"""Measured (FEA) error-map import -- docs/ui-spec-v0.2.md §E.

Pins, in order, the same build-order the feature was implemented against:

(a) a map that is all zeros changes nothing -- bit-identical to no map.
(b) a synthetic map equal to a pure analytic defocus increment broadens/
    shifts the Monte Carlo spot the same way applying that analytic change
    directly does (both are slope-only perturbations of the traced normal,
    never the hit point -- see :mod:`heliostat.geometry.errormap`'s module
    docstring -- so for a purely quadratic sag increment, whose gradient is
    exactly linear, bilinear interpolation of the map's precomputed slope
    grid reproduces the analytic gradient exactly at every ray, and the two
    paths should agree far tighter than Monte Carlo noise).
(c) a synthetic sinusoidal map's reported RMS slope matches the closed-form
    analytic RMS of its gradient, under the pooled/per-axis convention
    :mod:`heliostat.geometry.errormap` documents (matching how
    ``slope_error_mrad`` is actually applied in :mod:`heliostat.trace.mc`,
    not the glossary's loosely-worded "RMS angle").
(d) the cone backends never accept an ``error_map`` argument at all, so a
    map attached to a design is structurally inert there -- exercised at
    the HTTP layer via ``/api/trace``, comparing a request with and without
    ``error_map`` at ``mode="fast_accurate"``.
(e) trace-time: a 512x512 map costs within a generous margin of no map at
    all, confirming the "one bilinear lookup per ray, independent of map
    resolution" guarantee (§E) rather than an accidental O(grid) cost.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from heliostat.geometry.errormap import build_error_map, parse_error_map_csv
from heliostat.geometry.heliostat import zernike_sag_and_slopes
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM, _mirror_frame, _sun_vector, trace_heliostat

# ---------------------------------------------------------------------------
# shared geometry -- same construction tests/test_mc_physics_fixes.py uses:
# rot_az=0 / sun_az=90 puts the mirror normal and the sun vector in the
# world x-z plane, a clean 45-deg-incidence 2-D problem.
ROT_AZ, ROT_EL = 0.0, 75.0
SUN_AZ, SUN_EL = 90.0, 30.0
N_RAYS = 200_000
SEED = 7


def _chief_direction(c3=0.0, c4=0.0, c5=0.0):
    """World direction of the chief ray reflected off the mirror's own
    centre (x=y=0, where every figure here evaluates to zero slope), used
    only to place a receiver window that is guaranteed to catch the spot."""
    n, u, v = _mirror_frame(ROT_AZ, ROT_EL)
    s = _sun_vector(SUN_AZ, SUN_EL)
    d_in = -s
    d_out = d_in - 2.0 * np.dot(d_in, n) * n
    return d_out


def _receiver_at(d_out, z_mm=30_000.0, half_mm=8000.0):
    t = z_mm / d_out[2]
    cx, cy = d_out[0] * t, d_out[1] * t
    return FlatWindowReceiver(
        z_mm=z_mm, half_u_mm=half_mm, half_v_mm=half_mm, facing="down",
        center_x_mm=cx, center_y_mm=cy,
    )


def _grid_over_aperture(n=64):
    xs = np.linspace(-MIRROR_HALF_X_MM / 1000.0, MIRROR_HALF_X_MM / 1000.0, n)
    ys = np.linspace(-MIRROR_HALF_Y_MM / 1000.0, MIRROR_HALF_Y_MM / 1000.0, n)
    return xs, ys


def _trace(*, c3=0.0, c4=0.0, c5=0.0, error_map=None, seed=SEED, n_rays=N_RAYS, receiver=None):
    if receiver is None:
        receiver = _receiver_at(_chief_direction())
    return trace_heliostat(
        0.0, 0.0, ROT_AZ, ROT_EL, c3, c4, c5, SUN_AZ, SUN_EL,
        NoSecondary(), receiver, n_rays, np.random.default_rng(seed),
        source_disk_radius_mm="auto", error_map=error_map,
    )


# ---------------------------------------------------------------------------
# (a) all-zero map == no map, bit-identical


def test_zero_map_is_bit_identical_to_no_map():
    xs, ys = _grid_over_aperture(16)
    dz_mm = np.zeros((ys.size, xs.size))
    zero_map = build_error_map(xs, ys, dz_mm)

    receiver = _receiver_at(_chief_direction())
    with_map = _trace(error_map=zero_map, receiver=receiver)
    without_map = _trace(error_map=None, receiver=receiver)

    assert with_map["counters"] == without_map["counters"]
    np.testing.assert_array_equal(with_map["xy"], without_map["xy"])
    assert with_map["watts_per_ray"] == without_map["watts_per_ray"]


# ---------------------------------------------------------------------------
# (b) a map equal to a pure analytic defocus increment matches applying
# that increment directly


def test_map_equal_to_analytic_defocus_matches_direct_defocus():
    # A pure parabolic bowl dz(x, y) = k * (x^2 + y^2), x/y in meters, dz in
    # mm -- gradient (2kx, 2ky) is exactly linear, so a coarse grid's
    # bilinear-interpolated slope reproduces it exactly at every ray. Kept
    # small (~1 mm peak sag) because the error map ONLY perturbs the traced
    # NORMAL, never the mirror hit POINT (§E/mc.py's own convention -- same
    # simplification slope_error_mrad/specularity_mrad already make), while
    # an equivalent analytic c4 changes both (its sag also feeds the hit-
    # point Newton solve) -- at a large sag the two hit points diverge
    # enough to show up as a second-order centroid difference of their own,
    # unrelated to the slope-equivalence this test is pinning; small keeps
    # that second-order term well under the noise floor asserted below
    # (confirmed by hand: k=4.0 here gives a ~10 mm hit-point-driven
    # centroid gap, k=0.5 gives < 0.2 mm).
    k_mm_per_m2 = 0.5

    xs, ys = _grid_over_aperture(24)
    gx, gy = np.meshgrid(xs, ys)
    dz_mm = k_mm_per_m2 * (gx**2 + gy**2)
    defocus_map = build_error_map(xs, ys, dz_mm)

    # mc.trace_heliostat negates c4/c5 internally (its frame convention --
    # see the module docstring), and its Zernike sag is evaluated with x, y
    # in MILLIMETERS at normrad=1: c4 * sqrt(3) * (2x^2 + 2y^2 - 1). Solve
    # for the c4 ARGUMENT (pre-negation) that reproduces the same
    # dz = k_mm_per_m2 * (x_m^2 + y_m^2) = k_mm_per_m2/1e6 * (x_mm^2 + y_mm^2)
    # gradient: internal c4 * sqrt(3) * 2 = k_mm_per_m2 / 1e6, and the
    # traced formula uses -c4_arg internally, so c4_arg = -k / (2 sqrt(3) 1e6).
    sqrt3 = np.sqrt(3.0)
    c4_internal = (k_mm_per_m2 / 1.0e6) / (2.0 * sqrt3)
    c4_arg = -c4_internal
    # Sanity: the internal sign flip reproduces the intended gradient.
    _, dsdx_chk, dsdy_chk = zernike_sag_and_slopes(
        np.array([1000.0]), np.array([500.0]), 0.0, -c4_arg, 0.0
    )
    expected_dsdx = 2.0 * (k_mm_per_m2 / 1.0e6) * 1000.0
    assert dsdx_chk[0] == pytest.approx(expected_dsdx, rel=1e-9)

    receiver = _receiver_at(_chief_direction())
    via_map = _trace(error_map=defocus_map, receiver=receiver)
    via_analytic = _trace(c4=c4_arg, receiver=receiver)

    xy_map = via_map["xy"]
    xy_analytic = via_analytic["xy"]
    assert xy_map.shape[1] > N_RAYS * 0.2, "unexpectedly low landing fraction"
    # Both paths draw the exact same ray stream (same seed) and apply the
    # same slope field to well within float noise, so the landed-ray counts
    # should match almost exactly -- a handful of edge-of-window rays can
    # still flip either way from the tiny (~1e-12 rel) floating-point
    # differences between "evaluate a closed form" and "bilinearly
    # interpolate that same closed form's values off a grid".
    assert xy_map.shape[1] == pytest.approx(xy_analytic.shape[1], rel=1e-3), (
        f"map path landed {xy_map.shape[1]} rays vs analytic path's "
        f"{xy_analytic.shape[1]}"
    )

    # Spot centroid/spread should agree far tighter than Monte Carlo noise
    # (se(mean) ~ spot_sigma/sqrt(n) ~ a few mm at N_RAYS -- 2 mm is a
    # generous multiple of that).
    for axis in (0, 1):
        mean_map = float(np.mean(xy_map[axis]))
        mean_analytic = float(np.mean(xy_analytic[axis]))
        assert mean_map == pytest.approx(mean_analytic, abs=1.0), (
            f"axis {axis}: map-driven centroid {mean_map:.3f} mm vs "
            f"analytic-driven {mean_analytic:.3f} mm"
        )
        std_map = float(np.std(xy_map[axis]))
        std_analytic = float(np.std(xy_analytic[axis]))
        assert std_map == pytest.approx(std_analytic, rel=0.02), (
            f"axis {axis}: map-driven spread {std_map:.3f} mm vs "
            f"analytic-driven {std_analytic:.3f} mm"
        )


# ---------------------------------------------------------------------------
# (c) implied-RMS pin: a sinusoidal map's reported RMS matches the
# closed-form analytic RMS of its gradient


def test_implied_rms_matches_closed_form_sinusoid():
    # dz(x, y) = A * sin(2 pi x / L), x in meters, A in mm -- a purely
    # x-dependent sinusoid so the closed-form RMS is elementary.
    A_mm = 0.6
    L_m = 5.0
    n = 121
    xs, ys = _grid_over_aperture(n)
    gx, _gy = np.meshgrid(xs, ys)
    dz_mm = A_mm * np.sin(2.0 * np.pi * gx / L_m)
    smap = build_error_map(xs, ys, dz_mm)

    # dz/dx (radians, dz in meters) = (A_mm/1000) * (2 pi / L) * cos(...);
    # dz/dy = 0 identically. Per the module's pooled convention,
    # rms_slope_rad = sqrt(mean(dzdx^2 + dzdy^2) / 2) = sqrt(mean(dzdx^2) / 2)
    # = sqrt((amplitude^2 / 2) / 2) = amplitude / 2, since mean(cos^2) = 1/2
    # over a whole number of periods (aperture 5 m wide == exactly one
    # period of L_m = 5 m).
    amp_rad = (A_mm / 1000.0) * (2.0 * np.pi / L_m)
    expected_rms_mrad = (amp_rad / 2.0) * 1000.0

    assert smap.rms_slope_mrad == pytest.approx(expected_rms_mrad, rel=0.02), (
        f"reported {smap.rms_slope_mrad:.5f} mrad vs closed-form "
        f"{expected_rms_mrad:.5f} mrad"
    )
    assert smap.grid_shape == (n, n)
    assert smap.coverage_fraction == pytest.approx(1.0)


def test_csv_round_trip_preserves_grid_and_rms():
    """A Coddington-shaped §D export, re-parsed as an error map, reproduces
    the same grid and RMS -- the round-trip §E's own wording promises."""
    n = 31
    xs, ys = _grid_over_aperture(n)
    gx, gy = np.meshgrid(xs, ys)
    dz_mm = 0.3 * gx - 0.15 * gy

    lines = [
        "# units: x_m, y_m in meters (heliostat aperture frame); z_sag_mm in millimeters",
        "# heliostat: test · sun: az=180.00 deg, el=45.00 deg · mode: measured · timestamp: 2026-01-01T00:00:00Z",
        f"# grid: {n} x {n} samples",
    ]
    for i in range(n):
        for j in range(n):
            lines.append(f"{gx[i, j]:.6g},{gy[i, j]:.6g},{dz_mm[i, j]:.6g}")
    text = "\n".join(lines) + "\n"

    parsed = parse_error_map_csv(text)
    direct = build_error_map(xs, ys, dz_mm)
    assert parsed.grid_shape == direct.grid_shape
    assert parsed.rms_slope_mrad == pytest.approx(direct.rms_slope_mrad, rel=1e-6)
    assert parsed.coverage_fraction == pytest.approx(1.0)


def test_malformed_grid_rejected():
    """An irregular/scattered point cloud is not something the bilinear
    lookup can serve -- rejected at import with a message, not a silent
    best-effort grid."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2.0, 2.0, 50)
    ys = rng.uniform(-1.0, 1.0, 50)
    dz = np.zeros(50)
    lines = [f"{x:.6g},{y:.6g},{z:.6g}" for x, y, z in zip(xs, ys, dz)]
    with pytest.raises(ValueError, match="regular grid"):
        parse_error_map_csv("\n".join(lines))


# ---------------------------------------------------------------------------
# (e) trace-time guarantee: a fine map costs about the same as no map


# ---------------------------------------------------------------------------
# app.py: the import endpoint, and the (d) cone-is-inert / MC-changes
# acceptance scenario (docs/ui-spec-v0.2.md §J.147)


def _rect_design(surface="flat", **extra):
    d = {"type": "rect", "width_mm": 5000, "height_mm": 3000, "surface": surface}
    d.update(extra)
    return d


def _trace_payload(design, mode="monte_carlo", solar_el_deg=45.0, n_rays=None):
    body = {
        "design": design,
        "mode": mode,
        "optics": "prime_focus",
        "solar_az_deg": 180.0,
        "solar_el_deg": solar_el_deg,
    }
    if n_rays is not None:
        body["n_rays"] = n_rays
    return body


def _web_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from heliostat.web.app import create_app

    return TestClient(create_app())


def test_errormap_import_endpoint_reports_grid_coverage_and_rms():
    client = _web_client()
    n = 21
    xs, ys = _grid_over_aperture(n)
    gx, gy = np.meshgrid(xs, ys)
    dz_mm = 0.2 * gx
    lines = ["# units: x_m, y_m in meters; z_sag_mm in millimeters"]
    lines += [f"{gx[i, j]:.6g},{gy[i, j]:.6g},{dz_mm[i, j]:.6g}" for i in range(n) for j in range(n)]
    resp = client.post("/api/design/errormap/import", json={"csv": "\n".join(lines)})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grid_size"] == {"nx": n, "ny": n}
    assert data["coverage_fraction"] == pytest.approx(1.0)
    assert data["rms_slope_mrad"] > 0
    assert set(data["grid"].keys()) == {"x_m", "y_m", "dz_mm"}
    assert len(data["grid"]["dz_mm"]) == n
    assert len(data["grid"]["dz_mm"][0]) == n


def test_errormap_import_endpoint_rejects_malformed_csv():
    client = _web_client()
    resp = client.post("/api/design/errormap/import", json={"csv": "not,a,valid\nheader\n1,2"})
    assert resp.status_code == 422


def test_design_error_map_rejected_when_grid_irregular():
    client = _web_client()
    bad_grid = {"x_m": [0.0, 0.5, 1.3], "y_m": [0.0, 1.0], "dz_mm": [[0, 0, 0], [0, 0, 0]]}
    payload = _trace_payload(_rect_design(error_map=bad_grid))
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 422


def test_cone_mode_bit_identical_with_or_without_error_map():
    """(d): the cone backend never receives ``error_map`` at all, so a map
    attached to the design changes nothing at ``fast_accurate``."""
    client = _web_client()
    n = 17
    xs, ys = _grid_over_aperture(n)
    gx, gy = np.meshgrid(xs, ys)
    grid = {
        "x_m": xs.tolist(),
        "y_m": ys.tolist(),
        "dz_mm": (0.4 * np.sin(gx) * np.cos(gy)).tolist(),
    }
    without_map = client.post("/api/trace", json=_trace_payload(_rect_design(), mode="fast_accurate"))
    with_map = client.post(
        "/api/trace", json=_trace_payload(_rect_design(error_map=grid), mode="fast_accurate")
    )
    assert without_map.status_code == 200
    assert with_map.status_code == 200
    a, b = without_map.json(), with_map.json()
    assert a["power_w"] == b["power_w"]
    assert a["peak_flux_kw_m2"] == b["peak_flux_kw_m2"]


def test_measured_map_changes_mc_trace_export_reimport_scenario():
    """§J.147's own acceptance scenario: export the (flat, so nominal-zero)
    sag map, add a synthetic delta, reimport as a measured error map, and
    confirm a Monte Carlo trace changes while a cone trace does not."""
    client = _web_client()
    sag_resp = client.post("/api/design/sag.csv", json=_trace_payload(_rect_design(), mode="monte_carlo"))
    assert sag_resp.status_code == 200, sag_resp.text
    text = sag_resp.text

    lines_out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            lines_out.append(line)
            continue
        x_s, y_s, _z_s = line.split(",")
        x = float(x_s)
        # Synthetic delta: 1 mm/m of slope in x -- large enough to move the
        # Monte Carlo spot well above its own noise floor at this ray count.
        lines_out.append(f"{x_s},{y_s},{1.0 * x:.6g}")
    resp = client.post("/api/design/errormap/import", json={"csv": "\n".join(lines_out)})
    assert resp.status_code == 200, resp.text
    imported = resp.json()
    assert imported["rms_slope_mrad"] > 0

    design_with_map = _rect_design(error_map=imported["grid"])

    mc_payload_a = _trace_payload(_rect_design(), mode="monte_carlo", n_rays=200_000)
    mc_payload_b = _trace_payload(design_with_map, mode="monte_carlo", n_rays=200_000)
    mc_a = client.post("/api/trace", json=mc_payload_a).json()
    mc_b = client.post("/api/trace", json=mc_payload_b).json()
    assert mc_a["power_w"] != pytest.approx(mc_b["power_w"], rel=1e-6) or mc_a[
        "peak_flux_kw_m2"
    ] != pytest.approx(mc_b["peak_flux_kw_m2"], rel=1e-6), "measured map should visibly change the MC trace"

    cone_a = client.post(
        "/api/trace", json=_trace_payload(_rect_design(), mode="fast_accurate")
    ).json()
    cone_b = client.post(
        "/api/trace", json=_trace_payload(design_with_map, mode="fast_accurate")
    ).json()
    assert cone_a["power_w"] == cone_b["power_w"]
    assert cone_a["peak_flux_kw_m2"] == cone_b["peak_flux_kw_m2"]


def test_trace_time_independent_of_map_resolution():
    """A 512x512 map's bilinear lookup should cost about the same per ray
    as no map at all -- the whole point of pre-processing gradients once at
    import (§E) rather than evaluating a figure per ray. Timed with a
    generous margin (documented, not tuned) to avoid flaking on a loaded
    CI box; if this ever proves flaky in practice, replace the assertion
    with measured numbers in this docstring instead (per the task's own
    fallback) rather than tightening it.

    Measured on the dev machine this was written on: no-map ~ map time to
    within a few percent for 300k rays: the guarantee holds well inside the
    generous band asserted below.
    """
    xs, ys = _grid_over_aperture(512)
    gx, gy = np.meshgrid(xs, ys)
    dz_mm = 0.4 * np.sin(gx) * np.cos(gy)
    fine_map = build_error_map(xs, ys, dz_mm)

    receiver = _receiver_at(_chief_direction())
    n_rays = 300_000

    def _timed(error_map, reps=3):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            _trace(error_map=error_map, receiver=receiver, n_rays=n_rays, seed=SEED)
            best = min(best, time.perf_counter() - t0)
        return best

    t_no_map = _timed(None)
    t_with_map = _timed(fine_map)

    assert t_with_map < t_no_map * 1.5 + 0.05, (
        f"map trace took {t_with_map:.4f}s vs {t_no_map:.4f}s with no map -- "
        "cost should be roughly resolution-independent"
    )
