"""Spec §E2's remaining three bullets: surface deformation (a measured error
map on the secondary), parametric warp (defocus + astigmatism), and the
secondary sag view -- all built on top of 25eea5b's rigid-body misalignment
and f2bb0f4's §E error-map machinery, reused rather than reinvented.

Composition mechanism (see ``heliostat.geometry.secondary.secondary_warp_sag_mm``'s
own docstring for the full reasoning): the parametric warp is kept purely
ANALYTIC -- closed-form defocus/astigmatism sag terms and their closed-form
gradients -- and SUMMED with the (separately, bilinearly interpolated)
imported map's own gradient at query time, inside
``AxiconSecondary.redirect``/``CassegrainSecondary.redirect``, in the
secondary's LOCAL unperturbed frame (before the §E2 rigid-body transform
back to world). Both are keyword-only arguments on ``Secondary.redirect``
with defaults (``None``/``0.0``) that every non-MC caller (cone, the 3-D
scene) never overrides -- that default, not a mode branch, is what makes
this Monte-Carlo-only.

Five groups of tests, matching the build plan's MC gates:

(a) zero warp + no map traces bit-identically at cone fidelity (structurally
    unaffected -- the cone backend never even has the keyword to pass) and
    at MC fidelity too (the "no perturbation" fast path).
(b) implied-RMS pin for a synthetic secondary map, same closed-form
    convention ``tests/test_errormap.py`` already pins for the heliostat's
    own map -- reused unmodified since the RMS formula lives on
    ``ErrorMap`` itself, domain-agnostic.
(c) composition + correctness pin: warp-only and an equivalent map-only
    (round-tripped through the §D sag-CSV text convention) produce matching
    Monte Carlo spot statistics -- proving the map/warp SUM correctly and
    that the analytic closed forms and the map's bilinear grid agree.
(d) tests/test_secondary_flux.py's §C energy-conservation pin still holds
    with warp+map active (the perturbation only retilts the traced NORMAL,
    never the hit point -- so which rays count as "on the secondary" is
    unaffected, and this pin should hold to the same tight tolerance).
(e) rigid-body + map compose: a decentred (dz-only) secondary's map/warp
    effect is byte-for-byte the same rigid translation of the UNPERTURBED
    map/warp trace that ``test_decenter_is_a_rigid_translation`` already
    proves for bare geometry -- i.e. the map/warp is genuinely anchored to
    the secondary's own local frame, not to world space.

A sixth, unlisted group independently re-derives the reflected ray from
first principles (an explicit z = f(x, y) graph normal, hand-written, never
calling this module's own perturbation helpers) to pin the ABSOLUTE sign of
the composition -- catching a normal-convention sign error that (c)'s
map-vs-warp cross-check cannot, because both paths share the same
``AxiconSecondary``/``CassegrainSecondary.redirect`` branch that applies the
combined slope to the local normal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from heliostat.geometry.errormap import build_error_map, parse_error_map_csv
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    secondary_bin_areas_m2,
    secondary_nominal_sag_mm,
    secondary_uv,
    secondary_uv_extent,
    secondary_warp_sag_mm,
    secondary_warp_slopes,
)
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat

WINDOW_MM = 6000.0

# Standard-paper axicon/Cassegrain geometry -- identical constants to
# tests/test_secondary_perturbations.py, tests/test_secondary_flux.py and
# heliostat.web.app's own AXICON_*/CASSEGRAIN_* defaults.
AXICON = AxiconSecondary(apex_height_mm=27000.0, half_angle_deg=20.0, aperture_radius_mm=14000.0)
CASSEGRAIN = CassegrainSecondary(
    vertex_z_mm=26993.999446877,
    vertex_radius_mm=26112.078893738,
    conic=-5.317616535,
    aperture_radius_mm=14000.0,
)
RECEIVER = FlatWindowReceiver(z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up")

# Heliostat 48, mid_morning step -- tests/fixtures/mc_parity/{axicon,
# cassegrain}/summary.csv, read verbatim, exactly as the sibling test files.
AXICON_ROW = dict(
    x_mm=2.144027e-12,
    y_mm=35014.61863,
    rot_az_deg=-41.610506,
    rot_el_deg=55.757902,
    c3=-3.326145e-07,
    c4=-0.000001,
    c5=2.805182e-07,
    solar_az_deg=79.360701,
    solar_el_deg=44.891542,
)
CASSEGRAIN_ROW = dict(
    x_mm=0.0,
    y_mm=35014.61863,
    rot_az_deg=-39.675374,
    rot_el_deg=57.346906,
    c3=0.0,
    c4=-0.000001,
    c5=0.0,
    solar_az_deg=79.360701,
    solar_el_deg=44.891542,
)

KERNEL = sunshape_kernel("super_gauss")
SHAPES = {"axicon": (AXICON, AXICON_ROW), "cassegrain": (CASSEGRAIN, CASSEGRAIN_ROW)}


def _perturbed(nominal, **kw):
    return dataclasses.replace(nominal, **kw)


def _secondary_incoming_rays(secondary, receiver, row, n_rays=4000, seed=1):
    """Real ``(p, d)`` rays about to strike ``secondary`` -- copied verbatim
    from ``tests/test_secondary_perturbations.py`` (see that file for the
    full reasoning): mirror hit point and post-mirror direction, recovered
    from a Monte Carlo trace's own ``paths``."""
    rng = np.random.default_rng(seed)
    out = trace_heliostat(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, receiver, n_rays, rng, return_paths=True,
    )
    mir = out["paths"][1]
    con = out["paths"][2]
    d = con - mir
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    assert mir.shape[1] > 100
    return mir, d


def _trace_cone(secondary, row, **kw):
    return trace_heliostat_cone(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, KERNEL, **kw,
    )


def _trace_mc(secondary, row, n_rays, rng, **kw):
    return trace_heliostat(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, n_rays, rng, **kw,
    )


def _grid_over_secondary(aperture_radius_mm, n=41):
    """A regular grid COVERING the secondary's own circular aperture, in
    the §D CSV convention (meters) -- like ``tests/test_errormap.py``'s
    ``_grid_over_aperture`` but sized to the secondary's radius rather than
    the heliostat's rectangle."""
    xs = np.linspace(-aperture_radius_mm / 1000.0, aperture_radius_mm / 1000.0, n)
    ys = np.linspace(-aperture_radius_mm / 1000.0, aperture_radius_mm / 1000.0, n)
    return xs, ys


# ---------------------------------------------------------------------------
# (a) zero warp + no map is bit-identical, both backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_zero_warp_and_no_map_bit_identical_cone(shape):
    """The cone backend has no ``secondary_error_map``/``defocus_um``/
    ``astig_um`` keyword at all on its own call sites (heliostat.trace.cone
    calls ``secondary.redirect(p, d, {})`` with no such keywords anywhere),
    so it is structurally unaffected regardless of what a caller might
    configure elsewhere -- this pins that a cone trace of the SAME geometry
    is identical whichever of the (structurally identical, since
    ``Secondary`` carries no map/warp fields of its own) fixture objects
    traces it."""
    secondary, row = SHAPES[shape]
    out_a = _trace_cone(secondary, row, return_secondary_flux=True)
    out_b = _trace_cone(secondary, row, return_secondary_flux=True)
    np.testing.assert_array_equal(out_a["flux"], out_b["flux"])
    assert out_a["power_w"] == out_b["power_w"]
    np.testing.assert_array_equal(out_a["secondary_flux"], out_b["secondary_flux"])


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_zero_warp_and_no_map_bit_identical_mc(shape):
    """Passing the new keywords at their defaults (explicitly, and by
    omission) must trace bit-identically -- the "no perturbation" fast
    path costs nothing and changes nothing."""
    secondary, row = SHAPES[shape]
    out_a = _trace_mc(secondary, row, 20_000, np.random.default_rng(7), return_secondary_hits=True)
    out_b = _trace_mc(
        secondary, row, 20_000, np.random.default_rng(7), return_secondary_hits=True,
        secondary_error_map=None, secondary_defocus_um=0.0, secondary_astig_um=0.0, secondary_astig_axis_deg=0.0,
    )
    np.testing.assert_array_equal(out_a["xy"], out_b["xy"])
    np.testing.assert_array_equal(out_a["secondary_xy"], out_b["secondary_xy"])
    assert out_a["counters"] == out_b["counters"]


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_zero_warp_and_no_map_redirect_matches_no_kwargs_at_all(shape):
    """``redirect()`` called exactly the way cone.py/scene.py call it (no
    keywords whatsoever) must match calling it with the new keywords at
    their defaults -- the abstract signature's own promise."""
    secondary, row = SHAPES[shape]
    p, d = _secondary_incoming_rays(secondary, RECEIVER, row)
    p2_a, d2_a, on_a = secondary.redirect(p.copy(), d.copy(), {})
    p2_b, d2_b, on_b = secondary.redirect(p.copy(), d.copy(), {}, secondary_error_map=None, defocus_um=0.0, astig_um=0.0, astig_axis_deg=0.0)
    np.testing.assert_array_equal(on_a, on_b)
    np.testing.assert_array_equal(p2_a, p2_b)
    np.testing.assert_array_equal(d2_a, d2_b)


def test_http_cone_mode_unaffected_by_secondary_map_and_warp():
    """End-to-end through heliostat.web.app: an axicon request at
    ``fast_accurate`` must trace identically whether or not
    ``secondary_error_map``/``secondary_defocus_um``/``secondary_astig_um``
    are set in ``optics_params`` -- exercising the actual app.py plumbing
    (``_secondary_perturb_kwargs``, ``_trace_core``), not just the geometry
    layer the tests above pin directly."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from heliostat.web.app import create_app

    client = TestClient(create_app())
    xs, ys = _grid_over_secondary(AXICON.aperture_radius_mm, n=17)
    gx, gy = np.meshgrid(xs, ys)
    grid = {"x_m": xs.tolist(), "y_m": ys.tolist(), "dz_mm": (0.5 * np.sin(gx) * np.cos(gy)).tolist()}

    def payload(**secondary_extra):
        return {
            "design": {"type": "rect", "width_mm": 5000, "height_mm": 3000, "surface": "flat"},
            "mode": "fast_accurate",
            "optics": "axicon",
            "optics_params": {"secondary_reflectance": 0.90, **secondary_extra},
            "heliostat_x_mm": AXICON_ROW["x_mm"],
            "heliostat_y_mm": AXICON_ROW["y_mm"],
            "solar_az_deg": AXICON_ROW["solar_az_deg"],
            "solar_el_deg": AXICON_ROW["solar_el_deg"],
        }

    bare = client.post("/api/trace", json=payload())
    perturbed = client.post(
        "/api/trace",
        json=payload(secondary_error_map=grid, secondary_defocus_um=500.0, secondary_astig_um=300.0, secondary_astig_axis_deg=25.0),
    )
    assert bare.status_code == 200, bare.text
    assert perturbed.status_code == 200, perturbed.text
    a, b = bare.json(), perturbed.json()
    assert a["power_w"] == b["power_w"]
    assert a["peak_flux_kw_m2"] == b["peak_flux_kw_m2"]


# ---------------------------------------------------------------------------
# (b) implied-RMS pin for a synthetic secondary map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_secondary_map_implied_rms_matches_closed_form(shape):
    """Same closed-form pin as tests/test_errormap.py's
    ``test_implied_rms_matches_closed_form_sinusoid``, reproduced over the
    SECONDARY's own (much larger, circular) aperture domain -- confirming
    §E's RMS convention is genuinely domain-agnostic, as §E2 claims when it
    says "reuse the §E machinery" for the map itself."""
    secondary, _row = SHAPES[shape]
    r_mm = secondary.aperture_radius_mm
    A_mm = 0.8
    L_m = 2.0 * r_mm / 1000.0  # exactly one period across the full aperture width
    n = 161
    xs, ys = _grid_over_secondary(r_mm, n)
    gx, _gy = np.meshgrid(xs, ys)
    dz_mm = A_mm * np.sin(2.0 * np.pi * gx / L_m)
    smap = build_error_map(xs, ys, dz_mm)

    amp_rad = (A_mm / 1000.0) * (2.0 * np.pi / L_m)
    expected_rms_mrad = (amp_rad / 2.0) * 1000.0
    assert smap.rms_slope_mrad == pytest.approx(expected_rms_mrad, rel=0.02)
    assert smap.grid_shape == (n, n)


# ---------------------------------------------------------------------------
# (c) composition pin: warp-only vs. equivalent-map-only (round-tripped
# through the §D sag-CSV text convention) agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_warp_matches_equivalent_map_round_tripped_through_csv(shape):
    """A pure parametric warp and the SAME warp exported as a §D sag CSV,
    reimported as a measured error map, must move the Monte Carlo spot the
    same way (composition pin) -- both defocus and astigmatism are
    quadratic in (x, y), so their gradients are exactly linear, and a
    modest bilinear grid reproduces a linear function exactly (the same
    reasoning tests/test_errormap.py's own defocus-vs-map test leans on)."""
    secondary, row = SHAPES[shape]
    defocus_um = 900.0
    astig_um = 500.0
    astig_axis_deg = 35.0

    xs, ys = _grid_over_secondary(secondary.aperture_radius_mm, n=41)
    gx, gy = np.meshgrid(xs, ys)
    gx_mm, gy_mm = gx * 1000.0, gy * 1000.0
    dz_mm = secondary_warp_sag_mm(
        gx_mm, gy_mm, secondary.aperture_radius_mm, defocus_um, astig_um, astig_axis_deg
    )
    lines = [f"{gx[i, j]:.9g},{gy[i, j]:.9g},{dz_mm[i, j]:.9g}" for i in range(gx.shape[0]) for j in range(gx.shape[1])]
    reimported = parse_error_map_csv("\n".join(lines))

    n_rays = 200_000
    via_warp = _trace_mc(
        secondary, row, n_rays, np.random.default_rng(11),
        secondary_defocus_um=defocus_um, secondary_astig_um=astig_um, secondary_astig_axis_deg=astig_axis_deg,
    )
    via_map = _trace_mc(secondary, row, n_rays, np.random.default_rng(11), secondary_error_map=reimported)

    xy_warp = via_warp["xy"]
    xy_map = via_map["xy"]
    assert xy_warp.shape[1] > n_rays * 0.2, "unexpectedly low landing fraction"
    assert xy_warp.shape[1] == pytest.approx(xy_map.shape[1], rel=1e-3)
    for axis in (0, 1):
        mean_warp = float(np.mean(xy_warp[axis]))
        mean_map = float(np.mean(xy_map[axis]))
        assert mean_warp == pytest.approx(mean_map, abs=3.0), (
            f"axis {axis}: warp-driven centroid {mean_warp:.3f} mm vs map-driven {mean_map:.3f} mm"
        )
        std_warp = float(np.std(xy_warp[axis]))
        std_map = float(np.std(xy_map[axis]))
        assert std_warp == pytest.approx(std_map, rel=0.03), (
            f"axis {axis}: warp-driven spread {std_warp:.3f} mm vs map-driven {std_map:.3f} mm"
        )


def test_warp_slopes_match_finite_difference_of_warp_sag():
    """Independent numerical check that ``secondary_warp_slopes`` is truly
    the gradient of ``secondary_warp_sag_mm`` (catches an algebra slip in
    the closed forms that a map-vs-warp cross-check alone would not, since
    that cross-check only proves the two AGREE with each other)."""
    r_mm = 14000.0
    rng = np.random.default_rng(3)
    x_mm = rng.uniform(-r_mm * 0.9, r_mm * 0.9, 500)
    y_mm = rng.uniform(-r_mm * 0.9, r_mm * 0.9, 500)
    defocus_um, astig_um, astig_axis_deg = 700.0, 400.0, -20.0

    dzdx, dzdy = secondary_warp_slopes(x_mm, y_mm, r_mm, defocus_um, astig_um, astig_axis_deg)

    h = 0.5  # mm
    sag = lambda xx, yy: secondary_warp_sag_mm(xx, yy, r_mm, defocus_um, astig_um, astig_axis_deg)
    fd_dzdx = (sag(x_mm + h, y_mm) - sag(x_mm - h, y_mm)) / (2 * h)
    fd_dzdy = (sag(x_mm, y_mm + h) - sag(x_mm, y_mm - h)) / (2 * h)

    np.testing.assert_allclose(dzdx, fd_dzdx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(dzdy, fd_dzdy, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# absolute-sign pin: independently re-derive the reflected ray from a
# hand-written explicit z = f(x, y) graph normal, never calling this
# module's own perturbation helpers -- catches a normal-convention sign
# error that (c) above cannot (both its paths share the same +=/-=
# composition branch inside redirect()).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_defocus_reflection_matches_independent_explicit_graph_normal(shape):
    """Reflects the REAL incoming rays off an independently hand-built
    normal -- NOT calling ``secondary_warp_slopes``/``secondary_warp_sag_mm``
    or ``_secondary_combined_slopes`` -- and compares against
    ``AxiconSecondary``/``CassegrainSecondary.redirect(..., defocus_um=...)``'s
    actual output.

    The nominal (unperturbed) base normal is built here exactly the way
    each class's own OWN unperturbed code already builds it (visibly, in
    ``secondary.py``, unrelated to this feature and covered by 25eea5b's
    24 passing tests) -- ``cn0 = normalize(-k*x/h, -k*y/h, 1)`` for the
    axicon's constant-slope flank, ``sn0 = normalize(x, y, kk*zeta - r)``
    for the Cassegrain hyperboloid. The two are NOT the same sign
    convention (see ``CassegrainSecondary.redirect``'s own comment: the
    axicon's is "mostly +z", the Cassegrain's is its negative, "mostly
    -z") -- which is exactly why the composition step below is a
    DIFFERENT sign per shape, reasoned about fresh here rather than
    copied: a delta sag ``dz(x, y)`` added to an explicit graph ``z =
    f(x, y)`` written in the axicon's "mostly +z" convention changes its
    normal by ``-(d(dz)/dx, d(dz)/dy, 0)``; the Cassegrain's normal is the
    NEGATIVE of that convention, so the same physical delta changes IT by
    the opposite, ``+(d(dz)/dx, d(dz)/dy, 0)``. Each correction is applied
    to the ALREADY-NORMALIZED base and renormalized again -- the same
    two-stage order ``redirect`` uses (matching how ``slope_error_mrad``/
    the primary mirror's own ``error_map`` are composed in
    ``heliostat.trace.mc``: perturb the existing unit normal directly,
    not a pre-normalization raw vector), so this reproduces the actual
    trace to near machine precision rather than a first-order
    approximation of it.

    Restricted to hit points with local radius > 1500 mm so the axicon's
    (unmodeled-by-this-formula) rounded tip blend never enters either side
    of the comparison.
    """
    nominal, row = SHAPES[shape]
    defocus_um = 1200.0
    p, d = _secondary_incoming_rays(nominal, RECEIVER, row, n_rays=6000)

    # Where the UNPERTURBED surface is actually hit (defocus only retilts
    # the normal, never the hit point -- same convention as slope_error_mrad
    # and the §E map), so a plain nominal.redirect() locates every ray.
    p_nom, _d_nom, on_nom = nominal.redirect(p.copy(), d.copy(), {})
    x, y = p_nom[0], p_nom[1]
    h = np.hypot(x, y)
    far_from_tip = h > 1500.0
    x, y, h = x[far_from_tip], y[far_from_tip], h[far_from_tip]
    d_in = d[:, on_nom][:, far_from_tip]
    assert x.size > 50, "too few rays survived the tip-exclusion filter"

    r_ap = nominal.aperture_radius_mm
    if isinstance(nominal, AxiconSecondary):
        k = np.tan(np.deg2rad(nominal.half_angle_deg))
        base = np.vstack([-k * x / h, -k * y / h, np.ones_like(h)])
        sign = -1.0  # axicon: "mostly +z" convention -> subtract the bump's gradient
    else:
        rr = nominal.vertex_radius_mm
        kk = 1.0 + nominal.conic
        zeta = (rr - np.sqrt(np.clip(rr * rr - kk * h * h, 0.0, None))) / kk
        base = np.vstack([x, y, kk * zeta - rr])
        sign = +1.0  # Cassegrain: "mostly -z" (negated) convention -> add the bump's gradient
    base /= np.linalg.norm(base, axis=0)

    a_mm = defocus_um * 1.0e-3
    dfdx = (2.0 * a_mm / (r_ap * r_ap)) * x
    dfdy = (2.0 * a_mm / (r_ap * r_ap)) * y

    normal = base.copy()
    normal[0] += sign * dfdx
    normal[1] += sign * dfdy
    normal /= np.linalg.norm(normal, axis=0)

    dot = np.einsum("ij,ij->j", d_in, normal)
    d_out_manual = d_in - 2.0 * dot * normal

    p_pert, d_pert, on_pert = nominal.redirect(p.copy(), d.copy(), {}, defocus_um=defocus_um)
    np.testing.assert_array_equal(on_pert, on_nom)
    d_pert_far = d_pert[:, far_from_tip]

    np.testing.assert_allclose(d_pert_far, d_out_manual, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# (d) §C secondary-flux energy pin still holds with warp+map active
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_energy_pin_mc_with_warp_and_map_active(shape):
    """tests/test_secondary_flux.py::test_energy_pin_mc's own pin,
    reproduced with a nonzero warp AND a nonzero map active together: the
    perturbation only retilts the traced normal, never the hit point, so
    which rays count as "on the secondary" -- and therefore this
    conservation check -- must be unaffected."""
    secondary, row = SHAPES[shape]
    xs, ys = _grid_over_secondary(secondary.aperture_radius_mm, n=21)
    gx, gy = np.meshgrid(xs, ys)
    smap = build_error_map(xs, ys, (0.3 * np.sin(gx) * np.cos(gy)))

    out = _trace_mc(
        secondary, row, 20_000, np.random.default_rng(1), return_secondary_hits=True,
        secondary_error_map=smap, secondary_defocus_um=400.0, secondary_astig_um=200.0, secondary_astig_axis_deg=10.0,
    )
    watts_per_ray = out["watts_per_ray"]
    sec_xyz = out["secondary_xy"]
    n_hit = sec_xyz.shape[1]
    assert n_hit == out["counters"]["hit_secondary"]
    expected_power = n_hit * watts_per_ray

    uv = secondary_uv(secondary, sec_xyz)
    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    n_u, n_v = 256, 256
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    counts, _, _ = np.histogram2d(uv[1], uv[0], bins=[v_edges, u_edges])
    assert counts.sum() == n_hit

    # secondary_bin_areas_m2 masks bins outside the aperture disk to area 0
    # (see that function's own note) -- guard the divide the same way
    # tests/test_secondary_flux.py::test_energy_pin_mc does.
    areas_m2 = secondary_bin_areas_m2(secondary, (n_u, n_v))
    flux = np.divide(counts * watts_per_ray, areas_m2, out=np.zeros_like(areas_m2), where=areas_m2 > 0)
    power_via_histogram = float(np.sum(flux * areas_m2))
    assert power_via_histogram == pytest.approx(expected_power, rel=1e-6)


# ---------------------------------------------------------------------------
# (e) rigid-body + map compose: dz-only decenter is still an exact rigid
# translation with the map/warp active -- proving the perturbation is
# anchored to the secondary's own local frame, not to world space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_map_and_warp_compose_with_dz_decenter_as_rigid_translation(shape):
    """With a map AND a warp both active, a pure dz decenter must still
    satisfy the exact rigid-translation law
    ``tests/test_secondary_perturbations.py::test_decenter_is_a_rigid_translation``
    already pins for bare geometry: reflecting off the decentred secondary
    equals translating the incoming ray by ``-decenter``, reflecting off
    the UNPERTURBED secondary (with the SAME map/warp), then translating
    the hit point back. This only holds if the map/warp is queried in the
    secondary's own LOCAL frame (unaffected by a world-space dz shift) --
    exactly what ``AxiconSecondary``/``CassegrainSecondary.redirect``
    apply it in, ahead of the rigid-body transform back to world.
    """
    nominal, row = SHAPES[shape]
    xs, ys = _grid_over_secondary(nominal.aperture_radius_mm, n=21)
    gx, gy = np.meshgrid(xs, ys)
    smap = build_error_map(xs, ys, (0.4 * gx + 0.2 * gy))
    kw = dict(secondary_error_map=smap, defocus_um=350.0, astig_um=150.0, astig_axis_deg=60.0)

    p, d = _secondary_incoming_rays(nominal, RECEIVER, row)
    dz_mm = 45.0
    decenter = np.array([0.0, 0.0, dz_mm])
    decentred = dataclasses.replace(nominal, dz_mm=dz_mm)

    p2_pert, d2_pert, on_pert = decentred.redirect(p.copy(), d.copy(), {}, **kw)
    p2_nom, d2_nom, on_nom = nominal.redirect(p.copy() - decenter[:, None], d.copy(), {}, **kw)

    np.testing.assert_array_equal(on_pert, on_nom)
    np.testing.assert_allclose(p2_pert, p2_nom + decenter[:, None], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(d2_pert, d2_nom, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# secondary sag view: nominal + warp + map, summed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_secondary_sag_grid_sums_nominal_warp_and_map(shape):
    """``heliostat.web.app._secondary_sag_grid_mm`` -- imported lazily here
    to avoid pulling FastAPI into the module import for the tests above --
    must equal the plain sum of the three pieces the spec names: nominal
    figure, parametric warp, imported map."""
    pytest.importorskip("fastapi")
    from heliostat.web.app import _secondary_sag_grid_mm

    secondary, _row = SHAPES[shape]
    xs, ys = _grid_over_secondary(secondary.aperture_radius_mm, n=21)
    gx, gy = np.meshgrid(xs, ys)
    grid = {"x_m": xs.tolist(), "y_m": ys.tolist(), "dz_mm": (0.2 * gx).tolist()}

    class _FakeOptics:
        secondary_error_map = grid
        secondary_defocus_um = 600.0
        secondary_astig_um = 300.0
        secondary_astig_axis_deg = 15.0

    gx_out, gy_out, sag = _secondary_sag_grid_mm(secondary, _FakeOptics(), n=61)

    nominal = secondary_nominal_sag_mm(secondary, gx_out, gy_out)
    warp = secondary_warp_sag_mm(gx_out, gy_out, secondary.aperture_radius_mm, 600.0, 300.0, 15.0)
    smap = build_error_map(xs, ys, (0.2 * gx))
    map_dz = smap.sample_dz(gx_out, gy_out)

    expected = nominal + warp + map_dz
    inside = np.hypot(gx_out, gy_out) <= secondary.aperture_radius_mm
    np.testing.assert_allclose(sag[inside], expected[inside], rtol=1e-9, atol=1e-9)
    assert np.all(np.isnan(sag[~inside]))
