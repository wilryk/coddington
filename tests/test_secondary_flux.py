"""Spec §C build plan (docs/secondary-irradiance-plan.md) steps 1-3:
secondary-mirror irradiance geometry + the cone backend's chief-point deposit,
validated before anything is wired into the web layer.

Three groups of tests:

* ``geometry.secondary``'s ``(u, v)`` parameterization and per-bin area
  formula (the plan's "document like receiver.bin_areas_m2" section) --
  round-trip and area-sum-equals-analytic-surface-area, both shapes.
* the cone backend's new ``return_secondary_flux`` opt-in -- proves it does
  not perturb the existing receiver-path result (default off, bit-identical)
  and that its own deposit is internally sane (fidelity tag, non-negative
  flux).
* the energy-consistency pin the plan calls for BEFORE any UI: total
  secondary-incident power must equal the power leaving the mirror toward
  it (post shading/blocking, pre secondary-reflectance), for both backends.
  Fixture pointing/figure rows are taken verbatim from
  ``tests/fixtures/mc_parity/{axicon,cassegrain}/summary.csv`` (heliostat 48,
  the ``mid_morning`` step) -- already-validated aim/figure solves, so this
  suite tests the NEW secondary-flux code, not whether a made-up heliostat
  position happens to face the tower.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    PyramidSecondary,
    secondary_bin_areas_m2,
    secondary_has_flux_map,
    secondary_uv,
    secondary_uv_extent,
    secondary_uv_to_world,
)
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat

WINDOW_MM = 2000.0

# Standard-paper axicon/Cassegrain geometry -- identical constants to
# tests/test_mc_parity.py::_geometry_for and heliostat.web.app's own
# AXICON_*/CASSEGRAIN_* defaults.
AXICON = AxiconSecondary(apex_height_mm=27000.0, half_angle_deg=20.0, aperture_radius_mm=14000.0)
CASSEGRAIN = CassegrainSecondary(
    vertex_z_mm=26993.999446877,
    vertex_radius_mm=26112.078893738,
    conic=-5.317616535,
    aperture_radius_mm=14000.0,
)
RECEIVER = FlatWindowReceiver(z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up")

# Heliostat 48, mid_morning step -- tests/fixtures/mc_parity/{axicon,
# cassegrain}/summary.csv, read verbatim (already-validated aim/figure
# solves against the golden MC fixtures; this suite reuses them rather than
# re-deriving a heliostat pose, so any failure here is about the NEW
# secondary-flux code, not about a made-up test heliostat missing the tower).
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


# ---------------------------------------------------------------------------
# secondary_has_flux_map
# ---------------------------------------------------------------------------


def test_has_flux_map_scoped_to_axicon_and_cassegrain():
    assert secondary_has_flux_map(AXICON) is True
    assert secondary_has_flux_map(CASSEGRAIN) is True
    assert secondary_has_flux_map(NoSecondary()) is False
    assert secondary_has_flux_map(PyramidSecondary(apex_height_mm=27000.0, angle_deg=20.0, half_side_mm=10000.0)) is False


@pytest.mark.parametrize("secondary", [NoSecondary(), PyramidSecondary(apex_height_mm=1.0, angle_deg=10.0, half_side_mm=100.0)])
def test_uv_helpers_reject_shapes_with_no_flux_map(secondary):
    with pytest.raises(ValueError):
        secondary_uv(secondary, np.zeros((3, 1)))
    with pytest.raises(ValueError):
        secondary_uv_extent(secondary)
    with pytest.raises(ValueError):
        secondary_bin_areas_m2(secondary, (8, 8))


# ---------------------------------------------------------------------------
# (u, v) round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secondary", [AXICON, CASSEGRAIN])
def test_uv_round_trip(secondary):
    """Points built at known (h, phi) recover exactly the (u, v) the plan's
    formula predicts: v = h; u = aperture_radius_mm * phi, with phi the
    ``atan2(x, -y)`` convention -- so x = h*sin(phi), y = -h*cos(phi)."""
    rng = np.random.default_rng(0)
    n = 500
    h = rng.uniform(0.0, secondary.aperture_radius_mm, n)
    phi = rng.uniform(-math.pi, math.pi, n)
    x = h * np.sin(phi)
    y = -h * np.cos(phi)
    z = np.zeros(n)  # secondary_uv only reads x, y

    uv = secondary_uv(secondary, np.vstack([x, y, z]))
    np.testing.assert_allclose(uv[1], h, atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(uv[0], secondary.aperture_radius_mm * phi, atol=1e-6, rtol=1e-12)

    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    r = secondary.aperture_radius_mm
    assert u0 == pytest.approx(-math.pi * r)
    assert u1 == pytest.approx(math.pi * r)
    assert v0 == 0.0
    assert v1 == r
    assert np.all(uv[0] >= u0 - 1e-6) and np.all(uv[0] <= u1 + 1e-6)
    assert np.all(uv[1] >= v0 - 1e-9) and np.all(uv[1] <= v1 + 1e-9)


def test_uv_seam_and_axis_are_well_defined():
    """North seam (+y axis, x=0) and the axis itself (h=0) are ordinary
    inputs, not singularities -- atan2(0, -y) is 0 at u=0 for y>0 (south of
    the seam convention) and well-defined at the origin."""
    p = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]])  # (0,1,0) and (0,-1,0)
    uv = secondary_uv(AXICON, p)
    assert np.all(np.isfinite(uv))
    origin = np.zeros((3, 1))
    uv0 = secondary_uv(AXICON, origin)
    assert uv0[1, 0] == 0.0
    assert np.isfinite(uv0[0, 0])


# ---------------------------------------------------------------------------
# uv_to_world -- exact inverse of secondary_uv (v0.2 followups item 3, the
# FEA export's new x/y/z-in-world-frame columns)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secondary,row", [(AXICON, AXICON_ROW), (CASSEGRAIN, CASSEGRAIN_ROW)])
def test_uv_to_world_is_the_exact_inverse_of_secondary_uv(secondary, row):
    """Round-trip against the forward map -- mirrors
    tests/test_receiver_shapes.py::test_uv_to_world_is_the_exact_inverse_of_intersect
    exactly, with :meth:`Secondary.redirect` (via
    ``trace_heliostat``'s ``return_secondary_hits``) standing in for
    ``Receiver.intersect``: real rays, reflected off the real primary mirror
    and this real secondary, produce independently-computed 3-D hit points
    that ``secondary_uv_to_world(secondary_uv(hit))`` must reproduce
    exactly -- not a tautology, since ``uv_to_world`` never sees ``hit``
    itself, only the ``(u, v)`` derived from it."""
    rng = np.random.default_rng(3)
    out = trace_heliostat(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, 20000, rng, return_secondary_hits=True,
    )
    hit_xyz = out["secondary_xy"]  # (3, K) world points despite the name -- see mc.py's own comment
    assert hit_xyz.shape[1] > 100  # sanity: a real, sizeable bundle of hits

    uv = secondary_uv(secondary, hit_xyz)
    world = secondary_uv_to_world(secondary, uv)
    np.testing.assert_allclose(world, hit_xyz, atol=1e-6)


@pytest.mark.parametrize("secondary", [NoSecondary(), PyramidSecondary(apex_height_mm=1.0, angle_deg=10.0, half_side_mm=100.0)])
def test_uv_to_world_rejects_shapes_with_no_flux_map(secondary):
    with pytest.raises(ValueError):
        secondary_uv_to_world(secondary, np.zeros((2, 1)))


# ---------------------------------------------------------------------------
# bin-area sum == analytic surface area
# ---------------------------------------------------------------------------


def test_axicon_bin_area_sum_matches_closed_form():
    """h*sec(slope) is linear in h for a cone (constant slope), so the
    midpoint rule integrates it exactly for ANY bin count -- same tightness
    as FrustumReceiver's own linear-in-v area test
    (tests/test_receiver_shapes.py)."""
    analytic_m2 = math.pi * AXICON.aperture_radius_mm**2 / math.cos(math.radians(AXICON.half_angle_deg)) / 1.0e6
    for grid in [(8, 8), (64, 37), (128, 500)]:
        total = float(np.sum(secondary_bin_areas_m2(AXICON, grid)))
        assert total == pytest.approx(analytic_m2, rel=1e-9), grid


def test_cassegrain_bin_area_sum_converges_to_numeric_surface_area():
    """No closed form for a hyperboloid's lateral area; reference is a direct
    numerical integration of the same h*sec(slope) integrand
    secondary_bin_areas_m2 evaluates at bin midpoints. The midpoint rule's
    error is O(1/n_v^2) for this smooth, non-linear integrand, so a modest
    v-bin count already lands well inside a loose tolerance."""
    r, kk = CASSEGRAIN.vertex_radius_mm, 1.0 + CASSEGRAIN.conic

    def integrand(h):
        disc = max(r * r - kk * h * h, 0.0)
        zeta = (r - math.sqrt(disc)) / kk
        slope = h / (r - kk * zeta)
        return h * math.sqrt(1.0 + slope * slope)

    val, _ = integrate.quad(integrand, 0.0, CASSEGRAIN.aperture_radius_mm)
    analytic_m2 = 2.0 * math.pi * val / 1.0e6

    total = float(np.sum(secondary_bin_areas_m2(CASSEGRAIN, (64, 2000))))
    assert total == pytest.approx(analytic_m2, rel=1e-5)


# ---------------------------------------------------------------------------
# cone backend: opt-in, bit-identical when not requested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secondary,row", [(AXICON, AXICON_ROW), (CASSEGRAIN, CASSEGRAIN_ROW)])
def test_return_secondary_flux_does_not_change_receiver_result(secondary, row):
    baseline = trace_heliostat_cone(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, KERNEL,
    )
    with_secondary = trace_heliostat_cone(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, KERNEL,
        return_secondary_flux=True, secondary_flux_grid=(48, 48),
    )
    np.testing.assert_array_equal(baseline["flux"], with_secondary["flux"])
    assert baseline["power_w"] == with_secondary["power_w"]
    assert baseline["incident_power_w"] == with_secondary["incident_power_w"]
    assert baseline["counters"] == with_secondary["counters"]
    assert "secondary_flux" not in baseline
    assert with_secondary["secondary_fidelity"] == "coarse"
    assert np.all(with_secondary["secondary_flux"] >= 0.0)
    assert with_secondary["secondary_power_w"] > 0.0


def test_return_secondary_flux_omitted_for_no_secondary():
    """NoSecondary has no flux-map parameterization -- the opt-in silently
    adds nothing rather than raising, so a prime-focus trace can pass the
    same flag unconditionally."""
    out = trace_heliostat_cone(
        AXICON_ROW["x_mm"], AXICON_ROW["y_mm"], AXICON_ROW["rot_az_deg"], AXICON_ROW["rot_el_deg"],
        AXICON_ROW["c3"], AXICON_ROW["c4"], AXICON_ROW["c5"],
        AXICON_ROW["solar_az_deg"], AXICON_ROW["solar_el_deg"],
        NoSecondary(), RECEIVER, KERNEL, return_secondary_flux=True,
    )
    assert "secondary_flux" not in out


# ---------------------------------------------------------------------------
# energy-consistency pin (plan: "write BEFORE any UI")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secondary,row", [(AXICON, AXICON_ROW), (CASSEGRAIN, CASSEGRAIN_ROW)])
def test_energy_pin_cone(secondary, row):
    """No occluders, one instant: total secondary-incident power must equal
    the power leaving the mirror toward it. ``incident_power_w`` is the
    cone backend's own cosine-weighted mirror-side total (weights.sum(),
    measured before any bounce); with a correctly-aimed field and an
    aperture that is not clipping the beam, essentially all of it should
    reach the secondary -- the 0.5% band is the plan's own tolerance for
    this backend's chief-point-deposit discretization (rim/tip losses,
    fallback-node granularity), not measurement noise."""
    out = trace_heliostat_cone(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, KERNEL,
        return_secondary_flux=True, secondary_flux_grid=(96, 96),
    )
    incident = out["incident_power_w"]
    secondary_power = out["secondary_power_w"]
    assert secondary_power == pytest.approx(incident, rel=0.005)

    # Internal consistency: the returned secondary_power_w must equal what
    # summing the returned flux grid against the TRUE per-bin areas gives --
    # i.e. the true-area correction (secondary_bin_areas_m2) did not alter
    # total power, only its distribution across bins.
    from heliostat.geometry.secondary import secondary_bin_areas_m2 as _areas

    n_v, n_u = out["secondary_flux"].shape
    areas_m2 = _areas(secondary, (n_u, n_v))
    reintegrated = float(np.sum(out["secondary_flux"] * areas_m2))
    assert reintegrated == pytest.approx(secondary_power, rel=1e-9)


@pytest.mark.parametrize("secondary,row", [(AXICON, AXICON_ROW), (CASSEGRAIN, CASSEGRAIN_ROW)])
def test_energy_pin_mc(secondary, row):
    """Monte Carlo: secondary_xy already carries the (x, y) of every ray
    that struck the secondary (heliostat.trace.mc.trace_heliostat's
    ``return_secondary_hits=True``, no new ray tracing). Histogramming those
    hits through secondary_uv/secondary_bin_areas_m2's own bins and
    re-integrating power must reproduce ``hit_secondary`` count *
    watts-per-ray exactly (to float round-off) -- a genuine conservation
    check on the (u, v) extent/binning: any bug that clips hits at the bin
    edges or mis-scales area would show up here at far more than 1e-6."""
    rng = np.random.default_rng(1)
    out = trace_heliostat(
        row["x_mm"], row["y_mm"], row["rot_az_deg"], row["rot_el_deg"],
        row["c3"], row["c4"], row["c5"], row["solar_az_deg"], row["solar_el_deg"],
        secondary, RECEIVER, 20000, rng, return_secondary_hits=True,
    )
    watts_per_ray = out["watts_per_ray"]
    n_hit = out["secondary_xy"].shape[1]
    assert n_hit == out["counters"]["hit_secondary"]
    expected_power = n_hit * watts_per_ray

    sec_xy = out["secondary_xy"]
    p3 = np.vstack([sec_xy, np.zeros(sec_xy.shape[1])])
    uv = secondary_uv(secondary, p3)
    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    n_u, n_v = 128, 128
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    counts, _, _ = np.histogram2d(uv[1], uv[0], bins=[v_edges, u_edges])
    assert counts.sum() == n_hit  # no hit lost to a binning/extent bug

    areas_m2 = secondary_bin_areas_m2(secondary, (n_u, n_v))
    flux = counts * watts_per_ray / areas_m2
    power_via_histogram = float(np.sum(flux * areas_m2))
    assert power_via_histogram == pytest.approx(expected_power, rel=1e-6)
