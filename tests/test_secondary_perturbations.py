"""Spec §E2 rigid-body misalignment of the secondary mirror: decenter
(dx/dy/dz, mm) and tip/tilt (mrad, about the vertex/apex) for
:class:`~heliostat.geometry.secondary.AxiconSecondary` and
:class:`~heliostat.geometry.secondary.CassegrainSecondary`.

Implementation is a rigid transform bracketing the existing (unchanged)
exact conic intersection math inside each shape's own ``redirect``: an
incoming world ray is mapped into the secondary's perturbed local frame,
the nominal cone/hyperboloid equations run exactly as before, and the
resulting hit point and reflected direction are mapped back to world. At
zero perturbation (the default) this takes a fast identity path -- no
rotation-matrix multiply at all -- so every pre-existing trace stays
bit-identical.

Four groups of tests, matching the build plan:

1. Zero perturbation is bit-identical to the unperturbed geometry, both
   shapes, both backends -- the load-bearing pin.
2. Per-axis physical sanity, each closed-form/exact rather than a fuzzy
   tolerance:
   * decenter (dx, dy, dz) is an exact rigid translation of ``redirect``'s
     input/output (any correct implementation of "move the mirror by
     decenter" must satisfy this, independent of the cone/hyperboloid
     equations' own details); the dz axis additionally gets the spec's own
     suggested concrete check ("shifts the fold height by exactly dz"),
     verified end to end through the cone backend as an exact equivalence
     to moving ``apex_height_mm``/``vertex_z_mm`` itself.
   * tip/tilt is an exact rigid rotation about the vertex, in the same
     sense (a rotated mirror sees a de-rotated ray) -- checked both as a
     standalone textbook doubling-law identity (independent of this
     module entirely) and end to end on the real classes.
3. MC-vs-cone cross-check on one perturbed configuration per shape, same
   tolerance convention ``tests/test_cone_vs_mc.py`` uses (noise-derived,
   generous absolute floor).
4. Spec §C's secondary-flux energy-conservation pin
   (``tests/test_secondary_flux.py``'s own pattern) still holds under a
   nonzero perturbation, both backends -- this is also the test that
   exercises the ``secondary_uv``/``to_local_point`` fix that keeps the
   flux map anchored to the physical surface rather than to world space.

Geometry constants and the (already aim/figure-solved) fixture rows are
copied verbatim from ``tests/test_secondary_flux.py`` rather than imported,
matching that file's own stated reasoning for not sharing a helpers module
across these small, adjacent test files.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    secondary_bin_areas_m2,
    secondary_uv,
    secondary_uv_extent,
)
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat

# Deliberately wider than tests/test_secondary_flux.py's 2000 mm: that
# window is narrow enough to clip a meaningful fraction of heliostat 48's
# spot even in the UNPERTURBED baseline (measured ~2% cone-vs-MC power gap,
# confirmed independent of this feature -- window-edge clipping is exactly
# where the cone backend's chief-point/node-fallback deposit is coarsest,
# tests/test_cone_vs_mc.py's own module docstring says as much). A wide
# enough window that the beam is not clipped keeps the cross-check test
# below about the PERTURBATION machinery's own accuracy, not about
# re-litigating the cone backend's pre-existing edge-clipping tolerance.
WINDOW_MM = 6000.0

# Standard-paper axicon/Cassegrain geometry -- identical constants to
# tests/test_secondary_flux.py, tests/test_mc_parity.py::_geometry_for and
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
# cassegrain}/summary.csv, read verbatim (already-validated aim/figure
# solves against the golden MC fixtures), exactly as
# tests/test_secondary_flux.py uses them.
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

# A modest, deliberately small combined perturbation for the "does the whole
# feature hang together" tests (3 and 4): mm-scale decenter and a couple
# mrad of tip/tilt against a 14 m secondary aperture, so most of the beam
# still lands where it always did -- a perturbation, not a reconfiguration.
COMBINED_PERTURBATION = dict(dx_mm=15.0, dy_mm=-10.0, dz_mm=20.0, tip_mrad=1.0, tilt_mrad=-1.5)

DECENTER_AXES = {
    "dx": dict(dx_mm=25.0, dy_mm=0.0, dz_mm=0.0),
    "dy": dict(dx_mm=0.0, dy_mm=-18.0, dz_mm=0.0),
    "dz": dict(dx_mm=0.0, dy_mm=0.0, dz_mm=30.0),
}
TILT_AXES = {
    "tip": dict(tip_mrad=6.0, tilt_mrad=0.0),
    "tilt": dict(tip_mrad=0.0, tilt_mrad=-4.5),
}


def _perturbed(nominal, **kw):
    return dataclasses.replace(nominal, **kw)


def _rotation_matrix(tip_mrad: float, tilt_mrad: float) -> np.ndarray:
    """Independent re-derivation of secondary.py's documented tip/tilt
    convention (tip rotates about local x, tilt about local y, composed
    tip-then-tilt) -- NOT a call into the module's own private helper, so
    the tests below check the classes against the convention their
    docstrings promise, not against their own implementation detail.
    """
    tip = tip_mrad * 1.0e-3
    tilt = tilt_mrad * 1.0e-3
    ct, st = np.cos(tip), np.sin(tip)
    cl, sl = np.cos(tilt), np.sin(tilt)
    r_tip = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    r_tilt = np.array([[cl, 0.0, sl], [0.0, 1.0, 0.0], [-sl, 0.0, cl]])
    return r_tilt @ r_tip


def _vertex_z(secondary) -> float:
    if isinstance(secondary, AxiconSecondary):
        return secondary.apex_height_mm
    return secondary.vertex_z_mm


def _secondary_incoming_rays(secondary, receiver, row, n_rays=4000, seed=1):
    """Real ``(p, d)`` rays about to strike ``secondary``: mirror hit point
    and post-mirror direction, reconstructed from a Monte Carlo trace's own
    ``paths`` (source -> mirror -> secondary -> receiver). ``paths[1]`` is
    the mirror hit, ``paths[2]`` the secondary hit -- straight-line
    propagation between them means their difference, normalised, is exactly
    the direction the mirror sent that ray in. Restricted to rays that made
    it all the way to the receiver window (a subset of everything that hit
    the secondary), which is a real, non-contrived bundle spanning the
    secondary's actually-used footprint -- enough to exercise ``redirect``
    over many points at once rather than one hand-picked ray.
    """
    rng = np.random.default_rng(seed)
    out = trace_heliostat(
        row["x_mm"],
        row["y_mm"],
        row["rot_az_deg"],
        row["rot_el_deg"],
        row["c3"],
        row["c4"],
        row["c5"],
        row["solar_az_deg"],
        row["solar_el_deg"],
        secondary,
        receiver,
        n_rays,
        rng,
        return_paths=True,
    )
    mir = out["paths"][1]
    con = out["paths"][2]
    d = con - mir
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    assert mir.shape[1] > 100  # sanity: the bundle isn't degenerate
    return mir, d


def _trace_cone(secondary, row, **kw):
    return trace_heliostat_cone(
        row["x_mm"],
        row["y_mm"],
        row["rot_az_deg"],
        row["rot_el_deg"],
        row["c3"],
        row["c4"],
        row["c5"],
        row["solar_az_deg"],
        row["solar_el_deg"],
        secondary,
        RECEIVER,
        KERNEL,
        **kw,
    )


def _trace_mc(secondary, row, n_rays, rng, **kw):
    return trace_heliostat(
        row["x_mm"],
        row["y_mm"],
        row["rot_az_deg"],
        row["rot_el_deg"],
        row["c3"],
        row["c4"],
        row["c5"],
        row["solar_az_deg"],
        row["solar_el_deg"],
        secondary,
        RECEIVER,
        n_rays,
        rng,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. zero perturbation is bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_zero_perturbation_bit_identical_cone(shape):
    nominal, row = SHAPES[shape]
    zeroed = _perturbed(nominal, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, tip_mrad=0.0, tilt_mrad=0.0)

    out_a = _trace_cone(nominal, row, return_secondary_flux=True)
    out_b = _trace_cone(zeroed, row, return_secondary_flux=True)

    np.testing.assert_array_equal(out_a["flux"], out_b["flux"])
    assert out_a["power_w"] == out_b["power_w"]
    assert out_a["incident_power_w"] == out_b["incident_power_w"]
    assert out_a["counters"] == out_b["counters"]
    np.testing.assert_array_equal(out_a["secondary_flux"], out_b["secondary_flux"])
    assert out_a["secondary_power_w"] == out_b["secondary_power_w"]


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_zero_perturbation_bit_identical_mc(shape):
    nominal, row = SHAPES[shape]
    zeroed = _perturbed(nominal, dx_mm=0.0, dy_mm=0.0, dz_mm=0.0, tip_mrad=0.0, tilt_mrad=0.0)

    out_a = _trace_mc(nominal, row, 20_000, np.random.default_rng(7), return_secondary_hits=True)
    out_b = _trace_mc(zeroed, row, 20_000, np.random.default_rng(7), return_secondary_hits=True)

    np.testing.assert_array_equal(out_a["xy"], out_b["xy"])
    np.testing.assert_array_equal(out_a["secondary_xy"], out_b["secondary_xy"])
    assert out_a["counters"] == out_b["counters"]
    assert out_a["watts_per_ray"] == out_b["watts_per_ray"]


# ---------------------------------------------------------------------------
# 2a. decenter: exact rigid translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
@pytest.mark.parametrize("axis", ["dx", "dy", "dz"])
def test_decenter_is_a_rigid_translation(shape, axis):
    """A pure decenter (no tip/tilt) must satisfy the exact, general rigid
    translation law: reflecting off the DECENTRED secondary is the same as
    translating the incoming ray's origin by ``-decenter``, reflecting off
    the UNPERTURBED secondary, then translating the resulting hit point
    (only -- direction is unaffected by a pure translation) back by
    ``+decenter``. True for any correct "move the mirror" implementation,
    independent of the cone/hyperboloid equations' own details.
    """
    nominal, row = SHAPES[shape]
    p, d = _secondary_incoming_rays(nominal, RECEIVER, row)
    kw = DECENTER_AXES[axis]
    decenter = np.array([kw["dx_mm"], kw["dy_mm"], kw["dz_mm"]])
    perturbed = _perturbed(nominal, **kw)

    p2_pert, d2_pert, on_pert = perturbed.redirect(p.copy(), d.copy(), {})
    p2_nom, d2_nom, on_nom = nominal.redirect(p.copy() - decenter[:, None], d.copy(), {})

    np.testing.assert_array_equal(on_pert, on_nom)
    np.testing.assert_allclose(p2_pert, p2_nom + decenter[:, None], rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(d2_pert, d2_nom, rtol=1e-12, atol=1e-12)


def test_axicon_dz_decenter_equals_shifted_apex_height():
    """Spec's own suggested concrete check: a pure dz decenter of an axicon
    is exactly the same cone as one with ``apex_height_mm`` itself raised
    by dz -- "the fold height changes by exactly dz". Checked end to end
    through the cone backend's full receiver flux map (not just
    ``redirect``), using ``dz_mm`` on one instance against a directly
    shifted ``apex_height_mm`` on a second, ordinary, unperturbed instance.
    """
    z0, half_angle, aperture = 27000.0, 20.0, 14000.0
    dz = 450.0
    perturbed = AxiconSecondary(
        apex_height_mm=z0, half_angle_deg=half_angle, aperture_radius_mm=aperture, dz_mm=dz
    )
    shifted = AxiconSecondary(
        apex_height_mm=z0 + dz, half_angle_deg=half_angle, aperture_radius_mm=aperture
    )

    out_p = _trace_cone(perturbed, AXICON_ROW)
    out_s = _trace_cone(shifted, AXICON_ROW)

    np.testing.assert_allclose(out_p["flux"], out_s["flux"], rtol=1e-9, atol=1e-9)
    assert out_p["power_w"] == pytest.approx(out_s["power_w"], rel=1e-12)
    assert out_p["counters"] == out_s["counters"]


def test_cassegrain_dz_decenter_equals_shifted_vertex_z():
    """Same identity as the axicon check above, for the Cassegrain's
    ``vertex_z_mm``: the relay's internal geometry (``rim_z_mm``,
    ``zeta_max``) is built entirely from heights measured relative to
    ``vertex_z_mm``, so it too is translation-invariant along z.
    """
    vz, vr, k, aperture = 26993.999446877, 26112.078893738, -5.317616535, 14000.0
    dz = -300.0
    perturbed = CassegrainSecondary(
        vertex_z_mm=vz, vertex_radius_mm=vr, conic=k, aperture_radius_mm=aperture, dz_mm=dz
    )
    shifted = CassegrainSecondary(
        vertex_z_mm=vz + dz, vertex_radius_mm=vr, conic=k, aperture_radius_mm=aperture
    )

    out_p = _trace_cone(perturbed, CASSEGRAIN_ROW)
    out_s = _trace_cone(shifted, CASSEGRAIN_ROW)

    np.testing.assert_allclose(out_p["flux"], out_s["flux"], rtol=1e-9, atol=1e-9)
    assert out_p["power_w"] == pytest.approx(out_s["power_w"], rel=1e-12)
    assert out_p["counters"] == out_s["counters"]


# ---------------------------------------------------------------------------
# 2b. tip/tilt: exact rigid rotation
# ---------------------------------------------------------------------------


def test_flat_mirror_tilt_doubles_reflected_ray_rotation():
    """Textbook law, independent of this module entirely: tilting a flat
    mirror's normal by angle theta, ABOUT AN AXIS PERPENDICULAR TO THE
    PLANE OF INCIDENCE (the plane containing the incident ray and the
    normal), rotates the reflection of a FIXED incident ray by exactly
    2*theta about that same axis. (A tilt axis not perpendicular to that
    plane does not obey a simple doubling -- angle of incidence and angle
    of reflection only move in lockstep within the plane they are measured
    in -- so the axis choice below is not arbitrary.) This is the classic
    identity spec §E2's tip/tilt is built on (a rigid rotation of a
    locally-flat mirror patch); the end-to-end equivariance test below pins
    the general (unrestricted-axis) law on the real cone/hyperboloid
    classes, this one pins the textbook special case itself -- exact for
    any angle, not a small-angle approximation.
    """
    d_in = np.array([0.2, -0.1, -1.0])
    d_in /= np.linalg.norm(d_in)
    n0 = np.array([0.05, 0.03, 1.0])
    n0 /= np.linalg.norm(n0)

    def reflect(d, n):
        return d - 2.0 * np.dot(d, n) * n

    def rodrigues(axis, theta):
        c, s = np.cos(theta), np.sin(theta)
        k = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + s * k + (1.0 - c) * (k @ k)

    theta = 3.0e-3  # rad -- not infinitesimal; the law holds exactly anyway
    axis = np.cross(d_in, n0)
    axis /= np.linalg.norm(axis)
    r = rodrigues(axis, theta)
    r2 = rodrigues(axis, 2.0 * theta)

    d_out0 = reflect(d_in, n0)
    d_out1 = reflect(d_in, r @ n0)

    np.testing.assert_allclose(d_out1, r2 @ d_out0, atol=1e-10)


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
@pytest.mark.parametrize("axis", ["tip", "tilt"])
def test_tip_tilt_is_a_rigid_rotation(shape, axis):
    """A pure tip/tilt (no decenter) must satisfy the exact isometry law
    any rigid rotation of a mirror obeys: reflecting off the ROTATED
    secondary is the same as rotating the incoming ray (origin about the
    vertex, and direction) into the surface's own frame by R^-1, reflecting
    off the UNPERTURBED secondary, then rotating the result (hit point
    about the vertex, and direction) back by R. Exact for any rotation
    angle, independent of the cone/hyperboloid equations' own details.
    """
    nominal, row = SHAPES[shape]
    p, d = _secondary_incoming_rays(nominal, RECEIVER, row)
    vertex = np.array([0.0, 0.0, _vertex_z(nominal)])
    r = _rotation_matrix(**TILT_AXES[axis])
    perturbed = _perturbed(nominal, **TILT_AXES[axis])

    p2_pert, d2_pert, on_pert = perturbed.redirect(p.copy(), d.copy(), {})

    p_local = r.T @ (p - vertex[:, None]) + vertex[:, None]
    d_local = r.T @ d
    p2_nom, d2_nom, on_nom = nominal.redirect(p_local, d_local, {})

    np.testing.assert_array_equal(on_pert, on_nom)
    p2_expected = r @ (p2_nom - vertex[:, None]) + vertex[:, None]
    d2_expected = r @ d2_nom
    np.testing.assert_allclose(p2_pert, p2_expected, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(d2_pert, d2_expected, rtol=1e-8, atol=1e-9)


def test_secondary_uv_is_perturbation_invariant():
    """secondary_uv of a hit point must not depend on where in world space
    the secondary sits: the same LOCAL point on the physical surface
    reports the same (u, v) whether or not the secondary is
    decentred/tilted -- otherwise a spec §C secondary flux map would shift
    or smear under a perturbation instead of staying anchored to the part.
    """
    nominal = AXICON
    perturbed = _perturbed(nominal, **COMBINED_PERTURBATION)

    # A handful of points on the nominal cone's own flank.
    h = np.array([100.0, 5000.0, 12000.0])
    phi = np.array([0.3, -1.2, 2.5])
    x = h * np.sin(phi)
    y = -h * np.cos(phi)
    z = nominal.apex_height_mm + h * np.tan(np.deg2rad(nominal.half_angle_deg))
    p_local = np.vstack([x, y, z])

    uv_nominal = secondary_uv(nominal, p_local)

    # The SAME physical points, expressed in world space under the
    # perturbation (forward rigid transform).
    r = _rotation_matrix(perturbed.tip_mrad, perturbed.tilt_mrad)
    vertex = np.array([0.0, 0.0, nominal.apex_height_mm])
    decenter = np.array([perturbed.dx_mm, perturbed.dy_mm, perturbed.dz_mm])
    p_world = r @ (p_local - vertex[:, None]) + (vertex + decenter)[:, None]

    uv_perturbed = secondary_uv(perturbed, p_world)
    np.testing.assert_allclose(uv_perturbed, uv_nominal, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. MC vs cone agree under one perturbed configuration per shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_mc_vs_cone_agree_under_perturbation(shape):
    """Same tolerance convention as tests/test_cone_vs_mc.py: a fixed
    sanity band (power within 2%, centroid within 15 mm) around a 200,000-
    ray Monte Carlo trace, on a genuinely perturbed (decentred + tilted)
    secondary.

    ``_trace_cone`` (module-level ``KERNEL = sunshape_kernel("super_gauss")``)
    and ``_trace_mc`` (no ``sampler=``, so the app-wide Buie default per
    commit 7c4fd08) are a real sunshape mismatch, harmless for what this
    test asserts -- power and centroid, not shape/width -- because
    ``WINDOW_MM = 6000.0`` above is deliberately oversized precisely so
    neither backend's beam clips it (see that constant's own docstring);
    measured directly (both samplers, this geometry): power differs
    ~0.003% and centroid differs a few mm, both far under this test's own
    tolerances. With nothing clipped, total power and centroid position
    cannot depend on sunshape angular width -- only a peak-flux or
    map-shape version of this comparison would need the sampler matched."""
    nominal, row = SHAPES[shape]
    perturbed = _perturbed(nominal, **COMBINED_PERTURBATION)

    cone_out = _trace_cone(perturbed, row)
    mc_out = _trace_mc(perturbed, row, 200_000, np.random.default_rng(20260827))

    mc_n = mc_out["xy"].shape[1]
    mc_power = mc_n * mc_out["watts_per_ray"]
    assert mc_power == pytest.approx(cone_out["power_w"], rel=0.02)

    mc_centroid = mc_out["xy"].mean(axis=1)
    u_mid = 0.5 * (cone_out["u_edges"][:-1] + cone_out["u_edges"][1:])
    v_mid = 0.5 * (cone_out["v_edges"][:-1] + cone_out["v_edges"][1:])
    flux = cone_out["flux"]
    cone_centroid_u = float(np.sum(flux.sum(axis=0) * u_mid) / flux.sum())
    cone_centroid_v = float(np.sum(flux.sum(axis=1) * v_mid) / flux.sum())

    assert mc_centroid[0] == pytest.approx(cone_centroid_u, abs=15.0)
    assert mc_centroid[1] == pytest.approx(cone_centroid_v, abs=15.0)


# ---------------------------------------------------------------------------
# 4. spec §C secondary-flux energy pin holds under perturbation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_energy_pin_cone_under_perturbation(shape):
    """tests/test_secondary_flux.py::test_energy_pin_cone's own pin
    (secondary-incident power ~= power leaving the mirror toward it, and
    the returned flux grid re-integrates to the returned total power),
    reproduced against a perturbed (decentred + tilted) secondary."""
    nominal, row = SHAPES[shape]
    perturbed = _perturbed(nominal, **COMBINED_PERTURBATION)

    out = _trace_cone(perturbed, row, return_secondary_flux=True, secondary_flux_grid=(96, 96))

    incident = out["incident_power_w"]
    secondary_power = out["secondary_power_w"]
    assert secondary_power == pytest.approx(incident, rel=0.01)

    n_v, n_u = out["secondary_flux"].shape
    areas_m2 = secondary_bin_areas_m2(perturbed, (n_u, n_v))
    reintegrated = float(np.sum(out["secondary_flux"] * areas_m2))
    assert reintegrated == pytest.approx(secondary_power, rel=1e-9)


@pytest.mark.parametrize("shape", ["axicon", "cassegrain"])
def test_energy_pin_mc_under_perturbation(shape):
    """tests/test_secondary_flux.py::test_energy_pin_mc's own pin
    (histogramming raw secondary hits through secondary_uv/
    secondary_bin_areas_m2 must exactly reproduce hit_secondary count *
    watts-per-ray), reproduced against a perturbed secondary -- this is
    also what exercises mc.py's ``secondary_xy`` now carrying the full
    (x, y, z) world point rather than a plan (x, y) pair, and
    ``secondary_uv``'s world -> local undo, together."""
    nominal, row = SHAPES[shape]
    perturbed = _perturbed(nominal, **COMBINED_PERTURBATION)

    out = _trace_mc(perturbed, row, 20_000, np.random.default_rng(1), return_secondary_hits=True)
    watts_per_ray = out["watts_per_ray"]
    sec_xyz = out["secondary_xy"]
    assert sec_xyz.shape[0] == 3  # (x, y, z), not a plan-only pair
    n_hit = sec_xyz.shape[1]
    assert n_hit == out["counters"]["hit_secondary"]
    expected_power = n_hit * watts_per_ray

    uv = secondary_uv(perturbed, sec_xyz)
    (u0, u1), (v0, v1) = secondary_uv_extent(perturbed)
    n_u, n_v = 128, 128
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    counts, _, _ = np.histogram2d(uv[1], uv[0], bins=[v_edges, u_edges])
    assert counts.sum() == n_hit  # no hit lost to a binning/extent bug

    areas_m2 = secondary_bin_areas_m2(perturbed, (n_u, n_v))
    flux = counts * watts_per_ray / areas_m2
    power_via_histogram = float(np.sum(flux * areas_m2))
    assert power_via_histogram == pytest.approx(expected_power, rel=1e-6)
