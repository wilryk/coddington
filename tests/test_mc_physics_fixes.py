"""Pin tests for two Monte Carlo physics bugs fixed together:

1. **Specularity plane.** ``specularity_mrad`` must scatter the REFLECTED
   ray isotropically about ITSELF (the SolTrace convention: micro-facet
   roughness acts on the outgoing ray, after reflection), not as a
   tangent-plane tilt of the surface normal (that is the SLOPE-error
   convention -- correct for ``slope_error_mrad``, wrong for
   ``specularity_mrad``). The buggy version used the mirror's own
   ``(u, v)`` -- perpendicular to the NORMAL, not to the outgoing ray --
   which is only the same plane at normal incidence; at oblique incidence
   the two axes decompose as ``(u, v) = (sin(theta_i) d_out + cos(theta_i)
   t_hat, q_hat)`` (see the derivation this module's tests were built from,
   also in the fix's commit message / report), so the buggy perturbation's
   TANGENTIAL component (``t_hat``, in the incidence plane) comes out
   scaled by ``cos(theta_i)`` while the SAGITTAL component (``q_hat``,
   invariant under reflection about a fixed normal) is untouched -- an
   anisotropic, under-broadened scatter cone instead of an isotropic one.

2. **Source-disk centring.** The Monte Carlo source disk samples ray
   origins in a region sized/centred to cover the NOMINAL rectangular
   mirror; a custom design whose sketch is offset from the heliostat's
   pivot (a hand-drawn outline, a flower whose petals don't average back
   to the origin) could have part of its area sit outside that disk,
   silently losing the rays that should have illuminated it. The disk must
   be centred on the design's own bbox centroid instead.

Both tests below construct the exact geometry the bug/fix affects rather
than relying on any stored fixture, and cross-check against the cone
backend, which already implements both conventions correctly (see
``cone.py``'s ``sunshape_kernel`` and its own mirror-sample grid, which has
no source-disk concept to get wrong).
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.geometry.aperture import Rect
from heliostat.geometry.design import Facet, Flat, HeliostatDesign
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import _mirror_frame, _sun_vector, trace_heliostat

# ---------------------------------------------------------------------------
# shared geometry helpers


def _reflect(rot_az_deg: float, rot_el_deg: float, sun_az_deg: float, sun_el_deg: float):
    """The mirror normal, incoming, and (exact, unperturbed) outgoing ray
    direction for one pointing/sun-position pair -- the same construction
    :func:`heliostat.trace.mc.trace_heliostat` uses internally, computed
    once up front so the tests know exactly what geometry they are at."""
    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)
    s = _sun_vector(sun_az_deg, sun_el_deg)
    d_in = -s
    d_out = d_in - 2.0 * np.dot(d_in, n) * n
    return n, u, v, d_in, d_out


# ---------------------------------------------------------------------------
# bug 1: specularity must be isotropic about the reflected ray


# rot_az=0 / sun_az=90 puts both the mirror normal and the sun vector in the
# world x-z plane (see `_mirror_frame`'s plain-trig azimuth vs `_sun_vector`'s
# compass-bearing azimuth -- az=90 deg is the compass angle that cancels the
# "pi/2 - az" flip), so incidence is a clean 2-D problem with an exactly
# computable incidence angle: rot_el=75, sun_el=30 gives cos(theta_i) =
# |n . (-s)| = cos(45 deg) exactly (both are 45 deg off the shared x-z axis
# in opposite senses).
SPEC_ROT_AZ, SPEC_ROT_EL = 0.0, 75.0
SPEC_SUN_AZ, SPEC_SUN_EL = 90.0, 30.0
SPEC_MRAD = 15.0  # >> the pinned super-Gauss sigma (2.4 mrad), so the sun's
# own angular width is a small correction, not the dominant signal.
SPEC_N_RAYS = 400_000


def _specularity_only_receiver():
    """A FlatWindowReceiver placed so the 45-deg-incidence chief ray lands
    at its centre -- window tilt relative to the reflected ray does not
    matter for what this test measures (see below), only that every
    scattered ray in the cone actually reaches the window. Sized generously
    (6 m half-width against a combined ~21 mrad x 35 m ~= 0.7 m spot sigma)
    so neither backend's power comparison is contaminated by a systematic
    edge-clipping mismatch between MC's raw ray counting and the cone
    kernel's analytic tail handling -- a real, separate effect from the
    isotropy/magnitude bug this module pins, confirmed by sweeping window
    size until the two backends' power agreement stopped improving."""
    _, _, _, _, d_out = _reflect(SPEC_ROT_AZ, SPEC_ROT_EL, SPEC_SUN_AZ, SPEC_SUN_EL)
    z_mm = 30_000.0
    t = z_mm / d_out[2]
    cx, cy = d_out[0] * t, d_out[1] * t
    return FlatWindowReceiver(
        z_mm=z_mm, half_u_mm=6000.0, half_v_mm=6000.0, facing="down",
        center_x_mm=cx, center_y_mm=cy,
    )


def _tangential_sagittal_rms(specularity_mrad: float, seed: int = 123):
    """Trace the legacy rectangle at the fixed 45-deg-incidence geometry
    and return (rms_tangential, rms_sagittal, n_landed), in radians, of the
    REFLECTED RAY DIRECTION's deviation from the exact chief direction
    ``d_out`` -- reconstructed exactly from ``paths`` (mirror hit -> receiver
    hit is a straight line with no secondary in between, and the docstring
    of ``trace_heliostat``'s ``paths`` output guarantees an exact receiver
    hit reconstruction for a flat window), so this measurement is exact
    regardless of how the receiver window happens to be tilted relative to
    ``d_out`` -- unlike the footprint on the window itself, which would be
    foreshortened by that tilt and is deliberately not what is measured
    here.

    ``t_hat``/``q_hat`` are the tangential (in the incidence plane, spanned
    by the normal and the sun vector) and sagittal (perpendicular to it)
    axes relative to the TRUE outgoing chief ray -- the correct convention
    this fix targets, as opposed to the buggy mirror-(u, v) axes.
    """
    n, u, v, d_in, d_out = _reflect(SPEC_ROT_AZ, SPEC_ROT_EL, SPEC_SUN_AZ, SPEC_SUN_EL)
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()
    rng = np.random.default_rng(seed)
    out = trace_heliostat(
        0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
        secondary, receiver, SPEC_N_RAYS, rng,
        specularity_mrad=specularity_mrad, return_paths=True,
    )
    paths = out["paths"]
    mir, rec = paths[1], paths[3]
    dvec = rec - mir
    dvec /= np.linalg.norm(dvec, axis=0)

    t_hat = n - np.dot(n, d_out) * d_out
    t_hat /= np.linalg.norm(t_hat)
    q_hat = np.cross(d_out, t_hat)

    delta = dvec - d_out[:, None]
    tang = delta.T @ t_hat
    sag = delta.T @ q_hat
    return (
        float(np.sqrt(np.mean(tang**2))),
        float(np.sqrt(np.mean(sag**2))),
        dvec.shape[1],
    )


def test_specularity_isotropic_about_reflected_ray():
    """At 45-deg incidence with pure specularity (no slope error), the
    tangential and sagittal RMS widths of the reflected-ray scatter must
    agree within Monte Carlo noise -- the isotropy the SolTrace convention
    requires and the pre-fix code broke (it compressed the tangential
    component by cos(45 deg) = 0.7071, since it perturbed in the mirror's
    own (u, v), not in the plane perpendicular to the outgoing ray)."""
    rms_tang, rms_sag, n = _tangential_sagittal_rms(SPEC_MRAD)
    # The legacy 5x3 m rectangle is much smaller than the (oversized, fixed)
    # source disk it's sampled against, so only a fraction of emitted rays
    # ever hit the mirror at all (matching the ~27% hit rate the mc_parity
    # golden fixtures also see at similar geometry) -- this floor just
    # guards against a receiver/geometry mistake dropping the count to
    # near-zero, not against that expected, unrelated inefficiency.
    assert n > SPEC_N_RAYS * 0.2, "unexpectedly low landing fraction for this geometry"

    # Noise floor: each RMS is estimated from `n` roughly-Gaussian samples,
    # se(rms)/rms ~ 1/sqrt(2n). At n ~ 3.6e5 that is ~0.09%; use a generous
    # 3% band (30x that) to stay far from flaky without hiding the ~30%
    # ratio the pre-fix bug actually produced (see module docstring).
    ratio = rms_tang / rms_sag
    assert ratio == pytest.approx(1.0, rel=0.03), (
        f"tangential/sagittal RMS ratio {ratio:.4f} is not isotropic "
        f"(tang={rms_tang * 1e3:.3f} mrad, sag={rms_sag * 1e3:.3f} mrad); "
        f"the pre-fix bug gives ~cos(45deg)=0.7071 here"
    )

    # Combined 2-D RMS should recover sigma_spec * sqrt(2) (isotropic,
    # independent per-axis sigma == specularity_mrad), not the ~13% smaller
    # value the pre-fix bug gave.
    combined_mrad = np.hypot(rms_tang, rms_sag) * 1e3
    expected_mrad = SPEC_MRAD * np.sqrt(2.0)
    assert combined_mrad == pytest.approx(expected_mrad, rel=0.03), (
        f"combined RMS {combined_mrad:.3f} mrad vs expected {expected_mrad:.3f} mrad "
        f"(sigma_spec * sqrt(2)); pre-fix bug under-broadens this by ~13%"
    )


def test_specularity_matches_cone_backend():
    """Cone already applies ``specularity_mrad`` as an isotropic angular
    broadening with no doubling (:func:`heliostat.trace.cone.sunshape_kernel`);
    the fixed Monte Carlo backend must land on the same total power at the
    same 45-deg-incidence geometry, within the two backends' own noise."""
    secondary = NoSecondary()
    receiver = _specularity_only_receiver()

    rng = np.random.default_rng(99)
    mc = trace_heliostat(
        0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
        secondary, receiver, SPEC_N_RAYS, rng, specularity_mrad=SPEC_MRAD,
    )
    power_mc = mc["watts_per_ray"] * mc["counters"]["in_window"]

    kernel = sunshape_kernel("super_gauss", specularity_mrad=SPEC_MRAD)
    cone = trace_heliostat_cone(
        0.0, 0.0, SPEC_ROT_AZ, SPEC_ROT_EL, 0.0, 0.0, 0.0, SPEC_SUN_AZ, SPEC_SUN_EL,
        secondary, receiver, kernel,
    )
    power_cone = cone["power_w"]

    n_landed = mc["counters"]["in_window"]
    se_power = power_mc / np.sqrt(max(n_landed, 1))
    assert abs(power_mc - power_cone) < max(4 * se_power, 2.0), (
        f"MC power {power_mc:.2f}W vs cone {power_cone:.2f}W "
        f"(4se={4 * se_power:.2f}W, n_landed={n_landed})"
    )


# ---------------------------------------------------------------------------
# bug 2: source disk must be centred on the design's own bbox, not the pivot


OFF_ROT_AZ, OFF_ROT_EL = 0.0, 60.0
OFF_SUN_AZ, OFF_SUN_EL = 90.0, 45.0
OFF_N_RAYS = 300_000
# Offset chosen so the facet's nearest edge (3600 mm from the pivot) sits
# entirely outside the fixed default source disk (SOURCE_DISK_RADIUS_MM =
# 3500 mm): the pre-fix disk, centred on the pivot, cannot emit a single ray
# toward this facet at all.
FACET_OFFSET_MM = (4000.0, 0.0)
FACET_SIZE_MM = 800.0


def _offcentre_designs():
    off = HeliostatDesign(
        [Facet(region=Rect(FACET_SIZE_MM, FACET_SIZE_MM), surface=Flat(), offset_mm=FACET_OFFSET_MM)]
    )
    centred = HeliostatDesign(
        [Facet(region=Rect(FACET_SIZE_MM, FACET_SIZE_MM), surface=Flat(), offset_mm=(0.0, 0.0))]
    )
    return off, centred


def _offcentre_receiver():
    """Wide enough and shifted enough in v to catch BOTH the off-centre
    facet's image (whose hit point is shifted by the full 4 m offset along
    the mirror's own ``u`` -- here exactly the world y axis, see the
    in-file derivation in the specularity section above -- since the
    reflected direction has zero y-component at this geometry, that shift
    passes straight through to the receiver) and the centred facet's image,
    without needing separate windows per design."""
    n, u, v, d_in, d_out = _reflect(OFF_ROT_AZ, OFF_ROT_EL, OFF_SUN_AZ, OFF_SUN_EL)
    z_mm = 30_000.0
    t = z_mm / d_out[2]
    cx, cy = d_out[0] * t, d_out[1] * t
    return FlatWindowReceiver(
        z_mm=z_mm, half_u_mm=3000.0, half_v_mm=5000.0, facing="down",
        center_x_mm=cx, center_y_mm=cy + FACET_OFFSET_MM[0] / 2.0,
    )


def test_offcentre_outline_matches_centred_after_translation():
    """An off-centre custom outline must collect the same total power (and,
    since the facet is flat and merely translated with the source disk that
    now follows it, the exact same hit pattern) as the identical outline
    re-centred on the pivot -- after accounting for the small change in
    geometry a multi-metre translation makes to incidence/distance, which
    at these field scales (30 m source distance) is negligible next to
    Monte Carlo noise. Before the fix, the off-centre case collected
    exactly zero power (its facet sits entirely outside the fixed 3.5 m
    source disk when that disk is centred on the pivot instead of the
    facet)."""
    design_off, design_centred = _offcentre_designs()
    secondary = NoSecondary()
    receiver = _offcentre_receiver()

    results = {}
    for label, design in [("off", design_off), ("centred", design_centred)]:
        rng = np.random.default_rng(42)
        out = trace_heliostat(
            0.0, 0.0, OFF_ROT_AZ, OFF_ROT_EL, 0.0, 0.0, 0.0, OFF_SUN_AZ, OFF_SUN_EL,
            secondary, receiver, OFF_N_RAYS, rng, design=design,
        )
        results[label] = out

    n_off = results["off"]["counters"]["in_window"]
    n_cen = results["centred"]["counters"]["in_window"]
    assert n_off > 0, "off-centre outline collected zero rays -- the pre-fix bug"

    power_off = results["off"]["watts_per_ray"] * n_off
    power_cen = results["centred"]["watts_per_ray"] * n_cen

    # Same seed, same disk-relative sampling once each disk is centred on
    # its own facet, so for a flat, merely-translated facet the two traces
    # are bit-identical draws of the same underlying random stream --
    # `hit_mirror` should match exactly, not just approximately.
    assert results["off"]["counters"]["hit_mirror"] == results["centred"]["counters"]["hit_mirror"]

    se_power = power_cen / np.sqrt(max(n_cen, 1))
    assert abs(power_off - power_cen) < max(4 * se_power, 2.0), (
        f"off-centre power {power_off:.2f}W vs centred {power_cen:.2f}W "
        f"(4se={4 * se_power:.2f}W)"
    )


def test_offcentre_outline_mc_matches_cone():
    """MC total power for the off-centre outline must agree with the cone
    backend within noise -- cone has no source-disk concept to get wrong
    (it samples the mirror surface directly), so it is an independent
    reference unaffected by this bug."""
    design_off, _ = _offcentre_designs()
    secondary = NoSecondary()
    receiver = _offcentre_receiver()

    rng = np.random.default_rng(7)
    mc = trace_heliostat(
        0.0, 0.0, OFF_ROT_AZ, OFF_ROT_EL, 0.0, 0.0, 0.0, OFF_SUN_AZ, OFF_SUN_EL,
        secondary, receiver, OFF_N_RAYS, rng, design=design_off,
    )
    n_landed = mc["counters"]["in_window"]
    assert n_landed > 0, "off-centre outline collected zero rays in MC -- the pre-fix bug"
    power_mc = mc["watts_per_ray"] * n_landed

    kernel = sunshape_kernel("super_gauss")
    cone = trace_heliostat_cone(
        0.0, 0.0, OFF_ROT_AZ, OFF_ROT_EL, 0.0, 0.0, 0.0, OFF_SUN_AZ, OFF_SUN_EL,
        secondary, receiver, kernel, design=design_off,
    )
    power_cone = cone["power_w"]
    assert power_cone > 0, "off-centre outline collected zero power in cone -- test geometry bug"

    se_power = power_mc / np.sqrt(max(n_landed, 1))
    assert abs(power_mc - power_cone) < max(4 * se_power, 2.0), (
        f"MC power {power_mc:.2f}W vs cone {power_cone:.2f}W "
        f"(4se={4 * se_power:.2f}W, n_landed={n_landed})"
    )
