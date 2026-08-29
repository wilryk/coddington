"""Pin tests for the cone backend's slope-error BEAM-FRAME anisotropy.

Background (measured this session, see the owner's session report --
scratchpad/slope_anisotropy_report.md -- and this module's own docstrings
in ``heliostat.trace.cone``): ``slope_error_mrad`` (``mc.py``'s
``_perturb_unit``) perturbs the mirror's local surface normal with a
Gaussian that is isotropic in the MIRROR's own tangent frame -- the
physically correct convention, since surface waviness cannot depend on
where the sun is. Reflected into the beam, that mirror-isotropic
perturbation is ELLIPTICAL: tangential (in-plane-of-incidence) broadening
matches the full ``2 * slope_error_mrad``, while sagittal is compressed by
``cos(theta_i)`` -- confirmed to ~0.3% at every incidence angle tested up
to 60 deg. The cone backend previously folded slope error in as an
ISOTROPIC angular broadening (the same treatment correctly used for
specularity and pointing error, which really ARE beam-isotropic by
convention), disagreeing with Monte Carlo by ~5% total-spot-RMS on
realistically focused manuscript-field heliostats at high incidence and a
few mrad of slope error.

``heliostat.trace.cone.trace_heliostat_cone`` now folds slope error's
contribution in anisotropically, via a per-sample linear warp of that
sample's own Jacobian (mapping a fixed-shape, still fully ISOTROPIC kernel
through ``jac @ warp`` instead of ``jac`` alone) -- see that function's own
"slope-error beam-frame anisotropy" comment block for the moment-matching
derivation. ``sunshape_kernel``'s own returned kernel table is completely
unaffected (bit-identical in every case; only two private attributes,
``_slope_sigma_rad``/``_non_slope_var_rad2``, are attached for
``trace_heliostat_cone`` to read).

Three things are pinned here, matching gates 1-3 of the task brief (gate 4,
the existing parity suites passing unmodified, is verified by running them,
not by anything in this file; gate 5 is this file's own existence):

1. Zero slope error is bit-identical (both backends), and normal incidence
   is bit-identical at any slope error (cone) -- ``TestBitIdentity``.
2. The cone-vs-MC total-spot-RMS disagreement on realistically focused
   manuscript-field heliostats closes from several percent to within MC
   noise -- ``TestFocusedFieldDisagreementCloses``.
3. The cone spot's own tangential/sagittal second moments -- not just a
   smaller total RMS -- match what MC produces on the SAME real, focused
   geometry -- ``TestEllipseRatio``. NOT tested as a literal "ratio ==
   cos(theta_i)" on the receiver window directly: that comparison was tried
   first and rejected (see ``TestEllipseRatio``'s own module-level comment)
   because a fixed receiver window foreshortens the tangential/sagittal
   directions differently depending on the chief ray's own world
   orientation, a confound unrelated to slope error that a raw receiver-
   position ratio cannot separate out -- matching MC on the identical
   receiver cancels that confound, and MC's own reflected-RAY-DIRECTION
   ellipse (receiver-projection-independent by construction) already tracks
   cos(theta_i) to ~0.3pct, unchanged by this fix (see the report's Part 1).

Every MC-vs-cone comparison here explicitly pins BOTH sides to the SAME
sunshape (``sampler=BuieSampler()`` / ``sunshape_kernel("buie", ...)``) --
this session's report flags a sampler mismatch (cone on ``super_gauss``,
MC left on its ``buie`` default) that previously contaminated an earlier,
now-superseded version of the Part 2b field measurement with a
sunshape-width artefact on top of the real anisotropy signal, the same
pitfall ``tests/test_pointing_error.py`` already guards against.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.cone import RadialKernel, sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import _mirror_frame, _sun_vector, trace_heliostat
from heliostat.trace.samplers import BuieSampler
from heliostat.web.app import _geometry_for, _solve_for, resolve_optics_params

# ---------------------------------------------------------------------------
# shared geometry helpers (same construction as mc.trace_heliostat/cone.py's
# own internals, and the same pattern test_mc_physics_fixes._reflect /
# test_pointing_error.py use)


def _reflect(rot_az_deg: float, rot_el_deg: float, sun_az_deg: float, sun_el_deg: float):
    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)
    s = _sun_vector(sun_az_deg, sun_el_deg)
    d_in = -s
    d_out = d_in - 2.0 * np.dot(d_in, n) * n
    return n, u, v, d_in, d_out, s


def _receiver_for(rot_az_deg, rot_el_deg, sun_az_deg, sun_el_deg, z_mm=30_000.0, half_mm=16_000.0):
    """A generously-sized flat window centred on the exact unperturbed
    chief ray -- generous so slope error's tail never gets edge-clipped,
    which would contaminate a pure-anisotropy measurement with a receiver-
    window artefact (the same rationale ``test_mc_physics_fixes.
    _specularity_only_receiver`` and this session's own report use)."""
    _, _, _, _, d_out, _ = _reflect(rot_az_deg, rot_el_deg, sun_az_deg, sun_el_deg)
    t = z_mm / d_out[2]
    cx, cy = d_out[0] * t, d_out[1] * t
    return FlatWindowReceiver(
        z_mm=z_mm, half_u_mm=half_mm, half_v_mm=half_mm, facing="down",
        center_x_mm=cx, center_y_mm=cy,
    )


def _rms_about_own_centroid(xy: np.ndarray) -> float:
    if xy.shape[1] == 0:
        return float("nan")
    cx, cy = np.mean(xy[0]), np.mean(xy[1])
    return float(np.sqrt(np.mean((xy[0] - cx) ** 2 + (xy[1] - cy) ** 2)))


def _cone_moments(cone: dict):
    """(centroid_uv_mm, rms_mm) -- same recipe as test_cone_convergence's/
    test_pointing_error's own ``_cone_centroid_rms``/``_centroid_rms``."""
    flux = cone["flux"]
    u_mid = 0.5 * (cone["u_edges"][:-1] + cone["u_edges"][1:])
    v_mid = 0.5 * (cone["v_edges"][:-1] + cone["v_edges"][1:])
    total = flux.sum()
    cen_u = float((flux.sum(axis=0) * u_mid).sum() / total)
    cen_v = float((flux.sum(axis=1) * v_mid).sum() / total)
    uu, vv = np.meshgrid(u_mid, v_mid)
    rms = float(np.sqrt((((uu - cen_u) ** 2 + (vv - cen_v) ** 2) * flux).sum() / total))
    return (cen_u, cen_v), rms


def _cone_tang_sag_rms_mm(cone: dict, t_hat_uv, q_hat_uv):
    """Tangential/sagittal RMS (mm), about the flux map's OWN centroid, in
    the world-(x, y)-aligned ``(t_hat_uv, q_hat_uv)`` basis -- valid because
    ``FlatWindowReceiver``'s own ``uv`` IS world ``(x, y)`` (see that
    class's docstring)."""
    (cen_u, cen_v), _ = _cone_moments(cone)
    flux = cone["flux"]
    u_mid = 0.5 * (cone["u_edges"][:-1] + cone["u_edges"][1:])
    v_mid = 0.5 * (cone["v_edges"][:-1] + cone["v_edges"][1:])
    uu, vv = np.meshgrid(u_mid, v_mid)
    du, dv = uu - cen_u, vv - cen_v
    tang = du * t_hat_uv[0] + dv * t_hat_uv[1]
    sag = du * q_hat_uv[0] + dv * q_hat_uv[1]
    total = flux.sum()
    rms_t = float(np.sqrt((tang**2 * flux).sum() / total))
    rms_s = float(np.sqrt((sag**2 * flux).sum() / total))
    return rms_t, rms_s


def _tq_hat_uv(rot_az_deg, rot_el_deg, sun_az_deg, sun_el_deg):
    """World-(x, y) components of the tangential (in-plane-of-incidence)
    and sagittal (out-of-plane) unit directions at the mirror -- the same
    construction ``test_mc_physics_fixes._tangential_sagittal_rms`` and
    ``test_pointing_error._mean_tang_sag_deviation`` use in full 3-D, here
    projected onto a flat, world-(x, y)-aligned receiver window."""
    n, u, v, d_in, d_out, s = _reflect(rot_az_deg, rot_el_deg, sun_az_deg, sun_el_deg)
    t3 = n - np.dot(n, d_in) * d_in
    t3 /= np.linalg.norm(t3)
    q3 = np.cross(n, d_in)
    q3 /= np.linalg.norm(q3)
    t_uv = (t3[0] / np.hypot(t3[0], t3[1]), t3[1] / np.hypot(t3[0], t3[1]))
    q_uv = (q3[0] / np.hypot(q3[0], q3[1]), q3[1] / np.hypot(q3[0], q3[1]))
    return t_uv, q_uv


def _strip_aniso(kernel: RadialKernel) -> RadialKernel:
    """A kernel with the identical density table but WITHOUT the
    ``_slope_sigma_rad``/``_non_slope_var_rad2`` attributes ``sunshape_
    kernel`` attaches -- ``trace_heliostat_cone``'s ``getattr(..., 0.0)``
    default then makes it fall back to the OLD, fully isotropic treatment,
    exactly as if this session's fix did not exist. A plain shallow copy
    (not ``RadialKernel(kernel.theta_rad, kernel.density)``, which would
    RE-NORMALISE and perturb the density table at the ~1e-16 level) keeps
    the two kernels' density tables the same array objects, so any
    remaining difference in a trace is attributable ONLY to the anisotropy
    logic, not to a re-normalisation rounding artefact."""
    plain = copy.copy(kernel)
    del plain._slope_sigma_rad
    del plain._non_slope_var_rad2
    return plain


# ---------------------------------------------------------------------------
# 1. bit identity: zero slope error (both backends), normal incidence (cone)


OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL = 0.0, 75.0
OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL = 90.0, 30.0  # 45-deg incidence, well off-normal


class TestBitIdentity:
    def test_zero_slope_error_cone_kernel_unaffected(self):
        base = sunshape_kernel("buie")
        explicit_zero = sunshape_kernel("buie", slope_error_mrad=0.0)
        assert base.rms_radius_rad() == explicit_zero.rms_radius_rad()
        assert base._slope_sigma_rad == 0.0
        assert explicit_zero._slope_sigma_rad == 0.0

    def test_zero_slope_error_cone_trace_bit_identical(self):
        """Omitting ``slope_error_mrad`` and passing it explicitly at 0.0
        must trace bit-identically -- at OBLIQUE incidence, where a bug in
        the new warp-skip gate would most likely show up (the ``warp is
        None`` short-circuit means the anisotropy code touches nothing at
        all when slope error is exactly zero)."""
        secondary = NoSecondary()
        receiver = _receiver_for(OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL)
        args = (
            0.0, 0.0, OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, 0.0, 0.0, 0.0,
            OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL, secondary, receiver,
        )
        baseline = trace_heliostat_cone(*args, sunshape_kernel("buie"), grid=(20, 12), order=2)
        explicit_zero = trace_heliostat_cone(
            *args, sunshape_kernel("buie", slope_error_mrad=0.0), grid=(20, 12), order=2
        )
        assert baseline["power_w"] == explicit_zero["power_w"]
        assert np.array_equal(baseline["flux"], explicit_zero["flux"])
        assert np.array_equal(baseline["jacobians"], explicit_zero["jacobians"])

    def test_zero_slope_error_mc_trace_bit_identical(self):
        secondary = NoSecondary()
        receiver = _receiver_for(OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL)
        args = (
            0.0, 0.0, OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, 0.0, 0.0, 0.0,
            OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL, secondary, receiver, 5000,
        )
        baseline = trace_heliostat(*args, np.random.default_rng(99), return_paths=True)
        explicit_zero = trace_heliostat(
            *args, np.random.default_rng(99), return_paths=True, slope_error_mrad=0.0
        )
        assert np.array_equal(baseline["xy"], explicit_zero["xy"])
        assert np.array_equal(baseline["paths"], explicit_zero["paths"])
        assert baseline["counters"] == explicit_zero["counters"]

    def test_normal_incidence_bit_identical_at_nonzero_slope_error(self):
        """rot_az == sun_az == 45.0 (and rot_el == sun_el) makes the mirror
        normal and the sun vector literally the SAME array, bit for bit
        (``_mirror_frame``'s ``[cos(el)cos(az), cos(el)sin(az), sin(el)]``
        and ``_sun_vector``'s ``[cos(el)cos(pi/2-az), ...]`` coincide
        exactly at az=45 deg, since pi/2-45deg=45deg too) -- so
        ``cos_aoi = |normal . s|`` computes to EXACTLY 1.0, not just
        approximately, which is what this pin actually needs: the new
        ``is_normal = c_eff >= 1.0`` gate in ``trace_heliostat_cone`` forces
        the per-sample warp to the LITERAL identity matrix there (bypassing
        the floating-point ``t_hat``/``q_hat`` reconstruction, which would
        only be identity up to ~1e-16), so this comparison is exact, not
        approximate."""
        rot_az, rot_el = 45.0, 60.0
        sun_az, sun_el = 45.0, 60.0
        n, u, v = _mirror_frame(rot_az, rot_el)
        s = _sun_vector(sun_az, sun_el)
        assert np.array_equal(n, s), "test geometry must give an EXACT normal-incidence dot product"
        assert np.abs(n @ s) == 1.0

        secondary = NoSecondary()
        receiver = _receiver_for(rot_az, rot_el, sun_az, sun_el)
        args = (0.0, 0.0, rot_az, rot_el, 0.0, 0.0, 0.0, sun_az, sun_el, secondary, receiver)

        kernel_full = sunshape_kernel("buie", slope_error_mrad=3.0)
        kernel_plain = _strip_aniso(kernel_full)

        out_full = trace_heliostat_cone(*args, kernel_full, grid=(20, 12), order=2)
        out_plain = trace_heliostat_cone(*args, kernel_plain, grid=(20, 12), order=2)
        assert out_full["power_w"] == out_plain["power_w"]
        assert np.array_equal(out_full["flux"], out_plain["flux"])

    def test_sanity_oblique_incidence_DOES_differ(self):
        """Negative control for the test above: the same full-vs-plain
        kernel comparison, at the OBLIQUE geometry used everywhere else in
        this module, must NOT be bit-identical -- otherwise the normal-
        incidence test above would be vacuous (passing because the
        anisotropy code never does anything, not because the identity gate
        specifically fires at normal incidence)."""
        secondary = NoSecondary()
        receiver = _receiver_for(OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL)
        args = (
            0.0, 0.0, OBLIQUE_ROT_AZ, OBLIQUE_ROT_EL, 0.0, 0.0, 0.0,
            OBLIQUE_SUN_AZ, OBLIQUE_SUN_EL, secondary, receiver,
        )
        kernel_full = sunshape_kernel("buie", slope_error_mrad=3.0)
        kernel_plain = _strip_aniso(kernel_full)
        out_full = trace_heliostat_cone(*args, kernel_full, grid=(20, 12), order=2)
        out_plain = trace_heliostat_cone(*args, kernel_plain, grid=(20, 12), order=2)
        assert not np.array_equal(out_full["flux"], out_plain["flux"])
        _, rms_full = _cone_moments(out_full)
        _, rms_plain = _cone_moments(out_plain)
        # The fix COMPRESSES the sagittal axis relative to the old isotropic
        # treatment, so the anisotropic spot's total RMS must be smaller.
        assert rms_full < rms_plain


# ---------------------------------------------------------------------------
# 2. the cone spot's own tangential/sagittal second moments track cos(theta_i)
# -- via MATCHING MC, not a literal comparison to cos(theta_i) directly.
#
# A direct "cone ellipse ratio == cos(theta_i)" check was tried first, on
# the report's own controlled flat/unfigured-mirror sweep (Part 1's
# geometry), and rejected: it requires quadrature-subtracting a base
# (zero-slope) spot from an error-run spot on the RECEIVER WINDOW, and for
# an unfocused flat mirror the base spot is aperture-image-dominated
# (~1.5 m rms; see the report's own Part 2a) while slope error's own
# addition is a few percent of that -- subtracting two close, large numbers
# on top of the cone deposit's own bin-discretisation noise gives garbage,
# not the true ratio (confirmed: base/err rms 1586/1605 mm at theta=45,
# collapsing the ~250 mm broadening signal into the deposit grid's own
# resolution). Separately, a FIXED "down"-facing horizontal receiver
# foreshortens the tangential and sagittal directions DIFFERENTLY whenever
# the chief ray is not exactly vertical -- a real, direction-dependent
# projection effect unrelated to slope error, so even a perfectly precise
# receiver-position measurement would not reproduce the angle-space
# cos(theta_i) ratio literally (this is why the report's own Part 1 measured
# the reflected RAY DIRECTION directly, in angle space, from ``paths`` --
# sidestepping the receiver-projection question entirely -- rather than a
# receiver-position spot; the cone backend has no equivalent angle-space
# output to read directly, only a receiver-position flux map).
#
# What IS a robust, receiver-projection-independent check: cone and MC see
# the IDENTICAL receiver-projection geometry, so comparing the two DIRECTLY
# (no baseline subtraction, no comparison to an idealised angle-space
# formula) cancels the projection confound out. MC's own Part 1 measurement
# already established, independently of this fix, that MC's ray-direction
# ellipse tracks cos(theta_i) to ~0.3% up to 60 deg; cone matching MC here
# therefore transitively confirms cone reproduces the SAME (cos(theta_i))
# ellipse, at whatever precision this comparison holds.


class TestEllipseRatio:
    """Real, focused manuscript-field heliostats (the same two cases
    ``TestFocusedFieldDisagreementCloses`` uses one of), matched Buie
    sampler, direct (non-subtracted) tangential/sagittal second moments of
    the TOTAL spot -- both cone_after and MC see the same figure-driven
    baseline astigmatism AND the same receiver-projection foreshortening,
    so a close match here isolates the slope-anisotropy fix's own
    correctness rather than either confound."""

    @pytest.mark.parametrize(
        "x_mm,y_mm,aoi_deg,label,seed_key",
        [
            (-26303.242, 62521.30716, 17.153580445914926, "p10_id328", 1),
            (49799.08303, -53219.08275, 44.892565952957824, "p90_id367", 2),
        ],
    )
    def test_cone_tang_sag_matches_mc(self, x_mm, y_mm, aoi_deg, label, seed_key):
        params = resolve_optics_params("prime_focus", None)
        secondary, _ = _geometry_for("prime_focus", params)
        sol = _solve_for("prime_focus", x_mm, y_mm, FIELD_SUN_AZ, FIELD_SUN_EL, params)
        assert sol.aoi_deg == pytest.approx(aoi_deg, abs=0.1)
        receiver = FlatWindowReceiver(
            z_mm=FOCUS_HEIGHT_MM, half_u_mm=GENEROUS_HALF_MM, half_v_mm=GENEROUS_HALF_MM,
            facing="down",
        )
        trace_args = (
            x_mm, y_mm, sol.rot_az_deg, sol.rot_el_deg, sol.c3, sol.c4, sol.c5,
            FIELD_SUN_AZ, FIELD_SUN_EL, secondary, receiver,
        )
        t_uv, q_uv = _tq_hat_uv(sol.rot_az_deg, sol.rot_el_deg, FIELD_SUN_AZ, FIELD_SUN_EL)

        out_err = trace_heliostat(
            *trace_args, FIELD_N_RAYS, np.random.default_rng((6101, seed_key)),
            slope_error_mrad=FIELD_SLOPE_MRAD, sampler=BuieSampler(), return_paths=False,
        )
        xy = out_err["xy"]
        cx, cy = float(np.mean(xy[0])), float(np.mean(xy[1]))
        dx, dy = xy[0] - cx, xy[1] - cy
        mc_t = float(np.sqrt(np.mean((dx * t_uv[0] + dy * t_uv[1]) ** 2)))
        mc_s = float(np.sqrt(np.mean((dx * q_uv[0] + dy * q_uv[1]) ** 2)))

        kernel_after = sunshape_kernel("buie", slope_error_mrad=FIELD_SLOPE_MRAD)
        cone_after = trace_heliostat_cone(*trace_args, kernel_after, grid=(40, 24), order=2)
        cone_t, cone_s = _cone_tang_sag_rms_mm(cone_after, t_uv, q_uv)

        mc_ratio = mc_s / mc_t
        cone_ratio = cone_s / cone_t
        assert cone_ratio == pytest.approx(mc_ratio, rel=0.08), (
            f"{label}: cone sag/tang={cone_ratio:.4f} (t={cone_t:.1f}mm s={cone_s:.1f}mm) vs "
            f"MC sag/tang={mc_ratio:.4f} (t={mc_t:.1f}mm s={mc_s:.1f}mm), cos(aoi)={np.cos(np.radians(aoi_deg)):.4f}"
        )
        # Also directly against each backend's own total RMS, so a bug that
        # scales both axes together (passing this test by ratio alone) does
        # not slip through.
        assert cone_t == pytest.approx(mc_t, rel=0.05)
        assert cone_s == pytest.approx(mc_s, rel=0.08)


# ---------------------------------------------------------------------------
# 3. cone-vs-MC total-spot-RMS disagreement closes on real, focused,
# manuscript-field heliostats


# Heliostat 367 (field id) at the app's default sun position (az=165.2,
# el=61.4) -- the report's own field-90th-percentile-incidence pick
# (aoi_i=44.89 deg), reproduced here as fixed x/y (not re-derived from the
# field CSV's percentiles every test run, matching how test_pointing_error.py/
# test_cone_convergence.py hardcode a specific fixture heliostat rather than
# re-deriving it): see scratchpad/gate2_verify_results.json (this session)
# for the full before/after table across p10/p50/p90 x 1/2/3 mrad this one
# case is drawn from.
FIELD_SUN_AZ, FIELD_SUN_EL = 165.2, 61.4
P90_X_MM, P90_Y_MM = 49799.08303, -53219.08275
P90_AOI_DEG = 44.892565952957824
FOCUS_HEIGHT_MM = 35335.0
GENEROUS_HALF_MM = 8000.0
FIELD_N_RAYS = 150_000
FIELD_SLOPE_MRAD = 3.0


class TestFocusedFieldDisagreementCloses:
    def test_worst_case_p90_incidence_disagreement_closes(self):
        params = resolve_optics_params("prime_focus", None)
        secondary, _ = _geometry_for("prime_focus", params)
        sol = _solve_for("prime_focus", P90_X_MM, P90_Y_MM, FIELD_SUN_AZ, FIELD_SUN_EL, params)
        assert sol.aoi_deg == pytest.approx(P90_AOI_DEG, abs=0.1)
        receiver = FlatWindowReceiver(
            z_mm=FOCUS_HEIGHT_MM, half_u_mm=GENEROUS_HALF_MM, half_v_mm=GENEROUS_HALF_MM,
            facing="down",
        )
        trace_args = (
            P90_X_MM, P90_Y_MM, sol.rot_az_deg, sol.rot_el_deg, sol.c3, sol.c4, sol.c5,
            FIELD_SUN_AZ, FIELD_SUN_EL, secondary, receiver,
        )

        out_err = trace_heliostat(
            *trace_args, FIELD_N_RAYS, np.random.default_rng((6001, 3)),
            slope_error_mrad=FIELD_SLOPE_MRAD, sampler=BuieSampler(),
        )
        mc_rms = _rms_about_own_centroid(out_err["xy"])
        n_landed = out_err["xy"].shape[1]
        # se(rms) ~= rms / sqrt(2N) (test_cone_vs_mc.py's own noise-gate
        # formula); a generous 6*se band, floored, comfortably covers shot
        # noise at this ray count while still being far tighter than the
        # ~5% disagreement the pre-fix isotropic treatment showed here.
        se_rms = mc_rms / np.sqrt(2.0 * n_landed)
        noise_band_pct = max(6.0 * se_rms / mc_rms * 100.0, 1.0)

        kernel_after = sunshape_kernel("buie", slope_error_mrad=FIELD_SLOPE_MRAD)
        kernel_before = _strip_aniso(kernel_after)
        cone_after = trace_heliostat_cone(*trace_args, kernel_after, grid=(40, 24), order=2)
        cone_before = trace_heliostat_cone(*trace_args, kernel_before, grid=(40, 24), order=2)
        _, rms_after = _cone_moments(cone_after)
        _, rms_before = _cone_moments(cone_before)

        d_before_pct = (rms_before - mc_rms) / mc_rms * 100.0
        d_after_pct = (rms_after - mc_rms) / mc_rms * 100.0

        # The pre-fix (isotropic) disagreement at this case is a real,
        # sizeable, one-signed effect (report/gate2_verify.py: ~5%) -- assert
        # it is NOT already small, so a future change that quietly weakens
        # this fixture's own signal (e.g. shrinking the field pick's aoi)
        # cannot slip the "closes" assertion below by starting from nothing.
        assert d_before_pct > 3.0, (
            f"pre-fix disagreement {d_before_pct:+.3f}% is unexpectedly small for this fixture "
            "-- the case may no longer exercise the anisotropy gap"
        )
        assert abs(d_after_pct) < noise_band_pct, (
            f"post-fix disagreement {d_after_pct:+.3f}% exceeds the {noise_band_pct:.2f}% MC noise "
            f"band (mc_rms={mc_rms:.2f}mm, se={se_rms:.2f}mm, before={d_before_pct:+.3f}%)"
        )
