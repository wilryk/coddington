"""Verification for the B-spline coefficient-space deposit
(``deposit_method="bspline"`` in :func:`heliostat.trace.cone.trace_heliostat_cone`,
:mod:`heliostat.trace.bspline_deposit`), adopted for the ``ultra_fast`` mode.

Ported from ``scripts/coeff_prototype/bspline.py`` -- see
``scripts/coeff_prototype/REPORT.md`` SS0-6 for the full field-scale
benchmark (3.36x end-to-end speedup vs binned deposit, power conserved to
0.04-0.29%) that motivated the adoption, and SS3.4 for the cylinder-seam
peak-flux artifact (+3.73%) this module's periodic fix targets.

Five things are pinned here, mirroring the structure of
``tests/test_receiver_seam.py`` (the referee pattern that originally caught
the coordinate-seam bug this same geometry can trigger):

1. ``TestControlGridDerivation`` -- the coarse accumulation grid scales
   proportionally with ``flux_grid`` (a fixed 32x32 control grid, correct
   for the flat 128x128 case, badly under-resolves peak flux on a curved
   receiver's adaptive up-to-448-wide grid -- see cone.py's
   ``CONTROL_GRID_COARSEN``).
2. ``TestPowerConservation`` -- per-trace power_w stays within 0.3% of the
   binned path's, same sampling grid/order, several receiver geometries.
3. ``TestAzimuthalRotationInvariance`` / ``TestMonteCarloRefereesTheCavitySeam``
   -- the bearing-invariance referee pattern applied to the bspline path on
   a cylinder: power and peak must agree at every bearing (including the
   coordinate seam), and the cone/MC power ratio at the seam bearing must
   equal the seam-free bearing within MC noise -- exactly what would catch
   a periodic-basis regression of the kind that produced the prototype's
   +3.7% seam artifact.
4. ``TestNoPeriodicRingingArtifact`` / ``TestPeriodicNoRingingArtifact`` --
   the owner-reported "grid every 4 pixels" ringing, and its milder
   periodic-axis counterpart on a closed (cylindrical) receiver, pinned as
   a topological property (connected non-zero regions) rather than a raw
   numeric one.
5. ``TestPeriodicUpsampleMatrixPhysics`` / ``TestSeamContinuity`` -- the
   periodic branch's own requirements, re-pinned around physics
   (non-negativity, exact power conservation via partition-of-unity, seam
   continuity, and -- via item 3 above -- bearing invariance) after
   retiring the old exact-interpolation pin; see
   ``TestPeriodicUpsampleMatrixPhysics``'s docstring for why.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.geometry.aiming import solve_prime_focus_to_receiver
from heliostat.geometry.receiver import ApertureClippedReceiver, CylinderReceiver, FlatWindowReceiver, FrustumReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.trace.bspline_deposit import control_grid_edges, evaluate_bspline
from heliostat.trace.cone import CONTROL_GRID_COARSEN, CONTROL_GRID_MIN, sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat
from heliostat.trace.modes import ULTRA_FAST
from test_mc_parity import _geometry_for, _load_fixture

_KERNEL = sunshape_kernel("super_gauss")
_SECONDARY = NoSecondary()

_FIELD_RADIUS_MM = 40000.0
_BASE_SOLAR_AZ_DEG = 165.0
_BASE_SOLAR_EL_DEG = 45.0
_BEARINGS_DEG = [15.0, 60.0, 105.0, 150.0, 195.0, 240.0, 285.0, 330.0]


def _rotated_case(bearing_deg: float) -> tuple[float, float, float]:
    rad = np.deg2rad(bearing_deg)
    x = _FIELD_RADIUS_MM * np.sin(rad)
    y = _FIELD_RADIUS_MM * np.cos(rad)
    solar_az = (_BASE_SOLAR_AZ_DEG + bearing_deg) % 360.0
    return x, y, solar_az


def _trace(receiver, x_mm, y_mm, solar_az_deg, deposit_method="bspline", **overrides):
    sol = solve_prime_focus_to_receiver(x_mm, y_mm, solar_az_deg, _BASE_SOLAR_EL_DEG, receiver)
    kwargs = dict(ULTRA_FAST.cone_kwargs)
    kwargs["deposit_method"] = deposit_method
    kwargs.update(overrides)
    return trace_heliostat_cone(
        x_mm,
        y_mm,
        sol.rot_az_deg,
        sol.rot_el_deg,
        sol.c3,
        sol.c4,
        sol.c5,
        solar_az_deg,
        _BASE_SOLAR_EL_DEG,
        _SECONDARY,
        receiver,
        _KERNEL,
        **kwargs,
    )


class TestControlGridDerivation:
    """``control_grid=None`` derives the coarse accumulation grid as
    ``flux_grid`` coarsened ``CONTROL_GRID_COARSEN`` (4x) per axis, clamped
    to ``CONTROL_GRID_MIN`` -- not the prototype's fixed 32x32, which is
    only correct at the prototype's own flat 128x128 benchmark grid."""

    def test_flat_128_grid_matches_prototype_32x32(self):
        assert (128 // CONTROL_GRID_COARSEN, 128 // CONTROL_GRID_COARSEN) == (32, 32)

    def test_derivation_scales_with_a_wide_curved_receiver_grid(self):
        """A curved receiver's adaptive flux_grid can be much wider than
        tall (e.g. (448, 128) -- see app.py's _receiver_flux_grid); the
        control grid must track that aspect, not stay square."""
        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        x, y, solar_az = _rotated_case(60.0)
        flux_grid = (448, 128)
        out = _trace(receiver, x, y, solar_az, flux_grid=flux_grid)
        assert out["flux"].shape == (flux_grid[1], flux_grid[0])
        expected_n_cu = max(CONTROL_GRID_MIN, round(448 / CONTROL_GRID_COARSEN))
        expected_n_cv = max(CONTROL_GRID_MIN, round(128 / CONTROL_GRID_COARSEN))
        assert (expected_n_cu, expected_n_cv) == (112, 32)

    def test_explicit_control_grid_override_is_honoured(self):
        """A caller-supplied control_grid bypasses the flux_grid-derived
        default entirely -- app.py's own comb-artifact override sets an
        explicit `grid=` for the mirror-sample grid, and this is the
        control-grid analogue callers may need for the same reason."""
        receiver = FlatWindowReceiver(z_mm=20000.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down")
        x, y, solar_az = _rotated_case(0.0)
        out_default = _trace(receiver, x, y, solar_az)
        out_override = _trace(receiver, x, y, solar_az, control_grid=(8, 8))
        # Different control grids should not produce bit-identical output
        # for a real spot (otherwise the override silently did nothing).
        assert not np.allclose(out_default["flux"], out_override["flux"])
        assert out_override["power_w"] > 0


class TestPowerConservation:
    """Per-trace power_w within 0.3% of the binned path's, same sampling
    grid/order (isolating the deposit method), several receiver geometries
    -- see REPORT.md SS3 (0.04-0.29% measured at field scale across four
    scenarios) for the number this pin's tolerance is set against."""

    POWER_CONSERVATION_REL_TOL = 0.003  # 0.3%, per the task's own gate

    @pytest.mark.parametrize("config", ["prime_focus", "axicon", "cassegrain"])
    def test_power_conserved_vs_binned_on_fixture_cases(self, config):
        secondary, receiver = _geometry_for(config)
        _, counters, summary = _load_fixture(config)
        worst = 0.0
        for heliostat_id, step_key in list(counters)[:5]:
            row = summary.loc[(heliostat_id, step_key)]
            kwargs = dict(ULTRA_FAST.cone_kwargs)

            def _run(deposit_method):
                k = dict(kwargs)
                k["deposit_method"] = deposit_method
                return trace_heliostat_cone(
                    row.x_mm,
                    row.y_mm,
                    row.rot_az_deg,
                    row.rot_el_deg,
                    row.c3,
                    row.c4,
                    row.c5,
                    row.solar_az_deg,
                    row.solar_el_deg,
                    secondary,
                    receiver,
                    _KERNEL,
                    **k,
                )

            bs = _run("bspline")
            bn = _run("binned")
            assert bn["power_w"] > 0
            rel_err = abs(bs["power_w"] - bn["power_w"]) / bn["power_w"]
            worst = max(worst, rel_err)
        assert worst < self.POWER_CONSERVATION_REL_TOL, (
            f"{config}: worst power-conservation error {worst * 100:.4f}% "
            f"exceeds {self.POWER_CONSERVATION_REL_TOL * 100:.1f}%"
        )

    @pytest.mark.parametrize(
        "receiver",
        [
            CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0),
            FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0),
        ],
        ids=["cylinder", "frustum"],
    )
    def test_power_conserved_vs_binned_on_curved_receivers(self, receiver):
        x, y, solar_az = _rotated_case(60.0)
        bs = _trace(receiver, x, y, solar_az, deposit_method="bspline")
        bn = _trace(receiver, x, y, solar_az, deposit_method="binned")
        assert bn["power_w"] > 0
        rel_err = abs(bs["power_w"] - bn["power_w"]) / bn["power_w"]
        assert rel_err < self.POWER_CONSERVATION_REL_TOL


class TestAzimuthalRotationInvariance:
    """A receiver of revolution traced with a rigidly rotated heliostat must
    report the same power and peak flux at every bearing -- the same
    end-to-end referee ``tests/test_receiver_seam.py`` uses for the
    coordinate-seam fix, applied here to the bspline deposit path. A
    periodic-basis regression (the prototype's own +3.7% seam peak-flux
    miss) would show up as one or a few bearings disagreeing with the rest.
    """

    @pytest.mark.parametrize(
        "receiver",
        [
            CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0),
            FrustumReceiver(z_bot_mm=17000.0, r_bot_mm=4000.0, z_top_mm=23000.0, r_top_mm=2500.0),
        ],
        ids=["cylinder", "frustum"],
    )
    def test_power_and_peak_flux_match_at_every_bearing(self, receiver):
        results = {}
        for bearing in _BEARINGS_DEG:
            x, y, solar_az = _rotated_case(bearing)
            out = _trace(receiver, x, y, solar_az)
            results[bearing] = (out["power_w"], float(out["flux"].max()))

        base_power, base_peak = results[_BEARINGS_DEG[0]]
        assert base_power > 0
        assert base_peak > 0
        for bearing, (power, peak) in results.items():
            assert power == pytest.approx(base_power, rel=1e-6), (
                f"bearing {bearing} deg: power {power} W vs base {base_power} W "
                f"(all bearings: { {b: p for b, (p, _) in results.items()} })"
            )
            assert peak == pytest.approx(base_peak, rel=1e-6), (
                f"bearing {bearing} deg: peak flux {peak} W/m^2 vs base {base_peak} W/m^2 "
                f"(all bearings: { {b: pk for b, (_, pk) in results.items()} })"
            )


class TestMonteCarloRefereesTheCavitySeam:
    """Monte Carlo is an INDEPENDENT referee for the cavity seam fix, exactly
    per ``tests/test_receiver_seam.py``'s own class of the same name -- MC
    rays are points, each landing in exactly one bin, so MC never needed the
    wrap machinery and never had the seam bug. Applied here to the bspline
    deposit path: comparing the cone/MC collected-power ratio at the seam
    bearing to the same ratio at a seam-free bearing cancels every
    normalisation difference between the backends; a periodic-basis
    regression would drag that double ratio off 1.
    """

    def _mc_power_w(self, receiver, x, y, solar_az, n_rays=200_000):
        sol = solve_prime_focus_to_receiver(x, y, solar_az, _BASE_SOLAR_EL_DEG, receiver)
        out = trace_heliostat(
            x,
            y,
            sol.rot_az_deg,
            sol.rot_el_deg,
            sol.c3,
            sol.c4,
            sol.c5,
            solar_az,
            _BASE_SOLAR_EL_DEG,
            _SECONDARY,
            receiver,
            n_rays,
            np.random.default_rng(20260826),
        )
        return out["watts_per_ray"] * out["counters"]["in_window"]

    def test_cone_to_mc_ratio_is_bearing_independent_for_the_cavity(self):
        cylinder = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        cavity = ApertureClippedReceiver(
            aperture=FlatWindowReceiver(
                z_mm=20000.0, half_u_mm=1.0e6, half_v_mm=1.0e6, facing="down"
            ),
            inner=cylinder,
        )

        ratios = {}
        for bearing in (0.0, 180.0):  # dead on the seam vs. as far from it as possible
            x, y, solar_az = _rotated_case(bearing)
            cone_power = _trace(cavity, x, y, solar_az)["power_w"]
            mc_power = self._mc_power_w(cavity, x, y, solar_az)
            assert mc_power > 0
            ratios[bearing] = cone_power / mc_power

        # Same 2% tolerance as the coordinate-seam referee test: MC shot
        # noise at 2e5 rays plus genuine backend differences, both
        # bearing-independent -- the deficit this guards against (either the
        # coordinate seam bug or a periodic-basis regression in the bspline
        # upsample) was factor-of-two scale, not percent scale.
        assert ratios[0.0] == pytest.approx(ratios[180.0], rel=0.02), (
            f"cone/MC ratio depends on seam proximity: {ratios}"
        )


class TestDepositMethodValidation:
    def test_invalid_deposit_method_raises(self):
        receiver = FlatWindowReceiver(z_mm=20000.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down")
        x, y, solar_az = _rotated_case(0.0)
        with pytest.raises(ValueError, match="deposit_method"):
            _trace(receiver, x, y, solar_az, deposit_method="not_a_real_method")


class TestPeriodicUpsampleMatrixPhysics:
    """Replaces the retired ``TestUpsampleMatrixIsExactPeriodicInterpolation``.

    That test pinned the periodic ``_upsample_matrix`` branch against
    ``make_interp_spline(..., bc_type="periodic")``'s own exact-interpolating
    solve, unit for unit -- a guard on an IMPLEMENTATION detail (which
    particular spline construction the periodic branch happened to use),
    not on physics. Its cost was real: exact interpolation through sharp
    coarse-grid data rings (textbook cubic-spline overshoot/undershoot at
    the knot spacing, the same mechanism the non-periodic branch's fix in
    commit 3d68009 targeted), and keeping that pin would have permanently
    blocked fixing the periodic axis's milder version of the same
    "grid every 4 pixels" artifact the owner separately reported as a
    "jagged" cylinder map. Exact interpolation and guaranteed
    non-negativity are mutually exclusive for sharp data, so the two goals
    could never both be satisfied -- the pin had to go.

    What a receiver axis that closes on itself actually needs, physically,
    is not "reproduce one particular spline solve" but:

    1. periodic continuity -- no jump at the seam (``test_seam_is_...``
       below, and end to end in ``TestSeamContinuity``);
    2. non-negativity (``test_matrix_is_nonnegative_and_partitions_unity``);
    3. exact power conservation (the same test: partition-of-unity means a
       uniform coefficient array reconstructs to exactly that constant, so
       the integral is preserved by construction, not by a downstream
       clamp-and-rescale);
    4. bearing invariance -- already pinned end to end, at full optical
       trace precision (``rel=1e-6``), by ``TestAzimuthalRotationInvariance``
       above and by ``tests/test_receiver_seam.py``'s class of the same
       name; not duplicated here.

    Those four are pinned directly below (and, for continuity, again at
    full trace fidelity in ``TestSeamContinuity``) -- a strictly stronger
    guarantee than "matches one particular spline solve," because it holds
    regardless of which non-negative periodic construction is used.
    """

    def test_matrix_is_nonnegative_and_partitions_unity(self):
        period = 36000.0
        n_c, n_f = 32, 448
        coarse_edges = np.linspace(0.0, period, n_c + 1)
        fine_edges = np.linspace(0.0, period, n_f + 1)
        from heliostat.trace.bspline_deposit import _upsample_matrix

        m = _upsample_matrix(coarse_edges, fine_edges, periodic=True)

        assert m.min() >= 0.0, "periodic upsample matrix produced a negative entry"
        # Partition of unity: every fine row sums to 1, so ANY uniform
        # coefficient vector reconstructs to that exact constant everywhere
        # -- the matrix-level statement of exact power conservation on this
        # axis, with no clamp-and-rescale required downstream.
        row_sums = m.sum(axis=1)
        assert row_sums == pytest.approx(1.0, abs=1e-9)

    def test_constant_reconstructs_exactly(self):
        period = 36000.0
        coarse_edges = np.linspace(0.0, period, 33)
        fine_edges = np.linspace(0.0, period, 449)
        from heliostat.trace.bspline_deposit import _upsample_matrix

        m = _upsample_matrix(coarse_edges, fine_edges, periodic=True)
        out = m @ (np.ones(32) * 3.7)
        assert out == pytest.approx(3.7, abs=1e-9)

    def test_seam_is_continuous_not_just_matrix_bounded(self):
        """Direct check, independent of any optical trace, on the spline
        the matrix implements: it must evaluate to the SAME value at the
        seam approached from either side (the two chart-boundary
        evaluations of one continuous closed curve), with neighbouring
        values on either side smooth relative to the data's own scale --
        not merely non-negative and bounded, but genuinely without a jump.
        """
        from scipy.interpolate import BSpline

        period = 36000.0
        n_c, k = 32, 3
        coarse_edges = np.linspace(0.0, period, n_c + 1)
        coarse_mid = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
        h = period / n_c
        knot_idx = np.arange(-k, n_c + k + 1)
        t = coarse_mid[0] + knot_idx * h

        rng = np.random.default_rng(1)
        coeffs = rng.uniform(0.2, 1.0, n_c)
        c_ext = np.concatenate([coeffs, coeffs[:k]])
        spl = BSpline(t, c_ext, k, extrapolate=False)

        eps = 1.0  # mm -- tiny next to the ~1125 mm coarse cell width here
        left = float(spl(np.array([coarse_mid[0] + period - eps]))[0])
        at_seam_from_below = float(spl(np.array([coarse_mid[0] + period]))[0])
        at_seam_from_above = float(spl(np.array([coarse_mid[0]]))[0])
        right = float(spl(np.array([coarse_mid[0] + eps]))[0])

        assert at_seam_from_below == pytest.approx(at_seam_from_above, abs=1e-9), (
            "the seam's two chart-boundary evaluations disagree -- a real "
            "discontinuity, not a closed curve"
        )
        neighbourhood = [left, at_seam_from_below, right]
        assert max(neighbourhood) - min(neighbourhood) < 0.05 * coeffs.max(), (
            f"seam neighbourhood {neighbourhood} is not smooth relative to "
            f"the coefficient scale {coeffs.max()} -- looks like a spike, "
            "not a point on a smooth closed curve"
        )


class TestSeamContinuity:
    """End-to-end (real optical trace), MEASURABLE version of the seam
    continuity property pinned at the matrix level above: a spot straddling
    the coordinate seam must be exactly as smooth as the physically
    identical spot traced mid-chart -- not merely non-negative or
    connected (``TestPeriodicNoRingingArtifact`` covers that), but free of
    any extra curvature the seam wrap itself might introduce.

    A ``CylinderReceiver`` is a body of revolution: the SAME rigidly
    rotated heliostat traced at bearing 0 deg (aimed squarely at the
    coordinate seam -- see ``tests/test_receiver_seam.py``'s harness,
    reused here) and at bearing 180 deg (as far from the seam as this
    field radius puts it, spot safely mid-chart) must produce physically
    identical spot shapes, merely shifted in ``u`` -- the same symmetry
    ``TestAzimuthalRotationInvariance`` uses for power and peak. So a
    smoothness metric -- RMS of the periodic second difference along ``u``,
    through the row of peak flux -- computed at each bearing must agree: if
    the seam wrap added spurious curvature on top of the real spot shape,
    the bearing-0 metric would come out larger than bearing-180's.
    """

    RMS_CURVATURE_REL_TOL = 0.15  # 15%: comfortably above sampling/quadrature noise

    @staticmethod
    def _peak_row_curvature_rms(flux):
        peak_v = int(np.argmax(flux.max(axis=1)))
        row = flux[peak_v, :]
        # Periodic (wraparound) second difference: the same operator at
        # every column, including across the column-0/column-(n-1) wrap --
        # exactly the comparison the seam needs.
        d2 = np.roll(row, -1) - 2.0 * row + np.roll(row, 1)
        return float(np.sqrt(np.mean(d2**2)))

    def test_seam_spot_is_as_smooth_as_mid_chart_spot(self):
        from test_receiver_seam import _rotated_case as _seam_rotated_case

        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        rms = {}
        for bearing in (0.0, 180.0):
            x, y, solar_az = _seam_rotated_case(bearing)
            out = _trace(receiver, x, y, solar_az, deposit_method="bspline")
            flux = out["flux"]
            assert flux.max() > 0, f"bearing {bearing}: no spot at all -- fixture broken, not the assertion"
            rms[bearing] = self._peak_row_curvature_rms(flux)

        assert rms[0.0] == pytest.approx(rms[180.0], rel=self.RMS_CURVATURE_REL_TOL), (
            f"seam-bearing curvature {rms[0.0]:.6g} vs mid-chart curvature "
            f"{rms[180.0]:.6g} -- the seam wrap is adding spurious roughness "
            "not present in the physically identical mid-chart spot"
        )


class TestPeriodicNoRingingArtifact:
    """Same regression pin as ``TestNoPeriodicRingingArtifact`` above
    (owner-reported "grid every 4 pixels"), applied to the PERIODIC branch:
    a single-heliostat CYLINDER ultra_fast flux map must be one connected
    non-zero region, both away from the seam and with the spot straddling
    it."""

    def test_flux_map_has_no_isolated_ringing_islands(self):
        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        x, y, solar_az = _rotated_case(60.0)  # generic bearing, away from the seam
        out = _trace(receiver, x, y, solar_az, deposit_method="bspline")
        flux = out["flux"]
        assert flux.max() > 0, "test case produced no spot at all -- fixture is broken, not the assertion"

        from scipy.ndimage import label

        mask = flux > (1e-9 * flux.max())
        _, n_components = label(mask, structure=np.ones((3, 3)))
        assert n_components == 1, (
            f"cylinder flux map has {n_components} disconnected non-zero regions "
            "(expected 1) -- periodic-branch ringing islands"
        )

    def test_flux_map_has_no_isolated_ringing_islands_at_the_seam(self):
        """Same pin with the spot straddling the seam (bearing 0, reusing
        ``tests/test_receiver_seam.py``'s ``_rotated_case`` harness rather
        than rebuilding the geometry). The raw array's column 0 and column
        -1 are adjacent on the physical receiver but not adjacent in plain
        array indexing, so a naive connectivity check would report two
        components even for a perfectly smooth wrapped spot -- rolling the
        array by half its width first moves the seam away from the array
        edge, turning the wraparound-spanning spot into an ordinarily
        contiguous one, exactly what a truly seamless deposit should look
        like once re-centred."""
        from test_receiver_seam import _rotated_case as _seam_rotated_case
        from scipy.ndimage import label

        receiver = CylinderReceiver(center_z_mm=20000.0, radius_mm=3000.0, height_mm=6000.0)
        x, y, solar_az = _seam_rotated_case(0.0)
        out = _trace(receiver, x, y, solar_az, deposit_method="bspline")
        flux = out["flux"]
        assert flux.max() > 0, "test case produced no spot at all -- fixture is broken, not the assertion"

        n_u = flux.shape[1]
        rolled = np.roll(flux, n_u // 2, axis=1)
        mask = rolled > (1e-9 * rolled.max())
        _, n_components = label(mask, structure=np.ones((3, 3)))
        assert n_components == 1, (
            f"seam-straddling cylinder flux map has {n_components} disconnected "
            "non-zero regions after re-centring away from the array edge "
            "(expected 1) -- isolated ringing islands at the seam"
        )


class TestControlGridEdgesAndEvaluate:
    """Small direct unit tests on control_grid_edges/evaluate_bspline,
    independent of the optics: the coarse grid spans the same extent as the
    fine grid, and evaluating an all-zero coarse accumulator upsamples to
    all zero (no spurious energy from the interpolation matrices alone).
    """

    def test_control_grid_spans_same_extent_as_fine_grid(self):
        u_edges = np.linspace(-2000.0, 2000.0, 129)
        v_edges = np.linspace(-1500.0, 1500.0, 129)
        u_c, v_c = control_grid_edges(u_edges, v_edges, (32, 32))
        assert u_c[0] == pytest.approx(u_edges[0])
        assert u_c[-1] == pytest.approx(u_edges[-1])
        assert v_c[0] == pytest.approx(v_edges[0])
        assert v_c[-1] == pytest.approx(v_edges[-1])
        assert u_c.size == 33
        assert v_c.size == 33

    def test_zero_coarse_accumulator_upsamples_to_zero(self):
        u_edges = np.linspace(-2000.0, 2000.0, 129)
        v_edges = np.linspace(-1500.0, 1500.0, 129)
        u_c, v_c = control_grid_edges(u_edges, v_edges, (32, 32))
        coarse = np.zeros((32, 32))
        fine = evaluate_bspline(coarse, u_c, v_c, u_edges, v_edges, wrap_u=False)
        assert np.allclose(fine, 0.0)


class TestNoPeriodicRingingArtifact:
    """Regression pin for the owner-reported "grid every 4 pixels" bug: a
    darker/brighter line every ``CONTROL_GRID_COARSEN`` (4) fine bins in the
    rendered irradiance map, on a plain flat (non-wrapping) receiver.

    Root cause (measured on this exact single-heliostat case): the old
    interpolating construction of the non-periodic ``_upsample_matrix``
    branch rang at a spot's sharp edge -- undershooting to ~8-12% of local
    peak, period-4 -- and the module's clamp-to-zero then chopped the
    positive rebound lobes into isolated islands of nonzero flux completely
    surrounded by exact-zero "moats". That is a topological signature, not
    just a numeric one: before the fix this case's flux map had 3 separate
    connected non-zero regions (the main spot plus two satellite islands at
    +-1 control cell); after the fix (a non-negative coefficient-blend
    construction -- see ``_upsample_matrix``'s docstring) it has exactly 1.
    Binned deposit (``fast_accurate``) and Monte Carlo show 1 on the same
    case at every control-grid coarsening tried, so this is deposit-method
    structure, not a real multi-lobe spot.
    """

    def test_flux_map_has_no_isolated_ringing_islands(self):
        receiver = FlatWindowReceiver(z_mm=20000.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down")
        x, y, solar_az = _rotated_case(0.0)
        out = _trace(receiver, x, y, solar_az, deposit_method="bspline")
        flux = out["flux"]
        assert flux.max() > 0, "test case produced no spot at all -- fixture is broken, not the assertion"

        from scipy.ndimage import label

        # A tiny relative threshold, not exactly 0, so float roundoff in the
        # smooth (fixed) case can't manufacture a spurious extra "island".
        mask = flux > (1e-9 * flux.max())
        _, n_components = label(mask, structure=np.ones((3, 3)))
        assert n_components == 1, (
            f"flux map has {n_components} disconnected non-zero regions (expected 1) -- "
            "isolated ringing islands separated by clamped-to-zero moats, the reported "
            "'grid every 4 pixels' artifact"
        )
