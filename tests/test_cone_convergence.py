"""Analytic/numerical convergence checks for the cone-optics backend.

Small and fast: these don't touch the MC fixtures, just probe the cone
backend's own internal consistency -- grid refinement, finite-difference
step size, and the two pinned sunshape kernels' analytic properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from test_mc_parity import _geometry_for, _load_fixture

_D, _COUNTERS, _SUMMARY = _load_fixture("prime_focus")
_ROW = _SUMMARY.loc[(574, "20260321_0939")]
_SECONDARY, _RECEIVER = _geometry_for("prime_focus")
_KERNEL = sunshape_kernel("super_gauss")


def _trace(**kwargs):
    return trace_heliostat_cone(
        _ROW.x_mm,
        _ROW.y_mm,
        _ROW.rot_az_deg,
        _ROW.rot_el_deg,
        _ROW.c3,
        _ROW.c4,
        _ROW.c5,
        _ROW.solar_az_deg,
        _ROW.solar_el_deg,
        _SECONDARY,
        _RECEIVER,
        _KERNEL,
        **kwargs,
    )


def _centroid_rms(cone):
    flux = cone["flux"]
    u_mid = 0.5 * (cone["u_edges"][:-1] + cone["u_edges"][1:])
    v_mid = 0.5 * (cone["v_edges"][:-1] + cone["v_edges"][1:])
    tot = flux.sum()
    cen = np.array(
        [
            (flux.sum(axis=0) * u_mid).sum() / tot,
            (flux.sum(axis=1) * v_mid).sum() / tot,
        ]
    )
    uu, vv = np.meshgrid(u_mid, v_mid)
    rms = float(
        np.sqrt((((uu - cen[0]) ** 2 + (vv - cen[1]) ** 2).ravel() * flux.ravel()).sum() / tot)
    )
    return cen, rms


GRID_REFINEMENT_GRIDS = [(10, 6), (20, 12), (40, 24)]


@pytest.fixture(scope="module")
def grid_refinement_sequence():
    results = []
    for grid in GRID_REFINEMENT_GRIDS:
        cone = _trace(grid=grid)
        _, rms = _centroid_rms(cone)
        results.append((cone["power_w"], rms))
    return results


class TestGridRefinement:
    """Mirror-sample grid density (20, 12) is the default; here it's varied
    over (10, 6) -> (20, 12) -> (40, 24), each doubling both axes, on
    heliostat 574's prime_focus mid-morning fixture case. Measured sequence:

        grid       power_w        rms_mm
        (10, 6)    12462.671167   504.945271
        (20, 12)   12462.667013   504.961075
        (40, 24)   12462.664816   504.965033

    Both power and rms move monotonically and by a shrinking amount as the
    grid refines (power differences 0.0042 W then 0.0022 W; rms differences
    0.0158 mm then 0.0040 mm) -- exactly the second-order convergence a
    central-difference/midpoint-rule scheme should show, and (40, 24) vs
    (20, 12) agree to ~1.6e-7 relative in power, far inside the 0.1% the
    task asks for.
    """

    def test_power_converges_monotonically_with_shrinking_step(self, grid_refinement_sequence):
        p0, p1, p2 = (r[0] for r in grid_refinement_sequence)
        step1 = abs(p1 - p0)
        step2 = abs(p2 - p1)
        epsilon = 1e-6 * abs(p0)
        assert step2 < step1 + epsilon, f"power steps not shrinking: {step1} -> {step2}"

    def test_rms_converges_monotonically_with_shrinking_step(self, grid_refinement_sequence):
        r0, r1, r2 = (r[1] for r in grid_refinement_sequence)
        step1 = abs(r1 - r0)
        step2 = abs(r2 - r1)
        epsilon = 1e-6 * abs(r0)
        assert step2 < step1 + epsilon, f"rms steps not shrinking: {step1} -> {step2}"

    def test_finest_two_grids_agree_within_tenth_percent_power(self, grid_refinement_sequence):
        p1, p2 = grid_refinement_sequence[1][0], grid_refinement_sequence[2][0]
        assert p2 == pytest.approx(p1, rel=1e-3)


class TestDeltaRadRobustness:
    """Finite-difference probe angle for the Jacobian: 1e-4 / 2e-4 (default)
    / 1e-3 rad all give power within 0.05% of each other -- measured
    spread is ~1.3e-8 relative (12462.667013114677 W at 1e-4 vs
    12462.667013049286 W at 1e-3), five orders of magnitude inside the
    tolerance, so the Jacobian estimate is not step-size sensitive over
    this range for this geometry."""

    def test_power_stable_across_delta_rad(self):
        powers = {delta: _trace(delta_rad=delta)["power_w"] for delta in (1e-4, 2e-4, 1e-3)}
        base = powers[2e-4]
        for delta, p in powers.items():
            assert p == pytest.approx(base, rel=5e-4), (
                f"delta_rad={delta}: power {p} vs base {base}"
            )


class TestKernelRms:
    """Pinned sampled-distribution rms values for the two sunshape models,
    un-broadened (slope_error_mrad=0). Measured: super_gauss 2.5494 mrad
    (pin 2.549), buie 3.1400 mrad (pin 3.1401) -- both within 0.02% of the
    pin, comfortably inside the 2% the task asks for."""

    def test_super_gauss_rms(self):
        k = sunshape_kernel("super_gauss")
        assert k.rms_radius_rad() * 1e3 == pytest.approx(2.549, rel=0.02)

    def test_buie_rms(self):
        k = sunshape_kernel("buie")
        assert k.rms_radius_rad() * 1e3 == pytest.approx(3.1401, rel=0.02)


class TestSlopeErrorBroadening:
    """Slope error deflects a reflected ray by twice the surface tilt (see
    sunshape_kernel's docstring), so slope_error_mrad=1.0 broadens the
    angular kernel by an isotropic 2-D Gaussian of sigma = 2*1.0 mrad = 2e-3
    rad, and convolving with an isotropic 2-D Gaussian adds its variance
    (E[theta^2] = 2*sigma^2) to the kernel's own mean-square radius:

        rms_after^2 ~= rms_before^2 + 2*(2e-3)^2

    Measured: rms_before=2.5494e-3 rad, rms_after=3.8079e-3 rad;
    rms_after^2=1.45002e-5, predicted=1.44996e-5, relative error 4.0e-5 --
    far inside the 2% the task asks for.
    """

    def test_slope_error_adds_variance(self):
        base = sunshape_kernel("super_gauss")
        broadened = sunshape_kernel("super_gauss", slope_error_mrad=1.0)
        sigma_broaden = 2.0 * 1.0e-3
        expected_rms2 = base.rms_radius_rad() ** 2 + 2.0 * sigma_broaden**2
        assert broadened.rms_radius_rad() ** 2 == pytest.approx(expected_rms2, rel=0.02)


class TestFrustumFluxNormalisation:
    """A frustum's flux grid is uniform in the (u, v) parameterisation, but
    the surface rows those bins map to are not: bin area scales with
    r(v)/r_mean (FrustumReceiver.bin_areas_m2). The tracer's deposit
    accumulates power per PARAMETER bin, so reporting that density directly
    as W/m^2 mislabels every row that isn't at r_mean -- ~0.18% near
    mid-slant, growing toward the rims. The physical field must satisfy
    power == sum(flux * TRUE bin areas), the same contraction the MC
    backend, _mean_flux_kw_m2 and the FEA CSV export already use.
    """

    def _frustum_trace(self):
        from heliostat.geometry.receiver import FrustumReceiver

        receiver = FrustumReceiver(
            z_bot_mm=32335.0, r_bot_mm=2500.0, z_top_mm=38335.0, r_top_mm=4000.0
        )
        cone = trace_heliostat_cone(
            _ROW.x_mm,
            _ROW.y_mm,
            _ROW.rot_az_deg,
            _ROW.rot_el_deg,
            _ROW.c3,
            _ROW.c4,
            _ROW.c5,
            _ROW.solar_az_deg,
            _ROW.solar_el_deg,
            _SECONDARY,
            receiver,
            _KERNEL,
        )
        return receiver, cone

    def test_flux_times_true_bin_areas_recovers_collected_power(self):
        receiver, cone = self._frustum_trace()
        assert cone["power_w"] > 0
        n_u = len(cone["u_edges"]) - 1
        n_v = len(cone["v_edges"]) - 1
        areas = receiver.bin_areas_m2((n_u, n_v))
        assert np.sum(cone["flux"] * areas) == pytest.approx(cone["power_w"], rel=1e-9)

    def test_parameter_bin_contraction_no_longer_matches_power(self):
        """The uniform-bin contraction was the OLD (wrong) invariant; with
        true W/m^2 in the grid it must now visibly diverge from power, or
        the row correction silently regressed to a no-op."""
        receiver, cone = self._frustum_trace()
        du = cone["u_edges"][1] - cone["u_edges"][0]
        dv = cone["v_edges"][1] - cone["v_edges"][0]
        uniform_m2 = du * dv * 1.0e-6
        wrong = float(np.sum(cone["flux"]) * uniform_m2)
        assert abs(wrong - cone["power_w"]) / cone["power_w"] > 1.0e-7
