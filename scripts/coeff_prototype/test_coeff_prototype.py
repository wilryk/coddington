"""Correctness pins for the B2 coefficient-space prototype.

Run via: ``.venv\\Scripts\\python.exe -m pytest scripts/coeff_prototype/test_coeff_prototype.py -v``

Three groups, per the task brief:

1. ``sampling.py`` reproduces ``trace_heliostat_cone``'s own per-sample math
   bit-for-bit (validated via ``binned.py``, since that's the only way to
   observe the samples) -- the load-bearing test: every other test and the
   whole benchmark depend on this being true.
2. A single-Gaussian footprint reproduces analytic moments in all three
   deposits.
3. A uniform disk conserves power in all three deposits.

All seeds are fixed integers; no ``time.time()``/``datetime.now()`` seeding
anywhere in this file or the modules it exercises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT / "src", _ROOT / "tests", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.shading import MirrorGeometry
from heliostat.sweep import standard_optics
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.kernels import RadialKernel, deposit

from coeff_prototype import scenarios
from coeff_prototype.binned import deposit_binned, power_w
from coeff_prototype.bspline import control_grid_edges, deposit_bspline_coarse, evaluate_bspline
from coeff_prototype.hermite import HermiteBasis, accumulate_hermite, evaluate_hermite
from coeff_prototype.sampling import SampleBundle, flux_grid_edges, trace_heliostat_samples
from coeff_prototype.sparsity_sweep import grid_for_density, trace_field_at_grid

SIGMA = 0.0024  # rad, of order the solar half-width -- matches tests/test_kernels.py
K_MASK = 16


def _grid(half_mm=1000.0, n=200):
    edges = np.linspace(-half_mm, half_mm, n + 1)
    return edges, edges


def _moments(out, u_edges, v_edges):
    u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    total = out.sum() * du * dv
    w = out / out.sum()
    mu_u = (w.sum(axis=0) * u_mid).sum()
    mu_v = (w.sum(axis=1) * v_mid).sum()
    var_u = (w.sum(axis=0) * (u_mid - mu_u) ** 2).sum()
    var_v = (w.sum(axis=1) * (v_mid - mu_v) ** 2).sum()
    return total, (mu_u, mu_v), (var_u, var_v)


def _synthetic_bundle(kernel: RadialKernel, uv0, jac, weight: float, frac: float = 1.0,
                       node_ok=None) -> SampleBundle:
    """A one-sample bundle with no clipping/blocking, for the analytic pins."""
    axis_nodes = np.linspace(-kernel.support_rad, kernel.support_rad, K_MASK)
    au, av = np.meshgrid(axis_nodes, axis_nodes)
    w_nodes = kernel.value(np.hypot(au, av).ravel())
    if node_ok is None:
        node_ok = np.ones((1, K_MASK * K_MASK), dtype=bool)
    return SampleBundle(
        m=1,
        weights=np.array([weight]),
        uv0=np.array([[uv0[0]], [uv0[1]]]),
        jac=np.array([jac]),
        hess=None,
        smax=np.array([np.sqrt(np.linalg.eigvalsh(jac @ jac.T).max())]),
        frac=np.array([frac]),
        node_ok=node_ok,
        w_nodes=w_nodes,
        axis_nodes=axis_nodes,
        uv_nodes=np.zeros((2, 1, K_MASK * K_MASK)),
        chief_ok=np.array([True]),
        can_jac=np.array([True]),
        kernel=kernel,
        u_edges=np.array([-1000.0, 1000.0]),
        v_edges=np.array([-1000.0, 1000.0]),
        wrap_u=False,
        counters={},
    )


# ---------------------------------------------------------------------------
# 1. sampling.py reproduces trace_heliostat_cone bit-for-bit
# ---------------------------------------------------------------------------


class TestSamplingMatchesConeTrace:
    """binned.deposit_binned(trace_heliostat_samples(...)) must reproduce
    trace_heliostat_cone(...)'s own flux/power output -- proof that the
    prototype's three deposit methods are compared on identical samples,
    not merely similar ones.
    """

    def _compare(self, **kwargs):
        opt = standard_optics("prime_focus")
        kernel = sunshape_kernel("super_gauss")
        flux_grid = (128, 128)
        cone = trace_heliostat_cone(
            kwargs["x_mm"], kwargs["y_mm"], kwargs["rot_az_deg"], kwargs["rot_el_deg"],
            0.0, 0.0, 0.0, kwargs["solar_az_deg"], kwargs["solar_el_deg"],
            kwargs.get("secondary", opt.secondary), kwargs.get("receiver", opt.receiver), kernel,
            grid=(20, 12), flux_grid=flux_grid, order=1, mask_nodes=16,
            occluders=kwargs.get("occluders"),
        )
        bundle = trace_heliostat_samples(
            kwargs["x_mm"], kwargs["y_mm"], kwargs["rot_az_deg"], kwargs["rot_el_deg"],
            0.0, 0.0, 0.0, kwargs["solar_az_deg"], kwargs["solar_el_deg"],
            kwargs.get("secondary", opt.secondary), kwargs.get("receiver", opt.receiver), kernel,
            grid=(20, 12), order=1, mask_nodes=16, occluders=kwargs.get("occluders"),
        )
        u_edges, v_edges = flux_grid_edges(kwargs.get("receiver", opt.receiver), flux_grid)
        out = deposit_binned(bundle, u_edges, v_edges)
        flux_mine = out * 1.0e6
        p_mine = power_w(out, u_edges, v_edges)
        assert p_mine == pytest.approx(cone["power_w"], rel=1e-9, abs=1e-9)
        assert np.allclose(flux_mine, cone["flux"], rtol=1e-9, atol=1e-9 * max(cone["flux"].max(), 1.0))
        return cone, bundle

    def test_plain_case(self):
        opt = standard_optics("prime_focus")
        sol = opt.aim(40000.0, 10000.0, 150.0, 45.0)
        self._compare(
            x_mm=40000.0, y_mm=10000.0, rot_az_deg=sol.rot_az_deg, rot_el_deg=sol.rot_el_deg,
            solar_az_deg=150.0, solar_el_deg=45.0,
        )

    def test_with_occluders(self):
        opt = standard_optics("prime_focus")
        x_mm, y_mm, solar_az, solar_el = 40000.0, 10000.0, 150.0, 45.0
        sol = opt.aim(x_mm, y_mm, solar_az, solar_el)
        neighbours = []
        for dx in (-6000.0, 6000.0):
            for dy in (-6000.0, 6000.0):
                nx, ny = x_mm + dx, y_mm + dy
                nsol = opt.aim(nx, ny, solar_az, solar_el)
                neighbours.append(
                    MirrorGeometry.build(nx, ny, nsol.rot_az_deg, nsol.rot_el_deg, 2500.0, 1500.0)
                )
        cone, bundle = self._compare(
            x_mm=x_mm, y_mm=y_mm, rot_az_deg=sol.rot_az_deg, rot_el_deg=sol.rot_el_deg,
            solar_az_deg=solar_az, solar_el_deg=solar_el, occluders=neighbours,
        )
        # sanity: this case must actually exercise masking/blocking, or the
        # test would pass trivially without ever touching the mask path.
        assert cone["counters"]["masked"] > 0
        assert cone["counters"]["blocked"] > 0

    def test_window_clipping(self):
        opt = standard_optics("prime_focus")
        small_receiver = FlatWindowReceiver(
            z_mm=opt.receiver.z_mm, half_u_mm=700.0, half_v_mm=700.0, facing=opt.receiver.facing
        )
        x_mm, y_mm, solar_az, solar_el = 80000.0, 5000.0, 150.0, 45.0
        sol = opt.aim(x_mm, y_mm, solar_az, solar_el)
        cone, bundle = self._compare(
            x_mm=x_mm, y_mm=y_mm, rot_az_deg=sol.rot_az_deg, rot_el_deg=sol.rot_el_deg,
            solar_az_deg=solar_az, solar_el_deg=solar_el, receiver=small_receiver,
        )
        assert cone["counters"]["masked"] > 0  # this case must exercise clipping


# ---------------------------------------------------------------------------
# 2. single-Gaussian footprint reproduces analytic moments in all 3 deposits
# ---------------------------------------------------------------------------


class TestSingleGaussianMoments:
    SLANT = 100_000.0
    UV0 = (50.0, -30.0)
    WEIGHT = 7.5

    def _binned(self, kernel, u_edges, v_edges):
        out = np.zeros((200, 200))
        deposit(out, u_edges, v_edges, np.array(self.UV0), self.SLANT * np.eye(2), self.WEIGHT, kernel)
        return out

    def test_binned_reference_itself(self):
        # Pin the reference this class's other tests compare against --
        # matches tests/test_kernels.py::test_isotropic_map_conserves_power_and_spread.
        kernel = RadialKernel.gaussian(SIGMA)
        u_edges, v_edges = _grid()
        out = self._binned(kernel, u_edges, v_edges)
        total, mu, var = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(self.WEIGHT, rel=1e-4)
        assert mu[0] == pytest.approx(self.UV0[0], abs=0.5)
        assert mu[1] == pytest.approx(self.UV0[1], abs=0.5)
        assert np.sqrt(var[0]) == pytest.approx(self.SLANT * SIGMA, rel=2e-3)
        assert np.sqrt(var[1]) == pytest.approx(self.SLANT * SIGMA, rel=2e-3)

    def test_hermite_matches_analytic_moments(self):
        kernel = RadialKernel.gaussian(SIGMA)
        u_edges, v_edges = _grid()
        basis = HermiteBasis.build(kernel, order=6)
        bundle = _synthetic_bundle(kernel, self.UV0, self.SLANT * np.eye(2), self.WEIGHT)
        records, fallback = accumulate_hermite(bundle, basis)
        out = evaluate_hermite(records, fallback, basis, u_edges, v_edges, wrap_u=False)
        total, mu, var = _moments(out, u_edges, v_edges)
        # Renormalised, so total should match the binned pin tightly; a
        # Gaussian kernel is (almost) exactly representable by an order-6
        # Hermite-Gauss series about a matching width, so moments are tight
        # too -- looser than binned's own 1e-4/2e-3 tolerances since this
        # adds truncation + finite-quadrature error on top.
        assert total == pytest.approx(self.WEIGHT, rel=2e-3)
        assert mu[0] == pytest.approx(self.UV0[0], abs=1.0)
        assert mu[1] == pytest.approx(self.UV0[1], abs=1.0)
        assert np.sqrt(var[0]) == pytest.approx(self.SLANT * SIGMA, rel=5e-3)
        assert np.sqrt(var[1]) == pytest.approx(self.SLANT * SIGMA, rel=5e-3)

    def test_bspline_matches_analytic_moments(self):
        kernel = RadialKernel.gaussian(SIGMA)
        u_edges, v_edges = _grid()
        bundle = _synthetic_bundle(kernel, self.UV0, self.SLANT * np.eye(2), self.WEIGHT)
        u_c, v_c = control_grid_edges(u_edges, v_edges, (32, 32))
        coarse = deposit_bspline_coarse(bundle, u_c, v_c)
        out = evaluate_bspline(coarse, u_c, v_c, u_edges, v_edges, wrap_u=False)
        total, mu, var = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(self.WEIGHT, rel=2e-3)
        assert mu[0] == pytest.approx(self.UV0[0], abs=1.0)
        assert mu[1] == pytest.approx(self.UV0[1], abs=1.0)
        assert np.sqrt(var[0]) == pytest.approx(self.SLANT * SIGMA, rel=5e-3)
        assert np.sqrt(var[1]) == pytest.approx(self.SLANT * SIGMA, rel=5e-3)


# ---------------------------------------------------------------------------
# 3. uniform disk conserves power in all 3 deposits
# ---------------------------------------------------------------------------


class TestUniformDiskConservesPower:
    A = 0.005  # rad, angular radius
    WEIGHT = 3.0
    JAC = 80_000.0 * np.eye(2)

    def _kernel(self):
        return RadialKernel.from_profile(lambda t: (t <= self.A).astype(float), support_rad=self.A)

    def test_binned(self):
        kernel = self._kernel()
        u_edges, v_edges = _grid()
        out = np.zeros((200, 200))
        deposit(out, u_edges, v_edges, np.zeros(2), self.JAC, self.WEIGHT, kernel)
        total, _, _ = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(self.WEIGHT, rel=1e-4)

    def test_hermite(self):
        kernel = self._kernel()
        u_edges, v_edges = _grid()
        basis = HermiteBasis.build(kernel, order=6)
        bundle = _synthetic_bundle(kernel, (0.0, 0.0), self.JAC, self.WEIGHT)
        records, fallback = accumulate_hermite(bundle, basis)
        out = evaluate_hermite(records, fallback, basis, u_edges, v_edges, wrap_u=False)
        total, _, _ = _moments(out, u_edges, v_edges)
        # A top-hat is a much harder target for a Hermite series than a
        # Gaussian (sharp discontinuity at theta=a); renormalisation still
        # forces the ANALYTIC integral to match exactly, but the finite
        # bounding-box quadrature used to reconstruct it numerically here
        # leaves a small residual -- looser tolerance than the Gaussian case.
        assert total == pytest.approx(self.WEIGHT, rel=1e-2)

    def test_bspline(self):
        kernel = self._kernel()
        u_edges, v_edges = _grid()
        bundle = _synthetic_bundle(kernel, (0.0, 0.0), self.JAC, self.WEIGHT)
        u_c, v_c = control_grid_edges(u_edges, v_edges, (32, 32))
        coarse = deposit_bspline_coarse(bundle, u_c, v_c)
        out = evaluate_bspline(coarse, u_c, v_c, u_edges, v_edges, wrap_u=False)
        total, _, _ = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(self.WEIGHT, rel=5e-3)


# ---------------------------------------------------------------------------
# 4. sparsity_sweep.py: density->grid mapping and field-total conservation
# ---------------------------------------------------------------------------


class TestSparsityDensityGridMapping:
    """grid_for_density() must reproduce the current hardcoded 20x12 grid
    exactly at its own reference density on the manuscript's 5:3 mirror
    bbox -- if this ever drifts, every "vs reference" number in the sweep's
    report silently stops meaning what the report says it means."""

    def test_reference_density_reproduces_20x12_on_5x3_bbox(self):
        assert grid_for_density(16.0, 5.0, 3.0) == (20, 12)

    def test_grid_aspect_matches_bbox_and_scales_with_density(self):
        # 5*sqrt(4)=10, 3*sqrt(4)=6 -- exact, no rounding ambiguity.
        n_x, n_y = grid_for_density(4.0, 5.0, 3.0)
        assert (n_x, n_y) == (10, 6)
        assert n_x / n_y == pytest.approx(5.0 / 3.0, rel=1e-9)

    def test_min_grid_floor_at_very_low_density(self):
        n_x, n_y = grid_for_density(0.01, 5.0, 3.0, min_n=2)
        assert n_x >= 2
        assert n_y >= 2


class TestSparsitySweepFieldTotalConservation:
    """A sweep rung's reported field-total power must equal an
    independently computed sum -- both the sum of its own per-heliostat
    powers (catches a bug in trace_field_at_grid's bookkeeping) and a
    from-scratch re-trace/re-deposit of each case outside that function
    entirely (catches a bug in the accumulation itself, not just the
    arithmetic around it)."""

    def test_field_total_matches_independent_sum(self):
        scenario = scenarios.scenario_default_field()
        cases = scenario.cases[:5]  # small subset -- this pin is about bookkeeping, not scale
        u_edges, v_edges = flux_grid_edges(scenario.receiver, scenario.flux_grid)
        grid = grid_for_density(1.0, 5.0, 3.0)  # coarse, keeps the test fast

        result = trace_field_at_grid(cases, scenario.secondary, scenario.receiver, u_edges, v_edges, grid)
        assert result["field_total_power_w"] == pytest.approx(
            sum(result["per_heliostat_power_w"]), rel=1e-9
        )

        kernel = sunshape_kernel("super_gauss")
        du = u_edges[1] - u_edges[0]
        dv = v_edges[1] - v_edges[0]
        independent_total = 0.0
        for case in cases:
            bundle = trace_heliostat_samples(
                case.x_mm, case.y_mm, case.rot_az_deg, case.rot_el_deg,
                case.c3, case.c4, case.c5, case.solar_az_deg, case.solar_el_deg,
                scenario.secondary, scenario.receiver, kernel,
                grid=grid, order=2, mask_nodes=16, occluders=case.occluders,
            )
            out = np.zeros((v_edges.size - 1, u_edges.size - 1))
            deposit_binned(bundle, u_edges, v_edges, out=out)
            independent_total += float(out.sum() * du * dv)
        assert result["field_total_power_w"] == pytest.approx(independent_total, rel=1e-9)
