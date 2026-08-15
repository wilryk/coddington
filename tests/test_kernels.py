"""Analytic checks for the cone-optics kernel machinery."""

import numpy as np
import pytest

from heliostat.trace.kernels import RadialKernel, deposit

SIGMA = 0.0024  # rad, of order the solar half-width


class TestRadialKernel:
    def test_gaussian_normalised_and_rms(self):
        k = RadialKernel.gaussian(SIGMA)
        # 2-D Gaussian: E[theta^2] = 2 sigma^2
        assert k.rms_radius_rad() == pytest.approx(SIGMA * np.sqrt(2.0), rel=1e-4)
        integral = 2.0 * np.pi * np.trapz(k.density * k.theta_rad, k.theta_rad)
        assert integral == pytest.approx(1.0, rel=1e-12)

    def test_gaussian_convolution_adds_variances(self):
        s1, s2 = SIGMA, 1.5 * SIGMA
        conv = RadialKernel.gaussian(s1).convolve_gaussian(s2)
        expected = np.sqrt(2.0 * (s1**2 + s2**2))
        assert conv.rms_radius_rad() == pytest.approx(expected, rel=1e-3)

    def test_convolve_zero_sigma_is_identity(self):
        k = RadialKernel.gaussian(SIGMA)
        assert k.convolve_gaussian(0.0) is k

    def test_top_hat_profile(self):
        # Uniform disk of angular radius a: rms radius a / sqrt(2)
        a = 0.005
        k = RadialKernel.from_profile(lambda t: (t <= a).astype(float), support_rad=a)
        assert k.rms_radius_rad() == pytest.approx(a / np.sqrt(2.0), rel=1e-3)

    def test_rejects_bad_tables(self):
        with pytest.raises(ValueError):
            RadialKernel(np.array([0.1, 0.2]), np.array([1.0, 1.0]))  # not from 0
        with pytest.raises(ValueError):
            RadialKernel(np.array([0.0, 0.1]), np.array([0.0, 0.0]))  # zero weight


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
    uu, vv = np.meshgrid(u_mid, v_mid)
    cov_uv = (w * (uu - mu_u) * (vv - mu_v)).sum()
    return total, (mu_u, mu_v), np.array([[var_u, cov_uv], [cov_uv, var_v]])


class TestDeposit:
    def test_isotropic_map_conserves_power_and_spread(self):
        slant = 100_000.0  # mm/rad: 100 m slant range
        u_edges, v_edges = _grid()
        out = np.zeros((200, 200))
        kernel = RadialKernel.gaussian(SIGMA)
        deposit(out, u_edges, v_edges, np.array([50.0, -30.0]), slant * np.eye(2), 7.5, kernel)
        total, mu, cov = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(7.5, rel=1e-4)
        assert mu[0] == pytest.approx(50.0, abs=0.5)
        assert mu[1] == pytest.approx(-30.0, abs=0.5)
        assert np.sqrt(cov[0, 0]) == pytest.approx(slant * SIGMA, rel=2e-3)
        assert np.sqrt(cov[1, 1]) == pytest.approx(slant * SIGMA, rel=2e-3)

    def test_anisotropic_map_gives_jacobian_covariance(self):
        # Grid must contain the mapped footprint: largest singular value of
        # jac is ~1.24e5 mm/rad, so 6-sigma support reaches ~1.8 m.
        u_edges, v_edges = _grid(half_mm=2500.0, n=250)
        out = np.zeros((250, 250))
        jac = np.array([[120_000.0, 30_000.0], [0.0, 60_000.0]])
        kernel = RadialKernel.gaussian(SIGMA)
        deposit(out, u_edges, v_edges, np.zeros(2), jac, 1.0, kernel)
        total, _, cov = _moments(out, u_edges, v_edges)
        expected = SIGMA**2 * (jac @ jac.T)
        assert total == pytest.approx(1.0, rel=1e-4)
        assert np.allclose(cov, expected, rtol=5e-3)

    def test_rotation_preserves_power(self):
        u_edges, v_edges = _grid()
        kernel = RadialKernel.gaussian(SIGMA)
        c, s = np.cos(0.7), np.sin(0.7)
        rot = 80_000.0 * np.array([[c, -s], [s, c]])
        out = np.zeros((200, 200))
        deposit(out, u_edges, v_edges, np.zeros(2), rot, 3.0, kernel)
        total, _, _ = _moments(out, u_edges, v_edges)
        assert total == pytest.approx(3.0, rel=1e-4)

    def test_off_grid_footprint_deposits_nothing(self):
        u_edges, v_edges = _grid(half_mm=500.0, n=50)
        out = np.zeros((50, 50))
        kernel = RadialKernel.gaussian(SIGMA)
        deposit(out, u_edges, v_edges, np.array([50_000.0, 0.0]), 1e5 * np.eye(2), 1.0, kernel)
        assert out.sum() == 0.0

    def test_singular_jacobian_raises(self):
        u_edges, v_edges = _grid(n=10)
        out = np.zeros((10, 10))
        with pytest.raises(ValueError, match="singular"):
            deposit(
                out,
                u_edges,
                v_edges,
                np.zeros(2),
                np.zeros((2, 2)),
                1.0,
                RadialKernel.gaussian(SIGMA),
            )
