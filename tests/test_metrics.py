"""Tests for heliostat.metrics.

``cfg`` throughout is a minimal duck-typed stand-in for the project's real
config object -- metrics.py only ever reads ``cfg.source.watts_per_ray(n)``,
``cfg.optics.throughput``, and a handful of ``cfg.receiver`` geometry fields,
so a lightweight ``SimpleNamespace`` tree is enough to exercise it fully
without pulling in any config-loading machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from heliostat import metrics


def make_cfg(window_mm=150.0, grid_size=300, power_w=1000.0, throughput=0.81):
    bin_size_mm = 2.0 * window_mm / grid_size
    receiver = SimpleNamespace(
        window_mm=window_mm,
        grid_size=grid_size,
        bin_size_mm=bin_size_mm,
        bin_area_m2=(bin_size_mm / 1000.0) ** 2,
        edges=np.linspace(-window_mm, window_mm, grid_size + 1),
    )

    def _watts_per_ray(n):
        if n <= 0:
            raise ValueError("rays_emitted must be positive")
        return power_w / n

    source = SimpleNamespace(watts_per_ray=_watts_per_ray)
    optics = SimpleNamespace(throughput=throughput)
    return SimpleNamespace(receiver=receiver, source=source, optics=optics)


# ---------------------------------------------------------------------------
# scale_factor
# ---------------------------------------------------------------------------


class TestScaleFactor:
    def test_basic_value(self):
        cfg = make_cfg(power_w=1000.0, throughput=0.81)
        sf = metrics.scale_factor(cfg, rays_emitted=1000, dni_w_m2=1000.0)
        assert sf == pytest.approx((1000.0 / 1000) * 0.81 * 1.0)

    def test_scales_linearly_with_dni(self):
        cfg = make_cfg()
        sf_1000 = metrics.scale_factor(cfg, 1000, dni_w_m2=1000.0)
        sf_500 = metrics.scale_factor(cfg, 1000, dni_w_m2=500.0)
        assert sf_500 == pytest.approx(sf_1000 * 0.5)


# ---------------------------------------------------------------------------
# bin geometry helpers
# ---------------------------------------------------------------------------


class TestBinGeometry:
    def test_bin_centres_span_and_count(self):
        cfg = make_cfg(window_mm=100.0, grid_size=10)
        c = metrics.bin_centres(cfg)
        assert c.shape == (10,)
        assert c.min() > -100.0
        assert c.max() < 100.0
        # bins are symmetric about zero
        assert c[0] == pytest.approx(-c[-1])

    def test_bin_radius_shape_and_zero_at_centre_bin(self):
        cfg = make_cfg(window_mm=100.0, grid_size=10)  # even grid -> no exact-zero bin
        r = metrics.bin_radius(cfg)
        assert r.shape == (10, 10)
        assert np.all(r >= 0.0)

    def test_bin_radius_odd_grid_has_zero_centre(self):
        cfg = make_cfg(window_mm=99.0, grid_size=11)
        r = metrics.bin_radius(cfg)
        assert r.min() == pytest.approx(0.0, abs=1e-9)

    def test_radial_mask_matches_radial_masks_stack(self):
        cfg = make_cfg(window_mm=100.0, grid_size=20)
        radii = [10.0, 30.0, 60.0]
        stacked = metrics.radial_masks(cfg, radii)
        for i, radius in enumerate(radii):
            single = metrics.radial_mask(cfg, radius)
            assert np.array_equal(stacked[i], single)

    def test_radial_mask_grows_with_radius(self):
        cfg = make_cfg(window_mm=100.0, grid_size=20)
        small = metrics.radial_mask(cfg, 10.0)
        big = metrics.radial_mask(cfg, 50.0)
        assert small.sum() < big.sum()
        # small mask is a subset of big mask
        assert np.all(big[small])


# ---------------------------------------------------------------------------
# spot_metrics -- analytically known inputs
# ---------------------------------------------------------------------------

SPOT_METRICS_KEYS = {
    "rays_emitted",
    "rays_landed",
    "transmission",
    "power_w",
    "shading_blocking_efficiency",
    "centroid_x_mm",
    "centroid_y_mm",
    "rms_radius_mm",
    "r50_mm",
    "r90_mm",
    "peak_flux_w_m2",
    "spillage",
}


def gaussian_spot(sigma_mm, n, centre=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(centre[0], sigma_mm, n)
    y = rng.normal(centre[1], sigma_mm, n)
    return np.column_stack([x, y])


class TestSpotMetricsGaussian:
    """A symmetric isotropic 2D Gaussian has closed-form r50/r90/rms."""

    SIGMA = 10.0
    N = 200_000

    @pytest.fixture(scope="class")
    @classmethod
    def xy(cls):
        return gaussian_spot(cls.SIGMA, cls.N, centre=(0.0, 0.0), seed=42)

    @pytest.fixture(scope="class")
    @classmethod
    def cfg(cls):
        return make_cfg(window_mm=150.0, grid_size=300)

    def test_keys_pinned(self, xy, cfg):
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        assert set(out.keys()) == SPOT_METRICS_KEYS

    def test_centroid_near_origin(self, xy, cfg):
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        # standard error of the mean ~ sigma/sqrt(N) ~ 0.022 mm; generous margin
        assert abs(out["centroid_x_mm"]) < 0.3
        assert abs(out["centroid_y_mm"]) < 0.3

    def test_rms_radius_matches_theory(self, xy, cfg):
        # For an isotropic 2D Gaussian, E[r^2] = 2*sigma^2 -> rms = sigma*sqrt(2)
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        expected = self.SIGMA * np.sqrt(2.0)
        assert out["rms_radius_mm"] == pytest.approx(expected, rel=0.02)

    def test_r50_and_r90_match_rayleigh_theory(self, xy, cfg):
        # Radius from a 2D isotropic Gaussian is Rayleigh-distributed:
        # P(r < r_f) = f  =>  r_f = sigma * sqrt(-2 ln(1-f))
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        expected_r50 = self.SIGMA * np.sqrt(-2.0 * np.log(0.5))
        expected_r90 = self.SIGMA * np.sqrt(-2.0 * np.log(0.1))
        assert out["r50_mm"] == pytest.approx(expected_r50, rel=0.02)
        assert out["r90_mm"] == pytest.approx(expected_r90, rel=0.02)

    def test_r50_less_than_r90(self, xy, cfg):
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        assert out["r50_mm"] < out["r90_mm"]

    def test_transmission_full_when_all_rays_land(self, xy, cfg):
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg)
        assert out["transmission"] == pytest.approx(1.0)

    def test_power_scales_linearly_with_efficiency(self, xy, cfg):
        out1 = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg, efficiency=1.0)
        out2 = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg, efficiency=0.4)
        assert out2["power_w"] == pytest.approx(out1["power_w"] * 0.4)

    def test_power_scales_linearly_with_dni(self, xy, cfg):
        out1 = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg, dni_w_m2=1000.0)
        out2 = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg, dni_w_m2=250.0)
        assert out2["power_w"] == pytest.approx(out1["power_w"] * 0.25)

    def test_aperture_radius_reports_sensible_spillage(self, xy, cfg):
        out = metrics.spot_metrics(xy, rays_emitted=self.N, cfg=cfg, aperture_radius_mm=self.SIGMA)
        # aperture at 1 sigma of a Rayleigh-distributed radius captures
        # 1 - exp(-r^2/2sigma^2) = 1 - exp(-0.5) ~= 0.393
        expected_capture = 1.0 - np.exp(-0.5)
        assert (1.0 - out["spillage"]) == pytest.approx(expected_capture, abs=0.01)
        assert "power_in_aperture_w" in out


class TestSpotMetricsDeltaCluster:
    """All rays land at exactly the same point -- rms/r50/r90 must be exactly 0."""

    def test_all_zero_spread(self):
        cfg = make_cfg()
        xy = np.tile([12.5, -7.25], (500, 1)).astype(float)
        out = metrics.spot_metrics(xy, rays_emitted=500, cfg=cfg)
        assert out["centroid_x_mm"] == pytest.approx(12.5)
        assert out["centroid_y_mm"] == pytest.approx(-7.25)
        assert out["rms_radius_mm"] == pytest.approx(0.0, abs=1e-9)
        assert out["r50_mm"] == pytest.approx(0.0, abs=1e-9)
        assert out["r90_mm"] == pytest.approx(0.0, abs=1e-9)


class TestSpotMetricsEmpty:
    def test_no_rays_landed(self):
        cfg = make_cfg()
        xy = np.empty((0, 2))
        out = metrics.spot_metrics(xy, rays_emitted=1000, cfg=cfg)
        assert set(out.keys()) == SPOT_METRICS_KEYS
        assert out["rays_landed"] == 0
        assert out["transmission"] == 0.0
        assert out["power_w"] == 0.0
        assert np.isnan(out["centroid_x_mm"])
        assert np.isnan(out["rms_radius_mm"])
        assert np.isnan(out["r50_mm"])
        assert out["peak_flux_w_m2"] == 0.0

    def test_zero_rays_emitted_raises(self):
        # rays_emitted=0 is nonsensical (a heliostat with no ray budget at
        # all), and the watts-per-ray contract rejects it explicitly.
        cfg = make_cfg()
        xy = np.empty((0, 2))
        with pytest.raises(ValueError):
            metrics.spot_metrics(xy, rays_emitted=0, cfg=cfg)


# ---------------------------------------------------------------------------
# encircled energy: masked-sum vs raw-ray paths should agree
# ---------------------------------------------------------------------------


class TestEncircledEnergy:
    def test_rays_and_binned_paths_agree_approximately(self):
        cfg = make_cfg(window_mm=150.0, grid_size=300)
        xy = gaussian_spot(10.0, 100_000, seed=7)
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[cfg.receiver.edges] * 2)

        r_rays, p_rays, f_rays = metrics.encircled_energy_rays(xy, 100_000, cfg)
        r_bin, p_bin, f_bin = metrics.encircled_energy(counts, 100_000, cfg)

        assert np.allclose(r_rays, r_bin)
        # 1 mm bin quantization moves individual points across a radius step,
        # which is a large *relative* error where the cumulative power is
        # still small -- bound the disagreement as a fraction of total power
        # instead of pointwise.
        total = p_rays[-1]
        assert np.max(np.abs(p_rays - p_bin)) < 0.03 * total
        assert f_rays[-1] == pytest.approx(1.0, abs=1e-6)
        assert f_bin[-1] == pytest.approx(1.0, abs=1e-6)

    def test_encircled_energy_monotonic_nondecreasing(self):
        cfg = make_cfg(window_mm=150.0, grid_size=300)
        xy = gaussian_spot(10.0, 50_000, seed=3)
        _, power, _ = metrics.encircled_energy_rays(xy, 50_000, cfg)
        assert np.all(np.diff(power) >= -1e-9)

    def test_empty_rays_all_zero(self):
        cfg = make_cfg()
        xy = np.empty((0, 2))
        radii, power, frac = metrics.encircled_energy_rays(xy, 1000, cfg)
        assert np.all(power == 0.0)
        assert np.all(frac == 0.0)


class TestEncircledEnergyRadii:
    def test_matches_spot_metrics_r50_r90(self):
        xy = gaussian_spot(10.0, 100_000, seed=11)
        radii = metrics.encircled_energy_radii(xy, (0.5, 0.9))
        cfg = make_cfg()
        out = metrics.spot_metrics(xy, rays_emitted=100_000, cfg=cfg)
        assert radii[0.5] == pytest.approx(out["r50_mm"])
        assert radii[0.9] == pytest.approx(out["r90_mm"])

    def test_empty_returns_nan(self):
        radii = metrics.encircled_energy_radii(np.empty((0, 2)), (0.5, 0.9))
        assert np.isnan(radii[0.5])
        assert np.isnan(radii[0.9])


# ---------------------------------------------------------------------------
# aperture_metrics
# ---------------------------------------------------------------------------


class TestApertureMetrics:
    def test_full_aperture_captures_everything(self):
        cfg = make_cfg(window_mm=150.0, grid_size=300)
        xy = gaussian_spot(10.0, 50_000, seed=5)
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[cfg.receiver.edges] * 2)
        out = metrics.aperture_metrics(counts, 50_000, cfg, radius_mm=150.0)
        assert out["spillage"] == pytest.approx(0.0, abs=1e-9)
        assert out["power_w"] == pytest.approx(out["power_total_w"])

    def test_zero_aperture_captures_nothing(self):
        cfg = make_cfg(window_mm=150.0, grid_size=300)
        xy = gaussian_spot(10.0, 50_000, seed=5)
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[cfg.receiver.edges] * 2)
        out = metrics.aperture_metrics(counts, 50_000, cfg, radius_mm=0.0)
        assert out["spillage"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# map_metrics
# ---------------------------------------------------------------------------


class TestMapMetrics:
    def test_matches_spot_metrics_on_centroid_and_power(self):
        cfg = make_cfg(window_mm=150.0, grid_size=300)
        xy = gaussian_spot(10.0, 80_000, centre=(3.0, -2.0), seed=9)
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[cfg.receiver.edges] * 2)
        m = metrics.map_metrics(counts, 80_000, cfg)
        s = metrics.spot_metrics(xy, rays_emitted=80_000, cfg=cfg)
        assert m["centroid_x_mm"] == pytest.approx(s["centroid_x_mm"], abs=0.5)
        assert m["centroid_y_mm"] == pytest.approx(s["centroid_y_mm"], abs=0.5)
        assert m["power_w"] == pytest.approx(s["power_w"], rel=1e-6)

    def test_empty_map(self):
        cfg = make_cfg()
        counts = np.zeros((cfg.receiver.grid_size, cfg.receiver.grid_size))
        out = metrics.map_metrics(counts, 1000, cfg)
        assert out["rays_landed"] == 0
        assert out["power_w"] == 0.0
        assert np.isnan(out["centroid_x_mm"])


# ---------------------------------------------------------------------------
# rank_heliostats
# ---------------------------------------------------------------------------


class TestRankHeliostats:
    def _summary(self):
        return pd.DataFrame(
            {
                "heliostat_id": [1, 1, 2, 2, 3, 3],
                "x_m": [0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
                "y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "radius_m": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "hour": [8.0, 12.0, 8.0, 12.0, 8.0, 12.0],
                "power_w": [100.0, 200.0, 50.0, 50.0, 300.0, 300.0],
                "transmission": [0.9, 0.95, 0.5, 0.5, 0.99, 0.99],
            }
        )

    def test_sums_extensive_quantity(self):
        ranked = metrics.rank_heliostats(self._summary(), by="power_w", ascending=True)
        row = ranked[ranked["heliostat_id"] == 1].iloc[0]
        assert row["power_w_sum"] == pytest.approx(300.0)

    def test_averages_intensive_quantity(self):
        ranked = metrics.rank_heliostats(self._summary(), by="transmission", ascending=True)
        row = ranked[ranked["heliostat_id"] == 1].iloc[0]
        assert row["transmission_mean"] == pytest.approx(0.925)

    def test_rank_ordering_ascending(self):
        ranked = metrics.rank_heliostats(self._summary(), by="power_w", ascending=True)
        # heliostat 2 has lowest total power_w (100), then 1 (300), then 3 (600)
        assert list(ranked["heliostat_id"]) == [2, 1, 3]
        assert list(ranked["rank"]) == [1, 2, 3]
