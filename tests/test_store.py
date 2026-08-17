"""Tests for heliostat.store.

Exercises RunStore end-to-end on a tiny synthetic run (2 heliostats x 2
timesteps, a few hundred fake rays each) written and read back through the
same public API a real trace would use: no fixture files, no private-repo
run needed for this module's own contract.

``cfg`` is the same duck-typed shape used throughout this project's tests
(see ``tests/test_metrics.py``, ``tests/test_mc_parity.py``): a
``SimpleNamespace`` tree exposing exactly the fields RunStore reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from heliostat.geometry.design import rect_heliostat
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.store import INT16_MAX, RunStore, TimestepResult, flux_scale, scale_factor

WINDOW_MM = 200.0
GRID_SIZE = 16


def make_cfg(power_w=1000.0, throughput=0.81, raw_rays="all"):
    bin_size_mm = 2.0 * WINDOW_MM / GRID_SIZE
    receiver = SimpleNamespace(
        window_mm=WINDOW_MM,
        grid_size=GRID_SIZE,
        bin_size_mm=bin_size_mm,
        bin_area_m2=(bin_size_mm / 1000.0) ** 2,
        edges=np.linspace(-WINDOW_MM, WINDOW_MM, GRID_SIZE + 1),
    )

    def _watts_per_ray(n):
        return power_w / n

    return SimpleNamespace(
        receiver=receiver,
        source=SimpleNamespace(power_w=power_w, watts_per_ray=_watts_per_ray),
        optics=SimpleNamespace(throughput=throughput),
        trace=SimpleNamespace(rays_per_heliostat=200),
        storage=SimpleNamespace(raw_rays=raw_rays),
    )


def _fake_rays(rng, n, centre=(0.0, 0.0), sigma=20.0):
    xy = rng.normal(centre, sigma, size=(n, 2))
    return np.clip(xy, -WINDOW_MM + 1e-6, WINDOW_MM - 1e-6)


def write_synthetic_run(tmp_path, cfg, heliostat_ids=(1, 2), keys=("t0", "t1"), n_rays=300, seed=0):
    """2 heliostats x len(keys) timesteps, ~n_rays fake rays each."""
    store = RunStore(tmp_path, cfg=cfg, mode="w")
    rng = np.random.default_rng(seed)

    for ti, key in enumerate(keys):
        counts_list = []
        rows = []
        all_xy = []
        index_rows = []
        cursor = 0
        for hi, hid in enumerate(heliostat_ids):
            xy = _fake_rays(rng, n_rays, centre=(hi * 10.0, ti * 5.0))
            inside = RunStore.inside_window(xy, cfg.receiver.window_mm)
            xy = xy[inside]
            quant = RunStore.quantise(xy, cfg.receiver.window_mm)
            counts, _, _ = np.histogram2d(
                xy[:, 1], xy[:, 0], bins=[cfg.receiver.edges, cfg.receiver.edges]
            )
            counts_list.append(counts.astype(np.uint32))
            all_xy.append(quant)
            index_rows.append([hid, cursor, quant.shape[0]])
            cursor += quant.shape[0]
            rows.append(
                {
                    "date": "2026-01-01",
                    "hour": 10.0 + ti,
                    "timestep": key,
                    "heliostat_id": hid,
                    "rays_emitted": n_rays,
                    "rays_landed": int(xy.shape[0]),
                    "power_w": xy.shape[0]
                    * cfg.source.watts_per_ray(n_rays)
                    * cfg.optics.throughput,
                }
            )
        result = TimestepResult(
            key=key,
            date="2026-01-01",
            hour=10.0 + ti,
            solar_az_deg=180.0,
            solar_el_deg=45.0,
            heliostat_ids=np.array(heliostat_ids),
            rays_emitted=n_rays,
            counts=np.stack(counts_list),
            rays=np.concatenate(all_xy).astype(np.int16)
            if cfg.storage.raw_rays != "none"
            else None,
            index=np.array(index_rows, dtype=np.int64) if cfg.storage.raw_rays != "none" else None,
            rows=pd.DataFrame(rows),
        )
        store.write_timestep(result)

    store.write_manifest(cfg, extra={"heliostat_ids": list(heliostat_ids)})
    return store


# ---------------------------------------------------------------------------
# quantisation round-trip
# ---------------------------------------------------------------------------


class TestQuantisation:
    def test_inside_window_masks_correctly(self):
        xy = np.array([[0.0, 0.0], [WINDOW_MM + 1, 0.0], [0.0, -WINDOW_MM - 1], [50.0, -50.0]])
        mask = RunStore.inside_window(xy, WINDOW_MM)
        assert list(mask) == [True, False, False, True]

    def test_quantise_dequantise_round_trip_within_scale(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)

        raw_original = _fake_rays(np.random.default_rng(1), 500)
        inside = RunStore.inside_window(raw_original, WINDOW_MM)
        raw_original = raw_original[inside]
        q = RunStore.quantise(raw_original, WINDOW_MM)
        deq = store.dequantise(q)

        # Round-trip error bounded by one quantisation step.
        assert np.max(np.abs(deq - raw_original)) <= store.quant_scale + 1e-6

    def test_quantisation_scale_matches_window_over_int16max(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        assert store.quant_scale == pytest.approx(cfg.receiver.window_mm / INT16_MAX)

    def test_quantise_clips_to_window_not_before(self):
        # quantise() itself clips (a safety net); the real boundary control
        # is inside_window() filtering first.
        xy = np.array([[WINDOW_MM * 2, 0.0]])
        q = RunStore.quantise(xy, WINDOW_MM)
        assert q[0, 0] == INT16_MAX


# ---------------------------------------------------------------------------
# flux binning == histogram of raws
# ---------------------------------------------------------------------------


class TestFluxBinningMatchesRaws:
    def test_read_counts_matches_rebin_of_read_rays(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        key = store.timestep_keys()[0]

        counts = np.asarray(store.read_counts(key))
        for row, hid in enumerate((1, 2)):
            rebinned = store.rebin(
                key, cfg.receiver.grid_size, cfg.receiver.window_mm, heliostat_id=hid
            )
            assert np.array_equal(rebinned, counts[row])

    def test_read_rays_all_heliostats_concatenates_index(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        key = store.timestep_keys()[0]

        all_rays = store.read_rays(key)
        index = store.read_index(key)
        assert index[:, 2].sum() == all_rays.shape[0]

        for hid, start, count in index:
            one = store.read_rays(key, heliostat_id=int(hid))
            assert one.shape[0] == count
            np.testing.assert_array_equal(one, all_rays[start : start + count])

    def test_read_rays_missing_raises_when_raw_rays_off(self, tmp_path):
        cfg = make_cfg(raw_rays="none")
        store = write_synthetic_run(tmp_path, cfg)
        key = store.timestep_keys()[0]
        with pytest.raises(FileNotFoundError):
            store.read_rays(key)


# ---------------------------------------------------------------------------
# rebin power invariance
# ---------------------------------------------------------------------------


class TestRebinPowerInvariance:
    def test_rebin_at_finer_grid_preserves_total_power(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        key = store.timestep_keys()[0]

        native = np.asarray(store.read_counts(key)[0])
        fine = store.rebin(key, grid_size=64, window_mm=cfg.receiver.window_mm, heliostat_id=1)
        coarse = store.rebin(key, grid_size=8, window_mm=cfg.receiver.window_mm, heliostat_id=1)

        # Total landed-ray count is conserved under re-histogramming at any
        # resolution over the same window -- only where it lands differs.
        assert fine.sum() == pytest.approx(native.sum())
        assert coarse.sum() == pytest.approx(native.sum())

    def test_rebin_wider_window_still_conserves_rays_inside_original(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        key = store.timestep_keys()[0]

        native = np.asarray(store.read_counts(key)[0])
        wider = store.rebin(
            key,
            grid_size=cfg.receiver.grid_size,
            window_mm=cfg.receiver.window_mm * 2,
            heliostat_id=1,
        )
        # A wider window can only capture the same rays or more (none were
        # outside the original window here since inside_window filtered at
        # write time), so total count is unchanged.
        assert wider.sum() == pytest.approx(native.sum())


# ---------------------------------------------------------------------------
# flux_scale for both flux_kinds
# ---------------------------------------------------------------------------


class TestFluxScale:
    def test_ray_counts_matches_legacy_scale_factor(self):
        cfg = make_cfg(power_w=2000.0, throughput=0.81)
        sf = scale_factor(cfg, rays_emitted=1000, dni_w_m2=1000.0)
        fs = flux_scale(cfg, rays_emitted=1000, dni_w_m2=1000.0, flux_kind="ray_counts")
        assert fs == sf == pytest.approx((2000.0 / 1000) * 0.81)

    def test_analytic_ignores_ray_budget(self):
        cfg = make_cfg(power_w=2000.0, throughput=0.81)
        # analytic counts are already power; the ray budget must not enter.
        fs_a = flux_scale(cfg, rays_emitted=1000, dni_w_m2=1000.0, flux_kind="analytic")
        fs_b = flux_scale(cfg, rays_emitted=999999, dni_w_m2=1000.0, flux_kind="analytic")
        assert fs_a == fs_b == pytest.approx(0.81)

    def test_both_scale_linearly_with_dni(self):
        cfg = make_cfg()
        for kind in ("ray_counts", "analytic"):
            full = flux_scale(cfg, 1000, dni_w_m2=1000.0, flux_kind=kind)
            half = flux_scale(cfg, 1000, dni_w_m2=500.0, flux_kind=kind)
            assert half == pytest.approx(full * 0.5)

    def test_unknown_flux_kind_raises(self):
        cfg = make_cfg()
        with pytest.raises(ValueError):
            flux_scale(cfg, 1000, flux_kind="bogus")

    def test_field_flux_and_heliostat_flux_use_manifest_flux_kind(self, tmp_path):
        cfg = make_cfg(power_w=1000.0, throughput=0.5)
        store = write_synthetic_run(tmp_path, cfg)
        # Rewrite the manifest declaring analytic flux (as if counts already
        # held power) and confirm field_flux picks the analytic formula up
        # via self.flux_kind rather than a hardcoded ray_counts path.
        store.write_manifest(cfg, flux_kind="analytic", extra={"heliostat_ids": [1, 2]})
        assert store.flux_kind == "analytic"

        key = store.timestep_keys()[0]
        counts = np.asarray(store.read_counts(key)).astype(np.float64)
        expected = (
            counts.sum(axis=0) * flux_scale(cfg, 1, 1000.0, "analytic") / cfg.receiver.bin_area_m2
        )
        got = store.field_flux(key, cfg=cfg)
        np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# manifest design/receiver round-trip
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def test_absent_receiver_and_design_return_none(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg)
        assert store.receiver_from_manifest() is None
        assert store.design_from_manifest() is None

    def test_receiver_round_trips(self, tmp_path):
        cfg = make_cfg()
        store = RunStore(tmp_path, cfg=cfg, mode="w")
        receiver = FlatWindowReceiver(z_mm=7000.0, half_u_mm=200.0, half_v_mm=200.0, facing="up")
        store.write_manifest(cfg, receiver=receiver)

        rebuilt = store.receiver_from_manifest()
        assert isinstance(rebuilt, FlatWindowReceiver)
        assert rebuilt.z_mm == receiver.z_mm
        assert rebuilt.half_u_mm == receiver.half_u_mm
        assert rebuilt.facing == receiver.facing

    def test_design_round_trips(self, tmp_path):
        cfg = make_cfg()
        store = RunStore(tmp_path, cfg=cfg, mode="w")
        design = rect_heliostat(width_mm=5000.0, height_mm=3000.0)
        store.write_manifest(cfg, design=design)

        rebuilt = store.design_from_manifest()
        assert rebuilt.to_dict() == design.to_dict()

    def test_flux_kind_defaults_to_ray_counts(self, tmp_path):
        cfg = make_cfg()
        store = RunStore(tmp_path, cfg=cfg, mode="w")
        store.write_manifest(cfg)
        assert store.flux_kind == "ray_counts"
        assert store.manifest["flux_kind"] == "ray_counts"

    def test_extra_overrides_and_extends_payload(self, tmp_path):
        cfg = make_cfg()
        store = RunStore(tmp_path, cfg=cfg, mode="w")
        store.write_manifest(cfg, extra={"throughput": 0.42, "custom_note": "hi"})
        assert store.manifest["throughput"] == 0.42
        assert store.manifest["custom_note"] == "hi"

    def test_unknown_manifest_keys_tolerated(self, tmp_path):
        # A manifest from a future or foreign writer with extra keys must
        # not break reading -- RunStore does not validate the whole schema.
        cfg = make_cfg()
        store = RunStore(tmp_path, cfg=cfg, mode="w")
        store.write_manifest(cfg, extra={"some_future_field": {"nested": True}})
        reopened = RunStore(tmp_path)
        assert reopened.manifest["some_future_field"] == {"nested": True}
        assert reopened.flux_kind == "ray_counts"


# ---------------------------------------------------------------------------
# summary() and timestep bookkeeping
# ---------------------------------------------------------------------------


class TestSummaryAndBookkeeping:
    def test_summary_has_one_row_per_heliostat_per_timestep(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg, heliostat_ids=(1, 2), keys=("t0", "t1"))
        summary = store.summary()
        assert len(summary) == 4
        assert set(summary["heliostat_id"]) == {1, 2}

    def test_has_timestep_and_timestep_keys(self, tmp_path):
        cfg = make_cfg()
        store = write_synthetic_run(tmp_path, cfg, keys=("t0", "t1"))
        assert store.timestep_keys() == ["t0", "t1"]
        assert store.has_timestep("t0")
        assert not store.has_timestep("t9")

    def test_missing_store_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunStore(tmp_path / "does_not_exist")
