"""Tests for heliostat.sweep.

Kept under ~90 s total: small fields, small ray counts, ``workers=1`` (no
multiprocessing.Pool spin-up cost) everywhere except where the CLI test
exercises the real default path.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from heliostat import dni, energy
from heliostat.cli import main as cli_main
from heliostat.field import HeliostatField
from heliostat.field_layouts import generate
from heliostat.metrics import spot_metrics
from heliostat.store import RunStore
from heliostat.sweep import run_sweep, standard_optics
from heliostat.trace.mc import trace_heliostat

BASE_SEED = 20260811


def _far_field(ids=(11, 47, 203)) -> HeliostatField:
    """Three widely separated heliostats -- far enough apart that no pair
    ever shades or blocks another (eta_occlusion == 1.0 everywhere), so a
    seed-contract comparison is not entangled with the occlusion model."""
    xy_mm = np.array([[0.0, 90000.0], [90000.0, -20000.0], [-95000.0, 10000.0]])
    return HeliostatField(
        x_mm=xy_mm[:, 0],
        y_mm=xy_mm[:, 1],
        ids=np.array(ids),
        mirror_width_mm=5000.0,
        mirror_height_mm=3000.0,
        source="test_sweep._far_field",
    )


def _bare_cfg(throughput: float, window_mm: float = 2000.0, grid_size: int = 128):
    """Minimal duck-typed cfg for spot_metrics -- same shape as
    tests/test_mc_parity.py::_make_cfg."""
    bin_size_mm = 2.0 * window_mm / grid_size
    receiver = SimpleNamespace(
        window_mm=window_mm,
        grid_size=grid_size,
        bin_size_mm=bin_size_mm,
        bin_area_m2=(bin_size_mm / 1000.0) ** 2,
        edges=np.linspace(-window_mm, window_mm, grid_size + 1),
    )
    return SimpleNamespace(
        receiver=receiver,
        source=SimpleNamespace(watts_per_ray=lambda n: 38484.5 / n if n else 0.0),
        optics=SimpleNamespace(throughput=throughput),
    )


# ---------------------------------------------------------------------------
# 1. Seed-contract test
# ---------------------------------------------------------------------------


class TestSeedContract:
    """run_sweep's monte_carlo output must be exactly what a direct
    heliostat.trace.mc.trace_heliostat call produces with the fixture seed
    scheme: default_rng(SeedSequence((base_seed, int(step.key without '_'),
    heliostat_id))). This ties sweep output to that convention forever."""

    def test_power_centroid_rms_match_direct_trace(self, tmp_path):
        field = _far_field()
        n_rays = 4000

        store = run_sweep(
            field,
            [_dt.date(2026, 3, 21)],
            mode="monte_carlo",
            optics="prime_focus",
            n_rays=n_rays,
            workers=1,
            hour_step=6.0,
            base_seed=BASE_SEED,
            out_dir=tmp_path / "run",
            progress=lambda _msg: None,
        )
        summary = store.summary()
        assert len(summary) > 0

        opt = standard_optics("prime_focus")
        cfg = _bare_cfg(opt.throughput)

        for _, row in summary.iterrows():
            # eta_occlusion must be 1.0 everywhere for this field (see
            # _far_field docstring) -- assert it so a regression in the
            # occlusion geometry can't silently make this test meaningless.
            assert row["eta_occlusion"] == pytest.approx(1.0)

            step_int = int(row["timestep"].replace("_", ""))
            heliostat_id = int(row["heliostat_id"])
            x_mm, y_mm = row["x_m"] * 1000.0, row["y_m"] * 1000.0

            sol = opt.aim(x_mm, y_mm, row["solar_az_deg"], row["solar_el_deg"])
            assert sol.rot_az_deg == pytest.approx(row["rot_az_deg"], abs=1e-9)
            assert sol.rot_el_deg == pytest.approx(row["rot_el_deg"], abs=1e-9)

            rng = np.random.default_rng(np.random.SeedSequence((BASE_SEED, step_int, heliostat_id)))
            out = trace_heliostat(
                x_mm,
                y_mm,
                sol.rot_az_deg,
                sol.rot_el_deg,
                sol.c3,
                sol.c4,
                sol.c5,
                row["solar_az_deg"],
                row["solar_el_deg"],
                opt.secondary,
                opt.receiver,
                n_rays,
                rng,
            )
            computed = spot_metrics(out["xy"].T, n_rays, cfg, dni_w_m2=1000.0, efficiency=1.0)

            label = f"heliostat={heliostat_id} step={row['timestep']}"
            assert computed["power_w"] == pytest.approx(row["power_w"], rel=1e-9), label
            assert computed["centroid_x_mm"] == pytest.approx(row["centroid_x_mm"], abs=1e-9), label
            assert computed["centroid_y_mm"] == pytest.approx(row["centroid_y_mm"], abs=1e-9), label
            assert computed["rms_radius_mm"] == pytest.approx(row["rms_radius_mm"], abs=1e-9), label
            assert computed["rays_landed"] == row["rays_landed"], label


# ---------------------------------------------------------------------------
# 2. End-to-end small: fermat field, store round-trip, energy pipeline
# ---------------------------------------------------------------------------


class TestEndToEndSmall:
    def test_store_and_energy_pipeline(self, tmp_path):
        raw = generate("fermat", 25, a_m=4.5, b=0.55)
        field = replace(raw, mirror_width_mm=5000.0, mirror_height_mm=3000.0)

        # NOTE: energy.build_interpolator hard-requires >= 2 distinct
        # declinations (it raises ValueError below that) -- a single traced
        # date cannot feed annual_energy no matter how many timesteps it
        # has, so this uses two dates rather than the one the task brief
        # suggested. hour_step=4h keeps each date to ~3-4 timesteps.
        dates = [_dt.date(2026, 3, 21), _dt.date(2026, 6, 21)]
        store = run_sweep(
            field,
            dates,
            mode="ultra_fast",
            optics="prime_focus",
            workers=1,
            hour_step=4.0,
            out_dir=tmp_path / "run",
            progress=lambda _msg: None,
        )

        # -- store round-trip sanity --------------------------------------
        keys = store.timestep_keys()
        assert len(keys) >= 4
        summary = store.summary()
        assert set(summary["heliostat_id"]) == set(int(i) for i in field.ids)
        assert summary["timestep"].nunique() == len(keys)

        expected_columns = {
            "date",
            "hour",
            "timestep",
            "heliostat_id",
            "x_m",
            "y_m",
            "radius_m",
            "power_w",
            "centroid_x_mm",
            "centroid_y_mm",
            "rms_radius_mm",
            "r50_mm",
            "r90_mm",
            "peak_flux_w_m2",
            "solar_az_deg",
            "solar_el_deg",
            "rot_az_deg",
            "rot_el_deg",
            "aoi_deg",
            "cosine_efficiency",
            "eta_shade",
            "eta_secondary",
            "eta_block",
            "eta_occlusion",
        }
        assert expected_columns <= set(summary.columns)
        assert np.isfinite(summary["power_w"]).all()
        assert (summary["power_w"] >= 0).all()

        manifest = store.manifest
        assert manifest["flux_kind"] == "analytic"
        assert manifest["n_heliostats"] == 25
        assert len(manifest["timesteps"]) == len(keys)

        first_key = keys[0]
        counts = store.read_counts(first_key)
        assert counts.shape == (25, 128, 128)

        # -- energy pipeline: no reference value, just "it runs and is sane"
        provider = dni.ConstantDNI(1000.0)
        result = energy.annual_energy(summary, store.cfg, provider, year=2026, n_heliostats=25)
        assert np.isfinite(result["annual_energy_kwh"])
        assert result["annual_energy_kwh"] > 0.0


# ---------------------------------------------------------------------------
# 3. CLI smoke
# ---------------------------------------------------------------------------


class TestCLISmoke:
    def test_trace_command(self, tmp_path):
        from heliostat.field_layouts import write_field_csv

        field = generate("fermat", 12, a_m=4.5, b=0.55)
        field_csv = tmp_path / "field.csv"
        write_field_csv(field, field_csv)

        out_dir = tmp_path / "cli_run"
        rc = cli_main(
            [
                "trace",
                "--field",
                str(field_csv),
                "--date",
                "2026-03-21",
                "--mode",
                "ultra_fast",
                "--optics",
                "prime_focus",
                "--workers",
                "1",
                "--hour-step",
                "6",
                "-o",
                str(out_dir),
            ]
        )
        assert rc == 0
        store = RunStore(out_dir)
        assert len(store.timestep_keys()) > 0
        assert store.manifest["n_heliostats"] == 12


# ---------------------------------------------------------------------------
# 4. Occlusion columns
# ---------------------------------------------------------------------------


class TestOcclusionColumns:
    def test_low_sun_shading_shows_up_as_eta_occlusion_below_one(self, tmp_path):
        # Two heliostats 20 m apart (radially far from the tower so their
        # normals stay near-horizontal at low sun): the same geometry that
        # showed eta_occlusion ~= 0.35 at sunrise/sunset in manual testing.
        xy_mm = np.array([[0.0, 60000.0], [20000.0, 60000.0]])
        field = HeliostatField(
            x_mm=xy_mm[:, 0],
            y_mm=xy_mm[:, 1],
            ids=np.array([0, 1]),
            mirror_width_mm=5000.0,
            mirror_height_mm=3000.0,
            source="test_sweep.occlusion",
        )
        store = run_sweep(
            field,
            [_dt.date(2026, 3, 21)],
            mode="ultra_fast",
            optics="prime_focus",
            workers=1,
            hour_step=3.0,
            sunrise_margin_min=10.0,
            out_dir=tmp_path / "run",
            progress=lambda _msg: None,
        )
        summary = store.summary()

        low_sun = summary[summary["solar_el_deg"] < 15.0]
        assert len(low_sun) > 0
        assert (low_sun["eta_occlusion"] < 1.0).any()
        assert (low_sun["eta_shade"] < 1.0).any()

        near_noon = summary.loc[summary["solar_el_deg"].idxmax()]
        assert near_noon["eta_occlusion"] == pytest.approx(1.0)
