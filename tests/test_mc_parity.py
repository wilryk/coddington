"""Golden-fixture parity test for the Monte Carlo tracer.

``tests/fixtures/mc_parity/<config>/`` holds five heliostats x three sun
positions x three optical layouts (prime focus, axicon, Cassegrain), traced
with 20,000 rays each and stored as: a per-trace loss-chain ``counters.json``,
quantised int16 receiver rays (``rays_<id>_<step>.npy``), and a
``summary.csv`` of the pointing/figure inputs and derived spot metrics --
everything needed to reconstruct the exact trace call, listed in
``tests/fixtures/provenance.json``.

The gate is three-part, per fixture: the loss-chain counters, the quantised
receiver rays, and :func:`heliostat.metrics.spot_metrics` recomputed from
the reproduced rays. On the machine the fixtures were generated on (same
numpy version, recorded in provenance.json) Monte Carlo sampling with
``numpy.random.default_rng`` is bit-for-bit reproducible given the same seed
sequence, so all three assertions are exact. On any other platform the test
degrades to statistical checks: counters within 0.1% of the emitted ray
count, and the receiver distribution consistent with the fixture within its
own Monte Carlo noise (3-sigma centroid/RMS, a binned chi-squared test).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from heliostat import metrics
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import AxiconSecondary, CassegrainSecondary, NoSecondary
from heliostat.trace.mc import trace_heliostat
from heliostat.trace.samplers import SuperGaussSampler

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
MC_ROOT = FIXTURES_ROOT / "mc_parity"

PROVENANCE = json.loads((FIXTURES_ROOT / "provenance.json").read_text())

BASE_SEED = PROVENANCE["base_seed"]
N_RAYS = PROVENANCE["mc_parity"]["n_rays_per_trace"]
WINDOW_MM = 2000.0
QUANT_SCALE_MM = PROVENANCE["quantisation_scale_mm"]
INT16_MAX = 32767

# Exact assertions hold only where the ray-generation order and floating-point
# behaviour of numpy match the platform the fixtures were generated on.
SAME_PLATFORM = np.__version__ == PROVENANCE["numpy_version"]

# Mirror reflectivity 0.9 per bounce; prime focus has one bounce off the
# primary only, axicon and cassegrain have a second bounce off the secondary.
# Matches the standard-paper-geometry overrides the fixtures were traced at.
THROUGHPUT = {"prime_focus": 0.9, "axicon": 0.81, "cassegrain": 0.81}

CONFIGS = ["prime_focus", "axicon", "cassegrain"]


def _quantise(xy_mm: np.ndarray, window_mm: float = WINDOW_MM) -> np.ndarray:
    """Same int16 quantisation the fixtures were stored with."""
    scaled = np.clip(xy_mm / window_mm, -1.0, 1.0) * INT16_MAX
    return np.rint(scaled).astype(np.int16)


def _make_cfg(
    throughput: float, power_w: float = 38484.5, window_mm: float = WINDOW_MM, grid_size: int = 128
):
    """Minimal duck-typed cfg -- spot_metrics only reads these fields."""
    bin_size_mm = 2.0 * window_mm / grid_size
    receiver = SimpleNamespace(
        window_mm=window_mm,
        grid_size=grid_size,
        bin_size_mm=bin_size_mm,
        bin_area_m2=(bin_size_mm / 1000.0) ** 2,
        edges=np.linspace(-window_mm, window_mm, grid_size + 1),
    )

    def _watts_per_ray(n):
        return power_w / n

    return SimpleNamespace(
        receiver=receiver,
        source=SimpleNamespace(watts_per_ray=_watts_per_ray),
        optics=SimpleNamespace(throughput=throughput),
    )


def _geometry_for(config: str):
    """(secondary, receiver) at the standard paper geometry each fixture
    config was traced with -- see ``tests/fixtures/provenance.json`` and each
    config's own ``summary.csv`` for the traced pointing that resulted."""
    if config == "prime_focus":
        # Receiver at the shared focus, above the mirrors: rays arrive from
        # below, so the window faces down.
        secondary = NoSecondary()
        receiver = FlatWindowReceiver(
            z_mm=35335.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="down"
        )
    elif config == "axicon":
        # Beam-down ground receiver below the cone: rays arrive from above.
        secondary = AxiconSecondary(
            apex_height_mm=27000.0, half_angle_deg=20.0, aperture_radius_mm=14000.0
        )
        receiver = FlatWindowReceiver(
            z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
    elif config == "cassegrain":
        # Beam-down ground receiver below the hyperboloid, same side as the
        # axicon layout. Vertex/radius/conic are the built secondary's fixed
        # constants, independent of the shared-focus height.
        secondary = CassegrainSecondary(
            vertex_z_mm=26993.999446877,
            vertex_radius_mm=26112.078893738,
            conic=-5.317616535,
            aperture_radius_mm=14000.0,
        )
        receiver = FlatWindowReceiver(
            z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
    else:
        raise ValueError(f"unknown config {config!r}")
    return secondary, receiver


def _load_fixture(config: str):
    d = MC_ROOT / config
    counters = {
        (row["heliostat_id"], row["step_key"]): row
        for row in json.loads((d / "counters.json").read_text())
    }
    summary = pd.read_csv(d / "summary.csv").set_index(["heliostat_id", "step_key"])
    return d, counters, summary


def _cases():
    cases = []
    for config in CONFIGS:
        _, counters, _ = _load_fixture(config)
        for heliostat_id, step_key in counters:
            cases.append((config, heliostat_id, step_key))
    return cases


def _assert_statistically_consistent(
    xy_mm: np.ndarray, fixture_mm: np.ndarray, label: str, n_bins: int = 16
) -> None:
    """Fallback used off the pinned numpy version: same distribution within
    Monte Carlo noise, not bit-identical rays."""
    n_new, n_ref = xy_mm.shape[0], fixture_mm.shape[0]
    if n_ref == 0 or n_new == 0:
        assert abs(n_new - n_ref) <= max(1, int(0.02 * max(n_ref, 1))), label
        return

    c_new, c_ref = xy_mm.mean(axis=0), fixture_mm.mean(axis=0)
    r_new = np.hypot(xy_mm[:, 0] - c_new[0], xy_mm[:, 1] - c_new[1])
    r_ref = np.hypot(fixture_mm[:, 0] - c_ref[0], fixture_mm[:, 1] - c_ref[1])
    rms_new, rms_ref = np.sqrt(np.mean(r_new**2)), np.sqrt(np.mean(r_ref**2))

    se_centroid = rms_ref / np.sqrt(n_ref)
    assert abs(c_new[0] - c_ref[0]) < 3.0 * se_centroid, f"{label} centroid_x"
    assert abs(c_new[1] - c_ref[1]) < 3.0 * se_centroid, f"{label} centroid_y"

    se_rms = rms_ref / np.sqrt(2.0 * n_ref)
    assert abs(rms_new - rms_ref) < 3.0 * se_rms + 1.0, f"{label} rms_radius"

    edges = np.linspace(0.0, max(r_ref.max(), r_new.max()) * 1.001, n_bins + 1)
    obs, _ = np.histogram(r_new, bins=edges)
    exp, _ = np.histogram(r_ref, bins=edges)
    exp_scaled = exp * (obs.sum() / max(exp.sum(), 1))
    mask = exp_scaled > 5
    if mask.sum() >= 2:
        from scipy import stats

        chi2 = float(np.sum((obs[mask] - exp_scaled[mask]) ** 2 / exp_scaled[mask]))
        dof = int(mask.sum()) - 1
        pval = float(1.0 - stats.chi2.cdf(chi2, dof))
        assert pval > 1e-4, f"{label} chi2={chi2:.1f} dof={dof} pval={pval:.2e}"


POWER_INDEPENDENT_COLUMNS = [
    "rays_emitted",
    "rays_landed",
    "transmission",
    "centroid_x_mm",
    "centroid_y_mm",
    "rms_radius_mm",
    "r50_mm",
    "r90_mm",
]


@pytest.mark.parametrize("config,heliostat_id,step_key", _cases())
def test_mc_parity(config, heliostat_id, step_key):
    d, counters_by_key, summary = _load_fixture(config)
    row = summary.loc[(heliostat_id, step_key)]
    expected_counters = counters_by_key[(heliostat_id, step_key)]
    label = f"{config} heliostat={heliostat_id} step={step_key}"

    secondary, receiver = _geometry_for(config)
    sampler = SuperGaussSampler()  # provenance: source_model == "super_gauss"

    step_int = int(step_key.replace("_", ""))
    rng = np.random.default_rng(np.random.SeedSequence((BASE_SEED, step_int, heliostat_id)))

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
        N_RAYS,
        rng,
        sampler=sampler,
    )

    # --- (a) loss-chain counters -------------------------------------
    counter_keys = [
        "emitted",
        "hit_mirror",
        "tip_rays",
        "hit_secondary",
        "reached_receiver",
        "in_window",
    ]
    if config == "axicon":
        counter_keys.append("hit_cone")

    if SAME_PLATFORM:
        for k in counter_keys:
            assert out["counters"][k] == expected_counters[k], f"{label} counters[{k}]"
    else:
        emitted = expected_counters["emitted"]
        tol = max(1, int(0.001 * emitted))
        for k in counter_keys:
            assert abs(out["counters"][k] - expected_counters[k]) <= tol, f"{label} counters[{k}]"

    # --- (b) quantised receiver rays -----------------------------------
    xy_mm = out["xy"].T  # (K, 2)
    fixture_rays = np.load(d / f"rays_{heliostat_id}_{step_key}.npy")

    if SAME_PLATFORM:
        quant = _quantise(xy_mm)
        assert quant.shape == fixture_rays.shape, label
        assert np.array_equal(quant, fixture_rays), label
    else:
        fixture_mm = fixture_rays.astype(np.float64) * QUANT_SCALE_MM
        _assert_statistically_consistent(xy_mm, fixture_mm, label)

    # --- (c) spot metrics -------------------------------------------------
    cfg = _make_cfg(THROUGHPUT[config])
    computed = metrics.spot_metrics(xy_mm, N_RAYS, cfg, efficiency=1.0)

    if SAME_PLATFORM:
        for col in POWER_INDEPENDENT_COLUMNS:
            assert computed[col] == pytest.approx(row[col], rel=1e-9, abs=1e-9), f"{label} {col}"
        assert computed["power_w"] == pytest.approx(row["power_w"], rel=1e-9), f"{label} power_w"
        assert computed["peak_flux_w_m2"] == pytest.approx(row["peak_flux_w_m2"], rel=1e-9), (
            f"{label} peak_flux_w_m2"
        )
    else:
        for col in POWER_INDEPENDENT_COLUMNS:
            if col in ("rays_emitted", "rays_landed"):
                continue
            assert computed[col] == pytest.approx(row[col], rel=0.05, abs=5.0), f"{label} {col}"
