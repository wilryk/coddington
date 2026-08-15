"""Benchmark and visualize the cone-optics backend against Monte Carlo.

Runnable script, not a pytest test. For one heliostat per optical
configuration (prime_focus / axicon / cassegrain), this:

* times ``trace_heliostat_cone`` at its default grid against
  ``trace_heliostat`` (Monte Carlo) at 20k / 120k / 1M rays;
* treats the 1M-ray MC trace as a high-fidelity reference and measures the
  cone map's relative residual against it;
* calibrates how MC map noise scales with ray count from two independent
  20k-ray draws (noise ~ 1/sqrt(N)), then solves for the MC ray count whose
  own shot noise would match the cone-vs-high-fidelity residual -- a rough
  answer to "how many MC rays does the cone backend's zero-shot-noise map
  stand in for";
* writes a 3-panel PNG (MC 20k map, cone map, |difference|) per config.

Usage::

    python scripts/cone_benchmark.py [--out DIR]

Defaults to a temp directory if ``--out`` is omitted.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: must precede the pyplot import below
import matplotlib.pyplot as plt  # noqa: E402

from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone  # noqa: E402
from heliostat.trace.mc import trace_heliostat  # noqa: E402
from test_mc_parity import _geometry_for, _load_fixture  # noqa: E402

CONFIGS = ["prime_focus", "axicon", "cassegrain"]
HELIOSTAT_ID = 574  # widest fixture spot of the 5 fixture heliostats
STEP_KEY = "20260321_0939"  # mid-morning sun position
MC_RAY_COUNTS = [20_000, 120_000, 1_000_000]
HIGH_FIDELITY_N = 1_000_000
FLUX_GRID = (100, 100)
KERNEL = sunshape_kernel("super_gauss")


def _mc_flux_map(xy: np.ndarray, watts_per_ray: float, u_edges: np.ndarray, v_edges: np.ndarray):
    """Histogram MC receiver hits onto (u_edges, v_edges), in W/m^2."""
    counts, _, _ = np.histogram2d(xy[0], xy[1], bins=[u_edges, v_edges])
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    bin_area_m2 = (du * dv) / 1.0e6
    return counts.T * watts_per_ray / bin_area_m2  # (n_v, n_u)


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS(a - b) over bins where either map carries signal, / peak."""
    mask = (a > 0.02 * a.max()) | (b > 0.02 * b.max())
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)) / max(a.max(), b.max()))


def run_config(config: str, out_dir: Path) -> dict:
    d, _, summary = _load_fixture(config)
    row = summary.loc[(HELIOSTAT_ID, STEP_KEY)]
    secondary, receiver = _geometry_for(config)

    print(f"\n=== {config}  heliostat {HELIOSTAT_ID}  {STEP_KEY} ===")

    t0 = time.perf_counter()
    cone = trace_heliostat_cone(
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
        KERNEL,
        flux_grid=FLUX_GRID,
    )
    t_cone = time.perf_counter() - t0
    u_edges, v_edges = cone["u_edges"], cone["v_edges"]
    cone_flux = cone["flux"]
    print(f"cone (default sample grid):  {t_cone * 1000:8.1f} ms  power={cone['power_w']:.1f} W")

    mc_maps: dict[int, np.ndarray] = {}
    mc_times: dict[int, float] = {}
    for i, n in enumerate(MC_RAY_COUNTS):
        rng = np.random.default_rng(1000 + i)  # independent seed per ray count
        t0 = time.perf_counter()
        mc = trace_heliostat(
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
            n,
            rng,
        )
        dt = time.perf_counter() - t0
        mc_times[n] = dt
        mc_maps[n] = _mc_flux_map(mc["xy"], mc["watts_per_ray"], u_edges, v_edges)
        print(f"MC {n:>9,} rays:             {dt * 1000:8.1f} ms  landed={mc['xy'].shape[1]}")

    # A second, independent 20k-ray MC trace, purely to calibrate how map
    # noise scales with ray count: two independent maps at the same N differ
    # only by shot noise, and that noise ~ 1/sqrt(N).
    rng2 = np.random.default_rng(2000)
    mc_20k_b = trace_heliostat(
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
        MC_RAY_COUNTS[0],
        rng2,
    )
    mc_20k_b_flux = _mc_flux_map(mc_20k_b["xy"], mc_20k_b["watts_per_ray"], u_edges, v_edges)

    noise_20k = _relative_rms(mc_maps[MC_RAY_COUNTS[0]], mc_20k_b_flux)
    cone_residual = _relative_rms(cone_flux, mc_maps[HIGH_FIDELITY_N])

    # noise(N) ~= C / sqrt(N)  =>  C = noise_20k * sqrt(20k); solve for the N
    # whose own noise(N) equals the cone's residual against the 1M reference.
    calib_c = noise_20k * np.sqrt(MC_RAY_COUNTS[0])
    n_equivalent = (calib_c / cone_residual) ** 2 if cone_residual > 0 else float("inf")

    print(f"cone vs {HIGH_FIDELITY_N:,}-ray MC residual:  {cone_residual * 100:7.3f}% of peak")
    print(f"MC map noise at 20k (2 independent draws): {noise_20k * 100:7.3f}% of peak")
    print(f"MC ray count with matching map noise:      ~{n_equivalent:,.0f} rays")

    # --- figure: MC 20k, cone, |difference| -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmax = max(mc_maps[MC_RAY_COUNTS[0]].max(), cone_flux.max())
    extent = [u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]]

    im0 = axes[0].imshow(
        mc_maps[MC_RAY_COUNTS[0]], origin="lower", extent=extent, vmax=vmax, cmap="inferno"
    )
    axes[0].set_title(f"MC, {MC_RAY_COUNTS[0]:,} rays")
    im1 = axes[1].imshow(cone_flux, origin="lower", extent=extent, vmax=vmax, cmap="inferno")
    axes[1].set_title("Cone (default grid)")
    diff = np.abs(cone_flux - mc_maps[MC_RAY_COUNTS[0]])
    im2 = axes[2].imshow(diff, origin="lower", extent=extent, cmap="inferno")
    axes[2].set_title("|difference|")

    for ax in axes:
        ax.set_xlabel("u (mm)")
    axes[0].set_ylabel("v (mm)")
    fig.colorbar(im0, ax=axes[0], label="W/m^2")
    fig.colorbar(im1, ax=axes[1], label="W/m^2")
    fig.colorbar(im2, ax=axes[2], label="W/m^2")
    fig.suptitle(f"{config}: cone vs MC, heliostat {HELIOSTAT_ID}, {STEP_KEY}")

    out_path = out_dir / f"cone_benchmark_{config}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    return {
        "config": config,
        "t_cone_ms": t_cone * 1000,
        "mc_times_ms": {n: t * 1000 for n, t in mc_times.items()},
        "cone_residual_pct": cone_residual * 100,
        "noise_20k_pct": noise_20k * 100,
        "n_equivalent": n_equivalent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory for PNGs (default: a temp directory)",
    )
    args = parser.parse_args()
    out_dir = args.out or Path(tempfile.gettempdir()) / "heliostat_cone_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_config(config, out_dir) for config in CONFIGS]

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['config']:12s} cone={r['t_cone_ms']:7.1f}ms  "
            f"MC20k={r['mc_times_ms'][20_000]:7.1f}ms  "
            f"MC120k={r['mc_times_ms'][120_000]:8.1f}ms  "
            f"MC1M={r['mc_times_ms'][1_000_000]:9.1f}ms  "
            f"residual={r['cone_residual_pct']:6.3f}%  "
            f"equivalent_N={r['n_equivalent']:,.0f}"
        )
    print(f"\nPNGs written to {out_dir}")


if __name__ == "__main__":
    main()
