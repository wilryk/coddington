"""Measure heliostat.sweep.run_sweep's wall time on a realistic field.

600-heliostat fermat field (a=4.5, b=0.55), one full day (2026-03-21,
hour_step=1.0, ~13 timesteps), prime_focus optics. Three configurations:

  * ultra_fast,  workers=1
  * ultra_fast,  workers=cpu_count()
  * monte_carlo, workers=cpu_count(), 20,000 rays/heliostat

The monte_carlo config is deliberately run at 20k rays, not the 120k the
`monte_carlo` TraceMode defaults to -- 120k would blow the ~15 minute
benchmark budget. Its wall time is extrapolated linearly to 120k rays and
reported alongside the measured 20k number (ray count dominates the mc
backend's per-heliostat cost almost linearly -- see the module's own
per-call timing, ~13 ms/call at 20k rays scalar-loop overhead aside).

Writes results to a JSON file (default: this script's directory's
``sweep_benchmark_results.json``) and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

from heliostat.field_layouts import generate
from heliostat.sweep import run_sweep

N_HELIOSTATS = 600
A_M = 4.5
B = 0.55
DATE = "2026-03-21"
HOUR_STEP = 1.0
MC_RAYS_BENCHMARKED = 20_000
MC_RAYS_TARGET = 120_000
#: Judgment call: the task brief says only "the owner's 2-minute target" with
#: no further definition in scope here. Read literally as "one full day
#: (this field, hour_step=1.0) traces in under 2 minutes wall time" --
#: not a per-heliostat-step or per-1000-step rate, since neither of those
#: was stated anywhere available to this script.
TARGET_WALL_TIME_S = 120.0


def _build_field():
    raw = generate("fermat", N_HELIOSTATS, a_m=A_M, b=B)
    return replace(raw, mirror_width_mm=5000.0, mirror_height_mm=3000.0)


def _run_one(field, out_dir, *, mode, workers, n_rays=None):
    import datetime as _dt

    date = _dt.date.fromisoformat(DATE)
    t0 = time.perf_counter()
    store = run_sweep(
        field,
        [date],
        mode=mode,
        optics="prime_focus",
        workers=workers,
        n_rays=n_rays,
        hour_step=HOUR_STEP,
        out_dir=out_dir,
        progress=print,
    )
    elapsed = time.perf_counter() - t0
    n_steps = len(store.timestep_keys())
    heliostat_steps = n_steps * len(field)
    return {
        "mode": mode,
        "workers": workers,
        "n_rays": n_rays,
        "wall_time_s": elapsed,
        "n_timesteps": n_steps,
        "n_heliostats": len(field),
        "heliostat_steps": heliostat_steps,
        "ms_per_heliostat_step": (
            1000.0 * elapsed / heliostat_steps if heliostat_steps else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path(__file__).parent / "sweep_benchmark_results.json"),
        help="JSON results path.",
    )
    parser.add_argument(
        "--run-dir", default=None, help="root dir for the throwaway trace stores (default: temp)."
    )
    args = parser.parse_args()

    import tempfile

    run_root = (
        Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix="sweep_benchmark_"))
    )
    run_root.mkdir(parents=True, exist_ok=True)

    field = _build_field()
    cpu_count = mp.cpu_count()
    print(f"Field: {len(field)} heliostats. CPU count: {cpu_count}.")

    configs = [
        {"mode": "ultra_fast", "workers": 1, "n_rays": None},
        {"mode": "ultra_fast", "workers": cpu_count, "n_rays": None},
        {"mode": "monte_carlo", "workers": cpu_count, "n_rays": MC_RAYS_BENCHMARKED},
    ]

    results = []
    for i, cfg in enumerate(configs):
        rays_suffix = f"_{cfg['n_rays']}rays" if cfg["n_rays"] else ""
        label = f"{cfg['mode']}_workers{cfg['workers']}{rays_suffix}"
        print(f"\n=== [{i + 1}/{len(configs)}] {label} ===")
        out_dir = run_root / label
        row = _run_one(
            field, out_dir, mode=cfg["mode"], workers=cfg["workers"], n_rays=cfg["n_rays"]
        )
        row["label"] = label
        row["meets_2min_target"] = row["wall_time_s"] <= TARGET_WALL_TIME_S
        results.append(row)
        print(
            f"  wall time {row['wall_time_s']:.1f}s, {row['heliostat_steps']} heliostat-steps, "
            f"{row['ms_per_heliostat_step']:.2f} ms/heliostat-step"
        )

    # Extrapolate the monte_carlo row to the mode's real default (120k rays).
    mc_row = next(r for r in results if r["mode"] == "monte_carlo")
    extrapolated = {
        "label": "monte_carlo_workers%d_120000rays_EXTRAPOLATED" % mc_row["workers"],
        "mode": "monte_carlo",
        "workers": mc_row["workers"],
        "n_rays": MC_RAYS_TARGET,
        "wall_time_s": mc_row["wall_time_s"] * (MC_RAYS_TARGET / MC_RAYS_BENCHMARKED),
        "n_timesteps": mc_row["n_timesteps"],
        "n_heliostats": mc_row["n_heliostats"],
        "heliostat_steps": mc_row["heliostat_steps"],
        "ms_per_heliostat_step": (
            mc_row["ms_per_heliostat_step"] * (MC_RAYS_TARGET / MC_RAYS_BENCHMARKED)
        ),
        "extrapolated_linearly_from_20k_rays": True,
    }
    extrapolated["meets_2min_target"] = extrapolated["wall_time_s"] <= TARGET_WALL_TIME_S
    results.append(extrapolated)

    print("\n\n=== Summary ===")
    header = (
        f"{'mode':<14} {'workers':>7} {'rays':>8} {'wall_s':>9} "
        f"{'ms/h-step':>10} {'<=2min target':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        rays = r["n_rays"] if r["n_rays"] else "-"
        extrap = " (extrap)" if r.get("extrapolated_linearly_from_20k_rays") else ""
        print(
            f"{r['mode']:<14} {r['workers']:>7} {str(rays):>8} {r['wall_time_s']:>9.1f} "
            f"{r['ms_per_heliostat_step']:>10.2f} {str(r['meets_2min_target']):>14}{extrap}"
        )

    out_path = Path(args.output)
    payload = {"cpu_count": cpu_count, "n_heliostats": len(field), "results": results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
