"""Sampling-sparsity sweep: how few mirror-surface samples can the cone
tracer take before field-total accuracy degrades?

Motivation (see docs/ui-spec-v0.2.md and REPORT.md SS0-1 for the shared
machinery this reuses): ``trace_heliostat_cone``'s mirror-surface sampling
grid is a hardcoded 20x12 -- an aspect ratio baked to the manuscript's
5m x 3m rectangular mirror. This script instead derives the grid from a
requested sampling DENSITY (samples/m^2) applied to the mirror's own
aperture bounding box, with the grid's aspect ratio matched to that bbox,
and sweeps that density from very sparse to past the current hardcoded
resolution to find the sparsest rung that still holds field-total accuracy.

Everything here traces through the exact same ``sampling.py`` +
``binned.py`` path the rest of ``scripts/coeff_prototype/`` uses --
``trace_heliostat_samples`` (bit-for-bit faithful to
``trace_heliostat_cone``, see REPORT.md SS1) followed by ``deposit_binned``
(``kernels.deposit``, unmodified, ground truth). Nothing under ``src/`` is
touched. Deposit method is held fixed at binned throughout -- this sweep is
about sampling density, not about comparing deposit methods (see
``run_benchmark.py`` for that).

Reference = the density rung whose (n_x, n_y) reproduces the current
hardcoded 20x12 grid exactly: on the manuscript's 5m x 3m mirror,
5*sqrt(16)=20 and 3*sqrt(16)=12, so density=16.0 samples/m^2 IS the
production ("fast_accurate": order=2, grid=(20,12), mask_nodes=16) grid --
no separate reference run is needed, it is simply one rung of the ladder.

Usage::

    .venv\\Scripts\\python.exe scripts/coeff_prototype/sparsity_sweep.py [--quick] [--max-rung-s N]

``--quick`` subsamples the default field (every 20th heliostat) for
iterating on this script itself -- the real gate numbers need a full run.
``--max-rung-s`` stops the ladder (sparsest-first) if a rung's wall time
exceeds the budget, so a partial run still tells the accuracy story instead
of hanging on an expensive dense rung.

Writes ``scripts/coeff_prototype/sparsity_sweep_results.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT / "src", _ROOT / "tests", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from coeff_prototype import scenarios
from coeff_prototype.binned import deposit_binned
from coeff_prototype.sampling import flux_grid_edges, trace_heliostat_samples
from heliostat.trace.cone import sunshape_kernel

OUT_JSON = Path(__file__).parent / "sparsity_sweep_results.json"
KERNEL = sunshape_kernel("super_gauss")  # shared with run_benchmark.py's ULTRA_FAST-matching kernel

MIRROR_WIDTH_M = scenarios.MIRROR_WIDTH_MM / 1000.0  # 5.0 -- manuscript mirror aperture bbox
MIRROR_HEIGHT_M = scenarios.MIRROR_HEIGHT_MM / 1000.0  # 3.0

#: Reference density: exactly reproduces the hardcoded 20x12 grid on the
#: manuscript's 5x3 m mirror (5*sqrt(16)=20, 3*sqrt(16)=12).
REFERENCE_DENSITY = 16.0

#: 8 rungs, sparsest first (run_sweep() traces in this order so a partial
#: run under a wall-time budget still tells the accuracy story), spanning
#: ~0.5 samples/m^2 up to and past the reference 16 samples/m^2.
DENSITY_LADDER = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, REFERENCE_DENSITY, 24.0]

ORDER = 2  # "fast_accurate" (heliostat.trace.modes.FAST_ACCURATE)
MASK_NODES = 16  # "fast_accurate"


def grid_for_density(density: float, width_m: float, height_m: float, min_n: int = 2) -> tuple[int, int]:
    """``(n_x, n_y)`` for ``density`` samples/m^2 over a ``width_m x
    height_m`` aperture bbox, aspect ratio matched to the bbox.

    Derivation: requiring both ``n_x * n_y == density * width_m * height_m``
    (the requested sample count) and ``n_x / n_y == width_m / height_m``
    (aspect matched to the bbox) simultaneously gives, by substitution,
    ``n_x == width_m * sqrt(density)`` and ``n_y == height_m * sqrt(density)``
    -- each axis is independently its own physical length times
    ``sqrt(density)``, then rounded. (Sanity check: on the manuscript's
    5m x 3m mirror at density=16, this gives exactly (20, 12), the current
    hardcoded grid.)
    """
    s = math.sqrt(density)
    n_x = max(min_n, round(width_m * s))
    n_y = max(min_n, round(height_m * s))
    return n_x, n_y


def trace_field_at_grid(cases, secondary, receiver, u_edges, v_edges, grid) -> dict:
    """Trace every case in ``cases`` at sampling ``grid`` (n_x, n_y),
    order=2/mask_nodes=16/binned deposit (matching ``fast_accurate``) --
    returns per-heliostat power, field-total power/peak, and wall time.

    Per-heliostat power is obtained by depositing each bundle onto its own
    scratch grid before adding it into the shared field accumulator, so it
    costs one extra array add per heliostat and no extra tracing.
    """
    n_v, n_u = v_edges.size - 1, u_edges.size - 1
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    field_out = np.zeros((n_v, n_u))
    per_helio_power = np.empty(len(cases))
    t0 = time.perf_counter()
    for i, case in enumerate(cases):
        bundle = trace_heliostat_samples(
            case.x_mm, case.y_mm, case.rot_az_deg, case.rot_el_deg,
            case.c3, case.c4, case.c5, case.solar_az_deg, case.solar_el_deg,
            secondary, receiver, KERNEL,
            grid=grid, order=ORDER, mask_nodes=MASK_NODES, occluders=case.occluders,
        )
        tmp = np.zeros((n_v, n_u))
        deposit_binned(bundle, u_edges, v_edges, out=tmp)
        per_helio_power[i] = float(tmp.sum() * du * dv)
        field_out += tmp
    wall_s = time.perf_counter() - t0
    total_power_w = float(field_out.sum() * du * dv)
    peak_flux_w_m2 = float(field_out.max() * 1.0e6)
    return {
        "grid": list(grid),
        "wall_time_s": wall_s,
        "field_total_power_w": total_power_w,
        "peak_flux_w_m2": peak_flux_w_m2,
        "per_heliostat_power_w": per_helio_power.tolist(),
        "heliostat_ids": [c.heliostat_id for c in cases],
    }


def run_sweep(densities=DENSITY_LADDER, quick: bool = False, max_rung_s: float | None = None) -> dict:
    scenario = scenarios.scenario_default_field()
    cases = scenario.cases
    if quick:
        cases = cases[::20]
    u_edges, v_edges = flux_grid_edges(scenario.receiver, scenario.flux_grid)

    rungs = []
    skipped = []
    for idx, density in enumerate(densities):
        grid = grid_for_density(density, MIRROR_WIDTH_M, MIRROR_HEIGHT_M)
        n_x, n_y = grid
        samples_per_m2 = (n_x * n_y) / (MIRROR_WIDTH_M * MIRROR_HEIGHT_M)
        print(
            f"=== rung {idx + 1}/{len(densities)}: density={density:g} samples/m^2 -> "
            f"grid=({n_x},{n_y}) (actual {samples_per_m2:.3f}/m^2), n={len(cases)} heliostats ===",
            flush=True,
        )
        result = trace_field_at_grid(cases, scenario.secondary, scenario.receiver, u_edges, v_edges, grid)
        result["density_target"] = density
        result["density_actual"] = samples_per_m2
        print(
            f"    wall={result['wall_time_s']:.1f}s field_total={result['field_total_power_w']:.1f}W "
            f"peak={result['peak_flux_w_m2']:.1f}W/m^2",
            flush=True,
        )
        rungs.append(result)
        if max_rung_s is not None and result["wall_time_s"] > max_rung_s:
            print(f"    (exceeded {max_rung_s}s budget -- stopping ladder here)", flush=True)
            done = {r["density_target"] for r in rungs}
            skipped.extend(d for d in densities if d not in done)
            break

    return {
        "scenario_notes": scenario.notes,
        "n_heliostats": len(cases),
        "order": ORDER,
        "mask_nodes": MASK_NODES,
        "mirror_width_m": MIRROR_WIDTH_M,
        "mirror_height_m": MIRROR_HEIGHT_M,
        "reference_density": REFERENCE_DENSITY,
        "rungs": rungs,
        "skipped_densities": skipped,
    }


def summarize(sweep: dict) -> list[dict]:
    """Per-rung field-total/max-per-heliostat/peak-flux error (%) vs the
    reference-density rung, and speedup. Returns [] if the reference rung
    itself was skipped (e.g. a tight ``--max-rung-s`` stopped the ladder
    before reaching it)."""
    rungs = sweep["rungs"]
    ref = next((r for r in rungs if r["density_target"] == sweep["reference_density"]), None)
    if ref is None:
        return []
    ref_power = np.array(ref["per_heliostat_power_w"])
    ref_ids = ref["heliostat_ids"]
    summary = []
    for r in rungs:
        assert r["heliostat_ids"] == ref_ids, "heliostat id order mismatch between rungs"
        power = np.array(r["per_heliostat_power_w"])
        per_helio_err_pct = np.abs(power - ref_power) / np.abs(ref_power) * 100.0
        field_err_pct = (
            abs(r["field_total_power_w"] - ref["field_total_power_w"]) / ref["field_total_power_w"] * 100.0
        )
        peak_err_pct = abs(r["peak_flux_w_m2"] - ref["peak_flux_w_m2"]) / ref["peak_flux_w_m2"] * 100.0
        speedup = ref["wall_time_s"] / r["wall_time_s"] if r["wall_time_s"] > 0 else float("nan")
        summary.append(
            {
                "density_target": r["density_target"],
                "grid": r["grid"],
                "field_total_error_pct": field_err_pct,
                "max_per_heliostat_error_pct": float(per_helio_err_pct.max()),
                "peak_flux_error_pct": peak_err_pct,
                "speedup_vs_reference": speedup,
                "wall_time_s": r["wall_time_s"],
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="subsample the field for iteration")
    parser.add_argument(
        "--max-rung-s", type=float, default=None, help="stop the ladder if a rung exceeds this many seconds"
    )
    args = parser.parse_args()

    sweep = run_sweep(quick=args.quick, max_rung_s=args.max_rung_s)
    sweep["summary"] = summarize(sweep)

    OUT_JSON.write_text(json.dumps(sweep, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")

    print(
        f"\n{'density':>10} {'grid':>10} {'field_err%':>12} {'max_helio_err%':>16} "
        f"{'peak_err%':>10} {'speedup':>9}"
    )
    for s in sweep["summary"]:
        print(
            f"{s['density_target']:>10.2f} {str(tuple(s['grid'])):>10} {s['field_total_error_pct']:>12.4f} "
            f"{s['max_per_heliostat_error_pct']:>16.4f} {s['peak_flux_error_pct']:>10.4f} "
            f"{s['speedup_vs_reference']:>9.2f}"
        )
    if sweep["skipped_densities"]:
        print(f"\nSkipped (exceeded wall-time budget): {sweep['skipped_densities']}")


if __name__ == "__main__":
    main()
