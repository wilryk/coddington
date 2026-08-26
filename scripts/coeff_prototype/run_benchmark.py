"""B2 benchmark driver: binned vs Hermite-Gauss vs tensor-B-spline deposit,
on the same traced samples, across the four scenarios in ``scenarios.py``.

Usage::

    .venv\\Scripts\\python.exe scripts/coeff_prototype/run_benchmark.py [--quick] [--skip-mc]

``--quick`` subsamples the default field to a few dozen heliostats (for
iterating on this script itself); the real gate numbers need a full run.
``--skip-mc`` skips the Monte-Carlo reference (it is the slowest single
step) -- useful for re-running just the deposit-method comparison.

Writes ``scripts/coeff_prototype/benchmark_results.json`` (every number this
script computed) and prints a summary. ``REPORT.md`` is written separately,
by hand, from these numbers -- see that file for the actual writeup.
"""

from __future__ import annotations

import argparse
import json
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
from coeff_prototype.bspline import (
    DEFAULT_CONTROL_GRID,
    control_grid_edges,
    evaluate_bspline,
)
from coeff_prototype.hermite import DEFAULT_ORDER, HermiteBasis, accumulate_hermite, evaluate_hermite
from coeff_prototype.sampling import flux_grid_edges, trace_heliostat_samples
from heliostat.trace.cone import sunshape_kernel
from heliostat.trace.mc import SOURCE_POWER_W, trace_heliostat

MC_SEED_BASE = 424242
OUT_JSON = Path(__file__).parent / "benchmark_results.json"
KERNEL = sunshape_kernel("super_gauss")  # shared for every scenario, matching ULTRA_FAST's own


def _sample_scenario(scenario, order=1) -> tuple[list, list]:
    """Trace every case's samples once. Returns (bundles, sample_gen_times_s)."""
    bundles = []
    times = []
    for case in scenario.cases:
        t0 = time.perf_counter()
        bundle = trace_heliostat_samples(
            case.x_mm, case.y_mm, case.rot_az_deg, case.rot_el_deg,
            case.c3, case.c4, case.c5, case.solar_az_deg, case.solar_el_deg,
            scenario.secondary, scenario.receiver, KERNEL,
            grid=scenario.cone_kwargs.get("grid", (20, 12)), order=order,
            mask_nodes=scenario.cone_kwargs.get("mask_nodes", 16), occluders=case.occluders,
        )
        times.append(time.perf_counter() - t0)
        bundles.append(bundle)
    return bundles, times


def _binned_field(bundles, u_edges, v_edges):
    out = np.zeros((v_edges.size - 1, u_edges.size - 1))
    times = []
    for b in bundles:
        t0 = time.perf_counter()
        deposit_binned(b, u_edges, v_edges, out=out)
        times.append(time.perf_counter() - t0)
    return out, times


def _hermite_field(bundles, basis, u_edges, v_edges, wrap_u):
    all_records = []
    all_fallback = []
    accum_times = []
    for b in bundles:
        t0 = time.perf_counter()
        records, fallback = accumulate_hermite(b, basis)
        accum_times.append(time.perf_counter() - t0)
        all_records.extend(records)
        all_fallback.extend(fallback)
    t0 = time.perf_counter()
    out = evaluate_hermite(all_records, all_fallback, basis, u_edges, v_edges, wrap_u)
    eval_time = time.perf_counter() - t0
    return out, accum_times, eval_time


def _bspline_field(bundles, u_edges_coarse, v_edges_coarse, u_edges_fine, v_edges_fine, wrap_u):
    coarse = np.zeros((v_edges_coarse.size - 1, u_edges_coarse.size - 1))
    accum_times = []
    for b in bundles:
        t0 = time.perf_counter()
        deposit_binned(b, u_edges_coarse, v_edges_coarse, out=coarse)  # the accumulate step
        accum_times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    out = evaluate_bspline(coarse, u_edges_coarse, v_edges_coarse, u_edges_fine, v_edges_fine, wrap_u)
    eval_time = time.perf_counter() - t0
    return out, accum_times, eval_time


def _metrics(out_w_per_mm2, u_edges, v_edges, incident_power_w):
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    flux_w_m2 = out_w_per_mm2 * 1.0e6
    total_power_w = float(out_w_per_mm2.sum() * du * dv)
    peak_flux = float(flux_w_m2.max())
    intercept = total_power_w / incident_power_w if incident_power_w > 0 else float("nan")
    return {"total_power_w": total_power_w, "peak_flux_w_m2": peak_flux, "intercept_efficiency": intercept}


def _map_error_vs_binned(method_out, binned_out):
    diff = np.abs(method_out - binned_out) * 1.0e6  # W/m^2
    peak = binned_out.max() * 1.0e6
    return {
        "max_diff_pct_of_peak": float(diff.max() / peak * 100.0) if peak > 0 else float("nan"),
        "rms_diff_pct_of_peak": float(np.sqrt(np.mean(diff**2)) / peak * 100.0) if peak > 0 else float("nan"),
    }


def run_deposit_comparison(scenario, order=1, control_grid=DEFAULT_CONTROL_GRID, hermite_order=DEFAULT_ORDER):
    u_edges, v_edges = flux_grid_edges(scenario.receiver, scenario.flux_grid)
    u_edges_c, v_edges_c = control_grid_edges(u_edges, v_edges, control_grid)

    bundles, sample_gen_times = _sample_scenario(scenario, order=order)
    incident_power_w = float(sum(float(b.weights.sum()) for b in bundles))
    t_sample_total = sum(sample_gen_times)

    out_binned, t_binned_accum = _binned_field(bundles, u_edges, v_edges)
    m_binned = _metrics(out_binned, u_edges, v_edges, incident_power_w)

    basis = HermiteBasis.build(KERNEL, order=hermite_order)
    out_hermite, t_hermite_accum, t_hermite_eval = _hermite_field(bundles, basis, u_edges, v_edges, scenario.wrap_u)
    m_hermite = _metrics(out_hermite, u_edges, v_edges, incident_power_w)
    err_hermite = _map_error_vs_binned(out_hermite, out_binned)

    out_bspline, t_bspline_accum, t_bspline_eval = _bspline_field(
        bundles, u_edges_c, v_edges_c, u_edges, v_edges, scenario.wrap_u
    )
    m_bspline = _metrics(out_bspline, u_edges, v_edges, incident_power_w)
    err_bspline = _map_error_vs_binned(out_bspline, out_binned)

    def _power_conservation(m):
        return abs(m["total_power_w"] - m_binned["total_power_w"]) / m_binned["total_power_w"] * 100.0

    ring_radii = [c.ring_radius_m for c in scenario.cases]

    result = {
        "scenario": scenario.name,
        "notes": scenario.notes,
        "n_heliostats": len(scenario.cases),
        "incident_power_w": incident_power_w,
        "wall_time_s": {
            "sample_generation_total": t_sample_total,
            "binned_accumulate_total": sum(t_binned_accum),
            "hermite_accumulate_total": sum(t_hermite_accum),
            "hermite_evaluate_total": t_hermite_eval,
            "bspline_accumulate_total": sum(t_bspline_accum),
            "bspline_evaluate_total": t_bspline_eval,
            "binned_total": t_sample_total + sum(t_binned_accum),
            "hermite_total": t_sample_total + sum(t_hermite_accum) + t_hermite_eval,
            "bspline_total": t_sample_total + sum(t_bspline_accum) + t_bspline_eval,
        },
        "deposit_time_per_heliostat_vs_ring": {
            "ring_radius_m": ring_radii,
            "binned_s": t_binned_accum,
            "hermite_s": t_hermite_accum,
            "bspline_s": t_bspline_accum,
        },
        "metrics": {"binned": m_binned, "hermite": m_hermite, "bspline": m_bspline},
        "power_conservation_pct_vs_binned": {
            "hermite": _power_conservation(m_hermite),
            "bspline": _power_conservation(m_bspline),
        },
        "map_error_vs_binned": {"hermite": err_hermite, "bspline": err_bspline},
    }
    return result


def run_mc_reference(scenario, n_rays_per_heliostat: int, flux_grid, receiver_extent):
    (u0, u1), (v0, v1) = receiver_extent
    n_u, n_v = flux_grid
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    hist = np.zeros((n_v, n_u))
    t0 = time.perf_counter()
    incident_power_w = 0.0
    for case in scenario.cases:
        seed_seq = np.random.SeedSequence((MC_SEED_BASE, case.heliostat_id))
        rng = np.random.default_rng(seed_seq)
        out = trace_heliostat(
            case.x_mm, case.y_mm, case.rot_az_deg, case.rot_el_deg,
            case.c3, case.c4, case.c5, case.solar_az_deg, case.solar_el_deg,
            scenario.secondary, scenario.receiver, n_rays_per_heliostat, rng,
        )
        watts_per_ray = SOURCE_POWER_W / n_rays_per_heliostat
        # incident (cosine-weighted, on-mirror) power: emitted power scaled by
        # the counter chain's own hit_mirror/emitted fraction times the mean
        # cosine of incidence is not separately tracked by trace_heliostat's
        # counters, so use hit_mirror/emitted * SOURCE_POWER_W as the
        # (uncosine-weighted) mirror-incidence estimate -- consistent across
        # heliostats since every ray sample already carries the source's own
        # angular/spatial distribution.
        c = out["counters"]
        incident_power_w += SOURCE_POWER_W * c.get("hit_mirror", c.get("emitted", n_rays_per_heliostat)) / c.get(
            "emitted", n_rays_per_heliostat
        )
        xy = out["xy"]
        if xy.shape[1] > 0:
            counts, _, _ = np.histogram2d(xy[1], xy[0], bins=[v_edges, u_edges])
            hist += counts * watts_per_ray
    elapsed = time.perf_counter() - t0
    du, dv = u_edges[1] - u_edges[0], v_edges[1] - v_edges[0]
    bin_area_m2 = (du / 1000.0) * (dv / 1000.0)
    flux_w_m2 = hist / bin_area_m2
    total_power_w = float(hist.sum())
    peak_flux = float(flux_w_m2.max())
    return {
        "n_rays_per_heliostat": n_rays_per_heliostat,
        "n_heliostats": len(scenario.cases),
        "wall_time_s": elapsed,
        "total_power_w": total_power_w,
        "peak_flux_w_m2": peak_flux,
        "incident_power_w": incident_power_w,
        "intercept_efficiency": total_power_w / incident_power_w if incident_power_w > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="subsample default field for iteration")
    parser.add_argument("--skip-mc", action="store_true", help="skip the Monte Carlo reference")
    parser.add_argument("--mc-rays", type=int, default=20000, help="MC rays per heliostat")
    args = parser.parse_args()

    results = {}

    print("=== scenario 1: default field ===")
    scenario1 = scenarios.scenario_default_field()
    if args.quick:
        scenario1.cases = scenario1.cases[::20]
        scenario1.notes += " [QUICK: every 20th heliostat]"
    print(f"  {scenario1.notes}")
    r1 = run_deposit_comparison(scenario1)
    results["default_field"] = r1
    print(json.dumps({k: v for k, v in r1.items() if k != "deposit_time_per_heliostat_vs_ring"},
                      indent=2, default=str))

    if not args.skip_mc:
        print("\n=== scenario 1 MC reference ===")
        (u0, u1), (v0, v1) = scenario1.receiver.uv_extent()
        mc = run_mc_reference(scenario1, args.mc_rays, scenario1.flux_grid, ((u0, u1), (v0, v1)))
        results["default_field_mc_reference"] = mc
        print(json.dumps(mc, indent=2))

    for name in ("window_clipping", "heavy_blocking", "cylinder_seam"):
        print(f"\n=== scenario: {name} ===")
        scenario = scenarios.ALL_SCENARIOS[name]()
        print(f"  {scenario.notes}")
        r = run_deposit_comparison(scenario)
        results[name] = r
        print(json.dumps({k: v for k, v in r.items() if k != "deposit_time_per_heliostat_vs_ring"},
                          indent=2, default=str))

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
