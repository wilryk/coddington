#!/usr/bin/env python
"""Compare a ``reproduce.py`` run against the paper's published numbers.

Reads ``<out>/results/summary.csv`` (and the per-configuration JSON records
beside it), lines each value up against ``expected/`` and prints the
percentage difference. Tolerances are stated below with the measurement that
justifies each one; nothing here is a round number picked to make a table go
green.

A run that was not traced the paper's way -- quick mode, a reduced field, a
cone backend, a shortened date list -- is reported as ``n/c`` (not
comparable) rather than failed. Saying "FAIL" about a 60-heliostat smoke test
would be a false alarm, and saying "PASS" would be worse.

Exit status is 0 when every comparable value is inside tolerance (or when
nothing was comparable), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
EXPECTED_DIR = _HERE / "expected"

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
#
# Monte Carlo, 120,000 rays, the paper's seed contract. This package's tracer
# is a bit-exact port of the one the paper's runs were traced with (45 golden
# fixtures), so the only expected difference is Monte Carlo shot noise from a
# different seed stream, plus a ~42 mm position discrepancy on 11 of the 645
# heliostats between the shipped field file and the shipped figure tables
# (see the README).
#
# MEASURED, all nine configurations at the paper's instant, full 643-heliostat
# field, 120k rays -- 54 published values (see the README's validation table):
#
#     metric            worst |delta|   3x      tolerance set
#     window_kw             0.019%      0.06%   0.10%  (raised to the floor)
#     r90_mm                0.088%      0.26%   0.30%
#     power_720mm_kw        0.213%      0.64%   0.70%
#     frac_720mm            0.209%      0.63%   0.70%
#     conc_720mm_suns       0.222%      0.67%   0.70%
#     peak_kw_m2            0.644%      1.93%   2.00%
#
# peak_kw_m2 is the value of one bin out of 65,536 -- the noisiest statistic
# on the table by construction, and the only one whose band is wide.
#
# The 0.10% FLOOR on window_kw is deliberate. Three times an 0.019% agreement
# is 0.06%, which is tighter than the seed-to-seed spread a different Monte
# Carlo stream can produce, and a tolerance that fails on a legitimate reseed
# is a broken tolerance. 0.10% is still five times better than any other
# column has to be.
#
# The two ANNUAL columns get a wider band, and not because they are noisier.
# They are integrals over all 94 traced instants, so shot noise is smaller
# there, not larger. The band is wider because the two numbers being compared
# are not the same estimator:
#
#   * the paper's read path maps each untraced month onto the NEAREST TRACED
#     DECLINATION, and the paper quotes a maximum error of 2.69% for doing so;
#   * this package's heliostat.energy.build_interpolator INTERPOLATES across
#     declination instead (see the README, "How the year is filled in").
#
# So a difference between the two annual totals is a statement about the two
# integration methods, and only secondarily about the ray trace. A band
# narrower than the method difference the paper itself documents would fail on
# the method rather than on the reproduction, so 3% -- just above the paper's
# own 2.69% -- is the smallest defensible figure. The instant columns above
# are what actually test the optics.
MC_TOLERANCE_PCT = {
    "peak_kw_m2": 2.0,
    "window_kw": 0.1,
    "power_720mm_kw": 0.7,
    "frac_720mm": 0.7,
    "conc_720mm_suns": 0.7,
    "r90_mm": 0.3,
    "annual_MWh_1kW": 3.0,
    "annual_MWh_petrolina": 3.0,
}

# Cone-optics ("ultra_fast"/"fast_accurate") is a different numerical method,
# not a noisier version of the same one: it deposits an analytic sunshape
# through a measured optical Jacobian instead of sampling rays. Its band is
# therefore a systematic method difference, not shot noise, and 3x-the-noise
# is the wrong rule for it.
#
# MEASURED, ultra_fast, all nine configurations, full field, the paper's
# instant. The story is entirely about the FIGURE:
#
#   metric            twisting+spherical    flat        band set
#   power_720mm_kw          <=0.03%        0.33%          0.5%
#   conc_720mm_suns         <=0.03%        0.34%          0.5%
#   window_kw               <=0.09%        1.39%          2.0%
#   frac_720mm              <=0.07%        1.12%          2.0%
#   r90_mm                  <=0.42%        0.72%          1.5%
#   peak_kw_m2              2.1-5.2%      16-28%         40.0%
#
# On the two focused figures the cone backend reproduces the paper's power to
# a few hundredths of a percent -- better than the Monte Carlo run's own shot
# noise. On `flat` it does not, and peak flux is where it shows: the spot is
# metres across and spills past the window, so almost all of the map's
# structure is at the clipped edge, which is exactly what a linearised deposit
# smooths. The 40% band is not an endorsement -- it is the honest width of a
# column this method should not be trusted on. Use monte_carlo for peak flux.
CONE_TOLERANCE_PCT = {
    "peak_kw_m2": 40.0,
    "window_kw": 2.0,
    "power_720mm_kw": 0.5,
    "frac_720mm": 2.0,
    "conc_720mm_suns": 0.5,
    "r90_mm": 1.5,
    # Both effects at once: the cone method's own spread and the annual
    # estimator difference above.
    "annual_MWh_1kW": 5.0,
    "annual_MWh_petrolina": 5.0,
}

ANNUAL_COLS = ("annual_MWh_1kW", "annual_MWh_petrolina")
INSTANT_COLS = (
    "peak_kw_m2",
    "window_kw",
    "power_720mm_kw",
    "frac_720mm",
    "conc_720mm_suns",
    "r90_mm",
)


def load_expected() -> pd.DataFrame:
    """The paper's two tables, joined on (layout, figure)."""
    annual = pd.read_csv(EXPECTED_DIR / "annual_energy_720mm.csv")
    instant = pd.read_csv(EXPECTED_DIR / "instant_summary.csv")
    for name, frame in (("annual_energy_720mm.csv", annual), ("instant_summary.csv", instant)):
        if len(frame) != 9:
            raise ValueError(f"expected/{name} has {len(frame)} rows, expected 9")
    return annual.merge(instant, on=["layout", "figure"], how="outer")


def load_results(out_root: Path) -> tuple[pd.DataFrame, dict]:
    """``summary.csv`` plus the per-configuration JSON records beside it."""
    results_dir = out_root / "results"
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"No results at {summary_path}. Run reproduce.py --out {out_root} first.")
    summary = pd.read_csv(summary_path)
    records = {}
    for path in sorted(results_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        records[(rec["layout"], rec["figure"])] = rec
    return summary, records


def _tolerances(record: dict | None) -> dict:
    mode = (record or {}).get("mode", "monte_carlo")
    return MC_TOLERANCE_PCT if mode == "monte_carlo" else CONE_TOLERANCE_PCT


def _comparable(record: dict | None, column: str) -> bool:
    """Is this value a like-for-like comparison with the paper?"""
    if record is None:
        return False
    flag = record.get("paper_comparable")
    if isinstance(flag, dict):
        return bool(flag.get("annual" if column in ANNUAL_COLS else "instant"))
    return bool(flag)


def _reason(record: dict | None) -> str:
    """Why a record is not comparable, in one short phrase."""
    if record is None:
        return "no run record"
    bits = []
    if record.get("mode") != "monte_carlo":
        bits.append(f"mode={record.get('mode')}")
    if record.get("rays_per_heliostat") != 120_000:
        bits.append(f"rays={record.get('rays_per_heliostat')}")
    if record.get("n_heliostats") != 643:
        bits.append(f"heliostats={record.get('n_heliostats')}")
    if record.get("grid_size") != 256:
        bits.append(f"grid={record.get('grid_size')}")
    n_dates = len(record.get("traced_dates") or record.get("dates") or [])
    if n_dates != 7:
        bits.append(f"dates={n_dates}")
    return ", ".join(bits) or "not comparable"


def compare(out_root: Path, columns=None) -> tuple[pd.DataFrame, bool, int]:
    """Build the side-by-side table. Returns ``(table, ok, n_compared)``."""
    expected = load_expected()
    summary, records = load_results(out_root)
    got = summary.set_index(["layout", "figure"])
    columns = tuple(columns) if columns else ANNUAL_COLS + INSTANT_COLS

    rows = []
    ok = True
    n_compared = 0
    for _, exp in expected.iterrows():
        key = (exp["layout"], exp["figure"])
        record = records.get(key)
        if key not in got.index:
            continue
        mine = got.loc[key]
        tol = _tolerances(record)
        for col in columns:
            if col not in exp or pd.isna(exp[col]):
                continue
            value = mine.get(col, float("nan"))
            ref = float(exp[col])
            delta = float("nan") if pd.isna(value) else (float(value) - ref) / ref * 100.0
            if pd.isna(value):
                verdict = "missing"
            elif not _comparable(record, col):
                verdict = "n/c"
            else:
                n_compared += 1
                passed = abs(delta) <= tol[col]
                ok &= passed
                verdict = "PASS" if passed else "FAIL"
            rows.append(
                {
                    "layout": exp["layout"],
                    "figure": exp["figure"],
                    "metric": col,
                    "paper": ref,
                    "reproduced": float("nan") if pd.isna(value) else float(value),
                    "delta_pct": delta,
                    "tol_pct": tol[col],
                    "verdict": verdict,
                    "note": "" if verdict not in ("n/c", "missing") else _reason(record),
                }
            )
    return pd.DataFrame(rows), ok, n_compared


def render(table: pd.DataFrame) -> str:
    lines = []
    header = (
        f"{'layout':<12} {'figure':<10} {'metric':<21} "
        f"{'paper':>12} {'reproduced':>12} {'delta %':>9} {'tol %':>7}  verdict"
    )
    for (layout, figure), grp in table.groupby(["layout", "figure"], sort=False):
        if not lines:
            lines.append(header)
            lines.append("-" * len(header))
        for _, r in grp.iterrows():
            delta = "      n/a" if pd.isna(r["delta_pct"]) else f"{r['delta_pct']:+9.3f}"
            repro = "         --" if pd.isna(r["reproduced"]) else f"{r['reproduced']:12.4f}"
            note = f"  ({r['note']})" if r["note"] else ""
            lines.append(
                f"{layout:<12} {figure:<10} {r['metric']:<21} "
                f"{r['paper']:12.4f} {repro} {delta} {r['tol_pct']:7.2f}  "
                f"{r['verdict']}{note}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="runs/paper", help="the reproduce.py --out root")
    p.add_argument("--annual-only", action="store_true")
    p.add_argument("--instant-only", action="store_true")
    p.add_argument("--csv", default=None, help="also write the comparison table here")
    args = p.parse_args(argv)

    columns = None
    if args.annual_only:
        columns = ANNUAL_COLS
    elif args.instant_only:
        columns = INSTANT_COLS

    table, ok, n_compared = compare(Path(args.out), columns)
    if table.empty:
        print("Nothing to compare: no configuration in the results matches the paper's nine.")
        return 0
    print(render(table))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")

    n_nc = int((table["verdict"] == "n/c").sum())
    n_missing = int((table["verdict"] == "missing").sum())
    n_fail = int((table["verdict"] == "FAIL").sum())
    print(
        f"{n_compared} value(s) compared, {n_fail} outside tolerance; "
        f"{n_nc} not comparable, {n_missing} missing."
    )
    if n_compared == 0:
        print(
            "\nNo value was traced the paper's way, so nothing was checked -- the\n"
            "note beside each row says what differs (--quick, a reduced field, a\n"
            "cone backend, or a shortened date list). Nothing passed and nothing\n"
            "failed. Reproducing the paper needs monte_carlo at 120,000 rays on\n"
            "the full 643-heliostat field."
        )
        return 0
    worst = table.loc[table["verdict"].isin(["PASS", "FAIL"]), "delta_pct"].abs().max()
    print(f"worst comparable |delta| = {worst:.3f}%")
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
