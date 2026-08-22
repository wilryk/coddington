# Reproducing the paper

`examples/paper/` reproduces the nine configurations compared in the
companion SPIE paper (14212-5) from the paper's own inputs, using only this
package's public API. The paper's traces came from the research tracer this
package's Monte Carlo backend is a bit-exact port of, so this is a
reproduction rather than a re-derivation.

The [example's own README](https://github.com/wilryk/coddington/tree/main/examples/paper)
has the commands, the data provenance and the customization notes. This page
is the result.

## The nine

Three optical layouts — prime focus, Cassegrain, axicon — crossed with three
mirror figures: **twisting** (re-solved every timestep), **spherical** (one
frozen figure per heliostat) and **flat**. Shared across all nine: 643
heliostats of 5 × 3 m at 30–90 m, the Buie sunshape, a 4 × 4 m window,
120,000 rays per heliostat, and seven dates spanning both solstices.

## What it measured

Full field, full ray budget, at the paper's own instant (2026-12-21 09:06
solar). Worst difference across all nine:

| Metric | worst \|Δ\| |
| --- | --- |
| Total window power | 0.019% |
| r90 radius | 0.088% |
| Power inside 720 mm | 0.213% |
| Concentration (suns) | 0.222% |
| Peak flux | 0.644% |

Peak flux is one bin out of 65,536 — the noisiest statistic here by
construction.

Annual energy was validated end to end for one configuration:
`axicon:twisting` reproduces the paper to **+0.023%** on the constant-1 kW
basis and **+0.014%** on the Petrolina climatology. The cone backend lands
within +0.085%.

## Two deliberate differences

**The year is interpolated, not snapped.** The paper maps each untraced
month onto the nearest traced declination and reports a maximum 2.69% error
for doing so; this package interpolates across declination with a monotone
PCHIP that cannot overshoot an efficiency the trace never saw. The two are
different estimators of the same integral — measured difference, +0.02%.

**Stored flux maps are the field total, not per heliostat.** A per-heliostat
map is 168 MB per timestep at this field size. Maps add linearly and
occlusion weights are per-heliostat scalars, so the stored sum is exact;
per-heliostat power, pointing and occlusion live in the run's `summary.csv`.
