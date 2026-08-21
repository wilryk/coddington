# Reproducing the paper

The companion SPIE paper (14212-5) compares nine configurations of a solar
tower field. `examples/paper/` reproduces all nine from the paper's own
inputs, using nothing but this package's public API.

The paper's traces came from the research tracer this package's Monte Carlo
backend is a bit-exact port of, so this is a reproduction rather than a
re-derivation. **Every published value at the paper's own instant is
reproduced to better than 0.7%, and total collected power to better than
0.02%** — see [validation](#validation) below.

The [example's own README](https://github.com/wilryk/heliostat/tree/main/examples/paper)
is the full version, with the provenance of every data file and a
"customize it" section. This page is the summary.

## The nine situations

Three optical layouts × three mirror figures.

| Layout | Above the field | Receiver | Throughput |
| --- | --- | --- | --- |
| `prime_focus` | nothing | flat window at 35,335 mm, facing down | 0.90 |
| `cassegrain` | hyperboloid relay, 14 m aperture | flat window at 7,000 mm, facing up | 0.81 |
| `axicon` | cone, apex 27,000 mm, half-angle 20° | flat window at 7,000 mm, facing up | 0.81 |

| Figure | What the mirror does |
| --- | --- |
| `twisting` | Re-solves its astigmatic figure every timestep — the mirror twists as the sun moves |
| `spherical` | One figure per heliostat, frozen for the year |
| `flat` | No figure at all |

Shared across the nine: 643 heliostats of 5 × 3 m at 30–90 m radius, the
Buie sunshape, no slope or tracking error, a 4 × 4 m receiver window binned
256 × 256, 120,000 rays per heliostat, and seven traced dates spanning both
solstices (94 timesteps).

## Running it

```
cd examples/paper

python reproduce.py --quick                     # ~2 min, all nine, pipeline test
python reproduce.py --instant-only              # ~5 min, all nine, real numbers
python reproduce.py --configs axicon:twisting   # ~1 h 50 min, one configuration
python reproduce.py                             # ~16 h, the full reproduction

python check.py --out runs/paper                # compare against expected/
```

A full timestep costs about 70 s — 43 s of ray tracing and 27 s of shading —
and that second number is why the full run costs far more than scaling the
single-instant figure suggests. The shading neighbour list is sized from the
*lowest* sun elevation in the run, so tracing the true sunrise and sunset
edges puts 189 neighbours per heliostat into the occlusion test instead of
5.6. The example's README breaks the trade down.

`--quick` traces 60 heliostats at 20,000 rays on one date and says so loudly;
`check.py` reports its values as *not comparable* rather than failing them.

`--mode ultra_fast` switches to the deterministic cone backend. Measured
against the paper on the full field, it reproduces collected power to within
0.03% for the `twisting` and `spherical` figures — better than the Monte
Carlo run's own shot noise — but only to 1.4% for `flat`, and its peak flux
is 16–28% low on `flat`, where the spot is metres across and almost all of
the map's structure sits at the clipped window edge. It is a good way to
explore and a bad way to quote a peak flux.

## The two DNI bases

Every annual energy is quoted twice, and neither is a typical meteorological
year:

- **Constant 1,000 W/m²** — the *optical* result. No weather enters, so the
  difference between two configurations is purely a difference in optics.
- **Petrolina climatology** — the *energy* result. NASA POWER hourly all-sky
  DNI, 2001–2024, averaged on a (day-of-year, hour) grid across the 24 years
  and smoothed with a circular ±5-day window. The record integrates to
  1,848.6 kWh/m²/yr and the shipped climatology reproduces that to 1,848.0.

The record was taken 11.5° of longitude east of the site, so it is read at
the *site's* solar time rather than its own clock
([`SolarTimeAligned`](api.md)) — optical efficiency is keyed on declination
and hour angle, which do not depend on longitude, so the shift is exact
rather than approximate.

## Validation

Full field, full ray budget, at the paper's own instant (2026-12-21 09:06
solar). Worst percentage difference across all nine configurations:

| Metric | worst \|Δ\| |
| --- | --- |
| Total window power | 0.019% |
| r90 radius | 0.088% |
| Power inside 720 mm | 0.213% |
| Aperture fraction | 0.209% |
| Concentration (suns) | 0.222% |
| Peak flux | 0.644% |

Peak flux is the value of one bin out of 65,536 — the noisiest statistic on
the table by construction. `check.py` sets each tolerance at three times the
worst observed value.

The annual columns were validated at full scale for one configuration:
`axicon:twisting`, all 94 timesteps at 120,000 rays, reproduces the paper's
annual energy to **+0.023%** on the constant-1 kW basis and **+0.014%** on
the Petrolina climatology (1 h 43 m single-core). The same configuration
through the `ultra_fast` cone backend lands within **+0.085%** in 2 h 06 m.
The remaining eight configurations' annual columns therefore rest on
validated optics (the instant table above) plus a time-integration method
measured at +0.02% — notable because the time integration is exactly where
this package and the paper differ, as described next.

## Two things this pack does differently

Both are documented rather than smoothed over.

**The year is filled in by interpolation, not by nearest-declination
snapping.** The paper's read path maps each untraced month onto the nearest
traced declination, and reports a maximum error of 2.69% for doing so. This
package's [`energy.build_interpolator`](api.md) instead resamples each traced
day onto a normalised hour angle (zero at solar noon, ±1 at true sunrise and
sunset), anchors it to zero at the horizon, mirrors the samples about both
declination turning points, and interpolates across declination with a
monotone PCHIP that cannot overshoot an efficiency the trace never saw. With
seven declinations spanning both solstices, no hour of the year is
extrapolated. The published annual numbers and this pack's are therefore two
estimators of the same integral, not the same estimator run twice: a small
annual difference is a statement about the two methods, and the instant table
above is what tests the optics.

**Stored flux maps are the eta-weighted field total, not per heliostat.** A
per-heliostat map is 168 MB per timestep at this field size and grid — 140 GB
for the nine configurations. Flux maps add linearly and occlusion weights are
per-heliostat scalars, so the stored sum is the exact quantity every reported
number is built from; per-heliostat power, pointing and occlusion are in the
run's `summary.csv`.

## Cassegrain: 15 m or 14 m?

The paper's run manifests record a secondary shading body of radius 15,000 mm;
the surrounding prose describes a rim clearing 14,000 mm. Both are in the
source material and they disagree.

Measured at the paper's instant, with everything else fixed, the Cassegrain's
window power lands −0.02% from the published value at 15,000 mm and +0.36% at
14,000 mm, so the manifest is what the runs were traced with. The axicon
measures the other way and shades with its own 14,000 mm cone aperture. Both
are recorded in `SHADE_BODY_RADIUS_MM` and overridable with `--shade-radius`;
the traced optical apertures are 14,000 mm in both cases and are unaffected.
