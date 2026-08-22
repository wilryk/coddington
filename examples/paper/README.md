# Reproducing the companion paper

Reproduces the nine configurations compared in the companion SPIE paper
(14212-5) from the paper's own field file, frozen-figure tables and
irradiance record, using only this package's public API.

The paper's traces came from the research tracer this package's Monte Carlo
backend is a bit-exact port of, so this is a reproduction rather than a
re-derivation.

## The nine

Three optical layouts × three mirror figures.

| Layout | Above the field | Receiver |
| --- | --- | --- |
| `prime_focus` | nothing | flat window at 35,335 mm, facing down |
| `cassegrain` | hyperboloid relay, 14 m aperture | flat window at 7,000 mm, facing up |
| `axicon` | cone, apex 27,000 mm, half-angle 20° | flat window at 7,000 mm, facing up |

| Figure | The mirror |
| --- | --- |
| `twisting` | re-solves its astigmatic figure every timestep |
| `spherical` | one figure per heliostat, frozen for the year |
| `flat` | no figure at all |

Shared: 643 heliostats of 5 × 3 m at 30–90 m, the Buie sunshape (no
circumsolar aureole), no slope or tracking error, a 4 × 4 m window binned
256 × 256, 120,000 rays per heliostat, seven dates spanning both solstices
(94 timesteps). Shading and blocking are scalar union efficiencies applied
at read time.

## Commands

```
python reproduce.py --quick                     # ~2 min, pipeline test
python reproduce.py --instant-only              # ~5 min, real numbers
python reproduce.py --configs axicon:twisting   # ~1 h 20 min, one config
python reproduce.py                             # all nine

python check.py --out runs/paper                # compare against expected/
```

`--configs` also takes a bare layout (`axicon` → its three figures), a bare
figure (`twisting` → its three layouts), several at once, or `all`.

`--quick` traces 60 heliostats at 20,000 rays on one date; `check.py`
reports its values as *not comparable* rather than failing them. Runs are
skipped on re-run unless you pass `--rebuild`, so the full set can be done
a configuration at a time. Storage is ~70 MB per configuration.

`--mode ultra_fast` uses the deterministic cone backend. It reproduces
collected power to within 0.03% for `twisting` and `spherical` — better
than the Monte Carlo run's own shot noise — but only 1.4% for `flat`, where
its peak flux is 16–28% low. Good for exploring, bad for quoting a peak
flux.

## What the validation measured

Full field, full ray budget, at the paper's own instant (2026-12-21 09:06
solar). Worst difference across all nine configurations:

| Metric | worst \|Δ\| |
| --- | --- |
| Total window power | 0.019% |
| r90 radius | 0.088% |
| Power inside 720 mm | 0.213% |
| Concentration (suns) | 0.222% |
| Peak flux | 0.644% |

Peak flux is one bin out of 65,536, the noisiest statistic here by
construction. `check.py` sets each tolerance at three times the worst
observed.

Annual energy was validated at full scale for one configuration:
`axicon:twisting` reproduces the paper to **+0.023%** on the constant-1 kW
basis and **+0.014%** on the Petrolina climatology. The cone backend lands
within +0.085%.

## Two DNI bases

Every annual number is quoted twice, and neither is a typical
meteorological year:

- **Constant 1,000 W/m²** — the optical result; no weather, so a difference
  between configurations is purely optics.
- **Petrolina climatology** — the energy result. NASA POWER hourly all-sky
  DNI 2001–2024, averaged on a (day-of-year, hour) grid and smoothed with a
  circular ±5-day window; 1,848.6 kWh/m²/yr. Read at the *site's* solar
  time, since the record was taken 11.5° of longitude away.

## Where this differs from the paper

- **The year is interpolated, not snapped.** The paper maps each untraced
  month to the nearest traced declination (max error 2.69%); this package
  interpolates across declination with a monotone PCHIP. Measured
  difference: +0.02%.
- **Stored flux maps are the field total**, not per heliostat — a
  per-heliostat map is 168 MB per timestep here. Maps add linearly and
  occlusion weights are per-heliostat scalars, so the sum is exact;
  per-heliostat detail is in `summary.csv`.
- **Cassegrain shading body: 15 m.** The paper's manifests say 15,000 mm and
  its prose says 14,000 mm. Measured at the paper's instant, 15,000 mm lands
  −0.02% from the published value and 14,000 mm +0.36%, so the manifests are
  what the runs used. The axicon shades with its own 14,000 mm aperture.
  Both are in `SHADE_BODY_RADIUS_MM`; `--shade-radius` overrides.

## Data provenance

| File | What it is |
| --- | --- |
| `field_645.csv` | The paper's 645 heliostat positions, 30–90 m. Two coincident pairs (144=192, 241=289) are dropped at load, leaving 643. |
| `fixed_shapes_pf35335_spherical.csv`, `fixed_shapes_cass34892_spherical.csv` | Frozen per-heliostat figures for the `spherical` runs, 645 rows keyed by source id. |
| `fixed_shapes_axicon_medial.csv` | The axicon's medial-sphere baseline, R = 4·Ft·Fs/(Ft+Fs). **643 rows, renumbered 0–642** — matched on position, not id, since id-matching would shift every figure past heliostat 192. |
| `dni_nasa_hourly.csv.gz` | NASA POWER hourly DNI, 2001–2024, Petrolina. Acknowledge NASA POWER in work that uses it. |

Eleven positions differ by up to 42 mm between the field file and the
figure tables (the field file was snapped to round metres). Minimum spacing
is 5.8 m, so the position match is unambiguous; both files are shipped
unedited.

## Customize it

- **Your own field:** point `--field` at a CSV of x/y columns in metres.
- **Your own site:** `heliostat.dni.fetch("pvgis", cfg)` or
  `fetch("nasa_power", cfg)` — neither needs an API key.
- **Your own tower or mirror:** the web app (`heliostat`) edits geometry and
  design interactively; `reproduce.py`'s `paper_optics()` shows the same
  thing in code.
- **Your own sweep:** `heliostat trace --field … --date … -o runs/mine`.

## Files

`reproduce.py` traces and reports; `check.py` compares against
`expected/`; `data/` holds the four inputs above; `runs/` is output.
