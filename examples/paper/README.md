# Reproducing the companion paper

This directory reproduces the nine configurations compared in the companion
SPIE paper (14212-5), from the paper's own field file, its own frozen-figure
tables and its own irradiance record, using nothing but this package's public
API.

The paper's traces were produced by the research tracer that this package's
Monte Carlo backend is a bit-exact port of — the port is gated by 45 golden
fixtures — so this is a genuine reproduction rather than a re-derivation. The
[validation section](#what-the-validation-measured) reports how close it
actually lands: **every published value at the paper's own instant is
reproduced to better than 0.7%, and most to better than 0.2%.**

---

## The nine situations

Three optical layouts × three mirror figures.

| Layout | What sits above the field | Receiver | Throughput |
| --- | --- | --- | --- |
| `prime_focus` | nothing — every heliostat aims at the focus itself | flat window at 35,335 mm, facing **down** | 0.90 (1 mirror @ 0.9) |
| `cassegrain` | hyperboloid relay, vertex 26,994 mm, R 26,112 mm, conic −5.3176, aperture 14 m | flat window at 7,000 mm, facing **up** | 0.81 (2 mirrors @ 0.9) |
| `axicon` | cone, apex 27,000 mm, half-angle 20°, aperture 14 m | flat window at 7,000 mm, facing **up** | 0.81 (2 mirrors @ 0.9) |

| Figure | What the mirror does | Where the coefficients come from |
| --- | --- | --- |
| `twisting` | Re-solves its astigmatic figure at every timestep — the mirror physically twists as the sun moves. The best a figured heliostat can do. | `heliostat.geometry.aiming.solve_*`, per heliostat per timestep |
| `spherical` | One figure, frozen for the year. | `data/fixed_shapes_*.csv` (see [Provenance](#provenance-of-every-data-file)) |
| `flat` | No figure at all: `c3 = c4 = c5 = 0`. | — |

The paper calls the first of these **twisting** throughout; if you find the
word "adaptive" anywhere in this pack, it is a bug.

Everything else is shared across the nine: 643 heliostats of 5 × 3 m, the
Buie sunshape (4.65 mrad limb, no circumsolar aureole), no slope or tracking
error, a 4 m × 4 m receiver window binned 256 × 256, and 120,000 rays per
heliostat per timestep. Shading and blocking are scalar union efficiencies
(`heliostat.geometry.shading.polygon_occlusion`), applied to power at read
time; occluders are not themselves put in the ray path.

---

## Commands

Everything below runs from this directory. Install the package first
(`pip install -e .[dev]` from the repository root, or just `pip install
heliostat`).

### 1. Quick — does the pipeline work?

```
python reproduce.py --quick
python check.py --out runs/paper
```

**Two minutes** for all nine configurations (measured: 2 min 08 s). Quick
mode traces the 60 innermost heliostats at 20,000 rays on the December
solstice only, and prints a banner saying so. Annual energy is not computed
at all — the interpolation needs at least two declinations and one date gives
one.

`check.py` will report every value as `n/c` (not comparable) and exit 0. That
is the correct outcome: quick numbers are a pipeline test, not a
reproduction, and a checker that shouted FAIL at them would be crying wolf.

### 2. One configuration, for real

```
python reproduce.py --configs axicon:twisting --out runs/paper
python check.py --out runs/paper
```

**Just under two hours** — see [the timings below](#4-the-full-reproduction)
for where that goes. `--configs` also accepts a bare layout
(`--configs axicon` → all three of its figures), a bare figure
(`--configs twisting` → all three layouts), several at once, or `all`.

### 3. One configuration at the paper's instant only

```
python reproduce.py --configs axicon:twisting --instant-only --out runs/instant
python check.py --out runs/instant --instant-only
```

**36 seconds** per configuration; 5 min 11 s for all nine. Traces only
2026-12-21 09:06 solar — the instant the paper's spot table is quoted at —
with the full 643-heliostat field at the full ray budget. This is the
cheapest honest check that the whole optical chain agrees with the paper, and
it is what the validation table below was measured with.

### 4. The full reproduction

```
python reproduce.py --out runs/paper
python check.py --out runs/paper --csv comparison.csv
```

**About 16 hours** for all nine, single-process, on a 4-core/8-thread laptop
(i7-1185G7 class): roughly 1 hour 50 minutes per configuration. Runs are
written per configuration and skipped on a re-run unless you pass
`--rebuild`, so this can be done a configuration at a time.

> These timings were measured before the neighbour-search fix that cut
> occlusion work by roughly 3.8x over a full day, so they are now
> **conservative** — expect faster. They are left as measured rather than
> rescaled, because a derived number is not a measured one.

That comes from a direct profile of one timestep of the real thing, 643
heliostats, not from scaling the cheap number:

| Per timestep | as first measured | now |
| --- | --- | --- |
| Monte Carlo trace, 120,000 rays × 643 heliostats | **43 s** (67 ms each) | 43 s |
| Occlusion | **27 s** | **~8 s** |
| **Total** | ~70 s | ~51 s |

**The occlusion column changed, and it is worth knowing why**, because the
first version of this file explained the 27 s as a cost worth paying rather
than an inefficiency. It was both. The neighbour list used to be sized once
per run from the *lowest* sun elevation anywhere in it, because that is when
shadows reach furthest: the full grid runs from 1.8° to 83°, and at 1.8° a
shadow outruns the 60 m cap, so every heliostat was tested against every
neighbour within 60 m — **189 of them on average** — at every timestep, all
day, including noon, where 12 would have done.

Sizing that radius per timestep instead cuts occlusion over a full day by
about 3.8x (measured on this field: 47.5 min to 12.5 min). It is not simply
a smaller radius: shading reach shrinks as the sun climbs, but *blocking*
reach does not — it is set by the reflected beam's angle — so a radius taken
from the sun alone drops real blockers at high sun, moving `eta_block` by
0.11. The radius covers both reaches, and
`tests/test_polygon_shading.py` pins that it reproduces a whole-field
neighbour list exactly at every elevation.

So a full configuration should now come in nearer **1 h 20 min** than 1 h
50 min. That figure is derived from the two measurements above rather than
timed end to end, which is why the headline number in this file is still the
one that was actually clocked.

Dropping `--rays` to 20,000 cuts the trace to 6 s per timestep (33 s total,
~52 min per configuration). Shot noise on a whole-field total at 643 × 20,000
rays is a few hundredths of a percent, so the annual energies barely move —
but the instant table's peak flux does, and `check.py` will mark the run
not-comparable because it is not what the paper traced.

Storage is about 70 MB per configuration (49 MB of flux maps at 512 kB per
timestep, plus a ~19 MB `summary.csv` of 643 × 94 rows) — around 600 MB for
all nine.

### The honest fast alternative

```
python reproduce.py --mode ultra_fast --out runs/fast
```

The cone-optics backend deposits the analytic Buie sunshape through a
measured optical Jacobian instead of sampling rays: deterministic, no shot
noise, and faster. It is *not* the paper's method, and `check.py` reports its
results as not-comparable and widens its band. Measured against the paper at
the same instant, on the full field, the band depends almost entirely on the
**figure**:

| Metric | `twisting` + `spherical` | `flat` |
| --- | --- | --- |
| power inside 720 mm | ≤ 0.03% | 0.33% |
| total window power | ≤ 0.09% | 1.39% |
| r90 | ≤ 0.42% | 0.72% |
| **peak flux** | **2.1 – 5.2%** | **16 – 28%** |

On the two focused figures it reproduces the paper's collected power to a few
hundredths of a percent — better than the Monte Carlo run's own shot noise,
and for a fraction of the cost. On `flat` it does not, and peak flux is where
it shows: that spot is metres across and spills past the window, so nearly
all of the map's structure sits at the clipped edge, which is exactly what a
linearised deposit smooths. Use it to explore; use `monte_carlo` to
reproduce, and never quote a cone-mode peak flux.

---

## The two DNI bases

The paper quotes every annual energy twice, and neither number is a "typical
meteorological year".

**Constant 1,000 W/m² — the optical result.** Column `annual_MWh_1kW`. The
sun is assumed to deliver exactly the trace normalisation at every daylight
hour of the year. Nothing about the weather enters, so the difference between
two configurations is purely a difference in optics. This is the number to
compare layouts with.

**Petrolina climatology — the energy result.** Column
`annual_MWh_petrolina`. NASA POWER hourly all-sky DNI for 2001–2024 at
Petrolina (−9.4, −40.5), averaged on a (day-of-year, hour) grid across the 24
years and then smoothed with a circular ±5-day window — about 264 samples per
point. The raw record integrates to 1,848.6 kWh/m²/yr; the shipped
climatology reproduces that to 1,848.0 (−0.03%, and a smoothing that
preserves mass is the only kind worth using).

Two things about that record are worth stating plainly:

- **It was recorded 11.5° of longitude east of the site.** Solar noon happens
  46 minutes earlier at Petrolina than at the site, so pairing the two by
  clock hour would put the irradiance curve and the optical-efficiency curve
  out of step. `heliostat.dni.SolarTimeAligned` reads the table at the
  *site's* solar time instead. This is a shift of the lookup, not a fudge:
  optical efficiency is keyed on declination and hour angle, which do not
  depend on longitude at all.
- **The site's latitude is −10.0, not Petrolina's −9.4.** That is deliberate
  in the paper: the field is at a round −10.0 and the irradiance is borrowed
  from the nearest long-record station. Latitude is the part a longitude
  shift cannot fix — it changes the sun's path rather than its clock — and
  0.6° is small enough to ignore, but it is an approximation and not a
  coincidence.

Both providers are library code (`heliostat.dni.DailyClimatologyDNI` and
`SolarTimeAligned`); this example only wires them up.

---

## A recorded contradiction: 15 m vs 14 m

The Cassegrain's secondary shades the field. How big is it?

- The paper's run **manifests** record a shading body of radius **15,000 mm**.
- The surrounding **prose** describes a rim clearing **14,000 mm**.

Both are in the source material and they disagree. This pack follows the
manifest — and then measured which one the published numbers were actually
produced with. At the paper's own instant, with everything else held fixed:

| Cassegrain shading body | window power vs paper |
| --- | --- |
| 15,000 mm (manifest) | **−0.02%** |
| 14,000 mm (prose) | +0.36% |
| none | +3.51% |

So the manifest is what the runs were traced with, and 15,000 mm is what
`SHADE_BODY_RADIUS_MM["cassegrain"]` says. The hyperboloid's own **optical**
aperture stays at 14,000 mm — that is the surface rays actually reflect off,
and it is unaffected by this question.

The same measurement on the axicon points the other way: its cone shades with
its own 14,000 mm aperture (−0.02% against the paper, where 15,000 mm gives
−0.46%), so the two layouts genuinely use different shading radii. Prime
focus has nothing above the field at all. Override any of it with
`--shade-radius`.

---

## Provenance of every data file

Everything in `data/` came from the paper's own inputs. Nothing was
regenerated, refitted or rounded.

### `field_645.csv` — 16.5 kB, 645 rows

The project owner's heliostat layout, byte-identical to the source file; only
the filename changed. Two columns, `X (m)` and `Y (m)`, radii 30.0 to 89.6 m.

It contains **two coincident pairs**: heliostats 144 = 192 and 241 = 289 sit
at the same point to within a millimetre. Two mirrors cannot occupy one
position, and — because the shading test requires an occluder to be strictly
in front — they would not even shade each other, so each would be traced and
summed at full power. `heliostat.field.load_field` finds them by distance,
drops the higher id of each pair, and warns. **643 heliostats are traced**,
and the survivors keep the source file's numbering (so there is no heliostat
192 or 289, and heliostat 300 is still heliostat 300).

### `fixed_shapes_pf35335_spherical.csv` and `fixed_shapes_cass34892_spherical.csv` — 46 kB each, 645 rows

The frozen `spherical` figure for the prime-focus and Cassegrain layouts: one
sphere per heliostat, chosen once for the whole year. The header comments
record how each was built — a time-weighted median over 4,876 timesteps
across 365 days, weighted by daylight hours × cos(AOI), so a heliostat's
frozen figure is the one that serves it best over the hours that actually
matter. Columns: `heliostat, x_mm, y_mm, c3, c4, c5`, in the Zernike
convention the trace backends consume (`c3 = c5 = 0`, pure defocus).

### `fixed_shapes_axicon_medial.csv` — 46 kB, 643 rows

The axicon's `spherical` entry — and it is deliberately not a plain sphere.
A cone has optical power in one direction only, so the heliostat that best
serves it with a single fixed curvature is not the one focused at the
tangential distance but the one at the **medial** power between the
tangential and sagittal foci:

```
R = 4 · Ft · Fs / (Ft + Fs)
```

Written as a focal length that is `R = 2f` with `1/f = ½(1/Ft + 1/Fs)`: the
sphere whose optical *power* is the average of the tangential and sagittal
powers, rather than the one that matches either. It is analytic and
sun-independent. The paper uses it as the axicon's frozen-figure baseline,
and the plain `R = 2·Ft` sphere is what it beats.

Note the row count: this table carries the **643 survivors, renumbered
0..642**, while the other two carry all **645 source rows with the source
ids**. Matching frozen figures by id would therefore be right for two files
and silently wrong for the third, shifting every figure past heliostat 192 by
one. `reproduce.load_fixed_shapes` matches on the `x_mm`/`y_mm` columns
instead and requires the match to be one-to-one.

> **A discrepancy worth knowing about.** On 11 of the 645 heliostats, the
> position recorded in the figure tables differs from `field_645.csv` by up
> to 42 mm — the field file has those coordinates snapped to a round metre or
> decimetre, and the figure tables were built before the snap. The field's
> minimum spacing is 5,831 mm, so a 42 mm gap identifies a heliostat with no
> ambiguity whatsoever, and 42 mm on a 46 m focal distance moves the required
> curvature by one part in a thousand. It is left visible rather than
> reconciled: the field file is authoritative for positions, the tables are
> authoritative for figures, and neither was edited.

### `dni_nasa_hourly.csv.gz` — 930 kB gzipped (4.4 MB raw), 210,384 rows

NASA POWER hourly all-sky DNI, shipped gzipped (pandas reads `.gz`
natively). Columns: `year, month, day, hour, dni_w_m2`.

Fetch parameters, for anyone who wants to re-pull or move the site:

| | |
| --- | --- |
| Endpoint | `power.larc.nasa.gov/api/temporal/hourly/point` |
| Parameter | `ALLSKY_SFC_SW_DNI` |
| Community | `RE` |
| Time standard | `UTC` (**not** the endpoint's LST default — see below) |
| Latitude / longitude | −9.4, −40.5 (Petrolina) |
| Years | 2001–2024 |
| API key | none required |

The time standard matters. The hourly endpoint defaults to local solar time,
so a client that omits `time-standard` receives timestamps that are already
local and then converts them again — which shifts the whole diurnal curve
three hours early here. That leaves the annual total untouched (a shift
cannot change a sum) while misaligning every hour of sunlight against the
field's optical efficiency. `heliostat.dni.fetch("nasa", ...)` asks for UTC
explicitly.

---

## What the validation measured

The full 643-heliostat field, 120,000 rays per heliostat, the paper's Buie
sunshape and shading bodies, at the paper's own instant (2026-12-21 09:06
solar), against `expected/instant_summary.csv`:

Every one of the 54 published values, reproduced. Percentages are
(reproduced − paper) / paper.

| Layout / figure | peak kW/m² | window kW | 720 mm kW | frac | suns | r90 mm |
| --- | --- | --- | --- | --- | --- | --- |
| prime_focus / twisting | +0.011 | −0.002 | −0.028 | −0.030 | −0.028 | +0.041 |
| prime_focus / spherical | −0.031 | −0.002 | −0.053 | −0.048 | −0.053 | +0.055 |
| prime_focus / flat | −0.644 | +0.002 | −0.176 | −0.141 | −0.181 | +0.052 |
| cassegrain / twisting | +0.199 | −0.019 | −0.077 | −0.059 | −0.078 | +0.088 |
| cassegrain / spherical | −0.442 | −0.019 | −0.125 | −0.111 | −0.127 | +0.068 |
| cassegrain / flat | +0.001 | −0.012 | −0.213 | −0.151 | −0.222 | +0.061 |
| axicon / twisting | −0.185 | −0.015 | −0.071 | −0.059 | −0.071 | +0.047 |
| axicon / spherical | −0.202 | −0.016 | −0.113 | −0.104 | −0.113 | +0.058 |
| axicon / flat | −0.079 | −0.015 | −0.199 | −0.209 | −0.189 | +0.009 |
| **worst \|Δ\|** | **0.644** | **0.019** | **0.213** | **0.209** | **0.222** | **0.088** |

Total collected power lands within **0.02%** on all nine — better agreement
than Monte Carlo shot noise alone would predict, which is what a bit-exact
port of the same tracer should give. The one wide column, `peak_kw_m2`, is
the value of a single bin out of 65,536 and is the noisiest statistic on the
table by construction. `check.py` sets each tolerance at three times the
worst value in this table (with a 0.1% floor on `window_kw`, because three
times a 0.019% agreement is tighter than a legitimate reseed can hold).

Reproducing this table takes about 5 minutes:

```
python reproduce.py --instant-only --out runs/instant
python check.py --out runs/instant --instant-only
```

### The annual columns: validated at full scale for one configuration

`axicon:twisting` was reproduced end to end — all 94 timesteps, 643
heliostats, 120,000 rays — on 2026-08-20 (1 h 43 m single-core), and the
annual totals landed on the paper's:

| DNI basis | paper (MWh) | reproduced (MWh) | Δ |
| --- | --- | --- | --- |
| constant 1 kW/m² | 20,725.77 | 20,730.60 | **+0.023%** |
| Petrolina climatology | 9,338.35 | 9,339.69 | **+0.014%** |

The same configuration through the `ultra_fast` cone backend (2 h 06 m)
gives 20,743.41 / 9,345.56 MWh — **+0.085% / +0.077%** against the paper, so
the deterministic backend integrates the year to within a tenth of a percent
as well.

Two things follow. First, the instant table above shares everything with the
annual except the time integration — same trace, same occlusion, same
aperture mask — so with both now validated on this configuration, the other
eight configurations' annual columns rest on validated optics plus a
time-integration method measured at +0.02%. Second, that +0.02% is itself
notable, because the time integration is where this package and the paper
deliberately differ:

### How the year is filled in — and how that differs

The paper's read path maps each untraced month onto the **nearest traced
declination** and reports a maximum error of 2.69% for doing so.

This package does something else, and the difference is deliberate rather
than incidental. `heliostat.energy.build_interpolator` resamples each traced
day onto a normalised hour angle `u = ha / H0(declination)` — zero at solar
noon, ±1 at true sunrise and sunset, so a short winter day and a long summer
one overlap completely — anchors each day's curve to zero at the horizon,
mirrors the sample set about both declination turning points (±23.44°), and
interpolates across declination with a monotone PCHIP that cannot overshoot
into an efficiency the trace never saw. Then `annual_energy` walks all 8,760
hours of the year and evaluates that surface at each hour's own
(declination, hour angle).

So: **interpolation across declination, not nearest-neighbour snapping to
it.** With seven traced declinations spanning both solstices, no hour of the
year is extrapolated (`extrapolated_fraction` is recorded in each
`results/<config>.json` and should read 0.0). The published annual numbers
and the numbers this pack produces are therefore two estimators of the same
integral, not one estimator run twice.

`check.py` gives the annual columns a **3%** band — just above the 2.69% the
paper quotes for its own month mapping, since a band narrower than the method
difference the paper itself documents would fail on the method rather than on
the reproduction. In practice the two estimators agree far inside that band:
the full-scale `axicon:twisting` run above measured **+0.023%**, meaning the
paper's nearest-declination snapping and this package's PCHIP interpolation
land on essentially the same integral when fed seven declinations spanning
both solstices. The band is kept at 3% because one configuration is one data
point, not because a larger difference is expected.

### Where else this pack could not be literal

- **Stored flux maps are the field total, not per heliostat.** At 643
  heliostats on a 256 × 256 grid a per-heliostat map is 168 MB per timestep —
  15 GB per configuration, 140 GB for the nine. Flux maps add linearly and
  the occlusion weights are per-heliostat scalars, so the stored
  eta-weighted field sum is the *exact* quantity every reported number is
  built from, and the per-heliostat detail that matters (power into the
  window and into the 720 mm aperture, pointing, occlusion) is in
  `summary.csv`, one row per heliostat per timestep. The manifest records
  this under `counts_convention`.
- **`r90_mm` is measured about the receiver axis, not the spot centroid.** An
  aperture is a fixed hole in a fixed place; a spot whose centroid has
  wandered off axis really does spill, and a centroid-referenced radius would
  hide exactly that. All nine published `r90` values are reproduced to better
  than 0.09%, including the three `flat` rows where the spot is enormous and
  the two conventions would disagree loudly — which is good evidence the
  paper measured it the same way.
- **`heliostat.sweep.run_sweep` is not used.** It is the library's own sweep
  driver and it does most of this, but it fixes the flux grid at 128 bins,
  the sunshape at super-Gaussian, the figure at "whatever the solve
  returned", and passes no secondary to the shading test. All four are
  keywords on the functions underneath it, which is what `reproduce.py`
  calls. No physics is duplicated; the aiming solves, the tracer, the
  occlusion, the store and the energy integration are all imported.

---

## Customize it

The whole point of a reproduction pack is that you can move it off the paper.
Each of these is a small edit to `reproduce.py` or a command-line flag.

### Swap the field

Any CSV with x/y columns works — `X (m)`, `x`, `x (mm)` and a few other
spellings are all recognised (`heliostat.field.load_field`). Point
`FIELD_FILE` at it, or generate one:

```
heliostat layout fermat --n 600 --a 4.5 --b 0.55 -o my_field.csv
```

Coincident positions are detected and dropped automatically, whatever they
are. Note that the frozen-figure tables describe *this* field: `spherical`
runs will refuse to start on a field they do not cover, which is the intended
behaviour. Use `twisting` or `flat` with a new field, or rebuild the tables.

### Move the site and re-fetch DNI

Edit `SITE`, then pull a fresh irradiance record — no API key needed for
either source:

```python
from types import SimpleNamespace
from pathlib import Path
from heliostat import dni

cfg = SimpleNamespace(
    site=SimpleNamespace(latitude=-23.5, longitude=-46.6, timezone=-3),
    path=lambda rel: Path("data") / rel,
)
dni.fetch("nasa", cfg, year=2023)  # NASA POWER, one historical year
dni.fetch("pvgis", cfg)  # PVGIS typical meteorological year
```

Set `DNI_DATA_LONGITUDE` to the longitude you fetched at (or to `SITE[1]` if
you fetched at the site, which disables the solar-time shift). A single
historical year is a `TableDNI` or `MonthlyProfileDNI`, not a climatology —
`DailyClimatologyDNI` wants many years and will honestly describe itself as
`1 years` if you give it one.

### Change the tower geometry

`PRIME_FOCUS_HEIGHT_MM`, the `CASSEGRAIN_*` conic constants,
`AXICON_APEX_MM` / `AXICON_HALF_ANGLE_DEG` / `AXICON_APERTURE_RADIUS_MM`,
`GROUND_RECEIVER_Z_MM` and `WINDOW_MM` are all read in one place
(`paper_optics`), and each one is named to match the constructor argument of
the `heliostat.geometry.secondary` / `receiver` class it feeds, so the aiming
solve and the traced surface cannot drift apart. `APERTURE_RADIUS_MM`
changes what "inside the receiver" means for every reported number.

### Design a mirror in the web app

```
pip install heliostat[web]
heliostat serve
```

Build a faceted heliostat, canting and all, in the browser; watch it trace in
the 3-D scene; then pass the resulting `HeliostatDesign` to the tracer. Both
backends consume designs, and `run_config` can be given one — with a design
the `c3`/`c4`/`c5` figure lives on the design's own surfaces instead.

### Run your own sweep

For anything that is not the paper, the library's own driver is simpler:

```python
from heliostat.field import load_field
from heliostat.sweep import run_sweep

field = load_field("my_field.csv", mirror_width_mm=5000, mirror_height_mm=3000)
run_sweep(field, dates, mode="ultra_fast", optics="axicon", out_dir="runs/mine")
```

or from the command line, `heliostat trace --help`.

---

## Files

| Path | What it is |
| --- | --- |
| `reproduce.py` | Traces the configurations and writes `results/` |
| `check.py` | Compares `results/` against `expected/` with justified tolerances |
| `expected/annual_energy_720mm.csv` | The paper's annual energies, both DNI bases |
| `expected/instant_summary.csv` | The paper's spot table at 2026-12-21 09:06 |
| `data/` | The paper's own inputs (see [Provenance](#provenance-of-every-data-file)) |
