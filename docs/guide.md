# Concepts

This page describes the model the package is built around. It is deliberately
short on API detail — see the [API reference](api.md) for signatures.

## Trace once, weigh later

A ray trace is expensive; multiplying a number by 0.94 is not. So every trace
in this package runs at a **normalized 1000 W/m²** direct normal irradiance,
and everything that is merely a scalar multiplier is applied afterwards, at
read time:

- mirror reflectivity and other throughput terms,
- the actual DNI for the hour in question,
- shading and blocking efficiencies.

`heliostat.store.RunStore` stores ray **counts**, never scaled flux, for
exactly this reason: revising reflectivity or swapping a weather file never
requires re-tracing.

The same idea one level up drives annual energy. Optical efficiency depends
only on where the sun *is*, not on what day it is — two dates that put the
sun at the same declination and hour angle give identical optics. So the
trace samples a dimensionless surface

```
eta_optical(declination, hour_angle) = receiver power / (DNI x mirror area)
```

at the traced timesteps, and `heliostat.energy` integrates that surface over
all 8760 hours of the year against a full DNI series. A handful of traced
days yields a genuine annual number rather than four days multiplied by 91.

The cost is a sampling caveat, and it is worth taking seriously: the
declination axis is interpolated from however many distinct declinations you
traced. Two dates give two points across the ±23.4° range. A dozen dates give
a much better surface for proportionally more trace cost.
`energy.declination_coverage` reports what a given date set actually spans,
and `energy.suggest_sweep_dates` proposes one.

## Fields and layouts

A `heliostat.field.HeliostatField` is just heliostat centre positions in
millimetres, loaded from a CSV or XLSX of x/y columns (`field.load_field`) or
generated (`field_layouts.generate`).

The generator implemented today is the **Fermat spiral**, also called a
sunflower layout: `r = a * k**b` with successive heliostats separated by the
golden angle. It is a good default for a surround field — spacing grows
naturally with radius, and there are no radial gaps to align shadows along.

Layouts are shaped by composable *filters*, each of which simply drops
positions: `wedge_filter` (keep an angular sector), `ring_filter` (keep a
radial band), `road_corridors` (cut service roads at given azimuths) and
`min_spacing_filter` (enforce a minimum centre-to-centre distance). The
generator oversamples candidates, applies the filters and truncates to the
requested count, so the number you ask for is the number you get.

`field.coincident_pairs` and `field.downselect` help with real position
files: the first finds duplicate positions, the second takes a
representative subset (farthest-point or stratified) when you want a quick
answer from a big field.

## Designs and surfaces

A **design** (`heliostat.geometry.design.HeliostatDesign`) is what a single
heliostat physically is: a list of facets, each with an outline and an
optical figure.

Outlines come from `heliostat.geometry.aperture`, a small CAD-like sketching
layer — `Rect`, `Disc`, `Ellipse`, `Annulus`, `Polygon`, `regular_polygon`,
`CircularArray`, and boolean `Union` / `Intersection` / `Difference`, plus
`Translate` and `Rotate`. Builders assemble the common cases:
`rect_heliostat`, `grid_facets` (an n × m facet array) and `flower` (petals
around a hub).

Each facet carries a **surface**, and this is a separate axis from canting:

| Surface | What it is |
| --- | --- |
| `Flat` | No figure at all. Expect a mirror-shaped wash, not a spot. |
| `Spherical` | A spherical cap, focused at a given focal length or at the heliostat's own slant range. |
| `ZernikeAstig` | Defocus plus astigmatism — the figure an aiming solve produces for an off-axis mirror. |

**Canting** (`cant_on_axis`) is different: it tilts a whole facet so its
reflection points at the design's focal point. It does not curve anything. A
canted flat facet is still flat — it just looks somewhere else. Both axes
apply at once.

The web app exposes this as three surface modes:

- **Twisting** — the solve-driven choice. For a monolithic rectangle that is
  the aiming solve's own astigmatic figure, the twisting mirror of the
  companion paper. A faceted design has no monolithic surface to twist, so
  its facets get auto-focused spherical curvature instead.
- **Spherical** — a spherical cap on every facet, at the resolved cant focal.
- **Flat** — no figure anywhere.

## Optical layouts

Every layout shipped here delivers flux onto a **flat window receiver**.
`heliostat.geometry.receiver` also carries external-cylinder and
truncated-frustum implementations of the same interface, but no shipped
configuration or test exercises them — treat those as unvalidated.

Three ground-based layouts are covered by aiming solves in
`heliostat.geometry.aiming`, each paired with a secondary from
`heliostat.geometry.secondary`:

- **Prime focus** — no secondary. Every heliostat aims at and focuses on one
  shared point on the tower axis, where the receiver sits.
- **Axicon** — a secondary-mirror concentrator. A cone near the top of the
  tower redirects the combined beam onto a ground receiver below it. A cone
  has no focus, so each heliostat's aim point is derived from its own radial
  position; the cone also has optical power in one direction only, which the
  heliostat must pre-compensate with an extra figure correction.
- **Cassegrain** — the same tower-reflector idea with a hyperboloid instead
  of a cone. Upstream of the shared focus it is optically identical to prime
  focus, and because a hyperboloid is stigmatic between its two foci its
  relay adds no astigmatism, so the two share one solve.

A fourth solve, **pyramid** (an inverted four-sided pyramid — the axicon's
cone with its circular symmetry broken into four flats), is implemented but
is not covered by this package's golden fixtures.

## Fidelity modes

All three modes produce the same kind of flux map on the same receiver grid.

| Mode | Backend | What you get |
| --- | --- | --- |
| `ultra_fast` | Cone optics, linearised five-ray stencil | ~1,200 deterministic rays per heliostat. Total power and spot moments essentially exact; ~1%-of-peak curvature residual in local map detail. |
| `fast_accurate` | Cone optics, quadratic nine-ray stencil | Measures the optical map's curvature and deposits through it, removing the leading ultra-fast error for roughly twice the cost. |
| `monte_carlo` | Full Monte Carlo | The reference. Bit-reproducible from its seed; noise falls as 1/√rays. |

The **cone backend** is the interesting one. Rather than modelling
astigmatism, cone folding and magnification separately, it samples the
mirror, fires five (or nine) deterministic rays per sample through the real
optical chain, measures the resulting 2×2 Jacobian by central differences,
and deposits the analytic sunshape through it. The geometry carries the
optics; nothing is approximated away. The result has no shot noise at all,
which is why a fast mode can stand in for a very large Monte Carlo run.

Both backends share the same sunshapes (super-Gaussian and Buie) and the
same source → mirror → secondary → receiver chain, and both report a full
per-stage loss chain, so "where did the light go" is always answerable.

## Shading, blocking and occlusion

`heliostat.geometry.shading` computes what neighbouring mirrors take away:
**shading** (a neighbour blocks incoming sunlight) and **blocking** (a
neighbour intercepts the reflected beam on its way out).

Two things are worth knowing. First, occlusion is computed from each
design's own **silhouette**, so a flower-shaped heliostat shades as a flower,
not as its bounding rectangle. Second, `occlusion_efficiency` takes the
**union** of all occluded regions rather than multiplying per-neighbour
efficiencies together — overlapping shadows must not be counted twice, and
the product form measurably under-reports.

## Runs and the store

`heliostat.sweep.run_sweep` is the driver: a field, a list of dates, an
optical configuration and a fidelity mode in, a stored run out. It builds
the time grid from sunrise/sunset, solves pointing for every heliostat at
every timestep, applies occlusion, traces, and writes.

A run directory (`heliostat.store.RunStore`) holds:

```
manifest.json          run metadata and quantisation scale
summary.csv            one row per (timestep, heliostat)
raw/<key>_rays.npy     int16 receiver x/y for every ray in the window
raw/<key>_index.npy    which slice of that array belongs to which heliostat
flux/<key>.npy         per-heliostat binned counts
```

Raw rays are the source of truth; the flux maps are a cache, fully
reconstructible with `RunStore.rebin`. The int16 quantisation over the
receiver window is far finer than a flux bin and irrelevant next to Monte
Carlo noise. The summary is CSV, not Parquet, because it is small, needs no
extra dependency, and opens in a spreadsheet.

## Field mode in the web app

The single-heliostat view answers "what does this mirror do". **Field mode**
answers "what does this layout do": it generates or accepts a layout, solves
pointing for every heliostat at one instant, applies the same union occlusion
the sweep driver uses, and traces the lot.

The 3-D scene then tints each mirror's silhouette by its own efficiency, so
the shaded and blocked regions of the field — usually the inner ring and the
mirrors behind their neighbours — are visible immediately. A single-heliostat
field is bit-identical to the single-heliostat trace, which is pinned by a
test; field mode is the same physics at scale, not a second implementation.
