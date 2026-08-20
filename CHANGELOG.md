# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Everything below is the initial, unreleased body of work. The API is not yet
stable; see the status note in the README.

### Added

**Solar position, weather and metrics**

- `solar`: NOAA solar position, sunrise/sunset, declination/hour angle, and
  sweep time grids.
- `dni`: DNI providers (constant, TMY table, monthly profile, multi-year
  climatology, clear-sky) with key-free PVGIS and NASA POWER fetchers and
  solar-time alignment for tables recorded at a different longitude.
- `metrics`: per-heliostat spot metrics — power, rms radius, r50/r90, peak
  flux, centroid, spillage, and encircled energy from rays or from a map.

**Ray tracing**

- Monte Carlo tracer walking source → mirror → secondary → receiver, with
  super-Gaussian and Buie sunshapes and a full per-stage loss chain.
- Cone-optics backend: a five-ray Jacobian stencil per mirror sample that
  deposits the analytic sunshape through the real optical chain, with no
  shot noise.
- Three named fidelity modes — `ultra_fast` and `fast_accurate` (cone
  optics, first- and second-order deposits) and `monte_carlo`.
- Measured per-sample angular transmission, so secondary rims and receiver
  window edges clip with real penumbra rather than a hard mask.
- Golden-fixture parity suite: 45 cases reproduce the source research
  tracer bit-for-bit, plus a cone-vs-Monte-Carlo validation harness and an
  analytic convergence suite.

**Geometry**

- Receivers: flat window — the one every shipped optical configuration uses
  — plus external-cylinder and truncated-frustum implementations of the
  same interface, which no shipped configuration or test exercises yet.
- Secondaries: axicon cone, Cassegrain hyperboloid, pyramid, and the
  no-secondary prime-focus case.
- Aperture sketches: composable CAD-style regions — rectangle, disc,
  ellipse, annulus, polygon, circular array, and boolean union,
  intersection and difference.
- Design layer: `HeliostatDesign` of facets carrying flat, spherical or
  Zernike-astigmatic surfaces, on-axis canting, silhouette extraction and
  a footprint preview. Both tracers consume designs.
- Aiming solves: per-layout pointing and focusing for prime focus, axicon,
  Cassegrain and pyramid layouts, reproducing the fixture pointing rows to
  machine precision.
- Shading and blocking: silhouette-aware, exact polygon-projection
  occlusion (Sutherland–Hodgman) with a union efficiency that does not
  double-count overlapping shadows.

**Fields, sweeps and storage**

- `field`: heliostat position loading from CSV or XLSX, coincident-position
  detection, and downselection.
- `field_layouts`: Fermat-spiral (sunflower) generation with composable
  filters for angular wedges, radial rings, road corridors and minimum
  spacing.
- `sweep`: the multi-day, multi-timestep trace driver, with standard
  optical configurations and optional worker processes.
- `store`: `RunStore`, a compact on-disk run format — quantised receiver
  rays or binned flux, a per-heliostat summary, and a manifest recording
  what was traced.
- `energy`: optical efficiency from a stored run, hull-free interpolation
  over declination and hour angle, annual energy against any DNI provider,
  traced-day cross-checks and per-heliostat annual totals.
- CLI: `heliostat layout fermat`, `heliostat trace` and `heliostat serve`.

**Web app** (`pip install heliostat[web]`, then `heliostat serve`)

- Design panel: rectangle, facet grid and flower layouts, with a twisting /
  spherical / flat surface selector and separate facet canting.
- Trace panel: three optical layouts, three fidelity modes, an editable sun
  position and editable tower geometry, returning a flux map and spot
  metrics.
- Interactive 3-D scene rendered from the trace that just ran — real traced
  ray paths, orbit and zoom, click-to-inspect and edit the receiver,
  secondary or a heliostat, and drag the receiver along the tower axis.
- Field mode: trace a whole layout at once, with each mirror tinted by its
  own efficiency.

**Project**

- Repository bootstrap: packaging, CI (lint, tests on 3.11/3.12, docs) and
  documentation scaffolding.
- Documentation: concepts guide and API reference built with MkDocs.
