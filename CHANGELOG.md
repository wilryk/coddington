# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Repository bootstrap: packaging, CI, docs scaffolding.
- `solar` module: NOAA solar position, sunrise/sunset, sweep time grids.
- `dni` module: DNI providers (constant, TMY table, monthly profile,
  multi-year climatology, clear-sky) with key-free PVGIS/NASA POWER
  fetchers and solar-time alignment.
- `metrics` module: per-heliostat spot metrics (power, rms radius, r50/r90,
  peak flux, centroid, spillage).
