# Spec §C build plan: secondary-mirror irradiance & absorbed heat

Status: **design complete 2026-08-26, implementation not started.** Handed off by the study session; build directly from here. Scope: AxiconSecondary and CassegrainSecondary only (Pyramid/NoSecondary have no single radial coordinate / no surface — §C scopes to axicon/Cassegrain).

## Parameterization (document like receiver.bin_areas_m2)

- **u** = azimuthal arc length at the aperture rim: `aperture_radius_mm * atan2(x, -y)` — same −y/north-seam convention as `geometry/receiver.py`, app-wide consistency.
- **v** = radial distance from the axis in horizontal projection, `h = hypot(x, y)`. Chosen over true slant distance: needs no z, trivially invertible, and Cassegrain slant arc-length has no closed form (would need ODE integration); radial keeps both shapes on one code path.
- **Area weighting**: `dA = h · sec(local_slope(h)) · dh · du / aperture_radius_mm` per bin, m². Axicon: `sec(local_slope)` = constant `1/cos(half_angle_deg)`. Cassegrain: `sqrt(1 + slope(h)²)` with `slope` from the implicit hyperboloid (`ζ(h) = (R − sqrt(R² − (1+k)h²))/(1+k)`, `slope = h/(R − (1+k)ζ)`) — same `zeta`/`kk` convention as `CassegrainSecondary.redirect`.

## Hit points — already computed, zero new ray tracing

- MC: `trace_heliostat(..., return_secondary_hits=True)` already returns `secondary_xy` (mc.py ~447–451).
- Cone: `trace_heliostat_cone` computes the full-stencil secondary hit `pre` unconditionally at ~line 407; the chief leg's hit is recoverable by reshaping `on_sec`/`pre` to `(legs, m)` and indexing leg 0.

## Cone-mode fidelity: chief-ray-point deposit, disclosed as coarse

Each mirror sample deposits its full weight at its chief ray's secondary hit; weight = `weights[idx] * frac_secondary[idx]` where `frac_secondary` reuses the per-node shading+blocking mask (`node_ok_n` BEFORE the receiver-window/aperture filters are ANDed in, cone.py ~553–556) — a free byproduct captured one step earlier than the receiver deposit. Samples whose chief ray misses the secondary rim fall back to depositing at each passing node's own hit (`pre_n`), mirroring the existing `node_fallback` pattern (~617–629). Full footprint/Jacobian deposit onto the secondary (a second Jacobian in the secondary's own uv) is deliberately skipped. **UI must say "coarse in cone modes, exact in Monte Carlo" wherever the secondary map shows.**

## Energy-consistency pin (write BEFORE any UI)

Small fixture, axicon + Cassegrain, one instant, both backends, no occluders. Assert total secondary-incident power == power leaving the mirrors toward it (post shading/blocking, pre secondary-reflectance): tolerance 0.5% for cone chief-point fidelity, 1e-6 relative for MC (direct histogram, no approximation).

## Build order (file by file)

1. `geometry/secondary.py` — helpers `secondary_has_flux_map`, `secondary_uv`, `secondary_uv_extent`, `secondary_bin_areas_m2` (formulas above, ready to transcribe).
2. `trace/cone.py` — chief-point + node-fallback secondary flux dict; localized diff around lines 407 and 550–576, reusing `w_nodes`, `node_ok_n`, `pre_n` already in scope.
3. Tests: new `tests/test_secondary_flux.py` — uv round-trip, area sum equals full surface area, the energy pin (both backends). Physics validated before UI.
4. `web/app.py` — explicit `secondary_reflectance` field (default 0.90) on Receiver & Tower models, applied at the same choke point `store.flux_scale` uses for `throughput` — default physics stays bit-identical (0.81 = 0.9 × 0.9) while R becomes independently settable. Secondary payload block parallel to `_flux_grid_payload`, only when `optics != "prime_focus"`. MC field aggregator: pass `return_secondary_hits=True`, histogram `secondary_xy` through the new bins (mirrors receiver `histogram2d` at ~2277–2281).
5. JS — Receiver | Secondary selector on run-bar + Analysis maps (mockup M9), R input in Receiver & Tower panel, absorbed-heat readout (incident MW, absorbed MW at shown R, peak absorbed kW/m²).
6. `_fea_csv` helpers — secondary export with `x, y, flux, absorbed` columns, same commented-header convention.

CLAUDE.md resource rules apply throughout (one server, small serial traces, never the 643-heliostat field for verification).
