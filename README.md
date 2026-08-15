# heliostat

Heliostat-field simulation for concentrating solar power towers: field
layouts, Monte Carlo ray tracing, flux maps, shading and blocking, DNI
handling, and annual energy — as an installable Python library and CLI.

> **Status: pre-release.** The engine is being ported and generalized from a
> research codebase whose Monte Carlo tracer was validated to 0.15% annual
> agreement against a commercial optical CAD ray tracer. Expect the API to
> move until v0.1.0.

## Design principles

- **Trace once, weigh later.** Ray traces run at a normalized 1000 W/m².
  DNI, reflectivity, and shading/blocking are applied at read time, so
  changing weather data or optical assumptions never requires re-tracing.
- **Bring your own field — or generate one.** Load heliostat positions from
  CSV, or generate classic layouts (Fermat-spiral/sunflower first) with
  composable filters for roads and exclusion zones.
- **Real heliostats.** Multi-facet mirrors, on-axis canting to the
  slant-range sphere, spherical facet curvature, slope error.
- **Real receivers.** Flat windows, external cylinders, inverted-frustum
  cavities — plus beam-down secondaries (axicon, Cassegrain).
- **Fast answers.** Quick-look presets trace a full day for ~600 heliostats
  in well under two minutes on a laptop; annual energy for an arbitrary
  calendar day is instant via an efficiency interpolator.

## Install (once released)

```
pip install heliostat
```

## License

MIT. If this software contributes to published research, a citation of the
companion paper (reference forthcoming) is appreciated.
