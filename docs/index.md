# Coddington

A twisting heliostat software package: field layouts, ray tracing, flux
maps, shading and blocking, DNI, and annual energy. Installed and imported
as `heliostat`.

## Install

Download the Windows installer from
[releases](https://github.com/wilryk/coddington/releases) — no Python
needed.

From a clone of the repository instead:

```
pip install -e .[web]          # library, CLI and the app
heliostat                      # starts the app
```

There is no PyPI package yet. `heliostat shortcut` adds a Desktop
launcher.

## CLI quickstart

Generate a field:

```
heliostat layout fermat --n 600 --a 4.5 --b 0.55 -o field.csv
```

Trace it. Annual energy interpolates across solar *declination*, so trace at
least two dates at different declinations — the more dates, the better the
surface:

```
heliostat trace --field field.csv --optics axicon --mode ultra_fast \
    --date 2026-03-21 --date 2026-06-21 --date 2026-09-22 --date 2026-12-21 \
    -o runs/four_days
```

Integrate a year from the stored run. `cfg` is any object carrying the site
and mirror area, so a `SimpleNamespace` will do:

```python
from types import SimpleNamespace
from heliostat import dni, energy
from heliostat.store import RunStore

store = RunStore("runs/four_days")
cfg = SimpleNamespace(
    site=SimpleNamespace(**store.manifest["site"]),
    field=SimpleNamespace(mirror_area_m2=5.0 * 3.0),
)
result = energy.annual_energy(store.summary(), cfg, dni.ConstantDNI(1000.0), year=2026)
print(f"{result['annual_energy_mwh']:.1f} MWh/yr")
```

For real weather, swap `ConstantDNI` for `TableDNI`, `MonthlyProfileDNI`,
`ClearSkyDNI`, or a PVGIS / NASA POWER table from `dni.fetch` — none need an
API key.

## The app

Design a mirror, trace it, and look at the result four ways: flux map (kW/m²
with power, peak flux, rms radius), an interactive 3-D scene with real ray
paths, the mirror's sag, and an Analysis tab that traces a whole day and
plots collected power against time. Field mode traces a whole layout at
once. Day sweeps and flux maps export to CSV.

## Next

- [Concepts](guide.md) — the model this is built around
- [API reference](api.md)
- [Reproducing the paper](paper.md)
- [References](references.md) — what this borrows, and from whom
