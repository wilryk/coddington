# heliostat

Heliostat-field simulation for concentrating solar power towers: field
layouts, Monte Carlo and cone-optics ray tracing, flux maps, shading and
blocking, DNI handling, and annual energy — as an installable Python
library, a command-line tool, and a local web app.

!!! warning "Pre-release"
    The engine is being ported and generalized from a research codebase
    whose Monte Carlo tracer was validated to 0.15% annual agreement against
    a commercial optical CAD package. Expect the API to move until v0.1.0.

Start with the [concepts guide](guide.md) for the model this package is
built around, or the [API reference](api.md) for the modules themselves.

## Run it

```
pip install heliostat[web]
heliostat
```

Typing `heliostat` with no arguments starts the web app on your own machine
and opens a browser at it — `http://127.0.0.1:8420`, or the next free port
if something else already holds 8420, which the startup banner tells you.
Everything runs locally; nothing is uploaded anywhere. The console window it
runs in is the off switch: close it, or press Ctrl+C.

For a double-clickable launcher on your Desktop instead:

```
heliostat shortcut
```

That writes `heliostat.lnk` (Windows), `Heliostat.command` (macOS) or
`heliostat.desktop` (Linux) pointing at the `heliostat` you installed;
`--path DIR` puts it elsewhere, and an existing launcher is only replaced
with `--force`.

`pipx install "heliostat[web]"` installs the app into its own isolated
environment while keeping the `heliostat` command on your PATH (the quotes
stop a Unix shell from treating `[web]` as a glob).

Plain `pip install heliostat` — no extra — gives you the library and the
batch CLI without the app. From a clone, before the first release:

```
pip install -e .[dev,web]
```

## CLI quickstart

Generate a 600-heliostat Fermat-spiral field:

```
heliostat layout fermat --n 600 --a 4.5 --b 0.55 -o field.csv
```

Trace it over a few days and write a stored run. Annual energy interpolates
across solar *declination*, so trace at least two dates at different
declinations — the more you trace, the better the surface:

```
heliostat trace --field field.csv --optics axicon --mode ultra_fast \
    --date 2026-03-21 --date 2026-06-21 --date 2026-09-22 --date 2026-12-21 \
    -o runs/four_days
```

Then read that run and integrate a year from it. `cfg` is any object
carrying the site and mirror area — the library only reads those attributes,
so a `SimpleNamespace` is a perfectly good config:

```python
from types import SimpleNamespace
from heliostat import dni, energy
from heliostat.store import RunStore

store = RunStore("runs/four_days")
cfg = SimpleNamespace(
    site=SimpleNamespace(**store.manifest["site"]),   # the site that was traced
    field=SimpleNamespace(mirror_area_m2=5.0 * 3.0),  # one mirror, m^2
)
result = energy.annual_energy(store.summary(), cfg, dni.ConstantDNI(1000.0), year=2026)
print(f"{result['annual_energy_mwh']:.1f} MWh/yr, "
      f"eta = {result['annual_optical_efficiency']:.3f}")
```

`annual_energy` also returns the hourly and daily breakdowns it integrated,
so the total can be inspected rather than taken on faith. For real weather,
swap `ConstantDNI` for `dni.TableDNI`, `dni.MonthlyProfileDNI`,
`dni.ClearSkyDNI` or a PVGIS/NASA POWER table fetched by `dni.fetch` — none
of which need an API key.

## Web app

Started by a bare `heliostat` (see [Run it](#run-it)) or, if you want to
choose the details, by `heliostat serve --host … --port … --no-browser`. An
explicit `--port` that is already in use is an error rather than a silent
move to another one.

The **design** panel builds a mirror — a plain rectangle, a facet grid, or a
"flower" of petals — and gives it an optical figure: **twisting** (the
solve-driven astigmatic figure for a monolithic mirror, auto-focused
spherical facets for a faceted one), **spherical**, or **flat**. Facet
canting is a separate control, because a canted flat facet is still flat.

The **trace** panel picks one of three optical layouts (prime focus, axicon,
Cassegrain), one of three fidelity modes (ultra-fast and fast-accurate cone
optics, or Monte Carlo), a sun position, and the tower geometry, then traces
and returns a flux map with spot metrics — power, rms radius, centroid, and
the full loss chain from emitted rays to rays in the window.

Alongside the flux map is an interactive **3-D scene** drawn from the trace
that just ran: the facets as they were traced, the secondary revolved from
its own equations, the receiver, the sun, and real traced ray paths. Orbit
and zoom it, click the receiver, the secondary or a heliostat to inspect and
edit it, and drag the receiver along the tower axis. In **field mode** the
same panel traces a whole layout at once, tinting each mirror by its own
efficiency so shaded and blocked regions of the field are visible at a
glance.

## Validation

The Monte Carlo tracer is checked against 45 golden fixtures — five
heliostats × three sun positions × three optical layouts — exported from the
research code this package was ported from. On the platform the fixtures
were generated on, the reproduction is bit-for-bit: the loss-chain counters,
the quantised receiver rays, and the recomputed spot metrics all match
exactly; elsewhere the test degrades to statistical checks against the
fixtures' own Monte Carlo noise. That research tracer is the one validated
to 0.15% annual agreement against a commercial optical CAD package. The
cone-optics backend is then held to the same fixtures, and agrees with a
high-count Monte Carlo reference to about ±0.2% on power and rms radius.
Shading/blocking, the Fermat layout, the aiming solves and the stored-run
read path each have their own fixture parity gates.

## License

MIT. If this software contributes to published research, please cite the
companion paper (reference forthcoming) and this repository:
<https://github.com/wilryk/heliostat>.
