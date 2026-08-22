# heliostat

Heliostat-field simulation for concentrating solar power towers: field
layouts, Monte Carlo and cone-optics ray tracing, flux maps, shading and
blocking, DNI handling, and annual energy — as an installable Python
library, a command-line tool, and a local web app you can point a browser at
to design a mirror and watch it trace.

> **Status: pre-release.** The engine is being ported and generalized from a
> research codebase whose Monte Carlo tracer was validated to 0.15% annual
> agreement against a commercial optical CAD package. Expect the API to move
> until v0.1.0.

**Where everything came from:** the sunshape model, the solar-position
algorithm, the fast tracer's method lineage, the spiral field layout, and
both DNI datasets are other people's work. [**REFERENCES.md**](REFERENCES.md)
says which, and what this package changed about each.

## Design principles

- **Trace once, weigh later.** Ray traces run at a normalized 1000 W/m².
  DNI, reflectivity, and shading/blocking scalars are applied at read time,
  so changing weather data or optical assumptions never requires re-tracing.
- **Bring your own field — or generate one.** Load heliostat positions from
  CSV (or XLSX, with `openpyxl` installed), or generate a Fermat-spiral
  (sunflower) layout with composable filters for roads, rings, angular
  wedges and minimum spacing.
- **Real heliostats.** Multi-facet mirrors built from composable aperture
  regions (rectangles, discs, annuli, polygons, and boolean combinations of
  them), on-axis canting to the slant-range sphere, and flat, spherical or
  astigmatic facet figures.
- **Real sunshapes.** Super-Gaussian and Buie angular source models, shared
  by both tracers; the cone backend's kernel can additionally be broadened
  by mirror slope error and tracking error.
- **Two tracers, one geometry.** A Monte Carlo tracer for reference answers
  and a deterministic cone-optics backend for speed. Both walk the same
  source → mirror → secondary → receiver chain, so a design traced one way
  is the same design traced the other.
- **Honest about what is built.** Every optical configuration shipped here —
  in the CLI, the sweep driver and the web app — uses a flat window
  receiver. External-cylinder and inverted-frustum receivers implement the
  same `Receiver` interface and a tracer will accept one, but no shipped
  configuration, example or test exercises them, and they have no fixture
  parity gate. Treat them as unvalidated until they do.

### Performance, measured

These are wall times actually recorded on an 8-core laptop, not targets:

| Job | Time |
| --- | --- |
| 600-heliostat field, one full day (13 timesteps), ultra-fast mode | ~12 min |
| the same sweep, Monte Carlo at 20,000 rays/heliostat | ~3 min |
| 100-heliostat field, single instant, ultra-fast, in the web app | ~4 s |

The ultra-fast sweep being *slower* than Monte Carlo is a known problem, not
a property of the tracer: per-timestep field bookkeeping (aiming and polygon
occlusion) dominates that number and scales superlinearly with field size. A
performance pass toward a 2-minute full-day sweep is in progress;
`scripts/sweep_benchmark.py` is the yardstick. Annual energy, by contrast,
is already instant — it interpolates traced timesteps rather than re-tracing.

## Run it

```
pip install heliostat[web]
heliostat
```

That is the whole thing. Typing `heliostat` with no arguments starts the web
app on your own machine and opens a browser at it — `http://127.0.0.1:8420`,
or the next free port if something else already holds 8420, which the
startup banner tells you. Everything runs locally; nothing is uploaded
anywhere. The console window it runs in is the off switch: close it, or
press Ctrl+C.

Prefer an icon to a terminal?

```
heliostat shortcut
```

puts a double-clickable launcher on your Desktop — `heliostat.lnk` on
Windows, `Heliostat.command` on macOS, `heliostat.desktop` on Linux —
pointing at the `heliostat` you just installed. `--path DIR` puts it
somewhere else; an existing launcher is never replaced without `--force`.

`pipx install "heliostat[web]"` is a good alternative if you want the app in
its own isolated environment with the `heliostat` command still on your PATH
(the quotes are what stop a Unix shell from treating `[web]` as a glob).

Without the extra — plain `pip install heliostat` — you get the library and
the batch CLI, but not the app.

### The batch tool is the same executable

`heliostat layout`, `heliostat trace`, and `heliostat serve` (the app again,
with `--host`, `--port` and `--no-browser`). `heliostat --help` lists them.
Any argument at all means "run this subcommand" — only a bare `heliostat`
launches the app.

### From a clone

Until the first release:

```
pip install -e .[dev,web]
```

## Web app

The **design** panel builds a mirror — a plain rectangle or a facet grid —
and gives it an optical figure: **twisting** (the
solve-driven astigmatic figure for a monolithic mirror, auto-focused
spherical facets for a faceted one), **spherical**, or **flat**. Facet
canting is a separate control, because a canted flat facet is still flat.
The **trace** panel picks one of three optical layouts (prime focus, axicon,
Cassegrain), the receiver window size, one of three fidelity modes
(ultra-fast and fast-accurate cone optics, or Monte Carlo), a sun position
(typed directly, or computed from a latitude, longitude, date and clock
time), and the tower geometry, then traces
and returns a flux map in kW/m² with spot metrics — power, peak flux, rms
radius, centroid, and the full loss chain from emitted rays to rays in the
window.

Alongside the flux map is an interactive **3-D scene** drawn from the trace
that just ran: the facets as they were traced, the secondary revolved from
its own equations, the receiver, the sun, and real traced ray paths. Orbit
and zoom it, click the receiver, the secondary or a heliostat to inspect and
edit it, and drag the receiver along the tower axis. In **field mode** the
same panel traces a whole layout at once, tinting each mirror by its own
efficiency so shaded and blocked regions of the field are visible at a
glance.

## Quickstart (CLI)

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

## Reproducing the companion paper

[`examples/paper/`](examples/paper/) reproduces all nine configurations the
companion paper compares — three optical layouts × three mirror figures —
from the paper's own field file, frozen-figure tables and irradiance record.
Every published value at the paper's own instant is reproduced to better than
0.7%, and total collected power to better than 0.02%:

```
cd examples/paper
python reproduce.py --quick            # ~2 min, all nine, pipeline test
python reproduce.py --instant-only     # ~5 min, all nine, real numbers
python check.py --out runs/paper       # compare against the published values
```

The example's [README](examples/paper/README.md) documents the provenance of
every data file, the two DNI bases, the validation measurements, and how to
move the whole thing off the paper onto your own field, site and geometry.

## Documentation

Full documentation, including a concepts guide and the API reference, is
built from `docs/` with MkDocs:

```
pip install -e .[docs]
mkdocs serve
```

## Citation

If this software contributes to published research, please cite the
companion paper (reference forthcoming) and this repository:
<https://github.com/wilryk/heliostat>.

## License

MIT — see [LICENSE](LICENSE).
