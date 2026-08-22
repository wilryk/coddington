# Coddington

**A twisting heliostat software package.** Simulate a heliostat field for a
concentrating solar power tower: field layouts, ray tracing, flux maps,
shading and blocking, DNI, and annual energy — as a desktop app, a
command-line tool, and a Python library (installed and imported as
`heliostat`).

> **Pre-release.** The API will move until v0.1.0.

## Run it

**No Python?** Download the build for your platform from
[releases](https://github.com/wilryk/coddington/releases), unzip, and
double-click **Coddington**. A console window opens, then your browser.
Close the console to quit.

The builds are unsigned, so Windows shows "Windows protected your PC"
(*More info → Run anyway*) and macOS needs the app allowed through
Gatekeeper on first launch.

**With Python:**

```
pip install heliostat[web]
heliostat
```

`heliostat shortcut` puts a launcher on your Desktop. Without the `[web]`
extra you get the library and the batch CLI, but not the app.

## The app

**Design** builds a mirror — rectangle or facet grid — and gives it a
figure: **twisting** (re-solved as the sun moves), **spherical**, or
**flat**.

**Trace** picks the optics (prime focus, axicon, Cassegrain), the tower and
receiver geometry, a fidelity mode, and a sun position — typed, or computed
from a latitude, longitude and date.

Four views of the result:

| View | Shows |
| --- | --- |
| Flux map | kW/m² on the receiver, with power, peak flux, rms radius |
| 3-D scene | The traced geometry and real ray paths; click to inspect and edit |
| Mirror sag | The figure doing the focusing, in millimetres |
| Analysis | A whole day: collected power against time, and the day's energy |

**Field mode** traces a whole layout at once, each mirror tinted by its
shading efficiency. Day sweeps report progress and export to CSV; so does
any flux map.

## Command line

```
heliostat layout fermat --n 600 --a 4.5 --b 0.55 -o field.csv
heliostat trace --field field.csv --optics axicon --mode ultra_fast \
    --date 2026-03-21 --date 2026-06-21 -o runs/two_days
```

Annual energy reads a stored run — see the [quickstart](docs/index.md).

## Speed

600 heliostats over a full day (13 timesteps), 8 cores:

| Mode | Time |
| --- | --- |
| Monte Carlo, 20,000 rays | 1.0 min |
| Monte Carlo, 120,000 rays | 6.1 min |
| Ultra-fast cone optics | 7.0 min |

Cone optics carries no shot noise, so compare it against the 120,000-ray
row rather than the cheap one. Annual energy interpolates traced timesteps
rather than re-tracing, so it is effectively instant.

## Validation

The Monte Carlo tracer reproduces 45 golden fixtures from the research code
this was ported from, bit-for-bit; that code was validated to 0.15% annual
against a commercial optical CAD package. `examples/paper/` reproduces the
companion paper's nine configurations — every published instantaneous value
within 0.7%, collected power within 0.02%, annual energy within 0.03%.

Flat-window receivers are the only ones any shipped configuration
exercises. Cylinder and frustum receivers exist and satisfy the same
interface but have no fixture gate: treat them as unvalidated.

## More

- [Documentation](docs/index.md) — concepts, API, full quickstart
- [REFERENCES.md](REFERENCES.md) — the models, algorithms and datasets this
  borrows, and what was changed about each
- [examples/paper/](examples/paper/) — reproduce the companion paper

## Citation

Please cite the companion paper (reference forthcoming) and this repository
— see `CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).
