# Coddington 0.2 — secondary optics, a true sun, and reference fields

This release adds secondary-mirror analysis to the axicon and
Cassegrain layouts, replaces the placeholder sunshape with a
literature-validated one, and ships four famous solar-tower fields as
honestly-labelled reference projects. The workspace is also
reorganized into four explicit tabs, and a run of reported bugs —
rays, drag-panning, project save/load, and a silent startup — are
fixed.

## What's new

**A reorganized workspace.** Four tabs — Design, 3D View, Heliostat
Shape, and Analysis — replace the old single Workspace tab, and the
app now opens straight into the live 3-D scene. Design's plan and
elevation views switch with an explicit toggle instead of the old
automatic viewport morph.

**Secondary-mirror analysis**, new for axicon and Cassegrain layouts.
A Receiver | Secondary selector shows irradiance and absorbed heat on
the secondary mirror itself — read face-on as a disk with compass
markings — alongside a reflectance setting, an FEA export, and the
flux painted onto the secondary in the 3-D scene.

**Secondary perturbations.** Decenter and tip/tilt misalignment can be
dialed in on the secondary at every fidelity. Monte Carlo adds
measured deformation maps and parametric warp (defocus, astigmatism),
with a sag view showing the summed surface.

**Measured error-map import for heliostats.** A gridded sag-deviation
CSV can now be imported per heliostat in Monte Carlo, alongside the
implied slope error it represents next to the analytic setting.

**Pointing error**, a fourth optical-error input alongside slope,
specularity, and the new measured maps, quoted on the reflected beam.

**A true sun.** Every trace now uses the Buie sunshape — the model the
companion manuscript's own published runs used — with a
circumsolar-ratio setting for the aureole around the disk.
Coddington's results were checked against those published runs across
all three optical layouts and agree to within a tenth of a percent on
concentration. This changes results from 0.1: spots are somewhat
wider and peak flux somewhat lower. That is a correction, not a
regression — v0.1 used a simpler stand-in sun.

**Site DNI**, set as a constant or from a clear-sky model and stated
wherever a result depends on it. Previously most power and flux
numbers silently assumed 1000 W/m² regardless of what was shown
elsewhere.

**Four reference fields** — Gemasolar, PS10, Crescent Dunes, and a
Stellio-based Hami field — ship as built-in projects, each clearly
labelled a reconstruction from published parameters with citations.
None represents an operator's real layout, and each loads as an ideal
build with no optical errors applied.

**Analysis improvements**: a single traced instant is now a
first-class result you can study on its own, not just a step in a
sweep; clicking a day in the year estimate opens it as a full day
sweep; the aperture tool takes typed center and radius instead of
drag-only; footprints open in a full-resolution viewer; and the
irradiance map can color each heliostat in the field by the power it
delivered.

**Better maps.** Curved receivers label their horizontal axis in
compass directions (N/E/S/W); frustums offer a true developed "fan"
view alongside the unrolled rectangle; FEA exports carry real 3-D
coordinates; and a faint grid that used to print over irradiance maps
is gone.

**Faster, and bigger.** Ultra Fast mode is meaningfully quicker.
Fields of up to 15,000 heliostats can now be traced, and the 3-D view
draws a representative subset of very large fields to stay responsive.

**Tooltips throughout**, an optical-error glossary, and heliostat
positions shown and edited in meters instead of millimeters.

**Fixes**: the rays toggle now works reliably and is shared between
the Design and 3D View tabs; dragging to pan the plan or elevation
view no longer selects page text; project save and load, both broken,
work again; and the first trace after starting the app now says it is
starting worker processes instead of appearing to hang.

## Upgrading from 0.1

Because the sunshape changed, numbers from a v0.1 project — spot
size, peak flux, concentration — will differ slightly when you retrace
it under 0.2. Nothing about the saved project itself breaks; the
results simply move to reflect a more accurate sun.

# Coddington 0.1.0 — first public release

Coddington is a heliostat field and solar-tower optical design tool: lay
out a field of mirrors, pick a receiver and optics layout, and see the
traced flux on the receiver in a live 3-D workspace. This is the first
build made available outside the project — downloadable, no Python or
installation of anything else required.

## What's in this release

- **A live 3-D workspace.** Your tower, receiver, and field are visible in
  3-D from the moment the app opens. Editing a mirror, the field layout, or
  the receiver geometry updates the scene immediately; running a trace adds
  real ray paths and flux results on top.
- **Three optics layouts**: prime focus, axicon, and Cassegrain, each with
  its own receiver and tower geometry.
- **Three ray-tracing fidelities**: an ultra-fast analytic mode for quick
  iteration, a fast accurate mode for everyday use, and full Monte Carlo
  for a reference answer.
- **Field-wide tracing**: shading and blocking across the whole field,
  colored per heliostat, plus day sweeps and a day's collected energy.
- **A Heliostat Shape editor** and an **Analysis** tab for sweeps, flux
  maps, and CSV export, alongside the main Workspace tab.
- **Runs entirely on your machine.** The app is a small local web server
  your browser talks to at `http://127.0.0.1` — no data leaves your
  computer, and no internet connection is needed once it's downloaded.

## Installing

Download `Coddington-Setup-0.1.0.exe` from this release and run it. It
installs Coddington with a Start Menu shortcut, an optional desktop
shortcut, and an uninstaller — no other software required.

**About the SmartScreen warning:** this build isn't signed with a paid
code-signing certificate, so Windows will show *"Windows protected your
PC"* the first time you run the installer or the app. This is expected for
an independently-published app, not a sign anything is wrong. Click **More
info**, then **Run anyway** to continue.

See the [README](README.md#run-it) for the full install and launch
instructions, and the [quickstart](docs/index.md) for a tour of the app
itself.

## Known limitations

- **Pre-release API and UI.** Screens, project file formats, and the
  command-line interface will keep moving until v1.0.
- **Flat-window receivers** are the only receiver type validated against
  the reference tracer. Cylindrical and frustum receivers exist and share
  the same interface, but have no fixture coverage yet — treat their
  results as unconfirmed.
- **Windows only, and unsigned** (see the SmartScreen note above). The app
  runs on macOS and Linux from source, but no installer is built for them
  in this release.
- **Console window is the off switch.** Closing the console window that
  opens alongside your browser stops the app; there's no separate quit
  button yet.

## Feedback

This is early — if something looks wrong or a screen doesn't do what you
expect, please open an issue on the
[repository](https://github.com/wilryk/coddington).
