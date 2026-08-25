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

Download `Coddington-Setup-0.1.0.dev0.exe` from this release and run it. It
installs Coddington with a Start Menu shortcut, an optional desktop
shortcut, and an uninstaller — no other software required. Prefer not to
install anything? The `.zip` build works the same way: unzip it and
double-click `Coddington.exe` inside.

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
- **Unsigned build**, Windows and macOS only for this release (see the
  SmartScreen note above). A Linux build is produced from the same source
  but is not part of this installer.
- **Console window is the off switch.** Closing the console window that
  opens alongside your browser stops the app; there's no separate quit
  button yet.

## Feedback

This is early — if something looks wrong or a screen doesn't do what you
expect, please open an issue on the
[repository](https://github.com/wilryk/coddington).
