# Desktop build

A download-and-run build for people who should not have to install Python
first. It bundles the interpreter and every dependency, so the machine it
lands on needs nothing but a browser.

## What it produces

A folder named `Coddington` containing `Coddington.exe` (or `Coddington` on
macOS and Linux) and its runtime. Double-click it: a console window appears,
the app starts, and a browser opens at it. The console window is the off
switch — close it, or press Ctrl+C.

Measured on Windows x64: **181 MB unpacked, 75 MB zipped**. Most of that is
SciPy and pandas, which the tracer and the read path genuinely use.

Running it from a terminal gives the same subcommands as a pip install
(`Coddington.exe serve --port 9000`, `Coddington.exe layout fermat …`),
because the launcher forwards its arguments to the ordinary CLI rather than
being a second code path.

## Building it

From the repository root, in an environment that has the package and
PyInstaller:

```
pip install -e .[web] pyinstaller
python -m PyInstaller packaging/desktop/coddington.spec --noconfirm
```

The build lands in `dist/Coddington`. Zip that folder to hand it to someone.

**PyInstaller does not cross-compile.** A Windows build must be made on
Windows, a macOS build on macOS, a Linux build on Linux. `.github/workflows/
release.yml` does all three on GitHub's runners when a tag is pushed, which
is the only practical way to publish for platforms you do not have.

## Things worth knowing before shipping one

- **It is unsigned.** Windows SmartScreen will show "Windows protected your
  PC" on first run, and macOS Gatekeeper will refuse a downloaded build
  outright until it is signed and notarised. Signing needs a certificate
  (and, for macOS, an Apple Developer account); until then the honest thing
  is to tell people what they will see and why, rather than to act surprised
  when they report it.
- **One-folder, not one-file.** A one-file build unpacks the whole 181 MB to
  a temporary directory on *every* launch, turning a one-second start into
  twenty. The folder is the better trade for an app people open repeatedly.
- **The console window is deliberate.** A windowless server with no visible
  way to stop it is worse than a small console that says what it is doing
  and closes to quit.
- **`multiprocessing.freeze_support()` is load-bearing.** The sweep driver
  uses a process pool, and a frozen app without that call re-executes the
  whole bundle in every worker — one double-click becomes an unbounded fan
  of apps. It is the first thing the launcher does.

## Verified

The Windows build was launched and driven: it serves the app, and a trace of
the default rectangle returns 8226.0 W with a 505.3 mm rms spot — the same
pinned fixture values the test suite asserts, so the frozen build computes
identical physics to a source checkout. A 6-heliostat day sweep through it
returned 917.6 kWh over 7 timesteps.
