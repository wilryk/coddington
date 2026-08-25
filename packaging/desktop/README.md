# Desktop build

A download-and-run build: it bundles Python and every dependency, so the
target machine needs only a browser. Windows x64 is 181 MB unpacked; the
installer that wraps it is about 47 MB.

Double-click `Coddington.exe`; the console window that appears is the off
switch. Run it from a terminal and it takes the same subcommands as a pip
install (`Coddington.exe serve --port 9000`).

## Build

One command, from a clean checkout to a finished Windows installer:

```
powershell -File scripts\build_release.ps1
```

It creates its own virtual environment, builds the app, launches the
frozen build to confirm it actually serves the workspace with no external
references, then builds the installer if Inno Setup (`iscc`) is available.
`-SkipInstaller` stops after the app bundle; `-ReuseVenv` skips recreating
the virtual environment on a repeat run. See the script's header comment
for details.

To do the same steps by hand:

```
pip install -e .[web] pyinstaller
python -m PyInstaller packaging/desktop/coddington.spec --noconfirm
```

Output lands in `dist/Coddington`. PyInstaller does not cross-compile, so
each platform must be built on itself; `.github/workflows/release.yml` does
all three on tag push and smoke-tests each bundle before uploading.

## Windows installer

`coddington.iss` is an [Inno Setup](https://jrsoftware.org/isinfo.php)
script that wraps `dist/Coddington` into a single-file installer:

```
iscc packaging\desktop\coddington.iss /DMyAppVersion=0.1.0
```

(`scripts/build_release.ps1` passes `/DMyAppVersion` for you, read from
`heliostat.__version__`, and locates `iscc` even when it isn't on `PATH`.)
The installer needs no admin rights — it installs per-user unless run
elevated — and gives a Start Menu shortcut, an optional desktop shortcut
(unchecked by default), and a proper uninstaller. Output lands in
`dist/installer/Coddington-Setup-<version>.exe`.

`coddington.ico` is the app and installer icon, rasterized from
`src/heliostat/web/static/img/coddington-icon.svg`; regenerate it from that
source if the logo changes rather than editing the `.ico` directly.

## Known limitations

- **Unsigned.** Windows SmartScreen warns on first run; macOS Gatekeeper
  refuses downloaded builds until signed and notarised. Signing needs
  certificates.
- **One-folder, not one-file.** A one-file build unpacks 181 MB on every
  launch.
- `multiprocessing.freeze_support()` must stay first in the launcher — the
  sweep driver uses a process pool, and without it a frozen app re-executes
  itself in every worker.
