# Desktop build

A download-and-run build: it bundles Python and every dependency, so the
target machine needs only a browser. Windows x64 is 181 MB unpacked, 75 MB
zipped.

Double-click `Coddington.exe`; the console window that appears is the off
switch. Run it from a terminal and it takes the same subcommands as a pip
install (`Coddington.exe serve --port 9000`).

## Build

```
pip install -e .[web] pyinstaller
python -m PyInstaller packaging/desktop/coddington.spec --noconfirm
```

Output lands in `dist/Coddington`. PyInstaller does not cross-compile, so
each platform must be built on itself; `.github/workflows/release.yml` does
all three on tag push and smoke-tests each bundle before uploading.

## Known limitations

- **Unsigned.** Windows SmartScreen warns on first run; macOS Gatekeeper
  refuses downloaded builds until signed and notarised. Signing needs
  certificates.
- **One-folder, not one-file.** A one-file build unpacks 181 MB on every
  launch.
- `multiprocessing.freeze_support()` must stay first in the launcher — the
  sweep driver uses a process pool, and without it a frozen app re-executes
  itself in every worker.
