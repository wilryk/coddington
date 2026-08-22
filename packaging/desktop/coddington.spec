# PyInstaller spec for the downloadable desktop build.
#
# Produces a self-contained folder: Coddington.exe plus its runtime, needing
# no Python on the target machine. One-folder rather than one-file because a
# one-file build unpacks ~300 MB to a temp directory on every launch, which
# turns "double-click and wait a second" into "double-click and wait twenty".
#
# Build (from the repository root, in an environment with the package and
# PyInstaller installed):
#
#     python -m PyInstaller packaging/desktop/coddington.spec --noconfirm
#
# PyInstaller does not cross-compile: a Windows build must be made on
# Windows, a macOS build on macOS, a Linux build on Linux.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# The web app is a package-data payload (one HTML file), not an import, so
# it has to be collected explicitly or the frozen app serves a 500.
datas = collect_data_files("heliostat", includes=["web/static/*"])

hiddenimports = [
    # uvicorn resolves its protocol and lifespan implementations by string
    # at runtime, so static analysis never sees them.
    *collect_submodules("uvicorn"),
    # matplotlib's Agg backend is selected by name inside the request
    # handlers, for the same reason.
    "matplotlib.backends.backend_agg",
    # Reading .xlsx fields; pandas imports it lazily by name.
    "openpyxl",
]

excludes = [
    # Nothing in the app draws with Tk, and it pulls in a large runtime.
    "tkinter",
    # Test and notebook machinery that a shipped app has no use for.
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
]

block_cipher = None

a = Analysis(
    # SPECPATH is where this file lives; PyInstaller resolves relative
    # paths against it, not against the working directory.
    [os.path.join(SPECPATH, "coddington_launcher.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Coddington",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True on purpose: the window is the off switch, and a windowless
    # server with no visible way to stop it is worse than a small console.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Coddington",
)
