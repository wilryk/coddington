"""Command-line entry point.

Two audiences share one executable. Someone who types ``heliostat`` with no
arguments -- or double-clicks ``heliostat.exe`` from a desktop shortcut,
which is the same thing with an empty ``argv`` -- wants the web app, so that
is what an empty argv does: pick a port, print a short banner, open a
browser, serve. Someone who types a subcommand wants the batch tool, and
every subcommand (``serve``, ``layout``, ``trace``, ``shortcut``) behaves
exactly as it always did. ``--help``, ``-h`` and ``--version`` are argv, so
they are never the launcher.

The launcher stays a *console* script on purpose: the console window is the
off switch. A windowless entry point would leave a server running with no
obvious way to stop it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path

from heliostat import __version__

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420
#: Last port the automatic scan will try before giving up (inclusive).
PORT_SCAN_LAST = 8439


def _layout_fermat(args: argparse.Namespace) -> int:
    from heliostat.field_layouts import (
        generate,
        min_spacing_filter,
        ring_filter,
        road_corridors,
        wedge_filter,
        write_field_csv,
    )

    filters = []
    if args.wedge is not None:
        filters.append(wedge_filter(args.wedge[0], args.wedge[1]))
    if args.ring is not None:
        filters.append(ring_filter(args.ring[0], args.ring[1]))
    if args.road_width_m is not None:
        filters.append(road_corridors(args.road_width_m, azimuths_deg=tuple(args.road_az_deg)))
    if args.min_spacing_m is not None:
        filters.append(min_spacing_filter(args.min_spacing_m))

    params = {"a_m": args.a_m, "b": args.b, "k_start": args.k_start}
    if args.divergence is not None:
        params["divergence_rad"] = args.divergence

    field = generate("fermat", args.n, filters=tuple(filters), oversample=args.oversample, **params)
    write_field_csv(field, args.output)

    r_m = field.radius_mm / 1000.0
    r_min, r_max = float(r_m.min()), float(r_m.max())

    # Land-area coverage fraction, MATLAB's `ratlh`. Judgment call: this
    # module carries no mirror geometry (that's heliostat/aperture), so the
    # footprint used here is a CLI-only, reporting-only assumption -- a
    # square of side --mirror-width-m -- not a value stored on the field or
    # used anywhere else.
    land_r_min, land_r_max = (
        (args.ring[0], args.ring[1]) if args.ring is not None else (r_min, r_max)
    )
    angular_fraction = 1.0
    if args.wedge is not None:
        span = (args.wedge[1] - args.wedge[0]) % 360.0
        angular_fraction = (span or 360.0) / 360.0
    land_area_m2 = angular_fraction * math.pi * (land_r_max**2 - land_r_min**2)
    helio_area_m2 = len(field) * (args.mirror_width_m**2)
    coverage = helio_area_m2 / land_area_m2 if land_area_m2 > 0 else float("nan")

    print(
        f"{len(field)} heliostats, r {r_min:.1f}-{r_max:.1f} m, "
        f"land coverage {coverage * 100:.2f}% (assumed {args.mirror_width_m:.1f} m square mirrors)"
    )
    return 0


def _parse_date(value: str) -> _dt.date:
    return _dt.datetime.strptime(value, "%Y-%m-%d").date()


def _trace(args: argparse.Namespace) -> int:
    from heliostat.field import load_field
    from heliostat.sweep import DEFAULT_SITE, run_sweep

    field = load_field(
        args.field, mirror_width_mm=args.mirror_width_mm, mirror_height_mm=args.mirror_height_mm
    )
    dates = [_parse_date(d) for d in args.date]
    lat = DEFAULT_SITE[0] if args.lat is None else args.lat
    lon = DEFAULT_SITE[1] if args.lon is None else args.lon
    tz = DEFAULT_SITE[2] if args.tz is None else args.tz

    t_start = time.perf_counter()
    store = run_sweep(
        field,
        dates,
        mode=args.mode,
        optics=args.optics,
        site=(lat, lon, tz),
        out_dir=args.output,
        workers=args.workers,
        n_rays=args.rays,
        base_seed=args.base_seed,
        hour_step=args.hour_step,
        sunrise_margin_min=args.sunrise_margin_min,
        min_elevation_deg=args.min_elevation_deg,
        progress=print,
    )
    elapsed = time.perf_counter() - t_start
    n_steps = len(store.timestep_keys())
    heliostat_steps = n_steps * len(field)
    print(
        f"done in {elapsed:.1f}s, {heliostat_steps} heliostat-steps "
        f"({n_steps} timesteps x {len(field)} heliostats), store at {store.root}"
    )
    return 0


class PortBusyError(RuntimeError):
    """A port the user explicitly asked for is already in use."""


class NoFreePortError(RuntimeError):
    """Every port in the automatic scan range is in use."""


def _connect_host(host: str) -> str:
    """The address to *connect* to for a server bound to ``host``.

    A wildcard bind is not a connectable address on every platform, so the
    readiness poll and the browser URL both aim at the loopback instead.
    """
    if host in ("0.0.0.0", "", "::", "*"):
        return "127.0.0.1"
    return host


def port_is_free(host: str, port: int) -> bool:
    """True if a TCP server can bind ``host:port`` right now.

    Deliberately no ``SO_REUSEADDR``: on Windows that option lets a second
    bind succeed on a port already in use, which would answer a different
    question than the one uvicorn is about to ask.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(host: str, start: int = DEFAULT_PORT, last: int = PORT_SCAN_LAST) -> int:
    """First free port in ``[start, last]``, or raise :class:`NoFreePortError`."""
    for port in range(start, last + 1):
        if port_is_free(host, port):
            return port
    raise NoFreePortError(f"no free port between {start} and {last}")


def resolve_port(host: str, requested: int | None) -> int:
    """Decide which port to serve on.

    ``requested is None`` means "the default, or the next one up if it is
    busy" -- the no-arguments launcher must not fail just because something
    else already owns 8420. An explicit ``--port`` is an instruction, not a
    preference, so a busy one raises instead of quietly moving.
    """
    if requested is None:
        return find_free_port(host)
    if not port_is_free(host, requested):
        raise PortBusyError(f"port {requested} is already in use")
    return requested


def _open_browser_when_ready(
    url: str,
    host: str,
    port: int,
    *,
    timeout: float = 30.0,
    interval: float = 0.25,
    opener: Callable[[str], object] = webbrowser.open,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    """Open ``url`` once the server actually accepts connections.

    Polls with a real TCP connect rather than sleeping a fixed amount, so a
    slow first import cannot land the browser on a connection-refused page.
    Returns True if the browser was opened, False if the server never came
    up within ``timeout`` -- in which case nothing is opened at all, because
    a browser window showing an error is worse than no window.
    """
    target = _connect_host(host)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((target, port), timeout=1.0):
                pass
        except OSError:
            if time.monotonic() >= deadline:
                return False
            sleeper(interval)
            continue
        opener(url)
        return True


def _console_safe(text: str) -> str:
    """Downgrade the banner's dash unless the console clearly handles it.

    Only a UTF-8 stream is trusted with the em dash. A legacy code page can
    often *encode* it while whatever reads the output decodes the byte as
    something else, and a redirected or frozen-app stream may report no
    encoding at all -- both show up as a mojibake character in the one
    message a user is guaranteed to read.
    """
    encoding = getattr(sys.stdout, "encoding", None) or ""
    if encoding.lower().replace("-", "") not in {"utf8", "utf8sig"}:
        return text.replace("—", "-")
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.replace("—", "-")
    return text


def _banner(url: str, *, opening_browser: bool, moved_from: int | None) -> str:
    """The short human-facing block printed before uvicorn's own logging."""
    browser_note = "  (opening your browser)" if opening_browser else ""
    lines = [
        f"heliostat {__version__} — starting the web app",
        f"  {url}{browser_note}",
    ]
    if moved_from is not None:
        lines.append(f"  (port {moved_from} was busy, so it moved up to the next free one.)")
    lines.append("  Close this window or press Ctrl+C to stop.")
    lines.append("  Batch commands: heliostat --help")
    return "\n".join(lines)


_MISSING_WEB_EXTRA = (
    "heliostat: the web app needs the optional 'web' extra, which is not installed.\n"
    'Install it with:  pip install "heliostat[web]"'
)


def _serve(host: str, port: int | None, open_browser: bool) -> int:
    """Run the web app. ``port=None`` means "default port, or the next free one"."""
    try:
        import uvicorn

        from heliostat.web.app import create_app
    except ImportError:
        # A traceback here would be noise: the fix is one pip command, and
        # the person seeing this may well have double-clicked an icon.
        print(_MISSING_WEB_EXTRA, file=sys.stderr)
        return 1

    try:
        chosen = resolve_port(host, port)
    except PortBusyError as exc:
        print(f"heliostat: {exc}.", file=sys.stderr)
        print("Pick another with --port, or omit --port to scan for a free one.", file=sys.stderr)
        return 1
    except NoFreePortError as exc:
        print(f"heliostat: {exc}.", file=sys.stderr)
        print("Free one of those ports, or choose your own with --port.", file=sys.stderr)
        return 1

    url = f"http://{_connect_host(host)}:{chosen}/"
    moved_from = DEFAULT_PORT if (port is None and chosen != DEFAULT_PORT) else None
    print(_console_safe(_banner(url, opening_browser=open_browser, moved_from=moved_from)))
    print(flush=True)

    app = create_app()

    if open_browser:
        import threading

        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, host, chosen),
            daemon=True,
        ).start()

    uvicorn.run(app, host=host, port=chosen)
    return 0


class ShortcutError(RuntimeError):
    """A desktop launcher could not be created (reason is the message)."""


def heliostat_executable() -> Path | None:
    """Locate the installed ``heliostat`` program.

    The running interpreter's own bin directory first: a shortcut should
    point at the installation that is creating it, and when several
    installs coexist (a venv invoked by full path, an older system-Python
    install still on ``PATH``), ``shutil.which`` would silently pick the
    wrong one. ``which`` remains the fallback for layouts where the
    console script does not sit beside the interpreter.
    """
    bindir = Path(sys.executable).resolve().parent
    for candidate in (
        bindir / "heliostat.exe",
        bindir / "heliostat",
        bindir / "Scripts" / "heliostat.exe",
        bindir / "bin" / "heliostat",
    ):
        if candidate.is_file():
            return candidate
    found = shutil.which("heliostat")
    if found:
        return Path(found).resolve()
    return None


# One PowerShell call does the whole Windows job: it resolves the Desktop
# through the shell folder API (so a OneDrive-redirected Desktop is found,
# which %USERPROFILE%\Desktop would miss), enforces the no-clobber rule, and
# creates the .lnk through WScript.Shell -- no extra Python dependency.
# Paths travel in the environment rather than the command line so that
# spaces and quotes in them cannot be re-parsed by PowerShell.
_PS_MAKE_SHORTCUT = r"""
$ErrorActionPreference = 'Stop'
$target = $env:HELIOSTAT_SHORTCUT_TARGET
$dir = $env:HELIOSTAT_SHORTCUT_DIR
if ([string]::IsNullOrEmpty($dir)) { $dir = [Environment]::GetFolderPath('Desktop') }
if (-not (Test-Path -LiteralPath $dir)) { Write-Output "ERR:nodir:$dir"; exit 3 }
$lnk = Join-Path $dir 'heliostat.lnk'
if ((Test-Path -LiteralPath $lnk) -and ($env:HELIOSTAT_SHORTCUT_FORCE -ne '1')) {
    Write-Output "ERR:exists:$lnk"
    exit 4
}
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $target
$sc.WorkingDirectory = Split-Path -Parent $target
$sc.Description = 'Start the heliostat web app'
$sc.Save()
Write-Output "OK:$lnk"
"""


def _powershell() -> str:
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        raise ShortcutError("PowerShell was not found, so a .lnk cannot be created.")
    return exe


def _create_windows_shortcut(exe: Path, directory: Path | None, force: bool) -> Path:
    env = dict(os.environ)
    env["HELIOSTAT_SHORTCUT_TARGET"] = str(exe)
    env["HELIOSTAT_SHORTCUT_DIR"] = str(directory) if directory is not None else ""
    env["HELIOSTAT_SHORTCUT_FORCE"] = "1" if force else "0"
    proc = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _PS_MAKE_SHORTCUT,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    result = lines[-1] if lines else ""
    if result.startswith("OK:"):
        return Path(result[3:])
    if result.startswith("ERR:exists:"):
        existing = result.removeprefix("ERR:exists:")
        raise ShortcutError(f"{existing} already exists; pass --force to replace it.")
    if result.startswith("ERR:nodir:"):
        raise ShortcutError(f"{result.removeprefix('ERR:nodir:')} is not a directory.")
    detail = (proc.stderr.strip() or proc.stdout.strip() or "no output").splitlines()[0]
    raise ShortcutError(f"PowerShell could not create the shortcut: {detail}")


def macos_command_text(exe: Path) -> str:
    """Contents of the ``.command`` file macOS runs on double-click.

    ``as_posix`` so the text is a pure function of the path on every
    platform -- these two writers can then be unit-tested from Windows even
    though they only ever run elsewhere.
    """
    return f'#!/bin/sh\nexec "{exe.as_posix()}"\n'


def linux_desktop_text(exe: Path) -> str:
    """Contents of the freedesktop ``.desktop`` entry.

    ``Terminal=true`` on purpose: the terminal window it opens is how the
    user stops the server again.
    """
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Heliostat\n"
        "Comment=Start the heliostat web app\n"
        f'Exec="{exe.as_posix()}"\n'
        "Terminal=true\n"
        "Categories=Science;Education;\n"
    )


def _write_posix_launcher(path: Path, text: str, force: bool) -> Path:
    if not path.parent.is_dir():
        raise ShortcutError(f"{path.parent} is not a directory.")
    if path.exists() and not force:
        raise ShortcutError(f"{path} already exists; pass --force to replace it.")
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _shortcut(args: argparse.Namespace) -> int:
    exe = heliostat_executable()
    if exe is None:
        print(
            "heliostat: could not find the installed 'heliostat' program to point at.",
            file=sys.stderr,
        )
        print(
            'Install it with:  pip install "heliostat[web]"  '
            "(or activate the environment it is installed in), then try again.",
            file=sys.stderr,
        )
        return 1

    directory = Path(args.path).expanduser().resolve() if args.path else None
    try:
        if sys.platform == "win32":
            created = _create_windows_shortcut(exe, directory, args.force)
        elif sys.platform == "darwin":
            target_dir = directory if directory is not None else Path.home() / "Desktop"
            created = _write_posix_launcher(
                target_dir / "Heliostat.command", macos_command_text(exe), args.force
            )
        elif sys.platform.startswith("linux"):
            target_dir = directory if directory is not None else Path.home() / "Desktop"
            created = _write_posix_launcher(
                target_dir / "heliostat.desktop", linux_desktop_text(exe), args.force
            )
        else:
            print(f"heliostat: no launcher recipe for this platform ({sys.platform}).")
            print(f"The program itself is at: {exe}")
            print("Make a shortcut that runs it, or just type 'heliostat' in a terminal.")
            return 1
    except ShortcutError as exc:
        print(f"heliostat: {exc}", file=sys.stderr)
        return 1

    print("Created a double-clickable launcher:")
    print(f"  {created}")
    print(f"  -> {exe}")
    print("Double-click it to start the web app.")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="heliostat",
        description="Heliostat-field simulation for concentrating solar towers.",
    )
    parser.add_argument("--version", action="version", version=f"heliostat {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve", help="Run the local web app (same as running 'heliostat' with no arguments)."
    )
    serve_parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind host (default {DEFAULT_HOST})."
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Bind port. Omit to use {DEFAULT_PORT}, or the next free port up to "
        f"{PORT_SCAN_LAST} if it is busy; an explicit port that is busy is an error.",
    )
    serve_parser.add_argument(
        "--no-browser",
        "--no-open",
        action="store_true",
        dest="no_browser",
        help="Don't open a browser window automatically ('--no-open' is an alias).",
    )

    shortcut_parser = subparsers.add_parser(
        "shortcut", help="Create a double-clickable launcher for the web app on your Desktop."
    )
    shortcut_parser.add_argument(
        "--path",
        default=None,
        help="directory to create the launcher in (default: your Desktop).",
    )
    shortcut_parser.add_argument(
        "--force", action="store_true", help="replace an existing launcher of the same name."
    )

    layout_parser = subparsers.add_parser(
        "layout", help="Generate a synthetic heliostat field layout."
    )
    layout_sub = layout_parser.add_subparsers(dest="kind")

    fermat_parser = layout_sub.add_parser("fermat", help="Golden-ratio Fermat spiral layout.")
    fermat_parser.add_argument("--n", type=int, required=True, help="target heliostat count.")
    fermat_parser.add_argument(
        "--a", type=float, required=True, dest="a_m", help="spiral scale a in r = a * k**b, metres."
    )
    fermat_parser.add_argument(
        "--b", type=float, default=0.5, help="spiral exponent b (default 0.5)."
    )
    fermat_parser.add_argument(
        "--k-start", type=int, default=1, dest="k_start", help="first k value (default 1)."
    )
    fermat_parser.add_argument(
        "--divergence",
        type=float,
        default=None,
        help="override the divergence angle between k and k+1, radians (default: golden angle).",
    )
    fermat_parser.add_argument(
        "--wedge",
        type=float,
        nargs=2,
        metavar=("AZ_MIN_DEG", "AZ_MAX_DEG"),
        default=None,
        help="keep only the angular wedge [AZ_MIN_DEG, AZ_MAX_DEG] (math convention, atan2(y,x)).",
    )
    fermat_parser.add_argument(
        "--ring",
        type=float,
        nargs=2,
        metavar=("R_MIN_M", "R_MAX_M"),
        default=None,
        help="keep only radius in [R_MIN_M, R_MAX_M] metres.",
    )
    fermat_parser.add_argument(
        "--road-width",
        type=float,
        default=None,
        dest="road_width_m",
        help="half-width of a road corridor to remove, metres (needs --road-az).",
    )
    fermat_parser.add_argument(
        "--road-az",
        type=float,
        nargs="+",
        default=[180.0],
        dest="road_az_deg",
        help="compass azimuth(s) (deg, 0=+y/north, clockwise) of road corridors "
        "(default 180 = south).",
    )
    fermat_parser.add_argument(
        "--min-spacing",
        type=float,
        default=None,
        dest="min_spacing_m",
        help="minimum spacing between kept heliostats, metres.",
    )
    fermat_parser.add_argument(
        "--oversample",
        type=float,
        default=1.6,
        help="candidate-to-target ratio before filtering/truncation (default 1.6).",
    )
    fermat_parser.add_argument(
        "--mirror-width-m",
        type=float,
        default=6.0,
        help="assumed square mirror footprint for the coverage-fraction summary line only; "
        "does not affect generated positions (default 6.0, matching the MATLAB flat-to-flat).",
    )
    fermat_parser.add_argument("-o", "--output", required=True, help="output CSV path.")

    trace_parser = subparsers.add_parser(
        "trace", help="Trace a heliostat field across one or more days and write a stored run."
    )
    trace_parser.add_argument("--field", required=True, help="heliostat position CSV/XLSX.")
    trace_parser.add_argument(
        "--date",
        action="append",
        required=True,
        metavar="YYYY-MM-DD",
        help="trace date; repeat for multiple dates.",
    )
    trace_parser.add_argument(
        "--mode",
        choices=("ultra_fast", "fast_accurate", "monte_carlo"),
        default="ultra_fast",
        help="fidelity mode (default ultra_fast).",
    )
    trace_parser.add_argument(
        "--optics",
        choices=("prime_focus", "axicon", "cassegrain"),
        default="prime_focus",
        help="standard optical configuration (default prime_focus).",
    )
    trace_parser.add_argument("--lat", type=float, default=None, help="site latitude, deg.")
    trace_parser.add_argument("--lon", type=float, default=None, help="site longitude, deg.")
    trace_parser.add_argument("--tz", type=float, default=None, help="site timezone offset, hours.")
    trace_parser.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: cpu count)."
    )
    trace_parser.add_argument(
        "--rays", type=int, default=None, help="rays per heliostat (monte_carlo mode only)."
    )
    trace_parser.add_argument(
        "--mirror-width-mm", type=float, default=5000.0, help="mirror width, mm (default 5000)."
    )
    trace_parser.add_argument(
        "--mirror-height-mm", type=float, default=3000.0, help="mirror height, mm (default 3000)."
    )
    trace_parser.add_argument(
        "--hour-step",
        type=float,
        default=1.0,
        help="max hour spacing between timesteps (default 1.0).",
    )
    trace_parser.add_argument(
        "--sunrise-margin-min",
        type=float,
        default=10.0,
        help="minutes trimmed off sunrise/sunset (default 10).",
    )
    trace_parser.add_argument(
        "--min-elevation-deg",
        type=float,
        default=5.0,
        help=(
            "skip timesteps below this sun elevation, degrees (default 5.0); "
            "the integration window is shrunk to the elevation crossing, not "
            "just filtered, so the energy integral is not biased -- see "
            "heliostat.solar.build_time_grid's docstring."
        ),
    )
    trace_parser.add_argument(
        "--base-seed", type=int, default=20260811, help="Monte Carlo base seed (default 20260811)."
    )
    trace_parser.add_argument("-o", "--output", required=True, help="output run directory.")

    if not argv:
        # Truly empty argv only: no subcommand, no flags. This is what a
        # desktop shortcut or a double-clicked heliostat.exe sends, and what
        # someone who just types the name means. Anything else -- including
        # '--help' and '--version' -- goes to argparse untouched.
        return _serve(DEFAULT_HOST, None, open_browser=True)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args.host, args.port, open_browser=not args.no_browser)
    if args.command == "shortcut":
        return _shortcut(args)
    if args.command == "trace":
        return _trace(args)
    if args.command == "layout":
        if args.kind == "fermat":
            return _layout_fermat(args)
        layout_parser.print_help()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
