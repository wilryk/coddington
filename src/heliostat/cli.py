"""Command-line entry point.

Subcommands (``layout``, ``trace``, ``figures``, ``energy``, ``fetch-dni``,
``info``) are added as their modules are ported; ``serve`` (the local web
GUI) and ``trace`` (the sweep driver) are in. Until the rest land, a bare
``heliostat`` with no subcommand is still the stub it always was: print help
and exit 0.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import time

from heliostat import __version__


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


def _serve(host: str, port: int, open_browser: bool) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "the 'serve' command needs the web extra: pip install heliostat[web]"
        ) from exc

    from heliostat.web import create_app

    app = create_app()

    if open_browser:
        import threading
        import webbrowser

        def _open() -> None:
            webbrowser.open(f"http://{host}:{port}/")

        # Short delay so the browser doesn't race uvicorn's startup.
        threading.Timer(1.0, _open).start()

    uvicorn.run(app, host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heliostat",
        description="Heliostat-field simulation for concentrating solar towers.",
    )
    parser.add_argument("--version", action="version", version=f"heliostat {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the local web GUI.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    serve_parser.add_argument("--port", type=int, default=8420, help="Bind port (default 8420).")
    serve_parser.add_argument(
        "--no-open", action="store_true", help="Don't open a browser window automatically."
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
        "--base-seed", type=int, default=20260811, help="Monte Carlo base seed (default 20260811)."
    )
    trace_parser.add_argument("-o", "--output", required=True, help="output run directory.")

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args.host, args.port, open_browser=not args.no_open)
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
