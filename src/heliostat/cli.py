"""Command-line entry point.

Subcommands (``layout``, ``trace``, ``figures``, ``energy``, ``fetch-dni``,
``info``) are added as their modules are ported; ``serve`` (the local web
GUI) is the first one in. Until the rest land, a bare ``heliostat`` with no
subcommand is still the stub it always was: print help and exit 0.
"""

from __future__ import annotations

import argparse
import math

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

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args.host, args.port, open_browser=not args.no_open)
    if args.command == "layout":
        if args.kind == "fermat":
            return _layout_fermat(args)
        layout_parser.print_help()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
