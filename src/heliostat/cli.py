"""Command-line entry point.

Subcommands (``layout``, ``trace``, ``figures``, ``energy``, ``fetch-dni``,
``info``) are added as their modules are ported; ``serve`` (the local web
GUI) is the first one in. Until the rest land, a bare ``heliostat`` with no
subcommand is still the stub it always was: print help and exit 0.
"""

from __future__ import annotations

import argparse

from heliostat import __version__


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

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args.host, args.port, open_browser=not args.no_open)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
