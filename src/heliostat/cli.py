"""Command-line entry point.

Subcommands (``layout``, ``trace``, ``figures``, ``energy``, ``fetch-dni``,
``info``) are added as their modules are ported; until then this is a stub
so the console script installs cleanly.
"""

from __future__ import annotations

import argparse

from heliostat import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heliostat",
        description="Heliostat-field simulation for concentrating solar towers.",
    )
    parser.add_argument("--version", action="version", version=f"heliostat {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
