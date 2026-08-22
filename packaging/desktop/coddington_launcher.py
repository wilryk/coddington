"""Entry point for the frozen desktop build.

Running this is the same as typing ``heliostat`` with no arguments: start
the local web app, wait for the port, open a browser, and keep the console
open as the off switch. Everything it does lives in the ordinary CLI, so a
downloaded build and a ``pip install`` behave identically -- there is no
second code path to keep in step.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Without this, a frozen app that starts a multiprocessing Pool -- which
    # heliostat.sweep does -- re-executes the bundle in every worker instead
    # of forking a worker, so launching one app opens an unbounded fan of
    # them. It has to run before anything else touches multiprocessing.
    multiprocessing.freeze_support()

    from heliostat.cli import main as cli_main

    # Forward whatever was passed. A double-clicked icon passes nothing, so
    # that is still "start the app"; but someone who runs the exe from a
    # terminal gets the same subcommands as a pip install, rather than
    # having their arguments silently ignored.
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
