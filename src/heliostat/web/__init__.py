"""Local web GUI: design a heliostat, trace it, see the flux map.

This subpackage's only external dependencies (``fastapi``, ``uvicorn``) are
optional -- installed via ``pip install heliostat[web]``. Importing
:mod:`heliostat.web` itself is cheap and always safe; :func:`create_app` is
where the guarded import happens, so ``import heliostat.web`` never fails
just because the web extra is not installed.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Build and return the FastAPI application (see :mod:`heliostat.web.app`).

    Deferred here so ``import heliostat.web`` stays cheap and does not
    require FastAPI; the real guarded import lives in
    :func:`heliostat.web.app.create_app`.
    """
    from .app import create_app as _create_app

    return _create_app()
