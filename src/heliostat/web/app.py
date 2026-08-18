"""FastAPI application for the local heliostat web GUI.

The first slice: design a heliostat, pick sun position, optics layout and
fidelity mode, trace it, see the flux map. Everything here is a thin HTTP
skin over the existing library -- no new physics, no new geometry.

Optical-configuration geometry (secondary + receiver per ``optics``) is
copied from ``tests/test_mc_parity.py::_geometry_for`` rather than imported
from the test suite (tests are not a stable import surface). Keep the two
in step if that fixture geometry ever changes.

Pointing AND figure come from :mod:`heliostat.geometry.aiming`'s per-layout
solves (:func:`~heliostat.geometry.aiming.solve_prime_focus`,
:func:`~heliostat.geometry.aiming.solve_axicon`,
:func:`~heliostat.geometry.aiming.solve_cassegrain`), evaluated at the
requested sun position and heliostat position -- the same aiming and
focusing solves that reproduce the golden fixtures' pointing/figure
columns to machine precision (``tests/test_aiming.py``). The layout
constants passed to each solve (focus heights, axicon geometry) match
``_geometry_for``'s optics below exactly, since this module traces against
the identical fixture secondaries/receivers.
"""

from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("heliostat.web needs the 'web' extra: pip install heliostat[web]") from exc

from heliostat import __version__
from heliostat.geometry.aiming import Solution, solve_axicon, solve_cassegrain, solve_prime_focus
from heliostat.geometry.design import (
    Flat,
    HeliostatDesign,
    Spherical,
    ZernikeAstig,
    flower,
    grid_facets,
    rect_heliostat,
)
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import AxiconSecondary, CassegrainSecondary, NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat
from heliostat.trace.modes import MODES

STATIC_DIR = Path(__file__).parent / "static"

WINDOW_MM = 2000.0
FLUX_GRID = 128

# Layout constants for the aiming solves, matching _geometry_for's optics
# below exactly -- see tests/test_aiming.py's module docstring for where
# each of these numbers comes from (the private repo's config.toml
# [geometry] defaults plus each layout's own focus_height_mm override).
PRIME_FOCUS_HEIGHT_MM = 35335.0
CASSEGRAIN_FOCUS_HEIGHT_MM = 34892.4  # F1; independent of the physical
# receiver height, which sits behind the hyperboloid relay (see
# solve_cassegrain's docstring) -- _geometry_for's CassegrainSecondary
# below is the identical fixture secondary this height was solved against.
AXICON_APEX_HEIGHT_MM = 27000.0
AXICON_HALF_ANGLE_DEG = 20.0
AXICON_RECEIVER_Z_MM = 7000.0


def _solve_for(
    optics: str, x_mm: float, y_mm: float, solar_az_deg: float, solar_el_deg: float
) -> Solution:
    """Dispatch to this heliostat's pointing + figure solve for ``optics``."""
    if optics == "prime_focus":
        return solve_prime_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, PRIME_FOCUS_HEIGHT_MM)
    if optics == "axicon":
        return solve_axicon(
            x_mm,
            y_mm,
            solar_az_deg,
            solar_el_deg,
            AXICON_APEX_HEIGHT_MM,
            AXICON_HALF_ANGLE_DEG,
            AXICON_RECEIVER_Z_MM,
        )
    if optics == "cassegrain":
        return solve_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg, CASSEGRAIN_FOCUS_HEIGHT_MM)
    raise ValueError(f"unknown optics {optics!r}")  # pragma: no cover - Literal restricts this


# ---------------------------------------------------------------------------
# request models


class RectParams(BaseModel):
    type: Literal["rect"] = "rect"
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class GridParams(BaseModel):
    type: Literal["grid"] = "grid"
    n_u: int = Field(gt=0)
    n_v: int = Field(gt=0)
    facet_w_mm: float = Field(gt=0)
    facet_h_mm: float = Field(gt=0)
    gap_mm: float = Field(default=0.0, ge=0)
    # Blank/absent (None) auto-focuses at the trace's own slant range;
    # explicit 0 opts back into flat -- see _resolved_cant_focal_mm. Must
    # accept 0 for that to be expressible, hence ge=0 rather than gt=0.
    cant_focal_mm: float | None = Field(default=None, ge=0)


class FlowerParams(BaseModel):
    type: Literal["flower"] = "flower"
    n_petals: int = Field(gt=0)
    petal_length_mm: float = Field(gt=0)
    petal_width_mm: float = Field(gt=0)
    hub_radius_mm: float = Field(default=0.0, ge=0)
    cant_focal_mm: float | None = Field(default=None, ge=0)


DesignParams = Annotated[Union[RectParams, GridParams, FlowerParams], Field(discriminator="type")]


class PreviewRequest(BaseModel):
    design: DesignParams


class TraceRequest(BaseModel):
    design: DesignParams
    mode: Literal["ultra_fast", "fast_accurate", "monte_carlo"]
    optics: Literal["prime_focus", "axicon", "cassegrain"]
    solar_az_deg: float = Field(ge=0, le=360)
    solar_el_deg: float
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0

    @field_validator("solar_el_deg")
    @classmethod
    def _elevation_must_be_physical(cls, v: float) -> float:
        # Full rejection of a non-positive sun happens in the endpoint (it
        # needs a friendlier message than a bare pydantic constraint), but
        # anything past straight up is a plain typo -- reject it here.
        if v > 90.0:
            raise ValueError("solar_el_deg must be <= 90")
        return v


# ---------------------------------------------------------------------------
# design construction


def _resolved_cant_focal_mm(explicit: float | None, auto_focal_mm: float | None) -> float | None:
    """Turn a request's ``cant_focal_mm`` into the number (or ``None``, for
    flat) the grid/flower builders want.

    Blank/absent (``None``) auto-focuses at ``auto_focal_mm`` when the
    caller has one -- the trace endpoint computes it from this heliostat's
    solved slant range, so a default trace produces a concentrated spot
    instead of the flat design's mirror-shaped wash. The design-preview
    endpoint has no sun position and so no slant range; it calls this with
    ``auto_focal_mm=None``, so a blank field still previews flat. Explicit
    ``0`` always means flat, whichever endpoint calls this -- the one way
    a caller can opt back out of auto-focus.
    """
    if explicit is None:
        return auto_focal_mm
    if explicit == 0.0:
        return None
    return explicit


def _build_design(
    params: RectParams | GridParams | FlowerParams, auto_focal_mm: float | None = None
) -> HeliostatDesign:
    """Turn a validated param model into a :class:`HeliostatDesign`.

    Builder-level ``ValueError``s (a flower's petal width too wide for its
    length, say) are left to propagate; the endpoint maps them to a 422.

    Rectangle figures are not this function's business -- a rectangle's
    figure depends on a solve (sun position), which this function does not
    take; see the trace endpoint's own rect handling, which calls
    :func:`rect_heliostat` directly. This function's rect branch is a plain
    flat sketch, used only for ``/api/design/preview`` (preview draws
    footprint only, never figure).
    """
    if isinstance(params, RectParams):
        return rect_heliostat(width_mm=params.width_mm, height_mm=params.height_mm)
    if isinstance(params, GridParams):
        cant = _resolved_cant_focal_mm(params.cant_focal_mm, auto_focal_mm)
        surface = Spherical("slant") if cant is not None else None
        return grid_facets(
            n_u=params.n_u,
            n_v=params.n_v,
            facet_w_mm=params.facet_w_mm,
            facet_h_mm=params.facet_h_mm,
            gap_mm=params.gap_mm,
            surface=surface,
            cant_focal_mm=cant,
        )
    cant = _resolved_cant_focal_mm(params.cant_focal_mm, auto_focal_mm)
    surface = Spherical("slant") if cant is not None else None
    return flower(
        n_petals=params.n_petals,
        petal_length_mm=params.petal_length_mm,
        petal_width_mm=params.petal_width_mm,
        hub_radius_mm=params.hub_radius_mm,
        surface=surface,
        cant_focal_mm=cant,
    )


def _design_is_flat(design: HeliostatDesign | None, c3: float, c4: float, c5: float) -> bool:
    """True when the trace's mirror carries no focusing figure at all.

    The legacy path (``design is None``) is flat exactly when the solve's
    own figure is all-zero -- in practice this does not happen for a real
    solve (the defocus term ``c3`` is nonzero for any finite aim distance),
    so this branch is really only reachable in principle. The design path
    is flat when every facet's surface is :class:`Flat` or an all-zero
    :class:`ZernikeAstig` -- the only way there in this module is an
    explicit ``cant_focal_mm=0`` on a grid/flower design (see
    :func:`_resolved_cant_focal_mm`), since rect's non-default-size branch
    and the default grid/flower auto-focus both carry a real figure.
    """
    if design is None:
        return c3 == 0.0 and c4 == 0.0 and c5 == 0.0
    return all(
        isinstance(f.surface, Flat)
        or (
            isinstance(f.surface, ZernikeAstig)
            and f.surface.c3 == 0.0
            and f.surface.c4 == 0.0
            and f.surface.c5 == 0.0
        )
        for f in design.facets
    )


# ---------------------------------------------------------------------------
# optics geometry (copied from tests/test_mc_parity.py::_geometry_for --
# see the module docstring for why this is a copy, not an import)


def _geometry_for(optics: str):
    if optics == "prime_focus":
        secondary = NoSecondary()
        receiver = FlatWindowReceiver(
            z_mm=35335.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="down"
        )
    elif optics == "axicon":
        secondary = AxiconSecondary(
            apex_height_mm=27000.0, half_angle_deg=20.0, aperture_radius_mm=14000.0
        )
        receiver = FlatWindowReceiver(
            z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
    elif optics == "cassegrain":
        secondary = CassegrainSecondary(
            vertex_z_mm=26993.999446877,
            vertex_radius_mm=26112.078893738,
            conic=-5.317616535,
            aperture_radius_mm=14000.0,
        )
        receiver = FlatWindowReceiver(
            z_mm=7000.0, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
    else:  # pragma: no cover - pydantic Literal already restricts this
        raise ValueError(f"unknown optics {optics!r}")
    return secondary, receiver


# ---------------------------------------------------------------------------
# tracing + rendering helpers


def _mc_flux_and_metrics(xy: np.ndarray, watts_per_ray: float, receiver: FlatWindowReceiver):
    """2D-histogram flux map + spot metrics from raw Monte Carlo receiver hits."""
    (u0, u1), (v0, v1) = receiver.uv_extent()
    u_edges = np.linspace(u0, u1, FLUX_GRID + 1)
    v_edges = np.linspace(v0, v1, FLUX_GRID + 1)
    bin_area_m2 = ((u1 - u0) / FLUX_GRID / 1000.0) * ((v1 - v0) / FLUX_GRID / 1000.0)

    counts, _, _ = np.histogram2d(xy[1], xy[0], bins=[v_edges, u_edges])
    flux = counts * watts_per_ray / bin_area_m2  # (n_v, n_u), W/m^2

    if xy.shape[1] == 0:
        return flux, u_edges, v_edges, float("nan"), (float("nan"), float("nan"))

    cen = xy.mean(axis=1)
    r = np.hypot(xy[0] - cen[0], xy[1] - cen[1])
    rms = float(np.sqrt(np.mean(r * r)))
    return flux, u_edges, v_edges, rms, (float(cen[0]), float(cen[1]))


def _cone_metrics(flux: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray):
    """Spot centroid/RMS from a cone-backend flux grid (same recipe as the
    flower cross-backend test in ``tests/test_design_tracing.py``)."""
    u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
    total = flux.sum()
    if total <= 0:
        return float("nan"), (float("nan"), float("nan"))
    cen_u = float((flux.sum(axis=0) * u_mid).sum() / total)
    cen_v = float((flux.sum(axis=1) * v_mid).sum() / total)
    uu, vv = np.meshgrid(u_mid, v_mid)
    rms = float(np.sqrt((((uu - cen_u) ** 2 + (vv - cen_v) ** 2) * flux).sum() / total))
    return rms, (cen_u, cen_v)


def _render_flux_png(
    flux: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray, mode: str, elapsed_ms: float
) -> bytes:
    # Lazy import, same reasoning as HeliostatDesign.preview(): matplotlib
    # is a real dependency but no other endpoint in this module needs it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(
        flux,
        origin="lower",
        cmap="magma",
        extent=(float(u_edges[0]), float(u_edges[-1]), float(v_edges[0]), float(v_edges[-1])),
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("W/m²")
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title(f"{mode}, {elapsed_ms:.0f} ms")
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110)
    finally:
        plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# app


def create_app():
    """Build the FastAPI app. Import-guarded: see the module docstring."""
    app = FastAPI(title="heliostat", version=__version__)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"version": __version__})

    @app.post("/api/design/preview")
    def design_preview(body: PreviewRequest) -> Response:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            design = _build_design(body.design)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        try:
            design.preview(ax=ax)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=110)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            plt.close(fig)

        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/api/trace")
    def trace(body: TraceRequest) -> JSONResponse:
        if body.solar_el_deg <= 0:
            raise HTTPException(
                status_code=422,
                detail="solar_el_deg must be > 0 (the sun is below the horizon)",
            )

        sol = _solve_for(
            body.optics,
            body.heliostat_x_mm,
            body.heliostat_y_mm,
            body.solar_az_deg,
            body.solar_el_deg,
        )
        aim_x_mm = sol.extras["aim_x_mm"]
        aim_y_mm = sol.extras["aim_y_mm"]
        aim_z_mm = sol.extras["aim_z_mm"]
        slant_range_mm = float(
            np.hypot(
                np.hypot(aim_x_mm - body.heliostat_x_mm, aim_y_mm - body.heliostat_y_mm),
                aim_z_mm,
            )
        )

        # Pointing AND figure both come from the solve. Rectangle at the
        # engine's legacy default size (5000x3000 mm) uses the LEGACY
        # single-mirror path (design=None): this is bit-for-bit the
        # validated fixture physics (tests/test_aiming.py,
        # tests/test_design_tracing.py), so it stays the default trace.
        # Any other rectangle size, and every grid/flower design, goes
        # through the generalized facet path instead -- rect's own figure
        # is carried as ZernikeAstig(c3, -c4, -c5) per the sign convention
        # documented in tests/test_design_tracing.py (the legacy path
        # negates c4/c5 internally; a design equivalent to legacy (c3, c4,
        # c5) needs that flip applied up front). Grid/flower auto-focus at
        # this heliostat's own slant range when cant_focal_mm is blank.
        try:
            if isinstance(body.design, RectParams):
                if body.design.width_mm == 5000.0 and body.design.height_mm == 3000.0:
                    design = None
                else:
                    design = rect_heliostat(
                        width_mm=body.design.width_mm,
                        height_mm=body.design.height_mm,
                        surface=ZernikeAstig(sol.c3, -sol.c4, -sol.c5),
                    )
            else:
                design = _build_design(body.design, auto_focal_mm=slant_range_mm)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics)
        rot_az_deg, rot_el_deg = sol.rot_az_deg, sol.rot_el_deg

        mode = MODES[body.mode]
        t0 = time.perf_counter()

        if mode.backend == "mc":
            rng = np.random.default_rng(1)
            result = trace_heliostat(
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                rot_az_deg,
                rot_el_deg,
                sol.c3,
                sol.c4,
                sol.c5,
                body.solar_az_deg,
                body.solar_el_deg,
                secondary,
                receiver,
                mode.n_rays,
                rng,
                source_disk_radius_mm="auto",
                design=design,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            xy = result["xy"]
            counters = result["counters"]
            watts_per_ray = result["watts_per_ray"]
            power_w = watts_per_ray * counters.get("in_window", 0)
            incident_power_w = None
            flux, u_edges, v_edges, rms_mm, centroid = _mc_flux_and_metrics(
                xy, watts_per_ray, receiver
            )
        else:
            kernel = sunshape_kernel("super_gauss")
            cone_kwargs = dict(mode.cone_kwargs)
            if _design_is_flat(design, sol.c3, sol.c4, sol.c5):
                # A deliberately flat mirror (explicit cant_focal_mm=0 on a
                # grid/flower design -- see _design_is_flat) has no
                # focusing figure at all, so the cone backend's per-sample
                # kernels never overlap: at the mode's normal 20x12
                # sampling grid that shows up as a comb/ripple artifact
                # across the flux map (owner-reported). Denser sampling
                # closes the gaps between kernels; only worth the extra
                # cost for this deliberately-flat case, so it is not the
                # mode's own default.
                cone_kwargs["grid"] = (40, 24)
            result = trace_heliostat_cone(
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                rot_az_deg,
                rot_el_deg,
                sol.c3,
                sol.c4,
                sol.c5,
                body.solar_az_deg,
                body.solar_el_deg,
                secondary,
                receiver,
                kernel,
                design=design,
                **cone_kwargs,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            flux = result["flux"]
            u_edges, v_edges = result["u_edges"], result["v_edges"]
            power_w = result["power_w"]
            incident_power_w = result["incident_power_w"]
            counters = result["counters"]
            rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)

        png_bytes = _render_flux_png(flux, u_edges, v_edges, body.mode, elapsed_ms)

        # Standard JSON has no NaN token (JS's JSON.parse rejects it, even
        # though Python's json.dumps happily emits one) -- a landed=0 trace
        # would otherwise produce a response the frontend cannot parse.
        def _clean(x: float) -> float | None:
            return None if x is None or not np.isfinite(x) else float(x)

        return JSONResponse(
            {
                "power_w": _clean(power_w),
                "incident_power_w": _clean(incident_power_w),
                "rms_radius_mm": _clean(rms_mm),
                "centroid_mm": [_clean(centroid[0]), _clean(centroid[1])],
                "counters": {k: int(v) for k, v in counters.items()},
                "elapsed_ms": elapsed_ms,
                "mode": body.mode,
                "flux_png": base64.b64encode(png_bytes).decode("ascii"),
                "aim_point_mm": [aim_x_mm, aim_y_mm, aim_z_mm],
                "slant_range_m": slant_range_mm / 1000.0,
            }
        )

    return app
