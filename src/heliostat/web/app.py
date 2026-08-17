"""FastAPI application for the local heliostat web GUI.

The first slice: design a heliostat, pick sun position, optics layout and
fidelity mode, trace it, see the flux map. Everything here is a thin HTTP
skin over the existing library -- no new physics, no new geometry.

Optical-configuration geometry (secondary + receiver per ``optics``) is
copied from ``tests/test_mc_parity.py::_geometry_for`` rather than imported
from the test suite (tests are not a stable import surface). Keep the two
in step if that fixture geometry ever changes.

Pointing uses :func:`heliostat.geometry.heliostat.heliostat_orientation`
aimed at a fixed point per ``optics``: the shared focus at ``(0, 0, 35335)``
for ``prime_focus``, and ``(0, 0, 27000)`` -- the secondary's approximate
apex/vertex height -- for ``axicon``/``cassegrain``. Field sweeps solve a
proper per-heliostat aim strategy (spillage, canting, tracking); this demo
slice does not, which is why the UI footnotes it as "demo aiming".
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
from heliostat.geometry.design import (
    HeliostatDesign,
    Spherical,
    flower,
    grid_facets,
    rect_heliostat,
)
from heliostat.geometry.heliostat import heliostat_orientation
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import AxiconSecondary, CassegrainSecondary, NoSecondary
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import trace_heliostat
from heliostat.trace.modes import MODES

STATIC_DIR = Path(__file__).parent / "static"

WINDOW_MM = 2000.0
FLUX_GRID = 128

# Aim points for pointing solves, per the owner's demo-slice ruling: exact
# shared focus for prime_focus, the secondary's approximate apex/vertex
# height (a serviceable stand-in, not a solved aim strategy) for the two
# beam-down layouts.
AIM_POINTS = {
    "prime_focus": (0.0, 0.0, 35335.0),
    "axicon": (0.0, 0.0, 27000.0),
    "cassegrain": (0.0, 0.0, 27000.0),
}


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
    cant_focal_mm: float | None = Field(default=None, gt=0)


class FlowerParams(BaseModel):
    type: Literal["flower"] = "flower"
    n_petals: int = Field(gt=0)
    petal_length_mm: float = Field(gt=0)
    petal_width_mm: float = Field(gt=0)
    hub_radius_mm: float = Field(default=0.0, ge=0)
    cant_focal_mm: float | None = Field(default=None, gt=0)


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


def _build_design(params: RectParams | GridParams | FlowerParams) -> HeliostatDesign:
    """Turn a validated param model into a :class:`HeliostatDesign`.

    Builder-level ``ValueError``s (a flower's petal width too wide for its
    length, say) are left to propagate; the endpoint maps them to a 422.
    """
    if isinstance(params, RectParams):
        return rect_heliostat(width_mm=params.width_mm, height_mm=params.height_mm)
    if isinstance(params, GridParams):
        surface = Spherical("slant") if params.cant_focal_mm is not None else None
        return grid_facets(
            n_u=params.n_u,
            n_v=params.n_v,
            facet_w_mm=params.facet_w_mm,
            facet_h_mm=params.facet_h_mm,
            gap_mm=params.gap_mm,
            surface=surface,
            cant_focal_mm=params.cant_focal_mm,
        )
    surface = Spherical("slant") if params.cant_focal_mm is not None else None
    return flower(
        n_petals=params.n_petals,
        petal_length_mm=params.petal_length_mm,
        petal_width_mm=params.petal_width_mm,
        hub_radius_mm=params.hub_radius_mm,
        surface=surface,
        cant_focal_mm=params.cant_focal_mm,
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

        try:
            design = _build_design(body.design)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics)
        aim = AIM_POINTS[body.optics]
        mirror_pos = (body.heliostat_x_mm, body.heliostat_y_mm, 0.0)
        rot_az_deg, rot_el_deg, *_ = heliostat_orientation(
            aim, mirror_pos, body.solar_az_deg, body.solar_el_deg
        )

        mode = MODES[body.mode]
        t0 = time.perf_counter()

        if mode.backend == "mc":
            rng = np.random.default_rng(1)
            result = trace_heliostat(
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                rot_az_deg,
                rot_el_deg,
                0.0,
                0.0,
                0.0,
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
            result = trace_heliostat_cone(
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                rot_az_deg,
                rot_el_deg,
                0.0,
                0.0,
                0.0,
                body.solar_az_deg,
                body.solar_el_deg,
                secondary,
                receiver,
                kernel,
                design=design,
                **mode.cone_kwargs,
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
            }
        )

    return app
