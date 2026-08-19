"""FastAPI application for the local heliostat web GUI.

The first slice: design a heliostat, pick sun position, optics layout and
fidelity mode, trace it, see the flux map. Everything here is a thin HTTP
skin over the existing library -- no new physics, no new geometry.

Every trace response also carries a ``scene`` object -- facet outlines,
secondary profile, receiver window, sun vector and real ray paths -- for
the browser's 3-D view. It is built by :mod:`heliostat.web.scene` from the
same values the trace was given and is strictly additive: nothing in it
feeds back into the reported metrics or flux map.

Optical-configuration geometry (secondary + receiver per ``optics``) is
copied from ``tests/test_mc_parity.py::_geometry_for`` rather than imported
from the test suite (tests are not a stable import surface). Keep the two
in step if that fixture geometry ever changes. That fixture geometry is
only the *default*: a trace request may carry an ``optics_params`` object
overriding the tower numbers (see :class:`PrimeFocusOptics`,
:class:`AxiconOptics`, :class:`CassegrainOptics`). Whatever it resolves to
is fed to :func:`_solve_for` and :func:`_geometry_for` together, so the
aim-point solve and the traced geometry are always the same numbers --
that invariant is the whole reason both take the resolved object rather
than reading module constants of their own. The resolved values come back
on the response as ``optics_resolved``.

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
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("heliostat.web needs the 'web' extra: pip install heliostat[web]") from exc

from heliostat import __version__
from heliostat.geometry.aiming import Solution, solve_axicon, solve_cassegrain, solve_prime_focus
from heliostat.geometry.design import (
    Flat,
    HeliostatDesign,
    Spherical,
    Surface,
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
from heliostat.web.scene import build_scene

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
AXICON_APERTURE_RADIUS_MM = 14000.0
AXICON_RECEIVER_Z_MM = 7000.0

# The Cassegrain relay's own conic constants. Fixed, not exposed: they were
# solved together with CASSEGRAIN_FOCUS_HEIGHT_MM and the 7000 mm receiver,
# and only that triple describes a hyperboloid stigmatic between the two.
CASSEGRAIN_VERTEX_Z_MM = 26993.999446877
CASSEGRAIN_VERTEX_RADIUS_MM = 26112.078893738
CASSEGRAIN_CONIC = -5.317616535
CASSEGRAIN_APERTURE_RADIUS_MM = 14000.0
CASSEGRAIN_RECEIVER_Z_MM = 7000.0


# ---------------------------------------------------------------------------
# optics geometry parameters
#
# Every field defaults to the module constant above, so an absent
# ``optics_params`` object resolves to exactly the geometry this app traced
# before these models existed. The models are the single source of truth for
# both halves of a trace: _solve_for reads them for the aim point, and
# _geometry_for reads the same object for the secondary/receiver the rays
# actually meet. Pointing solved against one tower and rays traced against
# another is the failure mode this design exists to make impossible.


class PrimeFocusOptics(BaseModel):
    """Receiver sitting directly at the field's common focus.

    ``focus_height_mm`` is one number doing two jobs, and that is physical
    rather than a shortcut: for this layout the aim point *is* the receiver
    (see :func:`~heliostat.geometry.aiming.solve_prime_focus`), so there is
    no second height to disagree with.
    """

    model_config = ConfigDict(extra="forbid")

    focus_height_mm: float = Field(default=PRIME_FOCUS_HEIGHT_MM, gt=0)
    window_half_u_mm: float = Field(default=WINDOW_MM, gt=0)
    window_half_v_mm: float = Field(default=WINDOW_MM, gt=0)


class AxiconOptics(BaseModel):
    """Conical secondary above the field, flat receiver on the ground below.

    Field names match :class:`~heliostat.geometry.secondary.AxiconSecondary`'s
    and :func:`~heliostat.geometry.aiming.solve_axicon`'s own arguments, so
    the number a caller types is the number both of them read.
    """

    model_config = ConfigDict(extra="forbid")

    apex_height_mm: float = Field(default=AXICON_APEX_HEIGHT_MM, gt=0)
    half_angle_deg: float = Field(default=AXICON_HALF_ANGLE_DEG, gt=0, lt=90)
    aperture_radius_mm: float = Field(default=AXICON_APERTURE_RADIUS_MM, gt=0)
    receiver_z_mm: float = Field(default=AXICON_RECEIVER_Z_MM, gt=0)
    window_half_u_mm: float = Field(default=WINDOW_MM, gt=0)
    window_half_v_mm: float = Field(default=WINDOW_MM, gt=0)

    @model_validator(mode="after")
    def _receiver_below_the_cone(self) -> "AxiconOptics":
        # solve_axicon's whole construction is "the beam travels down from
        # the cone to the receiver": drop = apex_height_mm - receiver_z_mm.
        # A receiver at or above the apex makes that drop zero or negative
        # and the solve returns a plausible-looking aim point for a tower
        # that cannot exist.
        if self.receiver_z_mm >= self.apex_height_mm:
            raise ValueError(
                "receiver_z_mm must be below apex_height_mm -- the axicon "
                "reflects the beam downward onto a ground receiver"
            )
        return self


# Position/shape fields a caller might reasonably try to send for the
# Cassegrain layout, each of which would silently invalidate the relay
# solve. Named explicitly so the rejection can say *why* rather than
# leaving pydantic's generic "extra inputs are not permitted".
_CASSEGRAIN_FIXED_FIELDS = (
    "apex_height_mm",
    "aperture_radius_mm",
    "conic",
    "focus_height_mm",
    "half_angle_deg",
    "receiver_z_mm",
    "vertex_radius_mm",
    "vertex_z_mm",
    "z_mm",
)


class CassegrainOptics(BaseModel):
    """Hyperboloid relay: window size only, positions deliberately fixed.

    The hyperboloid's vertex, vertex radius and conic constant were solved
    for one specific ``(F1, receiver z)`` pair -- that is what makes the
    relay stigmatic between the two. Move either point and the stored conic
    constants no longer describe a hyperboloid with those foci, so the
    trace would still run and would quietly be wrong. Re-solving the relay
    is out of scope for this app, so position fields are rejected outright
    rather than accepted and ignored.
    """

    model_config = ConfigDict(extra="forbid")

    window_half_u_mm: float = Field(default=WINDOW_MM, gt=0)
    window_half_v_mm: float = Field(default=WINDOW_MM, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _reject_position_fields(cls, data):
        if isinstance(data, dict):
            offending = sorted(k for k in data if k in _CASSEGRAIN_FIXED_FIELDS)
            if offending:
                raise ValueError(
                    "cassegrain geometry is fixed: the hyperboloid relay (vertex, "
                    "vertex radius, conic) was solved for one specific focus/receiver "
                    "pair, and moving " + ", ".join(offending) + " would need that "
                    "relay re-solved -- which this app does not do. Only "
                    "window_half_u_mm and window_half_v_mm are adjustable here."
                )
        return data


OpticsParams = Union[PrimeFocusOptics, AxiconOptics, CassegrainOptics]

_OPTICS_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "prime_focus": PrimeFocusOptics,
    "axicon": AxiconOptics,
    "cassegrain": CassegrainOptics,
}


def resolve_optics_params(optics: str, raw: dict | None) -> OpticsParams:
    """Validate a request's ``optics_params`` against ``optics``'s own model.

    ``None`` (an absent object) resolves to that model's defaults, which are
    the module constants above -- so a request that says nothing about the
    tower gets byte-for-byte the geometry this app has always traced.

    Raises :class:`ValueError` with a flattened, human-readable message; the
    endpoint turns that into a 422.
    """
    model = _OPTICS_PARAM_MODELS[optics]
    try:
        return model.model_validate(raw if raw is not None else {})
    except ValidationError as exc:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        raise ValueError(f"optics_params for {optics!r} -- " + "; ".join(parts)) from exc


def _solve_for(
    optics: str,
    x_mm: float,
    y_mm: float,
    solar_az_deg: float,
    solar_el_deg: float,
    params: OpticsParams | None = None,
) -> Solution:
    """Dispatch to this heliostat's pointing + figure solve for ``optics``.

    ``params`` is the resolved geometry object (``None`` means "the
    defaults"); it must be the *same* object handed to :func:`_geometry_for`
    for the same trace.
    """
    if params is None:
        params = resolve_optics_params(optics, None)
    if optics == "prime_focus":
        return solve_prime_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, params.focus_height_mm)
    if optics == "axicon":
        return solve_axicon(
            x_mm,
            y_mm,
            solar_az_deg,
            solar_el_deg,
            params.apex_height_mm,
            params.half_angle_deg,
            params.receiver_z_mm,
        )
    if optics == "cassegrain":
        # F1 is a property of the relay, not of the window, so it stays the
        # module constant -- CassegrainOptics has no field that could move it.
        return solve_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg, CASSEGRAIN_FOCUS_HEIGHT_MM)
    raise ValueError(f"unknown optics {optics!r}")  # pragma: no cover - Literal restricts this


# ---------------------------------------------------------------------------
# request models


class _DesignBase(BaseModel):
    """Shared design fields -- in practice, the mirror's optical figure.

    **Two independent axes, and mixing them up is the easy mistake.**

    ``surface`` is the *optical figure* carried by each facet: the sag and
    slope of its own reflecting surface, which is what curves the light.
    Every design type has one.

    ``cant_focal_mm`` (grid and flower only, declared on those models) is
    *facet aiming*: the out-of-plane tilt that swings a whole facet so its
    reflection points at the design's focal point. It does not curve
    anything -- a canted flat facet is still flat, it just looks somewhere
    else. It also stays in charge under every ``surface`` setting, so
    ``surface="flat"`` means "flat facets, still canted wherever
    ``cant_focal_mm`` says", not "flat and parallel".

    ``surface`` values:

    * ``"adaptive"`` (default) -- whatever solve-driven figure this app
      judges best for the design type, i.e. exactly what it did before this
      field existed. For a rectangle that is the aiming solve's own
      astigmatic figure; for a grid or flower it is *spherical* facets
      auto-focused at the heliostat's slant range (blank ``cant_focal_mm``)
      or at the given focal. "Adaptive" names the choice being made for
      you, not a distinct kind of surface.
    * ``"spherical"`` -- a spherical cap on every facet (or on the single
      rectangle), at the resolved cant focal: blank ``cant_focal_mm`` means
      this heliostat's own slant range, an explicit value means that focal.
      A rectangle has no ``cant_focal_mm``, so it always figures at slant
      range.
    * ``"flat"`` -- no figure at all, anywhere. Expect a mirror-shaped wash
      rather than a spot.
    """

    surface: Literal["adaptive", "spherical", "flat"] = "adaptive"


class RectParams(_DesignBase):
    type: Literal["rect"] = "rect"
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class GridParams(_DesignBase):
    type: Literal["grid"] = "grid"
    n_u: int = Field(gt=0)
    n_v: int = Field(gt=0)
    facet_w_mm: float = Field(gt=0)
    facet_h_mm: float = Field(gt=0)
    gap_mm: float = Field(default=0.0, ge=0)
    # Facet AIMING, not figure -- see _DesignBase. Blank/absent (None)
    # auto-cants at the trace's own slant range; explicit 0 opts back into
    # uncanted -- see _resolved_cant_focal_mm. Must accept 0 for that to be
    # expressible, hence ge=0 rather than gt=0.
    cant_focal_mm: float | None = Field(default=None, ge=0)


class FlowerParams(_DesignBase):
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
    # Tower geometry overrides, validated against the chosen layout's own
    # model by resolve_optics_params (the model cannot be declared here --
    # which one applies depends on `optics`). Absent means "the defaults",
    # which are this module's constants.
    optics_params: dict | None = None

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


def _faceted(
    params: GridParams | FlowerParams,
    surface: Surface | None,
    cant_focal_mm: float | None,
) -> HeliostatDesign:
    """Call the grid or flower builder with one already-decided figure/cant pair."""
    if isinstance(params, GridParams):
        return grid_facets(
            n_u=params.n_u,
            n_v=params.n_v,
            facet_w_mm=params.facet_w_mm,
            facet_h_mm=params.facet_h_mm,
            gap_mm=params.gap_mm,
            surface=surface,
            cant_focal_mm=cant_focal_mm,
        )
    return flower(
        n_petals=params.n_petals,
        petal_length_mm=params.petal_length_mm,
        petal_width_mm=params.petal_width_mm,
        hub_radius_mm=params.hub_radius_mm,
        surface=surface,
        cant_focal_mm=cant_focal_mm,
    )


def _build_design(
    params: RectParams | GridParams | FlowerParams, auto_focal_mm: float | None = None
) -> HeliostatDesign:
    """Turn a validated param model into a :class:`HeliostatDesign`.

    Builder-level ``ValueError``s (a flower's petal width too wide for its
    length, say) are left to propagate; the endpoint maps them to a 422.

    This is the ``surface="adaptive"`` construction *and* the preview
    construction -- it ignores ``params.surface`` entirely. The design
    preview draws footprint only, never figure, and it has no sun position
    to resolve a figure against anyway; the trace endpoint goes through
    :func:`_build_trace_design`, which honours ``surface``.

    Rectangle figures are not this function's business -- a rectangle's
    figure depends on a solve (sun position), which this function does not
    take. This function's rect branch is a plain flat sketch.
    """
    if isinstance(params, RectParams):
        return rect_heliostat(width_mm=params.width_mm, height_mm=params.height_mm)
    cant = _resolved_cant_focal_mm(params.cant_focal_mm, auto_focal_mm)
    surface = Spherical("slant") if cant is not None else None
    return _faceted(params, surface, cant)


def _build_trace_design(
    params: RectParams | GridParams | FlowerParams,
    sol: Solution,
    slant_range_mm: float,
) -> HeliostatDesign | None:
    """The mirror a trace actually uses: ``surface`` mode crossed with design type.

    Returns ``None`` for the tracer's LEGACY single-mirror path, which is
    reached by exactly one combination -- an adaptive rectangle at the
    engine's default 5000x3000 size. That path is bit-for-bit the validated
    fixture physics (``tests/test_aiming.py``,
    ``tests/test_design_tracing.py``), so it stays the default trace; but it
    hard-codes the solve's astigmatic figure, so a default-size rectangle
    asking for any *other* surface has to be routed through the design path
    instead, or it would silently get the astigmatic figure it did not ask
    for.

    Rect's adaptive figure is carried as ``ZernikeAstig(c3, -c4, -c5)`` per
    the sign convention documented in ``tests/test_design_tracing.py`` (the
    legacy path negates c4/c5 internally; a design equivalent to legacy
    (c3, c4, c5) needs that flip applied up front).

    Canting stays on ``cant_focal_mm`` under every surface mode -- see
    :class:`_DesignBase` for why those are two axes and not one.
    """
    if isinstance(params, RectParams):
        if params.surface == "adaptive":
            if params.width_mm == 5000.0 and params.height_mm == 3000.0:
                return None
            figure: Surface = ZernikeAstig(sol.c3, -sol.c4, -sol.c5)
        elif params.surface == "flat":
            figure = Flat()
        else:
            # A rectangle has no cant_focal_mm to read, so "the resolved
            # cant focal" is the blank case: this heliostat's slant range.
            figure = Spherical(slant_range_mm)
        return rect_heliostat(
            width_mm=params.width_mm, height_mm=params.height_mm, surface=figure
        )

    if params.surface == "adaptive":
        return _build_design(params, auto_focal_mm=slant_range_mm)

    cant = _resolved_cant_focal_mm(params.cant_focal_mm, slant_range_mm)
    if params.surface == "flat":
        surface: Surface = Flat()
    elif cant is None:
        # cant_focal_mm=0 is the caller explicitly asking for no focal point
        # at all, which leaves a spherical figure nothing to be figured
        # against. Inventing a focal here (slant range, say) would trace a
        # perfectly plausible spot for a mirror nobody asked for.
        raise ValueError(
            "surface='spherical' needs a focal length to figure the facets at, "
            "but cant_focal_mm=0 asks for no focus at all. Leave cant_focal_mm "
            "blank to figure at this heliostat's slant range, or give it a "
            "positive focal length."
        )
    else:
        surface = Spherical("slant")
    return _faceted(params, surface, cant)


def _design_is_flat(design: HeliostatDesign | None, c3: float, c4: float, c5: float) -> bool:
    """True when the trace's mirror carries no focusing figure at all.

    The legacy path (``design is None``) is flat exactly when the solve's
    own figure is all-zero -- in practice this does not happen for a real
    solve (the defocus term ``c3`` is nonzero for any finite aim distance),
    so this branch is really only reachable in principle. The design path
    is flat when every facet's surface is :class:`Flat` or an all-zero
    :class:`ZernikeAstig`. Two ways to get there: ``surface="flat"`` on any
    design type (including a rectangle, which is then deliberately routed
    off the legacy path so this check can see it), or an explicit
    ``cant_focal_mm=0`` on an adaptive grid/flower design, which leaves the
    builders' own all-zero ``ZernikeAstig`` default in place (see
    :func:`_resolved_cant_focal_mm`). Canting does not count: a canted flat
    facet is still flat, and still needs the denser sampling.
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


def _geometry_for(optics: str, params: OpticsParams | None = None):
    """``(secondary, receiver)`` for ``optics``, built from ``params``.

    ``params=None`` resolves the layout's defaults, so calling this with the
    optics name alone still gives the fixture geometry verbatim. When a
    trace passes a resolved object it must be the same one it passed
    :func:`_solve_for`, or the aim point and the tower disagree.
    """
    if params is None:
        params = resolve_optics_params(optics, None)
    if optics == "prime_focus":
        secondary = NoSecondary()
        receiver = FlatWindowReceiver(
            z_mm=params.focus_height_mm,
            half_u_mm=params.window_half_u_mm,
            half_v_mm=params.window_half_v_mm,
            facing="down",
        )
    elif optics == "axicon":
        secondary = AxiconSecondary(
            apex_height_mm=params.apex_height_mm,
            half_angle_deg=params.half_angle_deg,
            aperture_radius_mm=params.aperture_radius_mm,
        )
        receiver = FlatWindowReceiver(
            z_mm=params.receiver_z_mm,
            half_u_mm=params.window_half_u_mm,
            half_v_mm=params.window_half_v_mm,
            facing="up",
        )
    elif optics == "cassegrain":
        secondary = CassegrainSecondary(
            vertex_z_mm=CASSEGRAIN_VERTEX_Z_MM,
            vertex_radius_mm=CASSEGRAIN_VERTEX_RADIUS_MM,
            conic=CASSEGRAIN_CONIC,
            aperture_radius_mm=CASSEGRAIN_APERTURE_RADIUS_MM,
        )
        receiver = FlatWindowReceiver(
            z_mm=CASSEGRAIN_RECEIVER_Z_MM,
            half_u_mm=params.window_half_u_mm,
            half_v_mm=params.window_half_v_mm,
            facing="up",
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
            optics_params = resolve_optics_params(body.optics, body.optics_params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        sol = _solve_for(
            body.optics,
            body.heliostat_x_mm,
            body.heliostat_y_mm,
            body.solar_az_deg,
            body.solar_el_deg,
            optics_params,
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

        # Pointing AND figure both come from the solve; which figure is
        # decided by the design's `surface` axis -- see _build_trace_design
        # for the legacy-path rule and _DesignBase for surface vs cant.
        try:
            design = _build_trace_design(body.design, sol, slant_range_mm)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics, optics_params)
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
                return_paths=True,
                design=design,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            traced_paths = result["paths"]
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
            traced_paths = None  # cone optics carries no rays; the scene samples its own
            flux = result["flux"]
            u_edges, v_edges = result["u_edges"], result["v_edges"]
            power_w = result["power_w"]
            incident_power_w = result["incident_power_w"]
            counters = result["counters"]
            rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)

        png_bytes = _render_flux_png(flux, u_edges, v_edges, body.mode, elapsed_ms)

        # The 3-D view's geometry, built from exactly the values the trace
        # above was given (see heliostat.web.scene). Strictly additive: it
        # reads the trace, never feeds it, so every other field of this
        # response is what it was before the scene existed.
        scene = build_scene(
            design,
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
            paths=traced_paths,
        )

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
                # What the tower geometry actually resolved to, so the
                # client can populate its inspector without keeping a second
                # copy of these defaults.
                "optics_resolved": optics_params.model_dump(),
                "scene": scene,
            }
        )

    return app
