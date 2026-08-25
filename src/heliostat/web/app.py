"""FastAPI application for the local heliostat web GUI.

The first slice: design a heliostat, pick sun position, optics layout and
fidelity mode, trace it, see the flux map. Everything here is a thin HTTP
skin over the existing library -- no new physics, no new geometry.

``/api/field/trace`` is the same trace over a whole layout at one instant:
per-heliostat solves and designs, mutual shading/blocking from
:mod:`heliostat.geometry.shading`, and the receiver maps summed. It is
built from the pieces :mod:`heliostat.sweep` uses rather than from
``run_sweep`` itself -- a browser request has no run store, no time grid
and no worker pool -- and both endpoints trace through the same
:func:`_trace_core`, so a one-heliostat field and a single trace cannot
drift apart.

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
import csv
import datetime as _dt
import time
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Union

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        TypeAdapter,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("heliostat.web needs the 'web' extra: pip install heliostat[web]") from exc

from heliostat import __version__
from heliostat.field import HeliostatField, load_field, neighbour_pairs
from heliostat.field_layouts import generate, ring_filter
from heliostat.geometry.aiming import (
    Solution,
    aim_points_mm,
    solve_axicon,
    solve_cassegrain,
    solve_prime_focus,
)
from heliostat.geometry.aperture import Polygon
from heliostat.geometry.design import (
    Flat,
    HeliostatDesign,
    Spherical,
    Surface,
    ZernikeAstig,
    custom_heliostat,
    flower,
    grid_facets,
    rect_heliostat,
)
from heliostat.geometry.heliostat import zernike_sag_and_slopes
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    solve_cassegrain_relay,
)
from heliostat.geometry.shading import (
    MirrorGeometry,
    min_beam_elevation_deg,
    mirror_basis,
    normal_from_angles,
    polygon_occlusion,
    search_radius_for,
)
from heliostat.solar import build_time_grid, sun_position, sunrise_sunset
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM, trace_heliostat
from heliostat.trace.modes import MODES, TraceMode
from heliostat.web.builtin_library import BUILTIN_DESIGNS, BUILTIN_RECEIVERS
from heliostat.web.jobs import JobRegistry
from heliostat.web.library import (
    LibraryError,
    delete_entry,
    list_entries,
    load_entry,
    save_entry,
)
from heliostat.web.scene import (
    build_field_scene,
    build_geometry_scene,
    build_scene,
    field_miss_detection,
)
from heliostat.web.setups import (
    SetupError,
    delete_setup,
    list_setups,
    load_setup,
    save_setup,
)

STATIC_DIR = Path(__file__).parent / "static"

#: Background runs (day sweeps). One registry per process; see
#: heliostat.web.jobs for why it is deliberately this small.
JOBS = JobRegistry()

WINDOW_MM = 2000.0
FLUX_GRID = 128

# ---------------------------------------------------------------------------
# the manuscript field: the paper's real 643-heliostat layout
#
# The app's default field used to REGENERATE a Fermat spiral standing in for
# the paper's own positions -- close in scale (643, 30-90 m), but not the
# points a reader comparing the app to the paper actually wants. This is the
# byte-identical fix: a packaged copy of examples/paper/data/field_645.csv,
# loaded through the exact loader the paper's own reproduce.py calls
# (load_paper_field), with the identical coincident-duplicate rule applied
# (144=192, 241=289 dropped by heliostat.field.load_field's own
# distance-based check, leaving 643). Parsed once per process below -- the
# file never changes at runtime.
MANUSCRIPT_FIELD_PATH = STATIC_DIR / "data" / "field_645.csv"

#: Mirror dims load_paper_field() passes load_field() (examples/paper/
#: reproduce.py's own MIRROR_WIDTH_MM/MIRROR_HEIGHT_MM). They do not affect
#: which positions load, only the field's own mirror_width_mm/height_mm
#: metadata -- carried here only so a caller reading `_load_manuscript_field()`
#: gets a field that matches the paper's in every respect, not just position.
_MANUSCRIPT_MIRROR_WIDTH_MM = 5000.0
_MANUSCRIPT_MIRROR_HEIGHT_MM = 3000.0

_manuscript_field: HeliostatField | None = None


def _load_manuscript_field() -> HeliostatField:
    """The paper's field, parsed once and cached for the process lifetime."""
    global _manuscript_field
    if _manuscript_field is None:
        _manuscript_field = load_field(
            MANUSCRIPT_FIELD_PATH,
            mirror_width_mm=_MANUSCRIPT_MIRROR_WIDTH_MM,
            mirror_height_mm=_MANUSCRIPT_MIRROR_HEIGHT_MM,
        )
    return _manuscript_field


# ---------------------------------------------------------------------------
# field-trace limits and layout defaults
#
# 1000 rather than the 600 of this package's reference field
# (``scripts/sweep_benchmark.py``): the companion paper's own field is 643
# heliostats, and a tool that cannot express the field it reproduces is
# the wrong tool. Large fields are slow in a browser request -- that is
# what the day-sweep job endpoint is for -- but slow is the caller's
# choice to make, not something to forbid.
MAX_FIELD_HELIOSTATS = 1000

# /api/scene/geometry's own, much larger cap. Placing and orienting a mirror
# (one aiming solve, no shading/blocking, no receiver trace) costs nothing
# like tracing it does, and the 3-D view's whole reason to exist is showing a
# field too big to trace in a browser request -- docs/ui-spec.md 2.1 states
# the scale target explicitly: "smooth orbiting up to 10,000 heliostats".
# Ten thousand analytic solves is still a fraction of a second; the trace
# cap above is unaffected.
MAX_GEOMETRY_HELIOSTATS = 10_000

# Fermat-spiral geometry for the ``{"type": "fermat", "n": ...}`` layout.
#
# JUDGMENT CALL: ``heliostat layout fermat`` has no default for ``--a`` (it is
# a required argument), so "the CLI's defaults" is only defined for b,
# k_start, divergence and oversample -- which are b=0.5, k_start=1, the golden
# angle and 1.6. The scale, and the exponent that goes with it, are taken from
# ``scripts/sweep_benchmark.py``'s 600-heliostat reference field (a=4.5,
# b=0.55) so that a field traced in the GUI is the same field the headless
# benchmark measures. Everything else is
# ``heliostat.field_layouts.generate``'s own default, so
# ``FermatLayout.positions_mm`` is the CLI's call with no filters.
FERMAT_A_M = 4.5
FERMAT_B = 0.55

#: Base seed for the field's per-heliostat Monte Carlo streams. Fixed, so two
#: identical field requests return identical numbers; per-heliostat, so 600
#: mirrors do not all draw the same sample pattern.
FIELD_MC_SEED = 20260818

#: Vertices in the occluder silhouette, matching
#: :meth:`heliostat.geometry.shading.MirrorGeometry.from_design`'s own default
#: -- the field endpoint builds the silhouette itself (once, shared) rather
#: than through ``from_design``, and this keeps the two identical.
OCCLUDER_SILHOUETTE_VERTICES = 72

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
                "reflects the beam onto a ground receiver beneath it"
            )
        return self


class CassegrainOptics(BaseModel):
    """Hyperboloid relay: adjustable, with the relay solved to match.

    The relay used to be fixed, because its vertex radius and conic constant
    were solved once for one focus/receiver pair and moving either would
    have left the stored constants describing a surface that no longer
    joined them. Re-solving is not hard, though -- a hyperboloid images one
    focus onto the other, and that pins the surface exactly (see
    :func:`~heliostat.geometry.secondary.solve_cassegrain_relay`) -- so the
    geometry is a set of heights the caller chooses, and the mirror that
    serves them is computed.

    ``vertex_z_mm`` is where the secondary sits, ``focus_height_mm`` the
    primary focus it relays (which is also what the aiming solve points the
    field at), and ``receiver_z_mm`` where the beam is delivered. The three
    have to describe a real hyperboloid; they are checked together.
    """

    model_config = ConfigDict(extra="forbid")

    vertex_z_mm: float = Field(default=CASSEGRAIN_VERTEX_Z_MM, gt=0)
    focus_height_mm: float = Field(default=CASSEGRAIN_FOCUS_HEIGHT_MM, gt=0)
    receiver_z_mm: float = Field(default=CASSEGRAIN_RECEIVER_Z_MM, gt=0)
    aperture_radius_mm: float = Field(default=CASSEGRAIN_APERTURE_RADIUS_MM, gt=0)
    window_half_u_mm: float = Field(default=WINDOW_MM, gt=0)
    window_half_v_mm: float = Field(default=WINDOW_MM, gt=0)

    @model_validator(mode="after")
    def _relay_must_be_solvable(self) -> "CassegrainOptics":
        # Solve it here rather than at trace time so an impossible tower is
        # a 422 naming the geometry, not a failure three layers down.
        try:
            solve_cassegrain_relay(self.vertex_z_mm, self.focus_height_mm, self.receiver_z_mm)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def relay(self) -> tuple[float, float]:
        """``(vertex_radius_mm, conic)`` for this geometry."""
        return solve_cassegrain_relay(self.vertex_z_mm, self.focus_height_mm, self.receiver_z_mm)


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
        # The field aims at the primary focus F1, and the relay is solved to
        # take that same F1 to the receiver -- so this must be the request's
        # focus height, not the module default. Reading them from different
        # places is precisely the drift this module exists to prevent.
        return solve_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg, params.focus_height_mm)
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

    * ``"twisting"`` (default) -- whatever solve-driven figure this app
      judges best for the design type, i.e. exactly what it did before this
      field existed. For a rectangle that is the aiming solve's own
      astigmatic figure, the twisting mirror of the companion paper; for a
      grid or flower it is *spherical* facets auto-focused at the
      heliostat's slant range (blank ``cant_focal_mm``) or at the given
      focal. "Twisting" names the choice being made for you, not a single
      distinct kind of surface.
    * ``"spherical"`` -- a spherical cap on every facet (or on the single
      rectangle), at the resolved cant focal: blank ``cant_focal_mm`` means
      this heliostat's own slant range, an explicit value means that focal.
      A rectangle has no ``cant_focal_mm``, so it always figures at slant
      range.
    * ``"flat"`` -- no figure at all, anywhere. Expect a mirror-shaped wash
      rather than a spot.

    ``slope_error_mrad``/``specularity_mrad``/``reflectance`` are optical
    errors, orthogonal to ``surface``: they blur and dim whatever figure
    ``surface`` already chose rather than describing a figure of their own.
    ``slope_error_mrad`` is a random per-ray tilt of the mirror's local
    surface normal (a manufacturing/mounting imperfection); it deflects a
    reflected ray by twice the tilt, same convention the cone backend's
    ``sunshape_kernel`` already documents. ``specularity_mrad`` is a random
    per-ray scatter of the reflected beam itself (a coating imperfection),
    with no such factor of two. Both default to ``0`` -- a perfect mirror --
    so an old request that has never heard of either field traces exactly as
    it always has. ``reflectance`` is the fraction of incident power that
    survives the bounce; it defaults to ``1.0`` for the same reason, so
    ``power_w``/the flux map are unscaled unless a caller opts in, and
    ``incident_power_w`` -- power arriving on the mirror, before the bounce
    -- never carries it.
    """

    surface: Literal["twisting", "spherical", "flat"] = "twisting"
    slope_error_mrad: float = Field(default=0.0, ge=0)
    specularity_mrad: float = Field(default=0.0, ge=0)
    reflectance: float = Field(default=1.0, gt=0, le=1)


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


class CustomParams(_DesignBase):
    """A single hand-drawn polygon facet -- the sketch-tool analogue of
    :class:`RectParams`.

    ``vertices_mm`` is the facet's own outline in the heliostat ``(u, v)``
    plane, one ``[u, v]`` pair per corner, in order around the perimeter
    (either winding). No ``cant_focal_mm``: a single facet has nothing to
    cant against another, exactly like a rectangle.
    """

    type: Literal["custom"] = "custom"
    vertices_mm: list[tuple[float, float]] = Field(min_length=3)

    @model_validator(mode="after")
    def _polygon_must_enclose_area(self) -> "CustomParams":
        pts = [self.vertices_mm[0]]
        for u, v in self.vertices_mm[1:]:
            # Consecutive duplicates contribute a zero-length edge -- drop
            # them before counting corners, so "a rectangle with one vertex
            # accidentally sent twice" is not silently treated as a
            # pentagon.
            if (u, v) != pts[-1]:
                pts.append((u, v))
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts.pop()
        if len(pts) < 3:
            raise ValueError(
                "vertices_mm needs at least 3 distinct points once consecutive "
                "duplicates are dropped, to describe a polygon at all"
            )
        arr = np.asarray(pts, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("vertices_mm must be finite")
        # Shoelace formula, twice the signed area; either winding is
        # accepted (the sign only says clockwise vs. counter-clockwise), so
        # only its magnitude is checked. Zero (collinear points, or a
        # polygon that folds back on itself to no net area) has no facet to
        # trace.
        u, v = arr[:, 0], arr[:, 1]
        signed_area2 = float(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))
        if signed_area2 == 0.0:
            raise ValueError(
                "vertices_mm must enclose a positive area -- these points are "
                "collinear or otherwise describe a zero-area polygon"
            )
        return self


DesignParams = Annotated[
    Union[RectParams, GridParams, FlowerParams, CustomParams], Field(discriminator="type")
]


class SetupRequest(BaseModel):
    """A named snapshot of the GUI's controls.

    ``document`` is deliberately unvalidated free-form JSON: it is the
    client's own state, and pinning a schema here would mean this module
    needs editing every time the panel gains a control. It is stored and
    handed back verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    document: dict


class SunRequest(BaseModel):
    """A place and a moment, for the sun-position endpoint.

    The GUI's own sun controls are azimuth and elevation, because that is
    what a trace needs. Those are awkward numbers to know for a real site,
    so this turns a site and a clock time into them via
    :func:`heliostat.solar.sun_position` -- the NOAA calculator method (see
    REFERENCES.md). Elevation carries NOAA's atmospheric-refraction
    correction, so it can be slightly positive with the sun geometrically
    just below the horizon.
    """

    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(default=-10.0, ge=-90.0, le=90.0)
    longitude_deg: float = Field(default=-52.0, ge=-180.0, le=180.0)
    timezone_h: float = Field(default=-3.0, ge=-14.0, le=14.0)
    year: int = Field(default=2026, ge=1901, le=2099)
    month: int = Field(default=3, ge=1, le=12)
    day: int = Field(default=21, ge=1, le=31)
    hour: float = Field(default=12.0, ge=0.0, lt=24.0)


class PreviewRequest(BaseModel):
    design: DesignParams


class _TraceRequestBase(BaseModel):
    """Everything a trace needs except *which mirrors* -- design, fidelity,
    tower, sun. Shared verbatim by the single-heliostat and whole-field
    endpoints so the two cannot drift in what they accept or default to; the
    subclasses add only the heliostat position (single) or the layout
    (field)."""

    design: DesignParams
    mode: Literal["ultra_fast", "fast_accurate", "monte_carlo"]
    optics: Literal["prime_focus", "axicon", "cassegrain"]
    solar_az_deg: float = Field(ge=0, le=360)
    solar_el_deg: float
    # Tower geometry overrides, validated against the chosen layout's own
    # model by resolve_optics_params (the model cannot be declared here --
    # which one applies depends on `optics`). Absent means "the defaults",
    # which are this module's constants.
    optics_params: dict | None = None
    #: Rays per heliostat, Monte Carlo only. ``None`` takes the mode's own
    #: budget (120,000), which is what every stored result was traced with.
    #: Lowering it is the fidelity/speed dial: Monte Carlo error falls as
    #: 1/sqrt(rays), so a tenth of the rays is about three times the noise.
    n_rays: int | None = Field(default=None, ge=100, le=2_000_000)

    def trace_mode(self) -> TraceMode:
        """The fidelity mode this request asks for, ray budget applied."""
        mode = MODES[self.mode]
        if self.n_rays is None or mode.backend != "mc":
            return mode
        return replace(mode, n_rays=self.n_rays)

    @field_validator("solar_el_deg")
    @classmethod
    def _elevation_must_be_physical(cls, v: float) -> float:
        # Full rejection of a non-positive sun happens in the endpoint (it
        # needs a friendlier message than a bare pydantic constraint), but
        # anything past straight up is a plain typo -- reject it here.
        if v > 90.0:
            raise ValueError("solar_el_deg must be <= 90")
        return v


class TraceRequest(_TraceRequestBase):
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0


# ---------------------------------------------------------------------------
# field layouts


class FermatLayout(BaseModel):
    """``n`` heliostats on a golden-ratio Fermat spiral.

    The same call ``heliostat layout fermat`` makes: no filters, generate's
    own oversample, ids ``0..n-1`` in spiral order (which runs outward from
    the tower). Deterministic -- the spiral is a pure function of ``k``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fermat"] = "fermat"
    n: int = Field(default=100, ge=1, le=MAX_FIELD_HELIOSTATS)
    #: Spiral scale, metres. ``None`` means "work it out": with a farthest
    #: radius given, it is solved so the requested count fits the envelope
    #: (see :meth:`resolved_a_m`). Without one it falls back to the default,
    #: so a plain ``{"type": "fermat", "n": N}`` is unchanged.
    a_m: float | None = Field(default=None, gt=0)
    b: float = Field(default=FERMAT_B, gt=0)
    # Ground-radius bounds: how close to the tower the nearest heliostat may
    # stand, and how far the farthest may.
    r_min_m: float | None = Field(default=None, ge=0)
    r_max_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _radii_ordered(self) -> "FermatLayout":
        if self.r_min_m is not None and self.r_max_m is not None:
            if self.r_max_m <= self.r_min_m:
                raise ValueError("r_max_m must be greater than r_min_m")
        return self

    def _k_for_radius(self, radius_m: float, a_m: float) -> float:
        """Spiral index at a ground radius, from ``r = a * k**b``."""
        return float((radius_m / a_m) ** (1.0 / self.b))

    def resolved_a_m(self) -> float:
        """The spiral scale actually used.

        A Fermat spiral's *density* is fixed by ``a``: the number of
        positions between two radii is
        ``(r_max**(1/b) - r_min**(1/b)) / a**(1/b)``, and no amount of
        generating further out adds any. So asking for 600 heliostats
        between 30 and 90 m at the default a = 4.5 m is not a matter of
        searching harder -- only about 200 positions exist there. Rather
        than refusing a perfectly reasonable field, solve ``a`` so the
        requested count fits the requested envelope.

        Only done when a farthest radius is given and ``a_m`` was not set
        explicitly; an explicit ``a_m`` is always honoured, because it is a
        physical spacing choice and this should not quietly overrule it.
        """
        if self.a_m is not None:
            return self.a_m
        if self.r_max_m is None:
            return FERMAT_A_M
        inv_b = 1.0 / self.b
        r_min = 0.0 if self.r_min_m is None else self.r_min_m
        span = self.r_max_m**inv_b - r_min**inv_b
        if span <= 0:  # pragma: no cover - the validator orders the radii
            raise ValueError("r_max_m must be greater than r_min_m")
        return float((span / self.n) ** self.b)

    def positions_mm(self) -> np.ndarray:
        a_m = self.resolved_a_m()
        filters = ()
        oversample = 1.6
        if self.r_min_m is not None or self.r_max_m is not None:
            filters = (
                ring_filter(
                    0.0 if self.r_min_m is None else self.r_min_m,
                    1.0e9 if self.r_max_m is None else self.r_max_m,
                ),
            )
            # generate() draws n * oversample candidates in spiral order and
            # only then filters, so the default 1.6 keeps nothing at all when
            # the ring sits outside the first few turns. Invert the radius
            # law to find how far out the ring reaches.
            k_needed = float(self.n)
            if self.r_max_m is not None:
                k_needed = max(k_needed, self._k_for_radius(self.r_max_m, a_m))
            if self.r_min_m is not None:
                k_needed = max(k_needed, self._k_for_radius(self.r_min_m, a_m) + self.n)
            oversample = max(oversample, 1.25 * k_needed / self.n)
        try:
            field = generate(
                "fermat",
                self.n,
                a_m=a_m,
                b=self.b,
                filters=filters,
                oversample=oversample,
            )
        except ValueError as exc:
            raise ValueError(self._capacity_message(a_m, exc)) from exc
        return field.xy_mm

    def _capacity_message(self, a_m: float, exc: Exception) -> str:
        """Explain a shortfall in terms the caller can act on.

        Reaching here means ``a_m`` was pinned by the caller, since a solved
        one fits by construction -- so the useful thing to say is which
        spacing *would* fit.
        """
        if self.r_max_m is None:
            return f"that layout does not fit {self.n} heliostats: {exc}"
        inv_b = 1.0 / self.b
        r_min = 0.0 if self.r_min_m is None else self.r_min_m
        capacity = int((self.r_max_m**inv_b - r_min**inv_b) / a_m**inv_b)
        fits = float(((self.r_max_m**inv_b - r_min**inv_b) / self.n) ** self.b)
        return (
            f"a Fermat spiral with a = {a_m:g} m holds only about {capacity} "
            f"heliostats between {r_min:g} and {self.r_max_m:g} m, not "
            f"{self.n}. Use a = {fits:.3g} m to fit {self.n} in that ring, "
            f"or leave a_m unset and it will be solved for you."
        )


class PositionsLayout(BaseModel):
    """Explicit heliostat positions, mm, in the tracers' world frame.

    The path a loaded field file will arrive on, and the path the GUI's
    inspector already uses to move one heliostat and re-trace: it sends the
    field back with that heliostat's row edited, so a moved mirror is one
    request rather than a stateful session.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["positions"] = "positions"
    xy_mm: list[tuple[float, float]] = Field(min_length=1, max_length=MAX_FIELD_HELIOSTATS)

    def positions_mm(self) -> np.ndarray:
        """Validated positions.

        The finiteness check lives here rather than in a pydantic
        ``field_validator`` on purpose. JSON has no NaN/Infinity literal, but
        Python's own ``json`` module both writes and reads all three, so a
        Python client can post one -- and a heliostat at infinity traces a
        plausible-looking nothing. Rejecting it in a validator does produce a
        422, but FastAPI's validation-error body echoes the offending input,
        and its JSON encoder then refuses to serialise the very NaN it is
        complaining about, so the client gets a server error instead of the
        message. Raising here puts the rejection somewhere the response is
        this module's to write.
        """
        xy = np.asarray(self.xy_mm, dtype=float)
        if not np.isfinite(xy).all():
            raise ValueError("xy_mm must be finite -- got a NaN or infinite coordinate")
        return xy


FieldLayout = Annotated[Union[FermatLayout, PositionsLayout], Field(discriminator="type")]


class FieldTraceRequest(_TraceRequestBase):
    layout: FieldLayout
    #: Layout indices to leave out. The surviving heliostats keep their
    #: original layout index as their id, so dropping one does not renumber
    #: the rest -- an id in a response means the same mirror across requests.
    exclude_ids: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# geometry-only field layouts
#
# /api/scene/geometry needs the same two layout shapes as a trace, only under
# the much larger MAX_GEOMETRY_HELIOSTATS cap -- and that cap is a literal
# Field(le=...) on FermatLayout.n and PositionsLayout.xy_mm, not a runtime
# check this module could apply after the fact. Subclassing and redeclaring
# just the capped field reuses every other line of the parent (a_m solving,
# the capacity-shortfall message, the NaN rejection) rather than forking the
# whole layout under a new name.


class GeometryFermatLayout(FermatLayout):
    n: int = Field(default=100, ge=1, le=MAX_GEOMETRY_HELIOSTATS)


class GeometryPositionsLayout(PositionsLayout):
    xy_mm: list[tuple[float, float]] = Field(min_length=1, max_length=MAX_GEOMETRY_HELIOSTATS)


GeometryFieldLayout = Annotated[
    Union[GeometryFermatLayout, GeometryPositionsLayout], Field(discriminator="type")
]


class GeometryRequest(BaseModel):
    """Where a field stands and points -- no trace, no flux.

    Everything a trace's phase 1 (:func:`_solve_field`) needs and nothing
    past it: design, optics/tower, sun, and either a field ``layout`` or a
    single heliostat's position, mirroring :class:`DayTraceRequest`'s own
    "layout, else the single position" convention. No ``mode`` and no
    ``n_rays`` -- there is no trace to budget.

    ``design`` defaults to the legacy 5000x3000 rectangle (the same size
    :class:`TraceRequest`'s hard-coded position defaults trace) rather than
    being required, since the 3-D view's whole point is showing *something*
    the instant it opens, before the sidebar's own design panel has been
    touched (docs/ui-spec.md 2.1, "Live from the first frame").
    """

    model_config = ConfigDict(extra="forbid")

    design: DesignParams = Field(
        default_factory=lambda: RectParams(width_mm=5000.0, height_mm=3000.0)
    )
    optics: Literal["prime_focus", "axicon", "cassegrain"]
    optics_params: dict | None = None
    solar_az_deg: float = Field(ge=0, le=360)
    solar_el_deg: float
    layout: GeometryFieldLayout | None = None
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0
    #: Chief rays from mirror corners through the secondary to the receiver
    #: (docs/ui-spec.md 2.1's "corner rays"); off by request for a caller
    #: that only wants positions.
    include_corner_rays: bool = True
    #: Cap on how many heliostats contribute corner rays -- ui-spec's own
    #: number ("For big fields they come from a spread-out subset (cap
    #: ~500 sources)"). Unlike MAX_GEOMETRY_HELIOSTATS this is not a hard
    #: field-size limit, just a picture budget, so it is adjustable.
    max_corner_sources: int = Field(default=500, ge=1, le=2000)

    @field_validator("solar_el_deg")
    @classmethod
    def _elevation_must_be_physical(cls, v: float) -> float:
        # Unlike a trace, a non-positive elevation is not rejected here (see
        # the endpoint): the sun below the horizon has to draw a scene, not a
        # 422 -- docs/ui-spec.md 2.1, "scene never goes blank". Only past
        # straight up is a plain typo.
        if v > 90.0:
            raise ValueError("solar_el_deg must be <= 90")
        return v


class DaySite(BaseModel):
    """Where and when. The sun angles come from this, per timestep."""

    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(default=-10.0, ge=-90.0, le=90.0)
    longitude_deg: float = Field(default=-52.0, ge=-180.0, le=180.0)
    timezone_h: float = Field(default=-3.0, ge=-14.0, le=14.0)
    year: int = Field(default=2026, ge=1901, le=2099)
    month: int = Field(default=3, ge=1, le=12)
    day: int = Field(default=21, ge=1, le=31)


class DayTraceRequest(_TraceRequestBase):
    """Trace one date from sunrise to sunset.

    Inherits the design, fidelity mode, optics and tower geometry a single
    trace takes; ``solar_az_deg``/``solar_el_deg`` are inherited too but
    ignored, because the whole point is that the sun moves. ``layout``
    traces a field at every timestep; without one it traces the single
    heliostat at ``heliostat_x_mm``/``heliostat_y_mm``.
    """

    site: DaySite = Field(default_factory=DaySite)
    #: Maximum spacing between samples, hours. The day is divided into equal
    #: intervals no wider than this, so the first and last land exactly on
    #: the daylight edges rather than being snapped inward.
    hour_step: float = Field(default=1.0, gt=0.05, le=6.0)
    sunrise_margin_min: float = Field(default=10.0, ge=0.0, le=120.0)
    layout: FieldLayout | None = None
    exclude_ids: list[int] = Field(default_factory=list)
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0


# ---------------------------------------------------------------------------
# library: named designs, receiver configs and projects
#
# heliostat.web.library is the file store (name-safe, atomic writes, skip-
# unparseable listing -- the same machinery heliostat.web.setups uses, see
# that module for why); it stores and returns documents without interpreting
# them, exactly like setups does. Everything that gives those documents a
# *shape* -- what a receiver or a project actually contains -- lives here,
# next to the request models it reuses, so a receiver document and a trace
# request's optics_params can never validate two different things called the
# same name.


class LibrarySaveRequest(BaseModel):
    """A name and a document, for any of the three library collections.

    The collection itself comes from the URL, not the body -- one request
    shape serves ``designs``, ``receivers`` and ``projects`` alike, and which
    schema the document must satisfy is decided by :func:`_validate_library_document`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    document: dict


class ReceiverDocument(BaseModel):
    """A tower: which optics layout, and its params.

    The shape of one ``receivers`` library entry's document, and of
    :class:`ProjectDocument`'s ``receiver`` field -- one schema describes a
    receiver everywhere it appears, whether saved alone or bundled in a
    project. ``params`` is validated the same way a trace's own
    ``optics_params`` is (:func:`resolve_optics_params`): absent/empty
    resolves to that layout's defaults, and an invalid value names the field
    that is wrong rather than failing three layers down.
    """

    model_config = ConfigDict(extra="forbid")

    optics: Literal["prime_focus", "axicon", "cassegrain"]
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _params_must_resolve(self) -> "ReceiverDocument":
        try:
            resolve_optics_params(self.optics, self.params)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self


class ProjectField(BaseModel):
    """Where the mirrors stand, mirroring a trace request's own
    "layout, else a single position" choice (see :class:`DayTraceRequest`)."""

    model_config = ConfigDict(extra="forbid")

    layout: FieldLayout | None = None
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0


class ProjectSun(BaseModel):
    """A sun direction, and optionally the site/time that produced it.

    ``site`` is carried alongside the angles rather than instead of them --
    reopening a project should not require re-deriving azimuth/elevation
    from a site and clock time it may not even have (a plain angle pair is a
    legal project), but when a site was used it is worth keeping so the
    project can be re-opened at a different hour.
    """

    model_config = ConfigDict(extra="forbid")

    azimuth_deg: float = Field(ge=0, le=360)
    elevation_deg: float = Field(le=90)
    site: DaySite | None = None


class ProjectRun(BaseModel):
    """The fidelity a project was (or should be) traced at."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["ultra_fast", "fast_accurate", "monte_carlo"] = "ultra_fast"
    n_rays: int | None = Field(default=None, ge=100, le=2_000_000)


class ProjectDocument(BaseModel):
    """Schema v1 of a saved project: design + field + receiver + sun + run,
    bundled as the Library's "save my work" unit (docs/ui-spec.md 5).

    ``schema_version`` is a required literal, not a default, on purpose: a
    document that does not say which version it is gets rejected rather than
    silently read under today's rules, so a future v2 can change what a
    project means without reinterpreting old ones.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    design: DesignParams
    receiver: ReceiverDocument
    field: ProjectField
    sun: ProjectSun
    run: ProjectRun = Field(default_factory=ProjectRun)


#: Which pydantic shape validates a document posted to each library
#: collection. ``designs`` validates against the same discriminated union a
#: trace's own ``design`` field takes -- not a single ``BaseModel`` subclass,
#: hence the ``TypeAdapter`` rather than a ``.model_validate()`` call.
_DESIGN_ADAPTER: TypeAdapter = TypeAdapter(DesignParams)


def _validate_library_document(collection: str, document: dict) -> None:
    """Validate ``document`` against ``collection``'s schema.

    Raises :class:`ValueError` with a flattened, human-readable message in
    the same style :func:`resolve_optics_params` uses, so a 422 from saving
    a library entry reads the same way whichever collection produced it.
    """
    try:
        if collection == "designs":
            _DESIGN_ADAPTER.validate_python(document)
        elif collection == "receivers":
            ReceiverDocument.model_validate(document)
        elif collection == "projects":
            ProjectDocument.model_validate(document)
        else:  # pragma: no cover - the endpoint 404s unknown collections first
            raise ValueError(f"unknown library collection {collection!r}")
    except ValidationError as exc:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        raise ValueError(f"{collection} document -- " + "; ".join(parts)) from exc


#: Built-in, read-only entries per collection -- see
#: heliostat.web.builtin_library for the numbers and where they come from.
#: ``projects`` has none: a project is always something a user built.
_BUILTIN_LIBRARY: dict[str, dict[str, dict]] = {
    "designs": BUILTIN_DESIGNS,
    "receivers": BUILTIN_RECEIVERS,
    "projects": {},
}


def _require_known_collection(collection: str) -> None:
    """404 for a collection name that is not one of the three -- every
    library route needs this same check first, so it is one function
    rather than four copies of the same ``if``."""
    if collection not in _BUILTIN_LIBRARY:
        raise HTTPException(status_code=404, detail=f"unknown library collection {collection!r}")


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
    params: RectParams | GridParams | FlowerParams | CustomParams,
    auto_focal_mm: float | None = None,
) -> HeliostatDesign:
    """Turn a validated param model into a :class:`HeliostatDesign`.

    Builder-level ``ValueError``s (a flower's petal width too wide for its
    length, say) are left to propagate; the endpoint maps them to a 422.

    This is the ``surface="twisting"`` construction *and* the preview
    construction -- it ignores ``params.surface`` entirely. The design
    preview draws footprint only, never figure, and it has no sun position
    to resolve a figure against anyway; the trace endpoint goes through
    :func:`_build_trace_design`, which honours ``surface``.

    Rectangle and custom-polygon figures are not this function's business --
    both depend on a solve (sun position), which this function does not
    take. Those two branches are plain flat sketches.
    """
    if isinstance(params, RectParams):
        return rect_heliostat(width_mm=params.width_mm, height_mm=params.height_mm)
    if isinstance(params, CustomParams):
        return custom_heliostat(vertices_mm=params.vertices_mm)
    cant = _resolved_cant_focal_mm(params.cant_focal_mm, auto_focal_mm)
    surface = Spherical("slant") if cant is not None else None
    return _faceted(params, surface, cant)


def _build_trace_design(
    params: RectParams | GridParams | FlowerParams | CustomParams,
    sol: Solution,
    slant_range_mm: float,
) -> HeliostatDesign | None:
    """The mirror a trace actually uses: ``surface`` mode crossed with design type.

    Returns ``None`` for the tracer's LEGACY single-mirror path, which is
    reached by exactly one combination -- a twisting rectangle at the
    engine's default 5000x3000 size. That path is bit-for-bit the validated
    fixture physics (``tests/test_aiming.py``,
    ``tests/test_design_tracing.py``), so it stays the default trace; but it
    hard-codes the solve's astigmatic figure, so a default-size rectangle
    asking for any *other* surface has to be routed through the design path
    instead, or it would silently get the astigmatic figure it did not ask
    for.

    Rect's twisting figure is carried as ``ZernikeAstig(c3, -c4, -c5)`` per
    the sign convention documented in ``tests/test_design_tracing.py`` (the
    legacy path negates c4/c5 internally; a design equivalent to legacy
    (c3, c4, c5) needs that flip applied up front).

    Canting stays on ``cant_focal_mm`` under every surface mode -- see
    :class:`_DesignBase` for why those are two axes and not one.
    """
    if isinstance(params, RectParams):
        if params.surface == "twisting":
            if params.width_mm == 5000.0 and params.height_mm == 3000.0:
                return None
            figure: Surface = ZernikeAstig(sol.c3, -sol.c4, -sol.c5)
        elif params.surface == "flat":
            figure = Flat()
        else:
            # A rectangle has no cant_focal_mm to read, so "the resolved
            # cant focal" is the blank case: this heliostat's slant range.
            figure = Spherical(slant_range_mm)
        return rect_heliostat(width_mm=params.width_mm, height_mm=params.height_mm, surface=figure)

    if isinstance(params, CustomParams):
        # Same three figures as a rectangle, same reasoning: one facet, no
        # cant_focal_mm to read, so "spherical" always figures at this
        # heliostat's own slant range.
        if params.surface == "twisting":
            figure = ZernikeAstig(sol.c3, -sol.c4, -sol.c5)
        elif params.surface == "flat":
            figure = Flat()
        else:
            figure = Spherical(slant_range_mm)
        return custom_heliostat(vertices_mm=params.vertices_mm, surface=figure)

    if params.surface == "twisting":
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
    ``cant_focal_mm=0`` on a twisting grid/flower design, which leaves the
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
        # The relay surface is solved from the three heights, so a moved
        # focus or receiver gets the hyperboloid that actually serves it
        # rather than the one that served the fixture.
        vertex_radius_mm, conic = params.relay()
        secondary = CassegrainSecondary(
            vertex_z_mm=params.vertex_z_mm,
            vertex_radius_mm=vertex_radius_mm,
            conic=conic,
            aperture_radius_mm=params.aperture_radius_mm,
        )
        receiver = FlatWindowReceiver(
            z_mm=params.receiver_z_mm,
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


def _trace_core(
    design: HeliostatDesign | None,
    x_mm: float,
    y_mm: float,
    sol: Solution,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary,
    receiver: FlatWindowReceiver,
    mode: TraceMode,
    *,
    mc_seed=1,
    mc_return_paths: bool = True,
    slope_error_mrad: float = 0.0,
    specularity_mrad: float = 0.0,
    reflectance: float = 1.0,
) -> dict:
    """Trace ONE heliostat -- the exact call both endpoints make.

    Pointing and figure come from ``sol``, the mirror from ``design``
    (``None`` is the tracer's legacy rectangle). The returned dict is the
    backend's own output plus a ``backend`` key saying which one ran, so the
    caller can pick the matching metric recipe; nothing is summarised here,
    because the single-heliostat and field endpoints summarise differently.

    ``mc_seed`` defaults to ``1``, the seed this module's single-heliostat
    trace has always used -- so routing that endpoint through this function
    changed nothing about what it returns. The field endpoint passes a
    per-heliostat seed instead (see :data:`FIELD_MC_SEED`).

    ``slope_error_mrad``/``specularity_mrad``/``reflectance`` are the
    design's own optical-error fields (see :class:`_DesignBase`), threaded
    through here rather than read off ``design`` because the legacy
    ``design=None`` path still needs them -- they describe the mirror's
    material, not its facet layout. The first two are forwarded into
    whichever backend ran; ``reflectance`` is applied once, here, as a
    scalar on the REFLECTED result (power and flux), after the backend has
    already reported ``incident_power_w`` -- see the module's
    ``_DesignBase`` docstring for why that field never carries it.
    """
    if mode.backend == "mc":
        result = {
            "backend": "mc",
            **trace_heliostat(
                x_mm,
                y_mm,
                sol.rot_az_deg,
                sol.rot_el_deg,
                sol.c3,
                sol.c4,
                sol.c5,
                solar_az_deg,
                solar_el_deg,
                secondary,
                receiver,
                mode.n_rays,
                np.random.default_rng(mc_seed),
                source_disk_radius_mm="auto",
                return_paths=mc_return_paths,
                design=design,
                slope_error_mrad=slope_error_mrad,
                specularity_mrad=specularity_mrad,
            ),
        }
        if reflectance != 1.0:
            # watts_per_ray is what every downstream reader (power_w, the
            # flux histogram, peak flux) scales from -- one multiply here
            # reaches all of them without touching incident power, which
            # this backend never reports in the first place.
            result["watts_per_ray"] = result["watts_per_ray"] * reflectance
        return result

    kernel = sunshape_kernel(
        "super_gauss", slope_error_mrad=slope_error_mrad, specularity_mrad=specularity_mrad
    )
    cone_kwargs = dict(mode.cone_kwargs)
    if _design_is_flat(design, sol.c3, sol.c4, sol.c5):
        # A deliberately flat mirror (explicit cant_focal_mm=0 on a
        # grid/flower design -- see _design_is_flat) has no focusing figure
        # at all, so the cone backend's per-sample kernels never overlap: at
        # the mode's normal 20x12 sampling grid that shows up as a
        # comb/ripple artifact across the flux map (owner-reported). Denser
        # sampling closes the gaps between kernels; only worth the extra cost
        # for this deliberately-flat case, so it is not the mode's own
        # default.
        cone_kwargs["grid"] = (40, 24)
    result = {
        "backend": "cone",
        **trace_heliostat_cone(
            x_mm,
            y_mm,
            sol.rot_az_deg,
            sol.rot_el_deg,
            sol.c3,
            sol.c4,
            sol.c5,
            solar_az_deg,
            solar_el_deg,
            secondary,
            receiver,
            kernel,
            design=design,
            **cone_kwargs,
        ),
    }
    if reflectance != 1.0:
        # incident_power_w is deliberately left untouched: it is measured
        # before the bounce, and this scalar models loss AT the bounce.
        result["flux"] = result["flux"] * reflectance
        result["power_w"] = result["power_w"] * reflectance
    return result


def _slant_range_mm(sol: Solution, x_mm: float, y_mm: float) -> float:
    """Straight-line distance from this heliostat to its own aim point."""
    return float(
        np.hypot(
            np.hypot(sol.extras["aim_x_mm"] - x_mm, sol.extras["aim_y_mm"] - y_mm),
            sol.extras["aim_z_mm"],
        )
    )


def _clean(x) -> float | None:
    """JSON-safe float. Standard JSON has no NaN token (JS's ``JSON.parse``
    rejects it, even though Python's ``json.dumps`` happily emits one) -- a
    trace where nothing landed would otherwise produce a response the
    frontend cannot parse."""
    return None if x is None or not np.isfinite(x) else float(x)


def _zero_power_note(counters: dict, power_w: float | None) -> str | None:
    """Why nothing arrived, when nothing arrived.

    A geometry that delivers no light is a legitimate answer -- an aperture
    too small for the field, a secondary the beam cannot clear -- but an
    empty flux map does not say which. The tracer's own counters do, so turn
    them into a sentence rather than leaving the caller to guess at a blank
    picture.
    """
    if power_w is None or power_w > 0.0:
        return None
    blocked = int(counters.get("blocked", 0))
    masked = int(counters.get("masked", 0))
    hit_mirror = counters.get("hit_mirror")
    reached = counters.get("reached_receiver")
    in_window = counters.get("in_window")

    if blocked and not masked:
        return (
            "No light reached the receiver: every sample was blocked before it "
            "got there. For a tower reflector that usually means the secondary "
            "cannot serve this field -- try a larger aperture radius, or move "
            "the secondary so the beam clears it."
        )
    if masked and not blocked:
        return (
            "No light reached the receiver: every sample fell outside the "
            "receiver window. Try a larger window, or check that the receiver "
            "is where the beam is aimed."
        )
    if reached is not None and in_window is not None and reached > 0 and in_window == 0:
        return (
            f"No light landed inside the window: {reached} rays reached the "
            "receiver plane but all fell outside it. A larger window, or a "
            "receiver nearer the aim point, would catch them."
        )
    if hit_mirror is not None and hit_mirror == 0:
        return (
            "No light reached the receiver: no ray even struck the mirror. "
            "Check the heliostat position and the sun direction."
        )
    return "No light reached the receiver with this geometry."


def _day_timesteps(req: "DayTraceRequest") -> list:
    """The day's sample times, from true sunrise to true sunset."""
    site = req.site
    cfg = SimpleNamespace(
        site=SimpleNamespace(
            latitude=site.latitude_deg,
            longitude=site.longitude_deg,
            timezone=site.timezone_h,
        ),
        sweep=SimpleNamespace(
            hour_step=req.hour_step,
            sunrise_margin_min=req.sunrise_margin_min,
            dates=[_dt.date(site.year, site.month, site.day)],
        ),
    )
    return build_time_grid(cfg, [_dt.date(site.year, site.month, site.day)])


def _trace_instant_metrics(
    req: "DayTraceRequest", solar_az_deg: float, solar_el_deg: float
) -> dict:
    """Power and spot metrics at one instant, for one heliostat or a field.

    Built from the same helpers the single and field endpoints use --
    :func:`_solve_for`, :func:`_build_trace_design`, :func:`_field_occlusion`
    and :func:`_trace_core` -- so a day's numbers are the numbers those
    endpoints would report, timestep by timestep. Nothing here re-implements
    physics; it only skips the parts a time series has no use for (the flux
    PNG, the 3-D scene, the per-heliostat table).
    """
    optics_params = resolve_optics_params(req.optics, req.optics_params)
    secondary, receiver = _geometry_for(req.optics, optics_params)
    mode = req.trace_mode()
    (u0, u1), (v0, v1) = receiver.uv_extent()
    u_edges = np.linspace(u0, u1, FLUX_GRID + 1)
    v_edges = np.linspace(v0, v1, FLUX_GRID + 1)
    bin_area_m2 = ((u1 - u0) / FLUX_GRID / 1000.0) * ((v1 - v0) / FLUX_GRID / 1000.0)

    if req.layout is None:
        xy_mm = np.array([[req.heliostat_x_mm, req.heliostat_y_mm]], dtype=float)
        ids = [0]
    else:
        xy_mm, ids = _field_positions(req.layout, req.exclude_ids)

    solutions = [
        _solve_for(req.optics, float(x), float(y), solar_az_deg, solar_el_deg, optics_params)
        for x, y in xy_mm
    ]
    designs = [
        _build_trace_design(req.design, sol, _slant_range_mm(sol, float(x), float(y)))
        for sol, (x, y) in zip(solutions, xy_mm)
    ]

    if len(ids) > 1:
        eta_shade, eta_block, eta_union, _outline = _field_occlusion(
            xy_mm, ids, solutions, designs[0], solar_az_deg, solar_el_deg
        )
    else:
        ones = np.ones(len(ids))
        eta_shade = eta_block = eta_union = ones

    flux = np.zeros((FLUX_GRID, FLUX_GRID))
    power_w = 0.0
    for i in range(len(ids)):
        result = _trace_core(
            designs[i],
            float(xy_mm[i, 0]),
            float(xy_mm[i, 1]),
            solutions[i],
            solar_az_deg,
            solar_el_deg,
            secondary,
            receiver,
            mode,
            mc_seed=np.random.SeedSequence((FIELD_MC_SEED, int(ids[i]))),
            mc_return_paths=False,
            slope_error_mrad=req.design.slope_error_mrad,
            specularity_mrad=req.design.specularity_mrad,
            reflectance=req.design.reflectance,
        )
        eta = float(eta_union[i])
        if result["backend"] == "mc":
            counts, _, _ = np.histogram2d(result["xy"][1], result["xy"][0], bins=[v_edges, u_edges])
            flux += counts * result["watts_per_ray"] / bin_area_m2 * eta
            power_w += result["watts_per_ray"] * result["counters"].get("in_window", 0) * eta
        else:
            flux += result["flux"] * eta
            power_w += result["power_w"] * eta

    rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)
    return {
        "power_w": float(power_w),
        "peak_flux_kw_m2": float(np.max(flux)) / 1000.0,
        "rms_radius_mm": rms_mm,
        "centroid_mm": list(centroid),
        "eta_shade_mean": float(np.mean(eta_shade)),
        "eta_block_mean": float(np.mean(eta_block)),
        "eta_mean": float(np.mean(eta_union)),
        "n_heliostats": len(ids),
    }


def _flux_grid_for(body: "TraceRequest") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flux_w_m2, u_edges, v_edges)`` for one single-heliostat request.

    The same solve/design/trace path :func:`_trace_core` gives the trace
    endpoint, kept separate only so an export does not have to render a PNG
    or build a scene to get at the numbers behind them.
    """
    optics_params = resolve_optics_params(body.optics, body.optics_params)
    secondary, receiver = _geometry_for(body.optics, optics_params)
    sol = _solve_for(
        body.optics,
        body.heliostat_x_mm,
        body.heliostat_y_mm,
        body.solar_az_deg,
        body.solar_el_deg,
        optics_params,
    )
    design = _build_trace_design(
        body.design, sol, _slant_range_mm(sol, body.heliostat_x_mm, body.heliostat_y_mm)
    )
    result = _trace_core(
        design,
        body.heliostat_x_mm,
        body.heliostat_y_mm,
        sol,
        body.solar_az_deg,
        body.solar_el_deg,
        secondary,
        receiver,
        body.trace_mode(),
        mc_return_paths=False,
        slope_error_mrad=body.design.slope_error_mrad,
        specularity_mrad=body.design.specularity_mrad,
        reflectance=body.design.reflectance,
    )
    if result["backend"] == "mc":
        flux, u_edges, v_edges, _rms, _cen = _mc_flux_and_metrics(
            result["xy"], result["watts_per_ray"], receiver
        )
        return flux, u_edges, v_edges
    return result["flux"], result["u_edges"], result["v_edges"]


def _render_day_png(steps: list[dict]) -> bytes:
    """Collected power and peak flux against local time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = [s["hour"] for s in steps]
    power_kw = [s["power_w"] / 1000.0 for s in steps]
    peak = [s["peak_flux_kw_m2"] for s in steps]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(hours, power_kw, "o-", color="#d97b29", label="collected power")
    ax.set_xlabel("local time (h)")
    ax.set_ylabel("collected power (kW)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    twin = ax.twinx()
    twin.plot(hours, peak, "s--", color="#3b6ea5", markersize=4, label="peak flux")
    twin.set_ylabel("peak flux (kW/m²)")
    twin.set_ylim(bottom=0)

    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110)
    finally:
        plt.close(fig)
    return buf.getvalue()


def _day_energy_kwh(steps: list[dict]) -> float:
    """Integrate collected power over the day, trapezoidally in local time.

    Trapezoid rather than a rectangle per sample because the samples land on
    the daylight edges, where power is near zero: a rectangle rule would
    charge each edge sample a full interval of its own value and overstate
    the ends.
    """
    if len(steps) < 2:
        return 0.0
    hours = np.array([s["hour"] for s in steps], dtype=float)
    power_kw = np.array([s["power_w"] for s in steps], dtype=float) / 1000.0
    return float(np.trapz(power_kw, hours))


#: Candidate contour spacings for the sag map, millimetres, smallest first.
_SAG_CONTOUR_INTERVALS_MM = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
#: Contour-line budget the interval is chosen against -- a manuscript rect's
#: ~9 mm peak-to-valley span lands on the 1.0 mm interval (9 lines), the
#: largest spacing that still keeps a hand-picked "roughly a dozen" ceiling.
_SAG_MAX_CONTOUR_LINES = 12


def _sag_contour_interval_mm(span_mm: float) -> float:
    """Smallest interval from :data:`_SAG_CONTOUR_INTERVALS_MM` giving at
    most :data:`_SAG_MAX_CONTOUR_LINES` contour lines across ``span_mm``.

    Falls back to the coarsest interval for a span so large that even that
    one draws more lines than the budget -- still the smallest overshoot
    available, rather than raising over a legitimately huge figure.
    """
    for interval in _SAG_CONTOUR_INTERVALS_MM:
        if span_mm / interval <= _SAG_MAX_CONTOUR_LINES:
            return interval
    return _SAG_CONTOUR_INTERVALS_MM[-1]


def _render_sag_png(
    design, sol, params, half_x_mm: float, half_y_mm: float
) -> tuple[bytes, float | None, float | None]:
    """Sag map of the mirror a trace would use, in millimetres.

    "Sag" is how far the reflecting surface departs from the flat plane
    through its own vertex -- the shape that turns a mirror into a lens.
    It is millimetres over metres of aperture, invisible in the 3-D scene
    (which draws facets flat for exactly that reason), so it gets its own
    view.

    Sampled from the same objects the trace uses: for the legacy path the
    solve's own astigmatic coefficients, for a design each facet's surface
    evaluated in that facet's frame. Points outside every facet -- the gaps
    in a grid, the space between petals -- are left blank rather than
    filled with the value a facet would have had if it were there.

    :returns: ``(png_bytes, peak_to_valley_mm, contour_interval_mm)``. Both
        numbers are ``None`` when no facet covers any sampled point ("no
        surface here"); ``contour_interval_mm`` is additionally ``None`` for
        a flat mirror (span at or below float noise), which draws no
        contours at all -- there is nothing for a spacing to describe.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 241
    xs = np.linspace(-half_x_mm, half_x_mm, n)
    ys = np.linspace(-half_y_mm, half_y_mm, n)
    gx, gy = np.meshgrid(xs, ys)
    sag = np.full(gx.shape, np.nan)

    if design is None:
        # Legacy single mirror: the tracer negates c4/c5 for its inherited
        # frame, so the surface actually traced is (c3, -c4, -c5).
        flat = zernike_sag_and_slopes(gx.ravel(), gy.ravel(), sol.c3, -sol.c4, -sol.c5)[0]
        sag = flat.reshape(gx.shape)
        inside = (np.abs(gx) <= half_x_mm) & (np.abs(gy) <= half_y_mm)
        sag = np.where(inside, sag, np.nan)
    else:
        for facet in design.facets:
            du = gx - facet.offset_mm[0]
            dv = gy - facet.offset_mm[1]
            inside = np.asarray(facet.region.contains(du.ravel(), dv.ravel())).reshape(gx.shape)
            if not inside.any():
                continue
            values = facet.surface.sag_and_slopes(du.ravel(), dv.ravel())[0].reshape(gx.shape)
            sag = np.where(inside, values, sag)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    finite = np.isfinite(sag)
    span: float | None = None
    interval: float | None = None
    if not finite.any():
        ax.text(0.5, 0.5, "no surface here", ha="center", va="center", transform=ax.transAxes)
    else:
        span = float(np.nanmax(sag) - np.nanmin(sag))
        im = ax.imshow(
            sag,
            origin="lower",
            cmap="jet",
            extent=(-half_x_mm, half_x_mm, -half_y_mm, half_y_mm),
            aspect="equal",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("sag (mm)")
        # Contours make a smooth figure legible; pointless on a flat mirror,
        # where the whole map is one value and matplotlib would warn.
        if span > 1e-9:
            interval = _sag_contour_interval_mm(span)
            lo, hi = float(np.nanmin(sag)), float(np.nanmax(sag))
            ax.contour(
                gx,
                gy,
                sag,
                levels=np.arange(lo, hi + interval, interval),
                colors="white",
                linewidths=0.4,
                alpha=0.6,
            )
            ax.set_title(f"peak-to-valley {span:.3f} mm · contours every {interval:g} mm")
        else:
            ax.set_title(f"peak-to-valley {span:.3f} mm")
    ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110)
    finally:
        plt.close(fig)
    return buf.getvalue(), span, interval


def _render_flux_png(
    flux: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray, mode: str, elapsed_ms: float
) -> bytes:
    # Lazy import, same reasoning as HeliostatDesign.preview(): matplotlib
    # is a real dependency but no other endpoint in this module needs it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    # Displayed in kW/m2: a tower receiver runs in the hundreds to
    # thousands of kW/m2, and five- and six-digit W/m2 tick labels are
    # harder to read than the two- and three-digit kW/m2 ones. The stored
    # and returned arrays stay in W/m2 -- this is a display unit only.
    im = ax.imshow(
        flux / 1000.0,
        origin="lower",
        cmap="magma",
        extent=(float(u_edges[0]), float(u_edges[-1]), float(v_edges[0]), float(v_edges[-1])),
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("kW/m²")
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
# field tracing
#
# The physics here is heliostat.sweep's, one timestep wide: per-heliostat
# aiming solves, silhouette-aware polygon occlusion against a neighbour list,
# one trace per heliostat, occlusion applied as a scalar on each heliostat's
# contribution. What this module does NOT reuse is run_sweep itself -- that
# function owns a RunStore, a time grid and a worker pool, none of which a
# browser request has or wants. The pieces underneath it are imported
# directly instead, so the two paths share the geometry rather than a
# workflow.


def _field_positions(layout, exclude_ids: list[int]) -> tuple[np.ndarray, list[int]]:
    """``(xy_mm, ids)`` for one layout, with ``exclude_ids`` dropped.

    Ids are layout indices and survive exclusion unchanged, so heliostat 7 is
    the same mirror whether or not heliostat 3 was dropped.
    """
    xy = layout.positions_mm()
    n = xy.shape[0]
    excluded = sorted(set(int(i) for i in exclude_ids))
    out_of_range = [i for i in excluded if not 0 <= i < n]
    if out_of_range:
        raise ValueError(
            f"exclude_ids {out_of_range} are not heliostat indices of this "
            f"{n}-heliostat layout (valid: 0..{n - 1})"
        )
    keep = [i for i in range(n) if i not in set(excluded)]
    if not keep:
        raise ValueError("exclude_ids drops every heliostat -- nothing left to trace")
    return xy[keep], keep


def _solve_field(
    optics: str,
    optics_params: OpticsParams,
    design: RectParams | GridParams | FlowerParams,
    xy_mm: np.ndarray,
    solar_az_deg: float,
    solar_el_deg: float,
) -> tuple[list[Solution], list[HeliostatDesign | None], list[float]]:
    """One pointing/figure solve and one design per heliostat -- ``/api/field/trace``'s
    former "phase 1", factored out so it and ``/api/scene/geometry`` share the
    identical loop rather than two copies that could drift. Everything past
    this (shading/blocking, the receiver trace) is trace-only business and
    stays in ``/api/field/trace``; the geometry endpoint has no use for it.

    :returns: ``(solutions, designs, slants)``, index-aligned with ``xy_mm``.
    """
    n = xy_mm.shape[0]
    solutions = [
        _solve_for(
            optics,
            float(xy_mm[i, 0]),
            float(xy_mm[i, 1]),
            solar_az_deg,
            solar_el_deg,
            optics_params,
        )
        for i in range(n)
    ]
    slants = [
        _slant_range_mm(solutions[i], float(xy_mm[i, 0]), float(xy_mm[i, 1])) for i in range(n)
    ]
    designs = [_build_trace_design(design, solutions[i], slants[i]) for i in range(n)]
    return solutions, designs, slants


def _field_geometry(design: HeliostatDesign | None):
    """``(region, outline_local_mm, half_width_mm, half_height_mm)`` for the mirror.

    One silhouette for the whole field. A design's silhouette depends on its
    facet regions and their offsets -- neither of which moves when a
    heliostat's own slant range changes its figure or its cant -- so every
    heliostat of a given design presents the identical outline, and building
    it once turns 600 radial silhouette traces into one. ``design=None`` is
    the tracer's legacy rectangle: ``region`` is ``None`` there, exactly as
    :func:`~heliostat.geometry.shading.build_geometries` leaves it, so
    occlusion falls back to the rectangle bounds test rather than a polygon
    that merely describes the same rectangle.

    The half-width/half-height pair is the same conservative *envelope*
    :meth:`~heliostat.geometry.shading.MirrorGeometry.from_design` computes
    (``max(|u0|, |u1|)``, not ``(u1 - u0) / 2``), since it sizes the raster
    window and the neighbour search radius rather than the material.
    """
    if design is None:
        outline = np.array(
            [
                [-MIRROR_HALF_X_MM, -MIRROR_HALF_Y_MM],
                [MIRROR_HALF_X_MM, -MIRROR_HALF_Y_MM],
                [MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM],
                [-MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM],
            ]
        )
        return None, outline, MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM

    u0, u1, v0, v1 = design.bbox
    facets = design.facets
    single_polygon = len(facets) == 1 and isinstance(facets[0].region, Polygon)
    if single_polygon and facets[0].offset_mm == (0.0, 0.0):
        # A custom design's single facet already IS its own outline -- the
        # vertices a caller drew, exactly. Radial-tracing it through
        # silhouette() below would only resample those same corners into
        # OCCLUDER_SILHOUETTE_VERTICES approximations of themselves, so a
        # hand-authored hexagon comes back as a hexagon, not a 72-gon that
        # merely looks like one.
        region = facets[0].region
    else:
        region = design.silhouette(OCCLUDER_SILHOUETTE_VERTICES)
    return region, region.vertices_mm, max(abs(u0), abs(u1)), max(abs(v0), abs(v1))


def _field_occlusion(
    xy_mm: np.ndarray,
    ids: list[int],
    solutions: list[Solution],
    design: HeliostatDesign | None,
    solar_az_deg: float,
    solar_el_deg: float,
):
    """Per-heliostat shading/blocking etas, exactly as ``heliostat.sweep`` gets them.

    Mirror geometry is built at each heliostat's own solved pointing, sharing
    the one silhouette from :func:`_field_geometry` (``region=None`` for the
    legacy rectangle, which is what
    :func:`~heliostat.geometry.shading.build_geometries` would have produced).
    Occluders are limited to a neighbour list sized by
    :func:`~heliostat.geometry.shading.search_radius_for` at this sun
    elevation *and* this field's flattest reflected beam -- the beam term
    matters, since blocking reach does not shrink as the sun climbs and
    omitting it loses real blockers at high sun. The secondary is not
    modelled as a shading body --
    ``polygon_occlusion(secondary=None)``, matching the sweep's own
    ``"traced_secondary": false``.

    :returns: ``(eta_shade, eta_block, eta_union)``. ``eta_union`` is the
        one applied to power: shaded and blocked areas overlap, so the union
        is the fraction of aperture actually delivering, and
        ``eta_shade * eta_block`` would charge that overlap twice.
    """
    region, outline, half_w, half_h = _field_geometry(design)

    geometries = []
    for i in range(xy_mm.shape[0]):
        normal = normal_from_angles(solutions[i].rot_az_deg, solutions[i].rot_el_deg)
        u, v = mirror_basis(normal)
        geometries.append(
            MirrorGeometry(
                centre=np.array([float(xy_mm[i, 0]), float(xy_mm[i, 1]), 0.0]),
                normal=normal,
                u=u,
                v=v,
                half_width=half_w,
                half_height=half_h,
                region=region,
            )
        )

    field = HeliostatField(
        x_mm=xy_mm[:, 0],
        y_mm=xy_mm[:, 1],
        ids=np.asarray(ids, dtype=int),
        mirror_width_mm=2.0 * half_w,
        mirror_height_mm=2.0 * half_h,
    )
    aims = aim_points_mm(solutions)
    centres = np.array([g.centre for g in geometries])
    neighbours = neighbour_pairs(
        field,
        search_radius_for(
            solar_el_deg,
            2.0 * half_h,
            2.0 * half_w,
            beam_elevation_deg=min_beam_elevation_deg(centres, aims),
        ),
    )
    eta_shade, eta_block, _eta_secondary, eta_union = polygon_occlusion(
        geometries, aims, solar_az_deg, solar_el_deg, neighbours
    )
    return eta_shade, eta_block, eta_union, outline


# ---------------------------------------------------------------------------
# app


def create_app():
    """Build the FastAPI app. Import-guarded: see the module docstring."""
    app = FastAPI(title="heliostat", version=__version__)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    # Additive: `/` keeps serving index.html through the explicit route
    # above exactly as before. This only adds a second, ordinary way to
    # reach the same files (and anything else dropped in static/, such as
    # the branding lockup in docs/ui-spec.md 5b) at their own paths, for
    # future markup that wants to reference `/static/...` directly rather
    # than everything being inlined into one served HTML string.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"version": __version__})

    @app.get("/api/field/manuscript")
    def field_manuscript() -> JSONResponse:
        """The paper's own 643-heliostat field, verbatim.

        Loaded from the packaged copy of ``examples/paper/data/field_645.csv``
        through the exact loader the paper's ``reproduce.py`` uses
        (:func:`heliostat.field.load_field`), so the two coincident-duplicate
        pairs (144=192, 241=289) are dropped here exactly as they are there --
        this endpoint cannot report a field that disagrees with the paper's
        own runs.

        ``ids`` are the loader's own surviving ids: 0-based file-row numbers,
        with the dropped duplicates' ids missing (so they run 0..644 with two
        gaps, not a clean 0..642). This is honest about what the ids mean and
        NOT what a trace renumbers them to -- a
        ``{"type": "positions", "xy_mm": ...}`` layout (the shape this
        endpoint's ``xy_mm`` feeds directly into) assigns every heliostat a
        fresh id 0..n-1 by array position when it traces, so an id reported
        here is not necessarily the id a trace on the same field reports back
        for the same mirror. ``xy_mm``'s order is preserved either way, which
        is what a trace/positions round-trip actually depends on.
        """
        field = _load_manuscript_field()
        xy_mm = [[round(float(x), 1), round(float(y), 1)] for x, y in field.xy_mm]
        return JSONResponse(
            {
                "xy_mm": xy_mm,
                "ids": [int(i) for i in field.ids],
                "n": len(field),
            }
        )

    @app.get("/api/setups")
    def setups_list() -> JSONResponse:
        return JSONResponse({"setups": list_setups()})

    @app.post("/api/setups")
    def setups_save(body: SetupRequest) -> JSONResponse:
        try:
            saved = save_setup(body.name, body.document)
        except SetupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:  # full disk, permissions, read-only home
            raise HTTPException(status_code=500, detail=f"could not save: {exc}") from exc
        return JSONResponse(saved)

    @app.get("/api/setups/{name}")
    def setups_load(name: str) -> JSONResponse:
        try:
            return JSONResponse(load_setup(name))
        except SetupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/setups/{name}")
    def setups_delete(name: str) -> JSONResponse:
        try:
            delete_setup(name)
        except SetupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"deleted": name})

    @app.post("/api/day/start")
    def day_start(body: DayTraceRequest) -> JSONResponse:
        """Trace one date end to end, on a background thread.

        Returns a job id immediately. A day is dozens of timesteps and a
        field is hundreds of mirrors, so this is minutes of work -- far too
        long to hold a request open with nothing to show. Poll
        ``/api/day/status/{job_id}``.
        """
        try:
            resolve_optics_params(body.optics, body.optics_params)
            steps = _day_timesteps(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not steps:
            raise HTTPException(
                status_code=422,
                detail=(
                    "the sun does not rise at that site on that date, so there is nothing to trace"
                ),
            )

        def work(job):
            rows = []
            for index, step in enumerate(steps):
                if job.cancelled():
                    break
                job.detail = f"{step.key} ({step.solar_el_deg:.1f}° elevation)"
                metrics = _trace_instant_metrics(body, step.solar_az_deg, step.solar_el_deg)
                rows.append(
                    {
                        "key": step.key,
                        "hour": round(float(step.hour), 4),
                        "solar_az_deg": round(float(step.solar_az_deg), 3),
                        "solar_el_deg": round(float(step.solar_el_deg), 3),
                        **{
                            k: (None if v is None or not np.isfinite(v) else round(float(v), 4))
                            for k, v in metrics.items()
                            if k not in ("centroid_mm", "n_heliostats")
                        },
                        "n_heliostats": metrics["n_heliostats"],
                    }
                )
                job.done = index + 1
            return {
                "steps": rows,
                "energy_kwh": round(_day_energy_kwh(rows), 3),
                "date": f"{body.site.year:04d}-{body.site.month:02d}-{body.site.day:02d}",
                "mode": body.mode,
                "optics": body.optics,
                "n_heliostats": rows[0]["n_heliostats"] if rows else 0,
            }

        job = JOBS.start(len(steps), work, label=f"day trace, {len(steps)} timesteps")
        return JSONResponse(job.snapshot())

    @app.get("/api/day/status/{job_id}")
    def day_status(job_id: str) -> JSONResponse:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return JSONResponse(job.snapshot())

    @app.post("/api/day/cancel/{job_id}")
    def day_cancel(job_id: str) -> JSONResponse:
        if not JOBS.cancel(job_id):
            raise HTTPException(status_code=409, detail="that job is not running")
        return JSONResponse({"cancelled": job_id})

    @app.get("/api/day/result/{job_id}")
    def day_result(job_id: str) -> JSONResponse:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        payload = dict(job.result or {})
        payload["state"] = job.state
        payload["elapsed_s"] = round(job.elapsed_s, 2)
        if payload.get("steps"):
            payload["plot_png"] = base64.b64encode(_render_day_png(payload["steps"])).decode(
                "ascii"
            )
        return JSONResponse(payload)

    @app.get("/api/day/export/{job_id}.csv")
    def day_export(job_id: str) -> Response:
        """The day's numbers as CSV, for analysis somewhere else."""
        job = JOBS.get(job_id)
        if job is None or not (job.result or {}).get("steps"):
            raise HTTPException(status_code=404, detail="no finished run with that id")
        rows = job.result["steps"]
        columns = list(rows[0].keys())
        out = StringIO()
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="heliostat-day-{job.result.get("date", job_id)}.csv"'
                )
            },
        )

    @app.post("/api/trace/flux.csv")
    def trace_flux_csv(body: TraceRequest) -> Response:
        """The flux map of one trace as CSV, in kW/m2.

        Row and column headers are the receiver-plane coordinates of each
        bin centre in millimetres, so the grid is self-describing rather
        than a bare block of numbers whose axes live in another document.
        """
        flux, u_edges, v_edges = _flux_grid_for(body)
        u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
        v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
        out = StringIO()
        writer = csv.writer(out)
        writer.writerow([r"v_mm \ u_mm"] + [f"{u:.1f}" for u in u_mid])
        for row_index, v in enumerate(v_mid):
            writer.writerow([f"{v:.1f}"] + [f"{x / 1000.0:.6g}" for x in flux[row_index]])
        return Response(
            content=out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="heliostat-flux-kW_m2.csv"'},
        )

    @app.post("/api/sun")
    def sun(body: SunRequest) -> JSONResponse:
        """Sun azimuth/elevation for a site and moment, plus that day's
        sunrise and sunset so the caller can offer a sensible time range."""
        try:
            az, el = sun_position(
                body.latitude_deg,
                body.longitude_deg,
                body.timezone_h,
                body.year,
                body.month,
                body.day,
                body.hour,
            )
            rise, set_ = sunrise_sunset(
                body.latitude_deg,
                body.longitude_deg,
                body.timezone_h,
                body.year,
                body.month,
                body.day,
            )
        except ValueError as exc:  # e.g. 31 February
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(
            {
                "solar_az_deg": round(float(az), 3),
                "solar_el_deg": round(float(el), 3),
                "sunrise_h": None if not np.isfinite(rise) else round(float(rise), 3),
                "sunset_h": None if not np.isfinite(set_) else round(float(set_), 3),
                "above_horizon": bool(el > 0.0),
            }
        )

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

    @app.post("/api/design/sag")
    def design_sag(body: TraceRequest) -> Response:
        """Sag map of the mirror this exact request would trace.

        Takes a full trace request because the figure depends on the solve:
        a twisting mirror's astigmatism is a function of where the sun is
        and where the heliostat stands, so there is no sag to draw without
        them.
        """
        if body.solar_el_deg <= 0:
            raise HTTPException(
                status_code=422,
                detail="solar_el_deg must be > 0 (the sun is below the horizon)",
            )
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
            sol = _solve_for(
                body.optics,
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                body.solar_az_deg,
                body.solar_el_deg,
                optics_params,
            )
            slant_range_mm = _slant_range_mm(sol, body.heliostat_x_mm, body.heliostat_y_mm)
            design = _build_trace_design(body.design, sol, slant_range_mm)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if design is None:
            half_x, half_y = MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM
        else:
            u0, u1, v0, v1 = design.bbox
            half_x = max(abs(u0), abs(u1))
            half_y = max(abs(v0), abs(v1))
        png, span_mm, interval_mm = _render_sag_png(design, sol, body.design, half_x, half_y)
        headers = {"X-Slant-Range-M": f"{slant_range_mm / 1000.0:.3f}"}
        if span_mm is not None:
            headers["X-Peak-To-Valley-Mm"] = f"{span_mm:.6g}"
        if interval_mm is not None:
            headers["X-Contour-Interval-Mm"] = f"{interval_mm:g}"
        return Response(content=png, media_type="image/png", headers=headers)

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
        slant_range_mm = _slant_range_mm(sol, body.heliostat_x_mm, body.heliostat_y_mm)

        # Pointing AND figure both come from the solve; which figure is
        # decided by the design's `surface` axis -- see _build_trace_design
        # for the legacy-path rule and _DesignBase for surface vs cant.
        try:
            design = _build_trace_design(body.design, sol, slant_range_mm)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics, optics_params)
        rot_az_deg, rot_el_deg = sol.rot_az_deg, sol.rot_el_deg

        mode = body.trace_mode()
        t0 = time.perf_counter()
        result = _trace_core(
            design,
            body.heliostat_x_mm,
            body.heliostat_y_mm,
            sol,
            body.solar_az_deg,
            body.solar_el_deg,
            secondary,
            receiver,
            mode,
            slope_error_mrad=body.design.slope_error_mrad,
            specularity_mrad=body.design.specularity_mrad,
            reflectance=body.design.reflectance,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if result["backend"] == "mc":
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

        return JSONResponse(
            {
                "power_w": _clean(power_w),
                "incident_power_w": _clean(incident_power_w),
                "note": _zero_power_note(counters, _clean(power_w)),
                "peak_flux_kw_m2": _clean(float(np.max(flux)) / 1000.0),
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

    @app.post("/api/field/trace")
    def field_trace(body: FieldTraceRequest) -> JSONResponse:
        """Trace a whole layout at one instant and sum it on the receiver.

        Per heliostat: its own aiming solve at its own position, its own
        design built at its own slant range, its own trace -- the identical
        calls ``/api/trace`` makes, through the same :func:`_trace_core`. On
        top of that, and only reachable with more than one mirror in the
        field: mutual shading and blocking, computed once for the whole field
        by :func:`_field_occlusion` and applied as a per-heliostat scalar on
        that heliostat's flux and power, which is the convention
        :mod:`heliostat.sweep` and the run store both use.

        Serial, one heliostat at a time. At n=100 ultra_fast that is a few
        seconds; making it fast is a separate piece of work, so the response
        carries ``timings_ms`` (solve, occlusion, trace, scene) to keep the
        cost measurable rather than anecdotal.

        Two honest differences from the single-heliostat endpoint, both
        consequences of summing rather than of different physics:

        * the combined spot's rms and centroid are moments of the SUMMED
          flux grid. For the cone backends that is the same recipe
          ``/api/trace`` uses. For Monte Carlo it is not: there, one
          heliostat's metrics come from exact hit coordinates, while a
          field's come from the 128x128 histogram those hits are summed in.
        * incident power is reported only for the cone backends, which
          measure it; Monte Carlo does not, and it stays ``null`` exactly as
          in a single trace.

        ``elapsed_ms`` is the solve + occlusion + trace total, i.e. what it
        cost to answer the physics question. Building the scene and rendering
        the flux PNG are reported separately in ``timings_ms`` rather than
        folded in, matching ``/api/trace``, whose ``elapsed_ms`` is likewise
        the trace alone.
        """
        if body.solar_el_deg <= 0:
            raise HTTPException(
                status_code=422,
                detail="solar_el_deg must be > 0 (the sun is below the horizon)",
            )

        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
            xy_mm, ids = _field_positions(body.layout, body.exclude_ids)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics, optics_params)
        mode = body.trace_mode()
        n = xy_mm.shape[0]

        # -- phase 1: one pointing/figure solve and one design per heliostat.
        # Shared with /api/scene/geometry via _solve_field, so the two cannot
        # trace and place the same field differently.
        t0 = time.perf_counter()
        try:
            solutions, designs, _slants = _solve_field(
                body.optics, optics_params, body.design, xy_mm, body.solar_az_deg, body.solar_el_deg
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        t_solve = time.perf_counter()

        # -- phase 2: mutual shading and blocking across the whole field.
        # designs[0] stands in for all of them: they differ only in figure
        # and cant focal, neither of which changes the outline an occluder
        # presents (see _field_geometry).
        eta_shade, eta_block, eta_union, outline = _field_occlusion(
            xy_mm, ids, solutions, designs[0], body.solar_az_deg, body.solar_el_deg
        )
        t_occlusion = time.perf_counter()

        # -- phase 3: one trace per heliostat, summed on the receiver.
        (u0, u1), (v0, v1) = receiver.uv_extent()
        u_edges = np.linspace(u0, u1, FLUX_GRID + 1)
        v_edges = np.linspace(v0, v1, FLUX_GRID + 1)
        bin_area_m2 = ((u1 - u0) / FLUX_GRID / 1000.0) * ((v1 - v0) / FLUX_GRID / 1000.0)

        flux = np.zeros((FLUX_GRID, FLUX_GRID))
        power_w = 0.0
        incident_power_w = 0.0 if mode.backend == "cone" else None
        counters: dict[str, float] = {}
        rows = []

        for i in range(n):
            result = _trace_core(
                designs[i],
                float(xy_mm[i, 0]),
                float(xy_mm[i, 1]),
                solutions[i],
                body.solar_az_deg,
                body.solar_el_deg,
                secondary,
                receiver,
                mode,
                # A field's rays are drawn by the scene's own side trace, so
                # the reported trace has no use for paths -- and 600
                # heliostats' worth of them is a lot of array for a picture
                # that will not use it.
                mc_seed=np.random.SeedSequence((FIELD_MC_SEED, int(ids[i]))),
                mc_return_paths=False,
                slope_error_mrad=body.design.slope_error_mrad,
                specularity_mrad=body.design.specularity_mrad,
                reflectance=body.design.reflectance,
            )
            eta = float(eta_union[i])
            if result["backend"] == "mc":
                counts, _, _ = np.histogram2d(
                    result["xy"][1], result["xy"][0], bins=[v_edges, u_edges]
                )
                own_power = result["watts_per_ray"] * result["counters"].get("in_window", 0)
                flux += counts * result["watts_per_ray"] / bin_area_m2 * eta
            else:
                own_power = result["power_w"]
                incident_power_w += result["incident_power_w"] * eta
                flux += result["flux"] * eta
            power_w += own_power * eta
            for k, v in result["counters"].items():
                counters[k] = counters.get(k, 0) + v
            rows.append(
                {
                    "id": int(ids[i]),
                    "x_mm": float(xy_mm[i, 0]),
                    "y_mm": float(xy_mm[i, 1]),
                    "eta_shade": float(eta_shade[i]),
                    "eta_block": float(eta_block[i]),
                    "eta": eta,
                    "power_w": _clean(own_power * eta),
                }
            )
        t_trace = time.perf_counter()

        rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)
        elapsed_ms = (t_trace - t0) * 1000.0
        png_bytes = _render_flux_png(flux, u_edges, v_edges, body.mode, elapsed_ms)

        scene = build_field_scene(
            [
                {
                    "id": int(ids[i]),
                    "x_mm": float(xy_mm[i, 0]),
                    "y_mm": float(xy_mm[i, 1]),
                    "rot_az_deg": solutions[i].rot_az_deg,
                    "rot_el_deg": solutions[i].rot_el_deg,
                    "c3": solutions[i].c3,
                    "c4": solutions[i].c4,
                    "c5": solutions[i].c5,
                    "design": designs[i],
                    "eta": float(eta_union[i]),
                }
                for i in range(n)
            ],
            outline,
            body.solar_az_deg,
            body.solar_el_deg,
            secondary,
            receiver,
        )
        t_scene = time.perf_counter()

        return JSONResponse(
            {
                "power_w": _clean(power_w),
                "incident_power_w": _clean(incident_power_w),
                "note": _zero_power_note(counters, _clean(power_w)),
                "peak_flux_kw_m2": _clean(float(np.max(flux)) / 1000.0),
                "rms_radius_mm": _clean(rms_mm),
                "centroid_mm": [_clean(centroid[0]), _clean(centroid[1])],
                "counters": {k: int(v) for k, v in counters.items()},
                "elapsed_ms": elapsed_ms,
                "timings_ms": {
                    "solve": (t_solve - t0) * 1000.0,
                    "occlusion": (t_occlusion - t_solve) * 1000.0,
                    "trace": (t_trace - t_occlusion) * 1000.0,
                    "scene": (t_scene - t_trace) * 1000.0,
                },
                "mode": body.mode,
                "flux_png": base64.b64encode(png_bytes).decode("ascii"),
                "n_heliostats": n,
                "eta_min": _clean(float(np.min(eta_union))),
                "eta_median": _clean(float(np.median(eta_union))),
                "eta_max": _clean(float(np.max(eta_union))),
                "optics_resolved": optics_params.model_dump(),
                "heliostats": rows,
                "scene": scene,
            }
        )

    @app.post("/api/scene/geometry")
    def scene_geometry(body: GeometryRequest) -> JSONResponse:
        """Where every mirror in a field stands and points -- no trace, no flux.

        This is ``/api/field/trace``'s phase 1 (:func:`_solve_field`) and
        nothing past it: no shading/blocking, no receiver trace, no flux map.
        The 3-D view calls this on every edit (docs/ui-spec.md 2.1, "Live
        from the first frame" and "Apply only where it's slow") -- placing
        and orienting mirrors is cheap enough to run live even for a field
        far too large to trace in a browser request, hence its own, larger
        cap (:data:`MAX_GEOMETRY_HELIOSTATS`, ten times the trace cap).

        The sun at or below the horizon is NOT an error here, unlike a
        trace. Every per-layout aiming solve (:func:`solve_prime_focus` and
        its siblings) divides by the sun's own elevation, so there is no
        pointing to compute -- but the view still has a field to show
        (docs/ui-spec.md 2.1: "Sun below horizon: scene never goes blank --
        heliostats hold their last pose, rays disappear, a banner
        explains"). This endpoint has no "last pose" to hold (it is not
        stateful), so the simplest honest answer is the one the scene
        already reports as ``null``-able: every heliostat comes back at its
        position with no orientation, ``rays`` is empty, and
        ``sun_below_horizon: true`` says why -- the client already has
        everything it needs to draw the banner and keep the last frame it
        drew, without this endpoint pretending to solve something that has
        no solution.

        ``miss`` is docs/ui-spec.md 2.3's amber "warning" tier: ``null``
        for prime focus (no secondary to miss), the sun below the horizon
        (no solved orientation to build a chief ray from), or an empty
        field; otherwise ``{needed_aperture_radius_mm, aperture_miss_ids,
        total_miss_ids, rays}`` from :func:`~heliostat.web.scene.field_miss_detection`
        plus the dropped-corner-ray polylines
        :func:`~heliostat.web.scene.build_geometry_scene` collects from the
        same strided sources as its own ``rays``. Nothing here is adjusted
        automatically -- the geometry solve above is untouched; this is
        purely a report on it.
        """
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
            if body.layout is None:
                xy_mm = np.array([[body.heliostat_x_mm, body.heliostat_y_mm]], dtype=float)
                ids: list[int] = [0]
            else:
                xy_mm, ids = _field_positions(body.layout, [])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        secondary, receiver = _geometry_for(body.optics, optics_params)
        sun_below_horizon = body.solar_el_deg <= 0.0

        if sun_below_horizon:
            heliostats = [
                {
                    "id": int(i),
                    "x_mm": float(x),
                    "y_mm": float(y),
                    "rot_az_deg": None,
                    "rot_el_deg": None,
                    "c3": None,
                    "c4": None,
                    "c5": None,
                    "design": None,
                    "slant_range_m": None,
                }
                for i, (x, y) in zip(ids, xy_mm)
            ]
            # No solve means no real design-driven silhouette either -- a
            # design's outline does not depend on the solve, but building
            # one only to draw no rays with it is pointless below the
            # horizon. The legacy rectangle's outline is a fine stand-in for
            # "some outline exists to place the (unoriented) mirrors with".
            _region, outline, _hw, _hh = _field_geometry(None)
        else:
            try:
                solutions, designs, slants = _solve_field(
                    body.optics,
                    optics_params,
                    body.design,
                    xy_mm,
                    body.solar_az_deg,
                    body.solar_el_deg,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            heliostats = [
                {
                    "id": int(ids[i]),
                    "x_mm": float(xy_mm[i, 0]),
                    "y_mm": float(xy_mm[i, 1]),
                    "rot_az_deg": solutions[i].rot_az_deg,
                    "rot_el_deg": solutions[i].rot_el_deg,
                    "c3": solutions[i].c3,
                    "c4": solutions[i].c4,
                    "c5": solutions[i].c5,
                    "design": designs[i],
                    "slant_range_m": round(slants[i] / 1000.0, 3),
                }
                for i in range(len(ids))
            ]
            # designs[0] stands in for the whole field's silhouette, exactly
            # as /api/field/trace's own occlusion pass assumes -- see
            # _field_geometry's docstring for why every heliostat of one
            # design presents the identical outline.
            _region, outline, _hw, _hh = _field_geometry(designs[0])

        scene = build_geometry_scene(
            heliostats,
            outline,
            body.solar_az_deg,
            body.solar_el_deg,
            secondary,
            receiver,
            include_corner_rays=body.include_corner_rays,
            max_corner_sources=body.max_corner_sources,
            sun_below_horizon=sun_below_horizon,
            include_miss_rays=True,
        )
        # The amber "warning" tier (docs/ui-spec.md 2.3): null for prime
        # focus (field_miss_detection's own NoSecondary check), an empty
        # field, or the sun below the horizon -- there is no solved
        # orientation to build a chief ray from then, so this must not be
        # called (see field_miss_detection's docstring). The dropped-ray
        # polylines come from build_geometry_scene's own strided corner-ray
        # sources, not a second pass over all 10,000 heliostats.
        miss_rays = scene.pop("miss_rays", [])
        miss = None if sun_below_horizon else field_miss_detection(
            heliostats, body.solar_az_deg, body.solar_el_deg, secondary, receiver
        )
        if miss is not None:
            miss["rays"] = miss_rays
        scene["miss"] = miss
        scene["optics_resolved"] = optics_params.model_dump()
        return JSONResponse(scene)

    # -- library: designs, receivers, projects ------------------------------

    @app.get("/api/library/{collection}")
    def library_list(collection: str) -> JSONResponse:
        _require_known_collection(collection)
        entries = [{"name": name, "builtin": True} for name in _BUILTIN_LIBRARY[collection]]
        entries += [
            {"name": e["name"], "builtin": False, "saved_at": e["saved_at"]}
            for e in list_entries(collection)
        ]
        return JSONResponse({"entries": entries})

    @app.post("/api/library/{collection}")
    def library_save(collection: str, body: LibrarySaveRequest) -> JSONResponse:
        _require_known_collection(collection)
        if body.name in _BUILTIN_LIBRARY[collection]:
            raise HTTPException(
                status_code=409,
                detail=f"{body.name!r} is a built-in {collection[:-1]} and cannot be overwritten",
            )
        try:
            _validate_library_document(collection, body.document)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            saved = save_entry(collection, body.name, body.document)
        except LibraryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:  # full disk, permissions, read-only home
            raise HTTPException(status_code=500, detail=f"could not save: {exc}") from exc
        return JSONResponse(saved)

    # {name:path} rather than the default {name}: FastAPI's default string
    # converter splits on "/", and one built-in name is genuinely
    # "Axicon 27 m / 20 deg / 14 m" (docs/ui-spec.md 5's own naming) -- a
    # plain {name} 404s on it, since the router sees an extra path segment
    # rather than one name containing a slash.
    @app.get("/api/library/{collection}/{name:path}")
    def library_load(collection: str, name: str) -> JSONResponse:
        _require_known_collection(collection)
        if name in _BUILTIN_LIBRARY[collection]:
            return JSONResponse(
                {"name": name, "builtin": True, "document": _BUILTIN_LIBRARY[collection][name]}
            )
        try:
            payload = load_entry(collection, name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({**payload, "builtin": False})

    @app.delete("/api/library/{collection}/{name:path}")
    def library_delete(collection: str, name: str) -> JSONResponse:
        _require_known_collection(collection)
        if name in _BUILTIN_LIBRARY[collection]:
            raise HTTPException(
                status_code=409,
                detail=f"{name!r} is a built-in {collection[:-1]} and cannot be deleted",
            )
        try:
            delete_entry(collection, name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"deleted": name})

    return app
