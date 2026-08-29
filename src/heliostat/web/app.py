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
solves (:func:`~heliostat.geometry.aiming.solve_prime_focus_to_receiver`,
:func:`~heliostat.geometry.aiming.solve_axicon`,
:func:`~heliostat.geometry.aiming.solve_cassegrain`), evaluated at the
requested sun position and heliostat position -- the same aiming and
focusing solves that reproduce the golden fixtures' pointing/figure
columns to machine precision (``tests/test_aiming.py``). The layout
constants passed to each solve (focus heights, axicon geometry) match
``_geometry_for``'s optics below exactly, since this module traces against
the identical fixture secondaries/receivers. Prime focus's own solve
targets whatever :func:`_prime_focus_receiver` resolves to (flat, on-axis
and identical to the golden fixture by default; cylindrical, frustum,
offset or off-axis when the request's ``optics_params`` says so).
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from typing import Annotated, ClassVar, Literal, Union

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
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

from heliostat import __version__, energy
from heliostat.field import HeliostatField, load_field, neighbour_pairs
from heliostat.field_layouts import generate, ring_filter
from heliostat.geometry.aiming import (
    Solution,
    aim_points_mm,
    solve_axicon,
    solve_cassegrain,
    solve_prime_focus_to_receiver,
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
from heliostat.geometry.errormap import ErrorMap, parse_error_map_csv
from heliostat.geometry.heliostat import zernike_sag_and_slopes
from heliostat.geometry.receiver import (
    ApertureClippedReceiver,
    CylinderReceiver,
    FlatWindowReceiver,
    FrustumReceiver,
    Receiver,
)
from heliostat.geometry.secondary import (
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
    secondary_bin_areas_m2,
    secondary_has_flux_map,
    secondary_nominal_sag_mm,
    secondary_uv,
    secondary_uv_extent,
    secondary_uv_to_world,
    secondary_warp_sag_mm,
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
from heliostat.dni import STANDARD_DNI, ClearSkyDNI, ConstantDNI, DNIProvider, ScaledDNI
from heliostat.solar import build_time_grid, sun_position, sunrise_sunset
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM, trace_heliostat
from heliostat.trace.modes import MODES, TraceMode
from heliostat.trace.samplers import BuieSampler
from heliostat.web.builtin_library import (
    BUILTIN_DESIGNS,
    BUILTIN_PROJECT_PROVENANCE,
    BUILTIN_PROJECTS,
    BUILTIN_RECEIVERS,
)
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

#: Cap on the adaptive u-bin count for curved receivers (below). 512 bounds
#: the deposit cost (a cone footprint touches ~4x the bins of a 128-wide
#: grid) and the flux-array size; the default cylinder and frustum land at
#: 448, comfortably inside it.
FLUX_GRID_MAX_U = 512


def _receiver_flux_grid(receiver) -> tuple[int, int]:
    """``(n_u, n_v)`` for a receiver's flux map.

    A flat window keeps the historical ``FLUX_GRID`` square. A curved
    receiver unrolls its FULL circumference into u while v spans only its
    height/slant, so a square grid leaves u-bins ~pi*(diameter/height) times
    coarser than v-bins (~146 mm vs 47 mm on the default cylinder) -- a
    single heliostat's spot lands in ~4 bins and every flux map goes stripy
    (both backends identically, since they share these edges). Scale n_u so
    u-bins match v-bins (square bins on the unrolled surface), rounded UP to
    a multiple of ``FLUX_GRID_TEXTURE_DIM`` so the drape texture's
    block-averaging divides evenly, clamped to
    ``[FLUX_GRID, FLUX_GRID_MAX_U]``.

    Deterministic from the receiver alone -- worker processes and the
    parent build their grids independently and MUST agree bin-for-bin.
    """
    if getattr(receiver, "is_planar", True):
        return (FLUX_GRID, FLUX_GRID)
    (u0, u1), (v0, v1) = receiver.uv_extent()
    ratio = (u1 - u0) / (v1 - v0)
    n_u = int(np.ceil(FLUX_GRID * ratio / FLUX_GRID_TEXTURE_DIM)) * FLUX_GRID_TEXTURE_DIM
    return (int(np.clip(n_u, FLUX_GRID, FLUX_GRID_MAX_U)), FLUX_GRID)


def _flux_edges(receiver) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The one shared ``(u_edges, v_edges, bin_area_m2)`` every flux-map
    producer bins against -- single vs field trace, MC vs cone, sweep
    steps -- so no two paths can disagree on resolution again."""
    (u0, u1), (v0, v1) = receiver.uv_extent()
    n_u, n_v = _receiver_flux_grid(receiver)
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    return u_edges, v_edges, receiver.bin_areas_m2((n_u, n_v))


def _secondary_flux_grid(secondary) -> tuple[int, int]:
    """``(n_u, n_v)`` for a secondary's own flux map.

    A secondary's ``(u, v)`` (see :mod:`heliostat.geometry.secondary`) is
    the same kind of unrolled parameterization a curved RECEIVER uses --
    full circumference in ``u``, a bounded span in ``v`` -- so this is the
    identical "square bins on the unrolled surface" rule
    :func:`_receiver_flux_grid` applies to a cylinder/frustum, deliberately
    NOT reusing that function's own edges: a secondary's aperture is its own
    surface with its own extent, unrelated to whatever the receiver's
    adaptive grid resolved to (a fresh change noted in this module's own
    history -- receiver flux edges now come from an adaptive grid for curved
    receivers, and a secondary must not silently inherit or collide with
    that unless its own geometry happens to agree).
    """
    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    ratio = (u1 - u0) / (v1 - v0)
    n_u = int(np.ceil(FLUX_GRID * ratio / FLUX_GRID_TEXTURE_DIM)) * FLUX_GRID_TEXTURE_DIM
    return (int(np.clip(n_u, FLUX_GRID, FLUX_GRID_MAX_U)), FLUX_GRID)


def _secondary_flux_edges(secondary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The secondary-map analogue of :func:`_flux_edges`: the one shared
    ``(u_edges, v_edges, bin_area_m2)`` every secondary-flux producer (single
    trace, field trace, MC vs cone) bins against."""
    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    n_u, n_v = _secondary_flux_grid(secondary)
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    return u_edges, v_edges, secondary_bin_areas_m2(secondary, (n_u, n_v))


def _mc_secondary_flux(
    secondary_xy: np.ndarray, watts_per_ray: float, secondary
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flux, u_edges, v_edges)`` W/m² on the secondary's own surface, from
    raw Monte Carlo secondary hits (``trace_heliostat(...,
    return_secondary_hits=True)``'s ``secondary_xy``) -- the secondary-map
    analogue of :func:`_mc_flux_and_metrics`. Exact accounting (every
    ray that struck the secondary lands in exactly one bin), unlike the cone
    backend's chief-point deposit -- see ``secondary_fidelity`` on the cone
    result."""
    u_edges, v_edges, bin_area_m2 = _secondary_flux_edges(secondary)
    # secondary_xy already carries the full (x, y, z) world point (see
    # heliostat.trace.mc.trace_heliostat's return_secondary_hits) -- needed
    # verbatim, not padded with a fake z=0, so secondary_uv can undo a spec
    # §E2 rigid-body misalignment exactly.
    uv = secondary_uv(secondary, secondary_xy)
    counts, _, _ = np.histogram2d(uv[1], uv[0], bins=[v_edges, u_edges])
    flux = counts * watts_per_ray / bin_area_m2
    return flux, u_edges, v_edges


def _secondary_maps_from_result(
    result: dict, secondary
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str] | None:
    """``(flux, u_edges, v_edges, power_w, fidelity)`` on the secondary's own
    surface from one :func:`_trace_core` result, or ``None`` when there is
    nothing to report -- the caller did not pass ``return_secondary_flux``,
    or ``secondary`` has no flux map (see
    :func:`~heliostat.geometry.secondary.secondary_has_flux_map`).

    ``fidelity`` is ``"exact"`` for Monte Carlo (every ray that struck the
    secondary is individually histogrammed, same as the receiver map) and
    ``"coarse"`` for the cone backends (chief-ray-point deposit -- spec §C's
    "coarse in cone modes, exact in Monte Carlo" disclosure, carried through
    verbatim from :func:`~heliostat.trace.cone.trace_heliostat_cone`).
    """
    if result["backend"] == "mc":
        if "secondary_xy" not in result:
            return None
        secondary_xy = result["secondary_xy"]
        watts_per_ray = result["watts_per_ray"]
        flux, u_edges, v_edges = _mc_secondary_flux(secondary_xy, watts_per_ray, secondary)
        power_w = float(secondary_xy.shape[1]) * watts_per_ray
        return flux, u_edges, v_edges, power_w, "exact"
    if "secondary_flux" not in result:
        return None
    return (
        result["secondary_flux"],
        result["secondary_u_edges"],
        result["secondary_v_edges"],
        result["secondary_power_w"],
        result["secondary_fidelity"],
    )


def _secondary_payload(
    flux: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    power_w: float,
    fidelity: str,
    secondary_reflectance: float,
    include_flux_grid: bool,
) -> dict:
    """Spec §C response block: the secondary's own irradiance map plus the
    absorbed-heat readout -- ``(1 - secondary_reflectance) * incident``,
    both as a total power and as a peak flux density, mirroring the
    receiver's own ``power_w``/``peak_flux_kw_m2`` pair so the client reads
    the two maps the same way. ``flux_grid`` (opt-in, same downsampling as
    :func:`_flux_grid_payload`) is what a future 3-D secondary drape would
    consume; this endpoint carries the numbers, the UI phase decides whether
    to draw them (mockup M9: a Receiver / Secondary map selector).
    """
    absorbed_fraction = 1.0 - secondary_reflectance
    peak_kw_m2 = float(np.max(flux)) / 1000.0 if flux.size else 0.0
    return {
        "power_w": _clean(power_w),
        "peak_flux_kw_m2": _clean(peak_kw_m2),
        "absorbed_power_w": _clean(power_w * absorbed_fraction),
        "peak_absorbed_kw_m2": _clean(peak_kw_m2 * absorbed_fraction),
        "secondary_reflectance": secondary_reflectance,
        # UI disclosure (spec §C / docs/secondary-irradiance-plan.md): the
        # cone backends' secondary deposit is a chief-ray-point
        # approximation ("coarse"), Monte Carlo's is per-ray exact.
        "fidelity": fidelity,
        "flux_grid": _flux_grid_payload(flux, u_edges, v_edges) if include_flux_grid else None,
    }

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
# Raised from 1000 to match MAX_GEOMETRY_HELIOSTATS (2026-08-29, owner's
# call): §P ships Gemasolar (2,650), Crescent Dunes (10,347) and a
# Stellio-based Hami field (14,500), and a tool that can draw a field but
# refuses to trace it is half a tool -- the same reasoning that set this
# above the companion paper's own 643 in the first place. Crescent Dunes
# is roughly sixteen times that field, so these runs are minutes rather
# than seconds; §Q's calibration exists to say so honestly before the
# caller commits. Slow remains the caller's choice to make, not something
# to forbid.
#: Kept equal to ``MAX_GEOMETRY_HELIOSTATS`` below (defined after this one,
#: so the value is written out rather than referenced).
MAX_FIELD_HELIOSTATS = 15_000

# /api/scene/geometry's own, much larger cap. Placing and orienting a mirror
# (one aiming solve, no shading/blocking, no receiver trace) costs nothing
# like tracing it does, and the 3-D view's whole reason to exist is showing a
# field too big to trace in a browser request -- docs/ui-spec.md 2.1 states
# the scale target explicitly: "smooth orbiting up to 10,000 heliostats".
# Ten thousand analytic solves is still a fraction of a second; the trace
# cap above is unaffected.
#
# Raised from 10,000 to 15,000 for docs/ui-spec-v0.2.md §P (signed off): the
# largest built-in reference project (the Stellio-based Hami reconstruction,
# heliostat.web.builtin_library.BUILTIN_PROJECTS) ships 14,500 heliostats --
# sbp's own official count for that plant, with sources disagreeing in a
# 14,000-15,000 range (see that module's provenance notes). This is purely
# the same validation ceiling moving to cover a real shipped field; the
# reasoning above (analytic solves only, no trace) is unchanged, and
# ui-spec.md's "10,000" scale target is accordingly understated as of this
# rider, worth a note back to Ryker rather than a silent rewrite of that
# doc's own number.
MAX_GEOMETRY_HELIOSTATS = 15_000

# Radial-staggered geometry for the ``{"type": "radial_stagger", ...}``
# layout: 12 rings in 3 bands (3 rings x 32, 4 rings x 48, 5 rings x 71 --
# 643 total), each band's own azimuthal pitch (360 / heliostats-per-ring),
# and explicit per-ring radii rather than a spacing scalar because the
# radial step is not constant, not even within a band. Reproduces the
# packaged field_645.csv field to 632/643 positions within 0.1 mm (max error
# ~42 mm, RMS ~3.9 mm) -- the remaining rows carry a hand-rounded coordinate
# in the source spreadsheet that no formula reproduces exactly.
RADIAL_STAGGER_BAND_COUNTS = [32, 48, 71]
RADIAL_STAGGER_BAND_RING_COUNTS = [3, 4, 5]
RADIAL_STAGGER_RING_RADII_M = [
    30.000000,
    35.014619,
    40.264159,
    46.095110,
    51.127524,
    56.419441,
    61.998057,
    67.829008,
    72.884974,
    78.188180,
    83.756681,
    89.609429,
]

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

#: Env var overriding the field-trace worker count below. A request's own
#: ``workers`` field (FieldTraceRequest) takes precedence over this when set.
TRACE_WORKERS_ENV = "HELIOSTAT_TRACE_WORKERS"


def _default_trace_workers() -> int:
    """``max(1, cpu_count - 1)``: one core left for the UI/server process
    itself. Measured per-worker cost on the 643-heliostat manuscript field
    is well under Zemax's own ~2 GB/thread rule of thumb (see the build
    report), so this default does not try to fit inside a memory budget the
    way a thread count sized for Zemax would.
    """
    override = os.environ.get(TRACE_WORKERS_ENV)
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - 1)

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

# Defaults for prime focus's cylindrical/frustum receiver shapes -- plain
# defaults, not derived from anything. The frustum defaults flare OUT toward
# the top (top radius > bottom radius): the light cone converging on a
# prime-focus receiver from the field below is widest where it first meets
# the receiver, so the absorbing surface should slope to face it there,
# narrowing toward the bottom rather than the top.
PRIME_FOCUS_CYLINDER_RADIUS_MM = 3000.0
PRIME_FOCUS_CYLINDER_HEIGHT_MM = 6000.0
PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM = 4000.0
PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM = 2500.0
PRIME_FOCUS_FRUSTUM_HEIGHT_MM = 6000.0

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


def _reject_non_finite(value):
    """Raise if ``value`` (or anything nested in it) is a non-finite float.

    ``gt``/``ge`` alone do not do this: ``inf > 0`` and ``inf >= 0`` are both
    true, so an unbounded field waves ``Infinity`` through, and a field with
    no bound at all (``receiver_center_x_mm``) waves both ``Infinity`` and
    ``NaN`` through. Recurses into lists/tuples so a vertex array or an
    ``xy_mm`` layout is covered the same as a scalar field.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("must be finite (Infinity/-Infinity/NaN are not accepted)")
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_non_finite(item)
    return value


def _json_safe(value):
    """Recursively swap a non-finite float for a JSON-safe placeholder.

    Only used to render a validation error's own ``input`` back to the
    caller: Starlette's ``JSONResponse`` encodes with ``allow_nan=False``,
    so a 422 body that quotes the rejected ``Infinity``/``NaN`` verbatim
    fails to serialize at all -- the caller would see a confusing generic
    JSON-encoding error instead of which field was the problem.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if value > 0:
            return "Infinity"
        if value < 0:
            return "-Infinity"
        return "NaN"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class _StrictModel(BaseModel):
    """Base for every request/document model in this module.

    JSON's own grammar has no ``Infinity``/``NaN`` literals, but Python's
    ``json`` module -- and the parser behind FastAPI's request body -- accept
    them as an extension. Unchecked, one of those reaches the physics as a
    NaN flux map instead of a 422. A single wildcard validator here covers
    every field of every model that inherits from it (directly or through
    another model in this file), so a field gains the same protection
    whether or not it also carries a ``gt``/``ge`` bound of its own.
    """

    @field_validator("*", mode="after")
    @classmethod
    def _finite_only(cls, value):
        return _reject_non_finite(value)


class PrimeFocusOptics(_StrictModel):
    """Receiver at the field's common focus; no secondary mirror.

    ``focus_height_mm``/``window_half_u_mm``/``window_half_v_mm`` describe
    the entrance aperture -- a flat opening at the focus, sized like the
    receiver window every layout has always had. ``receiver_type`` picks
    what actually absorbs behind it: ``"flat"`` (default) puts that
    absorbing surface AT the aperture, so ``aperture_to_receiver_mm = 0``
    reproduces today's behaviour exactly; ``"cylinder"``/``"frustum"`` are
    only meaningful here (a beam-down axicon/Cassegrain receiver sees the
    beam from above, where only a flat window makes sense).

    ``aperture_to_receiver_mm`` (default ``0``) sets the absorbing
    surface's height back from the aperture along the tower axis --
    ``0`` means "at the aperture", today's behaviour for every receiver
    type. ``receiver_center_x_mm``/``receiver_center_y_mm`` (default the
    tower axis) move the whole receiver off-axis; the aim solve
    (:func:`~heliostat.geometry.aiming.solve_prime_focus_to_receiver`)
    always targets the resolved receiver's own facing point, so pointing
    and geometry can never disagree about where it is.
    """

    model_config = ConfigDict(extra="forbid")

    focus_height_mm: float = Field(default=PRIME_FOCUS_HEIGHT_MM, gt=0)
    window_half_u_mm: float = Field(default=WINDOW_MM, gt=0)
    window_half_v_mm: float = Field(default=WINDOW_MM, gt=0)

    receiver_type: Literal["flat", "cylinder", "frustum"] = "flat"
    receiver_center_x_mm: float = 0.0
    receiver_center_y_mm: float = 0.0
    aperture_to_receiver_mm: float = Field(default=0.0, ge=0)

    cylinder_radius_mm: float = Field(default=PRIME_FOCUS_CYLINDER_RADIUS_MM, gt=0)
    cylinder_height_mm: float = Field(default=PRIME_FOCUS_CYLINDER_HEIGHT_MM, gt=0)

    frustum_top_radius_mm: float = Field(default=PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM, gt=0)
    frustum_bottom_radius_mm: float = Field(default=PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM, gt=0)
    frustum_height_mm: float = Field(default=PRIME_FOCUS_FRUSTUM_HEIGHT_MM, gt=0)

    @model_validator(mode="after")
    def _receiver_above_heliostat_plane(self) -> "PrimeFocusOptics":
        # A coarse, on-axis sanity check -- catches an obviously impossible
        # tower at 422 time. It cannot cover every heliostat's own aim
        # point for an off-axis or curved receiver (that depends on where
        # the heliostat stands); solve_prime_focus_to_receiver checks the
        # real one per heliostat at solve time.
        z = self.focus_height_mm + self.aperture_to_receiver_mm
        if self.receiver_type == "cylinder":
            z_bot = z - self.cylinder_height_mm / 2.0
        elif self.receiver_type == "frustum":
            z_bot = z - self.frustum_height_mm / 2.0
        else:
            z_bot = z
        if z_bot <= 0.0:
            raise ValueError(
                f"receiver sits at or below the heliostat plane (z = {z_bot:.0f} mm) -- "
                "raise focus_height_mm, reduce aperture_to_receiver_mm, or shrink the "
                "receiver's own height"
            )
        return self


class AxiconOptics(_StrictModel):
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
    #: Spec §C: fraction of secondary-incident power that leaves the
    #: secondary again (the rest is absorbed heat -- the §C readout is
    #: ``(1 - secondary_reflectance) * incident``). Default 0.90 is the
    #: value already baked into this package's combined 0.81 = 0.9 (mirror)
    #: x 0.9 (secondary) day-sweep throughput (heliostat.sweep.standard_optics)
    #: -- surfaced here as its own knob rather than staying buried in that
    #: product. Purely a NEW readout: it does not scale this endpoint's own
    #: receiver flux/power, which has never applied a secondary loss by
    #: default (this module's only existing loss knob is the mirror's own
    #: ``design.reflectance``, itself defaulting to 1.0) -- so introducing
    #: this field changes no existing response.
    secondary_reflectance: float = Field(default=0.90, gt=0, le=1)

    #: Spec §E2 rigid-body misalignment of the secondary -- decenter (mm)
    #: and tip/tilt (mrad, about the vertex/apex; see
    #: :func:`~heliostat.geometry.secondary._secondary_rotation_matrix` for
    #: the axis convention). Exact geometry, so it applies at every
    #: fidelity: passed straight through to
    #: :class:`~heliostat.geometry.secondary.AxiconSecondary`'s own
    #: identically-named fields. Defaults all zero, matching that class's
    #: own defaults -- an unperturbed request traces bit-identically to
    #: before this feature existed.
    secondary_dx_mm: float = 0.0
    secondary_dy_mm: float = 0.0
    secondary_dz_mm: float = 0.0
    secondary_tip_mrad: float = 0.0
    secondary_tilt_mrad: float = 0.0

    #: Spec §E2 surface deformation + parametric warp on the secondary,
    #: MONTE CARLO ONLY (see :func:`heliostat.trace.mc.trace_heliostat`'s
    #: own ``secondary_error_map`` docstring) -- the §E measured-error-map
    #: machinery reused verbatim (:attr:`_DesignBase.error_map`'s identical
    #: shape/validator, just parked on the secondary instead of the
    #: mirror), plus two analytic low-order terms. All default to "no
    #: perturbation", tracing bit-identically to before this feature
    #: existed at every fidelity, MC included.
    secondary_error_map: dict | None = None
    secondary_defocus_um: float = 0.0
    secondary_astig_um: float = 0.0
    secondary_astig_axis_deg: float = 0.0

    @field_validator("secondary_error_map")
    @classmethod
    def _secondary_error_map_must_be_a_valid_grid(cls, v: dict | None) -> dict | None:
        return _validate_error_map_grid(v, "secondary_error_map")

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


class CassegrainOptics(_StrictModel):
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
    #: See :attr:`AxiconOptics.secondary_reflectance` -- identical field,
    #: identical default, same reasoning.
    secondary_reflectance: float = Field(default=0.90, gt=0, le=1)

    #: See :attr:`AxiconOptics.secondary_dx_mm` et al. -- identical fields,
    #: identical defaults, passed through to
    #: :class:`~heliostat.geometry.secondary.CassegrainSecondary`'s own
    #: identically-named fields (rotation about its vertex rather than an
    #: axicon's apex, but the same spec §E2 convention).
    secondary_dx_mm: float = 0.0
    secondary_dy_mm: float = 0.0
    secondary_dz_mm: float = 0.0
    secondary_tip_mrad: float = 0.0
    secondary_tilt_mrad: float = 0.0

    #: See :attr:`AxiconOptics.secondary_error_map` et al. -- identical
    #: fields, identical defaults, identical MC-only convention.
    secondary_error_map: dict | None = None
    secondary_defocus_um: float = 0.0
    secondary_astig_um: float = 0.0
    secondary_astig_axis_deg: float = 0.0

    @field_validator("secondary_error_map")
    @classmethod
    def _secondary_error_map_must_be_a_valid_grid(cls, v: dict | None) -> dict | None:
        return _validate_error_map_grid(v, "secondary_error_map")

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
        receiver = _prime_focus_receiver(params)
        return solve_prime_focus_to_receiver(x_mm, y_mm, solar_az_deg, solar_el_deg, receiver)
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


def _validate_error_map_grid(v: dict | None, field_name: str) -> dict | None:
    """Shared body behind ``_DesignBase._error_map_must_be_a_valid_grid``
    (§E, the mirror's own map) and ``AxiconOptics``/``CassegrainOptics``'s
    ``_secondary_error_map_must_be_a_valid_grid`` (§E2, the same grid shape
    parked on the secondary instead) -- one validation, so a malformed grid
    is rejected identically regardless of which field it landed on.
    """
    if v is None:
        return None
    if not isinstance(v, dict) or not {"x_m", "y_m", "dz_mm"} <= v.keys():
        raise ValueError(
            f"{field_name} must be an object with 'x_m', 'y_m', 'dz_mm' "
            "(the shape /api/design/errormap/import returns under 'grid')"
        )
    try:
        ErrorMap.from_storage_dict(v)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{field_name}: {exc}") from exc
    return v


class _DesignBase(_StrictModel):
    """Shared design fields -- in practice, the mirror's optical figure.

    **Independent axes, and mixing them up is the easy mistake.**

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

    ``facet_focal_mm`` (grid and flower only) is a third, independent axis:
    the facet's own manufactured *curvature*, as a fixed spherical focal
    length (or explicitly none, at ``0``). Real heliostats often carry a
    fixed, weakly-focusing curvature that has nothing to do with how each
    heliostat is individually canted -- this field lets a design say so.
    Left blank, curvature follows ``surface``/``cant_focal_mm`` exactly as
    it always has; set, it overrides that figure on every facet regardless
    of ``surface`` or canting. See :func:`_build_trace_design`.

    ``surface`` values:

    * ``"twisting"`` (default) -- whatever solve-driven figure this app
      judges best for the design type. For a rectangle that is the aiming
      solve's own astigmatic figure, the twisting mirror of the companion
      paper; a grid or flower carries the same astigmatic figure, applied
      per facet in that facet's own local frame (an approximation -- see
      :func:`_build_trace_design`). ``facet_focal_mm``, when set on a grid
      or flower, overrides this with a fixed spherical (or flat) facet
      curvature instead.
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

    ``pointing_error_mrad`` (docs/ui-spec-v0.2.md §F) is a fourth optical
    error, orthogonal to the three above in a different way: it is the
    TRACKER's aiming inaccuracy (the whole mirror points slightly off its
    commanded direction), not a property of the mirror's own surface or
    coating. By the resolved spec convention it is quoted as the RMS
    angular deviation of the REFLECTED beam, so -- unlike
    ``slope_error_mrad``, a mirror-tilt RMS the reflection law doubles --
    no factor of two is applied to it anywhere; see
    :func:`heliostat.trace.mc.trace_heliostat`'s own docstring for the
    mirror-tilt-vs-beam-RMS bookkeeping this implies, and
    :func:`heliostat.trace.cone.sunshape_kernel`'s for the matching
    no-doubling fold at cone fidelity. Default ``0`` for the same reason
    every other error field is -- an old request/design that has never
    heard of this field traces exactly as it always has.

    ``error_map`` (docs/ui-spec-v0.2.md §E) is a fifth, independent optical
    error -- a MEASURED deformation grid (FEA or deflectometry) rather than
    a statistical description like the others above -- applied on top of
    whichever analytic figure ``surface`` chose, in MONTE CARLO ONLY. See
    the field's own comment for its shape and :func:`_build_error_map` for
    how it reaches the tracer.
    """

    surface: Literal["twisting", "spherical", "flat"] = "twisting"
    slope_error_mrad: float = Field(default=0.0, ge=0)
    specularity_mrad: float = Field(default=0.0, ge=0)
    reflectance: float = Field(default=1.0, gt=0, le=1)
    pointing_error_mrad: float = Field(default=0.0, ge=0)
    #: docs/ui-spec-v0.2.md §E: a measured/FEA sag-deviation grid, applied
    #: on top of the analytic figure above in MONTE CARLO ONLY (cone modes
    #: ignore it -- see heliostat.trace.mc.trace_heliostat's own
    #: error_map docstring). Shape is exactly what
    #: /api/design/errormap/import returns under "grid":
    #: {"x_m": [...], "y_m": [...], "dz_mm": [[...], ...]} -- the raw grid
    #: only (gradients/RMS are cheap to recompute, see
    #: heliostat.geometry.errormap.ErrorMap.from_storage_dict), so a client
    #: never builds this by hand, only round-trips what the import endpoint
    #: gave it. None (default) is "no map" -- an old request/design that
    #: has never heard of this field traces exactly as it always has.
    error_map: dict | None = None

    @field_validator("error_map")
    @classmethod
    def _error_map_must_be_a_valid_grid(cls, v: dict | None) -> dict | None:
        return _validate_error_map_grid(v, "error_map")


class ErrorMapImportRequest(_StrictModel):
    """§E's "Measured error map -- Import CSV..." request body.

    JSON-embedded rather than multipart: this app has no file-upload
    endpoint anywhere else and no ``python-multipart`` dependency to add
    one, and a CSV is plain text anyway -- the client reads the file with
    ``FileReader.readAsText`` and posts it as a string, same shape every
    other request body here already takes.
    """

    model_config = ConfigDict(extra="forbid")

    csv: str = Field(min_length=1)


class ErrorMapStatsRequest(_StrictModel):
    """Recompute an already-imported map's own chip stats (grid size,
    coverage, implied RMS) from its stored grid -- used when a design
    carrying an ``error_map`` loads from the Library or a project, where the
    client has the grid but not the import response that first reported on
    it (docs/ui-spec-v0.2.md §E's chip is not itself persisted -- only the
    grid is, see ``_DesignBase.error_map``)."""

    model_config = ConfigDict(extra="forbid")

    grid: dict


class SecondarySagRequest(_StrictModel):
    """Everything ``/api/secondary/sag``/``/api/secondary/sag.csv`` need:
    just the tower geometry -- no sun, no heliostat position, no mirror
    design -- since the secondary's own figure (nominal + §E2 parametric
    warp + imported map, summed) does not depend on any of those. Deliberately
    thinner than :class:`TraceRequest`, the same way :class:`GeometryRequest`
    is thinner than a full trace request for a different reason."""

    model_config = ConfigDict(extra="forbid")

    optics: Literal["axicon", "cassegrain"]
    optics_params: dict | None = None


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
    # Facet CURVATURE, independent of cant_focal_mm's aim -- see
    # _DesignBase. None follows surface/cant_focal_mm exactly as before this
    # field existed; 0 is explicitly flat facets; a positive value fixes
    # every facet to Spherical(facet_focal_mm) regardless of surface or
    # canting -- see _build_trace_design.
    facet_focal_mm: float | None = Field(default=None, ge=0)


class FlowerParams(_DesignBase):
    type: Literal["flower"] = "flower"
    n_petals: int = Field(gt=0)
    petal_length_mm: float = Field(gt=0)
    petal_width_mm: float = Field(gt=0)
    hub_radius_mm: float = Field(default=0.0, ge=0)
    cant_focal_mm: float | None = Field(default=None, ge=0)
    facet_focal_mm: float | None = Field(default=None, ge=0)


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


class SetupRequest(_StrictModel):
    """A named snapshot of the GUI's controls.

    ``document`` is deliberately unvalidated free-form JSON: it is the
    client's own state, and pinning a schema here would mean this module
    needs editing every time the panel gains a control. It is stored and
    handed back verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    document: dict


class SunRequest(_StrictModel):
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


class DNISetting(_StrictModel):
    """Spec §M.7's site DNI control: what sun this project assumes when it
    turns a trace/sweep/aperture reading into a real number.

    ``mode="constant"`` (the default) is a fixed W/m^2 value, applied at
    every elevation/time alike -- what every trace/day-sweep/aperture
    endpoint already assumed before this control existed, since none of
    them ever varied DNI with the sun's position (only the YEAR estimate
    did, hardcoded to :class:`~heliostat.dni.ClearSkyDNI` with no user
    control -- exactly the rider's complaint). Defaulting THIS control to
    ``constant`` at :data:`~heliostat.dni.STANDARD_DNI` (1000 W/m^2), rather
    than the rider's literally-stated "clear-sky model (default)", is a
    deliberate deviation: a clear-sky default would make a live single/field
    trace's power vary with ``solar_el_deg`` where it never has before,
    moving numbers the physics-regression test suite pins at flat 1000 for
    every existing test fixture. "Clear-sky model" is one click away in the
    Sun panel and, once chosen, applies to every surface consistently (the
    rider's actual ask) -- it just is not what a project gets for free.

    ``mode="clearsky"`` evaluates :class:`~heliostat.dni.ClearSkyDNI`,
    scaled by ``clearsky_scale`` (rider: "or a scale factor on the model" --
    1.0 is the model unscaled; e.g. 0.85 models a persistent haze/dust
    discount on an otherwise clear sky).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["constant", "clearsky"] = "constant"
    constant_w_m2: float = Field(default=STANDARD_DNI, gt=0)
    clearsky_scale: float = Field(default=1.0, gt=0)

    def dni_at_elevation(self, elevation_deg: float) -> float:
        """DNI in effect (W/m^2) at one instant, given only its solar
        elevation -- what a live single/field trace and one day/year-sweep
        timestep all have on hand, no site or calendar date required."""
        if self.mode == "constant":
            return self.constant_w_m2
        return self.clearsky_scale * ClearSkyDNI.dni_at_elevation(elevation_deg)

    def provider(self, site) -> DNIProvider:
        """A full ``(date, hour) -> W/m^2`` provider, for the surfaces that
        need one -- today, only the year estimate, whose hourly grid spans
        far more instants than it ever traces (see
        :func:`heliostat.energy.annual_energy`). ``site`` needs
        ``.latitude``/``.longitude``/``.timezone`` (a plain
        ``SimpleNamespace`` is what every caller here already builds one
        from -- see ``_year_energy_cfg``).
        """
        if self.mode == "constant":
            return ConstantDNI(self.constant_w_m2)
        base = ClearSkyDNI(site)
        return ScaledDNI(base, self.clearsky_scale) if self.clearsky_scale != 1.0 else base

    def describe(self) -> str:
        """Short, user-facing label -- spec §M.7: "stated on results (...)
        so a published number always says which sun it assumed." Distinct
        from a :class:`~heliostat.dni.DNIProvider`'s own, more technical
        ``describe()`` (kept separately, e.g. the year estimate's
        ``dni_provider`` diagnostic field) -- this one is the short rider
        wording, not the physics-parameter dump."""
        if self.mode == "constant":
            return f"{self.constant_w_m2:g} W/m² fixed"
        if self.clearsky_scale == 1.0:
            return "clear-sky model"
        return f"clear-sky model x{self.clearsky_scale:g}"


def _resolve_dni(setting: DNISetting, elevation_deg: float) -> tuple[float, float]:
    """``(scale, dni_w_m2)`` for one instant, given only its elevation.

    ``scale`` is the multiplier every already-1000-normalised power/flux
    number (the trace convention -- see ``heliostat.dni``'s own module
    docstring) gets multiplied by; ``dni_w_m2`` is the same number in W/m^2,
    for display. The one choke point every live-instant caller (single
    trace, field trace, one day-sweep timestep) applies its site DNI
    through, so "scale flux/power by dni/1000" is written in exactly one
    place rather than once per endpoint.
    """
    value = setting.dni_at_elevation(elevation_deg)
    return value / STANDARD_DNI, value


class PreviewRequest(_StrictModel):
    design: DesignParams


class _TraceRequestBase(_StrictModel):
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
    #: docs/ui-spec-v0.2.md §O: the Buie sunshape's circumsolar ratio --
    #: fraction of the sun's radiance in the hazy aureole around the disk
    #: versus the disk itself, the standard SolTrace/SolarPILOT convention.
    #: ``0`` (default) is today's shipped hard-cutoff, no-aureole Buie disk,
    #: BIT-IDENTICAL to before this field existed -- see
    #: :func:`heliostat.trace.cone.sunshape_kernel` and
    #: :class:`~heliostat.trace.samplers.BuieSampler` for the binding
    #: ``circumsolar_ratio == 0`` guarantee. Sibling of ``solar_az_deg``/
    #: ``solar_el_deg`` rather than a field on ``design`` -- it describes the
    #: SUN's angular shape, not the mirror -- but is threaded through
    #: exactly the same request/``_trace_core``/field/day/year/sweep path
    #: those two optical-error fields already take, so a single/field/day/
    #: year trace and a ``heliostat.sweep.run_sweep`` call move together at
    #: every fidelity.
    circumsolar_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
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
    #: Opt-in (spec §M.3): a small, downsampled raw flux grid alongside
    #: ``flux_png``, for the 3D scene's receiver drape (js/scene3d.js builds
    #: a THREE texture from it client-side rather than re-rendering the
    #: matplotlib PNG onto a mesh). ``False`` by default -- every caller that
    #: only wants the rendered PNG (scripts, the flux CSV endpoints, the
    #: existing test suite) pays nothing extra for it. See
    #: :func:`_flux_grid_payload` for the downsampling/size tradeoff.
    include_flux_grid: bool = False
    #: Opt-in (spec §C): incident flux on the secondary's own surface,
    #: alongside the receiver map -- the "Receiver / Secondary" map selector
    #: mockup M9 anticipates. ``False`` by default so an existing caller
    #: (scripts, the flux CSV endpoints, the test suite) traces exactly as
    #: before. Silently ignored -- no error, no payload -- for
    #: ``optics="prime_focus"`` and for any other secondary with no
    #: single-valued flux-map parameterization (see
    #: :func:`~heliostat.geometry.secondary.secondary_has_flux_map`).
    include_secondary_flux: bool = False
    #: Spec §M.7: the site DNI in effect for this trace/sweep. Absent means
    #: the default (constant, 1000 W/m^2) -- see :class:`DNISetting`'s own
    #: docstring for why that default, not the rider's literal "clear-sky",
    #: is what reproduces every prior response bit-for-bit.
    dni: DNISetting = Field(default_factory=DNISetting)

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
    #: Opt-in higher-resolution ``flux_png`` render (docs/ui-spec-v0.2.md
    #: §M.2's drill-down "click to expand" -- js/tabs/analysis.js's footprint
    #: overlay). ``None`` keeps the existing dpi=110 render exactly as
    #: before, so every other caller of this endpoint (scripts, the flux CSV
    #: endpoints, the existing test suite) is unaffected. Bounded well above
    #: the default so a client asking for the expanded view cannot demand an
    #: arbitrarily large server-side render.
    flux_png_dpi: int | None = Field(default=None, ge=110, le=320)
    #: v0.2 followups item 2: the frustum's TRUE developed (annular-sector,
    #: "fan") view, as an alternative to the default parameter-space
    #: rectangle (see _render_flux_png's own rim-distortion note and
    #: _render_flux_fan_png's docstring for why the rectangle distorts and
    #: what the fan view fixes). A view PREFERENCE, not a fidelity change --
    #: silently ignored (renders the rectangle) for any receiver that is not
    #: a frustum, so a client can leave this set across an optics change
    #: without a 422.
    flux_view: Literal["rect", "fan"] = "rect"


# ---------------------------------------------------------------------------
# field layouts


class FermatLayout(_StrictModel):
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


class PositionsLayout(_StrictModel):
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


class RadialStaggeredLayout(_StrictModel):
    """Concentric rings of heliostats, the classic DELSOL/Campo pattern.

    The field is organised into bands: each band has its own heliostat
    count per ring (``band_counts``) and its own number of rings
    (``band_ring_counts``), and every ring in the field has its own radius
    (``ring_radii_m``, one entry per ring across all bands, innermost
    first) -- the radial step is not constant, not even within a band, so
    it cannot be derived from a spacing scalar and is given explicitly.

    Within a ring the ``N`` heliostats are evenly spaced at pitch
    ``360 / N`` degrees. Consecutive rings within a band alternate a
    half-pitch stagger, and that alternation restarts at each band's first
    ring (which always carries the half-pitch offset) rather than
    continuing across the boundary, because the pitch itself changes
    there. The defaults reproduce the field this app has always shipped as
    a fixed 643-position CSV.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["radial_stagger"] = "radial_stagger"
    band_counts: list[int] = Field(default_factory=lambda: list(RADIAL_STAGGER_BAND_COUNTS))
    band_ring_counts: list[int] = Field(default_factory=lambda: list(RADIAL_STAGGER_BAND_RING_COUNTS))
    ring_radii_m: list[float] = Field(default_factory=lambda: list(RADIAL_STAGGER_RING_RADII_M))

    #: Total-heliostat cap for this layout. There is no single ``n`` field to
    #: pin a ``Field(le=...)`` on (the count is ``sum(band_counts *
    #: band_ring_counts)``), so it is checked in ``_shape_must_agree``
    #: instead; :class:`GeometryRadialStaggeredLayout` overrides this to
    #: ``MAX_GEOMETRY_HELIOSTATS`` the same way the Fermat geometry variant
    #: raises its own cap.
    _max_heliostats: ClassVar[int] = MAX_FIELD_HELIOSTATS

    @model_validator(mode="after")
    def _shape_must_agree(self) -> "RadialStaggeredLayout":
        if len(self.band_counts) != len(self.band_ring_counts):
            raise ValueError(
                "band_counts and band_ring_counts must be the same length -- "
                f"got {len(self.band_counts)} band_counts and "
                f"{len(self.band_ring_counts)} band_ring_counts"
            )
        if any(n <= 0 for n in self.band_counts):
            raise ValueError("band_counts must all be positive")
        if any(n <= 0 for n in self.band_ring_counts):
            raise ValueError("band_ring_counts must all be positive")
        if any(r <= 0 for r in self.ring_radii_m):
            raise ValueError("ring_radii_m must all be positive")
        total_rings = sum(self.band_ring_counts)
        if total_rings != len(self.ring_radii_m):
            raise ValueError(
                f"band_ring_counts sums to {total_rings} rings, but "
                f"ring_radii_m has {len(self.ring_radii_m)} -- one radius is "
                "needed per ring, across all bands"
            )
        total = sum(n * r for n, r in zip(self.band_counts, self.band_ring_counts))
        if total > self._max_heliostats:
            raise ValueError(
                f"band_counts x band_ring_counts totals {total} heliostats, "
                f"more than the {self._max_heliostats} this endpoint allows"
            )
        return self

    def positions_mm(self) -> np.ndarray:
        """Every ring's heliostats, band by band, innermost ring first.

        Azimuth follows :attr:`~heliostat.field.HeliostatField.azimuth_deg`'s
        own convention -- compass bearing, clockwise from +y -- so
        ``x = R sin(az)``, ``y = R cos(az)``. Positions come back in
        millimetres, matching every other layout.
        """
        xs = []
        ys = []
        ring = 0
        for band_count, n_rings in zip(self.band_counts, self.band_ring_counts):
            pitch_deg = 360.0 / band_count
            for local_ring in range(n_rings):
                radius_mm = self.ring_radii_m[ring] * 1000.0
                ring += 1
                # First ring of the band (and every other ring after it)
                # carries the half-pitch stagger -- see the class docstring.
                phase_deg = pitch_deg / 2.0 if local_ring % 2 == 0 else 0.0
                az_rad = np.radians(phase_deg + pitch_deg * np.arange(band_count))
                xs.append(radius_mm * np.sin(az_rad))
                ys.append(radius_mm * np.cos(az_rad))
        return np.column_stack((np.concatenate(xs), np.concatenate(ys)))


FieldLayout = Annotated[
    Union[FermatLayout, RadialStaggeredLayout, PositionsLayout], Field(discriminator="type")
]


class FieldTraceRequest(_TraceRequestBase):
    layout: FieldLayout
    #: Layout indices to leave out. The surviving heliostats keep their
    #: original layout index as their id, so dropping one does not renumber
    #: the rest -- an id in a response means the same mirror across requests.
    exclude_ids: list[int] = Field(default_factory=list)
    #: Process-pool size for the per-heliostat trace loop (see
    #: _trace_field_heliostats). ``None`` means "this endpoint's own
    #: default": 1 (serial, unchanged) for ``/api/field/trace``, and
    #: ``max(1, cpu_count - 1)`` (see HELIOSTAT_TRACE_WORKERS) for the
    #: ``/api/field/trace/start`` background job, which is the one meant for
    #: fields large enough that parallelism matters.
    workers: int | None = Field(default=None, ge=1, le=256)


# ---------------------------------------------------------------------------
# geometry-only field layouts
#
# /api/scene/geometry needs the same three layout shapes as a trace, only
# under the much larger MAX_GEOMETRY_HELIOSTATS cap -- for FermatLayout.n and
# PositionsLayout.xy_mm that cap is a literal Field(le=...), not a runtime
# check this module could apply after the fact; for RadialStaggeredLayout it
# is the ``_max_heliostats`` class variable its own validator reads.
# Subclassing and redeclaring just the capped field/attribute reuses every
# other line of the parent (a_m solving, the capacity-shortfall message, the
# NaN rejection, the ring/stagger math) rather than forking the whole layout
# under a new name.


class GeometryFermatLayout(FermatLayout):
    n: int = Field(default=100, ge=1, le=MAX_GEOMETRY_HELIOSTATS)


class GeometryRadialStaggeredLayout(RadialStaggeredLayout):
    _max_heliostats: ClassVar[int] = MAX_GEOMETRY_HELIOSTATS


class GeometryPositionsLayout(PositionsLayout):
    xy_mm: list[tuple[float, float]] = Field(min_length=1, max_length=MAX_GEOMETRY_HELIOSTATS)


GeometryFieldLayout = Annotated[
    Union[GeometryFermatLayout, GeometryRadialStaggeredLayout, GeometryPositionsLayout],
    Field(discriminator="type"),
]


class GeometryRequest(_StrictModel):
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


class DaySite(_StrictModel):
    """Where and when. The sun angles come from this, per timestep."""

    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(default=-10.0, ge=-90.0, le=90.0)
    longitude_deg: float = Field(default=-52.0, ge=-180.0, le=180.0)
    timezone_h: float = Field(default=-3.0, ge=-14.0, le=14.0)
    year: int = Field(default=2026, ge=1901, le=2099)
    month: int = Field(default=3, ge=1, le=12)
    day: int = Field(default=21, ge=1, le=31)
    #: Pre-existing bug fixed in passing (unrelated to spec §M.7): a day
    #: sweep itself never reads this (it traces every hour from sunrise to
    #: sunset, not one clock hour -- see _day_timesteps), but this same
    #: model is also ProjectSun.site, and the Sun panel's "site & time"
    #: fields always carry an ``hour`` (js/store.js's DEFAULT_DOC.sun.site,
    #: js/panels/sun.js's SUN_SITE_FIELDS) -- so every project save posted
    #: a ``sun.site.hour`` this model, with ``extra="forbid"``, rejected.
    #: Optional-with-default so an old saved project (which never carried
    #: one either way) keeps validating unchanged.
    hour: float = Field(default=12.0, ge=0.0, lt=24.0)


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
    #: Skip timesteps below this sun elevation -- they cost the same trace
    #: time as a noon one but collect almost no power. Shrinks the sampling
    #: window to the elevation crossing rather than filtering samples after
    #: the fact, so it does not bias the energy integral (see
    #: heliostat.solar.build_time_grid's docstring).
    min_elevation_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    layout: FieldLayout | None = None
    exclude_ids: list[int] = Field(default_factory=list)
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0


class YearSite(_StrictModel):
    """Where, and which calendar year -- a year estimate has no single
    clock time, so this drops :class:`DaySite`'s ``month``/``day``."""

    model_config = ConfigDict(extra="forbid")

    latitude_deg: float = Field(default=-10.0, ge=-90.0, le=90.0)
    longitude_deg: float = Field(default=-52.0, ge=-180.0, le=180.0)
    timezone_h: float = Field(default=-3.0, ge=-14.0, le=14.0)
    year: int = Field(default=2026, ge=1901, le=2099)


class YearTraceRequest(_TraceRequestBase):
    """Annual collection, estimated from a handful of traced days
    (docs/ui-spec.md 4, "Year estimate").

    Traces the dates :func:`heliostat.energy.suggest_sweep_dates` picks
    (``branch="ascending"``, December solstice to June solstice -- the
    single half-year that sweeps the whole declination range without
    tracing any sun direction twice), then integrates
    :func:`heliostat.energy.annual_energy` through the resulting
    (declination, hour-angle) efficiency surface, which already covers every
    hour of the year from as few as 7 traced days.

    ``fast_mode`` (default on) traces 7 dates rather than 12; the other 5 of
    the 12 reported sample days are reconstructed by mirroring a traced
    date's optics onto its declination twin on the far side of a solstice
    (see :func:`_year_report_days`), not by tracing them. DNI (spec §M.7)
    defaults to :class:`heliostat.dni.ClearSkyDNI` -- overriding the
    ``dni`` field's OWN default (``_TraceRequestBase``'s, "constant" at
    1000 W/m^2) because THIS endpoint's default has always been clear-sky,
    unconditionally, since before the site DNI control existed; keeping
    that default here (rather than the base class's) is what makes a year
    estimate posted with no ``dni`` field at all -- every year estimate
    ever posted before this control shipped -- keep reporting the exact
    same clear-sky upper bound it always did. An explicit ``dni`` (constant
    or a scaled clear-sky) overrides it like any other surface.
    """

    site: YearSite = Field(default_factory=YearSite)
    fast_mode: bool = True
    hour_step: float = Field(default=1.0, gt=0.05, le=6.0)
    sunrise_margin_min: float = Field(default=10.0, ge=0.0, le=120.0)
    #: See DayTraceRequest.min_elevation_deg -- same floor, same reasoning;
    #: on a year estimate this is the main lever on the ~93-timestep,
    #: ~1-hour full-field runtime (docs/ui-spec.md 4).
    min_elevation_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    layout: FieldLayout | None = None
    exclude_ids: list[int] = Field(default_factory=list)
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0
    dni: DNISetting = Field(default_factory=lambda: DNISetting(mode="clearsky"))


# ---------------------------------------------------------------------------
# library: named designs, receiver configs, projects and saved runs
#
# heliostat.web.library is the file store (name-safe, atomic writes, skip-
# unparseable listing -- the same machinery heliostat.web.setups uses, see
# that module for why); it stores and returns documents without interpreting
# them, exactly like setups does. Everything that gives those documents a
# *shape* -- what a receiver or a project actually contains -- lives here,
# next to the request models it reuses, so a receiver document and a trace
# request's optics_params can never validate two different things called the
# same name.


class LibrarySaveRequest(_StrictModel):
    """A name and a document, for any of the three library collections.

    The collection itself comes from the URL, not the body -- one request
    shape serves ``designs``, ``receivers`` and ``projects`` alike, and which
    schema the document must satisfy is decided by :func:`_validate_library_document`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    document: dict


class ReceiverDocument(_StrictModel):
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


class ProjectField(_StrictModel):
    """Where the mirrors stand, mirroring a trace request's own
    "layout, else a single position" choice (see :class:`DayTraceRequest`).

    ``layout`` deliberately takes the wider :data:`GeometryFieldLayout`
    (cap :data:`MAX_GEOMETRY_HELIOSTATS`) rather than a live trace's own
    :data:`FieldLayout` (cap :data:`MAX_FIELD_HELIOSTATS`, ten times
    smaller): a *saved* field is storage, not a trace request, and
    docs/ui-spec-v0.2.md §P's built-in reference projects (Gemasolar,
    Crescent Dunes, the Stellio-based Hami field) all carry more heliostats
    than :data:`MAX_FIELD_HELIOSTATS` allows. This does not loosen a live
    field trace's own cap -- :class:`FieldTraceRequest` still validates
    against the narrower :data:`FieldLayout`, so opening one of these
    projects and asking to trace the whole field still 422s there,
    unaffected by this widening. Backward compatible: every existing saved
    project has at most :data:`MAX_FIELD_HELIOSTATS` positions already, well
    inside the wider cap, so nothing that validated before stops validating
    now.
    """

    model_config = ConfigDict(extra="forbid")

    layout: GeometryFieldLayout | None = None
    heliostat_x_mm: float = 0.0
    heliostat_y_mm: float = -89609.0


class ProjectSun(_StrictModel):
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
    #: Spec §M.7, persisted like every other site setting. Absent (a
    #: project saved before this control existed) means the default --
    #: constant, 1000 W/m^2 -- which is exactly what that project already
    #: traced at, so it keeps reopening bit-identical.
    dni: DNISetting = Field(default_factory=DNISetting)
    #: Spec §O, persisted like every other Sun-stage setting. Absent (a
    #: project saved before this rider shipped) means the default -- ``0``,
    #: today's hard-cutoff Buie disk -- which is exactly the physics that
    #: project already traced at (§O's own binding bit-identity
    #: requirement), so it keeps reopening unchanged.
    circumsolar_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ProjectRun(_StrictModel):
    """The fidelity a project was (or should be) traced at."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["ultra_fast", "fast_accurate", "monte_carlo"] = "ultra_fast"
    n_rays: int | None = Field(default=None, ge=100, le=2_000_000)


class ProjectDocument(_StrictModel):
    """A saved project: design + field + receiver + sun + run, bundled as
    the Library's "save my work" unit (docs/ui-spec.md 5).

    ``schema_version`` is a required literal, not a default, on purpose: a
    document that does not say which version it is gets rejected rather than
    silently read under today's rules. It accepts ``1`` or ``2``: v1 predates
    saved runs and has no ``runs`` field (it defaults to empty, not
    "unmigrated"), v2 adds one. Both validate the same way, so a v1 project
    keeps opening; every fresh save writes ``2``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1, 2]
    design: DesignParams
    receiver: ReceiverDocument
    field: ProjectField
    sun: ProjectSun
    run: ProjectRun = Field(default_factory=ProjectRun)
    #: Names of ``runs`` library entries saved against this project
    #: (docs/ui-spec.md 4, "runs save with the project"), most recent last.
    runs: list[str] = Field(default_factory=list)


class SavedRunDocument(_StrictModel):
    """A finished day sweep or year estimate, persisted so it reopens
    without re-running (docs/ui-spec.md 4).

    ``request`` is the exact body the run was started with -- the client's
    own physics-key staleness check (js/tabs/analysis.js) runs against it
    unchanged, whether the run is still in memory or was just reloaded from
    here. ``result`` is the matching ``/api/day/result`` or
    ``/api/year/result`` payload. ``flux_pngs`` carries a day run's
    per-timestep irradiance maps, base64-encoded and keyed by step index as
    a string (empty for a year estimate, which renders none) -- the saved
    equivalent of a live job's in-memory ``Job.blobs``, so reopening a
    timestep costs nothing here either.
    """

    model_config = ConfigDict(extra="forbid")

    # docs/ui-spec-v0.2.md §R: "instant" persists one traced field/single
    # instant from the 3D View trace bar (ui.traceResult on the client) --
    # `request` is the exact FieldTraceRequest/TraceRequest body it was
    # traced with, `result` is that trace's own response with its `scene`
    # key dropped (3D View regenerates geometry cheaply from `request`;
    # Analysis's instruments never read it, and keeping it would roughly
    # double the saved size for no benefit here -- see js/tabs/analysis.js's
    # saveInstantRun). `flux_pngs` carries exactly one entry, key `"0"`
    # (the day/year machinery's own per-step-index convention, reused for
    # this run's one and only "step").
    kind: Literal["day", "year", "instant"]
    project_name: str | None = None
    request: dict
    result: dict
    flux_pngs: dict[str, str] = Field(default_factory=dict)
    #: §M.4's analysis aperture, frozen at save time -- js/tabs/analysis.js
    #: computes the circle and its readout live from the fetched flux grid;
    #: this field is only ever that already-computed snapshot, asked to be
    #: persisted verbatim. Reopening a saved run redraws the same circle and
    #: numbers with no recompute (mockup M17's own checknote: "reopening a
    #: saved analysis shows the same circle and readout without recomputing
    #: anything"). ``None`` when the run was saved with no aperture drawn.
    #: A loose dict, like ``request``/``result`` above, rather than a typed
    #: sub-model -- this is presentation metadata about a run, not physics,
    #: and its shape is owned by the one frontend module that reads and
    #: writes it.
    aperture: dict | None = None


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
        elif collection == "runs":
            SavedRunDocument.model_validate(document)
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
#: ``runs`` has none: a finished run is always something a user made, never
#: a manuscript default. ``projects`` gained its first four with
#: docs/ui-spec-v0.2.md §P (BUILTIN_PROJECTS: Gemasolar, PS10, Crescent
#: Dunes, the Stellio-based Hami field) -- previously always empty.
_BUILTIN_LIBRARY: dict[str, dict[str, dict]] = {
    "designs": BUILTIN_DESIGNS,
    "receivers": BUILTIN_RECEIVERS,
    "projects": BUILTIN_PROJECTS,
    "runs": {},
}


def _require_known_collection(collection: str) -> None:
    """404 for a collection name that is not one of the four -- every
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
    cant_focal_mm: float | str | None,
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
    if params.facet_focal_mm is not None:
        surface = Spherical(params.facet_focal_mm) if params.facet_focal_mm > 0 else Flat()
    else:
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
    (c3, c4, c5) needs that flip applied up front). A twisting grid or
    flower carries the same ``ZernikeAstig(c3, -c4, -c5)``, unchanged, on
    every facet, evaluated in that facet's own local frame -- the standard
    canted-and-figured approximation of a continuously twisted surface
    (every facet sees the same coefficients rather than its own re-solve).

    Canting stays on ``cant_focal_mm`` under every surface mode -- see
    :class:`_DesignBase`. A grid or flower's curvature is additionally
    overridable by ``facet_focal_mm``: when set it fixes every facet's own
    sphere (or removes curvature at 0) regardless of ``surface`` or
    ``cant_focal_mm``; left blank, curvature follows ``surface`` as above.
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

    cant = _resolved_cant_focal_mm(params.cant_focal_mm, slant_range_mm)
    # A blank cant means "per heliostat", and for a faceted design cut from a
    # solved surface that means following that surface's own local gradient
    # at each facet -- "the full calculated shape, just with aperture
    # cut-outs". An on-axis cant is rotationally symmetric and cannot
    # reproduce an astigmatic figure's slope away from centre. An explicit
    # focal (or an explicit 0) stays exactly what the caller asked for.
    auto_cant = "auto" if params.cant_focal_mm is None else cant

    if params.facet_focal_mm is not None:
        curvature: Surface = (
            Spherical(params.facet_focal_mm) if params.facet_focal_mm > 0 else Flat()
        )
        return _faceted(params, curvature, cant)

    if params.surface == "twisting":
        return _faceted(params, ZernikeAstig(sol.c3, -sol.c4, -sol.c5), auto_cant)
    if params.surface == "flat":
        return _faceted(params, Flat(), cant)
    if cant is None:
        # cant_focal_mm=0 with no facet_focal_mm either is the caller
        # explicitly asking for no focal point at all, which leaves a
        # spherical figure nothing to be figured against. Inventing a focal
        # here (slant range, say) would trace a perfectly plausible spot for
        # a mirror nobody asked for.
        raise ValueError(
            "surface='spherical' needs a focal length to figure the facets at, "
            "but cant_focal_mm=0 asks for no focus and facet_focal_mm is not "
            "set either. Leave cant_focal_mm blank to figure at this "
            "heliostat's slant range, give it a positive focal length, or set "
            "facet_focal_mm directly."
        )
    # A sphere is rotationally symmetric, so the on-axis cant already IS its
    # own local gradient -- only an astigmatic figure needs the surface to be
    # followed, and asking Spherical("slant") to supply a gradient before a
    # focal has resolved it is meaningless anyway.
    return _faceted(params, Spherical("slant"), cant)


def _build_error_map(params: RectParams | GridParams | FlowerParams | CustomParams) -> ErrorMap | None:
    """§E: the design's own measured error map, ready for
    :func:`heliostat.trace.mc.trace_heliostat`'s ``error_map`` argument.

    ``None`` when the design carries none -- the common case, and bit-
    identical to before this feature existed. ``_DesignBase``'s own
    validator (:func:`_DesignBase._error_map_must_be_a_valid_grid`) already
    proved ``params.error_map`` builds cleanly, so this cannot raise in
    practice; rebuilding here (rather than caching the validator's own
    object) keeps the request model a plain JSON-serialisable dict, not a
    numpy-carrying object pydantic would have to special-case.
    """
    if params.error_map is None:
        return None
    return ErrorMap.from_storage_dict(params.error_map)


def _build_secondary_error_map(optics_params: "OpticsParams") -> ErrorMap | None:
    """§E2 sibling of :func:`_build_error_map`: the SECONDARY's own measured
    error map, ready for :func:`heliostat.trace.mc.trace_heliostat`'s
    ``secondary_error_map`` argument.

    ``getattr`` rather than a direct attribute read: only
    :class:`AxiconOptics`/:class:`CassegrainOptics` carry
    ``secondary_error_map`` at all (:class:`PrimeFocusOptics` has no
    secondary to attach one to), so a prime-focus request -- or any caller
    that has not resolved ``optics_params`` to one of the two secondary
    types -- reads as "no map", same as the field's own default.
    """
    grid = getattr(optics_params, "secondary_error_map", None)
    if grid is None:
        return None
    return ErrorMap.from_storage_dict(grid)


def _secondary_perturb_kwargs(optics_params: "OpticsParams") -> dict:
    """§E2's four secondary-perturbation keyword arguments for
    :func:`_trace_core`/:func:`_trace_field_heliostats`
    (``secondary_error_map``/``secondary_defocus_um``/``secondary_astig_um``/
    ``secondary_astig_axis_deg``), read off ``optics_params`` once so every
    call site builds them the same way. ``getattr`` defaults mirror
    :func:`_build_secondary_error_map`'s own reasoning -- a prime-focus
    request (no secondary, no such fields) resolves to "no perturbation"
    exactly like an axicon/Cassegrain request that never touched these
    controls.
    """
    return {
        "secondary_error_map": _build_secondary_error_map(optics_params),
        "secondary_defocus_um": getattr(optics_params, "secondary_defocus_um", 0.0),
        "secondary_astig_um": getattr(optics_params, "secondary_astig_um", 0.0),
        "secondary_astig_axis_deg": getattr(optics_params, "secondary_astig_axis_deg", 0.0),
    }


def _design_is_flat(design: HeliostatDesign | None, c3: float, c4: float, c5: float) -> bool:
    """True when the trace's mirror carries no focusing figure at all.

    The legacy path (``design is None``) is flat exactly when the solve's
    own figure is all-zero -- in practice this does not happen for a real
    solve (the defocus term ``c3`` is nonzero for any finite aim distance),
    so this branch is really only reachable in principle. The design path
    is flat when every facet's surface is :class:`Flat` or an all-zero
    :class:`ZernikeAstig`: ``surface="flat"`` on any design type (including
    a rectangle, which is then deliberately routed off the legacy path so
    this check can see it), or an explicit ``facet_focal_mm=0`` on a
    grid/flower design (see :func:`_build_trace_design`). A twisting
    grid/flower's astigmatic figure is generally not all-zero even when the
    facets are left uncanted (``cant_focal_mm=0``), since it does not
    depend on canting -- so that combination no longer counts as flat.
    Canting does not count either way: a canted flat facet is still flat,
    and still needs the denser sampling.
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


def _prime_focus_receiver(params: PrimeFocusOptics) -> Receiver:
    """Prime focus's actual absorbing surface -- flat/cylinder/frustum.

    This is the receiver the aim solve targets (:func:`_solve_for`) AND the
    innermost surface the trace absorbs on (:func:`_geometry_for`, wrapped
    in :class:`ApertureClippedReceiver` when there is an offset) -- both
    read this one function so pointing and geometry can never disagree
    about where the receiver actually is.

    Equal top/bottom frustum radii collapse to :class:`CylinderReceiver`:
    a frustum's own geometry (its cone apex) is undefined there (see
    :class:`FrustumReceiver.__post_init__`), so a user typing matching
    radii -- the "frustum that is really a cylinder" degenerate case --
    gets a receiver that traces exactly, not a 422.
    """
    z = params.focus_height_mm + params.aperture_to_receiver_mm
    cx, cy = params.receiver_center_x_mm, params.receiver_center_y_mm
    if params.receiver_type == "cylinder":
        return CylinderReceiver(
            center_x_mm=cx,
            center_y_mm=cy,
            center_z_mm=z,
            radius_mm=params.cylinder_radius_mm,
            height_mm=params.cylinder_height_mm,
        )
    if params.receiver_type == "frustum":
        half_h = params.frustum_height_mm / 2.0
        if params.frustum_top_radius_mm == params.frustum_bottom_radius_mm:
            return CylinderReceiver(
                center_x_mm=cx,
                center_y_mm=cy,
                center_z_mm=z,
                radius_mm=params.frustum_top_radius_mm,
                height_mm=params.frustum_height_mm,
            )
        return FrustumReceiver(
            center_x_mm=cx,
            center_y_mm=cy,
            z_bot_mm=z - half_h,
            r_bot_mm=params.frustum_bottom_radius_mm,
            z_top_mm=z + half_h,
            r_top_mm=params.frustum_top_radius_mm,
        )
    return FlatWindowReceiver(
        z_mm=z,
        half_u_mm=params.window_half_u_mm,
        half_v_mm=params.window_half_v_mm,
        facing="down",
        center_x_mm=cx,
        center_y_mm=cy,
    )


def _prime_focus_geometry_receiver(params: PrimeFocusOptics) -> Receiver:
    """The receiver the trace absorbs on: :func:`_prime_focus_receiver`,
    behind the entrance aperture whenever ``aperture_to_receiver_mm > 0``.

    At the default offset of ``0`` the aperture and the receiver coincide,
    so wrapping would add nothing but a redundant clip -- this returns the
    bare receiver in that case, which is what keeps a request naming no
    offset tracing bit-identically to before this feature existed.
    """
    inner = _prime_focus_receiver(params)
    if params.aperture_to_receiver_mm <= 0.0:
        return inner
    aperture = FlatWindowReceiver(
        z_mm=params.focus_height_mm,
        half_u_mm=params.window_half_u_mm,
        half_v_mm=params.window_half_v_mm,
        facing="down",
        center_x_mm=params.receiver_center_x_mm,
        center_y_mm=params.receiver_center_y_mm,
    )
    return ApertureClippedReceiver(aperture=aperture, inner=inner)


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
        receiver = _prime_focus_geometry_receiver(params)
    elif optics == "axicon":
        secondary = AxiconSecondary(
            apex_height_mm=params.apex_height_mm,
            half_angle_deg=params.half_angle_deg,
            aperture_radius_mm=params.aperture_radius_mm,
            dx_mm=params.secondary_dx_mm,
            dy_mm=params.secondary_dy_mm,
            dz_mm=params.secondary_dz_mm,
            tip_mrad=params.secondary_tip_mrad,
            tilt_mrad=params.secondary_tilt_mrad,
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
            dx_mm=params.secondary_dx_mm,
            dy_mm=params.secondary_dy_mm,
            dz_mm=params.secondary_dz_mm,
            tip_mrad=params.secondary_tip_mrad,
            tilt_mrad=params.secondary_tilt_mrad,
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


def _mc_flux_and_metrics(xy: np.ndarray, watts_per_ray: float, receiver: Receiver):
    """2D-histogram flux map + spot metrics from raw Monte Carlo receiver hits."""
    # Per-bin area, not a scalar: uniform for a flat window or cylinder, but
    # a frustum's bins shrink toward its narrow end (Receiver.bin_areas_m2).
    u_edges, v_edges, bin_area_m2 = _flux_edges(receiver)

    counts, _, _ = np.histogram2d(xy[1], xy[0], bins=[v_edges, u_edges])
    flux = counts * watts_per_ray / bin_area_m2  # (n_v, n_u), W/m^2

    if xy.shape[1] == 0:
        return flux, u_edges, v_edges, float("nan"), (float("nan"), float("nan"))

    cen = xy.mean(axis=1)
    r = np.hypot(xy[0] - cen[0], xy[1] - cen[1])
    rms = float(np.sqrt(np.mean(r * r)))
    return flux, u_edges, v_edges, rms, (float(cen[0]), float(cen[1]))


def _mean_flux_kw_m2(flux: np.ndarray, bin_area_m2: np.ndarray | float) -> float:
    """Mean flux over the receiver's WHOLE modeled surface, kW/m^2.

    "Mean" here is the area-weighted average of the same per-bin flux grid
    ``peak_flux_kw_m2`` takes its max from -- ``sum(flux * bin_area) /
    sum(bin_area)`` -- over every bin the flux map covers (the full
    ``uv_extent``, dark bins included), not just the illuminated footprint.
    That is a deliberate choice: it is the definition under which
    ``mean <= max`` holds unconditionally, because a weighted average of a
    set of values can never exceed the largest one. Restricting the average
    to only the illuminated bins would still respect that bound, but "mean
    over the illuminated window" and "peak over the illuminated window" are
    both defensible; this file picks whole-surface for both so a caller
    comparing the two numbers is always comparing apples to apples.

    ``bin_area_m2`` may be a per-bin ``(n_v, n_u)`` array (a frustum's bins
    shrink toward its narrow end -- see ``Receiver.bin_areas_m2``) or a
    scalar (uniform bins, e.g. a flat window). Either broadcasts correctly
    against ``flux``.

    This replaced a frontend computation (``deriveMetrics`` in
    ``run.js``) that divided ``power_w`` by a box built from
    ``window_half_u_mm``/``window_half_v_mm`` -- the entrance APERTURE's own
    half-extents (see ``PrimeFocusOptics``), sized independently of the
    actual absorbing surface behind it. For a curved receiver (cylinder,
    frustum) that box has nothing to do with the receiver's true area, so
    the resulting "mean" could land on either side of the correctly
    per-bin-normalised peak -- observed in the field as peak 1007.1 kW/m^2
    but mean 1393.1 kW/m^2, which is impossible for two numbers drawn from
    one consistently-normalised flux field.
    """
    total_area = float(np.sum(bin_area_m2 * np.ones_like(flux)))
    if total_area <= 0.0:
        return 0.0
    return float(np.sum(flux * bin_area_m2)) / total_area / 1000.0


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
    receiver: Receiver,
    mode: TraceMode,
    *,
    mc_seed=1,
    mc_return_paths: bool = True,
    slope_error_mrad: float = 0.0,
    specularity_mrad: float = 0.0,
    reflectance: float = 1.0,
    error_map: ErrorMap | None = None,
    pointing_error_mrad: float = 0.0,
    pointing_rng: np.random.Generator | None = None,
    return_secondary_flux: bool = False,
    secondary_error_map: ErrorMap | None = None,
    secondary_defocus_um: float = 0.0,
    secondary_astig_um: float = 0.0,
    secondary_astig_axis_deg: float = 0.0,
    circumsolar_ratio: float = 0.0,
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

    ``error_map`` (§E, :func:`_build_error_map`) is forwarded to the MC
    backend only -- the cone backend's ``sunshape_kernel`` call below never
    receives it, by construction, so a map attached to a design is
    bit-identical-inert at every cone fidelity, exactly as the spec
    requires.

    ``pointing_error_mrad`` (§F) is the design's fourth optical-error
    field, forwarded into whichever backend ran exactly like
    ``slope_error_mrad``/``specularity_mrad`` are -- into
    :func:`~heliostat.trace.mc.trace_heliostat` for MC, into
    :func:`~heliostat.trace.cone.sunshape_kernel` for cone (see that
    function's docstring for why it carries no factor of two there, unlike
    ``slope_error_mrad``). ``pointing_rng``, MC only, lets a caller that
    loops over many timesteps at a fixed per-heliostat seed (the day/year
    sweep's :func:`_trace_instant_metrics`) redraw the pointing offset each
    timestep without touching that seed -- see
    :func:`~heliostat.trace.mc.trace_heliostat`'s own docstring; ``None``
    (default) draws the offset from ``mc_seed`` like every other MC error
    term, correct for a single-instant caller (single-heliostat trace,
    field trace).

    ``return_secondary_flux`` (spec §C, default ``False`` so an existing
    caller is unaffected) additionally traces the secondary's own incident
    flux, when ``secondary`` has one
    (:func:`~heliostat.geometry.secondary.secondary_has_flux_map` --
    axicon/Cassegrain only). Monte Carlo returns the raw hits
    (``secondary_xy``) for the caller to histogram with
    :func:`_mc_secondary_flux` -- exact accounting, same as every other MC
    receiver-side quantity here, which is always histogrammed outside
    ``_trace_core`` rather than inside it. The cone backend histograms its
    own chief-point deposit internally (``secondary_flux`` etc., see
    :func:`~heliostat.trace.cone.trace_heliostat_cone`) at
    :func:`_secondary_flux_grid`'s resolution, and ``reflectance`` (the
    mirror's own bounce loss, which the beam already carries by the time it
    reaches the secondary) is applied to it the same way it is applied to
    the receiver flux/power below -- a mirror that reflects less sends less
    to the secondary too. Silently a no-op for a secondary with no flux map.

    ``secondary_error_map``/``secondary_defocus_um``/``secondary_astig_um``/
    ``secondary_astig_axis_deg`` (spec §E2, :func:`_secondary_perturb_kwargs`)
    are forwarded into :func:`~heliostat.trace.mc.trace_heliostat`'s own
    identically-named parameters for the MC branch ONLY -- the cone branch
    below never reads them, so a secondary map/warp changes nothing about
    ``sunshape_kernel``/the cone deposit by construction, exactly the same
    MC-only guarantee ``error_map`` already has above.

    ``circumsolar_ratio`` (docs/ui-spec-v0.2.md §O) is the Buie sunshape's
    circumsolar ratio, forwarded into whichever backend ran: an explicit
    :class:`~heliostat.trace.samplers.BuieSampler` for MC (``None`` at
    ``circumsolar_ratio <= 0``, which is ``trace_heliostat``'s own pinned
    CSR=0 default sampler, so that path is bit-identical to before this
    parameter existed), and straight into ``sunshape_kernel``'s own
    ``circumsolar_ratio`` for cone. The one caller-visible number that
    drives both, per the spec's "one model, every fidelity" requirement.
    """
    want_secondary = return_secondary_flux and secondary_has_flux_map(secondary)
    if mode.backend == "mc":
        # docs/ui-spec-v0.2.md §O: circumsolar_ratio <= 0 passes sampler=None
        # through unchanged -- trace_heliostat's own pinned CSR=0
        # BuieSampler() default, bit-identical to before this parameter
        # existed. Only circumsolar_ratio > 0 builds an explicit sampler.
        sampler = BuieSampler(circumsolar_ratio=circumsolar_ratio) if circumsolar_ratio > 0 else None
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
                sampler=sampler,
                source_disk_radius_mm="auto",
                return_paths=mc_return_paths,
                return_secondary_hits=want_secondary,
                design=design,
                slope_error_mrad=slope_error_mrad,
                specularity_mrad=specularity_mrad,
                error_map=error_map,
                pointing_error_mrad=pointing_error_mrad,
                pointing_rng=pointing_rng,
                secondary_error_map=secondary_error_map,
                secondary_defocus_um=secondary_defocus_um,
                secondary_astig_um=secondary_astig_um,
                secondary_astig_axis_deg=secondary_astig_axis_deg,
            ),
        }
        if reflectance != 1.0:
            # watts_per_ray is what every downstream reader (power_w, the
            # flux histogram, peak flux) scales from -- one multiply here
            # reaches all of them without touching incident power, which
            # this backend never reports in the first place. Also the
            # SECONDARY histogram's scale, since a caller always builds it
            # from this same watts_per_ray via _mc_secondary_flux -- no
            # separate scaling needed there.
            result["watts_per_ray"] = result["watts_per_ray"] * reflectance
        return result

    kernel = sunshape_kernel(
        "buie",
        slope_error_mrad=slope_error_mrad,
        specularity_mrad=specularity_mrad,
        pointing_error_mrad=pointing_error_mrad,
        circumsolar_ratio=circumsolar_ratio,
    )
    cone_kwargs = dict(mode.cone_kwargs)
    # Same rule every flux-map consumer bins against (_flux_edges) -- a
    # worker's cone trace and the parent's accumulator grid must agree.
    cone_kwargs["flux_grid"] = _receiver_flux_grid(receiver)
    if want_secondary:
        cone_kwargs["return_secondary_flux"] = True
        cone_kwargs["secondary_flux_grid"] = _secondary_flux_grid(secondary)
    if _design_is_flat(design, sol.c3, sol.c4, sol.c5):
        # A deliberately flat mirror (explicit cant_focal_mm=0 on a
        # grid/flower design -- see _design_is_flat) has no focusing figure
        # at all, so the cone backend's per-sample kernels never overlap: at
        # the mode's normal sampling grid (20x12 fixed for fast_accurate,
        # density-derived -- 17x10 on the manuscript mirror -- for
        # ultra_fast) that shows up as a comb/ripple artifact across the flux
        # map (owner-reported). Denser sampling closes the gaps between
        # kernels; only worth the extra cost for this deliberately-flat case,
        # so it is not either mode's own default. An explicit tuple here
        # overrides both a fixed grid and a grid=None/density pair (the
        # tracer only resolves density when grid is exactly None).
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
        if "secondary_flux" in result:
            result["secondary_flux"] = result["secondary_flux"] * reflectance
            result["secondary_power_w"] = result["secondary_power_w"] * reflectance
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


# ---------------------------------------------------------------------------
# Parallel field tracing.
#
# Threads do not help here: the cone and Monte Carlo backends are Python/
# NumPy loops that hold the GIL for their whole run, so a thread pool
# measured only ~1.7x on 8 cores (owner-measured). A process pool sidesteps
# the GIL entirely, at the cost of pickling each task's own inputs across the
# process boundary.
#
# The pool itself is a module-level singleton (see _acquire_field_pool):
# starting one costs real wall-clock time (spawning worker processes), so a
# field trace reuses the same pool across requests rather than paying that on
# every Run click. That reuse is exactly why _field_trace_task takes every
# input a trace needs as part of its own task tuple instead of relying on an
# initializer to set shared state once per worker process -- an initializer
# only runs at pool creation, so a worker holding onto stale state from a
# previous, unrelated field would silently trace the wrong geometry for
# every request after the first. Module-level (not a closure), so it pickles
# under every start method -- including Windows' ``spawn``, which re-imports
# this module in each worker rather than forking it.
# ---------------------------------------------------------------------------

_field_pool_lock = threading.Lock()
_field_pool: ProcessPoolExecutor | None = None
_field_pool_size = 0
_field_pool_inflight = 0


def _init_field_worker() -> None:
    """Pin BLAS/OpenMP thread pools to 1, inside the worker process only.

    Runs once per worker as the ``ProcessPoolExecutor`` initializer, before
    that worker's first task -- never in the main server process, since
    this function is passed as the pool's ``initializer`` rather than
    called directly. Each worker traces one heliostat at a time, which
    includes tiny 2x2 eigenproblems (``np.linalg.eigvalsh`` in the
    Jacobian/reach path); left at their defaults, NumPy's OpenBLAS backend
    spins up its own multi-threaded pool for each of those, and with
    several worker *processes* already saturating the machine's cores that
    oversubscribes them -- every worker fighting every other worker for
    the same cores at the OS scheduler level. Setting these before any
    BLAS call keeps each worker single-threaded for linear algebra, so the
    process-level parallelism is the only parallelism in play.
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _acquire_field_pool(min_size: int) -> ProcessPoolExecutor:
    """The shared field-trace pool, sized to at least ``min_size`` workers.

    Grows (by replacing the pool) the first time a caller asks for more
    workers than it currently has; never shrinks, since idle worker
    processes cost little and losing a pool sized for an earlier large
    request would only force it to be rebuilt again later. Never replaced
    while another call is using it (see ``_field_pool_inflight``) --
    ``ProcessPoolExecutor.shutdown`` would cancel that call's own
    not-yet-started futures too, since they queue on the same pool.
    """
    global _field_pool, _field_pool_size, _field_pool_inflight
    with _field_pool_lock:
        if _field_pool is None or (_field_pool_size < min_size and _field_pool_inflight == 0):
            if _field_pool is not None:
                _field_pool.shutdown(wait=False)
            _field_pool = ProcessPoolExecutor(max_workers=min_size, initializer=_init_field_worker)
            _field_pool_size = min_size
        _field_pool_inflight += 1
        return _field_pool


def _field_pool_cold_start_expected(min_size: int) -> bool:
    """Whether the next :func:`_acquire_field_pool` call will (re)create the
    pool rather than reuse a live one.

    Read-only mirror of :func:`_acquire_field_pool`'s own condition -- never
    mutates pool state, so calling this has no effect on the pool that call
    would itself create. Exists purely so a caller can narrate a first-trace
    warmup honestly: on this machine, spawning ~7 worker processes and
    having each reimport this module (numpy/scipy/matplotlib, the
    manuscript-field CSV, the sunshape kernel table) costs several seconds
    of genuine wall time with nothing else to show for it -- see
    field_trace_start's own use of this.
    """
    with _field_pool_lock:
        return _field_pool is None or (_field_pool_size < min_size and _field_pool_inflight == 0)


def _release_field_pool() -> None:
    global _field_pool_inflight
    with _field_pool_lock:
        _field_pool_inflight = max(0, _field_pool_inflight - 1)


def _field_trace_task(task: tuple) -> tuple[int, dict]:
    """One heliostat's trace, run in a worker process. ``task`` bundles
    every input :func:`_trace_core` needs, not just the parts that vary per
    heliostat: the pool is reused across requests (see
    :func:`_acquire_field_pool`), so a worker cannot rely on state an
    initializer set for a previous, possibly different, field.

    ``error_map`` (an :class:`~heliostat.geometry.errormap.ErrorMap`, or
    ``None``) rides in the task tuple like everything else here -- it is a
    plain dataclass of numpy arrays, which :mod:`pickle` (how
    ``ProcessPoolExecutor`` ships a task to its worker) handles natively,
    so no special-casing is needed to cross the process boundary.
    """
    (
        index,
        heliostat_id,
        design,
        x_mm,
        y_mm,
        sol,
        secondary,
        receiver,
        mode,
        solar_az_deg,
        solar_el_deg,
        slope_error_mrad,
        specularity_mrad,
        reflectance,
        pointing_error_mrad,
        error_map,
        return_secondary_flux,
        secondary_error_map,
        secondary_defocus_um,
        secondary_astig_um,
        secondary_astig_axis_deg,
        circumsolar_ratio,
    ) = task
    result = _trace_core(
        design,
        x_mm,
        y_mm,
        sol,
        solar_az_deg,
        solar_el_deg,
        secondary,
        receiver,
        mode,
        mc_seed=np.random.SeedSequence((FIELD_MC_SEED, int(heliostat_id))),
        mc_return_paths=False,
        slope_error_mrad=slope_error_mrad,
        specularity_mrad=specularity_mrad,
        reflectance=reflectance,
        error_map=error_map,
        pointing_error_mrad=pointing_error_mrad,
        return_secondary_flux=return_secondary_flux,
        secondary_error_map=secondary_error_map,
        secondary_defocus_um=secondary_defocus_um,
        secondary_astig_um=secondary_astig_um,
        secondary_astig_axis_deg=secondary_astig_axis_deg,
        circumsolar_ratio=circumsolar_ratio,
    )
    return index, result


def _heliostat_progress_weights(xy_mm: np.ndarray) -> np.ndarray:
    """Expected relative trace cost per heliostat, for progress weighting only.

    The cone tracer's per-heliostat cost grows with slant range: a sample's
    footprint reach scales with the local Jacobian's top singular value
    (see ``kernels.deposit``), which grows with distance from the receiver,
    so a field's outer rings can cost several times what its inner rings
    do. Plain "N of {total} heliostats" progress races through the cheap
    inner rings and then stalls on the expensive outer ones. This weights
    each heliostat by its planar radius squared (mm^2) from the field
    origin -- a cheap, order-of-magnitude proxy for that cost, not a timing
    model -- so a job's progress fraction and ETA can track wall-time share
    instead of raw heliostat count. The ``+ 1.0`` floor keeps every weight
    strictly positive (and the total nonzero even for one heliostat sitting
    exactly at the origin).
    """
    return xy_mm[:, 0] ** 2 + xy_mm[:, 1] ** 2 + 1.0


def _trace_field_heliostats(
    designs: list[HeliostatDesign | None],
    xy_mm: np.ndarray,
    ids: list[int],
    solutions: list[Solution],
    eta_shade: np.ndarray,
    eta_block: np.ndarray,
    eta_union: np.ndarray,
    secondary,
    receiver: Receiver,
    mode: TraceMode,
    solar_az_deg: float,
    solar_el_deg: float,
    slope_error_mrad: float,
    specularity_mrad: float,
    reflectance: float,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    bin_area_m2: np.ndarray,
    *,
    error_map: ErrorMap | None = None,
    pointing_error_mrad: float = 0.0,
    workers: int = 1,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
    return_secondary_flux: bool = False,
    secondary_error_map: ErrorMap | None = None,
    secondary_defocus_um: float = 0.0,
    secondary_astig_um: float = 0.0,
    secondary_astig_axis_deg: float = 0.0,
    dni_w_m2: float = STANDARD_DNI,
    circumsolar_ratio: float = 0.0,
) -> dict:
    """One trace per heliostat, summed onto the receiver grid -- the whole
    field endpoint's "phase 3", shared by the synchronous endpoint and the
    background job so the two cannot compute it differently.

    Serial for ``workers <= 1`` or a field of one heliostat; otherwise a
    :class:`~concurrent.futures.ProcessPoolExecutor` sized to
    ``min(workers, n)``. Per-heliostat seeding (``FIELD_MC_SEED``, the
    heliostat's own id) does not depend on which worker traced it or when --
    but summing floats does depend on the order they're added in, and
    workers finish in whatever order the OS schedules them. So the parallel
    branch collects every heliostat's own result first and only sums them
    afterwards, in the same fixed index order the serial branch uses; that
    is what makes the summed result identical whatever ``workers`` is,
    bit for bit, not merely close.

    ``should_cancel``, checked at least every 0.25 s regardless of how long
    an in-flight heliostat trace takes, raises :class:`_TraceCancelled`
    rather than returning a partial sum: a field's flux is a physical total
    across every mirror, and half of one summed with the other half missing
    is not a smaller-but-valid answer the way a day sweep's finished
    timesteps are.

    ``return_secondary_flux`` (spec §C) additionally sums the secondary's
    own incident-flux map across the field -- mirrors the receiver
    ``histogram2d``/sum pattern above exactly, at
    :func:`_secondary_flux_edges`'s own resolution (fixed by ``secondary``
    alone, so every heliostat's per-trace map already shares one grid; no
    per-heliostat edges to reconcile). Silently contributes nothing when
    ``secondary`` has no flux map (:func:`secondary_has_flux_map` -- prime
    focus, pyramid).
    """
    n = xy_mm.shape[0]
    # Spec §M.7: the one choke point this field trace scales through -- every
    # per-heliostat result below is still at the trace's native DNI=1000
    # normalisation when consume() reads it; dni_scale turns that into the
    # real number, applied once per heliostat rather than once on the
    # already-summed total, so a per-heliostat `rows[i]["power_w"]` is
    # honest too, not just the field total.
    dni_scale = dni_w_m2 / STANDARD_DNI
    # Cost-weighted companion to the plain per-heliostat count, passed to
    # ``on_progress`` alongside it so a caller's ETA/progress-bar fraction
    # can track wall-time share instead of racing through cheap inner rings
    # and stalling on expensive outer ones (see _heliostat_progress_weights).
    progress_weight = _heliostat_progress_weights(xy_mm)
    flux = np.zeros((len(v_edges) - 1, len(u_edges) - 1))
    power_w = 0.0
    incident_power_w = 0.0 if mode.backend == "cone" else None
    counters: dict[str, float] = {}
    rows: list[dict | None] = [None] * n

    want_field_secondary = return_secondary_flux and secondary_has_flux_map(secondary)
    secondary_flux = secondary_power_w = secondary_u_edges = secondary_v_edges = None
    secondary_fidelity = None
    if want_field_secondary:
        secondary_u_edges, secondary_v_edges, _sec_bin_area_m2 = _secondary_flux_edges(secondary)
        secondary_flux = np.zeros((len(secondary_v_edges) - 1, len(secondary_u_edges) - 1))
        secondary_power_w = 0.0
    #: One entry per heliostat whose own trace raised -- occlusion already
    #: succeeded for it (that runs once, jointly, for the whole field before
    #: this loop), so its eta numbers are real even though it contributed no
    #: power. A single bad heliostat -- one numerically awkward geometry
    #: among hundreds -- ends up here instead of aborting a run the other
    #: 599 heliostats already finished.
    failed: list[dict] = []

    def record_failure(i: int, exc: BaseException) -> None:
        failed.append({"index": i, "id": int(ids[i]), "error": f"{type(exc).__name__}: {exc}"})
        rows[i] = {
            "id": int(ids[i]),
            "x_mm": float(xy_mm[i, 0]),
            "y_mm": float(xy_mm[i, 1]),
            "eta_shade": float(eta_shade[i]),
            "eta_block": float(eta_block[i]),
            "eta": float(eta_union[i]),
            "power_w": 0.0,
            "failed": True,
            "error": failed[-1]["error"],
        }

    def consume(i: int, result: dict) -> None:
        nonlocal power_w, incident_power_w, flux, secondary_flux, secondary_power_w, secondary_fidelity
        eta = float(eta_union[i])
        # Incident power is measured before the bounce (see _trace_core's
        # reflectance note), so it takes shading only: shading removes sun
        # before it ever reaches the mirror, but blocking removes the
        # REFLECTED ray on its way out, after the mirror already saw full
        # power. Charging incident power for blocking too would flatter
        # intercept efficiency (incident/collected) on a field with heavy
        # blocking -- collected power falls while the number it is divided
        # by falls right along with it.
        eta_incident = float(eta_shade[i])
        if result["backend"] == "mc":
            counts, _, _ = np.histogram2d(
                result["xy"][1], result["xy"][0], bins=[v_edges, u_edges]
            )
            watts_per_ray = result["watts_per_ray"] * dni_scale
            own_power = watts_per_ray * result["counters"].get("in_window", 0)
            flux += counts * watts_per_ray / bin_area_m2 * eta
        else:
            own_power = result["power_w"] * dni_scale
            incident_power_w += result["incident_power_w"] * dni_scale * eta_incident
            flux += result["flux"] * dni_scale * eta
        power_w += own_power * eta
        if want_field_secondary:
            sec_maps = _secondary_maps_from_result(result, secondary)
            if sec_maps is not None:
                s_flux, _s_u, _s_v, s_power_w, s_fidelity = sec_maps
                secondary_flux = secondary_flux + s_flux * dni_scale * eta
                secondary_power_w += s_power_w * dni_scale * eta
                secondary_fidelity = s_fidelity
        for k, v in result["counters"].items():
            counters[k] = counters.get(k, 0) + v
        rows[i] = {
            "id": int(ids[i]),
            "x_mm": float(xy_mm[i, 0]),
            "y_mm": float(xy_mm[i, 1]),
            "eta_shade": float(eta_shade[i]),
            "eta_block": float(eta_block[i]),
            "eta": eta,
            "power_w": _clean(own_power * eta),
        }

    def secondary_field_result() -> dict:
        if not want_field_secondary:
            return {}
        return {
            "secondary_flux": secondary_flux,
            "secondary_u_edges": secondary_u_edges,
            "secondary_v_edges": secondary_v_edges,
            "secondary_power_w": secondary_power_w,
            # None only if the field traced zero heliostats; every
            # contributing heliostat reports the same fidelity (fixed by
            # `mode`/backend, not by heliostat), so the last one written
            # (consume() above) speaks for the whole sum.
            "secondary_fidelity": secondary_fidelity,
        }

    if workers <= 1 or n <= 1:
        for i in range(n):
            if should_cancel is not None and should_cancel():
                raise _TraceCancelled
            try:
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
                    slope_error_mrad=slope_error_mrad,
                    specularity_mrad=specularity_mrad,
                    reflectance=reflectance,
                    error_map=error_map,
                    pointing_error_mrad=pointing_error_mrad,
                    return_secondary_flux=return_secondary_flux,
                    secondary_error_map=secondary_error_map,
                    secondary_defocus_um=secondary_defocus_um,
                    secondary_astig_um=secondary_astig_um,
                    secondary_astig_axis_deg=secondary_astig_axis_deg,
                    circumsolar_ratio=circumsolar_ratio,
                )
            except Exception as exc:  # noqa: BLE001 - isolated per heliostat, see record_failure
                record_failure(i, exc)
            else:
                consume(i, result)
            if on_progress is not None:
                on_progress(i + 1, float(progress_weight[: i + 1].sum()))
        return {
            "flux": flux,
            "power_w": power_w,
            "incident_power_w": incident_power_w,
            "counters": counters,
            "rows": rows,
            "failed": failed,
            **secondary_field_result(),
        }

    pool = _acquire_field_pool(min(workers, n))
    # Collected here, keyed by original index, but not summed until every
    # heliostat is in: workers finish in whatever order the OS schedules
    # them, and floating-point addition is not associative, so summing as
    # results arrive would make the result depend on worker count -- not
    # just slower with one worker, a DIFFERENT number. Summing in a fixed
    # pass afterwards (below) makes it the identical arithmetic the serial
    # branch above does, whatever `workers` is.
    raw_results: list[dict | None] = [None] * n
    try:
        future_index: dict = {}
        pending = set()
        for i in range(n):
            future = pool.submit(
                _field_trace_task,
                (
                    i,
                    int(ids[i]),
                    designs[i],
                    float(xy_mm[i, 0]),
                    float(xy_mm[i, 1]),
                    solutions[i],
                    secondary,
                    receiver,
                    mode,
                    solar_az_deg,
                    solar_el_deg,
                    slope_error_mrad,
                    specularity_mrad,
                    reflectance,
                    pointing_error_mrad,
                    error_map,
                    return_secondary_flux,
                    secondary_error_map,
                    secondary_defocus_um,
                    secondary_astig_um,
                    secondary_astig_axis_deg,
                    circumsolar_ratio,
                ),
            )
            future_index[future] = i
            pending.add(future)
        completed = 0
        weight_done = 0.0
        while pending:
            if should_cancel is not None and should_cancel():
                raise _TraceCancelled
            # A short timeout, not a plain wait for the next result: with a
            # handful of workers each mid-trace, "next completion" can be
            # most of a second away, and cancel has to be checked well
            # before that to land "within a couple of seconds" on a big
            # field.
            finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                idx = future_index[future]
                try:
                    i, result = future.result()
                except Exception as exc:  # noqa: BLE001 - isolated per heliostat
                    record_failure(idx, exc)
                else:
                    raw_results[i] = result
                completed += 1
                # Workers finish in schedule order, not submission order, so
                # this sums whichever heliostats actually landed so far --
                # the same weights the serial branch above sums by index,
                # just accumulated in a different order (floating-point sums
                # are order-dependent, but this feeds only a progress
                # estimate, never the trace result itself).
                weight_done += progress_weight[idx]
                if on_progress is not None:
                    on_progress(completed, weight_done)
    finally:
        # The pool itself is shared (see _acquire_field_pool) and outlives
        # this call, so a cancel must not shut it down -- only give up on
        # OUR OWN not-yet-started futures. Already-running ones finish on
        # their own in the background and are simply never collected here,
        # same as before; cancel() on those returns False and is a no-op.
        for future in pending:
            future.cancel()
        _release_field_pool()

    for i in range(n):
        if raw_results[i] is not None:
            consume(i, raw_results[i])

    return {
        "flux": flux,
        "power_w": power_w,
        "incident_power_w": incident_power_w,
        "counters": counters,
        "rows": rows,
        "failed": failed,
        **secondary_field_result(),
    }


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


#: Per-timestep flux maps a day sweep keeps, capped so a fine hour_step
#: cannot turn one job into an unbounded pile of PNGs. The finest allowed
#: hour_step (0.05 h) over a long day is worth hundreds of steps; at roughly
#: 100-200 KB per PNG and up to MAX_FINISHED_JOBS jobs alive at once,
#: keeping every one would be tens of megabytes per job. 60 covers a sweep
#: at a half-hour step or coarser (<=30 samples) with no striding at all,
#: and bounds the rest.
MAX_DAY_FLUX_MAPS = 60


def _day_flux_step_indices(total: int, cap: int = MAX_DAY_FLUX_MAPS) -> set[int]:
    """Which of ``total`` timesteps get a stored flux map.

    Every step if the day fits under ``cap``; otherwise an evenly spaced
    subset of ``cap`` of them, always including the first and last, so a
    coarse sweep still shows the day's extremes rather than an arbitrary
    prefix.
    """
    if total <= cap:
        return set(range(total))
    if cap <= 1:
        return {0} if total else set()
    return {round(i * (total - 1) / (cap - 1)) for i in range(cap)}


def _day_flux_blob_key(step: int) -> str:
    """The key a day job's flux PNG for ``step`` is stored under in
    ``Job.blobs`` -- one place both the writer and the reader use, so they
    cannot drift apart."""
    return f"day-flux/{step}"


def _day_flux_fea_blob_key(step: int) -> str:
    """Same idea as :func:`_day_flux_blob_key`, for that step's §D FEA CSV
    grid instead of its PNG -- a sibling blob, not a re-trace, computed
    alongside the PNG in ``day_start``'s work loop."""
    return f"day-flux-fea/{step}"


def _day_flux_grid_blob_key(step: int) -> str:
    """Same idea as :func:`_day_flux_blob_key`, for that step's raw flux
    grid instead of its PNG -- §M.4's Analysis-tab aperture needs the
    numbers behind the picture, not just the picture, and computes them
    entirely client-side (no re-trace). Reuses :func:`_flux_grid_payload`
    verbatim -- the same downsampled, JSON-safe grid shape §M.3's 3D
    receiver drape already carries over the wire in ``/api/trace``'s own
    response -- so the Analysis tab's aperture math and the 3D drape parse
    one grid convention between them, not two. Computed alongside the
    PNG/CSV in ``day_start``'s work loop, once, never re-traced."""
    return f"day-flux-grid/{step}"


def _day_secondary_grid_blob_key(step: int) -> str:
    """Same idea as :func:`_day_flux_blob_key`, for that step's secondary-
    surface flux map (spec §C) instead of the receiver's -- the Analysis
    tab's Receiver | Secondary selector's remaining honest gap: a stored day-
    sweep step used to carry no secondary data at all (only a live re-trace
    did), leaving the selector disabled with a tooltip for every stored
    step, always. Reuses :func:`_secondary_payload` verbatim (the same
    absorbed-heat numbers and ``flux_grid`` a live ``/api/trace`` response's
    ``secondary`` block carries), computed alongside the PNG/CSV/grid in
    ``day_start``'s work loop -- present only when that step kept a map at
    all (``want_flux``, same :data:`MAX_DAY_FLUX_MAPS` cap, not a second
    budget) AND the optics has a secondary flux map
    (:func:`~heliostat.geometry.secondary.secondary_has_flux_map` --
    axicon/Cassegrain; prime focus stores nothing extra here, ever)."""
    return f"day-secondary-grid/{step}"


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
            min_elevation_deg=req.min_elevation_deg,
            dates=[_dt.date(site.year, site.month, site.day)],
        ),
    )
    return build_time_grid(cfg, [_dt.date(site.year, site.month, site.day)])


class _TraceCancelled(Exception):
    """Raised out of a partially-traced field when its job was cancelled."""


def _trace_instant_metrics(
    req: "DayTraceRequest",
    solar_az_deg: float,
    solar_el_deg: float,
    want_flux: bool = False,
    want_secondary_flux: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    step_key: int | None = None,
) -> dict:
    """Power and spot metrics at one instant, for one heliostat or a field.

    Built from the same helpers the single and field endpoints use --
    :func:`_solve_for`, :func:`_build_trace_design`, :func:`_field_occlusion`
    and :func:`_trace_core` -- so a day's numbers are the numbers those
    endpoints would report, timestep by timestep. Nothing here re-implements
    physics; it only skips the parts a time series has no use for (the flux
    PNG, the 3-D scene, the per-heliostat table) unless ``want_flux`` asks
    for the grid back too, under ``flux``/``u_edges``/``v_edges``, so a
    caller that already paid for this trace can render a PNG from it instead
    of tracing the timestep again.

    ``want_secondary_flux`` is the same idea for spec §C's secondary-surface
    map -- summed across every heliostat exactly like
    :func:`_trace_field_heliostats`'s own ``return_secondary_flux`` does
    (same :func:`_secondary_maps_from_result`/``eta``-weighted sum), and
    silently a no-op (no ``secondary_*`` keys in the result) when
    ``secondary`` has no flux map, same as everywhere else §C touches. Kept
    as its own flag rather than folded into ``want_flux`` because a caller
    (``day_start``) wants the receiver grid and the secondary grid gated by
    two different conditions -- the day sweep's own kept-step cap, and
    whether the request asked for a secondary map at all.

    ``step_key`` is this call's own timestep discriminator inside a day/year
    sweep -- the caller's own loop index (int, distinct per timestep and
    stable given the same run) is sufficient, and is fed straight to
    :class:`numpy.random.SeedSequence` below, so it must be an int (or
    ``None``). Every
    timestep here traces each heliostat at the SAME ``mc_seed``
    (``FIELD_MC_SEED``, the heliostat's own id, no timestep) on purpose -- a
    day-sweep step is required to reproduce a direct ``/api/trace``/
    ``/api/field/trace`` call made at that step's own exact sun angles (see
    ``tests/test_web.py::test_day_flux_png_matches_a_direct_field_trace``),
    which only holds if that seed never varies by timestep. §F's pointing
    error, though, must be redrawn every timestep (quasi-static per
    instant) -- squarely in tension with reusing that same seed. Resolved
    by keeping the two on separate generators: ``mc_seed`` (and therefore
    ray sampling, ``slope_error_mrad``, ``specularity_mrad``) is untouched,
    while the pointing offset draws from its OWN generator, seeded by
    ``(FIELD_MC_SEED, heliostat_id, step_key)`` and passed as
    :func:`_trace_core`'s ``pointing_rng`` -- see
    :func:`~heliostat.trace.mc.trace_heliostat`'s own docstring for why
    that parameter exists. ``step_key=None`` (a caller outside a
    timestep loop) falls back to drawing the offset from ``mc_seed`` like
    every other MC error term, same as a single-instant caller elsewhere.
    """
    optics_params = resolve_optics_params(req.optics, req.optics_params)
    secondary, receiver = _geometry_for(req.optics, optics_params)
    mode = req.trace_mode()
    u_edges, v_edges, bin_area_m2 = _flux_edges(receiver)
    # Built once per call (not per heliostat/timestep below) -- the design's
    # map does not vary across a field or a day sweep's timesteps.
    error_map = _build_error_map(req.design)
    # Spec §E2 -- same "built once, not per heliostat/timestep" reasoning
    # as error_map above: the secondary's own map/warp is fixed by the
    # tower geometry, not by which heliostat or timestep is tracing.
    secondary_perturb_kwargs = _secondary_perturb_kwargs(optics_params)

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

    flux = np.zeros((len(v_edges) - 1, len(u_edges) - 1))
    power_w = 0.0
    # Spec §C -- same "built once, summed with an eta-weighted add per
    # heliostat" shape as the receiver flux above and as
    # _trace_field_heliostats's own want_field_secondary/consume(); the
    # secondary's own grid is fixed by `secondary` alone, so every
    # heliostat's per-trace map already shares one (u, v) edge set.
    want_secondary = want_secondary_flux and secondary_has_flux_map(secondary)
    secondary_flux = secondary_power_w = secondary_u_edges = secondary_v_edges = None
    secondary_fidelity = None
    if want_secondary:
        secondary_u_edges, secondary_v_edges, _sec_bin_area_m2 = _secondary_flux_edges(secondary)
        secondary_flux = np.zeros((len(secondary_v_edges) - 1, len(secondary_u_edges) - 1))
        secondary_power_w = 0.0
    for i in range(len(ids)):
        # Checked per heliostat, not per timestep: one timestep of a large
        # field runs for minutes, and a cancel that waits for it reads as a
        # hang. Reading a threading.Event costs nothing next to a trace.
        if should_cancel is not None and should_cancel():
            raise _TraceCancelled
        pointing_rng = (
            np.random.default_rng(
                np.random.SeedSequence((FIELD_MC_SEED, int(ids[i]), step_key))
            )
            if step_key is not None
            else None
        )
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
            error_map=error_map,
            pointing_error_mrad=req.design.pointing_error_mrad,
            pointing_rng=pointing_rng,
            return_secondary_flux=want_secondary_flux,
            circumsolar_ratio=req.circumsolar_ratio,
            **secondary_perturb_kwargs,
        )
        eta = float(eta_union[i])
        if result["backend"] == "mc":
            counts, _, _ = np.histogram2d(result["xy"][1], result["xy"][0], bins=[v_edges, u_edges])
            flux += counts * result["watts_per_ray"] / bin_area_m2 * eta
            power_w += result["watts_per_ray"] * result["counters"].get("in_window", 0) * eta
        else:
            flux += result["flux"] * eta
            power_w += result["power_w"] * eta
        if want_secondary:
            sec_maps = _secondary_maps_from_result(result, secondary)
            if sec_maps is not None:
                s_flux, _s_u, _s_v, s_power_w, s_fidelity = sec_maps
                secondary_flux = secondary_flux + s_flux * eta
                secondary_power_w += s_power_w * eta
                secondary_fidelity = s_fidelity

    rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)
    out = {
        "power_w": float(power_w),
        "peak_flux_kw_m2": float(np.max(flux)) / 1000.0,
        "mean_flux_kw_m2": _mean_flux_kw_m2(flux, bin_area_m2),
        "rms_radius_mm": rms_mm,
        "centroid_mm": list(centroid),
        "eta_shade_mean": float(np.mean(eta_shade)),
        "eta_block_mean": float(np.mean(eta_block)),
        "eta_mean": float(np.mean(eta_union)),
        "n_heliostats": len(ids),
    }
    if want_flux:
        out["flux"] = flux
        out["u_edges"] = u_edges
        out["v_edges"] = v_edges
    # None (rather than absent) only if secondary_fidelity never got set --
    # every heliostat failed, or the field traced zero of them; a real day
    # sweep step traces at least one, so this is a belt-and-suspenders case,
    # not the common one.
    if want_secondary and secondary_fidelity is not None:
        out["secondary_flux"] = secondary_flux
        out["secondary_u_edges"] = secondary_u_edges
        out["secondary_v_edges"] = secondary_v_edges
        out["secondary_power_w"] = secondary_power_w
        out["secondary_fidelity"] = secondary_fidelity
    return out


def _flux_grid_for(
    body: "TraceRequest",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, "Receiver"]:
    """``(flux_w_m2, u_edges, v_edges, receiver)`` for one single-heliostat
    request.

    The same solve/design/trace path :func:`_trace_core` gives the trace
    endpoint, kept separate only so an export does not have to render a PNG
    or build a scene to get at the numbers behind them. ``receiver`` rides
    along so a caller building an export's metadata (curved-receiver
    unrolling convention, receiver kind) never has to re-resolve optics
    params and re-build the geometry itself.
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
        error_map=_build_error_map(body.design),
        pointing_error_mrad=body.design.pointing_error_mrad,
        circumsolar_ratio=body.circumsolar_ratio,
        **_secondary_perturb_kwargs(optics_params),
    )
    # Spec §M.7: the same DNI an on-screen /api/trace call for this body
    # would apply -- so an exported CSV always matches what the UI shows
    # for identical inputs, never a silently-different flat-1000 number.
    dni_scale, _dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)
    if result["backend"] == "mc":
        flux, u_edges, v_edges, _rms, _cen = _mc_flux_and_metrics(
            result["xy"], result["watts_per_ray"] * dni_scale, receiver
        )
        return flux, u_edges, v_edges, receiver
    return result["flux"] * dni_scale, result["u_edges"], result["v_edges"], receiver


def _secondary_flux_grid_for(
    body: "TraceRequest",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, "Secondary"] | None:
    """``(flux_w_m2, u_edges, v_edges, secondary_reflectance, secondary)`` on
    the secondary's own surface for one single-heliostat request, or
    ``None`` when there is nothing to export (``optics="prime_focus"``, or
    any other secondary with no single-valued flux-map parameterization --
    see :func:`~heliostat.geometry.secondary.secondary_has_flux_map`). The
    ``secondary`` object itself rides along so a caller building the §D FEA
    export can turn its ``(u, v)`` grid into true world coordinates via
    :func:`~heliostat.geometry.secondary.secondary_uv_to_world`.

    The secondary-map analogue of :func:`_flux_grid_for`, sharing its own
    solve/design/trace path via :func:`_trace_core` so the export can never
    show a different secondary map than ``/api/trace``'s own ``secondary``
    payload block for identical inputs.
    """
    optics_params = resolve_optics_params(body.optics, body.optics_params)
    secondary, receiver = _geometry_for(body.optics, optics_params)
    if not secondary_has_flux_map(secondary):
        return None
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
        error_map=_build_error_map(body.design),
        pointing_error_mrad=body.design.pointing_error_mrad,
        return_secondary_flux=True,
        circumsolar_ratio=body.circumsolar_ratio,
        **_secondary_perturb_kwargs(optics_params),
    )
    secondary_maps = _secondary_maps_from_result(result, secondary)
    if secondary_maps is None:
        return None
    flux, u_edges, v_edges, _power_w, _fidelity = secondary_maps
    secondary_reflectance = getattr(optics_params, "secondary_reflectance", 0.90)
    # Spec §M.7 -- see _flux_grid_for's identical comment.
    dni_scale, _dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)
    return flux * dni_scale, u_edges, v_edges, secondary_reflectance, secondary


def _field_trace_phase(body: "FieldTraceRequest", *, return_secondary_flux: bool = False) -> dict:
    """Solve, occlude and trace one field instant -- ``/api/field/trace``'s
    own phases 1-3 (solve, occlusion, sum-onto-the-receiver), shared with the
    field-level FEA CSV export endpoints below so an export is never a
    second, possibly-drifted computation of the same field a live trace
    already summed. Skips phase 4+ (the flux PNG, the 3-D scene) -- callers
    that need those still go through ``/api/field/trace`` itself.

    docs/ui-spec-v0.2.md §R's own gap: a live field trace has no synchronous
    CSV export today, only the day-sweep job's internal per-kept-step call
    to :func:`_flux_fea_csv`/:func:`_secondary_flux_fea_csv`. This is the
    field-level analogue of :func:`_flux_grid_for`/:func:`_secondary_flux_grid_for`
    (the single-heliostat versions those two endpoints already share), kept
    as its own function rather than folded into ``field_trace`` itself so
    that well-tested, synchronous endpoint stays untouched.
    """
    if body.solar_el_deg <= 0:
        raise HTTPException(
            status_code=422,
            detail="solar_el_deg must be > 0 (the sun is below the horizon)",
        )
    optics_params = resolve_optics_params(body.optics, body.optics_params)
    try:
        xy_mm, ids = _field_positions(body.layout, body.exclude_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    secondary, receiver = _geometry_for(body.optics, optics_params)
    mode = body.trace_mode()
    try:
        solutions, designs, _slants = _solve_field(
            body.optics, optics_params, body.design, xy_mm, body.solar_az_deg, body.solar_el_deg
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    eta_shade, eta_block, eta_union, outline = _field_occlusion(
        xy_mm, ids, solutions, designs[0], body.solar_az_deg, body.solar_el_deg
    )
    u_edges, v_edges, bin_area_m2 = _flux_edges(receiver)
    _dni_scale, dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)
    traced = _trace_field_heliostats(
        designs,
        xy_mm,
        ids,
        solutions,
        eta_shade,
        eta_block,
        eta_union,
        secondary,
        receiver,
        mode,
        body.solar_az_deg,
        body.solar_el_deg,
        body.design.slope_error_mrad,
        body.design.specularity_mrad,
        body.design.reflectance,
        u_edges,
        v_edges,
        bin_area_m2,
        error_map=_build_error_map(body.design),
        pointing_error_mrad=body.design.pointing_error_mrad,
        workers=body.workers or 1,
        return_secondary_flux=return_secondary_flux,
        dni_w_m2=dni_w_m2,
        circumsolar_ratio=body.circumsolar_ratio,
        **_secondary_perturb_kwargs(optics_params),
    )
    return {
        "optics_params": optics_params,
        "xy_mm": xy_mm,
        "ids": ids,
        "secondary": secondary,
        "receiver": receiver,
        "u_edges": u_edges,
        "v_edges": v_edges,
        "traced": traced,
    }


# ---------------------------------------------------------------------------
# §D map exports: ANSYS-oriented FEA CSV grids (docs/ui-spec-v0.2.md §D).
#
# One convention, shared by every export below: three commented (``#``)
# metadata lines -- units, "heliostat / sun / mode / timestamp", and grid
# dimensions -- followed by a plain comma-separated numeric grid, one point
# per row. Deliberately no header row naming the columns: the spec's own
# wording ("plain comma-separated numeric grid ... preceded by commented
# metadata lines") describes the comments as the only non-numeric content,
# and ANSYS External Data's own CSV table import wants bare numeric columns
# after whatever it skips as header/metadata -- a trailing "x_m,y_m,..." text
# row would be one more line an importer has to be told to ignore. The units
# comment line names the columns instead (e.g. "x_m, y_m ... z_sag_mm ..."),
# so a human opening the file still knows what each column is.
# ---------------------------------------------------------------------------


def _fea_csv_header(
    units_line: str, subject_line: str, grid_line: str, extra_lines: tuple[str, ...] = ()
) -> str:
    """The three-or-more ``#`` comment lines every §D export starts with."""
    lines = [f"# units: {units_line}", f"# {subject_line}", f"# grid: {grid_line}"]
    lines.extend(f"# {line}" for line in extra_lines)
    return "\n".join(lines) + "\n"


def _fea_subject_line(heliostat_desc: str, solar_az_deg: float, solar_el_deg: float, mode: str) -> str:
    """The "heliostat / sun / mode / timestamp" comment line §D calls for.

    ``timestamp`` is wall-clock UTC at export time (traceability -- "this
    file was generated when"), not the simulated instant: the sun line
    already carries the simulated az/el that instant traced at.
    """
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"heliostat: {heliostat_desc} · sun: az={solar_az_deg:.2f} deg, el={solar_el_deg:.2f} deg "
        f"· mode: {mode} · timestamp: {ts}"
    )


def _fea_csv_bytes(header: str, rows) -> bytes:
    """``header`` (already newline-terminated) followed by one plain
    comma-separated numeric row per entry of ``rows``."""
    out = StringIO()
    out.write(header)
    writer = csv.writer(out, lineterminator="\n")
    for row in rows:
        writer.writerow([f"{v:.6g}" for v in row])
    return out.getvalue().encode("utf-8")


#: v0.2 followups item 3 (owner, verbatim: "yeah, we should do just x/y/z. I
#: don't think we need u/v at all. I think it would just make it harder for
#: someone using Ansys to import for thermal analysis."): the shared
#: world-frame convention statement for every §D flux export below --
#: receiver and secondary alike, one wording so the two never drift apart.
#: Replaces the old per-shape "unrolled (u, v)" note this constant used to
#: carry (see git history), since neither export unrolls anything any more.
_FEA_WORLD_FRAME_NOTE = (
    "3-D convention: x east, y north, z up (meters) -- the same field-frame "
    "heliostat.geometry.receiver's module docstring defines for every ray "
    "in this project; the tower axis sits at x = y = 0"
)


def _sag_fea_csv(
    design, sol, half_x_mm: float, half_y_mm: float, include_cant: bool, subject_line: str
) -> bytes:
    """§D sag-map export: ``x_m, y_m, z_sag_mm`` -- one row per grid point
    that lands on a facet, from the exact grid :func:`_render_sag_png` draws
    (:func:`_sag_grid_mm`), so an export can never show a different surface
    than the picture beside it.

    Points outside every facet (``NaN`` in the grid -- gaps in a grid
    design, the space between petals) are dropped rather than emitted as a
    row of ``nan``, which is not a number ANSYS's importer would accept.
    """
    gx_mm, gy_mm, sag_mm = _sag_grid_mm(design, sol, half_x_mm, half_y_mm, include_cant)
    finite = np.isfinite(sag_mm)
    n_total = sag_mm.size
    n_valid = int(finite.sum())
    header = _fea_csv_header(
        units_line="x_m, y_m in meters (heliostat aperture frame); z_sag_mm in millimeters",
        subject_line=subject_line,
        grid_line=(
            f"{sag_mm.shape[1]} x {sag_mm.shape[0]} samples over "
            f"±{half_x_mm / 1000.0:.4f} x ±{half_y_mm / 1000.0:.4f} m, "
            f"{n_valid} of {n_total} points on a facet"
        ),
    )
    xs_m = gx_mm[finite] / 1000.0
    ys_m = gy_mm[finite] / 1000.0
    zs_mm = sag_mm[finite]
    return _fea_csv_bytes(header, zip(xs_m, ys_m, zs_mm))


def _flux_fea_csv(
    flux_w_m2: np.ndarray, u_edges_mm: np.ndarray, v_edges_mm: np.ndarray, receiver, subject_line: str
) -> bytes:
    """§D irradiance-map export: ``x_m, y_m, z_m, flux_w_m2`` -- one row per
    bin centre, in meters and W/m² (never the display kW/m² the PNG and
    ``/api/trace/flux.csv`` use -- §D is explicit that units are always
    stated, never implied, and W/m² is the unit that header states).

    v0.2 followups item 3 (old format: ``x_m, y_m, flux_w_m2``, with x/y the
    unrolled ``(u, v)`` grid for a curved receiver -- see git history for the
    exact prior wording): every receiver kind now exports TRUE world
    coordinates via :meth:`Receiver.uv_to_world` -- the exact inverse of
    :meth:`Receiver.intersect` (pinned by
    ``tests/test_receiver_shapes.py::test_uv_to_world_is_the_exact_inverse_of_intersect``)
    -- so an ANSYS import places flux at the bin's real 3-D position on the
    surface rather than an arc-length coordinate an importer would otherwise
    read as flat. A flat receiver's ``z`` is simply its own plane height,
    exported uniformly alongside x, y like every other shape -- no branch
    needed, :meth:`FlatWindowReceiver.uv_to_world` already returns a
    constant z column.
    """
    u_mid = 0.5 * (u_edges_mm[:-1] + u_edges_mm[1:])
    v_mid = 0.5 * (v_edges_mm[:-1] + v_edges_mm[1:])
    gu, gv = np.meshgrid(u_mid, v_mid)  # (n_v, n_u), matches flux's own shape
    world_mm = receiver.uv_to_world(np.vstack([gu.ravel(), gv.ravel()]))
    header = _fea_csv_header(
        units_line="x_m, y_m, z_m in meters; flux_w_m2 in W/m²",
        subject_line=subject_line,
        grid_line=f"{gu.shape[1]} x {gu.shape[0]} bins",
        extra_lines=(_FEA_WORLD_FRAME_NOTE,),
    )
    xs_m = world_mm[0] / 1000.0
    ys_m = world_mm[1] / 1000.0
    zs_m = world_mm[2] / 1000.0
    flux_flat = flux_w_m2.ravel()
    return _fea_csv_bytes(header, zip(xs_m, ys_m, zs_m, flux_flat))


def _secondary_flux_fea_csv(
    flux_w_m2: np.ndarray,
    u_edges_mm: np.ndarray,
    v_edges_mm: np.ndarray,
    secondary,
    secondary_reflectance: float,
    subject_line: str,
) -> bytes:
    """Spec §C / §D irradiance-map export for the SECONDARY's own surface:
    ``x_m, y_m, z_m, flux_w_m2, absorbed_w_m2`` -- one row per bin centre,
    same commented-header convention as :func:`_flux_fea_csv`
    and :func:`_sag_fea_csv`, plus the fifth ``absorbed`` column §D calls
    for on this map specifically (``(1 - secondary_reflectance) *
    flux_w_m2``, the same formula the live absorbed-heat readout uses).

    v0.2 followups item 3 (old format: ``x_m, y_m, flux_w_m2, absorbed_w_m2``,
    with x/y the secondary's unrolled ``(u, v)``): ``x, y, z`` are now true
    world coordinates via
    :func:`~heliostat.geometry.secondary.secondary_uv_to_world` -- the exact
    inverse of :func:`~heliostat.geometry.secondary.secondary_uv` (pinned by
    ``tests/test_secondary_flux.py::test_uv_to_world_is_the_exact_inverse_of_secondary_uv``),
    in the surface's own NOMINAL frame moved back to world exactly as a real
    ray hit would be (spec §E2: a decenter/tilt relocates these points along
    with the physical part).
    """
    u_mid = 0.5 * (u_edges_mm[:-1] + u_edges_mm[1:])
    v_mid = 0.5 * (v_edges_mm[:-1] + v_edges_mm[1:])
    gu, gv = np.meshgrid(u_mid, v_mid)  # (n_v, n_u), matches flux's own shape
    world_mm = secondary_uv_to_world(secondary, np.vstack([gu.ravel(), gv.ravel()]))
    header = _fea_csv_header(
        units_line=(
            "x_m, y_m, z_m in meters; flux_w_m2, absorbed_w_m2 in W/m² "
            f"(absorbed = (1 - secondary_reflectance) * flux, secondary_reflectance={secondary_reflectance:g})"
        ),
        subject_line=subject_line,
        grid_line=f"{gu.shape[1]} x {gu.shape[0]} bins",
        extra_lines=(_FEA_WORLD_FRAME_NOTE,),
    )
    xs_m = world_mm[0] / 1000.0
    ys_m = world_mm[1] / 1000.0
    zs_m = world_mm[2] / 1000.0
    flux_flat = flux_w_m2.ravel()
    absorbed_flat = flux_flat * (1.0 - secondary_reflectance)
    return _fea_csv_bytes(header, zip(xs_m, ys_m, zs_m, flux_flat, absorbed_flat))


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


# ---------------------------------------------------------------------------
# year estimate (docs/ui-spec.md 4): a handful of traced days, integrated
# over the whole year through heliostat.energy's (declination, hour-angle)
# efficiency surface.
#
# Slow mode traces every reported sample day directly. Fast mode traces only
# the days needed to span the full declination range once (the December ->
# June solstice half-year -- see YearTraceRequest's docstring) and fills in
# the rest of the reported days by re-using a traced day's optics at its
# declination twin on the other side of a solstice
# (heliostat.energy.traced_day_energy's own ``source_date`` argument), which
# is exact wherever the twin's declination genuinely matches (real calendars
# are not perfectly symmetric about a solstice, so it is a very close
# approximation rather than identical).

#: Sample days requested by docs/ui-spec.md 4: 12 by default, 7 in fast mode
#: (the December and June solstices plus 5 interior declinations -- their 5
#: mirror twins fill out the other 5 of the 12 reported days).
YEAR_SLOW_N_DECLINATIONS = 12
YEAR_FAST_N_DECLINATIONS = 7


def _year_energy_cfg(req: "YearTraceRequest") -> SimpleNamespace:
    """The ``cfg``-shaped object :mod:`heliostat.energy` and
    :mod:`heliostat.solar` take -- a plain namespace, since neither module
    prescribes a concrete config class (see ``energy.annual_energy``'s
    docstring). ``field.mirror_area_m2`` starts at a placeholder; the caller
    fills it in once it knows the design's own footprint.
    """
    return SimpleNamespace(
        site=SimpleNamespace(
            latitude=req.site.latitude_deg,
            longitude=req.site.longitude_deg,
            timezone=req.site.timezone_h,
        ),
        sweep=SimpleNamespace(
            hour_step=req.hour_step,
            sunrise_margin_min=req.sunrise_margin_min,
            min_elevation_deg=req.min_elevation_deg,
            dates=[],
        ),
        field=SimpleNamespace(mirror_area_m2=1.0),
    )


def _year_mirror_area_m2(req: "YearTraceRequest", optics_params: OpticsParams) -> float:
    """One heliostat's own aperture footprint, m^2.

    This cancels exactly out of the annual MWh total: ``eta_optical``
    (:func:`heliostat.energy.optical_efficiency`) divides traced power by
    it, ``annual_energy`` multiplies the same factor back in. It only has to
    be honest for the diagnostic ``annual_optical_efficiency`` the result
    also carries, so a representative solve (an arbitrary but plausible sun
    position, not the field's real traced ones) is enough -- the footprint
    itself does not depend on where the sun is.
    """
    if req.layout is None:
        x0, y0 = req.heliostat_x_mm, req.heliostat_y_mm
    else:
        xy, _ids = _field_positions(req.layout, req.exclude_ids)
        x0, y0 = float(xy[0, 0]), float(xy[0, 1])
    sol = _solve_for(req.optics, x0, y0, 180.0, 45.0, optics_params)
    slant = _slant_range_mm(sol, x0, y0)
    design = _build_trace_design(req.design, sol, slant)
    _region, _outline, half_w, half_h = _field_geometry(design)
    return (2.0 * half_w) * (2.0 * half_h) / 1.0e6


def _year_trace_dates(cfg, year: int, fast_mode: bool) -> list[_dt.date]:
    """The calendar dates actually ray-traced for a year estimate."""
    n = YEAR_FAST_N_DECLINATIONS if fast_mode else YEAR_SLOW_N_DECLINATIONS
    return energy.suggest_sweep_dates(cfg, n_declinations=n, year=year, branch="ascending")


def _year_report_days(cfg, trace_dates: list[_dt.date], year: int, fast_mode: bool) -> list[dict]:
    """The (up to 12) sample days the year-estimate plot shows.

    Slow mode reports exactly the traced dates. Fast mode reports each
    traced date plus -- for every one except the two solstice extrema, which
    have no twin -- the calendar date on the *other* side of the nearest
    solstice whose declination is closest to it, found by a direct scan
    (declination is not perfectly symmetric about the calendar solstice, so
    this is a search rather than a reflection formula). Each entry's
    ``source_date`` is which traced day its optics actually came from;
    ``traced`` is false only for a mirrored entry.
    """
    if not fast_mode:
        return sorted(
            ({"date": d, "source_date": d, "traced": True} for d in trace_dates),
            key=lambda r: r["date"],
        )

    n_days = 366 if _dt.date(year, 12, 31).timetuple().tm_yday == 366 else 365
    all_days = [_dt.date(year, 1, 1) + _dt.timedelta(days=i) for i in range(n_days)]
    decs = np.array([energy._declination_of(cfg, d) for d in all_days])
    lo, hi = int(np.argmin(decs)), int(np.argmax(decs))
    ascending = set(range(lo, hi + 1)) if lo <= hi else set(range(lo, n_days)) | set(range(0, hi + 1))
    complement = [i for i in range(n_days) if i not in ascending]
    index_of = {d: i for i, d in enumerate(all_days)}

    by_declination = sorted(trace_dates, key=lambda d: decs[index_of[d]])
    extrema = {by_declination[0], by_declination[-1]}

    report = [{"date": d, "source_date": d, "traced": True} for d in trace_dates]
    for d in trace_dates:
        if d in extrema or not complement:
            continue
        target = decs[index_of[d]]
        twin = min(complement, key=lambda i: abs(decs[i] - target))
        report.append({"date": all_days[twin], "source_date": d, "traced": False})
    return sorted(report, key=lambda r: r["date"])


def _render_year_png(days: list[dict]) -> bytes:
    """Day energy across the year -- traced days solid, mirrored days
    hollow, so the reconstruction fast mode relies on is visible rather than
    a black box (docs/ui-spec.md 4)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [_dt.date.fromisoformat(d["date"]) for d in days]
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    energy_kwh = np.array([d["energy_kwh"] for d in days], dtype=float)
    traced = np.array([bool(d["traced"]) for d in days])
    order = np.argsort(doy)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(doy[order], energy_kwh[order], "-", color="#3b6ea5", alpha=0.4, zorder=1)
    ax.plot(doy[traced], energy_kwh[traced], "o", color="#d97b29", label="traced", zorder=2)
    if (~traced).any():
        ax.plot(
            doy[~traced],
            energy_kwh[~traced],
            "o",
            markerfacecolor="none",
            markeredgecolor="#d97b29",
            label="by symmetry",
            zorder=2,
        )
    ax.set_xlabel("day of year")
    ax.set_ylabel("day energy (kWh)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110)
    finally:
        plt.close(fig)
    return buf.getvalue()


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


# ---------------------------------------------------------------------------
# matplotlib figure reuse for the sag map and aperture-preview PNGs.
#
# Profiling both endpoints found the physics (the sag field itself, the
# facet membership test) essentially free next to matplotlib's own cost:
# building a fresh Figure/Axes/colorbar and laying out their text is most of
# a render, every render, because the previous code built all three from
# scratch on every request. A Figure is not thread-safe to share -- two
# requests drawing into the same one at once would interleave into a
# garbled PNG -- so each stays thread-local: built once per worker thread,
# then reused, cleared and redrawn, for every request that thread serves
# afterwards. Each gets its own Agg canvas wired up directly
# (``FigureCanvasAgg(fig)``) rather than going through ``matplotlib.use()``
# and ``pyplot``, so it needs no process-wide backend state and is never
# tracked by pyplot's global figure registry (nothing to ``plt.close()``).
# ---------------------------------------------------------------------------

_SAG_FIGURE_DPI = 100
_PREVIEW_FIGURE_DPI = 100
_render_tls = threading.local()


def _new_agg_figure(figsize: tuple[float, float], dpi: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def _sag_figure():
    """This thread's persistent ``(fig, ax)`` for :func:`_render_sag_png`."""
    fig = getattr(_render_tls, "sag_fig", None)
    if fig is None:
        fig = _new_agg_figure((5.6, 4.6), _SAG_FIGURE_DPI)
        _render_tls.sag_fig = fig
        _render_tls.sag_ax = fig.add_subplot(111)
        _render_tls.sag_cbar = None
        # Which of the two axes shapes (colorbar present or not) tight_layout
        # was last run for -- see the note in _render_sag_png.
        _render_tls.sag_layout_for = None
    return fig, _render_tls.sag_ax


def _preview_figure():
    """This thread's persistent ``(fig, ax)`` for the aperture preview."""
    fig = getattr(_render_tls, "preview_fig", None)
    if fig is None:
        fig = _new_agg_figure((6.0, 6.0), _PREVIEW_FIGURE_DPI)
        _render_tls.preview_fig = fig
        _render_tls.preview_ax = fig.add_subplot(111)
    return fig, _render_tls.preview_ax


def _warm_matplotlib() -> None:
    """Pay matplotlib's one-off font/text-layout warmup cost here, on a
    background thread at startup, instead of on a visitor's first render.

    Font discovery and Agg's glyph/layout caches are process-global --
    built by whichever thread draws text first, and free to every thread
    and every render after that. This draws a throwaway figure through the
    same calls the real endpoints use (imshow, colorbar, contour, legend,
    title/tick text, savefig) so that cost lands here, before anyone is
    looking, rather than on the first real request.
    """
    try:
        fig = _new_agg_figure((5.6, 4.6), _SAG_FIGURE_DPI)
        ax = fig.add_subplot(111)
        data = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        im = ax.imshow(data, cmap="jet")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("sag (mm)")
        ax.contour(data, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.4)
        ax.set_title("peak-to-valley 0.000 mm")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        (line,) = ax.plot([0, 1], [0, 1], linestyle="--", label="warmup")
        ax.legend(handles=[line], loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(BytesIO(), format="png", dpi=_SAG_FIGURE_DPI)
    except Exception:
        # Best-effort only: a failed warmup costs the first real request its
        # usual latency back, never correctness -- so nothing here should
        # ever reach an unhandled-exception log on its own thread.
        pass


#: Sample resolution (per side) for the sag grid -- shared by the sag PNG
#: and the sag CSV export so the CSV is, point for point, the same surface
#: the picture shows (docs/ui-spec-v0.2.md §D: "never re-deriving a
#: different surface").
_SAG_GRID_N = 241


def _sag_grid_mm(
    design, sol, half_x_mm: float, half_y_mm: float, include_cant: bool = True, n: int = _SAG_GRID_N
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gx_mm, gy_mm, sag_mm)`` -- the sag field :func:`_render_sag_png` and
    the FEA sag-CSV export both draw from, so a picture and its export can
    never silently disagree.

    Sampled from the same objects the trace uses: for the legacy path the
    solve's own astigmatic coefficients, for a design each facet's surface
    evaluated in that facet's frame. Points outside every facet -- the gaps
    in a grid, the space between petals -- come back ``NaN`` rather than
    filled with the value a facet would have had if it were there.

    With ``include_cant`` (the default) each facet is placed where its cant
    actually puts it, so a faceted design reads as the one continuous shape
    it was cut from, with the gaps punched out of it, and a canted flat
    heliostat shows the tilt of every facet. Turn it off to measure each
    facet from its own mounting plane instead -- what a facet fabricator
    needs, and what makes a grid look like a repeating tile.
    """
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
            if include_cant:
                # Put the facet back where the whole mirror holds it: at the
                # surface's own height above its centre (the piston), tilted
                # by its cant. With the facet's own figure on top, a design
                # cut from one surface adds back up to that surface -- which
                # is the point of drawing it this way.
                ou, ov = facet.offset_mm
                piston = float(
                    np.asarray(facet.surface.sag_and_slopes(np.array([ou]), np.array([ov]))[0])[0]
                )
                values = values + piston
                if facet.cant_normal is not None:
                    nx, ny, nz = (float(v) for v in facet.cant_normal)
                    if abs(nz) > 1e-12:
                        values = values - (nx * du + ny * dv) / nz
            sag = np.where(inside, values, sag)

    return gx, gy, sag


def _render_sag_png(
    design, sol, params, half_x_mm: float, half_y_mm: float, include_cant: bool = True
) -> tuple[bytes, float | None, float | None]:
    """Sag map of the mirror a trace would use, in millimetres.

    "Sag" is how far the reflecting surface departs from the flat plane
    through its own vertex -- the shape that turns a mirror into a lens.
    It is millimetres over metres of aperture, invisible in the 3-D scene
    (which draws facets flat for exactly that reason), so it gets its own
    view.

    Sampled by :func:`_sag_grid_mm`; see that function for what "sag" means
    here and what ``include_cant`` changes.

    :returns: ``(png_bytes, peak_to_valley_mm, contour_interval_mm)``. Both
        numbers are ``None`` when no facet covers any sampled point ("no
        surface here"); ``contour_interval_mm`` is additionally ``None`` for
        a flat mirror (span at or below float noise), which draws no
        contours at all -- there is nothing for a spacing to describe.
    """
    from matplotlib.ticker import MaxNLocator

    gx, gy, sag = _sag_grid_mm(design, sol, half_x_mm, half_y_mm, include_cant)

    fig, ax = _sag_figure()
    ax.clear()
    finite = np.isfinite(sag)
    span: float | None = None
    interval: float | None = None
    if not finite.any():
        ax.text(0.5, 0.5, "no surface here", ha="center", va="center", transform=ax.transAxes)
        # No image this render -- hide a colorbar left over from an earlier
        # (different heliostat's) render on this thread rather than showing
        # one with stale limits next to a blank axes.
        if _render_tls.sag_cbar is not None:
            _render_tls.sag_cbar.ax.set_visible(False)
    else:
        span = float(np.nanmax(sag) - np.nanmin(sag))
        im = ax.imshow(
            sag,
            origin="lower",
            cmap="jet",
            extent=(-half_x_mm, half_x_mm, -half_y_mm, half_y_mm),
            aspect="equal",
        )
        if _render_tls.sag_cbar is None:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(4))
            _render_tls.sag_cbar = cbar
        else:
            cbar = _render_tls.sag_cbar
            cbar.ax.set_visible(True)
            cbar.update_normal(im)
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
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    # tight_layout()'s own text-bbox measurement is the single biggest cost
    # in this render (it walks every tick label to size the margins) and it
    # only needs to run again when the axes' own shape changes -- a
    # colorbar present or not changes how much width the plot gets, but a
    # new heliostat's numbers inside the same shape do not.
    layout_for = "cbar" if finite.any() else "no_cbar"
    if _render_tls.sag_layout_for != layout_for:
        fig.tight_layout()
        _render_tls.sag_layout_for = layout_for

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=_SAG_FIGURE_DPI)
    return buf.getvalue(), span, interval


# ---------------------------------------------------------------------------
# §E2 "Secondary sag map" -- the secondary-mirror analogue of the heliostat
# sag map above: nominal figure + parametric warp + imported error map,
# SUMMED, same jet colormap/contour-interval recipe, same §D CSV export
# convention -- over the secondary's own circular aperture instead of the
# heliostat's rectangle.
# ---------------------------------------------------------------------------

_SECONDARY_SAG_GRID_N = 241


def _secondary_sag_grid_mm(
    secondary, optics_params: "OpticsParams", n: int = _SECONDARY_SAG_GRID_N
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gx_mm, gy_mm, sag_mm)`` for ``secondary``'s own §E2 sag map:
    nominal figure (:func:`~heliostat.geometry.secondary.secondary_nominal_sag_mm`)
    plus parametric warp (:func:`~heliostat.geometry.secondary.secondary_warp_sag_mm`)
    plus the imported map (:meth:`~heliostat.geometry.errormap.ErrorMap.sample_dz`),
    SUMMED -- sampled on a square grid over the aperture DISK (radius
    ``secondary.aperture_radius_mm``) and ``NaN``'d outside it, since the
    secondary's aperture is circular, unlike :func:`_sag_grid_mm`'s
    rectangle.

    Reads the exact same :class:`~heliostat.geometry.errormap.ErrorMap` and
    warp numbers a Monte Carlo trace of this ``optics_params`` would use
    (:func:`_build_secondary_error_map`, the same ``getattr`` defaults
    :func:`_secondary_perturb_kwargs` uses), so this view can never show a
    different surface than what that trace actually perturbs.
    """
    r = float(secondary.aperture_radius_mm)
    xs = np.linspace(-r, r, n)
    gx, gy = np.meshgrid(xs, xs)
    inside = np.hypot(gx, gy) <= r

    sag = secondary_nominal_sag_mm(secondary, gx, gy)

    error_map = _build_secondary_error_map(optics_params)
    if error_map is not None:
        sag = sag + error_map.sample_dz(gx, gy)

    defocus_um = getattr(optics_params, "secondary_defocus_um", 0.0)
    astig_um = getattr(optics_params, "secondary_astig_um", 0.0)
    astig_axis_deg = getattr(optics_params, "secondary_astig_axis_deg", 0.0)
    if defocus_um or astig_um:
        sag = sag + secondary_warp_sag_mm(gx, gy, r, defocus_um, astig_um, astig_axis_deg)

    return gx, gy, np.where(inside, sag, np.nan)


def _secondary_sag_figure():
    """This thread's persistent ``(fig, ax)`` for
    :func:`_render_secondary_sag_png` -- its OWN TLS slot, distinct from
    :func:`_sag_figure`'s, so a secondary-sag render never clobbers an
    interleaved heliostat-sag render's figure/colorbar state on the same
    worker thread (see :func:`_sag_figure`'s own docstring for why each
    render gets a dedicated, reused-not-rebuilt figure at all)."""
    fig = getattr(_render_tls, "sec_sag_fig", None)
    if fig is None:
        fig = _new_agg_figure((5.6, 4.6), _SAG_FIGURE_DPI)
        _render_tls.sec_sag_fig = fig
        _render_tls.sec_sag_ax = fig.add_subplot(111)
        _render_tls.sec_sag_cbar = None
        _render_tls.sec_sag_layout_for = None
    return fig, _render_tls.sec_sag_ax


def _render_secondary_sag_png(
    gx_mm: np.ndarray, gy_mm: np.ndarray, sag_mm: np.ndarray, aperture_radius_mm: float
) -> tuple[bytes, float | None, float | None]:
    """Sag map PNG for the SECONDARY -- mirrors :func:`_render_sag_png`
    exactly (jet colormap, :func:`_sag_contour_interval_mm`'s contour
    spacing, the same peak-to-valley title), over the circular aperture
    :func:`_secondary_sag_grid_mm` samples instead of a rectangle. Axes are
    the secondary's own local ``x``/``y`` (:mod:`heliostat.geometry.secondary`'s
    convention), not the heliostat sag map's ``u``/``v``.

    :returns: ``(png_bytes, peak_to_valley_mm, contour_interval_mm)`` -- see
        :func:`_render_sag_png` for what each means.
    """
    from matplotlib.ticker import MaxNLocator

    r = aperture_radius_mm
    fig, ax = _secondary_sag_figure()
    ax.clear()
    finite = np.isfinite(sag_mm)
    span: float | None = None
    interval: float | None = None
    if not finite.any():
        ax.text(0.5, 0.5, "no surface here", ha="center", va="center", transform=ax.transAxes)
        if _render_tls.sec_sag_cbar is not None:
            _render_tls.sec_sag_cbar.ax.set_visible(False)
    else:
        span = float(np.nanmax(sag_mm) - np.nanmin(sag_mm))
        im = ax.imshow(
            sag_mm,
            origin="lower",
            cmap="jet",
            extent=(-r, r, -r, r),
            aspect="equal",
        )
        if _render_tls.sec_sag_cbar is None:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(4))
            _render_tls.sec_sag_cbar = cbar
        else:
            cbar = _render_tls.sec_sag_cbar
            cbar.ax.set_visible(True)
            cbar.update_normal(im)
        cbar.set_label("sag (mm)")
        if span > 1e-9:
            interval = _sag_contour_interval_mm(span)
            lo, hi = float(np.nanmin(sag_mm)), float(np.nanmax(sag_mm))
            ax.contour(
                gx_mm,
                gy_mm,
                sag_mm,
                levels=np.arange(lo, hi + interval, interval),
                colors="white",
                linewidths=0.4,
                alpha=0.6,
            )
            ax.set_title(f"peak-to-valley {span:.3f} mm · contours every {interval:g} mm")
        else:
            ax.set_title(f"peak-to-valley {span:.3f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    layout_for = "cbar" if finite.any() else "no_cbar"
    if _render_tls.sec_sag_layout_for != layout_for:
        fig.tight_layout()
        _render_tls.sec_sag_layout_for = layout_for

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=_SAG_FIGURE_DPI)
    return buf.getvalue(), span, interval


def _secondary_sag_subject_line(optics: str, aperture_radius_mm: float) -> str:
    """The §D "subject" comment line for a secondary sag CSV. Unlike
    :func:`_fea_subject_line`, there is no sun/heliostat/trace-mode to
    report -- the secondary's own figure (nominal + warp + map) does not
    depend on any of those."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"{optics} secondary, aperture radius {aperture_radius_mm / 1000.0:.3f} m "
        f"(nominal figure + parametric warp + imported map, summed) · timestamp: {ts}"
    )


def _secondary_sag_fea_csv(
    gx_mm: np.ndarray, gy_mm: np.ndarray, sag_mm: np.ndarray, aperture_radius_mm: float, subject_line: str
) -> bytes:
    """§D sag-map export for the SECONDARY: ``x_m, y_m, z_sag_mm``, one row
    per grid point inside the aperture disk -- the secondary-map analogue of
    :func:`_sag_fea_csv`, from the exact grid :func:`_render_secondary_sag_png`
    draws (:func:`_secondary_sag_grid_mm`), so this export can never show a
    different surface than that picture.
    """
    finite = np.isfinite(sag_mm)
    n_total = sag_mm.size
    n_valid = int(finite.sum())
    header = _fea_csv_header(
        units_line="x_m, y_m in meters (secondary local aperture frame); z_sag_mm in millimeters",
        subject_line=subject_line,
        grid_line=(
            f"{sag_mm.shape[1]} x {sag_mm.shape[0]} samples over a "
            f"{aperture_radius_mm / 1000.0:.4f} m radius aperture, {n_valid} of {n_total} points inside"
        ),
    )
    xs_m = gx_mm[finite] / 1000.0
    ys_m = gy_mm[finite] / 1000.0
    zs_mm = sag_mm[finite]
    return _fea_csv_bytes(header, zip(xs_m, ys_m, zs_mm))


def _unwrap_receiver_for_map(receiver) -> "Receiver | None":
    """The receiver actually parameterizing a flux map's ``(u, v)`` grid --
    unwraps an :class:`ApertureClippedReceiver` to its inner absorbing
    surface, the same rule :func:`heliostat.web.scene._receiver_dict` uses
    for exactly this reason (a cavity's ``uv``/extent/``u_period_mm`` all
    already delegate to ``inner`` -- see that class's own methods -- so its
    map is described by the inner shape, not by the flat aperture in front
    of it)."""
    if receiver is None:
        return None
    return receiver.inner if isinstance(receiver, ApertureClippedReceiver) else receiver


#: v0.2 followups item 1: the N/W/S/E/N cardinal ordering left-to-right
#: across a periodic receiver's unrolled u-axis. PROOF (not assumed): u = R
#: * az with az = atan2(x, -y) measured from -y/south (receiver.py's module
#: docstring + CylinderReceiver/FrustumReceiver's own docstrings), so at
#: u=0 (az=0) x=0, y=-R -- due south. uv_to_world confirms the same sense
#: everywhere (x = R*sin(az), y = -R*cos(az)): az=+pi/2 gives x=+R, y=0 --
#: due EAST -- so u increasing (rightward on the chart, origin="lower"/
#: left-to-right x-axis) moves toward east, and az=+pi (u at its positive
#: extreme) gives x=0, y=+R -- due north, the seam. Symmetrically az=-pi/2
#: (u negative) is due west and az=-pi (u at its negative extreme) is the
#: SAME physical seam, north again. This is independently corroborated by
#: two other already-shipped renderings of the identical convention: (1)
#: js/scene3d.js's bakeCylindricalUV, whose own comment states plainly "the
#: compass sequence this produces (N . W . S . E . N left to right)"; (2)
#: js/main.js's CYLINDER_AXIS_COMPASS = ["N","W","S","E","N"], derived (see
#: that file's own comment) from the exact same atan2(x, -y) convention.
#: All three (this module, the 3D drape, and the flux-overlay caption) must
#: therefore never disagree -- they are three renderings of one proof.
_AZIMUTH_CARDINALS = (("N", -1.0), ("W", -0.5), ("S", 0.0), ("E", 0.5), ("N", 1.0))


def _azimuth_deg_ticks(u_period_mm: float) -> tuple[list[float], list[str]]:
    """Tick positions (mm, in ``u`` units) and two-line "letter / degrees"
    labels for :data:`_AZIMUTH_CARDINALS`, degrees measured from south
    (0 deg = S, the chart's own centre; +/-180 deg = N, the seam at both
    edges) rather than the conventional from-north bearing -- chosen so the
    numbers read in the SAME left-to-right sense u already has (u increases
    -> az increases -> degrees increase), with no wrap discontinuity inside
    the plotted range and no north labelled with two different numbers."""
    half = u_period_mm / 2.0
    positions = [frac * half for _letter, frac in _AZIMUTH_CARDINALS]
    labels = [f"{letter}\n{frac * 180.0:.0f}°" for letter, frac in _AZIMUTH_CARDINALS]
    return positions, labels


def _frustum_rect_distortion_note(frustum: "FrustumReceiver") -> str:
    """v0.2 followups item 2: the one-line disclosure §the rectangular flux
    map owes its reader -- the exact stretch/compression its own u-axis
    carries at each rim, per :func:`_render_flux_fan_png`'s docstring
    (``r_mean_mm / r(v)`` at ``v``'s own rim). Always stated as "narrow
    rim stretched / wide rim compressed" rather than "bottom/top", which
    would flip sign for an inverted (r_top < r_bot) frustum."""
    r_bot, r_top, r_mean = frustum.r_bot_mm, frustum.r_top_mm, frustum.r_mean_mm
    bot_pct = (r_mean / r_bot - 1.0) * 100.0
    top_pct = (r_mean / r_top - 1.0) * 100.0
    narrow_pct, wide_pct = (bot_pct, top_pct) if r_bot < r_top else (top_pct, bot_pct)
    return (
        f"rectangle view stretches the narrow rim {narrow_pct:+.0f}% and "
        f"compresses the wide rim {wide_pct:+.0f}% vs. true arc length -- "
        "see the fan view for the undistorted development"
    )


def _render_flux_png(
    flux: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    mode: str,
    elapsed_ms: float,
    dpi: int = 110,
    receiver=None,
) -> bytes:
    # Lazy import, same reasoning as HeliostatDesign.preview(): matplotlib
    # is a real dependency but no other endpoint in this module needs it.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # `dpi` only scales the OUTPUT raster (the savefig() call below) -- the
    # figure itself is still built at figsize=(5.6, 4.6) inches with the same
    # point-sized fonts/line widths, so a higher dpi yields a genuinely
    # sharper render of the identical layout (more pixels per element, not a
    # stretched/upscaled one), same idea as a "retina" screenshot. Every
    # caller except the footprint drill-down's expanded view passes the
    # default 110, so their PNG dimensions are unchanged (see TraceRequest's
    # `flux_png_dpi`, None by default).
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

    # v0.2 followups item 1: for a receiver that closes on itself (u_period_mm
    # not None -- the exact gate heliostat.trace.cone's own wrap machinery
    # uses, so "periodic" here always means what the tracer means by it), the
    # millimetre u-axis is misleading (exact only at the cylinder's radius /
    # the frustum's mean radius) and the owner asked for compass context
    # ("N / S / E / W labelled") instead -- see _AZIMUTH_CARDINALS above for
    # the proof of the ordering used here.
    unwrapped = _unwrap_receiver_for_map(receiver)
    u_period_mm = getattr(unwrapped, "u_period_mm", None)
    if u_period_mm:
        positions, labels = _azimuth_deg_ticks(u_period_mm)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        xlabel = "azimuth, degrees from south (seam/N at both edges)"
        if isinstance(unwrapped, FrustumReceiver):
            xlabel += "\n" + _frustum_rect_distortion_note(unwrapped)
        ax.set_xlabel(xlabel, fontsize=8 if isinstance(unwrapped, FrustumReceiver) else 10)
    else:
        ax.set_xlabel("u (mm)")
    ax.set_ylabel("v (mm)")
    ax.set_title(f"{mode}, {elapsed_ms:.0f} ms")
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi)
    finally:
        plt.close(fig)
    return buf.getvalue()


def _frustum_fan_sin_half_angle(frustum: "FrustumReceiver") -> float:
    """``sin`` of the frustum's own cone half-angle -- ``|r_top - r_bot| /
    slant_length`` (opposite over hypotenuse of the bottom-to-top-rim
    triangle). See :func:`_render_flux_fan_png`'s docstring for why this one
    ratio holds at every latitude of a right circular cone, which is what
    makes it the single scale factor the whole fan development needs."""
    return abs(frustum.r_top_mm - frustum.r_bot_mm) / frustum.slant_length_mm


def _frustum_fan_rho_direction(frustum: "FrustumReceiver") -> float:
    """``+1`` if distance-from-the-(virtual)-apex INCREASES from the bottom
    rim to the top rim, ``-1`` if it decreases -- i.e. whether the apex sits
    below the band (the common, "normal" ``r_top > r_bot`` case: climbing
    the band means climbing away from the apex) or above it (an INVERTED
    frustum, ``r_top < r_bot``: the band narrows going up, so climbing it
    means approaching the apex, and rho falls as ``v`` rises). ``v`` itself
    (:meth:`FrustumReceiver.intersect`'s slant distance from the bottom rim)
    is always non-negative and increasing regardless of orientation -- this
    sign is what turns that into the correct SIGNED step on rho."""
    return 1.0 if frustum.r_top_mm >= frustum.r_bot_mm else -1.0


def _frustum_fan_xy_grid_m(
    frustum: "FrustumReceiver", u_edges: np.ndarray, v_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` meshgrid, meters, of the frustum's true developed (fan)
    surface at every ``(u_edges, v_edges)`` bin corner -- the pure geometry
    :func:`_render_flux_fan_png` paints, split out so it can be pinned by
    ``tests/test_flux_fan.py`` without touching matplotlib. ``rho(v) =
    rho_bot_mm + direction*v`` (``v`` is already slant distance from the
    bottom rim -- ``FrustumReceiver.intersect``'s own docstring; ``direction``
    is :func:`_frustum_fan_rho_direction`, +1 for a normal frustum, -1 for
    an inverted one where the virtual apex sits above the band instead of
    below it); ``phi(u) = (u / r_mean_mm) * sin_half_angle`` scales the full
    azimuth down to the developed sector's true angle. Local ``(x, y) =
    (rho*sin(phi), -rho*cos(phi))`` is the same trig
    :meth:`FrustumReceiver.uv_to_world` uses for ``(r, az)``, so the apex
    sits at the origin and the sector opens downward, symmetric about the
    vertical."""
    sin_half_angle = _frustum_fan_sin_half_angle(frustum)
    rho_bot_mm = frustum.r_bot_mm / sin_half_angle
    direction = _frustum_fan_rho_direction(frustum)
    rho_edges_m = (rho_bot_mm + direction * v_edges) / 1000.0
    phi_edges = (u_edges / frustum.r_mean_mm) * sin_half_angle
    phi_grid, rho_grid = np.meshgrid(phi_edges, rho_edges_m)
    return rho_grid * np.sin(phi_grid), -rho_grid * np.cos(phi_grid)


def _frustum_fan_cardinal_points_m(
    frustum: "FrustumReceiver", u_edges: np.ndarray, v_edges: np.ndarray
) -> list[tuple[str, float, float]]:
    """``[(letter, x_m, y_m), ...]`` for :data:`_AZIMUTH_CARDINALS`, placed
    just outside the fan's own OUTER rim -- ``max(rho_bot, rho_top) * 1.05``,
    since an inverted frustum's outer (apex-farthest) rim is the bottom, not
    the top -- at each cardinal's true angular position: the same
    ``u``-fraction -> ``sin_half_angle``-scaled-angle map
    :func:`_frustum_fan_xy_grid_m` uses for the flux surface itself, so a
    marker always sits on the physical seam/E/W point it names, never a
    cosmetic quarter-split of the sector. Two entries share the label "N"
    (both edges are the one physical seam, at different developed-plane
    positions) -- a list, not a dict keyed by letter, so neither is lost."""
    sin_half_angle = _frustum_fan_sin_half_angle(frustum)
    rho_bot_mm = frustum.r_bot_mm / sin_half_angle
    rho_top_mm = rho_bot_mm + _frustum_fan_rho_direction(frustum) * float(v_edges[-1])
    rho_out_m = max(rho_bot_mm, rho_top_mm) * 1.05 / 1000.0
    half_circ_u = 0.5 * (float(u_edges[-1]) - float(u_edges[0]))
    points = []
    for letter, frac in _AZIMUTH_CARDINALS:
        phi = (frac * half_circ_u / frustum.r_mean_mm) * sin_half_angle
        points.append((letter, rho_out_m * np.sin(phi), -rho_out_m * np.cos(phi)))
    return points


def _render_flux_fan_png(
    flux: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    frustum: "FrustumReceiver",
    mode: str,
    elapsed_ms: float,
    dpi: int = 110,
) -> bytes:
    """v0.2 followups item 2: the frustum's TRUE developed view -- an
    annular sector, not the parameter-space rectangle :func:`_render_flux_png`
    always draws (see :func:`_frustum_rect_distortion_note` for exactly how
    much that rectangle distorts). The flux VALUES were already correct
    (``FrustumReceiver.bin_areas_m2`` divides by true per-row area); only the
    SHAPE drawn was wrong.

    Standard technical-drawing cone development: unrolling a right circular
    cone's lateral surface produces an annular sector whose radius from the
    (virtual) cone apex is the TRUE slant distance -- ``rho(v) = rho_bot +
    v`` since ``v`` is already slant distance from the bottom rim
    (``FrustumReceiver.intersect``'s own docstring) -- and whose angle is the
    full ``2*pi`` azimuth scaled by ``sin(half_angle)``. ``sin(half_angle) =
    |r_top - r_bot| / slant_length`` (opposite over hypotenuse of the
    bottom-to-top-rim triangle): a right circular cone has ``r(z)/rho(z)``
    constant at every latitude (both are linear in the same distance-from-
    apex parameter), which is what makes ONE sector angle correct for the
    whole band -- verified as the exact inverse of ``intersect``/``bin_areas_m2``
    by ``tests/test_flux_fan.py``'s own closed-form checks against
    ``r_bot_mm``/``r_top_mm``.

    Screen layout: local ``(x, y) = (rho*sin(phi), -rho*cos(phi))`` -- the
    SAME trig :meth:`FrustumReceiver.uv_to_world` uses for ``(r, az)`` --
    puts the (virtual) apex at the origin and the sector opening downward,
    symmetric about the vertical; cardinal markers use the identical
    u-fraction positions as :data:`_AZIMUTH_CARDINALS`, scaled through the
    same ``sin(half_angle)`` factor, so they land on the true physical
    seam/E/W points, not a cosmetic quarter-split of whatever angle the
    sector happens to draw at.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_grid, y_grid = _frustum_fan_xy_grid_m(frustum, u_edges, v_edges)
    cardinals_m = _frustum_fan_cardinal_points_m(frustum, u_edges, v_edges)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    im = ax.pcolormesh(x_grid, y_grid, flux / 1000.0, cmap="magma", shading="flat")
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label("kW/m²")

    for letter, x, y in cardinals_m:
        ax.annotate(
            letter,
            (x, y),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#3a3a3a",
        )

    # The cardinal markers sit deliberately OUTSIDE the pcolormesh data (just
    # past the outer rim -- see _frustum_fan_cardinal_points_m), which
    # pcolormesh's own autoscale knows nothing about: left alone, the N/E/W
    # letters render past the axes' auto-fit limits and are clipped off the
    # saved PNG (caught visually verifying this function's own output).
    # Explicitly folding the marker positions into the data limits before
    # autoscaling fixes that; harmless if a caller's markers ever happened
    # to already sit inside the mesh.
    xs = [x for _letter, x, _y in cardinals_m]
    ys = [y for _letter, _x, y in cardinals_m]
    ax.update_datalim(list(zip(xs, ys)))
    ax.autoscale_view()

    ax.set_xlabel("developed x (m)")
    ax.set_ylabel("developed y (m)")
    ax.set_title(f"{mode}, true developed (fan) view, {elapsed_ms:.0f} ms", fontsize=10)
    fig.tight_layout()

    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi)
    finally:
        plt.close(fig)
    return buf.getvalue()


#: Target resolution (per axis) for :func:`_flux_grid_payload` -- deliberately
#: coarser than FLUX_GRID (128): the drape is an orientation view ("where the
#: hot spot physically sits", spec §M.3), not the quantitative one, so it does
#: not need the same resolution the stored flux map and its CSV export do.
FLUX_GRID_TEXTURE_DIM = 64


def _flux_grid_payload(flux: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray) -> dict:
    """Downsampled raw flux grid for the 3D receiver drape (js/scene3d.js).

    ``flux`` is ``(n_v, n_u)`` W/m^2, the same array :func:`_render_flux_png`
    plots with ``origin="lower"`` -- row 0 is ``v_edges[0]`` (the bottom of
    the unrolled/plan map), increasing with row index, exactly like that
    plot's own y-axis. The client builds a canvas texture from ``values``
    (row-major, same row order) and maps it onto the receiver mesh with UVs
    baked from the identical physics convention (see scene3d.js's
    receiver-UV comment), so this payload only needs to carry the grid and
    its ``(u, v)`` extent in mm -- not the receiver's shape, which the
    existing ``scene.receiver`` block already describes.

    Downsamples to :data:`FLUX_GRID_TEXTURE_DIM` per axis by block-averaging
    (128x128 -> 64x64 for a flat window; a curved receiver's wider adaptive
    grid -- see :func:`_receiver_flux_grid` -- is sized as a multiple of
    this dim so it divides evenly too) and rounds to
    kW/m^2 with 2 decimal digits, both purely to keep this opt-in field
    small: 64*64 = 4096 numbers at ~5-6 bytes each (a value like "123.45,")
    is roughly 20-25 KB of JSON, versus ~100 KB+ for the full-resolution grid
    at the same rounding. Nothing here is stored -- it is recomputed from the
    same ``flux``/``u_edges``/``v_edges`` every other reading in the response
    already came from.
    """
    n_v, n_u = flux.shape
    factor_v = max(1, n_v // FLUX_GRID_TEXTURE_DIM)
    factor_u = max(1, n_u // FLUX_GRID_TEXTURE_DIM)
    trimmed_v = (n_v // factor_v) * factor_v
    trimmed_u = (n_u // factor_u) * factor_u
    down = flux[:trimmed_v, :trimmed_u].reshape(trimmed_v // factor_v, factor_v, trimmed_u // factor_u, factor_u).mean(
        axis=(1, 3)
    )
    kw_m2 = np.round(down / 1000.0, 2)
    return {
        "n_u": int(down.shape[1]),
        "n_v": int(down.shape[0]),
        "u_min_mm": float(u_edges[0]),
        "u_max_mm": float(u_edges[-1]),
        "v_min_mm": float(v_edges[0]),
        "v_max_mm": float(v_edges[-1]),
        "unit": "kW/m2",
        "values": [_clean(x) for x in kw_m2.flatten().tolist()],
    }


def _aperture_metrics(
    flux_w_m2: np.ndarray,
    u_min_mm: float,
    u_max_mm: float,
    v_min_mm: float,
    v_max_mm: float,
    center_u_mm: float,
    center_v_mm: float,
    radius_mm: float,
) -> dict:
    """Reference implementation of spec §M.4's analysis-aperture math: power
    within a circle of ``radius_mm`` centred on ``(center_u_mm, center_v_mm)``,
    read straight off an already-computed, uniform-bin flux grid -- no
    trace, and no receiver-specific knowledge beyond "bin area is uniform"
    (exact for a flat window and for a cylinder's own unrolled arc length;
    NOT exact for a frustum's position-dependent bin area -- see
    :meth:`~heliostat.geometry.receiver.FrustumReceiver.bin_areas_m2` --
    which is exactly why spec §M.4 scopes the aperture to flat receivers
    first, curved ones "later if wanted").

    This mirrors, bin for bin, ``js/tabs/analysis.js``'s own
    ``apertureMetrics`` -- the function that actually drives the live,
    drag-as-you-go readout from the grid the browser already fetched (spec
    §M.4: "ALL computed frontend-side from the flux grid", no server round
    trip per drag frame). It is not wired into any endpoint; it exists here,
    beside that JS twin, purely so the formula has an automated check --
    this repo runs pytest only, with no JS test runner, so a synthetic-grid
    analytic case (a uniform-flux disk, where the answer is exact
    arithmetic) is checked against this Python copy instead of the shipped
    JS directly. Keep the two in lockstep if the formula ever changes.

    A grid bin counts as "inside" the aperture when its CENTRE lies within
    ``radius_mm`` -- the standard discretization of encircled power. Average
    flux divides by the aperture's own ideal circular area (pi * r^2), not
    the discretized sum of included bin areas, matching mockup M17's own
    worked example (9.61 MW / (pi * 3.80 m^2) ~= 212 kW/m^2).
    """
    n_v, n_u = flux_w_m2.shape
    du_mm = (u_max_mm - u_min_mm) / n_u
    dv_mm = (v_max_mm - v_min_mm) / n_v
    u_mid = u_min_mm + (np.arange(n_u) + 0.5) * du_mm
    v_mid = v_min_mm + (np.arange(n_v) + 0.5) * dv_mm
    gu, gv = np.meshgrid(u_mid, v_mid)  # (n_v, n_u), matches flux_w_m2's own shape
    bin_area_m2 = (du_mm / 1000.0) * (dv_mm / 1000.0)
    inside = (gu - center_u_mm) ** 2 + (gv - center_v_mm) ** 2 <= radius_mm**2
    power_w = float(np.sum(flux_w_m2[inside]) * bin_area_m2)
    radius_m = radius_mm / 1000.0
    area_m2 = math.pi * radius_m**2
    avg_flux_w_m2 = power_w / area_m2 if area_m2 > 0 else 0.0
    return {
        "power_w": power_w,
        "avg_flux_w_m2": avg_flux_w_m2,
        "n_bins_inside": int(np.sum(inside)),
    }


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

    # Importing matplotlib itself is cheap; done here, synchronously, so
    # every request-serving thread finds it already fully loaded and never
    # races another thread's first import of it (that race is a real
    # circular-import crash, not just a slow path). The expensive part --
    # font discovery and first-text-layout caching -- runs on a background
    # thread below (_warm_matplotlib) so it lands before a visitor's first
    # Heliostat Shape view instead of during it.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    from matplotlib.figure import Figure  # noqa: F401
    from matplotlib.ticker import MaxNLocator  # noqa: F401

    threading.Thread(target=_warm_matplotlib, daemon=True, name="heliostat-mpl-warmup").start()

    @app.exception_handler(ValueError)
    def _value_error_is_422(request: Request, exc: ValueError) -> JSONResponse:
        # A safety net, not the primary path: most endpoints already catch
        # their own ValueError (invalid optics_params, an aim point below
        # the heliostat plane, ...) and raise HTTPException(422) with the
        # same message. This only catches whatever a future call site
        # forgets to -- a bad request should read as "you asked for
        # something impossible", never as an unhandled 500.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    def _validation_error_body_is_json_safe(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's own default handler already 422s a RequestValidationError
        # -- this exists only so the body still renders when the rejected
        # input was itself Infinity/NaN (see _StrictModel): unsanitized, it
        # sits in exc.errors()'s "input", and Starlette's JSONResponse
        # (allow_nan=False) fails to encode it, turning a clean 422 into an
        # opaque "not JSON compliant" one.
        detail = _json_safe(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """The workspace. Its modules and assets load from ``/static/``."""
        html = (STATIC_DIR / "next" / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @app.get("/legacy", response_class=HTMLResponse)
    def legacy_index() -> HTMLResponse:
        """The previous single-file UI, kept reachable for one release."""
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

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
            optics_params = resolve_optics_params(body.optics, body.optics_params)
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

        kept_steps = _day_flux_step_indices(len(steps), cap=MAX_DAY_FLUX_MAPS)
        # Cheap (no tracing) -- only for the FEA CSV export's header comment
        # (receiver kind/curved-unrolling note), so a kept timestep's export
        # never has to re-resolve optics params or rebuild the geometry.
        _, day_receiver = _geometry_for(body.optics, optics_params)
        if body.layout is None:
            day_heliostat_desc = (
                f"single heliostat at x={body.heliostat_x_mm / 1000.0:.3f} m, "
                f"y={body.heliostat_y_mm / 1000.0:.3f} m"
            )
        else:
            day_heliostat_desc = None  # filled in per-step once n_heliostats is known

        def work(job):
            rows = []
            #: Timesteps whose own trace raised -- kept out of `rows`
            #: entirely (a fabricated zero-power sample would quietly bias
            #: the day's energy integral) and reported here instead, so one
            #: bad timestep costs the day that one point, not the run.
            failed_steps: list[dict] = []
            for index, step in enumerate(steps):
                if job.cancelled():
                    break
                job.detail = f"{step.key} ({step.solar_el_deg:.1f}° elevation)"
                want_flux = index in kept_steps
                t0 = time.perf_counter()
                try:
                    metrics = _trace_instant_metrics(
                        body,
                        step.solar_az_deg,
                        step.solar_el_deg,
                        want_flux=want_flux,
                        # §C's remaining honest gap: a stored step used to
                        # carry no secondary map at all. Gated on want_flux
                        # too -- MAX_DAY_FLUX_MAPS is the one storage budget
                        # both the receiver grid and the secondary grid live
                        # under, not two separate caps -- and on the
                        # request's own opt-in (api.js's buildTraceRequest,
                        # reused verbatim by buildDayRequest, always sets
                        # it; a caller that doesn't ask pays nothing extra).
                        want_secondary_flux=want_flux and body.include_secondary_flux,
                        should_cancel=job.cancelled,
                        step_key=index,
                    )
                except _TraceCancelled:
                    break
                except Exception as exc:  # noqa: BLE001 - isolated per timestep, see failed_steps
                    failed_steps.append(
                        {
                            "key": step.key,
                            "hour": round(float(step.hour), 4),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    job.done = index + 1
                    continue
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                # Spec §M.7: this timestep's own site DNI, from its own
                # elevation -- applied HERE, once, rather than inside
                # _trace_instant_metrics (which the year endpoint also
                # calls, through its OWN choke point -- energy.py's
                # dni_provider -- and must not have this scaling applied a
                # second time; see year_start's own comment). A day sweep's
                # power/flux is therefore honestly the number this
                # project's DNI setting says it is at each timestep's own
                # sun elevation, not always flat 1000 regardless of where
                # the sun is (the gap the rider actually complains about).
                step_scale, step_dni_w_m2 = _resolve_dni(body.dni, step.solar_el_deg)
                metrics["power_w"] = metrics["power_w"] * step_scale
                metrics["peak_flux_kw_m2"] = metrics["peak_flux_kw_m2"] * step_scale
                metrics["mean_flux_kw_m2"] = metrics["mean_flux_kw_m2"] * step_scale
                if "flux" in metrics:
                    metrics["flux"] = metrics["flux"] * step_scale
                if "secondary_flux" in metrics:
                    metrics["secondary_flux"] = metrics["secondary_flux"] * step_scale
                    metrics["secondary_power_w"] = metrics["secondary_power_w"] * step_scale
                if want_flux:
                    job.blobs[_day_flux_blob_key(index)] = _render_flux_png(
                        metrics["flux"],
                        metrics["u_edges"],
                        metrics["v_edges"],
                        body.mode,
                        elapsed_ms,
                        receiver=day_receiver,
                    )
                    heliostat_desc = day_heliostat_desc or (
                        f"field of {metrics['n_heliostats']} heliostats"
                    )
                    subject = _fea_subject_line(
                        heliostat_desc, step.solar_az_deg, step.solar_el_deg, body.mode
                    )
                    job.blobs[_day_flux_fea_blob_key(index)] = _flux_fea_csv(
                        metrics["flux"], metrics["u_edges"], metrics["v_edges"], day_receiver, subject
                    )
                    # §M.4: the raw grid the Analysis-tab aperture reads,
                    # never re-traced -- same payload shape §M.3's 3D drape
                    # already sends over the wire (_flux_grid_payload).
                    job.blobs[_day_flux_grid_blob_key(index)] = json.dumps(
                        _flux_grid_payload(metrics["flux"], metrics["u_edges"], metrics["v_edges"])
                    ).encode("utf-8")
                    # §C: that same step's secondary-surface map, present in
                    # `metrics` exactly when _trace_instant_metrics found one
                    # (optics has a flux map -- axicon/Cassegrain; silently
                    # absent, so no blob at all, for prime_focus, keeping
                    # its blobs byte-identical to before this landed).
                    if "secondary_flux" in metrics:
                        job.blobs[_day_secondary_grid_blob_key(index)] = json.dumps(
                            _secondary_payload(
                                metrics["secondary_flux"],
                                metrics["secondary_u_edges"],
                                metrics["secondary_v_edges"],
                                metrics["secondary_power_w"],
                                metrics["secondary_fidelity"],
                                secondary_reflectance=getattr(
                                    optics_params, "secondary_reflectance", 0.90
                                ),
                                include_flux_grid=True,
                            )
                        ).encode("utf-8")
                rows.append(
                    {
                        "key": step.key,
                        "hour": round(float(step.hour), 4),
                        "solar_az_deg": round(float(step.solar_az_deg), 3),
                        "solar_el_deg": round(float(step.solar_el_deg), 3),
                        **{
                            k: (None if v is None or not np.isfinite(v) else round(float(v), 4))
                            for k, v in metrics.items()
                            if k
                            not in (
                                "centroid_mm",
                                "n_heliostats",
                                "flux",
                                "u_edges",
                                "v_edges",
                                "secondary_flux",
                                "secondary_u_edges",
                                "secondary_v_edges",
                                "secondary_power_w",
                                "secondary_fidelity",
                            )
                        },
                        "n_heliostats": metrics["n_heliostats"],
                        "has_flux_map": want_flux,
                        # Spec §M.7: the site DNI actually applied to THIS
                        # timestep's power/flux above (was previously a
                        # separate, display-only clear-sky-only computation
                        # that never affected power_w -- see this endpoint's
                        # git history; now the two are the same number by
                        # construction).
                        "dni_w_m2": round(step_dni_w_m2, 2),
                    }
                )
                job.done = index + 1
            return {
                "steps": rows,
                "failed_steps": failed_steps,
                "energy_kwh": round(_day_energy_kwh(rows), 3),
                "date": f"{body.site.year:04d}-{body.site.month:02d}-{body.site.day:02d}",
                "mode": body.mode,
                "optics": body.optics,
                "n_heliostats": rows[0]["n_heliostats"] if rows else 0,
                "dni_note": body.dni.describe(),
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

    @app.get("/api/day/flux/{job_id}/{step}.png")
    def day_flux_png(job_id: str, step: int) -> Response:
        """One timestep's flux map from a finished day sweep.

        Rendered once, during the sweep itself (see ``day_start``), and
        served back as-is here -- no re-trace. Same unknown-job/still-running
        behaviour as ``/api/day/result``. A step past the run's own count,
        or one the sweep did not keep a map for (``MAX_DAY_FLUX_MAPS``; each
        result row says which via ``has_flux_map``), is also a 404 -- there
        is nothing stored to serve either way.
        """
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        steps = (job.result or {}).get("steps") or []
        if not 0 <= step < len(steps):
            raise HTTPException(status_code=404, detail=f"no timestep {step} in that day's run")
        png_bytes = job.blobs.get(_day_flux_blob_key(step))
        if png_bytes is None:
            raise HTTPException(status_code=404, detail="no stored flux map for that timestep")
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/api/day/flux/{job_id}/{step}.csv")
    def day_flux_fea_csv(job_id: str, step: int) -> Response:
        """That same timestep's flux map as a §D-convention FEA CSV grid.

        Built once, alongside the PNG, during the sweep itself (see
        ``day_start``) -- no re-trace here either, and the same
        ``has_flux_map``/404 rules as ``.../{step}.png`` apply.
        """
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        steps = (job.result or {}).get("steps") or []
        if not 0 <= step < len(steps):
            raise HTTPException(status_code=404, detail=f"no timestep {step} in that day's run")
        csv_bytes = job.blobs.get(_day_flux_fea_blob_key(step))
        if csv_bytes is None:
            raise HTTPException(status_code=404, detail="no stored flux map for that timestep")
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="heliostat-day-flux-fea-{step}.csv"'
            },
        )

    @app.get("/api/day/flux/{job_id}/{step}.grid.json")
    def day_flux_grid_json(job_id: str, step: int) -> Response:
        """That same timestep's flux map as a compact JSON grid (§M.4): the
        raw numbers the Analysis tab's aperture reads and does its own
        arithmetic against, client-side -- power within a radius, average
        flux, average concentration, the encircled-power curve. Built once,
        alongside the PNG and the FEA CSV, during the sweep itself (see
        ``day_start``) -- no re-trace here either, and the same
        ``has_flux_map``/404 rules as ``.../{step}.png`` apply.
        """
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        steps = (job.result or {}).get("steps") or []
        if not 0 <= step < len(steps):
            raise HTTPException(status_code=404, detail=f"no timestep {step} in that day's run")
        grid_bytes = job.blobs.get(_day_flux_grid_blob_key(step))
        if grid_bytes is None:
            raise HTTPException(status_code=404, detail="no stored flux map for that timestep")
        return Response(content=grid_bytes, media_type="application/json")

    @app.get("/api/day/flux/{job_id}/{step}.secondary.json")
    def day_flux_secondary_json(job_id: str, step: int) -> Response:
        """That same timestep's secondary-surface flux map (spec §C), the
        Analysis tab's Receiver | Secondary selector's own stored-step gap:
        a stored day-sweep step used to carry no secondary data at all
        (only a live re-trace of that step's own sun angles did), leaving
        the selector disabled with a tooltip no matter what the sweep's
        optics was. Same shape as ``.../{step}.grid.json`` -- a
        :func:`_secondary_payload` block (``power_w``, ``peak_flux_kw_m2``,
        the absorbed-heat numbers, ``fidelity``, ``flux_grid``), built once
        alongside the PNG/FEA-CSV/grid during the sweep itself (see
        ``day_start``), never a re-trace.

        Same job-state rules and step-range 404 as ``.../{step}.grid.json``,
        plus its own narrower 404: this step's own blob is present only
        when ``day_start`` was asked for one (``include_secondary_flux``)
        AND the sweep's optics has a secondary flux map at all
        (axicon/Cassegrain -- prime focus, or a run finished before this
        endpoint existed, both 404 here exactly like a step the receiver
        grid cap left out).
        """
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        steps = (job.result or {}).get("steps") or []
        if not 0 <= step < len(steps):
            raise HTTPException(status_code=404, detail=f"no timestep {step} in that day's run")
        secondary_bytes = job.blobs.get(_day_secondary_grid_blob_key(step))
        if secondary_bytes is None:
            raise HTTPException(
                status_code=404, detail="no stored secondary flux map for that timestep"
            )
        return Response(content=secondary_bytes, media_type="application/json")

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

    @app.post("/api/year/start")
    def year_start(body: YearTraceRequest) -> JSONResponse:
        """Estimate annual collection, on a background thread. Same shape
        as ``/api/day/start``: returns a job id immediately, poll
        ``/api/year/status/{job_id}``."""
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        year = body.site.year
        cfg = _year_energy_cfg(body)
        try:
            trace_dates = _year_trace_dates(cfg, year, body.fast_mode)
            cfg.sweep.dates = trace_dates
            report_days = _year_report_days(cfg, trace_dates, year, body.fast_mode)
            steps = build_time_grid(cfg, trace_dates)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not steps:
            raise HTTPException(
                status_code=422,
                detail="the sun does not rise at that site on any of the sample dates",
            )
        try:
            cfg.field.mirror_area_m2 = _year_mirror_area_m2(body, optics_params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        steps_per_date: dict[_dt.date, int] = {}
        for step in steps:
            steps_per_date[step.date] = steps_per_date.get(step.date, 0) + 1

        def work(job):
            rows: list[dict] = []
            rows_per_date: dict[_dt.date, int] = {}
            #: Timesteps whose own trace raised. Left out of `rows`, exactly
            #: like a cancelled tail -- rows_per_date[step.date] then falls
            #: short of steps_per_date[step.date] below, so the "only
            #: fully-traced dates count" filter already excludes that date
            #: on its own; one bad timestep costs its date, not the year.
            failed_steps: list[dict] = []
            n_heliostats = 1
            for index, step in enumerate(steps):
                if job.cancelled():
                    break
                job.detail = f"{step.date:%Y-%m-%d} {step.hour:.2f}h ({step.solar_el_deg:.1f}° elevation)"
                try:
                    metrics = _trace_instant_metrics(
                        body,
                        step.solar_az_deg,
                        step.solar_el_deg,
                        should_cancel=job.cancelled,
                        step_key=index,
                    )
                except _TraceCancelled:
                    break
                except Exception as exc:  # noqa: BLE001 - isolated per timestep, see failed_steps
                    failed_steps.append(
                        {
                            "date": step.date.isoformat(),
                            "hour": float(step.hour),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    job.done = index + 1
                    continue
                n_heliostats = metrics["n_heliostats"]
                rows.append(
                    {
                        "date": step.date,
                        "hour": float(step.hour),
                        "heliostat_id": 0,
                        "power_w": float(metrics["power_w"]),
                        "solar_az_deg": float(step.solar_az_deg),
                        "solar_el_deg": float(step.solar_el_deg),
                    }
                )
                rows_per_date[step.date] = rows_per_date.get(step.date, 0) + 1
                job.done = index + 1

            # A date cut short by cancellation is not a real day -- its
            # partial trapezoid would read as a low-collection day rather
            # than an unfinished one, so only fully-traced dates count.
            complete = sorted(
                d
                for d in trace_dates
                if steps_per_date.get(d, 0) > 0 and rows_per_date.get(d, 0) == steps_per_date[d]
            )
            if len(complete) < 2:
                return {
                    "days": [],
                    "n_days_traced": len(complete),
                    "fast_mode": body.fast_mode,
                    "failed_steps": failed_steps,
                }

            summary = pd.DataFrame(rows)
            # Spec §M.7: this project's own site DNI, not a hardcoded
            # ClearSkyDNI -- energy.annual_energy/traced_day_energy are
            # THEIR OWN choke point for turning `rows`' flat-1000
            # `power_w` into a real number (they divide by STANDARD_DNI to
            # get eta_optical, then re-multiply by dni_provider.dni(date,
            # hour) -- see that module's docstring). YearTraceRequest.dni
            # defaults to clear-sky (overriding the base class's own
            # constant/1000 default -- see that field's comment), so a
            # request with no ``dni`` at all resolves to exactly the same
            # ClearSkyDNI(cfg.site) this endpoint has always used.
            dni_provider = body.dni.provider(cfg.site)
            annual = energy.annual_energy(summary, cfg, dni_provider, year=year, n_heliostats=n_heliostats)

            days_out = []
            for entry in report_days:
                if entry["source_date"] not in complete:
                    continue
                day = energy.traced_day_energy(
                    summary, cfg, dni_provider, date=entry["date"], source_date=entry["source_date"]
                )
                days_out.append(
                    {
                        "date": entry["date"].isoformat(),
                        "source_date": entry["source_date"].isoformat(),
                        "traced": entry["traced"],
                        "declination_deg": round(
                            float(energy._declination_of(cfg, entry["source_date"])), 3
                        ),
                        "energy_kwh": round(day["energy_kwh"], 3),
                        "peak_power_kw": round(day["peak_power_kw"], 3),
                    }
                )

            eff = annual["annual_optical_efficiency"]
            extrap = annual["extrapolated_fraction"]
            return {
                "annual_energy_mwh": round(annual["annual_energy_mwh"], 3),
                "annual_energy_kwh": round(annual["annual_energy_kwh"], 1),
                "annual_dni_kwh_m2": round(annual["annual_dni_kwh_m2"], 1),
                "annual_optical_efficiency": round(eff, 4) if np.isfinite(eff) else None,
                "mirror_area_m2": round(annual["mirror_area_m2"], 2),
                "n_heliostats": annual["n_heliostats"],
                "n_days_traced": len(complete),
                "fast_mode": body.fast_mode,
                "year": year,
                "dni_provider": dni_provider.describe(),
                "dni_note": body.dni.describe(),
                "extrapolated_fraction": round(extrap, 4) if np.isfinite(extrap) else None,
                "days": days_out,
                "failed_steps": failed_steps,
            }

        job = JOBS.start(len(steps), work, label=f"year estimate, {len(trace_dates)} dates traced")
        return JSONResponse(job.snapshot())

    @app.get("/api/year/status/{job_id}")
    def year_status(job_id: str) -> JSONResponse:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return JSONResponse(job.snapshot())

    @app.post("/api/year/cancel/{job_id}")
    def year_cancel(job_id: str) -> JSONResponse:
        if not JOBS.cancel(job_id):
            raise HTTPException(status_code=409, detail="that job is not running")
        return JSONResponse({"cancelled": job_id})

    @app.get("/api/year/result/{job_id}")
    def year_result(job_id: str) -> JSONResponse:
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
        if payload.get("days"):
            payload["plot_png"] = base64.b64encode(_render_year_png(payload["days"])).decode("ascii")
        return JSONResponse(payload)

    @app.post("/api/trace/flux.csv")
    def trace_flux_csv(body: TraceRequest) -> Response:
        """The flux map of one trace as CSV, in kW/m2.

        Row and column headers are the receiver-plane coordinates of each
        bin centre in millimetres, so the grid is self-describing rather
        than a bare block of numbers whose axes live in another document.
        """
        flux, u_edges, v_edges, _receiver = _flux_grid_for(body)
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

    @app.post("/api/trace/flux_fea.csv")
    def trace_flux_fea_csv(body: TraceRequest) -> Response:
        """The flux map of one trace as a §D-convention FEA CSV grid.

        Same trace as ``/api/trace/flux.csv`` (:func:`_flux_grid_for` is
        shared, so this is never a second, possibly-drifted computation of
        the same map) -- only the file format differs: meters and W/m²
        instead of millimeters and kW/m², one ``x, y, flux`` point per row
        behind commented metadata instead of a labelled matrix, targeting
        ANSYS External Data import rather than a spreadsheet.
        """
        flux, u_edges, v_edges, receiver = _flux_grid_for(body)
        subject = _fea_subject_line(
            f"single heliostat at x={body.heliostat_x_mm / 1000.0:.3f} m, "
            f"y={body.heliostat_y_mm / 1000.0:.3f} m",
            body.solar_az_deg,
            body.solar_el_deg,
            body.mode,
        )
        csv_bytes = _flux_fea_csv(flux, u_edges, v_edges, receiver, subject)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="heliostat-flux-fea.csv"'},
        )

    @app.post("/api/trace/secondary_flux_fea.csv")
    def trace_secondary_flux_fea_csv(body: TraceRequest) -> Response:
        """Spec §C/§D: the secondary's own incident-flux map as an FEA CSV
        grid -- ``x, y, z, flux, absorbed`` (:func:`_secondary_flux_fea_csv`),
        the same commented-header convention as
        ``/api/trace/flux_fea.csv``/the sag export. 404s for a layout with
        no secondary flux map (prime focus; any secondary with no
        single-valued (u, v) parameterization) rather than returning an
        empty file.
        """
        grid = _secondary_flux_grid_for(body)
        if grid is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"optics={body.optics!r} has no secondary irradiance map "
                    "(prime focus has no secondary; only axicon/Cassegrain do)"
                ),
            )
        flux, u_edges, v_edges, secondary_reflectance, secondary = grid
        subject = _fea_subject_line(
            f"single heliostat at x={body.heliostat_x_mm / 1000.0:.3f} m, "
            f"y={body.heliostat_y_mm / 1000.0:.3f} m",
            body.solar_az_deg,
            body.solar_el_deg,
            body.mode,
        )
        csv_bytes = _secondary_flux_fea_csv(flux, u_edges, v_edges, secondary, secondary_reflectance, subject)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="heliostat-secondary-flux-fea.csv"'},
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
        try:
            design = _build_design(body.design)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        fig, ax = _preview_figure()
        ax.clear()
        try:
            design.preview(ax=ax)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=_PREVIEW_FIGURE_DPI)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return Response(content=buf.getvalue(), media_type="image/png")

    def _resolve_sag_request(body: "TraceRequest"):
        """``(sol, design, slant_range_mm, half_x_mm, half_y_mm)`` for a sag
        request -- the solve-and-build-design step shared by
        ``/api/design/sag`` and ``/api/design/sag.csv`` so the CSV export is
        built from exactly the same design object the PNG is, never a
        second solve that could drift from it.

        :raises ValueError: an unsolvable geometry -- callers turn this into
            a 422, same as every other endpoint in this module.
        """
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
        if design is None:
            half_x, half_y = MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM
        else:
            u0, u1, v0, v1 = design.bbox
            half_x = max(abs(u0), abs(u1))
            half_y = max(abs(v0), abs(v1))
        return sol, design, slant_range_mm, half_x, half_y

    @app.post("/api/design/sag")
    def design_sag(body: TraceRequest, cant: bool = True) -> Response:
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
            sol, design, slant_range_mm, half_x, half_y = _resolve_sag_request(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        png, span_mm, interval_mm = _render_sag_png(
            design, sol, body.design, half_x, half_y, include_cant=cant
        )
        headers = {"X-Slant-Range-M": f"{slant_range_mm / 1000.0:.3f}"}
        if span_mm is not None:
            headers["X-Peak-To-Valley-Mm"] = f"{span_mm:.6g}"
        if interval_mm is not None:
            headers["X-Contour-Interval-Mm"] = f"{interval_mm:g}"
        return Response(content=png, media_type="image/png", headers=headers)

    @app.post("/api/design/sag.csv")
    def design_sag_csv(body: TraceRequest, cant: bool = True) -> Response:
        """The sag map of this exact request's mirror as a §D-convention FEA
        CSV grid: ``x_m, y_m, z_sag_mm``, one row per grid point that lands
        on a facet.

        Sibling of ``/api/design/sag`` -- same solve, same design, same
        sampling grid (:func:`_resolve_sag_request`, :func:`_sag_grid_mm`),
        so this can never show a different surface than that PNG. ``cant``
        means the same thing here as there: whole-mirror figure (default) vs
        each facet measured from its own mounting plane.
        """
        if body.solar_el_deg <= 0:
            raise HTTPException(
                status_code=422,
                detail="solar_el_deg must be > 0 (the sun is below the horizon)",
            )
        try:
            sol, design, slant_range_mm, half_x, half_y = _resolve_sag_request(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        subject = _fea_subject_line(
            f"single heliostat at x={body.heliostat_x_mm / 1000.0:.3f} m, "
            f"y={body.heliostat_y_mm / 1000.0:.3f} m, slant range "
            f"{slant_range_mm / 1000.0:.3f} m, {body.design.surface} figure"
            + ("" if cant else " (per-facet, tilt removed)"),
            body.solar_az_deg,
            body.solar_el_deg,
            body.mode,
        )
        csv_bytes = _sag_fea_csv(design, sol, half_x, half_y, cant, subject)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={
                "X-Slant-Range-M": f"{slant_range_mm / 1000.0:.3f}",
                "Content-Disposition": 'attachment; filename="heliostat-sag-fea.csv"',
            },
        )

    @app.post("/api/secondary/sag")
    def secondary_sag(body: SecondarySagRequest) -> Response:
        """Spec §E2 "View sag": the secondary's own sag map -- nominal
        figure + parametric warp + imported map, summed
        (:func:`_secondary_sag_grid_mm`) -- same jet colormap and stated
        contour interval as the heliostat's own ``/api/design/sag``.
        """
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        secondary, _receiver = _geometry_for(body.optics, optics_params)
        gx, gy, sag = _secondary_sag_grid_mm(secondary, optics_params)
        png, span_mm, interval_mm = _render_secondary_sag_png(gx, gy, sag, secondary.aperture_radius_mm)
        headers = {}
        if span_mm is not None:
            headers["X-Peak-To-Valley-Mm"] = f"{span_mm:.6g}"
        if interval_mm is not None:
            headers["X-Contour-Interval-Mm"] = f"{interval_mm:g}"
        return Response(content=png, media_type="image/png", headers=headers)

    @app.post("/api/secondary/sag.csv")
    def secondary_sag_csv(body: SecondarySagRequest) -> Response:
        """§D-convention CSV export of ``/api/secondary/sag``'s exact
        surface -- ``x_m, y_m, z_sag_mm`` over the secondary's aperture disk
        (:func:`_secondary_sag_fea_csv`), so the export can never show a
        different surface than that picture.
        """
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        secondary, _receiver = _geometry_for(body.optics, optics_params)
        gx, gy, sag = _secondary_sag_grid_mm(secondary, optics_params)
        subject = _secondary_sag_subject_line(body.optics, secondary.aperture_radius_mm)
        csv_bytes = _secondary_sag_fea_csv(gx, gy, sag, secondary.aperture_radius_mm, subject)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="secondary-sag-fea.csv"'},
        )

    @app.post("/api/design/errormap/import")
    def design_errormap_import(body: ErrorMapImportRequest) -> JSONResponse:
        """§E: parse an uploaded §D-convention sag CSV into a measured
        error map, reporting what was read (grid size, aperture coverage,
        implied RMS slope) and the raw grid a client attaches verbatim to
        ``_DesignBase.error_map`` for every subsequent request -- this
        endpoint does not itself store anything; the app is stateless per
        request, same as every other design field.
        """
        try:
            error_map = parse_error_map_csv(body.csv)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"could not read error map CSV: {exc}") from exc

        ny, nx = error_map.grid_shape
        return JSONResponse(
            {
                "grid": error_map.to_storage_dict(),
                "grid_size": {"nx": nx, "ny": ny},
                "coverage_fraction": error_map.coverage_fraction,
                "rms_slope_mrad": error_map.rms_slope_mrad,
            }
        )

    @app.post("/api/design/errormap/stats")
    def design_errormap_stats(body: ErrorMapStatsRequest) -> JSONResponse:
        """Sibling of ``/api/design/errormap/import`` for a grid the client
        already has (loaded from the Library/a project) rather than a fresh
        CSV -- same three numbers, so the chip reads the same either way."""
        try:
            error_map = ErrorMap.from_storage_dict(body.grid)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise HTTPException(status_code=422, detail=f"could not read error map grid: {exc}") from exc
        ny, nx = error_map.grid_shape
        return JSONResponse(
            {
                "grid_size": {"nx": nx, "ny": ny},
                "coverage_fraction": error_map.coverage_fraction,
                "rms_slope_mrad": error_map.rms_slope_mrad,
            }
        )

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

        try:
            sol = _solve_for(
                body.optics,
                body.heliostat_x_mm,
                body.heliostat_y_mm,
                body.solar_az_deg,
                body.solar_el_deg,
                optics_params,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            error_map=_build_error_map(body.design),
            pointing_error_mrad=body.design.pointing_error_mrad,
            return_secondary_flux=body.include_secondary_flux,
            circumsolar_ratio=body.circumsolar_ratio,
            **_secondary_perturb_kwargs(optics_params),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Spec §M.7: the site DNI in effect for this instant -- the ONE
        # choke point every downstream watt/flux number is scaled through
        # (see DNISetting/_resolve_dni). Every backend result above (and the
        # secondary maps below) is still at the trace's native DNI=1000
        # normalisation at this point; dni_scale is what turns that into the
        # real number, applied once, here, rather than at every consumer.
        dni_scale, dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)

        if result["backend"] == "mc":
            traced_paths = result["paths"]
            traced_miss_paths = result["miss_paths"]
            traced_miss_dirs = result["miss_dirs"]
            xy = result["xy"]
            counters = result["counters"]
            watts_per_ray = result["watts_per_ray"] * dni_scale
            power_w = watts_per_ray * counters.get("in_window", 0)
            incident_power_w = None
            flux, u_edges, v_edges, rms_mm, centroid = _mc_flux_and_metrics(
                xy, watts_per_ray, receiver
            )
        else:
            traced_paths = None  # cone optics carries no rays; the scene samples its own
            traced_miss_paths = None
            traced_miss_dirs = None
            flux = result["flux"] * dni_scale
            u_edges, v_edges = result["u_edges"], result["v_edges"]
            power_w = result["power_w"] * dni_scale
            incident_power_w = result["incident_power_w"] * dni_scale
            counters = result["counters"]
            rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)

        # v0.2 followups item 2: the fan view only ever applies to a frustum
        # -- silently falls back to the rectangle for any other receiver
        # (flat, cylinder, or a request left over from before an optics
        # change), rather than 422ing a view preference the client did not
        # actually get to choose meaningfully.
        unwrapped_receiver = _unwrap_receiver_for_map(receiver)
        if body.flux_view == "fan" and isinstance(unwrapped_receiver, FrustumReceiver):
            png_bytes = _render_flux_fan_png(
                flux,
                u_edges,
                v_edges,
                unwrapped_receiver,
                body.mode,
                elapsed_ms,
                dpi=body.flux_png_dpi or 110,
            )
        else:
            png_bytes = _render_flux_png(
                flux,
                u_edges,
                v_edges,
                body.mode,
                elapsed_ms,
                dpi=body.flux_png_dpi or 110,
                receiver=receiver,
            )

        # Spec §C: incident flux on the secondary's own surface, alongside
        # the receiver map -- only when requested and only for a secondary
        # with a single-valued (u, v) parameterization (axicon/Cassegrain;
        # silently absent for prime_focus/pyramid, see
        # _secondary_maps_from_result).
        secondary_payload = None
        if body.include_secondary_flux:
            secondary_maps = _secondary_maps_from_result(result, secondary)
            if secondary_maps is not None:
                s_flux, s_u_edges, s_v_edges, s_power_w, s_fidelity = secondary_maps
                secondary_payload = _secondary_payload(
                    s_flux * dni_scale,
                    s_u_edges,
                    s_v_edges,
                    s_power_w * dni_scale,
                    s_fidelity,
                    secondary_reflectance=getattr(optics_params, "secondary_reflectance", 0.90),
                    include_flux_grid=body.include_flux_grid,
                )

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
            miss_paths=traced_miss_paths,
            miss_dirs=traced_miss_dirs,
        )

        return JSONResponse(
            {
                "power_w": _clean(power_w),
                "incident_power_w": _clean(incident_power_w),
                "note": _zero_power_note(counters, _clean(power_w)),
                "peak_flux_kw_m2": _clean(float(np.max(flux)) / 1000.0),
                "mean_flux_kw_m2": _clean(_mean_flux_kw_m2(flux, receiver.bin_areas_m2((flux.shape[1], flux.shape[0])))),
                "rms_radius_mm": _clean(rms_mm),
                "centroid_mm": [_clean(centroid[0]), _clean(centroid[1])],
                "counters": {k: int(v) for k, v in counters.items()},
                "elapsed_ms": elapsed_ms,
                "mode": body.mode,
                "flux_png": base64.b64encode(png_bytes).decode("ascii"),
                "flux_grid": _flux_grid_payload(flux, u_edges, v_edges) if body.include_flux_grid else None,
                "secondary": secondary_payload,
                "aim_point_mm": [aim_x_mm, aim_y_mm, aim_z_mm],
                "slant_range_m": slant_range_mm / 1000.0,
                # What the tower geometry actually resolved to, so the
                # client can populate its inspector without keeping a second
                # copy of these defaults.
                "optics_resolved": optics_params.model_dump(),
                "scene": scene,
                # Spec §M.7: the DNI this response's power/flux were scaled
                # by, and a short label for what assumed it -- so a live
                # trace, like every other surface, states its own DNI
                # instead of leaving it implicit (commit 45d6515's "a live
                # trace carries no DNI to divide by" is exactly this gap).
                "dni_w_m2": _clean(dni_w_m2),
                "dni_note": body.dni.describe(),
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

        One heliostat at a time by default (``workers`` unset or 1) -- this
        endpoint's behaviour is pinned for the scripts and tests that already
        call it synchronously. Pass ``workers`` > 1 to trace the field across
        a process pool instead (see :func:`_trace_field_heliostats`); a big
        field is better traced through ``/api/field/trace/start``, which
        parallelises by default and can be polled and cancelled instead of
        holding this request open for minutes. The response carries
        ``timings_ms`` (solve, occlusion, trace, scene) either way, to keep
        the cost measurable rather than anecdotal.

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

        # -- phase 3: one trace per heliostat, summed on the receiver. A
        # field's rays are drawn by the scene's own side trace, so this pass
        # has no use for MC ray paths -- and 600 heliostats' worth of them is
        # a lot of array for a picture that will not use it (mc_return_paths
        # inside _trace_field_heliostats is always False).
        u_edges, v_edges, bin_area_m2 = _flux_edges(receiver)

        # Spec §M.7: this field's site DNI, resolved once from the sun
        # elevation the whole field shares -- passed into
        # _trace_field_heliostats so every per-heliostat contribution (and
        # the summed total) is scaled through that one function's own choke
        # point, not re-scaled here afterwards.
        _dni_scale, dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)

        traced = _trace_field_heliostats(
            designs,
            xy_mm,
            ids,
            solutions,
            eta_shade,
            eta_block,
            eta_union,
            secondary,
            receiver,
            mode,
            body.solar_az_deg,
            body.solar_el_deg,
            body.design.slope_error_mrad,
            body.design.specularity_mrad,
            body.design.reflectance,
            u_edges,
            v_edges,
            bin_area_m2,
            error_map=_build_error_map(body.design),
            pointing_error_mrad=body.design.pointing_error_mrad,
            workers=body.workers or 1,
            return_secondary_flux=body.include_secondary_flux,
            dni_w_m2=dni_w_m2,
            circumsolar_ratio=body.circumsolar_ratio,
            **_secondary_perturb_kwargs(optics_params),
        )
        flux = traced["flux"]
        power_w = traced["power_w"]
        incident_power_w = traced["incident_power_w"]
        counters = traced["counters"]
        rows = traced["rows"]
        failed = traced["failed"]
        t_trace = time.perf_counter()

        secondary_payload = None
        if body.include_secondary_flux and "secondary_flux" in traced:
            secondary_payload = _secondary_payload(
                traced["secondary_flux"],
                traced["secondary_u_edges"],
                traced["secondary_v_edges"],
                traced["secondary_power_w"],
                traced["secondary_fidelity"] or "coarse",
                secondary_reflectance=getattr(optics_params, "secondary_reflectance", 0.90),
                include_flux_grid=body.include_flux_grid,
            )

        rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)
        elapsed_ms = (t_trace - t0) * 1000.0
        png_bytes = _render_flux_png(flux, u_edges, v_edges, body.mode, elapsed_ms, receiver=receiver)

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
                "mean_flux_kw_m2": _clean(_mean_flux_kw_m2(flux, bin_area_m2)),
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
                "flux_grid": _flux_grid_payload(flux, u_edges, v_edges) if body.include_flux_grid else None,
                "secondary": secondary_payload,
                "n_heliostats": n,
                "eta_min": _clean(float(np.min(eta_union))),
                "eta_median": _clean(float(np.median(eta_union))),
                "eta_max": _clean(float(np.max(eta_union))),
                "optics_resolved": optics_params.model_dump(),
                "heliostats": rows,
                "failed_heliostats": failed,
                "scene": scene,
                # Spec §M.7 -- see /api/trace's identically-named fields.
                "dni_w_m2": _clean(dni_w_m2),
                "dni_note": body.dni.describe(),
            }
        )

    @app.post("/api/field/trace/start")
    def field_trace_start(body: FieldTraceRequest) -> JSONResponse:
        """The same trace as ``/api/field/trace``, run on a background job
        instead of held open on the request.

        Same shape as ``/api/day/start``: returns a job id immediately,
        progress is ``done``/``total`` heliostats via
        ``/api/field/trace/status/{job_id}``, and the finished payload --
        identical in every key to a synchronous ``/api/field/trace``
        response -- is collected from ``/api/field/trace/result/{job_id}``
        (409 while still running, matching the day endpoint).

        Unlike ``/api/field/trace``, ``workers`` here defaults to
        ``max(1, cpu_count - 1)`` rather than 1 -- this is the endpoint a big
        field is meant to go through, so it parallelises unless told not to.
        A cancelled run has no result to fetch: a field's flux is a sum
        across every mirror, and a sum missing half its terms is not a
        smaller-but-valid answer worth returning.
        """
        if body.solar_el_deg <= 0:
            raise HTTPException(
                status_code=422,
                detail="solar_el_deg must be > 0 (the sun is below the horizon)",
            )
        # Phase 1 (solve) runs here, synchronously, exactly as
        # /api/field/trace runs it -- it is not the cost centre (a plain
        # scalar loop, not a trace) and a bad layout/optics combination
        # should 422 immediately rather than only be discovered by polling a
        # job. Only the parts that actually cost seconds to minutes --
        # occlusion and the per-heliostat trace -- run inside the job.
        t0 = time.perf_counter()
        try:
            optics_params = resolve_optics_params(body.optics, body.optics_params)
            xy_mm, ids = _field_positions(body.layout, body.exclude_ids)
            secondary, receiver = _geometry_for(body.optics, optics_params)
            solutions, designs, _slants = _solve_field(
                body.optics, optics_params, body.design, xy_mm, body.solar_az_deg, body.solar_el_deg
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        t_solve = time.perf_counter()

        n = xy_mm.shape[0]
        mode = body.trace_mode()
        workers = body.workers or _default_trace_workers()

        def work(job):
            job.detail = "computing shading and blocking"
            eta_shade, eta_block, eta_union, outline = _field_occlusion(
                xy_mm, ids, solutions, designs[0], body.solar_az_deg, body.solar_el_deg
            )
            t_occlusion = time.perf_counter()

            u_edges, v_edges, bin_area_m2 = _flux_edges(receiver)

            # Cost-weighted total, set once up front so eta_s/snapshot's
            # `frac` can weight progress from the very first callback (see
            # _heliostat_progress_weights and Job.weight_done/weight_total).
            job.weight_total = float(_heliostat_progress_weights(xy_mm).sum())

            def on_progress(done: int, weight_done: float) -> None:
                job.done = done
                job.weight_done = weight_done
                job.detail = f"{done} / {n} heliostats"

            # A cold pool (first field trace since the server started, or
            # the first one asking for more workers than it currently has)
            # spends several seconds spawning worker processes before any
            # of them can report progress -- see _field_pool_cold_start_expected.
            # Left at "0 / n heliostats", that whole stall looks identical
            # to a hung trace: the count never advances, because nothing
            # has run yet to advance it. Narrating it honestly here costs
            # nothing on a warm pool -- on_progress's first callback
            # overwrites this within milliseconds, same as before.
            if _field_pool_cold_start_expected(min(workers, n)):
                job.detail = "starting worker processes"
            else:
                job.detail = f"0 / {n} heliostats"
            # Spec §M.7 -- see /api/field/trace's identical comment.
            _dni_scale, dni_w_m2 = _resolve_dni(body.dni, body.solar_el_deg)
            try:
                traced = _trace_field_heliostats(
                    designs,
                    xy_mm,
                    ids,
                    solutions,
                    eta_shade,
                    eta_block,
                    eta_union,
                    secondary,
                    receiver,
                    mode,
                    body.solar_az_deg,
                    body.solar_el_deg,
                    body.design.slope_error_mrad,
                    body.design.specularity_mrad,
                    body.design.reflectance,
                    u_edges,
                    v_edges,
                    bin_area_m2,
                    error_map=_build_error_map(body.design),
                    pointing_error_mrad=body.design.pointing_error_mrad,
                    workers=workers,
                    should_cancel=job.cancelled,
                    on_progress=on_progress,
                    return_secondary_flux=body.include_secondary_flux,
                    dni_w_m2=dni_w_m2,
                    circumsolar_ratio=body.circumsolar_ratio,
                    **_secondary_perturb_kwargs(optics_params),
                )
            except _TraceCancelled:
                return None
            t_trace = time.perf_counter()

            flux = traced["flux"]
            power_w = traced["power_w"]
            incident_power_w = traced["incident_power_w"]
            counters = traced["counters"]
            rows = traced["rows"]
            failed = traced["failed"]

            secondary_payload = None
            if body.include_secondary_flux and "secondary_flux" in traced:
                secondary_payload = _secondary_payload(
                    traced["secondary_flux"],
                    traced["secondary_u_edges"],
                    traced["secondary_v_edges"],
                    traced["secondary_power_w"],
                    traced["secondary_fidelity"] or "coarse",
                    secondary_reflectance=getattr(optics_params, "secondary_reflectance", 0.90),
                    include_flux_grid=body.include_flux_grid,
                )

            rms_mm, centroid = _cone_metrics(flux, u_edges, v_edges)
            elapsed_ms = (t_trace - t0) * 1000.0
            png_bytes = _render_flux_png(flux, u_edges, v_edges, body.mode, elapsed_ms, receiver=receiver)

            job.detail = "building scene"
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

            return {
                "power_w": _clean(power_w),
                "incident_power_w": _clean(incident_power_w),
                "note": _zero_power_note(counters, _clean(power_w)),
                "peak_flux_kw_m2": _clean(float(np.max(flux)) / 1000.0),
                "mean_flux_kw_m2": _clean(_mean_flux_kw_m2(flux, bin_area_m2)),
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
                "flux_grid": _flux_grid_payload(flux, u_edges, v_edges) if body.include_flux_grid else None,
                "secondary": secondary_payload,
                "n_heliostats": n,
                "eta_min": _clean(float(np.min(eta_union))),
                "eta_median": _clean(float(np.median(eta_union))),
                "eta_max": _clean(float(np.max(eta_union))),
                "optics_resolved": optics_params.model_dump(),
                "heliostats": rows,
                "failed_heliostats": failed,
                "scene": scene,
                "workers": workers,
                # Spec §M.7 -- see /api/trace's identically-named fields.
                "dni_w_m2": _clean(dni_w_m2),
                "dni_note": body.dni.describe(),
            }

        job = JOBS.start(n, work, label=f"field trace, {n} heliostats, {workers} workers")
        return JSONResponse(job.snapshot())

    @app.get("/api/field/trace/status/{job_id}")
    def field_trace_status(job_id: str) -> JSONResponse:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return JSONResponse(job.snapshot())

    @app.post("/api/field/trace/cancel/{job_id}")
    def field_trace_cancel(job_id: str) -> JSONResponse:
        if not JOBS.cancel(job_id):
            raise HTTPException(status_code=409, detail="that job is not running")
        return JSONResponse({"cancelled": job_id})

    @app.get("/api/field/trace/result/{job_id}")
    def field_trace_result(job_id: str) -> JSONResponse:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        if job.state == "running":
            raise HTTPException(status_code=409, detail="still running")
        if job.state == "error":
            raise HTTPException(status_code=500, detail=job.error or "the run failed")
        if job.result is None:
            # state == "cancelled": see field_trace_start's work() -- a
            # cancelled field trace has no partial answer worth returning.
            raise HTTPException(status_code=409, detail="that run was cancelled -- no result to fetch")
        payload = dict(job.result)
        payload["state"] = job.state
        payload["elapsed_s"] = round(job.elapsed_s, 2)
        return JSONResponse(payload)

    @app.post("/api/field/trace/flux_fea.csv")
    def field_trace_flux_fea_csv(body: FieldTraceRequest) -> Response:
        """docs/ui-spec-v0.2.md §R: the field-level analogue of
        ``/api/trace/flux_fea.csv`` -- a live field instant's receiver flux
        map as a §D FEA CSV grid, so the Analysis tab's Traced instant
        source has a synchronous export to call (today this grid only ever
        reaches a CSV inside the day-sweep job's per-kept-step blobs). Same
        commented-header convention, via the same :func:`_flux_fea_csv`
        helper, as every other §D export.
        """
        phase = _field_trace_phase(body, return_secondary_flux=False)
        subject = _fea_subject_line(
            f"field of {len(phase['ids'])} heliostats",
            body.solar_az_deg,
            body.solar_el_deg,
            body.mode,
        )
        csv_bytes = _flux_fea_csv(
            phase["traced"]["flux"], phase["u_edges"], phase["v_edges"], phase["receiver"], subject
        )
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="field-flux-fea.csv"'},
        )

    @app.post("/api/field/trace/secondary_flux_fea.csv")
    def field_trace_secondary_flux_fea_csv(body: FieldTraceRequest) -> Response:
        """docs/ui-spec-v0.2.md §R / §C / §D: the field-level analogue of
        ``/api/trace/secondary_flux_fea.csv`` -- a live field instant's
        SECONDARY flux map as a §D FEA CSV grid (``x_m, y_m, z_m,
        flux_w_m2, absorbed_w_m2``). 404s for a layout with no secondary
        flux map (prime focus; any secondary with no single-valued (u, v)
        parameterization), same as the single-heliostat endpoint.
        """
        phase = _field_trace_phase(body, return_secondary_flux=True)
        traced = phase["traced"]
        if "secondary_flux" not in traced:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"optics={body.optics!r} has no secondary irradiance map "
                    "(prime focus has no secondary; only axicon/Cassegrain do)"
                ),
            )
        secondary_reflectance = getattr(phase["optics_params"], "secondary_reflectance", 0.90)
        subject = _fea_subject_line(
            f"field of {len(phase['ids'])} heliostats",
            body.solar_az_deg,
            body.solar_el_deg,
            body.mode,
        )
        csv_bytes = _secondary_flux_fea_csv(
            traced["secondary_flux"],
            traced["secondary_u_edges"],
            traced["secondary_v_edges"],
            phase["secondary"],
            secondary_reflectance,
            subject,
        )
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="field-secondary-flux-fea.csv"'},
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

        ``miss`` carries two independent things under one key: docs/ui-spec.md
        2.3's amber "warning" tier (``needed_aperture_radius_mm``,
        ``aperture_miss_ids``, ``total_miss_ids`` from
        :func:`~heliostat.web.scene.field_miss_detection` -- ``null`` for
        prime focus, which has no secondary to miss at all) and 2.1's
        unconditional "rays that miss the optics ... draw dashed red rather
        than disappearing" (``rays``, the dropped-corner-ray polylines
        :func:`~heliostat.web.scene.build_geometry_scene` collects from the
        same strided sources as its own ``rays``). The second one is NOT
        gated on the first: a shrunk prime-focus cylinder/frustum drops
        corner rays the same way a too-small axicon aperture does, even
        though prime focus never gets an aperture-miss verdict. ``miss`` as
        a whole is ``null`` only when there is truly nothing to report --
        no verdict AND no dropped rays -- or the sun is below the horizon
        (no solved orientation to build a chief ray from in the first
        place). Nothing here is adjusted automatically -- the geometry
        solve above is untouched; this is purely a report on it.
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
        elif miss_rays and not sun_below_horizon:
            # field_miss_detection has no verdict to give -- prime focus has
            # no secondary to miss at all, so its own docstring says "None
            # means no warning to report" unconditionally -- but the corner
            # rays it was never asked about can still have missed a shrunk
            # cylinder/frustum receiver. That is docs/ui-spec.md 2.1's plain
            # "rays that miss ... draw dashed red" contract, not 2.3's amber
            # aperture-miss tier, and it must not be dropped just because
            # the tier that usually rides alongside it has nothing to say.
            miss = {
                "needed_aperture_radius_mm": None,
                "aperture_miss_ids": [],
                "total_miss_ids": [],
                "rays": miss_rays,
            }
        scene["miss"] = miss
        scene["optics_resolved"] = optics_params.model_dump()
        return JSONResponse(scene)

    # -- library: designs, receivers, projects ------------------------------

    @app.get("/api/library/{collection}")
    def library_list(collection: str) -> JSONResponse:
        _require_known_collection(collection)
        entries = [{"name": name, "builtin": True} for name in _BUILTIN_LIBRARY[collection]]
        entries += [
            {
                "name": e["name"],
                "builtin": False,
                "saved_at": e["saved_at"],
                "size_bytes": e["size_bytes"],
            }
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
            # docs/ui-spec-v0.2.md §P's binding provenance requirement: a
            # built-in project's citation/reconstruction metadata rides
            # alongside its document (never inside it -- ProjectDocument
            # forbids extra fields, same as every other library document),
            # so the client can badge the card and stamp the citation
            # without a second request. `None` for every collection/name
            # with no provenance entry (designs, receivers, and any project
            # this rider didn't add) -- not an error, just "nothing to show".
            return JSONResponse(
                {
                    "name": name,
                    "builtin": True,
                    "document": _BUILTIN_LIBRARY[collection][name],
                    "provenance": BUILTIN_PROJECT_PROVENANCE.get(name) if collection == "projects" else None,
                }
            )
        try:
            payload = load_entry(collection, name)
        except LibraryError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # `provenance` always present (None here) so a client can read
        # `data.provenance` uniformly rather than special-casing "the key
        # might be missing" -- only a built-in project ever has one.
        return JSONResponse({**payload, "builtin": False, "provenance": None})

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
