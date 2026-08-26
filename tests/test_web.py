"""HTTP-level gate for the local web GUI (``heliostat.web``).

Skipped entirely when the ``web`` extra is not installed -- these are the
only tests in the suite that need FastAPI, and the rest of the package must
stay importable/testable without it.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import json
import math
import time
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from heliostat import __version__, energy  # noqa: E402
from heliostat.dni import ClearSkyDNI  # noqa: E402
from heliostat.geometry.design import _petal_at_angle, flower, grid_facets  # noqa: E402
from heliostat.geometry.heliostat import zernike_sag_and_slopes  # noqa: E402
from heliostat.geometry.secondary import solve_cassegrain_relay  # noqa: E402
from heliostat.solar import build_time_grid  # noqa: E402
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM  # noqa: E402
from heliostat.web import app as app_module  # noqa: E402
from heliostat.web.app import (  # noqa: E402
    AXICON_APERTURE_RADIUS_MM,
    AXICON_APEX_HEIGHT_MM,
    AXICON_HALF_ANGLE_DEG,
    AXICON_RECEIVER_Z_MM,
    CASSEGRAIN_APERTURE_RADIUS_MM,
    CASSEGRAIN_CONIC,
    CASSEGRAIN_FOCUS_HEIGHT_MM,
    CASSEGRAIN_RECEIVER_Z_MM,
    CASSEGRAIN_VERTEX_RADIUS_MM,
    CASSEGRAIN_VERTEX_Z_MM,
    FERMAT_A_M,
    FERMAT_B,
    FLUX_GRID,
    MAX_DAY_FLUX_MAPS,
    MAX_FIELD_HELIOSTATS,
    PRIME_FOCUS_CYLINDER_HEIGHT_MM,
    PRIME_FOCUS_CYLINDER_RADIUS_MM,
    PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM,
    PRIME_FOCUS_FRUSTUM_HEIGHT_MM,
    PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM,
    PRIME_FOCUS_HEIGHT_MM,
    RADIAL_STAGGER_BAND_COUNTS,
    RADIAL_STAGGER_BAND_RING_COUNTS,
    RADIAL_STAGGER_RING_RADII_M,
    WINDOW_MM,
    DayTraceRequest,
    FermatLayout,
    FlowerParams,
    RadialStaggeredLayout,
    YearTraceRequest,
    _aperture_metrics,
    _build_trace_design,
    _day_flux_step_indices,
    _day_timesteps,
    _field_geometry,
    _geometry_for,
    _slant_range_mm,
    _solve_for,
    _year_energy_cfg,
    _year_trace_dates,
    create_app,
    resolve_optics_params,
)
from heliostat.web.scene import (  # noqa: E402
    FIELD_SILHOUETTE_VERTICES,
    MAX_SCENE_RAYS,
    _outline_sample_points,
    build_field_scene,
    decimate_outline,
    radial_outline,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

RECT_DESIGN = {"type": "rect", "width_mm": 5000, "height_mm": 3000}
GRID_DESIGN = {
    "type": "grid",
    "n_u": 2,
    "n_v": 2,
    "facet_w_mm": 1200,
    "facet_h_mm": 1000,
    "gap_mm": 20,
}
FLOWER_DESIGN = {
    "type": "flower",
    "n_petals": 5,
    "petal_length_mm": 2000,
    "petal_width_mm": 900,
    "hub_radius_mm": 200,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _trace_payload(design, mode="ultra_fast", optics="prime_focus", solar_el_deg=45.0):
    return {
        "design": design,
        "mode": mode,
        "optics": optics,
        "solar_az_deg": 180.0,
        "solar_el_deg": solar_el_deg,
    }


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"version": __version__}


def test_index_serves_the_workspace(client):
    """`/` is the workspace shell, not the previous single-file UI."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert 'id="app"' in resp.text
    assert "/static/js/main.js" in resp.text


def test_legacy_route_serves_the_previous_ui(client):
    """The previous single-file UI stays reachable at /legacy for one
    release. Compared line by line so a CRLF checkout does not matter."""
    resp = client.get("/legacy")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert resp.text.splitlines() == client.get("/static/index.html").text.splitlines()


@pytest.mark.parametrize("design", [RECT_DESIGN, GRID_DESIGN, FLOWER_DESIGN])
def test_design_preview_returns_png(client, design):
    resp = client.post("/api/design/preview", json={"design": design})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == PNG_MAGIC


def test_trace_ultra_fast_plausible(client):
    resp = client.post("/api/trace", json=_trace_payload(FLOWER_DESIGN))
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "ultra_fast"
    assert 1000 < data["power_w"] < 40000
    assert data["incident_power_w"] is not None
    assert isinstance(data["rms_radius_mm"], float)
    assert len(data["centroid_mm"]) == 2
    assert isinstance(data["counters"], dict) and data["counters"]
    assert data["elapsed_ms"] > 0
    assert len(data["aim_point_mm"]) == 3
    assert data["slant_range_m"] > 0
    png_bytes = base64.b64decode(data["flux_png"])
    assert png_bytes[:8] == PNG_MAGIC


def test_trace_rect_ultra_fast_is_concentrated(client):
    """Real aiming/focusing (heliostat.geometry.aiming), not the old "demo
    aiming" flat mirror: the default 5000x3000 rect at the default
    heliostat position must now trace a concentrated spot instead of a
    mirror-sized wash. Values are from an actual run of this endpoint:
    rms_radius_mm ~505, power_w ~8226 (was rms ~1400+, the unfocused
    mirror's own half-diagonal, before this fix)."""
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN))
    assert resp.status_code == 200
    data = resp.json()
    assert data["rms_radius_mm"] < 800
    assert data["power_w"] > 5000


def test_trace_flower_auto_focus_concentrates_spot(client):
    """Blank cant_focal_mm on a twisting grid/flower design auto-focuses
    (canted at the heliostat's own solved slant range, and figured with the
    solve's own astigmatism) against a genuinely flat, unfocused mirror
    (surface="flat" with cant_focal_mm=0 too, so neither curvature nor aim
    is doing anything) -- measured ~538 mm vs ~1415 mm at the default
    heliostat position/sun; the 0.5 factor below leaves comfortable margin.
    cant_focal_mm=0 alone is not this comparison's "flat" any more: a
    twisting design's astigmatic figure does not depend on canting, so an
    uncanted twisting mirror still carries it (see _build_trace_design)."""
    auto = client.post("/api/trace", json=_trace_payload(FLOWER_DESIGN)).json()
    flat = client.post(
        "/api/trace",
        json=_trace_payload({**FLOWER_DESIGN, "surface": "flat", "cant_focal_mm": 0}),
    ).json()
    assert auto["rms_radius_mm"] < 0.5 * flat["rms_radius_mm"]


def test_trace_fast_accurate_plausible(client):
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, mode="fast_accurate"))
    assert resp.status_code == 200
    data = resp.json()
    assert 1000 < data["power_w"] < 40000


def test_trace_monte_carlo(client):
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, mode="monte_carlo"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "monte_carlo"
    assert data["incident_power_w"] is None
    assert 1000 < data["power_w"] < 40000
    png_bytes = base64.b64decode(data["flux_png"])
    assert png_bytes[:8] == PNG_MAGIC


def test_trace_tower_reflector_optics(client):
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, optics="axicon"))
    assert resp.status_code == 200
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, optics="cassegrain"))
    assert resp.status_code == 200


def test_trace_flux_grid_is_opt_in(client):
    """`flux_grid` (spec §M.3's raw texture data for the 3D receiver drape)
    is absent unless a request asks for it -- every existing caller of
    /api/trace that doesn't know about it, this file's own payloads
    included, pays nothing extra for it."""
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN))
    assert resp.status_code == 200
    assert resp.json().get("flux_grid") is None


def test_trace_flux_grid_shape_and_extent(client):
    payload = _trace_payload(RECT_DESIGN)
    payload["include_flux_grid"] = True
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    grid = data["flux_grid"]
    assert grid is not None
    assert grid["unit"] == "kW/m2"
    assert grid["n_u"] * grid["n_v"] == len(grid["values"])
    # Downsampled from FLUX_GRID (128) -- see _flux_grid_payload -- never
    # coarser than requested and never finer than the stored map itself.
    assert 0 < grid["n_u"] <= FLUX_GRID
    assert 0 < grid["n_v"] <= FLUX_GRID
    # prime_focus's default flat window (RECT_DESIGN, no optics_params
    # override): FlatWindowReceiver.uv_extent is +-WINDOW_MM on both axes.
    assert grid["u_min_mm"] == pytest.approx(-WINDOW_MM)
    assert grid["u_max_mm"] == pytest.approx(WINDOW_MM)
    assert grid["v_min_mm"] == pytest.approx(-WINDOW_MM)
    assert grid["v_max_mm"] == pytest.approx(WINDOW_MM)
    values = [v for v in grid["values"] if v is not None]
    assert values and all(v >= 0 for v in values)
    # A block-averaged grid can only ever read at or below the true peak
    # (peak_flux_kw_m2 is the single hottest FLUX_GRID bin; averaging
    # several of those into one coarser bin never raises the max).
    assert max(values) <= data["peak_flux_kw_m2"] + 1e-6


def test_trace_flux_grid_matches_a_curved_receiver_extent(client):
    """Same opt-in field, a cylinder receiver this time -- extent must come
    from CylinderReceiver.uv_extent (+-pi*radius, +-height/2), not the flat
    window's."""
    payload = _trace_payload(RECT_DESIGN)
    payload["optics_params"] = {"receiver_type": "cylinder"}
    payload["include_flux_grid"] = True
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 200
    grid = resp.json()["flux_grid"]
    assert grid["u_min_mm"] == pytest.approx(-math.pi * PRIME_FOCUS_CYLINDER_RADIUS_MM, rel=1e-3)
    assert grid["u_max_mm"] == pytest.approx(math.pi * PRIME_FOCUS_CYLINDER_RADIUS_MM, rel=1e-3)
    assert grid["v_min_mm"] == pytest.approx(-PRIME_FOCUS_CYLINDER_HEIGHT_MM / 2.0, rel=1e-3)
    assert grid["v_max_mm"] == pytest.approx(PRIME_FOCUS_CYLINDER_HEIGHT_MM / 2.0, rel=1e-3)


def test_field_trace_flux_grid_opt_in(client):
    """Same opt-in contract on the whole-field endpoint (a small, 4-heliostat
    field -- see CLAUDE.md's resource rule against tracing the real field for
    verification)."""
    without = client.post("/api/field/trace", json=_field_payload(layout={"type": "fermat", "n": 4})).json()
    assert without.get("flux_grid") is None

    payload = _field_payload(layout={"type": "fermat", "n": 4})
    payload["include_flux_grid"] = True
    with_grid = client.post("/api/field/trace", json=payload).json()
    grid = with_grid["flux_grid"]
    assert grid is not None
    assert grid["n_u"] * grid["n_v"] == len(grid["values"])


def test_trace_elevation_at_or_below_horizon_is_422(client):
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, solar_el_deg=0.0))
    assert resp.status_code == 422
    resp = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, solar_el_deg=-5.0))
    assert resp.status_code == 422


def test_bad_design_type_is_422(client):
    resp = client.post(
        "/api/design/preview", json={"design": {"type": "hexagon", "radius_mm": 100}}
    )
    assert resp.status_code == 422


def test_bad_design_type_trace_is_422(client):
    resp = client.post("/api/trace", json=_trace_payload({"type": "hexagon", "radius_mm": 100}))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3-D scene (heliostat.web.scene)
#
# Facet counts per design, as the builders actually build them: rect is the
# one-facet parity anchor; grid_facets is n_u * n_v; flower() makes ONE FACET
# PER PETAL and no hub facet at all -- hub_radius_mm pushes the petals
# outward along their own axes rather than adding a disc in the middle.

SCENE_DESIGNS = {
    "rect": (RECT_DESIGN, 1),
    "grid": (GRID_DESIGN, GRID_DESIGN["n_u"] * GRID_DESIGN["n_v"]),
    "flower": (FLOWER_DESIGN, FLOWER_DESIGN["n_petals"]),
}

# Every facet outline vertex must sit within the design's own bbox
# half-diagonal of the pivot. The slack covers sag (sub-mm) and the tiny
# out-of-plane excursion of a canted facet's corners.
HALF_DIAGONAL_SLACK_MM = 100.0


def _half_diagonal_mm(design):
    """The traced mirror's bbox half-diagonal for one request payload.

    The default 5000x3000 rectangle traces through the tracer's legacy
    single-mirror path, whose aperture is the MIRROR_HALF_* constants rather
    than a design; everything else is the design builders' own footprint,
    which does not depend on the figure or cant the endpoint resolves.
    """
    if design["type"] == "rect":
        return math.hypot(MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM)
    if design["type"] == "grid":
        return grid_facets(
            n_u=design["n_u"],
            n_v=design["n_v"],
            facet_w_mm=design["facet_w_mm"],
            facet_h_mm=design["facet_h_mm"],
            gap_mm=design["gap_mm"],
        ).half_diagonal_mm
    return flower(
        n_petals=design["n_petals"],
        petal_length_mm=design["petal_length_mm"],
        petal_width_mm=design["petal_width_mm"],
        hub_radius_mm=design["hub_radius_mm"],
    ).half_diagonal_mm


@pytest.mark.parametrize("design_key", sorted(SCENE_DESIGNS))
@pytest.mark.parametrize("optics", ["prime_focus", "axicon", "cassegrain"])
def test_scene_is_well_formed(client, design_key, optics):
    design, n_facets = SCENE_DESIGNS[design_key]
    resp = client.post("/api/trace", json=_trace_payload(design, optics=optics))
    assert resp.status_code == 200
    scene = resp.json()["scene"]

    # -- heliostat: one closed polygon per facet, all finite, all inside the
    # design's own footprint about the pivot.
    assert len(scene["heliostat"]) == n_facets
    limit = _half_diagonal_mm(design) + HALF_DIAGONAL_SLACK_MM
    helio_xy = np.array([0.0, -89609.0, 0.0])  # _trace_payload's default position
    for poly in scene["heliostat"]:
        assert len(poly) >= 3
        pts = np.asarray(poly, dtype=float)
        assert pts.shape[1] == 3
        assert np.isfinite(pts).all()
        assert np.linalg.norm(pts - helio_xy, axis=1).max() <= limit

    # -- secondary: null only for prime focus; a monotonic radial profile
    # spanning 0 -> the real surface's aperture radius otherwise.
    secondary, receiver = _geometry_for(optics)
    if optics == "prime_focus":
        assert scene["secondary"] is None
    else:
        assert scene["secondary"]["kind"] == optics
        profile = np.asarray(scene["secondary"]["profile"], dtype=float)
        assert profile.shape[1] == 2
        assert np.isfinite(profile).all()
        assert profile[0, 0] == 0.0
        assert profile[-1, 0] == pytest.approx(secondary.aperture_radius_mm)
        assert np.all(np.diff(profile[:, 0]) > 0)

    # -- receiver: the fixture window, verbatim. "kind"/"center_x_mm"/
    # "center_y_mm" are additive (docs/ui-spec.md 2.2's positionable,
    # per-shape receiver) -- every fixture receiver here is still the
    # on-axis flat window this app has always traced.
    assert scene["receiver"] == {
        "kind": "flat",
        "z_mm": receiver.z_mm,
        "half_u_mm": receiver.half_u_mm,
        "half_v_mm": receiver.half_v_mm,
        "facing": receiver.facing,
        "center_x_mm": receiver.center_x_mm,
        "center_y_mm": receiver.center_y_mm,
    }

    # -- sun: a unit vector, above the horizon at the requested elevation.
    sun = np.asarray(scene["sun"], dtype=float)
    assert sun.shape == (3,)
    assert np.linalg.norm(sun) == pytest.approx(1.0, abs=1e-5)
    assert sun[2] == pytest.approx(math.sin(math.radians(45.0)), abs=1e-5)


@pytest.mark.parametrize("design_key", sorted(SCENE_DESIGNS))
@pytest.mark.parametrize("optics", ["prime_focus", "axicon", "cassegrain"])
def test_scene_rays_land_on_the_receiver(client, design_key, optics):
    """Each ray is a real 4-vertex path whose ends are where they must be:
    the last vertex on the receiver plane inside the window, the second
    (the mirror hit) inside the mirror's own footprint."""
    design, _ = SCENE_DESIGNS[design_key]
    resp = client.post("/api/trace", json=_trace_payload(design, optics=optics))
    scene = resp.json()["scene"]
    _, receiver = _geometry_for(optics)

    rays = np.asarray(scene["rays"], dtype=float)
    assert 0 < len(scene["rays"]) <= MAX_SCENE_RAYS
    assert rays.shape[1:] == (4, 3)
    assert np.isfinite(rays).all()

    # Rounding to 0.1 mm can nudge a hit a hair past the window edge.
    tol = 0.1
    assert np.abs(rays[:, 3, 2] - receiver.z_mm).max() <= 1e-6
    assert np.abs(rays[:, 3, 0]).max() <= receiver.half_u_mm + tol
    assert np.abs(rays[:, 3, 1]).max() <= receiver.half_v_mm + tol

    limit = _half_diagonal_mm(design) + HALF_DIAGONAL_SLACK_MM
    helio = np.array([0.0, -89609.0, 0.0])
    assert np.linalg.norm(rays[:, 1, :] - helio, axis=1).max() <= limit


def test_scene_rays_source_labels_the_backend(client):
    """Monte Carlo reports its own traced paths; the cone backends carry no
    rays at all, so the scene runs a small side trace and says so."""
    mc = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, mode="monte_carlo")).json()
    assert mc["scene"]["rays_source"] == "trace"
    for mode in ("ultra_fast", "fast_accurate"):
        cone = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, mode=mode)).json()
        assert cone["scene"]["rays_source"] == "mc_sample"


def test_prime_focus_rays_have_no_secondary_bounce(client):
    """With no secondary the middle two path vertices coincide -- the ray
    goes mirror -> receiver, and the scene still sends all four points."""
    scene = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()["scene"]
    rays = np.asarray(scene["rays"], dtype=float)
    assert np.abs(rays[:, 1, :] - rays[:, 2, :]).max() == 0.0


@pytest.mark.parametrize("mode", ["ultra_fast", "monte_carlo"])
def test_scene_is_deterministic_and_does_not_disturb_the_trace(client, mode):
    """Two identical requests agree ray for ray -- and the numbers the trace
    reports are untouched by the scene being built alongside them."""
    payload = _trace_payload(FLOWER_DESIGN, mode=mode, optics="axicon")
    first = client.post("/api/trace", json=payload).json()
    second = client.post("/api/trace", json=payload).json()

    assert first["scene"]["rays"] == second["scene"]["rays"]
    assert first["scene"]["heliostat"] == second["scene"]["heliostat"]
    assert first["power_w"] == second["power_w"]
    assert first["counters"] == second["counters"]
    assert first["rms_radius_mm"] == second["rms_radius_mm"]
    assert first["centroid_mm"] == second["centroid_mm"]


# ---------------------------------------------------------------------------
# mirror surface selector + editable optics geometry
#
# Everything in this block guards one property first and features second: a
# request that says nothing about `surface` or `optics_params` must trace
# exactly what this app traced before either field existed.


# Captured from this endpoint BEFORE the surface/optics_params work landed,
# with the default request _trace_payload builds (rect 5000x3000, ultra_fast,
# prime focus, sun az 180 el 45, heliostat at (0, -89609)). These pin the
# LEGACY single-mirror path: a default-size twisting rectangle is the one
# combination that still traces through design=None, and these are the
# numbers that path produced. A change here is a physics change, not a
# refactor -- do not re-base without understanding why it moved.
# Power was pinned at 8225.974283127302 until a deposit that could exceed the
# power its own sample carried was fixed: that value sat ABOVE the incident
# power pinned two lines down, i.e. the default trace collected more than
# arrived, by 15 parts per million. It now lands exactly on the incident
# power, which is the physical bound for a spot that all falls on the
# receiver.
PIN_DEFAULT_RECT_POWER_W = 8225.854187898512
PIN_DEFAULT_RECT_RMS_MM = 505.26411781070186  # moved 3e-10 relative by the same fix
PIN_DEFAULT_RECT_INCIDENT_W = 8225.854187898514
PIN_DEFAULT_RECT_SLANT_M = 96.32411487265273

# Fields whose value must be identical between two requests that describe the
# same trace. `elapsed_ms` is wall clock, and `flux_png` carries it in the
# plot title, so both differ between two runs of *identical* code -- they are
# the only response fields excluded here.
COMPARABLE_FIELDS = (
    "power_w",
    "incident_power_w",
    "rms_radius_mm",
    "centroid_mm",
    "counters",
    "mode",
    "aim_point_mm",
    "slant_range_m",
    "scene",
)


def _comparable(data):
    return {k: data[k] for k in COMPARABLE_FIELDS}


def test_default_rect_matches_the_pinned_legacy_path(client):
    """The default request still lands on the legacy single-mirror path and
    still produces the numbers it produced before `surface` existed."""
    data = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    assert data["power_w"] == pytest.approx(PIN_DEFAULT_RECT_POWER_W, rel=1e-12)
    assert data["rms_radius_mm"] == pytest.approx(PIN_DEFAULT_RECT_RMS_MM, rel=1e-12)
    assert data["incident_power_w"] == pytest.approx(PIN_DEFAULT_RECT_INCIDENT_W, rel=1e-12)
    assert data["slant_range_m"] == pytest.approx(PIN_DEFAULT_RECT_SLANT_M, rel=1e-12)


@pytest.mark.parametrize(
    ("design", "optics", "explicit_optics"),
    [
        (
            RECT_DESIGN,
            "prime_focus",
            {
                "focus_height_mm": PRIME_FOCUS_HEIGHT_MM,
                "window_half_u_mm": WINDOW_MM,
                "window_half_v_mm": WINDOW_MM,
                # docs/ui-spec.md 2.2's receiver-type/offset/position fields --
                # additive, so their defaults are today's flat, on-axis, no-
                # offset receiver.
                "receiver_type": "flat",
                "receiver_center_x_mm": 0.0,
                "receiver_center_y_mm": 0.0,
                "aperture_to_receiver_mm": 0.0,
                "cylinder_radius_mm": PRIME_FOCUS_CYLINDER_RADIUS_MM,
                "cylinder_height_mm": PRIME_FOCUS_CYLINDER_HEIGHT_MM,
                "frustum_top_radius_mm": PRIME_FOCUS_FRUSTUM_TOP_RADIUS_MM,
                "frustum_bottom_radius_mm": PRIME_FOCUS_FRUSTUM_BOTTOM_RADIUS_MM,
                "frustum_height_mm": PRIME_FOCUS_FRUSTUM_HEIGHT_MM,
            },
        ),
        (
            FLOWER_DESIGN,
            "axicon",
            {
                "apex_height_mm": AXICON_APEX_HEIGHT_MM,
                "half_angle_deg": AXICON_HALF_ANGLE_DEG,
                "aperture_radius_mm": AXICON_APERTURE_RADIUS_MM,
                "receiver_z_mm": AXICON_RECEIVER_Z_MM,
                "window_half_u_mm": WINDOW_MM,
                "window_half_v_mm": WINDOW_MM,
            },
        ),
        (
            GRID_DESIGN,
            "cassegrain",
            {
                "vertex_z_mm": CASSEGRAIN_VERTEX_Z_MM,
                "focus_height_mm": CASSEGRAIN_FOCUS_HEIGHT_MM,
                "receiver_z_mm": CASSEGRAIN_RECEIVER_Z_MM,
                "aperture_radius_mm": CASSEGRAIN_APERTURE_RADIUS_MM,
                "window_half_u_mm": WINDOW_MM,
                "window_half_v_mm": WINDOW_MM,
            },
        ),
    ],
)
def test_absent_surface_and_optics_params_equal_explicit_defaults(
    client, design, optics, explicit_optics
):
    """Saying nothing is the same as saying the defaults out loud -- every
    comparable response field, including the whole 3-D scene."""
    bare = client.post("/api/trace", json=_trace_payload(design, optics=optics)).json()
    spelled_out = _trace_payload({**design, "surface": "twisting"}, optics=optics)
    spelled_out["optics_params"] = explicit_optics
    explicit = client.post("/api/trace", json=spelled_out).json()

    assert _comparable(bare) == _comparable(explicit)
    assert bare["optics_resolved"] == explicit["optics_resolved"] == explicit_optics


@pytest.mark.parametrize("optics", ["prime_focus", "axicon", "cassegrain"])
def test_optics_resolved_echoes_the_defaults(client, optics):
    data = client.post("/api/trace", json=_trace_payload(RECT_DESIGN, optics=optics)).json()
    resolved = data["optics_resolved"]
    assert resolved["window_half_u_mm"] == WINDOW_MM
    assert resolved["window_half_v_mm"] == WINDOW_MM
    if optics == "prime_focus":
        assert resolved["focus_height_mm"] == PRIME_FOCUS_HEIGHT_MM
        assert "apex_height_mm" not in resolved
    elif optics == "axicon":
        assert resolved["apex_height_mm"] == AXICON_APEX_HEIGHT_MM
        assert resolved["half_angle_deg"] == AXICON_HALF_ANGLE_DEG
        assert resolved["aperture_radius_mm"] == AXICON_APERTURE_RADIUS_MM
        assert resolved["receiver_z_mm"] == AXICON_RECEIVER_Z_MM
    else:
        # The three heights that define the relay, plus its aperture. The
        # vertex radius and conic are absent because they are solved from
        # those heights rather than set -- see solve_cassegrain_relay.
        assert set(resolved) == {
            "vertex_z_mm",
            "focus_height_mm",
            "receiver_z_mm",
            "aperture_radius_mm",
            "window_half_u_mm",
            "window_half_v_mm",
        }
        assert resolved["vertex_z_mm"] == CASSEGRAIN_VERTEX_Z_MM
        assert resolved["focus_height_mm"] == CASSEGRAIN_FOCUS_HEIGHT_MM
        assert resolved["receiver_z_mm"] == CASSEGRAIN_RECEIVER_Z_MM


def test_flat_rect_washes_and_leaves_the_legacy_path(client):
    """surface="flat" on the default-size rectangle must NOT take the legacy
    path (which hard-codes the astigmatic figure): with no figure at all the
    spot is a mirror-sized wash, several times the focused rms, and the cone
    backend still reports real power through it."""
    twisting = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    flat = client.post("/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "flat"})).json()
    assert flat["rms_radius_mm"] > 3.0 * twisting["rms_radius_mm"]
    assert flat["power_w"] > 0


def test_spherical_rect_is_not_the_legacy_path(client):
    """A default-size rectangle asking for a spherical figure is routed
    through the design path, so it cannot come back with the legacy path's
    astigmatic answer."""
    twisting = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    spherical = client.post(
        "/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "spherical"})
    ).json()
    assert math.isfinite(spherical["rms_radius_mm"])
    assert spherical["rms_radius_mm"] != twisting["rms_radius_mm"]
    assert spherical["power_w"] > 0


@pytest.mark.parametrize("design", [GRID_DESIGN, FLOWER_DESIGN])
def test_flat_surface_equals_explicit_facet_focal_zero(client, design):
    """`surface="flat"` and an explicit `facet_focal_mm=0` describe the same
    mirror -- Flat() either way -- whatever `surface` itself says, since
    facet_focal_mm=0 overrides curvature regardless of surface mode."""
    named_flat = client.post("/api/trace", json=_trace_payload({**design, "surface": "flat"})).json()
    zero_facet_focal = client.post(
        "/api/trace",
        json=_trace_payload({**design, "surface": "twisting", "facet_focal_mm": 0}),
    ).json()
    assert _comparable(named_flat) == _comparable(zero_facet_focal)


@pytest.mark.parametrize("design", [GRID_DESIGN, FLOWER_DESIGN])
def test_twisting_cant_zero_is_no_longer_flat(client, design):
    """Curvature and canting are independent: an uncanted twisting design
    (cant_focal_mm=0) still carries the solve's astigmatic figure, so it
    must trace tighter than a mirror that is actually flat (surface="flat"
    with cant_focal_mm=0 too)."""
    uncanted_twisting = client.post(
        "/api/trace", json=_trace_payload({**design, "cant_focal_mm": 0})
    ).json()
    truly_flat = client.post(
        "/api/trace",
        json=_trace_payload({**design, "surface": "flat", "cant_focal_mm": 0}),
    ).json()
    assert uncanted_twisting["rms_radius_mm"] < 0.97 * truly_flat["rms_radius_mm"]


@pytest.mark.parametrize("design", [GRID_DESIGN, FLOWER_DESIGN])
def test_flat_surface_still_cants(client, design):
    """Surface and cant are two axes: flat facets aimed at the slant range
    (blank cant_focal_mm) are not the same mirror as flat facets left
    parallel (cant_focal_mm=0)."""
    canted = client.post("/api/trace", json=_trace_payload({**design, "surface": "flat"})).json()
    parallel = client.post(
        "/api/trace",
        json=_trace_payload({**design, "surface": "flat", "cant_focal_mm": 0}),
    ).json()
    assert canted["rms_radius_mm"] != parallel["rms_radius_mm"]


@pytest.mark.parametrize("design", [GRID_DESIGN, FLOWER_DESIGN])
def test_spherical_without_a_focal_is_422(client, design):
    """cant_focal_mm=0 asks for no focal point at all, which leaves a
    spherical figure nothing to be figured against -- rejected rather than
    silently figured at some invented distance."""
    resp = client.post(
        "/api/trace",
        json=_trace_payload({**design, "surface": "spherical", "cant_focal_mm": 0}),
    )
    assert resp.status_code == 422
    assert "cant_focal_mm" in resp.json()["detail"]


def test_bad_surface_value_is_422(client):
    resp = client.post("/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "parabolic"}))
    assert resp.status_code == 422


def test_the_old_adaptive_spelling_is_rejected(client):
    """The solve-driven surface mode was renamed "adaptive" -> "twisting"
    before this package was ever published, and deliberately WITHOUT a
    compatibility alias. This pins that: the old spelling must fail loudly
    rather than being quietly accepted as the default, which is the failure
    mode that would let stale client code keep working until it didn't."""
    resp = client.post("/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "adaptive"}))
    assert resp.status_code == 422


def test_surface_is_accepted_but_ignored_by_the_preview(client):
    """The preview draws footprint only and has no sun to resolve a figure
    against, so every surface mode previews the same picture."""
    for surface in ("twisting", "spherical", "flat"):
        resp = client.post(
            "/api/design/preview", json={"design": {**FLOWER_DESIGN, "surface": surface}}
        )
        assert resp.status_code == 200
        assert resp.content[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# facet_focal_mm: facet curvature, independent of cant_focal_mm's aim
#
# A real manuscript-field heliostat (id 627 of field_645.csv, off-axis
# rather than the arbitrary default position) is reused below where the
# comparison needs realistic astigmatism -- its default-figure rectangle
# traces to rms_radius_mm ~416, peak ~14.8 kW/m^2 on axicon/fast_accurate.
_MANUSCRIPT_HELIOSTAT_X_MM = -88536.98462
_MANUSCRIPT_HELIOSTAT_Y_MM = -13822.15877
_MANUSCRIPT_GRID_DESIGN = {
    "type": "grid",
    "n_u": 4,
    "n_v": 3,
    "facet_w_mm": 1250,
    "facet_h_mm": 1000,
    "gap_mm": 0,
}


def _manuscript_payload(design):
    payload = _trace_payload(design, mode="fast_accurate", optics="axicon")
    payload["heliostat_x_mm"] = _MANUSCRIPT_HELIOSTAT_X_MM
    payload["heliostat_y_mm"] = _MANUSCRIPT_HELIOSTAT_Y_MM
    return payload


def test_facet_focal_mm_three_states_are_distinct(client):
    """None (backward compatible), 0 (explicitly flat) and a positive value
    (a fixed sphere) are three different mirrors."""
    absent = client.post("/api/trace", json=_trace_payload(GRID_DESIGN)).json()
    zero = client.post(
        "/api/trace", json=_trace_payload({**GRID_DESIGN, "facet_focal_mm": 0})
    ).json()
    positive = client.post(
        "/api/trace", json=_trace_payload({**GRID_DESIGN, "facet_focal_mm": 90000})
    ).json()
    for data in (absent, zero, positive):
        assert data["power_w"] > 0
        assert math.isfinite(data["rms_radius_mm"])
    assert absent["rms_radius_mm"] != zero["rms_radius_mm"]
    assert absent["rms_radius_mm"] != positive["rms_radius_mm"]
    assert zero["rms_radius_mm"] != positive["rms_radius_mm"]


@pytest.mark.parametrize("extra", [{}, {"surface": "spherical"}, {"surface": "flat"}])
def test_facet_focal_mm_absent_matches_explicit_none(client, extra):
    """Leaving facet_focal_mm out and sending it as an explicit null must
    trace identically -- the field's whole backward-compatibility contract
    reduces to this at the wire level; pydantic gives both the same `None`."""
    without = client.post("/api/trace", json=_trace_payload({**GRID_DESIGN, **extra})).json()
    with_none = client.post(
        "/api/trace", json=_trace_payload({**GRID_DESIGN, **extra, "facet_focal_mm": None})
    ).json()
    assert _comparable(without) == _comparable(with_none)


def test_weakly_focusing_flat_beats_true_flat_but_not_slant_focus(client):
    """surface="flat" with a long facet_focal_mm is a real, buildable
    middle ground: gently curved facets, materially tighter than a truly
    flat mirror but looser than each facet auto-focused at its own slant
    range."""
    truly_flat = client.post(
        "/api/trace", json=_manuscript_payload({**_MANUSCRIPT_GRID_DESIGN, "surface": "flat"})
    ).json()
    weakly_focusing = client.post(
        "/api/trace",
        json=_manuscript_payload(
            {**_MANUSCRIPT_GRID_DESIGN, "surface": "flat", "facet_focal_mm": 200000}
        ),
    ).json()
    slant_focused = client.post(
        "/api/trace", json=_manuscript_payload({**_MANUSCRIPT_GRID_DESIGN, "surface": "spherical"})
    ).json()
    assert weakly_focusing["rms_radius_mm"] < 0.9 * truly_flat["rms_radius_mm"]
    assert weakly_focusing["rms_radius_mm"] > slant_focused["rms_radius_mm"]


def test_twisting_grid_beats_the_old_spherical_facet_grid(client):
    """A twisting grid now carries astigmatism rather than spherical facets
    auto-focused at slant range, so it traces tighter and brighter than the
    old default did -- surface="spherical" with blank cant_focal_mm is
    exactly the figure a twisting grid used to build."""
    old_style = client.post(
        "/api/trace", json=_manuscript_payload({**_MANUSCRIPT_GRID_DESIGN, "surface": "spherical"})
    ).json()
    twisting = client.post("/api/trace", json=_manuscript_payload(_MANUSCRIPT_GRID_DESIGN)).json()
    assert twisting["rms_radius_mm"] < old_style["rms_radius_mm"]
    assert twisting["peak_flux_kw_m2"] > old_style["peak_flux_kw_m2"]


def test_cant_focal_mm_still_controls_aim_alone(client):
    """With facet_focal_mm pinning curvature, cant_focal_mm still moves
    where the field aims -- it is aim, not figure, so the delivered power
    and spot still change with it."""
    fixed_curvature = {**GRID_DESIGN, "facet_focal_mm": 90000}
    near = client.post(
        "/api/trace", json=_trace_payload({**fixed_curvature, "cant_focal_mm": 95000})
    ).json()
    far = client.post(
        "/api/trace", json=_trace_payload({**fixed_curvature, "cant_focal_mm": 200000})
    ).json()
    assert near["power_w"] != far["power_w"]
    assert near["rms_radius_mm"] != far["rms_radius_mm"]


# ---------------------------------------------------------------------------
# optical errors: slope_error_mrad, specularity_mrad, reflectance
#
# All three default to zero-error/perfect-reflector, so every test above this
# block -- including the pinned legacy numbers -- is a standing proof the
# defaults changed nothing.


def test_reflectance_scales_power_and_flux_not_incident_power(client):
    """reflectance is applied once, after the bounce: power and peak flux
    scale by it exactly; incident power (measured before the bounce) does
    not move at all. The default request (no `reflectance`) still lands on
    the legacy design=None path, so this also proves reflectance reaches
    that path."""
    default = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    dimmed = client.post(
        "/api/trace", json=_trace_payload({**RECT_DESIGN, "reflectance": 0.9})
    ).json()
    assert dimmed["power_w"] == pytest.approx(0.9 * default["power_w"], rel=1e-9)
    assert dimmed["peak_flux_kw_m2"] == pytest.approx(0.9 * default["peak_flux_kw_m2"], rel=1e-9)
    assert dimmed["incident_power_w"] == pytest.approx(default["incident_power_w"], rel=1e-12)


def test_mc_slope_error_grows_the_spot(client):
    """A per-ray surface-normal tilt spreads the Monte Carlo spot -- same
    seed path (mc_seed defaults to 1), so the only thing that can move the
    number is the perturbation itself."""
    payload = _trace_payload(RECT_DESIGN, mode="monte_carlo")
    payload["n_rays"] = 5000
    base = client.post("/api/trace", json=payload).json()
    blurred = client.post(
        "/api/trace",
        json={**payload, "design": {**RECT_DESIGN, "slope_error_mrad": 3.0}},
    ).json()
    assert blurred["rms_radius_mm"] > base["rms_radius_mm"]


def test_mc_specularity_grows_the_spot(client):
    """Same direction as slope error, for the post-reflection scatter."""
    payload = _trace_payload(RECT_DESIGN, mode="monte_carlo")
    payload["n_rays"] = 5000
    base = client.post("/api/trace", json=payload).json()
    blurred = client.post(
        "/api/trace",
        json={**payload, "design": {**RECT_DESIGN, "specularity_mrad": 3.0}},
    ).json()
    assert blurred["rms_radius_mm"] > base["rms_radius_mm"]


def test_cone_fast_accurate_slope_error_grows_the_spot(client):
    """The cone backend broadens its kernel for the same slope error, with
    the identical directional effect -- and, being deterministic, needs no
    seed to hold still."""
    payload = _trace_payload(RECT_DESIGN, mode="fast_accurate")
    base = client.post("/api/trace", json=payload).json()
    blurred = client.post(
        "/api/trace",
        json={**payload, "design": {**RECT_DESIGN, "slope_error_mrad": 3.0}},
    ).json()
    assert blurred["rms_radius_mm"] > base["rms_radius_mm"]


# ---------------------------------------------------------------------------
# custom polygon designs


def _hexagon_vertices_mm(circumradius_mm: float = 2500.0) -> list:
    """A regular hexagon roughly the manuscript rectangle's own size."""
    return [
        [
            circumradius_mm * math.cos(math.radians(60 * i)),
            circumradius_mm * math.sin(math.radians(60 * i)),
        ]
        for i in range(6)
    ]


CUSTOM_HEX_DESIGN = {"type": "custom", "vertices_mm": _hexagon_vertices_mm()}


def test_custom_design_preview_returns_png(client):
    resp = client.post("/api/design/preview", json={"design": CUSTOM_HEX_DESIGN})
    assert resp.status_code == 200
    assert resp.content[:8] == PNG_MAGIC


def test_custom_design_traces_and_delivers_power(client):
    resp = client.post("/api/trace", json=_trace_payload(CUSTOM_HEX_DESIGN))
    assert resp.status_code == 200
    data = resp.json()
    assert data["power_w"] > 0


def test_custom_design_too_few_vertices_is_422(client):
    resp = client.post(
        "/api/design/preview",
        json={"design": {"type": "custom", "vertices_mm": [[0.0, 0.0], [1000.0, 0.0]]}},
    )
    assert resp.status_code == 422


def test_custom_design_zero_area_is_422(client):
    """Three collinear points describe a line, not a facet."""
    resp = client.post(
        "/api/design/preview",
        json={
            "design": {
                "type": "custom",
                "vertices_mm": [[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]],
            }
        },
    )
    assert resp.status_code == 422


# -- optics_params ----------------------------------------------------------


def test_prime_focus_height_moves_aim_scene_and_rays(client):
    """The solve and the traced geometry must come from the same number: the
    aim point, the scene's receiver and the rays' own endpoints all land at
    the requested height."""
    payload = _trace_payload(RECT_DESIGN, mode="monte_carlo")
    payload["optics_params"] = {"focus_height_mm": 30000.0}
    data = client.post("/api/trace", json=payload).json()

    assert data["aim_point_mm"] == [0.0, 0.0, 30000.0]
    assert data["scene"]["receiver"]["z_mm"] == 30000.0
    assert data["optics_resolved"]["focus_height_mm"] == 30000.0
    rays = np.asarray(data["scene"]["rays"], dtype=float)
    assert len(rays) > 0
    assert np.abs(rays[:, 3, 2] - 30000.0).max() <= 1e-6


def test_prime_focus_window_size_reaches_the_receiver(client):
    payload = _trace_payload(RECT_DESIGN)
    payload["optics_params"] = {"window_half_u_mm": 900.0, "window_half_v_mm": 1500.0}
    scene = client.post("/api/trace", json=payload).json()["scene"]
    assert scene["receiver"]["half_u_mm"] == 900.0
    assert scene["receiver"]["half_v_mm"] == 1500.0


def test_axicon_geometry_is_per_request(client):
    """A lowered cone and a lowered ground receiver show up in the scene
    profile and the receiver together, the aim point moves with them, and
    the rays still land on the receiver."""
    apex, recv = 25000.0, 6000.0
    payload = _trace_payload(RECT_DESIGN, mode="monte_carlo", optics="axicon")
    payload["optics_params"] = {"apex_height_mm": apex, "receiver_z_mm": recv}
    data = client.post("/api/trace", json=payload).json()

    assert data["scene"]["receiver"]["z_mm"] == recv
    assert data["optics_resolved"]["apex_height_mm"] == apex
    assert data["optics_resolved"]["receiver_z_mm"] == recv

    profile = np.asarray(data["scene"]["secondary"]["profile"], dtype=float)
    # Outer rim: the cone flank's own equation at the aperture radius.
    assert profile[-1, 0] == pytest.approx(AXICON_APERTURE_RADIUS_MM)
    assert profile[-1, 1] == pytest.approx(
        apex + AXICON_APERTURE_RADIUS_MM * math.tan(math.radians(AXICON_HALF_ANGLE_DEG)),
        abs=0.2,
    )

    # The aim point is the receiver's image in the cone flank, so lowering
    # the cone must move it: 25000 + (25000 - 6000) * cos(2 * 20 deg).
    drop = apex - recv
    assert data["aim_point_mm"][2] == pytest.approx(
        apex + drop * math.cos(math.radians(2 * AXICON_HALF_ANGLE_DEG)), abs=1e-6
    )

    rays = np.asarray(data["scene"]["rays"], dtype=float)
    assert len(rays) > 0
    assert np.abs(rays[:, 3, 2] - recv).max() <= 1e-6


@pytest.mark.parametrize(
    ("aperture_radius_mm", "half_angle_deg"), [(9000.0, 20.0), (14000.0, 18.0), (11000.0, 26.0)]
)
def test_axicon_aperture_and_angle_shape_the_profile(client, aperture_radius_mm, half_angle_deg):
    """Both cone dials reach the drawn surface. No ray assertion here: a
    shallower or smaller cone can put this heliostat's chief ray outside the
    rim entirely (real geometry, not a failure), and the point of this test
    is that the picture follows the request either way."""
    payload = _trace_payload(RECT_DESIGN, optics="axicon")
    payload["optics_params"] = {
        "aperture_radius_mm": aperture_radius_mm,
        "half_angle_deg": half_angle_deg,
    }
    data = client.post("/api/trace", json=payload).json()
    profile = np.asarray(data["scene"]["secondary"]["profile"], dtype=float)
    assert profile[-1, 0] == pytest.approx(aperture_radius_mm)
    assert profile[-1, 1] == pytest.approx(
        AXICON_APEX_HEIGHT_MM + aperture_radius_mm * math.tan(math.radians(half_angle_deg)),
        abs=0.2,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"focus_height_mm": 0.0},
        {"focus_height_mm": -100.0},
        {"window_half_u_mm": 0.0},
    ],
)
def test_prime_focus_optics_params_sanity_is_422(client, params):
    payload = _trace_payload(RECT_DESIGN)
    payload["optics_params"] = params
    assert client.post("/api/trace", json=payload).status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"half_angle_deg": 0.0},
        {"half_angle_deg": 90.0},
        {"half_angle_deg": 120.0},
        {"apex_height_mm": -1.0},
        {"aperture_radius_mm": 0.0},
        # Receiver at or above the cone: the drop from the cone down to the
        # receiver would be zero or negative, and the solve would answer for
        # a tower that cannot exist.
        {"receiver_z_mm": 27000.0},
        {"receiver_z_mm": 30000.0},
    ],
)
def test_axicon_optics_params_sanity_is_422(client, params):
    payload = _trace_payload(RECT_DESIGN, optics="axicon")
    payload["optics_params"] = params
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"vertex_z_mm": 25000.0},
        {"focus_height_mm": 33000.0},
        {"receiver_z_mm": 5000.0},
        {"aperture_radius_mm": 16000.0},
    ],
)
def test_cassegrain_geometry_is_adjustable(client, params):
    """The relay is solved for whatever geometry is asked for.

    It used to be refused: the stored vertex radius and conic were solved
    for one focus/receiver pair, so moving either left constants describing
    a surface that no longer joined them. Solving the hyperboloid from the
    three heights removes that limit — see
    :func:`heliostat.geometry.secondary.solve_cassegrain_relay`.
    """
    payload = _trace_payload(RECT_DESIGN, optics="cassegrain")
    payload["optics_params"] = params
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 200
    resolved = resp.json()["optics_resolved"]
    for key, value in params.items():
        assert resolved[key] == pytest.approx(value)


@pytest.mark.parametrize("field", ["conic", "vertex_radius_mm", "apex_height_mm", "z_mm"])
def test_cassegrain_rejects_fields_it_does_not_have(client, field):
    """The relay surface is a *result* of the three heights, so its conic and
    vertex radius are not inputs; nor are the axicon's fields."""
    payload = _trace_payload(RECT_DESIGN, optics="cassegrain")
    payload["optics_params"] = {field: 12345.0}
    assert client.post("/api/trace", json=payload).status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"focus_height_mm": 20000.0},  # focus below the vertex
        {"receiver_z_mm": 30000.0},  # receiver above the vertex
        {"vertex_z_mm": 26994.0, "focus_height_mm": 34892.0, "receiver_z_mm": 20000.0},
    ],
)
def test_cassegrain_unphysical_geometry_is_422(client, params):
    """Three heights that describe no hyperboloid are refused, with the
    reason, rather than traced against a surface that cannot exist."""
    payload = _trace_payload(RECT_DESIGN, optics="cassegrain")
    payload["optics_params"] = params
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 422
    assert "vertex" in str(resp.json()["detail"]).lower()


def test_cassegrain_relay_solver_round_trips_the_fixture_relay():
    """The solver must reproduce the constants this package's fixtures were
    traced with, or every Cassegrain result would shift."""
    vertex_radius_mm, conic = solve_cassegrain_relay(
        CASSEGRAIN_VERTEX_Z_MM, CASSEGRAIN_FOCUS_HEIGHT_MM, CASSEGRAIN_RECEIVER_Z_MM
    )
    assert vertex_radius_mm == pytest.approx(CASSEGRAIN_VERTEX_RADIUS_MM, rel=1e-9)
    assert conic == pytest.approx(CASSEGRAIN_CONIC, rel=1e-9)


def test_cassegrain_window_size_is_allowed(client):
    payload = _trace_payload(RECT_DESIGN, optics="cassegrain")
    payload["optics_params"] = {"window_half_u_mm": 1200.0, "window_half_v_mm": 1200.0}
    data = client.post("/api/trace", json=payload).json()
    assert data["scene"]["receiver"]["half_u_mm"] == 1200.0
    # The relay's own receiver height is untouched by a window resize.
    assert data["scene"]["receiver"]["z_mm"] == 7000.0


def test_unknown_optics_param_is_422(client):
    payload = _trace_payload(RECT_DESIGN)
    payload["optics_params"] = {"tower_colour": 3.0}
    assert client.post("/api/trace", json=payload).status_code == 422


def test_index_carries_the_surface_and_inspector_controls(client):
    """The legacy UI is one hand-written file with no build step, so the only
    cheap guard that its controls actually shipped is that the markup is
    there."""
    text = client.get("/legacy").text
    for marker in (
        'id="surface-tabs"',
        'data-surface="twisting"',
        'data-surface="spherical"',
        'data-surface="flat"',
        'id="surface-caption"',
        'id="inspector"',
        'id="inspector-apply"',
    ):
        assert marker in text, marker


def test_geometry_for_accepts_the_optics_name_alone(client):
    """The one-argument form still returns the fixture geometry verbatim --
    the existing scene tests call it that way."""
    secondary, receiver = _geometry_for("axicon")
    assert secondary.apex_height_mm == AXICON_APEX_HEIGHT_MM
    assert receiver.z_mm == AXICON_RECEIVER_Z_MM
    assert receiver.half_u_mm == WINDOW_MM


# ---------------------------------------------------------------------------
# field trace (/api/field/trace)
#
# The property these guard first is that the field path and the single path
# are the same physics: a one-heliostat field must equal the single trace of
# that heliostat, and everything the field adds on top (mutual shading, the
# summed map) must only ever take power away, never invent it.


def _field_payload(design=RECT_DESIGN, layout=None, mode="ultra_fast", **kw):
    payload = _trace_payload(design, mode=mode)
    payload["layout"] = layout if layout is not None else {"type": "fermat", "n": 4}
    payload.update(kw)
    return payload


def test_field_of_one_equals_the_single_trace(client):
    """A one-heliostat field is a single trace with extra bookkeeping. If
    these ever diverge, the two endpoints have stopped sharing their
    physics -- which is the whole reason _trace_core exists."""
    field = client.post("/api/field/trace", json=_field_payload(layout={"type": "fermat", "n": 1}))
    assert field.status_code == 200
    fd = field.json()
    assert fd["n_heliostats"] == 1

    row = fd["heliostats"][0]
    single = client.post(
        "/api/trace",
        json={
            **_trace_payload(RECT_DESIGN),
            "heliostat_x_mm": row["x_mm"],
            "heliostat_y_mm": row["y_mm"],
        },
    ).json()

    # Nothing shades a lone heliostat, so the whole of it delivers.
    assert row["eta_shade"] == row["eta_block"] == row["eta"] == 1.0
    assert fd["power_w"] == pytest.approx(single["power_w"], rel=1e-12)
    assert fd["rms_radius_mm"] == pytest.approx(single["rms_radius_mm"], rel=1e-9)
    assert fd["centroid_mm"] == pytest.approx(single["centroid_mm"], abs=1e-9)
    assert fd["incident_power_w"] == pytest.approx(single["incident_power_w"], rel=1e-12)
    assert fd["scene"]["receiver"] == single["scene"]["receiver"]
    assert fd["scene"]["sun"] == single["scene"]["sun"]


# Twelve heliostats 8 m apart on a north-south line, with the sun low in the
# south: near enough, and lit at a shallow enough angle, that the northern
# half of the row stands in the southern half's shadow. Values are geometry,
# not tuning -- a 3 m-tall mirror at 8 deg elevation throws a 21 m shadow.
DENSE_ROW_XY_MM = [[0.0, -100000.0 + 8000.0 * i] for i in range(12)]
DENSE_ROW_SUN_EL_DEG = 8.0


def test_dense_field_shades_itself(client):
    """Twelve neighbours at low sun must deliver less than twelve isolated
    heliostats -- the shading is real, applied, and visible in the total."""
    payload = _field_payload(
        layout={"type": "positions", "xy_mm": DENSE_ROW_XY_MM},
        solar_el_deg=DENSE_ROW_SUN_EL_DEG,
    )
    data = client.post("/api/field/trace", json=payload).json()

    assert data["n_heliostats"] == 12
    rows = data["heliostats"]
    assert len(rows) == 12
    assert [r["id"] for r in rows] == list(range(12))
    for r, xy in zip(rows, DENSE_ROW_XY_MM):
        assert (r["x_mm"], r["y_mm"]) == tuple(xy)
        for key in ("eta_shade", "eta_block", "eta"):
            assert 0.0 < r[key] <= 1.0

    # At least one heliostat actually loses aperture, and the reported
    # summary agrees with the rows it summarises.
    etas = [r["eta"] for r in rows]
    assert min(etas) < 1.0
    assert data["eta_min"] == pytest.approx(min(etas))
    assert data["eta_max"] == pytest.approx(max(etas))

    isolated = 0.0
    for xy in DENSE_ROW_XY_MM:
        single = client.post(
            "/api/trace",
            json={
                **_trace_payload(RECT_DESIGN, solar_el_deg=DENSE_ROW_SUN_EL_DEG),
                "heliostat_x_mm": xy[0],
                "heliostat_y_mm": xy[1],
            },
        ).json()
        isolated += single["power_w"]
    assert data["power_w"] < isolated
    assert data["power_w"] == pytest.approx(sum(r["power_w"] for r in rows))


# Seven heliostats reproducing the default manuscript field's own worst-blocked
# heliostat (id 394 at az=180, el=20) and its actual occluders, at a fraction
# of the cost of tracing all 643: heliostat 0 here gets exactly the same
# eta_shade/eta_block the full field gives it, because these six neighbours
# are everything within its search radius.
BLOCKING_SCENE_XY_MM = [
    [0.0, 72885.0],
    [3000.3, 67762.6],
    [-3000.3, 67762.6],
    [3458.5, 78111.7],
    [-3458.5, 78111.7],
    [6441.6, 72599.8],
    [-6441.6, 72599.8],
]
BLOCKING_SCENE_SUN_EL_DEG = 20.0


def test_field_incident_power_charges_shading_only_not_blocking(client):
    """Incident power is measured before the bounce (see _trace_core's
    reflectance note): shading removes sunlight before it ever reaches a
    mirror, so it belongs there, but blocking removes the REFLECTED ray on
    its way to the receiver, after the mirror already saw full power, so it
    does not. Folding eta_union (shading AND blocking) into incident power
    makes intercept efficiency (collected/incident) read higher than the
    field's real blocking loss -- on this scene, exactly 100%, hiding a
    real ~19% blocking loss entirely.
    """
    payload = _field_payload(
        layout={"type": "positions", "xy_mm": BLOCKING_SCENE_XY_MM},
        solar_el_deg=BLOCKING_SCENE_SUN_EL_DEG,
    )
    data = client.post("/api/field/trace", json=payload).json()
    rows = data["heliostats"]

    # This scene must actually exercise blocking, distinct from shading, or
    # the two etas would coincide and the test would prove nothing.
    assert min(r["eta_block"] for r in rows) < 0.99
    assert min(r["eta_shade"] for r in rows) < 0.99

    # Each heliostat's own incident power, isolated from field occlusion
    # entirely, from the single-heliostat endpoint the field path shares
    # its physics with (_trace_core) -- scaled by shading alone is what the
    # field total should match.
    expected_incident_w = 0.0
    for r in rows:
        single = client.post(
            "/api/trace",
            json={
                **_trace_payload(RECT_DESIGN, solar_el_deg=BLOCKING_SCENE_SUN_EL_DEG),
                "heliostat_x_mm": r["x_mm"],
                "heliostat_y_mm": r["y_mm"],
            },
        ).json()
        expected_incident_w += single["incident_power_w"] * r["eta_shade"]

    assert data["incident_power_w"] == pytest.approx(expected_incident_w, rel=1e-9)

    # The bug this guards against: scaling incident power by eta_union
    # (rather than eta_shade) collapses it onto collected power whenever
    # blocking is present, since both would then carry the identical
    # factor -- reporting 100% intercept efficiency on a field that is
    # demonstrably losing power to blocking.
    assert data["incident_power_w"] > data["power_w"]


def test_field_trace_is_deterministic(client):
    """Two identical field requests agree ray for ray and watt for watt."""
    payload = _field_payload(FLOWER_DESIGN, layout={"type": "fermat", "n": 5}, optics="axicon")
    first = client.post("/api/field/trace", json=payload).json()
    second = client.post("/api/field/trace", json=payload).json()

    assert first["scene"]["rays"] == second["scene"]["rays"]
    assert first["scene"]["heliostat"] == second["scene"]["heliostat"]
    assert first["power_w"] == second["power_w"]
    assert first["rms_radius_mm"] == second["rms_radius_mm"]
    assert first["heliostats"] == second["heliostats"]
    assert first["counters"] == second["counters"]


def test_positions_layout_round_trips(client):
    """The positions a fermat field reports back are a layout in their own
    right: re-posting them traces the identical field."""
    generated = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "fermat", "n": 6})
    ).json()
    xy = [[r["x_mm"], r["y_mm"]] for r in generated["heliostats"]]

    replayed = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "positions", "xy_mm": xy})
    ).json()
    assert replayed["power_w"] == generated["power_w"]
    assert replayed["heliostats"] == generated["heliostats"]
    assert replayed["scene"]["heliostat"] == generated["scene"]["heliostat"]


def test_moving_one_heliostat_changes_only_that_row(client):
    """What the inspector's Apply does: re-post the field with one row
    edited. The moved heliostat lands where it was told and the others are
    untouched (they can still see a different neighbour, which is why only
    the position is asserted, not the power)."""
    base = [[0.0, -90000.0], [6000.0, -90000.0], [0.0, -84000.0]]
    moved = [list(p) for p in base]
    moved[1] = [40000.0, -70000.0]

    before = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "positions", "xy_mm": base})
    ).json()
    after = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "positions", "xy_mm": moved})
    ).json()

    assert [(r["x_mm"], r["y_mm"]) for r in after["heliostats"]] == [tuple(p) for p in moved]
    assert [r["id"] for r in after["heliostats"]] == [r["id"] for r in before["heliostats"]]
    assert after["heliostats"][1]["power_w"] != before["heliostats"][1]["power_w"]


def test_exclude_ids_drops_heliostats_without_renumbering(client):
    full = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "fermat", "n": 5})
    ).json()
    thinned = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 5}, exclude_ids=[1, 3]),
    ).json()

    assert thinned["n_heliostats"] == 3
    assert [r["id"] for r in thinned["heliostats"]] == [0, 2, 4]
    kept = {r["id"]: (r["x_mm"], r["y_mm"]) for r in full["heliostats"]}
    for r in thinned["heliostats"]:
        assert (r["x_mm"], r["y_mm"]) == kept[r["id"]]
    assert thinned["power_w"] < full["power_w"]


@pytest.mark.parametrize(
    "layout",
    [
        {"type": "fermat", "n": MAX_FIELD_HELIOSTATS + 1},
        {"type": "fermat", "n": 0},
        {"type": "positions", "xy_mm": [[0.0, 0.0]] * (MAX_FIELD_HELIOSTATS + 1)},
        {"type": "positions", "xy_mm": []},
        {"type": "positions", "xy_mm": [[0.0, -90000.0, 1.0]]},
        {"type": "spiral_of_doom", "n": 4},
    ],
)
def test_bad_field_layout_is_422(client, layout):
    assert client.post("/api/field/trace", json=_field_payload(layout=layout)).status_code == 422


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_position_is_422(client, bad):
    """Posted as a raw body: JSON has no NaN/Infinity literal, so an HTTP
    client that follows the spec cannot even send one -- but Python's own
    json module writes and reads all three, and the server must not trace a
    heliostat at infinity."""
    body = json.dumps(_field_payload(layout={"type": "positions", "xy_mm": [[0.0, -90000.0]]}))
    body = body.replace("-90000.0", bad)
    resp = client.post(
        "/api/field/trace", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422
    assert "finite" in str(resp.json()["detail"])


@pytest.mark.parametrize("exclude", [[9], [-1], [0, 1, 2, 3]])
def test_bad_exclude_ids_is_422(client, exclude):
    """Out of range, or dropping the whole field -- both are a request that
    cannot be answered, not a field of zero heliostats."""
    resp = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 4}, exclude_ids=exclude),
    )
    assert resp.status_code == 422


def test_field_trace_below_the_horizon_is_422(client):
    resp = client.post("/api/field/trace", json=_field_payload(solar_el_deg=0.0))
    assert resp.status_code == 422


def test_field_monte_carlo_traces(client):
    """The Monte Carlo backend sums histograms rather than hit lists, so it
    reports no incident power (as a single trace does not) and its combined
    spot still has to be finite and inside the window."""
    data = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 3}, mode="monte_carlo"),
    ).json()
    assert data["mode"] == "monte_carlo"
    assert data["incident_power_w"] is None
    assert data["power_w"] > 0
    assert 0 < data["rms_radius_mm"] < WINDOW_MM
    assert data["counters"]["emitted"] == 3 * 120_000


def test_field_timings_are_reported(client):
    data = client.post(
        "/api/field/trace", json=_field_payload(layout={"type": "fermat", "n": 4})
    ).json()
    t = data["timings_ms"]
    assert set(t) == {"solve", "occlusion", "trace", "scene"}
    assert all(v >= 0 for v in t.values())
    # elapsed_ms is the physics: solve + occlusion + trace, scene excluded.
    assert data["elapsed_ms"] == pytest.approx(t["solve"] + t["occlusion"] + t["trace"], rel=1e-6)


# -- field scene ------------------------------------------------------------


@pytest.mark.parametrize("design_key", sorted(SCENE_DESIGNS))
def test_field_scene_is_one_silhouette_per_heliostat(client, design_key):
    """Not one polygon per facet: at field scale a mirror is its outline."""
    design, _n_facets = SCENE_DESIGNS[design_key]
    data = client.post(
        "/api/field/trace", json=_field_payload(design, layout={"type": "fermat", "n": 7})
    ).json()
    scene = data["scene"]

    assert len(scene["heliostat"]) == 7
    assert len(scene["field"]["heliostats"]) == 7
    for poly in scene["heliostat"]:
        assert 3 <= len(poly) <= FIELD_SILHOUETTE_VERTICES
        pts = np.asarray(poly, dtype=float)
        assert pts.shape[1] == 3
        assert np.isfinite(pts).all()

    # Each polygon sits at its own heliostat, and the table agrees with the
    # per-heliostat rows the metrics were built from.
    triples = zip(scene["heliostat"], scene["field"]["heliostats"], data["heliostats"])
    for poly, entry, row in triples:
        assert entry["id"] == row["id"]
        assert entry["eta"] == pytest.approx(row["eta"], abs=1e-4)
        centre = np.asarray(poly, dtype=float).mean(axis=0)
        assert abs(centre[0] - row["x_mm"]) < _half_diagonal_mm(design)
        assert abs(centre[1] - row["y_mm"]) < _half_diagonal_mm(design)

    # Four chief rays per heliostat, from every heliostat -- not a dense
    # bundle from a stride of them, which read as "only these were traced".
    # Rays that miss the receiver are not drawn, so this is an upper bound.
    assert 0 < len(scene["rays"]) <= 4 * 7
    assert scene["rays_source"] == "corner_chief"
    assert scene["field"]["ray_sources"] == 7


def test_field_scene_payload_at_600_stays_small():
    """The whole point of drawing silhouettes instead of facets: a full field
    has to fit in a response the browser will actually parse. Built directly
    rather than over HTTP -- 600 traces is a minute of wall time and this
    asserts nothing about the trace."""
    xy = FermatLayout(n=MAX_FIELD_HELIOSTATS).positions_mm()
    params = resolve_optics_params("prime_focus", None)
    secondary, receiver = _geometry_for("prime_focus", params)
    # A flower: the worst case for this payload, since its silhouette is a
    # sampled 72-gon rather than four rectangle corners.
    design_params = FlowerParams(**FLOWER_DESIGN)

    heliostats = []
    design = None
    for i in range(xy.shape[0]):
        sol = _solve_for("prime_focus", float(xy[i, 0]), float(xy[i, 1]), 180.0, 45.0, params)
        design = _build_trace_design(
            design_params, sol, _slant_range_mm(sol, float(xy[i, 0]), float(xy[i, 1]))
        )
        heliostats.append(
            {
                "id": i,
                "x_mm": float(xy[i, 0]),
                "y_mm": float(xy[i, 1]),
                "rot_az_deg": sol.rot_az_deg,
                "rot_el_deg": sol.rot_el_deg,
                "c3": sol.c3,
                "c4": sol.c4,
                "c5": sol.c5,
                "design": design,
                "eta": 0.9,
            }
        )

    _region, outline, _hw, _hh = _field_geometry(design)
    scene = build_field_scene(heliostats, outline, 180.0, 45.0, secondary, receiver)

    assert len(scene["heliostat"]) == MAX_FIELD_HELIOSTATS
    assert {len(p) for p in scene["heliostat"]} == {FIELD_SILHOUETTE_VERTICES}
    assert scene["field"]["decimated"] is True
    assert len(json.dumps(scene)) < 1_500_000


def test_decimate_outline_keeps_original_vertices():
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    outline = np.column_stack([np.cos(ang), np.sin(ang)])
    thinned = decimate_outline(outline, 24)
    assert thinned.shape == (24, 2)
    assert np.isin(thinned, outline).all()
    # A rectangle is already short enough and comes back untouched.
    rect = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    assert decimate_outline(rect, 24) is rect


# -- the field honours the design panel -------------------------------------


def test_field_surface_selector_applies_to_every_heliostat(client):
    """Flat facets have no figure, so a flat field washes: its combined spot
    is far broader than the same field focused."""
    focused = client.post(
        "/api/field/trace", json=_field_payload(GRID_DESIGN, layout={"type": "fermat", "n": 5})
    ).json()
    flat = client.post(
        "/api/field/trace",
        json=_field_payload({**GRID_DESIGN, "surface": "flat"}, layout={"type": "fermat", "n": 5}),
    ).json()
    assert flat["rms_radius_mm"] > 2.0 * focused["rms_radius_mm"]


def test_field_optics_params_move_the_whole_tower(client):
    """One tower, every heliostat aimed at it: a moved focus height shows up
    in the scene's receiver and in what the field resolved."""
    payload = _field_payload(layout={"type": "fermat", "n": 4})
    payload["optics_params"] = {"focus_height_mm": 30000.0}
    data = client.post("/api/field/trace", json=payload).json()

    assert data["optics_resolved"]["focus_height_mm"] == 30000.0
    assert data["scene"]["receiver"]["z_mm"] == 30000.0
    rays = np.asarray(data["scene"]["rays"], dtype=float)
    assert len(rays) > 0
    assert np.abs(rays[:, 3, 2] - 30000.0).max() <= 1e-6


def test_field_spherical_without_a_focal_is_422(client):
    """The design panel's own errors reach the field endpoint unchanged."""
    resp = client.post(
        "/api/field/trace",
        json=_field_payload({**GRID_DESIGN, "surface": "spherical", "cant_focal_mm": 0}),
    )
    assert resp.status_code == 422
    assert "cant_focal_mm" in resp.json()["detail"]


def test_fermat_layout_matches_the_library_generator():
    """The endpoint's spiral is field_layouts' spiral at this module's
    documented defaults -- not a second copy of the recipe."""
    from heliostat.field_layouts import generate

    expected = generate("fermat", 40, a_m=FERMAT_A_M, b=FERMAT_B).xy_mm
    assert np.array_equal(FermatLayout(n=40).positions_mm(), expected)


# ---------------------------------------------------------------------------
# radial-staggered field layout (the classic DELSOL/Campo pattern,
# docs/ui-spec.md 2.2) -- 12 rings in 3 bands, whose defaults reproduce the
# field the app has always shipped as a fixed CSV.


def test_radial_stagger_default_is_643_positions_in_twelve_rings():
    xy = RadialStaggeredLayout().positions_mm()
    assert xy.shape == (643, 2)
    r_m = np.hypot(xy[:, 0], xy[:, 1]) / 1000.0
    distinct_radii = np.unique(np.round(r_m, 3))
    assert distinct_radii.size == 12
    assert np.allclose(sorted(distinct_radii), RADIAL_STAGGER_RING_RADII_M, atol=1e-3)


def test_radial_stagger_ring_membership_matches_band_counts():
    """Each ring holds exactly its band's heliostat count, uniformly spaced
    in azimuth at pitch = 360 / that count."""
    xy = RadialStaggeredLayout().positions_mm()
    r_mm = np.hypot(xy[:, 0], xy[:, 1])
    az_deg = np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360.0

    expected_counts = []
    for band_count, n_rings in zip(RADIAL_STAGGER_BAND_COUNTS, RADIAL_STAGGER_BAND_RING_COUNTS):
        expected_counts.extend([band_count] * n_rings)

    for radius_m, expected_n in zip(RADIAL_STAGGER_RING_RADII_M, expected_counts):
        ring_mask = np.isclose(r_mm / 1000.0, radius_m, atol=1e-3)
        assert int(ring_mask.sum()) == expected_n
        ring_az = np.sort(az_deg[ring_mask])
        pitch = 360.0 / expected_n
        spacing = np.diff(np.concatenate([ring_az, [ring_az[0] + 360.0]]))
        assert np.allclose(spacing, pitch, atol=1e-6)


def test_radial_stagger_restarts_the_half_pitch_offset_each_band():
    """The half-pitch stagger alternates ring to ring, and restarts (the
    first ring of a band always carries the offset) at each band's first
    ring rather than continuing across the boundary, since the pitch itself
    changes there."""
    xy = RadialStaggeredLayout().positions_mm()
    r_mm = np.hypot(xy[:, 0], xy[:, 1])
    az_deg = np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360.0

    ring_i = 0
    for band_count, n_rings in zip(RADIAL_STAGGER_BAND_COUNTS, RADIAL_STAGGER_BAND_RING_COUNTS):
        pitch = 360.0 / band_count
        for local_ring in range(n_rings):
            radius_m = RADIAL_STAGGER_RING_RADII_M[ring_i]
            ring_i += 1
            ring_mask = np.isclose(r_mm / 1000.0, radius_m, atol=1e-3)
            first_az = np.sort(az_deg[ring_mask])[0]
            expected_phase = pitch / 2.0 if local_ring % 2 == 0 else 0.0
            assert np.isclose(first_az % pitch, expected_phase % pitch, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band_counts": [32, 48], "band_ring_counts": [3, 4, 5]},
        {"band_counts": [32, 48, 71], "band_ring_counts": [3, 4], "ring_radii_m": [30.0] * 7},
        {"band_counts": [0, 48, 71]},
        {"band_ring_counts": [3, 0, 5]},
        {"band_counts": [32, 48, 71], "band_ring_counts": [3, 4, 5], "ring_radii_m": [0.0] * 12},
    ],
)
def test_radial_stagger_rejects_mismatched_or_nonpositive_shapes(kwargs):
    with pytest.raises(ValidationError):
        RadialStaggeredLayout(**kwargs)


def test_radial_stagger_over_the_trace_cap_is_422(client):
    """band_counts x band_ring_counts exceeding MAX_FIELD_HELIOSTATS is
    rejected the same way an oversized Fermat ``n`` is."""
    layout = {"type": "radial_stagger", "band_counts": [2000], "band_ring_counts": [1], "ring_radii_m": [50.0]}
    assert client.post("/api/field/trace", json=_field_payload(layout=layout)).status_code == 422


def test_radial_stagger_layout_via_field_trace(client):
    """The trace endpoint accepts the layout and traces one heliostat per
    generated position.

    A deliberately tiny field: tracing the 643-position default here costs
    minutes to prove only that the layout reaches the tracer, which six
    mirrors establish just as well. The default's own shape is covered by
    test_radial_stagger_matches_the_manuscript_field, which needs no trace.
    """
    layout = {"type": "radial_stagger", "band_counts": [6], "band_ring_counts": [1], "ring_radii_m": [45.0]}
    resp = client.post("/api/field/trace", json=_field_payload(layout=layout))
    assert resp.status_code == 200
    assert len(resp.json()["heliostats"]) == 6


def test_radial_stagger_matches_the_manuscript_field(client):
    """The regression guard: the default radial-staggered field must still
    BE the field the app has always shipped, not merely the same size.

    Nearest-neighbour match against the manuscript's own positions
    (/api/field/manuscript) -- established bounds: every generated position
    within 100 mm of a real one, and at least 630 of 643 within 0.1 mm (11
    manuscript rows carry a hand-rounded coordinate no formula reproduces
    exactly)."""
    manuscript_xy = np.asarray(client.get("/api/field/manuscript").json()["xy_mm"])
    generated_xy = RadialStaggeredLayout().positions_mm()
    assert generated_xy.shape[0] == manuscript_xy.shape[0] == 643

    # O(n^2) nearest-neighbour distance, fine at n=643.
    diff = generated_xy[:, None, :] - manuscript_xy[None, :, :]
    dist_mm = np.sqrt((diff**2).sum(axis=-1)).min(axis=1)

    assert dist_mm.max() <= 100.0
    assert int((dist_mm <= 0.1).sum()) >= 630


def test_index_carries_the_field_controls(client):
    """Markup-only guard on the legacy UI, same reasoning as the
    surface/inspector one."""
    text = client.get("/legacy").text
    for marker in (
        'id="trace-mode-tabs"',
        'data-tracemode="single"',
        'data-tracemode="field"',
        'id="field-n"',
        'id="field-legend"',
        'id="row-eta"',
        "/api/field/trace",
    ):
        assert marker in text, marker


def test_radial_outline_traces_a_petal_boundary():
    """The outline sampler returns points that are inside the region (the
    bisection keeps the interior bracket) and reach its bbox extents."""
    region = _petal_at_angle(2000.0, 900.0, 0.0)
    pts = radial_outline(region, n_directions=48)
    assert pts.shape == (48, 2)
    assert np.isfinite(pts).all()
    assert region.contains(pts[:, 0], pts[:, 1]).all()

    # ...and that they reach the shape's true extents. Not the region's
    # bbox(): a lens built as two overlapping discs reports the intersection
    # of the discs' boxes, which is far looser than the petal (v from -336
    # for a petal that starts at 0). The honest reference is the membership
    # mask itself, rasterised finely.
    u0, u1, v0, v1 = region.bbox()
    n = 600
    uu, vv = np.meshgrid(np.linspace(u0, u1, n), np.linspace(v0, v1, n))
    mask = np.asarray(region.contains(uu, vv), dtype=bool)
    cell = max((u1 - u0) / n, (v1 - v0) / n)
    assert pts[:, 0].min() <= uu[mask].min() + 2 * cell
    assert pts[:, 0].max() >= uu[mask].max() - 2 * cell
    assert pts[:, 1].min() <= vv[mask].min() + 2 * cell
    assert pts[:, 1].max() >= vv[mask].max() - 2 * cell


def test_outline_samples_avoid_the_shape_s_own_symmetry_axes():
    """Regression pin: evenly spaced outline indices land in the gaps.

    A 2 x 2 facet grid has its gaps along +-u and +-v. Sampling the outline
    at indices 0, N/4, N/2, 3N/4 puts every sample in a gap, on backing
    structure rather than on a mirror, and the heliostat draws no rays at
    all. The half-offset puts them on the diagonals instead.
    """
    angles = 2.0 * np.pi * np.arange(24) / 24.0
    outline = np.column_stack([np.cos(angles), np.sin(angles)]) * 1000.0

    points = _outline_sample_points(outline, 4)
    sampled = np.rad2deg(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    for axis_angle in (0.0, 90.0, 180.0, 270.0):
        assert np.min(np.abs(sampled - axis_angle)) > 10.0, (
            f"a sample landed on the {axis_angle} deg axis, where a gapped design has no material"
        )


@pytest.mark.parametrize("design_key", sorted(SCENE_DESIGNS))
def test_every_heliostat_contributes_rays_whatever_its_shape(client, design_key):
    """Four rays per heliostat for rect, grid and flower alike.

    The shaped designs are the ones that catch a sampling bug: a point that
    misses the material is silently dropped, so a broken sampler shows up as
    an empty sky rather than as an error.
    """
    design, _n_facets = SCENE_DESIGNS[design_key]
    data = client.post(
        "/api/field/trace", json=_field_payload(design, layout={"type": "fermat", "n": 5})
    ).json()
    assert len(data["scene"]["rays"]) == 4 * 5


def test_field_layout_radius_bounds_are_honoured(client):
    """Nearest/farthest heliostat are the field-design controls the GUI
    exposes; a ring far from the tower must actually be reachable."""
    data = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 20, "r_min_m": 40, "r_max_m": 90}),
    ).json()
    radii = [
        math.hypot(h["x_mm"], h["y_mm"]) / 1000.0 for h in data["scene"]["field"]["heliostats"]
    ]
    assert len(radii) == 20
    assert min(radii) >= 40.0
    assert max(radii) <= 90.0


def test_far_ring_needs_more_candidates_than_the_default_oversample(client):
    """Regression pin. ``generate`` draws n * oversample candidates in spiral
    order and only then filters, so a ring beyond the first few turns kept
    nothing at all: 20 heliostats between 40 and 90 m returned "only 0 of 20
    survived". The layout now sizes its own oversample by inverting the
    spiral's radius law, so this must keep working."""
    resp = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 20, "r_min_m": 40, "r_max_m": 90}),
    )
    assert resp.status_code == 200


def test_reversed_radius_bounds_are_rejected(client):
    resp = client.post(
        "/api/field/trace",
        json=_field_payload(layout={"type": "fermat", "n": 10, "r_min_m": 90, "r_max_m": 40}),
    )
    assert resp.status_code == 422
    assert "r_max_m" in json.dumps(resp.json())


# -- non-finite input (Infinity/NaN) is rejected everywhere a float is
# accepted --------------------------------------------------------------
#
# httpx's own JSON encoder refuses to serialize Infinity/NaN at all
# (`json=` raises client-side before a request is even sent), so these post
# raw bytes built with the stdlib json module, which -- like the JSON a real
# browser's `JSON.stringify` on a bad number, or a hand-written client, can
# produce -- happily emits the non-standard `Infinity`/`-Infinity`/`NaN`
# tokens.


def _post_raw_json(client, url, payload):
    return client.post(url, content=json.dumps(payload), headers={"content-type": "application/json"})


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_infinite_top_level_float_is_rejected(client, bad):
    """`gt`/`ge` alone do not catch this: inf compares fine against either,
    so an ordinary Field(gt=0)/Field(ge=0) constraint waves it through."""
    payload = _trace_payload(RECT_DESIGN)
    payload["heliostat_x_mm"] = bad
    resp = _post_raw_json(client, "/api/design/sag", payload)
    assert resp.status_code == 422


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_infinite_optics_params_float_is_rejected(client, bad):
    """receiver_center_x_mm/receiver_center_y_mm carry no gt/ge bound at
    all -- the field this most needs to cover, since there is no partial
    constraint to even accidentally catch some of it."""
    payload = _trace_payload(RECT_DESIGN)
    payload["optics_params"] = {"receiver_center_x_mm": bad}
    resp = _post_raw_json(client, "/api/design/sag", payload)
    assert resp.status_code == 422
    assert "receiver_center_x_mm" in json.dumps(resp.json())


def test_nan_solar_elevation_is_rejected_by_name(client):
    """solar_el_deg carries no gt/ge of its own -- only a `> 90` check that
    a bare NaN passes too (NaN comparisons are always false) -- so this is
    the one field where, unfixed, NaN used to reach _solve_for's physics
    and turn into a NaN flux map. Asserting on the message, not just the
    status code: an unvalidated NaN elsewhere in the response can *also*
    422 by accident, when Starlette's own JSONResponse (allow_nan=False)
    fails to encode it back out -- that is a different failure with a
    generic message, not this field being validated up front."""
    payload = _field_payload(layout={"type": "fermat", "n": 4})
    payload["solar_el_deg"] = float("nan")
    resp = _post_raw_json(client, "/api/field/trace", payload)
    assert resp.status_code == 422
    assert "solar_el_deg" in json.dumps(resp.json())
    assert "finite" in json.dumps(resp.json())


# ---------------------------------------------------------------------------
# field trace as a background job (/api/field/trace/start|status|cancel|
# result) -- same job-registry shape as /api/day/*, and the same physics as
# a synchronous /api/field/trace, so most of these compare the two rather
# than re-deriving expected numbers. Kept deliberately small (well under
# MAX_FIELD_HELIOSTATS): the point is the job/cancel/parallel machinery, not
# giving the tracer real work -- the 643-heliostat default field is not
# something a test should ever run.

# Keys that legitimately differ between two otherwise-identical traces:
# flux_png embeds a wall-clock "traced in Xms" caption, and elapsed_ms/
# timings_ms/workers/state/elapsed_s describe the run itself, not the field.
_VOLATILE_TRACE_KEYS = {"flux_png", "elapsed_ms", "timings_ms", "workers", "state", "elapsed_s"}


def _strip_volatile(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _VOLATILE_TRACE_KEYS}


def _run_field_trace_job(client, payload):
    started = client.post("/api/field/trace/start", json=payload)
    assert started.status_code == 200, started.json()
    job_id = started.json()["job_id"]
    status = None
    for _ in range(600):
        status = client.get(f"/api/field/trace/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.02)
    return job_id, status


def test_field_trace_job_matches_the_synchronous_endpoint(client):
    """The job endpoint is the same trace as /api/field/trace, just run in
    the background -- everything but the volatile timing/caption fields must
    come back identical."""
    payload = _field_payload(layout={"type": "fermat", "n": 8}, workers=1)
    sync = client.post("/api/field/trace", json=payload)
    assert sync.status_code == 200

    job_id, status = _run_field_trace_job(client, payload)
    assert status["state"] == "done"
    assert status["done"] == status["total"] == 8

    result = client.get(f"/api/field/trace/result/{job_id}")
    assert result.status_code == 200
    assert _strip_volatile(result.json()) == _strip_volatile(sync.json())


def test_field_trace_job_one_bad_heliostat_does_not_lose_the_run(client, monkeypatch):
    """A field trace is hundreds of heliostats and can run for minutes; one
    of them raising (a numerically awkward geometry, say) must cost that
    one heliostat, not every heliostat already traced. Before the fix, any
    exception out of a single heliostat's trace propagated past
    _trace_field_heliostats entirely, into JobRegistry's generic handler,
    landing the whole job on state="error" and discarding the rest."""
    calls = {"n": 0}
    original = app_module._trace_core

    def flaky_trace_core(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated solve blowup")
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "_trace_core", flaky_trace_core)

    payload = _field_payload(layout={"type": "fermat", "n": 6}, workers=1)
    job_id, status = _run_field_trace_job(client, payload)
    assert status["state"] == "done", status

    result = client.get(f"/api/field/trace/result/{job_id}").json()
    assert len(result["failed_heliostats"]) == 1
    assert "simulated solve blowup" in result["failed_heliostats"][0]["error"]
    assert len(result["heliostats"]) == 6
    failed_rows = [r for r in result["heliostats"] if r.get("failed")]
    assert len(failed_rows) == 1
    assert failed_rows[0]["power_w"] == 0.0
    # The other five heliostats still traced and still contributed power.
    assert result["power_w"] > 0.0


def test_field_trace_job_progress_reports_done_out_of_total(client):
    payload = _field_payload(layout={"type": "fermat", "n": 12}, workers=1)
    started = client.post("/api/field/trace/start", json=payload)
    snap = started.json()
    assert snap["total"] == 12
    assert snap["done"] == 0
    job_id, status = _run_field_trace_job(client, payload)
    assert status["done"] == status["total"] == 12


def test_field_trace_result_409s_while_running_and_404s_when_unknown(client):
    assert client.get("/api/field/trace/status/nosuchjob").status_code == 404
    assert client.get("/api/field/trace/result/nosuchjob").status_code == 404
    assert client.post("/api/field/trace/cancel/nosuchjob").status_code == 409

    # A job that has already finished cannot be cancelled -- mirrors
    # /api/day/cancel's own 409. Wait on THIS job rather than starting a
    # second one and cancelling the first while it is still running.
    payload = _field_payload(layout={"type": "fermat", "n": 8}, workers=1)
    job_id, status = _run_field_trace_job(client, payload)
    assert status["state"] == "done", status
    assert client.post(f"/api/field/trace/cancel/{job_id}").status_code == 409


def test_field_trace_cancel_stops_before_completion(client, monkeypatch):
    """Serial path (workers=1): slow each heliostat trace down slightly so
    the cancel call, issued the moment the job starts, reliably lands before
    the job would finish on its own -- same technique as
    test_year_cancel_stops_the_job. This is the cooperative should_cancel
    check inside _trace_field_heliostats's serial loop, not the pool path
    below."""
    original = app_module._trace_core

    def slow_trace_core(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "_trace_core", slow_trace_core)

    payload = _field_payload(layout={"type": "fermat", "n": 30}, workers=1)
    t0 = time.perf_counter()
    started = client.post("/api/field/trace/start", json=payload)
    job_id = started.json()["job_id"]
    assert client.post(f"/api/field/trace/cancel/{job_id}").status_code == 200

    status = None
    for _ in range(600):
        status = client.get(f"/api/field/trace/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.02)
    elapsed = time.perf_counter() - t0

    assert status["state"] == "cancelled"
    assert status["done"] < status["total"], "cancel did not stop the trace before it finished"
    # 30 heliostats at the injected 0.05s each is 1.5s serial; a prompt
    # cancel should land in a small fraction of that.
    assert elapsed < 1.0, f"cancel took {elapsed:.2f}s to land"

    assert client.get(f"/api/field/trace/result/{job_id}").status_code == 409


def test_field_trace_cancel_stops_promptly_under_a_worker_pool(client):
    """Parallel path (workers>1): a real, unmocked trace -- monkeypatching
    _trace_core has no effect inside a worker process, so this proves the
    ProcessPoolExecutor's own should_cancel poll (in
    _trace_field_heliostats) actually interrupts a pool mid-run, using
    monte_carlo (slow enough per heliostat that a field of 30 does not
    finish before the cancel below can land)."""
    payload = _field_payload(
        design=RECT_DESIGN,
        layout={"type": "fermat", "n": 30},
        mode="monte_carlo",
        n_rays=20_000,
        workers=4,
    )
    started = client.post("/api/field/trace/start", json=payload)
    job_id = started.json()["job_id"]

    # Wait for real progress so the cancel is proven to interrupt a pool
    # that is actually mid-trace, not just to beat the job to its start.
    status = None
    for _ in range(600):
        status = client.get(f"/api/field/trace/status/{job_id}").json()
        if status["done"] > 2 or status["state"] != "running":
            break
        time.sleep(0.02)
    assert status["state"] == "running", "field finished before any cancel could be tested"

    t_cancel = time.perf_counter()
    assert client.post(f"/api/field/trace/cancel/{job_id}").status_code == 200
    for _ in range(600):
        status = client.get(f"/api/field/trace/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.02)
    elapsed = time.perf_counter() - t_cancel

    assert status["state"] == "cancelled"
    assert status["done"] < status["total"], "cancel did not stop the pool before it finished"
    assert elapsed < 5.0, f"cancel took {elapsed:.2f}s to land under a worker pool"


def test_field_trace_parallel_matches_serial(client):
    """Determinism: FIELD_MC_SEED is seeded per heliostat id, not by
    completion order, so a field traced across several worker processes must
    sum to exactly the same numbers as one traced serially -- monte_carlo
    is the more demanding case, since it is also seed-sensitive per ray."""
    payload = _field_payload(
        design=RECT_DESIGN,
        layout={"type": "fermat", "n": 10},
        mode="monte_carlo",
        n_rays=5_000,
    )
    serial = client.post("/api/field/trace", json={**payload, "workers": 1})
    parallel = client.post("/api/field/trace", json={**payload, "workers": 3})
    assert serial.status_code == parallel.status_code == 200
    assert _strip_volatile(serial.json()) == _strip_volatile(parallel.json())


def test_field_trace_parallel_matches_serial_via_the_job_endpoint(client):
    """Same determinism claim, exercised through the actual background-job
    path a big field goes through (this is the endpoint that defaults to
    parallel), not just the synchronous one."""
    payload = _field_payload(layout={"type": "fermat", "n": 8})
    _job1, status1 = _run_field_trace_job(client, {**payload, "workers": 1})
    _job2, status2 = _run_field_trace_job(client, {**payload, "workers": 3})
    assert status1["state"] == status2["state"] == "done"
    result1 = client.get(f"/api/field/trace/result/{_job1}").json()
    result2 = client.get(f"/api/field/trace/result/{_job2}").json()
    assert _strip_volatile(result1) == _strip_volatile(result2)


def test_synchronous_field_trace_default_workers_is_serial(client):
    """/api/field/trace's own behaviour is pinned: with no `workers` in the
    request it must trace one heliostat at a time, exactly as it always
    has -- the scripts and tests that already call it synchronously are not
    signing up for parallelism (or its process-pool startup cost) just
    because it is now available."""
    payload = _field_payload(layout={"type": "fermat", "n": 6})
    default = client.post("/api/field/trace", json=payload)
    explicit_serial = client.post("/api/field/trace", json={**payload, "workers": 1})
    assert default.status_code == explicit_serial.status_code == 200
    assert _strip_volatile(default.json()) == _strip_volatile(explicit_serial.json())


def test_field_trace_pool_is_reused_across_requests(client):
    """Starting a process pool is real wall-clock time (spawning worker
    processes), so a parallel field trace reuses one module-level pool
    across requests -- see _acquire_field_pool -- instead of building and
    tearing one down on every Run click. Before the fix,
    _trace_field_heliostats built a fresh ProcessPoolExecutor as a local
    variable every call; there was no module-level pool at all to find
    identical across two requests."""
    payload = _field_payload(layout={"type": "fermat", "n": 6}, workers=2)
    assert client.post("/api/field/trace", json=payload).status_code == 200
    pool_after_first = app_module._field_pool
    assert pool_after_first is not None

    assert client.post("/api/field/trace", json=payload).status_code == 200
    assert app_module._field_pool is pool_after_first


def test_default_trace_workers_env_override(monkeypatch):
    monkeypatch.delenv(app_module.TRACE_WORKERS_ENV, raising=False)
    monkeypatch.setattr(app_module.os, "cpu_count", lambda: 8)
    assert app_module._default_trace_workers() == 7

    monkeypatch.setenv(app_module.TRACE_WORKERS_ENV, "3")
    assert app_module._default_trace_workers() == 3

    monkeypatch.setenv(app_module.TRACE_WORKERS_ENV, "not-a-number")
    assert app_module._default_trace_workers() == 7


# -- saved setups -----------------------------------------------------------


@pytest.fixture
def setups_dir(tmp_path, monkeypatch):
    """Point the setups store at a temp dir, never the real home directory."""
    monkeypatch.setenv("HELIOSTAT_SETUPS_DIR", str(tmp_path / "setups"))
    return tmp_path / "setups"


def test_setup_round_trips(client, setups_dir):
    doc = {"version": 1, "values": {"sun-el": "37.5"}, "designType": "rect"}
    assert client.get("/api/setups").json() == {"setups": []}

    saved = client.post("/api/setups", json={"name": "My Tower", "document": doc})
    assert saved.status_code == 200
    assert saved.json()["name"] == "My Tower"

    listed = client.get("/api/setups").json()["setups"]
    assert [e["name"] for e in listed] == ["My Tower"]

    loaded = client.get("/api/setups/My Tower").json()
    assert loaded["document"] == doc

    assert client.delete("/api/setups/My Tower").status_code == 200
    assert client.get("/api/setups").json() == {"setups": []}


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "sub/dir",
        r"back\slash",
        ".hidden",
        "CON",
        "lpt1",
        "x" * 65,
    ],
)
def test_unsafe_setup_names_are_refused(client, setups_dir, name):
    """A setup name becomes a filename, so it must never be able to leave the
    setups directory or collide with a reserved device name."""
    resp = client.post("/api/setups", json={"name": name, "document": {}})
    assert resp.status_code == 422
    assert not list(setups_dir.glob("**/*.json")) if setups_dir.exists() else True


def test_saving_the_same_name_twice_overwrites(client, setups_dir):
    client.post("/api/setups", json={"name": "dup", "document": {"n": 1}})
    client.post("/api/setups", json={"name": "dup", "document": {"n": 2}})
    assert client.get("/api/setups/dup").json()["document"] == {"n": 2}
    assert len(client.get("/api/setups").json()["setups"]) == 1


def test_unreadable_setup_file_is_skipped_not_fatal(client, setups_dir):
    """One hand-edited file must not make the whole list unreadable."""
    client.post("/api/setups", json={"name": "good", "document": {"ok": True}})
    (setups_dir / "broken.json").write_text("{not json", encoding="utf-8")
    listed = client.get("/api/setups").json()["setups"]
    assert [e["name"] for e in listed] == ["good"]


# -- mirror sag -------------------------------------------------------------


@pytest.mark.parametrize("surface", ["twisting", "spherical", "flat"])
def test_sag_endpoint_renders_for_every_surface_mode(client, surface):
    resp = client.post("/api/design/sag", json=_trace_payload({**RECT_DESIGN, "surface": surface}))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_sag_matches_the_figure_the_trace_would_use(client):
    """The sag view must describe the mirror actually traced, not a
    plausible one. The legacy path negates c4/c5 for its inherited frame, so
    a sag map built from the raw solve would be mirrored about both axes."""
    payload = _trace_payload(RECT_DESIGN)
    sol = _solve_for(
        payload["optics"], 0.0, -89609.0, payload["solar_az_deg"], payload["solar_el_deg"]
    )
    # Sampled the way the renderer samples: the traced surface is
    # (c3, -c4, -c5).
    edge_u = zernike_sag_and_slopes(
        np.array([MIRROR_HALF_X_MM]), np.array([0.0]), sol.c3, -sol.c4, -sol.c5
    )[0][0]
    edge_v = zernike_sag_and_slopes(
        np.array([0.0]), np.array([MIRROR_HALF_Y_MM]), sol.c3, -sol.c4, -sol.c5
    )[0][0]
    centre = zernike_sag_and_slopes(np.array([0.0]), np.array([0.0]), sol.c3, -sol.c4, -sol.c5)[0][
        0
    ]

    # A focusing mirror: vertex at zero, both edges pulled the same way, and
    # the two axes differing -- that difference is what "twisting" names.
    # "Zero" is relative to the figure's own depth: the ANSI defocus term
    # carries a constant piston (Z4 is proportional to 2r^2 - 1, so it is
    # nonzero at r = 0), which lands around a nanometre against tens of
    # millimetres of sag. That is an offset, not a shape.
    assert abs(centre) < 1e-4 * max(edge_u, edge_v)
    assert edge_u > 0 and edge_v > 0
    assert not math.isclose(edge_u, edge_v, rel_tol=0.05)

    assert client.post("/api/design/sag", json=payload).status_code == 200


def test_sag_rejects_a_sun_below_the_horizon(client):
    """There is no solve, so there is no figure to draw."""
    resp = client.post("/api/design/sag", json=_trace_payload(RECT_DESIGN, solar_el_deg=0.0))
    assert resp.status_code == 422


_SAG_CONTOUR_CANDIDATES_MM = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def test_sag_headers_report_contour_interval_peak_to_valley_and_slant_range(client):
    """The default manuscript sag request carries its own numbers as
    headers, not just baked into the picture: a caption the client can
    render without re-deriving anything, and the same slant range /api/trace
    reports for the identical heliostat."""
    payload = _trace_payload(RECT_DESIGN)
    resp = client.post("/api/design/sag", json=payload)
    assert resp.status_code == 200
    assert resp.content[:8] == PNG_MAGIC

    span = float(resp.headers["X-Peak-To-Valley-Mm"])
    interval = float(resp.headers["X-Contour-Interval-Mm"])
    slant_range_m = float(resp.headers["X-Slant-Range-M"])

    assert span > 0
    assert interval in _SAG_CONTOUR_CANDIDATES_MM
    # At most 12 contour lines across the span, and no smaller candidate
    # interval would also have fit that budget.
    assert span / interval <= 12
    smaller = [c for c in _SAG_CONTOUR_CANDIDATES_MM if c < interval]
    if smaller:
        assert span / max(smaller) > 12

    assert slant_range_m > 0
    sol = _solve_for(
        payload["optics"], 0.0, -89609.0, payload["solar_az_deg"], payload["solar_el_deg"]
    )
    # The header is rounded to 3 decimals (the contract's own precision),
    # so compare at that granularity rather than bit-for-bit.
    expected_slant_m = _slant_range_mm(sol, 0.0, -89609.0) / 1000.0
    assert slant_range_m == pytest.approx(expected_slant_m, abs=5e-4)


# -- §D FEA CSV exports -------------------------------------------------------


def _parse_fea_csv(text: str) -> tuple[list[str], list[tuple[float, ...]]]:
    """``(comment_lines, data_rows)`` for a §D-convention export: every
    ``#``-prefixed line, verbatim, and every other non-blank line parsed as
    a tuple of floats."""
    comments = []
    rows = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
        else:
            rows.append(tuple(float(x) for x in line.split(",")))
    return comments, rows


def test_sag_csv_matches_the_png_endpoints_grid(client):
    """The FEA sag export must describe the exact same surface the sag PNG
    draws -- checked here by an invariant that does not require decoding
    pixels: the CSV's own peak-to-valley (max z_sag - min z_sag) must equal
    the PNG endpoint's X-Peak-To-Valley-Mm header for the identical
    request, since both are sampled by the same _sag_grid_mm grid."""
    payload = _trace_payload(RECT_DESIGN)
    png_resp = client.post("/api/design/sag", json=payload)
    assert png_resp.status_code == 200
    expected_span = float(png_resp.headers["X-Peak-To-Valley-Mm"])

    csv_resp = client.post("/api/design/sag.csv", json=payload)
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    _comments, rows = _parse_fea_csv(csv_resp.text)
    assert rows
    z = [r[2] for r in rows]
    span = max(z) - min(z)
    assert span == pytest.approx(expected_span, abs=5e-4)


def test_sag_csv_header_states_units_subject_and_grid(client):
    """§D: units, heliostat/sun/mode/timestamp, and grid dimensions are
    always three separate `#` comment lines -- never left implied."""
    payload = _trace_payload(RECT_DESIGN)
    resp = client.post("/api/design/sag.csv", json=payload)
    assert resp.status_code == 200
    comments, rows = _parse_fea_csv(resp.text)
    assert len(comments) >= 3

    units_line = next(c for c in comments if c.startswith("# units:"))
    assert "meters" in units_line
    assert "z_sag_mm" in units_line and "millimeters" in units_line

    subject_line = next(c for c in comments if c.startswith("# heliostat:"))
    assert "sun:" in subject_line and "mode:" in subject_line and "timestamp:" in subject_line
    assert f"az={payload['solar_az_deg']:.2f}" in subject_line

    grid_line = next(c for c in comments if c.startswith("# grid:"))
    assert "x" in grid_line

    # x, y columns are meters -- well inside the mirror's own half-extent in
    # metres, never the millimetre numbers the PNG's u/v axes use.
    half_x_m = MIRROR_HALF_X_MM / 1000.0
    half_y_m = MIRROR_HALF_Y_MM / 1000.0
    assert all(abs(r[0]) <= half_x_m + 1e-6 for r in rows)
    assert all(abs(r[1]) <= half_y_m + 1e-6 for r in rows)


def test_sag_csv_rejects_a_sun_below_the_horizon(client):
    resp = client.post("/api/design/sag.csv", json=_trace_payload(RECT_DESIGN, solar_el_deg=0.0))
    assert resp.status_code == 422


def test_flux_fea_csv_is_a_meters_and_w_m2_point_grid(client):
    """The new FEA convention must never be confused with the existing
    kW/m2 millimetre matrix export: same trace, different file. Cross-checked
    against /api/trace/flux.csv's own peak (kW/m2) so a unit slip (a factor
    of 1000, or mm vs m) would fail this test."""
    payload = _trace_payload(RECT_DESIGN)
    matrix_resp = client.post("/api/trace/flux.csv", json=payload)
    assert matrix_resp.status_code == 200
    matrix_rows = list(csv.reader(StringIO(matrix_resp.text)))
    matrix_peak_kw_m2 = max(
        float(x) for row in matrix_rows[1:] for x in row[1:]
    )

    resp = client.post("/api/trace/flux_fea.csv", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    comments, rows = _parse_fea_csv(resp.text)
    assert len(rows) == FLUX_GRID * FLUX_GRID

    units_line = next(c for c in comments if c.startswith("# units:"))
    assert "meters" in units_line
    assert "flux_w_m2" in units_line and "W/m" in units_line

    grid_line = next(c for c in comments if c.startswith("# grid:"))
    assert f"{FLUX_GRID}" in grid_line

    peak_w_m2 = max(r[2] for r in rows)
    assert peak_w_m2 == pytest.approx(matrix_peak_kw_m2 * 1000.0, rel=1e-4)
    assert all(r[2] >= 0 for r in rows)


def test_flux_fea_csv_notes_curved_receiver_unrolling(client):
    """A cylinder/frustum receiver's x/y are really an unrolled (arc length,
    height-or-slant) grid -- ANSYS maps flat coordinates, so §D requires the
    header to say so explicitly rather than leave "x" looking like a plain
    world coordinate."""
    flat_payload = _trace_payload(RECT_DESIGN)
    flat_resp = client.post("/api/trace/flux_fea.csv", json=flat_payload)
    assert flat_resp.status_code == 200
    flat_comments, _ = _parse_fea_csv(flat_resp.text)
    assert not any("arc length" in c for c in flat_comments)

    cyl_payload = _trace_payload(RECT_DESIGN)
    cyl_payload["optics_params"] = {"receiver_type": "cylinder"}
    cyl_resp = client.post("/api/trace/flux_fea.csv", json=cyl_payload)
    assert cyl_resp.status_code == 200
    cyl_comments, _ = _parse_fea_csv(cyl_resp.text)
    assert any("arc length" in c for c in cyl_comments)


def test_day_flux_fea_csv_matches_the_stored_pngs_peak(client):
    """The day-sweep timestep export is built once, alongside the PNG,
    during the sweep -- never a re-trace -- so its peak must agree with the
    PNG-adjacent peak_flux_kw_m2 already reported for that step."""
    job_id, data = _run_day(client, hour_step=2.0)
    kept = [i for i, s in enumerate(data["steps"]) if s["has_flux_map"]]
    assert kept
    step = data["steps"][kept[0]]

    resp = client.get(f"/api/day/flux/{job_id}/{kept[0]}.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    comments, rows = _parse_fea_csv(resp.text)
    assert rows
    units_line = next(c for c in comments if c.startswith("# units:"))
    assert "meters" in units_line and "W/m" in units_line

    peak_w_m2 = max(r[2] for r in rows)
    assert peak_w_m2 == pytest.approx(step["peak_flux_kw_m2"] * 1000.0, rel=1e-3)


def test_day_flux_fea_csv_404s_for_a_step_without_a_stored_map(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_DAY_FLUX_MAPS", 3)
    job_id, data = _run_day(client, hour_step=1.0)
    steps = data["steps"]
    assert len(steps) > 3
    skipped = [i for i, s in enumerate(steps) if not s["has_flux_map"]]
    assert skipped
    assert client.get(f"/api/day/flux/{job_id}/{skipped[0]}.csv").status_code == 404


def test_day_flux_fea_csv_404s_for_unknown_job(client):
    assert client.get("/api/day/flux/nosuchjob/0.csv").status_code == 404


# -- ray budget, day sweeps and exports --------------------------------------


def test_monte_carlo_ray_budget_is_adjustable(client):
    """Rays are the Monte Carlo fidelity dial; without control over them the
    only choice is 120,000 or a different backend."""
    payload = _trace_payload(RECT_DESIGN, mode="monte_carlo")
    payload["n_rays"] = 5000
    data = client.post("/api/trace", json=payload).json()
    assert data["counters"]["emitted"] == 5000


def test_ray_budget_is_ignored_by_the_cone_backends(client):
    """The cone backends sample the mirror on a fixed grid; there is no ray
    count to set, and pretending otherwise would be a lie in the UI."""
    payload = _trace_payload(RECT_DESIGN, mode="ultra_fast")
    payload["n_rays"] = 5000
    data = client.post("/api/trace", json=payload).json()
    assert data["counters"].get("emitted") is None
    assert data["power_w"] > 0


def _run_day(client, **overrides):
    payload = _trace_payload(RECT_DESIGN)
    payload.update(overrides)
    started = client.post("/api/day/start", json=payload)
    assert started.status_code == 200, started.json()
    job_id = started.json()["job_id"]
    for _ in range(600):
        status = client.get(f"/api/day/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "done", status
    return job_id, client.get(f"/api/day/result/{job_id}").json()


def test_day_trace_walks_the_daylight_hours(client):
    """Sunrise to sunset, with power rising and falling across it."""
    _job_id, data = _run_day(client, hour_step=2.0)
    steps = data["steps"]
    assert len(steps) >= 4
    assert steps == sorted(steps, key=lambda s: s["hour"])
    # The sun starts and ends near the horizon and is highest in between.
    assert steps[0]["solar_el_deg"] < max(s["solar_el_deg"] for s in steps)
    assert steps[-1]["solar_el_deg"] < max(s["solar_el_deg"] for s in steps)
    assert data["energy_kwh"] > 0
    assert all(s["power_w"] >= 0 for s in steps)


def test_day_energy_is_the_integral_of_its_own_power_curve(client):
    """The headline number must be the trapezoid of the table beneath it, or
    the two disagree in public."""
    _job_id, data = _run_day(client, hour_step=2.0)
    steps = data["steps"]
    hours = np.array([s["hour"] for s in steps])
    power_kw = np.array([s["power_w"] for s in steps]) / 1000.0
    # 1e-4 rather than tighter: the response rounds energy to 3 decimals
    # and each power to 4, so recomputing from the published table cannot
    # agree to more than that. The point is that it is the same integral,
    # not a different quantity.
    assert data["energy_kwh"] == pytest.approx(float(np.trapz(power_kw, hours)), rel=1e-4)


def test_min_elevation_deg_excludes_low_sun_and_never_increases_day_energy(client):
    """Pins heliostat.solar.build_time_grid's min_elevation_deg contract end
    to end, through the real /api/day/start path.

    Every surviving timestep must sit at or above the floor. And since the
    day sweep's headline energy_kwh is a plain trapezoid over just the
    surviving (non-negative-power) samples (heliostat.web.app
    ._day_energy_kwh -- no zero-anchored wings, unlike
    heliostat.energy.traced_day_energy), shrinking the sampled window can
    only remove area from that integral, never add to it: a 5 deg floor
    must score <= a 0 deg floor, and -- given the default site/date really
    has traced power near the horizon -- strictly less, not just equal.
    """
    _job0, data0 = _run_day(client, hour_step=1.0, min_elevation_deg=0.0)
    _job5, data5 = _run_day(client, hour_step=1.0, min_elevation_deg=5.0)

    steps5 = data5["steps"]
    assert steps5, "the floored sweep traced nothing"
    assert all(s["solar_el_deg"] >= 5.0 - 1e-6 for s in steps5)
    # The floor actually excluded some low-sun timesteps at this site/date.
    assert len(steps5) < len(data0["steps"])

    assert data5["energy_kwh"] < data0["energy_kwh"]


def test_day_progress_is_reported_and_result_waits_for_it(client):
    payload = _trace_payload(RECT_DESIGN)
    payload["hour_step"] = 2.0
    job_id = client.post("/api/day/start", json=payload).json()["job_id"]
    # Asking for a result before it exists is a 409, not a wrong answer.
    early = client.get(f"/api/day/result/{job_id}")
    assert early.status_code in (200, 409)
    for _ in range(600):
        status = client.get(f"/api/day/status/{job_id}").json()
        assert 0 <= status["done"] <= status["total"]
        if status["state"] != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "done"
    assert status["done"] == status["total"]


def test_day_one_bad_timestep_does_not_lose_the_whole_day(client, monkeypatch):
    """A day sweep is minutes of work; one timestep's solve blowing up (a
    numerically awkward sun angle, say) must cost that one point, not every
    timestep already traced. Before the fix, any exception out of
    _trace_instant_metrics propagated past day_start's own try/except
    (which only caught cancellation) straight into JobRegistry's generic
    handler, landing the whole job on state="error" with nothing to show
    for the timesteps that had already succeeded."""
    calls = {"n": 0}
    original = app_module._trace_core

    def flaky_trace_core(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated solve blowup")
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "_trace_core", flaky_trace_core)

    job_id, data = _run_day(client, hour_step=1.0)
    assert len(data["failed_steps"]) == 1
    assert "simulated solve blowup" in data["failed_steps"][0]["error"]
    # The failed timestep is excluded from steps (not faked as zero power),
    # so the day's own point count is short by exactly one.
    assert calls["n"] > len(data["steps"])
    assert data["energy_kwh"] > 0


def test_day_trace_of_a_field_carries_the_occlusion(client):
    """A field shades itself at low sun and not at noon; a day sweep that did
    not show that would not be tracing the field it claims to."""
    _job_id, data = _run_day(client, hour_step=2.0, layout={"type": "fermat", "n": 8})
    etas = [s["eta_mean"] for s in data["steps"]]
    assert all(0.0 < e <= 1.0 for e in etas)
    assert min(etas) < max(etas)
    assert data["n_heliostats"] == 8


def test_day_export_is_csv_with_a_row_per_timestep(client):
    job_id, data = _run_day(client, hour_step=2.0)
    resp = client.get(f"/api/day/export/{job_id}.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(StringIO(resp.text)))
    assert len(rows) == len(data["steps"])
    assert {"key", "hour", "solar_el_deg", "power_w"} <= set(rows[0])


def test_unknown_day_job_is_404(client):
    assert client.get("/api/day/status/nosuchjob").status_code == 404
    assert client.get("/api/day/result/nosuchjob").status_code == 404


# -- day sweep flux maps ------------------------------------------------------


def test_day_flux_step_indices_keeps_everything_under_the_cap():
    assert _day_flux_step_indices(5, cap=10) == set(range(5))


def test_day_flux_step_indices_strides_above_the_cap():
    kept = _day_flux_step_indices(100, cap=10)
    assert len(kept) == 10
    assert min(kept) == 0
    assert max(kept) == 99


def test_day_result_rows_say_which_have_a_stored_flux_map(client):
    """has_flux_map is the client's only way to know a click will be
    instant rather than falling back to a live trace."""
    _job_id, data = _run_day(client, hour_step=2.0)
    assert data["steps"]
    assert len(data["steps"]) <= MAX_DAY_FLUX_MAPS
    assert all("has_flux_map" in s for s in data["steps"])
    assert all(s["has_flux_map"] for s in data["steps"])


def test_day_flux_png_is_a_real_png(client):
    job_id, data = _run_day(client, hour_step=2.0)
    kept = [i for i, s in enumerate(data["steps"]) if s["has_flux_map"]]
    assert kept
    resp = client.get(f"/api/day/flux/{job_id}/{kept[0]}.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == PNG_MAGIC


def test_day_flux_png_matches_a_direct_trace_of_the_same_timestep(client):
    """The stored map for a step is the same trace a direct /api/trace at
    that step's own sun angles produces -- ultra_fast is a deterministic
    cone trace (no RNG), so the peak flux must match to the precision the
    day result reports it at (rounded to 4 decimals).

    The replay uses the timestep's exact, unrounded sun angles from
    _day_timesteps -- the JSON row itself rounds solar_az_deg/solar_el_deg
    to 3 decimals, and echoing that rounded angle back into a fresh trace
    would compare two different sun positions, not the same one twice.

    The PNG bytes are deliberately NOT compared: _render_flux_png titles the
    image with how long the trace took, so two renders of identical flux
    data still differ byte-for-byte. Asserting byte equality here would be
    flaky on timing alone, not a real regression signal.
    """
    payload = _trace_payload(RECT_DESIGN)
    payload["hour_step"] = 2.0
    job_id, data = _run_day(client, hour_step=2.0)
    step = data["steps"][0]
    assert step["has_flux_map"]
    resp = client.get(f"/api/day/flux/{job_id}/0.png")
    assert resp.status_code == 200
    assert resp.content[:8] == PNG_MAGIC

    exact_step = _day_timesteps(DayTraceRequest(**payload))[0]
    direct_payload = _trace_payload(RECT_DESIGN, solar_el_deg=exact_step.solar_el_deg)
    direct_payload["solar_az_deg"] = exact_step.solar_az_deg
    direct = client.post("/api/trace", json=direct_payload).json()
    assert direct["peak_flux_kw_m2"] == pytest.approx(step["peak_flux_kw_m2"], abs=5e-5)


def test_day_flux_png_matches_a_direct_field_trace(client):
    """Same determinism check as above, for a field: occlusion and the
    per-heliostat Monte Carlo seed (FIELD_MC_SEED) are both fixed by
    heliostat id, so a field's stored map is exactly reproducible too."""
    payload = _field_payload(RECT_DESIGN, layout={"type": "fermat", "n": 5})
    payload["hour_step"] = 3.0
    job_id, data = _run_day(client, hour_step=3.0, layout={"type": "fermat", "n": 5})
    step = data["steps"][0]
    assert step["has_flux_map"]
    resp = client.get(f"/api/day/flux/{job_id}/0.png")
    assert resp.status_code == 200

    exact_step = _day_timesteps(DayTraceRequest(**payload))[0]
    direct_payload = _field_payload(RECT_DESIGN, layout={"type": "fermat", "n": 5})
    direct_payload["solar_az_deg"] = exact_step.solar_az_deg
    direct_payload["solar_el_deg"] = exact_step.solar_el_deg
    direct = client.post("/api/field/trace", json=direct_payload).json()
    assert direct["peak_flux_kw_m2"] == pytest.approx(step["peak_flux_kw_m2"], abs=5e-5)


def test_day_flux_map_cap_strides_the_kept_steps(client, monkeypatch):
    """Above the cap, only a strided subset of steps keeps a map -- every
    row still says which via has_flux_map, and only those steps' endpoints
    serve a PNG. The rest are a 404, not a silently different response
    shape, so the client can always tell the two cases apart."""
    monkeypatch.setattr(app_module, "MAX_DAY_FLUX_MAPS", 3)
    job_id, data = _run_day(client, hour_step=1.0)
    steps = data["steps"]
    assert len(steps) > 3

    kept = [i for i, s in enumerate(steps) if s["has_flux_map"]]
    assert len(kept) == 3
    assert kept[0] == 0
    assert kept[-1] == len(steps) - 1
    for i in kept:
        assert client.get(f"/api/day/flux/{job_id}/{i}.png").status_code == 200

    skipped = [i for i in range(len(steps)) if i not in kept]
    assert skipped
    assert client.get(f"/api/day/flux/{job_id}/{skipped[0]}.png").status_code == 404


def test_day_flux_png_404s_for_unknown_job(client):
    assert client.get("/api/day/flux/nosuchjob/0.png").status_code == 404


def test_day_flux_png_404s_for_an_out_of_range_step(client):
    job_id, data = _run_day(client, hour_step=2.0)
    out_of_range = len(data["steps"])
    assert client.get(f"/api/day/flux/{job_id}/{out_of_range}.png").status_code == 404
    assert client.get(f"/api/day/flux/{job_id}/-1.png").status_code == 404


def test_day_flux_png_409s_while_running(client):
    """Same still-running semantics as /api/day/result: racy by nature (the
    background thread may finish before this request lands), which is why
    test_day_progress_is_reported_and_result_waits_for_it accepts the same
    two outcomes for /api/day/result."""
    payload = _trace_payload(RECT_DESIGN)
    payload["hour_step"] = 0.2
    payload["layout"] = {"type": "fermat", "n": 20}
    job_id = client.post("/api/day/start", json=payload).json()["job_id"]
    resp = client.get(f"/api/day/flux/{job_id}/0.png")
    assert resp.status_code in (200, 409)


# -- year estimate ------------------------------------------------------------
# hour_step is the widest the server allows (6h, ~3 samples/day) and the
# field small (single heliostat, or a small Fermat spiral) throughout -- a
# year estimate traces several days at once, so anything denser would make
# this section of the suite slow for no benefit.

_YEAR_SITE = {"latitude_deg": -10.0, "longitude_deg": -52.0, "timezone_h": -3.0, "year": 2026}


def _year_payload(design=RECT_DESIGN, optics="prime_focus", **overrides):
    payload = {
        "design": design,
        "mode": "ultra_fast",
        "optics": optics,
        "solar_az_deg": 180.0,
        "solar_el_deg": 45.0,
        "site": dict(_YEAR_SITE),
        "hour_step": 6.0,
    }
    payload.update(overrides)
    return payload


def _run_year(client, **overrides):
    payload = _year_payload(**overrides)
    started = client.post("/api/year/start", json=payload)
    assert started.status_code == 200, started.json()
    job_id = started.json()["job_id"]
    for _ in range(600):
        status = client.get(f"/api/year/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "done", status
    return job_id, client.get(f"/api/year/result/{job_id}").json()


def test_year_hour_step_scales_the_time_grid_like_the_day_sweep_does():
    """The Analysis tab's "Timestep (h)" field drives the day sweep and,
    since bug fix, the year estimate too -- YearTraceRequest.hour_step
    (default 1.0, same field/semantics as DayTraceRequest's) must actually
    coarsen or refine the traced time grid, not just be accepted and
    ignored (which is what shipped: the year estimate always traced at a
    hardcoded ~1 h regardless of what the day sweep's own field said).

    Pinned through heliostat.web.app's own _year_energy_cfg/
    _year_trace_dates/build_time_grid -- the exact three calls
    /api/year/start makes before it ever starts tracing -- rather than
    running a real (background-job) year estimate, so this is cheap and
    needs no polling loop.
    """

    def n_steps(hour_step):
        req = YearTraceRequest.model_validate(_year_payload(hour_step=hour_step, fast_mode=True))
        cfg = _year_energy_cfg(req)
        trace_dates = _year_trace_dates(cfg, req.site.year, req.fast_mode)
        cfg.sweep.dates = trace_dates
        return len(build_time_grid(cfg, trace_dates))

    n_fine = n_steps(0.5)
    n_coarse = n_steps(3.0)
    assert n_fine > n_coarse
    # Not just "more" -- roughly the 6x the hour_step ratio implies, which
    # is what pins the field as actually wired through rather than merely
    # nudging the count by some unrelated side effect.
    assert n_fine / n_coarse == pytest.approx(6.0, rel=0.35)


def test_year_start_reports_a_plausible_finite_annual_total(client):
    _job_id, data = _run_year(client)
    assert math.isfinite(data["annual_energy_mwh"])
    assert data["annual_energy_mwh"] > 0.0
    assert data["n_heliostats"] == 1
    assert data["state"] == "done"


def test_year_result_labels_the_dni_as_clearsky(client):
    """docs/ui-spec.md 4: ClearSkyDNI is a cloud-free upper bound, and the
    result must say so rather than presenting it as a weather-corrected
    estimate."""
    _job_id, data = _run_year(client)
    assert "ClearSky" in data["dni_provider"]
    assert "upper bound" in data["dni_provider"]


def test_year_fast_mode_traces_seven_dates_of_twelve_reported(client):
    _job_id, data = _run_year(client, fast_mode=True)
    assert data["fast_mode"] is True
    assert data["n_days_traced"] == 7
    days = data["days"]
    assert len(days) == 12
    assert sum(d["traced"] for d in days) == 7
    assert sum(not d["traced"] for d in days) == 5


def test_year_slow_mode_traces_all_twelve(client):
    _job_id, data = _run_year(client, fast_mode=False)
    assert data["fast_mode"] is False
    assert data["n_days_traced"] == 12
    days = data["days"]
    assert len(days) == 12
    assert all(d["traced"] for d in days)


def test_year_one_bad_timestep_costs_its_date_not_the_year(client, monkeypatch):
    """Same failure mode as the day job (see
    test_day_one_bad_timestep_does_not_lose_the_whole_day), a year estimate
    away: one timestep's solve raising must not throw out every date
    already traced. The existing "only fully-traced dates count" filter
    (a date cut short by cancellation is not a real day) already excludes
    whichever date the bad timestep falls on once that timestep is skipped
    rather than counted -- the fix only has to stop the exception from
    reaching JobRegistry's generic handler in the first place."""
    calls = {"n": 0}
    original = app_module._trace_instant_metrics

    def flaky_metrics(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("simulated solve blowup")
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "_trace_instant_metrics", flaky_metrics)

    _job_id, data = _run_year(client, fast_mode=False, hour_step=6.0)
    assert len(data["failed_steps"]) == 1
    assert "simulated solve blowup" in data["failed_steps"][0]["error"]
    # Slow mode traces all 12 dates; exactly the one date the bad timestep
    # landed on drops out, the rest still complete the year.
    assert data["n_days_traced"] == 11
    assert math.isfinite(data["annual_energy_mwh"])
    assert data["annual_energy_mwh"] > 0.0


def test_year_days_are_sorted_and_span_the_year(client):
    _job_id, data = _run_year(client, fast_mode=True)
    dates = [_dt.date.fromisoformat(d["date"]) for d in data["days"]]
    assert dates == sorted(dates)
    assert (dates[-1] - dates[0]).days > 300


def test_year_fast_mode_symmetry_reconstruction_matches_a_direct_trace(client):
    """A mirrored (untraced) reported day borrows its optics from a traced
    twin on the other side of a solstice. Check one against an independent,
    directly-traced day integrated the same way -- the two share no code
    path past ``power_w`` (see energy.traced_day_energy's own docstring on
    ``source_date``), so close agreement is real evidence the symmetry
    argument holds, not a tautology."""
    _job_id, data = _run_year(client, fast_mode=True)
    mirrored = [d for d in data["days"] if not d["traced"]]
    assert mirrored
    entry = mirrored[0]
    twin_date = _dt.date.fromisoformat(entry["date"])

    _day_job_id, day_data = _run_day(
        client,
        site={
            "latitude_deg": _YEAR_SITE["latitude_deg"],
            "longitude_deg": _YEAR_SITE["longitude_deg"],
            "timezone_h": _YEAR_SITE["timezone_h"],
            "year": twin_date.year,
            "month": twin_date.month,
            "day": twin_date.day,
        },
        hour_step=6.0,
    )
    rows = [
        {
            "date": twin_date,
            "hour": s["hour"],
            "heliostat_id": 0,
            "power_w": s["power_w"],
            "solar_az_deg": s["solar_az_deg"],
            "solar_el_deg": s["solar_el_deg"],
        }
        for s in day_data["steps"]
    ]
    summary = pd.DataFrame(rows)
    site = SimpleNamespace(
        latitude=_YEAR_SITE["latitude_deg"],
        longitude=_YEAR_SITE["longitude_deg"],
        timezone=_YEAR_SITE["timezone_h"],
    )
    direct = energy.traced_day_energy(summary, SimpleNamespace(site=site), ClearSkyDNI(site), date=twin_date)
    assert entry["energy_kwh"] == pytest.approx(direct["energy_kwh"], rel=0.08)


def test_year_cancel_stops_the_job(client, monkeypatch):
    """Slow each traced instant down slightly so the cancel call, issued the
    moment the job starts, reliably lands before the job would finish on its
    own -- otherwise this is a race the test cannot control."""
    original = app_module._trace_instant_metrics

    def slow_metrics(*args, **kwargs):
        time.sleep(0.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(app_module, "_trace_instant_metrics", slow_metrics)

    payload = _year_payload(fast_mode=False, hour_step=1.0)
    started = client.post("/api/year/start", json=payload)
    job_id = started.json()["job_id"]
    assert client.post(f"/api/year/cancel/{job_id}").status_code == 200

    status = None
    for _ in range(600):
        status = client.get(f"/api/year/status/{job_id}").json()
        if status["state"] != "running":
            break
        time.sleep(0.02)
    assert status["state"] == "cancelled"
    assert status["done"] < status["total"]

    result = client.get(f"/api/year/result/{job_id}")
    assert result.status_code == 200
    assert result.json()["state"] == "cancelled"


def test_year_field_trace_carries_occlusion(client):
    """A field, not just a single heliostat, must actually be traceable
    through the year job -- n_heliostats in the result says which."""
    _job_id, data = _run_year(client, layout={"type": "fermat", "n": 8}, fast_mode=True)
    assert data["n_heliostats"] == 8
    assert data["annual_energy_mwh"] > 0.0


def test_unknown_year_job_is_404(client):
    assert client.get("/api/year/status/nosuchjob").status_code == 404
    assert client.get("/api/year/result/nosuchjob").status_code == 404


def test_flux_csv_export_is_a_labelled_grid(client):
    """Self-describing: the axes travel with the numbers."""
    resp = client.post("/api/trace/flux.csv", json=_trace_payload(RECT_DESIGN))
    assert resp.status_code == 200
    rows = list(csv.reader(StringIO(resp.text)))
    assert len(rows) == FLUX_GRID + 1
    assert len(rows[0]) == FLUX_GRID + 1
    assert "u_mm" in rows[0][0]
    # Header and index are receiver coordinates in millimetres, ascending.
    u_axis = [float(x) for x in rows[0][1:]]
    assert u_axis == sorted(u_axis)
    assert max(float(x) for x in rows[1][1:]) >= 0.0


# ---------------------------------------------------------------------------
# docs/ui-spec-v0.2.md §M.4: the analysis aperture. _aperture_metrics is the
# Python reference implementation js/tabs/analysis.js's own apertureMetrics
# mirrors bin for bin -- see that function's docstring for why this repo
# checks the formula here rather than in JS (no JS test runner). Both tests
# below build a SYNTHETIC grid directly (no trace, no client, no job) --
# exactly the "post-processing only" contract §M.4 asks for: the aperture
# math must stand on its own as pure arithmetic over a grid.
# ---------------------------------------------------------------------------


def _synthetic_uniform_grid(flux_w_m2: float, n: int, half_extent_mm: float) -> tuple:
    """An n x n grid of constant flux over a square centred on the origin --
    ``(flux_array, u_min_mm, u_max_mm, v_min_mm, v_max_mm)``, the exact
    positional arguments :func:`_aperture_metrics` takes."""
    flux = np.full((n, n), flux_w_m2, dtype=float)
    return flux, -half_extent_mm, half_extent_mm, -half_extent_mm, half_extent_mm


def test_aperture_metrics_whole_grid_is_exact():
    """No circular-clipping ambiguity at all: the aperture radius is large
    enough that EVERY bin of a uniform-flux grid falls inside it, so the
    summed power has one unambiguous exact answer -- flux times the grid's
    own rectangular area, to floating-point precision, not just "close".
    Average flux then divides by the APERTURE's own ideal circular area
    (pi * r^2), which is much bigger than the grid it fully encloses -- this
    is what pins down that avg flux is not silently "whatever's inside
    divided by whatever's inside" but the aperture's own declared area.
    """
    flux_w_m2 = 850.0
    half_extent_mm = 1000.0
    flux, u0, u1, v0, v1 = _synthetic_uniform_grid(flux_w_m2, n=200, half_extent_mm=half_extent_mm)
    huge_radius_mm = 10.0 * half_extent_mm * math.sqrt(2)  # exceeds the grid's own half-diagonal

    out = _aperture_metrics(flux, u0, u1, v0, v1, center_u_mm=0.0, center_v_mm=0.0, radius_mm=huge_radius_mm)

    grid_area_m2 = ((u1 - u0) / 1000.0) * ((v1 - v0) / 1000.0)
    expected_power_w = flux_w_m2 * grid_area_m2
    assert out["power_w"] == pytest.approx(expected_power_w, rel=1e-9)
    assert out["n_bins_inside"] == flux.size  # every bin, none clipped

    expected_avg_flux = expected_power_w / (math.pi * (huge_radius_mm / 1000.0) ** 2)
    assert out["avg_flux_w_m2"] == pytest.approx(expected_avg_flux, rel=1e-9)
    # The aperture is mostly empty space beyond the grid, so its average
    # flux reads far below the field's own uniform value -- confirms this
    # isn't accidentally averaging over just the populated bins.
    assert out["avg_flux_w_m2"] < flux_w_m2


def test_aperture_metrics_uniform_disk_matches_the_analytic_answer():
    """The textbook case spec §M.4 calls out by name: a uniform-flux field,
    read through an aperture that actually clips a circle out of it. For any
    radius that fits inside the grid, the analytic answer is exact calculus
    (power = flux * pi * r^2, so avg flux = flux, independent of r) --
    checked here on a grid fine enough that the Cartesian rasterization of
    the circle's boundary is a small fraction of its interior, so the
    numeric answer converges tightly on the analytic one.
    """
    flux_w_m2 = 1000.0
    half_extent_mm = 1000.0
    # 500 bins across 2000 mm = 4 mm bins; a couple of test radii keep the
    # boundary-bin fraction (circumference / area, ~ 2/r in bin-widths) well
    # under 1%.
    flux, u0, u1, v0, v1 = _synthetic_uniform_grid(flux_w_m2, n=500, half_extent_mm=half_extent_mm)

    for radius_mm in (300.0, 600.0, 900.0):
        out = _aperture_metrics(flux, u0, u1, v0, v1, center_u_mm=0.0, center_v_mm=0.0, radius_mm=radius_mm)
        analytic_power_w = flux_w_m2 * math.pi * (radius_mm / 1000.0) ** 2
        assert out["power_w"] == pytest.approx(analytic_power_w, rel=5e-3)
        # Average flux of a uniform field over ANY sub-region is the field
        # value itself -- this is the invariant that makes "uniform disk" a
        # useful analytic check at all, independent of the radius chosen.
        assert out["avg_flux_w_m2"] == pytest.approx(flux_w_m2, rel=5e-3)


def test_aperture_metrics_off_centre_disk_still_averages_to_the_field_value():
    """Same invariant as above, off-axis -- avg flux of a uniform field does
    not depend on where the aperture sits, only on it staying inside the
    grid (a centre near the grid's own edge would clip against the grid
    boundary too, which is a different, deliberately untested case)."""
    flux_w_m2 = 400.0
    half_extent_mm = 1000.0
    flux, u0, u1, v0, v1 = _synthetic_uniform_grid(flux_w_m2, n=500, half_extent_mm=half_extent_mm)

    out = _aperture_metrics(flux, u0, u1, v0, v1, center_u_mm=250.0, center_v_mm=-150.0, radius_mm=400.0)
    assert out["avg_flux_w_m2"] == pytest.approx(flux_w_m2, rel=5e-3)


# ---------------------------------------------------------------------------
# docs/ui-spec-v0.2.md §M.4: the day sweep's own DNI per timestep -- the
# "avg concentration = avg flux / DNI" readout's denominator.
# ---------------------------------------------------------------------------


def test_day_sweep_reports_dni_per_timestep(client):
    """dni_w_m2 rides on every timestep row, display/analysis only -- never
    fed back into the trace."""
    _job_id, data = _run_day(client, hour_step=2.0)
    steps = data["steps"]
    assert steps
    for step in steps:
        assert "dni_w_m2" in step
        # Clear-sky DNI is between "nothing" and the solar constant -- a
        # loose sanity bound, not a re-derivation of ClearSkyDNI's own model.
        assert 0.0 < step["dni_w_m2"] < ClearSkyDNI.E0
    # Higher sun (less atmosphere to cross) means less attenuation -- DNI
    # should be at least as high at the day's peak elevation as at its
    # lowest-elevation kept sample.
    by_elevation = sorted(steps, key=lambda s: s["solar_el_deg"])
    assert by_elevation[-1]["dni_w_m2"] >= by_elevation[0]["dni_w_m2"]


# ---------------------------------------------------------------------------
# docs/ui-spec-v0.2.md §M.4: the day sweep's raw flux grid, fetched by the
# Analysis tab's aperture -- same has_flux_map/404 contract as the PNG/CSV
# siblings, built once alongside them, never re-traced.
# ---------------------------------------------------------------------------


def test_day_flux_grid_json_matches_the_stored_pngs_peak(client):
    job_id, data = _run_day(client, hour_step=2.0)
    kept = [i for i, s in enumerate(data["steps"]) if s["has_flux_map"]]
    assert kept
    step = data["steps"][kept[0]]

    resp = client.get(f"/api/day/flux/{job_id}/{kept[0]}.grid.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    grid = resp.json()
    assert grid["n_u"] > 0 and grid["n_v"] > 0
    assert len(grid["values"]) == grid["n_u"] * grid["n_v"]
    assert grid["u_max_mm"] > grid["u_min_mm"]
    assert grid["v_max_mm"] > grid["v_min_mm"]

    peak_kw_m2 = max(v for v in grid["values"] if v is not None)
    # This is the SAME payload shape _flux_grid_payload builds for §M.3's 3D
    # drape -- downsampled and rounded to kW/m2 with 2 decimals, so the
    # match is close but not bit-exact against the full-resolution PNG/CSV
    # peak: block-averaging a peak bin with its neighbours only ever lowers
    # the UNROUNDED coarse value below the true peak, but the 2-decimal
    # rounding on top of that can still nudge the displayed number up to
    # 0.005 either way -- the +0.02 slack below covers that rounding, not a
    # real excursion above the true peak.
    assert peak_kw_m2 <= step["peak_flux_kw_m2"] + 0.02
    assert peak_kw_m2 == pytest.approx(step["peak_flux_kw_m2"], rel=0.15)


def test_day_flux_grid_json_404s_for_a_step_without_a_stored_map(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_DAY_FLUX_MAPS", 3)
    job_id, data = _run_day(client, hour_step=1.0)
    steps = data["steps"]
    assert len(steps) > 3
    skipped = [i for i, s in enumerate(steps) if not s["has_flux_map"]]
    assert skipped
    assert client.get(f"/api/day/flux/{job_id}/{skipped[0]}.grid.json").status_code == 404


def test_day_flux_grid_json_404s_for_unknown_job(client):
    assert client.get("/api/day/flux/nosuchjob/0.grid.json").status_code == 404


def test_day_flux_grid_json_404s_for_an_out_of_range_step(client):
    job_id, data = _run_day(client, hour_step=2.0)
    out_of_range = len(data["steps"]) + 5
    assert client.get(f"/api/day/flux/{job_id}/{out_of_range}.grid.json").status_code == 404


# ---------------------------------------------------------------------------
# docs/ui-spec-v0.2.md §M.4: the aperture annotation round-trips through
# save/reopen, unrecomputed -- SavedRunDocument.aperture is a loose dict the
# client already computed; the library layer only has to store and return
# it byte for byte.
# ---------------------------------------------------------------------------


def test_saved_run_aperture_annotation_round_trips(client):
    document = {
        "kind": "day",
        "project_name": None,
        "request": {"design": RECT_DESIGN, "mode": "ultra_fast", "optics": "prime_focus"},
        "result": {"steps": [], "date": "2026-03-21"},
        "flux_pngs": {},
        "aperture": {
            "step_index": 2,
            "center_u_mm": 12.5,
            "center_v_mm": -30.0,
            "radius_mm": 803.6524,
            "power_w": 133443.515625,
            "frac_collected_pct": 95.51006886727252,
            "avg_flux_w_m2": 65767.46622041808,
            "dni_w_m2": 986.79,
            "avg_concentration": 66.64788477833996,
        },
    }
    name = "aperture-roundtrip-test"
    saved = client.post("/api/library/runs", json={"name": name, "document": document})
    assert saved.status_code == 200, saved.json()

    loaded = client.get(f"/api/library/runs/{name}")
    assert loaded.status_code == 200
    assert loaded.json()["document"]["aperture"] == document["aperture"]

    client.delete(f"/api/library/runs/{name}")


def test_saved_run_without_an_aperture_defaults_to_none(client):
    """A run saved with no aperture drawn (or one saved before §M.4 existed)
    must still validate -- the field is optional, not a breaking schema
    change for old saved runs. Library storage round-trips the posted dict
    verbatim rather than materializing pydantic defaults into it (true of
    every other optional field here too, e.g. flux_pngs), so an old
    document that never had the key stays keyless on reload -- the
    frontend's own `document.aperture || null` read (js/tabs/analysis.js's
    openSavedRun) already treats a missing key the same as an explicit
    null, which is what this checks with .get rather than an indexed [].
    """
    document = {
        "kind": "day",
        "project_name": None,
        "request": {"design": RECT_DESIGN, "mode": "ultra_fast", "optics": "prime_focus"},
        "result": {"steps": [], "date": "2026-03-21"},
        "flux_pngs": {},
    }
    name = "aperture-absent-test"
    saved = client.post("/api/library/runs", json={"name": name, "document": document})
    assert saved.status_code == 200, saved.json()

    loaded = client.get(f"/api/library/runs/{name}")
    assert loaded.status_code == 200
    assert loaded.json()["document"].get("aperture") is None

    client.delete(f"/api/library/runs/{name}")
