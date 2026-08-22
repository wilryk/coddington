"""HTTP-level gate for the local web GUI (``heliostat.web``).

Skipped entirely when the ``web`` extra is not installed -- these are the
only tests in the suite that need FastAPI, and the rest of the package must
stay importable/testable without it.
"""

from __future__ import annotations

import base64
import json
import math

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat import __version__  # noqa: E402
from heliostat.geometry.design import _petal_at_angle, flower, grid_facets  # noqa: E402
from heliostat.geometry.heliostat import zernike_sag_and_slopes  # noqa: E402
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM  # noqa: E402
from heliostat.web.app import (  # noqa: E402
    AXICON_APERTURE_RADIUS_MM,
    AXICON_APEX_HEIGHT_MM,
    AXICON_HALF_ANGLE_DEG,
    AXICON_RECEIVER_Z_MM,
    FERMAT_A_M,
    FERMAT_B,
    MAX_FIELD_HELIOSTATS,
    PRIME_FOCUS_HEIGHT_MM,
    WINDOW_MM,
    FermatLayout,
    FlowerParams,
    _build_trace_design,
    _field_geometry,
    _geometry_for,
    _slant_range_mm,
    _solve_for,
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


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "heliostat" in resp.text.lower()


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
    """Blank cant_focal_mm on a grid/flower design now auto-focuses at the
    heliostat's own solved slant range instead of defaulting to flat;
    explicit cant_focal_mm=0 still opts into flat. rms should drop well
    below the deliberately-flat case -- measured ~1092 mm (auto) vs
    ~1415 mm (flat) at the default heliostat position/sun; the 0.85 factor
    below leaves comfortable margin."""
    auto = client.post("/api/trace", json=_trace_payload(FLOWER_DESIGN)).json()
    flat = client.post(
        "/api/trace", json=_trace_payload({**FLOWER_DESIGN, "cant_focal_mm": 0})
    ).json()
    assert auto["rms_radius_mm"] < 0.85 * flat["rms_radius_mm"]


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

    # -- receiver: the fixture window, verbatim.
    assert scene["receiver"] == {
        "z_mm": receiver.z_mm,
        "half_u_mm": receiver.half_u_mm,
        "half_v_mm": receiver.half_v_mm,
        "facing": receiver.facing,
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
PIN_DEFAULT_RECT_POWER_W = 8225.974283127302
PIN_DEFAULT_RECT_RMS_MM = 505.2641179604239
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
            {"window_half_u_mm": WINDOW_MM, "window_half_v_mm": WINDOW_MM},
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
        # The relay's own numbers are deliberately not adjustable, so they
        # are not echoed as if they were.
        assert set(resolved) == {"window_half_u_mm", "window_half_v_mm"}


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
def test_flat_surface_equals_todays_explicit_cant_zero(client, design):
    """`surface="flat"` and today's `cant_focal_mm=0` describe the same
    mirror when both are asked for at once -- Flat() and the builders' own
    all-zero ZernikeAstig default are the same figure."""
    legacy_flat = client.post(
        "/api/trace", json=_trace_payload({**design, "cant_focal_mm": 0})
    ).json()
    named_flat = client.post(
        "/api/trace",
        json=_trace_payload({**design, "cant_focal_mm": 0, "surface": "flat"}),
    ).json()
    assert _comparable(legacy_flat) == _comparable(named_flat)


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
    "field",
    ["focus_height_mm", "receiver_z_mm", "apex_height_mm", "vertex_z_mm", "conic", "z_mm"],
)
def test_cassegrain_position_fields_are_422(client, field):
    """Moving the Cassegrain receiver or relay would need the hyperboloid
    re-solved, so the app refuses rather than tracing wrong optics."""
    payload = _trace_payload(RECT_DESIGN, optics="cassegrain")
    payload["optics_params"] = {field: 12345.0}
    resp = client.post("/api/trace", json=payload)
    assert resp.status_code == 422
    detail = str(resp.json()["detail"])
    assert field in detail
    assert "relay" in detail


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
    """The GUI is one hand-written file with no build step, so the only cheap
    guard that its new controls actually shipped is that the markup is there.
    Behaviour is the lead's visual check, not this test's."""
    text = client.get("/").text
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


def test_index_carries_the_field_controls(client):
    """Markup-only guard, same reasoning as the surface/inspector one."""
    text = client.get("/").text
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
