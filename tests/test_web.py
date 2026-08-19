"""HTTP-level gate for the local web GUI (``heliostat.web``).

Skipped entirely when the ``web`` extra is not installed -- these are the
only tests in the suite that need FastAPI, and the rest of the package must
stay importable/testable without it.
"""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat import __version__  # noqa: E402
from heliostat.geometry.design import _petal_at_angle, flower, grid_facets  # noqa: E402
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM  # noqa: E402
from heliostat.web.app import (  # noqa: E402
    AXICON_APERTURE_RADIUS_MM,
    AXICON_APEX_HEIGHT_MM,
    AXICON_HALF_ANGLE_DEG,
    AXICON_RECEIVER_Z_MM,
    PRIME_FOCUS_HEIGHT_MM,
    WINDOW_MM,
    _geometry_for,
    create_app,
)
from heliostat.web.scene import MAX_SCENE_RAYS, radial_outline  # noqa: E402

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


def test_trace_beam_down_optics(client):
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
# LEGACY single-mirror path: a default-size adaptive rectangle is the one
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
    spelled_out = _trace_payload({**design, "surface": "adaptive"}, optics=optics)
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
    adaptive = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    flat = client.post(
        "/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "flat"})
    ).json()
    assert flat["rms_radius_mm"] > 3.0 * adaptive["rms_radius_mm"]
    assert flat["power_w"] > 0


def test_spherical_rect_is_not_the_legacy_path(client):
    """A default-size rectangle asking for a spherical figure is routed
    through the design path, so it cannot come back with the legacy path's
    astigmatic answer."""
    adaptive = client.post("/api/trace", json=_trace_payload(RECT_DESIGN)).json()
    spherical = client.post(
        "/api/trace", json=_trace_payload({**RECT_DESIGN, "surface": "spherical"})
    ).json()
    assert math.isfinite(spherical["rms_radius_mm"])
    assert spherical["rms_radius_mm"] != adaptive["rms_radius_mm"]
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


def test_surface_is_accepted_but_ignored_by_the_preview(client):
    """The preview draws footprint only and has no sun to resolve a figure
    against, so every surface mode previews the same picture."""
    for surface in ("adaptive", "spherical", "flat"):
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
        # Receiver at or above the cone: the beam-down drop would be zero or
        # negative and the solve would answer for a tower that cannot exist.
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
        'data-surface="adaptive"',
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
