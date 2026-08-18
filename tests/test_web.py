"""HTTP-level gate for the local web GUI (``heliostat.web``).

Skipped entirely when the ``web`` extra is not installed -- these are the
only tests in the suite that need FastAPI, and the rest of the package must
stay importable/testable without it.
"""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat import __version__  # noqa: E402
from heliostat.web.app import create_app  # noqa: E402

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
