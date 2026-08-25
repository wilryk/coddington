"""A fast, always-green slice of ``scripts/stress.py``'s own findings.

Not the stress harness itself (that is a separate, slow, sampling run --
see ``docs/stress-test-plan.md`` section 4) but a handful of the specific
payloads it turned up that currently pass, pinned here as a regression
guard so a future change cannot silently reopen one of them. Every case
runs in well under a second: no Monte Carlo, no field larger than a
handful of heliostats, no day sweep.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.web.app import MAX_FIELD_HELIOSTATS, create_app  # noqa: E402

RECT_DESIGN = {"type": "rect", "width_mm": 5000.0, "height_mm": 3000.0}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def _trace_payload(**overrides):
    payload = {
        "design": RECT_DESIGN,
        "mode": "ultra_fast",
        "optics": "prime_focus",
        "solar_az_deg": 180.0,
        "solar_el_deg": 45.0,
    }
    payload.update(overrides)
    return payload


def _all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    return True


def test_basic_trace_is_plausible(client):
    """power_w must not exceed incident_power_w by more than ultra_fast's
    own documented total-power precision (about +-0.1%, trace/modes.py) --
    the two are integrated through slightly different quadratures, so an
    exact match is not expected, but a large excess would be."""
    resp = client.post("/api/trace", json=_trace_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert _all_finite(body)
    assert body["power_w"] >= 0.0
    assert body["power_w"] <= body["incident_power_w"] * 1.002


def test_self_intersecting_polygon_does_not_crash(client):
    """A bowtie facet (edges crossing) is a legal, if odd, polygon -- it
    must trace to a finite, non-negative answer, never a 5xx."""
    design = {
        "type": "custom",
        "vertices_mm": [[-2000.0, -1000.0], [2000.0, 1000.0], [2000.0, -1000.0], [-2000.0, 1000.0]],
        "surface": "flat",
    }
    resp = client.post("/api/trace", json=_trace_payload(design=design))
    assert resp.status_code < 500
    if resp.status_code == 200:
        body = resp.json()
        assert _all_finite(body)
        assert body["power_w"] >= 0.0
        if body["incident_power_w"]:
            assert body["power_w"] <= body["incident_power_w"] * 1.002


def test_grid_with_flat_facets_traces(client):
    design = {
        "type": "grid",
        "n_u": 2,
        "n_v": 2,
        "facet_w_mm": 1200.0,
        "facet_h_mm": 1000.0,
        "gap_mm": 20.0,
        "facet_focal_mm": 0.0,
        "surface": "twisting",
    }
    resp = client.post("/api/trace", json=_trace_payload(design=design))
    assert resp.status_code == 200
    body = resp.json()
    assert _all_finite(body)
    assert body["power_w"] >= 0.0


def test_field_of_one_matches_single_trace(client):
    single = client.post(
        "/api/trace",
        json=_trace_payload(heliostat_x_mm=0.0, heliostat_y_mm=-90000.0),
    ).json()
    field = client.post(
        "/api/field/trace",
        json={
            **_trace_payload(),
            "layout": {"type": "positions", "xy_mm": [[0.0, -90000.0]]},
        },
    ).json()
    assert field["power_w"] == pytest.approx(single["power_w"], rel=1e-9)
    assert field["n_heliostats"] == 1


def test_sun_below_horizon_geometry_is_not_an_error(client):
    resp = client.post(
        "/api/scene/geometry",
        json={
            "design": RECT_DESIGN,
            "optics": "prime_focus",
            "solar_az_deg": 180.0,
            "solar_el_deg": -5.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sun_below_horizon"] is True


def test_field_trace_over_the_cap_refuses_cleanly(client):
    resp = client.post(
        "/api/field/trace",
        json={**_trace_payload(), "layout": {"type": "fermat", "n": MAX_FIELD_HELIOSTATS + 1}},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]


def test_axicon_at_the_tower_base_is_a_clean_422(client):
    """No radial direction at the origin -- must 422, never 500."""
    resp = client.post(
        "/api/trace",
        json=_trace_payload(optics="axicon", heliostat_x_mm=0.0, heliostat_y_mm=0.0),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]


def test_non_finite_heliostat_position_is_a_clean_422(client):
    import json as _json

    body = _json.dumps(
        {**_trace_payload(), "layout": {"type": "positions", "xy_mm": [[0.0, -90000.0]]}}
    )
    payload = {**_trace_payload(), "layout": {"type": "positions", "xy_mm": [[0.0, -90000.0]]}}
    body = _json.dumps(payload).replace("-90000.0", "NaN")
    resp = client.post(
        "/api/field/trace", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_reflectance_scales_power_exactly(client):
    full = client.post("/api/trace", json=_trace_payload(design={**RECT_DESIGN, "reflectance": 1.0})).json()
    half = client.post("/api/trace", json=_trace_payload(design={**RECT_DESIGN, "reflectance": 0.9})).json()
    assert half["power_w"] == pytest.approx(0.9 * full["power_w"], rel=1e-6)


def test_design_round_trips_through_the_library(client, tmp_path, monkeypatch):
    monkeypatch.setenv("HELIOSTAT_LIBRARY_DIR", str(tmp_path / "library"))
    doc = {"type": "grid", "n_u": 2, "n_v": 2, "facet_w_mm": 1200.0, "facet_h_mm": 1000.0}
    saved = client.post("/api/library/designs", json={"name": "stress-quick-test", "document": doc})
    assert saved.status_code == 200
    loaded = client.get("/api/library/designs/stress-quick-test")
    assert loaded.status_code == 200
    assert loaded.json()["document"] == doc
    deleted = client.delete("/api/library/designs/stress-quick-test")
    assert deleted.status_code == 200


def test_bigger_axicon_aperture_never_collects_less(client):
    small = client.post(
        "/api/trace",
        json=_trace_payload(optics="axicon", optics_params={"aperture_radius_mm": 3000.0}),
    ).json()
    big = client.post(
        "/api/trace",
        json=_trace_payload(optics="axicon", optics_params={"aperture_radius_mm": 20000.0}),
    ).json()
    assert big["power_w"] >= small["power_w"] - 1e-6 * max(1.0, small["power_w"])
