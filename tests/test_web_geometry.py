"""HTTP-level gate for ``/api/scene/geometry`` -- the solve-only, no-trace
scene the 3-D view calls on every edit (docs/ui-spec.md 2.1).

Split out of ``test_web.py`` (already long) rather than appended to it, per
that module's own docstring convention of one file per HTTP surface area.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.web.app import (  # noqa: E402
    MAX_GEOMETRY_HELIOSTATS,
    create_app,
)

RECT_DESIGN = {"type": "rect", "width_mm": 5000, "height_mm": 3000}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _payload(optics="prime_focus", solar_el_deg=45.0, **kw):
    payload = {
        "optics": optics,
        "solar_az_deg": 180.0,
        "solar_el_deg": solar_el_deg,
    }
    payload.update(kw)
    return payload


def test_single_heliostat_geometry(client):
    resp = client.post("/api/scene/geometry", json=_payload())
    assert resp.status_code == 200
    data = resp.json()

    assert data["sun_below_horizon"] is False
    assert len(data["heliostats"]) == 1
    row = data["heliostats"][0]
    assert row["id"] == 0
    assert row["x_mm"] == 0.0
    assert row["y_mm"] == -89609.0
    assert row["rot_az_deg"] is not None
    assert row["rot_el_deg"] is not None

    assert data["outline_local"] is not None
    assert len(data["outline_local"]) == 4  # the default rect's four corners
    assert data["rays_source"] == "corner_chief"
    assert len(data["rays"]) == 4  # one heliostat, four corner rays
    assert data["receiver"] is not None
    assert len(data["sun"]) == 3
    assert data["optics_resolved"]["focus_height_mm"] == pytest.approx(35335.0)


def test_geometry_matches_the_trace_endpoints_solve(client):
    """No trace happens here, but the pointing this endpoint reports must be
    the exact pointing /api/trace would solve and use -- same helper
    (_solve_field / _solve_for), same inputs."""
    geom = client.post(
        "/api/scene/geometry",
        json=_payload(heliostat_x_mm=1000.0, heliostat_y_mm=-70000.0),
    ).json()
    trace = client.post(
        "/api/trace",
        json={
            "design": RECT_DESIGN,
            "mode": "ultra_fast",
            **_payload(heliostat_x_mm=1000.0, heliostat_y_mm=-70000.0),
        },
    ).json()

    # Both endpoints read the identical resolve/solve calls, so the receiver
    # geometry and the resolved tower must agree exactly -- only the picture
    # built on top (facet corners vs. one shared outline) differs.
    assert geom["receiver"] == trace["scene"]["receiver"]
    assert geom["optics_resolved"] == trace["optics_resolved"]
    assert geom["sun"] == trace["scene"]["sun"]


def test_field_geometry_no_trace_fields_leak_in(client):
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "fermat", "n": 12}),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["heliostats"]) == 12
    assert {h["id"] for h in data["heliostats"]} == set(range(12))
    for h in data["heliostats"]:
        assert h["rot_az_deg"] is not None
    # A geometry response has no flux, no power, no counters -- it never ran
    # a trace.
    for absent in ("power_w", "flux_png", "counters", "heliostats_traced"):
        assert absent not in data


def test_corner_rays_can_be_disabled(client):
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "fermat", "n": 5}, include_corner_rays=False),
    )
    assert resp.status_code == 200
    assert resp.json()["rays"] == []


def test_corner_ray_sources_are_strided_when_the_field_exceeds_the_cap(client):
    small = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "fermat", "n": 40}, max_corner_sources=500),
    ).json()
    assert len(small["rays"]) == 4 * 40  # every heliostat contributes

    capped = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "fermat", "n": 40}, max_corner_sources=10),
    ).json()
    # At most 4 rays per sourced heliostat, and at most max_corner_sources
    # heliostats contribute -- a stride, not a truncation of the field.
    assert len(capped["rays"]) <= 4 * 10
    assert len(capped["rays"]) > 0
    assert len(capped["heliostats"]) == 40  # every mirror is still placed


def test_ten_thousand_heliostats_is_within_the_geometry_cap(client):
    """The whole point of this endpoint: a field ten times the trace cap,
    placed and oriented, still answers -- and quickly, since nothing here
    shades, blocks or traces."""
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(
            layout={"type": "fermat", "n": MAX_GEOMETRY_HELIOSTATS},
            max_corner_sources=50,
        ),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["heliostats"]) == MAX_GEOMETRY_HELIOSTATS
    assert len(data["rays"]) <= 4 * 50


def test_over_the_geometry_cap_is_422(client):
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "fermat", "n": MAX_GEOMETRY_HELIOSTATS + 1}),
    )
    assert resp.status_code == 422


def test_sun_below_horizon_is_not_an_error(client):
    """Unlike a trace, a non-positive elevation returns 200: the scene never
    goes blank (docs/ui-spec.md 2.1)."""
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(solar_el_deg=0.0, layout={"type": "fermat", "n": 6}),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sun_below_horizon"] is True
    assert data["rays"] == []
    assert len(data["heliostats"]) == 6
    for h in data["heliostats"]:
        assert h["rot_az_deg"] is None
        assert h["rot_el_deg"] is None
        # Positions are still reported -- only the orientation is unknown.
        assert h["x_mm"] is not None and h["y_mm"] is not None


def test_negative_elevation_is_also_not_an_error(client):
    resp = client.post("/api/scene/geometry", json=_payload(solar_el_deg=-5.0))
    assert resp.status_code == 200
    assert resp.json()["sun_below_horizon"] is True


def test_elevation_past_straight_up_is_still_a_422(client):
    resp = client.post("/api/scene/geometry", json=_payload(solar_el_deg=95.0))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"focus_height_mm": -1.0},
        {"focus_height_mm": 0.0},
    ],
)
def test_bad_optics_params_is_422_same_style_as_trace(client, params):
    geom = client.post("/api/scene/geometry", json=_payload(optics_params=params))
    trace = client.post(
        "/api/trace",
        json={"design": RECT_DESIGN, "mode": "ultra_fast", **_payload(optics_params=params)},
    )
    assert geom.status_code == 422
    assert trace.status_code == 422
    assert geom.json()["detail"] == trace.json()["detail"]


def test_moved_optics_params_reach_the_receiver_and_scene(client):
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(optics_params={"focus_height_mm": 30000.0}),
    )
    data = resp.json()
    assert data["optics_resolved"]["focus_height_mm"] == 30000.0
    assert data["receiver"]["z_mm"] == 30000.0


def test_axicon_and_cassegrain_resolve_a_secondary(client):
    for optics in ("axicon", "cassegrain"):
        data = client.post("/api/scene/geometry", json=_payload(optics=optics)).json()
        assert data["secondary"] is not None
        assert data["secondary"]["kind"] == optics


def test_geometry_is_deterministic(client):
    payload = _payload(layout={"type": "fermat", "n": 8})
    first = client.post("/api/scene/geometry", json=payload).json()
    second = client.post("/api/scene/geometry", json=payload).json()
    assert first == second


def test_positions_layout_round_trips_through_geometry(client):
    generated = client.post(
        "/api/scene/geometry", json=_payload(layout={"type": "fermat", "n": 6})
    ).json()
    xy = [[h["x_mm"], h["y_mm"]] for h in generated["heliostats"]]
    replayed = client.post(
        "/api/scene/geometry",
        json=_payload(layout={"type": "positions", "xy_mm": xy}),
    ).json()
    assert [(h["x_mm"], h["y_mm"]) for h in replayed["heliostats"]] == [
        (h["x_mm"], h["y_mm"]) for h in generated["heliostats"]
    ]


def test_default_design_is_the_legacy_rectangle(client):
    """An absent `design` still shows something -- the same 5x3 m default
    /api/trace's own request model hard-codes for a bare position."""
    resp = client.post("/api/scene/geometry", json=_payload())
    assert resp.status_code == 200
    assert len(resp.json()["outline_local"]) == 4


def test_bad_design_type_is_422(client):
    resp = client.post("/api/scene/geometry", json=_payload(design={"type": "hexagon"}))
    assert resp.status_code == 422
