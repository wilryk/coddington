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


# ---------------------------------------------------------------------------
# miss detection -- docs/ui-spec.md 2.3's amber "warning" tier
#
# The manuscript field below is an explicit r_min_m/r_max_m Fermat spiral
# (30-90 m, ui-spec.md 2.2's own "the manuscript field") rather than a bare
# {"type": "fermat", "n": ...}: without radius bounds the layout falls back
# to its own default spiral density (see FermatLayout.resolved_a_m) and
# spreads well past 90 m for n=40, which is not the field these numbers were
# chosen against.

MANUSCRIPT_FIELD = {"type": "fermat", "n": 40, "r_min_m": 30, "r_max_m": 90}
MANUSCRIPT_AXICON = dict(
    optics="axicon", solar_az_deg=165.2, solar_el_deg=61.4, layout=MANUSCRIPT_FIELD
)


def test_manuscript_axicon_field_all_hit(client):
    """Every chief ray in the reference field clears the real 14 m
    aperture: both miss lists are empty, and the aperture that would be
    *needed* is a real number no bigger than the one already in use."""
    resp = client.post("/api/scene/geometry", json=_payload(**MANUSCRIPT_AXICON))
    assert resp.status_code == 200
    miss = resp.json()["miss"]
    assert miss is not None
    assert miss["aperture_miss_ids"] == []
    assert miss["total_miss_ids"] == []
    assert isinstance(miss["needed_aperture_radius_mm"], float)
    assert miss["needed_aperture_radius_mm"] <= 14000.0
    assert miss["rays"] == []


def test_undersized_aperture_reports_a_warning_without_adjusting_anything(client):
    """Shrinking the aperture below what the field needs raises the
    warning tier -- but the geometry stays exactly as requested: the same
    40 heliostats solve and come back, nothing is auto-adjusted."""
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(**{**MANUSCRIPT_AXICON, "optics_params": {"aperture_radius_mm": 8000.0}}),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["heliostats"]) == 40
    miss = data["miss"]
    assert miss["aperture_miss_ids"] != []
    assert miss["needed_aperture_radius_mm"] > 8000.0
    assert miss["rays"] != []
    # Dropped corner rays are the 3-point [source, mirror, extension]
    # overshoot polyline, not the usual 4-point hit path.
    for ray in miss["rays"]:
        assert len(ray) == 3


def test_prime_focus_has_no_miss_tier(client):
    """No secondary at all -- the warning tier does not apply."""
    resp = client.post("/api/scene/geometry", json=_payload(optics="prime_focus"))
    assert resp.status_code == 200
    assert resp.json()["miss"] is None


@pytest.mark.parametrize("solar_el_deg", [0.0, -5.0])
def test_sun_below_horizon_has_no_miss_tier(client, solar_el_deg):
    """No solved orientation below the horizon, so there is no chief ray to
    test -- same "not an error, just nothing to report" treatment as the
    rest of the response."""
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(
            optics="axicon", solar_el_deg=solar_el_deg, layout={"type": "fermat", "n": 6}
        ),
    )
    assert resp.status_code == 200
    assert resp.json()["miss"] is None


def test_total_miss_for_a_heliostat_the_secondary_cannot_reach(client):
    """The near-tower "cannot reach the secondary at all" case (ui-spec.md
    2.3's own example).

    The manuscript axicon (apex 27 m, half angle 20 deg, at its own 165.2/
    61.4 reference sun) turns out NOT to produce this case for any finite
    heliostat position, close to the tower or far past the real field: the
    axicon aiming solve (:func:`~heliostat.geometry.aiming.solve_axicon`)
    derives each heliostat's aim point AS its own intersection with the
    (infinite) cone surface, so the reflected chief ray is geometrically
    guaranteed to cross that surface somewhere, at a radius that stays
    bounded (a few tens of metres at most) even at absurd distances or
    sun angles -- confirmed by sweeping radial position, azimuth and sun
    angle broadly against a 20x-enlarged aperture and finding no miss. The
    only real failure at this geometry is the literal field origin
    (x=y=0), which is a pre-existing 422 from solve_axicon itself
    ("no defined radial direction"), not something this feature classifies.

    A genuine total miss does exist, though, for a steeper cone: a near-
    tower heliostat (radius ~1.5 m) under a fast, low sun and an ~83 degree
    half angle, found by a randomised parameter search. That is the case
    exercised here, to prove the total_miss_ids/needed_aperture_radius_mm
    machinery actually fires when a heliostat truly is stranded, and not
    only that it stays empty for the reference field.
    """
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(
            optics="axicon",
            solar_az_deg=134.72778005224947,
            solar_el_deg=8.99503878837469,
            heliostat_x_mm=-405.55017697219034,
            heliostat_y_mm=-1463.103878595684,
            optics_params={
                "apex_height_mm": 34722.50303425526,
                "half_angle_deg": 82.9688192172392,
                "receiver_z_mm": 9227.607073316292,
            },
        ),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["heliostats"]) == 1
    miss = data["miss"]
    assert miss["total_miss_ids"] == [0]
    assert miss["aperture_miss_ids"] == []
    assert miss["needed_aperture_radius_mm"] is None
    assert miss["rays"] != []
    for ray in miss["rays"]:
        assert len(ray) == 3


def test_cassegrain_also_gets_a_miss_tier(client):
    """The warning tier is not axicon-specific -- Cassegrain has a real
    secondary too."""
    resp = client.post(
        "/api/scene/geometry",
        json=_payload(optics="cassegrain", layout={"type": "fermat", "n": 10}),
    )
    assert resp.status_code == 200
    assert resp.json()["miss"] is not None


# ---------------------------------------------------------------------------
# /api/field/manuscript -- the paper's real 643-heliostat field, verbatim
#
# The app's default field used to REGENERATE a Fermat spiral standing in for
# the paper's own positions; this endpoint serves the actual points from
# examples/paper/data/field_645.csv (packaged as static/data/field_645.csv),
# through the same loader (heliostat.field.load_field) the paper's own
# reproduce.py calls, so the duplicate-drop rule can never disagree with the
# paper's own runs.

PAPER_N_HELIOSTATS = 643

# The two coincident pairs load_field's own distance check finds and drops
# the higher id of (see examples/paper/reproduce.py's PAPER_N_HELIOSTATS
# docstring) -- ids, not row numbers, since load_field assigns ids 0..N-1 in
# file order before dropping anything.
DROPPED_DUPLICATE_PAIRS = [(144, 192), (241, 289)]


def test_manuscript_field_is_200_and_643_heliostats(client):
    resp = client.get("/api/field/manuscript")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n"] == PAPER_N_HELIOSTATS
    assert len(data["xy_mm"]) == PAPER_N_HELIOSTATS
    assert len(data["ids"]) == PAPER_N_HELIOSTATS


def test_manuscript_field_first_position_matches_the_csv(client):
    # CSV row 1: X 2.94051421 m, Y -29.8555418 m -- mm, rounded to 0.1 mm.
    data = client.get("/api/field/manuscript").json()
    x0, y0 = data["xy_mm"][0]
    assert x0 == pytest.approx(2940.5, abs=0.05)
    assert y0 == pytest.approx(-29855.5, abs=0.05)


def test_manuscript_field_drops_exactly_one_id_from_each_duplicate_pair(client):
    data = client.get("/api/field/manuscript").json()
    ids = set(data["ids"])
    for lo, hi in DROPPED_DUPLICATE_PAIRS:
        # The lower id of each coincident pair survives; the higher one is
        # dropped -- never both, never neither.
        assert lo in ids
        assert hi not in ids


def test_manuscript_field_is_cached_and_deterministic(client):
    first = client.get("/api/field/manuscript").json()
    second = client.get("/api/field/manuscript").json()
    assert first == second
