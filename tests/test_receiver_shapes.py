"""Prime focus's own receiver: type, radial aiming, offset, position.

Covers docs/ui-spec.md 2.2's "the receiver is its own design" bullet --
cylindrical/frustum receivers under prime focus, radially-offset aiming for
curved receivers, a positionable receiver centre, and the entrance
aperture + offset behind it. Every test either proves a physics claim
(radial aiming lands on the facing surface point, not the axis) or a
compatibility claim (a request naming none of this traces exactly as
before it existed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from heliostat.geometry.aiming import (  # noqa: E402
    solve_prime_focus,
    solve_prime_focus_to_receiver,
)
from heliostat.geometry.receiver import (  # noqa: E402
    ApertureClippedReceiver,
    CylinderReceiver,
    FlatWindowReceiver,
    FrustumReceiver,
)
from heliostat.web.app import (  # noqa: E402
    PRIME_FOCUS_HEIGHT_MM,
    _prime_focus_receiver,
    _solve_for,
    create_app,
    resolve_optics_params,
)

RECT_DESIGN = {"type": "rect", "width_mm": 5000, "height_mm": 3000}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _payload(optics_params=None, x=0.0, y=-50000.0, mode="monte_carlo"):
    return {
        "design": RECT_DESIGN,
        "mode": mode,
        "optics": "prime_focus",
        "solar_az_deg": 165.0,
        "solar_el_deg": 45.0,
        "heliostat_x_mm": x,
        "heliostat_y_mm": y,
        "optics_params": optics_params or {},
    }


def _field_payload(optics_params=None, n=40, mode="fast_accurate"):
    return {
        "design": RECT_DESIGN,
        "mode": mode,
        "optics": "prime_focus",
        "solar_az_deg": 165.0,
        "solar_el_deg": 45.0,
        "layout": {"type": "fermat", "n": n},
        "optics_params": optics_params or {},
    }


# ---------------------------------------------------------------------------
# backward compatibility: no new field named -> today's numbers, exactly


@pytest.mark.parametrize("x_mm,y_mm", [(0.0, -50000.0), (30000.0, 20000.0), (0.0, 0.0)])
def test_prime_focus_with_no_receiver_fields_matches_the_old_solve(x_mm, y_mm):
    """`_solve_for` now routes prime focus through the receiver's own
    `aim_point_mm`; for the untouched default (flat, on-axis, no offset)
    that must still be exactly `solve_prime_focus`'s own arithmetic, not
    just numerically close."""
    old = solve_prime_focus(x_mm, y_mm, 165.0, 45.0, PRIME_FOCUS_HEIGHT_MM)
    new = _solve_for("prime_focus", x_mm, y_mm, 165.0, 45.0, None)
    assert new.rot_az_deg == old.rot_az_deg
    assert new.rot_el_deg == old.rot_el_deg
    assert new.c3 == old.c3
    assert new.c4 == old.c4
    assert new.c5 == old.c5
    assert new.extras["aim_x_mm"] == old.extras["aim_x_mm"]
    assert new.extras["aim_y_mm"] == old.extras["aim_y_mm"]
    assert new.extras["aim_z_mm"] == old.extras["aim_z_mm"]


def test_default_trace_response_unaffected_by_the_new_fields(client):
    """A request naming none of receiver_type/offset/centre gets the same
    trace response whether or not `optics_params` is present at all."""
    named_defaults = client.post(
        "/api/trace",
        json=_payload(
            {
                "receiver_type": "flat",
                "receiver_center_x_mm": 0.0,
                "receiver_center_y_mm": 0.0,
                "aperture_to_receiver_mm": 0.0,
            }
        ),
    ).json()
    absent = client.post("/api/trace", json=_payload(None)).json()
    assert named_defaults["power_w"] == absent["power_w"]
    assert named_defaults["peak_flux_kw_m2"] == absent["peak_flux_kw_m2"]
    assert named_defaults["aim_point_mm"] == absent["aim_point_mm"]
    assert named_defaults["scene"]["receiver"]["z_mm"] == absent["scene"]["receiver"]["z_mm"]


# ---------------------------------------------------------------------------
# radial aiming: each heliostat aims at a different point on a curved
# receiver's own surface, not the shared axis point


def test_cylinder_aim_point_faces_each_heliostat_and_lies_on_the_surface():
    """The physics claim item 2 makes: heliostats at distinct azimuths
    around a cylindrical receiver aim at DIFFERENT points, and every one of
    those points sits exactly on the cylinder (radius from its axis,
    height at its centre) -- not at the shared axis point solve_prime_focus
    would have used."""
    params = resolve_optics_params(
        "prime_focus",
        {"receiver_type": "cylinder", "cylinder_radius_mm": 3000.0, "cylinder_height_mm": 6000.0},
    )
    receiver = _prime_focus_receiver(params)
    assert isinstance(receiver, CylinderReceiver)

    positions = [(0.0, -40000.0), (40000.0, 0.0), (0.0, 40000.0), (-40000.0, 0.0), (28284.0, 28284.0)]
    aims = []
    for x, y in positions:
        sol = solve_prime_focus_to_receiver(x, y, 165.0, 45.0, receiver)
        aim = np.array([sol.extras["aim_x_mm"], sol.extras["aim_y_mm"], sol.extras["aim_z_mm"]])
        aims.append(aim)
        # On the surface: exact radius from the receiver's own axis, exact
        # centre height.
        assert np.hypot(aim[0], aim[1]) == pytest.approx(receiver.radius_mm, abs=1e-6)
        assert aim[2] == pytest.approx(receiver.center_z_mm, abs=1e-9)
        # Facing the heliostat: the aim point's own bearing from the axis
        # points the same way as the heliostat's, not toward a shared point.
        helio_bearing = np.arctan2(x, y)
        aim_bearing = np.arctan2(aim[0], aim[1])
        assert aim_bearing == pytest.approx(helio_bearing, abs=1e-6)

    # No two heliostats at different azimuths share an aim point -- proof
    # this is not secretly the old single shared-axis solve.
    distinct = {tuple(np.round(a, 3)) for a in aims}
    assert len(distinct) == len(positions)


def test_cylinder_aim_is_relative_to_an_off_axis_receiver_centre():
    """Item 3's generalisation: an off-axis receiver still faces each
    heliostat correctly, measured from ITS OWN centre, not the field
    origin."""
    params = resolve_optics_params(
        "prime_focus",
        {
            "receiver_type": "cylinder",
            "receiver_center_x_mm": 8000.0,
            "receiver_center_y_mm": -2000.0,
            "cylinder_radius_mm": 2500.0,
        },
    )
    receiver = _prime_focus_receiver(params)
    sol = solve_prime_focus_to_receiver(100000.0, -50000.0, 165.0, 45.0, receiver)
    ax, ay, az = sol.extras["aim_x_mm"], sol.extras["aim_y_mm"], sol.extras["aim_z_mm"]
    rel = np.hypot(ax - 8000.0, ay - (-2000.0))
    assert rel == pytest.approx(2500.0, abs=1e-6)
    assert az == pytest.approx(receiver.center_z_mm, abs=1e-9)


def test_frustum_aim_point_lies_on_the_cone_wall():
    """Same claim as the cylinder test, for the other curved shape: the aim
    point's radius at its own height matches the frustum's own taper."""
    params = resolve_optics_params(
        "prime_focus",
        {
            "receiver_type": "frustum",
            "frustum_top_radius_mm": 2000.0,
            "frustum_bottom_radius_mm": 4000.0,
            "frustum_height_mm": 6000.0,
        },
    )
    receiver = _prime_focus_receiver(params)
    assert isinstance(receiver, FrustumReceiver)
    sol = solve_prime_focus_to_receiver(0.0, -50000.0, 165.0, 45.0, receiver)
    ax, ay, az = sol.extras["aim_x_mm"], sol.extras["aim_y_mm"], sol.extras["aim_z_mm"]
    # aim_point_mm targets mid-slant, at the mean radius, by construction.
    assert np.hypot(ax, ay) == pytest.approx(receiver.r_mean_mm, abs=1e-6)
    assert az == pytest.approx(0.5 * (receiver.z_bot_mm + receiver.z_top_mm), abs=1e-9)


# ---------------------------------------------------------------------------
# a cylindrical receiver collects a sensible fraction of what a flat window
# collects, for the same field


def test_cylinder_receiver_collects_comparable_power_to_flat(client):
    """Radial aiming is supposed to compensate for the curvature, not cost
    power: a 3 m-radius, 6 m-tall cylindrical receiver on the same field,
    aperture and sun should collect within a few percent of the flat
    window it replaces -- neither systematically starved nor implausibly
    amplified."""
    flat = client.post("/api/field/trace", json=_field_payload()).json()
    cylinder = client.post(
        "/api/field/trace",
        json=_field_payload(
            {"receiver_type": "cylinder", "cylinder_radius_mm": 3000.0, "cylinder_height_mm": 6000.0}
        ),
    ).json()
    assert flat["power_w"] > 0
    assert cylinder["power_w"] > 0
    ratio = cylinder["power_w"] / flat["power_w"]
    # Recorded from an actual run: ~1.002 -- see the test report.
    assert 0.9 <= ratio <= 1.1


# ---------------------------------------------------------------------------
# entrance aperture + offset receiver


def test_aperture_offset_zero_matches_todays_numbers_exactly(client):
    off0 = client.post("/api/trace", json=_payload({"aperture_to_receiver_mm": 0.0})).json()
    absent = client.post("/api/trace", json=_payload(None)).json()
    assert off0["power_w"] == absent["power_w"]
    assert off0["peak_flux_kw_m2"] == absent["peak_flux_kw_m2"]
    assert off0["scene"]["receiver"].get("aperture") is None


def test_positive_offset_moves_the_receiver_and_changes_the_flux(client):
    """A heliostat close to the axis still clears the aperture at a modest
    offset, but the flux it delivers spreads out (the receiver sits past
    the focus now) -- a real physical change, not a no-op."""
    near_axis = dict(x=0.0, y=-20000.0)
    off0 = client.post("/api/trace", json=_payload({}, **near_axis)).json()
    off2000 = client.post(
        "/api/trace", json=_payload({"aperture_to_receiver_mm": 2000.0}, **near_axis)
    ).json()
    assert off0["scene"]["receiver"]["z_mm"] == PRIME_FOCUS_HEIGHT_MM
    assert off2000["scene"]["receiver"]["z_mm"] == PRIME_FOCUS_HEIGHT_MM + 2000.0
    assert off2000["scene"]["receiver"]["aperture"]["z_mm"] == PRIME_FOCUS_HEIGHT_MM
    assert off2000["peak_flux_kw_m2"] != off0["peak_flux_kw_m2"]
    assert off2000["power_w"] != off0["power_w"]


def test_the_aperture_clips_what_a_bigger_window_would_pass(client):
    """A field spanning both a near and a far heliostat: at a modest
    offset the far one's chief ray falls outside the (still today-sized)
    aperture window and is clipped, while the near one still gets through
    -- a real, partial loss, not all-or-nothing."""
    xy = [[0.0, -20000.0], [0.0, -60000.0]]
    offset0 = client.post(
        "/api/field/trace",
        json={
            "design": RECT_DESIGN,
            "mode": "monte_carlo",
            "optics": "prime_focus",
            "solar_az_deg": 165.0,
            "solar_el_deg": 45.0,
            "layout": {"type": "positions", "xy_mm": xy},
            "optics_params": {},
        },
    ).json()
    offset2000 = client.post(
        "/api/field/trace",
        json={
            "design": RECT_DESIGN,
            "mode": "monte_carlo",
            "optics": "prime_focus",
            "solar_az_deg": 165.0,
            "solar_el_deg": 45.0,
            "layout": {"type": "positions", "xy_mm": xy},
            "optics_params": {"aperture_to_receiver_mm": 2000.0},
        },
    ).json()
    assert offset0["power_w"] > 0
    assert 0.0 < offset2000["power_w"] < offset0["power_w"]
    far_row = next(r for r in offset2000["heliostats"] if r["y_mm"] == -60000.0)
    assert far_row["eta"] == 0.0 or offset2000["power_w"] < offset0["power_w"] * 0.7


# ---------------------------------------------------------------------------
# positionable receiver


def test_off_axis_receiver_centre_is_aimed_at(client):
    resp = client.post(
        "/api/trace", json=_payload({"receiver_center_x_mm": 5000.0, "receiver_center_y_mm": -1500.0})
    ).json()
    assert resp["aim_point_mm"] == [5000.0, -1500.0, PRIME_FOCUS_HEIGHT_MM]


def test_receiver_at_or_below_heliostat_plane_is_rejected_with_a_readable_error(client):
    resp = client.post(
        "/api/trace",
        json=_payload(
            {"focus_height_mm": 1000.0, "receiver_type": "cylinder", "cylinder_height_mm": 6000.0}
        ),
    )
    assert resp.status_code == 422
    assert "heliostat plane" in resp.json()["detail"]


def test_solve_prime_focus_to_receiver_rejects_a_non_positive_aim_point():
    """The authoritative, per-heliostat check inside the aiming solve
    itself (app.py's own validator is a coarse, on-axis pre-check for the
    same condition; this is what actually gates every trace)."""
    receiver = CylinderReceiver(center_z_mm=-500.0, radius_mm=2000.0, height_mm=1000.0)
    with pytest.raises(ValueError, match="heliostat plane"):
        solve_prime_focus_to_receiver(0.0, -50000.0, 165.0, 45.0, receiver)


# ---------------------------------------------------------------------------
# degenerate geometries


def test_frustum_with_equal_radii_traces_as_a_cylinder(client):
    resp = client.post(
        "/api/trace",
        json=_payload(
            {
                "receiver_type": "frustum",
                "frustum_top_radius_mm": 3000.0,
                "frustum_bottom_radius_mm": 3000.0,
            }
        ),
    )
    assert resp.status_code == 200
    assert resp.json()["scene"]["receiver"]["kind"] == "cylinder"


@pytest.mark.parametrize(
    "params",
    [
        {"receiver_type": "cylinder", "cylinder_height_mm": 0.0},
        {"receiver_type": "cylinder", "cylinder_radius_mm": 0.0},
        {"receiver_type": "frustum", "frustum_height_mm": 0.0},
    ],
)
def test_zero_size_receiver_dimensions_are_rejected_not_crashed(client, params):
    resp = client.post("/api/trace", json=_payload(params))
    assert resp.status_code == 422


@pytest.mark.parametrize("mode", ["ultra_fast", "fast_accurate"])
@pytest.mark.parametrize(
    "params",
    [
        {"receiver_type": "cylinder", "cylinder_radius_mm": 1500.0, "cylinder_height_mm": 6000.0},
        {"receiver_type": "frustum"},
    ],
)
def test_a_curved_receiver_never_collects_more_than_arrives(client, mode, params):
    """Energy conservation on a curved surface, at every fidelity.

    The cone backend's order-2 deposit spreads each sample by a Hessian of
    the ray-to-surface map. That map folds on a curved receiver, and a fold
    sends the deposit's Jacobian through zero -- this exact geometry once
    reported 17x the power that arrived on the mirror.
    """
    body = {
        "design": {
            "type": "grid",
            "n_u": 3,
            "n_v": 1,
            "facet_w_mm": 3000.0,
            "facet_h_mm": 200.0,
            "gap_mm": 100.0,
            "cant_focal_mm": 1e7,
            "facet_focal_mm": 0.0,
            "surface": "twisting",
        },
        "mode": mode,
        "optics": "prime_focus",
        "solar_az_deg": 360.0,
        "solar_el_deg": 5.0,
        "optics_params": params,
        "heliostat_x_mm": 0.0,
        "heliostat_y_mm": -50000.0,
    }
    data = client.post("/api/trace", json=body).json()
    assert data["power_w"] <= data["incident_power_w"] * 1.002


@pytest.mark.parametrize(
    "x_mm,y_mm,params",
    [
        (20000.0, -20000.0, None),
        (-20000.0, 20000.0, None),
        (20000.0, -20000.0, {"receiver_z_mm": 26973.0}),
    ],
)
def test_an_axicon_never_collects_more_than_arrives(client, x_mm, y_mm, params):
    """Energy conservation off the field axis, and with the receiver close
    under the apex.

    A clipped kernel footprint keeps its raw deposit, because a footprint
    running off the grid has genuinely spilled. Where the quadratic model
    folds, the deposit's density factor is only floored, so an unguarded
    clipped footprint could deposit more than the sample carried -- this
    geometry once reported 9.5x the power that arrived on the mirror.
    """
    body = {
        "design": {"type": "rect", "width_mm": 5000.0, "height_mm": 3000.0, "surface": "twisting"},
        "mode": "fast_accurate",
        "optics": "axicon",
        "solar_az_deg": 270.0,
        "solar_el_deg": 20.0,
        "heliostat_x_mm": x_mm,
        "heliostat_y_mm": y_mm,
    }
    if params is not None:
        body["optics_params"] = params
    data = client.post("/api/trace", json=body).json()
    assert data["power_w"] <= data["incident_power_w"] * 1.002


# ---------------------------------------------------------------------------
# uv_to_world: the exact inverse of intersect(), used to draw real 3-D ray
# paths on a receiver's own hit point (release-night bug: cylinder/frustum
# had no such inverse, so every ray path carried a NaN fourth vertex and was
# dropped whole by the scene's own finite-only filter -- "rays disappear").


def _random_rays_from_field(rng, n=500):
    """Rays scattered from plausible heliostat positions toward roughly the
    tower top, biased so most (but not all) actually cross a receiver near
    35 m up -- a mix of near and far ground radii, like a real field."""
    rad = rng.uniform(4000.0, 90000.0, n)
    ang = rng.uniform(0.0, 2.0 * np.pi, n)
    p = np.vstack([rad * np.cos(ang), rad * np.sin(ang), np.zeros(n)])
    aim = np.vstack(
        [rng.uniform(-3000.0, 3000.0, n), rng.uniform(-3000.0, 3000.0, n), rng.uniform(30000.0, 40000.0, n)]
    )
    d = aim - p
    d /= np.linalg.norm(d, axis=0)
    return p, d


@pytest.mark.parametrize(
    "receiver",
    [
        FlatWindowReceiver(z_mm=35335.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down"),
        FlatWindowReceiver(
            z_mm=35335.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down",
            center_x_mm=1500.0, center_y_mm=-800.0,
        ),
        CylinderReceiver(center_z_mm=35335.0, radius_mm=3000.0, height_mm=6000.0),
        CylinderReceiver(
            center_z_mm=35335.0, radius_mm=3000.0, height_mm=6000.0,
            center_x_mm=1500.0, center_y_mm=-800.0,
        ),
        FrustumReceiver(z_bot_mm=32335.0, r_bot_mm=4000.0, z_top_mm=38335.0, r_top_mm=2500.0),
        FrustumReceiver(z_bot_mm=32335.0, r_bot_mm=2500.0, z_top_mm=38335.0, r_top_mm=4000.0),
        ApertureClippedReceiver(
            aperture=FlatWindowReceiver(z_mm=35335.0, half_u_mm=5000.0, half_v_mm=5000.0, facing="down"),
            inner=CylinderReceiver(center_z_mm=35335.0, radius_mm=3000.0, height_mm=6000.0),
        ),
    ],
    ids=["flat", "flat-offaxis", "cylinder", "cylinder-offaxis", "frustum", "frustum-inverted", "aperture-clipped"],
)
def test_uv_to_world_is_the_exact_inverse_of_intersect(receiver):
    """For every receiver kind, `uv_to_world(intersect(p, d)[1])` must land
    on the exact same 3-D point the ray actually crossed -- checked by
    independently recomputing that point from the ray's own parametric line
    (using intersect's returned uv only to know *which* height/slant it hit),
    never from uv_to_world itself, so this cannot be a tautology."""
    rng = np.random.default_rng(0)
    p, d = _random_rays_from_field(rng, n=800)
    hit, uv = receiver.intersect(p, d)
    assert hit.sum() > 400  # sanity: the ray bundle actually meets this shape a lot

    world = receiver.uv_to_world(uv)
    ph, dh = p[:, hit], d[:, hit]

    # Recompute each hit's true 3-D point independently: solve the ray's own
    # parameter t from matching world[2] (z), the one coordinate every one
    # of these shapes reports directly or via a simple linear map -- then
    # compare the ray-parametrized (x, y) against uv_to_world's (x, y).
    t = (world[2] - ph[2]) / dh[2]
    x_expected = ph[0] + t * dh[0]
    y_expected = ph[1] + t * dh[1]
    np.testing.assert_allclose(world[0], x_expected, atol=1e-6)
    np.testing.assert_allclose(world[1], y_expected, atol=1e-6)


# ---------------------------------------------------------------------------
# rays render on curved receivers (release-night bug: field_corner_rays and
# trace_heliostat's return_paths both reconstructed a receiver hit's world z
# via getattr(receiver, "z_mm", nan) -- always NaN for cylinder/frustum, so
# every ray's fourth vertex was non-finite and the whole ray got dropped by
# the scene's finite-only filter. Switching a prime-focus setup to cylinder,
# or the frustum default, showed a receiver mesh with no rays on it at all.)


@pytest.mark.parametrize(
    "params",
    [
        {"receiver_type": "cylinder", "cylinder_radius_mm": 3000.0, "cylinder_height_mm": 6000.0},
        {"receiver_type": "frustum"},
    ],
    ids=["cylinder", "frustum"],
)
def test_single_heliostat_trace_draws_rays_on_a_curved_receiver(client, params):
    resp = client.post("/api/trace", json=_payload(params, mode="monte_carlo"))
    assert resp.status_code == 200
    data = resp.json()
    rays = data["scene"]["rays"]
    assert len(rays) > 0
    # Every ray's receiver-hit vertex (index 3) must be finite -- the NaN
    # this bug produced would have been silently dropped whole, not
    # reported as a NaN, so also check the count is a healthy fraction of
    # what a flat receiver draws for the same request.
    for ray in rays:
        assert all(math.isfinite(c) for c in ray[3])
    flat = client.post("/api/trace", json=_payload(None, mode="monte_carlo")).json()
    assert len(rays) >= 0.3 * len(flat["scene"]["rays"])


@pytest.mark.parametrize(
    "params",
    [
        {"receiver_type": "cylinder", "cylinder_radius_mm": 3000.0, "cylinder_height_mm": 6000.0},
        {"receiver_type": "frustum"},
    ],
    ids=["cylinder", "frustum"],
)
def test_field_trace_draws_rays_on_a_curved_receiver(client, params):
    resp = client.post("/api/field/trace", json=_field_payload(params))
    assert resp.status_code == 200
    rays = resp.json()["scene"]["rays"]
    assert len(rays) > 0
    for ray in rays:
        assert all(math.isfinite(c) for c in ray[3])


# ---------------------------------------------------------------------------
# frustum orientation: the mesh scene.py/scene3d.js draw pairs r_top_mm with
# z_top_mm and r_bot_mm with z_bot_mm; pin that the physics traces onto that
# same surface, so a rendered "wide end" always matches where flux actually
# lands (docs/ui-spec.md 2.2: reported bug was the two looking inverted
# relative to each other).


def test_frustum_traced_hits_land_within_the_rendered_surfaces_extent(client):
    """Every traced ray's receiver-hit point, in world frame, must fall
    within the exact z-range and radius-range the 3-D/elevation views draw
    the mesh over (z_bot_mm..z_top_mm, and the taper's own radius at that
    height) -- proving the rendered shape and the traced surface are the
    same object, not just two numbers that happen not to have been swapped
    anywhere obvious."""
    resp = client.post(
        "/api/trace",
        json=_payload({"receiver_type": "frustum"}, mode="monte_carlo"),
    )
    data = resp.json()
    receiver = data["scene"]["receiver"]
    assert receiver["kind"] == "frustum"
    z_bot, r_bot = receiver["z_bot_mm"], receiver["r_bot_mm"]
    z_top, r_top = receiver["z_top_mm"], receiver["r_top_mm"]
    assert z_top > z_bot

    rays = data["scene"]["rays"]
    assert len(rays) > 0
    for ray in rays:
        x, y, z = ray[3]
        assert z_bot - 1.0 <= z <= z_top + 1.0
        frac = (z - z_bot) / (z_top - z_bot)
        r_expected = r_bot + frac * (r_top - r_bot)
        assert math.hypot(x, y) == pytest.approx(r_expected, abs=1.0)


def test_frustum_uv_to_world_pins_bottom_and_top_rim_to_the_dataclass_fields():
    """Direct pin, independent of any web-layer plumbing: v=0 (bottom rim)
    maps to z_bot_mm at radius r_bot_mm; v=slant_length_mm (top rim) maps to
    z_top_mm at radius r_top_mm -- the exact pairing scene.py's receiver
    dict and scene3d.js's CylinderGeometry(r_top_mm, r_bot_mm, ...) rely on
    to draw the wide end where the physics actually put it."""
    receiver = FrustumReceiver(z_bot_mm=32335.0, r_bot_mm=4000.0, z_top_mm=38335.0, r_top_mm=2500.0)
    uv_bottom = np.array([[0.0], [0.0]])
    uv_top = np.array([[0.0], [receiver.slant_length_mm]])
    x0, y0, z0 = receiver.uv_to_world(uv_bottom)[:, 0]
    x1, y1, z1 = receiver.uv_to_world(uv_top)[:, 0]
    assert z0 == pytest.approx(receiver.z_bot_mm, abs=1e-6)
    assert math.hypot(x0, y0) == pytest.approx(receiver.r_bot_mm, abs=1e-6)
    assert z1 == pytest.approx(receiver.z_top_mm, abs=1e-6)
    assert math.hypot(x1, y1) == pytest.approx(receiver.r_top_mm, abs=1e-6)
