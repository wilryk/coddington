"""Tests for the exact polygon-projection occlusion path in
:mod:`heliostat.geometry.shading` (:func:`shadow_quad_uv`, :func:`block_quad_uv`,
:func:`polygon_occlusion`).

Three kinds of coverage, mirroring ``test_shading.py``'s structure:

(a) An analytic axis-aligned case where the projected/clipped polygon's area
    has a closed form -- checked *before* any rasterisation, so it isolates
    the projection + clip from the raster-quantisation step that follows.
(b) A fixture-scene comparison: :func:`polygon_occlusion` against both the
    existing 25x15 point grid (:func:`shading_blocking`/
    :func:`occlusion_efficiency`, the private-repo-matching reference) and a
    much finer 200x120 point grid, on the same real scene
    ``tests/fixtures/shading`` uses.
(c) A convergent-blocking case showing :func:`block_quad_uv`'s central
    projection agrees with the per-point converging reference where a
    parallel projection (:func:`shadow_quad_uv` misused with a fixed
    direction) does not -- the same point this codebase already makes in
    ``test_blocking_uses_per_point_direction_and_it_matters``, now for the
    polygon path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heliostat.field import HeliostatField, neighbour_pairs
from heliostat.geometry.shading import (
    MirrorGeometry,
    SecondaryCone,
    _fraction_unoccluded,
    _points_in_polygon,
    _polygon_area,
    _sutherland_hodgman,
    block_quad_uv,
    build_geometries,
    occlusion_efficiency,
    polygon_occlusion,
    shading_blocking,
    shadow_quad_uv,
)

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
SHADING_ROOT = FIXTURES_ROOT / "shading"


# ---------------------------------------------------------------------------
# (a) Analytic rectangle-overlap case
# ---------------------------------------------------------------------------


def _axis_aligned_mirror(cx: float, cy: float, cz: float, hw: float, hh: float) -> MirrorGeometry:
    """A mirror/occluder with normal exactly ``+z``, built by hand (no trig)
    so the projection below is exact to machine precision, not just to the
    precision of ``sun_vector``'s float trig evaluated near a right angle."""
    return MirrorGeometry(
        centre=np.array([cx, cy, cz]),
        normal=np.array([0.0, 0.0, 1.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        half_width=hw,
        half_height=hh,
    )


def test_shadow_quad_clip_area_matches_closed_form_before_rasterising():
    """Sun straight overhead, occluder directly above and offset in x so its
    shadow overlaps the mirror in a simple axis-aligned rectangle.

    Mirror: [-2500, 2500] x [-1500, 1500]. Occluder same size, centred at
    x=3000, so its shadow (straight down, since the sun is overhead) is
    [500, 5500] x [-1500, 1500]. Overlap with the mirror is exactly
    [500, 2500] x [-1500, 1500] = 2000 x 3000 = 6,000,000 mm^2 -- a closed
    form independent of any sampling resolution.
    """
    mirror = _axis_aligned_mirror(0.0, 0.0, 0.0, 2500.0, 1500.0)
    occ = _axis_aligned_mirror(3000.0, 0.0, 10000.0, 2500.0, 1500.0)
    to_sun = np.array([0.0, 0.0, 1.0])

    quad = shadow_quad_uv(occ, mirror, to_sun)
    assert quad is not None
    clipped = _sutherland_hodgman(quad, mirror.half_width, mirror.half_height)

    exact_area = 2000.0 * 3000.0
    area = _polygon_area(clipped)
    assert area == pytest.approx(exact_area, abs=1e-6), (
        f"clipped polygon area {area} vs closed form {exact_area} -- mismatch far "
        "above float noise, before any raster step"
    )

    # Now rasterise (via the same machinery polygon_occlusion uses) and check
    # eta_shade against the closed-form fraction, at a tolerance set by the
    # raster resolution rather than by the (exact) polygon step above.
    n_u, n_v = 2000, 1200
    su = (np.arange(n_u) + 0.5) / n_u * 2.0 - 1.0
    sv = (np.arange(n_v) + 0.5) / n_v * 2.0 - 1.0
    a, b = np.meshgrid(su * mirror.half_width, sv * mirror.half_height, indexing="ij")
    local_u, local_v = a.ravel(), b.ravel()
    shaded = _points_in_polygon(local_u, local_v, clipped)

    exact_eta = 1.0 - exact_area / ((2 * mirror.half_width) * (2 * mirror.half_height))
    got_eta = 1.0 - shaded.mean()
    assert got_eta == pytest.approx(exact_eta, abs=2.0 / min(n_u, n_v))


def test_shadow_quad_uv_none_when_occluder_behind():
    """The occluder-behind-the-mirror case from self_check, restated for the
    polygon path: t <= 0 for every corner must return None, not a bogus quad."""
    mirror = _axis_aligned_mirror(0.0, 0.0, 0.0, 2500.0, 1500.0)
    behind = _axis_aligned_mirror(0.0, 0.0, -10000.0, 20000.0, 20000.0)
    to_sun = np.array([0.0, 0.0, 1.0])
    assert shadow_quad_uv(behind, mirror, to_sun) is None


def test_block_quad_uv_none_beyond_aim_point():
    """An occluder farther from the mirror than the aim point (order
    mirror -> aim -> occluder) cannot block the beam -- the beam terminates
    at the aim -- and must come back None, not a quad the caller would
    wrongly rasterise as blocking."""
    mirror = _axis_aligned_mirror(0.0, 0.0, 0.0, 2500.0, 1500.0)
    aim = np.array([0.0, 0.0, 5000.0])
    far_occ = _axis_aligned_mirror(0.0, 0.0, 10000.0, 2500.0, 1500.0)
    assert block_quad_uv(far_occ, mirror, aim) is None

    near_occ = _axis_aligned_mirror(500.0, 0.0, 2000.0, 2500.0, 1500.0)
    assert block_quad_uv(near_occ, mirror, aim) is not None


# ---------------------------------------------------------------------------
# (b) Fixture-scene comparison
# ---------------------------------------------------------------------------


def _load_shading_scene(step_key: str):
    expected = json.loads((SHADING_ROOT / "expected.json").read_text(encoding="utf-8"))
    occlusion = pd.read_csv(SHADING_ROOT / "occlusion.csv")
    geometry = pd.read_csv(SHADING_ROOT / "geometry.csv")

    group = geometry[geometry["step_key"] == step_key].reset_index(drop=True)
    field = HeliostatField(
        x_mm=group["x_mm"].to_numpy(float),
        y_mm=group["y_mm"].to_numpy(float),
        ids=group["heliostat_id"].to_numpy(int),
        mirror_width_mm=expected["mirror_width_mm"],
        mirror_height_mm=expected["mirror_height_mm"],
    )
    neighbours = neighbour_pairs(field, expected["search_radius_mm"])
    rot_az = group["rot_az_deg"].to_numpy(float)
    rot_el = group["rot_el_deg"].to_numpy(float)
    aims = group[["aim_x_mm", "aim_y_mm", "aim_z_mm"]].to_numpy(float)
    geoms, aim_points = build_geometries(
        field, rot_az, rot_el, aims, pedestal_height_mm=expected["pedestal_height_mm"]
    )
    sun_row = next(s for label, s in expected["sun_positions"].items() if s["key"] == step_key)
    secondary = SecondaryCone(
        z_tip_mm=expected["secondary"]["z_tip_mm"],
        angle_deg=expected["secondary"]["angle_deg"],
        aperture_radius_mm=expected["secondary"]["aperture_radius_mm"],
    )
    want = occlusion[occlusion["step_key"] == step_key].set_index("heliostat_id")
    return field, geoms, aim_points, neighbours, sun_row, secondary, want


@pytest.mark.parametrize("step_key", ["20260321_0939", "20260321_1235"])
def test_polygon_occlusion_matches_fine_point_grid(step_key):
    """polygon_occlusion vs a 200x120 point-grid brute force, on the real
    shading fixture scene: agreement inside 2e-3 on every eta.

    Restricted to the mid_morning and highest_elevation steps. The third
    fixture step (lowest_elevation, 20260321_1828) was checked too during
    development: heliostat 156 there disagrees with the 200x120 reference by
    ~2.3e-3, which looks like a polygon_occlusion bug at first glance but
    is not one -- a much finer point grid (2000x1200) converges to
    0.2552354166666667, and polygon_occlusion at a matching raster resolution
    reproduces that number exactly (0.2552354166666667), while the 200x120
    grid itself sits at 0.25275, oscillating rather than monotonically
    converging as it's refined (25x15 -> 0.25333, 200x120 -> 0.25275,
    2000x1200 -> 0.25524). heliostat 156's shadow edge from its two dominant
    neighbours (107, 108) is close to axis-aligned with the sample grid at
    that sun position, which is exactly the case a point grid handles worst
    and a polygon clip is unaffected by. Per the task's own instruction not
    to force agreement with a coarse grid, that step is left out of the
    asserted comparison rather than loosening the tolerance for the other
    two steps.
    """
    field, geoms, aim_points, neighbours, sun_row, secondary, want = _load_shading_scene(step_key)

    eta_shade_25, eta_block_25, eta_sec_25 = shading_blocking(
        geoms,
        aim_points,
        sun_row["solar_az_deg"],
        sun_row["solar_el_deg"],
        neighbours,
        secondary=secondary,
    )
    eta_union_25 = occlusion_efficiency(
        geoms,
        aim_points,
        sun_row["solar_az_deg"],
        sun_row["solar_el_deg"],
        neighbours,
        secondary=secondary,
    )

    eta_shade_bf, eta_block_bf, eta_sec_bf = shading_blocking(
        geoms,
        aim_points,
        sun_row["solar_az_deg"],
        sun_row["solar_el_deg"],
        neighbours,
        nu=200,
        nv=120,
        secondary=secondary,
    )
    eta_union_bf = occlusion_efficiency(
        geoms,
        aim_points,
        sun_row["solar_az_deg"],
        sun_row["solar_el_deg"],
        neighbours,
        nu=200,
        nv=120,
        secondary=secondary,
    )

    eta_shade_p, eta_block_p, eta_sec_p, eta_union_p = polygon_occlusion(
        geoms,
        aim_points,
        sun_row["solar_az_deg"],
        sun_row["solar_el_deg"],
        neighbours,
        secondary=secondary,
        raster=(600, 360),
    )

    df = pd.DataFrame(
        {
            "id": field.ids,
            "shade_25": eta_shade_25,
            "shade_bf": eta_shade_bf,
            "shade_p": eta_shade_p,
            "block_25": eta_block_25,
            "block_bf": eta_block_bf,
            "block_p": eta_block_p,
            "sec_25": eta_sec_25,
            "sec_bf": eta_sec_bf,
            "sec_p": eta_sec_p,
            "union_25": eta_union_25,
            "union_bf": eta_union_bf,
            "union_p": eta_union_p,
        }
    ).set_index("id")
    sub = df.loc[want.index]

    for col in ("shade", "block", "sec", "union"):
        delta_bf = (sub[f"{col}_p"] - sub[f"{col}_bf"]).abs()
        delta_25 = (sub[f"{col}_p"] - sub[f"{col}_25"]).abs()
        print(
            f"\n{step_key} {col}: max|poly-200x120|={delta_bf.max():.6f} "
            f"max|poly-25x15|={delta_25.max():.6f} (polygon is the more exact of the two)"
        )
        assert delta_bf.max() < 2e-3, (
            f"{step_key} {col}: polygon vs 200x120 brute force exceeds 2e-3 "
            f"(max {delta_bf.max():.6f} at heliostat {delta_bf.idxmax()})"
        )


# ---------------------------------------------------------------------------
# (c) Convergent-blocking: central projection vs a wrongly-parallel one
# ---------------------------------------------------------------------------


def _raster_eta(quad, mirror, raster=(400, 400)):
    """eta (1 - covered fraction) for one occluder's clipped, rasterised quad."""
    if quad is None:
        return None
    clipped = _sutherland_hodgman(quad, mirror.half_width, mirror.half_height)
    if len(clipped) < 3:
        return 1.0
    n_u, n_v = raster
    su = (np.arange(n_u) + 0.5) / n_u * 2.0 - 1.0
    sv = (np.arange(n_v) + 0.5) / n_v * 2.0 - 1.0
    a, b = np.meshgrid(su * mirror.half_width, sv * mirror.half_height, indexing="ij")
    local_u, local_v = a.ravel(), b.ravel()
    covered = _points_in_polygon(local_u, local_v, clipped)
    return 1.0 - covered.mean()


def test_block_quad_uv_central_projection_beats_parallel_approximation():
    """Same scene as ``test_blocking_uses_per_point_direction_and_it_matters``:
    a mirror 80 m out aiming 60-odd metres away at a receiver near the tower,
    with a neighbour 6 m up-beam -- a case picked in that test specifically
    because the aim direction sweeps several degrees across the aperture.

    :func:`block_quad_uv`'s central projection (aim point at finite
    distance) should reproduce the per-point ``_blocked_mask`` reference
    (``converging``) closely; a parallel projection through one fixed
    direction (:func:`shadow_quad_uv` misused with ``aim - mirror.centre``,
    i.e. what ``collimated`` uses) should reproduce ``collimated`` instead
    and visibly miss the per-point reference -- the same gap
    ``test_blocking_uses_per_point_direction_and_it_matters`` already
    measures at ``> 1e-3``.
    """
    hw, hh = 2500.0, 1500.0
    aim = np.array([0.0, 0.0, 27000.0])
    g = MirrorGeometry.build(80000.0, 0.0, 4.0, 14.0, hw, hh)
    occ = MirrorGeometry.build(80000.0 - 6000.0, 0.0, 4.0, 14.0, hw, hh)

    pts = g.sample_points(401, 401)
    converging = _fraction_unoccluded(pts, aim - pts, [occ])
    collimated = _fraction_unoccluded(pts, aim - g.centre, [occ])
    assert abs(collimated - converging) > 1e-3  # the effect this case exists to show

    central_quad = block_quad_uv(occ, g, aim)
    assert central_quad is not None
    eta_central = _raster_eta(central_quad, g, raster=(600, 600))

    parallel_dir = aim - g.centre
    parallel_dir = parallel_dir / np.linalg.norm(parallel_dir)
    parallel_quad = shadow_quad_uv(occ, g, parallel_dir)
    assert parallel_quad is not None
    eta_parallel = _raster_eta(parallel_quad, g, raster=(600, 600))

    assert eta_central == pytest.approx(converging, abs=2e-3), (
        f"central-projection block_quad_uv {eta_central} vs per-point converging {converging}"
    )
    assert eta_parallel == pytest.approx(collimated, abs=2e-3), (
        f"parallel-projection (misused shadow_quad_uv) {eta_parallel} vs collimated {collimated}"
    )
    assert abs(eta_central - eta_parallel) > 1e-3, (
        "central and parallel projections should disagree meaningfully in this "
        "geometry -- if they don't, the test case no longer exercises the "
        "difference it's meant to"
    )
