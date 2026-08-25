"""Tests for the corridor pruning in :mod:`heliostat.geometry.shading`
(:func:`_shading_candidates`, :func:`_blocking_candidates`, and their use in
:func:`polygon_occlusion`/:func:`occlusion_efficiency`).

At low sun the neighbour list :func:`~heliostat.field.neighbour_pairs` builds
is a disc in the ground plane, sized to cover the longest possible shadow --
but a shadow only falls along the sun direction, so most of that disc is
occluders that sit well to the side and can never actually reach the target.
The prune drops those before the expensive quad-projection/clip step, using
a bound cheap enough to vectorise: no exact ray from the sun (parallel) or
from the aim point (a finite-distance point source) can pass through both
mirrors' bounding circles unless those circles overlap as seen along that
ray's own direction.

Two kinds of coverage:

(a) Hand-checkable cases for the two pruning primitives themselves.
(b) The headline proof: :func:`polygon_occlusion` with pruning enabled
    (today's code) against a locally-reimplemented, deliberately unpruned
    version of the same computation, on a dense field, at several sun
    elevations -- exact array equality, not a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.field import HeliostatField, neighbour_pairs
from heliostat.geometry.aiming import aim_points_mm, solve_prime_focus
from heliostat.geometry.shading import (
    _blocked_mask,
    _blocking_candidates,
    _points_in_polygon,
    _project_onto_plane,
    _shading_candidates,
    _sutherland_hodgman,
    block_quad_uv,
    build_geometries,
    min_beam_elevation_deg,
    occlusion_efficiency,
    polygon_occlusion,
    search_radius_for,
    shadow_quad_uv,
    sun_vector,
)

SOLAR_AZ_DEG = 165.2
TOWER_Z_MM = 27000.0


# ---------------------------------------------------------------------------
# (a) The two pruning primitives, by hand
# ---------------------------------------------------------------------------


def test_shading_candidates_keeps_overlap_and_drops_clear_separation():
    # Sum of bounding radii is 3.0 + 2.9 = 5.9: right at that distance the
    # bounding circles still touch (kept, "<="); a hair beyond, they cannot.
    target_proj = np.array([0.0, 0.0])
    cand_proj = np.array([[5.9, 0.0], [6.1, 0.0]])
    keep = _shading_candidates(target_proj, 3.0, cand_proj, np.array([2.9, 2.9]))
    assert list(keep) == [True, False]


def test_shading_candidates_is_symmetric_in_the_projected_plane():
    """Only the in-plane distance matters, not which axis it's along --
    the projection has already collapsed the sun direction away."""
    target_proj = np.array([0.0, 0.0])
    cand_proj = np.array([[4.0, 3.0], [3.0, 4.0]])  # both distance 5 away
    keep = _shading_candidates(target_proj, 3.0, cand_proj, np.array([2.9, 2.9]))
    assert list(keep) == [True, True]


def test_blocking_candidates_keeps_aligned_and_drops_off_axis():
    """Apex at the origin, target dead ahead at 100 m: a candidate near the
    same line of sight is kept, one far enough off it is dropped.

    Half-angles: asin(5/100) for the target and asin(5/~100) for either
    candidate, summing to about 5.73 deg (tan ~ 0.1003), so a lateral offset
    of 8 m at ~100 m range (about 4.6 deg) stays inside the cone and one of
    14 m (about 8 deg) does not.
    """
    apex = np.array([0.0, 0.0, 0.0])
    target_centre = np.array([0.0, 0.0, 100000.0])
    cand_centres = np.array(
        [
            [0.0, 8000.0, 100000.0],
            [0.0, 14000.0, 100000.0],
        ]
    )
    cand_radii = np.array([5000.0, 5000.0])
    keep = _blocking_candidates(apex, target_centre, 5000.0, cand_centres, cand_radii)
    assert list(keep) == [True, False]


def test_blocking_candidates_same_line_always_kept_whatever_the_distance():
    """A candidate directly behind the target on the same ray from the apex
    has zero angular separation, so it is kept regardless of how far along
    that ray it sits -- unlike the flat shading bound, distance along the
    beam itself never prunes a blocking candidate."""
    apex = np.array([0.0, 0.0, 0.0])
    target_centre = np.array([0.0, 0.0, 50000.0])
    cand_centres = np.array([[0.0, 0.0, 20000.0], [0.0, 0.0, 200000.0]])
    cand_radii = np.array([3000.0, 3000.0])
    keep = _blocking_candidates(apex, target_centre, 3000.0, cand_centres, cand_radii)
    assert list(keep) == [True, True]


# ---------------------------------------------------------------------------
# (b) The headline proof: pruned polygon_occlusion vs an unpruned reference
# ---------------------------------------------------------------------------


def _dense_field(radius_m: float = 94.0, pitch_m: float = 6.0, n_side: int = 9) -> HeliostatField:
    """A patch of heliostats dense enough to shade each other at low sun and
    far enough from the tower for their beams to be flat enough to block
    each other too (same tuning :func:`heliostat.geometry.shading` tests
    elsewhere use, just a larger patch so pruning has real work to do)."""
    offsets = (np.arange(n_side) - (n_side - 1) / 2) * pitch_m * 1000.0
    xs, ys = np.meshgrid(offsets, offsets + radius_m * 1000.0)
    return HeliostatField(
        x_mm=xs.ravel(),
        y_mm=ys.ravel(),
        ids=np.arange(xs.size),
        mirror_width_mm=5000.0,
        mirror_height_mm=3000.0,
    )


def _build_scene(field: HeliostatField, solar_el_deg: float):
    solutions = [
        solve_prime_focus(float(x), float(y), SOLAR_AZ_DEG, solar_el_deg, TOWER_Z_MM)
        for x, y in field.xy_mm
    ]
    geometries, aims = build_geometries(
        field,
        np.array([s.rot_az_deg for s in solutions]),
        np.array([s.rot_el_deg for s in solutions]),
        aim_points_mm(solutions),
        mirror_width_mm=field.mirror_width_mm,
        mirror_height_mm=field.mirror_height_mm,
    )
    return geometries, aims


def _neighbours_for(field: HeliostatField, geometries, aims, solar_el_deg: float):
    centres = np.array([g.centre for g in geometries])
    beam_el = min_beam_elevation_deg(centres, aims)
    radius = search_radius_for(
        solar_el_deg, field.mirror_height_mm, field.mirror_width_mm, beam_elevation_deg=beam_el
    )
    return neighbour_pairs(field, radius)


def _reference_polygon_occlusion(
    geometries, aim_points_mm_, solar_az_deg, solar_el_deg, neighbours, raster=(100, 60)
):
    """:func:`polygon_occlusion`'s exact computation with the corridor prune
    left out -- every candidate in ``neighbours[i]`` is quad-projected and
    clipped, full stop. Kept here (not in the shading module) purely as an
    independent ground truth for the exactness test below: it is built from
    the same public/private primitives (:func:`shadow_quad_uv`,
    :func:`block_quad_uv`, :func:`_sutherland_hodgman`,
    :func:`_points_in_polygon`, :func:`_blocked_mask`) polygon_occlusion
    itself still uses for the pairs it does not prune.
    """
    n = len(geometries)
    if solar_el_deg <= 0.0:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    to_sun = sun_vector(solar_az_deg, solar_el_deg)
    aim_points_mm_ = np.asarray(aim_points_mm_, dtype=float)
    n_u, n_v = raster
    su = (np.arange(n_u) + 0.5) / n_u * 2.0 - 1.0
    sv = (np.arange(n_v) + 0.5) / n_v * 2.0 - 1.0

    eta_shade = np.ones(n)
    eta_block = np.ones(n)
    eta_union = np.ones(n)

    for i, mirror in enumerate(geometries):
        nbrs = [geometries[j] for j in neighbours[i]]

        a, b = np.meshgrid(su * mirror.half_width, sv * mirror.half_height, indexing="ij")
        local_u, local_v = a.ravel(), b.ravel()
        world_pts = mirror.centre + local_u[:, None] * mirror.u + local_v[:, None] * mirror.v

        shaded = np.zeros(local_u.size, dtype=bool)
        blocked = np.zeros(local_u.size, dtype=bool)

        for occ in nbrs:
            quad = shadow_quad_uv(occ, mirror, to_sun)
            if quad is None:
                shaded |= _blocked_mask(world_pts, to_sun, [occ])
            else:
                clipped = _sutherland_hodgman(quad, mirror.half_width, mirror.half_height)
                if len(clipped) >= 3:
                    shaded |= _points_in_polygon(local_u, local_v, clipped)

            bquad = block_quad_uv(occ, mirror, aim_points_mm_[i])
            if bquad is None:
                blocked |= _blocked_mask(world_pts, aim_points_mm_[i] - world_pts, [occ])
            else:
                clipped_b = _sutherland_hodgman(bquad, mirror.half_width, mirror.half_height)
                if len(clipped_b) >= 3:
                    blocked |= _points_in_polygon(local_u, local_v, clipped_b)

        eta_shade[i] = 1.0 - shaded.mean()
        eta_block[i] = 1.0 - blocked.mean() if nbrs else 1.0
        eta_union[i] = 1.0 - (shaded | blocked).mean()

    return eta_shade, eta_block, eta_union


@pytest.mark.parametrize("solar_el_deg", [5.0, 10.0, 20.0, 30.0, 60.0])
def test_corridor_prune_is_bit_identical_to_unpruned(solar_el_deg):
    """The whole point of the prune: it may only skip pairs that provably
    cannot interact, so it must never move a single eta value."""
    field = _dense_field()
    geometries, aims = _build_scene(field, solar_el_deg)
    neighbours = _neighbours_for(field, geometries, aims, solar_el_deg)

    total_candidates = sum(len(idx) for idx in neighbours)
    assert total_candidates > 0, "test field has no neighbour candidates at this elevation"

    ref_shade, ref_block, ref_union = _reference_polygon_occlusion(
        geometries, aims, SOLAR_AZ_DEG, solar_el_deg, neighbours
    )
    got_shade, got_block, _got_sec, got_union = polygon_occlusion(
        geometries, aims, SOLAR_AZ_DEG, solar_el_deg, neighbours
    )

    assert np.array_equal(ref_shade, got_shade), "eta_shade changed under pruning"
    assert np.array_equal(ref_block, got_block), "eta_block changed under pruning"
    assert np.array_equal(ref_union, got_union), "eta_union changed under pruning"
    # Guard against a vacuous pass: the field must actually occlude here.
    assert np.any(got_union < 1.0), "test field has no real occlusion at this elevation"


@pytest.mark.parametrize("solar_el_deg", [5.0, 20.0, 60.0])
def test_occlusion_efficiency_prune_is_bit_identical_to_unpruned(solar_el_deg):
    """:func:`occlusion_efficiency` shares the same neighbour-list path and
    prunes the same way; check it against its own (pre-pruning) form:
    unfiltered neighbour lists straight into ``_blocked_mask``."""
    field = _dense_field(n_side=6)
    geometries, aims = _build_scene(field, solar_el_deg)
    neighbours = _neighbours_for(field, geometries, aims, solar_el_deg)
    to_sun = sun_vector(SOLAR_AZ_DEG, solar_el_deg)

    reference = np.ones(len(geometries))
    for i, geom in enumerate(geometries):
        nbrs = [geometries[j] for j in neighbours[i]]
        pts = geom.sample_points(25, 15)
        lost = _blocked_mask(pts, to_sun, nbrs) | _blocked_mask(pts, aims[i] - pts, nbrs)
        reference[i] = float(1.0 - lost.mean())

    got = occlusion_efficiency(
        geometries, aims, SOLAR_AZ_DEG, solar_el_deg, neighbours, nu=25, nv=15
    )

    assert np.array_equal(reference, got)
    assert np.any(got < 1.0), "test field has no real occlusion at this elevation"


def test_corridor_prune_actually_reduces_low_sun_candidates():
    """Proves the mechanism, not just its safety: at low sun the disc-shaped
    neighbour list is mostly occluders off to the side of the sun corridor,
    and the shading prune should visibly thin it out."""
    solar_el_deg = 5.0
    field = _dense_field()
    geometries, aims = _build_scene(field, solar_el_deg)
    neighbours = _neighbours_for(field, geometries, aims, solar_el_deg)

    all_centres = np.array([g.centre for g in geometries])
    all_radii = np.array([np.hypot(g.half_width, g.half_height) for g in geometries])
    to_sun = sun_vector(SOLAR_AZ_DEG, solar_el_deg)
    sun_proj = _project_onto_plane(to_sun, all_centres)

    pre_total = 0
    kept_total = 0
    for i in range(len(geometries)):
        idx = np.asarray(neighbours[i], dtype=int)
        pre_total += idx.size
        if idx.size:
            keep = _shading_candidates(sun_proj[i], all_radii[i], sun_proj[idx], all_radii[idx])
            kept_total += int(keep.sum())

    assert pre_total > 0
    assert kept_total < pre_total, "pruning did not reduce the shading candidate count at all"
