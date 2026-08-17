"""Tests for arbitrarily-shaped occluders in :mod:`heliostat.geometry.shading`.

:class:`~heliostat.geometry.shading.MirrorGeometry` gained an optional
``region`` (a design's silhouette, in the mirror's own local (u, v) mm
frame) so shading and blocking test a heliostat's actual outer-perimeter
outline rather than always its bounding rectangle. Four kinds of coverage:

(a) Rect parity: unchanged, covered by the existing
    ``tests/test_shading.py``/``tests/test_polygon_shading.py`` fixture
    suites, which this change must not move -- run alongside this file, not
    duplicated in it.
(b) Flower-shades-flower and flower-blocks-rect: :func:`polygon_occlusion`'s
    exact silhouette projection agrees with a brute-force point grid
    filtered by :meth:`MirrorGeometry.contains_local`, and the flower shape
    measurably changes the answer from what two bounding rectangles would
    give -- proof the silhouette, not just the envelope, drives the result.
(c) A cone-optics integration smoke test: :func:`trace_heliostat_cone` with
    a shaped occluder produces a partially-masked (penumbra) sample set, and
    its occluded/unoccluded power ratio is in the right ballpark against
    :func:`polygon_occlusion`'s ``eta_union`` for the matching geometry.
(d) A union raster mixing a plain-rectangle occluder and a silhouette
    occluder on the same target stays a valid fraction in [0, 1].
"""

from __future__ import annotations

import numpy as np

from heliostat.geometry.design import flower
from heliostat.geometry.heliostat import heliostat_orientation
from heliostat.geometry.receiver import FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.geometry.shading import (
    MirrorGeometry,
    _blocked_mask,
    polygon_occlusion,
    sun_vector,
)
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM

KERNEL = sunshape_kernel("super_gauss")


def _flower_design():
    return flower(n_petals=5, petal_length_mm=2000.0, petal_width_mm=900.0, hub_radius_mm=300.0)


# ---------------------------------------------------------------------------
# (b) Flower-shades-flower
# ---------------------------------------------------------------------------


def test_flower_shades_flower_matches_brute_force_and_beats_bbox():
    """Two flower silhouettes, one up-sun of the other (same recipe
    ``test_shading.py``'s aligned-pair cases use: same orientation, offset
    along ``to_sun`` with height zeroed, so the pair is parallel planes and
    the offset in the mirror's own frame does not depend on separation
    distance -- 12 m up-sun here).

    :func:`polygon_occlusion`'s exact silhouette projection must agree with
    a brute-force 300x300 point grid filtered by
    :meth:`MirrorGeometry.contains_local` to within 3e-3, and the flower
    shape must move the answer measurably (> 2 percentage points) from what
    two full bounding-box rectangles at the same positions would give --
    otherwise the silhouette isn't actually doing anything.
    """
    design = _flower_design()
    sun_az, sun_el = 88.0, 9.71
    rot_az, rot_el = 2.0, 28.0
    to_sun = sun_vector(sun_az, sun_el)

    target = MirrorGeometry.from_design(0.0, 0.0, rot_az, rot_el, design)
    shift = to_sun * 12000.0
    shift[2] = 0.0
    occluder = MirrorGeometry.from_design(shift[0], shift[1], rot_az, rot_el, design)

    aim = np.array([0.0, 0.0, 27000.0])
    geoms = [target, occluder]
    aims = np.tile(aim, (2, 1))
    neighbours = [np.array([1]), np.array([])]

    eta_shade_p, _, _, _ = polygon_occlusion(
        geoms, aims, sun_az, sun_el, neighbours, raster=(400, 400)
    )

    pts = target.sample_points(300, 300)
    shaded_bf = _blocked_mask(pts, to_sun, [occluder])
    eta_shade_bf = 1.0 - shaded_bf.mean()

    assert abs(eta_shade_p[0] - eta_shade_bf) < 3e-3, (
        f"polygon eta_shade {eta_shade_p[0]:.5f} vs brute force {eta_shade_bf:.5f}"
    )
    # Sanity: this case actually shades a meaningful, partial share.
    assert 0.1 < eta_shade_p[0] < 0.95

    target_rect = MirrorGeometry.build(
        0.0, 0.0, rot_az, rot_el, target.half_width, target.half_height
    )
    occluder_rect = MirrorGeometry.build(
        shift[0], shift[1], rot_az, rot_el, target.half_width, target.half_height
    )
    eta_shade_rect, _, _, _ = polygon_occlusion(
        [target_rect, occluder_rect], aims, sun_az, sun_el, neighbours, raster=(400, 400)
    )
    assert abs(eta_shade_p[0] - eta_shade_rect[0]) > 0.02, (
        "flower silhouette should shade measurably differently from its own "
        f"bounding rectangle: flower={eta_shade_p[0]:.4f} bbox={eta_shade_rect[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# (b) Flower blocks rect
# ---------------------------------------------------------------------------


def test_flower_blocks_rect_matches_brute_force():
    """A flower-silhouette occluder sitting between a plain rectangular
    mirror and its aim point: :func:`polygon_occlusion`'s ``eta_block``
    (exact silhouette projection via :func:`block_quad_uv`) against a
    brute-force point grid, same scene
    ``test_blocking_uses_per_point_direction_and_it_matters`` uses for its
    rectangle occluder, with the occluder swapped for a flower.
    """
    design = _flower_design()
    hw, hh = 2500.0, 1500.0
    rot_az, rot_el = 4.0, 14.0
    aim = np.array([0.0, 0.0, 27000.0])

    target = MirrorGeometry.build(80000.0, 0.0, rot_az, rot_el, hw, hh)
    occluder = MirrorGeometry.from_design(80000.0 - 6000.0, 0.0, rot_az, rot_el, design)

    geoms = [target, occluder]
    aims = np.tile(aim, (2, 1))
    neighbours = [np.array([1]), np.array([])]

    _, eta_block_p, _, _ = polygon_occlusion(geoms, aims, 90.0, 45.0, neighbours, raster=(400, 400))

    pts = target.sample_points(300, 300)
    blocked_bf = _blocked_mask(pts, aim - pts, [occluder])
    eta_block_bf = 1.0 - blocked_bf.mean()

    assert abs(eta_block_p[0] - eta_block_bf) < 3e-3, (
        f"polygon eta_block {eta_block_p[0]:.5f} vs brute force {eta_block_bf:.5f}"
    )
    # Sanity: partial, not all-or-nothing.
    assert 0.05 < eta_block_p[0] < 0.99


# ---------------------------------------------------------------------------
# (c) Cone-optics integration smoke test
# ---------------------------------------------------------------------------


def test_cone_trace_with_flower_occluder_matches_polygon_eta_union():
    """A shaped (flower) occluder passed through ``trace_heliostat_cone``'s
    ``occluders`` list should produce a real penumbra (``masked > 0``) and an
    occluded/unoccluded power ratio in the same ballpark as
    :func:`polygon_occlusion`'s ``eta_union`` for the matching geometry.

    Loose (0.02 absolute) on purpose: the cone backend measures true
    penumbra transmission through a ``mask_nodes`` angular raster per
    surviving sample, while ``polygon_occlusion`` is a sharp geometric
    area fraction with no penumbra at all -- the two are different physics,
    not two implementations of the same number, so this is a ballpark
    cross-check that the shaped occluder is doing approximately the right
    thing in the ray tracer, not a precision match.
    """
    solar_az, solar_el = 100.0, 40.0
    receiver_pos = np.array([0.0, 0.0, 35335.0])
    rot_az, rot_el, *_ = heliostat_orientation(
        receiver_pos, np.array([0.0, 0.0, 0.0]), solar_az, solar_el
    )

    secondary = NoSecondary()
    receiver = FlatWindowReceiver(z_mm=35335.0, half_u_mm=2000.0, half_v_mm=2000.0, facing="down")

    target = MirrorGeometry.build(0.0, 0.0, rot_az, rot_el, MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM)
    to_sun = sun_vector(solar_az, solar_el)
    design = _flower_design()

    shift = to_sun * 6000.0
    shift[2] = 0.0
    occluder = MirrorGeometry.from_design(shift[0], shift[1], rot_az, rot_el, design)

    geoms = [target, occluder]
    aims = np.tile(receiver_pos, (2, 1))
    neighbours = [np.array([1]), np.array([])]
    _, _, _, eta_union_p = polygon_occlusion(
        geoms, aims, solar_az, solar_el, neighbours, raster=(400, 400)
    )

    cone_unocc = trace_heliostat_cone(
        0.0,
        0.0,
        rot_az,
        rot_el,
        0.0,
        0.0,
        0.0,
        solar_az,
        solar_el,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
    )
    cone_occ = trace_heliostat_cone(
        0.0,
        0.0,
        rot_az,
        rot_el,
        0.0,
        0.0,
        0.0,
        solar_az,
        solar_el,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        occluders=[occluder],
    )
    counters = cone_occ["counters"]
    assert counters["masked"] > 0, "expected at least one partially-transmitted (penumbra) sample"

    ratio_cone = cone_occ["power_w"] / cone_unocc["power_w"]
    assert abs(ratio_cone - eta_union_p[0]) < 0.02, (
        f"cone occluded/unoccluded ratio {ratio_cone:.4f} vs polygon_occlusion "
        f"eta_union {eta_union_p[0]:.4f} differ by more than 0.02"
    )


# ---------------------------------------------------------------------------
# (d) Mixed rect + silhouette occluders in one union raster
# ---------------------------------------------------------------------------


def test_union_raster_with_mixed_rect_and_silhouette_occluders_stays_in_unit_interval():
    """A target shaded/blocked by one plain-rectangle neighbour and one
    flower-silhouette neighbour at once: the mixed occluder list must not
    break the raster union, and every eta must come back a valid [0, 1]
    fraction."""
    design = _flower_design()
    hw, hh = 2500.0, 1500.0
    rot_az, rot_el = 2.0, 28.0
    sun_az, sun_el = 88.0, 9.71
    to_sun = sun_vector(sun_az, sun_el)
    aim = np.array([0.0, 0.0, 27000.0])

    target = MirrorGeometry.build(0.0, 0.0, rot_az, rot_el, hw, hh)

    shift1 = to_sun * 6000.0
    shift1[2] = 0.0
    rect_occ = MirrorGeometry.build(shift1[0], shift1[1], rot_az, rot_el, hw, hh)

    shift2 = to_sun * 4000.0
    shift2[2] = 0.0
    flower_occ = MirrorGeometry.from_design(shift2[0] + hw, shift2[1], rot_az, rot_el, design)

    geoms = [target, rect_occ, flower_occ]
    aims = np.tile(aim, (3, 1))
    neighbours = [np.array([1, 2]), np.array([]), np.array([])]

    eta_shade, eta_block, eta_secondary, eta_union = polygon_occlusion(
        geoms, aims, sun_az, sun_el, neighbours, raster=(300, 300)
    )

    for name, eta in (
        ("eta_shade", eta_shade),
        ("eta_block", eta_block),
        ("eta_secondary", eta_secondary),
        ("eta_union", eta_union),
    ):
        assert np.all(eta >= 0.0) and np.all(eta <= 1.0), f"{name} out of [0, 1]: {eta}"
    # The union can only be <= either component alone.
    assert eta_union[0] <= eta_shade[0] + 1e-12
    assert eta_union[0] <= eta_block[0] + 1e-12
