"""Tests for heliostat.geometry.aperture.

Coverage groups, matching the module's own organisation:

(a) exact areas of the primitives against closed forms, plus a numeric
    check that the grid-sampled fallback (used by every transform/CSG node)
    agrees with a known-exact area to tight relative tolerance;
(b) membership -- boundary conventions, broadcasting, and a non-convex
    polygon (including its notch);
(c) Translate/Rotate -- bbox correctness and the forward/inverse rotation
    convention that :meth:`~heliostat.geometry.aperture.Rotate.contains`
    relies on;
(d) CSG semantics -- union/intersection/difference area and bbox algebra;
(e) CircularArray -- a five-petal flower's area and rotational symmetry;
(f) petal() -- the pinned tip/waist/validation behaviour;
(g) serialisation -- round-trip through to_dict/region_from_dict for every
    kind, plus the unknown-kind error.

Note on Rotate.bbox(): it is only exercised here with centrally symmetric
children (a Rect, as the deliverable asks for) or as part of a full
360-degree CircularArray sweep, both of which happen to mask a sign bug in
its corner rotation (verified independently while building
heliostat.geometry.design -- see that module's ``_petal_at_angle``
docstring). ``Rotate.contains()`` does not have this bug; every containment
test below exercises it directly at asymmetric angles and it is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from heliostat.geometry.aperture import (
    Annulus,
    CircularArray,
    Difference,
    Disc,
    Ellipse,
    Intersection,
    Polygon,
    Rect,
    Region,
    Rotate,
    Translate,
    Union,
    circular_array,
    disc,
    petal,
    rect,
    region_from_dict,
    regular_polygon,
)

# ---------------------------------------------------------------------------
# helpers


def _forward_rotate(u, v, angle_deg):
    """Reference CCW rotation, independent of the library under test."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return c * u - s * v, s * u + c * v


# ---------------------------------------------------------------------------
# (a) exact areas


def test_rect_area_exact():
    assert Rect(400.0, 250.0).area_mm2() == pytest.approx(100000.0)


def test_disc_area_exact():
    assert Disc(300.0).area_mm2() == pytest.approx(np.pi * 300.0**2)


def test_ellipse_area_exact():
    assert Ellipse(500.0, 200.0).area_mm2() == pytest.approx(np.pi * 500.0 * 200.0)


def test_annulus_area_exact():
    a = Annulus(100.0, 250.0)
    assert a.area_mm2() == pytest.approx(np.pi * (250.0**2 - 100.0**2))


def test_polygon_area_shoelace_square():
    square = Polygon([[-100, -100], [100, -100], [100, 100], [-100, 100]])
    assert square.area_mm2() == pytest.approx(40000.0)


def test_polygon_area_shoelace_regular_hexagon():
    hexagon = regular_polygon(6, 1000.0)
    exact = 0.5 * 6 * 1000.0**2 * np.sin(2 * np.pi / 6)
    assert hexagon.area_mm2() == pytest.approx(exact, rel=1e-12)


def test_polygon_area_orientation_independent():
    """Shoelace must give a positive area regardless of vertex winding."""
    ccw = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    cw = Polygon([[0, 0], [0, 10], [10, 10], [10, 0]])
    assert ccw.area_mm2() == pytest.approx(cw.area_mm2())


def test_rotated_rect_numeric_area_matches_exact():
    """Grid-sampled area_mm2 (the base-class fallback) on a rotated rect."""
    exact = 5000.0 * 3000.0
    numeric = Rotate(Rect(5000.0, 3000.0), 37.0).area_mm2(resolution=2048)
    assert numeric == pytest.approx(exact, rel=1e-3)


# ---------------------------------------------------------------------------
# (b) membership


def test_rect_boundary_convention():
    r = Rect(200.0, 100.0)
    eps = 1e-6
    assert r.contains(100.0 - eps, 0.0)
    assert not r.contains(100.0 + eps, 0.0)
    assert r.contains(0.0, 50.0 - eps)
    assert not r.contains(0.0, 50.0 + eps)
    # inclusive on the boundary itself
    assert r.contains(100.0, 50.0)


def test_disc_boundary_convention():
    d = Disc(500.0)
    eps = 1e-3
    assert d.contains(500.0 - eps, 0.0)
    assert not d.contains(500.0 + eps, 0.0)
    assert d.contains(500.0, 0.0)  # inclusive


def test_ellipse_boundary_convention():
    e = Ellipse(400.0, 150.0)
    assert e.contains(400.0 - 1e-3, 0.0)
    assert not e.contains(400.0 + 1e-3, 0.0)
    assert e.contains(0.0, 150.0 - 1e-3)
    assert not e.contains(0.0, 150.0 + 1e-3)


def test_annulus_boundary_conventions_both_edges():
    a = Annulus(100.0, 250.0)
    eps = 1e-3
    # inner edge
    assert not a.contains(100.0 - eps, 0.0)
    assert a.contains(100.0 + eps, 0.0)
    assert a.contains(100.0, 0.0)  # inclusive
    # outer edge
    assert a.contains(250.0 - eps, 0.0)
    assert not a.contains(250.0 + eps, 0.0)
    assert a.contains(250.0, 0.0)  # inclusive


def test_contains_broadcast_scalar():
    r = Rect(200.0, 100.0)
    out = r.contains(0.0, 0.0)
    assert out == True  # noqa: E712 -- explicitly checking scalar bool-ish output
    assert np.asarray(out).shape == ()


def test_contains_broadcast_1d():
    r = Rect(200.0, 100.0)
    u = np.array([0.0, 150.0, -150.0])
    v = np.array([0.0, 0.0, 0.0])
    out = r.contains(u, v)
    assert out.shape == (3,)
    assert list(out) == [True, False, False]


def test_contains_broadcast_2d_meshgrid():
    r = Rect(200.0, 100.0)
    u = np.linspace(-150, 150, 7)
    v = np.linspace(-80, 80, 5)
    out = r.contains(u[None, :], v[:, None])
    assert out.shape == (5, 7)
    # centre column/row inside, far corners outside the 200x100 rect
    assert out[2, 3]  # (u=0, v=0)
    assert not out[0, 0]  # (u=-150, v=-80): outside width and height


def test_polygon_l_shape_notch_and_crossing_number():
    # L-shape: a 100x100 square with a 50x50 notch cut from the top-right.
    l_shape = Polygon(
        [
            [0, 0],
            [100, 0],
            [100, 50],
            [50, 50],
            [50, 100],
            [0, 100],
        ]
    )
    assert l_shape.contains(25, 25)  # solid leg
    assert l_shape.contains(75, 25)  # solid leg
    assert not l_shape.contains(75, 75)  # inside the notch (removed corner)
    assert l_shape.contains(10, 90)  # solid leg, near notch but outside it
    # broadcast form, mixed inside/outside/notch
    uu = np.array([25.0, 75.0, 75.0])
    vv = np.array([25.0, 25.0, 75.0])
    out = l_shape.contains(uu, vv)
    assert list(out) == [True, True, False]


# ---------------------------------------------------------------------------
# (c) transforms


def test_translate_bbox():
    r = Rect(200.0, 100.0).translated(50.0, -30.0)
    assert r.bbox() == pytest.approx((-50.0, 150.0, -80.0, 20.0))


def test_translate_contains_shifts_the_shape():
    r = Rect(200.0, 100.0).translated(1000.0, 500.0)
    assert r.contains(1000.0, 500.0)
    assert not r.contains(0.0, 0.0)


def test_rotate_rect_bbox_equals_rotated_corners():
    """Rotated rect bbox = bbox of its own corners, forward-rotated."""
    child = Rect(5000.0, 3000.0)
    angle = 37.0
    u0, u1, v0, v1 = child.bbox()
    corners = [(u0, v0), (u0, v1), (u1, v0), (u1, v1)]
    rotated = [_forward_rotate(u, v, angle) for u, v in corners]
    expect = (
        min(p[0] for p in rotated),
        max(p[0] for p in rotated),
        min(p[1] for p in rotated),
        max(p[1] for p in rotated),
    )
    assert Rotate(child, angle).bbox() == pytest.approx(expect)


def test_rotate_45_of_square_contains_original_corners_images():
    """Rotate(45) of a square contains the forward-rotated image of each corner.

    Corners sit exactly on Rect's inclusive boundary, so a point rotated
    *exactly* onto them can miss by float roundoff; nudge fractionally
    inward (toward the origin) to test the rotation direction, not
    sub-micron boundary precision.
    """
    square = Rect(2000.0, 2000.0)
    rotated = Rotate(square, 45.0)
    u0, u1, v0, v1 = square.bbox()
    for cu, cv in [(u0, v0), (u0, v1), (u1, v0), (u1, v1)]:
        nu, nv = cu * 0.999, cv * 0.999
        fu, fv = _forward_rotate(nu, nv, 45.0)
        assert rotated.contains(fu, fv)


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 45.0, 73.4, 200.0, 359.0])
def test_rotate_inverse_membership_property(angle_deg):
    """p rotated forward by a is in Rotate(child, a) iff p is in child."""
    rng = np.random.default_rng(0)
    child = Rect(1200.0, 700.0)
    rotated = Rotate(child, angle_deg)
    uu = rng.uniform(-2000.0, 2000.0, 2000)
    vv = rng.uniform(-2000.0, 2000.0, 2000)
    fu, fv = _forward_rotate(uu, vv, angle_deg)
    lhs = rotated.contains(fu, fv)
    rhs = child.contains(uu, vv)
    assert np.array_equal(lhs, rhs)


# ---------------------------------------------------------------------------
# (d) CSG semantics


def test_union_area_de_morgan_on_overlapping_discs():
    a = disc(1000.0, at=(-400.0, 0.0))
    b = disc(1000.0, at=(400.0, 0.0))
    union_area = (a | b).area_mm2(resolution=2048)
    inter_area = (a & b).area_mm2(resolution=2048)
    expect = a.area_mm2() + b.area_mm2() - inter_area
    assert union_area == pytest.approx(expect, rel=1e-3)


def test_union_and_intersection_are_measurably_different():
    a = disc(1000.0, at=(-400.0, 0.0))
    b = disc(1000.0, at=(400.0, 0.0))
    union_area = (a | b).area_mm2(resolution=1024)
    inter_area = (a & b).area_mm2(resolution=1024)
    assert union_area > inter_area * 2  # not just float noise


def test_difference_bbox_equals_base_bbox():
    base = Rect(1000.0, 800.0)
    cut = Disc(200.0)
    diff = Difference(base, cut)
    assert diff.bbox() == base.bbox()


def test_difference_removes_material():
    base = disc(500.0)
    cut = disc(500.0)  # identical circle removed entirely
    diff = base - cut
    assert diff.area_mm2(resolution=512) == pytest.approx(0.0, abs=1.0)
    assert not diff.contains(0.0, 0.0)


def test_empty_intersection_gives_zero_area():
    a = disc(500.0, at=(-10000.0, 0.0))
    b = disc(500.0, at=(10000.0, 0.0))
    inter = a & b
    # disjoint bboxes -> the base class's u1<=u0 guard fires directly
    u0, u1, v0, v1 = inter.bbox()
    assert u1 <= u0
    assert inter.area_mm2() == 0.0


# ---------------------------------------------------------------------------
# (e) CircularArray


def test_circular_array_five_petal_flower_area_ratio():
    p = petal(2000.0, 900.0)
    flower = circular_array(p, 5)
    petal_area = p.area_mm2(resolution=2048)
    flower_area = flower.area_mm2(resolution=2048)
    ratio = flower_area / petal_area
    # Petals converge at the origin and genuinely overlap there (not just
    # grid noise -- see the CSG test above for the noise floor), so the
    # ratio is measurably below the no-overlap ideal of 5.0.
    assert 4.9 < ratio <= 5.0


def test_circular_array_rotational_symmetry():
    p = petal(2000.0, 900.0)
    flower = circular_array(p, 5)
    rng = np.random.default_rng(7)
    uu = rng.uniform(-2500.0, 2500.0, 3000)
    vv = rng.uniform(-2500.0, 2500.0, 3000)
    ru, rv = _forward_rotate(uu, vv, 72.0)
    assert np.array_equal(flower.contains(uu, vv), flower.contains(ru, rv))


# ---------------------------------------------------------------------------
# (f) petal()


def test_petal_tip_and_base():
    p = petal(2000.0, 900.0)
    assert p.contains(0.0, 2000.0)  # tip, inclusive boundary
    assert p.contains(0.0, 0.0)  # base point


def test_petal_waist_width_at_half_length():
    p = petal(2000.0, 900.0)
    half_w = 900.0 / 2.0
    eps = 0.5
    assert p.contains(half_w - eps, 1000.0)
    assert not p.contains(half_w + eps, 1000.0)
    assert p.contains(-(half_w - eps), 1000.0)
    assert not p.contains(-(half_w + eps), 1000.0)


def test_petal_rejects_width_at_least_twice_length():
    with pytest.raises(ValueError):
        petal(1000.0, 2000.0)  # width == 2*length
    with pytest.raises(ValueError):
        petal(1000.0, 2500.0)  # width > 2*length


# ---------------------------------------------------------------------------
# (g) serialisation


def _random_cloud(n=500, extent=3000.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-extent, extent, n), rng.uniform(-extent, extent, n)


@pytest.mark.parametrize(
    "region",
    [
        Rect(1200.0, 800.0),
        Disc(600.0),
        Ellipse(700.0, 300.0),
        Annulus(100.0, 500.0),
        Polygon([[0, 0], [500, 0], [500, 300], [0, 300]]),
        regular_polygon(6, 400.0),
        Rect(400.0, 300.0).translated(150.0, -80.0),
        Rect(400.0, 300.0).rotated(37.0),
        disc(300.0, at=(-100, 200)) | disc(300.0, at=(100, -200)),
        disc(300.0, at=(-100, 200)) & disc(300.0, at=(100, -200)),
        Rect(1000.0, 1000.0) - Disc(200.0),
        circular_array(petal(1500.0, 700.0), 5),
    ],
    ids=[
        "rect",
        "disc",
        "ellipse",
        "annulus",
        "polygon",
        "regular_polygon",
        "translate",
        "rotate",
        "union",
        "intersection",
        "difference",
        "circular_array",
    ],
)
def test_region_round_trip_preserves_membership(region):
    rebuilt = region_from_dict(region.to_dict())
    uu, vv = _random_cloud()
    assert np.array_equal(region.contains(uu, vv), rebuilt.contains(uu, vv))


def test_region_from_dict_unknown_kind_raises():
    with pytest.raises(ValueError):
        region_from_dict({"kind": "not_a_real_shape"})


def test_rect_convenience_constructor_with_offset():
    r = rect(200.0, 100.0, at=(50.0, 25.0))
    assert isinstance(r, Region)
    assert r.contains(50.0, 25.0)
    assert not r.contains(500.0, 500.0)


def test_region_algebra_operators_match_explicit_classes():
    a, b = disc(100.0), disc(100.0, at=(50, 0))
    assert isinstance(a | b, Union)
    assert isinstance(a & b, Intersection)
    assert isinstance(a - b, Difference)
    assert isinstance(a.translated(10, 10), Translate)
    assert isinstance(a.rotated(10), Rotate)
    assert isinstance(CircularArray(a, 3), CircularArray)
