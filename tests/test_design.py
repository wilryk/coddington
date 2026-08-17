"""Tests for heliostat.geometry.design.

Coverage:

(a) the cant law -- numeric reflection check for several offsets/focals;
(b) Surface implementations -- Spherical's numeric gradient, ZernikeAstig's
    exact delegation to heliostat.trace.mc._zernike_sag_and_slopes, Flat;
(c) rect_heliostat -- the parity anchor's exact geometry;
(d) grid_facets -- area, offsets, cant direction;
(e) flower -- facet count, area agreement between the two construction
    modes, round-trip serialisation;
(f) HeliostatDesign.silhouette -- rect/flower/gridded-with-gap cases;
(g) HeliostatDesign.preview -- headless smoke tests, matplotlib Agg backend.

Note on flower(petals_as_facets=True) and hub_radius_mm: with the default
hub_radius_mm=0, adjacent petals genuinely overlap near the hub (verified in
tests/test_aperture.py's CircularArray ratio test), so the *sum* of facet
areas legitimately exceeds the *union* sketch area by that overlap -- not a
numerical artefact. The area-agreement test below uses a hub radius large
enough to separate the petals so the two constructions describe the same
footprint and can be compared to tight tolerance.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from heliostat.geometry import aperture
from heliostat.geometry.design import (
    Facet,
    Flat,
    HeliostatDesign,
    Spherical,
    ZernikeAstig,
    cant_on_axis,
    flower,
    grid_facets,
    rect_heliostat,
    surface_from_dict,
)
from heliostat.trace.mc import _zernike_sag_and_slopes

# ---------------------------------------------------------------------------
# (a) cant law


@pytest.mark.parametrize(
    "offset,focal_mm",
    [
        ((0.0, 0.0), 100000.0),
        ((1000.0, 500.0), 100000.0),
        ((-2000.0, 1200.0), 50000.0),
        ((2500.0, 1500.0), 100000.0),
        ((-800.0, -1400.0), 75000.0),
    ],
)
def test_cant_on_axis_reflects_centre_to_focal_point(offset, focal_mm):
    facet = Facet(region=aperture.Rect(100.0, 100.0), surface=Flat(), offset_mm=offset)
    (canted,) = cant_on_axis([facet], focal_mm)
    n = canted.cant_normal
    assert n is not None
    assert np.linalg.norm(n) == pytest.approx(1.0)

    d_in = np.array([0.0, 0.0, -1.0])
    d_out = d_in - 2.0 * np.dot(d_in, n) * n

    p0 = np.array([offset[0], offset[1], 0.0])
    t = (focal_mm - p0[2]) / d_out[2]
    hit = p0 + t * d_out
    assert np.hypot(hit[0], hit[1]) < 1.0  # < 1 mm off-axis at the target plane
    assert hit[2] == pytest.approx(focal_mm)


def test_cant_on_axis_trivial_at_centre_facet():
    """A facet with no offset needs no tilt at all."""
    facet = Facet(region=aperture.Rect(100.0, 100.0), surface=Flat(), offset_mm=(0.0, 0.0))
    (canted,) = cant_on_axis([facet], 100000.0)
    assert canted.cant_normal == pytest.approx([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# (b) surfaces


def test_flat_surface_is_zero_everywhere():
    lu = np.array([0.0, 100.0, -250.0])
    lv = np.array([0.0, -80.0, 300.0])
    sag, dsdu, dsdv = Flat().sag_and_slopes(lu, lv)
    assert np.allclose(sag, 0.0)
    assert np.allclose(dsdu, 0.0)
    assert np.allclose(dsdv, 0.0)
    assert sag.shape == lu.shape


def test_spherical_sag_zero_at_origin():
    sag, _, _ = Spherical(100000.0).sag_and_slopes(0.0, 0.0)
    assert float(sag) == pytest.approx(0.0)


def test_spherical_slopes_match_numerical_gradient():
    surf = Spherical(85000.0)
    rng = np.random.default_rng(3)
    lu = rng.uniform(-2500.0, 2500.0, 200)
    lv = rng.uniform(-1500.0, 1500.0, 200)
    _, dsdu, dsdv = surf.sag_and_slopes(lu, lv)

    h = 1e-3
    sag_u1, _, _ = surf.sag_and_slopes(lu + h, lv)
    sag_u0, _, _ = surf.sag_and_slopes(lu - h, lv)
    num_dsdu = (sag_u1 - sag_u0) / (2 * h)
    sag_v1, _, _ = surf.sag_and_slopes(lu, lv + h)
    sag_v0, _, _ = surf.sag_and_slopes(lu, lv - h)
    num_dsdv = (sag_v1 - sag_v0) / (2 * h)

    assert np.allclose(dsdu, num_dsdu, rtol=1e-9, atol=1e-12)
    assert np.allclose(dsdv, num_dsdv, rtol=1e-9, atol=1e-12)


def test_spherical_unresolved_slant_raises():
    with pytest.raises(ValueError):
        Spherical("slant").sag_and_slopes(0.0, 0.0)


def test_zernike_astig_delegates_exactly_to_mc():
    rng = np.random.default_rng(5)
    lu = rng.uniform(-2500.0, 2500.0, 300)
    lv = rng.uniform(-1500.0, 1500.0, 300)
    c3, c4, c5 = 0.012, -0.034, 0.007

    got = ZernikeAstig(c3, c4, c5).sag_and_slopes(lu, lv)
    want = _zernike_sag_and_slopes(lu, lv, c3, c4, c5)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_surface_round_trip_and_unknown_kind():
    for surf in (Flat(), Spherical(90000.0), ZernikeAstig(0.01, -0.02, 0.03)):
        rebuilt = surface_from_dict(surf.to_dict())
        got = surf.sag_and_slopes(np.array([100.0, -50.0]), np.array([30.0, 200.0]))
        want = rebuilt.sag_and_slopes(np.array([100.0, -50.0]), np.array([30.0, 200.0]))
        assert all(np.allclose(g, w) for g, w in zip(got, want))

    with pytest.raises(ValueError):
        surface_from_dict({"kind": "not_a_surface"})


# ---------------------------------------------------------------------------
# (c) rect_heliostat


def test_rect_heliostat_is_the_parity_anchor():
    design = rect_heliostat()
    assert design.bbox == pytest.approx((-2500.0, 2500.0, -1500.0, 1500.0))
    assert design.area_mm2 == pytest.approx(15e6)
    assert design.half_diagonal_mm == pytest.approx(np.hypot(2500.0, 1500.0))
    assert len(design.facets) == 1
    facet = design.facets[0]
    assert facet.offset_mm == (0.0, 0.0)
    assert facet.cant_normal is None
    assert isinstance(facet.surface, ZernikeAstig)
    assert (facet.surface.c3, facet.surface.c4, facet.surface.c5) == (0.0, 0.0, 0.0)


def test_rect_heliostat_custom_size_and_surface():
    design = rect_heliostat(width_mm=4000.0, height_mm=2000.0, surface=Flat())
    assert design.bbox == pytest.approx((-2000.0, 2000.0, -1000.0, 1000.0))
    assert design.area_mm2 == pytest.approx(8e6)
    assert isinstance(design.facets[0].surface, Flat)


# ---------------------------------------------------------------------------
# (d) grid_facets


def test_grid_facets_area_is_sum_of_facet_areas():
    design = grid_facets(2, 2, 1000.0, 800.0, gap_mm=50.0)
    assert design.area_mm2 == pytest.approx(4 * 1000.0 * 800.0)
    assert len(design.facets) == 4


def test_grid_facets_offsets_correct():
    design = grid_facets(2, 2, 1000.0, 800.0, gap_mm=50.0)
    offsets = sorted(f.offset_mm for f in design.facets)
    half_u = (1000.0 + 50.0) / 2.0  # facet width + gap, halved: the 2x2 pitch
    half_v = (800.0 + 50.0) / 2.0  # facet height + gap, halved
    expect = sorted([(-half_u, -half_v), (-half_u, half_v), (half_u, -half_v), (half_u, half_v)])
    for got, want in zip(offsets, expect):
        assert got == pytest.approx(want)


def test_grid_facets_canted_normals_tilt_toward_axis():
    design = grid_facets(2, 2, 1000.0, 800.0, gap_mm=50.0, cant_focal_mm=80000.0)
    for f in design.facets:
        ou, ov = f.offset_mm
        cu, cv, _ = f.cant_normal
        if ou != 0.0:
            assert np.sign(cu) == -np.sign(ou)
        if ov != 0.0:
            assert np.sign(cv) == -np.sign(ov)


def test_grid_facets_no_cant_by_default():
    design = grid_facets(2, 2, 1000.0, 800.0)
    assert all(f.cant_normal is None for f in design.facets)


def test_grid_facets_slant_focal_resolves_per_facet():
    design = grid_facets(
        2, 2, 1000.0, 800.0, gap_mm=50.0, surface=Spherical("slant"), cant_focal_mm=80000.0
    )
    for f in design.facets:
        ou, ov = f.offset_mm
        assert isinstance(f.surface, Spherical)
        assert not isinstance(f.surface.focal_mm, str)
        expect = np.hypot(np.hypot(ou, ov), 80000.0)
        assert f.surface.focal_mm == pytest.approx(expect)


# ---------------------------------------------------------------------------
# (e) flower


def test_flower_petals_as_facets_true_has_n_facets():
    design = flower(n_petals=5, petals_as_facets=True)
    assert len(design.facets) == 5


def test_flower_facet_area_matches_single_sketch_area():
    # hub_radius_mm large enough that petals no longer overlap near the hub.
    hub = 200.0
    facets_design = flower(petals_as_facets=True, hub_radius_mm=hub)
    sketch_design = flower(petals_as_facets=False, hub_radius_mm=hub)

    area_from_facets = sum(f.region.area_mm2(resolution=2048) for f in facets_design.facets)
    area_from_sketch = sketch_design.facets[0].region.area_mm2(resolution=2048)
    assert area_from_facets == pytest.approx(area_from_sketch, rel=1e-3)


def test_flower_petals_as_facets_round_trip_preserves_membership_and_cant():
    design = flower(petals_as_facets=True, cant_focal_mm=60000.0)
    rebuilt = HeliostatDesign.from_dict(design.to_dict())

    rng = np.random.default_rng(11)
    for f1, f2 in zip(design.facets, rebuilt.facets):
        uu = rng.uniform(-3000.0, 3000.0, 300)
        vv = rng.uniform(-3000.0, 3000.0, 300)
        assert np.array_equal(f1.region.contains(uu, vv), f2.region.contains(uu, vv))
        assert f1.cant_normal is not None
        assert np.allclose(f1.cant_normal, f2.cant_normal)
        assert f1.offset_mm == pytest.approx(f2.offset_mm)


def test_flower_single_sketch_round_trip():
    design = flower(petals_as_facets=False)
    rebuilt = HeliostatDesign.from_dict(design.to_dict())
    rng = np.random.default_rng(12)
    uu = rng.uniform(-3000.0, 3000.0, 500)
    vv = rng.uniform(-3000.0, 3000.0, 500)
    assert np.array_equal(
        design.facets[0].region.contains(uu, vv), rebuilt.facets[0].region.contains(uu, vv)
    )


def test_heliostat_design_from_dict_unknown_facet_surface_kind_raises():
    design = rect_heliostat()
    data = design.to_dict()
    data["facets"][0]["surface"]["kind"] = "not_a_surface"
    with pytest.raises(ValueError):
        HeliostatDesign.from_dict(data)


def test_heliostat_design_rejects_empty_facet_list():
    with pytest.raises(ValueError):
        HeliostatDesign([])


# ---------------------------------------------------------------------------
# (f) silhouette


def test_silhouette_rect_matches_rectangle_to_tight_tolerance():
    design = rect_heliostat()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a plain rect must never hit the fallback
        sil = design.silhouette(n_vertices=360)
    assert sil.area_mm2() == pytest.approx(15e6, rel=5e-3)

    for u, v in sil.vertices_mm:
        d_u = 2500.0 - abs(u)
        d_v = 1500.0 - abs(v)
        assert min(abs(d_u), abs(d_v)) < 5.0  # every vertex within 5 mm of an edge


def test_silhouette_flower_area_and_star_shape():
    design = flower(petals_as_facets=True)  # hub_radius_mm=0: petals touch at centre
    sketch_area = flower(petals_as_facets=False).facets[0].region.area_mm2(resolution=2048)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # star-shaped about the centre: no fallback expected
        sil = design.silhouette(n_vertices=72)

    sil_area = sil.area_mm2(resolution=2048)
    # A 72-gon inscribed on a curved boundary chord-undershoots a little;
    # it must not be short by more than that sampling artefact.
    assert sil_area >= sketch_area * 0.98
    assert sil_area <= sketch_area * 1.05

    radii = np.hypot(sil.vertices_mm[:, 0], sil.vertices_mm[:, 1])
    assert radii.max() == pytest.approx(2000.0, abs=5.0)  # petal tips: hub(0) + length
    assert radii.min() < 0.5 * radii.max()  # a star, not a convex hull


def test_silhouette_grid_with_gap_fills_the_gap():
    design = grid_facets(2, 2, 1000.0, 800.0, gap_mm=100.0)
    facet_area_sum = sum(f.region.area_mm2() for f in design.facets)

    # The gap forms a "+" reaching the outer edge along the axis-aligned
    # azimuths, which are not star-shaped from the centroid -- expect (and
    # accept) the documented fallback warning there.
    with pytest.warns(UserWarning, match="silhouette"):
        sil = design.silhouette(n_vertices=72)

    sil_area = sil.area_mm2(resolution=2048)
    outer_area = (2 * 1000.0 + 100.0) * (2 * 800.0 + 100.0)
    assert sil_area > facet_area_sum  # the gap is filled in
    assert sil_area == pytest.approx(outer_area, rel=1e-2)


def test_silhouette_is_a_plain_polygon():
    sil = rect_heliostat().silhouette(n_vertices=36)
    assert isinstance(sil, aperture.Polygon)
    # No bespoke serialisation needed: it round-trips through the region
    # registry like any other polygon.
    rebuilt = aperture.region_from_dict(sil.to_dict())
    uu = np.linspace(-3000, 3000, 50)
    vv = np.linspace(-2000, 2000, 50)
    assert np.array_equal(
        sil.contains(uu[None, :], vv[:, None]), rebuilt.contains(uu[None, :], vv[:, None])
    )


# ---------------------------------------------------------------------------
# (g) preview


def test_preview_rect_heliostat_headless():
    ax = rect_heliostat().preview()
    try:
        assert len(ax.get_images()) >= 1
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert len(dashed) >= 1
    finally:
        plt.close(ax.figure)


def test_preview_flower_headless():
    ax = flower(petals_as_facets=True).preview()
    try:
        assert len(ax.get_images()) >= 1
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert len(dashed) >= 1
    finally:
        plt.close(ax.figure)


def test_preview_accepts_existing_axes():
    fig, ax = plt.subplots()
    try:
        returned = rect_heliostat().preview(ax=ax)
        assert returned is ax
    finally:
        plt.close(fig)


def test_preview_without_silhouette_has_no_dashed_line():
    ax = rect_heliostat().preview(show_silhouette=False)
    try:
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
        assert len(dashed) == 0
    finally:
        plt.close(ax.figure)
