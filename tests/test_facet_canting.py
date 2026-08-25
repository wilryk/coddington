"""Tests for the facet-canting fix in heliostat.geometry.design.

The bug: a faceted design tiling the same aperture as a solid rectangle,
with zero gaps, is the same optical surface merely cut into pieces -- it
should trace almost identically to the rectangle. Before this fix it did
not, whenever the underlying figure was astigmatic (``surface="twisting"``,
i.e. :class:`~heliostat.geometry.design.ZernikeAstig`): ``cant_on_axis``
gives every facet the same *rotationally-symmetric* aim, but an astigmatic
figure's true local slope at an off-axis facet centre is not rotationally
symmetric, so the cant and the figure disagreed by several milliradians --
doubled by reflection, that is metres of displacement at a few tens of
metres' slant range.

The fix, :func:`~heliostat.geometry.design.cant_to_surface`
(``cant_focal_mm="auto"`` on :func:`grid_facets`/:func:`flower`): cant each
facet to the continuous ``surface``'s own local gradient at that facet's
offset, computed via :meth:`Surface.sag_and_slopes` rather than an
independent on-axis focus -- "the facets reconstruct the continuous
surface, with steps at the joins" is the product framing this implements.

Coverage:

(a) a 1x1 grid must be EXACTLY the rectangle -- no offset, so no cant
    disagreement is possible; this is the invariant that isolated the bug
    in the first place, and it is checked at trace level (not just
    facet-structure level) so a future change cannot quietly break it while
    keeping the facet objects looking right;
(b) a 2x2 and 5x3 grid, tiling the same aperture with zero gaps, must
    reproduce the rectangle's peak flux and rms spot radius within a stated
    tolerance -- for both ``twisting`` (astigmatic) and ``spherical``
    figures, at two field positions;
(c) Spherical must not regress: ``cant_focal_mm="auto"`` on a
    :class:`~heliostat.geometry.design.Spherical` figure must give
    bit-identical ``cant_normal`` to today's ``cant_focal_mm=<the sphere's
    own focal>`` (:func:`~heliostat.geometry.design.cant_on_axis`);
(d) explicit canting (``cant_focal_mm=0`` and a positive number) is
    untouched by this change -- structural check that those paths still go
    through the same code (uncanted / :func:`cant_on_axis` respectively);
(e) a unit-level check of :func:`cant_to_surface` itself against a
    hand-computed tangent-plane normal, and its behaviour on ``Flat``
    (zero gradient everywhere -- no tilt) and on an unresolved
    ``Spherical(focal_mm="slant")`` (raises, since there is no single focal
    length to cant toward without a ``cant_focal_mm`` to resolve against).

Tolerance: 2%, on peak flux and rms spot radius, for the same-aperture
zero-gap 2x2/5x3 grids against the rectangle. Justification: a facet's own
figure is exactly the continuous surface's local Taylor expansion about its
offset for every quadratic figure this module ships (verified in
test_design.py's Surface tests: both Spherical and ZernikeAstig are
homogeneous quadratics, so the residual after subtracting the linear cant
term is exactly the facet's own local figure, with nothing left over) --
so the fix leaves no error from the figure/cant split itself. What remains
is a real, physical difference: each facet reflects from its own flat local
frame with a hard step at the joins, rather than the continuous surface's
smoothly-varying tangent plane -- an actual faceted mirror does this too.
Measured across two field positions, two figures and both grid sizes below,
that residual is 0.00-0.05% on power (conserved, as expected -- canting
does not change reflective area) and 0.00-1.10% on peak flux / rms radius
(the 5x3 spherical case is the worst, since its facets are smallest
relative to the aperture and thus most numerous joins); 2% keeps comfortable
margin above the observed worst case while still catching the original bug,
whose 2x2/5x3 peak-flux errors were 55-70% (before this fix, on-axis cant
under a twisting figure) -- two orders of magnitude past this tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heliostat.geometry import aperture
from heliostat.geometry.design import (
    Facet,
    Flat,
    Spherical,
    ZernikeAstig,
    cant_on_axis,
    cant_to_surface,
    flower,
    grid_facets,
    rect_heliostat,
)
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from test_mc_parity import MC_ROOT, _geometry_for

# ---------------------------------------------------------------------------
# shared fixtures: two real solved heliostats, axicon config, from the
# mc_parity golden fixtures (same provenance as test_design_tracing.py).

_SUMMARY = pd.read_csv(MC_ROOT / "axicon" / "summary.csv")
_SECONDARY, _RECEIVER = _geometry_for("axicon")
_KERNEL = sunshape_kernel("super_gauss")


def _row(heliostat_id: int, step_key: str = "20260321_0939"):
    return _SUMMARY[
        (_SUMMARY.heliostat_id == heliostat_id) & (_SUMMARY.step_key.astype(str) == step_key)
    ].iloc[0]


# Two field positions: one on the -y axis, one on the +x axis, both with a
# real solved astigmatic figure and a real slant range.
_CASES = [_row(241), _row(574)]


def _trace_args(row):
    return (
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        _SECONDARY,
        _RECEIVER,
    )


def _flux_metrics(result: dict) -> tuple[float, float, float]:
    """``(power_w, peak_flux_w_m2, rms_radius_mm)`` from a cone-trace result.

    ``rms_radius_mm`` is about the flux-weighted centroid, mirroring
    ``heliostat.metrics.spot_metrics``'s ray-based definition but computed
    from the deterministic flux grid the cone backend returns instead of
    sampled rays.
    """
    flux = result["flux"]
    u_edges, v_edges = result["u_edges"], result["v_edges"]
    uc = 0.5 * (u_edges[:-1] + u_edges[1:])
    vc = 0.5 * (v_edges[:-1] + v_edges[1:])
    uu, vv = np.meshgrid(uc, vc)
    bin_area_m2 = ((u_edges[1] - u_edges[0]) / 1000.0) * ((v_edges[1] - v_edges[0]) / 1000.0)
    weight = flux * bin_area_m2
    total = float(weight.sum())
    cx = float((uu * weight).sum() / total)
    cy = float((vv * weight).sum() / total)
    r2 = (uu - cx) ** 2 + (vv - cy) ** 2
    rms = float(np.sqrt((r2 * weight).sum() / total))
    return float(result["power_w"]), float(flux.max()), rms


def _twisting_figure(row) -> ZernikeAstig:
    # Legacy-path sign convention: the design frame negates c4/c5 relative
    # to the solve's own (see tests/test_design_tracing.py's module docstring).
    return ZernikeAstig(row.c3, -row.c4, -row.c5)


def _spherical_figure(row) -> Spherical:
    return Spherical(row.radius_m * 1000.0)


RECON_TOLERANCE = 0.02  # 2%, see module docstring for the measured residual


# ---------------------------------------------------------------------------
# (a) 1x1 grid is EXACTLY the rectangle -- no offset, no room for a cant
# disagreement, so this must hold regardless of surface or cant mode.


@pytest.mark.parametrize("row", _CASES, ids=lambda r: f"helio{r.heliostat_id}")
@pytest.mark.parametrize(
    "figure_fn", [_twisting_figure, _spherical_figure], ids=["twisting", "spherical"]
)
def test_grid_1x1_matches_rect_exactly(row, figure_fn):
    figure = figure_fn(row)
    args = _trace_args(row)

    rect = rect_heliostat(surface=figure)
    grid = grid_facets(
        n_u=1, n_v=1, facet_w_mm=5000.0, facet_h_mm=3000.0, gap_mm=0.0,
        surface=figure, cant_focal_mm="auto",
    )

    p0, pk0, rms0 = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=rect))
    p1, pk1, rms1 = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=grid))

    assert p1 == pytest.approx(p0, rel=1e-9)
    assert pk1 == pytest.approx(pk0, rel=1e-9)
    assert rms1 == pytest.approx(rms0, rel=1e-9)


# ---------------------------------------------------------------------------
# (b) 2x2 / 5x3 grids reconstruct the rectangle within RECON_TOLERANCE.


@pytest.mark.parametrize("row", _CASES, ids=lambda r: f"helio{r.heliostat_id}")
@pytest.mark.parametrize(
    "figure_fn", [_twisting_figure, _spherical_figure], ids=["twisting", "spherical"]
)
@pytest.mark.parametrize("n_u, n_v", [(2, 2), (5, 3)])
def test_grid_reconstructs_rect_within_tolerance(row, figure_fn, n_u, n_v):
    figure = figure_fn(row)
    args = _trace_args(row)

    rect = rect_heliostat(surface=figure)
    grid = grid_facets(
        n_u=n_u, n_v=n_v, facet_w_mm=5000.0 / n_u, facet_h_mm=3000.0 / n_v, gap_mm=0.0,
        surface=figure, cant_focal_mm="auto",
    )

    p0, pk0, rms0 = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=rect))
    p1, pk1, rms1 = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=grid))

    assert p1 == pytest.approx(p0, rel=RECON_TOLERANCE)
    assert pk1 == pytest.approx(pk0, rel=RECON_TOLERANCE)
    assert rms1 == pytest.approx(rms0, rel=RECON_TOLERANCE)


def test_grid_on_axis_cant_fails_the_reconstruction_bound_for_twisting():
    """The regression this whole file exists to catch: without the fix
    (``cant_focal_mm`` set to a plain on-axis focal, today's other explicit
    mode), a twisting 5x3 grid blows the 2% bound by a wide margin --
    proof that RECON_TOLERANCE is tight enough to matter, not a bound
    nothing could ever fail."""
    row = _CASES[0]
    figure = _twisting_figure(row)
    args = _trace_args(row)
    rect = rect_heliostat(surface=figure)
    on_axis_grid = grid_facets(
        n_u=5, n_v=3, facet_w_mm=1000.0, facet_h_mm=1000.0, gap_mm=0.0,
        surface=figure, cant_focal_mm=row.radius_m * 1000.0,
    )
    _, pk0, _ = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=rect))
    _, pk1, _ = _flux_metrics(trace_heliostat_cone(*args, _KERNEL, design=on_axis_grid))
    assert abs(pk1 - pk0) / pk0 > 5 * RECON_TOLERANCE


# ---------------------------------------------------------------------------
# (c) Spherical must not regress: "auto" == on-axis at the sphere's own focal.


@pytest.mark.parametrize("n_u, n_v", [(1, 1), (2, 2), (5, 3)])
def test_auto_cant_matches_on_axis_exactly_for_spherical_grid(n_u, n_v):
    focal_mm = 45000.0
    surface = Spherical(focal_mm)
    on_axis = grid_facets(
        n_u=n_u, n_v=n_v, facet_w_mm=5000.0 / n_u, facet_h_mm=3000.0 / n_v,
        surface=surface, cant_focal_mm=focal_mm,
    )
    auto = grid_facets(
        n_u=n_u, n_v=n_v, facet_w_mm=5000.0 / n_u, facet_h_mm=3000.0 / n_v,
        surface=surface, cant_focal_mm="auto",
    )
    for fa, fb in zip(on_axis.facets, auto.facets):
        assert np.array_equal(fa.cant_normal, fb.cant_normal)
        assert fa.offset_mm == fb.offset_mm


def test_auto_cant_matches_on_axis_exactly_for_spherical_flower():
    focal_mm = 30000.0
    surface = Spherical(focal_mm)
    on_axis = flower(hub_radius_mm=800.0, surface=surface, cant_focal_mm=focal_mm)
    auto = flower(hub_radius_mm=800.0, surface=surface, cant_focal_mm="auto")
    for fa, fb in zip(on_axis.facets, auto.facets):
        assert np.array_equal(fa.cant_normal, fb.cant_normal)


# ---------------------------------------------------------------------------
# (d) explicit canting is untouched: 0 -> uncanted, a number -> cant_on_axis.


def test_cant_focal_none_is_still_uncanted():
    design = grid_facets(2, 2, 2500.0, 1500.0, surface=ZernikeAstig(1e-7, 2e-7, -1e-7))
    assert all(f.cant_normal is None for f in design.facets)


def test_cant_focal_number_still_uses_cant_on_axis():
    surface = ZernikeAstig(1e-7, 2e-7, -1e-7)
    focal_mm = 50000.0
    design = grid_facets(2, 2, 2500.0, 1500.0, surface=surface, cant_focal_mm=focal_mm)
    uncanted = grid_facets(2, 2, 2500.0, 1500.0, surface=surface).facets
    expected = cant_on_axis(uncanted, focal_mm)
    for got, want in zip(design.facets, expected):
        assert np.allclose(got.cant_normal, want.cant_normal)


# ---------------------------------------------------------------------------
# (e) cant_to_surface unit-level checks.


def test_cant_to_surface_matches_hand_computed_gradient():
    c3, c4, c5 = 3e-8, -5e-8, 2e-8
    surface = ZernikeAstig(c3, c4, c5)
    ou, ov = 2000.0, 1000.0
    facet = Facet(region=aperture.Rect(1000.0, 1000.0), surface=surface, offset_mm=(ou, ov))
    (canted,) = cant_to_surface([facet], surface)

    _, dsdu, dsdv = surface.sag_and_slopes(ou, ov)
    expected = np.array([-dsdu, -dsdv, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(canted.cant_normal, expected)


def test_cant_to_surface_flat_is_uncanted_at_any_offset():
    surface = Flat()
    facet = Facet(region=aperture.Rect(1000.0, 1000.0), surface=surface, offset_mm=(1800.0, -900.0))
    (canted,) = cant_to_surface([facet], surface)
    assert np.allclose(canted.cant_normal, [0.0, 0.0, 1.0])


def test_cant_to_surface_rejects_unresolved_slant_placeholder():
    surface = Spherical("slant")
    facet = Facet(region=aperture.Rect(1000.0, 1000.0), surface=surface, offset_mm=(500.0, 500.0))
    with pytest.raises(ValueError, match="slant"):
        cant_to_surface([facet], surface)


def test_grid_facets_auto_cant_with_unresolved_slant_raises():
    with pytest.raises(ValueError, match="slant"):
        grid_facets(
            2, 2, 2500.0, 1500.0, surface=Spherical("slant"), cant_focal_mm="auto"
        )
