"""Cone-optics backend validated against Monte Carlo for mutual shading,
mutual blocking, and a tight secondary aperture rim -- the three edge kinds
``trace_heliostat_cone``'s ``occluders``/``shadow_body``/``mask_nodes``
machinery exists to resolve (see that function's docstring).

The MC side needs a reference that itself knows about neighbour occlusion,
which :func:`heliostat.trace.mc.trace_heliostat` does not model in-engine.
Rather than write a second, from-scratch MC tracer, :func:`_occluder_aware_trace`
below builds one by ray-vs-rectangle filtering ``trace_heliostat``'s own
``return_paths=True`` output -- the same ``_blocked_mask`` test the analytic
shading model (:mod:`heliostat.geometry.shading`) already uses, applied to
the traced rays' own source->mirror and mirror->secondary legs instead of to
a fixed sample grid. See its docstring for exactly what it does and does not
capture.

Runtime: the heaviest trace here is 1,000,000 rays (~0.5 s), cached in a
module-scoped fixture and reused across both cone orders and both apertures
in the rim test; everything else uses 100,000-500,000 rays. Total added
runtime is a few seconds, well inside the 120 s budget.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heliostat.field import HeliostatField, neighbour_pairs
from heliostat.geometry.shading import MirrorGeometry, _blocked_mask, sun_vector
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone
from heliostat.trace.mc import MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM, trace_heliostat
from test_mc_parity import _geometry_for, _load_fixture

FIXTURES_ROOT = Path(__file__).parent / "fixtures"
SHADING_ROOT = FIXTURES_ROOT / "shading"

# MC references here (``_occluder_aware_trace`` / ``big_mc_trace``) call
# ``trace_heliostat`` without ``sampler=``, so they draw from the app-wide
# default (Buie, since the sunshape swap) -- this module-level cone kernel
# must match, or every cross-backend comparison below is apples to oranges.
KERNEL = sunshape_kernel("buie")
MID_MORNING_STEP = "20260321_0939"
TIGHT_APERTURE_HELIOSTAT = 574
BLOCKING_HELIOSTAT = 414


def _occluder_aware_trace(
    row, secondary, receiver, n_rays, seed, occluders, **trace_kwargs
) -> dict:
    """MC reference *with* mutual shading/blocking, from ``trace_heliostat``'s
    own surviving rays -- not a from-scratch reimplementation.

    ``return_paths=True`` returns every surviving ray's ``[source, mirror,
    secondary, receiver]`` vertices. A ray is dropped if its source->mirror
    leg (shading) or its mirror->secondary leg (blocking) crosses any
    occluder rectangle, tested with :func:`_blocked_mask` -- per-ray
    directions, since both legs vary ray to ray (the source disk is sampled,
    and the reflected beam converges). Neither leg is bounded to stop
    exactly at its second vertex: the same unbounded-ray convention
    ``shading_blocking``/``occlusion_efficiency`` use for the (effectively
    infinite) sun direction and the aim-point direction, which the source
    disk and the secondary hit respectively stand in for here.

    ``return_paths`` only carries rays that already reached the receiver
    window, so this recomputes power (and centroid/rms for a non-empty
    result) over a subset of an already in-window population -- a
    power-in-window comparison, not a full loss-chain reproduction.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    out = trace_heliostat(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        n_rays,
        rng,
        return_paths=True,
        **trace_kwargs,
    )
    src, mir, con, rec = out["paths"]
    keep = np.ones(mir.shape[1], dtype=bool)
    if occluders:
        keep &= ~_blocked_mask(mir.T, (src - mir).T, occluders)
        keep &= ~_blocked_mask(mir.T, (con - mir).T, occluders)
    xy = rec[:2, keep].T
    n = int(keep.sum())
    watts_per_ray = out["watts_per_ray"]
    result = {"xy": xy, "n": n, "power_w": n * watts_per_ray, "watts_per_ray": watts_per_ray}
    if n > 0:
        centre = xy.mean(axis=0)
        r = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
        result["centroid"] = centre
        result["rms"] = float(np.sqrt(np.mean(r**2)))
    return result


def _assert_counter_invariant(counters: dict, label: str) -> None:
    parts = (
        counters["valid"]
        + counters["masked"]
        + counters["blocked"]
        + counters["node_fallback"]
        + counters["unresolved"]
    )
    assert parts == counters["samples"], f"{label}: counters {counters} do not sum to samples"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def axicon_h574_row():
    _, _, summary = _load_fixture("axicon")
    return summary.loc[(TIGHT_APERTURE_HELIOSTAT, MID_MORNING_STEP)]


@pytest.fixture(scope="module")
def secondary_footprint_p85(axicon_h574_row):
    """85th percentile of h574's secondary-footprint radius (mm), from a
    100,000-ray trace with ``return_secondary_hits`` -- the "tight" aperture
    radius the rim test probes (measured ~13278 mm, matching the task's
    quoted ~13270 mm)."""
    row = axicon_h574_row
    secondary, receiver = _geometry_for("axicon")
    rng = np.random.default_rng(np.random.SeedSequence((20260815, 574, 100_000)))
    out = trace_heliostat(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        100_000,
        rng,
        return_secondary_hits=True,
    )
    r = np.hypot(out["secondary_xy"][0], out["secondary_xy"][1])
    return float(np.percentile(r, 85))


RIM_MC_RAYS = 3_000_000


@pytest.fixture(scope="module")
def big_mc_trace(axicon_h574_row):
    """MC reference for h574's axicon mid_morning case, at the standard
    (14000 mm) secondary aperture -- computed once and reused for both cone
    orders and both apertures the rim test checks, since the tight-aperture
    subset is exactly the rays whose secondary-hit radius is within the
    tight radius (aperture is a pure post-hoc mask on which rays survive the
    cone; no upstream stage of the trace depends on its value).

    Judgment call: the task names "a 1M-ray MC" and a 0.15% standard-aperture
    tolerance. At 1,000,000 rays this case's own shot noise is
    se(power)/power ~= 0.14% (binomial, ~324,000 landed rays) -- close enough
    to the 0.15% target that whether the assertion passes depends on the RNG
    seed, which makes the test flaky rather than a real backend check (one
    seed measured a 0.19-0.24% cone/MC gap driven mostly by that noise, not
    by the backend). Runtime is not the constraint (1M rays traces in ~0.5s;
    the 120s budget has ample room), so this uses 3,000,000 rays instead,
    which brings shot noise down to ~0.08% -- the fixed seed below then
    measures cone/MC gaps of 0.027% (order 1) and 0.074% (order 2), both
    comfortably inside 0.15% with margin instead of sitting on the noise
    floor.
    """
    row = axicon_h574_row
    secondary, receiver = _geometry_for("axicon")
    rng = np.random.default_rng(np.random.SeedSequence((20260815, 574, RIM_MC_RAYS)))
    out = trace_heliostat(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        RIM_MC_RAYS,
        rng,
        return_paths=True,
    )
    return row, secondary, receiver, out


# ---------------------------------------------------------------------------
# 1. Tight-aperture rim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_tight_aperture_rim_power(order, big_mc_trace, secondary_footprint_p85):
    row, secondary, receiver, mc = big_mc_trace
    _src, _mir, con, _rec = mc["paths"]
    watts_per_ray = mc["watts_per_ray"]

    power_full_mc = con.shape[1] * watts_per_ray
    tight_r = secondary_footprint_p85
    n_tight = int((np.hypot(con[0], con[1]) <= tight_r).sum())
    power_tight_mc = n_tight * watts_per_ray

    cone_full = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        order=order,
    )
    tight_secondary = dataclasses.replace(secondary, aperture_radius_mm=tight_r)
    cone_tight = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        tight_secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        order=order,
    )
    _assert_counter_invariant(cone_full["counters"], f"order={order} standard aperture")
    _assert_counter_invariant(cone_tight["counters"], f"order={order} tight aperture")

    d_full = abs(cone_full["power_w"] - power_full_mc) / power_full_mc
    d_tight = abs(cone_tight["power_w"] - power_tight_mc) / power_tight_mc
    print(
        f"\norder={order} standard aperture (14000mm): cone={cone_full['power_w']:.2f}W "
        f"mc={power_full_mc:.2f}W diff={d_full * 100:.4f}%"
    )
    print(
        f"order={order} tight aperture ({tight_r:.1f}mm, p85): cone={cone_tight['power_w']:.2f}W "
        f"mc={power_tight_mc:.2f}W diff={d_tight * 100:.4f}%"
    )
    assert d_tight < 0.008, f"order={order}: tight-aperture power diff {d_tight * 100:.3f}% >= 0.8%"
    assert d_full < 0.0015, (
        f"order={order}: standard-aperture power diff {d_full * 100:.4f}% >= 0.15%"
    )


# ---------------------------------------------------------------------------
# 2. Penumbra shadow
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def penumbra_scene(axicon_h574_row):
    """A single occluder ~30 m toward the sun from h574, sized to shadow
    exactly the u > 0 half of the mirror by construction.

    Two mirrors with the same orientation are parallel planes, so a
    parallel-sun-projected rectangle from one onto the other's plane lands
    at an offset that is *independent of the distance along the sun
    direction* between them (see
    ``heliostat.geometry.shading.shadow_quad_uv``'s derivation): adding
    ``to_sun * 30000`` moves the occluder 30 m toward the sun without
    changing where its shadow falls in the mirror's own (u, v) frame, and
    adding ``mirror.half_width * mirror.u`` on top of that places the
    shadow's near edge exactly on the mirror's own centreline. The occluder
    is stretched 5x taller than the mirror so the edge is a straight vertical
    line across the full aperture height, not curved or partial in v.
    """
    row = axicon_h574_row
    mirror = MirrorGeometry.build(
        row.x_mm, row.y_mm, row.rot_az_deg, row.rot_el_deg, MIRROR_HALF_X_MM, MIRROR_HALF_Y_MM
    )
    to_sun = sun_vector(row.solar_az_deg, row.solar_el_deg)
    occ_centre = mirror.centre + to_sun * 30000.0 + mirror.half_width * mirror.u
    occluder = MirrorGeometry(
        centre=occ_centre,
        normal=mirror.normal,
        u=mirror.u,
        v=mirror.v,
        half_width=mirror.half_width,
        half_height=mirror.half_height * 5.0,
    )
    return row, occluder


def test_penumbra_shadow_ratio_matches_occluder_aware_mc(penumbra_scene):
    row, occluder = penumbra_scene
    secondary, receiver = _geometry_for("axicon")
    seed = (20260815, 574, 1)  # 1 = "penumbra" scenario tag

    mc_unocc = _occluder_aware_trace(row, secondary, receiver, 500_000, seed, [])
    mc_occ = _occluder_aware_trace(row, secondary, receiver, 500_000, seed, [occluder])
    ratio_mc = mc_occ["power_w"] / mc_unocc["power_w"]

    cone_unocc = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
    )
    cone_occ = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        occluders=[occluder],
    )
    ratio_cone = cone_occ["power_w"] / cone_unocc["power_w"]

    _assert_counter_invariant(cone_unocc["counters"], "penumbra unoccluded")
    _assert_counter_invariant(cone_occ["counters"], "penumbra occluded")

    c = cone_occ["counters"]
    print(f"\npenumbra: ratio_mc={ratio_mc:.5f} ratio_cone={ratio_cone:.5f} counters={c}")
    assert abs(ratio_cone - ratio_mc) < 0.005, (
        f"power ratio |d|={abs(ratio_cone - ratio_mc):.5f} exceeds 0.5% absolute "
        f"(mc={ratio_mc:.5f}, cone={ratio_cone:.5f})"
    )
    # Penumbra: some sample must be neither fully passed nor fully lost.
    # Per-sample transmitted fraction isn't itself returned, so this is
    # exactly the task's documented proxy: masked > 0.
    assert c["masked"] > 0, "expected at least one partially-transmitted (penumbra) sample"


# ---------------------------------------------------------------------------
# 3. Real blocking scene (from the shading fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blocking_scene():
    """Heliostat 414's real blocking neighbours at mid_morning.

    ``tests/fixtures/shading/occlusion.csv`` reports ``eta_block=0.824`` for
    (414, 20260321_0939) -- the shading fixture's own point-sampled
    reference, generated by the private repo. Of 414's six neighbours within
    the fixture's 10 m search radius, only 343 and 344 actually block
    anything (checked with a 201x201 point grid before writing this test);
    the other four contribute nothing and are left out.
    """
    _, _, summary = _load_fixture("axicon")
    row = summary.loc[(BLOCKING_HELIOSTAT, MID_MORNING_STEP)]

    expected = json.loads((SHADING_ROOT / "expected.json").read_text(encoding="utf-8"))
    geometry = pd.read_csv(SHADING_ROOT / "geometry.csv")
    group = geometry[geometry["step_key"] == MID_MORNING_STEP].reset_index(drop=True)
    field = HeliostatField(
        x_mm=group["x_mm"].to_numpy(float),
        y_mm=group["y_mm"].to_numpy(float),
        ids=group["heliostat_id"].to_numpy(int),
        mirror_width_mm=expected["mirror_width_mm"],
        mirror_height_mm=expected["mirror_height_mm"],
    )
    neighbours = neighbour_pairs(field, expected["search_radius_mm"])
    idx = int(np.flatnonzero(field.ids == BLOCKING_HELIOSTAT)[0])

    occluders = [
        MirrorGeometry.build(
            group.iloc[j].x_mm,
            group.iloc[j].y_mm,
            group.iloc[j].rot_az_deg,
            group.iloc[j].rot_el_deg,
            MIRROR_HALF_X_MM,
            MIRROR_HALF_Y_MM,
        )
        for j in neighbours[idx]
        if int(group.iloc[j].heliostat_id) in (343, 344)
    ]
    assert len(occluders) == 2
    return row, occluders


def test_blocking_scene_ratio_matches_occluder_aware_mc(blocking_scene):
    row, occluders = blocking_scene
    secondary, receiver = _geometry_for("axicon")
    seed = (20260815, 414, 2)  # 2 = "blocking" scenario tag

    mc_unocc = _occluder_aware_trace(row, secondary, receiver, 500_000, seed, [])
    mc_occ = _occluder_aware_trace(row, secondary, receiver, 500_000, seed, occluders)
    ratio_mc = mc_occ["power_w"] / mc_unocc["power_w"]

    # grid=(40, 24), not the (20, 12) default: at the default, this
    # blocking edge happens to sit close to the mirror sample grid's own
    # cell boundaries and the ratio comes out 0.8457 -- 2.2 points off,
    # purely spatial-sample aliasing (confirmed by sweeping grid density
    # before writing this test: (40,24)->0.8240, (80,48)->0.8246,
    # (160,96)->0.8244, all within 0.1% of each other and of the MC
    # reference below, while (20,12) sits alone at 0.8457). This is the
    # cone backend's spatial sample grid showing the same coarse-vs-fine
    # sensitivity test_polygon_shading.py documents for pure point sampling
    # -- not a backend bug, but real, and worth a finer grid for a sharp
    # nearby blocking edge; flagging per the task instructions.
    grid = (40, 24)
    cone_unocc = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        grid=grid,
    )
    cone_occ = trace_heliostat_cone(
        row.x_mm,
        row.y_mm,
        row.rot_az_deg,
        row.rot_el_deg,
        row.c3,
        row.c4,
        row.c5,
        row.solar_az_deg,
        row.solar_el_deg,
        secondary,
        receiver,
        KERNEL,
        mask_nodes=16,
        grid=grid,
        occluders=occluders,
    )
    ratio_cone = cone_occ["power_w"] / cone_unocc["power_w"]

    _assert_counter_invariant(cone_unocc["counters"], "blocking unoccluded")
    _assert_counter_invariant(cone_occ["counters"], "blocking occluded")

    print(f"\nblocking h{BLOCKING_HELIOSTAT}: ratio_mc={ratio_mc:.5f} ratio_cone={ratio_cone:.5f}")
    assert abs(ratio_cone - ratio_mc) < 0.007, (
        f"power ratio |d|={abs(ratio_cone - ratio_mc):.5f} exceeds 0.7% absolute "
        f"(mc={ratio_mc:.5f}, cone={ratio_cone:.5f})"
    )
