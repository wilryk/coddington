"""Sweep driver: trace a whole heliostat field across one or more days and
write a stored run.

Per timestep, this module:

1. solves the aiming pointing/figure for every heliostat (a plain scalar
   loop -- the aiming solves in :mod:`heliostat.geometry.aiming` are not
   vectorised, and 600 of them per timestep is not the cost centre);
2. builds mirror geometry + aim points and runs
   :func:`~heliostat.geometry.shading.polygon_occlusion` against a
   neighbour list sized for that step's own sun (and beam) elevation,
   to get per-heliostat ``eta_shade``/
   ``eta_block``/``eta_secondary``/``eta_union`` scalars;
3. dispatches one ray/cone trace per heliostat across a
   :class:`multiprocessing.Pool` (or serially when ``workers=1``);
4. writes one :class:`~heliostat.store.TimestepResult` per timestep: maps
   are stored **unoccluded** (the store's own convention -- see
   :mod:`heliostat.store`), with the occlusion scalars recorded as summary
   columns instead of baked into the maps.

Standard optics
----------------
:func:`standard_optics` reproduces the three configurations this package's
golden fixtures (``tests/fixtures/mc_parity/*``, ``tests/test_aiming.py``)
were traced/solved with: same secondary, receiver, throughput and aim-point
convention. Building a sweep on these numbers means a sweep run's per-
heliostat pointing and figure can be spot-checked against those fixtures at
matching heliostat positions and sun angles.

Judgment calls (flagged, not hidden)
-------------------------------------
* **Site default.** The task brief guessed ``(-9.4, -52.0, -3)``. The actual
  fixture (``tests/fixtures/energy/manifest.json`` -> ``"site"``) records
  latitude **-10.0**, not -9.4. :data:`DEFAULT_SITE` uses the real number.
* **Seed contract.** The brief's phrasing ``int(step.key)`` cannot work
  literally -- ``TimeStep.key`` is ``"20260320_0642"`` and ``int()`` on that
  string raises. ``tests/test_mc_parity.py`` (the fixture this convention is
  pinned to) computes ``int(step_key.replace("_", ""))``; this module does
  the same.
* **Mirror size is pinned upstream of this module.** With no ``design=``,
  :func:`heliostat.trace.mc.trace_heliostat` and
  :func:`heliostat.trace.cone.trace_heliostat_cone` always trace a fixed
  5 m x 3 m rectangle (``MIRROR_HALF_X_MM``/``MIRROR_HALF_Y_MM`` module
  constants) regardless of what a field's own ``mirror_width_mm``/
  ``mirror_height_mm`` say. Those field dimensions still drive the
  occlusion/shading geometry (mirror rectangles, neighbour search radius).
  ``run_sweep`` warns when the two disagree and no ``design`` was given.
* **No secondary self-shading.** ``polygon_occlusion`` is called with
  ``secondary=None`` (so ``eta_secondary`` is always 1.0), matching the
  fixture manifest's own ``"traced_secondary": false`` -- a secondary body
  that shades the field is a real, separate effect this sweep does not
  model, not an oversight of what the golden fixtures already skip.
* **Cone-mode flux maps store bin *power*, not bin flux.** The task brief
  says "float32 W/m2 maps"; but ``RunStore.field_flux``/``heliostat_flux``
  always compute ``counts.sum() * flux_scale(...) / bin_area_m2`` regardless
  of ``flux_kind`` (see ``tests/test_store.py::TestFluxScale`` -- the
  ``"analytic"`` test explicitly comments "as if counts already held
  power"). For that formula to reproduce W/m2 from a cone trace's flux map,
  the array written to ``flux/<key>.npy`` for a cone mode has to be
  ``out["flux"] * bin_area_m2`` (bin power at DNI=1000, throughput=1), not
  the flux map itself. Storing raw flux would make every read-path consumer
  double-apply the bin-area division. Documented here since it is the one
  place the code deliberately does not match the brief's literal wording.
* **Ray-derived summary columns are NaN for cone modes.** ``rays_emitted``,
  ``rays_landed``, ``transmission``, ``rays_outside_window`` have no cone-
  mode analogue (mirror-surface sample count is not a ray budget); left NaN
  rather than repurposed to avoid inventing a false ray count.
* **No raw-ray retention.** ``cfg.storage.raw_rays`` is always ``"none"``:
  ``run_sweep``'s signature (per the task brief) has no raw-ray toggle, and
  writing raw rays for a 600-heliostat sweep would multiply the store's size
  for no deliverable that needs it. Flux maps + summary rows are written
  either way.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from .field import HeliostatField, neighbour_pairs
from .geometry.aiming import Solution, solve_axicon, solve_cassegrain, solve_prime_focus
from .geometry.aiming import aim_points_mm as _aim_points_mm
from .geometry.receiver import FlatWindowReceiver, Receiver
from .geometry.secondary import AxiconSecondary, CassegrainSecondary, NoSecondary, Secondary
from .geometry.shading import (
    MirrorGeometry,
    build_geometries,
    min_beam_elevation_deg,
    polygon_occlusion,
    search_radius_for,
)
from .metrics import spot_metrics
from .solar import build_time_grid
from .store import RunStore, TimestepResult, flux_scale
from .trace import mc as _mc
from .trace.cone import grid_for_density, sunshape_kernel, trace_heliostat_cone
from .trace.mc import trace_heliostat
from .trace.modes import MODES, TraceMode
from .trace.samplers import BuieSampler

__all__ = ["run_sweep", "standard_optics", "OpticsSpec", "DEFAULT_SITE", "WINDOW_MM", "GRID_SIZE"]

# ---------------------------------------------------------------------------
# Standard optics -- see module docstring for where each number comes from.
# ---------------------------------------------------------------------------
WINDOW_MM = 2000.0
GRID_SIZE = 128

SECONDARY_HEIGHT_MM = 27000.0  # axicon apex height / cassegrain build height
RECEIVER_HEIGHT_MM = 7000.0  # ground receiver, axicon + cassegrain
AXICON_ANGLE_DEG = 20.0
AXICON_APERTURE_RADIUS_MM = 14000.0
PRIME_FOCUS_HEIGHT_MM = 35335.0
CASSEGRAIN_FOCUS_HEIGHT_MM = 34892.4
CASSEGRAIN_VERTEX_Z_MM = 26993.999446877
CASSEGRAIN_VERTEX_RADIUS_MM = 26112.078893738
CASSEGRAIN_CONIC = -5.317616535
CASSEGRAIN_APERTURE_RADIUS_MM = 14000.0

#: Site tests/fixtures/energy/manifest.json's "site" was traced at.
DEFAULT_SITE = (-10.0, -52.0, -3)

#: The pinned rectangle heliostat.trace.mc/cone use with no ``design``.
PINNED_MIRROR_WIDTH_MM = 2.0 * _mc.MIRROR_HALF_X_MM
PINNED_MIRROR_HEIGHT_MM = 2.0 * _mc.MIRROR_HALF_Y_MM


@dataclass(frozen=True)
class OpticsSpec:
    """One standard optical configuration: secondary, receiver, throughput
    and the aim-point solve to use for it."""

    name: str
    secondary: Secondary
    receiver: Receiver
    throughput: float
    aim: Callable[[float, float, float, float], Solution]
    window_mm: float = WINDOW_MM


def _aim_prime_focus(x_mm, y_mm, solar_az_deg, solar_el_deg) -> Solution:
    return solve_prime_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, PRIME_FOCUS_HEIGHT_MM)


def _aim_axicon(x_mm, y_mm, solar_az_deg, solar_el_deg) -> Solution:
    return solve_axicon(
        x_mm,
        y_mm,
        solar_az_deg,
        solar_el_deg,
        SECONDARY_HEIGHT_MM,
        AXICON_ANGLE_DEG,
        RECEIVER_HEIGHT_MM,
    )


def _aim_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg) -> Solution:
    return solve_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg, CASSEGRAIN_FOCUS_HEIGHT_MM)


def standard_optics(name: str) -> OpticsSpec:
    """The three standard optical configurations this package's golden
    fixtures (``tests/fixtures/mc_parity``, ``tests/test_aiming.py``) were
    traced/solved with. See the module docstring for provenance of every
    constant.
    """
    if name == "prime_focus":
        secondary = NoSecondary()
        receiver = FlatWindowReceiver(
            z_mm=PRIME_FOCUS_HEIGHT_MM, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="down"
        )
        return OpticsSpec("prime_focus", secondary, receiver, 0.9, _aim_prime_focus)

    if name == "axicon":
        secondary = AxiconSecondary(
            apex_height_mm=SECONDARY_HEIGHT_MM,
            half_angle_deg=AXICON_ANGLE_DEG,
            aperture_radius_mm=AXICON_APERTURE_RADIUS_MM,
        )
        receiver = FlatWindowReceiver(
            z_mm=RECEIVER_HEIGHT_MM, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
        return OpticsSpec("axicon", secondary, receiver, 0.81, _aim_axicon)

    if name == "cassegrain":
        secondary = CassegrainSecondary(
            vertex_z_mm=CASSEGRAIN_VERTEX_Z_MM,
            vertex_radius_mm=CASSEGRAIN_VERTEX_RADIUS_MM,
            conic=CASSEGRAIN_CONIC,
            aperture_radius_mm=CASSEGRAIN_APERTURE_RADIUS_MM,
        )
        receiver = FlatWindowReceiver(
            z_mm=RECEIVER_HEIGHT_MM, half_u_mm=WINDOW_MM, half_v_mm=WINDOW_MM, facing="up"
        )
        return OpticsSpec("cassegrain", secondary, receiver, 0.81, _aim_cassegrain)

    raise ValueError(f"unknown optics {name!r}; use 'prime_focus', 'axicon' or 'cassegrain'")


# ---------------------------------------------------------------------------
# cfg: a small duck-typed namespace tree, the same shape every other module
# in this package reads (see tests/test_store.py::make_cfg,
# tests/test_mc_parity.py::_make_cfg).
# ---------------------------------------------------------------------------


def _watts_per_ray(n: int) -> float:
    return _mc.SOURCE_POWER_W / n if n else 0.0


def _build_cfg(
    opt,
    trace_mode,
    n_rays,
    mirror_w,
    mirror_h,
    *,
    site,
    hour_step,
    sunrise_margin_min,
    dates,
    min_elevation_deg=5.0,
):
    bin_size_mm = 2.0 * WINDOW_MM / GRID_SIZE
    receiver_ns = SimpleNamespace(
        window_mm=WINDOW_MM,
        grid_size=GRID_SIZE,
        bin_size_mm=bin_size_mm,
        bin_area_m2=(bin_size_mm / 1000.0) ** 2,
        edges=np.linspace(-WINDOW_MM, WINDOW_MM, GRID_SIZE + 1),
    )
    if trace_mode.backend == "mc":
        rays_per_heliostat = n_rays or trace_mode.n_rays
    else:
        cone_grid = trace_mode.cone_kwargs.get("grid", (20, 12))
        if cone_grid is None:
            # Density-derived grid (ultra_fast) -- resolve against this
            # field's own mirror dimensions, same derivation the tracer
            # itself uses (heliostat.trace.cone.grid_for_density).
            density = trace_mode.cone_kwargs["density"]
            gx, gy = grid_for_density(density, mirror_w / 1000.0, mirror_h / 1000.0)
        else:
            gx, gy = cone_grid
        rays_per_heliostat = gx * gy  # informational only -- not a ray budget

    return SimpleNamespace(
        receiver=receiver_ns,
        source=SimpleNamespace(power_w=_mc.SOURCE_POWER_W, watts_per_ray=_watts_per_ray),
        optics=SimpleNamespace(throughput=opt.throughput),
        trace=SimpleNamespace(rays_per_heliostat=rays_per_heliostat),
        storage=SimpleNamespace(raw_rays="none"),
        site=SimpleNamespace(latitude=site[0], longitude=site[1], timezone=site[2]),
        sweep=SimpleNamespace(
            hour_step=hour_step,
            sunrise_margin_min=sunrise_margin_min,
            min_elevation_deg=min_elevation_deg,
            dates=tuple(dates),
        ),
        field=SimpleNamespace(mirror_area_m2=(mirror_w / 1000.0) * (mirror_h / 1000.0)),
    )


# ---------------------------------------------------------------------------
# Worker process state. Set once per worker by _init_worker (spawn-safe: the
# heavy shared objects -- secondary, receiver, design, kernel -- are pickled
# once per worker via Pool's initargs, not once per task).
# ---------------------------------------------------------------------------
_STATE: dict = {}


def _init_worker(
    trace_mode,
    secondary,
    receiver,
    design,
    kernel,
    base_seed,
    receiver_edges,
    cone_flux_grid,
    sampler=None,
):
    _STATE["mode"] = trace_mode
    _STATE["secondary"] = secondary
    _STATE["receiver"] = receiver
    _STATE["design"] = design
    _STATE["kernel"] = kernel
    _STATE["base_seed"] = base_seed
    _STATE["edges"] = receiver_edges
    _STATE["cone_flux_grid"] = cone_flux_grid
    # docs/ui-spec-v0.2.md §O: the MC counterpart of `kernel` above --
    # `None` (default) is trace_heliostat's own "use the pinned CSR=0
    # BuieSampler" default, so a caller that never heard of CSR sees no
    # change at all (see run_sweep's own circumsolar_ratio docstring).
    _STATE["sampler"] = sampler


def _trace_task(task: tuple) -> dict:
    """One heliostat's trace. Module-level (not a closure) so it pickles for
    ``multiprocessing.Pool.map`` on every start method, including Windows'
    ``spawn``."""
    heliostat_id, x, y, rot_az, rot_el, c3, c4, c5, solar_az, solar_el, step_int, n_rays = task
    mode: TraceMode = _STATE["mode"]

    if mode.backend == "mc":
        seed = np.random.SeedSequence((_STATE["base_seed"], step_int, heliostat_id))
        rng = np.random.default_rng(seed)
        out = trace_heliostat(
            x,
            y,
            rot_az,
            rot_el,
            c3,
            c4,
            c5,
            solar_az,
            solar_el,
            _STATE["secondary"],
            _STATE["receiver"],
            n_rays,
            rng,
            design=_STATE["design"],
            sampler=_STATE.get("sampler"),
        )
        xy = out["xy"].T  # (K, 2)
        edges = _STATE["edges"]
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[edges, edges])
        return {
            "heliostat_id": heliostat_id,
            "xy": xy,
            "counts": counts.astype(np.uint32),
            "counters": out["counters"],
            "rays_emitted": n_rays,
        }

    out = trace_heliostat_cone(
        x,
        y,
        rot_az,
        rot_el,
        c3,
        c4,
        c5,
        solar_az,
        solar_el,
        _STATE["secondary"],
        _STATE["receiver"],
        _STATE["kernel"],
        flux_grid=_STATE["cone_flux_grid"],
        design=_STATE["design"],
        **mode.cone_kwargs,
    )
    return {
        "heliostat_id": heliostat_id,
        "flux": out["flux"].astype(np.float32),
        "power_w": out["power_w"],
        "u_edges": out["u_edges"],
        "v_edges": out["v_edges"],
    }


# ---------------------------------------------------------------------------
# Per-heliostat summary rows
# ---------------------------------------------------------------------------


def _mc_row(
    field, i, hid, step, sol, r, eta_shade, eta_block, eta_secondary, eta_union, cfg
) -> dict:
    m = spot_metrics(r["xy"], r["rays_emitted"], cfg, dni_w_m2=1000.0, efficiency=float(eta_union))
    reached = r["counters"].get("reached_receiver", m["rays_landed"])
    in_window = r["counters"].get("in_window", m["rays_landed"])
    return {
        "date": step.date.isoformat(),
        "hour": step.hour,
        "timestep": step.key,
        "heliostat_id": hid,
        "x_m": float(field.x_m[i]),
        "y_m": float(field.y_m[i]),
        "radius_m": float(field.radius_mm[i] / 1000.0),
        "rays_emitted": m["rays_emitted"],
        "rays_landed": m["rays_landed"],
        "transmission": m["transmission"],
        "power_w": m["power_w"],
        "shading_blocking_efficiency": m["shading_blocking_efficiency"],
        "centroid_x_mm": m["centroid_x_mm"],
        "centroid_y_mm": m["centroid_y_mm"],
        "rms_radius_mm": m["rms_radius_mm"],
        "r50_mm": m["r50_mm"],
        "r90_mm": m["r90_mm"],
        "peak_flux_w_m2": m["peak_flux_w_m2"],
        "spillage": m["spillage"],
        "solar_az_deg": step.solar_az_deg,
        "solar_el_deg": step.solar_el_deg,
        "rot_az_deg": sol.rot_az_deg,
        "rot_el_deg": sol.rot_el_deg,
        "aoi_deg": sol.aoi_deg,
        "cosine_efficiency": sol.cosine_efficiency,
        "eta_shade": float(eta_shade),
        "eta_secondary": float(eta_secondary),
        "eta_block": float(eta_block),
        "rays_outside_window": int(reached - in_window),
        "eta_occlusion": float(eta_union),
    }


def _cone_map_metrics(
    flux, u_edges, v_edges, power_w_raw, cfg, efficiency, dni_w_m2=1000.0
) -> dict:
    """Power/peak/centroid/rms/r50/r90 from a cone-mode flux map's moments.

    ``flux`` is the raw (unoccluded) W/m^2 map at DNI=1000, throughput=1
    (:func:`heliostat.trace.cone.trace_heliostat_cone`'s own normalisation);
    ``power_w_raw`` is that same trace's map integral, in the same raw units
    -- passed through rather than recomputed from ``flux`` so this agrees
    exactly with what the trace itself measured (bin-centre quadrature of
    ``flux`` over ``u_edges``/``v_edges`` would only approximate it).
    Centroid/rms/r50/r90 are shape statistics and do not depend on
    ``efficiency`` -- the store's convention is a uniform scalar on power,
    not a geometric clip of the map (see module docstring).
    """
    scale = flux_scale(cfg, 0, dni_w_m2, flux_kind="analytic")
    total = float(flux.sum())
    if total <= 0:
        return {
            "power_w": 0.0,
            "peak_flux_w_m2": 0.0,
            "centroid_x_mm": float("nan"),
            "centroid_y_mm": float("nan"),
            "rms_radius_mm": float("nan"),
            "r50_mm": float("nan"),
            "r90_mm": float("nan"),
        }
    u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
    U, V = np.meshgrid(u_mid, v_mid)
    cx = float((U * flux).sum() / total)
    cy = float((V * flux).sum() / total)
    r = np.hypot(U - cx, V - cy)
    order = np.argsort(r.ravel())
    cum = np.cumsum(flux.ravel()[order]) / total
    r_sorted = r.ravel()[order]
    return {
        "power_w": float(power_w_raw) * scale * efficiency,
        "peak_flux_w_m2": float(flux.max()) * scale * efficiency,
        "centroid_x_mm": cx,
        "centroid_y_mm": cy,
        "rms_radius_mm": float(np.sqrt((flux * r**2).sum() / total)),
        "r50_mm": float(r_sorted[np.searchsorted(cum, 0.5)]),
        "r90_mm": float(r_sorted[np.searchsorted(cum, 0.9)]),
    }


def _cone_row(
    field, i, hid, step, sol, r, eta_shade, eta_block, eta_secondary, eta_union, cfg
) -> dict:
    m = _cone_map_metrics(
        r["flux"], r["u_edges"], r["v_edges"], r["power_w"], cfg, efficiency=float(eta_union)
    )
    return {
        "date": step.date.isoformat(),
        "hour": step.hour,
        "timestep": step.key,
        "heliostat_id": hid,
        "x_m": float(field.x_m[i]),
        "y_m": float(field.y_m[i]),
        "radius_m": float(field.radius_mm[i] / 1000.0),
        # Map-derived mode: no ray budget exists, so these ray-only columns
        # are NaN rather than repurposed (see module docstring).
        "rays_emitted": float("nan"),
        "rays_landed": float("nan"),
        "transmission": float("nan"),
        "power_w": m["power_w"],
        "shading_blocking_efficiency": float(eta_union),
        "centroid_x_mm": m["centroid_x_mm"],
        "centroid_y_mm": m["centroid_y_mm"],
        "rms_radius_mm": m["rms_radius_mm"],
        "r50_mm": m["r50_mm"],
        "r90_mm": m["r90_mm"],
        "peak_flux_w_m2": m["peak_flux_w_m2"],
        "spillage": float("nan"),
        "solar_az_deg": step.solar_az_deg,
        "solar_el_deg": step.solar_el_deg,
        "rot_az_deg": sol.rot_az_deg,
        "rot_el_deg": sol.rot_el_deg,
        "aoi_deg": sol.aoi_deg,
        "cosine_efficiency": sol.cosine_efficiency,
        "eta_shade": float(eta_shade),
        "eta_secondary": float(eta_secondary),
        "eta_block": float(eta_block),
        "rays_outside_window": float("nan"),
        "eta_occlusion": float(eta_union),
    }


# ---------------------------------------------------------------------------
# One timestep
# ---------------------------------------------------------------------------


def _trace_one_timestep(store, field, ids, step, opt, cfg, design, trace_mode, n_rays, pool):
    n = len(field)
    solutions = [
        opt.aim(float(field.x_mm[i]), float(field.y_mm[i]), step.solar_az_deg, step.solar_el_deg)
        for i in range(n)
    ]
    rot_az = np.array([s.rot_az_deg for s in solutions])
    rot_el = np.array([s.rot_el_deg for s in solutions])
    aims = _aim_points_mm(solutions)

    if design is None:
        geometries, aims = build_geometries(
            field,
            rot_az,
            rot_el,
            aims,
            mirror_width_mm=field.mirror_width_mm,
            mirror_height_mm=field.mirror_height_mm,
        )
    else:
        geometries = [
            MirrorGeometry.from_design(
                float(field.x_mm[i]), float(field.y_mm[i]), rot_az[i], rot_el[i], design
            )
            for i in range(n)
        ]

    # Neighbour list sized for THIS step, not once per run from the day's
    # lowest sun. The shading reach shrinks as the sun climbs and most of a
    # day is high sun, so a run-wide radius makes every noon step pay the
    # horizon's price: on a 643-heliostat field that is 189 candidate
    # occluders each where 12 suffice, and 24 s of occlusion where 2 s does.
    # The radius must still cover the blocking reach, which does *not*
    # shrink with the sun -- see search_radius_for, which measures what
    # omitting it costs.
    centres = np.array([g.centre for g in geometries])
    neighbours = neighbour_pairs(
        field,
        search_radius_for(
            step.solar_el_deg,
            float(field.mirror_height_mm),
            float(field.mirror_width_mm),
            beam_elevation_deg=min_beam_elevation_deg(centres, aims),
        ),
    )

    eta_shade, eta_block, eta_secondary, eta_union = polygon_occlusion(
        geometries, aims, step.solar_az_deg, step.solar_el_deg, neighbours
    )

    step_int = int(step.key.replace("_", ""))
    tasks = [
        (
            ids[i],
            float(field.x_mm[i]),
            float(field.y_mm[i]),
            float(rot_az[i]),
            float(rot_el[i]),
            float(solutions[i].c3),
            float(solutions[i].c4),
            float(solutions[i].c5),
            float(step.solar_az_deg),
            float(step.solar_el_deg),
            step_int,
            n_rays,
        )
        for i in range(n)
    ]
    results = pool.map(_trace_task, tasks) if pool is not None else [_trace_task(t) for t in tasks]

    rows = []
    if trace_mode.backend == "mc":
        counts = np.stack([r["counts"] for r in results]).astype(np.uint32)
        row_fn = _mc_row
    else:
        bin_area_m2 = cfg.receiver.bin_area_m2
        counts = np.stack([(r["flux"] * bin_area_m2).astype(np.float32) for r in results])
        row_fn = _cone_row

    for i, r in enumerate(results):
        rows.append(
            row_fn(
                field,
                i,
                ids[i],
                step,
                solutions[i],
                r,
                eta_shade[i],
                eta_block[i],
                eta_secondary[i],
                eta_union[i],
                cfg,
            )
        )

    result = TimestepResult(
        key=step.key,
        date=step.date.isoformat(),
        hour=step.hour,
        solar_az_deg=step.solar_az_deg,
        solar_el_deg=step.solar_el_deg,
        heliostat_ids=np.array(ids),
        rays_emitted=n_rays if trace_mode.backend == "mc" else 0,
        counts=counts,
        rays=None,
        index=None,
        rows=pd.DataFrame(rows),
    )
    store.write_timestep(result)


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def run_sweep(
    field: HeliostatField,
    dates,
    *,
    mode: str = "ultra_fast",
    optics: str = "prime_focus",
    site=DEFAULT_SITE,
    out_dir,
    design=None,
    workers: int | None = None,
    n_rays: int | None = None,
    base_seed: int = 20260811,
    hour_step: float = 1.0,
    sunrise_margin_min: float = 10.0,
    min_elevation_deg: float = 5.0,
    circumsolar_ratio: float = 0.0,
    progress: Callable[[str], None] = print,
) -> RunStore:
    """Trace ``field`` across ``dates`` and write a stored run to ``out_dir``.

    ``min_elevation_deg`` excludes timesteps below that sun elevation (see
    ``solar.build_time_grid``'s docstring for how -- the window is shrunk to
    the elevation crossing, not just filtered after the fact, so the
    collected-power integral is not biased). Pass ``float("-inf")`` for the
    pre-floor behaviour (every timestep from sunrise+margin to sunset-margin).

    ``circumsolar_ratio`` (docs/ui-spec-v0.2.md §O) is the Buie sunshape's
    circumsolar ratio, applied identically at every timestep and every
    heliostat -- the SAME value drives both the cone kernel
    (``sunshape_kernel("buie", circumsolar_ratio=...)``) and the Monte Carlo
    sampler (``BuieSampler(circumsolar_ratio=...)``, built once here and
    handed to every worker), so a field/day sweep run at CSR > 0 broadens
    identically to a single trace at the same CSR and mode. ``0`` (default)
    reproduces the pre-§O behaviour bit-identically -- unset ``kernel``/
    ``sampler`` are ``trace_heliostat_cone``/``trace_heliostat``'s own
    CSR=0 defaults, exactly as before this parameter existed.

    See the module docstring for the standard-optics constants, the cfg
    duck-type built internally, and every judgment call made along the way.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {sorted(MODES)}")
    trace_mode = MODES[mode]
    opt = standard_optics(optics)

    mirror_w = float(field.mirror_width_mm)
    mirror_h = float(field.mirror_height_mm)
    if not mirror_w or not mirror_h:
        raise ValueError(
            "field carries no mirror_width_mm/mirror_height_mm; load it via "
            "heliostat.field.load_field(..., mirror_width_mm=..., mirror_height_mm=...)"
        )
    if design is None and (mirror_w, mirror_h) != (PINNED_MIRROR_WIDTH_MM, PINNED_MIRROR_HEIGHT_MM):
        warnings.warn(
            f"field mirror dims ({mirror_w:g} x {mirror_h:g} mm) differ from the trace "
            f"backends' pinned rectangle ({PINNED_MIRROR_WIDTH_MM:g} x {PINNED_MIRROR_HEIGHT_MM:g} "
            "mm, heliostat.trace.mc.MIRROR_HALF_*_MM). Without a `design=`, the optical trace "
            "still uses the pinned rectangle; only occlusion/shading geometry uses the field's "
            "own dimensions. Pass a matching `design=` to change the traced mirror size.",
            stacklevel=2,
        )

    dates = tuple(dates)
    n = len(field)
    ids = [int(i) for i in field.ids]

    cfg = _build_cfg(
        opt,
        trace_mode,
        n_rays,
        mirror_w,
        mirror_h,
        site=site,
        hour_step=hour_step,
        sunrise_margin_min=sunrise_margin_min,
        min_elevation_deg=min_elevation_deg,
        dates=dates,
    )
    steps = build_time_grid(cfg, dates)
    if not steps:
        raise ValueError(
            "no daylight timesteps for the given dates/site/hour_step/margin/min_elevation_deg"
        )

    out_dir = Path(out_dir)
    store = RunStore(out_dir, cfg=cfg, mode="w")
    flux_kind = "ray_counts" if trace_mode.backend == "mc" else "analytic"
    workers_eff = workers or mp.cpu_count()
    store.write_manifest(
        cfg,
        receiver=opt.receiver,
        design=design,
        flux_kind=flux_kind,
        extra={
            "mode": mode,
            "optics": optics,
            "site": {"latitude": site[0], "longitude": site[1], "timezone": site[2]},
            "dates": [d.isoformat() for d in dates],
            "timesteps": [s.key for s in steps],
            "n_heliostats": n,
            "heliostat_ids": ids,
            "workers": workers_eff,
            "base_seed": base_seed,
            "seed_scheme": (
                "default_rng(SeedSequence((base_seed, "
                "int(step.key.replace('_', '')), heliostat_id)))"
            ),
            "occlusion_form": "union",
            "traced_secondary": False,
            "circumsolar_ratio": circumsolar_ratio,
        },
    )

    kernel = (
        sunshape_kernel("buie", circumsolar_ratio=circumsolar_ratio)
        if trace_mode.backend == "cone"
        else None
    )
    # docs/ui-spec-v0.2.md §O: the MC counterpart of `kernel` above, built
    # once (not per heliostat/timestep) like `kernel` itself -- `None` at
    # CSR=0 so trace_heliostat falls back to its own pinned default sampler,
    # bit-identical to before this parameter existed.
    sampler = BuieSampler(circumsolar_ratio=circumsolar_ratio) if circumsolar_ratio > 0 else None
    cone_flux_grid = (GRID_SIZE, GRID_SIZE)

    pool = None
    if workers_eff > 1:
        pool = mp.Pool(
            workers_eff,
            initializer=_init_worker,
            initargs=(
                trace_mode,
                opt.secondary,
                opt.receiver,
                design,
                kernel,
                base_seed,
                cfg.receiver.edges,
                cone_flux_grid,
                sampler,
            ),
        )
    else:
        _init_worker(
            trace_mode,
            opt.secondary,
            opt.receiver,
            design,
            kernel,
            base_seed,
            cfg.receiver.edges,
            cone_flux_grid,
            sampler,
        )

    n_rays_eff = n_rays or trace_mode.n_rays

    t_start = time.perf_counter()
    try:
        for si, step in enumerate(steps):
            _trace_one_timestep(
                store, field, ids, step, opt, cfg, design, trace_mode, n_rays_eff, pool
            )
            elapsed = time.perf_counter() - t_start
            avg = elapsed / (si + 1)
            eta = avg * (len(steps) - si - 1)
            progress(
                f"[{si + 1:>3}/{len(steps)}] {step.label}  {n} heliostats  "
                f"elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return store
