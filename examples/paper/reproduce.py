#!/usr/bin/env python
"""Reproduce the nine configurations of the companion SPIE paper (14212-5).

The paper compares three optical layouts -- ``prime_focus``, ``cassegrain``
and ``axicon`` -- each driven with three mirror *figures*:

``twisting``
    The heliostat re-solves its astigmatic figure at every timestep, so the
    mirror physically twists as the sun moves. This is what
    :func:`heliostat.geometry.aiming.solve_prime_focus` and friends return,
    and it is the upper bound on what a figured heliostat can do.
``spherical``
    One frozen figure per heliostat, chosen once and never changed. The
    coefficients ship in ``data/fixed_shapes_*.csv``; see the README for how
    each file was built.
``flat``
    No figure at all -- ``c3 = c4 = c5 = 0``.

Everything optical is imported from :mod:`heliostat`; this script only
assembles the paper's own numbers (site, field, dates, geometry, sunshape,
ray budget) and drives the library. No physics is reimplemented here. Where
this script does not simply call :mod:`heliostat.sweep.run_sweep`, it is
because the paper needs four things that driver does not expose -- a 256-bin
flux grid, the Buie sunshape, a per-heliostat *frozen* figure, and a
secondary shading body -- and every one of those is a keyword the underlying
library functions already take.

Run ``python reproduce.py --help`` for the command line; the README has
worked examples and runtimes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # run from a clone without installing
    sys.path.insert(0, str(_SRC))

from heliostat import dni as dni_mod  # noqa: E402
from heliostat import energy as energy_mod  # noqa: E402
from heliostat.field import HeliostatField, load_field, neighbour_pairs  # noqa: E402
from heliostat.geometry.aiming import (  # noqa: E402
    aim_points_mm,
    solve_axicon,
    solve_cassegrain,
    solve_prime_focus,
)
from heliostat.geometry.receiver import FlatWindowReceiver  # noqa: E402
from heliostat.geometry.secondary import (  # noqa: E402
    AxiconSecondary,
    CassegrainSecondary,
    NoSecondary,
)
from heliostat.geometry.shading import (  # noqa: E402
    SecondaryCone,
    SecondaryDisc,
    build_geometries,
    min_beam_elevation_deg,
    polygon_occlusion,
    search_radius_for,
)
from heliostat.metrics import bin_radius, radial_mask  # noqa: E402
from heliostat.solar import build_time_grid  # noqa: E402
from heliostat.store import RunStore, TimestepResult, flux_scale  # noqa: E402
from heliostat.trace import mc as mc_mod  # noqa: E402
from heliostat.trace.cone import sunshape_kernel, trace_heliostat_cone  # noqa: E402
from heliostat.trace.mc import trace_heliostat  # noqa: E402
from heliostat.trace.samplers import make_sampler  # noqa: E402

# ---------------------------------------------------------------------------
# The paper's own numbers. Every constant here is quoted in the README with
# where it came from; nothing in this block is a default of the library.
# ---------------------------------------------------------------------------

#: Latitude, longitude, timezone offset. NOT the DNI station's latitude --
#: the paper's site is a round -10.0 deg (see README, "Two DNI bases").
SITE = (-10.0, -52.0, -3)

#: Seven dates spanning December solstice to June solstice: one ascending
#: branch of the declination cycle, so no sun direction is traced twice.
DATES: tuple[_dt.date, ...] = (
    _dt.date(2026, 12, 21),
    _dt.date(2026, 1, 21),
    _dt.date(2026, 2, 20),
    _dt.date(2026, 3, 20),
    _dt.date(2026, 4, 21),
    _dt.date(2026, 5, 21),
    _dt.date(2026, 6, 21),
)

HOUR_STEP = 1.0  # maximum spacing between timesteps, hours
SUNRISE_MARGIN_MIN = 10.0  # skip the first/last 10 minutes of daylight
RAYS_PER_HELIOSTAT = 120_000
BASE_SEED = 20260811

MIRROR_WIDTH_MM = 5000.0
MIRROR_HEIGHT_MM = 3000.0

WINDOW_MM = 2000.0  # receiver window half-extent
GRID_SIZE = 256  # flux bins per axis (the paper overrides the library's 128)
APERTURE_RADIUS_MM = 720.0  # the paper's reported receiver aperture

#: The instant the paper's spot table is quoted at (a real timestep key of
#: the grid above -- see ``--list-timesteps``).
INSTANT_KEY = "20261221_0906"

#: Radius of the opaque body each secondary presents to the sun, for the
#: SHADING test only -- it never touches the traced optical surfaces, whose
#: apertures are the ``*_APERTURE_RADIUS_MM`` constants below.
#:
#: The Cassegrain's 15 m is the paper's run-manifest value and disagrees with
#: the surrounding prose, which describes a rim clearing 14 m (see the
#: README, "A recorded contradiction"). Measured at the paper's own instant,
#: 15 m reproduces the published Cassegrain window power to -0.02% and 14 m
#: to +0.36%, so the manifest is what the runs were actually traced with.
#: The axicon's cone shades with its own 14 m aperture on the same test:
#: 14 m gives -0.02%, 15 m gives -0.46%. Prime focus has nothing above the
#: field at all.
SHADE_BODY_RADIUS_MM = {
    "prime_focus": 0.0,
    "cassegrain": 15000.0,
    "axicon": 14000.0,
}

PRIME_FOCUS_HEIGHT_MM = 35335.0
CASSEGRAIN_F1_MM = 34892.4
CASSEGRAIN_VERTEX_Z_MM = 26993.999446877
CASSEGRAIN_VERTEX_RADIUS_MM = 26112.078893738
CASSEGRAIN_CONIC = -5.317616535
CASSEGRAIN_APERTURE_RADIUS_MM = 14000.0
AXICON_APEX_MM = 27000.0
AXICON_HALF_ANGLE_DEG = 20.0
AXICON_APERTURE_RADIUS_MM = 14000.0
GROUND_RECEIVER_Z_MM = 7000.0

LAYOUTS = ("prime_focus", "cassegrain", "axicon")
FIGURES = ("twisting", "spherical", "flat")

#: Heliostats the paper traced: 645 positions less the two coincident
#: duplicates (144=192, 241=289) the field loader drops.
PAPER_N_HELIOSTATS = 643

#: Frozen-figure table per layout. ``None`` for a layout+figure pair that
#: needs no table.
FIXED_SHAPE_FILES = {
    "prime_focus": "fixed_shapes_pf35335_spherical.csv",
    "cassegrain": "fixed_shapes_cass34892_spherical.csv",
    "axicon": "fixed_shapes_axicon_medial.csv",
}

FIELD_FILE = "field_645.csv"
DNI_FILE = "dni_nasa_hourly.csv.gz"

#: Longitude the NASA POWER record was pulled at (Petrolina), against the
#: site longitude above. The gap is read as a solar-time shift, not ignored.
DNI_DATA_LONGITUDE = -40.5
DNI_WINDOW_DAYS = 5

DATA_DIR = _HERE / "data"


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperOptics:
    """One layout's secondary, receiver, throughput, aim solve and shading body."""

    name: str
    secondary: object
    receiver: FlatWindowReceiver
    throughput: float
    aim: Callable
    shade_body: object | None


def paper_optics(layout: str, shade_radius_mm: float | None = None) -> PaperOptics:
    """Build the paper's optics for ``layout``.

    ``shade_radius_mm`` sizes the opaque body used by the *shading* test
    only; it never touches the traced surfaces. ``None`` takes the layout's
    entry in :data:`SHADE_BODY_RADIUS_MM`; ``0`` removes the body.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; use one of {LAYOUTS}")
    if shade_radius_mm is None:
        shade_radius_mm = SHADE_BODY_RADIUS_MM[layout]

    if layout == "prime_focus":
        return PaperOptics(
            name=layout,
            secondary=NoSecondary(),
            receiver=FlatWindowReceiver(
                z_mm=PRIME_FOCUS_HEIGHT_MM,
                half_u_mm=WINDOW_MM,
                half_v_mm=WINDOW_MM,
                facing="down",
            ),
            throughput=0.90,
            aim=lambda x, y, az, el: solve_prime_focus(x, y, az, el, PRIME_FOCUS_HEIGHT_MM),
            # Nothing is above the field: the receiver IS the target.
            shade_body=None,
        )

    if layout == "cassegrain":
        secondary = CassegrainSecondary(
            vertex_z_mm=CASSEGRAIN_VERTEX_Z_MM,
            vertex_radius_mm=CASSEGRAIN_VERTEX_RADIUS_MM,
            conic=CASSEGRAIN_CONIC,
            aperture_radius_mm=CASSEGRAIN_APERTURE_RADIUS_MM,
        )
        body = (
            SecondaryDisc(z_mm=secondary.rim_z_mm, radius_mm=shade_radius_mm)
            if shade_radius_mm > 0
            else None
        )
        return PaperOptics(
            name=layout,
            secondary=secondary,
            receiver=FlatWindowReceiver(
                z_mm=GROUND_RECEIVER_Z_MM,
                half_u_mm=WINDOW_MM,
                half_v_mm=WINDOW_MM,
                facing="up",
            ),
            throughput=0.81,
            aim=lambda x, y, az, el: solve_cassegrain(x, y, az, el, CASSEGRAIN_F1_MM),
            shade_body=body,
        )

    if layout == "axicon":
        body = (
            SecondaryCone(
                z_tip_mm=AXICON_APEX_MM,
                angle_deg=AXICON_HALF_ANGLE_DEG,
                aperture_radius_mm=shade_radius_mm,
            )
            if shade_radius_mm > 0
            else None
        )
        return PaperOptics(
            name=layout,
            secondary=AxiconSecondary(
                apex_height_mm=AXICON_APEX_MM,
                half_angle_deg=AXICON_HALF_ANGLE_DEG,
                aperture_radius_mm=AXICON_APERTURE_RADIUS_MM,
            ),
            receiver=FlatWindowReceiver(
                z_mm=GROUND_RECEIVER_Z_MM,
                half_u_mm=WINDOW_MM,
                half_v_mm=WINDOW_MM,
                facing="up",
            ),
            throughput=0.81,
            aim=lambda x, y, az, el: solve_axicon(
                x, y, az, el, AXICON_APEX_MM, AXICON_HALF_ANGLE_DEG, GROUND_RECEIVER_Z_MM
            ),
            shade_body=body,
        )

    raise AssertionError(f"unhandled layout {layout!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Field, figures, cfg
# ---------------------------------------------------------------------------


def load_paper_field(n_heliostats: int | None = None, path: Path | None = None) -> HeliostatField:
    """The paper's 645-position field, coincident duplicates dropped.

    :func:`heliostat.field.load_field` finds the two coincident pairs by
    distance and drops the higher id of each (144=192, 241=289), leaving 643
    traced heliostats whose ids keep the source file's numbering. That is
    exactly the rule the paper's own runs used.

    ``n_heliostats`` keeps only the innermost N by radius -- a smoke-test
    convenience, not a paper configuration.
    """
    path = Path(path) if path is not None else DATA_DIR / FIELD_FILE
    field = load_field(path, mirror_width_mm=MIRROR_WIDTH_MM, mirror_height_mm=MIRROR_HEIGHT_MM)
    if n_heliostats is not None and n_heliostats < len(field):
        order = np.argsort(field.radius_mm)[:n_heliostats]
        field = field.subset(field.ids[np.sort(order)])
    return field


#: How far a frozen-figure table's recorded position may sit from the field
#: position it describes, in mm. The shipped tables disagree with
#: ``field_645.csv`` by up to 42 mm on 11 of the 645 heliostats -- the field
#: file has those coordinates snapped to a round metre or decimetre and the
#: figure tables were built before that snap (see the README). The field's
#: own minimum spacing is 5,831 mm, so anything well under half of that
#: still identifies a heliostat uniquely; 500 mm is 12x the worst observed
#: residual and 6x below the ambiguity threshold.
SHAPE_MATCH_TOL_MM = 500.0


def load_fixed_shapes(
    path: Path, field: HeliostatField, tol_mm: float = SHAPE_MATCH_TOL_MM
) -> np.ndarray:
    """Per-heliostat frozen ``(c3, c4, c5)``, matched to ``field`` by position.

    The three shipped tables do not agree on how heliostats are numbered:
    the prime-focus and Cassegrain tables carry all 645 source rows with the
    source file's ids, while the axicon table carries the 643 survivors
    renumbered 0..642. Matching on id would silently mis-assign every figure
    past the first dropped duplicate in the axicon case -- so this matches on
    the ``x_mm``/``y_mm`` columns all three tables carry and never on the id.
    The match must be injective (no two heliostats may claim one table row),
    which is what turns a nearest-neighbour lookup into a proof that the
    table really describes this field.

    Returns an ``(N, 3)`` array in the field's own row order, in the same
    Zernike convention :class:`heliostat.geometry.aiming.Solution` reports --
    the trace backends' convention, which is what the tables were written in.
    """
    from scipy.spatial import cKDTree

    table = pd.read_csv(path, comment="#")
    needed = {"x_mm", "y_mm", "c3", "c4", "c5"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {sorted(missing)}")

    tree = cKDTree(table[["x_mm", "y_mm"]].to_numpy(float))
    dist, idx = tree.query(np.column_stack((field.x_mm, field.y_mm)))
    if float(dist.max()) > tol_mm:
        worst = int(np.argmax(dist))
        raise ValueError(
            f"{path.name}: heliostat {int(field.ids[worst])} at "
            f"({field.x_mm[worst]:.3f}, {field.y_mm[worst]:.3f}) mm has no figure "
            f"within {tol_mm:g} mm (nearest is {dist[worst]:.4f} mm away). The "
            f"figure table does not describe this field."
        )
    if len(set(idx.tolist())) != len(field):
        raise ValueError(
            f"{path.name}: {len(field)} heliostats matched only "
            f"{len(set(idx.tolist()))} distinct table rows -- the position match "
            f"is not one-to-one, so the table does not describe this field."
        )
    return table.loc[idx, ["c3", "c4", "c5"]].to_numpy(float)


def paper_cfg(
    optics: PaperOptics,
    *,
    n_rays: int,
    dates=DATES,
    grid_size: int = GRID_SIZE,
    site=SITE,
) -> SimpleNamespace:
    """The duck-typed cfg every :mod:`heliostat` read path expects.

    Same shape as :func:`heliostat.sweep._build_cfg` builds, with the paper's
    256-bin grid instead of the library default of 128.
    """
    bin_size_mm = 2.0 * WINDOW_MM / grid_size
    return SimpleNamespace(
        receiver=SimpleNamespace(
            window_mm=WINDOW_MM,
            grid_size=grid_size,
            bin_size_mm=bin_size_mm,
            bin_area_m2=(bin_size_mm / 1000.0) ** 2,
            edges=np.linspace(-WINDOW_MM, WINDOW_MM, grid_size + 1),
        ),
        source=SimpleNamespace(
            power_w=mc_mod.SOURCE_POWER_W,
            watts_per_ray=lambda n: mc_mod.SOURCE_POWER_W / n if n else 0.0,
        ),
        optics=SimpleNamespace(throughput=optics.throughput),
        trace=SimpleNamespace(rays_per_heliostat=n_rays),
        storage=SimpleNamespace(raw_rays="none"),
        site=SimpleNamespace(latitude=site[0], longitude=site[1], timezone=site[2]),
        sweep=SimpleNamespace(
            hour_step=HOUR_STEP,
            sunrise_margin_min=SUNRISE_MARGIN_MIN,
            dates=tuple(dates),
        ),
        field=SimpleNamespace(
            mirror_area_m2=(MIRROR_WIDTH_MM / 1000.0) * (MIRROR_HEIGHT_MM / 1000.0)
        ),
    )


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------


def config_name(layout: str, figure: str) -> str:
    return f"{layout}_{figure}"


def parse_configs(spec) -> list[tuple[str, str]]:
    """``["axicon:twisting", "prime_focus"]`` -> ``[(layout, figure), ...]``.

    A bare layout expands to all three of its figures, a bare figure to all
    three layouts, and ``all`` (or an empty spec) to all nine.
    """
    if not spec or list(spec) == ["all"]:
        return [(lay, fig) for lay in LAYOUTS for fig in FIGURES]
    out: list[tuple[str, str]] = []
    for item in spec:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                lay, fig = token.split(":", 1)
                lay, fig = lay.strip(), fig.strip()
                if lay not in LAYOUTS:
                    raise ValueError(f"unknown layout {lay!r}; use one of {LAYOUTS}")
                if fig not in FIGURES:
                    raise ValueError(f"unknown figure {fig!r}; use one of {FIGURES}")
                pairs = [(lay, fig)]
            elif token in LAYOUTS:
                pairs = [(token, fig) for fig in FIGURES]
            elif token in FIGURES:
                pairs = [(lay, token) for lay in LAYOUTS]
            else:
                raise ValueError(
                    f"cannot parse config {token!r}; use 'layout:figure', a bare "
                    f"layout {LAYOUTS}, a bare figure {FIGURES}, or 'all'"
                )
            for p in pairs:
                if p not in out:
                    out.append(p)
    return out


def _clear_run_dir(out_dir: Path) -> None:
    """Delete a previous run's artifacts from ``out_dir``, and only those.

    Named files and globs rather than :func:`shutil.rmtree`, so pointing
    ``--out`` at the wrong directory costs a confusing run and not somebody's
    data.
    """
    for path in (out_dir / "summary.csv", out_dir / "manifest.json"):
        path.unlink(missing_ok=True)
    for sub in ("flux", "raw"):
        for path in (out_dir / sub).glob("*.npy"):
            path.unlink()


def _figure_table(layout: str, figure: str, field: HeliostatField) -> np.ndarray | None:
    """``(N, 3)`` frozen coefficients, or ``None`` when the figure is solved."""
    if figure == "twisting":
        return None
    if figure == "flat":
        return np.zeros((len(field), 3))
    if figure == "spherical":
        return load_fixed_shapes(DATA_DIR / FIXED_SHAPE_FILES[layout], field)
    raise ValueError(f"unknown figure {figure!r}; use one of {FIGURES}")


def run_config(
    layout: str,
    figure: str,
    *,
    out_dir,
    dates=DATES,
    mode: str = "monte_carlo",
    n_rays: int = RAYS_PER_HELIOSTAT,
    n_heliostats: int | None = None,
    shade_radius_mm: float | None = None,
    base_seed: int = BASE_SEED,
    grid_size: int = GRID_SIZE,
    field: HeliostatField | None = None,
    only_keys: tuple[str, ...] | None = None,
    progress: Callable[[str], None] = print,
) -> RunStore:
    """Trace one (layout, figure) pair and write a stored run to ``out_dir``.

    What is stored
    --------------
    One map per timestep, shape ``(1, G, G)``: the whole field's counts
    already weighted by each heliostat's ``eta_union``. Per-heliostat maps
    would be 168 MB per timestep at 643 heliostats and a 256 grid -- 15 GB
    per configuration, 140 GB for the nine -- and nothing the paper reports
    needs them: flux maps add linearly and the occlusion weights are
    per-heliostat scalars, so the field map is the exact sum either way. The
    per-heliostat detail that *is* needed (power into the window and into the
    720 mm aperture, pointing, occlusion) is in ``summary.csv``, one row per
    heliostat per timestep. The manifest records this convention under
    ``counts_convention``.

    ``only_keys`` restricts the run to those timestep keys (used by the
    single-instant validation); ``None`` traces every step of ``dates``.

    What it costs
    -------------
    Profiled on 643 heliostats, one core of an i7-1185G7-class laptop: 43 s
    of Monte Carlo (67 ms per heliostat at 120,000 rays) plus 27 s of
    occlusion, so about 70 s per timestep and 1 h 50 min for the paper's 94.
    The occlusion half is set by ``only_keys``/``dates``, not by the field: a
    lone 09:06 instant sizes the neighbour search from a 40.6 deg sun (5.6
    occluders per heliostat, 1.4 s) while the full seven-date grid sizes it
    from a 1.8 deg sun (189 occluders, 27 s) -- see the README.

    An existing run at ``out_dir`` is cleared first. ``RunStore`` appends to
    ``summary.csv`` when it already exists -- that is what makes a sweep
    resumable -- so re-tracing into a directory that already holds a run
    would otherwise silently double every summary row, and every annual
    total with them.
    """
    opt = paper_optics(layout, shade_radius_mm)
    shade_radius_mm = (
        SHADE_BODY_RADIUS_MM[layout] if shade_radius_mm is None else float(shade_radius_mm)
    )
    field = load_paper_field(n_heliostats) if field is None else field
    table = _figure_table(layout, figure, field)

    cfg = paper_cfg(opt, n_rays=n_rays, dates=dates, grid_size=grid_size)
    steps = build_time_grid(cfg, dates)
    if only_keys is not None:
        wanted = set(only_keys)
        steps = [s for s in steps if s.key in wanted]
        if not steps:
            raise ValueError(f"none of {sorted(wanted)} is a timestep of these dates")
    if not steps:
        raise ValueError("no daylight timesteps for the given dates")

    if mode not in ("monte_carlo", "ultra_fast", "fast_accurate"):
        raise ValueError(f"unknown mode {mode!r}")
    backend = "mc" if mode == "monte_carlo" else "cone"
    cone_kwargs = {"order": 2 if mode == "fast_accurate" else 1, "grid": (20, 12), "mask_nodes": 16}

    sampler = make_sampler("buie") if backend == "mc" else None
    kernel = sunshape_kernel("buie") if backend == "cone" else None

    _clear_run_dir(Path(out_dir))
    store = RunStore(Path(out_dir), cfg=cfg, mode="w")
    store.write_manifest(
        cfg,
        receiver=opt.receiver,
        flux_kind="ray_counts" if backend == "mc" else "analytic",
        extra={
            "paper": "SPIE 14212-5",
            "layout": layout,
            "figure": figure,
            "mode": mode,
            "sunshape": "buie",
            "site": {"latitude": SITE[0], "longitude": SITE[1], "timezone": SITE[2]},
            "dates": [d.isoformat() for d in dates],
            "timesteps": [s.key for s in steps],
            "n_heliostats": len(field),
            "heliostat_ids": [int(i) for i in field.ids],
            "dropped_ids": [int(i) for i in field.dropped_ids],
            "mirror_width_mm": MIRROR_WIDTH_MM,
            "mirror_height_mm": MIRROR_HEIGHT_MM,
            "aperture_radius_mm": APERTURE_RADIUS_MM,
            "shade_body_radius_mm": shade_radius_mm,
            "traced_secondary": False,
            "occlusion_form": "union",
            "base_seed": base_seed,
            "seed_scheme": (
                "default_rng(SeedSequence((base_seed, "
                "int(step.key.replace('_', '')), heliostat_id)))"
            ),
            "counts_convention": (
                "one row per timestep: the whole field's counts already weighted "
                "by each heliostat's eta_union (see reproduce.run_config)"
            ),
        },
    )

    aperture = radial_mask(cfg, APERTURE_RADIUS_MM)
    edges = cfg.receiver.edges
    bin_area = cfg.receiver.bin_area_m2
    ids = [int(i) for i in field.ids]
    n = len(field)

    t0 = time.perf_counter()
    for si, step in enumerate(steps):
        solutions = [
            opt.aim(
                float(field.x_mm[i]), float(field.y_mm[i]), step.solar_az_deg, step.solar_el_deg
            )
            for i in range(n)
        ]
        rot_az = np.array([s.rot_az_deg for s in solutions])
        rot_el = np.array([s.rot_el_deg for s in solutions])
        geometries, aims = build_geometries(
            field,
            rot_az,
            rot_el,
            aim_points_mm(solutions),
            mirror_width_mm=MIRROR_WIDTH_MM,
            mirror_height_mm=MIRROR_HEIGHT_MM,
        )
        # Sized for this step rather than once per run from the day's
        # lowest sun: see heliostat.geometry.shading.search_radius_for,
        # including why the beam term is not optional.
        neighbours = neighbour_pairs(
            field,
            search_radius_for(
                step.solar_el_deg,
                MIRROR_HEIGHT_MM,
                MIRROR_WIDTH_MM,
                beam_elevation_deg=min_beam_elevation_deg(
                    np.array([g.centre for g in geometries]), aims
                ),
            ),
        )
        eta_shade, eta_block, eta_secondary, eta_union = polygon_occlusion(
            geometries,
            aims,
            step.solar_az_deg,
            step.solar_el_deg,
            neighbours,
            secondary=opt.shade_body,
        )

        step_int = int(step.key.replace("_", ""))
        scale = flux_scale(cfg, n_rays, 1000.0, store.flux_kind)
        field_counts = np.zeros((grid_size, grid_size))
        rows = []
        for i in range(n):
            c3, c4, c5 = (
                (solutions[i].c3, solutions[i].c4, solutions[i].c5)
                if table is None
                else (table[i, 0], table[i, 1], table[i, 2])
            )
            if backend == "mc":
                rng = np.random.default_rng(np.random.SeedSequence((base_seed, step_int, ids[i])))
                out = trace_heliostat(
                    float(field.x_mm[i]),
                    float(field.y_mm[i]),
                    float(rot_az[i]),
                    float(rot_el[i]),
                    float(c3),
                    float(c4),
                    float(c5),
                    step.solar_az_deg,
                    step.solar_el_deg,
                    opt.secondary,
                    opt.receiver,
                    n_rays,
                    rng,
                    sampler=sampler,
                )
                xy = out["xy"].T
                counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[edges, edges])
            else:
                out = trace_heliostat_cone(
                    float(field.x_mm[i]),
                    float(field.y_mm[i]),
                    float(rot_az[i]),
                    float(rot_el[i]),
                    float(c3),
                    float(c4),
                    float(c5),
                    step.solar_az_deg,
                    step.solar_el_deg,
                    opt.secondary,
                    opt.receiver,
                    kernel,
                    flux_grid=(grid_size, grid_size),
                    **cone_kwargs,
                )
                counts = out["flux"] * bin_area  # store's "analytic" convention

            eta = float(eta_union[i])
            field_counts += counts * eta
            total = float(counts.sum())
            inside = float(counts[aperture].sum())
            rows.append(
                {
                    "date": step.date.isoformat(),
                    "hour": step.hour,
                    "timestep": step.key,
                    "heliostat_id": ids[i],
                    "x_m": float(field.x_m[i]),
                    "y_m": float(field.y_m[i]),
                    "radius_m": float(field.radius_mm[i] / 1000.0),
                    "power_w": total * scale * eta,
                    "power_aperture_w": inside * scale * eta,
                    "solar_az_deg": step.solar_az_deg,
                    "solar_el_deg": step.solar_el_deg,
                    "rot_az_deg": float(rot_az[i]),
                    "rot_el_deg": float(rot_el[i]),
                    "aoi_deg": solutions[i].aoi_deg,
                    "cosine_efficiency": solutions[i].cosine_efficiency,
                    "c3": float(c3),
                    "c4": float(c4),
                    "c5": float(c5),
                    "eta_shade": float(eta_shade[i]),
                    "eta_block": float(eta_block[i]),
                    "eta_secondary": float(eta_secondary[i]),
                    "eta_occlusion": eta,
                }
            )

        store.write_timestep(
            TimestepResult(
                key=step.key,
                date=step.date.isoformat(),
                hour=step.hour,
                solar_az_deg=step.solar_az_deg,
                solar_el_deg=step.solar_el_deg,
                heliostat_ids=np.array([-1]),  # the field total, not a heliostat
                rays_emitted=n_rays if backend == "mc" else 0,
                counts=field_counts[None, :, :],
                rays=None,
                index=None,
                rows=pd.DataFrame(rows),
            )
        )
        elapsed = time.perf_counter() - t0
        progress(
            f"  [{si + 1:>3}/{len(steps)}] {step.label}  {n} heliostats  "
            f"elapsed {elapsed:7.1f}s  ETA {elapsed / (si + 1) * (len(steps) - si - 1):7.1f}s"
        )
    return store


# ---------------------------------------------------------------------------
# Reporting: the two tables the paper quotes
# ---------------------------------------------------------------------------


def instant_metrics(store: RunStore, cfg, key: str = INSTANT_KEY) -> dict:
    """The paper's spot table for one timestep, from the stored field map.

    ``r90_mm`` is measured about the receiver *axis*, not the spot centroid:
    an aperture is a fixed hole in a fixed place, and a spot whose centroid
    has walked off axis really does spill. Same convention as
    :func:`heliostat.metrics.bin_radius`.
    """
    flux = store.field_flux(key, cfg, dni_w_m2=1000.0)  # W/m^2, whole field
    power = flux * cfg.receiver.bin_area_m2  # W per bin
    window_w = float(power.sum())
    inside_w = float(power[radial_mask(cfg, APERTURE_RADIUS_MM)].sum())
    area_m2 = np.pi * (APERTURE_RADIUS_MM / 1000.0) ** 2

    rr = bin_radius(cfg).ravel()
    order = np.argsort(rr)
    cum = np.cumsum(power.ravel()[order])
    r90 = float(rr[order][np.searchsorted(cum, 0.9 * cum[-1])]) if cum[-1] > 0 else float("nan")

    return {
        "peak_kw_m2": float(flux.max()) / 1e3,
        "window_kw": window_w / 1e3,
        "power_720mm_kw": inside_w / 1e3,
        "frac_720mm": inside_w / window_w if window_w else float("nan"),
        "conc_720mm_suns": (inside_w / area_m2) / 1e3 if window_w else float("nan"),
        "r90_mm": r90,
    }


def petrolina_provider(path: Path | None = None, site=SITE):
    """The paper's climatology DNI: NASA POWER hourly, 2001-2024, at Petrolina.

    Two stages, both library code:
    :class:`heliostat.dni.DailyClimatologyDNI` averages the 24-year record on
    a (day-of-year, hour) grid with a circular +/-5-day window, and
    :class:`heliostat.dni.SolarTimeAligned` reads that table at the traced
    site's solar time rather than its own clock, since the record was taken
    11.5 deg of longitude to the east.
    """
    path = Path(path) if path is not None else DATA_DIR / DNI_FILE
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            frame = pd.read_csv(fh)
    else:
        frame = pd.read_csv(path)
    inner = dni_mod.DailyClimatologyDNI(frame, source=path.name, window_days=DNI_WINDOW_DAYS)
    return dni_mod.SolarTimeAligned(inner, DNI_DATA_LONGITUDE, site[1])


def annual_energies(store: RunStore, cfg, dni_petrolina=None) -> dict:
    """Annual MWh inside the 720 mm aperture, on both of the paper's DNI bases.

    Feeds :func:`heliostat.energy.annual_energy` a summary whose ``power_w``
    column is the power that landed *inside the aperture* -- which is
    additive over heliostats, so the per-heliostat column sums to the field's
    aperture power exactly. The efficiency surface, the declination
    interpolation and the 8760-hour walk are all library code from there on.
    """
    summary = store.summary()
    aperture_summary = summary.assign(power_w=summary["power_aperture_w"])
    n_helio = int(summary["heliostat_id"].nunique())

    out = {}
    for label, provider in (
        ("1kW", dni_mod.ConstantDNI(1000.0)),
        ("petrolina", dni_petrolina if dni_petrolina is not None else petrolina_provider()),
    ):
        res = energy_mod.annual_energy(
            aperture_summary, cfg, provider, year=cfg.sweep.dates[0].year, n_heliostats=n_helio
        )
        out[label] = {
            "annual_MWh": res["annual_energy_mwh"],
            "annual_dni_kwh_m2": res["annual_dni_kwh_m2"],
            "traced_timesteps": res["traced_timesteps"],
            "traced_declinations": res["traced_declinations"],
            "extrapolated_fraction": res["extrapolated_fraction"],
        }
    out["mirror_area_m2"] = float(cfg.field.mirror_area_m2 * n_helio)
    out["n_heliostats"] = n_helio
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

QUICK_HELIOSTATS = 60
QUICK_RAYS = 20_000

_QUICK_BANNER = """
============================================================================
QUICK MODE -- these numbers are NOT the paper's.
  {n} innermost heliostats (paper: 643), {rays:,} rays (paper: 120,000),
  1 date (paper: 7). Annual energy needs at least two declinations and is
  therefore not computed at all. This mode exists to prove the pipeline
  runs end to end, nothing more. check.py will refuse to compare it.
============================================================================
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce the nine configurations of SPIE paper 14212-5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--configs",
        nargs="*",
        default=["all"],
        help="e.g. 'axicon:twisting cassegrain' -- default all nine",
    )
    p.add_argument(
        "--mode",
        default="monte_carlo",
        choices=("monte_carlo", "ultra_fast", "fast_accurate"),
        help="monte_carlo reproduces the paper; the cone modes are a faster, "
        "documented-band alternative",
    )
    p.add_argument(
        "--rays", type=int, default=RAYS_PER_HELIOSTAT, help="Monte Carlo rays/heliostat"
    )
    p.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="ISO dates; default is the paper's seven",
    )
    p.add_argument("--out", default="runs/paper", help="output root (default runs/paper)")
    p.add_argument(
        "--quick",
        action="store_true",
        help=f"{QUICK_HELIOSTATS} innermost heliostats, {QUICK_RAYS} rays, first date only",
    )
    p.add_argument("--n-heliostats", type=int, default=None, help="keep only the innermost N")
    p.add_argument("--rebuild", action="store_true", help="re-trace runs that already exist")
    p.add_argument(
        "--instant-only",
        action="store_true",
        help="trace only the paper's instant (%s); skips annual energy" % INSTANT_KEY,
    )
    p.add_argument(
        "--shade-radius",
        type=float,
        default=None,
        help="override the secondary's shading-body radius for every layout, mm "
        "(0 disables it). Default is the per-layout value in SHADE_BODY_RADIUS_MM.",
    )
    p.add_argument("--base-seed", type=int, default=BASE_SEED)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE)
    p.add_argument("--list-timesteps", action="store_true", help="print the time grid and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    dates = tuple(_dt.date.fromisoformat(d) for d in args.dates) if args.dates else DATES
    n_heliostats = args.n_heliostats
    n_rays = args.rays
    if args.quick:
        dates = dates[:1]
        n_heliostats = n_heliostats or QUICK_HELIOSTATS
        n_rays = QUICK_RAYS
        print(_QUICK_BANNER.format(n=n_heliostats, rays=n_rays))

    if args.list_timesteps:
        from heliostat.solar import describe_time_grid

        print(describe_time_grid(paper_cfg(paper_optics("axicon"), n_rays=1, dates=dates), dates))
        return 0

    only_keys = (INSTANT_KEY,) if args.instant_only else None
    configs = parse_configs(args.configs)
    out_root = Path(args.out)
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)
        field = load_paper_field(n_heliostats)
    print(field.describe())
    if field.dropped_ids:
        print(f"  dropped coincident duplicates: {field.dropped_ids}")

    dni_petrolina = None
    rows = []
    for layout, figure in configs:
        name = config_name(layout, figure)
        run_dir = out_root / name
        print(f"\n=== {layout} / {figure} -> {run_dir} ===")
        t0 = time.perf_counter()
        if run_dir.exists() and not args.rebuild:
            print("  (existing run reused; pass --rebuild to re-trace)")
            store = RunStore(run_dir)
            cfg = paper_cfg(
                paper_optics(layout, args.shade_radius),
                n_rays=int(store.manifest.get("rays_per_heliostat", n_rays)),
                dates=dates,
                grid_size=int(store.manifest.get("grid_size", args.grid_size)),
            )
            store.cfg = cfg
        else:
            store = run_config(
                layout,
                figure,
                out_dir=run_dir,
                dates=dates,
                mode=args.mode,
                n_rays=n_rays,
                shade_radius_mm=args.shade_radius,
                base_seed=args.base_seed,
                grid_size=args.grid_size,
                field=field,
                only_keys=only_keys,
            )
            cfg = store.cfg
        trace_s = time.perf_counter() - t0

        record = {
            "layout": layout,
            "figure": figure,
            "run_dir": str(run_dir),
            "mode": str(store.manifest.get("mode", args.mode)),
            "rays_per_heliostat": int(store.manifest.get("rays_per_heliostat", n_rays)),
            "n_heliostats": int(store.manifest.get("n_heliostats", len(field))),
            "grid_size": int(store.manifest.get("grid_size", args.grid_size)),
            "dates": list(store.manifest.get("dates", [d.isoformat() for d in dates])),
            "shade_body_radius_mm": float(
                store.manifest.get(
                    "shade_body_radius_mm",
                    SHADE_BODY_RADIUS_MM[layout]
                    if args.shade_radius is None
                    else args.shade_radius,
                )
            ),
            "trace_seconds": trace_s,
            "paper_comparable": None,  # filled below
        }

        if store.has_timestep(INSTANT_KEY):
            record["instant"] = instant_metrics(store, cfg)
            record["instant_key"] = INSTANT_KEY
        else:
            record["instant"] = None

        # What was actually traced, not what was asked for: --instant-only
        # writes one timestep out of a seven-date grid.
        traced_dates = sorted({k.split("_")[0] for k in store.timestep_keys()})
        record["traced_dates"] = traced_dates
        record["traced_timestep_count"] = len(store.timestep_keys())

        if len(traced_dates) >= 2:
            if dni_petrolina is None:
                dni_petrolina = petrolina_provider()
            record["annual"] = annual_energies(store, cfg, dni_petrolina)
        else:
            record["annual"] = None
            print(
                "  annual energy skipped: needs at least two traced declinations "
                f"(this run traced {len(traced_dates)} date(s))"
            )

        # Two separate questions. The instant table needs the full field at
        # the paper's ray budget and grid; the annual table additionally
        # needs all seven dates. A single-instant validation run is
        # comparable on one table and not on the other.
        faithful = bool(
            record["mode"] == "monte_carlo"
            and record["rays_per_heliostat"] == RAYS_PER_HELIOSTAT
            and record["n_heliostats"] == PAPER_N_HELIOSTATS
            and record["grid_size"] == GRID_SIZE
            and record["shade_body_radius_mm"] == SHADE_BODY_RADIUS_MM[layout]
        )
        record["paper_comparable"] = {
            "instant": faithful and record["instant"] is not None,
            "annual": faithful and record["annual"] is not None and len(traced_dates) == len(DATES),
        }
        (results_dir / f"{name}.json").write_text(json.dumps(record, indent=2))

        row = {"layout": layout, "figure": figure}
        if record["annual"]:
            row["annual_MWh_1kW"] = record["annual"]["1kW"]["annual_MWh"]
            row["annual_MWh_petrolina"] = record["annual"]["petrolina"]["annual_MWh"]
        if record["instant"]:
            row.update(record["instant"])
        rows.append(row)
        print(f"  done in {trace_s:.1f}s")
        if record["instant"]:
            m = record["instant"]
            print(
                f"  instant {INSTANT_KEY}: peak {m['peak_kw_m2']:.1f} kW/m2, "
                f"window {m['window_kw']:.1f} kW, 720 mm {m['power_720mm_kw']:.1f} kW "
                f"({m['frac_720mm']:.4f}), {m['conc_720mm_suns']:.1f} suns, "
                f"r90 {m['r90_mm']:.1f} mm"
            )
        if record["annual"]:
            print(
                f"  annual: {record['annual']['1kW']['annual_MWh']:.2f} MWh @1kW, "
                f"{record['annual']['petrolina']['annual_MWh']:.2f} MWh @Petrolina"
            )

    summary = pd.DataFrame(rows)
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
