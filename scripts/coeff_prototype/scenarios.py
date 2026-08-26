"""The four B2 benchmark scenarios: default field, window-edge clipping,
heavy blocking, and cylinder-seam wrap.

Every scenario traces a **flat figure** (``c3 = c4 = c5 = 0``, no
manufacturing-error terms) for every heliostat in every scenario, not just
the default-field one -- a deliberate simplification kept consistent
throughout so no scenario's numbers are confounded by a different figure
convention than another's. Only the pointing angles (``rot_az_deg``/
``rot_el_deg``) come from the real aiming solve; the resulting figure
coefficients are discarded.

Scopes traced per docs/ui-spec-v0.2.md B2:

1. :func:`scenario_default_field` -- all 643 heliostats of the packaged
   manuscript field, one sun position, ``ultra_fast`` cone_kwargs, no
   occluders (matches how the app's field sweep runs ultra_fast today).
2. :func:`scenario_window_clipping` -- same field/sun, a shrunk receiver
   window (700 mm half-extent instead of 2000 mm) so outer-ring spots clip
   hard; a representative ~30-heliostat subset (inner to outer) rather than
   the full field, since the point is real clipped samples, not scale.
3. :func:`scenario_heavy_blocking` -- low sun, the field's smallest-radius
   ring, real neighbour occluders passed explicitly so the cone tracer's
   (and ``sampling.py``'s) actual per-sample transmission raster measures
   real occlusion discontinuities -- the non-production direct-occluder
   path section B's docstring calls out.
4. :func:`scenario_cylinder_seam` -- a :class:`CylinderReceiver`, north-
   sector heliostats (whose natural aim point IS the +y seam --
   see ``heliostat/geometry/receiver.py``'s module docstring), so real flux
   crosses ``u = +-half_circumference`` and wraps.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heliostat.field import HeliostatField, downselect, load_field, neighbour_pairs
from heliostat.geometry.aiming import solve_prime_focus_to_receiver
from heliostat.geometry.receiver import CylinderReceiver, FlatWindowReceiver
from heliostat.geometry.secondary import NoSecondary
from heliostat.geometry.shading import MirrorGeometry, search_radius_for
from heliostat.sweep import standard_optics
from heliostat.trace.modes import ULTRA_FAST

FIELD_PATH = _ROOT / "src" / "heliostat" / "web" / "static" / "data" / "field_645.csv"
MIRROR_WIDTH_MM = 5000.0
MIRROR_HEIGHT_MM = 3000.0
FLUX_GRID = (128, 128)

#: Scenario 1's sun position -- mid-morning-ish, arbitrary but reasonable
#: (matches neither fixture nor prior benchmark exactly, per the task's own
#: "pick something reasonable and state it" instruction).
DEFAULT_SOLAR_AZ_DEG = 150.0
DEFAULT_SOLAR_EL_DEG = 45.0

#: Scenario 3's low-sun position, chosen so shadows/blocking are long
#: relative to the field's own spacing.
LOW_SUN_AZ_DEG = 150.0
LOW_SUN_EL_DEG = 10.0

SEED = 20260826  # fixed, explicit -- used only for downselect()'s reproducible sampling


def load_default_field() -> HeliostatField:
    """The packaged 643-heliostat manuscript field, at the app's own
    mirror dimensions (``src/heliostat/web/app.py``'s
    ``_load_manuscript_field()`` uses the identical call)."""
    return load_field(FIELD_PATH, mirror_width_mm=MIRROR_WIDTH_MM, mirror_height_mm=MIRROR_HEIGHT_MM)


@dataclass
class HeliostatCase:
    """One heliostat's traced inputs at one instant -- everything
    ``sampling.trace_heliostat_samples`` needs beyond ``secondary``/
    ``receiver``/``kernel``, which the owning :class:`Scenario` carries."""

    heliostat_id: int
    x_mm: float
    y_mm: float
    rot_az_deg: float
    rot_el_deg: float
    solar_az_deg: float
    solar_el_deg: float
    ring_radius_m: float
    occluders: list | None = None
    c3: float = 0.0
    c4: float = 0.0
    c5: float = 0.0


@dataclass
class Scenario:
    name: str
    secondary: object
    receiver: object
    cases: list = dataclass_field(default_factory=list)
    cone_kwargs: dict = dataclass_field(default_factory=lambda: dict(ULTRA_FAST.cone_kwargs))
    flux_grid: tuple = FLUX_GRID
    wrap_u: bool = False
    notes: str = ""


def _case(field: HeliostatField, i: int, opt, solar_az: float, solar_el: float, occluders=None) -> HeliostatCase:
    x, y = float(field.x_mm[i]), float(field.y_mm[i])
    sol = opt.aim(x, y, solar_az, solar_el)
    return HeliostatCase(
        heliostat_id=int(field.ids[i]),
        x_mm=x,
        y_mm=y,
        rot_az_deg=sol.rot_az_deg,
        rot_el_deg=sol.rot_el_deg,
        solar_az_deg=solar_az,
        solar_el_deg=solar_el,
        ring_radius_m=float(field.radius_mm[i]) / 1000.0,
        occluders=occluders,
    )


def scenario_default_field() -> Scenario:
    """Scenario 1: full 643-heliostat field, one timestep, ultra_fast, no
    occluders -- matches how the app's field sweep runs ultra_fast today."""
    field = load_default_field()
    opt = standard_optics("prime_focus")
    cases = [_case(field, i, opt, DEFAULT_SOLAR_AZ_DEG, DEFAULT_SOLAR_EL_DEG) for i in range(len(field))]
    return Scenario(
        name="default_field",
        secondary=opt.secondary,
        receiver=opt.receiver,
        cases=cases,
        notes=f"{len(cases)} heliostats, flat figure, sun az={DEFAULT_SOLAR_AZ_DEG} el={DEFAULT_SOLAR_EL_DEG}",
    )


def scenario_window_clipping(n_subset: int = 30) -> Scenario:
    """Scenario 2: same field/sun, a shrunk receiver window (700 mm
    half-extent) so outer-ring spots clip hard -- a representative
    ``n_subset``-heliostat sample spanning inner to outer rings, chosen by
    ``heliostat.field.downselect``'s stratified-uniform method (radius-ring
    then azimuth stratified, reproducible from a fixed seed) rather than
    the full 643, since the point is real clipped samples, not scale.
    """
    field = load_default_field()
    opt = standard_optics("prime_focus")
    small_receiver = FlatWindowReceiver(
        z_mm=opt.receiver.z_mm, half_u_mm=700.0, half_v_mm=700.0, facing=opt.receiver.facing
    )
    idx = downselect(field, n_subset, method="uniform", seed=SEED)
    cases = [_case(field, int(i), opt, DEFAULT_SOLAR_AZ_DEG, DEFAULT_SOLAR_EL_DEG) for i in idx]
    return Scenario(
        name="window_clipping",
        secondary=opt.secondary,
        receiver=small_receiver,
        cases=cases,
        notes=(
            f"{len(cases)}/{len(field)} heliostats (downselect uniform, seed={SEED}), "
            f"half_u=half_v=700mm (vs standard 2000mm)"
        ),
    )


def scenario_heavy_blocking(n_cluster: int = 16, neighbour_factor: float = 3.0) -> Scenario:
    """Scenario 3: low sun, the field's smallest-radius (innermost) ring,
    real neighbour occluders built via ``MirrorGeometry.build`` and passed
    explicitly to the trace -- exercises the direct-occluder transmission
    raster section B's docstring flags as the non-production path B2 is
    meant to validate.

    ``neighbour_factor`` sizes the occluder search radius as a multiple of
    the mirror's full diagonal (``hypot(width, height)``); occluders come
    from the WHOLE field (not just the cluster), so a cluster heliostat's
    true nearest neighbours are used even if one sits just outside the
    cluster itself.
    """
    field = load_default_field()
    opt = standard_optics("prime_focus")
    order = np.argsort(field.radius_mm)
    cluster_idx = order[:n_cluster]

    diag_mm = float(np.hypot(MIRROR_WIDTH_MM, MIRROR_HEIGHT_MM))
    search_radius_mm = neighbour_factor * diag_mm
    neighbours = neighbour_pairs(field, search_radius_mm)

    cases = []
    for i in cluster_idx:
        i = int(i)
        nbr_idx = neighbours[i]
        occluders = []
        for j in nbr_idx:
            j = int(j)
            nx, ny = float(field.x_mm[j]), float(field.y_mm[j])
            nsol = opt.aim(nx, ny, LOW_SUN_AZ_DEG, LOW_SUN_EL_DEG)
            occluders.append(
                MirrorGeometry.build(
                    nx, ny, nsol.rot_az_deg, nsol.rot_el_deg,
                    MIRROR_WIDTH_MM / 2.0, MIRROR_HEIGHT_MM / 2.0,
                )
            )
        case = _case(field, i, opt, LOW_SUN_AZ_DEG, LOW_SUN_EL_DEG, occluders=occluders or None)
        cases.append(case)

    return Scenario(
        name="heavy_blocking",
        secondary=opt.secondary,
        receiver=opt.receiver,
        cases=cases,
        notes=(
            f"{len(cases)} innermost-ring heliostats, sun az={LOW_SUN_AZ_DEG} el={LOW_SUN_EL_DEG}, "
            f"neighbour search radius {search_radius_mm / 1000.0:.1f} m "
            f"({neighbour_factor:g}x mirror diagonal), "
            f"neighbour counts {[len(c.occluders or []) for c in cases]}"
        ),
    )


def scenario_cylinder_seam(n_heliostats: int = 8) -> Scenario:
    """Scenario 4: a :class:`CylinderReceiver`; ``n_heliostats`` closest to
    due-north of the tower, whose natural aim point (see
    ``heliostat/geometry/receiver.py``'s module docstring) sits exactly at
    the +y seam, so real flux crosses ``u = +-half_circumference`` and wraps.
    """
    field = load_default_field()
    receiver = CylinderReceiver(center_z_mm=27000.0, radius_mm=3000.0, height_mm=6000.0)
    secondary = NoSecondary()

    az = field.azimuth_deg  # compass bearing from tower, 0 = north
    dist_from_north = np.minimum(az, 360.0 - az)
    order = np.argsort(dist_from_north)
    idx = order[:n_heliostats]

    cases = []
    for i in idx:
        i = int(i)
        x, y = float(field.x_mm[i]), float(field.y_mm[i])
        sol = solve_prime_focus_to_receiver(x, y, DEFAULT_SOLAR_AZ_DEG, DEFAULT_SOLAR_EL_DEG, receiver)
        cases.append(
            HeliostatCase(
                heliostat_id=int(field.ids[i]),
                x_mm=x,
                y_mm=y,
                rot_az_deg=sol.rot_az_deg,
                rot_el_deg=sol.rot_el_deg,
                solar_az_deg=DEFAULT_SOLAR_AZ_DEG,
                solar_el_deg=DEFAULT_SOLAR_EL_DEG,
                ring_radius_m=float(field.radius_mm[i]) / 1000.0,
            )
        )
    return Scenario(
        name="cylinder_seam",
        secondary=secondary,
        receiver=receiver,
        cases=cases,
        wrap_u=True,
        notes=(
            f"{len(cases)} north-sector heliostats (closest to due-north azimuth), "
            f"cylinder radius 3000mm height 6000mm center_z 27000mm, "
            f"sun az={DEFAULT_SOLAR_AZ_DEG} el={DEFAULT_SOLAR_EL_DEG}"
        ),
    )


ALL_SCENARIOS = {
    "default_field": scenario_default_field,
    "window_clipping": scenario_window_clipping,
    "heavy_blocking": scenario_heavy_blocking,
    "cylinder_seam": scenario_cylinder_seam,
}
