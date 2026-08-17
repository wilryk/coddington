"""Synthetic heliostat field layouts.

Ported from the owner's MATLAB reference (``FermatSpiral.m``): a golden-ratio
Fermat spiral, plus the angular-wedge and "road" filters that script applies
to shape a spiral into a usable field boundary.

This module is deliberately a flat file, not a ``heliostat.field_layouts``
*package* -- there was no need this session to turn :mod:`heliostat.field`
into a package, and a flat module keeps that decision reversible later
without an import-path break. Flag if a package layout (``layouts/fermat.py``,
``layouts/registry.py``, ...) is preferred once more layout kinds land.

Everything here works in **metres** (matching the MATLAB script and
:meth:`heliostat.field.HeliostatField.x_m`/``y_m``), converting to the
field's internal millimetres only in :func:`generate`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from heliostat.field import HeliostatField

GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0

#: The MATLAB script's ``2*pi/phi_g^2`` divergence angle between successive
#: points on the spiral, in radians.
GOLDEN_ANGLE_RAD = 2.0 * math.pi / GOLDEN**2

Filter = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# The spiral itself
# ---------------------------------------------------------------------------


def fermat_spiral(
    n: int,
    a_m: float,
    b: float = 0.5,
    divergence_rad: float = GOLDEN_ANGLE_RAD,
    k_start: int = 1,
) -> np.ndarray:
    """Golden-ratio Fermat spiral positions, in metres.

    Exactly the MATLAB recipe::

        phi_g = (sqrt(5)+1)/2
        theta_k = 2*pi/phi_g^2 * k
        r_k = a*k^b
        x = r_k*cos(theta_k); y = r_k*sin(theta_k)

    generalised only in the divergence angle (``divergence_rad``, default the
    golden angle used by the reference) and in where ``k`` starts.

    **Mapping from the MATLAB script's ``k`` range.** ``FermatSpiral.m``
    doesn't start at ``k=1``: it picks a physical radius band first
    (``rmin``/``rmax``) and derives ``kmin = ceil((rmin/a)^(1/b))``,
    ``kmax = floor((rmax/a)^(1/b))``, then evaluates ``k = kmin:kmax``. That
    is reproduced two ways here, and they give identical points because the
    formulas are pure functions of ``k`` (no running state):

    - pass ``k_start=kmin`` and ``n=kmax-kmin+1`` directly, or
    - generate from ``k_start=1`` over a wider ``n`` and post-filter with
      :func:`ring_filter(rmin, rmax) <ring_filter>`.

    :func:`generate` takes the second approach (oversample, then filter) so
    that the same code path handles radius bounds and every other filter
    uniformly; this function alone takes the first, since it has no filter
    machinery of its own.

    Args:
        n: number of points, i.e. ``k`` runs ``k_start .. k_start + n - 1``.
        a_m: spiral scale, ``r = a * k**b``, metres.
        b: spiral exponent (MATLAB example: 0.5, then 0.56 for the field
            layout figure).
        divergence_rad: angle between successive ``k``, radians. Default is
            the golden angle, which is what makes the packing look like a
            sunflower head / phyllotaxis pattern; any other value still
            produces *a* spiral but not that one.
        k_start: the first ``k`` value (MATLAB is 1-based and, per above,
            frequently starts higher than 1).

    Returns:
        ``(n, 2)`` float array of ``(x, y)`` in metres.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    k = np.arange(k_start, k_start + n, dtype=float)
    theta = divergence_rad * k
    r = a_m * k**b
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


# ---------------------------------------------------------------------------
# Composable filters -- each takes (N,2) metres, returns an (N,) bool mask
# ---------------------------------------------------------------------------


def wedge_filter(az_min_deg: float, az_max_deg: float) -> Filter:
    """Keep points whose polar angle falls in ``[az_min_deg, az_max_deg]``.

    **Angle convention: this is the plain mathematical one, not compass
    bearing.** ``theta = atan2(y, x)`` in degrees, 0 deg on +x, 90 deg on +y,
    increasing counter-clockwise, range ``(-180, 180]`` -- exactly what
    MATLAB's ``atan2(y,x)`` gives. This is *not* the same convention as
    :attr:`heliostat.field.HeliostatField.azimuth_deg`, which is a compass
    bearing (0 deg = +y/north, clockwise, via ``atan2(x, y)``). Judgment
    call: the task spec named the parameters ``az_min_deg``/``az_max_deg``
    and pinned the mapping to MATLAB's ``pi/4 .. 3*pi/4`` example, which is
    the math convention, so that's what's implemented -- but the name
    invites confusion against ``HeliostatField.azimuth_deg`` and a caller
    reaching for "the south wedge" by compass intuition will get the wrong
    answer. Flagging rather than silently renaming the parameters, since the
    signature was specified.

    MATLAB's example (``theta>=pi/4 & theta<=3*pi/4``) is reproduced by
    ``wedge_filter(45, 135)``.

    If ``az_min_deg > az_max_deg`` the band is taken to wrap through +-180
    (e.g. ``wedge_filter(170, -170)`` keeps a 20 deg band straddling the
    negative x-axis).
    """

    def _filter(xy_m: np.ndarray) -> np.ndarray:
        theta_deg = np.degrees(np.arctan2(xy_m[:, 1], xy_m[:, 0]))
        if az_min_deg <= az_max_deg:
            return (theta_deg >= az_min_deg) & (theta_deg <= az_max_deg)
        return (theta_deg >= az_min_deg) | (theta_deg <= az_max_deg)

    return _filter


def ring_filter(r_min_m: float, r_max_m: float) -> Filter:
    """Keep points with radius in ``[r_min_m, r_max_m]`` (inclusive)."""

    def _filter(xy_m: np.ndarray) -> np.ndarray:
        r = np.hypot(xy_m[:, 0], xy_m[:, 1])
        return (r >= r_min_m) & (r <= r_max_m)

    return _filter


def road_corridors(half_width_m: float, azimuths_deg: tuple[float, ...] = (180.0,)) -> Filter:
    """Cut a straight corridor of the given half-width along each azimuth.

    Each corridor is a half-line from the origin (not a full infinite line):
    it only removes points on the side of the origin that its azimuth points
    towards, matching the MATLAB example, which blocks the *south* side only
    (``y < 0``), not both the north and south rays through the tower.

    ``azimuths_deg`` uses the same compass convention as
    :attr:`heliostat.field.HeliostatField.azimuth_deg` (0 deg = +y, clockwise)
    -- deliberately different from :func:`wedge_filter`'s math convention,
    because a "road" is a physical direction from the tower and compass
    bearing is the natural way to name one; see the :func:`wedge_filter`
    docstring for why that pair of conventions doesn't match and is worth a
    second look if it trips anyone up.

    MATLAB's south-corridor example (``|x|<=10 & y<0`` removed) is
    reproduced by ``road_corridors(10, azimuths_deg=(180,))``.
    """

    def _filter(xy_m: np.ndarray) -> np.ndarray:
        x = xy_m[:, 0]
        y = xy_m[:, 1]
        keep = np.ones(x.shape[0], dtype=bool)
        for az_deg in azimuths_deg:
            az_rad = math.radians(az_deg)
            sin_az, cos_az = math.sin(az_rad), math.cos(az_rad)
            d_along = x * sin_az + y * cos_az  # distance out along the road direction
            d_perp = x * cos_az - y * sin_az  # distance off the road centreline
            in_corridor = (d_along > 0) & (np.abs(d_perp) <= half_width_m)
            keep &= ~in_corridor
        return keep

    return _filter


def min_spacing_filter(min_m: float) -> Filter:
    """Keep a point unless it is within ``min_m`` of an already-kept point.

    Greedy KD-tree cull in index (= ``k``) order: point ``i`` survives iff no
    surviving point ``j < i`` is within ``min_m`` of it, so lower indices
    always win a conflict. Two survivors are guaranteed to be at least
    ``min_m`` apart, with the boundary (exactly ``min_m``) treated as *too
    close* (``scipy``'s ``query_ball_point`` default of ``<=``).
    """

    def _filter(xy_m: np.ndarray) -> np.ndarray:
        n = xy_m.shape[0]
        mask = np.zeros(n, dtype=bool)
        if n == 0:
            return mask
        tree = cKDTree(xy_m)
        neighbour_lists = tree.query_ball_point(xy_m, r=min_m)
        for i, neighbours in enumerate(neighbour_lists):
            conflict = any(j < i and mask[j] for j in neighbours)
            if not conflict:
                mask[i] = True
        return mask

    return _filter


# ---------------------------------------------------------------------------
# generate() and CSV export
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}


def _register(kind: str):
    def _decorator(fn):
        _REGISTRY[kind] = fn
        return fn

    return _decorator


@_register("fermat")
def _build_fermat(
    n_k: int,
    *,
    a_m: float,
    b: float = 0.5,
    divergence_rad: float = GOLDEN_ANGLE_RAD,
    k_start: int = 1,
) -> np.ndarray:
    return fermat_spiral(n_k, a_m, b=b, divergence_rad=divergence_rad, k_start=k_start)


# Left for later sessions: "radial_stagger", "rect_grid" register here the
# same way, each returning an (n_k, 2) metre array from its own params.


def generate(
    kind: str,
    n_target: int,
    *,
    filters: tuple[Filter, ...] = (),
    oversample: float = 1.6,
    **params,
) -> HeliostatField:
    """Build a :class:`HeliostatField` from a registered layout generator.

    Generates ``ceil(n_target * oversample)`` raw candidates from the
    ``kind`` generator (``k`` order), applies every filter in ``filters`` in
    sequence (AND-combined), then truncates the survivors to the first
    ``n_target`` **in generation order** -- i.e. the lowest-``k`` survivors
    are kept, matching "truncate to n_target by k order" rather than e.g.
    truncating by radius. Raises informatively (naming the shortfall and the
    oversample used) if fewer than ``n_target`` survive.

    Ids are assigned ``0..n_target-1`` in survivor order. ``field.source``
    records ``kind``, every param, the filter count, and the oversample, so
    a generated field is traceable back to the CLI invocation that made it.
    """
    if kind not in _REGISTRY:
        raise ValueError(f"unknown layout kind {kind!r}; registered: {sorted(_REGISTRY)}")
    if n_target <= 0:
        raise ValueError(f"n_target must be positive, got {n_target}")
    if oversample < 1.0:
        raise ValueError(f"oversample must be >= 1.0, got {oversample}")

    n_k = max(n_target, math.ceil(n_target * oversample))
    xy_m = _REGISTRY[kind](n_k, **params)

    mask = np.ones(xy_m.shape[0], dtype=bool)
    for filt in filters:
        mask &= filt(xy_m)

    survivor_idx = np.flatnonzero(mask)  # already ascending / k-order
    if survivor_idx.size < n_target:
        raise ValueError(
            f"layout {kind!r}: only {survivor_idx.size} of {n_target} requested heliostats "
            f"survived filtering out of {n_k} candidates (oversample={oversample}). "
            "Increase oversample, loosen the filters, or lower n_target."
        )
    keep_idx = survivor_idx[:n_target]
    xy_kept = xy_m[keep_idx]

    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    source = (
        f"generate(kind={kind!r}, n_target={n_target}, {param_str}, "
        f"n_filters={len(filters)}, oversample={oversample})"
    )
    return HeliostatField(
        x_mm=xy_kept[:, 0] * 1000.0,
        y_mm=xy_kept[:, 1] * 1000.0,
        ids=np.arange(n_target, dtype=int),
        source=source,
    )


def write_field_csv(field: HeliostatField, path) -> None:
    """Write ``field`` to a CSV that :func:`heliostat.field.load_field` reads back identically.

    Positions are written in **metres** with headers ``x (m)``/``y (m)`` --
    one of ``load_field``'s recognised aliases, and metres (not the ``mm``
    aliases) so no unit conversion happens on read. ``heliostat_id`` is
    written alongside for human readability; ``load_field`` ignores it and
    assigns its own ``0..N-1`` ids in row order, which round-trips exactly
    because :func:`generate` already produced sequential ``0..N-1`` ids in
    row order.
    """
    df = pd.DataFrame(
        {
            "heliostat_id": field.ids,
            "x (m)": field.x_m,
            "y (m)": field.y_m,
        }
    )
    df.to_csv(Path(path), index=False)
