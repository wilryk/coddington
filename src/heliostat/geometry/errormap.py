"""Measured (FEA) error-map import -- docs/ui-spec-v0.2.md §E.

A user with a real deformation map of a heliostat (gravity sag, wind load,
thermal, from FEA or deflectometry) can import it as a gridded
sag-deviation CSV -- ``x, y, dz`` from the nominal surface, in exactly the
grid convention :func:`heliostat.web.app._sag_fea_csv` exports (§D): ``x_m``
``y_m`` in meters (the heliostat's own aperture/mirror-plane frame -- the
same ``(u, v)``-derived local coordinates :mod:`heliostat.trace.mc`
evaluates figure slopes in), ``dz_mm`` in millimeters, preceded by ``#``
commented metadata lines. That symmetry is deliberate: a Coddington sag
export, annotated by an FEA tool with a deformation delta and reimported,
round-trips.

Two things happen once, at import (:func:`parse_error_map_csv` /
:func:`build_error_map`), never per ray:

1. The scattered ``(x, y, dz)`` rows are reassembled into the regular grid
   they were sampled from (§D's exporter drops points that fall outside
   every facet, so a faceted design's export is a grid with holes, not a
   dense rectangle -- reconstructed here from the surviving points' own
   coordinates, not assumed).
2. The grid's gradient -- ``(ddz/dx, ddz/dy)``, i.e. the LOCAL SLOPE the
   deviation implies at every grid point -- is precomputed once via central
   differences. Holes are filled by nearest-neighbour extrapolation purely
   so a bilinear lookup near a gap edge never reads a ``NaN``; the reported
   RMS below is computed from the ORIGINAL (unfilled) points only.

:meth:`ErrorMap.sample_slopes` is the per-ray cost: a vectorized bilinear
lookup into the two precomputed slope grids, independent of how fine the
imported grid is -- the trace-time guarantee §E promises.

Convention pin -- the implied RMS slope error
-----------------------------------------------
§G's glossary describes ``slope_error_mrad`` as "the local surface normal
deviates from the design surface by this RMS angle", but the number that
control actually feeds :func:`heliostat.trace.mc._perturb_unit` is the
PER-AXIS standard deviation applied independently to each of the mirror's
two tangent axes (``u``/``v``, or a facet's own ``fu``/``fv``) -- confirmed
by ``tests/test_mc_physics_fixes.py``'s ``test_specularity_isotropic_...``,
which recovers a combined 2-D RMS of ``sigma * sqrt(2)`` from a per-axis
sigma equal to the input value, for the sibling ``specularity_mrad`` field
built the same way. So a number DIRECTLY comparable to ``slope_error_mrad``
-- addable in quadrature, not double-counted -- is the map's own per-axis
slope RMS, pooling both gradient components under one isotropy assumption:

    rms_slope_rad = sqrt(mean(dzdx_rad**2 + dzdy_rad**2) / 2)

(the ``/ 2`` is what makes an isotropic map's reported number equal either
component's own RMS individually, matching what feeding that same number
into ``slope_error_mrad`` would apply per axis -- omitting it would silently
report the sqrt(2)-larger combined-magnitude RMS instead, double-counting
against the analytic control by that same factor).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

#: Relative tolerance for "the grid is regularly spaced" -- generous enough
#: to absorb the %.6g round-trip precision loss the §D exporter's own
#: ``_fea_csv_bytes`` formatting introduces, tight enough to catch a
#: genuinely irregular/scattered import.
_GRID_TOL = 1e-3


@dataclass(frozen=True)
class ErrorMap:
    """A parsed, pre-processed measured sag-deviation map.

    ``x_m``/``y_m`` are the 1-D regular grid axes (strictly ascending,
    meters, the heliostat's own aperture-frame convention -- see the module
    docstring). ``dz_mm`` is the ``(ny, nx)`` sag-deviation grid, ``NaN``
    where the source CSV had no point (gaps in a faceted design's export).
    ``dzdx_grid``/``dzdy_grid`` are the precomputed slope grids, radians,
    holes filled so a lookup near a gap edge is always defined. ``valid``
    is the original coverage mask (``True`` where the CSV actually had a
    point) -- used only for :attr:`coverage_fraction`; trace-time lookups
    always use the filled slope grids regardless of coverage.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    dz_mm: np.ndarray
    dzdx_grid: np.ndarray
    dzdy_grid: np.ndarray
    valid: np.ndarray
    coverage_fraction: float
    rms_slope_mrad: float

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(ny, nx)`` -- the number of samples along each axis."""
        return self.dz_mm.shape

    def sample_slopes(self, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear-interpolated ``(dzdx, dzdy)``, in radians, at each ray's
        mirror-point ``(x_mm, y_mm)`` (the same aperture-frame local
        coordinates the sag/slope evaluation already uses).

        Queries outside the imported grid's own extent are clamped to the
        nearest edge rather than extrapolated or zeroed -- the map is
        expected to cover the aperture it was measured over; a ray landing
        a hair outside due to float roundoff at the very edge should not
        silently lose its perturbation, and a ray landing well outside
        (a design edited larger than the map after import) degrades to the
        map's own edge value rather than blowing up.
        """
        xs, ys = self.x_m, self.y_m
        nx, ny = xs.size, ys.size
        x = np.asarray(x_mm, dtype=float) / 1000.0
        y = np.asarray(y_mm, dtype=float) / 1000.0

        dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else 1.0

        fx = np.clip((x - xs[0]) / dx, 0.0, nx - 1) if nx > 1 else np.zeros_like(x)
        fy = np.clip((y - ys[0]) / dy, 0.0, ny - 1) if ny > 1 else np.zeros_like(y)

        ix0 = np.clip(np.floor(fx).astype(int), 0, max(nx - 2, 0))
        iy0 = np.clip(np.floor(fy).astype(int), 0, max(ny - 2, 0))
        ix1 = np.minimum(ix0 + 1, nx - 1)
        iy1 = np.minimum(iy0 + 1, ny - 1)
        tx = np.clip(fx - ix0, 0.0, 1.0)
        ty = np.clip(fy - iy0, 0.0, 1.0)

        def _bilinear(grid: np.ndarray) -> np.ndarray:
            v00 = grid[iy0, ix0]
            v01 = grid[iy0, ix1]
            v10 = grid[iy1, ix0]
            v11 = grid[iy1, ix1]
            return (
                v00 * (1 - tx) * (1 - ty)
                + v01 * tx * (1 - ty)
                + v10 * (1 - tx) * ty
                + v11 * tx * ty
            )

        return _bilinear(self.dzdx_grid), _bilinear(self.dzdy_grid)

    def to_storage_dict(self) -> dict:
        """Compact JSON-safe payload for Library/project persistence: the
        raw grid only (axes + ``dz_mm`` with holes as ``None``) -- gradients
        and the reported RMS are cheap to recompute on load
        (:func:`build_error_map`), not worth doubling the payload to store.
        """
        dz = self.dz_mm
        rows = [
            [None if not np.isfinite(v) else round(float(v), 6) for v in row] for row in dz
        ]
        return {
            "x_m": [round(float(v), 9) for v in self.x_m],
            "y_m": [round(float(v), 9) for v in self.y_m],
            "dz_mm": rows,
        }

    @staticmethod
    def from_storage_dict(data: dict) -> "ErrorMap":
        """Inverse of :meth:`to_storage_dict` -- rebuilds gradients fresh."""
        x_m = np.asarray(data["x_m"], dtype=float)
        y_m = np.asarray(data["y_m"], dtype=float)
        dz_mm = np.array(
            [[np.nan if v is None else float(v) for v in row] for row in data["dz_mm"]],
            dtype=float,
        )
        return build_error_map(x_m, y_m, dz_mm)


def _validate_regular_spacing(axis: np.ndarray, name: str) -> None:
    """Raise :class:`ValueError` unless ``axis`` (already ascending/unique)
    is uniformly spaced to within :data:`_GRID_TOL`.

    Shared by :func:`_regular_axis` (scattered CSV rows, which still need
    dedup first) and :func:`build_error_map` (an already-gridded axis,
    e.g. from :meth:`ErrorMap.from_storage_dict` -- a stored/round-tripped
    document is untrusted input too, not guaranteed regular just because it
    came from this module originally) -- one check, so a malformed grid is
    rejected the same way regardless of which path it arrived by.
    """
    if axis.size < 2:
        raise ValueError(f"error map needs at least 2 distinct {name} coordinates, found {axis.size}")
    diffs = np.diff(axis)
    if np.any(diffs <= 0):
        raise ValueError(f"error map's {name} coordinates must be strictly ascending")
    spacing = float(np.median(diffs))
    if spacing <= 0 or not np.allclose(diffs, spacing, rtol=_GRID_TOL, atol=1e-9):
        raise ValueError(
            f"error map's {name} coordinates are not a regular grid "
            "(irregular/scattered point imports are not supported -- export a "
            "regular grid, e.g. from Coddington's own §D sag CSV export)"
        )


def _regular_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Unique, ascending grid coordinates along one axis, validated regular.

    Raises :class:`ValueError` (caught and reported to the user as a 4xx by
    the import endpoint) for fewer than 2 distinct coordinates or spacing
    that is not uniform to within :data:`_GRID_TOL` -- an irregular/
    scattered point cloud is not something the bilinear lookup in
    :meth:`ErrorMap.sample_slopes` can serve.
    """
    axis = np.unique(values)
    _validate_regular_spacing(axis, name)
    return axis


def build_error_map(x_m: np.ndarray, y_m: np.ndarray, dz_mm: np.ndarray) -> ErrorMap:
    """Build an :class:`ErrorMap` from an already-gridded ``(ny, nx)``
    ``dz_mm`` array and its ``x_m``/``y_m`` axes (ascending, regular).

    Used directly by :meth:`ErrorMap.from_storage_dict` (the grid is already
    reconstructed there) and internally by :func:`parse_error_map_csv`
    (which reconstructs the grid from scattered CSV rows first).
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    dz_mm = np.asarray(dz_mm, dtype=float)
    ny, nx = dz_mm.shape
    if x_m.size != nx or y_m.size != ny:
        raise ValueError(
            f"error map grid shape {dz_mm.shape} does not match axis lengths "
            f"(x: {x_m.size}, y: {y_m.size})"
        )
    _validate_regular_spacing(x_m, "x")
    _validate_regular_spacing(y_m, "y")

    valid = np.isfinite(dz_mm)
    n_valid = int(valid.sum())
    n_total = dz_mm.size
    if n_valid < 4:
        raise ValueError(
            f"error map has only {n_valid} valid point(s) -- need at least 4 to interpolate"
        )
    coverage_fraction = n_valid / n_total if n_total else 0.0

    # Fill holes (NaN, outside every facet) by nearest-neighbour before
    # differencing, so a bilinear lookup near a gap edge is always defined.
    # This never affects the reported RMS below, which reads only the
    # original valid points.
    if n_valid < n_total:
        from scipy.interpolate import NearestNDInterpolator

        gy, gx = np.meshgrid(y_m, x_m, indexing="ij")
        interp = NearestNDInterpolator(
            np.column_stack([gy[valid], gx[valid]]), dz_mm[valid]
        )
        dz_filled = dz_mm.copy()
        dz_filled[~valid] = interp(gy[~valid], gx[~valid])
    else:
        dz_filled = dz_mm

    # Gradient in radians: dz is millimeters, x/y are meters, so
    # d(dz_mm / 1000) / d(x_m) is a dimensionless slope -- the small-angle
    # radian convention every other optical-error field in this codebase
    # uses (see mc.py's own slope_error_mrad).
    dzdy_grid, dzdx_grid = np.gradient(dz_filled / 1000.0, y_m, x_m)

    rms_slope_rad = float(
        np.sqrt(np.mean(dzdx_grid[valid] ** 2 + dzdy_grid[valid] ** 2) / 2.0)
    )

    return ErrorMap(
        x_m=x_m,
        y_m=y_m,
        dz_mm=dz_mm,
        dzdx_grid=dzdx_grid,
        dzdy_grid=dzdy_grid,
        valid=valid,
        coverage_fraction=coverage_fraction,
        rms_slope_mrad=rms_slope_rad * 1000.0,
    )


def parse_error_map_csv(text: str | bytes) -> ErrorMap:
    """Parse a §D-convention sag CSV (``x_m, y_m, dz_mm`` rows, ``#``
    commented metadata lines tolerated and skipped) into an
    :class:`ErrorMap`.

    Tolerant of the exact §D header (units/subject/grid comment lines) but
    does not require it -- any ``#``-prefixed or blank line is skipped, so a
    bare numeric CSV imports too. Column order is fixed (``x, y, dz``),
    matching :func:`heliostat.web.app._sag_fea_csv`; there is no header row
    to name columns by (§D's exporter deliberately omits one).
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    reader = csv.reader(io.StringIO(text))
    for row_num, row in enumerate(reader, start=1):
        if not row:
            continue
        first = row[0].strip()
        if not first or first.startswith("#"):
            continue
        if len(row) < 3:
            raise ValueError(f"error map CSV row {row_num}: expected 3 columns (x, y, dz), got {len(row)}")
        try:
            x, y, z = float(row[0]), float(row[1]), float(row[2])
        except ValueError as exc:
            raise ValueError(f"error map CSV row {row_num}: non-numeric value ({exc})") from exc
        xs.append(x)
        ys.append(y)
        zs.append(z)

    if len(xs) < 4:
        raise ValueError(f"error map CSV has only {len(xs)} data row(s) -- need at least 4 to interpolate")

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    zs_arr = np.asarray(zs)

    x_axis = _regular_axis(xs_arr, "x")
    y_axis = _regular_axis(ys_arr, "y")

    dz_mm = np.full((y_axis.size, x_axis.size), np.nan)
    # Nearest-axis-index assignment (not exact equality) absorbs the same
    # %.6g formatting noise _regular_axis's tolerance already allows for.
    ix = np.searchsorted(x_axis, xs_arr)
    ix = np.clip(ix, 0, x_axis.size - 1)
    ix_left_closer = (ix > 0) & (np.abs(xs_arr - x_axis[np.maximum(ix - 1, 0)]) < np.abs(xs_arr - x_axis[ix]))
    ix = np.where(ix_left_closer, ix - 1, ix)
    iy = np.searchsorted(y_axis, ys_arr)
    iy = np.clip(iy, 0, y_axis.size - 1)
    iy_left_closer = (iy > 0) & (np.abs(ys_arr - y_axis[np.maximum(iy - 1, 0)]) < np.abs(ys_arr - y_axis[iy]))
    iy = np.where(iy_left_closer, iy - 1, iy)
    dz_mm[iy, ix] = zs_arr

    return build_error_map(x_axis, y_axis, dz_mm)
