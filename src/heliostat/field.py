"""Heliostat field: positions, downselection, and neighbour queries.

Positions are held in **millimetres** internally. Source files are commonly
in metres; the conversion happens once, here, rather than being repeated at
every call site.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_X_ALIASES = ("x (m)", "x_s (m)", "x", "x (mm)")
_Y_ALIASES = ("y (m)", "y_s (m)", "y", "y (mm)")


def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...], which: str) -> str:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise KeyError(f"No {which} column found. Looked for {aliases}, file has {list(df.columns)}")


def _read_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an x/y position table from .xlsx or .csv, in metres."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        # openpyxl is not a hard dependency of this package -- most fields
        # arrive as CSV, and a spreadsheet engine is weight the CSV path does
        # not need. Say so plainly rather than letting pandas' own
        # "missing optional dependency" surface from three frames down.
        try:
            df = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                f"reading {path.name} needs a spreadsheet engine: "
                "pip install openpyxl (or save the field as .csv)"
            ) from exc
    else:
        df = pd.read_csv(path)
    xcol = _pick_column(df, _X_ALIASES, "x")
    ycol = _pick_column(df, _Y_ALIASES, "y")
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    if "mm" in str(xcol).lower():
        x, y = x / 1000.0, y / 1000.0
    return x, y


@dataclass(frozen=True)
class HeliostatField:
    """Heliostat centre positions, in millimetres.

    ``mirror_width_mm``/``mirror_height_mm`` are carried on the field
    (rather than threaded through every downstream call as a separate cfg
    object) so that :mod:`heliostat.geometry.shading` can build mirror
    geometry directly from a field without a second source of dimensions
    that could disagree with the one the field was loaded with. They default
    to ``0.0`` for a field assembled without them (e.g. a synthetic one in a
    test); callers that need real geometry from such a field must pass
    dimensions explicitly.

    ``dropped_ids`` records heliostat ids removed at load time because they
    shared a position with a lower-indexed heliostat (see :func:`load_field`).
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    ids: np.ndarray
    mirror_width_mm: float = 0.0
    mirror_height_mm: float = 0.0
    dropped_ids: tuple = ()
    source: str = ""

    def __len__(self) -> int:
        return int(self.x_mm.size)

    @property
    def x_m(self) -> np.ndarray:
        return self.x_mm / 1000.0

    @property
    def y_m(self) -> np.ndarray:
        return self.y_mm / 1000.0

    @property
    def radius_mm(self) -> np.ndarray:
        return np.hypot(self.x_mm, self.y_mm)

    @property
    def azimuth_deg(self) -> np.ndarray:
        """Compass bearing of each heliostat from the tower, degrees CW from +y."""
        return np.mod(np.degrees(np.arctan2(self.x_mm, self.y_mm)), 360.0)

    @property
    def xy_mm(self) -> np.ndarray:
        return np.column_stack((self.x_mm, self.y_mm))

    def subset(self, heliostat_ids) -> "HeliostatField":
        """Select by heliostat ID, not array position.

        Ids keep whatever numbering the source file (or ``load_field``'s
        coincident-drop) gave them, so once heliostats have been dropped the
        two diverge and every caller means "heliostat 300", not "row 300".
        Unknown ids raise rather than silently select a neighbour.
        """
        wanted = [int(h) for h in np.atleast_1d(np.asarray(heliostat_ids))]
        row_of = {int(h): k for k, h in enumerate(self.ids)}
        missing = [h for h in wanted if h not in row_of]
        if missing:
            extra = (
                f" (dropped as coincident duplicates: {self.dropped_ids})"
                if (self.dropped_ids and any(m in self.dropped_ids for m in missing))
                else ""
            )
            raise KeyError(f"heliostat id(s) not in field: {missing}{extra}")
        idx = np.array([row_of[h] for h in wanted], dtype=int)
        return HeliostatField(
            x_mm=self.x_mm[idx],
            y_mm=self.y_mm[idx],
            ids=self.ids[idx],
            mirror_width_mm=self.mirror_width_mm,
            mirror_height_mm=self.mirror_height_mm,
            dropped_ids=self.dropped_ids,
            source=f"{self.source}[subset n={idx.size}]",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "heliostat_id": self.ids,
                "x_m": self.x_m,
                "y_m": self.y_m,
                "radius_m": self.radius_mm / 1000.0,
                "azimuth_deg": self.azimuth_deg,
            }
        )

    def describe(self) -> str:
        r = self.radius_mm / 1000.0
        return (
            f"{len(self)} heliostats from {self.source}\n"
            f"  x {self.x_m.min():.1f}..{self.x_m.max():.1f} m, "
            f"y {self.y_m.min():.1f}..{self.y_m.max():.1f} m\n"
            f"  radius {r.min():.1f}..{r.max():.1f} m"
        )


def coincident_pairs(field: HeliostatField, tol_mm: float = 1.0):
    """Positions shared by two or more heliostats.

    Two heliostats at the same point cannot both exist. They also fail to
    shade each other -- :func:`heliostat.geometry.shading` requires the
    occluder to be strictly ahead (``t > 1e-6``), and a coincident mirror is
    at ``t = 0`` -- so each would be traced and summed at full power,
    double-counting that position. Cheap to detect, and invisible in every
    downstream total, so it is checked at load rather than left to be
    noticed.
    """
    tree = cKDTree(field.xy_mm)
    return sorted({(int(min(i, j)), int(max(i, j))) for i, j in tree.query_pairs(r=float(tol_mm))})


def load_field(
    path,
    *,
    mirror_width_mm: float = 0.0,
    mirror_height_mm: float = 0.0,
    coincident_tol_mm: float = 1.0,
) -> HeliostatField:
    """Load a heliostat field from a position file, coincident duplicates removed.

    ``path`` is a ``.csv`` or ``.xlsx`` table with x/y columns matched by a
    flexible set of aliases (metres or millimetres, several common header
    spellings -- see ``_X_ALIASES``/``_Y_ALIASES``). Ids are assigned
    ``0..N-1`` in file order.

    Real position files can carry byte-identical duplicate rows -- a
    field-file authoring quirk, not a design intent. Any two heliostats
    found within ``coincident_tol_mm`` of each other are treated as one
    physical mirror recorded twice: the higher-indexed id of each pair is
    dropped, a :class:`UserWarning` names the ids, and the surviving field's
    ``dropped_ids`` records what was removed. This is generic distance-based
    detection -- it does not hardcode which ids a particular field happens to
    duplicate.
    """
    x, y = _read_xy(path)
    fld = HeliostatField(
        x_mm=x * 1000.0,
        y_mm=y * 1000.0,
        ids=np.arange(x.size, dtype=int),
        mirror_width_mm=mirror_width_mm,
        mirror_height_mm=mirror_height_mm,
        source=Path(path).name,
    )
    pairs = coincident_pairs(fld, tol_mm=coincident_tol_mm)
    if pairs:
        drop_rows = sorted({max(i, j) for i, j in pairs})
        dropped_ids = tuple(int(fld.ids[i]) for i in drop_rows)
        keep = np.ones(len(fld), dtype=bool)
        keep[drop_rows] = False
        listed = ", ".join(f"{int(fld.ids[i])}={int(fld.ids[j])}" for i, j in pairs)
        warnings.warn(
            f"{fld.source}: {len(pairs)} coincident heliostat position(s) "
            f"({listed}), within {coincident_tol_mm} mm. Dropping the higher id "
            f"of each pair ({dropped_ids}) -- they do not shade each other and "
            f"each would be traced at full power, double-counting that position.",
            stacklevel=2,
        )
        fld = HeliostatField(
            x_mm=fld.x_mm[keep],
            y_mm=fld.y_mm[keep],
            ids=fld.ids[keep],
            mirror_width_mm=mirror_width_mm,
            mirror_height_mm=mirror_height_mm,
            dropped_ids=dropped_ids,
            source=fld.source,
        )
    return fld


def downselect(
    field: HeliostatField,
    n: int,
    method: str = "farthest_point",
    seed: int = 0,
) -> np.ndarray:
    """Choose ``n`` representative heliostats. Returns indices into ``field``.

    ``farthest_point``
        Greedy max-dispersion, reproducible: it starts from the heliostat
        nearest the field centroid rather than a random one. Heavily weights
        the perimeter, which is good for showing the extremes.

    ``uniform``
        Stratified over radius rings and azimuth sectors, giving coverage
        that tracks heliostat density instead of over-representing the
        boundary. Better when the downselect is meant to stand in for the
        whole field.
    """
    if n >= len(field):
        return np.arange(len(field))
    if n <= 0:
        raise ValueError("n must be positive")

    if method == "farthest_point":
        return _farthest_point(field.xy_mm, n)
    if method == "uniform":
        return _stratified_uniform(field, n, seed)
    raise ValueError(f"unknown downselect method {method!r}")


def _farthest_point(points: np.ndarray, n: int) -> np.ndarray:
    centroid = points.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))

    selected = [start]
    min_dist = np.linalg.norm(points - points[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(sorted(selected), dtype=int)


def _stratified_uniform(field: HeliostatField, n: int, seed: int) -> np.ndarray:
    """Split into radius rings of equal population, then spread over azimuth."""
    rng = np.random.default_rng(seed)
    r = field.radius_mm
    az = field.azimuth_deg
    order = np.argsort(r)

    n_rings = max(1, int(round(np.sqrt(n))))
    rings = np.array_split(order, n_rings)

    base, extra = divmod(n, n_rings)
    chosen: list[int] = []
    for i, ring in enumerate(rings):
        take = base + (1 if i < extra else 0)
        if take <= 0 or ring.size == 0:
            continue
        take = min(take, ring.size)
        # Spread the picks evenly in azimuth within this ring.
        ring_sorted = ring[np.argsort(az[ring])]
        offset = rng.integers(0, max(1, ring_sorted.size // take))
        picks = np.linspace(0, ring_sorted.size, take, endpoint=False).astype(int) + offset
        chosen.extend(ring_sorted[np.clip(picks, 0, ring_sorted.size - 1)].tolist())

    chosen = sorted(set(chosen))
    # Backfill if de-duplication lost any.
    if len(chosen) < n:
        remaining = [i for i in np.argsort(r) if i not in set(chosen)]
        chosen.extend(remaining[: n - len(chosen)])
    return np.array(sorted(chosen[:n]), dtype=int)


def neighbour_pairs(field: HeliostatField, search_radius_mm: float):
    """Neighbours within ``search_radius_mm`` of each heliostat.

    Returns a list of index arrays, one per heliostat, excluding itself. Used
    by :mod:`heliostat.geometry.shading` to limit shading/blocking tests to
    plausible occluders instead of the whole field.
    """
    tree = cKDTree(field.xy_mm)
    groups = tree.query_ball_point(field.xy_mm, r=search_radius_mm)
    return [np.array([j for j in g if j != i], dtype=int) for i, g in enumerate(groups)]
