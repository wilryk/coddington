"""2-D aperture sketches: composable mirror outlines.

A heliostat's reflective outline is a *sketch* — 2-D primitives in the
mirror's own (u, v) plane, millimetres, combined the way CAD sketches
are: transformed, unioned, intersected, subtracted, and patterned. A
five-petal "flower" mirror is::

    petal = disc(1200, at=(0, 700)) & disc(1200, at=(0, -700))
    flower = circular_array(petal.translated(0, 1500), n=5)

The optical engine only ever asks a region three questions — membership,
bounding box, area — so anything expressible here works unchanged in the
Monte Carlo tracer (ray-hit tests), the cone backend (surface sample
grids), and shading/blocking (occlusion point tests). Regions serialise
to plain dicts (:meth:`Region.to_dict` / :func:`region_from_dict`), so a
design is a data file people can share, and a future graphical sketcher
is a front end over the same tree.

Conventions: operators ``|`` (union), ``&`` (intersection), ``-``
(subtract); every ``contains`` accepts scalar or array ``u, v`` and
broadcasts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Region(ABC):
    """A 2-D point set in the sketch plane, mm."""

    kind: str = "abstract"

    @abstractmethod
    def contains(self, u, v) -> np.ndarray:
        """Boolean membership, broadcasting over ``u``/``v``."""

    @abstractmethod
    def bbox(self) -> tuple[float, float, float, float]:
        """``(u_min, u_max, v_min, v_max)`` bounding box, mm."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-safe description; ``region_from_dict`` inverts it."""

    def area_mm2(self, resolution: int = 512) -> float:
        """Numeric area: membership fraction of a supersampled bbox grid.

        Relative accuracy ~(perimeter * cell) / area — better than 1e-4
        for sane shapes at the default resolution. Primitives with exact
        areas override this.
        """
        u0, u1, v0, v1 = self.bbox()
        if u1 <= u0 or v1 <= v0:
            return 0.0
        uu = np.linspace(u0, u1, resolution + 1)
        vv = np.linspace(v0, v1, resolution + 1)
        um = 0.5 * (uu[:-1] + uu[1:])
        vm = 0.5 * (vv[:-1] + vv[1:])
        inside = self.contains(um[None, :], vm[:, None])
        return float(inside.mean() * (u1 - u0) * (v1 - v0))

    # -- sketch algebra ---------------------------------------------------

    def __or__(self, other: "Region") -> "Region":
        return Union([self, other])

    def __and__(self, other: "Region") -> "Region":
        return Intersection([self, other])

    def __sub__(self, other: "Region") -> "Region":
        return Difference(self, other)

    def translated(self, du: float, dv: float) -> "Region":
        return Translate(self, du, dv)

    def rotated(self, angle_deg: float) -> "Region":
        return Rotate(self, angle_deg)


# ---------------------------------------------------------------------------
# primitives


class Rect(Region):
    """Axis-aligned rectangle centred on the origin."""

    kind = "rect"

    def __init__(self, width_mm: float, height_mm: float):
        if width_mm <= 0 or height_mm <= 0:
            raise ValueError("rectangle dimensions must be positive")
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)

    def contains(self, u, v):
        return (np.abs(u) <= self.width_mm / 2.0) & (np.abs(v) <= self.height_mm / 2.0)

    def bbox(self):
        return (-self.width_mm / 2, self.width_mm / 2, -self.height_mm / 2, self.height_mm / 2)

    def area_mm2(self, resolution: int = 512) -> float:
        return self.width_mm * self.height_mm

    def to_dict(self):
        return {"kind": self.kind, "width_mm": self.width_mm, "height_mm": self.height_mm}


class Disc(Region):
    """Circle centred on the origin."""

    kind = "disc"

    def __init__(self, radius_mm: float):
        if radius_mm <= 0:
            raise ValueError("radius must be positive")
        self.radius_mm = float(radius_mm)

    def contains(self, u, v):
        return np.asarray(u) ** 2 + np.asarray(v) ** 2 <= self.radius_mm**2

    def bbox(self):
        r = self.radius_mm
        return (-r, r, -r, r)

    def area_mm2(self, resolution: int = 512) -> float:
        return float(np.pi * self.radius_mm**2)

    def to_dict(self):
        return {"kind": self.kind, "radius_mm": self.radius_mm}


class Ellipse(Region):
    """Axis-aligned ellipse centred on the origin."""

    kind = "ellipse"

    def __init__(self, semi_u_mm: float, semi_v_mm: float):
        if semi_u_mm <= 0 or semi_v_mm <= 0:
            raise ValueError("semi-axes must be positive")
        self.semi_u_mm = float(semi_u_mm)
        self.semi_v_mm = float(semi_v_mm)

    def contains(self, u, v):
        return (np.asarray(u) / self.semi_u_mm) ** 2 + (np.asarray(v) / self.semi_v_mm) ** 2 <= 1.0

    def bbox(self):
        return (-self.semi_u_mm, self.semi_u_mm, -self.semi_v_mm, self.semi_v_mm)

    def area_mm2(self, resolution: int = 512) -> float:
        return float(np.pi * self.semi_u_mm * self.semi_v_mm)

    def to_dict(self):
        return {"kind": self.kind, "semi_u_mm": self.semi_u_mm, "semi_v_mm": self.semi_v_mm}


class Annulus(Region):
    """Ring between two radii, centred on the origin."""

    kind = "annulus"

    def __init__(self, r_inner_mm: float, r_outer_mm: float):
        if not 0 <= r_inner_mm < r_outer_mm:
            raise ValueError("need 0 <= r_inner < r_outer")
        self.r_inner_mm = float(r_inner_mm)
        self.r_outer_mm = float(r_outer_mm)

    def contains(self, u, v):
        rr = np.asarray(u) ** 2 + np.asarray(v) ** 2
        return (rr >= self.r_inner_mm**2) & (rr <= self.r_outer_mm**2)

    def bbox(self):
        r = self.r_outer_mm
        return (-r, r, -r, r)

    def area_mm2(self, resolution: int = 512) -> float:
        return float(np.pi * (self.r_outer_mm**2 - self.r_inner_mm**2))

    def to_dict(self):
        return {"kind": self.kind, "r_inner_mm": self.r_inner_mm, "r_outer_mm": self.r_outer_mm}


class Polygon(Region):
    """Simple polygon (convex or not) from an ordered vertex list."""

    kind = "polygon"

    def __init__(self, vertices_mm):
        verts = np.asarray(vertices_mm, dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] != 2:
            raise ValueError("vertices must be (N>=3, 2)")
        self.vertices_mm = verts

    def contains(self, u, v):
        # Crossing-number test, vectorised over broadcast (u, v).
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        inside = np.zeros(u.shape, dtype=bool)
        x, y = self.vertices_mm[:, 0], self.vertices_mm[:, 1]
        xj, yj = np.roll(x, 1), np.roll(y, 1)
        for xi, yi, xk, yk in zip(x, y, xj, yj):
            crosses = (yi > v) != (yk > v)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = xi + (v - yi) * (xk - xi) / (yk - yi)
            inside ^= crosses & (u < x_at)
        return inside

    def bbox(self):
        x, y = self.vertices_mm[:, 0], self.vertices_mm[:, 1]
        return (float(x.min()), float(x.max()), float(y.min()), float(y.max()))

    def area_mm2(self, resolution: int = 512) -> float:
        x, y = self.vertices_mm[:, 0], self.vertices_mm[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def to_dict(self):
        return {"kind": self.kind, "vertices_mm": self.vertices_mm.tolist()}


def regular_polygon(n_sides: int, circumradius_mm: float, phase_deg: float = 90.0) -> Polygon:
    """Regular n-gon; the default phase puts a vertex straight up."""
    if n_sides < 3:
        raise ValueError("need at least 3 sides")
    ang = np.deg2rad(phase_deg) + 2.0 * np.pi * np.arange(n_sides) / n_sides
    return Polygon(circumradius_mm * np.column_stack([np.cos(ang), np.sin(ang)]))


# ---------------------------------------------------------------------------
# transforms


class Translate(Region):
    kind = "translate"

    def __init__(self, child: Region, du_mm: float, dv_mm: float):
        self.child = child
        self.du_mm = float(du_mm)
        self.dv_mm = float(dv_mm)

    def contains(self, u, v):
        return self.child.contains(np.asarray(u) - self.du_mm, np.asarray(v) - self.dv_mm)

    def bbox(self):
        u0, u1, v0, v1 = self.child.bbox()
        return (u0 + self.du_mm, u1 + self.du_mm, v0 + self.dv_mm, v1 + self.dv_mm)

    def to_dict(self):
        return {
            "kind": self.kind,
            "child": self.child.to_dict(),
            "du_mm": self.du_mm,
            "dv_mm": self.dv_mm,
        }


class Rotate(Region):
    """Rotate the child about the origin, counter-clockwise degrees."""

    kind = "rotate"

    def __init__(self, child: Region, angle_deg: float):
        self.child = child
        self.angle_deg = float(angle_deg)

    def contains(self, u, v):
        a = np.deg2rad(self.angle_deg)
        c, s = np.cos(a), np.sin(a)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        # Inverse rotation into the child's frame.
        return self.child.contains(c * u + s * v, -s * u + c * v)

    def bbox(self):
        u0, u1, v0, v1 = self.child.bbox()
        a = np.deg2rad(self.angle_deg)
        c, s = np.cos(a), np.sin(a)
        corners = np.array([[u0, v0], [u0, v1], [u1, v0], [u1, v1]])
        rot = corners @ np.array([[c, s], [-s, c]]).T
        return (
            float(rot[:, 0].min()),
            float(rot[:, 0].max()),
            float(rot[:, 1].min()),
            float(rot[:, 1].max()),
        )

    def to_dict(self):
        return {"kind": self.kind, "child": self.child.to_dict(), "angle_deg": self.angle_deg}


# ---------------------------------------------------------------------------
# booleans and patterns


class Union(Region):
    kind = "union"

    def __init__(self, children: list[Region]):
        if not children:
            raise ValueError("union of nothing")
        self.children = list(children)

    def contains(self, u, v):
        out = self.children[0].contains(u, v)
        for child in self.children[1:]:
            out = out | child.contains(u, v)
        return out

    def bbox(self):
        boxes = np.array([c.bbox() for c in self.children])
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].max()),
            float(boxes[:, 2].min()),
            float(boxes[:, 3].max()),
        )

    def to_dict(self):
        return {"kind": self.kind, "children": [c.to_dict() for c in self.children]}


class Intersection(Region):
    kind = "intersection"

    def __init__(self, children: list[Region]):
        if not children:
            raise ValueError("intersection of nothing")
        self.children = list(children)

    def contains(self, u, v):
        out = self.children[0].contains(u, v)
        for child in self.children[1:]:
            out = out & child.contains(u, v)
        return out

    def bbox(self):
        boxes = np.array([c.bbox() for c in self.children])
        return (
            float(boxes[:, 0].max()),
            float(boxes[:, 1].min()),
            float(boxes[:, 2].max()),
            float(boxes[:, 3].min()),
        )

    def to_dict(self):
        return {"kind": self.kind, "children": [c.to_dict() for c in self.children]}


class Difference(Region):
    """Points in ``base`` and not in ``cut``."""

    kind = "difference"

    def __init__(self, base: Region, cut: Region):
        self.base = base
        self.cut = cut

    def contains(self, u, v):
        return self.base.contains(u, v) & ~self.cut.contains(u, v)

    def bbox(self):
        return self.base.bbox()

    def to_dict(self):
        return {"kind": self.kind, "base": self.base.to_dict(), "cut": self.cut.to_dict()}


class CircularArray(Region):
    """The union of ``n`` copies of the child, rotated about the origin.

    The CAD circular sketch pattern: place one petal off-origin, pattern
    it ``n`` times, get a flower.
    """

    kind = "circular_array"

    def __init__(self, child: Region, n: int, phase_deg: float = 0.0):
        if n < 1:
            raise ValueError("need n >= 1 copies")
        self.child = child
        self.n = int(n)
        self.phase_deg = float(phase_deg)

    def _copies(self) -> list[Region]:
        step = 360.0 / self.n
        return [Rotate(self.child, self.phase_deg + i * step) for i in range(self.n)]

    def contains(self, u, v):
        out = None
        for copy in self._copies():
            hit = copy.contains(u, v)
            out = hit if out is None else (out | hit)
        return out

    def bbox(self):
        boxes = np.array([c.bbox() for c in self._copies()])
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].max()),
            float(boxes[:, 2].min()),
            float(boxes[:, 3].max()),
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "child": self.child.to_dict(),
            "n": self.n,
            "phase_deg": self.phase_deg,
        }


# ---------------------------------------------------------------------------
# serialisation

_REGISTRY: dict[str, type] = {
    cls.kind: cls
    for cls in (
        Rect,
        Disc,
        Ellipse,
        Annulus,
        Polygon,
        Translate,
        Rotate,
        Union,
        Intersection,
        Difference,
        CircularArray,
    )
}


def region_from_dict(data: dict) -> Region:
    """Rebuild a region tree from :meth:`Region.to_dict` output."""
    data = dict(data)
    kind = data.pop("kind")
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown region kind {kind!r}; known: {sorted(_REGISTRY)}")
    if kind in ("union", "intersection"):
        return cls([region_from_dict(c) for c in data["children"]])
    if kind == "difference":
        return cls(region_from_dict(data["base"]), region_from_dict(data["cut"]))
    if kind in ("translate", "rotate", "circular_array"):
        child = region_from_dict(data.pop("child"))
        return cls(child, **data)
    return cls(**data)


# ---------------------------------------------------------------------------
# convenience constructors, sketch-vocabulary style


def rect(width_mm: float, height_mm: float, at: tuple[float, float] = (0.0, 0.0)) -> Region:
    r: Region = Rect(width_mm, height_mm)
    return r.translated(*at) if at != (0.0, 0.0) else r


def disc(radius_mm: float, at: tuple[float, float] = (0.0, 0.0)) -> Region:
    r: Region = Disc(radius_mm)
    return r.translated(*at) if at != (0.0, 0.0) else r


def petal(length_mm: float, width_mm: float) -> Region:
    """A vesica-style petal: the lens of two overlapping discs.

    The petal points along +v, base at the origin, ``length_mm`` tall and
    ``width_mm`` across at its waist.
    """
    if width_mm >= 2.0 * length_mm:
        raise ValueError("petal width must be less than twice its length")
    half_l = length_mm / 2.0
    half_w = width_mm / 2.0
    # Circle through the petal's tip and base with sagitta half_w at the waist.
    r = (half_l**2 + half_w**2) / (2.0 * half_w)
    lens = Disc(r).translated(r - half_w, 0.0) & Disc(r).translated(-(r - half_w), 0.0)
    return lens.translated(0.0, half_l)


def circular_array(child: Region, n: int, phase_deg: float = 0.0) -> Region:
    return CircularArray(child, n, phase_deg)
