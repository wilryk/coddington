"""Receiver surfaces and their flux parameterizations.

A receiver is the absorbing surface flux maps are recorded on. Every
receiver exposes the same contract so the tracer, the run store, and the
plotting code never branch on receiver type:

* :meth:`Receiver.intersect` — where rays meet the (unclipped) surface, in
  the receiver's own ``(u, v)`` surface coordinates, millimetres;
* :meth:`Receiver.uv_extent` — the finite window of that parameterization
  that counts as "on the receiver". Hitting the surface and landing inside
  the extent are deliberately separate tests, mirroring the tracer's
  ``reached_receiver`` / ``in_window`` loss counters;
* :meth:`Receiver.bin_edges` / :meth:`Receiver.bin_areas_m2` — the flux
  grid. Bin areas are returned as an array because they are *not* uniform
  for every shape (a frustum's bins shrink toward its narrow end); flux in
  W/m² must always divide counts by the per-bin area, never by a scalar.
* :meth:`Receiver.aim_point_mm` — the default aim point a heliostat at a
  given field position should target.

Ray inputs follow the tracer's convention: positions ``p`` and unit
directions ``d`` as ``(3, N)`` arrays in field coordinates (x east, y
north, z up, millimetres).

Curved surfaces are parameterized by *unrolled arc length*, not angle, so a
``(u, v)`` rectangle has true physical dimensions and a flux map of a
cylinder reads like a developed drawing of its shell. The azimuthal seam is
placed at the +y (north, tower-shadow) azimuth where flux is lowest, so
real spots never straddle the wrap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Receiver(ABC):
    """Common contract for absorbing receiver surfaces."""

    kind: str = "abstract"

    #: Whether the absorbing surface is a plane. The cone backend's
    #: second-order deposit differentiates the ray-to-surface map twice and
    #: assumes that map is well behaved; on a curved surface it can fold,
    #: which sends the deposit's Jacobian through zero and the flux through
    #: the roof. Curved receivers therefore take the first-order deposit.
    is_planar: bool = True

    @abstractmethod
    def intersect(self, p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Intersect rays with the unclipped surface.

        :param p: ``(3, N)`` ray origins, mm.
        :param d: ``(3, N)`` unit ray directions.
        :returns: ``(hit, uv)`` — ``hit`` is an ``(N,)`` bool mask of rays
            that meet the surface travelling toward its absorbing side at
            positive path length; ``uv`` is ``(2, K)`` surface coordinates
            (mm) for the ``K`` hits, in ray order. ``uv`` may lie outside
            :meth:`uv_extent`; callers clip with the extent to decide what
            is "on" the receiver.
        """

    @abstractmethod
    def uv_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((u_min, u_max), (v_min, v_max))`` of the absorbing window, mm."""

    @abstractmethod
    def aim_point_mm(self, helio_xy_mm: np.ndarray) -> np.ndarray:
        """Default aim point for a heliostat at field position ``(x, y)`` mm.

        :param helio_xy_mm: ``(2,)`` or ``(2, N)`` heliostat centre(s).
        :returns: ``(3,)`` or ``(3, N)`` aim point(s), mm.
        """

    def bin_edges(self, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Uniform ``(u_edges, v_edges)`` spanning :meth:`uv_extent`.

        ``grid`` is ``(n_u, n_v)``; the returned arrays have ``n_u + 1`` and
        ``n_v + 1`` entries.
        """
        (u0, u1), (v0, v1) = self.uv_extent()
        n_u, n_v = grid
        return np.linspace(u0, u1, n_u + 1), np.linspace(v0, v1, n_v + 1)

    def bin_areas_m2(self, grid: tuple[int, int]) -> np.ndarray:
        """True surface area of each flux bin, ``(n_v, n_u)`` in m².

        The base implementation is exact for any surface whose area element
        is independent of position in ``(u, v)`` — true for planes and
        cylinders parameterized by arc length. Shapes with position-
        dependent area elements (the frustum) override this.
        """
        (u0, u1), (v0, v1) = self.uv_extent()
        n_u, n_v = grid
        cell = ((u1 - u0) / n_u / 1000.0) * ((v1 - v0) / n_v / 1000.0)
        return np.full((n_v, n_u), cell)

    def to_manifest(self) -> dict:
        """JSON-safe description for the run manifest."""
        out = {"kind": self.kind}
        out.update({k: float(v) for k, v in vars(self).items()})
        return out

    @staticmethod
    def from_manifest(entry: dict) -> "Receiver":
        """Rebuild a receiver from :meth:`to_manifest` output."""
        entry = dict(entry)
        kind = entry.pop("kind")
        cls = _REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"unknown receiver kind {kind!r}; known: {sorted(_REGISTRY)}")
        return cls(**entry)


@dataclass
class FlatWindowReceiver(Receiver):
    """Horizontal rectangular window at height ``z_mm``.

    ``facing="up"`` absorbs rays arriving from above (a ground receiver
    below a tower reflector); ``facing="down"`` absorbs rays arriving from
    below (a prime-focus receiver at the top of a tower). ``uv`` is simply
    ``(x, y)`` at the plane.
    """

    z_mm: float
    half_u_mm: float
    half_v_mm: float
    facing: str = "up"
    #: World (x, y) this window is centred on; ``uv`` is world position minus
    #: this centre, so a receiver on-axis (the default) still reports raw
    #: world coordinates exactly as before.
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0

    kind = "flat"

    def __post_init__(self) -> None:
        if self.facing not in ("up", "down"):
            raise ValueError(f"facing must be 'up' or 'down', got {self.facing!r}")

    def intersect(self, p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dz = d[2]
        approach = dz < 0 if self.facing == "up" else dz > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.z_mm - p[2]) / dz
        hit = approach & np.isfinite(t) & (t > 0)
        uv = p[:2, hit] + t[hit] * d[:2, hit]
        uv[0] -= self.center_x_mm
        uv[1] -= self.center_y_mm
        return hit, uv

    def uv_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (-self.half_u_mm, self.half_u_mm), (-self.half_v_mm, self.half_v_mm)

    def aim_point_mm(self, helio_xy_mm: np.ndarray) -> np.ndarray:
        xy = np.asarray(helio_xy_mm, dtype=float)
        shape = (3,) if xy.ndim == 1 else (3, xy.shape[1])
        aim = np.zeros(shape)
        aim[0] = self.center_x_mm
        aim[1] = self.center_y_mm
        aim[2] = self.z_mm
        return aim

    def to_manifest(self) -> dict:
        return {
            "kind": self.kind,
            "z_mm": self.z_mm,
            "half_u_mm": self.half_u_mm,
            "half_v_mm": self.half_v_mm,
            "facing": self.facing,
            "center_x_mm": self.center_x_mm,
            "center_y_mm": self.center_y_mm,
        }


@dataclass
class CylinderReceiver(Receiver):
    """External vertical cylinder — the conventional tower receiver.

    Absorbs on its outer surface. ``u`` is unrolled azimuthal arc length
    ``R * wrap(azimuth - pi)`` with the seam at the +y (north) azimuth;
    ``v`` is height above the cylinder's mid-plane. Rays are accepted on
    the *near-side exterior*: the smaller positive root of the quadratic,
    travelling inward (``d . n_outward < 0``).
    """

    center_z_mm: float
    radius_mm: float
    height_mm: float
    #: World (x, y) of the cylinder's axis; 0, 0 is the tower axis, the
    #: only centre this shape ever used before it became positionable.
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0

    kind = "cylinder"
    is_planar = False

    def intersect(self, p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, py = p[0] - self.center_x_mm, p[1] - self.center_y_mm
        dx, dy = d[0], d[1]
        a = dx * dx + dy * dy
        b = 2.0 * (px * dx + py * dy)
        c = px * px + py * py - self.radius_mm**2
        disc = b * b - 4.0 * a * c
        ok = (disc >= 0) & (a > 0)

        sq = np.sqrt(np.where(ok, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_near = (-b - sq) / (2.0 * a)
        # The near root is the exterior hit; at t_near the outward normal is
        # (x, y)/R and d . n < 0 holds automatically for a ray whose origin
        # is outside (c > 0) and that reaches the surface. Rays starting
        # inside the cylinder (c < 0) are rejected — nothing in a heliostat
        # field emits from inside the receiver.
        hit = ok & (t_near > 0) & (c > 0)

        x = px[hit] + t_near[hit] * dx[hit]
        y = py[hit] + t_near[hit] * dy[hit]
        z = p[2, hit] + t_near[hit] * d[2, hit]
        # Azimuth measured from -y (south) so the wrap seam sits at +y
        # (north), behind the tower where flux is lowest.
        az = np.arctan2(x, -y)  # -pi..pi, 0 at south, seam at north
        u = self.radius_mm * az
        v = z - self.center_z_mm
        return hit, np.vstack((u, v))

    def uv_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half_circ = np.pi * self.radius_mm
        return (-half_circ, half_circ), (-self.height_mm / 2.0, self.height_mm / 2.0)

    def aim_point_mm(self, helio_xy_mm: np.ndarray) -> np.ndarray:
        """Aim at the surface generatrix facing the heliostat, mid-height.

        Aiming at the axis instead would land rays off-centre on the near
        surface and systematically under-fill the panel width. "Facing" is
        measured from this cylinder's own centre, not the field origin, so
        an off-axis receiver (:attr:`center_x_mm`/:attr:`center_y_mm`) still
        gets the correct per-heliostat surface point.
        """
        xy = np.asarray(helio_xy_mm, dtype=float)
        centre = np.array([self.center_x_mm, self.center_y_mm])
        rel = xy - (centre if xy.ndim == 1 else centre[:, None])
        norm = np.linalg.norm(rel, axis=0)
        norm = np.where(norm == 0, 1.0, norm)
        toward = rel / norm
        aim = np.empty((3,) if xy.ndim == 1 else (3, xy.shape[1]))
        aim[0] = self.center_x_mm + self.radius_mm * toward[0]
        aim[1] = self.center_y_mm + self.radius_mm * toward[1]
        aim[2] = self.center_z_mm
        return aim


@dataclass
class FrustumReceiver(Receiver):
    """Truncated-cone receiver, absorbing on its outer lateral surface.

    ``r_top_mm > r_bot_mm`` gives the "upside-down" (inverted) frustum. The
    surface is parameterized by ``v`` — slant distance from the bottom rim,
    ``0..slant_length`` — and ``u`` — unrolled azimuthal arc length *at the
    mean radius*, seam at +y. Because a circle of latitude at slant
    position ``v`` has radius ``r(v) != r_mean``, the true area of a bin
    scales with ``r(v) / r_mean``: :meth:`bin_areas_m2` is overridden and
    varies row by row.
    """

    z_bot_mm: float
    r_bot_mm: float
    z_top_mm: float
    r_top_mm: float
    #: World (x, y) of the frustum's axis; 0, 0 is the tower axis.
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0

    kind = "frustum"
    is_planar = False

    def __post_init__(self) -> None:
        if self.z_top_mm <= self.z_bot_mm:
            raise ValueError("z_top_mm must exceed z_bot_mm")
        if self.r_top_mm == self.r_bot_mm:
            raise ValueError("equal radii is a cylinder; use CylinderReceiver")

    @property
    def _slope(self) -> float:
        """dr/dz along the wall."""
        return (self.r_top_mm - self.r_bot_mm) / (self.z_top_mm - self.z_bot_mm)

    @property
    def slant_length_mm(self) -> float:
        dz = self.z_top_mm - self.z_bot_mm
        dr = self.r_top_mm - self.r_bot_mm
        return float(np.hypot(dz, dr))

    @property
    def r_mean_mm(self) -> float:
        return 0.5 * (self.r_top_mm + self.r_bot_mm)

    def _apex(self) -> tuple[float, float]:
        """(z_apex, k) of the full cone ``r = k * (z - z_apex)``."""
        m = self._slope
        z_apex = self.z_bot_mm - self.r_bot_mm / m
        return z_apex, m

    def intersect(self, p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_apex, m = self._apex()
        px, py, pz = p[0] - self.center_x_mm, p[1] - self.center_y_mm, p[2] - z_apex
        dx, dy, dz = d[0], d[1], d[2]
        m2 = m * m
        a = dx * dx + dy * dy - m2 * dz * dz
        b = 2.0 * (px * dx + py * dy - m2 * pz * dz)
        c = px * px + py * py - m2 * pz * pz
        disc = b * b - 4.0 * a * c

        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)
        # Exterior (outside, c > 0) rays: smallest positive root on the
        # correct nappe and travelling inward.
        t = np.where((t1 > 0) & ok, t1, np.where((t2 > 0) & ok, t2, np.nan))
        hit = ok & np.isfinite(t) & (c > 0)

        x = px[hit] + t[hit] * dx[hit]
        y = py[hit] + t[hit] * dy[hit]
        zc = pz[hit] + t[hit] * dz[hit]  # z above apex
        # Reject the wrong nappe (below the apex the mirror cone opens the
        # other way) — the physical wall satisfies sign(zc) == sign(m r).
        nappe_ok = (zc * m) > 0
        idx = np.flatnonzero(hit)
        hit[idx[~nappe_ok]] = False
        x, y, zc = x[nappe_ok], y[nappe_ok], zc[nappe_ok]

        z_field = zc + z_apex
        # Slant coordinate from the bottom rim; azimuth arc at mean radius.
        v = (z_field - self.z_bot_mm) / (self.z_top_mm - self.z_bot_mm) * self.slant_length_mm
        az = np.arctan2(x, -y)
        u = self.r_mean_mm * az
        return hit, np.vstack((u, v))

    def uv_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        half_circ = np.pi * self.r_mean_mm
        return (-half_circ, half_circ), (0.0, self.slant_length_mm)

    def bin_areas_m2(self, grid: tuple[int, int]) -> np.ndarray:
        """Per-bin area, larger toward the wide end: scales with r(v)/r_mean."""
        n_u, n_v = grid
        _, v_edges = self.bin_edges(grid)
        v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
        frac = v_mid / self.slant_length_mm
        r_mid = self.r_bot_mm + frac * (self.r_top_mm - self.r_bot_mm)
        (u0, u1), _ = self.uv_extent()
        du = (u1 - u0) / n_u
        dv = v_edges[1] - v_edges[0]
        row = (r_mid / self.r_mean_mm) * du * dv / 1.0e6
        return np.repeat(row[:, None], n_u, axis=1)

    def aim_point_mm(self, helio_xy_mm: np.ndarray) -> np.ndarray:
        """Aim at the facing generatrix at mid-slant, relative to this
        frustum's own centre (see :class:`CylinderReceiver`'s identical
        note)."""
        xy = np.asarray(helio_xy_mm, dtype=float)
        centre = np.array([self.center_x_mm, self.center_y_mm])
        rel = xy - (centre if xy.ndim == 1 else centre[:, None])
        norm = np.linalg.norm(rel, axis=0)
        norm = np.where(norm == 0, 1.0, norm)
        toward = rel / norm
        aim = np.empty((3,) if xy.ndim == 1 else (3, xy.shape[1]))
        aim[0] = self.center_x_mm + self.r_mean_mm * toward[0]
        aim[1] = self.center_y_mm + self.r_mean_mm * toward[1]
        aim[2] = 0.5 * (self.z_bot_mm + self.z_top_mm)
        return aim


@dataclass
class ApertureClippedReceiver(Receiver):
    """A flat entrance opening in front of the actual absorbing surface.

    Models a cavity receiver: ``aperture`` is a small flat window a beam
    must pass through before it can reach ``inner`` (any shape, including
    another :class:`FlatWindowReceiver`), which sits behind it and does the
    actual absorbing. A ray is only a hit if it clears the aperture's own
    window *and* then meets ``inner`` -- ``uv``/:meth:`uv_extent` always
    describe ``inner``, never the aperture, since that is what the flux map
    is drawn against.

    Not part of :meth:`Receiver.from_manifest`'s ``_REGISTRY`` -- nothing in
    this codebase round-trips one through a run manifest today (the web app
    rebuilds it fresh from ``optics_params`` on every request), so
    :meth:`to_manifest` is for inspection only.
    """

    aperture: FlatWindowReceiver
    inner: Receiver

    kind = "aperture_clipped"

    @property
    def is_planar(self) -> bool:
        """Whatever the surface behind the aperture is -- uv describes it."""
        return self.inner.is_planar

    def intersect(self, p: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ap_hit, ap_uv = self.aperture.intersect(p, d)
        (au0, au1), (av0, av1) = self.aperture.uv_extent()
        ap_inside = (ap_uv[0] >= au0) & (ap_uv[0] <= au1) & (ap_uv[1] >= av0) & (ap_uv[1] <= av1)
        # ap_hit already indexes into the original N rays; ap_uv/ap_inside
        # are already filtered to ap_hit's survivors (the same contract
        # Receiver.intersect documents), so this recovers the original
        # indices of the rays that cleared the aperture opening.
        cleared = np.flatnonzero(ap_hit)[ap_inside]
        in_hit, in_uv = self.inner.intersect(p[:, cleared], d[:, cleared])
        hit = np.zeros(p.shape[1], dtype=bool)
        hit[cleared[in_hit]] = True
        return hit, in_uv

    def uv_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.inner.uv_extent()

    def aim_point_mm(self, helio_xy_mm: np.ndarray) -> np.ndarray:
        return self.inner.aim_point_mm(helio_xy_mm)

    def bin_edges(self, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        return self.inner.bin_edges(grid)

    def bin_areas_m2(self, grid: tuple[int, int]) -> np.ndarray:
        return self.inner.bin_areas_m2(grid)

    def to_manifest(self) -> dict:
        return {"kind": self.kind, "aperture": self.aperture.to_manifest(), "inner": self.inner.to_manifest()}


_REGISTRY: dict[str, type] = {
    FlatWindowReceiver.kind: FlatWindowReceiver,
    CylinderReceiver.kind: CylinderReceiver,
    FrustumReceiver.kind: FrustumReceiver,
}
