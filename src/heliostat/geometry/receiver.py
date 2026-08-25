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
placed at the +y (north, tower-shadow) azimuth. A heliostat can legitimately
be aimed straight at that seam (radial aiming puts north-sector heliostats
exactly there), so :func:`_continuous_azimuth` below is what keeps a bundle
of rays cast from one point continuous across it -- see its docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _continuous_azimuth(
    x: np.ndarray, y: np.ndarray, ox: np.ndarray, oy: np.ndarray, oz: np.ndarray
) -> np.ndarray:
    """Azimuth of ``(x, y)`` from -y (south), continuous within each group
    of rays cast from the same origin ``(ox, oy, oz)``.

    Plain ``arctan2`` has a branch cut at +y (north): two rays landing on
    either side of it get azimuths ~2*pi apart even though the physical
    points are next to each other. That is invisible to a single ray, but
    :mod:`heliostat.trace.cone` measures its Jacobian ``d(uv)/d(angle)`` by
    finite-differencing this value between a handful of rays cast from one
    mirror point at a tiny angular offset -- for a stencil whose true
    azimuth sits near the seam, the spurious ~2*pi jump makes the measured
    Jacobian (and the flux/power it deposits) come out ~10^5 too large.

    Fix: unwrap each group of same-origin rays (a stencil's legs, a
    transmission raster's node fan) relative to its own circular mean, so
    members that genuinely straddle the seam still report mutually
    continuous values -- exactly what a caller differencing them needs.
    Rays with distinct origins (the common case: one physical ray each, as
    every caller outside the cone backend's finite-difference stencils
    sends) form singleton groups and get back exactly ``arctan2``'s own
    value, unchanged. A group whose own angular spread reaches all the way
    to the seam can still report a value just past the nominal +-pi*R
    window by up to that spread -- the unavoidable residual of any single
    branch cut on a closed surface, and bounded by one sample's own
    footprint rather than a full circumference.
    """
    az = np.arctan2(x, -y)
    n = az.size
    if n == 0:
        return az
    # Group rays by exact-match origin (cheap and exact: same-origin rays
    # reach here with bit-identical floats, since they are literally the
    # same point repeated by the caller, offset by the same receiver-frame
    # constants). Sort so identical rows become adjacent, then mark where a
    # new row starts.
    order = np.lexsort((oz, oy, ox))
    sx, sy, sz = ox[order], oy[order], oz[order]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    if n > 1:
        new_group[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1]) | (sz[1:] != sz[:-1])
    group_of_sorted = np.cumsum(new_group) - 1
    n_groups = int(group_of_sorted[-1]) + 1
    if n_groups == n:
        return az  # every ray its own group -- nothing to unwrap
    group = np.empty(n, dtype=np.intp)
    group[order] = group_of_sorted
    sin_sum = np.bincount(group, weights=np.sin(az), minlength=n_groups)
    cos_sum = np.bincount(group, weights=np.cos(az), minlength=n_groups)
    ref = np.arctan2(sin_sum, cos_sum)[group]
    return az - 2.0 * np.pi * np.round((az - ref) / (2.0 * np.pi))


class Receiver(ABC):
    """Common contract for absorbing receiver surfaces."""

    kind: str = "abstract"

    #: Width of one full turn of ``u``, mm, for a surface that closes on
    #: itself; ``None`` for one that does not. A cylinder's ``u`` is arc
    #: length around it, so a landing point at +pi and one at -pi are the
    #: same place, and any test against the window has to say so.
    u_period_mm: float | None = None

    #: Whether the absorbing surface is a plane. The cone backend's
    #: second-order deposit differentiates the ray-to-surface map twice and
    #: assumes that map is well behaved; on a curved surface it can fold,
    #: which sends the deposit's Jacobian through zero and the flux through
    #: the roof. A conservation cap in ``trace/kernels.py`` keeps the
    #: resulting TOTAL power correct, but peak flux at a fold is still
    #: unbounded -- curved receivers should take the first-order deposit
    #: instead. Nothing in this package currently reads this flag; the
    #: guard that would (``order = 1`` when ``not receiver.is_planar``,
    #: before the stencil is built) lives in ``trace/cone.py``, which this
    #: module does not own.
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


    @property
    def u_period_mm(self) -> float:
        (u0, u1), _ = self.uv_extent()
        return u1 - u0

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
        # (north); see _continuous_azimuth for why it isn't a plain arctan2.
        az = _continuous_azimuth(x, y, p[0, hit], p[1, hit], p[2, hit])
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


    @property
    def u_period_mm(self) -> float:
        (u0, u1), _ = self.uv_extent()
        return u1 - u0

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

        # ``c`` (the quadratic's constant term) is F = r^2 - (m*(z-z_apex))^2
        # evaluated AT THE RAY ORIGIN: positive when the origin sits outside
        # the infinite double cone's local silhouette, negative when inside
        # it -- true of most near heliostats under a frustum that only
        # flares out well above the ground, even though nothing physical
        # occupies that cone below the finite band. The old code used this
        # sign to reject such rays outright; a ray from an origin like that
        # can still validly reach the real (finite) band, so it is used here
        # only to pick which crossing TYPE is the physical one, never to
        # reject a ray outright.
        #
        # Along any ray, F is a quadratic in t and its derivative at a root
        # is +-sqrt(disc): unconditionally negative at t1 (F falling through
        # zero -- an ENTERING crossing) and unconditionally positive at t2
        # (F rising through zero -- an EXITING crossing), regardless of the
        # cone's opening direction. A ray that starts outside (F(0) > 0) can
        # only first meet the surface by entering it -- t1; one that starts
        # inside (F(0) <= 0, the near-heliostat case) can only first meet it
        # by exiting -- t2. Either way, the crossing found is on the correct
        # (outward-facing, absorbing) side of the thin shell by construction,
        # once combined with the nappe test below.
        origin_outside = c > 0

        def _root(t: np.ndarray, needs_origin_outside: bool) -> tuple[np.ndarray, ...]:
            x = px + t * dx
            y = py + t * dy
            zc = pz + t * dz
            # Correct nappe: below the apex the mirror cone opens the other
            # way, so the physical wall satisfies sign(zc) == sign(m * r).
            nappe_ok = (zc * m) > 0
            side_ok = origin_outside if needs_origin_outside else ~origin_outside
            valid = ok & np.isfinite(t) & (t > 0) & nappe_ok & side_ok
            return valid, x, y, zc

        v1, x1, y1, zc1 = _root(t1, needs_origin_outside=True)
        v2, x2, y2, zc2 = _root(t2, needs_origin_outside=False)
        # origin_outside partitions every ray between the two branches, so
        # v1 and v2 never both hold for the same ray -- no tie-break needed.
        hit = v1 | v2
        x = np.where(v1, x1, x2)[hit]
        y = np.where(v1, y1, y2)[hit]
        zc = np.where(v1, zc1, zc2)[hit]

        z_field = zc + z_apex
        # Slant coordinate from the bottom rim; azimuth arc at mean radius,
        # continuous across the +y seam (see _continuous_azimuth).
        v = (z_field - self.z_bot_mm) / (self.z_top_mm - self.z_bot_mm) * self.slant_length_mm
        az = _continuous_azimuth(x, y, p[0, hit], p[1, hit], p[2, hit])
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
