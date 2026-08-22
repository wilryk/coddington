"""Secondary reflectors between the primary mirrors and the receiver.

A secondary-mirror concentrator reflects sunlight twice: once off each
heliostat's own mirror, and once off a shared reflector near the top of the
tower that redirects the combined beam onto a ground-level receiver below.
This module models that second bounce for three secondary shapes
plus the trivial case of having none at all (a plain tower with a receiver
mounted directly at the primary focus).

Every :class:`Secondary` consumes the rays leaving the primary mirrors —
already reflected once — and returns the rays leaving its own surface,
together with a mask of which incoming rays actually struck it. Rays that
miss are simply absent from the receiver; nothing downstream needs to know
why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Secondary(ABC):
    """Common contract for the optics sitting above the primary mirrors."""

    @abstractmethod
    def redirect(
        self, p: np.ndarray, d: np.ndarray, counters: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reflect rays off this secondary.

        :param p: ``(3, N)`` ray origins (the primary-mirror hit points), mm.
        :param d: ``(3, N)`` unit ray directions after the primary
            reflection.
        :param counters: the trace's loss-chain counter dict, updated in
            place. Every implementation sets ``hit_secondary`` (rays that
            struck the surface) and ``tip_rays`` (``0`` for shapes with no
            rounded-tip treatment, so the key is always present).
        :returns: ``(p2, d2, on_secondary)`` — ``p2`` and ``d2`` are
            ``(3, K)`` for the ``K = on_secondary.sum()`` rays that struck
            the surface, reflected; ``on_secondary`` is an ``(N,)`` boolean
            mask into the input arrays.
        """


class NoSecondary(Secondary):
    """Identity passthrough: the plain-tower case with no secondary at all.

    Every primary-reflected ray flies straight on to the receiver. There is
    nothing here to miss, so every ray counts as a hit — recorded explicitly
    rather than by omitting the counter keys, so a caller summing the loss
    chain never has to special-case "no secondary".
    """

    def redirect(self, p, d, counters):
        n = p.shape[1]
        counters["tip_rays"] = 0
        counters["hit_secondary"] = int(n)
        return p, d, np.ones(n, dtype=bool)


def _axicon_tip_geometry(
    tip_model: str, tip_radius_mm: float, alpha_rad: float, apex_height_mm: float
) -> tuple[float, float, float]:
    """``(tangency ring radius, sphere centre z, sphere radius)`` for a tip model.

    A literal cone has an infinitely sharp apex, which no manufactured part
    has and which produces an unphysical concentration of rays at the axis.
    The tip is instead blended into a sphere that meets the cone flank
    tangentially, replacing the cone inside the ring where the two surfaces
    touch:

    * ``sphere_tangent`` — sphere of radius ``tip_radius_mm`` tangent to the
      flank; tangency at radius ``tip_radius_mm * sin(alpha)``, centred on
      the axis at ``apex_height_mm + tip_radius_mm / cos(alpha)``.
    * ``sphere_curvature`` — ``tip_radius_mm`` read as the radius of
      curvature at the vertex of a cap that still meets the flank
      tangentially; geometrically the same sphere as ``sphere_tangent``.
    * ``parabola`` — a parabolic cap approximated by its osculating sphere
      (radius ``tip_radius_mm``) out to the same tangency ring; differs from
      ``sphere_tangent`` only in where that ring sits.
    * ``none`` — sharp cone, no blend.
    """
    if tip_model == "none":
        return 0.0, 0.0, 0.0
    if tip_model in ("sphere_tangent", "sphere_curvature"):
        r = tip_radius_mm
        return r * np.sin(alpha_rad), apex_height_mm + r / np.cos(alpha_rad), r
    if tip_model == "parabola":
        r = tip_radius_mm
        return r * np.tan(alpha_rad), apex_height_mm + r / np.cos(alpha_rad), r
    raise ValueError(f"unknown tip_model {tip_model!r}")


@dataclass(frozen=True)
class AxiconSecondary(Secondary):
    """Cone secondary, apex up, reflecting beams off its underside.

    The surface is ``z = apex_height_mm + h * tan(half_angle_deg)`` for
    radial distance ``h`` from the axis, clipped to ``aperture_radius_mm``.
    Rays inside the tangency ring hit the rounded tip instead (see
    :func:`_axicon_tip_geometry`); they stay counted in ``tip_rays`` rather
    than being dropped, since they are still real hits on the physical part.
    """

    apex_height_mm: float
    half_angle_deg: float
    aperture_radius_mm: float
    tip_radius_mm: float = 500.0
    tip_model: str = "sphere_tangent"

    def redirect(self, p, d, counters):
        z0 = self.apex_height_mm
        k = np.tan(np.deg2rad(self.half_angle_deg))
        px, py, pz = p[0], p[1], p[2] - z0
        dx, dy, dz = d[0], d[1], d[2]

        a = k * k * (dx * dx + dy * dy) - dz * dz
        b = 2.0 * (k * k * (px * dx + py * dy) - pz * dz)
        c = k * k * (px * px + py * py) - pz * pz
        disc = b * b - 4.0 * a * c
        valid = disc >= 0
        sq = np.sqrt(np.where(valid, disc, 0.0))
        cone_t = np.full(px.shape, np.inf)
        for root in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
            z_at = pz + root * dz
            cand = valid & (root > 1e-6) & (z_at >= 0.0) & (root < cone_t)
            cone_t = np.where(cand, root, cone_t)
        hx = px + cone_t * dx
        hy = py + cone_t * dy
        h = np.hypot(hx, hy)
        on_cone = np.isfinite(cone_t) & (h <= self.aperture_radius_mm)
        counters["tip_rays"] = int((on_cone & (h < self.tip_radius_mm)).sum())
        counters["hit_cone"] = counters["hit_secondary"] = int(on_cone.sum())

        h, cone_t = h[on_cone], cone_t[on_cone]
        d = d[:, on_cone]
        cone_hit = p[:, on_cone] + cone_t * d

        # Surface normal from grad(z - z0 - k*h); sign is irrelevant to
        # reflection.
        with np.errstate(invalid="ignore", divide="ignore"):
            nx, ny = -k * cone_hit[0] / h, -k * cone_hit[1] / h
        cn = np.vstack([nx, ny, np.ones_like(h)])
        cn /= np.linalg.norm(cn, axis=0)

        alpha = np.deg2rad(self.half_angle_deg)
        h_t, centre_z, radius = _axicon_tip_geometry(self.tip_model, self.tip_radius_mm, alpha, z0)
        tip = h < h_t
        if tip.any() and radius > 0:
            centre = np.array([0.0, 0.0, centre_z])
            p0 = p[:, on_cone][:, tip] - centre[:, None]
            dt_ = d[:, tip]
            b2 = (p0 * dt_).sum(axis=0)
            c2 = (p0 * p0).sum(axis=0) - radius**2
            disc2 = b2 * b2 - c2
            good = disc2 >= 0
            troot = -b2 - np.sqrt(np.where(good, disc2, 0.0))  # first (underside) hit
            sph_hit = p0 + dt_ * troot
            replace = good & (troot > 1e-6)
            sn = sph_hit / radius  # unit normal
            idx = np.where(tip)[0][replace]
            cone_hit[:, idx] = sph_hit[:, replace] + centre[:, None]
            cn[:, idx] = sn[:, replace]

        dot = np.einsum("ij,ij->j", d, cn)
        dot *= 2.0
        d = d - dot * cn
        return cone_hit, d, on_cone


def solve_cassegrain_relay(
    vertex_z_mm: float, focus_height_mm: float, receiver_z_mm: float
) -> tuple[float, float]:
    """Vertex radius and conic constant of the hyperboloid joining two foci.

    A hyperboloid mirror images one focus onto the other, which is the whole
    job of a Cassegrain relay: it takes the beam converging on the primary
    focus ``focus_height_mm`` and re-images it at ``receiver_z_mm``. Placing
    the vertex fixes the rest, because for a conic of vertex radius ``R`` and
    conic constant ``k = -e**2`` the two foci sit at ``R / (1 + e)`` and
    ``R / (1 - e)`` from the vertex along the axis. Inverting that pair:

    ``e = (d2 - d1) / (d1 + d2)``, ``R = d1 (1 + e)``

    with ``d1``/``d2`` the signed vertex-to-focus distances.

    This is what makes the relay adjustable rather than a fixed triple of
    magic constants: change any of the three heights and the surface that
    serves them is solved, not looked up. Round-trips this package's own
    fixture relay to better than 1e-6.

    :returns: ``(vertex_radius_mm, conic)``.
    :raises ValueError: if the three heights do not describe a hyperboloid --
        the primary focus must sit above the vertex, and the receiver below
        it by more than the focus sits above, or the surface that would
        join them is not a hyperbola.
    """
    d1 = float(focus_height_mm) - float(vertex_z_mm)
    d2 = float(receiver_z_mm) - float(vertex_z_mm)
    if d1 <= 0.0:
        raise ValueError(
            "the primary focus must be above the secondary vertex "
            f"(focus {focus_height_mm:g} mm, vertex {vertex_z_mm:g} mm)"
        )
    if d2 >= 0.0:
        raise ValueError(
            "the receiver must be below the secondary vertex "
            f"(receiver {receiver_z_mm:g} mm, vertex {vertex_z_mm:g} mm)"
        )
    if abs(d2) <= d1:
        raise ValueError(
            "the receiver must be further below the vertex than the primary "
            f"focus is above it (focus is {d1:g} mm above, receiver "
            f"{abs(d2):g} mm below); otherwise the relay is not a hyperboloid"
        )
    eccentricity = (d2 - d1) / (d1 + d2)
    return float(d1 * (1.0 + eccentricity)), float(-eccentricity * eccentricity)


@dataclass(frozen=True)
class CassegrainSecondary(Secondary):
    """Hyperboloid relay secondary, relaying the primary focus to the receiver.

    Surface: ``x^2 + y^2 = 2 R zeta - (1 + k) zeta^2`` with ``zeta = z -
    vertex_z_mm``, ``R = vertex_radius_mm`` (opens toward the primary
    focus), ``k = conic`` (``< -1`` for a hyperboloid). The physical branch
    is the smallest positive path length landing inside the aperture on the
    near nappe.
    """

    vertex_z_mm: float
    vertex_radius_mm: float
    conic: float
    aperture_radius_mm: float

    @property
    def rim_z_mm(self) -> float:
        h2 = self.aperture_radius_mm**2
        r, k = self.vertex_radius_mm, self.conic
        return self.vertex_z_mm + h2 / (r * (1.0 + np.sqrt(1.0 - (1.0 + k) * h2 / r**2)))

    def redirect(self, p, d, counters):
        r = self.vertex_radius_mm
        kk = 1.0 + self.conic
        vz = self.vertex_z_mm
        zeta_max = self.rim_z_mm - vz + 0.5  # slack for float round-off
        px, py, pz = p[0], p[1], p[2] - vz
        dx, dy, dz = d[0], d[1], d[2]

        a = dx * dx + dy * dy + kk * dz * dz
        b = 2.0 * (px * dx + py * dy + kk * pz * dz - r * dz)
        c = px * px + py * py + kk * pz * pz - 2.0 * r * pz
        disc = b * b - 4.0 * a * c
        valid = disc >= 0
        sq = np.sqrt(np.where(valid, disc, 0.0))
        sec_t = np.full(px.shape, np.inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            for root in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
                zeta = pz + root * dz
                h2 = (px + root * dx) ** 2 + (py + root * dy) ** 2
                cand = (
                    valid
                    & (root > 1e-6)
                    & (zeta >= -0.5)
                    & (zeta <= zeta_max)
                    & (h2 <= self.aperture_radius_mm**2)
                    & (root < sec_t)
                )
                sec_t = np.where(cand, root, sec_t)
        on_sec = np.isfinite(sec_t)
        counters["tip_rays"] = 0
        counters["hit_secondary"] = int(on_sec.sum())

        sec_t = sec_t[on_sec]
        d = d[:, on_sec]
        sec_hit = p[:, on_sec] + sec_t * d
        zeta = sec_hit[2] - vz
        sn = np.vstack([sec_hit[0], sec_hit[1], kk * zeta - r])
        sn /= np.linalg.norm(sn, axis=0)
        dot = np.einsum("ij,ij->j", d, sn)
        dot *= 2.0
        d = d - dot * sn
        return sec_hit, d, on_sec


# Outward plan unit vectors of the pyramid's four faces, east/north/west/south.
_PYRAMID_FACE_U = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


@dataclass(frozen=True)
class PyramidSecondary(Secondary):
    """Inverted four-sided pyramid secondary: four flat facets, apex on axis.

    The apex sits at ``(0, 0, apex_height_mm)`` and the four faces rise
    outward from it, so beams strike the undersides exactly as they strike
    the axicon cone. Face ``f`` has outward plan unit vector ``u`` (east,
    north, west, south order) and surface ``z = H + k * (u . p)`` over the
    plan triangle ``u . p >= |v . p|``, ``u . p <= half_side_mm``, with
    ``k = tan(angle_deg)`` and ``v`` the +90-degree rotation of ``u``.
    Equivalently the whole surface is the graph ``z(p) = H + k * max(|p_x|,
    |p_y|)`` — the axicon's cone with the circle swapped for a square.

    These are the pyramid's own explicit parameters, independent of any
    other secondary's — sharing a config field between two unrelated shapes
    would be a coincidence future code should not have to know about.

    There is no tip model: a sharp apex is the point of the shape, and a ray
    landing exactly on a ridge may be assigned to either adjacent face — a
    measure-zero set that is not special-cased.
    """

    apex_height_mm: float
    angle_deg: float
    half_side_mm: float

    def redirect(self, p, d, counters):
        z0 = self.apex_height_mm
        k = np.tan(np.deg2rad(self.angle_deg))
        a = self.half_side_mm
        px, py, pz = p[0], p[1], p[2]
        dx, dy, dz = d[0], d[1], d[2]

        n = px.shape[0]
        t_face = np.full((4, n), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for f, (ux, uy) in enumerate(_PYRAMID_FACE_U):
                t = (z0 + k * (ux * px + uy * py) - pz) / (dz - k * (ux * dx + uy * dy))
                sx = px + t * dx
                sy = py + t * dy
                up = ux * sx + uy * sy  # outward plan coordinate
                vp = -uy * sx + ux * sy  # along-ridge coordinate
                good = np.isfinite(t) & (t > 1e-9) & (up >= np.abs(vp)) & (up <= a)
                t_face[f] = np.where(good, t, np.inf)

        face = np.argmin(t_face, axis=0)
        sec_t = t_face[face, np.arange(n)]
        on_sec = np.isfinite(sec_t)
        counters["tip_rays"] = 0
        counters["hit_secondary"] = int(on_sec.sum())

        sec_t = sec_t[on_sec]
        face = face[on_sec]
        d = d[:, on_sec]
        sec_hit = p[:, on_sec] + sec_t * d

        normals = np.vstack(
            [-k * _PYRAMID_FACE_U[:, 0], -k * _PYRAMID_FACE_U[:, 1], np.ones(4)]
        ) / np.sqrt(1.0 + k * k)
        sn = normals[:, face]
        dot = np.einsum("ij,ij->j", d, sn)
        dot *= 2.0
        d = d - dot * sn
        return sec_hit, d, on_sec
