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

    def to_local_point(self, p: np.ndarray) -> np.ndarray:
        """World-frame point(s) already known to lie on this secondary,
        expressed in its own nominal (unperturbed) design frame -- the
        inverse of whatever rigid-body transform :meth:`redirect` applies
        before running its intersection math.

        Identity for a secondary with no rigid-body perturbation to undo:
        every shape but :class:`AxiconSecondary`/:class:`CassegrainSecondary`
        (which override this), and those two themselves whenever their own
        perturbation is exactly zero. Used by :func:`secondary_uv` (so the
        §C flux map stays anchored to the physical surface, not to world
        coordinates, when the surface has been decentred/tilted) and by
        rim/aperture-clearance probes (:mod:`heliostat.trace.cone`'s
        transmission-skip test, :func:`~heliostat.web.scene.field_miss_detection`)
        that compare a hit point's distance from the axis against
        ``aperture_radius_mm``, a property of the part's own body, not of
        where that body currently sits in the world.
        """
        return p


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


def _secondary_rotation_matrix(tip_mrad: float, tilt_mrad: float) -> np.ndarray:
    """World-from-local rotation for a secondary's rigid tip/tilt (spec §E2).

    ``tip_mrad`` rotates about the local x-axis (east): +z rotates toward
    +y for a positive tip, tilting the surface north-south. ``tilt_mrad``
    rotates about the local y-axis (north): +z rotates toward +x for a
    positive tilt, tilting the surface east-west. Composed tip-then-tilt,
    ``R = Ry(tilt) @ Rx(tip)`` -- at the mrad scale these perturbations are
    meant to represent, the composition order has no physical significance,
    but a fixed convention keeps repeated calls (and tests) consistent.
    """
    tip = tip_mrad * 1.0e-3
    tilt = tilt_mrad * 1.0e-3
    ct, st = np.cos(tip), np.sin(tip)
    cl, sl = np.cos(tilt), np.sin(tilt)
    r_tip = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    r_tilt = np.array([[cl, 0.0, sl], [0.0, 1.0, 0.0], [-sl, 0.0, cl]])
    return r_tilt @ r_tip


def _secondary_is_unperturbed(
    dx_mm: float, dy_mm: float, dz_mm: float, tip_mrad: float, tilt_mrad: float
) -> bool:
    """True iff every rigid-perturbation parameter is exactly zero.

    Gates a fast identity path in the transform helpers below: a
    zero-perturbation secondary runs through exactly the same
    floating-point operations it always has, with no rotation-matrix
    multiply -- not even by the identity matrix -- so a request that leaves
    every perturbation field at its default zero traces bit-identically to
    before this feature existed. That is a numerical promise (spec §E2
    "defaults all zero, saved with the receiver config"), not just a
    default value.
    """
    return dx_mm == 0.0 and dy_mm == 0.0 and dz_mm == 0.0 and tip_mrad == 0.0 and tilt_mrad == 0.0


def _to_secondary_local(
    p: np.ndarray,
    d: np.ndarray,
    vertex_z_mm: float,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    tip_mrad: float,
    tilt_mrad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ``(p, d)`` -> the secondary's own perturbed-local frame.

    Spec §E2's rigid-body misalignment is a rotation by tip/tilt about the
    secondary's own vertex ``(0, 0, vertex_z_mm)``, followed by a decenter
    translation: ``world = R @ (local - vertex) + vertex + decenter``. This
    is that composition's inverse, mapping an incoming world ray into the
    frame where the physical (perturbed) surface sits exactly where the
    *nominal* design equations expect it to -- so the existing exact conic
    intersection math, left completely unchanged, finds where the real,
    displaced part is actually hit. ``redirect`` transforms back with
    :func:`_to_secondary_world` once that math is done.
    """
    if _secondary_is_unperturbed(dx_mm, dy_mm, dz_mm, tip_mrad, tilt_mrad):
        return p, d
    r = _secondary_rotation_matrix(tip_mrad, tilt_mrad)
    vertex = np.array([0.0, 0.0, float(vertex_z_mm)])
    decenter = np.array([dx_mm, dy_mm, dz_mm])
    p_local = r.T @ (p - (vertex + decenter)[:, None]) + vertex[:, None]
    d_local = r.T @ d
    return p_local, d_local


def _to_secondary_world(
    p: np.ndarray,
    d: np.ndarray,
    vertex_z_mm: float,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    tip_mrad: float,
    tilt_mrad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`_to_secondary_local`: local frame -> world.

    Applied to the hit point and reflected direction the unperturbed conic
    math computed in the local frame, so every caller downstream -- the
    receiver intersection, shading/blocking, the 3-D scene -- sees the ray
    the real, physically displaced surface actually produced.
    """
    if _secondary_is_unperturbed(dx_mm, dy_mm, dz_mm, tip_mrad, tilt_mrad):
        return p, d
    r = _secondary_rotation_matrix(tip_mrad, tilt_mrad)
    vertex = np.array([0.0, 0.0, float(vertex_z_mm)])
    decenter = np.array([dx_mm, dy_mm, dz_mm])
    p_world = r @ (p - vertex[:, None]) + (vertex + decenter)[:, None]
    d_world = r @ d
    return p_world, d_world


def _secondary_local_point(
    p: np.ndarray,
    vertex_z_mm: float,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    tip_mrad: float,
    tilt_mrad: float,
) -> np.ndarray:
    """Position-only :func:`_to_secondary_local`, for a world-frame point
    already known to lie on the secondary (a hit point ``redirect`` already
    returned) rather than a ray about to be traced. Backs
    :meth:`AxiconSecondary.to_local_point`/:meth:`CassegrainSecondary.to_local_point`.
    """
    if _secondary_is_unperturbed(dx_mm, dy_mm, dz_mm, tip_mrad, tilt_mrad):
        return p
    r = _secondary_rotation_matrix(tip_mrad, tilt_mrad)
    vertex = np.array([0.0, 0.0, float(vertex_z_mm)])
    decenter = np.array([dx_mm, dy_mm, dz_mm])
    return r.T @ (p - (vertex + decenter)[:, None]) + vertex[:, None]


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
    #: Spec §E2 rigid-body misalignment, about the vertex/apex -- decenter
    #: (mm) and tip/tilt (mrad, see :func:`_secondary_rotation_matrix`).
    #: Defaults all zero, and at all-zero :meth:`redirect` takes a fast
    #: identity path that is bit-identical to the unperturbed geometry.
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    tip_mrad: float = 0.0
    tilt_mrad: float = 0.0

    def to_local_point(self, p):
        return _secondary_local_point(
            p, self.apex_height_mm, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )

    def redirect(self, p, d, counters):
        p, d = _to_secondary_local(
            p, d, self.apex_height_mm, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )
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
        cone_hit, d = _to_secondary_world(
            cone_hit, d, z0, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )
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
    #: Spec §E2 rigid-body misalignment, about the vertex -- decenter (mm)
    #: and tip/tilt (mrad, see :func:`_secondary_rotation_matrix`). Defaults
    #: all zero, and at all-zero :meth:`redirect` takes a fast identity path
    #: that is bit-identical to the unperturbed geometry.
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    tip_mrad: float = 0.0
    tilt_mrad: float = 0.0

    @property
    def rim_z_mm(self) -> float:
        h2 = self.aperture_radius_mm**2
        r, k = self.vertex_radius_mm, self.conic
        return self.vertex_z_mm + h2 / (r * (1.0 + np.sqrt(1.0 - (1.0 + k) * h2 / r**2)))

    def to_local_point(self, p):
        return _secondary_local_point(
            p, self.vertex_z_mm, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )

    def redirect(self, p, d, counters):
        p, d = _to_secondary_local(
            p, d, self.vertex_z_mm, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )
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
        sec_hit, d = _to_secondary_world(
            sec_hit, d, vz, self.dx_mm, self.dy_mm, self.dz_mm, self.tip_mrad, self.tilt_mrad
        )
        return sec_hit, d, on_sec


def secondary_has_flux_map(secondary: Secondary) -> bool:
    """Whether ``secondary`` has the single-valued radial ``(u, v)``
    parameterization :func:`secondary_uv` needs for a flux map.

    True only for :class:`AxiconSecondary` and :class:`CassegrainSecondary`
    -- both are surfaces of revolution about the tower axis, so "horizontal
    distance from the axis" is a well-defined single radial coordinate.
    :class:`NoSecondary` has no surface to map at all, and
    :class:`PyramidSecondary` has four flat facets with no single radial
    coordinate (a point's "distance from the axis" does not determine which
    facet it is on, nor its height on that facet) -- spec §C scopes the
    secondary-irradiance feature to axicon/Cassegrain for exactly this
    reason.
    """
    return isinstance(secondary, (AxiconSecondary, CassegrainSecondary))


def secondary_uv(secondary: Secondary, p: np.ndarray) -> np.ndarray:
    """``(2, K)`` surface ``(u, v)`` mm for world points ``p`` (``(3, K)``,
    mm) already known to lie on ``secondary``'s surface -- a hit point from
    :meth:`Secondary.redirect`, exactly as :meth:`Receiver.intersect`'s
    callers hand it their own already-clipped world hits.

    ``p`` is first mapped through :meth:`Secondary.to_local_point` into the
    surface's own nominal (unperturbed) frame, so a spec §E2 rigid-body
    misalignment does not move the flux map in world space along with the
    part -- ``(u, v)`` stays anchored to wherever on the physical surface
    the ray actually landed, which is what a §C secondary map is for.
    Identity when there is no perturbation (or none to undo).

    ``u`` is azimuthal arc length measured at the APERTURE RIM: ``u =
    aperture_radius_mm * atan2(x, -y)``, the same ``-y``/north-seam
    convention :mod:`heliostat.geometry.receiver` uses for its cylinder and
    frustum (azimuth zero at south, seam at north/+y where a heliostat can
    legitimately be aimed dead centre). Because ``u`` is scaled by the
    aperture radius rather than each point's own local radius, ``du``
    converts to a true angular increment via a single division by
    ``aperture_radius_mm`` everywhere on the surface -- see
    :func:`secondary_bin_areas_m2`.

    ``v`` is radial distance from the axis in horizontal projection,
    ``hypot(x, y)`` -- not true slant distance along the surface. Horizontal
    distance needs no ``z`` and is trivially invertible; a Cassegrain's true
    slant arc-length has no closed form (it would need ODE integration), so
    radial distance is what keeps both shapes on one code path.

    Unlike :mod:`heliostat.geometry.receiver`'s azimuth, no seam-continuity
    unwrapping is needed here: this function reports one ``(u, v)`` per
    already-known 3-D point rather than finite-differencing a bundle of
    same-origin rays, so plain ``arctan2`` -- always principal-valued in
    ``(-pi*R, pi*R]``, i.e. already inside :func:`secondary_uv_extent` -- is
    exact.
    """
    if not secondary_has_flux_map(secondary):
        raise ValueError(
            f"{type(secondary).__name__} has no single-valued (u, v) flux-map "
            "parameterization -- secondary_has_flux_map() is False for it"
        )
    p_local = secondary.to_local_point(p)
    x, y = p_local[0], p_local[1]
    u = secondary.aperture_radius_mm * np.arctan2(x, -y)
    v = np.hypot(x, y)
    return np.vstack([u, v])


def secondary_uv_extent(secondary: Secondary) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((u_min, u_max), (v_min, v_max))`` of ``secondary``'s ``(u, v)``
    parameterization, mm -- mirrors :meth:`Receiver.uv_extent`.

    ``u`` spans one full turn, ``aperture_radius_mm * (-pi, pi]`` (the
    surface closes on itself, like a receiver cylinder/frustum); ``v`` spans
    ``0`` (the axis) to ``aperture_radius_mm`` (the rim). Both shapes are
    circular in plan, clipped at ``aperture_radius_mm``, so this needs no
    per-shape branch.
    """
    if not secondary_has_flux_map(secondary):
        raise ValueError(
            f"{type(secondary).__name__} has no single-valued (u, v) flux-map "
            "parameterization -- secondary_has_flux_map() is False for it"
        )
    r = secondary.aperture_radius_mm
    return (-np.pi * r, np.pi * r), (0.0, r)


def _secondary_sec_local_slope(secondary: Secondary, h_mm: np.ndarray) -> np.ndarray:
    """``sec(local_slope(h))`` -- ``sqrt(1 + (dz/dh)^2)`` at horizontal
    radius ``h_mm`` -- for :func:`secondary_bin_areas_m2`'s area element.

    Axicon: the flank is ``z = apex_height_mm + h * tan(half_angle_deg)``, a
    constant slope, so this is the constant ``sqrt(1 + tan(half_angle)^2) =
    1 / cos(half_angle_deg)`` everywhere.

    Cassegrain: the flank is the implicit hyperboloid ``h^2 = 2 R zeta - (1
    + k) zeta^2`` (``R = vertex_radius_mm``, ``k = conic``, ``zeta = z -
    vertex_z_mm``) that :class:`CassegrainSecondary.redirect` itself solves.
    Implicit differentiation gives ``zeta(h) = (R - sqrt(R^2 - (1+k)h^2)) /
    (1+k)`` (the near-vertex branch, matching ``zeta=0`` at ``h=0``) and
    ``dzeta/dh = h / (R - (1+k)*zeta)`` -- the same ``zeta``/``kk``
    convention :class:`CassegrainSecondary.redirect` already uses.
    """
    h = np.asarray(h_mm, dtype=float)
    if isinstance(secondary, AxiconSecondary):
        k = np.tan(np.deg2rad(secondary.half_angle_deg))
        return np.full(h.shape, np.sqrt(1.0 + k * k))
    if isinstance(secondary, CassegrainSecondary):
        r = secondary.vertex_radius_mm
        kk = 1.0 + secondary.conic
        disc = np.clip(r * r - kk * h * h, 0.0, None)
        zeta = (r - np.sqrt(disc)) / kk
        slope = h / (r - kk * zeta)
        return np.sqrt(1.0 + slope * slope)
    raise ValueError(
        f"{type(secondary).__name__} has no single-valued (u, v) flux-map "
        "parameterization -- secondary_has_flux_map() is False for it"
    )


def secondary_bin_areas_m2(secondary: Secondary, grid: tuple[int, int]) -> np.ndarray:
    """True surface area of each ``(u, v)`` flux bin, ``(n_v, n_u)`` in m² --
    mirrors :meth:`Receiver.bin_areas_m2`/``FrustumReceiver``'s own override.

    Per bin: ``dA = h * sec(local_slope(h)) * dh * du / aperture_radius_mm``,
    evaluated at each ``v``-bin's midpoint radius ``h`` (a row is constant
    across ``u`` by rotational symmetry, exactly like
    ``FrustumReceiver.bin_areas_m2``'s row-by-row scaling). ``du /
    aperture_radius_mm`` is the true angular width ``d(phi)`` of a bin,
    because :func:`secondary_uv`'s ``u`` is arc length AT THE RIM -- fixed
    radius ``aperture_radius_mm`` -- not at the bin's own (smaller) radius
    ``h``, so this one division converts it correctly everywhere.

    The midpoint rule integrates a LINEAR integrand exactly for any bin
    count: for the axicon (constant ``sec(slope)``, so ``h * sec(slope)`` is
    linear in ``h``) the bin-area sum equals the closed-form cone lateral
    area ``pi * aperture_radius_mm^2 / cos(half_angle_deg)`` to machine
    precision. The Cassegrain's integrand is not linear, so its bin-area sum
    only converges to the (numerically integrated) true area as ``n_v``
    grows -- the same approximation ``FrustumReceiver.bin_areas_m2`` makes
    for its own (also nonlinear in general, but there linear because a
    frustum's radius is linear in slant position) row scaling.
    """
    (u0, u1), (v0, v1) = secondary_uv_extent(secondary)
    n_u, n_v = grid
    v_edges = np.linspace(v0, v1, n_v + 1)
    v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
    du_mm = (u1 - u0) / n_u
    dv_mm = v_edges[1] - v_edges[0]
    sec_slope = _secondary_sec_local_slope(secondary, v_mid)
    row_m2 = (v_mid * sec_slope * dv_mm * du_mm / secondary.aperture_radius_mm) / 1.0e6
    return np.repeat(row_m2[:, None], n_u, axis=1)


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
