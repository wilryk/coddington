"""Mutual shading and blocking between heliostats.

Why analytically rather than by inserting opaque blocker geometry into the
ray trace
---------------------------------------------------------------------------
Both effects are pure geometric occlusion by opaque flat rectangles, so a
ray-rectangle intersection test gives the same answer as tracing real
blocker surfaces would -- without rebuilding scene geometry once per
heliostat per timestep. The whole field for a whole sweep costs seconds
here.

It also means shading is computed *outside* the ray trace, so the model can
be revised, re-tuned, or turned off entirely without re-tracing anything.

Definitions
-----------
**Shading**  a neighbour casts a shadow on this heliostat, so less sunlight
             arrives. Tested along the sun vector.
**Blocking** a neighbour intercepts this heliostat's reflected beam before it
             reaches the secondary. Tested along the beam to the aim point.

Both are returned as fractions in [0, 1] of the mirror aperture that remains
useful, and enter the flux calculation as scalar multipliers on that
heliostat's contribution.

Mirror orientation convention
------------------------------
The mirror rectangle spans ``mirror_width`` along the horizontal axis
``u = normalize(z x n)`` and ``mirror_height`` along ``v = n x u`` -- the same
convention :func:`heliostat.geometry.heliostat.heliostat_orientation` uses to
build its own basis, so a normal computed there and a normal reconstructed
here from ``rot_az``/``rot_el`` agree. If that convention were ever changed,
shading fractions would change slightly; :func:`self_check` exercises the
geometry against hand-checkable cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Z = np.array([0.0, 0.0, 1.0])


def mirror_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal (u) and up-slope (v) axes of a mirror with the given normal."""
    u = np.cross(_Z, normal)
    norm = np.linalg.norm(u)
    if norm < 1e-9:  # mirror faces straight up; any horizontal axis will do
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / norm
    v = np.cross(normal, u)
    return u, v / np.linalg.norm(v)


def normal_from_angles(rot_az_deg: float, rot_el_deg: float) -> np.ndarray:
    az = np.deg2rad(rot_az_deg)
    el = np.deg2rad(rot_el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def sun_vector(solar_az_deg: float, solar_el_deg: float) -> np.ndarray:
    """Unit vector toward the sun.

    ``solar_az_deg`` is a compass bearing (0 = north, clockwise); converted
    to the standard math convention (0 = +x, counter-clockwise) via
    ``pi/2 - az``, matching
    :func:`heliostat.geometry.heliostat.heliostat_orientation`.
    """
    az = np.deg2rad(solar_az_deg)
    el = np.deg2rad(solar_el_deg)
    v = np.array(
        [
            np.cos(el) * np.cos(np.pi / 2 - az),
            np.cos(el) * np.sin(np.pi / 2 - az),
            np.sin(el),
        ]
    )
    return v / np.linalg.norm(v)


@dataclass
class MirrorGeometry:
    """One heliostat's rectangle in world coordinates."""

    centre: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    half_width: float
    half_height: float

    @classmethod
    def build(cls, x_mm, y_mm, rot_az_deg, rot_el_deg, half_width, half_height, z_mm: float = 0.0):
        n = normal_from_angles(rot_az_deg, rot_el_deg)
        u, v = mirror_basis(n)
        return cls(
            centre=np.array([float(x_mm), float(y_mm), float(z_mm)]),
            normal=n,
            u=u,
            v=v,
            half_width=float(half_width),
            half_height=float(half_height),
        )

    def sample_points(self, nu: int = 25, nv: int = 15) -> np.ndarray:
        """Grid of points across the aperture, cell centres. Shape (nu*nv, 3)."""
        su = (np.arange(nu) + 0.5) / nu * 2.0 - 1.0
        sv = (np.arange(nv) + 0.5) / nv * 2.0 - 1.0
        a, b = np.meshgrid(su * self.half_width, sv * self.half_height, indexing="ij")
        return self.centre + a.reshape(-1, 1) * self.u + b.reshape(-1, 1) * self.v


@dataclass
class SecondaryCone:
    """A conical secondary, as an opaque body that shades the field.

    A conical shell on the tower axis: vertex on the axis at ``z_tip_mm``,
    the surface rising outward at ``angle_deg`` from horizontal to the
    aperture rim. For a geometry with the vertex well below the rim (e.g.
    27 m tip / 20 deg / 15 m aperture radius puts the rim 5.46 m above the
    vertex) the cone is wide enough directly over the field that its shadow
    can land on real mirrors at mid elevations -- thrown tens of metres at
    moderate sun angles, and clear of the field entirely at low sun.

    Only shading. The secondary must not enter the blocking test: it *is*
    what every heliostat aims at, so a beam reaching it is the beam
    arriving, not a beam obstructed.
    """

    z_tip_mm: float
    angle_deg: float
    aperture_radius_mm: float

    @property
    def rim_height_mm(self) -> float:
        return self.z_tip_mm + self.aperture_radius_mm * np.tan(np.deg2rad(self.angle_deg))

    def occludes(self, points: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Which points have their ray to the sun stopped by the cone.

        Exact ray-cone intersection rather than a disc at the mean height:
        the vertex and the rim can differ by several metres, which at low
        sun elevation displaces the shadow by tens of metres -- a disc would
        put it on the wrong heliostats.
        """
        k = np.tan(np.deg2rad(self.angle_deg))
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        q = np.asarray(points, float) - np.array([0.0, 0.0, self.z_tip_mm])

        dx, dy, dz = (d[..., 0], d[..., 1], d[..., 2])
        a = dz**2 - k**2 * (dx**2 + dy**2)
        b = 2.0 * (q[:, 2] * dz - k**2 * (q[:, 0] * dx + q[:, 1] * dy))
        c = q[:, 2] ** 2 - k**2 * (q[:, 0] ** 2 + q[:, 1] ** 2)

        a = np.broadcast_to(np.asarray(a, float), b.shape)
        hit = np.zeros(len(q), dtype=bool)
        disc = b**2 - 4.0 * a * c

        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.sqrt(np.where(disc > 0, disc, 0.0))
            near_linear = np.abs(a) < 1e-12
            for t in (
                np.where(
                    near_linear,
                    -c / np.where(b == 0, 1.0, b),
                    (-b - root) / (2.0 * np.where(near_linear, 1.0, a)),
                ),
                np.where(
                    near_linear,
                    -c / np.where(b == 0, 1.0, b),
                    (-b + root) / (2.0 * np.where(near_linear, 1.0, a)),
                ),
            ):
                z = q[:, 2] + t * dz
                # z >= 0 keeps the correct nappe: the mirrored cone below the
                # vertex is not there, and squaring the surface equation
                # invented it. z/k is the radius at the hit, which must be
                # inside the rim.
                hit |= (disc >= 0) & (t > 1e-6) & (z >= 0.0) & (z <= k * self.aperture_radius_mm)
        return hit


@dataclass
class SecondaryDisc:
    """A horizontal circular secondary body, as seen by the shading test.

    A hyperboloid-relay secondary's silhouette. Unlike the cone there is
    nothing to integrate along: the surface may be sagged, but every point
    of it lies inside the rim circle when projected vertically, so a
    horizontal disc at the rim height is the exact silhouette for any sun
    direction rather than an approximation. (The cone needs its full
    ray-cone test precisely because its vertex sits well *below* its rim and
    can poke out of the rim's projection at low sun.)

    Shading only, on the same argument as :class:`SecondaryCone`: the disc
    must not enter the blocking test, because it is what the beam is on its
    way to. Heliostats aim above the disc at the relay's downstream focus,
    so a beam reaching the disc is the beam arriving, not a beam obstructed.

    Duck-types :class:`SecondaryCone`: ``occludes``, ``rim_height_mm``,
    ``aperture_radius_mm``, so :func:`shading_blocking` and
    :func:`occlusion_efficiency` take either.
    """

    z_mm: float
    radius_mm: float

    @property
    def rim_height_mm(self) -> float:
        return float(self.z_mm)

    @property
    def aperture_radius_mm(self) -> float:
        """Alias, so callers written against ``SecondaryCone`` work unchanged."""
        return float(self.radius_mm)

    def occludes(self, points: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Which points have their ray to the sun stopped by the disc.

        A ray is stopped iff it crosses ``z = z_mm`` *ahead* of the point and
        does so inside the rim radius. A ray travelling horizontally never
        crosses the plane and is never stopped.
        """
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        p = np.asarray(points, float)

        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.z_mm - p[:, 2]) / np.where(np.abs(dz) < 1e-12, np.nan, dz)
            hit_x = p[:, 0] + t * d[..., 0]
            hit_y = p[:, 1] + t * d[..., 1]
            inside = hit_x**2 + hit_y**2 <= self.radius_mm**2
        return np.asarray(np.isfinite(t) & (t > 1e-6) & inside, dtype=bool)


def _blocked_mask(
    points: np.ndarray,
    direction: np.ndarray,
    occluders: list[MirrorGeometry],
) -> np.ndarray:
    """Which of ``points`` have their ray stopped by one of ``occluders``.

    Returned as a mask rather than a fraction so several kinds of
    obstruction -- neighbouring mirrors and the secondary -- can be unioned.
    Multiplying their separate efficiencies would count twice a point that
    both of them shade.

    ``direction`` is either one vector shared by every point -- correct for
    sunlight, which arrives collimated -- or an ``(N, 3)`` array giving each
    point its own direction, which is what the outgoing beam needs: it
    converges on the aim point rather than travelling parallel, and across a
    several-metre mirror aiming tens of metres away the directions differ by
    several degrees.
    """
    blocked = np.zeros(len(points), dtype=bool)
    if not occluders:
        return blocked

    d = np.asarray(direction, dtype=float)
    per_point = d.ndim == 2
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)

    for occ in occluders:
        denom = d @ occ.normal
        usable = np.abs(denom) > 1e-9 if per_point else abs(float(denom)) > 1e-9
        if not np.any(usable):  # ray parallel to the occluder plane
            continue
        # Distance along d from each point to the occluder plane.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((occ.centre - points) @ occ.normal) / denom
        ahead = (t > 1e-6) & usable
        if not np.any(ahead):
            continue
        step = d[ahead] if per_point else d
        hit = points[ahead] + t[ahead, None] * step
        rel = hit - occ.centre
        inside = (np.abs(rel @ occ.u) <= occ.half_width) & (np.abs(rel @ occ.v) <= occ.half_height)
        idx = np.flatnonzero(ahead)[inside]
        blocked[idx] = True
        if blocked.all():
            break

    return blocked


def _fraction_unoccluded(points, direction, occluders) -> float:
    """Fraction of ``points`` whose ray hits none of ``occluders``."""
    return float(1.0 - _blocked_mask(points, direction, occluders).mean())


def shading_blocking(
    geometries: list[MirrorGeometry],
    aim_points: np.ndarray,
    solar_az_deg: float,
    solar_el_deg: float,
    neighbours: list[np.ndarray],
    nu: int = 25,
    nv: int = 15,
    secondary: "SecondaryCone | SecondaryDisc | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shading and blocking efficiency for every heliostat at one instant.

    Returns ``(eta_shade, eta_block, eta_secondary)``. ``eta_shade`` is the
    fraction of the aperture that can see the sun at all, so it already
    includes anything the secondary blots out; ``eta_secondary`` is the
    secondary acting alone and is reported only so its cost can be read off
    separately. The two are unioned rather than multiplied, because a patch
    of mirror shaded by both a neighbour and the secondary is lost once, not
    twice.
    """
    to_sun = sun_vector(solar_az_deg, solar_el_deg)
    n = len(geometries)
    eta_shade = np.ones(n)
    eta_block = np.ones(n)
    eta_secondary = np.ones(n)

    if solar_el_deg <= 0.0:
        return np.zeros(n), np.zeros(n), np.ones(n)

    for i, geom in enumerate(geometries):
        nbrs = [geometries[j] for j in neighbours[i]]
        pts = geom.sample_points(nu, nv)

        # The sun is at infinity, so one direction serves every point.
        shaded = _blocked_mask(pts, to_sun, nbrs)
        if secondary is not None:
            by_cone = secondary.occludes(pts, to_sun)
            eta_secondary[i] = float(1.0 - by_cone.mean())
            shaded = shaded | by_cone
        eta_shade[i] = float(1.0 - shaded.mean())

        if not nbrs:
            continue

        # The aim point is not: each point on the aperture heads for it along
        # its own direction. Using one direction from the mirror centre
        # treats the outgoing beam as collimated, which is wrong by several
        # degrees across the aperture and misplaces the blocked region by
        # hundreds of millimetres at the neighbour.
        eta_block[i] = _fraction_unoccluded(pts, aim_points[i] - pts, nbrs)

    return eta_shade, eta_block, eta_secondary


def occlusion_efficiency(
    geometries: list[MirrorGeometry],
    aim_points: np.ndarray,
    solar_az_deg: float,
    solar_el_deg: float,
    neighbours: list[np.ndarray],
    nu: int = 25,
    nv: int = 15,
    secondary: "SecondaryCone | SecondaryDisc | None" = None,
) -> np.ndarray:
    """The fraction of each aperture that is lit *and* unblocked.

    Not ``eta_shade * eta_block``. That product treats the two losses as
    independent, and they are not: a patch of mirror lying in a neighbour's
    shadow sends no beam onward, so it cannot also be blocked. Where the
    shaded and blocked regions overlap the product removes the same patch
    twice and understates the delivered power -- measured 6 points low on a
    heavily occluded heliostat (0.278 against a union-form 0.335), which a
    ray trace of the same geometry settles at 0.332.

    Kept separate from :func:`shading_blocking` rather than replacing it,
    because ``eta_shade`` and ``eta_block`` are still the right things to
    report on their own; it is only their combination that has to be a
    union.
    """
    to_sun = sun_vector(solar_az_deg, solar_el_deg)
    eta = np.ones(len(geometries))
    if solar_el_deg <= 0.0:
        return np.zeros(len(geometries))

    for i, geom in enumerate(geometries):
        nbrs = [geometries[j] for j in neighbours[i]]
        pts = geom.sample_points(nu, nv)
        lost = _blocked_mask(pts, to_sun, nbrs) | _blocked_mask(pts, aim_points[i] - pts, nbrs)
        if secondary is not None:
            # Same union: a patch the secondary already shades cannot be
            # shaded by a neighbour as well, nor blocked on the way out.
            lost = lost | secondary.occludes(pts, to_sun)
        eta[i] = float(1.0 - lost.mean())
    return eta


def corner_shadow(geom: MirrorGeometry, direction: np.ndarray, ground_z: float = 0.0) -> np.ndarray:
    """The mirror's four corners projected along ``direction`` onto the ground.

    The classical formulation: cast the corners to a common plane and
    overlap the resulting polygons. It is exactly equivalent to intersecting
    rays with the mirror plane -- parallel projection is affine, so the
    overlap *fraction* of the target's own shadow is preserved whatever
    plane you land on -- and :func:`self_check` asserts the two agree to
    zero.

    The equivalence has one condition that is easy to lose: an occluder only
    shades what is *down*-sun of it. Ground shadows alone cannot tell which
    side of the target a neighbour sits on, because both lie on the same sun
    line, so overlapping every neighbour's shadow counts the ones behind the
    target as well and roughly doubles the apparent loss. Callers must
    filter to up-sun occluders, which :func:`_fraction_unoccluded` gets for
    free from ``t > 0``.
    """
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    corners = np.array(
        [
            geom.centre + su * geom.half_width * geom.u + sv * geom.half_height * geom.v
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    return corners - ((corners[:, 2] - ground_z) / d[2])[:, None] * d


def search_radius_for(
    min_elevation_deg: float,
    mirror_height_mm: float,
    mirror_width_mm: float,
    cap_mm: float = 60000.0,
) -> float:
    """How far a shadow can reach at the lowest traced sun elevation.

    Used to size the neighbour query so no plausible occluder is missed
    while keeping the per-heliostat neighbour list small.
    """
    el = max(float(min_elevation_deg), 1.0)
    reach = mirror_height_mm / np.tan(np.deg2rad(el))
    return float(min(cap_mm, reach + mirror_width_mm))


def build_geometries(
    field,
    rot_az_deg,
    rot_el_deg,
    aim_points_mm,
    *,
    mirror_width_mm: float | None = None,
    mirror_height_mm: float | None = None,
    pedestal_height_mm: float = 0.0,
) -> tuple[list[MirrorGeometry], np.ndarray]:
    """Mirror rectangles and aim points for a whole field at one instant.

    ``field`` is a :class:`heliostat.field.HeliostatField`. ``rot_az_deg``/
    ``rot_el_deg`` are per-heliostat pointing angles (e.g. the ``rot_az_deg``/
    ``rot_el_deg`` a :func:`heliostat.geometry.heliostat.heliostat_orientation`
    solve produced for each heliostat), and ``aim_points_mm`` is an ``(N, 3)``
    array of the world-coordinate point each heliostat is focused on -- the
    same information a per-heliostat solution's aim-point extras would carry,
    given here explicitly rather than through a solution object, so this
    module has no dependency on any particular secondary strategy's solve
    output.

    There is deliberately no fallback for a missing or wrong aim point: a
    caller that has not resolved the real aim point for every heliostat would
    get a blocking answer that is plausible and wrong rather than an error,
    which is worse than failing loudly. Resolve aim points before calling.

    ``mirror_width_mm``/``mirror_height_mm`` default to ``field``'s own (set
    by :func:`heliostat.field.load_field`); pass them explicitly for a field
    that was not loaded with them, e.g. a synthetic one in a test.
    """
    width = field.mirror_width_mm if mirror_width_mm is None else mirror_width_mm
    height = field.mirror_height_mm if mirror_height_mm is None else mirror_height_mm
    if not width or not height:
        raise ValueError(
            "mirror_width_mm/mirror_height_mm are required to build mirror "
            "geometry: the field carries none (mirror_width_mm="
            f"{field.mirror_width_mm}, mirror_height_mm={field.mirror_height_mm}) "
            "and neither was passed explicitly."
        )
    half_w, half_h = width / 2.0, height / 2.0

    aim_points_mm = np.asarray(aim_points_mm, dtype=float)
    if aim_points_mm.shape != (len(field), 3):
        raise ValueError(
            f"aim_points_mm must have shape ({len(field)}, 3), got {aim_points_mm.shape}"
        )

    geoms = [
        MirrorGeometry.build(
            field.x_mm[i],
            field.y_mm[i],
            rot_az_deg[i],
            rot_el_deg[i],
            half_w,
            half_h,
            pedestal_height_mm,
        )
        for i in range(len(field))
    ]
    return geoms, aim_points_mm


def self_check(verbose: bool = True) -> bool:
    """Hand-checkable cases, so the geometry can be trusted before a long run."""
    ok = True

    # A mirror directly south of another, sun low in the south: fully shaded.
    target = MirrorGeometry.build(0, 0, 90.0, 45.0, 2500, 1500)
    # Occluder placed one metre up-sun, large enough to cover completely.
    to_sun = sun_vector(180.0, 10.0)
    occ_centre = target.centre + to_sun * 8000.0
    occluder = MirrorGeometry(
        centre=occ_centre,
        normal=-to_sun,
        u=np.cross(_Z, -to_sun) / np.linalg.norm(np.cross(_Z, -to_sun)),
        v=np.cross(-to_sun, np.cross(_Z, -to_sun) / np.linalg.norm(np.cross(_Z, -to_sun))),
        half_width=20000,
        half_height=20000,
    )
    frac = _fraction_unoccluded(target.sample_points(), to_sun, [occluder])
    if verbose:
        print(f"  fully-occluded case: unoccluded fraction = {frac:.3f} (expect 0.000)")
    ok &= abs(frac) < 1e-9

    # Same occluder moved far to the side: no shading.
    far = MirrorGeometry(
        centre=occ_centre + np.array([500000.0, 0.0, 0.0]),
        normal=occluder.normal,
        u=occluder.u,
        v=occluder.v,
        half_width=2500,
        half_height=1500,
    )
    frac = _fraction_unoccluded(target.sample_points(), to_sun, [far])
    if verbose:
        print(f"  distant occluder   : unoccluded fraction = {frac:.3f} (expect 1.000)")
    ok &= abs(frac - 1.0) < 1e-9

    # Occluder behind the mirror relative to the sun: no shading.
    behind = MirrorGeometry(
        centre=target.centre - to_sun * 8000.0,
        normal=occluder.normal,
        u=occluder.u,
        v=occluder.v,
        half_width=20000,
        half_height=20000,
    )
    frac = _fraction_unoccluded(target.sample_points(), to_sun, [behind])
    if verbose:
        print(f"  occluder behind    : unoccluded fraction = {frac:.3f} (expect 1.000)")
    ok &= abs(frac - 1.0) < 1e-9

    # -- the two cases that pin down the *magnitude* of low-sun shading --------
    #
    # Two heliostats at the same ground height with the same normal are
    # parallel planes, so every point of the target maps onto the occluder
    # with the SAME offset: the shaded fraction is just the overlap of two
    # identical rectangles displaced by that offset, which is closed form.
    # This is the check that says the sampled answer is the right size, not
    # merely between 0 and 1.
    to_sun = sun_vector(88.0, 9.71)  # just after sunrise, sun due east
    hw, hh = 2500.0, 1500.0
    # Normals within the spread a real field shows at this hour: the sun's
    # mathematical azimuth is 90 - 88 = 2 deg, and the solved normals sit a
    # few tens of degrees either side of it.
    for rot_az, rot_el in ((2.0, 28.0), (20.0, 21.0), (-15.0, 40.0)):
        g = MirrorGeometry.build(0.0, 0.0, rot_az, rot_el, hw, hh)
        shift = to_sun * 6000.0
        shift[2] = 0.0  # neighbour 6 m up-sun, same height
        occ = MirrorGeometry.build(shift[0], shift[1], rot_az, rot_el, hw, hh)

        t = float((occ.centre - g.centre) @ occ.normal / (to_sun @ occ.normal))
        off = (g.centre + t * to_sun) - occ.centre
        overlap = max(0.0, 1.0 - abs(off @ occ.u) / (2 * hw)) * max(
            0.0, 1.0 - abs(off @ occ.v) / (2 * hh)
        )
        exact = 1.0 - overlap

        # A dense grid, because the tolerance is what makes the check
        # meaningful: the default 25 x 15 quantises the answer to 1/375, and
        # sampling cell centres costs about 1/n per axis at the shadow's
        # edge.
        n = 401
        got = _fraction_unoccluded(g.sample_points(n, n), to_sun, [occ])
        if verbose:
            print(
                f"  aligned pair az={rot_az:+6.1f} el={rot_el:4.1f}: "
                f"{got:.4f} vs closed form {exact:.4f}"
            )
        ok &= abs(got - exact) < 2.5 / n
        # Guard against a case that passes because nothing is shaded at all.
        ok &= exact < 0.9

        # Rows 2..4 up-sun shift by 2x, 3x, 4x the same offset, so each
        # shadows a strict subset of what row 1 already shadows -- shading
        # saturates at the nearest neighbour. This is *why* low-sun losses
        # are not larger, so it is worth failing loudly if a future change
        # breaks it.
        deeper = [
            MirrorGeometry.build(shift[0] * k, shift[1] * k, rot_az, rot_el, hw, hh)
            for k in range(1, 5)
        ]
        saturated = _fraction_unoccluded(g.sample_points(n, n), to_sun, deeper)
        if verbose:
            print(f"    + rows 2-4 up-sun  : {saturated:.4f} (expect no change)")
        ok &= abs(saturated - got) < 1e-12

    # -- the ground-shadow polygon method must agree exactly ------------------
    #
    # Independent formulation, and the one heliostat codes traditionally use:
    # project the corners of every rectangle to the ground along the sun and
    # overlap the polygons. Agreement here is what says the sampled answer is
    # not merely self-consistent.
    pedestal = 5000.0
    g = MirrorGeometry.build(0.0, 0.0, 4.0, 26.0, hw, hh, pedestal)
    occs = [
        MirrorGeometry.build(shift[0] * k, shift[1] * k, 4.0, 26.0, hw, hh, pedestal)
        for k in (1, 2)
    ]
    sampled = _fraction_unoccluded(g.sample_points(301, 301), to_sun, occs)

    tgt = corner_shadow(g, to_sun)[:, :2]
    n = 301
    a, b = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij")
    grid = tgt[0] + a.ravel()[:, None] * (tgt[1] - tgt[0]) + b.ravel()[:, None] * (tgt[3] - tgt[0])
    covered = np.zeros(len(grid), dtype=bool)
    for occ in occs:
        q = corner_shadow(occ, to_sun)[:, :2]
        edge = np.roll(q, -1, axis=0) - q
        side = np.sign(np.cross(edge[None, :, :], grid[:, None, :] - q[None, :, :]))
        covered |= np.all(side >= 0, axis=1) | np.all(side <= 0, axis=1)
    polygon = 1.0 - covered.mean()

    if verbose:
        print(f"  ground-shadow polygons: {polygon:.4f} vs ray sampling {sampled:.4f}")
    ok &= abs(polygon - sampled) < 1e-9

    # -- a common pedestal height cannot change mutual shading ----------------
    #
    # Every heliostat shares one height, and shifting the whole field
    # vertically leaves every mirror-to-mirror relationship identical.
    heights = []
    for z in (0.0, 5000.0, 20000.0):
        gz = MirrorGeometry.build(0.0, 0.0, 4.0, 26.0, hw, hh, z)
        oz = [
            MirrorGeometry.build(shift[0] * k, shift[1] * k, 4.0, 26.0, hw, hh, z) for k in (1, 2)
        ]
        heights.append(_fraction_unoccluded(gz.sample_points(101, 101), to_sun, oz))
    if verbose:
        print(
            f"  pedestal 0 / 5 / 20 m : {heights[0]:.6f} {heights[1]:.6f} "
            f"{heights[2]:.6f} (expect identical)"
        )
    ok &= max(heights) - min(heights) < 1e-12

    # -- blocking uses per-point directions, and it matters -------------------
    #
    # The outgoing beam converges on the aim point, so each aperture point
    # has its own direction. Asserting the two formulations *differ* stops a
    # future simplification back to one shared direction from passing
    # silently.
    aim = np.array([0.0, 0.0, 27000.0])
    g = MirrorGeometry.build(80000.0, 0.0, 4.0, 14.0, hw, hh)
    occs = [MirrorGeometry.build(80000.0 - 6000.0, 0.0, 4.0, 14.0, hw, hh)]
    pts = g.sample_points(101, 101)
    collimated = _fraction_unoccluded(pts, aim - g.centre, occs)
    converging = _fraction_unoccluded(pts, aim - pts, occs)

    # Brute force, one point at a time, as an independent implementation of
    # the per-point path -- the vectorised version has to index three arrays
    # in step and getting that subtly wrong would still return a plausible
    # number.
    slow = np.mean([_fraction_unoccluded(p[None, :], aim - p, occs) for p in pts[::37]])
    reference = _fraction_unoccluded(pts[::37], aim - pts[::37], occs)

    if verbose:
        print(f"  blocking, one direction {collimated:.4f} vs per-point {converging:.4f}")
        print(f"    per-point vs point-at-a-time: {reference:.6f} / {slow:.6f}")
    ok &= abs(collimated - converging) > 1e-3  # the correction is real
    ok &= 0.02 < converging < 0.98  # and the case actually blocks
    ok &= abs(reference - slow) < 1e-12  # vectorisation is faithful

    # -- the secondary shades the field, and only in shading ------------------
    cone = SecondaryCone(z_tip_mm=27000.0, angle_deg=20.0, aperture_radius_mm=15000.0)
    axis = np.array([[0.0, 0.0, 0.0]])

    # Straight up the axis from the centre: the vertex is directly overhead.
    ok &= bool(cone.occludes(axis, np.array([0.0, 0.0, 1.0]))[0])
    # Straight down: the cone is behind, not ahead.
    ok &= not bool(cone.occludes(axis, np.array([0.0, 0.0, -1.0]))[0])

    # A ray aimed just outside the rim must miss, and just inside must hit.
    # The rim is 5.46 m above the vertex, so the miss/hit boundary is nowhere
    # near where a flat disc at the vertex height would put it -- this is the
    # case that fails if the cone is ever simplified to a disc.
    rim_z = cone.rim_height_mm
    for offset, expect in ((-400.0, True), (400.0, False)):
        r = cone.aperture_radius_mm + offset
        start = np.array([[r, 0.0, 0.0]])
        # Aim at the rim point directly above r, straight up.
        ok &= bool(cone.occludes(start, np.array([0.0, 0.0, 1.0]))[0]) == expect
    if verbose:
        print(
            f"  secondary cone : rim {cone.aperture_radius_mm / 1000:.0f} m radius at "
            f"{rim_z / 1000:.2f} m, vertex {cone.z_tip_mm / 1000:.0f} m"
        )

    # A low sun throws the shadow clear of the field; a high sun drops it
    # inside.
    #
    # The sample point is offset 8 m beyond where the *vertex* alone would
    # put the shadow. Exactly at the vertex throw the ray grazes the apex,
    # the quadratic has a double root, and whether the discriminant lands
    # just above or just below zero is down to rounding -- a test on that
    # boundary would be flaky. It is also the wrong place to look: the rim
    # is 5.46 m higher than the vertex, so the shadow band lies *past* the
    # vertex throw, and it is that displacement a flat disc at vertex height
    # would get wrong.
    for el, shaded_at_30m in ((9.7, False), (45.0, True)):
        to_sun = sun_vector(90.0, el)
        throw = cone.z_tip_mm / np.tan(np.deg2rad(el)) / 1000.0
        rim_throw = cone.rim_height_mm / np.tan(np.deg2rad(el)) / 1000.0
        pt = np.array([[-(throw + 8.0) * 1000.0, 0.0, 0.0]])
        ok &= bool(cone.occludes(pt, to_sun)[0])
        near = np.array([[-30000.0, 0.0, 0.0]])
        got = bool(cone.occludes(near, to_sun)[0])
        if verbose:
            print(
                f"    sun el {el:4.1f}° -> shadow band {throw:5.1f}-{rim_throw:5.1f} m "
                f"from the axis; heliostat at 30 m {'in it' if got else 'clear'}"
            )
        ok &= got == shaded_at_30m

    if verbose:
        print("  PASS" if ok else "  FAIL")
    return bool(ok)
