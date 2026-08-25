"""Aiming strategies: per-layout aim-point choice and figure correction.

:mod:`heliostat.geometry.heliostat` answers "given an aim point, how is a
mirror pointed and what shape does it need" — the same question whatever
sits above the field. This module answers the layout-specific question in
front of that one: *where* does a given heliostat aim, and does its own
optics need any extra correction on top of the plain astigmatic figure.
Three ground-based layouts are covered:

* :func:`solve_prime_focus` and :func:`solve_cassegrain` — every heliostat
  aims at and focuses on one shared point ``F1 = (0, 0, focus_height_mm)``
  on the tower axis. The two layouts are optically identical upstream of
  F1 — a receiver sitting directly at F1 for prime focus, a hyperboloid
  relaying F1 down to a ground receiver for Cassegrain — so they share one
  solve (:func:`_solve_shared_focus`). Neither needs a figure correction:
  prime focus has no second optic, and a hyperboloid is stigmatic between
  its two foci so its relay contributes no extra astigmatism.
  :func:`solve_prime_focus_to_receiver` generalises the prime-focus half of
  this to any :mod:`heliostat.geometry.receiver` shape or position: each
  heliostat aims at that receiver's own ``aim_point_mm``, which is the
  shared axis point again for a receiver centred on-axis and a genuinely
  per-heliostat surface point for an off-axis, cylindrical or frustum one.
* :func:`solve_axicon` — a cone has no focus, so each heliostat's aim point
  is derived from its own radial field position (:func:`receiver_correction`
  solves where the mirror-to-receiver line meets the cone). The cone also
  has optical power in one direction only, which the heliostat has to
  pre-compensate; :func:`axicon_shape_correction` is that extra term.
* :func:`solve_pyramid` — an inverted four-sided pyramid, the axicon's cone
  with its circular symmetry broken into four flats. A flat facet has no
  optical power, so pointing is the plain shared-focus chain aimed at the
  mirror image of the receiver point in whichever facet
  (:func:`choose_face`) the chief ray actually lands on. Not covered by
  this package's golden fixtures; ported anyway since it needed no
  Quadoa-specific piece and no implicit config coupling once its two
  shared numbers (cone half-angle, cone aperture radius) became explicit
  parameters instead of being read by name off a shared config object.

Every ``solve_*`` function takes plain numbers — heliostat position, sun
position, and the layout's own geometry — and returns a :class:`Solution`.
Layout parameters are named to match the corresponding
:mod:`heliostat.geometry.secondary` class's constructor
(``apex_height_mm``/``half_angle_deg`` for :class:`~heliostat.geometry.secondary.AxiconSecondary`,
``apex_height_mm``/``angle_deg``/``half_side_mm`` for
:class:`~heliostat.geometry.secondary.PyramidSecondary`) or
:class:`~heliostat.geometry.receiver.FlatWindowReceiver` (``receiver_z_mm``
against its ``z_mm``), so a caller building the optics for a trace and the
aim point for the same heliostat reads the same numbers twice rather than
two different names for one physical quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .heliostat import heliostat_orientation, heliostat_shape

_SQRT3 = np.sqrt(3.0)
_SQRT6 = np.sqrt(6.0)


@dataclass(frozen=True)
class Solution:
    """Pointing and figure for one heliostat at one instant.

    ``rot_az_deg``/``rot_el_deg`` and ``c3``/``c4``/``c5`` are exactly what
    a trace call needs: the pointing angles
    :func:`~heliostat.geometry.heliostat.heliostat_orientation` would also
    produce, and figure coefficients in the convention
    :func:`~heliostat.geometry.heliostat.zernike_sag_and_slopes` (and the
    trace backends, which apply a further internal sign flip of their own —
    see :mod:`heliostat.trace.mc`/:mod:`heliostat.trace.cone`) expect.

    The ``extras`` aim-point contract
    ----------------------------------
    ``extras`` always carries::

        extras["aim_x_mm"], extras["aim_y_mm"], extras["aim_z_mm"]

    the world-coordinate point this heliostat is aimed at and focused on.
    :func:`~heliostat.geometry.shading.build_geometries` takes exactly this
    information as its ``aim_points_mm`` argument (one ``(N, 3)`` array for
    the whole field) to build the outgoing beam direction blocking is
    measured along; :func:`aim_points_mm` stacks a list of ``Solution``
    into that array. There is no fallback for a missing aim point on
    either side of that contract — a wrong or default aim point produces a
    blocking answer that is plausible and wrong, which is worse than
    failing loudly, so every ``solve_*`` here sets these three keys.

    Beyond those three, ``rot_astig_deg``, ``rad_s_mm``, ``rad_t_mm`` and
    ``focal_dist_s_mm`` are set by every layout that has a meaningful value
    for them; layout-specific diagnostics (the axicon's ``axicon_aoi_deg``,
    the pyramid's ``pyramid_face``/``face_margin_mm``) are each solve's own
    business.
    """

    rot_az_deg: float
    rot_el_deg: float
    c3: float
    c4: float
    c5: float
    aoi_deg: float = float("nan")
    focal_dist_mm: float = float("nan")
    cosine_efficiency: float = float("nan")
    extras: dict = field(default_factory=dict)


def aim_points_mm(solutions) -> np.ndarray:
    """Stack ``solutions``' aim points into the ``(N, 3)`` array
    :func:`~heliostat.geometry.shading.build_geometries` wants as
    ``aim_points_mm``."""
    return np.array(
        [[s.extras["aim_x_mm"], s.extras["aim_y_mm"], s.extras["aim_z_mm"]] for s in solutions],
        dtype=float,
    )


def to_trace_zernike(c3, c4, c5, c3_corr=0.0, c4_corr=0.0, c5_corr=0.0):
    """Sum a base figure and an optional secondary correction, in the
    normalised Zernike convention the trace backends consume.

    :func:`~heliostat.geometry.heliostat.heliostat_shape` returns
    ``c3``/``c4``/``c5`` as curvatures in the mirror's own local frame.
    The trace's Zernike-sag polynomial
    (:func:`~heliostat.geometry.heliostat.zernike_sag_and_slopes`, ANSI
    Z3..Z5, normrad = 1 mm) wants them normalised (divided by
    ``sqrt(3)``/``sqrt(6)``) and with ``c4``/``c5`` sign-flipped — the
    mirror's local curvature frame and the world sag frame the polynomial
    is written in disagree by exactly that reflection. A layout whose
    secondary contributes no extra astigmatism (prime focus and Cassegrain;
    a hyperboloid's relay is stigmatic between its two foci) passes the
    correction terms as zero and gets the base figure converted alone.

    This is a fixed bookkeeping convention pinned by the trace backends,
    not a free choice made here — the trace modules apply a further,
    separate c4/c5 sign flip of their own on top of whatever this returns
    (see their own comments), which is a different frame conversion this
    function does not anticipate.
    """
    tc3 = c3 / _SQRT6 + (-c3_corr / _SQRT6)
    tc4 = -c4 / _SQRT3 + (-c4_corr / _SQRT3)
    tc5 = -c5 / _SQRT6 + (c5_corr / _SQRT6)
    return tc3, tc4, tc5


# --------------------------------------------------------------------------
# Shared-focus layouts: prime focus and Cassegrain
# --------------------------------------------------------------------------


def _solve_to_aim(x_mm, y_mm, solar_az_deg, solar_el_deg, aim: np.ndarray) -> Solution:
    """Aim at an arbitrary world point, with no figure correction.

    Shared body for every solve that has one target and no secondary optic
    between the mirror and it: the fixed ``F1`` point
    (:func:`_solve_shared_focus`, used by :func:`solve_prime_focus` and
    :func:`solve_cassegrain`) and a per-heliostat point on a receiver's own
    surface (:func:`solve_prime_focus_to_receiver`). Nothing here assumes
    ``aim`` sits on the tower axis, so a heliostat at the field origin is
    well posed whatever ``aim`` is.
    """
    aim = np.asarray(aim, dtype=float)
    mirror_pos = np.array([float(x_mm), float(y_mm), 0.0], dtype=float)
    focal_dist = float(np.linalg.norm(aim - mirror_pos))

    (
        rot_az_deg,
        rot_el_deg,
        rot_astig_deg,
        rad_s,
        rad_t,
        aoi_deg,
        _u_mirror,
        _v_mirror,
    ) = heliostat_orientation(aim, mirror_pos, solar_az_deg, solar_el_deg)

    _c0, c3, c4, c5 = heliostat_shape(rot_astig_deg, rad_s, rad_t)
    tc3, tc4, tc5 = to_trace_zernike(c3, c4, c5)

    return Solution(
        rot_az_deg=float(rot_az_deg),
        rot_el_deg=float(rot_el_deg),
        c3=float(tc3),
        c4=float(tc4),
        c5=float(tc5),
        aoi_deg=float(aoi_deg),
        focal_dist_mm=focal_dist,
        cosine_efficiency=float(np.cos(np.deg2rad(aoi_deg))),
        extras={
            "rot_astig_deg": float(rot_astig_deg),
            "rad_s_mm": float(rad_s),
            "rad_t_mm": float(rad_t),
            # No secondary-induced sagittal shift, so the sagittal focal
            # distance is the plain one. Reported anyway so a summary table
            # has the same columns whichever layout produced it.
            "focal_dist_s_mm": focal_dist,
            "aim_x_mm": float(aim[0]),
            "aim_y_mm": float(aim[1]),
            "aim_z_mm": float(aim[2]),
        },
    )


def _solve_shared_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, focus_height_mm) -> Solution:
    """Aim at the one shared point ``F1 = (0, 0, focus_height_mm)``.

    :func:`_solve_to_aim` with that one point built for it -- see
    :func:`solve_prime_focus` and :func:`solve_cassegrain`, whose outputs
    are identical for identical inputs because everything that
    distinguishes the two layouts (whether a secondary sits above the
    field, how many reflections there are, the receiver's own position)
    lives outside pointing.
    """
    aim = np.array([0.0, 0.0, float(focus_height_mm)], dtype=float)
    return _solve_to_aim(x_mm, y_mm, solar_az_deg, solar_el_deg, aim)


def solve_prime_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, focus_height_mm) -> Solution:
    """Receiver at the field's common focus; no secondary mirror.

    ``focus_height_mm`` is both the aim height and the physical receiver
    height for this layout (pass the same number to
    ``FlatWindowReceiver(z_mm=focus_height_mm, ..., facing="down")``).
    """
    return _solve_shared_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, focus_height_mm)


def solve_cassegrain(x_mm, y_mm, solar_az_deg, solar_el_deg, focus_height_mm) -> Solution:
    """Hyperboloid secondary relaying the common focus onto a ground receiver.

    ``focus_height_mm`` is ``F1``, the virtual point every heliostat aims
    at and focuses on — NOT the physical receiver height, which sits lower
    behind the hyperboloid relay and is a property of the
    :class:`~heliostat.geometry.secondary.CassegrainSecondary`/receiver
    pair instead. Pointing only needs to know where the bundle is supposed
    to converge on its way to the secondary, and that is F1 by
    construction: a hyperboloid's defining property is that rays headed for
    one focus leave headed for the other, so this function never needs the
    hyperboloid's own conic constants.
    """
    return _solve_shared_focus(x_mm, y_mm, solar_az_deg, solar_el_deg, focus_height_mm)


def solve_prime_focus_to_receiver(x_mm, y_mm, solar_az_deg, solar_el_deg, receiver) -> Solution:
    """Prime focus aimed at ``receiver``'s own per-heliostat facing point.

    Generalises :func:`solve_prime_focus`: instead of the shared axis point
    ``(0, 0, focus_height_mm)``, each heliostat aims at
    ``receiver.aim_point_mm(x_mm, y_mm)`` --
    :mod:`heliostat.geometry.receiver`'s own answer to "which point on this
    surface faces this heliostat", which is the axis point itself for a
    receiver centred on-axis (so this reduces to exactly
    :func:`solve_prime_focus`'s aim point in that case) and a genuinely
    different point per heliostat for an off-axis, cylindrical or frustum
    receiver.

    Raises :class:`ValueError` if the resolved aim point sits at or below
    the heliostat plane (``z <= 0``) -- a receiver positioned or offset low
    enough to fail that has no physical field pointed at it.
    """
    aim = np.asarray(receiver.aim_point_mm(np.array([float(x_mm), float(y_mm)])), dtype=float)
    if aim[2] <= 0.0:
        raise ValueError(
            f"receiver aim point at ({aim[0]:.0f}, {aim[1]:.0f}, {aim[2]:.0f}) mm "
            "is at or below the heliostat plane (z = 0) -- raise the receiver "
            "or reduce the aperture-to-receiver offset"
        )
    return _solve_to_aim(x_mm, y_mm, solar_az_deg, solar_el_deg, aim)


# --------------------------------------------------------------------------
# Axicon
# --------------------------------------------------------------------------


def receiver_correction(mirror_radial_position, axicon_height, receiver_offset, axicon_angle_deg):
    """Where a ray from a heliostat meets the axicon cone, and the aim offset.

    Solves the intersection of the mirror-to-receiver line with the cone
    surface, in the radial/height plane containing the heliostat.
    """
    alpha = np.deg2rad(axicon_angle_deg)

    x_r = -receiver_offset * np.sin(2 * alpha)
    y_r = receiver_offset * np.cos(2 * alpha)

    x_m = mirror_radial_position
    y_m = -axicon_height

    slope = (y_r - y_m) / (x_r - x_m)
    x_a = (slope * x_m - y_m) / (slope - np.tan(alpha))
    y_a = np.tan(alpha) * x_a

    return x_r, y_r, x_a, y_a


def axicon_shape_correction(
    u, v, sagittal_vector, focal_dist, focal_dist_s, aoi_rad, foreshorten=None
):
    """Extra sagittal-only correction contributed by the axicon.

    The axicon has no tangential power, represented here by a very large
    tangential focal length.

    The correction is a cylinder, but its axis is the *axicon's* sagittal
    direction, which is not the mirror's own — so the plain sagittal
    relation ``rad = 2 f cos(aoi)`` does not apply to it unmodified.
    Projecting that direction onto the mirror yields two things: the angle
    the cylinder must run (used below, and correct as it stands), and the
    projection's length, which says how much of the direction lies in the
    mirror surface at all. A step across the mirror buys only that
    fraction of travel along the direction being corrected, and wavefront
    error grows as the square of distance, so the required curvature
    carries the squared length. It is 1 exactly when the axicon's sagittal
    direction lies in the mirror plane, where the plain relation is already
    right; ignoring it makes the correction ``1/L**2`` too strong, which is
    a factor of ~2 for the inner field.

    ``foreshorten`` overrides that factor; pass ``1.0`` to restore the
    plain (unforeshortened) relation.
    """
    d_power = 1.0 / focal_dist_s - 1.0 / focal_dist
    f_tangential = 1e20
    f_sagittal = 1.0 / d_power

    sag_u = np.dot(sagittal_vector, u)
    sag_v = np.dot(sagittal_vector, v)
    if foreshorten is None:
        foreshorten = (sag_u**2 + sag_v**2) / np.dot(sagittal_vector, sagittal_vector)

    rad_s = f_sagittal * 2.0 * np.cos(aoi_rad) / foreshorten
    rad_t = f_tangential * 2.0

    rot_astig = np.arctan2(sag_v, sag_u)
    return heliostat_shape(np.rad2deg(rot_astig), rad_s, rad_t)


def solve_axicon(
    x_mm,
    y_mm,
    solar_az_deg,
    solar_el_deg,
    apex_height_mm,
    half_angle_deg,
    receiver_z_mm,
    foreshorten=None,
) -> Solution:
    """Conical (axicon) secondary reflector.

    ``apex_height_mm``/``half_angle_deg`` match
    :class:`~heliostat.geometry.secondary.AxiconSecondary`'s own
    constructor arguments; ``receiver_z_mm`` matches
    ``FlatWindowReceiver(z_mm=...)`` for the ground receiver below the
    cone. ``foreshorten`` is exposed for :func:`axicon_shape_correction`
    (``None`` -- the physically-derived per-heliostat value -- unless a
    caller has a specific reason to override it).

    A heliostat is aimed by the bisector of the sun vector and the vector
    to its aim point, which for this layout is not the receiver itself but
    the point where the ray meets the axicon cone
    (:func:`receiver_correction`) -- pushed out along this heliostat's own
    radial direction, since the axicon has no single focus the whole field
    shares.
    """
    drop = apex_height_mm - receiver_z_mm
    field_radius = np.hypot(x_mm, y_mm)
    if field_radius == 0.0:
        raise ValueError("Heliostat at the field origin has no defined radial direction")

    (
        receiver_radial_offset,
        receiver_height_offset,
        axicon_radial_intersection,
        axicon_height_intersection,
    ) = receiver_correction(
        mirror_radial_position=field_radius,
        axicon_height=apex_height_mm,
        receiver_offset=drop,
        axicon_angle_deg=half_angle_deg,
    )

    # Push the aim point out along this heliostat's own radial direction.
    aim = np.array(
        [
            x_mm / field_radius * receiver_radial_offset,
            y_mm / field_radius * receiver_radial_offset,
            apex_height_mm + receiver_height_offset,
        ],
        dtype=float,
    )
    mirror_pos = np.array([x_mm, y_mm, 0.0], dtype=float)

    alpha = np.deg2rad(half_angle_deg)
    cone_dist = np.hypot(axicon_radial_intersection, axicon_height_intersection)
    s_prime = -np.hypot(drop + axicon_height_intersection, axicon_radial_intersection)
    radius_axicon = cone_dist / np.tan(alpha)

    axicon_aoi = np.rad2deg(
        np.arctan2(axicon_radial_intersection, drop + axicon_height_intersection) + alpha
    )
    s = 1.0 / (2.0 * np.cos(np.deg2rad(axicon_aoi)) / radius_axicon - 1.0 / s_prime)

    to_aim = aim - mirror_pos
    focal_dist = float(np.linalg.norm(to_aim))
    focal_dist_s = focal_dist + (s + s_prime)

    (
        rot_az_deg,
        rot_el_deg,
        rot_astig_deg,
        rad_s,
        rad_t,
        aoi_deg,
        u_mirror,
        v_mirror,
    ) = heliostat_orientation(aim, mirror_pos, solar_az_deg, solar_el_deg)

    c0, c3, c4, c5 = heliostat_shape(rot_astig_deg, rad_s, rad_t)

    # Sagittal direction of the beam, in the horizontal plane.
    focus_dir = to_aim / focal_dist
    focus_xy = np.array([focus_dir[0], focus_dir[1], 0.0])
    focus_xy /= np.linalg.norm(focus_xy)
    sagittal_vector = np.cross(focus_xy, np.array([0.0, 0.0, 1.0]))

    c0_c, c3_c, c4_c, c5_c = axicon_shape_correction(
        u_mirror,
        v_mirror,
        sagittal_vector,
        focal_dist,
        focal_dist_s,
        np.deg2rad(aoi_deg),
        foreshorten=foreshorten,
    )

    # Sum the base figure and the axicon's own correction, in the trace's
    # Zernike convention.
    tc3, tc4, tc5 = to_trace_zernike(c3, c4, c5, c3_c, c4_c, c5_c)

    return Solution(
        rot_az_deg=float(rot_az_deg),
        rot_el_deg=float(rot_el_deg),
        c3=float(tc3),
        c4=float(tc4),
        c5=float(tc5),
        aoi_deg=float(aoi_deg),
        focal_dist_mm=focal_dist,
        cosine_efficiency=float(np.cos(np.deg2rad(aoi_deg))),
        extras={
            "rot_astig_deg": float(rot_astig_deg),
            "rad_s_mm": float(rad_s),
            "rad_t_mm": float(rad_t),
            "axicon_aoi_deg": float(axicon_aoi),
            "focal_dist_s_mm": float(focal_dist_s),
            "aim_x_mm": float(aim[0]),
            "aim_y_mm": float(aim[1]),
            "aim_z_mm": float(aim[2]),
        },
    )


# --------------------------------------------------------------------------
# Pyramid (not gated by the golden fixtures -- see module docstring)
# --------------------------------------------------------------------------

# Outward plan unit vectors, in the index order reported as ``pyramid_face``:
# 0 = E, 1 = N, 2 = W, 3 = S. Same order as
# heliostat.geometry.secondary.PyramidSecondary's own face table.
FACE_U = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
FACE_NAMES = ("E", "N", "W", "S")


def _face_aim_points(apex_height_mm: float, angle_deg: float, drop_mm: float) -> np.ndarray:
    """The four virtual aim points, ``(4, 3)`` in world mm, face order E/N/W/S.

    Each face's aim point is the mirror image of the receiver point
    ``(0, 0, apex_height_mm - drop_mm)`` in that face's plane, so a chief
    ray aimed at it leaves the facet aimed at the receiver point exactly --
    a flat facet folds the path exactly, with no residual astigmatism to
    correct.
    """
    theta = np.deg2rad(angle_deg)
    plan = -drop_mm * np.sin(2.0 * theta)  # signed along u
    z = apex_height_mm + drop_mm * np.cos(2.0 * theta)
    aims = np.zeros((4, 3))
    aims[:, :2] = FACE_U * plan
    aims[:, 2] = z
    return aims


def _face_strike(mirror_pos, aim, u, apex_height_mm: float, k: float) -> np.ndarray:
    """Where the chief ray ``mirror_pos -> aim`` meets face ``u``'s PLANE.

    The plane is unclipped here; :func:`_face_margin` is what says whether
    the strike is on the actual triangular facet. Returns the 3-vector
    strike, or a point with non-finite coordinates when the ray runs
    parallel to the plane.
    """
    m = np.asarray(mirror_pos, dtype=float)
    a = np.asarray(aim, dtype=float)
    u = np.asarray(u, dtype=float)
    num = apex_height_mm + k * float(u @ m[:2]) - m[2]
    den = (a[2] - m[2]) - k * float(u @ (a[:2] - m[:2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / den
    return m + t * (a - m)


def _face_margin(strike, u, half_side_mm: float) -> float:
    """How far inside face ``u``'s plan triangle ``strike`` lands, in mm.

    The triangle is ``u.p >= |v.p|`` (between the two ridges) and
    ``u.p <= half_side_mm`` (inside the rim), with ``v`` the +90 deg
    rotation of ``u``. The margin is the smaller of the two slacks, so it
    is positive only when both hold and it is a distance either way --
    which is what makes "largest margin" a sensible tie-break for a
    heliostat that fits no facet.
    """
    s = np.asarray(strike, dtype=float)[:2]
    u = np.asarray(u, dtype=float)
    v = np.array([-u[1], u[0]])
    up = float(u @ s)
    vp = float(v @ s)
    return float(min(up - abs(vp), half_side_mm - up))


def choose_face(
    x_mm: float,
    y_mm: float,
    apex_height_mm: float,
    angle_deg: float,
    half_side_mm: float,
    receiver_z_mm: float,
) -> tuple[int, float, np.ndarray]:
    """``(face index, margin mm, aim point)`` for one heliostat.

    Nothing about the field says which facet a given heliostat should use,
    and the answer is not simply "the nearest one" -- the chief ray toward
    a face's own aim point can sail straight past that face and land on the
    far side of the apex, which is what happens to heliostats near the plan
    diagonals. This measures it: for each face it intersects the chief ray
    toward that face's own aim point with that face's plane, and scores the
    strike by how far inside the face's plan triangle it lands. Largest
    margin wins, even when every margin is negative -- a heliostat that
    cannot cleanly use any facet still has to point somewhere, and
    reporting the margin is what makes that visible instead of silent.
    Deterministic including ties: a heliostat exactly on a plan diagonal
    scores two faces identically and the lower index wins.
    """
    drop = apex_height_mm - receiver_z_mm
    k = np.tan(np.deg2rad(angle_deg))
    mirror_pos = np.array([float(x_mm), float(y_mm), 0.0])
    aims = _face_aim_points(apex_height_mm, angle_deg, drop)

    margins = np.empty(4)
    for f in range(4):
        strike = _face_strike(mirror_pos, aims[f], FACE_U[f], apex_height_mm, k)
        # A ray parallel to the plane has no strike at all; it cannot be the
        # best face, and -inf keeps it out of argmax without a second branch.
        margins[f] = (
            _face_margin(strike, FACE_U[f], half_side_mm)
            if np.all(np.isfinite(strike))
            else -np.inf
        )

    best = int(np.argmax(margins))
    return best, float(margins[best]), aims[best]


def solve_pyramid(
    x_mm,
    y_mm,
    solar_az_deg,
    solar_el_deg,
    apex_height_mm,
    angle_deg,
    half_side_mm,
    receiver_z_mm,
) -> Solution:
    """Inverted four-sided pyramid secondary; one flat facet per heliostat.

    ``apex_height_mm``/``angle_deg``/``half_side_mm`` match
    :class:`~heliostat.geometry.secondary.PyramidSecondary`'s own
    constructor arguments exactly; ``receiver_z_mm`` matches
    ``FlatWindowReceiver(z_mm=...)`` for the ground receiver below the
    pyramid, the same convention :func:`solve_axicon` uses.

    Not exercised by this package's golden fixtures (they cover prime
    focus, axicon and Cassegrain only); a heliostat at the field origin
    scores all four faces identically and takes face 0, same as the axicon
    layout is undefined there for a different reason (no radial direction).
    """
    face, margin, aim = choose_face(
        x_mm, y_mm, apex_height_mm, angle_deg, half_side_mm, receiver_z_mm
    )
    mirror_pos = np.array([float(x_mm), float(y_mm), 0.0], dtype=float)
    focal_dist = float(np.linalg.norm(aim - mirror_pos))

    (
        rot_az_deg,
        rot_el_deg,
        rot_astig_deg,
        rad_s,
        rad_t,
        aoi_deg,
        _u_mirror,
        _v_mirror,
    ) = heliostat_orientation(aim, mirror_pos, solar_az_deg, solar_el_deg)

    # A plane has no optical power, so there is no secondary-induced term to
    # add: the base astigmatic figure is the whole figure, exactly as for
    # the shared-focus layouts.
    _c0, c3, c4, c5 = heliostat_shape(rot_astig_deg, rad_s, rad_t)
    tc3, tc4, tc5 = to_trace_zernike(c3, c4, c5)

    return Solution(
        rot_az_deg=float(rot_az_deg),
        rot_el_deg=float(rot_el_deg),
        c3=float(tc3),
        c4=float(tc4),
        c5=float(tc5),
        aoi_deg=float(aoi_deg),
        focal_dist_mm=focal_dist,
        cosine_efficiency=float(np.cos(np.deg2rad(aoi_deg))),
        extras={
            "rot_astig_deg": float(rot_astig_deg),
            "rad_s_mm": float(rad_s),
            "rad_t_mm": float(rad_t),
            # No sagittal shift from a flat facet, so the sagittal focal
            # distance is the plain one -- same reasoning as the shared-focus
            # layouts.
            "focal_dist_s_mm": focal_dist,
            "aim_x_mm": float(aim[0]),
            "aim_y_mm": float(aim[1]),
            "aim_z_mm": float(aim[2]),
            # Which facet this heliostat was assigned to, and by how much it
            # cleared (or missed) that facet's triangle. Negative means the
            # chief ray toward the chosen aim does NOT land on the facet.
            "pyramid_face": int(face),
            "face_margin_mm": float(margin),
        },
    )
