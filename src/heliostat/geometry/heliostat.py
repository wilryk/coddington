"""Mirror pointing and figure common to every secondary layout.

These two functions answer "given an aim point, how is this heliostat
pointed and what shape does it need" — the same question whatever sits at
the top of the tower. Only the choice of aim point, and any extra
correction the specific secondary demands on top, is layout-specific and
lives elsewhere.
"""

from __future__ import annotations

import numpy as np


def heliostat_orientation(receiver_pos, mirror_pos, solar_az_deg, solar_el_deg):
    """Aim a flat mirror to send the sun to ``receiver_pos``.

    Returns pointing angles, the astigmatic rotation, the sagittal/
    tangential focal radii, the angle of incidence, and the mirror's local
    basis vectors.

    The mirror normal is the *bisector* of the direction to the sun and the
    direction to the aim point, both unit vectors from the mirror centre.
    Anything that reconstructs the normal from ``rot_az``/``rot_el`` has to
    use exactly this convention or it will disagree with the ray trace.
    """
    receiver_pos = np.asarray(receiver_pos, dtype=float)
    mirror_pos = np.asarray(mirror_pos, dtype=float)

    solar_az = np.deg2rad(solar_az_deg)
    solar_el = np.deg2rad(solar_el_deg)

    # Unit vector from mirror toward the sun. Azimuth is compass bearing,
    # hence the pi/2 - az conversion into standard math convention.
    to_sun = np.array(
        [
            np.cos(solar_el) * np.cos(np.pi / 2 - solar_az),
            np.cos(solar_el) * np.sin(np.pi / 2 - solar_az),
            np.sin(solar_el),
        ]
    )
    to_sun /= np.linalg.norm(to_sun)

    to_target = receiver_pos - mirror_pos
    focal_length = np.linalg.norm(to_target)
    to_target = to_target / focal_length

    normal = to_sun + to_target
    normal /= np.linalg.norm(normal)

    rot_el = np.arcsin(normal[2])
    rot_az = np.arctan2(normal[1], normal[0])

    aoi = 0.5 * np.arccos(np.clip(np.dot(to_sun, to_target), -1.0, 1.0))

    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    rot_astig = np.arctan2(np.dot(to_sun, v), np.dot(to_sun, u))

    radius = focal_length * 2.0
    rad_s = radius * np.cos(aoi)
    rad_t = radius / np.cos(aoi)

    return (
        np.rad2deg(rot_az),
        np.rad2deg(rot_el),
        np.rad2deg(rot_astig),
        rad_s,
        rad_t,
        np.rad2deg(aoi),
        u,
        v,
    )


def heliostat_shape(rot_astig_deg, rad_s, rad_t):
    """Curvature radii from :func:`heliostat_orientation` -> figure coefficients.

    An off-axis mirror used at nonzero angle of incidence is astigmatic:
    its sagittal and tangential focal radii differ. Correcting that exactly
    needs an astigmatic (per-timestep) figure — this converts the two focal
    radii and the astigmatic rotation into the three second-order terms
    (``c0`` isotropic curvature, ``c3``/``c4``/``c5`` the astigmatic and
    defocus terms) of that figure, in the mirror's own local frame.
    """
    rot_astig = np.deg2rad(rot_astig_deg)
    curv_t = 1.0 / rad_t
    curv_s = 1.0 / rad_s

    c0 = 0.125 * (curv_s + curv_t)
    c3 = 0.25 * (curv_t - curv_s) * np.sin(2 * rot_astig)
    c4 = 0.125 * (curv_t + curv_s)
    c5 = 0.25 * (curv_t - curv_s) * np.cos(2 * rot_astig)
    return c0, c3, c4, c5


_SQRT3 = np.sqrt(3.0)
_SQRT6 = np.sqrt(6.0)


def zernike_sag_and_slopes(x, y, c3, c4, c5):
    """ANSI Z3..Z5 sag and its partial derivatives, normrad = 1 (x, y in mm).

    Lives here (a numpy-only module) so both tracer backends and the design
    layer share one pinned polynomial without import cycles.
    """
    sag = (
        c3 * _SQRT6 * 2.0 * x * y
        + c4 * _SQRT3 * (2.0 * x * x + 2.0 * y * y - 1.0)
        + c5 * _SQRT6 * (x * x - y * y)
    )
    dsdx = c3 * _SQRT6 * 2.0 * y + c4 * _SQRT3 * 4.0 * x + c5 * _SQRT6 * 2.0 * x
    dsdy = c3 * _SQRT6 * 2.0 * x + c4 * _SQRT3 * 4.0 * y - c5 * _SQRT6 * 2.0 * y
    return sag, dsdx, dsdy
