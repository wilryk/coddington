"""Monte Carlo heliostat trace: one heliostat, one instant, many rays.

Traces the full optical chain sequentially — source disk, figured primary
mirror, secondary reflector, receiver — exactly as the physical light does.
A ray that misses any surface's aperture along the way is dropped there and
never considered again; the counter chain records where.

Convention pins that are not free to change
--------------------------------------------
- Super-Gaussian source: ``I(theta) = exp(-(theta^2 / 2 sigma^2)^n)``,
  ``sigma = 0.0024`` rad, ``n = 2``. See :mod:`heliostat.trace.samplers`.
- Figure sag is the ANSI Z80.28 astigmatic/defocus form, radius normalised
  to 1 mm: ``sag = c3*sqrt(6)*2xy + c4*sqrt(3)*(2x^2+2y^2-1) +
  c5*sqrt(6)*(x^2-y^2)``.
"""

from __future__ import annotations

import numpy as np

from ..geometry.receiver import Receiver
from ..geometry.secondary import Secondary
from .samplers import SuperGaussSampler

# Source geometry: a 3500 mm-radius disk, 30 m from each heliostat along the
# sun vector, emitting toward the mirror. Fixed for every run traced so far.
SOURCE_DISK_RADIUS_MM = 3500.0
SOURCE_DIST_MM = 30000.0
SOURCE_POWER_W = 38484.5  # = 1000 W/m^2 * pi * 3.5^2 m^2

# Mirror aperture half-sizes (rect, mm): 5 m wide along u, 3 m tall along v.
MIRROR_HALF_X_MM = 2500.0
MIRROR_HALF_Y_MM = 1500.0

_SQRT3 = np.sqrt(3.0)
_SQRT6 = np.sqrt(6.0)

_DEFAULT_SAMPLER: SuperGaussSampler | None = None


def _default_sampler() -> SuperGaussSampler:
    global _DEFAULT_SAMPLER
    if _DEFAULT_SAMPLER is None:
        _DEFAULT_SAMPLER = SuperGaussSampler()
    return _DEFAULT_SAMPLER


def _sun_vector(solar_az_deg: float, solar_el_deg: float) -> np.ndarray:
    """Unit vector from ground toward the sun (azimuth is compass bearing)."""
    az = np.deg2rad(solar_az_deg)
    el = np.deg2rad(solar_el_deg)
    s = np.array(
        [
            np.cos(el) * np.cos(np.pi / 2 - az),
            np.cos(el) * np.sin(np.pi / 2 - az),
            np.sin(el),
        ]
    )
    return s / np.linalg.norm(s)


def _mirror_frame(rot_az_deg: float, rot_el_deg: float):
    """Mirror normal and in-plane basis from the two stored pointing angles.

    Inverts the ``rot_el = arcsin(n_z)``, ``rot_az = atan2(n_y, n_x)`` map
    and rebuilds ``u``/``v`` the same way the pointing solve does, so the
    frame here is the frame the figure coefficients were computed in.
    """
    az = np.deg2rad(rot_az_deg)
    el = np.deg2rad(rot_el_deg)
    n = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up = np.array([0.0, 0.0, 1.0])
    u = np.cross(up, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return n, u, v


def _zernike_sag_and_slopes(x, y, c3, c4, c5):
    """ANSI Z3..Z5 sag and its partial derivatives, normrad = 1 (x, y in mm)."""
    sag = (
        c3 * _SQRT6 * 2.0 * x * y
        + c4 * _SQRT3 * (2.0 * x * x + 2.0 * y * y - 1.0)
        + c5 * _SQRT6 * (x * x - y * y)
    )
    dsdx = c3 * _SQRT6 * 2.0 * y + c4 * _SQRT3 * 4.0 * x + c5 * _SQRT6 * 2.0 * x
    dsdy = c3 * _SQRT6 * 2.0 * x + c4 * _SQRT3 * 4.0 * y - c5 * _SQRT6 * 2.0 * y
    return sag, dsdx, dsdy


def trace_heliostat(
    x_mm: float,
    y_mm: float,
    rot_az_deg: float,
    rot_el_deg: float,
    c3: float,
    c4: float,
    c5: float,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary: Secondary,
    receiver: Receiver,
    n_rays: int,
    rng: np.random.Generator,
    sampler: SuperGaussSampler | None = None,
    source_disk_radius_mm: float = SOURCE_DISK_RADIUS_MM,
    source_power_w: float = SOURCE_POWER_W,
    return_paths: bool = False,
    return_secondary_hits: bool = False,
) -> dict:
    """Trace one heliostat at one instant; return receiver hits and loss counts.

    Sequential: source -> mirror -> secondary -> receiver. A ray that misses
    any surface's aperture is dropped there. Returns a dict with ``xy``
    ``(2, K)`` receiver coordinates inside the receiver's extent and the
    counter chain ``emitted / hit_mirror / hit_secondary / tip_rays /
    reached_receiver / in_window`` (an axicon secondary additionally sets
    ``hit_cone``, equal to ``hit_secondary``, for backward-compatible
    naming). With ``return_paths`` the dict also carries ``paths``: ``(4
    vertices, 3, K)`` world-mm polylines source point -> mirror hit ->
    secondary hit -> receiver hit for every surviving ray.

    ``source_disk_radius_mm`` and ``source_power_w`` default to the values
    every stored trace was generated with; passing different ones changes
    the emitting disk and the reported ``watts_per_ray``, nothing else.
    """
    if sampler is None:
        sampler = _default_sampler()

    # heliostat_shape (../geometry/heliostat.py) computes the figure in a
    # frame whose y and z axes are the mirror's own; the Zernike sag
    # evaluated below is written in a frame with y and z flipped relative to
    # that. Carrying the flip through the ANSI Z3/Z4/Z5 terms negates c4 and
    # c5 while leaving c3 unchanged -- a bookkeeping correction between two
    # frame conventions that agree on everything else, not new physics.
    c4 = -c4
    c5 = -c5

    s = _sun_vector(solar_az_deg, solar_el_deg)
    helio = np.array([x_mm, y_mm, 0.0])

    # Source disk basis: any orthonormal pair perpendicular to the sun
    # vector works -- positions are uniform and the angular law is
    # axisymmetric.
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), s)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    centre = helio + SOURCE_DIST_MM * s

    r = source_disk_radius_mm * np.sqrt(rng.random(n_rays))
    phi = 2.0 * np.pi * rng.random(n_rays)
    p = (
        centre[:, None] + e1[:, None] * (r * np.cos(phi)) + e2[:, None] * (r * np.sin(phi))
    )  # (3, N)

    theta = sampler.sample(n_rays, rng)
    psi = 2.0 * np.pi * rng.random(n_rays)
    d = -s[:, None] * np.cos(theta) + (
        e1[:, None] * np.cos(psi) + e2[:, None] * np.sin(psi)
    ) * np.sin(theta)

    counters = {"emitted": n_rays}

    # --- primary mirror -----------------------------------------------
    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)
    dn = d.T @ n  # (N,)
    du, dv = d.T @ u, d.T @ v
    t = ((helio - p.T) @ n) / dn
    # One Newton correction for the sag, then a final evaluation: the
    # figure is millimetres over a 3 m half-aperture, so a second
    # correction is sub-micron -- far below receiver storage quantisation.
    hit = p + d * t
    rel = hit.T - helio
    lx, ly = rel @ u, rel @ v
    sag, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
    t -= (rel @ n - sag) / (dn - dsdx * du - dsdy * dv)
    hit = p + d * t
    rel = hit.T - helio
    lx, ly = rel @ u, rel @ v
    ok = (np.abs(lx) <= MIRROR_HALF_X_MM) & (np.abs(ly) <= MIRROR_HALF_Y_MM) & (t > 0)
    counters["hit_mirror"] = int(ok.sum())
    hit, d, lx, ly = hit[:, ok], d[:, ok], lx[ok], ly[ok]

    _, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
    normal = n[:, None] - u[:, None] * dsdx - v[:, None] * dsdy
    normal /= np.linalg.norm(normal, axis=0)
    # In-place reflection: d -= 2 (d.n) n, no fresh (3, M) temporaries.
    dot = np.einsum("ij,ij->j", d, normal)
    dot *= 2.0
    d -= dot * normal

    # --- secondary -------------------------------------------------------
    pre, d, on_sec = secondary.redirect(hit, d, counters)

    # --- receiver ----------------------------------------------------------
    hit_mask, uv = receiver.intersect(pre, d)
    counters["reached_receiver"] = int(hit_mask.sum())
    (u0, u1), (v0, v1) = receiver.uv_extent()
    inside = (uv[0] >= u0) & (uv[0] <= u1) & (uv[1] >= v0) & (uv[1] <= v1)
    counters["in_window"] = int(inside.sum())

    result = {
        "xy": uv[:, inside],
        "counters": counters,
        "source_power_w": source_power_w,
        "watts_per_ray": source_power_w / n_rays if n_rays else 0.0,
    }
    if return_secondary_hits:
        # Plan (x, y) of every ray that struck the secondary, whether or not
        # it reached the receiver window -- for irradiance maps on the
        # secondary itself.
        result["secondary_xy"] = pre[:2].copy()
    if return_paths:
        src = p[:, ok][:, on_sec][:, hit_mask][:, inside]
        mir = hit[:, on_sec][:, hit_mask][:, inside]
        con = pre[:, hit_mask][:, inside]  # == mir when there is no secondary
        rec_uv = uv[:, inside]
        # World z of the receiver hit: exact for a flat window (the only
        # shape every fixture and existing strategy uses), NaN for a
        # receiver whose (u, v) does not embed a single z -- a curved
        # receiver's path plot needs its own 3-D reconstruction, out of
        # scope here.
        rec_z = getattr(receiver, "z_mm", float("nan"))
        rec = np.vstack([rec_uv[0], rec_uv[1], np.full(int(inside.sum()), rec_z)])
        result["paths"] = np.stack([src, mir, con, rec])
    return result
