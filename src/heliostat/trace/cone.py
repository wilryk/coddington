"""Cone-optics trace: deterministic rays plus analytic sunshape deposition.

The Monte Carlo backend's noise comes from sampling the sun's angular
distribution with random rays; the distribution itself is known exactly.
This backend samples the *mirror surface* on a small deterministic grid
instead. Each sample point contributes five rays: a chief ray from the
sun's centre, and four rays perturbed by a small angle along two axes
perpendicular to the sun vector. The chief ray's landing point gives the
kernel centre; the four companions give the local Jacobian
``d(receiver uv)/d(source angle)`` by central differences. The sun's
angular density — optionally broadened by Gaussian slope and tracking
error — is then laid down through that Jacobian by
:func:`heliostat.trace.kernels.deposit`.

Because the Jacobian is *measured* through the real optical chain, every
geometric effect the Monte Carlo trace captures — off-axis astigmatism,
cone folding, hyperboloid magnification, receiver obliquity — is inherited
without being modelled. The two deliberate approximations, both geometric
and both reported in the counters:

* apertures act at sample granularity: a sample whose chief ray survives
  deposits its whole kernel, one whose chief ray is lost deposits nothing,
  so aperture edges are resolved to the mirror-grid cell size;
* the optical map is linearised across one kernel footprint. Where the map
  folds (an axicon tip) the five-ray stencil straddles the fold and the
  sample is flagged ``unresolved`` rather than silently mis-deposited.

A 20 x 12 grid (240 samples, 1200 deterministic rays) reproduces the
smooth flux structure that ~10^5 Monte Carlo rays estimate; error falls
with grid density, not with luck.
"""

from __future__ import annotations

import numpy as np

from ..geometry.receiver import Receiver
from ..geometry.secondary import Secondary
from .kernels import RadialKernel, deposit
from .mc import (
    MIRROR_HALF_X_MM,
    MIRROR_HALF_Y_MM,
    _mirror_frame,
    _sun_vector,
    _zernike_sag_and_slopes,
)
from .samplers import BUIE_LIMB_MRAD, SUPER_GAUSS_ORDER, SUPER_GAUSS_SIGMA_RAD

STANDARD_IRRADIANCE_W_MM2 = 1000.0e-6  # 1000 W/m^2, the trace normalisation


def sunshape_kernel(
    source_model: str = "super_gauss",
    slope_error_mrad: float = 0.0,
    tracking_error_mrad: float = 0.0,
) -> RadialKernel:
    """The angular kernel for a named sunshape, optionally error-broadened.

    Profiles are the same pinned forms the Monte Carlo samplers draw from.
    Mirror slope error deflects a reflected ray by twice the surface tilt,
    hence the factor 2 on ``slope_error_mrad``.
    """
    if source_model == "super_gauss":
        sig = SUPER_GAUSS_SIGMA_RAD

        def profile(t):
            return np.exp(-((t**2 / (2.0 * sig**2)) ** SUPER_GAUSS_ORDER))

        kernel = RadialKernel.from_profile(profile, support_rad=8.0 * sig)
    elif source_model == "buie":
        limb = BUIE_LIMB_MRAD * 1e-3

        def profile(t):
            t_mrad = np.minimum(t, limb) * 1e3
            return np.where(t <= limb, np.cos(0.326 * t_mrad) / np.cos(0.308 * t_mrad), 0.0)

        kernel = RadialKernel.from_profile(profile, support_rad=limb)
    else:
        raise ValueError(f"unknown source_model {source_model!r}")

    broadening = np.hypot(2.0 * slope_error_mrad, tracking_error_mrad) * 1e-3
    return kernel.convolve_gaussian(broadening) if broadening > 0 else kernel


def trace_heliostat_cone(
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
    kernel: RadialKernel,
    grid: tuple[int, int] = (20, 12),
    flux_grid: tuple[int, int] = (128, 128),
    delta_rad: float = 2.0e-4,
) -> dict:
    """Cone-optics trace of one heliostat at one instant.

    Same pointing, figure, secondary and receiver conventions as
    :func:`heliostat.trace.mc.trace_heliostat` (including the c4/c5 frame
    correction), so results are directly comparable at identical inputs.
    Returns a dict with ``flux`` (``(n_v, n_u)`` W/m² at the 1000 W/m²
    trace normalisation), ``power_w`` (its integral over the receiver
    window), ``incident_power_w`` (cosine-weighted power arriving on the
    mirror), and a counter chain ``samples / valid / blocked / unresolved``
    over mirror sample cells.

    ``delta_rad`` is the finite-difference probe angle; it must be small
    against the optics' scale of nonlinearity but large enough that
    receiver-position differences dominate roundoff — anything within an
    order of magnitude of the default works for metre-scale optics.
    """
    # Same frame bookkeeping as the MC trace: the figure coefficients
    # arrive in a convention whose y/z flip negates c4 and c5 here.
    c4 = -c4
    c5 = -c5

    s = _sun_vector(solar_az_deg, solar_el_deg)
    helio = np.array([x_mm, y_mm, 0.0])
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), s)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)

    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)

    # Mirror-surface sample grid: cell centres, equal areas.
    n_x, n_y = grid
    gx = (np.arange(n_x) + 0.5) / n_x * 2.0 * MIRROR_HALF_X_MM - MIRROR_HALF_X_MM
    gy = (np.arange(n_y) + 0.5) / n_y * 2.0 * MIRROR_HALF_Y_MM - MIRROR_HALF_Y_MM
    lx, ly = (a.ravel() for a in np.meshgrid(gx, gy))
    m = lx.size
    cell_area_mm2 = (2.0 * MIRROR_HALF_X_MM / n_x) * (2.0 * MIRROR_HALF_Y_MM / n_y)

    sag, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
    pts = helio[:, None] + u[:, None] * lx + v[:, None] * ly + n[:, None] * sag  # (3, M)
    normal = n[:, None] - u[:, None] * dsdx - v[:, None] * dsdy
    normal /= np.linalg.norm(normal, axis=0)

    # Five incoming directions, identical for every sample (the sun is at
    # infinity). Stencil order: [chief, +e1, -e1, +e2, -e2].
    offsets = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) * delta_rad
    dirs = -s[None, :] + offsets[:, 0, None] * e1[None, :] + offsets[:, 1, None] * e2[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # (5, 3)

    # Assemble all 5M rays, stencil-major: ray index k*m + i is stencil leg
    # k at sample i. Every leg starts at the sample point itself.
    p_flat = np.concatenate([pts] * 5, axis=1)
    normal5 = np.concatenate([normal] * 5, axis=1)
    d_flat = np.repeat(dirs.T, m, axis=1)  # (3, 5M), matching stencil-major order

    dot = 2.0 * np.einsum("ij,ij->j", d_flat, normal5)
    d_ref = d_flat - dot * normal5

    counters = {"samples": int(m)}
    pre, d_out, on_sec = secondary.redirect(p_flat, d_ref, {})
    hit_mask, uv_hits = receiver.intersect(pre, d_out)

    # Map survival back to (stencil, sample): redirect returns the filtered
    # survivor bundle plus its boolean mask over the input rays; intersect
    # filters again within the survivors.
    alive = np.zeros(5 * m, dtype=bool)
    survivors = np.flatnonzero(on_sec)
    alive[survivors[hit_mask]] = True
    uv = np.full((2, 5 * m), np.nan)
    uv[:, survivors[hit_mask]] = uv_hits

    alive = alive.reshape(5, m)
    uv = uv.reshape(2, 5, m)

    chief_ok = alive[0]
    stencil_ok = alive.all(axis=0)
    counters["blocked"] = int((~chief_ok).sum())
    counters["unresolved"] = int((chief_ok & ~stencil_ok).sum())
    counters["valid"] = int(stencil_ok.sum())

    # Central-difference Jacobian, mm per rad, per valid sample.
    sel = stencil_ok
    n_sel = int(sel.sum())
    jac = np.empty((n_sel, 2, 2))
    jac[:, :, 0] = ((uv[:, 1, sel] - uv[:, 2, sel]) / (2.0 * delta_rad)).T
    jac[:, :, 1] = ((uv[:, 3, sel] - uv[:, 4, sel]) / (2.0 * delta_rad)).T

    cos_aoi = np.abs(normal.T @ s)  # incoming is -s; |normal . s| is cos(aoi)
    weights = STANDARD_IRRADIANCE_W_MM2 * cell_area_mm2 * cos_aoi

    (u0, u1), (v0, v1) = receiver.uv_extent()
    n_u, n_v = flux_grid
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    out = np.zeros((n_v, n_u))

    uv0_sel = uv[:, 0, sel]
    w_sel = weights[sel]
    for i in range(n_sel):
        deposit(out, u_edges, v_edges, uv0_sel[:, i], jac[i], w_sel[i], kernel)

    bin_area_mm2 = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])
    power_w = float(out.sum() * bin_area_mm2)
    return {
        "flux": out * 1.0e6,  # W/mm^2 -> W/m^2
        "u_edges": u_edges,
        "v_edges": v_edges,
        "power_w": power_w,
        "incident_power_w": float(weights.sum()),
        "counters": counters,
        "chief_uv": uv0_sel,
        "jacobians": jac,
    }
