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
without being modelled.

Edges — secondary rims, receiver-window borders, neighbour shadows and
neighbour blocking — are handled in angle space, where they actually
live: every sample's transmission is *measured* on a ``mask_nodes``² node
grid spanning the kernel support (one vectorised bundle for the whole
mirror), and partially-clipped kernels deposit through the resulting
raster, penumbra included. Samples whose kernel centre is itself lost at
a rim deposit their surviving mass directly at the passing nodes'
landing points instead of being dropped.

A 20 x 12 grid (240 samples) reproduces the smooth flux structure that
~10^5 Monte Carlo rays estimate; error falls with grid and node density,
not with luck.
"""

from __future__ import annotations

import numpy as np

from ..geometry.receiver import Receiver
from ..geometry.secondary import Secondary
from ..geometry.shading import _blocked_mask
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

        # Support 4.5 sigma: the order-2 super-Gaussian falls as
        # exp(-(theta^2/2sigma^2)^2), ~1e-45 there. A generous-looking 8
        # sigma support would make every footprint's bounding box overlap
        # the grid edge and disable per-kernel mass renormalisation.
        kernel = RadialKernel.from_profile(profile, support_rad=4.5 * sig)
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
    order: int = 1,
    occluders: list | None = None,
    shadow_body=None,
    mask_nodes: int = 16,
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

    ``order`` selects the local model of the optical map. ``1`` (five rays
    per sample) linearises it — the ultra-fast mode, leaving a curvature
    residual of ~1% of peak in the flux map. ``2`` (nine rays per sample)
    also measures the Hessian by finite differences and deposits through
    the quadratic map, removing that leading error term for roughly twice
    the cost — the fast-and-accurate mode.

    Edges and occlusion. A sample whose kernel straddles any boundary —
    secondary rim, receiver-window edge, a neighbour's shadow on the
    incoming side, a neighbour blocking the outgoing beam — gets a
    ``mask_nodes``² transmission raster in angle space, built by testing
    that many deterministic node rays from the same surface point
    (mini-trace through secondary and receiver; ray-vs-rectangle tests
    against ``occluders``; ``shadow_body.occludes`` for an opaque
    secondary body). The kernel is deposited through the raster, so edges
    are resolved with true penumbra rather than at sample granularity.
    ``occluders`` is this heliostat's neighbour list as
    :class:`~heliostat.geometry.shading.MirrorGeometry` rectangles; leave
    it ``None`` for store-bound sweeps, where the store contract applies
    occlusion as read-time scalars instead. Counter invariant:
    ``valid + masked + blocked + node_fallback + unresolved == samples``.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
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

    # Incoming directions, identical for every sample (the sun is at
    # infinity). Stencil legs: [chief, +e1, -e1, +e2, -e2] and, at order 2,
    # the four diagonals [++, +-, -+, --] that resolve the mixed Hessian.
    stencil = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    if order == 2:
        stencil += [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    offsets = np.array(stencil) * delta_rad
    legs = offsets.shape[0]
    dirs = -s[None, :] + offsets[:, 0, None] * e1[None, :] + offsets[:, 1, None] * e2[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)  # (5, 3)

    # Assemble all legs*M rays, stencil-major: ray index k*m + i is stencil
    # leg k at sample i. Every leg starts at the sample point itself.
    p_flat = np.concatenate([pts] * legs, axis=1)
    normal5 = np.concatenate([normal] * legs, axis=1)
    d_flat = np.repeat(dirs.T, m, axis=1)  # (3, legs*M), stencil-major

    dot = 2.0 * np.einsum("ij,ij->j", d_flat, normal5)
    d_ref = d_flat - dot * normal5

    counters = {"samples": int(m)}
    pre, d_out, on_sec = secondary.redirect(p_flat, d_ref, {})
    hit_mask, uv_hits = receiver.intersect(pre, d_out)

    # Map survival back to (stencil, sample): redirect returns the filtered
    # survivor bundle plus its boolean mask over the input rays; intersect
    # filters again within the survivors.
    alive = np.zeros(legs * m, dtype=bool)
    survivors = np.flatnonzero(on_sec)
    alive[survivors[hit_mask]] = True
    uv = np.full((2, legs * m), np.nan)
    uv[:, survivors[hit_mask]] = uv_hits

    alive = alive.reshape(legs, m)
    uv = uv.reshape(2, legs, m)

    chief_ok = alive[0]
    full_stencil = alive.all(axis=0)

    # Jacobian per sample, mm per rad: central differences where both
    # partners of an axis survived, one-sided where only one did. A sample
    # deposits when its chief ray and at least one partner per axis exist.
    jac_all = np.full((m, 2, 2), np.nan)
    can_jac = chief_ok.copy()
    for axis, (leg_p, leg_m) in enumerate([(1, 2), (3, 4)]):
        both = alive[leg_p] & alive[leg_m]
        pos = alive[leg_p] & ~alive[leg_m] & chief_ok
        neg = alive[leg_m] & ~alive[leg_p] & chief_ok
        col = np.full((2, m), np.nan)
        col[:, both] = (uv[:, leg_p, both] - uv[:, leg_m, both]) / (2.0 * delta_rad)
        col[:, pos] = (uv[:, leg_p, pos] - uv[:, 0, pos]) / delta_rad
        col[:, neg] = (uv[:, 0, neg] - uv[:, leg_m, neg]) / delta_rad
        jac_all[:, :, axis] = col.T
        can_jac &= alive[leg_p] | alive[leg_m]

    hess_all = None
    if order == 2:
        # Second differences where the full stencil survived; a partial
        # stencil falls back to the linear deposit for that sample.
        d2 = delta_rad * delta_rad
        hess_all = np.full((m, 2, 2, 2), np.nan)
        f = full_stencil
        chief = uv[:, 0]
        hess_all[f, :, 0, 0] = ((uv[:, 1, f] - 2.0 * chief[:, f] + uv[:, 2, f]) / d2).T
        hess_all[f, :, 1, 1] = ((uv[:, 3, f] - 2.0 * chief[:, f] + uv[:, 4, f]) / d2).T
        mixed = (uv[:, 5, f] - uv[:, 6, f] - uv[:, 7, f] + uv[:, 8, f]) / (4.0 * d2)
        hess_all[f, :, 0, 1] = mixed.T
        hess_all[f, :, 1, 0] = mixed.T

    # --- angular transmission, measured on a node grid for EVERY sample --
    # The stencil spans only ~delta_rad and cannot detect a boundary lying
    # elsewhere inside the kernel's ~10 mrad support, so transmission is
    # not detected — it is measured: mask_nodes² node rays per sample, one
    # vectorised bundle for the whole mirror, through every clip the sun
    # cone can meet (neighbour shadow on the way in; neighbour blocking,
    # secondary aperture and receiver window on the way out).
    support = kernel.support_rad
    occluders = occluders or []
    k = mask_nodes
    kk = k * k
    axis_nodes = np.linspace(-support, support, k)
    au, av = np.meshgrid(axis_nodes, axis_nodes)  # rows = second angular axis
    w_nodes = kernel.value(np.hypot(au, av).ravel())  # (k²,) kernel weight per node
    w_sum = w_nodes.sum()
    d_in_nodes = -s[:, None] + au.ravel()[None, :] * e1[:, None] + av.ravel()[None, :] * e2[:, None]
    d_in_nodes /= np.linalg.norm(d_in_nodes, axis=0, keepdims=True)  # (3, k²)

    node_ok = np.ones((m, kk), dtype=bool)
    pts_t = pts.T
    if occluders or shadow_body is not None:
        for j in range(kk):
            toward_sun_j = -d_in_nodes[:, j]
            if occluders:
                node_ok[:, j] &= ~_blocked_mask(pts_t, toward_sun_j, occluders)
            if shadow_body is not None:
                node_ok[:, j] &= ~shadow_body.occludes(pts_t, toward_sun_j)

    # Reflect every node direction at every sample's normal and push the
    # whole bundle through the optical chain at once. Sample-major layout:
    # ray index i*k² + j is node j of sample i.
    dots = normal.T @ d_in_nodes  # (m, k²)
    d_out_nodes = d_in_nodes[:, None, :] - 2.0 * dots[None, :, :] * normal[:, :, None]
    d_out_flat = d_out_nodes.reshape(3, m * kk)
    p_nodes = np.repeat(pts, kk, axis=1)  # (3, m*k²)
    if occluders:
        blocked_out = _blocked_mask(p_nodes.T, d_out_flat.T, occluders).reshape(m, kk)
        node_ok &= ~blocked_out
    pre_n, d_n, on_n = secondary.redirect(p_nodes, d_out_flat.copy(), {})
    hit_n, uv_n = receiver.intersect(pre_n, d_n)
    pass_out = np.zeros(m * kk, dtype=bool)
    uv_nodes = np.full((2, m * kk), np.nan)
    surv = np.flatnonzero(on_n)[hit_n]
    (u0, u1), (v0, v1) = receiver.uv_extent()
    in_ext = (uv_n[0] >= u0) & (uv_n[0] <= u1) & (uv_n[1] >= v0) & (uv_n[1] <= v1)
    pass_out[surv[in_ext]] = True
    uv_nodes[:, surv] = uv_n
    node_ok &= pass_out.reshape(m, kk)
    uv_nodes = uv_nodes.reshape(2, m, kk)

    frac = (node_ok @ w_nodes) / w_sum  # kernel-weighted transmitted fraction

    # --- classify and deposit --------------------------------------------
    cos_aoi = np.abs(normal.T @ s)  # incoming is -s; |normal . s| is cos(aoi)
    weights = STANDARD_IRRADIANCE_W_MM2 * cell_area_mm2 * cos_aoi

    n_u, n_v = flux_grid
    u_edges = np.linspace(u0, u1, n_u + 1)
    v_edges = np.linspace(v0, v1, n_v + 1)
    out = np.zeros((n_v, n_u))
    bin_area_mm2 = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])

    n_valid = n_masked = n_blocked = n_node_fallback = 0
    for idx in range(m):
        if frac[idx] < 1.0e-6:
            n_blocked += 1
            continue
        if chief_ok[idx] and can_jac[idx]:
            full_pass = frac[idx] > 1.0 - 1.0e-9
            hess_i = None
            if hess_all is not None and full_stencil[idx]:
                hess_i = hess_all[idx]
            deposit(
                out,
                u_edges,
                v_edges,
                uv[:, 0, idx],
                jac_all[idx],
                float(weights[idx]),
                kernel,
                hess=hess_i,
                mask=None if full_pass else node_ok[idx].astype(float).reshape(k, k),
            )
            if full_pass:
                n_valid += 1
            else:
                n_masked += 1
        else:
            # The chief ray (or a whole stencil axis) is lost — usually a
            # rim-straddling sample whose kernel centre misses the aperture
            # while part of its sun cone still passes. Deposit that passing
            # mass directly at the surviving nodes' landing points,
            # kernel-weighted. Locally granular, but these slivers carry
            # little power and would otherwise be dropped entirely.
            ok_j = node_ok[idx]
            share = weights[idx] * w_nodes[ok_j] / w_sum / bin_area_mm2
            iu = np.clip(((uv_nodes[0, idx, ok_j] - u0) // (u_edges[1] - u_edges[0])), 0, n_u - 1)
            iv = np.clip(((uv_nodes[1, idx, ok_j] - v0) // (v_edges[1] - v_edges[0])), 0, n_v - 1)
            np.add.at(out, (iv.astype(np.intp), iu.astype(np.intp)), share)
            n_node_fallback += 1

    counters["valid"] = n_valid
    counters["masked"] = n_masked
    counters["blocked"] = n_blocked
    counters["node_fallback"] = n_node_fallback
    counters["unresolved"] = 0  # retained for counter-invariant compatibility

    bin_area_mm2 = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])
    power_w = float(out.sum() * bin_area_mm2)
    return {
        "flux": out * 1.0e6,  # W/mm^2 -> W/m^2
        "u_edges": u_edges,
        "v_edges": v_edges,
        "power_w": power_w,
        "incident_power_w": float(weights.sum()),
        "counters": counters,
        "chief_uv": uv[:, 0],
        "jacobians": jac_all,
    }
