"""Per-heliostat sample generation, shared by all three deposit methods.

This module exists so the coefficient-space prototype (``hermite.py``,
``bspline.py``) and the binned ground truth (``binned.py``) all consume
*exactly the same* per-sample data -- so any difference the benchmark
measures between deposit methods is attributable to the deposit step alone,
never to a difference in what got traced.

:func:`trace_heliostat_samples` mirrors ``heliostat.trace.cone
.trace_heliostat_cone``'s plain-rectangular-mirror branch (``design=None``)
up through computing every per-sample quantity ``kernels.deposit`` would
need, then stops -- it does not touch a flux grid at all. It is built by
importing and reusing that module's own (private) helpers rather than
copying their logic, so a change to the real tracer's stencil/transmission
math is not silently re-derived here.

**Scope limitation, stated explicitly**: only the ``design=None`` (plain
rectangular mirror, no custom facet sketch) branch of ``trace_heliostat_cone``
is reproduced. The custom-facet-design branch is ~80 additional lines this
prototype does not need -- none of the B2 gates involve a custom facet
design, and the production default field (``field_645.csv``) uses plain
rectangular mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heliostat.geometry.secondary import AxiconSecondary, CassegrainSecondary, NoSecondary
from heliostat.trace.cone import (
    RING_PROBES,
    SECONDARY_MARGIN_FLOOR_MM,
    SECONDARY_MARGIN_FRAC,
    STANDARD_IRRADIANCE_W_MM2,
    WINDOW_MARGIN_FLOOR_MM,
    WINDOW_SAFETY_FACTOR,
    DISABLE_TRANSMISSION_SKIP,
    RadialKernel,
    _effective_support_rad,
    _jac_smax_mm,
    _reach_mm,
    _secondary_ring_clears,
)
from heliostat.trace.mc import (
    MIRROR_HALF_X_MM,
    MIRROR_HALF_Y_MM,
    _mirror_frame,
    _sun_vector,
    _zernike_sag_and_slopes,
)


@dataclass
class SampleBundle:
    """Everything ``kernels.deposit`` (or a coefficient-space alternative)
    needs for one heliostat's mirror-surface samples at one instant.

    Shapes: ``m`` samples, ``k = mask_nodes``, ``kk = k*k``.
    """

    m: int
    weights: np.ndarray  # (m,) watts this sample carries
    uv0: np.ndarray  # (2, m) chief-ray landing point, mm
    jac: np.ndarray  # (m, 2, 2) local Jacobian d(uv)/d(angle), mm/rad
    hess: np.ndarray | None  # (m, 2, 2, 2) or None (order=1)
    smax: np.ndarray  # (m,) top singular value of jac
    frac: np.ndarray  # (m,) transmitted fraction (angular-mask-weighted)
    node_ok: np.ndarray  # (m, kk) bool pass/fail per transmission-raster node
    w_nodes: np.ndarray  # (kk,) kernel weight per node, shared across samples
    axis_nodes: np.ndarray  # (k,) node axis in angle space, linspace(-support, support, k)
    uv_nodes: np.ndarray  # (2, m, kk) node landing points (for node-fallback)
    chief_ok: np.ndarray  # (m,) bool
    can_jac: np.ndarray  # (m,) bool
    kernel: RadialKernel
    u_edges: np.ndarray
    v_edges: np.ndarray
    wrap_u: bool
    counters: dict


def trace_heliostat_samples(
    x_mm: float,
    y_mm: float,
    rot_az_deg: float,
    rot_el_deg: float,
    c3: float,
    c4: float,
    c5: float,
    solar_az_deg: float,
    solar_el_deg: float,
    secondary,
    receiver,
    kernel: RadialKernel,
    grid: tuple[int, int] = (20, 12),
    delta_rad: float = 2.0e-4,
    order: int = 1,
    occluders: list | None = None,
    shadow_body=None,
    mask_nodes: int = 16,
) -> SampleBundle:
    """Same call signature (minus ``flux_grid``, ``design``) as
    ``trace_heliostat_cone`` -- reproduces its per-sample data without
    depositing. ``flux_grid``/``u_edges``/``v_edges`` are still computed
    (callers need them to build a grid) using the receiver's own
    ``uv_extent`` at whatever ``flux_grid`` shape the caller wants; pass the
    grid shape to :func:`heliostat_flux_grid` separately -- this function
    does not touch the flux grid at all, it just carries a *placeholder*
    ``u_edges``/``v_edges`` for the receiver's default 128x128 for
    convenience. See ``binned.py`` for how a caller actually uses these.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    if not getattr(receiver, "is_planar", True):
        order = 1
    c4 = -c4
    c5 = -c5

    s = _sun_vector(solar_az_deg, solar_el_deg)
    helio = np.array([x_mm, y_mm, 0.0])
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), s)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)

    n, u, v = _mirror_frame(rot_az_deg, rot_el_deg)

    n_x, n_y = grid
    gx = (np.arange(n_x) + 0.5) / n_x * 2.0 * MIRROR_HALF_X_MM - MIRROR_HALF_X_MM
    gy = (np.arange(n_y) + 0.5) / n_y * 2.0 * MIRROR_HALF_Y_MM - MIRROR_HALF_Y_MM
    lx, ly = (a.ravel() for a in np.meshgrid(gx, gy))
    m = lx.size
    cell_area_mm2 = (2.0 * MIRROR_HALF_X_MM / n_x) * (2.0 * MIRROR_HALF_Y_MM / n_y)
    area_w = np.full(m, cell_area_mm2)

    sag, dsdx, dsdy = _zernike_sag_and_slopes(lx, ly, c3, c4, c5)
    pts = helio[:, None] + u[:, None] * lx + v[:, None] * ly + n[:, None] * sag
    normal = n[:, None] - u[:, None] * dsdx - v[:, None] * dsdy
    normal /= np.linalg.norm(normal, axis=0)

    stencil = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    if order == 2:
        stencil += [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    offsets = np.array(stencil) * delta_rad
    legs = offsets.shape[0]
    dirs = -s[None, :] + offsets[:, 0, None] * e1[None, :] + offsets[:, 1, None] * e2[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    p_flat = np.concatenate([pts] * legs, axis=1)
    normal5 = np.concatenate([normal] * legs, axis=1)
    d_flat = np.repeat(dirs.T, m, axis=1)

    dot = 2.0 * np.einsum("ij,ij->j", d_flat, normal5)
    d_ref = d_flat - dot * normal5

    counters = {"samples": int(m)}
    pre, d_out, on_sec = secondary.redirect(p_flat, d_ref, {})
    hit_mask, uv_hits = receiver.intersect(pre, d_out)

    alive = np.zeros(legs * m, dtype=bool)
    survivors = np.flatnonzero(on_sec)
    alive[survivors[hit_mask]] = True
    uv = np.full((2, legs * m), np.nan)
    uv[:, survivors[hit_mask]] = uv_hits

    alive = alive.reshape(legs, m)
    uv = uv.reshape(2, legs, m)

    chief_ok = alive[0]
    full_stencil = alive.all(axis=0)

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

    smax_all = _jac_smax_mm(jac_all)

    hess_all = None
    if order == 2:
        d2 = delta_rad * delta_rad
        hess_all = np.full((m, 2, 2, 2), np.nan)
        f = full_stencil
        chief = uv[:, 0]
        hess_all[f, :, 0, 0] = ((uv[:, 1, f] - 2.0 * chief[:, f] + uv[:, 2, f]) / d2).T
        hess_all[f, :, 1, 1] = ((uv[:, 3, f] - 2.0 * chief[:, f] + uv[:, 4, f]) / d2).T
        mixed = (uv[:, 5, f] - uv[:, 6, f] - uv[:, 7, f] + uv[:, 8, f]) / (4.0 * d2)
        hess_all[f, :, 0, 1] = mixed.T
        hess_all[f, :, 1, 0] = mixed.T

    support = kernel.support_rad
    occluders = occluders or []
    (u0, u1), (v0, v1) = receiver.uv_extent()
    u_period_mm = getattr(receiver, "u_period_mm", None)
    k = mask_nodes
    kk = k * k
    axis_nodes = np.linspace(-support, support, k)
    au, av = np.meshgrid(axis_nodes, axis_nodes)
    w_nodes = kernel.value(np.hypot(au, av).ravel())
    w_sum = w_nodes.sum()
    d_in_nodes = -s[:, None] + au.ravel()[None, :] * e1[:, None] + av.ravel()[None, :] * e2[:, None]
    d_in_nodes /= np.linalg.norm(d_in_nodes, axis=0, keepdims=True)

    can_skip = np.zeros(m, dtype=bool)
    if not occluders and shadow_body is None and not DISABLE_TRANSMISSION_SKIP:
        skip_support = _effective_support_rad(kernel)
        with np.errstate(invalid="ignore"):
            reach = _reach_mm(smax_all, hess_all, full_stencil, skip_support)
            uv0 = uv[:, 0, :]
            margin = reach * WINDOW_SAFETY_FACTOR + WINDOW_MARGIN_FLOOR_MM
            v_clears = (uv0[1] - margin >= v0) & (uv0[1] + margin <= v1)
            u_clears = True if u_period_mm else (uv0[0] - margin >= u0) & (uv0[0] + margin <= u1)
            within_window = chief_ok & can_jac & u_clears & v_clears
        if isinstance(secondary, NoSecondary):
            can_skip = within_window
        elif isinstance(secondary, (AxiconSecondary, CassegrainSecondary)) and np.any(within_window):
            cand = np.flatnonzero(within_window)
            ap_margin = max(
                SECONDARY_MARGIN_FRAC * secondary.aperture_radius_mm, SECONDARY_MARGIN_FLOOR_MM
            )
            ring_ok = _secondary_ring_clears(
                pts[:, cand],
                normal[:, cand],
                s,
                e1,
                e2,
                skip_support,
                secondary,
                ap_margin,
                RING_PROBES,
            )
            can_skip[cand] = ring_ok

    counters["transmission_skipped"] = int(can_skip.sum())
    need = ~can_skip
    frac = np.ones(m)
    node_ok = np.ones((m, kk), dtype=bool)
    uv_nodes = np.full((2, m, kk), np.nan)

    if np.any(need):
        idxn = np.flatnonzero(need)
        pts_n = pts[:, idxn]
        normal_n = normal[:, idxn]
        mn = idxn.size

        node_ok_n = np.ones((mn, kk), dtype=bool)
        pts_t = pts_n.T
        if occluders or shadow_body is not None:
            from heliostat.geometry.shading import _blocked_mask

            for j in range(kk):
                toward_sun_j = -d_in_nodes[:, j]
                if occluders:
                    node_ok_n[:, j] &= ~_blocked_mask(pts_t, toward_sun_j, occluders)
                if shadow_body is not None:
                    node_ok_n[:, j] &= ~shadow_body.occludes(pts_t, toward_sun_j)

        dots = normal_n.T @ d_in_nodes
        d_out_nodes = d_in_nodes[:, None, :] - 2.0 * dots[None, :, :] * normal_n[:, :, None]
        d_out_flat = d_out_nodes.reshape(3, mn * kk)
        p_nodes = np.repeat(pts_n, kk, axis=1)
        if occluders:
            from heliostat.geometry.shading import _blocked_mask

            blocked_out = _blocked_mask(p_nodes.T, d_out_flat.T, occluders).reshape(mn, kk)
            node_ok_n &= ~blocked_out
        pre_n, d_n, on_n = secondary.redirect(p_nodes, d_out_flat.copy(), {})
        hit_n, uv_n = receiver.intersect(pre_n, d_n)
        pass_out = np.zeros(mn * kk, dtype=bool)
        uv_nodes_n = np.full((2, mn * kk), np.nan)
        surv = np.flatnonzero(on_n)[hit_n]
        u_test = uv_n[0] if not u_period_mm else u0 + np.mod(uv_n[0] - u0, u_period_mm)
        in_ext = (u_test >= u0) & (u_test <= u1) & (uv_n[1] >= v0) & (uv_n[1] <= v1)
        pass_out[surv[in_ext]] = True
        uv_nodes_n[:, surv] = uv_n
        node_ok_n &= pass_out.reshape(mn, kk)
        uv_nodes_n = uv_nodes_n.reshape(2, mn, kk)

        frac[idxn] = (node_ok_n @ w_nodes) / w_sum
        node_ok[idxn] = node_ok_n
        uv_nodes[:, idxn] = uv_nodes_n

    cos_aoi = np.abs(normal.T @ s)
    weights = STANDARD_IRRADIANCE_W_MM2 * area_w * cos_aoi
    wrap_u = bool(u_period_mm)

    return SampleBundle(
        m=m,
        weights=weights,
        uv0=uv[:, 0, :],
        jac=jac_all,
        hess=hess_all,
        smax=smax_all,
        frac=frac,
        node_ok=node_ok,
        w_nodes=w_nodes,
        axis_nodes=axis_nodes,
        uv_nodes=uv_nodes,
        chief_ok=chief_ok,
        can_jac=can_jac,
        kernel=kernel,
        u_edges=np.array([u0, u1]),  # placeholder extent; see flux_grid_edges()
        v_edges=np.array([v0, v1]),
        wrap_u=wrap_u,
        counters=counters,
    )


def flux_grid_edges(receiver, flux_grid: tuple[int, int]):
    """``(u_edges, v_edges)`` for ``receiver`` at the requested grid shape --
    the same construction ``trace_heliostat_cone`` uses for its output grid.
    """
    (u0, u1), (v0, v1) = receiver.uv_extent()
    n_u, n_v = flux_grid
    return np.linspace(u0, u1, n_u + 1), np.linspace(v0, v1, n_v + 1)
