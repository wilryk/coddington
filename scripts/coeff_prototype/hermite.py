"""Hermite-Gauss coefficient accumulation (DELSOL3 precedent).

**The representation.** Every sample deposits, in the exact-deposit picture
(``kernels.deposit``), a density ``weight * k(|alpha|) / |det J|`` at receiver
point ``uv``, where ``alpha = J^-1 (uv - uv0)`` is the angular preimage. This
module represents ``k(|alpha|)`` -- a fixed, radially-symmetric function,
the *same* for every sample in the field (the kernel is shared) -- as a
truncated 2-D Hermite-function series in the dimensionless, kernel-width-
normalised coordinate ``xi = alpha / sigma``:

    k(sigma * |xi|)  ~=  sum_{n+m<=ORDER}  c[n,m] * psi_n(xi_u) * psi_m(xi_v)

with ``psi_n`` the standard quantum-harmonic-oscillator (physicists')
Hermite *functions*

    psi_n(x) = H_n(x) * exp(-x^2/2) / sqrt(2^n * n! * sqrt(pi))

which are exactly orthonormal on all of R (``integral psi_n psi_m dx =
delta_nm``), so a coefficient is a plain inner product:

    c[n,m] = integral integral  g(xi) * psi_n(xi_u) * psi_m(xi_v)  dxi_u dxi_v

**Unmasked samples** (the common case, no clipping/occlusion at this sample):
``g = k(sigma|xi|)`` is the SAME function for every sample, so its
coefficients are computed ONCE, at kernel-construction time, on a fine
quadrature grid (129x129) spanning the kernel's support -- not per sample,
not per heliostat. A sample's per-sample "accumulation" cost is then O(1):
store ``(uv0, J^-1, weight/|det J|)`` and a reference to this one shared
coefficient vector.

**Masked/clipped samples**: the local transmission raster
(``bundle.axis_nodes``, ``bundle.w_nodes``, ``bundle.node_ok[idx]``) --
already computed by ``sampling.py``/the real tracer, at no extra tracing
cost -- gives ``k(theta_j) * pass_j`` at a k x k grid of angular nodes. The
SAME projection formula, evaluated as a discrete sum over those k^2 nodes,
gives that one sample's own coefficients: ``O(k^2 * n_terms)``, independent
of the sample's footprint size in flux-grid cells. This is the mechanism
that is supposed to reproduce clipping -- and the mechanism whose failure
mode (Gibbs ringing at a hard edge, from a low truncation order) is exactly
what the B2 gates are designed to expose. See REPORT.md for how badly.

**Renormalisation.** Exactly mirroring ``kernels.deposit``'s own convention
(an unclipped footprint's raw deposit is rescaled so its integral matches
``weight`` exactly, removing bin-centre-evaluation error): each record's
amplitude is rescaled so the *truncated* series' own analytic-in-xi-space
integral matches ``target = weight`` (unmasked) or ``weight * frac``
(masked) exactly, removing the truncation-order's own conservation error at
evaluation time. The raw (pre-renormalisation) truncation error is also
computed and reported separately -- see :func:`HermiteBasis.reconstructed_mass`.

**Node-fallback samples** (chief ray lost at a rim): handled by direct point
deposit, identically to ``binned.py`` -- not through the coefficient
machinery at all. This is a shared simplification across all three methods
for a rare edge case, so it cannot bias the comparison; see REPORT.md.

**Evaluation** happens once, in :func:`evaluate_hermite`, over a local
bounding box per record sized by the same ``reach = smax * support_rad``
bound ``kernels.deposit`` itself uses -- so the evaluation phase is
deliberately kept in the same complexity class as the binned method's own
per-sample cost. The *accumulation* phase is where the O(1)-regardless-of-
footprint-size claim is tested; see ``run_benchmark.py``'s timing split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermvander

from .sampling import SampleBundle

DEFAULT_ORDER = 6  # total degree n+m <= 6 -> 28 terms; see REPORT.md for order=4


def _n_terms(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def _term_pairs(order: int) -> list[tuple[int, int]]:
    return [(n, m) for n in range(order + 1) for m in range(order + 1 - n)]


def _psi_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """``(len(x), order+1)`` matrix of ``psi_0(x)..psi_order(x)``."""
    h = hermvander(x, order)  # physicists' H_0..H_order at each x
    n = np.arange(order + 1)
    norm = np.sqrt((2.0**n) * np.array([math.factorial(int(i)) for i in n]) * np.sqrt(np.pi))
    gauss = np.exp(-0.5 * x * x)
    return h * gauss[:, None] / norm[None, :]


def _marginal_integrals(xi_axis: np.ndarray, order: int) -> np.ndarray:
    """``integral psi_n(xi) dxi`` over R, approximated on ``xi_axis`` (a
    uniform 1-D grid) by the same discrete-sum rule used to build the
    coefficients on that same grid -- so the mass-reconstruction check
    below is self-consistent with however the coefficients were computed,
    whether from the fine kernel-quadrature grid or the coarse mask-node
    raster.
    """
    psi = _psi_matrix(xi_axis, order)  # (len(axis), order+1)
    dxi = xi_axis[1] - xi_axis[0]
    return psi.sum(axis=0) * dxi  # (order+1,)


@dataclass
class HermiteBasis:
    """The kernel's own Hermite-Gauss decomposition, computed once, plus the
    reusable machinery masked samples need for their own per-sample
    projection.
    """

    sigma: float
    order: int
    terms: list[tuple[int, int]]
    shared_coeffs: np.ndarray  # (n_terms,) -- unmasked samples' coefficients
    shared_I: np.ndarray  # (order+1,) marginal integrals on the fine grid
    fine_axis: np.ndarray  # the fine xi-grid the shared coeffs were built on

    @classmethod
    def build(cls, kernel, order: int = DEFAULT_ORDER, n_fine: int = 129) -> "HermiteBasis":
        sigma = kernel.rms_radius_rad() / np.sqrt(2.0)
        support_xi = kernel.support_rad / sigma
        fine_axis = np.linspace(-support_xi, support_xi, n_fine)
        au, av = np.meshgrid(fine_axis, fine_axis)
        g = kernel.value(sigma * np.hypot(au, av))
        terms = _term_pairs(order)
        psi_u = _psi_matrix(fine_axis, order)  # (n_fine, order+1)
        psi_v = psi_u
        dxi = fine_axis[1] - fine_axis[0]
        coeffs = np.empty(len(terms))
        for i, (n, m) in enumerate(terms):
            integrand = g * psi_u[:, n][None, :] * psi_v[:, m][:, None]
            coeffs[i] = integrand.sum() * dxi * dxi
        shared_I = _marginal_integrals(fine_axis, order)
        return cls(
            sigma=sigma, order=order, terms=terms, shared_coeffs=coeffs,
            shared_I=shared_I, fine_axis=fine_axis,
        )

    def reconstructed_mass(self, coeffs: np.ndarray, marginal_I: np.ndarray) -> float:
        """``sum_{n,m} c[n,m] * I_n * I_m`` -- the truncated series' own
        analytic-in-xi integral. Equals the true ``1/sigma^2`` (unmasked) or
        ``frac/sigma^2`` (masked) only in the limit of no truncation; the
        gap is exactly the truncation's own power error, reported in
        REPORT.md before any renormalisation is applied.
        """
        total = 0.0
        for i, (n, m) in enumerate(self.terms):
            total += coeffs[i] * marginal_I[n] * marginal_I[m]
        return total

    def project_masked(self, axis_nodes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample coefficients from a k x k raster: ``values`` is the
        ``(k*k,)`` array ``kernel.value(theta) * node_ok`` (already
        available from ``SampleBundle``), ``axis_nodes`` the ``(k,)`` node
        axis in angle space (radians). Returns ``(coeffs, marginal_I)`` --
        the marginal integrals differ from the shared ones because they are
        computed on the coarser node grid, not the fine kernel-quadrature
        grid, so they travel with these coefficients rather than reusing
        ``self.shared_I``.
        """
        k = axis_nodes.size
        xi_axis = axis_nodes / self.sigma
        psi = _psi_matrix(xi_axis, self.order)  # (k, order+1)
        values_2d = values.reshape(k, k)  # rows = v-axis, cols = u-axis (meshgrid convention)
        dxi = xi_axis[1] - xi_axis[0]
        coeffs = np.empty(len(self.terms))
        for i, (n, m) in enumerate(self.terms):
            integrand = values_2d * psi[:, n][None, :] * psi[:, m][:, None]
            coeffs[i] = integrand.sum() * dxi * dxi
        marginal_I = _marginal_integrals(xi_axis, self.order)
        return coeffs, marginal_I


@dataclass
class HermiteRecord:
    uv0: np.ndarray  # (2,)
    jinv: np.ndarray  # (2,2), J^-1
    amplitude: float  # target / (sigma^2 * reconstructed_mass)
    coeffs: np.ndarray  # (n_terms,) -- may be the shared array (unmasked) or own (masked)
    reach_mm: float  # bounding-box half-width, mm


def accumulate_hermite(
    bundle: SampleBundle, basis: HermiteBasis
) -> tuple[list[HermiteRecord], list[tuple[np.ndarray, np.ndarray]]]:
    """Per-sample accumulation: O(1) for unmasked samples, O(k^2*n_terms)
    for masked ones. Returns ``(records, fallback_points)`` where
    ``fallback_points`` is a list of ``(iu_frac_like, share)`` handled
    identically to ``binned.py``'s node-fallback path -- see module
    docstring.
    """
    records: list[HermiteRecord] = []
    fallback: list[tuple[np.ndarray, np.ndarray]] = []
    sigma2 = basis.sigma * basis.sigma

    for idx in range(bundle.m):
        if bundle.frac[idx] < 1.0e-6:
            continue
        if bundle.chief_ok[idx] and bundle.can_jac[idx]:
            jac = bundle.jac[idx]
            det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
            jinv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det
            full_pass = bundle.frac[idx] > 1.0 - 1.0e-9
            target = float(bundle.weights[idx]) * (1.0 if full_pass else float(bundle.frac[idx]))

            if full_pass:
                coeffs = basis.shared_coeffs
                marginal_I = basis.shared_I
            else:
                values = bundle.kernel.value(
                    np.hypot(
                        *np.meshgrid(bundle.axis_nodes, bundle.axis_nodes)
                    ).ravel()
                ) * bundle.node_ok[idx].astype(float)
                coeffs, marginal_I = basis.project_masked(bundle.axis_nodes, values)

            recon = basis.reconstructed_mass(coeffs, marginal_I)
            if abs(recon) < 1.0e-300 or target <= 0:
                continue
            # density(uv) = weight/|det J| * g(xi); integrating over uv picks
            # up an extra |det J| * sigma^2 from d(uv) = |det J| sigma^2 dxi
            # (uv = uv0 + J @ (sigma*xi)) -- both factors belong in the
            # renormalisation, not just sigma^2 (see module docstring).
            amplitude = target / (abs(det) * sigma2 * recon)
            smax = float(bundle.smax[idx])
            reach = smax * bundle.kernel.support_rad
            records.append(
                HermiteRecord(uv0=bundle.uv0[:, idx], jinv=jinv, amplitude=amplitude,
                              coeffs=coeffs, reach_mm=reach)
            )
        else:
            ok_j = bundle.node_ok[idx]
            w_sum = bundle.w_nodes.sum()
            share = bundle.weights[idx] * bundle.w_nodes[ok_j] / w_sum
            fallback.append((bundle.uv_nodes[:, idx, ok_j], share))

    return records, fallback


def evaluate_hermite(
    records: list[HermiteRecord],
    fallback: list[tuple[np.ndarray, np.ndarray]],
    basis: HermiteBasis,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    wrap_u: bool,
) -> np.ndarray:
    """One pass over every accumulated record, evaluating its Hermite
    series onto a local bounding box of ``(u_edges, v_edges)`` -- the field
    map, in weight/mm^2 (matching ``binned.deposit_binned``'s convention).
    """
    n_u = u_edges.size - 1
    n_v = v_edges.size - 1
    out = np.zeros((n_v, n_u))
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    order = basis.order
    terms = basis.terms

    for rec in records:
        i0_raw = int(np.floor((rec.uv0[0] - rec.reach_mm - u_edges[0]) / du))
        i1_raw = int(np.ceil((rec.uv0[0] + rec.reach_mm - u_edges[0]) / du))
        j0_raw = int(np.floor((rec.uv0[1] - rec.reach_mm - v_edges[0]) / dv))
        j1_raw = int(np.ceil((rec.uv0[1] + rec.reach_mm - v_edges[0]) / dv))
        i0, i1 = (i0_raw, i1_raw) if wrap_u else (max(0, i0_raw), min(n_u, i1_raw))
        j0, j1 = max(0, j0_raw), min(n_v, j1_raw)
        if i0 >= i1 or j0 >= j1:
            continue

        u_mid = (
            u_edges[0] + (np.arange(i0, i1) + 0.5) * du
            if wrap_u
            else 0.5 * (u_edges[i0 : i1 + 1][:-1] + u_edges[i0 : i1 + 1][1:])
        )
        v_mid = 0.5 * (v_edges[j0 : j1 + 1][:-1] + v_edges[j0 : j1 + 1][1:])
        duv_u = u_mid[None, :] - rec.uv0[0]
        duv_v = v_mid[:, None] - rec.uv0[1]
        alpha_u = rec.jinv[0, 0] * duv_u + rec.jinv[0, 1] * duv_v
        alpha_v = rec.jinv[1, 0] * duv_u + rec.jinv[1, 1] * duv_v
        xi_u = (alpha_u / basis.sigma).ravel()
        xi_v = (alpha_v / basis.sigma).ravel()

        psi_u = _psi_matrix(xi_u, order)  # (npix, order+1)
        psi_v = _psi_matrix(xi_v, order)
        patch = np.zeros(xi_u.shape)
        for c, (n, m) in zip(rec.coeffs, terms):
            if c == 0.0:
                continue
            patch += c * psi_u[:, n] * psi_v[:, m]
        patch = patch.reshape(j1 - j0, i1 - i0) * rec.amplitude

        if wrap_u:
            np.add.at(out, (slice(j0, j1), np.arange(i0, i1) % n_u), patch)
        else:
            out[j0:j1, i0:i1] += patch

    for uv_pts, share in fallback:
        iu = np.clip(((uv_pts[0] - u_edges[0]) // du), 0, n_u - 1)
        iv = np.clip(((uv_pts[1] - v_edges[0]) // dv), 0, n_v - 1)
        np.add.at(out, (iv.astype(np.intp), iu.astype(np.intp)), share / (du * dv))

    return out
