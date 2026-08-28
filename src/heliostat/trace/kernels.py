"""Analytic kernel deposition for the cone-optics backend.

The Monte Carlo tracer's noise comes entirely from sampling the sun's
angular distribution with random rays. That distribution is known in
closed form, so the cone-optics backend replaces photon statistics with
geometry: a modest deterministic grid of points samples the *mirror
surface*, each point's central ray is traced through the optical chain,
and the sun's angular density is deposited around the landing point —
mapped through the local Jacobian ``J = d(receiver uv) / d(source angle)``
that the tracer measures with a few auxiliary rays. A few hundred sample
points reproduce what ~10^5 random rays estimate, with zero shot noise.

This module owns the two purely mathematical ingredients:

* :class:`RadialKernel` — a tabulated radially-symmetric angular density
  ``k(theta)`` (normalised so its 2-D plane integral is 1), built from a
  sunshape profile and optionally convolved with a Gaussian for slope and
  tracking error;
* :func:`deposit` — laying ``weight * k`` down on a flux grid after the
  linear map ``uv = uv0 + J @ alpha``, which turns the radial profile into
  the correct elliptical footprint with density ``k(J^-1 du) / |det J|``.

Everything here is exact for a locally-linear optical map; the error of
the backend is set by mirror-surface sampling density and by how far the
true map deviates from its linearisation across one kernel width — both
geometric quantities, not statistical ones.

Units: angles in radians, receiver coordinates in mm, so ``J`` carries
mm/rad and deposited values are ``weight`` per mm². Callers convert to
W/m² (or count-equivalents) themselves.
"""

from __future__ import annotations

import numpy as np


class RadialKernel:
    """Tabulated radially-symmetric 2-D angular density.

    ``theta`` is angular radius from the beam centre (rad). The stored
    table is the density ``k(theta)`` per unit solid angle (rad^-2),
    normalised so ``2*pi * integral(k(t) * t dt) == 1``.
    """

    def __init__(self, theta_rad: np.ndarray, density: np.ndarray):
        theta_rad = np.asarray(theta_rad, dtype=float)
        density = np.asarray(density, dtype=float)
        if theta_rad.ndim != 1 or theta_rad.shape != density.shape:
            raise ValueError("theta_rad and density must be matching 1-D arrays")
        if theta_rad[0] != 0.0 or np.any(np.diff(theta_rad) <= 0):
            raise ValueError("theta_rad must start at 0 and increase strictly")
        norm = 2.0 * np.pi * np.trapz(density * theta_rad, theta_rad)
        if norm <= 0:
            raise ValueError("kernel has non-positive total weight")
        self.theta_rad = theta_rad
        self.density = density / norm

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_profile(cls, profile, support_rad: float, n: int = 512) -> "RadialKernel":
        """Tabulate an arbitrary radial profile out to ``support_rad``."""
        theta = np.linspace(0.0, support_rad, n)
        return cls(theta, np.asarray(profile(theta), dtype=float))

    @classmethod
    def gaussian(cls, sigma_rad: float, n_sigma: float = 6.0, n: int = 512) -> "RadialKernel":
        theta = np.linspace(0.0, n_sigma * sigma_rad, n)
        return cls(theta, np.exp(-0.5 * (theta / sigma_rad) ** 2))

    # -- properties ------------------------------------------------------

    @property
    def support_rad(self) -> float:
        """Angular radius beyond which the kernel is treated as zero."""
        return float(self.theta_rad[-1])

    def rms_radius_rad(self) -> float:
        """Root-mean-square angular radius, ``sqrt(E[theta^2])``."""
        m2 = 2.0 * np.pi * np.trapz(self.density * self.theta_rad**3, self.theta_rad)
        return float(np.sqrt(m2))

    def value(self, theta: np.ndarray) -> np.ndarray:
        """Density at angular radius ``theta`` (0 outside the support)."""
        return np.interp(theta, self.theta_rad, self.density, left=self.density[0], right=0.0)

    # -- operations ------------------------------------------------------

    def convolve_gaussian(self, sigma_rad: float, n: int | None = None) -> "RadialKernel":
        """Convolve with an isotropic 2-D Gaussian (slope/tracking error).

        The convolution of two radial densities is radial, and for a
        Gaussian smoother it has the closed quadrature form

            (k * g)(r) = integral  k(t) * t/sigma^2 * exp(-(r^2+t^2)/(2 sigma^2))
                                   * I0(r t / sigma^2)  dt

        with ``I0`` the modified Bessel function. Evaluated with the
        exponentially-scaled ``i0e`` so large arguments stay finite.
        """
        if sigma_rad <= 0:
            return self
        from scipy.special import i0e

        if n is None:
            n = self.theta_rad.size
        support = self.support_rad + 6.0 * sigma_rad
        r = np.linspace(0.0, support, n)
        t = self.theta_rad
        # i0e(x) = exp(-x) I0(x)  =>  exp(-(r^2+t^2)/2s^2) I0(rt/s^2)
        #        = exp(-(r-t)^2 / 2s^2) * i0e(r t / s^2)
        rr = r[:, None]
        tt = t[None, :]
        s2 = sigma_rad**2
        integrand = (
            self.density[None, :]
            * (tt / s2)
            * np.exp(-((rr - tt) ** 2) / (2.0 * s2))
            * i0e(rr * tt / s2)
        )
        out = np.trapz(integrand, t, axis=1)
        return RadialKernel(r, out)


def _mask_lookup(mask: np.ndarray, a_u: np.ndarray, a_v: np.ndarray, support: float) -> np.ndarray:
    """Bilinear transmission lookup on a node grid spanning [-support, +support]²."""
    k = mask.shape[0]
    x = np.clip((a_u / (2.0 * support) + 0.5) * (k - 1), 0.0, k - 1.0)
    y = np.clip((a_v / (2.0 * support) + 0.5) * (k - 1), 0.0, k - 1.0)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, k - 1)
    y1 = np.minimum(y0 + 1, k - 1)
    fx = x - x0
    fy = y - y0
    return (
        mask[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + mask[y0, x1] * fx * (1.0 - fy)
        + mask[y1, x0] * (1.0 - fx) * fy
        + mask[y1, x1] * fx * fy
    )


def mask_transmitted_fraction(kernel: RadialKernel, mask: np.ndarray) -> float:
    """Fraction of the kernel's mass a transmission raster lets through.

    Evaluated on the mask's own node grid, kernel-weighted — the same
    resolution the deposit sees, so a masked deposit renormalised to
    ``weight * fraction`` is self-consistent.
    """
    k = mask.shape[0]
    s = kernel.support_rad
    nodes = np.linspace(-s, s, k)
    w = kernel.value(np.hypot(nodes[None, :], nodes[:, None]))
    denom = w.sum()
    if denom <= 0:
        return 0.0
    return float((w * mask).sum() / denom)


def deposit(
    out: np.ndarray,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    uv0: np.ndarray,
    jac: np.ndarray,
    weight: float,
    kernel: RadialKernel,
    hess: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    wrap_u: bool = False,
    jac_smax: float | None = None,
) -> None:
    """Accumulate one sample point's mapped kernel into ``out`` in place.

    :param out: ``(n_v, n_u)`` accumulator, units of ``weight`` per mm².
    :param u_edges: ``(n_u + 1,)`` bin edges, mm (uniform).
    :param v_edges: ``(n_v + 1,)`` bin edges, mm (uniform).
    :param uv0: ``(2,)`` central-ray landing point, mm.
    :param jac: ``(2, 2)`` Jacobian ``d(uv)/d(angle)``, mm/rad.
    :param weight: total deposited quantity (e.g. watts) for this sample.
    :param kernel: angular density to lay down.
    :param hess: optional ``(2, 2, 2)`` Hessian ``d²(uv_i)/d(angle_j)
        d(angle_k)``, mm/rad². When given, the deposit inverts the local
        *quadratic* model ``uv = uv0 + J a + H[a, a]/2`` instead of the
        linear one: the angular preimage of each bin becomes
        ``a ≈ J⁻¹ d − J⁻¹ H[J⁻¹ d, J⁻¹ d] / 2`` and the density factor
        ``1/|det J|`` becomes the pointwise ``1/|det(J + H[a, ·])|``. This
        removes the leading (curvature) term of the linearisation error at
        the cost of nothing but arithmetic — the residual is third order
        in kernel width.
    :param mask: optional ``(k, k)`` transmission raster in [0, 1] over the
        kernel's angular support square ``[-support, +support]²`` (rows are
        the second angular axis), bilinearly interpolated and multiplied
        into the kernel. This is how aperture rims, receiver-window edges,
        neighbour shadows and neighbour blocking clip the sun cone from
        one surface point — all of them are angular clips, so one raster
        expresses any combination, penumbra included. A masked deposit's
        conserved mass is ``weight * mask_transmitted_fraction(...)``.
    :param jac_smax: optional precomputed top singular value of ``jac``
        (``sqrt(eigvalsh(jac @ jac.T).max())``). A caller that already
        needs this value for its own footprint-reach bound (e.g. the cone
        tracer's transmission-skip test) can pass it here to avoid paying
        for the same eigenproblem twice; leaving it ``None`` computes it
        here instead, with an identical result either way.

    The kernel is evaluated at bin centres — exact in the limit of bins
    small against the kernel footprint, which holds by orders of magnitude
    for solar images (footprints ~10^2 mm vs bins ~10^1 mm). Power that
    the map carries outside the grid is simply not deposited; callers
    difference totals to measure spillage.
    """
    det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
    if det == 0.0:
        raise ValueError("singular Jacobian: degenerate optical map at this sample point")
    inv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det

    # Bounding box of the mapped support: the image of a circle of radius
    # support under jac is an ellipse whose largest reach is the largest
    # singular value of jac times the support radius; the quadratic term
    # can push the true image out by up to |H| support² / 2 more.
    smax = jac_smax if jac_smax is not None else np.sqrt(np.linalg.eigvalsh(jac @ jac.T).max())
    reach = smax * kernel.support_rad
    if hess is not None:
        reach += 0.5 * float(np.abs(hess).max()) * kernel.support_rad**2 * 2.0

    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    i0_raw = int(np.floor((uv0[0] - reach - u_edges[0]) / du))
    i1_raw = int(np.ceil((uv0[0] + reach - u_edges[0]) / du))
    j0_raw = int(np.floor((uv0[1] - reach - v_edges[0]) / dv))
    j1_raw = int(np.ceil((uv0[1] + reach - v_edges[0]) / dv))
    # On a surface that closes on itself the u axis has no edge to fall off:
    # a footprint running past the last column continues at the first, so it
    # is never clipped in u and none of its power is spillage there.
    n_u = u_edges.size - 1
    i0, i1 = (i0_raw, i1_raw) if wrap_u else (max(0, i0_raw), min(n_u, i1_raw))
    j0, j1 = max(0, j0_raw), min(v_edges.size - 1, j1_raw)
    if i0 >= i1 or j0 >= j1:
        return  # footprint entirely off-grid: pure spillage
    # A footprint that never touched the grid boundary must deposit exactly
    # its weight; renormalising to that removes both the bin-centre
    # evaluation error and (at order 2) the approximate-inverse mass error.
    # Clipped footprints keep their raw deposit — the shortfall is genuine
    # spillage the caller measures by differencing totals.
    unclipped = (j0_raw, j1_raw) == (j0, j1) and (wrap_u or (i0_raw, i1_raw) == (i0, i1))

    u_mid = (
        u_edges[0] + (np.arange(i0, i1) + 0.5) * du
        if wrap_u
        else 0.5 * (u_edges[i0 : i1 + 1][:-1] + u_edges[i0 : i1 + 1][1:])
    )
    v_mid = 0.5 * (v_edges[j0 : j1 + 1][:-1] + v_edges[j0 : j1 + 1][1:])
    duv_u = u_mid[None, :] - uv0[0]
    duv_v = v_mid[:, None] - uv0[1]
    alpha_u = inv[0, 0] * duv_u + inv[0, 1] * duv_v
    alpha_v = inv[1, 0] * duv_u + inv[1, 1] * duv_v

    if hess is None:
        a_u, a_v = alpha_u, alpha_v
        denom: float | np.ndarray = abs(det)
    else:
        # Quadratic correction of the preimage: a -= J^-1 H[a, a] / 2, with
        # a the linear preimage. One Newton step on the quadratic model —
        # ample, since the correction is already second order.
        q_u = 0.5 * (
            hess[0, 0, 0] * alpha_u * alpha_u
            + 2.0 * hess[0, 0, 1] * alpha_u * alpha_v
            + hess[0, 1, 1] * alpha_v * alpha_v
        )
        q_v = 0.5 * (
            hess[1, 0, 0] * alpha_u * alpha_u
            + 2.0 * hess[1, 0, 1] * alpha_u * alpha_v
            + hess[1, 1, 1] * alpha_v * alpha_v
        )
        a_u = alpha_u - (inv[0, 0] * q_u + inv[0, 1] * q_v)
        a_v = alpha_v - (inv[1, 0] * q_u + inv[1, 1] * q_v)

        # Pointwise volume factor: det(J + H[a, .]) at the corrected preimage.
        m00 = jac[0, 0] + hess[0, 0, 0] * a_u + hess[0, 0, 1] * a_v
        m01 = jac[0, 1] + hess[0, 0, 1] * a_u + hess[0, 1, 1] * a_v
        m10 = jac[1, 0] + hess[1, 0, 0] * a_u + hess[1, 0, 1] * a_v
        m11 = jac[1, 1] + hess[1, 0, 1] * a_u + hess[1, 1, 1] * a_v
        det_pt = np.abs(m00 * m11 - m01 * m10)
        denom = np.maximum(det_pt, 1e-12 * abs(det))  # guard folds; kernel≈0 there

    vals = kernel.value(np.hypot(a_u, a_v))
    if mask is not None:
        vals = vals * _mask_lookup(mask, a_u, a_v, kernel.support_rad)
    patch = weight * vals / denom
    target = weight if mask is None else weight * mask_transmitted_fraction(kernel, mask)
    total = patch.sum() * du * dv
    if unclipped:
        if total > 0 and target > 0:
            patch *= target / total
        elif target == 0:
            return
    elif total > target > 0:
        # A clipped footprint otherwise keeps its raw deposit, because what
        # runs off the grid is genuine spillage the caller measures by
        # differencing totals. That shortfall can never be an EXCESS though:
        # a sample cannot land more power than it carries. Where the
        # quadratic model folds, `denom` above is only floored at
        # 1e-12 |det|, so without this the density can be amplified without
        # bound and the trace reports more power collected than ever arrived.
        patch = patch * (target / total)
    if wrap_u:
        np.add.at(out, (slice(j0, j1), np.arange(i0, i1) % n_u), patch)
    else:
        out[j0:j1, i0:i1] += patch
