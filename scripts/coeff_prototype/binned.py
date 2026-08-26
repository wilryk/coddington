"""Ground-truth deposit: the current production method, on a grid of any
resolution.

This is a thin wrapper around ``heliostat.trace.kernels.deposit`` -- imported
and called unmodified, never re-implemented -- consuming a
:class:`~scripts.coeff_prototype.sampling.SampleBundle` instead of being
called inline inside ``trace_heliostat_cone``'s own loop. Two roles:

* On a 128x128 grid, this reproduces ``trace_heliostat_cone``'s own output
  bit-for-bit (see ``test_coeff_prototype.py``) -- the benchmark's ground
  truth and the "binned" column of every gate table.
* On the coarser 32x32 control grid, this exact same function is reused by
  ``bspline.py`` to accumulate its control-point coefficients -- "binning
  with smooth bins," per the spec's own framing, not a different algorithm.
"""

from __future__ import annotations

import numpy as np

from heliostat.trace.kernels import deposit

from .sampling import SampleBundle


def deposit_binned(
    bundle: SampleBundle,
    u_edges: np.ndarray,
    v_edges: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Deposit every sample in ``bundle`` onto a grid spanning
    ``u_edges``/``v_edges``, exactly the way ``trace_heliostat_cone``'s own
    final loop does. Returns the ``(n_v, n_u)`` accumulator, units of
    ``bundle.weights`` per mm^2 (multiply by 1e6 for W/m^2, matching
    ``trace_heliostat_cone``'s own convention).

    Pass an existing ``out`` (e.g. accumulated across several heliostats
    already) to add this bundle's samples into it in place -- this is how
    the field-level benchmark sums many heliostats onto one shared grid,
    matching production's own shared-flux-grid convention.
    """
    n_u = u_edges.size - 1
    n_v = v_edges.size - 1
    if out is None:
        out = np.zeros((n_v, n_u))
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    bin_area_mm2 = du * dv
    k = bundle.axis_nodes.size

    for idx in range(bundle.m):
        if bundle.frac[idx] < 1.0e-6:
            continue
        if bundle.chief_ok[idx] and bundle.can_jac[idx]:
            full_pass = bundle.frac[idx] > 1.0 - 1.0e-9
            hess_i = None
            if bundle.hess is not None and not np.any(np.isnan(bundle.hess[idx])):
                hess_i = bundle.hess[idx]
            deposit(
                out,
                u_edges,
                v_edges,
                bundle.uv0[:, idx],
                bundle.jac[idx],
                float(bundle.weights[idx]),
                bundle.kernel,
                hess=hess_i,
                mask=None if full_pass else bundle.node_ok[idx].astype(float).reshape(k, k),
                wrap_u=bundle.wrap_u,
                jac_smax=float(bundle.smax[idx]),
            )
        else:
            ok_j = bundle.node_ok[idx]
            w_sum = bundle.w_nodes.sum()
            share = bundle.weights[idx] * bundle.w_nodes[ok_j] / w_sum / bin_area_mm2
            iu = np.clip(((bundle.uv_nodes[0, idx, ok_j] - u_edges[0]) // du), 0, n_u - 1)
            iv = np.clip(((bundle.uv_nodes[1, idx, ok_j] - v_edges[0]) // dv), 0, n_v - 1)
            np.add.at(out, (iv.astype(np.intp), iu.astype(np.intp)), share)

    return out


def power_w(out: np.ndarray, u_edges: np.ndarray, v_edges: np.ndarray) -> float:
    """Total power (W) on a grid whose values are W/mm^2 (i.e. already
    multiplied by 1e6 the way ``deposit_binned``'s raw output is NOT --
    ``deposit_binned`` returns weight/mm^2 directly, so this expects that
    raw unit and does the mm^2->m^2 conversion itself: ``sum * bin_area_mm2``
    already gives watts directly, since ``weight`` is watts and density is
    watts/mm^2.)
    """
    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]
    return float(out.sum() * du * dv)
