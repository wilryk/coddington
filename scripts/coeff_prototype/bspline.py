"""Tensor B-spline coefficient accumulation -- "binning with smooth bins."

**Accumulate.** Exactly the binned method's own ``kernels.deposit``, called
unmodified, but targeting a coarse control grid (default 32x32) spanning the
same receiver window instead of the full 128x128 flux grid. This is not a
different algorithm from binned deposit -- it is the *same* energy-
conserving, mask-aware, wrap-aware deposit machinery, just on ~1/16th as
many cells (4x coarser per axis). A footprint that spans many 128-grid
cells spans far fewer 32-grid cells, which is the entire mechanism behind
any speed win this method shows: per the spec's own framing, "this is
binning with smooth bins," not a fundamentally different physical model.

**Evaluate once, at the end.** The accumulated 32x32 control-grid values are
treated as data on the coarse grid's cell centres and a *fixed* (built once,
independent of any trace data -- purely a function of the two grids'
geometry) cubic-B-spline interpolation matrix upsamples them to the full
128x128 grid: ``fine = M_v @ coarse @ M_u.T``. ``M_u``/``M_v`` come from
``scipy.interpolate.make_interp_spline(coarse_axis, e_j, k=3)`` evaluated at
the fine grid's cell centres, one unit vector ``e_j`` at a time (offline,
once per field trace's grid geometry -- not per heliostat, not per sample).
For a receiver whose ``u`` axis wraps (the cylinder-seam scenario),
``bc_type="periodic"`` keeps the spline (and hence the upsample) periodic
too.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from .binned import deposit_binned
from .sampling import SampleBundle

DEFAULT_CONTROL_GRID = (32, 32)  # (n_u, n_v); see REPORT.md for the resolution study


def control_grid_edges(u_edges: np.ndarray, v_edges: np.ndarray, control_grid: tuple[int, int]):
    """Coarse ``(u_edges, v_edges)`` spanning the same extent as the fine
    ``u_edges``/``v_edges`` but with ``control_grid`` cells.
    """
    n_cu, n_cv = control_grid
    return (
        np.linspace(u_edges[0], u_edges[-1], n_cu + 1),
        np.linspace(v_edges[0], v_edges[-1], n_cv + 1),
    )


def _upsample_matrix(coarse_edges: np.ndarray, fine_edges: np.ndarray, periodic: bool) -> np.ndarray:
    """``(n_fine, n_coarse)`` fixed linear operator: cubic-B-spline
    interpolation from coarse cell-centre values to fine cell centres.
    """
    coarse_mid = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    fine_mid = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    n_c = coarse_mid.size
    n_f = fine_mid.size
    m = np.zeros((n_f, n_c))
    bc_type = "periodic" if periodic else None
    for j in range(n_c):
        y = np.zeros(n_c)
        y[j] = 1.0
        if periodic:
            # A periodic interpolant needs its first/last y equal; wrap the
            # basis vector's period explicitly by using make_interp_spline's
            # own periodic extension (it requires y[0] == y[-1]).
            x = np.concatenate([coarse_mid, [coarse_mid[0] + (coarse_edges[-1] - coarse_edges[0])]])
            yy = np.concatenate([y, [y[0]]])
            spl = make_interp_spline(x, yy, k=3, bc_type="periodic")
            period = coarse_edges[-1] - coarse_edges[0]
            fm = coarse_mid[0] + np.mod(fine_mid - coarse_mid[0], period)
            m[:, j] = spl(fm)
        else:
            spl = make_interp_spline(coarse_mid, y, k=3)
            m[:, j] = spl(fine_mid)
    return m


def deposit_bspline_coarse(
    bundle: SampleBundle, u_edges_coarse: np.ndarray, v_edges_coarse: np.ndarray
) -> np.ndarray:
    """The accumulation step: bin, at coarse resolution. Identical call to
    ``binned.deposit_binned`` -- this function exists only to name the step
    distinctly in the benchmark's timing output.
    """
    return deposit_binned(bundle, u_edges_coarse, v_edges_coarse)


def evaluate_bspline(
    coarse: np.ndarray, u_edges_coarse, v_edges_coarse, u_edges_fine, v_edges_fine, wrap_u: bool
) -> np.ndarray:
    """Upsample a coarse accumulator to the fine grid via the fixed cubic
    B-spline matrices. Power is approximately (not exactly) conserved by
    construction -- see REPORT.md for the measured conservation error, which
    comes from the upsample, not from the (exact, deposit-conserving)
    accumulation step.
    """
    m_u = _upsample_matrix(u_edges_coarse, u_edges_fine, periodic=wrap_u)
    m_v = _upsample_matrix(v_edges_coarse, v_edges_fine, periodic=False)
    return m_v @ coarse @ m_u.T
