"""Tensor B-spline coefficient-space flux deposit -- "binning with smooth
bins," an alternative accumulation target for the cone-optics backend's
``ultra_fast`` mode.

Ported from the validated prototype at ``scripts/coeff_prototype/bspline.py``
(see ``REPORT.md`` SS0-6 there for the full benchmark this recommendation
comes from: 3.36x end-to-end speedup vs binned deposit at field scale, power
conserved to 0.04-0.29%). The math is unchanged from the prototype; this
module exists so :func:`heliostat.trace.cone.trace_heliostat_cone` can select
it as an accumulation target without reaching into ``scripts/``.

**Accumulate.** Exactly :func:`heliostat.trace.kernels.deposit` -- called
unmodified, by the tracer itself, once per sample -- but targeting a coarse
control grid (default 32x32) spanning the same receiver window instead of
the full fine flux grid. This is not a different physical model: a footprint
that spans many fine-grid cells spans far fewer coarse-grid cells, which is
the entire mechanism behind the speed win.

**Evaluate once, at the end.** The accumulated coarse control-grid values are
treated as data on the coarse grid's cell centres and a *fixed* (built once,
independent of any trace data -- purely a function of the two grids'
geometry) cubic-B-spline interpolation matrix upsamples them to the fine
grid: ``fine = M_v @ coarse @ M_u.T``. ``M_u``/``M_v`` come from
``scipy.interpolate.make_interp_spline(coarse_axis, e_j, k=3)`` evaluated at
the fine grid's cell centres, one unit vector ``e_j`` at a time.

**Cylinder-seam periodicity.** For a receiver whose ``u`` axis wraps
(``wrap_u``, e.g. a closed cylinder), the coarse control grid's own
accumulation already wraps correctly -- it goes through ``kernels.deposit``
unmodified, which has always wrapped bin indices modulo the column count for
the masked/full-pass path. The prototype's own upsample construction
(``_upsample_matrix`` below, ``bc_type="periodic"``) is likewise an exact
periodic cubic interpolant -- verified here by comparing it against directly
building the periodic spline through arbitrary coefficient values (they
agree to float64 roundoff, not merely approximately). The prototype's
measured seam artifact (+3.7% peak flux, REPORT.md SS3.4) traces instead to
its NODE-FALLBACK deposit path (the rare "chief ray lost at a rim, surviving
mass scattered directly at node landing points" branch): that branch
computed a bin index and *clipped* it to ``[0, n_coarse - 1]`` regardless of
``wrap_u``, silently dumping any node landing just past the wrap column back
onto the last column instead of column 0. The tracer's own accumulation loop
(:mod:`heliostat.trace.cone`) fixes this by wrapping that index modulo the
coefficient count when depositing onto a periodic coarse grid -- "coefficient
indices wrap modulo the coefficient count," both here (the upsample) and
there (the node-fallback accumulation).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

DEFAULT_CONTROL_GRID = (32, 32)  # (n_u, n_v); see REPORT.md SS3 for the resolution study


def control_grid_edges(
    u_edges: np.ndarray, v_edges: np.ndarray, control_grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Coarse ``(u_edges, v_edges)`` spanning the same extent as the fine
    ``u_edges``/``v_edges`` but with ``control_grid`` cells.
    """
    n_cu, n_cv = control_grid
    return (
        np.linspace(u_edges[0], u_edges[-1], n_cu + 1),
        np.linspace(v_edges[0], v_edges[-1], n_cv + 1),
    )


def _upsample_matrix(coarse_edges: np.ndarray, fine_edges: np.ndarray, periodic: bool) -> np.ndarray:
    """``(n_fine, n_coarse)`` fixed linear operator from coarse cell-centre
    values to fine cell centres.

    The two branches use different constructions on purpose.

    ``periodic=False`` (used for the ``v`` axis always, and for ``u`` on any
    non-wrapping receiver): a cubic-B-spline COEFFICIENT BLEND -- the
    coarse values are used directly as B-spline coefficients (a
    quasi-interpolation / control-point blend), not solved for as the
    control points of a spline forced to pass exactly through them. This is
    what makes the result non-negative for non-negative input with no
    downstream clamp needed: cubic B-spline basis functions are themselves
    non-negative and sum to 1 (the "convex hull" property: a point on the
    curve is a convex combination of nearby coefficients), so a blend of
    non-negative coefficients through this basis is non-negative *by
    construction* -- there is no Gibbs-style ringing at a sharp coarse edge
    to clip away in the first place. An exact-interpolating construction
    does not have that property: forcing a spline through a sharp edge
    drives it through under/overshoot lobes on either side, periodic at the
    coarse cell spacing; clipping the negative lobes to zero (this
    function's own earlier behaviour, before this fix) chopped the positive
    rebound lobes next to them into isolated islands surrounded by
    exact-zero moats -- the reported "grid every 4 pixels" (owner report;
    CONTROL_GRID_COARSEN=4 by default). Measured on a single-heliostat
    ultra_fast trace (flat, non-wrapping receiver, both axes non-periodic):
    the old interpolating construction's raw (pre-clip) undershoot reached
    ~8-12% of local peak right at a spot edge, spaced at exactly the
    control grid's cell width and scaling with it (~0.2% of peak at a 2x
    coarsen, ~8-12% at the default 4x, larger still at 8x) -- textbook
    cubic-spline ringing at a knot-spaced sharp feature. The same case's
    binned-deposit (fast_accurate) and Monte Carlo flux maps show no such
    structure at any amplitude. The trade is mild extra smoothing versus an
    interpolating spline (a coefficient blend does not reproduce each
    coarse cell's own value exactly at its own centre -- it blends with
    neighbours too), well inside the mode's own ~1% documented curvature
    residual and the 0.3% power-conservation tolerance this module is
    gated on (which ``evaluate_bspline`` enforces exactly anyway via its
    own rescale, independent of this matrix's shape) -- and it measurably
    moved the single-heliostat case above CLOSER to a high-ray-count Monte
    Carlo reference (whole-map relative L2 error 0.340 -> 0.280), not just
    quieter at the edge.

    ``periodic=True`` (the cylinder-seam ``u`` axis only): unchanged,
    still the exact interpolating construction -- see
    :class:`tests.test_bspline_deposit.TestUpsampleMatrixIsExactPeriodicInterpolation`,
    a unit-level pin on this exact technique (interpolate through coarse
    values with a periodic wrap-extension) as the guard against silently
    reintroducing the cylinder-seam off-by-one this module's own periodic
    handling was written to fix (see module docstring). A non-negative
    coefficient blend and an exact interpolant are mutually exclusive for
    sharp data -- so this axis, on a wrapping (cylindrical) receiver only,
    can still show a milder version of the ringing artifact above; fixing
    that too means first re-deriving (and re-pinning) the periodic
    construction's correctness some other way, which is out of scope here.
    """
    coarse_mid = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    fine_mid = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    n_c = coarse_mid.size
    n_f = fine_mid.size
    k = 3
    m = np.zeros((n_f, n_c))
    if periodic:
        for j in range(n_c):
            y = np.zeros(n_c)
            y[j] = 1.0
            # A periodic interpolant needs its first/last y equal; wrap the
            # basis vector's period explicitly by using make_interp_spline's
            # own periodic extension (it requires y[0] == y[-1]).
            period = coarse_edges[-1] - coarse_edges[0]
            x = np.concatenate([coarse_mid, [coarse_mid[0] + period]])
            yy = np.concatenate([y, [y[0]]])
            spl = make_interp_spline(x, yy, k=k, bc_type="periodic")
            fm = coarse_mid[0] + np.mod(fine_mid - coarse_mid[0], period)
            m[:, j] = spl(fm)
    else:
        # Coefficient-blend construction (see docstring above): the knot
        # vector is obtained via make_interp_spline purely for a correctly
        # clamped cubic knot vector at these coarse cell centres -- the y
        # values passed to it are never used, only its `.t` is read.
        t = make_interp_spline(coarse_mid, np.zeros(n_c), k=k).t
        for j in range(n_c):
            e = np.zeros(n_c)
            e[j] = 1.0
            # Outside the clamped basis's support (within half a coarse
            # cell of the window edge, where no basis function is nonzero)
            # is correctly zero, not an extrapolated value -- exactly the
            # true value of a compactly-supported basis function there.
            vals = BSpline(t, e, k, extrapolate=False)(fine_mid)
            m[:, j] = np.nan_to_num(vals)
    return m


def evaluate_bspline(
    coarse: np.ndarray,
    u_edges_coarse: np.ndarray,
    v_edges_coarse: np.ndarray,
    u_edges_fine: np.ndarray,
    v_edges_fine: np.ndarray,
    wrap_u: bool,
) -> np.ndarray:
    """Upsample a coarse accumulator to the fine grid via the fixed cubic
    B-spline matrices, then restore the two physical invariants the raw
    spline breaks:

    * **Nonnegativity** -- a cubic spline undershoots near sharp edges, so
      the raw upsample carries small negative lobes. Flux is nonnegative;
      downstream consumers (payloads, FEA CSV, drape) must never see a
      negative bin. Clamped at zero.
    * **Exact power conservation** -- the accumulation step conserves the
      deposit exactly, but the raw upsample only approximately preserves
      the integral (measured 0.04-0.29%, REPORT.md), and could land the
      collected power a hair ABOVE incident -- the release-night violation
      class at parts-per-billion scale. After clamping, the positive field
      is rescaled uniformly so the fine-grid total equals the coarse
      accumulator's deposited power exactly; the scale factor is within
      ~1e-3 of 1, a shape-preserving correction far below the mode's own
      accuracy.
    """
    m_u = _upsample_matrix(u_edges_coarse, u_edges_fine, periodic=wrap_u)
    m_v = _upsample_matrix(v_edges_coarse, v_edges_fine, periodic=False)
    fine = m_v @ coarse @ m_u.T
    coarse_area = (u_edges_coarse[1] - u_edges_coarse[0]) * (v_edges_coarse[1] - v_edges_coarse[0])
    fine_area = (u_edges_fine[1] - u_edges_fine[0]) * (v_edges_fine[1] - v_edges_fine[0])
    target = float(coarse.sum()) * coarse_area / fine_area
    np.clip(fine, 0.0, None, out=fine)
    total = float(fine.sum())
    if total > 0.0 and target > 0.0:
        fine *= target / total
    return fine
