"""Tensor B-spline coefficient-space flux deposit -- "binning with smooth
bins," an alternative accumulation target for the cone-optics backend's
``ultra_fast`` mode.

Ported from the validated prototype at ``scripts/coeff_prototype/bspline.py``
(see ``REPORT.md`` SS0-6 there for the original benchmark this module's
adoption came from: 3.36x end-to-end speedup vs binned deposit at field
scale, power conserved to 0.04-0.29% -- measured at a fixed 32x32 control
grid, 4x coarsening, on a FLAT receiver only). The math is unchanged from
the prototype; this module exists so
:func:`heliostat.trace.cone.trace_heliostat_cone` can select it as an
accumulation target without reaching into ``scripts/``.

**That 4x coarsening was later found to under-resolve peak flux badly on a
curved receiver** (a matched-sunshape re-measurement: cylinder/frustum peak
flux 40-43% low vs binned at the old default) -- see
:mod:`heliostat.trace.cone`'s ``CONTROL_GRID_COARSEN_PERIODIC``/
``CONTROL_GRID_COARSEN_NONPERIODIC`` for the fix (2x on both axes) and the
measurements behind it. At the corrected 2x setting, re-measured field-scale
timing (single ULTRA_FAST heliostat and a serial 40-heliostat field, cached
upsample matrices -- see ``_cached_upsample_matrix`` below) shows the speed
picture is shape-dependent, not the flat prototype's uniform 3.36x: a real
~1.5x win persists on a flat/planar receiver (16.3 ms/heliostat vs binned's
24.3 ms at n=40), but on a cylinder receiver the corrected setting is only
roughly at parity with binned (~20-24 ms/heliostat either way, run-to-run
noise on the order of the difference) -- the periodic axis's accuracy needs
and its speed potential are in real tension, and 2x coarsening spends most
of the periodic axis's available headroom on accuracy, not speed. See the
constants' own comment in ``cone.py`` for the full numbers and the
recommendation this leaves for curved receivers specifically.

**Accumulate.** Exactly :func:`heliostat.trace.kernels.deposit` -- called
unmodified, by the tracer itself, once per sample -- but targeting a coarse
control grid (now 2x coarsened per axis by default, not the prototype's
fixed 32x32/4x -- see above) spanning the same receiver window instead of
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
the masked/full-pass path. The prototype's node-fallback deposit path (the
rare "chief ray lost at a rim, surviving mass scattered directly at node
landing points" branch) had a separate, unrelated seam bug (+3.7% peak flux,
REPORT.md SS3.4): it computed a bin index and *clipped* it to
``[0, n_coarse - 1]`` regardless of ``wrap_u``, silently dumping any node
landing just past the wrap column back onto the last column instead of
column 0. The tracer's own accumulation loop (:mod:`heliostat.trace.cone`)
fixes that by wrapping that index modulo the coefficient count when
depositing onto a periodic coarse grid.

The upsample matrix's own periodic branch (``_upsample_matrix`` below) now
uses the same non-negative coefficient-blend technique as the non-periodic
branch -- a uniform *periodic* cubic B-spline built directly from the coarse
values as its control points (the standard closed-curve construction: knots
spaced uniformly around the period, coefficients wrapped cyclically), not an
exact-interpolating periodic spline solved to pass through them. The same
convex-hull argument applies: uniform B-spline basis functions are
non-negative and sum to 1 regardless of whether the knot vector happens to
be periodic, so a blend of non-negative coefficients through this basis is
non-negative by construction, on both axes, with no Gibbs ringing at a sharp
coarse-grid edge straddling the seam. See
:class:`tests.test_bspline_deposit.TestPeriodicNoRingingArtifact` and
:class:`tests.test_bspline_deposit.TestSeamContinuity`.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

DEFAULT_CONTROL_GRID = (32, 32)  # (n_u, n_v); see REPORT.md SS3 for the resolution study

#: Cache of built upsample matrices, keyed on exactly what
#: :func:`_upsample_matrix` depends on: the coarse and fine edge arrays
#: (bit-identical bytes -- both are deterministic functions of the receiver
#: geometry and control/flux grid shape, so every heliostat and every
#: timestep traced against the same receiver at the same grid resolution
#: produces bit-identical edges) and the periodicity flag. See
#: :func:`_cached_upsample_matrix`.
_MATRIX_CACHE: dict[tuple[bytes, bytes, bool], np.ndarray] = {}


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

    Both branches now use the same COEFFICIENT BLEND technique -- the
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
    ``CONTROL_GRID_COARSEN`` was 4 by default at the time -- see
    :mod:`heliostat.trace.cone`'s ``CONTROL_GRID_COARSEN_PERIODIC``/
    ``CONTROL_GRID_COARSEN_NONPERIODIC`` for the current, lower, per-axis
    default and the peak-accuracy measurement that replaced it). Measured on
    a single-heliostat
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

    ``periodic=True`` (the cylinder-seam ``u`` axis) applies the identical
    idea with a periodic (closed-curve) knot vector instead of a clamped
    one: knots spaced uniformly around the full period, extended by ``k``
    on each side by periodic wraparound, and coefficients wrapped cyclically
    (``c_ext[i] = e_j[i % n_c]``) so the resulting curve closes on itself
    with ``C^2`` continuity at the seam -- the standard "closed uniform
    cubic B-spline" construction. This used to be an exact-interpolating
    periodic spline (pinned unit-for-unit against ``make_interp_spline``'s
    own ``bc_type="periodic"`` solve); that exact-interpolation property
    was retired in favour of the same non-negative coefficient blend used
    above, because exact interpolation and guaranteed non-negativity cannot
    both hold for sharp data, and a receiver that closes on itself has no
    physical reason to privilege exact reproduction of the coarse grid's own
    sample values over the seam actually being non-negative, continuous,
    power-conserving and bearing-invariant -- see
    :mod:`tests.test_bspline_deposit`'s ``TestPeriodicNoRingingArtifact``,
    ``TestSeamContinuity`` and ``TestPeriodicConstantReconstructsExactly``.
    """
    coarse_mid = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])
    fine_mid = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    n_c = coarse_mid.size
    n_f = fine_mid.size
    k = 3
    m = np.zeros((n_f, n_c))
    if periodic:
        # Uniform periodic cubic B-spline: knots spaced at the coarse cell
        # width `h`, extended k on each side by periodic wraparound, with
        # coefficients wrapped cyclically so the curve closes on itself.
        # This is the direct periodic analogue of the coefficient-blend
        # branch below -- non-negative, sums to 1, no linear solve.
        period = coarse_edges[-1] - coarse_edges[0]
        h = period / n_c
        knot_idx = np.arange(-k, n_c + k + 1)
        t = coarse_mid[0] + knot_idx * h
        fm = coarse_mid[0] + np.mod(fine_mid - coarse_mid[0], period)
        for j in range(n_c):
            c_ext = np.zeros(n_c + k)
            # e_j repeated cyclically across the k extra wrap-around
            # coefficients (i.e. c_ext[i] = 1 wherever i % n_c == j).
            c_ext[j] = 1.0
            if j < k:
                c_ext[n_c + j] = 1.0
            spl = BSpline(t, c_ext, k, extrapolate=False)
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


def _cached_upsample_matrix(coarse_edges: np.ndarray, fine_edges: np.ndarray, periodic: bool) -> np.ndarray:
    """Cached wrapper around :func:`_upsample_matrix`.

    The matrix is a *fixed* function of the two grids' geometry alone --
    never of trace data -- so at field scale (many heliostats, many
    timesteps, all sharing one receiver and one flux/control grid pair) it
    only needs to be built once and reused for every subsequent call with
    the same edges. Measured (coarsen=2, 200-call average per branch): a
    fresh non-periodic 65-control/129-fine matrix build costs ~6.2 ms, a
    periodic 113-control/449-fine build ~9.6 ms; a cache hit of either
    costs ~0.004 ms -- roughly 1,700x cheaper. That is what makes a finer
    (less-coarsened, more accurate) control grid affordable at field scale:
    the accumulate-phase saving from a coarse grid is still real, but the
    evaluate-phase matrix cost that used to scale with heliostat count now
    does not -- it is paid once per (receiver, grid) pair, not once per
    heliostat-timestep.

    Keyed on the exact edge arrays' bytes (not on the grid shape or
    receiver identity): two different code paths that happen to produce the
    same edges -- e.g. two heliostats traced against the same receiver at
    the same resolution -- correctly share one cache entry, and any change
    to the edges (a different flux/control grid, a different receiver
    extent) correctly misses and rebuilds. No entry is ever evicted; a
    session touches at most a handful of distinct (receiver, grid) pairs,
    each matrix a few hundred KB at most, so unbounded growth within one
    process is not a practical concern.
    """
    key = (coarse_edges.tobytes(), fine_edges.tobytes(), periodic)
    m = _MATRIX_CACHE.get(key)
    if m is None:
        m = _upsample_matrix(coarse_edges, fine_edges, periodic)
        _MATRIX_CACHE[key] = m
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
    m_u = _cached_upsample_matrix(u_edges_coarse, u_edges_fine, periodic=wrap_u)
    m_v = _cached_upsample_matrix(v_edges_coarse, v_edges_fine, periodic=False)
    fine = m_v @ coarse @ m_u.T
    coarse_area = (u_edges_coarse[1] - u_edges_coarse[0]) * (v_edges_coarse[1] - v_edges_coarse[0])
    fine_area = (u_edges_fine[1] - u_edges_fine[0]) * (v_edges_fine[1] - v_edges_fine[0])
    target = float(coarse.sum()) * coarse_area / fine_area
    np.clip(fine, 0.0, None, out=fine)
    total = float(fine.sum())
    if total > 0.0 and target > 0.0:
        fine *= target / total
    return fine
