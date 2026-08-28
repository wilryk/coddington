"""Named fidelity modes: ultra-fast, fast-and-accurate, Monte Carlo.

Three ways to trade time for fidelity, all producing the same kind of
flux map on the same receiver grid:

``ultra_fast``
    Cone optics with a linearised (five-ray) stencil: ~1,200 deterministic
    rays per heliostat. Total power and spot moments are essentially exact
    (±0.1%); local map detail carries a ~1%-of-peak curvature residual.
    ~60 ms per heliostat-instant single-core.

``fast_accurate``
    Cone optics with the quadratic (nine-ray) stencil: measures the
    optical map's curvature and deposits through it, removing the
    ultra-fast mode's leading error for roughly twice the cost. Still
    deterministic, still noise-free.

``monte_carlo``
    The reference: full Monte Carlo ray trace, bit-reproducible from its
    seed, noise falling as 1/sqrt(rays). This is the backend the other
    two are validated against.

The mode objects only carry configuration; call sites unpack
``cone_kwargs`` into :func:`heliostat.trace.cone.trace_heliostat_cone` or
``n_rays`` into :func:`heliostat.trace.mc.trace_heliostat`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceMode:
    """A named point on the speed/fidelity curve."""

    name: str
    backend: str  # "cone" | "mc"
    cone_kwargs: dict = field(default_factory=dict)
    n_rays: int = 0  # mc only

    def __post_init__(self):
        if self.backend not in ("cone", "mc"):
            raise ValueError(f"backend must be 'cone' or 'mc', got {self.backend!r}")


ULTRA_FAST = TraceMode(
    "ultra_fast",
    backend="cone",
    # grid=None + density resolves the mirror-sample grid from the mirror's
    # own aperture bbox (12 samples/m^2, aspect-matched) instead of the
    # hardcoded (20, 12) fixed to the manuscript's 5m x 3m mirror -- see
    # heliostat.trace.cone.grid_for_density and
    # scripts/coeff_prototype/REPORT.md SS7 (0.073% field-total error vs the
    # hardcoded grid, on the full 643-heliostat field).
    # deposit_method="bspline" accumulates onto a coarse control grid and
    # upsamples once instead of depositing every sample onto the fine flux
    # grid directly -- see heliostat.trace.bspline_deposit and REPORT.md
    # SS0-6 (3.36x end-to-end speedup at field scale, power conserved to
    # 0.04-0.29%). fast_accurate below keeps the fixed (20, 12) grid and
    # exact binned deposit, both untouched.
    cone_kwargs={
        "order": 1,
        "grid": None,
        "density": 12.0,
        "mask_nodes": 16,
        "deposit_method": "bspline",
    },
)
FAST_ACCURATE = TraceMode(
    "fast_accurate", backend="cone", cone_kwargs={"order": 2, "grid": (20, 12), "mask_nodes": 16}
)
MONTE_CARLO = TraceMode("monte_carlo", backend="mc", n_rays=120_000)

MODES = {m.name: m for m in (ULTRA_FAST, FAST_ACCURATE, MONTE_CARLO)}
