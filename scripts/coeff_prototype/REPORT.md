# B2 prototype report — coefficient-space flux accumulation

**Status: prototype complete, benchmarked, recommendation below.**
Spec: `docs/ui-spec-v0.2.md` §B2. Scope: cone-optics ("Ultra Fast") backend
only. Nothing under `src/` was modified; everything lives under
`scripts/coeff_prototype/`.

All numbers in this report come from one real run of
`scripts/coeff_prototype/run_benchmark.py` (no `--quick`, full 643-heliostat
field, MC reference at 20,000 rays/heliostat), whose raw output is saved at
`scripts/coeff_prototype/benchmark_results.json`. Every seed used anywhere
in this prototype is a fixed integer (`SEED = 20260826` in `scenarios.py`,
`MC_SEED_BASE = 424242` in `run_benchmark.py`, `SeedSequence((base, id))`
per heliostat) — nothing is time-seeded.

## 0. What was built

```
scripts/coeff_prototype/
    sampling.py     SampleBundle + trace_heliostat_samples() -- faithfully
                    reproduces trace_heliostat_cone()'s per-sample math
                    (design=None branch only) without depositing; validated
                    bit-for-bit against the real tracer (see §1).
    binned.py       deposit_binned() -- kernels.deposit(), called unmodified,
                    consuming a SampleBundle. Ground truth for every gate.
    hermite.py      Hermite-Gauss coefficient accumulation + field evaluation.
    bspline.py      Tensor-B-spline coefficient accumulation + field evaluation.
    scenarios.py    The four benchmark scenarios.
    run_benchmark.py   The driver that produced every number below.
    test_coeff_prototype.py   9 correctness pins (all passing).
    benchmark_results.json    Raw numbers this report quotes.
```

**Scope limitation, stated up front**: `sampling.py` only reproduces
`trace_heliostat_cone`'s plain-rectangular-mirror (`design=None`) branch —
the custom-facet-design branch (~80 more lines) is not needed for any B2
gate and was not built. Every scenario traces a flat figure
(`c3=c4=c5=0`, no manufacturing-error terms) — the point of this prototype
is comparing deposit methods on identical samples, not reproducing exact
manuscript physics.

## 1. Sampling-fidelity validation (the load-bearing test)

`TestSamplingMatchesConeTrace` in `test_coeff_prototype.py` calls
`trace_heliostat_samples` + `deposit_binned` and compares the result against
calling `trace_heliostat_cone` directly, for three cases: a plain trace, a
trace with real neighbour occluders (exercising the masked/blocked path),
and a trace against a shrunk window (exercising clipping). All three match
**bit-for-bit** (`rtol=1e-9`, and the two cases with real masking are
asserted to actually exercise `masked > 0` / `blocked > 0` so the test
cannot pass trivially). This is the guarantee that every difference measured
below between binned/Hermite/B-spline is attributable to the deposit method
alone — every method consumes exactly the same traced samples.

All 9 tests pass:

```
9 passed in 1.7s
```

(`tests/test_kernels.py`, run as the required sanity check since nothing
under `src/` changed: 13 passed, unaffected.)

## 2. Method descriptions — actual formulas implemented

### 2.1 Binned deposit (ground truth)

Unmodified `heliostat.trace.kernels.deposit`: for each sample, maps the
angular kernel through the local Jacobian and writes `weight * k(|α|) /
|det J|` into every flux-grid cell within reach, `α = J⁻¹(uv − uv0)`. Cost
scales with the number of grid cells the footprint covers.

### 2.2 Hermite-Gauss coefficient accumulation

Each sample's contribution is the fixed function `k(|α|)` (the shared
angular kernel — same for every sample in the field), represented in the
dimensionless whitened coordinate `ξ = α / σ`, `σ = kernel.rms_radius_rad()
/ √2` (the kernel's Gaussian-equivalent one-axis standard deviation), as a
truncated series in the **physicists' Hermite functions**

```
ψ_n(x) = H_n(x) · exp(−x²/2) / √(2ⁿ n! √π)          (orthonormal on all of ℝ)

k(σ|ξ|)  ≈  Σ_{n+m ≤ ORDER}  c[n,m] · ψ_n(ξ_u) · ψ_m(ξ_v)

c[n,m] = ∬ g(ξ) ψ_n(ξ_u) ψ_m(ξ_v) dξ_u dξ_v     (numeric quadrature)
```

with **`ORDER = 6`** (28 terms; `(ORDER+1)(ORDER+2)/2`). Chosen because it
comfortably resolves the super-Gaussian sunshape's kurtosis (order-4 gave
visibly worse edge behaviour in early testing) while keeping per-masked-
sample projection cost bounded — see §4 for the timing this bought.

* **Unmasked samples** (the common case): `c[n,m]` is the **same for every
  sample in the field** — computed once, at `HermiteBasis.build()` time, on
  a 129×129 quadrature grid spanning the kernel's support. A sample's own
  accumulation is then genuinely O(1): store `(uv0, J⁻¹, weight/|det J|)`
  and a reference to the shared coefficient array.
* **Masked/clipped samples**: the `k×k=256`-node transmission raster
  `sampling.py` already computes (no extra tracing) gives `k(θ_j)·pass_j`
  at 256 angular nodes; the same projection formula, evaluated as a
  discrete sum over those 256 nodes, gives that one sample's own
  coefficients — `O(k²·n_terms)`, independent of the footprint's size on
  the flux grid.
* **Renormalisation** (mirrors `kernels.deposit`'s own convention exactly):
  each record's amplitude is rescaled so the *truncated* series' own
  analytic-in-ξ-space integral matches `target = weight` (unmasked) or
  `weight·frac` (masked) exactly —

  ```
  amplitude = target / (|det J| · σ² · Σ_{n,m} c[n,m]·I_n·I_m)
  ```

  where `I_n = ∫ ψ_n(ξ) dξ` (computed on the same grid the coefficients
  came from). This removes the truncation's own conservation error at
  evaluation time; §3/§4 report the *residual* map error this correction
  cannot remove (shape error, not scale error).
* **Node-fallback samples** (chief ray lost at a rim, rare): handled
  identically to `binned.py`'s direct point scatter, for all three methods —
  a shared simplification so this rare edge case cannot bias the
  comparison.
* **Evaluation**, once per field: each record's series is evaluated over a
  local bounding box (`reach = smax · kernel.support_rad`, the same bound
  `kernels.deposit` itself uses) via `numpy.polynomial.hermite.hermvander`
  to get `ψ_0..ψ_ORDER` at every pixel in one call per axis, then a
  Python-level sum over the 28 `(n,m)` terms.

### 2.3 Tensor B-spline coefficient accumulation

"Binning with smooth bins," literally: `kernels.deposit` — the **same**
function, unmodified — accumulates samples onto a coarse **32×32** control
grid spanning the same receiver window (chosen as a 4× coarsening per axis,
16× fewer cells than the 128×128 flux grid — see §4 for why this resolution
buys most of the available speedup). A footprint that spans many 128-grid
cells spans far fewer 32-grid cells, which is the entire mechanism behind
this method's speed win — not a different physical model.

Evaluation, once per field: a **fixed** cubic B-spline interpolation matrix
per axis, `M_u (128×32)` / `M_v (128×32)`, built once via
`scipy.interpolate.make_interp_spline(coarse_axis, e_j, k=3)` for each
one-hot control vector `e_j` (`bc_type="periodic"` for the cylinder-seam
scenario's wrapping `u` axis) — independent of any trace data, a pure
function of the two grids' geometry. The field map is then one matrix
product: `fine = M_v @ coarse @ M_u.T`.

## 3. Gate tables (measured, full 643-heliostat field where stated)

### 3.1 Scenario 1 — default field, one timestep, ultra_fast

643 heliostats, `prime_focus` optics, flat figure, sun az=150°/el=45°, no
occluders (matches how the app's own field sweeps run `ultra_fast` today —
occlusion applied as a scalar elsewhere, out of scope here). MC reference:
20,000 rays/heliostat, run once, serially (8.3 s total).

| Method | Total power (W) | vs binned | Peak flux (W/m²) | vs binned | Intercept eff. |
|---|---:|---:|---:|---:|---:|
| **Binned** (ground truth) | 4,687,276 | — | 337,235 | — | 0.5994 |
| Hermite (order 6) | 4,716,451 | **+0.62%** | 336,959 | −0.08% | 0.6031 |
| B-spline (32×32) | 4,693,770 | **+0.14%** | 333,436 | −1.13% | 0.6002 |
| **MC reference** (20k rays) | 4,788,655 | −2.12%* | 427,578 | −21.1%* | 0.6121 |

\* *vs MC row shows binned's own gap, for scale — see caveat below; Hermite
is −1.51% and B-spline −1.98% vs MC on total power, and all three cone
methods sit at essentially the same ≈−21% vs MC on peak flux.*

**MC caveat, stated honestly**: the ~21% peak-flux gap vs MC is **not a
deposit-method effect** — all three cone methods (which differ from each
other by ≤1.1%) show essentially the same gap against MC. It is the
well-known artifact of comparing a smooth analytic deposit's true maximum
against a single 20,000-ray MC realization's per-bin maximum: Poisson
counting noise on a finite sample systematically inflates the *observed*
max over many bins above the *true* smooth peak. The ~2% total-power gap
is more informative and is consistent with the cone backend's known
linearisation residual at this geometry.

Map error vs binned (map-level, not just power/peak) at field scale:

| Method | Max diff (% of binned peak) | RMS diff (% of binned peak) | Power conservation vs binned |
|---|---:|---:|---:|
| Hermite | 5.63% | 1.61% | 0.62% |
| B-spline | 11.46% | 1.31% | 0.14% |

### 3.2 Scenario 2 — window-edge clipping (deciding gate)

30/643 heliostats (stratified `downselect`, seed 20260826), receiver window
shrunk to 700 mm half-extent (from 2000 mm) so outer-ring spots clip hard.

| Method | Total power vs binned | Peak flux vs binned | Max map diff (% peak) | RMS map diff (% peak) | Power conservation |
|---|---:|---:|---:|---:|---:|
| Hermite | **+2.19%** | +1.65% | **11.73%** | 3.76% | 2.19% |
| B-spline | +0.04% | −0.01% | 7.74% | **0.30%** | 0.04% |

**Hermite rings at edges — quantified.** The 2.19% power-conservation
error and 11.7%/3.76% (max/RMS) map error are the largest of any scenario
for Hermite: an isotropic-kernel series truncated at total degree 6, cut by
a hard aperture edge, cannot represent the discontinuity — this is exactly
the DELSOL-style failure mode the spec anticipated. B-spline's power
conservation stays excellent (0.04%) and its RMS map error is small
(0.30%) — "blurred binning," not ringing — but its *max* error (7.74%) shows
the coarse control grid does produce a visible localized artifact at the
sharp edge, just a much smaller and better-conserved one than Hermite's.

### 3.3 Scenario 3 — heavy blocking (deciding gate)

16 innermost-ring heliostats (~30 m radius, the field's densest angular
packing), sun at 10° elevation, real neighbour occluders (`MirrorGeometry`,
~20 neighbours each within 3× mirror diagonal) passed explicitly to exercise
the direct-occluder transmission raster section B's docstring calls "non-
production" — exactly the path B2 asks to be validated.

| Method | Total power vs binned | Peak flux vs binned | Max map diff (% peak) | RMS map diff (% peak) | Power conservation |
|---|---:|---:|---:|---:|---:|
| Hermite | +0.17% | −0.42% | 6.70% | 0.92% | 0.17% |
| B-spline | +0.24% | +0.53% | **34.15%** | 2.52% | 0.24% |

This is the scenario where B-spline looks worst on max map error (34% of
peak) — occlusion penumbras from many different neighbours create several
independent, differently-shaped hard edges within one heliostat's spot, and
the 32×32 control grid (≈62.5 mm cells over the 2000 mm window) resolves
several of them poorly after upsampling. Both methods' *power* and *peak
flux* stay within roughly half a percent of binned, though — the 34% pixel
is a localized edge artifact, not a corruption of either headline number
(binned peak=12,857 W/m², B-spline peak=12,925 W/m² — the hot artifact
pixel is not the map's own maximum).

### 3.4 Scenario 4 — cylinder seam wrap (deciding gate)

8 north-sector heliostats (closest to due-north azimuth — radial aiming
puts them naturally at the seam), `CylinderReceiver` (r=3000 mm, h=6000 mm),
`wrap_u=True` exercised in all three deposit paths.

| Method | Total power vs binned | Peak flux vs binned | Max map diff (% peak) | RMS map diff (% peak) | Power conservation |
|---|---:|---:|---:|---:|---:|
| Hermite | +0.0003% | −0.03% | **2.32%** | 0.25% | **0.0003%** |
| B-spline | −0.29% | **+3.73%** | 17.52% | 2.19% | 0.29% |

Hermite does *best* here of any scenario — no hard clipping, just a smooth
wrap, which its isotropic-in-ξ representation handles cleanly. B-spline's
periodic upsample matrix is the weakest link of this scenario: peak flux is
off by 3.7% (the largest peak-flux miss of any method/scenario in this
report) and max map error is 17.5% — the periodic cubic-spline construction
evidently behaves less cleanly right at the wrap seam than the interior.
This is a concrete, actionable weak spot for any further iteration of the
B-spline method, not a fundamental limitation of "binning with smooth bins"
in general.

## 4. Timing

### 4.1 Deposit-phase time per heliostat vs ring radius (scenario 1, all 643)

Mean per-heliostat accumulation time, grouped by the field's 12 discrete
ring radii (innermost 30.0 m → outermost 89.6 m):

| Ring radius (m) | n | Binned (ms) | Hermite accum. (ms) | B-spline accum. (ms) |
|---:|---:|---:|---:|---:|
| 30.0 | 32 | 57.8 | 56.3 | 16.6 |
| 35.0 | 32 | 60.2 | 65.4 | 17.5 |
| 40.3 | 64 | 66.1 | 67.2 | 21.6 |
| 46.1 | 48 | 92.8 | 66.8 | 28.8 |
| 51.1 | 48 | 68.1 | 64.7 | 37.3 |
| 56.4 | 48 | 104.8 | 70.0 | 28.3 |
| 62.0 | 48 | 122.3 | 76.3 | 26.4 |
| 67.8 | 71 | 140.1 | 98.2 | 37.5 |
| 72.9 | 71 | 149.1 | 90.5 | 37.0 |
| 78.2 | 71 | 165.3 | 89.3 | 32.2 |
| 83.8 | 71 | 210.2 | 97.5 | 49.4 |
| 89.6 | 71 | 257.3 | 100.1 | 35.7 |
| **Growth, innermost→outermost** | | **4.45×** | **1.78×** | **2.16×** |

This is the headline evidence for the spec's own claim: binned deposit's
per-heliostat cost grows **4.45×** from the innermost to the outermost ring
(consistent with the ∝slant-range² profiling result that motivated B2);
Hermite's accumulation-only cost grows much more slowly (**1.78×** — mostly
from a growing share of masked samples needing the more expensive per-
sample projection at large radius, not from footprint size itself); B-spline
grows **2.16×** but from a much lower base (16.6→35.7 ms vs binned's
57.8→257.3 ms) — **consistently 3-4× cheaper than binned at every ring**.

Averaged over the whole field: binned accumulation = 139.8 ms/heliostat;
Hermite accumulation = 82.5 ms/heliostat (**1.69× cheaper**); B-spline
accumulation = 33.0 ms/heliostat (**4.23× cheaper**).

### 4.2 Total wall-clock, 643-heliostat field, one timestep

| Method | Sample gen. | Accumulate | Evaluate (once, whole field) | **Total** | vs binned |
|---|---:|---:|---:|---:|---:|
| Binned | 7.8 s | 89.9 s | — | **97.7 s** | — |
| Hermite | 7.8 s | 53.1 s | **161.0 s** | **221.9 s** | **2.27× slower** |
| B-spline | 7.8 s | 21.2 s | 0.01 s | **29.1 s** | **3.36× faster** |

**The honest headline finding**: the accumulation-phase win the spec is
built around is real for *both* coefficient methods (Hermite 1.69× cheaper,
B-spline 4.23× cheaper than binned, on average). But it is only the whole
story for B-spline, whose evaluation step is a single, cheap, fixed matrix
multiply (10 ms, independent of sample count) — its **total** pipeline is a
clear 3.36× win. Hermite's evaluation step, in this implementation (a
Python-level loop per accumulated record — of which there are ≈154,000 for
the full field — each evaluating a 28-term polynomial series over its own
local bounding box), is so expensive that it swamps the accumulation-phase
win entirely: Hermite's total wall-clock is the **slowest** of the three
methods tested, over twice as slow as plain binned deposit.

## 5. Correctness pins (`test_coeff_prototype.py`, 9/9 passing)

* Sampling fidelity vs `trace_heliostat_cone` (plain, occluded, clipped) —
  bit-for-bit, `rtol=1e-9`.
* Single-Gaussian footprint moments (mean, covariance, total power) in all
  three deposits, tolerances loosened from binned's own 1e-4/2e-3 pins to
  account for truncation + finite-quadrature error (Hermite/B-spline: total
  power `rel=2e-3`, centroid `abs=1.0mm`, spread `rel=5e-3`).
* Uniform-disk power conservation in all three deposits (binned `rel=1e-4`;
  Hermite `rel=1e-2` — a hard top-hat is a much harder target for a
  truncated Hermite series than a Gaussian; B-spline `rel=5e-3`).

## 6. Recommendation

**Hermite-Gauss (order 6): iterate, do not adopt as built.** The core
accumulation-phase concept is validated — real per-sample cost savings
(1.7× on average, flattest growth with ring radius of the two methods) that
get more pronounced at the field's outer, most expensive rings — but two
concrete problems block adoption:

1. **Evaluation is too slow as implemented.** A per-record Python loop
   evaluating a 28-term series over a local bounding box costs more in
   total than the entire binned pipeline. This is very plausibly fixable
   (batch the ≈90% of records that are unmasked — which all share one
   coefficient vector — into one large vectorised evaluation instead of a
   per-record loop; consider a lower truncation order; or evaluate onto a
   coarser intermediate grid and upsample, borrowing the B-spline method's
   own trick), but as built, it is a net loss.
2. **Real ringing at hard edges**, exactly as expected: the window-clipping
   gate shows Hermite's largest power-conservation error (2.19%) and map
   error (11.7% max / 3.76% RMS) of any scenario. A production version
   would need an explicit non-isotropic edge-correction term (e.g. an
   erf-based half-plane correction, following DELSOL3's own treatment more
   closely) rather than relying on the coarse node-raster projection this
   prototype uses for masked samples.

**Tensor B-spline (32×32 control grid, cubic upsample): adopt for Ultra
Fast, with an open item at the cylinder seam.** It wins decisively on
wall-clock (3.36× faster end-to-end, 4.2× faster accumulation alone),
conserves power well everywhere tested (0.04%-0.29%), and its total-power/
peak-flux numbers track binned within about 1% at field scale and within
about 4% even at the hardest edge cases — with one exception (below). Its
weakness is real and visible in max (not RMS) map error at hard edges —
up to 34% of peak in the heavy-blocking scenario — but confirmed to be a
localized artifact away from the map's own peak in every scenario tested,
not a corruption of the two headline gate metrics (total power, peak
flux). This residual profile is a good match for Ultra Fast's own already-
stated tolerance (§A: "approximate shadowing/blocking during sweeps, small
map-detail residual" is explicitly accepted for this mode) and should
**not** be extended to Fast Accurate or Monte Carlo, where the spec
promises exact edges.

**Open item before a production cutover**: the cylinder-seam scenario is
B-spline's single weakest result by peak-flux (+3.7%, the largest peak-flux
miss of any method/scenario in this report) — the periodic upsample
construction needs another look (a different periodic boundary condition,
or a higher-resolution control grid specifically around the seam column)
before B-spline is trusted on cylindrical receivers specifically. Flat
prime-focus windows (scenarios 1-2) and blocking-heavy fields (scenario 3)
show no comparable weakness.

**Bottom line**: this prototype does not support an unconditional cutover
of the binned deposit. It supports adopting the B-spline coefficient method
for the Ultra Fast mode specifically, once the cylinder-seam upsample is
fixed, with binned deposit remaining the reference for Fast Accurate and
Monte Carlo as already specified in B2. Hermite-Gauss is a validated
concept that is not yet a net win and needs the two fixes above before it
is worth re-benchmarking.
