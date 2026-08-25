# Coddington stress test — plan

Status: **plan, not yet built.** The goal is to find the failures nobody
thought to write a test for, by varying parameters in combination and
judging every outcome against rules that hold for *all* inputs.

The existing suite asks "does this case still give the number it gave
before". This asks a different question: **is there any combination of
settings that makes the software crash, hang, or lie?**

## 1. What counts as a failure

A run is judged against these rules. Anything that trips one is a finding
with a reproduction payload attached, not a line in a log.

| Class | Rule | Why it matters |
|---|---|---|
| **Crash** | No request may return 5xx. An impossible geometry is a 422 with a readable message. | A 500 is always a bug — either a missing validation or a real break. |
| **Hang** | No single request may exceed its time budget (geometry 2 s, trace 60 s, one day timestep 30 s). | The 1.9°-elevation occlusion cliff was exactly this, and no test caught it. |
| **Not a number** | No NaN or Inf anywhere in a numeric response. | These propagate silently into energy totals and plots. |
| **Impossible physics** | Collected power ≤ incident power; efficiencies in [0, 1]; flux ≥ 0; peak flux ≥ mean flux; rms radius > 0 when any power lands. | Catches sign errors and double-counting that still "look like numbers". |
| **Disagreement** | The geometry endpoint and the trace endpoint must agree about heliostat count, aim points and receiver placement for the same input. | The two solve paths have drifted before. |
| **Round trip** | Save a project, load it, serialize again — the second document must equal the first. Same for library entries. | Silent field loss on save is invisible until someone reopens old work. |
| **Monotonic sanity** | Where a direction is known, it must hold: more heliostats never collect less; a larger aperture never catches fewer rays; reflectance 0.9 collects exactly 0.9 of reflectance 1.0. | Cheap, and catches whole classes of wiring errors. |

## 2. What gets varied

Sampled in combination, not one axis at a time — the interesting failures
live in the corners.

- **Optics**: prime focus, axicon, Cassegrain. Apex/focus/vertex heights, half angle **including 0°**, aperture radius from tiny to absurd, receiver height above *and* below the apex, window half-sizes.
- **Receiver** (once it lands): flat / cylindrical / frustum, radius and height extremes, top radius == bottom radius, zero height, receiver centre on and off axis, aperture→receiver offset 0 and large.
- **Heliostat design**: rect, facet grid, custom polygon, flower. Surface twisting/spherical/flat × `facet_focal_mm` {absent, 0, small, huge} × `cant_focal_mm` {absent, 0, small, huge} — this matrix is where the recent curvature/canting split could hide a bad combination. Optical errors: 0 and deliberately large slope/specularity; reflectance 0→1.
- **Custom polygons specifically**: 3 vertices, many vertices, self-intersecting, near-zero area, huge, mirror symmetry on and off. (A self-intersecting polygon has already produced one odd result.)
- **Field**: radial staggered with varied bands and ring counts, Fermat across its range, single heliostat. Edge positions: **at the tower base**, extremely close, extremely far, and negative/zero radii.
- **Sun**: elevation −10° to 90° including the horizon crossing, azimuth right around 0/360, and the 1.9° low-sun case that cost 87 s.
- **Trace**: all three fidelities, ray budget at both limits.
- **Scale**: 1, 10, 643, 1 000 (trace cap) and 10 000 (geometry cap) heliostats, plus one over each cap to confirm a clean refusal.

## 3. How it runs

A standalone `scripts/stress.py`, driving the app **in-process** through
FastAPI's `TestClient` — no server, no browser, so a full pass is
CPU-bound rather than network-bound and can run thousands of cases.

- **Deterministic sampling.** A seeded generator picks combinations; every
  finding prints the seed and the exact JSON payload, so any failure
  replays as a one-line curl or a pasted test.
- **Three depths.** `--quick` (a few hundred cases, minutes, suitable for
  CI), `--full` (thousands, tens of minutes), `--focus <area>` to hammer
  one subsystem after a change.
- **Timing recorded for every call**, so the report ranks the slowest
  combinations. A performance cliff is a finding in its own right, not a
  footnote.
- **Findings, not logs.** Output is a report grouped by failure class,
  each with the smallest payload that still reproduces it. Where cheap,
  shrink the failing case automatically (drop heliostats, round
  parameters) before reporting.

## 4. What it deliberately does not do

- It does not replace the pinned physics tests. Those assert *specific
  numbers are still right*; this asserts *nothing is catastrophically
  wrong anywhere*.
- It does not drive the browser. UI behaviour gets a separate, much
  smaller smoke pass; the value here is in volume, and volume means the
  API.
- It does not run in the default `pytest` suite. A slow, sampling test
  that occasionally surfaces something new does not belong in a gate that
  must be green on every commit — `--quick` can be wired into CI once its
  findings are down to zero.

## 5. Order of work

1. Harness plus the crash / hang / NaN rules — the classes most likely to
   be hiding something today.
2. The design matrix (curvature × canting × surface × errors), since it is
   the newest physics and the least exercised.
3. Physics-plausibility and cross-endpoint agreement rules.
4. Round-trip and monotonic rules.
5. Fix what it finds, then keep `--quick` green.
