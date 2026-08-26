# Coddington v0.2 — specification addendum

Status: **SIGNED OFF by Ryker, 2026-08-26** (sections A–L and mockups M8–M13; §L desktop shell was Ryker's own 2026-08-26 request, approved in the same review). Companion to [ui-spec.md](ui-spec.md) (rev 5 signed off 2026-08-22; rev 6 draft). This document plus the v0.2 mockups are the build contract for the feature wave; deviations go back to Ryker before implementation. **§M below arrived after sign-off and awaits its own approval.**

Out of scope here (already in flight as bug fixes, no design sign-off needed): trace status bar moved to the top of the screen on all tabs; Analysis-tab panel overlap; the ~5° sun-elevation floor for sweeps; cylindrical-receiver retrace; frustum orientation; year-estimate duration warning; the two Monte Carlo physics corrections (specularity perturbation plane, source-region sizing); the >500-heliostat slowdown.

Also out of scope: **roads** and **drag-to-move/add heliostats** are fully specified in the signed-off rev 5 (§2.2) and will be built to that spec unchanged.

---

## A. Fidelity modes described by purpose

The run-bar fidelity picker stops describing algorithms and starts describing intent. Each mode gets a one-line subtitle (visible in the picker, expanded in a tooltip):

| Mode | Subtitle |
|---|---|
| **Ultra fast** | Field design optimization — explore layouts and geometry quickly. |
| **Fast accurate** | Compare a selected few options with confidence. |
| **Monte Carlo** | Model the final design with precision, including all error sources. |

The tooltip for each mode also states, honestly, what it trades away (Ultra fast: approximate shadowing/blocking during sweeps, small map-detail residual; Fast accurate: deterministic, noise-free, ~2× Ultra fast cost; Monte Carlo: noise falls as 1/√rays, only mode that applies measured error maps and pointing error per ray).

## B. Faster Ultra Fast: estimated shadowing & blocking

**Gate: the low-elevation occlusion validation (in flight) must first confirm the exact computation is correct; approximation is measured against that trusted reference.**

- **During day sweeps and year estimates**, Ultra fast computes exact per-heliostat shading/blocking factors only at **anchor timesteps** and interpolates the per-heliostat factors for the timesteps in between. Anchor spacing is chosen so interpolation error stays within ~1–2 % where the sun is low and tighter at high sun (error budget agreed 2026-08-25). Single-instant traces from the run bar have nothing to interpolate and stay exact.
- **Per-instant cost** in Ultra fast additionally drops by coarsening the occlusion mask resolution in that mode only (Fast accurate and Monte Carlo unchanged), bounded by the same error budget. **Caution from the 2026-08-25 validation:** the cone kernel's 16-node transmission raster already shows several-percent MC discrepancies in its (non-production) direct-occluder path once 12+ neighbouring penumbras overlap — so any coarsening must be re-validated against MC on dense inner-ring clusters specifically, not just the average heliostat.
- The Analysis tab's sweep header notes when a run used estimated occlusion ("shadowing/blocking interpolated between N anchors"), so a published number is never silently approximate.
- Fast accurate and Monte Carlo keep exact occlusion always — consistent with §A's promise.
- **Rides along with this work** (agreed 2026-08-25): characterize the cone kernel's direct-occluder raster — discrepancy vs. mask_nodes (16→32→64) and neighbour count on the dense inner-ring clusters — using the same MC validation harness this section builds anyway; fix the raster or document its limit in cone.py before that path is ever exposed for per-heliostat detailed traces.

## B2. Coefficient-space flux accumulation (experimental prototype, approved 2026-08-25)

Instead of depositing every heliostat's footprint into a shared flux grid ("binning space"), accumulate each heliostat's contribution as coefficients of a smooth basis and evaluate the summed map once at the end — the DELSOL3 approach (Hermite expansion per heliostat, summed in coefficient space). Motivation is doubled by the 2026-08-25 profiling result: per-heliostat deposit cost grows with footprint area (∝ slant range²), which is exactly what made the outer rings 4–8× slower; a coefficient deposit is O(basis terms) regardless of spot size.

- **Prototype compares two bases head-to-head** against the current binning on the default 643-heliostat field: **Hermite–Gauss** (DELSOL precedent; the cone stencil already measures the needed local moments) and **tensor B-splines** (locally supported, so the hard edges survive — receiver-window clipping and blocking penumbras are discontinuities a truncated global series rings at; with real blocking in every field this is why B-splines earn their comparison slot).
- **Pass/fail gates, judged against Monte Carlo:** total power, peak flux, intercept efficiency — with the clipped-edge cases (window edge, heavy blocking, cylinder seam wrap) as the deciding tests, since that is where our release-night energy-conservation bugs lived.
- Scope: cone backends only, Ultra Fast first. **Monte Carlo stays binned** — fitting a basis to rays is smoothing, which biases peak flux low in the one mode positioned as the reference.
- Outcome is a benchmark report + recommendation before any production cutover; the current binned deposit remains until the prototype beats it at the gates.

## C. Secondary-mirror irradiance and absorbed heat

For layouts with a secondary (axicon, Cassegrain):

- The irradiance-map view (Analysis timestep view and the run-bar flux thumbnail expansion) gains a **surface selector: Receiver / Secondary**. The secondary map shows **incident flux on the secondary surface** in the same colormap and units as the receiver map.
- Beside the secondary map, a readout reports **absorbed heat**: total absorbed power and peak absorbed flux density, computed as **(1 − reflectance of the secondary) × incident**. The secondary's reflectance becomes an explicit, visible input in the Receiver & Tower stage — **default 0.90, confirmed 2026-08-25** (the value already in use, surfaced rather than buried).
- Both maps export per §D.

## D. Map exports for FEA (ANSYS-oriented CSV grids)

One export convention, used everywhere a map exists:

- **Sag map export** (Heliostat Shape tab, next to the sag map): regular-grid CSV of `x, y, z_sag` for the previewed heliostat's exact traced surface.
- **Irradiance map export** (receiver and, when present, secondary): regular-grid CSV of `x, y, flux`; for the secondary, optionally `absorbed` as a fourth column.
- Format targets ANSYS External Data import: plain comma-separated numeric grid, one point per row, preceded by commented metadata lines (`# units`, `# heliostat / sun / mode / timestamp`, grid dimensions). Coordinates in **meters**, sag in **millimeters**, flux in **W/m²** — units always stated in the header, never implied.
- Buttons live beside each map ("Export CSV for FEA"), and day-sweep timestep maps export individually from the timestep view.

## E. Measured (FEA) error-map import — Monte Carlo only

For users who have a real deformation map of a heliostat (gravity sag, wind load, thermal) from FEA or deflectometry:

- In the Heliostat Shape tab's optical-errors section: **"Measured error map — Import CSV…"**. Accepts a gridded **sag-deviation** map (`x, y, Δz` from the nominal surface), the same grid convention as §D's export — so a Coddington sag export annotated by an FEA tool round-trips.
- On import the app reports what it read: grid size, coverage of the aperture, and the **RMS slope error implied by the map**, shown next to the analytic slope-error input so the two aren't double-counted blindly.
- The map applies **on top of** the analytic figure, **in Monte Carlo mode only** — a badge says so wherever the map is shown. Cone modes ignore it (their kernels assume the analytic surface); the fidelity picker's Monte Carlo subtitle already advertises this (§A).
- **Trace-time guarantee:** the map is pre-processed once at import into gradient (slope) grids; per-ray application is a bilinear lookup, so Monte Carlo cost is essentially unchanged regardless of map resolution.
- The imported map saves with the heliostat design in the Library and rides through project save/load. It does not ride through SolTrace/SolarPILOT export (no equivalent exists); the interchange report says so per rev 5 §5.

## E2. Secondary-mirror perturbations

For layouts with a secondary (axicon, Cassegrain), the Receiver & Tower stage's secondary controls gain a **Perturbations** group, so alignment and figure error of the secondary can be studied, not just the design geometry:

- **Rigid-body misalignment** — decenter (Δx, Δy, Δz in mm) and tip/tilt (two angles in mrad, about the vertex/apex). These are exact geometry changes, so they apply at **every fidelity** — the perturbed surface is simply what all three backends trace. Corner rays and the 3D scene show the perturbed secondary live, like any other geometry edit; defaults all zero, saved with the receiver config.
- **Surface deformation** — a measured **error map on the secondary** (gravity/thermal warp from FEA), using the same gridded Δ-sag CSV convention and import flow as the heliostat map (§E), applied in **Monte Carlo only** with the same pre-processed-gradient guarantee that trace time doesn't blow up. The import chip reports implied RMS slope error, same as §E.
- **Parametric warp** (confirmed 2026-08-25) — quick what-ifs without an FEA file: low-order terms added to the secondary's figure — **defocus (µm P–V)** and **astigmatism (µm P–V + axis angle)** over the secondary's aperture. Defaults zero; applied in Monte Carlo alongside the map (the two sum).
- **Secondary sag map** — a "View sag" for the secondary, mirroring the heliostat sag map: nominal figure + parametric warp + imported map, summed, jet colormap with stated contour interval, and the same "Export CSV for FEA" button (§D convention). Lives with the secondary's controls in Receiver & Tower, so perturb → view sag → retrace → watch the absorbed-heat view (§C) is one loop.

A fourth optical-error input alongside slope error, specularity, and reflectance: **pointing error (mrad RMS)** — the tracker's aiming inaccuracy, distinct from the mirror's surface errors.

- **Monte Carlo:** each heliostat draws one pointing offset per timestep (2-axis Gaussian) — quasi-static per instant, redrawn per timestep, reproducible from the seed.
- **Cone modes:** folded into the kernel as an additional broadening term, like slope error today — so the annual energy hit of a sloppy tracker shows up at every fidelity, while the per-instant "one heliostat misses left, one misses right" character is Monte Carlo's.
- Default **0 mrad** (matches the manuscript baseline; nothing changes for existing projects).
- **Convention (resolved 2026-08-25): reflected-beam RMS.** The quoted number is the RMS angular deviation of the *reflected beam* — no factor-of-two on reflection is applied to the user's number. The input label reads "pointing error (mrad RMS, on the reflected beam)" and the tooltip states it, so a vendor's mirror-normal spec must be doubled by the user before entry.

## G. Tooltips and the error glossary

- Explanatory prose currently living in code comments and docs moves into **hover tooltips** on the controls it describes, app-wide (every optical-error field, figure/canting controls, fidelity picker, receiver fields, sweep settings). One sentence each, plain language; longer background stays in the docs.
- The optical-error tooltips form a small glossary that draws the distinctions Ryker flagged — draft wording for sign-off:
  - **Slope error** — "Large-scale waviness of the mirror surface: the local surface normal deviates from the design surface by this RMS angle. Broadens the beam (doubled on reflection). Not pointing error (that's the tracker, §F) and not canting error (that's facet aiming)."
  - **Specularity** — "Micro-scale roughness: scatter of the reflected ray about the ideal specular direction, RMS, isotropic about the reflected beam. The 'polish' term — independent of shape, canting, and tracking."
  - **Pointing error** — "The tracker's aiming inaccuracy: the whole mirror points slightly off its commanded direction. Quasi-static per instant, not surface roughness."

## H. Heliostat positions in meters

- All heliostat positions **display and edit in meters**, rounded to **0.01 m** (plan view, inspector, sidebar, CSV interchange headers). Typed values accept any precision; storage keeps full precision (internally still mm) — rounding is display-only, so nothing in the physics or saved projects shifts.
- Layout parameters that are positions/radii (nearest/farthest radius, row spacing, road widths…) follow the same rule. Angles stay in degrees; mirror/receiver *dimensions* and sag stay in mm (**confirmed 2026-08-25** — they're fabrication numbers).

## I. Polar (North/South) heliostat fields

Research complete (2026-08-25). The literature's term is **polar field** (vs. *surround field*): heliostats only on the polar side of the tower — north in the northern hemisphere, south in the southern. The math is long settled: the radial-staggered method our default generator already follows originates with Lipps & Vant-Hull (1978); DELSOL3 and MUEEN bound it azimuthally by a designer-input "extended angle"; Collado & Guallar's *campo* (2012) is the modern systematization; and SolarPILOT exposes exactly an angular min/max field bound. **We adopt that parameterization rather than porting a new layout engine** — our shipped radial-staggered generator is already the validated member of this family, and restricting it to a sector is the published approach, not an invention.

- **Field generator:** a **Field span** control on the layout stage — *Full surround* (default, today's behavior) or *Sector*, with **center azimuth** (default 0° = North; flips to South for southern-hemisphere sites) and **half-span** (e.g. ±60°). Angles use the **compass convention** (0° = North, clockwise) the rest of the app speaks. Applies to the radial-staggered generator first; other layouts can adopt it later (the Fermat spiral does not truncate cleanly to a wedge and is out of scope).
- **Receiver aperture orientation:** the flat-window receiver generalizes from facing straight up/down to an **oriented plane**: *facing azimuth* + *tilt*, so a polar field aims at a vertical or tilted window facing the field. **Default tilt is "Auto (optimal)" (confirmed 2026-08-25):** the app aims the window normal at the field's flux-weighted centroid direction — the tilt the literature says is latitude- and tower-height-dependent (PS10's as-built is ~12.5° from vertical) — with the computed angle displayed and a manual override for studying off-optimal tilts. The cavity case composes with the existing entrance-aperture + offset-receiver mechanism (rev 5 §2.2): tilted aperture in front, absorber behind.
- **Known pitfall to test explicitly:** SolarPILOT's angular field bound historically failed silently for flat receivers (it only worked for cylindrical ones). The acceptance test for this feature is exactly that combination — sector field + flat tilted window — traced end to end.
- Roads, drag-to-move, and heliostat overrides interact with a sector field exactly as with full-surround layouts.
- **Citations** (added to REFERENCES.md with the implementation): Lipps & Vant-Hull 1978, *Solar Energy* 20(6); Collado & Guallar 2012, *Renewable Energy* 46; Wagner & Wendelin 2018, *Solar Energy* 171 (SolarPILOT); PS10 plant description + Wei et al. 2010, *Renewable Energy* 35(9) for aperture-tilt precedent.

## L. Desktop shell (added 2026-08-26)

The desktop build stops presenting as a webpage. Requested by Ryker after evaluating v0.1.0's launch-a-browser-tab behavior.

- **Own window:** the app opens in a native window hosting the existing UI via **WebView2** (`pywebview`) — the web engine already present on Windows 11 — with the Coddington title, taskbar icon (rev 5 §5b), and no browser chrome (no address bar, tabs, or bookmarks). The user's browser is never involved. Window size/position persist across launches.
- **No console:** the launcher builds windowed (no terminal window); backend output goes to a log file under the storage directory, surfaced via a "View log" link in the About dialog for diagnosis.
- The backend still binds localhost (unchanged architecture); a second launch focuses the existing window rather than starting a second server. Browser access to the localhost port keeps working for anyone who wants it — the shell is presentation, not a lockout.
- Dev mode (`uvicorn` + a browser tab) is unchanged — this section is about the shipped build.

## M. Post-sign-off additions (APPROVED 2026-08-26 with riders — M14 confirmed; M15 approved but its home is pending §N; M16 approved + compass rider; M17 approved)

Riders from the approval: **(a)** irradiance maps gain **N/E/S/W compass markings** (flat window: compass directions on the map edges; unwrapped cylinder/frustum: the azimuth axis labeled in compass terms) — consistent with the 3D scene's new compass markers; **(b)** the plan-view power coloring (M.1) may belong in the Ray Trace tab rather than the Workspace plan view — decided by §N below.

1. **Power per heliostat in the plan view.** After a field trace, the plan view can color each heliostat by its delivered power (or efficiency), with a small legend — answering "which heliostats are actually doing the work?" at a glance. Toggle on the plan view; colors update per trace and dim when stale, like other results.
2. **Sweep drill-down to one heliostat.** From any day-sweep or year-estimate timestep, in addition to the field irradiance map: pick a heliostat (click in a mini plan or by id) and see *that heliostat's own* flux footprint at that timestep. Computed on demand as a single-heliostat trace at the stored sun position — same mechanism the per-timestep field maps already use, so nothing new is stored.
3. **Irradiance map on the 3D receiver.** After a trace, the flux map drapes onto the receiver surface in the 3D scene (flat, cylinder, frustum), replacing the plain material until results go stale. The 2D map panel remains the quantitative view; the 3D drape is orientation — where the hot spot physically sits.
4. **Analysis aperture on the flux map (post-processing only).** On any irradiance map (flat receiver first; curved later if wanted): define a limiting aperture — a circle of radius r (or a rectangle) centered on a chosen point, default the map centroid — and read out, computed from the already-stored map with no re-trace: power within the aperture, fraction of collected power, average flux, and **average concentration** (average flux ÷ DNI). The aperture is draggable/resizable on the map and its numbers update live; it saves with the run as an annotation, and an optional encircled-power curve (power vs. aperture radius) gives the whole trade in one plot. Explicitly labeled as analysis, not geometry — the trace, intercept, and collected totals never change.
5. Not new items, recorded as confirmations: Ultra-fast ray-count reduction is covered by §B's coarser-stencil lever (Ryker's field-summed-error argument is exactly the §B validation gate); flat-window tip/tilt is §I's oriented plane and applies to any prime-focus flat window, not only polar fields; "easier radial staggered" and the North/South sector field land with the rev-5 layout picker + §I sector control.

Mockups: M14 (desktop shell, §L), M15 (plan-view power coloring + drill-down), M16 (3D receiver drape), M17 (analysis aperture + encircled-power curve) to be added to the mockup page for approval.

## N. Tab restructure: Design · 3D View · Heliostat Shape · Analysis (FINAL, decided 2026-08-26)

Evolved through two rounds on 2026-08-26. The morning's Ray Trace tab is **superseded** by the afternoon decision: the Workspace itself splits into an authoring tab and an observing/simulating tab, and the instant-trace instruments live with the 3D scene. Four tabs; the app **opens on 3D View** (preserving the signed-off vision: the plant is there in 3D, live, from the first frame).

- **Design** — where the plant is authored. The full four-stage sidebar (Heliostat, Field, Receiver & Tower, Sun) beside the precision 2D views: **plan view** for Field, **elevation** for Receiver & Tower, switched explicitly (a view toggle, not the old expand-a-stage auto-morph — that magic retires with the Workspace). Both views keep their new zoom/pan. Edits apply live to the shared model as always.
- **3D View** — where the plant is observed and simulated. The live 3D scene (edits from Design are there instantly), a compact trace bar (fidelity + Run, the §A purpose subtitles), and a **results dock**: the irradiance map (receiver/secondary selector, compass edges, the M.4 analysis aperture), the M.3 drape toggle, per-heliostat power coloring (M.1), single-heliostat footprint viewing (M.2), FEA exports. Editing stays light here: the **click-to-edit inspector keeps working** (click a heliostat, the receiver, the sun arrow — edit in place), plus a compact sun chip; the full control surface is Design's.
- **Heliostat Shape** — unchanged.
- **Analysis** — time-integrated studies, as already decided: day sweep, year estimate, energy plots, timestep table with **inline irradiance maps** for fast scrubbing (compass edges per the §M rider), the M.2 drill-down row, saved runs — with an "**Open in 3D View**" link on maps for the richer viewers one click deeper.

Mockup M18 is redrawn to this structure (M18a Design, M18b 3D View, M18c Analysis). The M18b "Workspace declutter" question is dissolved rather than answered: the old Workspace ceases to exist.

## J. Acceptance checklist (v0.2 additions)

1. Fidelity picker shows the three purpose subtitles; hovering any optical-error field shows its glossary tooltip.
2. Run an axicon trace: the irradiance view offers Receiver/Secondary; the secondary map shows incident flux and reports absorbed power = (1 − R_secondary) × incident; changing the secondary reflectance input moves the absorbed number, not the incident map.
3. Export the sag map and a receiver map as CSV; re-import the sag CSV (with a synthetic Δz added) as a measured error map: RMS slope is reported, a Monte Carlo trace changes, cone traces don't, and the MC-only badge is visible.
4. Set pointing error to 2 mrad: Monte Carlo spot at one instant shows per-heliostat offsets; Ultra fast total energy drops consistently with the broadening.
5. All plan-view and inspector positions read in meters at 0.01 m; a project saved before v0.2 reopens with identical physics.
6. Ultra fast day sweep reports "occlusion interpolated between N anchors"; Fast accurate and Monte Carlo never do; sweep totals differ from exact Ultra fast by within the stated budget.
7. Generate a North field (±60°) with a vertical prime-focus window: rays land on the window, plan view shows the sector, year estimate runs.
8. Decenter the axicon secondary 50 mm and tilt it 5 mrad: the 3D scene shows it, all three fidelities trace the perturbed surface, and the receiver map shifts accordingly; import a Δ-sag map onto the secondary: Monte Carlo changes, cone modes don't, MC-only badge visible. Add 200 µm of astigmatic warp: the secondary sag map shows it (nominal + warp + map summed), exports as CSV, and the Monte Carlo receiver map degrades.

## K. Open questions — all resolved 2026-08-25

1. ~~Pointing-error convention~~ → **reflected-beam RMS** (§F).
2. ~~Dimension units~~ → positions in m; mirror/receiver dimensions and sag stay in mm (§H).
3. ~~Secondary reflectance default~~ → **0.90**, the value already in use (§C).
4. ~~Ultra-fast anchor spacing~~ → **auto-chosen to the error budget**: anchors placed densely near sunrise/sunset where occlusion changes fast, sparse at midday (§B). Proposed as the default on 2026-08-25; flag at sign-off if a fixed spacing is preferred.
5. ~~Polar-field defaults~~ → half-span **±60°**; receiver tilt defaults to **Auto (optimal)** with manual override (§I).
6. ~~Secondary perturbations parametric warp~~ → yes — defocus + astigmatism terms, plus a secondary sag map view with FEA export (§E2).
