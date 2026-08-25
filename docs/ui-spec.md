# Coddington workspace UI — specification (rev 6 draft; rev 5 SIGNED OFF)

Status: rev 5 was **approved by Ryker, 2026-08-22**; this document plus the "Coddington Workspace" mockups are the build contract. Deviations go back to Ryker before implementation.

**Rev 6 (draft, awaiting sign-off)** records corrections Ryker made while testing the Heliostat Shape tab: facet curvature and canting are independent controls, a flat heliostat can be made weakly focusing, a twisting figure induces astigmatism on facets too, and the Workspace's Heliostat stage reports the aperture rather than editing it. Marked in §2.2 and §3.
Companion mockups: the "Coddington Workspace" artifact (seven screens, referenced as M1–M7).
Rev 2 incorporated the first review (axicon orientation, plan/elevation edit modes, warning-vs-error contract, interchange, Custom outlines, explicit canting, jet sag maps, per-timestep irradiance maps). Rev 3 incorporated the second (near-heliostat miss detection, layout picker + roads, heliostat-plane datum, choosable storage, saved analysis runs, slow-operation Apply). Rev 4 incorporated the third (flexible roads, honeycomb, optical errors, shape-preview selector, confirmed datum). Rev 5 incorporates the fourth: manuscript error defaults (90 % reflectance, no slope/specularity error), the receiver as a real design (types per layout + entrance-aperture offset), graceful degenerate geometries, the annual Year estimate, and Ryker Optics branding placement.

The one-sentence vision: **the 3D scene is the workspace, not a results viewer.** You open the app and your tower, receiver, and field are already there in 3D; every edit shows up immediately; tracing adds physics on top. The reference feel is SolidWorks: a live model in the middle, design controls around it, view modes that follow what you are editing, and a couple of full-screen editors for detail work.

---

## 1. Screen inventory

| Screen | What it is | Mockup |
|---|---|---|
| **Workspace** | The main screen. Live viewport, left sidebar of design stages, bottom run bar. | M1, M2 |
| **Field plan view** | The viewport's top-down mode, active while the Field stage is open. | M3 |
| **Tower elevation** | The viewport's dimensioned side-elevation mode, active while Receiver & Tower is open. | M4 |
| **Library** (slide-over) | Named heliostat designs, receiver configs, projects; locked manuscript built-ins; SolTrace/SolarPILOT/CSV interchange. | M5 |
| **Heliostat Shape** (full-screen tab) | Detail editor for the mirror: shape, figure, canting, aperture preview, sag map. | M6 |
| **Analysis** (full-screen tab) | Day sweeps, energy plot, timestep table, irradiance maps, CSV export. | M7 |

Top bar (always visible): project name, Workspace / Heliostat Shape / Analysis tabs, Library button, save state.

## 2. The Workspace

### 2.1 The viewport and its modes

- **Live from the first frame.** Opening the app shows the default project: ground, tower/secondary, receiver, and the field, with the sun direction indicated. No trace required, ever, to see geometry. Every edit updates the geometry within ~a third of a second.
- **Apply only where it's slow.** Edits apply live by default. Any operation the app expects to take more than a couple of seconds (regenerating a very large layout, an expensive re-solve) is deferred behind an explicit **Apply** button that appears in place, so typing never triggers a multi-second stall. The threshold is measured, not guessed: if the last refresh of that kind ran slow, the control switches to deferred mode.
- **The viewport follows the active stage.** Heliostat and Sun stages (and no stage) leave it in 3D. Expanding **Field** switches to the top-down plan view (§2.2); expanding **Receiver & Tower** switches to the dimensioned side elevation (§2.2). A view pill in the corner names the current mode and offers "back to 3D" at all times; collapsing the stage also returns to 3D.
- **The axicon draws apex-down**: the apex is the cone's lowest point (`z = apex + r·tan α`), the funnel opens upward. The 3D, plan (aperture circle), and elevation views all follow the real profile from the server.
- **Corner rays** are drawn automatically and refresh with every edit: chief rays from mirror corners, through the secondary, to the receiver. Labeled in-scene as *"corner chief rays — viewing aid, no shading."* For big fields they come from a spread-out subset (cap ~500 sources). Rays that miss the optics (e.g. beyond the aperture rim) draw dashed red rather than disappearing.
- The ground grid is deliberately faint so rays and geometry dominate; it never competes with content.
- After a real trace, traced rays and flux results augment the scene until the next edit makes them stale (stale results dim with a re-run hint).
- **Scale target:** smooth orbiting up to 10,000 heliostats. **Sun below horizon:** scene never goes blank — heliostats hold their last pose, rays disappear, a banner explains.

### 2.2 Left sidebar — four design stages

Collapsible sections in workflow order. Each edits live against the viewport.

**Heliostat** — which mirror design the field uses: picker from the Library + "Edit shape…" opening the Heliostat Shape tab, with a one-line summary (type, size, figure). **The aperture is reported here, not edited here** — shape and dimensions belong to the Heliostat Shape tab, so this stage stays a clean statement of what the heliostat is. The surface-figure toggle (Twisting / Spherical / Flat) is the one exception that stays in reach.

**Field** — where the mirrors stand.
- Mode: *Single heliostat* (x, y) or *Field*.
- **Layout picker**: **Fermat spiral** (default; count, nearest radius, farthest radius — defaults 30 m and 90 m, the manuscript field), **radial staggered** (the classic DELSOL/Campo pattern), **grid/cornfield** (row and column spacing), **honeycomb** (hex-staggered dense packing), and **imported positions** (CSV). Each layout keeps its own last-used parameters.
- **Roads**: the user can add roads of a chosen width in three forms — **straight at any angle**, with **partial extent** (e.g. an access road on the South half only, preserving the high-performance North field), and **rings** at a chosen radius. Heliostats a road crosses are cut out automatically; the sidebar counts how many. Roads draw in the plan view and persist in the project.
- Opening the stage shows the **plan view** (M3): rings at the layout radii, receiver and secondary aperture at center, roads, north arrow, scale bar, sun azimuth. **Drag a heliostat to move it** — an override of the parametric layout, counted in the sidebar with a one-click reset; double-click empty ground adds a heliostat. Overrides persist in the project.
- Viewing cap 10,000; tracing keeps its 1,000 cap and the Run bar says so when exceeded.

**Receiver & Tower** — the optics layout: **Prime focus / Axicon / Cassegrain**, each with honest per-layout labels (no shared "tower height" box):

| Prime focus | Axicon | Cassegrain |
|---|---|---|
| Focus height | Apex height | Secondary vertex height |
| Window ½ w / ½ h | **Half angle** | Primary focus height |
| | **Aperture radius** | Receiver height |
| | Receiver height | Aperture radius |
| | Window ½ w / ½ h | Window ½ w / ½ h |

- **Height datum: the heliostat plane.** All heights (apex, focus, receiver, vertex) are referenced to the plane containing the heliostat pivots — not the ground. Confirmed: the engine's z = 0 already *is* this plane, so this is a labeling/drawing contract, not a physics change. A separate **ground offset** parameter (heliostat plane above ground, **default 2 500 mm** — the bottom edge of the 3 m manuscript mirror clears grass and weeds by about a metre) positions the drawn ground; the elevation view draws the datum as a labeled dash-dot line with the ground offset as its own dimension.
- Opening the stage shows the **elevation view** (M4): a to-scale side elevation with SolidWorks-style dimension callouts (apex/focus heights, receiver height, aperture radius, half angle, ground offset) — every callout is an editable value box bound to the same fields as the sidebar; the receiver and cone drag along the mast.
- **The receiver is its own design** (a Receiver subsection of the stage):
  - **Type per layout**: *Flat window* (all layouts — the only sensible choice for axicon and Cassegrain, whose beams arrive from above); under **prime focus** also *Cylindrical* (radius, height) and *Frustum* (top radius, bottom radius, height). Both also take the **receiver center height** (relative to the heliostat-plane datum, drawn against the ground via the ground offset).
  - **Aiming for curved receivers**: each heliostat's focus vector must be **radially offset** to aim at the receiver surface point facing it (not the axis) — the aim solve takes the per-heliostat azimuth into account for cylindrical/frustum shapes.
  - **The receiver is positionable anywhere**, not just on the field axis: center (x, y, z), defaulting to the center of the field. The aiming math generalizes to any aim point; the one constraint is that aim points must sit **above the heliostat plane**.
  - **Entrance aperture + offset receiver** (a paper interest): the aperture sits at the focus position; **"Aperture → receiver (mm)"** places the receiver that distance behind it. 0 (default) = receiver at the aperture, today's behavior. Flux is reported at the receiver; the aperture clips what enters.
  - The receiver draws in all views (3D, elevation with its own dimensions, plan as footprint) and is part of the saved receiver config.
- **Degenerate geometries are legal and graceful.** Axicon half angle 0° is a flat disc mirror; a Cassegrain whose solve tends to R → ∞ is a flat fold mirror. Both trace exactly as flat mirrors, with a small info badge ("degenerate: tracing as flat mirror") — never a validation error. (Today's `gt=0` validators relax accordingly.)
- Switching layout type keeps each layout's last-used numbers. Receiver configs save to / load from the Library; swapping never touches the field.

**Sun** — azimuth + elevation, direct or via site & time (existing sun calculator).

### 2.3 Validation: warnings vs. errors

Two distinct severities, shown inline at the offending field in both the sidebar and the in-scene inspector:

- **Warning (amber)** — the geometry is legal but deficient. Canonical cases: raising the axicon half angle until the outer field misses the aperture, and **a near heliostat whose aim point cannot reach the secondary at all** (possible close to the tower). The message states the needed value ("needs ≥ 15 800 mm to catch the full field") and counts total-miss heliostats; those heliostats highlight red in every view (3D, plan) with their miss rays dashed red. **Nothing is adjusted automatically**; the geometry stays live. The same treatment applies whatever caused the miss — an optics edit or a heliostat dragged too close in plan view.
- **Error (red)** — the geometry is impossible (receiver above the axicon apex, unsolvable Cassegrain relay). The scene keeps the last valid geometry until the value is fixed; the next valid value recovers with no reload.

### 2.4 In-scene selection & inspector

- Clicking an object — a heliostat, the secondary, the receiver, the sun arrow — selects it and opens a compact floating inspector.
- **The inspector shows exactly the same fields as that object's sidebar stage, bound to the same values.** Type in either place and the other updates instantly. No separate edit pathway, no hidden overrides, no Apply. Warnings and errors appear in both places identically.
- Esc or clicking empty space deselects.

### 2.5 Run bar (bottom)

Fidelity (**Ultra fast / Fast accurate / Monte Carlo** + ray budget), one **Run** button, progress, cancel. Results strip: peak & mean flux, intercept efficiency, a flux-map thumbnail that expands to the full irradiance map, "Export flux CSV", and a trace timestamp. Results dim when edits make them stale.

## 3. Heliostat Shape tab (M6)

Full-screen. Left: shape controls — **Rectangle** (width/height), **Facet grid** (n×m, facet size, gap), **Custom** — plus figure, canting, and optical-error controls. Right: live aperture-layout preview and sag map (existing server renders). "Save to library as…" and "Done".

- **The previewed heliostat is always named.** The sag/figure depends on which heliostat (its slant range) and the sun, so the header chip and the sag panel state it explicitly ("Previewing on H-214 · r 45.2 m"), a locator mini-map shows where it sits in the field, and a ▾ selector switches to another. From the workspace, clicking any heliostat offers **"View shape"**, opening this tab locked to it. Default preview: a representative mid-field heliostat.
- **Optical errors** are part of the heliostat design: **slope error (mrad)**, **specularity (mrad)**, **reflectance (%)**. They feed the Monte Carlo trace and ride through SolTrace/SolarPILOT export. **Defaults are the manuscript's: 0 / 0 / 90 %.**

- **Custom** (replaces flower): sketch a closed outline of straight segments with typed dimensions, SolidWorks-sketch style; optional mirror symmetry. Arcs/constraints are out of scope for v1.
- **Surface figure**: Twisting / Spherical / Flat. Twisting induces both slant focusing and astigmatism, and does so on facets as well as on a solid mirror — a faceted twisting design carries the astigmatic figure per facet, not merely a spherical approximation of it.
- **Facet curvature and canting are independent controls**, because they are independent in practice: a mirror's own surface shape is one decision, where each facet points is another.
  - **Facet curvature**: none / follow canting / fixed focal. A **flat heliostat can be made weakly focusing** — the common real case, where nominally flat facets are built with a single long curvature — via an explicit choice on the Flat figure rather than by borrowing the canting number.
  - **Canting (facet aiming)**: Off (parallel facets) / per-heliostat (each heliostat cants to its own slant range, so different field positions are physically different heliostats) / fixed focal (one number for the whole field — one part number, at the cost of performance away from that range). Canting stays customisable per slant distance regardless of what the curvature is set to.
  - Under Twisting the figure and aim are solved together, so the canting control is locked and labeled as solved for you.
- **Sag map uses the jet colormap** (the standard one, full saturation), with the contour interval stated in the caption (default: contours every 1.0 mm of sag).

## 4. Analysis tab (M7)

Full-screen. The strip under the tabs restates the exact project being analyzed. Day sweep: date + timestep + fidelity, background job with progress/cancel, energy-through-the-day plot, per-step table, day CSV export.

- **Fidelity is one setting for the whole app**, not one per screen: the fidelity chosen here and the one in the workspace run bar are the same value seen twice. Changing it marks a finished sweep as no longer describing the current settings, like any other change that moves the numbers.
- A finished sweep says plainly when the heliostat, optics or field have changed since it ran — dimmed but still readable, and its timesteps still openable. The sweep's own date and timestep are settings for the *next* run and do not invalidate it.

- **Irradiance maps**: click any timestep row and its flux map appears beside the table; the workspace flux thumbnail expands to the same view. This is the answer to "how do I see irradiance maps" — one click from any timestep, one click from the run bar. The sweep keeps each timestep's map as it traces, so opening one costs nothing. Irradiance maps use the trace's own colormap (magma); the jet colormap applies to sag maps only.
- **Runs save with the project.** A finished sweep (including a large Monte Carlo run) persists in the project's storage and reopens without re-running. Per-run **"Discard this run"** covers the user who doesn't want that, and a **Manage saved runs** view lists what is stored with its disk footprint and delete controls.
- **Year estimate**: traces a set of **sample days across the year** (default 12, spaced in solar declination) using the day-sweep machinery, then **interpolates between them weighted by DNI** to report **annual collection in MWh** (with the per-day energies plotted across the year so the interpolation is visible, not a black box). Runs in the background like a day sweep and saves with the project.
  - **Fast mode** (default on): declination is symmetric about the solstices, so only **7 days are actually traced** and the other 5 come by symmetry; DNI weighting is applied afterward. A toggle traces all 12 for the skeptic.

## 5. Library (M5)

Three collections in one slide-over: **Heliostat designs**, **Receiver configs**, **Projects** (a project bundles design + field + receiver + sun + run settings and is the "save my work" unit).

- Built-ins, locked (usable, duplicable, never editable/deletable): the manuscript baseline — receiver configs **Prime focus 35.3 m**, **Axicon 27 m / 20° / 14 m**, **Cassegrain relay**, and the manuscript heliostat (5 × 3 m) in its three figures.
- One-click **"Use"** on a receiver config swaps the receiver while the field stays put.
- **Storage directory is the user's choice.** The Library footer shows where everything is stored, with a Change button — pick any working/project directory (a shared drive, a repo folder). Default: a Coddington folder under the home directory (replacing today's hidden `~/.heliostat`). Moving the directory migrates the files.
- **Interchange**: Import and Export for **SolTrace** and **SolarPILOT**, plus plain heliostat-positions CSV. Scope: field positions, heliostat geometry, receiver/secondary geometry, **and sun shape + optical-error settings wherever a direct equivalent exists** (both tools speak pillbox/Gaussian sun shapes and slope error); anything without an equivalent is reported at import/export rather than silently dropped. (SolTrace is NREL open-source freeware; reading and writing its file format is ordinary interoperability.)
- **Migration**: old saved setups appear under Projects with an import badge; importing converts what maps and lists what didn't. Originals untouched.

## 5b. Branding

Coddington is made by **Ryker Optics LLC**. Placement stays quiet:

- A small **Ryker Optics lockup** (the provided `ryker-optics-lockup.svg`) at the bottom of the workspace sidebar, with an **About** link.
- **About dialog**: full lockup, app version, license, storage-directory path, and third-party credits (three.js, SolTrace format notes).
- The **Coddington app logo** occupies the top-bar slot next to the app name — a dashed placeholder until the real mark exists (Ryker is generating the SVG separately).
- The desktop build's window/taskbar icon uses the app logo once it exists.

## 6. What this replaces

The current two-panel layout, the five separate tab strips, the results-only 3D view, the floating inspector's private edit state, and the "tower height" alias box. Saved setups remain readable for import; `/api/trace` and `/api/field/trace` keep working for scripts.

## 7. Open questions (to be answered at sign-off)

Resolved through rev 5: live apply everywhere except slow operations (explicit Apply); analysis runs save with the project; interchange includes sun shape and optical errors where equivalents exist; roads support arbitrary angle, partial extent, and rings; honeycomb layout; z = 0 = heliostat plane, ground offset default 2 500 mm; error defaults 0 / 0 / 90 % (manuscript); receiver types per layout + entrance-aperture offset; degenerate geometries trace as flat mirrors; annual Year estimate; Ryker Optics branding placement.

All questions resolved at sign-off (2026-08-22): year sampling is 12 declination-spaced days with a 7-day symmetry fast mode; cylindrical/frustum take radius(es) + height + center height, with radially-offset per-heliostat aiming; the receiver is positionable anywhere above the heliostat plane, defaulting to field center.

## 8. Acceptance checklist

1. Open the app cold: full 3D scene visible in under a second, no trace pressed; the axicon reads apex-down.
2. Change axicon half angle in the sidebar: cone reshapes and corner rays move within ~⅓ s; raise it far enough and the aperture-radius field warns amber with the needed minimum while missing rays draw dashed red — nothing auto-grows.
3. Click the cone, edit aperture radius in the inspector: instant update, sidebar shows the same number and the same warning state.
4. Enter an impossible value: red message at that field, scene keeps last valid geometry, recovery needs no reload.
5. Expand Field: plan view appears; drag a heliostat and see the override counted; reset restores the spiral; "back to 3D" always works. Add an 8 m South-only access road and a 6 m ring road at 60 m: the crossed heliostats disappear, the counts update, the North field is untouched. Switch layout to radial staggered (or honeycomb) and back: each keeps its parameters. Drag a heliostat close enough to the tower that it can't hit the cone: it highlights red and the warning counts it.
6. Expand Receiver & Tower: elevation appears with dimension callouts referenced to the heliostat-plane datum, ground offset as its own dimension; editing a callout edits the sidebar field.
7. Swap receiver config from the Library: field untouched.
8. Export the project to SolTrace, reimport it: same field, geometry, sun shape, and optical errors; unmapped settings reported.
9. In Analysis, click a timestep: its irradiance map appears immediately, from the map the sweep already made — no re-trace. Run a sweep, close the app, reopen: the run is still there; Discard removes it. Run a Year estimate: annual MWh reported with the per-day curve visible.
9a. Set axicon half angle to 0°: the scene shows a flat disc, the trace runs, an info badge says "degenerate: tracing as flat mirror" — no error. Same for a Cassegrain that solves to a flat relay. Set a positive aperture → receiver offset under prime focus: flux reports at the offset receiver.
9b. In Heliostat Shape, the previewed heliostat is named with its slant range; switching it re-solves the sag map; "View shape" from a workspace heliostat lands on that heliostat. Changing slope error changes the next Monte Carlo result.
10. 10,000-heliostat field: smooth orbiting; trace politely refuses over 1,000; the layout regeneration presents Apply instead of stalling.
11. Save project, close, reopen: identical state, overrides and roads included. Change the storage directory: everything moves and keeps working.
12. The desktop build behaves identically offline.
