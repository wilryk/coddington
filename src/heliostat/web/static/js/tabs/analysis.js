// Analysis tab (docs/ui-spec.md 4, mockup M7): day sweeps and year estimates
// against the current project -- background jobs with progress/cancel, the
// energy plots, a per-timestep table, CSV export, an on-demand irradiance
// map for whichever timestep is selected, and saved runs that persist with
// the project.
//
// Build-once/els/render(container, ctx) exactly like js/tabs/shape.js. All
// of it -- the day-sweep and year-estimate runs (job id, poll handle, last
// snapshot/result), the selected timestep, and the saved-runs state -- is
// module-local, not store state (js/library.js's drawer state is the same
// carve-out): a run has to survive a tab switch, so it cannot depend on
// render() being called at all. The poll loops run on their own setTimeout
// chains and only touch the DOM when this tab's section is not hidden.
//
// Saved runs (docs/ui-spec.md 4, "runs save with the project") live in the
// `runs` library collection -- api.js's generic getLibrary/getLibraryEntry/
// saveLibraryEntry/deleteLibraryEntry already work for it, so no dedicated
// wrappers were needed there. A run's document is {kind, project_name,
// request, result, flux_pngs} (heliostat.web.app.SavedRunDocument); this
// module is the only writer and the only reader of that shape. When a
// project is active (store ui.projectName set), saving a run also appends
// its name to ui.projectRuns, which js/project.js's serializeProject reads
// back into the saved project document's own `runs` field -- so this
// module never edits the project document directly, it only keeps that one
// array in step and lets the existing save path carry it along.
import { store } from "../store.js";
import { setVal, segButton } from "../fields.js";
import {
  receiverKindFor,
  apertureReceiverIsFlat,
  apertureDefaultCenter,
  apertureDefaultRadiusMm,
  clampToGridAxis,
  apertureClampRadius,
  apertureMetrics,
  apertureCurve,
  apertureDataToCanvas,
  apertureCanvasToData,
  paintApertureCanvas,
  paintApertureCurve,
  apertureCanvasEventPoint,
} from "../aperture.js";
import {
  buildDayRequest,
  buildFluxCsvRequest,
  buildTraceRequest,
  buildYearRequest,
  dayExportUrl,
  dayFluxUrl,
  dayFluxFeaCsvUrl,
  deleteLibraryEntry,
  getDayFluxGrid,
  getDaySecondaryGrid,
  getDayResult,
  getDayStatus,
  getLibrary,
  getLibraryEntry,
  getYearResult,
  getYearStatus,
  postDayCancel,
  postDayStart,
  postFieldFluxFeaCsv,
  postFieldSecondaryFluxFeaCsv,
  postFieldTrace,
  postFluxFeaCsv,
  postSecondaryFluxFeaCsv,
  postTrace,
  postYearCancel,
  postYearStart,
  saveLibraryEntry,
} from "../api.js";

const FIDELITY = [
  ["ultra_fast", "Ultra fast"],
  ["fast_accurate", "Fast accurate"],
  ["monte_carlo", "Monte Carlo"],
];

// docs/ui-spec-v0.2.md §A: one-line purpose subtitle plus what each mode
// trades away, verbatim from the signed-off table -- same wording as
// panels/run.js's own FIDELITY_TOOLTIPS (the workspace run bar and this
// tab are the same fidelity setting seen from two screens).
const FIDELITY_TOOLTIPS = {
  ultra_fast:
    "Field design optimization — explore layouts and geometry quickly. Trades shadowing/blocking accuracy and flux-map detail for speed.",
  // v0.2 followups item 2: Fast accurate stays the slower reference-cone
  // mode by owner decision -- its wording now says so, pointing individual-
  // heliostat work here and full-field work at Ultra fast instead. Kept
  // verbatim identical to panels/run.js's own FIDELITY_TOOLTIPS (see that
  // file's comment on why).
  fast_accurate:
    "Analyze a single heliostat with the highest peak-flux fidelity of any mode — deterministic and noise-free, but for full-field work, reach for Ultra fast instead.",
  monte_carlo:
    "Model the final design with precision, including all error sources — the only mode that applies measured error maps and pointing error per ray. More rays reduce noise, at the cost of speed.",
};

const DEFAULT_DATE = "2026-03-21"; // the server's own DaySite default
const DEFAULT_HOUR_STEP = 0.5;
const DEFAULT_MIN_ELEVATION_DEG = 5.0; // the server's own DayTraceRequest/YearTraceRequest default

let built = false;
let els = {};

let lastContainer = null;

// -- day-sweep run state (module-local -- see header) ----------------------
let formDate = DEFAULT_DATE;
// Shared by the day sweep and the year estimate -- both YearTraceRequest and
// DayTraceRequest take the same hour_step (max spacing between samples), so
// one control covers both rather than duplicating it per panel. See
// startYear()'s buildYearRequest call and estimateYearTimesteps.
let formHourStep = DEFAULT_HOUR_STEP;
// Shared the same way -- both skip timesteps below this sun elevation
// (heliostat.solar.build_time_grid's min_elevation_deg).
let formMinElevationDeg = DEFAULT_MIN_ELEVATION_DEG;

let jobId = null;
let jobSnapshot = null; // last /api/day/status (or /start) response
let dayResult = null; // /api/day/result payload, once terminal
let resultJobId = null; // which job dayResult belongs to (for the export link)
let dayError = null;
let starting = false;
let cancelling = false;
let pollTimer = null;

// -- selected timestep + its irradiance map ---------------------------------
let selectedStepIndex = null;
// The trace-shaped body the current run was started with, and per-step maps
// already traced for it, keyed `${resultJobId}:${stepIndex}`.
let sweepRequest = null;
let sweepPhysicsKey = null;
const fluxCache = new Map();
let fluxPngBase64 = null;
// Set instead of fluxPngBase64 when the sweep already stored this step's map
// server-side (has_flux_map) -- a real URL for the <img>, not a fetch at all.
let fluxSrcUrl = null;
let fluxPeakKwM2 = null;
// Spec §C: the current step's "secondary" response block (app.py's
// _secondary_payload), when there is one -- set either from a live re-trace
// (the cache/fetch branch of scheduleFluxFetch below) or, now that the
// stored-step gap is closed, from that step's own stored secondary blob
// (getDaySecondaryGrid, fetched in the has_flux_map/resultJobId branch).
// Absent (null) for a reopened saved run's flux_pngs, which carry no
// secondary data either way -- see scheduleFluxFetch's own comment there.
let fluxSecondary = null;
// v0.2 followups item 1, mockup M15: this step's own per-heliostat rows
// (id/x_mm/y_mm/power_w/...), when there are any -- ONLY ever set from a
// live on-demand re-trace of this step (the scheduleFluxFetch branch below
// that calls postFieldTrace, whose response carries its own `heliostats`
// array; app.py's field-trace endpoints). A step served from the sweep's
// own STORED map (fluxSrcUrl, the common case) never had a live field-trace
// response land for it -- has_flux_map only kept the PNG/CSV/grid blobs, not
// a per-heliostat breakdown -- so this stays null there, same honest gap a
// reopened saved run's flux_pngs already lives with. Field stays disabled
// with a tooltip in both of those cases (paintFluxSurfaceSelector).
let fluxHeliostats = null;
let fluxLoading = false;
let fluxError = null;
let fluxTimer = null;
let fluxController = null;
// A stored step's own secondary blob, cached per `${jobId}:${stepIndex}`
// once fetched -- a finished run's blob never changes, same idiom as
// fluxCache/apertureGridCache. `null` means "fetched, and this step has
// none" (prime focus, an older run), distinct from "not fetched yet" (key
// absent) so a confirmed absence is never refetched.
const storedSecondaryCache = new Map();
// True only while a stored step's own secondary blob is in flight --
// paintFluxSurfaceSelector uses it to show a "loading" tooltip instead of
// the honest "carries none" one during that brief window.
let storedSecondaryLoading = false;

// -- analysis aperture (docs/ui-spec-v0.2.md §M.4) --------------------------
// A draggable/resizable circle on the selected timestep's own flux grid,
// read out live: radius, power within, % of collected, average flux,
// average concentration (avg flux / DNI), plus an encircled-power curve.
// EVERYTHING below (apertureMetrics/apertureCurve/paintApertureCanvas) is
// pure post-processing on a grid already fetched from the server -- no
// function here ever calls postTrace/postFieldTrace or touches
// dayResult/jobSnapshot/sweepRequest; the trace, intercept and collected
// totals a sweep already produced never change because an aperture moved.
const apertureGridCache = new Map(); // `${resultJobId}:${stepIndex}` -> grid dict
let apertureGrid = null; // the SELECTED step's raw grid, once fetched
let apertureGridLoading = false;
let apertureGridError = null;
// User-set aperture, in the grid's own (u, v) mm frame -- null means "not
// yet touched", i.e. use the default (grid/step centroid, a radius derived
// from the step's own rms_radius_mm). Persists across a timestep switch
// within the same run on purpose: the whole point of the encircled-power
// curve is watching one FIXED physical region's catch change through the
// day, not a spot that redefines itself every time you scrub.
let apertureCenterUMm = null;
let apertureCenterVMm = null;
let apertureRadiusMm = null;
let apertureDrag = null; // {mode: "move"|"resize"} while a pointer drag is live
// A reopened saved run's frozen annotation (heliostat.web.app.SavedRunDocument
// .aperture) -- shown verbatim, never recomputed (see openSavedRun/paintAperturePanel).
let reopenedAperture = null;

// -- sweep drill-down to one heliostat (docs/ui-spec-v0.2.md §M.2) ----------
// Reuses the exact mechanism scheduleFluxFetch already uses for an
// uncached field timestep map -- a single-heliostat trace at that step's
// stored sun position -- except through /api/trace (never
// /api/field/trace: this is one mirror's own footprint, not a field sum),
// and with the chosen heliostat's own position rather than the field's.
// Cached per (heliostat id, step index), same idiom as fluxCache.
let drillHeliostatId = null;
const heliostatFootprintCache = new Map(); // `${id}:${stepIndex}` -> {png, peak}
let heliostatFootprintPngBase64 = null;
let heliostatFootprintPeakKwM2 = null;
let heliostatFootprintLoading = false;
let heliostatFootprintError = null;
let heliostatFootprintTimer = null;
let heliostatFootprintController = null;

// -- footprint "click to enlarge" overlay ------------------------------------
// The owner's ask (v0.2 followups): the inline drill-down thumbnail above is
// too small to read a hot spot closely, and CSS-stretching it would only
// blur the same 5.6x4.6in/dpi=110 PNG _render_flux_png always rendered --
// so this re-traces the SAME heliostat/step at a higher dpi (more pixels,
// not stretched ones; see app.py's TraceRequest.flux_png_dpi) instead of
// just reusing heliostatFootprintPngBase64 bigger. Cached separately from
// heliostatFootprintCache (same `${id}:${stepIndex}` key) since it is a
// different, heavier render of the same trace, fetched only on demand.
const FOOTPRINT_OVERLAY_DPI = 300;
const footprintOverlayCache = new Map(); // `${id}:${stepIndex}` -> pngBase64
let footprintOverlayPngBase64 = null;
let footprintOverlayLoading = false;
let footprintOverlayError = null;
let footprintOverlayController = null;

// -- year-estimate run state (module-local -- see header) -------------------
let yearFastMode = true;
let yearJobId = null;
let yearJobSnapshot = null;
let yearResult = null; // /api/year/result payload, once terminal
let yearResultJobId = null;
let yearError = null;
let yearStarting = false;
let yearCancelling = false;
let yearPollTimer = null;
let yearSweepRequest = null;
let yearSweepPhysicsKey = null;

// -- §R: traced instant, a third Analysis result source (docs/ui-spec-v0.2.md
// §R, mockup M22) ------------------------------------------------------------
// A traced instant needs none of the day/year machinery above (job polling,
// per-step fetch/cache): a live field/single trace response already carries
// everything the shared instruments below need in one payload (full
// per-heliostat rows, flux_grid, secondary -- §R's own point). So this reads
// straight off ui.traceResult/ui.traceRequest (the same shared state 3D
// View's own results dock reads -- js/main.js's traceSucceeded) rather than
// fetching or re-tracing anything, and the shared right-column instruments
// (paintFluxPanel/paintAperturePanel/paintDrillPanel and their supporting
// functions) are made source-aware by branching on `activeSource` /
// `currentStepForAperture()` / `currentAnyApertureGrid()` rather than forked
// into duplicate day/instant code paths wherever that is practical.
let activeSource = "day"; // "day" | "year" | "instant" -- which left-column result currently drives the right column.
// A saved SavedRunDocument(kind="instant"), once reopened -- {request,
// result, aperture}, the same shape a day/year run's own reopen already
// builds (aperture is this run's frozen §M.4 snapshot, or null). `null`
// means "show whatever ui.traceResult currently is" (the live case).
let reopenedInstant = null;
let instantSavedName = null;
let instantSaving = false;
let instantRunError = null;
// ui.traceTimestamp last synced -- a change means a FRESH trace just landed
// (as opposed to, say, the user toggling a checkbox), which supersedes
// whatever was reopened/saved before it (§R: "transient by default...
// gone on the next trace or reload").
let lastSeenInstantTimestamp = undefined;

// -- saved runs (docs/ui-spec.md 4: "runs save with the project") -----------
// `dayRunSavedName`/`yearRunSavedName` is the library entry name the run
// currently on screen was saved as (or reopened from) -- null means "not
// saved", which is what disables "Discard" and enables "Save".
let dayRunSavedName = null;
let dayRunSaving = false;
let dayRunError = null;
let yearRunSavedName = null;
let yearRunSaving = false;
let yearRunError = null;
// Set when the on-screen day result came from a saved run rather than a
// live job -- its flux maps live in this object, not on the (long gone)
// server-side job, so scheduleFluxFetch() reads them from here first.
let reopenedDayFluxPngs = null;

// The active project's own saved runs (ui.projectName's `runs` list,
// resolved to {name, kind, saved_at}), refreshed whenever that project
// changes -- see syncProjectRuns(). `undefined` (not null/a string) so the
// very first paint always triggers a sync, even for the no-project case.
let lastSyncedProjectName = undefined;
let projectRunEntries = [];
let projectRunsLoading = false;
let projectRunsError = null;

// "Manage saved runs" overlay: every run in the library, any project.
let manageOpen = false;
let manageLoading = false;
let manageError = null;
let manageEntries = []; // [{name, kind, project_name, saved_at, size_bytes}]

// -- formatting --------------------------------------------------------------

function fmtDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return null;
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const mins = Math.floor(seconds / 60);
  return `${mins} min ${Math.round(seconds - mins * 60)} s`;
}

function fmtHHMM(hour) {
  if (hour == null || !Number.isFinite(hour)) return "—";
  let h = Math.floor(hour);
  let m = Math.round((hour - h) * 60);
  if (m === 60) {
    m = 0;
    h += 1;
  }
  h = ((h % 24) + 24) % 24;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function fmtPower(w) {
  if (w == null || !Number.isFinite(w)) return "—";
  if (Math.abs(w) >= 1e6) return (w / 1e6).toFixed(2) + " MW";
  return (w / 1e3).toFixed(1) + " kW";
}

function fmtEnergy(kwh) {
  if (kwh == null || !Number.isFinite(kwh)) return "—";
  if (Math.abs(kwh) >= 1000) return (kwh / 1000).toFixed(2) + " MWh";
  return kwh.toFixed(1) + " kWh";
}

function fmtMWh(mwh) {
  if (mwh == null || !Number.isFinite(mwh)) return "—";
  return mwh.toFixed(mwh >= 100 ? 0 : mwh >= 10 ? 1 : 2) + " MWh";
}

function fmtBytes(n) {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + " KB";
  return n + " B";
}

function fmtFlux(kwM2) {
  if (kwM2 == null || !Number.isFinite(kwM2)) return "—";
  if (Math.abs(kwM2) >= 1000) return (kwM2 / 1000).toFixed(2) + " MW/m²";
  return kwM2.toFixed(1) + " kW/m²";
}

function mFmt(mm) {
  if (mm == null || !Number.isFinite(mm)) return "—";
  const m = Math.round((mm / 1000) * 10) / 10;
  return (Number.isInteger(m) ? String(m) : m.toFixed(1)) + " m";
}

// -- duration estimate: warn before a long run starts ------------------------
// A year estimate on the full 643-heliostat default field is ~93 timesteps at
// ~40s each -- about an hour -- started, until now, with no warning. This is
// a ROUGH pre-start estimate only (never the live job's own progress/ETA,
// which is exact): (heliostats * timesteps) scaled by a seconds-per-unit
// cost. That cost comes from this browser's own last finished run of the
// same fidelity mode (localStorage, survives a reload, not shared across
// machines) when one exists; otherwise a rough built-in guess per mode.
// Heliostat count and trace cost both depend on backend/ray count/mirror
// facets in ways this file has no business modelling -- honesty about being
// "rough" matters more than precision here.
const ROUGH_TRACE_COST_S_PER_UNIT = {
  ultra_fast: 0.015,
  fast_accurate: 0.05,
  monte_carlo: 0.2,
};

function traceCostStorageKey(mode) {
  return `heliostat.traceCostSPerUnit.${mode}`;
}

function recordTraceCost(mode, elapsedS, nHeliostats, nTimesteps) {
  const units = Math.max(1, nHeliostats || 1) * Math.max(1, nTimesteps || 0);
  if (!Number.isFinite(elapsedS) || elapsedS <= 0 || !(units > 0)) return;
  try {
    localStorage.setItem(traceCostStorageKey(mode), String(elapsedS / units));
  } catch (e) {
    // localStorage can be unavailable (private browsing, disabled) -- the
    // estimate just falls back to the rough built-in constant below.
  }
}

function traceCostSPerUnit(mode) {
  try {
    const stored = parseFloat(localStorage.getItem(traceCostStorageKey(mode)));
    if (Number.isFinite(stored) && stored > 0) return stored;
  } catch (e) {
    // see recordTraceCost
  }
  return ROUGH_TRACE_COST_S_PER_UNIT[mode] || ROUGH_TRACE_COST_S_PER_UNIT.ultra_fast;
}

// Rough timestep count for one day's sweep: daylight is roughly 10-12 h at
// the manuscript's near-equatorial site, and the elevation floor trims a bit
// more off each end -- not worth modelling exactly for a "rough estimate"
// line, so this just assumes an 11 h window, shrunk a little further for a
// stricter floor.
function estimateDayTimesteps(hourStep, minElevationDeg) {
  const daylightH = Math.max(1, 11 - (minElevationDeg || 0) * 0.15);
  return Math.max(2, Math.ceil(daylightH / Math.max(0.05, hourStep)) + 1);
}

function estimateYearTimesteps(fastMode, hourStep, minElevationDeg) {
  const nDates = fastMode ? 7 : 12;
  return nDates * estimateDayTimesteps(hourStep, minElevationDeg);
}

function currentHeliostatCount(ctx) {
  const doc = store.get("doc");
  if (doc.field.mode !== "field") return 1;
  const n = ctx && ctx.geometry && ctx.geometry.heliostats && ctx.geometry.heliostats.length;
  return n || 1;
}

function estimateDurationS(mode, nHeliostats, nTimesteps) {
  const units = Math.max(1, nHeliostats) * Math.max(1, nTimesteps);
  return traceCostSPerUnit(mode) * units;
}

// The estimate line ALWAYS shows (an estimate that vanishes when the run
// gets short reads as a bug -- user report, 2026-08-26: changing the
// timestep from 0.5 h to 3 h made the estimate "disappear"). The threshold
// now only decides the styling: past a couple of minutes it wears the
// amber warning class, under it a quiet hint.
const DURATION_WARNING_THRESHOLD_S = 120;

function durationWarningText(estimateS, nHeliostats, nTimesteps) {
  if (!(estimateS > 0)) return null;
  const dur = fmtDuration(estimateS) || `${Math.round(estimateS)} s`;
  const caveat =
    estimateS > DURATION_WARNING_THRESHOLD_S
      ? " This is a rough guess, not a promise -- actual time depends on hardware and settings."
      : "";
  return `Rough estimate: about ${dur} for ${nHeliostats} heliostat(s) x ${nTimesteps} timesteps.` + caveat;
}

function applyDurationHint(el, estimateS, text) {
  el.hidden = !text;
  if (!text) return;
  el.textContent = text;
  // fieldwarn is the amber warning look; fieldhint the quiet one.
  el.className = estimateS > DURATION_WARNING_THRESHOLD_S ? "fieldwarn" : "fieldhint";
}

// -- subject strip: what's being analyzed, read from doc + ctx.geometry -----

function opticsSummary(doc) {
  const p = doc.opticsParams[doc.optics] || {};
  if (doc.optics === "prime_focus") return `Prime focus ${mFmt(p.focus_height_mm)}`;
  if (doc.optics === "axicon") return `Axicon ${mFmt(p.apex_height_mm)} / ${p.half_angle_deg}° / ${mFmt(p.aperture_radius_mm)}`;
  if (doc.optics === "cassegrain") return `Cassegrain relay`;
  return doc.optics;
}

function fieldSummary(doc, geometry) {
  const count = (geometry && geometry.heliostats && geometry.heliostats.length) || 0;
  if (doc.field.mode !== "field") return "1 heliostat";
  if (doc.field.layout === "fermat") {
    const f = doc.field.fermat;
    const rmin = f.r_min_m != null ? f.r_min_m : "—";
    const rmax = f.r_max_m != null ? f.r_max_m : "—";
    return `${count} heliostats (Fermat ${rmin}–${rmax} m)`;
  }
  if (doc.field.layout === "manuscript") return `${count} heliostats (Manuscript 643)`;
  return `${count} heliostats (radial staggered)`;
}

function designSummary(doc) {
  const surface = doc.design.surface;
  const type = doc.design.type;
  if (type === "rect") {
    const p = doc.designParams.rect;
    return `${(p.width_mm / 1000).toFixed(1)} × ${(p.height_mm / 1000).toFixed(1)} m rectangle — ${surface}`;
  }
  if (type === "grid") {
    const p = doc.designParams.grid;
    return `${p.n_u}×${p.n_v} facet grid — ${surface}`;
  }
  const p = doc.designParams.custom;
  const n = (p && p.vertices_mm && p.vertices_mm.length) || 0;
  return `custom outline, ${n} vertices — ${surface}`;
}

// -- request building ---------------------------------------------------------

function siteFromForm() {
  const parts = (formDate || "").split("-").map(Number);
  if (parts.length !== 3 || parts.some((p) => !Number.isFinite(p))) return null;
  const [year, month, day] = parts;
  return { year, month, day };
}

// -- day-sweep lifecycle ------------------------------------------------------

function resetRunState() {
  if (jobId) publishAnalysisJob("day", null);
  jobId = null;
  jobSnapshot = null;
  dayResult = null;
  resultJobId = null;
  dayError = null;
  cancelling = false;
  selectedStepIndex = null;
  sweepRequest = null;
  sweepPhysicsKey = null;
  fluxCache.clear();
  clearFlux();
  clearAperture();
  clearHeliostatFootprint();
  drillHeliostatId = null;
  heliostatFootprintCache.clear();
  // A fresh sweep supersedes whatever was on screen before -- including a
  // reopened saved run, which has no live job to poll or discard through.
  dayRunSavedName = null;
  dayRunError = null;
  reopenedDayFluxPngs = null;
  // §R: starting (or reopening) a day sweep is the day source becoming the
  // freshest result -- switch the right column to it, same as picking it in
  // the "Viewing" selector would.
  activeSource = "day";
}

// -- top-of-screen trace bar -------------------------------------------------
// The #tracebar in main.js shows any running trace on every tab. Field traces
// publish through ui.traceBusy/ui.traceProgress; day sweeps and year estimates
// publish here through ui.analysisJob, cleared the moment the job stops being
// "running" (finished, failed, or cancelled -- the poll loop is the source of
// truth, same as the in-tab controls).
function publishAnalysisJob(kind, snap) {
  if (!snap || snap.state !== "running") {
    if (store.get("ui.analysisJob")) store.set("ui.analysisJob", null);
    return;
  }
  store.set("ui.analysisJob", {
    kind,
    detail: snap.detail || null,
    done: snap.done,
    total: snap.total,
    frac: snap.frac != null ? snap.frac : null,
    eta_s: snap.eta_s != null ? snap.eta_s : null,
  });
}

// Called by main.js's #tracebar Cancel when the running trace is one of ours.
export function cancelActiveAnalysisJob() {
  if (jobId && !cancelling) cancelSweep();
  else if (yearJobId && !yearCancelling) cancelYear();
}

function startSweep() {
  const site = siteFromForm();
  if (!site) {
    dayError = "Pick a date first.";
    paintIfVisible();
    return;
  }
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  resetRunState();
  starting = true;
  paintIfVisible();

  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildDayRequest(doc, ui, {
    site,
    hour_step: formHourStep,
    min_elevation_deg: formMinElevationDeg,
  });
  sweepRequest = body;
  sweepPhysicsKey = physicsKey(body);

  postDayStart(body)
    .then((snap) => {
      starting = false;
      jobId = snap.job_id;
      jobSnapshot = snap;
      publishAnalysisJob("day", snap);
      paintIfVisible();
      schedulePoll();
    })
    .catch((err) => {
      starting = false;
      dayError = (err && err.message) || "Could not start the day sweep.";
      paintIfVisible();
    });
}

function cancelSweep() {
  if (!jobId || cancelling) return;
  cancelling = true;
  paintIfVisible();
  postDayCancel(jobId).catch(() => {
    // The poll loop below is the source of truth for whether the job
    // actually stopped -- a failed cancel call just means the job may run
    // to completion, not that the UI should get stuck saying "cancelling".
  });
}

function schedulePoll() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(pollTick, 600);
}

function pollTick() {
  pollTimer = null;
  if (!jobId) return;
  const thisJob = jobId;
  getDayStatus(thisJob)
    .then((snap) => {
      if (jobId !== thisJob) return; // superseded by a newer run
      jobSnapshot = snap;
      publishAnalysisJob("day", snap);
      if (snap.state === "running") {
        paintIfVisible();
        schedulePoll();
        return;
      }
      cancelling = false;
      if (snap.state === "error") {
        dayError = snap.error || "The day sweep failed.";
        paintIfVisible();
        return;
      }
      fetchResult(thisJob);
    })
    .catch((err) => {
      if (jobId !== thisJob) return;
      dayError = (err && err.message) || "Lost track of the run.";
      cancelling = false;
      publishAnalysisJob("day", null);
      paintIfVisible();
    });
}

function fetchResult(forJob) {
  getDayResult(forJob)
    .then((data) => {
      if (jobId !== forJob) return;
      dayResult = data;
      resultJobId = forJob;
      dayError = null;
      if (dayResult.steps && dayResult.steps.length) {
        selectedStepIndex = pickDefaultStepIndex(dayResult.steps);
        scheduleFluxFetch();
        scheduleApertureGridFetch();
        scheduleHeliostatFootprintFetch();
      }
      // Feeds the NEXT run's rough duration estimate -- see recordTraceCost.
      // Uses done timesteps (not the requested total), so a cancelled run's
      // real cost still teaches the estimate something rather than nothing.
      if (jobSnapshot) {
        recordTraceCost(sweepRequest && sweepRequest.mode, jobSnapshot.elapsed_s, dayResult.n_heliostats, jobSnapshot.done);
      }
      paintIfVisible();
    })
    .catch((err) => {
      if (jobId !== forJob) return;
      dayError = (err && err.message) || "Could not load the day sweep result.";
      paintIfVisible();
    });
}

function pickDefaultStepIndex(steps) {
  let best = 0;
  let bestEl = -Infinity;
  steps.forEach((s, i) => {
    if (s.solar_el_deg != null && s.solar_el_deg > bestEl) {
      bestEl = s.solar_el_deg;
      best = i;
    }
  });
  return best;
}

// -- irradiance map for the selected timestep --------------------------------
// The day result carries no per-step flux image, so a click fetches one: a
// full field trace at that step's own sun angles, debounced and abortable
// exactly like js/tabs/shape.js's preview fetches, so clicking down the
// table doesn't pile up traces.

function clearFlux() {
  if (fluxTimer) {
    clearTimeout(fluxTimer);
    fluxTimer = null;
  }
  if (fluxController) {
    fluxController.abort();
    fluxController = null;
  }
  fluxPngBase64 = null;
  fluxSrcUrl = null;
  fluxPeakKwM2 = null;
  fluxHeliostats = null;
  fluxLoading = false;
  fluxError = null;
}

function clearAperture() {
  apertureGrid = null;
  apertureGridLoading = false;
  apertureGridError = null;
  apertureCenterUMm = null;
  apertureCenterVMm = null;
  apertureRadiusMm = null;
  apertureDrag = null;
  reopenedAperture = null;
}

function clearHeliostatFootprint() {
  if (heliostatFootprintTimer) {
    clearTimeout(heliostatFootprintTimer);
    heliostatFootprintTimer = null;
  }
  if (heliostatFootprintController) {
    heliostatFootprintController.abort();
    heliostatFootprintController = null;
  }
  heliostatFootprintPngBase64 = null;
  heliostatFootprintPeakKwM2 = null;
  heliostatFootprintLoading = false;
  heliostatFootprintError = null;
  closeFootprintOverlay();
}

// What a finished run measured: the mirror, the optics and where the
// heliostats stand. The sweep's own form fields (date, timestep, fidelity)
// are settings for the NEXT run, so changing them does not invalidate a
// result that is already on screen.
function physicsKey(body) {
  if (!body) return null;
  return JSON.stringify({
    design: body.design,
    optics: body.optics,
    optics_params: body.optics_params,
    layout: body.layout || null,
    heliostat_x_mm: body.heliostat_x_mm,
    heliostat_y_mm: body.heliostat_y_mm,
    // Fidelity is one app-wide setting, and it changes the numbers, so a
    // run traced at a different one no longer describes the current setup.
    mode: body.mode,
    n_rays: body.n_rays != null ? body.n_rays : null,
  });
}

// -- §R: traced instant -- reading the shared live/reopened state ----------

// The trace this source is currently showing -- the live 3D View result
// (ui.traceResult, exactly what its own results dock reads) unless a saved
// kind="instant" run has been reopened, in which case that frozen result
// wins (same "a reopened saved run supersedes the live job" precedent as
// the day/year sources' own resultJobId=null branch in openSavedRun).
function currentInstantResult() {
  if (reopenedInstant) return reopenedInstant.result;
  return store.get("ui.traceResult") || null;
}

// The EXACT request body that produced currentInstantResult() -- needed
// (not rebuildable from the live doc/ui) for the FEA export and "Save this
// run" below, both of which must describe the trace actually on screen, not
// whatever the workspace says right now if it has been edited since.
function currentInstantRequest() {
  if (reopenedInstant) return reopenedInstant.request;
  return store.get("ui.traceRequest") || null;
}

// A day-sweep-step-shaped view of the current instant, so the shared
// aperture/drill-down machinery below (written against a day sweep's own
// `steps[i]` shape) can read one without forking: `has_flux_map` stands in
// for "is there a grid to build an aperture from", `hour` has no meaning
// for a bare instant (no day/site to place it in) and is left null.
function instantStepLike() {
  const r = currentInstantResult();
  const req = currentInstantRequest();
  if (!r) return null;
  return {
    has_flux_map: !!r.flux_grid,
    hour: null,
    solar_az_deg: req ? req.solar_az_deg : null,
    solar_el_deg: req ? req.solar_el_deg : null,
    power_w: r.power_w,
    peak_flux_kw_m2: r.peak_flux_kw_m2,
    dni_w_m2: r.dni_w_m2,
    centroid_mm: r.centroid_mm,
    rms_radius_mm: r.rms_radius_mm,
    n_heliostats: r.n_heliostats,
  };
}

// A monotonic tag for "which distinct instant is this" -- bumped whenever a
// fresh trace lands or a different saved run is reopened, so a cache keyed
// off it (heliostatFootprintCache, footprintOverlayCache -- both shared with
// the day/year sources via cacheStepKey() below) can never confuse two
// different instants that happen to share the same heliostat id.
let instantGeneration = 0;

// Detects a fresh trace landing (or the source switching to instant while
// one is already on screen) and, when it happens, (a) drops whatever was
// reopened/saved -- a live trace always supersedes those (§R: "gone on the
// next trace or reload") -- and (b) bumps instantGeneration so any cached
// per-heliostat footprint from the PREVIOUS instant is never served for
// this one. Cheap (one number compare) -- called every paint(), same idiom
// as syncProjectRuns.
function syncInstantSourceIfNeeded() {
  const ts = store.get("ui.traceTimestamp");
  if (ts === lastSeenInstantTimestamp) return;
  lastSeenInstantTimestamp = ts;
  reopenedInstant = null;
  instantSavedName = null;
  instantRunError = null;
  instantGeneration += 1;
}

// Which "step" a footprint-drill-down/overlay cache entry belongs to --
// the real selectedStepIndex for day/year (unchanged), or instantGeneration
// for the instant source (see its own comment above) -- so
// scheduleHeliostatFootprintFetch/fetchFootprintOverlayImage's shared cache
// keys stay correct for all three sources without three copies of either
// function.
function cacheStepKey() {
  return activeSource === "instant" ? `instant-${instantGeneration}` : selectedStepIndex;
}

// docs/ui-spec-v0.2.md §M.4's aperture grid for whichever source is active --
// apertureGrid itself stays the day source's own fetched-grid variable
// (scheduleApertureGridFetch's own cache/fetch machinery is day-only and
// untouched), while the instant source's grid needs no fetch at all: a live
// trace's flux_grid (opt-in, always requested -- api.js's buildTraceRequest)
// is already sitting on the response. Every reader of "the current
// aperture's grid" (paintAperturePanel, the pointer-drag handlers,
// buildApertureSnapshotForSave) goes through this rather than the bare
// apertureGrid variable, so instant needs no separate copy of any of them.
function currentAnyApertureGrid() {
  if (activeSource === "instant") {
    const r = currentInstantResult();
    return (r && r.flux_grid) || null;
  }
  return apertureGrid;
}

function selectSource(source) {
  if (source === activeSource) return;
  closeFootprintOverlay(); // switching sources invalidates whatever the overlay was showing
  activeSource = source;
  if (source === "instant") syncInstantSourceIfNeeded();
  paintIfVisible();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// §R's own backend gap, closed: a live field trace has no synchronous FEA
// export today (only inside the day-sweep job's stored per-step blobs), so
// this always goes through the network (unlike the day source's own
// dayFluxFeaCsvUrl, a plain link to an already-rendered file) -- same
// fetch-then-download idiom as exportSecondaryFluxFeaCsvForStep below.
// `postFn` picks the field- vs single-heliostat endpoint by whether the
// traced request carried a `layout` (fluxRequestFor's own idiom, mirrored).
function exportInstantFluxFeaCsv() {
  const body = currentInstantRequest();
  if (!body) return;
  const postFn = body.layout ? postFieldFluxFeaCsv : postFluxFeaCsv;
  els.fluxFeaCsv.textContent = "Exporting…";
  postFn(body)
    .then((blob) => downloadBlob(blob, "heliostat-flux-fea.csv"))
    .catch((err) => {
      fluxError = (err && err.message) || "CSV export failed.";
      paintIfVisible();
    })
    .finally(() => {
      els.fluxFeaCsv.textContent = "Export CSV for FEA";
    });
}

function exportInstantSecondaryFluxFeaCsv() {
  const body = currentInstantRequest();
  if (!body) return;
  const postFn = body.layout ? postFieldSecondaryFluxFeaCsv : postSecondaryFluxFeaCsv;
  els.fluxSecFeaCsv.textContent = "Exporting…";
  postFn(body)
    .then((blob) => downloadBlob(blob, "heliostat-secondary-flux-fea.csv"))
    .catch((err) => {
      fluxError = (err && err.message) || "CSV export failed.";
      paintIfVisible();
    })
    .finally(() => {
      els.fluxSecFeaCsv.textContent = "Export CSV for FEA";
    });
}

// docs/ui-spec-v0.2.md §C leftover: the secondary's own FEA CSV for
// whichever timestep is selected -- same single-heliostat request shape as
// js/main.js's exportSecondaryFluxFeaCsv (buildFluxCsvRequest, no `layout`
// even in field mode: /api/trace/secondary_flux_fea.csv always reads
// heliostat_x_mm/y_mm), just at this step's own sun position instead of the
// live doc.sun one -- mirrors fluxRequestFor's own az/el override above.
function exportSecondaryFluxFeaCsvForStep() {
  if (activeSource === "instant") return exportInstantSecondaryFluxFeaCsv();
  const steps = dayResult && dayResult.steps;
  const step = steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;
  if (!step) return;
  const body = Object.assign(buildFluxCsvRequest(store.get("doc"), store.get("ui")), {
    solar_az_deg: step.solar_az_deg,
    solar_el_deg: step.solar_el_deg,
  });
  els.fluxSecFeaCsv.textContent = "Exporting…";
  postSecondaryFluxFeaCsv(body)
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "heliostat-secondary-flux-fea.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    })
    .catch((err) => {
      fluxError = (err && err.message) || "CSV export failed.";
      paintIfVisible();
    })
    .finally(() => {
      els.fluxSecFeaCsv.textContent = "Export CSV for FEA";
    });
}

function resultIsStale() {
  // §R: a live instant's own staleness is main.js's ui.staleResults (the
  // SAME rule 3D View's own results dock uses -- an edit made after this
  // trace landed); a reopened saved instant is a frozen record, never stale,
  // same as a reopened day/year run.
  if (activeSource === "instant") {
    return !reopenedInstant && !!store.get("ui.staleResults") && !!currentInstantResult();
  }
  if (!dayResult || !sweepPhysicsKey) return false;
  const current = physicsKey(buildTraceRequest(store.get("doc"), store.get("ui")));
  return current !== sweepPhysicsKey;
}

// The day job reports only scalars per timestep, so a map has to be traced
// on demand. It is traced from the body the sweep itself ran with, never
// from the live store: editing the workspace after a sweep must not silently
// render a map of a different heliostat than the sweep measured.
function fluxRequestFor(step) {
  const base = sweepRequest
    ? Object.assign({}, sweepRequest)
    : buildTraceRequest(store.get("doc"), store.get("ui"));
  delete base.site;
  delete base.hour_step;
  return Object.assign(base, {
    solar_az_deg: step.solar_az_deg,
    solar_el_deg: step.solar_el_deg,
  });
}

// Every exit path sets fluxLoading and repaints itself -- the loading
// indicator must never outlive the request it stands for, whether that
// request never happened (a stored map, a cache hit) or is still in flight.
function scheduleFluxFetch() {
  if (fluxTimer) clearTimeout(fluxTimer);
  if (fluxController) fluxController.abort();
  fluxTimer = null;
  fluxController = null;
  const steps = dayResult && dayResult.steps;
  const step = steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;
  if (!step) {
    fluxPngBase64 = null;
    fluxSrcUrl = null;
    fluxPeakKwM2 = null;
    fluxSecondary = null;
    fluxHeliostats = null;
    storedSecondaryLoading = false;
    fluxError = null;
    fluxLoading = false;
    paintIfVisible();
    return;
  }
  // A reopened saved run has no live job to serve a map from -- its maps
  // travel with the saved document itself instead (see SavedRunDocument).
  // SavedRunDocument keeps PNG bytes only, so a reopened run's secondary
  // map (if any) is gone the same way -- Secondary shows disabled with a
  // tooltip for it, same as any other secondary-less step.
  if (reopenedDayFluxPngs) {
    const png = reopenedDayFluxPngs[String(selectedStepIndex)];
    fluxPngBase64 = png || null;
    fluxSrcUrl = null;
    fluxPeakKwM2 = null;
    fluxSecondary = null;
    fluxHeliostats = null;
    storedSecondaryLoading = false;
    fluxError = png ? null : "This saved run kept no irradiance map for that timestep.";
    fluxLoading = false;
    paintIfVisible();
    return;
  }
  // The sweep already traced and rendered this timestep -- serve its own
  // map straight from the server, instantly, instead of re-tracing it.
  if (step.has_flux_map && resultJobId != null) {
    fluxPngBase64 = null;
    fluxSrcUrl = dayFluxUrl(resultJobId, selectedStepIndex);
    fluxPeakKwM2 = null; // the row's own peak_flux_kw_m2 covers the caption
    // A stored step's own blobs are the PNG/CSV/grid saved during the sweep
    // itself -- no live field-trace response ever landed for it, so there is
    // no per-heliostat breakdown to show (see fluxHeliostats's own comment).
    fluxHeliostats = null;
    fluxError = null;
    fluxLoading = false;
    // Spec §C's stored-step gap, closed: day_start now stores this step's
    // own secondary blob too, whenever the sweep asked for one and its
    // optics has a flux map (app.py's _day_secondary_grid_blob_key). Fetch
    // it here, cached per step since a finished run's blob never changes;
    // a 404 (prime focus, an older run from before this landed, or a step
    // the receiver-grid cap itself skipped) leaves fluxSecondary null, the
    // same honest disabled-selector state paintFluxSurfaceSelector already
    // renders -- just no longer a foregone conclusion for every stored step.
    const secKey = `${resultJobId}:${selectedStepIndex}`;
    if (storedSecondaryCache.has(secKey)) {
      fluxSecondary = storedSecondaryCache.get(secKey);
      storedSecondaryLoading = false;
    } else {
      fluxSecondary = null;
      storedSecondaryLoading = true;
      const jobIdAtRequest = resultJobId;
      const stepIndexAtRequest = selectedStepIndex;
      getDaySecondaryGrid(jobIdAtRequest, stepIndexAtRequest)
        .then((secondary) => {
          storedSecondaryCache.set(secKey, secondary);
          if (resultJobId === jobIdAtRequest && selectedStepIndex === stepIndexAtRequest) {
            fluxSecondary = secondary;
            storedSecondaryLoading = false;
            paintIfVisible();
          }
        })
        .catch(() => {
          // No stored blob for this step -- a routine, expected outcome
          // (not every step/optics has one), so cache the absence and move
          // on silently rather than surfacing it as a fluxError.
          storedSecondaryCache.set(secKey, null);
          if (resultJobId === jobIdAtRequest && selectedStepIndex === stepIndexAtRequest) {
            storedSecondaryLoading = false;
            paintIfVisible();
          }
        });
    }
    paintIfVisible();
    return;
  }
  fluxSrcUrl = null;
  // Every step of one finished run traces the same geometry at a different
  // sun position, so a map already fetched stays valid for that run.
  const cacheKey = `${resultJobId}:${selectedStepIndex}`;
  const cached = fluxCache.get(cacheKey);
  if (cached) {
    fluxPngBase64 = cached.png;
    fluxPeakKwM2 = cached.peak;
    fluxSecondary = cached.secondary || null;
    fluxHeliostats = cached.heliostats || null;
    fluxError = null;
    fluxLoading = false;
    paintIfVisible();
    return;
  }
  fluxLoading = true;
  fluxError = null;
  paintIfVisible();
  fluxTimer = setTimeout(() => {
    fluxTimer = null;
    const body = fluxRequestFor(step);
    fluxController = new AbortController();
    // A run with no layout is a single heliostat, which /api/field/trace
    // rejects -- that case belongs to /api/trace.
    const trace = body.layout ? postFieldTrace : postTrace;
    trace(body, fluxController.signal)
      .then((data) => {
        fluxController = null;
        fluxLoading = false;
        fluxPngBase64 = data.flux_png || null;
        fluxPeakKwM2 = data.peak_flux_kw_m2 != null ? data.peak_flux_kw_m2 : null;
        // Spec §C: buildTraceRequest (via fluxRequestFor) always asks for
        // this -- present whenever the current optics has a secondary flux
        // map (axicon/Cassegrain), null otherwise (app.py's
        // _secondary_maps_from_result).
        fluxSecondary = data.secondary || null;
        // v0.2 followups item 1: this on-demand branch is a real field trace
        // (or a single-heliostat one -- postTrace responses carry no
        // `heliostats` key at all, so this is naturally null there too,
        // same honest "single heliostat" disabled state as the overlay's).
        fluxHeliostats = Array.isArray(data.heliostats) ? data.heliostats : null;
        fluxError = fluxPngBase64 ? null : "No flux map came back for this timestep.";
        if (fluxPngBase64) {
          fluxCache.set(cacheKey, {
            png: fluxPngBase64,
            peak: fluxPeakKwM2,
            secondary: fluxSecondary,
            heliostats: fluxHeliostats,
          });
        }
        paintIfVisible();
      })
      .catch((err) => {
        fluxController = null;
        if (err && err.name === "AbortError") return;
        fluxLoading = false;
        fluxError = (err && err.message) || "Could not render the irradiance map.";
        paintIfVisible();
      });
  }, 250);
}

// -- analysis aperture: fetching the raw grid (docs/ui-spec-v0.2.md §M.4) ---
// The grid backing the aperture math only exists for a step the sweep
// itself stored a map for, served from a still-alive job -- exactly
// paintFluxPanel's own fluxSrcUrl/resultJobId condition for showing the §D
// FEA CSV export link. A reopened saved run has no live job to fetch a
// grid from; whatever aperture it carries is a frozen snapshot
// (reopenedAperture), shown as-is rather than recomputed -- see
// paintAperturePanel.
function apertureGridCacheKey(stepIndex) {
  return `${resultJobId}:${stepIndex}`;
}

function currentStepForAperture() {
  if (activeSource === "instant") return instantStepLike();
  const steps = dayResult && dayResult.steps;
  return steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;
}

function scheduleApertureGridFetch() {
  apertureDrag = null;
  const step = currentStepForAperture();
  if (!step || !step.has_flux_map || resultJobId == null) {
    apertureGrid = null;
    apertureGridLoading = false;
    apertureGridError = null;
    paintIfVisible();
    return;
  }
  const key = apertureGridCacheKey(selectedStepIndex);
  const cached = apertureGridCache.get(key);
  if (cached) {
    apertureGrid = cached;
    apertureGridLoading = false;
    apertureGridError = null;
    paintIfVisible();
    return;
  }
  // Clear the PREVIOUS step's grid rather than leaving it in place while
  // this one loads -- paintAperturePanel gates on apertureGrid being
  // truthy, and a stale grid there would render the wrong timestep's
  // picture/numbers for the gap between selecting a step and its own grid
  // landing.
  apertureGrid = null;
  apertureGridLoading = true;
  apertureGridError = null;
  paintIfVisible();
  getDayFluxGrid(resultJobId, selectedStepIndex)
    .then((grid) => {
      apertureGridCache.set(key, grid);
      apertureGrid = grid;
      apertureGridLoading = false;
      paintIfVisible();
    })
    .catch((err) => {
      apertureGrid = null;
      apertureGridLoading = false;
      apertureGridError = (err && err.message) || "Could not load the aperture grid.";
      paintIfVisible();
    });
}

// -- sweep drill-down: fetching one heliostat's own footprint (§M.2) --------

function currentFieldHeliostats() {
  // The live workspace's own resolved positions (ctx.geometry.heliostats,
  // set by render()'s lastCtx) -- the same source currentHeliostatCount
  // already reads, and subject to the identical caveat: if the field is
  // edited after this sweep ran, ids/positions here describe the CURRENT
  // field, not the one the sweep traced. resultIsStale() (physicsKey
  // already compares `layout`) already flags exactly that case across the
  // whole panel -- see paint()'s an-stale toggle on els.drillPanel below --
  // so this reads the live geometry without a bespoke staleness check of
  // its own.
  const geo = lastCtx && lastCtx.geometry;
  return (geo && geo.heliostats) || [];
}

function selectDrillHeliostat(id) {
  if (drillHeliostatId === id) return;
  closeFootprintOverlay(); // a new heliostat invalidates whatever the overlay was showing
  drillHeliostatId = id;
  scheduleHeliostatFootprintFetch();
  paintIfVisible();
}

// Same body-shaping idiom as fluxRequestFor above, but for ONE named
// heliostat rather than the field: drop layout/exclude_ids, set that
// heliostat's own position, keep the timestep's own stored sun angles.
// §R: the instant source's own base request (currentInstantRequest, the
// EXACT body that instant was traced with) takes the day-sweep's sweepRequest's
// place when active -- same "traced from the body the run itself used, never
// the live store" reasoning that comment already gives for the day case.
function heliostatFootprintRequestFor(step, heliostat) {
  const base =
    activeSource === "instant"
      ? Object.assign({}, currentInstantRequest())
      : sweepRequest
        ? Object.assign({}, sweepRequest)
        : buildTraceRequest(store.get("doc"), store.get("ui"));
  delete base.site;
  delete base.hour_step;
  delete base.layout;
  delete base.exclude_ids;
  return Object.assign(base, {
    solar_az_deg: step.solar_az_deg,
    solar_el_deg: step.solar_el_deg,
    heliostat_x_mm: heliostat.x_mm,
    heliostat_y_mm: heliostat.y_mm,
  });
}

function scheduleHeliostatFootprintFetch() {
  if (heliostatFootprintTimer) clearTimeout(heliostatFootprintTimer);
  if (heliostatFootprintController) heliostatFootprintController.abort();
  heliostatFootprintTimer = null;
  heliostatFootprintController = null;

  const step = currentStepForAperture();
  const heliostat =
    drillHeliostatId != null ? currentFieldHeliostats().find((h) => h.id === drillHeliostatId) : null;
  if (!step || !heliostat) {
    clearHeliostatFootprint();
    paintIfVisible();
    return;
  }
  const cacheKey = `${drillHeliostatId}:${cacheStepKey()}`;
  const cached = heliostatFootprintCache.get(cacheKey);
  if (cached) {
    heliostatFootprintPngBase64 = cached.png;
    heliostatFootprintPeakKwM2 = cached.peak;
    heliostatFootprintError = null;
    heliostatFootprintLoading = false;
    paintIfVisible();
    return;
  }
  heliostatFootprintLoading = true;
  heliostatFootprintError = null;
  paintIfVisible();
  heliostatFootprintTimer = setTimeout(() => {
    heliostatFootprintTimer = null;
    const body = heliostatFootprintRequestFor(step, heliostat);
    heliostatFootprintController = new AbortController();
    postTrace(body, heliostatFootprintController.signal)
      .then((data) => {
        heliostatFootprintController = null;
        heliostatFootprintLoading = false;
        heliostatFootprintPngBase64 = data.flux_png || null;
        heliostatFootprintPeakKwM2 = data.peak_flux_kw_m2 != null ? data.peak_flux_kw_m2 : null;
        heliostatFootprintError = heliostatFootprintPngBase64
          ? null
          : "No footprint came back for this heliostat.";
        if (heliostatFootprintPngBase64) {
          heliostatFootprintCache.set(cacheKey, {
            png: heliostatFootprintPngBase64,
            peak: heliostatFootprintPeakKwM2,
          });
        }
        paintIfVisible();
      })
      .catch((err) => {
        heliostatFootprintController = null;
        if (err && err.name === "AbortError") return;
        heliostatFootprintLoading = false;
        heliostatFootprintError = (err && err.message) || "Could not trace this heliostat's footprint.";
        paintIfVisible();
      });
  }, 250);
}

// -- footprint "click to enlarge" overlay ------------------------------------
// Same overlay chrome as "Manage saved runs" above (app.css's shared
// .overlay/.overlay-panel/.overlay-close -- the same lightbox the workspace
// run bar's flux thumbnail opens), built once in build() and toggled here.
// Opening it kicks off its own higher-dpi trace (see FOOTPRINT_OVERLAY_DPI
// above) rather than just showing heliostatFootprintPngBase64 bigger --
// that PNG is still the fixed dpi=110 render the inline thumbnail uses, and
// stretching it in CSS would only blur it, defeating the point of a closer
// look.
function openFootprintOverlay() {
  if (!heliostatFootprintPngBase64) return; // nothing loaded yet to zoom into
  const step = currentStepForAperture();
  const heliostat =
    drillHeliostatId != null ? currentFieldHeliostats().find((h) => h.id === drillHeliostatId) : null;
  if (!step || !heliostat) return;
  els.footprintOverlay.hidden = false;
  fetchFootprintOverlayImage(step, heliostat, `${drillHeliostatId}:${cacheStepKey()}`);
  paintIfVisible();
}

function closeFootprintOverlay() {
  if (footprintOverlayController) {
    footprintOverlayController.abort();
    footprintOverlayController = null;
  }
  footprintOverlayPngBase64 = null;
  footprintOverlayLoading = false;
  footprintOverlayError = null;
  if (built) els.footprintOverlay.hidden = true;
}

function fetchFootprintOverlayImage(step, heliostat, cacheKey) {
  const cached = footprintOverlayCache.get(cacheKey);
  if (cached) {
    footprintOverlayPngBase64 = cached;
    footprintOverlayLoading = false;
    footprintOverlayError = null;
    paintIfVisible();
    return;
  }
  if (footprintOverlayController) footprintOverlayController.abort();
  footprintOverlayLoading = true;
  footprintOverlayError = null;
  footprintOverlayPngBase64 = null;
  paintIfVisible();
  const body = Object.assign(heliostatFootprintRequestFor(step, heliostat), {
    flux_png_dpi: FOOTPRINT_OVERLAY_DPI,
  });
  footprintOverlayController = new AbortController();
  postTrace(body, footprintOverlayController.signal)
    .then((data) => {
      footprintOverlayController = null;
      footprintOverlayLoading = false;
      footprintOverlayPngBase64 = data.flux_png || null;
      footprintOverlayError = footprintOverlayPngBase64
        ? null
        : "No larger footprint came back for this heliostat.";
      if (footprintOverlayPngBase64) footprintOverlayCache.set(cacheKey, footprintOverlayPngBase64);
      paintIfVisible();
    })
    .catch((err) => {
      footprintOverlayController = null;
      if (err && err.name === "AbortError") return;
      footprintOverlayLoading = false;
      footprintOverlayError = (err && err.message) || "Could not load the larger footprint.";
      paintIfVisible();
    });
}

function selectStep(i) {
  if (selectedStepIndex === i) return;
  closeFootprintOverlay(); // a new timestep invalidates whatever the overlay was showing
  selectedStepIndex = i;
  scheduleFluxFetch();
  scheduleApertureGridFetch();
  scheduleHeliostatFootprintFetch();
  paintIfVisible();
}

// -- year-estimate lifecycle --------------------------------------------------
// Same background-job shape as the day sweep above, one endpoint family over
// (/api/year/* rather than /api/day/*), with no per-timestep flux map to fetch.

function resetYearRunState() {
  if (yearJobId) publishAnalysisJob("year", null);
  yearJobId = null;
  yearJobSnapshot = null;
  yearResult = null;
  yearResultJobId = null;
  yearError = null;
  yearCancelling = false;
  yearSweepRequest = null;
  yearSweepPhysicsKey = null;
  yearRunSavedName = null;
  yearRunError = null;
}

function startYear() {
  if (yearPollTimer) {
    clearTimeout(yearPollTimer);
    yearPollTimer = null;
  }
  resetYearRunState();
  activeSource = "year"; // §R: same "starting a run switches viewing to it" rule as resetRunState's day case.
  yearStarting = true;
  paintIfVisible();

  const doc = store.get("doc");
  const ui = store.get("ui");
  // No site fields exposed here, same as the day sweep's own date-only form
  // -- the server's YearSite defaults (the manuscript's site) fill the rest.
  const body = buildYearRequest(doc, ui, {
    site: {},
    fastMode: yearFastMode,
    hour_step: formHourStep,
    min_elevation_deg: formMinElevationDeg,
  });
  yearSweepRequest = body;
  yearSweepPhysicsKey = physicsKey(body);

  postYearStart(body)
    .then((snap) => {
      yearStarting = false;
      yearJobId = snap.job_id;
      yearJobSnapshot = snap;
      publishAnalysisJob("year", snap);
      paintIfVisible();
      scheduleYearPoll();
    })
    .catch((err) => {
      yearStarting = false;
      yearError = (err && err.message) || "Could not start the year estimate.";
      paintIfVisible();
    });
}

function cancelYear() {
  if (!yearJobId || yearCancelling) return;
  yearCancelling = true;
  paintIfVisible();
  postYearCancel(yearJobId).catch(() => {
    // As with the day sweep, the poll loop is the source of truth for
    // whether the job actually stopped.
  });
}

function scheduleYearPoll() {
  if (yearPollTimer) clearTimeout(yearPollTimer);
  yearPollTimer = setTimeout(yearPollTick, 600);
}

function yearPollTick() {
  yearPollTimer = null;
  if (!yearJobId) return;
  const thisJob = yearJobId;
  getYearStatus(thisJob)
    .then((snap) => {
      if (yearJobId !== thisJob) return;
      yearJobSnapshot = snap;
      publishAnalysisJob("year", snap);
      if (snap.state === "running") {
        paintIfVisible();
        scheduleYearPoll();
        return;
      }
      yearCancelling = false;
      if (snap.state === "error") {
        yearError = snap.error || "The year estimate failed.";
        paintIfVisible();
        return;
      }
      fetchYearResult(thisJob);
    })
    .catch((err) => {
      if (yearJobId !== thisJob) return;
      yearError = (err && err.message) || "Lost track of the run.";
      yearCancelling = false;
      publishAnalysisJob("year", null);
      paintIfVisible();
    });
}

function fetchYearResult(forJob) {
  getYearResult(forJob)
    .then((data) => {
      if (yearJobId !== forJob) return;
      yearResult = data;
      yearResultJobId = forJob;
      yearError = null;
      // See fetchResult's own call -- same rough-estimate bookkeeping, one
      // step up (timesteps across every traced date, not one day).
      if (yearJobSnapshot) {
        recordTraceCost(
          yearSweepRequest && yearSweepRequest.mode,
          yearJobSnapshot.elapsed_s,
          yearResult.n_heliostats,
          yearJobSnapshot.done
        );
      }
      paintIfVisible();
    })
    .catch((err) => {
      if (yearJobId !== forJob) return;
      yearError = (err && err.message) || "Could not load the year estimate result.";
      paintIfVisible();
    });
}

// physicsKey() is defined over exactly the fields YearTraceRequest shares
// with DayTraceRequest (design/optics/optics_params/layout/heliostat
// position/mode/n_rays), so the same function -- and the same staleness
// definition -- applies unchanged to a year result.
function yearResultIsStale() {
  if (!yearResult || !yearSweepPhysicsKey) return false;
  const current = physicsKey(buildTraceRequest(store.get("doc"), store.get("ui")));
  return current !== yearSweepPhysicsKey;
}

// -- saved runs (docs/ui-spec.md 4) -------------------------------------------

function runName(kind, label) {
  return `${kind}-${label}-${Date.now()}`;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error || new Error("could not read the flux map"));
    reader.readAsDataURL(blob);
  });
}

// Appends `name` to the ACTIVE project's own saved-run list and persists
// that one field back onto whatever project document is currently stored --
// fetched fresh rather than re-serialized from the live store, so this
// cannot silently resave unrelated workspace edits the user has not asked
// to save. No-op (but still updates the local cache) when no project is
// active: the run still saves standalone, just with nothing to attach to.
function attachRunToProject(name) {
  const projectName = store.get("ui.projectName");
  const current = Array.isArray(store.get("ui.projectRuns")) ? store.get("ui.projectRuns").slice() : [];
  if (current.indexOf(name) === -1) current.push(name);
  store.set("ui.projectRuns", current);
  if (!projectName) return Promise.resolve();
  return getLibraryEntry("projects", projectName).then((entry) => {
    const document = Object.assign({}, entry.document, { schema_version: 2, runs: current });
    return saveLibraryEntry("projects", projectName, document);
  });
}

function detachRunFromProjectNamed(projectName, name) {
  if (!projectName) return Promise.resolve();
  return getLibraryEntry("projects", projectName)
    .then((entry) => {
      const runs = (Array.isArray(entry.document.runs) ? entry.document.runs : []).filter((n) => n !== name);
      const document = Object.assign({}, entry.document, { schema_version: 2, runs });
      return saveLibraryEntry("projects", projectName, document);
    })
    .then(() => {
      if (store.get("ui.projectName") === projectName) {
        const current = Array.isArray(store.get("ui.projectRuns")) ? store.get("ui.projectRuns") : [];
        store.set("ui.projectRuns", current.filter((n) => n !== name));
      }
    })
    .catch(() => {
      // The run itself is already deleted by the time this runs (see the
      // callers) -- a project that fails to update its own list still
      // leaves a stale name in it, which openSavedRun's 404 handling below
      // treats as "not there anymore" rather than a hard failure.
    });
}

function detachRunFromProject(name) {
  return detachRunFromProjectNamed(store.get("ui.projectName"), name);
}

// projectRunEntries only ever changes through syncProjectRuns()'s own fetch,
// which runs on a ui.projectName CHANGE -- saving or discarding a run does
// not change that name, so without this the "Saved runs" bar would show a
// run that was just attached/detached only after the user switched projects
// and back. These update the local list directly instead of forcing a
// refetch.
function noteProjectRunAdded(name, kind) {
  if (!store.get("ui.projectName")) return;
  projectRunEntries = projectRunEntries.concat([{ name, kind, saved_at: new Date().toISOString() }]);
}

function noteProjectRunRemoved(name) {
  projectRunEntries = projectRunEntries.filter((e) => e.name !== name);
}

function saveDayRun() {
  if (!dayResult || !resultJobId || dayRunSaving || dayRunSavedName) return;
  dayRunSaving = true;
  dayRunError = null;
  paintIfVisible();

  const steps = dayResult.steps || [];
  const fluxIndices = [];
  steps.forEach((s, i) => {
    if (s.has_flux_map) fluxIndices.push(i);
  });

  Promise.all(
    fluxIndices.map((i) =>
      fetch(dayFluxUrl(resultJobId, i))
        .then((r) => r.blob())
        .then(blobToBase64)
        .then((b64) => [String(i), b64])
    )
  )
    .then((pairs) => {
      const flux_pngs = {};
      for (const [key, val] of pairs) flux_pngs[key] = val;
      const name = runName("day", dayResult.date || "run");
      const document = {
        kind: "day",
        project_name: store.get("ui.projectName") || null,
        request: sweepRequest,
        result: dayResult,
        flux_pngs,
        // §M.4: the analysis aperture, frozen at save time -- see
        // buildApertureSnapshotForSave and SavedRunDocument.aperture.
        aperture: buildApertureSnapshotForSave(),
      };
      return saveLibraryEntry("runs", name, document).then(() => attachRunToProject(name).then(() => name));
    })
    .then((name) => {
      dayRunSaving = false;
      dayRunSavedName = name;
      noteProjectRunAdded(name, "day");
      paintIfVisible();
    })
    .catch((err) => {
      dayRunSaving = false;
      dayRunError = (err && err.message) || "Could not save this run.";
      paintIfVisible();
    });
}

function discardDayRun() {
  if (!dayRunSavedName || dayRunSaving) return;
  const name = dayRunSavedName;
  dayRunSaving = true;
  paintIfVisible();
  deleteLibraryEntry("runs", name)
    .then(() => detachRunFromProject(name))
    .then(() => {
      dayRunSaving = false;
      dayRunSavedName = null;
      // The result on screen is still real, but it no longer has a saved
      // copy backing it -- a fresh timestep click should re-trace rather
      // than reach for a blob that is about to stop existing.
      reopenedDayFluxPngs = null;
      noteProjectRunRemoved(name);
      paintIfVisible();
    })
    .catch((err) => {
      dayRunSaving = false;
      dayRunError = (err && err.message) || "Could not discard the saved run.";
      paintIfVisible();
    });
}

function saveYearRun() {
  if (!yearResult || yearRunSaving || yearRunSavedName) return;
  yearRunSaving = true;
  yearRunError = null;
  paintIfVisible();

  const name = runName("year", yearResult.year || "run");
  const document = {
    kind: "year",
    project_name: store.get("ui.projectName") || null,
    request: yearSweepRequest,
    result: yearResult,
    flux_pngs: {},
  };
  saveLibraryEntry("runs", name, document)
    .then(() => attachRunToProject(name))
    .then(() => {
      yearRunSaving = false;
      yearRunSavedName = name;
      noteProjectRunAdded(name, "year");
      paintIfVisible();
    })
    .catch((err) => {
      yearRunSaving = false;
      yearRunError = (err && err.message) || "Could not save this run.";
      paintIfVisible();
    });
}

function discardYearRun() {
  if (!yearRunSavedName || yearRunSaving) return;
  const name = yearRunSavedName;
  yearRunSaving = true;
  paintIfVisible();
  deleteLibraryEntry("runs", name)
    .then(() => detachRunFromProject(name))
    .then(() => {
      yearRunSaving = false;
      yearRunSavedName = null;
      noteProjectRunRemoved(name);
      paintIfVisible();
    })
    .catch((err) => {
      yearRunSaving = false;
      yearRunError = (err && err.message) || "Could not discard the saved run.";
      paintIfVisible();
    });
}

// §R: "Save this run" -- persists the traced instant currently on screen as
// a SavedRunDocument(kind="instant"). Unlike a day run's own save (which
// fetches each kept step's PNG from the server job), the instant's one and
// only flux_png already sits base64-encoded right on the trace response --
// nothing to fetch. `scene` is dropped from the stored result (§R: "3D View
// regenerates geometry cheaply from the same request; Analysis's
// instruments never read it, and keeping it would roughly double the saved
// size for no benefit here"). Stays pointed at the LIVE ui.traceResult
// after saving (reopenedInstant is NOT set here), same as a day/year run's
// own save leaving its live job in place -- only an explicit reopen later
// switches to the frozen record.
function saveInstantRun() {
  if (activeSource !== "instant") return;
  const result = currentInstantResult();
  const request = currentInstantRequest();
  if (!result || !request || instantSaving || instantSavedName) return;
  instantSaving = true;
  instantRunError = null;
  paintIfVisible();

  const resultForSave = Object.assign({}, result);
  delete resultForSave.scene;
  const name = runName("instant", "run");
  const document = {
    kind: "instant",
    project_name: store.get("ui.projectName") || null,
    request,
    result: resultForSave,
    flux_pngs: result.flux_png ? { "0": result.flux_png } : {},
    aperture: buildApertureSnapshotForSave(0),
  };
  saveLibraryEntry("runs", name, document)
    .then(() => attachRunToProject(name).then(() => name))
    .then((savedName) => {
      instantSaving = false;
      instantSavedName = savedName;
      noteProjectRunAdded(savedName, "instant");
      paintIfVisible();
    })
    .catch((err) => {
      instantSaving = false;
      instantRunError = (err && err.message) || "Could not save this run.";
      paintIfVisible();
    });
}

function discardInstantRun() {
  if (!instantSavedName || instantSaving) return;
  const name = instantSavedName;
  instantSaving = true;
  paintIfVisible();
  deleteLibraryEntry("runs", name)
    .then(() => detachRunFromProject(name))
    .then(() => {
      instantSaving = false;
      instantSavedName = null;
      // A discarded reopen has nothing left to show frozen -- fall back to
      // whatever the live trace currently is, same as the day source falling
      // back to its (still-live) job after a discard.
      reopenedInstant = null;
      noteProjectRunRemoved(name);
      paintIfVisible();
    })
    .catch((err) => {
      instantSaving = false;
      instantRunError = (err && err.message) || "Could not discard the saved run.";
      paintIfVisible();
    });
}

// Loads a saved run in place of running a new job -- the whole point of
// docs/ui-spec.md 4's "reopens without re-running". `entry` is {name, kind}.
function openSavedRun(entry) {
  getLibraryEntry("runs", entry.name)
    .then((full) => {
      const document = full.document;
      if (document.kind === "day") {
        if (pollTimer) {
          clearTimeout(pollTimer);
          pollTimer = null;
        }
        resetRunState();
        dayResult = document.result;
        resultJobId = null;
        sweepRequest = document.request;
        sweepPhysicsKey = physicsKey(document.request);
        reopenedDayFluxPngs = document.flux_pngs || {};
        // §M.4: a frozen aperture annotation, if the run was saved with one
        // -- shown verbatim (paintAperturePanel), never recomputed. There is
        // no live job here to fetch a grid from, so this is the only
        // aperture state a reopened run can show.
        reopenedAperture = document.aperture || null;
        dayRunSavedName = entry.name;
        if (dayResult && dayResult.steps && dayResult.steps.length) {
          // Land on the aperture's own timestep when it has one, so the
          // frozen circle reappears on the map it was actually drawn on.
          selectedStepIndex =
            reopenedAperture &&
            Number.isInteger(reopenedAperture.step_index) &&
            reopenedAperture.step_index >= 0 &&
            reopenedAperture.step_index < dayResult.steps.length
              ? reopenedAperture.step_index
              : pickDefaultStepIndex(dayResult.steps);
          scheduleFluxFetch();
          scheduleHeliostatFootprintFetch();
        }
      } else if (document.kind === "year") {
        if (yearPollTimer) {
          clearTimeout(yearPollTimer);
          yearPollTimer = null;
        }
        resetYearRunState();
        activeSource = "year";
        yearResult = document.result;
        yearResultJobId = null;
        yearSweepRequest = document.request;
        yearSweepPhysicsKey = physicsKey(document.request);
        yearRunSavedName = entry.name;
        yearFastMode = !document.request || document.request.fast_mode !== false;
      } else {
        // §R: kind === "instant" -- reopens as a frozen record, exactly the
        // same "no live job, shown verbatim" shape a reopened day/year run
        // already has (reopenedAperture's own comment above). A fresh trace
        // still supersedes it the moment it lands (syncInstantSourceIfNeeded).
        closeFootprintOverlay();
        reopenedInstant = document;
        instantSavedName = entry.name;
        instantRunError = null;
        instantGeneration += 1;
        drillHeliostatId = null;
        clearHeliostatFootprint();
        apertureCenterUMm = null;
        apertureCenterVMm = null;
        apertureRadiusMm = null;
        apertureDrag = null;
        activeSource = "instant";
      }
      paintIfVisible();
    })
    .catch((err) => {
      manageError = (err && err.message) || "Could not load that saved run.";
      paintIfVisible();
    });
}

// The active project's own saved runs, kept in step with ui.projectName --
// called every paint() (cheap: it no-ops once synced) rather than through a
// store subscription, so it needs no teardown of its own.
function syncProjectRuns() {
  const name = store.get("ui.projectName");
  if (name === lastSyncedProjectName) return;
  lastSyncedProjectName = name;
  if (!name) {
    projectRunEntries = [];
    projectRunsLoading = false;
    projectRunsError = null;
    return;
  }
  projectRunsLoading = true;
  projectRunsError = null;
  getLibraryEntry("projects", name)
    .then((entry) => {
      if (store.get("ui.projectName") !== name) return; // superseded before this landed
      const names = Array.isArray(entry.document.runs) ? entry.document.runs : [];
      return Promise.all(
        names.map((n) =>
          getLibraryEntry("runs", n)
            .then((full) => ({ name: n, kind: full.document.kind, saved_at: full.saved_at }))
            .catch(() => null) // a run the project still lists but that was deleted elsewhere
        )
      ).then((entries) => {
        if (store.get("ui.projectName") !== name) return;
        projectRunEntries = entries.filter(Boolean);
        projectRunsLoading = false;
        paintIfVisible();
      });
    })
    .catch((err) => {
      if (store.get("ui.projectName") !== name) return;
      projectRunsLoading = false;
      projectRunsError = (err && err.message) || "Could not load this project's saved runs.";
      paintIfVisible();
    });
}

// -- "Manage saved runs" overlay: every run in the library, any project ------

function openManage() {
  manageOpen = true;
  loadManageEntries();
  paintIfVisible();
}

function closeManage() {
  manageOpen = false;
  paintIfVisible();
}

function loadManageEntries() {
  manageLoading = true;
  manageError = null;
  paintIfVisible();
  getLibrary("runs")
    .then((data) => {
      const entries = data.entries || [];
      return Promise.all(
        entries.map((e) =>
          getLibraryEntry("runs", e.name)
            .then((full) => ({
              name: e.name,
              saved_at: e.saved_at,
              size_bytes: e.size_bytes,
              kind: full.document.kind,
              project_name: full.document.project_name,
            }))
            .catch(() => ({
              name: e.name,
              saved_at: e.saved_at,
              size_bytes: e.size_bytes,
              kind: "?",
              project_name: null,
            }))
        )
      );
    })
    .then((entries) => {
      manageEntries = entries;
      manageLoading = false;
      paintIfVisible();
    })
    .catch((err) => {
      manageLoading = false;
      manageError = (err && err.message) || "Could not load saved runs.";
      paintIfVisible();
    });
}

function deleteManageEntry(name) {
  const entry = manageEntries.find((e) => e.name === name);
  deleteLibraryEntry("runs", name)
    .then(() => detachRunFromProjectNamed(entry && entry.project_name, name))
    .then(() => {
      if (dayRunSavedName === name) dayRunSavedName = null;
      if (yearRunSavedName === name) yearRunSavedName = null;
      lastSyncedProjectName = undefined; // force the project-runs chips to refresh too
      loadManageEntries();
    })
    .catch((err) => {
      manageError = (err && err.message) || "Could not delete that run.";
      paintIfVisible();
    });
}

// -- analysis aperture math (docs/ui-spec-v0.2.md §M.4) ----------------------
// Pure functions of (grid, center, radius, step): no store access, no
// fetch, no DOM. Mirrors heliostat.web.app._aperture_metrics bin for bin
// (kept in lockstep -- that Python twin is what tests/test_web.py's
// synthetic-disk test checks, since this repo runs pytest only and has no
// JS test runner to check this function directly). Post-processing ONLY:
// nothing below ever calls postTrace/postFieldTrace or writes to
// dayResult/sweepRequest/jobSnapshot -- the trace, intercept and collected
// totals a sweep already produced can never change because a circle moved.

// receiverKindFor/apertureReceiverIsFlat now live in ../aperture.js (v0.2
// followups item 3 -- shared with panels/run.js's own dock aperture).

// Compass rider (§M): flat window u = world x (east), v = world y (north) --
// FlatWindowReceiver.uv_extent (geometry/receiver.py) is plain (x, y), no
// rotation. Unrolled cylinder/frustum: u = radius * az with az = atan2(x,
// -y) (measured from -y/south, seam at +y/north -- CylinderReceiver's own
// module docstring and _continuous_azimuth) -- so the unrolled map's CENTER
// (u=0) is compass south and BOTH its edges (u = +/-half-circumference, the
// same physical seam) are compass north: moving right from center goes
// south -> east -> north; moving left goes south -> west -> north.
function compassCaptionFor(doc) {
  const kind = receiverKindFor(doc);
  if (kind === "flat") return "Compass: N top · S bottom · E right · W left";
  return "Compass (u-axis): S at center → E → N at the right edge · S at center → W → N at the left edge (seam)";
}

// apertureDefaultCenter/apertureDefaultRadiusMm now live in ../aperture.js.

function currentApertureCenterMm(grid, step) {
  if (apertureCenterUMm != null && apertureCenterVMm != null) {
    return { u: apertureCenterUMm, v: apertureCenterVMm };
  }
  return apertureDefaultCenter(grid, step);
}

function currentApertureRadiusMm(grid, step) {
  if (apertureRadiusMm != null) return apertureRadiusMm;
  return apertureDefaultRadiusMm(grid, step);
}

// clampToGridAxis/apertureMetrics/apertureCurve now live in ../aperture.js.

// `stepIndexForSave` is what the saved snapshot's own `step_index` reads
// back as (day: the real selectedStepIndex, for the aperture's "land on the
// saved circle's own timestep" reopen logic; instant: always 0, since §R's
// SavedRunDocument(kind="instant") carries exactly one timestep -- see
// saveInstantRun).
function buildApertureSnapshotForSave(stepIndexForSave) {
  // Resaving an already-reopened (frozen) run keeps its own annotation
  // rather than trying to recompute one from a grid that, for a reopened
  // run, does not exist (see openSavedRun/scheduleApertureGridFetch).
  if (activeSource !== "instant" && reopenedAperture) return reopenedAperture;
  const grid = currentAnyApertureGrid();
  if (!grid) return null;
  if (activeSource !== "instant" && selectedStepIndex == null) return null;
  const step = currentStepForAperture();
  if (!step) return null;
  const center = currentApertureCenterMm(grid, step);
  const radius = currentApertureRadiusMm(grid, step);
  const { powerW, avgFluxWM2 } = apertureMetrics(grid, center.u, center.v, radius);
  const collectedW = step.power_w;
  const dni = step.dni_w_m2;
  return {
    step_index: stepIndexForSave != null ? stepIndexForSave : selectedStepIndex,
    center_u_mm: center.u,
    center_v_mm: center.v,
    radius_mm: radius,
    power_w: powerW,
    frac_collected_pct: collectedW ? (100 * powerW) / collectedW : null,
    avg_flux_w_m2: avgFluxWM2,
    dni_w_m2: dni != null ? dni : null,
    avg_concentration: dni ? avgFluxWM2 / dni : null,
  };
}

// -- analysis aperture canvas: rendering + drag interaction ------------------
// A dedicated <canvas> rather than an overlay on the existing matplotlib
// <img> (fluxImg): that PNG carries axis labels/colorbar/title chrome this
// module has no exact pixel geometry for, so a draggable circle drawn on
// top of it could not be positioned reliably. This canvas is instead
// painted directly from the fetched grid, at a uniform mm-per-pixel scale
// in both axes (sizeApertureCanvas) so a physical-radius circle is a true
// circle on screen, not an ellipse.
// APERTURE_MAGMA_STOPS/magmaColor/sizeApertureCanvas/apertureDataToCanvas/
// apertureCanvasToData/paintApertureCanvas/paintApertureCurve/
// apertureCanvasEventPoint now live in ../aperture.js.

// Pointer-drag handling for the aperture canvas: click near the resize
// handle to change the radius, click anywhere else inside the circle to
// move it, click outside it to do nothing (no "click to place" -- the
// circle always has a defined default position).
function apertureHandlePointerDown(e) {
  const grid = currentAnyApertureGrid();
  if (!grid) return;
  const canvas = e.currentTarget;
  const step = currentStepForAperture();
  const center = currentApertureCenterMm(grid, step);
  const radius = currentApertureRadiusMm(grid, step);
  const [x, y] = apertureCanvasEventPoint(canvas, e);
  const [cx, cy] = apertureDataToCanvas(grid, canvas, center.u, center.v);
  const pxPerMm = canvas.width / (grid.u_max_mm - grid.u_min_mm);
  const rPx = radius * pxPerMm;
  const dHandle = Math.hypot(x - (cx + rPx), y - cy);
  const dCenter = Math.hypot(x - cx, y - cy);
  if (dHandle <= 10) {
    apertureDrag = { mode: "resize" };
  } else if (dCenter <= rPx + 10) {
    apertureDrag = { mode: "move" };
  } else {
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  e.preventDefault();
}

function apertureHandlePointerMove(e) {
  const grid = currentAnyApertureGrid();
  if (!apertureDrag || !grid) return;
  const canvas = e.currentTarget;
  const step = currentStepForAperture();
  const [x, y] = apertureCanvasEventPoint(canvas, e);
  const [uMm, vMm] = apertureCanvasToData(grid, canvas, x, y);
  if (apertureDrag.mode === "move") {
    apertureCenterUMm = clampToGridAxis(grid, uMm, "u");
    apertureCenterVMm = clampToGridAxis(grid, vMm, "v");
  } else {
    const center = currentApertureCenterMm(grid, step);
    const rMm = Math.hypot(uMm - center.u, vMm - center.v);
    apertureRadiusMm = apertureClampRadius(grid, rMm);
  }
  paintIfVisible();
}

// -- M.6: typed center u/v (mm) and radius (mm), alongside the drag above --
// Each handler clamps exactly like the drag handlers (clampToGridAxis /
// apertureClampRadius, the same functions the pointer-move code above
// calls) so a typed value can never land the circle somewhere a drag
// couldn't, then repaints -- the same live update a drag triggers.
//
// currentApertureCenterMm only trusts apertureCenterUMm/VMm as a pair --
// if either is still null it falls back to the DEFAULT center entirely
// (see that function, and currentApertureRadiusMm's own single-value
// equivalent has no such pairing issue). Typing just Center u first (the
// natural first edit) would otherwise leave v null and silently snap back
// to the default center on every keystroke -- both handlers seed the
// OTHER axis from the current effective center first so a lone edit holds.
function apertureHandleCenterUInput(e) {
  const grid = currentAnyApertureGrid();
  if (!grid) return;
  const v = parseFloat(e.target.value);
  if (!Number.isFinite(v)) return;
  const center = currentApertureCenterMm(grid, currentStepForAperture());
  apertureCenterUMm = clampToGridAxis(grid, v, "u");
  apertureCenterVMm = center.v;
  paintIfVisible();
}

function apertureHandleCenterVInput(e) {
  const grid = currentAnyApertureGrid();
  if (!grid) return;
  const v = parseFloat(e.target.value);
  if (!Number.isFinite(v)) return;
  const center = currentApertureCenterMm(grid, currentStepForAperture());
  apertureCenterVMm = clampToGridAxis(grid, v, "v");
  apertureCenterUMm = center.u;
  paintIfVisible();
}

function apertureHandleRadiusInput(e) {
  const grid = currentAnyApertureGrid();
  if (!grid) return;
  const v = parseFloat(e.target.value);
  if (!Number.isFinite(v) || v <= 0) return;
  apertureRadiusMm = apertureClampRadius(grid, v);
  paintIfVisible();
}

function apertureHandlePointerUp(e) {
  if (!apertureDrag) return;
  apertureDrag = null;
  const canvas = e.currentTarget;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (err) {
    // Capture may already be gone (pointer left the window, etc.) -- the
    // drag is over either way.
  }
}

// -- sweep drill-down rendering (§M.2) ---------------------------------------

function paintDrillMiniPlan(svg) {
  const heliostats = currentFieldHeliostats();
  svg.innerHTML = "";
  const bg = document.createElementNS(svg.namespaceURI, "rect");
  bg.setAttribute("width", "150");
  bg.setAttribute("height", "120");
  bg.setAttribute("fill", "#ffffff");
  svg.appendChild(bg);
  const cx = 75;
  const cy = 60;
  if (heliostats.length) {
    let maxR = 1;
    for (const h of heliostats) maxR = Math.max(maxR, Math.hypot(h.x_mm, h.y_mm));
    const scale = 52 / maxR;
    for (const h of heliostats) {
      const px = cx + h.x_mm * scale;
      // World y (north) is up on a plan view; SVG y grows downward -- flip.
      const py = cy - h.y_mm * scale;
      const sel = h.id === drillHeliostatId;
      const dot = document.createElementNS(svg.namespaceURI, "circle");
      dot.setAttribute("cx", px.toFixed(1));
      dot.setAttribute("cy", py.toFixed(1));
      dot.setAttribute("r", sel ? "3.6" : "2.2");
      dot.setAttribute("fill", sel ? "none" : "#cfe0ef");
      dot.setAttribute("stroke", sel ? "#0b5fd0" : "#33455c");
      dot.setAttribute("stroke-width", sel ? "1.8" : "0.6");
      dot.dataset.hid = String(h.id);
      dot.style.cursor = "pointer";
      svg.appendChild(dot);
    }
  }
  const tower = document.createElementNS(svg.namespaceURI, "circle");
  tower.setAttribute("cx", String(cx));
  tower.setAttribute("cy", String(cy));
  tower.setAttribute("r", "4");
  tower.setAttribute("fill", "#7b8794");
  svg.appendChild(tower);
}

// -- repaint helper: async callbacks (a poll tick, a flux fetch landing) hit
// this instead of a bare paint() call, so a run that keeps going after a
// tab switch never writes into a hidden section. --------------------------
function paintIfVisible() {
  if (built && lastContainer && !lastContainer.hidden) paint();
}

// -- build (once) -------------------------------------------------------------

function build(container) {
  container.innerHTML = "";
  container.className = "tabpage";

  // -- subject strip --------------------------------------------------------
  const subject = document.createElement("div");
  subject.className = "an-subject";
  subject.textContent = "Analyzing the current project: ";
  const subjOptics = document.createElement("strong");
  const subjSep1 = document.createTextNode(" · ");
  const subjField = document.createElement("strong");
  const subjSep2 = document.createTextNode(" · ");
  const subjDesign = document.createElement("strong");
  const subjLink = document.createElement("a");
  subjLink.href = "#";
  subjLink.textContent = "change in Design →";
  subjLink.addEventListener("click", (e) => {
    e.preventDefault();
    store.set("ui.tab", "design");
  });
  subject.appendChild(subjOptics);
  subject.appendChild(subjSep1);
  subject.appendChild(subjField);
  subject.appendChild(subjSep2);
  subject.appendChild(subjDesign);
  subject.appendChild(subjLink);

  // -- saved runs for this project (docs/ui-spec.md 4) ----------------------
  const savedRunsBar = document.createElement("div");
  savedRunsBar.className = "an-savedrunsbar";
  const savedRunsLabel = document.createElement("span");
  savedRunsLabel.className = "an-savedrunslabel";
  const savedRunsList = document.createElement("div");
  savedRunsList.className = "an-savedrunslist";
  const manageLink = document.createElement("a");
  manageLink.href = "#";
  manageLink.textContent = "Manage saved runs…";
  manageLink.addEventListener("click", (e) => {
    e.preventDefault();
    openManage();
  });
  savedRunsBar.appendChild(savedRunsLabel);
  savedRunsBar.appendChild(savedRunsList);
  savedRunsBar.appendChild(manageLink);

  // -- main content: left (sweep + energy + year), right (table + map) -----
  // v0.2 followups M.6: `content` used to be the flex ROW itself (left +
  // right side by side); it now stacks that row above a full-width strip
  // (an-contentrow / an-contentbottom) so the analysis aperture below can
  // lay its map and encircled-power curve out side by side and its readout
  // out as one horizontal row -- neither fits in the 360px right column
  // (.an-right) the rest of this tab's cards live in.
  const content = document.createElement("div");
  content.className = "tabcontent an-contentcol";

  const contentRow = document.createElement("div");
  contentRow.className = "an-contentrow";

  const left = document.createElement("div");
  left.className = "an-left";
  const right = document.createElement("div");
  right.className = "an-right";

  // -- day sweep panel --------------------------------------------------
  const sweepPanel = document.createElement("div");
  sweepPanel.className = "panel";
  const sweepH2 = document.createElement("h2");
  sweepH2.textContent = "Day sweep";
  sweepPanel.appendChild(sweepH2);

  const controlRow = document.createElement("div");
  controlRow.className = "an-controlrow";

  const dateField = document.createElement("div");
  dateField.className = "an-field";
  const dateLabel = document.createElement("label");
  dateLabel.textContent = "Date";
  const dateInput = document.createElement("input");
  dateInput.type = "date";
  dateInput.className = "val";
  dateField.title = "The day this sweep steps through, sunrise to sunset.";
  dateInput.addEventListener("input", () => {
    formDate = dateInput.value;
  });
  dateField.appendChild(dateLabel);
  dateField.appendChild(dateInput);

  const stepField = document.createElement("div");
  stepField.className = "an-field";
  const stepLabel = document.createElement("label");
  stepLabel.textContent = "Timestep (h)";
  const stepInput = document.createElement("input");
  stepInput.type = "number";
  stepInput.className = "val";
  stepInput.min = "0.1";
  stepInput.max = "6";
  stepInput.step = "0.1";
  stepField.title = "Maximum spacing between sampled timesteps -- sunrise and sunset are always sampled, in between the sun divides evenly into steps no larger than this.";
  stepInput.addEventListener("input", () => {
    const v = parseFloat(stepInput.value);
    if (Number.isFinite(v) && v > 0) formHourStep = v;
    // The rough duration estimate (both the day sweep's own line and the
    // year estimate's, which shares this same control) reads formHourStep
    // live -- without a repaint here it would only catch up the next time
    // something else happens to touch paint(), same bug as elevInput below.
    paintIfVisible();
  });
  stepField.appendChild(stepLabel);
  stepField.appendChild(stepInput);

  const elevField = document.createElement("div");
  elevField.className = "an-field";
  const elevLabel = document.createElement("label");
  elevLabel.textContent = "Min elevation (°)";
  const elevInput = document.createElement("input");
  elevInput.type = "number";
  elevInput.className = "val";
  elevInput.min = "0";
  elevInput.max = "45";
  elevInput.step = "0.5";
  elevInput.title =
    "Skip timesteps below this sun elevation -- they cost the same trace time as a noon one but collect almost no power.";
  elevInput.addEventListener("input", () => {
    const v = parseFloat(elevInput.value);
    if (Number.isFinite(v) && v >= 0) formMinElevationDeg = v;
    // See stepInput's own listener above -- same reason.
    paintIfVisible();
  });
  elevField.appendChild(elevLabel);
  elevField.appendChild(elevInput);

  const fidelitySeg = document.createElement("div");
  fidelitySeg.className = "seg";
  fidelitySeg.style.marginBottom = "0";
  const fidelityBtns = {};
  // One fidelity for the whole app: this and the workspace run bar are the
  // same setting seen from two screens.
  for (const [key, label] of FIDELITY) {
    fidelityBtns[key] = segButton(
      fidelitySeg,
      label,
      key === store.get("ui.fidelity"),
      () => {
        store.set("ui.fidelity", key);
      },
      FIDELITY_TOOLTIPS[key]
    );
  }

  const startBtn = document.createElement("div");
  startBtn.className = "btn primary";
  startBtn.textContent = "Trace day sweep";
  startBtn.addEventListener("click", () => {
    if (startBtn.classList.contains("disabled-link")) return;
    startSweep();
  });

  const cancelBtn = document.createElement("div");
  cancelBtn.className = "btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", () => cancelSweep());

  controlRow.appendChild(dateField);
  controlRow.appendChild(stepField);
  controlRow.appendChild(elevField);
  controlRow.appendChild(fidelitySeg);
  controlRow.appendChild(startBtn);
  controlRow.appendChild(cancelBtn);
  sweepPanel.appendChild(controlRow);

  // Pre-start rough duration estimate (docs/ui-spec.md 4) -- see
  // durationWarningText. Sits right under the controls, above the progress
  // bar, so it is the thing a user sees right before pressing Start.
  const durationHint = document.createElement("div");
  durationHint.className = "fieldwarn";
  durationHint.hidden = true;
  sweepPanel.appendChild(durationHint);

  const progressRow = document.createElement("div");
  progressRow.className = "an-progressrow";
  progressRow.hidden = true;
  const progressBar = document.createElement("div");
  progressBar.className = "an-progressbar";
  const progressFill = document.createElement("div");
  progressFill.className = "an-progressfill";
  progressBar.appendChild(progressFill);
  const progressText = document.createElement("span");
  progressText.className = "an-progresstext";
  progressRow.appendChild(progressBar);
  progressRow.appendChild(progressText);
  sweepPanel.appendChild(progressRow);

  const statusLine = document.createElement("div");
  statusLine.className = "hint";
  statusLine.hidden = true;
  sweepPanel.appendChild(statusLine);

  const staleChip = document.createElement("div");
  staleChip.className = "fieldwarn";
  staleChip.hidden = true;
  staleChip.textContent =
    "The heliostat, optics or field changed since this run — its numbers describe the old setup. Re-run the sweep to measure the current one.";
  sweepPanel.appendChild(staleChip);

  const sweepErr = document.createElement("div");
  sweepErr.className = "fielderr";
  sweepErr.hidden = true;
  sweepPanel.appendChild(sweepErr);

  const sweepHint = document.createElement("div");
  sweepHint.className = "hint";
  sweepHint.textContent =
    "A finished sweep lives in memory until 8 more replace it. Save it to keep it with the project instead.";
  sweepPanel.appendChild(sweepHint);

  const dayReopenBanner = document.createElement("div");
  dayReopenBanner.className = "an-reopenbanner";
  dayReopenBanner.hidden = true;
  sweepPanel.appendChild(dayReopenBanner);

  const dayRunRow = document.createElement("div");
  dayRunRow.className = "an-runrow";
  const dayRunSaveBtn = document.createElement("div");
  dayRunSaveBtn.className = "btn small";
  dayRunSaveBtn.textContent = "Save this run";
  dayRunSaveBtn.addEventListener("click", () => saveDayRun());
  const dayRunDiscardBtn = document.createElement("div");
  dayRunDiscardBtn.className = "btn small";
  dayRunDiscardBtn.textContent = "Discard this run";
  dayRunDiscardBtn.hidden = true;
  dayRunDiscardBtn.addEventListener("click", () => discardDayRun());
  const dayRunStatus = document.createElement("span");
  dayRunStatus.className = "an-runstatus";
  dayRunRow.appendChild(dayRunSaveBtn);
  dayRunRow.appendChild(dayRunDiscardBtn);
  dayRunRow.appendChild(dayRunStatus);
  sweepPanel.appendChild(dayRunRow);
  const dayRunErrEl = document.createElement("div");
  dayRunErrEl.className = "fielderr";
  dayRunErrEl.hidden = true;
  sweepPanel.appendChild(dayRunErrEl);

  left.appendChild(sweepPanel);

  // -- energy through the day panel --------------------------------------
  const energyPanel = document.createElement("div");
  energyPanel.className = "panel";
  energyPanel.style.flex = "1 1 auto";
  energyPanel.style.display = "flex";
  energyPanel.style.flexDirection = "column";
  // NOT min-height: 0. That override let this panel's own box be squeezed
  // by its flex-column siblings (Day sweep above, Year estimate below)
  // below its .frame child's fixed 260px min-height -- the frame doesn't
  // shrink with it, so with overflow:visible (the default) it spilled out
  // the bottom of this panel's box and over the Year estimate panel below,
  // hiding its Run button before any sweep has run (nothing shrinks the
  // panel below its content once yearFrame is showing a real plot, so the
  // bug was only visible in the pre-trace state where the panels are short
  // enough for space to actually run out). Leaving min-height at its
  // flex default (auto = content minimum, since overflow is visible) means
  // this panel can no longer be squashed smaller than its frame; .tabcontent's
  // own overflow:auto scrolls instead, which is honest rather than silently
  // hiding another panel.
  const energyH2 = document.createElement("h2");
  energyH2.textContent = "Energy through the day";
  energyPanel.appendChild(energyH2);

  const energyFrame = document.createElement("div");
  energyFrame.className = "frame";
  const energyImg = document.createElement("img");
  energyImg.alt = "Collected power through the day";
  energyImg.hidden = true;
  const energyPlaceholder = document.createElement("p");
  energyPlaceholder.className = "placeholder";
  energyPlaceholder.textContent = "Run a day sweep to see energy collected through the day.";
  energyFrame.appendChild(energyImg);
  energyFrame.appendChild(energyPlaceholder);
  energyPanel.appendChild(energyFrame);

  const energyFoot = document.createElement("div");
  energyFoot.className = "an-energyfoot";
  const energyTotal = document.createElement("span");
  energyTotal.className = "an-total";
  const energyCsv = document.createElement("a");
  energyCsv.href = "#";
  energyCsv.textContent = "Export day CSV";
  energyCsv.hidden = true;
  energyFoot.appendChild(energyTotal);
  energyFoot.appendChild(energyCsv);
  energyPanel.appendChild(energyFoot);

  left.appendChild(energyPanel);

  // -- year estimate (docs/ui-spec.md 4) -------------------------------------
  const yearPanel = document.createElement("div");
  yearPanel.className = "panel an-yearpanel";
  const yearRow = document.createElement("div");
  yearRow.className = "an-yearrow";
  const yearH2 = document.createElement("h2");
  yearH2.textContent = "Year estimate";
  const yearDesc = document.createElement("span");
  yearDesc.className = "an-yeardesc";
  yearDesc.textContent = "Sample days spaced in solar declination, DNI-weighted across the year";
  yearRow.appendChild(yearH2);
  yearRow.appendChild(yearDesc);
  yearPanel.appendChild(yearRow);

  const yearControlRow = document.createElement("div");
  yearControlRow.className = "an-controlrow";
  const yearFastSeg = document.createElement("div");
  yearFastSeg.className = "seg";
  yearFastSeg.style.marginBottom = "0";
  const yearFastBtn = segButton(
    yearFastSeg,
    "Fast (7 traced)",
    yearFastMode,
    () => {
      yearFastMode = true;
      paintIfVisible();
    },
    "Traces 7 representative days and fills the rest of the year by symmetry -- faster, at the cost of some accuracy on days far from those samples."
  );
  const yearAllBtn = segButton(
    yearFastSeg,
    "All 12 traced",
    !yearFastMode,
    () => {
      yearFastMode = false;
      paintIfVisible();
    },
    "Traces all 12 sample days directly -- slower, no symmetry fill-in."
  );
  const yearStartBtn = document.createElement("div");
  yearStartBtn.className = "btn primary";
  yearStartBtn.textContent = "Trace year estimate";
  yearStartBtn.addEventListener("click", () => {
    if (yearStartBtn.classList.contains("disabled-link")) return;
    startYear();
  });
  const yearCancelBtn = document.createElement("div");
  yearCancelBtn.className = "btn";
  yearCancelBtn.textContent = "Cancel";
  yearCancelBtn.addEventListener("click", () => cancelYear());
  yearControlRow.appendChild(yearFastSeg);
  yearControlRow.appendChild(yearStartBtn);
  yearControlRow.appendChild(yearCancelBtn);
  yearPanel.appendChild(yearControlRow);

  // See the day sweep's own durationHint -- a year estimate is the run this
  // warning matters most for (the full default field is ~1 hour).
  const yearDurationHint = document.createElement("div");
  yearDurationHint.className = "fieldwarn";
  yearDurationHint.hidden = true;
  yearPanel.appendChild(yearDurationHint);

  const yearProgressRow = document.createElement("div");
  yearProgressRow.className = "an-progressrow";
  yearProgressRow.hidden = true;
  const yearProgressBar = document.createElement("div");
  yearProgressBar.className = "an-progressbar";
  const yearProgressFill = document.createElement("div");
  yearProgressFill.className = "an-progressfill";
  yearProgressBar.appendChild(yearProgressFill);
  const yearProgressText = document.createElement("span");
  yearProgressText.className = "an-progresstext";
  yearProgressRow.appendChild(yearProgressBar);
  yearProgressRow.appendChild(yearProgressText);
  yearPanel.appendChild(yearProgressRow);

  const yearStatusLine = document.createElement("div");
  yearStatusLine.className = "hint";
  yearStatusLine.hidden = true;
  yearPanel.appendChild(yearStatusLine);

  const yearStaleChip = document.createElement("div");
  yearStaleChip.className = "fieldwarn";
  yearStaleChip.hidden = true;
  yearStaleChip.textContent =
    "The heliostat, optics or field changed since this run — its numbers describe the old setup. Re-run the estimate to measure the current one.";
  yearPanel.appendChild(yearStaleChip);

  const yearErr = document.createElement("div");
  yearErr.className = "fielderr";
  yearErr.hidden = true;
  yearPanel.appendChild(yearErr);

  // Spec §M.7: which sun this estimate assumed is stated per-run, next to
  // the annual total itself (paintYearResult's own dniLabel) -- this line
  // stays as the general, always-true explanation of what "clear-sky
  // model" means when that IS the site DNI setting in effect (the Sun
  // panel decides which one that is).
  const yearDniNote = document.createElement("div");
  yearDniNote.className = "hint";
  yearDniNote.textContent =
    "The clear-sky model (no clouds) is a cloud-free upper bound, not a weather-corrected forecast — set the site DNI in the Sun panel.";
  yearPanel.appendChild(yearDniNote);

  const yearTotal = document.createElement("div");
  yearTotal.className = "an-yeartotal";
  yearTotal.hidden = true;
  yearPanel.appendChild(yearTotal);

  const yearFrame = document.createElement("div");
  yearFrame.className = "frame";
  const yearImg = document.createElement("img");
  yearImg.alt = "Day energy across the year";
  yearImg.hidden = true;
  const yearPlaceholder = document.createElement("p");
  yearPlaceholder.className = "placeholder";
  yearPlaceholder.textContent = "Run a year estimate to see collection across the year.";
  yearFrame.appendChild(yearImg);
  yearFrame.appendChild(yearPlaceholder);
  // Hidden until there is a curve to show: an empty plot frame is 250 px of
  // nothing, and on a short window it pushes the Run button off the bottom
  // of the tab -- the one control the panel exists for.
  yearFrame.hidden = true;
  yearPanel.appendChild(yearFrame);

  // -- M.5: year -> day drill-in ---------------------------------------
  // Every sample day the year plot above shows can be opened as a full day
  // sweep -- the year machinery already traces these days with the day-
  // sweep code (this just reuses startSweep()'s own launch path with the
  // chosen date, same as clicking "Trace day sweep" by hand). See
  // openYearDay below.
  const yearDaysWrap = document.createElement("div");
  yearDaysWrap.className = "an-yeardayswrap";
  yearDaysWrap.hidden = true;
  const yearDaysHead = document.createElement("div");
  yearDaysHead.className = "an-drillhead";
  yearDaysHead.textContent = "Sampled days — click one to open it as a day sweep";
  yearDaysWrap.appendChild(yearDaysHead);
  const yearDaysTable = document.createElement("table");
  yearDaysTable.className = "an-table an-yeardaystable";
  yearDaysTable.innerHTML =
    "<thead><tr><th>Date</th><th>Declination</th><th>Energy</th><th>Peak</th></tr></thead><tbody></tbody>";
  const yearDaysTbody = yearDaysTable.querySelector("tbody");
  yearDaysTbody.addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-idx]");
    if (!row) return;
    openYearDay(Number(row.dataset.idx));
  });
  yearDaysWrap.appendChild(yearDaysTable);
  yearPanel.appendChild(yearDaysWrap);

  const yearReopenBanner = document.createElement("div");
  yearReopenBanner.className = "an-reopenbanner";
  yearReopenBanner.hidden = true;
  yearPanel.appendChild(yearReopenBanner);

  const yearRunRow = document.createElement("div");
  yearRunRow.className = "an-runrow";
  const yearRunSaveBtn = document.createElement("div");
  yearRunSaveBtn.className = "btn small";
  yearRunSaveBtn.textContent = "Save this run";
  yearRunSaveBtn.hidden = true;
  yearRunSaveBtn.addEventListener("click", () => saveYearRun());
  const yearRunDiscardBtn = document.createElement("div");
  yearRunDiscardBtn.className = "btn small";
  yearRunDiscardBtn.textContent = "Discard this run";
  yearRunDiscardBtn.hidden = true;
  yearRunDiscardBtn.addEventListener("click", () => discardYearRun());
  const yearRunStatus = document.createElement("span");
  yearRunStatus.className = "an-runstatus";
  yearRunRow.appendChild(yearRunSaveBtn);
  yearRunRow.appendChild(yearRunDiscardBtn);
  yearRunRow.appendChild(yearRunStatus);
  yearPanel.appendChild(yearRunRow);
  const yearRunErrEl = document.createElement("div");
  yearRunErrEl.className = "fielderr";
  yearRunErrEl.hidden = true;
  yearPanel.appendChild(yearRunErrEl);

  left.appendChild(yearPanel);

  // -- §R: traced instant, a third result source (docs/ui-spec-v0.2.md §R,
  // mockup M22) -- peer to Day sweep and Year estimate above, never a
  // Timesteps-table row. Shows whatever the 3D View trace bar's shared live
  // result currently is (or a reopened saved one), with a link to 3D View's
  // Run control when nothing has been traced yet -- clicking the summary
  // when a result exists selects it as the right column's source, exactly
  // like the "Viewing" selector below.
  const instantPanel = document.createElement("div");
  instantPanel.className = "panel an-instantpanel";
  const instantH2 = document.createElement("h2");
  instantH2.textContent = "Traced instant";
  instantPanel.appendChild(instantH2);

  const instantSummary = document.createElement("a");
  instantSummary.href = "#";
  instantSummary.className = "an-instantsummary";
  instantSummary.addEventListener("click", (e) => {
    e.preventDefault();
    if (instantSummary.classList.contains("disabled-link")) {
      store.set("ui.tab", "3dview");
      return;
    }
    selectSource("instant");
  });
  instantPanel.appendChild(instantSummary);

  const instantRunRow = document.createElement("div");
  instantRunRow.className = "an-runrow";
  const instantRunSaveBtn = document.createElement("div");
  instantRunSaveBtn.className = "btn small";
  instantRunSaveBtn.textContent = "Save this run";
  instantRunSaveBtn.addEventListener("click", () => saveInstantRun());
  const instantRunDiscardBtn = document.createElement("div");
  instantRunDiscardBtn.className = "btn small";
  instantRunDiscardBtn.textContent = "Discard this run";
  instantRunDiscardBtn.hidden = true;
  instantRunDiscardBtn.addEventListener("click", () => discardInstantRun());
  const instantRunStatus = document.createElement("span");
  instantRunStatus.className = "an-runstatus";
  instantRunRow.appendChild(instantRunSaveBtn);
  instantRunRow.appendChild(instantRunDiscardBtn);
  instantRunRow.appendChild(instantRunStatus);
  instantPanel.appendChild(instantRunRow);
  const instantRunErrEl = document.createElement("div");
  instantRunErrEl.className = "fielderr";
  instantRunErrEl.hidden = true;
  instantPanel.appendChild(instantRunErrEl);

  left.appendChild(instantPanel);

  // -- §R: "Viewing" -- which of the three sources drives the right column.
  const sourceRow = document.createElement("div");
  sourceRow.className = "an-sourcerow";
  const sourceLabel = document.createElement("span");
  sourceLabel.className = "an-sourcelabel";
  sourceLabel.textContent = "Viewing:";
  sourceRow.appendChild(sourceLabel);
  const sourceSeg = document.createElement("div");
  sourceSeg.className = "seg an-sourceseg";
  const sourceDayBtn = segButton(sourceSeg, "Day sweep", true, () => selectSource("day"));
  const sourceYearBtn = segButton(sourceSeg, "Year estimate", false, () => selectSource("year"));
  const sourceInstantBtn = segButton(sourceSeg, "Traced instant", false, () => selectSource("instant"));
  sourceRow.appendChild(sourceSeg);
  right.appendChild(sourceRow);

  // -- timesteps table -----------------------------------------------------
  const tsPanel = document.createElement("div");
  tsPanel.className = "panel an-tspanel";
  const tsH2 = document.createElement("h2");
  tsH2.textContent = "Timesteps";
  tsPanel.appendChild(tsH2);
  const tsWrap = document.createElement("div");
  tsWrap.className = "an-tswrap";
  const table = document.createElement("table");
  table.className = "an-table";
  table.innerHTML =
    "<thead><tr><th>Solar</th><th>El (°)</th><th>Power</th><th>Peak flux</th></tr></thead><tbody></tbody>";
  const tbody = table.querySelector("tbody");
  tbody.addEventListener("click", (e) => {
    const row = e.target.closest("tr[data-idx]");
    if (!row) return;
    selectStep(Number(row.dataset.idx));
  });
  tsWrap.appendChild(table);
  tsPanel.appendChild(tsWrap);
  const tsEmpty = document.createElement("p");
  tsEmpty.className = "hint";
  tsEmpty.textContent = "No timesteps yet — start a day sweep.";
  tsPanel.appendChild(tsEmpty);
  right.appendChild(tsPanel);

  // -- irradiance map for the selected timestep -----------------------------
  const fluxPanel = document.createElement("div");
  fluxPanel.className = "panel an-fluxpanel";
  const fluxHead = document.createElement("div");
  fluxHead.className = "an-fluxhead";
  const fluxH2 = document.createElement("h2");
  fluxH2.textContent = "Irradiance map";
  fluxHead.appendChild(fluxH2);
  // Spec §C / mockup M9: Receiver | Secondary selector -- only meaningfully
  // switchable when the currently-shown map came from a live single/field
  // trace that carried a "secondary" block (api.js's buildTraceRequest
  // always asks for one via include_secondary_flux). A step served from
  // the sweep's own stored PNG (fluxSrcUrl, the common case -- see
  // scheduleFluxFetch) has no raw grid behind it at all, so Secondary stays
  // disabled with a tooltip for that step instead of silently doing
  // nothing -- paintFluxPanel below decides which message applies.
  const fluxSurfaceSeg = document.createElement("div");
  fluxSurfaceSeg.className = "seg an-surfaceseg";
  const fluxSurfaceReceiverBtn = segButton(fluxSurfaceSeg, "Receiver", true, () =>
    store.set("ui.fluxSurface", "receiver")
  );
  const fluxSurfaceSecondaryBtn = segButton(fluxSurfaceSeg, "Secondary", false, () => {
    if (fluxSurfaceSecondaryBtn.classList.contains("disabled")) return;
    store.set("ui.fluxSurface", "secondary");
  });
  // v0.2 followups item 1, mockup M15: Field -- only ever available when
  // THIS step's map came from a live field-trace response (fluxHeliostats),
  // which a stored sweep step's own PNG/CSV/grid blobs never carry (see
  // fluxHeliostats's own comment above) -- disabled with an honest tooltip
  // otherwise, same "available" pattern as Secondary.
  const fluxSurfaceFieldBtn = segButton(fluxSurfaceSeg, "Field", false, () => {
    if (fluxSurfaceFieldBtn.classList.contains("disabled")) return;
    store.set("ui.fluxSurface", "field");
  });
  fluxHead.appendChild(fluxSurfaceSeg);
  // docs/ui-spec-v0.2.md §C2, mockup M23: the secondary map used to offer a
  // Plan/Unrolled (technical) toggle -- a display transform between the
  // radial (u, v) bins' native unrolled layout and a polar plan projection
  // of them. Now that secondary_uv is itself a Cartesian (x_local, y_local)
  // plan-projection pair (see heliostat.geometry.secondary's module notes),
  // the grid already IS the plan view -- there is no second, genuinely
  // different "unrolled" layout left to synthesize, so the toggle is
  // removed rather than kept as a no-op. paintSecondaryFluxCanvasPlan below
  // is now the map's only presentation.
  // docs/ui-spec-v0.2.md §N, mockup M18c: this inline map stays here for
  // fast scrubbing (decided, not moved) -- this link is the one-click-deeper
  // path to the richer 3D View viewers (drape, aperture, FEA export) mockup
  // M18c describes. The timestep itself isn't carried over (3D View shows
  // the live design's current trace, not a re-play of one sweep step) --
  // switching tabs is the same "closest existing idiom" every other
  // tab-jump link in the app already uses (js/inspector.js's "View shape",
  // js/panels/heliostat.js's "Edit shape…").
  const openIn3DLink = document.createElement("a");
  openIn3DLink.href = "#";
  openIn3DLink.className = "an-open3d";
  openIn3DLink.textContent = "Open in 3D View →";
  openIn3DLink.title = "3D View has the drape, per-heliostat inspector, and FEA export for the live design's own trace.";
  openIn3DLink.addEventListener("click", (e) => {
    e.preventDefault();
    store.set("ui.tab", "3dview");
  });
  fluxHead.appendChild(openIn3DLink);
  fluxPanel.appendChild(fluxHead);

  const fluxMapBody = document.createElement("div");
  fluxMapBody.className = "an-mapbody";
  const fluxFrame = document.createElement("div");
  fluxFrame.className = "frame";
  const fluxImg = document.createElement("img");
  fluxImg.alt = "Irradiance map for the selected timestep";
  fluxImg.hidden = true;
  // No server-rendered PNG exists for the secondary map (only the opt-in
  // raw flux_grid, app.py's _secondary_payload) -- painted client-side onto
  // this canvas, same magma ramp/orientation as js/scene3d.js's receiver
  // drape texture (see paintSecondaryFluxCanvas below).
  const fluxSecondaryCanvas = document.createElement("canvas");
  fluxSecondaryCanvas.className = "an-secondarycanvas";
  fluxSecondaryCanvas.hidden = true;
  // Field map (item 1, mockup M15): client-rendered, one dot per heliostat
  // at its own plan-view position, colored by power_w -- see
  // paintFieldMapCanvas below. Same "own canvas, no server PNG" reasoning as
  // the secondary canvas above.
  const fluxFieldCanvas = document.createElement("canvas");
  fluxFieldCanvas.className = "an-secondarycanvas an-fieldcanvas";
  fluxFieldCanvas.hidden = true;
  const fluxPlaceholder = document.createElement("p");
  fluxPlaceholder.className = "placeholder";
  fluxPlaceholder.textContent = "Click a timestep to render its irradiance map.";
  fluxFrame.appendChild(fluxImg);
  fluxFrame.appendChild(fluxSecondaryCanvas);
  fluxFrame.appendChild(fluxFieldCanvas);
  fluxFrame.appendChild(fluxPlaceholder);
  fluxMapBody.appendChild(fluxFrame);

  // Absorbed-heat readout (spec §C) -- shown beside the secondary map only,
  // same rmetric/rformula idiom as the aperture readout below and mockup
  // M9's own layout.
  const fluxSecondaryReadout = document.createElement("div");
  fluxSecondaryReadout.className = "readout an-secondaryreadout";
  fluxSecondaryReadout.hidden = true;
  const fluxSecReadoutH3 = document.createElement("h3");
  fluxSecReadoutH3.textContent = "Secondary readout";
  fluxSecondaryReadout.appendChild(fluxSecReadoutH3);
  function secRow(label) {
    const row = document.createElement("div");
    row.className = "rmetric";
    const lbl = document.createElement("div");
    lbl.className = "rlbl";
    lbl.textContent = label;
    const num = document.createElement("div");
    num.className = "rnum";
    row.appendChild(lbl);
    row.appendChild(num);
    fluxSecondaryReadout.appendChild(row);
    return { lbl, num };
  }
  const secIncidentRow = secRow("incident");
  const secAbsorbedRow = secRow("absorbed");
  const secPeakAbsorbedRow = secRow("peak absorbed");
  const fluxSecFormula = document.createElement("div");
  fluxSecFormula.className = "rformula";
  fluxSecFormula.textContent = "absorbed = (1 − R) × incident";
  fluxSecondaryReadout.appendChild(fluxSecFormula);
  // docs/secondary-irradiance-plan.md: "UI must say coarse in cone modes,
  // exact in Monte Carlo wherever the secondary map shows" -- visible text,
  // not a tooltip.
  const fluxSecFidelity = document.createElement("div");
  fluxSecFidelity.className = "rfidelity";
  fluxSecondaryReadout.appendChild(fluxSecFidelity);
  fluxMapBody.appendChild(fluxSecondaryReadout);

  // Field readout (item 1, mockup M15): heliostat count + total power, and
  // the legend M15 draws as a floating chip -- folded into this app's own
  // readout/rmetric idiom instead, same container class main.js's overlay
  // uses for its own field readout.
  const fluxFieldReadout = document.createElement("div");
  fluxFieldReadout.className = "readout an-secondaryreadout";
  fluxFieldReadout.hidden = true;
  const fluxFieldReadoutH3 = document.createElement("h3");
  fluxFieldReadoutH3.textContent = "Field power";
  fluxFieldReadout.appendChild(fluxFieldReadoutH3);
  function fieldRow(label) {
    const row = document.createElement("div");
    row.className = "rmetric";
    const lbl = document.createElement("div");
    lbl.className = "rlbl";
    lbl.textContent = label;
    const num = document.createElement("div");
    num.className = "rnum";
    row.appendChild(lbl);
    row.appendChild(num);
    fluxFieldReadout.appendChild(row);
    return num;
  }
  const fluxFieldCount = fieldRow("heliostats");
  const fluxFieldTotal = fieldRow("total power");
  const fluxFieldLegend = document.createElement("div");
  fluxFieldLegend.className = "fieldlegend";
  const fluxFieldLegendTitle = document.createElement("div");
  fluxFieldLegendTitle.className = "fieldlegend-title";
  fluxFieldLegendTitle.textContent = "kW delivered per heliostat";
  const fluxFieldLegendBar = document.createElement("div");
  fluxFieldLegendBar.className = "fieldlegend-bar";
  const fluxFieldLegendEnds = document.createElement("div");
  fluxFieldLegendEnds.className = "fieldlegend-ends";
  const fluxFieldLegendMin = document.createElement("span");
  const fluxFieldLegendMax = document.createElement("span");
  fluxFieldLegendEnds.appendChild(fluxFieldLegendMin);
  fluxFieldLegendEnds.appendChild(fluxFieldLegendMax);
  fluxFieldLegend.appendChild(fluxFieldLegendTitle);
  fluxFieldLegend.appendChild(fluxFieldLegendBar);
  fluxFieldLegend.appendChild(fluxFieldLegendEnds);
  fluxFieldReadout.appendChild(fluxFieldLegend);
  fluxMapBody.appendChild(fluxFieldReadout);

  fluxPanel.appendChild(fluxMapBody);

  // docs/ui-spec-v0.2.md §C2: "a fixed on-screen note under the map, visible
  // by default (no badge, no hover-only wording)" -- verbatim wording,
  // shown only while the plan projection is on screen (surface==="secondary"
  // && view==="plan").
  const fluxSecondaryPlanNote = document.createElement("div");
  fluxSecondaryPlanNote.className = "caption an-secplannote";
  fluxSecondaryPlanNote.textContent =
    "Plan projection — not equal-area. Flux values are true W/m² on the tilted surface; screen area is not proportional to surface area.";
  fluxSecondaryPlanNote.hidden = true;
  fluxPanel.appendChild(fluxSecondaryPlanNote);

  const fluxCaption = document.createElement("div");
  fluxCaption.className = "caption";
  fluxPanel.appendChild(fluxCaption);
  // Compass rider (§M): quiet caption text, same idiom as fluxCaption above
  // -- not a pixel overlay on this matplotlib PNG, which carries axis/
  // colorbar/title chrome this module has no exact pixel geometry for.
  const fluxCompass = document.createElement("div");
  fluxCompass.className = "caption an-compass";
  fluxPanel.appendChild(fluxCompass);
  // docs/ui-spec-v0.2.md §D: the FEA CSV grid for this timestep. Only a
  // link (like energyCsv above), not a fetch -- and only ever shown for a
  // step the sweep itself stored a map for (fluxSrcUrl set): the on-demand
  // (uncapped-step) trace and a reopened saved run's map are both plain
  // flux_png bytes with no raw grid behind them to export.
  const fluxFeaCsv = document.createElement("a");
  fluxFeaCsv.href = "#";
  fluxFeaCsv.textContent = "Export CSV for FEA";
  fluxFeaCsv.className = "an-fea-export";
  fluxFeaCsv.hidden = true;
  // §R: the instant source has no stored-blob URL to link to (see
  // exportInstantFluxFeaCsv's own comment on the backend gap it closes) --
  // for that source only, this intercepts the click and fetches instead of
  // following the href. The day source's own href (a real download URL,
  // set in paintFluxPanel below) is untouched, so its native navigation
  // still just works.
  fluxFeaCsv.addEventListener("click", (e) => {
    if (activeSource !== "instant") return;
    e.preventDefault();
    exportInstantFluxFeaCsv();
  });
  fluxPanel.appendChild(fluxFeaCsv);
  // docs/ui-spec-v0.2.md §C leftover: the secondary's own FEA CSV for this
  // timestep, same idiom as fluxFeaCsv above -- but the secondary map has no
  // stored-job grid to link to (only the live single-heliostat endpoint
  // /api/trace/secondary_flux_fea.csv, same limitation js/main.js's own
  // secondary export lives with), so this is a fetch-then-download button
  // like the run bar's exportFeaLink, not a plain href.
  const fluxSecFeaCsv = document.createElement("a");
  fluxSecFeaCsv.href = "#";
  fluxSecFeaCsv.textContent = "Export CSV for FEA";
  fluxSecFeaCsv.className = "an-fea-export";
  fluxSecFeaCsv.hidden = true;
  fluxSecFeaCsv.addEventListener("click", (e) => {
    e.preventDefault();
    exportSecondaryFluxFeaCsvForStep();
  });
  fluxPanel.appendChild(fluxSecFeaCsv);
  right.appendChild(fluxPanel);

  // -- sweep drill-down to one heliostat (docs/ui-spec-v0.2.md §M.2) --------
  const drillPanel = document.createElement("div");
  drillPanel.className = "panel an-drillpanel";
  const drillH2 = document.createElement("h2");
  drillH2.textContent = "Heliostat footprint";
  drillPanel.appendChild(drillH2);

  const drillIdRow = document.createElement("div");
  drillIdRow.className = "an-drillidrow";
  const drillIdLabel = document.createElement("label");
  drillIdLabel.textContent = "…or by id";
  const drillIdInput = document.createElement("input");
  drillIdInput.type = "number";
  drillIdInput.className = "val";
  drillIdInput.min = "0";
  drillIdInput.step = "1";
  drillIdInput.placeholder = "heliostat id";
  drillIdInput.addEventListener("change", () => {
    const v = parseInt(drillIdInput.value, 10);
    if (Number.isFinite(v)) selectDrillHeliostat(v);
  });
  drillIdRow.appendChild(drillIdLabel);
  drillIdRow.appendChild(drillIdInput);
  drillPanel.appendChild(drillIdRow);

  const drillRow = document.createElement("div");
  drillRow.className = "an-drillrow";

  const drillPlanCard = document.createElement("div");
  drillPlanCard.className = "an-drillcard";
  const drillPlanHead = document.createElement("div");
  drillPlanHead.className = "an-drillhead";
  drillPlanHead.textContent = "Mini plan — click to select";
  const drillPlanFrame = document.createElement("div");
  drillPlanFrame.className = "an-drillframe";
  const drillPlanSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  drillPlanSvg.setAttribute("viewBox", "0 0 150 120");
  drillPlanSvg.addEventListener("click", (e) => {
    const target = e.target.closest("[data-hid]");
    if (!target) return;
    selectDrillHeliostat(Number(target.dataset.hid));
  });
  drillPlanFrame.appendChild(drillPlanSvg);
  drillPlanCard.appendChild(drillPlanHead);
  drillPlanCard.appendChild(drillPlanFrame);

  const drillFootCard = document.createElement("div");
  drillFootCard.className = "an-drillcard";
  const drillFootHead = document.createElement("div");
  drillFootHead.className = "an-drillhead";
  drillFootHead.textContent = "Footprint";
  const drillFootFrame = document.createElement("div");
  drillFootFrame.className = "an-drillframe";
  const drillFootImg = document.createElement("img");
  drillFootImg.alt = "Selected heliostat's own flux footprint";
  drillFootImg.hidden = true;
  // v0.2 followups: "I'd like to see the Heliostat Footprint somewhat
  // bigger... I can't get a closer look at each spot" -- click-to-expand
  // into the same overlay chrome the workspace's flux thumbnail already
  // uses (js/panels/run.js's thumbImg, same idiom below), rather than
  // enlarging this thumbnail itself and reflowing the drill-down layout.
  drillFootImg.style.cursor = "pointer";
  drillFootImg.title = "Click to enlarge";
  drillFootImg.addEventListener("click", () => openFootprintOverlay());
  const drillFootPlaceholder = document.createElement("p");
  drillFootPlaceholder.className = "placeholder";
  drillFootPlaceholder.textContent = "Pick a heliostat to see its own footprint.";
  drillFootFrame.appendChild(drillFootImg);
  drillFootFrame.appendChild(drillFootPlaceholder);
  drillFootCard.appendChild(drillFootHead);
  drillFootCard.appendChild(drillFootFrame);

  drillRow.appendChild(drillPlanCard);
  drillRow.appendChild(drillFootCard);
  drillPanel.appendChild(drillRow);
  const drillCaption = document.createElement("div");
  drillCaption.className = "caption";
  drillCaption.textContent =
    "Computed on demand as a single-heliostat trace at the stored sun position -- the same mechanism the field map above already uses, so nothing new is stored.";
  drillPanel.appendChild(drillCaption);
  right.appendChild(drillPanel);

  // -- analysis aperture (docs/ui-spec-v0.2.md §M.4, riders on M.4 = M.6) ---
  // M.6 rearranges this panel: the encircled-power curve moves beside the
  // map (an-aperturebody, a row) instead of stacked below it, and the
  // readout becomes one horizontal strip beneath both (an-apreadoutbar)
  // instead of a tall 230px side card -- neither fits the 360px .an-right
  // column the rest of this tab's cards live in, so the whole panel is
  // full-width now (see contentRow/content above).
  const aperturePanel = document.createElement("div");
  aperturePanel.className = "panel an-aperturepanel";
  const apertureH2 = document.createElement("h2");
  apertureH2.textContent = "Analysis aperture";
  aperturePanel.appendChild(apertureH2);

  const apertureMsg = document.createElement("p");
  apertureMsg.className = "placeholder";
  aperturePanel.appendChild(apertureMsg);

  const apertureBody = document.createElement("div");
  apertureBody.className = "an-aperturebody";

  const apertureCanvasWrap = document.createElement("div");
  apertureCanvasWrap.className = "an-aperturecanvaswrap";
  const apertureCanvas = document.createElement("canvas");
  apertureCanvas.className = "an-aperturecanvas";
  apertureCanvas.addEventListener("pointerdown", apertureHandlePointerDown);
  apertureCanvas.addEventListener("pointermove", apertureHandlePointerMove);
  apertureCanvas.addEventListener("pointerup", apertureHandlePointerUp);
  apertureCanvas.addEventListener("pointercancel", apertureHandlePointerUp);
  apertureCanvasWrap.appendChild(apertureCanvas);
  const apertureCanvasCaption = document.createElement("div");
  apertureCanvasCaption.className = "caption";
  apertureCanvasCaption.textContent = "Dashed circle: drag to move · square: drag to resize.";
  apertureCanvasWrap.appendChild(apertureCanvasCaption);

  const apertureCurveFrame = document.createElement("div");
  apertureCurveFrame.className = "an-aperturecurveframe";
  const apertureCurveH3 = document.createElement("div");
  apertureCurveH3.className = "an-drillhead";
  apertureCurveH3.textContent = "Encircled power vs. aperture radius";
  const apertureCurveCanvas = document.createElement("canvas");
  apertureCurveCanvas.className = "an-aperturecurvecanvas";
  apertureCurveFrame.appendChild(apertureCurveH3);
  apertureCurveFrame.appendChild(apertureCurveCanvas);

  apertureBody.appendChild(apertureCanvasWrap);
  apertureBody.appendChild(apertureCurveFrame);
  aperturePanel.appendChild(apertureBody);

  // -- M.6: horizontal readout strip -- three typed fields (center u/v,
  // radius, all mm) that write straight into the same module state the
  // drag handlers above do, beside the same four read-only metrics as
  // before. One row, wrapping on a narrow window rather than the old
  // fixed-width side card.
  const apertureReadout = document.createElement("div");
  apertureReadout.className = "readout an-apreadoutbar";
  const apertureReadoutH3 = document.createElement("h3");
  apertureReadoutH3.textContent = "Aperture readout";
  apertureReadout.appendChild(apertureReadoutH3);

  const apertureReadoutRow = document.createElement("div");
  apertureReadoutRow.className = "an-apreadoutrow";
  apertureReadout.appendChild(apertureReadoutRow);

  function apInput(label, tooltip, onInput) {
    const field = document.createElement("div");
    field.className = "an-apfield";
    field.title = tooltip;
    const lbl = document.createElement("label");
    lbl.textContent = label;
    const input = document.createElement("input");
    input.type = "number";
    input.className = "val";
    input.step = "1";
    input.addEventListener("input", onInput);
    field.appendChild(lbl);
    field.appendChild(input);
    apertureReadoutRow.appendChild(field);
    return input;
  }
  // §G-style tooltips: one sentence, plain language, matching the existing
  // "Dashed circle: drag..." caption above rather than duplicating it.
  const apCenterUInput = apInput(
    "Center u (mm)",
    "East/west position of the aperture's center on the flux map, in millimeters -- typing moves the circle, same as dragging it.",
    apertureHandleCenterUInput
  );
  const apCenterVInput = apInput(
    "Center v (mm)",
    "North/south position of the aperture's center on the flux map, in millimeters -- typing moves the circle, same as dragging it.",
    apertureHandleCenterVInput
  );
  const apRadiusInput = apInput(
    "Radius (mm)",
    "Radius of the aperture circle, in millimeters -- typing resizes the circle, same as dragging its resize handle.",
    apertureHandleRadiusInput
  );

  function apRow(label) {
    const row = document.createElement("div");
    row.className = "rmetric an-apmetric";
    const lbl = document.createElement("div");
    lbl.className = "rlbl";
    lbl.textContent = label;
    const num = document.createElement("div");
    num.className = "rnum";
    row.appendChild(lbl);
    row.appendChild(num);
    apertureReadoutRow.appendChild(row);
    return num;
  }
  const apPower = apRow("power within");
  const apFrac = apRow("of collected");
  const apFlux = apRow("avg flux");
  const apConc = apRow("avg concentration");
  const apFormula = document.createElement("div");
  apFormula.className = "rformula";
  apFormula.textContent = "avg C = avg flux ÷ DNI";
  apertureReadout.appendChild(apFormula);
  const apFrozenNote = document.createElement("div");
  apFrozenNote.className = "hint";
  apFrozenNote.textContent = "Saved with this run -- shown as saved, not recomputed.";
  apFrozenNote.hidden = true;
  apertureReadout.appendChild(apFrozenNote);
  const apHint = document.createElement("div");
  apHint.className = "hint";
  apHint.textContent =
    "Type exact values or drag on the map above; saves with the run as an annotation. Post-processing only -- the trace, intercept and collected totals never change.";
  apertureReadout.appendChild(apHint);

  aperturePanel.appendChild(apertureReadout);

  contentRow.appendChild(left);
  contentRow.appendChild(right);
  content.appendChild(contentRow);
  content.appendChild(aperturePanel);

  // -- "Manage saved runs" overlay (docs/ui-spec.md 4) -----------------------
  // Reuses app.css's .overlay/.overlay-panel/.overlay-close (the same
  // flux-map lightbox the workspace run bar uses) rather than declaring a
  // second modal chrome here.
  const manageOverlay = document.createElement("div");
  manageOverlay.className = "overlay";
  manageOverlay.hidden = true;
  manageOverlay.addEventListener("click", (e) => {
    if (e.target === manageOverlay) closeManage();
  });
  const managePanelEl = document.createElement("div");
  managePanelEl.className = "overlay-panel an-managepanel";
  const manageClose = document.createElement("button");
  manageClose.className = "overlay-close";
  manageClose.textContent = "×";
  manageClose.addEventListener("click", () => closeManage());
  const manageH2 = document.createElement("h2");
  manageH2.textContent = "Manage saved runs";
  const manageBody = document.createElement("div");
  manageBody.className = "an-managebody";
  managePanelEl.appendChild(manageClose);
  managePanelEl.appendChild(manageH2);
  managePanelEl.appendChild(manageBody);
  manageOverlay.appendChild(managePanelEl);

  // -- footprint "click to enlarge" overlay (v0.2 followups) ------------------
  // Same chrome as manageOverlay just above -- app.css's shared
  // .overlay/.overlay-panel/.overlay-close, the same lightbox the workspace
  // run bar's flux thumbnail opens (js/main.js's #flux-overlay) -- but this
  // one is built here rather than in next/index.html/main.js because
  // opening it has to kick off ITS OWN higher-dpi trace of the currently
  // selected heliostat/step, which only this module has the request-shaping
  // (heliostatFootprintRequestFor) and drill-selection state for.
  const footprintOverlay = document.createElement("div");
  footprintOverlay.className = "overlay";
  footprintOverlay.hidden = true;
  footprintOverlay.addEventListener("click", (e) => {
    if (e.target === footprintOverlay) closeFootprintOverlay();
  });
  const footprintPanelEl = document.createElement("div");
  footprintPanelEl.className = "overlay-panel";
  const footprintClose = document.createElement("button");
  footprintClose.type = "button";
  footprintClose.className = "overlay-close";
  footprintClose.setAttribute("aria-label", "Close");
  footprintClose.textContent = "×";
  footprintClose.addEventListener("click", () => closeFootprintOverlay());
  const footprintOverlayImg = document.createElement("img");
  footprintOverlayImg.alt = "Selected heliostat's own flux footprint, enlarged";
  footprintOverlayImg.hidden = true;
  const footprintOverlayPlaceholder = document.createElement("p");
  footprintOverlayPlaceholder.className = "placeholder";
  const footprintOverlayCaption = document.createElement("div");
  footprintOverlayCaption.className = "caption";
  footprintPanelEl.appendChild(footprintClose);
  footprintPanelEl.appendChild(footprintOverlayImg);
  footprintPanelEl.appendChild(footprintOverlayPlaceholder);
  footprintPanelEl.appendChild(footprintOverlayCaption);
  footprintOverlay.appendChild(footprintPanelEl);

  // Esc closes it, same convention as js/main.js's #flux-overlay -- but
  // handled here, in the capture phase, so it can stop the keypress before
  // main.js's own document-level Escape handler also runs and backs the
  // whole Analysis tab out to 3D View in the same keystroke (that handler
  // has no notion of "a nested overlay already consumed this Escape").
  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "Escape" || footprintOverlay.hidden) return;
      closeFootprintOverlay();
      e.stopPropagation();
    },
    true
  );

  container.appendChild(subject);
  container.appendChild(savedRunsBar);
  container.appendChild(content);
  container.appendChild(manageOverlay);
  container.appendChild(footprintOverlay);

  els = {
    staleChip,
    sweepPanel,
    energyPanel,
    tsPanel,
    fluxPanel,
    instantPanel,
    instantSummary,
    instantRunSaveBtn,
    instantRunDiscardBtn,
    instantRunStatus,
    instantRunErrEl,
    sourceDayBtn,
    sourceYearBtn,
    sourceInstantBtn,
    subjOptics,
    subjField,
    subjDesign,
    dateInput,
    stepInput,
    elevInput,
    fidelityBtns,
    startBtn,
    cancelBtn,
    durationHint,
    progressRow,
    progressFill,
    progressText,
    statusLine,
    sweepErr,
    energyImg,
    energyPlaceholder,
    energyTotal,
    energyCsv,
    tbody,
    tsEmpty,
    tsWrap,
    fluxSurfaceReceiverBtn,
    fluxSurfaceSecondaryBtn,
    fluxSurfaceFieldBtn,
    fluxSecondaryPlanNote,
    fluxSecondaryCanvas,
    fluxSecondaryReadout,
    fluxFieldCanvas,
    fluxFieldReadout,
    fluxFieldCount,
    fluxFieldTotal,
    fluxFieldLegendMin,
    fluxFieldLegendMax,
    secIncidentRow,
    secAbsorbedRow,
    secPeakAbsorbedRow,
    fluxSecFidelity,
    fluxImg,
    fluxPlaceholder,
    fluxCaption,
    fluxCompass,
    openIn3DLink,
    fluxFeaCsv,
    fluxSecFeaCsv,
    drillPanel,
    drillIdInput,
    drillPlanSvg,
    drillFootHead,
    drillFootImg,
    drillFootPlaceholder,
    aperturePanel,
    apertureMsg,
    apertureBody,
    apertureCanvasWrap,
    apertureCanvas,
    apertureReadout,
    apCenterUInput,
    apCenterVInput,
    apRadiusInput,
    apPower,
    apFrac,
    apFlux,
    apConc,
    apFrozenNote,
    apertureCurveFrame,
    apertureCurveCanvas,
    savedRunsLabel,
    savedRunsList,
    dayReopenBanner,
    dayRunSaveBtn,
    dayRunDiscardBtn,
    dayRunStatus,
    dayRunErrEl,
    yearFastBtn,
    yearAllBtn,
    yearStartBtn,
    yearCancelBtn,
    yearDurationHint,
    yearProgressRow,
    yearProgressFill,
    yearProgressText,
    yearStatusLine,
    yearStaleChip,
    yearErr,
    yearTotal,
    yearFrame,
    yearImg,
    yearPlaceholder,
    yearDaysWrap,
    yearDaysTbody,
    yearReopenBanner,
    yearRunSaveBtn,
    yearRunDiscardBtn,
    yearRunStatus,
    yearRunErrEl,
    manageOverlay,
    manageBody,
    footprintOverlay,
    footprintOverlayImg,
    footprintOverlayPlaceholder,
    footprintOverlayCaption,
  };
  built = true;
}

// -- render (from live state, not just the store) ----------------------------

function paintSubject(doc, ctx) {
  els.subjOptics.textContent = opticsSummary(doc);
  els.subjField.textContent = fieldSummary(doc, (ctx && ctx.geometry) || null);
  els.subjDesign.textContent = designSummary(doc);
}

function paintSweepControls() {
  const running = jobSnapshot && jobSnapshot.state === "running";
  setVal(els.dateInput, formDate);
  setVal(els.stepInput, formHourStep);
  setVal(els.elevInput, formMinElevationDeg);
  const fidelity = store.get("ui.fidelity");
  for (const [key, btn] of Object.entries(els.fidelityBtns)) btn.classList.toggle("active", key === fidelity);

  const busy = starting || running;
  els.startBtn.classList.toggle("disabled-link", busy);
  els.startBtn.textContent = starting ? "Starting…" : running ? "Tracing…" : "Trace day sweep";
  els.cancelBtn.hidden = !running;
  els.cancelBtn.classList.toggle("disabled-link", cancelling);
  els.cancelBtn.textContent = cancelling ? "Cancelling…" : "Cancel";

  if (!busy) {
    const nHeliostats = currentHeliostatCount(lastCtx);
    const nTimesteps = estimateDayTimesteps(formHourStep, formMinElevationDeg);
    const estimateS = estimateDurationS(fidelity, nHeliostats, nTimesteps);
    const text = durationWarningText(estimateS, nHeliostats, nTimesteps);
    applyDurationHint(els.durationHint, estimateS, text);
  } else {
    els.durationHint.hidden = true;
  }

  els.progressRow.hidden = !running;
  if (running) {
    const snap = jobSnapshot;
    const pct = snap.total ? Math.min(100, (100 * snap.done) / snap.total) : 0;
    els.progressFill.style.width = pct.toFixed(1) + "%";
    const eta = fmtDuration(snap.eta_s);
    let text = `${snap.done} / ${snap.total} timesteps`;
    if (snap.detail) text += ` — ${snap.detail}`;
    text += ` — elapsed ${fmtDuration(snap.elapsed_s) || "0 s"}`;
    if (eta) text += ` — about ${eta} left`;
    if (cancelling) text += " — cancelling…";
    els.progressText.textContent = text;
  }

  if (!running && jobSnapshot) {
    const elapsed = fmtDuration(jobSnapshot.elapsed_s) || `${jobSnapshot.elapsed_s} s`;
    let text = null;
    if (jobSnapshot.state === "done") {
      text = `Finished ${jobSnapshot.total} timesteps in ${elapsed}.`;
    } else if (jobSnapshot.state === "cancelled") {
      text = `Cancelled after ${jobSnapshot.done} of ${jobSnapshot.total} timesteps (${elapsed}).`;
    }
    els.statusLine.hidden = !text;
    if (text) els.statusLine.textContent = text;
  } else {
    els.statusLine.hidden = true;
  }

  els.sweepErr.hidden = !dayError;
  if (dayError) els.sweepErr.textContent = dayError;
}

function paintEnergyPanel() {
  // §R: nothing here applies to a single instant -- no time axis to
  // integrate over, so the whole card (plot, day/year totals, CSV export)
  // stays hidden while Traced instant is the active source.
  els.energyPanel.hidden = activeSource === "instant";
  if (els.energyPanel.hidden) return;
  if (dayResult && dayResult.plot_png) {
    els.energyImg.src = "data:image/png;base64," + dayResult.plot_png;
    els.energyImg.hidden = false;
    els.energyPlaceholder.hidden = true;
  } else {
    els.energyImg.hidden = true;
    els.energyPlaceholder.hidden = false;
    if (jobSnapshot && jobSnapshot.state === "cancelled" && (!dayResult || !dayResult.steps || !dayResult.steps.length)) {
      els.energyPlaceholder.textContent = "Cancelled before any timestep finished — nothing to plot.";
    } else {
      els.energyPlaceholder.textContent = "Run a day sweep to see energy collected through the day.";
    }
  }

  const haveSteps = dayResult && dayResult.steps && dayResult.steps.length;
  if (haveSteps) {
    const label = dayResult.state === "cancelled" ? "Day total so far" : "Day total";
    els.energyTotal.innerHTML = "";
    els.energyTotal.appendChild(document.createTextNode(`${label}: `));
    const strong = document.createElement("strong");
    strong.textContent = fmtEnergy(dayResult.energy_kwh);
    els.energyTotal.appendChild(strong);
    // Spec §M.7: state the DNI this sweep's power/energy assumed.
    if (dayResult.dni_note) {
      els.energyTotal.appendChild(document.createTextNode(` (DNI: ${dayResult.dni_note})`));
    }
  } else {
    els.energyTotal.textContent = "";
  }
  els.energyCsv.hidden = !(haveSteps && resultJobId);
  if (haveSteps && resultJobId) els.energyCsv.href = dayExportUrl(resultJobId);
}

const tsRowEls = [];

function renderTimestepsRows(steps) {
  if (tsRowEls.length !== steps.length) {
    els.tbody.innerHTML = "";
    tsRowEls.length = 0;
    steps.forEach((_, i) => {
      const tr = document.createElement("tr");
      tr.dataset.idx = String(i);
      const tdSolar = document.createElement("td");
      const tdEl = document.createElement("td");
      const tdPower = document.createElement("td");
      const tdFlux = document.createElement("td");
      tr.appendChild(tdSolar);
      tr.appendChild(tdEl);
      tr.appendChild(tdPower);
      tr.appendChild(tdFlux);
      els.tbody.appendChild(tr);
      tsRowEls.push({ tr, tdSolar, tdEl, tdPower, tdFlux });
    });
  }
  steps.forEach((s, i) => {
    const row = tsRowEls[i];
    row.tdSolar.textContent = fmtHHMM(s.hour);
    row.tdEl.textContent = s.solar_el_deg != null ? s.solar_el_deg.toFixed(1) : "—";
    row.tdPower.textContent = fmtPower(s.power_w);
    row.tdFlux.textContent = fmtFlux(s.peak_flux_kw_m2);
    row.tr.classList.toggle("sel", i === selectedStepIndex);
  });
}

function paintTimestepsTable() {
  // §R: a traced instant is never a Timesteps-table row -- the whole table
  // stays hidden while it is the active source (there is no time axis for
  // one instant to sit on).
  els.tsPanel.hidden = activeSource === "instant";
  if (els.tsPanel.hidden) return;
  const steps = (dayResult && dayResult.steps) || [];
  els.tsWrap.hidden = steps.length === 0;
  els.tsEmpty.hidden = steps.length !== 0;
  if (steps.length) renderTimestepsRows(steps);
}

// docs/secondary-irradiance-plan.md: "UI must say coarse in cone modes,
// exact in Monte Carlo wherever the secondary map shows" -- visible text,
// not a tooltip. Same wording as js/main.js's identical helper for the run
// bar's flux overlay (mockup M9 draws this disclosure in both places).
function secondaryFidelityNote(fidelity) {
  if (fidelity === "exact") {
    return "Exact — Monte Carlo traces the full flux map on the secondary.";
  }
  return "Approximate in this mode — switch to Monte Carlo for an exact map.";
}

// Same compact magma approximation js/scene3d.js's fluxGridTexture and
// js/main.js's flux-overlay painter both use (see either's own comment for
// why these five stops are close enough to matplotlib's real table) -- its
// own copy here rather than a shared import, matching this app's existing
// per-file duplication idiom (this module already keeps its own
// APERTURE_MAGMA_STOPS/magmaColor above for the same reason).
const SECONDARY_FLUX_MAGMA_STOPS = [
  [0.0, 0, 0, 4],
  [0.2, 43, 17, 84],
  [0.4, 120, 28, 109],
  [0.6, 196, 60, 79],
  [0.8, 251, 135, 97],
  [1.0, 252, 253, 191],
];
function secondaryFluxMagmaColor(t) {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < SECONDARY_FLUX_MAGMA_STOPS.length; i++) {
    const [t0, r0, g0, b0] = SECONDARY_FLUX_MAGMA_STOPS[i - 1];
    const [t1, r1, g1, b1] = SECONDARY_FLUX_MAGMA_STOPS[i];
    if (x <= t1 || i === SECONDARY_FLUX_MAGMA_STOPS.length - 1) {
      const f = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
      return [Math.round(r0 + (r1 - r0) * f), Math.round(g0 + (g1 - g0) * f), Math.round(b0 + (b1 - b0) * f)];
    }
  }
  return [0, 0, 4];
}

// docs/ui-spec-v0.2.md §C2, mockup M23 (updated for the Cartesian
// plan-projection binning switch): paints app.py's _flux_grid_payload
// straight onto a 2D canvas -- there is no server-rendered PNG for the
// secondary map (only the opt-in raw grid, spec §C), unlike the receiver's
// own flux_png. This used to be two different presentations (a native
// "unrolled" arc-length/radius raster, and a polar-to-Cartesian "plan"
// projection of it) because the old radial (u, v) parameterization's native
// layout WASN'T a plan view. Now that secondary_uv is itself a Cartesian
// (x_local east, y_local north) pair over the aperture's own square
// bounding box, the grid already IS the plan view -- painting it is a
// direct raster, not a projection, so the old toggle and its second painter
// are gone (synthesizing a distinct "unrolled" layout from data that is
// already Cartesian would show nothing the plan view doesn't).
//
// `values` is row-major, row 0 = v_min (the bottom of the map, south) --
// canvas y grows down, so row 0 is drawn at the canvas's own bottom edge,
// same flip every other flux painter in this app uses (matplotlib's
// origin="lower", scene3d.js's fluxGridTexture). A bin whose centre falls
// outside the aperture disk is left unpainted (background shows through)
// rather than drawn in its own (always-zero, per
// secondary_bin_areas_m2's masking) color -- that is what makes the result
// read as a circular disk rather than a dark-cornered square.
function paintSecondaryFluxCanvasPlan(canvas, grid) {
  const { n_u, n_v, u_min_mm, u_max_mm, v_min_mm, v_max_mm, values } = grid;
  const apertureRadiusMm = Math.max((u_max_mm - u_min_mm) / 2, (v_max_mm - v_min_mm) / 2, 1e-6);
  const size = 380;
  canvas.width = size;
  canvas.height = size;
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const margin = 22; // room for the N/S/E/W labels outside the circle
  const scale = (size / 2 - margin) / apertureRadiusMm;

  let max = 0;
  for (const v of values) if (v != null && v > max) max = v;
  const duMm = (u_max_mm - u_min_mm) / n_u;
  const dvMm = (v_max_mm - v_min_mm) / n_v;
  const wPx = duMm * scale + 0.75; // slight overlap so adjacent bins don't
  const hPx = dvMm * scale + 0.75; // leave antialiasing seams between them

  for (let row = 0; row < n_v; row++) {
    const yMm = v_min_mm + (row + 0.5) * dvMm;
    for (let col = 0; col < n_u; col++) {
      const xMm = u_min_mm + (col + 0.5) * duMm;
      if (Math.hypot(xMm, yMm) > apertureRadiusMm) continue; // outside the aperture disk
      const val = values[row * n_u + col];
      const t = max > 0 && val != null ? val / max : 0;
      const [r, g, b] = secondaryFluxMagmaColor(t);
      ctx2d.fillStyle = `rgb(${r},${g},${b})`;
      const px = cx + xMm * scale - wPx / 2;
      const py = cy - yMm * scale - hPx / 2;
      ctx2d.fillRect(px, py, wPx, hPx);
    }
  }

  // Compass markers (§C2/§M rider) -- N/E/S/W just outside the rim, matching
  // the same "x east, y north" world-frame convention every other
  // compass-marked map/export in this app uses.
  ctx2d.fillStyle = "#334155";
  ctx2d.font = "11px sans-serif";
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "alphabetic";
  const rEdge = apertureRadiusMm * scale;
  ctx2d.fillText("N", cx, cy - rEdge - 8);
  ctx2d.fillText("S", cx, cy + rEdge + 15);
  ctx2d.textAlign = "left";
  ctx2d.fillText("E", cx + rEdge + 5, cy + 4);
  ctx2d.textAlign = "right";
  ctx2d.fillText("W", cx - rEdge - 5, cy + 4);
}

// Thin alias kept so call sites read the same as before this file's own
// toggle removal -- there is only one secondary presentation now.
function paintSecondaryFlux(canvas, grid) {
  paintSecondaryFluxCanvasPlan(canvas, grid);
}

// v0.2 followups item 1, mockup M15: plan-view power coloring -- one dot per
// heliostat at its own (x_mm, y_mm), colored by its own power_w. Same
// function as main.js's own paintFieldMapCanvas (own copy per this app's
// per-file duplication idiom for small painters -- see that function's own
// comment); returns {minKw, maxKw} so the caller paints the legend numbers
// from the same values this canvas was colored with.
function paintFieldMapCanvas(canvas, heliostats) {
  const size = 380;
  canvas.width = size;
  canvas.height = size;
  const ctx2d = canvas.getContext("2d");
  ctx2d.fillStyle = "#fdfdfe";
  ctx2d.fillRect(0, 0, size, size);

  let maxR = 1;
  let minKw = Infinity;
  let maxKw = -Infinity;
  for (const h of heliostats) {
    maxR = Math.max(maxR, Math.hypot(h.x_mm, h.y_mm));
    if (h.failed) continue;
    const kw = (h.power_w || 0) / 1000;
    if (kw < minKw) minKw = kw;
    if (kw > maxKw) maxKw = kw;
  }
  if (!Number.isFinite(minKw)) minKw = 0;
  if (!Number.isFinite(maxKw)) maxKw = 0;
  const span = maxKw - minKw;

  const cx = size / 2;
  const cy = size / 2;
  const scale = (size / 2 - 12) / maxR;
  const dotR = Math.max(1.2, Math.min(4, 220 / Math.sqrt(Math.max(1, heliostats.length))));
  for (const h of heliostats) {
    const px = cx + h.x_mm * scale;
    // World y (north) is up on a plan view; canvas y grows downward -- flip.
    const py = cy - h.y_mm * scale;
    ctx2d.beginPath();
    ctx2d.arc(px, py, dotR, 0, Math.PI * 2);
    if (h.failed) {
      ctx2d.fillStyle = "#c7cdd6";
    } else {
      const t = span > 0 ? ((h.power_w || 0) / 1000 - minKw) / span : 1;
      const [r, g, b] = secondaryFluxMagmaColor(t);
      ctx2d.fillStyle = `rgb(${r},${g},${b})`;
    }
    ctx2d.fill();
  }

  ctx2d.beginPath();
  ctx2d.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx2d.fillStyle = "#7b8794";
  ctx2d.fill();
  ctx2d.lineWidth = 1.3;
  ctx2d.strokeStyle = "#33455c";
  ctx2d.stroke();

  return { minKw, maxKw };
}

// Repaints the Receiver | Secondary selector against whether THIS step's
// currently-loaded map actually carries a secondary block, and returns
// whether the secondary map should be the one shown. Secondary stays
// disabled (with a tooltip explaining why) for prime_focus, for a step
// still loading/erroring, for a reopened saved run (SavedRunDocument keeps
// PNG bytes only, no secondary blob), and for a stored sweep step whose own
// secondary fetch (scheduleFluxFetch, via getDaySecondaryGrid) came back
// empty -- an older run from before the stored-step gap closed, a step the
// receiver-grid cap itself skipped, or a prime-focus sweep.
function paintFluxSurfaceSelector() {
  const doc = store.get("doc");
  const hasSecondaryOptics = doc.optics === "axicon" || doc.optics === "cassegrain";
  const secAvailable = !!(fluxSecondary && fluxSecondary.flux_grid);
  els.fluxSurfaceSecondaryBtn.classList.toggle("disabled", !secAvailable);
  let secTip = "";
  if (!secAvailable) {
    if (!hasSecondaryOptics) secTip = "Only axicon and Cassegrain layouts have a secondary flux map.";
    else if (fluxSrcUrl && storedSecondaryLoading) {
      secTip = "Loading this step's secondary flux data…";
    } else if (fluxSrcUrl || reopenedDayFluxPngs) {
      secTip = "This stored sweep step carries no secondary flux data.";
    } else secTip = "This trace carried no secondary flux map.";
  }
  els.fluxSurfaceSecondaryBtn.title = secTip;

  // v0.2 followups item 1: Field needs THIS step's own per-heliostat rows
  // (fluxHeliostats), which only ever exist after a live on-demand re-trace
  // of an uncached step -- never for a stored sweep step (fluxSrcUrl) or a
  // reopened saved run, and never for a single heliostat (doc.field.mode).
  const fieldAvailable = !!(fluxHeliostats && fluxHeliostats.length);
  els.fluxSurfaceFieldBtn.classList.toggle("disabled", !fieldAvailable);
  let fieldTip = "";
  if (!fieldAvailable) {
    if (doc.field.mode !== "field") fieldTip = "Field coloring needs a field, not a single heliostat.";
    else if (fluxLoading) fieldTip = "Tracing…";
    else if (fluxSrcUrl) {
      fieldTip = "Re-trace this step to color heliostats by power.";
    } else if (reopenedDayFluxPngs) {
      fieldTip = "Re-trace this run to color heliostats by power.";
    } else fieldTip = "Re-trace this step to color heliostats by power.";
  }
  els.fluxSurfaceFieldBtn.title = fieldTip;

  const requested = store.get("ui.fluxSurface");
  const showSecondary = requested === "secondary" && secAvailable;
  const showField = requested === "field" && fieldAvailable;
  const showReceiver = !showSecondary && !showField;
  els.fluxSurfaceReceiverBtn.classList.toggle("active", showReceiver);
  els.fluxSurfaceSecondaryBtn.classList.toggle("active", showSecondary);
  els.fluxSurfaceFieldBtn.classList.toggle("active", showField);
  if (showSecondary) return "secondary";
  if (showField) return "field";
  return "receiver";
}

// §R: the Traced instant source's own selector/map painters -- reuse the
// same low-level canvas painters (paintSecondaryFluxCanvas/
// paintFieldMapCanvas) and DOM elements as the day source below, but read
// straight off the live/reopened trace response (currentInstantResult())
// rather than any fetch/cache machinery -- there is nothing to fetch or
// re-trace, the response already carries it all (§R's own point).
function paintInstantFluxSurfaceSelector(result) {
  const doc = store.get("doc");
  const hasSecondaryOptics = doc.optics === "axicon" || doc.optics === "cassegrain";
  const secAvailable = !!(result && result.secondary && result.secondary.flux_grid);
  els.fluxSurfaceSecondaryBtn.classList.toggle("disabled", !secAvailable);
  els.fluxSurfaceSecondaryBtn.title = secAvailable
    ? ""
    : hasSecondaryOptics
      ? "This trace carried no secondary flux map."
      : "Only axicon and Cassegrain layouts have a secondary flux map.";

  const fieldAvailable = !!(result && Array.isArray(result.heliostats) && result.heliostats.length);
  els.fluxSurfaceFieldBtn.classList.toggle("disabled", !fieldAvailable);
  els.fluxSurfaceFieldBtn.title = fieldAvailable ? "" : "Field coloring needs a field, not a single heliostat.";

  const requested = store.get("ui.fluxSurface");
  const showSecondary = requested === "secondary" && secAvailable;
  const showField = requested === "field" && fieldAvailable;
  const showReceiver = !showSecondary && !showField;
  els.fluxSurfaceReceiverBtn.classList.toggle("active", showReceiver);
  els.fluxSurfaceSecondaryBtn.classList.toggle("active", showSecondary);
  els.fluxSurfaceFieldBtn.classList.toggle("active", showField);
  if (showSecondary) return "secondary";
  if (showField) return "field";
  return "receiver";
}

// docs/ui-spec-v0.2.md §C2, mockup M23: the fixed not-equal-area note is
// only meaningful while the secondary map is actually on screen -- shared
// by both paintFluxPanel (day/year source) and paintInstantFluxPanel (§R's
// instant source), so the two can never disagree about when to show it.
// (Used to also drive a Plan/Unrolled toggle -- removed now that
// secondary_uv's own binning is Cartesian, see paintSecondaryFluxCanvasPlan's
// comment.)
function paintSecondaryViewControls(showSecondary) {
  els.fluxSecondaryPlanNote.hidden = !showSecondary;
}

function paintInstantFluxPanel() {
  const result = currentInstantResult();
  const request = currentInstantRequest();
  const surface = paintInstantFluxSurfaceSelector(result);
  paintSecondaryViewControls(surface === "secondary");

  function showPlaceholder(text) {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = text;
    els.fluxCaption.textContent = "";
    els.fluxCompass.textContent = "";
    els.openIn3DLink.hidden = true;
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = true;
    paintSecondaryViewControls(false);
  }

  if (!result) {
    return showPlaceholder("Nothing traced yet — trace an instant in 3D View to see it here.");
  }

  els.fluxPlaceholder.hidden = true;
  els.openIn3DLink.hidden = false;

  const azEl =
    request && request.solar_az_deg != null && request.solar_el_deg != null
      ? `Az ${request.solar_az_deg.toFixed(1)}° El ${request.solar_el_deg.toFixed(1)}°`
      : "";

  if (surface === "secondary") {
    els.fluxImg.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    els.fluxSecondaryCanvas.hidden = false;
    paintSecondaryFlux(els.fluxSecondaryCanvas, result.secondary.flux_grid);
    els.fluxCaption.textContent = `incident flux on secondary, kW/m² · same colormap & units as the receiver map · peak ${fmtFlux(result.secondary.peak_flux_kw_m2)}`;
    els.fluxCompass.textContent =
      "Compass: N top · S bottom · E right · W left (plan projection, looking down the optical axis)";
    els.fluxSecondaryReadout.hidden = false;
    els.secIncidentRow.num.textContent = fmtPower(result.secondary.power_w);
    const rPct = (result.secondary.secondary_reflectance * 100).toFixed(1);
    els.secAbsorbedRow.lbl.textContent = `absorbed (R = ${rPct} %)`;
    els.secAbsorbedRow.num.textContent = fmtPower(result.secondary.absorbed_power_w);
    els.secPeakAbsorbedRow.num.textContent = fmtFlux(result.secondary.peak_absorbed_kw_m2);
    els.fluxSecFidelity.textContent = secondaryFidelityNote(result.secondary.fidelity);
    // §R: same reasoning as the day source's identically-placed comment --
    // the export's job is a full-fidelity ANSYS-ready grid, always a fresh
    // network call, never a cached blob.
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = false;
  } else if (surface === "field") {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = false;
    const { minKw, maxKw } = paintFieldMapCanvas(els.fluxFieldCanvas, result.heliostats);
    els.fluxCaption.textContent = `${azEl} · ${result.heliostats.length} heliostats`;
    els.fluxCompass.textContent = "";
    els.fluxFieldReadout.hidden = false;
    els.fluxFieldCount.textContent = result.heliostats.length.toLocaleString();
    const totalW = result.heliostats.reduce((sum, h) => sum + (h.failed ? 0 : h.power_w || 0), 0);
    els.fluxFieldTotal.textContent = fmtPower(totalW);
    els.fluxFieldLegendMin.textContent = minKw.toFixed(1);
    els.fluxFieldLegendMax.textContent = maxKw.toFixed(1);
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = true;
  } else {
    if (!result.flux_png) return showPlaceholder("This trace carried no flux map.");
    els.fluxImg.src = "data:image/png;base64," + result.flux_png;
    els.fluxImg.hidden = false;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    const nH = result.n_heliostats != null ? result.n_heliostats : (result.heliostats || []).length || 1;
    els.fluxCaption.textContent = `${azEl} · ${request && request.mode ? request.mode : ""} · ${nH} heliostat${nH === 1 ? "" : "s"} · peak ${fmtFlux(result.peak_flux_kw_m2)}`;
    // Compass rider (§M) -- see compassCaptionFor.
    els.fluxCompass.textContent = compassCaptionFor(store.get("doc"));
    els.fluxSecFeaCsv.hidden = true;
    // §R's own new endpoints (postFieldFluxFeaCsv/postFluxFeaCsv, via
    // exportInstantFluxFeaCsv) make this always available once there is a
    // request to replay -- no stored-blob gate like the day source's own.
    els.fluxFeaCsv.hidden = !request;
  }
}

function paintFluxPanel() {
  if (activeSource === "instant") return paintInstantFluxPanel();
  const steps = dayResult && dayResult.steps;
  const step = steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;
  const surface = paintFluxSurfaceSelector();
  paintSecondaryViewControls(surface === "secondary");

  function showPlaceholder(text) {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = text;
    els.fluxCaption.textContent = "";
    els.fluxCompass.textContent = "";
    els.openIn3DLink.hidden = true;
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = true;
    paintSecondaryViewControls(false);
  }

  if (!step) return showPlaceholder("Click a timestep to render its irradiance map.");
  if (fluxLoading) return showPlaceholder("Rendering…");
  if (fluxError) return showPlaceholder(fluxError);
  if (!(fluxSrcUrl || fluxPngBase64)) return showPlaceholder("Click a timestep to render its irradiance map.");

  els.fluxPlaceholder.hidden = true;
  els.openIn3DLink.hidden = false;

  if (surface === "secondary") {
    els.fluxImg.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    els.fluxSecondaryCanvas.hidden = false;
    paintSecondaryFlux(els.fluxSecondaryCanvas, fluxSecondary.flux_grid);
    els.fluxCaption.textContent = `incident flux on secondary, kW/m² · same colormap & units as the receiver map · peak ${fmtFlux(fluxSecondary.peak_flux_kw_m2)}`;
    // §C2/§M rider: the plan view carries the same compass convention as
    // every other map in the app.
    els.fluxCompass.textContent =
      "Compass: N top · S bottom · E right · W left (plan projection, looking down the optical axis)";
    els.fluxSecondaryReadout.hidden = false;
    els.secIncidentRow.num.textContent = fmtPower(fluxSecondary.power_w);
    const rPct = (fluxSecondary.secondary_reflectance * 100).toFixed(1);
    els.secAbsorbedRow.lbl.textContent = `absorbed (R = ${rPct} %)`;
    els.secAbsorbedRow.num.textContent = fmtPower(fluxSecondary.absorbed_power_w);
    els.secPeakAbsorbedRow.num.textContent = fmtFlux(fluxSecondary.peak_absorbed_kw_m2);
    els.fluxSecFidelity.textContent = secondaryFidelityNote(fluxSecondary.fidelity);
    // Deliberately still the live single-heliostat re-trace endpoint
    // (exportSecondaryFluxFeaCsvForStep above), not a stored-blob URL like
    // the receiver's dayFluxFeaCsvUrl below, even now that a stored step
    // can carry its own secondary blob for the ON-SCREEN map/readout --
    // the FEA export's job is an ANSYS-ready grid at full trace fidelity,
    // which the downsampled stored blob was never meant to serve.
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = false;
  } else if (surface === "field") {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = false;
    const { minKw, maxKw } = paintFieldMapCanvas(els.fluxFieldCanvas, fluxHeliostats);
    els.fluxCaption.textContent = `${fmtHHMM(step.hour)} solar · ${fluxHeliostats.length} heliostats`;
    els.fluxCompass.textContent = "";
    els.fluxFieldReadout.hidden = false;
    els.fluxFieldCount.textContent = fluxHeliostats.length.toLocaleString();
    const totalW = fluxHeliostats.reduce((sum, h) => sum + (h.failed ? 0 : h.power_w || 0), 0);
    els.fluxFieldTotal.textContent = fmtPower(totalW);
    els.fluxFieldLegendMin.textContent = minKw.toFixed(1);
    els.fluxFieldLegendMax.textContent = maxKw.toFixed(1);
    els.fluxFeaCsv.hidden = true;
    els.fluxSecFeaCsv.hidden = true;
  } else {
    els.fluxImg.src = fluxSrcUrl || "data:image/png;base64," + fluxPngBase64;
    els.fluxImg.hidden = false;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxFieldCanvas.hidden = true;
    els.fluxFieldReadout.hidden = true;
    els.fluxCaption.textContent = `${fmtHHMM(step.hour)} solar · peak ${fmtFlux(fluxPeakKwM2 != null ? fluxPeakKwM2 : step.peak_flux_kw_m2)}`;
    // Compass rider (§M) -- see compassCaptionFor.
    els.fluxCompass.textContent = compassCaptionFor(store.get("doc"));
    els.fluxSecFeaCsv.hidden = true;
    // Only a sweep-stored map (fluxSrcUrl, backed by a live job id) has a
    // raw grid on the server to export -- an on-demand trace's flux_png and
    // a reopened saved run's stored PNG are pixels only, nothing to grid.
    if (fluxSrcUrl && resultJobId != null) {
      els.fluxFeaCsv.href = dayFluxFeaCsvUrl(resultJobId, selectedStepIndex);
      els.fluxFeaCsv.hidden = false;
    } else {
      els.fluxFeaCsv.hidden = true;
    }
  }
}

// -- sweep drill-down to one heliostat (§M.2) --------------------------------

function paintDrillPanel() {
  const doc = store.get("doc");
  let haveStep;
  let isField;
  if (activeSource === "instant") {
    haveStep = !!currentInstantResult();
    const req = currentInstantRequest();
    isField = !!(req && req.layout);
  } else {
    const steps = dayResult && dayResult.steps;
    haveStep = !!(steps && steps.length);
    isField = doc.field.mode === "field";
  }
  els.drillPanel.hidden = !isField || !haveStep;
  if (els.drillPanel.hidden) return;

  paintDrillMiniPlan(els.drillPlanSvg);

  const step = currentStepForAperture();
  const heliostat =
    drillHeliostatId != null ? currentFieldHeliostats().find((h) => h.id === drillHeliostatId) : null;
  els.drillIdInput.value = drillHeliostatId != null ? String(drillHeliostatId) : "";
  els.drillFootHead.textContent = heliostat && step
    ? `H-${heliostat.id} footprint — ${activeSource === "instant" ? "traced instant" : fmtHHMM(step.hour)}`
    : "Footprint";

  if (!heliostat) {
    els.drillFootImg.hidden = true;
    els.drillFootPlaceholder.hidden = false;
    els.drillFootPlaceholder.textContent = "Pick a heliostat to see its own footprint.";
    return;
  }
  if (heliostatFootprintLoading) {
    els.drillFootImg.hidden = true;
    els.drillFootPlaceholder.hidden = false;
    els.drillFootPlaceholder.textContent = "Tracing…";
    return;
  }
  if (heliostatFootprintError) {
    els.drillFootImg.hidden = true;
    els.drillFootPlaceholder.hidden = false;
    els.drillFootPlaceholder.textContent = heliostatFootprintError;
    return;
  }
  if (heliostatFootprintPngBase64) {
    els.drillFootImg.src = "data:image/png;base64," + heliostatFootprintPngBase64;
    els.drillFootImg.hidden = false;
    els.drillFootPlaceholder.hidden = true;
  } else {
    els.drillFootImg.hidden = true;
    els.drillFootPlaceholder.hidden = false;
    els.drillFootPlaceholder.textContent = "Pick a heliostat to see its own footprint.";
  }
}

// -- analysis aperture (§M.4) -------------------------------------------------

// M.6: `frozen` also disables the three typed fields -- a reopened saved
// run has no live grid to recompute against (see paintAperturePanel's own
// reopenedAperture branch), so editing them would have nothing to act on.
function renderApertureReadout(data, frozen) {
  setVal(els.apCenterUInput, data.center_u_mm != null ? Math.round(data.center_u_mm) : "");
  setVal(els.apCenterVInput, data.center_v_mm != null ? Math.round(data.center_v_mm) : "");
  setVal(els.apRadiusInput, data.radius_mm != null ? Math.round(data.radius_mm) : "");
  els.apCenterUInput.disabled = !!frozen;
  els.apCenterVInput.disabled = !!frozen;
  els.apRadiusInput.disabled = !!frozen;
  els.apPower.textContent = data.power_w != null ? fmtPower(data.power_w) : "—";
  els.apFrac.textContent = data.frac_collected_pct != null ? data.frac_collected_pct.toFixed(1) + " %" : "—";
  els.apFlux.textContent = data.avg_flux_w_m2 != null ? fmtFlux(data.avg_flux_w_m2 / 1000) : "—";
  els.apConc.textContent = data.avg_concentration != null ? data.avg_concentration.toFixed(0) + "×" : "—";
  els.apFrozenNote.hidden = !frozen;
}

function paintAperturePanel() {
  const doc = store.get("doc");
  if (!apertureReceiverIsFlat(doc)) {
    // §M.4 scopes the aperture to flat receivers first ("curved later if
    // wanted") -- the readout math above assumes a uniform bin area, which
    // is not exact for a frustum.
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = false;
    els.apertureMsg.textContent = "The analysis aperture is available for flat receivers (curved receivers are not yet supported).";
    els.apertureBody.hidden = true;
    els.apertureCurveFrame.hidden = true;
    els.apertureReadout.hidden = true;
    return;
  }

  const step = currentStepForAperture();
  const grid = currentAnyApertureGrid();

  // A reopened run's frozen annotation -- shown verbatim, never recomputed
  // (mockup M17's own checknote). Day/year: reopenedAperture belongs to a
  // specific stored timestep (selectedStepIndex must still match it). §R: a
  // reopened instant has exactly one "timestep", so reopenedInstant.aperture
  // always applies whenever there is one -- no index to compare.
  const frozen =
    activeSource === "instant"
      ? reopenedInstant && reopenedInstant.aperture
      : reopenedAperture && selectedStepIndex === reopenedAperture.step_index
        ? reopenedAperture
        : null;
  if (frozen) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = true;
    els.apertureBody.hidden = false;
    els.apertureCanvasWrap.hidden = true; // no live grid to draw the circle against
    els.apertureReadout.hidden = false;
    els.apertureCurveFrame.hidden = true;
    renderApertureReadout(frozen, true);
    return;
  }
  if (activeSource !== "instant" && reopenedAperture) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = false;
    els.apertureMsg.textContent =
      "This run's saved aperture belongs to a different timestep -- select it to see the same circle and readout.";
    els.apertureBody.hidden = true;
    els.apertureCurveFrame.hidden = true;
    els.apertureReadout.hidden = true;
    return;
  }

  // Gate on the grid itself, not just the conditions that make one worth
  // fetching -- scheduleApertureGridFetch's own fetch is still in flight
  // the first time paint() runs right after a step becomes selected (it
  // sets apertureGridLoading and repaints before the network call lands),
  // so step.has_flux_map/resultJobId being fine is not enough to know
  // apertureGrid itself is populated yet. §R's own instant grid needs no
  // fetch at all (currentAnyApertureGrid reads it straight off the trace
  // response), so its only gate is "is there a step, and does it have one".
  const jobNotReady = activeSource === "day" && resultJobId == null;
  if (!step || !step.has_flux_map || jobNotReady || !grid) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = false;
    els.apertureMsg.textContent =
      activeSource === "instant"
        ? step
          ? "This trace carried no flux grid to build an aperture from."
          : "Trace an instant in 3D View to use the aperture here."
        : apertureGridLoading
          ? "Loading aperture grid…"
          : apertureGridError || "Select a timestep with a stored irradiance map to use the aperture.";
    els.apertureBody.hidden = true;
    els.apertureCurveFrame.hidden = true;
    els.apertureReadout.hidden = true;
    return;
  }

  els.aperturePanel.hidden = false;
  els.apertureMsg.hidden = true;
  els.apertureBody.hidden = false;
  els.apertureCanvasWrap.hidden = false;
  els.apertureReadout.hidden = false;
  els.apertureCurveFrame.hidden = false;

  const center = currentApertureCenterMm(grid, step);
  const radius = currentApertureRadiusMm(grid, step);
  paintApertureCanvas(els.apertureCanvas, grid, center.u, center.v, radius);

  const { powerW, avgFluxWM2 } = apertureMetrics(grid, center.u, center.v, radius);
  const collectedW = step.power_w;
  const dni = step.dni_w_m2;
  renderApertureReadout(
    {
      center_u_mm: center.u,
      center_v_mm: center.v,
      radius_mm: radius,
      power_w: powerW,
      frac_collected_pct: collectedW ? (100 * powerW) / collectedW : null,
      avg_flux_w_m2: avgFluxWM2,
      avg_concentration: dni ? avgFluxWM2 / dni : null,
    },
    false
  );

  const halfU = Math.abs(grid.u_max_mm - grid.u_min_mm) / 2;
  const halfV = Math.abs(grid.v_max_mm - grid.v_min_mm) / 2;
  const maxR = Math.min(halfU, halfV) * 0.98;
  const curve = apertureCurve(grid, center.u, center.v, maxR, 32);
  paintApertureCurve(els.apertureCurveCanvas, curve, radius, powerW, fmtPower);
}

// -- saved runs (docs/ui-spec.md 4) -------------------------------------------

function runRowLabel(entry) {
  const kind = entry.kind === "year" ? "Year estimate" : entry.kind === "instant" ? "Traced instant" : "Day sweep";
  const when = entry.saved_at ? entry.saved_at.slice(0, 16).replace("T", " ") : "";
  return `${kind} · ${when}`;
}

function paintSavedRunsBar() {
  const projectName = store.get("ui.projectName");
  els.savedRunsList.innerHTML = "";

  if (!projectName) {
    els.savedRunsLabel.textContent = "Saved runs: save the project to keep runs with it.";
    return;
  }
  if (projectRunsLoading) {
    els.savedRunsLabel.textContent = "Saved runs: loading…";
    return;
  }
  if (projectRunsError) {
    els.savedRunsLabel.textContent = `Saved runs: ${projectRunsError}`;
    return;
  }
  if (!projectRunEntries.length) {
    els.savedRunsLabel.textContent = "Saved runs: none yet for this project.";
    return;
  }
  els.savedRunsLabel.textContent = "Saved runs:";
  for (const entry of projectRunEntries) {
    const chip = document.createElement("a");
    chip.href = "#";
    chip.className = "an-runchip";
    chip.textContent = runRowLabel(entry);
    chip.addEventListener("click", (e) => {
      e.preventDefault();
      openSavedRun(entry);
    });
    els.savedRunsList.appendChild(chip);
  }
}

function paintDayRunControls() {
  const isReopened = !!(dayRunSavedName && !resultJobId);
  els.dayReopenBanner.hidden = !isReopened;
  if (isReopened) {
    els.dayReopenBanner.textContent = `Reopened saved run "${dayRunSavedName}" — no re-trace needed.`;
  }

  const haveResult = !!(dayResult && dayResult.steps && dayResult.steps.length);
  const busy = dayRunSaving;
  // Saving needs a live job id to fetch flux maps from (saveDayRun's own
  // guard) -- a discarded reopened run has none, so there is nothing left
  // to save until the sweep runs again.
  els.dayRunSaveBtn.hidden = !haveResult || !resultJobId || !!dayRunSavedName;
  els.dayRunSaveBtn.classList.toggle("disabled-link", busy);
  els.dayRunSaveBtn.textContent = busy && !dayRunSavedName ? "Saving…" : "Save this run";

  els.dayRunDiscardBtn.hidden = !dayRunSavedName;
  els.dayRunDiscardBtn.classList.toggle("disabled-link", busy);
  els.dayRunDiscardBtn.textContent = busy && dayRunSavedName ? "Discarding…" : "Discard this run";

  els.dayRunStatus.textContent = dayRunSavedName && !busy ? `Saved as "${dayRunSavedName}"` : "";
  els.dayRunErrEl.hidden = !dayRunError;
  if (dayRunError) els.dayRunErrEl.textContent = dayRunError;
}

// -- year estimate -------------------------------------------------------------

// -- M.5: year -> day drill-in (docs/ui-spec-v0.2.md §M item 5) -------------
// Reuses startSweep()'s own launch path verbatim, the same one "Trace day
// sweep" calls -- see this file's day-sweep lifecycle section. formDate is
// the only thing this sets before calling it; formHourStep/formMinElevation
// Deg/fidelity stay whatever the day-sweep controls already hold (the
// "current design/site settings" the drill-in is supposed to use), and
// startSweep()'s own resetRunState() clears any earlier day-sweep result on
// screen the same way a manual click would.
const yearDaysRowEls = [];

function renderYearDaysRows(days) {
  if (yearDaysRowEls.length !== days.length) {
    els.yearDaysTbody.innerHTML = "";
    yearDaysRowEls.length = 0;
    days.forEach((_, i) => {
      const tr = document.createElement("tr");
      tr.dataset.idx = String(i);
      const tdDate = document.createElement("td");
      const tdDec = document.createElement("td");
      const tdEnergy = document.createElement("td");
      const tdPeak = document.createElement("td");
      tr.appendChild(tdDate);
      tr.appendChild(tdDec);
      tr.appendChild(tdEnergy);
      tr.appendChild(tdPeak);
      els.yearDaysTbody.appendChild(tr);
      yearDaysRowEls.push({ tr, tdDate, tdDec, tdEnergy, tdPeak });
    });
  }
  days.forEach((d, i) => {
    const row = yearDaysRowEls[i];
    row.tdDate.textContent = d.traced ? d.date : `${d.date} (by symmetry)`;
    // Untraced (symmetry-filled) days have no sun-position trace of their
    // own -- clicking one opens the traced day it was mirrored from instead
    // (openYearDay below), so the day sweep that opens matches the numbers
    // already shown for it here. Traced twin named up front, not just on
    // hover, since it changes what a click actually does.
    row.tr.title = d.traced
      ? `Opens this day as a day sweep.`
      : `Filled in by symmetry from ${d.source_date} -- opens that traced day instead.`;
    row.tdDec.textContent = d.declination_deg != null ? d.declination_deg.toFixed(1) + "°" : "—";
    row.tdEnergy.textContent = fmtEnergy(d.energy_kwh);
    row.tdPeak.textContent = d.peak_power_kw != null ? fmtPower(d.peak_power_kw * 1000) : "—";
  });
}

function openYearDay(idx) {
  const days = yearResult && yearResult.days;
  const entry = days && days[idx];
  if (!entry) return;
  // See renderYearDaysRows's own comment -- source_date is entry.date
  // itself for a traced day, its traced twin's date for a symmetry-filled
  // one, so this line is correct either way.
  formDate = entry.source_date;
  startSweep();
  // "Land in the day view", not just fill in the date (owner's own words)
  // -- the day-sweep panel sits above the year panel that was just clicked
  // in, so scroll up to where the run everyone just started is happening.
  if (els.sweepPanel) {
    els.sweepPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function paintYearControls() {
  els.yearFastBtn.classList.toggle("active", yearFastMode);
  els.yearAllBtn.classList.toggle("active", !yearFastMode);

  const running = yearJobSnapshot && yearJobSnapshot.state === "running";
  const busy = yearStarting || running;
  els.yearStartBtn.classList.toggle("disabled-link", busy);
  els.yearStartBtn.textContent = yearStarting ? "Starting…" : running ? "Tracing…" : "Trace year estimate";
  els.yearCancelBtn.hidden = !running;
  els.yearCancelBtn.classList.toggle("disabled-link", yearCancelling);
  els.yearCancelBtn.textContent = yearCancelling ? "Cancelling…" : "Cancel";

  if (!busy) {
    const fidelity = store.get("ui.fidelity");
    const nHeliostats = currentHeliostatCount(lastCtx);
    const nTimesteps = estimateYearTimesteps(yearFastMode, formHourStep, formMinElevationDeg);
    const estimateS = estimateDurationS(fidelity, nHeliostats, nTimesteps);
    const text = durationWarningText(estimateS, nHeliostats, nTimesteps);
    applyDurationHint(els.yearDurationHint, estimateS, text);
  } else {
    els.yearDurationHint.hidden = true;
  }

  els.yearProgressRow.hidden = !running;
  if (running) {
    const snap = yearJobSnapshot;
    const pct = snap.total ? Math.min(100, (100 * snap.done) / snap.total) : 0;
    els.yearProgressFill.style.width = pct.toFixed(1) + "%";
    const eta = fmtDuration(snap.eta_s);
    let text = `${snap.done} / ${snap.total} timesteps`;
    if (snap.detail) text += ` — ${snap.detail}`;
    text += ` — elapsed ${fmtDuration(snap.elapsed_s) || "0 s"}`;
    if (eta) text += ` — about ${eta} left`;
    if (yearCancelling) text += " — cancelling…";
    els.yearProgressText.textContent = text;
  }

  if (!running && yearJobSnapshot) {
    const elapsed = fmtDuration(yearJobSnapshot.elapsed_s) || `${yearJobSnapshot.elapsed_s} s`;
    let text = null;
    if (yearJobSnapshot.state === "done") {
      text = `Finished in ${elapsed}.`;
    } else if (yearJobSnapshot.state === "cancelled") {
      text = `Cancelled after ${yearJobSnapshot.done} of ${yearJobSnapshot.total} timesteps (${elapsed}).`;
    }
    els.yearStatusLine.hidden = !text;
    if (text) els.yearStatusLine.textContent = text;
  } else {
    els.yearStatusLine.hidden = true;
  }

  els.yearErr.hidden = !yearError;
  if (yearError) els.yearErr.textContent = yearError;
}

function paintYearResult() {
  const days = yearResult && yearResult.days;
  const haveResult = !!(days && days.length);

  // The frame only takes up room once it has something in it, or once a run
  // is under way and about to.
  els.yearFrame.hidden = !(haveResult || (yearJobSnapshot && yearJobSnapshot.state === "running"));

  if (yearResult && yearResult.plot_png) {
    els.yearImg.src = "data:image/png;base64," + yearResult.plot_png;
    els.yearImg.hidden = false;
    els.yearPlaceholder.hidden = true;
  } else {
    els.yearImg.hidden = true;
    els.yearPlaceholder.hidden = false;
    els.yearPlaceholder.textContent =
      yearJobSnapshot && yearJobSnapshot.state === "cancelled" && !haveResult
        ? "Cancelled before any date finished — nothing to plot."
        : "Run a year estimate to see collection across the year.";
  }

  if (haveResult) {
    els.yearTotal.hidden = false;
    const traced = yearResult.n_days_traced;
    const modeLabel = yearResult.fast_mode ? `fast mode, ${traced} of ${days.length} days traced` : `all ${days.length} days traced`;
    els.yearTotal.innerHTML = "";
    const strong = document.createElement("strong");
    strong.textContent = fmtMWh(yearResult.annual_energy_mwh);
    els.yearTotal.appendChild(document.createTextNode("Annual collection: "));
    els.yearTotal.appendChild(strong);
    // Spec §M.7: the DNI this specific run actually assumed (constant W/m^2
    // fixed, or the clear-sky model) -- was hardcoded to "clear-sky upper
    // bound" text before the site DNI control existed, which is only true
    // when this run's own dni_note says so.
    const dniLabel = yearResult.dni_note ? `DNI: ${yearResult.dni_note}` : "clear-sky upper bound";
    els.yearTotal.appendChild(
      document.createTextNode(` per year (${dniLabel}) — ${modeLabel}, ${yearResult.n_heliostats} heliostat(s)`)
    );
  } else {
    els.yearTotal.hidden = true;
  }

  els.yearDaysWrap.hidden = !haveResult;
  if (haveResult) renderYearDaysRows(days);

  const isReopened = !yearResultJobId && !!yearRunSavedName;
  els.yearReopenBanner.hidden = !isReopened;
  if (isReopened) els.yearReopenBanner.textContent = `Reopened saved run "${yearRunSavedName}" — no re-trace needed.`;

  const busy = yearRunSaving;
  els.yearRunSaveBtn.hidden = !haveResult || !!yearRunSavedName;
  els.yearRunSaveBtn.classList.toggle("disabled-link", busy);
  els.yearRunSaveBtn.textContent = busy && !yearRunSavedName ? "Saving…" : "Save this run";

  els.yearRunDiscardBtn.hidden = !yearRunSavedName;
  els.yearRunDiscardBtn.classList.toggle("disabled-link", busy);
  els.yearRunDiscardBtn.textContent = busy && yearRunSavedName ? "Discarding…" : "Discard this run";

  els.yearRunStatus.textContent = yearRunSavedName && !busy ? `Saved as "${yearRunSavedName}"` : "";
  els.yearRunErrEl.hidden = !yearRunError;
  if (yearRunError) els.yearRunErrEl.textContent = yearRunError;

  els.yearStaleChip.hidden = !yearResultIsStale();
}

// -- "Manage saved runs" overlay -----------------------------------------------

function manageRow(entry) {
  const row = document.createElement("div");
  row.className = "an-managerow";
  const label = document.createElement("div");
  label.className = "an-managerowlabel";
  const kind =
    entry.kind === "year"
      ? "Year estimate"
      : entry.kind === "day"
        ? "Day sweep"
        : entry.kind === "instant"
          ? "Traced instant"
          : "Run";
  const when = entry.saved_at ? entry.saved_at.slice(0, 16).replace("T", " ") : "";
  label.textContent = `${kind} — ${entry.name}`;
  const meta = document.createElement("div");
  meta.className = "an-managerowmeta";
  meta.textContent = `${entry.project_name ? entry.project_name : "no project"} · ${when} · ${fmtBytes(entry.size_bytes)}`;
  const actions = document.createElement("div");
  actions.className = "an-managerowactions";
  const openBtn = document.createElement("div");
  openBtn.className = "btn small";
  openBtn.textContent = "Open";
  openBtn.addEventListener("click", () => {
    openSavedRun(entry);
    closeManage();
  });
  const delBtn = document.createElement("div");
  delBtn.className = "btn small";
  delBtn.textContent = "Delete";
  delBtn.addEventListener("click", () => deleteManageEntry(entry.name));
  actions.appendChild(openBtn);
  actions.appendChild(delBtn);
  row.appendChild(label);
  row.appendChild(meta);
  row.appendChild(actions);
  return row;
}

function paintManageOverlay() {
  els.manageOverlay.hidden = !manageOpen;
  if (!manageOpen) return;
  els.manageBody.innerHTML = "";
  if (manageLoading) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "Loading…";
    els.manageBody.appendChild(p);
    return;
  }
  if (manageError) {
    const p = document.createElement("div");
    p.className = "fielderr";
    p.textContent = manageError;
    els.manageBody.appendChild(p);
  }
  if (!manageEntries.length) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "No saved runs yet.";
    els.manageBody.appendChild(p);
    return;
  }
  const totalBytes = manageEntries.reduce((sum, e) => sum + (e.size_bytes || 0), 0);
  const totalLine = document.createElement("p");
  totalLine.className = "hint";
  totalLine.textContent = `${manageEntries.length} saved run(s), ${fmtBytes(totalBytes)} total.`;
  els.manageBody.appendChild(totalLine);
  for (const entry of manageEntries) els.manageBody.appendChild(manageRow(entry));
}

// -- footprint "click to enlarge" overlay ------------------------------------
function paintFootprintOverlay() {
  if (els.footprintOverlay.hidden) return;
  const step = currentStepForAperture();
  const heliostat =
    drillHeliostatId != null ? currentFieldHeliostats().find((h) => h.id === drillHeliostatId) : null;
  // Same caption text as the inline drill-down head (paintDrillPanel), plus
  // the peak flux the small render's own trace already carries -- the big
  // render is a higher-dpi picture of the identical trace, so it is the
  // same number, not a second fetch's worth.
  els.footprintOverlayCaption.textContent =
    heliostat && step
      ? `H-${heliostat.id} footprint — ${activeSource === "instant" ? "traced instant" : fmtHHMM(step.hour)}` +
        (heliostatFootprintPeakKwM2 != null ? ` · peak ${fmtFlux(heliostatFootprintPeakKwM2)}` : "")
      : "Footprint";

  if (footprintOverlayLoading) {
    els.footprintOverlayImg.hidden = true;
    els.footprintOverlayPlaceholder.hidden = false;
    els.footprintOverlayPlaceholder.textContent = "Rendering a sharper view…";
    return;
  }
  if (footprintOverlayError) {
    els.footprintOverlayImg.hidden = true;
    els.footprintOverlayPlaceholder.hidden = false;
    els.footprintOverlayPlaceholder.textContent = footprintOverlayError;
    return;
  }
  if (footprintOverlayPngBase64) {
    els.footprintOverlayImg.src = "data:image/png;base64," + footprintOverlayPngBase64;
    els.footprintOverlayImg.hidden = false;
    els.footprintOverlayPlaceholder.hidden = true;
  } else {
    els.footprintOverlayImg.hidden = true;
    els.footprintOverlayPlaceholder.hidden = false;
    els.footprintOverlayPlaceholder.textContent = "Pick a heliostat to see its own footprint.";
  }
}

// §R: the "Traced instant" left-column card -- a one-line summary of
// whatever ui.traceResult (or a reopened saved run) currently is, or a
// "Trace this instant →" link to 3D View's Run control when nothing has
// been traced yet.
function paintInstantCard() {
  const result = currentInstantResult();
  if (!result) {
    els.instantSummary.textContent = "Trace this instant in 3D View →";
    els.instantSummary.classList.add("disabled-link");
    els.instantRunSaveBtn.hidden = true;
    els.instantRunDiscardBtn.hidden = true;
    els.instantRunStatus.textContent = "";
    els.instantRunErrEl.hidden = true;
    return;
  }
  els.instantSummary.classList.remove("disabled-link");
  const req = currentInstantRequest();
  const azEl =
    req && req.solar_az_deg != null && req.solar_el_deg != null
      ? `Az ${req.solar_az_deg.toFixed(1)}° El ${req.solar_el_deg.toFixed(1)}°`
      : "";
  const fidelityEntry = FIDELITY.find(([key]) => key === (req && req.mode));
  const modeLabel = fidelityEntry ? fidelityEntry[1] : req && req.mode;
  const nH = result.n_heliostats != null ? result.n_heliostats : (result.heliostats || []).length || 1;
  const parts = [azEl, modeLabel, `${nH} heliostat${nH === 1 ? "" : "s"}`].filter(Boolean);
  const prefix = activeSource === "instant" ? "● Viewing: " : "";
  const suffix = activeSource === "instant" ? "" : " — click to view";
  els.instantSummary.textContent = prefix + parts.join(" · ") + suffix;

  const busy = instantSaving;
  els.instantRunSaveBtn.hidden = !!instantSavedName;
  els.instantRunSaveBtn.classList.toggle("disabled-link", busy);
  els.instantRunSaveBtn.textContent = busy && !instantSavedName ? "Saving…" : "Save this run";
  els.instantRunDiscardBtn.hidden = !instantSavedName;
  els.instantRunDiscardBtn.classList.toggle("disabled-link", busy);
  els.instantRunDiscardBtn.textContent = busy && instantSavedName ? "Discarding…" : "Discard this run";
  els.instantRunStatus.textContent = instantSavedName && !busy ? `Saved as "${instantSavedName}"` : "";
  els.instantRunErrEl.hidden = !instantRunError;
  if (instantRunError) els.instantRunErrEl.textContent = instantRunError;
}

// §R: the "Viewing" selector -- which of the three peer sources drives the
// right column's shared instruments right now.
function paintSourceSelector() {
  els.sourceDayBtn.classList.toggle("active", activeSource === "day");
  els.sourceYearBtn.classList.toggle("active", activeSource === "year");
  els.sourceInstantBtn.classList.toggle("active", activeSource === "instant");
}

function paint() {
  const doc = store.get("doc");
  syncInstantSourceIfNeeded();
  syncProjectRuns();
  paintSubject(doc, lastCtx);
  paintSavedRunsBar();
  paintSweepControls();
  paintDayRunControls();
  paintInstantCard();
  paintSourceSelector();
  paintEnergyPanel();
  paintTimestepsTable();
  paintFluxPanel();
  paintDrillPanel();
  paintAperturePanel();
  paintYearControls();
  paintYearResult();
  paintManageOverlay();
  paintFootprintOverlay();

  // Stale results stay readable -- they are still the truth about the setup
  // they were measured on, and a single timestep's map is still worth
  // looking at -- but they say plainly that they no longer describe the
  // project as it now stands.
  const stale = resultIsStale();
  els.staleChip.hidden = !stale;
  els.energyPanel.classList.toggle("an-stale", stale);
  els.tsPanel.classList.toggle("an-stale", stale);
  els.fluxPanel.classList.toggle("an-stale", stale);
  els.drillPanel.classList.toggle("an-stale", stale);
  els.aperturePanel.classList.toggle("an-stale", stale);
  els.instantPanel.classList.toggle("an-stale", stale && activeSource === "instant");
}

let lastCtx = null;

export function render(container, ctx) {
  if (!built) build(container);
  lastContainer = container;
  lastCtx = ctx;
  paint();
}
