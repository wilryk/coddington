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
  buildDayRequest,
  buildTraceRequest,
  buildYearRequest,
  dayExportUrl,
  dayFluxUrl,
  dayFluxFeaCsvUrl,
  deleteLibraryEntry,
  getDayFluxGrid,
  getDayResult,
  getDayStatus,
  getLibrary,
  getLibraryEntry,
  getYearResult,
  getYearStatus,
  postDayCancel,
  postDayStart,
  postFieldTrace,
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
// _secondary_payload), when there is one -- only ever set from a live
// re-trace (the cache/fetch branch of scheduleFluxFetch below), because
// that is the only path whose response body this module actually sees. A
// step served from the sweep's own stored PNG (fluxSrcUrl) or a reopened
// saved run's flux_pngs never carries one -- see scheduleFluxFetch's own
// comment on each of those branches.
let fluxSecondary = null;
let fluxLoading = false;
let fluxError = null;
let fluxTimer = null;
let fluxController = null;

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

function resultIsStale() {
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
    fluxError = png ? null : "This saved run kept no irradiance map for that timestep.";
    fluxLoading = false;
    paintIfVisible();
    return;
  }
  // The sweep already traced and rendered this timestep -- serve its own
  // map straight from the server, instantly, instead of re-tracing it.
  // /api/day/* was not extended for spec §C (docs/secondary-irradiance-
  // plan.md's build order stopped at the single/field trace endpoints), so
  // this stored PNG carries no secondary data either -- same as above.
  if (step.has_flux_map && resultJobId != null) {
    fluxPngBase64 = null;
    fluxSrcUrl = dayFluxUrl(resultJobId, selectedStepIndex);
    fluxPeakKwM2 = null; // the row's own peak_flux_kw_m2 covers the caption
    fluxSecondary = null;
    fluxError = null;
    fluxLoading = false;
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
        fluxError = fluxPngBase64 ? null : "No flux map came back for this timestep.";
        if (fluxPngBase64) {
          fluxCache.set(cacheKey, { png: fluxPngBase64, peak: fluxPeakKwM2, secondary: fluxSecondary });
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
  drillHeliostatId = id;
  scheduleHeliostatFootprintFetch();
  paintIfVisible();
}

// Same body-shaping idiom as fluxRequestFor above, but for ONE named
// heliostat rather than the field: drop layout/exclude_ids, set that
// heliostat's own position, keep the timestep's own stored sun angles.
function heliostatFootprintRequestFor(step, heliostat) {
  const base = sweepRequest
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
  const cacheKey = `${drillHeliostatId}:${selectedStepIndex}`;
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

function selectStep(i) {
  if (selectedStepIndex === i) return;
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
      } else {
        if (yearPollTimer) {
          clearTimeout(yearPollTimer);
          yearPollTimer = null;
        }
        resetYearRunState();
        yearResult = document.result;
        yearResultJobId = null;
        yearSweepRequest = document.request;
        yearSweepPhysicsKey = physicsKey(document.request);
        yearRunSavedName = entry.name;
        yearFastMode = !document.request || document.request.fast_mode !== false;
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

// §M.4 explicitly scopes the aperture to flat receivers first ("curved
// later if wanted") -- a frustum's bin area varies with position
// (FrustumReceiver.bin_areas_m2 in geometry/receiver.py), which the
// uniform-bin-area math below does not account for.
function receiverKindFor(doc) {
  if (doc.optics === "prime_focus") {
    return (doc.opticsParams.prime_focus || {}).receiver_type || "flat";
  }
  return "flat"; // axicon/cassegrain always target a flat receiver window
}

function apertureReceiverIsFlat(doc) {
  return receiverKindFor(doc) === "flat";
}

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

function apertureDefaultCenter(grid, step) {
  if (step && Array.isArray(step.centroid_mm) && step.centroid_mm.length === 2) {
    return { u: step.centroid_mm[0], v: step.centroid_mm[1] };
  }
  return { u: (grid.u_min_mm + grid.u_max_mm) / 2, v: (grid.v_min_mm + grid.v_max_mm) / 2 };
}

function apertureDefaultRadiusMm(grid, step) {
  const halfU = Math.abs(grid.u_max_mm - grid.u_min_mm) / 2;
  const halfV = Math.abs(grid.v_max_mm - grid.v_min_mm) / 2;
  const cap = 0.9 * Math.min(halfU, halfV);
  // A couple of RMS spot radii is a common, physically-motivated "captures
  // most of a roughly Gaussian-like spot" default -- step.rms_radius_mm is
  // already computed and stored (heliostat.web.app's _cone_metrics), so
  // this reads it rather than guessing a bare fraction of the grid.
  const rms = step && Number.isFinite(step.rms_radius_mm) ? step.rms_radius_mm : null;
  const guess = rms != null ? 2.0 * rms : cap * 0.4;
  return Math.max(1.0, Math.min(guess, cap));
}

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

function clampToGridAxis(grid, value, axis) {
  const lo = axis === "u" ? grid.u_min_mm : grid.v_min_mm;
  const hi = axis === "u" ? grid.u_max_mm : grid.v_max_mm;
  return Math.max(lo, Math.min(hi, value));
}

// grid.values are §M.3's own kW/m^2 convention (heliostat.web.app's
// _flux_grid_payload, row-major, row 0 = v_min_mm -- the bottom of the map,
// same as _render_flux_png's origin="lower") -- scaled back to W/m^2 here
// so power comes out in watts. A bin counts as "inside" when its CENTER
// lies within radiusMm of the aperture center (the standard encircled-power
// discretization); average flux divides by the aperture's own ideal
// circular area (pi * r^2), not the discretized sum of included bin areas,
// matching mockup M17's own worked example.
function apertureMetrics(grid, centerUMm, centerVMm, radiusMm) {
  const nU = grid.n_u;
  const nV = grid.n_v;
  const duMm = (grid.u_max_mm - grid.u_min_mm) / nU;
  const dvMm = (grid.v_max_mm - grid.v_min_mm) / nV;
  const binAreaM2 = (duMm / 1000) * (dvMm / 1000);
  const r2 = radiusMm * radiusMm;
  let powerW = 0;
  for (let row = 0; row < nV; row++) {
    const vMid = grid.v_min_mm + (row + 0.5) * dvMm;
    const dv = vMid - centerVMm;
    const dv2 = dv * dv;
    if (dv2 > r2) continue; // whole row is out of range -- skip its n_u work
    const rowBase = row * nU;
    for (let col = 0; col < nU; col++) {
      const uMid = grid.u_min_mm + (col + 0.5) * duMm;
      const du = uMid - centerUMm;
      if (du * du + dv2 > r2) continue;
      const kwM2 = grid.values[rowBase + col];
      if (kwM2 != null) powerW += kwM2 * 1000 * binAreaM2;
    }
  }
  const radiusM = radiusMm / 1000;
  const areaM2 = Math.PI * radiusM * radiusM;
  const avgFluxWM2 = areaM2 > 0 ? powerW / areaM2 : 0;
  return { powerW, avgFluxWM2 };
}

// Power vs. radius, mockup M17's encircled-power curve -- nSamples evenly
// spaced radii from 0 to maxRadiusMm.
function apertureCurve(grid, centerUMm, centerVMm, maxRadiusMm, nSamples) {
  const pts = [];
  for (let i = 0; i <= nSamples; i++) {
    const r = (maxRadiusMm * i) / nSamples;
    pts.push({ r, powerW: apertureMetrics(grid, centerUMm, centerVMm, r).powerW });
  }
  return pts;
}

function buildApertureSnapshotForSave() {
  // Resaving an already-reopened (frozen) run keeps its own annotation
  // rather than trying to recompute one from a grid that, for a reopened
  // run, does not exist (see openSavedRun/scheduleApertureGridFetch).
  if (reopenedAperture) return reopenedAperture;
  if (!apertureGrid || selectedStepIndex == null) return null;
  const step = currentStepForAperture();
  if (!step) return null;
  const grid = apertureGrid;
  const center = currentApertureCenterMm(grid, step);
  const radius = currentApertureRadiusMm(grid, step);
  const { powerW, avgFluxWM2 } = apertureMetrics(grid, center.u, center.v, radius);
  const collectedW = step.power_w;
  const dni = step.dni_w_m2;
  return {
    step_index: selectedStepIndex,
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
const APERTURE_MAGMA_STOPS = [
  [0.0, 0, 0, 4],
  [0.2, 59, 15, 112],
  [0.4, 140, 41, 129],
  [0.6, 222, 73, 104],
  [0.8, 254, 159, 109],
  [1.0, 252, 253, 191],
];

function magmaColor(t) {
  const c = Math.max(0, Math.min(1, t));
  for (let i = 1; i < APERTURE_MAGMA_STOPS.length; i++) {
    const [t0, r0, g0, b0] = APERTURE_MAGMA_STOPS[i - 1];
    const [t1, r1, g1, b1] = APERTURE_MAGMA_STOPS[i];
    if (c <= t1 || i === APERTURE_MAGMA_STOPS.length - 1) {
      const f = t1 > t0 ? (c - t0) / (t1 - t0) : 0;
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(0,0,4)";
}

function sizeApertureCanvas(canvas, grid, targetWidth) {
  const uExtent = Math.max(1e-6, grid.u_max_mm - grid.u_min_mm);
  const vExtent = Math.max(1e-6, grid.v_max_mm - grid.v_min_mm);
  const pxPerMm = targetWidth / uExtent;
  canvas.width = Math.round(targetWidth);
  canvas.height = Math.max(60, Math.round(vExtent * pxPerMm));
  return pxPerMm;
}

function apertureDataToCanvas(grid, canvas, uMm, vMm) {
  const x = ((uMm - grid.u_min_mm) / (grid.u_max_mm - grid.u_min_mm)) * canvas.width;
  // v (north/up in the data) grows upward; canvas y grows downward -- flip,
  // matching _render_flux_png's own origin="lower".
  const y = (1 - (vMm - grid.v_min_mm) / (grid.v_max_mm - grid.v_min_mm)) * canvas.height;
  return [x, y];
}

function apertureCanvasToData(grid, canvas, x, y) {
  const uMm = grid.u_min_mm + (x / canvas.width) * (grid.u_max_mm - grid.u_min_mm);
  const vMm = grid.v_min_mm + (1 - y / canvas.height) * (grid.v_max_mm - grid.v_min_mm);
  return [uMm, vMm];
}

function paintApertureCanvas(canvas, grid, centerUMm, centerVMm, radiusMm) {
  const pxPerMm = sizeApertureCanvas(canvas, grid, 380);
  const ctx2d = canvas.getContext("2d");
  const nU = grid.n_u;
  const nV = grid.n_v;
  const cellW = canvas.width / nU;
  const cellH = canvas.height / nV;
  let maxKw = 0;
  for (const v of grid.values) if (v != null && v > maxKw) maxKw = v;
  for (let row = 0; row < nV; row++) {
    // canvas row 0 is the TOP of the picture; grid row 0 is v_min_mm (the
    // bottom of the map) -- flip so the picture reads right-side up.
    const canvasRow = nV - 1 - row;
    const rowBase = row * nU;
    for (let col = 0; col < nU; col++) {
      const kw = grid.values[rowBase + col];
      const t = maxKw > 0 && kw != null ? kw / maxKw : 0;
      ctx2d.fillStyle = magmaColor(t);
      ctx2d.fillRect(col * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  const [cx, cy] = apertureDataToCanvas(grid, canvas, centerUMm, centerVMm);
  const rPx = radiusMm * pxPerMm;
  ctx2d.save();
  ctx2d.setLineDash([6, 4]);
  ctx2d.lineWidth = 2;
  ctx2d.strokeStyle = "#ffffff";
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, rPx, 0, Math.PI * 2);
  ctx2d.stroke();
  ctx2d.lineWidth = 1;
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.stroke();
  ctx2d.restore();

  // Resize handle: a small square at the circle's east edge (mockup M17).
  const hx = cx + rPx;
  ctx2d.fillStyle = "#ffffff";
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.lineWidth = 1.3;
  ctx2d.fillRect(hx - 5, cy - 5, 10, 10);
  ctx2d.strokeRect(hx - 5, cy - 5, 10, 10);

  // Compass rider (§M) -- the aperture is scoped to flat receivers (see
  // apertureReceiverIsFlat), so this canvas only ever needs the flat-window
  // convention: u = east/west, v = north/south, no rotation.
  ctx2d.fillStyle = "rgba(255,255,255,0.85)";
  ctx2d.font = "10px sans-serif";
  ctx2d.textAlign = "center";
  ctx2d.fillText("N", canvas.width / 2, 12);
  ctx2d.fillText("S", canvas.width / 2, canvas.height - 5);
  ctx2d.textAlign = "left";
  ctx2d.fillText("W", 4, canvas.height / 2 + 3);
  ctx2d.textAlign = "right";
  ctx2d.fillText("E", canvas.width - 4, canvas.height / 2 + 3);
}

function paintApertureCurve(canvas, curve, currentRadiusMm, currentPowerW) {
  canvas.width = 380;
  canvas.height = 160;
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  const padL = 46;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;
  const maxR = curve.length ? curve[curve.length - 1].r : 1;
  let maxP = 1e-9;
  for (const p of curve) if (p.powerW > maxP) maxP = p.powerW;
  const xOf = (r) => padL + (r / maxR) * plotW;
  const yOf = (p) => padT + plotH - (p / maxP) * plotH;

  ctx2d.strokeStyle = "#c7cdd6";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(padL, padT + plotH);
  ctx2d.lineTo(padL + plotW, padT + plotH);
  ctx2d.moveTo(padL, padT);
  ctx2d.lineTo(padL, padT + plotH);
  ctx2d.stroke();

  ctx2d.strokeStyle = "#45739e";
  ctx2d.lineWidth = 2;
  ctx2d.beginPath();
  curve.forEach((p, i) => {
    const x = xOf(p.r);
    const y = yOf(p.powerW);
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  });
  ctx2d.stroke();

  const markX = xOf(currentRadiusMm);
  const markY = yOf(currentPowerW);
  ctx2d.setLineDash([4, 4]);
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(markX, padT + plotH);
  ctx2d.lineTo(markX, markY);
  ctx2d.lineTo(padL, markY);
  ctx2d.stroke();
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = "#0b5fd0";
  ctx2d.beginPath();
  ctx2d.arc(markX, markY, 3.5, 0, Math.PI * 2);
  ctx2d.fill();

  ctx2d.fillStyle = "#64748b";
  ctx2d.font = "9.5px sans-serif";
  ctx2d.textAlign = "left";
  ctx2d.fillText("0", padL - 4, padT + plotH + 14);
  ctx2d.textAlign = "right";
  ctx2d.fillText((maxR / 1000).toFixed(1) + " m", padL + plotW, padT + plotH + 14);
  ctx2d.textAlign = "left";
  ctx2d.fillText(fmtPower(maxP), 2, padT + 8);
}

// Pointer-drag handling for the aperture canvas: click near the resize
// handle to change the radius, click anywhere else inside the circle to
// move it, click outside it to do nothing (no "click to place" -- the
// circle always has a defined default position).
function apertureCanvasEventPoint(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
}

function apertureHandlePointerDown(e) {
  if (!apertureGrid) return;
  const canvas = e.currentTarget;
  const step = currentStepForAperture();
  const grid = apertureGrid;
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
  if (!apertureDrag || !apertureGrid) return;
  const canvas = e.currentTarget;
  const grid = apertureGrid;
  const step = currentStepForAperture();
  const [x, y] = apertureCanvasEventPoint(canvas, e);
  const [uMm, vMm] = apertureCanvasToData(grid, canvas, x, y);
  if (apertureDrag.mode === "move") {
    apertureCenterUMm = clampToGridAxis(grid, uMm, "u");
    apertureCenterVMm = clampToGridAxis(grid, vMm, "v");
  } else {
    const center = currentApertureCenterMm(grid, step);
    const rMm = Math.hypot(uMm - center.u, vMm - center.v);
    const halfU = Math.abs(grid.u_max_mm - grid.u_min_mm) / 2;
    const halfV = Math.abs(grid.v_max_mm - grid.v_min_mm) / 2;
    apertureRadiusMm = Math.max(1, Math.min(rMm, 1.5 * Math.max(halfU, halfV)));
  }
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
  subjLink.textContent = "change in Workspace →";
  subjLink.addEventListener("click", (e) => {
    e.preventDefault();
    store.set("ui.tab", "workspace");
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
  const content = document.createElement("div");
  content.className = "tabcontent";

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
    fidelityBtns[key] = segButton(fidelitySeg, label, key === store.get("ui.fidelity"), () => {
      store.set("ui.fidelity", key);
    });
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
  const yearFastBtn = segButton(yearFastSeg, "Fast (7 traced)", yearFastMode, () => {
    yearFastMode = true;
    paintIfVisible();
  });
  const yearAllBtn = segButton(yearFastSeg, "All 12 traced", !yearFastMode, () => {
    yearFastMode = false;
    paintIfVisible();
  });
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

  const yearDniNote = document.createElement("div");
  yearDniNote.className = "hint";
  yearDniNote.textContent =
    "Assumes clear-sky DNI (no clouds) — a cloud-free upper bound on annual collection, not a weather-corrected forecast.";
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
  fluxHead.appendChild(fluxSurfaceSeg);
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
  const fluxPlaceholder = document.createElement("p");
  fluxPlaceholder.className = "placeholder";
  fluxPlaceholder.textContent = "Click a timestep to render its irradiance map.";
  fluxFrame.appendChild(fluxImg);
  fluxFrame.appendChild(fluxSecondaryCanvas);
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

  fluxPanel.appendChild(fluxMapBody);

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
  fluxPanel.appendChild(fluxFeaCsv);
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

  // -- analysis aperture (docs/ui-spec-v0.2.md §M.4) ------------------------
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

  const apertureReadout = document.createElement("div");
  apertureReadout.className = "readout";
  const apertureReadoutH3 = document.createElement("h3");
  apertureReadoutH3.textContent = "Aperture readout";
  apertureReadout.appendChild(apertureReadoutH3);

  function apRow(label) {
    const row = document.createElement("div");
    row.className = "rmetric";
    const lbl = document.createElement("div");
    lbl.className = "rlbl";
    lbl.textContent = label;
    const num = document.createElement("div");
    num.className = "rnum";
    row.appendChild(lbl);
    row.appendChild(num);
    apertureReadout.appendChild(row);
    return num;
  }
  const apRadius = apRow("aperture radius");
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
    "Draggable and resizable on the map above; saves with the run as an annotation. Post-processing only -- the trace, intercept and collected totals never change.";
  apertureReadout.appendChild(apHint);

  apertureBody.appendChild(apertureCanvasWrap);
  apertureBody.appendChild(apertureReadout);
  aperturePanel.appendChild(apertureBody);

  const apertureCurveFrame = document.createElement("div");
  apertureCurveFrame.className = "an-aperturecurveframe";
  const apertureCurveH3 = document.createElement("div");
  apertureCurveH3.className = "an-drillhead";
  apertureCurveH3.textContent = "Encircled power vs. aperture radius";
  const apertureCurveCanvas = document.createElement("canvas");
  apertureCurveCanvas.className = "an-aperturecurvecanvas";
  apertureCurveFrame.appendChild(apertureCurveH3);
  apertureCurveFrame.appendChild(apertureCurveCanvas);
  aperturePanel.appendChild(apertureCurveFrame);

  right.appendChild(aperturePanel);

  content.appendChild(left);
  content.appendChild(right);

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

  container.appendChild(subject);
  container.appendChild(savedRunsBar);
  container.appendChild(content);
  container.appendChild(manageOverlay);

  els = {
    staleChip,
    energyPanel,
    tsPanel,
    fluxPanel,
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
    fluxSecondaryCanvas,
    fluxSecondaryReadout,
    secIncidentRow,
    secAbsorbedRow,
    secPeakAbsorbedRow,
    fluxSecFidelity,
    fluxImg,
    fluxPlaceholder,
    fluxCaption,
    fluxCompass,
    fluxFeaCsv,
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
    apRadius,
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
    yearReopenBanner,
    yearRunSaveBtn,
    yearRunDiscardBtn,
    yearRunStatus,
    yearRunErrEl,
    manageOverlay,
    manageBody,
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
    return "Exact fidelity — Monte Carlo histograms every ray that actually struck the secondary.";
  }
  return "Coarse fidelity — this cone mode deposits each mirror's flux at its own chief ray's secondary hit, not a full footprint. Switch to Monte Carlo for an exact per-ray map.";
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

// Paints app.py's _flux_grid_payload straight onto a 2D canvas -- there is
// no server-rendered PNG for the secondary map (only the opt-in raw grid,
// spec §C), unlike the receiver's own flux_png. `values` is row-major, row
// 0 = v_min (the bottom of the map, matplotlib's own origin="lower"
// convention _render_flux_png uses) -- canvas row 0 is its TOP, so row 0 of
// `values` is drawn into the canvas's LAST row (same flip as scene3d.js's
// fluxGridTexture).
function paintSecondaryFluxCanvas(canvas, grid) {
  const { n_u, n_v, values } = grid;
  let max = 0;
  for (const v of values) if (v != null && v > max) max = v;
  canvas.width = n_u;
  canvas.height = n_v;
  const ctx2d = canvas.getContext("2d");
  const img = ctx2d.createImageData(n_u, n_v);
  for (let row = 0; row < n_v; row++) {
    const canvasRow = n_v - 1 - row;
    for (let col = 0; col < n_u; col++) {
      const val = values[row * n_u + col];
      const [r, g, b] = secondaryFluxMagmaColor(max > 0 && val != null ? val / max : 0);
      const idx = (canvasRow * n_u + col) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx2d.putImageData(img, 0, 0);
}

// Repaints the Receiver | Secondary selector against whether THIS step's
// currently-loaded map actually carries a secondary block, and returns
// whether the secondary map should be the one shown. Secondary stays
// disabled (with a tooltip explaining why) for prime_focus, for a step
// still loading/erroring, and for the common case of a sweep-stored PNG or
// a reopened saved run -- neither carries the raw grid the client-side
// paint needs (see scheduleFluxFetch's own comments on those branches).
function paintFluxSurfaceSelector() {
  const doc = store.get("doc");
  const hasSecondaryOptics = doc.optics === "axicon" || doc.optics === "cassegrain";
  const available = !!(fluxSecondary && fluxSecondary.flux_grid);
  els.fluxSurfaceSecondaryBtn.classList.toggle("disabled", !available);
  let tip = "";
  if (!available) {
    if (!hasSecondaryOptics) tip = "Only axicon and Cassegrain layouts have a secondary flux map.";
    else if (fluxSrcUrl || reopenedDayFluxPngs) {
      tip = "This stored sweep step carries no secondary flux data — only a freshly-traced map does.";
    } else tip = "This trace carried no secondary flux map.";
  }
  els.fluxSurfaceSecondaryBtn.title = tip;
  const requested = store.get("ui.fluxSurface");
  const showSecondary = requested === "secondary" && available;
  els.fluxSurfaceReceiverBtn.classList.toggle("active", !showSecondary);
  els.fluxSurfaceSecondaryBtn.classList.toggle("active", showSecondary);
  return showSecondary;
}

function paintFluxPanel() {
  const steps = dayResult && dayResult.steps;
  const step = steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;
  const showSecondary = paintFluxSurfaceSelector();

  function showPlaceholder(text) {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = text;
    els.fluxCaption.textContent = "";
    els.fluxCompass.textContent = "";
    els.fluxFeaCsv.hidden = true;
  }

  if (!step) return showPlaceholder("Click a timestep to render its irradiance map.");
  if (fluxLoading) return showPlaceholder("Rendering…");
  if (fluxError) return showPlaceholder(fluxError);
  if (!(fluxSrcUrl || fluxPngBase64)) return showPlaceholder("Click a timestep to render its irradiance map.");

  els.fluxPlaceholder.hidden = true;

  if (showSecondary) {
    els.fluxImg.hidden = true;
    els.fluxSecondaryCanvas.hidden = false;
    paintSecondaryFluxCanvas(els.fluxSecondaryCanvas, fluxSecondary.flux_grid);
    els.fluxCaption.textContent = `incident flux on secondary, kW/m² · same colormap & units as the receiver map · peak ${fmtFlux(fluxSecondary.peak_flux_kw_m2)}`;
    // No compass convention for the secondary (§C's u/v is arc length/
    // radius, not north-seam azimuth) -- receiver-only, see compassCaptionFor.
    els.fluxCompass.textContent = "";
    els.fluxSecondaryReadout.hidden = false;
    els.secIncidentRow.num.textContent = fmtPower(fluxSecondary.power_w);
    const rPct = (fluxSecondary.secondary_reflectance * 100).toFixed(1);
    els.secAbsorbedRow.lbl.textContent = `absorbed (R = ${rPct} %)`;
    els.secAbsorbedRow.num.textContent = fmtPower(fluxSecondary.absorbed_power_w);
    els.secPeakAbsorbedRow.num.textContent = fmtFlux(fluxSecondary.peak_absorbed_kw_m2);
    els.fluxSecFidelity.textContent = secondaryFidelityNote(fluxSecondary.fidelity);
    // No secondary CSV export wired here -- the day-sweep endpoints
    // (§D's dayFluxFeaCsvUrl) were not extended for spec §C; the run bar's
    // own single-trace export (js/main.js) is the supported path for now.
    els.fluxFeaCsv.hidden = true;
  } else {
    els.fluxImg.src = fluxSrcUrl || "data:image/png;base64," + fluxPngBase64;
    els.fluxImg.hidden = false;
    els.fluxSecondaryCanvas.hidden = true;
    els.fluxSecondaryReadout.hidden = true;
    els.fluxCaption.textContent = `${fmtHHMM(step.hour)} solar · peak ${fmtFlux(fluxPeakKwM2 != null ? fluxPeakKwM2 : step.peak_flux_kw_m2)}`;
    // Compass rider (§M) -- see compassCaptionFor.
    els.fluxCompass.textContent = compassCaptionFor(store.get("doc"));
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
  const steps = dayResult && dayResult.steps;
  const haveSteps = !!(steps && steps.length);
  const isField = doc.field.mode === "field";
  els.drillPanel.hidden = !isField || !haveSteps;
  if (els.drillPanel.hidden) return;

  paintDrillMiniPlan(els.drillPlanSvg);

  const step = currentStepForAperture();
  const heliostat =
    drillHeliostatId != null ? currentFieldHeliostats().find((h) => h.id === drillHeliostatId) : null;
  els.drillIdInput.value = drillHeliostatId != null ? String(drillHeliostatId) : "";
  els.drillFootHead.textContent =
    heliostat && step ? `H-${heliostat.id} footprint — ${fmtHHMM(step.hour)}` : "Footprint";

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

function renderApertureReadout(data, frozen) {
  els.apRadius.textContent = data.radius_mm != null ? (data.radius_mm / 1000).toFixed(2) + " m" : "—";
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
    return;
  }

  const step = currentStepForAperture();

  // A reopened run's frozen annotation, on its own timestep -- shown
  // verbatim, never recomputed (mockup M17's own checknote).
  if (reopenedAperture && selectedStepIndex === reopenedAperture.step_index) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = true;
    els.apertureBody.hidden = false;
    els.apertureCanvasWrap.hidden = true; // no live grid to draw the circle against
    els.apertureReadout.hidden = false;
    els.apertureCurveFrame.hidden = true;
    renderApertureReadout(reopenedAperture, true);
    return;
  }
  if (reopenedAperture) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = false;
    els.apertureMsg.textContent =
      "This run's saved aperture belongs to a different timestep -- select it to see the same circle and readout.";
    els.apertureBody.hidden = true;
    els.apertureCurveFrame.hidden = true;
    return;
  }

  // Gate on the grid itself, not just the conditions that make one worth
  // fetching -- scheduleApertureGridFetch's own fetch is still in flight
  // the first time paint() runs right after a step becomes selected (it
  // sets apertureGridLoading and repaints before the network call lands),
  // so step.has_flux_map/resultJobId being fine is not enough to know
  // apertureGrid itself is populated yet.
  if (!step || !step.has_flux_map || resultJobId == null || !apertureGrid) {
    els.aperturePanel.hidden = false;
    els.apertureMsg.hidden = false;
    els.apertureMsg.textContent = apertureGridLoading
      ? "Loading aperture grid…"
      : apertureGridError || "Select a timestep with a stored irradiance map to use the aperture.";
    els.apertureBody.hidden = true;
    els.apertureCurveFrame.hidden = true;
    return;
  }

  els.aperturePanel.hidden = false;
  els.apertureMsg.hidden = true;
  els.apertureBody.hidden = false;
  els.apertureCanvasWrap.hidden = false;
  els.apertureReadout.hidden = false;
  els.apertureCurveFrame.hidden = false;

  const grid = apertureGrid;
  const center = currentApertureCenterMm(grid, step);
  const radius = currentApertureRadiusMm(grid, step);
  paintApertureCanvas(els.apertureCanvas, grid, center.u, center.v, radius);

  const { powerW, avgFluxWM2 } = apertureMetrics(grid, center.u, center.v, radius);
  const collectedW = step.power_w;
  const dni = step.dni_w_m2;
  renderApertureReadout(
    {
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
  paintApertureCurve(els.apertureCurveCanvas, curve, radius, powerW);
}

// -- saved runs (docs/ui-spec.md 4) -------------------------------------------

function runRowLabel(entry) {
  const kind = entry.kind === "year" ? "Year estimate" : "Day sweep";
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
    els.yearTotal.appendChild(
      document.createTextNode(` per year (clear-sky upper bound) — ${modeLabel}, ${yearResult.n_heliostats} heliostat(s)`)
    );
  } else {
    els.yearTotal.hidden = true;
  }

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
  const kind = entry.kind === "year" ? "Year estimate" : entry.kind === "day" ? "Day sweep" : "Run";
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

function paint() {
  const doc = store.get("doc");
  syncProjectRuns();
  paintSubject(doc, lastCtx);
  paintSavedRunsBar();
  paintSweepControls();
  paintDayRunControls();
  paintEnergyPanel();
  paintTimestepsTable();
  paintFluxPanel();
  paintDrillPanel();
  paintAperturePanel();
  paintYearControls();
  paintYearResult();
  paintManageOverlay();

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
}

let lastCtx = null;

export function render(container, ctx) {
  if (!built) build(container);
  lastContainer = container;
  lastCtx = ctx;
  paint();
}
