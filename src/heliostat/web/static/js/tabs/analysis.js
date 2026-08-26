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
  deleteLibraryEntry,
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
let fluxLoading = false;
let fluxError = null;
let fluxTimer = null;
let fluxController = null;

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

// Only worth a line once the rough estimate clears a couple of minutes --
// anything shorter is not worth interrupting the flow to warn about.
const DURATION_WARNING_THRESHOLD_S = 120;

function durationWarningText(estimateS, nHeliostats, nTimesteps) {
  if (!(estimateS > DURATION_WARNING_THRESHOLD_S)) return null;
  const dur = fmtDuration(estimateS) || `${Math.round(estimateS)} s`;
  return (
    `Rough estimate: about ${dur} for ${nHeliostats} heliostat(s) x ${nTimesteps} timesteps. ` +
    "This is a rough guess, not a promise -- actual time depends on hardware and settings."
  );
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
    fluxError = null;
    fluxLoading = false;
    paintIfVisible();
    return;
  }
  // A reopened saved run has no live job to serve a map from -- its maps
  // travel with the saved document itself instead (see SavedRunDocument).
  if (reopenedDayFluxPngs) {
    const png = reopenedDayFluxPngs[String(selectedStepIndex)];
    fluxPngBase64 = png || null;
    fluxSrcUrl = null;
    fluxPeakKwM2 = null;
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
        fluxError = fluxPngBase64 ? null : "No flux map came back for this timestep.";
        if (fluxPngBase64) fluxCache.set(cacheKey, { png: fluxPngBase64, peak: fluxPeakKwM2 });
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

function selectStep(i) {
  if (selectedStepIndex === i) return;
  selectedStepIndex = i;
  scheduleFluxFetch();
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
        dayRunSavedName = entry.name;
        if (dayResult && dayResult.steps && dayResult.steps.length) {
          selectedStepIndex = pickDefaultStepIndex(dayResult.steps);
          scheduleFluxFetch();
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
  startBtn.textContent = "Start day sweep";
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
  yearStartBtn.textContent = "Run year estimate";
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
  const fluxH2 = document.createElement("h2");
  fluxH2.textContent = "Irradiance map";
  fluxPanel.appendChild(fluxH2);
  const fluxFrame = document.createElement("div");
  fluxFrame.className = "frame";
  const fluxImg = document.createElement("img");
  fluxImg.alt = "Irradiance map for the selected timestep";
  fluxImg.hidden = true;
  const fluxPlaceholder = document.createElement("p");
  fluxPlaceholder.className = "placeholder";
  fluxPlaceholder.textContent = "Click a timestep to render its irradiance map.";
  fluxFrame.appendChild(fluxImg);
  fluxFrame.appendChild(fluxPlaceholder);
  fluxPanel.appendChild(fluxFrame);
  const fluxCaption = document.createElement("div");
  fluxCaption.className = "caption";
  fluxPanel.appendChild(fluxCaption);
  right.appendChild(fluxPanel);

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
    fluxImg,
    fluxPlaceholder,
    fluxCaption,
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
  els.startBtn.textContent = starting ? "Starting…" : running ? "Running…" : "Start day sweep";
  els.cancelBtn.hidden = !running;
  els.cancelBtn.classList.toggle("disabled-link", cancelling);
  els.cancelBtn.textContent = cancelling ? "Cancelling…" : "Cancel";

  if (!busy) {
    const nHeliostats = currentHeliostatCount(lastCtx);
    const nTimesteps = estimateDayTimesteps(formHourStep, formMinElevationDeg);
    const estimateS = estimateDurationS(fidelity, nHeliostats, nTimesteps);
    const text = durationWarningText(estimateS, nHeliostats, nTimesteps);
    els.durationHint.hidden = !text;
    if (text) els.durationHint.textContent = text;
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

function paintFluxPanel() {
  const steps = dayResult && dayResult.steps;
  const step = steps && selectedStepIndex != null ? steps[selectedStepIndex] : null;

  if (!step) {
    els.fluxImg.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = "Click a timestep to render its irradiance map.";
    els.fluxCaption.textContent = "";
    return;
  }

  if (fluxLoading) {
    els.fluxImg.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = "Rendering…";
    els.fluxCaption.textContent = "";
    return;
  }

  if (fluxError) {
    els.fluxImg.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = fluxError;
    els.fluxCaption.textContent = "";
    return;
  }

  if (fluxSrcUrl || fluxPngBase64) {
    els.fluxImg.src = fluxSrcUrl || "data:image/png;base64," + fluxPngBase64;
    els.fluxImg.hidden = false;
    els.fluxPlaceholder.hidden = true;
    els.fluxCaption.textContent = `${fmtHHMM(step.hour)} solar · peak ${fmtFlux(fluxPeakKwM2 != null ? fluxPeakKwM2 : step.peak_flux_kw_m2)}`;
  } else {
    els.fluxImg.hidden = true;
    els.fluxPlaceholder.hidden = false;
    els.fluxPlaceholder.textContent = "Click a timestep to render its irradiance map.";
    els.fluxCaption.textContent = "";
  }
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
  els.yearStartBtn.textContent = yearStarting ? "Starting…" : running ? "Running…" : "Run year estimate";
  els.yearCancelBtn.hidden = !running;
  els.yearCancelBtn.classList.toggle("disabled-link", yearCancelling);
  els.yearCancelBtn.textContent = yearCancelling ? "Cancelling…" : "Cancel";

  if (!busy) {
    const fidelity = store.get("ui.fidelity");
    const nHeliostats = currentHeliostatCount(lastCtx);
    const nTimesteps = estimateYearTimesteps(yearFastMode, formHourStep, formMinElevationDeg);
    const estimateS = estimateDurationS(fidelity, nHeliostats, nTimesteps);
    const text = durationWarningText(estimateS, nHeliostats, nTimesteps);
    els.yearDurationHint.hidden = !text;
    if (text) els.yearDurationHint.textContent = text;
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
}

let lastCtx = null;

export function render(container, ctx) {
  if (!built) build(container);
  lastContainer = container;
  lastCtx = ctx;
  paint();
}
