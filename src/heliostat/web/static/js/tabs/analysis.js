// Analysis tab (docs/ui-spec.md 4, mockup M7): day sweeps against the
// current project -- a background job with progress/cancel, the
// energy-through-the-day plot, a per-timestep table, CSV export, and an
// on-demand irradiance map for whichever timestep is selected.
//
// Build-once/els/render(container, ctx) exactly like js/tabs/shape.js. The
// day-sweep run (job id, poll handle, last snapshot/result, the selected
// row, the flux-map fetch's abort controller) is module-local state, not
// store state -- js/library.js's drawer state is the same carve-out, and a
// run has to survive a tab switch, so it cannot depend on render() being
// called at all: the poll loop below runs on its own setTimeout chain and
// only touches the DOM when this tab's section is not hidden.
//
// Two backend pieces this screen deliberately does NOT build a UI for,
// because they don't exist yet: an annual estimate endpoint (the Year
// estimate strip is a disabled button and an honest line), and persisted
// runs (finished day-sweep jobs live in the server's memory only, capped at
// 8 -- there is no "saved runs" list here, just a note that says so).
import { store } from "../store.js";
import { setVal, segButton } from "../fields.js";
import {
  buildDayRequest,
  buildTraceRequest,
  dayExportUrl,
  dayFluxUrl,
  getDayResult,
  getDayStatus,
  postDayCancel,
  postDayStart,
  postFieldTrace,
  postTrace,
} from "../api.js";

const FIDELITY = [
  ["ultra_fast", "Ultra fast"],
  ["fast_accurate", "Fast accurate"],
  ["monte_carlo", "Monte Carlo"],
];

const DEFAULT_DATE = "2026-03-21"; // the server's own DaySite default
const DEFAULT_HOUR_STEP = 0.5;

let built = false;
let els = {};

let lastContainer = null;

// -- day-sweep run state (module-local -- see header) ----------------------
let formDate = DEFAULT_DATE;
let formHourStep = DEFAULT_HOUR_STEP;

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
  const body = buildDayRequest(doc, ui, { site, hour_step: formHourStep });
  sweepRequest = body;
  sweepPhysicsKey = physicsKey(body);

  postDayStart(body)
    .then((snap) => {
      starting = false;
      jobId = snap.job_id;
      jobSnapshot = snap;
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
  });
  stepField.appendChild(stepLabel);
  stepField.appendChild(stepInput);

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
  controlRow.appendChild(fidelitySeg);
  controlRow.appendChild(startBtn);
  controlRow.appendChild(cancelBtn);
  sweepPanel.appendChild(controlRow);

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
    "Finished runs are kept in memory for this session only (the most recent 8) — they are not saved with the project yet.";
  sweepPanel.appendChild(sweepHint);

  left.appendChild(sweepPanel);

  // -- energy through the day panel --------------------------------------
  const energyPanel = document.createElement("div");
  energyPanel.className = "panel";
  energyPanel.style.flex = "1 1 auto";
  energyPanel.style.display = "flex";
  energyPanel.style.flexDirection = "column";
  energyPanel.style.minHeight = "0";
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

  // -- year estimate strip (docs/ui-spec.md 4 -- no annual endpoint yet) --
  const yearPanel = document.createElement("div");
  yearPanel.className = "panel an-yearpanel";
  const yearRow = document.createElement("div");
  yearRow.className = "an-yearrow";
  const yearH2 = document.createElement("h2");
  yearH2.textContent = "Year estimate";
  const yearDesc = document.createElement("span");
  yearDesc.className = "an-yeardesc";
  yearDesc.textContent = "12 sample days, spaced in solar declination · DNI-weighted interpolation between them";
  const yearBtn = document.createElement("div");
  yearBtn.className = "btn primary disabled-link";
  yearBtn.textContent = "Run year";
  yearRow.appendChild(yearH2);
  yearRow.appendChild(yearDesc);
  yearRow.appendChild(yearBtn);
  yearPanel.appendChild(yearRow);
  const yearHint = document.createElement("div");
  yearHint.className = "hint";
  yearHint.textContent = "The annual-estimate endpoint doesn't exist on the server yet — this strip lights up once it does. No number is computed here in the meantime.";
  yearPanel.appendChild(yearHint);
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

  container.appendChild(subject);
  container.appendChild(content);

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
    fidelityBtns,
    startBtn,
    cancelBtn,
    progressRow,
    progressFill,
    progressText,
    statusLine,
    sweepErr,
    energyImg,
    energyPlaceholder,
    energyTotal,
    energyCsv,
    yearBtn,
    tbody,
    tsEmpty,
    tsWrap,
    fluxImg,
    fluxPlaceholder,
    fluxCaption,
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
  const fidelity = store.get("ui.fidelity");
  for (const [key, btn] of Object.entries(els.fidelityBtns)) btn.classList.toggle("active", key === fidelity);

  const busy = starting || running;
  els.startBtn.classList.toggle("disabled-link", busy);
  els.startBtn.textContent = starting ? "Starting…" : running ? "Running…" : "Start day sweep";
  els.cancelBtn.hidden = !running;
  els.cancelBtn.classList.toggle("disabled-link", cancelling);
  els.cancelBtn.textContent = cancelling ? "Cancelling…" : "Cancel";

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

function paint() {
  const doc = store.get("doc");
  paintSubject(doc, lastCtx);
  paintSweepControls();
  paintEnergyPanel();
  paintTimestepsTable();
  paintFluxPanel();

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
