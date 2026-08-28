// Wiring: store <-> panels <-> scene <-> api. This is the only module that
// talks to all four; every panel and scene3d.js is otherwise self-contained.
import { store } from "./store.js";
import {
  buildGeometryRequest,
  buildTraceRequest,
  buildFluxCsvRequest,
  createGeometryRequester,
  postGeometry,
  postTrace,
  postFieldTraceStart,
  getFieldTraceStatus,
  postFieldTraceCancel,
  getFieldTraceResult,
  postFluxCsv,
  postFluxFeaCsv,
  postSecondaryFluxFeaCsv,
  fetchManuscriptField,
  setManuscriptField,
} from "./api.js";
import { OPTICS_LABELS } from "./fields.js";
import { createScene } from "./scene3d.js";
import * as heliostatPanel from "./panels/heliostat.js";
import * as fieldPanel from "./panels/field.js";
import * as receiverPanel from "./panels/receiver.js";
import * as sunPanel from "./panels/sun.js";
import * as runPanel from "./panels/run.js";
import * as inspector from "./inspector.js";
import * as planView from "./views/plan.js";
import * as elevationView from "./views/elevation.js";
import * as library from "./library.js";
import * as shapeTab from "./tabs/shape.js";
import * as analysisTab from "./tabs/analysis.js";

const OPTICS_NAME = Object.fromEntries(OPTICS_LABELS);

const sceneContainer = document.getElementById("scene-container");
// docs/ui-spec.md 2.4: a click in the viewport selects a heliostat, the
// secondary, the receiver, or the sun -- scene3d.js owns the raycasting
// and click-vs-drag detection and reports back through this callback, so
// it never needs to import the store itself.
const scene = createScene(sceneContainer, { onSelect: (sel) => store.set("ui.selection", sel) });

const sunBanner = document.getElementById("sun-banner");
const errorStrip = document.getElementById("geometry-error-strip");
const raysChip = document.getElementById("rays-chip");
const raysChipText = document.getElementById("rays-chip-text");
const raysToggleLabel = document.getElementById("rays-toggle-label");
const raysToggle = document.getElementById("rays-toggle");
const refreshPill = document.getElementById("refresh-pill");
const orbitHint = document.getElementById("orbit-hint");
const inspectorEl = document.getElementById("inspector");

// docs/ui-spec.md 2.1: "the viewport follows the active stage" -- the plan
// and elevation views are siblings of #scene-container (index.html), shown
// one at a time via [hidden] so the 3D scene is never disposed on switch.
const planContainer = document.getElementById("plan-view");
const elevationContainer = document.getElementById("elevation-view");
const viewPill = document.getElementById("view-pill");
const viewPillMode = document.getElementById("view-pill-mode");
const viewPillBack = document.getElementById("view-pill-back");

const stageHeliostat = document.getElementById("stage-heliostat");
const stageField = document.getElementById("stage-field");
const stageReceiver = document.getElementById("stage-receiver");
const stageSun = document.getElementById("stage-sun");
const runbar = document.getElementById("runbar");

// -- trace status bar: relocated out of #runbar (Workspace-tab-only, see
// renderTabs) so a running field trace's progress and cancel control stay
// visible and usable from every app tab -----------------------------------
const tracebar = document.getElementById("tracebar");
const tracebarLabel = document.getElementById("tracebar-label");
const tracebarTrack = document.getElementById("tracebar-track");
const tracebarFill = document.getElementById("tracebar-fill");
const tracebarCancel = document.getElementById("tracebar-cancel");
tracebarCancel.addEventListener("click", () => {
  if (tracebarCancel.classList.contains("disabled-link")) return;
  tracebarCancel.classList.add("disabled-link");
  // The bar shows whichever trace is running: a workspace field trace, or an
  // Analysis-tab day sweep / year estimate (ui.analysisJob). Cancel routes to
  // the owner of the running job.
  if (store.get("ui.traceBusy")) cancelTrace();
  else analysisTab.cancelActiveAnalysisJob();
});

// -- top bar: Library button, save state, optics label (docs/ui-spec.md
// 1 + 5, mockup M5) -------------------------------------------------------
const libraryBtn = document.getElementById("library-btn");
const libraryDrawer = document.getElementById("library-drawer");
const libraryBackdrop = document.getElementById("library-backdrop");
const savestateEl = document.getElementById("savestate");
const opticsTagEl = document.getElementById("optics-tag");

libraryBtn.addEventListener("click", () => {
  store.set("ui.libraryOpen", !store.get("ui.libraryOpen"));
});

const apptabWorkspace = document.getElementById("apptab-workspace");
const apptabShape = document.getElementById("apptab-shape");
const apptabAnalysis = document.getElementById("apptab-analysis");
const tabShapeSection = document.getElementById("tab-shape");
const tabAnalysisSection = document.getElementById("tab-analysis");
const shellEl = document.querySelector(".shell");

apptabWorkspace.addEventListener("click", () => store.set("ui.tab", "workspace"));
apptabShape.addEventListener("click", () => store.set("ui.tab", "shape"));
apptabAnalysis.addEventListener("click", () => store.set("ui.tab", "analysis"));

// The 3D scene (and plan/elevation) keep running underneath -- only
// visibility toggles, nothing is destroyed. A tab module is rendered only
// while its own section is visible; both fetch on open rather than while
// hidden.
// Closing one stage while the other is still open hands the viewport to
// whichever stage remains, rather than dropping to 3D with the view pill
// hidden and an expanded stage that no longer matches what is shown.
function viewForOpenStage() {
  const expanded = store.get("ui.expanded");
  if (expanded.field) return "plan";
  if (expanded.receiver) return "elevation";
  return "3d";
}

function renderTabs() {
  const tab = store.get("ui.tab");
  apptabWorkspace.classList.toggle("active", tab === "workspace");
  apptabShape.classList.toggle("active", tab === "shape");
  apptabAnalysis.classList.toggle("active", tab === "analysis");
  shellEl.hidden = tab !== "workspace";
  runbar.hidden = tab !== "workspace";
  tabShapeSection.hidden = tab !== "shape";
  tabAnalysisSection.hidden = tab !== "analysis";
  if (tab === "shape") {
    shapeTab.render(tabShapeSection, { geometry: lastGeometryResponse });
  } else if (tab === "analysis") {
    analysisTab.render(tabAnalysisSection, { geometry: lastGeometryResponse });
  }
}

function renderTopbar() {
  const doc = store.get("doc");
  const ui = store.get("ui");
  libraryBtn.classList.toggle("primary", ui.libraryOpen);
  opticsTagEl.textContent = OPTICS_NAME[doc.optics] || doc.optics;
  if (!ui.projectName) {
    savestateEl.textContent = ui.dirty ? "Unsaved project" : "New project";
  } else {
    savestateEl.textContent = ui.dirty ? `${ui.projectName} — unsaved changes` : `${ui.projectName} — saved`;
  }
}

const fluxOverlay = document.getElementById("flux-overlay");
const fluxOverlayImg = document.getElementById("flux-overlay-img");
const fluxOverlayClose = document.getElementById("flux-overlay-close");
// v0.2 §M compass rider: N/E/S/W edges for a flat window, an axis caption in
// compass terms for the unwrapped cylinder/frustum maps (js/scene3d.js's own
// ground-plane compass markers are the 3D twin of these). Toggled in
// openFluxOverlay() below, off the same trace result's scene.receiver.kind
// the drape (§M.3) reads.
const fluxCompassN = document.getElementById("flux-compass-n");
const fluxCompassS = document.getElementById("flux-compass-s");
const fluxCompassE = document.getElementById("flux-compass-e");
const fluxCompassW = document.getElementById("flux-compass-w");
const fluxCompassAxis = document.getElementById("flux-compass-axis");

// Spec §C / mockup M9: Receiver | Secondary map selector + absorbed-heat
// readout, both living inside the same overlay the receiver map already
// opens into (js/tabs/analysis.js's own timestep panel is the other place
// mockup M9 draws this -- see that module's own fluxSurface handling).
const fluxSurfaceSeg = document.getElementById("flux-surfaceseg");
const fluxSurfaceReceiverBtn = document.getElementById("flux-surface-receiver");
const fluxSurfaceSecondaryBtn = document.getElementById("flux-surface-secondary");
const fluxSecondaryCanvas = document.getElementById("flux-secondary-canvas");
const fluxSecondaryCaption = document.getElementById("flux-secondary-caption");
const fluxSecondaryReadout = document.getElementById("flux-secondary-readout");
const fluxSecIncident = document.getElementById("flux-sec-incident");
const fluxSecAbsorbedLbl = document.getElementById("flux-sec-absorbed-lbl");
const fluxSecAbsorbed = document.getElementById("flux-sec-absorbed");
const fluxSecPeakAbsorbed = document.getElementById("flux-sec-peakabsorbed");
const fluxSecFidelity = document.getElementById("flux-sec-fidelity");
const fluxSecFeaExport = document.getElementById("flux-sec-fea-export");

// Field-trace progress and its cancel control, deliberately NOT gated by
// ui.tab (unlike #runbar's contents -- see renderTabs) -- a field trace is a
// background job that can run for minutes, so it must stay visible and
// cancellable no matter which tab the user switches to while it waits.
function renderTraceBar() {
  const ui = store.get("ui");
  if (!ui.traceBusy) {
    // Day sweeps and year estimates (Analysis tab jobs) share the same bar,
    // so a long-running sweep stays visible and cancellable from every tab
    // -- same contract as the field trace above.
    const aj = ui.analysisJob;
    if (aj) {
      tracebar.hidden = false;
      const kindLabel = aj.kind === "year" ? "Year estimate" : "Day sweep";
      let label = aj.detail || `${aj.done} / ${aj.total}`;
      if (aj.eta_s != null) {
        const etaS = Math.round(aj.eta_s);
        label += etaS >= 90 ? `, about ${Math.round(etaS / 60)} min left` : `, about ${etaS}s left`;
      }
      tracebarLabel.textContent = kindLabel + " — " + label;
      const haveTotal = aj.total > 0;
      tracebarTrack.hidden = !haveTotal;
      if (haveTotal) {
        const frac = aj.frac != null ? aj.frac : aj.done / aj.total;
        tracebarFill.style.width = Math.max(0, Math.min(100, 100 * frac)).toFixed(1) + "%";
      }
      tracebarCancel.hidden = false;
      return;
    }
    tracebar.hidden = true;
    tracebarCancel.classList.remove("disabled-link");
    return;
  }
  tracebar.hidden = false;
  const progress = ui.traceProgress;
  if (progress) {
    let label = progress.detail || `${progress.done} / ${progress.total} heliostats`;
    if (progress.eta_s != null) {
      const etaS = Math.round(progress.eta_s);
      label += etaS >= 90 ? `, about ${Math.round(etaS / 60)} min left` : `, about ${etaS}s left`;
    }
    tracebarLabel.textContent = "Tracing field — " + label;
    const haveTotal = progress.total > 0;
    tracebarTrack.hidden = !haveTotal;
    if (haveTotal) {
      // `frac` is cost-weighted (outer-ring heliostats trace slower than
      // inner-ring ones -- see heliostat.web.jobs.Job.snapshot) so the bar
      // tracks wall-time share instead of racing through cheap heliostats
      // and stalling on expensive ones; the label text above still counts
      // plain heliostats. Falls back to the plain count if a job never set
      // a weighted total.
      const frac = progress.frac != null ? progress.frac : progress.done / progress.total;
      const pct = Math.max(0, Math.min(100, 100 * frac));
      tracebarFill.style.width = pct.toFixed(1) + "%";
    }
    tracebarCancel.hidden = false;
  } else {
    // A single-heliostat trace is one plain request with no job behind it
    // to cancel or poll -- same "is this cancellable" test run.js's own
    // cancel button used before this moved out of it.
    tracebarLabel.textContent = "Tracing…";
    tracebarTrack.hidden = true;
    tracebarCancel.hidden = true;
  }
}

function renderAllPanels() {
  heliostatPanel.render(stageHeliostat);
  fieldPanel.render(stageField);
  receiverPanel.render(stageReceiver);
  sunPanel.render(stageSun);
  runPanel.render(runbar, runActions, {
    heliostatCount: (lastGeometryResponse && lastGeometryResponse.heliostats && lastGeometryResponse.heliostats.length) || 0,
  });
  // Needs the last geometry response (not store state) for a selected
  // heliostat's distance-from-axis readout -- see inspector.js's header comment.
  inspector.render(inspectorEl, { geometry: lastGeometryResponse });
  // Phase 3b: Library slide-over (docs/ui-spec.md 5). Fetches its own
  // listings on open / after a mutation rather than on every call here --
  // see library.js's header comment.
  library.render(libraryDrawer, libraryBackdrop);
  renderTopbar();
  renderTraceBar();
  renderTabs();
  renderRefreshPill();
  // Spec §C: repaints the flux overlay's Receiver | Secondary selector and
  // readout -- a no-op while the overlay is closed (paintFluxOverlay's own
  // guard), so toggling ui.fluxSurface or a fresh trace landing while it is
  // open both show up live.
  paintFluxOverlay();

  renderViewportMode();
}

// docs/ui-spec.md 2.1: shows exactly one of scene-container/plan-view/
// elevation-view per ui.view, updates the view pill and the bottom-left
// chip's view-specific text, and re-renders whichever view is actually
// showing -- at the same cadence as the sidebar panels (called from
// renderAllPanels, which every store change already triggers). The other
// two views simply don't render while hidden -- no wasted SVG rebuilds.
function renderViewportMode() {
  const view = store.get("ui.view");
  sceneContainer.hidden = view !== "3d";
  planContainer.hidden = view !== "plan";
  elevationContainer.hidden = view !== "elevation";
  // The orbit/pan/zoom hint describes the 3D controls only -- in plan or
  // elevation it would sit on top of the view's own chip and mislead.
  orbitHint.hidden = view !== "3d";

  viewPill.hidden = view === "3d";
  if (view === "plan") viewPillMode.textContent = "Plan";
  else if (view === "elevation") viewPillMode.textContent = "Elevation";

  raysToggleLabel.hidden = view !== "3d";
  if (view === "3d") {
    const heliostats = (lastGeometryResponse && lastGeometryResponse.heliostats) || [];
    const count = `${heliostats.length.toLocaleString()} heliostat${heliostats.length === 1 ? "" : "s"}`;
    raysChipText.textContent = raysVisible()
      ? `Corner chief rays — viewing aid, no shading · ${count}`
      : `Rays hidden · ${count}`;
  } else if (view === "plan") {
    raysChipText.textContent = "Plan view — click a heliostat to inspect it · drag-to-move lands with the layout picker";
  } else {
    raysChipText.textContent = "Elevation — dimensions are live and referenced to the heliostat plane";
  }

  if (view === "plan") planView.render(planContainer);
  else if (view === "elevation") elevationView.render(elevationContainer);
}

// -- geometry: live scene refresh on every doc edit ------------------------

// Geometry lands in two passes: the shapes first, then the corner rays.
// Measured on the 643-heliostat field, a geometry-only response takes about
// 130 ms against 220 ms with rays, so switching optics or layout redraws
// the scene noticeably sooner and the rays catch up. The ray pass is
// debounced longer than the shape pass, so a burst of typing pays only the
// cheap request until it settles.
let lastGeometryResponse = null;
let lastRaysResponse = null;
const scheduleGeometry = createGeometryRequester(300);
const scheduleRays = createGeometryRequester(450);

// ui.showRays is undefined until the toggle is first used, which reads as on.
function raysVisible() {
  return store.get("ui.showRays") !== false;
}

raysToggle.addEventListener("change", () => {
  store.set("ui.showRays", raysToggle.checked);
  applyRayVisibility();
  renderViewportMode();
});

// The plan and elevation views draw the same corner rays the 3D scene does,
// so they are fed from whichever pass last landed: the ray-bearing one when
// rays are on, otherwise the shapes with an empty ray list.
function viewGeometry() {
  if (raysVisible() && lastRaysResponse) return lastRaysResponse;
  if (!lastGeometryResponse) return null;
  return Object.assign({}, lastGeometryResponse, { rays: [] });
}

function pushGeometryToViews() {
  const geometry = viewGeometry();
  if (!geometry) return;
  planView.setGeometry(geometry);
  elevationView.setGeometry(geometry);
}

function applyRayVisibility() {
  if (raysVisible()) {
    if (lastRaysResponse) scene.updateRays(lastRaysResponse);
  } else {
    scene.updateRays({ rays: [], miss: null });
  }
  pushGeometryToViews();
}

// One "still working" indicator for both geometry passes and a trace, so a
// change that has not finished landing everywhere says so.
let pendingShapes = 0;
let pendingRays = 0;

function renderRefreshPill() {
  const busy = pendingShapes > 0 || pendingRays > 0 || store.get("ui.traceBusy");
  refreshPill.hidden = !busy;
  if (!busy) return;
  if (store.get("ui.traceBusy")) refreshPill.textContent = "Tracing…";
  else if (pendingShapes > 0) refreshPill.textContent = "Updating geometry…";
  else refreshPill.textContent = "Updating rays…";
}

function parseGeometryError(err) {
  const message = err && err.message ? err.message : "geometry request failed";
  // Best-effort placement: heliostat.web.app's resolve_optics_params
  // flattens pydantic errors as "optics_params for 'axicon' -- ..." --
  // anything naming optics_params belongs under the Receiver & Tower
  // stage; everything else is a general strip (docs/ui-spec.md 2.3's
  // warning/error contract, applied best-effort on the client).
  return { message, forReceiver: message.includes("optics_params") };
}

function handleGeometrySuccess(data) {
  pendingShapes = 0;
  store.set("ui.geometryError", null);
  if (data.sun_below_horizon) {
    store.set("ui.sunBelowHorizon", true);
    scene.clearAllRays();
    renderAllPanels();
    return;
  }
  store.set("ui.sunBelowHorizon", false);
  // docs/ui-spec.md 2.3: the `miss` key (aperture_miss_ids, total_miss_ids,
  // needed_aperture_radius_mm, dropped rays) may not be live on the backend
  // yet -- `data.miss` undefined/null both read as "no warnings" throughout
  // fields.js's apertureMissMessage() and scene3d.js's miss tinting.
  store.set("ui.miss", data.miss || null);
  lastGeometryResponse = data;
  // A heliostat can vanish from under the selection -- shrink the field, or
  // switch layout, and the inspector would otherwise stay bound to an id
  // that is no longer in the scene, editing a ghost.
  const sel = store.get("ui.selection");
  if (sel && sel.kind === "heliostat" && sel.id != null) {
    const present = (data.heliostats || []).some((h) => h.id === sel.id);
    if (!present) store.set("ui.selection", null);
  }
  // The shapes pass carries no rays, so anything the previous pass drew is
  // now stale -- drop it rather than hang mismatched rays off new geometry.
  lastRaysResponse = null;
  scene.updateGeometry(data);
  // Cache into the plan/elevation views too (never on error -- both keep
  // drawing their last valid geometry, same rule as the 3D scene) so
  // switching views renders instantly instead of waiting on the next edit.
  pushGeometryToViews();
  renderAllPanels();
}

function handleGeometryError(err) {
  pendingShapes = 0;
  const parsed = parseGeometryError(err);
  store.set("ui.geometryError", parsed);
  // Scene keeps its last valid geometry (docs/ui-spec.md 2.3) -- no scene
  // call here at all.
  renderAllPanels();
}

// Second pass: rays and the miss warnings that ride with them, applied
// without rebuilding the meshes the first pass just built.
function handleRaysSuccess(data) {
  pendingRays = 0;
  if (data.sun_below_horizon) return;
  lastRaysResponse = data;
  store.set("ui.miss", data.miss || null);
  if (lastGeometryResponse) lastGeometryResponse.miss = data.miss || null;
  applyRayVisibility(); // also refeeds the plan/elevation views
  renderAllPanels();
}

function geometryBody(withRays) {
  const body = buildGeometryRequest(store.get("doc"), { maxCornerSources: 500 });
  body.include_corner_rays = withRays;
  return body;
}

function requestRays() {
  if (!raysVisible()) {
    pendingRays = 0;
    return;
  }
  pendingRays = 1;
  scheduleRays(geometryBody(true), {
    onSuccess: handleRaysSuccess,
    // A failed ray pass is silent: the shape pass owns the error strip, and
    // reporting the same 422 twice would only fight with it.
    onError: () => {
      pendingRays = 0;
      renderRefreshPill();
    },
  });
}

function refreshGeometryNow() {
  pendingShapes = 1;
  renderRefreshPill();
  postGeometry(geometryBody(false)).then(handleGeometrySuccess).catch(handleGeometryError);
  requestRays();
}

function refreshGeometryDebounced() {
  pendingShapes = 1;
  renderRefreshPill();
  scheduleGeometry(geometryBody(false), {
    onSuccess: handleGeometrySuccess,
    onError: handleGeometryError,
  });
  requestRays();
}

// -- run bar: trace ----------------------------------------------------------

// A single-heliostat trace is one request/response, same as always. A field
// trace runs on a background job instead (heliostat.web.jobs, the same shape
// tabs/analysis.js's day sweep already polls) -- a field is hundreds of
// mirrors and, even parallel across cores, minutes of work at the 1000-
// heliostat cap, which is too long to hold a request open with nothing to
// show and no way to stop it. `traceJobId`/`tracePollTimer` are module-local
// (not store state) for the same reason analysis.js's own poll state is:
// they belong to this network loop, not to anything a re-render should own.
let traceJobId = null;
let tracePollTimer = null;
// Set when a doc edit lands while a trace is in flight, and the fidelity the
// run actually went out with -- results must be labelled with what traced
// them, not with whatever the run bar happens to show when they arrive.
let editedDuringTrace = false;
let tracedFidelity = null;

function traceSucceeded(data) {
  store.set("ui.traceBusy", false);
  store.set("ui.traceProgress", null);
  store.set("ui.traceResult", data);
  store.set("ui.traceTimestamp", Date.now());
  // Stale means "the project moved on since this run started", so an edit
  // made WHILE a trace was running leaves its results stale the moment they
  // land -- clearing the flag here unconditionally would dress old numbers,
  // and old rays over new geometry, as fresh.
  const stale = editedDuringTrace;
  editedDuringTrace = false;
  store.set("ui.staleResults", stale);
  store.set("ui.traceFidelity", tracedFidelity);
  if (!stale && data.scene && data.scene.rays) {
    scene.showTraceRays(data.scene.rays, data.scene.miss_rays);
  }
  // §M.3: the 3D receiver drape, same staleness rule as the rays above --
  // flux_grid is opt-in (buildTraceRequest always asks for it) and absent
  // for a receiver-less optics/scene, in which case scene.showFluxDrape is a
  // no-op.
  if (!stale && data.flux_grid) {
    scene.showFluxDrape(data.flux_grid);
  } else {
    scene.clearFluxDrape();
  }
  renderAllPanels();
}

function traceFailed(message) {
  store.set("ui.traceBusy", false);
  store.set("ui.traceProgress", null);
  store.set("ui.traceError", message || "trace failed");
  renderAllPanels();
}

function runTrace() {
  if (store.get("ui.traceBusy")) return;
  editedDuringTrace = false;
  tracedFidelity = store.get("ui.fidelity");
  store.set("ui.traceBusy", true);
  store.set("ui.traceError", null);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildTraceRequest(doc, ui);
  if (doc.field.mode === "field") {
    runFieldTraceJob(body);
    return;
  }
  postTrace(body).then(traceSucceeded).catch((err) => traceFailed(err && err.message));
}

function runFieldTraceJob(body) {
  if (tracePollTimer) {
    clearTimeout(tracePollTimer);
    tracePollTimer = null;
  }
  store.set("ui.traceProgress", null);
  postFieldTraceStart(body)
    .then((snap) => {
      traceJobId = snap.job_id;
      store.set("ui.traceProgress", snap);
      scheduleTracePoll();
    })
    .catch((err) => {
      traceJobId = null;
      traceFailed(err && err.message);
    });
}

function scheduleTracePoll() {
  if (tracePollTimer) clearTimeout(tracePollTimer);
  tracePollTimer = setTimeout(tracePollTick, 500);
}

function tracePollTick() {
  tracePollTimer = null;
  if (!traceJobId) return;
  const thisJob = traceJobId;
  getFieldTraceStatus(thisJob)
    .then((snap) => {
      if (traceJobId !== thisJob) return; // superseded by a newer run
      store.set("ui.traceProgress", snap);
      if (snap.state === "running") {
        scheduleTracePoll();
        return;
      }
      if (snap.state === "cancelled") {
        // No partial field to show -- see field_trace_start's own doc: a
        // field's flux is a sum across every mirror, so half of one is not
        // a smaller-but-valid answer.
        traceJobId = null;
        store.set("ui.traceBusy", false);
        store.set("ui.traceProgress", null);
        renderAllPanels();
        return;
      }
      if (snap.state === "error") {
        traceJobId = null;
        traceFailed(snap.error || "the field trace failed");
        return;
      }
      getFieldTraceResult(thisJob)
        .then((data) => {
          if (traceJobId !== thisJob) return;
          traceJobId = null;
          traceSucceeded(data);
        })
        .catch((err) => {
          if (traceJobId !== thisJob) return;
          traceJobId = null;
          traceFailed(err && err.message);
        });
    })
    .catch((err) => {
      if (traceJobId !== thisJob) return;
      traceJobId = null;
      traceFailed((err && err.message) || "lost track of the trace");
    });
}

function cancelTrace() {
  if (!traceJobId) return;
  postFieldTraceCancel(traceJobId).catch(() => {
    // The poll loop above is the source of truth for whether the job
    // actually stopped -- a failed cancel call just means it may run to
    // completion, not that the UI should get stuck saying "cancelling".
  });
}

function exportFluxCsv() {
  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildFluxCsvRequest(doc, ui);
  postFluxCsv(body)
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "heliostat-flux-kW_m2.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    })
    .catch((err) => {
      store.set("ui.traceError", "CSV export failed: " + ((err && err.message) || "unknown error"));
      renderAllPanels();
    });
}

// docs/ui-spec-v0.2.md §D: the ANSYS-oriented FEA CSV grid for the run
// bar's flux map -- same request body as exportFluxCsv above (same
// single-heliostat-map-even-in-field-mode limitation buildFluxCsvRequest
// documents), a different file: meters/W-m2 behind commented metadata
// instead of a labelled mm/kW-m2 matrix.
function exportFluxFeaCsv() {
  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildFluxCsvRequest(doc, ui);
  postFluxFeaCsv(body)
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "heliostat-flux-fea.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    })
    .catch((err) => {
      store.set("ui.traceError", "CSV export failed: " + ((err && err.message) || "unknown error"));
      renderAllPanels();
    });
}

// docs/ui-spec-v0.2.md §C leftover: the secondary's own FEA CSV, same body
// and download pattern as exportFluxFeaCsv above, just the secondary
// endpoint -- wired from the flux overlay's Secondary panel (fluxSecFeaExport
// below), since that is the only place with a secondary map to export from.
function exportSecondaryFluxFeaCsv() {
  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildFluxCsvRequest(doc, ui);
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
      store.set("ui.traceError", "CSV export failed: " + ((err && err.message) || "unknown error"));
      renderAllPanels();
    });
}

// v0.2 §M compass rider: N/E/S/W on a flat window's map edges; the unwrapped
// cylinder/frustum axis labelled in compass terms instead, seam-to-seam.
//
// The seam sits at +y (north) -- heliostat.geometry.receiver's module
// docstring and CylinderReceiver/FrustumReceiver's own ("azimuth arc...
// seam at +y"), with az = atan2(x, -y) measured from -y (south)
// (_continuous_azimuth). u = radius(_mean) * az spans uv_extent's
// (-half_circ, +half_circ), i.e. az over [-pi, pi] -- so u_min and u_max are
// BOTH the seam (north, the same physical generatrix), and the map's
// horizontal centre (u=0, az=0) is south. Walking left (u_min) to right
// (u_max): az runs -pi -> 0 -> +pi, passing -pi/2 (west) on the way to 0
// and +pi/2 (east) on the way out -- N . W . S . E . N, not the more
// guessable "seam-is-south" reading.
const CYLINDER_AXIS_COMPASS = ["N", "W", "S", "E", "N"];

// -- flux overlay: Receiver | Secondary (spec §C, mockup M9) ---------------

function fmtPower(w) {
  if (w == null || !Number.isFinite(w)) return "—";
  if (Math.abs(w) >= 1e6) return (w / 1e6).toFixed(2) + " MW";
  return (w / 1e3).toFixed(1) + " kW";
}

function fmtFlux(kwM2) {
  if (kwM2 == null || !Number.isFinite(kwM2)) return "—";
  if (Math.abs(kwM2) >= 1000) return (kwM2 / 1000).toFixed(2) + " MW/m²";
  return kwM2.toFixed(1) + " kW/m²";
}

// docs/secondary-irradiance-plan.md: "UI must say coarse in cone modes,
// exact in Monte Carlo wherever the secondary map shows" -- stated in
// plain text next to the readout, not tucked into a tooltip.
function secondaryFidelityNote(fidelity) {
  if (fidelity === "exact") {
    return "Exact fidelity — Monte Carlo histograms every ray that actually struck the secondary.";
  }
  return "Coarse fidelity — this cone mode deposits each mirror's flux at its own chief ray's secondary hit, not a full footprint. Switch to Monte Carlo for an exact per-ray map.";
}

// Same compact magma approximation as scene3d.js's fluxGridTexture (kept as
// its own copy here rather than a shared import -- this file draws to a
// plain 2D canvas for a modal, not a THREE.CanvasTexture for the 3D scene,
// and the app's own idiom is a small per-file copy over a shared util
// module; see that file's own comment for why these five stops are close
// enough to matplotlib's real 256-entry table).
const SECONDARY_MAGMA_STOPS = [
  [0.0, [0, 0, 4]],
  [0.2, [43, 17, 84]],
  [0.4, [120, 28, 109]],
  [0.6, [196, 60, 79]],
  [0.8, [251, 135, 97]],
  [1.0, [252, 253, 191]],
];
function secondaryMagmaColor(t) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < SECONDARY_MAGMA_STOPS.length; i++) {
    const [t0, c0] = SECONDARY_MAGMA_STOPS[i - 1];
    const [t1, c1] = SECONDARY_MAGMA_STOPS[i];
    if (x <= t1 || i === SECONDARY_MAGMA_STOPS.length - 1) {
      const f = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return SECONDARY_MAGMA_STOPS[SECONDARY_MAGMA_STOPS.length - 1][1];
}

// Paints app.py's _flux_grid_payload straight onto a 2D canvas -- there is
// no server-rendered PNG for the secondary (only the opt-in raw grid, spec
// §C), so this is the client's own rendering of it. `values` is row-major,
// row 0 = v_min (the bottom of the map, matplotlib's own origin="lower"
// convention _render_flux_png uses for the receiver PNG) -- canvas row 0 is
// its TOP, so row 0 of `values` is drawn into the canvas's LAST row,
// exactly mirroring scene3d.js's fluxGridTexture flip.
function paintSecondaryCanvas(canvas, grid) {
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
      const [r, g, b] = secondaryMagmaColor(max > 0 && val != null ? val / max : 0);
      const idx = (canvasRow * n_u + col) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx2d.putImageData(img, 0, 0);
}

// Repaints the overlay's Receiver | Secondary selector and body against
// whatever ui.traceResult/ui.fluxSurface currently say -- called both when
// the overlay is opened and from renderAllPanels (guarded on visibility)
// so toggling the shared ui.fluxSurface preference, or a fresh trace
// landing while the overlay happens to be open, both repaint it live.
function paintFluxOverlay() {
  if (fluxOverlay.hidden) return;
  const data = store.get("ui.traceResult");
  if (!data) return;
  const secondary = data.secondary;
  const optics = store.get("doc.optics");
  const hasSecondaryOptics = optics === "axicon" || optics === "cassegrain";
  const available = !!(secondary && secondary.flux_grid);

  fluxSurfaceSecondaryBtn.classList.toggle("disabled", !available);
  fluxSurfaceSecondaryBtn.title = available
    ? ""
    : hasSecondaryOptics
      ? "This trace carried no secondary flux map — retrace to get one."
      : "Only axicon and Cassegrain layouts have a secondary flux map.";

  const requested = store.get("ui.fluxSurface");
  const showSecondary = requested === "secondary" && available;
  fluxSurfaceReceiverBtn.classList.toggle("active", !showSecondary);
  fluxSurfaceSecondaryBtn.classList.toggle("active", showSecondary);

  if (showSecondary) {
    fluxOverlayImg.hidden = true;
    fluxCompassN.hidden = true;
    fluxCompassS.hidden = true;
    fluxCompassE.hidden = true;
    fluxCompassW.hidden = true;
    fluxCompassAxis.hidden = true;

    fluxSecondaryCanvas.hidden = false;
    paintSecondaryCanvas(fluxSecondaryCanvas, secondary.flux_grid);
    fluxSecondaryCaption.hidden = false;
    fluxSecondaryCaption.textContent = `incident flux on secondary, kW/m² · same colormap & units as the receiver map · peak ${fmtFlux(secondary.peak_flux_kw_m2)}`;

    fluxSecondaryReadout.hidden = false;
    fluxSecIncident.textContent = fmtPower(secondary.power_w);
    const rPct = (secondary.secondary_reflectance * 100).toFixed(1);
    fluxSecAbsorbedLbl.textContent = `absorbed (R = ${rPct} %)`;
    fluxSecAbsorbed.textContent = fmtPower(secondary.absorbed_power_w);
    fluxSecPeakAbsorbed.textContent = fmtFlux(secondary.peak_absorbed_kw_m2);
    fluxSecFidelity.textContent = secondaryFidelityNote(secondary.fidelity);
    // §C leftover: only ever exportable from a live trace's own grid --
    // same "the raw grid, not just the picture" gate as available above.
    fluxSecFeaExport.hidden = !available;
  } else {
    fluxOverlayImg.hidden = false;
    fluxSecondaryCanvas.hidden = true;
    fluxSecondaryCaption.hidden = true;
    fluxSecondaryReadout.hidden = true;
    fluxSecFeaExport.hidden = true;

    const kind = data.scene && data.scene.receiver ? data.scene.receiver.kind : null;
    const isFlat = kind === "flat";
    const isCurved = kind === "cylinder" || kind === "frustum";
    fluxCompassN.hidden = !isFlat;
    fluxCompassS.hidden = !isFlat;
    fluxCompassE.hidden = !isFlat;
    fluxCompassW.hidden = !isFlat;
    fluxCompassAxis.hidden = !isCurved;
    if (isCurved && !fluxCompassAxis.childElementCount) {
      for (const letter of CYLINDER_AXIS_COMPASS) {
        const span = document.createElement("span");
        span.textContent = letter;
        fluxCompassAxis.appendChild(span);
      }
    }
  }
}

fluxSurfaceReceiverBtn.addEventListener("click", () => store.set("ui.fluxSurface", "receiver"));
fluxSurfaceSecondaryBtn.addEventListener("click", () => {
  if (fluxSurfaceSecondaryBtn.classList.contains("disabled")) return;
  store.set("ui.fluxSurface", "secondary");
});
fluxSecFeaExport.addEventListener("click", (e) => {
  e.preventDefault();
  exportSecondaryFluxFeaCsv();
});

function openFluxOverlay() {
  const data = store.get("ui.traceResult");
  if (!data || !data.flux_png) return;
  fluxOverlayImg.src = "data:image/png;base64," + data.flux_png;
  fluxOverlay.hidden = false;
  paintFluxOverlay();
}

function closeFluxOverlay() {
  fluxOverlay.hidden = true;
}

const runActions = {
  onRunTrace: runTrace,
  onCancelTrace: cancelTrace,
  onExportCsv: exportFluxCsv,
  onExportFeaCsv: exportFluxFeaCsv,
  onOpenFlux: openFluxOverlay,
};

// docs/ui-spec.md 2.1: "a view pill in the corner ... offers 'back to 3D'
// at all times" -- the owning stage stays expanded (only ui.view resets).
viewPillBack.addEventListener("click", (e) => {
  e.preventDefault();
  // Collapsing the stage that opened this view is what returns to 3D (the
  // store subscriber below does it), and it leaves the sidebar showing only
  // what is still relevant.
  const view = store.get("ui.view");
  if (view === "plan") store.set("ui.expanded.field", false);
  else if (view === "elevation") store.set("ui.expanded.receiver", false);
  else store.set("ui.view", "3d");
});

fluxOverlayClose.addEventListener("click", closeFluxOverlay);
fluxOverlay.addEventListener("click", (e) => {
  if (e.target === fluxOverlay) closeFluxOverlay();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeFluxOverlay();
    // Phase 3b: the Library drawer takes priority -- Esc closes it first,
    // and only deselects (docs/ui-spec.md 2.4) once it's already shut.
    if (store.get("ui.libraryOpen")) {
      store.set("ui.libraryOpen", false);
    } else if (store.get("ui.tab") !== "workspace") {
      // Phase 3c wave 1: Esc backs out of a full-screen tab (docs/ui-spec.md
      // 3's "Done -- back to workspace" has a keyboard equivalent) -- one
      // layer per keypress, like the overlay and drawer above it.
      store.set("ui.tab", "workspace");
    } else {
      store.set("ui.selection", null);
    }
  }
});

// -- store wiring ------------------------------------------------------------

store.subscribe((path, value) => {
  // docs/ui-spec.md 2.1: "the viewport follows the active stage" -- wired
  // on the specific expand/collapse events (not derived from ui.expanded
  // as a whole), so the view pill's manual "back to 3D" isn't silently
  // re-overridden by a stage that's simply still open. Expanding switches
  // to that stage's view; collapsing only resets to "3d" if that stage was
  // the one currently showing (collapsing Field while in elevation, say,
  // must not touch it).
  if (path === "ui.expanded.field") {
    if (value) {
      store.set("ui.view", "plan");
      // v0.2 fix wave item 2: Field and Receiver & Tower are mutually
      // exclusive -- expanding one collapses the other so the viewport
      // (just switched to this stage's own view, above) is never left
      // showing a stage that isn't actually the one on screen.
      if (store.get("ui.expanded.receiver")) store.set("ui.expanded.receiver", false);
    } else if (store.get("ui.view") === "plan") store.set("ui.view", viewForOpenStage());
  } else if (path === "ui.expanded.receiver") {
    if (value) {
      store.set("ui.view", "elevation");
      if (store.get("ui.expanded.field")) store.set("ui.expanded.field", false);
    } else if (store.get("ui.view") === "elevation") store.set("ui.view", viewForOpenStage());
  }

  if (path === "ui.fidelity") {
    // Fidelity is part of what produced a number, so switching it makes the
    // results on screen describe a run nobody asked for any more.
    const ui = store.get("ui");
    if (ui.traceResult && !ui.staleResults && ui.traceFidelity && value !== ui.traceFidelity) {
      store.set("ui.staleResults", true);
      scene.clearTraceRays();
      scene.clearFluxDrape();
    }
  }

  if (path.startsWith("doc.")) {
    const ui = store.get("ui");
    if (ui.traceBusy) editedDuringTrace = true;
    if (ui.traceResult && !ui.staleResults) {
      store.set("ui.staleResults", true);
      scene.clearTraceRays();
      scene.clearFluxDrape();
    }
    // Phase 3b save state (docs/ui-spec.md 5): any edit dirties the
    // project -- guarded so a doc.* change doesn't re-set (and re-notify
    // subscribers over) an already-true flag on every single keystroke.
    // library.js clears this back to false right after a load/save.
    if (!ui.dirty) store.set("ui.dirty", true);
    renderAllPanels();
    refreshGeometryDebounced();
  } else if (path.startsWith("ui.")) {
    sunBanner.hidden = !store.get("ui.sunBelowHorizon");
    const err = store.get("ui.geometryError");
    if (err && !err.forReceiver) {
      errorStrip.hidden = false;
      errorStrip.textContent = err.message;
    } else {
      errorStrip.hidden = true;
    }
    // Keeps the 3D highlight (per-instance heliostat tint, secondary/
    // receiver edge, sun color) in sync with ui.selection however it
    // changed -- a scene click, Esc, or a sidebar interaction that
    // happens to also carry a selection (docs/ui-spec.md 2.4).
    scene.setSelection(store.get("ui.selection"));
    renderAllPanels();
  }
});

// -- first paint: live from the first frame ---------------------------------
//
// The default field is doc.field.layout === "radial_stagger" (store.js), a
// purely parametric layout -- the very first geometry request needs no
// network fetch to be correct, so it fires immediately. The manuscript
// field is fetched here too, in parallel rather than blocking that first
// request, so switching the Layout picker to "Manuscript 643" already has
// its positions cached instead of triggering a fetch on that click.

renderAllPanels();
refreshGeometryNow();
fetchManuscriptField()
  .then((data) => {
    setManuscriptField(data.xy_mm);
  })
  .catch((err) => {
    console.warn("could not prefetch the manuscript field:", err);
  })
  .finally(() => {
    renderAllPanels();
  });
