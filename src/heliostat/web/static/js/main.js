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
  postFieldTrace,
  postFluxCsv,
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

// -- full-screen tabs (docs/ui-spec.md 1, 3, phase 3c wave 1) -------------
// Workspace / Heliostat Shape / Analysis. Analysis stays inert (wave 2) --
// its click handler is a no-op while index.html keeps it .disabled.
const apptabWorkspace = document.getElementById("apptab-workspace");
const apptabShape = document.getElementById("apptab-shape");
const apptabAnalysis = document.getElementById("apptab-analysis");
const tabShapeSection = document.getElementById("tab-shape");
const shellEl = document.querySelector(".shell");

apptabWorkspace.addEventListener("click", () => store.set("ui.tab", "workspace"));
apptabShape.addEventListener("click", () => store.set("ui.tab", "shape"));
apptabAnalysis.addEventListener("click", () => {
  if (apptabAnalysis.classList.contains("disabled")) return;
  store.set("ui.tab", "analysis");
});

// The 3D scene (and plan/elevation) keep running underneath -- only
// visibility toggles, nothing is destroyed. shapeTab.render() is only
// called while its section is actually visible, matching its own "fetch on
// tab open, do nothing while hidden" refresh policy (js/tabs/shape.js).
function renderTabs() {
  const tab = store.get("ui.tab");
  apptabWorkspace.classList.toggle("active", tab === "workspace");
  apptabShape.classList.toggle("active", tab === "shape");
  apptabAnalysis.classList.toggle("active", tab === "analysis");
  shellEl.hidden = tab !== "workspace";
  runbar.hidden = tab !== "workspace";
  tabShapeSection.hidden = tab !== "shape";
  if (tab === "shape") {
    shapeTab.render(tabShapeSection, { geometry: lastGeometryResponse });
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

function renderAllPanels() {
  heliostatPanel.render(stageHeliostat);
  fieldPanel.render(stageField);
  receiverPanel.render(stageReceiver);
  sunPanel.render(stageSun);
  runPanel.render(runbar, runActions);
  // Needs the last geometry response (not store state) for a selected
  // heliostat's distance-from-axis readout -- see inspector.js's header comment.
  inspector.render(inspectorEl, { geometry: lastGeometryResponse });
  // Phase 3b: Library slide-over (docs/ui-spec.md 5). Fetches its own
  // listings on open / after a mutation rather than on every call here --
  // see library.js's header comment.
  library.render(libraryDrawer, libraryBackdrop);
  renderTopbar();
  renderTabs();

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

  if (view === "3d") {
    const heliostats = (lastGeometryResponse && lastGeometryResponse.heliostats) || [];
    raysChip.textContent = `Corner chief rays — viewing aid, no shading · ${heliostats.length.toLocaleString()} heliostat${heliostats.length === 1 ? "" : "s"}`;
  } else if (view === "plan") {
    raysChip.textContent = "Plan view — click a heliostat to inspect it · drag-to-move lands with the layout picker";
  } else {
    raysChip.textContent = "Elevation — dimensions are live and referenced to the heliostat plane";
  }

  if (view === "plan") planView.render(planContainer);
  else if (view === "elevation") elevationView.render(elevationContainer);
}

// -- geometry: live scene refresh on every doc edit ------------------------

let lastGeometryResponse = null;
const scheduleGeometry = createGeometryRequester(300);

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
  scene.updateGeometry(data);
  // Cache into the plan/elevation views too (never on error -- both keep
  // drawing their last valid geometry, same rule as the 3D scene) so
  // switching views renders instantly instead of waiting on the next edit.
  planView.setGeometry(data);
  elevationView.setGeometry(data);
  renderAllPanels();
}

function handleGeometryError(err) {
  const parsed = parseGeometryError(err);
  store.set("ui.geometryError", parsed);
  // Scene keeps its last valid geometry (docs/ui-spec.md 2.3) -- no scene
  // call here at all.
  renderAllPanels();
}

function refreshGeometryNow() {
  const body = buildGeometryRequest(store.get("doc"), { maxCornerSources: 500 });
  postGeometry(body).then(handleGeometrySuccess).catch(handleGeometryError);
}

function refreshGeometryDebounced() {
  const body = buildGeometryRequest(store.get("doc"), { maxCornerSources: 500 });
  scheduleGeometry(body, { onSuccess: handleGeometrySuccess, onError: handleGeometryError });
}

// -- run bar: trace ----------------------------------------------------------

function runTrace() {
  if (store.get("ui.traceBusy")) return;
  store.set("ui.traceBusy", true);
  store.set("ui.traceError", null);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const body = buildTraceRequest(doc, ui);
  const call = doc.field.mode === "field" ? postFieldTrace(body) : postTrace(body);
  call
    .then((data) => {
      store.set("ui.traceBusy", false);
      store.set("ui.traceResult", data);
      store.set("ui.traceTimestamp", Date.now());
      store.set("ui.staleResults", false);
      if (data.scene && data.scene.rays) scene.showTraceRays(data.scene.rays);
      renderAllPanels();
    })
    .catch((err) => {
      store.set("ui.traceBusy", false);
      store.set("ui.traceError", (err && err.message) || "trace failed");
      renderAllPanels();
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

function openFluxOverlay() {
  const data = store.get("ui.traceResult");
  if (!data || !data.flux_png) return;
  fluxOverlayImg.src = "data:image/png;base64," + data.flux_png;
  fluxOverlay.hidden = false;
}

function closeFluxOverlay() {
  fluxOverlay.hidden = true;
}

const runActions = { onRunTrace: runTrace, onExportCsv: exportFluxCsv, onOpenFlux: openFluxOverlay };

// docs/ui-spec.md 2.1: "a view pill in the corner ... offers 'back to 3D'
// at all times" -- the owning stage stays expanded (only ui.view resets).
viewPillBack.addEventListener("click", (e) => {
  e.preventDefault();
  store.set("ui.view", "3d");
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
    if (value) store.set("ui.view", "plan");
    else if (store.get("ui.view") === "plan") store.set("ui.view", "3d");
  } else if (path === "ui.expanded.receiver") {
    if (value) store.set("ui.view", "elevation");
    else if (store.get("ui.view") === "elevation") store.set("ui.view", "3d");
  }

  if (path.startsWith("doc.")) {
    const ui = store.get("ui");
    if (ui.traceResult && !ui.staleResults) {
      store.set("ui.staleResults", true);
      scene.clearTraceRays();
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
// The default field is doc.field.layout === "manuscript" (store.js), which
// needs the paper's actual positions in hand before the very first geometry
// request goes out -- otherwise that request would race the fetch and
// currentLayoutPayload() would fall back to the (wrong-looking) Fermat
// spiral for one frame. So the manuscript fetch is awaited here, before
// refreshGeometryNow(), rather than fired in parallel with it. A failed
// fetch is not fatal: the app still has the Fermat fallback
// (currentLayoutPayload) to draw something, so it warns and proceeds rather
// than blocking forever.

renderAllPanels();
fetchManuscriptField()
  .then((data) => {
    setManuscriptField(data.xy_mm);
  })
  .catch((err) => {
    console.warn("could not load the manuscript field, falling back to the Fermat spiral:", err);
    store.set("ui.geometryError", {
      message:
        "Could not load the manuscript field (" +
        ((err && err.message) || "network error") +
        ") -- showing the Fermat spiral instead.",
      forReceiver: false,
    });
    // Not sticky: the very next successful geometry request clears
    // ui.geometryError the same way any other transient error does
    // (handleGeometrySuccess), so this is a one-time notice, not a banner
    // that lingers after the fallback field has drawn fine.
  })
  .finally(() => {
    renderAllPanels();
    refreshGeometryNow();
  });
