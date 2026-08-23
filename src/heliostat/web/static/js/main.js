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
} from "./api.js";
import { createScene } from "./scene3d.js";
import * as heliostatPanel from "./panels/heliostat.js";
import * as fieldPanel from "./panels/field.js";
import * as receiverPanel from "./panels/receiver.js";
import * as sunPanel from "./panels/sun.js";
import * as runPanel from "./panels/run.js";

const sceneContainer = document.getElementById("scene-container");
const scene = createScene(sceneContainer);

const sunBanner = document.getElementById("sun-banner");
const errorStrip = document.getElementById("geometry-error-strip");
const raysChip = document.getElementById("rays-chip");

const stageHeliostat = document.getElementById("stage-heliostat");
const stageField = document.getElementById("stage-field");
const stageReceiver = document.getElementById("stage-receiver");
const stageSun = document.getElementById("stage-sun");
const runbar = document.getElementById("runbar");

const fluxOverlay = document.getElementById("flux-overlay");
const fluxOverlayImg = document.getElementById("flux-overlay-img");
const fluxOverlayClose = document.getElementById("flux-overlay-close");

function renderAllPanels() {
  heliostatPanel.render(stageHeliostat);
  fieldPanel.render(stageField);
  receiverPanel.render(stageReceiver);
  sunPanel.render(stageSun);
  runPanel.render(runbar, runActions);

  const heliostats = (lastGeometryResponse && lastGeometryResponse.heliostats) || [];
  raysChip.textContent = `Corner chief rays — viewing aid, no shading · ${heliostats.length.toLocaleString()} heliostat${heliostats.length === 1 ? "" : "s"}`;
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
  lastGeometryResponse = data;
  scene.updateGeometry(data);
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

fluxOverlayClose.addEventListener("click", closeFluxOverlay);
fluxOverlay.addEventListener("click", (e) => {
  if (e.target === fluxOverlay) closeFluxOverlay();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeFluxOverlay();
});

// -- store wiring ------------------------------------------------------------

store.subscribe((path) => {
  if (path.startsWith("doc.")) {
    const ui = store.get("ui");
    if (ui.traceResult && !ui.staleResults) {
      store.set("ui.staleResults", true);
      scene.clearTraceRays();
    }
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
    renderAllPanels();
  }
});

// -- first paint: live from the first frame ---------------------------------

renderAllPanels();
refreshGeometryNow();
