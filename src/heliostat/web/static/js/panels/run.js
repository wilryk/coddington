// Trace bar (fidelity, Run button, stale/error chips) + results dock (the
// flux thumbnail -> full overlay, peak/mean/intercept, CSV exports)
// (docs/ui-spec.md 2.5; docs/ui-spec-v0.2.md §N/mockup M18b: the same
// content, just split across two containers now -- a condensed bar above
// the 3D scene and a ~380px dock beside it, instead of one full-width
// bottom bar). Trace-shaped network calls live in main.js (this module only
// builds DOM and calls back into the `actions` it is given), so this file
// stays a pure view over the store plus those callbacks.
import { store } from "../store.js";
import {
  apertureReceiverIsFlat,
  apertureDefaultCenter,
  apertureDefaultRadiusMm,
  clampToGridAxis,
  apertureClampRadius,
  apertureMetrics,
  apertureDataToCanvas,
  apertureCanvasToData,
  paintApertureCanvas,
  apertureCanvasEventPoint,
} from "../aperture.js";

const FIDELITY = [
  ["ultra_fast", "Ultra fast"],
  ["fast_accurate", "Fast accurate"],
  ["monte_carlo", "Monte Carlo"],
];

// docs/ui-spec-v0.2.md §A: one-line purpose subtitle plus what each mode
// trades away, verbatim from the signed-off table.
const FIDELITY_TOOLTIPS = {
  ultra_fast:
    "Field design optimization — explore layouts and geometry quickly. Trades away exact shadowing/blocking during sweeps (interpolated between anchors) and a small map-detail residual.",
  // v0.2 followups item 2: Fast accurate stays the slower reference-cone
  // mode by owner decision -- its wording now says so, pointing individual-
  // heliostat work here and full-field work at Ultra fast instead.
  fast_accurate:
    "Analyze a single heliostat with the highest peak-flux fidelity of any mode — deterministic and noise-free, but for full-field work, reach for Ultra fast instead.",
  monte_carlo:
    "Model the final design with precision, including all error sources. Noise falls as 1/√rays; the only mode that applies measured error maps and pointing error per ray.",
};

let built = false;
let els = {};

function isFocused(el) {
  return el && document.activeElement === el;
}

function fmt(x, digits) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return x.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// The response carries peak_flux_kw_m2, mean_flux_kw_m2, power_w,
// incident_power_w and per-stage ray counters (heliostat.web.app's
// /api/trace and /api/field/trace) but no literal "intercept efficiency"
// field. mean_flux_kw_m2 is the backend's own area-weighted mean over the
// receiver's full modeled surface (see app.py's _mean_flux_kw_m2) -- this
// used to be derived here from power_w over a box built from
// optics_resolved's window_half_u/v_mm, but those describe the ENTRANCE
// APERTURE (see PrimeFocusOptics), not the absorbing surface behind it;
// for a curved receiver (cylinder/frustum) that box has nothing to do with
// the receiver's true area, so the derived "mean" could land above the
// correctly-normalised peak (observed: peak 1007.1 kW/m^2, mean
// 1393.1 kW/m^2). Reading the backend's own field keeps both numbers drawn
// from the same flux grid, so peak >= mean always. Intercept is
// power/incident for the cone backends, or in_window/hit_secondary from
// the counters for Monte Carlo, which carries no incident_power_w -- a
// judgment call, noted in the build report.
function deriveMetrics(data) {
  const out = { peak: null, mean: null, intercept: null };
  if (data.peak_flux_kw_m2 != null) out.peak = data.peak_flux_kw_m2;
  if (data.mean_flux_kw_m2 != null) out.mean = data.mean_flux_kw_m2;
  if (data.incident_power_w != null && data.incident_power_w > 0) {
    // The kernel integration can land a fraction of a percent above the
    // analytic incident power, and "100.1 %" reads as a bug rather than
    // as quadrature noise -- clamp at the physical ceiling.
    out.intercept = Math.min(1, data.power_w / data.incident_power_w) * 100;
  } else if (data.counters && data.counters.hit_secondary) {
    out.intercept = ((data.counters.in_window || 0) / data.counters.hit_secondary) * 100;
  }
  return out;
}

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

// -- results dock's own analysis aperture (v0.2 followups item 3, mockup
// M17) -- reads the LAST TRACE's own flux_grid (ui.traceResult.flux_grid,
// always present for a flat receiver -- api.js's buildTraceRequest always
// sets include_flux_grid). Math/painting come from ../aperture.js, shared
// verbatim with tabs/analysis.js's own aperture (see that module's header);
// only the drag STATE below is this file's own, same "small stateful glue,
// not shared" call that module's header explains.
let apCenterUMm = null;
let apCenterVMm = null;
let apRadiusMm = null;
let apDrag = null; // {mode: "move"|"resize"} while a pointer drag is live
// v0.2 followups item 4 (owner-reported): the aperture canvas replaced the
// plain thumbnail's whole-canvas click-to-open, so a plain click landed
// inside the circle used to just silently start a "move" drag from
// pointerdown -- the owner's "I lost the obvious way in" report. Fix: a
// pointerdown no longer engages a drag by itself, it only records what
// WOULD be dragged (apPending, from the same resize-handle/circle-body hit
// test apHandlePointerDown always did); apHandlePointerMove only promotes
// that to a real drag (apDrag) once the pointer has moved past a small
// threshold, and apHandlePointerUp treats "pointer went up with no drag
// ever engaged" as a plain click and opens the full map -- so a click
// anywhere on the canvas (handle, circle, or open space) opens the map,
// and only an actual drag gesture, started from the handle or the circle
// body, edits the aperture.
let apPending = null; // {mode: "move"|"resize"|null, startX, startY} while the pointer is down but no drag has engaged yet
let apDockActions = null; // actions.onOpenFlux, captured once in build()
const AP_CLICK_THRESHOLD_PX = 4;
let apGrid = null; // the CURRENT flux_grid the canvas/readout below read
let apMetricsLike = null; // ui.traceResult itself -- carries centroid_mm/rms_radius_mm/power_w
// Which trace this aperture's center/radius were last set against -- a new
// trace lands a different footprint, so a circle left over from the
// previous one would silently describe the wrong picture. Compared against
// ui.traceTimestamp in render() below.
let apLastTraceTimestamp = undefined;

function apCurrentCenter() {
  if (apCenterUMm != null && apCenterVMm != null) return { u: apCenterUMm, v: apCenterVMm };
  return apertureDefaultCenter(apGrid, apMetricsLike);
}

function apCurrentRadius() {
  if (apRadiusMm != null) return apRadiusMm;
  return apertureDefaultRadiusMm(apGrid, apMetricsLike);
}

// Repaints just the aperture canvas + readout -- called from render() and,
// directly (bypassing the store), from the pointer-move handler below, same
// idiom as tabs/analysis.js's own paintIfVisible()-triggered repaint: a drag
// is not a store change, so nothing else would repaint it.
function paintDockAperture() {
  if (!apGrid) return;
  const center = apCurrentCenter();
  const radius = apCurrentRadius();
  paintApertureCanvas(els.apCanvas, apGrid, center.u, center.v, radius, 300);
  const { powerW, avgFluxWM2 } = apertureMetrics(apGrid, center.u, center.v, radius);
  const collectedW = apMetricsLike && apMetricsLike.power_w;
  els.apRadiusNum.textContent = (radius / 1000).toFixed(2) + " m";
  els.apPowerNum.textContent = fmtPower(powerW);
  els.apFracNum.textContent = collectedW ? ((100 * powerW) / collectedW).toFixed(1) + " %" : "—";
  els.apFluxNum.textContent = fmtFlux(avgFluxWM2 / 1000);
  // Spec §M.7: a live trace now carries its own dni_w_m2 (see api.js's
  // buildTraceRequest and heliostat.web.app's trace()/field_trace()
  // responses), closing the gap commit 45d6515 left open -- "concentration
  // deliberately omitted [in the dock] -- a live trace carries no DNI to
  // divide by".
  const dni = apMetricsLike && apMetricsLike.dni_w_m2;
  els.apConcNum.textContent = dni ? (avgFluxWM2 / dni).toFixed(0) + "×" : "—";
}

// Which affordance a point hits: "resize" for the handle square at the
// circle's edge, "move" for anywhere inside the circle body, null for open
// canvas -- the same two explicit affordances the aperture always offered,
// just no longer engaged straight from pointerdown (see apPending's
// comment above).
function apHitAffordance(canvas, x, y) {
  const center = apCurrentCenter();
  const radius = apCurrentRadius();
  const [cx, cy] = apertureDataToCanvas(apGrid, canvas, center.u, center.v);
  const pxPerMm = canvas.width / (apGrid.u_max_mm - apGrid.u_min_mm);
  const rPx = radius * pxPerMm;
  const dHandle = Math.hypot(x - (cx + rPx), y - cy);
  const dCenter = Math.hypot(x - cx, y - cy);
  if (dHandle <= 10) return "resize";
  if (dCenter <= rPx + 10) return "move";
  return null;
}

function apHandlePointerDown(e) {
  if (!apGrid) return;
  const canvas = e.currentTarget;
  const [x, y] = apertureCanvasEventPoint(canvas, e);
  // Not a drag yet -- just remember what a drag from here WOULD be, and
  // where it started. Deliberately no setPointerCapture/preventDefault
  // here: a plain click (down+up with no meaningful movement) must still
  // read as a normal click for apHandlePointerUp to act on.
  apPending = { mode: apHitAffordance(canvas, x, y), startX: x, startY: y };
}

function apHandlePointerMove(e) {
  if (!apGrid) return;
  const canvas = e.currentTarget;
  const [x, y] = apertureCanvasEventPoint(canvas, e);
  if (!apDrag) {
    if (!apPending) {
      // Hover, not a press: swap the cursor between "click to open" and
      // "grab" so the two affordances (handle/circle) still look
      // draggable before the user commits to either gesture.
      canvas.style.cursor = apHitAffordance(canvas, x, y) ? "grab" : "pointer";
      return;
    }
    const moved = Math.hypot(x - apPending.startX, y - apPending.startY);
    if (moved <= AP_CLICK_THRESHOLD_PX || !apPending.mode) return;
    // Crossed the click-vs-drag threshold on an explicit affordance --
    // engage the drag now (not at pointerdown), so a plain click that never
    // crosses this threshold falls through to apHandlePointerUp's "open
    // the full map" instead.
    apDrag = { mode: apPending.mode };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  }
  const [uMm, vMm] = apertureCanvasToData(apGrid, canvas, x, y);
  if (apDrag.mode === "move") {
    apCenterUMm = clampToGridAxis(apGrid, uMm, "u");
    apCenterVMm = clampToGridAxis(apGrid, vMm, "v");
  } else {
    const center = apCurrentCenter();
    const rMm = Math.hypot(uMm - center.u, vMm - center.v);
    apRadiusMm = apertureClampRadius(apGrid, rMm);
  }
  paintDockAperture();
}

function apHandlePointerUp(e) {
  const wasDragging = !!apDrag;
  apDrag = null;
  try {
    e.currentTarget.releasePointerCapture(e.pointerId);
  } catch (err) {
    // Capture may already be gone (pointer left the window, etc.) -- the
    // drag is over either way.
  }
  if (!wasDragging && apPending && apDockActions) {
    // Pointer went down and back up without ever crossing the drag
    // threshold -- a plain click, wherever on the canvas it landed. Same
    // entry point the plain thumbnail's onClick used to give (see
    // thumbImg's own listener below) before the aperture replaced it.
    apDockActions.onOpenFlux();
  }
  apPending = null;
}

function build(container, dockContainer, actions) {
  container.innerHTML = "";
  container.className = "runbar";
  dockContainer.innerHTML = "";
  // build() only ever runs once (the `built` guard below) -- captured here
  // so apHandlePointerUp's plain-click case can reach it too.
  apDockActions = actions;

  const seg = document.createElement("div");
  seg.className = "seg";
  const fidelityBtns = {};
  for (const [key, label] of FIDELITY) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.title = FIDELITY_TOOLTIPS[key];
    btn.addEventListener("click", () => store.set("ui.fidelity", key));
    seg.appendChild(btn);
    fidelityBtns[key] = btn;
  }
  container.appendChild(seg);

  const raysRow = document.createElement("div");
  raysRow.className = "frow";
  raysRow.style.margin = "0";
  const raysLabel = document.createElement("label");
  raysLabel.textContent = "Rays";
  raysLabel.style.flex = "0 0 auto";
  const raysInput = document.createElement("input");
  raysInput.type = "number";
  raysInput.className = "val";
  raysInput.style.width = "84px";
  raysInput.placeholder = "120000";
  raysInput.min = "100";
  raysRow.title = "Number of Monte Carlo rays traced -- more rays reduce noise (falls as 1/√rays) at the cost of trace time.";
  raysInput.addEventListener("input", () => {
    const v = parseInt(raysInput.value, 10);
    store.set("ui.mcRays", Number.isFinite(v) && v > 0 ? v : null);
  });
  raysRow.appendChild(raysLabel);
  raysRow.appendChild(raysInput);
  container.appendChild(raysRow);

  const runBtn = document.createElement("div");
  runBtn.className = "btn primary";
  runBtn.textContent = "Run trace";
  runBtn.addEventListener("click", () => {
    if (runBtn.classList.contains("disabled-link")) return;
    actions.onRunTrace();
  });
  container.appendChild(runBtn);

  // The cancel control and live progress hint for a running field trace used
  // to live here, but this bar only ever shows on the 3D View tab (see
  // main.js's renderTabs) -- a field trace can run for minutes, so both
  // moved to #tracebar (main.js's renderTraceBar), which stays visible from
  // every tab. This bar keeps only the Run button's own "Running…" label.

  const staleChip = document.createElement("div");
  staleChip.className = "stalechip";
  staleChip.textContent = "Edited since last run — results below are stale";
  staleChip.hidden = true;
  container.appendChild(staleChip);

  const traceErr = document.createElement("div");
  traceErr.className = "fielderr";
  traceErr.hidden = true;
  container.appendChild(traceErr);

  // Everything below lives in the results dock (docs/ui-spec-v0.2.md §N,
  // mockup M18b), not the trace bar -- see this file's header comment.
  const results = document.createElement("div");
  results.className = "results";

  function metric(label) {
    const m = document.createElement("div");
    m.className = "metric";
    const num = document.createElement("div");
    num.className = "num";
    const lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.textContent = label;
    m.appendChild(num);
    m.appendChild(lbl);
    results.appendChild(m);
    return num;
  }
  const peakNum = metric("peak flux");
  const meanNum = metric("mean flux");
  const interceptNum = metric("intercept");

  const thumbWrap = document.createElement("div");
  thumbWrap.className = "fluxthumb";
  const thumbImg = document.createElement("img");
  thumbImg.style.width = "100%";
  thumbImg.style.height = "100%";
  thumbImg.style.objectFit = "cover";
  thumbImg.style.cursor = "pointer";
  thumbImg.addEventListener("click", () => actions.onOpenFlux());
  thumbWrap.appendChild(thumbImg);
  results.appendChild(thumbWrap);

  // v0.2 followups item 3, mockup M17: the results dock's own analysis
  // aperture -- a draggable/resizable circle on the LAST TRACE's own
  // flux_grid, live power-in-aperture/avg-flux readout. Shown INSTEAD of
  // thumbWrap above whenever the trace has a flux_grid on a flat receiver
  // (the only case the shared aperture math supports today -- see
  // ../aperture.js's apertureReceiverIsFlat); thumbWrap stays the fallback
  // for a curved receiver or a receiver-less scene. See render()'s own
  // toggle between the two.
  const apWrap = document.createElement("div");
  apWrap.className = "apwrap";
  apWrap.hidden = true;
  const apCanvas = document.createElement("canvas");
  apCanvas.className = "apcanvas";
  apCanvas.addEventListener("pointerdown", apHandlePointerDown);
  apCanvas.addEventListener("pointermove", apHandlePointerMove);
  apCanvas.addEventListener("pointerup", apHandlePointerUp);
  apCanvas.addEventListener("pointercancel", apHandlePointerUp);
  apWrap.appendChild(apCanvas);
  const apCaption = document.createElement("div");
  apCaption.className = "apcaption";
  apCaption.appendChild(
    document.createTextNode("Click to open the full map · drag the dashed circle to move it, its square to resize. ")
  );
  // The aperture canvas replaces the plain thumbnail's own click-to-open
  // handler (thumbImg below), so this link keeps that entry point reachable
  // -- the full overlay is still the only place with the Receiver |
  // Secondary | Field selector and the Secondary/Field readouts.
  const apOpenLink = document.createElement("a");
  apOpenLink.href = "#";
  apOpenLink.textContent = "Open full map →";
  apOpenLink.addEventListener("click", (e) => {
    e.preventDefault();
    actions.onOpenFlux();
  });
  apCaption.appendChild(apOpenLink);
  apWrap.appendChild(apCaption);
  const apReadout = document.createElement("div");
  apReadout.className = "apreadout";
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
    apReadout.appendChild(row);
    return num;
  }
  const apRadiusNum = apRow("aperture radius");
  const apPowerNum = apRow("power within");
  const apFracNum = apRow("of collected");
  const apFluxNum = apRow("avg flux");
  const apConcNum = apRow("avg concentration");
  apWrap.appendChild(apReadout);
  results.appendChild(apWrap);

  // Flux-map axis convention: only shown for a curved receiver, where u/v
  // aren't plain x/y (unrolled arc length + height/slant instead) -- see
  // heliostat.web.scene's receiver dict, kind "cylinder"/"frustum".
  const axisCaption = document.createElement("div");
  axisCaption.className = "hint";
  axisCaption.style.marginTop = "2px";
  axisCaption.hidden = true;
  results.appendChild(axisCaption);

  // v0.2 followups item 2, mockup M16: "Flux overlay" -- owner's own label,
  // not mockup M16's "drape" (docs/ui-spec-v0.2.md §M.3's toggle, wired to
  // ui.receiverFluxOverlay; main.js's applyFluxOverlayVisibility reads it).
  // Default ON, so a fresh page load matches the drape's old always-on
  // behavior until someone actually turns it off.
  //
  // v0.2 followups item 3: the SAME checkbox now also gates the secondary
  // mirror's own drape (main.js's applyFluxOverlayVisibility) rather than
  // getting a second toggle -- one "Flux overlay" idea, on-screen label
  // already generic, applying to whichever draped surfaces the current
  // optics actually has.
  const fluxOverlayToggleRow = document.createElement("label");
  fluxOverlayToggleRow.className = "fluxoverlaytoggle";
  fluxOverlayToggleRow.title = "Paint the traced flux map onto the receiver (and secondary, if any) surface in the 3D scene.";
  const fluxOverlayToggleInput = document.createElement("input");
  fluxOverlayToggleInput.type = "checkbox";
  fluxOverlayToggleInput.checked = true;
  fluxOverlayToggleInput.addEventListener("change", () => {
    store.set("ui.receiverFluxOverlay", fluxOverlayToggleInput.checked);
  });
  fluxOverlayToggleRow.appendChild(fluxOverlayToggleInput);
  fluxOverlayToggleRow.appendChild(document.createTextNode(" Flux overlay"));
  results.appendChild(fluxOverlayToggleRow);

  const stampWrap = document.createElement("div");
  const stamp = document.createElement("div");
  stamp.className = "stamp";
  const exportLink = document.createElement("a");
  exportLink.href = "#";
  exportLink.textContent = "Export flux CSV";
  exportLink.style.fontSize = "12px";
  exportLink.addEventListener("click", (e) => {
    e.preventDefault();
    actions.onExportCsv();
  });
  // docs/ui-spec-v0.2.md §D: the ANSYS-oriented FEA CSV grid, beside (never
  // instead of) the mm/kW-m2 export above -- same idiom, different file.
  const exportFeaLink = document.createElement("a");
  exportFeaLink.href = "#";
  exportFeaLink.textContent = "Export CSV for FEA";
  exportFeaLink.style.fontSize = "12px";
  exportFeaLink.style.marginLeft = "10px";
  exportFeaLink.addEventListener("click", (e) => {
    e.preventDefault();
    actions.onExportFeaCsv();
  });
  stampWrap.appendChild(stamp);
  stampWrap.appendChild(exportLink);
  stampWrap.appendChild(exportFeaLink);
  results.appendChild(stampWrap);

  dockContainer.appendChild(results);

  els = {
    fidelityBtns,
    raysRow,
    raysInput,
    runBtn,
    staleChip,
    traceErr,
    results,
    peakNum,
    meanNum,
    interceptNum,
    thumbWrap,
    thumbImg,
    apWrap,
    apCanvas,
    apRadiusNum,
    apPowerNum,
    apFracNum,
    apFluxNum,
    apConcNum,
    fluxOverlayToggleInput,
    axisCaption,
    stamp,
    exportLink,
    exportFeaLink,
  };
  built = true;
}

export function render(container, dockContainer, actions, ctx) {
  if (!built) build(container, dockContainer, actions);
  const ui = store.get("ui");

  for (const [key, btn] of Object.entries(els.fidelityBtns)) {
    btn.classList.toggle("active", key === ui.fidelity);
  }
  els.raysRow.style.display = ui.fidelity === "monte_carlo" ? "flex" : "none";
  if (!isFocused(els.raysInput)) {
    els.raysInput.value = ui.mcRays == null ? "" : String(ui.mcRays);
  }

  els.runBtn.textContent = ui.traceBusy ? "Running…" : "Run trace";
  els.runBtn.classList.toggle("disabled-link", ui.traceBusy);

  // The cancel control and live progress hint for a running field trace now
  // live in #tracebar (main.js's renderTraceBar), visible from every tab --
  // see the comment where they used to be built, above.

  els.traceErr.hidden = !ui.traceError;
  if (ui.traceError) els.traceErr.textContent = ui.traceError;

  els.fluxOverlayToggleInput.checked = ui.receiverFluxOverlay !== false;

  const data = ui.traceResult;
  els.staleChip.hidden = !(ui.staleResults && data);
  els.results.classList.toggle("stale", !!(ui.staleResults && data));
  els.exportLink.style.display = data ? "" : "none";
  els.exportFeaLink.style.display = data ? "" : "none";

  // v0.2 followups item 3: a fresh trace lands a different footprint, so a
  // circle carried over from the previous one would silently describe the
  // wrong picture -- reset to that trace's own sensible default the moment
  // a NEW trace timestamp shows up (not on every render(), which runs on
  // every store change including a plain drag).
  if (ui.traceTimestamp !== apLastTraceTimestamp) {
    apLastTraceTimestamp = ui.traceTimestamp;
    apCenterUMm = null;
    apCenterVMm = null;
    apRadiusMm = null;
    apDrag = null;
    apPending = null;
  }

  if (data) {
    const metrics = deriveMetrics(data);
    els.peakNum.textContent = metrics.peak == null ? "—" : fmt(metrics.peak, 1) + " kW/m²";
    els.meanNum.textContent = metrics.mean == null ? "—" : fmt(metrics.mean, 1) + " kW/m²";
    els.interceptNum.textContent = metrics.intercept == null ? "—" : fmt(metrics.intercept, 1) + " %";

    // v0.2 followups item 3: the interactive aperture replaces the plain
    // thumbnail whenever this trace has a flux_grid on a flat receiver --
    // the only case ../aperture.js's math supports (§M.4, "curved later if
    // wanted"). Otherwise the thumbnail is what there is to show.
    const flat = apertureReceiverIsFlat(store.get("doc"));
    if (data.flux_grid && flat) {
      apGrid = data.flux_grid;
      apMetricsLike = data;
      els.apWrap.hidden = false;
      els.thumbWrap.style.display = "none";
      paintDockAperture();
    } else {
      apGrid = null;
      apMetricsLike = null;
      els.apWrap.hidden = true;
      if (data.flux_png) {
        els.thumbImg.src = "data:image/png;base64," + data.flux_png;
        els.thumbWrap.style.display = "";
      } else {
        els.thumbWrap.style.display = "none";
      }
    }

    const receiverKind = data.scene && data.scene.receiver && data.scene.receiver.kind;
    if (receiverKind === "cylinder") {
      els.axisCaption.textContent = "u = arc length around receiver (seam at north) · v = height above centre";
      els.axisCaption.hidden = false;
    } else if (receiverKind === "frustum") {
      els.axisCaption.textContent = "u = arc length around receiver (seam at north) · v = slant distance from base";
      els.axisCaption.hidden = false;
    } else {
      els.axisCaption.hidden = true;
    }
    const when = ui.traceTimestamp ? new Date(ui.traceTimestamp).toLocaleTimeString() : "";
    // The fidelity that produced these numbers, not whichever one is
    // selected now -- switching the control must not relabel old results.
    const tracedAt = ui.traceFidelity || ui.fidelity;
    // Spec §M.7: state the DNI these numbers assumed, right beside the
    // fidelity that produced them -- a published/screenshotted number
    // should never leave that implicit.
    const dniNote = data.dni_note ? ` · DNI: ${data.dni_note}` : "";
    els.stamp.textContent = when ? `traced ${when} · ${tracedAt}${dniNote}` : "";
  } else {
    els.peakNum.textContent = "—";
    els.meanNum.textContent = "—";
    els.interceptNum.textContent = "—";
    els.thumbWrap.style.display = "none";
    apGrid = null;
    apMetricsLike = null;
    els.apWrap.hidden = true;
    els.axisCaption.hidden = true;
    els.stamp.textContent = "";
  }
}
