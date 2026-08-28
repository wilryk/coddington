// Trace bar (fidelity, Run button, stale/error chips) + results dock (the
// flux thumbnail -> full overlay, peak/mean/intercept, CSV exports)
// (docs/ui-spec.md 2.5; docs/ui-spec-v0.2.md §N/mockup M18b: the same
// content, just split across two containers now -- a condensed bar above
// the 3D scene and a ~380px dock beside it, instead of one full-width
// bottom bar). Trace-shaped network calls live in main.js (this module only
// builds DOM and calls back into the `actions` it is given), so this file
// stays a pure view over the store plus those callbacks.
import { store } from "../store.js";

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

function build(container, dockContainer, actions) {
  container.innerHTML = "";
  container.className = "runbar";
  dockContainer.innerHTML = "";

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

  // Flux-map axis convention: only shown for a curved receiver, where u/v
  // aren't plain x/y (unrolled arc length + height/slant instead) -- see
  // heliostat.web.scene's receiver dict, kind "cylinder"/"frustum".
  const axisCaption = document.createElement("div");
  axisCaption.className = "hint";
  axisCaption.style.marginTop = "2px";
  axisCaption.hidden = true;
  results.appendChild(axisCaption);

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

  const data = ui.traceResult;
  els.staleChip.hidden = !(ui.staleResults && data);
  els.results.classList.toggle("stale", !!(ui.staleResults && data));
  els.exportLink.style.display = data ? "" : "none";
  els.exportFeaLink.style.display = data ? "" : "none";

  if (data) {
    const metrics = deriveMetrics(data);
    els.peakNum.textContent = metrics.peak == null ? "—" : fmt(metrics.peak, 1) + " kW/m²";
    els.meanNum.textContent = metrics.mean == null ? "—" : fmt(metrics.mean, 1) + " kW/m²";
    els.interceptNum.textContent = metrics.intercept == null ? "—" : fmt(metrics.intercept, 1) + " %";
    if (data.flux_png) {
      els.thumbImg.src = "data:image/png;base64," + data.flux_png;
      els.thumbWrap.style.display = "";
    } else {
      els.thumbWrap.style.display = "none";
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
    els.stamp.textContent = when ? `traced ${when} · ${tracedAt}` : "";
  } else {
    els.peakNum.textContent = "—";
    els.meanNum.textContent = "—";
    els.interceptNum.textContent = "—";
    els.thumbWrap.style.display = "none";
    els.axisCaption.hidden = true;
    els.stamp.textContent = "";
  }
}
