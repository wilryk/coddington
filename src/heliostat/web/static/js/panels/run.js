// Run bar: fidelity, Run button, results strip (docs/ui-spec.md 2.5).
// Trace-shaped network calls live in main.js (this module only builds DOM
// and calls back into the `actions` it is given), so this file stays a
// pure view over the store plus those callbacks.
import { store } from "../store.js";

const FIDELITY = [
  ["ultra_fast", "Ultra fast"],
  ["fast_accurate", "Fast accurate"],
  ["monte_carlo", "Monte Carlo"],
];

let built = false;
let els = {};

function isFocused(el) {
  return el && document.activeElement === el;
}

function fmt(x, digits) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return x.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

// The response carries peak_flux_kw_m2, power_w, incident_power_w and
// per-stage ray counters (heliostat.web.app's /api/trace and
// /api/field/trace) but no literal "mean flux" or "intercept efficiency"
// field. Mean flux is derived from power over the receiver window's own
// area (optics_resolved's window_half_u/v_mm, echoed by both endpoints);
// intercept is power/incident for the cone backends, or in_window/
// hit_secondary from the counters for Monte Carlo, which carries no
// incident_power_w. Both are judgment calls, noted in the build report.
function deriveMetrics(data) {
  const out = { peak: null, mean: null, intercept: null };
  if (data.peak_flux_kw_m2 != null) out.peak = data.peak_flux_kw_m2;
  const resolved = data.optics_resolved || {};
  const halfU = resolved.window_half_u_mm;
  const halfV = resolved.window_half_v_mm;
  if (halfU && halfV && data.power_w != null) {
    const areaM2 = ((halfU * 2) / 1000) * ((halfV * 2) / 1000);
    if (areaM2 > 0) out.mean = data.power_w / areaM2 / 1000;
  }
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

function build(container, actions) {
  container.innerHTML = "";
  container.className = "runbar";

  const seg = document.createElement("div");
  seg.className = "seg";
  const fidelityBtns = {};
  for (const [key, label] of FIDELITY) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
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

  // Only a field trace runs as a cancellable job (main.js's runFieldTraceJob)
  // -- a single-heliostat trace is one request/response with nothing to
  // cancel, so this stays hidden for it (see render(), keyed on
  // ui.traceProgress being non-null).
  const cancelBtn = document.createElement("div");
  cancelBtn.className = "btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.hidden = true;
  cancelBtn.addEventListener("click", () => {
    if (cancelBtn.classList.contains("disabled-link")) return;
    cancelBtn.classList.add("disabled-link");
    actions.onCancelTrace();
  });
  container.appendChild(cancelBtn);

  // While a field trace's job is running this shows its live progress
  // (done/total heliostats + ETA, from heliostat.web.jobs' Job.snapshot);
  // otherwise hidden -- there is no honest pre-run estimate once the trace
  // is parallel, since wall clock now depends on core count.
  const costHint = document.createElement("div");
  costHint.className = "hint";
  costHint.style.margin = "0 0 0 4px";
  costHint.hidden = true;
  container.appendChild(costHint);

  const staleChip = document.createElement("div");
  staleChip.className = "stalechip";
  staleChip.textContent = "Edited since last run — results below are stale";
  staleChip.hidden = true;
  container.appendChild(staleChip);

  const traceErr = document.createElement("div");
  traceErr.className = "fielderr";
  traceErr.hidden = true;
  container.appendChild(traceErr);

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
  stampWrap.appendChild(stamp);
  stampWrap.appendChild(exportLink);
  results.appendChild(stampWrap);

  container.appendChild(results);

  els = {
    costHint,
    fidelityBtns,
    raysRow,
    raysInput,
    runBtn,
    cancelBtn,
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
  };
  built = true;
}

export function render(container, actions, ctx) {
  if (!built) build(container, actions);
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

  // A field trace runs as a cancellable background job (main.js's
  // runFieldTraceJob); a single-heliostat trace is one plain request with no
  // job behind it to cancel. ui.traceProgress is only ever set for the
  // former, so it doubles as "is this run cancellable".
  const progress = ui.traceProgress;
  const cancellable = ui.traceBusy && !!progress;
  els.cancelBtn.hidden = !cancellable;
  if (!cancellable) els.cancelBtn.classList.remove("disabled-link");

  // Once the trace is parallel, wall clock depends on core count -- there is
  // no honest pre-run estimate the way a fixed seconds-per-heliostat number
  // was. So this now shows only the running job's own live progress
  // (heliostat.web.jobs' Job.snapshot: done/total heliostats, detail, ETA),
  // nothing before Run is pressed.
  if (cancellable) {
    let label = progress.detail || `${progress.done} / ${progress.total} heliostats`;
    if (progress.eta_s != null) {
      const etaS = Math.round(progress.eta_s);
      label += etaS >= 90 ? `, about ${Math.round(etaS / 60)} min left` : `, about ${etaS}s left`;
    }
    els.costHint.textContent = label;
    els.costHint.hidden = false;
  } else {
    els.costHint.hidden = true;
  }

  els.traceErr.hidden = !ui.traceError;
  if (ui.traceError) els.traceErr.textContent = ui.traceError;

  const data = ui.traceResult;
  els.staleChip.hidden = !(ui.staleResults && data);
  els.results.classList.toggle("stale", !!(ui.staleResults && data));
  els.exportLink.style.display = data ? "" : "none";

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
