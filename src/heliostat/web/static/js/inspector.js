// Floating in-scene inspector (docs/ui-spec.md 2.4): shown whenever
// ui.selection is set, editing the exact same store paths as the matching
// sidebar stage via ./fields.js's shared descriptors -- "no separate edit
// pathway, no hidden overrides, no Apply". main.js re-renders this on
// every store change, the same as the four sidebar panels; it also needs
// the last /api/scene/geometry response (to look up a selected
// heliostat's position for its distance-from-axis readout), which isn't
// store state, so main.js passes it in as `ctx.geometry`.
import { store } from "./store.js";
import {
  numberRow,
  setVal,
  segButton,
  HELIOSTAT_RECT_FIELDS,
  HELIOSTAT_GRID_FIELDS,
  HELIOSTAT_SURFACE_OPTIONS,
  RECEIVER_FIELD_TABLE,
  OPTICS_LABELS,
  SUN_FIELDS,
  apertureMissMessage,
} from "./fields.js";

const OPTICS_NAME = Object.fromEntries(OPTICS_LABELS);

let built = false;
let els = {};

function build(container) {
  container.innerHTML = "";
  container.className = "inspector";

  const close = document.createElement("div");
  close.className = "close";
  close.textContent = "×";
  close.title = "Close";
  close.addEventListener("click", () => store.set("ui.selection", null));
  container.appendChild(close);

  const h3 = document.createElement("h3");
  container.appendChild(h3);

  const sub = document.createElement("div");
  sub.className = "sub";
  container.appendChild(sub);

  const body = document.createElement("div");
  container.appendChild(body);

  // -- heliostat: the Heliostat stage's own fields ------------------------
  const helioWrap = document.createElement("div");

  const typeSeg = document.createElement("div");
  typeSeg.className = "seg";
  const rectBtn = segButton(typeSeg, "Rectangle", true, () => store.set("doc.design.type", "rect"));
  const gridBtn = segButton(typeSeg, "Facet grid", false, () => store.set("doc.design.type", "grid"));
  helioWrap.appendChild(typeSeg);

  const rectFields = document.createElement("div");
  const rectInputs = {};
  for (const field of HELIOSTAT_RECT_FIELDS) rectInputs[field.key] = numberRow(rectFields, field);
  helioWrap.appendChild(rectFields);

  const gridFields = document.createElement("div");
  const gridInputs = {};
  for (const field of HELIOSTAT_GRID_FIELDS) gridInputs[field.key] = numberRow(gridFields, field);
  helioWrap.appendChild(gridFields);

  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const surfaceBtns = {};
  for (const [key, label] of HELIOSTAT_SURFACE_OPTIONS) {
    surfaceBtns[key] = segButton(surfaceSeg, label, key === "twisting", () => store.set("doc.design.surface", key));
  }
  helioWrap.appendChild(surfaceSeg);

  body.appendChild(helioWrap);

  // -- secondary / receiver: the Receiver & Tower stage's own fields ------
  const opticsWrap = document.createElement("div");

  const opticsSeg = document.createElement("div");
  opticsSeg.className = "seg";
  const opticsBtns = {};
  for (const [key, label] of OPTICS_LABELS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => store.set("doc.optics", key));
    opticsSeg.appendChild(btn);
    opticsBtns[key] = btn;
  }
  opticsWrap.appendChild(opticsSeg);

  const opticsFieldsByType = {};
  const opticsInputsByType = {};
  const opticsWarnByType = {};
  for (const [optics, fields] of Object.entries(RECEIVER_FIELD_TABLE)) {
    const wrap = document.createElement("div");
    const inputs = {};
    let warnBox = null;
    for (const field of fields) {
      inputs[field.key] = numberRow(wrap, field);
      if (field.key === "aperture_radius_mm") {
        warnBox = document.createElement("div");
        warnBox.className = "fieldwarn";
        warnBox.hidden = true;
        wrap.appendChild(warnBox);
      }
    }
    opticsWrap.appendChild(wrap);
    opticsFieldsByType[optics] = wrap;
    opticsInputsByType[optics] = inputs;
    opticsWarnByType[optics] = warnBox;
  }

  // Red geometry errors show here exactly as in the sidebar's Receiver &
  // Tower stage -- docs/ui-spec.md 2.4: "Warnings and errors appear in
  // both places identically."
  const opticsErrBox = document.createElement("div");
  opticsErrBox.className = "fielderr";
  opticsErrBox.hidden = true;
  opticsWrap.appendChild(opticsErrBox);

  body.appendChild(opticsWrap);

  // -- sun: the Sun stage's own fields --------------------------------------
  const sunWrap = document.createElement("div");
  const sunInputs = {};
  for (const field of SUN_FIELDS) sunInputs[field.key] = numberRow(sunWrap, field);
  body.appendChild(sunWrap);

  const foot = document.createElement("div");
  foot.className = "foot";
  foot.textContent = "Edits apply live and sync with the sidebar — there is no separate Apply.";
  body.appendChild(foot);

  els = {
    h3,
    sub,
    helioWrap,
    rectBtn,
    gridBtn,
    rectFields,
    gridFields,
    rectInputs,
    gridInputs,
    surfaceBtns,
    opticsWrap,
    opticsBtns,
    opticsFieldsByType,
    opticsInputsByType,
    opticsWarnByType,
    opticsErrBox,
    sunWrap,
    sunInputs,
  };
  built = true;
}

// The heliostat's own x/y from the last geometry response, as a "distance
// from the tower axis" readout (docs/ui-spec.md 3's "r 45.2 m" convention,
// reused here for the header per the phase-3b brief).
function heliostatDistanceLabel(id, geometry) {
  const list = (geometry && geometry.heliostats) || [];
  const h = list.find((x) => x.id === id);
  if (!h) return "";
  const rM = Math.hypot(h.x_mm, h.y_mm) / 1000;
  return `r ${rM.toFixed(1)} m`;
}

export function render(container, ctx) {
  if (!built) build(container);
  const sel = store.get("ui.selection");
  container.hidden = !sel;
  if (!sel) return; // [hidden] { display: none !important } also keeps this out of the click/orbit path

  const doc = store.get("doc");
  const ui = store.get("ui");
  const geometry = (ctx && ctx.geometry) || null;

  els.helioWrap.style.display = sel.kind === "heliostat" ? "" : "none";
  els.opticsWrap.style.display = sel.kind === "secondary" || sel.kind === "receiver" ? "" : "none";
  els.sunWrap.style.display = sel.kind === "sun" ? "" : "none";

  if (sel.kind === "heliostat") {
    els.h3.textContent = `Heliostat H-${sel.id}`;
    els.sub.textContent = heliostatDistanceLabel(sel.id, geometry) || "Selected in scene";

    const type = doc.design.type;
    els.rectBtn.classList.toggle("active", type === "rect");
    els.gridBtn.classList.toggle("active", type === "grid");
    els.rectFields.style.display = type === "rect" ? "" : "none";
    els.gridFields.style.display = type === "grid" ? "" : "none";
    const surface = doc.design.surface;
    for (const [key, btn] of Object.entries(els.surfaceBtns)) btn.classList.toggle("active", key === surface);

    if (type === "rect") {
      const p = doc.designParams.rect;
      setVal(els.rectInputs.width_mm, p.width_mm);
      setVal(els.rectInputs.height_mm, p.height_mm);
    } else {
      const p = doc.designParams.grid;
      setVal(els.gridInputs.n_u, p.n_u);
      setVal(els.gridInputs.n_v, p.n_v);
      setVal(els.gridInputs.facet_w_mm, p.facet_w_mm);
      setVal(els.gridInputs.facet_h_mm, p.facet_h_mm);
      setVal(els.gridInputs.gap_mm, p.gap_mm);
    }
  } else if (sel.kind === "secondary" || sel.kind === "receiver") {
    const optics = doc.optics;
    const name = OPTICS_NAME[optics] || optics;
    els.h3.textContent = sel.kind === "secondary" ? `Secondary — ${name}` : `Receiver — ${name}`;
    els.sub.textContent = "Selected in scene · same fields as Receiver & Tower";

    for (const [key, btn] of Object.entries(els.opticsBtns)) btn.classList.toggle("active", key === optics);
    for (const [key, wrap] of Object.entries(els.opticsFieldsByType)) {
      wrap.style.display = key === optics ? "" : "none";
    }

    const params = doc.opticsParams[optics];
    for (const [key, input] of Object.entries(els.opticsInputsByType[optics])) setVal(input, params[key]);

    const warnBox = els.opticsWarnByType[optics];
    if (warnBox) {
      const msg = apertureMissMessage(ui.miss);
      warnBox.hidden = !msg;
      if (msg) warnBox.textContent = msg;
    }

    const err = ui.geometryError;
    if (err && err.forReceiver) {
      els.opticsErrBox.hidden = false;
      els.opticsErrBox.textContent = err.message;
    } else {
      els.opticsErrBox.hidden = true;
    }
  } else if (sel.kind === "sun") {
    els.h3.textContent = "Sun";
    els.sub.textContent = "Selected in scene · same fields as Sun";
    setVal(els.sunInputs.az, doc.sun.az);
    setVal(els.sunInputs.el, doc.sun.el);
  }
}
