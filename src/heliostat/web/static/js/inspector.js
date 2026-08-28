// Floating in-scene inspector: shown whenever ui.selection is set, editing
// the exact same store paths as the matching sidebar stage via
// ./fields.js's shared descriptors -- no separate edit pathway. main.js
// re-renders this on every store change, the same as the sidebar panels;
// it also needs the last /api/scene/geometry response (to look up a
// selected heliostat's position for its distance-from-axis readout),
// which isn't store state, so main.js passes it in as `ctx.geometry`.
import { store } from "./store.js";
import {
  numberRow,
  setVal,
  segButton,
  sectionHeaderRow,
  HELIOSTAT_SURFACE_OPTIONS,
  RECEIVER_FIELD_TABLE,
  OPTICS_LABELS,
  RECEIVER_TYPE_OPTIONS,
  receiverFieldVisible,
  SUN_FIELDS,
  apertureMissMessage,
  apertureSummaryText,
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

  // Locks the Heliostat Shape tab's preview to this exact heliostat
  // rather than its own median-heliostat default.
  const viewShapeLink = document.createElement("a");
  viewShapeLink.href = "#";
  viewShapeLink.className = "hint";
  viewShapeLink.style.display = "block";
  viewShapeLink.style.marginBottom = "8px";
  viewShapeLink.textContent = "View shape →";
  viewShapeLink.addEventListener("click", (e) => {
    e.preventDefault();
    const sel = store.get("ui.selection");
    if (sel && sel.kind === "heliostat") store.set("ui.shapeHeliostatId", sel.id);
    store.set("ui.tab", "shape");
  });
  helioWrap.appendChild(viewShapeLink);

  const helioSummary = document.createElement("div");
  helioSummary.className = "summary";
  helioSummary.style.marginTop = "0";
  helioWrap.appendChild(helioSummary);

  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const surfaceBtns = {};
  const SURFACE_TOOLTIPS = {
    twisting: "Re-solves each facet's figure as the sun moves, so it stays perfectly focused at every instant.",
    spherical: "Freezes one figure — a long focal gives a weakly focusing, not-quite-flat facet that no longer re-solves with the sun.",
    flat: "No curvature by default — a true flat panel, though a facet grid can still be given a gentle fixed curvature.",
  };
  for (const [key, label] of HELIOSTAT_SURFACE_OPTIONS) {
    surfaceBtns[key] = segButton(
      surfaceSeg,
      label,
      key === "twisting",
      () => store.set("doc.design.surface", key),
      SURFACE_TOOLTIPS[key]
    );
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

  // prime_focus only -- receiver_type is a string, so it gets its own
  // segmented control rather than a RECEIVER_FIELD_TABLE/numberRow entry.
  const receiverTypeSeg = document.createElement("div");
  receiverTypeSeg.className = "seg";
  const receiverTypeBtns = {};
  for (const [key, label] of RECEIVER_TYPE_OPTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => store.set("doc.opticsParams.prime_focus.receiver_type", key));
    receiverTypeSeg.appendChild(btn);
    receiverTypeBtns[key] = btn;
  }
  opticsWrap.appendChild(receiverTypeSeg);

  const opticsFieldsByType = {};
  const opticsInputsByType = {};
  // Row element per field key, keyed by optics -- lets prime_focus's
  // cylinder/frustum-only rows (receiverFieldVisible) hide independently of
  // the rest of their own optics block.
  const opticsRowsByType = {};
  const opticsWarnByType = {};
  for (const [optics, fields] of Object.entries(RECEIVER_FIELD_TABLE)) {
    const wrap = document.createElement("div");
    const inputs = {};
    const rows = {};
    let warnBox = null;
    for (const field of fields) {
      // See panels/receiver.js's identical handling -- keeps the floating
      // inspector's fields visually identical to the sidebar's.
      if (field.sectionHeader) sectionHeaderRow(wrap, field.sectionHeader, field.sectionHeaderBadge);
      // secondary_error_map (§E2) has no numberRow -- a grid object isn't a
      // number -- and no compact-inspector equivalent of the sidebar's
      // import-chip UI; this floating panel skips it and shows only the
      // numeric warp fields below it, keeping the inspector's promise of
      // "the full control surface is Design's" (§N) for anything bigger
      // than a plain number.
      if (field.custom) continue;
      const input = numberRow(wrap, field);
      inputs[field.key] = input;
      rows[field.key] = input.parentElement;
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
    opticsRowsByType[optics] = rows;
    opticsWarnByType[optics] = warnBox;
  }

  // Red geometry errors show here exactly as in the sidebar's Receiver &
  // Tower stage.
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
    helioSummary,
    surfaceBtns,
    opticsWrap,
    opticsBtns,
    receiverTypeSeg,
    receiverTypeBtns,
    opticsFieldsByType,
    opticsInputsByType,
    opticsRowsByType,
    opticsWarnByType,
    opticsErrBox,
    sunWrap,
    sunInputs,
  };
  built = true;
}

// The heliostat's own x/y from the last geometry response, as a "distance
// from the tower axis" readout, shown in the header.
function heliostatDistanceLabel(id, geometry) {
  const list = (geometry && geometry.heliostats) || [];
  const h = list.find((x) => x.id === id);
  if (!h) return "";
  const rM = Math.hypot(h.x_mm, h.y_mm) / 1000;
  return `r ${rM.toFixed(2)} m`;
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

    els.helioSummary.textContent = apertureSummaryText(doc);
    const surface = doc.design.surface;
    for (const [key, btn] of Object.entries(els.surfaceBtns)) btn.classList.toggle("active", key === surface);
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

    els.receiverTypeSeg.style.display = optics === "prime_focus" ? "" : "none";
    if (optics === "prime_focus") {
      const rType = params.receiver_type || "flat";
      for (const [key, btn] of Object.entries(els.receiverTypeBtns)) {
        btn.classList.toggle("active", key === rType);
      }
      for (const field of RECEIVER_FIELD_TABLE.prime_focus) {
        els.opticsRowsByType.prime_focus[field.key].style.display = receiverFieldVisible(field, params) ? "" : "none";
      }
    }

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
