// Receiver & Tower stage: Prime focus / Axicon / Cassegrain, each with its
// own honest labels (docs/ui-spec.md 2.2's table -- no shared "tower
// height" alias). Field names match heliostat.web.app's *Optics models
// exactly, so what the user types is what /api/scene/geometry and
// /api/trace read back under `optics_params`.
import { store } from "../store.js";

let built = false;
let els = {};

const FIELD_TABLE = {
  prime_focus: [
    ["focus_height_mm", "Focus height (mm)"],
    ["window_half_u_mm", "Window ½ w (mm)"],
    ["window_half_v_mm", "Window ½ h (mm)"],
  ],
  axicon: [
    ["apex_height_mm", "Apex height (mm)"],
    ["half_angle_deg", "Half angle (°)"],
    ["aperture_radius_mm", "Aperture radius (mm)"],
    ["receiver_z_mm", "Receiver height (mm)"],
    ["window_half_u_mm", "Window ½ w (mm)"],
    ["window_half_v_mm", "Window ½ h (mm)"],
  ],
  cassegrain: [
    ["vertex_z_mm", "Secondary vertex height (mm)"],
    ["focus_height_mm", "Primary focus height (mm)"],
    ["receiver_z_mm", "Receiver height (mm)"],
    ["aperture_radius_mm", "Aperture radius (mm)"],
    ["window_half_u_mm", "Window ½ w (mm)"],
    ["window_half_v_mm", "Window ½ h (mm)"],
  ],
};

const OPTICS_LABELS = [
  ["prime_focus", "Prime focus"],
  ["axicon", "Axicon"],
  ["cassegrain", "Cassegrain"],
];

function isFocused(el) {
  return el && document.activeElement === el;
}

function setVal(input, value) {
  if (isFocused(input)) return;
  const s = value == null ? "" : String(value);
  if (input.value !== s) input.value = s;
}

function numberRow(parent, label, key) {
  const row = document.createElement("div");
  row.className = "frow";
  const lab = document.createElement("label");
  lab.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.className = "val";
  input.dataset.key = key;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v)) {
      const optics = store.get("doc.optics");
      store.set(`doc.opticsParams.${optics}.${key}`, v);
    }
  });
  row.appendChild(lab);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

function build(container) {
  container.innerHTML = "";
  container.className = "stage";

  const head = document.createElement("div");
  head.className = "stagehead";
  const chev = document.createElement("span");
  chev.className = "chev";
  chev.textContent = "▾";
  const h2 = document.createElement("h2");
  h2.textContent = "Receiver & Tower";
  head.appendChild(chev);
  head.appendChild(h2);
  head.addEventListener("click", () => {
    const ui = store.get("ui");
    store.set("ui.expanded.receiver", !ui.expanded.receiver);
  });

  const body = document.createElement("div");
  body.className = "stagebody";

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
  body.appendChild(opticsSeg);

  const fieldsByOptics = {};
  const inputsByOptics = {};
  for (const [optics, fields] of Object.entries(FIELD_TABLE)) {
    const wrap = document.createElement("div");
    const inputs = {};
    for (const [key, label] of fields) {
      inputs[key] = numberRow(wrap, label, key);
    }
    body.appendChild(wrap);
    fieldsByOptics[optics] = wrap;
    inputsByOptics[optics] = inputs;
  }

  const errorBox = document.createElement("div");
  errorBox.className = "fielderr";
  errorBox.hidden = true;
  body.appendChild(errorBox);

  const actions = document.createElement("div");
  actions.className = "stageactions";
  const saveBtn = document.createElement("div");
  saveBtn.className = "btn disabled-link";
  saveBtn.textContent = "Save config…";
  saveBtn.title = "Library -- coming in a later phase";
  const swapBtn = document.createElement("div");
  swapBtn.className = "btn disabled-link";
  swapBtn.textContent = "Swap from library…";
  swapBtn.title = "Library -- coming in a later phase";
  actions.appendChild(saveBtn);
  actions.appendChild(swapBtn);
  body.appendChild(actions);

  container.appendChild(head);
  container.appendChild(body);

  els = { chev, body, opticsBtns, fieldsByOptics, inputsByOptics, errorBox };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const optics = doc.optics;

  els.body.style.display = ui.expanded.receiver ? "" : "none";
  els.chev.style.transform = ui.expanded.receiver ? "rotate(90deg)" : "";

  for (const [key, btn] of Object.entries(els.opticsBtns)) {
    btn.classList.toggle("active", key === optics);
  }
  for (const [key, wrap] of Object.entries(els.fieldsByOptics)) {
    wrap.style.display = key === optics ? "" : "none";
  }

  const params = doc.opticsParams[optics];
  for (const [key, input] of Object.entries(els.inputsByOptics[optics])) {
    setVal(input, params[key]);
  }

  const err = ui.geometryError;
  if (err && err.forReceiver) {
    els.errorBox.hidden = false;
    els.errorBox.textContent = err.message;
    els.errorBox.className = err.severity === "warn" ? "fieldwarn" : "fielderr";
  } else {
    els.errorBox.hidden = true;
  }
}
