// Field stage: where the mirrors stand (docs/ui-spec.md 2.2). Phase 3a:
// Single heliostat (x, y) or Field (Fermat spiral -- the only layout
// built; the picker itself is a later phase). Plan-view drag-to-move is
// deferred; this stage only edits the parametric numbers.
import { store } from "../store.js";

const MAX_GEOMETRY_HELIOSTATS = 10000;
const MAX_TRACE_HELIOSTATS = 1000;

let built = false;
let els = {};

function isFocused(el) {
  return el && document.activeElement === el;
}

function numberRow(parent, label, path, opts) {
  const row = document.createElement("div");
  row.className = "frow";
  const lab = document.createElement("label");
  lab.textContent = label;
  const input = document.createElement("input");
  input.type = "number";
  input.className = "val";
  if (opts && opts.step !== undefined) input.step = opts.step;
  if (opts && opts.min !== undefined) input.min = opts.min;
  if (opts && opts.max !== undefined) input.max = opts.max;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v)) store.set(path, v);
  });
  row.appendChild(lab);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

function segButton(parent, label, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.addEventListener("click", onClick);
  parent.appendChild(btn);
  return btn;
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
  h2.textContent = "Field";
  head.appendChild(chev);
  head.appendChild(h2);
  head.addEventListener("click", () => {
    const ui = store.get("ui");
    store.set("ui.expanded.field", !ui.expanded.field);
  });

  const body = document.createElement("div");
  body.className = "stagebody";

  const modeSeg = document.createElement("div");
  modeSeg.className = "seg";
  const singleBtn = segButton(modeSeg, "Single heliostat", () => store.set("doc.field.mode", "single"));
  const fieldBtn = segButton(modeSeg, "Field", () => store.set("doc.field.mode", "field"));
  body.appendChild(modeSeg);

  const singleFields = document.createElement("div");
  const singleX = numberRow(singleFields, "X (mm)", "doc.field.single.x_mm");
  const singleY = numberRow(singleFields, "Y (mm)", "doc.field.single.y_mm");
  body.appendChild(singleFields);

  const fieldFields = document.createElement("div");
  const layoutRow = document.createElement("div");
  layoutRow.className = "frow";
  layoutRow.innerHTML = '<label>Layout</label><div class="val">Fermat spiral</div>';
  fieldFields.appendChild(layoutRow);
  const nInput = numberRow(fieldFields, "Heliostats", "doc.field.fermat.n", { min: 1, max: MAX_GEOMETRY_HELIOSTATS, step: 1 });
  const rMin = numberRow(fieldFields, "Nearest radius (m)", "doc.field.fermat.r_min_m", { min: 0 });
  const rMax = numberRow(fieldFields, "Farthest radius (m)", "doc.field.fermat.r_max_m", { min: 0 });
  const hint = document.createElement("div");
  hint.className = "hint";
  fieldFields.appendChild(hint);
  body.appendChild(fieldFields);

  container.appendChild(head);
  container.appendChild(body);

  els = { chev, body, singleBtn, fieldBtn, singleFields, fieldFields, singleX, singleY, nInput, rMin, rMax, hint };
  built = true;
}

function setVal(input, value) {
  if (isFocused(input)) return;
  const s = value == null ? "" : String(value);
  if (input.value !== s) input.value = s;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const mode = doc.field.mode;

  els.body.style.display = ui.expanded.field ? "" : "none";
  els.chev.style.transform = ui.expanded.field ? "rotate(90deg)" : "";

  els.singleBtn.classList.toggle("active", mode === "single");
  els.fieldBtn.classList.toggle("active", mode === "field");
  els.singleFields.style.display = mode === "single" ? "" : "none";
  els.fieldFields.style.display = mode === "field" ? "" : "none";

  setVal(els.singleX, doc.field.single.x_mm);
  setVal(els.singleY, doc.field.single.y_mm);

  const f = doc.field.fermat;
  setVal(els.nInput, f.n);
  setVal(els.rMin, f.r_min_m);
  setVal(els.rMax, f.r_max_m);

  els.hint.textContent =
    `Viewing up to ${MAX_GEOMETRY_HELIOSTATS.toLocaleString()} · tracing up to ${MAX_TRACE_HELIOSTATS.toLocaleString()}` +
    (f.n > MAX_TRACE_HELIOSTATS ? " (this field is too large to trace -- geometry only)" : "");
}
