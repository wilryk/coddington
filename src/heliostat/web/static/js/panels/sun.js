// Sun stage: azimuth + elevation (docs/ui-spec.md 2.2). Site & time entry
// is a later phase; Phase 3a is the direct az/el pair the trace endpoints
// take verbatim.
import { store } from "../store.js";

let built = false;
let els = {};

function isFocused(el) {
  return el && document.activeElement === el;
}

function setVal(input, value) {
  if (isFocused(input)) return;
  const s = value == null ? "" : String(value);
  if (input.value !== s) input.value = s;
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

function build(container) {
  container.innerHTML = "";
  container.className = "stage";

  const head = document.createElement("div");
  head.className = "stagehead";
  const chev = document.createElement("span");
  chev.className = "chev";
  chev.textContent = "▾";
  const h2 = document.createElement("h2");
  h2.textContent = "Sun";
  head.appendChild(chev);
  head.appendChild(h2);
  head.addEventListener("click", () => {
    const ui = store.get("ui");
    store.set("ui.expanded.sun", !ui.expanded.sun);
  });

  const body = document.createElement("div");
  body.className = "stagebody";
  const az = numberRow(body, "Azimuth (°)", "doc.sun.az", { min: 0, max: 360, step: 0.1 });
  const el = numberRow(body, "Elevation (°)", "doc.sun.el", { min: -90, max: 90, step: 0.1 });
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = "Site & time entry -- coming in a later phase.";
  body.appendChild(hint);

  container.appendChild(head);
  container.appendChild(body);

  els = { chev, body, az, el };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");

  els.body.style.display = ui.expanded.sun ? "" : "none";
  els.chev.style.transform = ui.expanded.sun ? "rotate(90deg)" : "";

  setVal(els.az, doc.sun.az);
  setVal(els.el, doc.sun.el);
}
