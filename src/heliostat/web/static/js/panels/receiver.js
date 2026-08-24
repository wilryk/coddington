// Receiver & Tower stage: Prime focus / Axicon / Cassegrain, each with its
// own honest labels (docs/ui-spec.md 2.2's table -- no shared "tower
// height" alias). Field descriptors now live in ../fields.js so the
// floating inspector (../inspector.js) can render the identical rows for
// whichever optics is selected (docs/ui-spec.md 2.4).
import { store } from "../store.js";
import { numberRow, setVal, RECEIVER_FIELD_TABLE, OPTICS_LABELS, apertureMissMessage } from "../fields.js";

let built = false;
let els = {};

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
  // The amber aperture-miss warning (docs/ui-spec.md 2.3) belongs right
  // under the aperture_radius_mm field -- only axicon and cassegrain have
  // one, so prime_focus simply gets no warnBox.
  const warnBoxByOptics = {};
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
    body.appendChild(wrap);
    fieldsByOptics[optics] = wrap;
    inputsByOptics[optics] = inputs;
    warnBoxByOptics[optics] = warnBox;
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

  els = { chev, body, opticsBtns, fieldsByOptics, inputsByOptics, warnBoxByOptics, errorBox };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const optics = doc.optics;

  els.body.style.display = ui.expanded.receiver ? "" : "none";
  els.chev.style.transform = ui.expanded.receiver ? "rotate(90deg)" : "";
  // docs/ui-spec.md 2.1 + mockup M4: highlighted while this stage owns the
  // viewport (elevation view). ui.view itself is driven by main.js's
  // store.subscribe on ui.expanded.receiver, not derived here.
  container.classList.toggle("selectedstage", ui.view === "elevation");

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

  const warnBox = els.warnBoxByOptics[optics];
  if (warnBox) {
    const msg = apertureMissMessage(ui.miss);
    warnBox.hidden = !msg;
    if (msg) warnBox.textContent = msg;
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
