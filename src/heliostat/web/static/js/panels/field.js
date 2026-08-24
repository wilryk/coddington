// Field stage: where the mirrors stand (docs/ui-spec.md 2.2). Phase 3a:
// Single heliostat (x, y) or Field (Fermat spiral -- the only layout
// built; the picker itself is a later phase). Plan-view drag-to-move is
// deferred; this stage only edits the parametric numbers. Field
// descriptors live in ../fields.js alongside the other three stages',
// even though the Field stage has no in-scene inspector counterpart today
// (docs/ui-spec.md 2.4 only selects a heliostat, the secondary, the
// receiver, or the sun) -- kept there for one consistent home per stage.
import { store } from "../store.js";
import { numberRow, setVal, segButton, FIELD_SINGLE_FIELDS, FIELD_FERMAT_FIELDS } from "../fields.js";
import { getManuscriptField } from "../api.js";

const MAX_GEOMETRY_HELIOSTATS = 10000;
const MAX_TRACE_HELIOSTATS = 1000;
const PAPER_N_HELIOSTATS = 643;

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
  const singleBtn = segButton(modeSeg, "Single heliostat", false, () => store.set("doc.field.mode", "single"));
  const fieldBtn = segButton(modeSeg, "Field", false, () => store.set("doc.field.mode", "field"));
  body.appendChild(modeSeg);

  const singleFields = document.createElement("div");
  const singleX = numberRow(singleFields, FIELD_SINGLE_FIELDS[0]);
  const singleY = numberRow(singleFields, FIELD_SINGLE_FIELDS[1]);
  body.appendChild(singleFields);

  const fieldFields = document.createElement("div");

  const layoutRow = document.createElement("div");
  layoutRow.className = "frow";
  const layoutLabel = document.createElement("label");
  layoutLabel.textContent = "Layout";
  layoutRow.appendChild(layoutLabel);
  const layoutSeg = document.createElement("div");
  layoutSeg.className = "seg";
  const manuscriptBtn = segButton(layoutSeg, "Manuscript 643", false, () =>
    store.set("doc.field.layout", "manuscript")
  );
  const fermatBtn = segButton(layoutSeg, "Fermat spiral", false, () => store.set("doc.field.layout", "fermat"));
  layoutRow.appendChild(layoutSeg);
  fieldFields.appendChild(layoutRow);

  const manuscriptHint = document.createElement("div");
  manuscriptHint.className = "hint";
  manuscriptHint.textContent = "The paper's exact 643 heliostat positions (field_645.csv)";
  fieldFields.appendChild(manuscriptHint);

  const fermatFields = document.createElement("div");
  const nInput = numberRow(fermatFields, FIELD_FERMAT_FIELDS[0]);
  const rMin = numberRow(fermatFields, FIELD_FERMAT_FIELDS[1]);
  const rMax = numberRow(fermatFields, FIELD_FERMAT_FIELDS[2]);
  fieldFields.appendChild(fermatFields);

  const hint = document.createElement("div");
  hint.className = "hint";
  fieldFields.appendChild(hint);
  body.appendChild(fieldFields);

  container.appendChild(head);
  container.appendChild(body);

  els = {
    chev,
    body,
    singleBtn,
    fieldBtn,
    singleFields,
    fieldFields,
    singleX,
    singleY,
    manuscriptBtn,
    fermatBtn,
    manuscriptHint,
    fermatFields,
    nInput,
    rMin,
    rMax,
    hint,
  };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const mode = doc.field.mode;

  els.body.style.display = ui.expanded.field ? "" : "none";
  els.chev.style.transform = ui.expanded.field ? "rotate(90deg)" : "";
  // docs/ui-spec.md 2.1 + mockup M3: highlighted while this stage owns the
  // viewport (plan view). ui.view itself is driven by main.js's
  // store.subscribe on ui.expanded.field, not derived here.
  container.classList.toggle("selectedstage", ui.view === "plan");

  els.singleBtn.classList.toggle("active", mode === "single");
  els.fieldBtn.classList.toggle("active", mode === "field");
  els.singleFields.style.display = mode === "single" ? "" : "none";
  els.fieldFields.style.display = mode === "field" ? "" : "none";

  setVal(els.singleX, doc.field.single.x_mm);
  setVal(els.singleY, doc.field.single.y_mm);

  const layout = doc.field.layout === "fermat" ? "fermat" : "manuscript";
  els.manuscriptBtn.classList.toggle("active", layout === "manuscript");
  els.fermatBtn.classList.toggle("active", layout === "fermat");
  els.manuscriptHint.style.display = layout === "manuscript" ? "" : "none";
  els.fermatFields.style.display = layout === "fermat" ? "" : "none";

  const f = doc.field.fermat;
  setVal(els.nInput, f.n);
  setVal(els.rMin, f.r_min_m);
  setVal(els.rMax, f.r_max_m);

  // The manuscript layout is always exactly 643 (the cached fetch's own
  // length when it landed, else the paper's known count) and always within
  // the trace cap, so its hint never carries the "geometry only" caveat.
  const manuscriptXY = getManuscriptField();
  const nHeliostats = layout === "manuscript" ? (manuscriptXY ? manuscriptXY.length : PAPER_N_HELIOSTATS) : f.n;
  els.hint.textContent =
    `Viewing up to ${MAX_GEOMETRY_HELIOSTATS.toLocaleString()} · tracing up to ${MAX_TRACE_HELIOSTATS.toLocaleString()}` +
    (nHeliostats > MAX_TRACE_HELIOSTATS ? " (this field is too large to trace -- geometry only)" : "");
}
