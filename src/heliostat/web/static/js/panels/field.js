// Field stage: where the mirrors stand (docs/ui-spec.md 2.2). Single
// heliostat (x, y) or Field, with a layout picker of Radial staggered
// (default), Fermat spiral, or Manuscript 643. Plan-view drag-to-move is
// deferred; this stage only edits the parametric numbers. Field
// descriptors live in ../fields.js alongside the other three stages',
// even though the Field stage has no in-scene inspector counterpart today
// (docs/ui-spec.md 2.4 only selects a heliostat, the secondary, the
// receiver, or the sun) -- kept there for one consistent home per stage.
import { store } from "../store.js";
import {
  numberRow,
  setVal,
  segButton,
  metersFromMm,
  FIELD_SINGLE_FIELDS,
  FIELD_FERMAT_FIELDS,
  FIELD_RADIAL_STAGGER_FIELDS,
} from "../fields.js";
import { radialStaggerBands, getManuscriptField } from "../api.js";

// Matches app.py's MAX_GEOMETRY_HELIOSTATS -- raised from 10,000 to 15,000
// for docs/ui-spec-v0.2.md §P's built-in reference projects (the
// Stellio-based Hami field ships 14,500 heliostats).
const MAX_GEOMETRY_HELIOSTATS = 15000;
const MAX_TRACE_HELIOSTATS = 1000;

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
  const radialBtn = segButton(layoutSeg, "Radial staggered", false, () =>
    store.set("doc.field.layout", "radial_stagger")
  );
  const fermatBtn = segButton(layoutSeg, "Fermat spiral", false, () => store.set("doc.field.layout", "fermat"));
  layoutRow.appendChild(layoutSeg);
  fieldFields.appendChild(layoutRow);

  const fermatFields = document.createElement("div");
  const nInput = numberRow(fermatFields, FIELD_FERMAT_FIELDS[0]);
  const rMin = numberRow(fermatFields, FIELD_FERMAT_FIELDS[1]);
  const rMax = numberRow(fermatFields, FIELD_FERMAT_FIELDS[2]);
  fieldFields.appendChild(fermatFields);

  const radialFields = document.createElement("div");
  const radialInputs = FIELD_RADIAL_STAGGER_FIELDS.map((f) => numberRow(radialFields, f));
  fieldFields.appendChild(radialFields);
  const radialHint = document.createElement("div");
  radialHint.className = "hint";
  radialHint.textContent = "Concentric staggered rings, band by band (the classic DELSOL/Campo pattern)";
  fieldFields.appendChild(radialHint);

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
    radialBtn,
    fermatBtn,
    fermatFields,
    nInput,
    rMin,
    rMax,
    radialFields,
    radialInputs,
    radialHint,
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

  setVal(els.singleX, metersFromMm(doc.field.single.x_mm));
  setVal(els.singleY, metersFromMm(doc.field.single.y_mm));

  const layout =
    doc.field.layout === "fermat" ? "fermat" : "radial_stagger";
  els.radialBtn.classList.toggle("active", layout === "radial_stagger");
  els.fermatBtn.classList.toggle("active", layout === "fermat");
  els.fermatFields.style.display = layout === "fermat" ? "" : "none";
  els.radialFields.style.display = layout === "radial_stagger" ? "" : "none";
  els.radialHint.style.display = layout === "radial_stagger" ? "" : "none";

  const f = doc.field.fermat;
  setVal(els.nInput, f.n);
  setVal(els.rMin, f.r_min_m);
  setVal(els.rMax, f.r_max_m);

  const bands = radialStaggerBands(doc);
  const radialValues = [
    bands[0].rings,
    bands[0].count,
    bands[1].rings,
    bands[1].count,
    bands[2].rings,
    bands[2].count,
  ];
  els.radialInputs.forEach((input, i) => setVal(input, radialValues[i]));
  const radialN = bands.reduce((sum, b) => sum + b.rings * b.count, 0);

  // "manuscript"/"positions" fall into the "radial_stagger" bucket above
  // for the picker's own highlight (neither has a matching button -- see
  // the `layout` fallback above), but the heliostat count shown here
  // should still be the field that's actually loaded, not the sidebar's
  // (possibly stale/default) radial-stagger numbers -- docs/ui-spec-v0.2.md
  // §P's built-in reference projects made this matter: without this, a
  // loaded 2,650-heliostat Gemasolar project would show "643" here.
  let nHeliostats = radialN;
  if (doc.field.layout === "fermat") {
    nHeliostats = f.n;
  } else if (doc.field.layout === "manuscript") {
    const xy = getManuscriptField();
    if (xy) nHeliostats = xy.length;
  } else if (doc.field.layout === "positions") {
    const xy = doc.field.positions && doc.field.positions.xy_mm;
    if (xy) nHeliostats = xy.length;
  }
  els.hint.textContent =
    `Viewing up to ${MAX_GEOMETRY_HELIOSTATS.toLocaleString()} · tracing up to ${MAX_TRACE_HELIOSTATS.toLocaleString()}` +
    (nHeliostats > MAX_TRACE_HELIOSTATS ? " (this field is too large to trace -- geometry only)" : "");
}
