// Heliostat stage: which mirror design the field uses (docs/ui-spec.md 2.2).
// Phase 3a: Rectangle / Facet grid, surface figure. "Edit shape..." opens
// the Heliostat Shape tab, deferred to a later phase -- shown disabled.
// Field descriptors live in ../fields.js so the floating inspector
// (../inspector.js) can render the identical rows for a selected
// heliostat (docs/ui-spec.md 2.4).
import { store } from "../store.js";
import { numberRow, setVal, segButton, HELIOSTAT_RECT_FIELDS, HELIOSTAT_GRID_FIELDS, HELIOSTAT_SURFACE_OPTIONS } from "../fields.js";

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
  h2.textContent = "Heliostat";
  const editLink = document.createElement("a");
  editLink.href = "#";
  editLink.textContent = "Edit shape…";
  editLink.style.fontSize = "11.5px";
  // Phase 3c wave 1: opens the full-screen Heliostat Shape tab
  // (docs/ui-spec.md 2.2, 3) on the current design -- no id to lock the
  // preview to, so js/tabs/shape.js falls back to its own median-heliostat
  // default.
  editLink.addEventListener("click", (e) => {
    e.preventDefault();
    store.set("ui.tab", "shape");
  });
  head.appendChild(chev);
  head.appendChild(h2);
  head.appendChild(editLink);
  head.addEventListener("click", (e) => {
    if (e.target === editLink) return;
    const ui = store.get("ui");
    store.set("ui.expanded.heliostat", !ui.expanded.heliostat);
  });

  const summary = document.createElement("div");
  summary.className = "summary";

  const body = document.createElement("div");
  body.className = "stagebody";

  const typeSeg = document.createElement("div");
  typeSeg.className = "seg";
  const rectBtn = segButton(typeSeg, "Rectangle", true, () => store.set("doc.design.type", "rect"));
  const gridBtn = segButton(typeSeg, "Facet grid", false, () => store.set("doc.design.type", "grid"));
  const customBtn = segButton(typeSeg, "Custom", false, () => store.set("doc.design.type", "custom"));
  body.appendChild(typeSeg);

  const rectFields = document.createElement("div");
  const rectWidth = numberRow(rectFields, HELIOSTAT_RECT_FIELDS[0]);
  const rectHeight = numberRow(rectFields, HELIOSTAT_RECT_FIELDS[1]);
  body.appendChild(rectFields);

  const gridFields = document.createElement("div");
  const gridNu = numberRow(gridFields, HELIOSTAT_GRID_FIELDS[0]);
  const gridNv = numberRow(gridFields, HELIOSTAT_GRID_FIELDS[1]);
  const gridFw = numberRow(gridFields, HELIOSTAT_GRID_FIELDS[2]);
  const gridFh = numberRow(gridFields, HELIOSTAT_GRID_FIELDS[3]);
  const gridGap = numberRow(gridFields, HELIOSTAT_GRID_FIELDS[4]);
  const cantHint = document.createElement("div");
  cantHint.className = "hint";
  cantHint.textContent = "Facet focal and canting are set in the Heliostat Shape tab.";
  gridFields.appendChild(cantHint);
  body.appendChild(gridFields);

  // Custom outline sketching (docs/ui-spec.md 3) needs a canvas -- lives in
  // the Heliostat Shape tab, not this narrow sidebar stage.
  const customFields = document.createElement("div");
  const customHint = document.createElement("div");
  customHint.className = "hint";
  customHint.textContent = "Custom outlines are sketched in the Heliostat Shape tab.";
  customFields.appendChild(customHint);
  body.appendChild(customFields);

  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const surfaceBtns = {};
  for (const [key, label] of HELIOSTAT_SURFACE_OPTIONS) {
    surfaceBtns[key] = segButton(surfaceSeg, label, key === "twisting", () => store.set("doc.design.surface", key));
  }
  body.appendChild(surfaceSeg);

  container.appendChild(head);
  container.appendChild(summary);
  container.appendChild(body);

  els = {
    chev,
    summary,
    body,
    rectBtn,
    gridBtn,
    customBtn,
    rectFields,
    gridFields,
    customFields,
    rectWidth,
    rectHeight,
    gridNu,
    gridNv,
    gridFw,
    gridFh,
    gridGap,
    surfaceBtns,
  };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const type = doc.design.type;

  els.body.style.display = ui.expanded.heliostat ? "" : "none";
  els.chev.style.transform = ui.expanded.heliostat ? "rotate(90deg)" : "";

  els.rectBtn.classList.toggle("active", type === "rect");
  els.gridBtn.classList.toggle("active", type === "grid");
  els.customBtn.classList.toggle("active", type === "custom");
  els.rectFields.style.display = type === "rect" ? "" : "none";
  els.gridFields.style.display = type === "grid" ? "" : "none";
  els.customFields.style.display = type === "custom" ? "" : "none";

  const surface = doc.design.surface;
  for (const [key, btn] of Object.entries(els.surfaceBtns)) {
    btn.classList.toggle("active", key === surface);
  }

  if (type === "rect") {
    const p = doc.designParams.rect;
    setVal(els.rectWidth, p.width_mm);
    setVal(els.rectHeight, p.height_mm);
    const wM = (p.width_mm / 1000).toFixed(1);
    const hM = (p.height_mm / 1000).toFixed(1);
    els.summary.innerHTML = `<strong>${wM} × ${hM} m</strong> — rectangle, ${surface} figure`;
  } else if (type === "grid") {
    const p = doc.designParams.grid;
    setVal(els.gridNu, p.n_u);
    setVal(els.gridNv, p.n_v);
    setVal(els.gridFw, p.facet_w_mm);
    setVal(els.gridFh, p.facet_h_mm);
    setVal(els.gridGap, p.gap_mm);
    els.summary.innerHTML = `<strong>${p.n_u}×${p.n_v} facet grid</strong> — ${surface} figure`;
  } else {
    const p = doc.designParams.custom;
    const n = (p && p.vertices_mm && p.vertices_mm.length) || 0;
    els.summary.innerHTML = `<strong>custom outline, ${n} vertices</strong> — ${surface} figure`;
  }
}
