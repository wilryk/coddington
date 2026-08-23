// Heliostat stage: which mirror design the field uses (docs/ui-spec.md 2.2).
// Phase 3a: Rectangle / Facet grid, surface figure. "Edit shape..." opens
// the Heliostat Shape tab, deferred to a later phase -- shown disabled.
import { store } from "../store.js";

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
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v)) store.set(path, v);
  });
  row.appendChild(lab);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

function segButton(parent, label, active, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.className = active ? "active" : "";
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
  h2.textContent = "Heliostat";
  const editLink = document.createElement("a");
  editLink.href = "#";
  editLink.textContent = "Edit shape…";
  editLink.style.fontSize = "11.5px";
  editLink.title = "Heliostat Shape editor -- coming in a later phase";
  editLink.setAttribute("aria-disabled", "true");
  editLink.classList.add("disabled-link");
  editLink.addEventListener("click", (e) => e.preventDefault());
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
  body.appendChild(typeSeg);

  const rectFields = document.createElement("div");
  const rectWidth = numberRow(rectFields, "Width (mm)", "doc.designParams.rect.width_mm", { min: 1 });
  const rectHeight = numberRow(rectFields, "Height (mm)", "doc.designParams.rect.height_mm", { min: 1 });
  body.appendChild(rectFields);

  const gridFields = document.createElement("div");
  const gridNu = numberRow(gridFields, "Columns (n_u)", "doc.designParams.grid.n_u", { min: 1, step: 1 });
  const gridNv = numberRow(gridFields, "Rows (n_v)", "doc.designParams.grid.n_v", { min: 1, step: 1 });
  const gridFw = numberRow(gridFields, "Facet width (mm)", "doc.designParams.grid.facet_w_mm", { min: 1 });
  const gridFh = numberRow(gridFields, "Facet height (mm)", "doc.designParams.grid.facet_h_mm", { min: 1 });
  const gridGap = numberRow(gridFields, "Gap (mm)", "doc.designParams.grid.gap_mm", { min: 0 });
  const cantHint = document.createElement("div");
  cantHint.className = "hint";
  cantHint.textContent = "Canting: Auto (slant range) -- editable in a later phase.";
  gridFields.appendChild(cantHint);
  body.appendChild(gridFields);

  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const twistBtn = segButton(surfaceSeg, "Twisting", true, () => store.set("doc.design.surface", "twisting"));
  const sphBtn = segButton(surfaceSeg, "Spherical", false, () => store.set("doc.design.surface", "spherical"));
  const flatBtn = segButton(surfaceSeg, "Flat", false, () => store.set("doc.design.surface", "flat"));
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
    rectFields,
    gridFields,
    rectWidth,
    rectHeight,
    gridNu,
    gridNv,
    gridFw,
    gridFh,
    gridGap,
    twistBtn,
    sphBtn,
    flatBtn,
  };
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
  const type = doc.design.type;

  els.body.style.display = ui.expanded.heliostat ? "" : "none";
  els.chev.style.transform = ui.expanded.heliostat ? "rotate(90deg)" : "";

  els.rectBtn.classList.toggle("active", type === "rect");
  els.gridBtn.classList.toggle("active", type === "grid");
  els.rectFields.style.display = type === "rect" ? "" : "none";
  els.gridFields.style.display = type === "grid" ? "" : "none";

  const surface = doc.design.surface;
  els.twistBtn.classList.toggle("active", surface === "twisting");
  els.sphBtn.classList.toggle("active", surface === "spherical");
  els.flatBtn.classList.toggle("active", surface === "flat");

  if (type === "rect") {
    const p = doc.designParams.rect;
    setVal(els.rectWidth, p.width_mm);
    setVal(els.rectHeight, p.height_mm);
    const wM = (p.width_mm / 1000).toFixed(1);
    const hM = (p.height_mm / 1000).toFixed(1);
    els.summary.innerHTML = `<strong>${wM} × ${hM} m</strong> — rectangle, ${surface} figure`;
  } else {
    const p = doc.designParams.grid;
    setVal(els.gridNu, p.n_u);
    setVal(els.gridNv, p.n_v);
    setVal(els.gridFw, p.facet_w_mm);
    setVal(els.gridFh, p.facet_h_mm);
    setVal(els.gridGap, p.gap_mm);
    els.summary.innerHTML = `<strong>${p.n_u}×${p.n_v} facet grid</strong> — ${surface} figure`;
  }
}
