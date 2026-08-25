// Heliostat Shape tab: the full-screen detail editor for the mirror --
// shape, figure, facet curvature, canting, optical errors, a live
// server-rendered aperture-layout preview, and the sag map.
//
// Two things live here that the sidebar panels don't have to deal with:
//
// 1. Server-rendered previews (aperture PNG, sag PNG) are not store state --
//    they are fetched over the network, debounced, abortable, and cached as
//    object URLs. Those object URLs, in-flight abort controllers, and the
//    last-request cache keys are module-locals -- nothing a user *reads a
//    value out of* lives only here.
// 2. The Custom outline sketch canvas is an interactive SVG editor (drag
//    vertices, click to add, click a length to type it exactly) layered
//    over the same store paths the vertex table's number inputs use --
//    doc.designParams.custom.vertices_mm.{i}.{0|1} -- so a drag and a typed
//    number are the same edit, not two pathways.
import { store } from "../store.js";
import {
  numberRow,
  setVal,
  segButton,
  HELIOSTAT_RECT_FIELDS,
  HELIOSTAT_GRID_FIELDS,
  HELIOSTAT_SURFACE_OPTIONS,
  apertureSummaryText,
} from "../fields.js";
import {
  buildSagRequest,
  currentDesignPayload,
  expandCustomVertices,
  postDesignPreview,
  postDesignSag,
  saveLibraryEntry,
} from "../api.js";

// -- local field descriptors -------------------------------------------
// Optical errors are only edited from this tab, so they don't need a home
// in fields.js's shared descriptor tables the way the sidebar-shared ones
// do.
const ERROR_FIELDS = [
  { key: "slope_error_mrad", label: "Slope error (mrad)", path: "doc.design.errors.slope_error_mrad", min: 0, step: 0.1 },
  { key: "specularity_mrad", label: "Specularity (mrad)", path: "doc.design.errors.specularity_mrad", min: 0, step: 0.1 },
  { key: "reflectance_pct", label: "Reflectance (%)", path: "doc.design.errors.reflectance_pct", min: 0, max: 100, step: 0.5 },
];

// cant_focal_mm is aim only: null = per-heliostat slant range, 0 =
// uncanted/parallel, >0 = one fixed focal for the whole field.
const CANT_FOCAL_FIELD = {
  key: "cant_focal_mm",
  label: "Field-wide focal (mm)",
  path: "doc.designParams.grid.cant_focal_mm",
  min: 1,
};

// facet_focal_mm is the facet's own curvature, independent of aim: null =
// follows the canting focal, 0 = truly flat facets, >0 = that focal.
const FACET_FOCAL_FIELD = {
  key: "facet_focal_mm",
  label: "Facet focal (mm)",
  path: "doc.designParams.grid.facet_focal_mm",
  min: 1,
};

// -- module state ---------------------------------------------------------

let built = false;
let els = {};

// Object URLs + fetch bookkeeping for the two server-rendered previews --
// see the header comment on why these are module-locals, not store state.
let previewObjectUrl = null;
let sagObjectUrl = null;
let lastPreviewKey = null;
let lastSagKey = null;
let lastSagResult = null; // {contourIntervalMm, peakToValleyMm, slantRangeM} of the currently-shown sag PNG

// Ephemeral view-only toggles -- these don't hold a value the user typed,
// just which of two already-live views is showing.
let popoverOpen = false;
let showServerRenderForCustom = false;
let saveOpen = false;
let saveError = null;
let saveSaved = false;

// Custom-sketch drag state + the projection its pointer handlers need to
// invert screen px back to design mm -- rebuilt every render (see
// computeSketchProjection), read by the window-level drag listeners below.
let dragState = null; // { index, u, v } | null
let sketchProj = null;
let locatorProj = null;

// Re-render hook set at the top of render() so async callbacks (a preview
// fetch landing, a save-as completing) and drag handlers can trigger a
// fresh paint without going through a store write.
let lastContainer = null;
let lastCtx = null;
function rerender() {
  if (lastContainer) render(lastContainer, lastCtx);
}

// -- pure helpers -----------------------------------------------------------

function clonePoints(vertices) {
  return (vertices || []).map((p) => [p[0], p[1]]);
}

function cantMode(cantFocalMm) {
  if (cantFocalMm === null || cantFocalMm === undefined) return "auto";
  if (cantFocalMm === 0) return "off";
  return "focal";
}

function curvatureMode(facetFocalMm) {
  if (facetFocalMm === null || facetFocalMm === undefined) return "auto";
  if (facetFocalMm === 0) return "off";
  return "focal";
}

function designSummaryText(doc) {
  return `${apertureSummaryText(doc)} — ${doc.design.surface} figure`;
}

// Deterministic median-by-radius pick so re-rendering the same geometry
// always lands on the same heliostat, ties broken by id.
function pickPreviewHeliostat(ui, geometry) {
  const list = (geometry && geometry.heliostats) || [];
  if (!list.length) return null;
  if (ui.shapeHeliostatId != null) {
    const found = list.find((h) => h.id === ui.shapeHeliostatId);
    if (found) return found;
  }
  const sorted = list.slice().sort((a, b) => {
    const ra = Math.hypot(a.x_mm, a.y_mm);
    const rb = Math.hypot(b.x_mm, b.y_mm);
    return ra !== rb ? ra - rb : a.id - b.id;
  });
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function slantRangeLabel(previewHeliostat) {
  if (previewHeliostat && previewHeliostat.slant_range_m != null) return previewHeliostat.slant_range_m.toFixed(1);
  if (lastSagResult && lastSagResult.slantRangeM != null) return lastSagResult.slantRangeM.toFixed(1);
  return null;
}

// -- debounced, abortable blob fetch (shared shape by the aperture and sag
// panels) ------------------------------------------------------------

function createBlobRequester(delay) {
  let timer = null;
  let controller = null;
  return function schedule(fetchFn, onSuccess, onError) {
    if (timer) clearTimeout(timer);
    if (controller) controller.abort();
    timer = setTimeout(() => {
      timer = null;
      controller = new AbortController();
      fetchFn(controller.signal)
        .then((result) => {
          controller = null;
          onSuccess(result);
        })
        .catch((err) => {
          controller = null;
          if (err && err.name === "AbortError") return;
          onError(err);
        });
    }, delay);
  };
}

const schedulePreview = createBlobRequester(400);
const scheduleSag = createBlobRequester(400);

// -- custom sketch: mm <-> px projection, drag, add/remove/length edit ----

function computeSketchProjection(vertices, mirror, w, h) {
  const pts = clonePoints(vertices);
  if (mirror) {
    for (const [u, v] of vertices) if (u !== 0) pts.push([-u, v]);
  }
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (const [u, v] of pts) {
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  if (!Number.isFinite(minU)) {
    minU = -1000; maxU = 1000; minV = -1000; maxV = 1000;
  }
  const spanU = Math.max(maxU - minU, 200);
  const spanV = Math.max(maxV - minV, 200);
  const margin = 34;
  const scale = Math.max(Math.min((w - 2 * margin) / spanU, (h - 2 * margin) / spanV), 0.001);
  const centerU = (minU + maxU) / 2;
  const centerV = (minV + maxV) / 2;
  return { scale, ox: w / 2 - centerU * scale, oy: h / 2 + centerV * scale };
}

function toPx(proj, u, v) {
  return [proj.ox + u * proj.scale, proj.oy - v * proj.scale];
}

function fromPx(proj, px, py) {
  return [(px - proj.ox) / proj.scale, (proj.oy - py) / proj.scale];
}

function sketchMarkup(vertices, mirror, w, h, proj) {
  const n = vertices.length;
  let s = `<rect x="0" y="0" width="${w}" height="${h}" fill="#ffffff"></rect>`;

  if (mirror) {
    // The mirror axis (u = 0), dash-dot like a drawing's centerline -- the
    // sketch is the right half, the ghost is what the axis reflects.
    const [ax] = toPx(proj, 0, 0);
    s += `<line class="sketch-axis" x1="${ax.toFixed(1)}" y1="0" x2="${ax.toFixed(1)}" y2="${h}"></line>`;
  }

  if (mirror) {
    let d = "";
    for (let i = 0; i < n; i++) {
      const [px, py] = toPx(proj, -vertices[i][0], vertices[i][1]);
      d += (i === 0 ? "M " : "L ") + px.toFixed(1) + " " + py.toFixed(1) + " ";
    }
    s += `<path class="sketch-ghost" d="${d}"></path>`;
  }

  let d = "";
  for (let i = 0; i < n; i++) {
    const [px, py] = toPx(proj, vertices[i][0], vertices[i][1]);
    d += (i === 0 ? "M " : "L ") + px.toFixed(1) + " " + py.toFixed(1) + " ";
  }
  if (!mirror) d += "Z";
  s += `<path class="sketch-outline" d="${d}"></path>`;

  // Segment length labels -- a closed loop's own edge (n-1 -> 0) only
  // exists when NOT mirrored (see api.js's expandCustomVertices comment:
  // mirror is what closes the shape, so the sketch itself is an open chain).
  const segCount = mirror ? n - 1 : n;
  for (let i = 0; i < segCount; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % n];
    const lenMm = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const [mx, my] = toPx(proj, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
    s += `<text class="sketch-length" data-length-index="${i}" x="${mx.toFixed(1)}" y="${my.toFixed(1)}" text-anchor="middle">${Math.round(lenMm)}</text>`;
  }

  for (let i = 0; i < n; i++) {
    const [px, py] = toPx(proj, vertices[i][0], vertices[i][1]);
    s += `<circle class="sketch-vertex" data-vertex-index="${i}" cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5.5"></circle>`;
  }
  return s;
}

function currentCustomVertices() {
  return store.get("doc.designParams.custom.vertices_mm") || [];
}

// Sutherland-Hodgman clip of a closed outline against the half-plane
// u >= 0. Turning mirror ON runs the current outline through this so a
// full-width symmetric outline (like the default rectangle) survives an
// on/off round-trip unchanged, instead of expanding into a self-overlapping
// polygon whose doubly-covered area the trace's even-odd membership test
// counts as OUTSIDE the mirror.
function clipRightHalf(vertices) {
  const out = [];
  const n = vertices.length;
  for (let i = 0; i < n; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % n];
    const aIn = a[0] >= 0;
    const bIn = b[0] >= 0;
    if (aIn) out.push([a[0], a[1]]);
    if (aIn !== bIn) out.push([0, Math.round(a[1] + (a[0] / (a[0] - b[0])) * (b[1] - a[1]))]);
  }
  const dedup = out.filter((p, i) => i === 0 || p[0] !== out[i - 1][0] || p[1] !== out[i - 1][1]);
  // The outline is closed implicitly, so a crossing that lands exactly on
  // the first vertex would otherwise leave a phantom duplicate row at the
  // end of the vertex table.
  if (dedup.length > 1) {
    const first = dedup[0];
    const last = dedup[dedup.length - 1];
    if (first[0] === last[0] && first[1] === last[1]) dedup.pop();
  }
  return dedup.length >= 3 ? dedup : null;
}

function setMirror(on) {
  const vertices = currentCustomVertices();
  if (on) {
    const half = clipRightHalf(vertices);
    // A sketch living entirely at u < 0 clips to nothing -- leave its
    // vertices alone and let the ghost show what the mirror would do.
    if (half) store.set("doc.designParams.custom.vertices_mm", half);
  } else {
    // Bake the mirrored outline into the sketch, so what was on screen is
    // exactly what stays editable.
    store.set("doc.designParams.custom.vertices_mm", expandCustomVertices(vertices, true));
  }
  store.set("doc.designParams.custom.mirror", on);
}

function renderSketchNow() {
  if (!els.sketchSvg) return;
  const w = Math.max(els.apertureFrame.clientWidth || 1, 100);
  const h = Math.max(els.apertureFrame.clientHeight || 1, 100);
  els.sketchSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const mirror = !!store.get("doc.designParams.custom.mirror");
  let vertices = clonePoints(currentCustomVertices());
  if (dragState) vertices[dragState.index] = [dragState.u, dragState.v];
  sketchProj = computeSketchProjection(vertices, mirror, w, h);
  els.sketchSvg.innerHTML = sketchMarkup(vertices, mirror, w, h, sketchProj);
}

function startDragVertex(index) {
  const vertices = currentCustomVertices();
  const v = vertices[index];
  if (!v) return;
  dragState = { index, u: v[0], v: v[1] };
  window.addEventListener("pointermove", onSketchPointerMove);
  window.addEventListener("pointerup", onSketchPointerUp);
}

function onSketchPointerMove(e) {
  if (!dragState || !sketchProj || !els.sketchSvg) return;
  const rect = els.sketchSvg.getBoundingClientRect();
  const [u, v] = fromPx(sketchProj, e.clientX - rect.left, e.clientY - rect.top);
  dragState.u = u;
  dragState.v = v;
  renderSketchNow();
}

function onSketchPointerUp() {
  window.removeEventListener("pointermove", onSketchPointerMove);
  window.removeEventListener("pointerup", onSketchPointerUp);
  if (dragState) {
    const i = dragState.index;
    store.set(`doc.designParams.custom.vertices_mm.${i}.0`, Math.round(dragState.u));
    store.set(`doc.designParams.custom.vertices_mm.${i}.1`, Math.round(dragState.v));
  }
  dragState = null;
}

function insertVertexNearestSegment(u, v) {
  const vertices = currentCustomVertices();
  const mirror = !!store.get("doc.designParams.custom.mirror");
  const n = vertices.length;
  const segCount = mirror ? n - 1 : n;
  let bestI = segCount - 1;
  let bestD = Infinity;
  for (let i = 0; i < segCount; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % n];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy || 1;
    let t = ((u - a[0]) * dx + (v - a[1]) * dy) / len2;
    t = Math.min(1, Math.max(0, t));
    const px = a[0] + t * dx;
    const py = a[1] + t * dy;
    const d = Math.hypot(u - px, v - py);
    if (d < bestD) {
      bestD = d;
      bestI = i;
    }
  }
  const next = vertices.slice();
  next.splice(bestI + 1, 0, [Math.round(u), Math.round(v)]);
  store.set("doc.designParams.custom.vertices_mm", next);
}

function removeCustomVertex(index) {
  const vertices = currentCustomVertices();
  if (vertices.length <= 3) return; // a closed outline needs >= 3 vertices
  const next = vertices.slice();
  next.splice(index, 1);
  store.set("doc.designParams.custom.vertices_mm", next);
}

function addCustomVertex() {
  const vertices = currentCustomVertices();
  const last = vertices[vertices.length - 1] || [0, 0];
  store.set("doc.designParams.custom.vertices_mm", vertices.concat([[last[0] + 300, last[1]]]));
}

function onLengthLabelClick(index) {
  const vertices = currentCustomVertices();
  const n = vertices.length;
  const a = vertices[index];
  const b = vertices[(index + 1) % n];
  if (!a || !b) return;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const curLen = Math.hypot(dx, dy);
  const input = window.prompt("Segment length (mm):", String(Math.round(curLen)));
  if (input == null) return;
  const newLen = parseFloat(input);
  if (!Number.isFinite(newLen) || newLen <= 0) return;
  const d = curLen || 1;
  const ux = dx / d;
  const uy = dy / d;
  const bIndex = (index + 1) % n;
  store.set(`doc.designParams.custom.vertices_mm.${bIndex}.0`, Math.round(a[0] + ux * newLen));
  store.set(`doc.designParams.custom.vertices_mm.${bIndex}.1`, Math.round(a[1] + uy * newLen));
}

// -- build (once) -----------------------------------------------------------

function build(container) {
  container.innerHTML = "";
  container.className = "tabpage";

  // -- editbar --------------------------------------------------------
  const editbar = document.createElement("div");
  editbar.className = "editbar";

  const nameEl = document.createElement("span");
  nameEl.className = "name";
  const fromEl = document.createElement("span");
  fromEl.className = "from";
  fromEl.textContent = "updates live in the workspace field";

  const chip = document.createElement("span");
  chip.className = "previewchip";
  chip.innerHTML =
    '<svg class="dot" width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#0b5fd0" stroke-width="1.6"><circle cx="8" cy="8" r="6"></circle><circle cx="8" cy="8" r="1.6" fill="#0b5fd0" stroke="none"></circle></svg>' +
    'Previewing on <strong data-ref="chip-id"></strong> <span data-ref="chip-r"></span> <span class="caret">▾</span>';
  const chipId = chip.querySelector('[data-ref="chip-id"]');
  const chipR = chip.querySelector('[data-ref="chip-r"]');
  const popover = document.createElement("div");
  popover.className = "previewpopover";
  popover.hidden = true;
  const popRow = document.createElement("div");
  popRow.className = "frow";
  const popLabel = document.createElement("label");
  popLabel.textContent = "Heliostat id";
  const popInput = document.createElement("input");
  popInput.type = "number";
  popInput.className = "val";
  popInput.step = "1";
  popRow.appendChild(popLabel);
  popRow.appendChild(popInput);
  const popApply = document.createElement("div");
  popApply.className = "btn small primary";
  popApply.textContent = "Apply";
  const popHint = document.createElement("div");
  popHint.className = "hint";
  popHint.style.margin = "6px 0 0 0";
  popHint.textContent = "or click a heliostat in the Workspace and choose View shape";
  popover.appendChild(popRow);
  popover.appendChild(popApply);
  popover.appendChild(popHint);
  chip.appendChild(popover);
  chip.addEventListener("click", (e) => {
    if (popover.contains(e.target) && e.target !== chip) return;
    popoverOpen = !popoverOpen;
    rerender();
  });
  popApply.addEventListener("click", (e) => {
    e.stopPropagation();
    const id = parseInt(popInput.value, 10);
    if (Number.isFinite(id)) store.set("ui.shapeHeliostatId", id);
    popoverOpen = false;
    rerender();
  });

  const rightWrap = document.createElement("div");
  rightWrap.style.marginLeft = "auto";
  rightWrap.style.display = "flex";
  rightWrap.style.gap = "8px";
  rightWrap.style.alignItems = "center";

  const saveWrap = document.createElement("div");
  const saveBtn = document.createElement("div");
  saveBtn.className = "btn";
  saveBtn.textContent = "Save to library as…";
  const saveInline = document.createElement("div");
  saveInline.className = "cardinline";
  saveInline.hidden = true;
  const saveNameInput = document.createElement("input");
  saveNameInput.type = "text";
  saveNameInput.className = "val";
  saveNameInput.placeholder = "Design name";
  const saveConfirm = document.createElement("div");
  saveConfirm.className = "btn small primary";
  saveConfirm.textContent = "Save";
  const saveCancel = document.createElement("div");
  saveCancel.className = "btn small";
  saveCancel.textContent = "Cancel";
  saveInline.appendChild(saveNameInput);
  saveInline.appendChild(saveConfirm);
  saveInline.appendChild(saveCancel);
  const saveErrEl = document.createElement("div");
  saveErrEl.className = "fielderr";
  saveErrEl.hidden = true;
  saveWrap.appendChild(saveBtn);
  saveWrap.appendChild(saveInline);
  saveWrap.appendChild(saveErrEl);

  saveBtn.addEventListener("click", () => {
    saveOpen = true;
    saveError = null;
    saveSaved = false;
    rerender();
    saveNameInput.focus();
  });
  saveCancel.addEventListener("click", () => {
    saveOpen = false;
    saveError = null;
    rerender();
  });
  saveConfirm.addEventListener("click", () => {
    const name = saveNameInput.value.trim();
    if (!name) {
      saveError = "Enter a name to save as.";
      rerender();
      return;
    }
    const doc = store.get("doc");
    saveLibraryEntry("designs", name, currentDesignPayload(doc))
      .then(() => {
        saveOpen = false;
        saveError = null;
        saveSaved = true;
        saveNameInput.value = "";
        rerender();
      })
      .catch((err) => {
        saveError = (err && err.message) || "Save failed.";
        rerender();
      });
  });

  const doneBtn = document.createElement("div");
  doneBtn.className = "btn primary";
  doneBtn.textContent = "Done — back to workspace";
  doneBtn.addEventListener("click", () => store.set("ui.tab", "workspace"));

  rightWrap.appendChild(saveWrap);
  rightWrap.appendChild(doneBtn);

  editbar.appendChild(nameEl);
  editbar.appendChild(fromEl);
  editbar.appendChild(chip);
  editbar.appendChild(rightWrap);

  // -- content: controls + previews ------------------------------------
  const content = document.createElement("div");
  content.className = "tabcontent";

  const controls = document.createElement("div");
  controls.className = "panel controls";
  const controlsH2 = document.createElement("h2");
  controlsH2.textContent = "Design";
  controls.appendChild(controlsH2);

  const typeSeg = document.createElement("div");
  typeSeg.className = "seg";
  const rectBtn = segButton(typeSeg, "Rectangle", false, () => store.set("doc.design.type", "rect"));
  const gridBtn = segButton(typeSeg, "Facet grid", false, () => store.set("doc.design.type", "grid"));
  const customBtn = segButton(typeSeg, "Custom", false, () => store.set("doc.design.type", "custom"));
  controls.appendChild(typeSeg);

  const rectFields = document.createElement("div");
  const rectInputs = {};
  for (const field of HELIOSTAT_RECT_FIELDS) rectInputs[field.key] = numberRow(rectFields, field);
  controls.appendChild(rectFields);

  const gridFields = document.createElement("div");
  const gridInputs = {};
  for (const field of HELIOSTAT_GRID_FIELDS) gridInputs[field.key] = numberRow(gridFields, field);
  controls.appendChild(gridFields);

  const customFields = document.createElement("div");
  const vertexTableWrap = document.createElement("div");
  customFields.appendChild(vertexTableWrap);
  const addVertexBtn = document.createElement("div");
  addVertexBtn.className = "btn small";
  addVertexBtn.textContent = "+ Add vertex";
  addVertexBtn.style.marginBottom = "8px";
  addVertexBtn.addEventListener("click", addCustomVertex);
  customFields.appendChild(addVertexBtn);
  const mirrorRow = document.createElement("div");
  mirrorRow.className = "frow";
  const mirrorLabel = document.createElement("label");
  mirrorLabel.textContent = "Mirror symmetry";
  const mirrorInput = document.createElement("input");
  mirrorInput.type = "checkbox";
  mirrorInput.addEventListener("change", () => setMirror(mirrorInput.checked));
  mirrorRow.appendChild(mirrorLabel);
  mirrorRow.appendChild(mirrorInput);
  customFields.appendChild(mirrorRow);
  controls.appendChild(customFields);

  const typeHint = document.createElement("div");
  typeHint.className = "hint";
  typeHint.style.marginLeft = "0";
  typeHint.textContent = "Custom lets you sketch your own outline — drag its corners, or click a side to type its exact length.";
  controls.appendChild(typeHint);

  const surfaceSub = document.createElement("div");
  surfaceSub.className = "subhead";
  surfaceSub.textContent = "Surface figure";
  controls.appendChild(surfaceSub);
  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const surfaceBtns = {};
  for (const [key, label] of HELIOSTAT_SURFACE_OPTIONS) {
    surfaceBtns[key] = segButton(surfaceSeg, label, key === "twisting", () => {
      store.set("doc.design.surface", key);
      if (key === "twisting") {
        if (store.get("doc.designParams.grid.cant_focal_mm") !== null) store.set("doc.designParams.grid.cant_focal_mm", null);
        if (store.get("doc.designParams.grid.facet_focal_mm") !== null) store.set("doc.designParams.grid.facet_focal_mm", null);
      } else if (key === "flat" && store.get("doc.designParams.grid.facet_focal_mm") === null) {
        store.set("doc.designParams.grid.facet_focal_mm", 0);
      }
    });
  }
  controls.appendChild(surfaceSeg);
  const surfaceHint = document.createElement("div");
  surfaceHint.className = "hint";
  surfaceHint.style.marginLeft = "0";
  surfaceHint.textContent =
    "Twisting re-solves the figure as the sun moves. Spherical freezes one figure (long focals give weakly focusing, not-quite-flat facets); flat has none by default, though a facet grid can still be made weakly focusing below.";
  controls.appendChild(surfaceHint);

  // -- facet curvature: the facet's own surface shape, independent of
  // where it's aimed. Locked under Twisting (the astigmatic figure is
  // solved for you there); simplified to a single checkbox under Flat.
  const curvSub = document.createElement("div");
  curvSub.className = "subhead";
  curvSub.textContent = "Facet curvature";
  controls.appendChild(curvSub);
  function curvDisabledNow() {
    const doc = store.get("doc");
    return doc.design.type !== "grid" || doc.design.surface === "twisting";
  }
  const curvSeg = document.createElement("div");
  curvSeg.className = "seg";
  const curvBtns = {
    off: segButton(curvSeg, "No curvature", false, () => {
      if (curvDisabledNow()) return;
      store.set("doc.designParams.grid.facet_focal_mm", 0);
    }),
    auto: segButton(curvSeg, "Follow canting", false, () => {
      if (curvDisabledNow()) return;
      store.set("doc.designParams.grid.facet_focal_mm", null);
    }),
    focal: segButton(curvSeg, "Fixed focal…", false, () => {
      if (curvDisabledNow()) return;
      const cur = store.get("doc.designParams.grid.facet_focal_mm");
      if (!(cur > 0)) store.set("doc.designParams.grid.facet_focal_mm", 60000);
    }),
  };
  controls.appendChild(curvSeg);
  const curvWeakRow = document.createElement("div");
  curvWeakRow.className = "frow";
  const curvWeakLabel = document.createElement("label");
  curvWeakLabel.textContent = "Weakly focusing";
  const curvWeakInput = document.createElement("input");
  curvWeakInput.type = "checkbox";
  curvWeakInput.addEventListener("change", () => {
    if (curvDisabledNow()) return;
    if (curvWeakInput.checked) {
      const cur = store.get("doc.designParams.grid.facet_focal_mm");
      store.set("doc.designParams.grid.facet_focal_mm", cur > 0 ? cur : 200000);
    } else {
      store.set("doc.designParams.grid.facet_focal_mm", 0);
    }
  });
  curvWeakRow.appendChild(curvWeakLabel);
  curvWeakRow.appendChild(curvWeakInput);
  controls.appendChild(curvWeakRow);
  const curvFocalRow = document.createElement("div");
  const curvFocalInput = numberRow(curvFocalRow, FACET_FOCAL_FIELD);
  controls.appendChild(curvFocalRow);
  const curvHint = document.createElement("div");
  curvHint.className = "hint";
  curvHint.style.marginLeft = "0";
  controls.appendChild(curvHint);

  // -- canting: where each facet is aimed, independent of its curvature.
  // Locked under Twisting for the same reason as curvature.
  const cantSub = document.createElement("div");
  cantSub.className = "subhead";
  cantSub.textContent = "Canting (facet aiming)";
  controls.appendChild(cantSub);
  const cantSeg = document.createElement("div");
  cantSeg.className = "seg";
  function cantDisabledNow() {
    const doc = store.get("doc");
    return doc.design.type !== "grid" || doc.design.surface === "twisting";
  }
  const cantBtns = {
    off: segButton(cantSeg, "Off", false, () => {
      if (cantDisabledNow()) return;
      store.set("doc.designParams.grid.cant_focal_mm", 0);
    }),
    auto: segButton(cantSeg, "Per heliostat", false, () => {
      if (cantDisabledNow()) return;
      store.set("doc.designParams.grid.cant_focal_mm", null);
    }),
    focal: segButton(cantSeg, "Fixed focal…", false, () => {
      if (cantDisabledNow()) return;
      const cur = store.get("doc.designParams.grid.cant_focal_mm");
      if (!(cur > 0)) store.set("doc.designParams.grid.cant_focal_mm", 60000);
    }),
  };
  controls.appendChild(cantSeg);
  const cantFocalRow = document.createElement("div");
  const cantFocalInput = numberRow(cantFocalRow, CANT_FOCAL_FIELD);
  controls.appendChild(cantFocalRow);
  const cantHint = document.createElement("div");
  cantHint.className = "hint";
  cantHint.style.marginLeft = "0";
  controls.appendChild(cantHint);

  const errorsSub = document.createElement("div");
  errorsSub.className = "subhead";
  errorsSub.textContent = "Optical errors";
  controls.appendChild(errorsSub);
  const errorInputs = {};
  for (const field of ERROR_FIELDS) errorInputs[field.key] = numberRow(controls, field);
  const errorHint = document.createElement("div");
  errorHint.className = "hint";
  errorHint.style.marginLeft = "0";
  errorHint.textContent = "Used by Monte Carlo traces and carried through SolTrace / SolarPILOT export.";
  controls.appendChild(errorHint);

  // -- previews: aperture layout + sag map -----------------------------
  const previews = document.createElement("div");
  previews.className = "previews";

  const aperturePanel = document.createElement("div");
  aperturePanel.className = "panel previewpanel";
  const apertureHead = document.createElement("div");
  apertureHead.style.display = "flex";
  apertureHead.style.alignItems = "center";
  const apertureH2 = document.createElement("h2");
  apertureH2.textContent = "Aperture layout";
  apertureH2.style.flex = "1 1 auto";
  const apertureToggle = document.createElement("div");
  apertureToggle.className = "btn small";
  apertureToggle.hidden = true;
  apertureHead.appendChild(apertureH2);
  apertureHead.appendChild(apertureToggle);
  aperturePanel.appendChild(apertureHead);
  const apertureFrame = document.createElement("div");
  apertureFrame.className = "frame";
  const apertureImg = document.createElement("img");
  apertureImg.alt = "Aperture layout preview";
  apertureImg.hidden = true;
  const sketchSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sketchSvg.setAttribute("class", "sketch");
  sketchSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  sketchSvg.hidden = true;
  const aperturePlaceholder = document.createElement("p");
  aperturePlaceholder.className = "placeholder";
  aperturePlaceholder.hidden = true;
  apertureFrame.appendChild(apertureImg);
  apertureFrame.appendChild(sketchSvg);
  apertureFrame.appendChild(aperturePlaceholder);
  aperturePanel.appendChild(apertureFrame);
  const apertureCaption = document.createElement("div");
  apertureCaption.className = "caption";
  aperturePanel.appendChild(apertureCaption);

  apertureToggle.addEventListener("click", () => {
    showServerRenderForCustom = !showServerRenderForCustom;
    lastPreviewKey = null; // force a re-fetch/redraw on the new mode
    rerender();
  });

  sketchSvg.addEventListener("pointerdown", (e) => {
    const vEl = e.target.closest && e.target.closest("[data-vertex-index]");
    if (!vEl) return;
    e.preventDefault();
    e.stopPropagation();
    startDragVertex(Number(vEl.dataset.vertexIndex));
  });
  sketchSvg.addEventListener("click", (e) => {
    const lEl = e.target.closest && e.target.closest("[data-length-index]");
    if (lEl) {
      onLengthLabelClick(Number(lEl.dataset.lengthIndex));
      return;
    }
    const vEl = e.target.closest && e.target.closest("[data-vertex-index]");
    if (vEl) return;
    if (!sketchProj) return;
    const rect = sketchSvg.getBoundingClientRect();
    const [u, v] = fromPx(sketchProj, e.clientX - rect.left, e.clientY - rect.top);
    insertVertexNearestSegment(u, v);
  });
  window.addEventListener("resize", () => {
    if (store.get("doc").design.type === "custom" && !showServerRenderForCustom && !sketchSvg.hidden) renderSketchNow();
  });

  const sagPanel = document.createElement("div");
  sagPanel.className = "panel previewpanel";
  const sagHead = document.createElement("div");
  sagHead.style.display = "flex";
  sagHead.style.alignItems = "flex-start";
  sagHead.style.gap = "10px";
  const sagH2 = document.createElement("h2");
  sagH2.style.flex = "1 1 auto";
  const locatorSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  locatorSvg.setAttribute("class", "locator");
  locatorSvg.setAttribute("width", "84");
  locatorSvg.setAttribute("height", "56");
  locatorSvg.setAttribute("viewBox", "0 0 84 56");
  sagHead.appendChild(sagH2);
  sagHead.appendChild(locatorSvg);
  sagPanel.appendChild(sagHead);
  const sagFrame = document.createElement("div");
  sagFrame.className = "frame";
  const sagImg = document.createElement("img");
  sagImg.alt = "Sag map";
  sagImg.hidden = true;
  const sagPlaceholder = document.createElement("p");
  sagPlaceholder.className = "placeholder";
  sagPlaceholder.hidden = true;
  sagFrame.appendChild(sagImg);
  sagFrame.appendChild(sagPlaceholder);
  sagPanel.appendChild(sagFrame);
  const sagCaption = document.createElement("div");
  sagCaption.className = "caption";
  const sagCaption2 = document.createElement("div");
  sagCaption2.className = "caption";
  sagPanel.appendChild(sagCaption);
  sagPanel.appendChild(sagCaption2);

  locatorSvg.addEventListener("click", (e) => {
    const geometry = (lastCtx && lastCtx.geometry) || null;
    const heliostats = (geometry && geometry.heliostats) || [];
    if (!heliostats.length || !locatorProj) return;
    const rect = locatorSvg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * 84;
    const py = ((e.clientY - rect.top) / rect.height) * 56;
    const um = ((px - locatorProj.cx) / locatorProj.scale) * 1000;
    const vm = (-(py - locatorProj.cy) / locatorProj.scale) * 1000;
    let best = null;
    let bestD = Infinity;
    for (const h of heliostats) {
      const d = Math.hypot(h.x_mm - um, h.y_mm - vm);
      if (d < bestD) {
        bestD = d;
        best = h;
      }
    }
    if (best) store.set("ui.shapeHeliostatId", best.id);
  });

  previews.appendChild(aperturePanel);
  previews.appendChild(sagPanel);

  content.appendChild(controls);
  content.appendChild(previews);

  container.appendChild(editbar);
  container.appendChild(content);

  els = {
    nameEl,
    chip,
    chipId,
    chipR,
    popover,
    popInput,
    saveWrap,
    saveBtn,
    saveInline,
    saveNameInput,
    saveErrEl,
    typeSeg,
    rectBtn,
    gridBtn,
    customBtn,
    rectFields,
    rectInputs,
    gridFields,
    gridInputs,
    customFields,
    typeHint,
    vertexTableWrap,
    mirrorInput,
    surfaceBtns,
    curvSeg,
    curvBtns,
    curvWeakRow,
    curvWeakInput,
    curvFocalRow,
    curvFocalInput,
    curvHint,
    cantSeg,
    cantBtns,
    cantFocalRow,
    cantFocalInput,
    cantHint,
    errorInputs,
    apertureH2,
    apertureToggle,
    apertureFrame,
    apertureImg,
    sketchSvg,
    aperturePlaceholder,
    apertureCaption,
    sagH2,
    locatorSvg,
    sagFrame,
    sagImg,
    sagPlaceholder,
    sagCaption,
    sagCaption2,
    // vertex-table row bookkeeping (rebuilt only when the count changes --
    // see renderCustomVertexRows)
    customRowEls: [],
  };
  built = true;
}

// -- custom vertex table: rebuild rows only on a structural change --------

function renderCustomVertexRows(vertices) {
  if (els.customRowEls.length !== vertices.length) {
    els.vertexTableWrap.innerHTML = "";
    els.customRowEls = [];
    vertices.forEach((_, i) => {
      const row = document.createElement("div");
      row.className = "frow";
      const uLabel = document.createElement("label");
      uLabel.textContent = `V${i} u/v`;
      uLabel.style.flex = "0 0 60px";
      const uInput = document.createElement("input");
      uInput.type = "number";
      uInput.className = "val";
      uInput.addEventListener("input", () => {
        const v = parseFloat(uInput.value);
        if (Number.isFinite(v)) store.set(`doc.designParams.custom.vertices_mm.${i}.0`, v);
      });
      const vInput = document.createElement("input");
      vInput.type = "number";
      vInput.className = "val";
      vInput.addEventListener("input", () => {
        const v = parseFloat(vInput.value);
        if (Number.isFinite(v)) store.set(`doc.designParams.custom.vertices_mm.${i}.1`, v);
      });
      const rmBtn = document.createElement("div");
      rmBtn.className = "btn small";
      rmBtn.textContent = "×";
      rmBtn.style.flex = "0 0 auto";
      rmBtn.addEventListener("click", () => removeCustomVertex(i));
      row.appendChild(uLabel);
      row.appendChild(uInput);
      row.appendChild(vInput);
      row.appendChild(rmBtn);
      els.vertexTableWrap.appendChild(row);
      els.customRowEls.push({ u: uInput, v: vInput, rm: rmBtn });
    });
  }
  vertices.forEach((v, i) => {
    const row = els.customRowEls[i];
    setVal(row.u, v[0]);
    setVal(row.v, v[1]);
    row.rm.classList.toggle("disabled-link", vertices.length <= 3);
  });
}

// -- render -------------------------------------------------------------

function renderEditbar(doc, ui, previewHeliostat) {
  els.nameEl.textContent = `Editing: ${designSummaryText(doc)}`;

  if (previewHeliostat) {
    els.chipId.textContent = `H-${previewHeliostat.id}`;
    const r = slantRangeLabel(previewHeliostat);
    els.chipR.textContent = r != null ? `· r ${r} m` : "· r —";
  } else {
    els.chipId.textContent = "—";
    els.chipR.textContent = "";
  }
  els.popover.hidden = !popoverOpen;
  if (popoverOpen) {
    setVal(els.popInput, ui.shapeHeliostatId != null ? ui.shapeHeliostatId : previewHeliostat ? previewHeliostat.id : "");
  }

  els.saveInline.hidden = !saveOpen;
  els.saveBtn.hidden = saveOpen;
  if (saveError) {
    els.saveErrEl.hidden = false;
    els.saveErrEl.textContent = saveError;
  } else if (saveSaved) {
    els.saveErrEl.hidden = false;
    els.saveErrEl.className = "hint";
    els.saveErrEl.textContent = "Saved.";
  } else {
    els.saveErrEl.hidden = true;
    els.saveErrEl.className = "fielderr";
  }
}

function renderDesignControls(doc, previewHeliostat) {
  const cantSlantM = slantRangeLabel(previewHeliostat);
  const cantPreviewId = previewHeliostat ? previewHeliostat.id : null;
  const type = doc.design.type;
  els.rectBtn.classList.toggle("active", type === "rect");
  els.gridBtn.classList.toggle("active", type === "grid");
  els.customBtn.classList.toggle("active", type === "custom");
  els.rectFields.style.display = type === "rect" ? "" : "none";
  els.gridFields.style.display = type === "grid" ? "" : "none";
  els.customFields.style.display = type === "custom" ? "" : "none";
  // Once Custom is active the sketch canvas caption explains itself.
  els.typeHint.style.display = type === "custom" ? "none" : "";

  if (type === "rect") {
    const p = doc.designParams.rect;
    setVal(els.rectInputs.width_mm, p.width_mm);
    setVal(els.rectInputs.height_mm, p.height_mm);
  } else if (type === "grid") {
    const p = doc.designParams.grid;
    setVal(els.gridInputs.n_u, p.n_u);
    setVal(els.gridInputs.n_v, p.n_v);
    setVal(els.gridInputs.facet_w_mm, p.facet_w_mm);
    setVal(els.gridInputs.facet_h_mm, p.facet_h_mm);
    setVal(els.gridInputs.gap_mm, p.gap_mm);
  } else {
    const p = doc.designParams.custom;
    renderCustomVertexRows(p.vertices_mm);
    if (document.activeElement !== els.mirrorInput) els.mirrorInput.checked = !!p.mirror;
  }

  const surface = doc.design.surface;
  for (const [key, btn] of Object.entries(els.surfaceBtns)) btn.classList.toggle("active", key === surface);

  const isGrid = type === "grid";
  const isTwisting = surface === "twisting";
  const isFlat = surface === "flat";
  const disabled = !isGrid || isTwisting;

  // -- facet curvature: seg (spherical) or a single checkbox (flat) -----
  const facetFocalMm = isGrid ? doc.designParams.grid.facet_focal_mm : null;
  const curvRawMode = isGrid ? curvatureMode(facetFocalMm) : "auto";
  const curvDisplayMode = disabled ? "auto" : curvRawMode;
  const showCurvSeg = isGrid && !isTwisting && !isFlat;
  const showCurvCheckbox = isGrid && !isTwisting && isFlat;
  els.curvSeg.style.display = showCurvSeg ? "" : "none";
  els.curvSeg.classList.toggle("disabled", disabled);
  els.curvBtns.off.classList.toggle("active", curvDisplayMode === "off");
  els.curvBtns.auto.classList.toggle("active", curvDisplayMode === "auto");
  els.curvBtns.focal.classList.toggle("active", curvDisplayMode === "focal");
  for (const btn of Object.values(els.curvBtns)) btn.disabled = disabled;

  els.curvWeakRow.style.display = showCurvCheckbox ? "" : "none";
  els.curvWeakInput.disabled = disabled;
  if (showCurvCheckbox && document.activeElement !== els.curvWeakInput) els.curvWeakInput.checked = facetFocalMm > 0;

  const showCurvFocal = (showCurvSeg && curvDisplayMode === "focal") || (showCurvCheckbox && facetFocalMm > 0);
  els.curvFocalRow.style.display = showCurvFocal ? "" : "none";
  if (showCurvFocal) setVal(els.curvFocalInput, facetFocalMm);

  if (!isGrid) {
    els.curvHint.textContent = "Single mirror — no facets to curve.";
  } else if (isTwisting) {
    els.curvHint.textContent = "Twisting solves each facet's own curvature for you, along with its aim.";
  } else if (isFlat) {
    els.curvHint.textContent =
      facetFocalMm > 0
        ? "Facets share one long fixed focal, giving the panel a gentle concentrating figure while staying flat to build."
        : "Facets are flat panels — check to give them a long, gentle fixed curvature without leaving Flat.";
  } else if (curvDisplayMode === "off") {
    els.curvHint.textContent = "Facets are flat panels — no curvature of their own.";
  } else if (curvDisplayMode === "auto") {
    els.curvHint.textContent = "Each facet curves to match its own canting distance, so it focuses exactly where it's aimed.";
  } else {
    els.curvHint.textContent = "One curvature for every facet in the field, independent of where each one is aimed.";
  }

  // -- canting: where each facet is aimed --------------------------------
  const cantRawMode = isGrid ? cantMode(doc.designParams.grid.cant_focal_mm) : "auto";
  const cantDisplayMode = disabled ? "auto" : cantRawMode;
  els.cantSeg.classList.toggle("disabled", disabled);
  els.cantBtns.off.classList.toggle("active", cantDisplayMode === "off");
  els.cantBtns.auto.classList.toggle("active", cantDisplayMode === "auto");
  els.cantBtns.focal.classList.toggle("active", cantDisplayMode === "focal");
  for (const btn of Object.values(els.cantBtns)) btn.disabled = disabled;
  els.cantFocalRow.style.display = cantDisplayMode === "focal" ? "" : "none";
  if (cantDisplayMode === "focal") setVal(els.cantFocalInput, doc.designParams.grid.cant_focal_mm);

  if (!isGrid) {
    els.cantHint.textContent = "Single mirror — no facets to cant.";
  } else if (isTwisting) {
    els.cantHint.textContent = "Twisting solves figure and aim together, so canting is chosen for you.";
  } else if (cantDisplayMode === "off") {
    els.cantHint.textContent = "Facets stay parallel — the mirror acts as one flat panel, so each facet sends its own separate spot.";
  } else if (cantDisplayMode === "auto") {
    els.cantHint.textContent =
      `Each heliostat aims its facets at its own distance to the aim point${cantSlantM ? ` — ${cantSlantM} m for the previewed H-${cantPreviewId}` : ""}, ` +
      "so every position in the field is individually optimal. Different field positions get physically different heliostats.";
  } else {
    els.cantHint.textContent =
      `One aim distance for the whole field — every heliostat points its facets the same way${cantSlantM ? `, whether it stands at ${cantSlantM} m like the previewed H-${cantPreviewId} or anywhere else` : ""}. ` +
      "That is the buildable case: one part number, at the cost of performance away from this range.";
  }

  const errors = doc.design.errors;
  setVal(els.errorInputs.slope_error_mrad, errors.slope_error_mrad);
  setVal(els.errorInputs.specularity_mrad, errors.specularity_mrad);
  setVal(els.errorInputs.reflectance_pct, errors.reflectance_pct);
}

function setPreviewImageBlob(blob) {
  if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
  previewObjectUrl = URL.createObjectURL(blob);
  els.apertureImg.src = previewObjectUrl;
  els.apertureImg.hidden = false;
  els.aperturePlaceholder.hidden = true;
  els.apertureCaption.textContent = "Live server render — updates as you type.";
}

function setPreviewError(err) {
  els.apertureImg.hidden = true;
  els.aperturePlaceholder.hidden = false;
  els.aperturePlaceholder.textContent = (err && err.message) || "Could not render the aperture layout.";
}

function renderAperturePreview(doc) {
  const type = doc.design.type;
  els.apertureToggle.hidden = type !== "custom";
  els.apertureToggle.textContent = showServerRenderForCustom ? "Show sketch" : "Show server render";

  const useSketch = type === "custom" && !showServerRenderForCustom;
  els.sketchSvg.hidden = !useSketch;
  els.apertureImg.hidden = useSketch || els.apertureImg.hidden;
  els.aperturePlaceholder.hidden = useSketch || els.aperturePlaceholder.hidden;

  if (useSketch) {
    els.apertureCaption.textContent =
      "Sketch — drag a vertex to move it, click empty space to add one, click a length to type it exactly.";
    renderSketchNow();
    return;
  }

  const design = currentDesignPayload(doc);
  const key = JSON.stringify(design);
  if (key === lastPreviewKey) return;
  lastPreviewKey = key;
  schedulePreview(
    (signal) => postDesignPreview(design, signal),
    (blob) => setPreviewImageBlob(blob),
    (err) => setPreviewError(err)
  );
}

function setSagImageBlob(result) {
  if (sagObjectUrl) URL.revokeObjectURL(sagObjectUrl);
  sagObjectUrl = URL.createObjectURL(result.blob);
  els.sagImg.src = sagObjectUrl;
  els.sagImg.hidden = false;
  els.sagPlaceholder.hidden = true;
  lastSagResult = result;
}

function setSagError(err) {
  els.sagImg.hidden = true;
  els.sagPlaceholder.hidden = false;
  els.sagPlaceholder.textContent = (err && err.message) || "Could not render the sag map.";
}

function renderSagPanel(doc, previewHeliostat) {
  const id = previewHeliostat ? previewHeliostat.id : null;
  els.sagH2.textContent = id != null ? `Sag map — H-${id}` : "Sag map";

  if (!previewHeliostat) {
    lastSagKey = null;
    setSagError({ message: "No heliostats in the field yet." });
    els.sagCaption.textContent = "";
    els.sagCaption2.textContent = "";
    return;
  }

  const body = buildSagRequest(doc, { x_mm: previewHeliostat.x_mm, y_mm: previewHeliostat.y_mm });
  const key = JSON.stringify(body);
  if (key !== lastSagKey) {
    lastSagKey = key;
    scheduleSag(
      (signal) => postDesignSag(body, signal),
      (result) => {
        setSagImageBlob(result);
        renderSagCaption(doc);
      },
      (err) => {
        setSagError(err);
        els.sagCaption.textContent = "";
        els.sagCaption2.textContent = "";
      }
    );
  } else {
    renderSagCaption(doc);
  }
}

function renderSagCaption(doc) {
  const contour = lastSagResult && lastSagResult.contourIntervalMm != null ? lastSagResult.contourIntervalMm.toFixed(1) : "—";
  // A faceted design's map is each facet's OWN figure, measured from its own
  // mounting plane -- the canting tilt that aims the facets is a rigid
  // rotation the trace applies separately, so it is not in these numbers.
  const perFacet = doc.design.type === "grid" ? " · per facet, canting removed" : "";
  els.sagCaption.textContent =
    `surface sag (mm) · contours every ${contour} mm${perFacet} · ${doc.design.surface} figure at ` +
    `Az ${doc.sun.az.toFixed(1)}° El ${doc.sun.el.toFixed(1)}°`;
  els.sagCaption2.textContent =
    "Figure solved for the workspace's current sun and this heliostat's slant range. Pick another heliostat from the " +
    "locator above, the ▾ selector, or by clicking one in the workspace and choosing “View shape”.";
}

function renderLocator(ui, geometry, previewHeliostat) {
  const heliostats = (geometry && geometry.heliostats) || [];
  const w = 84;
  const h = 56;
  let maxR = 0;
  for (const hh of heliostats) {
    const r = Math.hypot(hh.x_mm, hh.y_mm) / 1000;
    if (r > maxR) maxR = r;
  }
  if (maxR < 5) maxR = 5;
  const scale = Math.max((Math.min(w, h) / 2 - 6) / (maxR * 1.08), 0.001);
  const cx = w / 2;
  const cy = h / 2;
  locatorProj = { scale, cx, cy };
  const previewId = previewHeliostat ? previewHeliostat.id : null;
  let s = `<rect width="${w}" height="${h}" fill="#fdfdfe"></rect>`;
  for (const hh of heliostats) {
    const x = cx + (hh.x_mm / 1000) * scale;
    const y = cy - (hh.y_mm / 1000) * scale;
    const isSel = hh.id === previewId;
    if (isSel) {
      s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="none" stroke="#0b5fd0" stroke-width="1.4"></circle>`;
      s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.4" fill="#0b5fd0"></circle>`;
    } else {
      s += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1" fill="#8aa5c2"></circle>`;
    }
  }
  s += `<rect x="${(cx - 2).toFixed(1)}" y="${(cy - 2).toFixed(1)}" width="4" height="4" fill="rgba(217,123,41,0.8)"></rect>`;
  els.locatorSvg.innerHTML = s;
}

export function render(container, ctx) {
  if (!built) build(container);
  lastContainer = container;
  lastCtx = ctx;

  const doc = store.get("doc");
  const ui = store.get("ui");
  const geometry = (ctx && ctx.geometry) || null;
  const previewHeliostat = pickPreviewHeliostat(ui, geometry);

  renderEditbar(doc, ui, previewHeliostat);
  renderDesignControls(doc, previewHeliostat);
  renderAperturePreview(doc);
  renderLocator(ui, geometry, previewHeliostat);
  renderSagPanel(doc, previewHeliostat);
}
