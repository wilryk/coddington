// Tower elevation view (docs/ui-spec.md 2.2, mockup M4): a to-scale side
// elevation with SolidWorks-style dimension callouts, active while the
// Receiver & Tower stage is expanded (docs/ui-spec.md 2.1's "the viewport
// follows the active stage" -- wired in ../main.js). Same store-but-not-
// geometry pattern as ./plan.js: main.js pushes the latest
// /api/scene/geometry response in through setGeometry().
//
// SCOPE (phase 3b): the ground offset is drawn and dimensioned but NOT
// editable yet (its own callout says so on hover) -- the datum/ground-offset
// work is a later batch; only the per-optics height/aperture/angle fields
// are live callouts here.
//
// Vertical plane: horizontal = world y (the field lies south of the tower,
// so heliostats have y < 0 -- src/heliostat/web/scene.py's own frame),
// vertical = z, both in metres (mm / 1000). x is dropped throughout (every
// heliostat, ray and dimension is projected onto this one plane).
import { store } from "../store.js";
import { resolvePath, setVal, RECEIVER_FIELD_TABLE, receiverFieldVisible } from "../fields.js";

// The default heliostat-plane-above-ground offset (docs/ui-spec.md 2.2):
// "a separate ground offset parameter ... default 2 500 mm". Editing it is
// out of scope for this phase (SCOPE above) -- see the ground-offset
// callout's own non-editable box and title tooltip.
const GROUND_OFFSET_MM = 2500;

// Which RECEIVER_FIELD_TABLE keys get a dimension callout in this view, per
// optics (docs/ui-spec.md 2.2's per-layout table, minus the window ½w/½h
// fields -- those aren't dimensioned in the elevation, only in the sidebar).
const CALLOUT_KEYS = {
  prime_focus: [
    "focus_height_mm",
    "aperture_to_receiver_mm",
    "cylinder_radius_mm",
    "cylinder_height_mm",
    "frustum_top_radius_mm",
    "frustum_bottom_radius_mm",
    "frustum_height_mm",
  ],
  axicon: ["apex_height_mm", "receiver_z_mm", "aperture_radius_mm", "half_angle_deg"],
  cassegrain: ["vertex_z_mm", "focus_height_mm", "receiver_z_mm", "aperture_radius_mm"],
};

const CALLOUT_LABELS = {
  apex_height_mm: "apex height",
  vertex_z_mm: "vertex height",
  focus_height_mm: (optics) => (optics === "cassegrain" ? "primary focus height" : "focus height"),
  receiver_z_mm: "receiver height",
  aperture_radius_mm: "aperture radius",
  half_angle_deg: "half angle",
  aperture_to_receiver_mm: "aperture → receiver",
  cylinder_radius_mm: "cylinder radius",
  cylinder_height_mm: "cylinder height",
  frustum_top_radius_mm: "frustum top radius",
  frustum_bottom_radius_mm: "frustum bottom radius",
  frustum_height_mm: "frustum height",
};

let built = false;
let els = {};
let lastGeometry = null;
let builtContainer = null;

// v0.2 fix wave item 4: same zoom/pan mechanism as ./plan.js (see its own
// comment for the full rationale) -- null = auto-fit, else the live
// {scale, panX, panY} on top of the height-driven fit computeProjection
// would otherwise recompute from scratch on every render.
let manualView = null;
let lastProj = null;
let suppressNextClick = false;

// Called by main.js on every successful /api/scene/geometry response --
// never on error, so (like the 3D scene) this view keeps drawing the last
// valid geometry while ui.geometryError is set (docs/ui-spec.md 2.3).
export function setGeometry(data) {
  lastGeometry = data;
}

function resetView() {
  if (!manualView) return;
  manualView = null;
  if (builtContainer) render(builtContainer);
}

// Callouts live in a sibling HTML layer, not inside the SVG (see build()),
// so a click on one never reaches this handler at all -- anything that does
// arrive here is either a data-kind shape or empty ground.
function handleClick(e) {
  if (suppressNextClick) {
    // Mirrors plan.js's own guard: a pan drag that ended here must not also
    // fire the click semantics (select/deselect) under the pointer's rest
    // position -- see handlePointerUp below.
    suppressNextClick = false;
    return;
  }
  const el = e.target.closest && e.target.closest("[data-kind]");
  if (!el) {
    store.set("ui.selection", null); // empty ground deselects
    return;
  }
  const kind = el.dataset.kind;
  store.set("ui.selection", { kind, id: null });
}

// Wheel over the SVG zooms about the cursor -- identical construction to
// ./plan.js's own handleWheel, just against this view's (y, z) projection.
const ZOOM_MIN_FACTOR = 0.15;
const ZOOM_MAX_FACTOR = 40;

function handleWheel(e) {
  if (!lastProj) return;
  e.preventDefault();
  const rect = els.svg.getBoundingClientRect();
  const mx = ((e.clientX - rect.left) / rect.width) * lastProj.w;
  const my = ((e.clientY - rect.top) / rect.height) * lastProj.h;
  const proj = lastProj;
  const factor = Math.exp(-e.deltaY * 0.0015);
  const minScale = proj.fitScale * ZOOM_MIN_FACTOR;
  const maxScale = proj.fitScale * ZOOM_MAX_FACTOR;
  const newScale = Math.min(maxScale, Math.max(minScale, proj.scale * factor));

  // toScreen: sx = cx + ym*scale, sy = groundPx - (zm-zBottom)*scale --
  // solve for the world point under the cursor, then pick the new cx/groundPx
  // (as pan offsets from the fit basis) that put it back under the cursor.
  const ym = (mx - proj.cx) / proj.scale;
  const zm = proj.zBottom + (proj.groundPx - my) / proj.scale;
  const newCx = mx - ym * newScale;
  const newGroundPx = my + (zm - proj.zBottom) * newScale;
  manualView = { scale: newScale, panX: newCx - proj.baseCx, panY: newGroundPx - proj.baseGroundPx };
  render(builtContainer);
}

const PAN_DRAG_THRESHOLD_PX = 3;
let dragState = null;

function handlePointerDown(e) {
  if (e.button !== 0 || !lastProj) return;
  dragState = {
    startX: e.clientX,
    startY: e.clientY,
    startPanX: manualView ? manualView.panX : 0,
    startPanY: manualView ? manualView.panY : 0,
    moved: false,
  };
  els.svg.setPointerCapture(e.pointerId);
}

function handlePointerMove(e) {
  if (!dragState) return;
  const dx = e.clientX - dragState.startX;
  const dy = e.clientY - dragState.startY;
  if (!dragState.moved && Math.hypot(dx, dy) < PAN_DRAG_THRESHOLD_PX) return;
  dragState.moved = true;
  const scale = manualView ? manualView.scale : lastProj.fitScale;
  manualView = { scale, panX: dragState.startPanX + dx, panY: dragState.startPanY + dy };
  render(builtContainer);
}

function handlePointerUp(e) {
  if (dragState && dragState.moved) suppressNextClick = true;
  dragState = null;
  if (els.svg.hasPointerCapture && els.svg.hasPointerCapture(e.pointerId)) {
    els.svg.releasePointerCapture(e.pointerId);
  }
}

function build(container) {
  container.innerHTML =
    '<svg preserveAspectRatio="xMidYMid meet">' +
    '<rect x="0" y="0" fill="#fdfdfe"></rect>' +
    '<g data-layer="ground"></g>' +
    '<g data-layer="rays"></g>' +
    '<g data-layer="mast"></g>' +
    '<g data-layer="secondary"></g>' +
    '<g data-layer="receiver"></g>' +
    '<g data-layer="heliostats"></g>' +
    '<g data-layer="dims"></g>' +
    '<g data-layer="sun"></g>' +
    "</svg>";
  const svg = container.querySelector("svg");
  const bg = container.querySelector("rect");
  const layers = {};
  for (const name of ["ground", "rays", "mast", "secondary", "receiver", "heliostats", "dims", "sun"]) {
    layers[name] = container.querySelector('[data-layer="' + name + '"]');
  }
  svg.addEventListener("click", handleClick);
  svg.addEventListener("dblclick", (e) => {
    e.preventDefault();
    resetView();
  });
  svg.addEventListener("wheel", handleWheel, { passive: false });
  svg.addEventListener("pointerdown", handlePointerDown);
  svg.addEventListener("pointermove", handlePointerMove);
  svg.addEventListener("pointerup", handlePointerUp);
  svg.addEventListener("pointercancel", handlePointerUp);

  // Reset-view chip (v0.2 fix wave item 4's "double-click or a small reset
  // control"), a plain HTML button rather than SVG chrome -- simplest way to
  // share the callout layer's overlay pattern. Shown only while manualView
  // is non-null (render() below toggles it), top-left where nothing else in
  // this view ever draws.
  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.className = "elevation-reset-view";
  resetBtn.textContent = "Reset view";
  resetBtn.hidden = true;
  resetBtn.addEventListener("click", resetView);
  container.appendChild(resetBtn);

  // Callouts float in an HTML layer over the SVG (real <input>s, not SVG
  // text) so fields.js's setVal/focused-input guard works exactly as it
  // does in the sidebar -- built ONCE here and only repositioned/hidden on
  // each render, never recreated, or typing would lose focus every 300ms
  // debounce tick (same reasoning as the sidebar panels' own built/els split).
  const calloutLayer = document.createElement("div");
  calloutLayer.className = "elevation-callouts";
  container.appendChild(calloutLayer);

  const callouts = {};
  for (const [optics, keys] of Object.entries(CALLOUT_KEYS)) {
    for (const key of keys) {
      const id = optics + "." + key;
      const field = (RECEIVER_FIELD_TABLE[optics] || []).find((f) => f.key === key);
      if (!field) continue;
      const wrap = document.createElement("div");
      wrap.className = "callout";
      const input = document.createElement("input");
      input.type = "number";
      input.className = "val";
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        if (Number.isFinite(v)) {
          const doc = store.get("doc");
          store.set(resolvePath(field.path, doc), v);
        }
      });
      const lab = document.createElement("div");
      lab.className = "calloutlabel";
      const labelSource = CALLOUT_LABELS[key];
      lab.textContent = typeof labelSource === "function" ? labelSource(optics) : labelSource;
      wrap.appendChild(input);
      wrap.appendChild(lab);
      calloutLayer.appendChild(wrap);
      callouts[id] = { wrap, input };
    }
  }

  // Ground offset: same visual language, not editable yet (SCOPE above).
  const groundWrap = document.createElement("div");
  groundWrap.className = "callout";
  groundWrap.title = "ground offset — editable when the datum work lands";
  const groundVal = document.createElement("div");
  groundVal.className = "val noneditable";
  groundVal.textContent = GROUND_OFFSET_MM.toLocaleString();
  const groundLab = document.createElement("div");
  groundLab.className = "calloutlabel";
  groundLab.textContent = "ground offset";
  groundWrap.appendChild(groundVal);
  groundWrap.appendChild(groundLab);
  calloutLayer.appendChild(groundWrap);

  els = { svg, bg, layers, calloutLayer, callouts, groundWrap, resetBtn };
  built = true;
  builtContainer = container;
  window.addEventListener("resize", () => {
    if (!container.hidden) render(container);
  });
}

// -- projection: world (y, z) metres -> screen px, refit every render -----

function computeProjection(doc, geometry, w, h) {
  // Height-driven framing, like the mockup: the tower must fill the frame,
  // so the scale comes from the z span alone and the horizontal extent
  // simply clips -- a 90 m-radius field fitted edge-to-edge would squash a
  // 27 m tower into a sliver at the bottom (uniform scale either way: this
  // is a to-scale drawing, only the crop changes).
  const optics = doc.optics;
  const params = doc.opticsParams[optics] || {};
  let zTop = Math.max(5, tallestParamHeightM(params));
  const secondary = geometry && geometry.secondary;
  if (secondary && secondary.profile) {
    for (const pt of secondary.profile) zTop = Math.max(zTop, pt[1] / 1000);
  }
  const zBottom = -GROUND_OFFSET_MM / 1000;

  const marginPx = 90;
  const zSpanM = Math.max((zTop - zBottom) * 1.15, 1);
  const fitScale = Math.max((h - 2 * marginPx) / zSpanM, 0.01);
  const baseCx = w / 2; // tower axis (y = 0) centered; the far field clips
  const baseGroundPx = h - marginPx;

  // v0.2 fix wave item 4: manualView (see module comment) overrides the fit
  // with a live scale + pixel pan offset from this same fitted basis, so
  // zooming/panning survives a doc/geometry re-render instead of snapping
  // back to the fit every time.
  if (manualView) {
    return {
      scale: manualView.scale,
      cx: baseCx + manualView.panX,
      groundPx: baseGroundPx + manualView.panY,
      zBottom,
      visibleHalfYm: w / 2 / manualView.scale,
      fitScale,
      baseCx,
      baseGroundPx,
      w,
      h,
    };
  }
  return {
    scale: fitScale,
    cx: baseCx,
    groundPx: baseGroundPx,
    zBottom,
    visibleHalfYm: w / 2 / fitScale,
    fitScale,
    baseCx,
    baseGroundPx,
    w,
    h,
  };
}

function toScreen(proj, ym, zm) {
  return [proj.cx + ym * proj.scale, proj.groundPx - (zm - proj.zBottom) * proj.scale];
}

// The tallest of this optics' own height fields, in metres -- apex/vertex
// for axicon/cassegrain, but also focus/receiver height so prime focus
// (which has neither apex nor vertex) still gets a sane fallback.
function tallestParamHeightM(params) {
  let z = 0;
  for (const k of ["apex_height_mm", "vertex_z_mm", "focus_height_mm", "receiver_z_mm"]) {
    if (params[k] != null) z = Math.max(z, params[k] / 1000);
  }
  // prime_focus's own receiver can sit above focus_height_mm (the
  // aperture_to_receiver_mm offset) and, for a cylinder/frustum, extend
  // further still -- fold in its actual top so a tall or offset receiver
  // isn't cropped by the height-driven frame.
  if (params.receiver_type) {
    const baseZ = (params.focus_height_mm || 0) + (params.aperture_to_receiver_mm || 0);
    let halfHeight = 0;
    if (params.receiver_type === "cylinder") halfHeight = (params.cylinder_height_mm || 0) / 2;
    else if (params.receiver_type === "frustum") halfHeight = (params.frustum_height_mm || 0) / 2;
    z = Math.max(z, (baseZ + halfHeight) / 1000);
  }
  return z;
}

// The z a secondary's own surface reaches at its rim -- same quantity
// src/heliostat/web/scene.py's _secondary_top_height_mm draws the
// dropped-ray overshoot from. Falls back to the tallest optics param when
// there is no secondary at all (prime focus).
function secondaryRimZm(geometry, params) {
  const secondary = geometry && geometry.secondary;
  if (secondary && secondary.profile && secondary.profile.length) {
    return secondary.profile[secondary.profile.length - 1][1] / 1000;
  }
  return tallestParamHeightM(params);
}

// -- layers -----------------------------------------------------------------

function groundSvg(proj, w) {
  const groundY = proj.groundPx;
  let s = '<line x1="0" y1="' + groundY.toFixed(1) + '" x2="' + w + '" y2="' + groundY.toFixed(1) + '" stroke="#7b8794" stroke-width="2"></line>';
  for (let x = 20; x < w; x += 50) {
    s +=
      '<line x1="' + x + '" y1="' + groundY.toFixed(1) + '" x2="' + (x - 6) + '" y2="' + (groundY + 10).toFixed(1) +
      '" stroke="rgba(110,130,150,0.35)" stroke-width="1"></line>';
  }
  const datumY = toScreen(proj, 0, 0)[1];
  s +=
    '<line x1="0" y1="' + datumY.toFixed(1) + '" x2="' + w + '" y2="' + datumY.toFixed(1) +
    '" stroke="rgba(52,90,128,0.45)" stroke-width="1" stroke-dasharray="14 5 3 5"></line>';
  s +=
    '<text x="10" y="' + (datumY - 8).toFixed(1) + '" font-size="10.5" fill="#345a80">heliostat plane (datum for all heights)</text>';
  return s;
}

function mastSvg(doc, geometry, proj) {
  const params = doc.opticsParams[doc.optics] || {};
  const topZm = secondaryRimZm(geometry, params);
  const [x0, y0] = toScreen(proj, 0, proj.zBottom);
  const [x1, y1] = toScreen(proj, 0, topZm);
  return '<line x1="' + x0.toFixed(1) + '" y1="' + y0.toFixed(1) + '" x2="' + x1.toFixed(1) + '" y2="' + y1.toFixed(1) + '" stroke="#7b8794" stroke-width="5"></line>';
}

function secondarySvg(geometry, proj, ui) {
  const secondary = geometry && geometry.secondary;
  if (!secondary || !secondary.profile || secondary.profile.length < 2) return "";
  // Body of revolution about the tower axis: mirrored +-r about y=0 (spec:
  // "draw it mirrored, ±r about the axis").
  const right = secondary.profile.map((pt) => toScreen(proj, pt[0] / 1000, pt[1] / 1000));
  const left = secondary.profile
    .map((pt) => toScreen(proj, -pt[0] / 1000, pt[1] / 1000))
    .reverse();
  const pts = left.concat(right);
  const d = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  // Clickable like the plan view's aperture circle and the 3D cone
  // (docs/ui-spec.md 2.4 -- acceptance 3's "click the cone" must work in
  // whichever view is up), with the same select-blue highlight.
  const isSel = !!ui.selection && ui.selection.kind === "secondary";
  return (
    '<path data-kind="secondary" d="' + d + '" fill="rgba(126,158,196,0.30)" stroke="' +
    (isSel ? "#0b5fd0" : "rgba(52,90,128,0.75)") + '" stroke-width="' + (isSel ? 2.5 : 2) + '"></path>'
  );
}

// This view drops x throughout (see the module comment), so a receiver's
// horizontal extent (half_u_mm / radius_mm / r_top_mm / r_bot_mm -- all
// really x-axis quantities on an axisymmetric or on-axis shape) is drawn
// centered on its own center_y_mm, the same convention the pre-existing
// flat window used with an implicit center of 0.
function receiverSvg(geometry, proj, ui) {
  const receiver = geometry && geometry.receiver;
  if (!receiver) return "";
  const isSel = !!ui.selection && ui.selection.kind === "receiver";
  const stroke = isSel ? "#0b5fd0" : "#a8551a";
  const strokeWidth = isSel ? 2.2 : 1.5;
  const kind = receiver.kind || "flat";
  const yCenterM = (receiver.center_y_mm || 0) / 1000;

  let s;
  if (kind === "cylinder") {
    const halfM = receiver.radius_mm / 1000;
    const zTopM = (receiver.center_z_mm + receiver.height_mm / 2) / 1000;
    const zBotM = (receiver.center_z_mm - receiver.height_mm / 2) / 1000;
    const [x0, y0] = toScreen(proj, yCenterM - halfM, zTopM);
    const [x1, y1] = toScreen(proj, yCenterM + halfM, zBotM);
    s =
      '<rect data-kind="receiver" x="' + Math.min(x0, x1).toFixed(1) + '" y="' + Math.min(y0, y1).toFixed(1) +
      '" width="' + Math.abs(x1 - x0).toFixed(1) + '" height="' + Math.abs(y1 - y0).toFixed(1) +
      '" fill="rgba(217,123,41,0.35)" stroke="' + stroke + '" stroke-width="' + strokeWidth + '"></rect>';
  } else if (kind === "frustum") {
    const rTopM = receiver.r_top_mm / 1000;
    const rBotM = receiver.r_bot_mm / 1000;
    const [xTL, yT] = toScreen(proj, yCenterM - rTopM, receiver.z_top_mm / 1000);
    const [xTR] = toScreen(proj, yCenterM + rTopM, receiver.z_top_mm / 1000);
    const [xBR, yB] = toScreen(proj, yCenterM + rBotM, receiver.z_bot_mm / 1000);
    const [xBL] = toScreen(proj, yCenterM - rBotM, receiver.z_bot_mm / 1000);
    const d =
      "M" + xTL.toFixed(1) + " " + yT.toFixed(1) +
      " L" + xTR.toFixed(1) + " " + yT.toFixed(1) +
      " L" + xBR.toFixed(1) + " " + yB.toFixed(1) +
      " L" + xBL.toFixed(1) + " " + yB.toFixed(1) + " Z";
    s = '<path data-kind="receiver" d="' + d + '" fill="rgba(217,123,41,0.35)" stroke="' + stroke + '" stroke-width="' + strokeWidth + '"></path>';
  } else {
    const halfM = receiver.half_u_mm / 1000;
    const zm = receiver.z_mm / 1000;
    const [x0, y0] = toScreen(proj, yCenterM - halfM, zm);
    const [x1] = toScreen(proj, yCenterM + halfM, zm);
    const thickness = 10;
    s =
      '<rect data-kind="receiver" x="' + Math.min(x0, x1).toFixed(1) + '" y="' + (y0 - thickness / 2).toFixed(1) +
      '" width="' + Math.abs(x1 - x0).toFixed(1) + '" height="' + thickness + '" fill="rgba(217,123,41,0.7)" stroke="' +
      stroke + '" stroke-width="' + strokeWidth + '"></rect>';
  }

  // The entrance aperture (always flat, at focus_height_mm) only exists on
  // the wire once aperture_to_receiver_mm > 0 -- draw it as a thin dashed
  // line so the offset between it and the receiver above/below reads at a
  // glance.
  if (receiver.aperture) {
    const ap = receiver.aperture;
    const apHalfM = ap.half_u_mm / 1000;
    const apYCenterM = (ap.center_y_mm || 0) / 1000;
    const apZm = ap.z_mm / 1000;
    const [ax0, ay0] = toScreen(proj, apYCenterM - apHalfM, apZm);
    const [ax1] = toScreen(proj, apYCenterM + apHalfM, apZm);
    s +=
      '<line data-kind="receiver" x1="' + ax0.toFixed(1) + '" y1="' + ay0.toFixed(1) + '" x2="' + ax1.toFixed(1) +
      '" y2="' + ay0.toFixed(1) + '" stroke="#a8551a" stroke-width="1.5" stroke-dasharray="5 3"></line>';
  }

  return s;
}

function raysSvg(geometry, ui, proj) {
  let s = "";
  const rays = (geometry && geometry.rays) || [];
  for (const poly of rays) {
    const pts = poly.map((p) => toScreen(proj, p[1] / 1000, p[2] / 1000)); // drop x
    const ptStr = pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    s += '<polyline points="' + ptStr + '" fill="none" stroke="rgba(255,146,54,0.35)" stroke-width="1.2"></polyline>';
  }
  const missRays = (ui.miss && ui.miss.rays) || [];
  for (const poly of missRays) {
    const pts = poly.map((p) => toScreen(proj, p[1] / 1000, p[2] / 1000));
    const ptStr = pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    s += '<polyline points="' + ptStr + '" fill="none" stroke="#e0554a" stroke-width="1.2" stroke-dasharray="6 4"></polyline>';
  }
  return s;
}

// A handful of representative heliostats spanning the field's near/far
// radii on both sides (spec: "4 representative ones at the field's
// near/far radii on both sides"), by radius extremes + two intermediate
// points -- deterministic, no rng, same idea as scene.py's own strided
// corner-ray sources.
function representativeHeliostats(heliostats, visibleHalfYm) {
  // Only heliostats inside the height-driven crop (computeProjection) are
  // candidates -- a stroke placed off-frame is markup for nothing, and the
  // far field is deliberately cropped away in this view.
  const visible =
    visibleHalfYm == null
      ? heliostats
      : heliostats.filter((h) => Math.abs(h.y_mm / 1000) <= visibleHalfYm);
  if (visible.length <= 4) return visible;
  const sorted = visible.slice().sort((a, b) => Math.hypot(a.x_mm, a.y_mm) - Math.hypot(b.x_mm, b.y_mm));
  const idxs = [0, Math.floor(sorted.length * 0.33), Math.floor(sorted.length * 0.66), sorted.length - 1];
  const seen = new Set();
  const picked = [];
  for (const i of idxs) {
    if (seen.has(i)) continue;
    seen.add(i);
    picked.push(sorted[i]);
  }
  return picked;
}

function heliostatStrokesSvg(heliostats, proj) {
  let s = "";
  for (const h of representativeHeliostats(heliostats, proj.visibleHalfYm)) {
    const [sy, sz] = toScreen(proj, h.y_mm / 1000, 0); // pivot is z=0 (world frame)
    const len = 10;
    // Simplified 2D tilt from the mirror's elevation angle only -- the
    // azimuth component of its pointing is invisible in a y-z projection,
    // same simplification the profile-only side view makes everywhere else.
    let rot = 0;
    if (h.rot_el_deg != null) rot = -(90 - h.rot_el_deg);
    s +=
      '<line x1="' + (sy - len).toFixed(1) + '" y1="' + sz.toFixed(1) + '" x2="' + (sy + len).toFixed(1) + '" y2="' +
      sz.toFixed(1) + '" stroke="#33455c" stroke-width="2.2" transform="rotate(' + rot.toFixed(1) + " " + sy.toFixed(1) +
      " " + sz.toFixed(1) + ')"></line>';
  }
  return s;
}

function sunDiscSvg(w) {
  const cx = w - 60;
  const cy = 50;
  return (
    '<circle cx="' + cx + '" cy="' + cy + '" r="15" fill="#f0b429"></circle>' +
    '<g stroke="#f0b429" stroke-width="2.2">' +
    '<line x1="' + cx + '" y1="' + (cy - 26) + '" x2="' + cx + '" y2="' + (cy - 19) + '"></line>' +
    '<line x1="' + cx + '" y1="' + (cy + 19) + '" x2="' + cx + '" y2="' + (cy + 26) + '"></line>' +
    '<line x1="' + (cx - 26) + '" y1="' + cy + '" x2="' + (cx - 19) + '" y2="' + cy + '"></line>' +
    '<line x1="' + (cx + 19) + '" y1="' + cy + '" x2="' + (cx + 26) + '" y2="' + cy + '"></line>' +
    "</g>"
  );
}

// -- dimension callouts ------------------------------------------------------

// Anchor (the world point the value describes) + where its editable box
// sits on screen, for one callout key. Height-type dimensions run vertical
// leader lines back to the datum; aperture radius runs horizontal back to
// the tower axis; half angle gets a small arc near the apex.
function calloutAnchorAndBox(key, optics, doc, geometry, proj) {
  const params = doc.opticsParams[optics];
  const value = params[key];
  if (value == null) return null;

  if (key === "aperture_radius_mm") {
    const rimZm = secondaryRimZm(geometry, params);
    const anchor = toScreen(proj, value / 1000, rimZm);
    const axis = toScreen(proj, 0, rimZm);
    return { kind: "horizontal", anchor, axis, box: { x: (axis[0] + anchor[0]) / 2, y: anchor[1] - 46 } };
  }
  if (key === "half_angle_deg") {
    const apexZm = (params.apex_height_mm != null ? params.apex_height_mm : 0) / 1000;
    const apex = toScreen(proj, 0, apexZm);
    return { kind: "angle", anchor: apex, box: { x: apex[0] + 70, y: apex[1] + 22 } };
  }
  if (key === "aperture_to_receiver_mm") {
    // Vertical dimension from the aperture's own z (focus_height_mm) up to
    // the receiver's resolved z -- the offset IS the receiver's height
    // control (see heliostat.web.app's PrimeFocusOptics), there is no
    // separate receiver-height field to dimension instead.
    const apertureZm = (params.focus_height_mm || 0) / 1000;
    const receiverZm = apertureZm + value / 1000;
    const anchor = toScreen(proj, 0, receiverZm);
    const datum = toScreen(proj, 0, apertureZm);
    return { kind: "vertical", anchor, datum, box: { x: anchor[0] + 150, y: anchor[1] } };
  }
  if (key === "cylinder_height_mm" || key === "frustum_height_mm") {
    const centerZm = ((params.focus_height_mm || 0) + (params.aperture_to_receiver_mm || 0)) / 1000;
    const halfM = value / 2000;
    const anchor = toScreen(proj, 0, centerZm + halfM);
    const datum = toScreen(proj, 0, centerZm - halfM);
    return { kind: "vertical", anchor, datum, box: { x: anchor[0] + 190, y: (anchor[1] + datum[1]) / 2 } };
  }
  if (key === "cylinder_radius_mm") {
    const centerZm = ((params.focus_height_mm || 0) + (params.aperture_to_receiver_mm || 0)) / 1000;
    const anchor = toScreen(proj, value / 1000, centerZm);
    const axis = toScreen(proj, 0, centerZm);
    return { kind: "horizontal", anchor, axis, box: { x: (axis[0] + anchor[0]) / 2, y: anchor[1] - 46 } };
  }
  if (key === "frustum_top_radius_mm" || key === "frustum_bottom_radius_mm") {
    const centerZm = ((params.focus_height_mm || 0) + (params.aperture_to_receiver_mm || 0)) / 1000;
    const halfHeightM = (params.frustum_height_mm || 0) / 2000;
    const isTop = key === "frustum_top_radius_mm";
    const zm = centerZm + (isTop ? halfHeightM : -halfHeightM);
    const anchor = toScreen(proj, value / 1000, zm);
    const axis = toScreen(proj, 0, zm);
    const dy = isTop ? -46 : 46; // opposite sides so the two radii don't overlap
    return { kind: "horizontal", anchor, axis, box: { x: (axis[0] + anchor[0]) / 2, y: anchor[1] + dy } };
  }

  // apex_height_mm / vertex_z_mm / focus_height_mm / receiver_z_mm: a
  // vertical dimension from the heliostat-plane datum up to this height,
  // offset left (tower-side fields) or right (receiver) so callouts don't
  // stack on top of one another.
  const zm = value / 1000;
  const anchor = toScreen(proj, 0, zm);
  const datum = toScreen(proj, 0, 0);
  let dx = key === "receiver_z_mm" ? 110 : -110;
  if (key === "focus_height_mm" && optics === "cassegrain") dx = -190; // clear of the vertex callout
  return { kind: "vertical", anchor, datum, box: { x: anchor[0] + dx, y: anchor[1] } };
}

function dimSvg(info) {
  if (info.kind === "vertical") {
    const [ax, ay] = info.anchor;
    const [dx, dy] = info.datum;
    return (
      '<g stroke="#64748b" stroke-width="1" fill="none">' +
      '<line x1="' + ax.toFixed(1) + '" y1="' + ay.toFixed(1) + '" x2="' + dx.toFixed(1) + '" y2="' + dy.toFixed(1) + '" stroke-dasharray="2 3"></line>' +
      '<circle cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) + '" r="2.5" fill="#64748b" stroke="none"></circle>' +
      '<line x1="' + ax.toFixed(1) + '" y1="' + ay.toFixed(1) + '" x2="' + info.box.x.toFixed(1) + '" y2="' + info.box.y.toFixed(1) + '"></line>' +
      "</g>"
    );
  }
  if (info.kind === "horizontal") {
    const [ax, ay] = info.anchor;
    const [xx, xy] = info.axis;
    const midX = (xx + ax) / 2;
    const midY = (xy + ay) / 2;
    return (
      '<g stroke="#64748b" stroke-width="1" fill="none">' +
      '<line x1="' + xx.toFixed(1) + '" y1="' + xy.toFixed(1) + '" x2="' + ax.toFixed(1) + '" y2="' + ay.toFixed(1) + '"></line>' +
      '<circle cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) + '" r="2.5" fill="#64748b" stroke="none"></circle>' +
      '<line x1="' + midX.toFixed(1) + '" y1="' + midY.toFixed(1) + '" x2="' + info.box.x.toFixed(1) + '" y2="' + info.box.y.toFixed(1) + '"></line>' +
      "</g>"
    );
  }
  // angle
  const [ax, ay] = info.anchor;
  return (
    '<g stroke="#64748b" stroke-width="1" fill="none">' +
    '<path d="M' + ax.toFixed(1) + " " + (ay - 30).toFixed(1) + " A30 30 0 0 1 " + (ax + 27).toFixed(1) + " " + (ay - 12).toFixed(1) + '"></path>' +
    '<line x1="' + ax.toFixed(1) + '" y1="' + ay.toFixed(1) + '" x2="' + info.box.x.toFixed(1) + '" y2="' + info.box.y.toFixed(1) + '"></line>' +
    "</g>"
  );
}

// Callout boxes are anchored to the geometry they label, so several can want
// the same spot -- most visibly the receiver's own dimensions once it has a
// radius and a height as well as a position. Boxes are nudged down the screen
// until they stop overlapping, in the order they were placed.
const CALLOUT_BOX_H = 40;
const CALLOUT_BOX_W = 120;

function avoidOverlap(box, placed) {
  let moved = true;
  let guard = 0;
  while (moved && guard < 20) {
    moved = false;
    guard += 1;
    for (const other of placed) {
      const dx = Math.abs(box.x - other.x);
      const dy = Math.abs(box.y - other.y);
      if (dx < CALLOUT_BOX_W && dy < CALLOUT_BOX_H) {
        box.y = other.y + CALLOUT_BOX_H;
        moved = true;
      }
    }
  }
  placed.push({ x: box.x, y: box.y });
  return box;
}

function renderCallouts(doc, geometry, proj) {
  const optics = doc.optics;
  const params = doc.opticsParams[optics];
  const placed = [];
  let dims = "";
  for (const [id, c] of Object.entries(els.callouts)) {
    const dot = id.indexOf(".");
    const cOptics = id.slice(0, dot);
    const key = id.slice(dot + 1);
    let show = cOptics === optics;
    // prime_focus's cylinder/frustum-only callouts (same `group` tag the
    // sidebar/inspector rows use) only show while receiver_type matches.
    if (show && cOptics === "prime_focus") {
      const field = RECEIVER_FIELD_TABLE.prime_focus.find((f) => f.key === key);
      show = !field || receiverFieldVisible(field, params);
    }
    c.wrap.style.display = show ? "" : "none";
    if (!show) continue;
    const info = calloutAnchorAndBox(key, optics, doc, geometry, proj);
    if (!info) continue;
    avoidOverlap(info.box, placed);
    c.wrap.style.left = info.box.x.toFixed(1) + "px";
    c.wrap.style.top = info.box.y.toFixed(1) + "px";
    setVal(c.input, params[key]); // isFocused guard: typing survives re-renders
    dims += dimSvg(info);
  }

  // Ground offset: always shown, same visual language, not editable (SCOPE).
  const groundAnchor = toScreen(proj, 0, proj.zBottom);
  const groundBox = { x: groundAnchor[0] + 170, y: groundAnchor[1] };
  els.groundWrap.style.left = groundBox.x.toFixed(1) + "px";
  els.groundWrap.style.top = groundBox.y.toFixed(1) + "px";
  dims += dimSvg({ kind: "vertical", anchor: groundAnchor, datum: toScreen(proj, 0, 0), box: groundBox });

  els.layers.dims.innerHTML = dims;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const geometry = lastGeometry;

  const w = Math.max(container.clientWidth || 1, 100);
  const h = Math.max(container.clientHeight || 1, 100);
  els.svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  els.bg.setAttribute("width", w);
  els.bg.setAttribute("height", h);

  const proj = computeProjection(doc, geometry, w, h);
  lastProj = proj;
  els.resetBtn.hidden = !manualView;

  els.layers.ground.innerHTML = groundSvg(proj, w);
  els.layers.rays.innerHTML = raysSvg(geometry, ui, proj);
  els.layers.mast.innerHTML = mastSvg(doc, geometry, proj);
  els.layers.secondary.innerHTML = secondarySvg(geometry, proj, ui);
  els.layers.receiver.innerHTML = receiverSvg(geometry, proj, ui);
  els.layers.heliostats.innerHTML = heliostatStrokesSvg((geometry && geometry.heliostats) || [], proj);
  els.layers.sun.innerHTML = sunDiscSvg(w);

  renderCallouts(doc, geometry, proj);
}
