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
import { resolvePath, setVal, RECEIVER_FIELD_TABLE } from "../fields.js";

// The default heliostat-plane-above-ground offset (docs/ui-spec.md 2.2):
// "a separate ground offset parameter ... default 2 500 mm". Editing it is
// out of scope for this phase (SCOPE above) -- see the ground-offset
// callout's own non-editable box and title tooltip.
const GROUND_OFFSET_MM = 2500;

// Which RECEIVER_FIELD_TABLE keys get a dimension callout in this view, per
// optics (docs/ui-spec.md 2.2's per-layout table, minus the window ½w/½h
// fields -- those aren't dimensioned in the elevation, only in the sidebar).
const CALLOUT_KEYS = {
  prime_focus: ["focus_height_mm"],
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
};

let built = false;
let els = {};
let lastGeometry = null;

// Called by main.js on every successful /api/scene/geometry response --
// never on error, so (like the 3D scene) this view keeps drawing the last
// valid geometry while ui.geometryError is set (docs/ui-spec.md 2.3).
export function setGeometry(data) {
  lastGeometry = data;
}

// Callouts live in a sibling HTML layer, not inside the SVG (see build()),
// so a click on one never reaches this handler at all -- anything that does
// arrive here is either a data-kind shape or empty ground.
function handleClick(e) {
  const el = e.target.closest && e.target.closest("[data-kind]");
  if (!el) {
    store.set("ui.selection", null); // empty ground deselects
    return;
  }
  const kind = el.dataset.kind;
  store.set("ui.selection", { kind, id: null });
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

  els = { svg, bg, layers, calloutLayer, callouts, groundWrap };
  built = true;
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
  const scale = Math.max((h - 2 * marginPx) / zSpanM, 0.01);
  const cx = w / 2; // tower axis (y = 0) centered; the far field clips
  const groundPx = h - marginPx;
  return { scale, cx, groundPx, zBottom, visibleHalfYm: w / 2 / scale };
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

function receiverSvg(geometry, proj, ui) {
  const receiver = geometry && geometry.receiver;
  if (!receiver) return "";
  const halfM = receiver.half_u_mm / 1000;
  const zm = receiver.z_mm / 1000;
  const [x0, y0] = toScreen(proj, -halfM, zm);
  const [x1] = toScreen(proj, halfM, zm);
  const thickness = 10;
  const isSel = !!ui.selection && ui.selection.kind === "receiver";
  return (
    '<rect data-kind="receiver" x="' + Math.min(x0, x1).toFixed(1) + '" y="' + (y0 - thickness / 2).toFixed(1) +
    '" width="' + Math.abs(x1 - x0).toFixed(1) + '" height="' + thickness + '" fill="rgba(217,123,41,0.7)" stroke="' +
    (isSel ? "#0b5fd0" : "#a8551a") + '" stroke-width="' + (isSel ? 2.2 : 1.5) + '"></rect>'
  );
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

function renderCallouts(doc, geometry, proj) {
  const optics = doc.optics;
  const params = doc.opticsParams[optics];
  let dims = "";
  for (const [id, c] of Object.entries(els.callouts)) {
    const dot = id.indexOf(".");
    const cOptics = id.slice(0, dot);
    const key = id.slice(dot + 1);
    const show = cOptics === optics;
    c.wrap.style.display = show ? "" : "none";
    if (!show) continue;
    const info = calloutAnchorAndBox(key, optics, doc, geometry, proj);
    if (!info) continue;
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

  els.layers.ground.innerHTML = groundSvg(proj, w);
  els.layers.rays.innerHTML = raysSvg(geometry, ui, proj);
  els.layers.mast.innerHTML = mastSvg(doc, geometry, proj);
  els.layers.secondary.innerHTML = secondarySvg(geometry, proj, ui);
  els.layers.receiver.innerHTML = receiverSvg(geometry, proj, ui);
  els.layers.heliostats.innerHTML = heliostatStrokesSvg((geometry && geometry.heliostats) || [], proj);
  els.layers.sun.innerHTML = sunDiscSvg(w);

  renderCallouts(doc, geometry, proj);
}
