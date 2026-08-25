// Field plan view (docs/ui-spec.md 2.2, mockup M3): a top-down SVG drawing
// of the current geometry, active while the Field stage is expanded
// (docs/ui-spec.md 2.1's "the viewport follows the active stage" -- wired
// in ../main.js). Store-free except for reading selection/miss/doc state
// (the same pattern the sidebar panels use); the geometry itself is NOT
// store state, so main.js pushes the latest /api/scene/geometry response
// in through setGeometry() -- same reason inspector.js takes it as a ctx
// argument rather than reading it off the store.
//
// SCOPE (phase 3b): a live VIEW with click-select only. Drag-to-move,
// double-click-to-add, roads, and the layout picker land in a later batch
// -- the bottom-left chip says so explicitly (main.js owns that text).
//
// World frame matches the server exactly (src/heliostat/web/scene.py): x
// east, y north, mm on the wire. Drawn with north UP, so screen y is the
// negative of world y; everything here works in metres (mm / 1000).
import { store } from "../store.js";

const SELECT_COLOR = "#0b5fd0";
const MISS_COLOR = "#e0554a";
const HELIOSTAT_COLOR = "#c4daee";
const HELIOSTAT_STROKE = "#33455c";

let built = false;
let els = {};
let lastGeometry = null;

// Called by main.js on every successful /api/scene/geometry response --
// never on error, so (like the 3D scene) this view keeps drawing the last
// valid geometry while ui.geometryError is set (docs/ui-spec.md 2.3).
export function setGeometry(data) {
  lastGeometry = data;
}

function handleClick(e) {
  // Phase 3c wave 1: the selection card's "View shape" button
  // (docs/ui-spec.md 3, "clicking any heliostat in the workspace offers
  // 'View shape'") -- checked before [data-kind] since the button itself
  // isn't a heliostat/secondary/receiver shape.
  const actionEl = e.target.closest && e.target.closest("[data-action]");
  if (actionEl && actionEl.dataset.action === "view-shape") {
    const id = Number(actionEl.dataset.id);
    if (Number.isFinite(id)) {
      store.set("ui.shapeHeliostatId", id);
      store.set("ui.tab", "shape");
    }
    return;
  }
  const el = e.target.closest && e.target.closest("[data-kind]");
  if (!el) {
    store.set("ui.selection", null); // empty ground deselects
    return;
  }
  const kind = el.dataset.kind;
  if (kind === "heliostat") {
    const id = Number(el.dataset.id);
    if (Number.isFinite(id)) store.set("ui.selection", { kind: "heliostat", id });
  } else {
    // aperture -> "secondary", receiver square -> "receiver": the same two
    // kinds a 3D click on those objects reports (docs/ui-spec.md 2.4).
    store.set("ui.selection", { kind, id: null });
  }
}

function build(container) {
  container.innerHTML =
    '<svg preserveAspectRatio="xMidYMid meet">' +
    '<rect x="0" y="0" fill="#fdfdfe"></rect>' +
    '<g data-layer="rings"></g>' +
    '<g data-layer="aperture"></g>' +
    '<g data-layer="heliostats"></g>' +
    '<g data-layer="selection"></g>' +
    '<g data-layer="chrome"></g>' +
    "</svg>";
  const svg = container.querySelector("svg");
  const bg = container.querySelector("rect");
  const layers = {};
  for (const name of ["rings", "aperture", "heliostats", "selection", "chrome"]) {
    layers[name] = container.querySelector('[data-layer="' + name + '"]');
  }
  svg.addEventListener("click", handleClick);
  els = { svg, bg, layers };
  built = true;
  // Container is only ever the size the flex layout gives it; a pure
  // window resize while this view is showing gets no other signal to
  // re-render on (store-driven renders already cover every edit).
  window.addEventListener("resize", () => {
    if (!container.hidden) render(container);
  });
}

// -- projection: world metres -> screen px, refit every render ------------

function computeProjection(heliostats, w, h) {
  let maxRm = 0;
  for (const hh of heliostats) {
    const rm = Math.hypot(hh.x_mm, hh.y_mm) / 1000;
    if (rm > maxRm) maxRm = rm;
  }
  if (maxRm < 5) maxRm = 5; // a sane minimum span for an empty/near field
  const halfSpan = maxRm * 1.08; // spec: "fit the whole field with ~8% margin"
  const scale = Math.max((Math.min(w, h) / 2 - 40) / halfSpan, 0.01); // px/m
  return { scale, cx: w / 2, cy: h / 2 };
}

function toScreen(proj, xm, ym) {
  return [proj.cx + xm * proj.scale, proj.cy - ym * proj.scale]; // north up
}

// -- nice round metres for the scale bar -----------------------------------

function niceRoundMetres(targetPx, scale) {
  const targetM = Math.max(targetPx / scale, 0.1);
  const magnitude = Math.pow(10, Math.floor(Math.log10(targetM)));
  let best = magnitude;
  for (const k of [1, 2, 5, 10]) {
    const c = k * magnitude;
    if (Math.abs(c - targetM) < Math.abs(best - targetM)) best = c;
  }
  return best;
}

// -- layers -----------------------------------------------------------------

function ringsSvg(doc, proj) {
  // Rings at the layout radii only make sense for a parametric field
  // (docs/ui-spec.md "Faint rings at the layout radii ... only in field
  // mode"); single-heliostat mode draws just that one heliostat + tower.
  // The manuscript layout is real positions, not a radius-bounded spiral --
  // its r_min/r_max don't describe anything the field actually respects, so
  // there is nothing honest to draw a ring at; the positions speak for
  // themselves.
  if (doc.field.mode !== "field" || doc.field.layout === "manuscript") return "";
  const f = doc.field.fermat;
  let s = "";
  for (const rm of [f.r_min_m, f.r_max_m]) {
    if (!(rm > 0)) continue;
    const r = rm * proj.scale;
    s +=
      '<circle cx="' + proj.cx.toFixed(1) + '" cy="' + proj.cy.toFixed(1) + '" r="' + r.toFixed(1) +
      '" fill="none" stroke="rgba(110,130,150,0.35)" stroke-width="1" stroke-dasharray="3 5"></circle>';
    s +=
      '<text x="' + (proj.cx + r + 4).toFixed(1) + '" y="' + (proj.cy - 4).toFixed(1) +
      '" font-size="10.5" fill="#64748b">' + rm + " m</text>";
  }
  return s;
}

function apertureSvg(doc, geometry, ui, proj) {
  const optics = doc.optics;
  const params = doc.opticsParams[optics];
  const sel = ui.selection;
  let s = "";
  // prime_focus has no aperture_radius_mm field (no secondary at all).
  if (optics !== "prime_focus" && params && params.aperture_radius_mm) {
    const rm = params.aperture_radius_mm / 1000;
    const r = rm * proj.scale;
    const label = optics === "axicon" ? "axicon aperture r = " + rm.toFixed(1) + " m" : "aperture r = " + rm.toFixed(1) + " m";
    const isSel = !!sel && sel.kind === "secondary";
    s +=
      '<circle data-kind="secondary" cx="' + proj.cx.toFixed(1) + '" cy="' + proj.cy.toFixed(1) + '" r="' + r.toFixed(1) +
      '" fill="none" stroke="' + (isSel ? SELECT_COLOR : "rgba(52,90,128,0.55)") + '" stroke-width="' + (isSel ? 2.2 : 1.5) +
      '" stroke-dasharray="6 4"></circle>';
    s +=
      '<text x="' + proj.cx.toFixed(1) + '" y="' + (proj.cy - r - 8).toFixed(1) +
      '" font-size="11" fill="#345a80" text-anchor="middle">' + label + "</text>";
  }
  const receiver = geometry && geometry.receiver;
  if (receiver) {
    const size = 20;
    const isSel = !!sel && sel.kind === "receiver";
    s +=
      '<rect data-kind="receiver" x="' + (proj.cx - size / 2).toFixed(1) + '" y="' + (proj.cy - size / 2).toFixed(1) +
      '" width="' + size + '" height="' + size + '" fill="rgba(217,123,41,0.7)" stroke="' +
      (isSel ? SELECT_COLOR : "#a8551a") + '" stroke-width="' + (isSel ? 2.2 : 1.5) + '"></rect>';
  }
  return s;
}

function heliostatsSvg(heliostats, ui, proj) {
  const miss = ui.miss;
  const missIds = new Set([...((miss && miss.aperture_miss_ids) || []), ...((miss && miss.total_miss_ids) || [])]);
  const sel = ui.selection;
  let s = "";
  for (const h of heliostats) {
    const [sx, sy] = toScreen(proj, h.x_mm / 1000, h.y_mm / 1000);
    const isSel = !!sel && sel.kind === "heliostat" && sel.id === h.id;
    const isMiss = missIds.has(h.id);
    const fill = isSel ? SELECT_COLOR : isMiss ? MISS_COLOR : HELIOSTAT_COLOR;
    if (h.rot_az_deg == null) {
      // Sun below horizon (docs/ui-spec.md 2.1): no orientation solved --
      // draw an unoriented square rather than a rotated rect.
      const sz = 7;
      s +=
        '<rect data-kind="heliostat" data-id="' + h.id + '" x="' + (sx - sz / 2).toFixed(1) + '" y="' +
        (sy - sz / 2).toFixed(1) + '" width="' + sz + '" height="' + sz + '" fill="' + fill + '" stroke="' +
        HELIOSTAT_STROKE + '" stroke-width="0.8"></rect>';
    } else {
      // Screen rotate = -az_deg: rot_az_deg is a math-standard angle (0=+x
      // east, 90=+y north, per scene3d.js's mirrorFrame docstring), and
      // north-up screen space flips y, which negates the visual angle.
      const rot = -h.rot_az_deg;
      s +=
        '<rect data-kind="heliostat" data-id="' + h.id + '" x="' + (sx - 5).toFixed(1) + '" y="' + (sy - 3).toFixed(1) +
        '" width="10" height="6" fill="' + fill + '" stroke="' + HELIOSTAT_STROKE + '" stroke-width="0.8" transform="rotate(' +
        rot.toFixed(1) + " " + sx.toFixed(1) + " " + sy.toFixed(1) + ')"></rect>';
    }
  }
  return s;
}

function selectionSvg(heliostats, ui, proj, w, h) {
  const sel = ui.selection;
  if (!sel || sel.kind !== "heliostat") return "";
  const helio = heliostats.find((x) => x.id === sel.id);
  if (!helio) return "";
  const [sx, sy] = toScreen(proj, helio.x_mm / 1000, helio.y_mm / 1000);
  const xm = helio.x_mm / 1000;
  const ym = helio.y_mm / 1000;
  const rm = Math.hypot(helio.x_mm, helio.y_mm) / 1000;
  const cardW = 150;
  const cardH = 56;
  let cardX = sx + 20;
  let cardY = sy - 30;
  cardX = Math.min(Math.max(cardX, 4), w - cardW - 4);
  cardY = Math.min(Math.max(cardY, 4), h - cardH - 4);
  return (
    '<circle cx="' + sx.toFixed(1) + '" cy="' + sy.toFixed(1) + '" r="16" fill="none" stroke="rgba(11,95,208,0.5)" stroke-width="1.2" stroke-dasharray="3 3"></circle>' +
    '<g transform="translate(' + cardX.toFixed(1) + ", " + cardY.toFixed(1) + ')">' +
    '<rect width="' + cardW + '" height="' + cardH + '" rx="6" fill="#ffffff" stroke="#d8dee5"></rect>' +
    '<text x="10" y="15" font-size="11" fill="#1f2933" font-weight="600">H-' + helio.id + "</text>" +
    '<text x="10" y="28" font-size="10.5" fill="#64748b">x ' + xm.toFixed(1) + " m · y " + ym.toFixed(1) + " m · r " + rm.toFixed(1) + " m</text>" +
    '<g data-action="view-shape" data-id="' + helio.id + '" style="cursor:pointer">' +
    '<rect x="6" y="36" width="' + (cardW - 12) + '" height="16" rx="4" fill="#f2f7fd" stroke="#d8dee5"></rect>' +
    '<text x="' + (cardW / 2).toFixed(1) + '" y="47" font-size="10.5" fill="#345a80" text-anchor="middle" font-weight="600">View shape →</text>' +
    "</g>" +
    "</g>"
  );
}

function chromeSvg(doc, proj, w, h) {
  let s = "";
  // North arrow, top-left.
  s +=
    '<g transform="translate(50, 46)">' +
    '<circle r="16" fill="none" stroke="#64748b" stroke-width="1.2"></circle>' +
    '<path d="M0 -12 L4 4 L0 1 L-4 4 Z" fill="#64748b"></path>' +
    '<text y="32" font-size="11" fill="#64748b" text-anchor="middle">N</text>' +
    "</g>";
  // Scale bar, bottom-right -- a round number of metres for the current scale.
  const barM = niceRoundMetres(90, proj.scale);
  const barPx = barM * proj.scale;
  const bx = w - 60 - barPx;
  const by = h - 40;
  s +=
    '<g transform="translate(' + bx.toFixed(1) + ", " + by.toFixed(1) + ')">' +
    '<line x1="0" y1="0" x2="' + barPx.toFixed(1) + '" y2="0" stroke="#64748b" stroke-width="1.5"></line>' +
    '<line x1="0" y1="-4" x2="0" y2="4" stroke="#64748b" stroke-width="1.5"></line>' +
    '<line x1="' + barPx.toFixed(1) + '" y1="-4" x2="' + barPx.toFixed(1) + '" y2="4" stroke="#64748b" stroke-width="1.5"></line>' +
    '<text x="' + (barPx / 2).toFixed(1) + '" y="16" font-size="11" fill="#64748b" text-anchor="middle">' + barM + " m</text>" +
    "</g>";
  // Sun azimuth indicator, bottom-left. doc.sun.az is a compass bearing
  // (0=N, 90=E -- see scene3d.js's mirrorFrame comment on the *absence* of
  // this conversion for rot_az_deg, which is the opposite convention);
  // screen dx = sin(az), dy = -cos(az) turns that into the north-up frame.
  const azRad = (doc.sun.az * Math.PI) / 180;
  const dx = Math.sin(azRad);
  const dy = -Math.cos(azRad);
  const ox = 70;
  const oy = h - 60;
  s +=
    "<g>" +
    '<circle cx="' + ox + '" cy="' + oy + '" r="9" fill="#f0b429"></circle>' +
    '<line x1="' + (ox + dx * 13).toFixed(1) + '" y1="' + (oy + dy * 13).toFixed(1) + '" x2="' + (ox + dx * 34).toFixed(1) +
    '" y2="' + (oy + dy * 34).toFixed(1) + '" stroke="#f0b429" stroke-width="2.5"></line>' +
    '<text x="' + ox + '" y="' + (oy + 24) + '" font-size="10.5" fill="#64748b" text-anchor="middle">sun az ' +
    doc.sun.az.toFixed(0) + "°</text>" +
    "</g>";
  return s;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");
  const geometry = lastGeometry;
  const heliostats = (geometry && geometry.heliostats) || [];

  const w = Math.max(container.clientWidth || 1, 100);
  const h = Math.max(container.clientHeight || 1, 100);
  els.svg.setAttribute("viewBox", "0 0 " + w + " " + h);
  els.bg.setAttribute("width", w);
  els.bg.setAttribute("height", h);

  const proj = computeProjection(heliostats, w, h);

  els.layers.rings.innerHTML = ringsSvg(doc, proj);
  els.layers.aperture.innerHTML = apertureSvg(doc, geometry, ui, proj);
  els.layers.heliostats.innerHTML = heliostatsSvg(heliostats, ui, proj);
  els.layers.selection.innerHTML = selectionSvg(heliostats, ui, proj, w, h);
  els.layers.chrome.innerHTML = chromeSvg(doc, proj, w, h);
}
