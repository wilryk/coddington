// Shared field descriptors + row-building helpers for the four sidebar
// stages (Heliostat, Field, Receiver & Tower, Sun).
//
// Extracted out of panels/heliostat.js, panels/field.js, panels/receiver.js
// and panels/sun.js (which used to each carry their own numberRow/
// isFocused/setVal trio) so the floating in-scene inspector
// (js/inspector.js) can render "exactly the same fields as that object's
// sidebar stage, bound to the same values" (docs/ui-spec.md 2.4) without a
// second copy of the field lists. Both the sidebar panels and the
// inspector build their rows from the descriptors below, and every row's
// `input` event is the only place a typed value turns into a store.set --
// so there really is one edit pathway, not two kept in sync by hand.
import { store } from "./store.js";

export function isFocused(el) {
  return el && document.activeElement === el;
}

export function setVal(input, value) {
  if (isFocused(input)) return;
  const s = value == null ? "" : String(value);
  if (input.value !== s) input.value = s;
}

// A field's `path` is either a store path string or a function(doc) that
// returns one -- the Receiver & Tower fields need the latter because the
// real path depends on which optics layout (doc.optics) is selected.
// Exported so the elevation view's dimension callouts (js/views/elevation.js)
// can bind their own hand-built <input> boxes to the same field descriptors
// without a second copy of this one-liner (docs/ui-spec.md 2.2's "every
// callout is an editable value box bound to the same fields as the sidebar").
export function resolvePath(path, doc) {
  return typeof path === "function" ? path(doc) : path;
}

// Builds one <div class="frow"><label>...</label><input class="val"></div>
// row wired straight to the field's store path -- identical markup and
// behavior (focused-input guard included, via setVal called from each
// render()) to the panels' former private numberRow helpers.
export function numberRow(parent, field) {
  const row = document.createElement("div");
  row.className = "frow";
  const lab = document.createElement("label");
  lab.textContent = field.label;
  const input = document.createElement("input");
  input.type = "number";
  input.className = "val";
  input.dataset.key = field.key;
  if (field.step !== undefined) input.step = field.step;
  if (field.min !== undefined) input.min = field.min;
  if (field.max !== undefined) input.max = field.max;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v)) {
      const doc = store.get("doc");
      store.set(resolvePath(field.path, doc), v);
    }
  });
  row.appendChild(lab);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

export function segButton(parent, label, active, onClick) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.className = active ? "active" : "";
  btn.addEventListener("click", onClick);
  parent.appendChild(btn);
  return btn;
}

// -- Heliostat stage: design fields (rect, grid, surface figure) -----------

export const HELIOSTAT_RECT_FIELDS = [
  { key: "width_mm", label: "Width (mm)", path: "doc.designParams.rect.width_mm", min: 1 },
  { key: "height_mm", label: "Height (mm)", path: "doc.designParams.rect.height_mm", min: 1 },
];

export const HELIOSTAT_GRID_FIELDS = [
  { key: "n_u", label: "Columns (n_u)", path: "doc.designParams.grid.n_u", min: 1, step: 1 },
  { key: "n_v", label: "Rows (n_v)", path: "doc.designParams.grid.n_v", min: 1, step: 1 },
  { key: "facet_w_mm", label: "Facet width (mm)", path: "doc.designParams.grid.facet_w_mm", min: 1 },
  { key: "facet_h_mm", label: "Facet height (mm)", path: "doc.designParams.grid.facet_h_mm", min: 1 },
  { key: "gap_mm", label: "Gap (mm)", path: "doc.designParams.grid.gap_mm", min: 0 },
];

export const HELIOSTAT_SURFACE_OPTIONS = [
  ["twisting", "Twisting"],
  ["spherical", "Spherical"],
  ["flat", "Flat"],
];

// -- Field stage: single-heliostat (x, y) and Fermat-spiral fields ---------

export const FIELD_SINGLE_FIELDS = [
  { key: "x_mm", label: "X (mm)", path: "doc.field.single.x_mm" },
  { key: "y_mm", label: "Y (mm)", path: "doc.field.single.y_mm" },
];

export const FIELD_FERMAT_FIELDS = [
  { key: "n", label: "Heliostats", path: "doc.field.fermat.n", min: 1, max: 10000, step: 1 },
  { key: "r_min_m", label: "Nearest radius (m)", path: "doc.field.fermat.r_min_m", min: 0 },
  { key: "r_max_m", label: "Farthest radius (m)", path: "doc.field.fermat.r_max_m", min: 0 },
];

// -- Receiver & Tower stage --------------------------------------------------
// Field names match heliostat.web.app's *Optics models exactly, so what
// the user types is what /api/scene/geometry and /api/trace read back
// under `optics_params` (docs/ui-spec.md 2.2's per-layout table -- no
// shared "tower height" alias).

function opticsPath(key) {
  return (doc) => `doc.opticsParams.${doc.optics}.${key}`;
}

export const RECEIVER_FIELD_TABLE = {
  prime_focus: [
    { key: "focus_height_mm", label: "Focus height (mm)", path: opticsPath("focus_height_mm") },
    { key: "window_half_u_mm", label: "Window ½ w (mm)", path: opticsPath("window_half_u_mm") },
    { key: "window_half_v_mm", label: "Window ½ h (mm)", path: opticsPath("window_half_v_mm") },
  ],
  axicon: [
    { key: "apex_height_mm", label: "Apex height (mm)", path: opticsPath("apex_height_mm") },
    { key: "half_angle_deg", label: "Half angle (°)", path: opticsPath("half_angle_deg") },
    { key: "aperture_radius_mm", label: "Aperture radius (mm)", path: opticsPath("aperture_radius_mm") },
    { key: "receiver_z_mm", label: "Receiver height (mm)", path: opticsPath("receiver_z_mm") },
    { key: "window_half_u_mm", label: "Window ½ w (mm)", path: opticsPath("window_half_u_mm") },
    { key: "window_half_v_mm", label: "Window ½ h (mm)", path: opticsPath("window_half_v_mm") },
  ],
  cassegrain: [
    { key: "vertex_z_mm", label: "Secondary vertex height (mm)", path: opticsPath("vertex_z_mm") },
    { key: "focus_height_mm", label: "Primary focus height (mm)", path: opticsPath("focus_height_mm") },
    { key: "receiver_z_mm", label: "Receiver height (mm)", path: opticsPath("receiver_z_mm") },
    { key: "aperture_radius_mm", label: "Aperture radius (mm)", path: opticsPath("aperture_radius_mm") },
    { key: "window_half_u_mm", label: "Window ½ w (mm)", path: opticsPath("window_half_u_mm") },
    { key: "window_half_v_mm", label: "Window ½ h (mm)", path: opticsPath("window_half_v_mm") },
  ],
};

export const OPTICS_LABELS = [
  ["prime_focus", "Prime focus"],
  ["axicon", "Axicon"],
  ["cassegrain", "Cassegrain"],
];

// -- Sun stage ---------------------------------------------------------------

export const SUN_FIELDS = [
  { key: "az", label: "Azimuth (°)", path: "doc.sun.az", min: 0, max: 360, step: 0.1 },
  { key: "el", label: "Elevation (°)", path: "doc.sun.el", min: -90, max: 90, step: 0.1 },
];

// -- Aperture-radius miss warning (docs/ui-spec.md 2.3 + 2.4) --------------
//
// Shared by the Receiver & Tower panel and the inspector so the identical
// amber message appears in both places whenever ui.miss (set from
// /api/scene/geometry's top-level `miss` key) names any heliostats that
// miss the aperture or can't reach the secondary at all. Defensive against
// `miss` being undefined/null (the backend contract isn't necessarily live
// yet) or either id list being absent -- both read as "no warning".
export function apertureMissMessage(miss) {
  if (!miss) return null;
  const apertureIds = miss.aperture_miss_ids || [];
  const totalIds = miss.total_miss_ids || [];
  if (!apertureIds.length && !totalIds.length) return null;

  // Two different physics, two different messages (Ryker's correction):
  // an aperture miss is fixable -- the reflected ray would land in the
  // receiver window, the rim just cuts it off first, so "needs >= X" is a
  // real purchase recommendation. A total miss is not: that heliostat's
  // light never reaches the receiver via this secondary (a near heliostat
  // that can't reach the cone, or a steep cone folding light away), and no
  // aperture size helps, so the message must not suggest one.
  let message;
  if (apertureIds.length) {
    const needed = miss.needed_aperture_radius_mm;
    const neededMm = needed == null ? null : Math.round(needed / 100) * 100;
    const neededTxt = neededMm == null ? "" : `needs ≥ ${neededMm.toLocaleString()} mm to catch the full field — `;
    message = `${neededTxt}${apertureIds.length} heliostat${apertureIds.length === 1 ? "" : "s"} miss the aperture`;
    if (totalIds.length) message += `; ${totalIds.length} can't reach the receiver at any aperture`;
  } else {
    message = `${totalIds.length} heliostat${totalIds.length === 1 ? "" : "s"} can't reach the receiver at this geometry — no aperture size catches them`;
  }
  return message;
}
