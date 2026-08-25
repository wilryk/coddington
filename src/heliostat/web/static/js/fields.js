// Shared field descriptors + row-building helpers for the sidebar stages
// (Heliostat, Field, Receiver & Tower, Sun) and the floating inspector,
// which renders the same fields bound to the same store paths. Every row's
// `input` event is the only place a typed value turns into a store.set --
// one edit pathway, not two kept in sync by hand.
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
export function resolvePath(path, doc) {
  return typeof path === "function" ? path(doc) : path;
}

// Builds one <div class="frow"><label>...</label><input class="val"></div>
// row wired straight to the field's store path, with setVal's focus guard
// applied on every render.
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

// -- Heliostat Shape tab: design fields (rect, grid, surface figure) -------

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

// Read-only report of the current aperture: type + key dimensions. Shared
// by the sidebar's Heliostat stage and the floating inspector so both say
// the same thing about the same design.
export function apertureSummaryText(doc) {
  const type = doc.design.type;
  if (type === "rect") {
    const p = doc.designParams.rect;
    const wM = (p.width_mm / 1000).toFixed(1);
    const hM = (p.height_mm / 1000).toFixed(1);
    return `${wM} × ${hM} m rectangle`;
  }
  if (type === "grid") {
    const p = doc.designParams.grid;
    return `${p.n_u}×${p.n_v} facet grid · ${p.facet_w_mm} × ${p.facet_h_mm} mm facets`;
  }
  const p = doc.designParams.custom;
  const n = (p && p.vertices_mm && p.vertices_mm.length) || 0;
  return `custom outline, ${n} vertices`;
}

// -- Field stage: single-heliostat (x, y), Fermat-spiral and radial-
// staggered fields -----------------------------------------------------

export const FIELD_SINGLE_FIELDS = [
  { key: "x_mm", label: "X (mm)", path: "doc.field.single.x_mm" },
  { key: "y_mm", label: "Y (mm)", path: "doc.field.single.y_mm" },
];

export const FIELD_FERMAT_FIELDS = [
  { key: "n", label: "Heliostats", path: "doc.field.fermat.n", min: 1, max: 10000, step: 1 },
  { key: "r_min_m", label: "Nearest radius (m)", path: "doc.field.fermat.r_min_m", min: 0 },
  { key: "r_max_m", label: "Farthest radius (m)", path: "doc.field.fermat.r_max_m", min: 0 },
];

// Rings per band and heliostats per ring, one pair per band -- the field's
// radii are not user-editable here (see api.js's radialStaggerPayload for
// how an edited ring count reshapes them).
export const FIELD_RADIAL_STAGGER_FIELDS = [
  { key: "band0Rings", label: "Inner rings", path: "doc.field.radialStagger.band0Rings", min: 1, step: 1 },
  {
    key: "band0Count",
    label: "Inner heliostats/ring",
    path: "doc.field.radialStagger.band0Count",
    min: 1,
    step: 1,
  },
  { key: "band1Rings", label: "Middle rings", path: "doc.field.radialStagger.band1Rings", min: 1, step: 1 },
  {
    key: "band1Count",
    label: "Middle heliostats/ring",
    path: "doc.field.radialStagger.band1Count",
    min: 1,
    step: 1,
  },
  { key: "band2Rings", label: "Outer rings", path: "doc.field.radialStagger.band2Rings", min: 1, step: 1 },
  {
    key: "band2Count",
    label: "Outer heliostats/ring",
    path: "doc.field.radialStagger.band2Count",
    min: 1,
    step: 1,
  },
];

// -- Receiver & Tower stage --------------------------------------------------
// Field names match heliostat.web.app's *Optics models exactly, so what
// the user types is what /api/scene/geometry and /api/trace read back
// under `optics_params` -- no shared "tower height" alias.

function opticsPath(key) {
  return (doc) => `doc.opticsParams.${doc.optics}.${key}`;
}

// Fields with a `group` show only while prime_focus's own receiver_type
// (a string, kept out of this table -- see RECEIVER_TYPE_OPTIONS below)
// equals that group; fields with no `group` are common to every receiver
// type. window_half_u/v_mm has no group -- it now sizes the entrance
// aperture (a flat rectangle) whichever receiver_type sits behind it.
export const RECEIVER_FIELD_TABLE = {
  prime_focus: [
    { key: "focus_height_mm", label: "Focus height (mm)", path: opticsPath("focus_height_mm") },
    { key: "window_half_u_mm", label: "Window ½ w (mm)", path: opticsPath("window_half_u_mm") },
    { key: "window_half_v_mm", label: "Window ½ h (mm)", path: opticsPath("window_half_v_mm") },
    { key: "receiver_center_x_mm", label: "Receiver centre X (mm)", path: opticsPath("receiver_center_x_mm") },
    { key: "receiver_center_y_mm", label: "Receiver centre Y (mm)", path: opticsPath("receiver_center_y_mm") },
    {
      key: "aperture_to_receiver_mm",
      label: "Aperture → receiver (mm)",
      path: opticsPath("aperture_to_receiver_mm"),
      min: 0,
    },
    {
      key: "cylinder_radius_mm",
      label: "Cylinder radius (mm)",
      path: opticsPath("cylinder_radius_mm"),
      min: 1,
      group: "cylinder",
    },
    {
      key: "cylinder_height_mm",
      label: "Cylinder height (mm)",
      path: opticsPath("cylinder_height_mm"),
      min: 1,
      group: "cylinder",
    },
    {
      key: "frustum_top_radius_mm",
      label: "Frustum top radius (mm)",
      path: opticsPath("frustum_top_radius_mm"),
      min: 1,
      group: "frustum",
    },
    {
      key: "frustum_bottom_radius_mm",
      label: "Frustum bottom radius (mm)",
      path: opticsPath("frustum_bottom_radius_mm"),
      min: 1,
      group: "frustum",
    },
    {
      key: "frustum_height_mm",
      label: "Frustum height (mm)",
      path: opticsPath("frustum_height_mm"),
      min: 1,
      group: "frustum",
    },
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

// receiver_type only exists for prime_focus -- axicon/cassegrain always
// have a flat window and carry no such field.
export const RECEIVER_TYPE_OPTIONS = [
  ["flat", "Flat window"],
  ["cylinder", "Cylindrical"],
  ["frustum", "Frustum"],
];

// Shared by the sidebar's Receiver & Tower stage and the inspector so a
// cylinder/frustum-only row (RECEIVER_FIELD_TABLE's `group`) shows in both
// places exactly when the current receiver_type matches it.
export function receiverFieldVisible(field, params) {
  return !field.group || field.group === params.receiver_type;
}

// -- Sun stage ---------------------------------------------------------------

export const SUN_FIELDS = [
  { key: "az", label: "Azimuth (°)", path: "doc.sun.az", min: 0, max: 360, step: 0.1 },
  { key: "el", label: "Elevation (°)", path: "doc.sun.el", min: -90, max: 90, step: 0.1 },
];

// -- Aperture-radius miss warning ------------------------------------------
//
// Shared by the Receiver & Tower panel and the inspector so the identical
// amber message appears in both places. Defensive against `miss` being
// undefined/null or either id list being absent -- both read as "no
// warning".
export function apertureMissMessage(miss) {
  if (!miss) return null;
  const apertureIds = miss.aperture_miss_ids || [];
  const totalIds = miss.total_miss_ids || [];
  if (!apertureIds.length && !totalIds.length) return null;

  // An aperture miss is fixable -- the reflected ray would land in the
  // receiver window, the rim just cuts it off first, so "needs >= X" is a
  // real purchase recommendation. A total miss is not: that heliostat's
  // light never reaches the receiver via this secondary, and no aperture
  // size helps, so the message must not suggest one.
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
