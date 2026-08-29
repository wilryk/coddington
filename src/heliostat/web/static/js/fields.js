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

// docs/ui-spec-v0.2.md §H: a position/radius field stored in mm can declare
// `unit: "m"` to display and edit in meters (rounded to 0.01 m on display --
// storage keeps full mm precision; a typed value converts back to mm on
// write, at any precision the user types). metersFromMm is exported so
// non-numberRow readouts (plan view, inspector) round the same way.
export function metersFromMm(mm) {
  if (mm == null || !Number.isFinite(mm)) return null;
  return Math.round(mm / 10) / 100; // nearest 0.01 m
}

// Builds one <div class="frow"><label>...</label><input class="val"></div>
// row wired straight to the field's store path, with setVal's focus guard
// applied on every render. `field.tooltip`, when present, becomes the row's
// native hover title -- the one mechanism docs/ui-spec-v0.2.md §G asks for,
// since every sidebar/inspector field already renders through here.
export function numberRow(parent, field) {
  const row = document.createElement("div");
  row.className = "frow";
  if (field.tooltip) row.title = field.tooltip;
  const lab = document.createElement("label");
  lab.textContent = field.label;
  // docs/ui-spec-v0.2.md §E2: a field carrying `badge` (e.g. "Monte Carlo
  // only") gets the same `.badge.import` pill the Shape tab's measured
  // error-map chip uses (f2bb0f4) -- rigid-body fields stay badge-free
  // since they apply at every fidelity; the map/warp fields do not.
  if (field.badge) {
    const badge = document.createElement("span");
    badge.className = "badge import fieldbadge";
    badge.textContent = field.badge;
    lab.appendChild(badge);
  }
  const input = document.createElement("input");
  input.type = "number";
  input.className = "val";
  input.dataset.key = field.key;
  if (field.step !== undefined) input.step = field.step;
  else if (field.unit === "m") input.step = 0.01;
  if (field.min !== undefined) input.min = field.min;
  if (field.max !== undefined) input.max = field.max;
  input.addEventListener("input", () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v)) {
      const doc = store.get("doc");
      const stored = field.unit === "m" ? v * 1000 : v;
      store.set(resolvePath(field.path, doc), stored);
    }
  });
  row.appendChild(lab);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

// A subgroup heading within a stage/inspector's field list -- the
// `sectionHeader`/`sectionHeaderBadge` a field descriptor can carry (see
// RECEIVER_FIELD_TABLE's "Perturbations" group). Shared by
// panels/receiver.js and inspector.js so both stay visually identical, the
// same reasoning their own comments already give for numberRow.
export function sectionHeaderRow(parent, text, badgeText) {
  const sub = document.createElement("div");
  sub.className = "subhead";
  sub.textContent = text;
  if (badgeText) {
    const badge = document.createElement("span");
    badge.className = "badge import fieldbadge";
    badge.textContent = badgeText;
    sub.appendChild(badge);
  }
  parent.appendChild(sub);
  return sub;
}

export function segButton(parent, label, active, onClick, tooltip) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.className = active ? "active" : "";
  if (tooltip) btn.title = tooltip;
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

// docs/ui-spec-v0.2.md §H: heliostat positions display and edit in meters
// (rounded to 0.01 m) though storage stays full-precision mm -- unit: "m"
// tells numberRow to convert on read/write (see its own comment above).
export const FIELD_SINGLE_FIELDS = [
  {
    key: "x_mm",
    label: "X (m)",
    path: "doc.field.single.x_mm",
    unit: "m",
    tooltip: "East/west position of this heliostat, relative to the tower axis.",
  },
  {
    key: "y_mm",
    label: "Y (m)",
    path: "doc.field.single.y_mm",
    unit: "m",
    tooltip: "North/south position of this heliostat, relative to the tower axis.",
  },
];

export const FIELD_FERMAT_FIELDS = [
  { key: "n", label: "Heliostats", path: "doc.field.fermat.n", min: 1, max: 10000, step: 1, tooltip: "How many heliostats the spiral places." },
  {
    key: "r_min_m",
    label: "Nearest radius (m)",
    path: "doc.field.fermat.r_min_m",
    min: 0,
    step: 0.01,
    tooltip: "Radius from the tower axis where the spiral starts placing heliostats.",
  },
  {
    key: "r_max_m",
    label: "Farthest radius (m)",
    path: "doc.field.fermat.r_max_m",
    min: 0,
    step: 0.01,
    tooltip: "Radius from the tower axis where the spiral stops placing heliostats.",
  },
];

// Rings per band and heliostats per ring, one pair per band -- the field's
// radii are not user-editable here (see api.js's radialStaggerPayload for
// how an edited ring count reshapes them).
export const FIELD_RADIAL_STAGGER_FIELDS = [
  {
    key: "band0Rings",
    label: "Inner rings",
    path: "doc.field.radialStagger.band0Rings",
    min: 1,
    step: 1,
    tooltip: "Number of concentric rings in the innermost band, closest to the tower.",
  },
  {
    key: "band0Count",
    label: "Inner heliostats/ring",
    path: "doc.field.radialStagger.band0Count",
    min: 1,
    step: 1,
    tooltip: "Heliostats staggered around each ring of the inner band.",
  },
  {
    key: "band1Rings",
    label: "Middle rings",
    path: "doc.field.radialStagger.band1Rings",
    min: 1,
    step: 1,
    tooltip: "Number of concentric rings in the middle band.",
  },
  {
    key: "band1Count",
    label: "Middle heliostats/ring",
    path: "doc.field.radialStagger.band1Count",
    min: 1,
    step: 1,
    tooltip: "Heliostats staggered around each ring of the middle band.",
  },
  {
    key: "band2Rings",
    label: "Outer rings",
    path: "doc.field.radialStagger.band2Rings",
    min: 1,
    step: 1,
    tooltip: "Number of concentric rings in the outermost band, farthest from the tower.",
  },
  {
    key: "band2Count",
    label: "Outer heliostats/ring",
    path: "doc.field.radialStagger.band2Count",
    min: 1,
    step: 1,
    tooltip: "Heliostats staggered around each ring of the outer band.",
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
// type. window_half_u/v_mm and aperture_to_receiver_mm are tagged
// group: "flat" -- they describe the flat entrance aperture / its offset
// from the absorbing surface, which only reads as a meaningful control
// when that absorbing surface is itself the flat window (v0.2 fix wave
// item 1: they don't apply -- visibly -- to a cylinder/frustum receiver,
// so they hide there; values are preserved and reappear under Flat).
// Tooltips below are one-sentence restatements of this table's own labels
// and the design comments that used to be the only place this prose lived
// (docs/ui-spec-v0.2.md §G). Receiver/tower fields stay in mm throughout --
// docs/ui-spec-v0.2.md §H confirms mirror/receiver *dimensions* are
// fabrication numbers, unlike heliostat positions.
export const RECEIVER_FIELD_TABLE = {
  prime_focus: [
    {
      key: "focus_height_mm",
      label: "Focus height (mm)",
      path: opticsPath("focus_height_mm"),
      tooltip: "Height above the heliostat plane where the primary field focuses its beam.",
    },
    {
      key: "window_half_u_mm",
      label: "Window ½ w (mm)",
      path: opticsPath("window_half_u_mm"),
      group: "flat",
      tooltip: "Half-width of the flat entrance aperture the receiver looks through.",
    },
    {
      key: "window_half_v_mm",
      label: "Window ½ h (mm)",
      path: opticsPath("window_half_v_mm"),
      group: "flat",
      tooltip: "Half-height of the flat entrance aperture the receiver looks through.",
    },
    {
      key: "receiver_center_x_mm",
      label: "Receiver centre X (mm)",
      path: opticsPath("receiver_center_x_mm"),
      tooltip: "East/west offset of the receiver's centre from the tower axis.",
    },
    {
      key: "receiver_center_y_mm",
      label: "Receiver centre Y (mm)",
      path: opticsPath("receiver_center_y_mm"),
      tooltip: "North/south offset of the receiver's centre from the tower axis.",
    },
    {
      key: "aperture_to_receiver_mm",
      label: "Aperture → receiver (mm)",
      path: opticsPath("aperture_to_receiver_mm"),
      min: 0,
      group: "flat",
      tooltip: "Distance from the flat entrance aperture back to the absorbing surface behind it.",
    },
    {
      key: "cylinder_radius_mm",
      label: "Cylinder radius (mm)",
      path: opticsPath("cylinder_radius_mm"),
      min: 1,
      group: "cylinder",
      tooltip: "Radius of the cylindrical absorbing surface.",
    },
    {
      key: "cylinder_height_mm",
      label: "Cylinder height (mm)",
      path: opticsPath("cylinder_height_mm"),
      min: 1,
      group: "cylinder",
      tooltip: "Height of the cylindrical absorbing surface.",
    },
    {
      key: "frustum_top_radius_mm",
      label: "Frustum top radius (mm)",
      path: opticsPath("frustum_top_radius_mm"),
      min: 1,
      group: "frustum",
      tooltip: "Radius at the top of the conical (frustum) absorbing surface.",
    },
    {
      key: "frustum_bottom_radius_mm",
      label: "Frustum bottom radius (mm)",
      path: opticsPath("frustum_bottom_radius_mm"),
      min: 1,
      group: "frustum",
      tooltip: "Radius at the bottom of the conical (frustum) absorbing surface.",
    },
    {
      key: "frustum_height_mm",
      label: "Frustum height (mm)",
      path: opticsPath("frustum_height_mm"),
      min: 1,
      group: "frustum",
      tooltip: "Height of the conical (frustum) absorbing surface.",
    },
  ],
  axicon: [
    {
      key: "apex_height_mm",
      label: "Apex height (mm)",
      path: opticsPath("apex_height_mm"),
      tooltip: "Height above the heliostat plane of the axicon secondary's apex.",
    },
    {
      key: "half_angle_deg",
      label: "Half angle (°)",
      path: opticsPath("half_angle_deg"),
      tooltip: "Cone half-angle of the axicon secondary, measured from its axis.",
    },
    {
      key: "aperture_radius_mm",
      label: "Aperture radius (mm)",
      path: opticsPath("aperture_radius_mm"),
      tooltip: "Outer radius of the axicon secondary's reflecting surface.",
    },
    {
      key: "receiver_z_mm",
      label: "Receiver height (mm)",
      path: opticsPath("receiver_z_mm"),
      tooltip: "Height above the heliostat plane of the receiver behind the secondary.",
    },
    {
      key: "window_half_u_mm",
      label: "Window ½ w (mm)",
      path: opticsPath("window_half_u_mm"),
      tooltip: "Half-width of the receiver's flat entrance aperture.",
    },
    {
      key: "window_half_v_mm",
      label: "Window ½ h (mm)",
      path: opticsPath("window_half_v_mm"),
      tooltip: "Half-height of the receiver's flat entrance aperture.",
    },
    // Spec §C: default 0.90, the value already in use, now a visible input
    // (docs/ui-spec-v0.2.md §K.3) instead of an assumed constant. A plain
    // 0-1 fraction, matching AxiconOptics.secondary_reflectance's own wire
    // units exactly -- no percent conversion layer, unlike design.errors'
    // reflectance_pct (see store.js's DEFAULT_DOC comment).
    {
      key: "secondary_reflectance",
      label: "Secondary reflectance (R)",
      path: opticsPath("secondary_reflectance"),
      min: 0,
      max: 1,
      step: 0.01,
      tooltip: "Fraction of secondary-incident power the secondary itself reflects back out; absorbed heat is (1 − R) × incident.",
    },
    // docs/ui-spec-v0.2.md §E2: rigid-body misalignment of the secondary --
    // exact geometry at every fidelity (unlike §C's flux-map fidelity note,
    // no caveat needed). Wire units match AxiconOptics.secondary_dx_mm et
    // al. exactly -- mm/mrad, no conversion layer -- and §H keeps
    // mirror/receiver fabrication numbers in mm, so these stay unconverted
    // like the rest of this table. sectionHeader groups them under their
    // own "Perturbations" subhead (see panels/receiver.js + inspector.js).
    {
      key: "secondary_dx_mm",
      label: "Decenter Δx (mm)",
      path: opticsPath("secondary_dx_mm"),
      sectionHeader: "Perturbations",
      tooltip: "East/west rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_dy_mm",
      label: "Decenter Δy (mm)",
      path: opticsPath("secondary_dy_mm"),
      tooltip: "North/south rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_dz_mm",
      label: "Decenter Δz (mm)",
      path: opticsPath("secondary_dz_mm"),
      tooltip: "Vertical rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_tip_mrad",
      label: "Tip (mrad)",
      path: opticsPath("secondary_tip_mrad"),
      tooltip: "Rigid-body rotation of the secondary about its vertex, tilting the surface north/south.",
    },
    {
      key: "secondary_tilt_mrad",
      label: "Tilt (mrad)",
      path: opticsPath("secondary_tilt_mrad"),
      tooltip: "Rigid-body rotation of the secondary about its vertex, tilting the surface east/west.",
    },
    // docs/ui-spec-v0.2.md §E2: parametric warp -- quick what-ifs without an
    // FEA file, summed with an imported measured map (below) in Monte Carlo
    // ONLY (unlike the rigid-body fields above, which apply at every
    // fidelity -- hence the badge here and not there). Its own
    // sectionHeader starts a new subgroup within "Perturbations" (see
    // panels/receiver.js/inspector.js's sectionHeaderRow).
    // §E2's measured error map on the secondary -- the §E machinery reused
    // verbatim (same gridded Δ-sag CSV convention, same import/stats
    // endpoints, same storage shape), just parked under
    // secondary_error_map instead of design.errors.error_map. `custom:
    // true` means this key has no numberRow -- panels/receiver.js renders
    // its own import-chip UI for it (mirroring tabs/shape.js's) -- but it
    // must still be a real table entry so api.js's currentOpticsParams
    // does not treat it as a foreign key and drop it at the request
    // boundary. Its sectionHeader opens the "Surface deformation & warp"
    // subgroup that the parametric-warp fields below continue.
    {
      key: "secondary_error_map",
      custom: true,
      sectionHeader: "Surface deformation & warp",
      sectionHeaderBadge: "Monte Carlo only",
    },
    {
      key: "secondary_defocus_um",
      label: "Defocus (µm P-V)",
      path: opticsPath("secondary_defocus_um"),
      tooltip: "Peak-to-valley defocus added to the secondary's figure over its aperture. Applied in Monte Carlo traces only; sums with an imported map and defaults to zero.",
    },
    {
      key: "secondary_astig_um",
      label: "Astigmatism (µm P-V)",
      path: opticsPath("secondary_astig_um"),
      tooltip: "Peak-to-valley astigmatism added to the secondary's figure over its aperture, along the axis angle below. Applied in Monte Carlo traces only; sums with an imported map and defaults to zero.",
    },
    {
      key: "secondary_astig_axis_deg",
      label: "Astigmatism axis (deg)",
      path: opticsPath("secondary_astig_axis_deg"),
      tooltip: "Orientation of the astigmatism above, degrees from the local x-axis.",
    },
  ],
  cassegrain: [
    {
      key: "vertex_z_mm",
      label: "Secondary vertex height (mm)",
      path: opticsPath("vertex_z_mm"),
      tooltip: "Height above the heliostat plane of the secondary mirror's vertex.",
    },
    {
      key: "focus_height_mm",
      label: "Primary focus height (mm)",
      path: opticsPath("focus_height_mm"),
      tooltip: "Height above the heliostat plane where the primary field would focus without the secondary.",
    },
    {
      key: "receiver_z_mm",
      label: "Receiver height (mm)",
      path: opticsPath("receiver_z_mm"),
      tooltip: "Height above the heliostat plane of the receiver behind the secondary.",
    },
    {
      key: "aperture_radius_mm",
      label: "Aperture radius (mm)",
      path: opticsPath("aperture_radius_mm"),
      tooltip: "Outer radius of the secondary mirror's reflecting surface.",
    },
    {
      key: "window_half_u_mm",
      label: "Window ½ w (mm)",
      path: opticsPath("window_half_u_mm"),
      tooltip: "Half-width of the receiver's flat entrance aperture.",
    },
    {
      key: "window_half_v_mm",
      label: "Window ½ h (mm)",
      path: opticsPath("window_half_v_mm"),
      tooltip: "Half-height of the receiver's flat entrance aperture.",
    },
    // See axicon's identical field above -- CassegrainOptics.secondary_reflectance.
    {
      key: "secondary_reflectance",
      label: "Secondary reflectance (R)",
      path: opticsPath("secondary_reflectance"),
      min: 0,
      max: 1,
      step: 0.01,
      tooltip: "Fraction of secondary-incident power the secondary itself reflects back out; absorbed heat is (1 − R) × incident.",
    },
    // See axicon's identical fields above -- CassegrainOptics.secondary_dx_mm
    // et al., rotation about the Cassegrain secondary's own vertex_z_mm.
    {
      key: "secondary_dx_mm",
      label: "Decenter Δx (mm)",
      path: opticsPath("secondary_dx_mm"),
      sectionHeader: "Perturbations",
      tooltip: "East/west rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_dy_mm",
      label: "Decenter Δy (mm)",
      path: opticsPath("secondary_dy_mm"),
      tooltip: "North/south rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_dz_mm",
      label: "Decenter Δz (mm)",
      path: opticsPath("secondary_dz_mm"),
      tooltip: "Vertical rigid-body decenter of the secondary from its design position.",
    },
    {
      key: "secondary_tip_mrad",
      label: "Tip (mrad)",
      path: opticsPath("secondary_tip_mrad"),
      tooltip: "Rigid-body rotation of the secondary about its vertex, tilting the surface north/south.",
    },
    {
      key: "secondary_tilt_mrad",
      label: "Tilt (mrad)",
      path: opticsPath("secondary_tilt_mrad"),
      tooltip: "Rigid-body rotation of the secondary about its vertex, tilting the surface east/west.",
    },
    // See axicon's identical fields above -- CassegrainOptics.secondary_defocus_um et al.
    // §E2's measured error map on the secondary -- the §E machinery reused
    // verbatim (same gridded Δ-sag CSV convention, same import/stats
    // endpoints, same storage shape), just parked under
    // secondary_error_map instead of design.errors.error_map. `custom:
    // true` means this key has no numberRow -- panels/receiver.js renders
    // its own import-chip UI for it (mirroring tabs/shape.js's) -- but it
    // must still be a real table entry so api.js's currentOpticsParams
    // does not treat it as a foreign key and drop it at the request
    // boundary. Its sectionHeader opens the "Surface deformation & warp"
    // subgroup that the parametric-warp fields below continue.
    {
      key: "secondary_error_map",
      custom: true,
      sectionHeader: "Surface deformation & warp",
      sectionHeaderBadge: "Monte Carlo only",
    },
    {
      key: "secondary_defocus_um",
      label: "Defocus (µm P-V)",
      path: opticsPath("secondary_defocus_um"),
      tooltip: "Peak-to-valley defocus added to the secondary's figure over its aperture. Applied in Monte Carlo traces only; sums with an imported map and defaults to zero.",
    },
    {
      key: "secondary_astig_um",
      label: "Astigmatism (µm P-V)",
      path: opticsPath("secondary_astig_um"),
      tooltip: "Peak-to-valley astigmatism added to the secondary's figure over its aperture, along the axis angle below. Applied in Monte Carlo traces only; sums with an imported map and defaults to zero.",
    },
    {
      key: "secondary_astig_axis_deg",
      label: "Astigmatism axis (deg)",
      path: opticsPath("secondary_astig_axis_deg"),
      tooltip: "Orientation of the astigmatism above, degrees from the local x-axis.",
    },
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
  {
    key: "az",
    label: "Azimuth (°)",
    path: "doc.sun.az",
    min: 0,
    max: 360,
    step: 0.1,
    tooltip: "Compass bearing the sun is at (0° = North, clockwise).",
  },
  {
    key: "el",
    label: "Elevation (°)",
    path: "doc.sun.el",
    min: -90,
    max: 90,
    step: 0.1,
    tooltip: "Angle of the sun above the horizon.",
  },
];

// docs/ui-spec-v0.2.md §O: circumsolar ratio -- a Sunshape-stage field, kept
// out of SUN_FIELDS above deliberately (inspector.js renders every
// SUN_FIELDS entry in the 3D View's compact sun chip too; §O/mockup M19
// place CSR only in the Design tab's full Sun stage, not that chip), so
// panels/sun.js renders this one directly instead of through SUN_FIELDS.
export const SUN_CSR_FIELD = {
  key: "csr",
  label: "CSR",
  path: "doc.sun.csr",
  min: 0,
  max: 1,
  step: 0.01,
  tooltip:
    "Circumsolar ratio: the fraction of the sun's radiance in the hazy aureole around the disk versus the disk itself. 0 is a clean cutoff disk (today's default); typical clear-sky sites read under 0.1, hazy skies higher.",
};

// Where and when, which the server turns into an azimuth and an elevation.
export const SUN_SITE_FIELDS = [
  {
    key: "latitude_deg",
    label: "Latitude (°)",
    path: "doc.sun.site.latitude_deg",
    min: -90,
    max: 90,
    step: 0.0001,
    tooltip: "Site latitude, used with date and time to solve the sun's azimuth and elevation.",
  },
  {
    key: "longitude_deg",
    label: "Longitude (°)",
    path: "doc.sun.site.longitude_deg",
    min: -180,
    max: 180,
    step: 0.0001,
    tooltip: "Site longitude, used with date and time to solve the sun's azimuth and elevation.",
  },
  {
    key: "timezone_h",
    label: "UTC offset (h)",
    path: "doc.sun.site.timezone_h",
    min: -14,
    max: 14,
    step: 0.25,
    tooltip: "Hours offset from UTC the site's local time and date are given in.",
  },
  {
    key: "hour",
    label: "Local time (h)",
    path: "doc.sun.site.hour",
    min: 0,
    max: 23.99,
    step: 0.25,
    tooltip: "Local time of day the sun position is solved for.",
  },
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
