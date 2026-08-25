// Project schema v1 <-> store, and the legacy-setup importer (Phase 3b,
// docs/ui-spec.md 5's "Projects" collection + "Migration"). Pure functions
// over a `doc`/`ui` pair or a document dict, except applyProject() and
// convertLegacySetup()'s caller, which write the store the same way
// main.js's own helpers do -- js/library.js is the only thing that calls
// into this module, and it owns the network/DOM side of the Library
// drawer, so this file stays free of both.
import { store, DEFAULT_DOC } from "./store.js";
import {
  DESIGN_ERROR_KEYS,
  currentDesignPayload,
  currentLayoutPayload,
  currentOpticsParams,
  errorsFromDesignDocument,
  getManuscriptField,
} from "./api.js";
import { OPTICS_LABELS } from "./fields.js";

const OPTICS_KEYS = OPTICS_LABELS.map(([key]) => key);
const DESIGN_TYPES = ["rect", "grid", "custom"]; // the three this client's UI understands (see applyProject)

function stripKeys(obj, keys) {
  const out = Object.assign({}, obj);
  for (const k of keys) delete out[k];
  return out;
}

// A positions layout applies as doc.field.layout = "manuscript" exactly
// when its xy_mm is the cached manuscript field (same length, every pair
// equal within 0.1 mm -- the endpoint's own rounding, see
// heliostat.web.app's field_manuscript docstring). Any other positions
// layout is a shape this client has no picker for yet -- see applyProject's
// error message.
function xyMatchesManuscriptField(xy) {
  const cached = getManuscriptField();
  if (!cached || !Array.isArray(xy) || xy.length !== cached.length) return false;
  for (let i = 0; i < xy.length; i++) {
    const a = xy[i];
    const b = cached[i];
    if (!Array.isArray(a) || a.length < 2) return false;
    if (Math.abs(a[0] - b[0]) > 0.1 || Math.abs(a[1] - b[1]) > 0.1) return false;
  }
  return true;
}

// -- store -> ProjectDocument (app.ProjectDocument, schema_version 1) ------

export function serializeProject(doc, ui) {
  const design = currentDesignPayload(doc);
  // currentOpticsParams rather than doc.opticsParams[doc.optics] for two
  // reasons: the saved document must never share an object with the live
  // store (a later caller mutating the document would silently edit the
  // workspace), and the server's ReceiverDocument forbids foreign keys the
  // same way a trace request does -- same whitelist, same boundary.
  const receiver = { optics: doc.optics, params: currentOpticsParams(doc) };
  const field =
    doc.field.mode === "field"
      ? {
          layout: currentLayoutPayload(doc),
          heliostat_x_mm: doc.field.single.x_mm,
          heliostat_y_mm: doc.field.single.y_mm,
        }
      : {
          layout: null,
          heliostat_x_mm: doc.field.single.x_mm,
          heliostat_y_mm: doc.field.single.y_mm,
        };
  const sun = { azimuth_deg: doc.sun.az, elevation_deg: doc.sun.el };
  // Note: `doc.sun` carries no `site` today -- the Sun stage is a plain
  // az/el pair in this phase (docs/ui-spec.md 2.2, "site & time entry is a
  // later phase") -- so re-saving a project that started life as an
  // imported legacy setup silently drops whatever `site` that setup had.
  // Acceptable for phase 3b per the build brief; a future Sun stage that
  // keeps the site around would restore it here.
  const run = {
    mode: ui.fidelity,
    n_rays: ui.fidelity === "monte_carlo" ? ui.mcRays : null,
  };
  return { schema_version: 1, design, receiver, field, sun, run };
}

// -- ProjectDocument -> store -----------------------------------------------
//
// Validates the whole document *before* writing anything, so a document
// this client can't fully represent (an unsupported design type, a
// positions field layout) leaves the store untouched rather than half
// applied. Returns null on success, or a human-readable error string.
export function applyProject(document) {
  const design = document && document.design;
  if (!design || DESIGN_TYPES.indexOf(design.type) === -1) {
    return `design type ${design && design.type ? `"${design.type}"` : "(missing)"} isn't supported by this workspace yet`;
  }
  const field = (document && document.field) || {};
  if (field.layout && field.layout.type === "positions") {
    if (!xyMatchesManuscriptField(field.layout.xy_mm)) {
      return "custom position layouts arrive with the layout picker";
    }
  } else if (field.layout && field.layout.type !== "fermat") {
    return `field layout "${field.layout.type}" isn't supported yet (only the Fermat spiral is)`;
  }
  const receiver = document && document.receiver;
  if (!receiver || OPTICS_KEYS.indexOf(receiver.optics) === -1) {
    return `receiver optics ${receiver && receiver.optics ? `"${receiver.optics}"` : "(missing)"} isn't recognized`;
  }

  // -- validated: now write --------------------------------------------
  store.set("doc.design.type", design.type);
  store.set("doc.design.surface", design.surface || "twisting");
  // The optical-error fields ride flat in the design document (phase 3c);
  // they belong under doc.design.errors, not in doc.designParams, so strip
  // them before the params merge. A custom design's document carries its
  // mirror-expanded vertex list, so it loads with mirror off -- same shape,
  // just no longer editable as a half-sketch.
  store.set("doc.design.errors", errorsFromDesignDocument(design));
  const designParams = stripKeys(design, ["type", "surface"].concat(DESIGN_ERROR_KEYS));
  store.set(`doc.designParams.${design.type}`, Object.assign({}, DEFAULT_DOC.designParams[design.type], designParams));

  store.set("doc.optics", receiver.optics);
  store.set(
    `doc.opticsParams.${receiver.optics}`,
    Object.assign({}, DEFAULT_DOC.opticsParams[receiver.optics], receiver.params || {})
  );

  if (field.layout && field.layout.type === "fermat") {
    store.set("doc.field.mode", "field");
    store.set("doc.field.layout", "fermat");
    store.set("doc.field.fermat", {
      n: field.layout.n,
      r_min_m: field.layout.r_min_m != null ? field.layout.r_min_m : null,
      r_max_m: field.layout.r_max_m != null ? field.layout.r_max_m : null,
    });
  } else if (field.layout && field.layout.type === "positions") {
    // Validated above: xy_mm matches the cached manuscript field exactly,
    // so this project's field IS the manuscript layout -- store it as that
    // symbolic choice rather than a frozen positions blob, so it stays live
    // (and re-fetchable/re-serializable) exactly like a freshly-defaulted
    // document's field does.
    store.set("doc.field.mode", "field");
    store.set("doc.field.layout", "manuscript");
  } else {
    store.set("doc.field.mode", "single");
  }
  store.set("doc.field.single", {
    x_mm: field.heliostat_x_mm != null ? field.heliostat_x_mm : DEFAULT_DOC.field.single.x_mm,
    y_mm: field.heliostat_y_mm != null ? field.heliostat_y_mm : DEFAULT_DOC.field.single.y_mm,
  });

  const sun = (document && document.sun) || {};
  store.set("doc.sun", {
    az: sun.azimuth_deg != null ? sun.azimuth_deg : DEFAULT_DOC.sun.az,
    el: sun.elevation_deg != null ? sun.elevation_deg : DEFAULT_DOC.sun.el,
  });
  // `sun.site`, if present, is intentionally not stored anywhere -- see
  // serializeProject()'s comment on why a resave then drops it.

  const run = (document && document.run) || {};
  store.set("ui.fidelity", run.mode || "ultra_fast");
  store.set("ui.mcRays", run.n_rays != null ? run.n_rays : null);

  return null;
}

// -- legacy setup (heliostat.web.setups' free-form document) -> ProjectDocument
//
// The old GUI's saved-setup shape: {version, values: {<control id>: string},
// designType, surface, mode, traceMode, opticsEdits: {prime_focus, axicon,
// cassegrain}}. Field names in `values` are the old control ids (kept
// exactly as the brief lists them); `opticsEdits`' dicts already use the
// server's own field names (half_angle_deg and friends), same as
// app.ReceiverDocument.params. Returns {document, unmapped} -- `document`
// is not yet validated against ProjectDocument (the save call does that
// server-side and its 422 detail, if any, belongs in the caller's report).
const KNOWN_VALUES_KEYS = [
  "rect-width",
  "rect-height",
  "grid-nu",
  "grid-nv",
  "grid-fw",
  "grid-fh",
  "grid-gap",
  "grid-cant",
  "optics",
  "helio-x",
  "helio-y",
  "field-n",
  "field-rmin",
  "field-rmax",
  "sun-az",
  "sun-el",
  "site-lat",
  "site-lon",
  "site-tz",
  "site-date",
  "site-hour", // known, but never mapped -- see below
  "tower-height",
  "recv-half-u",
  "recv-half-v",
];

export function convertLegacySetup(old) {
  const unmapped = [];
  const values = (old && old.values) || {};

  function num(key) {
    const raw = values[key];
    if (raw === undefined || raw === null || raw === "") return undefined;
    const v = parseFloat(raw);
    return Number.isFinite(v) ? v : undefined;
  }

  // -- design ---------------------------------------------------------
  const designType = old && old.designType === "grid" ? "grid" : "rect";
  const surface = (old && old.surface) || "twisting";
  const design = { type: designType, surface };
  if (designType === "rect") {
    design.width_mm = num("rect-width") != null ? num("rect-width") : DEFAULT_DOC.designParams.rect.width_mm;
    design.height_mm = num("rect-height") != null ? num("rect-height") : DEFAULT_DOC.designParams.rect.height_mm;
  } else {
    const gridDefaults = DEFAULT_DOC.designParams.grid;
    design.n_u = num("grid-nu") != null ? num("grid-nu") : gridDefaults.n_u;
    design.n_v = num("grid-nv") != null ? num("grid-nv") : gridDefaults.n_v;
    design.facet_w_mm = num("grid-fw") != null ? num("grid-fw") : gridDefaults.facet_w_mm;
    design.facet_h_mm = num("grid-fh") != null ? num("grid-fh") : gridDefaults.facet_h_mm;
    design.gap_mm = num("grid-gap") != null ? num("grid-gap") : gridDefaults.gap_mm;
    const cant = num("grid-cant");
    design.cant_focal_mm = cant != null ? cant : null; // blank -> null, same as the old UI's own auto-cant
  }

  // -- receiver: opticsEdits merged over the manuscript defaults, then the
  // selected layout's own panel-box values win (mirrors the old UI's own
  // tracePayload precedence: panel boxes over stale inspector edits) -----
  let optics = "axicon";
  if (typeof values.optics === "string") {
    if (OPTICS_KEYS.indexOf(values.optics) !== -1) optics = values.optics;
    else unmapped.push(`values.optics ${JSON.stringify(values.optics)} isn't a recognized optics layout`);
  }
  const edits = (old && old.opticsEdits) || {};
  const paramsByOptics = {};
  for (const key of OPTICS_KEYS) {
    paramsByOptics[key] = Object.assign({}, DEFAULT_DOC.opticsParams[key], edits[key] || {});
  }
  const towerHeight = num("tower-height");
  if (towerHeight !== undefined) {
    if (optics === "prime_focus") paramsByOptics.prime_focus.focus_height_mm = towerHeight;
    else if (optics === "axicon") paramsByOptics.axicon.apex_height_mm = towerHeight;
    else unmapped.push("tower-height (the old UI had no single tower-height box for Cassegrain)");
  }
  const halfU = num("recv-half-u");
  const halfV = num("recv-half-v");
  if (halfU !== undefined) paramsByOptics[optics].window_half_u_mm = halfU;
  if (halfV !== undefined) paramsByOptics[optics].window_half_v_mm = halfV;
  const receiver = { optics, params: paramsByOptics[optics] };

  // -- field ------------------------------------------------------------
  const traceMode = old && old.traceMode === "field" ? "field" : "single";
  const heliostatX = num("helio-x") != null ? num("helio-x") : DEFAULT_DOC.field.single.x_mm;
  const heliostatY = num("helio-y") != null ? num("helio-y") : DEFAULT_DOC.field.single.y_mm;
  let field;
  if (traceMode === "field") {
    const layout = { type: "fermat", n: Math.round(num("field-n") != null ? num("field-n") : DEFAULT_DOC.field.fermat.n) };
    const rMin = num("field-rmin");
    const rMax = num("field-rmax");
    if (rMin !== undefined) layout.r_min_m = rMin;
    if (rMax !== undefined) layout.r_max_m = rMax;
    field = { layout, heliostat_x_mm: heliostatX, heliostat_y_mm: heliostatY };
  } else {
    field = { layout: null, heliostat_x_mm: heliostatX, heliostat_y_mm: heliostatY };
  }

  // -- sun (+ site, kept even though applyProject() ignores it -- it is
  // still part of schema v1's ProjectSun, so a saved import keeps it) ----
  const az = num("sun-az") != null ? num("sun-az") : DEFAULT_DOC.sun.az;
  const el = num("sun-el") != null ? num("sun-el") : DEFAULT_DOC.sun.el;
  const sun = { azimuth_deg: az, elevation_deg: el };
  const lat = num("site-lat");
  const lon = num("site-lon");
  if (lat !== undefined && lon !== undefined) {
    const site = { latitude_deg: lat, longitude_deg: lon };
    const tz = num("site-tz");
    if (tz !== undefined) site.timezone_h = tz;
    const dateRaw = values["site-date"];
    if (typeof dateRaw === "string") {
      const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateRaw.trim());
      if (m) {
        site.year = parseInt(m[1], 10);
        site.month = parseInt(m[2], 10);
        site.day = parseInt(m[3], 10);
      }
    }
    sun.site = site;
  }
  if (values["site-hour"] !== undefined && values["site-hour"] !== "") {
    unmapped.push("site-hour (schema v1 has no home for a bare clock hour)");
  }

  // -- run ----------------------------------------------------------------
  const validModes = ["ultra_fast", "fast_accurate", "monte_carlo"];
  const runMode = old && validModes.indexOf(old.mode) !== -1 ? old.mode : "ultra_fast";
  const run = { mode: runMode };

  // -- anything in `values` this table doesn't know about ----------------
  for (const key of Object.keys(values)) {
    if (KNOWN_VALUES_KEYS.indexOf(key) === -1) unmapped.push(`values.${key} (no schema v1 home)`);
  }

  const document = { schema_version: 1, design, receiver, field, sun, run };
  return { document, unmapped };
}
