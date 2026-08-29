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

// -- store -> ProjectDocument (app.ProjectDocument, schema_version 2) ------

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
  // The site rides along whether or not the sun is currently being solved
  // from it, so reopening a project restores the place and moment its
  // author was working in, not just the angles that came out.
  const sun = {
    azimuth_deg: doc.sun.az,
    elevation_deg: doc.sun.el,
    site: doc.sun.site,
    // Spec §M.7: persisted like every other site setting.
    dni: doc.sun.dni,
  };
  const run = {
    mode: ui.fidelity,
    n_rays: ui.fidelity === "monte_carlo" ? ui.mcRays : null,
  };
  // `ui.projectRuns` is the currently-loaded project's saved-run names
  // (js/tabs/analysis.js keeps it in step with the `runs` library
  // collection -- see that module's header comment). Reading it here rather
  // than taking a parameter means every save path -- this tab's own resave
  // after attaching a run, and the Library drawer's ordinary "Save
  // project", which calls this function unchanged -- carries the same list
  // forward without either one needing to know about the other.
  const runs = Array.isArray(ui.projectRuns) ? ui.projectRuns.slice() : [];
  return { schema_version: 2, design, receiver, field, sun, run, runs };
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
  } else if (field.layout && field.layout.type === "radial_stagger") {
    const bc = field.layout.band_counts;
    const brc = field.layout.band_ring_counts;
    // A bare {"type": "radial_stagger"} (no band_counts/band_ring_counts)
    // is the all-defaults layout and always loads. An explicit shape only
    // loads if it is the 3-band structure this workspace's editor knows how
    // to represent -- anything else (a different band count) has nowhere to
    // land in the sidebar.
    if (bc !== undefined || brc !== undefined) {
      if (!Array.isArray(bc) || !Array.isArray(brc) || bc.length !== 3 || brc.length !== 3) {
        return "radial-staggered layouts with other than 3 bands aren't supported by this workspace's editor";
      }
    }
  } else if (field.layout && field.layout.type !== "fermat") {
    return `field layout "${field.layout.type}" isn't supported yet (only Fermat spiral and radial staggered are)`;
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
  } else if (field.layout && field.layout.type === "radial_stagger") {
    // Validated above: either the bare default, or an explicit 3-band shape
    // this editor can represent.
    const defaults = DEFAULT_DOC.field.radialStagger;
    const bc = field.layout.band_counts;
    const brc = field.layout.band_ring_counts;
    store.set("doc.field.mode", "field");
    store.set("doc.field.layout", "radial_stagger");
    store.set("doc.field.radialStagger", {
      band0Rings: brc ? brc[0] : defaults.band0Rings,
      band0Count: bc ? bc[0] : defaults.band0Count,
      band1Rings: brc ? brc[1] : defaults.band1Rings,
      band1Count: bc ? bc[1] : defaults.band1Count,
      band2Rings: brc ? brc[2] : defaults.band2Rings,
      band2Count: bc ? bc[2] : defaults.band2Count,
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
    // Pre-existing bug fixed in passing (unrelated to spec §M.7): this
    // referenced an undefined `doc_sun_mode(sun)`, throwing on every
    // project load. ProjectSun does not persist a mode at all (only
    // azimuth_deg/elevation_deg/site/dni) -- az/el are the authoritative
    // numbers regardless of how they were arrived at (see this function's
    // own "site rides along" comment above serializeProject), so reopening
    // always lands in "direct" and shows exactly the angles the project
    // was saved at; "Site & time" is one click away if a re-solve from the
    // carried site is wanted.
    mode: "direct",
    az: sun.azimuth_deg != null ? sun.azimuth_deg : DEFAULT_DOC.sun.az,
    el: sun.elevation_deg != null ? sun.elevation_deg : DEFAULT_DOC.sun.el,
    site: Object.assign({}, DEFAULT_DOC.sun.site, sun.site || {}),
    // A document saved before spec §M.7 simply has no dni block, and falls
    // back to the same default (constant, 1000 W/m^2) that project already
    // traced at -- so it keeps reopening bit-identical.
    dni: Object.assign({}, DEFAULT_DOC.sun.dni, sun.dni || {}),
  });

  const run = (document && document.run) || {};
  store.set("ui.fidelity", run.mode || "ultra_fast");
  store.set("ui.mcRays", run.n_rays != null ? run.n_rays : null);

  // v1 documents have no `runs` field at all (they predate saved runs, not
  // "have none yet") -- either way an absent field means the same as an
  // empty one here.
  store.set("ui.projectRuns", Array.isArray(document && document.runs) ? document.runs.slice() : []);

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
