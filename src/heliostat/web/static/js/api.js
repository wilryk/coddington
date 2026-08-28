// Fetch wrappers for the Coddington backend, plus the pure request-body
// builders and the debounced-geometry helper main.js drives the 3-D view
// with. No store/DOM access here -- everything is a function of its
// arguments, so it is easy to reason about what a given `doc` produces.
// (fields.js is imported for its RECEIVER_FIELD_TABLE constant only --
// currentOpticsParams filters against the same per-layout field lists the
// UI renders from; nothing here reads or writes the store.)
import { RECEIVER_FIELD_TABLE } from "./fields.js";

const API_BASE = "/api";

// Shared response handling for every wrapper below (postJSON included): a
// non-2xx response is turned into an Error carrying both a human-readable
// `message` (the server's `detail` when the body parsed as JSON, else the
// status text) and the raw `status`/`detail`, so callers (library.js's
// inline error boxes especially) can tell a 409 name collision from a 422
// validation failure without re-parsing anything themselves.
async function handleResponse(resp) {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (data && data.detail !== undefined) detail = data.detail;
    } catch (_err) {
      // body wasn't JSON -- keep statusText
    }
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    const err = new Error(message);
    err.status = resp.status;
    err.detail = detail;
    throw err;
  }
  if (resp.status === 204) return null;
  return resp.json();
}

async function postJSON(path, body, signal) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return handleResponse(resp);
}

async function getJSON(path, signal) {
  const resp = await fetch(API_BASE + path, { signal });
  return handleResponse(resp);
}

async function deleteJSON(path, signal) {
  const resp = await fetch(API_BASE + path, { method: "DELETE", signal });
  return handleResponse(resp);
}

export function postGeometry(body, signal) {
  return postJSON("/scene/geometry", body, signal);
}

export function postTrace(body, signal) {
  return postJSON("/trace", body, signal);
}

export function postFieldTrace(body, signal) {
  return postJSON("/field/trace", body, signal);
}

// A whole-field trace on a background job -- same request/response shape as
// postFieldTrace, but for fields large enough that holding the request open
// is not acceptable (docs/ui-spec.md 2.5). Same poll contract as the day
// sweep below: /result 409s while the job is still running, and again if it
// was cancelled (a field's flux is a sum across every mirror, so a partial
// one is not a valid answer to hand back).
export function postFieldTraceStart(body, signal) {
  return postJSON("/field/trace/start", body, signal);
}

export function getFieldTraceStatus(jobId) {
  return getJSON(`/field/trace/status/${encodeURIComponent(jobId)}`);
}

export function postFieldTraceCancel(jobId) {
  return postJSON(`/field/trace/cancel/${encodeURIComponent(jobId)}`, {});
}

export function getFieldTraceResult(jobId) {
  return getJSON(`/field/trace/result/${encodeURIComponent(jobId)}`);
}

export async function postFluxCsv(body) {
  const resp = await fetch(API_BASE + "/trace/flux.csv", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (data && data.detail !== undefined) detail = data.detail;
    } catch (_err) {
      // ignore
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return resp.blob();
}

// docs/ui-spec-v0.2.md §D: the ANSYS-oriented FEA CSV grid (meters, W/m²,
// one x/y/flux point per row behind commented metadata) -- a different file
// format from postFluxCsv's mm/kW-m2 labelled matrix above, not a
// replacement for it. Same request body shape (buildFluxCsvRequest), same
// single-heliostat-map-even-in-field-mode limitation that function already
// documents; only the endpoint and the resulting file differ.
export function postFluxFeaCsv(body, signal) {
  return postForBlob("/trace/flux_fea.csv", body, signal).then((resp) => resp.blob());
}

// docs/ui-spec-v0.2.md §C leftover: the secondary's own incident-flux map,
// same ANSYS-oriented FEA CSV convention as postFluxFeaCsv above (x, y,
// flux, absorbed rows behind commented metadata -- see app.py's
// _secondary_flux_fea_csv), just on the secondary's own surface instead of
// the receiver's. Same single-heliostat-map-even-in-field-mode request
// shape (the endpoint reads body.heliostat_x_mm/y_mm unconditionally, same
// as flux_fea.csv); 404s for prime_focus or a secondary with no
// single-valued flux-map parameterization.
export function postSecondaryFluxFeaCsv(body, signal) {
  return postForBlob("/trace/secondary_flux_fea.csv", body, signal).then((resp) => resp.blob());
}

// ---------------------------------------------------------------------------
// library: named designs, receiver configs and projects (docs/ui-spec.md 5)
// -- plus the read-only legacy `/api/setups` used for the Projects tab's
// import section. `name` may contain slashes (one built-in receiver is
// literally "Axicon 27 m / 20 deg / 14 m" -- docs/ui-spec.md 5's own
// naming), so it goes through encodeURIComponent like any other path
// segment; the server's `{name:path}` route decodes the resulting `%2F`
// back into a real slash, so this "just works" without special-casing.

export function getLibrary(collection) {
  return getJSON(`/library/${encodeURIComponent(collection)}`);
}

export function getLibraryEntry(collection, name) {
  return getJSON(`/library/${encodeURIComponent(collection)}/${encodeURIComponent(name)}`);
}

export function saveLibraryEntry(collection, name, document) {
  return postJSON(`/library/${encodeURIComponent(collection)}`, { name, document });
}

export function deleteLibraryEntry(collection, name) {
  return deleteJSON(`/library/${encodeURIComponent(collection)}/${encodeURIComponent(name)}`);
}

export function getSetups() {
  return getJSON("/setups");
}

export function getSetup(name) {
  return getJSON(`/setups/${encodeURIComponent(name)}`);
}

// ---------------------------------------------------------------------------
// day sweeps: a background job on the server, polled by the caller.
// /api/day/result 409s while the job is still running.

export function postDayStart(body, signal) {
  return postJSON("/day/start", body, signal);
}

export function getDayStatus(jobId) {
  return getJSON(`/day/status/${encodeURIComponent(jobId)}`);
}

export function postDayCancel(jobId) {
  return postJSON(`/day/cancel/${encodeURIComponent(jobId)}`, {});
}

export function getDayResult(jobId) {
  return getJSON(`/day/result/${encodeURIComponent(jobId)}`);
}

export function dayExportUrl(jobId) {
  return `${API_BASE}/day/export/${encodeURIComponent(jobId)}.csv`;
}

// A finished sweep's own flux map for one timestep, already rendered
// server-side -- an <img> src, not a fetch wrapper, since that is all it is
// for. 404s if that step's map was not kept (see the result row's
// `has_flux_map`) or the job has since been evicted.
export function dayFluxUrl(jobId, stepIndex) {
  return `${API_BASE}/day/flux/${encodeURIComponent(jobId)}/${stepIndex}.png`;
}

// That same stored timestep's flux map as a §D FEA CSV grid (docs/ui-spec-v0.2.md
// §D) -- built once, alongside the PNG, during the sweep itself. Only valid
// for a step the sweep kept a map for (has_flux_map), exactly like
// dayFluxUrl above; a direct download link, not a fetch wrapper, for the
// same reason dayExportUrl is.
export function dayFluxFeaCsvUrl(jobId, stepIndex) {
  return `${API_BASE}/day/flux/${encodeURIComponent(jobId)}/${stepIndex}.csv`;
}

// docs/ui-spec-v0.2.md §M.4: that same stored timestep's flux map as a
// compact JSON grid -- built once alongside the PNG/CSV during the sweep
// itself, never a re-trace (see heliostat.web.app's _day_flux_grid_blob_key).
// The Analysis tab's aperture reads this and does its own arithmetic
// against it client-side (power within a radius, avg flux, avg
// concentration, the encircled-power curve) -- a real fetch wrapper, unlike
// dayFluxUrl/dayFluxFeaCsvUrl above, because the caller needs the parsed
// numbers, not just a URL to hand to an <img>/<a>. Same has_flux_map/404
// rules as those two siblings.
export function getDayFluxGrid(jobId, stepIndex, signal) {
  return getJSON(`/day/flux/${encodeURIComponent(jobId)}/${stepIndex}.grid.json`, signal);
}

// `hour_step` is a MAXIMUM spacing: the server divides sunrise-to-sunset into
// equal intervals no larger than it, so both ends are always sampled. The
// request's own sun angles are ignored -- the sweep computes its own per step.
export function buildDayRequest(doc, ui, opts) {
  const body = buildTraceRequest(doc, ui);
  body.site = opts.site;
  body.hour_step = opts.hour_step;
  if (opts.min_elevation_deg != null) body.min_elevation_deg = opts.min_elevation_deg;
  return body;
}

// ---------------------------------------------------------------------------
// year estimate: the same background-job shape as a day sweep, one level up
// (docs/ui-spec.md 4). `site` here carries no month/day -- a year estimate
// samples dates across the whole year itself.

export function postYearStart(body, signal) {
  return postJSON("/year/start", body, signal);
}

export function getYearStatus(jobId) {
  return getJSON(`/year/status/${encodeURIComponent(jobId)}`);
}

export function postYearCancel(jobId) {
  return postJSON(`/year/cancel/${encodeURIComponent(jobId)}`, {});
}

export function getYearResult(jobId) {
  return getJSON(`/year/result/${encodeURIComponent(jobId)}`);
}

export function buildYearRequest(doc, ui, opts) {
  const body = buildTraceRequest(doc, ui);
  body.site = opts.site;
  body.fast_mode = opts.fastMode !== false;
  // Same "Timestep (h)" control the day sweep uses -- YearTraceRequest takes
  // hour_step itself (default 1.0, same as DayTraceRequest's), so without
  // this the year job silently traced at 1 h regardless of what the day
  // sweep's own field said.
  if (opts.hour_step != null) body.hour_step = opts.hour_step;
  if (opts.min_elevation_deg != null) body.min_elevation_deg = opts.min_elevation_deg;
  return body;
}

// ---------------------------------------------------------------------------
// the manuscript field: the paper's real 643-heliostat positions
// (/api/field/manuscript), fetched once at startup (main.js) and cached here
// module-level -- the same reason FIELD_MC_SEED-style caching lives on the
// server: this data never changes, so every caller in this tab (
// currentLayoutPayload, project.js's serialize/apply) reads the identical
// array rather than re-fetching or risking two slightly different copies.

let manuscriptFieldXY = null;

export function setManuscriptField(xy) {
  manuscriptFieldXY = xy;
}

export function getManuscriptField() {
  return manuscriptFieldXY;
}

export function fetchManuscriptField() {
  return getJSON("/field/manuscript");
}

// ---------------------------------------------------------------------------
// pure request-body builders -- a function of the store's `doc` (and, for
// trace, `ui.fidelity`/`ui.mcRays`) and nothing else, so the same doc always
// produces the same request.

// Phase 3c wave 1 (docs/ui-spec.md 3): "the sketch is the right half,
// mirrored to close" -- `vertices` is what the Custom sketch canvas edits
// (js/tabs/shape.js), always the u>=0 half. When `mirror` is false the
// sketch already IS the whole closed outline (an ordinary polygon) and goes
// out verbatim. When `mirror` is true, the wire polygon continues past the
// sketch's last point with the same points reflected across u=0 and walked
// back in REVERSE order -- that's what keeps the result a simple polygon
// (out along the sketch, back along its mirror) instead of a self-crossing
// one. A point already sitting on the axis (u === 0, e.g. a vertex the user
// deliberately pinned to the centerline) is skipped in the mirrored half:
// its own reflection is itself, so repeating it would double a vertex.
export function expandCustomVertices(vertices, mirror) {
  const sketch = (vertices || []).map((p) => [p[0], p[1]]);
  if (!mirror) return sketch;
  const mirrored = [];
  for (let i = sketch.length - 1; i >= 0; i--) {
    const [u, v] = sketch[i];
    if (u === 0) continue;
    mirrored.push([-u, v]);
  }
  return sketch.concat(mirrored);
}

// The optical-error fields ride flat inside a design document/payload (wire
// units: reflectance as a 0-1 fraction) while the store keeps them under
// doc.design.errors with reflectance as a percent. These two exports are
// the only place that mapping lives -- currentDesignPayload writes it
// outbound, and library.js/project.js use them to route the fields back
// into doc.design.errors when loading, instead of letting them land as
// stray keys in doc.designParams. error_map (docs/ui-spec-v0.2.md §E) rides
// the same way, verbatim (an object or null) -- there is no percent/unit
// conversion for it, unlike the others. pointing_error_mrad (§F) rides like
// slope_error_mrad/specularity_mrad -- a plain mrad number, no conversion.
export const DESIGN_ERROR_KEYS = [
  "slope_error_mrad",
  "specularity_mrad",
  "reflectance",
  "pointing_error_mrad",
  "error_map",
];

export function errorsFromDesignDocument(d) {
  return {
    slope_error_mrad: d && d.slope_error_mrad != null ? d.slope_error_mrad : 0,
    specularity_mrad: d && d.specularity_mrad != null ? d.specularity_mrad : 0,
    // Absent means the document predates the field, which the server treats
    // as a perfect mirror -- loading it must reproduce that (100%), not the
    // fresh-document 90% default, or an old project would trace 10% dimmer
    // than it used to.
    reflectance_pct: (d && d.reflectance != null ? d.reflectance : 1.0) * 100,
    // Absent means the document predates §F -- 0 mrad (a perfect tracker)
    // reproduces its old, pointing-error-free trace exactly.
    pointing_error_mrad: d && d.pointing_error_mrad != null ? d.pointing_error_mrad : 0,
    error_map: (d && d.error_map) || null,
  };
}

export function currentDesignPayload(doc) {
  const type = doc.design.type;
  const errors = doc.design.errors || {};
  // docs/ui-spec.md 3: optical errors are part of the design, on every
  // type, wire default reflectance = 1.0 -- but this client's fresh
  // DEFAULT_DOC always carries the manuscript's 90%, so `?? 90` here is
  // only a defensive fallback for a hand-built `doc` missing the field.
  const errorFields = {
    slope_error_mrad: errors.slope_error_mrad || 0,
    specularity_mrad: errors.specularity_mrad || 0,
    reflectance: (errors.reflectance_pct != null ? errors.reflectance_pct : 90) / 100,
    pointing_error_mrad: errors.pointing_error_mrad || 0,
    error_map: errors.error_map || null,
  };
  if (type === "custom") {
    const custom = doc.designParams.custom || {};
    return Object.assign(
      { type: "custom", vertices_mm: expandCustomVertices(custom.vertices_mm, !!custom.mirror) },
      errorFields,
      { surface: doc.design.surface }
    );
  }
  const params = doc.designParams[type] || {};
  return Object.assign({ type }, params, errorFields, { surface: doc.design.surface });
}

// The selected layout's optics params, filtered to that layout's own legal
// fields (fields.js's RECEIVER_FIELD_TABLE -- the same lists the sidebar,
// inspector, and elevation callouts render from). The server's pydantic
// models forbid extra keys, so a foreign field leaking into a layout's
// params (seen twice in the wild: half_angle_deg inside the cassegrain
// params, producing a user-facing 422 on every request) must never reach
// the wire. The filter is a boundary guard, not a fix for the leak itself
// -- hence the loud console.warn: if the writer ever fires again, the
// evidence names the keys while the page state is still alive to inspect.
export function currentOpticsParams(doc) {
  const optics = doc.optics;
  const params = doc.opticsParams[optics] || {};
  const legal = new Set((RECEIVER_FIELD_TABLE[optics] || []).map((f) => f.key));
  const out = {};
  const dropped = [];
  for (const [key, value] of Object.entries(params)) {
    if (legal.has(key)) out[key] = value;
    // receiver_type is a string (a segmented control, not a numberRow), kept
    // out of RECEIVER_FIELD_TABLE on purpose -- added back explicitly below
    // instead of being treated as a foreign-key leak.
    else if (optics === "prime_focus" && key === "receiver_type") continue;
    else dropped.push(key);
  }
  if (optics === "prime_focus") out.receiver_type = params.receiver_type || "flat";
  if (dropped.length) {
    console.warn(
      `optics_params for '${optics}' carried foreign key(s) [${dropped.join(", ")}] -- ` +
        "dropped at the request boundary. This means something wrote across layouts; please report."
    );
  }
  return out;
}

// Radial-staggered layout: per-band [innermost, outermost] radius bounds in
// metres, matching heliostat.web.app.RADIAL_STAGGER_RING_RADII_M's own
// default split into its three bands. Used only to reshape a band's rings
// when its ring count is edited away from the default -- the exact default
// radii (not this interpolation) are what actually reproduces the paper's
// field, and that happens server-side via RadialStaggeredLayout's own
// defaults whenever no band has been touched (see radialStaggerPayload).
const RADIAL_STAGGER_BAND_RADIUS_BOUNDS_M = [
  [30.0, 40.264159],
  [46.09511, 61.998057],
  [67.829008, 89.609429],
];

const RADIAL_STAGGER_DEFAULT_BANDS = [
  { rings: 3, count: 32 },
  { rings: 4, count: 48 },
  { rings: 5, count: 71 },
];

export function radialStaggerBands(doc) {
  const r = doc.field.radialStagger;
  return [
    { rings: r.band0Rings, count: r.band0Count },
    { rings: r.band1Rings, count: r.band1Count },
    { rings: r.band2Rings, count: r.band2Count },
  ];
}

function linspace(min, max, n) {
  if (n <= 1) return [max];
  const out = [];
  for (let i = 0; i < n; i++) out.push(min + ((max - min) * i) / (n - 1));
  return out;
}

// The radial-staggered layout payload for the current doc. Untouched bands
// (still the default 3/32, 4/48, 5/71) send a bare `{type: "radial_stagger"}`
// so the server's own model defaults apply -- byte-for-byte the field this
// app reproduces. An edited band count must also carry its own ring radii
// (the server requires one radius per ring, and its default list no longer
// has the right length), reshaped by evenly spacing that band's rings
// between its fixed radius bounds above.
export function radialStaggerPayload(doc) {
  const bands = radialStaggerBands(doc);
  const isDefault = bands.every(
    (b, i) => b.rings === RADIAL_STAGGER_DEFAULT_BANDS[i].rings && b.count === RADIAL_STAGGER_DEFAULT_BANDS[i].count
  );
  if (isDefault) return { type: "radial_stagger" };
  const band_counts = bands.map((b) => b.count);
  const band_ring_counts = bands.map((b) => b.rings);
  const ring_radii_m = bands.flatMap((b, i) => {
    const [lo, hi] = RADIAL_STAGGER_BAND_RADIUS_BOUNDS_M[i];
    return linspace(lo, hi, b.rings);
  });
  return { type: "radial_stagger", band_counts, band_ring_counts, ring_radii_m };
}

// Exported so js/project.js's serializeProject() builds the exact same
// layout shape a geometry/trace request sends -- one function, not two
// copies that could drift. doc.field.layout picks which of three shapes:
// "radial_stagger" sends the parametric staggered-ring layout above (the
// default), "manuscript" sends the paper's own positions verbatim (the
// cache fetchManuscriptField() filled at startup -- see main.js), "fermat"
// sends the parametric `{type: "fermat", n, r_min_m?, r_max_m?}` spiral. A
// manuscript request with an empty cache (the startup fetch failed) falls
// back to the fermat payload so the app still has something to draw rather
// than sending an empty positions list.
export function currentLayoutPayload(doc) {
  if (doc.field.layout === "manuscript") {
    const xy = getManuscriptField();
    if (xy && xy.length) return { type: "positions", xy_mm: xy };
  }
  if (doc.field.layout === "radial_stagger") {
    return radialStaggerPayload(doc);
  }
  const f = doc.field.fermat;
  const layout = { type: "fermat", n: f.n };
  if (f.r_min_m !== null && f.r_min_m !== undefined) layout.r_min_m = f.r_min_m;
  if (f.r_max_m !== null && f.r_max_m !== undefined) layout.r_max_m = f.r_max_m;
  return layout;
}

export function buildGeometryRequest(doc, opts) {
  const options = opts || {};
  const body = {
    design: currentDesignPayload(doc),
    optics: doc.optics,
    optics_params: currentOpticsParams(doc),
    solar_az_deg: doc.sun.az,
    solar_el_deg: doc.sun.el,
    include_corner_rays: options.includeCornerRays !== false,
  };
  if (options.maxCornerSources) body.max_corner_sources = options.maxCornerSources;
  if (doc.field.mode === "field") {
    body.layout = currentLayoutPayload(doc);
  } else {
    body.heliostat_x_mm = doc.field.single.x_mm;
    body.heliostat_y_mm = doc.field.single.y_mm;
  }
  return body;
}

export function buildTraceRequest(doc, ui) {
  const body = {
    design: currentDesignPayload(doc),
    mode: ui.fidelity,
    optics: doc.optics,
    solar_az_deg: doc.sun.az,
    solar_el_deg: doc.sun.el,
    optics_params: currentOpticsParams(doc),
    // §M.3: the 3D receiver drape's raw texture data (see app.py's
    // _flux_grid_payload) -- opt-in on the request so callers that only
    // want the rendered PNG (buildFluxCsvRequest below builds its own,
    // separate body and does not set this) never pay for it.
    include_flux_grid: true,
    // Spec §C: the secondary's own irradiance map + absorbed-heat readout
    // (mockup M9's Receiver | Secondary selector). Declared on every
    // TraceRequest-family model (heliostat.web.app's _TraceRequestBase), so
    // this is safe to always ask for -- a no-op response field for
    // prime_focus (server test: test_include_secondary_flux_is_absent_for_
    // prime_focus) and for a day/year sweep, which does not read it at all.
    // No new ray tracing either way (docs/secondary-irradiance-plan.md):
    // it reuses hits the trace already computed.
    include_secondary_flux: true,
  };
  if (ui.fidelity === "monte_carlo" && ui.mcRays) body.n_rays = ui.mcRays;
  if (doc.field.mode === "field") {
    body.layout = currentLayoutPayload(doc);
  } else {
    body.heliostat_x_mm = doc.field.single.x_mm;
    body.heliostat_y_mm = doc.field.single.y_mm;
  }
  return body;
}

// The flux-CSV endpoint (/api/trace/flux.csv) only accepts a single
// heliostat's TraceRequest, field or no -- mirroring the old UI's own
// export, which drops `layout` and exports the single-heliostat position
// even when the workspace is in field mode (its comment: "one heliostat's
// map; a field's is the sum"). Phase 3a keeps that same, honestly limited,
// behavior rather than inventing a field-summed CSV the API has no route
// for.
export function buildFluxCsvRequest(doc, ui) {
  const body = {
    design: currentDesignPayload(doc),
    mode: ui.fidelity,
    optics: doc.optics,
    solar_az_deg: doc.sun.az,
    solar_el_deg: doc.sun.el,
    optics_params: currentOpticsParams(doc),
    heliostat_x_mm: doc.field.single.x_mm,
    heliostat_y_mm: doc.field.single.y_mm,
  };
  if (ui.fidelity === "monte_carlo" && ui.mcRays) body.n_rays = ui.mcRays;
  return body;
}

// ---------------------------------------------------------------------------
// Heliostat Shape tab (docs/ui-spec.md 3, mockup M6): server-rendered
// aperture-layout preview and sag map. Both are PNGs, not JSON, so they get
// their own small blob-fetch helper rather than going through
// postJSON/handleResponse -- same non-2xx -> Error(message, {status,
// detail}) convention as handleResponse, just returning a Response instead
// of parsed JSON so callers can also read headers off it.

async function postForBlob(path, body, signal) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (data && data.detail !== undefined) detail = data.detail;
    } catch (_err) {
      // body wasn't JSON -- keep statusText
    }
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    const err = new Error(message);
    err.status = resp.status;
    err.detail = detail;
    throw err;
  }
  return resp;
}

// /api/design/preview takes just {design} -- no trace context, so no figure
// (the aperture layout is the same silhouette whatever the sun is doing).
export async function postDesignPreview(design, signal) {
  const resp = await postForBlob("/design/preview", { design }, signal);
  return resp.blob();
}

function floatHeader(resp, name) {
  const raw = resp.headers.get(name);
  if (raw == null || raw === "") return null;
  const v = parseFloat(raw);
  return Number.isFinite(v) ? v : null;
}

// /api/design/sag takes a full TraceRequest-shaped body (design + mode +
// optics + optics_params + sun + heliostat position) -- unlike the preview,
// the sag depends on the solve (which heliostat, where the sun is), see
// buildSagRequest below. Ships back the PNG plus the three headers the
// caption/warning need; a 422 (sun below horizon, unsolvable geometry)
// throws exactly like postForBlob's other callers, `err.detail` carrying
// the server's message verbatim.
export async function postDesignSag(body, signal, includeCant = true) {
  const path = includeCant ? "/design/sag" : "/design/sag?cant=false";
  const resp = await postForBlob(path, body, signal);
  const blob = await resp.blob();
  return {
    blob,
    contourIntervalMm: floatHeader(resp, "X-Contour-Interval-Mm"),
    peakToValleyMm: floatHeader(resp, "X-Peak-To-Valley-Mm"),
    slantRangeM: floatHeader(resp, "X-Slant-Range-M"),
  };
}

// docs/ui-spec-v0.2.md §D: the sag map's own "Export CSV for FEA" button --
// /api/design/sag.csv is a sibling of /api/design/sag above (same solve,
// same design, same sampling grid; only the file format differs), so it
// takes the identical body and `cant` flag.
export function postSagFeaCsv(body, signal, includeCant = true) {
  const path = includeCant ? "/design/sag.csv" : "/design/sag.csv?cant=false";
  return postForBlob(path, body, signal).then((resp) => resp.blob());
}

// docs/ui-spec-v0.2.md §E: "Measured error map -- Import CSV..." -- posts
// the raw CSV text (read client-side via FileReader.readAsText, since this
// app has no multipart upload anywhere else) and gets back what the server
// read: grid size, aperture coverage, the implied RMS slope error, and the
// raw grid itself under `grid` -- store that verbatim as
// doc.design.errors.error_map (see DESIGN_ERROR_KEYS below); it rides
// through currentDesignPayload unchanged and is exactly what a client
// hands back on the next trace/save, never rebuilt by hand.
export function postErrorMapImport(csvText, signal) {
  return postJSON("/design/errormap/import", { csv: csvText }, signal);
}

// Sibling of postErrorMapImport for a grid the client already has (loaded
// from the Library or a project) rather than a fresh CSV -- same three
// numbers back, so the chip in js/tabs/shape.js reads the same whichever
// way the map arrived.
export function postErrorMapStats(grid, signal) {
  return postJSON("/design/errormap/stats", { grid }, signal);
}

// docs/ui-spec-v0.2.md §E2: the SECONDARY's own sag map -- nominal figure +
// parametric warp + imported map, summed (heliostat.web.app._secondary_sag_grid_mm)
// -- a sibling of postDesignSag/postSagFeaCsv above, but the body is just
// {optics, optics_params}: unlike the heliostat's own sag map, the
// secondary's figure depends on none of design/sun/heliostat position, only
// the tower geometry. The map import itself reuses postErrorMapImport/
// postErrorMapStats verbatim (the endpoint is domain-agnostic -- see their
// own comments) -- a caller just stores the returned grid under
// doc.opticsParams[optics].secondary_error_map instead of
// doc.design.errors.error_map.
export async function postSecondarySag(optics, opticsParams, signal) {
  const resp = await postForBlob("/secondary/sag", { optics, optics_params: opticsParams }, signal);
  const blob = await resp.blob();
  return {
    blob,
    contourIntervalMm: floatHeader(resp, "X-Contour-Interval-Mm"),
    peakToValleyMm: floatHeader(resp, "X-Peak-To-Valley-Mm"),
  };
}

// §D "Export CSV for FEA" sibling of postSecondarySag -- same body, same
// exact surface, only the file format differs (see _secondary_sag_fea_csv).
export function postSecondarySagFeaCsv(optics, opticsParams, signal) {
  return postForBlob("/secondary/sag.csv", { optics, optics_params: opticsParams }, signal).then((resp) => resp.blob());
}

// The sag map is always for ONE named heliostat (docs/ui-spec.md 3), never
// a field -- `heliostat` is {x_mm, y_mm} for whichever one
// js/tabs/shape.js is currently previewing. mode is pinned to "ultra_fast":
// the sag map cares about the figure the solve produces, not ray-traced
// flux, so there is no reason to pay for a slower fidelity here.
export function buildSagRequest(doc, heliostat) {
  return {
    design: currentDesignPayload(doc),
    mode: "ultra_fast",
    optics: doc.optics,
    optics_params: currentOpticsParams(doc),
    solar_az_deg: doc.sun.az,
    solar_el_deg: doc.sun.el,
    heliostat_x_mm: heliostat.x_mm,
    heliostat_y_mm: heliostat.y_mm,
  };
}

// ---------------------------------------------------------------------------
// debounced geometry requester: 300 ms after the last call, aborting
// whatever request (queued or in flight) came before it.

export function createGeometryRequester(delay) {
  const wait = delay || 300;
  let timer = null;
  let controller = null;

  return function schedule(body, handlers) {
    const { onSuccess, onError } = handlers || {};
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    if (controller) {
      controller.abort();
      controller = null;
    }
    timer = setTimeout(() => {
      timer = null;
      controller = new AbortController();
      postGeometry(body, controller.signal)
        .then((data) => {
          controller = null;
          if (onSuccess) onSuccess(data);
        })
        .catch((err) => {
          controller = null;
          if (err && err.name === "AbortError") return;
          if (onError) onError(err);
        });
    }, wait);
  };
}
