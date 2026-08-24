// Fetch wrappers for the Coddington backend, plus the pure request-body
// builders and the debounced-geometry helper main.js drives the 3-D view
// with. No store/DOM access here -- everything is a function of its
// arguments, so it is easy to reason about what a given `doc` produces.

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

export function currentDesignPayload(doc) {
  const type = doc.design.type;
  const params = doc.designParams[type] || {};
  return Object.assign({ type }, params, { surface: doc.design.surface });
}

// Exported so js/project.js's serializeProject() builds the exact same
// layout shape a geometry/trace request sends -- one function, not two
// copies that could drift. doc.field.layout picks which of two shapes:
// "manuscript" sends the paper's own positions verbatim (the cache
// fetchManuscriptField() filled at startup -- see main.js), "fermat" sends
// the parametric `{type: "fermat", n, r_min_m?, r_max_m?}` spiral it always
// did. A manuscript request with an empty cache (the startup fetch failed)
// falls back to the fermat payload so the app still has something to draw
// rather than sending an empty positions list.
export function currentLayoutPayload(doc) {
  if (doc.field.layout === "manuscript") {
    const xy = getManuscriptField();
    if (xy && xy.length) return { type: "positions", xy_mm: xy };
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
    optics_params: doc.opticsParams[doc.optics],
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
    optics_params: doc.opticsParams[doc.optics],
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
    optics_params: doc.opticsParams[doc.optics],
    heliostat_x_mm: doc.field.single.x_mm,
    heliostat_y_mm: doc.field.single.y_mm,
  };
  if (ui.fidelity === "monte_carlo" && ui.mcRays) body.n_rays = ui.mcRays;
  return body;
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
