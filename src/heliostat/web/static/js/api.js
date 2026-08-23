// Fetch wrappers for the Coddington backend, plus the pure request-body
// builders and the debounced-geometry helper main.js drives the 3-D view
// with. No store/DOM access here -- everything is a function of its
// arguments, so it is easy to reason about what a given `doc` produces.

const API_BASE = "/api";

async function postJSON(path, body, signal) {
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
  return resp.json();
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
// pure request-body builders -- a function of the store's `doc` (and, for
// trace, `ui.fidelity`/`ui.mcRays`) and nothing else, so the same doc always
// produces the same request.

export function currentDesignPayload(doc) {
  const type = doc.design.type;
  const params = doc.designParams[type] || {};
  return Object.assign({ type }, params, { surface: doc.design.surface });
}

function currentLayoutPayload(doc) {
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
