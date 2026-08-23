// Single source of truth for the Coddington workspace (Phase 3a).
//
// `doc` is the design under edit -- everything a geometry/trace request is
// built from. `ui` is transient view state: which stage is expanded, the
// run bar's fidelity pick, the last trace result, error banners. Nothing
// here talks to the network or the DOM; main.js and the panels do that by
// reading/writing through get/set and reacting to subscribe().
//
// docs/ui-spec.md 2.2's per-layout defaults (the manuscript baseline):
// rect 5000x3000 twisting, axicon selected (apex 27000, half angle 20,
// aperture 14000, receiver 7000, window 2000/2000), field mode "field"
// with a 643-heliostat Fermat spiral 30-90 m, sun az 165.2 / el 61.4.
// prime_focus and cassegrain keep their own manuscript numbers too, so
// switching optics layout shows real geometry rather than a blank slate
// (spec 2.2, "each layout keeps its own last-used numbers").

function clone(x) {
  return JSON.parse(JSON.stringify(x));
}

const DEFAULT_DOC = {
  design: {
    type: "rect",
    surface: "twisting",
  },
  designParams: {
    rect: { width_mm: 5000, height_mm: 3000 },
    grid: {
      n_u: 4,
      n_v: 3,
      facet_w_mm: 1200,
      facet_h_mm: 1200,
      gap_mm: 40,
      cant_focal_mm: null,
    },
  },
  optics: "axicon",
  opticsParams: {
    prime_focus: {
      focus_height_mm: 35335,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
    },
    axicon: {
      apex_height_mm: 27000,
      half_angle_deg: 20,
      aperture_radius_mm: 14000,
      receiver_z_mm: 7000,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
    },
    cassegrain: {
      vertex_z_mm: 26993.999446877,
      focus_height_mm: 34892.4,
      receiver_z_mm: 7000,
      aperture_radius_mm: 14000,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
    },
  },
  field: {
    mode: "field",
    single: { x_mm: 0, y_mm: -89609 },
    fermat: { n: 643, r_min_m: 30, r_max_m: 90 },
  },
  sun: { az: 165.2, el: 61.4 },
};

const DEFAULT_UI = {
  expanded: { heliostat: true, field: true, receiver: true, sun: true },
  fidelity: "fast_accurate",
  mcRays: null,
  geometryPending: false,
  geometryError: null, // { message, forReceiver }
  sunBelowHorizon: false,
  traceBusy: false,
  traceError: null,
  traceResult: null, // last successful trace response, plus derived fields
  staleResults: false,
  fluxOverlayOpen: false,
};

function createStore() {
  const state = { doc: clone(DEFAULT_DOC), ui: clone(DEFAULT_UI) };
  const subscribers = new Set();

  function get(path) {
    if (!path) return state;
    return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), state);
  }

  function set(path, value) {
    const parts = path.split(".");
    const last = parts.pop();
    let obj = state;
    for (const p of parts) {
      if (obj[p] == null || typeof obj[p] !== "object") obj[p] = {};
      obj = obj[p];
    }
    obj[last] = value;
    for (const fn of subscribers) fn(path, value);
  }

  function subscribe(fn) {
    subscribers.add(fn);
    return () => subscribers.delete(fn);
  }

  return { get, set, subscribe };
}

export const store = createStore();
export { DEFAULT_DOC, DEFAULT_UI };
