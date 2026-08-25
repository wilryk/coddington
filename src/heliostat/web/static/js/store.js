// Single source of truth for the Coddington workspace.
//
// `doc` is the design under edit -- everything a geometry/trace request is
// built from. `ui` is transient view state: which stage is expanded, the
// run bar's fidelity pick, the last trace result, error banners. Nothing
// here talks to the network or the DOM; main.js and the panels do that by
// reading/writing through get/set and reacting to subscribe().
//
// doc.field.layout picks the field's shape: "radial_stagger" (the default --
// concentric staggered rings, doc.field.radialStagger) is the parametric
// layout whose own defaults reproduce the paper's field; "fermat" is the
// golden-angle spiral alternative (doc.field.fermat); "manuscript" is that
// same paper field as a fixed dataset, served byte-exact by
// /api/field/manuscript.

function clone(x) {
  return JSON.parse(JSON.stringify(x));
}

const DEFAULT_DOC = {
  design: {
    type: "rect",
    surface: "twisting",
    // Reflectance is kept in the operator's usual unit (percent);
    // api.js's currentDesignPayload converts to the wire's 0-1 fraction.
    errors: { slope_error_mrad: 0, specularity_mrad: 0, reflectance_pct: 90 },
  },
  designParams: {
    rect: { width_mm: 5000, height_mm: 3000 },
    grid: {
      n_u: 4,
      n_v: 3,
      facet_w_mm: 1200,
      facet_h_mm: 1200,
      gap_mm: 40,
      // cant_focal_mm is aim only: null = per-heliostat slant range, 0 =
      // uncanted/parallel, >0 = one fixed focal for the whole field.
      cant_focal_mm: null,
      // facet_focal_mm is the facet's own curvature, independent of aim:
      // null = follows the canting focal (today's behaviour), 0 = truly
      // flat facets, >0 = that focal.
      facet_focal_mm: null,
    },
    // `vertices_mm` is the SKETCH the user edits -- the right half of the
    // shape when `mirror` is true (see api.js's currentDesignPayload /
    // expandCustomVertices for how mirror symmetry closes it into the
    // wire's full vertex list).
    custom: { vertices_mm: [[-2500, -1500], [2500, -1500], [2500, 1500], [-2500, 1500]], mirror: false },
  },
  optics: "axicon",
  opticsParams: {
    prime_focus: {
      focus_height_mm: 35335,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
      receiver_type: "flat",
      receiver_center_x_mm: 0,
      receiver_center_y_mm: 0,
      aperture_to_receiver_mm: 0,
      cylinder_radius_mm: 3000,
      cylinder_height_mm: 6000,
      frustum_top_radius_mm: 2500,
      frustum_bottom_radius_mm: 4000,
      frustum_height_mm: 6000,
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
    layout: "radial_stagger", // "radial_stagger" | "fermat" | "manuscript"
    single: { x_mm: 0, y_mm: -89609 },
    fermat: { n: 643, r_min_m: 30, r_max_m: 90 },
    // Three bands, innermost first -- matches the server's own
    // RadialStaggeredLayout defaults (heliostat.web.app), which is what
    // reproduces the paper's 643-heliostat field. Ring radii are not
    // exposed for editing here; see api.js's radialStaggerPayload for how
    // an edited band count reshapes them.
    radialStagger: {
      band0Rings: 3,
      band0Count: 32,
      band1Rings: 4,
      band1Count: 48,
      band2Rings: 5,
      band2Count: 71,
    },
  },
  // `az`/`el` are what every trace request carries. `mode` picks how they
  // are arrived at: "site" solves them from where and when, "direct" takes
  // them as typed. `site` is kept either way, so switching back and forth
  // does not lose the place you set.
  sun: {
    // "direct" on a fresh project so the shipped scene keeps the azimuth and
    // elevation it was built around; switching to "site" solves them from
    // where and when instead, and is the first control in the stage.
    mode: "direct",
    az: 165.2,
    el: 61.4,
    site: {
      latitude_deg: -10.0,
      longitude_deg: -52.0,
      timezone_h: -3.0,
      year: 2026,
      month: 3,
      day: 21,
      hour: 12.0,
    },
  },
};

const DEFAULT_UI = {
  // All collapsed on open: the workspace opens on the 3D scene, and an
  // expanded Field or Receiver stage would immediately swap it for that
  // stage's own plan or elevation view.
  expanded: { heliostat: false, field: false, receiver: false, sun: false },
  // "3d" | "plan" | "elevation". Not derived fresh from ui.expanded on
  // read, so a manual "back to 3D" isn't clobbered by an already-expanded
  // stage.
  view: "3d",
  fidelity: "fast_accurate",
  mcRays: null,
  geometryPending: false,
  geometryError: null, // { message, forReceiver }
  sunBelowHorizon: false,
  traceBusy: false,
  traceError: null,
  traceResult: null,
  // The fidelity the on-screen results were actually traced at.
  traceFidelity: null, // last successful trace response, plus derived fields
  // A field trace's job snapshot while ui.traceBusy (heliostat.web.jobs'
  // Job.snapshot() -- done/total/detail/eta_s/state) -- null for a
  // single-heliostat trace, which has no job behind it to poll.
  traceProgress: null,
  staleResults: false,
  fluxOverlayOpen: false,
  // In-scene selection + miss warnings.
  selection: null, // null | { kind: "heliostat" | "secondary" | "receiver" | "sun", id: number|null }
  miss: null, // /api/scene/geometry's top-level `miss` key, verbatim (or null if absent/not-yet-live)
  // `dirty` is set on the first doc.* write after a load/save; `projectName`
  // is null until a project has been saved to or loaded from the library.
  libraryOpen: false,
  libraryTab: "receivers", // "designs" | "receivers" | "projects"
  projectName: null,
  dirty: false,
  // Which full-screen tab is showing -- "workspace" | "shape" | "analysis".
  // `shapeHeliostatId` is which heliostat the Heliostat Shape tab previews;
  // null means "no explicit pick yet", so js/tabs/shape.js falls back to a
  // deterministic median-radius heliostat from the live field.
  tab: "workspace",
  shapeHeliostatId: null,
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
