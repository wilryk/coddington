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
      frustum_top_radius_mm: 4000,
      frustum_bottom_radius_mm: 2500,
      frustum_height_mm: 6000,
    },
    axicon: {
      apex_height_mm: 27000,
      half_angle_deg: 20,
      aperture_radius_mm: 14000,
      receiver_z_mm: 7000,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
      // Spec §C: fraction of secondary-incident power the secondary itself
      // reflects back out; 1 - this is the absorbed-heat readout's own
      // fraction. 0.90 matches AxiconOptics.secondary_reflectance's own
      // default (docs/secondary-irradiance-plan.md).
      secondary_reflectance: 0.9,
      // docs/ui-spec-v0.2.md §E2: rigid-body misalignment, all zero so an
      // unperturbed project traces bit-identically to before this feature
      // existed (heliostat.geometry.secondary's own default-zero promise).
      secondary_dx_mm: 0,
      secondary_dy_mm: 0,
      secondary_dz_mm: 0,
      secondary_tip_mrad: 0,
      secondary_tilt_mrad: 0,
    },
    cassegrain: {
      vertex_z_mm: 26993.999446877,
      focus_height_mm: 34892.4,
      receiver_z_mm: 7000,
      aperture_radius_mm: 14000,
      window_half_u_mm: 2000,
      window_half_v_mm: 2000,
      // See AxiconOptics's identical field above -- CassegrainOptics.
      // secondary_reflectance shares the same 0.90 default.
      secondary_reflectance: 0.9,
      // See axicon's identical fields above -- CassegrainOptics.secondary_dx_mm et al.
      secondary_dx_mm: 0,
      secondary_dy_mm: 0,
      secondary_dz_mm: 0,
      secondary_tip_mrad: 0,
      secondary_tilt_mrad: 0,
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
  // All collapsed on open (docs/ui-spec-v0.2.md §N): the app opens on the
  // 3D View tab, and the Design tab's sidebar stages no longer drive which
  // 2D view shows (that auto-morph retired -- see ui.view below).
  expanded: { heliostat: false, field: false, receiver: false, sun: false },
  // "plan" | "elevation" -- which of Design's two 2D views is showing,
  // switched only by its own explicit toggle (main.js's design-view-plan /
  // design-view-elevation buttons). 3D View has no view state of its own:
  // it always shows the 3D scene. Pre-§N this also held "3d" and was
  // coupled to ui.expanded.field/receiver (the "auto-morph" mockup M18a
  // retires); that coupling is gone.
  view: "plan",
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
  // Which top-level tab is showing -- "design" | "3dview" | "shape" |
  // "analysis" (docs/ui-spec-v0.2.md §N: the old single "workspace" tab
  // split into Design, the authoring tab, and 3D View, the observing/
  // simulating tab the app now opens on). `shapeHeliostatId` is which
  // heliostat the Heliostat Shape tab previews; null means "no explicit
  // pick yet", so js/tabs/shape.js falls back to a deterministic
  // median-radius heliostat from the live field.
  tab: "3dview",
  shapeHeliostatId: null,
  // Spec §C / mockup M9: Receiver | Secondary selector shown wherever a
  // trace flux map is on screen (run bar's flux overlay, Analysis tab's
  // timestep map) -- one shared preference rather than a copy per view, so
  // picking "Secondary" in one place is what you meant everywhere else too.
  // Only ever meaningful for axicon/cassegrain; a prime-focus doc simply
  // has nothing to show for it (see fluxSecondaryAvailable helpers).
  fluxSurface: "receiver",
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
