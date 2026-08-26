// three.js viewport for the Coddington workspace.
//
// World frame matches the server exactly (src/heliostat/web/scene.py's own
// docstring): x east, y north, z up, heliostat pivot at z = 0, tower axis
// at x = y = 0. Millimetres on the wire, metres here (divide by 1000) to
// keep camera/controls numbers sane. Z-up throughout: THREE.Object3D's
// default up vector is repointed at +Z before anything is constructed, so
// OrbitControls orbits about the world's real "up" instead of Y.
//
// Mirror orientation convention -- derived, not guessed, from
// src/heliostat/trace/mc.py's `_mirror_frame`:
//
//   az = rot_az_deg (radians), el = rot_el_deg (radians)
//   n  = (cos(el)*cos(az), cos(el)*sin(az), sin(el))
//   u  = normalize(up x n)          -- up = (0, 0, 1)
//   v  = normalize(n x u)
//
// That is the exact inverse of the pointing solve's own
// `rot_el = arcsin(n_z)`, `rot_az = atan2(n_y, n_x)` map, i.e. n already
// *is* the mirror's outward normal in world space -- no compass-bearing
// conversion (unlike `_sun_vector`, which does apply one for
// `solar_az_deg`). A facet's outline is drawn flat in the mirror's own
// (u, v) plane (src/heliostat/web/scene.py: "facets are drawn flat in
// their canted planes"), so a THREE.ShapeGeometry built from
// `outline_local` in the local XY plane, then placed with a rotation
// matrix whose columns are [u, v, n], reproduces the server's own
// `helio + lu*u + lv*v` placement (see `_facet_polygons`) exactly.
//
// Selection + picking (docs/ui-spec.md 2.4) and the miss-visualization
// (docs/ui-spec.md 2.3) live in this module too, but store-free as always
// -- picks are reported through the `onSelect` callback passed into
// createScene(), and the current selection/miss set is pushed in from
// outside via setSelection()/updateGeometry(). main.js is the only module
// that reads/writes the store and also holds a reference to this scene.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const MM = 1 / 1000;

const HELIOSTAT_COLOR = 0x7ea3c8;
const SECONDARY_FILL = 0x7e9ec4;
const SECONDARY_EDGE = 0x345a80;
const RECEIVER_FILL = 0xd97b29;
const RAY_COLOR = 0xff9236;
const SUN_COLOR = 0xf0b429;
const GRID_COLOR = 0x6e8296;
// Heliostat plane above the ground, mm -- the same datum the elevation view
// dimensions against (js/views/elevation.js).
const GROUND_OFFSET_MM = 2500;
// --select (app.css) and --error-border (app.css) -- the same blue/red the
// sidebar and inspector use for selection and warning/error state, so the
// 3D view reads as the same language as the 2D chrome around it.
const SELECT_COLOR = 0x0b5fd0;
const MISS_COLOR = 0xe0554a;

// docs/ui-spec.md 2.4: a click selects, an orbit drag must not. OrbitControls
// doesn't expose "was this interaction a drag", so pointerdown/pointerup are
// tracked independently here and only counted as a click if the pointer
// moved less than this many CSS pixels within this many ms.
const CLICK_MAX_DISTANCE_PX = 5;
const CLICK_MAX_DURATION_MS = 300;

function disposeObject(obj) {
  if (!obj) return;
  obj.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
    }
  });
}

// Compass indicators (v0.2 fix wave item 3, extended per user request): quiet
// ground-plane marks at the grid's own edges -- +y is true north per this
// module's world-frame docstring above (x east, y north, z up) -- so they
// read as the 3D twin of the plan view's north arrow (js/views/plan.js's
// chromeSvg) rather than a second, disagreeing convention. Kept as faint as
// the grid they sit on (docs/ui-spec.md 2.1's "quiet" chrome): same
// GRID_COLOR, low opacity, no outline. Labels are camera-facing sprites
// (canvas textures) so they stay legible from any orbit angle without real
// 3D text geometry. North alone keeps the arrowhead -- one primary direction,
// three supporting letters, not four shouting arrows.
const compassLabelTextures = {};
function compassLabelSprite(letter) {
  if (!compassLabelTextures[letter]) {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.font = "600 42px system-ui, sans-serif";
    ctx.fillStyle = "rgba(110, 130, 150, 0.8)"; // GRID_COLOR (0x6e8296), same family as plan.js's #64748b
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(letter, 32, 34);
    compassLabelTextures[letter] = new THREE.CanvasTexture(canvas);
  }
  // A fresh material per call (disposeObject below reclaims it on rebuild),
  // but the canvas textures themselves are cached module-wide -- disposing a
  // material never disposes the texture it points at, so this is safe.
  const material = new THREE.SpriteMaterial({
    map: compassLabelTextures[letter],
    transparent: true,
    depthWrite: false,
  });
  return new THREE.Sprite(material);
}

function buildCompassMarkers(extentM) {
  const group = new THREE.Group();
  const z = -GROUND_OFFSET_MM * MM + 0.01; // a hair above the ground grid, avoids z-fighting
  const labelSize = Math.max(1.2, extentM * 0.045);

  // A small, flat, faint arrowhead pointing north (+y), lying on the ground
  // the same way the grid itself does. North only -- see the note above.
  const len = Math.max(1.5, extentM * 0.035);
  const shape = new THREE.Shape();
  shape.moveTo(0, len);
  shape.lineTo(len * 0.32, 0);
  shape.lineTo(-len * 0.32, 0);
  shape.closePath();
  const arrowMat = new THREE.MeshBasicMaterial({
    color: GRID_COLOR,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
  });
  const arrow = new THREE.Mesh(new THREE.ShapeGeometry(shape), arrowMat);
  arrow.position.set(0, extentM - len, z);
  group.add(arrow);

  // Compass convention: +y north, +x east (screen-truth for this scene), so
  // E sits at +x, S at -y, W at -x, each just past its own grid edge.
  const edges = [
    { letter: "N", x: 0, y: extentM + labelSize * 0.6 },
    { letter: "E", x: extentM + labelSize * 0.6, y: 0 },
    { letter: "S", x: 0, y: -extentM - labelSize * 0.6 },
    { letter: "W", x: -extentM - labelSize * 0.6, y: 0 },
  ];
  for (const { letter, x, y } of edges) {
    const label = compassLabelSprite(letter);
    label.scale.set(labelSize, labelSize, 1);
    label.position.set(x, y, z);
    group.add(label);
  }

  return group;
}

// -- irradiance drape (v0.2 spec §M.3, mockup M16) --------------------------
//
// After a trace, the flux map textures the receiver mesh in place of its
// plain material, until results go stale (same lifecycle as showTraceRays/
// clearTraceRays below -- main.js calls showFluxDrape/clearFluxDrape at
// exactly the same points it calls those). The texture itself is a
// THREE.CanvasTexture built client-side from the raw (downsampled) flux
// grid app.py's `/api/trace` and `/api/field/trace*` now return under
// `flux_grid` when a request opts in (see api.js's buildTraceRequest) --
// this module never touches flux_png, the rendered PNG the 2D quantitative
// panel uses instead (docs mockup M16: "3D view is orientation, panel is
// quantitative").
//
// Orientation is the whole point of this feature, so the receiver mesh's UV
// attribute is NOT left at whatever THREE's primitive geometries default to
// -- it is recomputed per vertex here, straight from
// heliostat.geometry.receiver's own contract (module docstring + each
// class's intersect()/uv_extent()):
//   * flat window: uv IS local (x, y) (FlatWindowReceiver.intersect) -- a
//     planar UV, u along local x, v along local y.
//   * cylinder/frustum: u is unrolled arc length `radius(_mean) * az`, az =
//     atan2(x, -y) (continuous-azimuth's base case; see _continuous_azimuth
//     in receiver.py) -- measured from -y (south), seam at +y (north). v is
//     height above centre (cylinder) or slant distance from the bottom rim,
//     which is just a linear remap of local z (frustum). Baking UV this way
//     -- rather than trusting THREE.CylinderGeometry's own theta-based UVs
//     -- is what makes the hot-spot orientation test below meaningful: it
//     verifies THIS mapping, not a coincidence of THREE's defaults.
function bakePlanarUV(geometry, halfUM, halfVM) {
  const pos = geometry.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    uv[i * 2] = (pos.getX(i) + halfUM) / (2 * halfUM);
    uv[i * 2 + 1] = (pos.getY(i) + halfVM) / (2 * halfVM);
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
}

function bakeCylindricalUV(geometry, vMinM, vMaxM) {
  const pos = geometry.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  const span = vMaxM - vMinM || 1;
  // atan2(x, -y): az=0 at -y (south), az=+-pi at +y (north) -- the exact
  // convention _continuous_azimuth documents. u_extent is (-piR, piR)
  // (CylinderReceiver.uv_extent), i.e. az in [-pi, pi], so normalizing
  // (az+pi)/(2*pi) lands the seam at north and u=0.5 at south -- see the
  // module-level comment above for the compass sequence this produces
  // (N . W . S . E . N left to right).
  //
  // The seam needs one extra step. CylinderGeometry's closed wrap
  // DUPLICATES the seam vertex column, and a naive per-vertex atan2 gives
  // both duplicates the same u -- so one mesh segment interpolated u all
  // the way back across the whole texture, smearing a reversed copy of the
  // map into a visible stripe (user-reported, screenshot 2026-08-26). The
  // fix: CylinderGeometry's own NATIVE uv.x already runs linearly
  // 0..1 around the wrap and is the one thing that tells the two seam
  // duplicates apart (0 vs 1). So measure once how native u maps to the
  // physical azimuth (anchor angle at native u=0, direction from the first
  // angular step) and assign u LINEARLY in native u -- continuous
  // everywhere, seam duplicates exactly one full turn apart. Sampling then
  // needs wrapS = RepeatWrapping on the drape texture (set in
  // showFluxDrape), which is also what makes the seam bins blend
  // physically -- the flux grid wraps there too (Receiver.u_period_mm).
  const nativeUv = geometry.attributes.uv;
  const azOf = (i) => Math.atan2(pos.getX(i), -pos.getY(i));
  let anchor = 0;
  let step = 1;
  for (let i = 0; i < pos.count; i++) {
    if (nativeUv.getX(i) === 0) anchor = i;
  }
  let bestFrac = Infinity;
  for (let i = 0; i < pos.count; i++) {
    const nu = nativeUv.getX(i);
    if (nu > 0 && nu < bestFrac) {
      bestFrac = nu;
      step = i;
    }
  }
  const azAnchor = azOf(anchor);
  let delta = azOf(step) - azAnchor;
  if (delta > Math.PI) delta -= 2 * Math.PI;
  if (delta < -Math.PI) delta += 2 * Math.PI;
  const dir = delta >= 0 ? 1 : -1;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    const az = azAnchor + dir * 2 * Math.PI * nativeUv.getX(i);
    uv[i * 2] = (az + Math.PI) / (2 * Math.PI);
    uv[i * 2 + 1] = (z - vMinM) / span;
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
}

// Compact approximation of matplotlib's "magma" colormap (the same one
// _render_flux_png uses for the 2D map) -- a handful of anchor stops, linearly
// interpolated, rather than the full 256-entry LUT: close enough that the 3D
// drape and the 2D panel read as the same palette (docs mockup M16's own
// gradient stops are this same approximation), without shipping matplotlib's
// table to the browser.
const MAGMA_STOPS = [
  [0.0, [0, 0, 4]],
  [0.2, [43, 17, 84]],
  [0.4, [120, 28, 109]],
  [0.6, [196, 60, 79]],
  [0.8, [251, 135, 97]],
  [1.0, [252, 253, 191]],
];
function magmaColor(t) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < MAGMA_STOPS.length; i++) {
    const [t0, c0] = MAGMA_STOPS[i - 1];
    const [t1, c1] = MAGMA_STOPS[i];
    if (x <= t1 || i === MAGMA_STOPS.length - 1) {
      const f = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return MAGMA_STOPS[MAGMA_STOPS.length - 1][1];
}

// Builds a THREE.CanvasTexture from a `flux_grid` payload (app.py's
// _flux_grid_payload): `values` is row-major, n_v rows of n_u columns, row 0
// at v_min -- exactly the UV's own v=0 (bakeCylindricalUV/bakePlanarUV above
// both put v=0 at the local-position minimum). THREE.Texture defaults to
// flipY=true (so a canvas drawn top-down displays right-side-up under
// standard bottom-left-origin UVs), which means canvas row 0 (its top, as
// putImageData addresses it) samples at V=1 and the canvas's LAST row
// samples at V=0. Since V=0 must be v_min, row 0 of `values` (v_min) is
// drawn into the canvas's last row here -- getting this backwards is
// exactly the kind of orientation bug the hot-spot test below exists to
// catch.
function fluxGridTexture(grid) {
  const { n_u, n_v, values } = grid;
  let max = 0;
  for (const v of values) if (v != null && v > max) max = v;
  const canvas = document.createElement("canvas");
  canvas.width = n_u;
  canvas.height = n_v;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(n_u, n_v);
  for (let row = 0; row < n_v; row++) {
    const canvasRow = n_v - 1 - row; // flip: values[0] is v_min, canvas y=0 is the top
    for (let col = 0; col < n_u; col++) {
      const val = values[row * n_u + col] || 0;
      const [r, g, b] = magmaColor(max > 0 ? val / max : 0);
      const idx = (canvasRow * n_u + col) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function mirrorFrame(rotAzDeg, rotElDeg) {
  const az = THREE.MathUtils.degToRad(rotAzDeg);
  const el = THREE.MathUtils.degToRad(rotElDeg);
  const n = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
  const up = new THREE.Vector3(0, 0, 1);
  const u = new THREE.Vector3().crossVectors(up, n);
  if (u.lengthSq() < 1e-12) u.set(1, 0, 0);
  else u.normalize();
  const v = new THREE.Vector3().crossVectors(n, u).normalize();
  return { n, u, v };
}

function raysToLineSegments(rays) {
  const positions = [];
  for (const poly of rays || []) {
    for (let i = 0; i < poly.length - 1; i++) {
      const a = poly[i];
      const b = poly[i + 1];
      positions.push(a[0] * MM, a[1] * MM, a[2] * MM, b[0] * MM, b[1] * MM, b[2] * MM);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: RAY_COLOR,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  });
  return new THREE.LineSegments(geometry, material);
}

// Dropped corner rays from the `miss` contract (docs/ui-spec.md 2.3):
// "rays that miss the optics ... draw dashed red rather than disappearing."
// THREE.Line's own computeLineDistances() walks the whole buffer as one
// continuous polyline, which is wrong for LineSegments' disjoint a-b pairs
// (a dash phase would leak across unrelated segments) -- so the
// `lineDistance` attribute LineDashedMaterial reads is built by hand here,
// resetting to 0 at the start of every pair instead.
function missRaysToLineSegments(rays) {
  const positions = [];
  const distances = [];
  for (const poly of rays || []) {
    for (let i = 0; i < poly.length - 1; i++) {
      const a = poly[i];
      const b = poly[i + 1];
      const ax = a[0] * MM;
      const ay = a[1] * MM;
      const az = a[2] * MM;
      const bx = b[0] * MM;
      const by = b[1] * MM;
      const bz = b[2] * MM;
      positions.push(ax, ay, az, bx, by, bz);
      const d = Math.hypot(bx - ax, by - ay, bz - az);
      distances.push(0, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("lineDistance", new THREE.Float32BufferAttribute(distances, 1));
  const material = new THREE.LineDashedMaterial({
    color: MISS_COLOR,
    dashSize: 0.6,
    gapSize: 0.4,
    transparent: true,
    opacity: 0.8,
    depthWrite: false,
  });
  return new THREE.LineSegments(geometry, material);
}

export function createScene(container, callbacks) {
  const onSelect = (callbacks && callbacks.onSelect) || null;

  const scene = new THREE.Scene();
  scene.background = null; // page-level CSS sky gradient shows through

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
  camera.up.set(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 8);
  camera.position.set(70, -110, 55);
  controls.update();

  const groundGroup = new THREE.Group();
  const fieldGroup = new THREE.Group();
  const secondaryGroup = new THREE.Group();
  const receiverGroup = new THREE.Group();
  const raysGroup = new THREE.Group();
  const missRaysGroup = new THREE.Group();
  const sunGroup = new THREE.Group();
  scene.add(groundGroup, fieldGroup, secondaryGroup, receiverGroup, raysGroup, missRaysGroup, sunGroup);

  const state = {
    heliostatMesh: null,
    heliostatIndexToId: [], // instance index -> heliostat id (rebuildHeliostats skips null-orientation heliostats, so this is NOT the same as the geometry response's own array index)
    grid: null,
    hasFit: false,
    lastFitRadius: 0,
    cornerRays: [],
    traceRays: null, // null => show corner rays; [] or [...] => show these instead
    missIds: new Set(), // heliostat ids in miss.aperture_miss_ids or miss.total_miss_ids
    missRays: [], // dashed misses for the live corner-ray view (aperture-rim probe)
    traceMissRays: [], // dashed misses for the current real trace (build_scene/build_field_scene's own miss_rays)
    selection: null, // null | {kind: "heliostat"|"secondary"|"receiver"|"sun", id}
    secondaryMesh: null,
    secondaryFillMat: null,
    secondaryEdgeMat: null,
    receiverMesh: null,
    receiverMat: null,
    receiverEdgeLines: null,
    receiverEdgeMat: null,
    receiverDraped: false, // true while a flux texture replaces the plain material (§M.3)
    receiverDrapeTexture: null,
    sunMesh: null,
    sunMat: null,
  };

  function resize() {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  }
  resize();
  // The viewport can be hidden when this runs -- the app opens on whichever
  // view the active stage wants -- and a renderer sized against a hidden
  // container stays 1x1 until something resizes the window. A
  // ResizeObserver fires again when the container gets a real box back.
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  window.addEventListener("resize", resize);

  let rafId = null;
  function tick() {
    controls.update();
    renderer.render(scene, camera);
    rafId = requestAnimationFrame(tick);
  }
  tick();

  function rebuildGround(radiusM) {
    disposeObject(groundGroup);
    groundGroup.clear();
    const extent = Math.max(20, radiusM * 2.5);
    const divisions = Math.min(60, Math.max(10, Math.round(extent / 5)));
    const grid = new THREE.GridHelper(extent * 2, divisions, GRID_COLOR, GRID_COLOR);
    grid.rotation.x = Math.PI / 2; // GridHelper is authored in the XZ plane; tip it into world XY (Z-up)
    // z = 0 is the heliostat PIVOT plane, and a mirror hangs half its height
    // below that -- a grid drawn there cuts through every heliostat in the
    // field. The ground is the pivot plane less the ground offset, which is
    // where it belongs anyway.
    grid.position.z = -GROUND_OFFSET_MM * MM;
    grid.material.transparent = true;
    grid.material.opacity = 0.12;
    groundGroup.add(grid);
    groundGroup.add(buildCompassMarkers(extent));
  }

  // Per-instance color for one heliostat id: selection wins over the miss
  // tint (docs/ui-spec.md 2.3's red-miss and 2.4's selection highlight can
  // both apply to the same heliostat -- e.g. the near-heliostat total-miss
  // case is exactly the kind of object a user would click to investigate).
  function heliostatColorFor(id) {
    const sel = state.selection;
    if (sel && sel.kind === "heliostat" && sel.id === id) return SELECT_COLOR;
    if (state.missIds.has(id)) return MISS_COLOR;
    return HELIOSTAT_COLOR;
  }

  function applyHeliostatColors() {
    const mesh = state.heliostatMesh;
    if (!mesh) return;
    const color = new THREE.Color();
    for (let i = 0; i < mesh.count; i++) {
      color.set(heliostatColorFor(state.heliostatIndexToId[i]));
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  function rebuildHeliostats(outlineLocalMm, heliostats) {
    if (state.heliostatMesh) {
      fieldGroup.remove(state.heliostatMesh);
      disposeObject(state.heliostatMesh);
      state.heliostatMesh = null;
    }
    state.heliostatIndexToId = [];
    if (!outlineLocalMm || !outlineLocalMm.length || !heliostats || !heliostats.length) return;

    const shapePoints = outlineLocalMm.map(([x, y]) => new THREE.Vector2(x * MM, y * MM));
    const shape = new THREE.Shape(shapePoints);
    const geometry = new THREE.ShapeGeometry(shape);
    // Base color is white: per-instance color (set below, every instance,
    // every rebuild) fully determines the drawn color, so it can be swapped
    // to the select/miss tint without fighting a non-white material.color.
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.96,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, heliostats.length);
    mesh.userData.pickKind = "heliostat";
    const m = new THREE.Matrix4();
    let count = 0;
    for (const h of heliostats) {
      if (h.rot_az_deg == null || h.rot_el_deg == null) continue;
      const { n, u, v } = mirrorFrame(h.rot_az_deg, h.rot_el_deg);
      m.makeBasis(u, v, n);
      m.setPosition(h.x_mm * MM, h.y_mm * MM, 0);
      mesh.setMatrixAt(count, m);
      state.heliostatIndexToId[count] = h.id;
      count++;
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    fieldGroup.add(mesh);
    state.heliostatMesh = mesh;
    applyHeliostatColors();
  }

  function rebuildSecondary(secondary) {
    disposeObject(secondaryGroup);
    secondaryGroup.clear();
    state.secondaryMesh = null;
    state.secondaryFillMat = null;
    state.secondaryEdgeMat = null;
    if (!secondary || !secondary.profile || !secondary.profile.length) return;
    const points = secondary.profile.map(([r, z]) => new THREE.Vector2(Math.max(r, 0) * MM, z * MM));
    if (points.length < 2) return;
    const lathe = new THREE.LatheGeometry(points, 48);
    // LatheGeometry revolves about its local Y axis; rotating +90 deg about
    // X puts that axis along world Z (a body of revolution has no
    // handedness to get backwards by the accompanying angular mirroring).
    const material = new THREE.MeshBasicMaterial({
      color: SECONDARY_FILL,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(lathe, material);
    mesh.rotation.x = Math.PI / 2;
    mesh.userData.pickKind = "secondary";
    secondaryGroup.add(mesh);

    const edges = new THREE.EdgesGeometry(lathe, 20);
    const edgeMat = new THREE.LineBasicMaterial({ color: SECONDARY_EDGE, transparent: true, opacity: 0.55 });
    const edgeLines = new THREE.LineSegments(edges, edgeMat);
    edgeLines.rotation.x = Math.PI / 2;
    secondaryGroup.add(edgeLines);

    state.secondaryMesh = mesh;
    state.secondaryFillMat = material;
    state.secondaryEdgeMat = edgeMat;

    // Mast: thin cylinder from the ground to the profile's lowest point.
    const lowestZ = Math.min(...points.map((p) => p.y));
    if (lowestZ > 0) {
      const mastGeom = new THREE.CylinderGeometry(0.12, 0.12, lowestZ, 12);
      mastGeom.rotateX(Math.PI / 2);
      mastGeom.translate(0, 0, lowestZ / 2);
      const mastMat = new THREE.MeshBasicMaterial({ color: 0x7b8794 });
      secondaryGroup.add(new THREE.Mesh(mastGeom, mastMat));
    }
  }

  function rebuildReceiver(receiver) {
    disposeObject(receiverGroup);
    receiverGroup.clear();
    state.receiverMesh = null;
    state.receiverMat = null;
    state.receiverEdgeLines = null;
    state.receiverEdgeMat = null;
    // A geometry rebuild always means a fresh, plain material (the drape, if
    // any, belonged to the material instance just disposed above) -- see
    // showFluxDrape/clearFluxDrape's own comment for why main.js also clears
    // the drape explicitly on the same doc-edit/fidelity-change events that
    // get here.
    state.receiverDraped = false;
    if (state.receiverDrapeTexture) {
      state.receiverDrapeTexture.dispose();
      state.receiverDrapeTexture = null;
    }
    if (!receiver) return;

    const kind = receiver.kind || "flat";
    let geometry;
    let posX = 0;
    let posY = 0;
    let posZ = 0;

    if (kind === "cylinder") {
      // CylinderReceiver: an open-ended body of revolution about its own
      // vertical axis (heliostat.geometry.receiver.CylinderReceiver -- only
      // the lateral surface absorbs, no top/bottom caps). THREE's cylinder
      // is authored Y-up; rotateX(Math.PI/2) is the same Y-up -> Z-up trick
      // the secondary's mast cylinder above uses.
      if (!(receiver.radius_mm > 0) || !(receiver.height_mm > 0)) return;
      geometry = new THREE.CylinderGeometry(receiver.radius_mm * MM, receiver.radius_mm * MM, receiver.height_mm * MM, 32, 1, true);
      geometry.rotateX(Math.PI / 2);
      bakeCylindricalUV(geometry, -receiver.height_mm * MM * 0.5, receiver.height_mm * MM * 0.5);
      posX = receiver.center_x_mm;
      posY = receiver.center_y_mm;
      posZ = receiver.center_z_mm;
    } else if (kind === "frustum") {
      // FrustumReceiver: same open-ended lateral-surface-only convention as
      // the cylinder, top/bottom radii can differ.
      const heightMm = receiver.z_top_mm - receiver.z_bot_mm;
      if (!(receiver.r_top_mm > 0) || !(receiver.r_bot_mm > 0) || !(heightMm > 0)) return;
      geometry = new THREE.CylinderGeometry(receiver.r_top_mm * MM, receiver.r_bot_mm * MM, heightMm * MM, 32, 1, true);
      geometry.rotateX(Math.PI / 2);
      // CylinderGeometry's local z (post-rotateX) runs -height/2..+height/2,
      // which IS a linear proxy for FrustumReceiver's own v (slant distance
      // from the bottom rim, receiver.py's uv_to_world docstring: `frac =
      // v/slant_length` interpolates height exactly the same way) -- so the
      // same bake used for the cylinder's v works here unchanged.
      bakeCylindricalUV(geometry, -heightMm * MM * 0.5, heightMm * MM * 0.5);
      posX = receiver.center_x_mm;
      posY = receiver.center_y_mm;
      posZ = 0.5 * (receiver.z_top_mm + receiver.z_bot_mm);
    } else {
      // FlatWindowReceiver: a horizontal window at z_mm, +-half_u_mm in x,
      // +-half_v_mm in y (heliostat.geometry.receiver.FlatWindowReceiver --
      // intersect() tests against p[2], uv_extent() returns the half-widths
      // directly as x/y bounds). THREE.PlaneGeometry is already authored in
      // the XY plane, so no rotation is needed in this Z-up scene.
      const halfUM = receiver.half_u_mm * MM;
      const halfVM = receiver.half_v_mm * MM;
      const w = halfUM * 2;
      const h = halfVM * 2;
      if (!(w > 0) || !(h > 0)) return;
      geometry = new THREE.PlaneGeometry(w, h);
      bakePlanarUV(geometry, halfUM, halfVM);
      posX = receiver.center_x_mm || 0;
      posY = receiver.center_y_mm || 0;
      posZ = receiver.z_mm;
    }

    const material = new THREE.MeshBasicMaterial({
      color: RECEIVER_FILL,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(posX * MM, posY * MM, posZ * MM);
    mesh.userData.pickKind = "receiver";
    receiverGroup.add(mesh);

    // Selection outline (docs/ui-spec.md 2.4): hidden by default, shown in
    // --select blue only while the receiver is the current selection.
    const edges = new THREE.EdgesGeometry(geometry);
    const edgeMat = new THREE.LineBasicMaterial({ color: SELECT_COLOR, transparent: true, opacity: 0.9 });
    const edgeLines = new THREE.LineSegments(edges, edgeMat);
    edgeLines.position.copy(mesh.position);
    edgeLines.visible = false;
    receiverGroup.add(edgeLines);

    state.receiverMesh = mesh;
    state.receiverMat = material;
    state.receiverKind = kind; // showFluxDrape: curved kinds wrap their drape texture
    state.receiverEdgeLines = edgeLines;
    state.receiverEdgeMat = edgeMat;
  }

  function rebuildSun(sunUnit) {
    disposeObject(sunGroup);
    sunGroup.clear();
    state.sunMesh = null;
    state.sunMat = null;
    if (!sunUnit || sunUnit.length !== 3) return;
    const dir = new THREE.Vector3(sunUnit[0], sunUnit[1], sunUnit[2]);
    if (dir.lengthSq() < 1e-9) return;
    dir.normalize();
    const dist = Math.max(60, state.lastFitRadius * 1.6 || 100);
    const pos = dir.clone().multiplyScalar(dist);
    const sphereMat = new THREE.MeshBasicMaterial({ color: SUN_COLOR });
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(dist * 0.03, 12, 12), sphereMat);
    sphere.position.copy(pos);
    sphere.userData.pickKind = "sun";
    sunGroup.add(sphere);
    const arrow = new THREE.ArrowHelper(dir.clone().negate(), pos, dist * 0.35, SUN_COLOR, dist * 0.05, dist * 0.03);
    sunGroup.add(arrow);

    state.sunMesh = sphere;
    state.sunMat = sphereMat;
  }

  function renderRays() {
    disposeObject(raysGroup);
    raysGroup.clear();
    const rays = state.traceRays !== null ? state.traceRays : state.cornerRays;
    if (rays && rays.length) raysGroup.add(raysToLineSegments(rays));

    // Dropped miss rays ride alongside whichever ray set is showing:
    // the live corner-ray view's own aperture-rim probe (docs/ui-spec.md
    // 2.3) while state.traceRays is null, or a real trace's own
    // build_scene/build_field_scene misses once one lands (2.1: "rays that
    // miss the optics ... draw dashed red rather than disappearing" --
    // curved-receiver misses included, not just the aperture rim). Either
    // way this never mixes the two: the corner-ray probe's misses are
    // computed against a different (enlarged-aperture) heuristic and would
    // mislabel a real trace's own rays.
    disposeObject(missRaysGroup);
    missRaysGroup.clear();
    const missRays = state.traceRays !== null ? state.traceMissRays : state.missRays;
    if (missRays && missRays.length) {
      missRaysGroup.add(missRaysToLineSegments(missRays));
    }
  }

  function fieldRadiusM(heliostats) {
    let r = 0;
    for (const h of heliostats || []) {
      const d = Math.hypot(h.x_mm, h.y_mm) * MM;
      if (d > r) r = d;
    }
    return r;
  }

  function maybeFitCamera(radiusM) {
    const r = Math.max(radiusM, 10);
    if (!state.hasFit) {
      const dist = r * 1.9;
      camera.position.set(dist * 0.55, -dist * 0.85, dist * 0.5);
      controls.target.set(0, 0, Math.min(r * 0.15, 15));
      controls.update();
      state.hasFit = true;
      state.lastFitRadius = r;
      return;
    }
    // "Never re-fit after first load unless field radius changes
    // drastically" (docs/ui-spec.md 2.1) -- a >2x growth or <0.5x shrink.
    if (r > state.lastFitRadius * 2 || r < state.lastFitRadius * 0.5) {
      const dist = r * 1.9;
      camera.position.set(dist * 0.55, -dist * 0.85, dist * 0.5);
      controls.target.set(0, 0, Math.min(r * 0.15, 15));
      controls.update();
      state.lastFitRadius = r;
    }
  }

  // Re-applies the current selection's highlight to whatever objects exist
  // right now -- needed both after setSelection() and after a geometry
  // rebuild recreates the secondary/receiver/sun materials from scratch.
  function applySelectionVisuals() {
    const sel = state.selection;

    if (state.secondaryFillMat && state.secondaryEdgeMat) {
      const isSel = !!sel && sel.kind === "secondary";
      state.secondaryFillMat.opacity = isSel ? 0.5 : 0.3;
      state.secondaryEdgeMat.color.set(isSel ? SELECT_COLOR : SECONDARY_EDGE);
      state.secondaryEdgeMat.opacity = isSel ? 0.9 : 0.55;
    }

    if (state.receiverMat && state.receiverEdgeLines) {
      const isSel = !!sel && sel.kind === "receiver";
      // Draped (§M.3): the texture needs to read clearly, so it sits
      // noticeably more opaque than the translucent plain material -- still
      // nudged up a touch further on selection, same as the plain case.
      state.receiverMat.opacity = state.receiverDraped ? (isSel ? 0.98 : 0.92) : isSel ? 0.8 : 0.55;
      state.receiverEdgeLines.visible = isSel;
    }

    if (state.sunMat) {
      const isSel = !!sel && sel.kind === "sun";
      state.sunMat.color.set(isSel ? SELECT_COLOR : SUN_COLOR);
    }

    applyHeliostatColors();
  }

  // -- picking: click vs. orbit-drag (docs/ui-spec.md 2.4) -------------------

  const raycaster = new THREE.Raycaster();
  let pointerDown = null;

  function pointerNDC(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
      y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
    };
  }

  function pick(e) {
    if (!onSelect) return;
    const ndc = pointerNDC(e);
    raycaster.setFromCamera(ndc, camera);
    const objects = [state.heliostatMesh, state.secondaryMesh, state.receiverMesh, state.sunMesh].filter(Boolean);
    const hits = objects.length ? raycaster.intersectObjects(objects, false) : [];
    if (!hits.length) {
      onSelect(null); // clicking empty space deselects
      return;
    }
    const hit = hits[0];
    const kind = hit.object.userData.pickKind;
    if (kind === "heliostat") {
      const id = state.heliostatIndexToId[hit.instanceId];
      if (id == null) return;
      onSelect({ kind: "heliostat", id });
    } else if (kind) {
      onSelect({ kind, id: null });
    }
  }

  function handlePointerDown(e) {
    if (e.button !== 0) return;
    pointerDown = { x: e.clientX, y: e.clientY, t: performance.now() };
  }

  function handlePointerUp(e) {
    if (e.button !== 0 || !pointerDown) return;
    const dx = e.clientX - pointerDown.x;
    const dy = e.clientY - pointerDown.y;
    const dt = performance.now() - pointerDown.t;
    pointerDown = null;
    if (Math.hypot(dx, dy) > CLICK_MAX_DISTANCE_PX || dt > CLICK_MAX_DURATION_MS) return;
    pick(e);
  }

  renderer.domElement.addEventListener("pointerdown", handlePointerDown);
  renderer.domElement.addEventListener("pointerup", handlePointerUp);

  return {
    camera,
    controls,

    // Applies a valid /api/scene/geometry response. Never called for a
    // sun-below-horizon response (main.js keeps the last valid frame
    // instead, per docs/ui-spec.md 2.1).
    updateGeometry(resp) {
      const heliostats = resp.heliostats || [];
      const radius = fieldRadiusM(heliostats);
      maybeFitCamera(radius);
      rebuildGround(Math.max(radius, state.lastFitRadius));
      rebuildHeliostats(resp.outline_local, heliostats);
      rebuildSecondary(resp.secondary);
      rebuildReceiver(resp.receiver);
      rebuildSun(resp.sun);
      state.cornerRays = resp.rays || [];

      // `miss` (docs/ui-spec.md 2.3) may not be live on the backend yet --
      // undefined/null reads as "nothing misses", same as empty lists.
      const miss = resp.miss || null;
      const apertureIds = (miss && miss.aperture_miss_ids) || [];
      const totalIds = (miss && miss.total_miss_ids) || [];
      state.missIds = new Set([...apertureIds, ...totalIds]);
      state.missRays = (miss && miss.rays) || [];

      renderRays();
      applySelectionVisuals(); // materials were just rebuilt from scratch
    },

    // Corner and miss rays from a second, ray-bearing geometry response,
    // without touching the meshes the first one already built.
    updateRays(resp) {
      state.cornerRays = resp.rays || [];
      const miss = resp.miss || null;
      const apertureIds = (miss && miss.aperture_miss_ids) || [];
      const totalIds = (miss && miss.total_miss_ids) || [];
      state.missIds = new Set([...apertureIds, ...totalIds]);
      state.missRays = (miss && miss.rays) || [];
      renderRays();
      applySelectionVisuals(); // miss ids recolour heliostats
    },

    // A real trace's own rays (response.scene.rays) and its own miss rays
    // (response.scene.miss_rays -- rays that reflected off their mirror,
    // and secondary if any, but never reached the receiver), shown in place
    // of the live corner rays (and their own miss rays) until the next edit
    // makes results stale.
    showTraceRays(rays, missRays) {
      state.traceRays = rays || [];
      state.traceMissRays = missRays || [];
      renderRays();
    },

    // Falls back to whatever corner (and miss) rays the last geometry
    // response drew.
    clearTraceRays() {
      state.traceRays = null;
      state.traceMissRays = [];
      renderRays();
    },

    // §M.3: textures the receiver mesh with a trace's flux map (`flux_grid`
    // on the trace response, opt-in via api.js's buildTraceRequest -- see
    // app.py's _flux_grid_payload), replacing the plain material. No-op if
    // the current geometry has no receiver mesh (e.g. an aperture-clipped
    // shape rebuildReceiver declined to build one) -- there is nothing to
    // drape onto, same as a trace with no receiver at all.
    showFluxDrape(fluxGrid) {
      const mat = state.receiverMat;
      if (!mat || !fluxGrid || !fluxGrid.values || !fluxGrid.values.length) return;
      if (state.receiverDrapeTexture) state.receiverDrapeTexture.dispose();
      const texture = fluxGridTexture(fluxGrid);
      // Curved receivers wrap: the baked u runs linearly across the seam
      // (bakeCylindricalUV), landing the duplicate seam column exactly one
      // turn past its twin, so sampling must wrap too -- and the flux grid
      // is physically periodic there (Receiver.u_period_mm), making
      // repeat-sampling the correct blend, not just a cosmetic one. Flat
      // windows keep the clamped default.
      if (state.receiverKind === "cylinder" || state.receiverKind === "frustum") {
        texture.wrapS = THREE.RepeatWrapping;
        texture.needsUpdate = true;
      }
      state.receiverDrapeTexture = texture;
      mat.map = texture;
      mat.color.set(0xffffff); // let the texture's own colors show, untinted
      mat.needsUpdate = true;
      state.receiverDraped = true;
      applySelectionVisuals(); // sets the draped opacity
    },

    // Reverts the receiver to its plain material -- called wherever
    // clearTraceRays() is (main.js: doc edits, fidelity changes, a fresh
    // trace superseding a stale one), matching this feature's own lifecycle
    // rule (spec §M.3: "replacing the plain material until results go
    // stale"). Also happens implicitly on the next rebuildReceiver (a new
    // mesh/material always starts plain), so this covers the "stale but the
    // mesh hasn't rebuilt yet" window explicitly.
    clearFluxDrape() {
      const mat = state.receiverMat;
      if (state.receiverDrapeTexture) {
        state.receiverDrapeTexture.dispose();
        state.receiverDrapeTexture = null;
      }
      state.receiverDraped = false;
      if (mat) {
        mat.map = null;
        mat.color.set(RECEIVER_FILL);
        mat.needsUpdate = true;
      }
      applySelectionVisuals();
    },

    // Sun below the horizon: rays disappear, everything else holds its
    // last pose (docs/ui-spec.md 2.1).
    clearAllRays() {
      state.cornerRays = [];
      state.traceRays = null;
      state.missRays = [];
      state.traceMissRays = [];
      renderRays();
    },

    // Applies the current in-scene selection (docs/ui-spec.md 2.4):
    // sel is null or {kind: "heliostat"|"secondary"|"receiver"|"sun", id}.
    // main.js calls this whenever store's ui.selection changes.
    setSelection(sel) {
      state.selection = sel;
      applySelectionVisuals();
    },

    dispose() {
      cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      controls.dispose();
      disposeObject(groundGroup);
      disposeObject(fieldGroup);
      disposeObject(secondaryGroup);
      disposeObject(receiverGroup);
      disposeObject(raysGroup);
      disposeObject(missRaysGroup);
      disposeObject(sunGroup);
      if (state.receiverDrapeTexture) state.receiverDrapeTexture.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    },
  };
}
