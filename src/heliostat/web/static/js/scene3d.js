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
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const MM = 1 / 1000;

const HELIOSTAT_COLOR = 0xc4daee;
const SECONDARY_FILL = 0x7e9ec4;
const SECONDARY_EDGE = 0x345a80;
const RECEIVER_FILL = 0xd97b29;
const RAY_COLOR = 0xff9236;
const SUN_COLOR = 0xf0b429;
const GRID_COLOR = 0x6e8296;

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

export function createScene(container) {
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
  const sunGroup = new THREE.Group();
  scene.add(groundGroup, fieldGroup, secondaryGroup, receiverGroup, raysGroup, sunGroup);

  const state = {
    heliostatMesh: null,
    grid: null,
    hasFit: false,
    lastFitRadius: 0,
    cornerRays: [],
    traceRays: null, // null => show corner rays; [] or [...] => show these instead
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
    grid.material.transparent = true;
    grid.material.opacity = 0.12;
    groundGroup.add(grid);
  }

  function rebuildHeliostats(outlineLocalMm, heliostats) {
    if (state.heliostatMesh) {
      fieldGroup.remove(state.heliostatMesh);
      disposeObject(state.heliostatMesh);
      state.heliostatMesh = null;
    }
    if (!outlineLocalMm || !outlineLocalMm.length || !heliostats || !heliostats.length) return;

    const shapePoints = outlineLocalMm.map(([x, y]) => new THREE.Vector2(x * MM, y * MM));
    const shape = new THREE.Shape(shapePoints);
    const geometry = new THREE.ShapeGeometry(shape);
    const material = new THREE.MeshBasicMaterial({
      color: HELIOSTAT_COLOR,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.96,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, heliostats.length);
    const m = new THREE.Matrix4();
    let count = 0;
    for (const h of heliostats) {
      if (h.rot_az_deg == null || h.rot_el_deg == null) continue;
      const { n, u, v } = mirrorFrame(h.rot_az_deg, h.rot_el_deg);
      m.makeBasis(u, v, n);
      m.setPosition(h.x_mm * MM, h.y_mm * MM, 0);
      mesh.setMatrixAt(count, m);
      count++;
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    fieldGroup.add(mesh);
    state.heliostatMesh = mesh;
  }

  function rebuildSecondary(secondary) {
    disposeObject(secondaryGroup);
    secondaryGroup.clear();
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
    secondaryGroup.add(mesh);

    const edges = new THREE.EdgesGeometry(lathe, 20);
    const edgeMat = new THREE.LineBasicMaterial({ color: SECONDARY_EDGE, transparent: true, opacity: 0.55 });
    const edgeLines = new THREE.LineSegments(edges, edgeMat);
    edgeLines.rotation.x = Math.PI / 2;
    secondaryGroup.add(edgeLines);

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
    if (!receiver) return;
    // FlatWindowReceiver: a horizontal window at z_mm, +-half_u_mm in x,
    // +-half_v_mm in y (heliostat.geometry.receiver.FlatWindowReceiver --
    // intersect() tests against p[2], uv_extent() returns the half-widths
    // directly as x/y bounds). THREE.PlaneGeometry is already authored in
    // the XY plane, so no rotation is needed in this Z-up scene.
    const w = receiver.half_u_mm * 2 * MM;
    const h = receiver.half_v_mm * 2 * MM;
    if (!(w > 0) || !(h > 0)) return;
    const geometry = new THREE.PlaneGeometry(w, h);
    const material = new THREE.MeshBasicMaterial({
      color: RECEIVER_FILL,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, receiver.z_mm * MM);
    receiverGroup.add(mesh);
  }

  function rebuildSun(sunUnit) {
    disposeObject(sunGroup);
    sunGroup.clear();
    if (!sunUnit || sunUnit.length !== 3) return;
    const dir = new THREE.Vector3(sunUnit[0], sunUnit[1], sunUnit[2]);
    if (dir.lengthSq() < 1e-9) return;
    dir.normalize();
    const dist = Math.max(60, state.lastFitRadius * 1.6 || 100);
    const pos = dir.clone().multiplyScalar(dist);
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(dist * 0.03, 12, 12),
      new THREE.MeshBasicMaterial({ color: SUN_COLOR })
    );
    sphere.position.copy(pos);
    sunGroup.add(sphere);
    const arrow = new THREE.ArrowHelper(dir.clone().negate(), pos, dist * 0.35, SUN_COLOR, dist * 0.05, dist * 0.03);
    sunGroup.add(arrow);
  }

  function renderRays() {
    disposeObject(raysGroup);
    raysGroup.clear();
    const rays = state.traceRays !== null ? state.traceRays : state.cornerRays;
    if (rays && rays.length) raysGroup.add(raysToLineSegments(rays));
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
      renderRays();
    },

    // A real trace's own rays (response.scene.rays), shown in place of the
    // live corner rays until the next edit makes results stale.
    showTraceRays(rays) {
      state.traceRays = rays || [];
      renderRays();
    },

    // Falls back to whatever corner rays the last geometry response drew.
    clearTraceRays() {
      state.traceRays = null;
      renderRays();
    },

    // Sun below the horizon: rays disappear, everything else holds its
    // last pose (docs/ui-spec.md 2.1).
    clearAllRays() {
      state.cornerRays = [];
      state.traceRays = null;
      renderRays();
    },

    dispose() {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
      controls.dispose();
      disposeObject(groundGroup);
      disposeObject(fieldGroup);
      disposeObject(secondaryGroup);
      disposeObject(receiverGroup);
      disposeObject(raysGroup);
      disposeObject(sunGroup);
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    },
  };
}
