// Shared analysis-aperture math + canvas rendering (docs/ui-spec-v0.2.md
// §M.4, mockup M17): a draggable/resizable circle on a flat receiver's own
// flux grid, read out live (power within, avg flux, avg concentration) plus
// an encircled-power curve. Pure functions of (grid, center, radius) --
// nothing here touches the store, the DOM beyond the canvas it is handed, or
// the network; a caller supplies the grid (app.py's _flux_grid_payload
// shape: n_u/n_v/u_min_mm/u_max_mm/v_min_mm/v_max_mm/values, kW/m^2
// row-major from v_min) and a "metricsLike" object carrying whatever of
// `centroid_mm`/`rms_radius_mm` it has, for a sensible default center/radius.
//
// v0.2 followups item 3: this module exists so the Analysis tab's aperture
// (js/tabs/analysis.js, the original home of every function below) and the
// 3D View results dock's own aperture (js/panels/run.js, reading the last
// live trace's own flux_grid instead of a stored sweep step's) share the
// exact same math and painting -- refactored out here rather than
// duplicated, since a discrepancy in this arithmetic would be the kind of
// bug that is easy to ship and hard to notice. Each caller keeps its OWN
// thin, stateful pointer-drag glue (center/radius/drag-mode variables, the
// three pointer event handlers) -- that part is small and already
// per-file-duplicated elsewhere in this app (e.g. every file's own magma
// color-ramp copy), so it stays that way here too rather than forcing a
// shared stateful controller on callers with different DOM/store shapes.

// §M.4 explicitly scopes the aperture to flat receivers first ("curved
// later if wanted") -- a frustum's bin area varies with position
// (FrustumReceiver.bin_areas_m2 in geometry/receiver.py), which the
// uniform-bin-area math below does not account for.
export function receiverKindFor(doc) {
  if (doc.optics === "prime_focus") {
    return (doc.opticsParams.prime_focus || {}).receiver_type || "flat";
  }
  return "flat"; // axicon/cassegrain always target a flat receiver window
}

export function apertureReceiverIsFlat(doc) {
  return receiverKindFor(doc) === "flat";
}

export function apertureDefaultCenter(grid, metricsLike) {
  if (metricsLike && Array.isArray(metricsLike.centroid_mm) && metricsLike.centroid_mm.length === 2) {
    return { u: metricsLike.centroid_mm[0], v: metricsLike.centroid_mm[1] };
  }
  return { u: (grid.u_min_mm + grid.u_max_mm) / 2, v: (grid.v_min_mm + grid.v_max_mm) / 2 };
}

export function apertureDefaultRadiusMm(grid, metricsLike) {
  const halfU = Math.abs(grid.u_max_mm - grid.u_min_mm) / 2;
  const halfV = Math.abs(grid.v_max_mm - grid.v_min_mm) / 2;
  const cap = 0.9 * Math.min(halfU, halfV);
  // A couple of RMS spot radii is a common, physically-motivated "captures
  // most of a roughly Gaussian-like spot" default -- rms_radius_mm is
  // already computed and stored on the caller's own metrics object
  // (heliostat.web.app's _cone_metrics), so this reads it rather than
  // guessing a bare fraction of the grid.
  const rms = metricsLike && Number.isFinite(metricsLike.rms_radius_mm) ? metricsLike.rms_radius_mm : null;
  const guess = rms != null ? 2.0 * rms : cap * 0.4;
  return Math.max(1.0, Math.min(guess, cap));
}

export function clampToGridAxis(grid, value, axis) {
  const lo = axis === "u" ? grid.u_min_mm : grid.v_min_mm;
  const hi = axis === "u" ? grid.u_max_mm : grid.v_max_mm;
  return Math.max(lo, Math.min(hi, value));
}

// grid.values are §M.3's own kW/m^2 convention (heliostat.web.app's
// _flux_grid_payload, row-major, row 0 = v_min_mm -- the bottom of the map,
// same as _render_flux_png's origin="lower") -- scaled back to W/m^2 here
// so power comes out in watts. A bin counts as "inside" when its CENTER
// lies within radiusMm of the aperture center (the standard encircled-power
// discretization); average flux divides by the aperture's own ideal
// circular area (pi * r^2), not the discretized sum of included bin areas,
// matching mockup M17's own worked example.
export function apertureMetrics(grid, centerUMm, centerVMm, radiusMm) {
  const nU = grid.n_u;
  const nV = grid.n_v;
  const duMm = (grid.u_max_mm - grid.u_min_mm) / nU;
  const dvMm = (grid.v_max_mm - grid.v_min_mm) / nV;
  const binAreaM2 = (duMm / 1000) * (dvMm / 1000);
  const r2 = radiusMm * radiusMm;
  let powerW = 0;
  for (let row = 0; row < nV; row++) {
    const vMid = grid.v_min_mm + (row + 0.5) * dvMm;
    const dv = vMid - centerVMm;
    const dv2 = dv * dv;
    if (dv2 > r2) continue; // whole row is out of range -- skip its n_u work
    const rowBase = row * nU;
    for (let col = 0; col < nU; col++) {
      const uMid = grid.u_min_mm + (col + 0.5) * duMm;
      const du = uMid - centerUMm;
      if (du * du + dv2 > r2) continue;
      const kwM2 = grid.values[rowBase + col];
      if (kwM2 != null) powerW += kwM2 * 1000 * binAreaM2;
    }
  }
  const radiusM = radiusMm / 1000;
  const areaM2 = Math.PI * radiusM * radiusM;
  const avgFluxWM2 = areaM2 > 0 ? powerW / areaM2 : 0;
  return { powerW, avgFluxWM2 };
}

// Power vs. radius, mockup M17's encircled-power curve -- nSamples evenly
// spaced radii from 0 to maxRadiusMm.
export function apertureCurve(grid, centerUMm, centerVMm, maxRadiusMm, nSamples) {
  const pts = [];
  for (let i = 0; i <= nSamples; i++) {
    const r = (maxRadiusMm * i) / nSamples;
    pts.push({ r, powerW: apertureMetrics(grid, centerUMm, centerVMm, r).powerW });
  }
  return pts;
}

// -- canvas rendering + coordinate mapping -----------------------------------
// A dedicated <canvas> rather than an overlay on a server-rendered PNG (which
// carries axis labels/colorbar/title chrome no caller has exact pixel
// geometry for): painted directly from the fetched grid, at a uniform
// mm-per-pixel scale in both axes (sizeApertureCanvas) so a physical-radius
// circle is a true circle on screen, not an ellipse.
export const APERTURE_MAGMA_STOPS = [
  [0.0, 0, 0, 4],
  [0.2, 59, 15, 112],
  [0.4, 140, 41, 129],
  [0.6, 222, 73, 104],
  [0.8, 254, 159, 109],
  [1.0, 252, 253, 191],
];

export function magmaColor(t) {
  const c = Math.max(0, Math.min(1, t));
  for (let i = 1; i < APERTURE_MAGMA_STOPS.length; i++) {
    const [t0, r0, g0, b0] = APERTURE_MAGMA_STOPS[i - 1];
    const [t1, r1, g1, b1] = APERTURE_MAGMA_STOPS[i];
    if (c <= t1 || i === APERTURE_MAGMA_STOPS.length - 1) {
      const f = t1 > t0 ? (c - t0) / (t1 - t0) : 0;
      const r = Math.round(r0 + (r1 - r0) * f);
      const g = Math.round(g0 + (g1 - g0) * f);
      const b = Math.round(b0 + (b1 - b0) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(0,0,4)";
}

export function sizeApertureCanvas(canvas, grid, targetWidth) {
  const uExtent = Math.max(1e-6, grid.u_max_mm - grid.u_min_mm);
  const vExtent = Math.max(1e-6, grid.v_max_mm - grid.v_min_mm);
  const pxPerMm = targetWidth / uExtent;
  canvas.width = Math.round(targetWidth);
  canvas.height = Math.max(60, Math.round(vExtent * pxPerMm));
  return pxPerMm;
}

export function apertureDataToCanvas(grid, canvas, uMm, vMm) {
  const x = ((uMm - grid.u_min_mm) / (grid.u_max_mm - grid.u_min_mm)) * canvas.width;
  // v (north/up in the data) grows upward; canvas y grows downward -- flip,
  // matching _render_flux_png's own origin="lower".
  const y = (1 - (vMm - grid.v_min_mm) / (grid.v_max_mm - grid.v_min_mm)) * canvas.height;
  return [x, y];
}

export function apertureCanvasToData(grid, canvas, x, y) {
  const uMm = grid.u_min_mm + (x / canvas.width) * (grid.u_max_mm - grid.u_min_mm);
  const vMm = grid.v_min_mm + (1 - y / canvas.height) * (grid.v_max_mm - grid.v_min_mm);
  return [uMm, vMm];
}

export function paintApertureCanvas(canvas, grid, centerUMm, centerVMm, radiusMm, targetWidth) {
  const pxPerMm = sizeApertureCanvas(canvas, grid, targetWidth || 380);
  const ctx2d = canvas.getContext("2d");
  const nU = grid.n_u;
  const nV = grid.n_v;
  const cellW = canvas.width / nU;
  const cellH = canvas.height / nV;
  let maxKw = 0;
  for (const v of grid.values) if (v != null && v > maxKw) maxKw = v;
  for (let row = 0; row < nV; row++) {
    // canvas row 0 is the TOP of the picture; grid row 0 is v_min_mm (the
    // bottom of the map) -- flip so the picture reads right-side up.
    const canvasRow = nV - 1 - row;
    const rowBase = row * nU;
    for (let col = 0; col < nU; col++) {
      const kw = grid.values[rowBase + col];
      const t = maxKw > 0 && kw != null ? kw / maxKw : 0;
      ctx2d.fillStyle = magmaColor(t);
      ctx2d.fillRect(col * cellW, canvasRow * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  const [cx, cy] = apertureDataToCanvas(grid, canvas, centerUMm, centerVMm);
  const rPx = radiusMm * pxPerMm;
  ctx2d.save();
  ctx2d.setLineDash([6, 4]);
  ctx2d.lineWidth = 2;
  ctx2d.strokeStyle = "#ffffff";
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, rPx, 0, Math.PI * 2);
  ctx2d.stroke();
  ctx2d.lineWidth = 1;
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.stroke();
  ctx2d.restore();

  // Resize handle: a small square at the circle's east edge (mockup M17).
  const hx = cx + rPx;
  ctx2d.fillStyle = "#ffffff";
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.lineWidth = 1.3;
  ctx2d.fillRect(hx - 5, cy - 5, 10, 10);
  ctx2d.strokeRect(hx - 5, cy - 5, 10, 10);

  // Compass rider (§M) -- the aperture is scoped to flat receivers (see
  // apertureReceiverIsFlat), so this canvas only ever needs the flat-window
  // convention: u = east/west, v = north/south, no rotation.
  ctx2d.fillStyle = "rgba(255,255,255,0.85)";
  ctx2d.font = "10px sans-serif";
  ctx2d.textAlign = "center";
  ctx2d.fillText("N", canvas.width / 2, 12);
  ctx2d.fillText("S", canvas.width / 2, canvas.height - 5);
  ctx2d.textAlign = "left";
  ctx2d.fillText("W", 4, canvas.height / 2 + 3);
  ctx2d.textAlign = "right";
  ctx2d.fillText("E", canvas.width - 4, canvas.height / 2 + 3);
}

// `fmtPowerFn` is a caller-supplied formatter (e.g. tabs/analysis.js's own
// fmtPower) -- kept out of this module so it stays free of any one file's
// display-string conventions.
export function paintApertureCurve(canvas, curve, currentRadiusMm, currentPowerW, fmtPowerFn) {
  canvas.width = 380;
  canvas.height = 160;
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  const padL = 46;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = canvas.width - padL - padR;
  const plotH = canvas.height - padT - padB;
  const maxR = curve.length ? curve[curve.length - 1].r : 1;
  let maxP = 1e-9;
  for (const p of curve) if (p.powerW > maxP) maxP = p.powerW;
  const xOf = (r) => padL + (r / maxR) * plotW;
  const yOf = (p) => padT + plotH - (p / maxP) * plotH;

  ctx2d.strokeStyle = "#c7cdd6";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(padL, padT + plotH);
  ctx2d.lineTo(padL + plotW, padT + plotH);
  ctx2d.moveTo(padL, padT);
  ctx2d.lineTo(padL, padT + plotH);
  ctx2d.stroke();

  ctx2d.strokeStyle = "#45739e";
  ctx2d.lineWidth = 2;
  ctx2d.beginPath();
  curve.forEach((p, i) => {
    const x = xOf(p.r);
    const y = yOf(p.powerW);
    if (i === 0) ctx2d.moveTo(x, y);
    else ctx2d.lineTo(x, y);
  });
  ctx2d.stroke();

  const markX = xOf(currentRadiusMm);
  const markY = yOf(currentPowerW);
  ctx2d.setLineDash([4, 4]);
  ctx2d.strokeStyle = "#0b5fd0";
  ctx2d.lineWidth = 1;
  ctx2d.beginPath();
  ctx2d.moveTo(markX, padT + plotH);
  ctx2d.lineTo(markX, markY);
  ctx2d.lineTo(padL, markY);
  ctx2d.stroke();
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = "#0b5fd0";
  ctx2d.beginPath();
  ctx2d.arc(markX, markY, 3.5, 0, Math.PI * 2);
  ctx2d.fill();

  ctx2d.fillStyle = "#64748b";
  ctx2d.font = "9.5px sans-serif";
  ctx2d.textAlign = "left";
  ctx2d.fillText("0", padL - 4, padT + plotH + 14);
  ctx2d.textAlign = "right";
  ctx2d.fillText((maxR / 1000).toFixed(1) + " m", padL + plotW, padT + plotH + 14);
  ctx2d.textAlign = "left";
  const fmt = fmtPowerFn || ((w) => (w == null ? "—" : String(w)));
  ctx2d.fillText(fmt(maxP), 2, padT + 8);
}

// -- pointer-event geometry helper (shared math; the drag STATE stays with
// each caller -- see this file's header) ------------------------------------
export function apertureCanvasEventPoint(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
}
