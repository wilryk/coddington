// Sun stage: where and when, or a typed azimuth and elevation.
//
// A trace only ever carries an azimuth and an elevation, so those stay the
// stored values and the site solves into them through /api/sun (the same
// NOAA calculator the sweeps use). Field descriptors live in ../fields.js so
// the floating inspector can render the identical rows when the sun is
// selected in the scene.
import { store } from "../store.js";
import { numberRow, setVal, segButton, SUN_FIELDS, SUN_SITE_FIELDS } from "../fields.js";

let built = false;
let els = {};

// Solved from the last site request: az/el are pushed into the store, the
// rest is read-only context for the panel.
let solved = null;
let solveError = null;
let solveTimer = null;
let solveController = null;

function siteBody(doc) {
  const s = doc.sun.site;
  return {
    latitude_deg: s.latitude_deg,
    longitude_deg: s.longitude_deg,
    timezone_h: s.timezone_h,
    year: s.year,
    month: s.month,
    day: s.day,
    hour: s.hour,
  };
}

// Debounced so dragging a time or latitude does not fire a request per
// keystroke; aborted so only the latest answer is applied.
function scheduleSolve() {
  if (solveTimer) clearTimeout(solveTimer);
  if (solveController) solveController.abort();
  solveTimer = setTimeout(() => {
    solveTimer = null;
    const doc = store.get("doc");
    if (doc.sun.mode !== "site") return;
    solveController = new AbortController();
    fetch("/api/sun", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(siteBody(doc)),
      signal: solveController.signal,
    })
      .then((r) => r.json().then((data) => ({ ok: r.ok, data })))
      .then(({ ok, data }) => {
        solveController = null;
        if (!ok) {
          solveError = typeof data.detail === "string" ? data.detail : "Could not solve the sun position.";
          solved = null;
          store.set("ui.sunSolveNonce", (store.get("ui.sunSolveNonce") || 0) + 1);
          return;
        }
        solveError = null;
        solved = data;
        // Writing az/el is what actually moves the scene.
        store.set("doc.sun.az", Math.round(data.solar_az_deg * 100) / 100);
        store.set("doc.sun.el", Math.round(data.solar_el_deg * 100) / 100);
      })
      .catch((err) => {
        solveController = null;
        if (err && err.name === "AbortError") return;
        solveError = "Could not reach the sun calculator.";
        solved = null;
      });
  }, 250);
}

function dateRow(parent) {
  const row = document.createElement("div");
  row.className = "frow";
  const label = document.createElement("label");
  label.textContent = "Date";
  const input = document.createElement("input");
  input.type = "date";
  input.className = "val";
  input.addEventListener("input", () => {
    const parts = (input.value || "").split("-").map(Number);
    if (parts.length !== 3 || parts.some((p) => !Number.isFinite(p))) return;
    store.set("doc.sun.site.year", parts[0]);
    store.set("doc.sun.site.month", parts[1]);
    store.set("doc.sun.site.day", parts[2]);
  });
  row.appendChild(label);
  row.appendChild(input);
  parent.appendChild(row);
  return input;
}

function build(container) {
  container.innerHTML = "";
  container.className = "stage";

  const head = document.createElement("div");
  head.className = "stagehead";
  const chev = document.createElement("span");
  chev.className = "chev";
  chev.textContent = "▾";
  const h2 = document.createElement("h2");
  h2.textContent = "Sun";
  head.appendChild(chev);
  head.appendChild(h2);
  head.addEventListener("click", () => {
    const ui = store.get("ui");
    store.set("ui.expanded.sun", !ui.expanded.sun);
  });

  const body = document.createElement("div");
  body.className = "stagebody";

  const modeSeg = document.createElement("div");
  modeSeg.className = "seg";
  const modeBtns = {
    site: segButton(modeSeg, "Site & time", true, () => {
      store.set("doc.sun.mode", "site");
      scheduleSolve();
    }),
    direct: segButton(modeSeg, "Azimuth & elevation", false, () => store.set("doc.sun.mode", "direct")),
  };
  body.appendChild(modeSeg);

  const siteFields = document.createElement("div");
  const lat = numberRow(siteFields, SUN_SITE_FIELDS[0]);
  const lon = numberRow(siteFields, SUN_SITE_FIELDS[1]);
  const tz = numberRow(siteFields, SUN_SITE_FIELDS[2]);
  const date = dateRow(siteFields);
  const hour = numberRow(siteFields, SUN_SITE_FIELDS[3]);
  body.appendChild(siteFields);

  const solvedLine = document.createElement("div");
  solvedLine.className = "summary";
  siteFields.appendChild(solvedLine);

  const siteErr = document.createElement("div");
  siteErr.className = "fielderr";
  siteErr.hidden = true;
  siteFields.appendChild(siteErr);

  const directFields = document.createElement("div");
  const az = numberRow(directFields, SUN_FIELDS[0]);
  const el = numberRow(directFields, SUN_FIELDS[1]);
  body.appendChild(directFields);

  // -- spec §M.7: site DNI -- independent of how az/el were arrived at
  // (site & time, or a typed angle pair), so this shows either way rather
  // than living inside siteFields/directFields above.
  const dniHead = document.createElement("div");
  dniHead.className = "subhead";
  dniHead.textContent = "Sun intensity (DNI)";
  body.appendChild(dniHead);

  const dniSeg = document.createElement("div");
  dniSeg.className = "seg";
  const dniBtns = {
    constant: segButton(
      dniSeg,
      "Constant",
      true,
      () => store.set("doc.sun.dni.mode", "constant"),
      "A fixed direct normal irradiance, applied the same at every sun position. The default (1000 W/m² -- the trace's own native normalisation), so a fresh project's numbers match what every trace already assumed."
    ),
    clearsky: segButton(
      dniSeg,
      "Clear-sky model",
      false,
      () => store.set("doc.sun.dni.mode", "clearsky"),
      "Beer-Lambert clear-sky DNI (heliostat.dni.ClearSkyDNI): highest near solar noon, falling toward sunrise/sunset as the beam crosses more atmosphere. No weather, no clouds -- a cloud-free upper bound, not a forecast."
    ),
  };
  body.appendChild(dniSeg);

  const dniConstant = numberRow(body, {
    key: "dni_constant_w_m2",
    label: "DNI (W/m²)",
    path: "doc.sun.dni.constant_w_m2",
    min: 1,
    max: 1500,
    step: 1,
    tooltip:
      "Direct normal irradiance assumed at every sun position -- scales power, flux, and concentration consistently across every trace, sweep, and estimate.",
  });

  const dniNote = document.createElement("div");
  dniNote.className = "summary";
  body.appendChild(dniNote);

  container.appendChild(head);
  container.appendChild(body);

  els = {
    chev,
    body,
    modeBtns,
    siteFields,
    directFields,
    lat,
    lon,
    tz,
    date,
    hour,
    az,
    el,
    solvedLine,
    siteErr,
    dniBtns,
    dniConstant,
    dniNote,
  };
  built = true;
}

// Mirrors heliostat.web.app's DNISetting.describe() -- the short, rider-
// wording label ("DNI: clear-sky model" / "DNI: 850 W/m² fixed"), NOT the
// verbose heliostat.dni provider describe() strings a day/year result also
// carries as a diagnostic. Kept in lockstep with that Python method if it
// ever changes.
function describeDni(dni) {
  const d = dni || { mode: "constant", constant_w_m2: 1000 };
  if (d.mode !== "clearsky") {
    const v = d.constant_w_m2 != null ? d.constant_w_m2 : 1000;
    return `DNI: ${v} W/m² fixed`;
  }
  const scale = d.clearsky_scale;
  return scale && scale !== 1 ? `DNI: clear-sky model x${scale}` : "DNI: clear-sky model";
}

function fmtHour(h) {
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");

  els.body.style.display = ui.expanded.sun ? "" : "none";
  els.chev.style.transform = ui.expanded.sun ? "rotate(90deg)" : "";

  const mode = doc.sun.mode || "site";
  els.modeBtns.site.classList.toggle("active", mode === "site");
  els.modeBtns.direct.classList.toggle("active", mode === "direct");
  els.siteFields.style.display = mode === "site" ? "" : "none";
  els.directFields.style.display = mode === "direct" ? "" : "none";

  const s = doc.sun.site;
  setVal(els.lat, s.latitude_deg);
  setVal(els.lon, s.longitude_deg);
  setVal(els.tz, s.timezone_h);
  setVal(els.hour, s.hour);
  const iso = `${s.year}-${String(s.month).padStart(2, "0")}-${String(s.day).padStart(2, "0")}`;
  if (document.activeElement !== els.date && els.date.value !== iso) els.date.value = iso;

  setVal(els.az, doc.sun.az);
  setVal(els.el, doc.sun.el);

  els.siteErr.hidden = !solveError;
  if (solveError) els.siteErr.textContent = solveError;

  if (mode === "site") {
    if (solved) {
      const rise = solved.sunrise_h == null ? "—" : fmtHour(solved.sunrise_h);
      const set = solved.sunset_h == null ? "—" : fmtHour(solved.sunset_h);
      els.solvedLine.innerHTML =
        `<strong>Az ${doc.sun.az.toFixed(1)}° · El ${doc.sun.el.toFixed(1)}°</strong>` +
        `<br />sunrise ${rise} · sunset ${set}` +
        (solved.above_horizon ? "" : " · the sun is below the horizon");
    } else {
      els.solvedLine.textContent = "Solving…";
    }
  }

  // -- spec §M.7: site DNI. The mode toggle always shows; the constant
  // input hides under "Clear-sky model" (it has no effect there), and the
  // note line states the assumption in effect either way -- the rider's
  // core ask: never leave DNI an implicit, unstated default.
  const dni = doc.sun.dni || { mode: "constant", constant_w_m2: 1000, clearsky_scale: 1 };
  const dniClearsky = dni.mode === "clearsky";
  els.dniBtns.constant.classList.toggle("active", !dniClearsky);
  els.dniBtns.clearsky.classList.toggle("active", dniClearsky);
  els.dniConstant.parentElement.style.display = dniClearsky ? "none" : "";
  setVal(els.dniConstant, dni.constant_w_m2);
  els.dniNote.textContent = describeDni(dni);
}

// Any site edit re-solves; main.js's subscriber calls render() on every
// store change, so this is the one hook needed.
store.subscribe((path) => {
  if (path.startsWith("doc.sun.site") || path === "doc.sun.mode") scheduleSolve();
});
