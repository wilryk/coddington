// Receiver & Tower stage: Prime focus / Axicon / Cassegrain, each with its
// own honest labels (docs/ui-spec.md 2.2's table -- no shared "tower
// height" alias). Field descriptors now live in ../fields.js so the
// floating inspector (../inspector.js) can render the identical rows for
// whichever optics is selected (docs/ui-spec.md 2.4).
import { store } from "../store.js";
import {
  numberRow,
  setVal,
  sectionHeaderRow,
  RECEIVER_FIELD_TABLE,
  OPTICS_LABELS,
  RECEIVER_TYPE_OPTIONS,
  receiverFieldVisible,
  apertureMissMessage,
} from "../fields.js";
import { postErrorMapImport, postErrorMapStats, postSecondarySag, postSecondarySagFeaCsv } from "../api.js";

let built = false;
let els = {};
let lastContainer = null;

// docs/ui-spec-v0.2.md §E2: the secondary's own measured error-map import
// (module-local UI state -- mirrors tabs/shape.js's own errorMapImportBusy/
// errorMapImportError pattern exactly, just for the secondary's map instead
// of the heliostat's). Keyed by nothing -- only one optics is ever active
// at a time, so one set of transient state is enough regardless of which
// optics's own store path the import writes into.
let errorMapImportBusy = false;
let errorMapImportError = null;

// View-sag overlay state: the fetched PNG (as an object URL, revoked on the
// next fetch/close) plus the peak-to-valley/contour-interval numbers the
// response headers carry, an in-flight/error flag, and an export-busy flag
// for the overlay's own "Export CSV for FEA" button.
let sagOpen = false;
let sagLoading = false;
let sagError = null;
let sagImgUrl = null;
let sagPeakToValleyMm = null;
let sagContourIntervalMm = null;
let sagExportBusy = false;
let sagExportError = null;
// The optics the currently-open (or in-flight) sag view is FOR -- captured
// at open time so a stray re-render after the user has since switched
// optics doesn't relabel or re-request against the wrong tower.
let sagOptics = null;

function rerender() {
  if (lastContainer) render(lastContainer);
}

// Cached /design/errormap/stats result for the secondary's own map -- same
// "keyed by object identity, refetched only when it changes" reasoning as
// tabs/shape.js's own errorMapStatsGrid/errorMapStats (see that file's
// ensureErrorMapStats): an import already returns its own stats inline, so
// this only fires for a map that arrived some other way (Library load,
// project reopen).
let secondaryErrorMapStatsGrid = null;
let secondaryErrorMapStats = null;
let secondaryErrorMapStatsBusy = false;

function ensureSecondaryErrorMapStats(map) {
  if (map === secondaryErrorMapStatsGrid || secondaryErrorMapStatsBusy) return;
  secondaryErrorMapStatsBusy = true;
  postErrorMapStats(map)
    .then((data) => {
      secondaryErrorMapStatsBusy = false;
      secondaryErrorMapStatsGrid = map;
      secondaryErrorMapStats = data;
      rerender();
    })
    .catch(() => {
      secondaryErrorMapStatsBusy = false;
      // Best-effort background refresh, same as shape.js's own -- the chip
      // just shows "attached" with no numbers until this succeeds.
    });
}

// §E2's "Measured error map" section for the secondary -- mirrors
// tabs/shape.js's own error-map import UI (f2bb0f4) almost verbatim, just
// writing to doc.opticsParams.<optics>.secondary_error_map instead of
// doc.design.errors.error_map, plus a "View sag" trigger the heliostat's
// own section has no equivalent for (that tab has its own always-visible
// sag panel instead -- this compact sidebar group opens one on demand).
function buildSecondaryMapUi(wrap) {
  const row = document.createElement("div");
  row.className = "frow";
  row.style.marginTop = "2px";
  const label = document.createElement("label");
  label.textContent = "Measured error map";
  label.title =
    "A real deformation map (gravity sag, wind load, thermal) from FEA or deflectometry, applied on top of the secondary's figure. Monte Carlo only -- cone modes ignore it.";
  row.appendChild(label);
  const importBtn = document.createElement("div");
  importBtn.className = "btn small";
  importBtn.textContent = "Import CSV…";
  row.appendChild(importBtn);
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".csv,text/csv";
  fileInput.hidden = true;
  row.appendChild(fileInput);
  wrap.appendChild(row);

  const chip = document.createElement("div");
  chip.className = "errormapchip";
  chip.hidden = true;
  const badge = document.createElement("span");
  badge.className = "badge import";
  badge.textContent = "Monte Carlo only";
  badge.title =
    "Applied on top of the secondary's figure in Monte Carlo traces only -- cone modes (Ultra fast, Fast accurate) ignore an attached map entirely.";
  const statsText = document.createElement("span");
  statsText.className = "errormapstats";
  const removeBtn = document.createElement("span");
  removeBtn.className = "btn small";
  removeBtn.textContent = "Remove";
  chip.appendChild(badge);
  chip.appendChild(statsText);
  chip.appendChild(removeBtn);
  wrap.appendChild(chip);

  const errEl = document.createElement("div");
  errEl.className = "fielderr";
  errEl.hidden = true;
  wrap.appendChild(errEl);

  function startImport(file) {
    if (!file) return;
    errorMapImportBusy = true;
    errorMapImportError = null;
    rerender();
    const reader = new FileReader();
    reader.onload = () => {
      postErrorMapImport(String(reader.result))
        .then((data) => {
          errorMapImportBusy = false;
          const doc = store.get("doc");
          store.set(`doc.opticsParams.${doc.optics}.secondary_error_map`, data.grid);
          // Same inline-cache trick as tabs/shape.js: the import response
          // already carries this map's own stats, so the chip shows them
          // on the very next render with no extra round trip.
          secondaryErrorMapStatsGrid = data.grid;
          secondaryErrorMapStats = {
            grid_size: data.grid_size,
            coverage_fraction: data.coverage_fraction,
            rms_slope_mrad: data.rms_slope_mrad,
          };
          rerender();
        })
        .catch((err) => {
          errorMapImportBusy = false;
          errorMapImportError = (err && err.message) || "Could not import the error map CSV.";
          rerender();
        });
    };
    reader.onerror = () => {
      errorMapImportBusy = false;
      errorMapImportError = "Could not read the file.";
      rerender();
    };
    reader.readAsText(file);
  }
  importBtn.addEventListener("click", () => {
    if (errorMapImportBusy) return;
    fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = ""; // allow re-selecting the same file later
    startImport(file);
  });
  removeBtn.addEventListener("click", () => {
    const doc = store.get("doc");
    store.set(`doc.opticsParams.${doc.optics}.secondary_error_map`, null);
    errorMapImportError = null;
    rerender();
  });

  // docs/ui-spec-v0.2.md §E2's third bullet: "View sag" -- nominal figure +
  // parametric warp + imported map, summed. Lives right under the map/warp
  // controls it visualises, so perturb -> view sag -> retrace is one loop.
  const sagRow = document.createElement("div");
  sagRow.className = "frow";
  sagRow.style.marginTop = "2px";
  const sagLabel = document.createElement("label");
  sagLabel.textContent = "Secondary sag";
  sagLabel.title = "Nominal figure + parametric warp + imported map, summed -- same jet colormap and stated contour interval as the heliostat's own sag map.";
  sagRow.appendChild(sagLabel);
  const viewSagBtn = document.createElement("div");
  viewSagBtn.className = "btn small";
  viewSagBtn.textContent = "View sag";
  viewSagBtn.addEventListener("click", () => openSagView());
  sagRow.appendChild(viewSagBtn);
  wrap.appendChild(sagRow);

  return { row, chip, badge, statsText, removeBtn, errEl, importBtn, fileInput, viewSagBtn };
}

function openSagView() {
  const doc = store.get("doc");
  sagOptics = doc.optics;
  sagOpen = true;
  fetchSagView();
}

function closeSagView() {
  sagOpen = false;
  if (sagImgUrl) {
    URL.revokeObjectURL(sagImgUrl);
    sagImgUrl = null;
  }
  sagError = null;
  sagExportError = null;
  rerender();
}

function fetchSagView() {
  const doc = store.get("doc");
  const optics = sagOptics || doc.optics;
  sagLoading = true;
  sagError = null;
  rerender();
  postSecondarySag(optics, doc.opticsParams[optics])
    .then(({ blob, peakToValleyMm, contourIntervalMm }) => {
      sagLoading = false;
      if (sagImgUrl) URL.revokeObjectURL(sagImgUrl);
      sagImgUrl = URL.createObjectURL(blob);
      sagPeakToValleyMm = peakToValleyMm;
      sagContourIntervalMm = contourIntervalMm;
      rerender();
    })
    .catch((err) => {
      sagLoading = false;
      sagError = (err && err.message) || "Could not render the secondary sag map.";
      rerender();
    });
}

function exportSagCsv() {
  const doc = store.get("doc");
  const optics = sagOptics || doc.optics;
  sagExportBusy = true;
  sagExportError = null;
  rerender();
  postSecondarySagFeaCsv(optics, doc.opticsParams[optics])
    .then((blob) => {
      sagExportBusy = false;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "secondary-sag-fea.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      rerender();
    })
    .catch((err) => {
      sagExportBusy = false;
      sagExportError = (err && err.message) || "Could not export the sag CSV.";
      rerender();
    });
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
  h2.textContent = "Receiver & Tower";
  head.appendChild(chev);
  head.appendChild(h2);
  head.addEventListener("click", () => {
    const ui = store.get("ui");
    store.set("ui.expanded.receiver", !ui.expanded.receiver);
  });

  const body = document.createElement("div");
  body.className = "stagebody";

  const opticsSeg = document.createElement("div");
  opticsSeg.className = "seg";
  const opticsBtns = {};
  for (const [key, label] of OPTICS_LABELS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => store.set("doc.optics", key));
    opticsSeg.appendChild(btn);
    opticsBtns[key] = btn;
  }
  body.appendChild(opticsSeg);

  // prime_focus only -- receiver_type is a string, so it gets its own
  // segmented control rather than a RECEIVER_FIELD_TABLE/numberRow entry.
  const receiverTypeSeg = document.createElement("div");
  receiverTypeSeg.className = "seg";
  const receiverTypeBtns = {};
  for (const [key, label] of RECEIVER_TYPE_OPTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => store.set("doc.opticsParams.prime_focus.receiver_type", key));
    receiverTypeSeg.appendChild(btn);
    receiverTypeBtns[key] = btn;
  }
  body.appendChild(receiverTypeSeg);

  const fieldsByOptics = {};
  const inputsByOptics = {};
  // Row element per field key, keyed by optics -- lets prime_focus's
  // cylinder/frustum-only rows (receiverFieldVisible) hide independently of
  // the rest of their own optics block.
  const rowsByOptics = {};
  // The amber aperture-miss warning (docs/ui-spec.md 2.3) belongs right
  // under the aperture_radius_mm field -- only axicon and cassegrain have
  // one, so prime_focus simply gets no warnBox.
  const warnBoxByOptics = {};
  // §E2: the secondary's own measured-error-map import chip + "View sag"
  // trigger, one set per optics (axicon/cassegrain only -- prime_focus's
  // field list carries no `custom: true` entry, so it never builds one).
  const secondaryMapUiByOptics = {};
  for (const [optics, fields] of Object.entries(RECEIVER_FIELD_TABLE)) {
    const wrap = document.createElement("div");
    const inputs = {};
    const rows = {};
    let warnBox = null;
    for (const field of fields) {
      // docs/ui-spec-v0.2.md §E2: a field carrying `sectionHeader` starts a
      // named subgroup ("Perturbations", or the map/warp's own "Surface
      // deformation & warp") within its optics block -- same "subhead"
      // convention tabs/shape.js uses for Surface figure / Facet curvature
      // / Optical errors. `sectionHeaderBadge` adds the same "Monte Carlo
      // only" pill tabs/shape.js's own error-map chip uses (f2bb0f4) --
      // present only on the map/warp subgroup, since rigid-body fields
      // apply at every fidelity.
      if (field.sectionHeader) sectionHeaderRow(wrap, field.sectionHeader, field.sectionHeaderBadge);
      if (field.custom) {
        // secondary_error_map: no numberRow (a grid object isn't a number)
        // -- build the import-chip UI here instead, mirroring
        // tabs/shape.js's own "Measured error map" section exactly, just
        // writing to doc.opticsParams.<optics>.secondary_error_map.
        secondaryMapUiByOptics[optics] = buildSecondaryMapUi(wrap);
        continue;
      }
      const input = numberRow(wrap, field);
      inputs[field.key] = input;
      rows[field.key] = input.parentElement;
      if (field.key === "aperture_radius_mm") {
        warnBox = document.createElement("div");
        warnBox.className = "fieldwarn";
        warnBox.hidden = true;
        wrap.appendChild(warnBox);
      }
    }
    body.appendChild(wrap);
    fieldsByOptics[optics] = wrap;
    inputsByOptics[optics] = inputs;
    rowsByOptics[optics] = rows;
    warnBoxByOptics[optics] = warnBox;
  }

  const errorBox = document.createElement("div");
  errorBox.className = "fielderr";
  errorBox.hidden = true;
  body.appendChild(errorBox);

  const actions = document.createElement("div");
  actions.className = "stageactions";
  const saveBtn = document.createElement("div");
  saveBtn.className = "btn disabled-link";
  saveBtn.textContent = "Save config…";
  saveBtn.title = "Library -- coming in a later phase";
  const swapBtn = document.createElement("div");
  swapBtn.className = "btn disabled-link";
  swapBtn.textContent = "Swap from library…";
  swapBtn.title = "Library -- coming in a later phase";
  actions.appendChild(saveBtn);
  actions.appendChild(swapBtn);
  body.appendChild(actions);

  // §E2's "View sag" overlay -- reuses app.css's .overlay/.overlay-panel/
  // .overlay-close, the same lightbox chrome the run bar's flux map and the
  // Analysis tab's "Manage saved runs" panel already use, rather than
  // declaring a second modal chrome here.
  const sagOverlay = document.createElement("div");
  sagOverlay.className = "overlay";
  sagOverlay.hidden = true;
  sagOverlay.addEventListener("click", (e) => {
    if (e.target === sagOverlay) closeSagView();
  });
  const sagOverlayPanel = document.createElement("div");
  sagOverlayPanel.className = "overlay-panel";
  const sagOverlayClose = document.createElement("button");
  sagOverlayClose.className = "overlay-close";
  sagOverlayClose.textContent = "×";
  sagOverlayClose.addEventListener("click", () => closeSagView());
  const sagOverlayH2 = document.createElement("h2");
  sagOverlayH2.textContent = "Secondary sag";
  const sagOverlayImg = document.createElement("img");
  sagOverlayImg.alt = "Secondary sag map";
  sagOverlayImg.hidden = true;
  const sagOverlayPlaceholder = document.createElement("p");
  sagOverlayPlaceholder.className = "placeholder";
  const sagOverlayCaption = document.createElement("div");
  sagOverlayCaption.className = "caption";
  const sagOverlayErrEl = document.createElement("div");
  sagOverlayErrEl.className = "fielderr";
  sagOverlayErrEl.hidden = true;
  const sagOverlayExportBtn = document.createElement("div");
  sagOverlayExportBtn.className = "btn small";
  sagOverlayExportBtn.textContent = "Export CSV for FEA";
  sagOverlayExportBtn.addEventListener("click", () => exportSagCsv());
  sagOverlayPanel.appendChild(sagOverlayClose);
  sagOverlayPanel.appendChild(sagOverlayH2);
  sagOverlayPanel.appendChild(sagOverlayImg);
  sagOverlayPanel.appendChild(sagOverlayPlaceholder);
  sagOverlayPanel.appendChild(sagOverlayCaption);
  sagOverlayPanel.appendChild(sagOverlayExportBtn);
  sagOverlayPanel.appendChild(sagOverlayErrEl);
  sagOverlay.appendChild(sagOverlayPanel);

  container.appendChild(head);
  container.appendChild(body);
  container.appendChild(sagOverlay);

  els = {
    chev,
    body,
    opticsBtns,
    receiverTypeSeg,
    receiverTypeBtns,
    fieldsByOptics,
    inputsByOptics,
    rowsByOptics,
    warnBoxByOptics,
    secondaryMapUiByOptics,
    errorBox,
    sagOverlay,
    sagOverlayImg,
    sagOverlayPlaceholder,
    sagOverlayCaption,
    sagOverlayErrEl,
    sagOverlayExportBtn,
  };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  lastContainer = container;
  const doc = store.get("doc");
  const ui = store.get("ui");
  const optics = doc.optics;

  els.body.style.display = ui.expanded.receiver ? "" : "none";
  els.chev.style.transform = ui.expanded.receiver ? "rotate(90deg)" : "";
  // docs/ui-spec.md 2.1 + mockup M4: highlighted while this stage owns the
  // viewport (elevation view). ui.view itself is driven by main.js's
  // store.subscribe on ui.expanded.receiver, not derived here.
  container.classList.toggle("selectedstage", ui.view === "elevation");

  for (const [key, btn] of Object.entries(els.opticsBtns)) {
    btn.classList.toggle("active", key === optics);
  }
  for (const [key, wrap] of Object.entries(els.fieldsByOptics)) {
    wrap.style.display = key === optics ? "" : "none";
  }

  const params = doc.opticsParams[optics];

  els.receiverTypeSeg.style.display = optics === "prime_focus" ? "" : "none";
  if (optics === "prime_focus") {
    const rType = params.receiver_type || "flat";
    for (const [key, btn] of Object.entries(els.receiverTypeBtns)) {
      btn.classList.toggle("active", key === rType);
    }
    for (const field of RECEIVER_FIELD_TABLE.prime_focus) {
      els.rowsByOptics.prime_focus[field.key].style.display = receiverFieldVisible(field, params) ? "" : "none";
    }
  }

  for (const [key, input] of Object.entries(els.inputsByOptics[optics])) {
    setVal(input, params[key]);
  }

  const warnBox = els.warnBoxByOptics[optics];
  if (warnBox) {
    const msg = apertureMissMessage(ui.miss);
    warnBox.hidden = !msg;
    if (msg) warnBox.textContent = msg;
  }

  const err = ui.geometryError;
  if (err && err.forReceiver) {
    els.errorBox.hidden = false;
    els.errorBox.textContent = err.message;
    els.errorBox.className = err.severity === "warn" ? "fieldwarn" : "fielderr";
  } else {
    els.errorBox.hidden = true;
  }

  const mapUi = els.secondaryMapUiByOptics[optics];
  if (mapUi) renderSecondaryMapUi(mapUi, params);

  renderSagOverlay();
}

// §E2: the current optics's secondary_error_map chip -- badge, implied-RMS
// stats (fetched once per distinct grid, see ensureSecondaryErrorMapStats),
// and the import-in-progress/error states. Mirrors
// tabs/shape.js::renderErrorMapSection almost exactly.
function renderSecondaryMapUi(mapUi, params) {
  mapUi.importBtn.textContent = errorMapImportBusy ? "Importing…" : "Import CSV…";
  mapUi.importBtn.classList.toggle("disabled-link", errorMapImportBusy);
  const map = params.secondary_error_map;
  if (map) {
    ensureSecondaryErrorMapStats(map);
    mapUi.chip.hidden = false;
    const stats = map === secondaryErrorMapStatsGrid ? secondaryErrorMapStats : null;
    mapUi.statsText.textContent = stats
      ? ` ${stats.grid_size.nx}×${stats.grid_size.ny} grid, ` +
        `${(stats.coverage_fraction * 100).toFixed(0)}% aperture coverage, ` +
        `${stats.rms_slope_mrad.toFixed(3)} mrad implied RMS slope`
      : " loading grid stats…";
  } else {
    mapUi.chip.hidden = true;
  }
  mapUi.errEl.hidden = !errorMapImportError;
  mapUi.errEl.textContent = errorMapImportError || "";
}

// §E2's "View sag" overlay -- fetched on demand (openSagView), not live-
// debounced like the Shape tab's always-visible sag panel: this is a
// sidebar-triggered popup, so paying network cost only while it is
// actually open is the right default.
function renderSagOverlay() {
  els.sagOverlay.hidden = !sagOpen;
  if (!sagOpen) return;
  const label = sagOptics === "cassegrain" ? "Cassegrain" : "Axicon";
  if (sagLoading) {
    els.sagOverlayImg.hidden = true;
    els.sagOverlayPlaceholder.hidden = false;
    els.sagOverlayPlaceholder.textContent = "Rendering…";
    els.sagOverlayCaption.textContent = "";
  } else if (sagError) {
    els.sagOverlayImg.hidden = true;
    els.sagOverlayPlaceholder.hidden = false;
    els.sagOverlayPlaceholder.textContent = sagError;
    els.sagOverlayCaption.textContent = "";
  } else if (sagImgUrl) {
    els.sagOverlayImg.hidden = false;
    els.sagOverlayImg.src = sagImgUrl;
    els.sagOverlayPlaceholder.hidden = true;
    const pv = sagPeakToValleyMm != null ? `peak-to-valley ${sagPeakToValleyMm.toFixed(3)} mm` : "";
    const ci = sagContourIntervalMm != null ? ` · contours every ${sagContourIntervalMm} mm` : "";
    els.sagOverlayCaption.textContent = `${label} secondary -- nominal figure + parametric warp + imported map, summed. ${pv}${ci}`;
  }
  els.sagOverlayExportBtn.textContent = sagExportBusy ? "Exporting…" : "Export CSV for FEA";
  els.sagOverlayExportBtn.classList.toggle("disabled-link", sagExportBusy);
  els.sagOverlayErrEl.hidden = !sagExportError;
  els.sagOverlayErrEl.textContent = sagExportError || "";
}
