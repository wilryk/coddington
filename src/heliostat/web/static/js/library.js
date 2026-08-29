// Library slide-over (docs/ui-spec.md 5, mockup M5): heliostat designs,
// receiver configs and projects, built-ins locked, plus the legacy-setup
// importer under Projects. Like the sidebar panels, this is a build-once/
// render-from-store module -- but unlike them, it also owns the network
// side of its own collections (fetch on open, refetch after any mutation)
// and a slab of ephemeral, per-drawer-open UI state (which card is mid-
// rename, the save-as name box, the last import's report) that has no
// business living in the global store: it is meaningless the instant the
// drawer closes, and nothing outside this file ever reads it.
//
// main.js calls render(drawer, backdrop) on every store change exactly like
// it calls the sidebar panels; this module fetches on its own schedule
// (open transition, after a mutation) and re-renders itself directly
// afterwards rather than routing through a store write, since a listing
// arriving over the network isn't state main.js or any other module needs
// to react to.
import { store, DEFAULT_DOC } from "./store.js";
import {
  DESIGN_ERROR_KEYS,
  currentDesignPayload,
  deleteLibraryEntry,
  errorsFromDesignDocument,
  getLibrary,
  getLibraryEntry,
  getSetup,
  getSetups,
  saveLibraryEntry,
} from "./api.js";
import { OPTICS_LABELS } from "./fields.js";
import { serializeProject, applyProject, convertLegacySetup } from "./project.js";

const OPTICS_NAME = Object.fromEntries(OPTICS_LABELS);

const TABS = [
  ["designs", "Heliostat designs"],
  ["receivers", "Receiver configs"],
  ["projects", "Projects"],
];

let built = false;
let els = {};

// Per-collection entry lists. `designs`/`receivers` entries carry their
// full `.document` (fetched alongside the listing -- see refreshCollection)
// so the card subtitle and IN USE check have something to read; `projects`
// entries stay listing-only (name/builtin/saved_at) per the brief's "name +
// date is enough" for that tab.
const cache = { designs: null, receivers: null, projects: null, setups: null };

const state = {
  wasOpen: false,
  listError: {}, // collection -> message
  renameTarget: null, // {collection, name} | null
  renameValue: "",
  deleteConfirm: null, // {collection, name} | null
  saveAsName: "",
  saveError: null, // {collection, message} | null
  importReport: null, // {setupName, unmapped, savedAs?, applyError?, error?} | null
};

function resetEphemeral() {
  state.renameTarget = null;
  state.deleteConfirm = null;
  state.saveError = null;
}

// -- formatting ---------------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtNum(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  return Math.round(v).toLocaleString();
}

function receiverSubtitle(doc) {
  const p = doc.params || {};
  if (doc.optics === "prime_focus") {
    return `Focus ${fmtNum(p.focus_height_mm)} mm · window ${fmtNum(p.window_half_u_mm * 2)} × ${fmtNum(p.window_half_v_mm * 2)} mm`;
  }
  if (doc.optics === "axicon") {
    return `Apex ${fmtNum(p.apex_height_mm)} mm · half angle ${p.half_angle_deg}° · aperture r ${fmtNum(p.aperture_radius_mm)} mm · receiver ${fmtNum(p.receiver_z_mm)} mm`;
  }
  if (doc.optics === "cassegrain") {
    return `Vertex ${fmtNum(p.vertex_z_mm)} mm · focus ${fmtNum(p.focus_height_mm)} mm · receiver ${fmtNum(p.receiver_z_mm)} mm · aperture r ${fmtNum(p.aperture_radius_mm)} mm`;
  }
  return OPTICS_NAME[doc.optics] || doc.optics;
}

function designSubtitle(doc) {
  if (doc.type === "rect") {
    const wM = (doc.width_mm / 1000).toFixed(1);
    const hM = (doc.height_mm / 1000).toFixed(1);
    return `${wM} × ${hM} m rectangle — ${doc.surface}`;
  }
  if (doc.type === "grid") {
    return `${doc.n_u}×${doc.n_v} facet grid — ${doc.surface}`;
  }
  if (doc.type === "custom") {
    const n = (doc.vertices_mm && doc.vertices_mm.length) || 0;
    return `custom outline, ${n} vertices — ${doc.surface}`;
  }
  return `${doc.type} — ${doc.surface}`;
}

function projectSubtitle(entry) {
  if (!entry.saved_at) return "";
  const when = new Date(entry.saved_at);
  return Number.isNaN(when.getTime()) ? `Saved ${entry.saved_at}` : `Saved ${when.toLocaleString()}`;
}

// docs/ui-spec-v0.2.md §P / mockup M20: a built-in reference field's card
// subtitle reads "<field kind> · <N> heliostats · <receiver summary>" --
// e.g. "Surround field · 2,650 heliostats · 140 m tower · external
// cylindrical molten-salt receiver" -- everything the provenance metadata
// (heliostat.web.builtin_library.BUILTIN_PROJECT_PROVENANCE, fetched
// alongside the document in refreshCollection) carries.
function builtinProjectSubtitle(prov) {
  if (!prov) return "";
  const count = Number.isFinite(prov.heliostat_count) ? prov.heliostat_count.toLocaleString() : "?";
  return `${prov.field_kind} · ${count} heliostats · ${prov.receiver_summary}`;
}

// -- IN USE matching (loose numeric equality, 1e-6 relative) -------------

function numsClose(a, b) {
  if (typeof a !== "number" || typeof b !== "number") return a === b;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
  const scale = Math.max(1, Math.abs(a), Math.abs(b));
  return Math.abs(a - b) <= 1e-6 * scale;
}

function receiverMatchesCurrent(doc, entryDocument) {
  if (entryDocument.optics !== doc.optics) return false;
  const current = doc.opticsParams[doc.optics] || {};
  const params = entryDocument.params || {};
  for (const key of Object.keys(params)) {
    if (!numsClose(current[key], params[key])) return false;
  }
  return true;
}

// numsClose over nested arrays too -- a custom design's vertices_mm is a
// list of [u, v] pairs, and phase 3c's error fields make the wire payload
// (not doc.designParams) the honest thing to compare a saved document
// against: currentDesignPayload carries reflectance as the same 0-1
// fraction the document does, and the mirror-expanded vertex list.
function valuesClose(a, b) {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => valuesClose(v, b[i]));
  }
  return numsClose(a, b);
}

function designMatchesCurrent(doc, entryDocument) {
  const current = currentDesignPayload(doc);
  if (entryDocument.type !== current.type) return false;
  if ((entryDocument.surface || "twisting") !== current.surface) return false;
  for (const key of Object.keys(entryDocument)) {
    if (key === "type" || key === "surface") continue;
    if (!valuesClose(current[key], entryDocument[key])) return false;
  }
  return true;
}

// -- fetching -------------------------------------------------------------

function findEntry(collection, name) {
  const list = cache[collection] || [];
  return list.find((e) => e.name === name);
}

function fetchDocumentFor(collection, name) {
  const entry = findEntry(collection, name);
  if (entry && entry.document) return Promise.resolve(entry.document);
  return getLibraryEntry(collection, name).then((full) => full.document);
}

function refreshCollection(collection) {
  return getLibrary(collection)
    .then((data) => {
      if (collection === "projects") {
        // A user's own saved project stays listing-only (name + date is
        // enough, per the original brief this module's header comment
        // still describes). Built-ins are different since §P: the four
        // reference-field cards need their heliostat count, field kind,
        // receiver summary and (binding, §P) their citation -- all of
        // which live in the document/provenance the {name} route returns,
        // not the plain listing. Only four entries ever match, so this
        // costs four extra requests, once per drawer-open/mutation, not
        // per render.
        const builtinNames = data.entries.filter((e) => e.builtin).map((e) => e.name);
        return Promise.all(
          builtinNames.map((name) =>
            getLibraryEntry("projects", name)
              .then((full) => [name, { document: full.document, provenance: full.provenance }])
              .catch(() => [name, null])
          )
        ).then((pairs) => {
          const byName = new Map(pairs);
          cache.projects = data.entries.map((e) => {
            const extra = byName.get(e.name);
            return extra ? Object.assign({}, e, extra) : e;
          });
          state.listError.projects = null;
          update();
        });
      }
      // designs/receivers: fetch every entry's document alongside the
      // listing -- the collections are small (three built-ins plus
      // whatever the user has saved), and the card subtitle + IN USE
      // check both need the document, which /api/library/{collection}
      // deliberately does not include (that's what the {name} route is
      // for).
      return Promise.all(
        data.entries.map((e) =>
          getLibraryEntry(collection, e.name)
            .then((full) => Object.assign({}, e, { document: full.document }))
            .catch((err) => Object.assign({}, e, { document: null, loadError: err.message }))
        )
      ).then((withDocs) => {
        cache[collection] = withDocs;
        state.listError[collection] = null;
        update();
      });
    })
    .catch((err) => {
      cache[collection] = [];
      state.listError[collection] = err.message;
      update();
    });
}

function refreshSetups() {
  return getSetups()
    .then((data) => {
      cache.setups = data.setups;
      update();
    })
    .catch(() => {
      cache.setups = [];
      update();
    });
}

function refreshAll() {
  refreshCollection("designs");
  refreshCollection("receivers");
  refreshCollection("projects");
  refreshSetups();
}

// -- actions ----------------------------------------------------------------

// docs/ui-spec.md 5: "Use" on a receiver swaps doc.optics + that layout's
// params; the field and heliostat design are never touched.
function useReceiver(name) {
  const entry = findEntry("receivers", name);
  if (!entry || !entry.document) return;
  const params = Object.assign({}, DEFAULT_DOC.opticsParams[entry.document.optics], entry.document.params || {});
  store.set("doc.optics", entry.document.optics);
  store.set(`doc.opticsParams.${entry.document.optics}`, params);
}

function useDesign(name) {
  const entry = findEntry("designs", name);
  if (!entry || !entry.document) return;
  const d = entry.document;
  const params = Object.assign({}, DEFAULT_DOC.designParams[d.type]);
  for (const key of Object.keys(d)) {
    // The optical-error fields ride flat in the document but live under
    // doc.design.errors in the store (phase 3c) -- routed there below, not
    // into designParams, where they'd be stray keys the request builder
    // then shadows.
    if (key === "type" || key === "surface" || DESIGN_ERROR_KEYS.indexOf(key) !== -1) continue;
    params[key] = d[key];
  }
  store.set("doc.design.type", d.type);
  store.set("doc.design.surface", d.surface || "twisting");
  store.set("doc.design.errors", errorsFromDesignDocument(d));
  store.set(`doc.designParams.${d.type}`, params);
}

function loadProject(name) {
  getLibraryEntry("projects", name)
    .then((full) => {
      const err = applyProject(full.document);
      if (err) {
        state.saveError = { collection: "projects", message: "Couldn't load that project: " + err };
      } else {
        store.set("ui.projectName", name);
        // docs/ui-spec-v0.2.md §P: while a built-in reconstruction is
        // loaded, its citation stamps into the top bar (main.js's
        // renderTopbar()) so a result screenshot carries the provenance,
        // not just the plant name. `full.provenance` is null for every
        // non-reconstruction project (a user's own, or a built-in with no
        // provenance entry).
        store.set("ui.projectProvenance", full.provenance || null);
        store.set("ui.dirty", false);
        state.saveError = null;
      }
      update();
    })
    .catch((err) => {
      state.saveError = { collection: "projects", message: err.message };
      update();
    });
}

function nextDuplicateName(collection, baseName, existingNames) {
  let candidate = `Copy of ${baseName}`;
  if (!existingNames.has(candidate)) return candidate;
  let i = 2;
  while (existingNames.has(`Copy ${i} of ${baseName}`)) i++;
  return `Copy ${i} of ${baseName}`;
}

function duplicateEntry(collection, name) {
  fetchDocumentFor(collection, name)
    .then((document) => {
      const existing = new Set((cache[collection] || []).map((e) => e.name));
      const newName = nextDuplicateName(collection, name, existing);
      return saveLibraryEntry(collection, newName, document);
    })
    .then(() => {
      state.saveError = null;
      return refreshCollection(collection);
    })
    .catch((err) => {
      state.saveError = { collection, message: err.message };
      update();
    });
}

function confirmRename(collection, oldName) {
  const newName = state.renameValue.trim();
  if (!newName || newName === oldName) {
    state.renameTarget = null;
    update();
    return;
  }
  // Load -> save under the new name -> delete the old one. A name that
  // collides with a built-in 409s on the save, before anything is deleted
  // -- that 409's message is exactly "'X' is a built-in ... and cannot be
  // overwritten", which reads fine surfaced verbatim as the rename's error.
  fetchDocumentFor(collection, oldName)
    .then((document) => saveLibraryEntry(collection, newName, document))
    .then(() => deleteLibraryEntry(collection, oldName))
    .then(() => {
      state.renameTarget = null;
      state.saveError = null;
      if (collection === "projects" && store.get("ui.projectName") === oldName) {
        store.set("ui.projectName", newName);
      }
      return refreshCollection(collection);
    })
    .catch((err) => {
      state.saveError = { collection, message: err.message };
      update();
    });
}

function confirmDelete(collection, name) {
  deleteLibraryEntry(collection, name)
    .then(() => {
      state.deleteConfirm = null;
      state.saveError = null;
      if (collection === "projects" && store.get("ui.projectName") === name) {
        store.set("ui.projectName", null);
        store.set("ui.projectProvenance", null);
      }
      return refreshCollection(collection);
    })
    .catch((err) => {
      state.saveError = { collection, message: err.message };
      update();
    });
}

function saveProjectAs(name) {
  if (!name) {
    state.saveError = { collection: "projects", message: "Enter a name to save as." };
    update();
    return;
  }
  const document = serializeProject(store.get("doc"), store.get("ui"));
  saveLibraryEntry("projects", name, document)
    .then(() => {
      state.saveAsName = "";
      state.saveError = null;
      store.set("ui.projectName", name);
      // A save always produces the user's own project, not a
      // reconstruction, even if the workspace was last loaded from a
      // built-in -- the top-bar citation stamp (§P) belongs to the
      // built-in it came from, not to a saved copy of it.
      store.set("ui.projectProvenance", null);
      store.set("ui.dirty", false);
      return refreshCollection("projects");
    })
    .catch((err) => {
      state.saveError = { collection: "projects", message: err.message };
      update();
    });
}

function saveProjectOverwrite() {
  const name = store.get("ui.projectName");
  if (!name) return;
  const document = serializeProject(store.get("doc"), store.get("ui"));
  saveLibraryEntry("projects", name, document)
    .then(() => {
      state.saveError = null;
      store.set("ui.dirty", false);
      return refreshCollection("projects");
    })
    .catch((err) => {
      state.saveError = { collection: "projects", message: err.message };
      update();
    });
}

// docs/ui-spec.md 5, "Migration": converts, saves under the setup's own
// name (falling back to "name (imported)" on a collision -- checked
// against a fresh listing here rather than relying on a 409, so this
// works whether or not the name happens to collide with a built-in;
// docs/ui-spec-v0.2.md §P gave `projects` its first four built-ins, and
// `existing` below already includes them since it comes from the same
// merged listing the Library drawer renders), loads it into the
// workspace, and reports what didn't map. The original setup is never
// touched -- no DELETE call anywhere in this path.
function importSetup(setupName) {
  state.importReport = null;
  getSetup(setupName)
    .then((setup) => {
      const { document, unmapped } = convertLegacySetup(setup.document);
      return getLibrary("projects").then((data) => {
        const existing = new Set(data.entries.map((e) => e.name));
        const targetName = existing.has(setupName) ? `${setupName} (imported)` : setupName;
        return saveLibraryEntry("projects", targetName, document).then(() => {
          const applyErr = applyProject(document);
          store.set("ui.projectName", targetName);
          store.set("ui.projectProvenance", null); // an imported setup is always the user's own
          store.set("ui.dirty", false);
          state.importReport = { setupName, unmapped, savedAs: targetName, applyError: applyErr };
          return refreshCollection("projects");
        });
      });
    })
    .catch((err) => {
      state.importReport = { setupName, unmapped: [], error: err.message };
      update();
    });
}

// -- DOM: card + section markup ---------------------------------------------

// docs/ui-spec-v0.2.md §P, mockup M20: the RECONSTRUCTION badge's citation
// and (when present) its disclosed-limitation caveats, directly beneath the
// card subtitle -- "the most important clause in this rider": a reference
// field that ships without its citation visible on the card is not
// acceptable, so this renders whenever `opts.provenance` is given rather
// than behind any further click.
function provenanceHtml(prov) {
  if (!prov) return "";
  let html = `<div class="idealnote">Ideal build: slope error, specularity, pointing error all 0</div>`;
  html += `<div class="citation"><strong>Source:</strong> ${esc(prov.citation)}</div>`;
  if (prov.caveats && prov.caveats.length) {
    html += '<ul class="citation caveats">' + prov.caveats.map((c) => `<li>${esc(c)}</li>`).join("") + "</ul>";
  }
  return html;
}

function cardHtml(collection, entry, opts) {
  const badges = [];
  if (opts.isBuiltin) badges.push('<span class="badge builtin">BUILT-IN</span>');
  if (opts.reconstruction) badges.push('<span class="badge recon">RECONSTRUCTION</span>');
  if (opts.isCurrent) badges.push('<span class="badge inuse">IN USE</span>');
  const lock = opts.isBuiltin
    ? '<svg class="lock" width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="1.6"><rect x="3.5" y="7" width="9" height="6.5" rx="1"></rect><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2"></path></svg>'
    : "";
  const nameEsc = esc(entry.name);
  let actions;
  if (opts.isBuiltin) {
    actions =
      `<div class="btn small" data-action="${opts.useAction}" data-collection="${collection}" data-name="${nameEsc}">${opts.useLabel}</div>` +
      `<div class="btn small" data-action="duplicate" data-collection="${collection}" data-name="${nameEsc}">Duplicate</div>`;
  } else {
    const useBtn = opts.isCurrent
      ? ""
      : `<div class="btn small" data-action="${opts.useAction}" data-collection="${collection}" data-name="${nameEsc}">${opts.useLabel}</div>`;
    actions =
      useBtn +
      `<div class="btn small" data-action="rename" data-collection="${collection}" data-name="${nameEsc}">Rename</div>` +
      `<div class="btn small" data-action="duplicate" data-collection="${collection}" data-name="${nameEsc}">Duplicate</div>` +
      `<div class="btn small" data-action="delete" data-collection="${collection}" data-name="${nameEsc}">Delete</div>`;
  }
  let extra = "";
  if (state.renameTarget && state.renameTarget.collection === collection && state.renameTarget.name === entry.name) {
    extra += `<div class="cardinline">
      <input type="text" class="val" data-role="rename-input" value="${esc(state.renameValue)}">
      <div class="btn small primary" data-action="rename-confirm" data-collection="${collection}" data-name="${nameEsc}">Save</div>
      <div class="btn small" data-action="rename-cancel">Cancel</div>
    </div>`;
  }
  if (state.deleteConfirm && state.deleteConfirm.collection === collection && state.deleteConfirm.name === entry.name) {
    extra += `<div class="cardinline">
      <span class="fielderr" style="margin:0; flex:1 1 auto">Delete "${nameEsc}"? This can't be undone.</span>
      <div class="btn small" data-action="delete-confirm" data-collection="${collection}" data-name="${nameEsc}">Delete</div>
      <div class="btn small" data-action="delete-cancel">Cancel</div>
    </div>`;
  }
  return `<div class="card${opts.isCurrent ? " current" : ""}">
    <div class="cardhead">${lock}<span class="name">${nameEsc}</span>${badges.join("")}</div>
    <div class="cardsub">${esc(opts.subtitle || "")}</div>
    ${provenanceHtml(opts.provenance)}
    <div class="cardactions">${actions}</div>
    ${extra}
  </div>`;
}

function cardForEntry(collection, entry, doc, isBuiltin) {
  let subtitle = "";
  let isCurrent = false;
  let provenance = null;
  if (collection === "receivers" && entry.document) {
    subtitle = receiverSubtitle(entry.document);
    isCurrent = receiverMatchesCurrent(doc, entry.document);
  } else if (collection === "designs" && entry.document) {
    subtitle = designSubtitle(entry.document);
    isCurrent = designMatchesCurrent(doc, entry.document);
  } else if (collection === "projects") {
    // §P built-ins carry provenance (fetched alongside the document in
    // refreshCollection); a user's own saved project never does, and falls
    // back to the plain "Saved <date>" subtitle exactly as before this
    // rider.
    subtitle = entry.provenance ? builtinProjectSubtitle(entry.provenance) : projectSubtitle(entry);
    provenance = entry.provenance || null;
    isCurrent = store.get("ui.projectName") === entry.name;
  }
  if (entry.loadError) subtitle = "Could not load: " + entry.loadError;
  return cardHtml(collection, entry, {
    subtitle,
    isCurrent,
    isBuiltin,
    reconstruction: !!(provenance && provenance.reconstruction),
    provenance,
    useLabel: collection === "projects" ? "Load" : "Use",
    useAction: collection === "projects" ? "load" : "use",
  });
}

function projectsSaveBarHtml() {
  const ui = store.get("ui");
  let html = '<div class="libsavebar">';
  if (ui.projectName) {
    html += `<div class="btn small primary" data-action="save-overwrite">Save "${esc(ui.projectName)}"</div>`;
  }
  html += `<input type="text" class="val" data-role="saveas-input" placeholder="Save current as…" value="${esc(state.saveAsName)}">`;
  html += `<div class="btn small" data-action="save-new">Save as new</div>`;
  html += "</div>";
  return html;
}

function importReportHtml() {
  const r = state.importReport;
  if (!r) return "";
  if (r.error) {
    return `<div class="importreport"><div class="fielderr">Import of "${esc(r.setupName)}" failed: ${esc(r.error)}</div><div class="btn small" data-action="dismiss-report">Dismiss</div></div>`;
  }
  const bits = [];
  bits.push(`Imported "${esc(r.setupName)}" as "${esc(r.savedAs)}".`);
  bits.push(r.unmapped.length ? `Not carried over: ${r.unmapped.map(esc).join("; ")}.` : "Everything mapped cleanly.");
  if (r.applyError) bits.push(`Saved, but couldn't load into the workspace: ${esc(r.applyError)}`);
  return `<div class="importreport"><div class="fieldwarn">${bits.join(" ")}</div><div class="btn small" data-action="dismiss-report">Dismiss</div></div>`;
}

function legacySetupsHtml() {
  const setups = cache.setups || [];
  if (!setups.length) return "";
  let html = '<div class="sectionlabel">Legacy setups</div>';
  for (const s of setups) {
    const nameEsc = esc(s.name);
    const when = s.saved_at ? new Date(s.saved_at) : null;
    const whenTxt = when && !Number.isNaN(when.getTime()) ? when.toLocaleString() : s.saved_at || "";
    html += `<div class="card">
      <div class="cardhead"><span class="name">${nameEsc}</span><span class="badge import">LEGACY SETUP</span></div>
      <div class="cardsub">${esc(whenTxt)}</div>
      <div class="cardactions"><div class="btn small" data-action="import" data-name="${nameEsc}">Import</div></div>
    </div>`;
  }
  return html;
}

// -- DOM: build once, rebuild the card list on demand ------------------------

function onBodyClick(e) {
  const el = e.target.closest("[data-action]");
  if (!el) return;
  const action = el.dataset.action;
  const collection = el.dataset.collection;
  const name = el.dataset.name;
  switch (action) {
    case "use":
      if (collection === "receivers") useReceiver(name);
      else if (collection === "designs") useDesign(name);
      break;
    case "load":
      loadProject(name);
      break;
    case "duplicate":
      duplicateEntry(collection, name);
      break;
    case "rename":
      state.renameTarget = { collection, name };
      state.renameValue = name;
      update();
      break;
    case "rename-confirm":
      confirmRename(collection, name);
      break;
    case "rename-cancel":
      state.renameTarget = null;
      update();
      break;
    case "delete":
      state.deleteConfirm = { collection, name };
      update();
      break;
    case "delete-cancel":
      state.deleteConfirm = null;
      update();
      break;
    case "delete-confirm":
      confirmDelete(collection, name);
      break;
    case "save-new":
      saveProjectAs(state.saveAsName.trim());
      break;
    case "save-overwrite":
      saveProjectOverwrite();
      break;
    case "import":
      importSetup(name);
      break;
    case "dismiss-report":
      state.importReport = null;
      update();
      break;
    default:
      break;
  }
}

function onBodyInput(e) {
  // No re-render here on purpose: renderTabBody()'s own focused-input guard
  // (below) already keeps a rebuild from yanking focus mid-keystroke, so
  // there is nothing to gain from also re-rendering on every character --
  // just record the value for the eventual Save/rename-confirm click.
  if (e.target.dataset.role === "rename-input") state.renameValue = e.target.value;
  else if (e.target.dataset.role === "saveas-input") state.saveAsName = e.target.value;
}

function build(container, backdrop) {
  backdrop.addEventListener("click", () => store.set("ui.libraryOpen", false));

  container.className = "drawer";
  container.innerHTML = "";

  const closeBtn = document.createElement("div");
  closeBtn.className = "close";
  closeBtn.textContent = "×";
  closeBtn.title = "Close";
  closeBtn.addEventListener("click", () => store.set("ui.libraryOpen", false));
  container.appendChild(closeBtn);

  const h2 = document.createElement("h2");
  h2.textContent = "Library";
  container.appendChild(h2);

  const tabsEl = document.createElement("div");
  tabsEl.className = "libtabs";
  const tabButtons = {};
  for (const [key, label] of TABS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", () => {
      resetEphemeral();
      store.set("ui.libraryTab", key);
    });
    tabsEl.appendChild(b);
    tabButtons[key] = b;
  }
  container.appendChild(tabsEl);

  const body = document.createElement("div");
  body.className = "libbody";
  body.addEventListener("click", onBodyClick);
  body.addEventListener("input", onBodyInput);
  container.appendChild(body);

  const foot = document.createElement("div");
  foot.className = "drawerfoot";
  foot.textContent =
    '“Use” swaps the receiver while your field and heliostat design stay put. Built-ins are the manuscript’s numbers — duplicate one to tweak it.';
  container.appendChild(foot);

  els = { closeBtn, h2, tabsEl, tabButtons, body, foot };
  built = true;
}

function renderTabBody() {
  // Focused-input guard (fields.js's setVal does the same job for the
  // sidebar's persistent inputs): this function throws away and rebuilds
  // the whole card list every call, which would steal focus/keystrokes out
  // from under the rename or save-as text box the instant the user typed
  // into it -- so skip the rebuild entirely while focus is inside.
  if (els.body.contains(document.activeElement)) return;

  const doc = store.get("doc");
  const ui = store.get("ui");
  const tab = ui.libraryTab;
  let html = "";

  if (tab === "projects") html += projectsSaveBarHtml();

  // One inline error box per tab -- covers save/rename/duplicate/delete
  // failures alike (a 409 name collision, a 422 validation detail, a
  // network error), whichever collection the active tab is showing.
  if (state.saveError && state.saveError.collection === tab) {
    html += `<div class="fielderr">${esc(state.saveError.message)}</div>`;
  }

  const entries = cache[tab];
  if (entries == null) {
    html += '<div class="hint">Loading…</div>';
  } else if (state.listError[tab]) {
    html += `<div class="fielderr">${esc(state.listError[tab])}</div>`;
  } else {
    const builtins = entries.filter((e) => e.builtin);
    const yours = entries.filter((e) => !e.builtin);
    if (builtins.length) {
      // docs/ui-spec-v0.2.md §P, mockup M20: the Projects tab's built-ins
      // are reconstructed reference fields, not the manuscript's own
      // numbers (designs/receivers keep the label that's always been true
      // of them).
      const label = tab === "projects" ? "Built-in — manuscript & reference fields" : "Built-in — manuscript baseline";
      html += `<div class="sectionlabel">${label}</div>`;
      for (const e of builtins) html += cardForEntry(tab, e, doc, true);
    }
    html += '<div class="sectionlabel">Yours</div>';
    html += yours.length ? yours.map((e) => cardForEntry(tab, e, doc, false)).join("") : '<div class="hint">Nothing saved yet.</div>';
  }

  if (tab === "projects") {
    html += importReportHtml();
    html += legacySetupsHtml();
  }

  els.body.innerHTML = html;
}

let drawerEl = null;
let backdropEl = null;

function update() {
  const ui = store.get("ui");
  const open = ui.libraryOpen;
  drawerEl.hidden = !open;
  backdropEl.hidden = !open;
  if (open && !state.wasOpen) refreshAll();
  state.wasOpen = open;
  if (!open) return;

  const tab = ui.libraryTab;
  for (const [key, btn] of Object.entries(els.tabButtons)) btn.classList.toggle("active", key === tab);
  renderTabBody();
}

export function render(container, backdrop) {
  drawerEl = container;
  backdropEl = backdrop;
  if (!built) build(container, backdrop);
  update();
}
