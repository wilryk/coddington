// Sun stage: azimuth + elevation (docs/ui-spec.md 2.2). Site & time entry
// is a later phase; Phase 3a is the direct az/el pair the trace endpoints
// take verbatim. Field descriptors live in ../fields.js so the floating
// inspector (../inspector.js) can render the identical rows when the sun
// is selected in scene (docs/ui-spec.md 2.4).
import { store } from "../store.js";
import { numberRow, setVal, SUN_FIELDS } from "../fields.js";

let built = false;
let els = {};

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
  const az = numberRow(body, SUN_FIELDS[0]);
  const el = numberRow(body, SUN_FIELDS[1]);
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = "Site & time entry -- coming in a later phase.";
  body.appendChild(hint);

  container.appendChild(head);
  container.appendChild(body);

  els = { chev, body, az, el };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");

  els.body.style.display = ui.expanded.sun ? "" : "none";
  els.chev.style.transform = ui.expanded.sun ? "rotate(90deg)" : "";

  setVal(els.az, doc.sun.az);
  setVal(els.el, doc.sun.el);
}
