// Heliostat stage: reports the current aperture and lets the field-wide
// surface figure be toggled. Aperture shape/dimensions are edited on the
// Heliostat Shape tab, not here.
import { store } from "../store.js";
import { segButton, HELIOSTAT_SURFACE_OPTIONS, apertureSummaryText } from "../fields.js";

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
  h2.textContent = "Heliostat";
  const editLink = document.createElement("a");
  editLink.href = "#";
  editLink.textContent = "Edit shape…";
  editLink.style.fontSize = "11.5px";
  editLink.addEventListener("click", (e) => {
    e.preventDefault();
    store.set("ui.tab", "shape");
  });
  head.appendChild(chev);
  head.appendChild(h2);
  head.appendChild(editLink);
  head.addEventListener("click", (e) => {
    if (e.target === editLink) return;
    const ui = store.get("ui");
    store.set("ui.expanded.heliostat", !ui.expanded.heliostat);
  });

  const summary = document.createElement("div");
  summary.className = "summary";

  const body = document.createElement("div");
  body.className = "stagebody";

  const surfaceSeg = document.createElement("div");
  surfaceSeg.className = "seg";
  const surfaceBtns = {};
  const SURFACE_TOOLTIPS = {
    twisting: "Re-solves each facet's figure as the sun moves, so it stays perfectly focused at every instant.",
    spherical: "Freezes one figure — a long focal gives a weakly focusing, not-quite-flat facet that no longer re-solves with the sun.",
    flat: "No curvature by default — a true flat panel, though a facet grid can still be given a gentle fixed curvature.",
  };
  for (const [key, label] of HELIOSTAT_SURFACE_OPTIONS) {
    surfaceBtns[key] = segButton(
      surfaceSeg,
      label,
      key === "twisting",
      () => store.set("doc.design.surface", key),
      SURFACE_TOOLTIPS[key]
    );
  }
  body.appendChild(surfaceSeg);

  container.appendChild(head);
  container.appendChild(summary);
  container.appendChild(body);

  els = { chev, summary, body, surfaceBtns };
  built = true;
}

export function render(container) {
  if (!built) build(container);
  const doc = store.get("doc");
  const ui = store.get("ui");

  els.body.style.display = ui.expanded.heliostat ? "" : "none";
  els.chev.style.transform = ui.expanded.heliostat ? "rotate(90deg)" : "";

  const surface = doc.design.surface;
  for (const [key, btn] of Object.entries(els.surfaceBtns)) {
    btn.classList.toggle("active", key === surface);
  }

  els.summary.innerHTML = `<strong>${apertureSummaryText(doc)}</strong>`;
}
