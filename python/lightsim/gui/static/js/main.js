// Shell: theme, navigation between the three views, shared state.

import { api, toast } from "./util.js";
import { SetupView } from "./setup.js";
import { RunView } from "./run.js";
import { ResultsView } from "./results.js";

const state = { info: null, views: {}, current: null, listeners: [] };
export const app = {
  state,
  onThemeChange(f) { state.listeners.push(f); },
  async refreshResults() {
    if (state.views.results) await state.views.results.load();
    updateNav();
  },
  go(view) { location.hash = "#" + view; },
};

// theme: follows the system until the user picks one -------------------------
function systemDark() { return matchMedia("(prefers-color-scheme: dark)").matches; }
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  document.getElementById("theme-toggle").textContent = t === "dark" ? "Light theme" : "Dark theme";
  for (const f of state.listeners) f();
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("lightsim-theme"); } catch { /* storage unavailable */ }
  applyTheme(saved || (systemDark() ? "dark" : "light"));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    let s = null;
    try { s = localStorage.getItem("lightsim-theme"); } catch { /* */ }
    if (!s) applyTheme(systemDark() ? "dark" : "light");
  });
  document.getElementById("theme-toggle").addEventListener("click", (e) => {
    e.preventDefault();
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("lightsim-theme", t); } catch { /* */ }
    applyTheme(t);
  });
}

function updateNav() {
  const info = state.info;
  for (const a of document.querySelectorAll("#nav a")) {
    const v = a.dataset.view;
    a.classList.toggle("active", v === state.current);
    a.classList.toggle("disabled", (v === "setup" || v === "run") && !info.has_spec);
  }
}

function show() {
  let v = (location.hash || "").slice(1) || (state.info.has_spec ? "setup" : "results");
  if (!["setup", "run", "results"].includes(v)) v = "setup";
  if (!state.info.has_spec && v !== "results") v = "results";
  state.current = v;
  for (const s of document.querySelectorAll(".view")) s.hidden = s.id !== "view-" + v;
  updateNav();
  if (!state.views[v]) {
    const C = { setup: SetupView, run: RunView, results: ResultsView }[v];
    state.views[v] = new C(app);
  }
  state.views[v].shown && state.views[v].shown();
}

async function main() {
  initTheme();
  try {
    state.info = await api("info");
  } catch (e) {
    document.querySelector("main").textContent = "Could not reach the lightsim server: " + e.message;
    return;
  }
  const t = state.info.title || "";
  document.getElementById("sim-title").textContent = t;
  document.title = t ? `${t} · lightsim` : "lightsim";
  window.addEventListener("hashchange", show);
  show();
}

window.addEventListener("error", (e) => toast(String(e.message || e)));
window.addEventListener("unhandledrejection", (e) => toast(String(e.reason && e.reason.message ? e.reason.message : e.reason)));
main();
