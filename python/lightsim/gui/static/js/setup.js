// Set-up view: cuts of the rasterised material with overlays, the list of
// structures, sources and monitors, and a summary of the grid and the run.

import { api, el, decode, fmt, fmtBytes, fmtCount, C0, AXES, download, csv, safeName, debounce, themeColors } from "./util.js";
import { MapPlot } from "./mapplot.js";
import { buildOverlays, planeAxes } from "./overlays.js";

export class SetupView {
  constructor(app) {
    this.app = app;
    this.info = app.state.info;
    const info = this.info;
    const inv = info.invariant;
    this.axis = inv !== null && inv !== undefined ? inv : 2;
    this.index = {};
    for (const a of [0, 1, 2]) this.index[a] = Math.floor(info.n[a] / 2);
    // start the xy cut at the height of the first structure that is not a wafer slab, or the first source
    const d = info.run_defaults;
    if (d && d.axis === 2) this.index[2] = d.index;
    if (inv !== null && inv !== undefined) this.index[inv] = 0;
    this.q = "n";
    this.show = { sources: true, monitors: true, pml: true, sym: true, mesh: false, ports: true };
    this.selected = null;
    this.plot = new MapPlot(document.getElementById("setup-figure"), { emptyText: "loading the cut…" });
    this.buildControls();
    this.buildSide();
    app.onThemeChange(() => this.redraw());
    this.load();
  }

  shown() { this.plot.resize(); }

  buildControls() {
    const info = this.info, box = document.getElementById("setup-controls");
    box.innerHTML = "";
    const inv = info.invariant;
    const planes = [[2, "xy"], [1, "xz"], [0, "yz"]].filter(([a]) => inv === null || inv === undefined || a === inv);
    this.planeLinks = planes.map(([a, name]) => el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.setAxis(a); } }, name));
    box.append(el("span", { class: "group" }, el("label", {}, "Plane"), el("span", { class: "seg" }, this.planeLinks)));
    if (inv !== null && inv !== undefined) box.lastChild.append(el("span", { class: "note" }, ` 2-D, invariant along ${AXES[inv]}`));
    this.slider = el("input", { type: "range", min: 0, max: 1, step: 1, value: 0 });
    this.posText = el("span", { class: "value" });
    const deb = debounce(() => this.load(), 60);
    this.slider.addEventListener("input", () => { this.index[this.axis] = +this.slider.value; this.updatePos(); deb(); });
    this.posGroup = el("span", { class: "group" }, el("label", {}, "Position"), this.slider, this.posText);
    box.append(this.posGroup);
    this.qSel = el("select", { onchange: () => { this.q = this.qSel.value; this.load(); } }, info.quantities.map((q) => el("option", { value: q.id }, q.label)));
    box.append(el("span", { class: "group" }, el("label", {}, "Show"), this.qSel));
    const chk = (k, label) => {
      const c = el("input", { type: "checkbox" });
      c.checked = this.show[k];
      c.addEventListener("change", () => { this.show[k] = c.checked; this.redraw(); });
      return el("label", {}, c, label);
    };
    const over = el("span", { class: "group" }, chk("sources", "sources"), chk("monitors", "monitors"), chk("pml", "absorbing layers"), chk("sym", "symmetry"), chk("mesh", "mesh"));
    if (info.ports && info.ports.length) over.append(chk("ports", "ports"));
    box.append(over);
    box.append(el("span", { class: "links" },
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.plot.exportPNG(this.fileName()); } }, "PNG"),
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.plot.exportSVG(this.fileName()); } }, "SVG"),
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.exportCSV(); } }, "CSV")));
    this.updatePlane();
  }

  fileName() { return safeName(`${this.info.title}_${this.q}_${["yz", "xz", "xy"][this.axis]}_${this.index[this.axis]}`); }

  setAxis(a) { this.axis = a; this.updatePlane(); this.load(); }

  updatePlane() {
    const info = this.info;
    this.planeLinks.forEach((l) => l.classList.toggle("on", l.textContent === ["yz", "xz", "xy"][this.axis]));
    const n = info.n[this.axis];
    this.slider.max = n - 1;
    this.slider.value = this.index[this.axis];
    this.posGroup.hidden = n <= 1;
    this.updatePos();
  }

  updatePos() {
    const a = this.axis, i = this.index[a], nodes = this.info.nodes[a];
    this.posText.textContent = `${AXES[a]} = ${nodes[i].toFixed(3)} µm  (cell ${i + 1} of ${nodes.length - 1})`;
  }

  async load() {
    const a = this.axis, i = this.index[a], q = this.q;
    const key = `${a}/${i}/${q}`;
    this.want = key;
    let cut;
    try {
      cut = await api(`cut?axis=${a}&index=${i}&q=${q}`);
    } catch (e) {
      this.plot.m = null; this.plot.opts.emptyText = e.message; this.plot.render();
      return;
    }
    if (this.want !== key) return;
    this.cut = cut;
    this.redraw(true);
  }

  redraw(newData = false) {
    const cut = this.cut;
    if (!cut) return;
    const colors = themeColors();
    const qinfo = this.info.quantities.find((x) => x.id === cut.quantity);
    const [u, v] = planeAxes(cut.axis);
    let vmin = cut.min, vmax = cut.max;
    if (vmax <= vmin) { vmax = vmin + (Math.abs(vmin) || 1) * 0.01; }
    const present = cut.categories ? [...new Set(Array.from(decodeOnce(cut)).filter(Number.isFinite).map(Math.round))].sort((x, y) => x - y) : null;
    const m = {
      data: decodeOnce(cut), nu: cut.nu, nv: cut.nv, uEdges: cut.u_edges, vEdges: cut.v_edges,
      uTitle: `${AXES[u]} (µm)`, vTitle: `${AXES[v]} (µm)`, uName: AXES[u], vName: AXES[v],
      cmap: "viridis", vmin, vmax, categories: cut.categories, present, unit: cut.unit,
      cbTitle: qinfo ? shortLabel(qinfo) : cut.quantity, valueName: qinfo ? shortLabel(qinfo) : "value", nanText: "PEC",
      keepView: !newData || (this.plot.m && this.plot.m.axis === cut.axis), axis: cut.axis,
      cellLabel: (iu, iv) => `cell (${AXES[u]} ${iu * cut.stride[0] + 1}, ${AXES[v]} ${iv * cut.stride[1] + 1})`,
    };
    if (cut.categories && cut.quantity !== "medium") m.catColors = [colors.nan, colors.warn];
    m.overlays = buildOverlays(this.info, cut, this.show, this.selected, colors);
    const parts = [];
    if (this.info.dims === 2) parts.push(`The plane of the 2-D simulation (invariant along ${AXES[cut.axis]}).`);
    else parts.push(`Cut normal to ${AXES[cut.axis]} through cell ${cut.index + 1} (${AXES[cut.axis]} = ${cut.coord_um.toFixed(4)} µm).`);
    const what = { n: "Index √ε, ε the mean of the three E samples of the cell,", medium: "Medium that owns each E sample", sigma: "Conductivity of the medium of each sample,", dispersive: "Cells with a dispersive medium", nonlinear: "Cells with a nonlinear medium" }[cut.quantity] || `Permittivity of the ${cut.quantity.slice(-1)} component of E,`;
    parts.push(cut.quantity === "medium" || cut.quantity === "dispersive" || cut.quantity === "nonlinear" ? what + "." : what + " as the solver uses it" + (this.info.summary.subpixel ? ", after sub-pixel smoothing." : " (sub-pixel smoothing is off for this simulation)."));
    if (cut.stride[0] > 1 || cut.stride[1] > 1) parts.push(`Shown every ${cut.stride[0]} × ${cut.stride[1]} cells.`);
    const inPml = this.info.boundaries["xyz"[cut.axis]].some((s) => (s.type === "pml" || s.type === "absorber") && cut.coord_um >= s.min - 1e-9 && cut.coord_hi_um <= s.max + 1e-9);
    if (inPml) parts.push("This cut lies in an absorbing layer.");
    parts.push("Wheel to zoom, drag to pan, double-click to reset.");
    document.getElementById("setup-caption").textContent = parts.join(" ");
    this.plot.set(m);
    if (this.show.mesh && this.plot.meshEvery > 1) {
      document.getElementById("setup-caption").textContent += ` Mesh: every ${this.plot.meshEvery}th line at this zoom.`;
    }
  }

  exportCSV() {
    const cut = this.cut;
    if (!cut) return;
    const d = decodeOnce(cut);
    const [u, v] = planeAxes(cut.axis);
    const cu = cut.u_edges.slice(0, -1).map((x, i) => (x + cut.u_edges[i + 1]) / 2);
    const cv = cut.v_edges.slice(0, -1).map((x, i) => (x + cut.v_edges[i + 1]) / 2);
    const lines = [`# ${cut.quantity} on the cut normal to ${AXES[cut.axis]} at ${cut.coord_um} um; rows: ${AXES[u]} (um), columns: ${AXES[v]} (um)`];
    lines.push([`${AXES[u]}\\${AXES[v]}`, ...cv.map((x) => x.toFixed(5))].join(","));
    for (let i = 0; i < cut.nu; i++) {
      const row = [cu[i].toFixed(5)];
      for (let j = 0; j < cut.nv; j++) { const x = d[i * cut.nv + j]; row.push(Number.isFinite(x) ? x.toPrecision(7) : ""); }
      lines.push(row.join(","));
    }
    download(this.fileName() + ".csv", lines.join("\n") + "\n", "text/csv");
  }

  // -- side panel ------------------------------------------------------------
  buildSide() {
    const info = this.info, s = info.summary, side = document.getElementById("setup-side");
    side.innerHTML = "";
    side.append(el("h3", {}, "Summary"));
    const n = s.n;
    const inv = info.invariant;
    const rows = [
      ["grid", n.map((x, i) => (i === inv ? "1" : x)).join(" × ")],
      ["cells", fmtCount(s.cells)],
      ["cell size", `${fmtRange(s.min_step_um, s.max_step_um, inv)} nm`],
      ["time step", `${(s.dt * 1e15).toPrecision(4)} fs`],
      ["steps", `${s.nsteps.toLocaleString("en-US")}${s.shutoff ? " at most" : ""}`],
      ["run time", `${(s.run_time * 1e15).toPrecision(4)} fs`],
      ["shutoff", s.shutoff ? `${fmt(s.shutoff, 2)} (every ${s.shutoff_interval} steps)` : "none"],
      ["memory, 1 device", fmtBytes(s.memory.per_device["1"])],
      ["memory, 2 devices", `${fmtBytes(s.memory.per_device["2"])} each`],
    ];
    if (s.band) {
      rows.push(["wavelengths", `${trim(s.band.lambda_min_um * 1e3)} – ${trim(s.band.lambda_max_um * 1e3)} nm`]);
      rows.push(["frequencies", `${(s.band.f_min / 1e12).toFixed(2)} – ${(s.band.f_max / 1e12).toFixed(2)} THz`]);
    }
    if (s.densest) {
      rows.push(["densest medium", `${s.densest.medium}, n = ${s.densest.n.toFixed(3)}`]);
      rows.push(["cells per λ there", `${s.densest.cells_per_wavelength.toFixed(1)}`]);
    }
    if (s.steps_per_period) rows.push(["steps per period", s.steps_per_period.toFixed(1)]);
    side.append(kv(rows));
    if (s.band) side.append(el("p", { class: "note" }, `Band from the ${s.band.from}; cells per wavelength at the shortest wavelength, from the largest step of the cells that overlap the densest medium. Memory: field and coefficient arrays, absorbing layers and monitors (${s.memory.bytes_per_real * 8}-bit numbers), without dispersive auxiliary fields.`));
    for (const w of s.warnings) side.append(el("div", { class: "warn" }, w));

    const colors = themeColors();
    const section = (title, items, color, kindOf) => {
      if (!items.length) return;
      side.append(el("h3", {}, `${title} (${items.length})`));
      const ul = el("ul", { class: "list" });
      for (const it of items) {
        const li = el("li", { "data-id": it.id, onclick: () => this.select(it) },
          color ? el("span", { class: "dot", style: `background:${color}` }) : null,
          el("span", {}, it.name), el("span", { class: "kind" }, kindOf(it)));
        ul.append(li);
      }
      side.append(ul);
    };
    section("Structures", info.structures, null, (s) => `${s.shape}, ${s.medium}`);
    section("Sources", info.sources, colors.src, (s) => s.type.replace(/_/g, " "));
    section("Monitors", info.monitors, colors.mon, (m) => m.type.replace(/_/g, " "));
    this.paramBox = el("div", { class: "params", hidden: true });
    side.append(this.paramBox);
    side.append(el("h3", {}, "Boundaries"));
    const brows = [];
    for (const a of [0, 1, 2]) {
      const b = info.boundaries[AXES[a]];
      const txt = b.map((x) => (x.type === "pml" ? `PML ${x.layers}${x.inside ? " inside" : ""}` : x.type === "absorber" ? `absorber ${x.layers}` : x.type === "bloch" ? `Bloch k=${fmt(x.k, 3)}` : x.type)).join(" / ");
      brows.push([AXES[a], a === inv ? "invariant (periodic)" : txt]);
    }
    side.append(kv(brows));
    side.append(el("h3", {}, "Media"));
    side.append(kv(info.media.map((m) => [m.name, m.pec ? "PEC" : `n = ${Math.sqrt(Math.max(...m.eps)).toFixed(4)}${m.dispersive ? ", dispersive" : ""}${m.lossy ? ", lossy" : ""}${m.nonlinear ? ", nonlinear" : ""}${m.anisotropic ? ", anisotropic" : ""}`])));
    side.append(el("p", { class: "note" }, `lightsim ${info.version}, ${info.precision}`));
  }

  select(it) {
    this.selected = this.selected === it.id ? null : it.id;
    for (const li of document.querySelectorAll("#setup-side li")) li.classList.toggle("sel", li.dataset.id === this.selected);
    const pb = this.paramBox;
    if (!this.selected) { pb.hidden = true; this.redraw(); return; }
    pb.hidden = false;
    pb.innerHTML = "";
    pb.append(el("div", { class: "head", style: "margin-bottom:0.3rem" }, it.name));
    const rows = Object.entries(it.params).map(([k, v]) => [k, Array.isArray(v) ? v.map((x) => (typeof x === "number" ? fmt(x, 5) : x)).join(", ") : typeof v === "object" ? JSON.stringify(v) : typeof v === "number" ? fmt(v, 6) : String(v)]);
    pb.append(kv(rows));
    pb.append(el("p", { class: "note" }, "Lengths in µm."));
    // move the cut to the object if it does not meet it
    const g = it.geometry || it.outline;
    if (g && g.min && this.info.n[this.axis] > 1) {
      const a = this.axis, nodes = this.info.nodes[a], i = this.index[a];
      const lo = nodes[i], hi = nodes[i + 1];
      const gmin = Math.max(g.min[a], nodes[0]), gmax = Math.min(g.max[a], nodes[nodes.length - 1]);
      if (!(gmin <= hi && gmax >= lo)) {
        const c = (gmin + gmax) / 2;
        let k = nodes.findIndex((x) => x > c) - 1;
        if (k < 0) k = 0;
        this.index[a] = Math.min(k, nodes.length - 2);
        this.updatePlane();
        this.load();
        return;
      }
    }
    this.redraw();
  }
}

const cache = new WeakMap();
function decodeOnce(cut) {
  if (!cache.has(cut)) cache.set(cut, decode(cut.data));
  return cache.get(cut);
}

function kv(rows) {
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of rows) dl.append(el("dt", {}, k), el("dd", {}, v));
  return dl;
}

function fmtRange(lo, hi, inv) {
  const a = lo.filter((_, i) => i !== inv), b = hi.filter((_, i) => i !== inv);
  const mn = Math.min(...a) * 1e3, mx = Math.max(...b) * 1e3;
  return Math.abs(mx - mn) < 0.05 ? mn.toFixed(1) : `${mn.toFixed(1)} – ${mx.toFixed(1)}`;
}

function shortLabel(q) {
  return { n: "index n", eps_x: "ε of Ex", eps_y: "ε of Ey", eps_z: "ε of Ez", sigma: "σ (S/m)", medium: "medium", dispersive: "dispersive", nonlinear: "nonlinear" }[q.id] || q.label;
}

function trim(v) {
  return Number(v.toFixed(1)).toString();
}

export { kv, decodeOnce, C0 };
