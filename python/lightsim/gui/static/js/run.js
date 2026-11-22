// Run view: device choice, start and stop, progress, field decay and a live
// field slice taken during the run.

import { api, el, decode, fmt, fmtTime, AXES, themeColors, debounce, safeName } from "./util.js";
import { MapPlot } from "./mapplot.js";
import { LinePlot } from "./lineplot.js";
import { buildOverlays, planeAxes } from "./overlays.js";
import { kv } from "./setup.js";

const COMPS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"];

export class RunView {
  constructor(app) {
    this.app = app;
    this.info = app.state.info;
    const d = this.info.run_defaults;
    this.comp = d.comp;
    this.axis = d.axis;
    this.index = { 0: Math.floor(this.info.n[0] / 2), 1: Math.floor(this.info.n[1] / 2), 2: Math.floor(this.info.n[2] / 2) };
    this.index[d.axis] = d.index;
    this.outlines = {};
    this.state = null;
    this.sliceVersion = -1;
    const wrap = document.getElementById("run-figure").parentElement;
    this.decayBox = el("div", { class: "figure short", style: "flex:none;height:190px;position:relative" });
    wrap.after(this.decayBox);
    wrap.dataset.reserve = 200;
    this.plot = new MapPlot(document.getElementById("run-figure"), { emptyText: "Start a run to see the field on this plane." });
    this.decay = new LinePlot(this.decayBox);
    this.buildSide();
    this.buildControls();
    app.onThemeChange(() => { this.drawSlice(); this.drawDecay(); });
    this.connect();
    api("run/state").then((s) => this.update(s));
    this.drawDecay();
  }

  shown() { this.plot.resize(); this.decay.resize(); }

  buildSide() {
    const side = document.getElementById("run-side");
    side.innerHTML = "";
    side.append(el("h3", {}, "Device"));
    this.devSel = el("select", {}, this.info.devices.map((d) => el("option", { value: d.id }, d.label)));
    side.append(this.devSel);
    side.append(el("p", { class: "note" }, "The devices this lightsim build reports. A run here uses the same engine as Simulation.run."));
    this.startBtn = el("button", { class: "primary", onclick: () => this.start() }, "Start");
    this.stopBtn = el("button", { onclick: () => this.stop(), disabled: true }, "Stop");
    side.append(el("div", { style: "display:flex;gap:0.6rem;margin:0.9rem 0 0.4rem" }, this.startBtn, this.stopBtn));
    side.append(el("h3", {}, "Progress"));
    this.statusEl = el("div", { class: "status" }, "Not started");
    this.bar = el("i", { style: "width:0" });
    side.append(this.statusEl, el("div", { class: "progress" }, this.bar));
    this.kvBox = el("div");
    side.append(this.kvBox);
    this.after = el("div");
    side.append(this.after);
    const s = this.info.summary;
    side.append(el("h3", {}, "This simulation"));
    side.append(kv([["cells", s.cells.toLocaleString("en-US")], ["steps", `${s.nsteps.toLocaleString("en-US")}${s.shutoff ? " at most" : ""}`], ["shutoff", s.shutoff ? fmt(s.shutoff, 2) : "none"]]));
    side.append(el("p", { class: "note" }, "The live slice is taken from the device every few tenths of a second while the run goes on; on a GPU each slice copies one field component to the host. Stop ends the run after the current step and keeps what the monitors recorded so far."));
  }

  buildControls() {
    const info = this.info, box = document.getElementById("run-controls");
    box.innerHTML = "";
    this.compSel = el("select", { onchange: () => { this.comp = this.compSel.value; this.sendPlane(); } }, COMPS.map((c) => el("option", { value: c }, c)));
    this.compSel.value = this.comp;
    box.append(el("span", { class: "group" }, el("label", {}, "Field"), this.compSel));
    const inv = info.invariant;
    const planes = [[2, "xy"], [1, "xz"], [0, "yz"]].filter(([a]) => inv === null || inv === undefined || a === inv);
    this.planeLinks = planes.map(([a, name]) => el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.axis = a; this.updatePlane(); this.sendPlane(); } }, name));
    box.append(el("span", { class: "group" }, el("label", {}, "Plane"), el("span", { class: "seg" }, this.planeLinks)));
    this.slider = el("input", { type: "range", min: 0, max: 1, step: 1 });
    this.posText = el("span", { class: "value" });
    const deb = debounce(() => this.sendPlane(), 120);
    this.slider.addEventListener("input", () => { this.index[this.axis] = +this.slider.value; this.updatePos(); deb(); });
    this.posGroup = el("span", { class: "group" }, el("label", {}, "Position"), this.slider, this.posText);
    box.append(this.posGroup);
    this.holdChk = el("input", { type: "checkbox" });
    box.append(el("label", {}, this.holdChk, "hold colour scale"));
    box.append(el("span", { class: "links" },
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.plot.exportPNG(safeName(`${info.title}_live_${this.comp}`)); } }, "PNG"),
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.plot.exportSVG(safeName(`${info.title}_live_${this.comp}`)); } }, "SVG")));
    this.updatePlane();
  }

  updatePlane() {
    this.planeLinks.forEach((l) => l.classList.toggle("on", l.textContent === ["yz", "xz", "xy"][this.axis]));
    const n = this.info.n[this.axis];
    this.slider.max = n - 1;
    this.slider.value = this.index[this.axis];
    this.posGroup.hidden = n <= 1;
    this.updatePos();
  }

  updatePos() {
    const a = this.axis, i = this.index[a], nodes = this.info.nodes[a];
    this.posText.textContent = `${AXES[a]} = ${nodes[i].toFixed(3)} µm`;
  }

  async sendPlane() {
    if (this.state && (this.state.status === "running" || this.state.status === "compiling")) {
      await api("run/plane", { comp: this.comp, axis: this.axis, index: this.index[this.axis] }).catch(() => {});
    }
  }

  async start() {
    this.startBtn.disabled = true;
    this.held = null;
    try {
      const s = await api("run/start", { device: this.devSel.value, comp: this.comp, axis: this.axis, index: this.index[this.axis] });
      this.update(s);
    } catch (e) {
      this.startBtn.disabled = false;
      this.statusEl.textContent = e.message;
      this.statusEl.className = "status error";
    }
  }

  async stop() {
    this.stopBtn.disabled = true;
    this.update(await api("run/stop", {}));
  }

  connect() {
    if (this.es) this.es.close();
    this.es = new EventSource("api/run/events");
    this.es.onmessage = (e) => this.update(JSON.parse(e.data));
    this.es.onerror = () => { /* the browser reconnects */ };
  }

  async update(s) {
    const prev = this.state;
    this.state = s;
    const active = s.status === "running" || s.status === "compiling";
    this.startBtn.disabled = active;
    this.stopBtn.disabled = !active || s.error;
    const label = { idle: "Not started", compiling: "Compiling the plan…", running: `Running on ${s.device}`, done: "Finished", stopped: "Stopped", error: "Failed" }[s.status];
    this.statusEl.textContent = label;
    this.statusEl.className = "status " + (s.status === "compiling" ? "running" : s.status);
    const frac = s.nsteps ? s.step / s.nsteps : 0;
    this.bar.style.width = `${(100 * Math.min(1, frac)).toFixed(1)}%`;
    const rows = [];
    if (s.status !== "idle") {
      rows.push(["step", `${s.step.toLocaleString("en-US")} / ${s.nsteps.toLocaleString("en-US")}`]);
      rows.push(["simulated time", `${(s.time * 1e15).toFixed(2)} fs`]);
      rows.push(["field decay", s.decay === null || s.decay === undefined ? "–" : fmt(s.decay, 3)]);
      rows.push(["shutoff at", s.shutoff ? fmt(s.shutoff, 2) : "none"]);
      rows.push(["elapsed", fmtTime(s.status === "compiling" ? s.wall : s.elapsed)]);
      if (active) rows.push(["remaining, at most", fmtTime(s.remaining_max)]);
      if (s.elapsed > 0 && s.step > 0) rows.push(["speed", `${((this.info.summary.cells * s.step) / s.elapsed / 1e6).toFixed(0)} Mcells/s`]);
    }
    this.kvBox.innerHTML = "";
    this.kvBox.append(kv(rows));
    this.after.innerHTML = "";
    if (s.error) this.after.append(el("div", { class: "warn" }, s.error));
    if ((s.status === "done" || s.status === "stopped") && (!prev || prev.status !== s.status)) {
      await this.app.refreshResults();
    }
    if (s.status === "done" || s.status === "stopped") {
      this.after.append(el("p", {}, el("a", { href: "#results" }, "Open the results")));
      if (s.status === "stopped") this.after.append(el("p", { class: "note" }, "The run was stopped: spectra hold what was recorded up to that step."));
    }
    this.drawDecay();
    if (s.slice_version !== this.sliceVersion && s.slice_version > 0) {
      this.sliceVersion = s.slice_version;
      this.slice = await api("run/slice");
      this.drawSlice();
    }
  }

  async outlineFor(axis, index) {
    const k = `${axis}/${index}`;
    if (!this.outlines[k]) this.outlines[k] = api(`outline?axis=${axis}&index=${index}`).then((r) => r.segments);
    return this.outlines[k];
  }

  async drawSlice() {
    const s = this.slice;
    if (!s || !s.data) return;
    const colors = themeColors();
    const [u, v] = planeAxes(s.axis);
    let lim = Math.max(Math.abs(s.min), Math.abs(s.max)) || 1;
    if (this.holdChk.checked) { this.held = this.held ? Math.max(this.held, lim) : lim; lim = this.held; } else this.held = null;
    const nodes = this.info.nodes[s.axis];
    const cut = { axis: s.axis, coord_um: nodes[s.index], coord_hi_um: nodes[s.index + 1] };
    const segs = await this.outlineFor(s.axis, s.index);
    const ov = [{ t: "segments", segs, color: colors.text, width: 0.9, opacity: 0.55 }, ...buildOverlays(this.info, cut, { sources: true, monitors: true, pml: true, sym: false, mesh: false, ports: false }, null, colors)];
    this.plot.set({
      data: decode(s.data), nu: s.nu, nv: s.nv, uEdges: s.u_edges, vEdges: s.v_edges,
      uTitle: `${AXES[u]} (µm)`, vTitle: `${AXES[v]} (µm)`, uName: AXES[u], vName: AXES[v],
      cmap: "diverging", vmin: -lim, vmax: lim, cbTitle: `${s.comp} (V/m)`, valueName: s.comp, unit: "V/m", keepView: true, overlays: ov,
      title: `${s.comp} at step ${s.step.toLocaleString("en-US")}, t = ${(s.time * 1e15).toFixed(2)} fs${this.info.dims === 2 ? "" : `, ${AXES[s.axis]} = ${nodes[s.index].toFixed(3)} µm`}`,
    });
    const isH = s.comp.startsWith("H");
    document.getElementById("run-caption").textContent = `Instantaneous ${s.comp}${isH ? " (A/m times 376.73 Ω, in V/m)" : ""} on the plane, signed; the colour scale is symmetric about zero${this.holdChk.checked ? " and held at the largest value seen" : ", ± the largest value of this slice"}. Lines: medium boundaries, sources, monitors; hatched: absorbing layers.${s.stride[0] > 1 || s.stride[1] > 1 ? ` Every ${s.stride[0]} × ${s.stride[1]} cells.` : ""}`;
  }

  drawDecay() {
    const s = this.state;
    const colors = themeColors();
    const hist = (s && s.decay_hist) || [];
    this.decay.set({
      series: hist.length ? [{ name: "field energy / peak", x: hist.map((h) => h[0]), y: hist.map((h) => h[1]), color: colors.accent, markers: false }] : [],
      yScale: "log", xTitle: "step", yTitle: "energy / peak", xName: "step", hlines: s && s.shutoff ? [{ y: s.shutoff, label: `shutoff ${fmt(s.shutoff, 2)}` }] : [],
      emptyText: "The field decay appears here during a run (sampled every shutoff interval).", keepZoom: true,
    });
  }
}
