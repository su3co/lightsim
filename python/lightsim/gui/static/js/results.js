// Results view: what each monitor holds, as line plots and maps with
// read-outs, linear / dB scales and CSV, PNG and SVG export.

import { api, el, decode, fmt, fmtTime, C0, AXES, download, csv, safeName, themeColors, SERIES } from "./util.js";
import { LinePlot } from "./lineplot.js";
import { MapPlot } from "./mapplot.js";
import { planeAxes } from "./overlays.js";
import { kv } from "./setup.js";

const TYPE_LABEL = {
  flux: "flux", flux_box: "flux box", mode: "mode monitor", dft_field: "field (frequency domain)", field_time: "field (time domain)",
  probe: "probe", far_field: "far field", energy: "energy", time: "time series",
};

export class ResultsView {
  constructor(app) {
    this.app = app;
    this.info = app.state.info;
    this.xUnit = "nm"; // or "THz"
    this.scale = "lin";
    this.figs = [];
    this.sel = null;
    app.onThemeChange(() => this.select(this.sel, true));
    this.load();
  }

  shown() { for (const f of this.figs) f.resize(); }

  async load() {
    this.index = await api("results");
    this.buildSide();
    if (!this.index.available) {
      this.stage().innerHTML = "";
      this.stage().append(el("div", { class: "figure-wrap" }, el("div", { class: "figure" }, el("div", { class: "empty" }, this.info.has_spec ? "No results yet. Start a run in the Run view, or pass results to view()." : "No results."))));
      return;
    }
    const keep = this.sel && this.items.find((i) => i.key === this.sel.key);
    this.select(keep || this.items.find((i) => i.kind === "sparams") || this.items.find((i) => i.kind !== "run") || this.items[0]);
  }

  stage() { return document.getElementById("results-stage"); }

  buildSide() {
    const side = document.getElementById("results-side");
    side.innerHTML = "";
    this.items = [];
    if (!this.index.available) { side.append(el("p", { class: "note" }, "Nothing to show yet.")); return; }
    const ix = this.index;
    const add = (group, item) => { item.group = group; this.items.push(item); };
    add("Run", { key: "run", kind: "run", name: "Run summary" });
    if (ix.sparams && ix.sparams.monitors.length && ix.sparams.reference) add("Run", { key: "sparams", kind: "sparams", name: "S-parameters" });
    for (const m of ix.monitors) add("Monitors", { key: "m:" + m.name, kind: m.type, name: m.name, m });
    let group = null, ul = null;
    for (const it of this.items) {
      if (it.group !== group) {
        group = it.group;
        side.append(el("h3", {}, group));
        ul = el("ul", { class: "list" });
        side.append(ul);
      }
      it.li = el("li", { onclick: () => this.select(it) }, el("span", {}, it.name), it.m ? el("span", { class: "kind" }, TYPE_LABEL[it.kind] || it.kind) : null);
      ul.append(it.li);
    }
    side.append(el("h3", {}, "Spectral axis"));
    const seg = el("span", { class: "seg" });
    for (const [k, label] of [["nm", "wavelength (nm)"], ["THz", "frequency (THz)"]]) {
      const a = el("a", { href: "#", class: this.xUnit === k ? "on" : "", onclick: (e) => { e.preventDefault(); this.xUnit = k; seg.querySelectorAll("a").forEach((x) => x.classList.toggle("on", x === a)); this.select(this.sel, true); } }, label);
      seg.append(a);
    }
    side.append(seg);
  }

  clear() {
    for (const f of this.figs) f.destroy();
    this.figs = [];
    const s = this.stage();
    s.innerHTML = "";
    return s;
  }

  header(title, sub) {
    const s = this.stage();
    s.append(el("h2", { class: "view-title" }, title));
    if (sub) s.append(el("p", { class: "sub" }, sub));
  }

  controls() {
    const c = el("div", { class: "controls" });
    this.stage().append(c);
    return c;
  }

  figure(cls = "") {
    const plots = this.stage().querySelector(".plots") || (() => { const p = el("div", { class: "plots" }); this.stage().append(p); return p; })();
    const d = el("div", { class: "figure " + cls });
    plots.append(d);
    return d;
  }

  exportLinks(fig, name, csvFn) {
    const l = el("span", { class: "links" });
    if (csvFn) l.append(el("a", { href: "#", onclick: (e) => { e.preventDefault(); download(safeName(name) + ".csv", csvFn(), "text/csv"); } }, "CSV"));
    l.append(el("a", { href: "#", onclick: (e) => { e.preventDefault(); fig.exportPNG(safeName(name)); } }, "PNG"));
    l.append(el("a", { href: "#", onclick: (e) => { e.preventDefault(); fig.exportSVG(safeName(name)); } }, "SVG"));
    return l;
  }

  seg(options, value, onchange) {
    const s = el("span", { class: "seg" });
    for (const [k, label] of options) {
      const a = el("a", { href: "#", class: k === value ? "on" : "", onclick: (e) => { e.preventDefault(); s.querySelectorAll("a").forEach((x) => x.classList.toggle("on", x === a)); onchange(k); } }, label);
      s.append(a);
    }
    return s;
  }

  // spectral x axis
  spectral(freqs) {
    const nm = this.xUnit === "nm";
    return {
      x: freqs.map((f) => (nm ? (C0 / f) * 1e9 : f / 1e12)),
      title: nm ? "wavelength (nm)" : "frequency (THz)",
      name: nm ? "λ" : "f",
      unit: nm ? "nm" : "THz",
      format: (v, full) => (full ? (nm ? v.toFixed(2) : v.toFixed(4)) : undefined),
    };
  }

  select(it, force = false) {
    if (!it) return;
    if (!force && this.sel && this.sel.key === it.key && this.figs.length) return;
    this.sel = it;
    for (const x of this.items) x.li && x.li.classList.toggle("sel", x === it);
    this.clear();
    const k = it.kind;
    const p = k === "run" ? this.showRun()
      : k === "sparams" ? this.showSparams()
        : k === "flux" || k === "flux_box" ? this.showFlux(it.m)
          : k === "mode" ? this.showMode(it.m)
            : k === "dft_field" ? this.showMap(it.m)
              : k === "probe" || k === "field_time" || k === "time" ? this.showTime(it.m)
                : k === "far_field" ? this.showFar(it.m)
                  : k === "energy" ? this.showEnergy(it.m) : null;
    if (p && p.catch) p.catch((e) => this.stage().append(el("div", { class: "warn" }, e.message)));
  }

  // -- run summary ----------------------------------------------------------------
  showRun() {
    const st = this.index.stats;
    this.header("Run summary", "What the engine reports for the run these results come from.");
    const rows = [["device", st.backend], ["cells", Number(st.cells).toLocaleString("en-US")], ["steps", Number(st.steps).toLocaleString("en-US")],
      ["wall time", fmtTime(st.seconds)], ["speed", `${st.mcells_per_s.toFixed(0)} Mcells/s`], ["ended", st.stopped_early ? "early (shutoff reached, or stopped)" : "at run_time"]];
    if (st.dt) rows.push(["simulated time", `${(st.dt * st.steps * 1e15).toFixed(2)} fs`]);
    const box = el("div", { style: "max-width:30rem" });
    box.append(kv(rows));
    this.stage().append(box);
    const f = new LinePlot(this.figure("tall"));
    this.figs.push(f);
    const e = st.energy || [], mx = Math.max(...e, 0);
    const c = this.controls();
    c.append(this.exportLinks(f, `${this.info.title}_decay`, () => csv(["step", "energy_over_peak"], [st.energy_steps, e.map((v) => v / mx)])));
    f.set({ series: e.length ? [{ name: "field energy / peak", x: st.energy_steps, y: e.map((v) => v / mx), color: themeColors().accent }] : [], yScale: "log", xTitle: "step", yTitle: "energy / peak", xName: "step", hlines: st.shutoff ? [{ y: st.shutoff, label: `shutoff ${fmt(st.shutoff, 2)}` }] : [], emptyText: "no energy samples" });
  }

  // -- flux -------------------------------------------------------------------
  async showFlux(m) {
    const d = await api(`results/flux?name=${encodeURIComponent(m.name)}`);
    this.header(m.name, `${TYPE_LABEL[m.type] || m.type}: power through the ${m.type === "flux_box" ? "closed box (positive outwards)" : "plane (positive towards +axis)"}, per frequency.`);
    const others = this.index.monitors.filter((x) => x.name !== m.name && (x.type === "flux" || x.type === "flux_box" || x.type === "mode"));
    const c = this.controls();
    const norm = el("select", {}, el("option", { value: "raw" }, "flux (W)"), d.source_power ? el("option", { value: "src" }, "flux / |S(f)|²") : null,
      others.map((o) => el("option", { value: "m:" + o.name }, `flux / flux of ${o.name}`)));
    norm.value = d.source_power ? "src" : "raw";
    c.append(el("label", {}, "Show", norm));
    let scale = this.scale;
    c.append(el("label", {}, "Scale", this.seg([["lin", "linear"], ["db", "dB"]], scale, (k) => { scale = this.scale = k; draw(); })));
    const fig = new LinePlot(this.figure("tall"));
    this.figs.push(fig);
    const note = el("p", { class: "note" });
    this.stage().append(note);
    let cur = null;
    c.append(this.exportLinks(fig, `${m.name}_flux`, () => csv(["frequency_Hz", "wavelength_nm", cur.label], [d.freqs, d.freqs.map((f) => (C0 / f) * 1e9), cur.y])));
    const ref = {};
    const draw = async () => {
      let y = d.flux, label = "flux_W", yTitle = "flux (W)", name = "flux";
      const v = norm.value;
      if (v === "src") { y = d.flux.map((f, i) => f / d.source_power[i]); label = "flux_over_S2"; yTitle = "flux / |S|² (W per unit amplitude²)"; name = "flux / |S|²"; }
      if (v.startsWith("m:")) {
        const o = v.slice(2);
        ref[o] = ref[o] || (await api(`results/flux?name=${encodeURIComponent(o)}`));
        const r = ref[o];
        if (r.freqs.length !== d.freqs.length) { note.textContent = `${o} records other frequencies.`; return; }
        y = d.flux.map((f, i) => f / r.flux[i]); label = `flux_over_${o}`; yTitle = `flux / flux of ${o}`; name = `${m.name} / ${o}`;
      }
      if (scale === "db") { y = y.map((q) => 10 * Math.log10(Math.abs(q))); yTitle = yTitle.replace(/ \(.*\)$/, "") + " (dB)"; }
      cur = { y, label };
      const sp = this.spectral(d.freqs);
      fig.set({ series: [{ name, x: sp.x, y }], xTitle: sp.title, yTitle, xName: sp.name, xUnit: sp.unit, xFormat: sp.format });
      note.textContent = v === "src" ? "S(f) is the spectrum of the source signal over the steps that ran (Results.source_spectrum). For a mode source with a unit-power profile, flux / |S|² is the transmitted fraction up to the coupling of the source to the grid (about 1 %); dividing by a monitor next to the source is exact." : scale === "db" ? "dB = 10 log10 |value|." : "";
    };
    norm.addEventListener("change", draw);
    draw();
  }

  // -- mode monitors ---------------------------------------------------------------
  async showMode(m) {
    const d = await api(`results/mode?name=${encodeURIComponent(m.name)}`);
    const sp0 = this.index.sparams;
    const neffs = d.modes.map((q) => q.neff).filter((x) => x !== null && x !== undefined);
    this.header(m.name, `mode monitor: amplitudes of ${d.modes.length} mode profile${d.modes.length > 1 ? "s" : ""}${neffs.length ? " (n_eff " + neffs.map((x) => x.toFixed(4)).join(", ") + ")" : ""}, forward (+) and backward (−) along the monitor axis.`);
    const c = this.controls();
    let tab = "amp";
    c.append(el("label", {}, "View", this.seg([["amp", "amplitudes"], ["prof", "profiles"]], tab, (k) => { tab = k; render(); })));
    const body = el("div", { style: "display:flex;flex-direction:column;gap:0.6rem;flex:1;min-height:0" });
    this.stage().append(body);
    const render = () => {
      for (const f of this.figs) f.destroy();
      this.figs = [];
      body.innerHTML = "";
      if (tab === "amp") this.modeAmplitudes(body, m, d, sp0);
      else this.modeProfiles(body, m);
    };
    render();
  }

  async modeAmplitudes(body, m, d, sp0) {
    const c = el("div", { class: "controls" });
    body.append(c);
    const refName = sp0 && sp0.reference;
    let refData = null;
    if (refName && refName !== m.name) refData = await api(`results/mode?name=${encodeURIComponent(refName)}`);
    if (refName === m.name) refData = d;
    const norm = el("select", {}, el("option", { value: "raw" }, "|a|² (W)"), d.source_re ? el("option", { value: "src" }, "|a|² / |S(f)|²") : null,
      refData ? el("option", { value: "ref" }, `|a / a_in|², a_in from ${refName}`) : null);
    norm.value = refData ? "ref" : d.source_re ? "src" : "raw";
    c.append(el("label", {}, "Show", norm));
    let scale = this.scale;
    c.append(el("label", {}, "Scale", this.seg([["lin", "linear"], ["db", "dB"]], scale, (k) => { scale = this.scale = k; draw(); })));
    const plots = el("div", { class: "plots" });
    body.append(plots);
    const f1 = new LinePlot(el("div", { class: "figure" }));
    const f2 = new LinePlot(el("div", { class: "figure short" }));
    plots.append(f1.container, f2.container);
    f1.resize(); f2.resize();
    this.figs.push(f1, f2);
    let cols = null;
    c.append(this.exportLinks(f1, `${m.name}_modes`, () => csv(cols.header, cols.data)));
    c.append(el("span", { class: "links" }, el("a", { href: "#", onclick: (e) => { e.preventDefault(); f2.exportPNG(safeName(`${m.name}_phase`)); } }, "phase PNG"), el("a", { href: "#", onclick: (e) => { e.preventDefault(); f2.exportSVG(safeName(`${m.name}_phase`)); } }, "phase SVG")));
    const note = el("p", { class: "note" });
    body.append(note);
    const draw = () => {
      const v = norm.value;
      const sp = this.spectral(d.freqs);
      const s1 = [], s2 = [];
      cols = { header: ["frequency_Hz", "wavelength_nm"], data: [d.freqs, d.freqs.map((f) => (C0 / f) * 1e9)] };
      let ain = null;
      if (v === "ref") {
        const md = refData.modes[0];
        const dirPlus = !this.index.sparams || this.index.sparams.direction !== "-";
        ain = d.freqs.map((_, i) => (dirPlus ? [md.plus_re[i], md.plus_im[i]] : [md.minus_re[i], md.minus_im[i]]));
      } else if (v === "src") ain = d.freqs.map((_, i) => [d.source_re[i], d.source_im[i]]);
      d.modes.forEach((q, k) => {
        for (const [dir, re, im] of [["+", q.plus_re, q.plus_im], ["−", q.minus_re, q.minus_im]]) {
          const z = re.map((r, i) => (ain ? cdiv([r, im[i]], ain[i]) : [r, im[i]]));
          let p = z.map(([a, b]) => a * a + b * b);
          const ph = z.map(([a, b]) => (Math.atan2(b, a) * 180) / Math.PI);
          if (scale === "db") p = p.map((x) => 10 * Math.log10(x));
          const name = `mode ${k}${q.neff ? ` (${q.neff.toFixed(3)})` : ""} ${dir}`;
          const color = SERIES[(2 * k + (dir === "+" ? 0 : 1)) % SERIES.length];
          s1.push({ name, x: sp.x, y: p, color, dash: dir === "+" ? undefined : "5 3" });
          s2.push({ name, x: sp.x, y: ph, color, dash: dir === "+" ? undefined : "5 3" });
          cols.header.push(`mode${k}_${dir === "+" ? "plus" : "minus"}_power`, `mode${k}_${dir === "+" ? "plus" : "minus"}_phase_deg`);
          cols.data.push(p, ph);
        }
      });
      const yt = { raw: "|a|² (W)", src: "|a|² / |S|²", ref: "|a / a_in|²" }[v] + (scale === "db" ? " (dB)" : "");
      f1.set({ series: s1, xTitle: sp.title, yTitle: yt, xName: sp.name, xUnit: sp.unit, xFormat: sp.format, legend: true });
      f2.set({ series: s2, xTitle: sp.title, yTitle: "phase (°)", xName: sp.name, xUnit: sp.unit, xFormat: sp.format, yRange: [-180, 180], wrap: 360, legend: false });
      note.textContent = v === "ref" ? `a_in: amplitude of mode 0 in the source direction at ${refName}, the first mode monitor after the mode source. Phases are those of a / a_in.` : v === "src" ? "S(f): spectrum of the source signal over the steps that ran." : "For unit-power profiles |a|² is the power in the mode, scaled by the squared source spectrum.";
    };
    norm.addEventListener("change", draw);
    draw();
  }

  async modeProfiles(body, m) {
    const d = await api(`results/profiles?name=${encodeURIComponent(m.name)}`);
    const c = el("div", { class: "controls" });
    body.append(c);
    let k = 0, comp = "|E|";
    const modeSel = el("select", {}, d.profiles.map((p, i) => el("option", { value: i }, `mode ${i}: n_eff ${p.neff.toFixed(5)}${Math.abs(p.neff_imag || 0) > 1e-9 ? (p.neff_imag >= 0 ? " + " : " − ") + Math.abs(p.neff_imag).toExponential(2) + "i" : ""}`)));
    const comps = Object.keys(d.profiles[0].maps);
    const compSel = el("select", {}, comps.map((x) => el("option", { value: x }, x)));
    c.append(el("label", {}, "Profile", modeSel), el("label", {}, "Field", compSel));
    const wrap = el("div", { class: "figure-wrap" });
    const fd = el("div", { class: "figure" });
    wrap.append(fd);
    body.append(wrap);
    const fig = new MapPlot(fd);
    this.figs.push(fig);
    c.append(this.exportLinks(fig, `${m.name}_mode${k}`));
    const info = el("p", { class: "note" });
    body.append(info);
    const [u, v] = [d.u, d.v];
    const draw = () => {
      const p = d.profiles[k], mp = p.maps[comp];
      fig.set({
        data: decode(mp.data), nu: mp.nu, nv: mp.nv, uEdges: mp.u_edges, vEdges: mp.v_edges, uTitle: `${AXES[u]} (µm)`, vTitle: `${AXES[v]} (µm)`, uName: AXES[u], vName: AXES[v],
        cmap: "viridis", vmin: 0, vmax: mp.max || 1, cbTitle: `${comp} (a.u.)`, valueName: comp, keepView: true,
        overlays: [{ t: "segments", segs: d.outline, color: themeColors().text, width: 1, opacity: 0.7 }],
        title: `mode ${k}, n_eff = ${p.neff.toFixed(5)}, TE fraction ${p.te_fraction.toFixed(2)}${p.freq ? `, λ = ${((C0 / p.freq) * 1e9).toFixed(1)} nm` : ""}`,
      });
      info.textContent = `Profile on the monitor window, normal to ${AXES[d.axis]} at ${d.coord_um.toFixed(3)} µm; unit power. TE fraction: share of |E|² in E${AXES[u]}. Lines: medium boundaries.`;
    };
    modeSel.addEventListener("change", () => { k = +modeSel.value; draw(); });
    compSel.addEventListener("change", () => { comp = compSel.value; draw(); });
    draw();
  }

  // -- S-parameters --------------------------------------------------------------
  async showSparams() {
    const sp0 = this.index.sparams;
    const ref = sp0.reference;
    const all = {};
    for (const n of sp0.monitors) all[n] = await api(`results/mode?name=${encodeURIComponent(n)}`);
    const monSpec = Object.fromEntries(this.info.monitors.map((m) => [m.name, m]));
    this.header("S-parameters", `From the mode monitors: S = a / a_in, with a_in the amplitude of mode 0 travelling away from the source "${sp0.source}" at ${ref}, the first mode monitor after it.`);
    const c = this.controls();
    let scale = "db", what = "mag", dirs = "out", modes = "0";
    c.append(el("label", {}, "Magnitude", this.seg([["lin", "|S|"], ["pow", "|S|²"], ["db", "dB"]], scale, (k) => { scale = k; draw(); })));
    c.append(el("label", {}, "Waves", this.seg([["out", "leaving"], ["all", "both directions"]], dirs, (k) => { dirs = k; draw(); })));
    const maxModes = Math.max(...Object.values(all).map((d) => d.modes.length));
    if (maxModes > 1) c.append(el("label", {}, "Modes", this.seg([["0", "mode 0"], ["all", `all ${maxModes}`]], modes, (k) => { modes = k; draw(); })));
    // the wave leaving through a monitor travels towards the nearer face of the region
    const leaving = (n) => {
      const g = monSpec[n] && monSpec[n].geometry;
      if (!g) return null;
      const a = g.axis, nodes = this.info.nodes[a], p = g.min[a];
      return p - nodes[0] > nodes[nodes.length - 1] - p ? "+" : "−";
    };
    const fig = new LinePlot(this.figure());
    const fph = new LinePlot(this.figure("short"));
    this.figs.push(fig, fph);
    let cols = null;
    c.append(this.exportLinks(fig, `${this.info.title}_sparams`, () => csv(cols.header, cols.data)));
    c.append(el("span", { class: "links" }, el("a", { href: "#", onclick: (e) => { e.preventDefault(); fph.exportPNG(safeName(`${this.info.title}_sparams_phase`)); } }, "phase PNG"), el("a", { href: "#", onclick: (e) => { e.preventDefault(); fph.exportSVG(safeName(`${this.info.title}_sparams_phase`)); } }, "phase SVG")));
    const note = el("p", { class: "note" });
    this.stage().append(note);
    void what;
    const draw = () => {
      const r = all[ref];
      const plus = sp0.direction !== "-";
      const ain = r.freqs.map((_, i) => (plus ? [r.modes[0].plus_re[i], r.modes[0].plus_im[i]] : [r.modes[0].minus_re[i], r.modes[0].minus_im[i]]));
      const sp = this.spectral(r.freqs);
      const s1 = [], s2 = [];
      cols = { header: ["frequency_Hz", "wavelength_nm"], data: [r.freqs, r.freqs.map((f) => (C0 / f) * 1e9)] };
      let k = 0;
      for (const n of sp0.monitors) {
        const d = all[n];
        if (d.freqs.length !== r.freqs.length) continue;
        const ax = monSpec[n] ? AXES[monSpec[n].geometry.axis] : "";
        d.modes.forEach((q, mi) => {
          if (modes === "0" && mi > 0) return;
          for (const [dir, re, im] of [["+", q.plus_re, q.plus_im], ["−", q.minus_re, q.minus_im]]) {
            if (n === ref && mi === 0 && ((dir === "+") === plus)) continue; // a_in itself
            if (dirs === "out" && leaving(n) && dir !== leaving(n)) continue;
            const z = re.map((a, i) => cdiv([a, im[i]], ain[i]));
            const mag = z.map(([a, b]) => Math.hypot(a, b));
            const y = scale === "db" ? mag.map((x) => 20 * Math.log10(x)) : scale === "pow" ? mag.map((x) => x * x) : mag;
            const ph = z.map(([a, b]) => (Math.atan2(b, a) * 180) / Math.PI);
            const name = `${n}${d.modes.length > 1 ? ` mode ${mi}` : ""} ${dir}${ax}`;
            const color = SERIES[k % SERIES.length];
            k++;
            s1.push({ name, x: sp.x, y, color, dash: dir === "+" ? undefined : "5 3" });
            s2.push({ name, x: sp.x, y: ph, color, dash: dir === "+" ? undefined : "5 3" });
            cols.header.push(`${safeName(n)}_mode${mi}_${dir === "+" ? "plus" : "minus"}_abs`, `${safeName(n)}_mode${mi}_${dir === "+" ? "plus" : "minus"}_phase_deg`);
            cols.data.push(mag, ph);
          }
        });
      }
      const yt = { lin: "|S|", pow: "|S|²", db: "|S|² (dB)" }[scale];
      fig.set({ series: s1, xTitle: sp.title, yTitle: yt, xName: sp.name, xUnit: sp.unit, xFormat: sp.format, legend: true });
      fph.set({ series: s2, xTitle: sp.title, yTitle: "phase (°)", xName: sp.name, xUnit: sp.unit, xFormat: sp.format, yRange: [-180, 180], wrap: 360, legend: false });
      note.textContent = `Each curve is one mode at one monitor and one direction along the monitor axis (solid +, dashed −). "Leaving" keeps the wave that travels towards the nearer face of the region at each monitor: the reflection at ${ref}, the transmission at the others. dB = 20 log10 |S|. The CSV holds the curves shown, |S| and the phase in degrees.`;
    };
    draw();
  }

  // -- field maps ------------------------------------------------------------------
  async showMap(m) {
    const freqs = m.freqs;
    this.header(m.name, `${TYPE_LABEL[m.type] || m.type}: complex amplitude of each component at each frequency${this.info.has_spec ? ", with the medium boundaries" : ""}.`);
    const c = this.controls();
    let comp = m.components.find((x) => x === "Ey") || m.components[0], fi = Math.floor(freqs.length / 2), part = "abs", axis = null, index = null;
    const compSel = el("select", {}, m.components.map((x) => el("option", { value: x }, x)));
    compSel.value = comp;
    const fSel = el("select", {}, freqs.map((f, i) => el("option", { value: i }, `${((C0 / f) * 1e9).toFixed(1)} nm · ${(f / 1e12).toFixed(3)} THz`)));
    fSel.value = fi;
    c.append(el("label", {}, "Component", compSel), el("label", {}, "Frequency", fSel));
    c.append(el("label", {}, "Part", this.seg([["abs", "|·|"], ["intensity", "|·|²"], ["re", "Re"], ["im", "Im"], ["phase", "phase"]], part, (k) => { part = k; load(); })));
    const planeBox = el("span", { class: "group" });
    c.append(planeBox);
    const wrap = el("div", { class: "figure-wrap" });
    const fd = el("div", { class: "figure" });
    wrap.append(fd);
    this.stage().append(wrap);
    const fig = new MapPlot(fd);
    this.figs.push(fig);
    let cur = null;
    c.append(this.exportLinks(fig, `${m.name}_${comp}`, () => mapCSV(cur)));
    const note = el("p", { class: "note" });
    this.stage().append(note);
    const load = async () => {
      const q = `results/map?name=${encodeURIComponent(m.name)}&comp=${comp}&f=${fi}&part=${part}${axis !== null ? `&axis=${axis}` : ""}${index !== null ? `&index=${index}` : ""}`;
      const d = await api(q);
      cur = d;
      if (axis === null) { axis = d.axis; buildPlane(d); }
      const [u, v] = [d.u, d.v];
      const signed = d.signed;
      let vmin = d.min, vmax = d.max;
      if (part === "phase") { vmin = -Math.PI; vmax = Math.PI; } else if (signed) { const l = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1; vmin = -l; vmax = l; } else { vmin = 0; vmax = vmax || 1; }
      const unit = part === "phase" ? "rad" : part === "intensity" ? "(V/m)²" : "V/m";
      const partName = { abs: `|${comp}|`, intensity: `|${comp}|²`, re: `Re ${comp}`, im: `Im ${comp}`, phase: `arg ${comp}` }[part];
      fig.set({
        data: decode(d.data), nu: d.nu, nv: d.nv, uEdges: d.u_edges, vEdges: d.v_edges, uTitle: `${AXES[u]} (µm)`, vTitle: `${AXES[v]} (µm)`, uName: AXES[u], vName: AXES[v],
        cmap: signed ? "diverging" : "viridis", vmin, vmax, cbTitle: `${partName} (${unit})`, valueName: partName, unit, keepView: true,
        overlays: d.outline && d.outline.length ? [{ t: "segments", segs: d.outline, color: themeColors().text, width: 1, opacity: 0.65 }] : [],
        title: `${partName} at ${((C0 / d.freq) * 1e9).toFixed(1)} nm (${(d.freq / 1e12).toFixed(3)} THz)${d.count > 1 ? `, ${AXES[d.axis]} = ${d.coord_um?.toFixed(3)} µm` : ""}`,
      });
      note.textContent = `Fields scale with the source spectrum S(f). Magnetic components are multiplied by 376.73 Ω (V/m), as Results.field returns them.${part === "phase" ? " Phase on a diverging map from −π to π." : ""}${d.stride[0] > 1 || d.stride[1] > 1 ? ` Every ${d.stride[0]} × ${d.stride[1]} samples.` : ""}${d.cells_only ? " No simulation spec: axes are sample indices." : ""}`;
    };
    const buildPlane = (d) => {
      planeBox.innerHTML = "";
      if (d.count <= 1 && !m.volume) return;
      const sl = el("input", { type: "range", min: 0, max: d.count - 1, value: d.index });
      const txt = el("span", { class: "value" }, `${AXES[d.axis]} = ${d.coord_um?.toFixed(3)} µm`);
      sl.addEventListener("change", () => { index = +sl.value; load().then(() => (txt.textContent = `${AXES[cur.axis]} = ${cur.coord_um?.toFixed(3)} µm`)); });
      planeBox.append(el("label", {}, `Plane ⟂ ${AXES[d.axis]}`), sl, txt);
    };
    compSel.addEventListener("change", () => { comp = compSel.value; load(); });
    fSel.addEventListener("change", () => { fi = +fSel.value; load(); });
    await load();
  }

  // -- time series -------------------------------------------------------------------
  async showTime(m) {
    let comp = m.components[0];
    this.header(m.name, `${TYPE_LABEL[m.type] || m.type}: field against time and the magnitude of its Fourier transform.`);
    const c = this.controls();
    const compSel = el("select", {}, m.components.map((x) => el("option", { value: x }, x)));
    c.append(el("label", {}, "Component", compSel));
    let scale = this.scale;
    c.append(el("label", {}, "Spectrum", this.seg([["lin", "linear"], ["db", "dB"]], scale, (k) => { scale = this.scale = k; draw(); })));
    const f1 = new LinePlot(this.figure("short"));
    const f2 = new LinePlot(this.figure());
    this.figs.push(f1, f2);
    let d = null;
    c.append(this.exportLinks(f1, `${m.name}_${comp}_time`, () => csv(["time_s", comp], [d.times, d.values])));
    c.append(el("span", { class: "links" }, el("a", { href: "#", onclick: (e) => { e.preventDefault(); download(safeName(`${m.name}_${comp}_spectrum`) + ".csv", csv(["frequency_Hz", "abs_fft_Vs_per_m"], [d.spec_freqs || [], d.spec || []]), "text/csv"); } }, "spectrum CSV"), el("a", { href: "#", onclick: (e) => { e.preventDefault(); f2.exportPNG(safeName(`${m.name}_${comp}_spectrum`)); } }, "spectrum PNG"), el("a", { href: "#", onclick: (e) => { e.preventDefault(); f2.exportSVG(safeName(`${m.name}_${comp}_spectrum`)); } }, "spectrum SVG")));
    const note = el("p", { class: "note" });
    this.stage().append(note);
    const draw = () => {
      if (!d) return;
      f1.set({ series: [{ name: comp, x: d.times.map((t) => t * 1e15), y: d.values }], xTitle: "time (fs)", yTitle: `${comp} (V/m)`, xName: "t", xUnit: "fs" });
      if (d.spec) {
        const keep = d.spec_freqs.map((f, i) => i).filter((i) => d.spec_freqs[i] > 0);
        const fr = keep.map((i) => d.spec_freqs[i]);
        let y = keep.map((i) => d.spec[i]);
        const mx = Math.max(...y);
        if (scale === "db") y = y.map((v) => 20 * Math.log10(v / mx));
        const sp = this.spectral(fr);
        f2.set({ series: [{ name: `|FFT ${comp}|`, x: sp.x, y, markers: false }], xTitle: sp.title, yTitle: scale === "db" ? "|FFT| (dB, re. peak)" : `|FFT ${comp}| (V s/m)`, xName: sp.name, xUnit: sp.unit, xFormat: sp.format, yRange: scale === "db" ? [-80, 3] : undefined });
      }
      const box = d.box.reduce((a, b) => a * b, 1) > 1 ? ` The box holds ${d.box.join(" × ")} samples; the trace is that of the centre sample (${d.point.join(", ")}).` : "";
      note.textContent = `${d.times.length} samples. Spectrum: |Σ f(t) dt e^{iωt}|, zero-padded; ${this.xUnit === "nm" ? "wavelength" : "frequency"} axis as chosen in the side panel.${box}${comp.startsWith("H") ? " Magnetic samples are multiplied by 376.73 Ω and taken half a step earlier." : ""}`;
    };
    const load = async () => { d = await api(`results/time?name=${encodeURIComponent(m.name)}&comp=${comp}`); draw(); };
    compSel.addEventListener("change", () => { comp = compSel.value; load(); });
    await load();
  }

  async showEnergy(m) {
    const d = await api(`results/energy?name=${encodeURIComponent(m.name)}`);
    this.header(m.name, "energy monitor: electromagnetic energy in the box against time.");
    const c = this.controls();
    let scale = "lin";
    c.append(el("label", {}, "Scale", this.seg([["lin", "linear"], ["log", "log"]], scale, (k) => { scale = k; draw(); })));
    const f = new LinePlot(this.figure("tall"));
    this.figs.push(f);
    c.append(this.exportLinks(f, `${m.name}_energy`, () => csv(["time_s", "energy_J"], [d.times, d.energy])));
    const draw = () => f.set({ series: [{ name: "energy", x: d.times.map((t) => t * 1e15), y: d.energy, markers: false }], xTitle: "time (fs)", yTitle: "energy (J)", xName: "t", xUnit: "fs", yUnit: "J", yScale: scale === "log" ? "log" : "linear" });
    draw();
  }

  async showFar(m) {
    let fi = 0;
    let d = await api(`results/far?name=${encodeURIComponent(m.name)}&f=${fi}`);
    this.header(m.name, "far field: radiant intensity U(θ, φ) in W/sr; θ from +z, φ from +x in the xy plane.");
    const c = this.controls();
    const fSel = el("select", {}, d.freqs.map((f, i) => el("option", { value: i }, `${((C0 / f) * 1e9).toFixed(1)} nm · ${(f / 1e12).toFixed(3)} THz`)));
    c.append(el("label", {}, "Frequency", fSel));
    let scale = "lin";
    c.append(el("label", {}, "Scale", this.seg([["lin", "linear"], ["db", "dB"]], scale, (k) => { scale = k; draw(); })));
    const wrap = el("div", { class: "figure-wrap" });
    const fd = el("div", { class: "figure" });
    wrap.append(fd);
    this.stage().append(wrap);
    const fig = new MapPlot(fd);
    const cutFig = new LinePlot(this.figure("short"));
    this.figs.push(fig, cutFig);
    c.append(this.exportLinks(fig, `${m.name}_far`, () => {
      const lines = ["# radiant intensity (W/sr); rows: theta (deg), columns: phi (deg)", ["theta\\phi", ...d.phi].join(",")];
      d.theta.forEach((t, i) => lines.push([t, ...d.intensity[i]].join(",")));
      return lines.join("\n") + "\n";
    }));
    const draw = () => {
      let a = decode(d.data), vmin = 0, vmax = d.max || 1;
      let unit = "W/sr";
      if (scale === "db") { a = a.map((v) => 10 * Math.log10(Math.max(v, 1e-300) / (d.max || 1))); vmin = -40; vmax = 0; unit = "dB"; }
      fig.set({ data: a, nu: d.nu, nv: d.nv, uEdges: d.u_edges, vEdges: d.v_edges, uTitle: "φ (°)", vTitle: "θ (°)", uName: "φ", vName: "θ", coordUnit: "°", coordDigits: 1,
        cmap: "viridis", vmin, vmax, cbTitle: `U (${unit})`, valueName: "U", unit, keepView: true, title: `U at ${((C0 / d.freq) * 1e9).toFixed(1)} nm` });
      // cuts at the azimuths nearest 0° and 90°
      const series = [];
      for (const target of [0, 90]) {
        let j = 0;
        d.phi.forEach((p, k) => { if (Math.abs(p - target) < Math.abs(d.phi[j] - target)) j = k; });
        let y = d.theta.map((_, i) => d.intensity[i][j]);
        if (scale === "db") y = y.map((v) => 10 * Math.log10(Math.max(v, 1e-300) / (d.max || 1)));
        if (!series.some((s) => s.name === `φ = ${d.phi[j].toFixed(1)}°`)) series.push({ name: `φ = ${d.phi[j].toFixed(1)}°`, x: d.theta, y });
      }
      cutFig.set({ series, xTitle: "θ (°)", yTitle: `U (${unit})`, xName: "θ", xUnit: "°", legend: true });
    };
    fSel.addEventListener("change", async () => { fi = +fSel.value; d = await api(`results/far?name=${encodeURIComponent(m.name)}&f=${fi}`); draw(); });
    draw();
  }
}

function cdiv([a, b], [c, d]) {
  const den = c * c + d * d;
  return [(a * c + b * d) / den, (b * c - a * d) / den];
}

function mapCSV(d) {
  if (!d) return "";
  const a = decode(d.data);
  const cu = d.u_edges.slice(0, -1).map((x, i) => (x + d.u_edges[i + 1]) / 2);
  const cv = d.v_edges.slice(0, -1).map((x, i) => (x + d.v_edges[i + 1]) / 2);
  const lines = [`# ${d.part} of ${d.comp} at ${d.freq} Hz; rows: ${AXES[d.u]} (um), columns: ${AXES[d.v]} (um)`, [`${AXES[d.u]}\\${AXES[d.v]}`, ...cv.map((x) => x.toFixed(5))].join(",")];
  for (let i = 0; i < d.nu; i++) {
    const row = [cu[i].toFixed(5)];
    for (let j = 0; j < d.nv; j++) row.push(a[i * d.nv + j].toPrecision(7));
    lines.push(row.join(","));
  }
  return lines.join("\n") + "\n";
}
