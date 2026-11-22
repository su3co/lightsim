// Line plots with a hover read-out, linear or logarithmic y, drag to zoom
// along x (double-click resets).

import { Figure, ticks, logTicks, tickText, drawAxes } from "./figure.js";
import { svg, SERIES, fmt } from "./util.js";

export class LinePlot extends Figure {
  constructor(container, opts = {}) {
    super(container, opts);
    this.data = { series: [] };
    this.xr = null;
    this.svg.addEventListener("mousemove", (e) => this.hover(e));
    this.svg.addEventListener("mouseleave", () => { this.hideReadout(); this.cursor && this.cursor.remove(); this.cursor = null; });
    this.svg.addEventListener("mousedown", (e) => this.dragStart(e));
    this.svg.addEventListener("dblclick", () => { this.xr = null; this.render(); });
  }

  set(data) {
    this.data = data;
    for (const s of data.series) {
      // sort by x so that hover can bisect
      const idx = s.x.map((_, i) => i).sort((a, b) => s.x[a] - s.x[b]);
      s.xs = idx.map((i) => s.x[i]);
      s.ys = idx.map((i) => s.y[i]);
    }
    if (!data.keepZoom) this.xr = null;
    this.render();
  }

  legendLayout() {
    const items = this.data.series.filter((s) => !s.noLegend);
    const rows = [];
    let row = [], wsum = 0;
    const maxw = this.w - 80;
    for (const s of items) {
      const w = 22 + 6.6 * s.name.length + 16;
      if (row.length && wsum + w > maxw) { rows.push(row); row = []; wsum = 0; }
      row.push({ s, w });
      wsum += w;
    }
    if (row.length) rows.push(row);
    return rows;
  }

  draw() {
    const d = this.data, c = this.colors;
    if (!d) return;
    if (!d.series || !d.series.length) return this.empty(d.emptyText || "nothing to plot");
    const legend = d.legend === false ? [] : d.series.length > 1 || d.legend ? this.legendLayout() : [];
    const top = 12 + legend.length * 17;
    const rect = { x: 70, y: top + 6, w: this.w - 70 - 18, h: this.h - top - 6 - 48 };
    if (rect.w < 40 || rect.h < 30) return;
    this.rect = rect;
    // ranges
    let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
    for (const s of d.series) for (const v of s.xs) { if (Number.isFinite(v)) { xlo = Math.min(xlo, v); xhi = Math.max(xhi, v); } }
    if (this.xr) [xlo, xhi] = this.xr;
    const log = d.yScale === "log";
    for (const s of d.series) {
      s.xs.forEach((xv, i) => {
        const v = s.ys[i];
        if (xv < xlo || xv > xhi || v === null || !Number.isFinite(v) || (log && v <= 0)) return;
        ylo = Math.min(ylo, v); yhi = Math.max(yhi, v);
      });
    }
    for (const h of d.hlines || []) if (Number.isFinite(h.y) && (!log || h.y > 0)) { ylo = Math.min(ylo, h.y); yhi = Math.max(yhi, h.y); }
    if (d.yRange) [ylo, yhi] = d.yRange;
    if (!Number.isFinite(ylo)) { ylo = log ? 1e-6 : 0; yhi = 1; }
    if (xhi === xlo) { xlo -= 0.5; xhi += 0.5; }
    if (log) {
      if (yhi <= ylo) { yhi = ylo * 10; }
      ylo = Math.pow(10, Math.floor(Math.log10(ylo))); yhi = Math.pow(10, Math.ceil(Math.log10(yhi)));
    } else {
      if (yhi === ylo) { const a = Math.abs(yhi) || 1; ylo -= 0.5 * a; yhi += 0.5 * a; }
      const pad = 0.06 * (yhi - ylo);
      if (!d.yRange) { ylo -= d.yZero && ylo >= 0 ? 0 : pad; yhi += pad; if (d.yZero && ylo > 0) ylo = 0; }
    }
    const xt = ticks(xlo, xhi, Math.max(3, Math.floor(rect.w / 90)));
    const sx = (v) => rect.x + ((v - xlo) / (xhi - xlo)) * rect.w;
    const ly = (v) => Math.log10(v);
    const sy = log ? (v) => rect.y + rect.h - ((ly(v) - ly(ylo)) / (ly(yhi) - ly(ylo))) * rect.h
      : (v) => rect.y + rect.h - ((v - ylo) / (yhi - ylo)) * rect.h;
    let yt = log ? { values: logTicks(ylo, yhi), step: 1 } : ticks(ylo, yhi, Math.max(3, Math.floor(rect.h / 50)));
    if (d.yRange && d.yRange[0] === -180 && d.yRange[1] === 180) yt = { values: rect.h > 140 ? [-180, -90, 0, 90, 180] : [-180, 0, 180], step: 90 };
    this.sx = sx; this.sy = sy; this.xlo = xlo; this.xhi = xhi;
    drawAxes(this, rect,
      { ticks: xt, map: sx, label: (v) => (d.xFormat && d.xFormat(v)) || tickText(v, xt.step), title: d.xTitle },
      { ticks: yt, map: sy, label: (v) => (log ? tickText(v, v) : tickText(v, yt.step)), title: d.yTitle },
      { grid: true, yTitleOffset: 56 });
    // clip
    const id = "clip" + Math.random().toString(36).slice(2);
    const defs = svg("defs");
    const cp = svg("clipPath", { id });
    cp.append(svg("rect", { x: rect.x, y: rect.y, width: rect.w, height: rect.h }));
    defs.append(cp);
    this.svg.append(defs);
    const g = svg("g", { "clip-path": `url(#${id})` });
    this.svg.append(g);
    for (const h of d.hlines || []) {
      if (!Number.isFinite(h.y) || (log && h.y <= 0)) continue;
      const py = sy(h.y);
      g.append(svg("line", { x1: rect.x, x2: rect.x + rect.w, y1: py, y2: py, stroke: h.color || c.warn, "stroke-dasharray": "5 4", "stroke-width": 1.2 }));
      if (h.label) this.text(rect.x + rect.w - 6, py - 5, h.label, { anchor: "end", size: 10.5, fill: h.color || c.warn, parent: g });
    }
    d.series.forEach((s, k) => {
      s.color = s.color || SERIES[k % SERIES.length];
      let path = "", pen = false, last = null;
      s.xs.forEach((xv, i) => {
        const v = s.ys[i];
        if (v === null || !Number.isFinite(v) || (log && v <= 0)) { pen = false; return; }
        // phases: no line across a wrap
        if (d.wrap && pen && Math.abs(v - last) > d.wrap / 2) pen = false;
        last = v;
        path += (pen ? "L" : "M") + sx(xv).toFixed(2) + "," + sy(v).toFixed(2);
        pen = true;
      });
      g.append(svg("path", { d: path, fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-dasharray": s.dash, "stroke-linejoin": "round" }));
      if (s.markers || s.xs.length <= 24) {
        s.xs.forEach((xv, i) => {
          const v = s.ys[i];
          if (v === null || !Number.isFinite(v) || (log && v <= 0)) return;
          g.append(svg("circle", { cx: sx(xv), cy: sy(v), r: 2.3, fill: s.color }));
        });
      }
    });
    // legend
    let ly0 = 14;
    for (const row of legend) {
      let lx = rect.x;
      for (const { s, w } of row) {
        this.svg.append(svg("line", { x1: lx, x2: lx + 16, y1: ly0 - 4, y2: ly0 - 4, stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash }));
        this.text(lx + 22, ly0, s.name, { size: 11.5, fill: c.text });
        lx += w;
      }
      ly0 += 17;
    }
    if (d.note) this.text(rect.x + rect.w, this.h - 6, d.note, { anchor: "end", size: 10.5, fill: c.muted });
  }

  hover(e) {
    if (!this.rect || this.dragging) return;
    const b = this.svg.getBoundingClientRect();
    const mx = e.clientX - b.left, my = e.clientY - b.top;
    const r = this.rect;
    if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) { this.hideReadout(); this.cursor && this.cursor.remove(); this.cursor = null; return; }
    const xv = this.xlo + ((mx - r.x) / r.w) * (this.xhi - this.xlo);
    const d = this.data;
    // nearest sample of the first series sets the x of the read-out
    let best = null;
    for (const s of d.series) {
      const i = nearest(s.xs, xv);
      if (i < 0) continue;
      const dist = Math.abs(s.xs[i] - xv);
      if (!best || dist < best.dist) best = { dist, x: s.xs[i] };
    }
    if (!best) return;
    const lines = [`${d.xName || "x"} = ${(d.xFormat && d.xFormat(best.x, true)) || fmt(best.x, 6)}${d.xUnit ? " " + d.xUnit : ""}`];
    if (this.cursor) this.cursor.remove();
    this.cursor = svg("g", { "data-noexport": "1", "pointer-events": "none" });
    const px = this.sx(best.x);
    this.cursor.append(svg("line", { x1: px, x2: px, y1: r.y, y2: r.y + r.h, stroke: this.colors.muted, "stroke-width": 1, "stroke-dasharray": "2 3" }));
    for (const s of d.series) {
      const i = nearest(s.xs, best.x);
      if (i < 0 || Math.abs(s.xs[i] - best.x) > 1e-9 * Math.max(1, Math.abs(best.x)) + Math.abs(this.xhi - this.xlo) * 0.02) continue;
      const v = s.ys[i];
      lines.push(`${s.name}: ${fmt(v, 5)}${d.yUnit ? " " + d.yUnit : ""}`);
      if (v !== null && Number.isFinite(v) && !(d.yScale === "log" && v <= 0)) this.cursor.append(svg("circle", { cx: this.sx(s.xs[i]), cy: this.sy(v), r: 3.5, fill: "none", stroke: s.color, "stroke-width": 1.5 }));
    }
    this.svg.append(this.cursor);
    this.showReadout(mx, my, lines);
  }

  dragStart(e) {
    if (!this.rect || e.button !== 0) return;
    const b = this.svg.getBoundingClientRect();
    const x0 = e.clientX - b.left;
    const r = this.rect;
    if (x0 < r.x || x0 > r.x + r.w) return;
    this.dragging = true;
    const band = svg("rect", { x: x0, y: r.y, width: 0, height: r.h, fill: this.colors.accent, opacity: 0.12, "data-noexport": "1" });
    this.svg.append(band);
    const move = (ev) => {
      const x1 = Math.max(r.x, Math.min(r.x + r.w, ev.clientX - b.left));
      band.setAttribute("x", Math.min(x0, x1));
      band.setAttribute("width", Math.abs(x1 - x0));
    };
    const up = (ev) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      this.dragging = false;
      const x1 = Math.max(r.x, Math.min(r.x + r.w, ev.clientX - b.left));
      band.remove();
      if (Math.abs(x1 - x0) > 8) {
        const inv = (px) => this.xlo + ((px - r.x) / r.w) * (this.xhi - this.xlo);
        this.xr = [inv(Math.min(x0, x1)), inv(Math.max(x0, x1))];
        this.render();
      }
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }
}

function nearest(xs, v) {
  if (!xs.length) return -1;
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xs[m] < v) lo = m; else hi = m;
  }
  return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
}
