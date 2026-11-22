// Figure base: an SVG layer (axes, labels, overlays) over an optional canvas
// (raster), resized with its container, exportable as SVG and PNG. Colours
// are written as attributes (not CSS classes), so an exported file looks
// like the screen without the page's style sheet.

import { el, svg, themeColors, download, SVGNS } from "./util.js";

export function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

export function ticks(lo, hi, count = 6) {
  if (!(hi > lo)) return { values: [lo], step: 1 };
  const step = niceStep(hi - lo, count);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { values: out, step };
}

export function logTicks(lo, hi) {
  const out = [];
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  const every = Math.max(1, Math.ceil((b - a) / 7));
  for (let e = a; e <= b; e += every) {
    const v = Math.pow(10, e);
    if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
  }
  return out;
}

export function tickText(v, step) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    const e = Math.floor(Math.log10(a));
    const m = v / Math.pow(10, e);
    return `${Number(m.toPrecision(3))}e${e}`.replace(/-/g, "−");
  }
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(d).replace(/-/g, "−");
}

export class Figure {
  constructor(container, opts = {}) {
    this.container = container;
    container.innerHTML = "";
    this.opts = opts;
    if (opts.raster) {
      this.canvas = el("canvas");
      container.append(this.canvas);
    }
    this.svg = svg("svg", { xmlns: SVGNS });
    container.append(this.svg);
    this.readout = el("div", { class: "readout", hidden: true });
    container.append(this.readout);
    this.w = 0;
    this.h = 0;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
  }

  destroy() {
    this.ro.disconnect();
    this.container.innerHTML = "";
  }

  resize() {
    const r = this.container.getBoundingClientRect();
    const w = Math.max(50, Math.floor(r.width)), h = Math.max(50, Math.floor(r.height));
    if (w === this.w && h === this.h) return;
    this.w = w;
    this.h = h;
    this.svg.setAttribute("width", w);
    this.svg.setAttribute("height", h);
    this.svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    this.render();
  }

  render() {
    this.colors = themeColors();
    this.svg.innerHTML = "";
    this.svg.setAttribute("font-family", this.colors.fontHead);
    this.draw();
  }

  draw() {}

  empty(msg) {
    this.svg.innerHTML = "";
    if (this.canvas) this.canvas.getContext("2d").clearRect(0, 0, this.canvas.width, this.canvas.height);
    const c = this.colors || themeColors();
    const t = svg("text", { x: this.w / 2, y: this.h / 2, "text-anchor": "middle", fill: c.muted, "font-size": 14, "font-family": c.fontHead });
    t.textContent = msg;
    this.svg.append(t);
  }

  // text helpers
  text(x, y, s, o = {}) {
    const t = svg("text", {
      x, y, fill: o.fill || this.colors.text, "font-size": o.size || 12, "text-anchor": o.anchor || "start",
      "dominant-baseline": o.baseline || "auto", "font-family": o.mono ? this.colors.fontMono : this.colors.fontHead,
      "font-weight": o.weight, transform: o.rotate ? `rotate(${o.rotate} ${x} ${y})` : undefined,
    });
    t.textContent = s;
    (o.parent || this.svg).append(t);
    return t;
  }

  // a standalone SVG of the figure (raster embedded as a PNG image)
  toSVGString() {
    const s = this.svg.cloneNode(true);
    s.setAttribute("xmlns", SVGNS);
    s.setAttribute("width", this.w);
    s.setAttribute("height", this.h);
    s.querySelectorAll("[data-noexport]").forEach((n) => n.remove());
    const bg = svg("rect", { x: 0, y: 0, width: this.w, height: this.h, fill: this.colors.panel });
    s.insertBefore(bg, s.firstChild);
    if (this.canvas && this.plotRect) {
      const p = this.plotRect;
      const img = svg("image", { x: p.x, y: p.y, width: p.w, height: p.h, preserveAspectRatio: "none", href: this.canvas.toDataURL("image/png") });
      s.insertBefore(img, bg.nextSibling);
    }
    return new XMLSerializer().serializeToString(s);
  }

  exportSVG(name) {
    download(name + ".svg", this.toSVGString(), "image/svg+xml");
  }

  exportPNG(name, scale = 2) {
    const src = this.toSVGString();
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = this.w * scale;
      c.height = this.h * scale;
      const g = c.getContext("2d");
      g.scale(scale, scale);
      g.drawImage(img, 0, 0);
      c.toBlob((b) => download(name + ".png", b), "image/png");
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(src);
  }

  showReadout(x, y, lines) {
    const r = this.readout;
    r.textContent = lines.join("\n");
    r.hidden = false;
    const bw = r.offsetWidth, bh = r.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + bw > this.w - 4) left = x - bw - 14;
    if (top + bh > this.h - 4) top = y - bh - 14;
    r.style.left = Math.max(2, left) + "px";
    r.style.top = Math.max(2, top) + "px";
  }

  hideReadout() {
    this.readout.hidden = true;
  }
}

// axis drawing: returns nothing, draws ticks + grid + label on the svg
export function drawAxes(fig, rect, x, y, opt = {}) {
  const c = fig.colors;
  const g = svg("g", {});
  fig.svg.append(g);
  // frame
  g.append(svg("rect", { x: rect.x, y: rect.y, width: rect.w, height: rect.h, fill: "none", stroke: c.lineStrong, "stroke-width": 1 }));
  const xt = x.ticks;
  for (const v of xt.values) {
    const px = x.map(v);
    if (px < rect.x - 0.5 || px > rect.x + rect.w + 0.5) continue;
    if (opt.grid) g.append(svg("line", { x1: px, x2: px, y1: rect.y, y2: rect.y + rect.h, stroke: c.grid, "stroke-width": 1 }));
    g.append(svg("line", { x1: px, x2: px, y1: rect.y + rect.h, y2: rect.y + rect.h + 4, stroke: c.lineStrong }));
    fig.text(px, rect.y + rect.h + 16, x.label(v), { anchor: "middle", size: 10.5, mono: true, fill: c.muted, parent: g });
  }
  const yt = y.ticks;
  for (const v of yt.values) {
    const py = y.map(v);
    if (py < rect.y - 0.5 || py > rect.y + rect.h + 0.5) continue;
    if (opt.grid) g.append(svg("line", { x1: rect.x, x2: rect.x + rect.w, y1: py, y2: py, stroke: c.grid, "stroke-width": 1 }));
    g.append(svg("line", { x1: rect.x - 4, x2: rect.x, y1: py, y2: py, stroke: c.lineStrong }));
    fig.text(rect.x - 7, py, y.label(v), { anchor: "end", baseline: "middle", size: 10.5, mono: true, fill: c.muted, parent: g });
  }
  if (x.title) fig.text(rect.x + rect.w / 2, rect.y + rect.h + 34, x.title, { anchor: "middle", size: 12, fill: c.text, parent: g });
  if (y.title) fig.text(rect.x - (opt.yTitleOffset || 48), rect.y + rect.h / 2, y.title, { anchor: "middle", size: 12, fill: c.text, rotate: -90, parent: g });
  return g;
}
