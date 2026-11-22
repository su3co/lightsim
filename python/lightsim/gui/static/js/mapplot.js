// 2-D maps on a rectilinear (possibly non-uniform) grid with true aspect
// ratio, a colour bar, overlays in data coordinates, a hover read-out, wheel
// zoom, drag to pan and double-click to reset.

import { Figure, ticks, tickText, drawAxes } from "./figure.js";
import { svg, fmt } from "./util.js";
import { MAPS, CATEGORICAL } from "./colormaps.js";

function bisect(edges, v) {
  // index i with edges[i] <= v < edges[i+1], or -1 outside
  const n = edges.length - 1;
  if (!(v >= edges[0]) || !(v <= edges[n])) return -1;
  let lo = 0, hi = n;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (edges[m] <= v) lo = m; else hi = m;
  }
  return lo;
}

function parseColor(s) {
  const c = document.createElement("canvas").getContext("2d");
  c.fillStyle = s;
  const h = c.fillStyle; // #rrggbb or rgba()
  if (h.startsWith("#")) return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16), 255];
  const m = h.match(/[\d.]+/g).map(Number);
  return [m[0], m[1], m[2], Math.round((m[3] ?? 1) * 255)];
}

export class MapPlot extends Figure {
  constructor(container, opts = {}) {
    super(container, { raster: true, ...opts });
    this.m = null;
    this.view = null;
    this.svg.addEventListener("mousemove", (e) => this.hover(e));
    this.svg.addEventListener("mouseleave", () => this.hideReadout());
    this.svg.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    this.svg.addEventListener("mousedown", (e) => this.pan(e));
    this.svg.addEventListener("dblclick", () => { this.view = null; this.render(); });
  }

  set(m) {
    const sameExtent = this.m && m.keepView && this.m.uEdges[0] === m.uEdges[0] && this.m.uEdges[m.uEdges.length - 1] === m.uEdges[m.uEdges.length - 1]
      && this.m.vEdges[0] === m.vEdges[0] && this.m.vEdges[m.vEdges.length - 1] === m.vEdges[m.vEdges.length - 1];
    this.m = m;
    if (!sameExtent) this.view = null;
    this._colors = null;
    this.render();
  }

  setOverlays(ov) {
    if (!this.m) return;
    this.m.overlays = ov;
    this.render();
  }

  extent() {
    const m = this.m;
    return [m.uEdges[0], m.uEdges[m.uEdges.length - 1], m.vEdges[0], m.vEdges[m.vEdges.length - 1]];
  }

  sampleColors() {
    const m = this.m, c = this.colors;
    const n = m.nu * m.nv;
    const out = new Uint32Array(n);
    const nan = parseColor(c.nan);
    const pack = (r, g, b, a = 255) => (a << 24) | (b << 16) | (g << 8) | r;
    const nanc = pack(...nan);
    if (m.categories) {
      const cols = m.categories.map((_, i) => parseColor(m.catColors ? m.catColors[i] : CATEGORICAL[i % CATEGORICAL.length]));
      for (let i = 0; i < n; i++) {
        const v = m.data[i];
        const k = Math.round(v);
        out[i] = Number.isFinite(v) && cols[k] ? pack(...cols[k]) : nanc;
      }
      return out;
    }
    const map = m.cmap === "diverging" ? (c.dark ? MAPS.divergingDark : MAPS.divergingLight) : MAPS.viridis;
    const lo = m.vmin, hi = m.vmax, span = hi - lo || 1;
    const log = m.log;
    const l0 = log ? Math.log10(Math.max(lo, 1e-300)) : 0, l1 = log ? Math.log10(Math.max(hi, 1e-300)) : 1;
    for (let i = 0; i < n; i++) {
      const v = m.data[i];
      if (!Number.isFinite(v)) { out[i] = nanc; continue; }
      let t = log ? (Math.log10(Math.max(v, 1e-300)) - l0) / (l1 - l0 || 1) : (v - lo) / span;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const j = Math.round(t * 255) * 3;
      out[i] = pack(map[j], map[j + 1], map[j + 2]);
    }
    return out;
  }

  draw() {
    const m = this.m, c = this.colors;
    const ctx = this.canvas.getContext("2d");
    if (!m) { this.plotRect = null; ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); return this.empty(this.opts.emptyText || "no data"); }
    const right = m.categories ? Math.min(200, 40 + 7 * Math.max(...m.categories.map((s) => s.length))) : 92;
    const avail = { x: 66, y: 12 + (m.title ? 18 : 0), w: this.w - 66 - right, h: this.h - 12 - (m.title ? 18 : 0) - 48 };
    if (avail.w < 40 || avail.h < 40) return;
    const [ux0, ux1, vy0, vy1] = this.extent();
    let v = this.view || [ux0, ux1, vy0, vy1];
    // narrow windows: the figure takes the height its width allows (no empty bands)
    const wrap = this.container.parentElement;
    if (wrap && wrap.classList.contains("figure-wrap") && !this.view) {
      // the figure takes the height its width allows, so that a wide domain leaves no empty bands
      const narrow = window.innerWidth <= 900;
      const cap = narrow ? window.innerHeight * 0.8 : Math.max(320, window.innerHeight - 230 - (+wrap.dataset.reserve || 0));
      const want = Math.round(Math.min(Math.max(((v[3] - v[2]) / (v[1] - v[0])) * avail.w + (this.h - avail.h), 260), cap));
      if (Math.abs(want - this.h) > 3) { wrap.style.flex = "none"; wrap.style.height = want + "px"; }
    }
    // true aspect: one µm is the same number of pixels on both axes
    const scale = Math.min(avail.w / (v[1] - v[0]), avail.h / (v[3] - v[2]));
    if (!this.view) {
      // the full extent fitted; centre the unused direction
      const pw = (v[1] - v[0]) * scale, ph = (v[3] - v[2]) * scale;
      this.rect = { x: avail.x, y: avail.y + (avail.h - ph) / 2, w: pw, h: ph };
    } else {
      // a zoomed view fills the available area (the window grows to keep the aspect)
      const cx = (v[0] + v[1]) / 2, cy = (v[2] + v[3]) / 2;
      const hw = avail.w / scale / 2, hh = avail.h / scale / 2;
      v = [cx - hw, cx + hw, cy - hh, cy + hh];
      this.rect = { ...avail };
    }
    this.v = v;
    this.scale = scale;
    const r = this.rect;
    this.plotRect = r;
    const sx = (x) => r.x + ((x - v[0]) / (v[1] - v[0])) * r.w;
    const sy = (y) => r.y + r.h - ((y - v[2]) / (v[3] - v[2])) * r.h;
    this.sx = sx; this.sy = sy;
    // raster
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const pw = Math.max(1, Math.round(r.w * dpr)), ph = Math.max(1, Math.round(r.h * dpr));
    this.canvas.width = pw;
    this.canvas.height = ph;
    this.canvas.style.left = r.x + "px";
    this.canvas.style.top = r.y + "px";
    this.canvas.style.width = r.w + "px";
    this.canvas.style.height = r.h + "px";
    if (!this._colors) this._colors = this.sampleColors();
    const cols = new Int32Array(pw), rows = new Int32Array(ph);
    for (let i = 0; i < pw; i++) cols[i] = bisect(m.uEdges, v[0] + ((i + 0.5) / pw) * (v[1] - v[0]));
    for (let j = 0; j < ph; j++) rows[j] = bisect(m.vEdges, v[3] - ((j + 0.5) / ph) * (v[3] - v[2]));
    const img = ctx.createImageData(pw, ph);
    const out = new Uint32Array(img.data.buffer);
    const sc = this._colors, nv = m.nv;
    for (let j = 0; j < ph; j++) {
      const iv = rows[j];
      if (iv < 0) continue;
      const base = j * pw;
      for (let i = 0; i < pw; i++) {
        const iu = cols[i];
        if (iu >= 0) out[base + i] = sc[iu * nv + iv];
      }
    }
    ctx.putImageData(img, 0, 0);
    // axes
    const xt = ticks(v[0], v[1], Math.max(2, Math.floor(r.w / 80)));
    const yt = ticks(v[2], v[3], Math.max(2, Math.floor(r.h / 48)));
    drawAxes(this, r, { ticks: xt, map: sx, label: (t) => tickText(t, xt.step), title: m.uTitle }, { ticks: yt, map: sy, label: (t) => tickText(t, yt.step), title: m.vTitle }, { grid: false, yTitleOffset: 50 });
    if (m.title) this.text(r.x, avail.y - 6, m.title, { size: 12, fill: c.text });
    this.drawOverlays();
    this.drawColorbar(r, right);
    if (m.note) this.text(this.w - 6, this.h - 6, m.note, { anchor: "end", size: 10.5, fill: c.muted });
  }

  drawOverlays() {
    const m = this.m, c = this.colors, r = this.rect;
    if (!m.overlays || !m.overlays.length) return;
    const id = "mclip" + Math.random().toString(36).slice(2);
    const defs = svg("defs");
    const cp = svg("clipPath", { id });
    cp.append(svg("rect", { x: r.x, y: r.y, width: r.w, height: r.h }));
    defs.append(cp);
    const pat = svg("pattern", { id: id + "h", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.append(svg("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: c.pml, "stroke-width": 1, opacity: 0.4 }));
    defs.append(pat);
    this.svg.append(defs);
    const g = svg("g", { "clip-path": `url(#${id})` });
    this.svg.append(g);
    const labels = [];
    const sx = this.sx, sy = this.sy;
    for (const o of m.overlays) {
      const stroke = o.color || c.mon;
      const w = o.width || 1.6;
      const halo = o.halo !== false;
      const add = (el) => {
        if (halo && o.t !== "hatch" && o.t !== "segments") {
          const h = el.cloneNode();
          h.setAttribute("stroke", c.halo);
          h.setAttribute("stroke-width", w + 2.6);
          h.setAttribute("fill", "none");
          h.removeAttribute("stroke-dasharray");
          g.append(h);
        }
        g.append(el);
      };
      const base = { stroke, "stroke-width": w, fill: o.fill || "none", "stroke-dasharray": o.dash, opacity: o.opacity };
      if (o.t === "rect" || o.t === "hatch") {
        const x0 = sx(Math.min(o.x0, o.x1)), x1 = sx(Math.max(o.x0, o.x1)), y0 = sy(Math.max(o.y0, o.y1)), y1 = sy(Math.min(o.y0, o.y1));
        if (o.t === "hatch") {
          g.append(svg("rect", { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0), fill: `url(#${id}h)`, stroke: "none" }));
          g.append(svg("rect", { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0), fill: "none", stroke: c.pml, "stroke-width": 0.8, opacity: 0.6 }));
        } else add(svg("rect", { ...base, x: x0, y: y0, width: Math.max(0.5, x1 - x0), height: Math.max(0.5, y1 - y0) }));
        if (o.label) labels.push({ x: x0 + 3, y: y0 - 4, s: o.label, color: stroke, p: o.priority });
      } else if (o.t === "line") {
        add(svg("line", { ...base, x1: sx(o.x0), y1: sy(o.y0), x2: sx(o.x1), y2: sy(o.y1) }));
        if (o.label) labels.push({ x: Math.max(sx(o.x0), sx(o.x1)) + 4, y: Math.min(sy(o.y0), sy(o.y1)) + 10, s: o.label, color: stroke, p: o.priority });
      } else if (o.t === "point") {
        const px = sx(o.x), py = sy(o.y);
        add(svg("circle", { ...base, cx: px, cy: py, r: o.r || 4.5 }));
        add(svg("line", { ...base, x1: px - 8, x2: px + 8, y1: py, y2: py }));
        add(svg("line", { ...base, x1: px, x2: px, y1: py - 8, y2: py + 8 }));
        if (o.label) labels.push({ x: px + 8, y: py - 8, s: o.label, color: stroke, p: o.priority });
      } else if (o.t === "poly") {
        add(svg("polygon", { ...base, points: o.pts.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ") }));
      } else if (o.t === "circle") {
        add(svg("ellipse", { ...base, cx: sx(o.x), cy: sy(o.y), rx: Math.abs(sx(o.x + o.r) - sx(o.x)), ry: Math.abs(sy(o.y + o.r) - sy(o.y)) }));
      } else if (o.t === "segments") {
        let d = "";
        for (const s of o.segs) d += `M${sx(s[0]).toFixed(1)},${sy(s[1]).toFixed(1)}L${sx(s[2]).toFixed(1)},${sy(s[3]).toFixed(1)}`;
        g.append(svg("path", { d, stroke, "stroke-width": w, fill: "none", opacity: o.opacity ?? 1, "stroke-dasharray": o.dash }));
      } else if (o.t === "mesh") {
        // grid lines inside the view; every k-th when they would be closer than 4 px
        const v = this.v;
        let d = "", every = 1;
        for (const [nodes, horiz] of [[o.u, false], [o.v, true]]) {
          const lo = horiz ? v[2] : v[0], hi = horiz ? v[3] : v[1];
          const inside = nodes.filter((q) => q >= lo && q <= hi);
          const px = (inside.length > 1 ? (hi - lo) / inside.length : 1) * this.scale;
          const k = Math.max(1, Math.ceil(4 / Math.max(px, 1e-9)));
          every = Math.max(every, k);
          for (let i = 0; i < inside.length; i += k) {
            const q = inside[i];
            d += horiz ? `M${r.x},${sy(q).toFixed(1)}H${r.x + r.w}` : `M${sx(q).toFixed(1)},${r.y}V${r.y + r.h}`;
          }
        }
        g.append(svg("path", { d, stroke: o.color, "stroke-width": 0.6, opacity: 0.35, fill: "none" }));
        this.meshEvery = every;
      } else if (o.t === "arrow") {
        const x0 = sx(o.x), y0 = sy(o.y);
        const L = 22, ang = Math.atan2(-o.dy, o.dx);
        const x1 = x0 + L * Math.cos(ang), y1 = y0 + L * Math.sin(ang);
        const hx = (a) => x1 - 7 * Math.cos(ang + a), hy = (a) => y1 - 7 * Math.sin(ang + a);
        add(svg("path", { ...base, d: `M${x0},${y0}L${x1},${y1}M${hx(0.45)},${hy(0.45)}L${x1},${y1}L${hx(-0.45)},${hy(-0.45)}` }));
        if (o.label) labels.push({ x: x1 + 4, y: y1 - 4, s: o.label, color: stroke });
      }
    }
    // labels: skip those that would overlap an earlier one
    const placed = [];
    labels.sort((a, b) => (b.p ? 1 : 0) - (a.p ? 1 : 0));
    for (const l of labels) {
      const w = 6.4 * l.s.length, h = 12;
      const box = { x: l.x, y: l.y - h, w, h };
      if (placed.some((p) => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y)) continue;
      placed.push(box);
      // the halo has to contrast with the map under the label, not with the page: viridis and the
      // dark-centred diverging map are dark where most labels sit, the light-centred one is light
      const onDark = this.m.cmap !== "diverging" || c.dark;
      const t = this.text(l.x, l.y, l.s, { size: 11, fill: onDark ? (c.dark ? l.color : "#f3f5f9") : l.color, parent: g });
      t.setAttribute("paint-order", "stroke");
      t.setAttribute("stroke", onDark ? "rgba(10, 12, 16, 0.86)" : c.halo);
      t.setAttribute("stroke-width", 3);
    }
  }

  drawColorbar(r, right) {
    const m = this.m, c = this.colors;
    const x = r.x + r.w + 18;
    if (m.categories) {
      let y = r.y + 4;
      const present = m.present || m.categories.map((_, i) => i);
      for (const i of present) {
        const col = m.catColors ? m.catColors[i] : CATEGORICAL[i % CATEGORICAL.length];
        this.svg.append(svg("rect", { x, y, width: 11, height: 11, fill: col, stroke: c.lineStrong }));
        this.text(x + 17, y + 9.5, m.categories[i], { size: 11.5, fill: c.text });
        y += 17;
        if (y > r.y + r.h + 30) break;
      }
      if (m.cbTitle) this.text(x, r.y + r.h + 34, m.cbTitle, { size: 11, fill: c.muted });
      return;
    }
    let h = Math.max(60, Math.min(r.h, 420)), y0 = r.y + (r.h - h) / 2;
    const w = 12;
    if (m.cbTitle && y0 < 20) {
      // keep room above the bar for its title
      h -= 20 - y0;
      y0 = 20;
    }
    const map = m.cmap === "diverging" ? (c.dark ? MAPS.divergingDark : MAPS.divergingLight) : MAPS.viridis;
    const N = 64;
    for (let k = 0; k < N; k++) {
      const t = (k + 0.5) / N, j = Math.round(t * 255) * 3;
      this.svg.append(svg("rect", { x, y: y0 + h - ((k + 1) / N) * h, width: w, height: h / N + 0.6, fill: `rgb(${map[j]},${map[j + 1]},${map[j + 2]})`, stroke: "none" }));
    }
    this.svg.append(svg("rect", { x, y: y0, width: w, height: h, fill: "none", stroke: c.lineStrong }));
    const lo = m.vmin, hi = m.vmax;
    if (m.log) {
      const l0 = Math.log10(lo), l1 = Math.log10(hi);
      for (let e = Math.ceil(l0); e <= Math.floor(l1); e++) {
        const py = y0 + h - ((e - l0) / (l1 - l0)) * h;
        this.svg.append(svg("line", { x1: x + w, x2: x + w + 4, y1: py, y2: py, stroke: c.lineStrong }));
        this.text(x + w + 7, py, `1e${e}`.replace(/-/g, "−"), { size: 10.5, mono: true, fill: c.muted, baseline: "middle" });
      }
    } else if (hi > lo) {
      const t = ticks(lo, hi, Math.max(2, Math.floor(h / 45)));
      for (const v of t.values) {
        const py = y0 + h - ((v - lo) / (hi - lo)) * h;
        if (py < y0 - 0.5 || py > y0 + h + 0.5) continue;
        this.svg.append(svg("line", { x1: x + w, x2: x + w + 4, y1: py, y2: py, stroke: c.lineStrong }));
        this.text(x + w + 7, py, tickText(v, t.step), { size: 10.5, mono: true, fill: c.muted, baseline: "middle" });
      }
    }
    if (m.cbTitle) {
      const words = m.cbTitle;
      this.text(x - 2, y0 - 9, words, { size: 11, fill: c.text });
    }
  }

  dataAt(e) {
    if (!this.m || !this.rect) return null;
    const b = this.svg.getBoundingClientRect();
    const mx = e.clientX - b.left, my = e.clientY - b.top, r = this.rect;
    if (mx < r.x || mx > r.x + r.w || my < r.y || my > r.y + r.h) return null;
    const x = this.v[0] + ((mx - r.x) / r.w) * (this.v[1] - this.v[0]);
    const y = this.v[3] - ((my - r.y) / r.h) * (this.v[3] - this.v[2]);
    return { mx, my, x, y };
  }

  hover(e) {
    if (this.panning) return;
    const p = this.dataAt(e);
    const m = this.m;
    if (!p) return this.hideReadout();
    const iu = bisect(m.uEdges, p.x), iv = bisect(m.vEdges, p.y);
    const unit = m.coordUnit ?? "µm";
    const lines = [`${m.uName || "u"} = ${p.x.toFixed(m.coordDigits ?? 4)} ${unit}`, `${m.vName || "v"} = ${p.y.toFixed(m.coordDigits ?? 4)} ${unit}`];
    if (iu >= 0 && iv >= 0) {
      const val = m.data[iu * m.nv + iv];
      if (m.categories) lines.push(`${m.valueName || "value"}: ${Number.isFinite(val) ? m.categories[Math.round(val)] : "–"}`);
      else lines.push(`${m.valueName || "value"} = ${Number.isFinite(val) ? fmt(val, 5) : (m.nanText || "–")}${m.unit ? " " + m.unit : ""}`);
      if (m.cellLabel) lines.push(m.cellLabel(iu, iv));
      if (m.extra) lines.push(...m.extra(iu, iv));
    }
    this.showReadout(p.mx, p.my, lines);
  }

  wheel(e) {
    const p = this.dataAt(e);
    if (!p) return;
    e.preventDefault();
    const f = Math.exp((e.deltaY > 0 ? 1 : -1) * Math.min(0.3, Math.abs(e.deltaY) / 300 + 0.08));
    const v = this.v;
    let nv = [p.x - (p.x - v[0]) * f, p.x + (v[1] - p.x) * f, p.y - (p.y - v[2]) * f, p.y + (v[3] - p.y) * f];
    const [ux0, ux1, vy0, vy1] = this.extent();
    if (nv[1] - nv[0] >= (ux1 - ux0) && nv[3] - nv[2] >= (vy1 - vy0)) nv = null;
    this.view = nv;
    this.hideReadout();
    this.schedule();
  }

  pan(e) {
    if (e.button !== 0 || !this.view) return;
    const p0 = this.dataAt(e);
    if (!p0) return;
    this.panning = true;
    const v0 = this.v.slice();
    const kx = (v0[1] - v0[0]) / this.rect.w, ky = (v0[3] - v0[2]) / this.rect.h;
    const sx0 = e.clientX, sy0 = e.clientY;
    const move = (ev) => {
      const dx = (ev.clientX - sx0) * kx, dy = (ev.clientY - sy0) * ky;
      this.view = [v0[0] - dx, v0[1] - dx, v0[2] + dy, v0[3] + dy];
      this.schedule();
    };
    const up = () => { this.panning = false; window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  schedule() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this.render(); });
  }

  render() {
    this._colors = this._colorsTheme === document.documentElement.dataset.theme ? this._colors : null;
    this._colorsTheme = document.documentElement.dataset.theme;
    super.render();
  }
}
