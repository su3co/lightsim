// Overlays of the set-up and run views: sources, monitors, ports, absorbing
// layers, symmetry planes, mesh lines and the outline of a selected
// structure, intersected with the current cut. Coordinates in µm.

const ALL = [0, 1, 2];

export function planeAxes(axis) {
  return { 0: [1, 2], 1: [0, 2], 2: [0, 1] }[axis];
}

// does a box [min, max] meet the cut slab [lo, hi] along the normal?
function meets(g, a, lo, hi) {
  const t = 1e-9;
  return g.min[a] <= hi + t && g.max[a] >= lo - t;
}

function boxOverlay(g, u, v, style) {
  const du = Math.abs(g.max[u] - g.min[u]), dv = Math.abs(g.max[v] - g.min[v]);
  if (du === 0 && dv === 0) return [{ t: "point", x: g.min[u], y: g.min[v], ...style }];
  if (du === 0 || dv === 0) return [{ t: "line", x0: g.min[u], y0: g.min[v], x1: g.max[u], y1: g.max[v], ...style }];
  return [{ t: "rect", x0: g.min[u], y0: g.min[v], x1: g.max[u], y1: g.max[v], ...style }];
}

// clip infinite extents to the region
function clipToRegion(g, info) {
  const out = { ...g, min: g.min.slice(), max: g.max.slice() };
  const n = info.nodes;
  for (const a of ALL) {
    out.min[a] = Math.max(out.min[a], n[a][0]);
    out.max[a] = Math.min(out.max[a], n[a][n[a].length - 1]);
  }
  return out;
}

export function buildOverlays(info, cut, show, selected, colors) {
  const a = cut.axis;
  const [u, v] = planeAxes(a);
  const lo = cut.coord_um, hi = cut.coord_hi_um;
  const ov = [];
  const nodes = info.nodes;
  // absorbing layers
  if (show.pml) {
    for (const [k, ax] of [[u, u], [v, v]]) {
      const sides = info.boundaries["xyz"[ax]];
      sides.forEach((s) => {
        if (s.type !== "pml") return;
        const other = ax === u ? v : u;
        const o0 = nodes[other][0], o1 = nodes[other][nodes[other].length - 1];
        if (ax === u) ov.push({ t: "hatch", x0: s.min, x1: s.max, y0: o0, y1: o1 });
        else ov.push({ t: "hatch", y0: s.min, y1: s.max, x0: o0, x1: o1 });
      });
      void k;
    }
  }
  if (show.mesh) ov.push({ t: "mesh", u: nodes[u], v: nodes[v], color: colors.mesh });
  // symmetry planes and walls
  if (show.sym) {
    for (const s of info.symmetry) {
      if (s.axis === a) continue;
      const style = { color: colors.sym, dash: "7 3 2 3", width: 1.4, label: s.kind };
      if (s.axis === u) ov.push({ t: "line", x0: s.position, x1: s.position, y0: nodes[v][0], y1: nodes[v][nodes[v].length - 1], ...style });
      else ov.push({ t: "line", y0: s.position, y1: s.position, x0: nodes[u][0], x1: nodes[u][nodes[u].length - 1], ...style });
    }
  }
  // selected structure outline
  const sel = selected;
  for (const s of info.structures) {
    if (!sel || sel !== s.id) continue;
    const o = s.outline;
    const g = clipToRegion(o, info);
    const style = { color: colors.text, width: 2.2, label: s.name, priority: true };
    if (!meets(g, a, lo, hi)) { ov.push(...boxOverlay(g, u, v, { ...style, dash: "4 3" })); continue; }
    if (o.type === "poly_slab" && o.axis === a) ov.push({ t: "poly", pts: o.vertices, ...style });
    else if ((o.type === "cylinder" && o.axis === a) || o.type === "sphere") {
      let r = o.radius / 1e-6;
      if (o.type === "sphere") { const d = (lo + hi) / 2 - o.center[a] / 1e-6; r = Math.sqrt(Math.max(0, r * r - d * d)); }
      ov.push({ t: "circle", x: o.center[u] / 1e-6, y: o.center[v] / 1e-6, r, ...style });
    } else ov.push(...boxOverlay(g, u, v, style));
  }
  const objects = [];
  if (show.sources) for (const s of info.sources) objects.push({ ...s, color: colors.src });
  if (show.monitors) for (const m of info.monitors) objects.push({ ...m, color: colors.mon });
  for (const o of objects) {
    const g = clipToRegion(o.geometry, info);
    const isSel = sel === o.id;
    const style = { color: isSel ? colors.text : o.color, width: isSel ? 3.2 : 1.6, label: o.name, priority: isSel };
    if (!meets(g, a, lo, hi)) {
      if (isSel) ov.push(...boxOverlay(g, u, v, { ...style, dash: "4 3", opacity: 0.8 }));
      continue;
    }
    ov.push(...boxOverlay(g, u, v, style));
    const d = o.geometry.direction;
    if (d && o.geometry.axis !== undefined && o.geometry.axis !== a) {
      const ax = o.geometry.axis;
      const cx = (g.min[u] + g.max[u]) / 2, cy = (g.min[v] + g.max[v]) / 2;
      const sgn = d === "+" ? 1 : -1;
      ov.push({ t: "arrow", x: cx, y: cy, dx: ax === u ? sgn : 0, dy: ax === v ? sgn : 0, color: o.color, width: 1.6 });
    }
  }
  if (show.ports && info.ports && info.ports.length && a === 2) {
    for (const p of info.ports) {
      const ang = (p.orientation || 0) * Math.PI / 180;
      const c = p.center, w = p.width || 0.5;
      const tx = -Math.sin(ang), ty = Math.cos(ang);
      ov.push({ t: "line", x0: c[0] - tx * w / 2, y0: c[1] - ty * w / 2, x1: c[0] + tx * w / 2, y1: c[1] + ty * w / 2, color: colors.warn, width: 2.4, label: p.name });
      ov.push({ t: "arrow", x: c[0], y: c[1], dx: Math.cos(ang), dy: Math.sin(ang), color: colors.warn, width: 1.4 });
    }
  }
  return ov;
}
