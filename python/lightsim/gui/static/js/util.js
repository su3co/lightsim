// Shared helpers: requests, decoding, number formatting, theme colours, files.

export const C0 = 299792458;

export async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("api/" + path, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

export function decode(b64) {
  // base64 little-endian float32 -> Float32Array
  const s = atob(b64);
  const buf = new ArrayBuffer(s.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
  return new Float32Array(buf);
}

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

export const SVGNS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  return e;
}

// numbers ---------------------------------------------------------------
export function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(Math.max(0, digits - 1)).replace(/-/g, "−");
  return Number(v.toPrecision(digits)).toString().replace(/-/g, "−");
}
export function fmtFixed(v, d = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  return v.toFixed(d).replace(/-/g, "−");
}
export function fmtBytes(b) {
  const u = ["B", "kB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b.toPrecision(3)} ${u[i]}`;
}
export function fmtTime(s) {
  if (s === null || s === undefined || !Number.isFinite(s)) return "–";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60), r = Math.round(s - 60 * m);
  if (m < 60) return `${m} min ${r} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}
export function fmtCount(n) {
  if (n >= 1e9) return (n / 1e9).toPrecision(3) + " G";
  if (n >= 1e6) return (n / 1e6).toPrecision(3) + " M";
  if (n >= 1e4) return (n / 1e3).toPrecision(3) + " k";
  return String(n);
}

// theme -------------------------------------------------------------------
export function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (k) => cs.getPropertyValue(k).trim();
  return {
    bg: g("--bg"), panel: g("--panel"), line: g("--line"), lineStrong: g("--line-strong"), text: g("--text"), muted: g("--muted"),
    faint: g("--faint"), accent: g("--accent"), warn: g("--warn"), src: g("--src"), mon: g("--mon"), sym: g("--sym"), pml: g("--pml"),
    mesh: g("--mesh"), halo: g("--halo"), grid: g("--grid"), nan: g("--nan"), bad: g("--bad"),
    fontHead: g("--font-head"), fontMono: g("--font-mono"), fontText: g("--font-text"),
    dark: document.documentElement.dataset.theme === "dark" || (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches),
  };
}
export const SERIES = ["#4fa3c7", "#e0a43a", "#d4626b", "#7dbb6a", "#a479c9", "#5fc4b0", "#d96fa6", "#9aa86a", "#8f9fd9", "#c98a5f"];

// files ---------------------------------------------------------------
export function download(name, blobOrText, type = "text/plain") {
  const blob = blobOrText instanceof Blob ? blobOrText : new Blob([blobOrText], { type });
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
export function csv(header, columns) {
  const n = Math.max(...columns.map((c) => c.length));
  const lines = [header.join(",")];
  for (let i = 0; i < n; i++) lines.push(columns.map((c) => (c[i] === undefined || c[i] === null ? "" : String(c[i]))).join(","));
  return lines.join("\n") + "\n";
}
export function safeName(s) {
  return String(s).replace(/[^A-Za-z0-9_.-]+/g, "_");
}

export function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 6000);
}

export function debounce(f, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); };
}

// spectral axes ----------------------------------------------------------
export const AXES = ["x", "y", "z"];
