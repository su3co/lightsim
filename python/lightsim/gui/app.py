"""Data side of the graphical interface: everything the browser asks for,
computed from a simulation spec, the compiled core and a results dict.

Nothing here knows about HTTP; :mod:`lightsim.gui.server` maps requests to
the methods of :class:`Session`. Lengths sent to the browser are in
micrometres, frequencies in hertz, times in seconds; 2-D arrays are
row-major ``[u, v]`` float32, base64-encoded, with ``u`` the horizontal and
``v`` the vertical axis of the picture.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from collections import OrderedDict

import numpy as np

from .. import _core
from ..spec import C0

UM = 1e-6
AXES = "xyz"
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
# the largest picture side sent for a cut or a field map (samples)
MAX_SIDE = 900
# the live field slice of a run
LIVE_SIDE = 600


def b64(a) -> str:
    """A float array as base64 of little-endian float32."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def plane_axes(axis: int) -> tuple[int, int]:
    """In-plane axes (u, v) of a plane normal to ``axis``, in increasing order
    (as the core's mode windows and cuts)."""
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


def stride_for(n: int, side: int) -> int:
    return max(1, int(math.ceil(n / side)))


def downsample(a: np.ndarray, edges_u, edges_v, side: int = MAX_SIDE):
    """Every k-th sample of a 2-D array (nearest, not averaged, so that values
    stay those of the grid) and the matching cell edges."""
    ku, kv = stride_for(a.shape[0], side), stride_for(a.shape[1], side)
    eu = np.asarray(edges_u, dtype=float)
    ev = np.asarray(edges_v, dtype=float)
    a = a[::ku, ::kv]
    eu = np.append(eu[: -1 : ku][: a.shape[0]], eu[-1])
    ev = np.append(ev[: -1 : kv][: a.shape[1]], ev[-1])
    return a, eu, ev, (ku, kv)


def finite_range(a) -> tuple[float, float]:
    f = np.asarray(a, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return 0.0, 1.0
    return float(f.min()), float(f.max())


def _num(v):
    """JSON-safe float (None for NaN / inf)."""
    v = float(v)
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# results from the command line (`lightsim run --out`) as the Python dict
# ---------------------------------------------------------------------------


def results_from_cli_json(d: dict) -> dict:
    """Convert the results file of ``lightsim run --out`` (flat arrays with
    ``dims``) into the dict ``lightsim.Results`` wraps."""
    raw = {"dt": d["dt"], "stats": dict(d["stats"]), "flux": {k: np.asarray(v) for k, v in d.get("flux", {}).items()}}
    dft = {}
    for name, r in d.get("dft", {}).items():
        fields = {}
        for f in r["fields"]:
            key, t = f["comp"], 1
            while key in fields:
                key = f"{f['comp']}_{t}"
                t += 1
            dims = f["dims"]
            fields[key] = {"origin": f["origin"], "re": np.asarray(f["re"], dtype=np.float32).reshape(dims), "im": np.asarray(f["im"], dtype=np.float32).reshape(dims)}
        dft[name] = {"freqs": np.asarray(r["freqs"]), "stride": r.get("stride", 0), "fields": fields}
    raw["dft"] = dft
    modes = {}
    for name, m in d.get("modes", {}).items():
        def cplx(re, im):
            return [{"re": np.asarray(a), "im": np.asarray(b)} for a, b in zip(re, im)]

        modes[name] = {"freqs": np.asarray(m["freqs"]), "plus": cplx(m["amp_plus_re"], m["amp_plus_im"]), "minus": cplx(m["amp_minus_re"], m["amp_minus_im"]), "flux": np.asarray(m["flux"])}
    raw["modes"] = modes
    time_ = {}
    for name, t in d.get("time", {}).items():
        time_[name] = {"times": np.asarray(t["times"]), "fields": {f["comp"]: np.asarray(f["data"], dtype=np.float32).reshape(f["dims"]) for f in t["fields"]}}
    raw["time"] = time_
    far = {}
    for name, f in d.get("far_field", {}).items():
        dims = (len(f["freqs"]), len(f["theta"]), len(f["phi"]))
        far[name] = {k: np.asarray(f[k]) for k in ("freqs", "theta", "phi", "flux")}
        for k in ("intensity", "e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im"):
            far[name][k] = np.asarray(f.get(k) or np.zeros(dims)).reshape(dims)
    raw["far_field"] = far
    raw["energy"] = {k: {"times": np.asarray(e["times"]), "energy": np.asarray(e["energy"])} for k, e in d.get("energy", {}).items()}
    raw["sources"] = [{"name": s["name"], "signal": json.dumps(s["signal"]), "analytic": bool(s.get("analytic", False))} for s in d.get("sources", [])]
    raw["source_spectra"] = {}
    return raw


# ---------------------------------------------------------------------------
# geometry of the spec objects (for overlays and highlighting only)
# ---------------------------------------------------------------------------


def shape_outline(shape: dict) -> dict:
    """Bounding box (m) of a shape, plus what is needed to draw its outline on
    a cut: the polygon of a ``poly_slab``, the circle of a cylinder or a
    sphere."""
    t = shape.get("type")
    out: dict = {"type": t}
    if t in ("box", "index_grid"):
        lo, hi = list(shape["min"]), list(shape["max"])
    elif t == "sphere":
        c, r = shape["center"], shape["radius"]
        lo, hi = [x - r for x in c], [x + r for x in c]
        out["center"], out["radius"] = list(c), r
    elif t == "cylinder":
        c, r, a = shape["center"], shape["radius"], int(shape.get("axis") or 2)
        L = shape.get("length") or 0.0
        lo, hi = [x - r for x in c], [x + r for x in c]
        lo[a], hi[a] = c[a] - 0.5 * L, c[a] + 0.5 * L
        out["center"], out["radius"], out["axis"] = list(c), r, a
    elif t == "poly_slab":
        a = int(shape.get("axis") or 2)
        u, v = plane_axes(a)
        vs = np.asarray(shape["vertices"], dtype=float).reshape(-1, 2)
        lo, hi = [0.0] * 3, [0.0] * 3
        lo[u], hi[u] = float(vs[:, 0].min()), float(vs[:, 0].max())
        lo[v], hi[v] = float(vs[:, 1].min()), float(vs[:, 1].max())
        lo[a], hi[a] = float(shape["slab_min"]), float(shape["slab_max"])
        out["axis"] = a
        out["vertices"] = (vs / UM).round(6).tolist()
    else:
        lo, hi = [-1e9] * 3, [1e9] * 3
    out["min"], out["max"] = lo, hi
    return out


def _vec(v):
    return None if v is None else [float(x) for x in v]


def source_geometry(s: dict, region_min, region_max) -> dict:
    """Where a source acts: a point, a plane window or a box (m)."""
    t = s.get("type")
    if t == "point_dipole":
        p = _vec(s["position"])
        return {"kind": "point", "min": p, "max": p}
    if t == "tfsf":
        return {"kind": "box", "min": _vec(s["min"]), "max": _vec(s["max"])}
    a = int(s.get("axis", 0))
    lo = _vec(s.get("min")) or list(region_min)
    hi = _vec(s.get("max")) or list(region_max)
    lo, hi = list(lo), list(hi)
    lo[a] = hi[a] = float(s["position"])
    return {"kind": "plane", "axis": a, "min": lo, "max": hi, "direction": s.get("direction")}


def monitor_geometry(m: dict) -> dict:
    t = m.get("type")
    if t == "probe":
        p = _vec(m["position"])
        return {"kind": "point", "min": p, "max": p}
    if t in ("flux", "mode"):
        a = int(m["axis"])
        lo, hi = list(_vec(m["min"])), list(_vec(m["max"]))
        lo[a] = hi[a] = float(m["position"])
        return {"kind": "plane", "axis": a, "min": lo, "max": hi}
    lo, hi = _vec(m["min"]), _vec(m["max"])
    flat = [a for a in range(3) if hi[a] == lo[a]]
    return {"kind": "plane" if len(flat) == 1 else ("box" if not flat else "line"), "axis": flat[0] if len(flat) == 1 else None, "min": lo, "max": hi}


def _um(v):
    return None if v is None else [round(x / UM, 6) for x in v]


def _summary_params(d: dict, skip=("type", "name", "profile", "modes", "signal", "vertices", "eps", "shape")) -> dict:
    """Short parameters of a spec object for the side list (lengths in µm)."""
    out = {}
    for k, v in d.items():
        if k in skip or v is None:
            continue
        if k in ("min", "max", "position", "center") and isinstance(v, (list, tuple)):
            out[k] = [round(x / UM, 4) for x in v] if isinstance(v, (list, tuple)) else v
        elif k in ("position", "waist", "radius", "length", "slab_min", "slab_max", "focus_distance") and isinstance(v, (int, float)):
            out[k] = round(v / UM, 4)
        elif k == "freqs":
            f = np.asarray(v, dtype=float)
            out["freqs"] = f"{len(f)} from {f.min() / 1e12:.3f} to {f.max() / 1e12:.3f} THz" if len(f) > 1 else f"{f[0] / 1e12:.3f} THz"
        elif k in ("theta", "phi"):
            out[k] = f"{len(v)} angles"
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 6 and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict) and len(v) <= 6:
            out[k] = {a: b for a, b in v.items() if isinstance(b, (int, float, str, bool))}
    if "signal" in d and isinstance(d["signal"], dict):
        sg = d["signal"]
        out["signal"] = sg.get("type", "")
        if "freq0" in sg:
            out["freq0"] = f"{sg['freq0'] / 1e12:.3f} THz ({C0 / sg['freq0'] / UM * 1e3:.1f} nm)"
        if "fwidth" in sg:
            out["fwidth"] = f"{sg['fwidth'] / 1e12:.3f} THz"
    if "profile" in d and isinstance(d["profile"], dict):
        out["neff"] = round(float(d["profile"].get("neff", 0.0)), 5)
    if "modes" in d:
        out["modes"] = f"{len(d['modes'])} profiles: neff " + ", ".join(f"{float(p.get('neff', 0)):.4f}" for p in d["modes"][:6])
    return out


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------


class Session:
    """A simulation (spec dict), its grid and media, optional results, and
    the state of a run started from the interface."""

    def __init__(self, spec, results=None, ports=None, title: str | None = None):
        if spec is None and results is None:
            raise ValueError("give a simulation, a spec or results")
        self.spec = None
        if spec is not None:
            if hasattr(spec, "to_dict"):
                spec = spec.to_dict()
            elif isinstance(spec, str):
                spec = json.loads(spec)
            self.spec = dict(spec)
        if results is not None and hasattr(results, "raw"):
            results = results.raw
        self.results = results
        self.ports = list(ports or [])
        self.title = title or (self.spec or {}).get("name") or "simulation"
        self._lock = threading.Lock()
        self._cuts: OrderedDict = OrderedDict()
        self.run = RunState()
        if self.spec is not None:
            self.spec_json = json.dumps(self.spec)
            g = dict(self.spec, sources=[], monitors=[])
            self.geom_json = json.dumps(g)
            self.nodes = [np.asarray(a, dtype=float) for a in _core.build_grid(self.geom_json)]
            self.media = list(_core.media_info(self.geom_json))
            self.dt, self.nsteps = _core.time_info(self.spec_json)

    # -- basic facts --------------------------------------------------------
    @property
    def has_spec(self) -> bool:
        return self.spec is not None

    def dims(self) -> int:
        return 2 if self.spec.get("dimensions") == 2 else 3

    def invariant(self):
        if self.dims() != 2:
            return None
        a = self.spec.get("invariant_axis", 2)
        return AXES.index(a) if isinstance(a, str) else int(a)

    def n(self) -> list[int]:
        return [len(a) - 1 for a in self.nodes]

    def region(self):
        r = self.spec["region"]
        return [float(v) for v in r["min"]], [float(v) for v in r["max"]]

    # -- set-up -------------------------------------------------------------
    def info(self) -> dict:
        out = {"title": self.title, "has_spec": self.has_spec, "has_results": self.results is not None,
               "devices": devices(), "precision": getattr(_core, "precision", "f32"), "version": _core.__version__}
        if not self.has_spec:
            return out
        rmin, rmax = self.region()
        out.update(
            dims=self.dims(),
            invariant=self.invariant(),
            polarization=self.spec.get("polarization", "both") if self.dims() == 2 else None,
            n=self.n(),
            nodes=[(a / UM).round(6).tolist() for a in self.nodes],
            region={"min": _um(rmin), "max": _um(rmax)},
            boundaries=self.boundaries(),
            symmetry=self.symmetry_planes(),
            media=self.media,
            structures=self.structures(),
            sources=self.sources(),
            monitors=self.monitors(),
            ports=[{"name": p.get("name"), "center": _um(list(p["center"]) + [0.0] * (3 - len(p["center"]))), "orientation": p.get("orientation"), "width": p.get("width", 0) / UM} for p in self.ports],
            summary=self.summary(),
            quantities=self.quantities(),
            run_defaults=self.run_defaults(),
        )
        return out

    def boundaries(self) -> dict:
        """Per axis and side: the condition and, for absorbing layers, the
        extent (µm) of the layer."""
        out = {}
        b = self.spec.get("boundaries") or {}
        for a, k in enumerate(AXES):
            e = self.nodes[a]
            sides = []
            for s, bd in enumerate(b.get(k, [{"type": "pml"}, {"type": "pml"}])):
                t = bd.get("type", "pml")
                d = {"type": t}
                if t in ("pml", "absorber"):
                    L = int(bd.get("layers", 12 if t == "pml" else 40))
                    L = min(L, len(e) - 1)
                    lo, hi = (e[0], e[L]) if s == 0 else (e[-1 - L], e[-1])
                    d.update(layers=L, min=round(lo / UM, 6), max=round(hi / UM, 6), inside=bool(bd.get("inside", False)))
                elif t == "bloch":
                    d["k"] = bd.get("k")
                sides.append(d)
            out[k] = sides
        return out

    def symmetry_planes(self) -> list:
        """Mirror planes: those of an automatic mesh, and electric or magnetic
        walls on the faces of the region (the planes a symmetric run keeps)."""
        out = []
        mesh = self.spec.get("mesh", {})
        for a, c in enumerate(mesh.get("symmetry") or []):
            if c is not None:
                out.append({"axis": a, "position": round(float(c) / UM, 6), "kind": "mesh symmetry"})
        b = self.spec.get("boundaries") or {}
        for a, k in enumerate(AXES):
            for s, bd in enumerate(b.get(k, [])):
                if bd.get("type") in ("pec", "pmc"):
                    e = self.nodes[a]
                    out.append({"axis": a, "position": round((e[0] if s == 0 else e[-1]) / UM, 6), "kind": f"{bd['type'].upper()} wall"})
        return out

    def structures(self) -> list:
        out = []
        media = {m["name"]: m for m in self.media}
        for i, s in enumerate(self.spec.get("structures", [])):
            o = shape_outline(s["shape"])
            m = media.get(s["medium"], {})
            eps = m.get("eps", [1.0])
            out.append({"id": f"structure:{i}", "name": s.get("name") or f"{s['shape'].get('type')} {i}", "medium": s["medium"],
                        "shape": s["shape"].get("type"), "outline": dict(o, min=_um(o["min"]), max=_um(o["max"])),
                        "params": dict(_summary_params(s["shape"]), medium=s["medium"], n=round(math.sqrt(max(eps)), 4) if not m.get("pec") else "PEC")})
        return out

    def sources(self) -> list:
        rmin, rmax = self.region()
        out = []
        for i, s in enumerate(self.spec.get("sources", [])):
            g = source_geometry(s, rmin, rmax)
            out.append({"id": f"source:{i}", "name": s.get("name") or f"source{i}", "type": s.get("type"),
                        "geometry": dict(g, min=_um(g["min"]), max=_um(g["max"])), "params": _summary_params(s)})
        return out

    def monitors(self) -> list:
        out = []
        for i, m in enumerate(self.spec.get("monitors", [])):
            g = monitor_geometry(m)
            out.append({"id": f"monitor:{i}", "name": m.get("name"), "type": m.get("type"),
                        "geometry": dict(g, min=_um(g["min"]), max=_um(g["max"])), "params": _summary_params(m)})
        return out

    def quantities(self) -> list:
        q = [("n", "index n = √ε", ""), ("eps_x", "ε of Ex", ""), ("eps_y", "ε of Ey", ""), ("eps_z", "ε of Ez", ""), ("medium", "medium", "")]
        if any(any(s > 0 for s in m["sigma"]) for m in self.media):
            q.append(("sigma", "conductivity σ", "S/m"))
        if any(m["dispersive"] for m in self.media):
            q.append(("dispersive", "dispersive cells", ""))
        if any(m["nonlinear"] for m in self.media):
            q.append(("nonlinear", "nonlinear cells", ""))
        return [{"id": a, "label": b, "unit": c} for a, b, c in q]

    def freq_band(self):
        """(f_min, f_max) of the monitors, or of the sources if no monitor
        records frequencies."""
        fs = [float(f) for m in self.spec.get("monitors", []) for f in (m.get("freqs") or [])]
        if fs:
            return min(fs), max(fs), "monitors"
        lo, hi = [], []
        for s in self.spec.get("sources", []):
            sg = s.get("signal") or {}
            if "freq0" in sg:
                w = float(sg.get("fwidth", 0.0))
                lo.append(sg["freq0"] - 2 * w)
                hi.append(sg["freq0"] + 2 * w)
        if hi:
            return max(min(lo), 1.0), max(hi), "sources (centre ± 2 widths)"
        return None, None, None

    def summary(self) -> dict:
        n = self.n()
        cells = int(np.prod(n))
        out = {"n": n, "cells": cells, "dt": self.dt, "nsteps": int(self.nsteps), "run_time": float(self.spec.get("run_time", 0.0)),
               "shutoff": self.spec.get("shutoff"), "shutoff_interval": self.spec.get("shutoff_interval"), "courant": self.spec.get("courant", 0.99),
               "subpixel": bool(self.spec.get("subpixel", True)), "warnings": []}
        d = [np.diff(a) for a in self.nodes]
        inv = self.invariant()
        act = [a for a in range(3) if a != inv]
        out["min_step_um"] = [round(float(x.min()) / UM, 5) for x in d]
        out["max_step_um"] = [round(float(x.max()) / UM, 5) for x in d]
        out["memory"] = self.memory()
        fmin, fmax, src = self.freq_band()
        if fmax:
            out["band"] = {"f_min": fmin, "f_max": fmax, "lambda_min_um": C0 / fmax / UM, "lambda_max_um": C0 / fmin / UM, "from": src}
            nmax, name, dmax = self.densest()
            lam = C0 / fmax / nmax
            cpw = lam / dmax
            out["densest"] = {"medium": name, "n": round(nmax, 4), "max_step_um": round(dmax / UM, 5), "cells_per_wavelength": round(cpw, 2)}
            if cpw < 10:
                out["warnings"].append(f"coarse mesh: {cpw:.1f} cells per wavelength in {name} (n = {nmax:.3f}) at {C0 / fmax / UM * 1e3:.0f} nm; 10 or more is the usual minimum, 15 to 20 for accurate phase")
        if inv is None and cells > 5e9:
            out["warnings"].append(f"{cells / 1e9:.1f} billion cells")
        steps_per_period = None
        if fmax:
            steps_per_period = 1.0 / (fmax * self.dt)
            out["steps_per_period"] = round(steps_per_period, 1)
        if self.spec.get("shutoff") is None and any((s.get("signal") or {}).get("type") == "gaussian_pulse" for s in self.spec.get("sources", [])):
            out["warnings"].append("no shutoff: the run lasts the full run_time")
        if not self.spec.get("sources"):
            out["warnings"].append("no source")
        if not self.spec.get("monitors"):
            out["warnings"].append("no monitor: a run returns nothing to look at")
        _ = act
        return out

    def densest(self):
        """(index, medium name, largest cell size in it): the medium of the
        highest index among those used, and the largest step of the cells that
        overlap its structures (all cells for the background)."""
        used = {self.spec.get("background", "vacuum")} | {s["medium"] for s in self.spec.get("structures", [])}
        best = (1.0, self.spec.get("background", "vacuum"))
        for m in self.media:
            if m["name"] in used and not m["pec"]:
                nn = math.sqrt(max(max(m["eps"]), 1e-12))
                if nn > best[0]:
                    best = (nn, m["name"])
        for s in self.spec.get("structures", []):
            sh = s["shape"]
            if sh.get("type") == "index_grid" and sh.get("eps"):
                nn = math.sqrt(float(np.max(sh["eps"])))
                if nn > best[0]:
                    best = (nn, s.get("name") or "index grid")
        nmax, name = best
        inv = self.invariant()
        boxes = [shape_outline(s["shape"]) for s in self.spec.get("structures", []) if s["medium"] == name or (s.get("name") == name)]
        dmax = 0.0
        for a in range(3):
            if a == inv:
                continue
            e = self.nodes[a]
            c0, c1 = e[:-1], e[1:]
            mask = np.zeros(len(c0), dtype=bool)
            if name == self.spec.get("background", "vacuum") or not boxes:
                mask[:] = True
            for o in boxes:
                mask |= (c1 > o["min"][a]) & (c0 < o["max"][a])
            if mask.any():
                dmax = max(dmax, float((c1 - c0)[mask].max()))
        return nmax, name, max(dmax, 1e-30)

    def memory(self) -> dict:
        """Approximate device memory (bytes): field and coefficient arrays,
        absorbing layers, frequency-domain and time monitors. Dispersive
        auxiliary fields are not counted."""
        r = 8 if getattr(_core, "precision", "f32") == "f64" else 4
        n = self.n()
        ny, nz = n[1], n[2]
        storage = (n[0] + 2) * ny * nz
        lossy = any(m["lossy"] for m in self.media)
        magnetic = any(m["magnetic"] for m in self.media)
        b = self.boundaries()
        # an adiabatic absorber needs the loss coefficients of both fields on the whole grid
        absorber = any(s["type"] == "absorber" for k in AXES for s in b[k])
        arrays = 9 + (3 if lossy or absorber else 0) + (3 if magnetic or absorber else 0)
        fields = arrays * storage * r
        pml = 0
        for a, k in enumerate(AXES):
            for s in b[k]:
                if s["type"] == "absorber":
                    # no auxiliary fields; one magnetic-loss entry (index and factor) per H sample
                    face = int(np.prod([n[x] for x in range(3) if x != a]))
                    pml += 3 * s["layers"] * face * (8 + r)
                if s["type"] == "pml":
                    face = int(np.prod([n[x] for x in range(3) if x != a]))
                    pml += 4 * s["layers"] * face * r
        mon = 0
        e = self.nodes

        def cells_in(lo, hi, flat_axis=None):
            c = 1
            for a in range(3):
                if a == flat_axis:
                    continue
                i0 = int(np.searchsorted(e[a], lo[a], side="right")) - 1
                i1 = int(np.searchsorted(e[a], hi[a], side="left"))
                c *= max(1, min(i1, n[a]) - max(i0, 0))
            return c

        for m in self.spec.get("monitors", []):
            t = m.get("type")
            nf = len(m.get("freqs") or [])
            if t in ("flux", "mode"):
                mon += 4 * cells_in(m["min"], m["max"], int(m["axis"])) * nf * 2 * r
            elif t in ("flux_box", "far_field"):
                lo, hi = m["min"], m["max"]
                for a in range(3):
                    mon += 2 * 4 * cells_in(lo, hi, a) * nf * 2 * r
            elif t == "dft_field":
                mon += len(m.get("components", [])) * cells_in(m["min"], m["max"]) * nf * 2 * r
            elif t == "field_time":
                mon += len(m.get("components", [])) * cells_in(m["min"], m["max"]) * (self.nsteps // max(1, int(m.get("interval", 1))) + 1) * r
            elif t == "probe":
                mon += len(m.get("components", [])) * self.nsteps * r
            elif t == "energy":
                mon += 6 * cells_in(m["min"], m["max"]) * (self.nsteps // max(1, int(m.get("interval", 1))) + 1) * r
        total = fields + pml + mon
        per = {}
        for ndev in (1, 2, 4):
            # slabs along x: fields and layers split, plus two ghost planes each
            per[str(ndev)] = int(total / ndev + 2 * arrays * ny * nz * r)
        return {"total": int(total), "fields": int(fields), "pml": int(pml), "monitors": int(mon), "per_device": per, "bytes_per_real": r}

    # -- cuts ---------------------------------------------------------------
    def _cut(self, axis: int, index: int) -> dict:
        key = (axis, index)
        with self._lock:
            if key in self._cuts:
                self._cuts.move_to_end(key)
                return self._cuts[key]
        c = _core.material_cut(self.geom_json, axis, index)
        with self._lock:
            self._cuts[key] = c
            while len(self._cuts) > 12:
                self._cuts.popitem(last=False)
        return c

    def cut_quantity(self, axis: int, index: int, q: str):
        """A 2-D array ``[u, v]`` of quantity ``q`` on the cut, the categories
        of a categorical quantity, and its unit."""
        c = self._cut(axis, index)
        eps = [np.asarray(x, dtype=float) for x in c["eps"]]
        med = [np.asarray(x, dtype=np.int64) for x in c["medium"]]
        pec = np.array([m["pec"] for m in self.media], dtype=bool)
        cats = None
        unit = ""
        if q == "n":
            a = np.sqrt(np.maximum((eps[0] + eps[1] + eps[2]) / 3.0, 0.0))
            a[pec[med[0]] | pec[med[1]] | pec[med[2]]] = np.nan
        elif q in ("eps_x", "eps_y", "eps_z"):
            k = "xyz".index(q[-1])
            a = eps[k].copy()
            a[pec[med[k]]] = np.nan
        elif q == "sigma":
            sig = np.array([m["sigma"] for m in self.media], dtype=float)
            a = (sig[med[0], 0] + sig[med[1], 1] + sig[med[2], 2]) / 3.0
            unit = "S/m"
        elif q in ("dispersive", "nonlinear"):
            flag = np.array([m[q] for m in self.media], dtype=bool)
            a = (flag[med[0]] | flag[med[1]] | flag[med[2]]).astype(float)
            cats = ["no", q]
        elif q == "medium":
            u, _ = plane_axes(axis)
            a = med[u].astype(float)
            cats = [m["name"] for m in self.media]
        else:
            raise ValueError(f"unknown quantity {q!r}")
        return a, cats, unit

    def cut(self, axis: int, index: int, q: str = "n", side: int = MAX_SIDE) -> dict:
        n = self.n()
        axis = int(axis)
        index = int(min(max(int(index), 0), n[axis] - 1))
        a, cats, unit = self.cut_quantity(axis, index, q)
        u, v = plane_axes(axis)
        full_min, full_max = finite_range(a)
        a2, eu, ev, (ku, kv) = downsample(a, self.nodes[u], self.nodes[v], side)
        e = self.nodes[axis]
        return {"axis": axis, "index": index, "coord_um": round(e[index] / UM, 6), "coord_hi_um": round(e[index + 1] / UM, 6),
                "u": u, "v": v, "nu": int(a2.shape[0]), "nv": int(a2.shape[1]), "stride": [ku, kv],
                "u_edges": (eu / UM).round(6).tolist(), "v_edges": (ev / UM).round(6).tolist(),
                "data": b64(a2), "min": full_min, "max": full_max, "unit": unit, "quantity": q, "categories": cats}

    def outline_segments(self, axis: int, index: int, lo_um=None, hi_um=None, side: int = MAX_SIDE) -> list:
        """Boundaries between media on a cut, as segments ``[u0, v0, u1, v1]``
        (µm), optionally only inside ``[lo, hi]``: the structure outline drawn
        on field maps."""
        c = self._cut(axis, index)
        u, v = plane_axes(axis)
        key = np.asarray(c["medium"][u], dtype=np.int64)
        eu, ev = self.nodes[u] / UM, self.nodes[v] / UM
        iu0, iu1, iv0, iv1 = 0, key.shape[0], 0, key.shape[1]
        if lo_um is not None:
            iu0 = max(0, int(np.searchsorted(eu, lo_um[u], side="right")) - 2)
            iu1 = min(key.shape[0], int(np.searchsorted(eu, hi_um[u], side="left")) + 2)
            iv0 = max(0, int(np.searchsorted(ev, lo_um[v], side="right")) - 2)
            iv1 = min(key.shape[1], int(np.searchsorted(ev, hi_um[v], side="left")) + 2)
        k = key[iu0:iu1, iv0:iv1]
        ku, kv = stride_for(k.shape[0], side), stride_for(k.shape[1], side)
        k = k[::ku, ::kv]
        eus = np.append(eu[iu0:iu1][::ku][: k.shape[0]], eu[min(iu0 + k.shape[0] * ku, len(eu) - 1)])
        evs = np.append(ev[iv0:iv1][::kv][: k.shape[1]], ev[min(iv0 + k.shape[1] * kv, len(ev) - 1)])
        segs = []
        di = np.nonzero(k[1:, :] != k[:-1, :])
        for i, j in zip(*di):
            segs.append([eus[i + 1], evs[j], eus[i + 1], evs[j + 1]])
        dj = np.nonzero(k[:, 1:] != k[:, :-1])
        for i, j in zip(*dj):
            segs.append([eus[i], evs[j + 1], eus[i + 1], evs[j + 1]])
        if len(segs) > 40000:
            segs = segs[:: len(segs) // 40000 + 1]
        return np.round(np.asarray(segs, dtype=float).reshape(-1, 4), 5).tolist()

    def run_defaults(self) -> dict:
        """The plane and component the live view starts with."""
        n = self.n()
        inv = self.invariant()
        comp = "Ey"
        srcs = self.spec.get("sources", [])
        pol = next((s.get("polarization") for s in srcs if s.get("polarization")), None)
        if inv is not None:
            w = AXES[inv]
            p = self.spec.get("polarization", "both")
            comp = {"te": f"H{w}", "tm": f"E{w}"}.get(p, pol or f"E{w}")
            return {"comp": comp, "axis": inv, "index": 0}
        if pol:
            comp = pol
        axis = 2
        pos = None
        for s in srcs:
            g = source_geometry(s, *self.region())
            pos = 0.5 * (g["min"][2] + g["max"][2])
            break
        e = self.nodes[2]
        idx = int(np.clip(np.searchsorted(e, pos) - 1, 0, n[2] - 1)) if pos is not None else n[2] // 2
        return {"comp": comp, "axis": axis, "index": idx}

    # -- results ------------------------------------------------------------
    def results_index(self) -> dict:
        if self.results is None:
            return {"available": False, "monitors": []}
        r = self.results
        specs = {m["name"]: m for m in (self.spec or {}).get("monitors", [])}
        out = []
        for name, f in r.get("flux", {}).items():
            if name in r.get("modes", {}) or name in r.get("far_field", {}):
                continue
            out.append({"name": name, "type": specs.get(name, {}).get("type", "flux"), "views": ["spectrum"]})
        for name in r.get("modes", {}):
            out.append({"name": name, "type": "mode", "views": ["amplitudes", "profiles"]})
        for name, d in r.get("dft", {}).items():
            if name in r.get("flux", {}) or name in r.get("modes", {}) or name in r.get("far_field", {}):
                continue
            out.append({"name": name, "type": specs.get(name, {}).get("type", "dft_field"), "views": ["map"], "components": list(d["fields"].keys()), "freqs": np.asarray(d["freqs"]).tolist()})
        for name, t in r.get("time", {}).items():
            out.append({"name": name, "type": specs.get(name, {}).get("type", "time"), "views": ["trace"], "components": list(t["fields"].keys())})
        for name in r.get("far_field", {}):
            out.append({"name": name, "type": "far_field", "views": ["pattern"]})
        for name in r.get("energy", {}):
            out.append({"name": name, "type": "energy", "views": ["trace"]})
        order = {m: i for i, m in enumerate(specs)}
        out.sort(key=lambda d: order.get(d["name"], 1e9))
        sp = self.sparam_setup()
        st = r["stats"]
        stats = {k: st[k] for k in ("backend", "cells", "steps", "seconds", "mcells_per_s", "stopped_early") if k in st}
        stats["energy_steps"] = np.asarray(st.get("energy_steps", [])).tolist()
        stats["energy"] = np.asarray(st.get("energy", [])).tolist()
        stats["dt"] = r.get("dt")
        stats["shutoff"] = (self.spec or {}).get("shutoff")
        return {"available": True, "monitors": out, "sparams": sp, "stats": stats, "sources": [s["name"] for s in r.get("sources", [])]}

    def _spectrum(self, monitor: str, source: str | None = None):
        from ..spec import Results

        res = Results(self.results)
        names = [s["name"] for s in self.results.get("sources", [])]
        if not names:
            return None
        try:
            _, S = res.source_spectrum(source or names[0], monitor=monitor)
            return S
        except Exception:
            return None

    def flux_data(self, name: str) -> dict:
        r = self.results
        if name in r.get("modes", {}):
            freqs = np.asarray(r["modes"][name]["freqs"], dtype=float)
        elif name in r.get("dft", {}):
            freqs = np.asarray(r["dft"][name]["freqs"], dtype=float)
        else:
            freqs = np.asarray(next((m.get("freqs") for m in (self.spec or {}).get("monitors", []) if m.get("name") == name), []), dtype=float)
        flux = np.asarray(r["flux"][name], dtype=float)
        S = self._spectrum(name)
        out = {"name": name, "freqs": freqs.tolist(), "flux": flux.tolist()}
        if S is not None:
            out["source_power"] = (np.abs(S) ** 2).tolist()
        return out

    def mode_data(self, name: str) -> dict:
        m = self.results["modes"][name]
        freqs = np.asarray(m["freqs"], dtype=float)
        S = self._spectrum(name)
        spec = next((x for x in (self.spec or {}).get("monitors", []) if x.get("name") == name), None)
        neffs = [float(p.get("neff", float("nan"))) for p in (spec or {}).get("modes", [])]
        modes = []
        for k, (p, q) in enumerate(zip(m["plus"], m["minus"])):
            modes.append({"plus_re": np.asarray(p["re"]).tolist(), "plus_im": np.asarray(p["im"]).tolist(),
                          "minus_re": np.asarray(q["re"]).tolist(), "minus_im": np.asarray(q["im"]).tolist(),
                          "neff": neffs[k] if k < len(neffs) else None})
        out = {"name": name, "freqs": freqs.tolist(), "modes": modes, "flux": np.asarray(m["flux"], dtype=float).tolist()}
        if S is not None:
            out["source_re"] = S.real.tolist()
            out["source_im"] = S.imag.tolist()
        return out

    def sparam_setup(self):
        """The mode monitors and which one measures the launched amplitude:
        the first mode monitor downstream of the first mode source on its axis
        and inside its window (``None`` without a mode source)."""
        if not self.results or not self.results.get("modes") or not self.spec:
            return None
        mons = [m for m in self.spec.get("monitors", []) if m.get("type") == "mode" and m["name"] in self.results["modes"]]
        src = next((s for s in self.spec.get("sources", []) if s.get("type") == "mode"), None)
        ref = None
        if src is not None:
            a, x0, d = int(src["axis"]), float(src["position"]), src.get("direction", "+")
            sgn = 1 if d == "+" else -1
            cands = [(sgn * (float(m["position"]) - x0), m["name"]) for m in mons if int(m["axis"]) == a and sgn * (float(m["position"]) - x0) > 0]
            if cands:
                ref = min(cands)[1]
        return {"monitors": [m["name"] for m in mons], "reference": ref, "source": src.get("name") if src else None,
                "direction": src.get("direction") if src else None}

    def mode_profiles(self, name: str) -> dict:
        """The profiles a mode monitor (or source) projects on, with their
        window (µm) and effective index."""
        spec = next((x for x in self.spec.get("monitors", []) + self.spec.get("sources", []) if x.get("name") == name and x.get("type") == "mode"), None)
        if spec is None:
            raise KeyError(name)
        profs = spec.get("modes") or [spec.get("profile")]
        a = int(spec["axis"])
        info = _core.plane_info(self.geom_json, a, float(spec["position"]), list(spec["min"]), list(spec["max"]), None)
        u, v = info["axes"]
        out = []
        for p in profs:
            du, dv = p["dims"]
            eu = np.asarray(p["e_re"][0]) + 1j * np.asarray(p["e_im"][0])
            ev = np.asarray(p["e_re"][1]) + 1j * np.asarray(p["e_im"][1])
            mag = np.sqrt(np.abs(eu) ** 2 + np.abs(ev) ** 2).reshape(du, dv)
            un, vn = np.asarray(info["u_nodes"]), np.asarray(info["v_nodes"])
            arrays = {"|E|": mag, f"|E{AXES[u]}|": np.abs(eu).reshape(du, dv), f"|E{AXES[v]}|": np.abs(ev).reshape(du, dv)}
            maps = {}
            for k, arr in arrays.items():
                a2, e1, e2, _ = downsample(arr, un, vn, MAX_SIDE)
                maps[k] = {"data": b64(a2), "nu": int(a2.shape[0]), "nv": int(a2.shape[1]), "u_edges": (e1 / UM).round(6).tolist(), "v_edges": (e2 / UM).round(6).tolist(), "min": 0.0, "max": float(np.nanmax(a2)) if a2.size else 1.0}
            te = float(np.sum(np.abs(eu) ** 2) / max(np.sum(np.abs(eu) ** 2) + np.sum(np.abs(ev) ** 2), 1e-300))
            out.append({"neff": p.get("neff"), "neff_imag": p.get("neff_imag", 0.0), "freq": p.get("freq"), "te_fraction": round(te, 3), "maps": maps})
        seg = self.outline_segments(a, int(info["p"]), _um(spec["min"]), _um(spec["max"]))
        return {"name": name, "axis": a, "u": u, "v": v, "coord_um": round(info["coord"] / UM, 6), "profiles": out, "outline": seg}

    def dft_map(self, name: str, comp: str, fi: int, part: str = "abs", axis=None, index=None) -> dict:
        """One component of a frequency-domain field monitor at one frequency,
        on a plane of its box (µm), with the outline of the structures."""
        d = self.results["dft"][name]
        f = d["fields"][comp]
        arr = np.asarray(f["re"])[fi] + 1j * np.asarray(f["im"])[fi]  # [x, y, z]
        o = list(f["origin"])
        shape = arr.shape
        flat = [a for a in range(3) if shape[a] == 1]
        if axis is None:
            axis = flat[-1] if flat else 2
            if self.has_spec and self.invariant() is not None and shape[self.invariant()] == 1:
                axis = self.invariant()
        axis = int(axis)
        if index is None:
            index = shape[axis] // 2
        index = int(min(max(int(index), 0), shape[axis] - 1))
        sl = [slice(None)] * 3
        sl[axis] = index
        a2 = arr[tuple(sl)]
        u, v = plane_axes(axis)
        if part == "abs":
            vals = np.abs(a2)
        elif part == "re":
            vals = a2.real
        elif part == "im":
            vals = a2.imag
        elif part == "phase":
            vals = np.angle(a2)
        elif part == "intensity":
            vals = np.abs(a2) ** 2
        else:
            raise ValueError(part)
        out = {"name": name, "comp": comp, "freq": float(np.asarray(d["freqs"])[fi]), "part": part, "axis": axis, "index": index,
               "count": int(shape[axis]), "u": u, "v": v}
        if self.has_spec:
            gi = o[axis] + index
            nodes = self.nodes
            eu = nodes[u][o[u] : o[u] + shape[u] + 1]
            ev = nodes[v][o[v] : o[v] + shape[v] + 1]
            if len(eu) < shape[u] + 1:
                eu = np.append(eu, eu[-1] + (np.arange(1, shape[u] + 2 - len(eu))) * (eu[-1] - eu[-2] if len(eu) > 1 else 1e-9))
            if len(ev) < shape[v] + 1:
                ev = np.append(ev, ev[-1] + (np.arange(1, shape[v] + 2 - len(ev))) * (ev[-1] - ev[-2] if len(ev) > 1 else 1e-9))
            out["coord_um"] = round(nodes[axis][min(gi, len(nodes[axis]) - 1)] / UM, 6)
            gi = min(gi, self.n()[axis] - 1)
            out["outline"] = self.outline_segments(axis, gi, _bounds(u, v, eu, ev, 0), _bounds(u, v, eu, ev, 1))
        else:
            eu = np.arange(shape[u] + 1) * UM
            ev = np.arange(shape[v] + 1) * UM
            out["outline"] = []
            out["cells_only"] = True
        lo, hi = finite_range(vals)
        a3, e1, e2, stride = downsample(vals, eu, ev, MAX_SIDE)
        out.update(data=b64(a3), nu=int(a3.shape[0]), nv=int(a3.shape[1]), u_edges=(e1 / UM).round(6).tolist(), v_edges=(e2 / UM).round(6).tolist(),
                   min=lo, max=hi, stride=list(stride), signed=part in ("re", "im", "phase"))
        return out

    def time_data(self, name: str, comp: str) -> dict:
        t = self.results["time"][name]
        times = np.asarray(t["times"], dtype=float)
        a = np.asarray(t["fields"][comp], dtype=float)  # [sample, x, y, z]
        idx = tuple(s // 2 for s in a.shape[1:])
        trace = a[(slice(None),) + idx]
        # spectrum of the trace: |FFT| zero-padded to 8x
        out = {"name": name, "comp": comp, "times": times.tolist(), "values": trace.tolist(), "point": list(idx), "box": list(a.shape[1:])}
        if len(times) > 3:
            dt = float(np.median(np.diff(times)))
            nfft = 1 << int(math.ceil(math.log2(len(trace) * 8)))
            sp = np.abs(np.fft.rfft(trace, nfft)) * dt
            fr = np.fft.rfftfreq(nfft, dt)
            # keep the band where the spectrum is above 1e-6 of its peak
            keep = sp > 1e-6 * sp.max() if sp.max() > 0 else np.ones_like(sp, bool)
            if keep.any():
                i1 = min(len(fr), int(np.nonzero(keep)[0].max() * 1.3) + 2)
            else:
                i1 = len(fr)
            k = max(1, i1 // 2000)
            out["spec_freqs"] = fr[:i1:k].tolist()
            out["spec"] = sp[:i1:k].tolist()
        return out

    def energy_data(self, name: str) -> dict:
        e = self.results["energy"][name]
        return {"name": name, "times": np.asarray(e["times"]).tolist(), "energy": np.asarray(e["energy"]).tolist()}

    def far_data(self, name: str, fi: int) -> dict:
        f = self.results["far_field"][name]
        I = np.asarray(f["intensity"], dtype=float)[fi]  # [theta, phi]
        th = np.degrees(np.asarray(f["theta"], dtype=float))
        ph = np.degrees(np.asarray(f["phi"], dtype=float))

        def edges(c):
            if len(c) == 1:
                return np.array([c[0] - 0.5, c[0] + 0.5])
            m = 0.5 * (c[1:] + c[:-1])
            return np.concatenate([[c[0] - (m[0] - c[0])], m, [c[-1] + (c[-1] - m[-1])]])

        # picture: phi horizontal, theta vertical
        a = I.T
        return {"name": name, "freq": float(np.asarray(f["freqs"])[fi]), "freqs": np.asarray(f["freqs"]).tolist(), "theta": th.tolist(), "phi": ph.tolist(),
                "data": b64(a), "nu": int(a.shape[0]), "nv": int(a.shape[1]), "u_edges": edges(ph).tolist(), "v_edges": edges(th).tolist(),
                "min": 0.0, "max": float(np.nanmax(I)) if I.size else 1.0, "flux": np.asarray(f["flux"]).tolist(), "intensity": I.tolist()}


def _bounds(u, v, eu, ev, which):
    out = [0.0, 0.0, 0.0]
    out[u] = float((eu[0] if which == 0 else eu[-1]) / UM)
    out[v] = float((ev[0] if which == 0 else ev[-1]) / UM)
    return out


def devices() -> list:
    """Devices the build reports, as choices for a run (``cuda:0,1`` when
    there are two GPUs)."""
    out = [{"id": "cpu", "label": "CPU"}]
    gpus = []
    for d in _core.devices():
        if d.startswith("cuda:"):
            idx, _, name = d.partition(" ")
            gpus.append(idx)
            out.append({"id": idx, "label": f"{idx}  {name}".strip()})
    if len(gpus) > 1:
        out.append({"id": "cuda:" + ",".join(g.split(":")[1] for g in gpus), "label": f"cuda:{','.join(g.split(':')[1] for g in gpus)}  all {len(gpus)} GPUs"})
    return out


# ---------------------------------------------------------------------------
# runs started from the interface
# ---------------------------------------------------------------------------


class RunState:
    """A run in a background thread with the engine's run hook: progress,
    the field-decay history, a live field slice, stop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.reset()

    def reset(self):
        self.status = "idle"  # idle, compiling, running, done, stopped, error
        self.device = None
        self.step = 0
        self.nsteps = 0
        self.dt = 0.0
        self.elapsed = 0.0
        self.decay = None
        self.decay_hist: list = []
        self.shutoff = None
        self.error = None
        self.stop_flag = False
        self.slice = None
        self.slice_version = 0
        self.want = None
        self.started = None
        self.finished = None
        self.wall = 0.0
        self.version = 0

    def snapshot(self) -> dict:
        with self.lock:
            rem = None
            if self.status == "running" and self.step > 0 and self.nsteps > self.step:
                rem = self.elapsed / self.step * (self.nsteps - self.step)
            return {"status": self.status, "device": self.device, "step": self.step, "nsteps": self.nsteps, "dt": self.dt,
                    "time": self.step * self.dt, "elapsed": self.elapsed, "remaining_max": rem, "decay": self.decay,
                    "shutoff": self.shutoff, "decay_hist": self.decay_hist[-400:], "error": self.error,
                    "slice_version": self.slice_version, "want": self.want, "wall": (time.time() - self.started) if self.started and self.status in ("compiling", "running") else self.wall,
                    "version": self.version}

    def running(self) -> bool:
        return self.status in ("compiling", "running")

    def start(self, session: Session, device: str, comp: str, axis: int, index: int, every: float = 0.3) -> dict:
        with self.lock:
            if self.status in ("compiling", "running"):
                raise RuntimeError("a run is in progress")
        allowed = {d["id"] for d in devices()}
        if device not in allowed:
            raise ValueError(f"device {device!r} is not one of {sorted(allowed)}")
        if comp not in COMPONENTS:
            raise ValueError(f"unknown component {comp!r}")
        self.reset()
        with self.lock:
            self.status = "compiling"
            self.device = device
            self.want = [comp, int(axis), int(index)]
            self.started = time.time()
            self.nsteps = int(session.nsteps)
            self.dt = float(session.dt)
            self.shutoff = session.spec.get("shutoff")
        nodes = session.nodes

        def hook(d):
            with self.lock:
                self.status = "running"
                self.step, self.nsteps, self.dt, self.elapsed = d["step"], d["nsteps"], d["dt"], d["elapsed"]
                self.decay, self.shutoff = d["decay"], d["shutoff"]
                if d["decay"] is not None and (not self.decay_hist or self.decay_hist[-1][0] != d["step"]):
                    self.decay_hist.append([d["step"], d["decay"]])
                s = d.get("slice")
                if s is not None:
                    a = np.asarray(s["data"], dtype=float)
                    u, v = plane_axes(s["axis"])
                    a2, eu, ev, st = downsample(a, nodes[u], nodes[v], LIVE_SIDE)
                    lim = float(np.nanmax(np.abs(a2))) if a2.size else 0.0
                    self.slice = {"comp": s["comp"], "axis": s["axis"], "index": s["index"], "u": u, "v": v, "nu": int(a2.shape[0]), "nv": int(a2.shape[1]),
                                  "u_edges": (eu / UM).round(6).tolist(), "v_edges": (ev / UM).round(6).tolist(), "data": b64(a2), "min": -lim, "max": lim,
                                  "step": d["step"], "time": d["step"] * d["dt"], "stride": list(st)}
                    self.slice_version += 1
                self.version += 1
                want = tuple(self.want) if self.want else None
                return {"stop": self.stop_flag, "slice": want}

        def work():
            try:
                raw = _core.run(session.spec_json, device, 0, hook, every)
                with self.lock:
                    stopped = self.stop_flag
                    self.step = int(raw["stats"]["steps"])
                    self.elapsed = float(raw["stats"]["seconds"])
                    e = np.asarray(raw["stats"].get("energy", []), dtype=float)
                    st = np.asarray(raw["stats"].get("energy_steps", []))
                    if e.size and e.max() > 0:
                        self.decay = float(e[-1] / e.max())
                        self.decay_hist = [[int(a), float(b / e.max())] for a, b in zip(st, e)]
                    self.status = "stopped" if stopped else "done"
                session.results = raw
            except Exception as exc:  # reported to the browser
                with self.lock:
                    self.status = "error"
                    self.error = str(exc)
            finally:
                with self.lock:
                    self.finished = time.time()
                    self.wall = self.finished - (self.started or self.finished)
                    self.version += 1

        self.thread = threading.Thread(target=work, name="lightsim-gui-run", daemon=True)
        self.thread.start()
        return self.snapshot()

    def stop(self) -> dict:
        with self.lock:
            self.stop_flag = True
        return self.snapshot()

    def set_plane(self, comp: str, axis: int, index: int) -> dict:
        if comp not in COMPONENTS:
            raise ValueError(f"unknown component {comp!r}")
        with self.lock:
            self.want = [comp, int(axis), int(index)]
        return self.snapshot()

    def get_slice(self):
        with self.lock:
            return self.slice

    def join(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)
