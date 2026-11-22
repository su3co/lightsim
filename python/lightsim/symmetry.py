"""Automatic mirror-symmetry reduction (cf. FDTDX ``config.symmetry``,
Lumerical symmetric/anti-symmetric boundaries).

``run_symmetric(sim, (sx, sy, sz))`` runs only the upper half of every axis
with ``s != 0`` and unfolds the results back to the full domain:

* ``s = +1``: tangential E is even about the mid-plane (PMC wall),
* ``s = -1``: tangential E is odd (PEC wall).

The domain, mesh, structures and sources must be mirror-symmetric with a grid
node on each mid-plane. The reduced run uses exactly the full domain's upper-half
nodes (explicit mesh), so its fields equal the full run's.

After the run:

* DFT field monitors are mirrored with the parity of each Yee component
  (E_t: s, E_n: -s, H_t: -s, H_n: s) and cover the symmetric extent of the
  recorded region;
* flux planes cut by a mid-plane and closed flux boxes are scaled by 2 per
  reduced axis.
"""

from __future__ import annotations

import copy

import numpy as np

from . import _core
from .spec import Results

_REAL = np.float64 if _core.precision == "f64" else np.float32

_AX = "xyz"


def _region_nodes(sim):
    nodes = []
    for a, g in enumerate(sim.grid()):
        g = np.asarray(g)
        lo, hi = sim.min[a], sim.max[a]
        tol = 1e-9 * max(hi - lo, 1e-30)
        nodes.append(g[(g >= lo - tol) & (g <= hi + tol)])
    return nodes


def reduce(sim, symmetry):
    """Return (reduced simulation, info) for mirror symmetry ``(sx, sy, sz)``."""
    sym = tuple(int(s) for s in symmetry)
    if any(s not in (-1, 0, 1) for s in sym):
        raise ValueError("symmetry entries must be -1, 0 or +1")
    nodes = _region_nodes(sim)
    red = copy.deepcopy(sim)
    centers = [None, None, None]
    for a, s in enumerate(sym):
        if s == 0:
            continue
        lo, hi = sim.min[a], sim.max[a]
        c = 0.5 * (lo + hi)
        n = nodes[a]
        tol = 1e-6 * float(np.min(np.diff(n)))
        if np.min(np.abs(n - c)) > tol:
            raise ValueError(f"symmetry along {_AX[a]}: no grid node on the mid-plane {c}")
        if not np.allclose(n - c, -(n - c)[::-1], atol=tol):
            raise ValueError(f"symmetry along {_AX[a]}: the mesh is not mirror-symmetric")
        if sim.boundaries[_AX[a]][0] != sim.boundaries[_AX[a]][1]:
            raise ValueError(f"symmetry along {_AX[a]}: the two boundaries differ")
        centers[a] = c
        nodes[a] = n[n >= c - tol]
        nodes[a][0] = c
        red.min = list(red.min)
        red.min[a] = c
        red.boundaries = dict(red.boundaries)
        red.boundaries[_AX[a]] = [{"type": "pmc" if s > 0 else "pec"}, red.boundaries[_AX[a]][1]]
    red.mesh = {"type": "explicit", "x": [float(v) for v in nodes[0]], "y": [float(v) for v in nodes[1]], "z": [float(v) for v in nodes[2]]}
    # drop sources entirely in a removed half; clip boxes at the mid-planes
    keep = []
    for src in red.sources:
        drop = False
        for a, c in enumerate(centers):
            if c is None:
                continue
            if "position" in src and isinstance(src["position"], list) and src["position"][a] < c - 1e-15:
                drop = True
            if src.get("type") == "tfsf":
                src["min"] = list(src["min"])
                src["min"][a] = max(src["min"][a], c)
        if not drop:
            keep.append(src)
    red.sources = keep
    # zero-thickness sheets: drop those normal to a symmetry axis in the removed half
    red.sheets = [sh for sh in getattr(red, "sheets", []) if not any(c is not None and sh["axis"] == a and sh["position"] < c - 1e-15 for a, c in enumerate(centers))]
    for m in red.monitors:
        for a, c in enumerate(centers):
            if c is None or "min" not in m:
                continue
            if m["type"] == "flux" and m["axis"] == a:
                if m["position"] < c:
                    raise ValueError("flux planes normal to a symmetry axis must lie in the upper half")
                continue
            lo, hi = m["min"][a], m["max"][a]
            ext = max(abs(lo - c), abs(hi - c))
            m["min"] = list(m["min"])
            m["max"] = list(m["max"])
            m["min"][a] = c
            m["max"][a] = c + ext
    info = {"symmetry": sym, "centers": centers}
    return red, info


def _node_component(comp: str, axis: int) -> bool:
    """Is component `comp` located on grid nodes along `axis`?"""
    kind, ca = comp[0], _AX.index(comp[1].lower())
    return (ca != axis) if kind == "E" else (ca == axis)


def _parity(comp: str, axis: int, s: int) -> int:
    kind, ca = comp[0], _AX.index(comp[1].lower())
    tangential = ca != axis
    if kind == "E":
        return s if tangential else -s
    return -s if tangential else s


def _unfold_field(comp: str, f: dict, axis: int, s: int, p0: int) -> dict:
    """Mirror one DFT component (origin in full-grid indices) about node p0."""
    org = list(f["origin"])
    data = np.asarray(f["re"]) + 1j * np.asarray(f["im"])  # (nf, nx, ny, nz)
    node = _node_component(comp, axis)
    par = _parity(comp, axis, s)
    lo, n = org[axis], data.shape[axis + 1]
    hi = lo + n - 1

    def mirror(i):
        return 2 * p0 - i if node else 2 * p0 - 1 - i

    new_lo = min(lo, mirror(hi))
    new_hi = max(hi, mirror(lo))
    shape = list(data.shape)
    shape[axis + 1] = new_hi - new_lo + 1
    out = np.zeros(shape, dtype=complex)
    for i in range(lo, hi + 1):
        src = [slice(None)] * 4
        dst = [slice(None)] * 4
        src[axis + 1] = i - lo
        dst[axis + 1] = i - new_lo
        out[tuple(dst)] = data[tuple(src)]
        j = mirror(i)
        if j != i:
            dst[axis + 1] = j - new_lo
            out[tuple(dst)] = par * data[tuple(src)]
    org[axis] = new_lo
    return {"origin": org, "re": out.real.astype(_REAL), "im": out.imag.astype(_REAL)}


def run_symmetric(sim, symmetry, device: str = "auto", progress: int = 0) -> Results:
    """Run the symmetry-reduced simulation and unfold its results."""
    red, info = reduce(sim, symmetry)
    res = red.run(device=device, progress=progress)
    raw = res.raw
    full_grid = [np.asarray(g) for g in sim.grid()]
    red_grid = [np.asarray(g) for g in red.grid()]
    for a, s in enumerate(info["symmetry"]):
        if s == 0:
            continue
        c = info["centers"][a]
        p0 = int(np.argmin(np.abs(full_grid[a] - c)))
        r0 = int(np.argmin(np.abs(red_grid[a] - c)))
        off = p0 - r0
        for m in sim.monitors:
            if m["type"] != "dft_field" or m["name"] not in raw.get("dft", {}):
                continue
            fields = raw["dft"][m["name"]]["fields"]
            for comp in list(fields.keys()):
                f = dict(fields[comp])
                f["origin"] = list(f["origin"])
                f["origin"][a] += off
                fields[comp] = _unfold_field(comp, f, a, s, p0)
        # flux planes cut by this mid-plane carry half the power
        for m in sim.monitors:
            if m["type"] == "flux" and m["axis"] != a and m["name"] in raw.get("flux", {}):
                raw["flux"][m["name"]] = [2.0 * v for v in raw["flux"][m["name"]]]
    # closed flux boxes: the face on a symmetry wall is skipped by the core
    # (no power crosses a PEC/PMC wall); the rest is 1/2 per reduced axis
    nsym = sum(1 for s in info["symmetry"] if s != 0)
    for m in sim.monitors:
        if m["type"] == "flux_box" and m["name"] in raw.get("flux", {}):
            raw["flux"][m["name"]] = [v * 2.0**nsym for v in raw["flux"][m["name"]]]
    # indices along non-reduced axes may differ only by PML offsets, which are equal
    raw["symmetry"] = list(info["symmetry"])
    return Results(raw)
