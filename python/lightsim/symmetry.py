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
  reduced axis;
* a mode monitor cut by a mid-plane projects on the upper half of its
  profiles; an amplitude is the overlap over the profile's own, so it needs
  no factor, and the flux through the window is scaled like a flux plane.

Mode sources and mode monitors: a window cut by a mid-plane must be symmetric
about it. The run then uses the upper half of the profile, which must have
the parity of the wall (a source of the other parity, or of none, is
refused; a monitor mode of the other parity reads zero, as in the full run).
A window that lies entirely in the upper half is used as it is. Flux planes
and windows of mode sources and monitors that are neither are refused.
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


_PORT_FIELDS = (("e_re", "e_im", "E"), ("h_re", "h_im", "H"))


def _window(item, a, c, tol, what):
    """How the window of a mode source, mode monitor or flux plane lies about
    the mid-plane ``c`` along axis ``a``: ``"above"`` (entirely in the upper
    half) or ``"cut"`` (symmetric about the plane)."""
    lo, hi = item["min"][a], item["max"][a]
    if lo >= c - tol:
        return "above"
    if abs((c - lo) - (hi - c)) <= tol:
        return "cut"
    raise ValueError(f"symmetry along {_AX[a]}: the window of {what} is neither symmetric about the mid-plane nor entirely in the upper half")


def _half_profile(prof, port_axis, a, m, s, what):
    """The upper half of a mode profile about the mid-plane normal to ``a``,
    ``m`` cells above the low edge of its window, and its parity: +1 when the
    profile has the symmetry ``s`` of the wall, -1 when it has the other one.
    A profile with neither is refused. The half returned is the mean of the
    upper half and the mirror image of the lower one with the profile's own
    parity (a solved profile is symmetric to a few per cent of its tails:
    the solver's window is not centred on the plane to the half cell)."""
    t_axes = [x for x in range(3) if x != port_axis]  # the profile's two axes, in this order
    k = t_axes.index(a)
    dims = [int(d) for d in prof["dims"]]
    if dims[k] < 2 * m or m < 1:
        raise ValueError(f"symmetry along {_AX[a]}: the profile of {what} does not cover its window")
    out = {key: v for key, v in prof.items() if key not in ("e_re", "e_im", "h_re", "h_im")}
    even = odd = norm = 0.0
    parts = []
    for re_key, im_key, kind in _PORT_FIELDS:
        for t, q in enumerate(t_axes):
            # E along a and H across a sit half a cell off the plane; the others on its nodes
            on_nodes = (q != a) if kind == "E" else (q == a)
            par = (s if q != a else -s) if kind == "E" else (-s if q != a else s)
            halves = []
            for key in (re_key, im_key):
                flat = np.asarray(prof.get(key, [[], []])[t], dtype=float)
                if flat.size == 0:
                    halves.append(None)
                    continue
                x = np.moveaxis(flat.reshape(dims), k, 0)
                up = x[m:]
                low = x[m::-1] if on_nodes else x[m - 1 :: -1]  # mirror images of up[0], up[1], ...
                n = min(len(up), len(low))
                even += float(np.sum((up[:n] - par * low[:n]) ** 2))
                odd += float(np.sum((up[:n] + par * low[:n]) ** 2))
                norm += float(np.sum(up[:n] ** 2) + np.sum(low[:n] ** 2))
                halves.append((up, low[:n], par))
            parts.append((re_key, im_key, halves))
    # (a weakly guided mode in a small window is symmetric to ten per cent only: the part with the parity found
    # is kept, which is at least four fifths of it)
    if norm == 0.0 or 4.0 * even <= odd:
        parity = 1
    elif 4.0 * odd <= even:
        parity = -1
    else:
        raise ValueError(f"symmetry along {_AX[a]}: the profile of {what} is neither even nor odd about the mid-plane")
    for re_key, im_key, halves in parts:
        for key, h in zip((re_key, im_key), halves):
            if h is None:
                out.setdefault(key, []).append([])
                continue
            up, low, par = h
            mean = up.copy()
            mean[: len(low)] = 0.5 * (up[: len(low)] + parity * par * low)
            out.setdefault(key, []).append(np.moveaxis(mean, 0, k).ravel().tolist())
    out["dims"] = [d - m if i == k else d for i, d in enumerate(dims)]
    if prof.get("others"):
        out["others"] = [_half_profile(o, port_axis, a, m, s, what)[0] for o in prof["others"]]
    return out, parity


def reduce(sim, symmetry):
    """Return (reduced simulation, info) for mirror symmetry ``(sx, sy, sz)``."""
    sym = tuple(int(s) for s in symmetry)
    if any(s not in (-1, 0, 1) for s in sym):
        raise ValueError("symmetry entries must be -1, 0 or +1")
    nodes = _region_nodes(sim)
    red = copy.deepcopy(sim)
    full_grid = [np.asarray(g) for g in sim.grid()]
    centers = [None, None, None]
    # per mode monitor: the factor on each mode's amplitudes (1; 0 for a mode of the other parity, which the
    # fields of this run cannot hold) and on the flux through its window; per flux plane: the factor on its flux
    mode_scale, flux_scale = {}, {}
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
            if src.get("type") == "mode" and src["axis"] != a:
                # a window cut by the mid-plane: the upper half of the profile, which must have the wall's parity
                what = f"mode source '{src.get('name', '')}'"
                tol = 1e-6 * float(np.min(np.diff(full_grid[a])))
                if _window(src, a, c, tol, what) == "cut":
                    m = int(np.argmin(np.abs(full_grid[a] - c))) - int(np.searchsorted(full_grid[a], src["min"][a] + tol, side="right") - 1)
                    src["profile"], parity = _half_profile(src["profile"], src["axis"], a, m, sym[a], what)
                    if parity < 0:
                        raise ValueError(f"symmetry along {_AX[a]}: the profile of {what} has the other parity (use {-sym[a]:+d} for this axis)")
                    src["min"] = list(src["min"])
                    src["min"][a] = c
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
            tol = 1e-6 * float(np.min(np.diff(full_grid[a])))
            if m["type"] == "flux":
                # cut by the mid-plane: half the plane, twice the flux; entirely above: as it is
                if _window(m, a, c, tol, f"flux plane '{m['name']}'") == "cut":
                    m["min"] = list(m["min"])
                    m["min"][a] = c
                    flux_scale[m["name"]] = 2.0 * flux_scale.get(m["name"], 1.0)
                continue
            if m["type"] == "mode":
                if m["axis"] == a:
                    if m["position"] < c:
                        raise ValueError("mode monitors normal to a symmetry axis must lie in the upper half")
                    continue
                what = f"mode monitor '{m['name']}'"
                scale = mode_scale.setdefault(m["name"], [1.0] * len(m["modes"]))
                if _window(m, a, c, tol, what) == "cut":
                    cells = int(np.argmin(np.abs(full_grid[a] - c))) - int(np.searchsorted(full_grid[a], m["min"][a] + tol, side="right") - 1)
                    halves = [_half_profile(q, m["axis"], a, cells, sym[a], what) for q in m["modes"]]
                    m["modes"] = [h[0] for h in halves]
                    for i, (_, parity) in enumerate(halves):
                        scale[i] *= 1.0 if parity > 0 else 0.0
                    for key in (m["name"], "mode:" + m["name"]):  # the plane's flux, kept under both
                        flux_scale[key] = 2.0 * flux_scale.get(key, 1.0)
                    m["min"] = list(m["min"])
                    m["min"][a] = c
                continue
            lo, hi = m["min"][a], m["max"][a]
            ext = max(abs(lo - c), abs(hi - c))
            m["min"] = list(m["min"])
            m["max"] = list(m["max"])
            m["min"][a] = c
            m["max"][a] = c + ext
    info = {"symmetry": sym, "centers": centers, "mode_scale": mode_scale, "flux_scale": flux_scale}
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
    # flux planes cut by a mid-plane carry half the power per plane that cuts them
    for name, k in info["flux_scale"].items():
        if not name.startswith("mode:") and name in raw.get("flux", {}):
            raw["flux"][name] = k * np.asarray(raw["flux"][name])
    # mode monitors cut by a mid-plane: a mode of the other parity is not in the fields at all (its half profile
    # would still overlap them); the flux through the window is twice that of its half per plane
    for name, scales in info["mode_scale"].items():
        res_m = raw.get("modes", {}).get(name)
        if res_m is None:
            continue
        per_frequency = len(res_m["plus"]) == 1 and len(scales) > 1
        for i in range(len(res_m["plus"])):
            k = (min(scales) if min(scales) == max(scales) else None) if per_frequency else scales[i]
            if k is None:
                raise ValueError(f"mode monitor '{name}': its profiles per frequency do not share one parity about the mid-plane")
            for side in ("plus", "minus"):
                res_m[side][i] = {"re": k * np.asarray(res_m[side][i]["re"]), "im": k * np.asarray(res_m[side][i]["im"])}
        if "flux" in res_m and "mode:" + name in info["flux_scale"]:
            res_m["flux"] = info["flux_scale"]["mode:" + name] * np.asarray(res_m["flux"])
    # closed flux boxes: the face on a symmetry wall is skipped by the core
    # (no power crosses a PEC/PMC wall); the rest is 1/2 per reduced axis
    nsym = sum(1 for s in info["symmetry"] if s != 0)
    for m in sim.monitors:
        if m["type"] == "flux_box" and m["name"] in raw.get("flux", {}):
            raw["flux"][m["name"]] = [v * 2.0**nsym for v in raw["flux"][m["name"]]]
    # indices along non-reduced axes may differ only by PML offsets, which are equal
    raw["symmetry"] = list(info["symmetry"])
    return Results(raw)
