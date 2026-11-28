"""S-parameters of gdsfactory components with lightsim.

Workflow (mirrors gplugins' `write_sparameters_*`):

1. extend the component's ports so waveguides run into the PML,
2. extrude the layer stack into lightsim structures,
3. solve the fundamental (or chosen) waveguide mode at every port on the
   lightsim Yee grid,
4. for each input port run one simulation with a mode source and mode
   monitors on all ports,
5. S[j, i](f) = a_out_j(f) / a_in_i(f).
"""

from __future__ import annotations

import numpy as np

from . import gds as _gds
from .modes import profile_over_band, solve_modes
from .spec import C0, AutoMesh, GaussianPulse, ModeMonitor, ModeSource, PML, Simulation, um


def _port_plane(port, offset, zmin, zmax, half_width):
    """(axis, position, direction_into_device, min, max) for a port.

    The plane sits `offset` *outside* the device on the (extended) port
    waveguide, where the guide is straight and isolated.
    """
    x, y = port["center"]
    o = port["orientation"]
    hw = half_width
    if abs(o - 0) < 1 or abs(o - 360) < 1:
        return 0, x + offset, "-", [x + offset, y - hw, zmin], [x + offset, y + hw, zmax]
    if abs(o - 180) < 1:
        return 0, x - offset, "+", [x - offset, y - hw, zmin], [x - offset, y + hw, zmax]
    if abs(o - 90) < 1:
        return 1, y + offset, "-", [x - hw, y + offset, zmin], [x + hw, y + offset, zmax]
    if abs(o - 270) < 1:
        return 1, y - offset, "+", [x - hw, y - offset, zmin], [x + hw, y - offset, zmax]
    raise ValueError(f"port {port['name']} has non-Manhattan orientation {o}")


def _half_width(port, ports, lateral):
    """Port window half-width: width/2 + lateral margin, clamped to half the
    distance to the nearest parallel port on the same face so that windows
    never contain a neighbouring waveguide."""
    hw = port["width"] / 2 + lateral
    x, y = port["center"]
    horiz = abs(port["orientation"] % 180) < 1  # faces +-x: transverse axis is y
    for q in ports:
        if q is port or abs(q["orientation"] - port["orientation"]) > 1:
            continue
        qx, qy = q["center"]
        same_face = abs(qx - x) < 1e-9 if horiz else abs(qy - y) < 1e-9
        if not same_face:
            continue
        d = abs(qy - y) if horiz else abs(qx - x)
        if d > 0:
            hw = min(hw, d / 2)
    return hw


def detect_port_symmetries(component, ports=None) -> list[dict]:
    """Mirror symmetries of a component that map its ports onto each other.

    The layout is compared with its mirror image about the horizontal and
    the vertical centre line of its bounding box, layer by layer and exactly
    (integer database units). For each mirror that leaves the layout
    unchanged and maps every optical port onto a port of the same width
    facing the mirrored way, the port map ``{name: image}`` is returned.
    Maps that send every port to itself are left out. The result is the
    list of generators :func:`write_sparameters` accepts as
    ``port_symmetry``; it is empty for a component without such a symmetry.
    """
    import klayout.db as kdb

    ports = _gds.port_info(component) if ports is None else ports
    bb = component.dbbox()
    dbu = component.kcl.dbu
    # twice the centre of the bounding box in integer database units
    sx, sy = int(round((bb.left + bb.right) / dbu)), int(round((bb.bottom + bb.top) / dbu))
    # mirror about y = centre: M0 flips y, then shift by bottom + top; about x = centre likewise with M90
    mirrors = {"y": kdb.Trans(kdb.Trans.M0, 0, sy), "x": kdb.Trans(kdb.Trans.M90, sx, 0)}
    regions = {}
    for layer, polys in component.get_polygons().items():
        r = kdb.Region()
        for q in polys:
            r.insert(q)
        regions[layer] = r.merged()
    out = []
    xc, yc = 0.5 * (bb.left + bb.right) * um, 0.5 * (bb.bottom + bb.top) * um
    tol = 2 * dbu * um
    for axis, trans in mirrors.items():
        if any(not (r ^ r.transformed(trans)).is_empty() for r in regions.values()):
            continue
        mapping = {}
        for a in ports:
            x, y = a["center"]
            if axis == "y":
                image, facing = (x, 2 * yc - y), (-a["orientation"]) % 360.0
            else:
                image, facing = (2 * xc - x, y), (180.0 - a["orientation"]) % 360.0
            match = [
                b["name"]
                for b in ports
                if abs(b["center"][0] - image[0]) < tol and abs(b["center"][1] - image[1]) < tol and abs((b["orientation"] - facing + 180.0) % 360.0 - 180.0) < 1.0 and abs(b["width"] - a["width"]) < tol
            ]
            if len(match) != 1:
                mapping = None
                break
            mapping[a["name"]] = match[0]
        if mapping and any(k != v for k, v in mapping.items()):
            out.append(mapping)
    return out


def _symmetry_plan(names, port_symmetry, inputs):
    """Which ports to excite, and how the others follow from them.

    ``port_symmetry`` is a list of port maps. Each swaps ports in pairs:
    ``{"o2": "o3"}`` also maps o3 to o2, and ports it does not mention map to
    themselves. Returns ``(run, source_of)``:
    ``run`` lists the ports to excite and ``source_of[p] = (r, inverse)``
    gives, for an input ``p`` that is not excited, the excited port ``r``
    and the permutation with ``S[j, p] = S[inverse[j], r]``.
    """
    gens = []
    for g in port_symmetry:
        full = {n: n for n in names}
        for a, b in g.items():
            if a not in full or b not in full:
                raise ValueError(f"port_symmetry names port {a!r} -> {b!r}, but the ports are {names}")
            full[a] = b
            if b not in g:  # {"o2": "o3"} also says o3 -> o2
                full[b] = a
        if sorted(full.values()) != sorted(names) or any(full[full[n]] != n for n in names):
            raise ValueError(f"port_symmetry {g} is not a pairing of the ports (each map must swap ports in pairs)")
        gens.append(full)
    run, source_of, reached = [], {}, {}
    for p in inputs:
        if p in reached:
            source_of[p] = reached[p]
            continue
        run.append(p)
        # all images of p under the group generated by the maps: breadth first over permutations
        ident = {n: n for n in names}
        frontier, seen = [ident], {p: ident}
        while frontier:
            nxt = []
            for perm in frontier:
                for g in gens:
                    comp = {n: g[perm[n]] for n in names}  # apply perm, then g
                    if comp[p] not in seen:
                        seen[comp[p]] = comp
                        nxt.append(comp)
            frontier = nxt
        for q, perm in seen.items():
            if q != p and q not in reached:
                reached[q] = (p, {v: k for k, v in perm.items()})
    return run, source_of


def _resolve_port_symmetry(component, ports, port_symmetry):
    if port_symmetry is None:
        return []
    if isinstance(port_symmetry, str):
        if port_symmetry != "auto":
            raise ValueError("port_symmetry must be None, 'auto', a port map or a list of port maps")
        return detect_port_symmetries(component, ports)
    return [port_symmetry] if isinstance(port_symmetry, dict) else list(port_symmetry)


def build_simulation(
    component,
    wavelengths,
    layer_stack=None,
    material_index=None,
    steps_per_wavelength: float = 20.0,
    xy_margin: float = 1.0 * um,
    z_margin: float = 0.8 * um,
    port_extension: float = 2.0 * um,
    background: str = "sio2",
    run_time_factor: float = 6.0,
    zmin: float | None = None,
    zmax: float | None = None,
    shutoff: float = 1e-5,
    exclude=None,
    wafer_levels=None,
):
    """Build the base simulation (no sources/monitors) and port geometry."""
    import gdsfactory as gf

    ports = _gds.port_info(component)
    ext = gf.components.extend_ports(component, length=(port_extension + xy_margin) / um)
    bb = component.dbbox()
    rmin = [bb.left * um - xy_margin, bb.bottom * um - xy_margin, 0.0]
    rmax = [bb.right * um + xy_margin, bb.top * um + xy_margin, 0.0]
    structures, media, layers = _gds.component_to_structures(ext, layer_stack, material_index, rmin, rmax, exclude=exclude, wafer_levels=wafer_levels)
    pat = [l for l in layers if l.polygons]
    core_lo = min(l.zmin for l in pat) if pat else 0.0
    core_hi = max(l.zmax for l in pat) if pat else 0.22 * um
    rmin[2] = zmin if zmin is not None else core_lo - z_margin
    rmax[2] = zmax if zmax is not None else core_hi + z_margin
    wl = np.asarray(wavelengths, dtype=float)
    lam0 = float(np.mean(wl))
    idx = _gds.material_table(material_index)
    if not any(m["name"] == background for m in media):
        bg = _gds._medium_for(background, idx, "background")  # None: a medium the core defines
        if bg is not None:
            media = media + [bg]
    nmax = max([m["eps"][0] for m in media] + [1.0]) ** 0.5
    length = max(rmax[0] - rmin[0], rmax[1] - rmin[1])
    run_time = run_time_factor * nmax * length / C0 + 2e-12 * 0
    sim = Simulation(
        min=rmin,
        max=rmax,
        mesh=AutoMesh(lam0, steps_per_wavelength=steps_per_wavelength, grading=1.3),
        run_time=run_time,
        boundaries={"x": [PML(), PML()], "y": [PML(), PML()], "z": [PML(), PML()]},
        background=background,
        shutoff=shutoff,
    )
    for m in media:
        sim.add_medium(m)
    sim.structures.extend(structures)
    return sim, ports, (core_lo, core_hi)


def write_sparameters(
    component,
    wavelengths=None,
    layer_stack=None,
    material_index=None,
    steps_per_wavelength: float = 20.0,
    port_offset: float = 0.5 * um,
    port_lateral_margin: float = 1.0 * um,
    port_z_margin: float = 0.6 * um,
    mode_index: int = 0,
    num_modes: int = 2,
    device: str = "auto",
    input_ports=None,
    progress: int = 0,
    return_results: bool = False,
    port_symmetry=None,
    port_profiles: str = "center",
    **kwargs,
):
    """Compute the S-matrix of a gdsfactory component.

    Returns a dict with ``wavelengths`` (m) and complex arrays ``"o2@0,o1@0"``
    = S(o2 <- o1), following the gdsfactory/gplugins naming convention, plus
    ``"mode_neff"`` per port.

    One simulation is run per input port. ``input_ports`` limits the inputs
    to the ports named. ``port_symmetry`` uses a symmetry of the device to
    run fewer simulations for the same result: ``"auto"`` looks for mirror
    symmetries of the layout (:func:`detect_port_symmetries`), a dict such as
    ``{"o2": "o3"}`` states that a symmetry of the device swaps those ports,
    and a list of dicts gives several symmetries. Ports that are images of
    each other are then excited once, and the columns of the others are
    copied: ``S[j, image(i)] = S[image(j), i]``. With a mirror-symmetric mesh,
    which the automatic mesh is for a mirror-symmetric layout, the copied
    entries are what their own runs would give. ``None`` (the default) runs
    every input.
    """
    if wavelengths is None:
        wavelengths = np.linspace(1.5e-6, 1.6e-6, 21)
    wl = np.asarray(wavelengths, dtype=float)
    freqs = C0 / wl
    f0 = float(np.mean(freqs))
    fw = float(np.ptp(freqs)) if len(freqs) > 1 else 0.2 * f0
    sim, ports, (zc0, zc1) = build_simulation(component, wl, layer_stack, material_index, steps_per_wavelength, **kwargs)
    pz0 = zc0 - port_z_margin
    pz1 = zc1 + port_z_margin
    planes = {}
    profiles = {}
    neffs = {}
    for p in ports:
        axis, pos, direc, pmin, pmax = _port_plane(p, port_offset, pz0, pz1, _half_width(p, ports, port_lateral_margin))
        modes = solve_modes(sim, axis, pos, pmin, pmax, f0, num_modes=max(num_modes, mode_index + 1))
        modes = sorted(modes, key=lambda m: -m.neff.real)
        planes[p["name"]] = (axis, pos, direc, pmin, pmax)
        profiles[p["name"]] = modes[mode_index].profile
        if port_profiles == "band":
            profiles[p["name"]] = profile_over_band(sim, axis, pos, pmin, pmax, profiles[p["name"]], freqs)
        elif port_profiles != "center":
            raise ValueError("port_profiles must be 'center' or 'band'")
        neffs[p["name"]] = modes[mode_index].neff
    signal = GaussianPulse(f0, max(fw, 0.1 * f0))
    out = {"wavelengths": wl, "mode_neff": neffs}
    results = {}
    names = [p["name"] for p in ports]
    inputs = list(input_ports or names)
    run_ports, source_of = _symmetry_plan(names, _resolve_port_symmetry(component, ports, port_symmetry), inputs)
    for src in run_ports:
        s = Simulation(**{k: v for k, v in sim.__dict__.items()})
        s.sources = []
        s.monitors = []
        axis, pos, direc, pmin, pmax = planes[src]
        s.add_source(ModeSource(axis, pos, direc, pmin, pmax, profiles[src], signal, name=f"src_{src}"))
        for p in ports:
            a, ps, d, mn, mx = planes[p["name"]]
            if p["name"] == src:
                # monitor just downstream of the source (towards the device)
                # measures the incident and reflected modes
                shift = 2 * (sim_step(sim, a, ps))
                ps = ps + (shift if d == "+" else -shift)
                mn = list(mn)
                mx = list(mx)
                mn[a] = mx[a] = ps
            s.add_monitor(ModeMonitor(f"m_{p['name']}", a, ps, mn, mx, freqs, [profiles[p["name"]]]))
        r = s.run(device=device, progress=progress)
        results[src] = r
        a_plus, a_minus = r.mode_amplitudes(f"m_{src}")
        a_in = a_plus if direc == "+" else a_minus
        for p in ports:
            ap, am = r.mode_amplitudes(f"m_{p['name']}")
            d = planes[p["name"]][2]
            a_out = am if d == "+" else ap  # leaving the device = against the into-device direction
            if p["name"] == src:
                a_out = am if direc == "+" else ap
            out[f"{p['name']}@0,{src}@0"] = a_out / a_in
    _copy_symmetric_columns(out, names, source_of)
    if return_results:
        return out, results
    return out


def _copy_symmetric_columns(out, names, source_of):
    """Fill the columns of inputs that were not excited: S[j, p] = S[inverse[j], r]."""
    for p, (r, inverse) in source_of.items():
        for j in names:
            out[f"{j}@0,{p}@0"] = out[f"{inverse[j]}@0,{r}@0"].copy()


def sim_step(sim, axis, position):
    """Local grid spacing at `position` along `axis`."""
    g = sim.grid()[axis]
    i = int(np.clip(np.searchsorted(g, position), 1, len(g) - 1))
    return float(g[i] - g[i - 1])
