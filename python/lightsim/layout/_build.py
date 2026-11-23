"""Simulation set-up shared by ``lightsim.gplugins`` and ``lightsim.layout``.

Both fronts reduce a layout to the same three things: the layer-stack
levels with their polygons (:class:`lightsim.gds.StackLayer`, metres; the
port waveguides already extended through the absorber), the region in x and
y, and the ports (``lightsim.gds.port_info`` dicts). This module turns them
into a lightsim :class:`~lightsim.Simulation` (3-D, or the 2-D
effective-index / bulk-index model) and, given the ports, into the
per-source simulations with mode sources and monitors placed the way
gplugins.gmeep places them.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import gds as _gds
from ..modes import solve_modes
from ..sparams import _half_width
from ..spec import C0, AutoMesh, Box, GaussianPulse, Medium, ModeMonitor, ModeSource, PML, PolySlab, Simulation, Structure, UniformMesh, um


def mesh_and_pml(wl_m, resolution, steps_per_wavelength, tpml):
    """(mesh, PML layers, PML thickness in m) for the gplugins resolution arguments."""
    lam0 = float(np.mean(wl_m))
    dx = 1e-6 / resolution
    mesh = AutoMesh(lam0, steps_per_wavelength=steps_per_wavelength, grading=1.3) if steps_per_wavelength else UniformMesh(dx)
    step = dx if not steps_per_wavelength else lam0 / steps_per_wavelength
    pml_layers = max(8, int(round(tpml * 1e-6 / step))) if tpml is not None else 12
    return mesh, pml_layers, pml_layers * step


def build_from_levels(
    levels,
    region,
    wl_m,
    material_index,
    *,
    is_3d,
    effective_index,
    polarization,
    mesh,
    pml_layers,
    zmargin_top,
    zmargin_bot,
    background,
    decay_by,
):
    """The simulation without sources and monitors, and the z range of the ports.

    ``levels``: StackLayer list (wafer levels have no polygons); ``region``:
    ``((x0, y0), (x1, y1))`` in metres; margins in micrometres (0 means
    0.8 um, as in lightsim.gplugins); ``background``: the cladding material.
    """
    (x0, y0), (x1, y1) = region
    idx = _gds.material_table(material_index)
    wafer = [l for l in levels if not l.polygons]
    pattern = [l for l in levels if l.polygons]
    if not pattern:
        raise ValueError("the layout has no polygons on the levels of the layer stack")
    if is_3d:
        media, structures = {}, []

        def register(l):
            if l.material not in media:
                m = _gds._medium_for(l.material, idx, f"layer stack level '{l.name}'")
                if m is not None:
                    media[l.material] = m

        for l in wafer:
            register(l)
            structures.append(Structure(Box([x0 - 1e-3, y0 - 1e-3, l.zmin], [x1 + 1e-3, y1 + 1e-3, l.zmax]), l.material, name=l.name))
        for l in sorted(pattern, key=lambda l: -l.mesh_order):
            register(l)
            for i, poly in enumerate(l.polygons):
                structures.append(Structure(PolySlab(poly, l.zmin, l.zmax, axis=2), l.material, name=f"{l.name}_{i}"))
        media = list(media.values())
        lo, hi = min(l.zmin for l in pattern), max(l.zmax for l in pattern)
        z0 = lo - (zmargin_bot * um if zmargin_bot else 0.8 * um)
        z1 = hi + (zmargin_top * um if zmargin_top else 0.8 * um)
        if not any(m["name"] == background for m in media):
            bg = _gds._medium_for(background, idx, "clad_material")
            if bg is not None:
                media = media + [bg]
        nmax = max([m["eps"][0] for m in media] + [1.0]) ** 0.5
        sim = Simulation(
            min=[x0, y0, z0],
            max=[x1, y1, z1],
            mesh=mesh,
            run_time=1.0,
            boundaries={a: [PML(layers=pml_layers), PML(layers=pml_layers)] for a in "xyz"},
            background=background,
            shutoff=decay_by,
        )
        for m in media:
            sim.add_medium(m)
        sim.structures.extend(structures)
        zr = (z0, z1)
    else:
        from ..eim import _vertical_stack, slab_modes

        pol = (polarization or "te").lower()
        lam0 = float(np.mean(wl_m))
        if effective_index:
            zc = np.mean([0.5 * (l.zmin + l.zmax) for l in pattern])
            n_bg = _gds.material_index_of(background, idx, "background")
            for w in reversed(wafer):
                if w.zmin < zc < w.zmax:
                    n_bg = _gds.material_index_of(w.material, idx, f"layer stack level '{w.name}'")
                    break
            neff = {}
            for l in pattern:
                modes = slab_modes(*_vertical_stack(l, wafer, idx, background), lam0, pol)
                neff[l.name] = modes[0] if modes else n_bg
        else:
            # the plain 2-D cut with the bulk indices, as gplugins.gmeep does for is_3d=False
            neff = {l.name: _gds.material_index_of(l.material, idx, f"layer stack level '{l.name}'") for l in pattern}
            n_bg = _gds.material_index_of(background, idx, "clad_material")
        pml = lambda: PML(layers=pml_layers)  # noqa: E731
        sim = Simulation(
            min=[x0, y0, 0.0],
            max=[x1, y1, 1e-6],
            mesh=mesh,
            run_time=1.0,
            boundaries={"x": [pml(), pml()], "y": [pml(), pml()], "z": [{"type": "periodic"}, {"type": "periodic"}]},
            background="eim_background",
            shutoff=decay_by,
            dimensions=2,
            invariant_axis="z",
            polarization=pol,
        )
        sim.add_medium(Medium("eim_background", n=n_bg))
        for l in sorted(pattern, key=lambda l: neff.get(l.name, n_bg)):  # higher index painted last
            name = f"eim_{l.name}"
            sim.add_medium(Medium(name, n=neff.get(l.name, n_bg)))
            for i, poly in enumerate(l.polygons):
                sim.structures.append(Structure(PolySlab(poly, -1.0, 1.0, axis=2), name, name=f"{l.name}_{i}"))
        if not effective_index and polarization is None:
            sim.polarization = "both"  # the mode of highest index decides, as Meep's band 1 does
        nmax = max([n_bg, *neff.values()])
        zr = (float(sim.min[2]), float(sim.max[2]))
    length = max(x1 - x0, y1 - y0)
    sim.run_time = 30.0 * nmax * length / C0 + 2e-12
    return sim, zr


def port_plane(port, offset: float, z0: float, z1: float, hw: float):
    """``(axis, position, direction into the device, min, max)`` of a port
    plane ``offset`` (m) outside the port (negative: inside)."""
    x, y = port["center"]
    o = port["orientation"] % 360.0
    if min(abs(o), abs(o - 360)) < 1:
        axis, pos, d, c = 0, x + offset, "-", y
    elif abs(o - 180) < 1:
        axis, pos, d, c = 0, x - offset, "+", y
    elif abs(o - 90) < 1:
        axis, pos, d, c = 1, y + offset, "-", x
    elif abs(o - 270) < 1:
        axis, pos, d, c = 1, y - offset, "+", x
    else:
        raise ValueError(f"port {port['name']} has orientation {o}; lightsim needs ports along x or y")
    t = 1 - axis
    pmin, pmax = [0.0, 0.0, z0], [0.0, 0.0, z1]
    pmin[axis] = pmax[axis] = pos
    pmin[t], pmax[t] = c - hw, c + hw
    return axis, pos, d, pmin, pmax


def _solve(sim, plane, freq, num_modes, cache):
    key = (plane[0], round(plane[1] * 1e12), tuple(round(v * 1e12) for v in plane[3] + plane[4]))
    if key not in cache:
        axis, pos, _, pmin, pmax = plane
        cache[key] = sorted(solve_modes(sim, axis, pos, pmin, pmax, freq, num_modes=num_modes), key=lambda m: -m.neff.real)
    modes = cache[key]
    if len(modes) < num_modes:
        raise ValueError(f"the mode solver found {len(modes)} modes at a port, {num_modes} are needed")
    return modes


class PortRuns:
    """The per-source simulations of a set of ports and their evaluation.

    Mode sources ``port_source_offset`` outside the source port (negative:
    inside), monitors ``port_monitor_offset`` outside every port and
    ``gap`` behind the source on the source port (all in metres), as
    gplugins.gmeep places them; ``S = a_out / a_in``.
    """

    def __init__(self, sim, zr, ports, *, port_margin, port_monitor_offset, port_source_offset, gap, port_modes, port_source_modes, freqs, signal):
        self.sim, self.ports = sim, ports
        self.names = [p["name"] for p in ports]
        self.by_name = {p["name"]: p for p in ports}
        self.zr = zr
        self.port_margin = port_margin
        self.port_source_offset = port_source_offset
        self.gap = gap
        self.port_modes = list(port_modes)
        self.freqs = freqs
        self.f0 = float(np.mean(freqs))
        self.signal = signal
        self.nmodes = max(self.port_modes + [m for ms in port_source_modes.values() for m in ms]) + 1
        self.cache = {}
        self.monitor_planes = {p["name"]: self.plane(p, port_monitor_offset) for p in ports}

    def plane(self, p, offset):
        z0, z1 = p.get("z") or self.zr  # a port file may give the mode window's height
        return port_plane(p, offset, z0, z1, _half_width(p, self.ports, self.port_margin))

    def build(self, src, src_mode=0):
        """``(simulation, planes)`` with the mode source at ``src`` (mode ``src_mode``)."""
        sim = self.sim
        s = Simulation(**{k: v for k, v in sim.__dict__.items()})
        s.sources, s.monitors = [], []
        p = self.by_name[src]
        sp_ = self.plane(p, self.port_source_offset)
        mode = _solve(sim, sp_, self.f0, self.nmodes, self.cache)[src_mode]
        s.add_source(ModeSource(sp_[0], sp_[1], sp_[2], sp_[3], sp_[4], mode.profile, self.signal, name=f"src_{src}"))
        planes = dict(self.monitor_planes)
        planes[src] = self.plane(p, self.port_source_offset - self.gap)
        for name in self.names:
            a, pos, _, mn, mx = planes[name]
            profiles = [m.profile for m in _solve(sim, planes[name], self.f0, self.nmodes, self.cache)[: max(self.port_modes) + 1]]
            s.add_monitor(ModeMonitor(f"m_{name}", a, pos, mn, mx, self.freqs, profiles))
        return s, planes

    def evaluate(self, result, src, src_mode, planes) -> dict:
        out = {}
        a_plus, a_minus = result.mode_amplitudes(f"m_{src}", src_mode)
        a_in = a_plus if planes[src][2] == "+" else a_minus
        for name in self.names:
            for m in self.port_modes:
                ap, am = result.mode_amplitudes(f"m_{name}", m)
                a_out = am if planes[name][2] == "+" else ap  # leaving the device
                out[f"{name}@{m},{src}@{src_mode}"] = np.asarray(a_out / a_in)
        return out

    def run(self, runs, device="auto", progress=0) -> list[dict]:
        """Run the (source, mode) pairs ``runs``; one S-parameter column dict each.

        ``device`` is a device string or a list, over which the runs are
        spread one at a time per entry."""
        devices = [device] if isinstance(device, str) else list(device)
        pool: queue.Queue = queue.Queue()
        for d in devices:
            pool.put(d)
        for job in runs:  # solve the port modes once, before the threads share the cache
            self.build(*job)

        def work(job):
            s, planes = self.build(*job)
            dev = pool.get()
            try:
                r = s.run(device=dev, progress=progress)
            finally:
                pool.put(dev)
            return self.evaluate(r, job[0], job[1], planes)

        if len(devices) > 1 and len(runs) > 1:
            with ThreadPoolExecutor(max_workers=len(devices)) as ex:
                return list(ex.map(work, runs))
        return [work(job) for job in runs]


def gaussian_signal(freqs, dfcen=0.2):
    f0 = float(np.mean(freqs))
    return GaussianPulse(f0, max(dfcen * f0, float(np.ptp(freqs))))


def extend_ports(levels, ports, length, port_layers=None):
    """Add the port waveguides (``length`` m beyond each port, the port's
    width) to the patterned level that holds the port: the level named in
    ``port_layers[name]`` or, by default, the patterned level with a polygon
    edge through the port centre (else the first patterned level)."""
    out = [l for l in levels]
    pattern = [l for l in out if l.polygons]
    for p in ports:
        x, y = p["center"]
        o = np.radians(p["orientation"])
        d = np.array([np.cos(o), np.sin(o)])
        n = np.array([-d[1], d[0]]) * p["width"] / 2
        c = np.array([x, y])
        rect = [tuple(c - n), tuple(c - n + d * length), tuple(c + n + d * length), tuple(c + n)]
        target = None
        if port_layers and p["name"] in port_layers:
            target = next((l for l in pattern if l.name == port_layers[p["name"]]), None)
        if target is None:
            target = next((l for l in pattern if _touches(l.polygons, c)), pattern[0] if pattern else None)
        if target is not None:
            target.polygons = list(target.polygons) + [rect]
    return out


def _touches(polygons, c, tol=2e-9) -> bool:
    for poly in polygons:
        pts = np.asarray(poly)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            ab = b - a
            L2 = float(ab @ ab)
            if L2 == 0:
                continue
            t = np.clip(float((c - a) @ ab) / L2, 0, 1)
            if np.linalg.norm(a + t * ab - c) < tol:
                return True
    return False
