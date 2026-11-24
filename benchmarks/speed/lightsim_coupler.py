"""lightsim version of FDTDX's own speed benchmark
(fdtdx/performance/directional_coupler.py): an SOI directional coupler from
`coupler.gds`, non-uniform grid, 12-cell CPML (placed inside the domain like
FDTDX), TE mode source at port o1 and mode detectors at o1 (normalisation),
o4 (through) and o3 (cross). The grid is built with exactly the same node
construction as FDTDX, so both codes advance the same number of cells for the
same number of steps.

Usage: python lightsim_coupler.py --gds /path/to/coupler.gds --cells-per-lambda 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import lightsim as ls

LAM = 1550e-9
EPS_SI = 12.25
EPS_OX = 2.25
NSI = math.sqrt(EPS_SI)
NOX = math.sqrt(EPS_OX)
DOM = (42e-6, 7e-6, 4e-6)
PML = 12
TSI = 220e-9
SRC_SPAN_Y = 2e-6
GDS_CENTER = (10e-6, 0.368e-6)
PORTS = {
    "o1": (-10e-6, -1.632e-6, 180.0),
    "o2": (-10e-6, 2.368e-6, 180.0),
    "o3": (30e-6, 2.368e-6, 0.0),
    "o4": (30e-6, -1.632e-6, 0.0),
}
PORT_W = 0.5e-6


def edges(cpl: int):
    dx_f = LAM / (NSI * cpl)
    dx_c = LAM / (NOX * cpl)
    um = 1e-6
    x = np.linspace(0.0, DOM[0], round(DOM[0] / dx_f) + 1)
    y = np.concatenate(
        [
            np.linspace(0.0, 0.5 * um, round(0.5 * um / dx_c) + 1),
            np.linspace(0.5 * um, DOM[1] - 0.5 * um, round((DOM[1] - 1 * um) / dx_f) + 1)[1:],
            np.linspace(DOM[1] - 0.5 * um, DOM[1], round(0.5 * um / dx_c) + 1)[1:],
        ]
    )
    z = np.concatenate(
        [
            np.linspace(0.0, 1.0 * um, round(1.0 * um / dx_c) + 1),
            np.linspace(1.0 * um, 3.0 * um, round(2.0 * um / dx_f) + 1)[1:],
            np.linspace(3.0 * um, DOM[2], round(1.0 * um / dx_c) + 1)[1:],
        ]
    )
    return x, y, z, dx_f


def to_sim(xg, yg):
    return xg - GDS_CENTER[0] + DOM[0] / 2, yg - GDS_CENTER[1] + DOM[1] / 2


def polygons(gds_path: str):
    """Coupler polygons on layer 1 (sim coordinates) plus port stubs to the
    domain edges (as FDTDX's extend_gds_with_port_stubs)."""
    import gdstk

    lib = gdstk.read_gds(gds_path)
    cell = {c.name: c for c in lib.cells if not c.name.startswith("$$$")}["coupler"]
    polys = []
    for p in cell.get_polygons(depth=None):
        if p.layer != 1:
            continue
        pts = np.asarray(p.points) * 1e-6
        xs, ys = to_sim(pts[:, 0], pts[:, 1])
        polys.append(list(zip(xs.tolist(), ys.tolist())))
    for name, (px, py, o) in PORTS.items():
        sx, sy = to_sim(px, py)
        x0, x1 = (-1e-6, sx) if abs(o - 180) < 1 else (sx, DOM[0] + 1e-6)
        polys.append([(x0, sy - PORT_W / 2), (x1, sy - PORT_W / 2), (x1, sy + PORT_W / 2), (x0, sy + PORT_W / 2)])
    return polys


def build(gds_path: str, cpl: int):
    x, y, z, dx_f = edges(cpl)
    f0 = ls.C0 / LAM
    sim_time = 2.0 * NSI * DOM[0] / ls.C0
    pml = ls.PML(PML, inside=True)
    sim = ls.Simulation(
        min=[0, 0, 0],
        max=list(DOM),
        mesh={"type": "explicit", "x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
        run_time=sim_time,
        boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]},
        background="oxide",
    )
    sim.add_medium(ls.Medium("oxide", eps=EPS_OX))
    sim.add_medium(ls.Medium("si", eps=EPS_SI))
    z0 = DOM[2] / 2 - TSI / 2
    for poly in polygons(gds_path):
        sim.add_structure(ls.PolySlab(poly, z0, z0 + TSI), "si")
    planes = {}
    for name, (px, py, _) in PORTS.items():
        sx, sy = to_sim(px, py)
        planes[name] = (sx, [sx, sy - SRC_SPAN_Y / 2, 0.0], [sx, sy + SRC_SPAN_Y / 2, DOM[2]])
    return sim, planes, f0, dx_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gds", required=True)
    ap.add_argument("--cells-per-lambda", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="limit the run (throughput comparisons)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    sim, planes, f0, dx_f = build(a.gds, a.cells_per_lambda)
    sx, mn, mx = planes["o1"]
    modes = ls.solve_modes(sim, 0, sx, mn, mx, f0, num_modes=2)
    te = max(modes, key=lambda m: (m.te_fraction > 0.5, m.neff.real))
    sig = ls.GaussianPulse(f0, f0 / 10)
    sim.add_source(ls.ModeSource(0, sx, "+", mn, mx, te.profile, sig, name="src"))
    det = {"det_source": ("o1", dx_f), "det_thru": ("o4", 0.0), "det_cross": ("o3", 0.0)}
    for dname, (port, shift) in det.items():
        px, pmn, pmx = planes[port]
        pmn = list(pmn)
        pmx = list(pmx)
        pmn[0] = pmx[0] = px + shift
        # same profile for all ports (identical waveguides)
        sim.add_monitor(ls.ModeMonitor(dname, 0, px + shift, pmn, pmx, [f0], [te.profile]))
    if a.max_steps:
        dt0, _ = sim.time_info()
        sim.run_time = a.max_steps * dt0 * 0.999999
    setup = time.perf_counter() - t0
    dt, nsteps = sim.time_info()
    t1 = time.perf_counter()
    res = sim.run(device=a.device)
    wall = time.perf_counter() - t1
    norm = abs(res.mode_amplitudes("det_source")[0][0]) ** 2
    t_thru = abs(res.mode_amplitudes("det_thru")[0][0]) ** 2 / norm
    t_cross = abs(res.mode_amplitudes("det_cross")[0][0]) ** 2 / norm
    gx, gy, gz = sim.grid()
    shape = [len(gx) - 1, len(gy) - 1, len(gz) - 1]
    out = {
        "solver": "lightsim",
        "cells_per_lambda": a.cells_per_lambda,
        "grid_shape": shape,
        "time_steps": res.stats["steps"],
        "run_seconds": res.stats["seconds"],
        "wall_seconds_incl_compile": wall,
        "mode_solve_and_setup_seconds": setup,
        "mcups": res.stats["mcells_per_s"],
        "backend": res.stats["backend"],
        "neff": te.neff.real,
        "transmissions": {"det_thru": t_thru, "det_cross": t_cross},
    }
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
