"""lightsim inverse-design gradient benchmark matching fdtdx_gradient.py.

Same scene: straight 400 x 400 nm silicon waveguides in air through a
1.6 x 1.6 x 0.4 um design region, 20 nm grid, 10-cell CPML inside the volume
(same cell count as FDTDX), same number of time steps. The gradient of the
output mode power with respect to the permittivity of every design-region Yee
cell comes from lightsim's adjoint method (one forward and one adjoint run,
`lightsim.adjoint.mode_power_gradient`).

Usage: python lightsim_gradient.py --length 8e-6 --width 4e-6 --time 100e-15 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import lightsim as ls
from lightsim.adjoint import mode_power_gradient
from lightsim.spec import PML

C0 = 299792458.0


def build(length, width, height, sim_time):
    dx = 20e-9
    lam = 1.55e-6
    f0 = C0 / lam
    pml = PML(layers=10, inside=True)
    sim = ls.Simulation(
        min=[-length / 2, -width / 2, -height / 2],
        max=[length / 2, width / 2, height / 2],
        mesh=ls.UniformMesh(dx),
        run_time=sim_time,
        boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]},
    )
    sim.add_medium(ls.Medium("si", eps=12.25))
    sim.add_medium(ls.Medium("design", eps=0.5 * (1 + 12.25)))
    sim.add_structure(ls.Box([-length, -200e-9, -200e-9], [length, 200e-9, 200e-9]), "si")
    dmin, dmax = [-0.8e-6, -0.8e-6, -200e-9], [0.8e-6, 0.8e-6, 200e-9]
    sim.add_structure(ls.Box(dmin, dmax), "design")
    xs = -length / 2 + 14 * dx
    xo = length / 2 - 15 * dx
    bmin, bmax = [xs, -width / 2 + 12 * dx, -height / 2 + 12 * dx], [xs, width / 2 - 12 * dx, height / 2 - 12 * dx]
    mode = ls.solve_modes(sim, 0, xs, bmin, bmax, f0, num_modes=1)[0]
    sim.add_source(ls.ModeSource(0, xs, "+", bmin, bmax, mode.profile, ls.GaussianPulse(f0, 0.1 * f0), name="src"))
    out = {"axis": 0, "position": xo, "min": [xo, bmin[1], bmin[2]], "max": [xo, bmax[1], bmax[2]], "profile": mode.profile, "direction": "+"}
    return sim, out, f0, dmin, dmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=8e-6)
    ap.add_argument("--width", type=float, default=4e-6)
    ap.add_argument("--height", type=float, default=1.5e-6)
    ap.add_argument("--time", type=float, default=50e-15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    sim, out, f0, dmin, dmax = build(a.length, a.width, a.height, a.time)
    setup = time.perf_counter() - t0
    g = sim.grid()
    cells = (len(g[0]) - 1) * (len(g[1]) - 1) * (len(g[2]) - 1)
    tf, tg = [], []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        r = sim.run(device=a.device)
        tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        res = mode_power_gradient(sim, dmin, dmax, out, f0, device=a.device)
        tg.append(time.perf_counter() - t0)
    gl1 = float(sum(np.sum(np.abs(v[1])) for v in res.grad.values()))
    rep = {
        "solver": "lightsim",
        "device": r.stats["backend"],
        "cells": int(cells),
        "steps": int(r.stats["steps"]),
        "forward_seconds": min(tf),
        "gradient_seconds": min(tg),
        "setup_seconds": setup,
        "fom": res.fom,
        "grad_l1": gl1,
        "finite": bool(all(np.all(np.isfinite(v[1])) for v in res.grad.values())),
    }
    print(json.dumps(rep))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
