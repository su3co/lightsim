"""Monolithic lightsim FDTD cost of one run through a free-form N-port
device (2.5-D, random binary design of depth = width, as in fdfd2d.py),
against N: cells, time steps to a 1e-5 decay, wall time.

    python fdtd_scaling.py --Ns 8,16,32 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mesh2d as M
import window_fdtd as W
import lightsim as ls
from lightsim.spec import C0, Box, IndexGrid, Medium

um = 1e-6


def build(N, pitch=1.5, seed=0):
    class A:
        pass

    a = A()
    a.N, a.pitch, a.depth, a.seed, a.contrast = N, pitch, 0.0, seed, 0.0
    eps, xs, ys, py, xin, xout = W.build_case("rand_binary", a)
    X, Y = xs[-1] - xs[0] + W.DX_UM, ys[-1] - ys[0] + W.DX_UM
    sim = M._sim(-X / 2, X / 2, -Y / 2, Y / 2, run_time=40 * M.N_SLAB * X * um / C0 + 1e-12)
    sim.shutoff = 1e-5
    for y in py:
        sim.add_structure(Box([-1.0, (y - 0.25) * um, -1.0], [1.0, (y + 0.25) * um, 1.0]), "eim_si")
    L = N * pitch
    sel = (np.abs(xs) <= L / 2)
    sely = (np.abs(ys) <= L / 2)
    e = eps[np.ix_(sel, sely)]
    sim.add_medium(Medium("grid", eps=float(e.max())))
    sim.structures.append(ls.spec.Structure(IndexGrid([xs[sel][0] * um, ys[sely][0] * um, -1.0], [xs[sel][-1] * um, ys[sely][-1] * um, 1.0], e[:, :, None]), "grid", name="dev"))
    f0 = C0 / (M.LAM0 * um)
    p = M.make_port("in", xin, py[0], "+", f0)
    sim.add_source(ls.ModeSource(0, p.position, "+", p.min, p.max, p.profiles[0], ls.GaussianPulse(f0, 0.1 * f0), name="src"))
    q = M.make_port("out", xout, py[-1], "-", f0)
    sim.add_monitor(ls.ModeMonitor("out", 0, q.position, q.min, q.max, [f0], q.profiles))
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="8,16,32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = []
    for N in [int(v) for v in a.Ns.split(",")]:
        sim = build(N)
        t0 = time.time()
        r = sim.run(device=a.device)
        wall = time.time() - t0
        st = r.stats
        row = {"N": N, "device": a.device, "backend": st.get("backend", ""), "cells": st["cells"], "steps": st["steps"], "wall_s": wall, "engine_s": st.get("seconds", wall), "cell_updates_per_s": st["cells"] * st["steps"] / st.get("seconds", wall)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).mkdir(exist_ok=True)
    tag = a.device.replace(":", "").replace(",", "")
    Path(a.out, f"fdtd_scaling_{tag}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
