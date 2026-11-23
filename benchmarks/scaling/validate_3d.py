"""3-D check of the composite solver: a two-layer mesh of 2-port blocks in
220 nm silicon (4 ports), composite S against the monolithic 3-D FDTD.

    python validate_3d.py --device cpu --kind wg_bumps

The block designs are waveguides through the design region with random
bumps (low loss: the composite approximation is meant for blocks that guide
most of the light), or random binary patterns (``--kind binary``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import lightsim as ls
from lightsim import invdes
from lightsim.spec import C0, Box, Medium

um = 1e-6
DX, NZF = 0.04, 5  # in-plane cell (um), cells through the film
T_SI, N_SI, N_OX = 0.22, 3.476, 1.444
PITCH, W_WG, LB, STUB, GAP = 1.6, 0.5, 2.4, 1.0, 1.0
MARGIN, MARGIN_Y = 0.8, 0.8


def snap(v):
    return float(np.round(v / DX) * DX)


def zgrid():
    dz = T_SI / NZF
    nclad = int(round(0.7 / dz))
    z0 = T_SI / 2 - (NZF // 2 + nclad) * dz
    return z0, z0 + (2 * nclad + NZF) * dz, dz


def sim3d(x0, x1, y0, y1, ys):
    z0, z1, dz = zgrid()
    s = ls.Simulation(min=[snap(x0) * um, snap(y0) * um, z0 * um], max=[snap(x1) * um, snap(y1) * um, z1 * um], mesh=ls.UniformMesh([DX * um, DX * um, dz * um]), run_time=1.0e-12, background="sio2", shutoff=1e-5)
    s.add_medium(Medium("sio2", n=N_OX))
    s.add_medium(Medium("si", n=N_SI))
    for y in ys:
        s.add_structure(Box([-1.0, (snap(y) - W_WG / 2) * um, 0.0], [1.0, (snap(y) + W_WG / 2) * um, T_SI * um]), "si")
    return s


_PROF = {}


def profile(f0):
    if "p" not in _PROF:
        s = sim3d(-1.6, 1.6, -1.6, 1.6, [0.0])
        p = invdes.port_mode(s, "ref", 0, 0.0, [0.0, -PITCH / 2 * um, -0.6 * um], [0.0, PITCH / 2 * um, (T_SI + 0.6) * um], f0, "+")
        _PROF["p"] = p.profiles[0]
    return _PROF["p"]


def port(name, x, y, d, f0):
    return invdes.Port(name, 0, snap(x) * um, [snap(x) * um, (snap(y) - PITCH / 2) * um, -0.6 * um], [snap(x) * um, (snap(y) + PITCH / 2) * um, (T_SI + 0.6) * um], [profile(f0)], d)


def design(n, xc, yc):
    h = (n * PITCH - GAP) / 2
    nx, ny = int(round(LB / DX)) + 1, int(round(2 * h / DX)) + 1
    return invdes.Design.slab([(xc - LB / 2) * um, (yc - h) * um], [(xc + LB / 2) * um, (yc + h) * um], (0.0, T_SI * um), N_OX**2, N_SI**2, shape=(nx, ny))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kind", default="wg_bumps")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    f0 = C0 / (1.55 * um)
    freqs = [f0]
    N, n = 4, 2
    ys = [(j - (N - 1) / 2) * PITCH for j in range(N)]
    Lx = LB + 2 * STUB
    xs = [0.0, snap(Lx)]
    # layer 0: blocks on ports (0,1) and (2,3); layer 1: (1,2), ports 0 and 3 pass straight through
    blocks = [(0, 0, 2), (0, 2, 2), (1, 1, 2)]
    rng = np.random.default_rng(1)
    designs, rhos = [], []
    for layer, first, k in blocks:
        yc = 0.5 * (ys[first] + ys[first + k - 1])
        d = design(k, xs[layer], yc)
        X, Y = d.coordinates()
        rho = np.zeros(d.shape)
        for j in range(k):
            yy = ys[first + j]
            rho[:, np.abs(Y / um - yy) <= W_WG / 2] = 1.0
        if a.kind == "binary":
            rho = (invdes.filter_density(rng.standard_normal(d.shape), invdes.conic_kernel(3)) > 0).astype(float)
        else:
            bump = invdes.filter_density(rng.standard_normal(d.shape), invdes.conic_kernel(3))
            rho = np.clip(rho + 0.6 * (bump > 0.5 * bump.std()), 0, 1)
        designs.append(d)
        rhos.append(rho)
    eps = [e for d, r in zip(designs, rhos) for e in d.eps(r, 16.0)]
    t0 = time.time()
    blks = []
    for (layer, first, k), d in zip(blocks, designs):
        xc = xs[layer]
        py = ys[first : first + k]
        s = sim3d(xc - LB / 2 - STUB - MARGIN, xc + LB / 2 + STUB + MARGIN, py[0] - PITCH / 2 - MARGIN_Y, py[-1] + PITCH / 2 + MARGIN_Y, py)
        ports = [port(f"L{j}", xc - LB / 2 - STUB, y, "+", f0) for j, y in enumerate(py)] + [port(f"R{j}", xc + LB / 2 + STUB, y, "-", f0) for j, y in enumerate(py)]
        prob = invdes.SMatrixProblem(s, d.grids(), ports[:k], ports[k:], freqs, devices=a.device)
        blks.append(invdes.Block(prob, list(range(k)), list(range(k, 2 * k)), ports=2 * k))
    # pass-through straight waveguide (1 port per side), solved once
    xc = xs[1]
    s = sim3d(xc - LB / 2 - STUB - MARGIN, xc + LB / 2 + STUB + MARGIN, -PITCH / 2 - MARGIN_Y, PITCH / 2 + MARGIN_Y, [0.0])
    pp = [port("L", xc - LB / 2 - STUB, 0.0, "+", f0), port("R", xc + LB / 2 + STUB, 0.0, "-", f0)]
    sp = invdes.SMatrixProblem(s, [], [pp[0]], [pp[1]], freqs, devices=a.device).solve([], adjoint=False).S
    straight = np.zeros((2, 2, 1), complex)
    straight[1, 0] = sp[0, 0]
    blks = blks[:2] + [invdes.Block(straight.copy()), blks[2], invdes.Block(straight.copy())]
    # network: blocks 0,1 (layer 0), 2 = pass (port 0), 3 = block (1,2), 4 = pass (port 3)
    conns = [((0, 2), (2, 0)), ((0, 3), (3, 0)), ((1, 2), (3, 1)), ((1, 3), (4, 0))]
    ext = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 2), (3, 3), (4, 1)]
    sides = [["in"] * (P // 2) + ["out"] * (P // 2) for P in (4, 4, 2, 4, 2)]
    net = invdes.Network([4, 4, 2, 4, 2], conns, ext, sides=sides)
    cp = invdes.CompositeProblem(blks, net, list(range(4)), list(range(4, 8)), devices=a.device, reflections=False)
    Sc = cp.solve(eps, adjoint=False).S
    t_c = time.time() - t0
    t0 = time.time()
    ms = sim3d(xs[0] - LB / 2 - STUB - MARGIN, xs[1] + LB / 2 + STUB + MARGIN, ys[0] - PITCH / 2 - MARGIN_Y, ys[-1] + PITCH / 2 + MARGIN_Y, ys)
    ins = [port(f"in{q}", xs[0] - LB / 2 - STUB, y, "+", f0) for q, y in enumerate(ys)]
    outs = [port(f"out{q}", xs[1] + LB / 2 + STUB, y, "-", f0) for q, y in enumerate(ys)]
    grids = []
    for i, d in enumerate(designs):
        g = d.grids()[0]
        g.name, g.medium = f"d{i}", dict(g.medium, name=f"invdes_d{i}")
        grids.append(g)
    mp = invdes.SMatrixProblem(ms, grids, ins, outs, freqs, devices=a.device)
    msol = mp.solve(eps, adjoint=False)
    Sm = msol.S
    t_m = time.time() - t0
    err = float(np.linalg.norm(Sc - Sm) / np.linalg.norm(Sm))
    res = {"kind": a.kind, "rel_err_fro": err, "max_abs_err": float(np.abs(Sc - Sm).max()), "throughput_mono": float(np.sum(np.abs(Sm) ** 2) / N), "composite_s": t_c, "mono_s": t_m, "mono_cells": msol.stats["cells"],
           "Sc": [Sc[..., 0].real.tolist(), Sc[..., 0].imag.tolist()], "Sm": [Sm[..., 0].real.tolist(), Sm[..., 0].imag.tolist()]}
    print(json.dumps({k: v for k, v in res.items() if k not in ("Sc", "Sm")}))
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"validate3d_{a.kind}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
