"""Composite (block network) S-matrix against the monolithic FDTD of the same
mesh, 2.5-D.

    python validate_mesh.py --N 8 --n 4 --D 2 --kind binary --device cpu

Prints and saves (json) the transmission matrices of both, their difference,
the per-block losses and the timings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mesh2d as M
from lightsim import invdes
from lightsim.spec import C0


def densities(blocks, n, kind, seed, beta):
    out = []
    rng = np.random.default_rng(seed)
    for bl in blocks:
        if not bl["designed"]:
            continue
        d = M.block_design(bl["n"])
        r = d.random(int(rng.integers(1 << 30)), 0.5, 1.0)
        if kind == "wg":  # straight waveguides through the block: no scattering
            out.append((d, r, d.eps(d.fixed_solid.any(axis=0)[None, :] * np.ones(d.shape), beta)))
            continue
        if kind == "grey":  # low contrast: a weak modulation about the slab index
            e = d.eps(r, 1.0)
            out.append((d, r, [0.75 * M.N_SLAB**2 + 0.25 * M.N_OX**2 + 0.15 * (ei - ei.mean()) for ei in e]))
        else:
            out.append((d, r, d.eps(r, beta)))
    return out


def composite(N, n, D, freqs, devices, stub, reflections, rho_eps=None, adjoint=False):
    net, blocks = M.mesh_network(N, n, D)
    blks = []
    fixed_cache = {}
    for bl in blocks:
        if bl["designed"]:
            prob, des, ins, outs = M.block_problem(bl["n"], True, freqs, devices=devices[0], stub=stub, sides="both" if reflections else "ff")
            blks.append(invdes.Block(prob, ins, outs, ports=2 * bl["n"]))
        else:
            k = bl["n"]
            if k not in fixed_cache:
                prob, _, ins, outs = M.block_problem(k, False, freqs, devices=devices[0], stub=stub, sides="both" if reflections else "ff")
                s = prob.solve([], adjoint=False).S
                full = np.zeros((2 * k, 2 * k, len(freqs)), complex)
                full[np.ix_(outs, ins)] = s
                fixed_cache[k] = full
            blks.append(invdes.Block(fixed_cache[k]))
    ext_in = list(range(N))
    ext_out = list(range(N, 2 * N))
    cp = invdes.CompositeProblem(blks, net, ext_in, ext_out, devices=devices, reflections=reflections)
    return cp


def composite_layers(N, n, D, freqs, devices, stub, colours=2):
    """Feed-forward composite in which every layer's designed blocks are
    solved together (one LayerProblem per layer)."""
    net, blocks = M.mesh_network(N, n, D)
    members = {}
    for layer in range(D):
        lp, idx = M.layer_problem(N, n, layer, freqs, devices=devices[0], stub=stub, colours=colours)
        for i, b in enumerate(idx):
            members[b] = invdes.LayerMember(lp, i)
    blks, fixed_cache = [], {}
    for b, bl in enumerate(blocks):
        k = bl["n"]
        if bl["designed"]:
            blks.append(invdes.Block(members[b], list(range(k)), list(range(k, 2 * k)), ports=2 * k))
        else:
            if k not in fixed_cache:
                prob, _, ins, outs = M.block_problem(k, False, freqs, devices=devices[0], stub=stub, sides="ff")
                s = prob.solve([], adjoint=False).S
                full = np.zeros((2 * k, 2 * k, len(freqs)), complex)
                full[np.ix_(outs, ins)] = s
                fixed_cache[k] = full
            blks.append(invdes.Block(fixed_cache[k]))
    return invdes.CompositeProblem(blks, net, list(range(N)), list(range(N, 2 * N)), devices=devices, reflections=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--kind", default="binary", choices=["binary", "grey", "wg"])
    ap.add_argument("--beta", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stub", type=float, default=M.STUB)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-mono", action="store_true")
    ap.add_argument("--rho", default=None, help="npz with block densities 'rho' [B, nx, ny] and 'beta'")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    devices = a.device.split(";")
    freqs = np.array([C0 / (M.LAM0 * 1e-6)])
    _, blocks = M.mesh_network(a.N, a.n, a.D)
    if a.rho:
        z = np.load(a.rho)
        ds = [M.block_design(bl["n"]) for bl in blocks if bl["designed"]]
        eps = [e for d, r in zip(ds, z["rho"]) for e in d.eps(r, float(z["beta"]))]
    else:
        dens = densities(blocks, a.n, a.kind, a.seed, a.beta)
        eps = [e for _, _, es in dens for e in es]
    res = {"args": vars(a)}
    for refl in (False, True):
        t0 = time.time()
        cp = composite(a.N, a.n, a.D, freqs, devices, a.stub, refl)
        sol = cp.solve(eps, adjoint=False)
        res["composite_refl" if refl else "composite_ff"] = {"S": [sol.S[..., 0].real.tolist(), sol.S[..., 0].imag.tolist()], "wall": time.time() - t0, "sim_seconds": sol.stats["sim_seconds"], "runs": len(sol.stats["runs"])}
        if refl:
            # block losses: 1 - sum_p |S_pq|^2, per input column, designed blocks
            loss = [float(1 - np.min(np.sum(np.abs(Sb[..., 0]) ** 2, axis=0))) for Sb, blk in zip(sol.block_S, cp.blocks) if not blk.fixed]
            res["block_max_loss"] = loss
        print(("refl" if refl else "ff  "), f"wall {time.time() - t0:.1f}s runs {len(sol.stats['runs'])}", flush=True)
    if not a.no_mono:
        t0 = time.time()
        mp, _ = M.monolithic_problem(a.N, a.n, a.D, freqs, devices=devices[0], stub=a.stub)
        msol = mp.solve(eps, adjoint=False)
        res["mono"] = {"S": [msol.S[..., 0].real.tolist(), msol.S[..., 0].imag.tolist()], "wall": time.time() - t0, "sim_seconds": msol.stats["sim_seconds"], "cells": msol.stats["cells"], "steps": msol.stats["steps"]}
        Sm = msol.S[..., 0]
        nm = np.linalg.norm(Sm)
        for key in ("composite_ff", "composite_refl"):
            Sc = np.array(res[key]["S"][0]) + 1j * np.array(res[key]["S"][1])
            err = np.linalg.norm(Sc - Sm) / nm
            res[key]["rel_err_fro"] = float(err)
            res[key]["max_abs_err"] = float(np.abs(Sc - Sm).max())
            print(key, f"relative Frobenius error {err:.4f}, max |dS| {np.abs(Sc - Sm).max():.4f}")
        print("mono throughput", float(np.sum(np.abs(Sm) ** 2) / a.N), "wall", time.time() - t0)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    tag = f"mesh_N{a.N}_n{a.n}_D{a.D}_{a.kind}_s{a.seed}_stub{a.stub:g}" + ("_opt" if a.rho else "")
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
