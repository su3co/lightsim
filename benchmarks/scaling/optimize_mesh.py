"""Inverse design of a mesh of free-form blocks with the composite solver.

    python optimize_mesh.py --N 8 --n 4 --D 4 --iters 160 --device "cpu;cpu;cpu;cpu"

The objective is the matrix fidelity to the N-point DFT (one frequency by
default). Checkpoints in results/opt_<tag>.npz (resumable); the final
densities are saved as results/rho_<tag>.npz for validate_mesh.py --rho.
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
from validate_mesh import composite, composite_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reflections", action="store_true")
    ap.add_argument("--wavelengths", default="1.55")
    ap.add_argument("--beta", default="4,2,40,64", help="start,factor,every,max")
    ap.add_argument("--layers", action="store_true", help="one LayerProblem per layer")
    ap.add_argument("--rho0", default=None, help="npz with a starting density (key best_rho or rho)")
    ap.add_argument("--init", default="random", choices=["random", "wg"], help="wg: straight waveguides through every block plus noise (starts with little loss)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    devices = a.device.split(";")
    freqs = np.array([C0 / (float(w) * 1e-6) for w in a.wavelengths.split(",")])
    cp = composite_layers(a.N, a.n, a.D, freqs, devices, M.STUB) if a.layers else composite(a.N, a.n, a.D, freqs, devices, M.STUB, a.reflections)
    _, blocks = M.mesh_network(a.N, a.n, a.D)
    design = invdes.CompositeDesign([M.block_design(bl["n"]) for bl in blocks if bl["designed"]])
    target = M.dft_target(a.N)
    obj = lambda S: invdes.matrix_fidelity(S, target)  # noqa: E731
    inv = invdes.InverseDesign(cp, design, objective=obj)
    tag = f"N{a.N}_n{a.n}_D{a.D}" + ("_refl" if a.reflections else "") + a.tag
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    b0, bf, be, bm = (float(v) for v in a.beta.split(","))
    rho0 = design.random(seed=a.seed, mean=0.5, amplitude=0.3)
    if a.init == "wg":
        rng = np.random.default_rng(a.seed)
        parts = []
        for d in design.designs:
            r = np.where(d.fixed_solid.any(axis=0)[None, :], 0.8, 0.25) + 0.1 * rng.standard_normal(d.shape)
            parts.append(np.clip(r, 0, 1))
        rho0 = np.stack(parts)
    if a.rho0:
        z = np.load(a.rho0)
        rho0 = z["best_rho"] if "best_rho" in z.files else z["rho"]
    t0 = time.time()

    def cb(it, rho, val, beta, info):
        m = invdes.matrix_metrics(info["S"], target)
        print(f"it {it:4d} fom {val:.4f} fid {m['fidelity'].mean():.4f} thr {m['throughput'].mean():.4f} beta {beta:g} runs {info['simulations']} t {time.time() - t0:.0f}s", flush=True)

    rho, hist = invdes.optimize(inv, rho0, a.iters, beta=invdes.beta_schedule(b0, bf, int(be), bm), lr=a.lr, checkpoint=str(out / f"opt_{tag}.npz"), callback=cb, verbose=False)
    beta_end = float(hist["beta"][-1])
    np.savez(out / f"rho_{tag}.npz", rho=hist["best_rho"], beta=beta_end, rho_last=rho)
    (out / f"hist_{tag}.json").write_text(json.dumps({k: np.asarray(v).tolist() for k, v in hist.items() if k not in ("best_rho",)}, default=float))
    print("done", tag, "best", hist["best_fom"], "wall", time.time() - t0)


if __name__ == "__main__":
    main()
