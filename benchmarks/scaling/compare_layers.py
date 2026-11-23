"""Layer-batched composite (one simulation per layer and colour) against the
per-block composite and the monolithic FDTD: S and gradient.

    python compare_layers.py --N 8 --D 4 --rho results/rho_N8_n4_D4_A.npz
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
    ap.add_argument("--rho", default=None)
    ap.add_argument("--colours", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    devices = a.device.split(";")
    freqs = np.array([C0 / (M.LAM0 * 1e-6)])
    _, blocks = M.mesh_network(a.N, a.n, a.D)
    design = invdes.CompositeDesign([M.block_design(bl["n"]) for bl in blocks if bl["designed"]])
    if a.rho:
        z = np.load(a.rho)
        rho, beta = z["rho"], float(z["beta"])
    else:
        rho, beta = design.random(seed=5, mean=0.5, amplitude=0.4), 8.0
    target = M.dft_target(a.N)
    obj = lambda S: invdes.matrix_fidelity(S, target)  # noqa: E731
    res = {"N": a.N, "D": a.D, "colours": a.colours}
    G = {}
    for name, cp in [("block", composite(a.N, a.n, a.D, freqs, devices, M.STUB, False)), ("layer", composite_layers(a.N, a.n, a.D, freqs, devices, M.STUB, a.colours))] + ([("mono", M.monolithic_problem(a.N, a.n, a.D, freqs, devices=devices[0])[0])] if a.mono else []):
        inv = invdes.InverseDesign(cp, design, objective=obj)
        t0 = time.time()
        v, g, info = inv.value_and_grad(rho, beta)
        G[name] = (info["S"], g)
        res[name] = {"fom": v, "wall": time.time() - t0, "runs": info["simulations"], "sim_seconds": info["stats"]["sim_seconds"]}
        print(name, json.dumps(res[name]), flush=True)
    for x in G:
        for y in G:
            if x < y:
                Sx, gx = G[x]
                Sy, gy = G[y]
                res[f"{x}_vs_{y}"] = {"S_rel": float(np.linalg.norm(Sx - Sy) / np.linalg.norm(Sy)), "grad_cos": float(np.sum(gx * gy) / np.linalg.norm(gx) / np.linalg.norm(gy)), "grad_rel": float(np.linalg.norm(gx - gy) / np.linalg.norm(gy))}
                print(x, y, res[f"{x}_vs_{y}"], flush=True)
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"layers_N{a.N}_D{a.D}_c{a.colours}{'_opt' if a.rho else ''}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
