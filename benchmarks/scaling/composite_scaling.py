"""Cost of one exact gradient (all N^2 matrix entries) of a brick-wall mesh
of 4-port blocks with D = N/2 layers (enough for every input to reach every
output), composite solver against monolithic FDTD.

    python composite_scaling.py --Ns 8,16,32 --mono-Ns 8,16 --device cuda:0
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
from validate_mesh import composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="8,16,32")
    ap.add_argument("--mono-Ns", default="8")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ff", action="store_true", help="feed-forward block model (n fwd + n adj runs per block)")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    devices = a.device.split(";")
    freqs = np.array([C0 / (M.LAM0 * 1e-6)])
    rows = []
    mono = [int(v) for v in a.mono_Ns.split(",") if v]
    for N in [int(v) for v in a.Ns.split(",")]:
        D = max(2, N // 2)
        _, blocks = M.mesh_network(N, a.n, D)
        design = invdes.CompositeDesign([M.block_design(bl["n"]) for bl in blocks if bl["designed"]])
        rho = design.random(seed=0, mean=0.5, amplitude=0.4)
        eps = design.eps(rho, 4.0)
        target = M.dft_target(N)
        t0 = time.time()
        cp = composite(N, a.n, D, freqs, devices, M.STUB, not a.ff)
        t_setup = time.time() - t0
        t0 = time.time()
        sol = cp.solve(eps, adjoint=True)
        v, g = invdes.matrix_fidelity(sol.S, target)
        grads = sol.vjp(g)
        t_c = time.time() - t0
        row = {"N": N, "D": D, "blocks_designed": len(design.designs), "composite_s": t_c, "composite_setup_s": t_setup, "composite_runs": len(sol.stats["runs"]), "composite_sim_s": sol.stats["sim_seconds"], "network_s": sol.stats["network_seconds"], "block_cells": int(np.mean(sol.stats["cells"])), "device": a.device}
        if N in mono:
            mp, _ = M.monolithic_problem(N, a.n, D, freqs, devices=devices[0], engine=a.engine)
            t0 = time.time()
            ms = mp.solve(eps, adjoint=True)
            _, gm = invdes.matrix_fidelity(ms.S, target)
            ms.vjp(gm)
            row.update({"mono_s": time.time() - t0, "mono_runs": len(ms.stats["runs"]), "mono_cells": ms.stats["cells"], "mono_steps": int(np.mean(ms.stats["steps"])), "mono_sim_s": ms.stats["sim_seconds"], "rel_err_S": float(np.linalg.norm(sol.S - ms.S) / np.linalg.norm(ms.S))})
        rows.append(row)
        print(json.dumps(row), flush=True)
        del grads
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"composite_scaling{a.tag}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
