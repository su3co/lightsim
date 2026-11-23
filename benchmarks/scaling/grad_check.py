"""Gradient of the composite solver against (a) finite differences of its own
objective and (b) the adjoint gradient of the monolithic FDTD problem.

    python grad_check.py --N 8 --n 4 --D 4 --rho results/rho_N8_n4_D4.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mesh2d as M
from lightsim import invdes
from lightsim.spec import C0
from validate_mesh import composite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--D", type=int, default=4)
    ap.add_argument("--rho", default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--h", type=float, default=0.01)
    ap.add_argument("--dirs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reflections", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    devices = a.device.split(";")
    freqs = np.array([C0 / (M.LAM0 * 1e-6)])
    _, blocks = M.mesh_network(a.N, a.n, a.D)
    design = invdes.CompositeDesign([M.block_design(bl["n"]) for bl in blocks if bl["designed"]])
    if a.rho:
        z = np.load(a.rho)
        rho, beta = z["rho"], float(z["beta"]) if a.beta is None else a.beta
    else:
        rho, beta = design.random(seed=3, mean=0.5, amplitude=0.4), a.beta or 4.0
    target = M.dft_target(a.N)
    obj = lambda S: invdes.matrix_fidelity(S, target)  # noqa: E731
    cp = composite(a.N, a.n, a.D, freqs, devices, M.STUB, a.reflections)
    inv_c = invdes.InverseDesign(cp, design, objective=obj)
    mp, _ = M.monolithic_problem(a.N, a.n, a.D, freqs, devices=devices[0])
    inv_m = invdes.InverseDesign(mp, design, objective=obj)
    vc, gc, _ = inv_c.value_and_grad(rho, beta)
    vm, gm, _ = inv_m.value_and_grad(rho, beta)
    rng = np.random.default_rng(0)
    rows = []
    for k in range(a.dirs):
        d = np.stack([invdes.filter_density(rng.standard_normal(rho.shape[1:]), invdes.conic_kernel(4)) for _ in range(rho.shape[0])])
        d /= np.abs(d).max()
        fc = lambda r: obj(inv_c.s_matrix(r, beta))[0]  # noqa: E731
        fd = (fc(np.clip(rho + a.h * d, 0, 1)) - fc(np.clip(rho - a.h * d, 0, 1))) / (2 * a.h)
        row = {"dir": k, "composite_adjoint": float(np.sum(gc * d)), "composite_fd": float(fd), "monolithic_adjoint": float(np.sum(gm * d))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    cos = float(np.sum(gc * gm) / np.linalg.norm(gc) / np.linalg.norm(gm))
    res = {"value_composite": vc, "value_monolithic": vm, "cosine_composite_vs_monolithic_gradient": cos, "rel_norm_diff": float(np.linalg.norm(gc - gm) / np.linalg.norm(gm)), "directions": rows, "beta": beta}
    print(json.dumps({k: v for k, v in res.items() if k != "directions"}))
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"gradcheck_N{a.N}_n{a.n}_D{a.D}{'_opt' if a.rho else '_rand'}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
