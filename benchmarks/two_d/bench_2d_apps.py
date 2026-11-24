"""Time per run of the two applications that moved to the 2-D path: a 2.5-D
effective-index S-parameter run and one planar inverse-design gradient.

``eim``      ``lightsim.eim.write_sparameters_2d`` on the 1x2 MMI splitter of
             gdsfactory's generic PDK (220 nm silicon in oxide, 1.50-1.60 um,
             11 wavelengths, 18 cells per wavelength), one input port.
``gradient`` the ``planar`` scene of ``../gradients/lightsim_autodiff.py``: a 2.5-D design
             region between two input and two output guides, three
             frequencies; one solve + vjp of L = sum |S|^2 through the native
             engine (persistent session) and through the Python engine.

The script only uses API that exists before and after the 2-D mode, so the
same file times both trees. Run it once per tree and device:

    python benchmarks/two_d/bench_2d_apps.py --device cuda:0 --json after_gpu.json
    LIGHTSIM_2D=0 python benchmarks/two_d/bench_2d_apps.py ...    # 2-D problem on the 3-D kernels

Each figure is the best of ``--reps`` runs after one warm-up run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# the planar scene of the gradient benchmark
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gradients"))


def best(fn, reps):
    fn()  # warm-up: kernel loading, session set-up, caches
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def eim_run(device, reps):
    import gdsfactory as gf

    from lightsim import eim

    gf.gpdk.PDK.activate()
    mmi = gf.components.mmi1x2()
    wl = np.linspace(1.5e-6, 1.6e-6, 11)

    def one():
        return eim.write_sparameters_2d(mmi, wavelengths=wl, steps_per_wavelength=18, input_ports=["o1"], device=device, return_results=True)

    t, (sp, runs) = best(one, reps)
    st = runs["o1"].stats
    return {"seconds": t, "cells": int(st["cells"]), "steps": int(st["steps"]), "backend": st["backend"], "split_1550": float(abs(sp["o2@0,o1@0"][5]) ** 2)}


def gradient_run(device, reps):
    from lightsim import invdes
    from lightsim_autodiff import planar_scene

    sim, ports, des, freqs = planar_scene()
    out = {}
    eps = des.eps(des.random(seed=0, mean=0.5, amplitude=0.4), 4.0)
    for engine in ("native", "python"):
        prob = invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=device, engine=engine)

        def one():
            sol = prob.solve(eps)
            return sol, sol.vjp(2 * sol.S)

        t, (sol, _) = best(one, reps)
        out[engine] = {"seconds": t, "steps": [int(s) for s in sol.stats["steps"]], "fom": float(np.sum(np.abs(sol.S) ** 2))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", choices=["eim", "gradient"], default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rep = {"device": a.device, "label": a.label, "LIGHTSIM_2D": os.environ.get("LIGHTSIM_2D", "")}
    if a.only in (None, "eim"):
        rep["eim"] = eim_run(a.device, a.reps)
        print("eim", rep["eim"], flush=True)
    if a.only in (None, "gradient"):
        rep["gradient"] = gradient_run(a.device, a.reps)
        print("gradient", rep["gradient"], flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
