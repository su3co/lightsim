"""Lumerical FDTD timing on the lightsim / FDTDX benchmark scenes.

UNTESTED: no Lumerical install or license was available while writing this.
It uses the documented `lumapi` scripting calls; property names can differ
between Lumerical versions (check `?set;` in your version if a `set` fails).

Scenes (same as `lightsim bench --scene ...` and `benchmarks/fdtdx_*.py`):

* ``box``     - vacuum box of NX x NY x NZ cells (25 nm, uniform mesh),
                12-cell PML, centred Ez dipole, fixed number of steps
* ``lorentz`` - the same box filled with a Lorentz medium (eps_inf 2.25,
                delta_eps 1, f0 = 2 x 200 THz, gamma = 1e13 rad/s)

The time step is set to match lightsim/FDTDX (Courant 0.99 of the 3-D limit)
and early shutoff is disabled, so all three codes run the same number of
cell updates. The script reports wall time of `run()` and Gcells/s.

Usage (on a machine with Lumerical and a license; `lumapi` on PYTHONPATH,
e.g. /opt/lumerical/v241/api/python):

    python lumerical_bench.py --scene box --n 400,400,400 --steps 200 --gpu
"""

from __future__ import annotations

import argparse
import json
import math
import time

C0 = 299792458.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["box", "lorentz"], default="box")
    ap.add_argument("--n", default="400,400,400")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gpu", action="store_true", help="use Lumerical's GPU solver (express mode)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import lumapi  # noqa: PLC0415  (only available with a Lumerical install)

    n = [int(v) for v in a.n.split(",")]
    dx = 25e-9
    pml = 12
    span = [(c - 2 * pml) * dx for c in n]  # PML outside the region, as in `lightsim bench`
    dt = 0.99 * dx / (C0 * math.sqrt(3))
    t_sim = a.steps * dt

    fdtd = lumapi.FDTD(hide=True)
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    for ax, s in zip("xyz", span):
        fdtd.set(ax, 0.0)
        fdtd.set(f"{ax} span", s)
    fdtd.set("mesh type", "uniform")
    for ax in "xyz":
        fdtd.set(f"define {ax} mesh by", "maximum mesh step")
        fdtd.set(f"d{ax}", dx)
    fdtd.set("pml layers", pml)
    fdtd.set("simulation time", t_sim)
    fdtd.set("set time step stability factor", True)
    fdtd.set("time step stability factor", 0.99)
    fdtd.set("use early shutoff", False)
    if a.gpu:
        fdtd.set("express mode", True)
        # GPU resource (Lumerical 2023 R2+); adjust to your resource setup
        fdtd.setresource("FDTD", 1, "GPU", True)
    if a.scene == "lorentz":
        w = 2 * math.pi * 2e14
        fdtd.setmaterial(fdtd.addmaterial("Lorentz"), "name", "lorentz_bench")
        fdtd.setmaterial("lorentz_bench", "Permittivity", 2.25)
        fdtd.setmaterial("lorentz_bench", "Lorentz Permittivity", 1.0)
        fdtd.setmaterial("lorentz_bench", "Lorentz Resonance", 2 * w)
        fdtd.setmaterial("lorentz_bench", "Lorentz Linewidth", 1e13)
        fdtd.addrect()
        for ax, s in zip("xyz", span):
            fdtd.set(ax, 0.0)
            fdtd.set(f"{ax} span", 2 * s)
        fdtd.set("material", "lorentz_bench")
    fdtd.adddipole()
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("z", 0.0)
    fdtd.set("theta", 0.0)  # polarization along z
    fdtd.set("center wavelength", C0 / 2e14)
    fdtd.set("wavelength span", 0.5e-6)
    fdtd.save("lightsim_bench.fsp")
    t0 = time.perf_counter()
    fdtd.run()
    secs = time.perf_counter() - t0
    cells = n[0] * n[1] * n[2]
    res = {
        "solver": "lumerical",
        "scene": a.scene,
        "gpu": a.gpu,
        "grid": n,
        "cells": cells,
        "steps": a.steps,
        "seconds": secs,
        "gcells_per_s": cells * a.steps / secs / 1e9,
        "note": "wall time of run(), includes Lumerical's meshing and solver start-up",
    }
    print(json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    fdtd.close()


if __name__ == "__main__":
    main()
