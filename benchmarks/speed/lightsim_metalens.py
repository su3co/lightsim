"""A metalens scene for throughput at the scale of published commercial
benchmarks: dielectric pillars of varying radius on a glass substrate, a
plane wave from the substrate, CPML on all six faces, DFT field monitors on
the focal plane and on a cross-section, and a flux monitor above the lens.

The pillar radius follows a hyperbolic lens phase through a linear
radius-to-phase map. That is not a calibrated pillar library, so the lens
focuses but is not an optimised design; the scene is meant to load the solver
the way a metalens simulation does (thousands of sub-wavelength pillars,
optionally dispersive, sources and frequency-domain monitors), not to be a
reference design.

    python benchmarks/speed/lightsim_metalens.py --n 1200,1200,590 --steps 2000 --device cuda:0
    python benchmarks/speed/lightsim_metalens.py --n 400,400,300 --device cuda:0     # runs to the shutoff level

``--dispersive`` gives the pillars one Lorentz pole (a TiO2-like index of
2.39 at 633 nm with normal dispersion) in place of a constant permittivity.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import lightsim as ls

LAM = 633e-9
N_PILLAR = 2.39
EPS_GLASS = 2.13
PERIOD = 400e-9
HEIGHT = 600e-9
R_MIN, R_MAX = 60e-9, 160e-9
LAM_POLE = 280e-9


def build(n, dx, dispersive, steps):
    lx, ly, lz = (v * dx for v in n)
    f0 = ls.C0 / LAM
    pml = ls.PML(12)
    z_glass = 1.0e-6                      # top of the substrate
    z_top = z_glass + HEIGHT              # top of the pillars
    margin = 24 * dx
    radius = 0.5 * min(lx, ly) - margin   # lens radius
    focal = min(0.75 * radius, lz - z_top - 2.0e-6)
    sim = ls.Simulation(
        min=[0, 0, 0],
        max=[lx, ly, lz],
        mesh=ls.UniformMesh(dx),
        run_time=12.0 * lz / ls.C0,          # upper bound; the shutoff level ends the run
        boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]},
        shutoff=None if steps else 1e-5,
    )
    sim.add_medium(ls.Medium("glass", eps=EPS_GLASS))
    if dispersive:
        w_pole = 2 * math.pi * ls.C0 / LAM_POLE
        d_eps = (N_PILLAR**2 - 1.0) * (1.0 - (LAM_POLE / LAM) ** 2)
        sim.add_medium(ls.Medium("pillar", eps=1.0, poles=[ls.Lorentz(d_eps, w_pole, 1e13)]))
    else:
        sim.add_medium(ls.Medium("pillar", eps=N_PILLAR**2))
    sim.add_structure(ls.Box([-1, -1, -1], [lx + 1, ly + 1, z_glass]), "glass")
    cx, cy = 0.5 * lx, 0.5 * ly
    m = int(radius // PERIOD)
    count = 0
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            x, y = i * PERIOD, j * PERIOD
            r = math.hypot(x, y)
            if r > radius - 0.5 * PERIOD:
                continue
            phase = (-2 * math.pi / LAM * (math.sqrt(r * r + focal * focal) - focal)) % (2 * math.pi)
            rad = R_MIN + (R_MAX - R_MIN) * phase / (2 * math.pi)
            sim.add_structure(ls.Cylinder([cx + x, cy + y, z_glass + 0.5 * HEIGHT], rad, HEIGHT, axis=2), "pillar")
            count += 1
    sim.add_source(ls.PlaneWave(2, 0.5e-6, "+", "Ex", ls.GaussianPulse(f0, f0 / 10), name="pw"))
    freqs = [0.95 * f0, f0, 1.05 * f0]
    z_focus = z_top + focal
    sim.add_monitor(ls.DFTMonitor("focal_plane", [0, 0, z_focus], [lx, ly, z_focus], freqs))
    sim.add_monitor(ls.DFTMonitor("cross_section", [0, cy, 0], [lx, cy, lz], freqs))
    sim.add_monitor(ls.FluxMonitor("above", 2, z_top + 0.5e-6, [0, 0, 0], [lx, ly, lz], freqs))
    if steps:
        dt0, _ = sim.time_info()
        sim.run_time = steps * dt0 * 0.999999
    return sim, count, z_top, focal, radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="400,400,300", help="cells along x, y, z")
    ap.add_argument("--dx", type=float, default=20e-9)
    ap.add_argument("--steps", type=int, default=None, help="fixed number of steps (throughput); default: run to the shutoff level")
    ap.add_argument("--dispersive", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    n = [int(v) for v in a.n.split(",")]
    t0 = time.perf_counter()
    sim, count, z_top, focal, radius = build(n, a.dx, a.dispersive, a.steps)
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = sim.run(device=a.device)
    wall = time.perf_counter() - t1
    out = {
        "solver": "lightsim",
        "scene": "metalens, dispersive pillars" if a.dispersive else "metalens",
        "grid_shape": n,
        "cells": int(np.prod(n)),
        "cell_size_nm": a.dx * 1e9,
        "pillars": count,
        "lens_radius_um": radius * 1e6,
        "focal_length_um": focal * 1e6,
        "time_steps": res.stats["steps"],
        "run_seconds": res.stats["seconds"],
        "wall_seconds_incl_setup": wall,
        "build_seconds": build_s,
        "mcups": res.stats["mcells_per_s"],
        "backend": res.stats["backend"],
        "stopped_early": bool(res.stats.get("stopped_early", False)),
    }
    if not a.steps:
        # where the lens focuses: the maximum of |Ex|^2 on the axis above the pillars
        ex = np.asarray(res.field("cross_section", "Ex"))   # [frequency, x, y, z]
        ex = ex.reshape(ex.shape[0], ex.shape[1], -1, ex.shape[-1])[:, :, 0, :]
        axis_line = np.abs(ex[1, ex.shape[1] // 2, :]) ** 2
        zs = np.linspace(0.0, n[2] * a.dx, axis_line.size)
        above = (zs > z_top + 0.5e-6) & (zs < n[2] * a.dx - 14 * a.dx)
        k = int(np.flatnonzero(above)[np.argmax(axis_line[above])])
        incident = float(np.median(axis_line[(zs > 0.55e-6) & (zs < 0.95e-6)])) * EPS_GLASS**0.5
        out["focus_z_above_lens_um"] = (zs[k] - z_top) * 1e6
        out["focus_intensity_over_incident"] = float(axis_line[k] / incident)
        # transmitted fraction of the incident plane wave through the plane above the lens
        _, spec = res.source_spectrum("pw", freqs=res.freqs("above"))
        incident = EPS_GLASS**0.5 * np.abs(spec) ** 2 * (n[0] * a.dx) * (n[1] * a.dx) / (2 * 376.730313668)
        out["transmission_above"] = [float(v) for v in np.ravel(np.asarray(res.flux("above"))) / incident]
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
