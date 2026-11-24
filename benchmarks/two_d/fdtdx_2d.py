"""FDTDX throughput on 2-D grids, matching `lightsim bench --n NU,NV`.

FDTDX has no 2-D solver: a 2-D problem is a 3-D grid one cell thick along z
with periodic z boundaries, on which all six field components are updated.
This script times exactly that: NX x NY x 1 cells (25 nm), 12-cell CPML on
the x and y faces (inside the box, as FDTDX places boundaries), periodic
along z, one point dipole in the centre. The forward simulation is
jit-compiled once for exactly `--steps` steps and then timed over `--reps`
executions (compile time reported separately).

Usage: python fdtdx_2d.py --n 1024,1024 --steps 1000 [--polarization 2] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.fdtd.initialization import apply_params, place_objects


def build(n: tuple[int, int], steps: int, polarization: int):
    dx = 25e-9
    cfg = fdtdx.SimulationConfig(time=1e-12, grid=fdtdx.UniformGrid(spacing=dx), dtype=jnp.float32, courant_factor=0.99)
    total_time = steps * cfg.time_step_duration * 1.0000001
    cfg = fdtdx.SimulationConfig(time=total_time, grid=fdtdx.UniformGrid(spacing=dx), dtype=jnp.float32, courant_factor=0.99, gradient_config=None)
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(n[0] * dx, n[1] * dx, dx))
    objects.append(volume)
    bc = fdtdx.BoundaryConfig.from_uniform_bound(thickness=12, override_types={"min_z": "periodic", "max_z": "periodic"})
    bounds, c = fdtdx.boundary_objects_from_config(bc, volume)
    objects.extend(bounds.values())
    constraints.extend(c)
    src = fdtdx.PointDipoleSource(name="dipole", partial_grid_shape=(1, 1, 1), polarization=polarization, wave_character=fdtdx.WaveCharacter(wavelength=1.5e-6))
    constraints.append(src.place_at_center(volume))
    objects.append(src)
    key = jax.random.PRNGKey(0)
    obj, arrays, params, cfg, _ = place_objects(objects, cfg, constraints, key)
    arrays, obj, _ = apply_params(arrays, obj, params, key)
    return obj, arrays, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1024,1024")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--polarization", type=int, default=2, help="dipole axis: 2 (E along z: the TM set) or 0 / 1 (E in the plane: the TE set)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    n = tuple(int(v) for v in a.n.split(","))
    t0 = time.perf_counter()
    obj, arrays, cfg = build(n, a.steps, a.polarization)
    setup = time.perf_counter() - t0
    steps = min(cfg.time_steps_total, a.steps)
    kwargs = dict(arrays=arrays, objects=obj, config=cfg, key=jax.random.PRNGKey(0), reset_container=True, record_detectors=False, start_time=0, end_time=steps, show_progress=False)
    static = ("reset_container", "record_detectors", "start_time", "end_time", "show_progress")
    t0 = time.perf_counter()
    compiled = jax.jit(custom_fdtd_forward, static_argnames=static).lower(**kwargs).compile()
    compile_s = time.perf_counter() - t0
    dyn = {k: v for k, v in kwargs.items() if k not in static}
    times = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        out = compiled(**dyn)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    e = out[1].fields.E
    peak = float(jax.jit(lambda x: jnp.max(jnp.abs(x)))(e))
    cells = n[0] * n[1]
    res = {
        "solver": "fdtdx",
        "device": jax.devices()[0].device_kind,
        "grid": [n[0], n[1], 1],
        "cells": cells,
        "steps": steps,
        "seconds": best,
        "all_seconds": times,
        "compile_seconds": compile_s,
        "setup_seconds": setup,
        "mcells_per_s": cells * steps / best / 1e6,
        "max_abs_e": peak,
    }
    print(json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
