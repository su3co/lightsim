"""FDTDX throughput on the feature scenes of `lightsim bench --scene ...`.

Scenes (25 nm cells, centred Ez dipole, lambda = 1.5 um):

* ``lorentz`` - the whole domain is a Lorentz medium (eps_inf 2.25,
  delta_eps 1, omega0 = 2 omega, gamma = 1e13), 12-cell CPML on every face;
* ``bloch``   - Bloch boundaries along x and y (k = 0.3 k0, 0.2 k0), CPML in z;
* ``tensor``  - the whole domain is a uniaxial crystal (n_e^2 = 3.24,
  n_o^2 = 2.25) rotated 45 degrees about z (off-diagonal eps_xy), CPML on
  every face.

Grid sizes count all cells including the CPML, as in `lightsim bench`.

Usage: python fdtdx_scenes.py --scene lorentz --n 400,400,400 --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp

import fdtdx
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.fdtd.initialization import apply_params, place_objects

C0 = 299792458.0


def build(scene: str, n, steps: int):
    dx = 25e-9
    wl = 1.5e-6
    omega = 2 * math.pi * C0 / wl
    cfg = fdtdx.SimulationConfig(time=1e-12, grid=fdtdx.UniformGrid(spacing=dx), dtype=jnp.float32, courant_factor=0.99)
    total_time = steps * cfg.time_step_duration * 1.0000001
    cfg = fdtdx.SimulationConfig(
        time=total_time, grid=fdtdx.UniformGrid(spacing=dx), dtype=jnp.float32, courant_factor=0.99, gradient_config=None
    )
    if scene == "lorentz":
        mat = fdtdx.Material(
            permittivity=2.25,
            dispersion=fdtdx.DispersionModel(
                poles=(fdtdx.LorentzPole(resonance_frequency=2 * omega, damping=1e13, delta_epsilon=1.0),)
            ),
        )
    elif scene == "tensor":
        a, b = 0.5 * (3.24 + 2.25), 0.5 * (3.24 - 2.25)
        mat = fdtdx.Material(permittivity=(a, b, 0.0, b, a, 0.0, 0.0, 0.0, 2.25))
    else:
        mat = fdtdx.Material()
    volume = fdtdx.SimulationVolume(partial_real_shape=(n[0] * dx, n[1] * dx, n[2] * dx), material=mat)
    objects, constraints = [volume], []
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=12)
    if scene == "bloch":
        k0 = 2 * math.pi / wl
        for side in ("minx", "maxx", "miny", "maxy"):
            bcfg = bcfg.aset(f"boundary_type_{side}", "bloch")
        bcfg = bcfg.aset("bloch_vector", (0.3 * k0, 0.2 * k0, 0.0))
    bounds, c = fdtdx.boundary_objects_from_config(bcfg, volume)
    objects.extend(bounds.values())
    constraints.extend(c)
    src = fdtdx.PointDipoleSource(
        name="dipole", partial_grid_shape=(1, 1, 1), polarization=2, wave_character=fdtdx.WaveCharacter(wavelength=wl)
    )
    constraints.append(src.place_at_center(volume))
    objects.append(src)
    key = jax.random.PRNGKey(0)
    obj, arrays, params, cfg, _ = place_objects(objects, cfg, constraints, key)
    arrays, obj, _ = apply_params(arrays, obj, params, key)
    return obj, arrays, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=["lorentz", "bloch", "tensor"])
    ap.add_argument("--n", default="400,400,400")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    n = tuple(int(v) for v in a.n.split(","))
    t0 = time.perf_counter()
    obj, arrays, cfg = build(a.scene, n, a.steps)
    setup = time.perf_counter() - t0
    steps = min(cfg.time_steps_total, a.steps)
    kwargs = dict(
        arrays=arrays,
        objects=obj,
        config=cfg,
        key=jax.random.PRNGKey(0),
        reset_container=True,
        record_detectors=False,
        start_time=0,
        end_time=steps,
        show_progress=False,
    )
    static = ("reset_container", "record_detectors", "start_time", "end_time", "show_progress")
    t0 = time.perf_counter()
    compiled = jax.jit(custom_fdtd_forward, static_argnames=static).lower(**kwargs).compile()
    compile_s = time.perf_counter() - t0
    dyn = {k: v for k, v in kwargs.items() if k not in static}
    times = []
    out = None
    for _ in range(a.reps):
        t0 = time.perf_counter()
        out = compiled(**dyn)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    e = out[1].fields.E
    finite = bool(jnp.all(jnp.isfinite(e)))
    cells = n[0] * n[1] * n[2]
    res = {
        "solver": "fdtdx",
        "scene": a.scene,
        "device": jax.devices()[0].device_kind,
        "devices": jax.device_count(),
        "grid": list(n),
        "cells": cells,
        "steps": steps,
        "seconds": best,
        "all_seconds": times,
        "compile_seconds": compile_s,
        "setup_seconds": setup,
        "mcells_per_s": cells * steps / best / 1e6,
        "fields_finite": finite,
    }
    print(json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
