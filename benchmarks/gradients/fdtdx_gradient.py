"""FDTDX inverse-design gradient benchmark (after examples/optimize_ceviche_corner.py).

Straight silicon waveguides (400 x 400 nm) in air feed a 1.6 x 1.6 x 0.4 um
design region (FDTDX ``Device``, 20 nm voxels, Gaussian smoothing + subpixel
projection). A mode source launches the TE mode; Poynting-flux detectors
measure input and output power, and loss = -P_out / P_in. The gradient with
respect to the design parameters comes from ``jax.value_and_grad`` through
``fdtdx.run_fdtd`` (FDTDX's reversible time-stepping with the recorder settings
of its own example). 20 nm grid, f32.

Usage: python fdtdx_gradient.py --length 8e-6 --width 4e-6 --time 50e-15
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import fdtdx


def build(length, width, height, sim_time, recorder=True):
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)
    config = fdtdx.SimulationConfig(time=sim_time, grid=fdtdx.UniformGrid(spacing=20e-9), dtype=jnp.float32, courant_factor=0.99)
    period_steps = round(period / config.time_step_duration)
    all_steps = list(range(config.time_steps_total))
    if recorder:
        gc = fdtdx.GradientConfig(
            recorder=fdtdx.Recorder(modules=[fdtdx.LinearReconstructEveryK(k=5), fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz)])
        )
        config = config.aset("gradient_config", gc)
    objs, cons = [], []
    volume = fdtdx.SimulationVolume(partial_real_shape=(length, width, height))
    objs.append(volume)
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10)
    bd, c = fdtdx.boundary_objects_from_config(bcfg, volume)
    cons.extend(c)
    objs.extend(bd.values())
    mats = {"Air": fdtdx.Material(permittivity=1.0), "Silicon": fdtdx.Material(permittivity=12.25)}
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(1.6e-6, 1.6e-6, 400e-9),
        materials=mats,
        param_transforms=[fdtdx.GaussianSmoothing2D(std_discrete=3), fdtdx.SubpixelSmoothedProjection()],
        partial_voxel_real_shape=(20e-9, 20e-9, 400e-9),
    )
    cons.append(device.place_at_center(volume))
    objs.append(device)
    wg_in = fdtdx.UniformMaterialObject(partial_real_shape=(None, 400e-9, 400e-9), material=mats["Silicon"])
    cons.extend([wg_in.place_at_center(device, axes=(1, 2)), wg_in.extend_to(device, axis=0, direction="+")])
    objs.append(wg_in)
    wg_out = fdtdx.UniformMaterialObject(partial_real_shape=(None, 400e-9, 400e-9), material=mats["Silicon"])
    cons.extend([wg_out.place_at_center(device, axes=(1, 2)), wg_out.extend_to(device, axis=0, direction="-")])
    objs.append(wg_out)
    src = fdtdx.ModePlaneSource(partial_grid_shape=(1, None, None), wave_character=fdtdx.WaveCharacter(wavelength=wavelength), direction="+")
    cons.append(src.place_relative_to(volume, axes=(0,), other_positions=(-1,), own_positions=(-1,), grid_margins=(bcfg.thickness_grid_minx + 4,)))
    objs.append(src)
    det_in = fdtdx.PoyntingFluxDetector(
        name="in flux", partial_grid_shape=(1, None, None), direction="+", switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_steps[-period_steps:])
    )
    cons.append(det_in.place_relative_to(src, axes=0, own_positions=1, other_positions=1, grid_margins=2))
    objs.append(det_in)
    det_out = fdtdx.PoyntingFluxDetector(
        name="out flux", partial_grid_shape=(1, None, None), direction="+", switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_steps[-period_steps:])
    )
    cons.append(det_out.place_relative_to(volume, axes=0, own_positions=1, other_positions=1, grid_margins=-bcfg.thickness_grid_maxx - 5))
    objs.append(det_out)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=key)
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key, beta=1.0)
    return objects, arrays, params, config, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=float, default=8e-6)
    ap.add_argument("--width", type=float, default=4e-6)
    ap.add_argument("--height", type=float, default=1.5e-6)
    ap.add_argument("--time", type=float, default=50e-15)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    objects, arrays, params, config, key = build(a.length, a.width, a.height, a.time)
    setup = time.perf_counter() - t0

    def loss(params, arrays, key):
        arrays, new_objects, _ = fdtdx.apply_params(arrays, objects, params, key, beta=1.0)
        _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=new_objects, config=config, key=key)
        p_out = arrays.detector_states["out flux"]["poynting_flux"].sum()
        p_in = arrays.detector_states["in flux"]["poynting_flux"].sum()
        return -p_out / p_in

    t0 = time.perf_counter()
    fwd = jax.jit(loss).lower(params, arrays, key).compile()
    grad = jax.jit(jax.value_and_grad(loss)).lower(params, arrays, key).compile()
    compile_s = time.perf_counter() - t0
    tf, tg = [], []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        v = jax.block_until_ready(fwd(params, arrays, key))
        tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        val, g = jax.block_until_ready(grad(params, arrays, key))
        tg.append(time.perf_counter() - t0)
    gl = jax.tree_util.tree_leaves(g)
    gnorm = float(sum(jnp.sum(jnp.abs(x)) for x in gl))
    shape = [int(s) for s in arrays.fields.E.shape[1:]] if hasattr(arrays, "fields") else None
    res = {
        "solver": "fdtdx",
        "device": jax.devices()[0].device_kind,
        "grid": shape,
        "cells": int(shape[0] * shape[1] * shape[2]) if shape else None,
        "steps": int(config.time_steps_total),
        "forward_seconds": min(tf),
        "gradient_seconds": min(tg),
        "compile_seconds": compile_s,
        "setup_seconds": setup,
        "loss": float(val),
        "grad_l1": gnorm,
        "finite": bool(all(bool(jnp.all(jnp.isfinite(x))) for x in gl)),
    }
    print(json.dumps(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
