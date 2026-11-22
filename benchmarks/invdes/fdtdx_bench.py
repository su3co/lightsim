"""FDTDX (JAX, GPU) on the shared benchmark scenes: time-reversible autodiff.

    CUDA_VISIBLE_DEVICES=0 python fdtdx_bench.py --scene wg3d_6M
    CUDA_VISIBLE_DEVICES=0 python fdtdx_bench.py --scene mm3d_44nm --recorder exact

Single precision, one GPU (pin it with CUDA_VISIBLE_DEVICES; JAX otherwise
takes every visible GPU). The gradient is ``jax.value_and_grad`` through
``fdtdx.run_fdtd`` with ``GradientConfig(method="reversible")``: a forward
run that records the fields at the absorbing-layer interfaces, then a
backward run that rebuilds the forward fields by time reversal while the
adjoint is stepped. With N inputs this is done N times inside one compiled
function.

Scene mapping
-------------
* FDTDX materials are per cell, so the design enters as one value per cell,
  the mean of the four surrounding node samples (the bilinear interpolant at
  the cell centre); the gradient is taken with respect to the node samples
  through that map. The permittivity is written straight into
  ``inv_permittivities`` (what ``Device`` does for a continuous design).
* Mode source with FDTDX's Gaussian pulse of the same spectral width (its
  peak is at 6 sigma_t instead of 5), mode-overlap detectors at the scene's
  wavelengths; S = overlap(output) / overlap(2 cells after the source).
* The mode profiles of sources and detectors are computed once, before the
  timed part (they do not depend on the design).

``--recorder``: ``example`` is the recorder of FDTDX's own optimisation
examples (every 5th interface sample, 8-bit floats); ``fp16`` the same with
16-bit floats, ``fp16k10`` every 10th step in 16-bit floats (for 86 M cells,
where ``fp16`` does not fit) (added here because the 8-bit recorder fails the
finite-difference check and the exact one does not fit in GPU memory above
6 M cells); ``exact`` records every step in single precision.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common
import problems

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fdtdx  # noqa: E402


def build(sc: problems.Scene, recorder: str):
    lo, hi = sc.lo, sc.hi
    if recorder == "example":
        rec = fdtdx.Recorder(modules=[fdtdx.LinearReconstructEveryK(k=5), fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz)])
    elif recorder == "fp16":
        rec = fdtdx.Recorder(modules=[fdtdx.LinearReconstructEveryK(k=5), fdtdx.DtypeConversion(dtype=jnp.float16)])
    elif recorder == "fp16k10":
        rec = fdtdx.Recorder(modules=[fdtdx.LinearReconstructEveryK(k=10), fdtdx.DtypeConversion(dtype=jnp.float16)])
    elif recorder == "exact":
        rec = fdtdx.Recorder(modules=[])
    else:
        raise ValueError(recorder)
    config = fdtdx.SimulationConfig(time=sc.run_time, grid=fdtdx.UniformGrid(spacing=sc.dx), dtype=jnp.float32, courant_factor=problems.COURANT, gradient_config=fdtdx.GradientConfig(recorder=rec))
    objs, cons = [], []
    size = sc.size if sc.ndim == 3 else (sc.size[0], sc.size[1], sc.dx)
    lo3 = lo if sc.ndim == 3 else (lo[0], lo[1], -0.5 * sc.dx)
    hi3 = hi if sc.ndim == 3 else (hi[0], hi[1], 0.5 * sc.dx)
    volume = fdtdx.SimulationVolume(partial_real_shape=size, material=fdtdx.Material(permittivity=sc.eps_bg), name="volume")
    objs.append(volume)
    over = {} if sc.ndim == 3 else {"min_z": "periodic", "max_z": "periodic"}
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=sc.npml, override_types=over)
    bd, c = fdtdx.boundary_objects_from_config(bcfg, volume)
    objs.extend(bd.values())
    cons.extend(c)

    def at(obj, center):
        return obj.place_relative_to(volume, axes=(0, 1, 2), own_positions=(0.0, 0.0, 0.0), other_positions=(-1.0, -1.0, -1.0), margins=tuple(float(c - l) for c, l in zip(center, lo3)))

    def snap(v, l, how):
        k = (v - l) / sc.dx
        k = np.floor(k + 1e-6) if how == "floor" else np.ceil(k - 1e-6) if how == "ceil" else np.round(k)
        return float(l + k * sc.dx)

    core = fdtdx.Material(permittivity=sc.eps_core)
    for i, (mn, mx) in enumerate(sc.waveguides):
        # clipped to the domain and to whole cells (FDTDX rasterises per cell);
        # a stub that ends half a cell inside the design ends at its edge
        a = [snap(max(mn[k], lo3[k]), lo3[k], "ceil" if k == 0 else "round") for k in range(3)]
        b = [snap(min(mx[k], hi3[k]), lo3[k], "floor" if k == 0 else "round") for k in range(3)]
        wg = fdtdx.UniformMaterialObject(partial_real_shape=tuple(q - p for p, q in zip(a, b)), material=core, name=f"wg{i}")
        objs.append(wg)
        cons.append(at(wg, [0.5 * (p + q) for p, q in zip(a, b)]))

    nx, ny = sc.design_n
    z0, z1 = sc.design_z if sc.ndim == 3 else (lo3[2], hi3[2])
    dz = z1 - z0
    device = fdtdx.Device(
        name="design",
        partial_real_shape=(nx * sc.dx, ny * sc.dx, dz),
        materials={"lo": fdtdx.Material(permittivity=sc.eps_lo), "hi": fdtdx.Material(permittivity=sc.eps_hi)},
        param_transforms=[],
        partial_voxel_real_shape=(sc.dx, sc.dx, dz),
    )
    objs.append(device)
    cons.append(at(device, (0.0, 0.0, 0.5 * (z0 + z1))))

    wave0 = fdtdx.WaveCharacter(wavelength=problems.LAM0)
    profile = fdtdx.GaussianPulseProfile(center_wave=wave0, spectral_width=fdtdx.WaveCharacter(frequency=problems.PULSE_FWIDTH * problems.F0))
    waves = tuple(fdtdx.WaveCharacter(wavelength=w) for w in sc.wavelengths)

    def plane_cells(p):
        (cy, cz), (hy, hz) = p.center, p.half
        ncy = int(round(2 * hy / sc.dx))
        ncz = int(round(2 * min(hz, 0.5 * size[2]) / sc.dx)) if sc.ndim == 3 else 1
        return (1, ncy, ncz), (cy, cz if sc.ndim == 3 else 0.0)

    pol = None if sc.ndim == 3 else "tm"
    names_in, names_norm, names_out = [], [], []
    for p in sc.inputs:
        shape, (cy, cz) = plane_cells(p)
        src = fdtdx.ModePlaneSource(partial_grid_shape=shape, wave_character=wave0, temporal_profile=profile, direction="+", mode_index=0, filter_pol=pol, name=f"src_{p.name}")
        objs.append(src)
        cons.append(at(src, (p.x + 0.5 * sc.dx, cy, cz)))
        det = fdtdx.ModeOverlapDetector(partial_grid_shape=shape, wave_characters=waves, direction="+", mode_index=0, filter_pol=pol, name=f"norm_{p.name}")
        objs.append(det)
        cons.append(at(det, (p.x + 2.5 * sc.dx, cy, cz)))
        names_in.append(src.name)
        names_norm.append(det.name)
    for p in sc.outputs:
        shape, (cy, cz) = plane_cells(p)
        det = fdtdx.ModeOverlapDetector(partial_grid_shape=shape, wave_characters=waves, direction="+", mode_index=0, filter_pol=pol, name=f"out_{p.name}")
        objs.append(det)
        cons.append(at(det, (p.x + 0.5 * sc.dx, cy, cz)))
        names_out.append(det.name)
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=key)
    return objects, arrays, params, config, key, (names_in, names_norm, names_out)


def objective_jax(sc, S):
    """The scene's objective on a JAX array S[M, N, K]."""
    if sc.target is None:
        return jnp.mean(jnp.abs(S) ** 2)
    T = jnp.asarray(sc.target, dtype=S.dtype)
    t2 = float(np.sum(np.abs(sc.target) ** 2))
    c = jnp.einsum("mn,mnk->k", jnp.conj(T), S)
    return jnp.sum(jnp.abs(c) ** 2) / (S.shape[2] * t2**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="wg3d_6M")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--recorder", default="example", choices=["example", "fp16", "fp16k10", "exact"])
    ap.add_argument("--fd", type=int, default=4)
    ap.add_argument("--fd-step", type=float, default=0.4)
    ap.add_argument("--direction", type=int, default=1, help="also check the derivative along a smooth direction")
    ap.add_argument("--warmup", type=int, default=1, help="untimed calls of the compiled gradient before the timed ones")
    ap.add_argument("--run-time", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--straight", action="store_true", help="only evaluate S for the base scene (waveguide through the design region)")
    a = ap.parse_args()
    sc = problems.make(a.scene)
    if a.run_time:
        sc.run_time = a.run_time
    gpus = common.gpu_names()
    state_before = common.gpu_state()
    t0 = time.perf_counter()
    objects, arrays, params, config, key, (names_in, names_norm, names_out) = build(sc, a.recorder)
    place_seconds = time.perf_counter() - t0
    grid_shape = [int(s) for s in arrays.inv_permittivities.shape[1:]]

    # One object container per input: the other sources are switched off
    # first, then apply_params computes the mode profiles of every source and
    # detector and the on/off schedule of the sources (once, before the timed
    # part; neither depends on the design).
    t0 = time.perf_counter()
    p0 = {k: (jnp.zeros_like(v) if hasattr(v, "shape") else {kk: jnp.zeros_like(vv) for kk, vv in v.items()}) for k, v in params.items()}
    per_input, arrays0 = [], None
    for name in names_in:
        o = objects
        for other in names_in:
            if other != name:
                o = o.aset(f"object_list->[{o.index(other)}]->switch->is_always_off", True)
        arr, o, _ = fdtdx.apply_params(arrays, o, p0, key)
        per_input.append(o)
        arrays0 = arr if arrays0 is None else arrays0
    jax.block_until_ready(arrays0.inv_permittivities)
    apply_seconds = time.perf_counter() - t0
    device = per_input[0]["design"]
    sl = device.grid_slice
    inv0 = arrays0.inv_permittivities
    block_shape = inv0[:, *sl].shape
    config_fwd = config.aset("gradient_config", None)

    def s_matrix(eps, cfg):
        ev = 0.25 * (eps[:-1, :-1] + eps[1:, :-1] + eps[:-1, 1:] + eps[1:, 1:])
        block = jnp.broadcast_to((1.0 / ev)[None, :, :, None], block_shape).astype(inv0.dtype)
        arr = arrays0.at["inv_permittivities"].set(inv0.at[:, *sl].set(block))
        cols = []
        for n, o in enumerate(per_input):
            _, out = fdtdx.run_fdtd(arrays=arr, objects=o, config=cfg, key=key, show_progress=False)
            ov = lambda nm: o[nm].compute_overlap(out.detector_states[nm])  # noqa: E731
            a_in = ov(names_norm[n])
            cols.append(jnp.stack([ov(nm) for nm in names_out]) / a_in)
        return jnp.stack(cols, axis=1)  # [M, N, K]

    def loss(eps, cfg):
        S = s_matrix(eps, cfg)
        return objective_jax(sc, S), S

    eps0 = jnp.asarray(sc.design_eps(), dtype=jnp.float32)
    if a.straight:
        S = np.asarray(jax.jit(lambda e: s_matrix(e, config_fwd))(jnp.asarray(sc.straight_eps(), dtype=jnp.float32)))
        common.save({"tool": "fdtdx", "scene": sc.name, "check": "base scene as the design", "S_abs2": np.abs(S) ** 2}, f"straight_fdtdx_{sc.name}", out=a.out)
        print(f"{sc.name} fdtdx straight: |S|^2 = {np.abs(S).ravel() ** 2}")
        return
    t0 = time.perf_counter()
    fwd = jax.jit(lambda e: loss(e, config_fwd)).lower(eps0).compile()
    compile_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    vg = jax.jit(jax.value_and_grad(lambda e: loss(e, config), has_aux=True)).lower(eps0).compile()
    compile_grad = time.perf_counter() - t0

    def gradient(eps=eps0):
        (L, S), g = vg(eps)
        return float(L), np.asarray(jax.block_until_ready(g), dtype=float), np.asarray(S)

    def value(eps):
        return float(jax.block_until_ready(fwd(jnp.asarray(eps, dtype=jnp.float32))[0]))

    first, _ = common.timed(gradient, 1) if a.warmup else ([None], None)
    ts, (L, grad, S) = common.timed(gradient, a.reps)
    tf, Lf = common.timed(lambda: value(eps0), max(1, min(a.reps, 2)))
    try:
        peak = int(jax.devices()[0].memory_stats()["peak_bytes_in_use"])
    except Exception:
        peak = None
    N = len(names_in)
    rep = {
        "tool": "fdtdx",
        "version": __import__("importlib.metadata").metadata.version("fdtdx"),
        "jax": jax.__version__,
        **sc.describe(),
        "fdtdx_grid": grid_shape,
        "steps": [int(config.time_steps_total)],
        "variant": f"reversible_{a.recorder}",
        "method": "reverse-mode autodiff through the time loop (time-reversible, interface recorder: %s)" % a.recorder,
        "hardware": f"1x {gpus[0] if gpus else 'GPU'}",
        "jax_devices": [str(d) for d in jax.devices()],
        "cpu": common.cpu_name(),
        "precision": "single",
        "reps": a.reps,
        "seconds_per_gradient": min(ts),
        "seconds_all": ts,
        "first_call_seconds": first[0],
        "compile_seconds": compile_grad,
        "compile_forward_seconds": compile_fwd,
        "forward_only_seconds": min(tf),
        "included": "N forward runs with interface recording + N backward runs, mode overlaps, objective, gradient for every design sample (steady state: compiled function called again)",
        "excluded": "XLA compilation (compile_seconds), object placement (place_seconds), mode solves of sources and detectors (apply_seconds)",
        "place_seconds": place_seconds,
        "apply_seconds": apply_seconds,
        "simulations": {"forward": N, "reverse_pass": N},
        "forward_equivalents": min(ts) / (min(tf) / N),
        "objective": L,
        "objective_forward_only": Lf,
        "S_abs2": np.abs(S) ** 2,
        "peak_device_bytes": peak,
        "peak_host_bytes": common.host_peak_bytes(),
        "grad_l1": float(np.abs(grad).sum()),
        "grad_finite": bool(np.all(np.isfinite(grad))),
        "gpu_state_before": state_before,
    }
    e0 = np.asarray(sc.design_eps(), dtype=float)
    if a.fd:
        rows = []
        for i0, i1, j0, j1 in sc.check_patches(a.fd):
            ep, em = e0.copy(), e0.copy()
            ep[i0:i1, j0:j1] += a.fd_step
            em[i0:i1, j0:j1] -= a.fd_step
            rows.append((i0, j0, float(np.sum(grad[i0:i1, j0:j1])), (value(ep) - value(em)) / (2 * a.fd_step)))
        rep["fd"] = common.fd_summary(rows)
        rep["fd"].update(patch_samples=int(sc.extra.get("patch", 1)) ** 2, step=a.fd_step)
    if a.direction:
        d, h = sc.check_direction(), 0.5 * a.fd_step
        fd, ad = (value(e0 + h * d) - value(e0 - h * d)) / (2 * h), float(np.sum(grad * d))
        rep.setdefault("fd", {})["direction"] = {"gradient": ad, "finite_difference": fd, "rel": common.rel(ad, fd), "step": h}
    rep["gpu_state_after"] = common.gpu_state()
    common.save(rep, f"fdtdx_{a.recorder}_{sc.name}_{a.tag or '1gpu'}", {"grad": grad, "S": S, "eps": e0}, a.out)
    print(f"{sc.name} fdtdx ({a.recorder}): {min(ts):.3f} s per gradient (forward only {min(tf):.3f} s, compile {compile_grad:.1f} s), L = {L:.6g}, fd {rep.get('fd', {}).get('max_rel_to_largest')}", flush=True)


if __name__ == "__main__":
    main()
