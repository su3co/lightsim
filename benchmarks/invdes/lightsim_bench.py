"""lightsim: time per gradient on the shared benchmark scenes.

    python lightsim_bench.py --scene wg3d_6M --devices cuda:0 --engines python,native,reverse
    python lightsim_bench.py --scene mm3d_44nm --devices cuda:0+cuda:1 --engines native

Variants (``--engines``)
------------------------
``python``   ``lightsim.invdes.SMatrixProblem(engine="python")``: N forward and
             M adjoint simulations driven from Python, adjoint product in NumPy.
``native``   the same S-matrix gradient in the persistent Rust session
             (``engine="native"``).
``reverse``  reverse-mode differentiation through the time loop
             (``lightsim.autodiff.ReverseSession``) of a time-domain objective,
             the energy through the output plane. One source only.

With ``LIGHTSIM_FUSED=0`` in the environment the solver steps with its split
update kernels instead of the fused one; the results are then stored as
variants ``python_split``, ``native_split``, ``reverse_split``.

One "gradient" is everything from a design permittivity to dL/d(eps) for
every design sample: the simulations, the S-matrix, the objective and the
pull-back onto the samples. One-time set-up (mode solves, rasterisation,
device allocation) is reported separately.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common
import problems

import lightsim as ls  # noqa: E402
from lightsim import autodiff, invdes  # noqa: E402
from lightsim.spec import PERIODIC, PML, Box, Medium  # noqa: E402

Z_CELL = 1e-6  # thickness of the single periodic cell of a 2-D scene


def build(sc: problems.Scene):
    """Base simulation, ports and design grid of a scene."""
    pml = PML(layers=sc.npml, inside=True)
    lo, hi = sc.lo, sc.hi
    if sc.ndim == 2:
        sim = ls.Simulation(
            min=[lo[0], lo[1], 0.0],
            max=[hi[0], hi[1], Z_CELL],
            mesh={"type": "uniform", "dx": [sc.dx, sc.dx, Z_CELL]},
            run_time=sc.run_time,
            boundaries={"x": [pml, pml], "y": [pml, pml], "z": [PERIODIC, PERIODIC]},
            background="clad",
        )
    else:
        sim = ls.Simulation(min=list(lo), max=list(hi), mesh=ls.UniformMesh(sc.dx), run_time=sc.run_time, boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]}, background="clad")
    sim.add_medium(Medium("clad", eps=sc.eps_bg))
    sim.add_medium(Medium("core", eps=sc.eps_core))
    for mn, mx in sc.waveguides:
        sim.add_structure(Box(list(mn), list(mx)), "core")

    def port(p):
        (cy, cz), (hy, hz) = p.center, p.half
        if sc.ndim == 2:
            return invdes.port_mode(sim, p.name, 0, p.x, [p.x, cy - hy, 0.0], [p.x, cy + hy, Z_CELL], problems.F0, p.direction, "tm")
        return invdes.port_mode(sim, p.name, 0, p.x, [p.x, cy - hy, cz - hz], [p.x, cy + hy, cz + hz], problems.F0, p.direction)

    ins, outs = [port(p) for p in sc.inputs], [port(p) for p in sc.outputs]
    z0, z1 = (-1.0, 1.0) if sc.ndim == 2 else sc.design_z
    grid = invdes.DesignGrid([sc.design_min[0], sc.design_min[1], z0], [sc.design_max[0], sc.design_max[1], z1], (sc.design_n[0] + 1, sc.design_n[1] + 1), Medium("design", eps=sc.eps_hi))
    return sim, ins, outs, grid


def bump(eps, patch, delta):
    e = eps.copy()
    i0, i1, j0, j1 = patch
    e[i0:i1, j0:j1] += delta
    return e


def fd_checks(sc, value, eps, grad, step, npatch, direction=True):
    if not npatch:
        d, h = sc.check_direction(), 0.5 * step
        fd, ad = (value(eps + h * d) - value(eps - h * d)) / (2 * h), float(np.sum(grad * d))
        return {"direction": {"gradient": ad, "finite_difference": fd, "rel": common.rel(ad, fd), "step": h}}
    """Central differences of ``value(eps)`` on patches of samples and along
    a smooth direction, against the gradient ``grad``."""
    rows = []
    for patch in sc.check_patches(npatch):
        i0, i1, j0, j1 = patch
        fd = (value(bump(eps, patch, step)) - value(bump(eps, patch, -step))) / (2 * step)
        rows.append((i0, j0, float(np.sum(grad[i0:i1, j0:j1])), fd))
    out = common.fd_summary(rows)
    out["patch_samples"] = int(sc.extra.get("patch", 1)) ** 2
    out["step"] = step
    if direction:
        d = sc.check_direction()
        h = 0.5 * step
        lp, lm = value(eps + h * d), value(eps - h * d)
        fd, ad = (lp - lm) / (2 * h), float(np.sum(grad * d))
        out["direction"] = {"gradient": ad, "finite_difference": fd, "rel": common.rel(ad, fd), "step": h}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="wg2d_45nm")
    ap.add_argument("--devices", default="cuda:0", help="'cpu', 'cuda:0' or 'cuda:0+cuda:1'")
    ap.add_argument("--engines", default="python,native,reverse")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="untimed evaluations first (the first call of 'native' builds the session)")
    ap.add_argument("--fd", type=int, default=4, help="number of finite-difference patches (0: skip)")
    ap.add_argument("--fd-engines", default="python,native,reverse", help="variants that get the finite-difference check")
    ap.add_argument("--direction", type=int, default=1, help="also check the derivative along a smooth direction")
    ap.add_argument("--fd-step", type=float, default=0.4, help="permittivity step of the finite differences")
    ap.add_argument("--run-time", type=float, default=None, help="override the simulated time (convergence checks)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--straight", action="store_true", help="only evaluate the objective of the base scene (waveguide through the design region)")
    a = ap.parse_args()

    sc = problems.make(a.scene)
    if a.run_time:
        sc.run_time = a.run_time
    devices = a.devices.split("+")
    dev = devices if len(devices) > 1 else devices[0]
    gpus = common.gpu_names()
    hardware = "CPU" if devices[0] == "cpu" else f"{len(devices)}x {gpus[0] if gpus else 'GPU'}"
    eps0 = sc.design_eps()
    t0 = time.perf_counter()
    sim, ins, outs, grid = build(sc)
    build_seconds = time.perf_counter() - t0
    base = {
        "tool": "lightsim",
        "version": getattr(ls, "__version__", "0.1.0"),
        "build": os.environ.get("LIGHTSIM_BUILD", "the build before the 2-D mode"),
        **sc.describe(),
        "hardware": hardware,
        "cpu": common.cpu_name(),
        "precision": "single",
        "devices": devices,
        "reps": a.reps,
        "scene_build_seconds": build_seconds,
        "gpu_state_before": common.gpu_state(),
    }
    tag = a.tag or ("cpu" if devices[0] == "cpu" else f"{len(devices)}gpu")
    # LIGHTSIM_FUSED=0 selects the split update kernels instead of the fused one
    split = "_split" if os.environ.get("LIGHTSIM_FUSED") == "0" else ""
    base["kernels"] = "split (LIGHTSIM_FUSED=0)" if split else "default"
    engines = [e for e in a.engines.split(",") if e]

    if a.straight:
        prob = invdes.SMatrixProblem(sim, [grid], ins, outs, sc.freqs, devices=dev, fwidth=problems.PULSE_FWIDTH * problems.F0, engine="native")
        S = prob.solve([sc.straight_eps()], adjoint=False).S
        common.save({"tool": "lightsim", "scene": sc.name, "check": "base scene as the design", "S_abs2": np.abs(S) ** 2}, f"straight_lightsim_{sc.name}", out=a.out)
        print(f"{sc.name} lightsim straight: |S|^2 = {np.abs(S).ravel() ** 2}")
        return

    for engine in [e for e in engines if e in ("python", "native")]:
        with common.GpuMemory() as mem:
            prob = invdes.SMatrixProblem(sim, [grid], ins, outs, sc.freqs, devices=dev, fwidth=problems.PULSE_FWIDTH * problems.F0, engine=engine)

            def gradient(eps=eps0):
                sol = prob.solve([eps])
                L, g = sc.objective(sol.S)
                return L, sol.vjp(g)[0][:, :, 0], sol

            def value(eps):
                return sc.objective(prob.solve([eps], adjoint=False).S)[0]

            first, _ = common.timed(gradient, 1) if a.warmup else ([None], None)  # includes the session set-up of "native"
            ts, (L, grad, sol) = common.timed(gradient, a.reps)
            if not a.warmup:
                first = ts[:1]
            tf, _ = common.timed(lambda: value(eps0), max(1, min(a.reps, 2)))
        rep = dict(base)
        rep.update(
            variant=engine + split,
            method="continuous adjoint (frequency-domain product of forward and adjoint DFT fields)",
            seconds_per_gradient=min(ts),
            seconds_all=ts,
            first_call_seconds=first[0],
            forward_only_seconds=min(tf),
            included="N forward + M adjoint simulations, S-matrix, objective, gradient for every design sample",
            excluded="mode solves and scene build (scene_build_seconds); for 'native' the session set-up is in first_call_seconds" + ("" if a.warmup else " (no warm-up here: it is inside the first timing)"),
            warmup=a.warmup,
            simulations={"forward": len(ins), "adjoint": len(outs)},
            forward_equivalents=min(ts) / (min(tf) / len(ins)),
            sim_seconds=sol.stats["sim_seconds"],
            steps=[int(s) for s in sol.stats["steps"]],
            objective=L,
            S_abs2=np.abs(sol.S) ** 2,
            peak_device_bytes=mem.peak,
            peak_host_bytes=common.host_peak_bytes(),
            grad_l1=float(np.abs(grad).sum()),
            grad_finite=bool(np.all(np.isfinite(grad))),
        )
        if engine == "native":
            rep["session_setup_seconds"] = prob.native_session().setup_seconds
            rep["device_names"] = prob.native_session().devices
        if (a.fd or a.direction) and engine in a.fd_engines.split(","):
            rep["fd"] = fd_checks(sc, value, eps0, grad, a.fd_step, a.fd, bool(a.direction))
        rep["gpu_state_after"] = common.gpu_state()
        common.save(rep, f"lightsim_{engine}{split}_{sc.name}_{tag}", {"grad": grad, "S": sol.S, "eps": eps0}, a.out)
        print(f"{sc.name} {engine}: {min(ts):.3f} s per gradient (forward only {min(tf):.3f} s), L = {L:.6g}, fd {rep.get('fd', {}).get('max_rel_to_largest')}")
        del prob

    if "reverse" in engines:
        if len(ins) != 1:
            print("reverse mode: one source only, skipped")
            return
        p_in, p_out, pdef = ins[0], outs[0], sc.outputs[0]
        s1 = ls.Simulation(**{k: v for k, v in sim.__dict__.items()})
        s1.media, s1.structures, s1.sources, s1.monitors = list(sim.media), list(sim.structures), [], []
        s1.add_source(ls.ModeSource(0, p_in.position, "+", p_in.min, p_in.max, p_in.profiles[0], ls.GaussianPulse(problems.F0, problems.PULSE_FWIDTH * problems.F0), name="src"))
        obj = autodiff.flux_energy(0, pdef.x, p_out.min, p_out.max)
        with common.GpuMemory() as mem:
            t0 = time.perf_counter()
            sess = autodiff.ReverseSession(s1, grid, obj, device=devices[0])
            setup = time.perf_counter() - t0

            def gradient(eps=eps0):
                sess.set_design([eps])
                val = sess.forward()
                return val, sess.backward()[0][:, :, 0], sess.stats()

            def value(eps):
                sess.set_design([eps])
                return sess.forward()

            first, _ = common.timed(gradient, 1) if a.warmup else ([None], None)
            ts, (L, grad, st) = common.timed(gradient, a.reps)
            if not a.warmup:
                first = ts[:1]
            tf, _ = common.timed(lambda: value(eps0), max(1, min(a.reps, 2)))
        rep = dict(base)
        rep.update(
            variant="reverse" + split,
            hardware="CPU" if devices[0] == "cpu" else f"1x {gpus[0] if gpus else 'GPU'}",
            devices=devices[:1],
            method="discrete reverse mode through the time loop (forward fields rebuilt by time reversal)",
            seconds_per_gradient=min(ts),
            seconds_all=ts,
            first_call_seconds=first[0],
            forward_only_seconds=min(tf),
            included="one forward simulation and one reverse pass; gradient of the energy through the output plane for every design sample",
            excluded="mode solve, scene build and session set-up (session_setup_seconds)",
            session_setup_seconds=setup,
            simulations={"forward": 1, "reverse_pass": 1},
            forward_equivalents=min(ts) / min(tf),
            forward_seconds=st["forward_seconds"],
            reverse_seconds=st["reverse_seconds"],
            steps=[int(st["steps"])],
            objective=L,
            objective_name="energy through the output plane (J)",
            recorded_bytes=int((st["shell_floats_per_step"] + st["monitor_floats_per_step"]) * 4 * st["steps"]),
            peak_device_bytes=mem.peak,
            peak_host_bytes=common.host_peak_bytes(),
            grad_l1=float(np.abs(grad).sum()),
            grad_finite=bool(np.all(np.isfinite(grad))),
        )
        if (a.fd or a.direction) and "reverse" in a.fd_engines.split(","):
            rep["fd"] = fd_checks(sc, value, eps0, grad, a.fd_step, a.fd, bool(a.direction))
        rep["gpu_state_after"] = common.gpu_state()
        common.save(rep, f"lightsim_reverse{split}_{sc.name}_{tag if len(devices) == 1 else '1gpu'}", {"grad": grad, "eps": eps0}, a.out)
        print(f"{sc.name} reverse: {min(ts):.3f} s per gradient (forward {st['forward_seconds']:.3f} + reverse {st['reverse_seconds']:.3f}), L = {L:.6g}, fd {rep.get('fd', {}).get('max_rel_to_largest')}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
