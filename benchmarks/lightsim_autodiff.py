"""Time per gradient: Python path, native session, time-domain reverse mode.

Scenes
------
``gradient``  the scene of lightsim_gradient.py / fdtdx_gradient.py: straight
              400 x 400 nm silicon waveguides in air through a
              1.6 x 1.6 x 0.4 um design region, 20 nm grid, 10-cell CPML inside
              the volume. The design region is an index grid with one sample
              per Yee cell (81 x 81 x 21).
``planar``    a small 2.5-D problem (two inputs, two outputs, three
              frequencies, about 2e4 cells): per-iteration overhead dominates.

Paths (one "gradient" = everything needed for dL/d(eps) of one design)
----------------------------------------------------------------------
``adjoint``   lightsim.adjoint.mode_power_gradient (the published number of
              lightsim_gradient.py: uniform design block, gradient per Yee cell)
``python``    invdes.SMatrixProblem(engine="python"): solve + vjp of L = sum |S|^2
``native``    the same through engine="native" (persistent Rust session)
``reverse``   lightsim.autodiff reverse mode: energy through the output plane

Usage:
  python lightsim_autodiff.py --scene gradient --length 8e-6 --width 4e-6 --height 1.5e-6 --time 100e-15 --device cuda:0 --json out.json
  python lightsim_autodiff.py --scene planar --device cuda:0 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import lightsim as ls
from lightsim import autodiff, eim, invdes
from lightsim.adjoint import mode_power_gradient
from lightsim.spec import PML, Box, Medium

C0 = 299792458.0
UM = 1e-6


def gradient_scene(length, width, height, sim_time):
    dx = 20e-9
    f0 = C0 / 1.55e-6
    pml = PML(layers=10, inside=True)
    sim = ls.Simulation(
        min=[-length / 2, -width / 2, -height / 2],
        max=[length / 2, width / 2, height / 2],
        mesh=ls.UniformMesh(dx),
        run_time=sim_time,
        boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]},
    )
    sim.add_medium(ls.Medium("si", eps=12.25))
    sim.add_structure(ls.Box([-length, -200e-9, -200e-9], [length, 200e-9, 200e-9]), "si")
    dmin, dmax = [-0.8e-6, -0.8e-6, -200e-9], [0.8e-6, 0.8e-6, 200e-9]
    xs = -length / 2 + 14 * dx
    xo = length / 2 - 15 * dx
    bmin, bmax = [xs, -width / 2 + 12 * dx, -height / 2 + 12 * dx], [xs, width / 2 - 12 * dx, height / 2 - 12 * dx]
    mode = ls.solve_modes(sim, 0, xs, bmin, bmax, f0, num_modes=1)[0]
    p_in = invdes.Port("in", 0, xs, bmin, bmax, [mode.profile], "+")
    p_out = invdes.Port("out", 0, xo, [xo, bmin[1], bmin[2]], [xo, bmax[1], bmax[2]], [mode.profile], "-")
    grid = invdes.DesignGrid(dmin, dmax, (81, 81, 21), Medium("design", eps=12.25))
    return sim, mode, p_in, p_out, grid, f0, (dmin, dmax)


def planar_scene():
    lam = 1.55 * UM
    f0 = C0 / lam
    n_si = eim.slab_neff(3.476, 220e-9, lam, 1.444)
    n_bg = 1.444
    sim = invdes.planar_simulation([0, -1.9 * UM], [6.4 * UM, 1.9 * UM], lam, n_bg, n_si, steps_per_wavelength=14, run_time=500e-15, shutoff=1e-7)
    sim.add_medium(Medium("core", n=n_si))
    for y in (-0.8 * UM, 0.8 * UM):
        sim.add_structure(Box([-5 * UM, y - 0.25 * UM, -1], [12 * UM, y + 0.25 * UM, 1]), "core")
    freqs = C0 / (np.array([1.53, 1.55, 1.57]) * UM)

    def port(name, x, y, d):
        return invdes.port_mode(sim, name, 0, x, [x, y - 0.75 * UM, 0], [x, y + 0.75 * UM, 1 * UM], f0, d, "te")

    ports = [port("i1", 1 * UM, -0.8 * UM, "+"), port("i2", 1 * UM, 0.8 * UM, "+"), port("o1", 5.4 * UM, -0.8 * UM, "-"), port("o2", 5.4 * UM, 0.8 * UM, "-")]
    des = invdes.Design.effective_index([2 * UM, -1.4 * UM], [4.4 * UM, 1.4 * UM], n_bg, n_si, pixel=50e-9, filter_radius=120e-9)
    return sim, ports, des, freqs


def timed(fn, reps):
    """(min, mean, last result) of `reps` calls."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.mean(ts)), out


def designs(shape, lo, hi, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.uniform(lo, hi, shape) for _ in range(n)]


def smatrix_paths(make_problem, eps_list, reps, rep, devices):
    """solve + vjp(L = sum |S|^2) through both engines; fills `rep`."""
    out = {}
    for engine in ("python", "native"):
        prob = make_problem(engine)
        it = iter(eps_list * 4)

        def one():
            sol = prob.solve(next(it))
            return sol, sol.vjp(2 * sol.S)

        tfirst, _, _ = timed(one, 1)  # includes the session set-up for "native"
        tmin, tmean, (sol, g) = timed(one, reps)
        out[engine] = (sol, g)
        rep[f"{engine}_first_seconds"] = tfirst
        rep[f"{engine}_gradient_seconds"] = tmin
        rep[f"{engine}_gradient_seconds_mean"] = tmean
        rep[f"{engine}_sim_seconds"] = sol.stats["sim_seconds"]
        rep[f"{engine}_steps"] = [int(s) for s in sol.stats["steps"]]
        if engine == "native":
            rep["native_setup_seconds"] = prob.native_session().setup_seconds
            rep["native_devices"] = prob.native_session().devices
    # same design through both engines: the gradients agree
    ep = eps_list[0]
    sp = make_problem("python").solve(ep)
    sn = out["native"][0]._problem.solve(ep)
    gp, gn = sp.vjp(2 * sp.S), sn.vjp(2 * sn.S)
    rep["S_max_rel_diff"] = float(np.abs(sn.S - sp.S).max() / np.abs(sp.S).max())
    rep["grad_max_rel_diff"] = float(max(np.abs(a - b).max() for a, b in zip(gn, gp)) / max(np.abs(b).max() for b in gp))
    rep["fom"] = float(np.sum(np.abs(sp.S) ** 2))
    rep["devices"] = devices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gradient", choices=["gradient", "planar"])
    ap.add_argument("--length", type=float, default=8e-6)
    ap.add_argument("--width", type=float, default=4e-6)
    ap.add_argument("--height", type=float, default=1.5e-6)
    ap.add_argument("--time", type=float, default=100e-15)
    ap.add_argument("--device", default="cuda:0", help="device, or a comma-free list separated by '+' (cuda:0+cuda:1)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma-separated paths to skip (adjoint, smatrix, reverse)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    devices = a.device.split("+")
    dev0 = devices[0]
    skip = set(a.skip.split(","))
    rep = {"solver": "lightsim", "scene": a.scene, "reps": a.reps}

    if a.scene == "planar":
        sim, ports, des, freqs = planar_scene()
        rhos = [des.random(seed=s, mean=0.5, amplitude=0.4) for s in range(4)]
        eps_list = [des.eps(r, 4.0) for r in rhos]
        g = sim.grid()
        rep["cells"] = int((len(g[0]) - 1) * (len(g[1]) - 1) * (len(g[2]) - 1))
        rep["shape_MNK"] = [2, 2, 3]
        smatrix_paths(lambda engine: invdes.SMatrixProblem(sim, des.grids(), ports[:2], ports[2:], freqs, devices=devices if len(devices) > 1 else dev0, engine=engine), eps_list, a.reps, rep, devices)
    else:
        t0 = time.perf_counter()
        sim, mode, p_in, p_out, grid, f0, (dmin, dmax) = gradient_scene(a.length, a.width, a.height, a.time)
        rep["scene_seconds"] = time.perf_counter() - t0
        g = sim.grid()
        rep["cells"] = int((len(g[0]) - 1) * (len(g[1]) - 1) * (len(g[2]) - 1))
        rep["design_samples"] = int(np.prod(grid.dims))
        eps_list = designs(grid.dims, 2.0, 12.0, 4)
        sig = ls.GaussianPulse(f0, 0.1 * f0)

        # ---- published path: uniform design block, gradient per Yee cell ----
        if "adjoint" not in skip:
            s0 = ls.Simulation(**{k: v for k, v in sim.__dict__.items()})
            s0.media, s0.structures, s0.sources, s0.monitors = list(sim.media), list(sim.structures), [], []
            s0.add_medium(ls.Medium("design0", eps=0.5 * (1 + 12.25)))
            s0.add_structure(ls.Box(dmin, dmax), "design0")
            s0.add_source(ls.ModeSource(0, p_in.position, "+", p_in.min, p_in.max, mode.profile, sig, name="src"))
            out = {"axis": 0, "position": p_out.position, "min": p_out.min, "max": p_out.max, "profile": mode.profile, "direction": "+"}
            tf, _, r = timed(lambda: s0.run(device=dev0), a.reps)
            tg, _, _ = timed(lambda: mode_power_gradient(s0, dmin, dmax, out, f0, device=dev0), a.reps)
            rep.update(forward_seconds=tf, adjoint_gradient_seconds=tg, steps=int(r.stats["steps"]), backend=r.stats["backend"])

        # ---- S-matrix: python engine and native session ----
        if "smatrix" not in skip:
            smatrix_paths(lambda engine: invdes.SMatrixProblem(sim, [grid], [p_in], [p_out], [f0], devices=devices if len(devices) > 1 else dev0, fwidth=0.1 * f0, engine=engine), eps_list, a.reps, rep, devices)

        # ---- reverse mode: energy through the output waveguide cross-section ----
        if "reverse" not in skip:
            s1 = ls.Simulation(**{k: v for k, v in sim.__dict__.items()})
            s1.media, s1.structures, s1.sources, s1.monitors = list(sim.media), list(sim.structures), [], []
            s1.add_source(ls.ModeSource(0, p_in.position, "+", p_in.min, p_in.max, mode.profile, sig, name="src"))
            xo = p_out.position
            obj = autodiff.flux_energy(0, xo, [xo, -0.6e-6, -0.6e-6], [xo, 0.6e-6, 0.6e-6])
            t0 = time.perf_counter()
            sess = autodiff.ReverseSession(s1, grid, obj, device=dev0)
            rep["reverse_setup_seconds"] = time.perf_counter() - t0
            it = iter(eps_list * 4)

            def one():
                sess.set_design(next(it))
                val = sess.forward()
                return val, sess.backward()[0], sess.stats()

            timed(one, 1)
            tmin, tmean, (val, gr, st) = timed(one, a.reps)
            rep.update(
                reverse_gradient_seconds=tmin,
                reverse_gradient_seconds_mean=tmean,
                reverse_forward_seconds=st["forward_seconds"],
                reverse_backward_seconds=st["reverse_seconds"],
                reverse_steps=int(st["steps"]),
                reverse_device=st["device"],
                reverse_shell_bytes=int(st["shell_floats_per_step"] * 4 * st["steps"]),
                reverse_monitor_bytes=int(st["monitor_floats_per_step"] * 4 * st["steps"]),
                reverse_value=val,
                reverse_grad_l1=float(np.abs(gr).sum()),
                reverse_finite=bool(np.all(np.isfinite(gr))),
            )
            # finite difference of the reverse-mode gradient along sign(grad),
            # with a step that changes L by about 2 %: a random direction in
            # 1e5 dimensions is nearly orthogonal to the gradient and its
            # difference quotient drowns in single-precision noise
            e0 = eps_list[0]
            sess.set_design(e0)
            l0 = sess.forward()
            g0 = sess.backward()[0]
            d = np.sign(g0)
            h = float(min(0.05, 0.02 * abs(l0) / np.abs(g0).sum()))
            sess.set_design(e0 + h * d)
            lp = sess.forward()
            sess.set_design(e0 - h * d)
            lm = sess.forward()
            fd, ad = (lp - lm) / (2 * h), float(np.sum(g0 * d))
            rep.update(reverse_fd=fd, reverse_adjoint=ad, reverse_fd_rel=abs(ad - fd) / abs(fd), reverse_fd_step=h, reverse_fd_dL_over_L=abs(lp - lm) / abs(l0))

    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
