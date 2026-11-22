"""Meep adjoint solver (``meep.adjoint``) on the shared benchmark scenes.

    mpirun -np 64 python meep_bench.py --scene wg3d_6M
    mpirun -np 16 python meep_bench.py --scene mm2d_45nm --fd 4

Double precision, CPU, MPI. The design is a ``MaterialGrid`` whose samples
sit on the design nodes (Meep's material grid is pixel-centred, so the block
is one cell larger than the design rectangle), linear in permittivity, no
projection. Sub-pixel averaging of the fixed waveguides is Meep's default.

One gradient = N forward runs (one per input, all outputs and wavelengths
monitored) + N adjoint runs (one per input: every output launched backwards
at once, weighted by dL/dS of that column) + the overlap integrals. Each
forward run re-initialises the structure, as ``OptimizationProblem`` does at
every iteration; this is part of the time.

Simulated time: forward runs stop at the scene's run time. The stock adjoint
pulse of ``meep.adjoint`` is narrow-band (100 optical periods, 517 fs), so by
default the mode monitors here use the bandwidth of the forward pulse
instead (``--stock-pulse`` restores Meep's own). With several wavelengths
Meep's adjoint source is a filtered source of length 1 / (frequency spacing)
(805 fs for the three wavelengths); the adjoint run lasts that long plus the
ring-down time the forward run has after its pulse. (Stopping at the end of
the filtered source leaves the gradient 5 to 28 % off.)
"""

from __future__ import annotations

import argparse
import resource
import time

import autograd.numpy as anp
import numpy as np

import common
import problems

import meep as mp  # noqa: E402
import meep.adjoint as mpa  # noqa: E402

UM = 1e-6  # Meep length unit
TU = UM / problems.C0  # Meep time unit (s)
FWIDTH = 2 * np.pi * problems.PULSE_FWIDTH  # Meep's fwidth is 1 / sigma_t


class MatchedCoefficient(mpa.EigenmodeCoefficient):
    """Mode coefficient whose adjoint pulse has the bandwidth of the forward
    pulse (the stock adjoint pulse is 0.1 f wide in Meep's convention: 100
    periods long)."""

    def _create_time_profile(self, fwidth_frac=FWIDTH, adj_cutoff=problems.PULSE_OFFSET):
        return super()._create_time_profile(fwidth_frac=fwidth_frac, adj_cutoff=adj_cutoff)


def v3(x, y, z, ndim):
    return mp.Vector3(x / UM, y / UM, z / UM if ndim == 3 else 0)


class Column:
    """One input of the scene: a simulation, its monitors and its
    ``OptimizationProblem``; the objective is the linearised one,
    ``Re sum conj(g) S``, with ``g`` set after all forward runs."""

    def __init__(self, sc, n, stock_pulse, courant):
        self.sc, self.n = sc, n
        nd = sc.ndim
        self.fcen = UM / problems.LAM0
        freqs = [UM / w for w in sc.wavelengths]
        core = mp.Medium(epsilon=sc.eps_core)
        geometry = []
        for mn, mx in sc.waveguides:
            c = [0.5 * (a + b) for a, b in zip(mn, mx)]
            s = [b - a for a, b in zip(mn, mx)]
            geometry.append(mp.Block(center=v3(*c, nd), size=mp.Vector3(*[min(v / UM, 1e20) for v in s]) if nd == 3 else mp.Vector3(min(s[0] / UM, 1e20), s[1] / UM, mp.inf), material=core))
        nx, ny = sc.design_n
        self.mg = mp.MaterialGrid(mp.Vector3(nx + 1, ny + 1, 0), mp.Medium(epsilon=sc.eps_lo), mp.Medium(epsilon=sc.eps_hi), weights=np.zeros((nx + 1, ny + 1)), do_averaging=False)
        dsize = [(nx + 1) * sc.dx, (ny + 1) * sc.dx]
        dz = (sc.design_z[1] - sc.design_z[0]) if nd == 3 else 0.0
        dvol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(dsize[0] / UM, dsize[1] / UM, dz / UM))
        geometry.append(mp.Block(center=dvol.center, size=mp.Vector3(dsize[0] / UM, dsize[1] / UM, dz / UM if nd == 3 else mp.inf), material=self.mg))
        self.region = mpa.DesignRegion(self.mg, volume=dvol)
        parity = mp.ODD_Z if nd == 2 else mp.NO_PARITY  # 2-D: E out of plane

        def plane(p):
            (cy, cz), (hy, hz) = p.center, p.half
            return dict(center=v3(p.x, cy, cz, nd), size=mp.Vector3(0, 2 * hy / UM, 2 * hz / UM if nd == 3 else 0))

        pin = sc.inputs[n]
        src = mp.EigenModeSource(
            mp.GaussianSource(self.fcen, fwidth=FWIDTH * self.fcen, cutoff=problems.PULSE_OFFSET),
            eig_band=1,
            eig_parity=parity,
            eig_match_freq=True,
            direction=mp.X,
            **plane(pin),
        )
        self.sim = mp.Simulation(
            cell_size=v3(*sc.size, nd),
            resolution=UM / sc.dx,
            boundary_layers=[mp.PML(sc.pml / UM)],
            geometry=geometry,
            sources=[src],
            default_material=mp.Medium(epsilon=sc.eps_bg),
            Courant=courant,
            dimensions=nd,
        )
        coeff = mpa.EigenmodeCoefficient if stock_pulse else MatchedCoefficient

        def mon(p, shift):
            pl = plane(p)
            pl["center"] = pl["center"] + mp.Vector3(shift * sc.dx / UM)
            return coeff(self.sim, mp.Volume(**pl), mode=1, forward=True, eig_parity=parity)

        # input amplitude 4 cells after the source, outputs at their planes
        self.monitors = [mon(pin, 4)] + [mon(p, 0) for p in sc.outputs]
        self.g = np.zeros((len(sc.outputs), len(freqs)), dtype=complex)

        def J(a_in, *a_out):
            S = anp.stack([a / a_in for a in a_out])
            return anp.real(anp.sum(anp.conj(self.g) * S))

        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=[J],
            objective_arguments=self.monitors,
            design_regions=[self.region],
            frequencies=freqs,
            decay_by=0.0,  # never: the runs stop at the scene's simulated time
            maximum_run_time=sc.run_time / TU,
        )
        self.steps = {}
        self.fixed_time = None  # timing runs: every forward and adjoint run lasts this long (Meep units)

    def forward(self, u):
        self.opt.update_design([u.ravel()])
        self.opt.current_state = "INIT"
        self.opt.maximum_run_time = self.fixed_time or self.sc.run_time / TU
        self.opt.forward_run()
        self.steps["forward"] = int(self.sim.fields.t)
        a = [np.asarray(r) for r in self.opt.results_list]
        return np.stack([ao / a[0] for ao in a[1:]])  # [M, K]

    def full_steps(self):
        """Steps of the full forward and adjoint runs of the scene (what a
        gradient takes without ``fixed_time``)."""
        dt = self.sim.fields.dt
        fwd = self.sc.run_time / TU
        adj = fwd
        if len(self.sc.wavelengths) > 1:
            fr = np.sort([UM / w for w in self.sc.wavelengths])
            adj = float(np.max(1 / np.diff(fr))) + fwd - 2 * problems.PULSE_OFFSET / (FWIDTH * self.fcen)
        return int(np.ceil(fwd / dt)), int(np.ceil(adj / dt))

    def adjoint(self, g_col):
        self.g[...] = g_col
        if self.fixed_time:
            self.opt.maximum_run_time = self.fixed_time
        elif len(self.sc.wavelengths) > 1:
            # the multi-wavelength adjoint source is a filtered source of length
            # 1 / (frequency spacing); after it the fields get the ring-down
            # time the forward run has after its pulse
            fr = np.sort([UM / w for w in self.sc.wavelengths])
            pulse_end = 2 * problems.PULSE_OFFSET / (FWIDTH * self.fcen)
            self.opt.maximum_run_time = float(np.max(1 / np.diff(fr))) + self.sc.run_time / TU - pulse_end
        self.opt.adjoint_run()
        self.steps["adjoint"] = int(self.sim.fields.t)
        self.opt.calculate_gradient()
        grad = np.asarray(self.opt.gradient)
        if grad.ndim == 2:  # [parameters, wavelengths]
            grad = grad.sum(axis=1)
        nx, ny = self.sc.design_n
        return grad.reshape(nx + 1, ny + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="wg2d_45nm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=0, help="untimed gradient evaluations first (nothing is compiled or cached, so 0)")
    ap.add_argument("--fd", type=int, default=4)
    ap.add_argument("--fd-step", type=float, default=0.4)
    ap.add_argument("--direction", type=int, default=1, help="also check the derivative along a smooth direction")
    ap.add_argument("--no-forward", action="store_true", help="skip the separate forward-only timing (taken from the gradient run)")
    ap.add_argument("--stock-pulse", action="store_true", help="keep meep.adjoint's own adjoint pulse")
    ap.add_argument("--courant", type=float, default=None, help="default: the same time step as the GPU codes")
    ap.add_argument("--run-time", type=float, default=None)
    ap.add_argument("--timing-steps", type=int, default=None, help="timing only: stop every forward and adjoint run after this many steps; meep.adjoint never stops a run before its sources have ended, so the runs last at least that long (see meep_extrapolate.py)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--straight", action="store_true", help="only evaluate S for the base scene (waveguide through the design region)")
    a = ap.parse_args()
    mp.verbosity(0)
    sc = problems.make(a.scene)
    if a.run_time:
        sc.run_time = a.run_time
    courant = a.courant or problems.COURANT / np.sqrt(sc.ndim)
    nproc = mp.count_processors()
    span = sc.eps_hi - sc.eps_lo
    eps0 = sc.design_eps()
    u_of = lambda eps: np.clip((eps - sc.eps_lo) / span, 0.0, 1.0)  # noqa: E731

    t0 = time.perf_counter()
    cols = [Column(sc, n, a.stock_pulse, courant) for n in range(len(sc.inputs))]
    build_seconds = time.perf_counter() - t0
    if a.timing_steps:
        dt = courant * sc.dx / UM  # Meep's time step, in Meep units (resolution = UM / dx)
        for c in cols:
            c.fixed_time = a.timing_steps * dt
        a.fd, a.direction = 0, 0
    parts = {}

    def s_matrix(eps):
        return np.stack([c.forward(u_of(eps)) for c in cols], axis=1)  # [M, N, K]

    def value(eps):
        return sc.objective(s_matrix(eps))[0]

    def gradient(eps=eps0):
        t = time.perf_counter()
        S = s_matrix(eps)
        L, g = sc.objective(S)
        t1 = time.perf_counter()
        grad = sum(c.adjoint(g[:, n, :]) for n, c in enumerate(cols)) / span  # dL/d eps
        parts.update(forward=t1 - t, adjoint=time.perf_counter() - t1)
        return L, grad, S

    if a.straight:
        S = s_matrix(sc.straight_eps())
        if mp.am_master():
            common.save({"tool": "meep", "scene": sc.name, "check": "base scene as the design", "S_abs2": np.abs(S) ** 2}, f"straight_meep_{sc.name}", out=a.out)
            print(f"{sc.name} meep np={nproc} straight: |S|^2 = {np.abs(S).ravel() ** 2}", flush=True)
        return
    if a.warmup:
        common.timed(gradient, a.warmup)
    ts, (L, grad, S) = common.timed(gradient, a.reps)
    first = ts
    # forward only: timed alone, or the forward part of the last gradient
    tf = [parts["forward"]] if a.no_forward else common.timed(lambda: value(eps0), 1)[0]
    rep = {
        "tool": "meep",
        "version": mp.__version__,
        **sc.describe(),
        "variant": "adjoint" + ("_stock_pulse" if a.stock_pulse else ""),
        "method": "continuous adjoint (meep.adjoint): forward and adjoint FDTD runs, DFT fields in the design region",
        "hardware": f"{nproc} MPI processes",
        "cpu": common.cpu_name(),
        "precision": "double",
        "mpi_processes": nproc,
        "courant": courant,
        "steps": [cols[0].steps.get("forward"), cols[0].steps.get("adjoint")],
        "steps_forward": cols[0].steps.get("forward"),
        "steps_adjoint": cols[0].steps.get("adjoint"),
        "timing_only": bool(a.timing_steps),
        "full_steps": list(cols[0].full_steps()),
        "reps": a.reps,
        "seconds_per_gradient": min(ts),
        "seconds_all": ts,
        "first_call_seconds": first[0],
        "forward_only_seconds": min(tf),
        "seconds_forward_part": parts["forward"],
        "seconds_adjoint_part": parts["adjoint"],
        "included": "N forward + N adjoint runs (structure re-initialised in every forward run), mode coefficients, objective, gradient for every design sample",
        "excluded": "building the Simulation objects (scene_build_seconds)",
        "scene_build_seconds": build_seconds,
        "simulations": {"forward": len(cols), "adjoint": len(cols)},
        "forward_equivalents": min(ts) / (min(tf) / len(cols)),
        "objective": L,
        "S_abs2": np.abs(S) ** 2,
        "grad_l1": float(np.abs(grad).sum()),
        "grad_finite": bool(np.all(np.isfinite(grad))),
    }
    try:
        from mpi4py import MPI

        rep["peak_host_bytes"] = int(MPI.COMM_WORLD.allreduce(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, op=MPI.SUM)) * 1024
    except Exception:
        rep["peak_host_bytes"] = common.host_peak_bytes()
    if a.fd:
        rows = []
        for patch in sc.check_patches(a.fd):
            i0, i1, j0, j1 = patch
            ep, em = eps0.copy(), eps0.copy()
            ep[i0:i1, j0:j1] += a.fd_step
            em[i0:i1, j0:j1] -= a.fd_step
            rows.append((i0, j0, float(np.sum(grad[i0:i1, j0:j1])), (value(ep) - value(em)) / (2 * a.fd_step)))
        rep["fd"] = common.fd_summary(rows)
        rep["fd"].update(patch_samples=int(sc.extra.get("patch", 1)) ** 2, step=a.fd_step)
    if a.direction:
        d, h = sc.check_direction(), 0.5 * a.fd_step
        fd, ad = (value(eps0 + h * d) - value(eps0 - h * d)) / (2 * h), float(np.sum(grad * d))
        rep.setdefault("fd", {})["direction"] = {"gradient": ad, "finite_difference": fd, "rel": common.rel(ad, fd), "step": h}
    if mp.am_master():
        tag = a.tag or (f"{nproc}p_{a.timing_steps}steps" if a.timing_steps else f"{nproc}p")
        name = f"meep_{rep['variant']}_{sc.name}_{tag}" if not a.timing_steps else f"meeptiming_{rep['variant']}_{sc.name}_{tag}"
        common.save(rep, name, None if a.timing_steps else {"grad": grad, "S": S, "eps": eps0}, a.out)
        print(f"{sc.name} meep np={nproc}: {min(ts):.3f} s per gradient (forward {parts['forward']:.3f} + adjoint {parts['adjoint']:.3f}), steps {rep['steps']}, L = {L:.6g}, fd {rep.get('fd', {}).get('max_rel_to_largest')}", flush=True)


if __name__ == "__main__":
    main()
