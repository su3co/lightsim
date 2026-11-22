"""Ceviche (autograd FDFD, 2-D, CPU) on the shared 2-D benchmark scenes.

    python ceviche_bench.py --scene wg2d_45nm
    python ceviche_bench.py --scene mm2d_15nm --variant lu

Double precision, frequency domain: there is no time stepping, so "the same
simulated time" does not apply; the cost is sparse direct solves. E out of
plane (``fdfd_ez``), permittivity on the same nodes as the other tools.

Variants
--------
``stock``   ceviche as shipped: ``autograd.value_and_grad`` through
            ``fdfd_ez.solve``. Every solve factorises its matrix again, so a
            gradient costs 2 N K factorisations (N inputs, K wavelengths: one
            forward and one adjoint system each).
``lu``      ceviche's operators with a hand-written adjoint that factorises
            once per wavelength and reuses the factors for the N forward and
            the N transposed (adjoint) right-hand sides: K factorisations.
            This is what a direct frequency-domain solver can do; it is not a
            ceviche feature.

ceviche's phasors follow the opposite time convention to the time-domain
tools, so S is conjugated (it matters for the matrix objective, not for
|S|^2). Sources are line currents with the port mode profile (ceviche's own
mode solver). The launched amplitude is calibrated once, before the timed part, on
a straight waveguide through the whole domain; S = modal amplitude at the
output line / launched amplitude. The solver is MKL PARDISO through pyMKL
when it is installed (set MKL_NUM_THREADS), SciPy's SuperLU otherwise.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common
import problems

import autograd.numpy as npa  # noqa: E402
import ceviche  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
from autograd import value_and_grad  # noqa: E402
from ceviche import fdfd_ez  # noqa: E402
from ceviche import solvers as cv_solvers  # noqa: E402
from ceviche.constants import EPSILON_0  # noqa: E402
from ceviche.modes import get_modes  # noqa: E402


class Setup:
    def __init__(self, sc: problems.Scene):
        if sc.ndim != 2:
            raise SystemExit("ceviche is 2-D only")
        self.sc = sc
        self.base, (self.i0, self.j0) = sc.node_eps()
        self.nx, self.ny = self.base.shape
        self.omegas = 2 * np.pi * sc.freqs
        self.npml = [sc.npml, sc.npml]
        ix = lambda x: int(round((x - sc.lo[0]) / sc.dx))  # noqa: E731
        jy = lambda y: int(round((y - sc.lo[1]) / sc.dx))  # noqa: E731

        def line(p, shift=0):
            return ix(p.x) + shift, slice(jy(p.center[0] - p.half[0]), jy(p.center[0] + p.half[0]) + 1)

        self.src_lines = [line(p) for p in sc.inputs]
        self.in_lines = [line(p, 4) for p in sc.inputs]
        self.out_lines = [line(p) for p in sc.outputs]
        K = len(self.omegas)
        # mode profiles (per wavelength) on every line, and unit-amplitude projectors
        self.sources = [[None] * K for _ in sc.inputs]
        self.proj_out = [[None] * K for _ in sc.outputs]
        self.a_in = np.zeros((len(sc.inputs), K), dtype=complex)
        for k, om in enumerate(self.omegas):
            for m, (i, js) in enumerate(self.out_lines):
                e = self._mode(self.base[i, js], om)
                pr = np.zeros((self.nx, self.ny), dtype=complex)
                pr[i, js] = e / np.sum(e * e)
                self.proj_out[m][k] = pr
            for n, ((i, js), (ii, _)) in enumerate(zip(self.src_lines, self.in_lines)):
                e = self._mode(self.base[i, js], om)
                src = np.zeros((self.nx, self.ny), dtype=complex)
                src[i, js] = e
                self.sources[n][k] = src
                # calibration: the same source in a straight waveguide across the domain
                ref = np.tile(self.base[i, :][None, :], (self.nx, 1))
                ez = fdfd_ez(om, sc.dx, ref, self.npml).solve(src)[2]
                self.a_in[n, k] = np.sum(ez[ii, js] * e) / np.sum(e * e)

    def _mode(self, eps_line, omega):
        """Fundamental mode of a line, real, positive at its maximum, unit
        norm (the eigenvector comes back with an arbitrary complex phase)."""
        _, vec = get_modes(eps_line, omega, self.sc.dx, 0, m=1, filtering=False)
        e = np.atleast_2d(vec)[:, 0]
        e = np.real(e * np.exp(-1j * np.angle(e[np.argmax(np.abs(e))])))
        return e / np.sqrt(np.sum(e * e))

    def full_eps(self, design, xp=np):
        """Insert the design samples into the node permittivity."""
        i0, j0 = self.i0, self.j0
        i1, j1 = i0 + design.shape[0], j0 + design.shape[1]
        b = self.base
        mid = xp.concatenate([b[i0:i1, :j0], design, b[i0:i1, j1:]], axis=1)
        return xp.concatenate([b[:i0, :], mid, b[i1:, :]], axis=0)

    # -- stock ceviche -------------------------------------------------
    def s_matrix_ag(self, design):
        eps = self.full_eps(design, npa)
        cols = []
        for n in range(len(self.sc.inputs)):
            per_k = []
            for k, om in enumerate(self.omegas):
                ez = fdfd_ez(om, self.sc.dx, eps, self.npml).solve(self.sources[n][k])[2]
                per_k.append(npa.stack([npa.sum(ez * self.proj_out[m][k]) for m in range(len(self.sc.outputs))]) / self.a_in[n, k])
            cols.append(npa.stack(per_k, axis=1))  # [M, K]
        # ceviche's phasors use the opposite time convention: conjugate, so S
        # has the phases of the time-domain tools (exp(-i w t))
        return npa.conj(npa.stack(cols, axis=1))  # [M, N, K]

    def objective_ag(self, S):
        sc = self.sc
        if sc.target is None:
            return npa.mean(npa.abs(S) ** 2)
        t2 = float(np.sum(np.abs(sc.target) ** 2))
        c = npa.sum(npa.conj(sc.target)[:, :, None] * S, axis=(0, 1))
        return npa.sum(npa.abs(c) ** 2) / (S.shape[2] * t2**2)

    # -- one factorisation per wavelength ------------------------------
    def _matrix(self, om, eps):
        sim = fdfd_ez(om, self.sc.dx, eps, self.npml)
        entries, indices = sim._make_A(eps.flatten())
        n = eps.size
        return sp.csr_matrix((entries, indices), shape=(n, n)), sim

    def gradient_lu(self, design, need_grad=True):
        sc, M, N, K = self.sc, len(self.sc.outputs), len(self.sc.inputs), len(self.omegas)
        eps = self.full_eps(design)
        facs, xs = [], [[None] * K for _ in range(N)]
        S = np.zeros((M, N, K), dtype=complex)
        for k, om in enumerate(self.omegas):
            A, _ = self._matrix(om, eps)
            f = Factor(A)
            facs.append(f)
            for n in range(N):
                x = f.solve(1j * om * self.sources[n][k].flatten())
                xs[n][k] = x
                for m in range(M):
                    S[m, n, k] = np.conj(np.sum(x * self.proj_out[m][k].flatten()) / self.a_in[n, k])  # exp(-i w t) convention
        L, g = sc.objective(S)
        if not need_grad:
            for f in facs:
                f.clear()
            return L, None, S
        grad = np.zeros(eps.size)
        for k, om in enumerate(self.omegas):
            for n in range(N):
                # dL = Re(conj(g) dS) and S is the conjugate of the solver's amplitude a: dL = Re(g da)
                w = sum(g[m, n, k] * self.proj_out[m][k].flatten() for m in range(M)) / self.a_in[n, k]
                lam = facs[k].solve(w, transpose=True)
                grad += np.real(EPSILON_0 * om**2 * lam * xs[n][k])  # dA/d eps = -eps0 w^2 on the diagonal
            facs[k].clear()
        grad = grad.reshape(eps.shape)[self.i0 : self.i0 + design.shape[0], self.j0 : self.j0 + design.shape[1]]
        return L, grad, S


class Factor:
    """Sparse LU kept for several right-hand sides (PARDISO if available)."""

    def __init__(self, A):
        self.mkl = cv_solvers.HAS_MKL
        if self.mkl:
            from pyMKL import pardisoSolver

            self.At = None
            self.A = A
            self.p = pardisoSolver(A, mtype=13)
            self.p.factor()
        else:
            self.lu = spl.splu(A.tocsc())

    def solve(self, b, transpose=False):
        if self.mkl:
            self.p.iparm[11] = 2 if transpose else 0  # solve the transposed system with the same factors
            x = self.p.solve(np.asarray(b, dtype=complex))
            self.p.iparm[11] = 0
            return x.reshape(-1)
        return self.lu.solve(np.asarray(b, dtype=complex), trans="T" if transpose else "N")

    def clear(self):
        if self.mkl:
            self.p.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="wg2d_45nm")
    ap.add_argument("--variant", default="stock", choices=["stock", "lu"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fd", type=int, default=4)
    ap.add_argument("--fd-step", type=float, default=0.4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sc = problems.make(a.scene)
    t0 = time.perf_counter()
    st = Setup(sc)
    setup_seconds = time.perf_counter() - t0
    eps0 = sc.design_eps()
    M, N, K = sc.shape_mnk

    if a.variant == "stock":
        vg = value_and_grad(lambda d: st.objective_ag(st.s_matrix_ag(d)))

        def gradient(eps=eps0):
            L, g = vg(eps)
            return float(L), np.asarray(g, dtype=float)

        def value(eps):
            return float(st.objective_ag(st.s_matrix_ag(eps)))

        S = np.asarray(st.s_matrix_ag(eps0))
        sims = {"forward_factorisations": N * K, "adjoint_factorisations": N * K}
        method = "frequency domain, autograd adjoint; every solve factorises again"
    else:

        def gradient(eps=eps0):
            L, g, _ = st.gradient_lu(eps)
            return L, g

        def value(eps):
            return st.gradient_lu(eps, need_grad=False)[0]

        S = st.gradient_lu(eps0, need_grad=False)[2]
        sims = {"factorisations": K, "forward_substitutions": N * K, "adjoint_substitutions": N * K}
        method = "frequency domain, hand-written adjoint on ceviche's operators; one factorisation per wavelength"

    first, _ = common.timed(gradient, 1)
    ts, (L, grad) = common.timed(gradient, a.reps)
    tf, _ = common.timed(lambda: value(eps0), max(1, min(a.reps, 2)))
    threads = os.environ.get("MKL_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or str(os.cpu_count())
    rep = {
        "tool": "ceviche",
        "version": ceviche.__version__,
        **sc.describe(),
        "steps": None,
        "variant": a.variant,
        "method": method,
        "solver": "MKL PARDISO (pyMKL)" if cv_solvers.HAS_MKL else "SciPy SuperLU",
        "hardware": f"CPU, {threads} threads",
        "threads": int(threads),
        "cpu": common.cpu_name(),
        "precision": "double",
        "reps": a.reps,
        "seconds_per_gradient": min(ts),
        "seconds_all": ts,
        "first_call_seconds": first[0],
        "forward_only_seconds": min(tf),
        "included": "matrix assembly, factorisations and solves for every input and wavelength, objective, gradient for every design sample",
        "excluded": "mode profiles and the calibration solves on straight waveguides (setup_seconds)",
        "setup_seconds": setup_seconds,
        "simulations": sims,
        "forward_equivalents": min(ts) / (min(tf) / (N * K)) if a.variant == "stock" else None,
        "objective": L,
        "S_abs2": np.abs(S) ** 2,
        "peak_host_bytes": common.host_peak_bytes(),
        "grad_l1": float(np.abs(grad).sum()),
        "grad_finite": bool(np.all(np.isfinite(grad))),
    }
    if a.fd:
        rows = []
        for i0, i1, j0, j1 in sc.check_patches(a.fd):
            ep, em = eps0.copy(), eps0.copy()
            ep[i0:i1, j0:j1] += a.fd_step
            em[i0:i1, j0:j1] -= a.fd_step
            rows.append((i0, j0, float(np.sum(grad[i0:i1, j0:j1])), (value(ep) - value(em)) / (2 * a.fd_step)))
        rep["fd"] = common.fd_summary(rows)
        rep["fd"].update(patch_samples=int(sc.extra.get("patch", 1)) ** 2, step=a.fd_step)
        d, h = sc.check_direction(), 0.5 * a.fd_step
        fd, ad = (value(eps0 + h * d) - value(eps0 - h * d)) / (2 * h), float(np.sum(grad * d))
        rep["fd"]["direction"] = {"gradient": ad, "finite_difference": fd, "rel": common.rel(ad, fd), "step": h}
    common.save(rep, f"ceviche_{a.variant}_{sc.name}_{a.tag or threads + 't'}", {"grad": grad, "S": S, "eps": eps0}, a.out)
    print(f"{sc.name} ceviche ({a.variant}, {threads} threads): {min(ts):.3f} s per gradient (forward only {min(tf):.3f} s), L = {L:.6g}, fd {rep.get('fd', {}).get('max_rel_to_largest')}", flush=True)


if __name__ == "__main__":
    main()
