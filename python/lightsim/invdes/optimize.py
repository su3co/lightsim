"""Optimisation driver: :class:`InverseDesign` (problem + parametrisation +
objective) and :func:`optimize` (Adam or L-BFGS-B with beta continuation,
history and npz checkpoint / resume)."""

from __future__ import annotations

import os
import time

import numpy as np

__all__ = ["InverseDesign", "ProbeDesign", "optimize", "beta_schedule"]


class InverseDesign:
    """Ties an :class:`SMatrixProblem` to a :class:`Design`.

    ``S`` has the axes (outputs, inputs, frequencies), as in
    :class:`SMatrixProblem`: ``S[m, n, k]``, shape ``[M, N, K]``.

    ``forward(rho, beta)`` returns ``(S, pullback)`` where ``pullback(g)`` is
    dL/d(rho) for the upstream gradient ``g = dL/dRe S + i dL/dIm S``: the
    S-matrix as a differentiable layer in the density.

    ``objective(S) -> (value, g)`` (see :mod:`lightsim.invdes.objectives`) is
    used by :meth:`value_and_grad`. ``robust`` = threshold offset d adds the
    eroded (eta + d) and dilated (eta - d) variants; the objective is then the
    mean over the three (three times the simulations).
    """

    def __init__(self, problem, design, objective=None, robust=None):
        self.problem = problem
        self.design = design
        self.objective = objective
        self.robust = robust
        self.last = None

    def s_matrix(self, rho, beta, eta=None):
        """S only (N forward runs)."""
        return self.problem.solve(self.design.eps(rho, beta, eta), adjoint=False).S

    def forward(self, rho, beta, eta=None):
        """S-matrix and its pullback for the density ``rho`` at projection
        strength ``beta`` (threshold ``eta``, default that of the design).

        Returns ``(S, pullback)``: ``S`` is the complex array ``[M, N, K]``
        (N forward and M adjoint runs are made), and ``pullback(g)`` maps the
        upstream gradient ``g = dL/dRe S + i dL/dIm S`` of a real loss, an
        array of the same shape, to ``dL/d(rho)`` with the shape of ``rho``:
        through the S-matrix and through the filter and projection of the
        design. This is the pair an autodiff framework needs.
        """
        rho = np.asarray(rho, dtype=float)
        sol = self.problem.solve(self.design.eps(rho, beta, eta), adjoint=True)
        self.last = sol

        def pullback(g):
            return self.design.vjp(rho, beta, sol.vjp(g), eta)

        return sol.S, pullback

    def value_and_grad(self, rho, beta):
        """(L, dL/drho, info) for the stored objective."""
        etas = [None]
        if self.robust:
            etas = [None, self.design.eta + self.robust, self.design.eta - self.robust]
        val, grad, Ss, wall, sims = 0.0, 0.0, [], 0.0, 0
        for eta in etas:
            S, pb = self.forward(rho, beta, eta)
            v, g = self.objective(S)
            val += v / len(etas)
            grad = grad + pb(g) / len(etas)
            Ss.append(S)
            wall += self.last.stats["wall"]
            sims += len(self.last.stats["runs"])
        info = {"S": Ss[0], "S_variants": Ss, "wall": wall, "simulations": sims, "stats": self.last.stats}
        return val, grad, info



class ProbeDesign:
    """Stochastic gradient of :func:`~lightsim.invdes.objectives.matrix_distance`
    from random probe vectors, for devices with many ports.

    Each call of :meth:`value_and_grad` draws ``probes`` random complex input
    vectors ``x`` (``E[x x^H] = I``), runs one forward simulation per vector
    with all inputs excited at once (:meth:`SMatrixProblem.solve_combined`)
    and one adjoint simulation per vector and frequency with the outputs
    driven by the residual ``r = S x - e^{i phi} T x``. The mean of
    ``|r|^2 / ||T||^2`` is an unbiased estimate of the distance and its
    gradient an unbiased estimate of the distance's gradient (the global
    phase ``phi`` is estimated from the same probes). A gradient costs
    ``probes * (1 + K)`` simulations whatever the number of ports N, against
    ``2 N`` for the exact :class:`InverseDesign`; the price is noise, which
    Adam averages over iterations. Use with ``optimize(..., maximize=False)``.
    """

    def __init__(self, problem, design, target, probes=2, seed=0, weights=None):
        self.problem = problem
        self.design = design
        self.target = np.asarray(target, dtype=complex)
        self.probes = int(probes)
        self.rng = np.random.default_rng(seed)
        self.weights = weights
        self.last = None

    def value_and_grad(self, rho, beta):
        """(estimated distance, estimated gradient, info)."""
        from .objectives import probe_vectors

        rho = np.asarray(rho, dtype=float)
        T = self.target
        M, N = T.shape
        sol = self.problem.solve_combined(self.design.eps(rho, beta), probe_vectors(N, self.probes, self.rng))
        X, Y = sol.X, sol.Y  # [P, N, K], [M, P, K]
        K = Y.shape[2]
        w = np.full(K, 1.0 / K) if self.weights is None else np.asarray(self.weights, dtype=float) / np.sum(self.weights)
        t2 = float(np.sum(np.abs(T) ** 2))
        TX = np.einsum("mn,pnk->mpk", T, X)
        ph = np.exp(1j * np.angle(np.einsum("mpk,mpk->k", np.conj(TX), Y)))
        R = Y - ph[None, None, :] * TX
        P = Y.shape[1]
        val = float(np.sum(w * np.sum(np.abs(R) ** 2, axis=(0, 1))) / (P * t2))
        g = 2.0 * w[None, None, :] * R / (P * t2)
        grads = sol.vjp(g)
        self.last = sol
        runs = sol.stats["runs"] + sol.adjoint_stats["runs"]
        info = {"wall": sol.stats["wall"] + sol.adjoint_stats["wall"], "simulations": len(runs), "runs": runs, "Y": Y, "X": X, "g_mismatch": float(np.linalg.norm(sol.g_eff - g) / max(np.linalg.norm(g), 1e-300))}
        return val, self.design.vjp(rho, beta, grads), info


def directional_derivative(f, x, direction, h=0.01):
    """Finite-difference derivative of ``f`` at ``x`` along ``direction``,
    for checking a gradient: ``np.sum(gradient * direction)`` should match it.

    Central differences with steps ``h`` and ``2 h`` combined by Richardson
    extrapolation, ``(4 D(h) - D(2 h)) / 3``, whose error is of order h**4
    instead of h**2. A gradient of the whole chain (filter, projection,
    S-matrix, objective) needs it: the projection can make the objective
    strongly curved in the density, and a plain central difference with a
    step large enough to rise above the single-precision noise of the runs
    (two runs of one design differ in the last bits) is then off by its
    h**2 term. It costs four evaluations of ``f``. Keep ``h`` well above the
    noise: about 1e-2 of the density range in single precision, 1e-3 in
    the double-precision build.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(direction, dtype=float)

    def central(step):
        return (f(x + step * d) - f(x - step * d)) / (2 * step)

    return (4 * central(h) - central(2 * h)) / 3


def beta_schedule(start=1.0, factor=2.0, every=20, maximum=64.0):
    """beta(iteration) = min(maximum, start * factor ** (iteration // every))."""

    def f(it):
        return float(min(maximum, start * factor ** (it // every)))

    return f


def optimize(
    fn,
    rho0,
    iterations,
    beta=None,
    method="adam",
    lr=0.02,
    checkpoint=None,
    callback=None,
    maximize=True,
    betas=(0.9, 0.999),
    verbose=True,
    save_every=1,
    reset_on_beta=True,
):
    """Optimise a density in [0, 1].

    ``fn`` is an :class:`InverseDesign` (its ``value_and_grad`` is used) or a
    callable ``fn(rho, beta) -> (value, grad, info)``. ``beta`` is a number or
    a callable of the iteration (see :func:`beta_schedule`).

    ``method="adam"`` (projected Adam; the moments restart when beta changes
    unless ``reset_on_beta=False``: a restart makes the next step ``lr`` on
    every pixel, which can throw a narrow-band design off its resonances)
    or ``"lbfgs"`` (scipy L-BFGS-B, restarted at every beta stage; an
    "iteration" is then one function evaluation).

    ``checkpoint`` (npz path): the density, optimiser state and history are
    saved every ``save_every`` iterations and the run resumes from the file
    if it exists. History entries: ``iteration, fom, beta, binarization,
    seconds`` (wall time of the iteration) plus what ``info`` provides
    (``sim_seconds``: wall time of the forward + adjoint runs).

    Returns ``(rho, history)`` with ``history`` a dict of arrays.
    """
    vg = fn.value_and_grad if hasattr(fn, "value_and_grad") else fn
    design = getattr(fn, "design", None)
    bfun = beta if callable(beta) else (lambda it: float(beta if beta is not None else 8.0))
    rho = np.clip(np.array(rho0, dtype=float), 0, 1)
    m = np.zeros_like(rho)
    v = np.zeros_like(rho)
    t_adam = 0
    hist = {k: [] for k in ("iteration", "fom", "beta", "binarization", "seconds", "sim_seconds")}
    best = (-np.inf, rho.copy())
    start = 0
    if checkpoint and os.path.exists(checkpoint):
        ck = np.load(checkpoint)
        rho, m, v, t_adam = ck["rho"], ck["m"], ck["v"], int(ck["t_adam"])
        for k in hist:
            hist[k] = list(ck["hist_" + k])
        start = len(hist["iteration"])
        best = (float(ck["best_fom"]), ck["best_rho"])
        if verbose:
            print(f"resumed from {checkpoint} at iteration {start}", flush=True)
    sign = 1.0 if maximize else -1.0

    def save():
        if checkpoint:
            tmp = checkpoint + ".tmp.npz"
            np.savez(tmp, rho=rho, m=m, v=v, t_adam=t_adam, best_fom=best[0], best_rho=best[1], **{"hist_" + k: np.asarray(x) for k, x in hist.items()})
            os.replace(tmp, checkpoint)

    def record(it, val, b, t0, info, r):
        hist["iteration"].append(it)
        hist["fom"].append(float(val))
        hist["beta"].append(b)
        hist["binarization"].append(design.binarization(r, b) if design is not None else np.nan)
        hist["seconds"].append(time.time() - t0)
        hist["sim_seconds"].append(float(info.get("wall", np.nan)) if isinstance(info, dict) else np.nan)
        if verbose:
            print(f"it {it:4d}  fom {val:+.5f}  beta {b:5.1f}  bin {hist['binarization'][-1]:.3f}  {hist['seconds'][-1]:.1f} s", flush=True)
        if callback is not None:
            callback(it, r, val, b, info)

    if method == "adam":
        b_prev = bfun(start - 1) if start > 0 else None
        for it in range(start, iterations):
            t0 = time.time()
            b = bfun(it)
            if reset_on_beta and b_prev is not None and b != b_prev:
                m[:], v[:], t_adam = 0.0, 0.0, 0
            b_prev = b
            val, grad, info = vg(rho, b)
            if sign * val > sign * best[0] or it == start and start == 0:
                best = (float(val), rho.copy())
            record(it, val, b, t0, info, rho)
            g = sign * np.asarray(grad)
            t_adam += 1
            m = betas[0] * m + (1 - betas[0]) * g
            v = betas[1] * v + (1 - betas[1]) * g * g
            step = lr * (m / (1 - betas[0] ** t_adam)) / (np.sqrt(v / (1 - betas[1] ** t_adam)) + 1e-8)
            rho = np.clip(rho + step, 0.0, 1.0)
            if (it + 1) % save_every == 0 or it == iterations - 1:
                save()
    elif method == "lbfgs":
        from scipy.optimize import minimize

        it = start
        while it < iterations:
            b = bfun(it)
            end = it
            while end < iterations and bfun(end) == b:
                end += 1
            count = [it]

            class _Stop(Exception):
                pass

            def f(x, b=b, end=end):
                nonlocal rho, best
                if count[0] >= end:
                    raise _Stop
                t0 = time.time()
                r = x.reshape(rho.shape)
                val, grad, info = vg(r, b)
                if sign * val > sign * best[0]:
                    best = (float(val), r.copy())
                rho = r.copy()
                record(count[0], val, b, t0, info, r)
                count[0] += 1
                if count[0] % save_every == 0:
                    save()
                return -sign * val, (-sign * np.asarray(grad)).ravel()

            try:
                res = minimize(f, rho.ravel(), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * rho.size, options={"maxfun": end - it, "maxiter": end - it})
                rho = res.x.reshape(rho.shape)
            except _Stop:
                pass
            if count[0] == it:  # converged without an evaluation: stop
                break
            it = end if count[0] >= end else end  # next beta stage
            save()
    else:
        raise ValueError("method must be 'adam' or 'lbfgs'")
    out = {k: np.asarray(x) for k, x in hist.items()}
    out["best_fom"] = best[0]
    out["best_rho"] = best[1]
    return rho, out
