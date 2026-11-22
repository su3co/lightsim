"""Validation suite of :mod:`lightsim.fem` against analytic references.

Run it with ``python -m lightsim.fem.validate`` (all cases) or
``python -m lightsim.fem.validate CASE ...``; ``python/tests/test_fem_*.py``
run the same functions. Each case returns a :class:`CaseReport` with checks
``(quantity, measured, expected, tolerance)`` like the FDTD suite of the
Rust core (``lightsim validate``), plus measured convergence orders and run
times in ``notes``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CaseReport", "CASES", "run"]


@dataclass
class CaseReport:
    name: str
    checks: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    seconds: float = 0.0

    def check(self, what: str, measured: float, expected: float, tol: float):
        """Relative check |m - e| <= tol |e| (absolute when e == 0)."""
        self.checks.append((what, float(measured), float(expected), float(tol), expected == 0))

    def check_abs(self, what: str, measured: float, expected: float, tol: float):
        self.checks.append((what, float(measured), float(expected), float(tol), True))

    @staticmethod
    def _err(m, e, a):
        return abs(m - e) if a else abs(m - e) / abs(e)

    @property
    def passed(self) -> bool:
        return all(self._err(m, e, a) <= t for _, m, e, t, a in self.checks)

    def summary(self) -> str:
        s = f"{self.name} [{'PASS' if self.passed else 'FAIL'}] ({self.seconds:.2f} s)\n"
        for w, m, e, t, a in self.checks:
            err = self._err(m, e, a)
            s += f"    {w:44} measured {m:12.6e}  expected {e:12.6e}  {'abs' if a else 'rel'} err {err:9.3e} (tol {t:.1e})\n"
        for n in self.notes:
            s += f"    note: {n}\n"
        return s


def _orders(hs, errs):
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    return np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])


def _mms_errors(mesh, order, exact, grad_exact, source, bc=None):
    """Solve -lap u = f with u = exact on the boundary; (L2, H1) errors."""
    from . import _scalar
    from .mesh import Field

    xy = mesh.dof_coords(order)
    bd = mesh.edge_dofs(mesh.boundary_edges(), order)
    k = np.tile([1.0, 1.0, 0.0], (mesh.n_tris, 1))
    ones = np.ones(mesh.n_tris)
    r = _scalar.solve(mesh, order, k, ones, None, dirichlet=[(bd, 0.0)], matrices=True)
    M = _scalar.csr(r["capacity"])
    load = M @ source(xy[:, 0], xy[:, 1])
    # boundary values: one Dirichlet group per DOF value
    from .. import _core

    vals = (exact if bc is None else bc)(xy[bd, 0], xy[bd, 1])
    r = _core.fem_scalar_solve(*mesh._args(), order, k, np.zeros(mesh.n_tris), np.zeros(mesh.n_tris), bd.tolist(), [vals.tolist()], ([], []), ([], [], []), "direct", False, False, np.ascontiguousarray(load))
    q = Field(mesh, order, r["u"][0]).quadrature(8)
    p = q["points"]
    e0 = math.sqrt(np.sum(q["weights"] * (q["values"] - exact(p[:, 0], p[:, 1])) ** 2))
    g = grad_exact(p[:, 0], p[:, 1])
    e1 = math.sqrt(np.sum(q["weights"] * ((q["grads"][:, 0] - g[0]) ** 2 + (q["grads"][:, 1] - g[1]) ** 2)))
    return e0, e1


def poisson_convergence() -> CaseReport:
    """Manufactured solution u = sin(pi x) sin(pi y) e^x of -lap u = f on
    the unit square, meshed by the lightsim mesher (unstructured) at three
    sizes: convergence orders 2 / 1 (P1) and 3 / 2 (P2) in L2 / H1."""
    from .geometry import Geometry, Rectangle

    rep = CaseReport("poisson_convergence")
    pi = math.pi
    exact = lambda x, y: np.sin(pi * x) * np.sin(pi * y) * np.exp(x)
    grad = lambda x, y: ((pi * np.cos(pi * x) + np.sin(pi * x)) * np.sin(pi * y) * np.exp(x), np.sin(pi * x) * pi * np.cos(pi * y) * np.exp(x))
    src = lambda x, y: (2 * pi * pi * np.sin(pi * x) - 2 * pi * np.cos(pi * x) - np.sin(pi * x)) * np.sin(pi * y) * np.exp(x)
    hs = [0.1, 0.05, 0.025]
    for order, (w0, w1) in ((1, (2, 1)), (2, (3, 2))):
        errs = []
        hmeas = []
        for h in hs:
            m = Geometry().add(Rectangle(0, 0, 1, 1), "sq").mesh(default_size=h)
            hmeas.append(float(np.sqrt(np.mean(m.areas()) * 2)))
            errs.append(_mms_errors(m, order, exact, grad, src))
        o0 = _orders(hmeas, [e[0] for e in errs])[-1]
        o1 = _orders(hmeas, [e[1] for e in errs])[-1]
        rep.check_abs(f"P{order} L2 order", o0, w0, 0.25)
        rep.check_abs(f"P{order} H1 order", o1, w1, 0.25)
        rep.notes.append(f"P{order}: L2 errors {', '.join(f'{e[0]:.2e}' for e in errs)}; H1 {', '.join(f'{e[1]:.2e}' for e in errs)}")
    return rep


def isoparametric_disk() -> CaseReport:
    """-lap u = f on the unit disk with u = (1 - r^2) e^x: P2 on a mesh whose
    boundary edges are curved (isoparametric) converges at order 3 in L2;
    the same P2 on the straight-sided mesh is limited by the polygonal
    boundary."""
    from .geometry import Circle, Geometry

    rep = CaseReport("isoparametric_disk")
    exact = lambda x, y: (1 - x * x - y * y) * np.exp(x)
    grad = lambda x, y: ((-2 * x + 1 - x * x - y * y) * np.exp(x), -2 * y * np.exp(x))
    src = lambda x, y: np.exp(x) * (3 + 4 * x + x * x + y * y)
    hs, curved, straight = [], [], []
    for h in (0.2, 0.1, 0.05):
        m = Geometry().add(Circle((0, 0), 1.0, size=h), "disk").mesh(default_size=h)
        hs.append(float(np.sqrt(np.mean(m.areas()) * 2)))
        zero = lambda x, y: 0 * x  # the boundary condition u = 0 on the circle
        curved.append(_mms_errors(m, 2, exact, grad, src, zero)[0])
        m.edge_mid[:] = np.nan
        straight.append(_mms_errors(m, 2, exact, grad, src, zero)[0])
    oc = _orders(hs, curved)[-1]
    os_ = _orders(hs, straight)[-1]
    rep.check_abs("P2 curved L2 order", oc, 3.0, 0.3)
    rep.check_abs("straight-sided L2 order (polygonal boundary)", os_, 2.0, 0.3)
    rep.notes.append(f"curved L2 errors {', '.join(f'{e:.2e}' for e in curved)} (order {oc:.2f}); straight {', '.join(f'{e:.2e}' for e in straight)} (order {os_:.2f})")
    return rep


CASES = {
    "poisson_convergence": poisson_convergence,
    "isoparametric_disk": isoparametric_disk,
}


def run(names=None, verbose: bool = True) -> list[CaseReport]:
    """Run the named cases (all by default) and print their summaries."""
    out = []
    for name in names or list(CASES):
        t = time.perf_counter()
        rep = CASES[name]()
        rep.seconds = time.perf_counter() - t
        if verbose:
            print(rep.summary(), end="", flush=True)
        out.append(rep)
    return out
