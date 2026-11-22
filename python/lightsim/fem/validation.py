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


# ----------------------------------------------------------------------------
# heat conduction
# ----------------------------------------------------------------------------


def heat_slab() -> CaseReport:
    """Two-layer slab (k = 1.4 below, 150 above, 1 um each, insulated
    sides), uniform source in the upper layer, T = 0 at the bottom and
    convection h = 1e5 W/(m^2 K) to T_inf = 0 at the top: the exact
    temperature is piecewise quadratic; P2 reproduces it to round-off and P1
    at its nodes (1-D problem)."""
    from .geometry import Geometry, Rectangle
    from .heat import solve_heat

    rep = CaseReport("heat_slab")
    um = 1e-6
    k1, k2, L, q, h = 1.4, 150.0, 1 * um, 1e14, 1e5  # q in W/m^3
    g = Geometry().add(Rectangle(0, 0, 0.5 * um, L), "lower", "a").add(Rectangle(0, L, 0.5 * um, 2 * L), "upper", "b")
    m = g.mesh(default_size=0.1 * um)
    p = q * 0.5 * um * L  # power per unit length of the upper layer
    # exact: flux at the top F = h T(2L) leaves; all heat q L goes down or up.
    # Lower: T = A y. Upper: T = A L + (A k1/k2)(y - L) - q (y - L)^2 / (2 k2).
    # Top: -k2 T'(2L) = h T(2L) -> solve for A.
    # T'(2L) = A k1/k2 - q L / k2 ; T(2L) = A L + A k1 L / k2 - q L^2 / (2 k2)
    a = (q * L + h * q * L * L / (2 * k2)) / (k1 + h * (L + k1 * L / k2))

    def exact(y):
        return np.where(y <= L, a * y, a * L + a * k1 / k2 * (y - L) - q * (y - L) ** 2 / (2 * k2))

    for order in (1, 2):
        r = solve_heat(m, {"upper": p}, {"a": k1, "b": k2}, fixed={"bottom": 0.0}, convection={"top": (h, 0.0)}, order=order)
        xy = m.dof_coords(order)
        err = np.max(np.abs(r.temperature.values - exact(xy[:, 1]))) / np.max(exact(xy[:, 1]))
        rep.check_abs(f"P{order} max nodal error / max T", err, 0.0, 1e-9 if order == 2 else 1e-4)
    rep.check("top temperature (P2)", float(r.temperature(np.array([0.25 * um]), np.array([2 * L]))[0]), float(exact(np.array([2 * L]))[0]), 1e-9)
    return rep


def heat_cylinder() -> CaseReport:
    """Concentric cylinders: a heated rod (radius a = 1 um, k1 = 28,
    p = 10 W/m) inside a shell (b = 5 um, k2 = 1.4) held at T = 0 on r = b.
    Exact: T = p ln(b/r) / (2 pi k2) outside, plus p (1 - r^2/a^2) / (4 pi k1)
    inside. Checks the centre temperature and the L2 error with curved P2
    (convergence order) and P1."""
    from .geometry import Circle, Geometry
    from .heat import solve_heat

    rep = CaseReport("heat_cylinder")
    um = 1e-6
    a, b, k1, k2, p = 1 * um, 5 * um, 28.0, 1.4, 10.0

    def exact(x, y):
        r = np.hypot(x, y)
        out = p * np.log(b / np.maximum(r, a)) / (2 * np.pi * k2)
        return out + np.where(r < a, p * (1 - r**2 / a**2) / (4 * np.pi * k1), 0.0)

    t0 = p * np.log(b / a) / (2 * np.pi * k2) + p / (4 * np.pi * k1)
    for order in (2, 1):
        errs, hs = [], []
        for hsz in (0.8 * um, 0.4 * um, 0.2 * um):
            g = Geometry().add(Circle((0, 0), b, size=hsz), "shell", "oxide", mesh_order=10).add(Circle((0, 0), a, size=hsz), "rod", "metal", mesh_order=1)
            m = g.mesh(default_size=hsz)
            r = solve_heat(m, {"rod": p}, {"oxide": k2, "metal": k1}, fixed={lambda x, y: np.hypot(x, y) > 0.99 * b: 0.0}, order=order)
            q = r.temperature.quadrature(8)
            pt = q["points"]
            errs.append(math.sqrt(np.sum(q["weights"] * (q["values"] - exact(pt[:, 0], pt[:, 1])) ** 2) / np.sum(q["weights"] * exact(pt[:, 0], pt[:, 1]) ** 2)))
            hs.append(hsz)
        o = _orders(hs, errs)[-1]
        rep.check_abs(f"P{order} L2 order", o, 3.0 if order == 2 else 2.0, 0.35)
        rep.check(f"P{order} centre temperature", float(r.temperature(np.array([0.0]), np.array([0.0]))[0]), t0, 2e-4 if order == 2 else 5e-3)
        rep.notes.append(f"P{order} relative L2 errors {', '.join(f'{e:.2e}' for e in errs)}")
    return rep


def heat_transient_slab() -> CaseReport:
    """Transient decay of the first slab mode: T(y, 0) = sin(pi y / L),
    T = 0 at y = 0 and L, insulated sides: T = sin(pi y / L) exp(-t / tau)
    with tau = rho c L^2 / (k pi^2). Crank-Nicolson, P2; checks the
    amplitude after one and three time constants and the fitted tau."""
    from .geometry import Geometry, Rectangle
    from .heat import solve_heat_transient

    rep = CaseReport("heat_transient_slab")
    um = 1e-6
    L, k, rc = 2 * um, 1.4, 1.66e6
    tau = rc * L * L / (k * np.pi**2)
    m = Geometry().add(Rectangle(0, 0, 0.4 * um, L), "slab", "oxide").mesh(default_size=0.1 * um)
    nsteps = 300
    tr = solve_heat_transient(
        m, None, t_end=3 * tau, nsteps=nsteps, conductivity={"oxide": k}, capacity={"oxide": rc},
        fixed={"bottom": 0.0, "top": 0.0}, initial=lambda x, y: np.sin(np.pi * y / L), order=2, theta=0.5, snapshots=[nsteps // 3],
    )
    f1 = tr.snapshots[0]
    y = np.array([0.5 * L])
    x = np.array([0.2 * um])
    rep.check("T(L/2) at t = tau", float(f1(x, y)[0]), math.exp(-1), 2e-4)
    rep.check("T(L/2) at t = 3 tau", float(tr.final(x, y)[0]), math.exp(-3), 1e-3)
    # time constant from a log-linear fit of the mid-plane value
    rep.notes.append(f"tau = {tau * 1e6:.4f} us, dt = tau/{nsteps // 3}")
    return rep


CASES = {
    "poisson_convergence": poisson_convergence,
    "isoparametric_disk": isoparametric_disk,
    "heat_slab": heat_slab,
    "heat_cylinder": heat_cylinder,
    "heat_transient_slab": heat_transient_slab,
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
