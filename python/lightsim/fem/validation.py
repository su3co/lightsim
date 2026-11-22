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


# ----------------------------------------------------------------------------
# electrostatics
# ----------------------------------------------------------------------------

_EPS0 = 8.8541878128e-12


def es_coax() -> CaseReport:
    """Coaxial line (a = 1, b = 3.5 um, eps_r = 2.1): C = 2 pi eps0 eps_r /
    ln(b/a). Curved P2 converges at order 4 in the energy (capacitance);
    P1 at order 2."""
    from .electrostatics import capacitance_matrix
    from .geometry import Circle, Geometry

    rep = CaseReport("es_coax")
    um = 1e-6
    a, b, er = 1 * um, 3.5 * um, 2.1
    exact = 2 * math.pi * _EPS0 * er / math.log(b / a)
    for order in (2, 1):
        errs, hs = [], []
        for h in (0.6 * um, 0.3 * um, 0.15 * um):
            g = Geometry().add(Circle((0, 0), b, size=h), "outer", "teflon", mesh_order=10).add(Circle((0, 0), a, size=h), "inner", "metal", mesh_order=1)
            m = g.mesh(default_size=h)
            C, _ = capacitance_matrix(m, ["inner"], {"teflon": er}, boundary="ground", order=order)
            errs.append(abs(C[0, 0] / exact - 1))
            hs.append(h)
        rep.check(f"P{order} C (finest)", C[0, 0], exact, 1e-6 if order == 2 else 3e-3)
        rep.check_abs(f"P{order} order of the C error", _orders(hs, errs)[-1], 4.0 if order == 2 else 2.0, 0.6)
        rep.notes.append(f"P{order} relative errors {', '.join(f'{e:.2e}' for e in errs)}")
    return rep


def es_anisotropic_ellipses() -> CaseReport:
    """Anisotropic dielectric (eps = diag(4, 9) rotated by 30 degrees)
    between two similar, equally rotated ellipses x^2/eps_x + y^2/eps_y = s^2
    (s = 1 and 3 um): exactly C = 2 pi eps0 sqrt(det eps) / ln 3 (the
    stretched-coordinate image of a coaxial line). Checks the full in-plane
    tensor including the off-diagonal term."""
    from .electrostatics import capacitance_matrix, rotate_tensor
    from .geometry import Geometry, Polygon

    rep = CaseReport("es_anisotropic_ellipses")
    um = 1e-6
    ex, ey, ang = 4.0, 9.0, 30.0
    exact = 2 * math.pi * _EPS0 * math.sqrt(ex * ey) / math.log(3.0)
    c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    def ellipse(sc, n=720):
        t = 2 * np.pi * np.arange(n) / n
        x, y = sc * math.sqrt(ex) * np.cos(t), sc * math.sqrt(ey) * np.sin(t)
        return Polygon(np.stack([c * x - s_ * y, s_ * x + c * y], 1).tolist())

    g = Geometry().add(ellipse(3 * um), "outer", "crystal", mesh_order=10).add(ellipse(1 * um), "inner", "metal", mesh_order=1)
    m = g.mesh(default_size=0.25 * um)
    C, _ = capacitance_matrix(m, ["inner"], {"crystal": rotate_tensor(ex, ey, ang)}, boundary="ground", order=2)
    rep.check("C with the rotated tensor", C[0, 0], exact, 2e-4)
    # the same ellipses with the unrotated tensor give a different capacitance
    C2, _ = capacitance_matrix(m, ["inner"], {"crystal": (ex, ey)}, boundary="ground", order=2)
    rep.notes.append(f"unrotated tensor on the rotated ellipses: C/C_exact = {C2[0, 0] / exact:.4f} (must differ from 1)")
    rep.checks.append(("unrotated tensor differs (|C/C_exact - 1| > 1e-2)", float(abs(C2[0, 0] / exact - 1) > 1e-2), 1.0, 0.0, True))
    return rep


def es_parallel_plates() -> CaseReport:
    """Two thin parallel plates (width w = 10 d, thickness d/50) in a large
    insulated box: capacitance with fringing against Palmer's formula
    C/eps0 = (w/d) [1 + d/(pi w) (1 + ln(2 pi w/d))] (J. Inst. Electr. Eng.
    81, 363 (1937)), an approximation for thin plates; the measured
    difference is quoted, the tolerance is 1 %."""
    from .electrostatics import capacitance_matrix
    from .geometry import Geometry, Rectangle

    rep = CaseReport("es_parallel_plates")
    um = 1e-6
    d, w, t = 1 * um, 10 * um, 0.02 * um
    L = 40 * w
    g = Geometry().add(Rectangle(-L / 2, -L / 2, L / 2, L / 2), "air", "air", mesh_order=10)
    g.add(Rectangle(-w / 2, d / 2, w / 2, d / 2 + t), "top", "metal", size=t, mesh_order=1)
    g.add(Rectangle(-w / 2, -d / 2 - t, w / 2, -d / 2), "bottom", "metal", size=t, mesh_order=1)
    m = g.mesh(default_size=L / 20, refine={"top": t, "bottom": t}, grading=0.25)
    C, _ = capacitance_matrix(m, ["top", "bottom"], {"air": 1.0}, order=2)
    c_pp = -C[0, 1]  # two-terminal capacitance with insulating outer boundary: -C12 (C11 = -C12 up to the box)
    palmer = _EPS0 * (w / d) * (1 + d / (math.pi * w) * (1 + math.log(2 * math.pi * w / d)))
    rep.check("C between the plates vs Palmer", c_pp, palmer, 1e-2)
    rep.notes.append(f"{m.n_tris} triangles; C/(eps0 w/d) = {c_pp / (_EPS0 * w / d):.4f}, Palmer {palmer / (_EPS0 * w / d):.4f}")
    return rep


def _hj_microstrip(u: float, er: float, t: float = 0.0):
    """Hammerstad & Jensen (IEEE MTT-S Digest 1980, 407): (Z0_air, eps_eff)
    of a microstrip with width/height u, relative thickness t/h."""
    eta0 = 376.730313412
    if t > 0:
        du1 = t / math.pi * math.log(1 + 4 * math.e / (t * (1 / math.tanh(math.sqrt(6.517 * u))) ** 2))
        dur = 0.5 * (1 + 1 / math.cosh(math.sqrt(er - 1))) * du1
        u1, ur = u + du1, u + dur
    else:
        u1 = ur = u

    def z(uu):
        f = 6 + (2 * math.pi - 6) * math.exp(-((30.666 / uu) ** 0.7528))
        return eta0 / (2 * math.pi) * math.log(f / uu + math.sqrt(1 + (2 / uu) ** 2))

    def ee(uu):
        a = 1 + math.log((uu**4 + (uu / 52) ** 2) / (uu**4 + 0.432)) / 49 + math.log(1 + (uu / 18.1) ** 3) / 18.7
        b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
        return (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / uu) ** (-a * b)

    z_air = z(u1)
    eps_eff = ee(ur) * (z(u1) / z(ur)) ** 2
    return z_air, eps_eff


def es_microstrip() -> CaseReport:
    """Microstrip (w = h, t = h/100, eps_r = 9.8) over a ground plane in a
    large box: capacitance per length in air C0 = 1 / (c Z0_air) and the
    effective permittivity C / C0 against Hammerstad & Jensen's closed forms
    with their thickness correction (stated accuracy 0.01 % for Z0_air and
    0.2 % for eps_eff with t = 0)."""
    from .electrostatics import capacitance_matrix
    from .geometry import Geometry, Rectangle

    rep = CaseReport("es_microstrip")
    um = 1e-6
    h, w, t, er = 1 * um, 1 * um, 0.01 * um, 9.8
    L, H = 60 * h, 40 * h
    c0 = 299792458.0
    res = {}
    for name, eps_sub in (("air", 1.0), ("sub", er)):
        g = Geometry()
        g.add(Rectangle(-L / 2, 0, L / 2, h), "substrate", "sub", mesh_order=5)
        g.add(Rectangle(-L / 2, h, L / 2, H), "air", "air", mesh_order=5)
        g.add(Rectangle(-w / 2, h, w / 2, h + t), "strip", "metal", size=t, mesh_order=1)
        m = g.mesh(default_size=H / 15, refine={"strip": t / 2}, grading=0.2)
        C, _ = capacitance_matrix(m, ["strip"], {"sub": eps_sub, "air": 1.0}, boundary={"bottom": 0.0, "top": 0.0, "left": 0.0, "right": 0.0}, order=2)
        res[name] = C[0, 0]
    z_air, eps_eff = _hj_microstrip(w / h, er, t / h)
    rep.check("C0 (air) vs 1/(c Z0_air)", res["air"], 1 / (c0 * z_air), 3e-3)
    rep.check("eps_eff = C/C0", res["sub"] / res["air"], eps_eff, 3e-3)
    rep.notes.append(f"Z0_air FEM {1 / (c0 * res['air']):.3f} ohm vs HJ {z_air:.3f}; eps_eff FEM {res['sub'] / res['air']:.4f} vs HJ {eps_eff:.4f}")
    return rep


def eo_uniform_field() -> CaseReport:
    """Pockels shift in a uniform field: an x-cut LiNbO3 slab (0.6 um,
    optic axis horizontal along u) between SiO2 layers, the side walls at
    1 V and 0 V give E_u = V / W exactly; the TE slab mode (E along u, no u
    dependence) shifts by the change of a slab of index n_e - n_e^3 r33 E/2,
    computed with the exact multilayer solver. Checks dneff/dV and V_pi L."""
    from ..eim import slab_modes
    from ..materials import refractive_index
    from .electrostatics import electro_optic_vpil
    from .geometry import Geometry, Rectangle

    rep = CaseReport("eo_uniform_field")
    um = 1e-6
    lam = 1.55e-6
    W, tl, tc = 4 * um, 0.6 * um, 1.5 * um
    g = Geometry()
    g.add(Rectangle(-W / 2, -tc, W / 2, 0), "box", "SiO2")
    g.add(Rectangle(-W / 2, 0, W / 2, tl), "ln", "LiNbO3")
    g.add(Rectangle(-W / 2, tl, W / 2, tl + tc), "clad", "SiO2")
    g.add(Rectangle(-W / 2 - 0.2 * um, -tc, -W / 2, tl + tc), "left", "Au")
    g.add(Rectangle(W / 2, -tc, W / 2 + 0.2 * um, tl + tc), "right", "Au")
    m = g.mesh(default_size=0.1 * um)
    r = electro_optic_vpil(m, "left", "right", "ln", "ln", lam, cut="x-cut", voltage=1.0, mode_window=(-W / 2 + 20e-9, -tc, W / 2 - 20e-9, tl + tc), mode_resolution=10e-9, permittivity={"SiO2": 3.9})
    # the static field in the slab: uniform V / W? (permittivity is layered in v only)
    E = r["static"].field(np.array([0.0, 1.0 * um]), np.array([0.3 * um, 0.3 * um]))
    rep.check("static E_u in the crystal (V/m)", float(E[0, 0]), 1.0 / W, 1e-9)
    ne = refractive_index("LiNbO3_e", lam).real
    no2 = refractive_index("SiO2", lam).real
    dn = -0.5 * ne**3 * 30.8e-12 * (1.0 / W)
    # a 1-V shift is ~1e-7: compare at 1e4 V scale to stay above the root-finding tolerance
    s = 1e4
    a = slab_modes([no2, ne, no2], [tl], lam, "te")[0]
    b = slab_modes([no2, ne + s * dn, no2], [tl], lam, "te")[0]
    rep.check("dneff/dV vs the slab dispersion relation", r["dneff_dV"], (b - a) / s, 5e-3)
    rep.notes.append(f"neff FD {r['neff'].real:.6f} vs slab {a:.6f}; V_pi L = {r['VpiL'] * 100:.3f} V cm (field over a {W / um:.0f} um gap)")
    return rep


CASES = {
    "poisson_convergence": poisson_convergence,
    "isoparametric_disk": isoparametric_disk,
    "heat_slab": heat_slab,
    "heat_cylinder": heat_cylinder,
    "heat_transient_slab": heat_transient_slab,
    "es_coax": es_coax,
    "es_anisotropic_ellipses": es_anisotropic_ellipses,
    "es_parallel_plates": es_parallel_plates,
    "es_microstrip": es_microstrip,
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
