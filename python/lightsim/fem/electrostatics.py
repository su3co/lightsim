"""Electrostatics on a cross-section: potentials, fields, capacitance
matrices per unit length and electro-optic (Pockels) modulators.

Solves ``-div(eps0 eps_r grad phi) = rho`` with P1 or P2 elements, where
``eps_r`` is a scalar or a symmetric in-plane tensor ``(eps_uu, eps_vv,
eps_uv)`` per region. Conductors are regions held at fixed potentials (their
interior is meshed and fixed). The outer boundary is insulating (zero normal
displacement, ``"neumann"``) or grounded (``"ground"``).

The Maxwell capacitance matrix per unit length is ``C_ij = u_i^T K u_j``
(K: stiffness, u_j: the solution with conductor j at 1 V and every other
conductor at 0 V), i.e. the field energy, which converges at twice the
order of the potential. This is the 2-D counterpart of gplugins'
capacitance simulations with Palace and Elmer.

:func:`electro_optic_vpil` computes the static field of a modulator, the
index change of a Pockels crystal (LiNbO3: r13, r22, r33, r51 of
:data:`lightsim.materials.ELECTRO_OPTIC`; diagonal terms of the index
ellipsoid in the cross-section frame, off-diagonal terms dropped), the
resulting effective-index change of the optical mode and the half-wave
voltage-length product ``V_pi L = lambda / (2 dn_eff/dV)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _materials, _scalar
from .mesh import Field, Mesh

EPS0 = 8.8541878128e-12

__all__ = ["ElectrostaticResult", "solve_electrostatics", "capacitance_matrix", "electro_optic_vpil", "rotate_tensor", "crystal_tensor"]

#: Crystal axes along the cross-section axes (u: in-plane horizontal, v:
#: vertical, w: propagation) for common cuts.
CUTS = {
    "x-cut": ("z", "x", "y"),  # x-cut, propagation along y: optic axis in-plane, horizontal
    "x-cut-z-prop": ("y", "x", "z"),
    "z-cut": ("y", "z", "x"),  # optic axis vertical, propagation along x
    "y-cut": ("z", "y", "x"),
}


def crystal_tensor(diag, axes=("x", "y", "z")) -> tuple:
    """(eps_uu, eps_vv, eps_ww) of a crystal-frame diagonal tensor for the
    crystal axes lying along (u, v, w)."""
    d = dict(zip("xyz", np.broadcast_to(np.asarray(diag, float), (3,))))
    return tuple(d[a] for a in axes)


def rotate_tensor(eps_uu: float, eps_vv: float, angle_deg: float) -> tuple:
    """In-plane tensor (eps_uu, eps_vv, eps_uv) of a diagonal tensor rotated by ``angle_deg``."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * c * eps_uu + s * s * eps_vv, s * s * eps_uu + c * c * eps_vv, c * s * (eps_uu - eps_vv))


@dataclass
class ElectrostaticResult:
    """Potential (V) of one solve, with the field and energy helpers."""

    mesh: Mesh
    potential: Field
    permittivity: np.ndarray  # per triangle (eps_uu, eps_vv, eps_uv), relative
    conductors: dict

    def field(self, x, y) -> np.ndarray:
        """Electric field E = -grad(phi) (V/m) at points, shape (..., 2)."""
        return -self.potential.grad(x, y)

    def field_per_triangle(self) -> np.ndarray:
        """Field at the triangle centroids (M, 2)."""
        c = self.mesh.centroids()
        return self.field(c[:, 0], c[:, 1])

    def energy(self) -> float:
        """Electrostatic energy per unit length (J/m), from the field."""
        q = self.potential.quadrature(6)
        g = q["grads"]
        e = self.permittivity[q["tri"]]
        dens = e[:, 0] * g[:, 0] ** 2 + e[:, 1] * g[:, 1] ** 2 + 2 * e[:, 2] * g[:, 0] * g[:, 1]
        return 0.5 * EPS0 * float(np.sum(q["weights"] * dens))


def _eps_rows(mesh: Mesh, permittivity: dict | None, conductor_names) -> np.ndarray:
    cond_tags = set()
    for n in conductor_names:
        cond_tags.update(mesh.tags_of(n))
    perm = dict(permittivity or {})
    for t in cond_tags:  # conductors: any value (their potential is fixed)
        perm.setdefault(mesh.region_names[t], 1.0)
    vals = mesh.per_region(perm, default=_materials.static_permittivity, what="permittivity")
    rows = []
    for v in vals:
        v = np.atleast_1d(np.asarray(v, float))
        if v.size == 3 and not np.isfinite(v).all():
            raise ValueError("infinite permittivity: list the region as a conductor instead")
        rows.append(v)
    out = _scalar.k_tensor(rows)
    out[~np.isfinite(out)] = 1.0
    return out[mesh.tags]


def _outer(mesh: Mesh, boundary, order):
    if boundary in (None, "neumann", "open"):
        return []
    if boundary in ("ground", "dirichlet"):
        return [(mesh.edge_dofs(mesh.boundary_edges(), order), 0.0)]
    if isinstance(boundary, dict):
        return [(mesh.edge_dofs(mesh.boundary_edges(w), order), float(v)) for w, v in boundary.items()]
    raise ValueError(f"boundary must be 'neumann', 'ground' or a dict, got {boundary!r}")


def solve_electrostatics(mesh: Mesh, conductors: dict, permittivity: dict | None = None, boundary="neumann", charge: dict | None = None, order: int = 2) -> ElectrostaticResult:
    """Potential of conductors at fixed voltages.

    Parameters
    ----------
    conductors : dict
        ``{region name: potential (V)}``.
    permittivity : dict, optional
        ``{region or material: eps_r}``, a scalar or an in-plane tensor
        ``(eps_uu, eps_vv)`` / ``(eps_uu, eps_vv, eps_uv)`` (see
        :func:`crystal_tensor`, :func:`rotate_tensor`); others from the
        static-permittivity table.
    boundary : str or dict
        ``"neumann"`` (insulating outer boundary, default), ``"ground"``
        (outer boundary at 0 V) or ``{side: V}``.
    charge : dict, optional
        ``{region: charge density (C/m^3)}``.
    """
    eps = _eps_rows(mesh, permittivity, set(conductors))
    rho = np.zeros(mesh.n_tris)
    for name, q in (charge or {}).items():
        rho[mesh.region_mask(name)] += float(q) / EPS0  # solve with eps_r and rho/eps0
    dirichlet = _outer(mesh, boundary, order) + [(mesh.region_dofs(n, order), float(v)) for n, v in conductors.items()]
    if not dirichlet:
        raise ValueError("no conductor and an insulating boundary: the potential is undetermined")
    r = _scalar.solve(mesh, order, eps, None, rho, dirichlet=dirichlet)
    return ElectrostaticResult(mesh, Field(mesh, order, r["u"][0]), eps, dict(conductors))


def capacitance_matrix(mesh: Mesh, conductors, permittivity: dict | None = None, ground=(), boundary="neumann", order: int = 2):
    """Maxwell capacitance matrix per unit length (F/m).

    ``conductors``: names of the terminal regions (the matrix order);
    ``ground``: regions always at 0 V (not terminals); ``boundary`` as in
    :func:`solve_electrostatics`. Diagonal entries are self-capacitances,
    off-diagonal entries the negative mutual capacitances (the convention of
    gplugins' ElectrostaticResults). Returns ``(C, names)``; one sparse
    factorisation serves all terminals.
    """
    names = list(conductors)
    eps = _eps_rows(mesh, permittivity, set(names) | set(ground))
    outer = _outer(mesh, boundary, order)
    groups = outer + [(mesh.region_dofs(g, order), 0.0) for g in ground] + [(mesh.region_dofs(n, order), 0.0) for n in names]
    n0 = len(outer) + len(ground)
    cases = []
    for i in range(len(names)):
        vals = [0.0] * len(groups)
        vals[n0 + i] = 1.0
        cases.append(vals)
    r = _scalar.solve(mesh, order, eps, None, None, dirichlet=groups, cases=cases, matrices=True)
    K = _scalar.csr(r["stiffness"])
    U = np.asarray(r["u"])
    C = EPS0 * (U @ (K @ U.T))
    return 0.5 * (C + C.T), names


def _pockels_dn(E_crystal: np.ndarray, n_o: float, n_e: float, coeffs: dict) -> np.ndarray:
    """Diagonal index changes (dn_x, dn_y, dn_z) in the crystal frame of a 3m
    crystal (LiNbO3) for static fields E (..., 3) in the crystal frame."""
    r13, r22, r33 = coeffs["r13"], coeffs["r22"], coeffs["r33"]
    Ex, Ey, Ez = E_crystal[..., 0], E_crystal[..., 1], E_crystal[..., 2]
    d_eta = np.stack([-r22 * Ey + r13 * Ez, r22 * Ey + r13 * Ez, r33 * Ez], -1)
    n = np.array([n_o, n_o, n_e])
    return -0.5 * n**3 * d_eta


def electro_optic_vpil(
    mesh: Mesh,
    signal: str,
    ground,
    crystal: str,
    waveguide: str,
    wavelength: float = 1.55e-6,
    cut="x-cut",
    voltage: float = 1.0,
    permittivity: dict | None = None,
    index: dict | None = None,
    order: int = 2,
    mode_window: float = 1.5e-6,
    mode_resolution: float = 20e-9,
    mode_method: str = "fd",
    polarization: str = "te",
) -> dict:
    """Half-wave voltage-length product of an electro-optic phase shifter.

    ``signal``: the electrode region at ``voltage``; ``ground``: region(s) at
    0 V; ``crystal``: the region(s) of Pockels material (LiNbO3) whose
    crystal axes lie along the cross-section axes as given by ``cut`` (a key
    of :data:`CUTS` or a tuple of crystal axes along (u, v, w)). The static
    permittivity of the crystal follows the same axes unless given.
    ``waveguide``: the region around which the optical mode is solved, in a
    window ``mode_window`` (m) larger on every side (or an explicit window
    ``(x0, y0, x1, y1)``).
    ``mode_method``: ``"fd"`` (lightsim's finite-difference solver on the
    perturbed, cell-averaged index) or ``"fem"`` (:func:`lightsim.fem.solve_modes`);
    of the two modes of highest index the one with the larger (``"te"``) or
    smaller (``"tm"``) TE fraction is used.

    Returns a dict with ``dneff_dV`` (1/V), ``VpiL`` (V m), ``neff``, the
    :class:`ElectrostaticResult` ``static`` and the per-triangle crystal-frame
    field ``E_crystal`` (V/m at ``voltage``).
    """
    from ..materials import electro_optic_coefficients, refractive_index

    axes = CUTS[cut] if isinstance(cut, str) else tuple(cut)
    coeffs = electro_optic_coefficients("LiNbO3")
    cryst_names = [crystal] if isinstance(crystal, str) else list(crystal)
    perm = dict(permittivity or {})
    for c in cryst_names:
        if c not in perm:
            uu, vv, _ = crystal_tensor(coeffs["eps_static"], axes)
            perm[c] = (uu, vv)
    grounds = [ground] if isinstance(ground, str) else list(ground)
    static = solve_electrostatics(mesh, {signal: voltage, **{g: 0.0 for g in grounds}}, perm, order=order)
    Et = static.field_per_triangle()  # (M, 2) in (u, v)
    E3 = np.zeros((mesh.n_tris, 3))
    for k, a in enumerate(axes[:2]):
        E3[:, "xyz".index(a)] = Et[:, k]
    n_o = float(refractive_index("LiNbO3_o", wavelength).real)
    n_e = float(refractive_index("LiNbO3_e", wavelength).real)
    dn_cr = _pockels_dn(E3, n_o, n_e, coeffs)  # (M, 3) crystal frame
    mask = mesh.region_mask(cryst_names)
    # unperturbed index per triangle and axis (u, v, w)
    n_axes = np.zeros((mesh.n_tris, 3))
    for t, name in enumerate(mesh.materials):
        sel = mesh.tags == t
        if name in cryst_names or mesh.region_names[t] in cryst_names:
            n_axes[sel] = crystal_tensor((n_o, n_o, n_e), axes)
        else:
            try:
                n_axes[sel] = _materials.refractive_index(name, wavelength, index).real
            except KeyError:
                n_axes[sel] = np.nan
    dn_axes = np.zeros_like(n_axes)
    for k, a in enumerate(axes):
        dn_axes[:, k] = np.where(mask, dn_cr[:, "xyz".index(a)], 0.0)
    lo = mesh.nodes[mesh.tris[mesh.region_mask(waveguide)].ravel()].min(0)
    hi = mesh.nodes[mesh.tris[mesh.region_mask(waveguide)].ravel()].max(0)
    if np.ndim(mode_window) == 0:
        win = (lo[0] - mode_window, lo[1] - mode_window, hi[0] + mode_window, hi[1] + mode_window)
    else:
        win = tuple(float(v) for v in mode_window)
    if mode_method == "fd":
        from .heat import neff_fd

        def pick(eps):
            ns, raw = neff_fd(mesh, None, wavelength, win, mode_resolution, eps_tri=eps, num_modes=2)
            te = [float(np.sum(np.abs(r[1]) ** 2 * r[5]) / (np.sum(np.abs(r[1]) ** 2 * r[5]) + np.sum(np.abs(r[2]) ** 2 * r[6]))) for r in raw]
            i = int(np.argmax(te)) if polarization == "te" else int(np.argmin(te))
            return ns[i]

        n0 = pick(n_axes**2)
        d = pick((n_axes + dn_axes) ** 2) - n0
    else:
        from .modes import solve_modes

        kw = dict(num_modes=2, window=win)
        m0 = solve_modes(mesh, wavelength, eps=n_axes**2, **kw)
        m1 = solve_modes(mesh, wavelength, eps=(n_axes + dn_axes) ** 2, **kw)
        pick = (lambda ms: max(ms, key=lambda m: m.te_fraction)) if polarization == "te" else (lambda ms: min(ms, key=lambda m: m.te_fraction))
        a0, a1 = pick(m0), pick(m1)
        n0, d = a0.neff, a1.neff - a0.neff
    dndv = d.real / voltage
    return {"neff": n0, "dneff_dV": dndv, "VpiL": wavelength / (2 * abs(dndv)) if dndv != 0 else float("inf"), "static": static, "E_crystal": E3}
