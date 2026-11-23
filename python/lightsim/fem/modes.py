"""Full-vectorial finite-element waveguide modes.

``solve_modes`` discretises Maxwell's equations on a cross-section mesh with
first-order Nedelec edge elements for the transverse field and P1 Lagrange
elements for the longitudinal one (the formulation of Vardapetyan and
Demkowicz, as in femwell), and finds the modes with a shift-invert Arnoldi
eigen-solver (Rust core). Permittivities may be complex (loss, metals) and
anisotropic (a full in-plane tensor plus eps_zz); perfectly matched layers
absorb leaky and radiating fields; a bend radius turns the problem into the
modes of a bent waveguide (conformal transformation), whose imaginary
effective index is the bend loss.

The modes are :class:`FemMode` objects, a subclass of
:class:`lightsim.modes.Mode`: ``neff``, ``freq``, ``te_fraction``, ``E`` and
``H`` (sampled on a regular grid, see ``info``) as for the finite-difference
solver, plus the fields anywhere (:meth:`FemMode.fields`), the group index,
the effective area, overlaps, and the profile a lightsim ``ModeSource`` or
``ModeMonitor`` takes (:meth:`FemMode.profile_for`).
"""

from __future__ import annotations

import math

import numpy as np

from ..modes import C0, ETA0, Mode
from . import _materials
from .mesh import Mesh

__all__ = ["FemMode", "solve_modes", "eps_from"]


def eps_from(mesh: Mesh, wavelength: float, index=None, eps=None) -> np.ndarray:
    """Per-triangle permittivity (M, 4) = (eps_xx, eps_yy, eps_xy, eps_zz), complex.

    ``eps``: a dict ``{region or material: value}`` or a per-triangle
    array; a value is a scalar, ``(eps_xx, eps_yy, eps_zz)`` (diagonal) or
    ``(eps_xx, eps_yy, eps_xy, eps_zz)``. ``index``: the same with
    refractive indices (scalars). Otherwise, and for regions missing from a
    dict, the index comes from lightsim's material library at
    ``wavelength``.
    """
    M = mesh.n_tris

    def row(v):
        v = np.atleast_1d(np.asarray(v, complex))
        if v.size == 1:
            return np.array([v[0], v[0], 0, v[0]])
        if v.size == 3:
            return np.array([v[0], v[1], 0, v[2]])
        if v.size == 4:
            return v
        raise ValueError(f"a permittivity needs 1, 3 or 4 values, got {v.size}")

    if eps is not None and not isinstance(eps, dict):
        a = np.asarray(eps, complex)
        if a.ndim == 1:
            return np.stack([a, a, 0 * a, a], 1)
        if a.shape == (M, 3):
            return np.stack([a[:, 0], a[:, 1], 0 * a[:, 0], a[:, 2]], 1)
        if a.shape == (M, 4):
            return a
        raise ValueError(f"eps array must be (M,), (M, 3) or (M, 4) for M = {M} triangles, got {a.shape}")
    if index is not None and not isinstance(index, dict):
        n = np.asarray(index, complex)
        return np.stack([n**2, n**2, 0 * n, n**2], 1)
    rows = []
    for t, (name, mat) in enumerate(zip(mesh.region_names, mesh.materials)):
        v = None
        if isinstance(eps, dict):
            for key in (name, mat):
                if key in eps:
                    v = row(eps[key])
                    break
        if v is None:
            n = None
            if isinstance(index, dict):
                for key in (name, mat):
                    if key in index:
                        n = complex(index[key])
                        break
            if n is None:
                n = _materials.refractive_index(mat, wavelength, index if isinstance(index, dict) else None)
            v = row(n * n)
        rows.append(v)
    return np.asarray(rows, complex)[mesh.tags]


class FemMode(Mode):
    """A mode of :func:`solve_modes` (a :class:`lightsim.modes.Mode`).

    Extra attributes: ``mesh``, ``wavelength``, ``beta`` (1/m),
    ``group_index`` (from the field energy; material dispersion not
    included), ``effective_area`` (m^2, ``(int |E|^2)^2 / int |E|^4``),
    ``power`` (W, 1 after normalisation).
    """

    def __init__(self, neff, freq, te_fraction, E, H, info, *, mesh, problem, x, beta, ints):
        super().__init__(neff=complex(neff), freq=float(freq), te_fraction=float(te_fraction), profile={}, E=E, H=H, info=info)
        self.mesh = mesh
        self._problem = problem
        self._x = x
        self.beta = complex(beta)
        self.wavelength = C0 / freq
        self.group_index = float(ints[5] / ints[0]) if ints[0] != 0 else float("nan")
        self.effective_area = float(ints[3] ** 2 / ints[4]) if ints[4] > 0 else float("nan")
        self.power = 1.0

    def __repr__(self) -> str:
        return f"<FemMode neff={self.neff.real:.6f}{self.neff.imag:+.2e}j TE={self.te_fraction:.2f} ng={self.group_index:.4f}>"

    # femwell-compatible names
    @property
    def n_eff(self) -> complex:
        return self.neff

    @property
    def tm_fraction(self) -> float:
        return 1.0 - self.te_fraction

    @property
    def loss_db_per_m(self) -> float:
        """Power loss (dB/m) from Im(neff): 20 log10(e) k0 Im(neff)."""
        return 20 / math.log(10) * 2 * math.pi / self.wavelength * self.neff.imag

    def calculate_propagation_loss(self, distance: float) -> float:
        """Loss in dB over ``distance`` (m)."""
        return self.loss_db_per_m * distance

    def fields(self, x, y):
        """``(E, H)`` at points (arrays in m): complex (..., 3) each, E in
        V/m and H in A/m for 1 W; NaN outside the mesh."""
        from .. import _core

        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        pts = np.ascontiguousarray(np.stack([x.ravel(), y.ravel()], 1))
        p = self._problem
        d = _core.fem_mode_fields(self.mesh.nodes, self.mesh.tris, self.mesh.tags, p["eps"], p["k0"], (self.beta.real, self.beta.imag), self._x, pts, p["pml"], p["bend"])
        return d["E"].reshape(x.shape + (3,)), d["H"].reshape(x.shape + (3,)) / ETA0

    def quadrature(self, degree: int = 4) -> dict:
        """Quadrature points, weights, triangle index, E and H (A/m) of the mode."""
        from .. import _core

        p = self._problem
        d = _core.fem_mode_fields(self.mesh.nodes, self.mesh.tris, self.mesh.tags, p["eps"], p["k0"], (self.beta.real, self.beta.imag), self._x, None, p["pml"], p["bend"], degree)
        d["H"] = d["H"] / ETA0
        return d

    def calculate_overlap(self, other: "FemMode") -> complex:
        """``1/2 int (E_self x H_other*) . z dA`` over the mesh (both modes on
        the same mesh); 1 for a mode with itself (unit power)."""
        if other.mesh is not self.mesh:
            raise ValueError("overlaps need both modes on the same mesh")
        a = self.quadrature()
        b = other.quadrature()
        e, h, w = a["E"], b["H"], a["weights"]
        return complex(0.5 * np.sum(w * (e[:, 0] * np.conj(h[:, 1]) - e[:, 1] * np.conj(h[:, 0]))))

    def calculate_confinement_factor(self, region) -> float:
        """Share of ``int |E|^2`` in the named region(s)."""
        q = self.quadrature()
        e2 = np.sum(np.abs(q["E"]) ** 2, 1) * q["weights"]
        mask = self.mesh.region_mask(region)[q["tri"]]
        return float(np.sum(e2[mask]) / np.sum(e2))

    def profile_for(self, sim, axis: int, position: float, min, max, offset=(0.0, 0.0)) -> dict:
        """The ``profile`` dict of lightsim's ``ModeSource`` / ``ModeMonitor``
        for a window of a :class:`lightsim.Simulation`, sampled from the FEM
        fields at the exact staggered positions the FDTD engine uses.

        The cross-section coordinates (x, y) of the mesh map to the plane's
        in-plane axes (u, v) as ``u = x + offset[0]``, ``v = y + offset[1]``;
        the propagation direction is +``axis``. For ``axis=1`` the in-plane
        axes (x, z) are left-handed around +y, so the mode's x maps to z and
        its y to x as in :func:`lightsim.solve_modes`.
        """
        from .. import _core

        spec = sim.geometry_json() if hasattr(sim, "geometry_json") else sim
        info = _core.plane_info(spec, int(axis), float(position), [float(v) for v in min], [float(v) for v in max], float(self.freq))
        un, vn = np.asarray(info["u_nodes"]), np.asarray(info["v_nodes"])
        nu, nv = int(info["dims"][0]), int(info["dims"][1])
        uc, vc = 0.5 * (un[:-1] + un[1:]), 0.5 * (vn[:-1] + vn[1:])
        swap = axis == 1
        k0 = 2 * math.pi * self.freq / C0

        def at(U, V):
            U, V = np.broadcast_arrays(U, V)
            if swap:  # plane (u, v) = (x, z); mode (x, y) = (z, x)
                return self.fields(V - offset[0], U - offset[1])
            return self.fields(U - offset[0], V - offset[1])

        Eu_pos = at(uc[:, None], vn[None, :nv])  # E_u and H_v positions
        Ev_pos = at(un[:nu, None], vc[None, :])  # E_v and H_u positions
        cu, cv = (1, 0) if swap else (0, 1)
        eu = np.nan_to_num(Eu_pos[0][..., cu])
        ev = np.nan_to_num(Ev_pos[0][..., cv])
        hv = np.nan_to_num(Eu_pos[1][..., cv])
        hu = np.nan_to_num(Ev_pos[1][..., cu])
        shift = np.exp(-1j * self.neff * k0 * info["h_offset"])
        hu_p, hv_p = hu * shift, hv * shift
        return {
            "neff": float(self.neff.real),
            "neff_imag": float(self.neff.imag),
            "dims": [nu, nv],
            "e_re": [np.real(eu).ravel().tolist(), np.real(ev).ravel().tolist()],
            "e_im": [np.imag(eu).ravel().tolist(), np.imag(ev).ravel().tolist()],
            "h_re": [np.real(hu_p).ravel().tolist(), np.real(hv_p).ravel().tolist()],
            "h_im": [np.imag(hu_p).ravel().tolist(), np.imag(hv_p).ravel().tolist()],
            "freq": float(self.freq),
            "h_offset": float(info["h_offset"]),
        }

    def plot(self, component: str = "Ex", part: str = "abs", ax=None, n: int = 300):
        """Plot one field component (``Ex``, ``Ey``, ``Ez``, ``Hx``, ...,
        ``E`` for |E|) on a regular grid (needs matplotlib)."""
        import matplotlib.pyplot as plt

        x0, y0, x1, y1 = self.info["window"]
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, max(2, int(n * (y1 - y0) / (x1 - x0))))
        X, Y = np.meshgrid(xs, ys)
        E, H = self.fields(X, Y)
        if component in ("E", "H"):
            v = np.sqrt(np.sum(np.abs(E if component == "E" else H) ** 2, -1))
        else:
            v = (E if component[0] == "E" else H)[..., "xyz".index(component[1])]
            v = {"abs": np.abs, "real": np.real, "imag": np.imag}[part](v)
        if ax is None:
            _, ax = plt.subplots()
        im = ax.pcolormesh(xs * 1e6, ys * 1e6, v, shading="auto")
        plt.colorbar(im, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"{component} neff={self.neff.real:.4f}")
        return ax


def _pml_spec(mesh: Mesh, pml, strength: float):
    if pml is None or pml is False:
        return None
    x0, y0, x1, y1 = mesh.bounds
    if isinstance(pml, (int, float)):
        t = [float(pml)] * 4
    elif isinstance(pml, dict):
        t = [float(pml.get(k, 0.0)) for k in ("left", "bottom", "right", "top")]
    else:
        t = [float(v) for v in pml]
    inner = [x0 + t[0], y0 + t[1], x1 - t[2], y1 - t[3]]
    return (inner, t, float(strength))


def solve_modes(
    mesh: Mesh,
    wavelength: float,
    num_modes: int = 2,
    index=None,
    eps=None,
    n_guess: float | None = None,
    radius: float | None = None,
    bend_axis_x: float = 0.0,
    pml=None,
    pml_strength: float = 3.0,
    boundary="pec",
    window=None,
    grid: int = 120,
    tol: float = 1e-10,
    keep_pml_modes: bool = False,
) -> list[FemMode]:
    """Waveguide modes of a cross-section mesh.

    Parameters
    ----------
    mesh : Mesh
        Cross-section mesh (x horizontal, y vertical, propagation along z).
    wavelength : float
        Vacuum wavelength (m).
    num_modes : int
        Number of modes, highest effective index first (near ``n_guess``).
    index, eps
        Materials (see :func:`eps_from`); default from the material library.
    n_guess : float, optional
        Effective index the eigen-solver targets; default 1.05 times the
        largest index (the modes of highest index come first).
    radius : float, optional
        Bend radius (m); the bend centre lies at ``x = bend_axis_x -
        radius``, so positive x is the outside of the bend. Give the mesh a
        PML on the outer side to see the radiation (bend) loss.
    pml : float, sequence or dict, optional
        Thickness of perfectly matched layers inside the mesh boundary:
        one value for all sides, ``(left, bottom, right, top)``, or a dict.
        ``pml_strength`` is the imaginary stretch at the outer edge.
    boundary : str, sequence or callable
        Outer boundary: ``"pec"`` (electric wall, default), ``"pmc"``
        (magnetic wall, the natural condition), or the sides that are PEC
        (``["left", "right"]``, a function of edge midpoints).
    window : (x0, y0, x1, y1), optional
        Region of the grid samples ``Mode.E`` / ``Mode.H`` (default: the
        mesh without its PML); ``grid`` points along the longer side.

    keep_pml_modes : bool
        With a PML, modes with more than half of ``int |E|^2`` inside the
        layers (PML and radiation modes) are dropped unless this is True;
        a bend makes the outer cladding look like a high index, so give
        ``n_guess`` near the expected index (e.g. the straight guide's).

    Returns a list of :class:`FemMode`, sorted by decreasing Re(neff)
    (possibly fewer than ``num_modes`` when PML modes were dropped).
    """
    from .. import _core

    e4 = np.ascontiguousarray(eps_from(mesh, wavelength, index, eps), complex)
    k0 = 2 * math.pi / wavelength
    if n_guess is None:
        n_guess = 1.05 * float(np.sqrt(np.max(np.real(e4[:, [0, 1, 3]]))))
    if boundary == "pec":
        pec = mesh.boundary_edges()
    elif boundary == "pmc" or boundary is None:
        pec = np.zeros(0, np.int64)
    else:
        pec = mesh.boundary_edges(boundary)
    pmls = _pml_spec(mesh, pml, pml_strength)
    bend = (float(radius), float(bend_axis_x)) if radius is not None and np.isfinite(radius) else None
    d = _core.fem_modes(mesh.nodes, mesh.tris, mesh.tags, e4, k0, int(num_modes), float(n_guess), pec.tolist(), pmls, bend, float(tol))
    problem = {"eps": e4, "k0": k0, "pml": pmls, "bend": bend}
    if window is None:
        window = tuple(pmls[0]) if pmls is not None else mesh.bounds
    x0, y0, x1, y1 = window
    nx = grid if (x1 - x0) >= (y1 - y0) else max(8, int(grid * (x1 - x0) / (y1 - y0)))
    ny = grid if (y1 - y0) > (x1 - x0) else max(8, int(grid * (y1 - y0) / (x1 - x0)))
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.ascontiguousarray(np.stack([X.ravel(), Y.ravel()], 1))
    out = []
    freq = C0 / wavelength
    for beta, x, ints in zip(d["beta"], d["x"], d["integrals"]):
        p = ints[0] / ETA0  # power of the raw vector (W)
        if not np.isfinite(p) or p == 0:
            continue
        s = 1 / math.sqrt(abs(p))
        x = x * s
        f = _core.fem_mode_fields(mesh.nodes, mesh.tris, mesh.tags, e4, k0, (beta.real, beta.imag), x, pts, pmls, bend)
        E = np.nan_to_num(f["E"])
        idx = np.argmax(np.abs(E[:, :2]).max(1))
        comp = int(np.argmax(np.abs(E[idx, :2])))
        ph = np.exp(-1j * np.angle(E[idx, comp]))
        x = x * ph
        E = (f["E"] * ph).reshape(nx, ny, 3)
        H = (f["H"] * ph / ETA0).reshape(nx, ny, 3)
        te = ints[1] / (ints[1] + ints[2]) if ints[1] + ints[2] > 0 else 0.5
        ints_n = np.array(ints) * np.array([s * s, s * s, s * s, s * s, s**4, s * s])
        info = {"u_nodes": xs, "v_nodes": ys, "window": (x0, y0, x1, y1), "Ez": E[..., 2], "Hz": H[..., 2], "solver": "fem"}
        out.append(FemMode(beta / k0, freq, te, (E[..., 0], E[..., 1]), (H[..., 0], H[..., 1]), info, mesh=mesh, problem=problem, x=x, beta=beta, ints=ints_n))
    if pmls is not None and not keep_pml_modes:
        x0i, y0i, x1i, y1i = pmls[0]
        kept = []
        for md in out:
            q = md.quadrature(2)
            e2 = np.sum(np.abs(q["E"]) ** 2, 1) * q["weights"]
            p = q["points"]
            inside = (p[:, 0] >= x0i) & (p[:, 0] <= x1i) & (p[:, 1] >= y0i) & (p[:, 1] <= y1i)
            md.pml_fraction = float(np.sum(e2[~inside]) / np.sum(e2))
            if md.pml_fraction < 0.5:
                kept.append(md)
        out = kept
    out.sort(key=lambda m: -m.neff.real)
    return out
