"""Full-vectorial finite-difference eigenmode solver on the lightsim Yee grid.

The cross-section permittivity comes from the Rust core (`plane_info`), i.e.
exactly the sub-pixel-smoothed values the FDTD engine uses, sampled at the
staggered positions of the transverse Yee lattice. The solver discretizes
Maxwell's equations for fields ``~ exp(i(beta*n - w t))`` with the same
staggering (E_u at (u+1/2, v), E_v at (u, v+1/2), E_n at nodes, H dual), so
mode profiles can be injected and projected without interpolation.

Formulation (normalized, H~ = eta0 H, k0 = w/c):
    beta E_t = P H_t,   beta H_t = Q E_t   ->   beta^2 E_t = P Q E_t
solved with shift-invert Arnoldi (scipy). With off-diagonal permittivity the
four transverse components are kept and the linear problem
``beta [E_t; H_t] = M [E_t; H_t]`` is solved instead.

Options on top of the closed (zero-field) window:

* perfectly matched layers by complex coordinate stretching
  (``s = 1 + (kappa - 1) rho^m + i sigma rho^m`` in the outer cells), for
  leaky modes and substrate leakage;
* bent waveguides by the exact conformal map of a cylindrical bend onto a
  straight guide: ``eps' = eps diag(r/R, r/R, R/r)`` and the same for
  ``mu`` (r the distance from the bend centre, R the bend radius);
* diagonal ``mu`` and the full permittivity tensor.

The field arrays use the staggering of the simulation grid in the window's
natural axes (u, v): E_u at (u+1/2, v), E_v at (u, v+1/2), E_n at (u, v),
H_u at (u, v+1/2), H_v at (u+1/2, v), H_n at (u+1/2, v+1/2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C0 = 299_792_458.0
ETA0 = 376.730313412
EPS0 = 8.8541878128e-12


@dataclass
class Mode:
    """A waveguide mode returned by :func:`solve_modes` (or by
    :meth:`lightsim.xsection.CrossSection.solve`).

    Attributes
    ----------
    neff : complex
        Effective index; a positive imaginary part is loss. For a bent
        guide it refers to the bend radius: the field goes as
        ``exp(i k0 neff R phi)``.
    freq : float
        Frequency (Hz) the mode was solved at.
    te_fraction : float
        Share of the transverse electric field energy in the first in-plane
        axis, between 0 and 1.
    profile : dict
        The field pattern in the form ``ModeSource`` and ``ModeMonitor`` take,
        normalised to unit power.
    E : tuple
        ``(E_u, E_v)``: the two transverse electric components, complex
        arrays over the mode window (V/m for unit power).
    H : tuple
        ``(H_u, H_v)`` in A/m, at the plane of ``E``.
    info : dict
        The mode window: ``u_nodes``, ``v_nodes`` (m), ``dims``, ``axes``,
        ``origin``, the permittivity maps ``eps_u``, ``eps_v``, ``eps_n``
        (and ``eps_*_im``), ``coord`` and ``h_offset``. Modes from a
        :class:`~lightsim.xsection.CrossSection` also carry ``fill`` (the
        fill fraction of every material at the three electric positions)
        and ``eps_material``.
    E_n, H_n : ndarray or None
        Longitudinal components (E at the nodes, H at the cell centres), or
        ``None`` if the solver that made the mode did not compute them.
    n_group : float or None
        Group index, when it was asked for (``group_index=True``).
    bend_radius : float or None
        Bend radius (m) the mode was solved for; ``None`` for a straight
        guide.
    """

    neff: complex
    freq: float
    te_fraction: float
    profile: dict  # ModeProfile dict for lightsim sources/monitors
    E: tuple  # (E_u, E_v) complex arrays on the plane (natural order)
    H: tuple  # (H_u, H_v) complex arrays (A/m) at the E plane
    info: dict
    E_n: np.ndarray | None = None
    H_n: np.ndarray | None = None
    n_group: float | None = None
    bend_radius: float | None = None

    def __repr__(self) -> str:
        ng = f" ng={self.n_group:.4f}" if self.n_group is not None else ""
        return f"<Mode neff={self.neff.real:.5f}{self.neff.imag:+.2e}j TE={self.te_fraction:.2f}{ng}>"

    # -- scalar quantities -------------------------------------------------
    @property
    def wavelength(self) -> float:
        """Vacuum wavelength (m)."""
        return C0 / self.freq

    @property
    def beta(self) -> complex:
        """Propagation constant ``k0 * neff`` (1/m)."""
        return 2 * np.pi * self.freq / C0 * self.neff

    @property
    def fraction_te(self) -> float:
        """Same as ``te_fraction`` (gplugins / Tidy3D name)."""
        return self.te_fraction

    @property
    def fraction_tm(self) -> float:
        """``1 - te_fraction``."""
        return 1.0 - self.te_fraction

    @property
    def alpha(self) -> float:
        """Power attenuation coefficient ``2 k0 Im(neff)`` (1/m)."""
        return 2 * 2 * np.pi * self.freq / C0 * float(np.imag(self.neff))

    @property
    def loss_db_per_cm(self) -> float:
        """Propagation loss in dB/cm, ``10 log10(e) * alpha / 100``. For a
        bend, per cm of path along the bend radius."""
        return 10 * np.log10(np.e) * self.alpha / 100.0

    def loss_db_per_90deg(self) -> float:
        """Loss of a 90-degree turn (dB), for a mode solved with
        ``bend_radius``."""
        if not self.bend_radius:
            raise ValueError("the mode was not solved for a bend (bend_radius is None)")
        return 10 * np.log10(np.e) * self.alpha * np.pi / 2 * abs(self.bend_radius)

    # -- field-derived quantities ------------------------------------------
    def effective_area(self) -> float:
        """Effective mode area ``(∫|E|² dA)² / ∫|E|⁴ dA`` (m²), with all
        three electric components (the definition Tidy3D's ``mode_area``
        uses)."""
        return mode_effective_area(self)

    def energy_fraction(self, material=None, region=None) -> float:
        """Share of the electric energy ``∫ eps |E|² dA`` inside a material
        (name, for cross-section modes) or a rectangular ``region``
        ``((u0, v0), (u1, v1))`` in metres. See :func:`mode_confinement`."""
        return mode_confinement(self, material=material, region=region, kind="energy")

    def confinement(self, material=None, region=None, kind: str = "energy") -> float:
        """Confinement factor in a material or region; ``kind`` is
        ``"energy"`` (electric energy) or ``"power"`` (longitudinal Poynting
        flux). See :func:`mode_confinement`."""
        return mode_confinement(self, material=material, region=region, kind=kind)

    def power_in(self, region) -> float:
        """Share of the mode power (Poynting flux through the plane) inside
        the rectangle ``region = ((u0, v0), (u1, v1))`` (m)."""
        return mode_confinement(self, region=region, kind="power")

    def overlap(self, other: "Mode", conjugate: bool = True) -> complex:
        """Overlap with another mode, see :func:`mode_overlap`."""
        return mode_overlap(self, other, conjugate=conjugate)

    def fields_at_centres(self):
        """All six components averaged to the cell centres: a dict with
        ``u``, ``v`` (centre coordinates, m) and ``Eu, Ev, En, Hu, Hv, Hn``
        (``En``/``Hn`` zero if not computed)."""
        return _fields_at_centres(self)


# ----------------------------------------------------------------------------
# discretisation
# ----------------------------------------------------------------------------


def _diff_ops(nodes_x, nodes_y, sx=None, sy=None, periodic=(False, False)):
    """Forward (E->H) and backward (H->E) difference operators on an Nx x Ny lattice.

    ``sx``/``sy``: optional complex stretch factors ``(at_centres, at_nodes)``
    (perfectly matched layers). ``periodic``: wrap the lattice along x / y.
    """
    dxp = np.diff(nodes_x)
    dyp = np.diff(nodes_y)
    Nx, Ny = len(dxp), len(dyp)
    px, py = periodic

    def dual(dp, per):
        dd = np.empty(len(dp))
        dd[0] = 0.5 * (dp[-1] + dp[0]) if per else dp[0]
        dd[1:] = 0.5 * (dp[:-1] + dp[1:])
        return dd

    dxd, dyd = dual(dxp, px), dual(dyp, py)
    sxc, sxn = sx if sx is not None else (1.0, 1.0)
    syc, syn = sy if sy is not None else (1.0, 1.0)

    def fwd(dp, s, per):
        n = len(dp)
        h = dp * s
        m = sp.diags([-1.0 / h, 1.0 / h[:-1]], [0, 1], shape=(n, n), format="lil")
        if per and n > 1:
            m[n - 1, 0] = 1.0 / h[n - 1]
        elif per:  # one cell: f[i+1] = f[i]
            m = sp.lil_matrix((1, 1))
        return m.tocsr()

    def bwd(dd, s, per):
        n = len(dd)
        h = dd * s
        m = sp.diags([1.0 / h, -1.0 / h[1:]], [0, -1], shape=(n, n), format="lil")
        if per and n > 1:
            m[0, n - 1] = -1.0 / h[0]
        elif per:
            m = sp.lil_matrix((1, 1))
        return m.tocsr()

    fx, bx = fwd(dxp, sxc, px), bwd(dxd, sxn, px)
    fy, by = fwd(dyp, syc, py), bwd(dyd, syn, py)
    Ix = sp.identity(Nx, format="csr")
    Iy = sp.identity(Ny, format="csr")
    # 1-D forward: (f[i+1]-f[i])/dxp[i], f[N]=0 ; backward: (g[i]-g[i-1])/dxd[i], g[-1]=0
    Dxe = sp.kron(fx, Iy, format="csr")
    Dxh = sp.kron(bx, Iy, format="csr")
    Dye = sp.kron(Ix, fy, format="csr")
    Dyh = sp.kron(Ix, by, format="csr")
    return Dxe, Dxh, Dye, Dyh, dxp, dxd, dyp, dyd


def _avg_ops(Nx, Ny, periodic=(False, False)):
    """Averaging operators between the staggered positions (for off-diagonal
    permittivity): x-centres <-> x-nodes and y-centres <-> y-nodes."""

    def c2n(n, per):  # value at node i from centres i-1/2, i+1/2
        if per and n == 1:
            return sp.identity(1, format="csr")
        m = sp.diags([0.5 * np.ones(n), 0.5 * np.ones(n - 1)], [0, -1], shape=(n, n), format="lil")
        if per:
            m[0, n - 1] = 0.5
        return m.tocsr()

    def n2c(n, per):  # value at centre i+1/2 from nodes i, i+1
        if per and n == 1:
            return sp.identity(1, format="csr")
        m = sp.diags([0.5 * np.ones(n), 0.5 * np.ones(n - 1)], [0, 1], shape=(n, n), format="lil")
        if per:
            m[n - 1, 0] = 0.5
        return m.tocsr()

    Ix, Iy = sp.identity(Nx, format="csr"), sp.identity(Ny, format="csr")
    return {
        "x_c2n": sp.kron(c2n(Nx, periodic[0]), Iy, format="csr"),
        "x_n2c": sp.kron(n2c(Nx, periodic[0]), Iy, format="csr"),
        "y_c2n": sp.kron(Ix, c2n(Ny, periodic[1]), format="csr"),
        "y_n2c": sp.kron(Ix, n2c(Ny, periodic[1]), format="csr"),
    }


def pml_stretch(nodes, cells_lo: int, cells_hi: int, freq: float, strength: float = 4.0, kappa: float = 1.0, order: float = 2.0):
    """Complex stretch factors of a perfectly matched layer along one axis.

    The layer covers the outer ``cells_lo`` / ``cells_hi`` cells of
    ``nodes``. With the depth ``rho`` (0 at the inner edge of the layer, 1
    at the wall) the factor is ``s = 1 + (kappa - 1) rho^order + i strength
    rho^order``; a wave ``exp(i k x)`` is damped by ``exp(-k Im ∫(s-1) dx)``,
    i.e. by ``exp(-k * strength * thickness / (order + 1))`` in one pass.

    Returns ``(s_at_centres, s_at_nodes)`` for the cell centres and for the
    nodes ``nodes[:-1]`` (the last node is the closed wall).
    """
    nodes = np.asarray(nodes, dtype=float)
    cen = 0.5 * (nodes[:-1] + nodes[1:])
    nod = nodes[:-1]
    N = len(nodes) - 1

    def prof(x):
        rho = np.zeros_like(x)
        if cells_lo > 0:
            x0 = nodes[cells_lo]
            L = x0 - nodes[0]
            rho = np.maximum(rho, np.clip((x0 - x) / L, 0.0, 1.0))
        if cells_hi > 0:
            x1 = nodes[N - cells_hi]
            L = nodes[N] - x1
            rho = np.maximum(rho, np.clip((x - x1) / L, 0.0, 1.0))
        r = rho**order
        return 1.0 + (kappa - 1.0) * r + 1j * strength * r

    return prof(cen), prof(nod)


def _pml_cells(pml, swap: bool = False):
    """Normalise a ``pml`` argument to (cells u_lo, u_hi, v_lo, v_hi), strength, kappa, order."""
    if pml is None or pml is False:
        return None
    if pml is True:
        pml = 12
    if isinstance(pml, (int, np.integer)):
        cells = (int(pml),) * 4
        opts = {}
    elif isinstance(pml, dict):
        c = pml.get("cells", 12)
        cells = (int(c),) * 4 if isinstance(c, (int, np.integer)) else tuple(int(v) for v in c)
        opts = {k: pml[k] for k in ("strength", "kappa", "order") if k in pml}
    else:
        cells = tuple(int(v) for v in pml)
        opts = {}
    if len(cells) != 4:
        raise ValueError("pml cells must be one number or four (u_lo, u_hi, v_lo, v_hi)")
    if swap:
        cells = (cells[2], cells[3], cells[0], cells[1])
    return cells, float(opts.get("strength", 4.0)), float(opts.get("kappa", 1.0)), float(opts.get("order", 2.0))


def _positions(nodes_x, nodes_y):
    """Coordinates of the staggered positions: dict name -> (X, Y) arrays (Nx, Ny)."""
    xn, yn = np.asarray(nodes_x, float), np.asarray(nodes_y, float)
    xc, yc = 0.5 * (xn[:-1] + xn[1:]), 0.5 * (yn[:-1] + yn[1:])
    xn, yn = xn[:-1], yn[:-1]
    g = lambda a, b: np.meshgrid(a, b, indexing="ij")  # noqa: E731
    return {"ex": g(xc, yn), "ey": g(xn, yc), "ez": g(xn, yn), "hx": g(xn, yc), "hy": g(xc, yn), "hz": g(xc, yc)}


def bend_transform(eps, mu, eps_off, nodes_x, nodes_y, radius: float, bend_axis: int = 0, center: float = 0.0):
    """Material of the straight guide that is equivalent to a bend.

    A bend of radius ``radius`` about an axis parallel to the solver's other
    transverse axis maps exactly onto a straight guide with
    ``eps' = eps diag(r/R, r/R, R/r)`` and ``mu' = mu diag(r/R, r/R, R/r)``,
    ``r = R + (x - center)`` the distance from the centre of curvature,
    ``x`` the coordinate along ``bend_axis`` (0 or 1) of the solver
    lattice. A positive radius puts the centre of curvature on the low side.
    Off-diagonal terms scale with ``(r/R) / (h_i h_j)``.
    """
    R = float(radius)
    pos = _positions(nodes_x, nodes_y)

    def ratio(name):
        X = pos[name][bend_axis]
        r = R + (X - center)
        if np.any(r <= 0):
            raise ValueError("the bend centre lies inside the mode window; use a larger radius or a smaller window")
        return r / R

    ex, ey, ez = eps
    eps_b = (ex * ratio("ex"), ey * ratio("ey"), ez / ratio("ez"))
    mx, my, mz = mu if mu is not None else (1.0, 1.0, 1.0)
    mu_b = (mx * ratio("hx"), my * ratio("hy"), mz / ratio("hz"))
    off_b = None
    if eps_off:
        rowpos = {"xy": "ex", "xz": "ex", "yx": "ey", "yz": "ey", "zx": "ez", "zy": "ez"}
        off_b = {}
        for k, v in eps_off.items():
            q = ratio(rowpos[k])
            # h = (1, 1, q): eps'_ij = eps_ij q / (h_i h_j)
            hz = sum(1 for c in k if c == "z")
            off_b[k] = v * q / q**hz
    return eps_b, mu_b, off_b


_FACTOR_CACHE: dict = {}


def _array_key(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        if isinstance(a, dict):
            for k in sorted(a):
                h.update(k.encode())
                h.update(np.ascontiguousarray(a[k]).tobytes())
            continue
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def clear_cache() -> None:
    """Forget the cached mode solutions (see ``cache=True`` in the solvers)."""
    _FACTOR_CACHE.clear()


def solve_fields(
    eps,
    nodes_x,
    nodes_y,
    freq: float,
    num_modes: int = 2,
    target_neff: float | None = None,
    *,
    mu=None,
    eps_off=None,
    stretch=None,
    periodic=(False, False),
    tol: float = 1e-10,
    cache: bool = False,
    timings: dict | None = None,
):
    """Core eigen-solve on the solver lattice (right-handed x, y, propagation).

    Parameters
    ----------
    eps : (eps_x, eps_y, eps_z)
        Complex (Nx, Ny) arrays at the E_x, E_y, E_z positions.
    mu : (mu_x, mu_y, mu_z), optional
        Diagonal permeability at the H_x, H_y, H_z positions (default 1).
    eps_off : dict, optional
        Off-diagonal permittivity: keys ``"xy", "xz"`` (arrays at the E_x
        positions), ``"yx", "yz"`` (at E_y), ``"zx", "zy"`` (at E_z). When
        given, the full four-component problem is solved.
    stretch : ((sx_c, sx_n), (sy_c, sy_n)), optional
        PML stretch factors from :func:`pml_stretch` (``None`` per axis for
        none).
    periodic : (bool, bool)
        Periodic lattice along x / y instead of closed walls.
    cache : bool
        Reuse the result of an identical earlier call (same arrays and
        settings) instead of solving again.
    timings : dict, optional
        Filled with the seconds spent assembling, factorising and iterating.

    Returns a list (highest real effective index first) of dicts with
    ``neff`` and the six field components ``Ex, Ey, Ez, Hx, Hy, Hz``
    (H normalised as ``eta0 H``), normalised to unit power, and the areas
    ``a_ex``, ``a_ey`` used for the power.
    """
    import time

    t0 = time.perf_counter()
    ex, ey, ez = (np.asarray(e, dtype=complex) for e in eps)
    Nx, Ny = ex.shape
    N = Nx * Ny
    k0 = 2 * np.pi * freq / C0
    if mu is None:
        mx = my = mz = np.ones((Nx, Ny), dtype=complex)
    else:
        mx, my, mz = (np.broadcast_to(np.asarray(m, dtype=complex), (Nx, Ny)) for m in mu)
    eps_off = {k: np.asarray(v, dtype=complex) for k, v in (eps_off or {}).items() if np.any(np.asarray(v) != 0)}
    nmax = float(np.sqrt(np.max(np.real(np.concatenate([ex.ravel(), ey.ravel()]) * np.concatenate([my.ravel(), mx.ravel()])))))
    tn = target_neff if target_neff is not None else nmax
    key = None
    if cache:
        st = [] if stretch is None else [np.concatenate([np.atleast_1d(a) for a in ax]) if ax is not None else np.zeros(1) for ax in stretch]
        key = _array_key(ex, ey, ez, mx, my, mz, eps_off, np.asarray(nodes_x), np.asarray(nodes_y), np.asarray([freq, num_modes, tn, tol]), np.asarray(periodic), *st)
        if key in _FACTOR_CACHE:
            if timings is not None:
                timings.update(assemble=0.0, factor=0.0, iterate=0.0, cached=True)
            return [dict(m) for m in _FACTOR_CACHE[key]]
    sx, sy = stretch if stretch is not None else (None, None)
    Dxe, Dxh, Dye, Dyh, dxp, dxd, dyp, dyd = _diff_ops(nodes_x, nodes_y, sx, sy, periodic)
    D = sp.diags
    iez = D(1.0 / ez.ravel())
    imz = D(1.0 / mz.ravel())
    MX, MY = D(mx.ravel()), D(my.ravel())
    EX, EY = D(ex.ravel()), D(ey.ravel())
    general = bool(eps_off)
    if not general:
        # A = P Q with P = P_mu + P_d (material and derivative parts) and
        # Q = Q_eps + Q_d. P_d Q_d is identically zero (the difference
        # operators along x and along y commute), so A is assembled as
        # P_mu Q + P_d Q_eps: the same matrix without the structurally
        # present zeros of P_d Q_d, which would make the factorisation
        # fill in about 1.5 times as much
        Q = sp.bmat(
            [
                [Dxh @ imz @ Dye, -(k0**2) * EY - Dxh @ imz @ Dxe],
                [k0**2 * EX + Dyh @ imz @ Dye, -Dyh @ imz @ Dxe],
            ],
            format="csc",
        ) / k0
        P_mu = sp.bmat([[None, k0**2 * MY], [-(k0**2) * MX, None]], format="csr") / k0
        P_d = sp.bmat([[-Dxe @ iez @ Dyh, Dxe @ iez @ Dxh], [-Dye @ iez @ Dyh, Dye @ iez @ Dxh]], format="csr") / k0
        Q_eps = sp.bmat([[None, -(k0**2) * EY], [k0**2 * EX, None]], format="csr") / k0
        A = (P_mu @ Q + P_d @ Q_eps).tocsc()
        sigma = (k0 * tn) ** 2
        dim = 2 * N
    else:
        av = _avg_ops(Nx, Ny, periodic)
        z = sp.csr_matrix((N, N))
        off = lambda k: D(eps_off[k].ravel()) if k in eps_off else None  # noqa: E731
        # E_z = iez [ (i/k0)(Dxh Hy - Dyh Hx) - eps_zx <Ex>_n - eps_zy <Ey>_n ]
        Lx = -iez @ off("zx") @ av["x_c2n"] if "zx" in eps_off else z
        Ly = -iez @ off("zy") @ av["y_c2n"] if "zy" in eps_off else z
        Mx = -(1j / k0) * iez @ Dyh
        My = (1j / k0) * iez @ Dxh
        # H_z = (Dxe Ey - Dye Ex) / (i k0 mu_z)
        Kx = (1j / k0) * imz @ Dye
        Ky = -(1j / k0) * imz @ Dxe
        # (eps E)_x at E_x: eps_xx Ex + eps_xy <Ey> + eps_xz <Ez>
        Axy = off("xy") @ av["x_n2c"] @ av["y_c2n"] if "xy" in eps_off else z
        Axz = off("xz") @ av["x_n2c"] if "xz" in eps_off else None
        Ayx = off("yx") @ av["x_c2n"] @ av["y_n2c"] if "yx" in eps_off else z
        Ayz = off("yz") @ av["y_n2c"] if "yz" in eps_off else None
        ez_ex, ez_ey, ez_hx, ez_hy = Lx, Ly, Mx, My
        # (eps E)_x and (eps E)_y as rows over [Ex, Ey, Hx, Hy]
        dx_row = [EX + (Axz @ ez_ex if Axz is not None else z), Axy + (Axz @ ez_ey if Axz is not None else z), (Axz @ ez_hx if Axz is not None else z), (Axz @ ez_hy if Axz is not None else z)]
        dy_row = [Ayx + (Ayz @ ez_ex if Ayz is not None else z), EY + (Ayz @ ez_ey if Ayz is not None else z), (Ayz @ ez_hx if Ayz is not None else z), (Ayz @ ez_hy if Ayz is not None else z)]
        # beta Ex = k0 mu_y Hy - i Dxe Ez ; beta Ey = -i Dye Ez - k0 mu_x Hx
        r_ex = [-1j * Dxe @ ez_ex, -1j * Dxe @ ez_ey, -1j * Dxe @ ez_hx, k0 * MY - 1j * Dxe @ ez_hy]
        r_ey = [-1j * Dye @ ez_ex, -1j * Dye @ ez_ey, -k0 * MX - 1j * Dye @ ez_hx, -1j * Dye @ ez_hy]
        # beta Hx = -i Dxh Hz - k0 (eps E)_y ; beta Hy = -i Dyh Hz + k0 (eps E)_x
        r_hx = [-1j * Dxh @ Kx - k0 * dy_row[0], -1j * Dxh @ Ky - k0 * dy_row[1], -k0 * dy_row[2], -k0 * dy_row[3]]
        r_hy = [-1j * Dyh @ Kx + k0 * dx_row[0], -1j * Dyh @ Ky + k0 * dx_row[1], k0 * dx_row[2], k0 * dx_row[3]]
        A = sp.bmat([r_ex, r_ey, r_hx, r_hy], format="csc")
        sigma = k0 * tn
        dim = 4 * N
    t1 = time.perf_counter()
    k = min(num_modes, dim - 2)
    # lossless media without PML give a real matrix: factorise in real
    # arithmetic (about three times faster than complex)
    if not np.iscomplexobj(A.data) or not np.any(np.imag(A.data)):
        A = sp.csc_matrix((np.real(A.data), A.indices, A.indptr), shape=A.shape) if np.iscomplexobj(A.data) else A
    lu = None
    if dim <= 600:  # small problems (1-D slabs, uniform sections): dense
        k = min(num_modes, dim)
        w, v = np.linalg.eig(A.toarray())
        pick = np.argsort(np.abs(w - sigma))[:k]
        vals, vecs = w[pick], v[:, pick]
        t2 = t3 = time.perf_counter()
    else:
        shifted = (A - sigma * sp.identity(dim, format="csc")).tocsc()
        lu = spla.splu(shifted, permc_spec="COLAMD")
        t2 = time.perf_counter()
        op = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=A.dtype)
        ncv = min(dim - 1, max(2 * k + 1, k + 16))
        # a fixed start vector: ARPACK otherwise draws a random one per call,
        # and the modes then differ in the last digits from run to run
        v0 = np.random.default_rng(0).standard_normal(dim).astype(A.dtype)
        vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM", OPinv=op, tol=tol, ncv=ncv, v0=v0)
        t3 = time.perf_counter()
    if general:
        beta = vals.astype(complex)
        order = np.argsort(-beta.real)
        beta, vecs = beta[order], vecs[:, order]
        Exs, Eys, Hxs, Hys = vecs[:N], vecs[N : 2 * N], vecs[2 * N : 3 * N], vecs[3 * N :]
    else:
        beta = np.sqrt(vals.astype(complex))
        # forward modes: decaying along +n (Im beta > 0); among propagating
        # modes (Im beta = 0) the ones with Re beta > 0. Complex modes of a
        # lossless guide come in conjugate pairs of beta^2, whose decaying
        # roots are x + iy and -x + iy.
        tiny = 1e-9 * np.abs(beta)
        flip = (beta.imag < -tiny) | ((np.abs(beta.imag) <= tiny) & (beta.real < 0))
        beta = np.where(flip, -beta, beta)
        order = np.argsort(-beta.real)
        beta = beta[order]
        vecs = vecs[:, order]
        Hs = (Q @ vecs) / beta[None, :]
        Exs, Eys, Hxs, Hys = vecs[:N], vecs[N:], Hs[:N], Hs[N:]
    # longitudinal components
    Ezs = (1j / k0) * (iez @ (Dxh @ Hys - Dyh @ Hxs))
    if general:
        Ezs = Lx @ Exs + Ly @ Eys + Ezs
    Hzs = (imz @ (Dxe @ Eys - Dye @ Exs)) / (1j * k0)
    a_ex = np.outer(dxp, dyd)
    a_ey = np.outer(dxd, dyp)
    out = []
    for m in range(len(beta)):
        f = [c[:, m].reshape(Nx, Ny) for c in (Exs, Eys, Ezs, Hxs, Hys, Hzs)]
        p = 0.5 / ETA0 * np.real(np.sum(f[0] * np.conj(f[4]) * a_ex) - np.sum(f[1] * np.conj(f[3]) * a_ey))
        s = 1.0 / np.sqrt(abs(p)) if p != 0 else 1.0
        if p < 0:  # flip propagation sign convention (shouldn't happen for beta>0)
            s = -s
        f = [c * s for c in f]
        # fix the global phase: largest E component real positive
        big = f[0] if np.max(np.abs(f[0])) >= np.max(np.abs(f[1])) else f[1]
        idx = np.unravel_index(np.argmax(np.abs(big)), big.shape)
        ph = np.exp(-1j * np.angle(big[idx]))
        f = [c * ph for c in f]
        out.append({"neff": beta[m] / k0, "Ex": f[0], "Ey": f[1], "Ez": f[2], "Hx": f[3], "Hy": f[4], "Hz": f[5], "a_ex": a_ex, "a_ey": a_ey})
    if timings is not None:
        timings.update(assemble=t1 - t0, factor=t2 - t1, iterate=t3 - t2, cached=False, unknowns=dim, nnz=int(A.nnz), lu_nnz=int(lu.L.nnz + lu.U.nnz) if lu is not None else 0)
    if key is not None:
        _FACTOR_CACHE[key] = [dict(m) for m in out]
    return out


def solve_cross_section(
    eps_x,
    eps_y,
    eps_z,
    nodes_x,
    nodes_y,
    freq: float,
    num_modes: int = 2,
    target_neff: float | None = None,
    **options,
):
    """Solve the transverse eigenproblem on a (right-handed x, y, propagation)
    cross-section. ``eps_*`` are (Nx, Ny) arrays at the Yee positions.
    Returns a list of ``(neff, Ex, Ey, Hx, Hy, a_ex, a_ey)`` with H
    normalized (H~ = eta0 H).

    Keyword ``options`` are passed to :func:`solve_fields` (``mu``,
    ``eps_off``, ``stretch``, ``periodic``, ``cache``, ``timings``).
    """
    res = solve_fields((eps_x, eps_y, eps_z), nodes_x, nodes_y, freq, num_modes, target_neff, **options)
    return [(m["neff"], m["Ex"], m["Ey"], m["Hx"], m["Hy"], m["a_ex"], m["a_ey"]) for m in res]


def _profile(Eu, Ev, Hu_a, Hv_a, neff, freq, dims, h_offset):
    k0 = 2 * np.pi * freq / C0
    shift = np.exp(-1j * neff * k0 * h_offset)  # H plane half a cell upstream
    hu_p = Hu_a * shift
    hv_p = Hv_a * shift
    return {
        "neff": float(np.real(neff)),
        "neff_imag": float(np.imag(neff)),
        "dims": [int(dims[0]), int(dims[1])],
        "e_re": [np.real(Eu).ravel().tolist(), np.real(Ev).ravel().tolist()],
        "e_im": [np.imag(Eu).ravel().tolist(), np.imag(Ev).ravel().tolist()],
        "h_re": [np.real(hu_p).ravel().tolist(), np.real(hv_p).ravel().tolist()],
        "h_im": [np.imag(hu_p).ravel().tolist(), np.imag(hv_p).ravel().tolist()],
        "freq": float(freq),
        "h_offset": float(h_offset),
    }


def solve_plane(
    eps_u,
    eps_v,
    eps_n,
    u_nodes,
    v_nodes,
    freq: float,
    num_modes: int = 2,
    target_neff: float | None = None,
    *,
    swap: bool = False,
    info: dict | None = None,
    pml=None,
    bend_radius: float | None = None,
    bend_axis: str = "u",
    bend_center: float | None = None,
    eps_off: dict | None = None,
    periodic=(False, False),
    sort: str = "neff",
    cache: bool = False,
    timings: dict | None = None,
    profile: bool = True,
):
    """Solve the modes of permittivity maps given in a plane's natural axes
    (u, v) and return :class:`Mode` objects.

    ``eps_u``, ``eps_v``, ``eps_n`` are complex (Nu, Nv) arrays at the E_u,
    E_v and E_n positions; ``swap`` is true when (u, v, normal) is
    left-handed (planes normal to y), in which case the solver works in
    (v, u). ``eps_off`` holds off-diagonal terms in natural axes, keys
    ``"uv", "un", "vu", "vn", "nu", "nv"`` at the positions of the row
    component. The other options are those of :func:`solve_modes`.
    """
    un, vn = np.asarray(u_nodes, float), np.asarray(v_nodes, float)
    eu, ev, en = (np.asarray(a, dtype=complex) for a in (eps_u, eps_v, eps_n))
    info = dict(info or {})
    # natural -> solver axes
    if swap:
        e_sol = (ev.T, eu.T, en.T)
        nx, ny = vn, un
    else:
        e_sol = (eu, ev, en)
        nx, ny = un, vn
    off_sol = None
    if eps_off:
        nat2sol = {"u": "y", "v": "x", "n": "z"} if swap else {"u": "x", "v": "y", "n": "z"}
        off_sol = {}
        for k_, v_ in eps_off.items():
            kk = nat2sol[k_[0]] + nat2sol[k_[1]]
            off_sol[kk] = np.asarray(v_, dtype=complex).T if swap else np.asarray(v_, dtype=complex)
    nmax = float(np.sqrt(np.max(np.real(np.concatenate([eu.ravel(), ev.ravel()])))))
    tn = target_neff if target_neff is not None else nmax
    mu = None
    if bend_radius:
        ax_nat = {"u": 0, "v": 1}[bend_axis]
        ax_sol = (1 - ax_nat) if swap else ax_nat
        nodes_nat = un if ax_nat == 0 else vn
        c = bend_center if bend_center is not None else 0.5 * (nodes_nat[0] + nodes_nat[-1])
        e_sol, mu, off_sol = bend_transform(e_sol, None, off_sol, nx, ny, float(bend_radius), ax_sol, c)
    pc = _pml_cells(pml, swap)
    stretch = None
    if pc is not None:
        (cxl, cxh, cyl, cyh), strength, kappa, order = pc
        sx = pml_stretch(nx, cxl, cxh, freq, strength, kappa, order) if (cxl or cxh) else None
        sy = pml_stretch(ny, cyl, cyh, freq, strength, kappa, order) if (cyl or cyh) else None
        stretch = (sx, sy)
    per = (periodic[1], periodic[0]) if swap else tuple(periodic)
    if bend_radius and target_neff is None:
        # search next to the modes of the straight guide: the outer cladding
        # of the mapped bend has a higher index than the core further out
        e_str = (ev.T, eu.T, en.T) if swap else (eu, ev, en)
        straight = solve_fields(e_str, nx, ny, freq, max(1, num_modes), nmax, eps_off=None, stretch=stretch, periodic=per)
        tn = float(np.real(straight[0]["neff"])) * 1.01
    if bend_radius:
        # the conformal map raises the index of the outer cladding, which
        # carries radiation modes with a large real part: solve more modes
        # and keep the least lossy ones (the guided modes of the bend)
        if pml is None:
            import warnings

            warnings.warn("bend_radius without pml: the outer wall reflects the radiation, so the bend loss is not computed", stacklevel=3)
        nb = max(3 * num_modes, num_modes + 6)
        res = solve_fields(e_sol, nx, ny, freq, nb, tn, mu=mu, eps_off=off_sol, stretch=stretch, periodic=per, cache=cache, timings=timings)
        # keep the modes nearest the target (the straight guide's index): the
        # guided bend modes, not the lossy radiation of the outer cladding
        # nor modes trapped against the inner wall
        res = sorted(res, key=lambda r: abs(r["neff"] - tn))[:num_modes]
        res.sort(key=lambda r: abs(np.imag(r["neff"])))
    else:
        res = solve_fields(e_sol, nx, ny, freq, num_modes, tn, mu=mu, eps_off=off_sol, stretch=stretch, periodic=per, cache=cache, timings=timings)
    dims = (len(un) - 1, len(vn) - 1)
    h_offset = float(info.get("h_offset", 0.0))
    info.setdefault("u_nodes", un)
    info.setdefault("v_nodes", vn)
    info.setdefault("dims", list(dims))
    for name, arr in (("eps_u", eu), ("eps_v", ev), ("eps_n", en)):
        info.setdefault(name, np.real(arr))
        info.setdefault(name + "_im", np.imag(arr))
    info["solver"] = {"pml": pml, "bend_radius": bend_radius, "bend_axis": bend_axis, "bend_center": bend_center, "periodic": tuple(periodic), "swap": swap, "anisotropic": bool(eps_off)}
    modes = []
    for r in res:
        neff = r["neff"]
        if swap:
            Eu, Ev, En = r["Ey"].T, r["Ex"].T, r["Ez"].T
            Hu, Hv, Hn = r["Hy"].T, r["Hx"].T, r["Hz"].T
            a_u, a_v = r["a_ey"].T, r["a_ex"].T
        else:
            Eu, Ev, En = r["Ex"], r["Ey"], r["Ez"]
            Hu, Hv, Hn = r["Hx"], r["Hy"], r["Hz"]
            a_u, a_v = r["a_ex"], r["a_ey"]
        su = np.sum(np.abs(Eu) ** 2 * a_u)
        te = float(su / (su + np.sum(np.abs(Ev) ** 2 * a_v)))
        Hu_a, Hv_a, Hn_a = Hu / ETA0, Hv / ETA0, Hn / ETA0
        prof = _profile(Eu, Ev, Hu_a, Hv_a, neff, freq, dims, h_offset) if profile else {}
        modes.append(Mode(neff=complex(neff), freq=freq, te_fraction=te, profile=prof, E=(Eu, Ev), H=(Hu_a, Hv_a), info=info, E_n=En, H_n=Hn_a, bend_radius=bend_radius))
    if sort == "te":
        modes.sort(key=lambda m: (-round(m.te_fraction), -m.neff.real))
    return modes


def solve_modes(
    sim,
    axis: int,
    position: float,
    min,
    max,
    freq: float,
    num_modes: int = 2,
    target_neff: float | None = None,
    sort: str = "neff",
    *,
    pml=None,
    bend_radius: float | None = None,
    bend_axis: str = "u",
    bend_center: float | None = None,
    group_index: bool = False,
    cache: bool = False,
):
    """Solve waveguide modes on a plane of a :class:`lightsim.Simulation`.

    Returns a list of :class:`Mode` sorted by effective index (or by TE
    fraction first when ``sort="te"``).

    In a 2-D simulation the plane is a line and the modes are those of the
    1-D solver :func:`lightsim.eim.solve_port_mode`, for the polarization(s)
    the simulation contains: ``te_fraction`` is 1 for the modes with E in
    the plane and 0 for those with E along the invariant axis.

    Keyword-only options (3-D simulations):

    pml : int, tuple or dict, optional
        Perfectly matched layers in the outer cells of the window: a number
        of cells on all four sides, four numbers ``(u_lo, u_hi, v_lo, v_hi)``
        or a dict ``{"cells": ..., "strength": 4.0, "kappa": 1.0,
        "order": 2}`` (see :func:`pml_stretch`). Needed for leaky modes,
        substrate leakage and bends. Default: closed walls.
    bend_radius : float, optional
        Solve the modes of a bend of this radius (m). The bend turns in the
        plane of ``bend_axis`` (``"u"``: the first in-plane axis is radial)
        about a centre at ``bend_center - bend_radius`` on that axis
        (``bend_center`` defaults to the middle of the window). ``neff``
        then refers to the radius ``bend_radius``. Use with ``pml``. The
        bend modes nearest the effective index of the straight guide (or
        ``target_neff``) are returned, least lossy first.
    group_index : bool
        Also compute the group index ``n_group`` from the energy-flux
        relation, with the dispersion of the simulation's media.
    cache : bool
        Reuse an identical earlier solve.
    """
    from . import _core

    if getattr(sim, "dimensions", 3) == 2:
        from .eim import solve_port_mode

        pols = ("te", "tm") if sim.polarization == "both" else (sim.polarization,)
        modes = [m for pol in pols for m in solve_port_mode(sim, axis, position, min, max, freq, pol, num_modes=num_modes)]
        modes.sort(key=(lambda m: (-round(m.te_fraction), -m.neff.real)) if sort == "te" else (lambda m: -m.neff.real))
        return modes[:num_modes]

    spec = sim.geometry_json() if hasattr(sim, "geometry_json") else sim
    pargs = (spec, int(axis), float(position), [float(v) for v in min], [float(v) for v in max])
    info = _core.plane_info(*pargs, float(freq))
    # complex permittivity at the mode frequency (lossy / dispersive media)
    eu = info["eps_u"] + 1j * info["eps_u_im"]
    ev = info["eps_v"] + 1j * info["eps_v_im"]
    en = info["eps_n"] + 1j * info["eps_n_im"]
    un, vn = info["u_nodes"], info["v_nodes"]
    swap = axis == 1  # natural (x, z) is left-handed around +y: solve in (z, x)
    modes = solve_plane(eu, ev, en, un, vn, freq, num_modes, target_neff, swap=swap, info=info, pml=pml, bend_radius=bend_radius, bend_axis=bend_axis, bend_center=bend_center, sort=sort, cache=cache)
    if group_index:
        rel = 1e-3
        grp = []
        for s in (-1, 1):
            i2 = _core.plane_info(*pargs, float(freq * (1 + s * rel)))
            grp.append(tuple(i2[k] + 1j * i2[k + "_im"] for k in ("eps_u", "eps_v", "eps_n")))
        w = 2 * np.pi * freq
        dw = 2 * np.pi * freq * rel
        # d(w eps)/dw by central differences
        eps_g = tuple((w + dw) * grp[1][i] / (2 * dw) - (w - dw) * grp[0][i] / (2 * dw) for i in range(3))
        for m in modes:
            m.n_group = group_index_of(m, eps_g)
    return modes


def gaussian_mode(
    sim,
    axis: int,
    position: float,
    min,
    max,
    freq: float,
    waist: float,
    center=None,
    polarization: str = "v",
    n: float = 1.0,
    focus_distance: float = 0.0,
    theta: float = 0.0,
    phi: float = 0.0,
):
    """Gaussian-beam reference profile for :func:`ModeMonitor` / :func:`ModeSource`
    (cf. FDTDX ``GaussianModeOverlapDetector``).

    The beam travels towards +``axis`` in a medium of index ``n``, with waist
    ``waist`` (1/e field radius at the focus, m), sampled on the Yee
    positions of the plane window ``min``..``max`` at ``position``. It is the
    beam of :func:`lightsim.GaussianBeam` with the same parameters:

    Parameters
    ----------
    center : (x, y, z), optional
        Where the beam axis crosses the plane (its two in-plane coordinates
        are used); default: the centre of the window.
    polarization : "u" or "v"
        The electric field points along the plane's first (``"u"``) or second
        (``"v"``) in-plane axis, projected onto the plane normal to the beam
        axis. The in-plane axes are (y, z), (x, z), (x, y) for axis 0, 1, 2.
    focus_distance : float
        Distance (m) from the plane to the focus along the beam axis
        (negative: the focus lies before the plane).
    theta, phi : float
        Polar angle (rad) between the beam axis and +``axis``, and azimuth of
        the tilt from the first in-plane axis towards the second.

    In a simulation that is periodic along an in-plane axis with a period
    below twice the waist (a 2-D simulation), the beam is uniform along that
    axis (a sheet beam), as for the source.

    Returns ``(profile, power)``. ``power`` = 1/2 Re ∫ E × H* dA of the
    profile, so the power a monitor couples into the beam is
    ``abs(a_plus)**2 * power``.
    """
    from . import _core

    spec = sim.geometry_json() if hasattr(sim, "geometry_json") else sim
    info = _core.plane_info(spec, int(axis), float(position), [float(v) for v in min], [float(v) for v in max])
    un, vn = np.asarray(info["u_nodes"]), np.asarray(info["v_nodes"])
    nu, nv = int(info["dims"][0]), int(info["dims"][1])
    uc, vc = 0.5 * (un[:-1] + un[1:]), 0.5 * (vn[:-1] + vn[1:])
    u_ax, v_ax = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]
    c = list(center) if center is not None else [0.5 * (min[i] + max[i]) for i in range(3)]
    k = 2 * np.pi * freq * n / C0
    zr = k * waist**2 / 2
    if polarization not in ("u", "v"):
        raise ValueError("polarization must be 'u' or 'v'")
    if not abs(theta) < np.pi / 2:
        raise ValueError("theta must be below pi/2")
    # beam axis d, polarization e (projected onto the plane normal to d), h = d x e
    d = np.zeros(3)
    d[axis], d[u_ax], d[v_ax] = np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    e = np.zeros(3)
    e[u_ax if polarization == "u" else v_ax] = 1.0
    e = e - np.dot(e, d) * d
    if np.linalg.norm(e) < 1e-6:
        raise ValueError("the beam axis is along the polarization axis")
    e = e / np.linalg.norm(e)
    h = np.cross(d, e)
    # uniform along a periodic in-plane axis that is narrower than the beam (2-D)
    flat = {u_ax: False, v_ax: False}
    if hasattr(sim, "boundaries"):
        for q in (u_ax, v_ax):
            periodic = str(sim.boundaries["xyz"[q]][0].get("type", "")) in ("periodic", "bloch")
            flat[q] = periodic and (float(sim.max[q]) - float(sim.min[q])) < 2 * waist
            if flat[q] and abs(d[q]) > 1e-9:
                raise ValueError(f"the simulation is uniform along axis {q}, so the beam axis must have no component along it")
    sheet = flat[u_ax] or flat[v_ax]

    def g(U, V):
        ru = np.zeros_like(U + V) if flat[u_ax] else (U - c[u_ax]) + 0 * V
        rv = np.zeros_like(U + V) if flat[v_ax] else (V - c[v_ax]) + 0 * U
        along = ru * d[u_ax] + rv * d[v_ax]
        r2 = np.maximum(ru**2 + rv**2 - along**2, 0.0)
        z = along - focus_distance  # along the beam axis, from the focus
        wz = waist * np.sqrt(1 + (z / zr) ** 2)
        gouy = np.arctan(z / zr)
        amp = np.sqrt(waist / wz) if sheet else waist / wz
        # phasors ~ exp(+i k z) in lightsim's DFT convention
        return amp * np.exp(-r2 / wz**2) * np.exp(1j * (k * along + k * r2 * z / (2 * (z**2 + zr**2)) - (0.5 if sheet else 1.0) * gouy))

    at_eu = g(uc[:, None], vn[None, :nv])  # E_u and H_v positions
    at_ev = g(un[:nu, None], vc[None, :])  # E_v and H_u positions
    shift = np.exp(-1j * k * np.cos(theta) * info["h_offset"])  # H plane half a cell upstream
    eu, ev = e[u_ax] * at_eu, e[v_ax] * at_ev
    hu, hv = h[u_ax] * n / ETA0 * at_ev * shift, h[v_ax] * n / ETA0 * at_eu * shift
    area = np.outer(np.diff(un), np.diff(vn))
    handed = -1.0 if axis == 1 else 1.0  # (u, v, normal) is left-handed for axis 1
    power = handed * 0.5 * float(np.real(np.sum((eu * np.conj(hv / shift) - ev * np.conj(hu / shift)) * area)))
    profile = {
        "neff": float(n * np.cos(theta)),
        "neff_imag": 0.0,
        "dims": [nu, nv],
        "e_re": [np.real(eu).ravel().tolist(), np.real(ev).ravel().tolist()],
        "e_im": [np.imag(eu).ravel().tolist(), np.imag(ev).ravel().tolist()],
        "h_re": [np.real(hu).ravel().tolist(), np.real(hv).ravel().tolist()],
        "h_im": [np.imag(hu).ravel().tolist(), np.imag(hv).ravel().tolist()],
        "freq": float(freq),
        "h_offset": float(info["h_offset"]),
    }
    return profile, power


# ----------------------------------------------------------------------------
# derived quantities
# ----------------------------------------------------------------------------


def _steps(nodes):
    nodes = np.asarray(nodes, dtype=float)
    dp = np.diff(nodes)
    dd = np.empty(len(dp))
    dd[0] = dp[0]
    dd[1:] = 0.5 * (dp[:-1] + dp[1:])
    return dp, dd


def _areas(mode):
    """Cell areas of the six staggered positions in natural (u, v) layout."""
    dup, dud = _steps(mode.info["u_nodes"])
    dvp, dvd = _steps(mode.info["v_nodes"])
    return {
        "eu": np.outer(dup, dvd),
        "ev": np.outer(dud, dvp),
        "en": np.outer(dud, dvd),
        "hu": np.outer(dud, dvp),
        "hv": np.outer(dup, dvd),
        "hn": np.outer(dup, dvp),
    }


def _handed(mode) -> float:
    """+1 if (u, v, normal) is right-handed for the mode's plane, -1 if not
    (planes normal to y)."""
    s = mode.info.get("solver", {})
    if "swap" in s:
        return -1.0 if s["swap"] else 1.0
    return -1.0 if int(mode.info.get("axis", 0)) == 1 and "p" in mode.info else 1.0


def _natural_positions(mode):
    """Coordinates of the staggered positions in natural (u, v) layout."""
    p = _positions(mode.info["u_nodes"], mode.info["v_nodes"])
    return {"eu": p["ex"], "ev": p["ey"], "en": p["ez"], "hu": p["hx"], "hv": p["hy"], "hn": p["hz"]}


def _fields_at_centres(mode):
    un, vn = np.asarray(mode.info["u_nodes"], float), np.asarray(mode.info["v_nodes"], float)
    Eu, Ev = mode.E
    Hu, Hv = mode.H
    z = np.zeros_like(Eu)
    En = mode.E_n if mode.E_n is not None else z
    Hn = mode.H_n if mode.H_n is not None else z

    def v_avg(a):  # (c, n) -> (c, c)
        b = np.zeros_like(a)
        b[:, :-1] = 0.5 * (a[:, :-1] + a[:, 1:])
        b[:, -1] = 0.5 * a[:, -1]
        return b

    def u_avg(a):  # (n, c) -> (c, c)
        b = np.zeros_like(a)
        b[:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])
        b[-1, :] = 0.5 * a[-1, :]
        return b

    return {
        "u": 0.5 * (un[:-1] + un[1:]),
        "v": 0.5 * (vn[:-1] + vn[1:]),
        "Eu": v_avg(Eu),
        "Ev": u_avg(Ev),
        "En": u_avg(v_avg(En)),
        "Hu": u_avg(Hu),
        "Hv": v_avg(Hv),
        "Hn": Hn,
    }


def mode_effective_area(mode) -> float:
    """Effective area ``(∫|E|² dA)² / ∫|E|⁴ dA`` (m²) of a :class:`Mode`,
    with the three electric components averaged to the cell centres."""
    f = _fields_at_centres(mode)
    dup, _ = _steps(mode.info["u_nodes"])
    dvp, _ = _steps(mode.info["v_nodes"])
    a = np.outer(dup, dvp)
    e2 = np.abs(f["Eu"]) ** 2 + np.abs(f["Ev"]) ** 2 + np.abs(f["En"]) ** 2
    return float(np.sum(e2 * a) ** 2 / np.sum(e2**2 * a))


def _region_mask(X, Y, region):
    (u0, v0), (u1, v1) = region
    return ((X >= min(u0, u1)) & (X <= max(u0, u1)) & (Y >= min(v0, v1)) & (Y <= max(v0, v1))).astype(float)


def mode_confinement(mode, material=None, region=None, kind: str = "energy") -> float:
    """Confinement factor of a mode in a material or a region.

    ``kind="energy"``: share of the electric energy,
    ``∫_A Re(eps) |E|² dA / ∫ Re(eps) |E|² dA``, with all three components
    at their own grid positions. In a cell that two materials share, the
    energy is split in proportion to ``fill * eps`` of each material.

    ``kind="power"``: share of the longitudinal Poynting flux,
    ``∫_A Re(E × H*)·n dA / ∫ Re(E × H*)·n dA``.

    ``material`` is a material name of a cross-section mode (one made by
    :class:`lightsim.xsection.CrossSection`, which records the fill fraction
    of every material); ``region`` is a rectangle ``((u0, v0), (u1, v1))``
    in metres. Give one of them, or both (their intersection).
    """
    if material is None and region is None:
        raise ValueError("give a material or a region")
    pos = _natural_positions(mode)
    a = _areas(mode)
    Eu, Ev = mode.E
    Hu, Hv = mode.H
    En = mode.E_n if mode.E_n is not None else np.zeros_like(Eu)
    fill = mode.info.get("fill")
    if material is not None and not fill:
        raise ValueError("this mode carries no material map; use region=")
    if material is not None and material not in fill:
        raise ValueError(f"no material {material!r} in this cross-section; materials: {sorted(fill)}")

    def weight(comp):  # share of the material at the positions of an E component (0: u, 1: v, 2: n)
        w = np.ones_like(np.real(Eu))
        if material is not None:
            em = mode.info["eps_material"]
            num = np.real(em[material][comp]) * fill[material][comp]
            den = sum(np.real(em[m][comp]) * fill[m][comp] for m in fill)
            w = num / np.where(den == 0, 1.0, den) if kind == "energy" else fill[material][comp]
        if region is not None:
            key = ("eu", "ev", "en")[comp]
            w = w * _region_mask(*pos[key], region)
        return w

    if kind == "energy":
        eps = [np.asarray(mode.info["eps_u"]), np.asarray(mode.info["eps_v"]), np.asarray(mode.info["eps_n"])]
        parts = [eps[0] * np.abs(Eu) ** 2 * a["eu"], eps[1] * np.abs(Ev) ** 2 * a["ev"], eps[2] * np.abs(En) ** 2 * a["en"]]
        tot = sum(np.sum(p) for p in parts)
        return float(sum(np.sum(p * weight(i)) for i, p in enumerate(parts)) / tot)
    if kind == "power":
        su = np.real(Eu * np.conj(Hv)) * a["eu"]
        sv = -np.real(Ev * np.conj(Hu)) * a["ev"]
        tot = np.sum(su) + np.sum(sv)
        return float((np.sum(su * weight(0)) + np.sum(sv * weight(1))) / tot)
    raise ValueError("kind must be 'energy' or 'power'")


def group_index_of(mode, eps_group, mu=None) -> float:
    """Group index of a solved mode from the energy-flux relation
    ``n_g = c W / P = (∫ d(w eps)/dw |E|² + mu |eta0 H|² dA) / (2 Re ∫ E × eta0 H* · n dA)``.

    ``eps_group = (eg_u, eg_v, eg_n)`` holds ``d(w eps)/dw`` at the E_u,
    E_v, E_n positions (the permittivity itself for non-dispersive media).
    """
    a = _areas(mode)
    Eu, Ev = mode.E
    Hu, Hv = (h * ETA0 for h in mode.H)
    En = mode.E_n if mode.E_n is not None else np.zeros_like(Eu)
    Hn = mode.H_n * ETA0 if mode.H_n is not None else np.zeros_like(Eu)
    eg = [np.real(np.asarray(e)) for e in eps_group]
    we = np.sum(eg[0] * np.abs(Eu) ** 2 * a["eu"]) + np.sum(eg[1] * np.abs(Ev) ** 2 * a["ev"]) + np.sum(eg[2] * np.abs(En) ** 2 * a["en"])
    mu = mu if mu is not None else (1.0, 1.0, 1.0)
    wm = np.sum(np.real(mu[0]) * np.abs(Hu) ** 2 * a["hu"]) + np.sum(np.real(mu[1]) * np.abs(Hv) ** 2 * a["hv"]) + np.sum(np.real(mu[2]) * np.abs(Hn) ** 2 * a["hn"])
    p = _handed(mode) * np.real(np.sum(Eu * np.conj(Hv) * a["eu"]) - np.sum(Ev * np.conj(Hu) * a["ev"]))
    return float(0.5 * (we + wm) / p)


def _resample(mode, target):
    """Fields of ``mode`` at the staggered positions of ``target`` (linear
    interpolation, zero outside ``mode``'s window)."""
    from scipy.interpolate import RegularGridInterpolator

    src = _natural_positions(mode)
    dst = _natural_positions(target)
    out = {}
    for name, arr in (("eu", mode.E[0]), ("ev", mode.E[1]), ("hu", mode.H[0]), ("hv", mode.H[1])):
        X, Y = src[name]
        f = RegularGridInterpolator((X[:, 0], Y[0, :]), arr, bounds_error=False, fill_value=0.0)
        Xd, Yd = dst[name]
        out[name] = f(np.stack([Xd.ravel(), Yd.ravel()], axis=-1)).reshape(Xd.shape)
    return out


def _same_grid(m1, m2) -> bool:
    a, b = m1.info, m2.info
    return len(a["u_nodes"]) == len(b["u_nodes"]) and len(a["v_nodes"]) == len(b["v_nodes"]) and np.allclose(a["u_nodes"], b["u_nodes"], rtol=0, atol=1e-15) and np.allclose(a["v_nodes"], b["v_nodes"], rtol=0, atol=1e-15)


def mode_overlap(mode1, mode2, conjugate: bool = True) -> complex:
    """Overlap integral of two modes on the plane, with fields as stored
    (unit power for solver modes).

    ``conjugate=True`` (the power overlap, gplugins ``innerprod_trapz`` and
    Tidy3D ``outer_dot(conjugate=True)``)::

        1/4 ∫ (E1* × H2 + E2 × H1*) · n dA

    ``|overlap|²`` is the share of the power of ``mode2`` that couples into
    ``mode1`` for lossless, power-normalised modes, and the self-overlap of
    a solver mode is 1.

    ``conjugate=False`` (the reciprocity form used for lossy and leaky
    modes, Tidy3D ``outer_dot(conjugate=False)``)::

        1/4 ∫ (E1 × H2 + E2 × H1) · n dA

    Modes on different grids are compared by interpolating ``mode2``
    linearly onto ``mode1``'s grid (zero outside ``mode2``'s window); the
    two windows must share their coordinate frame.
    """
    a = _areas(mode1)
    E1u, E1v = mode1.E
    H1u, H1v = mode1.H
    if _same_grid(mode1, mode2):
        E2u, E2v = mode2.E
        H2u, H2v = mode2.H
    else:
        r = _resample(mode2, mode1)
        E2u, E2v, H2u, H2v = r["eu"], r["ev"], r["hu"], r["hv"]
    c = np.conj if conjugate else (lambda x: x)
    t1 = np.sum(c(E1u) * H2v * a["eu"]) - np.sum(c(E1v) * H2u * a["ev"])
    t2 = np.sum(E2u * c(H1v) * a["eu"]) - np.sum(E2v * c(H1u) * a["ev"])
    return complex(_handed(mode1) * 0.25 * (t1 + t2))


def overlap_matrix(modes1, modes2, conjugate: bool = True) -> np.ndarray:
    """``O[i, j] = mode_overlap(modes1[i], modes2[j], conjugate)``."""
    return np.array([[mode_overlap(a, b, conjugate) for b in modes2] for a in modes1])
