"""Full-vectorial finite-difference eigenmode solver on the lightsim Yee grid.

The cross-section permittivity comes from the Rust core (`plane_info`), i.e.
exactly the sub-pixel-smoothed values the FDTD engine uses, sampled at the
staggered positions of the transverse Yee lattice. The solver discretizes
Maxwell's equations for fields ``~ exp(i(beta*n - w t))`` with the same
staggering (E_u at (u+1/2, v), E_v at (u, v+1/2), E_n at nodes, H dual), so
mode profiles can be injected and projected without interpolation.

Formulation (normalized, H~ = eta0 H, k0 = w/c):
    beta E_t = P H_t,   beta H_t = Q E_t   ->   beta^2 E_t = P Q E_t
solved with shift-invert Arnoldi (scipy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

C0 = 299_792_458.0
ETA0 = 376.730313412


@dataclass
class Mode:
    """A waveguide mode returned by :func:`solve_modes`.

    Attributes
    ----------
    neff : complex
        Effective index; a positive imaginary part is loss.
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
        (and ``eps_*_im``), ``coord`` and ``h_offset``.
    """

    neff: complex
    freq: float
    te_fraction: float
    profile: dict  # ModeProfile dict for lightsim sources/monitors
    E: tuple  # (E_u, E_v) complex arrays on the plane (natural order)
    H: tuple  # (H_u, H_v) complex arrays (A/m) at the E plane
    info: dict

    def __repr__(self) -> str:
        return f"<Mode neff={self.neff.real:.5f}{self.neff.imag:+.2e}j TE={self.te_fraction:.2f}>"


def _diff_ops(nodes_x, nodes_y):
    """Forward (E->H) and backward (H->E) difference operators on an Nx x Ny lattice."""
    dxp = np.diff(nodes_x)
    dyp = np.diff(nodes_y)
    Nx, Ny = len(dxp), len(dyp)
    dxd = np.empty(Nx)
    dxd[0] = dxp[0]
    dxd[1:] = 0.5 * (dxp[:-1] + dxp[1:])
    dyd = np.empty(Ny)
    dyd[0] = dyp[0]
    dyd[1:] = 0.5 * (dyp[:-1] + dyp[1:])
    Ix = sp.identity(Nx, format="csr")
    Iy = sp.identity(Ny, format="csr")
    # 1-D forward: (f[i+1]-f[i])/dxp[i], f[N]=0 ; backward: (g[i]-g[i-1])/dxd[i], g[-1]=0
    fx = sp.diags([-1.0 / dxp, 1.0 / dxp[:-1]], [0, 1], shape=(Nx, Nx), format="csr")
    bx = sp.diags([1.0 / dxd, -1.0 / dxd[1:]], [0, -1], shape=(Nx, Nx), format="csr")
    fy = sp.diags([-1.0 / dyp, 1.0 / dyp[:-1]], [0, 1], shape=(Ny, Ny), format="csr")
    by = sp.diags([1.0 / dyd, -1.0 / dyd[1:]], [0, -1], shape=(Ny, Ny), format="csr")
    Dxe = sp.kron(fx, Iy, format="csr")
    Dxh = sp.kron(bx, Iy, format="csr")
    Dye = sp.kron(Ix, fy, format="csr")
    Dyh = sp.kron(Ix, by, format="csr")
    return Dxe, Dxh, Dye, Dyh, dxp, dxd, dyp, dyd


def solve_cross_section(
    eps_x,
    eps_y,
    eps_z,
    nodes_x,
    nodes_y,
    freq: float,
    num_modes: int = 2,
    target_neff: float | None = None,
):
    """Solve the transverse eigenproblem on a (right-handed x, y, propagation)
    cross-section. ``eps_*`` are (Nx, Ny) arrays at the Yee positions.
    Returns (neffs, Ex, Ey, Hx, Hy) with H normalized (H~ = eta0 H).
    """
    eps_x = np.asarray(eps_x, dtype=complex)
    eps_y = np.asarray(eps_y, dtype=complex)
    eps_z = np.asarray(eps_z, dtype=complex)
    Nx, Ny = eps_x.shape
    N = Nx * Ny
    k0 = 2 * np.pi * freq / C0
    Dxe, Dxh, Dye, Dyh, dxp, dxd, dyp, dyd = _diff_ops(nodes_x, nodes_y)
    ex = sp.diags(eps_x.ravel())
    ey = sp.diags(eps_y.ravel())
    iez = sp.diags(1.0 / eps_z.ravel())
    mux = sp.identity(N)
    muy = sp.identity(N)
    imuz = sp.identity(N)
    P = sp.bmat(
        [
            [-Dxe @ iez @ Dyh, k0**2 * muy + Dxe @ iez @ Dxh],
            [-(k0**2) * mux - Dye @ iez @ Dyh, Dye @ iez @ Dxh],
        ],
        format="csc",
    ) / k0
    Q = sp.bmat(
        [
            [Dxh @ imuz @ Dye, -(k0**2) * ey - Dxh @ imuz @ Dxe],
            [k0**2 * ex + Dyh @ imuz @ Dye, -Dyh @ imuz @ Dxe],
        ],
        format="csc",
    ) / k0
    A = (P @ Q).tocsc()
    nmax = float(np.sqrt(np.max(np.real(np.concatenate([eps_x.ravel(), eps_y.ravel()])))))
    tn = target_neff if target_neff is not None else nmax
    sigma = (k0 * tn) ** 2
    k = min(num_modes, 2 * N - 2)
    vals, vecs = spla.eigs(A, k=k, sigma=sigma, which="LM")
    beta = np.sqrt(vals.astype(complex))
    beta = np.where(beta.real < 0, -beta, beta)
    order = np.argsort(-beta.real)
    beta = beta[order]
    vecs = vecs[:, order]
    E = vecs
    H = (Q @ E) / beta[None, :]
    Ex = E[:N].T.reshape(-1, Nx, Ny)
    Ey = E[N:].T.reshape(-1, Nx, Ny)
    Hx = H[:N].T.reshape(-1, Nx, Ny)
    Hy = H[N:].T.reshape(-1, Nx, Ny)
    # areas for power: Ex/Hy at (i+1/2, j), Ey/Hx at (i, j+1/2)
    a_ex = np.outer(dxp, dyd)
    a_ey = np.outer(dxd, dyp)
    out = []
    for m in range(len(beta)):
        p = 0.5 / ETA0 * np.real(np.sum(Ex[m] * np.conj(Hy[m]) * a_ex) - np.sum(Ey[m] * np.conj(Hx[m]) * a_ey))
        s = 1.0 / np.sqrt(abs(p)) if p != 0 else 1.0
        if p < 0:  # flip propagation sign convention (shouldn't happen for beta>0)
            s = -s
        ex_, ey_, hx_, hy_ = Ex[m] * s, Ey[m] * s, Hx[m] * s, Hy[m] * s
        # fix the global phase: largest E component real positive
        big = ex_ if np.max(np.abs(ex_)) >= np.max(np.abs(ey_)) else ey_
        idx = np.unravel_index(np.argmax(np.abs(big)), big.shape)
        ph = np.exp(-1j * np.angle(big[idx]))
        out.append((beta[m] / k0, ex_ * ph, ey_ * ph, hx_ * ph, hy_ * ph, a_ex, a_ey))
    return out


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
):
    """Solve waveguide modes on a plane of a :class:`lightsim.Simulation`.

    Returns a list of :class:`Mode` sorted by effective index (or by TE
    fraction first when ``sort="te"``).

    In a 2-D simulation the plane is a line and the modes are those of the
    1-D solver :func:`lightsim.eim.solve_port_mode`, for the polarization(s)
    the simulation contains: ``te_fraction`` is 1 for the modes with E in
    the plane and 0 for those with E along the invariant axis.
    """
    from . import _core

    if getattr(sim, "dimensions", 3) == 2:
        from .eim import solve_port_mode

        pols = ("te", "tm") if sim.polarization == "both" else (sim.polarization,)
        modes = [m for pol in pols for m in solve_port_mode(sim, axis, position, min, max, freq, pol, num_modes=num_modes)]
        modes.sort(key=(lambda m: (-round(m.te_fraction), -m.neff.real)) if sort == "te" else (lambda m: -m.neff.real))
        return modes[:num_modes]

    spec = sim.geometry_json() if hasattr(sim, "geometry_json") else sim
    info = _core.plane_info(spec, int(axis), float(position), [float(v) for v in min], [float(v) for v in max], float(freq))
    # complex permittivity at the mode frequency (lossy / dispersive media)
    eu = info["eps_u"] + 1j * info["eps_u_im"]
    ev = info["eps_v"] + 1j * info["eps_v_im"]
    en = info["eps_n"] + 1j * info["eps_n_im"]
    un, vn = info["u_nodes"], info["v_nodes"]
    swap = axis == 1  # natural (x, z) is left-handed around +y: solve in (z, x)
    if swap:
        res = solve_cross_section(ev.T, eu.T, en.T, vn, un, freq, num_modes, target_neff)
    else:
        res = solve_cross_section(eu, ev, en, un, vn, freq, num_modes, target_neff)
    modes = []
    for neff, ex, ey, hx, hy, a_ex, a_ey in res:
        if swap:
            Eu, Ev, Hu, Hv = ey.T, ex.T, hy.T, hx.T
            a_u, a_v = a_ey.T, a_ex.T
        else:
            Eu, Ev, Hu, Hv = ex, ey, hx, hy
            a_u, a_v = a_ex, a_ey
        te = float(np.sum(np.abs(Eu) ** 2 * a_u) / (np.sum(np.abs(Eu) ** 2 * a_u) + np.sum(np.abs(Ev) ** 2 * a_v)))
        k0 = 2 * np.pi * freq / C0
        shift = np.exp(-1j * neff * k0 * info["h_offset"])  # H plane half a cell upstream
        Hu_a = Hu / ETA0
        Hv_a = Hv / ETA0
        hu_p = Hu_a * shift
        hv_p = Hv_a * shift
        profile = {
            "neff": float(np.real(neff)),
            "neff_imag": float(np.imag(neff)),
            "dims": [int(info["dims"][0]), int(info["dims"][1])],
            "e_re": [np.real(Eu).ravel().tolist(), np.real(Ev).ravel().tolist()],
            "e_im": [np.imag(Eu).ravel().tolist(), np.imag(Ev).ravel().tolist()],
            "h_re": [np.real(hu_p).ravel().tolist(), np.real(hv_p).ravel().tolist()],
            "h_im": [np.imag(hu_p).ravel().tolist(), np.imag(hv_p).ravel().tolist()],
            "freq": float(freq),
            "h_offset": float(info["h_offset"]),
        }
        modes.append(Mode(neff=complex(neff), freq=freq, te_fraction=te, profile=profile, E=(Eu, Ev), H=(Hu_a, Hv_a), info=info))
    if sort == "te":
        modes.sort(key=lambda m: (-round(m.te_fraction), -m.neff.real))
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
