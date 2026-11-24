"""Media: permittivity, permeability, conductivity, poles, nonlinearity,
gain, director fields, plug-ins; zero-thickness sheets."""

from __future__ import annotations

from typing import Iterable, Sequence

from ._util import _axis, _v3, Vec3


def Medium(
    name: str,
    *,
    n: float | Sequence[float] | None = None,
    eps: float | Sequence[float] | None = None,
    mu: float | Sequence[float] = 1.0,
    sigma: float | Sequence[float] = 0.0,
    sigma_m: float | Sequence[float] = 0.0,
    poles: Iterable[dict] = (),
    pole_axes: Sequence[bool] = (True, True, True),
    pec: bool = False,
    chi2: float | Sequence[float] = 0.0,
    chi3: float | Sequence[float] = 0.0,
    eps_off: Sequence[float] = (0.0, 0.0, 0.0),
    sigma_off: Sequence[float] = (0.0, 0.0, 0.0),
    rotation=None,
    oriented: Iterable[tuple[dict, Sequence[float]]] = (),
    raman: dict | None = None,
    gyro: Iterable[tuple[dict, Sequence[float]]] = (),
    director: dict | None = None,
    gain: dict | None = None,
    plugin: dict | None = None,
    gain2: dict | None = None,
    eps_complex=None,
) -> dict:
    """A (possibly anisotropic, dispersive, conductive, nonlinear) medium.

    ``sigma`` is the electric conductivity (S/m), ``sigma_m`` the magnetic
    conductivity (Ohm/m, the magnetic current sigma_m H in Faraday's law); a
    medium with ``sigma_m / (mu0 mu) = sigma / (eps0 eps)`` has the impedance
    of the lossless medium at every frequency.

    ``chi2`` (m/V) and ``chi3`` (m^2/V^2) are instantaneous per-axis
    susceptibilities: P_i/eps0 = chi_lin E_i + chi2 E_i^2 + chi3 E_i^3
    (Lumerical's chi2/chi3 material convention).

    ``eps_off`` = (eps_xy, eps_xz, eps_yz) and ``sigma_off`` add off-diagonal
    terms to symmetric permittivity / conductivity tensors. ``rotation`` (3x3,
    columns = crystal axes in lab coordinates) rotates the diagonal
    ``eps``/``n`` and ``sigma`` into the lab frame (tilted uniaxial crystals,
    lossy or not).

    ``oriented`` = [(pole, u), ...] adds dispersive poles acting along unit
    vectors ``u`` (chi(w) u u^T), e.g. ``[(Lorentz(1.0, w0, g), (1, 1, 0))]``
    for a dispersive crystal whose optic axis is rotated in the xy plane.

    ``raman`` = ``Raman(fraction, omega, delta)`` makes a fraction of ``chi3``
    a delayed (Raman) response: P_i/eps0 = chi3 E_i [(1 - f) E_i^2 + f (g_R * E_i^2)].

    ``gyro`` = [(pole, bias), ...] adds gyrotropic (magneto-optic) poles: a
    ``Lorentz`` or ``Drude`` pole in a static bias, ``bias`` being the
    gyration vector b in rad/s (for free electrons the cyclotron frequency
    e B / m along B). The polarization obeys P'' + gamma P' + omega0^2 P -
    b x P' = s E, which gives a permittivity tensor with antisymmetric
    off-diagonal terms (eps_xy = -eps_yx = i g for a bias along z); see
    :func:`lightsim.materials.gyrotropic_permittivity`. Needs a lossless
    diagonal ``eps`` and no other ``poles``.

    ``director`` = ``DirectorField(...)`` makes the medium uniaxial with an
    optic axis that varies from cell to cell (a liquid crystal); without
    ``n`` / ``eps`` the medium's ``eps`` (read only by the automatic mesher)
    is set to the larger of eps_o and eps_e.

    ``gain`` = ``FourLevel(...)`` adds a four-level atomic gain medium with
    pumping (rate equations coupled to a Lorentz polarization of the lasing
    transition); ``eps`` is then the background permittivity.

    ``plugin`` = ``lightsim.plugins.use(name, params)`` runs a user-defined
    material model registered under ``name`` (see :mod:`lightsim.plugins`)
    on top of the background ``eps`` and ``sigma``.

    ``gain2`` = ``FourLevelTwoElectron(...)`` adds Chang and Taflove's
    four-level two-electron gain medium (Lumerical's "4-level 2-electron"
    model), pumped optically by a field at its pump transition.

    ``eps_complex`` = a complex 3x3 relative permittivity tensor given
    directly, Hermitian and positive definite (e.g. ``[[e, 0, 0], [0, e,
    1j * g], [0, -1j * g, e]]`` for a magneto-optic medium biased along x).
    The medium is non-dispersive and lossless; the run then uses complex
    fields, and the tensor holds at every frequency (exact only as a
    non-dispersive model). It cannot be combined with ``eps``/``n``, poles,
    ``sigma`` or other tensors.
    """

    def tri(x):
        return [float(x)] * 3 if isinstance(x, (int, float)) else [float(v) for v in x]

    ec = None
    if eps_complex is not None:
        import numpy as _np

        t = _np.asarray(eps_complex, dtype=complex)
        if t.shape != (3, 3):
            raise ValueError("eps_complex must be a 3x3 matrix")
        if not _np.allclose(t, t.conj().T, rtol=0, atol=1e-12 * _np.max(_np.abs(t))):
            raise ValueError("eps_complex must be Hermitian (eps_ij = conj(eps_ji))")
        if _np.min(_np.linalg.eigvalsh(t)) <= 0:
            raise ValueError("eps_complex must be positive definite")
        if n is not None or eps is not None:
            raise ValueError("give eps_complex alone (its real diagonal becomes eps)")
        ec = [[[float(t[i, j].real), float(t[i, j].imag)] for j in range(3)] for i in range(3)]
        eps = [float(t[i, i].real) for i in range(3)]

    if n is not None:
        nn = tri(n)
        epsv = [v * v for v in nn]
    elif eps is not None:
        epsv = tri(eps)
    elif director is not None:
        epsv = [max(director["eps_o"], director["eps_e"])] * 3
    else:
        epsv = [1.0, 1.0, 1.0]
    off = [float(v) for v in eps_off]
    sig = tri(sigma)
    soff = [float(v) for v in sigma_off]
    if rotation is not None:
        import numpy as _np

        r = _np.asarray(rotation, dtype=float)

        def rot(d, o):
            t = r @ _np.diag(d) @ r.T + _np.array([[0, o[0], o[1]], [o[0], 0, o[2]], [o[1], o[2], 0]])
            return [float(t[0, 0]), float(t[1, 1]), float(t[2, 2])], [float(t[0, 1]), float(t[0, 2]), float(t[1, 2])]

        epsv, off = rot(epsv, off)
        sig, soff = rot(sig, soff)
    return {
        "name": name,
        "eps": epsv,
        "eps_off": off,
        "mu": tri(mu),
        "sigma": sig,
        "sigma_off": soff,
        "sigma_m": tri(sigma_m),
        "oriented": [{"pole": p, "u": [float(v) for v in u]} for p, u in oriented],
        "poles": list(poles),
        "pole_axes": list(pole_axes),
        "pec": pec,
        "chi2": tri(chi2),
        "chi3": tri(chi3),
        **({"raman": dict(raman)} if raman is not None else {}),
        **({"gyro": [{"pole": p, "bias": [float(v) for v in b]} for p, b in gyro]} if gyro else {}),
        **({"director": dict(director)} if director is not None else {}),
        **({"gain": dict(gain)} if gain is not None else {}),
        **({"plugin": dict(plugin)} if plugin is not None else {}),
        **({"gain2": dict(gain2)} if gain2 is not None else {}),
        **({"eps_complex": ec} if ec is not None else {}),
    }


def FourLevel(density: float, omega: float, gamma: float, tau32: float, tau21: float, tau10: float, pump_rate: float = 0.0, coupling: float | None = None, initial=(1.0, 0.0, 0.0, 0.0)) -> dict:
    """A four-level atomic gain medium with pumping, for ``Medium(gain=...)``.

    Rate equations coupled to a Lorentz polarization of the 2 -> 1
    transition (fields exp(-i w t), N in m^-3, P in C/m^2)::

        P'' + gamma P' + omega^2 P = xi (N1 - N2) E
        dN3/dt = W_p N0 - N3/tau32
        dN2/dt = N3/tau32 - N2/tau21 + E.P' / (hbar omega)
        dN1/dt = N2/tau21 - N1/tau10 - E.P' / (hbar omega)
        dN0/dt = N1/tau10 - W_p N0

    ``density`` is the total atom density (m^-3), ``omega`` and ``gamma`` the
    transition's angular frequency and width (rad/s), the ``tau`` the
    lifetimes (s), ``pump_rate`` W_p (1/s) the pump from level 0 to 3, and
    ``coupling`` xi (C^2/kg, default the radiative value
    6 pi eps0 c^3 / (omega^2 tau21)). ``initial`` gives the starting
    populations as fractions of ``density``. At fixed populations the
    transition adds chi(w) = (xi (N1 - N2) / eps0) / (omega^2 - w^2 - i gamma w):
    gain where N2 > N1. Each E-component position carries its own atoms,
    coupled only to its own field component. The rate equations are explicit:
    the time step must stay below half of every lifetime and of 1/W_p.
    ``Results.gain_levels()`` returns the atoms in each level at the end.
    """
    out = {"density": float(density), "omega": float(omega), "gamma": float(gamma), "tau32": float(tau32), "tau21": float(tau21), "tau10": float(tau10), "pump_rate": float(pump_rate), "initial": [float(v) for v in initial]}
    if coupling is not None:
        out["coupling"] = float(coupling)
    return out


def FourLevelTwoElectron(
    density: float,
    omega_a: float,
    omega_b: float,
    gamma_a: float,
    gamma_b: float,
    t30: float,
    t32: float,
    t21: float,
    t10: float,
    initial=None,
    kappa_a: float | None = None,
    kappa_b: float | None = None,
) -> dict:
    """The four-level two-electron gain medium of Chang and Taflove (Opt.
    Express 12, 3827 (2004)), for ``Medium(gain2=...)``; the parameters of
    Lumerical's "4-level 2-electron" material (N density, omega a / b, gamma
    a / b, t30, t32, t21, t10, initial populations)::

        Pa'' + gamma_a Pa' + omega_a^2 Pa = kappa_a (N1 - N2) E
        Pb'' + gamma_b Pb' + omega_b^2 Pb = kappa_b (N0 - N3) E
        dN3/dt =  E.Pb'/(hbar omega_b) - N3 (1 - N2/Nd)/t32 - N3 (1 - N0/Nd)/t30
        dN2/dt =  N3 (1 - N2/Nd)/t32 + E.Pa'/(hbar omega_a) - N2 (1 - N1/Nd)/t21
        dN1/dt =  N2 (1 - N1/Nd)/t21 - E.Pa'/(hbar omega_a) - N1 (1 - N0/Nd)/t10
        dN0/dt =  N3 (1 - N0/Nd)/t30 + N1 (1 - N0/Nd)/t10 - E.Pb'/(hbar omega_b)

    ``density`` Nd in m^-3 (two electrons per atom: the populations sum to
    2 Nd, each at most Nd), angular frequencies and widths in rad/s,
    lifetimes in s; ``initial`` = (N0, N1, N2, N3) in m^-3, default
    (Nd, Nd, 0, 0); ``kappa_a``, ``kappa_b`` (C^2/kg) default to
    6 pi eps0 c^3 / (omega^2 t) with t21 and t30. Pump it with a source at
    ``omega_b``; ``Results.gain_levels()`` returns the electrons in each
    level at the end. The rate equations are explicit: the time step must
    stay below half of every lifetime.
    """
    out = {"density": float(density), "omega_a": float(omega_a), "omega_b": float(omega_b), "gamma_a": float(gamma_a), "gamma_b": float(gamma_b), "t30": float(t30), "t32": float(t32), "t21": float(t21), "t10": float(t10)}
    if initial is not None:
        out["initial"] = [float(v) for v in initial]
    if kappa_a is not None:
        out["kappa_a"] = float(kappa_a)
    if kappa_b is not None:
        out["kappa_b"] = float(kappa_b)
    return out


def DirectorField(min: Vec3, max: Vec3, director, *, eps_o=None, eps_e=None, n_o=None, n_e=None, axes: Sequence[int] | None = None) -> dict:
    """A director field for ``Medium(director=...)``: a uniaxial medium
    (a liquid crystal) whose optic axis varies from cell to cell.

    ``director`` is a numpy array of director vectors, its last axis of
    length 3 (x, y, z components; the length and the sign do not matter).
    The leading axes are the sample grid, whose nodes span ``[min, max]``
    inclusive, as in :func:`IndexGrid`: 3-D grids are indexed
    ``[ix, iy, iz]``; a 1-D or 2-D grid varies along x or (x, y) unless
    ``axes`` says otherwise (``axes=(2,)`` for a director that turns along
    z); a single vector is a uniform director. Give the ordinary and
    extraordinary permittivities (``eps_o``, ``eps_e``) or indices (``n_o``,
    ``n_e``). At each field position the solver interpolates the director
    tensor d d^T trilinearly, normalises it to unit trace and uses
    eps = eps_o I + (eps_e - eps_o) d d^T; outside the box the nearest
    face value is used.
    """
    import numpy as _np

    if (eps_o is None) == (n_o is None) or (eps_e is None) == (n_e is None):
        raise ValueError("give eps_o or n_o, and eps_e or n_e")
    eo = float(eps_o) if eps_o is not None else float(n_o) ** 2
    ee = float(eps_e) if eps_e is not None else float(n_e) ** 2
    d = _np.asarray(director, dtype=float)
    if d.shape[-1:] != (3,) or d.ndim > 4:
        raise ValueError(f"director must be an array of 3-vectors (shape (..., 3), at most 3 grid axes), got {d.shape}")
    g = d.ndim - 1
    ax = tuple(range(g)) if axes is None else tuple(int(x) for x in axes)
    if len(ax) != g or len(set(ax)) != g or any(x not in (0, 1, 2) for x in ax):
        raise ValueError(f"axes={axes!r} does not match a director grid of {g} axes")
    if g > 1:
        d = _np.transpose(d, list(_np.argsort(ax)) + [g])
    shape = [1, 1, 1]
    for n_, x in zip(d.shape[:-1], sorted(ax)):
        shape[x] = n_
    d = d.reshape(shape + [3])
    if not _np.all(_np.isfinite(d)) or _np.any(_np.linalg.norm(d, axis=-1) == 0):
        raise ValueError("director samples must be finite non-zero vectors")
    return {"eps_o": eo, "eps_e": ee, "min": _v3(min), "max": _v3(max), "dims": shape, "director": d.ravel(order="C").tolist()}


def Raman(fraction: float, omega: float, delta: float) -> dict:
    """The delayed (Raman) part of a chi3 response, for ``Medium(raman=...)``.

    A fraction ``fraction`` (f, 0 to 1) of ``chi3`` acts through the
    damped-oscillator response g_R(t) = (omega^2 / nu) exp(-delta t) sin(nu t),
    nu = sqrt(omega^2 - delta^2), whose integral is 1:

        P_i/eps0 = chi3_i E_i [(1 - f) E_i^2 + f (g_R * E_i^2)(t)],
        g_R(w) = omega^2 / (omega^2 - w^2 - 2j delta w).

    ``omega`` is the phonon (resonance) angular frequency and ``delta`` the
    damping rate, in rad/s (0 <= delta < omega). The common fibre form
    g_R = (tau1^2 + tau2^2) / (tau1 tau2^2) exp(-t/tau2) sin(t/tau1) is
    omega = sqrt(1/tau1^2 + 1/tau2^2), delta = 1/tau2; for silica, f = 0.18,
    tau1 = 12.2 fs and tau2 = 32 fs give omega = 8.77e13 rad/s (14.0 THz)
    and delta = 3.13e13 rad/s.
    """
    return {"fraction": float(fraction), "omega": float(omega), "delta": float(delta)}


def Lorentz(delta_eps: float, omega0: float, gamma: float) -> dict:
    """A Lorentz pole for ``Medium(poles=...)``.

    Adds the susceptibility ``delta_eps * omega0**2 / (omega0**2 - w**2 - 1j * gamma * w)``
    to the permittivity.

    Parameters
    ----------
    delta_eps : float
        Strength: the contribution of the pole to the static permittivity.
    omega0 : float
        Resonance angular frequency (rad/s).
    gamma : float
        Damping rate (rad/s); the full width of the absorption line.

    Returns the pole dict.
    """
    return {"type": "lorentz", "delta_eps": delta_eps, "omega0": omega0, "gamma": gamma}


def Drude(omega_p: float, gamma: float) -> dict:
    """A Drude pole for ``Medium(poles=...)``, the response of free carriers.

    Adds the susceptibility ``-omega_p**2 / (w**2 + 1j * gamma * w)``.

    Parameters
    ----------
    omega_p : float
        Plasma angular frequency (rad/s).
    gamma : float
        Collision rate (rad/s).

    Returns the pole dict.
    """
    return {"type": "drude", "omega_p": omega_p, "gamma": gamma}


def Debye(delta_eps: float, tau: float) -> dict:
    """A Debye pole for ``Medium(poles=...)``, a relaxation.

    Adds the susceptibility ``delta_eps / (1 - 1j * w * tau)``.

    Parameters
    ----------
    delta_eps : float
        Strength: the contribution of the pole to the static permittivity.
    tau : float
        Relaxation time (s). It should be resolved by the time step; a pole with
        ``tau`` far below the time step makes the run refuse to start.

    Returns the pole dict.
    """
    return {"type": "debye", "delta_eps": delta_eps, "tau": tau}


PEC = Medium("PEC", pec=True)


def Sheet(axis, position: float, conductivity: dict, *, min: Vec3 | None = None, max: Vec3 | None = None, polygon=None, name: str = "") -> dict:
    """A zero-thickness conducting sheet, for :meth:`Simulation.add_sheet`.

    The plane ``axis = position`` (``axis`` "x", "y", "z" or 0, 1, 2)
    carries the surface current ``J_s = sigma_s(w) E_t`` (tangential E
    only), with ``conductivity`` a surface conductivity in siemens from
    :func:`lightsim.sheets.sheet_conductivity`, :func:`lightsim.sheets.graphene`
    (without a thickness) or :func:`lightsim.sheets.SheetConductivity`.
    Without ``min`` / ``max`` / ``polygon`` the sheet fills the plane;
    ``min`` / ``max`` (box corners, the components along ``axis`` ignored)
    make it a rectangle, ``polygon`` (vertices as ``(u, v)`` pairs in the two
    other axes in increasing order: y, z for a sheet normal to x; x, z for
    y; x, y for z) a polygon. The sheet need not lie on a grid node; see
    ``docs/modules/sheets.md`` for the discretisation and its accuracy. In a
    2-D simulation a sheet is a line, normal to one of the in-plane axes.
    """
    out = {"name": name, "axis": _axis(axis), "position": float(position), "conductivity": dict(conductivity)}
    if (min is None) != (max is None):
        raise ValueError("give both min and max (a rectangle), or neither")
    if min is not None:
        out["min"], out["max"] = _v3(min), _v3(max)
    if polygon is not None:
        out["polygon"] = [[float(p[0]), float(p[1])] for p in polygon]
    return out
