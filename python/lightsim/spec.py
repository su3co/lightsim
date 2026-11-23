"""Pythonic builders for the lightsim simulation specification.

Every builder returns a plain ``dict`` matching the JSON schema consumed by the
Rust core (``lightsim_core::spec::Simulation``). Units are SI (metres,
seconds, Hz); helper ``um`` converts micrometres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

C0 = 299_792_458.0
um = 1e-6
nm = 1e-9

Vec3 = Sequence[float]


def _v3(v: Vec3) -> list[float]:
    """A point or box corner: 3 components, or the 2 in-plane components of a
    2-D simulation (completed by :class:`Simulation` along its invariant
    axis)."""
    v = [float(x) for x in v]
    if len(v) not in (2, 3):
        raise ValueError(f"expected 3 components (or 2 in a 2-D simulation), got {v}")
    return v


_AXES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def _axis(a) -> int:
    try:
        return _AXES[a.lower() if isinstance(a, str) else int(a)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"unknown axis {a!r} (use 'x', 'y', 'z' or 0, 1, 2)") from None


def _lift(v, w: int, c: float) -> list[float]:
    """In-plane components (a, b) of a 2-D simulation invariant along axis
    ``w`` -> 3 components, with ``c`` along ``w``. 3-vectors pass through."""
    v = list(v)
    if len(v) == 3:
        return v
    return v[:w] + [float(c)] + v[w:]


# ----------------------------------------------------------------------------
# materials
# ----------------------------------------------------------------------------


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

# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


def Box(min: Vec3, max: Vec3) -> dict:
    """Axis-aligned box.

    Parameters
    ----------
    min, max : (x, y, z)
        Opposite corners (m). The box may extend beyond the simulation region;
        only the part inside is used.

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {"type": "box", "min": _v3(min), "max": _v3(max)}


def Sphere(center: Vec3, radius: float) -> dict:
    """Sphere.

    Parameters
    ----------
    center : (x, y, z)
        Centre (m).
    radius : float
        Radius (m).

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {"type": "sphere", "center": _v3(center), "radius": float(radius)}


def Cylinder(center: Vec3, radius: float, length: float | None = None, axis: int | str | None = None) -> dict:
    """Circular cylinder.

    Parameters
    ----------
    center : (x, y, z) or (a, b)
        Centre of the cylinder (m), half-way along its length. In a 2-D
        simulation two in-plane coordinates are enough.
    radius : float
        Radius (m).
    length : float, optional
        Length (m) along ``axis``, centred on ``center``. Required in 3-D; in
        a 2-D simulation it defaults to more than the slab.
    axis : int or str, optional
        Direction of the cylinder axis: 0, 1, 2 or "x", "y", "z". Default z in
        3-D and the invariant axis in 2-D, where ``Cylinder([a, b], radius)``
        is a disc.

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {
        "type": "cylinder",
        "center": _v3(center),
        "radius": float(radius),
        "length": None if length is None else float(length),
        "axis": None if axis is None else _axis(axis),
    }


def PolySlab(vertices: Iterable[Sequence[float]], slab_min: float | None = None, slab_max: float | None = None, axis: int | str | None = None) -> dict:
    """Polygon extruded along an axis.

    Parameters
    ----------
    vertices : sequence of (a, b)
        Corners of the polygon (m) in the plane normal to ``axis``, as pairs of
        coordinates on the two other axes in increasing axis order: (x, y) for
        ``axis=2``, (x, z) for ``axis=1``, (y, z) for ``axis=0``. The polygon is
        closed automatically, may be concave and is filled by the even-odd
        rule. It cannot have holes: cut such a shape into pieces.
    slab_min, slab_max : float, optional
        Extent (m) of the extrusion along ``axis``. Required in 3-D; in a 2-D
        simulation the extrusion covers the slab by default.
    axis : int or str, optional
        Extrusion axis: 0, 1, 2 or "x", "y", "z". Default z in 3-D and the
        invariant axis in 2-D, where ``PolySlab(vertices)`` is the polygon
        itself.

    Returns the shape dict for ``Simulation.add_structure``. Consecutive
    polygons of one medium and one extent are painted as their union.
    """
    return {
        "type": "poly_slab",
        "vertices": [[float(p[0]), float(p[1])] for p in vertices],
        "axis": None if axis is None else _axis(axis),
        "slab_min": None if slab_min is None else float(slab_min),
        "slab_max": None if slab_max is None else float(slab_max),
    }


def IndexGrid(min: Vec3, max: Vec3, eps=None, *, n=None, axes: Sequence[int] | None = None, eps_axes=None, n_axes=None) -> dict:
    """Imported / graded permittivity on a regular grid (Lumerical "import (n,k)").

    ``eps`` (relative permittivity) or ``n`` (refractive index) is a real numpy
    array of samples whose nodes span the box ``[min, max]`` inclusive along
    every axis it varies on; inside the box the solver uses the trilinear
    interpolation of the samples at each Yee E-component position. 3-D arrays
    are indexed ``[ix, iy, iz]``. 1-D and 2-D arrays are padded with length-1
    (= uniform) axes: by default a 1-D array varies along x and a 2-D array
    along (x, y); pass ``axes`` to choose, e.g. ``axes=(1, 2)`` for a (y, z)
    cross-section map of a waveguide running along x. A scalar gives a
    constant box.

    ``eps_axes=(eps_x, eps_y, eps_z)`` / ``n_axes`` (three arrays of the same
    shape) give a diagonal-anisotropic grid.

    The grid *replaces* the permittivity of the structure's medium inside the
    box: the medium passed to ``Simulation.add_structure`` only supplies its
    name, conductivity (loss), chi2/chi3 and permeability, and its ``eps`` is
    what the automatic mesher uses to pick the cell size, so set it to the
    largest permittivity of the grid. Dispersive, PEC and off-diagonal media
    are not allowed. The box faces are staircased (no sub-pixel smoothing, the
    grid is already a sampled field); to embed a grid in another material,
    add that structure first and let the grid overlap it.
    """
    import numpy as _np

    given = [x is not None for x in (eps, n, eps_axes, n_axes)]
    if sum(given) != 1:
        raise ValueError("give exactly one of eps, n, eps_axes, n_axes")

    def prep(a, is_index):
        if _np.iscomplexobj(a):
            raise ValueError("IndexGrid samples must be real; put loss in the medium's sigma")
        a = _np.asarray(a, dtype=float)
        if a.ndim > 3:
            raise ValueError(f"IndexGrid array must be at most 3-D, got shape {a.shape}")
        ax = tuple(range(a.ndim)) if axes is None else tuple(int(x) for x in axes)
        if len(ax) != a.ndim or len(set(ax)) != len(ax) or any(x not in (0, 1, 2) for x in ax):
            raise ValueError(f"axes={axes!r} does not match a {a.ndim}-D array")
        # order the array axes by simulation axis, then insert the uniform ones
        a = _np.transpose(a, _np.argsort(ax)) if a.ndim > 1 else a
        shape = [1, 1, 1]
        for d, x in zip(a.shape, sorted(ax)):
            shape[x] = d
        a = a.reshape(shape)
        if a.size == 0 or not _np.all(_np.isfinite(a)):
            raise ValueError("IndexGrid samples must be finite and non-empty")
        if is_index:
            a = a * a
        if _np.any(a <= 0.0):
            raise ValueError("IndexGrid permittivity must be positive")
        return a

    lo, hi = _v3(min), _v3(max)
    if any(h <= l for l, h in zip(lo, hi)):
        raise ValueError("IndexGrid needs max > min along every axis")
    out = {"type": "index_grid", "min": lo, "max": hi}
    if eps is not None or n is not None:
        a = prep(eps if eps is not None else n, n is not None)
        out["dims"] = list(a.shape)
        out["eps"] = a.ravel(order="C").tolist()
    else:
        src = eps_axes if eps_axes is not None else n_axes
        if len(src) != 3:
            raise ValueError("eps_axes / n_axes need three arrays (x, y, z)")
        arrs = [prep(a, n_axes is not None) for a in src]
        if len({a.shape for a in arrs}) != 1:
            raise ValueError("eps_axes / n_axes arrays must have the same shape")
        out["dims"] = list(arrs[0].shape)
        out["eps"] = []
        out["eps_axes"] = [a.ravel(order="C").tolist() for a in arrs]
    return out


def Structure(shape: dict, medium: str, name: str = "") -> dict:
    """A shape filled with a medium, as stored in ``Simulation.structures``.

    Parameters
    ----------
    shape : dict
        A shape from ``Box``, ``Sphere``, ``Cylinder``, ``PolySlab`` or
        ``IndexGrid``.
    medium : str
        Name of the medium, defined in the simulation or built in.
    name : str
        A label, used in error messages.

    Returns the structure dict. Normally created by ``Simulation.add_structure``.
    """
    return {"name": name, "shape": shape, "medium": medium}


# ----------------------------------------------------------------------------
# signals, sources, monitors
# ----------------------------------------------------------------------------


def GaussianPulse(freq0: float, fwidth: float, offset: float = 5.0, phase: float = 0.0, amplitude: float = 1.0) -> dict:
    """Gaussian pulse, the usual signal for a spectrum in one run.

    ``amplitude * cos(2 pi freq0 (t - t0) + phase) * exp(-(t - t0)**2 / (2 tau**2))``
    with ``tau = 1 / (2 pi fwidth)`` and ``t0 = offset * tau``. Its spectrum is a
    Gaussian of standard deviation ``fwidth`` around ``freq0``.

    Parameters
    ----------
    freq0 : float
        Centre frequency (Hz).
    fwidth : float
        Spectral width (Hz), the standard deviation of the spectrum.
    offset : float
        Peak time in units of ``tau``; 5 starts the pulse from a negligible value.
    phase : float
        Carrier phase (rad).
    amplitude : float
        Peak value, a factor on the source amplitude.

    Returns the signal dict for a source.
    """
    return {"type": "gaussian_pulse", "freq0": freq0, "fwidth": fwidth, "offset": offset, "phase": phase, "amplitude": amplitude}


def ContinuousWave(freq0: float, fwidth: float, offset: float = 5.0, phase: float = 0.0, amplitude: float = 1.0) -> dict:
    """Continuous wave at one frequency, switched on smoothly.

    ``amplitude * cos(2 pi freq0 t + phase)``, multiplied before ``t0 = offset * tau``
    by a Gaussian ramp of width ``tau = 1 / (2 pi fwidth)``.

    Parameters
    ----------
    freq0 : float
        Frequency (Hz).
    fwidth : float
        Sets the duration of the turn-on, ``tau = 1 / (2 pi fwidth)`` (Hz).
    offset : float
        End of the ramp in units of ``tau``.
    phase : float
        Phase (rad).
    amplitude : float
        Amplitude, a factor on the source amplitude.

    Returns the signal dict for a source. The signal never ends, so ``shutoff``
    has no effect and frequency-domain monitors sample every step.
    """
    return {"type": "continuous_wave", "freq0": freq0, "fwidth": fwidth, "offset": offset, "phase": phase, "amplitude": amplitude}


def PointDipole(position: Vec3, polarization: str, signal: dict, amplitude: float = 1.0, name: str = "") -> dict:
    """Point dipole at a single field sample.

    Parameters
    ----------
    position : (x, y, z)
        Position (m); the dipole is placed on the grid position of the component
        ``polarization`` nearest to it.
    polarization : str
        ``"Ex"``, ``"Ey"`` or ``"Ez"``: an electric current element of moment
        ``amplitude * signal(t)`` in A m. ``"Hx"``, ``"Hy"`` or ``"Hz"``: a
        magnetic current element in V m.
    signal : dict
        Time signal.
    amplitude : float
        Dipole moment amplitude (A m, or V m for a magnetic dipole).
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. An electric dipole of moment ``p`` in a medium of
    index ``n`` radiates ``n eta0 k0**2 abs(p)**2 / (12 pi)`` watts, with
    ``k0 = 2 pi f / c``.
    """
    return {
        "type": "point_dipole",
        "name": name,
        "position": _v3(position),
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
    }


def PlaneWave(
    axis: int,
    position: float,
    direction: str,
    polarization: str,
    signal: dict,
    amplitude: float = 1.0,
    angle: float = 0.0,
    min: Vec3 | None = None,
    max: Vec3 | None = None,
    name: str = "",
    fixed_angle: bool = False,
) -> dict:
    """One-sided plane-wave source on a plane.

    The wave is launched towards ``direction`` only; behind the plane there is
    no incident field.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` (0, 1, 2) at this coordinate (m),
        moved to the nearest grid node.
    direction : "+" or "-"
        Propagation direction along ``axis``.
    polarization : str
        A field component in the plane. An electric component (``"Ey"`` for
        ``axis=0``) gives E along that axis; with ``angle`` set the wave is
        s-polarised. A magnetic component (``"Hz"`` for ``axis=0``) gives H
        along that axis and E along the other in-plane axis; with ``angle``
        set the wave is p-polarised.
    signal : dict
        Time signal.
    amplitude : float
        Peak electric field (V/m), times the signal.
    angle : float
        Tilt of the wavevector (rad) towards the in-plane axis perpendicular to
        the polarisation axis: s-polarisation for an E component,
        p-polarisation for an H component (then E has the tangential part
        ``amplitude * cos(angle)`` and a normal part ``amplitude * sin(angle)``).
        It needs Bloch boundaries along the tilt axis with
        ``k = n k0 sin(angle)``, and a ``GaussianPulse`` or ``ContinuousWave``
        signal. The angle is exact at the centre frequency of the signal.
    min, max : (x, y, z), optional
        Corners limiting the extent of the plane (m); the whole plane by default.
    name : str
        Name, for ``Results.source_spectrum``.
    fixed_angle : bool
        Hold ``angle`` at every frequency of the pulse instead of only at its
        centre (broadband oblique incidence in one run). The boundaries along
        the tilt axis must then be ``PERIODIC`` (the run imposes the phase of
        each frequency itself), the other transverse axis periodic, and the
        source must be the only one; any real signal works. The run solves
        for the transformed fields ``E(r, t + n sin(angle) u / c)`` and
        returns physical frequency-domain data. See "Broadband oblique
        incidence at a fixed angle" in the sources guide for what it
        supports and costs.

    Returns the source dict. The index of the medium is taken at the centre of
    the plane; in a medium of index ``n`` the wave carries
    ``n abs(amplitude * S)**2 cos(angle) / (2 eta0)`` watts per square metre of
    the plane.
    """
    return {
        "type": "plane_wave",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
        "angle": angle,
        "min": None if min is None else _v3(min),
        "max": None if max is None else _v3(max),
        **({"fixed_angle": True} if fixed_angle else {}),
    }


def GaussianBeam(
    axis: int,
    position: float,
    direction,
    polarization: str,
    center: Vec3,
    waist: float,
    signal: dict,
    amplitude: float = 1.0,
    focus_distance: float = 0.0,
    name: str = "",
    theta: float = 0.0,
    phi: float = 0.0,
) -> dict:
    """One-sided Gaussian-beam source, at any angle to its plane.

    The beam is injected on the plane normal to ``axis`` at ``position`` and
    travels away from it towards ``+axis`` or ``-axis``.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` (0, 1, 2) at this coordinate
        (m), moved to the nearest grid node.
    direction : "+", "-" or a vector
        Side of the plane the beam is launched to. A vector ``(dx, dy, dz)``
        gives the direction of the beam axis itself; its component along
        ``axis`` sets the side, and ``theta`` and ``phi`` are computed from
        it (they must then be left at 0).
    polarization : str
        An electric component in the plane, e.g. ``"Ey"`` for ``axis=0``.
        The electric field of the beam points along the projection of this
        axis onto the plane normal to the beam axis: exactly along it when
        the tilt is perpendicular to it (s polarisation), tilted with the
        beam otherwise (p polarisation).
    center : (x, y, z)
        A point of the beam axis; its two in-plane coordinates are where the
        axis crosses the source plane (m).
    waist : float
        Radius (m) at which the field falls to 1/e at the focus.
    signal : dict
        Time signal; a tilted or defocused beam needs ``GaussianPulse`` or
        ``ContinuousWave``. The pattern is that of the centre frequency.
    amplitude : float
        Peak electric field at the focus (V/m), times the signal.
    focus_distance : float
        Distance (m) from ``center`` to the focus along the beam axis;
        negative for a beam that is already diverging at the plane. The waist
        plane is normal to the beam axis.
    theta, phi : float
        Polar angle (rad) between the beam axis and the plane normal, below
        pi/2, and azimuth of the tilt, measured from the first in-plane axis
        towards the second. The in-plane axes are (y, z) for ``axis=0``,
        (x, z) for ``axis=1`` and (x, y) for ``axis=2``.

    The pattern is the paraxial Gaussian beam: accurate for waists of about a
    wavelength or more. It spans the whole plane, so the region must be
    several waists wide where the beam crosses it. In a simulation that is
    periodic along an in-plane axis with a period below twice the waist (a
    two-dimensional simulation) the beam is uniform along that axis, a sheet
    beam, and must travel in the other plane. The beam carries the power
    ``n |amplitude S|^2 pi waist^2 / (4 eta0)``, or
    ``n |amplitude S|^2 waist sqrt(pi/2) / (2 eta0)`` per unit length of a
    sheet beam, whatever its angle.
    """
    if not isinstance(direction, str):
        import math

        if theta or phi:
            raise ValueError("give the beam direction as a vector, or as '+'/'-' with theta and phi, not both")
        d = _v3(direction)
        norm = math.sqrt(sum(c * c for c in d))
        if norm == 0.0 or d[axis] == 0.0:
            raise ValueError("the direction vector of a Gaussian beam needs a component along the source axis")
        u, v = [i for i in range(3) if i != axis]
        theta = math.acos(min(1.0, abs(d[axis]) / norm))
        phi = math.atan2(d[v], d[u]) if theta > 0 else 0.0
        direction = "+" if d[axis] > 0 else "-"
    return {
        "type": "gaussian_beam",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "center": _v3(center),
        "waist": waist,
        "signal": signal,
        "amplitude": amplitude,
        "focus_distance": focus_distance,
        "theta": float(theta),
        "phi": float(phi),
    }


def ModeSource(axis, position, direction, min, max, profile, signal, amplitude=1.0, name="") -> dict:
    """One-sided waveguide-mode source on a plane window.

    Parameters
    ----------
    axis, position : int, float
        The source plane is normal to ``axis`` at this coordinate (m), moved to
        the nearest grid node.
    direction : "+" or "-"
        Direction in which the mode is launched.
    min, max : (x, y, z)
        Corners of the mode window (m): those given to the mode solver.
    profile : dict
        Field pattern: ``Mode.profile`` from ``solve_modes`` or
        ``eim.solve_port_mode``, or a profile from ``gaussian_mode``.
    signal : dict
        Time signal; a complex profile (a lossy mode, a tilted beam) needs
        ``GaussianPulse`` or ``ContinuousWave``.
    amplitude : float
        Factor on the profile. A profile from the mode solver carries unit
        power, so the launched power spectrum is ``abs(amplitude * S)**2`` watts.
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. The part of the window inside an absorbing layer
    is not injected.
    """
    return {
        "type": "mode",
        "name": name,
        "axis": axis,
        "position": position,
        "direction": direction,
        "min": _v3(min),
        "max": _v3(max),
        "profile": profile,
        "signal": signal,
        "amplitude": amplitude,
    }


def TFSF(min: Vec3, max: Vec3, axis: int, direction: str, polarization: str, signal: dict, amplitude=1.0, name="") -> dict:
    """Total-field/scattered-field box with a plane wave inside.

    The incident wave exists inside the box only; outside it only the field
    scattered by what the box contains remains.

    Parameters
    ----------
    min, max : (x, y, z)
        Corners of the box (m), snapped to grid nodes.
    axis : int
        Propagation axis of the incident wave (normal incidence on the faces).
    direction : "+" or "-"
        Propagation direction along ``axis``.
    polarization : str
        Electric component of the wave, perpendicular to ``axis``.
    signal : dict
        Time signal.
    amplitude : float
        Peak electric field (V/m), times the signal.
    name : str
        Name, for ``Results.source_spectrum``.

    Returns the source dict. The medium around the box must be uniform (it may
    be dispersive). A box that spans a periodic axis completely has no faces
    normal to that axis, which gives the incident wave of a 2-D problem.
    """
    return {
        "type": "tfsf",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "axis": axis,
        "direction": direction,
        "polarization": polarization.capitalize(),
        "signal": signal,
        "amplitude": amplitude,
    }


def FieldTimeMonitor(name, min, max, components=("Ex", "Ey", "Ez"), interval=1, start=0.0) -> dict:
    """Fields in a box at regular time intervals.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.time_series(name, component)``.
    min, max : (x, y, z)
        Corners of the box (m). A zero extent records a plane, a line or a point.
    components : sequence of str
        Field components to record, from ``"Ex"`` ... ``"Hz"``.
    interval : int
        Record every ``interval`` steps.
    start : float
        Time (s) of the first sample.

    Returns the monitor dict. All samples stay in memory until the end of the
    run: 4 bytes per cell, component and sample.
    """
    return {"type": "field_time", "name": name, "min": _v3(min), "max": _v3(max), "components": list(components), "interval": interval, "start": start}


def ProbeMonitor(name, position, components=("Ez",)) -> dict:
    """Fields at one point at every time step.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.time_series(name, component)``.
    position : (x, y, z)
        Position (m); each component is taken at its nearest grid position.
    components : sequence of str
        Field components to record.

    Returns the monitor dict.
    """
    return {"type": "probe", "name": name, "position": _v3(position), "components": list(components)}


def DFTMonitor(name, min, max, freqs, components=("Ex", "Ey", "Ez"), stride=None, apodization=None) -> dict:
    """Frequency-domain fields in a box, a plane, a line or a point.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.field(name, component)``.
    min, max : (x, y, z)
        Corners of the box (m). Along an axis of zero extent the nearest layer of
        each component is taken.
    freqs : sequence of float
        Frequencies (Hz).
    components : sequence of str
        Field components to record, from ``"Ex"`` ... ``"Hz"``.
    stride : int, optional
        Accumulate the Fourier sums every ``stride`` steps. ``None`` takes
        ``Simulation.dft_stride``, which by default is chosen from the bandwidth
        of the sources so that the result equals that of every step.
    apodization : dict, optional
        Time window of the sums: keys ``start``, ``end`` (s) and ``width`` (s)
        of a Gaussian taper at each edge, all optional.

    Returns the monitor dict. The monitor stores two 4-byte numbers per sample,
    component and frequency.
    """
    return {
        "type": "dft_field",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "components": list(components),
        "freqs": [float(f) for f in freqs],
        "stride": 0 if stride is None else int(stride),
        "apodization": apodization or {},
    }


def FluxMonitor(name, axis, position, min, max, freqs, apodization=None) -> dict:
    """Power through a plane, per frequency.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.flux(name)``.
    axis, position : int, float
        The plane is normal to ``axis`` at this coordinate (m), moved to the
        nearest grid node.
    min, max : (x, y, z)
        Corners limiting the plane (m); they may lie outside the region.
    freqs : sequence of float
        Frequencies (Hz).
    apodization : dict, optional
        Time window of the Fourier sums, as for ``DFTMonitor``.

    Returns the monitor dict. The power is that of the time-averaged Poynting
    vector of the recorded spectra, in watts times the squared source spectrum,
    positive towards ``+axis``.
    """
    return {
        "type": "flux",
        "name": name,
        "axis": axis,
        "position": position,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "apodization": apodization or {},
    }


def FluxBoxMonitor(name, min, max, freqs) -> dict:
    """Net power leaving a closed box, per frequency.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.flux(name)``.
    min, max : (x, y, z)
        Corners of the box (m), snapped to grid nodes.
    freqs : sequence of float
        Frequencies (Hz).

    Returns the monitor dict. The result is positive for a source inside the
    box, negative for absorption inside it; around a ``TFSF`` box it is the
    scattered power. A face on an electric or magnetic wall is skipped.
    """
    return {"type": "flux_box", "name": name, "min": _v3(min), "max": _v3(max), "freqs": [float(f) for f in freqs]}


def ModeMonitor(name, axis, position, min, max, freqs, modes) -> dict:
    """Amplitudes of waveguide modes on a plane window.

    Parameters
    ----------
    name : str
        Monitor name; read the result with ``Results.mode_amplitudes(name, mode)``.
    axis, position : int, float
        The plane is normal to ``axis`` at this coordinate (m), moved to the
        nearest grid node.
    min, max : (x, y, z)
        Corners of the mode window (m): those given to the mode solver.
    freqs : sequence of float
        Frequencies (Hz).
    modes : sequence of dict
        Mode profiles (``Mode.profile``, or from ``gaussian_mode``) solved on
        this window. If there are as many profiles as frequencies, and more
        than one, they are taken as one mode with one profile per frequency.

    Returns the monitor dict. For unit-power profiles ``abs(a)**2`` is the power
    in the mode. Monitors on one plane with the same frequencies share the
    fields they record, so many of them cost what one costs.
    """
    return {
        "type": "mode",
        "name": name,
        "axis": axis,
        "position": position,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "modes": list(modes),
    }


def FarFieldMonitor(name, min, max, freqs, theta, phi, n_medium: float = 1.0, apodization=None) -> dict:
    """Far field radiated by everything inside a box.

    The fields on the six faces of the box ``min``..``max`` (m, snapped to
    grid nodes) are recorded at the frequencies ``freqs`` (Hz), turned into
    equivalent surface currents and radiated into a homogeneous medium of
    refractive index ``n_medium``. Read the result with
    ``Results.far_field(name)``.

    Parameters
    ----------
    theta, phi : sequences of float
        Directions (radians): ``theta`` is the polar angle from the +z axis,
        ``phi`` the azimuth from the +x axis in the xy plane. The pattern is
        returned on the grid ``theta`` x ``phi``.
    n_medium : float
        Index of the medium outside the box, which must be homogeneous: a
        substrate or other layers outside the box are not accounted for.
    apodization : dict, optional
        Time window of the Fourier sums, as for :func:`DFTMonitor`.

    The box must be closed, lie inside the simulation region and enclose
    all sources and scatterers whose radiation is wanted.
    """
    return {
        "type": "far_field",
        "name": name,
        "min": _v3(min),
        "max": _v3(max),
        "freqs": [float(f) for f in freqs],
        "theta": [float(t) for t in theta],
        "phi": [float(p) for p in phi],
        "n_medium": float(n_medium),
        "apodization": apodization or {},
    }


def EnergyMonitor(name, min, max, interval=1) -> dict:
    """Electromagnetic energy stored in a box, versus time.

    Every ``interval`` steps the monitor evaluates

        U = eps0/2 * sum( eps |E|^2 + mu |eta0 H|^2 ) dV      (joules)

    over the cells of the box ``min``..``max`` (m), with the permittivity
    of each cell as the engine uses it. Cells cut by the box count with the
    part inside it. Read the result with ``Results.energy(name)``.

    The electric field is sampled at the end of a step and the magnetic field
    half a step earlier, so the sum ripples at twice the optical frequency by
    a fraction of about ``(omega dt / 2)**2``. For a dispersive medium ``eps``
    is its high-frequency value: the energy stored in the polarisation of the
    poles is not included. The six field components of the box are kept in
    memory for every sample until the end of the run (24 bytes per cell and
    sample), so keep the box small or the interval long.
    """
    return {"type": "energy", "name": name, "min": _v3(min), "max": _v3(max), "interval": int(interval)}


# ----------------------------------------------------------------------------
# boundaries and mesh
# ----------------------------------------------------------------------------


def PML(
    layers: int = 12,
    order: float = 3.0,
    sigma_scale: float = 1.0,
    kappa_max: float = 2.0,
    alpha_max: float = 0.0,
    inside: bool = False,
    stabilize: bool = True,
) -> dict:
    """CPML boundary. ``inside=True`` puts the layers on the outermost cells of
    the region (FDTDX convention) instead of adding them outside.

    ``stabilize`` (on by default) changes, inside the layers only and when the
    region holds a structure of another medium than the background, the
    lossless poles of transparent media that lie outside the band of the
    sources: the far-infrared pole of a fitted glass
    (:func:`lightsim.materials.fitted_medium`) is damped, and a strong
    ultraviolet pole (silicon fitted with ``eps_inf = 1``) is frozen at its
    value at the centre of the band. Without it a waveguide of such media
    running into the layer makes a long run diverge: the medium is metal-like
    in a band above each pole, the guide carries backward modes there, and a
    PML amplifies backward waves. The change reflects about 1e-7 of the power
    at the face of the layer for fused silica near 1.55 um, and at most 1e-4
    at the ends of a wide band; poles in or near the band are left as they
    are. Damping the infrared pole has removed the growth in every case
    tried; freezing the ultraviolet pole removes it for silicon guides in a
    glass cladding and only delays it for some other structures (the
    boundaries guide has the measurements). :func:`Absorber` is the boundary
    that cannot amplify."""
    return {
        "type": "pml",
        "layers": layers,
        "params": {"order": order, "sigma_scale": sigma_scale, "kappa_max": kappa_max, "alpha_max": alpha_max, "stabilize": stabilize},
        "inside": inside,
    }


def Absorber(layers: int = 40, order: float = 3.0, sigma_scale: float = 1.0) -> dict:
    """Adiabatic absorber: a boundary that can replace a PML where a PML is
    unstable. ``layers`` cells are added outside the region, in which an
    electric and a magnetic conductivity rise smoothly (as the power ``order``
    of the depth) from zero. The two are matched to each other in every
    medium, at the centre of the source band, so a wave decays without a
    change of its impedance or of its mode profile.

    An absorber is a passive medium: it cannot amplify any wave. A PML can:
    it amplifies backward waves, which fitted (dispersive) media with a band
    of negative permittivity, metals with surface plasmons, and some photonic
    crystal guides carry (:func:`PML` describes the usual case and what
    ``stabilize`` does about it). Use an absorber on the faces that such a
    structure runs through when a run with PML diverges.

    The price is thickness. An absorber reflects wherever its conductivity
    changes, so the change must be slow on the scale of the wavelength. At
    1.55 um a layer 1 um thick (one wavelength in a glass cladding: 40 layers
    of 25 nm) reflects 1e-5 to 1e-4 of a guided mode, 0.5 um about 5e-4, and
    1.5 um as little as a 12-layer PML (a few 1e-6): choose ``layers`` from
    the thickness. The boundaries guide gives the measured values. ``sigma_scale`` multiplies the peak conductivity: the default
    attenuates a plane wave in the background medium by 6 nepers in amplitude
    (1e-5 in power) on the way to the outer wall and back, and a guided mode
    in proportion to its group index.

    Absorbers and PML can be mixed, face by face. Only media with a diagonal
    permittivity, conductivity and per-axis poles can run through an
    absorber; the fixed-angle engine and the ``reverse`` gradient engine do
    not take one, and design regions must stay outside it."""
    return {"type": "absorber", "layers": layers, "params": {"order": order, "sigma_scale": sigma_scale}}


PERIODIC = {"type": "periodic"}
PEC_WALL = {"type": "pec"}
PMC_WALL = {"type": "pmc"}


def Bloch(k: float) -> dict:
    """Bloch-periodic boundary condition.

    The field at the upper face of the axis is the field at the lower face times
    ``exp(1j * k * L)``, with ``L`` the length of the region along the axis.

    Parameters
    ----------
    k : float
        Wavevector component along the axis (rad/m); ``n k0 sin(theta)`` for a
        plane wave incident at ``theta`` in a medium of index ``n``.

    Returns the boundary dict, to be set on both faces of the axis with
    ``Simulation.set_boundaries``. A run with ``k != 0`` uses complex fields.
    """
    return {"type": "bloch", "k": k}


def UniformMesh(dx: float | Sequence[float]) -> dict:
    """Uniform mesh.

    Parameters
    ----------
    dx : float or (dx, dy, dz)
        Cell size (m), one value or one per axis. It is adjusted so that a whole
        number of cells fits the region along each axis. In a 2-D simulation
        ``dx`` may hold the two in-plane spacings; the spacing along the
        invariant axis is not used.

    Returns the mesh dict for ``Simulation``.
    """
    d = [float(dx)] * 3 if isinstance(dx, (int, float)) else [float(v) for v in dx]
    return {"type": "uniform", "dx": d}


def AutoMesh(wavelength: float, steps_per_wavelength: float = 20.0, grading: float = 1.3, min_step=None, max_step=None, overrides=(), symmetry=None) -> dict:
    """Automatic graded mesh.

    The cell size in each medium is ``wavelength / (n * steps_per_wavelength)``
    with ``n`` the refractive index of the medium, and it grows away from a
    structure by at most the factor ``grading`` per cell.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength (m) the resolution refers to; use the shortest
        wavelength of interest.
    steps_per_wavelength : float
        Cells per wavelength in each medium.
    grading : float
        Largest ratio of neighbouring cell sizes, about.
    min_step, max_step : float, optional
        Lower bound (m) on the cell size chosen for structures, upper bound
        on the cell size of the background.
    overrides : sequence of (min, max, (dx, dy, dz))
        Boxes with an explicit largest cell size (m) per axis.
    symmetry : (x0, y0, z0), optional
        Mirror planes of the mesh, one coordinate (m) per axis or ``None``.
        A plane gets a grid node and the cells on its two sides are mirror
        images of each other. Without it the mesh is built outwards from the
        centre of the region along each axis, which is already symmetric for
        structures that are symmetric about that centre; name a plane when
        the symmetry plane of the device is somewhere else.

    Returns the mesh dict for :class:`Simulation`.
    """
    sym = [None, None, None] if symmetry is None else [None if v is None else float(v) for v in symmetry]
    if len(sym) != 3:
        raise ValueError("symmetry needs three entries (x0, y0, z0), each a coordinate or None")
    return {
        "type": "auto",
        "wavelength": wavelength,
        "steps_per_wavelength": steps_per_wavelength,
        "grading": grading,
        "min_step": min_step,
        "max_step": max_step,
        "overrides": [{"min": _v3(o[0]), "max": _v3(o[1]), "dx": _v3(o[2])} for o in overrides],
        "symmetry": sym,
    }


def ExplicitMesh(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> dict:
    """Mesh given by its node coordinates.

    ``x``, ``y`` and ``z`` are increasing lists of the cell edges (m) along
    each axis, at least two per axis. The first and the last node of each
    list should be the ``min`` and ``max`` of the simulation region;
    absorbing layers are added outside them with the size of the outermost
    cell. :meth:`Simulation.frozen_mesh` returns the grid of a simulation in
    this form.
    """
    out = {"type": "explicit"}
    for k, v in zip("xyz", (x, y, z)):
        nodes = [float(c) for c in v]
        if len(nodes) < 2 or any(b <= a for a, b in zip(nodes[:-1], nodes[1:])):
            raise ValueError(f"explicit mesh: the {k} nodes must be at least two increasing coordinates")
        out[k] = nodes
    return out


def _complete(obj, w: int, lo: float, hi: float):
    """Copy of a source / monitor / shape / mesh dict of a 2-D simulation
    with every 2-component vector completed along the invariant axis ``w``
    (the slab is ``[lo, hi]``): box corners span the slab, points sit on its
    mid-plane, cylinders and polygons without an axis are normal to the
    plane."""
    if not isinstance(obj, dict):
        return obj
    mid = 0.5 * (lo + hi)
    out = dict(obj)
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _complete(v, w, lo, hi)
        elif k == "overrides" and isinstance(v, list):
            out[k] = [_complete(o, w, lo, hi) for o in v]
        elif k in ("min", "max", "position", "center", "dx") and isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = _lift(v, w, {"min": lo, "max": hi, "dx": hi - lo}.get(k, mid))
    kind = obj.get("type")
    if kind == "cylinder":
        if out.get("axis") is None:
            out["axis"] = w
        if out.get("length") is None:
            out["length"] = 4.0 * (hi - lo)
    elif kind == "poly_slab":
        if out.get("axis") is None:
            out["axis"] = w
        if out.get("slab_min") is None:
            out["slab_min"] = lo - (hi - lo)
        if out.get("slab_max") is None:
            out["slab_max"] = hi + (hi - lo)
    return out


def _complete_3d(obj):
    """3-D simulations: defaults of the shapes that a 2-D one would infer,
    and a clear error for 2-component vectors."""
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _complete_3d(v)
        elif k in ("min", "max", "position", "center") and isinstance(v, (list, tuple)) and len(v) == 2:
            raise ValueError(f"{k}={list(v)} has 2 components: give 3, or make the simulation 2-D (Simulation(..., dimensions=2))")
    kind = obj.get("type")
    if kind in ("cylinder", "poly_slab") and out.get("axis") is None:
        out["axis"] = 2
    if kind == "cylinder" and out.get("length") is None:
        raise ValueError("Cylinder needs a length in a 3-D simulation")
    if kind == "poly_slab" and (out.get("slab_min") is None or out.get("slab_max") is None):
        raise ValueError("PolySlab needs slab_min and slab_max in a 3-D simulation")
    return out


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------


@dataclass
class Simulation:
    """A lightsim simulation. Build it, then call :meth:`run`.

    **3-D** (default): ``min`` / ``max`` are the corners of the region.

    **2-D** (``dimensions=2``, or ``min`` / ``max`` with two components): the
    fields do not depend on the coordinate along ``invariant_axis`` ("x",
    "y" or "z", default "z"). The grid has one cell along that axis, the
    boundaries there are periodic, the time step follows the 2-D Courant
    limit, and the solver updates three field components per polarization
    instead of six:

    * ``polarization="te"``: E in the plane, H along the invariant axis --
      (Ex, Ey, Hz) for the invariant axis z;
    * ``polarization="tm"``: E along the invariant axis, H in the plane --
      (Ez, Hx, Hy) for z;
    * ``polarization="both"`` (default): the two sets, which are independent;
      only the sets that a source drives are updated.

    (The naming of Lumerical's 2-D FDTD, Meep and Taflove's TE_z / TM_z. In
    a top view of a chip, z-invariant, "te" holds the quasi-TE waveguide
    modes; in a side view of a slab, e.g. y-invariant, a slab TE mode has E
    along y and belongs to "tm".)

    The region's extent along the invariant axis is the thickness of the
    slab the 2-D model stands for: with two-component ``min`` / ``max`` it is
    ``thickness`` (default 1 m, centred on 0), so fluxes and dipole powers
    are per metre. Structures are cut by the mid-plane of the slab, whatever
    their extent along the invariant axis, and sources and monitors span it.
    Inside a 2-D simulation every point or box corner may be given by its two
    in-plane components.
    """

    min: Vec3
    max: Vec3
    mesh: dict
    run_time: float
    boundaries: dict | None = None
    media: list[dict] = field(default_factory=list)
    structures: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    monitors: list[dict] = field(default_factory=list)
    background: str = "vacuum"
    shutoff: float | None = None
    shutoff_interval: int = 200
    courant: float = 0.99
    subpixel: bool = True
    name: str = ""
    dft_stride: int | None = None
    dimensions: int = 3
    invariant_axis: int | str = "z"
    polarization: str = "both"
    thickness: float = 1.0
    force_3d_kernels: bool = False
    sheets: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.min, self.max = list(self.min), list(self.max)
        if len(self.min) != len(self.max) or len(self.min) not in (2, 3):
            raise ValueError("min and max need 3 components each (or 2 each for a 2-D simulation)")
        if len(self.min) == 2:
            self.dimensions = 2
        if self.dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3")
        self.invariant_axis = _axis(self.invariant_axis)
        self.polarization = str(self.polarization).lower()
        if self.polarization not in ("te", "tm", "both"):
            raise ValueError("polarization must be 'te', 'tm' or 'both'")
        if self.dimensions == 3 and self.polarization != "both":
            raise ValueError("polarization applies to 2-D simulations: pass dimensions=2 (or 2-component min / max)")
        w = self.invariant_axis
        if len(self.min) == 2:
            half = 0.5 * float(self.thickness)
            self.min, self.max = _lift(self.min, w, -half), _lift(self.max, w, half)
        if self.boundaries is None:
            self.boundaries = {"x": [PML(), PML()], "y": [PML(), PML()], "z": [PML(), PML()]}
        if self.dimensions == 2:
            self.boundaries = dict(self.boundaries)
            self.boundaries["xyz"[w]] = [dict(PERIODIC), dict(PERIODIC)]

    @property
    def slab(self) -> tuple[int, float, float] | None:
        """2-D simulations: (invariant axis, low face, high face) of the slab
        the model stands for; ``None`` for a 3-D simulation."""
        if self.dimensions != 2:
            return None
        w = self.invariant_axis
        return w, float(self.min[w]), float(self.max[w])

    def _resolve(self, obj: dict) -> dict:
        s = self.slab
        return _complete_3d(obj) if s is None else _complete(obj, *s)

    # -- builders --
    def add_medium(self, m: dict) -> "Simulation":
        """Register a medium.

        A medium whose name is already registered is ignored. ``m`` is a dict from
        ``Medium``. Returns the simulation.
        """
        if not any(x["name"] == m["name"] for x in self.media):
            self.media.append(m)
        return self

    def add_structure(self, shape: dict, medium: str | dict, name: str = "") -> "Simulation":
        """Add a shape filled with a medium.

        ``medium`` is the name of a medium, or a ``Medium`` dict, which is registered
        as well. Structures are painted in the order they are added: where two
        overlap, the later one wins. Returns the simulation.
        """
        if isinstance(medium, dict):
            self.add_medium(medium)
            medium = medium["name"]
        self.structures.append(Structure(self._resolve(shape), medium, name))
        return self

    def add_source(self, s: dict) -> "Simulation":
        """Add a source (a dict from ``PointDipole``, ``PlaneWave``, ``GaussianBeam``,
        ``ModeSource`` or ``TFSF``). A source name may be used once. Returns the
        simulation.
        """
        name = s.get("name", "")
        if name and any(q.get("name", "") == name for q in self.sources):
            raise ValueError(f"a source named {name!r} exists already; source spectra are looked up by name")
        self.sources.append(self._resolve(s))
        return self

    def add_monitor(self, m: dict) -> "Simulation":
        """Add a monitor (a dict from one of the ``...Monitor`` builders). Every
        monitor needs its own name, whatever its type: results are looked up by
        name. Returns the simulation.
        """
        name = m.get("name", "")
        if any(q.get("name", "") == name for q in self.monitors):
            raise ValueError(f"a monitor named {name!r} exists already; results are looked up by name, so every monitor needs its own (whatever its type)")
        self.monitors.append(self._resolve(m))
        return self

    def add_sheet(self, sheet: dict) -> "Simulation":
        """Add a zero-thickness conducting sheet (a dict from :func:`Sheet`).
        Returns the simulation."""
        self.sheets.append(self._resolve(sheet))
        return self

    def set_boundaries(self, x=None, y=None, z=None) -> "Simulation":
        """Set the boundary conditions of the given axes.

        Each of ``x``, ``y``, ``z`` is one condition for both faces of that axis, or
        a pair ``[low, high]``: ``PML(...)``, ``PERIODIC``, ``Bloch(k)``,
        ``PEC_WALL`` or ``PMC_WALL``. Periodic and Bloch conditions must be set on
        both faces. Axes left at ``None`` keep their conditions. Returns the
        simulation.
        """
        for k, v in (("x", x), ("y", y), ("z", z)):
            if v is not None:
                self.boundaries[k] = list(v) if isinstance(v, (list, tuple)) else [v, v]
        if self.dimensions == 2:
            self.boundaries["xyz"[self.invariant_axis]] = [dict(PERIODIC), dict(PERIODIC)]
        return self

    # -- serialization --
    def to_dict(self) -> dict:
        """The simulation as a plain dictionary in the JSON format of the core.
        """
        r = self._resolve
        mesh = r(self.mesh)
        d = {
            "name": self.name,
            "region": {"min": [float(v) for v in self.min], "max": [float(v) for v in self.max]},
            "mesh": mesh,
            "media": self.media,
            "background": self.background,
            "structures": [dict(s, shape=r(s["shape"])) for s in self.structures],
            "sources": [{k: v for k, v in r(s).items() if k != "fixed_angle"} for s in self.sources],
            "monitors": [r(m) for m in self.monitors],
            "boundaries": self.boundaries,
            "run_time": self.run_time,
            "shutoff": self.shutoff,
            "shutoff_interval": self.shutoff_interval,
            "courant": self.courant,
            "subpixel": self.subpixel,
            "dft_stride": self.dft_stride,
        }
        if self.dimensions == 2:
            d["dimensions"] = 2
            d["invariant_axis"] = self.invariant_axis
            d["polarization"] = self.polarization
        if self.force_3d_kernels:
            d["force_3d_kernels"] = True
        # a plane wave with fixed_angle=True makes the run a fixed-angle run
        # (a simulation-level switch in the JSON format of the core)
        if any(s.get("fixed_angle") for s in self.sources):
            d["fixed_angle"] = True
        if self.sheets:
            d["sheets"] = [r(q) for q in self.sheets]
        return d

    def to_json(self) -> str:
        """The simulation as a JSON string, as read by ``lightsim run``.
        """
        return json.dumps(self.to_dict())

    def geometry_json(self) -> str:
        """The simulation as JSON without its sources and monitors: all that
        the grid and the permittivity of a plane depend on. The mode solvers
        send this to the core, so that their cost does not grow with the mode
        profiles already attached to monitors and sources."""
        d = self.to_dict()
        d["sources"] = []
        d["monitors"] = []
        return json.dumps(d)

    # -- core calls --
    def grid(self):
        """Node coordinates ``(x, y, z)`` of the grid as numpy arrays (m), absorbing
        layers included. Nothing is run.
        """
        from . import _core

        return _core.build_grid(self.to_json())

    def frozen_mesh(self) -> dict:
        """The grid of this simulation as an explicit mesh.

        Returns the mesh dict (see :func:`ExplicitMesh`) with the node
        coordinates the simulation has now, inside the region; absorbing
        layers added outside the region are not part of a mesh and come back
        when the mesh is used. An automatic mesh depends on the structures,
        so a copy of the simulation with structures removed or added gets
        another grid. Give the copy this mesh to keep the grid, for example
        to solve port modes on a single waveguide of a larger device:

            mesh = device.frozen_mesh()
            single = Simulation(min=device.min, max=device.max, mesh=mesh, ...)
        """
        import numpy as np

        nodes = self.grid()
        out = {"type": "explicit"}
        for k, g in zip("xyz", nodes):
            g = np.asarray(g, dtype=float)
            lo, hi = (int(b.get("layers", 40)) if b.get("type") == "absorber" else int(b.get("layers", 12)) if b.get("type") == "pml" and not b.get("inside", False) else 0 for b in self.boundaries[k])
            out[k] = g[lo : len(g) - hi].tolist()
        return out

    def freeze_mesh(self) -> "Simulation":
        """Replace the mesh by the explicit mesh of the current grid
        (:meth:`frozen_mesh`), so that later changes of the structures no
        longer move the grid. Returns the simulation."""
        self.mesh = self.frozen_mesh()
        return self

    def time_info(self) -> tuple[float, int]:
        """``(dt, steps)``: the time step in seconds and the number of steps that
        ``run_time`` gives. The time step is the Courant step of the mesh times
        ``courant``, lowered if a dispersive medium needs it. Nothing is run.
        """
        from . import _core

        return _core.time_info(self.to_json())

    def run(self, device: str = "auto", progress: int = 0) -> "Results":
        """Run the simulation.

        Parameters
        ----------
        device : str
            ``"auto"`` (the first GPU if there is one, else the CPU), ``"cpu"``,
            ``"cuda"``, ``"cuda:1"``, ``"cuda:0,1"`` (split over several GPUs) or
            ``"cuda:all"``.
        progress : int
            Print a line every ``progress`` steps; 0 prints nothing.

        Returns a ``Results``. A run whose fields grow without bound raises a
        ``RuntimeError``; one that reaches ``run_time`` before the fields have
        decayed below ``shutoff`` returns with a warning.
        """
        from . import _core

        res = Results(_core.run(self.to_json(), device, progress))
        left = res.decay()
        if self.shutoff and not res.stats["stopped_early"] and left is not None and left > self.shutoff:
            # every source has an end (a continuous wave never decays)
            if all(q["signal"].get("type") in ("gaussian_pulse", "ricker") for q in self.sources):
                import warnings

                warnings.warn(
                    f"the run reached run_time before the fields had decayed: their energy is {left:.1e} of its peak, "
                    f"above shutoff = {self.shutoff:.1e}. Spectra are less converged than in a run that stops on "
                    "shutoff; increase run_time, or look at Results.stats['energy'] to see how the fields decay.",
                    stacklevel=2,
                )
        return res

    def view(self, results=None, **kwargs) -> str:
        """Open the graphical interface for this simulation (and optionally
        its ``results``) and return its URL. Keyword arguments go to
        :func:`lightsim.gui.view` (``host``, ``port``, ``open_browser``,
        ``block``)."""
        from . import gui

        return gui.view(self, results, **kwargs)


class Results:
    """Thin wrapper around the result dict returned by the core."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def stats(self) -> dict:
        """Run statistics: ``backend``, ``cells``, ``steps`` (executed),
        ``seconds``, ``mcells_per_s`` (million cell updates per second),
        ``stopped_early``, and the arrays ``energy_steps`` and ``energy``, the
        field energy that ``shutoff`` looks at, every ``shutoff_interval`` steps.
        """
        return self.raw["stats"]

    @property
    def dt(self) -> float:
        """Time step in seconds.
        """
        return self.raw["dt"]

    def decay(self):
        """Field energy at the end of the run as a fraction of its peak, the
        number that ``shutoff`` is compared with (``None`` if the run was too
        short for a sample). ``stats["energy_steps"]`` and
        ``stats["energy"]`` hold the whole history."""
        import numpy as np

        e = np.asarray(self.stats.get("energy", []), dtype=float)
        if e.size == 0 or not np.max(e) > 0:
            return None
        return float(e[-1] / np.max(e))

    def flux(self, name: str):
        """Power per frequency of a ``FluxMonitor`` or ``FluxBoxMonitor`` (also the
        total power through the plane of a ``ModeMonitor`` and through the box of a
        ``FarFieldMonitor``), as an array in the order of the monitor frequencies.
        """
        return self.raw["flux"][name]

    def field(self, monitor: str, comp: str):
        """One field component of a ``DFTMonitor``: a complex array indexed
        ``[frequency, x, y, z]``, in single precision. Magnetic components are
        multiplied by the impedance of free space (376.73 ohm), so they have the
        unit of the electric field; divide by it to obtain A/m.
        """
        import numpy as np

        f = self.raw["dft"][monitor]["fields"][comp]
        return np.asarray(f["re"]) + 1j * np.asarray(f["im"])

    def freqs(self, monitor: str):
        """Frequencies (Hz) of a frequency-domain monitor, in the order given.
        """
        return self.raw["dft"][monitor]["freqs"]

    def mode_amplitudes(self, monitor: str, mode: int = 0):
        """(a_plus, a_minus) complex arrays over frequency."""
        import numpy as np

        m = self.raw["modes"][monitor]
        p = m["plus"][mode]
        q = m["minus"][mode]
        return np.asarray(p["re"]) + 1j * np.asarray(p["im"]), np.asarray(q["re"]) + 1j * np.asarray(q["im"])

    def time_series(self, monitor: str, comp: str):
        """``(times, values)`` of one component of a ``ProbeMonitor`` or
        ``FieldTimeMonitor``: the sample times in seconds and an array indexed
        ``[sample, x, y, z]``. The times are those of the electric field; magnetic
        samples are half a time step earlier and are multiplied by the impedance
        of free space.
        """
        t = self.raw["time"][monitor]
        return t["times"], t["fields"][comp]

    def gain_levels(self):
        """Gain media: the number of atoms in levels 0, 1, 2 and 3 at the end
        of the run, summed over all gain cells (density times the volume of
        each E-field sample), as a numpy array; ``None`` without gain media.
        For four-level two-electron media these are electrons (two per
        atom)."""
        import numpy as np

        v = self.raw.get("gain_levels")
        return np.asarray(v) if v else None

    def far_field(self, monitor: str) -> dict:
        """Far field of a :func:`FarFieldMonitor`, as a dict of numpy arrays:

        - ``freqs`` (Hz), ``theta``, ``phi`` (radians): the axes;
        - ``intensity``: radiant intensity ``U`` in W/sr, shape
          ``[frequency, theta, phi]``; its integral over the sphere,
          ``sum U sin(theta) dtheta dphi``, is the radiated power;
        - ``e_theta``, ``e_phi``: complex far-field amplitudes in volts, same
          shape. Far from the box the electric field is
          ``(e_theta theta^ + e_phi phi^) exp(i k r) / r`` with ``r`` the
          distance from the coordinate origin, and
          ``U = (|e_theta|^2 + |e_phi|^2) / (2 eta)`` with ``eta`` the wave
          impedance of the outer medium, 376.73 ohm / ``n_medium``;
        - ``flux``: power through the box per frequency (W), for comparison
          with the integral of ``U``.

        Like every frequency-domain result, the values scale with the source
        spectrum (fields with ``S``, intensity and flux with ``abs(S)**2``).
        """
        import numpy as np

        f = self.raw["far_field"][monitor]
        out = {k: np.asarray(f[k]) for k in ("freqs", "theta", "phi", "intensity", "flux")}
        out["e_theta"] = np.asarray(f["e_theta_re"]) + 1j * np.asarray(f["e_theta_im"])
        out["e_phi"] = np.asarray(f["e_phi_re"]) + 1j * np.asarray(f["e_phi_im"])
        return out

    def energy(self, monitor: str):
        """``(times, energy)`` of an :func:`EnergyMonitor`: the sample times
        in seconds and the electromagnetic energy in the monitor box in
        joules, as numpy arrays."""
        e = self.raw["energy"][monitor]
        return e["times"], e["energy"]

    def source_spectrum(self, name: str | None = None, freqs=None, monitor: str | None = None):
        """Spectrum of a source, for normalising monitor data.

        Returns ``(freqs, spectrum)``: the frequencies in Hz and the complex
        Fourier transform ``S`` of the source's signal at those frequencies,
        in the order given. Dividing a monitor field by ``S``, or a power by
        ``abs(S)**2``, gives the response per unit source ``amplitude``.

        Parameters
        ----------
        name : str, optional
            Name of the source; unnamed sources are ``"source0"``,
            ``"source1"``, ... May be omitted if there is one source.
        freqs : sequence of float, optional
            Frequencies (Hz) to evaluate at, in any order.
        monitor : str, optional
            Evaluate at the frequencies of this frequency-domain monitor, in
            the monitor's order, so that the result lines up with
            ``flux(monitor)``, ``field(monitor, ...)`` or
            ``mode_amplitudes(monitor)``.

        With neither ``freqs`` nor ``monitor``, the frequencies are those of
        the frequency-domain monitors if they all record the same list in the
        same order. If they differ, a ``ValueError`` asks for one of the two
        arguments, because a spectrum on some other grid would be divided
        into the wrong monitor samples without any sign of it.

        ``S`` is computed from the time samples of the run that was made
        (same time step and number of steps), so it includes the truncation
        of a run that stopped early. For a source with a complex field
        pattern in a run with complex fields (an oblique plane wave with
        Bloch boundaries) it is the spectrum of the complex signal that was
        injected, which is twice that of the real pulse.
        """
        import numpy as np

        from . import _core

        sources = self.raw.get("sources", [])
        if name is None:
            if len(sources) != 1:
                raise ValueError(f"the run has {len(sources)} sources ({[q['name'] for q in sources]}); name one")
            src = sources[0]
        else:
            match = [q for q in sources if q["name"] == name]
            if not match:
                raise KeyError(f"no source named {name!r}; the sources are {[q['name'] for q in sources]}")
            src = match[0]
        if freqs is not None and monitor is not None:
            raise ValueError("give freqs or monitor, not both")
        if monitor is not None:
            freqs = self.freqs(monitor)
        elif freqs is None:
            lists = {m: np.asarray(d["freqs"], dtype=float) for m, d in self.raw["dft"].items()}
            if not lists:
                raise ValueError("the run has no frequency-domain monitor; pass freqs=")
            first = next(iter(lists.values()))
            if any(len(f) != len(first) or np.any(f != first) for f in lists.values()):
                raise ValueError(
                    "the monitors record different frequency lists ("
                    + ", ".join(f"{m}: {len(f)}" for m, f in lists.items())
                    + "); pass monitor=<name> or freqs=<frequencies> to say which the spectrum is for"
                )
            freqs = first
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        re, im = _core.signal_spectrum(src["signal"], f.tolist(), float(self.dt), int(self.stats["steps"]), bool(src["analytic"]))
        return f, np.asarray(re) + 1j * np.asarray(im)

    def __repr__(self) -> str:
        s = self.stats
        return f"<lightsim.Results {s['cells']} cells x {s['steps']} steps on {s['backend']}: {s['mcells_per_s']:.0f} Mcells/s>"
