"""Sheet (two-dimensional) materials given by a surface conductivity.

A conducting sheet such as graphene has a surface conductivity
``sigma_s(w)`` (siemens) and no thickness. There are two ways to put one in
a simulation:

* **Zero-thickness sheets** (the default when no ``thickness`` is given):
  :func:`graphene`, :func:`sheet_medium` and :func:`sheet_conductivity`
  return a surface conductivity, placed with :func:`lightsim.Sheet` and
  :meth:`lightsim.Simulation.add_sheet`. The solver applies the surface
  current on the plane of the tangential E nodes at the sheet, divided by
  the dual cell normal to it; there is nothing to align (see
  ``docs/modules/sheets.md``).
* **One-cell layers** (with ``thickness``): a thin volumetric layer of
  thickness ``t`` whose permittivity carries the same sheet current:

    eps(w) = eps_b + 1j * sigma_s(w) / (eps0 * w * t)

``t`` should be the cell size of the mesh normal to the sheet (or a whole
number of cells), because dispersive media are not sub-pixel smoothed: a
layer thinner than its cell either fills the cell or disappears.

* A Drude sheet ``sigma_s = 1j * D / (w + 1j * gamma)`` maps exactly onto one
  Drude pole with ``omega_p**2 = D / (eps0 * t)``.
* Any other ``sigma_s(w)`` is fitted over a band with
  :func:`lightsim.materials.fit_poles`.
* :func:`graphene` uses the Kubo conductivity: the intraband term (a Drude
  sheet at any temperature) is mapped exactly; the interband term, when a
  band is given, is fitted with Lorentz poles.

Conventions: SI units, fields ~ ``exp(-i w t)``, so a lossy sheet has
``Re sigma_s > 0``; an inductive (Drude-like) one has ``Im sigma_s > 0``.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .spec import C0, Drude, Lorentz, Medium

__all__ = [
    "SheetConductivity",
    "DrudeSheetPole",
    "LorentzSheetPole",
    "sheet_conductivity",
    "sheet",
    "sigma_of",
    "sheet_medium",
    "graphene",
    "graphene_conductivity",
    "drude_sheet_conductivity",
    "sheet_layer_conductivity",
    "sheet_rt",
]

EPS0 = 8.8541878128e-12
ETA0 = 376.730313412
Q_E = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23


def drude_sheet_conductivity(freq, drude_weight: float, gamma: float):
    """Surface conductivity ``1j * D / (w + 1j * gamma)`` (S) of a Drude sheet.

    ``freq`` in Hz, ``drude_weight`` ``D`` in S/s (siemens per second),
    ``gamma`` the collision rate in rad/s.
    """
    w = 2 * math.pi * np.asarray(freq, dtype=float)
    return 1j * drude_weight / (w + 1j * gamma)


def _graphene_drude_weight(fermi_level_eV: float, temperature: float) -> float:
    """Drude weight ``D`` (S/s) of graphene's intraband conductivity."""
    ef = abs(fermi_level_eV) * Q_E
    if temperature <= 0:
        return Q_E**2 * ef / (math.pi * HBAR**2)
    kt = K_B * temperature
    x = ef / (2 * kt)
    # 2 kT ln(2 cosh(Ef / 2kT)), written to avoid overflow
    eff = 2 * kt * (x + math.log1p(math.exp(-2 * x)))
    return Q_E**2 * eff / (math.pi * HBAR**2)


def _h_occ(e, ef, kt):
    """sinh(e/kT) / (cosh(Ef/kT) + cosh(e/kT)) for e >= 0, overflow-safe."""
    e = np.asarray(e, dtype=float)
    if kt <= 0:
        return np.where(e > ef, 1.0, np.where(e == ef, 0.5, 0.0))
    x = e / kt
    m = ef / kt
    with np.errstate(over="ignore"):
        em2x = np.exp(-2 * x)
        return (1 - em2x) / (np.exp(m - x) + np.exp(-m - x) + 1 + em2x)


def graphene_conductivity(freq, fermi_level_eV: float, scattering_time: float, temperature: float = 300.0, interband: bool = True):
    """Kubo surface conductivity of graphene (S), at frequencies ``freq`` (Hz).

    ``sigma = sigma_intra + sigma_inter`` with (Falkovsky, J. Phys. Conf. Ser.
    129, 012004, 2008; Hanson, J. Appl. Phys. 103, 064302, 2008, in the
    ``exp(-i w t)`` convention)

    * ``sigma_intra = 1j * D / (w + 1j / tau)``, ``D = (2 e^2 kT / (pi hbar^2))
      ln(2 cosh(Ef / 2kT))``, which is ``e^2 Ef / (pi hbar^2)`` at T = 0;
    * ``sigma_inter = (e^2 / 4 hbar) [H(w/2) + (4j w / pi) int_0^inf (H(x) -
      H(w/2)) / (w^2 - 4 x^2) dx]``, ``H(x) = sinh(hbar x/kT) / (cosh(Ef/kT)
      + cosh(hbar x/kT))``; at T = 0 this is ``(e^2/4 hbar) [step(hbar w -
      2 Ef) + (1j/pi) ln|(hbar w - 2 Ef) / (hbar w + 2 Ef)|]``.

    The interband term is computed without broadening. ``scattering_time``
    ``tau`` is in s, ``temperature`` in K. Spatial dispersion and magnetic
    fields are not included.
    """
    from scipy.integrate import quad

    f = np.atleast_1d(np.asarray(freq, dtype=float))
    w = 2 * math.pi * f
    sig = drude_sheet_conductivity(f, _graphene_drude_weight(fermi_level_eV, temperature), 1.0 / scattering_time)
    if interband:
        ef = abs(fermi_level_eV) * Q_E
        kt = K_B * max(temperature, 0.0)
        s0 = Q_E**2 / (4 * HBAR)
        out = np.empty(f.shape, dtype=complex)
        for i, wi in enumerate(w):
            e_half = 0.5 * HBAR * wi
            if kt <= 0:
                re = 1.0 if e_half > ef else 0.0
                im = math.log(abs((e_half - ef) / (e_half + ef))) / math.pi if e_half != ef else -np.inf
                out[i] = s0 * (re + 1j * im)
                continue
            h_half = float(_h_occ(e_half, ef, kt))
            scale = max(ef, kt, e_half)

            # integrand in energy e = hbar x: (H(e) - H(w/2)) / ((hbar w)^2 - 4 e^2), times hbar w
            def g(u):
                e = u * scale
                return float(_h_occ(e, ef, kt) - h_half) / ((2 * e_half / scale) ** 2 - 4 * u * u)

            u_half = e_half / scale
            u_far = (ef + e_half + 40 * kt) / scale + 1.0
            parts = [quad(g, 0.0, u_half, limit=400)[0], quad(g, u_half, u_far, limit=400)[0], quad(g, u_far, np.inf, limit=200)[0]]
            integral = sum(parts) / scale  # in 1/energy
            out[i] = s0 * (h_half + 1j * (4 * 2 * e_half / math.pi) * integral)
        sig = sig + out
    return sig if np.ndim(freq) else complex(sig[0])


def DrudeSheetPole(drude_weight: float, gamma: float) -> dict:
    """A Drude term ``sigma = 1j * D / (w + 1j * gamma)`` of a surface
    conductivity: ``drude_weight`` D in S/s, ``gamma`` in rad/s."""
    return {"type": "drude", "drude_weight": float(drude_weight), "gamma": float(gamma)}


def LorentzSheetPole(strength: float, omega0: float, gamma: float) -> dict:
    """A Lorentz term ``sigma = -1j * w * s / (w0**2 - w**2 - 1j * gamma * w)``
    of a surface conductivity: ``strength`` s in S rad/s (the conductance at
    resonance is s / gamma), ``omega0`` and ``gamma`` in rad/s."""
    return {"type": "lorentz", "strength": float(strength), "omega0": float(omega0), "gamma": float(gamma)}


def SheetConductivity(conductance: float = 0.0, susceptibility: float = 0.0, poles: Sequence[dict] = (), name: str = "") -> dict:
    """A surface conductivity for a zero-thickness :func:`lightsim.Sheet`:

    ``sigma_s(w) = conductance - 1j * w * eps0 * susceptibility + sum(poles)``

    in siemens, with ``conductance`` (S) a constant conductance,
    ``susceptibility`` (m) a constant surface susceptibility and ``poles``
    from :func:`DrudeSheetPole` and :func:`LorentzSheetPole`. All terms must
    be passive (non-negative)."""
    return {"name": name, "conductance": float(conductance), "susceptibility": float(susceptibility), "poles": [dict(p) for p in poles]}


def sigma_of(conductivity: dict, freq):
    """``sigma_s`` (S) of a :func:`SheetConductivity` dict at ``freq`` (Hz)."""
    w = 2 * math.pi * np.asarray(freq, dtype=float)
    s = conductivity.get("conductance", 0.0) - 1j * w * EPS0 * conductivity.get("susceptibility", 0.0) + 0j * w
    for p in conductivity.get("poles", []):
        if p["type"] == "drude":
            s = s + 1j * p["drude_weight"] / (w + 1j * p["gamma"])
        else:
            s = s - 1j * w * p["strength"] / (p["omega0"] ** 2 - w**2 - 1j * p["gamma"] * w)
    return s


def _fit_sheet(sigma_vals, freqs, n_lorentz, drude, name, **fit_kwargs):
    """Fit sampled ``sigma_s`` with passive sheet poles: through the layer
    permittivity of a reference thickness chosen so that the susceptibility
    is of order one (fit_poles weighs errors relative to |eps|), then mapped
    back to sheet poles (exact for any thickness)."""
    f = np.asarray(freqs, dtype=float)
    w = 2 * math.pi * f
    s = np.asarray(sigma_vals, dtype=complex)
    t_ref = float(np.max(np.abs(s) / (EPS0 * w))) / 3.0
    eps_t = 1.0 + 1j * s / (EPS0 * w * t_ref)
    fit = _fit_layer(eps_t, f, n_lorentz, drude, 1.0, name, **fit_kwargs)
    poles = [LorentzSheetPole(EPS0 * t_ref * p["delta_eps"] * p["omega0"] ** 2, p["omega0"], p["gamma"]) for p in fit["lorentz"]]
    if fit.get("drude"):
        poles.append(DrudeSheetPole(EPS0 * t_ref * fit["drude"]["omega_p"] ** 2, fit["drude"]["gamma"]))
    cond = SheetConductivity(susceptibility=t_ref * (fit["eps_inf"] - 1.0), poles=poles, name=name)
    return cond, fit


def sheet_conductivity(
    name: str = "sheet",
    *,
    conductance: float = 0.0,
    drude_weight: float | None = None,
    gamma: float = 0.0,
    sigma: Callable | Sequence[complex] | None = None,
    freqs: Sequence[float] | None = None,
    band: tuple[float, float] | None = None,
    n_lorentz: int = 2,
    drude: bool = True,
    samples: int = 81,
    return_fit: bool = False,
    **fit_kwargs,
):
    """A surface conductivity for a zero-thickness :func:`lightsim.Sheet`.

    Give a constant ``conductance`` (S), and/or a Drude sheet
    (``drude_weight`` D in S/s and ``gamma`` in rad/s, ``sigma = 1j D / (w +
    1j gamma)``, exact at every frequency), or ``sigma``: a callable
    ``sigma(freq_Hz)`` with ``band = (f_min, f_max)``, or an array on
    ``freqs``, fitted with ``n_lorentz`` Lorentz terms, (``drude=True``) one
    Drude term and a constant susceptibility; the fit holds in the band only.
    Returns the :func:`SheetConductivity` dict, or with ``return_fit=True`` a
    dict with ``conductivity``, ``fit`` and ``max_sigma_error`` (the largest
    relative error over the samples, 0 for exact terms).
    """
    if sigma is not None and drude_weight is not None:
        raise ValueError("give drude_weight or sigma, not both")
    fit, err = None, 0.0
    if sigma is not None:
        if callable(sigma):
            if band is None:
                raise ValueError("a callable sigma needs band=(f_min, f_max) in Hz")
            f = np.linspace(min(band), max(band), int(samples))
            sv = np.asarray(sigma(f), dtype=complex)
        else:
            if freqs is None:
                raise ValueError("an array sigma needs freqs")
            f = np.asarray(freqs, dtype=float)
            sv = np.asarray(sigma, dtype=complex)
        cond, fit = _fit_sheet(sv - conductance, f, n_lorentz, drude, name, **fit_kwargs)
        cond["conductance"] = float(conductance)
        err = float(np.max(np.abs(sigma_of(cond, f) - sv) / np.abs(sv)))
    else:
        poles = []
        if drude_weight is not None:
            if drude_weight < 0 or gamma < 0:
                raise ValueError("drude_weight and gamma must be >= 0")
            poles.append(DrudeSheetPole(drude_weight, gamma))
        if conductance < 0:
            raise ValueError("conductance must be >= 0")
        cond = SheetConductivity(conductance=conductance, poles=poles, name=name)
    if return_fit:
        return {"conductivity": cond, "fit": fit, "max_sigma_error": err}
    return cond


def sheet(axis, position: float, conductivity: dict, **kwargs) -> dict:
    """:func:`lightsim.Sheet`: a zero-thickness sheet carrying ``conductivity``."""
    from .spec import Sheet

    return Sheet(axis, position, conductivity, **kwargs)


def sheet_layer_conductivity(medium: dict, freq, thickness: float, eps_background: float = 1.0, axis: int = 0):
    """Surface conductivity that a layer of ``medium`` and ``thickness`` carries.

    ``sigma_s = -1j * eps0 * w * t * (eps(w) - eps_background)``: the inverse
    of the mapping used by :func:`sheet_medium`, to check a fit against the
    target conductivity.
    """
    from .materials import pole_permittivity

    w = 2 * math.pi * np.asarray(freq, dtype=float)
    eps = pole_permittivity(medium, w, axis=axis)
    return -1j * EPS0 * w * thickness * (eps - eps_background)


def sheet_rt(freq, sigma_s, n1: float = 1.0, n2: float = 1.0, angle: float = 0.0, polarization: str = "s"):
    """Analytic power reflection and transmission of a sheet.

    The sheet lies between media of indices ``n1`` (incidence side) and
    ``n2``. ``sigma_s`` is a complex array over ``freq``, a callable of
    ``freq`` or a :func:`SheetConductivity` dict. At normal incidence
    ``r = (n1 - n2 - eta0 sigma) / (n1 + n2 + eta0 sigma)``,
    ``t = 2 n1 / (n1 + n2 + eta0 sigma)``, ``T = (n2/n1)|t|^2``; for a
    free-standing sheet ``T = |2 / (2 + eta0 sigma)|^2``. At ``angle`` (rad,
    in medium 1) the indices become the wave admittances ``n cos(theta)``
    (``polarization="s"``) or ``n / cos(theta)`` (``"p"``). Returns ``(R, T)``.
    """
    if isinstance(sigma_s, dict):
        s = sigma_of(sigma_s, freq)
    else:
        s = sigma_s(freq) if callable(sigma_s) else np.asarray(sigma_s)
    c1 = math.cos(angle)
    st2 = n1 * math.sin(angle) / n2
    c2 = np.sqrt(complex(1.0 - st2 * st2))
    if polarization == "s":
        y1, y2 = n1 * c1, n2 * c2
    elif polarization == "p":
        y1, y2 = n1 / c1, n2 / c2
    else:
        raise ValueError("polarization must be 's' or 'p'")
    d = y1 + y2 + ETA0 * s
    return np.abs((y1 - y2 - ETA0 * s) / d) ** 2, (y2.real / y1) * np.abs(2 * y1 / d) ** 2


def _fit_layer(eps_target, freqs, n_lorentz, drude, eps_background, name, **fit_kwargs):
    from .materials import fit_poles

    lam = C0 / np.asarray(freqs, dtype=float)
    n = np.sqrt(np.asarray(eps_target, dtype=complex))
    n = np.where(n.imag < 0, -n, n)
    return fit_poles(lam, n, None, n_lorentz=n_lorentz, drude=drude, name=name, **fit_kwargs)


def sheet_medium(
    name: str,
    thickness: float | None = None,
    *,
    drude_weight: float | None = None,
    gamma: float = 0.0,
    sigma: Callable | Sequence[complex] | None = None,
    freqs: Sequence[float] | None = None,
    band: tuple[float, float] | None = None,
    eps_background: float = 1.0,
    n_lorentz: int = 2,
    drude: bool = True,
    samples: int = 81,
    return_fit: bool = False,
    **fit_kwargs,
):
    """A sheet conductivity: zero-thickness (``thickness=None``, the default)
    or as a medium whose layer of ``thickness`` (m) carries it.

    Without ``thickness`` this is :func:`sheet_conductivity`: the returned
    surface conductivity goes into :func:`lightsim.Sheet`. With a
    ``thickness``, give either

    * ``drude_weight`` (S/s) and ``gamma`` (rad/s): the Drude sheet
      ``sigma_s = 1j * D / (w + 1j * gamma)``, mapped exactly onto one Drude
      pole with ``omega_p**2 = D / (eps0 * thickness)``; or
    * ``sigma``: a callable ``sigma(freq_Hz) -> complex`` (S) with ``band =
      (f_min, f_max)`` in Hz, or an array of values on ``freqs``. The layer
      permittivity ``eps_b + 1j sigma / (eps0 w t)`` is fitted with
      ``n_lorentz`` Lorentz poles and (``drude=True``) one Drude pole by
      :func:`lightsim.materials.fit_poles`, so the result holds in that band
      only, with the fit error reported by ``return_fit=True``.

    ``eps_background`` is the permittivity of the layer without the sheet
    (1 for a free-standing sheet). Place the layer as a box exactly
    ``thickness`` thick, on a mesh whose cell normal to the sheet is
    ``thickness``, centred on the grid nodes of the tangential E components
    (dispersive media are staircased, not smoothed). Returns the
    :func:`lightsim.spec.Medium` dict; with ``return_fit=True`` a dict with the
    medium, the fit (or ``None``) and the largest relative conductivity error
    over the fitted samples (0 for a Drude sheet).
    """
    if (drude_weight is None) == (sigma is None):
        raise ValueError("give either drude_weight (and gamma) or sigma")
    if thickness is None:
        if eps_background != 1.0:
            raise ValueError("eps_background belongs to a one-cell layer; a zero-thickness sheet takes the media around it")
        return sheet_conductivity(name, drude_weight=drude_weight, gamma=gamma, sigma=sigma, freqs=freqs, band=band, n_lorentz=n_lorentz, drude=drude, samples=samples, return_fit=return_fit, **fit_kwargs)
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    fit = None
    err = 0.0
    if drude_weight is not None:
        if drude_weight < 0 or gamma < 0:
            raise ValueError("drude_weight and gamma must be >= 0")
        wp = math.sqrt(drude_weight / (EPS0 * thickness))
        med = Medium(name, eps=eps_background, poles=[Drude(wp, gamma)])
    else:
        if callable(sigma):
            if band is None:
                raise ValueError("a callable sigma needs band=(f_min, f_max) in Hz")
            f = np.linspace(min(band), max(band), int(samples))
            s = np.asarray(sigma(f), dtype=complex)
        else:
            if freqs is None:
                raise ValueError("an array sigma needs freqs")
            f = np.asarray(freqs, dtype=float)
            s = np.asarray(sigma, dtype=complex)
        w = 2 * math.pi * f
        eps_t = eps_background + 1j * s / (EPS0 * w * thickness)
        fit = _fit_layer(eps_t, f, n_lorentz, drude, eps_background, name, **fit_kwargs)
        med = fit["medium"]
        s_fit = sheet_layer_conductivity(med, f, thickness, eps_background)
        err = float(np.max(np.abs(s_fit - s) / np.abs(s)))
    if return_fit:
        return {"medium": med, "fit": fit, "max_sigma_error": err}
    return med


def graphene(
    fermi_level_eV: float,
    scattering_time: float,
    thickness: float | None = None,
    temperature: float = 300.0,
    *,
    band: tuple[float, float] | None = None,
    name: str = "graphene",
    eps_background: float = 1.0,
    n_lorentz: int = 2,
    samples: int = 81,
    return_fit: bool = False,
    **fit_kwargs,
):
    """Graphene from the Kubo conductivity: a zero-thickness surface
    conductivity (``thickness=None``, the default, for :func:`lightsim.Sheet`)
    or a layer of ``thickness`` (m).

    ``fermi_level_eV`` is the chemical potential (eV, its sign does not
    matter), ``scattering_time`` the intraband relaxation time (s),
    ``temperature`` in K. The intraband term is one exact Drude pole, valid at
    every frequency. With ``band = (f_min, f_max)`` (Hz) the interband term of
    :func:`graphene_conductivity` is added as ``n_lorentz`` Lorentz poles
    fitted over that band (and only valid there); without ``band`` it is left
    out, which is accurate well below the interband threshold ``2 Ef / h``
    (compare the two terms with :func:`graphene_conductivity`).

    Returns the :func:`lightsim.spec.Medium` dict, or with ``return_fit=True``
    a dict with the medium, the interband fit and the largest relative error
    of the layer's sheet conductivity against the Kubo formula over the band.
    """
    d = _graphene_drude_weight(fermi_level_eV, temperature)
    gamma = 1.0 / scattering_time
    if thickness is None:
        if eps_background != 1.0:
            raise ValueError("eps_background belongs to a one-cell layer; a zero-thickness sheet takes the media around it")
        cond = SheetConductivity(poles=[DrudeSheetPole(d, gamma)], name=name)
        fit, err = None, None
        if band is not None:
            f = np.linspace(min(band), max(band), int(samples))
            s_inter = graphene_conductivity(f, fermi_level_eV, scattering_time, temperature) - drude_sheet_conductivity(f, d, gamma)
            inter, fit = _fit_sheet(s_inter, f, n_lorentz, False, name, **fit_kwargs)
            cond = SheetConductivity(susceptibility=inter["susceptibility"], poles=inter["poles"] + cond["poles"], name=name)
            s_full = graphene_conductivity(f, fermi_level_eV, scattering_time, temperature)
            err = float(np.max(np.abs(sigma_of(cond, f) - s_full) / np.abs(s_full)))
        if return_fit:
            return {"conductivity": cond, "fit": fit, "max_sigma_error": err}
        return cond
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    wp = math.sqrt(d / (EPS0 * thickness))
    fit = None
    err = None
    if band is None:
        med = Medium(name, eps=eps_background, poles=[Drude(wp, gamma)])
    else:
        f = np.linspace(min(band), max(band), int(samples))
        w = 2 * math.pi * f
        s_inter = graphene_conductivity(f, fermi_level_eV, scattering_time, temperature) - drude_sheet_conductivity(f, d, gamma)
        eps_t = eps_background + 1j * s_inter / (EPS0 * w * thickness)
        fit = _fit_layer(eps_t, f, n_lorentz, False, eps_background, name, **fit_kwargs)
        poles = [Lorentz(p["delta_eps"], p["omega0"], p["gamma"]) for p in fit["lorentz"]] + [Drude(wp, gamma)]
        med = Medium(name, eps=fit["eps_inf"], poles=poles)
    if band is not None:
        s_full = graphene_conductivity(f, fermi_level_eV, scattering_time, temperature)
        s_fit = sheet_layer_conductivity(med, f, thickness, eps_background)
        err = float(np.max(np.abs(s_fit - s_full) / np.abs(s_full)))
    if return_fit:
        return {"medium": med, "fit": fit, "max_sigma_error": err}
    return med
