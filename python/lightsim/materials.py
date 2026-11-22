"""Material library and dispersion fitting.

Two things live here:

* a small library of published dispersion formulas for common photonic
  materials (:func:`refractive_index`), and
* a constrained multi-pole fit (:func:`fit_poles`, :func:`fitted_medium`) that
  turns sampled ``n``/``k`` data into the Lorentz / Drude ADE poles the solver
  time-steps, i.e. a ready :func:`lightsim.spec.Medium`.

Conventions follow the core: SI units (wavelengths in metres, angular
frequencies in rad/s), fields ~ ``exp(-i w t)`` so loss is ``Im(eps) > 0`` and
``k > 0``; a Lorentz pole is ``chi = delta_eps w0^2 / (w0^2 - w^2 - i gamma w)``
and a Drude pole ``chi = -wp^2 / (w^2 + i gamma w)``.

Library (coefficients as tabulated by refractiveindex.info, wavelengths in um
inside the formulas):

=============  ==========================================  ===============
name           source                                      valid range (um)
=============  ==========================================  ===============
``SiO2``       Malitson 1965 (fused silica, 20 C)          0.21 - 6.7
``Si3N4``      Luke et al. 2015 (LPCVD film)               0.31 - 5.504
``Si``         Li 1980 (293 K)                             1.2 - 14
``LiNbO3_o``   Zelmon, Small & Jundt 1997 (ordinary)       0.4 - 5.0
``LiNbO3_e``   Zelmon, Small & Jundt 1997 (extraordinary)  0.4 - 5.0
``Al2O3_o``    Malitson & Dodge 1972 (ordinary)            0.2 - 5.0
``Al2O3_e``    Malitson & Dodge 1972 (extraordinary)       0.2 - 5.0
``GaAs``       Skauli et al. 2003 (22 C)                   0.97 - 17
``Ge``         Burnett et al. 2016 (22 C)                  2.0 - 14
=============  ==========================================  ===============

All of these are transparent-window Sellmeier-type formulas, so the returned
index is real (``k = 0``); the return type is complex for uniformity with
user-supplied absorbing data.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Sequence

import numpy as np

from .spec import C0, Drude, Lorentz, Medium

__all__ = [
    "refractive_index",
    "permittivity",
    "available_materials",
    "material_info",
    "fit_poles",
    "fitted_medium",
    "pole_permittivity",
    "gyrotropic_permittivity",
]

# ----------------------------------------------------------------------------
# dispersion formulas (wavelength in micrometres -> n^2)
# ----------------------------------------------------------------------------


def _sellmeier(lam, a0, terms):
    """n^2 = 1 + a0 + sum_i B_i lam^2 / (lam^2 - C_i^2), terms = [(B_i, C_i)]."""
    l2 = lam * lam
    out = 1.0 + a0 + 0.0 * l2
    for b, c in terms:
        out = out + b * l2 / (l2 - c * c)
    return out


def _sellmeier2(lam, a0, terms):
    """n^2 = 1 + a0 + sum_i B_i lam^2 / (lam^2 - C_i), terms = [(B_i, C_i)]."""
    l2 = lam * lam
    out = 1.0 + a0 + 0.0 * l2
    for b, c in terms:
        out = out + b * l2 / (l2 - c)
    return out


def _sio2(lam):
    """Fused silica, 20 C.

    I. H. Malitson, "Interspecimen comparison of the refractive index of fused
    silica", J. Opt. Soc. Am. 55, 1205-1208 (1965). Three-term Sellmeier,
    measured 0.21-3.71 um (verified to 6.7 um by C. Z. Tan, J. Non-Cryst.
    Solids 223, 158 (1998)).
    """
    return _sellmeier(lam, 0.0, [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)])


def _si3n4(lam):
    """Stoichiometric LPCVD silicon nitride film.

    K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, M. Lipson, "Broadband
    mid-infrared frequency comb generation in a Si3N4 microresonator", Opt.
    Lett. 40, 4823-4826 (2015). Two-term Sellmeier, 0.31-5.504 um.
    """
    return _sellmeier(lam, 0.0, [(3.0249, 0.1353406), (40314.0, 1239.842)])


def _si(lam):
    """Crystalline silicon at 293 K.

    H. H. Li, "Refractive index of silicon and germanium and its wavelength
    and temperature derivatives", J. Phys. Chem. Ref. Data 9, 561-658 (1980),
    Eq. (27): n^2 = eps(T) + L(T) / lam^2 * (A0 + A1 T + A2 T^2), valid
    1.2-14 um, evaluated at T = 293 K where the thermal-expansion factor
    L = exp(-3 dL/L) is 1 by definition (dL/L is referenced to 293 K).
    Reproduces Li's recommended table to 1e-4 (n(1.55 um) = 3.4757).
    """
    t = 293.0
    eps = 11.4445 + 2.7739e-4 * t + 1.7050e-6 * t**2 - 8.1347e-10 * t**3
    a = 0.8948 + 4.3977e-4 * t + 7.3835e-8 * t**2
    return eps + a / (lam * lam)


def _linbo3_o(lam):
    """Congruent lithium niobate, ordinary ray, 21 C.

    D. E. Zelmon, D. L. Small, D. Jundt, "Infrared corrected Sellmeier
    coefficients for congruently grown lithium niobate and 5 mol.% magnesium
    oxide-doped lithium niobate", J. Opt. Soc. Am. B 14, 3319-3322 (1997).
    0.4-5.0 um.
    """
    return _sellmeier2(lam, 0.0, [(2.6734, 0.01764), (1.2290, 0.05914), (12.614, 474.60)])


def _linbo3_e(lam):
    """Congruent lithium niobate, extraordinary ray, 21 C (Zelmon et al. 1997,
    see :func:`_linbo3_o`). 0.4-5.0 um."""
    return _sellmeier2(lam, 0.0, [(2.9804, 0.02047), (0.5981, 0.0666), (8.9543, 416.08)])


def _al2o3_o(lam):
    """Synthetic sapphire, ordinary ray, 20 C.

    I. H. Malitson and M. J. Dodge, "Refractive index and birefringence of
    synthetic sapphire", J. Opt. Soc. Am. 62, 1405 (1972); formula from
    M. J. Dodge, "Refractive index", Handbook of Laser Science and Technology
    Vol. IV, CRC Press (1986), p. 30. 0.2-5.0 um.
    """
    return _sellmeier(lam, 0.0, [(1.4313493, 0.0726631), (0.65054713, 0.1193242), (5.3414021, 18.028251)])


def _al2o3_e(lam):
    """Synthetic sapphire, extraordinary ray, 20 C (Malitson & Dodge 1972, see
    :func:`_al2o3_o`). 0.2-5.0 um."""
    return _sellmeier(lam, 0.0, [(1.5039759, 0.0740288), (0.55069141, 0.1216529), (6.5927379, 20.072248)])


def _gaas(lam):
    """Gallium arsenide, 22 C.

    T. Skauli, P. S. Kuo, K. L. Vodopyanov, T. J. Pinguet, O. Levi, L. A.
    Eyres, J. S. Harris, M. M. Fejer, B. Gerard, L. Becouarn, E. Lallier,
    "Improved dispersion relations for GaAs and applications to nonlinear
    optics", J. Appl. Phys. 94, 6447-6455 (2003). 0.97-17 um.
    """
    return _sellmeier(lam, 4.372514, [(5.466742, 0.4431307), (0.02429960, 0.8746453), (1.957522, 36.9166)])


def _ge(lam):
    """Crystalline germanium, 22 C.

    J. H. Burnett, S. G. Kaplan, E. Stover, A. Phenis, "Refractive index
    measurements of Ge", Proc. SPIE 9974, 99740X (2016). 2-14 um (Ge absorbs
    below ~1.8 um, so there is no telecom-band entry).
    """
    return _sellmeier2(lam, 0.0, [(0.4886331, 1.393959), (14.5142535, 0.1626427), (0.0091224, 752.190)])


# name -> (eps(lam_um), (lam_min_um, lam_max_um), citation)
_LIBRARY: dict[str, tuple[Callable, tuple[float, float], str]] = {
    "SiO2": (_sio2, (0.21, 6.7), "Malitson, J. Opt. Soc. Am. 55, 1205 (1965)"),
    "Si3N4": (_si3n4, (0.31, 5.504), "Luke et al., Opt. Lett. 40, 4823 (2015)"),
    "Si": (_si, (1.2, 14.0), "Li, J. Phys. Chem. Ref. Data 9, 561 (1980), 293 K"),
    "LiNbO3_o": (_linbo3_o, (0.4, 5.0), "Zelmon, Small & Jundt, J. Opt. Soc. Am. B 14, 3319 (1997), ordinary"),
    "LiNbO3_e": (_linbo3_e, (0.4, 5.0), "Zelmon, Small & Jundt, J. Opt. Soc. Am. B 14, 3319 (1997), extraordinary"),
    "Al2O3_o": (_al2o3_o, (0.2, 5.0), "Malitson & Dodge, J. Opt. Soc. Am. 62, 1405 (1972), ordinary"),
    "Al2O3_e": (_al2o3_e, (0.2, 5.0), "Malitson & Dodge, J. Opt. Soc. Am. 62, 1405 (1972), extraordinary"),
    "GaAs": (_gaas, (0.97, 17.0), "Skauli et al., J. Appl. Phys. 94, 6447 (2003)"),
    "Ge": (_ge, (2.0, 14.0), "Burnett et al., Proc. SPIE 9974, 99740X (2016)"),
}

_ALIASES = {
    "sio2": "SiO2",
    "silica": "SiO2",
    "fused_silica": "SiO2",
    "si3n4": "Si3N4",
    "sin": "Si3N4",
    "nitride": "Si3N4",
    "si": "Si",
    "silicon": "Si",
    "linbo3": "LiNbO3_o",
    "linbo3_o": "LiNbO3_o",
    "linbo3_e": "LiNbO3_e",
    "al2o3": "Al2O3_o",
    "sapphire": "Al2O3_o",
    "al2o3_o": "Al2O3_o",
    "al2o3_e": "Al2O3_e",
    "gaas": "GaAs",
    "ge": "Ge",
    "germanium": "Ge",
}


def _resolve(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIASES:
        raise KeyError(f"unknown material {name!r}; available: {', '.join(available_materials())}")
    return _ALIASES[key]


def available_materials() -> list[str]:
    """Canonical names of the library materials."""
    return list(_LIBRARY)


def material_info(name: str) -> dict:
    """``{"name", "range": (wl_min_m, wl_max_m), "reference"}`` of a library entry."""
    key = _resolve(name)
    _, (lo, hi), ref = _LIBRARY[key]
    return {"name": key, "range": (lo * 1e-6, hi * 1e-6), "reference": ref}


def permittivity(name: str, wavelength_m, strict: bool = False):
    """Relative permittivity ``eps = (n + i k)^2`` of a library material.

    Same arguments as :func:`refractive_index`.
    """
    key = _resolve(name)
    fn, (lo, hi), _ = _LIBRARY[key]
    lam = np.asarray(wavelength_m, dtype=float) * 1e6
    if np.any(lam < lo * (1 - 1e-12)) or np.any(lam > hi * (1 + 1e-12)):
        msg = f"{key}: wavelength outside the formula's validity range {lo}-{hi} um"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
    eps = np.asarray(fn(lam), dtype=complex)
    return complex(eps) if eps.ndim == 0 else eps


def refractive_index(name: str, wavelength_m, strict: bool = False):
    """Complex refractive index ``n + i k`` of a library material.

    ``name`` is case-insensitive (``"SiO2"``, ``"Si3N4"``, ``"Si"``,
    ``"LiNbO3_o"``, ``"LiNbO3_e"``, ``"Al2O3_o"``, ``"Al2O3_e"``, ``"GaAs"``,
    ``"Ge"``; see the module docstring for the sources). ``wavelength_m`` is
    the vacuum wavelength in metres, scalar (returns ``complex``) or array
    (returns a complex ``ndarray``). Outside the published validity range a
    warning is issued (``strict=True`` raises instead).
    """
    eps = permittivity(name, wavelength_m, strict=strict)
    n = np.sqrt(np.asarray(eps, dtype=complex))
    n = np.where(n.imag < 0, -n, n)
    return complex(n) if n.ndim == 0 else n


# ----------------------------------------------------------------------------
# pole model
# ----------------------------------------------------------------------------


def pole_permittivity(medium_or_fit: dict, omega, dt: float | None = None, axis: int = 0):
    """Permittivity of a pole model at angular frequency ``omega`` (rad/s).

    ``medium_or_fit`` is a :func:`lightsim.spec.Medium` dict or the dict
    returned by :func:`fit_poles`. With ``dt`` given, the *discrete-time*
    permittivity seen by the ADE update with time step ``dt`` is returned
    instead (same expression as ``Pole::chi_discrete`` in the core); use it to
    compare FDTD results with the model at coarse time steps.
    """
    med = medium_or_fit.get("medium", medium_or_fit)
    w = np.asarray(omega, dtype=float)
    eps = np.full(w.shape, complex(med["eps"][axis]))
    if dt is None:
        w2 = w * w
        gw = w
    else:
        # central differences: d2/dt2 -> -(2/dt)^2 sin^2(w dt/2), d/dt -> -i sin(w dt)/dt
        w2 = (2.0 / dt * np.sin(0.5 * w * dt)) ** 2
        gw = np.sin(w * dt) / dt
    for p in med["poles"]:
        if p["type"] == "lorentz":
            w0 = p["omega0"]
            eps = eps + p["delta_eps"] * w0 * w0 / (w0 * w0 - w2 - 1j * p["gamma"] * gw)
        elif p["type"] == "drude":
            eps = eps - p["omega_p"] ** 2 / (w2 + 1j * p["gamma"] * gw)
        elif p["type"] == "debye":
            if dt is None:
                eps = eps + p["delta_eps"] / (1.0 - 1j * w * p["tau"])
            else:
                r = p["tau"] / dt
                z = np.exp(-1j * w * dt)
                eps = eps + (p["delta_eps"] / (r + 0.5)) / (z - (r - 0.5) / (r + 0.5))
        else:
            raise ValueError(f"unknown pole type {p['type']!r}")
    sigma = med.get("sigma", [0.0, 0.0, 0.0])[axis]
    if sigma:
        eps = eps + 1j * sigma / (8.8541878128e-12 * w)
    return complex(eps) if eps.ndim == 0 else eps


def gyrotropic_permittivity(medium: dict, omega: float, dt: float | None = None):
    """3 x 3 complex permittivity tensor of a medium with gyrotropic poles
    (``Medium(gyro=...)``) at angular frequency ``omega`` (rad/s), fields as
    exp(-i w t).

    Each pole adds chi = s (D I + i w [b x])^-1 with D = omega0^2 - w^2 -
    i gamma w, s = delta_eps omega0^2 (Lorentz) or omega_p^2 (Drude, omega0 =
    0) and [b x] the matrix of v -> b x v. For a bias along z, chi_xy =
    -chi_yx = i s w b / (D^2 - w^2 b^2), and the circular polarizations
    (1, +-1j, 0) see eps_inf + s / (D +- w b). With ``dt`` given, the
    tensor of the discrete update with time step ``dt`` is returned instead
    (as ``GyroPole::chi_discrete`` in the core). Ordinary ``poles`` and
    ``sigma`` of the medium are added on the diagonal.
    """
    w = float(omega)
    eps = np.diag([pole_permittivity(medium, w, dt, axis=a) for a in range(3)]).astype(complex)
    for g in medium.get("gyro", []):
        p, b = g["pole"], np.asarray(g["bias"], dtype=float)
        if p["type"] == "lorentz":
            s, w0sq, gam = p["delta_eps"] * p["omega0"] ** 2, p["omega0"] ** 2, p["gamma"]
        elif p["type"] == "drude":
            s, w0sq, gam = p["omega_p"] ** 2, 0.0, p["gamma"]
        else:
            raise ValueError("a gyrotropic pole must be a Lorentz or a Drude pole")
        W = np.array([[0.0, -b[2], b[1]], [b[2], 0.0, -b[0]], [-b[1], b[0], 0.0]])
        I = np.eye(3)
        if dt is None:
            chi = s * np.linalg.inv((w0sq - w * w - 1j * gam * w) * I + 1j * w * W)
        else:
            a, beta = 1.0 / dt**2 + gam / (2 * dt), 1.0 / (2 * dt)
            Mi = np.linalg.inv(a * I - beta * W)
            A = Mi * (2.0 / dt**2 - w0sq)
            B = Mi @ (-(1.0 / dt**2 - gam / (2 * dt)) * I - beta * W)
            C = Mi * s
            z = np.exp(-1j * w * dt)
            chi = np.linalg.inv(z * I - A - B / z) @ C
        eps = eps + chi
    return eps


def _model(x, ws, n_lorentz, drude, eps_inf_fixed, lossless):
    """Model permittivity in scaled units (ws = omega / omega_scale)."""
    i = 0
    if eps_inf_fixed is None:
        eps = np.full(ws.shape, complex(x[0]))
        i = 1
    else:
        eps = np.full(ws.shape, complex(eps_inf_fixed))
    w2 = ws * ws
    for _ in range(n_lorentz):
        de, w0 = x[i], x[i + 1]
        g = 0.0 if lossless else x[i + 2]
        i += 2 if lossless else 3
        eps = eps + de * w0 * w0 / (w0 * w0 - w2 - 1j * g * ws)
    if drude:
        wp = x[i]
        g = 0.0 if lossless else x[i + 1]
        eps = eps - wp * wp / (w2 + 1j * g * ws)
    return eps


def fit_poles(
    wavelengths_m: Sequence[float],
    n: Sequence[float],
    k: Sequence[float] | None = None,
    n_lorentz: int = 2,
    drude: bool = False,
    eps_inf: float | None = None,
    *,
    name: str = "fitted",
    weights: Sequence[float] | None = None,
    omega0_max: float | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> dict:
    """Fit ``eps(w) = eps_inf + sum Lorentz (+ Drude)`` to sampled ``n`` (and ``k``).

    The fit is a bounded nonlinear least squares on the complex permittivity
    error relative to ``|eps|`` (times optional per-sample ``weights``), with
    several deterministic starting points. The bounds ``delta_eps >= 0``,
    ``gamma >= 0``, ``omega0 > 0``, ``omega_p >= 0`` and ``eps_inf >= 1`` make
    every pole passive (``Im eps >= 0`` at all frequencies). The explicit ADE
    update of strong poles on a small ``eps_inf`` is stable only below the
    Courant step of the mesh; the solver reduces the time step when a medium
    needs it.

    Parameters
    ----------
    wavelengths_m, n, k
        Sample vacuum wavelengths (m), refractive index and extinction
        coefficient. ``n`` may be complex (then ``k`` must be ``None``). With no
        absorption data at all (``k`` is ``None`` and ``n`` real) the poles are
        fitted lossless (``gamma = 0`` exactly).
    n_lorentz, drude
        Number of Lorentz poles and whether to add one Drude pole.
    eps_inf
        Fix the high-frequency permittivity; ``None`` fits it (>= 1). For
        transparent-window data the best fit can end at 1 with one strong
        ultraviolet pole (silicon); a fixed larger value, up to about half
        the in-band permittivity, avoids the time-step reduction that such a
        medium needs, for a slightly larger fit error.
    name
        Name of the returned medium.
    omega0_max
        Upper bound on the Lorentz resonance frequencies (rad/s); default 6x
        the highest sampled frequency. A pole far above the band forces a
        small time step, so the default keeps the fit usable down to about
        10 cells per shortest vacuum wavelength of the band; raise it for a
        closer fit on fine grids, lower it for coarser ones.
    n_starts, seed
        Number of starting points (the first is structured, the rest are
        drawn from a generator seeded with ``seed``; the fit is deterministic).

    Returns
    -------
    dict with keys ``eps_inf``, ``lorentz`` (list of ``{"delta_eps", "omega0",
    "gamma"}``), ``drude`` (``{"omega_p", "gamma"}`` or ``None``), ``poles``
    (spec pole dicts), ``medium`` (a ready :func:`lightsim.spec.Medium`),
    ``max_n_error`` / ``max_k_error`` (max ``|n - n_fit|``, ``|k - k_fit|`` over
    the samples), ``rms_eps_error`` (relative) and ``band`` (wl_min, wl_max).
    """
    from scipy.optimize import least_squares

    lam = np.asarray(wavelengths_m, dtype=float).ravel()
    nk = np.asarray(n).ravel()
    if lam.size != nk.size:
        raise ValueError("wavelengths_m and n must have the same length")
    if lam.size < 2:
        raise ValueError("need at least two samples")
    if np.iscomplexobj(nk):
        if k is not None:
            raise ValueError("pass either complex n or separate k, not both")
        lossless = False
        nk = nk.astype(complex)
    elif k is None:
        lossless = True
        nk = nk.astype(complex)
    else:
        kk = np.asarray(k, dtype=float).ravel()
        if kk.size != lam.size:
            raise ValueError("k must have the same length as n")
        if np.any(kk < 0):
            raise ValueError("k must be >= 0 (exp(-i w t) convention, loss is k > 0)")
        lossless = False
        nk = nk.astype(float) + 1j * kk
    if n_lorentz < 0 or (n_lorentz == 0 and not drude):
        raise ValueError("need at least one pole")
    if eps_inf is not None and eps_inf < 1.0:
        raise ValueError("eps_inf must be >= 1")

    eps = nk * nk
    omega = 2.0 * math.pi * C0 / lam
    scale = float(np.sqrt(omega.min() * omega.max()))
    ws = omega / scale
    wgt = 1.0 / np.abs(eps)
    if weights is not None:
        wgt = wgt * np.asarray(weights, dtype=float).ravel()
    w0_hi = (omega0_max / scale) if omega0_max is not None else 6.0 * ws.max()
    w0_lo = 1e-2 * ws.min()
    npp = 2 if lossless else 3

    lo, hi = [], []
    if eps_inf is None:
        lo.append(1.0)
        hi.append(np.inf)
    for _ in range(n_lorentz):
        lo += [0.0, w0_lo] + ([] if lossless else [0.0])
        hi += [np.inf, w0_hi] + ([] if lossless else [np.inf])
    if drude:
        lo += [0.0] + ([] if lossless else [0.0])
        hi += [np.inf] + ([] if lossless else [np.inf])
    lo = np.array(lo)
    hi = np.array(hi)

    def resid(x):
        d = (_model(x, ws, n_lorentz, drude, eps_inf, lossless) - eps) * wgt
        return np.concatenate([d.real, d.imag])

    rng = np.random.default_rng(seed)
    eps_mean = float(np.mean(eps.real))
    best = None
    for s in range(max(1, n_starts)):
        x0 = []
        if eps_inf is None:
            x0.append(1.0 if s % 2 == 0 else max(1.0, 0.5 * eps_mean))
        if s == 0:
            # structured start: resonances stacked above the band (UV), last one below (IR)
            w0s = [ws.max() * 3.0 * 1.6**j for j in range(n_lorentz)]
            if n_lorentz >= 2:
                w0s[-1] = ws.min() * 0.1
        else:
            w0s = list(np.exp(rng.uniform(np.log(ws.min() * 0.05), np.log(min(w0_hi, ws.max() * 10.0)), n_lorentz)))
        for w0 in w0s:
            w0 = float(np.clip(w0, w0_lo * 1.01, w0_hi * 0.99))
            de = max(abs(eps_mean - 1.0), 0.1) / max(n_lorentz, 1) if w0 > 1.0 else 0.1
            x0 += [de, w0] + ([] if lossless else [0.01 * w0 if s % 3 else 0.2 * w0])
        if drude:
            x0 += [float(rng.uniform(0.5, 5.0)) if s else 1.0] + ([] if lossless else [0.05])
        x0 = np.clip(np.array(x0), lo, np.where(np.isfinite(hi), hi, 1e30))
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=600)
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(r.fun)):
            continue
        if best is None or r.cost < best.cost:
            best = r
        if best.cost < 1e-16 * resid(x0).size:
            break
    if best is None:
        raise RuntimeError("pole fit failed for every starting point")
    x = np.clip(best.x, lo, hi)

    i = 0
    if eps_inf is None:
        e_inf = float(x[0])
        i = 1
    else:
        e_inf = float(eps_inf)
    lorentz = []
    for _ in range(n_lorentz):
        de, w0 = float(x[i]), float(x[i + 1]) * scale
        g = 0.0 if lossless else float(x[i + 2]) * scale
        i += npp
        if de > 0.0:
            lorentz.append({"delta_eps": de, "omega0": w0, "gamma": g})
    lorentz.sort(key=lambda p: p["omega0"])
    dr = None
    if drude:
        wp = float(x[i]) * scale
        g = 0.0 if lossless else float(x[i + 1]) * scale
        if wp > 0.0:
            dr = {"omega_p": wp, "gamma": g}

    poles = [Lorentz(p["delta_eps"], p["omega0"], p["gamma"]) for p in lorentz]
    if dr is not None:
        poles.append(Drude(dr["omega_p"], dr["gamma"]))
    medium = Medium(name, eps=e_inf, poles=poles)
    eps_fit = pole_permittivity(medium, omega)
    n_fit = np.sqrt(eps_fit)
    n_fit = np.where(n_fit.imag < 0, -n_fit, n_fit)
    return {
        "eps_inf": e_inf,
        "lorentz": lorentz,
        "drude": dr,
        "poles": poles,
        "medium": medium,
        "max_n_error": float(np.max(np.abs(n_fit.real - nk.real))),
        "max_k_error": float(np.max(np.abs(n_fit.imag - nk.imag))),
        "rms_eps_error": float(np.sqrt(np.mean(np.abs((eps_fit - eps) / eps) ** 2))),
        "band": (float(lam.min()), float(lam.max())),
    }


def fitted_medium(
    name: str,
    band: tuple[float, float] = (1.2e-6, 1.7e-6),
    n_lorentz: int = 2,
    *,
    drude: bool = False,
    eps_inf: float | None = None,
    samples: int = 101,
    medium_name: str | None = None,
    tol: float | None = 1e-3,
    return_fit: bool = False,
    **fit_kwargs,
):
    """Library data -> pole fit -> ``Medium`` for the wavelength ``band`` (m).

    Samples the library formula for ``name`` at ``samples`` points uniform in
    frequency over ``band = (wl_min, wl_max)``, runs :func:`fit_poles` and
    returns the :func:`lightsim.spec.Medium` dict (named ``medium_name``,
    default the library name). A warning is issued if the fit misses the data
    by more than ``tol`` in ``n`` (``None`` disables the check). With
    ``return_fit=True`` the full :func:`fit_poles` dict is returned instead.

    Uniaxial crystals are fitted one ray at a time (``"LiNbO3_o"``,
    ``"LiNbO3_e"``, ...); the result is an isotropic medium for that ray.
    """
    key = _resolve(name)
    wl_min, wl_max = sorted(float(b) for b in band)
    if not wl_max > wl_min > 0:
        raise ValueError("band must be (wl_min, wl_max) with 0 < wl_min < wl_max")
    lam = C0 / np.linspace(C0 / wl_max, C0 / wl_min, int(samples))
    nk = refractive_index(key, lam)
    lossy = bool(np.any(nk.imag > 0))
    fit = fit_poles(lam, nk.real, nk.imag if lossy else None, n_lorentz=n_lorentz, drude=drude, eps_inf=eps_inf, name=medium_name or key, **fit_kwargs)
    if tol is not None and fit["max_n_error"] > tol:
        warnings.warn(f"{key}: pole fit error max|n - n_fit| = {fit['max_n_error']:.2e} exceeds {tol:.1e}; try more poles", stacklevel=2)
    return fit if return_fit else fit["medium"]


# ----------------------------------------------------------------------------
# thermal, thermo-optic and electro-optic properties (used by lightsim.fem)
# ----------------------------------------------------------------------------

#: Room-temperature (about 300 K) thermal and thermo-optic properties of the
#: library materials: thermal conductivity ``k`` (W/(m K)), specific heat
#: ``cp`` (J/(kg K)), density ``rho`` (kg/m^3) and thermo-optic coefficient
#: ``dndT`` (1/K, near 1550 nm unless noted; ``None`` where no value is
#: given). Thin films can differ strongly from bulk values (deposited oxide
#: and nitride films conduct heat less well than the bulk solids); pass your
#: own values to the solvers when you know them.
THERMAL_PROPERTIES: dict[str, dict] = {
    "Si": {
        "k": 148.0, "cp": 712.0, "rho": 2329.0, "dndT": 1.86e-4,
        "reference": "k: Glassbrenner & Slack, Phys. Rev. 134, A1058 (1964); cp, rho: CRC Handbook of Chemistry and Physics; "
        "dn/dT at 1.5 um: Cocorullo & Rendina, Electron. Lett. 28, 83 (1992)",
    },
    "SiO2": {
        "k": 1.38, "cp": 754.0, "rho": 2203.0, "dndT": 0.95e-5,
        "reference": "fused silica; k: Touloukian et al., Thermophysical Properties of Matter vol. 2 (1970), 300 K; "
        "cp (45.3 J/(mol K)), rho: CRC Handbook of Chemistry and Physics; dn/dT at 1550 nm (SiOx film): "
        "Arbabi & Goddard, Opt. Lett. 38, 3878 (2013)",
    },
    "Si3N4": {
        "k": 2.1, "cp": 700.0, "rho": 3100.0, "dndT": 2.45e-5,
        "reference": "thin LPCVD film; k: the value gplugins/femwell use for SiN films (bulk ceramic: about 30); cp, rho: "
        "typical film values (bulk 3170-3290 kg/m^3); dn/dT at 1550 nm: Arbabi & Goddard, Opt. Lett. 38, 3878 (2013)",
    },
    "LiNbO3": {
        "k": 4.6, "cp": 628.0, "rho": 4648.0, "dndT": None,
        "reference": "congruent LiNbO3; rho: Weis & Gaylord, Appl. Phys. A 37, 191 (1985); k, cp: typical crystal-vendor "
        "values; dn/dT not given (strongly anisotropic, see Moretti et al., J. Appl. Phys. 98, 036101 (2005))",
    },
    "Al2O3": {
        "k": 35.0, "cp": 763.0, "rho": 3980.0, "dndT": None,
        "reference": "sapphire, 300 K, perpendicular to c; Touloukian et al., Thermophysical Properties of Matter (1970); "
        "rho, cp: CRC Handbook of Chemistry and Physics",
    },
    "GaAs": {
        "k": 55.0, "cp": 330.0, "rho": 5320.0, "dndT": 2.35e-4,
        "reference": "k, cp, rho: Ioffe Institute semiconductor database (NSM archive), 300 K; dn/dT at 1.5 um: "
        "Della Corte et al., Appl. Phys. Lett. 77, 1614 (2000)",
    },
    "Ge": {
        "k": 58.0, "cp": 310.0, "rho": 5323.0, "dndT": 4.0e-4,
        "reference": "k, cp, rho: Ioffe Institute semiconductor database (NSM archive), 300 K; dn/dT (mid-infrared, "
        "3-12 um): Icenogle, Platt & Wolfe, Appl. Opt. 15, 2348 (1976)",
    },
}
THERMAL_PROPERTIES["LiNbO3_o"] = THERMAL_PROPERTIES["LiNbO3"]
THERMAL_PROPERTIES["LiNbO3_e"] = THERMAL_PROPERTIES["LiNbO3"]
THERMAL_PROPERTIES["Al2O3_o"] = THERMAL_PROPERTIES["Al2O3"]
THERMAL_PROPERTIES["Al2O3_e"] = THERMAL_PROPERTIES["Al2O3"]

#: Linear electro-optic (Pockels) coefficients in pm/V (contracted notation,
#: crystal axes, z = optic axis) and the low-frequency relative
#: permittivities. Congruent LiNbO3, unclamped (low-frequency) values at
#: 633 nm from R. S. Weis and T. K. Gaylord, "Lithium niobate: summary of
#: physical properties and crystal structure", Appl. Phys. A 37, 191-203
#: (1985): r33 = 30.8, r13 = 8.6, r22 = 3.4, r51 = 28; eps_11 = 85,
#: eps_33 = 28.7 (unclamped, low frequency).
ELECTRO_OPTIC: dict[str, dict] = {
    "LiNbO3": {
        "r13": 8.6e-12, "r22": 3.4e-12, "r33": 30.8e-12, "r51": 28.0e-12,
        "eps_static": (85.0, 85.0, 28.7),
        "reference": "Weis & Gaylord, Appl. Phys. A 37, 191 (1985), unclamped values at 633 nm",
    },
}


def thermal_properties(name: str) -> dict:
    """Thermal and thermo-optic properties of a library material (see
    :data:`THERMAL_PROPERTIES`): ``{"k", "cp", "rho", "dndT", "reference"}``."""
    key = _resolve(name)
    if key not in THERMAL_PROPERTIES:
        raise KeyError(f"no thermal data for {key}")
    return dict(THERMAL_PROPERTIES[key], name=key)


def electro_optic_coefficients(name: str) -> dict:
    """Pockels coefficients (m/V) and static permittivities of a material
    (see :data:`ELECTRO_OPTIC`; LiNbO3 only)."""
    key = _resolve(name)
    key = "LiNbO3" if key.startswith("LiNbO3") else key
    if key not in ELECTRO_OPTIC:
        raise KeyError(f"no electro-optic data for {key}")
    return dict(ELECTRO_OPTIC[key], name=key)


__all__ += ["THERMAL_PROPERTIES", "ELECTRO_OPTIC", "thermal_properties", "electro_optic_coefficients"]
