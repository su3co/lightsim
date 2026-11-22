"""Electro-optic helpers: free-carrier plasma dispersion in silicon.

Turns a carrier-density map (e.g. exported from a device/TCAD simulation of a
pn or pin modulator) into a permittivity perturbation that the FDTD engine and
the mode solver see through :func:`lightsim.spec.IndexGrid`.

Plasma-dispersion model
-----------------------
Free electrons (``dN``) and holes (``dP``), in cm^-3, change the refractive
index and the absorption coefficient of crystalline silicon as

    dn     = -(A_e * dN**p_e + A_h * dP**p_h)
    dalpha =   B_e * dN**q_e + B_h * dP**q_h          [cm^-1]

``model="soref1987"`` (default) - R. A. Soref and B. R. Bennett, "Electrooptical
effects in silicon", IEEE J. Quantum Electron. 23, 123-129 (1987), in the
widely used fitted form (e.g. G. T. Reed et al., Nature Photon. 4, 518 (2010)):

    ==========  ========  ====  ========  ====  ========  ===  ========  ===
    wavelength  A_e       p_e   A_h       p_h   B_e       q_e  B_h       q_h
    ==========  ========  ====  ========  ====  ========  ===  ========  ===
    1550 nm     8.8e-22   1     8.5e-18   0.8   8.5e-18   1    6.0e-18   1
    1310 nm     6.2e-22   1     6.0e-18   0.8   6.0e-18   1    4.0e-18   1
    ==========  ========  ====  ========  ====  ========  ===  ========  ===

``model="nedeljkovic2011"`` - M. Nedeljkovic, R. Soref and G. Z. Mashanovich,
"Free-carrier electrorefraction and electroabsorption modulation predictions
for silicon over the 1-14 um infrared wavelength range", IEEE Photon. J. 3,
1171-1180 (2011), the updated fit:

    ==========  ========  =====  ========  =====  ========  =====  ========  =====
    wavelength  A_e       p_e    A_h       p_h    B_e       q_e    B_h       q_h
    ==========  ========  =====  ========  =====  ========  =====  ========  =====
    1550 nm     5.40e-22  1.011  1.53e-18  0.838  8.88e-21  1.167  5.84e-20  1.109
    1310 nm     2.98e-22  1.016  1.25e-18  0.835  3.48e-22  1.229  1.02e-19  1.089
    ==========  ========  =====  ========  =====  ========  =====  ========  =====

Reference values (soref1987, 1550 nm, 1e18 cm^-3): electrons dn = -8.8e-4,
dalpha = 8.5 cm^-1; holes dn = -2.1e-3, dalpha = 6.0 cm^-1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .spec import C0, IndexGrid, Medium

EPS0 = 8.8541878128e-12

#: (A_e, p_e, A_h, p_h, B_e, q_e, B_h, q_h) per model and wavelength [nm];
#: densities in cm^-3, absorption in cm^-1.
COEFFICIENTS = {
    "soref1987": {
        1550: (8.8e-22, 1.0, 8.5e-18, 0.8, 8.5e-18, 1.0, 6.0e-18, 1.0),
        1310: (6.2e-22, 1.0, 6.0e-18, 0.8, 6.0e-18, 1.0, 4.0e-18, 1.0),
    },
    "nedeljkovic2011": {
        1550: (5.40e-22, 1.011, 1.53e-18, 0.838, 8.88e-21, 1.167, 5.84e-20, 1.109),
        1310: (2.98e-22, 1.016, 1.25e-18, 0.835, 3.48e-22, 1.229, 1.02e-19, 1.089),
    },
}

#: Refractive index of undoped crystalline silicon at room temperature.
N_SILICON = {1550: 3.476, 1310: 3.503}


def _band(wavelength: float) -> int:
    """Nearest tabulated wavelength (1310 or 1550 nm); ``wavelength`` in metres."""
    nm = float(wavelength) * 1e9
    for w in (1310, 1550):
        if abs(nm - w) <= 30.0:
            return w
    raise ValueError(f"plasma-dispersion coefficients are tabulated at 1310 and 1550 nm only, got {nm:.1f} nm")


def soref_bennett(dN, dP, wavelength: float = 1550e-9, model: str = "soref1987"):
    """Carrier-induced index and absorption change of silicon.

    ``dN`` / ``dP``: free electron / hole densities in **cm^-3** (scalars or
    arrays, >= 0). ``wavelength`` in metres, within 30 nm of 1310 or 1550 nm.
    Returns ``(dn, dalpha)``: the (negative) index change and the absorption
    change in **cm^-1** (power attenuation coefficient).
    """
    if model not in COEFFICIENTS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(COEFFICIENTS)}")
    ae, pe, ah, ph, be, qe, bh, qh = COEFFICIENTS[model][_band(wavelength)]
    dN = np.asarray(dN, dtype=float)
    dP = np.asarray(dP, dtype=float)
    if np.any(dN < 0) or np.any(dP < 0):
        raise ValueError("carrier densities must be >= 0 (cm^-3)")
    dn = -(ae * dN**pe + ah * dP**ph)
    dalpha = be * dN**qe + bh * dP**qh
    if dn.ndim == 0:
        return float(dn), float(dalpha)
    return dn, dalpha


def alpha_to_sigma(alpha_per_m, n):
    """Conductivity (S/m) giving a power absorption ``alpha`` (1/m) in a medium
    of index ``n``: sigma = n * alpha * eps0 * c (valid for kappa << n)."""
    return np.asarray(n, dtype=float) * np.asarray(alpha_per_m, dtype=float) * EPS0 * C0


def carrier_index_grid(
    min: Sequence[float],
    max: Sequence[float],
    dN,
    dP,
    *,
    wavelength: float = 1550e-9,
    n_base=None,
    axes: Sequence[int] | None = None,
    model: str = "soref1987",
    mask=None,
) -> dict:
    """An :func:`~lightsim.spec.IndexGrid` holding ``(n_base + dn)**2``.

    ``dN`` / ``dP`` are 1-D, 2-D or 3-D carrier maps in cm^-3 on a regular grid
    whose nodes span ``[min, max]`` (see ``IndexGrid`` for ``axes``; a (y, z)
    cross-section of a waveguide along x is ``axes=(1, 2)``). ``n_base`` is
    the unperturbed index: a scalar (default: silicon at ``wavelength``) or an
    array of the same shape, e.g. the silicon/oxide index map of the
    cross-section. ``mask`` (boolean, same shape) restricts the perturbation
    to the silicon; elsewhere the index stays ``n_base``.

    Only the real index change is carried by the grid. For the free-carrier
    loss give the structure a conductive medium, see :func:`carrier_medium`.
    """
    dN, dP = np.broadcast_arrays(np.asarray(dN, dtype=float), np.asarray(dP, dtype=float))
    dn, _ = soref_bennett(dN, dP, wavelength, model)
    dn = np.asarray(dn, dtype=float)
    if mask is not None:
        dn = np.where(np.asarray(mask, dtype=bool), dn, 0.0)
    if n_base is None:
        n_base = N_SILICON[_band(wavelength)]
    n = np.asarray(n_base, dtype=float) + dn
    return IndexGrid(min, max, n=n, axes=axes)


def carrier_medium(name: str, dN, dP, *, wavelength: float = 1550e-9, n_base=None, model: str = "soref1987", weights=None) -> dict:
    """Medium to pair with :func:`carrier_index_grid`.

    An ``IndexGrid`` takes one (spatially uniform) conductivity from its
    medium, so the free-carrier absorption is represented by its average over
    the map (optionally weighted, e.g. by the optical mode intensity
    ``weights``): sigma = n * <dalpha> * eps0 * c. The medium's ``eps`` is
    ``n_base**2`` (used by the auto-mesher only; the grid replaces it).
    """
    _, dalpha = soref_bennett(dN, dP, wavelength, model)
    if n_base is None:
        n_base = N_SILICON[_band(wavelength)]
    n0 = float(np.max(n_base))
    mean_alpha = float(np.average(np.asarray(dalpha, dtype=float), weights=weights))  # cm^-1
    sigma = float(alpha_to_sigma(100.0 * mean_alpha, n0))
    return Medium(name, n=n0, sigma=sigma)
