"""Mode-solver sweeps and derived waveguide quantities (SI units).

Every sweep returns a tidy table, one row per (parameter value, mode): a
``pandas.DataFrame`` when pandas is installed, otherwise a dict of lists
with the same columns. The cross-sections come from
:meth:`lightsim.xsection.CrossSection.strip` (rectangular, rib and coupled
waveguides) unless a ``make`` function is passed that builds a
:class:`~lightsim.xsection.CrossSection` from the swept parameter.

Columns
-------
``mode``: index (0 = highest effective index, or least lossy for bends);
``neff``, ``neff_imag``; ``ng`` (group index, energy-flux relation with the
material dispersion); ``te_fraction``; ``aeff`` (effective area, m²);
``loss_db_cm``; sweep-specific columns as documented per function.

gplugins-style functions with micrometre arguments are in
:mod:`lightsim.gplugins_modes`.
"""

from __future__ import annotations

import math

import numpy as np

from .modes import C0, Mode, mode_overlap
from .xsection import CrossSection

#: Thermo-optic coefficients dn/dT (1/K) near 1550 nm and room temperature.
THERMO_OPTIC = {
    "si": 1.86e-4,  # Komma et al., Appl. Phys. Lett. 101, 041905 (2012)
    "silicon": 1.86e-4,
    "sio2": 0.95e-5,  # thermal oxide / fused silica
    "silica": 0.95e-5,
    "si3n4": 2.45e-5,  # Arbabi & Goddard, Opt. Lett. 38, 3878 (2013)
    "sin": 2.45e-5,
    "linbo3_e": 3.3e-5,
    "linbo3_o": 0.4e-5,
}


def table(rows: list[dict]):
    """``pandas.DataFrame`` of the rows, or a dict of lists without pandas."""
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:  # pragma: no cover
        cols = list(rows[0]) if rows else []
        return {c: [r.get(c) for r in rows] for c in cols}


def mode_row(m: Mode, **extra) -> dict:
    """The standard columns of one mode."""
    row = dict(extra)
    row.update(
        neff=float(np.real(m.neff)),
        neff_imag=float(np.imag(m.neff)),
        ng=m.n_group,
        te_fraction=m.te_fraction,
        aeff=m.effective_area(),
        loss_db_cm=m.loss_db_per_cm,
    )
    return row


def _solve(xs: CrossSection, wavelength, num_modes, group_index, sort, **kw):
    return xs.solve(wavelength, num_modes, group_index=group_index, sort=sort, **kw)


def sweep(make, values, wavelength: float = 1.55e-6, num_modes: int = 2, parameter: str = "value", group_index: bool = True, sort: str = "neff", **solve_kwargs):
    """Solve ``make(value)`` (a function returning a :class:`CrossSection`)
    for every value and tabulate the modes; the swept value is the column
    ``parameter``."""
    rows = []
    for v in values:
        xs = make(v)
        for i, m in enumerate(_solve(xs, wavelength, num_modes, group_index, sort, **solve_kwargs)):
            rows.append(mode_row(m, **{parameter: v, "mode": i}))
    return table(rows)


def neff_vs_width(widths, wavelength: float = 1.55e-6, num_modes: int = 2, group_index: bool = True, **strip_kwargs):
    """Effective and group index against core width (m), column ``width``."""
    return sweep(lambda w: CrossSection.strip(core_width=w, **strip_kwargs), widths, wavelength, num_modes, "width", group_index)


def neff_vs_thickness(thicknesses, wavelength: float = 1.55e-6, num_modes: int = 2, group_index: bool = True, **strip_kwargs):
    """Effective and group index against core thickness (m), column ``thickness``."""
    return sweep(lambda t: CrossSection.strip(core_thickness=t, **strip_kwargs), thicknesses, wavelength, num_modes, "thickness", group_index)


def neff_vs_wavelength(wavelengths, num_modes: int = 1, xs: CrossSection | None = None, **strip_kwargs):
    """Dispersion: effective index, group index and group-velocity
    dispersion against wavelength (m).

    Extra columns: ``wavelength``, ``D_ps_nm_km`` (``D = (1/c) d n_g / d
    lambda``, from central differences of ``ng`` between neighbouring
    wavelengths; NaN at the ends of the list) and ``beta2_ps2_km``
    (``beta_2 = -lambda² D / (2 pi c)``). Modes are matched between
    wavelengths by their index in the list, so keep ``num_modes`` small
    where modes cross.
    """
    xs = xs or CrossSection.strip(**strip_kwargs)
    wl = np.asarray(wavelengths, dtype=float)
    rows = []
    for w in wl:
        for i, m in enumerate(xs.solve(w, num_modes, group_index=True)):
            rows.append(mode_row(m, wavelength=float(w), mode=i))
    for i in range(num_modes):
        sel = [r for r in rows if r["mode"] == i]
        ng = np.array([r["ng"] for r in sel])
        lam = np.array([r["wavelength"] for r in sel])
        D = np.full(len(sel), np.nan)
        if len(sel) >= 3:
            D[1:-1] = (ng[2:] - ng[:-2]) / (lam[2:] - lam[:-2]) / C0
        for r, d in zip(sel, D):
            r["D_ps_nm_km"] = d * 1e6  # s/m² -> ps/(nm km)
            r["beta2_ps2_km"] = -(r["wavelength"] ** 2) * d / (2 * math.pi * C0) * 1e27
    return table(rows)


def with_temperature(material, delta_t: float, dn_dt: float | None = None):
    """Material spec at ``delta_t`` kelvin above the reference: ``n + dn/dT
    delta_t`` with ``dn/dT`` from :data:`THERMO_OPTIC` (by name) unless
    given."""
    from .xsection import _index_of, _material_name

    name = _material_name(material)
    coef = dn_dt if dn_dt is not None else THERMO_OPTIC.get(name)
    if coef is None:
        raise ValueError(f"no thermo-optic coefficient for {name!r}; pass dn_dT={{{name!r}: value}}")

    def f(wavelength):
        return _index_of(material, wavelength) + coef * delta_t

    f.__name__ = name
    return f


def neff_vs_temperature(temperatures, wavelength: float = 1.55e-6, num_modes: int = 1, t0: float = 25.0, dn_dT: dict | None = None, core_material="si", clad_material="sio2", **strip_kwargs):
    """Effective and group index against temperature (degrees C), with linear
    thermo-optic coefficients (:data:`THERMO_OPTIC`, or ``dn_dT`` per
    material name). Extra column ``dneff_dT`` (1/K, central differences)."""
    dn_dT = {k.lower(): v for k, v in (dn_dT or {}).items()}

    def mat(m, dt):
        from .xsection import _material_name

        return with_temperature(m, dt, dn_dT.get(_material_name(m)))

    tab = sweep(lambda T: CrossSection.strip(core_material=mat(core_material, T - t0), clad_material=mat(clad_material, T - t0), **strip_kwargs), temperatures, wavelength, num_modes, "temperature", True)
    rows = tab.to_dict("records") if hasattr(tab, "to_dict") else [dict(zip(tab, v)) for v in zip(*tab.values())]
    for i in range(num_modes):
        sel = [r for r in rows if r["mode"] == i]
        T = np.array([r["temperature"] for r in sel], dtype=float)
        n = np.array([r["neff"] for r in sel])
        d = np.gradient(n, T) if len(sel) > 1 else np.full(len(sel), np.nan)
        for r, v in zip(sel, d):
            r["dneff_dT"] = float(v)
    return table(rows)


def supermodes(gap: float, wavelength: float = 1.55e-6, num_modes: int = 4, xs: CrossSection | None = None, **strip_kwargs):
    """Supermodes of two parallel guides (``CrossSection.strip(gap=...)``)."""
    xs = xs or CrossSection.strip(gap=gap, **strip_kwargs)
    return xs.solve(wavelength, num_modes, sort="te")


def coupling_from_supermodes(n_even, n_odd, wavelength: float, power_ratio: float = 1.0) -> dict:
    """Coupling of a symmetric directional coupler from its supermodes.

    ``kappa = pi (n_even - n_odd) / lambda`` (1/m) is the coupling
    coefficient of coupled-mode theory, ``Lc = pi / (2 kappa) = lambda /
    (2 dn)`` the length for full transfer and ``L(power_ratio) = arcsin(
    sqrt(power_ratio)) / kappa`` the length for a fraction ``power_ratio``
    of the power in the cross port.
    """
    dn = float(np.real(n_even - n_odd))
    kappa = math.pi * dn / wavelength
    return {"dn": dn, "kappa": kappa, "lc": wavelength / (2 * dn), "length": math.asin(math.sqrt(power_ratio)) / kappa}


def coupling_vs_gap(gaps, wavelength: float = 1.55e-6, polarization: str = "te", power_ratio: float = 1.0, **strip_kwargs):
    """Supermodes and coupling against gap (m).

    Columns: ``gap``, ``n_even``, ``n_odd``, ``dn``, ``kappa`` (1/m),
    ``lc`` (m, full transfer), ``length`` (m, for ``power_ratio``). The
    even and odd supermodes are the two highest-index modes of the chosen
    polarization (``"te"``: TE fraction above 0.5).
    """
    rows = []
    for g in gaps:
        ms = supermodes(g, wavelength, num_modes=6, **strip_kwargs)
        sel = [m for m in ms if (m.te_fraction > 0.5) == (polarization == "te")]
        sel.sort(key=lambda m: -m.neff.real)
        if len(sel) < 2:
            raise RuntimeError(f"fewer than two {polarization} supermodes at gap {g}")
        c = coupling_from_supermodes(sel[0].neff, sel[1].neff, wavelength, power_ratio)
        rows.append({"gap": g, "n_even": sel[0].neff.real, "n_odd": sel[1].neff.real, **c})
    return table(rows)


def bend_sweep(radii, wavelength: float = 1.55e-6, num_modes: int = 1, pml_cells: int = 20, outer_margin: float = 2.0e-6, vertical_pml: bool = False, xs_factory=None, **strip_kwargs):
    """Bend modes against radius (m).

    For each radius the cross-section is widened by ``outer_margin`` on the
    outer side and terminated there by ``pml_cells`` of PML (also below and
    above with ``vertical_pml``, for leakage into a substrate; a PML that
    reaches the evanescent tail of the mode adds loss of its own). Columns: ``radius``, ``mode``, ``neff`` (at the
    radius), ``loss_db_90`` (loss of a 90-degree turn), ``loss_db_cm``,
    ``mismatch_db`` (loss of one straight-to-bend junction, ``-10
    log10 |<bend|straight>|²`` with the conjugated overlap of
    :func:`lightsim.modes.mode_overlap`), ``shift`` (m, outward shift of
    the power centroid relative to the straight guide).
    """
    def make(r=None):
        if xs_factory is not None:
            return xs_factory(r)
        kw = dict(strip_kwargs)
        side = kw.pop("side_margin", 1.5e-6)
        xs = CrossSection.strip(side_margin=side, **kw)
        (u0, v0), (u1, v1) = xs.window
        xs.window = ((u0, v0), (u1 + outer_margin, v1))
        xs.mesh(xs.window, step=kw.get("step", 20e-9), max_step=kw.get("max_step"))
        return xs

    xs = make()
    vp = pml_cells if vertical_pml else 0
    pml = {"cells": (0, pml_cells, vp, vp)}
    straight = xs.solve(wavelength, max(num_modes, 2), pml=pml)
    rows = []
    for r in radii:
        bends = xs.solve(wavelength, num_modes, pml=pml, bend_radius=r)
        for i, b in enumerate(bends):
            s = max(straight, key=lambda m: abs(mode_overlap(m, b)))
            ov = abs(mode_overlap(s, b)) ** 2
            rows.append(
                {
                    "radius": r,
                    "mode": i,
                    "neff": float(np.real(b.neff)),
                    "neff_imag": float(np.imag(b.neff)),
                    "te_fraction": b.te_fraction,
                    "loss_db_90": b.loss_db_per_90deg(),
                    "loss_db_cm": b.loss_db_per_cm,
                    "mismatch_db": -10 * math.log10(ov),
                    "shift": _centroid(b) - _centroid(s),
                }
            )
    return table(rows)


def _centroid(m: Mode) -> float:
    f = m.fields_at_centres()
    s = np.real(f["Eu"] * np.conj(f["Hv"]) - f["Ev"] * np.conj(f["Hu"]))
    w = np.sum(s, axis=1)
    return float(np.sum(w * f["u"]) / np.sum(w))


def overlap_table(modes1, modes2, conjugate: bool = True):
    """Overlap matrix of two mode lists as a tidy table (columns ``i``,
    ``j``, ``overlap``, ``power`` = ``|overlap|²``); see
    :func:`lightsim.modes.mode_overlap` for the definitions."""
    rows = []
    for i, a in enumerate(modes1):
        for j, b in enumerate(modes2):
            o = mode_overlap(a, b, conjugate)
            rows.append({"i": i, "j": j, "overlap": o, "power": abs(o) ** 2})
    return table(rows)
