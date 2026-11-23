"""Frequency-domain circuit simulation: the S-matrix of a netlist of
components, each described by an S-parameter model.

Conventions (the same as SAX, so the models here drop into SAX unchanged
and SAX models drop in here):

* a **model** is a function of keyword arguments with defaults, among them
  ``wl``, the vacuum wavelength in **micrometres** (a scalar or an array);
  lengths are in micrometres, as in gdsfactory;
* it returns an **SDict**: ``{(port_in, port_out): value}``, the complex
  amplitude leaving ``port_out`` per unit amplitude entering ``port_in``.
  Pairs that are absent are zero. Port names may carry a mode index,
  ``"o1@0"``, ``"o1@1"`` (several modes per port);
* in matrix form this is ``S[out, in] = sdict[(in, out)]``, the transpose
  of SAX's ``sdense`` layout, and lightsim's own S-parameter files key the
  same entry as ``"out@m,in@n"`` (see :func:`model_from_npz`).

:func:`circuit` takes a netlist (gdsfactory's ``component.get_netlist()``,
flat or recursive, or SAX's ``instances``/``connections``/``ports`` form)
and a dict of models and returns a function with the same calling
convention: ``circuit_fn(wl=..., **instance_overrides) -> SDict`` over the
external ports. Every instance's model receives the instance's
``settings`` (and the numbers in its ``info``, e.g. the length of a
gdsfactory Euler bend) for the arguments the model accepts.

The network is solved exactly at every wavelength: with ``S`` the
block-diagonal matrix of all instance ports and ``C`` the connection
permutation, the external S-matrix is
``S_EE + S_EI C (I - S_II C)^-1 S_IE`` (linear solve, batched over
wavelengths; sparse LU for large circuits), so feedback loops such as rings
are handled. This is a linear, steady-state (frequency-domain) simulation:
time-domain signals, modulation dynamics and nonlinear effects (Kerr,
two-photon absorption, thermal or carrier effects that depend on the
power) need a transient circuit solver, which lightsim does not have.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Callable

import numpy as np

__all__ = [
    "circuit",
    "simulate",
    "sdense",
    "sdict",
    "reciprocal",
    "straight",
    "bend",
    "phase_shifter",
    "attenuator",
    "coupler",
    "coupler_ideal",
    "coupler_ring",
    "mmi1x2",
    "mmi2x2",
    "grating_coupler",
    "ring_single",
    "ring_double",
    "model_from_npz",
    "model_from_sparameters",
    "straight_from_cross_section",
    "bend_from_cross_section",
    "coupler_from_cross_section",
    "DEFAULT_MODELS",
]


# ----------------------------------------------------------------------------
# array namespace: numpy, or jax.numpy when called with jax arrays (inside
# SAX's jitted circuits)
# ----------------------------------------------------------------------------


def _xp(*args):
    for a in args:
        if type(a).__module__.startswith("jax"):
            import jax.numpy as jnp

            return jnp
    return np


def reciprocal(sd: dict) -> dict:
    """Add ``(b, a)`` for every ``(a, b)`` (passive reciprocal device)."""
    out = dict(sd)
    for (a, b), v in sd.items():
        out.setdefault((b, a), v)
    return out


def sdense(sd: dict, ports=None):
    """SDict -> ``(S, ports)`` with ``S[..., out, in]`` (lightsim / usual
    order; SAX's ``sdense`` is the transpose)."""
    names = list(ports) if ports is not None else sorted({p for k in sd for p in k}, key=_port_key)
    idx = {p: i for i, p in enumerate(names)}
    shape = np.broadcast_shapes(*[np.shape(v) for v in sd.values()]) if sd else ()
    S = np.zeros(shape + (len(names), len(names)), dtype=complex)
    for (a, b), v in sd.items():
        if a in idx and b in idx:
            S[..., idx[b], idx[a]] = v
    return S, names


def sdict(S, ports) -> dict:
    """``(S[..., out, in], ports)`` -> SDict."""
    return {(a, b): S[..., j, i] for i, a in enumerate(ports) for j, b in enumerate(ports)}


def _port_key(p: str):
    import re

    name, _, mode = p.partition("@")
    m = re.match(r"([A-Za-z_]*)(\d*)", name)
    return (m.group(1), int(m.group(2)) if m.group(2) else -1, name, int(mode) if mode else 0)


# ----------------------------------------------------------------------------
# analytic models (SAX-compatible signatures; gdsfactory port names)
# ----------------------------------------------------------------------------


def straight(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.0) -> dict:
    """Waveguide of ``length`` µm with first-order dispersion: ``n_eff(wl) =
    neff - (wl - wl0) (ng - neff) / wl0``; transmission ``10^(-loss_dB_cm
    length 1e-4 / 20) exp(2 pi i n_eff length / wl)`` (as ``sax.models.
    straight``). Ports o1, o2."""
    xp = _xp(wl, neff, ng, length)
    wl = xp.asarray(wl)
    n = neff - (wl - wl0) * (ng - neff) / wl0
    t = 10 ** (-1e-4 * loss_dB_cm * length / 20) * xp.exp(2j * xp.pi * n * length / wl)
    return reciprocal({("o1", "o2"): t})


def bend(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.1, loss_dB=0.0) -> dict:
    """Bend as a straight of the bend's path ``length`` (gdsfactory reports it
    in the instance ``info``), plus a lumped ``loss_dB`` (mode mismatch and
    radiation of the whole bend). Ports o1, o2."""
    s = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)
    a = 10 ** (-loss_dB / 20)
    return {k: v * a for k, v in s.items()}


def phase_shifter(*, wl=1.55, neff=2.34, voltage=0.0, length=10.0, loss=0.0, v_pi=1.0) -> dict:
    """Phase shifter: ``exp(i (2 pi neff length / wl + pi voltage / v_pi))``,
    amplitude ``10^(-loss length / 20)`` (``loss`` in dB/µm, as
    ``sax.models.phase_shifter`` with ``v_pi = 1``). Ports o1, o2."""
    xp = _xp(wl, voltage)
    wl = xp.asarray(wl)
    t = 10 ** (-loss * length / 20) * xp.exp(1j * (2 * xp.pi * neff * length / wl + xp.pi * voltage / v_pi))
    return reciprocal({("o1", "o2"): t})


def attenuator(*, wl=1.55, loss=0.0) -> dict:
    """Flat attenuation of ``loss`` dB. Ports o1, o2."""
    xp = _xp(wl)
    return reciprocal({("o1", "o2"): 10 ** (-loss / 20) * xp.ones_like(xp.asarray(wl, dtype=float)) + 0j})


def coupler_ideal(*, wl=1.55, coupling=0.5) -> dict:
    """Lossless 2x2 coupler with power ``coupling`` to the cross port.
    gdsfactory ports: o1, o2 on the left (bottom, top), o3, o4 on the right
    (top, bottom); bar o1-o4 and o2-o3, cross o1-o3 and o2-o4 (cross
    amplitude ``i sqrt(coupling)``)."""
    xp = _xp(wl, coupling)
    one = xp.ones_like(xp.asarray(wl, dtype=float))
    k = xp.sqrt(coupling) * one
    t = xp.sqrt(1 - coupling) * one
    return reciprocal({("o1", "o4"): t + 0j, ("o2", "o3"): t + 0j, ("o1", "o3"): 1j * k, ("o2", "o4"): 1j * k})


def coupler(*, wl=1.55, wl0=1.55, length=0.0, coupling0=0.2, dk1=1.2435, dk2=5.3022, dn=0.02, dn1=0.1169, dn2=0.4821) -> dict:
    """Dispersive directional coupler, the model of ``sax.models.coupler``:
    ``phi = kappa0(wl) + pi dn(wl) length / wl``, bar ``cos(phi)``, cross
    ``-i sin(phi)``, with ``kappa0 = coupling0 + dk1 dwl + dk2 dwl²/2``
    (bends) and ``dn = dn + dn1 dwl + dn2 dwl²/2`` (supermode index
    difference), ``dwl = wl - wl0``. Ports as :func:`coupler_ideal`."""
    xp = _xp(wl, length)
    wl = xp.asarray(wl)
    dwl = wl - wl0
    d = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    k0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    phi = k0 + xp.pi * d / wl * length
    tau, kap = xp.cos(phi) + 0j, -xp.sin(phi)
    return reciprocal({("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): 1j * kap, ("o2", "o4"): 1j * kap})


def coupler_ring(*, wl=1.55, coupling=0.1, loss_dB=0.0) -> dict:
    """Bus-ring coupler with gdsfactory's ``coupler_ring`` ports: o1 (bus,
    west), o4 (bus, east), o2 and o3 (ring, west and east). Bar o1-o4 and
    o2-o3, cross o1-o3 and o2-o4 with power ``coupling``."""
    xp = _xp(wl, coupling)
    one = xp.ones_like(xp.asarray(wl, dtype=float))
    a = 10 ** (-loss_dB / 20)
    t, k = a * xp.sqrt(1 - coupling) * one + 0j, a * 1j * xp.sqrt(coupling) * one
    return reciprocal({("o1", "o4"): t, ("o2", "o3"): t, ("o1", "o3"): k, ("o2", "o4"): k})


def _mmi_amp(wl, wl0, fwhm, loss_dB):
    xp = _xp(wl)
    f, f0 = 1 / xp.asarray(wl), 1 / wl0
    bw = 1 / (wl0 - fwhm / 2) - 1 / (wl0 + fwhm / 2)
    sigma = bw / (2 * math.sqrt(2 * math.log(2)))
    return 10 ** (-abs(loss_dB) / 20) * xp.exp(-((f - f0) ** 2) / (2 * sigma**2))


def mmi1x2(*, wl=1.55, wl0=1.55, fwhm=0.2, loss_dB=0.3) -> dict:
    """1x2 splitter (o1 -> o2, o3), Gaussian in frequency, as
    ``sax.models.mmi1x2``."""
    t = _mmi_amp(wl, wl0, fwhm, loss_dB) / math.sqrt(2) + 0j
    return reciprocal({("o1", "o2"): t, ("o1", "o3"): t})


def mmi2x2(*, wl=1.55, wl0=1.55, fwhm=0.2, loss_dB=0.3, shift=0.0, splitting_ratio_cross=0.5, splitting_ratio_thru=0.5) -> dict:
    """2x2 MMI, as ``sax.models.mmi2x2``: thru o1-o3 and o2-o4, cross
    (``i``) o1-o4 and o2-o3 (gdsfactory ``mmi2x2`` ports)."""
    thru = _mmi_amp(wl, wl0, fwhm, loss_dB) * math.sqrt(splitting_ratio_thru) + 0j
    cross = 1j * _mmi_amp(wl, wl0 + shift, fwhm, loss_dB) * math.sqrt(splitting_ratio_cross)
    return reciprocal({("o1", "o3"): thru, ("o2", "o4"): thru, ("o1", "o4"): cross, ("o2", "o3"): cross})


def grating_coupler(*, wl=1.55, wl0=1.55, loss=0.0, reflection=0.0, reflection_fiber=0.0, bandwidth=0.040) -> dict:
    """Fibre grating coupler: Gaussian transmission of ``bandwidth`` (µm,
    full width at half maximum of the power... as in ``sax.models.
    grating_coupler``: the amplitude is ``exp(-(wl - wl0)² / (4 sigma²))``,
    ``sigma = bandwidth / (2 sqrt(2 ln 2))``), peak loss ``loss`` dB,
    reflections at the waveguide (o1) and fibre (o2) sides."""
    xp = _xp(wl)
    wl = xp.asarray(wl, dtype=float)
    one = xp.ones_like(wl)
    sigma = bandwidth / (2 * math.sqrt(2 * math.log(2)))
    t = 10 ** (-loss / 20) * xp.exp(-((wl - wl0) ** 2) / (4 * sigma**2)) + 0j
    return {("o1", "o1"): reflection * one + 0j, ("o1", "o2"): t, ("o2", "o1"): t, ("o2", "o2"): reflection_fiber * one + 0j}


def ring_single(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=50.0, loss_dB_cm=0.0, coupling=0.1) -> dict:
    """All-pass ring in closed form: ``t = (r - a e^{i phi}) / (1 - r a
    e^{i phi})`` with ``r = sqrt(1 - coupling)``, round trip ``a e^{i phi}``
    of a ring of circumference ``length`` µm (:func:`straight` dispersion).
    Ports o1 (in), o2 (through)."""
    rt = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    r = math.sqrt(1 - coupling)
    t = (r - rt) / (1 - r * rt)
    return reciprocal({("o1", "o2"): t})


def ring_double(*, wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=50.0, loss_dB_cm=0.0, coupling1=0.1, coupling2=0.1) -> dict:
    """Add-drop ring in closed form (couplers ``coupling1`` at the input bus,
    ``coupling2`` at the drop bus, couplers at diametrically opposite
    points). Ports o1 (in), o2 (through), o3 (drop, on the input side), o4
    (add)."""
    xp = _xp(wl)
    rt = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    half = straight(wl=wl, wl0=wl0, neff=neff, ng=ng, length=length / 2, loss_dB_cm=loss_dB_cm)[("o1", "o2")]
    r1, r2 = math.sqrt(1 - coupling1), math.sqrt(1 - coupling2)
    k1, k2 = math.sqrt(coupling1), math.sqrt(coupling2)
    den = 1 - r1 * r2 * rt
    thru = (r1 - r2 * rt) / den
    drop = -k1 * k2 * half / den
    return reciprocal({("o1", "o2"): thru, ("o1", "o3"): drop, ("o4", "o3"): (r2 - r1 * rt) / den, ("o4", "o2"): drop})


# ----------------------------------------------------------------------------
# models from data and from the lightsim solvers
# ----------------------------------------------------------------------------


def model_from_sparameters(sp: dict, key_order: str = "auto", interpolation: str = "polar", keep_modes: bool | None = None) -> Callable:
    """Model that interpolates an S-parameter dictionary in wavelength.

    ``sp`` has ``"wavelengths"`` and complex arrays under keys
    ``"a@m,b@n"``. ``key_order``: ``"out,in"`` (lightsim's
    :func:`~lightsim.sparams.write_sparameters` and the gplugins Meep
    writer) or ``"in,out"`` (gplugins' Tidy3D writer and SAX); ``"auto"``
    takes ``"out,in"`` when the wavelengths are in metres (a lightsim file)
    and ``"in,out"`` otherwise. For reciprocal devices the two agree.
    Wavelengths in metres (lightsim) or micrometres (gplugins) are both
    accepted; the model takes ``wl`` in µm. ``interpolation``: ``"polar"``
    (magnitude and unwrapped phase, linear) or ``"linear"`` (real and
    imaginary parts, as gplugins). Port names drop ``@0`` when every key is
    mode 0, unless ``keep_modes``.
    """
    x = np.asarray(sp["wavelengths"], dtype=float).ravel()
    metres = x.max() < 1e-3
    if metres:
        x = x * 1e6
    order = ("out,in" if metres else "in,out") if key_order == "auto" else key_order
    keys = [k for k in sp if "@" in k and "," in k]
    modes = {p.split("@")[1] for k in keys for p in k.split(",")}
    km = keep_modes if keep_modes is not None else modes != {"0"}
    idx = np.argsort(x)
    x = x[idx]
    data = {}
    for k in keys:
        a, b = k.split(",")
        if not km:
            a, b = a.split("@")[0], b.split("@")[0]
        pin, pout = (b, a) if order == "out,in" else (a, b)
        v = np.asarray(sp[k], dtype=complex).ravel()[idx]
        data[(pin, pout)] = v
    phase = {k: np.unwrap(np.angle(v)) for k, v in data.items()}
    mag = {k: np.abs(v) for k, v in data.items()}

    def model(*, wl=1.55) -> dict:
        w = np.asarray(wl, dtype=float)
        out = {}
        for k, v in data.items():
            if len(x) == 1:
                out[k] = np.broadcast_to(v[0], w.shape).astype(complex)
            elif interpolation == "linear":
                out[k] = np.interp(w, x, v.real) + 1j * np.interp(w, x, v.imag)
            else:
                out[k] = np.interp(w, x, mag[k]) * np.exp(1j * np.interp(w, x, phase[k]))
        return out

    model.ports = sorted({p for k in data for p in k}, key=_port_key)
    return model


def model_from_npz(filepath, **kwargs) -> Callable:
    """Model from an S-parameter ``.npz`` file (the layout lightsim and
    gplugins share: keys ``"o1@0,o2@0"`` and ``wavelengths``); see
    :func:`model_from_sparameters` for the options."""
    with np.load(filepath) as f:
        sp = {k: f[k] for k in f.files}
    return model_from_sparameters(sp, **kwargs)


def _poly_neff(xs, wavelengths_um, mode, solve_kwargs):
    lam = np.asarray(wavelengths_um, dtype=float)
    ns = []
    for w in lam:
        ms = xs.solve(w * 1e-6, mode + 1, **solve_kwargs)
        ns.append(ms[mode].neff)
    ns = np.array(ns)
    deg = min(3, len(lam) - 1)
    return np.polyfit(lam, ns.real, deg), np.polyfit(lam, ns.imag, deg), ns


def straight_from_cross_section(xs, wavelengths=(1.5, 1.55, 1.6), mode: int = 0, extra_loss_dB_cm: float = 0.0, **solve_kwargs) -> Callable:
    """Straight-waveguide model from the mode solver: the complex effective
    index of ``mode`` of the :class:`~lightsim.xsection.CrossSection` is
    solved at ``wavelengths`` (µm) and fitted with a polynomial (degree up
    to 3), so the phase carries the full dispersion (group index and GVD)
    and the imaginary part the propagation loss (e.g. substrate leakage
    with a PML). ``extra_loss_dB_cm`` adds loss the solver does not see
    (roughness). The model is ``f(*, wl=1.55, length=10.0) -> SDict`` with
    ports o1, o2; ``f.neff(wl)`` gives the fitted index."""
    pr, pi_, ns = _poly_neff(xs, wavelengths, mode, solve_kwargs)

    def neff(wl):
        return np.polyval(pr, wl) + 1j * np.polyval(pi_, wl)

    def model(*, wl=1.55, length=10.0) -> dict:
        w = np.asarray(wl, dtype=float)
        n = neff(w)
        t = np.exp(2j * np.pi * n * length / w) * 10 ** (-1e-4 * extra_loss_dB_cm * length / 20)
        return reciprocal({("o1", "o2"): t})

    model.neff = neff
    model.samples = (np.asarray(wavelengths, dtype=float), ns)
    return model


def bend_from_cross_section(xs, radius: float, angle: float = 90.0, wavelengths=(1.5, 1.55, 1.6), mode: int = 0, pml_cells: int = 20, **solve_kwargs) -> Callable:
    """Circular-bend model from the mode solver: effective index (at the
    radius) and radiation loss of the bend mode of radius ``radius`` µm
    over ``angle`` degrees, and the two straight-to-bend junction losses
    from the overlap of the bend and straight modes, at ``wavelengths``
    (µm), fitted in wavelength. The cross-section needs room for the PML
    on its outer side (+u). Model ``f(*, wl=1.55) -> SDict``, ports o1, o2.
    """
    from .modes import mode_overlap

    lam = np.asarray(wavelengths, dtype=float)
    ns, junction = [], []
    pml = {"cells": (0, pml_cells, 0, 0)}
    for w in lam:
        s = xs.solve(w * 1e-6, mode + 1, **solve_kwargs)[mode]
        b = xs.solve(w * 1e-6, mode + 1, pml=pml, bend_radius=radius * 1e-6, **solve_kwargs)[mode]
        ns.append(b.neff)
        junction.append(mode_overlap(s, b, conjugate=False))
    ns, junction = np.array(ns), np.array(junction)
    deg = min(3, len(lam) - 1)
    pr, pim = np.polyfit(lam, ns.real, deg), np.polyfit(lam, ns.imag, deg)
    jr, ji = np.polyfit(lam, junction.real, deg), np.polyfit(lam, junction.imag, deg)
    length = radius * np.deg2rad(angle)

    def model(*, wl=1.55) -> dict:
        w = np.asarray(wl, dtype=float)
        n = np.polyval(pr, w) + 1j * np.polyval(pim, w)
        j = np.polyval(jr, w) + 1j * np.polyval(ji, w)
        return reciprocal({("o1", "o2"): j**2 * np.exp(2j * np.pi * n * length / w)})

    model.samples = (lam, ns, junction)
    return model


def coupler_from_cross_section(xs, wavelengths=(1.5, 1.55, 1.6), polarization: str = "te", coupling0: float = 0.0) -> Callable:
    """Directional-coupler model from the supermodes of the coupled section
    (``xs``, e.g. ``CrossSection.strip(gap=...)``): ``phi = coupling0 + pi
    dn(wl) length / wl``, bar ``cos(phi)``, cross ``-i sin(phi)``, with the
    supermode index difference ``dn`` solved at ``wavelengths`` (µm) and
    fitted. ``coupling0`` is the phase picked up in the bends (0: ignored).
    Model ``f(*, wl=1.55, length=10.0) -> SDict`` with the ports of
    :func:`coupler`; ``f.n_mean(wl)`` is the mean supermode index, used for
    the common phase."""
    lam = np.asarray(wavelengths, dtype=float)
    dn, nm = [], []
    for w in lam:
        ms = [m for m in xs.solve(w * 1e-6, 6, sort="te") if (m.te_fraction > 0.5) == (polarization == "te")]
        ms.sort(key=lambda m: -m.neff.real)
        dn.append(ms[0].neff.real - ms[1].neff.real)
        nm.append(0.5 * (ms[0].neff.real + ms[1].neff.real))
    deg = min(3, len(lam) - 1)
    pd, pm = np.polyfit(lam, dn, deg), np.polyfit(lam, nm, deg)

    def model(*, wl=1.55, length=10.0) -> dict:
        w = np.asarray(wl, dtype=float)
        phi = coupling0 + np.pi * np.polyval(pd, w) * length / w
        common = np.exp(2j * np.pi * np.polyval(pm, w) * length / w)
        tau, kap = common * np.cos(phi), -common * np.sin(phi)
        return reciprocal({("o1", "o4"): tau, ("o2", "o3"): tau, ("o1", "o3"): 1j * kap, ("o2", "o4"): 1j * kap})

    model.n_mean = lambda wl: np.polyval(pm, wl)
    model.dn = lambda wl: np.polyval(pd, wl)
    return model


#: Models for the gdsfactory components of common circuits (SAX defaults).
DEFAULT_MODELS = {
    "straight": straight,
    "bend_euler": bend,
    "bend_circular": bend,
    "bend_s": bend,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "coupler": coupler,
    "coupler_ring": coupler_ring,
    "grating_coupler_elliptical": grating_coupler,
    "grating_coupler_elliptical_te": grating_coupler,
    "phase_shifter": phase_shifter,
}


# ----------------------------------------------------------------------------
# netlists
# ----------------------------------------------------------------------------


def _split(ref: str):
    inst, port = ref.split(",")
    return inst.strip(), port.strip()


def _normalise(netlist: dict) -> dict:
    """gdsfactory (``nets``) or SAX (``connections``) netlist -> instances,
    pairs, ports."""
    inst = {}
    for name, v in netlist["instances"].items():
        if isinstance(v, str):
            inst[name] = {"component": v, "settings": {}, "info": {}}
        else:
            inst[name] = {"component": v.get("component"), "settings": dict(v.get("settings") or {}), "info": dict(v.get("info") or {})}
    pairs = []
    if "nets" in netlist and netlist["nets"]:
        for n in netlist["nets"]:
            pairs.append((_split(n["p1"]), _split(n["p2"])))
    for a, b in (netlist.get("connections") or {}).items():
        pairs.append((_split(a), _split(b)))
    ports = {k: _split(v) for k, v in (netlist.get("ports") or {}).items()}
    return {"instances": inst, "pairs": pairs, "ports": ports}


def _call(model, settings: dict, info: dict, wl, overrides: dict):
    try:
        params = inspect.signature(model).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kw = {}
    for src in (info, settings, overrides):
        for k, v in src.items():
            if (k in params or takes_all) and k != "wl" and (isinstance(v, (int, float, complex, np.ndarray)) or not isinstance(v, (dict, list, str, type(None)))):
                if isinstance(v, bool):
                    if k in params:
                        kw[k] = v
                    continue
                kw[k] = v
    if "wl" in params or takes_all:
        kw["wl"] = wl
    return model(**kw)


def _expand_ports(sd: dict) -> list:
    return sorted({p for k in sd for p in k}, key=_port_key)


def _modes_of(ports: list, port: str) -> list:
    """Port names of ``port`` in a model's port list: ``port`` itself, or
    ``port@0``, ``port@1``, ..."""
    if port in ports:
        return [port]
    return sorted([p for p in ports if p.split("@")[0] == port and "@" in p], key=_port_key)


def circuit(netlist: dict, models: dict | None = None, backend: str = "auto", recursive_netlists: dict | None = None, use_info: bool = True):
    """Build a circuit function from a netlist and models.

    Parameters
    ----------
    netlist : dict
        ``component.get_netlist()`` (gdsfactory: ``instances``, ``nets``,
        ``ports``), the dict of ``component.get_netlist(recursive=True)``
        (the top cell is the one no other cell instantiates), or a SAX
        netlist (``instances``, ``connections``, ``ports``).
    models : dict
        Component name -> model function. Missing names fall back to
        :data:`DEFAULT_MODELS` and, in a recursive netlist, to the cell's
        own netlist (a sub-circuit).
    backend : ``"dense"``, ``"sparse"`` or ``"auto"`` (sparse above 400
        internal ports).
    use_info : bool
        Also pass the numbers of each instance's gdsfactory ``info`` (the
        path length of a bend, for example) to the models, as SAX 0.18 does
        with gdsfactory 9 netlists. ``False`` passes ``settings`` only.

    Returns ``(circuit_fn, info)``. ``circuit_fn(wl=1.55, **overrides)``
    returns the SDict of the external ports; ``overrides`` maps an instance
    name to a dict of model arguments (``circuit_fn(wl=wl, top={"length":
    20})``) or sets a global argument passed to every model that takes it.
    ``info`` lists the instances, their models and the port count.
    """
    models = dict(DEFAULT_MODELS, **(models or {}))
    if "instances" not in netlist:  # recursive netlist dict {cell: netlist}
        recursive_netlists = netlist
        used = {v["component"] if isinstance(v, dict) else v for n in netlist.values() for v in n.get("instances", {}).values()}
        tops = [k for k in netlist if k not in used]
        if len(tops) != 1:
            raise ValueError(f"cannot find the top cell of the recursive netlist (candidates: {tops})")
        netlist = netlist[tops[0]]
    net = _normalise(netlist)
    sub = {}
    for name, v in net["instances"].items():
        comp = v["component"]
        if comp in models:
            continue
        if recursive_netlists and comp in recursive_netlists:
            if comp not in sub:
                sub[comp], _ = circuit(recursive_netlists[comp], models, backend, recursive_netlists, use_info)
            continue
        raise KeyError(f"no model for component {comp!r} (instance {name!r}); pass models={{{comp!r}: model}}")

    def fn(wl=1.55, **overrides):
        globals_ = {k: v for k, v in overrides.items() if k not in net["instances"]}
        sdicts = {}
        for name, v in net["instances"].items():
            comp = v["component"]
            ov = dict(globals_)
            ov.update(overrides.get(name, {}) if isinstance(overrides.get(name), dict) else {})
            if comp in models:
                sdicts[name] = _call(models[comp], v["settings"], v["info"] if use_info else {}, wl, ov)
            else:
                sdicts[name] = sub[comp](wl=wl, **globals_)
        return solve_network(sdicts, net["pairs"], net["ports"], backend=backend)

    info = {"instances": {n: v["component"] for n, v in net["instances"].items()}, "subcircuits": list(sub), "ports": list(net["ports"])}
    return fn, info


def solve_matrix(S, E, I, Cm, backend: str = "auto", adjoint: bool = False):
    """The linear algebra of :func:`solve_network`, on index arrays.

    ``S[k, out, in]`` holds all instance ports (``k`` over wavelengths),
    ``E`` and ``I`` are the external and internal port indices and ``Cm`` the
    symmetric connection matrix on ``I``. Returns ``(Sx, CX, RI)``:
    ``Sx = S_EE + S_EI C (1 - S_II C)^-1 S_IE`` and, with ``adjoint``, the
    per-wavelength factors ``CX = C (1 - S_II C)^-1 S_IE`` and
    ``RI = S_EI C (1 - S_II C)^-1`` from which the gradient with respect to
    ``S`` follows (:mod:`lightsim.invdes.network`); else ``None, None``."""
    S_EE = S[:, E][:, :, E]
    nI = len(I)
    nw = S.shape[0]
    if nI == 0:
        return S_EE, ([None] * nw if adjoint else None), ([None] * nw if adjoint else None)
    S_EI, S_IE, S_II = S[:, E][:, :, I], S[:, I][:, :, E], S[:, I][:, :, I]
    use_sparse = backend == "sparse" or (backend == "auto" and nI > 400)
    CXs, RIs = ([], []) if adjoint else (None, None)
    if use_sparse:
        import scipy.sparse as sp_
        import scipy.sparse.linalg as spla

        Cs = sp_.csr_matrix(Cm)
        Sx = np.empty_like(S_EE)
        for k in range(nw):
            A = sp_.identity(nI, format="csc") - sp_.csr_matrix(S_II[k]) @ Cs
            lu = spla.splu(A.tocsc())
            CX = Cs @ lu.solve(S_IE[k])
            Sx[k] = S_EE[k] + S_EI[k] @ CX
            if adjoint:
                CXs.append(CX)
                RIs.append(lu.solve(np.ascontiguousarray((S_EI[k] @ Cs.toarray()).T), trans="T").T)
    else:
        A = np.eye(nI)[None] - S_II @ Cm[None]
        CX = Cm[None] @ np.linalg.solve(A, S_IE)
        Sx = S_EE + S_EI @ CX
        if adjoint:
            RI = np.swapaxes(np.linalg.solve(np.swapaxes(A, 1, 2), np.swapaxes(S_EI @ Cm[None], 1, 2)), 1, 2)
            CXs, RIs = list(CX), list(RI)
    return Sx, CXs, RIs


def solve_network(sdicts: dict, pairs: list, ports: dict, backend: str = "auto") -> dict:
    """External SDict of instances ``sdicts`` (name -> SDict) connected by
    ``pairs`` (((inst, port), (inst, port)), ...) with external ``ports``
    (name -> (inst, port)). Several modes per port connect mode by mode."""
    names = []  # (inst, portname)
    index = {}
    plist = {}
    for inst, sd in sdicts.items():
        ps = _expand_ports(sd)
        plist[inst] = ps
        for p in ps:
            index[(inst, p)] = len(names)
            names.append((inst, p))
    N = len(names)
    # wavelength shape
    vals = [np.asarray(v) for sd in sdicts.values() for v in sd.values()]
    shape = np.broadcast_shapes(*[v.shape for v in vals]) if vals else ()  # np.broadcast takes at most 64 arrays
    nw = int(np.prod(shape)) if shape else 1
    S = np.zeros((nw, N, N), dtype=complex)
    for inst, sd in sdicts.items():
        for (a, b), v in sd.items():
            S[:, index[(inst, b)], index[(inst, a)]] = np.broadcast_to(np.asarray(v, dtype=complex), shape).reshape(nw)
    # connections, mode by mode
    conn = []
    for (ia, pa), (ib, pb) in pairs:
        ma, mb = _modes_of(plist.get(ia, []), pa), _modes_of(plist.get(ib, []), pb)
        if not ma or not mb:
            raise KeyError(f"connection {ia},{pa} - {ib},{pb}: port not in the models' S-parameters")
        if len(ma) != len(mb):
            warnings.warn(f"connection {ia},{pa} - {ib},{pb}: {len(ma)} and {len(mb)} modes; connecting the first {min(len(ma), len(mb))}", stacklevel=2)
        for x, y in zip(ma, mb):
            conn.append((index[(ia, x)], index[(ib, y)]))
    ext_names, ext_idx = [], []
    for pname, (inst, p) in ports.items():
        ms = _modes_of(plist.get(inst, []), p)
        if not ms:
            raise KeyError(f"external port {pname} -> {inst},{p}: port not in the model's S-parameters")
        for m in ms:
            ext_names.append(pname + (m[len(p):] if "@" in m else ""))
            ext_idx.append(index[(inst, m)])
    internal = sorted({i for c in conn for i in c})
    E = np.array(ext_idx, dtype=int)
    I = np.array(internal, dtype=int)
    pos = {g: k for k, g in enumerate(internal)}
    nI = len(I)
    Cm = np.zeros((nI, nI))
    for a, b in conn:
        Cm[pos[a], pos[b]] = Cm[pos[b], pos[a]] = 1.0
    Sx = solve_matrix(S, E, I, Cm, backend)[0]
    out = {}
    for i, a in enumerate(ext_names):
        for j, b in enumerate(ext_names):
            v = Sx[:, j, i]
            out[(a, b)] = v.reshape(shape) if shape else v[0]
    return out


def simulate(component, models: dict | None = None, wl=1.55, recursive: bool = True, **overrides) -> dict:
    """S-parameters of a gdsfactory component from its netlist:
    ``circuit(component.get_netlist(recursive=...), models)(wl=wl,
    **overrides)``. ``wl`` in µm."""
    nl = component.get_netlist(recursive=True) if recursive else component.get_netlist()
    fn, _ = circuit(nl, models)
    return fn(wl=wl, **overrides)
