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


_WL_DEFAULT = object()  # circuit_fn called without wl


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
    spec = _ModelCall(model, settings, info)
    return spec(wl, overrides)


#: speed of light in micrometres per second (``wl`` in µm from ``f`` in Hz)
_C_UM = 299_792_458.0e6


def _accepts(v) -> bool:
    return isinstance(v, (int, float, complex, np.ndarray)) or not isinstance(v, (dict, list, str, type(None)))


class _ModelCall:
    """An instance's model with the arguments that do not change between
    calls (its ``info`` and ``settings``) filtered once; ``__call__`` adds
    the per-call overrides and the wavelength with the rules of
    :func:`_call`."""

    __slots__ = ("model", "params", "takes_all", "static", "wants_wl", "wants_f")

    def __init__(self, model, settings: dict, info: dict):
        try:
            params = inspect.signature(model).parameters
        except (TypeError, ValueError):  # pragma: no cover
            params = {}
        self.model = model
        self.params = params
        self.takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        self.static = self._filter({}, (info, settings))
        self.wants_wl = "wl" in params or self.takes_all
        self.wants_f = "f" in params

    def _filter(self, kw: dict, sources) -> dict:
        params, takes_all = self.params, self.takes_all
        for src in sources:
            for k, v in src.items():
                if (k in params or takes_all) and k != "wl" and _accepts(v):
                    if isinstance(v, bool):
                        if k in params:
                            kw[k] = v
                        continue
                    kw[k] = v
        return kw

    def kwargs(self, wl, overrides: dict, wl_f=None, f_wl=None) -> dict:
        """The model's keyword arguments. ``wl_f``: the wavelength for
        models that do not take ``f`` (from a frequency sweep); ``f_wl``:
        the frequency for models that take ``f`` but not ``wl`` (from a
        wavelength given explicitly)."""
        kw = self._filter(dict(self.static), (overrides,)) if overrides else dict(self.static)
        if self.wants_wl:
            kw["wl"] = wl_f if (wl_f is not None and not self.wants_f) else wl
        elif f_wl is not None and self.wants_f and "f" not in kw:
            kw["f"] = f_wl
        return kw

    def __call__(self, wl, overrides: dict, wl_f=None, f_wl=None):
        return self.model(**self.kwargs(wl, overrides, wl_f, f_wl))


def _frozen(v, digests: dict):
    """A hashable stand-in for a model argument and whether it stays valid
    between calls: numbers, strings and tuples by value, numpy arrays by
    shape, type and a digest of their bytes (computed once per object and
    call, in ``digests``), anything else by identity (valid within one call
    only)."""
    if v is None or isinstance(v, (bool, int, float, complex, str)):
        return (type(v).__name__, v), True
    if isinstance(v, tuple):
        parts = [_frozen(x, digests) for x in v]
        return ("tuple",) + tuple(p for p, _ in parts), all(ok for _, ok in parts)
    if isinstance(v, (np.ndarray, np.generic)):
        d = digests.get(id(v))
        if d is None:
            import hashlib

            a = np.ascontiguousarray(v)
            d = digests[id(v)] = (a.shape, a.dtype.str, hashlib.blake2b(a.view(np.uint8).reshape(-1) if a.size else b"", digest_size=16).digest())
        return ("array",) + d, True
    return ("id", id(v)), False


def _expand_ports(sd: dict) -> list:
    return sorted({p for k in sd for p in k}, key=_port_key)


def _modes_of(ports: list, port: str) -> list:
    """Port names of ``port`` in a model's port list: ``port`` itself, or
    ``port@0``, ``port@1``, ..."""
    if port in ports:
        return [port]
    return sorted([p for p in ports if p.split("@")[0] == port and "@" in p], key=_port_key)


def circuit(netlist: dict, models: dict | None = None, backend: str = "auto", recursive_netlists: dict | None = None, use_info: bool = True, threads: int | None = None, reuse_identical: bool = False, stack_instances: bool = False):
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
    backend : ``"auto"``, ``"joins"``, ``"dense"`` or ``"sparse"``.
        ``"joins"`` joins the instances one connection at a time (compiled,
        in parallel over the wavelengths); ``"dense"`` solves the matrix of
        all internal ports at every wavelength (batched), ``"sparse"`` with a
        sparse LU. ``"auto"`` takes the joins unless their operation count
        exceeds that of the dense solve, then dense (sparse above 400
        internal ports). All give the same S-matrix to rounding.
    use_info : bool
        Also pass the numbers of each instance's gdsfactory ``info`` (the
        path length of a bend, for example) to the models, as SAX 0.18 does
        with gdsfactory 9 netlists. ``False`` passes ``settings`` only.
    threads : int, optional
        Threads of the network solve (default: all cores, or
        ``RAYON_NUM_THREADS``). Given and above 1, the instances' models are
        also evaluated concurrently on that many threads when a call has
        enough points to gain from it (numpy releases the interpreter lock in
        its array loops); leave it unset for models that are not thread-safe.
    reuse_identical : bool
        Evaluate a model once for all instances that call it with the same
        arguments, and not at all for an instance whose model and arguments
        are those of the previous call (a Monte Carlo trial or an optimiser
        step re-evaluates only the instances it changes). Numbers, strings
        and tuples are compared by value, numpy arrays by their contents
        (a 128-bit digest), other objects by identity and within one call
        only. Only for models that are pure functions of their arguments (no
        random draws, no state); off by default.
    stack_instances : bool
        Evaluate all instances of one model in one call: arguments that
        differ between them (numbers, numpy arrays) are stacked along a new
        leading axis and the model's outputs split along it. For models
        written with array operations that broadcast (numpy semantics). The
        first stacked call of every group is checked against one call per
        instance; a group whose results differ, or whose model raises, is
        evaluated instance by instance from then on. Off by default.

    Returns ``(circuit_fn, info)``. ``circuit_fn(wl=1.55, **overrides)``
    returns the SDict of the external ports; ``overrides`` maps an instance
    name to a dict of model arguments (``circuit_fn(wl=wl, top={"length":
    20})``) or sets a global argument passed to every model that takes it.
    ``circuit_fn(f=...)`` sweeps the frequency in hertz instead: models that
    take ``f`` receive it, models that take only ``wl`` receive ``wl = c /
    f`` in µm (and with ``wl`` given, models that take only ``f`` receive
    ``f = c / wl``). ``info`` lists the instances, their models and the port count.
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
                sub[comp], _ = circuit(recursive_netlists[comp], models, backend, recursive_netlists, use_info, threads, reuse_identical, stack_instances)
            continue
        raise KeyError(f"no model for component {comp!r} (instance {name!r}); pass models={{{comp!r}: model}}")

    inst_names = net["instances"]
    S0 = frozenset(inst_names)
    # sub-circuits are inlined into one network (one solve per call, and the
    # instances of all cells share the model reuse and stacking); a netlist
    # that cannot be inlined is solved cell by cell as before
    flat = None
    if sub and _FLATTEN:
        try:
            flat = _flatten(net, models, recursive_netlists, {})
        except (KeyError, ValueError):
            flat = None
    if flat is not None:
        leaves, f_pairs, f_ports = flat
        subinsts = {}
    else:
        leaves = {(n,): (v, ((n, S0),)) for n, v in inst_names.items() if v["component"] in models}
        f_pairs = [(((a[0],), a[1]), ((b[0],), b[1])) for a, b in net["pairs"]]
        f_ports = {k: ((t[0],), t[1]) for k, t in net["ports"].items()}
        subinsts = {(n,): v["component"] for n, v in inst_names.items() if v["component"] not in models}
    calls = {key: _ModelCall(models[v["component"]], v["settings"], v["info"] if use_info else {}) for key, (v, _) in leaves.items()}
    levels = {key: lv for key, (_, lv) in leaves.items()}
    plans = {}
    last = {}  # reuse_identical: model results of the previous call, by arguments
    stacking = {}  # stack_instances: group key -> True (checked), False (not stackable)

    def fn(wl=_WL_DEFAULT, **overrides):
        # the frequency sweep variable: ``f`` (Hz) gives the wavelength of the
        # models that take ``wl`` only, ``wl`` (µm) the frequency of those
        # that take ``f`` only
        f = overrides.get("f") if "f" not in S0 else None
        wl_f = f_wl = None
        sub_wl = {} if wl is _WL_DEFAULT else {"wl": wl}
        if wl is _WL_DEFAULT:
            wl = 1.55
            if f is not None:
                wl_f = _C_UM / np.asarray(f, dtype=float)
        elif f is None:
            f_wl = _C_UM / np.asarray(wl, dtype=float)
        # overrides as the cell-by-cell solve applies them: an instance
        # override at its own level, every other key passed down to the
        # sub-circuits (where a key naming one of their instances overrides it)
        gcache = {}

        def down(O, S):
            g = gcache.get((id(O), S))
            if g is None:
                g = gcache[(id(O), S)] = {a: b for a, b in O.items() if a not in S}
            return g

        sdicts = {}
        jobs = []
        for key, lv in levels.items():
            O = overrides
            for _, S in lv[:-1]:
                O = down(O, S)
            n, S = lv[-1]
            g = down(O, S)
            ov = O.get(n)
            if isinstance(ov, dict) and ov:
                o = dict(g)
                o.update(ov)
            else:
                o = g
            jobs.append((key, o))
        for key, comp in subinsts.items():  # in this thread (their models may use the pool)
            sdicts[key] = sub[comp](**sub_wl, **down(overrides, S0))
        work = [(calls[name].model, calls[name].kwargs(wl, o, wl_f, f_wl)) for name, o in jobs]
        owner = list(range(len(jobs)))
        res = {}
        if reuse_identical:
            # one evaluation per distinct (model, arguments) in this call, and
            # none for those whose arguments equal the previous call's
            seen, digests, keep, keys = {}, {}, {}, [None] * len(jobs)
            for k, (m, kw) in enumerate(work):
                parts = [(a, _frozen(v, digests)) for a, v in kw.items()]
                key = (id(m), tuple(sorted((a, p) for a, (p, _) in parts)))
                owner[k] = seen.setdefault(key, k)
                if all(ok for _, (_, ok) in parts):
                    keys[k] = key
                    if owner[k] == k and key in last:
                        res[k] = last[key]
        todo = [k for k in range(len(jobs)) if owner[k] == k and k not in res]
        pooled = threads is not None and threads > 1 and _points(work, todo) >= _MODEL_POOL_MIN_POINTS
        tasks = _stack_groups(work, todo, stacking) if stack_instances else [[k] for k in todo]
        if pooled and stack_instances:  # split the stacked groups over the threads
            tasks = [t[i : i + max(2, -(-len(t) // threads))] for t in tasks for i in range(0, len(t), max(2, -(-len(t) // threads)))]

        def run(t):
            if len(t) == 1:
                return [work[t[0]][0](**work[t[0]][1])]
            return _call_stacked(work, t, stacking)

        if pooled and len(tasks) > 1:
            outs = list(_pool(threads).map(run, tasks))
        else:
            outs = [run(t) for t in tasks]
        for t, o in zip(tasks, outs):
            res.update(zip(t, o))
        for k, (name, _) in enumerate(jobs):
            sdicts[name] = res[owner[k]]
        if reuse_identical:
            keep = {keys[k]: res[k] for k in res if keys[k] is not None}
            last.clear()
            last.update(keep)
        return _solve(sdicts, f_pairs, f_ports, backend, threads, plans, fixed=True)

    info = {"instances": {n: v["component"] for n, v in net["instances"].items()}, "subcircuits": list(sub), "ports": list(net["ports"])}
    return fn, info


_FLATTEN = True  # inline sub-circuits (False: solve them cell by cell)


def _flatten(net: dict, models: dict, recnets: dict, memo: dict):
    """Inline the sub-circuits of a normalised netlist. Returns ``(leaves,
    pairs, ports)``: ``leaves`` maps an instance path (tuple of names from
    the top) to ``(instance, levels)``, ``levels`` being the ``(name, names
    of that level)`` of each level of the path; pairs and ports refer to
    ``(path, port)``. A connection to a port a sub-circuit does not export
    raises KeyError."""
    S = frozenset(net["instances"])
    leaves, sub_ports, pairs, ports = {}, {}, [], {}
    for name, v in net["instances"].items():
        comp = v["component"]
        if comp in models:
            leaves[(name,)] = (v, ((name, S),))
            continue
        if comp not in memo:
            memo[comp] = None  # recursion guard
            memo[comp] = _flatten(_normalise(recnets[comp]), models, recnets, memo)
        if memo[comp] is None:
            raise ValueError(f"recursive cell {comp!r}")
        sl, sp, sx = memo[comp]
        for k, (vv, lv) in sl.items():
            leaves[(name,) + k] = (vv, ((name, S),) + lv)
        pairs += [(((name,) + a[0], a[1]), ((name,) + b[0], b[1])) for a, b in sp]
        sub_ports[name] = {pn: ((name,) + t[0], t[1]) for pn, t in sx.items()}

    def at(inst, port):
        if inst not in sub_ports:
            return ((inst,), port)
        sp = sub_ports[inst]
        if port in sp:
            return sp[port]
        base, _, mode = port.partition("@")
        if mode and base in sp and "@" not in sp[base][1]:
            return (sp[base][0], sp[base][1] + "@" + mode)
        raise KeyError(f"{inst},{port}: not a port of the sub-circuit")

    for a, b in net["pairs"]:
        pairs.append((at(*a), at(*b)))
    for pn, t in net["ports"].items():
        ports[pn] = at(*t)
    return leaves, pairs, ports


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


def solve_network(sdicts: dict, pairs: list, ports: dict, backend: str = "auto", threads: int | None = None) -> dict:
    """External SDict of instances ``sdicts`` (name -> SDict) connected by
    ``pairs`` (((inst, port), (inst, port)), ...) with external ``ports``
    (name -> (inst, port)). Several modes per port connect mode by mode.
    ``backend`` and ``threads`` as in :func:`circuit`."""
    return _solve(sdicts, pairs, ports, backend, threads, _PLANS)


_PLANS: dict = {}  # network plans of solve_network calls, by port lists and connections
_PORTS: dict = {}  # sorted port lists, by the keys of an SDict


def _ports_of(sd: dict) -> list:
    try:
        key = tuple(sd)
        ps = _PORTS.get(key)
    except TypeError:  # pragma: no cover  (unhashable port names)
        return _expand_ports(sd)
    if ps is None:
        if len(_PORTS) > 4096:
            _PORTS.clear()
        ps = _PORTS[key] = _expand_ports(sd)
    return ps


def _numeric(v) -> bool:
    return isinstance(v, (int, float, complex, np.number, np.ndarray)) and not isinstance(v, bool) and not (isinstance(v, np.ndarray) and v.dtype.kind not in "biufc")


def _stack_groups(work, todo, stacking) -> list:
    """Group the model calls ``todo`` (indices into ``work``) that can share
    one stacked call: same model, same argument names, equal non-numeric
    arguments; groups marked not stackable are split again."""
    groups = {}
    for k in todo:
        m, kw = work[k]
        key = (id(m), tuple(sorted((a, None if _numeric(v) else _frozen(v, {})[0]) for a, v in kw.items())))
        groups.setdefault(key, []).append(k)
    tasks = []
    for key, ks in groups.items():
        if len(ks) > 1 and stacking.get(key) is not False:
            tasks.append(ks)
        else:
            tasks += [[k] for k in ks]
    return tasks


def _call_stacked(work, ks, stacking) -> list:
    """One call of the model for the instances ``ks``: the numeric arguments
    that differ are stacked along a new leading axis (in front of the batch
    axes of every argument), the outputs are split along it."""
    m, kw0 = work[ks[0]]
    key = (id(m), tuple(sorted((a, None if _numeric(v) else _frozen(v, {})[0]) for a, v in kw0.items())))
    G = len(ks)
    nd = max((np.ndim(v) for k in ks for v in work[k][1].values() if _numeric(v)), default=0)
    kw = {}
    for a, v0 in kw0.items():
        vs = [work[k][1][a] for k in ks]
        if not _numeric(v0) or all(v is v0 for v in vs) or all(np.ndim(v) == 0 and v == v0 for v in vs):
            kw[a] = v0
            continue
        shape = np.broadcast_shapes(*[np.shape(v) for v in vs])
        st = np.stack([np.broadcast_to(np.asarray(v), shape) for v in vs])
        kw[a] = st.reshape((G,) + (1,) * (nd - len(shape)) + shape)
    singles = None
    try:
        out = m(**kw)
        parts = []
        for g in range(G):
            parts.append({p: (v[g] if np.ndim(v) == nd + 1 and np.shape(v)[0] == G else v) for p, v in out.items()})
    except Exception:  # the model does not broadcast over these arguments
        stacking[key] = False
        return [work[k][0](**work[k][1]) for k in ks]
    if key not in stacking:  # first stacked call of this group: check it
        singles = [work[k][0](**work[k][1]) for k in ks]
        ok = True
        for a, b in zip(parts, singles):
            if a.keys() != b.keys():
                ok = False
                break
            for p in a:
                x, y = np.asarray(a[p]), np.asarray(b[p])
                if np.broadcast_shapes(x.shape, y.shape) != y.shape and x.shape != y.shape:
                    ok = False
                    break
                if not np.allclose(x, y, rtol=1e-12, atol=1e-14 * max(1.0, float(np.max(np.abs(y), initial=0.0)))):
                    ok = False
                    break
            if not ok:
                break
        stacking[key] = ok
        if not ok:
            return singles
    return parts


class _Plan:
    """Everything about a network solve that depends only on the port lists
    of the instances, the connections and the external ports: the global
    port numbering, the external names, the dense solve's index arrays and
    connection matrix, and the join program."""

    def __init__(self, plist: dict, pairs: list, ports: dict):
        names, index = [], {}
        for inst, ps in plist.items():
            for p in ps:
                index[(inst, p)] = len(names)
                names.append((inst, p))
        self.index, self.N = index, len(names)
        conn = []
        for (ia, pa), (ib, pb) in pairs:
            ma, mb = _modes_of(plist.get(ia, []), pa), _modes_of(plist.get(ib, []), pb)
            if not ma or not mb:
                raise KeyError(f"connection {_nm(ia)},{pa} - {_nm(ib)},{pb}: port not in the models' S-parameters")
            if len(ma) != len(mb):
                warnings.warn(f"connection {_nm(ia)},{pa} - {_nm(ib)},{pb}: {len(ma)} and {len(mb)} modes; connecting the first {min(len(ma), len(mb))}", stacklevel=4)
            for x, y in zip(ma, mb):
                conn.append((index[(ia, x)], index[(ib, y)]))
        ext_names, ext_idx = [], []
        for pname, (inst, p) in ports.items():
            ms = _modes_of(plist.get(inst, []), p)
            if not ms:
                raise KeyError(f"external port {pname} -> {_nm(inst)},{p}: port not in the model's S-parameters")
            for m in ms:
                ext_names.append(pname + (m[len(p):] if "@" in m else ""))
                ext_idx.append(index[(inst, m)])
        self.ext_names = ext_names
        internal = sorted({i for c in conn for i in c})
        self.E = np.array(ext_idx, dtype=int)
        self.I = np.array(internal, dtype=int)
        pos = {g: k for k, g in enumerate(internal)}
        nI = len(internal)
        self.Cm = np.zeros((nI, nI))
        for a, b in conn:
            self.Cm[pos[a], pos[b]] = self.Cm[pos[b], pos[a]] = 1.0
        self.inst_of = {inst: k for k, inst in enumerate(plist)}
        self._joins(plist, conn, ext_idx)

    def _joins(self, plist, conn, ext_idx):
        """The join program (:mod:`lightsim_core::circuit`): blocks are the
        instances' used ports (ports neither connected nor external are
        matched and dropped); connections are taken greedily, the one that
        leaves the smallest block first."""
        import heapq

        self.joins_ok = False
        ends = [i for c in conn for i in c]
        used = set(ends) | set(ext_idx)
        # a port joined twice, joined to itself, or both joined and external:
        # only the matrix solve defines the result
        if len(set(ends)) != len(ends) or set(ends) & set(ext_idx):
            return
        blocks, local, where = [], {}, {}
        self.fill_map = {}  # instance -> (block, {port name: local index})
        for inst, ps in plist.items():
            mine = [p for p in ps if self.index[(inst, p)] in used]
            if not mine:
                continue
            b = len(blocks)
            g = [self.index[(inst, p)] for p in mine]
            blocks.append(g)
            self.fill_map[inst] = (b, {p: k for k, p in enumerate(mine)})
            for k, x in enumerate(g):
                local[x] = k
                where[x] = b
        sizes = [len(g) for g in blocks]
        n_init = len(blocks)
        ports_of = dict(enumerate(blocks))

        def after(k):
            x, y = conn[k]
            bx, by = where[x], where[y]
            return len(ports_of[bx]) + (len(ports_of[by]) if bx != by else 0) - 2

        heap = [(after(k), k) for k in range(len(conn))]
        heapq.heapify(heap)
        done = [False] * len(conn)
        ops = []
        while heap:
            key, k = heapq.heappop(heap)
            if done[k]:
                continue
            cur = after(k)
            if cur > key:
                heapq.heappush(heap, (cur, k))
                continue
            done[k] = True
            x, y = conn[k]
            bx, by = where[x], where[y]
            new = len(sizes)
            if bx != by:
                px, py = ports_of.pop(bx), ports_of.pop(by)
                ops.append((0, bx, px.index(x), by, py.index(y), new))
                newp = [p for p in px if p != x] + [p for p in py if p != y]
            else:
                px = ports_of.pop(bx)
                ops.append((1, bx, px.index(x), 0, px.index(y), new))
                newp = [p for p in px if p != x and p != y]
            sizes.append(len(newp))
            ports_of[new] = newp
            for p in newp:
                where[p] = new
        self.sizes, self.n_init = sizes, n_init
        self.ops = np.array(ops, dtype=np.int64).reshape(-1, 6)
        nE = len(ext_idx)
        outs = []
        for i, gi in enumerate(ext_idx):
            for j, gj in enumerate(ext_idx):
                b = where[gi]
                if where[gj] == b:
                    pb = ports_of[b]
                    outs.append((i * nE + j, b, pb.index(gj), pb.index(gi)))
        self.outputs = np.array(outs, dtype=np.int64).reshape(-1, 4)
        self.n_out = nE * nE
        # operation counts per batch point: the joins against a dense LU of
        # the internal ports and its solves (complex multiply-adds)
        self.cost_joins = sum(n * n + 4 for n in sizes[n_init:]) + sum(n * n for n in sizes[:n_init])
        nI = len(self.I)
        self.cost_dense = nI**3 / 3 + nI * nI * nE + nE * nE * nI
        self.joins_ok = True

    def fill_order(self, keys):
        """For the S-parameter keys ``keys`` (per instance, in the order the
        SDicts hold them): which entries enter the join program, and their
        place ``(block, out, in)``, so that a call only looks up its values."""
        self.fill_mask, rows = [], []
        if not self.joins_ok:
            return
        for inst, ks in keys:
            fm = self.fill_map.get(inst)
            if fm is None:
                self.fill_mask.append(())
                continue
            b, loc = fm
            mask = []
            for a, o in ks:
                ia, io = loc.get(a), loc.get(o)
                use = ia is not None and io is not None
                mask.append(use)
                if use:
                    rows.append((b, io, ia))
            self.fill_mask.append(tuple(mask))
        self.fill_static = np.array(rows, dtype=np.int64).reshape(-1, 3)


def _nm(inst) -> str:
    """An instance name for messages (a path of an inlined sub-circuit as a.b)."""
    return ".".join(map(str, inst)) if isinstance(inst, tuple) else str(inst)


def _plan(sdicts: dict, pairs: list, ports: dict, plans: dict, fixed: bool = False):
    """The cached plan of a network. ``fixed``: ``pairs`` and ``ports`` are
    those of every call with this ``plans`` dict (a circuit function's), so
    the key is the instances' S-parameter keys alone."""
    try:
        keys = tuple((i, tuple(sd)) for i, sd in sdicts.items())
        key = keys if fixed else (keys, tuple(pairs), tuple(ports.items()))
        plan = plans.get(key)
    except TypeError:  # pragma: no cover  (unhashable names)
        return _Plan({inst: _ports_of(sd) for inst, sd in sdicts.items()}, pairs, ports)
    if plan is None:
        if len(plans) > 256:
            plans.clear()
        plan = plans[key] = _Plan({inst: _ports_of(sd) for inst, sd in sdicts.items()}, pairs, ports)
        plan.fill_order(keys)
    return plan


def _solve(sdicts: dict, pairs: list, ports: dict, backend: str, threads, plans: dict, fixed: bool = False) -> dict:
    if backend not in ("auto", "joins", "dense", "sparse"):
        raise ValueError(f"backend must be 'auto', 'joins', 'dense' or 'sparse', not {backend!r}")
    plan = _plan(sdicts, pairs, ports, plans, fixed)
    seen = {}
    for sd in sdicts.values():
        for v in sd.values():
            if id(v) not in seen:
                seen[id(v)] = np.shape(v)
    shape = np.broadcast_shapes(*seen.values()) if seen else ()  # np.broadcast takes at most 64 arrays
    run = getattr(_core(), "circuit_network_run", None)
    use_joins = run is not None and plan.joins_ok and (backend == "joins" or (backend == "auto" and plan.cost_joins <= plan.cost_dense))
    if backend == "joins" and not plan.joins_ok:
        raise ValueError("backend='joins' cannot solve this network (a port is connected twice, to itself, or connected and external)")
    if use_joins:
        R = _solve_joins(run, plan, sdicts, shape, threads)
        out = {}
        nE = len(plan.ext_names)
        for i, a in enumerate(plan.ext_names):
            for j, b in enumerate(plan.ext_names):
                v = R[i * nE + j]
                out[(a, b)] = v.reshape(shape) if shape else v[0]
        return out
    nw = int(np.prod(shape)) if shape else 1
    S = np.zeros((nw, plan.N, plan.N), dtype=complex)
    index = plan.index
    for inst, sd in sdicts.items():
        for (a, b), v in sd.items():
            S[:, index[(inst, b)], index[(inst, a)]] = np.broadcast_to(np.asarray(v, dtype=complex), shape).reshape(nw)
    Sx = solve_matrix(S, plan.E, plan.I, plan.Cm, "auto" if backend == "joins" else backend)[0]
    out = {}
    for i, a in enumerate(plan.ext_names):
        for j, b in enumerate(plan.ext_names):
            v = Sx[:, j, i]
            out[(a, b)] = v.reshape(shape) if shape else v[0]
    return out


_POOLS: dict = {}

#: Smallest batch (points per call) for which the models are evaluated on
#: the thread pool. Below it the interpreter lock and the hand-off cost more
#: than the threads give: on a GB10 (8 threads) a 42-instance circuit took
#: 11.3 ms serial against 14.5 ms pooled at 5,001 points, 28 against 24 ms at
#: 20,004 and 47 against 39 ms at 40,008. With ``stack_instances`` the
#: stacked groups are split over the threads above the same size. The network
#: solve is parallel in every case.
_MODEL_POOL_MIN_POINTS = 32768


def _points(work, todo) -> int:
    """Batch points of a call: the broadcast size of the array arguments."""
    shapes = [np.shape(v) for k in todo for v in work[k][1].values() if isinstance(v, np.ndarray)]
    try:
        return int(np.prod(np.broadcast_shapes(*shapes))) if shapes else 1
    except ValueError:
        return max((int(np.prod(s)) for s in shapes), default=1)


def _pool(n: int):
    from concurrent.futures import ThreadPoolExecutor

    p = _POOLS.get(n)
    if p is None:
        p = _POOLS[n] = ThreadPoolExecutor(max_workers=n, thread_name_prefix="lightsim-circuit")
    return p


def _core():
    from . import _core as core

    return core


def _solve_joins(run, plan: _Plan, sdicts: dict, shape: tuple, threads):
    if not hasattr(plan, "fill_static"):  # a plan built outside _plan
        plan.fill_order(tuple((i, tuple(sd)) for i, sd in sdicts.items()))
    values, ids, ks = [], {}, []
    for sd, mask in zip(sdicts.values(), plan.fill_mask):
        if not mask:
            continue
        for v, use in zip(sd.values(), mask):
            if not use:
                continue
            k = ids.get(id(v))
            if k is None:
                arr = np.asarray(v, dtype=complex)
                if arr.shape != shape:
                    arr = np.broadcast_to(arr, shape)
                if any(st < 0 for st in arr.strides) or not arr.dtype.isnative:
                    arr = np.ascontiguousarray(arr)
                k = ids[id(v)] = len(values)
                values.append(arr)
            ks.append(k)
    fill = np.empty((len(ks), 4), dtype=np.int64)
    fill[:, :3] = plan.fill_static
    fill[:, 3] = ks
    nb = int(np.prod(shape)) if shape else 1
    if len(values) > 1 and len(values) * nb <= _PACK_MAX:
        # small batches: one array to hand over instead of one per value
        packed = np.array(values, dtype=complex).reshape(len(values), nb)
        return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, [], list(shape), threads, packed)
    return run(plan.sizes, plan.n_init, plan.ops, fill, plan.outputs, plan.n_out, values, list(shape), threads)


#: Largest number of complex values (values x batch points) copied into one
#: array for the network solve: handing an array to the compiled solve costs
#: about 1 us, the copy about 1 ns per value, so packing pays for small
#: batches; large ones are read in place through their strides.
_PACK_MAX = 1 << 16


def simulate(component, models: dict | None = None, wl=1.55, recursive: bool = True, **overrides) -> dict:
    """S-parameters of a gdsfactory component from its netlist:
    ``circuit(component.get_netlist(recursive=...), models)(wl=wl,
    **overrides)``. ``wl`` in µm."""
    nl = component.get_netlist(recursive=True) if recursive else component.get_netlist()
    fn, _ = circuit(nl, models)
    return fn(wl=wl, **overrides)
