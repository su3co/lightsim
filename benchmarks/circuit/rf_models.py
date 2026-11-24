"""Circuit models of the quantum RF PDK's resonator test chip (qpdk 0.4.3),
written once with numpy (or jax.numpy when called with jax arrays), so the
same functions run in lightsim.circuit and in SAX, jitted or not.

Every model takes the frequency ``f`` in hertz (the PDK's convention, and
lightsim.circuit's frequency sweep variable) and the CPW cross-section as two
numbers, centre ``width`` and ``gap`` in micrometres, instead of a
gdsfactory cross-section object, so a Monte Carlo trial is an array
argument. The physics is the PDK's: conformal-mapping CPW on a 500 um
silicon substrate with the thickness correction of Gupta et al., dielectric
loss from the loss tangent, matched lines, ideal tees, and the coupling
capacitance interpolated from the PDK's Palace dataset. The launcher is a
wide line and a 100-step taper; the coupled quarter-wave resonator is the
PDK's sub-circuit reduced to closed form. ``check_models`` compares them with
the PDK's own SAX models (``bench_rf.py --check``).
"""

from __future__ import annotations

import math

import numpy as np

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

# ---- the PDK's layer stack and materials (qpdk.tech) -------------------------
EPS_SI = 11.45  # silicon at millikelvin temperatures
TAND_SI = 2.7e-6  # substrate loss tangent
H_SUB = 500.0  # um, substrate thickness
T_FILM = 0.2  # um, M1 (Nb) thickness
W0, G0 = 10.0, 6.0  # um, nominal CPW centre width and gap
LAUNCHER_BIG = (200.0, 110.0)  # um, launcher_cross_section_big (width, gap)
TAPER_STEPS = 100  # taper_cross_section default npoints


def _xp(*args):
    """numpy, or jax.numpy if any argument is a jax array (or tracer)."""
    for a in args:
        if type(a).__module__.startswith("jax"):
            import jax.numpy as jnp

            return jnp
    return np


def _ellipk(m, xp):
    if xp is np:
        from scipy.special import ellipk

        return ellipk(m)
    import jaxellip

    return jaxellip.ellipk(m)


def ellipk_ratio(m, xp=np):
    """K(m) / K(1 - m), parameter convention (m = k^2)."""
    return _ellipk(m, xp) / _ellipk(1 - m, xp)


# ---- CPW parameters ----------------------------------------------------------
class AnalyticCPW:
    """The PDK's CPW: conformal mapping on a finite substrate plus the
    Gupta et al. thickness correction (qpdk.models.cpw.cpw_parameters with the
    default layer stack). ``eps_eff(w, s)`` and ``z0(w, s)`` take um."""

    name = "analytic (conformal mapping)"

    def __init__(self, h=H_SUB, t=T_FILM, eps_r=EPS_SI):
        self.h, self.t, self.eps_r = h, t, eps_r

    def both(self, w, s):
        xp = _xp(w, s)
        w = xp.asarray(w, dtype=float) * 1e-6
        s = xp.asarray(s, dtype=float) * 1e-6
        h, t = self.h * 1e-6, self.t * 1e-6
        k0 = w / (w + 2 * s)
        k1 = xp.sinh(xp.pi * w / (4 * h)) / xp.sinh(xp.pi * (w + 2 * s) / (4 * h))
        q0 = ellipk_ratio(k0**2, xp)
        q1 = ellipk_ratio(k1**2, xp) / q0
        ep = 1 + q1 * (self.eps_r - 1) / 2
        if self.t <= 0:
            return ep, 30 * xp.pi / (xp.sqrt(ep) * q0)
        delta = (1.25 * t / xp.pi) * (1 + xp.log(4 * xp.pi * w / t))
        ke = xp.clip(k0 + (1 - k0**2) * delta / (2 * s), 1e-12, 1 - 1e-12)
        ep_t = ep - (0.7 * (ep - 1) * t / s) / (q0 + 0.7 * t / s)
        z0_t = 30 * xp.pi / (xp.sqrt(ep_t) * ellipk_ratio(ke**2, xp))
        return ep_t, z0_t

    def eps_eff(self, w, s):
        return self.both(w, s)[0]

    def z0(self, w, s):
        return self.both(w, s)[1]


# ---- coupling capacitance -----------------------------------------------------
class _Multilinear:
    """Linear interpolation on a regular (rectilinear) grid, the algorithm of
    scipy's and jax's RegularGridInterpolator(method="linear"), NaN outside."""

    def __init__(self, coords, values):
        self.coords = [np.asarray(c, float) for c in coords]
        self.values = np.asarray(values, float)  # (*grid, ncomp)

    def __call__(self, pts):
        xp = _xp(pts)
        pts = xp.asarray(pts, dtype=float)
        shape = pts.shape[:-1]
        pts = pts.reshape(-1, len(self.coords))
        vals = xp.asarray(self.values)
        idx, frac = [], []
        inside = xp.ones(pts.shape[0], dtype=bool)
        for d, c in enumerate(self.coords):
            c = xp.asarray(c)
            x = pts[:, d]
            i = xp.clip(xp.searchsorted(c, x) - 1, 0, len(c) - 2)
            idx.append(i)
            frac.append((x - c[i]) / (c[i + 1] - c[i]))
            inside = inside & (x >= c[0]) & (x <= c[-1])
        out = 0.0
        nd = len(self.coords)
        for corner in range(2**nd):
            wgt = 1.0
            ii = []
            for d in range(nd):
                bit = (corner >> d) & 1
                wgt = wgt * (frac[d] if bit else 1 - frac[d])
                ii.append(idx[d] + bit)
            out = out + wgt[:, None] * vals[tuple(ii)]
        out = xp.where(inside[:, None], out, xp.nan)
        return out.reshape(*shape, -1)


class PalaceCoupling:
    """Mutual capacitance per length of the probeline-resonator section from
    the PDK's Palace datasets (``cpw_coupling_ground_strip_palace``, with
    ``cpw_coupling_palace`` where the two slots touch), interpolated as the
    PDK does: log coordinates and log magnitudes on the ground-strip grid,
    linear on the etched grid. ``c_pul(w, s, gap)`` in F/m; gap is the
    distance between the two centre conductors (um)."""

    name = "PDK table (Palace, 2-D, zero-thickness sheets)"

    def __init__(self):
        from qpdk.models.datasets import Dataset

        grounded = Dataset("cpw_coupling_ground_strip_palace")
        etched = Dataset("cpw_coupling_palace")
        g = grounded.grid("maxwell_capacitance")
        e = etched.grid("maxwell_capacitance")
        st = grounded.metadata.provenance["settings"]
        self.slice_m = float(st["slice_length_um"]) * 1e-6
        self.sim_eps = float(st["permittivity"])
        self.domain = dict(g.domain)
        vg = np.asarray(g.values).reshape(*np.asarray(g.values).shape[:3], 4)
        self._grounded = _Multilinear([np.log(np.asarray(c, float)) for c in g.coords], np.log(np.abs(vg)))
        self._etched = _Multilinear([np.asarray(c, float) for c in e.coords], np.asarray(e.values).reshape(*np.asarray(e.values).shape[:3], 4))
        self.e_domain = dict(e.domain)
        self.grid_coords = [np.asarray(c, float) for c in g.coords]

    def c_pul(self, w, s, gap):
        xp = _xp(w, s, gap)
        w, s, gap = xp.broadcast_arrays(xp.asarray(w, dtype=float), xp.asarray(s, dtype=float), xp.asarray(gap, dtype=float))
        strip = gap - 2 * s
        lo, hi = self.domain["ground_strip_width"]
        tol = 8 * np.finfo(float).eps * xp.maximum(xp.abs(gap), 1)
        pts = xp.stack([xp.log(w), xp.log(s), xp.log(xp.clip(strip, lo, hi))], -1)
        c01_g = -xp.exp(self._grounded(pts)[..., 1])
        inside = xp.ones(w.shape, dtype=bool)
        for name, val in (("width", w), ("cpw_gap", s), ("ground_strip_width", strip)):
            a, b = self.domain[name]
            if name == "ground_strip_width":
                a, b = a - tol, b + tol
            inside = inside & (val >= a) & (val <= b)
        c01_g = xp.where(inside, c01_g, xp.nan)
        c01_e = self._etched(xp.stack([w, s, gap], -1))[..., 1]
        for name, val in (("width", w), ("cpw_gap", s), ("gap", gap)):
            a, b = self.e_domain[name]
            c01_e = xp.where((val >= a) & (val <= b), c01_e, xp.nan)
        c01 = xp.where(strip <= 0, c01_e, c01_g)
        eps_ratio = ((1 + EPS_SI) / 2) / ((1 + self.sim_eps) / 2)
        return -c01 / self.slice_m * eps_ratio


# ---- line and component models (lightsim convention: wl in um) ---------------
def gamma(wl, eps_eff, eps_r=EPS_SI, tand=TAND_SI):
    """Propagation constant (1/m) of a quasi-TEM line at vacuum wavelength wl (um):
    dielectric loss from the substrate loss tangent (Pozar 3.8), as the PDK."""
    xp = _xp(wl, eps_eff)
    lam = xp.asarray(wl) * 1e-6
    beta = 2 * xp.pi * xp.sqrt(eps_eff) / lam
    alpha = xp.pi / lam * (eps_r / xp.sqrt(eps_eff)) * ((eps_eff - 1) / (eps_r - 1)) * tand
    return alpha + 1j * beta


class Chip:
    """The chip's models with a choice of CPW and coupling plug-ins.

    ``models()`` returns a dict ``component name -> model(wl, ...)`` for
    lightsim.circuit; every model also takes ``width`` and ``gap`` (um) of its
    CPW cross-section, which the Monte Carlo overrides per instance."""

    def __init__(self, cpw=None, coupling=None, eps_r=EPS_SI, tand=TAND_SI):
        self.cpw = cpw or AnalyticCPW()
        self.coupling = coupling or PalaceCoupling()
        self.eps_r, self.tand = eps_r, tand

    # -- pieces
    def phase(self, wl, length, w, s):
        """exp(-gamma L) of a matched line of length L (um)."""
        xp = _xp(wl, w, s)
        ep = self.cpw.eps_eff(w, s)
        return xp.exp(-gamma(wl, ep, self.eps_r, self.tand) * (xp.asarray(length) * 1e-6))

    def straight(self, wl=40_000.0, length=10.0, width=W0, gap=G0):
        t = self.phase(wl, length, width, gap)
        z = 0 * t
        return {("o1", "o2"): t, ("o2", "o1"): t, ("o1", "o1"): z, ("o2", "o2"): z}

    def launcher(self, wl=40_000.0, straight_length=200.0, taper_length=100.0):
        """Wide line, then a taper of TAPER_STEPS matched sections (the PDK's launcher)."""
        xp = _xp(wl)
        (wb, gb) = LAUNCHER_BIG
        ws = np.linspace(wb, W0, TAPER_STEPS)  # fixed numbers: the launcher is not perturbed
        gs = np.linspace(gb, G0, TAPER_STEPS)
        ep_big = self.cpw_launcher_eps(wb, gb)
        theta = gamma(wl, ep_big, self.eps_r, self.tand) * (straight_length * 1e-6)
        seg = taper_length / TAPER_STEPS * 1e-6
        for w, g in zip(ws, gs):
            theta = theta + gamma(wl, self.cpw_launcher_eps(w, g), self.eps_r, self.tand) * seg
        t = xp.exp(-theta)
        z = 0 * t
        return {("waveport", "o1"): t, ("o1", "waveport"): t, ("waveport", "waveport"): z, ("o1", "o1"): z}

    def cpw_launcher_eps(self, w, g):
        """The launcher is not perturbed and lies outside the FEM tables: always analytic."""
        return AnalyticCPW().eps_eff(w, g)

    def resonator(self, wl=40_000.0, length=4000.0, coupling_gap=16.0, coupling_straight_length=200.0, width=W0, gap=G0, width_nr=W0, gap_nr=G0):
        """Quarter-wave resonator coupled to the probeline, in closed form.

        The PDK's sub-circuit: probeline halves (coupling_1, coupling_2), an ideal
        tee, the coupling capacitor, a second tee into an open stub of
        coupling_straight_length / 2 and a shorted stub of the rest. With every
        line matched to its own impedance the shunt branch seen from the
        probeline is a normalised admittance y and S21 = 2 / (2 + y)."""
        xp = _xp(wl, width, gap)
        lam_m = xp.asarray(wl) * 1e-6
        omega = 2 * xp.pi * C0 / lam_m
        ep, z0 = self.cpw.both(width, gap)
        g_r = gamma(wl, ep, self.eps_r, self.tand)
        c = self.coupling.c_pul(width, gap, coupling_gap) * coupling_straight_length * 1e-6
        l1 = coupling_straight_length / 2 * 1e-6
        l2 = (length - coupling_straight_length / 2) * 1e-6
        g_open = xp.exp(-2 * g_r * l1)  # open end: reflection +1
        g_short = -xp.exp(-2 * g_r * l2)  # shorted end: reflection -1
        y_node = (1 - g_open) / (1 + g_open) + (1 - g_short) / (1 + g_short)
        z_branch = 1 / (1j * omega * c * z0) + 1 / y_node
        y = 1 / z_branch
        t_c = self.phase(wl, coupling_straight_length, width_nr, gap_nr)
        s21 = t_c * 2 / (2 + y)
        s11 = t_c * (-y / (2 + y))
        return {("coupling_o1", "coupling_o2"): s21, ("coupling_o2", "coupling_o1"): s21, ("coupling_o1", "coupling_o1"): s11, ("coupling_o2", "coupling_o2"): s11}

    def models(self):
        return {"straight": self.straight, "bend_s": self.straight, "launcher": self.launcher, "quarter_wave_resonator_coupled": self.resonator}


def wl_of_f(f):
    """Vacuum wavelength (um) of a frequency (Hz)."""
    return C0 / np.asarray(f, dtype=float) * 1e6


def frequency_models(chip):
    """The models of ``chip`` with ``f`` (Hz) in place of ``wl`` (um)."""

    def straight(f=5e9, length=10.0, width=W0, gap=G0):
        return chip.straight(wl=wl_of_f_x(f), length=length, width=width, gap=gap)

    def launcher(f=5e9, straight_length=200.0, taper_length=100.0):
        return chip.launcher(wl=wl_of_f_x(f), straight_length=straight_length, taper_length=taper_length)

    def resonator(f=5e9, length=4000.0, coupling_gap=16.0, coupling_straight_length=200.0, width=W0, gap=G0):
        return chip.resonator(wl=wl_of_f_x(f), length=length, coupling_gap=coupling_gap, coupling_straight_length=coupling_straight_length, width=width, gap=gap)

    return {"straight": straight, "bend_s": straight, "launcher": launcher, "quarter_wave_resonator_coupled": resonator}


def wl_of_f_x(f):
    xp = _xp(f)
    return C0 / xp.asarray(f, dtype=float) * 1e6
