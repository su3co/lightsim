"""Lossy-core waveguide: complex n_eff from the mode solver and matching FDTD
decay (cf. FDTDX test_mode_source lossy-core tests)."""

import math
import os

import numpy as np

import lightsim as ls
from lightsim.spec import ModeMonitor, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
EPS0 = 8.8541878128e-12


def build(sigma):
    lam = 1.55 * um
    f0 = ls.C0 / lam
    sim = ls.Simulation(min=[0, -1.5 * um, -1.0 * um], max=[5.0 * um, 1.5 * um, 1.0 * um], mesh=ls.UniformMesh(40e-9), run_time=250e-15, background="ox")
    sim.add_medium(ls.Medium("ox", n=1.444))
    sim.add_medium(ls.Medium("core", n=2.0, sigma=sigma))
    sim.add_structure(ls.Box([-1, -0.4 * um, -0.2 * um], [5 * um, 0.4 * um, 0.2 * um]), "core")
    return sim, f0


def test_lossy_core_complex_neff_and_decay():
    lam = 1.55 * um
    w = 2 * math.pi * ls.C0 / lam
    sigma = 0.25 * w * EPS0  # Im(eps_core) = 0.25
    bmin, bmax = [0.6 * um, -1.2 * um, -0.8 * um], [0.6 * um, 1.2 * um, 0.8 * um]
    sim0, f0 = build(0.0)
    m0 = ls.solve_modes(sim0, 0, 0.6 * um, bmin, bmax, f0, num_modes=1)[0]
    sim, _ = build(sigma)
    m = ls.solve_modes(sim, 0, 0.6 * um, bmin, bmax, f0, num_modes=1)[0]
    # premise: the loss makes n_eff complex; the lossless guide stays real
    assert m.neff.imag > 1e-2, m.neff
    assert abs(m0.neff.imag) < 1e-6, m0.neff
    sim.add_source(ls.ModeSource(0, 0.6 * um, "+", bmin, bmax, m.profile, ls.GaussianPulse(f0, 0.1 * f0), name="src"))
    xs = [2.0 * um, 4.0 * um]
    for i, x in enumerate(xs):
        sim.add_monitor(ModeMonitor(f"m{i}", 0, x, [x, bmin[1], bmin[2]], [x, bmax[1], bmax[2]], [f0], [m.profile]))
    r = sim.run(device=DEVICE)
    a0 = r.mode_amplitudes("m0")[0][0]
    a1 = r.mode_amplitudes("m1")[0][0]
    g = sim.grid()[0]
    dx_true = g[np.argmin(abs(g - xs[1]))] - g[np.argmin(abs(g - xs[0]))]
    alpha_fdtd = -math.log(abs(a1) / abs(a0)) / dx_true
    alpha_mode = 2 * math.pi / lam * m.neff.imag
    print(f"neff={m.neff:.5f} lossless={m0.neff.real:.5f} alpha_fdtd/alpha_mode={alpha_fdtd / alpha_mode:.4f}")
    assert abs(alpha_fdtd / alpha_mode - 1) < 0.05, (alpha_fdtd, alpha_mode)
    # the phase advance follows Re(n_eff)
    k_fdtd = (np.angle(a1 / a0)) % (2 * math.pi)
    k_mode = (2 * math.pi / lam * m.neff.real * dx_true) % (2 * math.pi)
    dphi = (k_fdtd - k_mode + math.pi) % (2 * math.pi) - math.pi
    print(f"phase error {dphi:.4f} rad")
    assert abs(dphi) < 0.08, (k_fdtd, k_mode)  # Yee dispersion at 24 cells per wavelength
