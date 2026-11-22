"""Sheet materials: a one-cell layer carrying a surface conductivity against
the analytic transmission and reflection of a zero-thickness sheet,
T = |2 / (2 + eta0 sigma)|^2 and R = |eta0 sigma / (2 + eta0 sigma)|^2."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import sheets

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
E2_4HBAR = sheets.Q_E**2 / (4 * sheets.HBAR)


def sheet_spectra(medium, f0, freqs, dx):
    """R and T of a free-standing layer one cell thick (normal incidence, Ey)."""
    lam0 = ls.C0 / f0
    nx = int(round(3 * lam0 / dx))
    length = nx * dx
    xs = round(0.55 * nx) * dx  # on a grid node: the layer owns one Ey plane

    def run(with_sheet):
        sim = ls.Simulation(min=[0, 0, 0], max=[length, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=60 / f0, shutoff=1e-7)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        if with_sheet:
            sim.add_medium(medium)
            sim.add_structure(ls.Box([xs - dx / 2, -1, -1], [xs + dx / 2, 1, 1]), medium["name"])
        sim.add_source(ls.PlaneWave(0, 0.35 * length, "+", "Ey", ls.GaussianPulse(f0, 0.25 * f0)))
        for name, x in (("R", 0.2 * length), ("T", 0.8 * length)):
            sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], freqs))
        return sim.run(device=DEVICE)

    empty, layer = run(False), run(True)
    incident = np.asarray(empty.flux("T"))
    return -np.asarray(layer.flux("R")) / incident, np.asarray(layer.flux("T")) / incident


def test_graphene_conductivity_limits():
    """Drude weight e^2 Ef / (pi hbar^2) at T = 0; the finite-temperature
    interband integral at 1 K against the closed T = 0 form away from the
    threshold; the universal e^2 / 4 hbar far above it."""
    ef = 0.2
    d0 = sheets.Q_E**2 * ef * sheets.Q_E / (np.pi * sheets.HBAR**2)
    assert abs(sheets._graphene_drude_weight(ef, 0.0) / d0 - 1) < 1e-12
    assert abs(sheets._graphene_drude_weight(ef, 1.0) / d0 - 1) < 1e-9
    f = np.array([10e12, 60e12, 150e12, 300e12])
    cold = sheets.graphene_conductivity(f, ef, 1e-13, temperature=1.0) - sheets.drude_sheet_conductivity(f, d0, 1e13)
    exact = sheets.graphene_conductivity(f, ef, 1e-13, temperature=0.0) - sheets.drude_sheet_conductivity(f, d0, 1e13)
    print("interband / (e^2/4hbar) at 1 K:", np.round(cold / E2_4HBAR, 5))
    assert np.max(np.abs(cold - exact)) / E2_4HBAR < 1e-4
    far = sheets.graphene_conductivity(3e15, ef, 1e-13, temperature=300.0, interband=True)
    assert abs(far.real / E2_4HBAR - 1) < 1e-3


def test_drude_sheet_one_cell_and_convergence():
    """A Drude sheet (graphene's intraband term, Ef = 0.2 eV, tau = 0.1 ps)
    over 2.25-3.75 THz is one exact Drude pole: at 40 cells per wavelength the
    one-cell layer is within 1e-3 of the sheet formulas, and the error falls
    as the second power of the cell size (the layer thickness)."""
    f0 = 3e12
    freqs = np.linspace(0.75, 1.25, 11) * f0
    d = sheets._graphene_drude_weight(0.2, 300.0)
    gamma = 1e13
    r_ex, t_ex = sheets.sheet_rt(freqs, sheets.drude_sheet_conductivity(freqs, d, gamma))
    errs = []
    for cells in (40, 80, 160):
        dx = ls.C0 / f0 / cells
        med = sheets.sheet_medium("drude_sheet", dx, drude_weight=d, gamma=gamma)
        r, t = sheet_spectra(med, f0, freqs, dx)
        errs.append(max(np.max(np.abs(t - t_ex)), np.max(np.abs(r - r_ex))))
        print(f"{cells} cells per wavelength (t = {dx * 1e6:.3f} um): max |T - T_sheet| = {np.max(np.abs(t - t_ex)):.2e}, "
              f"max |R - R_sheet| = {np.max(np.abs(r - r_ex)):.2e}  (T {t_ex.min():.3f}-{t_ex.max():.3f}, R {r_ex.min():.3f}-{r_ex.max():.3f})")
    assert errs[0] < 1e-3
    assert errs[0] / errs[1] > 3 and errs[1] / errs[2] > 3


def test_graphene_terahertz():
    """Graphene, Ef = 0.4 eV, tau = 0.1 ps, 300 K, 1.4-2.6 THz, Kubo
    conductivity with the interband term fitted: R and T against the sheet
    formulas with the Kubo sigma."""
    f0 = 2e12
    band = (0.7 * f0, 1.3 * f0)
    freqs = np.linspace(*band, 13)
    dx = ls.C0 / f0 / 40
    out = sheets.graphene(0.4, 1e-13, dx, 300.0, band=band, return_fit=True)
    r_ex, t_ex = sheets.sheet_rt(freqs, sheets.graphene_conductivity(freqs, 0.4, 1e-13, 300.0))
    r, t = sheet_spectra(out["medium"], f0, freqs, dx)
    print(f"sigma fit error {out['max_sigma_error']:.1e}; max |T - T_sheet| = {np.max(np.abs(t - t_ex)):.2e}, "
          f"max |R - R_sheet| = {np.max(np.abs(r - r_ex)):.2e}  (T {t_ex.min():.3f}-{t_ex.max():.3f}, R {r_ex.min():.3f}-{r_ex.max():.3f})")
    assert out["max_sigma_error"] < 1e-3
    assert np.max(np.abs(t - t_ex)) < 3e-3
    assert np.max(np.abs(r - r_ex)) < 2e-3


def test_graphene_mid_infrared_interband():
    """Graphene, Ef = 0.1 eV, tau = 0.1 ps, 300 K, 35-65 THz: the band crosses
    the interband threshold 2 Ef / h = 48 THz, where the fitted interband term
    dominates the absorption. Absorbance 1 - R - T and T against the Kubo
    sheet formulas."""
    f0 = 50e12
    band = (0.7 * f0, 1.3 * f0)
    freqs = np.linspace(*band, 13)
    dx = ls.C0 / f0 / 40
    out = sheets.graphene(0.1, 1e-13, dx, 300.0, band=band, n_lorentz=3, return_fit=True)
    sigma = sheets.graphene_conductivity(freqs, 0.1, 1e-13, 300.0)
    r_ex, t_ex = sheets.sheet_rt(freqs, sigma)
    r, t = sheet_spectra(out["medium"], f0, freqs, dx)
    a, a_ex = 1 - r - t, 1 - r_ex - t_ex
    err_a = np.max(np.abs(a - a_ex) / a_ex)
    print(f"sigma fit error {out['max_sigma_error']:.1e}; absorbance {a_ex.min():.4f}-{a_ex.max():.4f}, max relative error {err_a:.2e}; "
          f"max |T - T_sheet| = {np.max(np.abs(t - t_ex)):.2e}, max |R - R_sheet| = {np.max(np.abs(r - r_ex)):.2e}")
    assert out["max_sigma_error"] < 2e-3
    assert err_a < 0.02
    assert np.max(np.abs(t - t_ex)) < 5e-4


def test_general_sigma_fit_matches_graphene_helper():
    """sheet_medium with a callable conductivity fits the whole sigma (Drude +
    Lorentz poles); over a terahertz band it reproduces graphene's sigma."""
    band = (1.4e12, 2.6e12)
    out = sheets.sheet_medium("g", 1e-6, sigma=lambda f: sheets.graphene_conductivity(f, 0.4, 1e-13), band=band, return_fit=True)
    assert out["max_sigma_error"] < 1e-3
    f = np.linspace(*band, 7)
    s = sheets.sheet_layer_conductivity(out["medium"], f, 1e-6)
    assert np.max(np.abs(s / sheets.graphene_conductivity(f, 0.4, 1e-13) - 1)) < 1e-3
    with pytest.raises(ValueError):
        sheets.sheet_medium("g", 1e-6)
