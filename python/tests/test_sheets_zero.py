"""Zero-thickness sheets (``lightsim.Sheet``): R and T of graphene and of a
Drude sheet between two dielectrics against the analytic sheet formulas at
three mesh steps, with the sheet at a position that is not a grid node, and
the refusals (gradient engines, a sheet normal to the invariant axis of a
2-D simulation)."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import sheets
from lightsim.adjoint import check_gradient_media

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def sheet_rt_run(cond, f0, freqs, dx, n1, n2, xs_frac=0.5517):
    """R and T of a zero-thickness sheet at x = xs between n1 (incidence
    side) and a substrate n2 (normal incidence, Ey, periodic y and z)."""
    lam0 = ls.C0 / f0
    length = 3 * lam0
    xs = xs_frac * length  # not a grid node for any of the meshes used

    def run(full):
        sim = ls.Simulation(min=[0, 0, 0], max=[length, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=80 / f0, shutoff=1e-7)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        sim.add_medium(ls.Medium("n1", n=n1))
        sim.background = "n1"
        if full:
            sim.add_structure(ls.Box([xs, -1, -1], [2 * length, 1, 1]), ls.Medium("n2", n=n2))
            sim.add_sheet(ls.Sheet("x", xs, cond))
        sim.add_source(ls.PlaneWave(0, 0.3 * length, "+", "Ey", ls.GaussianPulse(f0, 0.25 * f0)))
        for name, x in (("R", 0.15 * length), ("T", 0.8 * length)):
            sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], freqs))
        return sim.run(device=DEVICE)

    ref, res = run(False), run(True)
    inc = np.asarray(ref.flux("T"))
    return -np.asarray(res.flux("R")) / inc, np.asarray(res.flux("T")) / inc


@pytest.mark.parametrize("kind", ["graphene", "drude"])
def test_zero_thickness_sheet_between_dielectrics(kind):
    """Graphene (Kubo, Ef = 0.4 eV, tau = 0.1 ps, 300 K, interband term
    fitted over the band) and a Drude sheet, between n1 = 1.45 and n2 = 1.95
    over 1.4-2.6 THz: the errors against the sheet formulas at 30, 60 and
    120 cells per wavelength (the sheet off the nodes; no thickness)."""
    f0 = 2e12
    freqs = np.linspace(0.7, 1.3, 7) * f0
    n1, n2 = 1.45, 1.95
    if kind == "graphene":
        fit = sheets.graphene(0.4, 1e-13, temperature=300, band=(0.7 * f0, 1.3 * f0), return_fit=True)
        cond = fit["conductivity"]
        assert fit["max_sigma_error"] < 1e-6
        sig = sheets.graphene_conductivity(freqs, 0.4, 1e-13, 300)
    else:
        cond = sheets.sheet_conductivity("drude", drude_weight=sheets._graphene_drude_weight(0.2, 300.0), gamma=1e13)
        sig = sheets.sigma_of(cond, freqs)
    r_ex, t_ex = sheets.sheet_rt(freqs, sig, n1, n2)
    errs = []
    for cells in (30, 60, 120):
        dx = ls.C0 / f0 / cells
        r, t = sheet_rt_run(cond, f0, freqs, dx, n1, n2)
        e = max(np.max(np.abs(r - r_ex)), np.max(np.abs(t - t_ex)))
        errs.append(e)
        print(f"{kind}, {cells} cells per wavelength: max |R - R_a| = {np.max(np.abs(r - r_ex)):.2e}, max |T - T_a| = {np.max(np.abs(t - t_ex)):.2e}"
              f"  (R {r_ex.min():.3f}-{r_ex.max():.3f}, T {t_ex.min():.3f}-{t_ex.max():.3f})")
    # second order: the mesh puts a node on the sheet
    assert errs[2] < 1e-3
    assert errs[0] / errs[2] > 6


def test_sheet_conductivity_fit_and_layer_route():
    """A fitted sheet conductivity (graphene's Kubo formula through the
    general ``sigma`` route) matches the dedicated helper, and the one-cell
    layer route still returns a medium."""
    band = (35e12, 65e12)
    s = sheets.sheet_conductivity("g", sigma=lambda f: sheets.graphene_conductivity(f, 0.1, 1e-13, 300), band=band, n_lorentz=3, return_fit=True)
    assert s["max_sigma_error"] < 2e-3
    f = np.linspace(*band, 9)
    ref = sheets.graphene_conductivity(f, 0.1, 1e-13, 300)
    assert np.max(np.abs(sheets.sigma_of(s["conductivity"], f) - ref) / np.abs(ref)) < 2e-3
    layer = sheets.graphene(0.4, 1e-13, 1e-6)
    assert "poles" in layer and "eps" in layer
    with pytest.raises(ValueError):
        sheets.graphene(0.4, 1e-13, eps_background=2.0)


def test_sheet_refusals():
    """Gradient engines refuse sheets; a sheet normal to the invariant axis
    of a 2-D simulation, a polygon with two vertices and an active sheet
    (negative conductance) are refused by the core."""
    cond = sheets.sheet_conductivity("c", conductance=1e-3)
    sim = ls.Simulation(min=[0, 0, 0], max=[2e-6, 0.1e-6, 0.1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim.add_sheet(ls.Sheet("x", 1e-6, cond, name="s"))
    with pytest.raises(ValueError, match="sheet 's'"):
        check_gradient_media(sim)
    sim2 = ls.Simulation(min=[0, 0], max=[2e-6, 1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim2.add_sheet(ls.Sheet("z", 0.0, cond))
    with pytest.raises(Exception, match="invariant axis"):
        sim2.time_info()
    sim3 = ls.Simulation(min=[0, 0, 0], max=[2e-6, 0.1e-6, 0.1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim3.add_sheet(ls.Sheet("x", 1e-6, cond, polygon=[(0, 0), (1, 1)]))
    with pytest.raises(Exception, match="three vertices"):
        sim3.time_info()
    sim4 = ls.Simulation(min=[0, 0, 0], max=[2e-6, 0.1e-6, 0.1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim4.add_sheet(ls.Sheet("x", 1e-6, sheets.SheetConductivity(conductance=-1e-3)))
    with pytest.raises(Exception, match="passive"):
        sim4.time_info()


def test_sheet_patch_full_cell_equals_plane():
    """A rectangular patch that covers the whole periodic cell gives the
    result of the full plane (same cells, coverage 1)."""
    f0 = 2e12
    dx = ls.C0 / f0 / 30
    cond = sheets.sheet_conductivity("d", drude_weight=4e10, gamma=1e13)

    def run(patch):
        sim = ls.Simulation(min=[0, 0, 0], max=[90 * dx, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=40 / f0)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        kw = dict(min=[0, -1, -1], max=[0, 1, 1]) if patch else {}
        sim.add_sheet(ls.Sheet("x", 50.3 * dx, cond, **kw))
        sim.add_source(ls.PlaneWave(0, 20 * dx, "+", "Ey", ls.GaussianPulse(f0, 0.25 * f0)))
        sim.add_monitor(ls.FluxMonitor("T", 0, 70 * dx, [70 * dx, -1, -1], [70 * dx, 1, 1], [f0]))
        return sim.run(device=DEVICE).flux("T")[0]

    assert run(True) == run(False)
