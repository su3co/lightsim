"""Broadband oblique incidence at a fixed angle in one run
(``PlaneWave(..., fixed_angle=True)``): a dielectric slab against the Airy
formula across +-30 %, the diffraction efficiencies of a lamellar grating
at several frequencies against RCWA from one run, the JSON form, and the
refusals (Bloch boundaries along the tilt axis, the gradient engines)."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import gratings as gr
from lightsim.adjoint import check_gradient_media
from lightsim.spec import um
from rcwa_reference import efficiencies

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")

LAM0 = 1.0 * um
F0 = ls.C0 / LAM0


def slab_airy(freqs, theta, n, d, pol):
    k0 = 2 * np.pi * np.asarray(freqs) / ls.C0
    c1 = np.cos(theta)
    c2 = np.sqrt(1 - (np.sin(theta) / n) ** 2)
    r12 = (c1 - n * c2) / (c1 + n * c2) if pol == "s" else (n * c1 - c2) / (n * c1 + c2)
    e = np.exp(2j * k0 * n * d * c2)
    r = r12 * (1 - e) / (1 - r12**2 * e)
    return np.abs(r) ** 2


@pytest.mark.parametrize("pol, tol", [("s", 0.01), ("p", 3e-3)])
def test_slab_spectrum_at_45_degrees_from_one_run(pol, tol):
    """The slab of test_sweep_angle (n = 1.5, 2 um, 60 cells per vacuum
    wavelength) at 45 degrees: R and T over +-30 % from one run (and one
    empty reference run) against the Airy formula at 45 degrees, and the
    incident power against n |S|^2 cos(theta) area / (2 eta0)."""
    dx = LAM0 / 60
    ly, lz = 8 * dx, 2 * dx
    theta = np.radians(45.0)
    freqs = np.linspace(0.7, 1.3, 13) * F0
    n_slab, d_slab = 1.5, 120 * dx

    def run(slab):
        sim = ls.Simulation(min=[0, 0, 0], max=[7 * um, ly, lz], mesh=ls.UniformMesh(dx), run_time=300e-15, shutoff=1e-6)
        sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
        if slab:
            sim.add_structure(ls.Box([3 * um, -1, -1], [3 * um + d_slab, 1, 1]), ls.Medium("glass", n=n_slab))
        sim.add_source(ls.PlaneWave(0, 1.5 * um, "+", "Ez" if pol == "s" else "Hz", ls.GaussianPulse(F0, 0.25 * F0), angle=theta, name="pw", fixed_angle=True))
        for name, x in (("R", 1.0 * um), ("T", 6.0 * um)):
            sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], freqs))
        return sim.run(device=DEVICE)

    ref, res = run(False), run(True)
    incident = ref.flux("T")
    r, t = -res.flux("R") / incident, res.flux("T") / incident
    r_ex = slab_airy(freqs, theta, n_slab, d_slab, pol)
    _, spec = ref.source_spectrum("pw", monitor="T")
    p_formula = np.abs(spec) ** 2 * np.cos(theta) * ly * lz / (2 * 376.730313412)
    print(f"{pol}: max |R - R_Airy| = {np.max(np.abs(r - r_ex)):.2e}, max |T - T_Airy| = {np.max(np.abs(t - (1 - r_ex))):.2e}, "
          f"incident / formula {np.min(incident / p_formula):.4f}-{np.max(incident / p_formula):.4f}; {res.stats['steps']} steps")
    assert "fixed angle" in res.stats["backend"]
    assert np.max(np.abs(incident / p_formula - 1)) < 0.01
    assert np.max(np.abs(r - r_ex)) < tol
    assert np.max(np.abs(t - (1 - r_ex))) < tol


@pytest.mark.parametrize("pol", ["Ez", "Hz"])
def test_grating_orders_at_fixed_angle_against_rcwa(pol):
    """A lamellar grating (period 1.5 um, n = 1.5 ridges of half the period,
    0.5 um deep, on n = 1.5) at 20 degrees in 2-D: the efficiencies of every
    propagating order at 0.85, 0.95 and 1.15 f0 from one fixed-angle run
    (gratings with one k_inc per frequency) against RCWA at 20 degrees. (At
    f0, 1.3 % below the Rayleigh anomaly of the reflected order +1, the
    40-cell grid is 1.4e-2 off RCWA, as a narrow-band Bloch run is: the
    validation case fixed_angle_grating finds the two within 2e-4; at 0.95 f0
    the TE error falls from 1.2e-2 at 40 to 6.9e-3 at 60 cells per vacuum
    wavelength.)"""
    period, depth, n2 = 1.5 * um, 0.5 * um, 1.5
    theta = np.radians(20.0)
    dx = LAM0 / 50
    freqs = np.array([0.85, 0.95, 1.15]) * F0
    kx = 2 * np.pi * freqs / ls.C0 * np.sin(theta)
    tan = ("Ey", "Ez", "Hy", "Hz")

    def run(grating):
        sim = ls.Simulation(min=[0, 0], max=[4.5 * um, period], mesh=ls.UniformMesh(dx), run_time=800e-15, shutoff=1e-6)
        sim.set_boundaries(x=ls.PML(), y=ls.PERIODIC)
        if grating:
            sim.add_structure(ls.Box([2.0 * um, 0.25 * period], [2.0 * um + depth, 0.75 * period]), ls.Medium("ridge", n=n2))
            sim.add_structure(ls.Box([2.0 * um + depth, -1], [10 * um, 1]), ls.Medium("sub", n=n2))
        sim.add_source(ls.PlaneWave(0, 1.0 * um, "+", pol, ls.GaussianPulse(F0, 0.25 * F0), angle=theta, name="pw", fixed_angle=True))
        for name, x in (("t", 3.5 * um), ("r", 0.5 * um)):
            sim.add_monitor(ls.DFTMonitor(name, [x, 0], [x, period], freqs, tan))
        return sim, sim.run(device=DEVICE)

    sref, ref = run(False)
    p_inc = gr.incident_power(ref, "t", 0, 1, n_medium=1.0, kx_inc=kx, sim=sref)
    sim, res = run(True)
    eff = gr.grating_efficiencies(res, "t", "r", 0, 1, n_t=n2, n_r=1.0, kx_inc=kx, sim=sim, incident_power=p_inc)
    t, r = eff["T"], eff["R"]
    worst = 0.0
    for fi, f in enumerate(freqs):
        orders, r_ref, t_ref = efficiencies(ls.C0 / f, period, depth, 0.5, 1.0, n2, n2, 1.0, theta, pol == "Ez")
        for m, rr, tt in zip(orders, r_ref, t_ref):
            if tt > 0:
                worst = max(worst, abs(t.efficiency(int(m))[fi] - tt))
                print(f"  {f / F0:.2f} f0 T{int(m):+d}: {t.efficiency(int(m))[fi]:.4f} (RCWA {tt:.4f})")
            if rr > 0:
                worst = max(worst, abs(r.efficiency(int(m))[fi] - rr))
                print(f"  {f / F0:.2f} f0 R{int(m):+d}: {r.efficiency(int(m))[fi]:.4f} (RCWA {rr:.4f})")
        total = eff["T_total"][fi] + eff["R_total"][fi]
        assert abs(total - 1) < 0.01, total
    print(f"{pol}: max |efficiency - RCWA| over the orders at 0.85, 0.95, 1.15 f0: {worst:.2e}")
    assert worst < 0.01


def test_json_form_and_refusals():
    sim = ls.Simulation(min=[0, 0], max=[2 * um, 0.4 * um], mesh=ls.UniformMesh(25e-9), run_time=50e-15)
    sim.set_boundaries(x=ls.PML(), y=ls.PERIODIC)
    sim.add_source(ls.PlaneWave(0, 0.5 * um, "+", "Ez", ls.GaussianPulse(F0, 0.2 * F0), angle=np.radians(30), fixed_angle=True))
    sim.add_monitor(ls.FluxMonitor("T", 0, 1.5 * um, [1.5 * um, -1], [1.5 * um, 1], [F0]))
    d = sim.to_dict()
    assert d["fixed_angle"] is True and all("fixed_angle" not in s for s in d["sources"])
    with pytest.raises(ValueError, match="fixed-angle"):
        check_gradient_media(sim)
    sim.set_boundaries(y=ls.Bloch(1e6))
    with pytest.raises(RuntimeError, match="Bloch"):
        sim.run(device="cpu")
