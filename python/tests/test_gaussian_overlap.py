"""Gaussian-beam overlap against analytic coupling (cf. FDTDX test_mode_overlap)."""

import os

import numpy as np

import lightsim as ls
from lightsim.spec import GaussianBeam, ModeMonitor, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def test_gaussian_overlap_follows_analytic_coupling():
    lam, w0 = 1.0 * um, 1.0 * um
    f0 = ls.C0 / lam
    sim = ls.Simulation(min=[-2.5 * um, -2.5 * um, 0], max=[2.5 * um, 2.5 * um, 2.0 * um], mesh=ls.UniformMesh(50e-9), run_time=60e-15)
    sim.add_source(GaussianBeam(2, 0.5 * um, "+", "Ex", [0, 0, 0.5 * um], w0, ls.GaussianPulse(f0, 0.1 * f0), name="gb"))
    zp, half = 1.0 * um, 2.0 * um
    bmin, bmax = [-half, -half, zp], [half, half, zp]
    matched, pm = ls.gaussian_mode(sim, 2, zp, bmin, bmax, f0, w0, center=[0, 0, zp], polarization="u", focus_distance=-0.5 * um)
    narrow, pn = ls.gaussian_mode(sim, 2, zp, bmin, bmax, f0, 0.6 * w0, center=[0, 0, zp], polarization="u", focus_distance=-0.5 * um)
    wide, pw = ls.gaussian_mode(sim, 2, zp, bmin, bmax, f0, 1.5 * w0, center=[0, 0, zp], polarization="u", focus_distance=-0.5 * um)
    for nm, prof in [("m", matched), ("n", narrow), ("w", wide)]:
        sim.add_monitor(ModeMonitor(nm, 2, zp, bmin, bmax, [f0], [prof]))
    r = sim.run(device=DEVICE)
    flux = r.raw["modes"]["m"]["flux"][0]
    ap, am = r.mode_amplitudes("m")
    eta_m = abs(ap[0]) ** 2 * pm / flux
    eta_n = abs(r.mode_amplitudes("n")[0][0]) ** 2 * pn / flux
    eta_w = abs(r.mode_amplitudes("w")[0][0]) ** 2 * pw / flux
    two_gauss = lambda a, b: (2 * a * b / (a * a + b * b)) ** 2  # noqa: E731
    # the launched beam is the matched Gaussian; it carries no backward power
    assert eta_m > 0.97, eta_m
    assert abs(am[0]) ** 2 * pm / flux < 1e-3
    # waist-mismatched references follow the two-Gaussian overlap formula
    assert abs(eta_n / eta_m - two_gauss(w0, 0.6 * w0)) < 0.02, (eta_n / eta_m, two_gauss(w0, 0.6 * w0))
    assert abs(eta_w / eta_m - two_gauss(w0, 1.5 * w0)) < 0.02, (eta_w / eta_m, two_gauss(w0, 1.5 * w0))
    assert eta_m > max(eta_n, eta_w)


def test_tilted_beam_is_collected_by_the_tilted_reference_beam():
    """A sheet beam launched at 25 degrees couples fully into the reference
    beam of the same direction, waist and focus, and not at all into the
    beam along the normal; for E out of the plane and E in the plane."""
    import pytest

    lam = 1.0 * um
    f0 = ls.C0 / lam
    h = lam / 20
    w0 = 2.5 * um
    theta = np.radians(25.0)
    xs, yc, xm = 2 * um, -1 * um, 9 * um
    for pol_source, pol_mode in (("Ez", "v"), ("Ey", "u")):
        sim = ls.Simulation(min=[0, -9 * um, 0], max=[12 * um, 9 * um, h], mesh=ls.UniformMesh(h), run_time=160e-15, shutoff=1e-8)
        sim.set_boundaries(z=ls.PERIODIC)
        sim.add_source(ls.GaussianBeam(0, xs, "+", pol_source, [xs, yc, 0], w0, ls.GaussianPulse(f0, 0.08 * f0), focus_distance=3 * um, theta=theta))
        x = np.asarray(sim.grid()[0])
        travel = x[np.argmin(np.abs(x - xm))] - x[np.argmin(np.abs(x - xs))]
        centre = [xm, yc + travel * np.tan(theta), 0]
        wmin, wmax = [xm, -8.5 * um, 0], [xm, 8.5 * um, h]
        matched, p_matched = ls.gaussian_mode(sim, 0, xm, wmin, wmax, f0, w0, center=centre, polarization=pol_mode, focus_distance=3 * um - travel / np.cos(theta), theta=theta)
        normal, p_normal = ls.gaussian_mode(sim, 0, xm, wmin, wmax, f0, w0, center=centre, polarization=pol_mode)
        sim.add_monitor(ls.ModeMonitor("matched", 0, xm, wmin, wmax, [f0], [matched]))
        sim.add_monitor(ls.ModeMonitor("normal", 0, xm, wmin, wmax, [f0], [normal]))
        sim.add_monitor(ls.FluxMonitor("P", 0, xm, [xm, -20 * um, -1], [xm, 20 * um, 1], [f0]))
        res = sim.run(device=DEVICE)
        power = res.flux("P")[0]
        a_plus, a_minus = res.mode_amplitudes("matched")
        eta = abs(a_plus[0]) ** 2 * p_matched / power
        back = abs(a_minus[0]) ** 2 * p_matched / power
        eta_normal = abs(res.mode_amplitudes("normal")[0][0]) ** 2 * p_normal / power
        print(f"{pol_source}: matched {eta:.4f}, backward {back:.1e}, beam along the normal {eta_normal:.1e}")
        assert abs(eta - 1) < 0.01 and back < 1e-3 and eta_normal < 1e-3
        # the power of the reference beam is that of the formula for a sheet beam
        assert p_matched == pytest.approx(w0 * np.sqrt(np.pi / 2) * h / (2 * 376.730313412), rel=2e-3)
