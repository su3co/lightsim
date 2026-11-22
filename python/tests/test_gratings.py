"""Diffraction-order analysis (lightsim.gratings) against analytic results and
a reference 1-D RCWA (Moharam, Grann, Pommet & Gaylord, JOSA A 12, 1068 (1995)).

Run with `pytest python/tests/test_gratings.py` (CPU unless LIGHTSIM_DEVICE is set).
"""

import math
import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import gratings as gr
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")

LAM = 1 * um
F0 = ls.C0 / LAM
K0 = 2 * math.pi / LAM
PERIOD = 1.5 * LAM
DEPTH = 0.5 * LAM
FILL = 0.5
N_RIDGE = 1.5
N_SUB = 1.5

X_SRC, X_R, X_TOP, X_T, X_MAX = 1.0 * um, 0.5 * um, 2.0 * um, 3.5 * um, 4.5 * um
TAN = ("Ey", "Ez", "Hy", "Hz")


# ----------------------------------------------------------------------------
# reference RCWA: lamellar grating, TE (E along the grooves), one layer
# ----------------------------------------------------------------------------


def rcwa_te(lam, period, depth, fill, n_ridge, n_groove, n_inc, n_sub, theta=0.0, n_harm=41):
    """Diffraction efficiencies {m: R_m}, {m: T_m} of the propagating orders.

    Order m has in-plane wavevector k0 n_inc sin(theta) + 2 pi m / period.
    """
    k0 = 2 * math.pi / lam
    half = n_harm // 2
    i = np.arange(-half, half + 1)
    n = i.size
    kx = n_inc * math.sin(theta) - i * lam / period  # Moharam's sign: order i <-> m = -i
    def kz(nm):
        v = np.sqrt((nm**2 - kx**2).astype(complex))
        return np.where(v.imag > 0, -v, v)  # exp(+jwt): decaying = negative imaginary part
    y1, y2 = kz(n_inc), kz(n_sub)
    # Toeplitz matrix of the permittivity harmonics (ridge centred in the period)
    h = i[:, None] - i[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_h = np.where(h == 0, n_ridge**2 * fill + n_groove**2 * (1 - fill), (n_ridge**2 - n_groove**2) * np.sin(math.pi * h * fill) / (math.pi * h))
    a = np.diag(kx**2) - eps_h
    q2, w = np.linalg.eig(a)
    q = np.sqrt(q2.astype(complex))
    q = np.where(q.real < 0, -q, q)
    v = w * q[None, :]
    x = np.diag(np.exp(-k0 * q * depth))
    eye = np.eye(n)
    d0 = (i == 0).astype(complex)
    z = np.zeros((n, n))
    # unknowns [R, c+, c-, T]
    m = np.block(
        [
            [-eye, w, w @ x, z],
            [1j * np.diag(y1), v, -v @ x, z],
            [z, w @ x, w, -eye],
            [z, v @ x, -v, -1j * np.diag(y2)],
        ]
    )
    rhs = np.concatenate([d0, 1j * n_inc * math.cos(theta) * d0, np.zeros(2 * n)])
    sol = np.linalg.solve(m, rhs)
    r, t = sol[:n], sol[3 * n :]
    de_r = np.abs(r) ** 2 * (y1 / (n_inc * math.cos(theta))).real
    de_t = np.abs(t) ** 2 * (y2 / (n_inc * math.cos(theta))).real
    rr = {int(-ii): float(e) for ii, e, k in zip(i, de_r, y1) if abs(k.imag) < 1e-12 and k.real > 0}
    tt = {int(-ii): float(e) for ii, e, k in zip(i, de_t, y2) if abs(k.imag) < 1e-12 and k.real > 0}
    return rr, tt


def test_rcwa_reference_sanity():
    # fill = 1 on an index-matched substrate: a single Fresnel interface
    for theta in (0.0, math.radians(30)):
        r, t = rcwa_te(LAM, PERIOD, DEPTH, 1.0, 1.5, 1.0, 1.0, 1.5, theta)
        ct = math.cos(theta)
        c2 = math.sqrt(1.5**2 - math.sin(theta) ** 2)
        fres = ((ct - c2) / (ct + c2)) ** 2
        assert abs(r[0] - fres) < 1e-10 and abs(t[0] - (1 - fres)) < 1e-10
        assert sum(v for m, v in r.items() if m) < 1e-12
    # lossless grating: energy conservation and truncation convergence
    r, t = rcwa_te(LAM, PERIOD, DEPTH, FILL, N_RIDGE, 1.0, 1.0, N_SUB)
    assert sorted(r) == [-1, 0, 1] and sorted(t) == [-2, -1, 0, 1, 2]
    assert abs(sum(r.values()) + sum(t.values()) - 1) < 1e-10
    r2, t2 = rcwa_te(LAM, PERIOD, DEPTH, FILL, N_RIDGE, 1.0, 1.0, N_SUB, n_harm=81)
    assert max(abs(t[m] - t2[m]) for m in t) < 2e-4 and max(abs(r[m] - r2[m]) for m in r) < 2e-4


# ----------------------------------------------------------------------------
# FDTD unit cell: x = propagation, y = grating period, z = invariant
# ----------------------------------------------------------------------------


def cell(grating, pol="Ez", theta=0.0, dx=LAM / 40, run_time=120e-15, fwidth=0.15, wz=None, freqs=(F0,)):
    ky = K0 * math.sin(theta)
    wz = 2 * dx if wz is None else wz
    sim = ls.Simulation(min=[0, 0, 0], max=[X_MAX, PERIOD, wz], mesh=ls.UniformMesh(dx), run_time=run_time)
    sim.set_boundaries(y=ls.Bloch(ky) if theta else ls.PERIODIC, z=ls.PERIODIC)
    if grating:
        sim.add_structure(ls.Box([X_TOP, 0.5 * (1 - FILL) * PERIOD, -1], [X_TOP + DEPTH, 0.5 * (1 + FILL) * PERIOD, 1]), ls.Medium("ridge", n=N_RIDGE))
        sim.add_structure(ls.Box([X_TOP + DEPTH, -1, -1], [1, 1, 1]), ls.Medium("sub", n=N_SUB))
    sim.add_source(ls.PlaneWave(0, X_SRC, "+", pol, ls.GaussianPulse(F0, fwidth * F0), angle=theta, name="pw"))
    for name, x in (("t", X_T), ("r", X_R)):
        sim.add_monitor(ls.DFTMonitor(name, [x, 0, 0], [x, PERIOD, wz], freqs, TAN))
        sim.add_monitor(ls.FluxMonitor(name.upper(), 0, x, [x, -1, -1], [x, 1, 1], freqs))
    return sim, ky


@pytest.mark.parametrize("pol", ["Ez", "Ey"])
def test_no_grating_normal_incidence(pol):
    sim, _ = cell(False, pol=pol, run_time=60e-15)
    res = sim.run(device=DEVICE)
    o = gr.diffraction_orders(res, "t", 0, 1, [F0], 1.0, sim=sim)
    assert o.orders[:, 0].tolist() == [-1, 0, 1] and o.propagating.all()
    i0 = o.index(0)
    flux = res.flux("T")[0]
    assert abs(o.fraction[0, i0] - 1) < 1e-6
    assert abs(o.theta[0, i0]) < 1e-12
    np.testing.assert_allclose(np.abs(o.angles[0, [0, 2], 0]), math.asin(LAM / PERIOD), rtol=1e-12)
    assert abs(o.total[0] / flux - 1) < 1e-3
    assert o.total_backward[0] / flux < 1e-6
    # polarisation split: E along z is s (E normal to the x-y plane), E along y is p
    s, p = o.power_s_forward[0, i0] / flux, o.power_p_forward[0, i0] / flux
    assert (s > 0.999 and p < 1e-6) if pol == "Ez" else (p > 0.999 and s < 1e-6)
    # nothing behind the one-sided source
    r = gr.diffraction_orders(res, "r", 0, 1, [F0], 1.0, sim=sim)
    assert (r.total_forward[0] + r.total_backward[0]) / flux < 1e-6


def test_no_grating_oblique_incidence():
    theta = math.radians(30)
    f2 = 1.04 * F0
    sim, ky = cell(False, theta=theta, run_time=150e-15, fwidth=0.06, freqs=(F0, f2))
    res = sim.run(device=DEVICE)
    o = gr.diffraction_orders(res, "t", 0, 1, None, 1.0, ky, sim=sim)
    i0 = o.index(0)
    flux = np.asarray(res.flux("T"))
    # all power in order 0, at the launch angle (fixed Bloch k: sin(theta) scales as 1/f)
    assert np.all(np.abs(o.fraction[:, i0] - 1) < 1e-6)
    assert abs(o.angles[0, i0, 0] - theta) < 1e-9 and abs(o.theta[0, i0] - theta) < 1e-9
    assert abs(o.angles[1, i0, 0] - math.asin(math.sin(theta) * F0 / f2)) < 1e-9
    np.testing.assert_allclose(o.total, flux, rtol=1e-3)
    assert np.all(o.total_backward / flux < 1e-6)
    # the other propagating orders follow the grating equation
    ords, ang = gr.grating_angles(F0, PERIOD, 1.0, ky)
    assert ords.tolist() == o.orders[o.propagating[0], 0].tolist() == [-2, -1, 0]
    np.testing.assert_allclose(o.angles[0, o.propagating[0], 0], ang, atol=1e-12)
    for m in (-2, -1):
        assert abs(math.sin(ang[ords == m][0]) - (math.sin(theta) + m * LAM / PERIOD)) < 1e-12
    # the analysis with the wrong Bloch vector does not reproduce the flux
    bad = gr.diffraction_orders(res, "t", 0, 1, [F0], 1.0, -ky, sim=sim)
    assert abs(bad.total[0] / flux[0] - 1) > 0.05


def test_two_dimensional_orders_vacuum():
    # both in-plane axes periodic: orders (m, n); a plane wave is order (0, 0)
    sim, _ = cell(False, dx=LAM / 20, run_time=60e-15, wz=PERIOD)
    res = sim.run(device=DEVICE)
    o = gr.diffraction_orders(res, "t", 0, (1, 2), [F0], 1.0, sim=sim)
    assert len(o.orders) == 9  # |m|, |n| <= 1 all propagate for a 1.5 lambda square cell
    flux = res.flux("T")[0]
    assert abs(o.efficiency(0, 0)[0] - 1) < 1e-6 and abs(o.total[0] / flux - 1) < 1e-3
    j = o.index(1, -1)
    assert abs(o.theta[0, j] - math.asin(math.sqrt(2) * LAM / PERIOD)) < 1e-12
    assert abs(o.phi[0, j] + math.pi / 4) < 1e-12
    assert abs(o.area - PERIOD * PERIOD) < 1e-9 * PERIOD**2


# ----------------------------------------------------------------------------
# binary lamellar grating on a substrate vs RCWA
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_power():
    out = {}
    for pol in ("Ez", "Ey"):
        sim, _ = cell(False, pol=pol, run_time=60e-15)
        out[pol] = gr.incident_power(sim.run(device=DEVICE), "t", 0, 1, [F0], 1.0, sim=sim)
    return out


def test_lamellar_grating_te_vs_rcwa(reference_power):
    sim, _ = cell(True, pol="Ez", run_time=250e-15)
    res = sim.run(device=DEVICE)
    eff = gr.grating_efficiencies(res, "t", "r", 0, 1, [F0], n_t=N_SUB, n_r=1.0, sim=sim, incident_power=reference_power["Ez"])
    t, r = eff["T"], eff["R"]
    r_ref, t_ref = rcwa_te(LAM, PERIOD, DEPTH, FILL, N_RIDGE, 1.0, 1.0, N_SUB)
    assert sorted(t.as_dict()) == [(m, 0) for m in sorted(t_ref)]
    assert sorted(r.as_dict()) == [(m, 0) for m in sorted(r_ref)]
    report = []
    for name, got, ref in (("T", t, t_ref), ("R", r, r_ref)):
        for m in sorted(ref):
            g = got.efficiency(m)[0]
            report.append(f"{name}{m:+d}: fdtd {g:.4f} rcwa {ref[m]:.4f}")
            assert abs(g - ref[m]) < 0.01, report
    print("\n".join(report))
    # angles: grating equation in the substrate and in air
    for m in t_ref:
        assert abs(math.sin(t.angles[0, t.index(m), 0]) - m * LAM / (N_SUB * PERIOD)) < 1e-12
    for m in r_ref:
        assert abs(math.sin(r.angles[0, r.index(m), 0]) - m * LAM / PERIOD) < 1e-12
    # TE in, TE out; energy conservation; consistency with the flux monitors
    assert t.power_p_forward.sum() / t.power_s_forward.sum() < 1e-6
    total = eff["T_total"][0] + eff["R_total"][0]
    print(f"T {eff['T_total'][0]:.4f} R {eff['R_total'][0]:.4f} sum {total:.5f}")
    assert abs(total - 1) < 0.01
    assert abs(t.total[0] / res.flux("T")[0] - 1) < 1e-3
    assert abs(-r.total[0] / -res.flux("R")[0] - 1) < 1e-3


def test_lamellar_grating_tm_energy_conservation(reference_power):
    # H along the grooves (p polarisation): orders stay p polarised and power is conserved
    sim, _ = cell(True, pol="Ey", run_time=250e-15)
    res = sim.run(device=DEVICE)
    eff = gr.grating_efficiencies(res, "t", "r", 0, 1, [F0], n_t=N_SUB, n_r=1.0, sim=sim, incident_power=reference_power["Ey"])
    t = eff["T"]
    assert t.power_s_forward.sum() / t.power_p_forward.sum() < 1e-6
    total = eff["T_total"][0] + eff["R_total"][0]
    print(f"TM: T {eff['T_total'][0]:.4f} R {eff['R_total'][0]:.4f} sum {total:.5f}", t.as_dict(), eff["R"].as_dict())
    assert abs(total - 1) < 0.01
    assert abs(t.efficiency(1)[0] - t.efficiency(-1)[0]) < 1e-3  # mirror symmetry
    assert abs(t.total[0] / res.flux("T")[0] - 1) < 1e-3


def test_input_validation():
    sim, _ = cell(False, run_time=2e-15)
    sim.add_monitor(ls.DFTMonitor("half", [X_T, 0, 0], [X_T, PERIOD / 2, 1], [F0], TAN))
    sim.add_monitor(ls.DFTMonitor("eonly", [X_T, 0, 0], [X_T, PERIOD, 1], [F0], ("Ey", "Ez")))
    res = sim.run(device=DEVICE)
    with pytest.raises(ValueError, match="one period"):
        gr.diffraction_orders(res, "half", 0, 1, [F0], 1.0, sim=sim)
    with pytest.raises(KeyError, match="tangential"):
        gr.diffraction_orders(res, "eonly", 0, 1, [F0], 1.0, sim=sim)
    with pytest.raises(ValueError, match="period_axes"):
        gr.diffraction_orders(res, "t", 0, 0, [F0], 1.0, sim=sim)
    with pytest.raises(ValueError, match="not recorded"):
        gr.diffraction_orders(res, "t", 0, 1, [1.3 * F0], 1.0, sim=sim)
    o = gr.diffraction_orders(res, "t", 0, 1, [F0], 1.0, sim=sim, max_order=3)
    assert o.orders[:, 0].tolist() == [-3, -2, -1, 0, 1, 2, 3]
    assert o.propagating[0].tolist() == [False, False, True, True, True, False, False]
    assert np.all(o.power_forward[0, ~o.propagating[0]] == 0) and np.isnan(o.theta[0, 0])
