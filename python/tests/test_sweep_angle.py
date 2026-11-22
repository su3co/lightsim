"""Fixed-angle spectra from a sweep of oblique runs (lightsim.sweep_angle):
reflection and transmission of a dielectric slab at 45 degrees across a
+-15 % band against the Airy formula at constant angle, for s and p
polarisation, and the broadband fixed-k run for comparison, whose angle
drifts with frequency."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
# devices for the concurrent-dispatch test, separated by spaces ("cuda:0 cuda:1")
SWEEP_DEVICES = os.environ.get("LIGHTSIM_SWEEP_DEVICES", f"{DEVICE} {DEVICE}").split()

LAM0 = 1.0 * um
F0 = ls.C0 / LAM0
DX = LAM0 / 60
LY, LZ = 8 * DX, 2 * DX
N_SLAB = 1.5
D_SLAB = 120 * DX  # 2 um
X_SLAB = 3.0 * um
THETA = np.radians(45.0)
FREQS = np.linspace(0.85, 1.15, 9) * F0


def airy(freqs, theta, pol):
    """R, T of a slab (index N_SLAB, thickness D_SLAB) in vacuum at angle theta."""
    k0 = 2 * np.pi * np.asarray(freqs) / ls.C0
    c1 = np.cos(theta)
    c2 = np.sqrt(1 - (np.sin(theta) / N_SLAB) ** 2)
    r12 = (c1 - N_SLAB * c2) / (c1 + N_SLAB * c2) if pol == "s" else (N_SLAB * c1 - c2) / (N_SLAB * c1 + c2)
    e = np.exp(2j * k0 * N_SLAB * D_SLAB * c2)
    r = r12 * (1 - e) / (1 - r12**2 * e)
    t = (1 - r12**2) * np.sqrt(e) / (1 - r12**2 * e)
    return np.abs(r) ** 2, np.abs(t) ** 2


def builder(pol, slab, monitor_freqs=None):
    def build(f, signal, k_t):
        sim = ls.Simulation(min=[0, 0, 0], max=[7 * um, LY, LZ], mesh=ls.UniformMesh(DX), run_time=400e-15, shutoff=1e-6)
        sim.set_boundaries(y=ls.Bloch(k_t), z=ls.PERIODIC)
        if slab:
            sim.add_medium(ls.Medium("glass", n=N_SLAB))
            sim.add_structure(ls.Box([X_SLAB, -1, -1], [X_SLAB + D_SLAB, 1, 1]), "glass")
        # s: E along z (normal to the xy plane of incidence); p: H along z
        sim.add_source(ls.PlaneWave(0, 1.5 * um, "+", "Ez" if pol == "s" else "Hz", signal, angle=THETA, name="pw"))
        fm = [f] if monitor_freqs is None else list(monitor_freqs)
        for name, x in (("R", 1.0 * um), ("T", 6.0 * um)):
            sim.add_monitor(ls.FluxMonitor(name, 0, x, [x, -1, -1], [x, 1, 1], fm))
        return sim

    return build


@pytest.mark.parametrize("pol, tol", [("s", 0.01), ("p", 3e-3)])
def test_fixed_angle_slab_matches_airy(pol, tol):
    """One run per frequency, k_t = k0(f) sin(45 deg): R and T against the
    Airy formula at 45 degrees; the incident power against the formula."""
    ref = ls.sweep_angle(builder(pol, False), FREQS, THETA, devices=DEVICE)
    sweep = ls.sweep_angle(builder(pol, True), FREQS, THETA, devices=DEVICE)
    incident = ref.flux("T")
    r, t = -sweep.flux("R") / incident, sweep.flux("T") / incident
    r_ex, t_ex = airy(FREQS, THETA, pol)
    p_ratio = incident / ref.incident_power(LY * LZ)
    print(f"{pol}: max |R - R_Airy| = {np.max(np.abs(r - r_ex)):.2e}, max |T - T_Airy| = {np.max(np.abs(t - t_ex)):.2e} "
          f"(R {r_ex.min():.3f}-{r_ex.max():.3f}); incident power / formula {p_ratio.min():.4f}-{p_ratio.max():.4f}")
    assert np.allclose(sweep.k_t, 2 * np.pi * FREQS / ls.C0 * np.sin(THETA))
    assert np.max(np.abs(p_ratio - 1)) < 0.01
    assert np.max(np.abs(r - r_ex)) < tol
    assert np.max(np.abs(t - t_ex)) < tol


@pytest.mark.parametrize("pol", ["s", "p"])
def test_broadband_fixed_k_run_drifts_in_angle(pol):
    """One broadband run with k_t fixed at the centre frequency: its
    transmission follows the Airy formula at theta(f) = asin(sin 45 deg f0/f),
    not at 45 degrees."""
    k_t = 2 * np.pi * F0 / ls.C0 * np.sin(THETA)
    signal = ls.GaussianPulse(F0, 0.08 * F0)
    ref, slab = (builder(pol, s, FREQS)(F0, signal, k_t).run(device=DEVICE) for s in (False, True))
    t = np.asarray(slab.flux("T")) / np.asarray(ref.flux("T"))
    theta_f = np.arcsin(np.sin(THETA) * F0 / FREQS)
    t_drift = np.array([airy(f, th, pol)[1] for f, th in zip(FREQS, theta_f)])
    _, t_45 = airy(FREQS, THETA, pol)
    print(f"{pol}: max |T - T_Airy(theta(f))| = {np.max(np.abs(t - t_drift)):.2e}, max |T - T_Airy(45)| = {np.max(np.abs(t - t_45)):.2e} "
          f"(theta from {np.degrees(theta_f.min()):.1f} to {np.degrees(theta_f.max()):.1f} deg)")
    assert np.max(np.abs(t - t_drift)) < 0.01
    assert np.max(np.abs(t - t_45)) > (0.1 if pol == "s" else 0.02)


def test_runs_dispatched_over_several_devices_match_serial():
    """The same sweep dispatched concurrently over a list of devices gives the
    same spectrum as one device."""
    f = FREQS[::4]
    one = ls.sweep_angle(builder("s", True), f, THETA, devices=DEVICE)
    many = ls.sweep_angle(builder("s", True), f, THETA, devices=SWEEP_DEVICES)
    print(f"devices {SWEEP_DEVICES}: max relative difference {np.max(np.abs(many.flux('T') / one.flux('T') - 1)):.1e}")
    assert np.allclose(many.flux("T"), one.flux("T"), rtol=1e-5)
    assert np.allclose(many.flux("R"), one.flux("R"), rtol=1e-4, atol=1e-12 * np.max(np.abs(one.flux("T"))))
