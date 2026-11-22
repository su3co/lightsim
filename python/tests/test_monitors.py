"""Cost and bookkeeping of frequency-domain monitors: many mode monitors on
one plane, the automatic sampling stride, flux boxes with many frequencies."""

import os
import time

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def _beam_scene(steps=1500):
    lam = 1.55 * um
    f0 = ls.C0 / lam
    h = lam / 20
    sim = ls.Simulation(min=[0, -5 * um, 0], max=[12 * um, 5 * um, h], mesh=ls.UniformMesh(h), run_time=1.0)
    sim.set_boundaries(z=ls.PERIODIC)
    sim.add_source(ls.GaussianBeam(0, 2 * um, "+", "Ez", [2 * um, 0, 0], 2.5 * um, ls.GaussianPulse(f0, 0.1 * f0)))
    sim.run_time = steps * sim.time_info()[0]
    return sim, f0, h


def test_many_mode_monitors_on_one_plane_cost_one():
    """A scan of collecting beams: 120 single-frequency mode monitors on one
    plane. They share one set of recorded planes, so the run costs what one
    monitor costs, and every monitor returns what it returns alone."""
    sim, f0, h = _beam_scene()
    x = 9 * um
    wmin, wmax = [x, -4 * um, 0], [x, 4 * um, h]
    profiles = []
    t0 = time.time()
    for i in range(120):
        waist = (1.5 + 2.0 * (i % 12) / 11) * um
        centre = (-1.5 + 3.0 * (i // 12) / 9) * um
        profile, _ = ls.gaussian_mode(sim, 0, x, wmin, wmax, f0, waist, center=[x, centre, 0], polarization="v")
        profiles.append(profile)
        sim.add_monitor(ls.ModeMonitor(f"m{i}", 0, x, wmin, wmax, [f0], [profile]))
    t_profiles = time.time() - t0
    res = sim.run(device=DEVICE)
    many = np.array([res.mode_amplitudes(f"m{i}")[0][0] for i in range(120)])

    # the same projections from one monitor that holds all the profiles ...
    one, _, _ = _beam_scene()
    one.add_monitor(ls.ModeMonitor("all", 0, x, wmin, wmax, [f0], profiles))
    r1 = one.run(device=DEVICE)
    together = np.array([r1.mode_amplitudes("all", i)[0][0] for i in range(120)])
    assert np.allclose(many, together, rtol=1e-6, atol=0)
    assert np.abs(many).max() > 0 and len(set(np.round(np.abs(many) / np.abs(many).max(), 6))) > 50  # they differ
    # ... at about the cost of that one monitor (it was 2 to 50 times, CPU to GPU)
    print(f"120 monitors {res.stats['seconds']:.2f} s, one monitor {r1.stats['seconds']:.2f} s, profiles {t_profiles:.2f} s")
    assert res.stats["seconds"] < 1.5 * r1.stats["seconds"] + 0.5
    # building the profiles does not get slower as monitors pile up on the simulation
    assert t_profiles < 5.0
    # every monitor still has its fields and its flux
    assert res.raw["dft"]["m7"]["fields"]["Ez"]["re"].shape == res.raw["dft"]["m0"]["fields"]["Ez"]["re"].shape
    assert res.flux("m7")[0] == pytest.approx(res.flux("m0")[0], rel=1e-12)


def _column(dft_stride, signal, monitor_stride=None):
    lam = 1.0 * um
    dx = lam / 40
    freqs = np.linspace(0.9, 1.1, 5) * ls.C0 / lam
    sim = ls.Simulation(min=[0, 0, 0], max=[4 * um, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=120e-15, shutoff=1e-8, dft_stride=dft_stride)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_structure(ls.Box([2 * um, -1, -1], [2.5 * um, 1, 1]), ls.Medium("glass", n=2.0))
    sim.add_source(ls.PlaneWave(0, 1 * um, "+", "Ey", signal))
    sim.add_monitor(ls.FluxMonitor("T", 0, 3.5 * um, [3.5 * um, -1, -1], [3.5 * um, 1, 1], freqs))
    sim.add_monitor(ls.DFTMonitor("E", [3.5 * um, 0, 0], [3.5 * um, 0, 0], freqs, components=("Ey",), stride=monitor_stride))
    return sim


def test_automatic_dft_stride_matches_every_step():
    """Frequency-domain monitors sample every few steps, as far apart as the
    bandwidth of the pulse allows; the result is that of sampling every step."""
    f0 = ls.C0 / (1.0 * um)
    pulse = ls.GaussianPulse(f0, 0.3 * f0)
    auto = _column(None, pulse).run(device=DEVICE)
    every = _column(1, pulse).run(device=DEVICE)
    s = auto.raw["dft"]["T"]["stride"]
    dt = auto.dt
    # 1 / (s dt) >= highest monitor frequency + f0 + 7 fwidth, and s is the largest such stride
    edge = 1.1 * f0 + f0 + 7 * 0.3 * f0
    assert s > 1 and 1 / (s * dt) >= edge > 1 / ((s + 1) * dt)
    assert auto.raw["dft"]["E"]["stride"] == s and every.raw["dft"]["T"]["stride"] == 1
    t_auto, t_every = np.asarray(auto.flux("T")), np.asarray(every.flux("T"))
    print("stride", s, "max relative flux difference", np.max(np.abs(t_auto / t_every - 1)))
    assert np.allclose(t_auto, t_every, rtol=2e-4, atol=0)
    assert np.allclose(auto.field("E", "Ey"), every.field("E", "Ey"), rtol=0, atol=2e-4 * np.abs(every.field("E", "Ey")).max())
    # a monitor's own stride wins; a continuous wave has no band edge, so every step is taken
    assert _column(None, pulse, monitor_stride=3).run(device=DEVICE).raw["dft"]["E"]["stride"] == 3
    cw = _column(None, ls.ContinuousWave(f0, 0.3 * f0))
    cw.shutoff = None
    cw.run_time = 20e-15
    assert cw.run(device=DEVICE).raw["dft"]["T"]["stride"] == 1


def test_flux_boxes_with_many_frequencies_do_not_dominate_the_run():
    """A scattering run: TFSF box and two flux boxes at 41 frequencies. The
    Fourier sums used to take several times as long as the field update."""
    lam = 0.6 * um
    f0 = ls.C0 / lam
    half = 0.25 * um

    def scene(nfreq):
        sim = ls.Simulation(min=[-half] * 3, max=[half] * 3, mesh=ls.UniformMesh(10e-9), run_time=1.0)
        sim.add_structure(ls.Sphere([0, 0, 0], 0.06 * um), ls.Medium("bead", n=2.0))
        sim.add_source(ls.TFSF([-0.6 * half] * 3, [0.6 * half] * 3, 0, "+", "Ez", ls.GaussianPulse(f0, 0.3 * f0), name="tf"))
        if nfreq:
            freqs = np.linspace(0.8, 1.2, nfreq) * f0
            sim.add_monitor(ls.FluxBoxMonitor("sca", [-0.8 * half] * 3, [0.8 * half] * 3, freqs))
            sim.add_monitor(ls.FluxBoxMonitor("abs", [-0.4 * half] * 3, [0.4 * half] * 3, freqs))
        sim.run_time = 3000 * sim.time_info()[0]
        return sim

    bare = scene(0).run(device=DEVICE)
    full = scene(41).run(device=DEVICE)
    print(f"no monitors {bare.stats['mcells_per_s']:.0f} Mcells/s, 2 flux boxes x 41 frequencies {full.stats['mcells_per_s']:.0f} Mcells/s, stride {full.raw['dft']['sca']['stride']}")
    assert full.raw["dft"]["sca"]["stride"] >= 8
    assert full.stats["seconds"] < 1.6 * bare.stats["seconds"] + 0.3
    # a lossless bead: what leaves the outer box left the inner one; nothing is absorbed inside the TFSF box
    sca, absorbed = np.asarray(full.flux("sca")), np.asarray(full.flux("abs"))
    assert np.all(sca > 0) and np.max(np.abs(absorbed)) < 0.02 * sca.max()

