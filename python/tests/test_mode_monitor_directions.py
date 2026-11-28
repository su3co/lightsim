"""A mode monitor separates the two directions of travel: the forward amplitude does not pick up a reflection, a
straight guide reads no reflection, and |a+|^2 - |a-|^2 is the flux through the plane.

The monitor's magnetic field is the mean of the two half-cell planes around it, which for a guided wave is the field
on the plane times cos(beta dx / 2). Without undoing that factor each direction leaks into the other by
(1 - c) / (1 + c) of its amplitude: 2.1 % at the 45 nm cells of this test."""

import os

import numpy as np

import lightsim as ls
from lightsim.spec import Box, ModeMonitor, ModeSource

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = ls.C0 / 1.55e-6
DX = 45e-9


def guide(x_high):
    nx, ny = round(6.0e-6 / DX), round(4.0e-6 / DX)
    pml = ls.PML(layers=12, inside=True)
    sim = ls.Simulation(min=[0, -ny * DX / 2], max=[nx * DX, ny * DX / 2], mesh=ls.UniformMesh(DX), run_time=600e-15, polarization="te", background="clad",
                        boundaries={"x": [pml, x_high], "y": [pml, pml]}, shutoff=1e-7)
    sim.add_medium(ls.Medium("clad", eps=1.444**2))
    sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.25e-6], [1.0, 0.25e-6]), "core")
    h = ny * DX / 2 - 14 * DX

    def plane(x):
        return [x, -h], [x, h]

    xs = round(1.5e-6 / DX) * DX
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.05 * F0), name="src"))
    return sim, mode, plane


def test_forward_amplitude_ignores_a_total_reflection():
    # the guide ends on a conducting wall: the mode comes back whole, and |a+| must be the same on every plane
    sim, mode, plane = guide(ls.PEC_WALL)
    m0, n = round(3.0e-6 / DX), 20
    for m in range(n):
        sim.add_monitor(ModeMonitor(f"m{m}", 0, (m0 + m) * DX, *plane((m0 + m) * DX), [F0], [mode.profile]))
    res = sim.run(device=DEVICE)
    ap = np.array([abs(res.mode_amplitudes(f"m{m}")[0][0]) for m in range(n)])
    am = np.array([abs(res.mode_amplitudes(f"m{m}")[1][0]) for m in range(n)])
    ripple = (ap.max() - ap.min()) / (ap.max() + ap.min())
    assert ripple < 2e-3, ripple  # 5.3e-4 measured; 2.2e-2 before the two-plane mean was undone
    assert abs(am.mean() / ap.mean() - 1.0) < 2e-3


def test_straight_guide_reads_no_reflection_and_the_flux():
    sim, mode, plane = guide(ls.PML(layers=12, inside=True))
    xm = round(3.5e-6 / DX) * DX
    sim.add_monitor(ModeMonitor("m", 0, xm, *plane(xm), [F0], [mode.profile]))
    sim.add_monitor(ls.FluxMonitor("f", 0, xm, *plane(xm), [F0]))
    res = sim.run(device=DEVICE)
    ap, am = res.mode_amplitudes("m")
    assert abs(am[0]) / abs(ap[0]) < 3e-3  # 4.6e-4 measured; 2.2e-2 before
    # |a+|^2 - |a-|^2 is the power through the plane, in the units of a flux monitor
    assert abs((abs(ap[0]) ** 2 - abs(am[0]) ** 2) / res.flux("f")[0] - 1.0) < 1e-3
