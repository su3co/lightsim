"""What the profiles of a mode monitor mean when there are as many of them as frequencies.

Run with `pytest python/tests/test_mode_monitor_profiles.py` (CPU unless LIGHTSIM_DEVICE is set).
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import Box, ModeMonitor, ModeSource

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
DX = 45e-9
FREQS = ls.C0 / np.array([1.50e-6, 1.55e-6, 1.60e-6])
F0 = ls.C0 / 1.55e-6


def test_as_many_profiles_as_frequencies_are_modes_or_one_mode_per_frequency():
    """Two modes read at two frequencies are two modes (they were once taken as one mode with a profile per
    frequency); one mode solved at each of the two frequencies is one mode; ``per_frequency`` says it outright."""
    dx = DX
    nx, ny = round(6.0e-6 / dx), round(4.0e-6 / dx)
    pml = ls.PML(layers=14, inside=True)
    sim = ls.Simulation(min=[0, -ny * dx / 2], max=[nx * dx, ny * dx / 2], mesh=ls.UniformMesh(dx), run_time=400e-15, polarization="te", background="clad", boundaries={"x": [pml, pml], "y": [pml, pml]})
    sim.add_medium(ls.Medium("clad", eps=1.444**2))
    sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.25e-6], [1.0, 0.25e-6]), "core")
    h = ny * dx / 2 - 16 * dx
    plane = lambda x: ([x, -h], [x, h])
    xs, xm = round(2.0e-6 / dx) * dx, round(4.0e-6 / dx) * dx
    f2 = [FREQS[0], FREQS[2]]
    two = [m.profile for m in ls.solve_modes(sim, 0, xm, *plane(xm), F0, num_modes=2)]
    each = [ls.solve_modes(sim, 0, xm, *plane(xm), f, num_modes=1)[0].profile for f in f2]
    src = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0].profile
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), src, ls.GaussianPulse(F0, 0.10 * F0)))
    sim.add_monitor(ModeMonitor("modes", 0, xm, *plane(xm), f2, two))
    sim.add_monitor(ModeMonitor("modes_said", 0, xm, *plane(xm), f2, two, per_frequency=False))
    sim.add_monitor(ModeMonitor("first", 0, xm, *plane(xm), f2, [two[0]]))
    sim.add_monitor(ModeMonitor("second", 0, xm, *plane(xm), f2, [two[1]]))
    sim.add_monitor(ModeMonitor("each", 0, xm, *plane(xm), f2, each))
    sim.add_monitor(ModeMonitor("each_said", 0, xm, *plane(xm), f2, list(reversed(each)), per_frequency=True))
    for i, f in enumerate(f2):
        sim.add_monitor(ModeMonitor(f"alone{i}", 0, xm, *plane(xm), [f], [each[i]]))
    res = sim.run(device=DEVICE)
    amp = lambda name, mode=0: np.asarray(res.mode_amplitudes(name, mode)[0])
    # two modes at two frequencies: both there, each what a monitor of its own gives
    for name in ("modes", "modes_said"):
        assert len(res.raw["modes"][name]["plus"]) == 2
        assert np.array_equal(amp(name, 0), amp("first")) and np.array_equal(amp(name, 1), amp("second"))
    assert np.abs(amp("modes", 1)).max() < 0.05 * np.abs(amp("modes", 0)).min()
    # one mode solved at each frequency: one mode, each frequency on its own profile
    assert len(res.raw["modes"]["each"]["plus"]) == 1
    assert np.array_equal(amp("each"), np.concatenate([amp("alone0"), amp("alone1")]))
    # said outright, the profile of position i is used at frequency i (here the wrong way round, on purpose)
    assert len(res.raw["modes"]["each_said"]["plus"]) == 1 and not np.array_equal(amp("each_said"), amp("each"))
    # as many profiles as frequencies that are neither: refused until said
    sim.add_monitor(ModeMonitor("unclear", 0, xm, *plane(xm), f2, list(reversed(each))))
    with pytest.raises(Exception, match="per_frequency"):
        sim.run(device=DEVICE)
    with pytest.raises(Exception, match="as many profiles as frequencies"):
        sim.monitors[-1] = ModeMonitor("unclear", 0, xm, *plane(xm), f2, [two[0]], per_frequency=True)
        sim.run(device=DEVICE)
