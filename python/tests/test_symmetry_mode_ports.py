"""``run_symmetric`` with mode sources and mode monitors cut by the mirror plane: the symmetric run against the plain run.

Run with `pytest python/tests/test_symmetry_mode_ports.py` (CPU unless LIGHTSIM_DEVICE is set).
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import Box, DFTMonitor, FluxMonitor, ModeMonitor, ModeSource

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
DX = 40e-9
F0 = ls.C0 / 1.55e-6
FREQS = [ls.C0 / 1.60e-6, F0, ls.C0 / 1.50e-6]


def scene(centres, source_at, monitors_at, modes=1, run_time=150e-15):
    """Silicon wires 480 x 220 nm in oxide along x, centred on z = 0 and on the y given; a mode source on one of
    them and mode monitors (windows 1.2 x 0.8 um, cut by the plane z = 0) on those given."""
    half_y = max(abs(c) for c in centres) + 1.0e-6
    pml = ls.PML(layers=8, inside=True)
    sim = ls.Simulation(min=[0, -half_y, -0.8e-6], max=[3.2e-6, half_y, 0.8e-6], mesh=ls.UniformMesh(DX), run_time=run_time, background="clad", boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]})
    sim.add_medium(ls.Medium("clad", eps=1.444**2))
    sim.add_medium(ls.Medium("core", eps=12.25))
    for c in centres:
        sim.add_structure(Box([-1.0, c - 0.24e-6, -0.11e-6], [1.0, c + 0.24e-6, 0.11e-6]), "core")
    xs, xm = 20 * DX, 55 * DX
    win = lambda x, c: ([x, c - 0.6e-6, -0.4e-6], [x, c + 0.6e-6, 0.4e-6])
    solved = {c: ls.solve_modes(sim, 0, xs, *win(xs, c), F0, num_modes=modes) for c in set([source_at, *monitors_at])}
    sim.add_source(ModeSource(0, xs, "+", *win(xs, source_at), solved[source_at][0].profile, ls.GaussianPulse(F0, 0.1 * F0), name="in"))
    for i, c in enumerate(monitors_at):
        sim.add_monitor(ModeMonitor(f"m{i}", 0, xm, *win(xm, c), FREQS, [m.profile for m in solved[c]], per_frequency=False))
    return sim, xm


def amplitudes(res, name, mode=0):
    return np.asarray(res.mode_amplitudes(name, mode)[0])


def test_a_wire_with_a_mode_source_and_a_mode_monitor_cut_by_the_plane():
    """Before this was handled the symmetric run read the full-window profile from its first cell and gave 0.128 of
    the amplitude."""
    sim, xm = scene([0.0], 0.0, [0.0], modes=2)
    sim.add_monitor(DFTMonitor("field", [xm, -0.6e-6, -0.4e-6], [xm, 0.6e-6, 0.4e-6], [F0], components=("Ey",)))
    sim.add_monitor(FluxMonitor("through", 0, xm, [xm, -0.6e-6, -0.4e-6], [xm, 0.6e-6, 0.4e-6], [F0]))
    # a flux window that the plane does not cut (entirely above it): the same window in both runs
    sim.add_monitor(FluxMonitor("above", 0, xm, [xm, -0.6e-6, 0.04e-6], [xm, 0.6e-6, 0.4e-6], [F0]))
    plain = sim.run(device=DEVICE)
    # tangential E even about z = 0 (+1); and the quarter domain: E_y of this mode is normal to y = 0 and even (-1)
    for symmetry in ((0, 0, 1), (0, -1, 1)):
        sym = ls.run_symmetric(sim, symmetry, device=DEVICE)
        a, b = amplitudes(plain, "m0"), amplitudes(sym, "m0")
        print(symmetry, "forward amplitude, symmetric over plain:", np.abs(b) / np.abs(a), "phase difference", np.angle(b / a))
        assert np.abs(b / a - 1.0).max() < 5e-3
        ea, eb = np.asarray(plain.field("field", "Ey")), np.asarray(sym.field("field", "Ey"))
        assert ea.shape == eb.shape and np.abs(eb - ea).max() < 5e-3 * np.abs(ea).max()
        fa, fb = float(np.asarray(plain.flux("through"))[0]), float(np.asarray(sym.flux("through"))[0])
        assert abs(fb / fa - 1.0) < 5e-3
        # the flux read by the mode monitor itself, and its second mode (the other parity about z = 0: not launched)
        assert np.abs(np.asarray(sym.raw["modes"]["m0"]["flux"]) / np.asarray(plain.raw["modes"]["m0"]["flux"]) - 1.0).max() < 5e-3
        assert np.abs(np.asarray(sym.flux("m0")) / np.asarray(plain.flux("m0")) - 1.0).max() < 5e-3
        assert np.abs(amplitudes(sym, "m0", 1)).max() <= 2e-2 * np.abs(b).min() and np.abs(amplitudes(plain, "m0", 1)).max() < 2e-2 * np.abs(a).min()
        if symmetry == (0, 0, 1):
            ua, ub = float(np.asarray(plain.flux("above"))[0]), float(np.asarray(sym.flux("above"))[0])
            assert abs(ub / ua - 1.0) < 5e-3 and ua < 0.75 * fa


def test_a_coupler_with_mode_ports_cut_by_the_plane():
    """Two wires 200 nm apart: the source on one, mode monitors on both; through and cross amplitudes of the
    symmetric run against the plain run."""
    gap = 0.2e-6
    c = 0.5 * (0.48e-6 + gap)
    sim, _ = scene([-c, c], -c, [-c, c], run_time=180e-15)
    plain = sim.run(device=DEVICE)
    sym = ls.run_symmetric(sim, (0, 0, 1), device=DEVICE)
    for name in ("m0", "m1"):
        a, b = amplitudes(plain, name), amplitudes(sym, name)
        print(name, "|a| plain", np.abs(a), "symmetric", np.abs(b))
        assert np.abs(b - a).max() < 5e-3 * np.abs(amplitudes(plain, "m0")).max()
    assert np.abs(amplitudes(plain, "m1")).min() > 0.02 * np.abs(amplitudes(plain, "m0")).max()  # light has crossed


def test_what_cannot_be_reduced_is_refused():
    sim, xm = scene([0.0], 0.0, [0.0])
    # the wrong parity for this mode about z = 0
    with pytest.raises(ValueError, match="other parity"):
        ls.run_symmetric(sim, (0, 0, -1), device=DEVICE)
    # a mode monitor whose window is neither symmetric about the plane nor above it
    sim.add_monitor(ModeMonitor("low", 0, xm, [xm, -0.6e-6, -0.6e-6], [xm, 0.6e-6, 0.2e-6], [F0], [sim.sources[0]["profile"]]))
    with pytest.raises(ValueError, match="neither symmetric about the mid-plane nor entirely in the upper half"):
        ls.run_symmetric(sim, (0, 0, 1), device=DEVICE)
