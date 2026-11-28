"""``lightsim.run_many``: many simulations at once, each result what its own run gives.

Run with `pytest python/tests/test_run_many.py` (CPU unless LIGHTSIM_DEVICE is set).
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import Box, DFTMonitor, IndexGrid, ModeMonitor, ModeSource

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
H = 45e-9
F0 = ls.C0 / 1.55e-6


def scenes(n, pol="tm"):
    """n small 2-D scenes: two offset guides and a square of seeded grey permittivity between them."""
    out = []
    mode = None
    for seed in range(n):
        pml = ls.PML(layers=10, inside=True)
        sim = ls.Simulation(min=[-60 * H, -44 * H], max=[60 * H, 44 * H], mesh=ls.UniformMesh(H), run_time=150e-15, polarization=pol, background="clad", boundaries={"x": [pml, pml], "y": [pml, pml]}, shutoff=0)
        sim.add_medium(ls.Medium("clad", eps=1.444**2))
        sim.add_medium(ls.Medium("core", eps=12.25))
        sim.add_medium(ls.Medium("design", eps=12.25))
        d = 20 * H
        sim.add_structure(Box([-1.0, -9 * H - 2.5 * H], [-d, -9 * H + 2.5 * H]), "core")
        sim.add_structure(Box([d, 9 * H - 2.5 * H], [1.0, 9 * H + 2.5 * H]), "core")
        eps = np.random.default_rng(seed).uniform(1.444**2, 12.25, (21, 21, 1))
        sim.add_structure(IndexGrid([-d, -d, -1.0], [d, d, 1.0], eps), "design")
        xi, xo, ph = -40 * H, 40 * H, 18 * H
        if mode is None:
            mode = ls.solve_modes(sim, 0, xi, [xi, -9 * H - ph], [xi, -9 * H + ph], F0, num_modes=1)[0]
        sim.add_source(ModeSource(0, xi, "+", [xi, -9 * H - ph], [xi, -9 * H + ph], mode.profile, ls.GaussianPulse(F0, 0.1 * F0)))
        sim.add_monitor(ModeMonitor("out", 0, xo, [xo, 9 * H - ph], [xo, 9 * H + ph], [F0], [mode.profile]))
        sim.add_monitor(DFTMonitor("field", [-d, -d], [d, d], [F0]))
        out.append(sim)
    return out


def test_run_many_equals_one_run_after_the_other():
    sims = scenes(7)
    one = [s.run(device=DEVICE) for s in sims]
    for workers in (None, 1, 3):
        many = ls.run_many(sims, device=DEVICE, workers=workers)
        assert len(many) == len(sims)
        for a, b in zip(one, many):
            assert np.array_equal(np.asarray(a.mode_amplitudes("out")[0]), np.asarray(b.mode_amplitudes("out")[0]))
            assert np.array_equal(a.field("field", "Ez"), b.field("field", "Ez")) and np.abs(a.field("field", "Ez")).max() > 0
            assert a.stats["steps"] == b.stats["steps"]
    # the scenes differ, and the results come back in the order given
    amps = [complex(np.asarray(r.mode_amplitudes("out")[0])[0]) for r in one]
    assert len({f"{abs(a):.6e}" for a in amps}) == len(amps) and min(abs(a) for a in amps) > 0


def test_run_many_takes_a_list_of_devices_and_reports_the_failing_simulation():
    sims = scenes(3)
    assert ls.run_many([], device=DEVICE) == []
    many = ls.run_many(sims, device=[DEVICE, DEVICE], workers=1)
    one = [s.run(device=DEVICE) for s in sims]
    assert all(np.array_equal(np.asarray(a.mode_amplitudes("out")[0]), np.asarray(b.mode_amplitudes("out")[0])) for a, b in zip(one, many))
    with pytest.raises(ValueError, match="workers"):
        ls.run_many(sims, device=DEVICE, workers=0)
    with pytest.raises(ValueError, match="on_error"):
        ls.run_many(sims, device=DEVICE, on_error="skip")


def test_a_failing_simulation_does_not_stop_the_others():
    sims = scenes(5)
    one = [s.run(device=DEVICE) for s in sims]
    amp = lambda r: np.asarray(r.mode_amplitudes("out")[0])
    sims[1].add_structure(Box([0, 0], [H, H]), "no such medium")  # fails when it is compiled
    sims[3].monitors[0] = dict(sims[3].monitors[0], axis=7)  # stops inside the run
    for workers in (1, 3):
        with pytest.raises(ls.RunManyError, match="2 of 5 simulations failed") as caught:
            ls.run_many(sims, device=DEVICE, workers=workers)
        e = caught.value
        assert sorted(e.errors) == [1, 3] and "simulation 1" in str(e.errors[1]) and "no such medium" in str(e.errors[1])
        assert e.results[1] is None and e.results[3] is None
        # the others ran to the end and are what their own runs give
        assert all(np.array_equal(amp(e.results[i]), amp(one[i])) for i in (0, 2, 4))
        back = ls.run_many(sims, device=DEVICE, workers=workers, on_error="return")
        assert [isinstance(r, Exception) for r in back] == [False, True, False, True, False]
        assert all(np.array_equal(amp(back[i]), amp(one[i])) for i in (0, 2, 4))
