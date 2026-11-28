"""Port profiles that follow a band of frequencies (``lightsim.profile_over_band``, ``port_profiles="band"``).

Run with `pytest python/tests/test_port_profiles_band.py` (CPU unless LIGHTSIM_DEVICE is set).
"""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import invdes
from lightsim.spec import Box, ModeMonitor, ModeSource

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
DX = 45e-9
LAMS = np.array([1.50e-6, 1.55e-6, 1.60e-6])
FREQS = ls.C0 / LAMS
F0 = ls.C0 / 1.55e-6


def guide_on_a_wall(pol="te"):
    """A 0.5 um silicon guide in oxide that ends on a conducting wall: the mode comes back whole."""
    nx, ny = round(6.0e-6 / DX), round(4.0e-6 / DX)
    pml = ls.PML(layers=12, inside=True)
    sim = ls.Simulation(min=[0, -ny * DX / 2], max=[nx * DX, ny * DX / 2], mesh=ls.UniformMesh(DX), run_time=700e-15, polarization=pol, background="clad", boundaries={"x": [pml, ls.PEC_WALL], "y": [pml, pml]}, shutoff=1e-7)
    sim.add_medium(ls.Medium("clad", eps=1.444**2))
    sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.25e-6], [1.0, 0.25e-6]), "core")
    h = ny * DX / 2 - 14 * DX
    return sim, (lambda x: ([x, -h], [x, h]))


def test_one_frequency_leaves_the_profile_as_it_is():
    sim, plane = guide_on_a_wall()
    x = round(1.5e-6 / DX) * DX
    mode = ls.solve_modes(sim, 0, x, *plane(x), F0, num_modes=1)[0]
    assert ls.profile_over_band(sim, 0, x, *plane(x), mode.profile, [F0]) is mode.profile
    band = ls.profile_over_band(sim, 0, x, *plane(x), mode.profile, FREQS)
    assert len(band["others"]) >= 2 and "others" not in mode.profile
    assert all(abs(o["freq"] - F0) > 1e-6 * F0 for o in band["others"])
    # the index falls with the wavelength, and the profile keeps its sign
    lo = min(band["others"], key=lambda o: o["freq"])
    hi = max(band["others"], key=lambda o: o["freq"])
    assert lo["neff"] < mode.profile["neff"] < hi["neff"]
    e0 = np.asarray(mode.profile["e_re"]).ravel()
    assert all(np.dot(e0, np.asarray(o["e_re"]).ravel()) > 0 for o in band["others"])


def test_a_profile_of_another_solver_is_refused():
    sim, plane = guide_on_a_wall()
    x = round(1.5e-6 / DX) * DX
    mode = ls.solve_modes(sim, 0, x, *plane(x), F0, num_modes=1)[0]
    wrong = dict(mode.profile, neff=mode.profile["neff"] * 1.01)
    with pytest.raises(ValueError, match="reproduces the profile"):
        ls.profile_over_band(sim, 0, x, *plane(x), wrong, FREQS)


def test_band_profile_removes_the_leak_at_the_ends_of_the_band():
    """The forward amplitude on a row of planes in front of the wall: its ripple is the share of the returning wave
    read as forward. With the centre profile it is 0.3 to 0.4 % at 3 % detuning; with the band profile it is what a
    profile solved at each frequency gives."""
    sim, plane = guide_on_a_wall()
    xs = round(1.5e-6 / DX) * DX
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    band = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, FREQS)
    every = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, FREQS, nodes="all")
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.10 * F0)))
    m0, n = round(3.0e-6 / DX), 16
    for m in range(n):
        x = (m0 + m) * DX
        for tag, prof in (("c", mode.profile), ("b", band), ("e", every)):
            sim.add_monitor(ModeMonitor(f"{tag}{m}", 0, x, *plane(x), FREQS, [prof]))
    res = sim.run(device=DEVICE)

    def ripple(tag):
        a = np.abs(np.array([res.mode_amplitudes(f"{tag}{m}")[0] for m in range(n)]))
        return (a.max(0) - a.min(0)) / (a.max(0) + a.min(0))

    centre, banded, each = ripple("c"), ripple("b"), ripple("e")
    print("ripple of |a+| at 1.50, 1.55, 1.60 um: centre", centre, "band", banded, "every frequency", each)
    assert centre[0] > 2.5e-3 and centre[2] > 2.0e-3
    assert banded.max() < 1.0e-3
    assert np.abs(banded - each).max() < 5e-5
    assert abs(banded[1] - centre[1]) < 1e-6  # the centre frequency reads the profile itself


def closed_guide(port_profiles, freqs):
    """A 225 nm guide between two conducting plates, cut by a gap: nothing can radiate, so |S11|^2 + |S21|^2 = 1."""
    nx, ny = round(6.0e-6 / DX), 2 * round(0.54e-6 / DX)
    pml = ls.PML(layers=16, inside=True)
    sim = ls.Simulation(min=[0, -ny * DX / 2, 0.0], max=[nx * DX, ny * DX / 2, 1e-6], mesh={"type": "uniform", "dx": [DX, DX, 1e-6]}, run_time=5e-12, shutoff=1e-8, boundaries={"x": [pml, pml], "y": [ls.PEC_WALL, ls.PEC_WALL], "z": [ls.PERIODIC, ls.PERIODIC]}, background="clad")
    sim.add_medium(ls.Medium("clad", eps=1.444**2))
    sim.add_medium(ls.Medium("core", eps=12.25))
    xc = nx * DX / 2
    sim.add_structure(Box([-1.0, -112.5e-9, -1.0], [xc - 90e-9, 112.5e-9, 1.0]), "core")
    sim.add_structure(Box([xc + 90e-9, -112.5e-9, -1.0], [1.0, 112.5e-9, 1.0]), "core")
    lo, hi = [-ny * DX / 2, 0.0], [ny * DX / 2, 1e-6]
    xi, xo = round(1.6e-6 / DX) * DX, round(4.4e-6 / DX) * DX
    p_in = invdes.port_mode(sim, "in", 0, xi, [xi, *lo], [xi, *hi], F0, "+", "tm")
    p_out = invdes.port_mode(sim, "out", 0, xo, [xo, *lo], [xo, *hi], F0, "-", "tm")
    grid = invdes.DesignGrid([xc - DX, hi[0] - 3 * DX, -1.0], [xc + DX, hi[0] - DX, 1.0], (2, 2), ls.Medium("design", eps=1.444**2))
    prob = invdes.SMatrixProblem(sim, [grid], [p_in], [p_out, p_in], freqs, devices=DEVICE, fwidth=0.1 * F0, engine="python", port_profiles=port_profiles)
    return np.asarray(prob.solve([np.full((2, 2), 1.444**2)], adjoint=False).S)


def test_band_ports_close_the_energy_balance_of_a_closed_guide():
    centre, band = closed_guide("center", FREQS), closed_guide("band", FREQS)
    bal = lambda S: np.abs(S[0, 0]) ** 2 + np.abs(S[1, 0]) ** 2 - 1.0
    print("|S11|^2 + |S21|^2 - 1 at 1.50, 1.55, 1.60 um: centre", bal(centre), "band", bal(band), "|S11|", np.abs(band[1, 0]))
    assert np.abs(band[1, 0]).min() > 0.5  # a strong reflection
    assert np.abs(bal(centre)).max() > 4e-3
    assert np.abs(bal(band)).max() < 2.5e-3
    # the centre frequency is the profile's own: the same S either way
    assert np.abs(band[:, 0, 1] - centre[:, 0, 1]).max() < 1e-6


def test_band_ports_change_nothing_for_one_frequency():
    centre, band = closed_guide("center", [F0]), closed_guide("band", [F0])
    assert np.array_equal(centre, band)
    with pytest.raises(ValueError, match="port_profiles"):
        closed_guide("every", [F0])
