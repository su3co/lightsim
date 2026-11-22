"""Automatic mirror-symmetry reduction (cf. FDTDX test_simulation_symmetry and
test_symmetry_structured_field): the reduced, unfolded run reproduces the full
domain."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.spec import DFTMonitor, FluxBoxMonitor, PointDipole, Sphere, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def graded_axis(h0, h1, half):
    """Symmetric graded nodes: spacing h0 at the centre growing to h1 at +-half."""
    xs, x, h = [0.0], 0.0, h0
    while x + h < half - 1e-12:
        x += h
        xs.append(x)
        h = min(h * 1.08, h1)
    xs.append(half)
    xs = np.array(xs)
    return np.concatenate([-xs[:0:-1], xs]).tolist()


def build(mesh, half=0.8 * um):
    lam = 1.0 * um
    f0 = ls.C0 / lam
    sim = ls.Simulation(min=[-half, -half, -half], max=[half, half, half], mesh=mesh, run_time=30e-15)
    sim.add_medium(ls.Medium("bead", n=1.8))
    sim.add_structure(Sphere([0, 0, 0.2 * um], 0.25 * um), "bead")
    g = sim.grid()[2]
    zc = 0.5 * (g[np.searchsorted(g, 0.0)] + g[np.searchsorted(g, 0.0) + 1])  # an Ez location
    sim.add_source(PointDipole([0, 0, zc], "Ez", ls.GaussianPulse(f0, 0.3 * f0), amplitude=1e-12, name="d"))
    sim.add_monitor(DFTMonitor("pl", [-0.6 * um, -0.6 * um, 0.3 * um], [0.6 * um, 0.6 * um, 0.3 * um], [f0], components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))
    sim.add_monitor(FluxBoxMonitor("P", [-0.5 * um] * 3, [0.5 * um] * 3, [f0]))
    return sim


@pytest.mark.parametrize("graded", [False, True])
def test_unfolded_symmetric_run_matches_full_domain(graded):
    if graded:
        ax = graded_axis(25e-9, 50e-9, 0.8 * um)
        mesh = {"type": "explicit", "x": ax, "y": ax, "z": ax}
    else:
        mesh = ls.UniformMesh(40e-9)
    sim = build(mesh)
    full = sim.run(device=DEVICE)
    red = ls.run_symmetric(sim, (1, 1, 0), device=DEVICE)
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        a = full.raw["dft"]["pl"]["fields"][comp]
        b = red.raw["dft"]["pl"]["fields"][comp]
        fa = np.asarray(a["re"]) + 1j * np.asarray(a["im"])
        fb = np.asarray(b["re"]) + 1j * np.asarray(b["im"])
        assert list(a["origin"]) == list(b["origin"]) and fa.shape == fb.shape, (comp, a["origin"], b["origin"], fa.shape, fb.shape)
        err = np.max(np.abs(fa - fb)) / np.max(np.abs(fa))
        assert err < 1e-4, (comp, err)
    pf, pr = full.raw["flux"]["P"][0], red.raw["flux"]["P"][0]
    assert abs(pr / pf - 1) < 1e-4, (pf, pr)
    # the reduced run is ~4x smaller
    assert red.stats["cells"] < 0.3 * full.stats["cells"]
