"""The four-level two-electron gain medium (``FourLevelTwoElectron``)
through the Python API: a plane of atoms pumped at omega_b moves electrons
up (N3, N2 grow) while their number stays 2 Nd per atom; gradient engines
refuse the medium."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim.adjoint import check_gradient_media

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
LAM = 1e-6
F0 = ls.C0 / LAM


def medium():
    w0 = 2 * np.pi * F0
    nd = 1e24
    eps0 = 8.8541878128e-12
    g = ls.FourLevelTwoElectron(nd, w0, 1.3 * w0, 0.1 * w0, 0.13 * w0, 300e-15, 20e-15, 500e-15, 20e-15,
                                kappa_a=0.05 * 0.1 * w0 * w0 * eps0 / nd, kappa_b=0.02 * 0.13 * w0 * 1.3 * w0 * eps0 / nd)
    return ls.Medium("g2", eps=1.0, gain2=g), nd


def test_pumped_atoms_and_electron_number():
    med, nd = medium()
    dx = LAM / 60
    sim = ls.Simulation(min=[0, 0, 0], max=[4e-6, 2 * dx, 2 * dx], mesh=ls.UniformMesh(dx), run_time=1e-12)
    sim.set_boundaries(y=ls.PERIODIC, z=ls.PERIODIC)
    sim.add_structure(ls.Box([2e-6 - dx / 2, -1, -1], [2e-6 + dx / 2, 1, 1]), med)
    sim.add_source(ls.PlaneWave(0, 1e-6, "+", "Ey", ls.ContinuousWave(1.3 * F0, 0.13 * F0), amplitude=7e7))
    r = sim.run(device=DEVICE)
    lv = np.asarray(r.gain_levels())
    vc = dx**3
    driven = (lv - 8 * vc * np.array([nd, nd, 0, 0])) / (4 * vc) / nd
    print("driven populations / Nd:", np.round(driven, 4))
    assert abs(lv.sum() / (2 * nd * 12 * vc) - 1) < 1e-9
    assert driven[2] > 0.5 and driven[3] > 0.1 and driven[1] < 0.5


def test_refused_by_gradients():
    med, _ = medium()
    sim = ls.Simulation(min=[0, 0, 0], max=[2e-6, 0.1e-6, 0.1e-6], mesh=ls.UniformMesh(0.05e-6), run_time=1e-15)
    sim.add_structure(ls.Box([0.5e-6, -1, -1], [1e-6, 1, 1]), med)
    with pytest.raises(ValueError, match="two-electron"):
        check_gradient_media(sim)
