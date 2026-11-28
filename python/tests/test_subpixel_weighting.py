"""The weight of the sub-pixel average (`subpixel_weighting`): the default spec is unchanged, and with the tent the
index of a slab mode on the grid hardly depends on where the slab's faces fall inside a cell."""

import numpy as np
import pytest
from scipy.optimize import brentq

import lightsim as ls
from lightsim import invdes

LAM = 1.55e-6
F0 = ls.C0 / LAM
N_CORE, N_CLAD, THICK = 3.5, 1.444, 0.22e-6


def slab_index(weighting, shift, dx=20e-9):
    """Mode with E parallel to the faces of a silicon slab in a 2-D simulation, the slab moved by `shift` cells."""
    pml = ls.PML(layers=8, inside=True)
    sim = ls.Simulation(min=[-0.6e-6, -1.2e-6], max=[0.6e-6, 1.2e-6], mesh=ls.UniformMesh(dx), run_time=1e-14, boundaries={"x": [pml, pml], "y": [pml, pml]},
                        background="clad", polarization="tm", subpixel_weighting=weighting)
    sim.add_medium(ls.Medium("clad", eps=N_CLAD**2))
    sim.add_medium(ls.Medium("core", eps=N_CORE**2))
    sim.add_structure(ls.Box([-1.0, -THICK / 2 + shift * dx], [1.0, THICK / 2 + shift * dx]), "core")
    h = 1.2e-6 - 10 * dx
    port = invdes.port_mode(sim, "in", 0, 0.0, [0.0, -h], [0.0, h], F0, "+", polarization="tm")
    return float(np.real(np.asarray(port.neff).ravel()[0]))


def slab_exact():
    k0 = 2 * np.pi / LAM

    def f(n):
        kx, g = k0 * np.sqrt(N_CORE**2 - n * n), k0 * np.sqrt(n * n - N_CLAD**2)
        return np.tan(kx * THICK / 2) - g / kx

    return brentq(f, 2.6, 3.0)


def test_default_spec_is_unchanged():
    sim = ls.Simulation(min=[0, 0, 0], max=[1e-6, 1e-6, 1e-6], mesh=ls.UniformMesh(50e-9), run_time=1e-15)
    assert sim.subpixel_weighting == "cell"
    assert "subpixel_weighting" not in sim.to_dict()
    sim.subpixel_weighting = "tent"
    assert sim.to_dict()["subpixel_weighting"] == "tent"
    with pytest.raises(ValueError):
        ls.Simulation(min=[0, 0, 0], max=[1e-6, 1e-6, 1e-6], mesh=ls.UniformMesh(50e-9), run_time=1e-15, subpixel_weighting="box")


def test_tent_index_does_not_depend_on_placement():
    exact = slab_exact()
    shifts = (0.0, 0.25, 0.5)
    cell = [slab_index("cell", s) - exact for s in shifts]
    tent = [slab_index("tent", s) - exact for s in shifts]
    # measured at 20 nm: cell +4.1e-3, -8.7e-4, -2.5e-3; tent -2.41e-3, -2.47e-3, -2.54e-3
    assert max(cell) - min(cell) > 5e-3
    assert max(tent) - min(tent) < 3e-4
    assert all(-3.0e-3 < e < -2.0e-3 for e in tent), tent
