"""lightsim.fem full-vectorial modes: analytic references, FD solver, gplugins-style API."""

import numpy as np
import pytest

import lightsim as ls
from lightsim import fem
from lightsim.fem import validation

um = 1e-6


@pytest.mark.parametrize(
    "case",
    ["modes_slab", "modes_fiber", "modes_strip_vs_fd", "modes_bend_loss", "modes_plasmon", "modes_lossy_perturbation", "eo_uniform_field"],
)
def test_validation_case(case):
    rep = validation.CASES[case]()
    assert rep.passed, rep.summary()


def _strip_mesh(h=0.02 * um):
    g = fem.Geometry().add(fem.Rectangle(-1.25 * um, -1.11 * um, 1.25 * um, 1.11 * um), "clad", "SiO2", mesh_order=5)
    g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=h, mesh_order=1)
    return g.mesh(default_size=0.1 * um)


def test_mode_object_and_profile_for_fdtd():
    m = _strip_mesh()
    modes = fem.solve_modes(m, 1.55e-6, 2, index={"Si": 3.4777, "SiO2": 1.444})
    te = modes[0]
    assert isinstance(te, ls.Mode) and te.te_fraction > 0.95
    assert abs(te.calculate_overlap(te) - 1) < 1e-9 and abs(te.calculate_overlap(modes[1])) < 1e-6
    assert 3.9 < te.group_index < 4.3 and 0.1 * um**2 < te.effective_area() < 0.3 * um**2
    assert te.E[0].shape == (len(te.info["u_nodes"]), len(te.info["v_nodes"]))
    # the profile of a lightsim simulation plane: it matches the FD mode of that plane
    sim = ls.Simulation(min=[0, -1.25 * um, -1.11 * um], max=[0.2 * um, 1.25 * um, 1.11 * um], mesh=ls.UniformMesh(20e-9), run_time=1e-15, background="ox")
    sim.add_medium(ls.Medium("ox", n=1.444))
    sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]), ls.Medium("si_strip", n=3.4777))
    lo, hi = [0.1 * um, -1.25 * um, -1.11 * um], [0.1 * um, 1.25 * um, 1.11 * um]
    fd = ls.solve_modes(sim, 0, 0.1 * um, lo, hi, ls.C0 / 1.55e-6, num_modes=1)[0]
    prof = te.profile_for(sim, 0, 0.1 * um, lo, hi)
    for c in (0, 1):
        a = np.array(prof["e_re"][c]) + 1j * np.array(prof["e_im"][c])
        b = np.array(fd.profile["e_re"][c]) + 1j * np.array(fd.profile["e_im"][c])
        if np.linalg.norm(b) > 1e-3 * np.linalg.norm(a):
            corr = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert corr > (0.995 if c == 0 else 0.98), (c, corr)  # c = 1: the minor component
    assert prof["dims"] == fd.profile["dims"]


def test_bend_mode_shift_and_pml_leaky():
    m = _strip_mesh(0.03 * um)
    straight = fem.solve_modes(m, 1.55e-6, 1, index={"Si": 3.4777, "SiO2": 1.444})[0]
    g = fem.Geometry().add(fem.Rectangle(-2 * um, -2 * um, 3.5 * um, 2 * um), "clad", "SiO2", mesh_order=5)
    g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=0.03 * um, mesh_order=1)
    mb = g.mesh(default_size=0.1 * um)
    bent = fem.solve_modes(mb, 1.55e-6, 1, index={"Si": 3.4777, "SiO2": 1.444}, radius=2 * um, pml={"right": 1.0 * um}, n_guess=straight.neff.real)[0]
    # a 2-um bend pushes the mode outwards: higher index at the reference radius, finite loss
    assert bent.neff.real > straight.neff.real
    assert 0 < bent.loss_db_per_m < 1e5


def test_gplugins_compute_cross_section_modes():
    gf = pytest.importorskip("gdsfactory")
    from gdsfactory.gpdk import LAYER_STACK
    from gdsfactory.technology import LayerStack

    from lightsim.fem import gplugins as gp

    gf.gpdk.PDK.activate()
    stack = LayerStack(layers={k: LAYER_STACK.layers[k].model_copy(deep=True) for k in ("core", "clad", "slab90", "box")})
    stack.layers["core"].thickness = 0.2
    modes = gp.compute_cross_section_modes("rib", stack, wavelength=1.55, num_modes=2, resolution_specs={"core": 0.03, "slab90": 0.06})
    assert len(modes) == 2 and isinstance(modes, gp.Modes)
    assert 2.45 < modes.n_effs[0].real < 2.55 and modes[0].te_fraction > 0.95
