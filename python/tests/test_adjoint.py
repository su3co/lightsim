"""Adjoint gradients versus finite differences (cf. FDTDX test_adjoint_gradient)."""

import copy
import os

import numpy as np

import lightsim as ls
from lightsim.adjoint import mode_power_gradient
from lightsim.spec import ModeMonitor, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")


def build(delta_eps=0.0):
    lam = 1.55 * um
    f0 = ls.C0 / lam
    dx = 40e-9
    sim = ls.Simulation(min=[0, -1.2 * um, -0.9 * um], max=[4 * um, 1.2 * um, 0.9 * um], mesh=ls.UniformMesh(dx), run_time=220e-15, background="ox")
    sim.add_medium(ls.Medium("ox", n=1.444))
    sim.add_medium(ls.Medium("sin", n=2.0))
    sim.add_structure(ls.Box([-1, -0.4 * um, -0.2 * um], [5 * um, 0.4 * um, 0.2 * um]), "sin")
    blk = ([1.8 * um, 0.4 * um, -0.2 * um], [2.2 * um, 0.6 * um, 0.2 * um])  # block beside the core
    if delta_eps != 0.0:
        sim.add_medium(ls.Medium("pert", eps=1.444**2 + delta_eps))
        sim.add_structure(ls.Box(*blk), "pert")
    box_min, box_max = [0.8 * um, -0.9 * um, -0.7 * um], [0.8 * um, 0.9 * um, 0.7 * um]
    mode = ls.solve_modes(sim, 0, 0.8 * um, box_min, box_max, f0, num_modes=1)[0]
    sim.add_source(ls.ModeSource(0, 0.8 * um, "+", box_min, box_max, mode.profile, ls.GaussianPulse(f0, 0.1 * f0), name="src"))
    out = {"axis": 0, "position": 3.2 * um, "min": [3.2 * um, -0.9 * um, -0.7 * um], "max": [3.2 * um, 0.9 * um, 0.7 * um], "profile": mode.profile, "direction": "+"}
    return sim, out, f0, blk


def amplitude(sim, out, f0):
    s = copy.deepcopy(sim)
    s.add_monitor(ModeMonitor("o", 0, out["position"], out["min"], out["max"], [f0], [out["profile"]]))
    r = s.run(device=DEVICE)
    _, sf = r.source_spectrum("src")
    ap, _ = r.mode_amplitudes("o")
    return ap[0] / sf[0]


def test_adjoint_matches_finite_difference():
    sim, out, f0, blk = build()
    res = mode_power_gradient(sim, [1.6 * um, 0.2 * um, -0.4 * um], [2.4 * um, 0.8 * um, 0.4 * um], out, f0, device=DEVICE)
    d = 0.2
    ap = amplitude(build(+d)[0], out, f0)
    am = amplitude(build(-d)[0], out, f0)
    # complex d(a_+)/d(eps): magnitude and phase from one adjoint run
    da_fd = (ap - am) / (2 * d)
    da_adj = res.amplitude_derivative_in_box(sim, *blk)
    assert abs(da_adj - da_fd) / abs(da_fd) < 0.08, (da_adj, da_fd)
    # power figure of merit: same sign (this FOM is phase-dominated, so the
    # projection Re(conj(a) da) is small compared with |da|)
    g_fd = (abs(ap) ** 2 - abs(am) ** 2) / (2 * d)
    g_adj = res.total_gradient_in_box(sim, *blk)
    assert np.sign(g_adj) == np.sign(g_fd), (g_adj, g_fd)
    assert 0.5 < res.fom < 1.2


# ----------------------------------------------------------------------------
# 2.5-D (one cell thick, periodic z) grids
# ----------------------------------------------------------------------------


def build_2d(pol, delta_eps=0.0, dz=40e-9):
    """A strip of effective index 2.85 in 1.444 on a grid one cell thick."""
    from lightsim import eim

    lam = 1.55 * um
    f0 = ls.C0 / lam
    dx = 40e-9
    sim = ls.Simulation(min=[0, -1.5 * um, 0], max=[4 * um, 1.5 * um, dz], mesh=ls.UniformMesh([dx, dx, dz]), run_time=250e-15, background="clad", shutoff=1e-7)
    sim.set_boundaries(z=ls.PERIODIC)
    sim.add_medium(ls.Medium("clad", n=1.444))
    sim.add_structure(ls.Box([-1, -0.25 * um, -1], [5 * um, 0.25 * um, 1]), ls.Medium("core", n=2.85))
    # block in the cladding next to the core, uniform along z; its faces lie on
    # grid nodes, so that every cell it touches mixes permittivities linearly
    blk = ([1.8 * um, 0.40 * um, -1], [2.2 * um, 0.60 * um, 1])
    if delta_eps != 0.0:
        sim.add_structure(ls.Box(*blk), ls.Medium("pert", eps=1.444**2 + delta_eps))
    wmin, wmax = [0.8 * um, -1.2 * um, 0], [0.8 * um, 1.2 * um, dz]
    mode = eim.solve_port_mode(sim, 0, 0.8 * um, wmin, wmax, f0, pol)
    sim.add_source(ls.ModeSource(0, 0.8 * um, "+", wmin, wmax, mode.profile, ls.GaussianPulse(f0, 0.1 * f0), name="src"))
    out = {"axis": 0, "position": 3.2 * um, "min": [3.2 * um, -1.2 * um, 0], "max": [3.2 * um, 1.2 * um, dz], "profile": mode.profile, "direction": "+"}
    return sim, out, f0, blk


import pytest  # noqa: E402


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_adjoint_on_one_cell_thick_grid(pol):
    """TE has the in-plane components Ex, Ey, which sit on the z nodes of the
    single periodic cell: their cell volume is the whole cell, not half."""
    sim, out, f0, blk = build_2d(pol)
    res = mode_power_gradient(sim, [1.6 * um, 0.2 * um, 0.0], [2.4 * um, 0.8 * um, 40e-9], out, f0, device=DEVICE)
    d = 0.1
    ap = amplitude(build_2d(pol, +d)[0], out, f0)
    am = amplitude(build_2d(pol, -d)[0], out, f0)
    da_fd = (ap - am) / (2 * d)
    da_adj = res.amplitude_derivative_in_box(sim, *blk)
    print(pol, "adjoint", da_adj, "finite difference", da_fd, "ratio", abs(da_adj) / abs(da_fd))
    assert abs(da_adj - da_fd) / abs(da_fd) < 0.05, (da_adj, da_fd)
    # the same derivative from the per-cell arrays: a block that covers part of the cell thickness
    half = res.amplitude_derivative_in_box(sim, [blk[0][0], blk[0][1], 0.0], [blk[1][0], blk[1][1], 20e-9])
    assert abs(half / da_adj - 0.5) < 1e-6
    assert 0.5 < res.fom < 1.2
