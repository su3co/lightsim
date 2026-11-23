"""Adiabatic absorber boundaries (``lightsim.Absorber``): a passive alternative
to the PML for structures a PML amplifies."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import materials as mat
from lightsim.spec import Box, FluxBoxMonitor, ModeMonitor, ModeSource, PointDipole, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = ls.C0 / (1.55 * um)


def test_absorber_adds_its_layers_outside_the_region():
    sim = ls.Simulation(min=[0, 0, 0], max=[1 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=1e-14)
    sim.set_boundaries(x=ls.Absorber(layers=30), y=ls.PML(layers=8), z=[ls.Absorber(), ls.PML()])
    x, y, z = (np.asarray(g) for g in sim.grid())
    assert len(x) - 1 == 20 + 60 and len(y) - 1 == 20 + 16 and len(z) - 1 == 20 + 40 + 12
    assert np.isclose(x[0], -30 * 50e-9) and np.isclose(z[0], -40 * 50e-9) and np.isclose(z[-1], 1 * um + 12 * 50e-9)
    # the frozen mesh is the mesh of the region, without the layers
    frozen = sim.frozen_mesh()
    assert len(frozen["x"]) == 21 and len(frozen["z"]) == 21
    assert ls.Absorber() == {"type": "absorber", "layers": 40, "params": {"order": 3.0, "sigma_scale": 1.0}}


def guide(boundary, core, cladding, dx=25e-9, lx=4.0, ly=1.5, half_width=0.4, run_time=400e-15, shutoff=1e-9):
    """A slab guide along x (2-D, E in the plane) that runs through the x faces."""
    sim = ls.Simulation(min=[-lx * um, -ly * um], max=[lx * um, ly * um], mesh=ls.UniformMesh(dx), run_time=run_time,
                        invariant_axis="z", polarization="te", background="cladding", shutoff=shutoff, shutoff_interval=1000)
    sim.set_boundaries(x=boundary, y=boundary)
    sim.add_medium(cladding)
    sim.add_medium(core)
    sim.add_structure(Box([-20 * um, -half_width * um], [20 * um, half_width * um]), "core")
    return sim


def test_absorber_reflects_little_of_a_guided_mode():
    """A nitride guide in fitted silica through a 40-layer absorber (1 um at
    25 nm cells): the guided mode is reflected below 2e-4 in power across the
    band (a 12-layer PML reflects 5e-6 here)."""
    silica = mat.fitted_medium("SiO2", band=(1.45e-6, 1.65e-6), n_lorentz=2, medium_name="cladding")
    sim = guide(ls.Absorber(layers=40), ls.Medium("core", n=1.996), silica)
    plane = lambda x: ([x, -1.3 * um], [x, 1.3 * um])
    xs, xm, xo = -3.0 * um, -2.5 * um, 3.0 * um
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.06 * F0)))
    freqs = [ls.C0 / (w * um) for w in (1.60, 1.55, 1.50)]
    for i, f in enumerate(freqs):
        prof = ls.solve_modes(sim, 0, xm, *plane(xm), f, num_modes=1)[0].profile
        sim.add_monitor(ModeMonitor(f"in{i}", 0, xm, *plane(xm), [f], [prof]))
        sim.add_monitor(ModeMonitor(f"out{i}", 0, xo, *plane(xo), [f], [prof]))
    res = sim.run(device=DEVICE)
    for i in range(3):
        (a, b), (c, _) = res.mode_amplitudes(f"in{i}"), res.mode_amplitudes(f"out{i}")
        assert abs(b[0] / a[0]) ** 2 < 2e-4, i
        assert abs(abs(c[0] / a[0]) ** 2 - 1.0) < 2e-3, i


def silicon_cladding(boundary, steps):
    """A silica core in a fitted-silicon cladding, 12.5 nm cells: above its
    ultraviolet pole the silicon is metal-like, and the core carries backward
    modes there."""
    dx = 12.5e-9
    si = mat.fitted_medium("Si", band=(1.45e-6, 1.65e-6), n_lorentz=1, medium_name="cladding")
    sim = guide(boundary, ls.Medium("core", n=1.444), si, dx=dx, lx=1.0, ly=0.5, half_width=0.25, run_time=steps * 0.99 * dx / (ls.C0 * 2**0.5), shutoff=None)
    sim.add_source(PointDipole([0.1 * um, 0.03 * um], "Ey", ls.GaussianPulse(F0, 0.06 * F0)))
    return sim


def test_absorber_is_stable_where_a_pml_amplifies():
    """The plain PML amplifies the backward modes of this guide until the run
    is stopped (near step 65 000); the absorber is a passive medium and only
    lets the energy fall."""
    steps = 80_000
    with pytest.raises(RuntimeError, match="unstable"):
        silicon_cladding(ls.PML(stabilize=False), steps).run(device=DEVICE)
    res = silicon_cladding(ls.Absorber(layers=20), steps).run(device=DEVICE)
    e = np.asarray(res.stats["energy"])
    late = e[len(e) // 2 :]
    assert e[-1] < 1e-8 * e.max()
    assert late[-1] <= 1.05 * late.min()


def test_absorber_and_pml_agree_on_the_radiated_power():
    """3-D: the power a dipole radiates into silica, through a box around it,
    with PML and with absorbers on every face (corners and edges included)."""
    freqs = [ls.C0 / (w * um) for w in (1.65, 1.55, 1.45)]

    def power(boundary):
        sim = ls.Simulation(min=[-0.6 * um] * 3, max=[0.6 * um] * 3, mesh=ls.UniformMesh(50e-9), run_time=300e-15, background="silica", shutoff=1e-8)
        sim.set_boundaries(x=boundary, y=boundary, z=boundary)
        sim.add_medium(ls.Medium("silica", n=1.444))
        sim.add_source(PointDipole([0.0, 0.0, 0.0], "Ez", ls.GaussianPulse(F0, 0.1 * F0)))
        sim.add_monitor(FluxBoxMonitor("box", [-0.4 * um] * 3, [0.4 * um] * 3, freqs))
        return np.asarray(sim.run(device=DEVICE).flux("box"))

    p_pml, p_abs = power(ls.PML()), power(ls.Absorber(layers=24))
    assert np.all(p_pml > 0)
    assert np.allclose(p_abs, p_pml, rtol=1e-2)


def test_absorber_refuses_media_it_cannot_take():
    def run(x_max):
        sim = ls.Simulation(min=[-0.5 * um] * 3, max=[0.5 * um] * 3, mesh=ls.UniformMesh(50e-9), run_time=20e-15)
        sim.set_boundaries(x=ls.Absorber(layers=8), y=ls.PML(layers=6), z=ls.PML(layers=6))
        sim.add_medium(ls.Medium("kerr", n=2.0, chi3=1e-30))
        sim.add_structure(Box([-0.3 * um, -0.1 * um, -0.1 * um], [x_max, 0.1 * um, 0.1 * um]), "kerr")
        sim.add_source(PointDipole([0.0, 0.0, 0.0], "Ez", ls.GaussianPulse(F0, 0.1 * F0), amplitude=1e-15))
        return sim.run(device=DEVICE)

    # a nonlinear guide that runs into the absorber is refused, with the reason
    with pytest.raises(RuntimeError, match="reaches into an Absorber boundary"):
        run(5 * um)
    # the same guide ending before the absorber runs
    run(0.3 * um)
