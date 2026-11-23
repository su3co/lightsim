"""Absorbing layers behind waveguides of fitted (dispersive) transparent media.

A lossless Lorentz pole makes its medium metal-like (Re eps < 0) in a band
above its resonance; a dielectric guide next to such a medium carries backward
modes there, and a PML amplifies backward waves, so a guide that runs into the
layers made a long run diverge at a frequency far from the sources'. Two kinds
of pole do this: the far-infrared pole that ``fitted_medium`` gives a glass
(the infrared term of the Sellmeier equation), and the strong ultraviolet pole
of silicon fitted with ``eps_inf = 1``. Inside themselves the layers now damp
the first kind and freeze the second (``PML(stabilize=True)``, the default)."""

import os

import numpy as np
import pytest

import lightsim as ls
from lightsim import materials as mat
from lightsim.spec import Box, ModeMonitor, ModeSource, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
F0 = ls.C0 / (1.55 * um)
WAVELENGTHS = (1.60, 1.55, 1.50)


def guide(stabilize, run_time, shutoff):
    """A 0.8 um core (n = 1.996, constant) in fitted silica, along x, through the PML at both ends."""
    silica = mat.fitted_medium("SiO2", band=(1.45e-6, 1.65e-6), n_lorentz=2, medium_name="silica")
    assert min(p["omega0"] for p in silica["poles"]) < 0.1 * 2 * np.pi * F0, "the fit has a far-infrared pole"
    sim = ls.Simulation(
        min=[-2 * um, -1 * um], max=[2 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=run_time,
        invariant_axis="z", polarization="te", background="silica", shutoff=shutoff, shutoff_interval=1000,
    )
    pml = ls.PML(stabilize=stabilize)
    sim.set_boundaries(x=pml, y=pml)
    sim.add_medium(silica)
    sim.add_medium(ls.Medium("core", n=1.996))
    sim.add_structure(Box([-20 * um, -0.4 * um], [20 * um, 0.4 * um]), "core")
    plane = lambda x: ([x, -0.8 * um], [x, 0.8 * um])  # noqa: E731
    mode = ls.solve_modes(sim, 0, -1 * um, *plane(-1 * um), F0, num_modes=1)[0]
    sim.add_source(ModeSource(0, -1 * um, "+", *plane(-1 * um), mode.profile, ls.GaussianPulse(F0, 0.06 * F0)))
    for name, x in [("in", -0.5 * um), ("out", 1 * um)]:
        for i, w in enumerate(WAVELENGTHS):
            f = ls.C0 / (w * um)
            profile = ls.solve_modes(sim, 0, x, *plane(x), f, num_modes=1)[0].profile
            sim.add_monitor(ModeMonitor(f"{name}{i}", 0, x, *plane(x), [f], [profile]))
    return sim


def transmission_reflection(res):
    out = []
    for i in range(len(WAVELENGTHS)):
        (a, b), (c, _) = res.mode_amplitudes(f"in{i}"), res.mode_amplitudes(f"out{i}")
        out.append((abs(c[0] / a[0]) ** 2, abs(b[0] / a[0]) ** 2))
    return np.array(out)


def test_guide_in_fitted_glass_through_pml_is_stable():
    """40 000 steps (4.7 ps) after a 0.2 ps pulse: the energy keeps falling
    with the stabilised layers; with ``stabilize=False`` the run diverges."""
    long = 4.7e-12
    res = guide(True, long, None).run(device=DEVICE)
    energy = np.asarray(res.stats["energy"])
    late = energy[len(energy) // 2 :]
    print("stabilised: energy end / peak", energy[-1] / energy.max(), "late max / min", late.max() / late.min())
    assert energy[-1] < 1e-9 * energy.max()
    assert late.max() < 10 * late.min(), "no growth in the second half of the run"
    with pytest.raises(RuntimeError, match="unstable"):
        guide(False, long, None).run(device=DEVICE)


def test_stabilised_layers_do_not_change_the_reflection():
    """The guided mode is absorbed as before: the transmission along the guide
    and the reflection from the layers, at three wavelengths, are those of the
    plain layers (compared on a run short enough for the plain layers)."""
    tr = [transmission_reflection(guide(s, 400e-15, 1e-9).run(device=DEVICE)) for s in (True, False)]
    print("T, R stabilised:", tr[0].tolist(), "plain:", tr[1].tolist())
    assert np.allclose(tr[0][:, 0], tr[1][:, 0], atol=2e-4)
    assert np.all(tr[0][:, 1] < 2e-4)
    assert np.allclose(tr[0][:, 1], tr[1][:, 1], rtol=0.05, atol=1e-6)


def silicon_strip(stabilize, run_time, shutoff, dx=10e-9, half=(1.0, 0.5), subpixel=True):
    """A 0.5 um silicon strip, fitted with one ultraviolet pole (eps_inf = 1), in constant silica,
    along x, through the PML at both ends."""
    si = mat.fitted_medium("Si", band=(1.45e-6, 1.65e-6), n_lorentz=1, medium_name="si")
    assert si["eps"][0] < 1.01 and len(si["poles"]) == 1 and si["poles"][0]["omega0"] > 3 * 2 * np.pi * F0
    lx, ly = half
    sim = ls.Simulation(
        min=[-lx * um, -ly * um], max=[lx * um, ly * um], mesh=ls.UniformMesh(dx), run_time=run_time,
        invariant_axis="z", polarization="te", background="silica", shutoff=shutoff, shutoff_interval=1000, subpixel=subpixel,
    )
    pml = ls.PML(stabilize=stabilize)
    sim.set_boundaries(x=pml, y=pml)
    sim.add_medium(ls.Medium("silica", n=1.444))
    sim.add_medium(si)
    sim.add_structure(Box([-20 * um, -0.25 * um], [20 * um, 0.25 * um]), "si")
    plane = lambda x: ([x, -(ly - 0.1) * um], [x, (ly - 0.1) * um])  # noqa: E731
    xs = -(lx - 0.4) * um
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.06 * F0)))
    for name, x in [("in", xs + 0.25 * um), ("out", (lx - 0.4) * um)]:
        for i, w in enumerate(WAVELENGTHS):
            f = ls.C0 / (w * um)
            profile = ls.solve_modes(sim, 0, x, *plane(x), f, num_modes=1)[0].profile
            sim.add_monitor(ModeMonitor(f"{name}{i}", 0, x, *plane(x), [f], [profile]))
    return sim


def test_fitted_silicon_strip_through_pml_is_stable():
    """The ultraviolet pole: on 10 nm cells the plain layers diverge near step
    40 000; with the pole frozen inside the layers 80 000 steps show no growth.
    The strip is staircased here: with its surface smoothed (the default) the
    plain layers of this 2-D guide no longer diverge, so the control would not
    show the growth; in 3-D they do either way."""
    long = 1.8e-12
    res = silicon_strip(True, long, None, subpixel=False).run(device=DEVICE)
    energy = np.asarray(res.stats["energy"])
    late = energy[len(energy) // 2 :]
    print("frozen: steps", res.stats.get("steps"), "energy end / peak", energy[-1] / energy.max(), "late max / min", late.max() / late.min())
    assert len(energy) >= 75 and energy[-1] < 1e-8 * energy.max()
    assert late.max() < 10 * late.min(), "no growth in the second half of the run"
    with pytest.raises(RuntimeError, match="unstable"):
        silicon_strip(False, long, None, subpixel=False).run(device=DEVICE)


def test_frozen_pole_does_not_change_the_reflection():
    """The guided mode of the fitted-silicon strip is absorbed as before: the
    same reflection at the band centre, and a change below 1e-5 at its ends."""
    tr = [transmission_reflection(silicon_strip(s, 400e-15, 1e-9, dx=25e-9, half=(2.0, 1.0)).run(device=DEVICE)) for s in (True, False)]
    print("T, R frozen:", tr[0].tolist(), "plain:", tr[1].tolist())
    assert np.allclose(tr[0][:, 0], tr[1][:, 0], atol=2e-4)
    assert np.all(tr[0][:, 1] < 1e-4)
    assert abs(tr[0][1, 1] - tr[1][1, 1]) < 0.02 * tr[1][1, 1], "band centre"
    assert np.allclose(tr[0][:, 1], tr[1][:, 1], atol=1e-5)


def test_tilted_fitted_guide_through_pml_is_stable():
    """A fitted nitride core (sub-pixel smoothed with the fitted silica
    around it: mixed voxels along its tilted faces, also inside the layers)
    crossing the PML at 25 degrees: 40 000 steps after the pulse, the energy
    falls to the floor of single precision (about 1e-10 of its peak, where
    it fluctuates) and does not grow out of it late in the run."""
    band = (1.45e-6, 1.65e-6)
    silica = mat.fitted_medium("SiO2", band=band, n_lorentz=2, medium_name="silica")
    nitride = mat.fitted_medium("Si3N4", band=band, n_lorentz=2, medium_name="nitride")
    sim = ls.Simulation(
        min=[-2 * um, -1.5 * um], max=[2 * um, 1.5 * um], mesh=ls.UniformMesh(50e-9), run_time=4.7e-12,
        invariant_axis="z", polarization="te", background="silica", shutoff_interval=1000,
    )
    pml = ls.PML(stabilize=True)
    sim.set_boundaries(x=pml, y=pml)
    sim.add_medium(silica)
    th = np.radians(25.0)
    u, v = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    half_l, half_w = 10 * um, 0.3 * um
    verts = [tuple(p) for p in (-half_l * u - half_w * v, half_l * u - half_w * v, half_l * u + half_w * v, -half_l * u + half_w * v)]
    sim.add_structure(ls.PolySlab(verts), nitride)
    p = -1.0 * um * u
    sim.add_source(ls.PointDipole([p[0], p[1]], "Ey", ls.GaussianPulse(F0, 0.06 * F0)))
    res = sim.run(device=DEVICE)
    energy = np.asarray(res.stats["energy"])
    late = energy[len(energy) // 2 :]
    print("tilted fitted guide: energy end / peak", energy[-1] / energy.max(), "late max / peak", late.max() / energy.max())
    assert energy[-1] < 1e-8 * energy.max()
    assert late.max() < 1e-8 * energy.max(), "no growth in the second half of the run"
