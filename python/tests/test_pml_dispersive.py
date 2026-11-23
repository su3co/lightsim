"""Absorbing layers behind a waveguide clad in a fitted glass.

``fitted_medium`` of a glass over a telecom band puts one Lorentz pole in the
far infrared (the infrared term of the Sellmeier equation). Above that pole the
lossless model is metal-like (Re eps < 0), a dielectric core in it carries
backward modes there, and a PML amplifies backward waves: a guide that runs
into the layers made a long run diverge at a frequency far below the sources'.
The layers now damp such poles inside themselves (``PML(stabilize=True)``, the
default)."""

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
