"""Mie scattering of a dielectric sphere (TFSF illumination) versus Mie theory.

    python examples/mie_sphere.py --device cuda:0
"""

import argparse

import numpy as np

import lightsim as ls
from lightsim.spec import um

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
a = ap.parse_args()

lam = 1 * um
f0 = ls.C0 / lam
n_s, r = 2.0, 0.25 * um
sim = ls.Simulation(min=[-0.75 * um] * 3, max=[0.75 * um] * 3, mesh=ls.UniformMesh(lam / 50), run_time=45e-15)
sim.add_structure(ls.Sphere([0, 0, 0], r), ls.Medium("bead", n=n_s))
sig = ls.GaussianPulse(f0, 0.3 * f0)
sim.add_source(ls.TFSF([-0.4 * um] * 3, [0.4 * um] * 3, 0, "+", "Ey", sig, name="tf"))
freqs = np.linspace(0.85, 1.15, 7) * f0
sim.add_monitor(ls.FluxBoxMonitor("sca", [-0.52 * um] * 3, [0.52 * um] * 3, freqs))
res = sim.run(device=a.device)
print(res)
# incident intensity from the source spectrum (E-time samples)
eta0 = 376.730313412
_, S = res.source_spectrum("tf", monitor="sca")
for i, f in enumerate(freqs):
    sigma = res.flux("sca")[i] / (0.5 / eta0 * abs(S[i]) ** 2)
    print(f"f/f0 = {f / f0:.3f}  sigma_sca / (pi r^2) = {sigma / (np.pi * r**2):.3f}")
