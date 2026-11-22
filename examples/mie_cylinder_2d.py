"""2-D simulation: scattering by a dielectric cylinder, both polarizations,
against the 2-D Mie series.

    python examples/mie_cylinder_2d.py --device cpu

The simulation is two-dimensional (two-component corners): nothing varies
along z, and each polarization updates three field components. "tm" has the
electric field along the cylinder axis, "te" across it.
"""

import argparse

import numpy as np
from scipy import special as sp

import lightsim as ls
from lightsim.spec import um

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
a = ap.parse_args()

lam = 1 * um
f0 = ls.C0 / lam
n_c, r = 2.0, 0.3 * um
freqs = np.linspace(0.85, 1.15, 7) * f0
eta0 = 376.730313412


def q_sca(m, x, e_along_axis, nmax=40):
    """Scattering efficiency of an infinite cylinder at normal incidence
    (Bohren & Huffman, ch. 8)."""
    q = 0.0
    for n in range(nmax + 1):
        jm, jmp = sp.jv(n, m * x), sp.jvp(n, m * x)
        jx, jxp = sp.jv(n, x), sp.jvp(n, x)
        hx, hxp = sp.hankel1(n, x), sp.h1vp(n, x)
        if e_along_axis:
            c = (jm * jxp - m * jmp * jx) / (jm * hxp - m * jmp * hx)
        else:
            c = (m * jm * jxp - jmp * jx) / (m * jm * hxp - jmp * hx)
        q += abs(c) ** 2 * (1 if n == 0 else 2)
    return 2 / x * q


for polarization, field in (("tm", "Ez"), ("te", "Ey")):
    sim = ls.Simulation(min=[-1.1 * um, -1.1 * um], max=[1.1 * um, 1.1 * um], mesh=ls.UniformMesh(lam / 60), run_time=70e-15, polarization=polarization)
    sim.add_structure(ls.Cylinder([0, 0], r), ls.Medium("rod", n=n_c))
    sim.add_source(ls.TFSF([-0.55 * um, -0.55 * um], [0.55 * um, 0.55 * um], 0, "+", field, ls.GaussianPulse(f0, 0.3 * f0), name="tf"))
    sim.add_monitor(ls.FluxBoxMonitor("sca", [-0.75 * um, -0.75 * um], [0.75 * um, 0.75 * um], freqs))
    res = sim.run(device=a.device)
    print(res)
    _, s = res.source_spectrum("tf")
    intensity = np.abs(s) ** 2 / (2 * eta0)
    # the slab is 1 m thick: the flux is the scattered power per metre of cylinder
    q_fdtd = np.asarray(res.flux("sca")) / intensity / (2 * r)
    for f, q in zip(freqs, q_fdtd):
        q_mie = q_sca(n_c, 2 * np.pi * f / ls.C0 * r, field == "Ez")
        print(f"  {polarization}  f/f0 = {f / f0:.2f}   Q_sca FDTD {q:.4f}   Mie {q_mie:.4f}   ({100 * (q / q_mie - 1):+.2f} %)")
