"""2-D Mie scattering by an infinite dielectric cylinder (cf. FDTDX
test_mie_scattering): z-invariant domain (periodic z), a TFSF box spanning the
periodic axis, E along the cylinder axis (Bohren & Huffman case I)."""

import os

import numpy as np
from scipy import special as sp

import lightsim as ls
from lightsim.spec import TFSF, Cylinder, FluxMonitor, um

DEVICE = os.environ.get("LIGHTSIM_DEVICE", "cpu")
ETA0 = 376.730313412


def q_sca_tm(m, x, nmax=30):
    """Scattering efficiency of a cylinder, E parallel to the axis."""
    q = 0.0
    for n in range(0, nmax + 1):
        jm, jmp = sp.jv(n, m * x), sp.jvp(n, m * x)
        jx, jxp = sp.jv(n, x), sp.jvp(n, x)
        hx, hxp = sp.hankel1(n, x), sp.h1vp(n, x)
        b = (jm * jxp - m * jmp * jx) / (jm * hxp - m * jmp * hx)
        q += abs(b) ** 2 * (1 if n == 0 else 2)
    return 2 / x * q


def test_cylinder_scattering_matches_2d_mie():
    lam, r, n_c = 1.0 * um, 0.3 * um, 1.5
    f0 = ls.C0 / lam
    dx = 20e-9
    sim = ls.Simulation(min=[-1.3 * um, -1.3 * um, 0], max=[1.3 * um, 1.3 * um, 2 * dx], mesh=ls.UniformMesh(dx), run_time=80e-15)
    sim.set_boundaries(z={"type": "periodic"})
    sim.add_medium(ls.Medium("rod", n=n_c))
    sim.add_structure(Cylinder([0, 0, 0], r, 10 * um, axis=2), "rod")
    sim.add_source(TFSF([-0.6 * um, -0.6 * um, -1], [0.6 * um, 0.6 * um, 1], 0, "+", "Ez", ls.GaussianPulse(f0, 0.3 * f0), name="tf"))
    freqs = [0.85 * f0, f0, 1.15 * f0]
    b = 0.9 * um
    for nm, ax, pos in [("xp", 0, b), ("xm", 0, -b), ("yp", 1, b), ("ym", 1, -b)]:
        lo = [-b, -b, -1.0]
        hi = [b, b, 1.0]
        lo[ax] = hi[ax] = pos
        sim.add_monitor(FluxMonitor(nm, ax, pos, lo, hi, freqs))
    res = sim.run(device=DEVICE)
    lz = 2 * dx
    p_sca = np.asarray(res.flux("xp")) - np.asarray(res.flux("xm")) + np.asarray(res.flux("yp")) - np.asarray(res.flux("ym"))
    _, s = res.source_spectrum("tf")
    inc = 0.5 / ETA0 * np.abs(s) ** 2
    for i, f in enumerate(freqs):
        x = 2 * np.pi * f / ls.C0 * r
        c_sca = q_sca_tm(n_c, x) * 2 * r  # per unit length
        q_meas = p_sca[i] / lz / inc[i] / (2 * r)
        q_th = c_sca / (2 * r)
        print(f"f/f0={f / f0:.2f}: Q_sca FDTD {q_meas:.4f}  Mie {q_th:.4f}")
        assert abs(q_meas / q_th - 1) < 0.03, (f / f0, q_meas, q_th)
