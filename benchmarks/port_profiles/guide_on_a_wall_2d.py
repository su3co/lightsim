"""Band-edge leak of a mode monitor: a 2-D silicon guide (0.5 um, in oxide) ending on a conducting wall, so the launched
mode comes back whole. A row of mode monitors reads 1.50 .. 1.60 um; the forward amplitude should have the same modulus
on every plane, and a backward wave read as forward shows as a ripple. Three profiles: the centre profile alone (1.55 um),
the centre profile with the mode at three frequencies over the band (lightsim.profile_over_band), and the same with the
mode solved at every frequency. usage: band_leak.py PKGDIR DX_NM[,..] [te|tm] [device]"""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import lightsim as ls
from lightsim.spec import Box, ModeMonitor, ModeSource
pol = sys.argv[3] if len(sys.argv) > 3 else "te"
dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
F0 = ls.C0 / 1.55e-6
lams = np.array([1.50, 1.5125, 1.525, 1.5375, 1.55, 1.5625, 1.575, 1.5875, 1.60]) * 1e-6
freqs = ls.C0 / lams
for dx_nm in [float(v) for v in sys.argv[2].split(",")]:
    dx = dx_nm * 1e-9
    nx, ny = round(6.0e-6 / dx), round(4.0e-6 / dx)
    pml = ls.PML(layers=12, inside=True)
    sim = ls.Simulation(min=[0, -ny * dx / 2], max=[nx * dx, ny * dx / 2], mesh=ls.UniformMesh(dx), run_time=900e-15, polarization=pol, background="clad",
                        boundaries={"x": [pml, ls.PEC_WALL], "y": [pml, pml]}, shutoff=1e-7)
    sim.add_medium(ls.Medium("clad", eps=1.444**2)); sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.25e-6], [1.0, 0.25e-6]), "core")
    h = ny * dx / 2 - 14 * dx
    plane = lambda x: ([x, -h], [x, h])
    xs = round(1.5e-6 / dx) * dx
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    t0 = time.perf_counter(); band = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, freqs); t1 = time.perf_counter()
    every = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, freqs, nodes="all"); t2 = time.perf_counter()
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.10 * F0)))
    m0 = round(3.0e-6 / dx); nmon = max(16, round(0.9e-6 / dx))
    for m in range(nmon):
        x = (m0 + m) * dx
        for tag, prof in (("c", mode.profile), ("b", band), ("e", every)):
            sim.add_monitor(ModeMonitor(f"{tag}{m}", 0, x, *plane(x), freqs, [prof]))
    res = sim.run(device=dev)
    rip = lambda v: (np.abs(v).max(0) - np.abs(v).min(0)) / (np.abs(v).max(0) + np.abs(v).min(0))
    print(f"{pol} dx {dx_nm:.0f} nm on {dev}: set-up: three-frequency profile {1e3 * (t1 - t0):.1f} ms ({len(band['others'])} extra solves), every frequency {1e3 * (t2 - t1):.1f} ms ({len(every['others'])} extra solves)")
    print("  wavelength um      " + " ".join(f"{l * 1e6:7.4f}" for l in lams))
    for tag, nm in (("c", "centre profile     "), ("b", "three frequencies  "), ("e", "every frequency    ")):
        ap = np.array([res.mode_amplitudes(f"{tag}{m}")[0] for m in range(nmon)])
        am = np.array([res.mode_amplitudes(f"{tag}{m}")[1] for m in range(nmon)])
        print(f"  {nm}" + " ".join(f"{100 * v:6.3f}%" for v in rip(ap)) + "   |a-|/|a+| mean " + " ".join(f"{v:.4f}" for v in (np.abs(am) / np.abs(ap)).mean(0)[[0, 4, 8]]), flush=True)
