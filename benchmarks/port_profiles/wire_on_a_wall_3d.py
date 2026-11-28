"""Band-edge leak and energy balance of a mode monitor in 3-D: a 480 x 220 nm silicon wire in oxide ending on a conducting
wall, so the launched mode comes back whole (|S11| = 1). Mode monitors on a row of planes read 1.50 .. 1.60 um with the
centre profile alone, with the mode at three frequencies over the band (lightsim.profile_over_band) and with the mode
solved at every frequency. Printed per wavelength: the ripple of |a+| along the row (a backward wave read as forward)
and |a-|^2 / |a+|^2 - 1 (the energy balance of the reflection). usage: band_leak_3d.py PKGDIR DX_NM[,..] [device]"""
import sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
import lightsim as ls
from lightsim.spec import Box, ModeMonitor, ModeSource
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
F0 = ls.C0 / 1.55e-6
lams = np.array([1.50, 1.525, 1.55, 1.575, 1.60]) * 1e-6
freqs = ls.C0 / lams
for dx_nm in [float(v) for v in sys.argv[2].split(",")]:
    dx = dx_nm * 1e-9
    nx, ny, nz = round(4.8e-6 / dx), 2 * round(1.2e-6 / dx), 2 * round(0.92e-6 / dx)
    pml = ls.PML(layers=10, inside=True)
    sim = ls.Simulation(min=[0, -ny * dx / 2, -nz * dx / 2], max=[nx * dx, ny * dx / 2, nz * dx / 2], mesh=ls.UniformMesh(dx), run_time=700e-15, background="clad",
                        boundaries={"x": [pml, ls.PEC_WALL], "y": [pml, pml], "z": [pml, pml]}, shutoff=1e-7)
    sim.add_medium(ls.Medium("clad", eps=1.444**2)); sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.24e-6, -0.11e-6], [1.0, 0.24e-6, 0.11e-6]), "core")
    hy, hz = ny * dx / 2 - 12 * dx, nz * dx / 2 - 12 * dx
    plane = lambda x: ([x, -hy, -hz], [x, hy, hz])
    xs = round(1.2e-6 / dx) * dx
    t0 = time.perf_counter(); mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]; t1 = time.perf_counter()
    band = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, freqs); t2 = time.perf_counter()
    every = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, freqs, nodes="all"); t3 = time.perf_counter()
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), mode.profile, ls.GaussianPulse(F0, 0.10 * F0)))
    m0 = round(2.6e-6 / dx); nmon = max(8, round(0.5e-6 / dx))
    for m in range(nmon):
        x = (m0 + m) * dx
        for tag, prof in (("c", mode.profile), ("b", band), ("e", every)):
            sim.add_monitor(ModeMonitor(f"{tag}{m}", 0, x, *plane(x), freqs, [prof]))
    t4 = time.perf_counter(); res = sim.run(device=dev); t5 = time.perf_counter()
    rip = lambda v: (np.abs(v).max(0) - np.abs(v).min(0)) / (np.abs(v).max(0) + np.abs(v).min(0))
    print(f"3-D wire, dx {dx_nm:g} nm, {nx}x{ny}x{nz} cells on {dev}: n_eff {mode.neff.real:.4f}; one mode solve {t1 - t0:.2f} s; three-frequency profile +{t2 - t1:.2f} s ({len(band['others'])} solves); every frequency +{t3 - t2:.2f} s ({len(every['others'])} solves); run {t5 - t4:.1f} s")
    print("  wavelength um                       " + " ".join(f"{l * 1e6:8.4f}" for l in lams))
    for tag, nm in (("c", "centre profile   "), ("b", "three frequencies"), ("e", "every frequency  ")):
        ap = np.array([res.mode_amplitudes(f"{tag}{m}")[0] for m in range(nmon)])
        am = np.array([res.mode_amplitudes(f"{tag}{m}")[1] for m in range(nmon)])
        bal = (np.abs(am) ** 2 / np.abs(ap) ** 2 - 1)
        print(f"  {nm} ripple of |a+|      " + " ".join(f"{100 * v:7.4f}%" for v in rip(ap)))
        print(f"  {nm} |S11|^2 - 1, worst  " + " ".join(f"{100 * v:+7.4f}%" for v in bal[np.abs(bal).argmax(0), range(len(lams))]), flush=True)
