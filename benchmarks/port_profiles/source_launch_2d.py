"""What a mode source launches backwards over a band: a 0.5 um silicon guide in oxide through absorbing layers at both
ends, a source in the middle, mode monitors (profiles that follow the band) behind and ahead of it. Printed: the
modulus of the backward amplitude behind the source over the forward amplitude ahead, at 1.50 to 1.60 um, for a source
with the profile of 1.55 um alone and for a source whose profile follows the band.
usage: PYTHONPATH=<lightsim>/python python3 source_launch_2d.py [cell_nm ...] [--pol te|tm] [--device cpu]"""
import sys, time
import numpy as np
import lightsim as ls
from lightsim.spec import Box, ModeMonitor, ModeSource

opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
args = [a for i, a in enumerate(sys.argv) if i > 0 and not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
pol, device = opt.get("pol", "te"), opt.get("device", "cpu")
LAMS = np.array([1.50e-6, 1.525e-6, 1.55e-6, 1.575e-6, 1.60e-6]); FREQS = ls.C0 / LAMS; F0 = ls.C0 / 1.55e-6


def launch(dx, kind):
    nx, ny = round(8.0e-6 / dx), round(4.0e-6 / dx)
    pml = ls.PML(layers=max(12, round(0.6e-6 / dx)), inside=True)
    sim = ls.Simulation(min=[0, -ny * dx / 2], max=[nx * dx, ny * dx / 2], mesh=ls.UniformMesh(dx), run_time=600e-15, polarization=pol, background="clad", boundaries={"x": [pml, pml], "y": [pml, pml]})
    sim.add_medium(ls.Medium("clad", eps=1.444**2)); sim.add_medium(ls.Medium("core", eps=12.25))
    sim.add_structure(Box([-1.0, -0.25e-6], [1.0, 0.25e-6]), "core")
    h = ny * dx / 2 - (pml["layers"] + 2) * dx
    plane = lambda x: ([x, -h], [x, h])
    xs = round(4.0e-6 / dx) * dx
    mode = ls.solve_modes(sim, 0, xs, *plane(xs), F0, num_modes=1)[0]
    t0 = time.perf_counter()
    band = ls.profile_over_band(sim, 0, xs, *plane(xs), mode.profile, FREQS)
    t_band = time.perf_counter() - t0
    sim.add_source(ModeSource(0, xs, "+", *plane(xs), band if kind == "band" else mode.profile, ls.GaussianPulse(F0, 0.10 * F0)))
    xb, xa = round(2.5e-6 / dx) * dx, round(5.5e-6 / dx) * dx
    sim.add_monitor(ModeMonitor("behind", 0, xb, *plane(xb), FREQS, [band]))
    sim.add_monitor(ModeMonitor("ahead", 0, xa, *plane(xa), FREQS, [band]))
    t0 = time.perf_counter(); res = sim.run(device=device); t_run = time.perf_counter() - t0
    back = np.abs(np.asarray(res.mode_amplitudes("behind")[1])).ravel(); fwd = np.abs(np.asarray(res.mode_amplitudes("ahead")[0])).ravel()
    return back / fwd, fwd, t_band, t_run


for cell in [float(a) for a in args] or [45.0, 22.5]:
    for kind in ("center", "band"):
        r, fwd, tb, tr = launch(cell * 1e-9, kind)
        print(f"{pol} {cell:g} nm source {kind:6s}: backward / forward amplitude at 1.50 1.525 1.55 1.575 1.60 um: " + " ".join(f"{100 * v:.4f} %" for v in r) + f" | forward amplitude relative to 1.55 um: " + " ".join(f"{v / fwd[2]:.4f}" for v in fwd) + f" | band solves {1e3 * tb:.0f} ms, run {tr:.2f} s", flush=True)
