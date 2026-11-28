"""Energy balance of a reflective device that cannot radiate, over 1.50 .. 1.60 um, with the port profiles of the centre
frequency and with profiles that follow the band.
 2-D: a 225 nm silicon guide in oxide between two conducting plates 1.08 um apart (E out of plane), cut by a gap that is
      symmetric about the guide axis: the only propagating mode of that symmetry carries all the power, so
      |S11|^2 + |S21|^2 = 1.
usage: closed_guide_balance.py PKGDIR 2d DX_NM[,..] [device]"""
import sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
import lightsim as ls
from lightsim import invdes
C0 = ls.C0; lams = np.linspace(1.50e-6, 1.60e-6, 9); freqs = C0 / lams; F0 = float(np.mean(freqs))
kind = sys.argv[2]; dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
for dx_nm in [float(v) for v in sys.argv[3].split(",")]:
    dx = dx_nm * 1e-9
    L = 6.0e-6; nx = round(L / dx); pml = ls.PML(layers=16, inside=True)
    if kind != "2d":
        raise SystemExit("only the 2-D device is set up")
    if True:
        ny = 2 * round(0.54e-6 / dx)
        sim = ls.Simulation(min=[0, -ny * dx / 2, 0.0], max=[nx * dx, ny * dx / 2, 1e-6], mesh={"type": "uniform", "dx": [dx, dx, 1e-6]}, run_time=6e-12, shutoff=1e-8,
                            boundaries={"x": [pml, pml], "y": [ls.PEC_WALL, ls.PEC_WALL], "z": [ls.PERIODIC, ls.PERIODIC]}, background="clad")
        sim.add_medium(ls.Medium("clad", eps=1.444**2)); sim.add_medium(ls.Medium("core", eps=12.25))
        w2 = 112.5e-9; g2 = 90e-9; xc = nx * dx / 2
        sim.add_structure(ls.Box([-1.0, -w2, -1.0], [xc - g2, w2, 1.0]), "core"); sim.add_structure(ls.Box([xc + g2, -w2, -1.0], [1.0, w2, 1.0]), "core")
        lo, hi = [-ny * dx / 2, 0.0], [ny * dx / 2, 1e-6]; pol = "tm"
    xi, xo = round(1.6e-6 / dx) * dx, round(4.4e-6 / dx) * dx
    t0 = time.perf_counter()
    p_in = invdes.port_mode(sim, "in", 0, xi, [xi, *lo], [xi, *hi], F0, "+", pol)
    p_out = invdes.port_mode(sim, "out", 0, xo, [xo, *lo], [xo, *hi], F0, "-", pol)
    t_modes = time.perf_counter() - t0
    # a design grid of one oxide sample far from the guide axis: the S-matrix problem needs one, it changes nothing
    grid = invdes.DesignGrid([xc - dx, hi[0] - 3 * dx, -1.0], [xc + dx, hi[0] - dx, 1.0], (2, 2), ls.Medium("design", eps=1.444**2))
    print(f"{kind} dx {dx_nm:g} nm on {dev}: n_eff at the centre {p_in.neff[0].real:.4f}; two port solves {1e3 * t_modes:.0f} ms", flush=True)
    for pp in ("center", "band"):
        t0 = time.perf_counter()
        prob = invdes.SMatrixProblem(sim, [grid], [p_in], [p_out, p_in], freqs, devices=dev, fwidth=0.1 * F0, engine="python", port_profiles=pp)
        t_setup = time.perf_counter() - t0
        S = np.asarray(prob.solve([np.full((2, 2), 1.444**2)], adjoint=False).S)
        bal = abs(S[0, 0]) ** 2 + abs(S[1, 0]) ** 2
        print(f"  port_profiles={pp:6s} set-up {1e3 * t_setup:6.0f} ms  |S11| {abs(S[1,0]).min():.3f}..{abs(S[1,0]).max():.3f}  |S11|^2+|S21|^2 - 1 in %: " + " ".join(f"{100 * (b - 1):+.3f}" for b in bal) + f"  (largest {100 * abs(bal - 1).max():.3f} %)", flush=True)
