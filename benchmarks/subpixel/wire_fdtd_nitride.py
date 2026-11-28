"""3-D FDTD convergence of a silicon nitride wire (1000 x 400 nm, n = 2.0, in oxide, 1.55 um; reference index 1.640709): S21 of a straight 3.6 um section between two
mode ports, against mesh size, run length and sub-pixel smoothing. The reference for the level of theory is the phase
beta L with the effective index of the finite-difference mode solver extrapolated to zero cell size.
usage: cd benchmarks/invdes && python ../subpixel/wire_fdtd_nitride.py DEVICE OUT.jsonl DX_NM[,DX_NM..] [RUN_TIME_FS[,..]] [subpixel 1|0] [pml_um] [ZSHIFT_CELLS[,..]] [cell|tent]
(wire_fdtd.py with the nitride wire of placement_study.py in a wider box)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import problems  # noqa: E402
import lightsim as ls  # noqa: E402
from lightsim import invdes  # noqa: E402
from lightsim.spec import PML, Box, Medium  # noqa: E402

dev, out = sys.argv[1], sys.argv[2]
dxs = [float(v) for v in sys.argv[3].split(",")]
times = [float(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "300").split(",")]
subpixel = (sys.argv[5] != "0") if len(sys.argv) > 5 else True
pml_um = float(sys.argv[6]) if len(sys.argv) > 6 else 0.32
weighting = sys.argv[8] if len(sys.argv) > 8 else "cell"
zshifts = [float(v) for v in (sys.argv[7] if len(sys.argv) > 7 else "0").split(",")]  # wire moved along z by this fraction of a cell
LX, LY, LZ, XP, XM = 6.0e-6, 4.8e-6, 3.2e-6, 1.8e-6, 0.6e-6
WY, WZ = 0.5e-6, 0.2e-6
EPS_CORE = 4.0   # silicon nitride, n = 2.0
for dx_nm in dxs:
    dx = dx_nm * 1e-9
    npml = max(4, round(pml_um * 1e-6 / dx))
    for t_fs, zs in [(t, z) for t in times for z in zshifts]:
        pml = PML(layers=npml, inside=True)
        sim = ls.Simulation(min=[-LX / 2, -LY / 2, -LZ / 2], max=[LX / 2, LY / 2, LZ / 2], mesh=ls.UniformMesh(dx), run_time=t_fs * 1e-15,
                            boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]}, background="clad")
        sim.subpixel = subpixel; sim.subpixel_weighting = weighting
        sim.add_medium(Medium("clad", eps=problems.EPS_OX)); sim.add_medium(Medium("core", eps=EPS_CORE))
        sim.add_structure(Box([-1.0, -WY, -WZ + zs * dx], [1.0, WY, WZ + zs * dx]), "core")
        hy, hz = LY / 2 - (npml + 2) * dx, LZ / 2 - (npml + 2) * dx
        t0 = time.perf_counter()
        pin = invdes.port_mode(sim, "in", 0, -XP, [-XP, -hy, -hz], [-XP, hy, hz], problems.F0, "+")
        pout = invdes.port_mode(sim, "out", 0, XP, [XP, -hy, -hz], [XP, hy, hz], problems.F0, "-")
        pmid = invdes.port_mode(sim, "mid", 0, XM, [XM, -hy, -hz], [XM, hy, hz], problems.F0, "-")
        d = 0.4e-6
        grid = invdes.DesignGrid([-d, -d, -WZ + zs * dx], [d, d, WZ + zs * dx], (21, 21), Medium("design", eps=EPS_CORE))
        x = np.linspace(-d, d, 21); X, Y = np.meshgrid(x, x, indexing="ij")
        eps = np.where(np.abs(Y) <= WY + 1e-12, EPS_CORE, problems.EPS_OX).astype(float)
        prob = invdes.SMatrixProblem(sim, [grid], [pin], [pout, pmid], [problems.F0], devices=dev, fwidth=problems.PULSE_FWIDTH * problems.F0, engine="native")
        t1 = time.perf_counter(); sol = prob.solve([eps], adjoint=False); t2 = time.perf_counter()
        sol = prob.solve([eps], adjoint=False); t3 = time.perf_counter()
        Sv = np.asarray(sol.S).ravel()
        S, Sm = complex(Sv[0]), complex(Sv[1])
        k0 = 2 * np.pi / problems.LAM0
        ratio = S / Sm
        # the phase advance over XP - XM gives the effective index on the FDTD grid (branch nearest the port mode's index)
        ph = abs(np.angle(ratio)); nm = float(np.real(np.asarray(getattr(pin, "neff", 2.44)).ravel()[0])); best = None
        for m in range(0, 40):
            for sg in (1, -1):
                c = (sg * ph + 2 * np.pi * m) / (k0 * (XP - XM))
                if best is None or abs(c - nm) < abs(best - nm): best = c
        n_mode = float(np.real(np.asarray(getattr(pin, "neff", np.nan)).ravel()[0]))
        cells = [round(LX / dx), round(LY / dx), round(LZ / dx)]
        rec = dict(weighting=weighting, zshift=zs, dx_nm=dx_nm, run_fs=t_fs, subpixel=subpixel, npml=npml, cells=cells, s_re=S.real, s_im=S.imag, s_abs=abs(S), phase=float(np.angle(S)), mid_abs=abs(Sm), ratio_abs=abs(ratio), neff_fdtd=float(best),
                   n_mode_port=n_mode, setup_s=t1 - t0, first_s=t2 - t1, solve_s=t3 - t2, device=dev, stats=str(getattr(prob, "adjoint_stats", ""))[:300])
        open(out, "a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("## conv3d done", flush=True)
