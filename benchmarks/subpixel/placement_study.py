"""Accuracy of the two sub-pixel weightings ("cell", "tent") on waveguide cross-sections, against the position of the
body inside a cell. For each case the effective index of the port mode on the FDTD grid (the rasterised, smoothed
permittivity the run uses) is compared with a converged reference (finite-difference mode solver on meshes that
conform to the body, 20 / 10 / 5 nm, Richardson) or with the analytic slab solution (2-D case).
usage: python benchmarks/subpixel/placement_study.py OUT.jsonl [CASE,CASE..|all] [DX_NM,..]
Every body is shifted by s cells along both transverse axes, s = 0, 0.25, 0.5 (and 0.125, 0.375 for the range)."""
import sys, json, time
import numpy as np
import lightsim as ls
from lightsim import invdes, modes
from lightsim.spec import PML, Box, Medium, Lorentz
from lightsim.xsection import CrossSection

out = sys.argv[1]
want = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else None
dxs = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [40.0, 30.0, 20.0, 15.0, 10.0]
C0 = 299792458.0; LAM = 1.55e-6; F0 = C0 / LAM
SHIFTS = (0.0, 0.125, 0.25, 0.375, 0.5)
N_OX, N_SI, N_SIN = 1.444, 3.5, 2.0

def reference(kw, num):
    """Richardson limit of the conforming-mesh mode solver: [values], observed order."""
    v = {}
    for st in (20e-9, 10e-9, 5e-9):
        modes.clear_cache()
        xs = CrossSection.strip(step=st, max_step=st, **kw)
        v[st] = np.array([m.neff.real for m in xs.solve(LAM, num_modes=num)])
    a, b, c = v[20e-9], v[10e-9], v[5e-9]
    p = np.log2(np.abs((a - b) / (b - c)))
    return c + (c - b) / (2.0**p - 1.0), p

def port_index(dx, weighting, boxes, core, clad, half, num, shift):
    ly, lz = half
    pml = PML(layers=8, inside=True)
    sim = ls.Simulation(min=[-0.6e-6, -ly, -lz], max=[0.6e-6, ly, lz], mesh=ls.UniformMesh(dx), run_time=1e-14,
                        boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]}, background="clad", subpixel_weighting=weighting)
    sim.add_medium(clad); sim.add_medium(core)
    for (y0, y1, z0, z1) in boxes:
        sim.add_structure(Box([-1.0, y0 + shift * dx, z0 + shift * dx], [1.0, y1 + shift * dx, z1 + shift * dx]), "core")
    hy, hz = ly - 10 * dx, lz - 10 * dx
    p = invdes.port_mode(sim, "in", 0, 0.0, [0.0, -hy, -hz], [0.0, hy, hz], F0, "+", num_modes=num)
    return np.real(np.asarray(p.neff).ravel())[:num]

def slab_exact():
    from scipy.optimize import brentq
    k0 = 2 * np.pi / LAM; e1, e2, t = N_SI**2, N_OX**2, 0.22e-6
    def f(n, pol):
        kx = k0 * np.sqrt(e1 - n * n); g = k0 * np.sqrt(n * n - e2)
        return np.tan(kx * t / 2) - (e1 / e2 if pol else 1.0) * g / kx
    return brentq(lambda n: f(n, 0), 2.6, 3.0), brentq(lambda n: f(n, 1), 1.6, 2.3)

def slab_2d(dx, weighting, shift, pol):
    """2-D simulation (invariant z): a 220 nm silicon slab along x; the lateral port solver of the 2-D run."""
    pml = PML(layers=8, inside=True)
    sim = ls.Simulation(min=[-0.6e-6, -1.2e-6], max=[0.6e-6, 1.2e-6], mesh=ls.UniformMesh(dx), run_time=1e-14, boundaries={"x": [pml, pml], "y": [pml, pml]},
                        background="clad", polarization=pol, subpixel_weighting=weighting)
    sim.add_medium(Medium("clad", eps=N_OX**2)); sim.add_medium(Medium("core", eps=N_SI**2))
    sim.add_structure(Box([-1.0, -0.11e-6 + shift * dx], [1.0, 0.11e-6 + shift * dx]), "core")
    h = 1.2e-6 - 10 * dx
    p = invdes.port_mode(sim, "in", 0, 0.0, [0.0, -h], [0.0, h], F0, "+", polarization=pol)
    return float(np.real(np.asarray(p.neff).ravel()[0]))

si = Medium("core", eps=N_SI**2); ox = Medium("clad", eps=N_OX**2); sin = Medium("core", eps=N_SIN**2)
# a transparent dispersive silicon: one lossless Lorentz pole at 0.4 um, eps(1.55 um) = 12.25
w0 = 2 * np.pi * C0 / 0.4e-6; fac = 1.0 / (1.0 - (0.4 / 1.55) ** 2)
si_disp = Medium("core", eps=2.0, poles=[Lorentz((N_SI**2 - 2.0) / fac, w0, 0.0)])
MI = {"si": N_SI, "sio2": N_OX, "sin": N_SIN}
CASES = {
    "silicon_wire": dict(boxes=[(-0.24e-6, 0.24e-6, -0.11e-6, 0.11e-6)], core=si, half=(1.2e-6, 0.96e-6), num=2, names=["TE0", "TM0"],
                         ref=dict(core_width=0.48e-6, core_thickness=0.22e-6, side_margin=0.96e-6, clad_thickness=0.85e-6, box_thickness=0.85e-6, material_index=MI)),
    "nitride_wire": dict(boxes=[(-0.5e-6, 0.5e-6, -0.2e-6, 0.2e-6)], core=sin, half=(2.4e-6, 1.6e-6), num=2, names=["TE0", "TM0"],
                         ref=dict(core_width=1.0e-6, core_thickness=0.4e-6, core_material="sin", side_margin=1.9e-6, clad_thickness=1.4e-6, box_thickness=1.4e-6, material_index=MI)),
    "coupled_wires": dict(boxes=[(-0.58e-6, -0.1e-6, -0.11e-6, 0.11e-6), (0.1e-6, 0.58e-6, -0.11e-6, 0.11e-6)], core=si, half=(1.6e-6, 0.96e-6), num=2, names=["even", "odd"],
                          ref=dict(core_width=0.48e-6, core_thickness=0.22e-6, gap=0.2e-6, side_margin=1.02e-6, clad_thickness=0.85e-6, box_thickness=0.85e-6, material_index=MI)),
    "dispersive_silicon_wire": dict(boxes=[(-0.24e-6, 0.24e-6, -0.11e-6, 0.11e-6)], core=si_disp, half=(1.2e-6, 0.96e-6), num=2, names=["TE0", "TM0"], ref=None),
}
def emit(rec):
    open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)

refs = {}
for name, c in CASES.items():
    if want and name not in want: continue
    if c["ref"] is not None:
        t0 = time.perf_counter(); r, p = reference(c["ref"], c["num"]); refs[name] = r
        emit(dict(case=name, reference=[float(x) for x in r], order=[float(x) for x in p], names=c["names"], seconds=time.perf_counter() - t0))
    for dx_nm in dxs:
        for wt in ("cell", "tent"):
            for s in SHIFTS:
                v = port_index(dx_nm * 1e-9, wt, c["boxes"], c["core"], ox, c["half"], c["num"], s)
                emit(dict(case=name, dx_nm=dx_nm, weighting=wt, shift=s, neff=[float(x) for x in v]))
if not want or "slab_2d" in want:
    te, tm = slab_exact()
    emit(dict(case="slab_2d", reference=[te, tm], names=["E parallel to the faces", "E normal to the faces"]))
    for dx_nm in dxs:
        for wt in ("cell", "tent"):
            for s in SHIFTS:
                emit(dict(case="slab_2d", dx_nm=dx_nm, weighting=wt, shift=s, neff=[slab_2d(dx_nm * 1e-9, wt, s, "tm"), slab_2d(dx_nm * 1e-9, wt, s, "te")]))
print("## placement study done", flush=True)
