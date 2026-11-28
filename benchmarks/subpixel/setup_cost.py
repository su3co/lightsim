"""Set-up cost of the two sub-pixel weightings: wall time of `lightsim run` for a run of one time step on the CPU
(grid, rasterisation of the six components, coefficients, one step), the two weightings alternating, N repeats.
usage: python benchmarks/subpixel/setup_cost.py LIGHTSIM_CLI OUT.jsonl [REPEATS] ; scenes of 400 x 300 x 110 cells (20 nm)."""
import sys, json, os, subprocess, tempfile, time, math
import numpy as np
import lightsim as ls
cli, out = sys.argv[1], sys.argv[2]
rep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dx = 20e-9; L = [8.0e-6, 6.0e-6, 2.2e-6]
def base():
    pml = ls.PML(layers=8, inside=True)
    sim = ls.Simulation(min=[0, 0, 0], max=L, mesh=ls.UniformMesh(dx), run_time=1e-18, boundaries={"x": [pml, pml], "y": [pml, pml], "z": [pml, pml]}, background="clad")
    sim.add_medium(ls.Medium("clad", eps=1.444**2)); sim.add_medium(ls.Medium("core", eps=12.25)); sim.add_medium(ls.Medium("box", eps=2.0)); sim.add_medium(ls.Medium("air", eps=1.0))
    sim.add_source(ls.PointDipole([4e-6, 3e-6, 1.1e-6], "Ey", ls.GaussianPulse(1.93e14, 2e13)))
    return sim
def wire():
    s = base(); s.add_structure(ls.Box([-1, 2.76e-6, 0.99e-6], [1, 3.24e-6, 1.21e-6]), "core"); return s
def ring():
    s = base(); s.add_structure(ls.Box([-1, 0.76e-6, 0.99e-6], [1, 1.24e-6, 1.21e-6]), "core")
    c = [4.003e-6, 3.507e-6]; th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for r, m in ((2.0e-6, "core"), (1.52e-6, "clad")):
        s.add_structure(ls.PolySlab([[c[0] + r * math.cos(t), c[1] + r * math.sin(t)] for t in th], 0.99e-6, 1.21e-6), m)
    return s
def stack():
    # abutting bodies over the whole plane: buried oxide, slab, ridge and an air top (every voxel next to a shared face is composited again)
    s = base(); s.add_structure(ls.Box([-1, -1, -1], [1, 1, 0.99e-6]), "box"); s.add_structure(ls.Box([-1, -1, 0.99e-6], [1, 1, 1.06e-6]), "core")
    s.add_structure(ls.Box([-1, 2.76e-6, 1.06e-6], [1, 3.24e-6, 1.21e-6]), "core"); s.add_structure(ls.Box([-1, -1, 1.61e-6], [1, 1, 3e-6]), "air"); return s
def sphere():
    s = base(); s.add_structure(ls.Sphere([4.003e-6, 3.007e-6, 1.101e-6], 0.9e-6), "core"); return s
def cylinders():
    s = base()
    for i in range(12):
        for j in range(8):
            s.add_structure(ls.Cylinder([1.0e-6 + 0.5e-6 * i, 1.2e-6 + 0.5e-6 * j, 1.1e-6], 0.15e-6, 0.22e-6, "z"), "air")
    return s
d = tempfile.mkdtemp()
for name, f in (("wire", wire), ("ring_polygons", ring), ("stack_of_abutting_boxes", stack), ("sphere", sphere), ("cylinder_array", cylinders)):
    spec = os.path.join(d, name + ".json"); open(spec, "w").write(f().to_json())
    t = {"cell": [], "tent": []}
    for r in range(rep):
        for wt in ("cell", "tent") if r % 2 == 0 else ("tent", "cell"):
            t0 = time.perf_counter()
            p = subprocess.run([cli, "run", spec, "--device", "cpu", "--subpixel-weighting", wt, "--out", os.path.join(d, "o.json")], capture_output=True, text=True)
            el = time.perf_counter() - t0
            if p.returncode != 0: print(p.stderr[-400:]); sys.exit(1)
            t[wt].append(el)
    rec = dict(scene=name, cells=[round(L[a] / dx) for a in range(3)], cell_s=sorted(t["cell"]), tent_s=sorted(t["tent"]), load1=os.getloadavg()[0], threads=os.environ.get("RAYON_NUM_THREADS", "all"))
    open(out, "a").write(json.dumps(rec) + "\n")
    med = lambda v: sorted(v)[len(v) // 2]
    print(f"{name:26s} cell {med(t['cell']):.2f} s ({min(t['cell']):.2f}-{max(t['cell']):.2f})  tent {med(t['tent']):.2f} s ({min(t['tent']):.2f}-{max(t['tent']):.2f})  +{med(t['tent']) - med(t['cell']):.2f} s", flush=True)
print("## setup cost done", flush=True)
