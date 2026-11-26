"""Waveguide mode solving: lightsim's finite-difference solver against
Tidy3D's local mode solver (the same method: full-vector finite differences
on a uniform grid with conducting walls) and MPB (plane waves in a periodic
cell: a different method, reported in its own column).

    python bench_modes.py --engine ls --step 0.02
    python bench_modes.py --engine tidy3d --step 0.02
    python bench_modes.py --engine mpb --step 0.02      # in an environment with meep

The guide: a 500 x 220 nm silicon strip (n = 3.476) in silica (n = 1.444)
at 1.55 um, two modes, a 3.5 x 2.22 um window (1.5 um either side, 1 um
above and below), a uniform grid of ``--step`` um (the core edges on grid
lines). Recorded: the effective indices, the time of the first solve (the
set-up included) and the median of repeated solves.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_SI, N_SIO2, WL = 3.476, 1.444, 1.55
W, T, SIDE, CLAD, BOX = 0.5, 0.22, 1.5, 1.0, 1.0


def ls_solver(step):
    from lightsim.xsection import CrossSection

    def solve():
        xs = CrossSection.strip(core_width=W * 1e-6, core_thickness=T * 1e-6, side_margin=SIDE * 1e-6, clad_thickness=CLAD * 1e-6, box_thickness=BOX * 1e-6, step=step * 1e-6, max_step=step * 1e-6, material_index={"si": N_SI, "sio2": N_SIO2})
        ms = xs.solve(WL * 1e-6, num_modes=2)
        return [complex(m.neff) for m in ms], (len(xs.u_nodes), len(xs.v_nodes))

    return solve


def tidy3d_solver(step):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    def solve():
        core = td.Structure(geometry=td.Box(center=(0, T / 2, 0), size=(W, T, td.inf)), medium=td.Medium(permittivity=N_SI**2))
        size = (W + 2 * SIDE, T + CLAD + BOX)
        centre = (0, (T + CLAD - BOX) / 2)
        sim = td.Simulation(size=(size[0], size[1], 1.0), center=(centre[0], centre[1], 0), grid_spec=td.GridSpec.uniform(dl=step), structures=[core], medium=td.Medium(permittivity=N_SIO2**2), run_time=1e-12, boundary_spec=td.BoundarySpec(x=td.Boundary.pec(), y=td.Boundary.pec(), z=td.Boundary.periodic()))
        plane = td.Box(center=(centre[0], centre[1], 0), size=(size[0], size[1], 0))
        ms = ModeSolver(simulation=sim, plane=plane, mode_spec=td.ModeSpec(num_modes=2, precision="double"), freqs=[td.C_0 / WL])
        data = ms.solve()
        n = np.asarray(data.n_complex.values).ravel()
        return [complex(x) for x in n], tuple(int(len(c)) for c in (sim.grid.boundaries.x, sim.grid.boundaries.y))

    return solve


def mpb_solver(step):
    import meep as mp
    from meep import mpb

    def solve():
        geom = [mp.Block(size=mp.Vector3(W, T, mp.inf), center=mp.Vector3(0, T / 2), material=mp.Medium(index=N_SI))]
        size = mp.Vector3(W + 2 * SIDE, T + CLAD + BOX)
        lat = mp.Lattice(size=size)
        ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mp.Medium(index=N_SIO2), resolution=int(round(1 / step)), num_bands=2, verbose=False)
        ms.geometry_center = mp.Vector3(0, (T + CLAD - BOX) / 2)
        f = 1 / WL
        ks = ms.find_k(mp.NO_PARITY, f, 1, 2, mp.Vector3(0, 0, 1), 1e-6, f * N_SI, f * N_SIO2, f * N_SI)
        return [complex(k / f) for k in ks], (int(size.x / step), int(size.y / step))

    return solve


def runnable():
    """Instantaneous count of runnable threads on the host (/proc/loadavg), or None."""
    try:
        with open("/proc/loadavg") as fh:
            return int(fh.read().split()[3].split("/")[0])
    except (OSError, ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="ls")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(HERE / "results" / "modes.jsonl"))
    a = ap.parse_args()
    make = {"ls": ls_solver, "tidy3d": tidy3d_solver, "mpb": mpb_solver}[a.engine]
    solve = make(a.step)
    la0, ru0 = os.getloadavg()[0], runnable()
    t = time.perf_counter()
    neff, grid = solve()
    first = time.perf_counter() - t
    ts = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        solve()
        ts.append(time.perf_counter() - t)
    rec = {"workload": "strip_modes", "engine": a.engine, "step_um": a.step, "neff": [[x.real, x.imag] for x in neff], "grid": grid, "first": first, "repeat_median": float(np.median(ts)) if ts else first, "repeat_min": float(np.min(ts)) if ts else first, "threads": a.threads, "label": a.label, "machine": platform.machine(), "load": la0, "load_before": la0, "load_after": os.getloadavg()[0], "runnable_before": ru0, "runnable_after": runnable(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("engine", "step_um", "neff", "grid", "first", "repeat_median")}), flush=True)


if __name__ == "__main__":
    main()
