"""Meep time-stepping rate against the number of MPI processes.

    mpirun -np 64 python meep_rate.py --scene wg3d_6M --steps 300

Builds the scene of meep_bench.py (source, mode monitors, design-region DFT
monitors, as in a forward run of the adjoint solver) and times a fixed number
of steps. Used to pick the process count that gives Meep its best time.
"""

from __future__ import annotations

import argparse
import json
import time

import meep as mp

import problems
from meep_bench import TU, Column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="wg3d_6M")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    mp.verbosity(0)
    sc = problems.make(a.scene)
    col = Column(sc, 0, False, problems.COURANT / sc.ndim**0.5)
    col.opt.update_design([(sc.design_eps().ravel() - sc.eps_lo) / (sc.eps_hi - sc.eps_lo)])
    t0 = time.perf_counter()
    col.opt.prepare_forward_run()
    col.sim.init_sim()
    init = time.perf_counter() - t0
    dt = col.sim.fields.dt
    col.sim.run(until=20 * dt)  # warm-up
    t0 = time.perf_counter()
    col.sim.run(until=a.steps * dt)
    run = time.perf_counter() - t0
    if mp.am_master():
        print(json.dumps({"scene": sc.name, "mpi_processes": mp.count_processors(), "steps": a.steps, "init_seconds": init, "step_seconds": run, "cells_per_second": sc.n_cells * a.steps / run, "seconds_per_forward_run": init + run / a.steps * sc.run_time / (dt * TU)}), flush=True)


if __name__ == "__main__":
    main()
