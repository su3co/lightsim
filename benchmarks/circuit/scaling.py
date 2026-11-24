"""Strong and weak scaling of the circuit network solve over threads.

    taskset -c 0-63 python scaling.py --threads 1,2,4,8,16,32,64 --label x86

Strong scaling: fixed problems (a chain of 1000 two-ports, 32 ring
resonators, a 16 x 16 lattice, a 16-mode Clements mesh), each at a fixed
batch, timed with 1, 2, 4, ... threads; speed-up and parallel efficiency
against one thread. Weak scaling: the batch grows with the threads (a fixed
number of points per thread), time per point against threads; flat is ideal.
Only the solve is timed (instance S-parameters built once), so the curves
show the parallelisation of the compiled kernel, not of the models.
Results are appended to ``results/scaling.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import generality as G
from lightsim import circuit as C

HERE = Path(__file__).resolve().parent

CASES = [("chain", G.chain, 1000, 20000), ("ring", G.rings, 32, 20000), ("lattice", G.lattice, 16, 2000), ("clements", G.clements, 16, 4000)]
WEAK_PER_THREAD = 2000


def best(f, rep=5):
    f()
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(HERE / "results" / "scaling.jsonl"))
    a = ap.parse_args()
    threads = [int(x) for x in a.threads.split(",")]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {"label": a.label, "machine": platform.machine(), "cpus": len(os.sched_getaffinity(0)), "load_before": os.getloadavg()[0]}
    for name, gen, size, nw in CASES:
        rng = np.random.default_rng(1)
        sd, pairs, ports = gen(rng, size, nw)
        C.solve_network(sd, pairs, ports, backend="joins", threads=1)
        t1 = None
        for t in threads:
            med, lo, hi = best(lambda: C.solve_network(sd, pairs, ports, backend="joins", threads=t))
            t1 = t1 or med
            rec = {**meta, "kind": "strong", "case": name, "size": size, "nw": nw, "threads": t, "time": med, "min": lo, "max": hi, "speedup": t1 / med, "efficiency": t1 / med / t}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"strong {name:9s} t={t:3d} {med * 1e3:9.2f} ms  speed-up {t1 / med:6.2f}  eff {t1 / med / t:5.2f}", flush=True)
        # weak: WEAK_PER_THREAD points per thread
        tw1 = None
        for t in threads:
            rng = np.random.default_rng(1)
            sd, pairs, ports = gen(rng, size, WEAK_PER_THREAD * t)
            med, lo, hi = best(lambda: C.solve_network(sd, pairs, ports, backend="joins", threads=t), rep=3)
            tw1 = tw1 or med
            # weak efficiency: time with the first thread count over time with t threads (same work per thread)
            rec = {**meta, "kind": "weak", "case": name, "size": size, "nw": WEAK_PER_THREAD * t, "threads": t, "time": med, "efficiency": tw1 / med}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"weak   {name:9s} t={t:3d} nw={WEAK_PER_THREAD * t:7d} {med * 1e3:9.2f} ms  eff {tw1 / med:5.2f}", flush=True)
    meta_end = os.getloadavg()[0]
    print(f"## load before {meta['load_before']:.2f} after {meta_end:.2f}")


if __name__ == "__main__":
    main()
