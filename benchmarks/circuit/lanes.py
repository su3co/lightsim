"""Time of the network solve against the number of lanes per chunk (set
through LIGHTSIM_CIRCUIT_LANES in a fresh process per value), for the
topologies of scaling.py, at the thread count given by RAYON_NUM_THREADS.

    RAYON_NUM_THREADS=10 python lanes.py --label gb10_t10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANES = [8, 16, 32, 64, 128, 256, 512, 1024, 4096]

CHILD = r"""
import json, sys, time, numpy as np
sys.path.insert(0, %r)
import generality as G
from lightsim import circuit as C
out = {}
for name, gen, size, nw in [("chain", G.chain, 1000, 20000), ("ring", G.rings, 32, 20000), ("lattice", G.lattice, 16, 2000), ("clements", G.clements, 16, 4000), ("chip", G.chain, 40, 125000)]:
    sd, pairs, ports = gen(np.random.default_rng(1), size, nw)
    C.solve_network(sd, pairs, ports, backend="joins")
    ts = []
    for _ in range(5):
        t = time.perf_counter(); C.solve_network(sd, pairs, ports, backend="joins"); ts.append(time.perf_counter() - t)
    out[name] = float(np.median(ts))
print(json.dumps(out))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--auto-only", action="store_true", help="only the built-in choice of lanes")
    ap.add_argument("--out", default=str(HERE / "results" / "lanes.jsonl"))
    a = ap.parse_args()
    for lanes in ([0] if a.auto_only else LANES + [0]):
        env = dict(os.environ)
        if lanes:
            env["LIGHTSIM_CIRCUIT_LANES"] = str(lanes)
        else:
            env.pop("LIGHTSIM_CIRCUIT_LANES", None)
        r = subprocess.run([sys.executable, "-c", CHILD % str(HERE)], env=env, capture_output=True, text=True)
        times = json.loads(r.stdout.strip().splitlines()[-1])
        rec = {"label": a.label, "lanes": lanes or "auto", "threads": os.environ.get("RAYON_NUM_THREADS"), **times}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
