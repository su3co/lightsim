"""Throughput sweep: lightsim vs FDTDX on the vacuum-box benchmark.

    python benchmarks/speed/sweep.py --label "1x RTX PRO 6000" --device cuda:0 \
        --sizes 128,192,256,320,400,512 --fdtdx-python /path/to/fdtdx-venv/bin/python \
        --out benchmarks/speed/results/sweep_rtx6000_1gpu.json

For several GPUs pass --device cuda:0,1; FDTDX then uses every visible GPU
(set CUDA_VISIBLE_DEVICES accordingly). A size N means an N^3 box; "NX:NY:NZ"
gives other shapes. Each point is the best of `lightsim bench` (fused kernel)
and of `fdtdx_box.py --reps 2`, both over the same number of steps: 200, or
more on small grids so that every run does at least --min-updates cell updates
(a 128^3 box finishes 200 steps in under 20 ms, too short to time reliably).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dims(s):
    return [int(v) for v in s.split(":")] if ":" in s else [int(s)] * 3


def run_lightsim(binary, n, device, steps):
    out = subprocess.run([binary, "bench", "--n", ",".join(map(str, n)), "--steps", str(steps), "--device", device], capture_output=True, text=True, check=True).stdout
    m = re.search(r"([\d.]+) Mcells/s", out)
    return float(m.group(1)) / 1000 if m else None


def run_fdtdx(python, n, steps):
    try:
        out = subprocess.run([python, str(HERE / "fdtdx_box.py"), "--n", ",".join(map(str, n)), "--steps", str(steps), "--reps", "2"], capture_output=True, text=True, timeout=3600).stdout
    except subprocess.TimeoutExpired:
        return None
    for line in out.splitlines():
        if line.startswith("{"):
            return json.loads(line)["mcells_per_s"] / 1000
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sizes", default="128,192,256,320,400,512")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--min-updates", type=float, default=2e10, help="lengthen runs on small grids to at least this many cell updates")
    ap.add_argument("--lightsim", default=str(HERE.parent.parent / "target/release/lightsim"))
    ap.add_argument("--fdtdx-python", default=None, help="python of an environment with FDTDX installed (skip FDTDX if omitted)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.sizes.split(","):
        n = dims(s)
        cells = n[0] * n[1] * n[2]
        steps = max(a.steps, -(-int(a.min_updates) // cells))
        row = {"label": a.label, "grid": n, "cells": cells, "steps": steps, "lightsim": run_lightsim(a.lightsim, n, a.device, steps)}
        if a.fdtdx_python:
            row["fdtdx"] = run_fdtdx(a.fdtdx_python, n, steps)
        print(json.dumps(row), flush=True)
        rows.append(row)
    Path(a.out).write_text(json.dumps({"label": a.label, "device": a.device, "rows": rows}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
