"""Throughput of 2-D grids: the 2-D update, the same one-cell grid on the
general 3-D update, and FDTDX on a one-cell 3-D grid.

For square grids of 1e4 to 1e8 cells (CPML included) this runs
`lightsim bench --n N,N` with one polarization (te) and with both, with the
2-D update and with `--kernels 3d`, and optionally `benchmarks/fdtdx_2d.py`
on the same grids, and writes one JSON file with every run.

    python benchmarks/bench_2d.py --device cuda:0 --out benchmarks/results/twod_rtx6000.json
    python benchmarks/bench_2d.py --device cpu --out benchmarks/results/twod_cpu.json
    python benchmarks/bench_2d.py --device cuda:0 --solvers fdtdx --fdtdx /path/to/fdtdx/python --out ...
    python benchmarks/bench_2d.py --table benchmarks/results/twod_*.json     # markdown tables

`--lightsim` is the command-line program (default: target/release/lightsim
of this repository). Each configuration is run `--reps` times; the best is
kept. Run it on an otherwise idle device: a timing taken while another job
shares the device is not meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIDES = [100, 316, 1000, 3162, 10000]


def steps_for(cells: int, device: str) -> int:
    """Enough steps for a stable figure: more for small grids, where a step
    takes microseconds."""
    budget = 3e9 if device == "cpu" else 3e10
    return int(min(20000, max(40 if device == "cpu" else 200, budget / cells)))


def lightsim_bench(program, device, side, steps, polarization, kernels, reps):
    best = None
    for _ in range(reps):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            cmd = [program, "bench", "--device", device, "--n", f"{side},{side}", "--steps", str(steps), "--polarization", polarization, "--json", f.name]
            if kernels == "3d":
                cmd += ["--kernels", "3d"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            r = json.load(open(f.name))
        if best is None or r["mcells_per_s"] > best["mcells_per_s"]:
            best = r
    return best


def fdtdx_bench(python, side, steps, reps, gpu):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = subprocess.run([python, str(ROOT / "benchmarks" / "fdtdx_2d.py"), "--n", f"{side},{side}", "--steps", str(steps), "--reps", str(reps)], check=True, capture_output=True, text=True, env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def table(paths) -> None:
    """Print markdown tables (Gcells/s) of result files written by this
    script; files of the same device are merged into one table."""
    by_device = {}
    for path in paths:
        data = json.load(open(path))
        by_device.setdefault(data["device"], []).extend(data["runs"])
    for device, runs in by_device.items():
        rows = {}
        for r in runs:
            rows.setdefault(r["cells"], {})[(r["kernels"], r["polarization"])] = r["mcells_per_s"] / 1e3
        fdtdx = any(("fdtdx", "tm") in v for v in rows.values())
        print(f"\n{device}, Gcells/s\n")
        print("| cells | 2-D, TE | 2-D, both | one-cell 3-D route, TE | one-cell 3-D route, both |" + (" FDTDX (one-cell 3-D grid) |" if fdtdx else ""))
        print("|---|---|---|---|---|" + ("---|" if fdtdx else ""))
        for cells in sorted(rows):
            v = rows[cells]
            cell = lambda k: f"{v[k]:.2f}" if k in v else ""  # noqa: E731
            line = f"| {cells:,} | {cell(('2d', 'te'))} | {cell(('2d', 'both'))} | {cell(('3d', 'te'))} | {cell(('3d', 'both'))} |"
            if fdtdx:
                line += f" {cell(('fdtdx', 'tm'))} |"
            print(line.replace(",", " "))


def main() -> None:
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--table":
        return table(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lightsim", default=str(ROOT / "target" / "release" / "lightsim"))
    ap.add_argument("--fdtdx", default=None, help="python interpreter of an environment with FDTDX")
    ap.add_argument("--solvers", default="lightsim", help="comma-separated: lightsim, fdtdx (fdtdx needs --fdtdx)")
    ap.add_argument("--sides", default=",".join(str(s) for s in SIDES))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-3d-above", type=float, default=2e8, help="skip the 3-D-kernel runs on larger grids (cells)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    gpu = a.device.split(":")[1] if a.device.startswith("cuda:") else None
    solvers = set(a.solvers.split(","))
    if "fdtdx" in solvers and not a.fdtdx:
        ap.error("--solvers fdtdx needs --fdtdx PYTHON")
    for side in (int(s) for s in a.sides.split(",")):
        cells = side * side
        steps = steps_for(cells, a.device)
        for pol in ("te", "both") if "lightsim" in solvers else ():
            for kernels in ("2d", "3d"):
                if kernels == "3d" and cells > a.skip_3d_above:
                    continue
                # the 3-D update on a one-cell grid is slow: fewer steps
                st = steps if kernels == "2d" else max(40, steps // 10 if a.device != "cpu" else steps // 3)
                r = lightsim_bench(a.lightsim, a.device, side, st, pol, kernels, a.reps)
                r.update(solver="lightsim", polarization=pol, kernels=kernels)
                runs.append(r)
                print(f"{side:6d}^2 {pol:4s} {kernels}: {r['mcells_per_s'] / 1e3:8.3f} Gcells/s  ({r['steps']} steps, {r['seconds']:.3f} s)  [{r['backend']}]", flush=True)
        if "fdtdx" in solvers:
            try:
                r = fdtdx_bench(a.fdtdx, side, steps // 10 if cells >= 1e7 else steps, a.reps, gpu)
                r.update(polarization="tm", kernels="fdtdx")
                runs.append(r)
                print(f"{side:6d}^2 fdtdx  : {r['mcells_per_s'] / 1e3:8.3f} Gcells/s  ({r['steps']} steps, {r['seconds']:.3f} s, compile {r['compile_seconds']:.1f} s)", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"{side:6d}^2 fdtdx failed: {e.stderr[-400:] if e.stderr else e}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"device": a.device, "runs": runs}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
