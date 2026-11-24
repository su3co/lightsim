"""Network solves on a seeded spread of circuit topologies: every back end of
lightsim.circuit.solve_network against the dense solve, results and times.

    python generality.py [--quick] [--threads N] [--out results/generality.jsonl]

Topologies (sizes from a handful of instances to thousands, 1 to 20,000
batch points, one or several modes per port):

  chain      two-ports in series (a waveguide route, a probeline)
  tree       1x2 splitters into a binary tree (a fan-out)
  ring       a bus of couplers, each with a ring (loops)
  clements   an N x N Clements mesh of MZIs (a lattice with many loops)
  lattice    a 2-D grid of 4-port junctions (meshes with loops everywhere)
  dense      random graph of 2- to 6-port blocks (no structure)
  multimode  a chain of two-ports with two modes per port
  star       one N-port hub with N two-ports on its arms (many external ports)

Every instance's S-matrix is a random contraction (passive) with a
wavelength dependence ``exp(i k L)`` per entry, so loops resonate. The dense
solve (``backend="dense"``, the solver of lightsim before the joins) is the
reference; ``auto`` is what users get. Times are medians of repeated solves
on the same inputs (instance S-parameters built once).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from lightsim import circuit as C

HERE = Path(__file__).resolve().parent


def rand_block(rng, n, nw, modes=1):
    """SDict of a random passive n-port (per mode), wavelength dependent."""
    ports = [f"o{k + 1}" for k in range(n)] if modes == 1 else [f"o{k + 1}@{m}" for k in range(n) for m in range(modes)]
    m = len(ports)
    A = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))
    A *= 0.95 / np.linalg.norm(A, 2)
    ph = rng.uniform(0, 40, size=(m, m))
    x = np.linspace(0, 1, nw)
    sd = {}
    for i, a in enumerate(ports):
        for j, b in enumerate(ports):
            sd[(a, b)] = A[j, i] * np.exp(1j * ph[j, i] * x) if nw > 1 else np.array([A[j, i]])
    return sd


def chain(rng, n, nw):
    sd = {f"i{k}": rand_block(rng, 2, nw) for k in range(n)}
    pairs = [((f"i{k}", "o2"), (f"i{k + 1}", "o1")) for k in range(n - 1)]
    return sd, pairs, {"in": ("i0", "o1"), "out": (f"i{n - 1}", "o2")}


def tree(rng, depth, nw):
    sd, pairs, ports = {}, [], {"in": ("s0", "o1")}
    k = 0
    level = ["s0"]
    sd["s0"] = rand_block(rng, 3, nw)
    for d in range(depth - 1):
        nxt = []
        for s in level:
            for p in ("o2", "o3"):
                k += 1
                c = f"s{k}"
                sd[c] = rand_block(rng, 3, nw)
                pairs.append(((s, p), (c, "o1")))
                nxt.append(c)
        level = nxt
    for s in level:
        ports[f"{s}_a"] = (s, "o2")
        ports[f"{s}_b"] = (s, "o3")
    return sd, pairs, ports


def rings(rng, n, nw):
    sd, pairs = {}, []
    for k in range(n):
        sd[f"c{k}"] = rand_block(rng, 4, nw)
        sd[f"r{k}"] = rand_block(rng, 2, nw)
        pairs += [((f"c{k}", "o3"), (f"r{k}", "o1")), ((f"r{k}", "o2"), (f"c{k}", "o2"))]
        if k:
            pairs.append(((f"c{k - 1}", "o4"), (f"c{k}", "o1")))
    return sd, pairs, {"in": ("c0", "o1"), "out": (f"c{n - 1}", "o4")}


def clements(rng, N, nw):
    """N x N mesh: N columns of 2x2 MZIs (two couplers, one phase shifter)."""
    sd, pairs = {}, []
    last = {m: None for m in range(N)}  # mode -> (inst, port) currently at the front
    ports = {}
    for col in range(N):
        for top in range(col % 2, N - 1, 2):
            a, b, p = f"a{col}_{top}", f"b{col}_{top}", f"p{col}_{top}"
            sd[a], sd[b], sd[p] = rand_block(rng, 4, nw), rand_block(rng, 4, nw), rand_block(rng, 2, nw)
            pairs += [((a, "o3"), (p, "o1")), ((p, "o2"), (b, "o1")), ((a, "o4"), (b, "o2"))]
            for m, port in ((top, "o1"), (top + 1, "o2")):
                if last[m] is None:
                    ports[f"in{m}"] = (a, port)
                else:
                    pairs.append((last[m], (a, port)))
            last[top], last[top + 1] = (b, "o3"), (b, "o4")
    for m in range(N):
        if last[m] is not None:
            ports[f"out{m}"] = last[m]
    return sd, pairs, ports


def lattice(rng, n, nw):
    """n x n grid of 4-ports (o1 west, o2 north, o3 east, o4 south)."""
    sd, pairs, ports = {}, [], {}
    for i in range(n):
        for j in range(n):
            sd[f"g{i}_{j}"] = rand_block(rng, 4, nw)
            if j:
                pairs.append(((f"g{i}_{j - 1}", "o3"), (f"g{i}_{j}", "o1")))
            if i:
                pairs.append(((f"g{i - 1}_{j}", "o4"), (f"g{i}_{j}", "o2")))
    for i in range(n):
        ports[f"w{i}"] = (f"g{i}_0", "o1")
        ports[f"e{i}"] = (f"g{i}_{n - 1}", "o3")
    return sd, pairs, ports


def dense_graph(rng, n, nw):
    sizes = rng.integers(2, 7, size=n)
    sd = {f"d{k}": rand_block(rng, int(sizes[k]), nw) for k in range(n)}
    free = [(f"d{k}", f"o{p + 1}") for k in range(n) for p in range(sizes[k])]
    rng.shuffle(free)
    n_ext = max(2, len(free) // 10)
    ports = {f"e{k}": free[k] for k in range(n_ext)}
    rest = free[n_ext:]
    pairs = [(rest[2 * k], rest[2 * k + 1]) for k in range(len(rest) // 2)]
    return sd, pairs, ports


def multimode(rng, n, nw):
    sd = {f"i{k}": rand_block(rng, 2, nw, modes=2) for k in range(n)}
    pairs = [((f"i{k}", "o2"), (f"i{k + 1}", "o1")) for k in range(n - 1)]
    return sd, pairs, {"in": ("i0", "o1"), "out": (f"i{n - 1}", "o2")}


def star(rng, n, nw):
    sd = {"hub": rand_block(rng, n, nw)}
    pairs, ports = [], {}
    for k in range(n):
        sd[f"a{k}"] = rand_block(rng, 2, nw)
        pairs.append((("hub", f"o{k + 1}"), (f"a{k}", "o1")))
        ports[f"p{k}"] = (f"a{k}", "o2")
    return sd, pairs, ports


CASES_FULL = [
    ("chain", chain, [2, 10, 100, 1000, 3000]),
    ("tree", tree, [2, 4, 7, 10]),
    ("ring", rings, [1, 4, 32, 256]),
    ("clements", clements, [2, 4, 8, 16, 32]),
    ("lattice", lattice, [2, 4, 8, 16]),
    ("dense", dense_graph, [4, 16, 64, 256]),
    ("multimode", multimode, [2, 20, 200]),
    ("star", star, [4, 16, 64]),
]
NWS_FULL = [1, 100, 2000, 20000]


def timed(f, budget=1.0, max_rep=20):
    t = time.perf_counter()
    r = f()
    dt = [time.perf_counter() - t]
    while sum(dt) < budget and len(dt) < max_rep:
        t = time.perf_counter()
        f()
        dt.append(time.perf_counter() - t)
    return r, float(np.median(dt)), float(np.min(dt)), len(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results" / "generality.jsonl"))
    ap.add_argument("--label", default="")
    ap.add_argument("--max-dense-bytes", type=float, default=8e9, help="skip the dense reference above this matrix size")
    a = ap.parse_args()
    cases = CASES_FULL if not a.quick else [(n, f, s[:3]) for n, f, s in CASES_FULL]
    nws = NWS_FULL if not a.quick else [1, 2000]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for name, gen, sizes in cases:
        for size in sizes:
            for nw in nws:
                rng = np.random.default_rng(hash((name, size, nw)) % 2**32)
                sd, pairs, ports = gen(rng, size, nw)
                nports = sum(len({p for k in s for p in k}) for s in sd.values())
                if nports * nports * 16 * nw > a.max_dense_bytes or nports * nw > 4e8:
                    continue
                plan = C._plan(sd, pairs, ports, {})
                rec = {"case": name, "size": size, "nw": nw, "instances": len(sd), "ports": nports, "internal": int(len(plan.I)), "external": len(plan.ext_names), "threads": a.threads, "label": a.label, "machine": platform.machine(), "cpus": len(os.sched_getaffinity(0))}
                rec["joins_cost"] = getattr(plan, "cost_joins", None)
                rec["dense_cost"] = getattr(plan, "cost_dense", None)
                rec["max_block"] = max(plan.sizes) if plan.joins_ok else None
                ref, rec["t_dense"], _, rec["n_dense"] = timed(lambda: C.solve_network(sd, pairs, ports, backend="dense"))
                for be in ("joins", "auto"):
                    if be == "joins" and not plan.joins_ok:
                        continue
                    S, t_med, t_min, n = timed(lambda: C.solve_network(sd, pairs, ports, backend=be, threads=a.threads))
                    scale = max(np.abs(np.asarray(v)).max() for v in ref.values())
                    rec[f"t_{be}"] = t_med
                    rec[f"err_{be}"] = float(max(np.abs(np.asarray(S[k]) - np.asarray(ref[k])).max() for k in ref) / max(scale, 1e-300))
                rec["speedup_auto"] = rec["t_dense"] / rec["t_auto"]
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{name:9s} {size:5d} nw={nw:6d} ports={nports:6d} int={rec['internal']:6d} ext={rec['external']:4d} blk={rec['max_block']} dense {rec['t_dense'] * 1e3:9.2f} ms  joins {rec.get('t_joins', float('nan')) * 1e3:9.2f} ms  auto {rec['t_auto'] * 1e3:9.2f} ms  x{rec['speedup_auto']:7.1f}  err {rec.get('err_joins', 0):.1e}", flush=True)


if __name__ == "__main__":
    main()
