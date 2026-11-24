"""FDTDX multi-node throughput benchmark (one process per GPU, JAX distributed).

Same scene as `fdtdx_box.py` (vacuum box, 25 nm cells, 12-cell CPML inside the
box, one dipole); FDTDX shards the grid along x over all global devices.
Run one process per machine, e.g. on three GB10 nodes:

    # on node0, node1, node2 (process 0, 1, 2)
    NCCL_SOCKET_IFNAME=<interface> python fdtdx_multinode.py --n 1200,640,512 \
        --coordinator <node0-address>:<port> --num-processes 3 --process-id $R

Process 0 prints the JSON result (and writes --json).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1200,640,512")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--coordinator", required=True)
    ap.add_argument("--num-processes", type=int, required=True)
    ap.add_argument("--process-id", type=int, required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--checksum", action="store_true", help="report sum(E^2) and max|E| of the final field")
    a = ap.parse_args()
    jax.distributed.initialize(a.coordinator, a.num_processes, a.process_id)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fdtdx.fdtd.fdtd import custom_fdtd_forward

    from fdtdx_box import build

    n = tuple(int(v) for v in a.n.split(","))
    t0 = time.perf_counter()
    obj, arrays, cfg = build(n, a.steps)
    setup = time.perf_counter() - t0
    steps = min(cfg.time_steps_total, a.steps)
    kwargs = dict(
        arrays=arrays,
        objects=obj,
        config=cfg,
        key=jax.random.PRNGKey(0),
        reset_container=True,
        record_detectors=False,
        start_time=0,
        end_time=steps,
        show_progress=False,
    )
    static = ("reset_container", "record_detectors", "start_time", "end_time", "show_progress")
    t0 = time.perf_counter()
    compiled = jax.jit(custom_fdtd_forward, static_argnames=static).lower(**kwargs).compile()
    compile_s = time.perf_counter() - t0
    dyn = {k: v for k, v in kwargs.items() if k not in static}
    times = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        out = compiled(**dyn)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    best = min(times)
    checksum = None
    if a.checksum:
        import jax.numpy as jnp

        e = out[1].fields.E
        checksum = [float(jax.jit(lambda x: jnp.sum(jnp.square(x)))(e)), float(jax.jit(lambda x: jnp.max(jnp.abs(x)))(e))]
    cells = n[0] * n[1] * n[2]
    res = {
        "solver": "fdtdx",
        "device": jax.devices()[0].device_kind,
        "processes": jax.process_count(),
        "devices": jax.device_count(),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        "grid": list(n),
        "cells": cells,
        "steps": steps,
        "seconds": best,
        "all_seconds": times,
        "compile_seconds": compile_s,
        "setup_seconds": setup,
        "mcells_per_s": cells * steps / best / 1e6,
        "checksum": checksum,
    }
    if jax.process_index() == 0:
        print(json.dumps(res), flush=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=2)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
