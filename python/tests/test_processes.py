"""lightsim in worker processes started with fork (multiprocessing's default
on Linux): a child forked after the parent ran a simulation must be able to
run its own, with the parent's thread-count setting."""

import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
import pytest

_SCRIPT = r"""
import json, multiprocessing as mp, sys
import numpy as np
import lightsim as ls
from lightsim import circuit as C
from lightsim.spec import C0, GaussianPulse, Medium, PointDipole, um


def fdtd():
    dx = 50e-9
    f0 = C0 / (1.55 * um)
    sim = ls.Simulation(min=[0, 0, 0], max=[3.2 * um, 2.0 * um, dx], mesh=ls.UniformMesh(dx), run_time=20e-15, background="ox",
                        boundaries={"x": [ls.PML(), ls.PML()], "y": [ls.PML(), ls.PML()], "z": [ls.PERIODIC, ls.PERIODIC]})
    sim.add_medium(Medium("ox", n=1.444))
    sim.add_source(PointDipole([0.5 * um, 1.0 * um, 0], "Ez", GaussianPulse(f0, 0.25 * f0), amplitude=1e-15))
    r = sim.run(device="cpu")
    return [float(v) for v in r.stats["energy"]], r.stats["backend"]


def circuit():
    models = {"dc": lambda wl=1.55: C.coupler_ideal(wl=wl, coupling=0.05),
              "wg": lambda wl=1.55, length=60.0: C.straight(wl=wl, neff=2.4, ng=4.2, length=length, loss_dB_cm=30.0)}
    net = {"instances": {"A": "dc", "L1": "wg"}, "connections": {"A,o3": "L1,o1", "L1,o2": "A,o2"}, "ports": {"o1": "A,o1", "o2": "A,o4"}}
    s = C.circuit(net, models, threads=2)[0](wl=np.linspace(1.5, 1.6, 301))[("o1", "o2")]
    return [float(v) for v in np.abs(s)]


def work():
    e, b = fdtd()
    return {"energy": e, "backend": b, "circuit": circuit()}


def worker(q):
    q.put(work())


if __name__ == "__main__":
    out = {"parent": work() if sys.argv[1] == "parent-first" else None}
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(q,), daemon=True) for _ in range(2)]
    for p in ps:
        p.start()
    out["children"] = [q.get(timeout=240) for _ in ps]
    for p in ps:
        p.join(60)
    if out["parent"] is None:
        out["parent"] = work()
    print(json.dumps(out))
"""


@pytest.mark.skipif(not hasattr(os, "fork") or "fork" not in mp.get_all_start_methods(), reason="needs the fork start method")
@pytest.mark.parametrize("order", ["parent-first", "children-first"])
def test_forked_workers_run_simulations(order):
    """Two forked workers each run a CPU simulation and a threaded circuit
    solve, after the parent ran both (once a hang: the children had none of
    the parent's pool threads) or before the parent ran anything. Their
    results equal the parent's, and they use the thread count the parent
    was given (RAYON_NUM_THREADS)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RAYON_NUM_THREADS="3")
    p = subprocess.run([sys.executable, "-c", _SCRIPT, order], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert "3 threads" in out["parent"]["backend"]
    for c in out["children"]:
        assert c["backend"] == out["parent"]["backend"] and c["circuit"] == out["parent"]["circuit"]
        # (the energy is a sum over threads: its last digits follow the order)
        assert np.allclose(c["energy"], out["parent"]["energy"], rtol=1e-12, atol=0)
