"""Smoke test of an installed lightsim wheel: import, devices, a tiny run.

    python smoke_test.py            # any build, on a machine without a GPU
    python smoke_test.py --gpu      # also run on cuda:0 and compare with the CPU

Without --gpu it checks that the machine has no usable GPU from lightsim's
point of view: devices() lists the CPU only, device="auto" runs on the CPU
and device="cuda:0" raises an error that says why.
"""

import argparse
import sys
import time

import numpy as np

import lightsim as ls
from lightsim.spec import um


def tiny_sim():
    f0 = ls.C0 / (1.0 * um)
    sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 1 * um, 1 * um], mesh=ls.UniformMesh(50e-9), run_time=30e-15)
    sim.add_source(ls.PointDipole([1 * um, 0.5 * um, 0.5 * um], "Ez", ls.GaussianPulse(f0, 0.3 * f0), name="dipole"))
    sim.add_monitor(ls.DFTMonitor("slice", [0, 0, 0.5 * um], [2 * um, 1 * um, 0.5 * um], [f0], components=("Ez",)))
    return sim


def run(device):
    t = time.perf_counter()
    res = tiny_sim().run(device=device)
    ez = res.field("slice", "Ez")
    print(f"  device={device!r}: backend {res.stats['backend']!r}, {res.stats['cells']} cells, {res.stats['steps']} steps, {time.perf_counter() - t:.2f} s")
    assert np.all(np.isfinite(ez)) and np.abs(ez).max() > 0
    return res, ez


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="expect a GPU and run on cuda:0")
    args = ap.parse_args()

    print(f"lightsim {ls.__version__} ({ls.precision}) from {ls.__file__}")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
    devs = ls.devices()
    print(f"devices: {devs}")
    assert devs[0] == "cpu"
    assert ls.precision == "f32"

    print("runs:")
    _, ez_cpu = run("cpu")
    res_auto, ez_auto = run("auto")

    if args.gpu:
        assert any(d.startswith("cuda:0") for d in devs), "no GPU listed"
        assert res_auto.stats["backend"].lower().startswith("cuda") or "gpu" in res_auto.stats["backend"].lower(), res_auto.stats["backend"]
        _, ez_gpu = run("cuda:0")
        err = np.abs(ez_gpu - ez_cpu).max() / np.abs(ez_cpu).max()
        print(f"  max |Ez(cuda:0) - Ez(cpu)| / max |Ez(cpu)| = {err:.2e}")
        assert err < 1e-3, err
    else:
        assert devs == ["cpu"], devs
        assert res_auto.stats["backend"].startswith("cpu"), res_auto.stats["backend"]
        assert np.array_equal(ez_auto, ez_cpu)
        try:
            tiny_sim().run(device="cuda:0")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f'  device="cuda:0" raised {type(e).__name__}: {msg}')
            assert any(w in msg.lower() for w in ("cuda", "gpu", "driver")), msg
        else:
            raise AssertionError('device="cuda:0" ran without a GPU')
    print("smoke test passed")


if __name__ == "__main__":
    main()
