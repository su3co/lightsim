"""Photonic circuits from gdsfactory netlists: lightsim.circuit against SAX.

    python bench_photonic.py --engine ls --circuit mzi --nw 1000 --threads 1

Circuits (gdsfactory 9 generic PDK): ``mzi`` (``gf.components.mzi()``),
``mzi4`` (four MZIs of increasing arm imbalance in series, recursive
netlist), ``ring`` (``gf.components.ring_single()``, recursive netlist) and
``lattice`` (``gf.components.mzi_lattice()`` with four couplers, recursive).
Models: gplugins' ``test_mzi`` models (straight, Euler bend, ideal 1x2
splitter) plus a ring coupler and a directional coupler, written once with
numpy or jax.numpy (the array type of the call decides), so both engines
run the same formulas.

Engines: ``ls`` (this checkout), ``ls_fast`` (with ``reuse_identical`` and
``stack_instances``: the models are pure and broadcast), ``ls_trace`` (those
two and ``trace_models``), ``ls_ref`` (``--ref-module``: the
circuit.py of an older commit), ``sax`` (SAX's default back end, not
jitted), ``sax_jit`` (default back end under jax.jit), ``sax_jit_fg``
(Filipsson-Gunnar back end under jax.jit). Recorded: build, first call
(compilation for the jitted engines), repeated calls (median and spread of
at least 10 calls or 1 s), and the largest difference against lightsim's
dense solve.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _xp(*args):
    for a in args:
        if type(a).__module__.startswith("jax"):
            import jax.numpy as jnp

            return jnp
    return np


def runnable():
    """Instantaneous count of runnable threads on the host (/proc/loadavg), or None."""
    try:
        with open("/proc/loadavg") as fh:
            return int(fh.read().split()[3].split("/")[0])
    except (OSError, ValueError, IndexError):
        return None


def reciprocal(sd):
    out = dict(sd)
    for (a, b), v in sd.items():
        out.setdefault((b, a), v)
    return out


# gplugins/sax/tests/test_mzi.py, array type chosen by the call
def straight(wl=1.5, length=10.0, neff=2.4):
    xp = _xp(wl)
    return reciprocal({("o1", "o2"): xp.exp(2j * xp.pi * neff * length / wl)})


def mmi1x2():
    return reciprocal({("o1", "o2"): 0.5**0.5, ("o1", "o3"): 0.5**0.5})


def bend_euler(wl=1.5, length=20.0):
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


def coupler(wl=1.5, coupling=0.5):
    xp = _xp(wl)
    one = xp.ones_like(xp.asarray(wl, dtype=float))
    k, t = (coupling**0.5) * one, ((1 - coupling) ** 0.5) * one
    return reciprocal({("o1", "o4"): t + 0j, ("o2", "o3"): t + 0j, ("o1", "o3"): 1j * k, ("o2", "o4"): 1j * k})


def coupler_ring(wl=1.5, coupling=0.1):
    return coupler(wl=wl, coupling=coupling)


MODELS = {"straight": straight, "bend_euler": bend_euler, "mmi1x2": mmi1x2, "coupler": coupler, "coupler_ring": coupler_ring, "mmi2x2": coupler}


def netlist(name):
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "mzi":
            return gf.components.mzi().get_netlist()
        if name == "mzi4":

            @gf.cell
            def mzi4_chain():
                c = gf.Component()
                refs = [c << gf.components.mzi(delta_length=10.0 * (k + 1)) for k in range(4)]
                for a, b in zip(refs, refs[1:]):
                    b.connect("o1", a.ports["o2"])
                c.add_port("o1", port=refs[0].ports["o1"])
                c.add_port("o2", port=refs[-1].ports["o2"])
                return c

            return mzi4_chain().get_netlist(recursive=True)
        if name == "ring":
            return gf.components.ring_single().get_netlist(recursive=True)
        if name == "lattice":
            return gf.components.mzi_lattice(coupler_lengths=(10.0, 20.0, 30.0, 40.0), coupler_gaps=(0.2, 0.3, 0.4, 0.5), delta_lengths=(10.0, 20.0, 30.0)).get_netlist(recursive=True)
    raise SystemExit(name)


class LS:
    def __init__(self, nl, module, threads=None, fast=False, trace=False):
        kw = {"threads": threads} if threads and "threads" in module.circuit.__code__.co_varnames else {}
        if fast:
            kw.update(reuse_identical=True, stack_instances=True)
        if trace:
            kw.update(trace_models=True)
        self.fn, _ = module.circuit(nl, MODELS, **kw)

    def __call__(self, wl):
        return self.fn(wl=wl)


class Sax:
    def __init__(self, nl, backend, jit):
        import jax
        import sax

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        fn, _ = sax.circuit(netlist=nl, models=MODELS, backend=backend)
        self.fn = jax.jit(lambda wl: fn(wl=wl)) if jit else (lambda wl: fn(wl=wl))

    def __call__(self, wl):
        S = self.fn(self.jax.numpy.asarray(wl))
        return {k: np.asarray(v.block_until_ready()) for k, v in S.items()}


def make(engine, nl, ref_module=None, threads=None):
    if engine in ("ls", "ls_fast", "ls_trace"):
        from lightsim import circuit as lc

        return LS(nl, lc, threads, fast=engine != "ls", trace=engine == "ls_trace")
    if engine == "ls_ref":
        spec = importlib.util.spec_from_file_location("circuit_ref", ref_module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return LS(nl, mod)
    if engine == "sax":
        return Sax(nl, "default", False)
    if engine == "sax_jit":
        return Sax(nl, "default", True)
    if engine == "sax_jit_fg":
        return Sax(nl, "filipsson_gunnar", True)
    raise SystemExit(engine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="ls")
    ap.add_argument("--circuit", default="mzi")
    ap.add_argument("--nw", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ref-module", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(HERE / "results" / "photonic.jsonl"))
    a = ap.parse_args()
    nl = netlist(a.circuit)
    wl = np.linspace(1.5, 1.6, a.nw)
    # imports are not part of the build time
    import lightsim.circuit  # noqa: F401

    if a.engine.startswith("sax"):
        import jax  # noqa: F401
        import sax  # noqa: F401
    la0, ru0 = os.getloadavg()[0], runnable()
    t = time.perf_counter()
    eng = make(a.engine, nl, a.ref_module, a.threads or None)
    build = time.perf_counter() - t
    t = time.perf_counter()
    S = eng(wl)
    first = time.perf_counter() - t
    times = []
    while len(times) < 10 or (sum(times) < 1.0 and len(times) < 2000):
        # wavelengths shifted by a negligible amount on every call, so that
        # every engine does the whole evaluation (reuse_identical would
        # otherwise return the previous call's model results)
        x = wl + (len(times) + 1) * 1e-13
        t = time.perf_counter()
        eng(x)
        times.append(time.perf_counter() - t)
    from lightsim import circuit as lc

    ref = lc.circuit(nl, MODELS, backend="dense")[0](wl=wl)
    keys = [k for k in ref if k in S]
    err = max(float(np.max(np.abs(np.asarray(S[k]) - ref[k]))) for k in keys)
    rec = {
        "workload": f"photonic_{a.circuit}",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": a.engine,
        "label": a.label,
        "nw": a.nw,
        "threads": a.threads,
        "cpus_visible": len(os.sched_getaffinity(0)),
        "machine": platform.machine(),
        "load": os.getloadavg()[0],
        "load_before": la0,
        "device": (str(__import__("jax").devices()[0]) if a.engine.startswith("sax") else "cpu"),
        "runnable_before": ru0,
        "runnable_after": runnable(),
        "build": build,
        "first_call": first,
        "repeat_median": float(np.median(times)),
        "repeat_p10": float(np.percentile(times, 10)),
        "repeat_p90": float(np.percentile(times, 90)),
        "repeats": len(times),
        "max_diff_vs_dense": err,
        "ports": len({p for k in ref for p in k}),
        "env": {k: os.environ.get(k, "") for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS")},
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("workload", "engine", "nw", "threads", "build", "first_call", "repeat_median", "max_diff_vs_dense")}), flush=True)


if __name__ == "__main__":
    main()
