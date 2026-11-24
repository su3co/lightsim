"""Monte Carlo of the quantum RF PDK's resonator test chip: lightsim.circuit
against SAX, same netlist, models, frequencies and trials.

    python bench_rf.py --engine ls --trials 25 --threads 1
    python bench_rf.py --engine sax_jit_fg --trials 25
    python bench_rf.py --check            # models against the PDK's, engines against each other

The workload is the one of qpdk's notebook ``monte_carlo_fabrication_
tolerance``: the 42-instance netlist of ``resonator_test_chip_yaml``, S21 of
the top probeline at 5001 frequencies from 5 to 9 GHz, ``T`` global trials
(seed 42; width sigma 0.5 um, gap sigma 0.3 um on the 38 CPW instances), the
notches refined on 257-point local grids and the eight deepest kept. Every
engine evaluates the same arrays in the same chunks (``--chunk`` trials of
the coarse sweep per call; the refinement points of those trials in calls of
the same size). The thread count is set by the caller (``taskset``, and
``RAYON_NUM_THREADS``/``XLA_FLAGS`` from ``run_rf.sh``).

Engines
  ls          lightsim.circuit (this checkout), models of rf_models.py
  ls_f        the same, swept with ``f=`` (Hz) instead of ``wl=``
  ls_full     the same, with every input broadcast to the full batch shape
              (what the SAX engines need), so the models do the same work
  ls_reuse    ls with ``reuse_identical=True`` (the four identical launchers
              evaluated once; the models are pure)
  ls_full_reuse  ls_full with ``reuse_identical=True``
  ls_fast     ls with ``reuse_identical=True`` and ``stack_instances=True``
              (the instances of one model evaluated in one call)
  ls_full_fast   ls_full with both options
  ls_ref      lightsim.circuit from another file (``--ref-module``: the
              circuit.py of an older commit), same models
  sax_jit_fg  SAX, Filipsson-Gunnar back end, the same models, the whole
              chunk under jax.jit (compiled once per array shape)
  sax_jit_klu SAX, KLU back end (klujax), under jax.jit
  sax_fg      SAX, Filipsson-Gunnar back end, not jitted
  sax_pdk     SAX with the PDK's own models, as the notebook calls it
              (klu back end, not jitted, one call per trial)

Each run appends a JSON record to ``results/rf.jsonl`` (phases, trials,
threads, versions, host load) and stores the resonances it found in
``results/dips.npz`` under the run's name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rf_models as M  # noqa: E402

F = np.linspace(5e9, 9e9, 5001)
WIDTH_SIGMA, GAP_SIGMA, SEED = 0.5, 0.3, 42
N_RES, PROMINENCE, MIN_DISTANCE, REFINE = 8, 0.3, 30, 257


def netlist():
    import gdsfactory as gf
    from qpdk import PATH, PDK

    PDK.activate()
    chip = gf.read.from_yaml(PATH.samples / "resonator_test_chip_yaml.pic.yml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return chip.get_netlist()


def cpw_instances(nl):
    keep = ("straight", "bend_s", "quarter_wave_resonator_coupled")
    return sorted(n for n, v in nl["instances"].items() if v["component"] in keep)


def draws(n):
    rng = np.random.default_rng(SEED)
    return rng.normal(0, WIDTH_SIGMA, n), rng.normal(0, GAP_SIGMA, n)


def db(x):
    return 20 * np.log10(np.abs(np.asarray(x)))


# ---- engines: run(wl, w, g) -> S21 on the broadcast shape --------------------
class LightsimEngine:
    def __init__(self, nl, module, freq=False, full=False, threads=None, reuse=False, stack=False):
        self.names = cpw_instances(nl)
        self.freq, self.full = freq, full
        t = time.perf_counter()
        kw = {"threads": threads} if threads and "threads" in module.circuit.__code__.co_varnames else {}
        if reuse:
            kw["reuse_identical"] = True
        if stack:
            kw["stack_instances"] = True
        self.fn, _ = module.circuit(nl, M.Chip().models(), **kw)
        self.build = time.perf_counter() - t

    def run(self, wl, w, g):
        if self.full:  # the inputs SAX gets: every array on the full batch shape
            shape = np.broadcast_shapes(wl.shape, w.shape)
            wl, w, g = (np.ascontiguousarray(np.broadcast_to(x, shape)) for x in (wl, w, g))
        ov = {n: {"width": w, "gap": g} for n in self.names}
        if self.freq:
            S = self.fn(f=M.C0 / (wl * 1e-6), **ov)
        else:
            S = self.fn(wl=wl, **ov)
        return np.broadcast_to(S[("o1", "o2")], np.broadcast_shapes(wl.shape, w.shape))


class SaxEngine:
    def __init__(self, nl, backend, jit):
        import jax
        import sax

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.names = cpw_instances(nl)
        t = time.perf_counter()
        fn, _ = sax.circuit(netlist=nl, models=M.Chip().models(), on_internal_port="ignore", backend=backend)
        self.build = time.perf_counter() - t

        def s21(wl, w, g):
            return fn(wl=wl, **{n: {"width": w, "gap": g} for n in self.names})["o1", "o2"]

        self.f = jax.jit(s21) if jit else s21

    def run(self, wl, w, g):
        # SAX's back ends need every model output on the same batch shape:
        # the inputs are broadcast to the full shape
        jnp = self.jax.numpy
        shape = np.broadcast_shapes(wl.shape, w.shape)
        r = self.f(*(jnp.asarray(np.ascontiguousarray(np.broadcast_to(x, shape))) for x in (wl, w, g)))
        return np.asarray(r.block_until_ready())


class SaxPdkEngine:
    """The notebook's way: the PDK models, cross-section objects as overrides."""

    def __init__(self, nl):
        import jax
        import sax
        from qpdk.models import models
        from qpdk.tech import coplanar_waveguide

        jax.config.update("jax_enable_x64", True)
        self.xs = coplanar_waveguide
        self.names = cpw_instances(nl)
        t = time.perf_counter()
        self.fn, _ = sax.circuit(netlist=nl, models=models, on_internal_port="ignore")
        self.build = time.perf_counter() - t

    def run(self, wl, w, g):
        # one call per trial geometry (the cross-section is an object): rows
        # with the same width and gap are evaluated together, as the notebook
        # evaluates all refinement points of a trial in one call
        shape = np.broadcast_shapes(wl.shape, w.shape)
        wl, w, g = (np.broadcast_to(a, shape) for a in (wl, w, g))
        out = np.empty(shape, complex)
        groups = {}
        for r in range(shape[0]):
            groups.setdefault((float(w[r, 0]), float(g[r, 0])), []).append(r)
        for (wr, gr), rows in groups.items():
            xs = self.xs(width=wr, gap=gr)
            S = self.fn(f=(M.C0 / (wl[rows] * 1e-6)).ravel(), **{n: {"cross_section": xs} for n in self.names})
            out[rows] = np.asarray(S["o1", "o2"]).reshape(len(rows), -1)
        return out


def make_engine(name, nl, ref_module=None, threads=None):
    if name in ("ls", "ls_f", "ls_full", "ls_reuse", "ls_full_reuse", "ls_fast", "ls_full_fast"):
        from lightsim import circuit as lc

        fast = name.endswith("_fast")
        return LightsimEngine(nl, lc, freq=name == "ls_f", full=name.startswith("ls_full"), threads=threads, reuse=name.endswith("_reuse") or fast, stack=fast)
    if name == "ls_ref":
        spec = importlib.util.spec_from_file_location("circuit_ref", ref_module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return LightsimEngine(nl, mod)
    if name == "sax_jit_fg":
        return SaxEngine(nl, "filipsson_gunnar", True)
    if name == "sax_jit_klu":
        return SaxEngine(nl, "klu", True)
    if name == "sax_fg":
        return SaxEngine(nl, "filipsson_gunnar", False)
    if name == "sax_pdk":
        return SaxPdkEngine(nl)
    raise SystemExit(f"unknown engine {name}")


# ---- the Monte Carlo ------------------------------------------------------------
def monte_carlo(eng, dw, dg, chunk):
    """Coarse sweep and refinement of every trial; returns dips and phase times."""
    from scipy.signal import find_peaks

    T = len(dw)
    wl = M.wl_of_f(F)
    W, G = M.W0 + dw, M.G0 + dg
    t = time.perf_counter()
    s21 = np.empty((T, len(F)))
    for a in range(0, T, chunk):
        rows = slice(a, min(a + chunk, T))
        n = rows.stop - rows.start
        w = np.full((chunk, 1), M.W0)
        g = np.full((chunk, 1), M.G0)
        w[:n, 0], g[:n, 0] = W[rows], G[rows]
        s21[rows] = db(eng.run(wl[None, :], w, g)[:n])
    t_coarse = time.perf_counter() - t
    # refinement: every candidate notch of every trial on a 257-point grid,
    # in calls of `rows` candidates (the same number of points as a coarse call)
    t = time.perf_counter()
    df = F[1] - F[0]
    cand, owner = [], []
    for k in range(T):
        p, _ = find_peaks(-s21[k], distance=MIN_DISTANCE)
        cand.append(p)
        owner += [k] * len(p)
    owner = np.asarray(owner, dtype=int)
    centres = np.concatenate([F[p] for p in cand]) if cand else np.zeros(0)
    fr = centres[:, None] + df * np.linspace(-1, 1, REFINE)[None, :]
    rows = max(1, chunk * len(F) // REFINE)
    vals = np.empty(fr.shape)
    for a in range(0, len(fr), rows):
        sl = slice(a, min(a + rows, len(fr)))
        n = sl.stop - sl.start
        x = np.full((rows, REFINE), F[0])
        w = np.full((rows, 1), M.W0)
        g = np.full((rows, 1), M.G0)
        x[:n], w[:n, 0], g[:n, 0] = fr[sl], W[owner[sl]], G[owner[sl]]
        vals[sl] = db(eng.run(M.wl_of_f(x), w, g)[:n])
    # the notebook's selection: deepest notches above the prominence
    dips = np.full((T, N_RES), np.nan)
    for k in range(T):
        idx = np.nonzero(owner == k)[0]
        if not len(idx):
            continue
        v = vals[idx]
        m = np.argmin(v, axis=1)
        depth = np.maximum(v[:, 0], v[:, -1]) - v[np.arange(len(idx)), m]
        sel = np.argsort(depth)[::-1][:N_RES]
        sel = sel[depth[sel] >= PROMINENCE]
        d = np.sort(fr[idx[sel], m[sel]])
        dips[k, : len(d)] = d
    t_refine = time.perf_counter() - t
    return dips, s21, {"coarse": t_coarse, "refine": t_refine, "candidates": int(len(fr))}


def versions():
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for m in ("lightsim", "sax", "jax", "jaxlib", "klujax", "qpdk", "gdsfactory", "scipy"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "")
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="ls")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--threads", type=int, default=0, help="recorded; set with taskset and the environment")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ref-module", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(HERE / "results" / "rf.jsonl"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    nl = netlist()
    if a.check:
        return check(nl, a)
    import lightsim.circuit  # noqa: F401  (imports are not part of the build time)

    if a.engine.startswith("sax"):
        import jax  # noqa: F401
        import sax  # noqa: F401
    t0 = time.perf_counter()
    eng = make_engine(a.engine, nl, a.ref_module, a.threads or None)
    build = time.perf_counter() - t0
    # first call: one coarse chunk at the nominal geometry (compiles the jitted engines)
    t = time.perf_counter()
    eng.run(M.wl_of_f(F)[None, :], np.full((a.chunk, 1), M.W0), np.full((a.chunk, 1), M.G0))
    first = time.perf_counter() - t
    dw, dg = draws(a.trials)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for rep in range(a.repeats):
        la0 = os.getloadavg()[0]
        t = time.perf_counter()
        dips, s21, ph = monte_carlo(eng, dw, dg, a.chunk)
        total = time.perf_counter() - t
        rec = {
            "workload": "rf_chip_mc",
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "engine": a.engine,
            "label": a.label,
            "trials": a.trials,
            "chunk": a.chunk,
            "threads": a.threads,
            "cpus_visible": len(os.sched_getaffinity(0)),
            "machine": platform.machine(),
            "load_before": la0,
            "load_after": os.getloadavg()[0],
            "build": build,
            "first_call": first,
            "mc": total,
            "repeat": rep,
            **ph,
            "per_trial": total / a.trials,
            "versions": versions(),
            "env": {k: os.environ.get(k, "") for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
        }
        tag = f"{a.engine}_{a.trials}_{a.threads}_{a.label or 'x'}"
        store = Path(a.out).parent / "dips.npz"
        sets = dict(np.load(store)) if store.exists() else {}
        sets[tag] = dips
        np.savez_compressed(store, **sets)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("engine", "trials", "threads", "build", "first_call", "mc", "coarse", "refine", "per_trial")}), flush=True)


def check(nl, a):
    """The models against the PDK's (whole circuit, nominal) and every engine
    against lightsim's dense solve (a few trials), largest differences."""
    import jax
    import sax
    from qpdk.models import models

    jax.config.update("jax_enable_x64", True)
    from lightsim import circuit as lc

    wl = M.wl_of_f(F)
    pdk, _ = sax.circuit(netlist=nl, models=models, on_internal_port="ignore")
    Sp = pdk(f=F)
    fn, _ = lc.circuit(nl, M.Chip().models())
    Sl = fn(wl=wl)
    Sf = fn(f=F)
    for key in [("o1", "o2"), ("o1", "o1"), ("o3", "o4")]:
        d = np.abs(np.asarray(Sp[key]) - Sl[key]).max()
        print(f"models vs PDK {key}: max |dS| = {d:.2e}  (|S| max {np.abs(Sl[key]).max():.3f})")
    print(f"lightsim f= vs wl=: {max(np.abs(Sf[k] - Sl[k]).max() for k in Sl):.2e}")
    dw, dg = draws(3)
    w, g = (M.W0 + dw)[:, None], (M.G0 + dg)[:, None]
    ref = lc.circuit(nl, M.Chip().models(), backend="dense")[0]
    names = cpw_instances(nl)
    Sr = ref(wl=wl[None, :], **{n: {"width": w, "gap": g} for n in names})
    for e in ("ls", "sax_jit_fg", "sax_jit_klu", "sax_fg"):
        eng = make_engine(e, nl)
        s = eng.run(wl[None, :], w, g)
        d = np.abs(s - Sr[("o1", "o2")]).max()
        print(f"{e} vs lightsim dense: max |dS21| = {d:.2e}")


if __name__ == "__main__":
    main()
