"""Write docs/comparison.md from the committed result files of the
benchmark groups (one JSON record per line). Run after adding results:

    python scripts/make_comparison.py

A CPU timing is used only if the host's one-minute load was at most
``MAX_LOAD`` before and after it (other users' jobs make timings invalid);
records above it stay in the files but not in the tables.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "benchmarks" / "circuit" / "results"
MAX_LOAD = 8.0
#: label prefixes of the runs shown, per CPU type (the final measurement of
#: each machine; earlier rounds stay in the files)
FINAL_LABELS = {"aarch64": ("gb4",), "x86_64": ("x86final",)}

MACHINES = {"aarch64": "GB10 (20-core Arm CPU: 10 Cortex-X925 + 10 Cortex-A725)", "x86_64": "Threadripper 9980X (64 cores)"}
ENGINES = {
    "ls_fast": "lightsim, `reuse_identical` + `stack_instances`",
    "ls_reuse": "lightsim, `reuse_identical`",
    "ls": "lightsim, defaults",
    "ls_ref": "lightsim before (dffd2bf, dense solve)",
    "sax_jit_fg": "SAX jitted, Filipsson-Gunnar",
    "sax_jit_klu": "SAX jitted, KLU",
    "sax_fg": "SAX not jitted, Filipsson-Gunnar",
    "sax_pdk": "SAX + PDK models, as the notebook (not jitted, KLU)",
}


def load(name):
    p = RES / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def quiet(r):
    return r.get("load_before", r.get("load", 0)) <= MAX_LOAD and r.get("load_after", r.get("load", 0)) <= MAX_LOAD


def fmt(x, unit="s"):
    if x is None:
        return "–"
    if unit == "s":
        return f"{x:.3f} s" if x < 10 else f"{x:.1f} s"
    if unit == "ms":
        return f"{x * 1e3:.2f} ms" if x < 0.1 else f"{x * 1e3:.0f} ms"
    return str(x)


def rf_section():
    recs = [r for r in load("rf.jsonl") if quiet(r) and r.get("trials") == 25 and r.get("label", "").startswith(FINAL_LABELS.get(r["machine"], ("-",)))]
    out = []
    groups = defaultdict(list)
    for r in recs:
        groups[(r["machine"], r["threads"], r["engine"], r["chunk"])].append(r)
    for machine in sorted({k[0] for k in groups}):
        for threads in sorted({k[1] for k in groups if k[0] == machine}):
            rows = []
            for eng in ENGINES:
                cands = [(k, v) for k, v in groups.items() if k[0] == machine and k[1] == threads and k[2] == eng]
                if not cands:
                    continue
                best = None
                for k, v in cands:
                    v = sorted(v, key=lambda r: r["repeat"])
                    first = v[0]
                    steady = [r["mc"] for r in v[1:]] if (eng.startswith("sax_jit") and len(v) > 1) else [r["mc"] for r in v]
                    med = st.median(steady)
                    item = (med, k[3], first, min(steady), max(steady), len(steady))
                    if best is None or med < best[0]:
                        best = item
                med, chunk, first, lo, hi, n = best
                whole = first["build"] + first["first_call"] + first["mc"]
                rows.append((eng, chunk, first["build"] + first["first_call"], med, lo, hi, n, whole))
            if not rows:
                continue
            out.append(f"\n**{MACHINES.get(machine, machine)}, {threads} thread{'s' if threads > 1 else ''}** (pinned)\n")
            out.append("| engine | trials per call | set-up and first call | 25 trials, steady (median, range, runs) | whole job: set-up + first 25 trials |")
            out.append("|---|---|---|---|---|")
            for eng, chunk, setup, med, lo, hi, n, whole in sorted(rows, key=lambda x: x[3]):
                out.append(f"| {ENGINES[eng]} | {chunk} | {fmt(setup)} | {fmt(med)} ({fmt(lo)}–{fmt(hi)}, {n}) | {fmt(whole)} |")
    return "\n".join(out)


def rf_scaling_section():
    recs = [r for r in load("rf.jsonl") if quiet(r) and any(t in r.get("label", "") for t in ("strong", "weak")) and r.get("label", "").startswith(tuple(p for v in FINAL_LABELS.values() for p in v) + ("x86",))]
    if not recs:
        return "_No Monte Carlo scaling records yet._"
    out = []
    for machine in sorted({r["machine"] for r in recs}):
        for kind in ("strong", "weak"):
            rs = [r for r in recs if r["machine"] == machine and kind in r["label"]]
            if not rs:
                continue
            threads = sorted({r["threads"] for r in rs})
            what = "250 trials" if kind == "strong" else "25 trials per thread"
            out.append(f"\n**{MACHINES.get(machine, machine)}, Monte Carlo {kind} scaling** ({what}, 25 trials per call; steady time, parallel efficiency in brackets)\n")
            out.append("| engine | " + " | ".join(f"{t} thr" for t in threads) + " |")
            out.append("|---|" + "---|" * len(threads))
            for eng in ENGINES:
                er = [r for r in rs if r["engine"] == eng]
                if not er:
                    continue
                best = {}
                for t in threads:
                    tr = sorted([r for r in er if r["threads"] == t], key=lambda r: r["repeat"])
                    if tr:
                        steady = [r["mc"] for r in tr[1:]] or [tr[0]["mc"]]
                        best[t] = min(steady)
                t0 = min(best) if best else None
                cells = []
                for t in threads:
                    if t not in best:
                        cells.append("–")
                        continue
                    eff = (best[t0] / best[t] / (t / t0)) if kind == "strong" else (best[t0] / best[t])
                    cells.append(f"{fmt(best[t])} ({eff:.2f})")
                out.append(f"| {ENGINES[eng]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def gpu_section():
    recs = load("rf_gpu.jsonl")
    if not recs:
        return "_No GPU records yet._"
    out = ["| engine | trials | trials per call | set-up and first call | Monte Carlo, steady | per trial | host load |", "|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["engine"], r["trials"], r["chunk"], r.get("label", ""))].append(r)
    for (eng, T, ch, lab), v in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        v = sorted(v, key=lambda r: r["repeat"])
        steady = [r["mc"] for r in v[1:]] or [v[0]["mc"]]
        m = min(steady)
        out.append(f"| {ENGINES.get(eng, eng)} ({lab}) | {T} | {ch} | {fmt(v[0]['build'] + v[0]['first_call'])} | {fmt(m)} | {fmt(m / T, 'ms')} | {v[0]['load_before']:.0f} |")
    return "\n".join(out)


def photonic_section():
    recs = [r for r in load("photonic.jsonl") if quiet(r) and r.get("label", "").startswith(FINAL_LABELS.get(r["machine"], ("-",)))]
    if not recs:
        return "_No photonic records yet._"
    eng = [("ls", "lightsim"), ("ls_fast", "lightsim, reuse + stack"), ("ls_ref", "lightsim before"), ("sax", "SAX"), ("sax_jit", "SAX jitted"), ("sax_jit_fg", "SAX jitted, F-G")]
    out = []
    for machine, threads in sorted({(r["machine"], r["threads"]) for r in recs}):
        rs = [r for r in recs if r["machine"] == machine and r["threads"] == threads]
        out.append(f"\n**{MACHINES.get(machine, machine)}, {threads} thread{'s' if threads > 1 else ''}** (pinned). Repeated call (median); first call in brackets where it differs.\n")
        out.append("| circuit | wavelengths | " + " | ".join(n for _, n in eng) + " | lightsim faster than the fastest SAX by | largest difference (SAX vs lightsim) |")
        out.append("|---|---|" + "---|" * (len(eng) + 2))
        for wl in sorted({(r["workload"], r["nw"]) for r in rs}):
            cells, best_sax, ls_t, diff = [], None, None, 0.0
            for e, _ in eng:
                m = [r for r in rs if (r["workload"], r["nw"]) == wl and r["engine"] == e]
                if not m:
                    cells.append("–")
                    continue
                r = min(m, key=lambda r: r["repeat_median"])
                t, f = r["repeat_median"], r["build"] + r["first_call"]
                cells.append(fmt(t, "ms") + (f" ({fmt(f, 's')})" if f > 3 * t + 0.05 else ""))
                if e == "ls":
                    ls_t = t
                if e.startswith("sax"):
                    best_sax = t if best_sax is None else min(best_sax, t)
                    diff = max(diff, r["max_diff_vs_dense"])
            ratio = f"{best_sax / ls_t:.1f}x" if (best_sax and ls_t) else "–"
            out.append(f"| {wl[0].replace('photonic_', '')} | {wl[1]} | " + " | ".join(cells) + f" | {ratio} | {diff:.1e} |")
    return "\n".join(out)


def generality_section():
    recs = [r for r in load("generality.jsonl") if "final" in r.get("label", "")]
    if not recs:
        return "_No generality records yet._"
    out = []
    for lab in sorted({r["label"] for r in recs}):
        rs = [r for r in recs if r["label"] == lab]
        sp = sorted(r["speedup_auto"] for r in rs)
        errs = [r.get("err_auto", 0) for r in rs]
        worst = min(rs, key=lambda r: r["speedup_auto"])
        out.append(f"\n**{lab}**: {len(rs)} cases; speed-up of `auto` over the dense solve: median {st.median(sp):.1f}x, worst {sp[0]:.2f}x ({worst['case']} {worst['size']}, {worst['nw']} points), best {sp[-1]:.0f}x; largest relative difference {max(errs):.1e}.\n")
        out.append("| topology | cases | median speed-up | worst | best | largest block | largest difference |")
        out.append("|---|---|---|---|---|---|---|")
        for case in dict.fromkeys(r["case"] for r in rs):
            c = [r for r in rs if r["case"] == case]
            s = sorted(r["speedup_auto"] for r in c)
            out.append(f"| {case} | {len(c)} | {st.median(s):.1f}x | {s[0]:.2f}x | {s[-1]:.0f}x | {max((r['max_block'] or 0) for r in c)} | {max(r.get('err_auto', 0) for r in c):.1e} |")
    return "\n".join(out)


def scaling_section():
    recs = [r for r in load("scaling.jsonl") if "final" in r.get("label", "")]
    if not recs:
        return "_No scaling records yet._"
    out = []
    for lab in sorted({r["label"] for r in recs}):
        rs = [r for r in recs if r["label"] == lab]
        threads = sorted({r["threads"] for r in rs})
        for kind in ("strong", "weak"):
            out.append(f"\n**{lab}, {kind} scaling** ({'fixed batch' if kind == 'strong' else '2000 points per thread'}; parallel efficiency, time in ms)\n")
            out.append("| case | " + " | ".join(f"{t} thr" for t in threads) + " |")
            out.append("|---|" + "---|" * len(threads))
            for case in dict.fromkeys(r["case"] for r in rs):
                cells = []
                for t in threads:
                    m = [r for r in rs if r["case"] == case and r["kind"] == kind and r["threads"] == t]
                    cells.append(f"{m[-1]['efficiency']:.2f} ({m[-1]['time'] * 1e3:.1f})" if m else "–")
                out.append(f"| {case} | " + " | ".join(cells) + " |")
    return "\n".join(out)


PAGE = """# Speed against other packages

This page compares lightsim with open packages that do the same job, on the
same machine with the same inputs; results are checked equal before the
timings count. It is generated by `scripts/make_comparison.py` from the
result files in `benchmarks/*/results/`, which the scripts in the same
folders write. A timing taken while other jobs loaded the host
(one-minute load above {max_load:g}) is kept in the files but left out of
the tables. Thread counts are stated per table; processes are pinned with
`taskset` to physical cores (on the GB10, to its Cortex-X925 cores) and
BLAS, OpenMP and rayon are given the same thread count; XLA sizes its pool
from the same affinity mask. GB10 timings, CPU ones included, run inside a
reservation of the node, which pauses its idle GPU workload. Every run is
sampled about once a second by `scripts/monitor.py` (cores used,
threads, memory, context switches, and per card utilisation, memory and
power); the samples are in `results/mon/`.

## Circuits: lightsim.circuit against SAX

### Microwave Monte Carlo: the quantum RF PDK's resonator chip

The workload of qpdk's notebook `monte_carlo_fabrication_tolerance`
(`benchmarks/circuit/bench_rf.py`): the 42-instance netlist of
`resonator_test_chip_yaml` (16 quarter-wave resonators on two probelines),
S21 at 5001 frequencies from 5 to 9 GHz, 25 global trials of the CPW width
and gap of 38 instances, and every notch refined on a 257-point grid (about
450 refinement grids). Both engines run the same models
(`rf_models.py`: the PDK's physics written once for numpy and jax.numpy;
against the PDK's own models the circuit S-parameters agree to 1.5e-11),
the same netlist, frequencies and draws, in calls of the same size; each
engine is shown at the call size (trials per call) that suits it best.
lightsim's S21 equals its dense solve to 2.6e-15 and SAX's to 1.4e-11 (the
difference comes from the elliptic integrals inside the models: scipy for
numpy, jaxellip for jax); the resonance frequencies found agree to well
below the 0.8 MHz grid step. Versions: SAX 0.18.2, JAX and jaxlib 0.9.2,
klujax 0.5.3, qpdk 0.4.3, gdsfactory 9.51.0, numpy 2.5.5, scipy 1.18.1,
Python 3.12, double precision throughout.

{rf}

"Set-up and first call" is building the circuit plus one call of the first
size (for jitted SAX, the compilation; SAX compiles again for each new array
shape, which the refinement needs). "Whole job" is what a user waits for in
a fresh process.

**Scaling over threads, the whole Monte Carlo** (models, solves and the
search for the notches; pinned cores, the fast cores first on the GB10):

{rf_scaling}

**On a GPU.** SAX with JAX's CUDA build on one RTX PRO 6000 (lightsim's
circuit solver runs on the CPU):

{gpu}

### Photonic circuits from gdsfactory netlists

gplugins' `test_mzi` models on gdsfactory 9 netlists
(`benchmarks/circuit/bench_photonic.py`).

{photonic}

### The same solver on circuits it was not tuned on

`benchmarks/circuit/generality.py` builds a seeded spread of
topologies (chains, splitter trees, rings, Clements meshes, 2-D lattices,
random graphs of 2- to 6-ports, two-mode chains, many-port stars) from 2 to
3000 instances and 1 to 20,000 batch points, with random passive
S-matrices, and compares `solve_network`'s default (`auto`) with the dense
solve it replaced (results and time; the instance S-parameters are built
once).

{generality}

### Scaling of the network solve over threads

`benchmarks/circuit/scaling.py` (solve only, instance S-parameters
built once).

{scaling}
"""


def main():
    page = PAGE.format(max_load=MAX_LOAD, rf=rf_section(), rf_scaling=rf_scaling_section(), gpu=gpu_section(), photonic=photonic_section(), generality=generality_section(), scaling=scaling_section())
    (ROOT / "docs" / "comparison.md").write_text(page)
    print("wrote docs/comparison.md")


if __name__ == "__main__":
    main()
