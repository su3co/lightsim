"""Write docs/comparison.md from the committed result files of the
benchmark groups (one JSON record per line). Run after adding results:

    python scripts/make_comparison.py

A CPU timing is used only if other jobs kept at most ``MAX_LOAD``
processes runnable besides the benchmark's own before and after it (for
records without that count, the host's one-minute load); records above it
stay in the files but not in the tables.
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
FINAL_LABELS = {"aarch64": ("gb5",), "x86_64": ("x86final",)}
#: the same for the resonator chip's per-engine tables (re-timed after the
#: circuit solver's later changes; the scaling tables keep FINAL_LABELS)
TABLE_LABELS = {"aarch64": ("gb8",), "x86_64": ("x86final8",)}

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


LABEL_NAMES = {
    "gb10_t1": "GB10, 1 thread on a Cortex-X925 core",
    "gb10_t10": "GB10, 10 threads on the Cortex-X925 cores",
    "gb10_x925": "GB10, the 10 Cortex-X925 cores",
    "gb10_all": "GB10, all 20 cores (10 Cortex-X925 + 10 Cortex-A725)",
    "x86_64cores": "Threadripper 9980X, up to 64 cores",
}


def nice(label: str) -> str:
    """What a result set is, from its run label (labels are internal)."""
    for k in sorted(LABEL_NAMES, key=len, reverse=True):
        if label.startswith(k):
            return LABEL_NAMES[k]
    return label


def load(name):
    p = RES / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def quiet(r):
    """Other jobs' load at the start and end of a run: the processes
    runnable besides the benchmark's own when the record has that count
    (the one-minute load also counts the benchmark's threads), else the
    one-minute load."""
    if "runnable_before" in r:
        return r["runnable_before"] - 1 <= MAX_LOAD and r.get("runnable_after", 1) - 1 <= MAX_LOAD
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
    recs = [r for r in load("rf.jsonl") if quiet(r) and r.get("trials") == 25 and r.get("label", "").startswith(TABLE_LABELS.get(r["machine"], ("-",)))]
    monitors = load("monitor.jsonl")
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
            out.append("| engine | trials per call | set-up and first call | 25 trials, steady (median, range, runs) | whole job: set-up + first 25 trials | cores busy (mean of the run) |")
            out.append("|---|---|---|---|---|---|")
            for eng, chunk, setup, med, lo, hi, n, whole in sorted(rows, key=lambda x: x[3]):
                mon = [m for m in monitors if m.get("engine") == eng and m.get("threads") == threads and m.get("chunk") == chunk and m.get("trials") == 25 and str(m.get("label", "")).startswith(TABLE_LABELS.get(machine, ("-",)))]
                busy = f"{st.median([m['mean_cores_used'] for m in mon if m.get('mean_cores_used') is not None]):.2f} of {threads}" if mon else "–"
                out.append(f"| {ENGINES[eng]} | {chunk} | {fmt(setup)} | {fmt(med)} ({fmt(lo)}–{fmt(hi)}, {n}) | {fmt(whole)} | {busy} |")
            if machine == "aarch64" and threads == 8:
                out.append("\n" + layout_note())
    return "\n".join(out)


def layout_note():
    """The refinement's batching (geometry per candidate row or per point) and
    the traced models, measured back to back on one GB10 (label gb10_trace)."""
    recs = [r for r in load("layout.jsonl") if quiet(r) and r.get("label") == "gb10_trace"]
    if not recs:
        return ""
    cells = defaultdict(list)
    for r in recs:
        if r["repeat"] == 0:  # the first repeat compiles the jitted engines' refinement shapes
            continue
        cells[(r["threads"], r["layout"], r["engine"], r["chunk"])].append(r["mc"])
    med = {k: st.median(v) for k, v in cells.items()}

    def best(t, lay, eng):
        c = {k[3]: v for k, v in med.items() if k[:3] == (t, lay, eng)}
        return min(c.values()) if c else None

    names = {"ls_fast": "lightsim, reuse + stack", "ls_trace_fast": "lightsim, reuse + stack + `trace_models`", "sax_jit_fg": "SAX jitted, F-G"}
    out = [
        "How the refinement is batched matters to lightsim's untraced models and not to SAX. The tables "
        "above give the width and gap per candidate row (257 points share one geometry). If instead all "
        "refinement points are packed into calls of 25 x 5001 points with the geometry given per point, "
        "as the examples repository's harness does, the numpy models compute the CPW parameters "
        "(elliptic integrals, the coupling table) at every point. With `trace_models` the per-point work "
        "of the models runs inside the compiled solve, and lightsim is ahead of SAX in both layouts. "
        "Measured back to back on one GB10 (node claimed), 25 trials, each engine at its best number of "
        "trials per call (5 or 25; every call size is in `results/layout.jsonl`), median of the steady repeats:",
        "",
        "| threads | refinement layout | " + " | ".join(names.values()) + " |",
        "|---|---|---|---|---|",
    ]
    for t in (1, 8):
        for lay, lname in (("rows", "geometry per row"), ("points", "geometry per point")):
            out.append(f"| {t} | {lname} | " + " | ".join(fmt(best(t, lay, e)) for e in names) + " |")
    slow = [(k, v) for k, v in med.items() if k[2] == "ls_trace_fast" and (k[0], k[1], "ls_fast", k[3]) in med and v > med[(k[0], k[1], "ls_fast", k[3])]]
    for (t, lay, _, ch), v in slow:
        out.append(
            f"\nOne configuration is slower with `trace_models`: {t} thread{'s' if t > 1 else ''}, geometry per "
            f"{'point' if lay == 'points' else 'row'}, {ch} trials per call ({fmt(v)} against {fmt(med[(t, lay, 'ls_fast', ch)])} untraced)."
        )
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
    # SAX rows only from runs whose record names a CUDA device (SAX's dependency
    # klujax switches JAX to the CPU on import; such runs are not GPU rows);
    # lightsim's rows are its CPU path on the same host, same job
    recs = [r for r in load("rf_gpu.jsonl") if quiet(r) and r.get("trials") == 2500 and r.get("chunk") == 250]
    recs = [r for r in recs if str(r.get("device", "")).startswith("cuda") or r["engine"].startswith("ls")]
    if not any(str(r.get("device", "")).startswith("cuda") for r in recs):
        return "lightsim's circuit solver runs on the CPU only; SAX with JAX's CUDA build has not been measured cleanly yet."
    groups = defaultdict(list)
    for r in recs:
        gpu = str(r.get("device", "")).startswith("cuda")
        groups[(r["machine"], r["engine"], gpu, 0 if gpu else int(r["threads"]))].append(r)
    gpu_name = {"x86_64": "RTX PRO 6000 Blackwell Max-Q", "aarch64": "GB10"}
    rows, best = [], {}
    for (mach, eng, gpu, thr), v in sorted(groups.items(), key=lambda kv: (kv[0][0], not kv[0][2], kv[0][3])):
        # runs of one process: the first Monte Carlo includes the refinement's compilation
        firsts = [r for r in v if r["repeat"] == 0]
        steady = [r["mc"] for r in v if r["repeat"] > 0]
        if not steady:
            continue
        f0 = st.median([r["build"] + r["first_call"] for r in firsts]) if firsts else None
        m0 = st.median([r["mc"] for r in firsts]) if firsts else None
        m = st.median(steady)
        where = f"{gpu_name[mach]} GPU" if gpu else f"CPU, {thr} threads"
        name = "SAX jitted, Filipsson-Gunnar" if eng.startswith("sax") else "lightsim, reuse + stack + `trace_models`"
        rows.append(f"| {MACHINES[mach].split(' (')[0]} | {name} | {where} | {fmt(f0)} | {fmt(m0)} | {fmt(m)} ({fmt(min(steady))}–{fmt(max(steady))}, {len(steady)}) |")
        k = (mach, "gpu" if gpu else "cpu")
        if k not in best or m < best[k][0]:
            best[k] = (m, f0, thr)
    out = [
        "The resonator chip's Monte Carlo at 2,500 trials, 250 trials per call: SAX with JAX's CUDA build "
        "(the device each run used is in its record) against lightsim's CPU path on the same host. "
        "Set-up and first call, the first Monte Carlo (which includes the compilations for the refinement's "
        "shapes: SAX's jit and lightsim's traced models) and the median of the later ones:",
        "",
        "| host | engine | runs on | set-up and first call | first Monte Carlo | Monte Carlo, steady (range, runs) |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    parts = []
    for mach in sorted({k[0] for k in best}):
        if (mach, "gpu") not in best or (mach, "cpu") not in best:
            continue
        g, c = best[(mach, "gpu")], best[(mach, "cpu")]
        host = MACHINES[mach].split(" (")[0]
        first = f"set-up and first call take {fmt(c[1])} against {fmt(g[1])}" if c[1] is not None and g[1] is not None else ""
        if c[0] <= g[0]:
            parts.append(f"on the {host}, lightsim's CPU path ({c[2]} threads) is {g[0] / c[0]:.1f}x faster than SAX on the {gpu_name[mach]} GPU in the steady Monte Carlo ({fmt(c[0])} against {fmt(g[0])}) and its {first}")
        else:
            parts.append(f"on the {host}, SAX on the {gpu_name[mach]} GPU is {c[0] / g[0]:.2f}x faster than lightsim's CPU path at its best ({c[2]} threads) in the steady Monte Carlo ({fmt(g[0])} against {fmt(c[0])}), while lightsim's {first}")
    if parts:
        s_ = "; ".join(parts)
        out.append(s_[0].upper() + s_[1:] + ".")
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
        out.append(f"\n**{nice(lab)}**: {len(rs)} cases; speed-up of `auto` over the dense solve: median {st.median(sp):.1f}x, worst {sp[0]:.2f}x ({worst['case']} {worst['size']}, {worst['nw']} points), best {sp[-1]:.0f}x; largest relative difference {max(errs):.1e}.\n")
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
            out.append(f"\n**{nice(lab)}, {kind} scaling** ({'fixed batch' if kind == 'strong' else '2000 points per thread'}; parallel efficiency, time in ms)\n")
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
folders write. A timing taken while other jobs loaded the host (more
than {max_load:g} runnable processes besides the benchmark's own when it
started or ended; for older records without that count, a one-minute load
above {max_load:g}) is kept in the files but left out of the tables. Thread counts are stated per table; processes are pinned with
`taskset` to physical cores (on the GB10, to its Cortex-X925 cores) and
BLAS, OpenMP and rayon are given the same thread count; XLA sizes its pool
from the same affinity mask. GB10 timings, CPU ones included, run inside a
reservation of the node, which pauses its idle GPU workload. Every run is
sampled about once a second by `scripts/monitor.py` (cores used,
threads, memory, context switches, and per card utilisation, memory and
power); one summary line per run is in `results/monitor.jsonl`, and the
tables show the mean number of busy cores it measured.

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
numpy, jaxellip for jax); the resonance frequencies found are identical
(0 Hz difference) for every engine, call size and thread count, SAX with
the PDK's own models included (`results/dips.npz`). Versions: SAX 0.18.2, JAX and jaxlib 0.9.2,
klujax 0.5.3, qpdk 0.4.3, gdsfactory 9.51.0, numpy 2.5.5, scipy 1.18.1,
Python 3.12, double precision throughout.

{rf}

"Set-up and first call" is building the circuit from the netlist plus one
call of the first size (for jitted SAX, the compilation; SAX compiles again
for each new array shape, which the refinement needs). "Whole job" is set-up
plus the first 25-trial Monte Carlo (with the refinement's compilation for
SAX), timed inside a process in which Python, the packages and the netlist
are already loaded: starting the interpreter and the imports are not
included (on the GB10, one core: `import lightsim.circuit` 0.40 s, of which
0.24 s is scipy.signal through lightsim's inverse-design package; `import
jax` 0.26 s, `import sax` 1.46 s; the benchmark's own `import qpdk` 2.05 s
and `import gdsfactory` 0.94 s, which both engines pay). The trials per call
are those that suit each engine best: SAX's jitted Filipsson-Gunnar back end
is fastest with one trial per call (0.30 s on one core, against 0.57 s with
25 trials per call; its compiled code keeps a single trial's arrays in
cache), lightsim with 5 or 25.

**Scaling over threads, the whole Monte Carlo** (models, solves and the
search for the notches; pinned cores, the fast cores first on the GB10):

{rf_scaling}

**On a GPU.**

{gpu}

A GPU path for lightsim's own circuit solve was prototyped and not adopted:
compiling its generated CUDA kernels takes 11 s on the GB10 and 30 s on the
RTX PRO 6000, ten to forty times lightsim's whole first call on the CPU, for
a gain in the steady Monte Carlo of under a second.

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
