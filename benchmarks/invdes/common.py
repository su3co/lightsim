"""Helpers shared by the benchmark drivers (standard library and NumPy only)."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def timed(fn, reps):
    """Wall times of ``reps`` calls and the last result."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return ts, out


def cpu_name():
    model, cores, threads = platform.processor() or "CPU", os.cpu_count(), os.cpu_count()
    try:
        txt = Path("/proc/cpuinfo").read_text()
        names = [line.split(":", 1)[1].strip() for line in txt.splitlines() if line.startswith("model name")]
        if names:
            model, threads = names[0], len(names)
        ids = {(b.split("physical id")[1].split("\n")[0], b.split("core id")[1].split("\n")[0]) for b in txt.split("\n\n") if "core id" in b}
        cores = len(ids) or threads
    except Exception:
        pass
    return {"model": model, "cores": cores, "threads": threads}


def gpu_names():
    """Model names of the visible NVIDIA GPUs (honours CUDA_VISIBLE_DEVICES)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    rows = [line.split(",", 1) for line in out.strip().splitlines() if line.strip()]
    names = {int(i): n.strip() for i, n in rows}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [names[int(v)] for v in vis.split(",") if v.strip().isdigit() and int(v) in names]
    return [names[i] for i in sorted(names)]


def gpu_state():
    """Utilisation of the visible GPUs and the other processes computing on
    them. A timing is trusted only if, before it started, the GPUs it uses
    were idle (see ``contended``)."""
    try:
        u = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = {int(v) for v in vis.split(",") if v.strip().isdigit()} if vis else None
    gpus, by_uuid = {}, {}
    for r in u.strip().splitlines():
        i, uuid, util, mem = [v.strip() for v in r.split(",")]
        by_uuid[uuid] = int(i)
        if visible is None or int(i) in visible:
            gpus[int(i)] = {"utilization": int(util), "memory_used_mib": int(mem)}
    me, others = os.getpid(), []
    for r in q.strip().splitlines():
        if not r.strip():
            continue
        uuid, pid, mem = [v.strip() for v in r.split(",")]
        gi = by_uuid.get(uuid, -1)
        if int(pid) != me and gi in gpus:
            others.append({"gpu": gi, "pid": int(pid), "memory_mib": int(mem)})
    return {"gpus": gpus, "other_processes": others, "contended": any(g["utilization"] > 15 for g in gpus.values())}


class GpuMemory:
    """Peak device memory of this process, polled from nvidia-smi."""

    def __init__(self, period=0.25):
        self.period, self.peak, self._stop = period, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        me = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                tot = sum(int(r.split(",")[1]) for r in out.strip().splitlines() if r.split(",")[0].strip() == me)
                self.peak = max(self.peak, tot * 1024 * 1024)
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)


def host_peak_bytes():
    """Peak resident memory of this process (Linux: ru_maxrss is in kB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) * (1 if platform.system() == "Darwin" else 1024)


def rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))


def fd_summary(rows):
    """rows: [(i, j, adjoint, finite_difference)] -> dict for the report."""
    rows = [(int(i), int(j), float(a), float(f)) for i, j, a, f in rows]
    scale = max(abs(f) for *_, f in rows) or 1.0
    return {
        "pixels": [{"ij": [i, j], "gradient": a, "finite_difference": f, "rel": rel(a, f)} for i, j, a, f in rows],
        # worst pixel, relative to the largest finite difference of the set (a
        # near-zero entry does not dominate)
        "max_rel_to_largest": float(max(abs(a - f) for *_, a, f in rows) / scale),
        "max_rel": float(max(rel(a, f) for *_, a, f in rows)),
    }


MAX_SAVED_SAMPLES = 100_000  # gradient maps above this size are not stored


def slim(arrays):
    """What is kept next to a result: the gradient map (single precision) and
    S, for designs of up to MAX_SAVED_SAMPLES samples. The design itself is
    regenerated from problems.py."""
    out = {}
    for k, v in (arrays or {}).items():
        v = np.asarray(v)
        if k == "eps" or v.size > MAX_SAVED_SAMPLES:
            continue
        out[k] = v.astype(np.complex64 if np.iscomplexobj(v) else np.float32)
    return out


def save(rep, name, arrays=None, out=None):
    out = Path(out) if out else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"invdes_{name}.json"
    p.write_text(json.dumps(rep, indent=2, default=_default) + "\n")
    arrays = slim(arrays)
    if arrays:
        np.savez_compressed(out / f"invdes_{name}.npz", **arrays)
    print("wrote", p)
    return p


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))
