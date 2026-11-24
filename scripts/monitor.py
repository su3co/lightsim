"""Run a command and sample the resources it uses, about once a second.

    python monitor.py --jsonl results/monitor.jsonl [--gpu] -- python bench_rf.py ...

Linux only (reads /proc). Appends one summary line to ``--jsonl`` (with the
command's ``--engine``, ``--trials``, ``--threads``, ``--chunk`` and
``--label`` arguments, so that the line can be matched with the timing
record of the same run); ``--out PREFIX`` also writes every sample to
``PREFIX.csv``. The command's exit code is passed through.

Per sample: wall time, the cores of the affinity mask that are busy (from
/proc/stat), the command's process tree CPU use in cores (user and system
separately), its thread count, resident memory, context switches, the number
of its threads busy above half a core, and with ``--gpu`` every card's
utilisation, memory-controller utilisation, memory used, power, temperature,
SM clock and throttle reasons (nvidia-smi, sampled in the same loop), for
the cards the job holds (``GET_COMPUTE_GPUS`` or ``CUDA_VISIBLE_DEVICES``).

The summary gives the mean and peak cores used against the cores given (the
parallel efficiency of the run as the hardware saw it), the share of the
wall time in which one thread did the work while the rest idled (a serial
section: Python, set-up), the kernel-time share, the peak memory and, per
card, the busy fraction and mean power.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")


def cpu_times():
    """{cpu index: (busy, total)} jiffies from /proc/stat."""
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                v = list(map(int, f[1:]))
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out[int(f[0][3:])] = (sum(v) - idle, sum(v))
    return out


def tree(pid):
    """The pid and all its descendants."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as fh:
                    s = fh.read()
                pp = int(s[s.rindex(")") + 2 :].split()[1])
                kids.setdefault(pp, []).append(int(d))
            except (OSError, ValueError):
                pass
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def proc_sample(pids):
    """CPU (user, system jiffies per thread), threads, RSS, context switches."""
    ut = st = rss = vcs = ncs = 0
    threads = {}
    for p in pids:
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/stat") as fh:
                    s = fh.read()
                f = s[s.rindex(")") + 2 :].split()
                u, k = int(f[11]), int(f[12])
                threads[(p, int(t))] = u + k
                ut += u
                st += k
            with open(f"/proc/{p}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
                    elif line.startswith("voluntary_ctxt_switches:"):
                        vcs += int(line.split()[1])
                    elif line.startswith("nonvoluntary_ctxt_switches:"):
                        ncs += int(line.split()[1])
        except OSError:
            pass
    return ut, st, rss, vcs, ncs, threads


GPU_Q = "index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks_throttle_reasons.active"


def _own_gpus():
    """Physical indices of the cards this job holds (the queue helper's
    GET_COMPUTE_GPUS, else CUDA_VISIBLE_DEVICES when it lists indices), or
    None for all cards."""
    for var in ("GET_COMPUTE_GPUS", "CUDA_VISIBLE_DEVICES"):
        v = os.environ.get(var)
        if v:
            try:
                return {int(x) for x in v.split(",") if x.strip()}
            except ValueError:
                continue
    return None


def gpu_sample():
    own = _own_gpus()
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            if own is not None and int(f[0]) not in own:
                continue
            rows.append({"gpu": int(f[0]), "util": float(f[1]), "mem_util": float(f[2]), "mem_mib": float(f[3]), "power_w": float(f[4]) if f[4] not in ("[N/A]", "N/A") else None, "temp_c": float(f[5]), "sm_mhz": float(f[6]) if f[6] not in ("[N/A]", "N/A") else None, "throttle": f[7]})
        except (ValueError, IndexError):
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="also write the samples to OUT.csv")
    ap.add_argument("--jsonl", default=None, help="append the summary as one JSON line")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    for f in (a.out, a.jsonl):
        if f:
            os.makedirs(os.path.dirname(os.path.abspath(f)), exist_ok=True)
    mask = sorted(os.sched_getaffinity(0))
    c0 = cpu_times()
    t0 = time.time()
    p = subprocess.Popen(cmd)
    rows = []
    last_cpu, last_t, last_thr = c0, t0, {}
    last_proc = (0, 0)
    while True:
        try:
            p.wait(timeout=a.interval)
            done = True
        except subprocess.TimeoutExpired:
            done = False
        now = time.time()
        c1 = cpu_times()
        ut, st, rss, vcs, ncs, thr = proc_sample(tree(p.pid)) if not done else (last_proc[0], last_proc[1], 0, 0, 0, {})
        dt = max(now - last_t, 1e-9)
        busy_mask = sum((c1[c][0] - last_cpu[c][0]) / max(c1[c][1] - last_cpu[c][1], 1) for c in mask if c in c1 and c in last_cpu)
        busy_all = sum((c1[c][0] - last_cpu[c][0]) / max(c1[c][1] - last_cpu[c][1], 1) for c in c1 if c in last_cpu)
        active = sum(1 for k, v in thr.items() if (v - last_thr.get(k, v)) / TICK / dt > 0.5)
        row = {"t": now - t0, "mask_busy_cores": busy_mask, "host_busy_cores": busy_all, "proc_user_cores": (ut - last_proc[0]) / TICK / dt, "proc_sys_cores": (st - last_proc[1]) / TICK / dt, "threads": len(thr), "busy_threads": active, "rss_mb": rss / 2**20, "vol_cs": vcs, "invol_cs": ncs}
        if a.gpu:
            for g in gpu_sample():
                for k, v in g.items():
                    if k != "gpu":
                        row[f"gpu{g['gpu']}_{k}"] = v
        if not done:
            rows.append(row)
            last_proc, last_thr = (ut, st), thr
        last_cpu, last_t = c1, now
        if done:
            break
    wall = time.time() - t0
    if a.out:
        keys = sorted({k for r in rows for k in r})
        with open(a.out + ".csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    used = [r["proc_user_cores"] + r["proc_sys_cores"] for r in rows]
    summ = {
        "cmd": [os.path.basename(c) if os.sep in c else c for c in cmd],  # no local paths in the record
        "wall_s": wall,
        "samples": len(rows),
        "cores_given": len(mask),
        "mean_cores_used": sum(used) / len(used) if used else None,
        "peak_cores_used": max(used) if used else None,
        "efficiency_vs_given": (sum(used) / len(used) / len(mask)) if used else None,
        "sys_share": (sum(r["proc_sys_cores"] for r in rows) / max(sum(used), 1e-9)) if used else None,
        "serial_share": (sum(1 for r in rows if r["busy_threads"] <= 1 and r["proc_user_cores"] + r["proc_sys_cores"] < 1.5) / len(rows)) if rows else None,
        "peak_rss_mb": max((r["rss_mb"] for r in rows), default=None),
        "peak_threads": max((r["threads"] for r in rows), default=None),
        "host_busy_cores_mean": (sum(r["host_busy_cores"] for r in rows) / len(rows)) if rows else None,
        "mask_busy_cores_mean": (sum(r["mask_busy_cores"] for r in rows) / len(rows)) if rows else None,
        "returncode": p.returncode,
    }
    if a.gpu and rows:
        for g in sorted({k.split("_")[0] for r in rows for k in r if k.startswith("gpu")}):
            u = [r[f"{g}_util"] for r in rows if f"{g}_util" in r]
            pw = [r[f"{g}_power_w"] for r in rows if r.get(f"{g}_power_w") is not None]
            summ[g] = {"busy_mean": sum(u) / len(u) if u else None, "busy_fraction_over_50": sum(1 for x in u if x > 50) / len(u) if u else None, "power_mean_w": sum(pw) / len(pw) if pw else None, "mem_peak_mib": max((r.get(f"{g}_mem_mib") or 0) for r in rows), "throttle_seen": sorted({r.get(f"{g}_throttle") for r in rows if r.get(f"{g}_throttle")})}
    args = {}
    for i, c in enumerate(cmd):
        if c in ("--engine", "--trials", "--threads", "--chunk", "--label", "--circuit", "--nw") and i + 1 < len(cmd):
            v = cmd[i + 1]
            args[c[2:]] = int(v) if v.isdigit() else v
    script = next((os.path.basename(c) for c in cmd if c.endswith(".py")), "")
    summ = {"script": script, **args, **{k: v for k, v in summ.items() if k != "cmd"}}
    if a.jsonl:
        with open(a.jsonl, "a") as fh:
            fh.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}) + "\n")
    print("## monitor", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items() if k != "cmd"}), file=sys.stderr, flush=True)
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
