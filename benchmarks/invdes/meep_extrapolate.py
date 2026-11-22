"""Meep gradients too slow to run in full: extrapolation by step count and
measured stepping rate.

    for s in wg3d_6M wg3d_32M wg3d_86M mm3d_44nm mm3d_22nm; do for n in 100 200; do
        mpirun -np 64 python meep_rate.py --scene $s --steps $n | grep '^{' >> ../results/invdes_meep_rate_scaling.jsonl
    done; done
    python meep_extrapolate.py

A full Meep gradient of ``wg3d_32M``, ``wg3d_86M`` and ``mm3d_22nm`` takes
much longer than the 15 minutes a benchmark job may run. Shortening the runs
does not help much: ``meep.adjoint`` runs every forward and adjoint
simulation until its sources have ended (2158 steps for the 82 fs pulse on a
20 nm grid; for the multi-wavelength adjoint source, 805 fs), whatever the
requested run time.

So the time of a full gradient of scene S is scaled from a gradient of the
same family that was measured in full (C: ``wg3d_6M`` or ``mm3d_44nm``, same
process count):

    T_S = T_C * (W_S / W_C) * (R_C / R_S)

W = cells x (time steps of all forward and adjoint runs of one gradient), R =
cells per second of Meep's time stepping in that scene, measured with
``meep_rate.py`` at 100 and 200 steps (the two must agree: constant cost per
step). The overheads that do not scale with steps (mode solves, gradient
assembly) are thereby assumed to scale with W too; they are a few per cent of
T_C.

Check: one gradient of ``wg3d_32M`` with every run stopped at the end of the
source (``meep_bench.py --timing-steps``, 2158 + 2158 steps) was measured
(``invdes_meeptiming_*``); the same formula predicts it, and the ratio is
printed and stored as ``check_ratio``.

Output: ``invdes_meep_adjoint_<scene>_<np>p_extrapolated.json``
(``extrapolated: true``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import problems

RES = Path(__file__).resolve().parent.parent / "results"
CAL = {"wg3d": "wg3d_6M", "mm3d": "mm3d_44nm"}
TARGETS = ["wg3d_32M", "wg3d_86M", "mm3d_22nm"]
NPROC = 64


def full_steps(sc):
    """Steps of one forward and one adjoint run of meep_bench.py on scene sc."""
    dt = sc.dt()
    fwd = int(np.ceil(sc.run_time / dt))
    if len(sc.wavelengths) == 1:
        return fwd, fwd
    f = np.sort(problems.C0 / np.asarray(sc.wavelengths))
    pulse_end = 2 * problems.PULSE_OFFSET / (2 * np.pi * problems.PULSE_FWIDTH * problems.F0)
    adj = float(np.max(1 / np.diff(f))) + sc.run_time - pulse_end
    return fwd, int(np.ceil(adj / dt))


def work(sc, steps):
    return sc.n_cells * len(sc.inputs) * (steps[0] + steps[1])


def rates():
    out = {}
    p = RES / "invdes_meep_rate_scaling.jsonl"
    for line in p.read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r["mpi_processes"] == NPROC:
                out.setdefault(r["scene"], {})[r["steps"]] = r["cells_per_second"]
    return out


def main():
    R = rates()

    def rate(s):
        v = R[s]
        spread = (max(v.values()) - min(v.values())) / np.mean(list(v.values()))
        return float(np.mean(list(v.values()))), {str(k): float(x) for k, x in sorted(v.items())}, float(spread)

    for s in TARGETS:
        sc = problems.make(s)
        cs = CAL[s[:4]]
        cal = json.loads((RES / f"invdes_meep_adjoint_{cs}_{NPROC}p.json").read_text())
        scc = problems.make(cs)
        assert [cal["steps_forward"], cal["steps_adjoint"]] == list(full_steps(scc)), (cs, cal["steps"], full_steps(scc))
        r_s, r_s_all, spread_s = rate(s)
        r_c, r_c_all, spread_c = rate(cs)
        fs = full_steps(sc)
        scale = work(sc, fs) / work(scc, full_steps(scc)) * r_c / r_s
        t = cal["seconds_per_gradient"] * scale
        rep = {
            "tool": "meep",
            "version": cal["version"],
            **sc.describe(),
            "variant": "adjoint",
            "method": cal["method"],
            "hardware": cal["hardware"],
            "cpu": cal["cpu"],
            "precision": "double",
            "mpi_processes": NPROC,
            "simulations": {"forward": len(sc.inputs), "adjoint": len(sc.inputs)},
            "steps": list(fs),
            "steps_forward": fs[0],
            "steps_adjoint": fs[1],
            "extrapolated": True,
            "calibration_scene": cs,
            "calibration_seconds": cal["seconds_per_gradient"],
            "cells_per_second": r_s,
            "cells_per_second_by_steps": r_s_all,
            "rate_spread": spread_s,
            "calibration_cells_per_second": r_c,
            "calibration_cells_per_second_by_steps": r_c_all,
            "seconds_per_gradient": float(t),
            "seconds_all": [float(t)],
            "note": f"extrapolated, not measured: the full gradient of {cs} ({cal['seconds_per_gradient']:.0f} s) scaled by cells x steps and by the stepping rate measured on both scenes ({r_c / 1e6:.0f} and {r_s / 1e6:.0f} M cells/s at 100 and 200 steps); no finite-difference check",
        }
        msg = ""
        chk = RES / f"invdes_meeptiming_adjoint_{s}_{NPROC}p_clipped.json"
        if chk.exists():
            m = json.loads(chk.read_text())
            pred = cal["seconds_per_gradient"] * work(sc, m["steps"]) / work(scc, full_steps(scc)) * r_c / r_s
            rep["check_measured_seconds"] = m["seconds_per_gradient"]
            rep["check_steps"] = m["steps"]
            rep["check_predicted_seconds"] = float(pred)
            rep["check_ratio"] = float(m["seconds_per_gradient"] / pred)
            msg = f"; check: runs stopped at {m['steps']} steps measured {m['seconds_per_gradient']:.0f} s, predicted {pred:.0f} s (measured / predicted {m['seconds_per_gradient'] / pred:.2f})"
        name = f"invdes_meep_adjoint_{s}_{NPROC}p_extrapolated.json"
        (RES / name).write_text(json.dumps(rep, indent=2) + "\n")
        print(f"{s}: {fs[0]} + {fs[1]} steps x {len(sc.inputs)} inputs, rate {r_s / 1e6:.0f} M cells/s (spread {100 * spread_s:.1f} %), calibration {cs} {cal['seconds_per_gradient']:.0f} s at {r_c / 1e6:.0f} M cells/s -> {t:.0f} s{msg}")


if __name__ == "__main__":
    main()
