"""How the simulated time of each scene was chosen (lightsim, native engine).

    python convergence.py --scene wg3d_6M --times 150e-15,220e-15,330e-15 --device cuda:0

For each simulated time: the objective, and the directional derivative from
the adjoint gradient against central finite differences. The time used by
the benchmark is the shortest one after which the objective changes by less
than 1 % and the two derivatives agree to about 1 %: a run that is cut while
the fields still ring gives a different derivative in every tool.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common
import problems
from lightsim_bench import build

from lightsim import invdes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--times", required=True, help="comma-separated simulated times (s)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for t in [float(v) for v in a.times.split(",")]:
        sc = problems.make(a.scene)
        sc.run_time = t
        sim, ins, outs, grid = build(sc)
        prob = invdes.SMatrixProblem(sim, [grid], ins, outs, sc.freqs, devices=a.device, fwidth=problems.PULSE_FWIDTH * problems.F0, engine="native")
        eps0, d = sc.design_eps(), sc.check_direction()
        sol = prob.solve([eps0])
        L, g = sc.objective(sol.S)
        ad = float(np.sum(sol.vjp(g)[0][:, :, 0] * d))
        val = lambda e: sc.objective(prob.solve([e], adjoint=False).S)[0]  # noqa: E731
        fd = (val(eps0 + a.step * d) - val(eps0 - a.step * d)) / (2 * a.step)
        rows.append({"run_time": t, "steps": int(sol.stats["steps"][0]), "objective": L, "adjoint": ad, "finite_difference": fd, "rel": common.rel(ad, fd)})
        print(json.dumps(rows[-1]), flush=True)
        del prob
    common.save({"tool": "lightsim", "scene": a.scene, "rows": rows}, f"convergence_{a.scene}", out=a.out)


if __name__ == "__main__":
    main()
