"""The two S-parameter normalisations on the 2-D inverse-design scene, with the port profiles of the centre frequency
and with profiles that follow the band (port_profiles="band"), at 45 nm / r cells.
usage: PYTHONPATH=<lightsim>/python python3 two_normalisations_2d.py [engine] [r,r,..]

From the reproduction of the defect in the monitor decomposition:

Scene: lightsim's 2-D inverse-design benchmark scene (E out of plane, two
offset silicon guides in oxide, a 2.43 um design square between them, 45 nm
cells, port planes 0.945 um from the design square).

For a handful of fixed (seeded) designs the script solves the same problem
with ``SMatrixProblem(normalize="monitor")`` (the default: S per unit
incident mode amplitude, the amplitude being measured by two mode monitors
2 and 4 cells downstream of the source) and with ``normalize="source"`` (S
per unit source spectrum). The source is a fixed current sheet in a linear
problem, so the amplitude it launches cannot depend on what is downstream:
the ratio of the two S must be one constant per frequency. It is not.

The control is the same scene with a straight guide through the design
square (no reflection). Its ratio is the calibration constant, and the
reflection lightsim reports for it should be zero.

Only lightsim's public API and NumPy. About 10 s on one CPU core.

    PYTHONPATH=<lightsim>/python python3 monitor_and_source_normalisation_disagree.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

import lightsim as ls
from lightsim import invdes

C0 = 299_792_458.0
F0 = C0 / 1.55e-6
H = 45e-9  # all lengths of the scene are multiples of this
EPS_SI, EPS_OX = 12.25, 1.444**2
WAVELENGTHS = (1.50e-6, 1.55e-6, 1.60e-6)
SEED = 20261127


class Scene:
    """The benchmark scene. ``straight`` puts the output guide in line with
    the input guide (the control). ``r`` divides the cell (45 nm / r).
    ``extra`` moves the input port plane (and the left edge of the domain)
    ``extra`` x 45 nm further from the design square."""

    def __init__(self, straight=False, r=1, extra=0, normalize="monitor", engine="python", device="cpu", port_profiles="center"):
        self.r, self.extra = r, extra
        dx = self.dx = H / r
        self.w2 = 2.5 * H  # half width of the guides
        self.yi = -9 * H
        self.yo = self.yi if straight else 9 * H
        self.d = 27 * H  # half size of the design square
        self.xp_in, self.xp_out = -(48 + extra) * H, 48 * H
        lo = [-(72 + extra) * H, -54 * H]
        hi = [72 * H, 54 * H]
        self.freqs = C0 / np.asarray(WAVELENGTHS)
        pml = ls.PML(layers=12 * r, inside=True)
        sim = ls.Simulation(
            min=[lo[0], lo[1], 0.0],
            max=[hi[0], hi[1], 1e-6],
            mesh={"type": "uniform", "dx": [dx, dx, 1e-6]},
            run_time=3e-12,
            boundaries={"x": [pml, pml], "y": [pml, pml], "z": [ls.PERIODIC, ls.PERIODIC]},
            background="clad",
            shutoff=1e-8,
        )
        sim.add_medium(ls.Medium("clad", eps=EPS_OX))
        sim.add_medium(ls.Medium("core", eps=EPS_SI))
        sim.add_structure(ls.Box([-1.0, self.yi - self.w2, -1.0], [-self.d + 0.5 * dx, self.yi + self.w2, 1.0]), "core")
        sim.add_structure(ls.Box([self.d - 0.5 * dx, self.yo - self.w2, -1.0], [1.0, self.yo + self.w2, 1.0]), "core")
        ph = 22 * H
        self.p_in = invdes.port_mode(sim, "in", 0, self.xp_in, [self.xp_in, self.yi - ph, 0.0], [self.xp_in, self.yi + ph, 1e-6], F0, "+", "tm")
        self.p_out = invdes.port_mode(sim, "out", 0, self.xp_out, [self.xp_out, self.yo - ph, 0.0], [self.xp_out, self.yo + ph, 1e-6], F0, "-", "tm")
        self.n = 55  # design samples per side, at the 45 nm nodes
        grid = invdes.DesignGrid([-self.d, -self.d, -1.0], [self.d, self.d, 1.0], (self.n, self.n), ls.Medium("design", eps=EPS_SI))
        self.sim = sim
        self.problem = invdes.SMatrixProblem(sim, [grid], [self.p_in], [self.p_out, self.p_in], self.freqs, devices=device, fwidth=0.1 * F0, normalize=normalize, engine=engine, port_profiles=port_profiles)

    def with_perimeter(self, design):
        """The design with its outermost samples set to the fixed scene:
        silicon where a guide meets the square, oxide elsewhere."""
        e = np.array(design, dtype=float)
        y = np.linspace(-self.d, self.d, self.n)
        e[0, :] = np.where(np.abs(y - self.yi) <= self.w2 + 1e-12, EPS_SI, EPS_OX)
        e[-1, :] = np.where(np.abs(y - self.yo) <= self.w2 + 1e-12, EPS_SI, EPS_OX)
        e[:, 0] = e[:, -1] = EPS_OX
        return e

    def straight_guide(self):
        y = np.linspace(-self.d, self.d, self.n)
        return np.tile(np.where(np.abs(y - self.yi) <= self.w2 + 1e-12, EPS_SI, EPS_OX), (self.n, 1))

    def solve(self, design):
        """(S21, S11) at the three wavelengths."""
        S = np.asarray(self.problem.solve([np.asarray(design, dtype=float)], adjoint=False).S)
        return S[0, 0], S[1, 0]


def smooth_noise(rng, n, corr):
    """Gaussian random field on n x n samples, correlation length ``corr`` samples."""
    k = np.fft.fftfreq(n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * np.exp(-2.0 * (np.pi * corr) ** 2 * k2)).real
    return (f - f.mean()) / f.std()


def fixed_designs(n=55, count=6, seed=SEED):
    """Named permittivity maps (before the perimeter rule): two uniform
    fills and ``count`` seeded random patterns, binary and grey in turn."""
    rng = np.random.default_rng(seed)
    out = [("all oxide (guide ends)", np.full((n, n), EPS_OX)), ("all silicon", np.full((n, n), EPS_SI))]
    for i in range(count):
        corr = (2.0, 3.0, 5.0)[i % 3]
        f = smooth_noise(rng, n, corr)
        if i % 2 == 0:
            out.append((f"binary, features ~{corr * 45:.0f} nm, #{i}", np.where(f > 0.0, EPS_SI, EPS_OX)))
        else:
            out.append((f"grey, features ~{corr * 45:.0f} nm, #{i}", EPS_OX + (EPS_SI - EPS_OX) / (1.0 + np.exp(-2.0 * f))))
    return out


def run(engine="python", count=6, out=sys.stdout, r=1, port_profiles="center"):
    t0 = time.time()
    rows = []
    cal = None
    for straight in (True, False):
        mon = Scene(straight=straight, normalize="monitor", engine=engine, r=r, port_profiles=port_profiles)
        src = Scene(straight=straight, normalize="source", engine=engine, r=r, port_profiles=port_profiles)
        designs = [("CONTROL: straight guide", mon.straight_guide())] if straight else [(nm, mon.with_perimeter(e)) for nm, e in fixed_designs(count=count)]
        for name, eps in designs:
            s21m, s11m = mon.solve(eps)
            s21s, s11s = src.solve(eps)
            ratio = s21m / s21s  # = source spectrum / measured launch amplitude
            if cal is None:
                cal = ratio  # the control: one constant per wavelength
            rows.append({"name": name, "ratio": ratio / cal, "s21_monitor": s21m, "s11_monitor": s11m, "s21_source": s21s * cal, "s11_source": s11s * cal})
    print(f"lightsim {getattr(ls, '__version__', '?')}, engine={engine}, cpu, {45 / r:g} nm cells, port_profiles={port_profiles}; wavelengths 1.50 / 1.55 / 1.60 um", file=out)
    print("ratio = S21(normalize='monitor') / S21(normalize='source'), divided by the same ratio of the control.", file=out)
    print("A fixed source in a linear problem launches the same amplitude whatever the design: every ratio should be 1.\n", file=out)
    print(f"{'design':34s} {'|ratio|-1 in % at 3 wavelengths':>34s} {'arg(ratio) in mrad':>24s} {'|S11|':>6s} {'|S21|':>6s} {'T+R monitor':>12s} {'T+R source':>11s}", file=out)
    for r in rows:
        dm = 100.0 * (np.abs(r["ratio"]) - 1.0)
        ph = 1e3 * np.angle(r["ratio"])
        tr_m = np.abs(r["s21_monitor"]) ** 2 + np.abs(r["s11_monitor"]) ** 2
        tr_s = np.abs(r["s21_source"]) ** 2 + np.abs(r["s11_source"]) ** 2
        print(
            f"{r['name']:34s} {dm[0]:+10.3f} {dm[1]:+10.3f} {dm[2]:+10.3f}   {ph[0]:+7.2f} {ph[1]:+7.2f} {ph[2]:+7.2f} "
            f"{np.abs(r['s11_monitor'][1]):6.3f} {np.abs(r['s21_monitor'][1]):6.3f} {tr_m.max():12.4f} {tr_s.max():11.4f}",
            file=out,
        )
    dev = np.array([r["ratio"] for r in rows[1:]])
    c = rows[0]
    print(file=out)
    print(f"designs (control excluded): |ratio| spread, std {100 * np.std(np.abs(dev)):.2f} %, range {100 * (np.abs(dev).min() - 1):+.2f} % .. {100 * (np.abs(dev).max() - 1):+.2f} %;", file=out)
    print(f"  |ratio|^2 (power) range {100 * (np.abs(dev).min() ** 2 - 1):+.2f} % .. {100 * (np.abs(dev).max() ** 2 - 1):+.2f} %; phase std {1e3 * np.std(np.angle(dev)):.1f} mrad", file=out)
    print(f"control: |S21| = {np.abs(c['s21_monitor']).round(5).tolist()}, reported |S11| = {np.abs(c['s11_monitor']).round(5).tolist()} (a straight guide: should be 0)", file=out)
    print(f"(T+R columns: largest of the three wavelengths; 'source' = normalize='source' times the control's constant)", file=out)
    print(f"wall time {time.time() - t0:.1f} s", file=out)
    return rows


if __name__ == "__main__":
    for r in [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1").split(",")]:
        for pp in ("center", "band"):
            run(engine=sys.argv[1] if len(sys.argv) > 1 else "python", r=r, port_profiles=pp)
            print()
