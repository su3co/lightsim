"""Resonance of a dielectric disc in a 2-D simulation (a curved interface drawn as a Cylinder), both sub-pixel
weightings, against the analytic whispering-gallery resonance (complex root of the Bessel / Hankel matching condition).
The disc centre is shifted by s cells along x and y. A point dipole inside the rim excites the mode; the spectrum at a
point on the far side, taken after the pulse and the low-Q resonances have gone, is fitted with one pole plus a
linear background.
usage: python benchmarks/subpixel/disc_resonance.py OUT.jsonl DEVICE DX_NM[,..] [POL,..] [SHIFT,..]
lightsim "tm" = E along the cylinder axis (parallel to the surface everywhere); "te" = E in the plane."""
import sys, json, time
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
import lightsim as ls

out, dev = sys.argv[1], sys.argv[2]
dxs = [float(v) for v in sys.argv[3].split(",")]
pols = sys.argv[4].split(",") if len(sys.argv) > 4 else ["tm", "te"]
shifts = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.0, 0.25, 0.5]
C0 = 299792458.0
ND, NB = 3.5, 1.444
RADIUS = {"tm": 0.6e-6, "te": 0.7e-6}   # one resonance with Q of 200 to 300 near 1.55 um for each
R = 0.6e-6

def char(k, m, pol):
    x1, x2 = ND * k * R, NB * k * R
    if pol == "tm":   # E_z: n_d J'/J = n_b H'/H
        return ND * jvp(m, x1) * hankel1(m, x2) - NB * jv(m, x1) * h1vp(m, x2)
    return NB * jvp(m, x1) * hankel1(m, x2) - ND * jv(m, x1) * h1vp(m, x2)

def roots(pol, lam_lo=1.35e-6, lam_hi=1.75e-6):
    found = []
    for m in range(2, 14):
        for kr in np.linspace(2 * np.pi / lam_hi, 2 * np.pi / lam_lo, 60):
            k = complex(kr, -1e3)
            for _ in range(60):
                f = char(k, m, pol); h = 1e-6 * abs(k)
                d = (char(k + h, m, pol) - char(k - h, m, pol)) / (2 * h)
                step = f / d
                k -= step
                if abs(step) < 1e-13 * abs(k): break
            else:
                continue
            if k.imag < 0 and 2 * np.pi / lam_hi < k.real < 2 * np.pi / lam_lo and abs(char(k, m, pol)) < 1e-6 * abs(char(k * 1.01, m, pol)):
                if not any(abs(k - r[1]) < 1e-6 * abs(k) and m == r[0] for r in found):
                    found.append((m, k, k.real / (-2 * k.imag)))
    return found

def pick(pol):
    # the resonance with the highest Q below 500 (it must ring down inside the run)
    c = [r for r in roots(pol) if 150 < r[2] < 500]
    return max(c, key=lambda r: r[2])

def run(dx, pol, weighting, s, m, kref, q):
    f0 = kref.real * C0 / (2 * np.pi)
    half = R + 1.0e-6
    t_run = 9.0 * 2 * q / (2 * np.pi * f0) + 60e-15
    sim = ls.Simulation(min=[-half, -half], max=[half, half], mesh=ls.UniformMesh(dx), run_time=t_run, polarization=pol, background="clad", subpixel_weighting=weighting)
    sim.add_medium(ls.Medium("clad", eps=NB**2)); sim.add_medium(ls.Medium("core", eps=ND**2))
    c = [s * dx, s * dx]
    sim.add_structure(ls.Cylinder(c, R), "core")
    comp = "Ez" if pol == "tm" else "Ey"
    sim.add_source(ls.PointDipole([c[0] + 0.8 * R, c[1]], comp, ls.GaussianPulse(f0, 0.06 * f0), name="d"))
    p = [c[0] - 0.8 * R, c[1]]
    sim.add_monitor(ls.ProbeMonitor("p", p, components=(comp,)))
    t0 = time.perf_counter(); res = sim.run(device=dev); el = time.perf_counter() - t0
    t, v = res.time_series("p", comp)
    t, v = np.asarray(t, float).ravel(), np.asarray(v, float).ravel()
    # late part of the ring-down only (the pulse and the low-Q resonances are gone after 0.4 ps): its spectrum is one pole
    sel = t > 0.4e-12
    t, v = t[sel] - t[sel][0], v[sel]   # (time from the start of the window: the spectrum is then the bare pole)
    x = np.linspace(-0.012, 0.012, 193)
    e = np.array([np.sum(v * np.exp(2j * np.pi * f0 * (1 + xx) * t)) for xx in x])
    best = None
    for w in (0.012, 0.006, 0.003):
        sl = np.abs(x - (best.real if best is not None else x[np.argmax(np.abs(e))])) <= w
        A = np.stack([e[sl], np.ones(sl.sum()), x[sl]], axis=1)
        best = np.linalg.lstsq(A, e[sl] * x[sl], rcond=None)[0][0]
    f_res = f0 * (1 + best.real); q_fit = (1 + best.real) / (-2 * best.imag) if best.imag != 0 else float("nan")
    return dict(dx_nm=dx * 1e9, pol=pol, weighting=weighting, shift=s, m=m, f_ref=f0, q_ref=q, f_res=f_res, rel_err=f_res / f0 - 1.0, q_fit=q_fit, seconds=el, steps=int(t_run / res.dt))

import warnings; warnings.simplefilter("ignore")
for pol in pols:
    R = RADIUS[pol]
    m, k, q = pick(pol)
    print(f"## {pol}: radius {R*1e9:.0f} nm, m = {m}, wavelength {2*np.pi/k.real*1e9:.3f} nm, Q {q:.1f}", flush=True)
    for dx_nm in dxs:
        for wt in ("cell", "tent"):
            for s in shifts:
                rec = run(dx_nm * 1e-9, pol, wt, s, m, k, q)
                open(out, "a").write(json.dumps(rec) + "\n")
                print(json.dumps({a: (round(b, 7) if isinstance(b, float) and abs(b) < 1e6 else b) for a, b in rec.items()}), flush=True)
print("## disc resonance done", flush=True)
