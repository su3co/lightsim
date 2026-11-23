"""Activity-driven ("moving window") 2-D FDTD: a prototype to measure what
skipping inactive tiles saves and what it costs in accuracy.

The domain is cut into square tiles. After every ``check`` steps the peak
energy density of every tile is compared with ``tau`` times the largest
energy density seen so far in the run; tiles above it, dilated by one tile
(so that a wave cannot cross an inactive tile between two checks), are the
active set. Inactive tiles are not updated and their fields are set to zero:
the energy they held is dropped and accounted for.

Error certificate. The Yee scheme in a lossless medium conserves a discrete
energy, and the absorbing layer used here (matched electric and magnetic
conductivity) only removes energy. The difference between the windowed and
the full run is the response of the full scheme to the dropped fields
(injected with the opposite sign) and to the fields that an inactive tile
would have received; by linearity and the energy bound, its energy norm at
any time is at most the sum of the energy norms of all dropped fields. The
run reports this sum relative to the injected energy (``bound``), next to
the observed error of the port amplitudes.

Pure NumPy; the windowed update is emulated by masking the full update, so
the reported speed-up is the ratio of updated cells, not wall time.

    python window_fdtd.py --case i1 --taus 1e-2,1e-4,1e-6,1e-8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DX_UM = 1.0 / 30.0
LAM_UM = 1.55
N_SI_SLAB = None


def slab_index():
    try:
        from mesh2d import N_SLAB

        return N_SLAB
    except Exception:  # pragma: no cover
        return 2.848


class Grid:
    def __init__(self, eps, pml=30, dt=0.5):
        self.eps = np.asarray(eps, float)
        nx, ny = self.eps.shape
        self.nx, self.ny = nx, ny
        self.dt = dt
        # conductivity profile (quadratic) on the cell-centre grid
        def prof(n):
            s = np.zeros(n)
            d = np.arange(pml)[::-1] + 0.5
            s[:pml] = (d / pml) ** 2
            s[n - pml :] = (d[::-1] / pml) ** 2
            return s

        # quadratic profile: one-way field attenuation int(sigma / sqrt(eps)) dx = smax * pml / (3 sqrt(eps)) = 5
        # in the densest medium (round trip e^-10)
        smax = 15.0 * np.sqrt(self.eps.max()) / pml
        sig = smax * np.maximum(prof(nx)[:, None], prof(ny)[None, :])
        self.sig = sig
        e = self.eps
        # Ex on (nx, ny+1): eps from the cell centres on either side in y
        epx = np.pad(e, ((0, 0), (1, 1)), mode="edge")
        epx = 0.5 * (epx[:, 1:] + epx[:, :-1])
        sgx = np.pad(sig, ((0, 0), (1, 1)), mode="edge")
        sgx = 0.5 * (sgx[:, 1:] + sgx[:, :-1])
        epy = np.pad(e, ((1, 1), (0, 0)), mode="edge")
        epy = 0.5 * (epy[1:] + epy[:-1])
        sgy = np.pad(sig, ((1, 1), (0, 0)), mode="edge")
        sgy = 0.5 * (sgy[1:] + sgy[:-1])
        self.cax = (1 - sgx * dt / (2 * epx)) / (1 + sgx * dt / (2 * epx))
        self.cbx = (dt / epx) / (1 + sgx * dt / (2 * epx))
        self.cay = (1 - sgy * dt / (2 * epy)) / (1 + sgy * dt / (2 * epy))
        self.cby = (dt / epy) / (1 + sgy * dt / (2 * epy))
        sm = sig / e  # matched magnetic conductivity (mu = 1)
        self.da = (1 - sm * dt / 2) / (1 + sm * dt / 2)
        self.db = dt / (1 + sm * dt / 2)
        self.epx, self.epy = epx, epy


def slab_mode(eps_line, k0):
    """Fundamental TE (Hz) mode of a 1-D cross-section, unit cell spacing."""
    e = np.asarray(eps_line, float)
    n = len(e)
    ie = 1.0 / (0.5 * (e[1:] + e[:-1]))
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = k0**2 * e[i]
        if i > 0:
            A[i, i] -= e[i] * ie[i - 1]
            A[i, i - 1] += e[i] * ie[i - 1]
        if i < n - 1:
            A[i, i] -= e[i] * ie[i]
            A[i, i + 1] += e[i] * ie[i]
    w, v = np.linalg.eig(A)
    k = int(np.argmax(w.real))
    h = np.real(v[:, k])
    h = h / np.sqrt(np.sum(h**2 / e))
    if h[np.argmax(np.abs(h))] < 0:
        h = -h
    return h, float(np.sqrt(w[k].real))


def run(g: Grid, src, outs, freqs, steps, tau=0.0, tile=32, check=8, record_every=0, shutoff=1e-5):
    """One run. ``src`` = (i, j0, j1, profile): a soft Ey current line at x
    index i over y cells j0..j1. ``outs`` = list of (i, j0, j1, profile)
    where the Hz DFT is projected on the profile. Returns dict."""
    nx, ny, dt = g.nx, g.ny, g.dt
    Hz = np.zeros((nx, ny))
    Ex = np.zeros((nx, ny + 1))
    Ey = np.zeros((nx + 1, ny))
    tx, ty = -(-nx // tile), -(-ny // tile)
    active = np.ones((tx, ty), bool) if tau <= 0 else np.zeros((tx, ty), bool)
    f0 = float(np.mean(freqs))
    fw = max(0.1 * f0, float(np.ptp(freqs)))
    t0 = 4.0 / (2 * np.pi * fw) * 2
    si, sj0, sj1, sprof = src
    if tau > 0:
        active[si // tile, sj0 // tile : (sj1 - 1) // tile + 1] = True
    w = 2 * np.pi * np.asarray(freqs)
    acc = np.zeros((len(outs), len(freqs)), complex)
    src_spec = np.zeros(len(freqs), complex)
    updated = 0
    e_ref = 1e-300
    injected = 0.0
    dropped_norm = 0.0
    peak_norm = 1e-300
    frac_hist = []
    trace = []
    cellmask = None

    def expand(m):
        return np.kron(m, np.ones((tile, tile), bool))[:nx, :ny]

    for n in range(steps):
        t = n * dt
        if tau > 0 and (n % check == 0):
            # energy density per cell (cell-centred approximations)
            u = g.eps * (0.5 * (Ex[:, 1:] ** 2 + Ex[:, :-1] ** 2) * 0.5 + 0) + Hz**2
            u = u + g.eps * 0.5 * (Ey[1:] ** 2 + Ey[:-1] ** 2) * 0.5
            up = np.zeros((tx * tile, ty * tile))
            up[:nx, :ny] = u
            tmax = up.reshape(tx, tile, ty, tile).max(axis=(1, 3))
            e_ref = max(e_ref, float(tmax.max()))
            peak_norm = max(peak_norm, float(np.sqrt(0.5 * np.sum(u))))
            hot = tmax > tau * e_ref
            # the source tiles are hot while the source is on
            if t < 2 * t0:
                hot[si // tile, sj0 // tile : (sj1 - 1) // tile + 1] = True
            new = hot.copy()
            new[1:] |= hot[:-1]
            new[:-1] |= hot[1:]
            new[:, 1:] |= new[:, :-1].copy()
            new[:, :-1] |= new[:, 1:].copy()
            # drop fields of tiles that become inactive
            off = active & ~new
            if off.any():
                m = expand(off)
                d = np.sum(g.eps[m] * 0) + 0.0
                mx = np.zeros_like(Ex, bool)
                mx[:, :ny] |= m
                my = np.zeros_like(Ey, bool)
                my[:nx] |= m
                d = float(np.sum(Hz[m] ** 2) + np.sum(g.epx[mx] * Ex[mx] ** 2) + np.sum(g.epy[my] * Ey[my] ** 2))
                dropped_norm += np.sqrt(0.5 * d)
                Hz[m] = 0
                Ex[mx] = 0
                Ey[my] = 0
            active = new
            cellmask = expand(active)
            mEx = np.zeros_like(Ex, bool)
            mEx[:, :ny] |= cellmask
            mEx[:, 1:] |= cellmask
            mEy = np.zeros_like(Ey, bool)
            mEy[:nx] |= cellmask
            mEy[1:] |= cellmask
        # H update
        dH = (Ey[1:, :] - Ey[:-1, :]) - (Ex[:, 1:] - Ex[:, :-1])
        Hz = g.da * Hz - g.db * dH
        if tau > 0:
            leak = float(np.sum(Hz[~cellmask] ** 2))
            Hz *= cellmask
        # E update
        Ex[:, 1:-1] = g.cax[:, 1:-1] * Ex[:, 1:-1] + g.cbx[:, 1:-1] * (Hz[:, 1:] - Hz[:, :-1])
        Ey[1:-1, :] = g.cay[1:-1, :] * Ey[1:-1, :] - g.cby[1:-1, :] * (Hz[1:, :] - Hz[:-1, :])
        # source: soft Ey current
        s = np.exp(-(((t + 0.5 * dt) - t0) ** 2) * (2 * np.pi * fw) ** 2 / 2) * np.sin(2 * np.pi * f0 * (t + 0.5 * dt))
        Ey[si, sj0:sj1] -= g.cby[si, sj0:sj1] * s * sprof
        injected += float(np.sum(g.cby[si, sj0:sj1] * s * sprof) ** 2)
        src_spec += s * np.exp(1j * w * (t + 0.5 * dt)) * dt
        if tau > 0:
            leak += float(np.sum(g.epx[~mEx] * Ex[~mEx] ** 2) + np.sum(g.epy[~mEy] * Ey[~mEy] ** 2))
            if leak > 0:
                dropped_norm += np.sqrt(0.5 * leak)
            Ex *= mEx
            Ey *= mEy
            updated += int(cellmask.sum())
        else:
            updated += nx * ny
        # DFT of Hz projected on the output profiles
        ph = np.exp(1j * w * (t + dt)) * dt
        for o, (oi, j0, j1, prof) in enumerate(outs):
            acc[o] += float(np.dot(Hz[oi, j0:j1], prof)) * ph
        if record_every and n % record_every == 0:
            frac_hist.append(float(cellmask.mean()) if (tau > 0 and cellmask is not None) else 1.0)
        if tau <= 0 and n % 50 == 0:
            trace.append((n + 1, acc.copy()))
            en = float(np.sum(Hz**2) + np.sum(g.epx * Ex**2) + np.sum(g.epy * Ey**2))
            np_peak[0] = max(np_peak[0], en)
            if t > 2 * t0 and en < shutoff**2 * np_peak[0]:
                break
    return {"amps": acc / src_spec[None, :], "updated": updated, "steps": n + 1, "full": (n + 1) * nx * ny, "dropped_norm": dropped_norm / peak_norm, "frac_hist": frac_hist, "trace": [(m, a / src_spec[None, :]) for m, a in trace]}


np_peak = [0.0]


def build_case(case, args):
    """Return eps map (nx, ny) in cells, source list and output list."""
    ns = slab_index()
    e_hi, e_lo = ns**2, 1.444**2
    pitch = args.pitch
    if case == "i1":
        z = np.load(args.i1)
        rh = z["rho_hat"]
        L = 4.0
        dpitch = 1.0
        nports = 4
        stub, mx, my = 1.2, 2.2, 1.0
        X = L + 2 * mx
        Y = L + 2 * my
        nx, ny = int(round(X / DX_UM)), int(round(Y / DX_UM))
        xs = (np.arange(nx) + 0.5) * DX_UM - X / 2
        ys = (np.arange(ny) + 0.5) * DX_UM - Y / 2
        eps = np.full((nx, ny), e_lo)
        py = [(j - 1.5) * dpitch for j in range(nports)]
        for y in py:
            eps[:, np.abs(ys - y) <= 0.25] = e_hi
        from scipy.interpolate import RegularGridInterpolator

        gx = np.linspace(-L / 2, L / 2, rh.shape[0])
        it = RegularGridInterpolator((gx, gx), rh, bounds_error=False, fill_value=None)
        inside = (np.abs(xs)[:, None] <= L / 2) & (np.abs(ys)[None, :] <= L / 2)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        r = it(np.stack([XX[inside], YY[inside]], -1))
        eps[inside] = e_lo + (e_hi - e_lo) * np.clip(r, 0, 1)
        xin, xout = -L / 2 - stub, L / 2 + stub
        return eps, xs, ys, py, xin, xout
    if case.startswith("rand"):
        # a free-form random scatterer of N ports: N*pitch wide, depth = width
        N = args.N
        W = N * pitch
        L = args.depth if args.depth else W
        stub, mx, my = 1.2, 2.2, 1.0
        X, Y = L + 2 * mx, W + 2 * my
        nx, ny = int(round(X / DX_UM)), int(round(Y / DX_UM))
        xs = (np.arange(nx) + 0.5) * DX_UM - X / 2
        ys = (np.arange(ny) + 0.5) * DX_UM - Y / 2
        eps = np.full((nx, ny), e_lo)
        py = [(j - (N - 1) / 2) * pitch for j in range(N)]
        for y in py:
            eps[:, np.abs(ys - y) <= 0.25] = e_hi
        rng = np.random.default_rng(args.seed)
        inside = (np.abs(xs)[:, None] <= L / 2) & (np.abs(ys)[None, :] <= W / 2)
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(rng.standard_normal((nx, ny)), 0.08 / DX_UM)
        noise /= noise.std()
        if case == "rand_binary":  # high contrast, strongly scattering
            fill = (noise > 0).astype(float)
            eps[inside] = (e_lo + (e_hi - e_lo) * fill)[inside]
        else:  # weak: slab index with a small modulation (rand_weak)
            eps[inside] = (e_hi + args.contrast * (e_hi - e_lo) * np.tanh(noise))[inside]
        return eps, xs, ys, py, -L / 2 - stub, L / 2 + stub
    if case == "eps":
        z = np.load(args.eps)
        return z["eps"], z["xs"], z["ys"], list(z["py"]), float(z["xin"]), float(z["xout"])
    raise ValueError(case)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="i1")
    ap.add_argument("--i1", default="design.npz")
    ap.add_argument("--eps", default=None)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--pitch", type=float, default=1.5)
    ap.add_argument("--depth", type=float, default=0.0)
    ap.add_argument("--contrast", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--taus", default="1e-2,1e-3,1e-4,1e-5,1e-6,1e-8")
    ap.add_argument("--tile", type=int, default=30)
    ap.add_argument("--inputs", default="0", help="comma list of input ports, or 'all'")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    eps, xs, ys, py, xin, xout = build_case(a.case, a)
    g = Grid(eps)
    k0 = 2 * np.pi * DX_UM / LAM_UM
    f0 = 1.0 / (LAM_UM / DX_UM)  # c = 1 cell / unit time
    freqs = f0 * np.array([1.55 / 1.56, 1.0, 1.55 / 1.54])
    hw = int(round(0.5 * a.pitch / DX_UM)) if a.case != "i1" else 15
    ports = []
    for y in py:
        j = int(np.argmin(np.abs(ys - y)))
        ports.append((j - hw, j + hw))
    jl = ports[0]
    prof, beta = slab_mode(eps[int(np.argmin(np.abs(xs - xin))), jl[0] : jl[1]], k0)
    iin = int(np.argmin(np.abs(xs - xin)))
    iout = int(np.argmin(np.abs(xs - xout)))
    outs = [(iout, j0, j1, prof) for j0, j1 in ports]
    L_um = xs[-1] - xs[0]
    steps = a.steps or int(30 * slab_index() * L_um / DX_UM / g.dt + 6000)
    ins = list(range(len(py))) if a.inputs == "all" else [int(v) for v in a.inputs.split(",")]
    res = {"case": a.case, "args": vars(a), "cells": int(eps.size), "runs": []}
    for q in ins:
        src = (iin, ports[q][0], ports[q][1], prof)
        np_peak[0] = 0.0
        t0 = time.time()
        ref = run(g, src, outs, freqs, steps, tau=0.0)
        n_ref = ref["steps"]
        print(f"input {q}: reference {n_ref} steps, {time.time() - t0:.0f}s, |S|^2 sum {np.sum(np.abs(ref['amps'][:, 1]) ** 2):.3f} (arbitrary norm)", flush=True)
        for tau in [float(v) for v in a.taus.split(",")]:
            t0 = time.time()
            r = run(g, src, outs, freqs, n_ref, tau=tau, tile=a.tile, record_every=50)
            err = float(np.max(np.abs(r["amps"] - ref["amps"])) / np.max(np.abs(ref["amps"])))
            inj = np.sqrt(np.sum(np.abs(ref["amps"]) ** 2))
            # baseline: the full run stopped early at the same number of cell updates
            n_eq = r["updated"] / (g.nx * g.ny)
            m_eq, a_eq = min(ref["trace"], key=lambda ma: abs(ma[0] - n_eq))
            err_trunc = float(np.max(np.abs(a_eq - ref["amps"])) / np.max(np.abs(ref["amps"])))
            row = {"input": q, "err_truncated_same_cost": err_trunc, "tau": tau, "speedup": ref["full"] / max(1, r["updated"]), "active_fraction": r["updated"] / ref["full"], "rel_err": err, "dropped_norm": r["dropped_norm"], "frac_hist": r["frac_hist"]}
            res["runs"].append(row)
            print(f"  tau {tau:.0e}: active {row['active_fraction']:.3f} (speed-up {row['speedup']:.2f}), max rel. port error {err:.2e} (full run stopped at equal cost: {err_trunc:.2e}), bound (dropped / peak energy norm) {r['dropped_norm']:.2e}  [{time.time() - t0:.0f}s]", flush=True)
            del inj
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    tag = f"window_{a.case}" + (f"_N{a.N}_d{a.depth:g}_c{a.contrast:g}" if a.case.startswith("rand") else "") + (Path(a.eps).stem if a.eps else "")
    (out / f"{tag}.json").write_text(json.dumps(res))
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
