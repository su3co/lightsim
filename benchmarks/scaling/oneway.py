"""One-way propagation (leading term of the Bremmer series) against the full
frequency-domain solution, as a function of the index contrast of the design.

Scalar 2-D (TM, Ez) problem  u_xx + u_yy + k0^2 eps u = -J on the grid of
fdfd2d.py. The one-way field is marched column by column with the exact
square-root propagator of every column's transverse operator
H = D_yy + k0^2 eps(x, .) (PML in y), u <- V exp(i beta) V^-1 u with
2 - 2 cos(beta) = eig(H), the discrete counterpart of the FDFD stencil along
x. Within a column this is exact for x-invariant media; what it leaves out
is every reflection, i.e. the Bremmer terms of order >= 1. For transmission
the first omitted term is second order in the local reflection
coefficients, so the error is expected to follow the reflected power.

    python oneway.py --contrasts 0.01,0.03,0.1,0.3,1 --depth 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import fdfd2d as F
import window_fdtd as W


def y_pml_operator(eps_col, k0, npml=30, strength=2.0):
    ny = len(eps_col)
    sc, sf = F.pml_s(ny, npml, k0, strength)
    D = np.zeros((ny + 1, ny), complex)
    for i in range(ny):
        D[i, i] = 1.0
        D[i + 1, i] = -1.0
    H = -(np.diag(1 / sc) @ D.T @ np.diag(1 / sf) @ D) + k0**2 * np.diag(eps_col)
    return H


def tm_mode(eps_col, k0):
    """Fundamental mode of the closed (no PML) transverse operator on a window."""
    ny = len(eps_col)
    H = np.diag(-2.0 * np.ones(ny)) + np.diag(np.ones(ny - 1), 1) + np.diag(np.ones(ny - 1), -1) + k0**2 * np.diag(eps_col)
    w, v = np.linalg.eigh(H)
    f = v[:, -1]
    if f[np.argmax(np.abs(f))] < 0:
        f = -f
    lam = w[-1]
    beta = np.arccos(1 - lam / 2)
    return f / np.linalg.norm(f), float(beta)


def oneway(eps, k0, i0, i1, u0):
    """March u0 from column i0 to column i1 with the square-root propagator
    of every column (identical columns share one eigendecomposition)."""
    u = u0.astype(complex)
    cache = {}
    for i in range(i0, i1):
        key = eps[i].tobytes()
        if key not in cache:
            H = y_pml_operator(eps[i], k0)
            lam, V = np.linalg.eig(H)
            b = np.arccos(1 - lam.astype(complex) / 2)
            b = np.where(b.imag < 0, -b, b)
            cache[key] = V @ np.diag(np.exp(1j * b)) @ np.linalg.inv(V)
        u = cache[key] @ u
    return u


def incident(eps_col, k0, J):
    """Field launched towards +x at the column of a line source J
    (u_xx + H u = -J, discrete): V diag(i / (2 sin beta)) V^-1 J."""
    H = y_pml_operator(eps_col, k0)
    lam, V = np.linalg.eig(H)
    b = np.arccos(1 - lam.astype(complex) / 2)
    b = np.where(b.imag < 0, -b, b)
    return V @ (1j / (2 * np.sin(b)) * np.linalg.solve(V, J))


def case(contrast, N, pitch, depth, seed, layered=False):
    """A slab region (effective index of the silicon film) containing a
    design of depth ``depth`` and width N * pitch: the random binary pattern
    of window_fdtd.py with its permittivity pulled towards the slab value,
    eps = eps_slab - contrast * (eps_slab - eps_binary); contrast 1 is
    silicon / oxide. With ``layered`` the design is a stack of thin random
    layers (100 nm every 1.5 um, metalines) in the unpatterned slab. No
    waveguides: the input is a beam launched in the slab and the output the
    field on a plane behind the design, so that only the design scatters."""

    class A:
        pass

    a = A()
    a.N, a.pitch, a.depth, a.seed, a.contrast = N, pitch, depth, seed, 0.0
    eps, xs, ys, py, xin, xout = W.build_case("rand_binary", a)
    e_hi = W.slab_index() ** 2
    inside_x = np.abs(xs) <= depth / 2
    inside_y = np.abs(ys) <= N * pitch / 2
    des = np.zeros_like(eps, bool)
    des[np.ix_(inside_x, inside_y)] = True
    if layered:
        for i in np.nonzero(inside_x)[0]:
            if (xs[i] + depth / 2) % 1.5 > 0.1:
                des[i] = False
    eps = np.where(des, e_hi - contrast * (e_hi - eps), e_hi)
    return eps, xs, ys, py, xin, xout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts", default="0.01,0.03,0.1,0.3,1")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--pitch", type=float, default=1.5)
    ap.add_argument("--depth", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layered", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    k0 = 2 * np.pi * W.DX_UM / W.LAM_UM
    rows = []
    for c in [float(v) for v in a.contrasts.split(",")]:
        eps, xs, ys, py, xin, xout = case(c, a.N, a.pitch, a.depth, a.seed, a.layered)
        nx, ny = eps.shape
        # 20 cells clear of the x PML (30 cells) on both sides
        iin, iout = int(np.argmin(np.abs(xs - xin))) + 20, int(np.argmin(np.abs(xs - xout))) - 20
        t0 = time.time()
        lu = F.factor(F.operator(eps, k0, pol="tm"), eps.shape)
        t_fac = time.time() - t0
        errs, refl, t_ow = [], [], 0.0
        for q in range(len(py)):
            # a Gaussian beam of 0.5 um waist on input port q
            J = np.exp(-(((ys - py[q]) / 0.5) ** 2))
            Jf = np.zeros((nx, ny))
            Jf[iin] = J
            u = lu.solve(-Jf.ravel().astype(complex)).reshape(nx, ny)
            u_inc = incident(eps[iin], k0, J)
            t1 = time.time()
            u_ow = oneway(eps, k0, iin, iout, u_inc)
            t_ow += time.time() - t1
            sel = np.abs(ys) <= a.N * a.pitch / 2 + 1.0
            errs.append(float(np.linalg.norm(u_ow[sel] - u[iout, sel]) / np.linalg.norm(u[iout, sel])))
            # reflected field: the full solution upstream minus the source's own backward wave
            up = iin - 15
            direct = oneway(eps, k0, 0, 15, u_inc)
            refl.append(float(np.linalg.norm(u[up] - direct) / np.linalg.norm(direct)))
        row = {"contrast": c, "layered": a.layered, "rel_err_mean": float(np.mean(errs)), "rel_err_max": float(np.max(errs)), "reflected_amplitude": float(np.mean(refl)), "fdfd_factor_s": t_fac, "oneway_s": t_ow}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"oneway_N{a.N}_d{a.depth:g}{'_layered' if a.layered else ''}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
