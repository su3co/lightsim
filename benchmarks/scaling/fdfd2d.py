"""2-D frequency-domain (FDFD) reference and scaling measurement.

TE polarisation (Hz out of plane, E in plane) on the Yee grid of
window_fdtd.py, with stretched-coordinate PML:

    d/dx (1/(eps sx) d/dx Hz)/sx... written as  Dx^e S_x^-1 eps^-1 ... + k0^2 Hz = -J

Used for (i) the cost of a sparse direct solve, factorised once and reused
for every right-hand side (all N forward and N adjoint problems of an exact
gradient at one frequency), against the size of an N-port free-form device,
and (ii) the reference for the one-way (beam propagation, Bremmer series)
comparison in oneway.py.

    python fdfd2d.py --scaling 8,16,32,64
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

DX_UM = 1.0 / 30.0
LAM_UM = 1.55


def pml_s(n, npml, k0, strength=2.0):
    """Complex stretching on the cell centres (c) and on the faces (f)."""
    def s(pos):
        d = np.zeros_like(pos)
        lo = npml - pos
        hi = pos - (n - npml)
        d = np.where(lo > 0, lo, 0) + np.where(hi > 0, hi, 0)
        sig = strength * (d / npml) ** 3
        return 1 + 1j * sig / (k0 * 1.0) * 3.0 / npml * 6

    c = s(np.arange(n) + 0.5)
    f = s(np.arange(n + 1).astype(float))
    return c, f


def operator(eps, k0, npml=30, pol="te"):
    """Sparse matrix A with A h = -source for the TE Hz field (unit cells).

    -(Dx^T Ex-like) ... assembled as  A = Dxb Sx_f^-1 Ey_inv Dxf + Dyb Sy_f^-1 Ex_inv Dyf (with c-stretch)
    + k0^2 I, all on cell centres, PEC behind the PML."""
    nx, ny = eps.shape
    sxc, sxf = pml_s(nx, npml, k0)
    syc, syf = pml_s(ny, npml, k0)
    # permittivities on the faces (harmonic mean is not needed: arithmetic as the FDTD prototype)
    ex = np.pad(eps, ((1, 1), (0, 0)), mode="edge")
    eps_xf = 0.5 * (ex[1:] + ex[:-1])  # (nx+1, ny): faces normal to x (Ey positions)
    ey = np.pad(eps, ((0, 0), (1, 1)), mode="edge")
    eps_yf = 0.5 * (ey[:, 1:] + ey[:, :-1])  # (nx, ny+1): faces normal to y (Ex positions)
    # 1-D difference matrices: centres -> faces (forward, PEC: field outside = mirrored? use zero-flux)
    def dmat(n):
        # faces 0..n, centres 0..n-1; (D h)_f = h_f - h_{f-1}, with h_{-1} = h_n = 0 (PEC for H -> E tangential 0 is simplest)
        return sp.diags([np.ones(n), -np.ones(n)], [0, -1], shape=(n + 1, n), format="csr")

    Dx1, Dy1 = dmat(nx), dmat(ny)
    Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    Dx = sp.kron(Dx1, Iy, format="csr")  # (nx+1)*ny x nx*ny
    Dy = sp.kron(Ix, Dy1, format="csr")  # nx*(ny+1) x nx*ny
    if pol == "tm":  # Ez on the cell centres: Laplacian + k0^2 eps
        eps_xf, eps_yf = np.ones_like(eps_xf), np.ones_like(eps_yf)
    wx = (1.0 / (eps_xf * sxf[:, None])).ravel()
    wy = (1.0 / (eps_yf * syf[None, :])).ravel()
    Lx = sp.diags((1.0 / sxc[:, None] * np.ones((1, ny))).ravel()) @ (Dx.T @ sp.diags(wx) @ Dx)
    Ly = sp.diags((1.0 / syc[None, :] * np.ones((nx, 1))).ravel()) @ (Dy.T @ sp.diags(wy) @ Dy)
    A = -(Lx + Ly) + k0**2 * (sp.diags(eps.ravel()) if pol == "tm" else sp.identity(nx * ny))
    return A.tocsc()


def nested_dissection(nx, ny, leaf=64):
    """Geometric nested-dissection ordering of an nx x ny grid (row-major
    index i * ny + j): recursively split the longer side by a separator
    line, order both halves first and the separator last."""
    order = []

    def rec(i0, i1, j0, j1):
        if (i1 - i0) * (j1 - j0) <= leaf:
            for i in range(i0, i1):
                order.extend(range(i * ny + j0, i * ny + j1))
            return
        if i1 - i0 >= j1 - j0:
            m = (i0 + i1) // 2
            rec(i0, m, j0, j1)
            rec(m + 1, i1, j0, j1)
            order.extend(range(m * ny + j0, m * ny + j1))
        else:
            m = (j0 + j1) // 2
            rec(i0, i1, j0, m)
            rec(i0, i1, m + 1, j1)
            order.extend(m + i * ny for i in range(i0, i1))

    import sys

    sys.setrecursionlimit(10000)
    rec(0, nx, 0, ny)
    return np.asarray(order)


class NDFactor:
    """Sparse LU (SuperLU) of A in a nested-dissection ordering."""

    def __init__(self, A, shape):
        self.p = nested_dissection(*shape)
        Ap = A[self.p][:, self.p].tocsc()
        self.lu = spl.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        self.L, self.U = self.lu.L, self.lu.U

    def solve(self, b):
        b = np.asarray(b)
        x = np.empty_like(b, dtype=complex)
        x[self.p] = self.lu.solve(np.ascontiguousarray(b[self.p]).astype(complex))
        return x


def factor(A, shape=None):
    if shape is not None:
        return NDFactor(A, shape)
    return spl.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def scaling(Ns, pitch=1.5, nrhs=4, out="results"):
    import window_fdtd as W

    k0 = 2 * np.pi * DX_UM / LAM_UM
    rows = []
    for N in Ns:
        class A_:
            pass

        a = A_()
        a.N, a.pitch, a.depth, a.seed, a.contrast = N, pitch, 0.0, 0, 0.0
        eps, xs, ys, py, xin, xout = W.build_case("rand_binary", a)
        t0 = time.time()
        A = operator(eps, k0)
        t_asm = time.time() - t0
        t0 = time.time()
        lu = factor(A, eps.shape)
        t_fac = time.time() - t0
        nnz = lu.L.nnz + lu.U.nnz
        b = np.zeros((A.shape[0], nrhs), complex)
        rng = np.random.default_rng(0)
        for r in range(nrhs):
            b[rng.integers(A.shape[0]), r] = 1.0
        t0 = time.time()
        lu.solve(b)
        t_sol = (time.time() - t0) / nrhs
        row = {"N": N, "cells": int(eps.size), "assemble_s": t_asm, "factor_s": t_fac, "solve_s_per_rhs": t_sol, "factor_nnz": int(nnz), "factor_GB": nnz * 16 / 1e9, "peak_rss_GB": rss_gb()}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del lu, A
    Path(out).mkdir(exist_ok=True)
    Path(out, "fdfd_scaling.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", default="8,16,32")
    ap.add_argument("--pitch", type=float, default=1.5)
    a = ap.parse_args()
    scaling([int(v) for v in a.scaling.split(",")], a.pitch)
