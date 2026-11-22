"""Benchmark problems shared by every tool (NumPy only, no solver imported).

Each tool's script builds the *same* scene from a :class:`Scene`: domain,
cell size, materials, absorbing layers (inside the domain, same thickness),
waveguides, ports, pulse, simulated time, design permittivity and objective.

Conventions
-----------
* Lengths in metres. The origin is the centre of the domain; grid nodes are
  at ``lo + i * dx``. Waveguides run along x; every port plane is normal to x.
* A design is a 2-D permittivity map on the *nodes* of the design rectangle,
  ``(n + 1) x (n + 1)`` samples spanning it inclusively (extruded through the
  film in 3-D). Between nodes the permittivity is the (bi)linear
  interpolation of the samples. The gradient that is timed and checked is
  ``dL / d eps`` for every sample.
* The samples on the perimeter of the rectangle carry the permittivity of the
  surrounding scene (core where a waveguide meets the region, cladding
  elsewhere), so it does not matter whether a tool lets the design or the
  background own the perimeter.
* ``S[m, n, k]``: amplitude of the fundamental mode leaving through output
  ``m`` per unit amplitude of the fundamental mode launched at input ``n``,
  at wavelength ``k`` (power-normalised modes, so ``|S|^2`` is a transmitted
  power fraction).
* Pulse: Gaussian envelope of spectral standard deviation
  ``PULSE_FWIDTH * F0`` (``sigma_t = 1 / (2 pi sigma_f)``), peak at
  ``PULSE_OFFSET * sigma_t``.

Scenes
------
``wg2d``     2-D (E out of plane): two offset silicon waveguides in oxide and
             a 2.43 um square design between them. Cell 45, 15 or 5 nm.
``wg3d``     3-D: a 480 x 220 nm silicon wire in oxide through a
             1.6 x 1.6 um design region, 20 nm cells; 6 M, 32 M and 86 M
             cells (the domain grows, the design region does not).
``mm2d``     2-D: four inputs, four outputs, three wavelengths; the objective
             is the fidelity of the 4 x 4 transmission matrix to the discrete
             Fourier transform. Cell 45, 15 or 5 nm.
``mm3d``     the same device in 220 nm silicon on a cubic 44 nm (or 22 nm)
             grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C0 = 299_792_458.0
LAM0 = 1.55e-6
F0 = C0 / LAM0
EPS_SI = 12.25
EPS_OX = 1.444**2
PULSE_FWIDTH = 0.1  # spectral standard deviation of the pulse, in units of F0
PULSE_OFFSET = 5.0  # pulse peak, in units of sigma_t = 1 / (2 pi PULSE_FWIDTH F0)
COURANT = 0.99  # fraction of the stability limit used by the GPU FDTD codes


@dataclass
class Port:
    name: str
    x: float  # plane position
    center: tuple  # (y, z) of the waveguide axis
    half: tuple  # (hy, hz) half-size of the mode window
    direction: str  # "+" or "-": pointing INTO the device


@dataclass
class Scene:
    name: str
    ndim: int
    dx: float
    cells: tuple  # (nx, ny, nz); nz = 1 in 2-D
    npml: int  # absorbing-layer cells on every open face, inside the domain
    eps_bg: float
    eps_core: float
    waveguides: list  # [(min_xyz, max_xyz)] boxes of eps_core (the base scene)
    design_min: tuple  # (x, y)
    design_max: tuple
    design_z: tuple  # (z0, z1) of the extruded design (3-D)
    design_n: tuple  # (n_x, n_y) cells across the design: samples are (n_x + 1, n_y + 1)
    inputs: list
    outputs: list
    wavelengths: tuple
    run_time: float
    target: object = None  # complex [M, N] target matrix, or None for |S|^2
    eps_lo: float = EPS_OX
    eps_hi: float = EPS_SI
    seed: int = 0
    feature: float = 0.30e-6  # shortest wavelength of the random density
    contrast: float = 0.2  # the density spans 0.5 +- contrast
    extra: dict = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def size(self):
        return tuple(n * self.dx for n in self.cells)

    @property
    def lo(self):
        return tuple(-0.5 * s for s in self.size)

    @property
    def hi(self):
        return tuple(0.5 * s for s in self.size)

    @property
    def n_cells(self):
        return int(np.prod(self.cells))

    @property
    def freqs(self):
        return C0 / np.asarray(self.wavelengths, dtype=float)

    @property
    def pml(self):
        return self.npml * self.dx

    def dt(self, courant=COURANT):
        """Time step of an explicit Yee scheme at ``courant`` times the limit."""
        return courant * self.dx / (C0 * np.sqrt(self.ndim))

    def steps(self, courant=COURANT):
        return int(np.ceil(self.run_time / self.dt(courant)))

    @property
    def shape_mnk(self):
        return (len(self.outputs), len(self.inputs), len(self.wavelengths))

    # -- design ------------------------------------------------------------
    def design_nodes(self):
        x = np.linspace(self.design_min[0], self.design_max[0], self.design_n[0] + 1)
        y = np.linspace(self.design_min[1], self.design_max[1], self.design_n[1] + 1)
        return x, y

    def density(self):
        """Smooth random density in [0.3, 0.7] on the design nodes: a fixed set
        of plane waves (seeded), so the same physical pattern at every cell
        size. A grey design with moderate contrast, as early in an
        optimisation; the fields ring down within the simulated time."""
        x, y = self.design_nodes()
        X, Y = np.meshgrid(x, y, indexing="ij")
        rng = np.random.default_rng(self.seed)
        nwaves = 48
        kmag = 2 * np.pi / self.feature * np.sqrt(rng.uniform(0.0, 1.0, nwaves))
        ang = rng.uniform(0.0, 2 * np.pi, nwaves)
        ph = rng.uniform(0.0, 2 * np.pi, nwaves)
        amp = rng.standard_normal(nwaves)
        f = np.zeros_like(X)
        for k, a, p, c in zip(kmag, ang, ph, amp):
            f += c * np.cos(k * (np.cos(a) * X + np.sin(a) * Y) + p)
        sigma = np.sqrt(0.5 * np.sum(amp**2))
        return 0.5 + self.contrast * np.clip(f / (2.0 * sigma), -1.0, 1.0)

    def base_eps_xy(self, x, y):
        """Permittivity of the base scene (waveguides in cladding) at in-plane
        points, in the film mid-plane."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        eps = np.full(x.shape, self.eps_bg)
        tol = 1e-3 * self.dx
        for mn, mx in self.waveguides:
            inside = (x >= mn[0] - tol) & (x <= mx[0] + tol) & (y >= mn[1] - tol) & (y <= mx[1] + tol)
            eps[inside] = self.eps_core
        return eps

    def design_eps(self):
        """Design permittivity on the nodes, ``(n_x + 1, n_y + 1)``; the
        perimeter carries the base scene."""
        eps = self.eps_lo + self.density() * (self.eps_hi - self.eps_lo)
        x, y = self.design_nodes()
        X, Y = np.meshgrid(x, y, indexing="ij")
        base = self.base_eps_xy(X, Y)
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            eps[sl] = base[sl]
        return eps

    def straight_eps(self):
        """Design samples that reproduce the base scene (the waveguides
        continued through the design region): a check of each tool's S
        normalisation, since a straight wire transmits all of its mode."""
        x, y = self.design_nodes()
        X, Y = np.meshgrid(x, y, indexing="ij")
        return self.base_eps_xy(X, Y)

    def node_eps(self, design=None):
        """2-D scenes: permittivity on the nodes ``lo + (i, j) dx``,
        ``i < nx``, ``j < ny`` (the E_z points), with the design samples
        inserted; and the index of the first design node along x and y."""
        nx, ny = self.cells[0], self.cells[1]
        x = self.lo[0] + self.dx * np.arange(nx)
        y = self.lo[1] + self.dx * np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        eps = self.base_eps_xy(X, Y)
        i0 = int(round((self.design_min[0] - self.lo[0]) / self.dx))
        j0 = int(round((self.design_min[1] - self.lo[1]) / self.dx))
        d = self.design_eps() if design is None else design
        eps[i0 : i0 + d.shape[0], j0 : j0 + d.shape[1]] = d
        return eps, (i0, j0)

    def check_patches(self, count=4):
        """Design samples used for the finite-difference check: ``count``
        square patches of ``patch`` x ``patch`` samples at the same physical
        places in every tool (fractions of the design side). The patch has
        the same physical size at every cell size (one sample of the
        coarsest grid), so the perturbation stays well above single-precision
        noise on fine grids. Returns [(i0, i1, j0, j1)] (half-open)."""
        nx, ny = self.design_n
        p = int(self.extra.get("patch", 1))
        fr = [(0.30, 0.42), (0.52, 0.61), (0.71, 0.36), (0.45, 0.24), (0.18, 0.70), (0.83, 0.55)][:count]
        out = []
        for a, b in fr:
            i, j = int(round(a * nx)) - p // 2, int(round(b * ny)) - p // 2
            out.append((i, i + p, j, j + p))
        return out

    def check_direction(self):
        """Smooth direction (unit maximum, zero on the perimeter) for the
        directional finite-difference check."""
        x, y = self.design_nodes()
        X, Y = np.meshgrid(x, y, indexing="ij")
        rng = np.random.default_rng(self.seed + 1)
        f = np.zeros_like(X)
        for _ in range(16):
            k = 2 * np.pi / (2.5 * self.feature) * np.sqrt(rng.uniform())
            a, ph = rng.uniform(0, 2 * np.pi, 2)
            f += rng.standard_normal() * np.cos(k * (np.cos(a) * X + np.sin(a) * Y) + ph)
        f /= np.abs(f).max()
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0
        return f

    # -- objective ---------------------------------------------------------
    def objective(self, S):
        """``(L, g)`` with ``g = dL/dRe(S) + i dL/dIm(S)`` (so
        ``dL = Re(conj(g) dS)``). ``L = |S|^2`` for one port pair, the matrix
        fidelity ``mean_k |<T, S_k>|^2 / ||T||^4`` when a target is set."""
        S = np.asarray(S, dtype=complex)
        if self.target is None:
            return float(np.mean(np.abs(S) ** 2)), 2.0 * S / S.size
        T = np.asarray(self.target, dtype=complex)
        K = S.shape[2]
        t2 = float(np.sum(np.abs(T) ** 2))
        c = np.einsum("mn,mnk->k", np.conj(T), S)
        return float(np.sum(np.abs(c) ** 2) / (K * t2**2)), 2.0 * (c / K)[None, None, :] * T[:, :, None] / t2**2

    def describe(self):
        return {
            "scene": self.name,
            "ndim": self.ndim,
            "dx": self.dx,
            "grid": list(self.cells),
            "cells": self.n_cells,
            "pml_cells": self.npml,
            "run_time": self.run_time,
            "steps": self.steps(),
            "design_samples": int((self.design_n[0] + 1) * (self.design_n[1] + 1)),
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
            "wavelengths": list(self.wavelengths),
        }


def dft_target(n=4):
    m, k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.exp(-2j * np.pi * m * k / n) / np.sqrt(n)


# ---------------------------------------------------------------------------
# 2-D scenes (E out of plane). All lengths are multiples of H2 = 45 nm and the
# cell is H2 / r with r odd, so waveguide edges stay half a cell from the
# nodes at every size.
# ---------------------------------------------------------------------------
H2 = 45e-9
BIG = 1.0  # "infinite" waveguide length (m)


def _check_refine(dx_nm):
    r = 45.0 / float(dx_nm)
    if abs(r - round(r)) > 1e-9 or int(round(r)) % 2 == 0:
        raise ValueError("2-D cell sizes are 45 nm / (odd integer): 45, 15, 9, 5 ...")
    return int(round(r))


def wg2d(dx_nm=45, run_time=500e-15):
    r = _check_refine(dx_nm)
    h, dx = H2, H2 / r
    w2 = 2.5 * h  # half width (5 nodes of the coarsest grid)
    yi, yo = -9 * h, 9 * h
    d = 27 * h
    wgs = [((-BIG, yi - w2, -BIG), (-d + 0.5 * dx, yi + w2, BIG)), ((d - 0.5 * dx, yo - w2, -BIG), (BIG, yo + w2, BIG))]
    xp = 48 * h
    return Scene(
        name=f"wg2d_{dx_nm:g}nm",
        ndim=2,
        dx=dx,
        cells=(144 * r, 108 * r, 1),
        npml=12 * r,
        eps_bg=EPS_OX,
        eps_core=EPS_SI,
        waveguides=wgs,
        design_min=(-d, -d),
        design_max=(d, d),
        design_z=(-BIG, BIG),
        design_n=(54 * r, 54 * r),
        inputs=[Port("in", -xp, (yi, 0.0), (22 * h, BIG), "+")],
        outputs=[Port("out", xp, (yo, 0.0), (22 * h, BIG), "-")],
        wavelengths=(LAM0,),
        run_time=run_time,
        extra={"patch": r},
    )


def mm2d(dx_nm=45, run_time=600e-15):
    r = _check_refine(dx_nm)
    h, dx = H2, H2 / r
    w2 = 2.5 * h
    d = 48 * h
    ys = [-36 * h, -12 * h, 12 * h, 36 * h]
    wgs = [((-BIG, y - w2, -BIG), (-d + 0.5 * dx, y + w2, BIG)) for y in ys] + [((d - 0.5 * dx, y - w2, -BIG), (BIG, y + w2, BIG)) for y in ys]
    xp = 75 * h
    return Scene(
        name=f"mm2d_{dx_nm:g}nm",
        ndim=2,
        dx=dx,
        cells=(196 * r, 144 * r, 1),
        npml=12 * r,
        eps_bg=EPS_OX,
        eps_core=EPS_SI,
        waveguides=wgs,
        design_min=(-d, -d),
        design_max=(d, d),
        design_z=(-BIG, BIG),
        design_n=(96 * r, 96 * r),
        inputs=[Port(f"in{j}", -xp, (y, 0.0), (12 * h, BIG), "+") for j, y in enumerate(ys)],
        outputs=[Port(f"out{j}", xp, (y, 0.0), (12 * h, BIG), "-") for j, y in enumerate(ys)],
        wavelengths=(1.54e-6, 1.55e-6, 1.56e-6),
        run_time=run_time,
        target=dft_target(4),
        extra={"patch": r},
    )


# ---------------------------------------------------------------------------
# 3-D scenes
# ---------------------------------------------------------------------------
WG3D_SIZES = {
    # name: (cells, run time): the simulated time grows with the waveguide length.
    # nz is even, so the film mid-plane is on a node and its faces are half a cell from the nodes.
    "6M": ((400, 200, 76), 220e-15),
    "32M": ((800, 400, 100), 340e-15),
    "86M": ((1200, 600, 120), 460e-15),
}


def wg3d(size="6M", run_time=None):
    cells, t = WG3D_SIZES[size]
    dx = 20e-9
    L = cells[0] * dx
    wy, wz = 0.24e-6, 0.11e-6  # half width, half height: a 480 x 220 nm wire
    d = 0.8e-6
    npml = 10
    wgs = [((-BIG, -wy, -wz), (BIG, wy, wz))]
    xs, xo = -L / 2 + 14 * dx, L / 2 - 15 * dx
    half = (cells[1] * dx / 2 - 12 * dx, cells[2] * dx / 2 - 12 * dx)
    return Scene(
        name=f"wg3d_{size}",
        ndim=3,
        dx=dx,
        cells=cells,
        npml=npml,
        eps_bg=EPS_OX,
        eps_core=EPS_SI,
        waveguides=wgs,
        design_min=(-d, -d),
        design_max=(d, d),
        design_z=(-wz, wz),
        design_n=(80, 80),
        inputs=[Port("in", xs, (0.0, 0.0), half, "+")],
        outputs=[Port("out", xo, (0.0, 0.0), half, "-")],
        wavelengths=(LAM0,),
        run_time=t if run_time is None else run_time,
        extra={"patch": 2},
    )


def mm3d(dx_nm=44, run_time=400e-15):
    r = 44.0 / float(dx_nm)
    if abs(r - round(r)) > 1e-9:
        raise ValueError("mm3d cell sizes are 44 nm / integer")
    r = int(round(r))
    h, dx = 44e-9, 44e-9 / r
    w2 = 5.5 * h  # 484 nm wide
    t2 = 2.5 * h  # 220 nm film
    d = 48 * h
    ys = [-36 * h, -12 * h, 12 * h, 36 * h]
    wgs = [((-BIG, y - w2, -t2), (-d + 0.5 * dx, y + w2, t2)) for y in ys] + [((d - 0.5 * dx, y - w2, -t2), (BIG, y + w2, t2)) for y in ys]
    xp = 75 * h
    npml = 12 * r
    nz = 54 * r
    return Scene(
        name=f"mm3d_{dx_nm:g}nm",
        ndim=3,
        dx=dx,
        cells=(196 * r, 144 * r, nz),
        npml=npml,
        eps_bg=EPS_OX,
        eps_core=EPS_SI,
        waveguides=wgs,
        design_min=(-d, -d),
        design_max=(d, d),
        design_z=(-t2, t2),
        design_n=(96 * r, 96 * r),
        inputs=[Port(f"in{j}", -xp, (y, 0.0), (12 * h, 15 * h), "+") for j, y in enumerate(ys)],
        outputs=[Port(f"out{j}", xp, (y, 0.0), (12 * h, 15 * h), "-") for j, y in enumerate(ys)],
        wavelengths=(1.54e-6, 1.55e-6, 1.56e-6),
        run_time=run_time,
        target=dft_target(4),
        extra={"patch": r},
    )


def make(name):
    """Scene from a name: ``wg2d_45nm``, ``wg3d_6M``, ``mm2d_15nm``, ``mm3d_44nm``."""
    kind, _, arg = name.partition("_")
    if kind == "wg3d":
        return wg3d(arg or "6M")
    nm = float(arg[:-2]) if arg else None
    if kind == "wg2d":
        return wg2d(nm or 45)
    if kind == "mm2d":
        return mm2d(nm or 45)
    if kind == "mm3d":
        return mm3d(nm or 44)
    raise ValueError(f"unknown scene {name!r}")


ALL = ["wg2d_45nm", "wg2d_15nm", "wg2d_5nm", "wg3d_6M", "wg3d_32M", "wg3d_86M", "mm2d_45nm", "mm2d_15nm", "mm2d_5nm", "mm3d_44nm", "mm3d_22nm"]


if __name__ == "__main__":
    for n in ALL:
        s = make(n)
        d = s.describe()
        print(f"{n:11s} grid {d['grid']}  cells {d['cells']:>10,d}  steps {d['steps']:>6d}  design {d['design_samples']:>7d}  S {s.shape_mnk}")
