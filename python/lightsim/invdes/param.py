"""Density parametrisation of a planar design region.

    rho in [0, 1]  --filter-->  rho~  --tanh projection (beta, eta)-->  rho^
    eps = eps_base + sum_l coeff_l u_l(rho^)

The density lives on a 2-D grid (one etch layer extruded through the device
slab). With two density fields (partial etch) the level fields are
``u_0 = rho^_0`` (material present in the lower part of the slab) and
``u_1 = rho^_0 rho^_1`` (material present over the full height), so the upper
level never overhangs the lower one.

Every step has a hand-written transpose, so ``Design.vjp`` maps dL/d(eps) on
the grid samples back to dL/d(rho) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .smatrix import DesignGrid

__all__ = ["Design", "conic_kernel", "gaussian_kernel", "filter_density", "project", "binarization"]


def conic_kernel(radius_px: float):
    """Conic (hat) filter of radius ``radius_px`` pixels: w = max(0, 1 - d/R)."""
    r = int(np.ceil(radius_px))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.clip(1.0 - np.hypot(x, y) / max(radius_px, 1e-12), 0.0, None)
    return k / k.sum()


def gaussian_kernel(radius_px: float):
    """Gaussian filter whose full width at half maximum is ``radius_px``."""
    sig = max(radius_px, 1e-12) / 2.3548
    r = int(np.ceil(3 * sig))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sig * sig))
    return k / k.sum()


def filter_density(rho, kernel, transpose=False):
    """Normalised convolution ``(K * rho) / (K * 1)`` (zero padding, so the
    weights renormalise at the edges). ``transpose=True`` applies the adjoint
    linear map."""
    if kernel is None or kernel.size == 1:
        return np.array(rho, dtype=float)
    from scipy.signal import fftconvolve  # here: scipy.signal costs 0.2 s to import

    norm = fftconvolve(np.ones_like(rho, dtype=float), kernel, mode="same")
    if transpose:
        return fftconvolve(rho / norm, kernel[::-1, ::-1], mode="same")
    return fftconvolve(rho, kernel, mode="same") / norm


def project(x, beta: float, eta: float = 0.5):
    """Smoothed Heaviside: (tanh(b e) + tanh(b (x - e))) / (tanh(b e) + tanh(b (1 - e)))."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / den


def _project_grad(x, beta, eta):
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta / np.cosh(np.clip(beta * (x - eta), -300, 300)) ** 2 / den


def binarization(u):
    """1 for a fully binary density, 0 for uniform grey: 1 - mean(4 u (1 - u))."""
    u = np.asarray(u, dtype=float)
    return float(1.0 - np.mean(4.0 * u * (1.0 - u)))


@dataclass
class _Layer:
    zmin: float
    zmax: float
    eps_base: float
    coeffs: tuple
    eps_max: float


class Design:
    """A planar density design.

    Parameters
    ----------
    min, max : (x, y) corners of the design rectangle (metres).
    shape : (nx, ny) density samples; the sample nodes span the rectangle
        inclusive, like :func:`lightsim.IndexGrid`.
    layers : list of ``(zmin, zmax, eps_base, coeffs)``: one index grid per
        entry with ``eps = eps_base + sum_l coeffs[l] u_l``. Use the
        constructors :meth:`slab`, :meth:`partial_etch` and :meth:`effective_index`.
    fields : number of density fields (1, or 2 for a partial etch).
    filter_radius : radius of the conic filter (metres); the minimum feature
        size is about twice the radius. ``filter="gaussian"`` uses a Gaussian
        of that FWHM.
    eta : projection threshold.
    fixed_solid / fixed_void : boolean (nx, ny) masks of pixels held at 1 / 0
        (waveguide stubs, cladding gaps). Applied before the filter (so the
        filter sees them) and again after the projection.
    symmetry : subset of ``("x", "y")``: mirror-symmetrise the density about
        the centre line normal to that axis.
    """

    def __init__(self, min, max, shape, layers, fields=1, filter_radius=0.0, filter="conic", eta=0.5, fixed_solid=None, fixed_void=None, symmetry=(), medium_sigma=0.0, name="design"):
        self.min = [float(v) for v in min[:2]]
        self.max = [float(v) for v in max[:2]]
        self.shape = (int(shape[0]), int(shape[1]))
        self.fields = int(fields)
        self.layers = [_Layer(float(z0), float(z1), float(e0), tuple(float(c) for c in cs), float(e0 + sum(c for c in cs if c > 0))) for z0, z1, e0, cs in layers]
        for l in self.layers:
            if len(l.coeffs) != self.fields:
                raise ValueError("every layer needs one coefficient per density field")
        self.eta = float(eta)
        self.pixel = ((self.max[0] - self.min[0]) / (self.shape[0] - 1), (self.max[1] - self.min[1]) / (self.shape[1] - 1))
        rpx = float(filter_radius) / (0.5 * (self.pixel[0] + self.pixel[1]))
        self.filter_radius = float(filter_radius)
        self.kernel = None if rpx <= 0.5 else (conic_kernel(rpx) if filter == "conic" else gaussian_kernel(rpx))
        self.fixed_solid = None if fixed_solid is None else np.asarray(fixed_solid, dtype=bool)
        self.fixed_void = None if fixed_void is None else np.asarray(fixed_void, dtype=bool)
        self.symmetry = tuple(symmetry)
        self.name = name
        self.medium_sigma = medium_sigma

    # -- constructors --------------------------------------------------
    @classmethod
    def slab(cls, min, max, z, eps_lo, eps_hi, shape=None, pixel=None, **kw):
        """Fully etched layer between ``z = (zmin, zmax)`` (3-D)."""
        shape = cls._shape(min, max, shape, pixel)
        return cls(min, max, shape, [(z[0], z[1], eps_lo, (eps_hi - eps_lo,))], fields=1, **kw)

    @classmethod
    def partial_etch(cls, min, max, z, eps_lo, eps_hi, shape=None, pixel=None, **kw):
        """Two etch depths (3-D): ``z = (zmin, zmid, zmax)``. Field 0 is the
        lower part of the slab, field 1 the upper part (only where field 0 is)."""
        shape = cls._shape(min, max, shape, pixel)
        d = eps_hi - eps_lo
        return cls(min, max, shape, [(z[0], z[1], eps_lo, (d, 0.0)), (z[1], z[2], eps_lo, (0.0, d))], fields=2, **kw)

    @classmethod
    def effective_index(cls, min, max, n_lo, n_hi, n_partial=None, shape=None, pixel=None, **kw):
        """2.5-D effective-index design: one z-invariant grid whose
        permittivity runs from ``n_lo^2`` (etched) to ``n_hi^2`` (slab);
        ``n_partial`` (slab index of the partially etched film) adds the
        second density field."""
        shape = cls._shape(min, max, shape, pixel)
        if n_partial is None:
            return cls(min, max, shape, [(-1.0, 1.0, n_lo**2, (n_hi**2 - n_lo**2,))], fields=1, **kw)
        return cls(min, max, shape, [(-1.0, 1.0, n_lo**2, (n_partial**2 - n_lo**2, n_hi**2 - n_partial**2))], fields=2, **kw)

    @staticmethod
    def _shape(mn, mx, shape, pixel):
        if shape is not None:
            return shape
        if pixel is None:
            raise ValueError("give shape or pixel")
        return (int(round((mx[0] - mn[0]) / pixel)) + 1, int(round((mx[1] - mn[1]) / pixel)) + 1)

    # -- simulation side -----------------------------------------------
    def grids(self):
        """The :class:`DesignGrid` list for :class:`SMatrixProblem`."""
        from ..spec import Medium

        out = []
        for i, l in enumerate(self.layers):
            name = self.name if len(self.layers) == 1 else f"{self.name}_{i}"
            med = Medium(f"invdes_{name}", eps=l.eps_max, sigma=self.medium_sigma)
            out.append(DesignGrid([self.min[0], self.min[1], l.zmin], [self.max[0], self.max[1], l.zmax], self.shape, med, name=name))
        return out

    def coordinates(self):
        """``(x, y)``: the coordinates (metres) of the density samples along
        the two axes of the design rectangle, ``shape[0]`` and ``shape[1]``
        values from ``min`` to ``max`` inclusive."""
        return np.linspace(self.min[0], self.max[0], self.shape[0]), np.linspace(self.min[1], self.max[1], self.shape[1])

    def random(self, seed=0, mean=0.5, amplitude=0.0):
        """A starting density: ``mean`` plus smooth noise of ``amplitude``."""
        rng = np.random.default_rng(seed)
        r = np.full((self.fields,) + self.shape, float(mean))
        if amplitude:
            for f in range(self.fields):
                n = filter_density(rng.standard_normal(self.shape), self.kernel)
                r[f] += amplitude * n / (np.abs(n).max() + 1e-30)
        r = np.clip(r, 0, 1)
        return r[0] if self.fields == 1 else r

    # -- the chain -----------------------------------------------------
    def _as_fields(self, rho):
        rho = np.asarray(rho, dtype=float)
        want = (self.fields,) + self.shape
        if rho.shape == self.shape and self.fields == 1:
            return rho[None]
        if rho.shape != want:
            raise ValueError(f"density has shape {rho.shape}, expected {want if self.fields > 1 else self.shape}")
        return rho

    def _sym(self, a):
        for s in self.symmetry:
            ax = {"x": 0, "y": 1}[s]
            a = 0.5 * (a + np.flip(a, axis=ax))
        return a

    def _fix(self, a, grad=False):
        if self.fixed_solid is not None:
            a = np.where(self.fixed_solid, 0.0 if grad else 1.0, a)
        if self.fixed_void is not None:
            a = np.where(self.fixed_void, 0.0, a)
        return a

    def filtered(self, rho):
        """rho~ : symmetrised, clamped at the fixed pixels, filtered."""
        return np.stack([filter_density(self._fix(self._sym(r)), self.kernel) for r in self._as_fields(rho)])

    def projected(self, rho, beta, eta=None):
        """rho^ per density field (fixed pixels enforced), shape (fields, nx, ny)."""
        eta = self.eta if eta is None else eta
        return np.stack([self._fix(project(f, beta, eta)) for f in self.filtered(rho)])

    def levels(self, rho, beta, eta=None):
        """The level fields u_l (cumulative products of rho^)."""
        return np.cumprod(self.projected(rho, beta, eta), axis=0)

    def eps(self, rho, beta, eta=None):
        """Permittivity arrays, one per grid (shape (nx, ny, 1))."""
        u = self.levels(rho, beta, eta)
        return [(l.eps_base + np.tensordot(np.asarray(l.coeffs), u, axes=1))[:, :, None] for l in self.layers]

    def vjp(self, rho, beta, grads, eta=None):
        """dL/d(rho) from dL/d(eps) on the grids (``grads``: one array per grid)."""
        eta = self.eta if eta is None else eta
        ft = self.filtered(rho)
        ph = np.stack([self._fix(project(f, beta, eta)) for f in ft])
        u = np.cumprod(ph, axis=0)
        gu = np.zeros_like(u)
        for l, g in zip(self.layers, grads):
            g2 = np.asarray(g, dtype=float).reshape(self.shape)
            for i, c in enumerate(l.coeffs):
                gu[i] += c * g2
        # u_l = prod_{j <= l} ph_j
        gp = np.zeros_like(ph)
        for j in range(self.fields):
            for l in range(j, self.fields):
                others = np.ones(self.shape)
                for t in range(l + 1):
                    if t != j:
                        others = others * ph[t]
                gp[j] += gu[l] * others
        out = np.empty_like(ph)
        for j in range(self.fields):
            g = self._fix(gp[j], grad=True) * _project_grad(ft[j], beta, eta)
            g = filter_density(g, self.kernel, transpose=True)
            out[j] = self._sym(self._fix(g, grad=True))
        return out[0] if self.fields == 1 else out

    def binarization(self, rho, beta, eta=None):
        """Binarisation of the projected density over the free pixels."""
        ph = self.projected(rho, beta, eta)
        free = np.ones(self.shape, dtype=bool)
        if self.fixed_solid is not None:
            free &= ~self.fixed_solid
        if self.fixed_void is not None:
            free &= ~self.fixed_void
        return binarization(ph[:, free])
