"""Shape parametrisations: polygons and level sets on the design grid.

    parameters --shape--> density on the design grid --> permittivity --> S --> objective

A :class:`ShapeDesign` replaces the free pixel density of a :class:`Design`
by a few shape parameters. The density at every sample of the design grid is
computed from the shapes:

* :class:`Polygon`: the exact fraction of the sample's pixel cell (the
  ``pixel[0] x pixel[1]`` rectangle centred on the sample) that the polygon
  covers (area-fraction, anti-aliased rasterisation, :func:`polygon_density`).
  The fraction is a continuous, piecewise-quadratic function of the vertex
  coordinates, so the objective is a smooth function of them except where an
  edge is exactly parallel to a grid axis.
* :class:`LevelSet`: a smoothed Heaviside of a parametrised signed function
  ``phi(p, x, y)`` (positive inside), ``rho = H(phi / width)``, for shapes of
  fixed topology described by a function, for example a width profile w(x)
  (:meth:`LevelSet.width_profile`).

Each step has its exact transpose (vector-Jacobian product), so
``ShapeDesign.vjp`` turns the permittivity gradient that
:class:`SMatrixProblem` returns into ``d objective / d parameter``. The
derivative of the covered area with respect to a vertex is the boundary
integral of the outward normal times the hat function of that vertex along
its two edges, clipped to each pixel cell; it is the exact derivative of the
rasterised area, not an approximation of it.

The parameters seen by the optimiser are normalised to ``[0, 1]`` between
per-parameter bounds, so :func:`optimize` (projected Adam or L-BFGS-B on
``[0, 1]``) and :class:`InverseDesign` work unchanged and the bounds are hard
constraints (a minimum width, a vertex that must stay in the design region).

See ``docs/modules/shapes.md``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["polygon_density", "polygon_density_vjp", "polygon_is_simple", "smooth_heaviside", "Polygon", "LevelSet", "ShapeDesign"]


# ----------------------------------------------------------------------------
# area-fraction rasterisation of a polygon and its transpose
# ----------------------------------------------------------------------------


def _cell_edges(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("a design axis needs at least two samples")
    p = (x[-1] - x[0]) / (x.size - 1)
    return x[0] - 0.5 * p + p * np.arange(x.size + 1), p


def _signed_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _pieces(a, b, xe, ye):
    """Split the segment a -> b at the cell lines. Returns the parameter
    interval (t0, t1) of every piece, the cell (ix, iy) holding it (-1 or
    n outside the grid) and its midpoint."""
    d = b - a
    ts = [np.array([0.0, 1.0])]
    for k, e in ((0, xe), (1, ye)):
        if d[k] != 0.0:
            t = (e - a[k]) / d[k]
            ts.append(t[(t > 0.0) & (t < 1.0)])
    t = np.unique(np.concatenate(ts))
    t0, t1 = t[:-1], t[1:]
    pm = a[None, :] + (0.5 * (t0 + t1))[:, None] * d[None, :]
    ix = np.searchsorted(xe, pm[:, 0], side="right") - 1
    iy = np.searchsorted(ye, pm[:, 1], side="right") - 1
    ix = np.where(pm[:, 0] >= xe[-1], xe.size - 1, ix)
    iy = np.where(pm[:, 1] >= ye[-1], ye.size - 1, iy)
    return t0, t1, ix, iy, pm


def polygon_density(vertices, x, y):
    """Area fraction of every pixel cell covered by a simple polygon.

    ``vertices``: (n, 2) array of xy points (metres), either orientation, the
    last vertex joined to the first. ``x``, ``y``: the uniformly spaced sample
    coordinates of the design grid (``Design.coordinates()``); the cell of
    sample (i, j) is the pixel rectangle centred on it. Returns an
    ``(len(x), len(y))`` array in [0, 1].

    The area in each cell is the sum over the edges of the signed area
    between the edge and the bottom of the cell (Green's theorem column by
    column), so it is exact for any simple polygon, including the parts of
    it that lie outside the grid. Edges that cross each other count with
    their winding number; check with :func:`polygon_is_simple`."""
    v = np.asarray(vertices, dtype=float)
    xe, px = _cell_edges(x)
    ye, py = _cell_edges(y)
    nx, ny = xe.size - 1, ye.size - 1
    sigma = np.sign(_signed_area(v)) or 1.0
    part = np.zeros((nx, ny))
    full = np.zeros((nx, ny + 1))  # full[i, k]: full-height strips added to rows below k
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        t0, t1, ix, iy, pm = _pieces(a, b, xe, ye)
        dx = (t1 - t0) * (b[0] - a[0])
        ok = (ix >= 0) & (ix < nx) & (iy >= 0)
        inside = ok & (iy < ny)
        np.add.at(part, (ix[inside], iy[inside]), dx[inside] * (pm[inside, 1] - ye[iy[inside]]))
        np.add.at(full, (ix[ok], np.minimum(iy[ok], ny)), dx[ok] * py)
    below = np.cumsum(full[:, ::-1], axis=1)[:, ::-1][:, 1:]  # sum over k > row
    return -sigma * (part + below) / (px * py)


def polygon_density_vjp(vertices, x, y, g):
    """Transpose of :func:`polygon_density`: ``dL/d vertices`` (n, 2) from
    ``g = dL/d density`` on the grid.

    For a vertex v between the edges (u, v) and (v, w), the derivative of
    the area of cell c is ``sum over the two edges of n_e L_e int_{t in c}
    phi_v(t) dt``: the outward normal times the edge length times the
    integral of the vertex's hat function over the part of the edge inside
    the cell."""
    v = np.asarray(vertices, dtype=float)
    xe, px = _cell_edges(x)
    ye, py = _cell_edges(y)
    nx, ny = xe.size - 1, ye.size - 1
    ga = np.asarray(g, dtype=float).reshape(nx, ny) / (px * py)
    sigma = np.sign(_signed_area(v)) or 1.0
    out = np.zeros_like(v)
    n = len(v)
    for i in range(n):
        j = (i + 1) % n
        a, b = v[i], v[j]
        t0, t1, ix, iy, _ = _pieces(a, b, xe, ye)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if not inside.any():
            continue
        w = ga[ix[inside], iy[inside]]
        t0, t1 = t0[inside], t1[inside]
        it = 0.5 * (t1 * t1 - t0 * t0)  # int t dt
        i1 = (t1 - t0) - it  # int (1 - t) dt
        nl = sigma * np.array([b[1] - a[1], a[0] - b[0]])  # outward normal x length
        out[i] += nl * float(np.sum(w * i1))
        out[j] += nl * float(np.sum(w * it))
    return out


def polygon_is_simple(vertices, tol=0.0):
    """True if no two non-adjacent edges of the closed polygon intersect
    (and no edge has zero length)."""
    v = np.asarray(vertices, dtype=float)
    n = len(v)
    a, b = v, np.roll(v, -1, axis=0)
    if np.any(np.hypot(*(b - a).T) <= tol):
        return False

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    A, B = a[:, None, :], b[:, None, :]
    C, D = a[None, :, :], b[None, :, :]
    d1, d2 = orient(A, B, C), orient(A, B, D)
    d3, d4 = orient(C, D, A), orient(C, D, B)
    cross = (d1 * d2 < 0) & (d3 * d4 < 0)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    adjacent = (np.abs(i - j) <= 1) | (np.abs(i - j) == n - 1)
    return not bool(np.any(cross & ~adjacent))


# ----------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------


def smooth_heaviside(s):
    """C1 smoothed step: 0 for s <= -1, 1 for s >= 1, ``1/2 + s (3 - s^2) / 4``
    between. ``H(s) - step(s)`` is odd, so a flat boundary keeps its area."""
    s = np.clip(s, -1.0, 1.0)
    return 0.5 + 0.25 * s * (3.0 - s * s)


def _smooth_heaviside_grad(s):
    return np.where(np.abs(s) < 1.0, 0.75 * (1.0 - s * s), 0.0)


def _fd_jacobian(fn, p, steps):
    p = np.asarray(p, dtype=float)
    cols = []
    for k in range(p.size):
        e = np.zeros_like(p)
        e[k] = steps[k]
        cols.append((np.asarray(fn(p + e), dtype=float) - np.asarray(fn(p - e), dtype=float)) / (2 * steps[k]))
    return np.stack(cols, axis=-1) if cols else np.zeros(np.shape(fn(p)) + (0,))


def _bounds(b, n):
    if b is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    lo, hi = b
    return np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy(), np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()


class Polygon:
    """A polygon whose vertices are (functions of) shape parameters.

    Parameters
    ----------
    vertices : (n, 2) xy points in metres. Without ``mirror`` they are the
        closed polygon; with ``mirror`` they are one half of it, an open
        chain whose first and last points lie on the mirror line.
    free : which coordinates are parameters: ``True`` (all), ``False``
        (none: a fixed polygon), a boolean (n,) mask of free vertices or a
        boolean (n, 2) mask of free coordinates. The parameter vector lists
        the free coordinates in row order (x0, y0, x1, y1, ...).
    mirror : ``None``, ``"x"`` (mirror about the line x = ``axis``) or
        ``"y"`` (about y = ``axis``). The full polygon is the chain followed
        by its mirror image traversed backwards; the coordinate of the two end
        points normal to the line is held on it.
    bounds : ``(lo, hi)``, scalars or arrays with one value per parameter.
        Unbounded entries take the extent of the design rectangle along the
        coordinate's axis (:class:`ShapeDesign`).
    field : density field the polygon draws into (0, or 1 for the upper
        level of a two-level design).
    check : raise ``ValueError`` when the polygon is not simple (edges that
        cross); the density of a self-intersecting polygon counts winding
        numbers and is wrong.

    Use :meth:`from_function` for vertices computed from fewer parameters
    (spline control points, a width profile).
    """

    def __init__(self, vertices, free=True, mirror=None, axis=0.0, bounds=None, field=0, check=True):
        v = np.array(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or len(v) < 2:
            raise ValueError("vertices must be an (n, 2) array")
        m = np.asarray(free, dtype=bool)
        if m.ndim == 0:
            m = np.full(v.shape, bool(m))
        elif m.ndim == 1:
            m = np.repeat(m[:, None], 2, axis=1)
        if m.shape != v.shape:
            raise ValueError("free must be a bool, an (n,) or an (n, 2) mask")
        self.mirror, self.axis = mirror, float(axis)
        if mirror is not None:
            k = {"x": 0, "y": 1}[mirror]
            for e in (0, -1):
                if abs(v[e, k] - self.axis) > 1e-12:
                    raise ValueError("with mirror, the first and last vertex must lie on the mirror line")
                v[e, k] = self.axis
            m[[0, -1], k] = False
        self._base, self._free = v, m
        self._fn, self._jac = None, None
        self.p0 = v[m].copy()
        self.coord_axis = np.nonzero(m)[1] if m.any() else np.zeros(0, int)
        self.lo, self.hi = _bounds(bounds, int(m.sum()))
        self.field, self.check = int(field), check

    @classmethod
    def from_function(cls, fn, p0, bounds, jac=None, mirror=None, axis=0.0, field=0, check=True):
        """Vertices ``fn(p) -> (n, 2)`` from a parameter vector ``p``
        (``p0``: initial values, ``bounds``: ``(lo, hi)``, finite). ``jac(p)
        -> (n, 2, P)`` is the Jacobian; without it a central difference of
        ``fn`` (step 1e-6 of each parameter's range) is used, which is exact
        to rounding for maps linear in ``p`` (splines through control
        points). ``mirror`` applies to the vertices ``fn`` returns."""
        p0 = np.asarray(p0, dtype=float).ravel()
        obj = cls(fn(p0), free=False, mirror=mirror, axis=axis, field=field, check=check)
        obj._fn, obj._jac = fn, jac
        obj.p0 = p0.copy()
        obj.coord_axis = np.full(p0.size, -1)
        obj.lo, obj.hi = _bounds(bounds, p0.size)
        if not np.all(np.isfinite(obj.lo) & np.isfinite(obj.hi)):
            raise ValueError("from_function needs finite bounds")
        return obj

    @property
    def size(self):
        """Number of parameters."""
        return self.p0.size

    def chain(self, p):
        """The vertices before mirroring."""
        if self._fn is not None:
            c = np.array(self._fn(np.asarray(p, dtype=float)), dtype=float)
            if self.mirror is not None:
                c[[0, -1], {"x": 0, "y": 1}[self.mirror]] = self.axis
            return c
        c = self._base.copy()
        c[self._free] = p
        return c

    def _reflect(self, c):
        r = np.array(c, dtype=float)
        k = {"x": 0, "y": 1}[self.mirror]
        r[..., k] = 2 * self.axis - r[..., k]
        return r

    def vertices(self, p):
        """The closed polygon (m, 2) for parameters ``p`` (physical units)."""
        c = self.chain(p)
        if self.mirror is None:
            return c
        return np.concatenate([c, self._reflect(c[-2:0:-1])])

    def _chain_vjp(self, p, gc):
        if self._fn is None:
            return gc[self._free]
        if self._jac is not None:
            J = np.asarray(self._jac(np.asarray(p, dtype=float)), dtype=float)
        else:
            J = _fd_jacobian(self._fn, p, 1e-6 * (self.hi - self.lo))
        if self.mirror is not None:
            J = J.copy()
            J[[0, -1], {"x": 0, "y": 1}[self.mirror]] = 0.0
        return np.tensordot(gc, J, axes=([0, 1], [0, 1]))

    def density(self, p, x, y):
        """Area fractions on the samples ``x``, ``y`` (:func:`polygon_density`);
        ``ValueError`` if the polygon is not simple and ``check`` is set."""
        v = self.vertices(p)
        if self.check and not polygon_is_simple(v):
            raise ValueError("polygon edges cross (or an edge has zero length): the shape is no longer simple")
        return polygon_density(v, x, y)

    def vjp(self, p, x, y, g):
        """dL/dp from dL/d density."""
        gv = polygon_density_vjp(self.vertices(p), x, y, g)
        if self.mirror is not None:
            n = len(self.chain(p))
            gc = gv[:n].copy()
            gc[-2:0:-1] += self._reflect_grad(gv[n:])
        else:
            gc = gv
        return self._chain_vjp(p, gc)

    def _reflect_grad(self, g):
        r = np.array(g, dtype=float)
        r[..., {"x": 0, "y": 1}[self.mirror]] *= -1.0
        return r


class LevelSet:
    """``rho = H(phi(p, X, Y) / width)``: a smoothed Heaviside
    (:func:`smooth_heaviside`, a ramp from -width to +width) of a signed
    function that is positive inside the shape.

    Parameters
    ----------
    phi : ``phi(p, X, Y) -> array`` with the shape of ``X``; ``X``, ``Y``
        are the sample coordinates (``meshgrid(x, y, indexing="ij")``). It
        should be close to a signed distance (|grad phi| ~ 1) near the
        boundary, so that ``width`` is a length.
    p0, bounds : initial parameters and ``(lo, hi)`` (finite).
    jac : ``jac(p, X, Y) -> (P,) + X.shape``, d phi / d p. Default: central
        differences of ``phi`` (step 1e-6 of each parameter's range).
    width : half-width of the smoothed step (metres). Default: the mean
        pixel of the design.
    field : density field (0 or 1).
    """

    def __init__(self, phi, p0, bounds, jac=None, width=None, field=0):
        self.phi, self.jac = phi, jac
        self.p0 = np.asarray(p0, dtype=float).ravel().copy()
        self.lo, self.hi = _bounds(bounds, self.p0.size)
        if not np.all(np.isfinite(self.lo) & np.isfinite(self.hi)):
            raise ValueError("LevelSet needs finite bounds")
        self.coord_axis = np.full(self.p0.size, -1)
        self.width, self.field = width, int(field)

    @classmethod
    def width_profile(cls, x, widths, bounds, free=None, center=0.0, kind="cubic", axis="x", width=None, field=0):
        """A strip along ``axis`` whose full width w(s) interpolates
        ``widths`` at the stations ``x`` (``kind = "linear"`` or ``"cubic"``,
        a natural cubic spline), centred on the line ``center``:
        ``phi = w(s) / 2 - |t - center|`` (s along ``axis``, t across).
        ``free``: boolean mask of the stations whose width is a parameter
        (default: all); the others keep their value in ``widths`` (the two
        ends, where the strip meets fixed waveguides). ``bounds = (wmin,
        wmax)`` on the free widths: ``wmin`` is a minimum feature size the
        optimisation cannot go below. Outside ``[x[0], x[-1]]`` the end
        widths continue."""
        xs = np.asarray(x, dtype=float)
        w0 = np.asarray(widths, dtype=float)
        free = np.ones(xs.size, bool) if free is None else np.asarray(free, dtype=bool)
        k = {"x": 0, "y": 1}[axis]

        def basis(s):
            s = np.clip(s, xs[0], xs[-1])
            eye = np.eye(xs.size)
            if kind == "linear":
                return np.stack([np.interp(s, xs, e) for e in eye], axis=-1)
            from scipy.interpolate import CubicSpline

            return CubicSpline(xs, eye, axis=0, bc_type="natural")(s)

        cache = {}

        def B(S):
            key = (S.shape, S.tobytes())
            if key not in cache:
                cache[key] = basis(S)
            return cache[key]

        def full(p):
            w = w0.copy()
            w[free] = p
            return w

        def phi(p, X, Y):
            S, T = (X, Y) if k == 0 else (Y, X)
            return 0.5 * (B(S) @ full(p)) - np.abs(T - center)

        def jac(p, X, Y):
            S = X if k == 0 else Y
            return np.moveaxis(0.5 * B(S)[..., free], -1, 0)

        obj = cls(phi, w0[free], bounds, jac=jac, width=width, field=field)
        obj.stations, obj.free_stations, obj.all_widths = xs, free, full
        return obj

    @property
    def size(self):
        """Number of parameters."""
        return self.p0.size

    def _width(self, x, y):
        if self.width is not None:
            return float(self.width)
        return 0.5 * ((x[-1] - x[0]) / (x.size - 1) + (y[-1] - y[0]) / (y.size - 1))

    def density(self, p, x, y):
        """``H(phi / width)`` on the samples ``x``, ``y``."""
        X, Y = np.meshgrid(x, y, indexing="ij")
        return smooth_heaviside(self.phi(np.asarray(p, dtype=float), X, Y) / self._width(x, y))

    def vjp(self, p, x, y, g):
        """dL/dp from dL/d density."""
        X, Y = np.meshgrid(x, y, indexing="ij")
        p = np.asarray(p, dtype=float)
        w = self._width(x, y)
        gs = np.asarray(g, dtype=float).reshape(X.shape) * _smooth_heaviside_grad(self.phi(p, X, Y) / w) / w
        if self.jac is not None:
            J = np.asarray(self.jac(p, X, Y), dtype=float)
        else:
            J = np.moveaxis(_fd_jacobian(lambda q: self.phi(q, X, Y), p, 1e-6 * (self.hi - self.lo)), -1, 0)
        return np.tensordot(J, gs, axes=([1, 2], [0, 1]))


# ----------------------------------------------------------------------------
# shapes -> density -> permittivity
# ----------------------------------------------------------------------------


class ShapeDesign:
    """Shapes drawn on the grid of a :class:`Design`.

    ``ShapeDesign(design, shapes)`` has the interface :class:`InverseDesign`
    and :func:`optimize` expect from a design (``eps``, ``vjp``, ``grids``,
    ``binarization``), with the normalised parameter vector ``u`` in
    ``[0, 1]^P`` in place of the density: ``p = lo + u (hi - lo)``, the
    physical parameters of all shapes concatenated.

    Parameters
    ----------
    design : the :class:`Design` that sets the grid (rectangle, samples),
        the layers (``slab``, ``partial_etch``, ``effective_index``) and the
        permittivities.
    shapes : :class:`Polygon` and :class:`LevelSet` objects. Each draws into
        its ``field``; the densities of one field add up and must not
        overlap (a density above 1 raises ``ValueError``).
    projection : ``False`` (default): the density goes straight into the
        level fields, ``eps = eps_base + sum_l coeff_l u_l`` with ``u_0 =
        rho_0``, ``u_1 = rho_0 rho_1``; the ``beta`` and ``eta`` arguments
        are ignored and the filter, projection, masks and symmetry of
        ``design`` are not used. ``True``: the density goes through the
        design's own chain (filter, tanh projection at ``beta``, masks), for
        example to smooth the boundary with a filter.
    """

    def __init__(self, design, shapes, projection=False):
        self.design = design
        self.shapes = list(shapes)
        self.projection = bool(projection)
        self.x, self.y = design.coordinates()
        for s in self.shapes:
            if not 0 <= s.field < design.fields:
                raise ValueError(f"shape field {s.field} but the design has {design.fields} density field(s)")
        lo, hi, p0, self._slices = [], [], [], []
        ext = np.array([design.min, design.max])  # (2, 2): rows min/max, columns x/y
        start = 0
        for s in self.shapes:
            l, h = s.lo.copy(), s.hi.copy()
            for i, ax in enumerate(s.coord_axis):
                if ax >= 0:
                    l[i] = ext[0, ax] if not np.isfinite(l[i]) else l[i]
                    h[i] = ext[1, ax] if not np.isfinite(h[i]) else h[i]
            if np.any(h <= l) and s.size:
                raise ValueError("every parameter needs lo < hi")
            lo.append(l)
            hi.append(h)
            p0.append(s.p0)
            self._slices.append(slice(start, start + s.size))
            start += s.size
        self.lo = np.concatenate(lo) if lo else np.zeros(0)
        self.hi = np.concatenate(hi) if hi else np.zeros(0)
        self.p0 = np.concatenate(p0) if p0 else np.zeros(0)
        if np.any(self.p0 < self.lo - 1e-15) or np.any(self.p0 > self.hi + 1e-15):
            raise ValueError("initial parameters outside their bounds")

    # -- parameters ----------------------------------------------------
    @property
    def size(self):
        """Number of shape parameters."""
        return self.p0.size

    def params(self, u):
        """Physical parameters ``lo + u (hi - lo)``."""
        return self.lo + np.asarray(u, dtype=float) * (self.hi - self.lo)

    def normalize(self, p):
        """Normalised parameters ``(p - lo) / (hi - lo)``."""
        return (np.asarray(p, dtype=float) - self.lo) / (self.hi - self.lo)

    def initial(self):
        """The normalised initial parameters (from the shapes' ``p0``)."""
        return self.normalize(self.p0)

    def shape_params(self, u):
        """List of the physical parameters of each shape."""
        p = self.params(u)
        return [p[sl] for sl in self._slices]

    def polygons(self, u):
        """``[(field, vertices)]`` of the :class:`Polygon` shapes (metres),
        for GDS export; level sets are traced with :func:`density_polygons`
        on :meth:`density`."""
        return [(s.field, s.vertices(p)) for s, p in zip(self.shapes, self.shape_params(u)) if isinstance(s, Polygon)]

    # -- density -------------------------------------------------------
    def density(self, u):
        """The density on the design grid, ``(nx, ny)`` (or ``(fields, nx,
        ny)``)."""
        rho = np.zeros((self.design.fields,) + self.design.shape)
        for s, p in zip(self.shapes, self.shape_params(u)):
            rho[s.field] += s.density(p, self.x, self.y)
        if rho.max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError(f"shapes overlap: density {rho.max():.3f} > 1")
        return rho[0] if self.design.fields == 1 else rho

    def density_vjp(self, u, g):
        """dL/du from dL/d density."""
        g = np.asarray(g, dtype=float).reshape((self.design.fields,) + self.design.shape)
        out = np.zeros(self.size)
        for s, p, sl in zip(self.shapes, self.shape_params(u), self._slices):
            if s.size:
                out[sl] = s.vjp(p, self.x, self.y, g[s.field])
        return out * (self.hi - self.lo)

    # -- the Design interface --------------------------------------------
    def grids(self):
        """The :class:`DesignGrid` list (that of ``design``)."""
        return self.design.grids()

    def eps(self, u, beta=None, eta=None):
        """Permittivity arrays, one per grid (shape (nx, ny, 1))."""
        rho = self.density(u)
        if self.projection:
            return self.design.eps(rho, beta, eta)
        lv = np.cumprod(self.design._as_fields(rho), axis=0)
        return [(l.eps_base + np.tensordot(np.asarray(l.coeffs), lv, axes=1))[:, :, None] for l in self.design.layers]

    def vjp(self, u, beta, grads, eta=None):
        """dL/du from dL/d(eps) on the grids."""
        rho = self.density(u)
        if self.projection:
            return self.density_vjp(u, self.design.vjp(rho, beta, grads, eta))
        r = self.design._as_fields(rho)
        nf = self.design.fields
        gu = np.zeros_like(r)
        for l, g in zip(self.design.layers, grads):
            g2 = np.asarray(g, dtype=float).reshape(self.design.shape)
            for i, c in enumerate(l.coeffs):
                gu[i] += c * g2
        gr = np.zeros_like(r)
        for j in range(nf):
            for lev in range(j, nf):
                others = np.ones(self.design.shape)
                for t in range(lev + 1):
                    if t != j:
                        others = others * r[t]
                gr[j] += gu[lev] * others
        return self.density_vjp(u, gr)

    def binarization(self, u, beta=None, eta=None):
        """``1 - mean(4 rho (1 - rho))`` of the density (the fraction of
        grey boundary pixels, not a convergence measure for shapes)."""
        from .param import binarization

        return binarization(self.density(u))
