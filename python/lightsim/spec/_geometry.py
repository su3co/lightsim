"""Shapes and structures."""

from __future__ import annotations

from typing import Iterable, Sequence

from ._util import _axis, _v3, Vec3


def Box(min: Vec3, max: Vec3) -> dict:
    """Axis-aligned box.

    Parameters
    ----------
    min, max : (x, y, z)
        Opposite corners (m). The box may extend beyond the simulation region;
        only the part inside is used.

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {"type": "box", "min": _v3(min), "max": _v3(max)}


def Sphere(center: Vec3, radius: float) -> dict:
    """Sphere.

    Parameters
    ----------
    center : (x, y, z)
        Centre (m).
    radius : float
        Radius (m).

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {"type": "sphere", "center": _v3(center), "radius": float(radius)}


def Cylinder(center: Vec3, radius: float, length: float | None = None, axis: int | str | None = None) -> dict:
    """Circular cylinder.

    Parameters
    ----------
    center : (x, y, z) or (a, b)
        Centre of the cylinder (m), half-way along its length. In a 2-D
        simulation two in-plane coordinates are enough.
    radius : float
        Radius (m).
    length : float, optional
        Length (m) along ``axis``, centred on ``center``. Required in 3-D; in
        a 2-D simulation it defaults to more than the slab.
    axis : int or str, optional
        Direction of the cylinder axis: 0, 1, 2 or "x", "y", "z". Default z in
        3-D and the invariant axis in 2-D, where ``Cylinder([a, b], radius)``
        is a disc.

    Returns the shape dict for ``Simulation.add_structure``.
    """
    return {
        "type": "cylinder",
        "center": _v3(center),
        "radius": float(radius),
        "length": None if length is None else float(length),
        "axis": None if axis is None else _axis(axis),
    }


def PolySlab(vertices: Iterable[Sequence[float]], slab_min: float | None = None, slab_max: float | None = None, axis: int | str | None = None) -> dict:
    """Polygon extruded along an axis.

    Parameters
    ----------
    vertices : sequence of (a, b)
        Corners of the polygon (m) in the plane normal to ``axis``, as pairs of
        coordinates on the two other axes in increasing axis order: (x, y) for
        ``axis=2``, (x, z) for ``axis=1``, (y, z) for ``axis=0``. The polygon is
        closed automatically, may be concave and is filled by the even-odd
        rule. It cannot have holes: cut such a shape into pieces.
    slab_min, slab_max : float, optional
        Extent (m) of the extrusion along ``axis``. Required in 3-D; in a 2-D
        simulation the extrusion covers the slab by default.
    axis : int or str, optional
        Extrusion axis: 0, 1, 2 or "x", "y", "z". Default z in 3-D and the
        invariant axis in 2-D, where ``PolySlab(vertices)`` is the polygon
        itself.

    Returns the shape dict for ``Simulation.add_structure``. Consecutive
    polygons of one medium and one extent are painted as their union.
    """
    return {
        "type": "poly_slab",
        "vertices": [[float(p[0]), float(p[1])] for p in vertices],
        "axis": None if axis is None else _axis(axis),
        "slab_min": None if slab_min is None else float(slab_min),
        "slab_max": None if slab_max is None else float(slab_max),
    }


def IndexGrid(min: Vec3, max: Vec3, eps=None, *, n=None, axes: Sequence[int] | None = None, eps_axes=None, n_axes=None) -> dict:
    """Imported / graded permittivity on a regular grid (Lumerical "import (n,k)").

    ``eps`` (relative permittivity) or ``n`` (refractive index) is a real numpy
    array of samples whose nodes span the box ``[min, max]`` inclusive along
    every axis it varies on; inside the box the solver uses the trilinear
    interpolation of the samples at each Yee E-component position. 3-D arrays
    are indexed ``[ix, iy, iz]``. 1-D and 2-D arrays are padded with length-1
    (= uniform) axes: by default a 1-D array varies along x and a 2-D array
    along (x, y); pass ``axes`` to choose, e.g. ``axes=(1, 2)`` for a (y, z)
    cross-section map of a waveguide running along x. A scalar gives a
    constant box.

    ``eps_axes=(eps_x, eps_y, eps_z)`` / ``n_axes`` (three arrays of the same
    shape) give a diagonal-anisotropic grid.

    The grid *replaces* the permittivity of the structure's medium inside the
    box: the medium passed to ``Simulation.add_structure`` only supplies its
    name, conductivity (loss), chi2/chi3 and permeability, and its ``eps`` is
    what the automatic mesher uses to pick the cell size, so set it to the
    largest permittivity of the grid. Dispersive, PEC and off-diagonal media
    are not allowed. The box faces are staircased (no sub-pixel smoothing, the
    grid is already a sampled field); to embed a grid in another material,
    add that structure first and let the grid overlap it.
    """
    import numpy as _np

    given = [x is not None for x in (eps, n, eps_axes, n_axes)]
    if sum(given) != 1:
        raise ValueError("give exactly one of eps, n, eps_axes, n_axes")

    def prep(a, is_index):
        if _np.iscomplexobj(a):
            raise ValueError("IndexGrid samples must be real; put loss in the medium's sigma")
        a = _np.asarray(a, dtype=float)
        if a.ndim > 3:
            raise ValueError(f"IndexGrid array must be at most 3-D, got shape {a.shape}")
        ax = tuple(range(a.ndim)) if axes is None else tuple(int(x) for x in axes)
        if len(ax) != a.ndim or len(set(ax)) != len(ax) or any(x not in (0, 1, 2) for x in ax):
            raise ValueError(f"axes={axes!r} does not match a {a.ndim}-D array")
        # order the array axes by simulation axis, then insert the uniform ones
        a = _np.transpose(a, _np.argsort(ax)) if a.ndim > 1 else a
        shape = [1, 1, 1]
        for d, x in zip(a.shape, sorted(ax)):
            shape[x] = d
        a = a.reshape(shape)
        if a.size == 0 or not _np.all(_np.isfinite(a)):
            raise ValueError("IndexGrid samples must be finite and non-empty")
        if is_index:
            a = a * a
        if _np.any(a <= 0.0):
            raise ValueError("IndexGrid permittivity must be positive")
        return a

    lo, hi = _v3(min), _v3(max)
    if any(h <= l for l, h in zip(lo, hi)):
        raise ValueError("IndexGrid needs max > min along every axis")
    out = {"type": "index_grid", "min": lo, "max": hi}
    if eps is not None or n is not None:
        a = prep(eps if eps is not None else n, n is not None)
        out["dims"] = list(a.shape)
        out["eps"] = a.ravel(order="C").tolist()
    else:
        src = eps_axes if eps_axes is not None else n_axes
        if len(src) != 3:
            raise ValueError("eps_axes / n_axes need three arrays (x, y, z)")
        arrs = [prep(a, n_axes is not None) for a in src]
        if len({a.shape for a in arrs}) != 1:
            raise ValueError("eps_axes / n_axes arrays must have the same shape")
        out["dims"] = list(arrs[0].shape)
        out["eps"] = []
        out["eps_axes"] = [a.ravel(order="C").tolist() for a in arrs]
    return out


def Structure(shape: dict, medium: str, name: str = "") -> dict:
    """A shape filled with a medium, as stored in ``Simulation.structures``.

    Parameters
    ----------
    shape : dict
        A shape from ``Box``, ``Sphere``, ``Cylinder``, ``PolySlab`` or
        ``IndexGrid``.
    medium : str
        Name of the medium, defined in the simulation or built in.
    name : str
        A label, used in error messages.

    Returns the structure dict. Normally created by ``Simulation.add_structure``.
    """
    return {"name": name, "shape": shape, "medium": medium}
