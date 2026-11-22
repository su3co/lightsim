"""Density -> polygons (for GDS export of an optimised design)."""

from __future__ import annotations

import numpy as np

__all__ = ["density_polygons"]


def density_polygons(density, min, max, level=0.5, upsample=4, pad=True):
    """Contours of a 2-D density at ``level`` as polygons.

    ``density[ix, iy]`` has its nodes spanning ``[min, max]`` inclusive. It is
    resampled ``upsample`` times (bilinear, the same interpolation as the
    index grid) and contoured with marching squares (``contourpy``, shipped
    with matplotlib). With ``pad`` the outside of the rectangle counts as
    empty so every polygon is closed.

    Returns a list of ``(exterior, holes)``: ``exterior`` an (n, 2) array of
    xy points in metres, ``holes`` a list of such arrays.
    """
    import contourpy
    from scipy.ndimage import zoom

    d = np.asarray(density, dtype=float)
    nx, ny = d.shape
    if upsample > 1:
        d = zoom(d, ((nx - 1) * upsample + 1) / nx, order=1, mode="nearest", grid_mode=False)
        d = d[: (nx - 1) * upsample + 1, : (ny - 1) * upsample + 1]
    x = np.linspace(min[0], max[0], d.shape[0])
    y = np.linspace(min[1], max[1], d.shape[1])
    if pad:
        dx, dy = x[1] - x[0], y[1] - y[0]
        d = np.pad(d, 1, constant_values=0.0)
        # the padding ring sits a vanishing distance outside, so walls are sharp
        x = np.concatenate([[x[0] - 1e-3 * dx], x, [x[-1] + 1e-3 * dx]])
        y = np.concatenate([[y[0] - 1e-3 * dy], y, [y[-1] + 1e-3 * dy]])
    gen = contourpy.contour_generator(x=x, y=y, z=d.T, fill_type=contourpy.FillType.OuterOffset)
    polys, offsets = gen.filled(level, np.inf)
    out = []
    for pts, off in zip(polys, offsets):
        rings = [pts[off[i] : off[i + 1]] for i in range(len(off) - 1)]
        out.append((rings[0], rings[1:]))
    return out
