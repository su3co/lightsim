"""Cross-section geometry for the finite-element solvers.

A :class:`Geometry` is a list of :class:`Region` objects: a shape (polygon,
rectangle, circle, trapezoid) with a name, a material, a target mesh size and
a painting priority (``mesh_order``: where regions overlap, the lowest value
wins, as in gdsfactory). :meth:`Geometry.mesh` meshes it with the Rust mesher
(constrained Delaunay triangulation with quality and size refinement, no
gmsh). :func:`cross_section_geometry` builds one by cutting a gdsfactory
component along a line through its layer stack, the way gplugins prepares
femwell meshes.

Coordinates are in metres (``lightsim.um`` = 1e-6); in a cross-section the
first coordinate is the in-plane position (along the cut) and the second the
height ``z``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Polygon", "Rectangle", "Circle", "Trapezoid", "Region", "Geometry", "cross_section_geometry", "cut_component"]


@dataclass
class Polygon:
    """A polygon: an outer ring and optional holes, each a list of (x, y)
    vertices in metres. ``arcs`` optionally marks segments as circular arcs
    (``arcs[ring][i] = (cx, cy, r)`` for the segment starting at vertex i)."""

    points: list
    holes: list = field(default_factory=list)
    arcs: list | None = None

    def rings(self) -> list:
        return [np.asarray(self.points, float)] + [np.asarray(h, float) for h in self.holes]

    def contains(self, x, y) -> np.ndarray:
        """Even-odd point-in-polygon test (vectorised over points)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        inside = np.zeros(np.broadcast(x, y).shape, bool)
        for ring in self.rings():
            xa, ya = ring[:, 0], ring[:, 1]
            xb, yb = np.roll(xa, 1), np.roll(ya, 1)
            for i in range(len(xa)):
                c = ((ya[i] > y) != (yb[i] > y)) & (x < (xb[i] - xa[i]) * (y - ya[i]) / (yb[i] - ya[i] + 1e-300) + xa[i])
                inside ^= c
        return inside

    @property
    def bounds(self):
        p = np.asarray(self.points, float)
        return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def Rectangle(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    """Axis-aligned rectangle with corners (x0, y0) and (x1, y1)."""
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def Trapezoid(x_center: float, width_bottom: float, width_top: float, y0: float, y1: float) -> Polygon:
    """Symmetric trapezoid (a waveguide with slanted sidewalls)."""
    return Polygon(
        [
            (x_center - width_bottom / 2, y0),
            (x_center + width_bottom / 2, y0),
            (x_center + width_top / 2, y1),
            (x_center - width_top / 2, y1),
        ]
    )


def Circle(center, radius: float, n: int | None = None, size: float | None = None) -> Polygon:
    """Circle as a polygon whose segments are marked as arcs: the mesher
    puts the subdivision points on the circle and P2 elements are curved.
    ``n`` (default 32) is the number of input vertices; the mesher subdivides
    them further to the local mesh size."""
    if n is None:
        n = 32 if size is None else max(12, int(math.ceil(2 * math.pi * radius / size)))
    cx, cy = float(center[0]), float(center[1])
    t = 2 * np.pi * np.arange(n) / n
    pts = np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t)], 1)
    return Polygon(pts.tolist(), arcs=[[(cx, cy, float(radius))] * n])


@dataclass
class Region:
    """A tagged region of a cross-section.

    Attributes
    ----------
    shape : Polygon
        Its outline (see :func:`Rectangle`, :func:`Circle`, ...).
    name : str
        Name used to address the region (sources, conductors, mesh sizes).
    material : str
        Material name, looked up in the property tables of each solver.
    size : float or None
        Target edge length inside the region (m); ``None``: the default.
    mesh_order : int
        Priority where regions overlap; the lowest value wins.
    """

    shape: Polygon
    name: str
    material: str = ""
    size: float | None = None
    mesh_order: int = 10


class Geometry:
    """A list of regions to mesh.

    >>> g = Geometry()
    >>> g.add(Rectangle(-2e-6, -1e-6, 2e-6, 0), "box", "SiO2")       # doctest: +SKIP
    """

    def __init__(self, regions=None):
        self.regions: list[Region] = list(regions or [])

    def add(self, shape: Polygon, name: str, material: str = "", size: float | None = None, mesh_order: int = 10) -> "Geometry":
        """Append a region; returns the geometry (calls can be chained)."""
        self.regions.append(Region(shape, name, material or name, size, mesh_order))
        return self

    def names(self) -> list[str]:
        out = []
        for r in self.regions:
            if r.name not in out:
                out.append(r.name)
        return out

    @property
    def bounds(self):
        b = np.array([r.shape.bounds for r in self.regions])
        return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()

    def mesh(
        self,
        default_size: float | None = None,
        sizes: dict | None = None,
        refine: dict | None = None,
        grading: float = 0.3,
        min_angle: float = 25.0,
        max_vertices: int = 2_000_000,
    ):
        """Mesh the geometry.

        Parameters
        ----------
        default_size : float, optional
            Largest edge length (m); default: a tenth of the diagonal.
        sizes : dict, optional
            ``{region name or material: edge length}``, overriding the
            regions' own ``size``.
        refine : dict, optional
            ``{region name: edge length}``: refine along the boundary of
            these regions (interfaces), growing away from it by ``grading``.
        grading : float
            Size growth per unit distance away from small regions.
        min_angle : float
            Minimum angle (degrees) of the quality refinement (at most ~30).

        Returns a :class:`lightsim.fem.Mesh`. Region names become the tags
        (``mesh.region_names``); regions sharing a name share a tag.
        """
        from .. import _core
        from .mesh import Mesh

        sizes = dict(sizes or {})
        names = self.names()
        tag_of = {n: i for i, n in enumerate(names)}
        regions = []
        for r in self.regions:
            size = sizes.get(r.name, sizes.get(r.material, r.size))
            rings = [ring.tolist() for ring in r.shape.rings()]
            arcs = r.shape.arcs if r.shape.arcs is not None else None
            regions.append({"rings": rings, "arcs": arcs, "tag": tag_of[r.name], "priority": int(r.mesh_order), "size": float(size or 0.0)})
        refinements = []
        for name, size in (refine or {}).items():
            idx = [i for i, r in enumerate(self.regions) if r.name == name or r.material == name]
            if not idx:
                raise ValueError(f"refine: no region named {name!r}; regions: {names}")
            refinements.append((idx, float(size)))
        if default_size is None:
            x0, y0, x1, y1 = self.bounds
            default_size = math.hypot(x1 - x0, y1 - y0) / 10
        d = _core.fem_mesh(regions, float(default_size), float(grading), float(min_angle), refinements, int(max_vertices))
        materials = {}
        for r in self.regions:
            materials.setdefault(r.name, r.material)
        m = Mesh(d["nodes"], d["tris"], d["tags"], region_names=names, materials=[materials[n] for n in names], edge_mid=d["edge_mid"], _info=d)
        m.min_angle_deg = float(d["min_angle"])
        return m


# ----------------------------------------------------------------------------
# gdsfactory: cut a component through its layer stack
# ----------------------------------------------------------------------------


def _intervals_on_line(polys, p0, p1) -> list:
    """Parameter intervals [t0, t1] (0..1 along p0 -> p1) inside the union of polygons."""
    p0 = np.asarray(p0, float)
    d = np.asarray(p1, float) - p0
    out = []
    for poly in polys:
        P = np.asarray(poly, float)
        ts = []
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            e = b - a
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-30:
                continue
            w = a - p0
            t = (w[0] * e[1] - w[1] * e[0]) / den
            s = (w[0] * d[1] - w[1] * d[0]) / den
            if -1e-12 <= s < 1 - 1e-12:
                ts.append(t)
        ts = sorted(ts)
        for k in range(0, len(ts) - 1, 2):
            a, b = max(ts[k], 0.0), min(ts[k + 1], 1.0)
            if b > a:
                out.append([a, b])
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def cut_component(component, layer_stack=None, xsection_bounds=None, exclude=(), wafer_levels=None, u_offset: float = 0.0):
    """Cut a gdsfactory component along a line through its layer stack.

    ``xsection_bounds = ((x0, y0), (x1, y1))`` in **micrometres** (gdsfactory
    units, as in gplugins). Returns a list of dicts ``{name, material,
    mesh_order, zmin, zmax, intervals, sidewall_angle, width_to_z}``: for
    every layer-stack level the intervals of the cut (metres, measured from
    the start of the line plus ``u_offset``) where the level has material.
    Wafer levels (``background=True``, a ``WAFER`` layer or listed in
    ``wafer_levels``) cover the whole line.
    """
    import gdsfactory as gf

    from ..gds import _is_wafer_level, _level_polygons
    from ..spec import um

    if layer_stack is None:
        layer_stack = gf.pdk.get_layer_stack()
    (x0, y0), (x1, y1) = xsection_bounds
    p0 = np.array([x0, y0]) * um
    p1 = np.array([x1, y1]) * um
    L = float(np.hypot(*(p1 - p0)))
    out = []
    for name, lev in layer_stack.layers.items():
        if name in exclude:
            continue
        th = float(lev.thickness)
        z0 = float(lev.zmin)
        zmin, zmax = (z0, z0 + th) if th >= 0 else (z0 + th, z0)
        if _is_wafer_level(name, lev, wafer_levels):
            iv = [[0.0, 1.0]]
        else:
            iv = _intervals_on_line(_level_polygons(component, lev), p0, p1)
        if not iv:
            continue
        info = getattr(lev, "info", None) or {}
        mesh_order = info.get("mesh_order", getattr(lev, "mesh_order", 3))
        out.append(
            {
                "name": name,
                "material": str(lev.material) if lev.material is not None else "air",
                "mesh_order": int(mesh_order if mesh_order is not None else 3),
                "zmin": zmin * um,
                "zmax": zmax * um,
                "intervals": [(a * L + u_offset, b * L + u_offset) for a, b in iv],
                "sidewall_angle": float(getattr(lev, "sidewall_angle", 0.0) or 0.0),
                "width_to_z": float(getattr(lev, "width_to_z", 0.0) or 0.0),
                "full": iv == [[0.0, 1.0]],
            }
        )
    return out


def cross_section_geometry(
    component,
    layer_stack=None,
    xsection_bounds=None,
    exclude=(),
    wafer_levels=None,
    sizes: dict | None = None,
    background: str | None = None,
    u_offset: float = 0.0,
) -> Geometry:
    """The cross-section of a gdsfactory component along a line as a :class:`Geometry`.

    Every layer-stack level becomes regions named after the level (material
    from the stack, priority from its ``mesh_order``), rectangles in the
    (u, z) plane, trapezoids where the level has a ``sidewall_angle`` (the
    drawn width applies at the fraction ``width_to_z`` of the thickness, and
    the sidewalls lean inwards for a positive angle, as in gdsfactory).
    ``xsection_bounds`` is in micrometres; the geometry is in metres with u
    measured from the start of the line plus ``u_offset``. ``background``
    (a material name) adds a region of lowest priority covering the bounding
    box, for stacks that leave gaps (air above the cladding).
    """
    cuts = cut_component(component, layer_stack, xsection_bounds, exclude, wafer_levels, u_offset)
    if not cuts:
        raise ValueError("the cut line crosses no layer-stack level")
    g = Geometry()
    sizes = sizes or {}
    for c in cuts:
        h = c["zmax"] - c["zmin"]
        ang = math.radians(c["sidewall_angle"]) if not c["full"] else 0.0
        for a, b in c["intervals"]:
            size = sizes.get(c["name"], sizes.get(c["material"]))
            if ang == 0.0:
                shape = Rectangle(a, c["zmin"], b, c["zmax"])
            else:
                zr = c["zmin"] + c["width_to_z"] * h
                off_bot = (c["zmin"] - zr) * math.tan(ang)  # negative below the reference: wider
                off_top = (c["zmax"] - zr) * math.tan(ang)
                pts = [(a + off_bot, c["zmin"]), (b - off_bot, c["zmin"]), (b - off_top, c["zmax"]), (a + off_top, c["zmax"])]
                shape = Polygon(pts)
            g.add(shape, c["name"], c["material"], size=size, mesh_order=c["mesh_order"])
    if background is not None:
        x0, y0, x1, y1 = g.bounds
        g.add(Rectangle(x0, y0, x1, y1), "background", background, mesh_order=10_000)
    return g
