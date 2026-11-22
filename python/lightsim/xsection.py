"""Waveguide cross-sections for the mode solver, without a 3-D simulation.

A :class:`CrossSection` is a list of shapes (rectangles, trapezoids,
polygons and full-width layers) in a plane with axes (u, v), each filled
with a material, on a rectilinear grid. :meth:`CrossSection.solve` samples
the permittivity at the staggered Yee positions with sub-pixel smoothing and
calls the finite-difference solver of :mod:`lightsim.modes`; the modes come
back as :class:`lightsim.modes.Mode` objects, propagating along the normal
``n = u x v``.

Units are SI (metres). Materials are given as

* a number (refractive index, complex for loss: ``3.48 + 0.001j``);
* a name: a dispersive formula of :mod:`lightsim.materials` (``"si"``,
  ``"sio2"``, ``"si3n4"``, ``"LiNbO3_e"``, ...), else the constant table of
  :mod:`lightsim.gds`; ``material_index={"name": n}`` overrides both;
* three indices ``(n_u, n_v, n_n)`` (diagonal anisotropy) or a 3x3
  permittivity tensor in (u, v, n) axes (see :func:`uniaxial`);
* a function of the wavelength (m) returning any of the above.

Builders: :meth:`CrossSection.strip` (rectangular, rib and coupled
waveguides with the parameters of gplugins), :func:`from_gdsfactory` (a
gdsfactory cross-section and layer stack) and :func:`from_component` (a cut
through a gdsfactory component).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .modes import C0, Mode, group_index_of, solve_plane

um = 1e-6

__all__ = ["CrossSection", "uniaxial", "from_gdsfactory", "from_component", "graded_nodes"]


# ----------------------------------------------------------------------------
# materials
# ----------------------------------------------------------------------------


def uniaxial(n_o, n_e, optic_axis=(0.0, 0.0, 1.0)):
    """Permittivity tensor of a uniaxial crystal as a function of wavelength.

    ``n_o``, ``n_e``: ordinary and extraordinary indices (numbers, library
    names such as ``"LiNbO3_o"`` / ``"LiNbO3_e"``, or functions of the
    wavelength in m). ``optic_axis``: direction of the optic axis in the
    cross-section's (u, v, n) axes. Returns ``f(wavelength) -> 3x3``.

    Example: x-cut lithium niobate with the waveguide at an angle ``a`` to
    the crystal y axis in the chip plane (u horizontal, v vertical, n along
    the guide): ``uniaxial("LiNbO3_o", "LiNbO3_e", (cos(a), 0, sin(a)))``.
    """
    c = np.asarray(optic_axis, dtype=float)
    c = c / np.linalg.norm(c)

    def f(wavelength):
        no = _index_of(n_o, wavelength)
        ne = _index_of(n_e, wavelength)
        return no**2 * np.eye(3) + (ne**2 - no**2) * np.outer(c, c)

    f.__name__ = f"uniaxial({n_o},{n_e})"
    return f


def _index_of(spec, wavelength, material_index=None) -> complex:
    if callable(spec):
        return complex(spec(wavelength))
    if isinstance(spec, str):
        return complex(np.sqrt(_eps_of_name(spec, wavelength, material_index)))
    return complex(spec)


def _eps_of_name(name: str, wavelength: float, material_index=None) -> complex:
    key = name.lower()
    if material_index:
        mi = {str(k).lower(): v for k, v in material_index.items()}
        if key in mi:
            v = mi[key]
            n = v(wavelength) if callable(v) else v
            return complex(n) ** 2
    from . import materials

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return complex(materials.permittivity(name, wavelength))
    except KeyError:
        pass
    from .gds import MATERIAL_INDEX

    if key in MATERIAL_INDEX:
        return complex(MATERIAL_INDEX[key]) ** 2
    raise ValueError(f"material {name!r} is unknown: give a refractive index, or material_index={{{name!r}: n}}")


def material_tensor(spec, wavelength: float, material_index=None) -> np.ndarray:
    """3x3 complex permittivity tensor (u, v, n axes) of a material spec at
    ``wavelength`` (m)."""
    if callable(spec) and not isinstance(spec, str):
        v = spec(wavelength)
        if np.ndim(v) == 0:
            return complex(v) ** 2 * np.eye(3, dtype=complex)
        return material_tensor(v, wavelength, material_index)
    if isinstance(spec, str):
        return _eps_of_name(spec, wavelength, material_index) * np.eye(3, dtype=complex)
    a = np.asarray(spec, dtype=complex)
    if a.ndim == 0:
        return complex(a) ** 2 * np.eye(3, dtype=complex)
    if a.shape == (3,):
        return np.diag(a**2)
    if a.shape == (3, 3):
        return a
    raise ValueError(f"cannot read material {spec!r}")


def _material_name(spec) -> str:
    if isinstance(spec, str):
        return spec.lower()
    if callable(spec):
        return getattr(spec, "__name__", "material")
    a = np.asarray(spec)
    if a.ndim == 0:
        return f"n={complex(a).real:g}" + (f"{complex(a).imag:+g}j" if complex(a).imag else "")
    return "anisotropic"


# ----------------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------------


def graded_nodes(lo: float, hi: float, fine_lo: float, fine_hi: float, step: float, max_step: float | None = None, grading: float = 1.2) -> np.ndarray:
    """Nodes from ``lo`` to ``hi``: uniform ``step`` in ``[fine_lo, fine_hi]``,
    growing by ``grading`` per cell outside it up to ``max_step``."""
    max_step = max_step or step
    fine_lo, fine_hi = max(lo, fine_lo), min(hi, fine_hi)
    n = max(1, int(math.ceil((fine_hi - fine_lo) / step - 1e-9)))
    mid = list(np.linspace(fine_lo, fine_hi, n + 1))

    def grow(start, end, sign):
        pts = []
        x, h = start, step
        while (end - x) * sign > 1e-12:
            h = min(h * grading, max_step)
            nxt = x + sign * h
            if (end - nxt) * sign < 0.5 * h:
                nxt = end
            pts.append(nxt)
            x = nxt
        return pts

    left = grow(fine_lo, lo, -1)[::-1]
    right = grow(fine_hi, hi, +1)
    return np.array(left + mid + right)


def conforming_nodes(lo: float, hi: float, edges, fine_lo: float, fine_hi: float, step: float, max_step: float | None = None, grading: float = 1.2) -> np.ndarray:
    """Nodes from ``lo`` to ``hi`` that contain every coordinate in ``edges``
    (the shape boundaries), with cells of at most ``step`` in ``[fine_lo,
    fine_hi]`` and cells growing by ``grading`` up to ``max_step`` outside.

    With every interface on a node, the discretisation error of the
    effective index is a smooth function of the cell size (it falls as the
    square of the step), which makes width sweeps smooth and allows the
    Richardson extrapolation of :meth:`CrossSection.solve`.
    """
    max_step = max_step or step
    fine_lo, fine_hi = max(lo, fine_lo), min(hi, fine_hi)
    bps = sorted(set([lo, hi, fine_lo, fine_hi] + [float(e) for e in edges if lo < e < hi]))
    tol = 1e-6 * step
    clean = [bps[0]]
    for b in bps[1:]:
        if b - clean[-1] > tol:
            clean.append(b)
    clean[-1] = hi
    out = [clean[0]]
    for a, b in zip(clean[:-1], clean[1:]):
        if b <= fine_lo + tol and a < fine_lo - tol:  # left of the fine region: grade towards a
            d = (fine_lo - b)  # distance of b from the fine region
            h0 = min(max_step, step * grading ** (d / step) if d > 0 else step)
            pts = _graded_segment(b, a, h0, max_step, grading)  # from b going left
            out += pts[::-1][1:]
        elif a >= fine_hi - tol and b > fine_hi + tol:  # right of the fine region
            d = a - fine_hi
            h0 = min(max_step, step * grading ** (d / step) if d > 0 else step)
            out += _graded_segment(a, b, h0, max_step, grading)[1:]
        else:
            n = max(1, int(math.ceil((b - a) / step - 1e-9)))
            out += list(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def _graded_segment(start, end, h0, max_step, grading):
    """Points from ``start`` to ``end`` (either direction) with steps growing
    from ``h0`` by ``grading`` up to ``max_step``; the last step is merged so
    the segment ends exactly at ``end``."""
    sign = 1.0 if end >= start else -1.0
    length = abs(end - start)
    steps = []
    h, tot = h0, 0.0
    while tot + h < length - 1e-12:
        steps.append(h)
        tot += h
        h = min(h * grading, max_step)
    rem = length - tot
    if steps and rem < 0.5 * steps[-1]:
        steps[-1] += rem
    else:
        steps.append(rem)
    pts = [start]
    for h in steps:
        pts.append(pts[-1] + sign * h)
    pts[-1] = end
    return pts


def refine_nodes(nodes) -> np.ndarray:
    """Every cell split in two."""
    nodes = np.asarray(nodes, dtype=float)
    out = np.empty(2 * len(nodes) - 1)
    out[0::2] = nodes
    out[1::2] = 0.5 * (nodes[:-1] + nodes[1:])
    return out


# ----------------------------------------------------------------------------
# shapes
# ----------------------------------------------------------------------------


@dataclass
class _Shape:
    kind: str  # "rect", "poly", "layer"
    data: tuple
    material: object
    name: str
    priority: float

    def bbox(self):
        if self.kind == "rect":
            u0, v0, u1, v1 = self.data
            return u0, v0, u1, v1
        if self.kind == "layer":
            v0, v1 = self.data
            return -np.inf, v0, np.inf, v1
        p = np.asarray(self.data)
        return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()

    def contains(self, U, V):
        """Boolean mask of the points (U, V) (broadcastable arrays) inside."""
        if self.kind == "rect":
            u0, v0, u1, v1 = self.data
            return (U >= u0) & (U < u1) & (V >= v0) & (V < v1)
        if self.kind == "layer":
            v0, v1 = self.data
            return (V >= v0) & (V < v1) & np.ones_like(U, dtype=bool)
        return _in_polygon(U, V, np.asarray(self.data, dtype=float))


def _in_polygon(U, V, poly):
    """Even-odd point-in-polygon test, vectorised over the points."""
    U, V = np.broadcast_arrays(U, V)
    inside = np.zeros(U.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        u1, v1 = poly[i]
        u2, v2 = poly[(i + 1) % n]
        if v1 == v2:
            continue
        cond = (V >= min(v1, v2)) & (V < max(v1, v2))
        x = u1 + (V - v1) * (u2 - u1) / (v2 - v1)
        inside ^= cond & (U < x)
    return inside


class CrossSection:
    """A waveguide cross-section: shapes and materials on a (u, v) grid.

    Parameters
    ----------
    background : material
        Material of everything not covered by a shape.
    u_nodes, v_nodes : array, optional
        Grid nodes (m). If omitted, :meth:`mesh` builds a graded grid around
        the shapes when the cross-section is solved.
    material_index : dict, optional
        Refractive index (or function of wavelength) per material name; it
        overrides the library for those names.
    subpixel : int
        Sub-samples per cell and direction for the sub-pixel smoothing
        (even; 0 or 1: no smoothing, the material at the sample point).

    Shapes added later are painted over earlier ones unless a ``priority``
    says otherwise (higher wins).
    """

    def __init__(self, background="sio2", u_nodes=None, v_nodes=None, material_index=None, subpixel: int = 8):
        self.background = background
        self.shapes: list[_Shape] = []
        self.material_index = dict(material_index or {})
        self.u_nodes = None if u_nodes is None else np.asarray(u_nodes, dtype=float)
        self.v_nodes = None if v_nodes is None else np.asarray(v_nodes, dtype=float)
        self.subpixel = int(subpixel)
        self._geom = None
        self.window = None  # ((u0, v0), (u1, v1)) used by mesh()
        self.mesh_options = {}

    # -- building ----------------------------------------------------------
    def _add(self, kind, data, material, name, priority):
        pr = float(len(self.shapes)) if priority is None else float(priority)
        self.shapes.append(_Shape(kind, data, material, name or _material_name(material), pr))
        self._geom = None
        return self

    def add_rect(self, u0, v0, u1, v1, material, name=None, priority=None) -> "CrossSection":
        """Rectangle ``[u0, u1] x [v0, v1]`` (m)."""
        return self._add("rect", (min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)), material, name, priority)

    def add_layer(self, v0, v1, material, name=None, priority=None) -> "CrossSection":
        """Layer ``v0 <= v < v1`` across the whole width."""
        return self._add("layer", (min(v0, v1), max(v0, v1)), material, name, priority)

    def add_polygon(self, vertices, material, name=None, priority=None) -> "CrossSection":
        """Polygon with vertices ``[(u, v), ...]`` (m)."""
        return self._add("poly", tuple(map(tuple, np.asarray(vertices, dtype=float))), material, name, priority)

    def add_trapezoid(self, u_center, width, v0, v1, material, sidewall_angle: float = 0.0, name=None, priority=None) -> "CrossSection":
        """Waveguide core with sloped sidewalls: ``width`` at ``v0``, narrowing
        towards ``v1`` by ``sidewall_angle`` (degrees from the vertical, as in
        gdsfactory and Tidy3D)."""
        if not sidewall_angle:
            return self.add_rect(u_center - width / 2, v0, u_center + width / 2, v1, material, name, priority)
        d = abs(v1 - v0) * math.tan(math.radians(sidewall_angle))
        w0, w1 = width / 2, width / 2 - d
        return self.add_polygon([(u_center - w0, v0), (u_center + w0, v0), (u_center + w1, v1), (u_center - w1, v1)], material, name, priority)

    def set_mesh(self, u_nodes, v_nodes) -> "CrossSection":
        """Use this grid."""
        self.u_nodes = np.asarray(u_nodes, dtype=float)
        self.v_nodes = np.asarray(v_nodes, dtype=float)
        self._geom = None
        return self

    def mesh(self, window=None, step: float = 20e-9, max_step: float | None = None, margin: float = 0.3e-6, grading: float = 1.2, conforming: bool = True) -> "CrossSection":
        """Build a graded grid over ``window = ((u0, v0), (u1, v1))``: cells of
        at most ``step`` over the patterned shapes plus ``margin``, growing by
        ``grading`` up to ``max_step`` (default ``4 * step``) outside. With
        ``conforming`` (default) every rectangle and layer edge and every
        polygon vertex is a grid node (see :func:`conforming_nodes`)."""
        window = window or self.window
        if window is None:
            raise ValueError("give a window ((u0, v0), (u1, v1)) or set_mesh(...)")
        (u0, v0), (u1, v1) = window
        fu0, fu1 = np.inf, -np.inf
        ev, eu, evv = [], [], []
        for s in self.shapes:
            a, b, c, d = s.bbox()
            if s.kind != "layer":
                fu0, fu1 = min(fu0, a), max(fu1, c)
                ev += [b, d]
            else:
                ev += [e for e in (b, d) if v0 < e < v1]
            if s.kind == "poly":
                p = np.asarray(s.data)
                eu += list(p[:, 0])
                evv += list(p[:, 1])
            else:
                if np.isfinite(a):
                    eu += [a, c]
                evv += [b, d]
        fv0, fv1 = (min(ev), max(ev)) if ev else (np.inf, -np.inf)
        if not np.isfinite(fu0):
            fu0, fu1 = u0, u1
        if not np.isfinite(fv0):
            fv0, fv1 = v0, v1
        ms = max_step or 4 * step
        if conforming:
            self.u_nodes = conforming_nodes(u0, u1, eu, fu0 - margin, fu1 + margin, step, ms, grading)
            self.v_nodes = conforming_nodes(v0, v1, evv, fv0 - margin, fv1 + margin, step, ms, grading)
        else:
            self.u_nodes = graded_nodes(u0, u1, fu0 - margin, fu1 + margin, step, ms, grading)
            self.v_nodes = graded_nodes(v0, v1, fv0 - margin, fv1 + margin, step, ms, grading)
        self.window = window
        self.mesh_options = dict(step=step, max_step=max_step, margin=margin, grading=grading, conforming=conforming)
        self._geom = None
        return self

    # -- standard geometries -------------------------------------------------
    @classmethod
    def strip(
        cls,
        core_width: float = 0.5e-6,
        core_thickness: float = 0.22e-6,
        core_material="si",
        clad_material="sio2",
        box_material=None,
        slab_thickness: float = 0.0,
        slab_material=None,
        sidewall_angle: float = 0.0,
        gap: float | None = None,
        core_width2: float | None = None,
        side_margin: float = 1.5e-6,
        clad_thickness: float = 1.0e-6,
        box_thickness: float = 1.0e-6,
        step: float = 20e-9,
        max_step: float | None = None,
        material_index=None,
    ) -> "CrossSection":
        """Strip, rib or coupled waveguide, as gplugins' ``Waveguide`` and
        ``WaveguideCoupler`` (Tidy3D) and ``get_mode_solver_rib`` /
        ``get_mode_solver_coupler`` (MPB) describe them, in metres.

        The core sits on the box at ``v = 0`` and is centred at ``u = 0``
        (with ``gap``, the two cores are at ``u = -(gap + w1)/2`` and
        ``+(gap + w2)/2``). ``slab_thickness`` adds a slab of
        ``slab_material`` (default: the core material) under the core; the
        top cladding is ``clad_material`` and the bottom cladding
        ``box_material`` (default: the cladding).
        """
        xs = cls(background=clad_material, material_index=material_index)
        box = box_material if box_material is not None else clad_material
        if box_material is not None:
            xs.add_layer(-box_thickness - 1.0, 0.0, box, name=_material_name(box) + "_box")
        if slab_thickness > 0:
            xs.add_layer(0.0, slab_thickness, slab_material if slab_material is not None else core_material, name="slab")
        cores = []
        if gap is None:
            cores.append((0.0, core_width))
        else:
            w2 = core_width2 if core_width2 is not None else core_width
            cores += [(-(gap + core_width) / 2, core_width), ((gap + w2) / 2, w2)]
        for i, (uc, w) in enumerate(cores):
            xs.add_trapezoid(uc, w, 0.0, core_thickness, core_material, sidewall_angle, name=_material_name(core_material))
        half = max(abs(c) + w / 2 for c, w in cores) + side_margin
        xs.window = ((-half, -box_thickness), (half, core_thickness + clad_thickness))
        xs.mesh(xs.window, step=step, max_step=max_step)
        return xs

    # -- permittivity ----------------------------------------------------------
    def materials(self) -> dict:
        """Material name -> spec, background first."""
        out = {_material_name(self.background): self.background}
        for s in self.shapes:
            out.setdefault(s.name, s.material)
        return out

    def _geometry(self):
        """Fill fractions and first moments of every material in the cells of
        the E_u, E_v, E_n positions.

        The plane is cut into sub-cells by the grid nodes, the cell centres
        and the edges of the rectangles and layers, so rectilinear shapes
        get exact fill fractions; polygons are sampled with ``subpixel``
        sub-cells per cell and direction on top.
        """
        if self._geom is not None:
            return self._geom
        if self.u_nodes is None or self.v_nodes is None:
            self.mesh()
        import scipy.sparse as sp

        un, vn = self.u_nodes, self.v_nodes
        Nu, Nv = len(un) - 1, len(vn) - 1
        polys = any(s.kind == "poly" for s in self.shapes)
        S = max(1, self.subpixel) if polys else 1

        def breaks(nodes, edges):
            cen = 0.5 * (nodes[:-1] + nodes[1:])
            pts = [nodes, cen]
            if S > 1:
                d = np.diff(nodes)
                pts.append((nodes[:-1, None] + d[:, None] * (np.arange(1, S) / S)[None, :]).ravel())
            e = np.asarray([x for x in edges if nodes[0] < x < nodes[-1]], dtype=float)
            pts.append(e)
            b = np.unique(np.concatenate(pts))
            b = b[np.concatenate([[True], np.diff(b) > 1e-6 * np.min(np.diff(nodes))])]
            if b[-1] < nodes[-1]:
                b = np.append(b, nodes[-1])
            mid, w = 0.5 * (b[:-1] + b[1:]), np.diff(b)
            prim = np.clip(np.searchsorted(nodes, mid) - 1, 0, len(nodes) - 2)
            dual = np.searchsorted(cen, mid)  # node i owns [cen[i-1], cen[i]]
            n = len(nodes) - 1

            def assign(idx):
                ok = idx < n
                return sp.csr_matrix((w[ok], (idx[ok], np.nonzero(ok)[0])), shape=(n, len(mid)))

            return mid, assign(prim), assign(dual)

        ue, ve = [], []
        for s in self.shapes:
            a, b, c, d = s.bbox()
            if s.kind in ("rect", "layer"):
                ue += [a, c] if np.isfinite(a) else []
                ve += [b, d]
            else:
                p = np.asarray(s.data)
                ue += list(p[:, 0])
                ve += list(p[:, 1])
        Uf, Bu_c, Bu_n = breaks(un, ue)
        Vf, Bv_c, Bv_n = breaks(vn, ve)
        names = list(self.materials())
        ids = np.zeros((len(Uf), len(Vf)), dtype=np.int16)
        for s in sorted(self.shapes, key=lambda s: s.priority):
            a, b, c, d = s.bbox()
            iu = np.nonzero((Uf >= a) & (Uf < c))[0] if np.isfinite(a) else np.arange(len(Uf))
            iv = np.nonzero((Vf >= b) & (Vf < d))[0]
            if len(iu) == 0 or len(iv) == 0:
                continue
            sl = (slice(iu[0], iu[-1] + 1), slice(iv[0], iv[-1] + 1))
            m = s.contains(Uf[sl[0], None], Vf[None, sl[1]])
            sub = ids[sl]
            sub[m] = names.index(s.name)
        geo = {"names": names, "Nu": Nu, "Nv": Nv}
        for pos, (Bu, Bv) in {"u": (Bu_c, Bv_n), "v": (Bu_n, Bv_c), "n": (Bu_n, Bv_n)}.items():
            su, sv = Bu @ np.ones_like(Uf), Bv @ np.ones_like(Vf)
            area = su[:, None] * sv[None, :]
            area = np.where(area == 0, 1.0, area)
            uc = (Bu @ Uf) / np.where(su == 0, 1, su)
            vc = (Bv @ Vf) / np.where(sv == 0, 1, sv)
            fills, mom = {}, {}
            BvT = Bv.T.tocsr()
            for mi, name in enumerate(names):
                chi = (ids == mi).astype(float)
                if not chi.any():
                    continue
                f = np.asarray((Bu @ (BvT.T @ chi.T).T)) / area
                fills[name] = f
                mu_ = np.asarray(Bu @ (BvT.T @ (chi * Uf[:, None]).T).T) / area - f * uc[:, None]
                mv_ = np.asarray(Bu @ (BvT.T @ (chi * Vf[None, :]).T).T) / area - f * vc[None, :]
                mom[name] = (mu_, mv_)
            geo[pos] = {"fill": fills, "moment": mom}
        self._geom = geo
        return geo

    def permittivity(self, wavelength: float):
        """Smoothed permittivity at ``wavelength`` (m).

        Returns ``(eps_u, eps_v, eps_n, eps_off, fill, eps_material)``: the
        diagonal maps at the E_u, E_v, E_n positions, the off-diagonal terms
        (dict, empty for diagonal media), the fill fraction of every
        material at the three positions and the diagonal permittivity of
        every material. In a cell with an interface the diagonal term along
        the interface normal ``n`` is ``n_i² <1/eps>^-1 + (1 - n_i²) <eps>``
        (the normal comes from the first moment of the permittivity).
        """
        geo = self._geometry()
        names = geo["names"]
        mats = self.materials()
        tens = {name: material_tensor(mats[name], wavelength, self.material_index) for name in names}
        out = {}
        off = {}
        fill_out = {}
        for ci, pos in enumerate(("u", "v", "n")):
            fills = geo[pos]["fill"]
            present = list(fills)
            A = sum(fills[m] * tens[m][ci, ci] for m in present)
            if len(present) == 1:
                out[pos] = np.broadcast_to(A, (geo["Nu"], geo["Nv"])).astype(complex)
            else:
                Hm = 1.0 / sum(fills[m] / tens[m][ci, ci] for m in present)
                # normal from the permittivity-weighted first moments
                eref = np.mean([np.real(np.trace(tens[m])) / 3 for m in present])
                mu_ = sum((np.real(np.trace(tens[m])) / 3 - eref) * geo[pos]["moment"][m][0] for m in present)
                mv_ = sum((np.real(np.trace(tens[m])) / 3 - eref) * geo[pos]["moment"][m][1] for m in present)
                nrm = np.hypot(mu_, mv_)
                ok = nrm > 0
                nu2 = np.where(ok, (mu_ / np.where(ok, nrm, 1)) ** 2, 0.0)
                nv2 = np.where(ok, (mv_ / np.where(ok, nrm, 1)) ** 2, 0.0)
                p = (nu2, nv2, np.zeros_like(nu2))[ci]
                out[pos] = p * Hm + (1 - p) * A
            # off-diagonal terms of the row component
            for cj, other in enumerate(("u", "v", "n")):
                if cj == ci:
                    continue
                val = sum(fills[m] * tens[m][ci, cj] for m in present)
                if np.any(np.abs(val) > 0):
                    off[pos + other] = np.broadcast_to(val, (geo["Nu"], geo["Nv"])).astype(complex)
        for name in names:
            fill_out[name] = tuple(geo[p]["fill"].get(name, np.zeros((geo["Nu"], geo["Nv"]))) for p in ("u", "v", "n"))
        eps_mat = {name: tuple(tens[name][i, i] for i in range(3)) for name in names}
        return out["u"], out["v"], out["n"], off, fill_out, eps_mat

    # -- solving -------------------------------------------------------------
    def solve(
        self,
        wavelength: float,
        num_modes: int = 2,
        target_neff: float | None = None,
        *,
        pml=None,
        bend_radius: float | None = None,
        bend_center: float = 0.0,
        group_index: bool = False,
        sort: str = "neff",
        cache: bool = False,
        timings: dict | None = None,
        profile: bool = True,
        periodic=(False, False),
        extrapolate: bool = False,
    ) -> list[Mode]:
        """Modes of the cross-section at ``wavelength`` (m), highest effective
        index first (``sort="te"``: TE-like first).

        ``pml``, ``bend_radius`` as for :func:`lightsim.solve_modes`; the
        bend turns about a centre at ``u = bend_center - bend_radius``
        (default: the core centred on ``u = 0``). ``group_index=True`` adds
        ``n_group`` from the energy-flux relation with the materials'
        dispersion (central difference of ``w eps`` over ±0.1 % in
        frequency).

        ``extrapolate=True`` solves again on the grid with every cell split
        in two and returns the modes of the fine grid with the Richardson
        extrapolation ``(4 n_fine - n_coarse) / 3`` of ``neff`` (and of
        ``n_group``); the two values are kept in ``info["neff_grids"]``. It
        removes the leading (second-order) discretisation error when every
        interface lies on a grid node (the default :meth:`mesh`), and costs
        about five times one solve.
        """
        if extrapolate:
            kw = dict(num_modes=num_modes, target_neff=target_neff, bend_radius=bend_radius, bend_center=bend_center, group_index=group_index, sort=sort, cache=cache, periodic=periodic)
            coarse = self.solve(wavelength, pml=pml, profile=False, **kw)
            saved = (self.u_nodes, self.v_nodes)
            un = saved[0] if (periodic[0] and len(saved[0]) == 2) else refine_nodes(saved[0])
            vn = saved[1] if (periodic[1] and len(saved[1]) == 2) else refine_nodes(saved[1])
            self.set_mesh(un, vn)
            try:
                fine = self.solve(wavelength, pml=_double_pml(pml), profile=profile, timings=timings, **kw)
            finally:
                self.set_mesh(*saved)
            for m in fine:
                c = min(coarse, key=lambda c: abs(c.neff - m.neff) + abs(c.te_fraction - m.te_fraction))
                m.info = dict(m.info)
                m.info["neff_grids"] = (c.neff, m.neff)
                m.neff = complex((4 * m.neff - c.neff) / 3)
                if m.n_group is not None and c.n_group is not None:
                    m.n_group = float((4 * m.n_group - c.n_group) / 3)
                if m.profile:
                    m.profile["neff"], m.profile["neff_imag"] = float(m.neff.real), float(m.neff.imag)
            return fine
        eu, ev, en, off, fill, epsm = self.permittivity(wavelength)
        freq = C0 / wavelength
        info = {"fill": fill, "eps_material": epsm, "wavelength": wavelength, "h_offset": 0.0}
        modes = solve_plane(
            eu, ev, en, self.u_nodes, self.v_nodes, freq, num_modes, target_neff,
            info=info, pml=pml, bend_radius=bend_radius, bend_center=bend_center,
            eps_off=off or None, sort=sort, cache=cache, timings=timings, profile=profile, periodic=periodic,
        )
        if group_index:
            self.add_group_index(modes, wavelength)
        return modes

    def group_permittivity(self, wavelength: float, rel: float = 1e-3):
        """``d(w eps)/dw`` at the E_u, E_v, E_n positions (central
        difference over ``±rel`` in frequency)."""
        f = C0 / wavelength
        lo = self.permittivity(C0 / (f * (1 - rel)))
        hi = self.permittivity(C0 / (f * (1 + rel)))
        return tuple(((1 + rel) * hi[i] - (1 - rel) * lo[i]) / (2 * rel) for i in range(3))

    def add_group_index(self, modes, wavelength: float) -> None:
        """Set ``n_group`` of modes solved on this cross-section."""
        eg = self.group_permittivity(wavelength)
        for m in modes:
            mu = None
            if m.bend_radius:
                eg_m, mu = _bend_scaled(eg, self.u_nodes, self.v_nodes, m.bend_radius, m.info["solver"].get("bend_center") or 0.0)
            else:
                eg_m = eg
            m.n_group = group_index_of(m, eg_m, mu)

    def plot(self, wavelength: float = 1.55e-6, ax=None, component: str = "n"):
        """Plot the real index map ``sqrt(eps)`` of one position set (needs
        matplotlib)."""
        import matplotlib.pyplot as plt

        eu, ev, en, *_ = self.permittivity(wavelength)
        e = {"u": eu, "v": ev, "n": en}[component]
        ax = ax or plt.gca()
        un, vn = self.u_nodes, self.v_nodes
        im = ax.pcolormesh(un * 1e6, vn * 1e6, np.sqrt(e).real.T, shading="flat")
        ax.set_aspect("equal")
        ax.set_xlabel("u (um)")
        ax.set_ylabel("v (um)")
        plt.colorbar(im, ax=ax, label="n")
        return ax


def _double_pml(pml):
    """The PML of a grid with every cell split in two (same thickness)."""
    if pml is None or pml is False:
        return pml
    if pml is True:
        return 24
    if isinstance(pml, (int, np.integer)):
        return 2 * int(pml)
    if isinstance(pml, dict):
        d = dict(pml)
        c = d.get("cells", 12)
        d["cells"] = 2 * int(c) if isinstance(c, (int, np.integer)) else tuple(2 * int(v) for v in c)
        return d
    return tuple(2 * int(v) for v in pml)


def _bend_scaled(eps3, un, vn, radius, center):
    from .modes import _positions

    p = _positions(un, vn)
    r = lambda name: (radius + (p[name][0] - center)) / radius  # noqa: E731
    eps_b = (eps3[0] * r("ex"), eps3[1] * r("ey"), eps3[2] / r("ez"))
    mu_b = (r("hx"), r("hy"), 1.0 / r("hz"))
    return eps_b, mu_b


# ----------------------------------------------------------------------------
# gdsfactory
# ----------------------------------------------------------------------------


def _interval_cuts(poly, c: float, along: int):
    """Intervals where the line ``coord[along] = c`` crosses a polygon:
    sorted list of (t0, t1) in the other coordinate."""
    p = np.asarray(poly, dtype=float)
    other = 1 - along
    hits = []
    n = len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        if (a[along] <= c < b[along]) or (b[along] <= c < a[along]):
            t = a[other] + (c - a[along]) * (b[other] - a[other]) / (b[along] - a[along])
            hits.append(t)
    hits.sort()
    return [(hits[i], hits[i + 1]) for i in range(0, len(hits) - 1, 2)]


def from_component(
    component,
    x: float | None = None,
    y: float | None = None,
    layer_stack=None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    margin: float = 1.5e-6,
    v_margin: float = 1.0e-6,
    span=None,
    step: float = 20e-9,
    max_step: float | None = None,
    background="sio2",
    sidewall: bool = False,
) -> CrossSection:
    """Cross-section of a gdsfactory component on the plane ``x = x`` (light
    along x; u = y, v = z) or ``y = y`` (light along y; u = x, v = z),
    coordinates in metres.

    The polygons of every patterned level of the layer stack are cut by the
    plane; wafer levels (buried oxide, cladding) become full-width layers.
    Levels are painted as in :func:`lightsim.gds.component_to_structures`
    (lower ``mesh_order`` wins). ``span = (u0, u1)`` sets the lateral extent
    of the window, otherwise it is the extent of the cut plus ``margin``;
    vertically the window covers the patterned levels plus ``v_margin``.
    ``sidewall=True`` draws the levels' sidewall angles (the 3-D import
    draws vertical walls).
    """
    from .gds import DEFAULT_EXCLUDE, stack_layers

    if (x is None) == (y is None):
        raise ValueError("give exactly one of x= or y=")
    import gdsfactory as gf

    ls_ = layer_stack if layer_stack is not None else gf.pdk.get_layer_stack()
    layers = stack_layers(component, ls_, exclude=exclude if exclude is not None else DEFAULT_EXCLUDE, wafer_levels=wafer_levels)
    along = 0 if x is not None else 1
    c = x if x is not None else y
    xs = CrossSection(background=background, material_index=material_index)
    wafer = [l for l in layers if not l.polygons]
    pattern = sorted([l for l in layers if l.polygons], key=lambda l: -l.mesh_order)
    for l in wafer:
        xs.add_layer(l.zmin, l.zmax, l.material, name=l.material)
    lo, hi = np.inf, -np.inf
    zlo, zhi = np.inf, -np.inf
    for l in pattern:
        ang = 0.0
        if sidewall:
            ang = float(getattr(ls_.layers[l.name], "sidewall_angle", 0.0) or 0.0)
        for poly in l.polygons:
            for t0, t1 in _interval_cuts(poly, c, along):
                if t1 - t0 <= 0:
                    continue
                xs.add_trapezoid(0.5 * (t0 + t1), t1 - t0, l.zmin, l.zmax, l.material, ang, name=l.material)
                lo, hi = min(lo, t0), max(hi, t1)
                zlo, zhi = min(zlo, l.zmin), max(zhi, l.zmax)
    if not np.isfinite(lo):
        raise ValueError(f"the plane {'x' if along == 0 else 'y'} = {c} does not cut any patterned level of the component")
    u0, u1 = span if span is not None else (lo - margin, hi + margin)
    xs.window = ((u0, zlo - v_margin), (u1, zhi + v_margin))
    xs.mesh(xs.window, step=step, max_step=max_step)
    xs.cut = {"axis": along, "position": c}
    return xs


def from_gdsfactory(
    cross_section="strip",
    layer_stack=None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    margin: float = 1.5e-6,
    v_margin: float = 1.0e-6,
    step: float = 20e-9,
    max_step: float | None = None,
    background="sio2",
    sidewall: bool = False,
    **cross_section_kwargs,
) -> CrossSection:
    """Cross-section of a gdsfactory cross-section (a name such as
    ``"strip"`` or ``"rib"``, a ``CrossSection``, or a function returning
    one) extruded with a ``LayerStack`` (default: the active PDK's).

    The cross-section is drawn as a short straight and cut in the middle
    (see :func:`from_component`), so derived layers and slab sections are
    handled as in the 3-D import. The guide runs along x: u = y, v = z.
    """
    import gdsfactory as gf

    xsec = gf.get_cross_section(cross_section, **cross_section_kwargs)
    comp = gf.components.straight(length=2.0, cross_section=xsec)
    return from_component(comp, x=1.0 * um, layer_stack=layer_stack, material_index=material_index, exclude=exclude, wafer_levels=wafer_levels, margin=margin, v_margin=v_margin, step=step, max_step=max_step, background=background, sidewall=sidewall)
