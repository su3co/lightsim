"""Eigenmode expansion (EME): S-matrices of devices that vary along their
propagation axis, from the modes of their cross-sections.

A device is a sequence of sections, each a :class:`~lightsim.xsection.
CrossSection` and a length, on one shared transverse grid. In every section
the field is expanded in the section's modes (from the finite-difference
solver of :mod:`lightsim.modes`); a mode travels through the section as
``exp(i beta L)``. At the junction of two sections the transverse fields are
matched with the unconjugated overlaps ``<E_i, H_j> = ∫ (E_i x H_j) · n dA``
of the two mode sets (E continuity tested with the modes of the far side, H
continuity with those of the near side; see :func:`interface`), which gives
the reflection and transmission matrices of the junction. The S-matrices
of junctions and sections are combined with the Redheffer star product;
``N`` identical periods are combined by repeated squaring (``log2 N``
products). Reflections are kept everywhere (bidirectional EME).

Modes are normalised to ``<E_i, H_i> = 1``. For lossless guided modes this
is unit power up to a factor two, so ``|S_ji|²`` is the power transfer.

Entry points:

* :class:`EME` -- sections given by hand;
* :func:`component_sections` -- sections cut from a gdsfactory component
  (light along x) with a layer stack, merging identical neighbours;
* :func:`sparameters` / :class:`EMEComponent` -- the S-parameters of a
  gdsfactory component, as ``gplugins.meow.MEOW`` returns them;
* :func:`stack_transfer_matrix` -- the exact 1-D multilayer reference.

Key convention of the S-parameter dictionaries: ``"o2@0,o1@0"`` is the
amplitude leaving port ``o2`` in mode 0 per unit amplitude entering port
``o1`` in mode 0, as in :func:`lightsim.sparams.write_sparameters` (gplugins'
Tidy3D writer orders keys the other way round; for the reciprocal devices
EME models the two orders give the same matrix).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .modes import C0, _areas
from .xsection import CrossSection

um = 1e-6

__all__ = ["EME", "SMatrix", "star", "power", "component_sections", "sparameters", "EMEComponent", "stack_transfer_matrix"]


# ----------------------------------------------------------------------------
# S-matrix algebra
# ----------------------------------------------------------------------------


@dataclass
class SMatrix:
    """Scattering matrix of a two-sided section in the mode bases of its
    left and right ends.

    ``s11`` (left -> left, reflection), ``s21`` (left -> right,
    transmission), ``s12`` (right -> left), ``s22`` (right -> right). The
    first index of every block is the outgoing mode, the second the
    incoming one.
    """

    s11: np.ndarray
    s12: np.ndarray
    s21: np.ndarray
    s22: np.ndarray

    @classmethod
    def identity(cls, n: int) -> "SMatrix":
        z = np.zeros((n, n), dtype=complex)
        return cls(z.copy(), np.eye(n, dtype=complex), np.eye(n, dtype=complex), z.copy())

    @classmethod
    def propagation(cls, beta, length: float) -> "SMatrix":
        p = np.diag(np.exp(1j * np.asarray(beta) * length))
        z = np.zeros_like(p)
        return cls(z, p, p.copy(), z.copy())

    def full(self) -> np.ndarray:
        """``[[s11, s12], [s21, s22]]`` as one matrix (left modes first)."""
        return np.block([[self.s11, self.s12], [self.s21, self.s22]])


def star(a: SMatrix, b: SMatrix) -> SMatrix:
    """Redheffer star product: ``a`` followed by ``b`` (``a``'s right basis is
    ``b``'s left basis)."""
    n = a.s22.shape[0]
    I = np.eye(n, dtype=complex)
    m1 = np.linalg.solve(I - b.s11 @ a.s22, np.eye(n))  # (I - B11 A22)^-1
    m2 = np.linalg.solve(I - a.s22 @ b.s11, np.eye(n))  # (I - A22 B11)^-1
    return SMatrix(
        s11=a.s11 + a.s12 @ m1 @ b.s11 @ a.s21,
        s12=a.s12 @ m1 @ b.s12,
        s21=b.s21 @ m2 @ a.s21,
        s22=b.s22 + b.s21 @ m2 @ a.s22 @ b.s12,
    )


def power(s: SMatrix, n: int) -> SMatrix:
    """``s`` cascaded ``n`` times (``n >= 1``) by repeated squaring."""
    if n < 1:
        raise ValueError("n must be at least 1")
    result = None
    base = s
    while n:
        if n & 1:
            result = base if result is None else star(result, base)
        n >>= 1
        if n:
            base = star(base, base)
    return result


def interface(left, right) -> SMatrix:
    """S-matrix of the junction between two mode sets on the same grid.

    With ``O = <E_i^L, H_j^R>`` (unconjugated overlaps, modes normalised to
    ``<E_i, H_i> = 1``), continuity of the transverse E field tested with
    the modes of the right side and of the transverse H field tested with
    the modes of the left side give, for light from the left,

        T_LR = 2 O^T (I + O O^T)^-1,    R_LR = (I + O O^T)^-1 (I - O O^T),

    and the same with ``O' = <E_i^R, H_j^L>`` for light from the right. For
    lossless guided modes with real fields the junction conserves power
    exactly whatever the number of modes, the two mode sets may have
    different sizes, and a single mode reproduces the Fresnel coefficients.
    """
    O_LR = left.overlap(right)  # <E_i^L, H_j^R>, (NL, NR)
    O_RL = right.overlap(left)  # <E_i^R, H_j^L>, (NR, NL)

    def side(O):
        n = O.shape[0]
        W = np.linalg.inv(np.eye(n) + O @ O.T)
        return W @ (np.eye(n) - O @ O.T), 2 * O.T @ W  # reflection, transmission

    r_lr, t_lr = side(O_LR)
    r_rl, t_rl = side(O_RL)
    return SMatrix(s11=r_lr, s12=t_rl, s21=t_lr, s22=r_rl)


# ----------------------------------------------------------------------------
# mode sets
# ----------------------------------------------------------------------------


@dataclass
class _ModeSet:
    """Transverse fields of a set of modes, normalised to <E_i, H_i> = 1."""

    beta: np.ndarray
    Eu: np.ndarray  # (M, Nu, Nv)
    Ev: np.ndarray
    Hu: np.ndarray
    Hv: np.ndarray
    a_u: np.ndarray
    a_v: np.ndarray
    modes: list = field(default_factory=list)

    @classmethod
    def from_modes(cls, modes, k0: float) -> "_ModeSet":
        a = _areas(modes[0])
        Eu = np.stack([m.E[0] for m in modes])
        Ev = np.stack([m.E[1] for m in modes])
        Hu = np.stack([m.H[0] for m in modes])
        Hv = np.stack([m.H[1] for m in modes])
        s = cls(np.array([k0 * m.neff for m in modes]), Eu, Ev, Hu, Hv, a["eu"], a["ev"], list(modes))
        norm = np.sum(Eu * Hv * a["eu"], axis=(1, 2)) - np.sum(Ev * Hu * a["ev"], axis=(1, 2))
        c = 1.0 / np.sqrt(norm.astype(complex))
        # keep the solver's phase convention: the square root's sign is chosen
        # so that the largest field component stays real positive for lossless modes
        c = np.where(np.real(c) < 0, -c, c)
        s.Eu, s.Ev = s.Eu * c[:, None, None], s.Ev * c[:, None, None]
        s.Hu, s.Hv = s.Hu * c[:, None, None], s.Hv * c[:, None, None]
        return s

    def overlap(self, other: "_ModeSet") -> np.ndarray:
        """``O[i, j] = <E_i^self, H_j^other>`` (unconjugated)."""
        M = self.Eu.shape[0]
        Mo = other.Eu.shape[0]
        e_u = self.Eu.reshape(M, -1)
        e_v = self.Ev.reshape(M, -1)
        h_v = (other.Hv * self.a_u).reshape(Mo, -1)
        h_u = (other.Hu * self.a_v).reshape(Mo, -1)
        return e_u @ h_v.T - e_v @ h_u.T

    def subset(self, idx) -> "_ModeSet":
        idx = list(idx)
        return _ModeSet(self.beta[idx], self.Eu[idx], self.Ev[idx], self.Hu[idx], self.Hv[idx], self.a_u, self.a_v, [self.modes[i] for i in idx])


def _xs_key(xs: CrossSection, wavelength: float) -> str:
    eu, ev, en, off, *_ = xs.permittivity(wavelength)
    h = hashlib.sha1()
    for a in (xs.u_nodes, xs.v_nodes, eu, ev, en):
        h.update(np.ascontiguousarray(a).tobytes())
    for k in sorted(off or {}):
        h.update(k.encode())
        h.update(np.ascontiguousarray(off[k]).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# EME
# ----------------------------------------------------------------------------


class EME:
    """Eigenmode expansion of a sequence of sections.

    Parameters
    ----------
    sections : list of (CrossSection, length)
        The device from left to right; lengths in metres (zero-length
        sections are allowed, e.g. at the ports). All cross-sections must
        share the same grid. A section may also be ``("period", sections,
        n)``: ``n`` repetitions of a sub-list, combined by repeated squaring.
    num_modes : int
        Modes per section (the same in every section).
    pml : optional
        PML of the mode solves (see :func:`lightsim.solve_modes`).

    After :meth:`solve`, ``smatrix`` is the :class:`SMatrix` in the mode
    bases of the first and last sections and ``timings`` holds the seconds
    spent solving modes, computing overlaps and cascading.
    """

    def __init__(self, sections, num_modes: int = 10, pml=None, target_neff=None):
        self.sections = list(sections)
        self.num_modes = int(num_modes)
        self.pml = pml
        self.target_neff = target_neff
        self.smatrix = None
        self.timings = {}
        self._modes_cache = {}
        self.unique_cross_sections = 0

    # -- modes --------------------------------------------------------------
    def _modeset(self, xs: CrossSection, wavelength: float) -> _ModeSet:
        key = _xs_key(xs, wavelength)
        if key not in self._modes_cache:
            ms = xs.solve(wavelength, self.num_modes, self.target_neff, pml=self.pml, profile=False)
            if len(ms) < self.num_modes:
                raise RuntimeError(f"the solver returned {len(ms)} of {self.num_modes} modes")
            self._modes_cache[key] = _ModeSet.from_modes(ms, 2 * np.pi / wavelength)
        return self._modes_cache[key]

    def _flat(self, sections):
        """Yield ("xs", xs, length) and ("period", list, n) items."""
        for s in sections:
            if isinstance(s, tuple) and len(s) == 3 and s[0] == "period":
                yield s
            else:
                xs, length = s
                yield ("xs", xs, float(length))

    def _cascade(self, sections, wavelength, t):
        """S-matrix of a list of sections; returns (S, first_set, last_set)."""
        S, first, last = None, None, None
        for item in self._flat(sections):
            if item[0] == "period":
                sub, first_sub, last_sub = self._cascade(item[1], wavelength, t)
                # close the period: junction from its last section back to its first
                t0 = time.perf_counter()
                closing = interface(last_sub, first_sub) if last_sub is not first_sub else SMatrix.identity(len(first_sub.beta))
                t["overlaps"] += time.perf_counter() - t0
                unit = star(sub, closing)
                t0 = time.perf_counter()
                block = star(power(unit, item[2] - 1), sub) if item[2] > 1 else sub
                t["cascade"] += time.perf_counter() - t0
                cur_first, cur_last = first_sub, last_sub
            else:
                _, xs, length = item
                t0 = time.perf_counter()
                ms = self._modeset(xs, wavelength)
                t["modes"] += time.perf_counter() - t0
                block = SMatrix.propagation(ms.beta, length)
                cur_first = cur_last = ms
            if S is None:
                S, first = block, cur_first
            else:
                t0 = time.perf_counter()
                J = interface(last, cur_first) if last is not cur_first else None
                t["overlaps"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                S = star(S, J) if J is not None else S
                S = star(S, block)
                t["cascade"] += time.perf_counter() - t0
            last = cur_last
        return S, first, last

    def solve(self, wavelength: float) -> SMatrix:
        """S-matrix of the device at ``wavelength`` (m), in the mode bases of
        the first and last sections (reference planes at the two ends)."""
        t = {"modes": 0.0, "overlaps": 0.0, "cascade": 0.0}
        t0 = time.perf_counter()
        n_before = len(self._modes_cache)
        S, first, last = self._cascade(self.sections, wavelength, t)
        t["total"] = time.perf_counter() - t0
        self.timings = t
        self.smatrix, self.left, self.right = S, first, last
        self.unique_cross_sections = len(self._modes_cache) - n_before
        return S

    def port_smatrix(self, left_ports, right_ports) -> tuple[np.ndarray, list]:
        """S-matrix between port modes after :meth:`solve`.

        ``left_ports`` / ``right_ports`` are lists of ``(name, modes)`` with
        modes (from :meth:`CrossSection.solve` on the grid of the end
        sections) that represent each port, e.g. the mode of one output
        guide of a splitter. Amplitudes are converted with the symmetric
        unconjugated projection ``C[p, m] = (<E_m, H_p> + <E_p, H_m>) / 2``
        between the port modes (normalised to ``<E_p, H_p> = 1``) and the
        modes of the end section. Returns ``(S, names)``, names
        ``"port@mode"`` in the order of the matrix.
        """
        k0 = self.left.beta[0] / self.left.modes[0].neff
        names, C_L, C_R = [], [], []

        def coeffs(ports, end):
            rows = []
            for name, modes in ports:
                ps = _ModeSet.from_modes(modes, k0)
                c = 0.5 * (end.overlap(ps).T + ps.overlap(end))  # (P, M)
                for i in range(len(modes)):
                    rows.append((f"{name}@{i}", c[i]))
            return rows

        L = coeffs(left_ports, self.left)
        R = coeffs(right_ports, self.right)
        CL = np.array([r[1] for r in L])
        CR = np.array([r[1] for r in R])
        S = self.smatrix
        top = np.hstack([CL @ S.s11 @ CL.T, CL @ S.s12 @ CR.T])
        bot = np.hstack([CR @ S.s21 @ CL.T, CR @ S.s22 @ CR.T])
        return np.vstack([top, bot]), [r[0] for r in L] + [r[0] for r in R]


# ----------------------------------------------------------------------------
# 1-D reference
# ----------------------------------------------------------------------------


def stack_transfer_matrix(indices, thicknesses, wavelength: float):
    """Exact normal-incidence reflection and transmission amplitudes of a
    multilayer stack (indices of the input medium, the layers and the
    output medium; layer thicknesses in m). Returns ``(r, t, r_back,
    t_back)`` for fields normalised to unit power flux in each medium
    (``|r|² + |t|² = 1`` without loss)."""
    n = np.asarray(indices, dtype=complex)
    k0 = 2 * np.pi / wavelength
    M = np.eye(2, dtype=complex)
    for i in range(len(n) - 1):
        a, b = n[i], n[i + 1]
        # interface i -> i+1 for (forward, backward) amplitudes of E
        D = 0.5 * np.array([[1 + a / b, 1 - a / b], [1 - a / b, 1 + a / b]])
        M = D @ M
        if i + 1 < len(n) - 1:
            ph = k0 * n[i + 1] * thicknesses[i]
            M = np.diag([np.exp(1j * ph), np.exp(-1j * ph)]) @ M
    # M maps (a_in, b_in) -> (a_out, b_out); incidence from the left: b_out = 0
    r = -M[1, 0] / M[1, 1]
    t = M[0, 0] + M[0, 1] * r
    # back incidence: a_in = 0 -> b_in = 1/M11 * b_out ...
    t_back = 1.0 / M[1, 1]
    r_back = M[0, 1] * t_back
    s_in, s_out = np.sqrt(n[0].real), np.sqrt(n[-1].real)
    return r, t * s_out / s_in, r_back, t_back * s_in / s_out


# ----------------------------------------------------------------------------
# gdsfactory components
# ----------------------------------------------------------------------------


def _cut_signature(layers, x: float):
    """Hashable description of the cut of the patterned levels at ``x``."""
    from .xsection import _interval_cuts

    sig = []
    for l in layers:
        if not l.polygons:
            continue
        iv = []
        for poly in l.polygons:
            iv += [(round(a, 13), round(b, 13)) for a, b in _interval_cuts(poly, x, 0) if b > a]
        sig.append((l.name, tuple(sorted(iv))))
    return tuple(sig)


def component_sections(
    component,
    layer_stack=None,
    cell_length: float = 0.5e-6,
    x_range=None,
    span=None,
    margin: float = 1.0e-6,
    v_margin: float = 0.8e-6,
    step: float = 25e-9,
    max_step: float | None = None,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    background="sio2",
    merge: bool = True,
):
    """Sections of a gdsfactory component cut along x (light along +x).

    The extent ``x_range = (x0, x1)`` (m; default the component's x extent)
    is cut at every polygon edge normal to x (abrupt changes such as grating
    teeth or the ends of an MMI body), and each interval into equal cells of
    at most ``cell_length``; each cell takes the cross-section at its centre
    (a staircase approximation of tapers and bends).
    With ``merge``, neighbouring cells with the same cut are joined into one
    section, so straight parts cost one mode solve. A zero-length section
    with the cross-section at each end gives the reference planes ``x0``
    and ``x1``. All sections share one grid: ``step`` over the component's
    y extent (``span`` sets the lateral window, default the extent plus
    ``margin``) and over the patterned levels in z (plus ``v_margin``).

    Returns ``(sections, info)``; ``info`` has the cell boundaries and the
    number of distinct cross-sections.
    """
    import gdsfactory as gf

    from .gds import DEFAULT_EXCLUDE, stack_layers
    from .xsection import conforming_nodes, from_component, graded_nodes

    ls_ = layer_stack if layer_stack is not None else gf.pdk.get_layer_stack()
    layers = stack_layers(component, ls_, exclude=exclude if exclude is not None else DEFAULT_EXCLUDE, wafer_levels=wafer_levels)
    bb = component.bbox() if callable(getattr(component, "bbox", None)) else component.bbox
    try:
        xmin, ymin, xmax, ymax = component.xmin * um, component.ymin * um, component.xmax * um, component.ymax * um
    except AttributeError:  # pragma: no cover
        (xmin, ymin), (xmax, ymax) = np.asarray(bb) * um
    x0, x1 = x_range if x_range is not None else (xmin, xmax)
    # abrupt changes along x (polygon edges normal to x: grating teeth, the
    # ends of an MMI body) become cell boundaries; between them the cells
    # are equal and at most cell_length long
    tol = 1e-4 * cell_length
    brk = {x0, x1}
    for l in layers:
        for poly in l.polygons:
            p = np.asarray(poly)
            q = np.roll(p, -1, axis=0)
            for xa in p[np.abs(p[:, 0] - q[:, 0]) < 1e-12, 0]:
                if x0 + tol < xa < x1 - tol:
                    brk.add(float(xa))
    brk = sorted(brk)
    clean = [brk[0]]
    for b in brk[1:]:
        if b - clean[-1] > tol:
            clean.append(b)
    clean[-1] = x1
    edges = [clean[0]]
    for a, b in zip(clean[:-1], clean[1:]):
        n = max(1, int(math.ceil((b - a) / cell_length - 1e-9)))
        edges += list(np.linspace(a, b, n + 1)[1:])
    edges = np.array(edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    u0, u1 = span if span is not None else (ymin - margin, ymax + margin)
    zs = [z for l in layers if l.polygons for z in (l.zmin, l.zmax)]
    z0, z1 = min(zs) - v_margin, max(zs) + v_margin
    ms = max_step or 4 * step
    u_nodes = graded_nodes(u0, u1, ymin - 0.3e-6, ymax + 0.3e-6, step, ms)
    z_edges = sorted(set(z for l in layers for z in (l.zmin, l.zmax) if z0 < z < z1))
    v_nodes = conforming_nodes(z0, z1, z_edges, min(zs) - 0.3e-6, max(zs) + 0.3e-6, step, ms)
    kw = dict(layer_stack=ls_, material_index=material_index, exclude=exclude, wafer_levels=wafer_levels, span=(u0, u1), background=background)
    eps_x = 1e-9 * (x1 - x0)

    def xs_at(x):
        xs = from_component(component, x=x, **kw)
        xs.set_mesh(u_nodes, v_nodes)
        return xs

    sections = []
    sigs = []
    cur_sig, cur_len, cur_x = None, 0.0, None
    for a, b, m in zip(edges[:-1], edges[1:], mids):
        sig = _cut_signature(layers, m)
        if merge and sig == cur_sig:
            cur_len += b - a
            continue
        if cur_sig is not None:
            sections.append((xs_at(cur_x), cur_len))
            sigs.append(cur_sig)
        cur_sig, cur_len, cur_x = sig, b - a, m
    sections.append((xs_at(cur_x), cur_len))
    sigs.append(cur_sig)
    ends = [(xs_at(x0 + eps_x), 0.0), (xs_at(x1 - eps_x), 0.0)]
    sections = [ends[0]] + sections + [ends[1]]
    return sections, {"edges": edges, "distinct": len(set(sigs)), "u_nodes": u_nodes, "v_nodes": v_nodes, "x_range": (x0, x1)}


def _port_cross_sections(component, side_ports, x: float, sections_info, layer_stack, material_index, background, half_gap=None):
    """Cross-section of each port on one end: the cut at ``x`` keeping only
    the shapes within the port's lateral window."""
    from .xsection import CrossSection as XS
    from .xsection import from_component

    full = from_component(component, x=x, layer_stack=layer_stack, material_index=material_index, span=(sections_info["u_nodes"][0], sections_info["u_nodes"][-1]), background=background)
    ys = sorted(p["center"][1] for p in side_ports)
    out = []
    for p in side_ports:
        yc, w = p["center"][1], p["width"]
        others = [y for y in ys if y != yc]
        lim = min([abs(y - yc) for y in others] + [np.inf]) / 2
        lo, hi = yc - lim, yc + lim
        xs = XS(background=full.background, material_index=full.material_index)
        for s in full.shapes:
            if s.kind == "layer":
                xs.shapes.append(s)
                continue
            a, b, c, d = s.bbox()
            if c > lo and a < hi:
                xs.shapes.append(s)
        xs.set_mesh(sections_info["u_nodes"], sections_info["v_nodes"])
        out.append((p["name"], xs))
    return out


def sparameters(
    component,
    layer_stack=None,
    wavelengths=1.55e-6,
    num_modes: int = 10,
    port_modes: int = 1,
    cell_length: float = 0.5e-6,
    step: float = 25e-9,
    margin: float = 1.0e-6,
    v_margin: float = 0.8e-6,
    material_index=None,
    exclude=None,
    wafer_levels=None,
    background="sio2",
    pml=None,
    return_eme: bool = False,
):
    """S-parameters of a gdsfactory component by EME, light along x.

    Ports facing west (180 degrees) are on the left end, ports facing east
    (0 degrees) on the right end; other ports raise an error. Each port is
    represented by the ``port_modes`` highest-index modes of the end cut
    restricted to that port's guide (shapes within half the distance to the
    neighbouring port); the device is expanded in ``num_modes`` modes per
    section. Returns a dict as :func:`lightsim.sparams.write_sparameters`:
    ``"o2@0,o1@0"`` (out, in) complex arrays over ``wavelengths`` (m), plus
    ``"wavelengths"`` and ``"timings"`` (seconds per stage, summed over
    wavelengths). Lengths and coordinates in metres.
    """
    from .gds import port_info

    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    ports = port_info(component)
    west = sorted([p for p in ports if abs(p["orientation"] - 180) < 1e-6], key=lambda p: p["center"][1])
    east = sorted([p for p in ports if abs(p["orientation"]) < 1e-6 or abs(p["orientation"] - 360) < 1e-6], key=lambda p: p["center"][1])
    if len(west) + len(east) != len(ports):
        raise ValueError("EME needs every optical port to face west (180) or east (0)")
    xl = min(p["center"][0] for p in west) if west else None
    xr = max(p["center"][0] for p in east) if east else None
    sections, info = component_sections(component, layer_stack, cell_length=cell_length, x_range=(xl, xr), step=step, margin=margin, v_margin=v_margin, material_index=material_index, exclude=exclude, wafer_levels=wafer_levels, background=background)
    eme = EME(sections, num_modes=num_modes, pml=pml)
    eps_x = 1e-9 * (xr - xl)
    lp = _port_cross_sections(component, west, xl + eps_x, info, layer_stack, material_index, background)
    rp = _port_cross_sections(component, east, xr - eps_x, info, layer_stack, material_index, background)
    out = {}
    timings = {"modes": 0.0, "overlaps": 0.0, "cascade": 0.0, "ports": 0.0, "total": 0.0}
    names = None
    for i, w in enumerate(wl):
        t0 = time.perf_counter()
        eme.solve(w)
        t1 = time.perf_counter()
        left = [(n, xs.solve(w, port_modes, profile=False)) for n, xs in lp]
        right = [(n, xs.solve(w, port_modes, profile=False)) for n, xs in rp]
        S, names = eme.port_smatrix(left, right)
        t2 = time.perf_counter()
        for k in ("modes", "overlaps", "cascade"):
            timings[k] += eme.timings[k]
        timings["ports"] += t2 - t1
        timings["total"] += t2 - t0
        for a, na in enumerate(names):
            for b, nb in enumerate(names):
                out.setdefault(f"{na},{nb}", np.zeros(len(wl), dtype=complex))[i] = S[a, b]
    out["wavelengths"] = wl
    out["timings"] = timings
    out["sections"] = len(sections)
    out["distinct_cross_sections"] = info["distinct"]
    if return_eme:
        return out, eme
    return out


class EMEComponent:
    """EME of a gdsfactory component with the interface of
    ``gplugins.meow.MEOW`` (lengths in µm).

    ``compute_sparameters()`` returns ``{"o1@0,o2@0": ..., "wavelengths":
    ...}`` with the modes of the two end cross-sections as port modes
    (``o1@i``, ``o2@j`` for ``i, j < num_modes``), wavelengths in µm, as
    meow does. The component must have exactly two optical ports, ``o1``
    facing west and ``o2`` facing east. ``resolution_x`` / ``resolution_y``
    are replaced by the grid ``step`` (µm, default 0.025);
    ``spacing_x`` / ``spacing_y`` are the window margins.
    """

    def __init__(self, component, layer_stack=None, wavelength: float = 1.55, temperature: float = 25.0, num_modes: int = 4, cell_length: float = 0.5, spacing_x: float = 2.0, spacing_y: float = 2.0, step: float = 0.025, material_index=None, background="sio2", **_ignored):
        ports = {p.name: p for p in component.ports}
        if len(ports) != 2 or ports.get("o1") is None or ports.get("o2") is None:
            raise ValueError("EMEComponent needs exactly two optical ports o1 (west) and o2 (east)")
        if ports["o1"].orientation != 180 or ports["o2"].orientation != 0:
            raise ValueError("port o1 must face west (180) and o2 east (0)")
        self.component, self.layer_stack = component, layer_stack
        self.wavelength, self.num_modes, self.cell_length = wavelength, num_modes, cell_length
        self.spacing_x, self.spacing_y, self.step = spacing_x, spacing_y, step
        self.material_index, self.background = material_index, background
        self.eme = None
        self.S = None
        self.port_map = None

    def compute_sparameters(self) -> dict:
        sections, info = component_sections(self.component, self.layer_stack, cell_length=self.cell_length * um, margin=self.spacing_x / 2 * um, v_margin=self.spacing_y / 2 * um, step=self.step * um, material_index=self.material_index, background=self.background)
        self.eme = EME(sections, num_modes=self.num_modes)
        S = self.eme.solve(self.wavelength * um).full()
        n = self.num_modes
        names = [f"o1@{i}" for i in range(n)] + [f"o2@{i}" for i in range(n)]
        self.S, self.port_map = S, {nm: i for i, nm in enumerate(names)}
        sp = {f"{a},{b}": np.asarray(S[self.port_map[b], self.port_map[a]]) for a in names for b in names}
        sp["wavelengths"] = np.asarray(self.wavelength)
        return sp
