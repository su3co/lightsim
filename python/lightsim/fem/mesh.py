"""Triangle meshes and finite-element fields of :mod:`lightsim.fem`."""

from __future__ import annotations

import numpy as np

__all__ = ["Mesh", "Field", "rectangle_mesh"]


class Mesh:
    """A triangle mesh of a cross-section with named regions.

    Attributes
    ----------
    nodes : ndarray (N, 2)
        Node coordinates (m).
    tris : ndarray (M, 3)
        Counter-clockwise triangles.
    tags : ndarray (M,)
        Region index of every triangle (into ``region_names``).
    region_names, materials : list of str
        Name and material of every region index.
    edges : ndarray (E, 2)
        Global edges (lower node first); ``edge_tris`` (E, 2) the triangles
        on each (-1 for none), ``tri_edges`` (M, 3) the edges of each
        triangle (edge k joins local nodes k and k+1).
    edge_mid : ndarray (E, 2)
        Midpoint of curved edges on the true boundary (NaN: straight).
    """

    def __init__(self, nodes, tris, tags, region_names=None, materials=None, edge_mid=None, _info=None):
        from .. import _core

        nodes = np.ascontiguousarray(nodes, dtype=float)
        tris = np.ascontiguousarray(tris, dtype=np.int64)
        tags = np.ascontiguousarray(tags, dtype=np.int64)
        info = _info if _info is not None else _core.fem_mesh_info(nodes, tris, tags)
        self.nodes = np.asarray(info["nodes"], float)
        self.tris = np.asarray(info["tris"], np.int64)
        self.tags = np.asarray(info["tags"], np.int64)
        self.edges = np.asarray(info["edges"], np.int64)
        self.edge_tris = np.asarray(info["edge_tris"], np.int64)
        self.tri_edges = np.asarray(info["tri_edges"], np.int64)
        em = np.asarray(edge_mid if edge_mid is not None else info["edge_mid"], float)
        self.edge_mid = em if em.shape == (len(self.edges), 2) else np.full((len(self.edges), 2), np.nan)
        ntag = int(self.tags.max()) + 1 if len(self.tags) else 0
        self.region_names = list(region_names) if region_names is not None else [f"region{i}" for i in range(ntag)]
        self.materials = list(materials) if materials is not None else list(self.region_names)
        self.min_angle_deg = float(np.degrees(self._min_angles().min())) if len(self.tris) else 0.0

    # ------------------------------------------------------------------
    def _args(self):
        return self.nodes, self.tris, self.tags, self.edge_mid

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    def __repr__(self) -> str:
        return f"<Mesh {self.n_nodes} nodes, {self.n_tris} triangles, regions {self.region_names}, min angle {self.min_angle_deg:.1f} deg>"

    def tag(self, name: str) -> int:
        """Region index of a region name (or material name)."""
        if name in self.region_names:
            return self.region_names.index(name)
        hits = [i for i, m in enumerate(self.materials) if m == name]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"no region named {name!r}; regions: {self.region_names}")

    def tags_of(self, names) -> list[int]:
        """Region indices of one name or several (names or materials)."""
        if isinstance(names, str):
            names = [names]
        out = []
        for n in names:
            if n in self.region_names:
                out.append(self.region_names.index(n))
            else:
                hits = [i for i, m in enumerate(self.materials) if m == n]
                if not hits:
                    raise KeyError(f"no region or material named {n!r}; regions: {self.region_names}, materials: {self.materials}")
                out.extend(hits)
        return out

    def region_mask(self, names) -> np.ndarray:
        """Boolean mask of the triangles of the named regions."""
        return np.isin(self.tags, self.tags_of(names))

    def areas(self) -> np.ndarray:
        p = self.nodes[self.tris]
        return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))

    def centroids(self) -> np.ndarray:
        return self.nodes[self.tris].mean(axis=1)

    def _min_angles(self) -> np.ndarray:
        p = self.nodes[self.tris]
        out = np.full(len(p), np.pi)
        for k in range(3):
            u = p[:, (k + 1) % 3] - p[:, k]
            v = p[:, (k + 2) % 3] - p[:, k]
            c = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
            out = np.minimum(out, np.arccos(np.clip(c, -1, 1)))
        return out

    def edge_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.nodes[self.edges[:, 1]] - self.nodes[self.edges[:, 0]], axis=1)

    @property
    def bounds(self):
        return self.nodes[:, 0].min(), self.nodes[:, 1].min(), self.nodes[:, 0].max(), self.nodes[:, 1].max()

    # ------------------------------------------------------------------
    def boundary_edges(self, where=None) -> np.ndarray:
        """Indices of outer-boundary edges.

        ``where``: ``None`` (all), ``"top"``, ``"bottom"``, ``"left"``,
        ``"right"`` (the sides of the bounding box), a list of those, or a
        function ``f(x, y) -> bool`` of the edge midpoints.
        """
        e = np.nonzero(self.edge_tris[:, 1] < 0)[0]
        if where is None:
            return e
        mid = 0.5 * (self.nodes[self.edges[e, 0]] + self.nodes[self.edges[e, 1]])
        x0, y0, x1, y1 = self.bounds
        tol = 1e-9 * max(x1 - x0, y1 - y0)
        sides = {
            "bottom": lambda x, y: np.abs(y - y0) < tol,
            "top": lambda x, y: np.abs(y - y1) < tol,
            "left": lambda x, y: np.abs(x - x0) < tol,
            "right": lambda x, y: np.abs(x - x1) < tol,
        }
        if callable(where):
            keep = np.asarray(where(mid[:, 0], mid[:, 1]), bool)
        else:
            keep = np.zeros(len(e), bool)
            for w in [where] if isinstance(where, str) else where:
                if w not in sides:
                    raise ValueError(f"unknown boundary {w!r}; use top, bottom, left, right or a function f(x, y)")
                keep |= sides[w](mid[:, 0], mid[:, 1])
        return e[keep]

    def interface_edges(self, a, b=None) -> np.ndarray:
        """Edges between region(s) ``a`` and region(s) ``b`` (any other
        region, or the outer boundary, if ``b`` is None)."""
        ta = self.tags_of(a)
        t0, t1 = self.edge_tris[:, 0], self.edge_tris[:, 1]
        g0 = self.tags[t0]
        g1 = np.where(t1 >= 0, self.tags[np.maximum(t1, 0)], -1)
        in0, in1 = np.isin(g0, ta), np.isin(g1, ta)
        if b is None:
            return np.nonzero(in0 ^ in1)[0]
        tb = self.tags_of(b)
        return np.nonzero((in0 & np.isin(g1, tb)) | (in1 & np.isin(g0, tb)))[0]

    def edge_dofs(self, edges, order: int) -> np.ndarray:
        """Lagrange DOFs on edges: their nodes, plus the midside DOF for P2."""
        edges = np.asarray(edges, np.int64)
        d = [self.edges[edges].ravel()]
        if order == 2:
            d.append(self.n_nodes + edges)
        return np.unique(np.concatenate(d))

    def region_dofs(self, names, order: int) -> np.ndarray:
        """Lagrange DOFs of the closed triangles of named regions."""
        mask = self.region_mask(names)
        d = [self.tris[mask].ravel()]
        if order == 2:
            d.append(self.n_nodes + self.tri_edges[mask].ravel())
        return np.unique(np.concatenate(d))

    def dof_coords(self, order: int) -> np.ndarray:
        if order == 1:
            return self.nodes.copy()
        mid = 0.5 * (self.nodes[self.edges[:, 0]] + self.nodes[self.edges[:, 1]])
        curved = np.isfinite(self.edge_mid[:, 0])
        mid[curved] = self.edge_mid[curved]
        return np.concatenate([self.nodes, mid])

    def ndof(self, order: int) -> int:
        return self.n_nodes + (len(self.edges) if order == 2 else 0)

    def refine(self) -> "Mesh":
        """Uniformly refined mesh (each triangle split into four)."""
        nn = self.n_nodes
        mid = self.dof_coords(2)[nn:]
        nodes = np.concatenate([self.nodes, mid])
        m = nn + self.tri_edges
        t = self.tris
        tris = np.concatenate(
            [
                np.stack([t[:, 0], m[:, 0], m[:, 2]], 1),
                np.stack([m[:, 0], t[:, 1], m[:, 1]], 1),
                np.stack([m[:, 2], m[:, 1], t[:, 2]], 1),
                np.stack([m[:, 0], m[:, 1], m[:, 2]], 1),
            ]
        )
        tags = np.concatenate([self.tags] * 4)
        return Mesh(nodes, tris, tags, self.region_names, self.materials)

    def per_region(self, values: dict, default=None, what: str = "value") -> np.ndarray:
        """Per-triangle array from ``{region name or material: value}``
        (region names take precedence over materials)."""
        out = [None] * len(self.region_names)
        for i, (n, m) in enumerate(zip(self.region_names, self.materials)):
            if n in values:
                out[i] = values[n]
            elif m in values:
                out[i] = values[m]
            elif m.lower() in {str(k).lower(): k for k in values}:
                out[i] = {str(k).lower(): v for k, v in values.items()}[m.lower()]
            elif default is not None:
                out[i] = default(m) if callable(default) else default
            else:
                raise KeyError(f"no {what} for region {n!r} (material {m!r})")
        return out

    # ------------------------------------------------------------------
    def plot(self, ax=None, show_regions: bool = True, lw: float = 0.3):
        """Plot the mesh with region colours (needs matplotlib)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if show_regions:
            ax.tripcolor(self.nodes[:, 0] * 1e6, self.nodes[:, 1] * 1e6, self.tris, facecolors=self.tags.astype(float), cmap="tab20", alpha=0.5)
        ax.triplot(self.nodes[:, 0] * 1e6, self.nodes[:, 1] * 1e6, self.tris, lw=lw, color="k")
        ax.set_aspect("equal")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        return ax


class Field:
    """A scalar Lagrange (P1 or P2) field on a mesh.

    Call it with points to evaluate: ``f(x, y)`` (arrays, metres) returns the
    values (NaN outside the mesh); :meth:`grad` the gradient.
    """

    def __init__(self, mesh: Mesh, order: int, values):
        self.mesh = mesh
        self.order = order
        self.values = np.asarray(values, float)

    def _eval(self, x, y):
        from .. import _core

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shape = np.broadcast(x, y).shape
        pts = np.stack(np.broadcast_arrays(x, y), -1).reshape(-1, 2)
        v, g = _core.fem_eval(*self.mesh._args(), self.order, self.values, np.ascontiguousarray(pts))
        return v[0].reshape(shape), g[0].reshape(shape + (2,))

    def __call__(self, x, y):
        return self._eval(x, y)[0]

    def grad(self, x, y):
        """Gradient (..., 2) at points."""
        return self._eval(x, y)[1]

    def at_nodes(self) -> np.ndarray:
        return self.values[: self.mesh.n_nodes]

    def per_triangle(self) -> np.ndarray:
        """Average over the vertices of every triangle."""
        return self.at_nodes()[self.mesh.tris].mean(axis=1)

    def integrate(self, region=None, degree: int = 6) -> float:
        """Integral of the field over the mesh or over named regions."""
        q = self.quadrature(degree)
        w = q["weights"]
        if region is not None:
            w = w * self.mesh.region_mask(region)[q["tri"]]
        return float(np.sum(w * q["values"]))

    def quadrature(self, degree: int = 6) -> dict:
        """Points, weights, triangle index, values and gradients at quadrature points."""
        from .. import _core

        return _core.fem_quadrature(*self.mesh._args(), self.order, self.values, degree)

    def max(self) -> float:
        return float(np.max(self.values))

    def plot(self, ax=None, colorbar: bool = True, **kw):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        m = self.mesh
        tp = ax.tripcolor(m.nodes[:, 0] * 1e6, m.nodes[:, 1] * 1e6, m.tris, self.at_nodes(), shading="gouraud", **kw)
        if colorbar:
            plt.colorbar(tp, ax=ax)
        ax.set_aspect("equal")
        return ax


def rectangle_mesh(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int, name: str = "domain", material: str = "") -> Mesh:
    """Structured triangle mesh of a rectangle (two triangles per cell)."""
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    tris = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i
            if (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    tris = np.array(tris)
    return Mesh(nodes, tris, np.zeros(len(tris), int), [name], [material or name])
