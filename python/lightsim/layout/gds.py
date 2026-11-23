"""GDSII layouts without gdsfactory: :func:`read_gds` and :class:`Layout`.

The reading is done by the Rust core (``lightsim._core.read_gds``), which
flattens one cell: polygons, paths (``PATHTYPE`` 0, 1, 2, 4, mitred
corners), boxes, references and arrays with rotation, magnification and
reflection, text labels, and the units of the file. Coordinates are in
metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _layer(spec) -> tuple[int, int]:
    """``(layer, datatype)`` from ``(1, 0)``, ``[1, 0]``, ``"1/0"`` or ``1``."""
    if isinstance(spec, str):
        a, _, b = spec.partition("/")
        return int(a), int(b or 0)
    if isinstance(spec, int | np.integer):
        return int(spec), 0
    return int(spec[0]), int(spec[1])


@dataclass
class Layout:
    """One flattened cell of a GDSII file (metres).

    Attributes
    ----------
    cell : str
        The cell read.
    polygons : dict
        ``{(layer, datatype): [array (N, 2), ...]}``; paths and boxes are
        included as polygons.
    paths : list
        ``(layer, datatype, width, pathtype, points (N, 2))``: the centre
        lines of the paths (pins are paths in SiEPIC's convention).
    labels : list
        ``(text, layer, texttype, x, y)``.
    cells, top_cells : list of str
        All cells of the file and those no other cell references.
    database_unit_m, user_unit_m : float
        The file's units.
    """

    cell: str
    polygons: dict
    paths: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    top_cells: list = field(default_factory=list)
    database_unit_m: float = 1e-9
    user_unit_m: float = 1e-6
    path: str = ""

    def layer(self, spec) -> list:
        """The polygons of one layer, ``(layer, datatype)`` or ``"1/0"``."""
        return self.polygons.get(_layer(spec), [])

    @property
    def layers(self) -> list[tuple[int, int]]:
        """The ``(layer, datatype)`` pairs that have polygons, sorted."""
        return sorted(self.polygons)

    def bbox(self, layers=None) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((xmin, ymin), (xmax, ymax))`` of the polygons (of ``layers`` if given)."""
        keys = self.layers if layers is None else [_layer(l) for l in layers]
        pts = [p for k in keys for p in self.polygons.get(k, [])]
        if not pts:
            raise ValueError(f"no polygons on layers {keys}")
        allp = np.concatenate(pts)
        return (float(allp[:, 0].min()), float(allp[:, 1].min())), (float(allp[:, 0].max()), float(allp[:, 1].max()))


def read_gds(path, cell: str | None = None) -> Layout:
    """Read and flatten ``cell`` of a GDSII file (default: its only top cell).

    No gdsfactory, gdstk or KLayout is needed. For an OASIS file, convert it
    first, for example with KLayout: ``klayout -zz -rd input=chip.oas -rd
    output=chip.gds -r convert.py`` or, interactively, File > Save As with
    the GDS2 format.
    """
    from .. import _core

    p = str(Path(path))
    if p.lower().endswith((".oas", ".oasis")):
        raise ValueError(f"{p}: OASIS is not read; convert it to GDSII first (KLayout: open it and save as GDS2)")
    d = _core.read_gds(p, cell)
    polys = {tuple(int(v) for v in k): [np.asarray(a) for a in v] for k, v in d["polygons"].items()}
    paths = [(int(l), int(dt), float(w), int(pt), np.asarray(pts)) for l, dt, w, pt, pts in d.get("paths", [])]
    return Layout(
        cell=d["cell"],
        polygons=polys,
        paths=paths,
        labels=[(t, int(l), int(tt), float(x), float(y)) for t, l, tt, x, y in d["labels"]],
        cells=list(d["cells"]),
        top_cells=list(d["top_cells"]),
        database_unit_m=float(d["database_unit_m"]),
        user_unit_m=float(d["user_unit_m"]),
        path=p,
    )
