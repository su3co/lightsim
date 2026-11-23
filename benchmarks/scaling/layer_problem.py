"""LayerProblem: all blocks of one mesh layer in one simulation (one run per
port and colour), as a member type for lightsim.invdes.CompositeProblem.

Research code kept to reproduce the measurement in docs/modules/composite.md:
it needs fewer runs than one problem per block but was slower in every case
measured (a strip holding a whole layer takes far longer to decay), so it is
not part of the lightsim API.
"""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from lightsim.invdes.smatrix import E_COMPONENTS, SMatrixProblem, SMatrixSolution

# ---------------------------------------------------------------------------
# several blocks in one simulation


@dataclass
class LayerMember:
    """Block ``index`` of a :class:`LayerProblem`, as the ``problem`` of a
    :class:`Block`."""

    layer: object
    index: int

    @property
    def n_inputs(self):
        return len(self.layer.members[self.index]["in"])

    @property
    def n_outputs(self):
        return len(self.layer.members[self.index]["out"])

    @property
    def grids(self):
        return list(self.layer.members[self.index]["grids"])

    @property
    def freqs(self):
        return self.layer.freqs


class _GridView:
    def __init__(self, grids, components, meta):
        self.grids, self.components, self._meta = grids, components, meta


class LayerSolution:
    """Result of :meth:`LayerProblem.solve`: :meth:`block` gives each
    member's :class:`lightsim.invdes.SMatrixSolution` (S and vjp)."""

    def __init__(self, blocks, stats):
        self._blocks = blocks
        self.stats = stats

    def block(self, i):
        return self._blocks[i]


class LayerProblem(SMatrixProblem):
    """Several blocks that do not interact, simulated together.

    The blocks sit side by side in one simulation ``sim`` (for example one
    layer of a mesh). Every forward run excites the same local input of all
    the blocks of one colour at once and every adjoint run the same local
    output, so the number of runs is ``colours * (n_in + n_out)`` however
    many blocks there are; the blocks of one colour must be far enough apart
    that what one scatters does not reach another's ports (with two colours,
    neighbours are never excited together). The fields in each block's
    design grids and the amplitudes at its ports are attributed to that block
    alone: this is the approximation of the composite model, made inside one
    simulation.

    ``members``: list of dicts with ``grids`` (the block's DesignGrid
    list), ``ports`` (its :class:`lightsim.invdes.Port` list, in the block's
    port numbering), ``in`` and ``out`` (port numbers of the inputs and
    outputs) and ``colour`` (an integer). All members must have the same
    numbers of inputs and outputs.
    """

    def __init__(self, sim, members, freqs, devices="cpu", fwidth=None, components=E_COMPONENTS, progress=0):
        self.members = []
        grids, ins, outs = [], [], []
        for m in members:
            g0 = len(grids)
            grids.extend(m["grids"])
            mi = {"grids": list(m["grids"]), "gidx": list(range(g0, len(grids))), "in": [int(p) for p in m["in"]], "out": [int(p) for p in m["out"]], "colour": int(m.get("colour", 0))}
            mi["in_flat"] = list(range(len(ins), len(ins) + len(mi["in"])))
            ins.extend(m["ports"][p] for p in mi["in"])
            mi["out_flat"] = list(range(len(outs), len(outs) + len(mi["out"])))
            outs.extend(m["ports"][p] for p in mi["out"])
            self.members.append(mi)
        if len({len(m["in"]) for m in self.members}) != 1 or len({len(m["out"]) for m in self.members}) != 1:
            raise ValueError("all members of a layer need the same numbers of inputs and outputs")
        super().__init__(sim, grids, ins, outs, freqs, devices=devices, fwidth=fwidth, normalize="monitor", components=components, progress=progress, engine="python")
        self.colours = sorted({m["colour"] for m in self.members})

    @property
    def n_in(self):
        return len(self.members[0]["in"])

    @property
    def n_out(self):
        return len(self.members[0]["out"])

    def _run_group(self, kind, colour, li, eps, device):
        from ..spec import DFTMonitor, ModeMonitor, ModeSource

        t0 = time.time()
        s = self.simulation(eps)
        mem = [b for b, m in enumerate(self.members) if m["colour"] == colour]
        src = {}
        for b in mem:
            m = self.members[b]
            port, mode = (self.inputs[m["in_flat"][li]] if kind == "fwd" else self.outputs[m["out_flat"][li]])
            src[b] = (port, mode)
            s.add_source(ModeSource(port.axis, port.position, port.direction, port.min, port.max, port.profiles[mode], self.signal, name=f"__src{b}__"))
            for t, (x, mn, mx) in enumerate(self._norm_planes(port)):
                s.add_monitor(ModeMonitor(f"__norm{b}_{t}__", port.axis, x, mn, mx, self.freqs, port.profiles))
            for gi in m["gidx"]:
                lo, hi = self._monitor_box(self.grids[gi])
                s.add_monitor(DFTMonitor(f"__des{gi}__", lo, hi, self.freqs, components=self.components))
            if kind == "fwd":
                for k in m["out_flat"]:
                    p, _ = self.outputs[k]
                    if p is not port:
                        s.add_monitor(ModeMonitor(f"__out_{k}__", p.axis, p.position, p.min, p.max, self.freqs, p.profiles))
        r = s.run(device=device, progress=self.progress)
        back = lambda p: "-" if p.direction == "+" else "+"  # noqa: E731
        fields, amps = {}, {}
        for b in mem:
            port, mode = src[b]
            a1 = self._amp(r, f"__norm{b}_0__", mode, port.direction)
            a2 = self._amp(r, f"__norm{b}_1__", mode, port.direction)
            q = a1 / a2
            norm = a1 * q
            m = self.members[b]
            for gi in m["gidx"]:
                d = {}
                for c in self.components:
                    f = r.raw["dft"][f"__des{gi}__"]["fields"][c]
                    if self._meta_pending:
                        self._origins[(gi, c)] = (list(f["origin"]), tuple(np.shape(f["re"])[1:]))
                    d[c] = ((np.asarray(f["re"]) + 1j * np.asarray(f["im"])) / norm[:, None, None, None]).astype(np.complex128)
                fields[gi] = d
            if kind == "fwd":
                a = np.zeros((len(m["out_flat"]), len(self.freqs)), dtype=complex)
                for j, k in enumerate(m["out_flat"]):
                    p, md = self.outputs[k]
                    if p is port:
                        a[j] = self._amp(r, f"__norm{b}_0__", md, back(p)) / q / norm
                    else:
                        a[j] = self._amp(r, f"__out_{k}__", md, back(p)) / norm
                amps[b] = a
        st = dict(r.stats)
        st["wall"] = time.time() - t0
        st["kind"], st["index"], st["colour"], st["device"] = kind, li, colour, device
        return fields, amps, st

    def solve(self, eps, adjoint=True) -> LayerSolution:
        eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
        t0 = time.time()
        self._meta_pending = self._meta is None
        self._origins = {}
        jobs = [("fwd", c, i) for c in self.colours for i in range(self.n_in)]
        if adjoint:
            jobs += [("adj", c, i) for c in self.colours for i in range(self.n_out)]
        devq = queue.Queue()
        for d in self.devices:
            devq.put(d)

        def work(job):
            dev = devq.get()
            try:
                return self._run_group(job[0], job[1], job[2], eps, dev)
            finally:
                devq.put(dev)

        if len(self.devices) == 1:
            res = [work(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
                res = list(ex.map(work, jobs))
        if self._meta_pending:
            self._build_meta()
        K = len(self.freqs)
        runs = [r[2] for r in res]
        stats = {"wall": time.time() - t0, "runs": runs, "forward_runs": sum(1 for j in jobs if j[0] == "fwd"), "adjoint_runs": sum(1 for j in jobs if j[0] == "adj"), "cells": runs[0]["cells"], "steps": [r["steps"] for r in runs], "sim_seconds": float(sum(r["seconds"] for r in runs))}
        out = []
        for b, m in enumerate(self.members):
            S = np.zeros((self.n_out, self.n_in, K), dtype=complex)
            fwd = [None] * self.n_in
            adj = [None] * self.n_out if adjoint else None
            for (kind, c, i), (fields, amps, _) in zip(jobs, res):
                if c != m["colour"]:
                    continue
                f = [fields[gi] for gi in m["gidx"]]
                if kind == "fwd":
                    S[:, i, :] = amps[b]
                    fwd[i] = f
                else:
                    adj[i] = f
            view = _GridView(m["grids"], self.components, [self._meta[gi] for gi in m["gidx"]])
            out.append(SMatrixSolution(S=S, freqs=self.freqs, stats=stats, _problem=view, _fwd=fwd, _adj=adj))
        return LayerSolution(out, stats)
