"""Composite S-matrix problems: a large device as a network of blocks, each
block simulated on its own by FDTD with its adjoint, the network solved by
scattering-matrix composition, and the gradient carried back through the
composition to every block.

The approximation is that blocks interact only through their ports: every
block is simulated alone, with absorbing boundaries where its neighbours
would be, and joined to them through the guided modes of the port
waveguides. What it leaves out is (i) the coupling through the non-guided
part of the field at a port plane (evanescent and radiation modes of the
connecting waveguide), which decays with the distance between the design
region and the port plane, and (ii) the light a block scatters out of its
ports that reaches another block through the cladding. By energy
conservation and reciprocity, (ii) is bounded per pair of blocks by the
geometric mean of the fractions of power the two blocks scatter out of their
ports (``docs/modules/composite.md``). Both are measured against the
monolithic FDTD simulation of the same layout in the documentation.

``CompositeProblem`` behaves like :class:`lightsim.invdes.SMatrixProblem`:
``solve(eps)`` returns an object with ``S`` and ``vjp(g)``, and the design
grids are those of the blocks, concatenated. With
:class:`CompositeDesign` (one :class:`lightsim.invdes.Design` per designed
block) it plugs into :class:`lightsim.invdes.InverseDesign` and
:func:`lightsim.invdes.optimize` unchanged.
"""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .network import Network

__all__ = ["Block", "CompositeProblem", "CompositeSolution", "CompositeDesign"]


def _is_member(problem):
    """A block that is one member of a problem solving several blocks at
    once (an object with ``layer``, ``index``, ``n_inputs``, ``n_outputs``,
    ``grids`` and ``freqs``, whose ``layer.solve`` returns an object with
    ``block(index)``)."""
    return hasattr(problem, "layer") and hasattr(problem, "index")


@dataclass
class Block:
    """One block of a :class:`CompositeProblem`.

    ``problem`` is an :class:`lightsim.invdes.SMatrixProblem` of the block
    alone, or a fixed S-matrix ``[P, P, K]`` (a block without design, such as
    a straight waveguide, computed once). ``in_ports`` / ``out_ports`` are
    the block port numbers (0 ... P-1, the numbering of the network) of the
    problem's inputs and outputs, in order; entries of the block S-matrix
    that no input-output pair measures are zero in the network (reflections
    left out, for instance). ``ports`` is P (default: the largest port
    number plus one)."""

    problem: object
    in_ports: list = None
    out_ports: list = None
    ports: int = None
    name: str = ""

    def __post_init__(self):
        if isinstance(self.problem, np.ndarray):
            P = self.problem.shape[0]
            self.ports = P if self.ports is None else int(self.ports)
            return
        if _is_member(self.problem):
            nin, nout = self.problem.n_inputs, self.problem.n_outputs
        else:
            nin, nout = len(self.problem.inputs), len(self.problem.outputs)
        self.in_ports = list(range(nin)) if self.in_ports is None else [int(p) for p in self.in_ports]
        self.out_ports = list(range(nout)) if self.out_ports is None else [int(p) for p in self.out_ports]
        if len(self.in_ports) != nin or len(self.out_ports) != nout:
            raise ValueError(f"block '{self.name}': in_ports / out_ports must list one port per input / output of its problem")
        if self.ports is None:
            self.ports = max(self.in_ports + self.out_ports) + 1

    @property
    def fixed(self):
        return isinstance(self.problem, np.ndarray)

    @property
    def unit(self):
        """The object that runs the simulations: the problem itself, or the
        group that solves several blocks at once (see :func:`_is_member`)."""
        return self.problem.layer if _is_member(self.problem) else self.problem

    @property
    def grids(self):
        return [] if self.fixed else list(self.problem.grids)


@dataclass
class CompositeSolution:
    """Result of :meth:`CompositeProblem.solve`: ``S[m, n, k]`` between the
    chosen external outputs and inputs, and :meth:`vjp`."""

    S: np.ndarray
    freqs: np.ndarray
    stats: dict
    T: np.ndarray = None  #: all external ports, ``[E, E, K]``
    block_S: list = None  #: per block ``[P, P, K]`` as used in the network
    _problem: object = None
    _net: object = None
    _sols: list = field(default=None, repr=False)

    def vjp(self, g):
        """dL/d(eps) on the samples of every design grid (the blocks' grids,
        concatenated in block order) for the upstream gradient ``g`` of the
        shape of ``S``."""
        p = self._problem
        g = np.asarray(g, dtype=complex)
        if g.shape != self.S.shape:
            raise ValueError(f"upstream gradient has shape {g.shape}, S has {self.S.shape}")
        gT = np.zeros(self.T.shape, dtype=complex)
        gT[np.ix_(p.outputs, p.inputs)] = g
        gS = self._net.vjp(gT)
        out = []
        for b, blk in enumerate(p.blocks):
            if blk.fixed:
                continue
            gb = gS[b][np.ix_(blk.out_ports, blk.in_ports)]
            if blk.grids:
                out.extend(self._sols[b].vjp(gb))
        return out


class CompositeProblem:
    """S-matrix of a network of blocks, each solved by its own FDTD problem.

    Parameters
    ----------
    blocks : list of :class:`Block`.
    network : the :class:`lightsim.invdes.Network` that joins them (its
        ``ports`` must equal the blocks' port counts).
    inputs, outputs : indices into the network's external ports; ``S`` is
        ``T[outputs][:, inputs]``.
    devices : a device string or a list of them; the blocks are solved
        concurrently, one block at a time per entry (``["cuda:0",
        "cuda:1"]``). Each block's own ``devices`` setting is replaced.
    reflections : keep the internal reflections of the network (the exact
        composition of the given block S-matrices). ``False`` drops the
        entries between ports on the same side (``network.sides``), which is
        the zeroth order of the multiple-reflection series.
    """

    def __init__(self, blocks, network: Network, inputs, outputs, devices="cpu", reflections=True):
        self.blocks = list(blocks)
        self.network = network
        if [b.ports for b in self.blocks] != network.ports:
            raise ValueError(f"block port counts {[b.ports for b in self.blocks]} differ from the network's {network.ports}")
        self.inputs = [int(i) for i in inputs]
        self.outputs = [int(o) for o in outputs]
        self.devices = [devices] if isinstance(devices, str) else list(devices)
        self.reflections = bool(reflections)
        fr = [b.problem.freqs for b in self.blocks if not b.fixed]
        self.freqs = np.asarray(fr[0]) if fr else None
        for f in fr:
            if not np.allclose(f, self.freqs):
                raise ValueError("all blocks must use the same frequencies")
        self.grids = [g for b in self.blocks for g in b.grids]
        self._split = np.cumsum([0] + [len(b.grids) for b in self.blocks])

    @property
    def shape(self):
        K = len(self.freqs) if self.freqs is not None else self.blocks[0].problem.shape[2]
        return (len(self.outputs), len(self.inputs), K)

    def block_eps(self, eps):
        """Split the concatenated permittivity list into one list per block."""
        eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
        if len(eps) != len(self.grids):
            raise ValueError(f"expected {len(self.grids)} permittivity arrays, got {len(eps)}")
        return [eps[self._split[b] : self._split[b + 1]] for b in range(len(self.blocks))]

    def solve(self, eps, adjoint=True) -> CompositeSolution:
        """Solve every designed block (forward runs, and adjoint runs with
        ``adjoint``), compose the network and return the solution."""
        t0 = time.time()
        per = self.block_eps(eps)
        devq = queue.Queue()
        for d in self.devices:
            devq.put(d)
        units, unit_of = [], {}
        for b, blk in enumerate(self.blocks):
            if blk.fixed:
                continue
            u = blk.unit
            k = next((i for i, x in enumerate(units) if x is u), None)
            if k is None:
                units.append(u)
                k = len(units) - 1
            unit_of[b] = k
        unit_eps = {}
        for b, k in unit_of.items():
            if _is_member(self.blocks[b].problem):
                unit_eps.setdefault(k, {})[self.blocks[b].problem.index] = per[b]
            else:
                unit_eps[k] = per[b]

        def eps_of(k):
            e = unit_eps[k]
            if isinstance(e, dict):  # a layer: its members' arrays in member order
                return [a for i in range(len(units[k].members)) for a in e[i]]
            return e

        def work(k):
            dev = devq.get()
            try:
                prob = units[k]
                prob.devices = [dev]
                return prob.solve(eps_of(k), adjoint=adjoint)
            finally:
                devq.put(dev)

        ks = list(range(len(units)))
        if len(self.devices) == 1:
            usols = [work(k) for k in ks]
        else:
            with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
                usols = list(ex.map(work, ks))
        sols = {}
        for b, k in unit_of.items():
            m = self.blocks[b].problem
            sols[b] = usols[k].block(m.index) if _is_member(m) else usols[k]
        K = self.shape[2]
        S_blocks, all_sols = [], [None] * len(self.blocks)
        for b, blk in enumerate(self.blocks):
            if blk.fixed:
                S_blocks.append(np.asarray(blk.problem, dtype=complex))
                continue
            Sb = np.zeros((blk.ports, blk.ports, K), dtype=complex)
            Sb[np.ix_(blk.out_ports, blk.in_ports)] = sols[b].S
            S_blocks.append(Sb)
            all_sols[b] = sols[b]
        t1 = time.time()
        net = self.network.solve(S_blocks, reflections=self.reflections, store=adjoint)
        runs = [r for u in usols for r in u.stats["runs"]]
        stats = {
            "wall": time.time() - t0,
            "network_seconds": time.time() - t1,
            "runs": runs,
            "units_solved": len(units),
            "blocks_solved": len(unit_of),
            "forward_runs": sum(u.stats["forward_runs"] for u in usols),
            "adjoint_runs": sum(u.stats["adjoint_runs"] for u in usols),
            "sim_seconds": float(sum(u.stats["sim_seconds"] for u in usols)),
            "cells": [u.stats["cells"] for u in usols],
        }
        S = net.T[np.ix_(self.outputs, self.inputs)]
        return CompositeSolution(S=S, freqs=self.freqs, stats=stats, T=net.T, block_S=S_blocks, _problem=self, _net=net, _sols=all_sols)


class CompositeDesign:
    """Several :class:`lightsim.invdes.Design` objects seen as one: the
    density is the list (or, when all have the same shape, the stacked array
    ``[B, nx, ny]``) of the blocks' densities, ``eps`` concatenates their
    permittivity arrays in the order of :attr:`CompositeProblem.grids`, and
    ``vjp`` splits the gradient back. It has the methods
    :class:`lightsim.invdes.InverseDesign` and :func:`lightsim.invdes.optimize`
    use."""

    def __init__(self, designs):
        self.designs = list(designs)
        self.shapes = [((d.fields,) if d.fields > 1 else ()) + d.shape for d in self.designs]
        self.stacked = len(set(self.shapes)) == 1
        self.eta = self.designs[0].eta
        self.shape = (len(self.designs),) + self.shapes[0] if self.stacked else None
        self.fields = 1

    def _parts(self, rho):
        if self.stacked:
            rho = np.asarray(rho, dtype=float)
            return [rho[i] for i in range(len(self.designs))]
        return list(rho)

    def _join(self, parts):
        return np.stack(parts) if self.stacked else parts

    def grids(self):
        """The design grids of all blocks, block after block, in the order of
        ``eps`` and of the gradients ``vjp`` takes."""
        return [g for d in self.designs for g in d.grids()]

    def random(self, seed=0, mean=0.5, amplitude=0.0):
        """A starting density for every block (see
        :meth:`lightsim.invdes.Design.random`); block ``i`` uses ``seed + i``."""
        return self._join([d.random(seed + i, mean, amplitude) for i, d in enumerate(self.designs)])

    def eps(self, rho, beta, eta=None):
        """The permittivity arrays of all blocks for the density ``rho``, at
        projection strength ``beta``, in the order of :meth:`grids`."""
        return [e for d, r in zip(self.designs, self._parts(rho)) for e in d.eps(r, beta, eta)]

    def vjp(self, rho, beta, grads, eta=None):
        """Gradient with respect to the density from the gradients with
        respect to the permittivity arrays (``grads``, in the order of
        :meth:`grids`): each block's part goes through its own filter and
        projection."""
        out, i = [], 0
        for d, r in zip(self.designs, self._parts(rho)):
            n = len(d.layers)
            out.append(d.vjp(r, beta, grads[i : i + n], eta))
            i += n
        return self._join(out)

    def projected(self, rho, beta, eta=None):
        """The filtered and projected density of every block."""
        return self._join([d.projected(r, beta, eta) for d, r in zip(self.designs, self._parts(rho))])

    def binarization(self, rho, beta, eta=None):
        """Mean over the blocks of their binarisation (1 for a design of
        only the two materials)."""
        vals = [d.binarization(r, beta, eta) for d, r in zip(self.designs, self._parts(rho))]
        return float(np.mean(vals))
