"""Gradients computed inside the engine (Rust, CPU and GPU).

Two entry points, both for designs given as :func:`lightsim.IndexGrid`
structures (:class:`lightsim.invdes.DesignGrid`) painted after every other
structure:

* :class:`GradientSession` -- a persistent multi-port, multi-frequency
  S-matrix session. The scene is rasterised and the sources / monitors of
  every excitation are compiled once; ``set_design`` rewrites only the design
  box on the device; ``forward`` returns the complex ``S[M, N, K]``;
  ``backward(g)`` runs the adjoint simulations, forms
  ``Re[conj(g) (i w eps0 / 4) E_adj E_fwd] dV`` on the device and applies the
  transpose of the rasterisation. This is what
  ``lightsim.invdes.SMatrixProblem(..., engine="native")`` uses.

* :class:`ReverseSession` / :func:`value_and_grad` -- discrete reverse-mode
  differentiation through the time loop for time-domain objectives (the
  energy crossing a plane in a time window, or any function of recorded
  field samples). The forward fields are reconstructed by exact time
  reversal while the adjoint is stepped, so nothing of the field history is
  stored and the result is the gradient of the discrete scheme itself.

Gradient convention (same as :mod:`lightsim.invdes`): the upstream gradient of
a real loss ``L`` with respect to a complex ``S`` entry is
``g = dL/dRe(S) + i dL/dIm(S)``, so that ``dL = Re(conj(g) dS)``; this is what
``torch.Tensor.grad`` holds for a complex leaf (JAX cotangents are its
conjugate). Gradients with respect to the design are plain real arrays
``dL/d(eps)`` with the shape of the index-grid samples.

See ``docs/modules/autodiff.md``.
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import _core
from .spec import GaussianPulse, Simulation

__all__ = ["GradientSession", "ReverseSession", "value_and_grad", "flux_energy", "field_samples"]


def _grids(grids):
    from .invdes.smatrix import DesignGrid

    return [grids] if isinstance(grids, DesignGrid) else list(grids)


def _base(sim: Simulation, grids, keep_sources: bool) -> dict:
    """Spec dict of the base simulation with the design media added."""
    d = sim.to_dict()
    d["monitors"] = []
    if not keep_sources:
        d["sources"] = []
    media = list(d["media"])
    for g in grids:
        if not any(m["name"] == g.medium["name"] for m in media):
            media.append(g.medium)
    d["media"] = media
    return d


def _grid_specs(grids) -> list:
    return [{"name": g.name, "min": [float(v) for v in g.min], "max": [float(v) for v in g.max], "dims": [int(v) for v in g.dims], "medium": g.medium["name"]} for g in grids]


def _eps_list(grids, eps) -> list:
    eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
    if len(eps) != len(grids):
        raise ValueError(f"expected {len(grids)} permittivity arrays, got {len(eps)}")
    return [np.ascontiguousarray(np.asarray(e, dtype=np.float64).reshape(g.dims)) for g, e in zip(grids, eps)]


class GradientSession:
    """Persistent native S-matrix session.

    Parameters are those of :class:`lightsim.invdes.SMatrixProblem`:
    ``sim`` (base simulation; its sources and monitors are ignored), ``grids``
    (:class:`lightsim.invdes.DesignGrid`), ``inputs`` (``Port``,
    ``(Port, mode)`` or ``Excitation``), ``outputs`` (``Port`` or
    ``(Port, mode)``), ``freqs`` (Hz). ``devices`` is a device string or a
    list of them (``["cuda:0", "cuda:1"]``): the independent runs are
    dispatched over the list, one at a time per entry, and the design-region
    fields are gathered on the first device for the adjoint product.

        sess = ls.autodiff.GradientSession(sim, grids, [p_in], [p_out], freqs, devices="cuda:0")
        sess.set_design(eps)                 # only the design box is re-rasterised and uploaded
        S = sess.forward()                   # complex [M, N, K]
        grads = sess.backward(dL_dS)         # list of dL/d(eps), one per grid

    ``forward(adjoint=True)`` also runs every adjoint simulation in the same
    dispatch (useful with several devices); otherwise ``backward`` runs the
    adjoint simulations the upstream gradient needs (outputs whose rows of
    ``g`` are zero are skipped) and keeps them for further ``backward`` calls
    on the same design.

    The design-region fields and the port monitors are sampled every
    ``dft_stride`` time steps (attribute of the session): the stride of
    ``sim`` (``Simulation.dft_stride``), by default the automatic one, which
    is as large as the bandwidth of the source allows.
    """

    def __init__(self, sim, grids, inputs, outputs, freqs, devices="cpu", fwidth=None, normalize="monitor", components=("Ex", "Ey", "Ez"), signal=None):
        from .invdes.smatrix import Excitation, Port

        self.grids = _grids(grids)
        self.freqs = np.asarray(freqs, dtype=float).reshape(-1)
        devices = [devices] if isinstance(devices, str) else list(devices)
        f0 = 0.5 * (self.freqs.min() + self.freqs.max())
        self.signal = signal or GaussianPulse(f0, float(fwidth) if fwidth else max(float(np.ptp(self.freqs)), 0.1 * f0))
        ports: list = []

        def port_index(p):
            for i, q in enumerate(ports):
                if q is p:
                    return i
            ports.append(p)
            return len(ports) - 1

        def channel(c, allow_excitation):
            if isinstance(c, Excitation):
                if not allow_excitation:
                    raise ValueError("outputs must be port modes")
                norm = None
                if c.norm is not None:
                    v = np.broadcast_to(np.asarray(c.norm, dtype=complex), self.freqs.shape)
                    norm = [[float(z.real), float(z.imag)] for z in v]
                return {"type": "source", "name": c.name, "source": dict(c.source), "norm": norm}
            port, mode = (c, 0) if isinstance(c, Port) else c
            return {"type": "port", "port": port_index(port), "mode": int(mode)}

        ins = [channel(c, True) for c in inputs]
        outs = [channel(c, False) for c in outputs]
        spec = {
            "sim": _base(sim, self.grids, keep_sources=False),
            "grids": _grid_specs(self.grids),
            "ports": [
                {"name": p.name, "axis": int(p.axis), "position": float(p.position), "min": [float(v) for v in p.min], "max": [float(v) for v in p.max], "profiles": list(p.profiles), "direction": p.direction}
                for p in ports
            ],
            "inputs": ins,
            "outputs": outs,
            "freqs": [float(f) for f in self.freqs],
            "signal": self.signal,
            "normalize": normalize,
            "components": list(components),
        }
        self._s = _core.GradientSession(json.dumps(spec), devices)
        self.shape = tuple(self._s.shape)
        self.devices = list(self._s.devices)
        self.cells = self._s.cells
        self.setup_seconds = self._s.setup_seconds
        self.dft_stride = self._s.dft_stride

    def set_design(self, eps) -> None:
        """Set the index-grid permittivities (one array per grid)."""
        self._s.set_design(_eps_list(self.grids, eps))

    def forward(self, adjoint: bool = False) -> np.ndarray:
        """Run the N excitations; complex ``S[M, N, K]``."""
        re, im = self._s.forward(adjoint)
        return np.asarray(re) + 1j * np.asarray(im)

    def _g(self, g):
        g = np.asarray(g, dtype=complex)
        if g.shape != self.shape:
            raise ValueError(f"upstream gradient has shape {g.shape}, S has {self.shape}")
        return np.ascontiguousarray(g.real), np.ascontiguousarray(g.imag)

    def backward(self, g) -> list:
        """dL/d(eps) on the samples of every design grid (list of arrays of
        shape ``grid.dims``) for the upstream gradient ``g`` (see the module
        docstring for the convention)."""
        return [np.asarray(a) for a in self._s.backward(*self._g(g))]

    def backward_yee(self, g) -> list:
        """dL/d(eps_r) on the Yee voxels of every design box: list over grids
        of ``{comp: (origin, real array [i, j, k])}``."""
        return [{c: (list(o), np.asarray(a)) for c, (o, a) in d.items()} for d in self._s.backward_yee(*self._g(g))]

    def yee_permittivity(self) -> dict:
        """What the engine uses: ``{comp: (origin, array)}`` over the design
        bounding box, NaN where no grid owns the voxel."""
        return {c: (list(o), np.asarray(a)) for c, (o, a) in self._s.yee_permittivity().items()}

    def stats(self) -> list:
        """Runs done since the last ``set_design``."""
        return list(self._s.stats())


def flux_energy(axis, position, min, max, t_start=None, t_end=None, taper=None) -> dict:
    """Objective: the energy (J) crossing the plane ``axis = position``
    (restricted to ``[min, max]``) towards ``+axis`` within a time window,
    1 on ``[t_start, t_end]`` (defaults: the whole run) with raised-cosine
    edges of width ``taper``. In a 2.5-D simulation it is the energy per
    z cell."""
    return {"type": "flux", "axis": int(axis), "position": float(position), "min": [float(v) for v in min], "max": [float(v) for v in max], "t_start": t_start, "t_end": t_end, "taper": taper}


def field_samples(min, max, components=("Ex", "Ey", "Ez")) -> dict:
    """Objective evaluated by the caller on the recorded samples of
    ``components`` in the box ``[min, max]`` (every step): pass a function
    ``fn(series) -> (L, cotangents)`` to :func:`value_and_grad` or use
    :meth:`ReverseSession.series` / :meth:`ReverseSession.backward`."""
    return {"type": "fields", "min": [float(v) for v in min], "max": [float(v) for v in max], "components": list(components)}


class ReverseSession:
    """Time-domain reverse mode for a simulation with a design region.

    ``sim`` keeps its sources (its monitors are ignored); ``objective`` is
    :func:`flux_energy` or :func:`field_samples`. The session is persistent:
    ``set_design`` only uploads the design box.

        sess = ls.autodiff.ReverseSession(sim, grids, ls.autodiff.flux_energy(0, x, mn, mx), device="cuda:0")
        sess.set_design(eps)
        L = sess.forward()          # runs the simulation, records the monitor (and the PML shell)
        grads = sess.backward()     # one reverse pass: list of dL/d(eps), one per grid

    With :func:`field_samples`, ``series()`` returns the recorded samples as
    ``{comp: array [step, i, j, k]}`` (E at ``(n + 1) dt``, H at
    ``(n + 1/2) dt``) and ``backward(cotangents)`` takes ``dL/d(sample)`` in
    the same layout.
    """

    def __init__(self, sim, grids, objective, device="cpu"):
        self.grids = _grids(grids)
        self.objective = dict(objective)
        spec = {"sim": _base(sim, self.grids, keep_sources=True), "grids": _grid_specs(self.grids), "objective": self.objective}
        self._s = _core.ReverseSession(json.dumps(spec), device)
        self.device = self._s.device
        self.steps = self._s.steps
        self.dt = self._s.dt

    def set_design(self, eps) -> None:
        """Set the index-grid permittivities (one array per grid)."""
        self._s.set_design(_eps_list(self.grids, eps))

    def forward(self) -> float:
        """Run the simulation, recording the objective monitor and the PML
        shell on the device; the value of a :func:`flux_energy` objective
        (0 for :func:`field_samples`, whose value the caller computes)."""
        return float(self._s.forward())

    def series(self) -> dict:
        """Recorded samples of the last ``forward``:
        ``{comp: array [step, i, j, k]}``. A flux plane records several boxes
        of one component; their keys are ``comp``, ``comp_1``, ..."""
        out = {}
        for comp, origin, a in self._s.series():
            key, t = comp, 1
            while key in out:  # flux planes hold several boxes of one component
                key, t = f"{comp}_{t}", t + 1
            out[key] = np.asarray(a)
        return out

    def times(self, comp: str) -> np.ndarray:
        """Sample times of component ``comp`` of :meth:`series`."""
        n = np.arange(self.steps)
        return (n + (1.0 if comp[0] == "E" else 0.5)) * self.dt

    def backward(self, cotangents=None) -> list:
        """One reverse pass: ``dL/d(eps)`` on the samples of every design
        grid (list of arrays of shape ``grid.dims``). ``cotangents`` holds
        ``dL/d(sample)`` in the layout of :meth:`series` (a dict with its
        keys, or a list in its order); omit it for :func:`flux_energy`."""
        if cotangents is None:
            return [np.asarray(a) for a in self._s.backward()]
        if isinstance(cotangents, dict):
            cotangents = list(cotangents.values())
        return [np.asarray(a) for a in self._s.backward_with([np.ascontiguousarray(c, dtype=np.float64) for c in cotangents])]

    def stats(self) -> dict:
        """Timings and sizes of the last forward run and reverse pass
        (``forward_seconds``, ``reverse_seconds``, ``steps``, ``cells``,
        ``shell_floats_per_step``, ...)."""
        return dict(self._s.stats())


def value_and_grad(sim, grids, eps, objective, device="cpu", fn=None, session=None):
    """Objective value and its gradient with respect to the index-grid
    permittivities by reverse-mode differentiation through the time loop.

    ``objective`` is :func:`flux_energy` (built in) or :func:`field_samples`
    with ``fn(series) -> (L, cotangents)``, where ``series`` is the dict of
    :meth:`ReverseSession.series` and ``cotangents`` a dict with the same
    keys holding ``dL/d(sample)``.

    Returns ``(L, grads, info)``: ``grads`` is a list with one array per grid
    (a single array when one grid was given as such), ``info`` holds the
    timings and the session (pass it back as ``session=`` to reuse the
    device buffers for the next design).
    """
    t0 = time.time()
    s = session or ReverseSession(sim, grids, objective, device)
    s.set_design(eps)
    val = s.forward()
    if s.objective["type"] == "fields":
        if fn is None:
            raise ValueError("a field_samples objective needs fn(series) -> (L, cotangents)")
        series = s.series()
        val, cot = fn(series)
        grads = s.backward([cot[k] for k in series])
    else:
        grads = s.backward()
    info = dict(s.stats())
    info["wall"] = time.time() - t0
    info["session"] = s
    single = not isinstance(grids, (list, tuple))
    return float(val), (grads[0] if single else grads), info
