"""Multi-port, multi-frequency S-matrix with its vector-Jacobian product.

``S[m, n, k]`` is the complex amplitude of output channel ``m`` (a waveguide
mode leaving the device through a port) per unit excitation ``n`` (a mode
launched into a port, or any lightsim source) at frequency ``k``.

One *forward* run per excitation records the field in the design region at
all K frequencies and the mode amplitudes at every output port. One *adjoint*
run per output channel launches that mode backwards into the device with a
broadband pulse and records the same design-region field. Lorentz reciprocity
(see :mod:`lightsim.adjoint`) then gives every entry of the Jacobian,

    dS[m, n, k] / d eps_c(r) = (i w_k eps0 / 4) E_adj,m(r) E_fwd,n(r) dV_c(r),

for each Yee E component ``c``, with ``E_fwd,n`` per unit excitation and
``E_adj,m`` per unit launched amplitude. N + M simulations therefore give the
gradient of *any* real function L(S) with respect to every design voxel.

Complex-gradient convention
---------------------------
Everywhere in :mod:`lightsim.invdes` the upstream gradient of a real loss L
with respect to a complex S entry is

    g = dL/dRe(S) + i dL/dIm(S)          (so that dL = Re(conj(g) dS)),

i.e. twice the conjugate Wirtinger derivative. This is what
``torch.Tensor.grad`` holds for a complex leaf; JAX cotangents are the complex
conjugate of it. The wrappers in ``lightsim.invdes.torch`` and
``lightsim.invdes.jax`` convert, so both frameworks see their own convention.

The design permittivity enters the simulation as :func:`lightsim.IndexGrid`
structures (one per :class:`DesignGrid`), appended after all the structures
of the base simulation. :func:`grid_weights` reproduces the rasteriser's
trilinear sampling of an index grid at the Yee positions and its transpose
maps the per-voxel gradient back onto the grid samples exactly.

Engines
-------
``SMatrixProblem(..., engine="python")`` (default) drives every run from
Python: each one is a fresh simulation, the design-region fields come back to
the host and the product and the transposed sampling are NumPy.
``engine="native"`` uses :class:`lightsim.autodiff.GradientSession` instead:
the scene is rasterised and the sources and monitors compiled once, a design
update uploads only the design box, the fields stay on the device, where the
product is formed, and the transposed sampling is the rasteriser's own
stencil. Same S, same normalisation, same gradient convention; the public
API (``solve``, ``SMatrixSolution.S / vjp / yee_vjp / jacobian``) is
unchanged. ``engine="reverse"`` uses :class:`lightsim.autodiff.ReverseSMatrix`:
the same S, and ``vjp`` by one reverse pass through the time loop per input,
the exact derivative of the discrete S (the other two use the continuous
adjoint formula, about 1 % off, a few per cent on boundary parameters); it
costs about four times the time and memory of ``"native"`` and has no
``yee_vjp``. The environment variable ``LIGHTSIM_INVDES_ENGINE`` sets the
default.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .. import _core
from ..adjoint import _positions, _volumes, amplitude_sensitivity
from ..spec import DFTMonitor, GaussianPulse, IndexGrid, Medium, ModeMonitor, ModeSource, Simulation, Structure

# design-region fields are kept in the precision of the core (complex64, or
# complex128 in a double-precision build)
_COMPLEX = np.complex128 if _core.precision == "f64" else np.complex64

__all__ = ["Port", "Excitation", "DesignGrid", "SMatrixProblem", "SMatrixSolution", "NativeSolution", "grid_weights"]

E_COMPONENTS = ("Ex", "Ey", "Ez")


@dataclass
class Port:
    """A waveguide port: a mode plane and the direction pointing *into* the
    device (``"+"`` or ``"-"`` along ``axis``). ``profiles`` are the mode
    profile dicts (``Mode.profile``) of the modes used at this port."""

    name: str
    axis: int
    position: float
    min: list
    max: list
    profiles: list
    direction: str

    def __post_init__(self):
        if isinstance(self.profiles, dict):
            self.profiles = [self.profiles]
        if self.direction not in ("+", "-"):
            raise ValueError("direction must be '+' or '-'")


@dataclass
class Excitation:
    """A forward excitation that is not a port mode (a beam, a plane wave, a
    dipole...). ``source`` is any lightsim source dict. S is normalised by the
    spectrum of the source signal times ``norm`` (one complex or real number
    per frequency; e.g. the square root of the incident power per unit
    spectrum, so that ``|S|^2`` is a coupling efficiency)."""

    name: str
    source: dict
    norm: object = None


@dataclass
class DesignGrid:
    """One index grid of the design: samples ``dims = (nx, ny, nz)`` spanning
    the box ``[min, max]`` (see :func:`lightsim.IndexGrid`); a 2-tuple means
    one sample along z (a layer extruded through ``[zmin, zmax)``).
    ``medium`` is the medium dict that carries the grid (its ``eps`` should be
    the largest permittivity the grid can take: the mesher uses it).

    ``permeability=True`` makes the samples relative permeabilities: the grid
    then owns H voxels, its E voxels take the medium's ``eps``, and the
    medium's ``mu`` must be the largest permeability the grid takes (above 1).
    Permeability grids are differentiated by reverse mode
    (:mod:`lightsim.autodiff`), not by the S-matrix paths."""

    min: list
    max: list
    dims: tuple
    medium: dict
    name: str = "design"
    permeability: bool = False

    def __post_init__(self):
        d = tuple(int(v) for v in self.dims)
        self.dims = d + (1,) * (3 - len(d))
        self.min = [float(v) for v in self.min]
        self.max = [float(v) for v in self.max]


def _own_tol(sim_grid) -> float:
    """Ownership tolerance of the rasteriser: a face within a millionth of the
    smallest cell of a field position lies on it (lower face inside, upper
    face outside), whatever the rounding of the coordinates says."""
    return 1e-6 * min(float(np.min(np.diff(np.asarray(n, dtype=float)))) for n in sim_grid)


def _axis_weights(p, lo, hi, n, tol=0.0):
    """Rows: Yee positions ``p`` along one axis; columns: the ``n`` grid
    samples. Same arithmetic as ``Shape::grid_eps`` (clamped linear
    interpolation) restricted to the positions the grid owns, ``[lo, hi)``
    with the rasteriser's tolerance ``tol`` (:func:`_own_tol`)."""
    p = np.asarray(p, dtype=float).reshape(-1)
    owned = (p >= lo - tol) & (p < hi - tol)
    w = np.zeros((len(p), n))
    if n == 1:
        w[:, 0] = owned
        return w
    t = np.clip((p - lo) / (hi - lo), 0.0, 1.0) * (n - 1)
    f = np.minimum(np.floor(t).astype(int), n - 2)
    u = t - f
    rows = np.arange(len(p))
    w[rows, f] = np.where(owned, 1.0 - u, 0.0)
    w[rows, f + 1] = np.where(owned, u, 0.0)
    return w


def grid_weights(comp, sim_grid, origin, shape, grid: DesignGrid):
    """Separable sampling matrices (Wx, Wy, Wz) of an index grid for the Yee
    positions of component ``comp`` in the index box ``origin``/``shape``:

        eps_yee[i, j, k] = sum_abc Wx[i, a] Wy[j, b] Wz[k, c] eps_grid[a, b, c]

    on the voxels the grid owns (rows of the others are zero). The transpose
    is the exact adjoint of the IndexGrid rasterisation."""
    pos = _positions(comp, sim_grid, origin, shape)
    tol = _own_tol(sim_grid)
    return [_axis_weights(pos[a], grid.min[a], grid.max[a], grid.dims[a], tol) for a in range(3)]


def _apply(ws, eps):
    return np.einsum("ia,jb,kc,abc->ijk", ws[0], ws[1], ws[2], eps, optimize=True)


def _apply_t(ws, g):
    return np.einsum("ia,jb,kc,ijk->abc", ws[0], ws[1], ws[2], g, optimize=True)


def _owned_range(comp, sim_grid, grid: DesignGrid):
    """Index range [lo, hi) per axis of the Yee positions the grid owns."""
    axis = "xyz".index(comp[1].lower())
    tol = _own_tol(sim_grid)
    out = []
    for a in range(3):
        nodes = np.asarray(sim_grid[a])
        p = 0.5 * (nodes[:-1] + nodes[1:]) if a == axis else nodes[:-1]  # one sample per cell
        idx = np.nonzero((p >= grid.min[a] - tol) & (p < grid.max[a] - tol))[0]
        out.append((int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0))
    return out


@dataclass
class SMatrixSolution:
    """Result of :meth:`SMatrixProblem.solve`."""

    S: np.ndarray  #: complex [M, N, K]
    freqs: np.ndarray
    stats: dict
    _problem: object = None
    _fwd: list = None  # per excitation: list over grids of {comp: complex [K, i, j, k]}
    _adj: list = None  # per output channel: same
    fields: list = None  #: per excitation: {monitor name: {component: complex array, "origin": {...}}}

    def yee_vjp(self, g):
        """dL/d(eps_r) on the Yee voxels of every design-grid monitor box for
        the upstream gradient ``g`` (see the module docstring):
        list over grids of ``{comp: (origin, real array [i, j, k])}``."""
        if self._adj is None:
            raise RuntimeError("solve(..., adjoint=True) is needed for gradients")
        p = self._problem
        g = np.asarray(g, dtype=complex)
        if g.shape != self.S.shape:
            raise ValueError(f"upstream gradient has shape {g.shape}, S has {self.S.shape}")
        M, N, K = g.shape
        out = []
        for gi in range(len(p.grids)):
            d = {}
            for c in p.components:
                meta = p._meta[gi][c]
                acc = np.zeros(meta["shape"])
                for k in range(K):
                    tot = 0
                    for m in range(M):
                        if not np.any(g[m, :, k]):
                            continue
                        b = 0
                        for n in range(N):
                            if g[m, n, k] != 0:
                                b = b + np.conj(g[m, n, k]) * self._fwd[n][gi][c][k]
                        tot = tot + self._adj[m][gi][c][k] * b
                    if np.ndim(tot):
                        acc += np.real(amplitude_sensitivity(tot, 1.0, 1.0, self.freqs[k]))
                d[c] = (meta["origin"], acc * meta["vol"])
            out.append(d)
        return out

    def vjp(self, g):
        """dL/d(eps) on the samples of every design grid (list of arrays of
        shape ``grid.dims``) for the upstream gradient ``g``."""
        p = self._problem
        yee = self.yee_vjp(g)
        out = []
        for gi, grid in enumerate(p.grids):
            tot = np.zeros(grid.dims)
            for c in p.components:
                tot += _apply_t(p._meta[gi][c]["w"], yee[gi][c][1])
            out.append(tot)
        return out

    def jacobian(self, m, n, k):
        """Complex dS[m, n, k]/d(eps) on the samples of every design grid."""
        p = self._problem
        out = []
        for gi, grid in enumerate(p.grids):
            tot = np.zeros(grid.dims, dtype=complex)
            for c in p.components:
                meta = p._meta[gi][c]
                j = amplitude_sensitivity(self._fwd[n][gi][c][k], self._adj[m][gi][c][k], meta["vol"], self.freqs[k])
                tot += _apply_t(meta["w"], j.real) + 1j * _apply_t(meta["w"], j.imag)
            out.append(tot)
        return out


class NativeSolution:
    """Result of :meth:`SMatrixProblem.solve` with ``engine="native"``: the
    same interface as :class:`SMatrixSolution`, backed by the fields the
    :class:`lightsim.autodiff.GradientSession` keeps on the device.

    The session holds one design at a time. A solution used after the
    problem has been solved for another design stays valid: its design is
    put back and its runs are repeated first (once), which costs one more
    set of simulations; the usual loop (solve, back-propagate, next design)
    never does."""

    def __init__(self, problem, S, stats, token, eps):
        self.S = S
        self.freqs = problem.freqs
        self.stats = stats
        self._problem = problem
        self._token = token
        self._eps = [np.array(e, dtype=float) for e in eps]

    def _session(self):
        p = self._problem
        sess = p.native_session()
        if p._native_token != self._token:
            sess.set_design(self._eps)
            sess.forward(adjoint=True)
            p._native_token += 1
            self._token = p._native_token
        return sess

    def _check(self, g):
        g = np.asarray(g, dtype=complex)
        if g.shape != self.S.shape:
            raise ValueError(f"upstream gradient has shape {g.shape}, S has {self.S.shape}")
        return g

    def yee_vjp(self, g):
        """dL/d(eps_r) on the Yee voxels of every design-grid monitor box:
        list over grids of ``{comp: (origin, real array [i, j, k])}``."""
        return self._session().backward_yee(self._check(g))

    def vjp(self, g):
        """dL/d(eps) on the samples of every design grid (list of arrays of
        shape ``grid.dims``) for the upstream gradient ``g``."""
        return self._session().backward(self._check(g))

    def jacobian(self, m, n, k):
        """Complex dS[m, n, k]/d(eps) on the samples of every design grid."""
        g = np.zeros(self.S.shape, dtype=complex)
        g[m, n, k] = 1.0
        re = self._session().backward(g)  # L = Re S
        im = self._session().backward(1j * g)  # L = Im S
        return [a + 1j * b for a, b in zip(re, im)]


class ReverseSolution(NativeSolution):
    """Result of :meth:`SMatrixProblem.solve` with ``engine="reverse"``: the
    S of the other engines; :meth:`vjp` runs one reverse pass through the
    time loop per input (the exact derivative of the discrete scheme). As
    for :class:`NativeSolution`, a solution used after another design was
    solved puts its design back and repeats its forward runs first."""

    def _session(self):
        p = self._problem
        rs = p.reverse_smatrix()
        if p._reverse_token != self._token:
            rs.set_design(self._eps)
            rs.forward()
            p._reverse_token += 1
            self._token = p._reverse_token
        return rs

    def yee_vjp(self, g):
        raise NotImplementedError("per-voxel gradients are not available with engine='reverse'; use vjp, or engine='native' / 'python'")


class SMatrixProblem:
    """S-matrix of a device whose design region is a set of index grids.

    Parameters
    ----------
    sim : the base :class:`lightsim.Simulation` (fixed structures, mesh,
        boundaries, run time). Its sources and monitors are ignored.
    grids : list of :class:`DesignGrid` (or one). They are appended after the
        structures of ``sim``, so the design covers what lies under it; no
        structure of ``sim`` may be meant to cover the design.
    inputs : excitations: :class:`Port` (mode 0), ``(Port, mode_index)`` or
        :class:`Excitation`.
    outputs : output channels: :class:`Port` or ``(Port, mode_index)``.
    freqs : the K frequencies (Hz).
    devices : a device string or a list of them. The N + M runs of one
        :meth:`solve` are independent and are dispatched concurrently, one at
        a time per list entry (``["cuda:0", "cuda:1"]``; with
        ``engine="python"`` an entry may itself be a multi-GPU string such as
        ``"cuda:0,1"``, which splits each run over those GPUs and gives the
        one-GPU S and gradient bit for bit; the native engine runs each
        simulation on one device).
    fwidth : bandwidth of the Gaussian pulses (default: the span of
        ``freqs``, at least 10 % of the centre frequency).
    monitors : extra DFT field monitors (``lightsim.DFTMonitor`` dicts) added
        to every forward run (recorded at ``freqs``); their fields, per unit excitation, are returned
        in ``SMatrixSolution.fields[n][name][component]`` (field maps of the
        final design: by linearity any input vector is a weighted sum).
    normalize : ``"monitor"`` (default) measures the amplitude launched by
        every mode source with two mode monitors just downstream and refers
        it to the source plane, so S is exactly "per unit incident mode
        amplitude"; ``"source"`` divides by the source signal spectrum only.
    engine : ``"reverse"`` (one forward run and one reverse pass through the
        time loop per input: the exact derivative of the discrete scheme; see
        :class:`lightsim.autodiff.ReverseSMatrix`),
        ``"python"`` (every run driven from Python, products in NumPy)
        or ``"native"`` (the persistent Rust session of
        :mod:`lightsim.autodiff`: buffers kept on the device across
        ``solve`` calls, adjoint product and transposed rasterisation in the
        engine). Default: ``$LIGHTSIM_INVDES_ENGINE`` or ``"python"``.
    """

    def __init__(self, sim, grids, inputs, outputs, freqs, devices="cpu", fwidth=None, normalize="monitor", components=E_COMPONENTS, progress=0, monitors=(), engine=None):
        self.engine = engine or os.environ.get("LIGHTSIM_INVDES_ENGINE", "python")
        if self.engine not in ("python", "native", "reverse"):
            raise ValueError("engine must be 'python', 'native' or 'reverse'")
        if self.engine in ("native", "reverse") and monitors:
            if engine in ("native", "reverse"):
                raise ValueError("extra field monitors need engine='python'")
            self.engine = "python"  # selected by the environment only: field maps run on the Python path
        self._native = None
        self._native_token = 0
        self._reverse = None
        self._reverse_token = 0
        from ..adjoint import check_gradient_media

        check_gradient_media(sim)
        self.sim = sim
        self.monitors = list(monitors)
        self.grids = [grids] if isinstance(grids, DesignGrid) else list(grids)
        if any(getattr(g, "permeability", False) for g in self.grids):
            raise ValueError("permeability design grids are differentiated by reverse mode (lightsim.autodiff), not by the S-matrix problem")
        self.inputs = [self._channel(i, True) for i in inputs]
        self.outputs = [self._channel(o, False) for o in outputs]
        self._unit_power_ports()
        self.freqs = np.asarray(freqs, dtype=float).reshape(-1)
        self.devices = [devices] if isinstance(devices, str) else list(devices)
        f0 = 0.5 * (self.freqs.min() + self.freqs.max())
        self.signal = GaussianPulse(f0, float(fwidth) if fwidth else max(float(np.ptp(self.freqs)), 0.1 * f0))
        if normalize not in ("monitor", "source"):
            raise ValueError("normalize must be 'monitor' or 'source'")
        self.normalize = normalize
        self.components = tuple(components)
        self.progress = progress
        self._sim_grid = None
        self._meta = None

    def _unit_power_ports(self):
        """Scale the port profiles to 1 W as the mode overlap measures the
        power (``lightsim._core.port_power``), as the native session does:
        S is normalised with that overlap, and the adjoint formula needs the
        profiles to carry unit power in the same measure. The ports given by
        the caller are left as they are (copies are scaled)."""
        import json

        from .. import _core

        spec = self.sim.to_json()
        new = {}

        def scaled(port):
            if id(port) not in new:
                d = {"name": port.name, "axis": int(port.axis), "position": float(port.position), "min": [float(v) for v in port.min], "max": [float(v) for v in port.max], "profiles": list(port.profiles), "direction": port.direction}
                profs = []
                for prof, pw in zip(port.profiles, _core.port_power(spec, json.dumps(d))):
                    if not np.isfinite(pw) or pw == 0.0:
                        raise ValueError(f"port '{port.name}': a mode profile carries no power through the port window")
                    f = 1.0 / np.sqrt(abs(pw))
                    q = dict(prof)
                    for key in ("e_re", "e_im", "h_re", "h_im"):
                        if key in q:
                            q[key] = [(np.asarray(a, dtype=float) * f).tolist() for a in q[key]]
                    profs.append(q)
                new[id(port)] = dataclasses.replace(port, profiles=profs)
            return new[id(port)]

        self.inputs = [(scaled(c[0]), c[1]) if isinstance(c, tuple) else c for c in self.inputs]
        self.outputs = [(scaled(c[0]), c[1]) for c in self.outputs]

    # -- description ---------------------------------------------------
    @staticmethod
    def _channel(c, allow_excitation):
        if isinstance(c, Port):
            return (c, 0)
        if isinstance(c, Excitation):
            if not allow_excitation:
                raise ValueError("outputs must be port modes")
            return c
        port, mode = c
        return (port, int(mode))

    @property
    def shape(self):
        """``(M, N, K)``: the numbers of outputs, inputs and frequencies, the
        shape of the S-matrix that :meth:`solve` returns."""
        return (len(self.outputs), len(self.inputs), len(self.freqs))

    def simulation(self, eps):
        """The simulation (no sources, no monitors) with the design grids
        holding the permittivity arrays ``eps`` (one per grid)."""
        eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
        if len(eps) != len(self.grids):
            raise ValueError(f"expected {len(self.grids)} permittivity arrays")
        s = Simulation(**{k: v for k, v in self.sim.__dict__.items()})
        s.media = list(self.sim.media)
        s.structures = list(self.sim.structures)
        s.sources, s.monitors = [], []
        for g, e in zip(self.grids, eps):
            e = np.asarray(e, dtype=float).reshape(g.dims)
            if not any(m["name"] == g.medium["name"] for m in s.media):
                s.media.append(g.medium)
            s.structures.append(Structure(IndexGrid(g.min, g.max, e), g.medium["name"], name=g.name))
        return s

    def _grid(self):
        if self._sim_grid is None:
            self._sim_grid = [np.asarray(a, dtype=float) for a in self.sim.grid()]
        return self._sim_grid

    def _monitor_box(self, g):
        """Design box grown by more than a cell so every owned Yee position
        is recorded whatever the monitor's inclusion rule."""
        sg = self._grid()
        lo, hi = [], []
        for a in range(3):
            n = sg[a]
            i0 = int(np.clip(np.searchsorted(n, g.min[a]), 2, len(n) - 1))
            i1 = int(np.clip(np.searchsorted(n, g.max[a]), 0, len(n) - 3))
            lo.append(max(float(n[i0 - 2]), float(n[0])))
            hi.append(min(float(n[i1 + 2]), float(n[-1])))
        return lo, hi

    def _norm_planes(self, port):
        """Two planes just downstream of a mode source (2 and 4 cells)."""
        n = self._grid()[port.axis]
        i0 = int(np.argmin(np.abs(n - port.position)))
        s = 1 if port.direction == "+" else -1
        out = []
        for d in (2, 4):
            x = float(n[i0 + s * d])
            mn, mx = list(port.min), list(port.max)
            mn[port.axis] = mx[port.axis] = x
            out.append((x, mn, mx))
        return out

    # -- one run -------------------------------------------------------
    def _run(self, kind, index, eps, device):
        t0 = time.time()
        s = self.simulation(eps)
        ch = self.inputs[index] if kind == "fwd" else self.outputs[index]
        src_port = None
        if isinstance(ch, Excitation):
            src = dict(ch.source)
            src["name"] = "__src__"
        else:
            src_port, mode = ch
            src = ModeSource(src_port.axis, src_port.position, src_port.direction, src_port.min, src_port.max, src_port.profiles[mode], self.signal, name="__src__")
        s.add_source(src)
        for gi, g in enumerate(self.grids):
            lo, hi = self._monitor_box(g)
            s.add_monitor(DFTMonitor(f"__des{gi}__", lo, hi, self.freqs, components=self.components))
        if src_port is not None:
            for t, (x, mn, mx) in enumerate(self._norm_planes(src_port)):
                s.add_monitor(ModeMonitor(f"__norm{t}__", src_port.axis, x, mn, mx, self.freqs, src_port.profiles))
        ports = []
        if kind == "fwd":
            for mon in self.monitors:
                s.add_monitor({**mon, "freqs": [float(f) for f in self.freqs]})
            for port, _ in self.outputs:
                if port is not src_port and all(port is not q for q in ports):
                    ports.append(port)
                    s.add_monitor(ModeMonitor(f"__out_{len(ports) - 1}__", port.axis, port.position, port.min, port.max, self.freqs, port.profiles))
        r = s.run(device=device, progress=self.progress)
        # ---- normalisation ----
        # the spectrum of the source at the frequencies of the problem, in their order
        _, spec = r.source_spectrum("__src__", freqs=self.freqs)
        spec = np.asarray(spec)
        q = None
        if src_port is not None and self.normalize == "monitor":
            a1 = self._amp(r, "__norm0__", mode, src_port.direction)
            a2 = self._amp(r, "__norm1__", mode, src_port.direction)
            q = a1 / a2  # one-step phase factor of the launched mode over 2 cells
            norm = a1 * q  # referred back to the source plane
        else:
            norm = np.asarray(spec, dtype=complex)
            if isinstance(ch, Excitation) and ch.norm is not None:
                norm = norm * np.asarray(ch.norm)
        fields = []
        for gi in range(len(self.grids)):
            d = {}
            for c in self.components:
                f = r.raw["dft"][f"__des{gi}__"]["fields"][c]
                if self._meta_pending:
                    self._origins[(gi, c)] = (list(f["origin"]), tuple(np.shape(f["re"])[1:]))
                d[c] = ((np.asarray(f["re"]) + 1j * np.asarray(f["im"])) / norm[:, None, None, None]).astype(_COMPLEX)
            fields.append(d)
        amps = None
        if kind == "fwd":
            amps = np.zeros((len(self.outputs), len(self.freqs)), dtype=complex)
            back = lambda p: "-" if p.direction == "+" else "+"  # noqa: E731
            for m, (port, md) in enumerate(self.outputs):
                if port is src_port:
                    # reflection: measured 2 cells inside, referred back to the port plane
                    a = self._amp(r, "__norm0__", md, back(port))
                    amps[m] = a / (q if q is not None else 1.0) / norm
                else:
                    pi = next(i for i, p_ in enumerate(ports) if p_ is port)
                    amps[m] = self._amp(r, f"__out_{pi}__", md, back(port)) / norm
        st = dict(r.stats)
        st["wall"] = time.time() - t0
        st["kind"], st["index"], st["device"] = kind, index, device
        extra = {}
        if kind == "fwd":
            for mon in self.monitors:
                fs = r.raw["dft"][mon["name"]]["fields"]
                nd = lambda a: norm.reshape((-1,) + (1,) * (np.ndim(a) - 1))  # noqa: E731
                extra[mon["name"]] = {c: ((np.asarray(f["re"]) + 1j * np.asarray(f["im"])) / nd(f["re"])).astype(_COMPLEX) for c, f in fs.items()}
                extra[mon["name"]]["origin"] = {c: list(f["origin"]) for c, f in fs.items()}
        return fields, amps, st, extra

    @staticmethod
    def _amp(r, name, mode, direction):
        ap, am = r.mode_amplitudes(name, mode)
        return np.asarray(ap if direction == "+" else am, dtype=complex)

    # -- solve ---------------------------------------------------------
    def solve(self, eps, adjoint=True) -> SMatrixSolution:
        """Run the N forward (and, with ``adjoint``, the M adjoint)
        simulations for the grid permittivities ``eps`` (one array per
        grid)."""
        eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
        if self.engine == "native":
            return self._solve_native(eps, adjoint)
        if self.engine == "reverse":
            return self._solve_reverse(eps)
        t0 = time.time()
        self._meta_pending = self._meta is None
        self._origins = {}
        jobs = [("fwd", n) for n in range(len(self.inputs))]
        if adjoint:
            jobs += [("adj", m) for m in range(len(self.outputs))]
        devq = queue.Queue()
        for d in self.devices:
            devq.put(d)

        def work(job):
            dev = devq.get()
            try:
                return self._run(job[0], job[1], eps, dev)
            finally:
                devq.put(dev)

        if len(self.devices) == 1:
            res = [work(j) for j in jobs]
        else:
            with ThreadPoolExecutor(max_workers=len(self.devices)) as ex:
                res = list(ex.map(work, jobs))
        N = len(self.inputs)
        S = np.stack([res[n][1] for n in range(N)], axis=1)  # [M, N, K]
        if self._meta_pending:
            self._build_meta()
        runs = [r[2] for r in res]
        stats = {
            "wall": time.time() - t0,
            "runs": runs,
            "forward_runs": N,
            "adjoint_runs": len(jobs) - N,
            "cells": runs[0]["cells"],
            "steps": [r["steps"] for r in runs],
            "sim_seconds": float(sum(r["seconds"] for r in runs)),
        }
        return SMatrixSolution(
            S=S,
            freqs=self.freqs,
            stats=stats,
            _problem=self,
            _fwd=[res[n][0] for n in range(N)],
            _adj=[res[N + m][0] for m in range(len(jobs) - N)] if adjoint else None,
            fields=[res[n][3] for n in range(N)],
        )

    def native_session(self):
        """The :class:`lightsim.autodiff.GradientSession` behind
        ``engine="native"`` (built on first use)."""
        if self._native is None:
            from ..autodiff import GradientSession

            self._native = GradientSession(
                self.sim,
                self.grids,
                [c if isinstance(c, Excitation) else tuple(c) for c in self.inputs],
                [tuple(c) for c in self.outputs],
                self.freqs,
                devices=self.devices,
                normalize=self.normalize,
                components=self.components,
                signal=self.signal,
            )
        return self._native

    def reverse_smatrix(self):
        """The :class:`lightsim.autodiff.ReverseSMatrix` behind
        ``engine="reverse"`` (built on first use)."""
        if self._reverse is None:
            from ..autodiff import ReverseSMatrix

            self._reverse = ReverseSMatrix(
                self.sim,
                self.grids,
                [c if isinstance(c, Excitation) else tuple(c) for c in self.inputs],
                [tuple(c) for c in self.outputs],
                self.freqs,
                devices=self.devices,
                normalize=self.normalize,
                components=self.components,
                signal=self.signal,
            )
        return self._reverse

    def _solve_reverse(self, eps):
        t0 = time.time()
        rs = self.reverse_smatrix()
        rs.set_design(eps)
        S = rs.forward()
        self._reverse_token += 1
        runs = [dict(r, kind="fwd", index=r["input"]) for r in rs.stats()]
        stats = {
            "wall": time.time() - t0,
            "runs": runs,
            "forward_runs": len(runs),
            "adjoint_runs": 0,
            "steps": [r["steps"] for r in runs],
            "sim_seconds": float(sum(r["forward_seconds"] for r in runs)),
            "engine": "reverse",
        }
        return ReverseSolution(self, S, stats, self._reverse_token, eps)

    def _solve_native(self, eps, adjoint):
        t0 = time.time()
        sess = self.native_session()
        sess.set_design(eps)
        S = sess.forward(adjoint=adjoint)
        self._native_token += 1
        runs = [dict(r, cells=sess.cells) for r in sess.stats()]
        stats = {
            "wall": time.time() - t0,
            "runs": runs,
            "forward_runs": sum(r["kind"] == "fwd" for r in runs),
            "adjoint_runs": sum(r["kind"] == "adj" for r in runs),
            "cells": sess.cells,
            "steps": [r["steps"] for r in runs],
            "sim_seconds": float(sum(r["seconds"] for r in runs)),
            "engine": "native",
        }
        return NativeSolution(self, S, stats, self._native_token, eps)

    def _build_meta(self):
        sg = self._grid()
        periodic = tuple(self.sim.boundaries[a][0].get("type") in ("periodic", "bloch") for a in "xyz")
        meta = []
        for gi, g in enumerate(self.grids):
            d = {}
            for c in self.components:
                origin, shape = self._origins[(gi, c)]
                for a, (lo, hi) in enumerate(_owned_range(c, sg, g)):
                    if hi > lo and (lo < origin[a] or hi > origin[a] + shape[a]):
                        raise RuntimeError(f"design monitor does not cover the grid '{g.name}' along axis {a} for {c}")
                d[c] = {
                    "origin": origin,
                    "shape": shape,
                    "vol": _volumes(c, sg, origin, shape, periodic),
                    "w": grid_weights(c, sg, origin, shape, g),
                }
            meta.append(d)
        self._meta = meta

    def yee_permittivity(self, eps):
        """What the solver uses: the permittivity at the Yee positions of
        every design monitor box interpolated from the grids (zero where the
        grid does not own the voxel). Needs one earlier :meth:`solve` with
        ``engine="python"``; the native session reports the same through
        ``native_session().yee_permittivity()``."""
        eps = [eps] if isinstance(eps, np.ndarray) else list(eps)
        if self._meta is None:
            raise RuntimeError("yee_permittivity needs one earlier solve with engine='python'")
        return [{c: _apply(self._meta[gi][c]["w"], np.asarray(e, dtype=float).reshape(g.dims)) for c in self.components} for gi, (g, e) in enumerate(zip(self.grids, eps))]
