"""The :class:`Simulation` and the completion of 2-D specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ._boundaries import PERIODIC, PML
from ._geometry import Structure
from ._results import Results
from ._util import _axis, _lift, Vec3


def _complete(obj, w: int, lo: float, hi: float):
    """Copy of a source / monitor / shape / mesh dict of a 2-D simulation
    with every 2-component vector completed along the invariant axis ``w``
    (the slab is ``[lo, hi]``): box corners span the slab, points sit on its
    mid-plane, cylinders and polygons without an axis are normal to the
    plane."""
    if not isinstance(obj, dict):
        return obj
    mid = 0.5 * (lo + hi)
    out = dict(obj)
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _complete(v, w, lo, hi)
        elif k == "overrides" and isinstance(v, list):
            out[k] = [_complete(o, w, lo, hi) for o in v]
        elif k in ("min", "max", "position", "center", "dx") and isinstance(v, (list, tuple)) and len(v) == 2:
            out[k] = _lift(v, w, {"min": lo, "max": hi, "dx": hi - lo}.get(k, mid))
    kind = obj.get("type")
    if kind == "cylinder":
        if out.get("axis") is None:
            out["axis"] = w
        if out.get("length") is None:
            out["length"] = 4.0 * (hi - lo)
    elif kind == "poly_slab":
        if out.get("axis") is None:
            out["axis"] = w
        if out.get("slab_min") is None:
            out["slab_min"] = lo - (hi - lo)
        if out.get("slab_max") is None:
            out["slab_max"] = hi + (hi - lo)
    return out


def _complete_3d(obj):
    """3-D simulations: defaults of the shapes that a 2-D one would infer,
    and a clear error for 2-component vectors."""
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _complete_3d(v)
        elif k in ("min", "max", "position", "center") and isinstance(v, (list, tuple)) and len(v) == 2:
            raise ValueError(f"{k}={list(v)} has 2 components: give 3, or make the simulation 2-D (Simulation(..., dimensions=2))")
    kind = obj.get("type")
    if kind in ("cylinder", "poly_slab") and out.get("axis") is None:
        out["axis"] = 2
    if kind == "cylinder" and out.get("length") is None:
        raise ValueError("Cylinder needs a length in a 3-D simulation")
    if kind == "poly_slab" and (out.get("slab_min") is None or out.get("slab_max") is None):
        raise ValueError("PolySlab needs slab_min and slab_max in a 3-D simulation")
    return out


@dataclass
class Simulation:
    """A lightsim simulation. Build it, then call :meth:`run`.

    **3-D** (default): ``min`` / ``max`` are the corners of the region.

    **2-D** (``dimensions=2``, or ``min`` / ``max`` with two components): the
    fields do not depend on the coordinate along ``invariant_axis`` ("x",
    "y" or "z", default "z"). The grid has one cell along that axis, the
    boundaries there are periodic, the time step follows the 2-D Courant
    limit, and the solver updates three field components per polarization
    instead of six:

    * ``polarization="te"``: E in the plane, H along the invariant axis --
      (Ex, Ey, Hz) for the invariant axis z;
    * ``polarization="tm"``: E along the invariant axis, H in the plane --
      (Ez, Hx, Hy) for z;
    * ``polarization="both"`` (default): the two sets, which are independent;
      only the sets that a source drives are updated.

    (The naming of Lumerical's 2-D FDTD, Meep and Taflove's TE_z / TM_z. In
    a top view of a chip, z-invariant, "te" holds the quasi-TE waveguide
    modes; in a side view of a slab, e.g. y-invariant, a slab TE mode has E
    along y and belongs to "tm".)

    The region's extent along the invariant axis is the thickness of the
    slab the 2-D model stands for: with two-component ``min`` / ``max`` it is
    ``thickness`` (default 1 m, centred on 0), so fluxes and dipole powers
    are per metre. Structures are cut by the mid-plane of the slab, whatever
    their extent along the invariant axis, and sources and monitors span it.
    Inside a 2-D simulation every point or box corner may be given by its two
    in-plane components.
    """

    min: Vec3
    max: Vec3
    mesh: dict
    run_time: float
    boundaries: dict | None = None
    media: list[dict] = field(default_factory=list)
    structures: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    monitors: list[dict] = field(default_factory=list)
    background: str = "vacuum"
    shutoff: float | None = None
    shutoff_interval: int = 200
    courant: float = 0.99
    subpixel: bool = True
    name: str = ""
    dft_stride: int | None = None
    dimensions: int = 3
    invariant_axis: int | str = "z"
    polarization: str = "both"
    thickness: float = 1.0
    force_3d_kernels: bool = False
    sheets: list[dict] = field(default_factory=list)
    subpixel_weighting: str = "cell"

    def __post_init__(self):
        self.min, self.max = list(self.min), list(self.max)
        if len(self.min) != len(self.max) or len(self.min) not in (2, 3):
            raise ValueError("min and max need 3 components each (or 2 each for a 2-D simulation)")
        if len(self.min) == 2:
            self.dimensions = 2
        if self.subpixel_weighting not in ("cell", "tent"):
            raise ValueError("subpixel_weighting must be 'cell' or 'tent'")
        if self.dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3")
        self.invariant_axis = _axis(self.invariant_axis)
        self.polarization = str(self.polarization).lower()
        if self.polarization not in ("te", "tm", "both"):
            raise ValueError("polarization must be 'te', 'tm' or 'both'")
        if self.dimensions == 3 and self.polarization != "both":
            raise ValueError("polarization applies to 2-D simulations: pass dimensions=2 (or 2-component min / max)")
        w = self.invariant_axis
        if len(self.min) == 2:
            half = 0.5 * float(self.thickness)
            self.min, self.max = _lift(self.min, w, -half), _lift(self.max, w, half)
        if self.boundaries is None:
            self.boundaries = {"x": [PML(), PML()], "y": [PML(), PML()], "z": [PML(), PML()]}
        if self.dimensions == 2:
            self.boundaries = dict(self.boundaries)
            self.boundaries["xyz"[w]] = [dict(PERIODIC), dict(PERIODIC)]

    @property
    def slab(self) -> tuple[int, float, float] | None:
        """2-D simulations: (invariant axis, low face, high face) of the slab
        the model stands for; ``None`` for a 3-D simulation."""
        if self.dimensions != 2:
            return None
        w = self.invariant_axis
        return w, float(self.min[w]), float(self.max[w])

    def _resolve(self, obj: dict) -> dict:
        s = self.slab
        return _complete_3d(obj) if s is None else _complete(obj, *s)

    # -- builders --
    def add_medium(self, m: dict) -> "Simulation":
        """Register a medium.

        A medium whose name is already registered is ignored. ``m`` is a dict from
        ``Medium``. Returns the simulation.
        """
        if not any(x["name"] == m["name"] for x in self.media):
            self.media.append(m)
        return self

    def add_structure(self, shape: dict, medium: str | dict, name: str = "") -> "Simulation":
        """Add a shape filled with a medium.

        ``medium`` is the name of a medium, or a ``Medium`` dict, which is registered
        as well. Structures are painted in the order they are added: where two
        overlap, the later one wins. Returns the simulation.
        """
        if isinstance(medium, dict):
            self.add_medium(medium)
            medium = medium["name"]
        self.structures.append(Structure(self._resolve(shape), medium, name))
        return self

    def add_source(self, s: dict) -> "Simulation":
        """Add a source (a dict from ``PointDipole``, ``PlaneWave``, ``GaussianBeam``,
        ``ModeSource`` or ``TFSF``). A source name may be used once. Returns the
        simulation.
        """
        name = s.get("name", "")
        if name and any(q.get("name", "") == name for q in self.sources):
            raise ValueError(f"a source named {name!r} exists already; source spectra are looked up by name")
        self.sources.append(self._resolve(s))
        return self

    def add_monitor(self, m: dict) -> "Simulation":
        """Add a monitor (a dict from one of the ``...Monitor`` builders). Every
        monitor needs its own name, whatever its type: results are looked up by
        name. Returns the simulation.
        """
        name = m.get("name", "")
        if any(q.get("name", "") == name for q in self.monitors):
            raise ValueError(f"a monitor named {name!r} exists already; results are looked up by name, so every monitor needs its own (whatever its type)")
        self.monitors.append(self._resolve(m))
        return self

    def add_sheet(self, sheet: dict) -> "Simulation":
        """Add a zero-thickness conducting sheet (a dict from :func:`Sheet`).
        Returns the simulation."""
        self.sheets.append(self._resolve(sheet))
        return self

    def set_boundaries(self, x=None, y=None, z=None) -> "Simulation":
        """Set the boundary conditions of the given axes.

        Each of ``x``, ``y``, ``z`` is one condition for both faces of that axis, or
        a pair ``[low, high]``: ``PML(...)``, ``PERIODIC``, ``Bloch(k)``,
        ``PEC_WALL`` or ``PMC_WALL``. Periodic and Bloch conditions must be set on
        both faces. Axes left at ``None`` keep their conditions. Returns the
        simulation.
        """
        for k, v in (("x", x), ("y", y), ("z", z)):
            if v is not None:
                self.boundaries[k] = list(v) if isinstance(v, (list, tuple)) else [v, v]
        if self.dimensions == 2:
            self.boundaries["xyz"[self.invariant_axis]] = [dict(PERIODIC), dict(PERIODIC)]
        return self

    # -- serialization --
    def to_dict(self) -> dict:
        """The simulation as a plain dictionary in the JSON format of the core.
        """
        r = self._resolve
        mesh = r(self.mesh)
        d = {
            "name": self.name,
            "region": {"min": [float(v) for v in self.min], "max": [float(v) for v in self.max]},
            "mesh": mesh,
            "media": self.media,
            "background": self.background,
            "structures": [dict(s, shape=r(s["shape"])) for s in self.structures],
            "sources": [{k: v for k, v in r(s).items() if k != "fixed_angle"} for s in self.sources],
            "monitors": [r(m) for m in self.monitors],
            "boundaries": self.boundaries,
            "run_time": self.run_time,
            "shutoff": self.shutoff,
            "shutoff_interval": self.shutoff_interval,
            "courant": self.courant,
            "subpixel": self.subpixel,
            "dft_stride": self.dft_stride,
        }
        if self.dimensions == 2:
            d["dimensions"] = 2
            d["invariant_axis"] = self.invariant_axis
            d["polarization"] = self.polarization
        if self.force_3d_kernels:
            d["force_3d_kernels"] = True
        if self.subpixel_weighting != "cell":
            d["subpixel_weighting"] = self.subpixel_weighting
        # a plane wave with fixed_angle=True makes the run a fixed-angle run
        # (a simulation-level switch in the JSON format of the core)
        if any(s.get("fixed_angle") for s in self.sources):
            d["fixed_angle"] = True
        if self.sheets:
            d["sheets"] = [r(q) for q in self.sheets]
        return d

    def to_json(self) -> str:
        """The simulation as a JSON string, as read by ``lightsim run``.
        """
        return json.dumps(self.to_dict())

    def geometry_json(self) -> str:
        """The simulation as JSON without its sources and monitors: all that
        the grid and the permittivity of a plane depend on. The mode solvers
        send this to the core, so that their cost does not grow with the mode
        profiles already attached to monitors and sources."""
        d = self.to_dict()
        d["sources"] = []
        d["monitors"] = []
        return json.dumps(d)

    # -- core calls --
    def grid(self):
        """Node coordinates ``(x, y, z)`` of the grid as numpy arrays (m), absorbing
        layers included. Nothing is run.
        """
        from .. import _core

        return _core.build_grid(self.to_json())

    def frozen_mesh(self) -> dict:
        """The grid of this simulation as an explicit mesh.

        Returns the mesh dict (see :func:`ExplicitMesh`) with the node
        coordinates the simulation has now, inside the region; absorbing
        layers added outside the region are not part of a mesh and come back
        when the mesh is used. An automatic mesh depends on the structures,
        so a copy of the simulation with structures removed or added gets
        another grid. Give the copy this mesh to keep the grid, for example
        to solve port modes on a single waveguide of a larger device:

            mesh = device.frozen_mesh()
            single = Simulation(min=device.min, max=device.max, mesh=mesh, ...)
        """
        import numpy as np

        nodes = self.grid()
        out = {"type": "explicit"}
        for k, g in zip("xyz", nodes):
            g = np.asarray(g, dtype=float)
            lo, hi = (int(b.get("layers", 40)) if b.get("type") == "absorber" else int(b.get("layers", 12)) if b.get("type") == "pml" and not b.get("inside", False) else 0 for b in self.boundaries[k])
            out[k] = g[lo : len(g) - hi].tolist()
        return out

    def freeze_mesh(self) -> "Simulation":
        """Replace the mesh by the explicit mesh of the current grid
        (:meth:`frozen_mesh`), so that later changes of the structures no
        longer move the grid. Returns the simulation."""
        self.mesh = self.frozen_mesh()
        return self

    def time_info(self) -> tuple[float, int]:
        """``(dt, steps)``: the time step in seconds and the number of steps that
        ``run_time`` gives. The time step is the Courant step of the mesh times
        ``courant``, lowered if a dispersive medium needs it. Nothing is run.
        """
        from .. import _core

        return _core.time_info(self.to_json())

    def run(self, device: str = "auto", progress: int = 0) -> "Results":
        """Run the simulation.

        Parameters
        ----------
        device : str
            ``"auto"`` (the first GPU if there is one, else the CPU), ``"cpu"``,
            ``"cuda"``, ``"cuda:1"``, ``"cuda:0,1"`` (split over several GPUs) or
            ``"cuda:all"``.
        progress : int
            Print a line every ``progress`` steps; 0 prints nothing.

        Returns a ``Results``. A run whose fields grow without bound raises a
        ``RuntimeError``; one that reaches ``run_time`` before the fields have
        decayed below ``shutoff`` returns with a warning.
        """
        from .. import _core

        res = Results(_core.run(self.to_json(), device, progress))
        left = res.decay()
        if self.shutoff and not res.stats["stopped_early"] and left is not None and left > self.shutoff:
            # every source has an end (a continuous wave never decays)
            if all(q["signal"].get("type") in ("gaussian_pulse", "ricker") for q in self.sources):
                import warnings

                warnings.warn(
                    f"the run reached run_time before the fields had decayed: their energy is {left:.1e} of its peak, "
                    f"above shutoff = {self.shutoff:.1e}. Spectra are less converged than in a run that stops on "
                    "shutoff; increase run_time, or look at Results.stats['energy'] to see how the fields decay.",
                    stacklevel=2,
                )
        return res

    def view(self, results=None, **kwargs) -> str:
        """Open the graphical interface for this simulation (and optionally
        its ``results``) and return its URL. Keyword arguments go to
        :func:`lightsim.gui.view` (``host``, ``port``, ``open_browser``,
        ``block``)."""
        from .. import gui

        return gui.view(self, results, **kwargs)
