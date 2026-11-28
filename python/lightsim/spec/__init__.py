"""Pythonic builders for the lightsim simulation specification.

Every builder returns a plain ``dict`` matching the JSON schema consumed by the
Rust core (``lightsim_core::spec::Simulation``). Units are SI (metres,
seconds, Hz); helper ``um`` converts micrometres.

The builders live in private submodules by kind (``_media``, ``_geometry``,
``_sources``, ``_monitors``, ``_boundaries``, ``_simulation``, ``_results``);
everything is used from here, as ``lightsim.spec.<name>`` or
``lightsim.<name>``.
"""

from __future__ import annotations

# (also names of this module before it became a package; `import lightsim`
# re-exports every public name of it)
import json  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Iterable, Sequence  # noqa: F401

from ._util import C0, um, nm, Vec3, _v3, _AXES, _axis, _lift  # noqa: F401
from ._media import Medium, FourLevel, FourLevelTwoElectron, DirectorField, Raman, Lorentz, Drude, Debye, PEC, Sheet  # noqa: F401
from ._geometry import Box, Sphere, Cylinder, PolySlab, IndexGrid, Structure  # noqa: F401
from ._sources import GaussianPulse, ContinuousWave, PointDipole, PlaneWave, GaussianBeam, ModeSource, TFSF  # noqa: F401
from ._monitors import FieldTimeMonitor, ProbeMonitor, DFTMonitor, FluxMonitor, FluxBoxMonitor, ModeMonitor, FarFieldMonitor, EnergyMonitor  # noqa: F401
from ._boundaries import PML, Absorber, PERIODIC, PEC_WALL, PMC_WALL, Bloch, UniformMesh, AutoMesh, ExplicitMesh  # noqa: F401
from ._simulation import _complete, _complete_3d, Simulation, run_many  # noqa: F401
from ._results import Results  # noqa: F401
