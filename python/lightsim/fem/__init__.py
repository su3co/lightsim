"""Finite-element solvers on 2-D cross-sections.

* geometry and meshes: :class:`Geometry`, :func:`cross_section_geometry`
  (gdsfactory component + layer stack), :class:`Mesh`, :class:`Field`;
* heat conduction: :func:`solve_heat`, :func:`solve_heat_transient`,
  :func:`thermal_phase_shifter`;
* electrostatics: :func:`solve_electrostatics`, :func:`capacitance_matrix`,
  :func:`electro_optic_vpil`;
* waveguide modes: :func:`solve_modes` (full-vectorial Nedelec / Lagrange);
* gplugins-compatible entry points in :mod:`lightsim.fem.gplugins`.

The numerics (mesher, assembly, sparse direct and iterative solvers,
eigen-solver) run in the Rust crate ``lightsim-fem``.
"""

from .geometry import Circle, Geometry, Polygon, Rectangle, Region, Trapezoid, cross_section_geometry, cut_component  # noqa: F401
from .mesh import Field, Mesh, rectangle_mesh  # noqa: F401
from .heat import Current, HeatResult, TransientResult, solve_heat, solve_heat_transient, thermal_phase_shifter  # noqa: F401
from .electrostatics import ElectrostaticResult, capacitance_matrix, crystal_tensor, electro_optic_vpil, rotate_tensor, solve_electrostatics  # noqa: F401
from .modes import FemMode, eps_from, solve_modes  # noqa: F401
