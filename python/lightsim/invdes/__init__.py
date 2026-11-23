"""Inverse design with adjoint back-propagation through the S-matrix.

* :class:`SMatrixProblem` -- S[m, n, k] of N excitations, M output modes and
  K frequencies from N forward + M adjoint runs, with the vector-Jacobian
  product for any upstream gradient dL/dS,
* :class:`Design` -- density -> filter -> projection -> index grids, with the
  exact transpose of every step (3-D slab, two-level partial etch, 2.5-D
  effective index),
* :class:`CompositeProblem` -- a large device as a network of blocks, each
  solved by its own FDTD problem, joined by S-matrix composition
  (:mod:`lightsim.invdes.network`), with the gradient through the network,
* :class:`InverseDesign`, :func:`optimize` -- the optimisation loop (Adam or
  L-BFGS-B, beta continuation, checkpoints),
* :mod:`lightsim.invdes.shapes` -- shape parametrisations (polygons with
  movable vertices, level sets) drawn on the grid of a :class:`Design`, with
  their exact transposes (:class:`ShapeDesign`, :class:`Polygon`,
  :class:`LevelSet`; also importable from here),
* :mod:`lightsim.invdes.objectives` -- matrix fidelity, band transmission
  with crosstalk, averaged coupling,
* ``lightsim.invdes.torch`` / ``lightsim.invdes.jax`` -- the S-matrix as a
  ``torch.autograd.Function`` / ``jax.custom_vjp`` (imported on demand).

``SMatrixProblem(..., engine="native")`` computes S and its gradient in the
persistent Rust session of :mod:`lightsim.autodiff` (fields and adjoint
product on the device) instead of driving every run from Python.

See ``docs/modules/invdes.md``.
"""

from .composite import Block, CompositeDesign, CompositeProblem, CompositeSolution  # noqa: F401
from .geometry import density_polygons  # noqa: F401
from .network import Network, NetworkSolution, chain  # noqa: F401
from .objectives import band_transmission, matrix_fidelity, matrix_metrics, mean_coupling, softmin  # noqa: F401
from .optimize import InverseDesign, beta_schedule, optimize  # noqa: F401
from .param import Design, binarization, conic_kernel, filter_density, gaussian_kernel, project  # noqa: F401
from .planar import planar_simulation, port_mode  # noqa: F401
from .shapes import LevelSet, Polygon, ShapeDesign  # noqa: F401  (documented in lightsim.invdes.shapes)
from .smatrix import DesignGrid, Excitation, NativeSolution, Port, SMatrixProblem, SMatrixSolution, grid_weights  # noqa: F401

__all__ = [
    "SMatrixProblem",
    "CompositeProblem",
    "CompositeSolution",
    "CompositeDesign",
    "Block",
    "Network",
    "NetworkSolution",
    "chain",
    "SMatrixSolution",
    "NativeSolution",
    "Port",
    "Excitation",
    "DesignGrid",
    "grid_weights",
    "Design",
    "InverseDesign",
    "optimize",
    "beta_schedule",
    "matrix_fidelity",
    "matrix_metrics",
    "band_transmission",
    "mean_coupling",
    "softmin",
    "density_polygons",
    "planar_simulation",
    "port_mode",
    "project",
    "filter_density",
    "conic_kernel",
    "gaussian_kernel",
    "binarization",
]
