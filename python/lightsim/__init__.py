"""lightsim — GPU-first FDTD electromagnetic solver.

Kernels are written in Rust and compiled to PTX with NVIDIA cuda-oxide; the
Python package builds simulations (or imports gdsfactory layouts), solves
waveguide modes on the exact Yee grid and runs everything through the Rust
core on CPU, one GPU, several GPUs, or several machines.

    import lightsim as ls
    sim = ls.Simulation(min=[0, 0, 0], max=[4e-6, 2e-6, 2e-6],
                        mesh=ls.UniformMesh(40e-9), run_time=100e-15)
    ...
    res = sim.run(device="cuda:0")
"""

from ._core import __version__, devices, release_gpus  # noqa: F401
from ._core import precision  # noqa: F401  ("f32", or "f64" in a double-precision build)
from .spec import *  # noqa: F401,F403
from .spec import Results, Simulation  # noqa: F401
from .modes import Mode, gaussian_mode, solve_modes  # noqa: F401
from .symmetry import run_symmetric  # noqa: F401
from .sweep import AngleSweep, sweep_angle  # noqa: F401
from .sparameters import SParameters  # noqa: F401  (S-parameters: Touchstone, INTERCONNECT text, npz, SAX)
from . import invdes  # noqa: F401,E402  (inverse design: adjoint S-matrix gradients, parametrisation, optimiser)
from . import autodiff  # noqa: F401,E402  (gradients inside the engine: persistent S-matrix session, reverse mode)

__all__ = ["Simulation", "Results", "solve_modes", "gaussian_mode", "Mode", "devices", "release_gpus", "sweep_angle", "AngleSweep", "SParameters", "invdes", "autodiff", "__version__"]
