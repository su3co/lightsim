"""Simulation and port helpers for inverse design (2.5-D and 3-D)."""

from __future__ import annotations

import numpy as np

from ..spec import C0, PML, Medium, Simulation, um
from .smatrix import Port

__all__ = ["planar_simulation", "port_mode"]


def planar_simulation(min, max, wavelength, n_background, n_max, steps_per_wavelength=20.0, z_cell=1.0 * um, pml_layers=12, run_time=None, shutoff=1e-5, background="eim_background", polarization="both"):
    """Empty 2.5-D effective-index simulation: a 2-D simulation invariant
    along z (``dimensions=2``) on a uniform xy grid, standing for a slab of
    thickness ``z_cell``, with PML in x and y (the model of
    :mod:`lightsim.eim`, built without a layout). ``min`` / ``max`` are the xy
    corners; ``n_max`` (largest effective index) sets the cell size
    ``wavelength / (n_max * steps_per_wavelength)``. Add effective-index
    structures as z-spanning boxes / poly-slabs (z from -1 to 1) and the
    design from :meth:`lightsim.invdes.Design.effective_index`.

    ``polarization`` = ``"te"`` / ``"tm"`` restricts the simulation to one
    field set; with ``"both"`` the engine updates the sets the port modes
    drive, which for one mode family is the same thing."""
    dx = wavelength / (n_max * steps_per_wavelength)
    length = float(np.hypot(max[0] - min[0], max[1] - min[1]))
    pml = lambda: PML(layers=pml_layers)  # noqa: E731
    sim = Simulation(
        min=[float(min[0]), float(min[1]), 0.0],
        max=[float(max[0]), float(max[1]), float(z_cell)],
        mesh={"type": "uniform", "dx": [dx, dx, float(z_cell)]},
        run_time=run_time if run_time is not None else 8.0 * n_max * length / C0,
        boundaries={"x": [pml(), pml()], "y": [pml(), pml()], "z": [{"type": "periodic"}, {"type": "periodic"}]},
        background=background,
        shutoff=shutoff,
        dimensions=2,
        invariant_axis="z",
        polarization=polarization,
    )
    sim.add_medium(Medium(background, n=n_background))
    return sim


def port_mode(sim, name, axis, position, min, max, freq, direction, polarization=None, num_modes=1):
    """Solve the port modes and return a :class:`Port`.

    ``polarization`` = ``"te"`` / ``"tm"`` selects the 1-D lateral solver of a
    2-D / 2.5-D simulation (:func:`lightsim.eim.solve_port_mode`); ``None``
    uses :func:`lightsim.solve_modes` (the full-vectorial solver in 3-D).
    Passing a polarization in a 3-D simulation is an error: the lateral
    solver would read one line of cells and return a mode of the cladding.
    ``direction`` points into the device."""
    if getattr(sim, "slab", None) is not None:  # 2-D: two in-plane components span the slab
        box = sim._resolve({"min": list(min), "max": list(max)})
        min, max = box["min"], box["max"]
    if polarization is not None:
        from ..eim import solve_port_mode

        modes = solve_port_mode(sim, axis, position, min, max, freq, polarization, num_modes=num_modes)
    else:
        from ..modes import solve_modes

        modes = solve_modes(sim, axis, position, min, max, freq, num_modes=num_modes)
    p = Port(name, axis, float(position), [float(v) for v in min], [float(v) for v in max], [m.profile for m in modes], direction)
    p.neff = [complex(m.neff) for m in modes]
    return p
