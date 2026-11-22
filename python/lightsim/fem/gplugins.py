"""gplugins-compatible entry points of the finite-element solvers.

The functions here take the argument names gdsfactory users know from
gplugins (``gplugins.femwell``, ``gplugins.palace``, ``gplugins.elmer``) and
return objects with the same fields, so that a thin ``gplugins/lightsim``
wrapper can forward calls. As in gplugins, lengths are in **micrometres**
(``wavelength``, ``xsection_bounds``, ``wafer_padding``, mesh resolutions);
results are SI unless stated.

==========================================  ===========================================
gplugins                                     lightsim.fem.gplugins
==========================================  ===========================================
femwell.compute_cross_section_modes          :func:`compute_cross_section_modes`
femwell.compute_component_slice_modes        :func:`compute_component_slice_modes`
femwell (femwell.thermal.solve_thermal)      :func:`solve_thermal`
(femwell heater notebook)                    :func:`compute_component_slice_thermal`
palace / elmer run_capacitive_simulation_*   :func:`run_capacitive_simulation` (2-D)
==========================================  ===========================================
"""

from __future__ import annotations

import numpy as np

from .geometry import cross_section_geometry
from .heat import Current, HeatResult, solve_heat
from .mesh import Mesh

__all__ = ["solve_thermal", "compute_component_slice_thermal"]

UM = 1e-6


def _sizes_um(resolutions: dict | None) -> dict:
    """``{layer: size_um}`` or meshwell-style ``{layer: [ConstantInField(resolution=...)]}``
    or femwell-style ``{layer: {"resolution": r}}`` -> sizes in metres."""
    out = {}
    for k, v in (resolutions or {}).items():
        if isinstance(v, (int, float)):
            out[k] = float(v) * UM
        elif isinstance(v, dict) and "resolution" in v:
            out[k] = float(v["resolution"]) * UM
        else:
            vs = v if isinstance(v, (list, tuple)) else [v]
            res = [getattr(x, "resolution", None) for x in vs]
            res = [r for r in res if r is not None]
            if res:
                out[k] = float(min(res)) * UM
    return out


def solve_thermal(
    mesh: Mesh,
    thermal_conductivity: dict | None = None,
    specific_conductivity: dict | None = None,
    current_densities: dict | None = None,
    fixed_boundaries: dict | None = None,
    order: int = 1,
    powers: dict | None = None,
    convection: dict | None = None,
) -> HeatResult:
    """Steady heat conduction with Joule heating, after
    ``femwell.thermal.solve_thermal``.

    ``thermal_conductivity``: ``{region or material: k}`` in W/(m K) (others
    from the material tables). ``specific_conductivity``: ``{region:
    sigma}`` in S/m. ``current_densities``: ``{region: J}`` in A/m^2 (SI,
    unlike femwell's per-um^2 when meshing in um), heating ``J^2/sigma``.
    ``fixed_boundaries``: ``{boundary: T}`` (``"bottom"``, ``"top"``, a
    function of the edge midpoints, or a region name). ``powers``:
    ``{region: W/m}`` as an alternative to currents. Returns a
    :class:`~lightsim.fem.HeatResult` (``.temperature`` is the field).
    """
    sources = dict(powers or {})
    for name, j in (current_densities or {}).items():
        sources[name] = {"current_density": float(j), "sigma": float((specific_conductivity or {})[name])}
    return solve_heat(mesh, sources, thermal_conductivity, fixed_boundaries or {"bottom": 0.0}, convection, order=order)


def compute_component_slice_thermal(
    component,
    xsection_bounds,
    layer_stack=None,
    heater: str = "heater",
    power_per_length: float | None = None,
    current: float | None = None,
    thermal_conductivity: dict | None = None,
    resolutions: dict | None = None,
    default_resolution_max: float = 0.5,
    order: int = 2,
    fixed_boundaries: dict | None = None,
    convection: dict | None = None,
    exclude=("substrate",),
    background: str | None = None,
):
    """Temperature in a cross-section of a gdsfactory component.

    The component is cut along ``xsection_bounds`` (um) through
    ``layer_stack`` (default: the active PDK's), meshed with ``resolutions``
    (``{layer: um}``, also meshwell / femwell resolution specs), and the
    ``heater`` level dissipates ``power_per_length`` (W/m) or carries
    ``current`` (A, conductivity from the metal table). The substrate level
    is excluded by default and the bottom of what remains is held at 0 (the
    result is the rise above the substrate temperature). Returns ``(mesh,
    HeatResult)``.
    """
    g = cross_section_geometry(component, layer_stack, xsection_bounds, exclude=exclude, sizes=_sizes_um(resolutions), background=background)
    m = g.mesh(default_size=default_resolution_max * UM)
    if (power_per_length is None) == (current is None):
        raise ValueError("give exactly one of power_per_length and current")
    src = {heater: power_per_length if current is None else Current(current)}
    return m, solve_heat(m, src, thermal_conductivity, fixed_boundaries, convection, order=order)

