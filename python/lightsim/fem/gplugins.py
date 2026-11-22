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

__all__ = ["solve_thermal", "compute_component_slice_thermal", "ElectrostaticResults", "run_capacitive_simulation"]

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



class ElectrostaticResults:
    """Capacitance result with the fields of gplugins' ``ElectrostaticResults``.

    ``capacitance_matrix``: ``{(terminal_i, terminal_j): C}`` (Maxwell
    convention: self-capacitances on the diagonal, negative mutual
    capacitances off it); ``raw_capacitance_matrix``: the same as an array.
    Values are per unit length (F/m), or in F when a ``length`` (um) was
    given. ``mesh`` and ``solutions`` (the potential of each terminal at 1 V)
    replace gplugins' mesh and field file locations.
    """

    def __init__(self, matrix, names, mesh, per_length: bool):
        self.raw_capacitance_matrix = np.asarray(matrix)
        self.terminals = list(names)
        self.capacitance_matrix = {(a, b): float(matrix[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)}
        self.mesh = mesh
        self.per_length = per_length
        self.mesh_location = None
        self.field_file_location = None

    def __repr__(self) -> str:
        unit = "F/m" if self.per_length else "F"
        return f"<ElectrostaticResults terminals={self.terminals} ({unit})>"


def run_capacitive_simulation(
    component,
    xsection_bounds,
    layer_stack=None,
    material_spec: dict | None = None,
    element_order: int = 2,
    resolutions: dict | None = None,
    default_resolution_max: float = 1.0,
    conductor_refinement: float | None = None,
    terminals=None,
    ground=(),
    boundary: str = "neumann",
    length: float | None = None,
    background: str | None = "vacuum",
    exclude=(),
) -> ElectrostaticResults:
    """2-D capacitance matrix of a cross-section of a gdsfactory component,
    after ``run_capacitive_simulation_palace`` / ``_elmer``.

    ``material_spec`` follows gplugins (``{material: {"relative_permittivity":
    eps}}``); materials with an infinite permittivity are conductors, and
    every connected conductor piece of the cut is a terminal named
    ``"<level>_<k>"`` (k counts along the cut). ``terminals`` restricts and
    orders them; ``ground`` lists pieces held at 0 V that are not terminals.
    ``length`` (um, e.g. the component length along the cut normal) turns
    the per-length matrix into a total. A 2-D cut neglects the end effects a
    3-D solve includes.
    """
    material_spec = material_spec or {}
    g = cross_section_geometry(component, layer_stack, xsection_bounds, exclude=exclude, sizes=_sizes_um(resolutions), background=background)
    eps = {}
    conductor_mats = set()
    for mat, spec in material_spec.items():
        e = float(spec.get("relative_permittivity", 1.0))
        if np.isinf(e):
            conductor_mats.add(str(mat))
        else:
            eps[str(mat)] = e
    if background is not None:
        eps.setdefault(background, 1.0)
    # one region name per conductor piece
    counts = {}
    names = []
    for r in g.regions:
        if r.material in conductor_mats:
            k = counts.get(r.name, 0)
            counts[r.name] = k + 1
            r.name = f"{r.name}_{k}"
            r.mesh_order = min(r.mesh_order, 0)
            names.append(r.name)
    if not names:
        raise ValueError("no conductor in the cut: give the metals relative_permittivity = inf in material_spec")
    refine = {n: conductor_refinement * UM for n in names} if conductor_refinement else None
    m = g.mesh(default_size=default_resolution_max * UM, refine=refine)
    terms = list(terminals) if terminals is not None else [n for n in names if n not in ground]
    from .electrostatics import capacitance_matrix

    C, terms = capacitance_matrix(m, terms, eps, ground=list(ground), boundary=boundary, order=element_order)
    if length is not None:
        C = C * length * UM
    return ElectrostaticResults(C, terms, m, length is None)
