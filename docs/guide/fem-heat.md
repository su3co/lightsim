# Heat and thermo-optic phase shifters (FEM)

After this page you can compute the temperature in a cross-section heated by
a metal heater, the index change it causes, the phase shift per power of a
thermo-optic phase shifter, its P_pi, the thermal crosstalk to a neighbouring
waveguide and its thermal time constant, from plain shapes or from a
gdsfactory component.

## Equations and elements

`lightsim.fem.solve_heat` solves `-div(k grad T) = q` and
`solve_heat_transient` solves `rho c_p dT/dt - div(k grad T) = q` with
Lagrange P1 or P2 elements (`order=2` by default, isoparametric on curved
boundaries). The conductivity `k` is per region (a scalar or an anisotropic
in-plane tensor). The source `q` sits in named regions: a power per unit
length (W/m), a current along the waveguide axis through a conductor
(`fem.Current(I, sigma)`, heating `J^2 / sigma` with `J = I / A`) or a
current density as in femwell. Boundaries: a fixed temperature (`fixed`,
default `{"bottom": 0.0}`, so the result is the rise above the substrate),
convection `-k dT/dn = h (T - T_inf)` (`convection`), a heat flux
(`heat_flux`); edges without a condition are insulated. The transient uses
the theta scheme on one sparse factorisation.

Properties not given come from `lightsim.materials.THERMAL_PROPERTIES`
(Si, SiO2, Si3N4, GaAs, Ge) and from the thermal conductivities of Al, Cu,
Au and W in `lightsim.fem._materials.METALS`; every value there is checked
against the source listed in the table of the
[module page](../modules/fem.md#material-data), with its conditions (bulk
or film, temperature, wavelength). A quantity without a confirmed source has
no default: heater alloys such as TiN, electrical conductivities, and the
heat capacity of Si3N4 must be given, and the solver says which. Thin films
differ from bulk values; pass your own when you know them.

## A heater over a silicon strip

```python
import numpy as np
from lightsim import fem

um = 1e-6
g = fem.Geometry()
g.add(fem.Rectangle(-8 * um, -2.11 * um, 8 * um, -0.11 * um), "box", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-8 * um, -0.11 * um, 8 * um, 2.69 * um), "clad", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=0.03 * um, mesh_order=1)
g.add(fem.Rectangle(4.75 * um, -0.11 * um, 5.25 * um, 0.11 * um), "core2", "Si", size=0.03 * um, mesh_order=1)
g.add(fem.Rectangle(-1 * um, 2.11 * um, 1 * um, 2.25 * um), "heater", "TiN", size=0.05 * um, mesh_order=1)
mesh = g.mesh(default_size=0.5 * um, grading=0.5)

# TiN film: the inputs of femwell's heater example (no lightsim default)
tin_k, tin_sigma = {"TiN": 28.0}, 2.3e6
heat = fem.solve_heat(mesh, {"heater": fem.Current(5e-3, sigma=tin_sigma)}, conductivity=tin_k)
print(f"power {heat.total_power:.1f} W/m, core rise {heat.mean_temperature('core'):.2f} K, "
      f"heater {heat.mean_temperature('heater'):.2f} K")
```

`heat.temperature` is a `Field`: call it at points (`heat.temperature(x,
y)`), take its gradient, integrate it over regions or plot it
(`heat.temperature.plot()`).

## Phase shift, P_pi, crosstalk and time constant

`thermal_phase_shifter` solves the heat problem, maps the temperature to an
index change `dn/dT * dT`, and re-solves the fundamental mode of the
waveguide with lightsim's finite-difference mode solver on the perturbed
index (permittivity averaged over each Yee cell of a grid of
`mode_resolution`). The problem is linear in the power, so the phase per
power `dphi/dP = 2 pi / lambda * dneff/dp` does not depend on the heater
length (for a heater as long as the arm) and `P_pi = pi / (dphi/dP)`.

```python
tin_rho_cp = {"TiN": 3.1e6}  # J/(m^3 K), an illustrative input for the transient
r = fem.thermal_phase_shifter(mesh, "heater", "core", neighbour="core2", power_per_length=50.0,
                              conductivity=tin_k, capacity=tin_rho_cp, mode_resolution=25e-9,
                              transient=True, t_end=100e-6, nsteps=200)
print(f"P_pi = {r['P_pi'] * 1e3:.1f} mW, crosstalk {r['crosstalk']:.2f}, "
      f"time constant {r['time_constant'] * 1e6:.1f} us")
```

```text
P_pi = 21.1 mW, crosstalk 0.38, time constant 10.1 us
```

The crosstalk is the index change of the neighbouring guide (5 um away)
relative to that of the heated one. The time constant is the time the mean
temperature of the waveguide needs to cover 1 - 1/e of its rise after a
step of power.

## From a gdsfactory component

```python
import gdsfactory as gf
from lightsim.fem import gplugins as gp

gf.gpdk.PDK.activate()
c = gf.components.straight_heater_metal(length=50)
m, res = gp.compute_component_slice_thermal(c, [(25, -6), (25, 6)], power_per_length=50.0,
                                             resolutions={"core": 0.03, "heater": 0.1},
                                             thermal_conductivity={"TiN": 28.0})
print(m.region_names, f"core rise {res.mean_temperature('core'):.2f} K")
```

The component is cut along `xsection_bounds` (um) through the active PDK's
layer stack (`cross_section_geometry`); the substrate level is left out and
the bottom of the buried oxide is held at 0. `gp.solve_thermal(mesh,
thermal_conductivity, specific_conductivity, current_densities,
fixed_boundaries)` takes femwell's arguments (SI current densities).

## Accuracy

Validation cases (`python -m lightsim.fem.validate heat_slab heat_cylinder
heat_transient_slab`): a two-layer slab with a source and convection is
reproduced to 7e-12 with P2; a heated rod in a cylindrical shell converges
at order 2.97 (P2, curved) and 1.91 (P1), the centre temperature within
1e-7 (P2); the decay of the first slab mode is within 8.5e-6 after one time
constant (Crank-Nicolson, 100 steps per time constant). On the TiN heater
of gplugins' heater notebook, at femwell's triangle count (6 682 against
6 636), lightsim's core temperature is within 0.01 % (P2) and 0.18 % (P1)
of the converged value (14.6555 K, P2 on 76 000 to 320 000 triangles) and
its index change within 0.16 % (P2) and 0.33 % (P1) of the converged
2.944e-3. lightsim and femwell agree on the core temperature to 0.04 % (P1)
and 0.13 % (P2) and on the index change to 0.02 % (P1) and 0.15 % (P2); femwell's
default mesh is itself 0.14 % (temperature) and 0.30 % (index change) from the
converged values (see [the module page](../modules/fem.md)).
lightsim's region size is a largest edge length: its meshes have about
2.5 times the triangles of gmsh's at the same numbers (see
[What a mesh size means](../modules/fem.md#what-a-mesh-size-means)).

## Limits

The solver is linear: conductivities do not depend on temperature, there
is no radiation and no convection inside the stack. dn/dT is a constant per
material (none for GaAs, LiNbO3 or Al2O3; give it with `dndT=`, or the
mode solve stops with an error if such a region lies in the mode window). Joule
heating assumes a uniform current density over the heater cross-section.
