# Waveguide modes with finite elements

After this page you can solve the full-vectorial modes of a waveguide
cross-section on a triangle mesh, including leaky, lossy, anisotropic,
metallic and bent waveguides, read their effective index, group index,
mode area, TE fraction and loss, and use a mode as the source or monitor
profile of a lightsim FDTD simulation.

## Equations and elements

`lightsim.fem.solve_modes` discretises Maxwell's equations for fields
`E(x, y) exp(i (beta z - omega t))` with first-order Nedelec edge elements
for the transverse field and P1 Lagrange elements for the longitudinal one
(the formulation of femwell) and finds the modes closest to a target index
with a shift-invert Arnoldi eigen-solver. The permittivity is per triangle:
complex (loss, metals) and anisotropic (`(eps_xx, eps_yy, eps_zz)` or
`(eps_xx, eps_yy, eps_xy, eps_zz)`). Perfectly matched layers (`pml=`) are
complex coordinate stretches inside the mesh boundary; a bend radius
(`radius=`) maps the bent waveguide onto a straight one with the exact
cylindrical-coordinate transformation, so the imaginary part of the
effective index is the bend loss. The outer boundary is an electric wall
(`boundary="pec"`, default), a magnetic wall (`"pmc"`, femwell's default)
or electric on chosen sides.

Compared with `lightsim.solve_modes` (finite differences on the FDTD grid
of a `Simulation`), the FEM solver works on any mesh with curved and
slanted boundaries, local refinement and PML, and converges smoothly with
the mesh; the FD solver gives exactly the discrete modes of the FDTD grid.

## A silicon strip

```python
import numpy as np
from lightsim import fem

um = 1e-6
g = fem.Geometry()
g.add(fem.Rectangle(-1.25 * um, -1.11 * um, 1.25 * um, 1.11 * um), "clad", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=0.02 * um, mesh_order=1)
mesh = g.mesh(default_size=0.1 * um)
modes = fem.solve_modes(mesh, 1.55e-6, num_modes=2, index={"Si": 3.4777, "SiO2": 1.444})
for m in modes:
    print(f"neff {m.neff.real:.5f}  TE {m.te_fraction:.2f}  ng {m.group_index:.3f}  "
          f"Aeff {m.effective_area() / um**2:.3f} um^2")
print(f"overlap TE0-TM0: {abs(modes[0].calculate_overlap(modes[1])):.0e}")
```

```text
neff 2.44520  TE 0.98  ng 4.054  Aeff 0.191 um^2
neff 1.77118  TE 0.05  ng 3.647  Aeff 0.341 um^2
overlap TE0-TM0: 1e-14
```

The modes are `FemMode` objects, a subclass of `lightsim.Mode`: `neff`,
`te_fraction`, `E` and `H` (sampled on a regular grid, coordinates in
`info["u_nodes"]`, `info["v_nodes"]`) as for the FD solver, and in
addition `fields(x, y)` (E in V/m and H in A/m for 1 W anywhere),
`quadrature()`, `calculate_overlap`, `calculate_confinement_factor`,
`group_index` (field energy; material dispersion not included),
`effective_area()` and `loss_db_per_m`. Materials missing from `index` come
from lightsim's material library at the wavelength.

## A mode as an FDTD source

`profile_for(sim, axis, position, min, max)` samples the FEM fields at the
staggered positions the FDTD engine uses on a mode window of a
`Simulation`, which is what `ModeSource` and `ModeMonitor` take:

```python
import lightsim as ls

sim = ls.Simulation(min=[0, -1.25 * um, -1.11 * um], max=[3 * um, 1.25 * um, 1.11 * um],
                    mesh=ls.UniformMesh(40e-9), run_time=50e-15, background="ox")
sim.add_medium(ls.Medium("ox", n=1.444))
sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]), ls.Medium("si_strip", n=3.4777))
lo, hi = [0.5 * um, -1.2 * um, -1.0 * um], [0.5 * um, 1.2 * um, 1.0 * um]
profile = modes[0].profile_for(sim, 0, 0.5 * um, lo, hi)
f0 = ls.C0 / 1.55e-6
sim.add_source(ls.ModeSource(0, 0.5 * um, "+", lo, hi, profile, ls.GaussianPulse(f0, 0.2 * f0)))
print(profile["dims"], round(profile["neff"], 5))
```

The mesh coordinates (x, y) map to the plane's in-plane axes (shifted by
`offset`), with the same axis convention as `lightsim.solve_modes`.

## Leaky modes, bends and metals

A PML on the outer side of a bend absorbs the radiation; give `n_guess`
near the expected index, because the conformal map makes the far outer
cladding look like a high index. Modes with most of their energy inside the
PML are dropped (`keep_pml_modes=True` keeps them).

```python
g = fem.Geometry().add(fem.Rectangle(-2 * um, -2 * um, 3.5 * um, 2 * um), "clad", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-0.25 * um, -0.11 * um, 0.25 * um, 0.11 * um), "core", "Si", size=0.03 * um, mesh_order=1)
bend_mesh = g.mesh(default_size=0.1 * um)
bent = fem.solve_modes(bend_mesh, 1.55e-6, 1, index={"Si": 3.4777, "SiO2": 1.444}, radius=2 * um,
                       pml={"right": 1.0 * um}, n_guess=2.45)[0]
print(f"R = 2 um: neff {bent.neff:.6f}, loss {bent.loss_db_per_m / 100:.2f} dB/cm")
```

Metals and lossy media take complex permittivities; refine the mesh near
metal interfaces (the skin depth of gold at 1550 nm is about 25 nm):

```python
w = 0.05 * um
g = fem.Geometry().add(fem.Rectangle(0, -0.3 * um, w, 0), "metal", "metal").add(fem.Rectangle(0, 0, w, 8 * um), "glass", "glass")
spp_mesh = g.mesh(default_size=0.4 * um, refine={"metal": 0.002 * um}, grading=0.2)
walls = lambda x, y: (np.abs(x) > 1e-12) & (np.abs(x - w) > 1e-12)  # electric top and bottom only
spp = fem.solve_modes(spp_mesh, 1.55e-6, 1, eps={"metal": -95 + 10j, "glass": 2.25}, boundary=walls, n_guess=1.52)[0]
print(spp.neff, np.sqrt((-95 + 10j) * 2.25 / (-95 + 10j + 2.25)))
```

## From gdsfactory: the gplugins functions

`lightsim.fem.gplugins.compute_cross_section_modes` and
`compute_component_slice_modes` take the arguments of
`gplugins.femwell` (micrometres; `resolution_specs` as `{layer: um}` or
meshwell specs; `material_name_to_index`) and return `Modes`, a list of
`FemMode` with `n_effs`:

```python
import gdsfactory as gf
from gdsfactory.gpdk import LAYER_STACK
from gdsfactory.technology import LayerStack
from lightsim.fem import gplugins as gp

gf.gpdk.PDK.activate()
stack = LayerStack(layers={k: LAYER_STACK.layers[k].model_copy(deep=True) for k in ("core", "clad", "slab90", "box")})
stack.layers["core"].thickness = 0.2
rib = gp.compute_cross_section_modes("rib", stack, wavelength=1.55, num_modes=2,
                                     resolution_specs={"core": 0.03, "slab90": 0.06})
print(rib.n_effs.real, [round(m.te_fraction, 2) for m in rib])
```

The cut keeps the sidewall angles of the layer stack (trapezoids).
femwell's `order=2` is not available (first-order elements only).

## Accuracy and run time

Validation (`python -m lightsim.fem.validate modes_slab modes_fiber
modes_strip_vs_fd modes_bend_loss modes_plasmon modes_lossy_perturbation`):
slab TE0 and TM0 within 1.5e-5 of the exact dispersion relation at 10 nm,
second-order convergence; step-index fibre HE11 within 1.3e-5, TE01 and
TM01 within 9e-5; a bent slab within 8e-7 (real part) and 8.4e-4 (loss) of
the exact ring solution; a surface plasmon within 6.8e-6 (real part) and
1.5e-4 (imaginary part) of the closed form; the 500 x 220 nm strip within
4e-4 of lightsim's FD solver at 10 nm. On gplugins' rib example lightsim
and femwell agree within 6e-4 at matched core resolution. Run time on an
Apple M1 Pro: one mode of the strip with a 20 nm core mesh (10 700
triangles) 0.5 s, 40 600 triangles 2.2 s; four modes take about five times
as long as one (Arnoldi restarts, about 6 ms per sparse triangular solve
for 18 000 unknowns).

## Limits

First-order elements only (no femwell `order=2`); straight-sided elements
(curved boundaries are polygonal at the mesh size); the corners of
rectangular cores limit the convergence to about order 1.5. The group
index excludes material dispersion. PML and off-diagonal permittivity
components are not combined consistently. No magnetic materials.
