# Finite-element solvers

`lightsim.fem` solves heat conduction, electrostatics and full-vectorial
waveguide modes with the finite-element method on 2-D cross-sections. It is
the counterpart of the finite-element backends of gplugins (`femwell` for
modes and heaters, `palace` and `elmer` for capacitance), without gmsh or any
other external program. The numerics are in the Rust crate
`crates/lightsim-fem`; the Python package builds geometries, cuts gdsfactory
components and post-processes. The guide pages show how to use each solver:
[heat](../guide/fem-heat.md), [electrostatics](../guide/fem-electrostatics.md),
[waveguide modes](../guide/fem-modes.md).

## Overview

```python
import numpy as np
from lightsim import fem

um = 1e-6
g = fem.Geometry()
g.add(fem.Rectangle(-2 * um, -2 * um, 2 * um, 0), "box", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-2 * um, 0, 2 * um, 2 * um), "clad", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-0.25 * um, 0, 0.25 * um, 0.22 * um), "core", "Si", size=0.03 * um, mesh_order=1)
mesh = g.mesh(default_size=0.25 * um, refine={"core": 0.015 * um})
print(mesh)
modes = fem.solve_modes(mesh, 1.55e-6, num_modes=2)
print(modes[0].neff, modes[0].te_fraction)
```

A `Geometry` is a list of regions: a shape, a name, a material, a target
mesh size and a painting priority (`mesh_order`; where regions overlap the
lowest value wins, as in gdsfactory). `fem.cross_section_geometry(component,
layer_stack, xsection_bounds)` builds one by cutting a gdsfactory component
along a line, the way gplugins prepares femwell meshes. Coordinates are in
metres; the solvers take materials by region or material name and fall back
on lightsim's material library.

## Mesher

The mesher is written for lightsim on the `spade` crate (constrained
Delaunay triangulation, MIT OR Apache-2.0); gmsh is not used (its licence is
GPL). Steps:

1. A size field `h(x)` is the smallest of the default size, the size of each
   region graded away from it (`size + grading * distance`) and the sizes of
   the interface refinements (`refine={"core": 10e-9}`), graded the same way.
2. The input segments are split at their intersections and collinear
   overlaps, so polygons of different regions share their points; each
   atomic segment is subdivided to the local size. Segments marked as arcs
   (`fem.Circle`) are subdivided on the arc.
3. The segments are constraints of a constrained Delaunay triangulation.
   Interior points are seeded at the centres of a quadtree refined to `h`,
   then the triangulation is refined to a minimum angle (25 degrees by
   default; spade's Ruppert-type refinement, which splits encroached
   constraints).
4. Each triangle takes the tag of the highest-priority region containing its
   centroid, so overlapping polygons need no Boolean operations. Vertices on
   arc chords are projected onto the arc and arc edges get curved midpoints:
   P2 elements are isoparametric on curved boundaries.

Measured: a 6 x 4 um cross-section with a 20 nm core mesh (15 000
triangles) takes 22 ms; minimum angles are 25 degrees except where input
edges meet at smaller angles. A circle of radius 0.3 um meshed at 50 nm has
an isoparametric area error of 2e-5 (straight-sided: 2e-3).

## Elements and solvers

| piece | what | where |
|---|---|---|
| Lagrange P1, P2 | scalar fields; P2 isoparametric on curved edges | `element.rs` |
| Nedelec (Whitney, first order) | transverse electric field of the mode solver | `element.rs`, `maxwell.rs` |
| quadrature | Dunavant rules of degree 1 to 8, Gauss-Legendre on edges | `quadrature.rs` |
| assembly | element loops in parallel (rayon), CSR from triplets | `sparse.rs`, `scalar.rs` |
| boundary conditions | Dirichlet (edges, regions), Neumann, convective (Robin) | `scalar.rs` |
| direct solver | faer sparse Cholesky and LU, real and complex | `linsolve.rs` |
| iterative solvers | conjugate gradients (Jacobi or symmetric Gauss-Seidel), BiCGSTAB (ILU(0)) | `linsolve.rs` |
| eigen-solver | shift-invert implicitly restarted Arnoldi, complex, generalised problems | `eigen.rs` |
| transient | theta scheme (backward Euler, Crank-Nicolson) on one factorisation | `scalar.rs` |

Everything is `f64` (complex `f64` for modes) in both precision builds of
lightsim: the `Real` type of `lightsim-core` is the type of the FDTD fields.

The mode solver uses the formulation of Vardapetyan and Demkowicz (also
femwell's): with `zeta = i beta E_z / k0^2`,

```text
A_tt = (1/k0^2) (1/mu_zz) curl e . curl v - (eps_t e) . v
A_tz = (mut^-1 grad zeta) . v
A_zt = (eps_t e) . grad g
A_zz = -k0^2 eps_zz zeta g
B_tt = -(1/k0^2) (mut^-1 e) . v            A x = beta^2 B x
```

with a full in-plane permittivity tensor, perfectly matched layers as
complex coordinate stretches `s = 1 + i S (d/L)^2` and bends through the
exact cylindrical-coordinate map `eps_t (1 + x/R)`, `eps_zz / (1 + x/R)`.

## Validation

`python -m lightsim.fem.validate` runs the FEM suite (`--list` lists the
cases); `pytest python/tests/test_fem_*.py` runs the same cases and the
comparisons below. Measured on an Apple M1 Pro:

| case | reference | measured |
|---|---|---|
| poisson_convergence | manufactured solution on mesher meshes (h = 0.1, 0.05, 0.025) | P1 orders 2.02 (L2) / 1.00 (H1); P2 3.05 / 2.02; P2 L2 error 2.0e-6 |
| isoparametric_disk | manufactured solution on a disk, u = 0 on the circle | curved P2 order 3.12 (4.3e-6); straight P2 order 1.97 (6.8e-4) |
| heat_slab | two-layer slab with source and convection (exact quadratic) | P2 7e-12, P1 4.6e-6 |
| heat_cylinder | heated rod in a shell (exact logarithmic solution) | P2 order 2.97, centre temperature 1e-7; P1 order 1.91, 4.5e-4 |
| heat_transient_slab | decay of the first slab mode, exp(-t/tau) | 8.5e-6 at tau, 2.5e-5 at 3 tau (Crank-Nicolson, dt = tau/100) |
| es_coax | C = 2 pi eps / ln(b/a) | P2 5.1e-7 (order 3.6), P1 1.0e-3 (order 1.9) |
| es_anisotropic_ellipses | rotated anisotropic tensor between similar ellipses, exact | 1.2e-7 |
| es_parallel_plates | thin plates with fringing vs Palmer's formula | 0.7 % (Palmer is approximate; the plates are d/50 thick) |
| es_microstrip | Hammerstad-Jensen Z0 (air) and eps_eff | 0.06 %, 0.08 % |
| modes_slab | exact slab dispersion, TE0 and TM0 | 1.5e-5 at h = 10 nm, order 2.1 (TE) and 1.9 (TM) |
| modes_fiber | exact HE11, TE01, TM01 of a step-index fibre | 1.3e-5, 7.4e-5, 8.7e-5 |
| modes_strip_vs_fd | 500 x 220 nm silicon strip, lightsim's FD solver | 4.0e-4 apart at 10 nm; FEM extrapolates to 2.44708 |
| modes_bend_loss | bent slab vs the exact ring (Bessel/Hankel, complex order) | Re neff 8e-7, loss 8.4e-4; d ln(alpha)/dR 0.5 % from Marcuse's exponent |
| modes_plasmon | surface plasmon, sqrt(eps_m eps_d / (eps_m + eps_d)) | Re 6.8e-6, Im 1.5e-4 |
| modes_lossy_perturbation | lossy core vs first-order perturbation theory | 1e-7 |
| eo_uniform_field | LiNbO3 slab in a uniform field vs the exact slab with n_e - n_e^3 r33 E / 2 | 1e-4 |

The bent-slab case also prints the loss of the asymptotic formula
`alpha = gamma^2 kappa^2 exp(gamma d) exp(-2 gamma^3 R / (3 beta^2)) /
(beta (1 + gamma d/2) (kappa^2 + gamma^2))`: it decays with R at the same
rate but is 1.56 times the exact loss at these parameters, so the case uses
the exact ring solution as its reference.

## Comparison with femwell and the FD solver

At test time only (femwell is not a dependency): `python
python/tests/fem_femwell_reference.py` and `test_fem_compare_femwell.py`.

| problem | femwell 0.1.12 / scikit-fem 12.0.2 | lightsim.fem |
|---|---|---|
| TiN heater over a Si strip (gplugins heater notebook geometry), 7.4 mA | core 14.635 K, max 39.14 K; dneff 2.935e-3 | core 14.650 K (P1) / 14.655 K (P2), max 39.27 K (P2); dneff 2.936e-3 |
| same, run time | gmsh 0.24 s, heat 0.03 s, mode 0.4 s (6 800 triangles) | mesh 0.02 s, heat 0.02 s (P1) / 0.11 s (P2) (16 600 triangles); FD mode solve 3.4 s (20 nm grid) |
| rib waveguide of gplugins' femwell test, TE0, core mesh 40 / 20 / 10 nm | 2.49791 / 2.50094 / 2.50163; mode solve 2.5 / 6.4 / 17.5 s (10 900 / 22 800 / 52 200 triangles) | 2.49881 / 2.50059 / 2.50107; mesh + modes 1.4 / 2.0 / 10.7 s (13 100 / 15 900 / 27 100 triangles) |

The two FEM solvers converge to the same index (within 6e-4 at matched
core resolution). For the rib, femwell's own gmsh mesher gave a
non-conforming mesh when the core and the slab were separate overlapping
polygons; the comparison meshes the rib as one polygon (gplugins uses
meshwell, which was not installed). lightsim's finite-difference solver on
the same strip gives 2.44584, 2.44560, 2.44683 at 40, 20, 10 nm cells (not
monotone); the FEM gives 2.44163, 2.44520, 2.44644 at core meshes of 40, 20,
10 nm (observed order 1.5, limited by the corner singularities).

## gplugins entry points

| gplugins | lightsim |
|---|---|
| `gplugins.femwell.compute_cross_section_modes` | `lightsim.fem.gplugins.compute_cross_section_modes` |
| `gplugins.femwell.compute_component_slice_modes` | `lightsim.fem.gplugins.compute_component_slice_modes` |
| `femwell.thermal.solve_thermal` (used by gplugins' heater notebook) | `lightsim.fem.gplugins.solve_thermal` |
| heater notebook (cut, mesh, heat) | `lightsim.fem.gplugins.compute_component_slice_thermal` |
| `gplugins.palace.run_capacitive_simulation_palace`, `gplugins.elmer.run_capacitive_simulation_elmer` | `lightsim.fem.gplugins.run_capacitive_simulation` (2-D cut, per length) |
| `ElectrostaticResults` | `lightsim.fem.gplugins.ElectrostaticResults` |
| femwell `Modes`, `Mode.n_eff`, `te_fraction`, `calculate_overlap`, `calculate_propagation_loss`, `calculate_confinement_factor` | `Modes`, `FemMode` with the same names |

Arguments follow gplugins: lengths in micrometres (`wavelength`,
`xsection_bounds`, resolutions, which may be meshwell `ConstantInField`
specs), `material_spec` with an infinite permittivity for conductors. The
results are SI; meshes are in metres.

## Limits

- 2-D cross-sections only. A 3-D electrostatic or driven RF solver (the
  rest of Palace's role) needs tetrahedral meshing of a layer stack,
  Nedelec elements on tetrahedra, iterative solvers with multigrid or
  domain decomposition for 10^6-10^7 unknowns and port definitions; none of
  it is started.
- The mode solver has first-order Nedelec / P1 elements only (femwell's
  `order=1`); `order=2` falls back to 1 with a warning. Corners of
  rectangular cores limit the convergence to order 1.5.
- Mode elements are straight-sided (curved boundaries are polygonal, error
  O(h^2) like the elements). PML and anisotropic permittivity with
  off-diagonal terms are not combined consistently (the off-diagonal term is
  kept as is inside the PML).
- The group index comes from the field energy and excludes material
  dispersion.
- The Pockels model keeps the diagonal of the index-ellipsoid change in the
  cross-section frame (r13, r22, r33) and drops the off-diagonal terms (r51,
  r22 shear); LiNbO3 is the only Pockels material.
- Thermo-optic and thermal data are room-temperature constants; the heat
  solver is linear (no temperature-dependent conductivity, no radiation).
- Sparse LU solves dominate the mode solver: about 6 ms per Arnoldi step
  for 18 000 unknowns, so four modes of a 9 000-triangle strip take 2.4 s,
  one mode 0.4 s.
- Drift-diffusion carriers (the devsim counterpart) are not implemented.
  The pieces it would reuse exist (mesher with doping regions from GDS
  levels, sparse LU, `lightsim.eo.soref_bennett`, the mode solver). Options:
  (a) a box-method (control-volume) discretisation on the Delaunay mesh with
  Scharfetter-Gummel edge fluxes and cotangent edge weights, Gummel
  iterations for low bias and a coupled Newton solve (Poisson, electrons,
  holes; Slotboom or quasi-Fermi variables) for forward bias, SRH and Auger
  recombination, ohmic contacts; validated against the abrupt-junction
  depletion width and built-in potential and the short-base ideal-diode
  law; (b) a 1-D drift-diffusion solver along the junction direction for
  lateral PN/PIN modulators, with the carrier profile extruded over the
  waveguide, which covers the common depletion modulator at a fraction of
  the work. Neither is started.

## GPU acceleration later

Worth doing: element assembly (embarrassingly parallel), the iterative
solvers for large heat and electrostatic problems (CG with a GPU
preconditioner) and batched eigen-solves over wavelength or bias sweeps.
The sparse direct factorisations, which dominate the 2-D problems, gain
little on a GPU at these sizes; none of this is started.

## Crates and licences

| crate | use | licence |
|---|---|---|
| spade 2.15 | constrained Delaunay triangulation, refinement | MIT OR Apache-2.0 |
| faer 0.24 | sparse LU / Cholesky, dense eigen-decomposition | MIT |
| rayon, num-complex, anyhow, thiserror | workspace crates | MIT OR Apache-2.0 |
