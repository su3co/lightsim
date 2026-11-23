# Electrostatics, capacitance and electro-optic modulators (FEM)

After this page you can compute the potential and field of conductors in a
cross-section, the capacitance matrix per unit length of a set of
electrodes, and the half-wave voltage-length product V_pi L of a lithium
niobate modulator, from plain shapes or from a gdsfactory component.

## Equations and elements

`lightsim.fem.solve_electrostatics` solves `-div(eps0 eps_r grad phi) =
rho` with Lagrange P1 or P2 elements (P2 by default). `eps_r` is per region:
a scalar, a diagonal in-plane tensor `(eps_uu, eps_vv)` or a full one
`(eps_uu, eps_vv, eps_uv)` (`fem.rotate_tensor`, `fem.crystal_tensor`).
Conductors are regions held at a potential; their interior is meshed and
fixed. The outer boundary is insulating (`boundary="neumann"`, zero normal
displacement) or grounded (`"ground"`), or set side by side (`{"bottom":
0.0}`). Static permittivities not given come from a table
(`lightsim.fem._materials.STATIC_PERMITTIVITY`: SiO2 3.9, Si 11.7, Si3N4 7.5,
LiNbO3 (85, 85, 28.7) in the crystal frame, ...).

`fem.capacitance_matrix` returns the Maxwell matrix per unit length (F/m):
`C_ij = u_i^T K u_j` with `u_j` the potential when conductor j is at 1 V and
the others at 0 V, i.e. the field energy, which converges at twice the
order of the potential. One sparse factorisation serves all terminals.

## Capacitance of a coaxial line and of two wires

```python
import numpy as np
from lightsim import fem

um, eps0 = 1e-6, 8.8541878128e-12
g = fem.Geometry()
g.add(fem.Circle((0, 0), 3.5 * um, size=0.15 * um), "outer", "teflon", mesh_order=10)
g.add(fem.Circle((0, 0), 1.0 * um, size=0.15 * um), "inner", "metal", mesh_order=1)
mesh = g.mesh(default_size=0.15 * um)
C, names = fem.capacitance_matrix(mesh, ["inner"], {"teflon": 2.1}, boundary="ground")
exact = 2 * np.pi * eps0 * 2.1 / np.log(3.5)
print(f"C = {C[0, 0] * 1e12:.6f} pF/m, error {abs(C[0, 0] / exact - 1):.1e}")
```

```text
C = 93.256520 pF/m, error 5.1e-07
```

Several terminals give the full matrix; diagonal entries are
self-capacitances, off-diagonal entries negative mutual capacitances:

```python
g = fem.Geometry().add(fem.Rectangle(-10 * um, -10 * um, 10 * um, 10 * um), "air", "air")
g.add(fem.Circle((-2 * um, 0), 0.5 * um, size=0.1 * um), "a", "metal", mesh_order=1)
g.add(fem.Circle((2 * um, 0), 0.5 * um, size=0.1 * um), "b", "metal", mesh_order=1)
mesh2 = g.mesh(default_size=1 * um)
C2, names = fem.capacitance_matrix(mesh2, ["a", "b"], boundary="ground")
print(names, np.round(C2 * 1e12, 3), "pF/m")
r = fem.solve_electrostatics(mesh2, {"a": 1.0, "b": 0.0}, boundary="ground")
print("E at the midpoint (V/m):", r.field(np.array([0.0]), np.array([0.0]))[0])
```

## V_pi L of a thin-film lithium niobate modulator

`fem.electro_optic_vpil` solves the static field, converts it to the
crystal frame of the given cut, computes the Pockels index change of
LiNbO3 (r13, r22, r33 from `lightsim.materials.ELECTRO_OPTIC`, Weis and
Gaylord 1985; the diagonal of the index ellipsoid in the cross-section frame
is kept), re-solves the optical mode on the perturbed index and returns
`dneff/dV` and `V_pi L = lambda / (2 |dneff/dV|)` for one arm.

```python
g = fem.Geometry()
g.add(fem.Rectangle(-10 * um, -2 * um, 10 * um, 0), "box", "SiO2", mesh_order=5)
g.add(fem.Rectangle(-10 * um, 0, 10 * um, 0.3 * um), "slab", "LiNbO3", mesh_order=3)
g.add(fem.Trapezoid(0, 1.6 * um, 1.2 * um, 0.3 * um, 0.6 * um), "ridge", "LiNbO3", size=0.05 * um, mesh_order=2)
g.add(fem.Rectangle(-10 * um, 0.3 * um, 10 * um, 3 * um), "air", "air", mesh_order=10)
g.add(fem.Rectangle(-6.5 * um, 0.3 * um, -2.5 * um, 0.8 * um), "gnd", "Au", mesh_order=1)
g.add(fem.Rectangle(2.5 * um, 0.3 * um, 6.5 * um, 0.8 * um), "sig", "Au", mesh_order=1)
mesh3 = g.mesh(default_size=0.5 * um, refine={"gnd": 0.05 * um, "sig": 0.05 * um})
r = fem.electro_optic_vpil(mesh3, "sig", "gnd", ["slab", "ridge"], "ridge", cut="x-cut",
                           mode_method="fem")
print(f"neff {r['neff'].real:.4f}, V_pi L = {r['VpiL'] * 100:.2f} V cm (one arm)")
```

With `mode_method="fem"` the optical mode comes from `fem.solve_modes` and
the electrodes become electric walls of the optical problem;
`mode_method="fd"` uses lightsim's finite-difference solver on a grid
around the waveguide. Cuts: `"x-cut"` (optic axis in the chip plane, across
the guide, propagation along crystal y), `"z-cut"`, `"y-cut"`,
`"x-cut-z-prop"`, or a tuple of the crystal axes along (u, v, propagation).

## From a gdsfactory component

`lightsim.fem.gplugins.run_capacitive_simulation` takes gplugins'
`material_spec` (an infinite `relative_permittivity` marks a conductor)
and returns an object with the fields of gplugins' `ElectrostaticResults`
(`capacitance_matrix` keyed by terminal pairs, `raw_capacitance_matrix`).
Each conductor piece of the cut is a terminal named `"<level>_<k>"`.

```python
import gdsfactory as gf
from gdsfactory.gpdk import LAYER
from gdsfactory.technology import LayerLevel, LayerStack
from lightsim.fem import gplugins as gp

gf.gpdk.PDK.activate()
stack = LayerStack(layers=dict(
    substrate=LayerLevel(layer=LAYER.WAFER, thickness=5, zmin=0, material="Si", mesh_order=99),
    metal=LayerLevel(layer=LAYER.M1, thickness=0.2, zmin=5, material="Nb", mesh_order=2),
    vacuum=LayerLevel(layer=LAYER.WAFER, thickness=5, zmin=5, material="vacuum", mesh_order=100),
))
c = gf.Component()
c.add_polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)], layer=LAYER.WAFER)
for x in (-15.0, 15.0):
    c.add_polygon([(x - 10, -10), (x + 10, -10), (x + 10, 10), (x - 10, 10)], layer=LAYER.M1)
spec = {"Si": {"relative_permittivity": 11.45}, "Nb": {"relative_permittivity": np.inf},
        "vacuum": {"relative_permittivity": 1}}
res = gp.run_capacitive_simulation(c, [(-50, 0), (50, 0)], stack, spec, resolutions={"metal": 0.5},
                                   default_resolution_max=5.0, conductor_refinement=0.1)
print(res, np.round(res.raw_capacitance_matrix * 1e12, 2), "pF/m")
```

A 2-D cut gives capacitances per unit length; `length=` (um) multiplies
them, which neglects the end effects a 3-D solve includes.

## Accuracy

Validation (`python -m lightsim.fem.validate es_coax es_anisotropic_ellipses
es_parallel_plates es_microstrip eo_uniform_field`): coaxial line 5.1e-7
(P2, order 3.6) and 1.0e-3 (P1); a rotated anisotropic tensor between two
similar ellipses 1.2e-7 against the exact `2 pi eps0 sqrt(det eps) /
ln(b/a)`; parallel plates with fringing 0.7 % from Palmer's approximate
formula; microstrip within 0.06 % (C0) and 0.08 % (eps_eff) of
Hammerstad-Jensen; the Pockels shift of a LiNbO3 slab in a uniform field
within 1e-4 of the exact slab solution.

## Limits

2-D only (no 3-D capacitance or driven RF, see the
[module page](../modules/fem.md)). Conductors are perfect; there is no
frequency dependence, no space charge from carriers (no drift-diffusion)
and no piezo-electric or photo-elastic coupling. LiNbO3 is the only Pockels
material; the off-diagonal (shear) index changes are dropped.
