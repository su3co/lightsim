# Mode solver

`lightsim.xsection`, `lightsim.modes`, `lightsim.mode_sweeps` and
`lightsim.gplugins_modes` compute the modes of a waveguide cross-section
without setting up a 3-D simulation: effective and group index, loss
(leaky, absorbing and bent guides), dispersion, effective area,
confinement, polarisation, supermodes of coupled guides and overlaps. The
same solver gives the port modes of [`solve_modes`](../guide/modes-sparams.md)
in a simulation and the cell modes of [eigenmode expansion](eme.md).

## A cross-section

A `CrossSection` holds shapes in a plane with axes (u, v), each filled with
a material, on a rectilinear grid. The modes propagate along the normal
n = u x v. Lengths are in metres.

```python
import numpy as np
from lightsim.xsection import CrossSection

xs = CrossSection(background="sio2")                    # cladding
xs.add_rect(-0.25e-6, 0.0, 0.25e-6, 0.22e-6, "si")       # 500 x 220 nm core on v = 0
xs.window = ((-1.5e-6, -1.0e-6), (1.5e-6, 1.22e-6))
xs.mesh(xs.window, step=20e-9)                           # graded grid, 20 nm over the core
modes = xs.solve(1.55e-6, num_modes=2, group_index=True)
for m in modes:
    print(f"neff {m.neff.real:.4f}  ng {m.n_group:.4f}  TE {m.te_fraction:.2f}  "
          f"Aeff {m.effective_area() * 1e12:.3f} um2  in Si {m.confinement('si'):.3f}")
```

```text
neff 2.4435  ng 4.1827  TE 0.98  Aeff 0.191 um2  in Si 0.914
neff 1.7767  ng 3.7690  TE 0.04  Aeff 0.339 um2  in Si 0.570
```

Shapes: `add_rect`, `add_layer` (a full-width layer), `add_polygon` and
`add_trapezoid` (sloped sidewalls, angle from the vertical as in
gdsfactory). Later shapes are painted over earlier ones unless a
`priority` says otherwise. `CrossSection.strip(...)` builds strip, rib and
coupled guides with the parameters of gplugins' `Waveguide`.

Materials are a refractive index (complex for loss), a name (the
dispersive formulas of [Materials](materials.md), else the constant table
of the gdsfactory import; `material_index={"name": n}` overrides both),
three indices for diagonal anisotropy, a 3x3 permittivity tensor in (u, v,
n) axes, or a function of the wavelength. `uniaxial(n_o, n_e, axis)` gives
the tensor of a tilted uniaxial crystal such as lithium niobate.

### From gdsfactory

`from_gdsfactory(cross_section, layer_stack)` draws a gdsfactory
cross-section (`"strip"`, `"rib"`, a `CrossSection` object) with a layer
stack (default: the active PDK's) and cuts it; `from_component(component,
x=...)` cuts any component on the plane x = const (light along x, u = y,
v = z) or y = const. Patterned levels are cut, wafer levels become layers,
derived layers and painting order follow the 3-D import (see
[gdsfactory](../guide/gdsfactory.md)).

```python
import gdsfactory as gf
from lightsim.xsection import from_gdsfactory

gf.gpdk.PDK.activate()
rib = from_gdsfactory("rib", step=25e-9)
print([round(m.neff.real, 4) for m in rib.solve(1.55e-6, 2)])
```

```text
[2.5654, 2.1116]
```

On the same grid the cross-section rasteriser and the Rust core (the
sub-pixel smoothing the FDTD engine uses) give the same effective index of
a strip to 4e-8.

## Options of the solver

| Option | Meaning |
|---|---|
| `pml` | Perfectly matched layers in the outer cells: a number of cells on all four sides, `(u_lo, u_hi, v_lo, v_hi)`, or `{"cells": ..., "strength": 4.0}`. The layer stretches the coordinate by `s = 1 + i strength rho^2` (rho: depth in the layer). Needed for leaky modes, substrate leakage and bends. |
| `bend_radius` | Bend of this radius (m) about a centre at `u = bend_center - bend_radius`. Exact conformal map: the bend becomes a straight guide with `eps' = eps diag(r/R, r/R, R/r)` and the same `mu`. `neff` refers to the radius R. Bend modes are returned least lossy first. |
| `group_index` | Group index from `n_g = (∫ d(w eps)/dw abs(E)^2 + abs(eta0 H)^2) / (2 Re ∫ E x eta0 H*)`, with the materials' dispersion. |
| `extrapolate` | Solve again with every cell split in two and return `(4 n_fine - n_coarse) / 3` (Richardson); see accuracy below. |
| `periodic` | Periodic grid along u and/or v (slab problems with one cell). |
| `cache` | Reuse the result of an identical earlier solve. |

`solve_modes(sim, ...)` on a simulation plane takes the same `pml`,
`bend_radius`, `group_index` and `cache` options.

A `Mode` has, besides the attributes listed in [Modes and
S-parameters](../guide/modes-sparams.md): `E_n` and `H_n` (longitudinal
fields), `n_group`, `bend_radius`, and the derived quantities `wavelength`,
`beta`, `alpha` (power attenuation, 1/m), `loss_db_per_cm`,
`loss_db_per_90deg()`, `fraction_te`, `fraction_tm`, `effective_area()`
(`(∫abs(E)^2)^2 / ∫abs(E)^4`, all three components), `confinement(material
or region, kind="energy" or "power")`, `power_in(region)` and
`overlap(other, conjugate=True)`.

Overlaps (`lightsim.modes.mode_overlap`): the conjugated form
`1/4 ∫ (E1* x H2 + E2 x H1*) · n` (gplugins' `innerprod_trapz`, Tidy3D's
`outer_dot`) gives the power coupled between lossless modes as its square
magnitude; the unconjugated form `1/4 ∫ (E1 x H2 + E2 x H1) · n` is the
one that stays orthogonal for lossy and leaky modes. Solver modes carry
unit power, so a self-overlap is 1. Modes on different grids are compared
by interpolating the second onto the first.

### A leaky guide and a bend

A silicon slab on 0.5 µm of oxide over a silicon substrate leaks into the
substrate. With a PML at the bottom of the window the loss comes out
directly:

```python
xs = CrossSection(background=1.444)
xs.add_layer(-5e-6, -0.5e-6, 3.476, name="substrate")
xs.add_layer(0.0, 0.22e-6, 3.476, name="core")
step = 10e-9
xs.set_mesh([0, step], np.arange(-1.5e-6, 1.22e-6 + 1e-12, step))
modes = xs.solve(1.55e-6, 4, 2.85, periodic=(True, False), pml={"cells": (0, 0, 30, 0)})
te = min([m for m in modes if m.te_fraction > 0.5], key=lambda m: abs(m.neff - 2.85))
print(f"{te.neff:.5f}  {te.loss_db_per_cm:.2f} dB/cm")
```

```text
2.84716+0.00002j  6.64 dB/cm
```

The exact answer is 6.602 dB/cm (complex root of the slab dispersion
relation). The other modes of this window are substrate modes with large
loss, hence the choice of the mode nearest the guided index. A bend needs room and a PML on its outer side:

```python
from lightsim import mode_sweeps

print(mode_sweeps.bend_sweep([3e-6, 5e-6], step=25e-9)[["radius", "neff", "loss_db_90", "mismatch_db"]])
```

```text
     radius      neff    loss_db_90  mismatch_db
0  0.000003  2.446569  6.428336e-07     0.017451
1  0.000005  2.443987 -9.036493e-11     0.006160
```

`mismatch_db` is the loss of one straight-to-bend junction from the mode
overlap. `loss_db_90` at 5 µm is below the solver's resolution (a tiny
negative number); the radiation loss of a 500 x 220 nm silicon strip is negligible
at these radii (the bend loss observed in practice comes from substrate
leakage, roughness and the junctions).

## Sweeps

`lightsim.mode_sweeps` returns tidy tables (a pandas `DataFrame`, one row
per value and mode; a dict of lists without pandas): `neff_vs_width`,
`neff_vs_thickness`, `neff_vs_wavelength` (adds `D_ps_nm_km` and
`beta2_ps2_km`), `neff_vs_temperature` (linear thermo-optic coefficients,
`THERMO_OPTIC`, adds `dneff_dT`), `coupling_vs_gap` (`n_even`, `n_odd`,
`kappa = pi dn / lambda`, `lc = lambda / (2 dn)`), `bend_sweep`, `sweep`
(any parameter through a function that builds the cross-section) and
`overlap_table`.

```python
tab = mode_sweeps.neff_vs_wavelength(np.linspace(1.5e-6, 1.6e-6, 5), step=25e-9)
print(tab[["neff", "ng", "D_ps_nm_km"]].round(4))
```

```text
     neff      ng  D_ps_nm_km
0  2.4984  4.1705         NaN
1  2.4705  4.1749    583.0182
2  2.4425  4.1792    557.6780
3  2.4145  4.1833    516.2070
4  2.3864  4.1869         NaN
```

## gplugins-style functions

`lightsim.gplugins_modes` has the functions of `gplugins.modes` (MPB) and
`gplugins.tidy3d.modes` with their argument names, micrometre units and
return types; the last rows list the EME and circuit counterparts:

| gplugins | lightsim | Notes |
|---|---|---|
| `gplugins.modes.find_modes_waveguide` | `gplugins_modes.find_modes_waveguide` | `{mode_number: Mode}` with `neff`, `ng`, `fraction_te`, `effective_area`, fields, `y`, `z` |
| `find_modes_coupler` | `find_modes_coupler` | `core_widths`, `gaps` |
| `find_neff_vs_width` | `find_neff_vs_width` | DataFrame, columns `1..nmodes`, `width` |
| `find_mode_dispersion` | `find_mode_dispersion` | three-wavelength `ng`, as gplugins |
| `find_neff_ng_dw_dh` | `find_neff_ng_dw_dh` | DataFrame `dw`, `dh`, `neff`, `ng` |
| `find_coupling_vs_gap`, `coupling_length` | same names | DataFrame `gap`, `ne`, `no`, `lc`, `dn` |
| `modes.overlap.innerprod_trapz` | `innerprod_trapz` | conjugated overlap; lightsim modes have unit power |
| `gplugins.tidy3d.modes.Waveguide`, `WaveguideCoupler` | `Waveguide`, `WaveguideCoupler` | `n_eff`, `n_group`, `fraction_te/tm`, `mode_area`, `loss_dB_per_cm`, `overlap`, `coupling_length` |
| `sweep_n_eff`, `sweep_n_group`, `sweep_fraction_te`, `sweep_mode_area` | same names | numpy arrays shaped like Tidy3D's (no xarray) |
| `sweep_bend_mismatch`, `sweep_coupling_length` | same names | |
| `get_mode_solver_rib`, `get_mode_solver_coupler` | `CrossSection.strip(...)` | SI units |
| `gplugins.meow.MEOW` | `lightsim.eme.EMEComponent`, `lightsim.eme.sparameters` | see [Eigenmode expansion](eme.md) |
| `gplugins.sax.read.model_from_npz` | `lightsim.circuit.model_from_npz` | both key orders and units, see [Circuit simulation](circuit.md) |
| `sax.circuit` with `gplugins.sax.models` | `lightsim.circuit.circuit` and its models | same model convention |

The SI functions of `mode_sweeps` are the native interface; the
micrometre ones forward to them. MPB's `parity` arguments are accepted and
ignored (the solver returns every mode). The MPB window is periodic,
lightsim's is closed (zero field at the wall); a mode that reaches the wall
(the 200 nm wide guide of gplugins' `find_neff_vs_width` test in a 2 x 2
µm window) therefore differs between the two.

```python
from lightsim import gplugins_modes as gm

modes = gm.find_modes_waveguide(core_width=0.45, resolution=32)
print(round(modes[1].neff, 4), round(modes[2].neff, 4))
```

```text
2.3431 1.7371
```

## Method

The solver discretises Maxwell's equations on the staggered (Yee) grid of
the plane. Without off-diagonal permittivity it solves
`beta^2 E_t = P Q E_t` for the two transverse electric components; with a
full tensor (terms coupling the transverse and longitudinal components,
such as a tilted optic axis) it keeps the four transverse components and
solves the linear problem `beta [E_t; H_t] = M [E_t; H_t]`, twice the
unknowns. The eigenproblem is solved by shift-invert Arnoldi (SuperLU
factorisation of the shifted matrix, ARPACK), in real arithmetic when the
media are lossless and there is no PML. Evanescent and complex modes are
taken on the branch that decays along +n.

Permittivity: every rectangle, layer edge and polygon vertex is cut out of
the cells exactly (polygons are sub-sampled), giving fill fractions at the
E_u, E_v and E_n positions. In a cell with an interface the component along
the interface normal is `n_i^2 <1/eps>^-1 + (1 - n_i^2) <eps>`. `mesh()`
puts every interface on a grid node (`conforming=True`), which makes the
error a smooth function of the step.

## Accuracy

Measured against exact solutions (tests in
`python/tests/test_mode_solver.py`):

| Case | Reference | Result |
|---|---|---|
| Si slab 220 nm in SiO2, TE and TM | exact dispersion relation | error 2.5e-3 at 22 nm cells, ratio 4.0 per halving; Richardson from 22 nm: below 2e-5 |
| leaky slab (0.5 µm oxide on Si), TE 6.602 dB/cm, TM 242.2 dB/cm | complex root, outgoing substrate wave | loss within 0.6 % (TE) and 1.0 % (TM) with 10 nm cells and 300 nm of PML |
| step-index fibre, n 2.0 / 1.444, radius 0.5 µm | HE11 vector eigenvalue equation | 6e-5 at 25 nm, 9e-6 at 12.5 nm; the degenerate pair splits by 1e-12 |
| 2-D slab bend, n 1.60 / 1.444, width 1 µm, R = 20 and 30 µm | Bessel/Hankel functions of complex order | loss within 1.3–2.4 %, Re(neff) within 2e-5 (both polarisations) |
| same bend | asymptotic (WKB) bend-loss formula | ratio 1.32, 1.03, 0.85 at 20, 30, 40 µm: the formula is asymptotic |
| coupled Si slabs, gap 200 nm, 5 nm cells | exact supermodes | supermode indices within 6e-4, coupling length within 0.08 % |
| uniaxial slab (n_o 2.21, n_e 2.14), optic axis in the (u,v), (u,n), (v,n) planes and tilted in 3-D | 4x4 transfer matrix (Berreman) | within 2e-5 |
| strip 500 x 220 nm, group index | finite difference of neff over wavelength | 2e-7 |

For a 450 x 220 nm strip (n 3.47 / 1.44) the effective index converges as
2.341332, 2.344402, 2.345315, 2.345501 at 40, 20, 10 and 5 nm cells; with
extrapolation 2.345707, 2.345613, 2.345565, 2.345548. At 20 nm with
extrapolation the error is about 7e-5 and the solve takes 1.4 s on a
laptop CPU.

## Speed

The cost is the sparse LU factorisation of the shifted matrix and the
Arnoldi iterations with it. On an Apple-silicon laptop core, a 176 x 124
grid (43,648 unknowns) takes 0.7 s to factorise and 0.2 s (one mode) to
1.0 s (four modes) to iterate; a 288 x 232 grid takes 5.6 s for two modes.
Real arithmetic (lossless media, no PML) is 2.3 times faster than complex.
Full anisotropy doubles the unknowns and needs complex arithmetic: a
lithium-niobate ridge on a 210 x 120 grid takes 0.84 s with the optic axis
along u (diagonal tensor) and 5.1 s with it tilted by 29 degrees towards
the propagation axis; the tilted solve leaves a spurious imaginary part of
about 1e-9 in neff.

Sweeps re-factorise at every point because the matrix changes with every
geometry and wavelength; `cache=True` and EME reuse identical
cross-sections. A port of the factorisation to a multithreaded supernodal
solver (or a GPU sparse direct solver) would speed up large grids several
times; the Python assembly is under 5 % of the time.

## Compared with other mode solvers

Tidy3D's local mode solver (through gplugins' `RectangularDielectric`, the
geometry of `gplugins.tidy3d.modes.Waveguide`, 500 x 220 nm strip, n 3.476
/ 1.444) gives effective indices of 2.511, 2.412, 2.423, 2.445 and 2.451
at `grid_resolution` 15, 20, 30, 40 and 60 (0.6–3.3 s); lightsim converges
to 2.44540 (TE) and 1.77031 (TM), and gives 2.44543 / 1.77039 at 20 nm with
extrapolation in 1.7 s. For a coupler of two 450 nm guides with a 200 nm
gap the two solvers agree to 1e-5 on the TE supermodes at Tidy3D's
resolution 40 (2.37096, 2.33894). MPB, femwell and Tidy3D's remote solver
were not run.

## Limits

- Second-order finite differences: 1e-3 accuracy needs about 10 nm cells
  in silicon, or 20 nm with `extrapolate=True`.
- Closed walls unless a PML is given; the window must be large enough for
  the mode to decay (or end in a PML).
- Magnetic materials: diagonal `mu` only (the bend map uses it).
- Group index with full anisotropy uses the diagonal of `d(w eps)/dw`.
- `Waveguide` does not model gplugins' sidewall/surface loss layers;
  build them as shapes of a `CrossSection` instead.
