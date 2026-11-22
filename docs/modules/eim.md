# 2.5-D effective-index simulations

`lightsim.eim` collapses a planar photonic circuit along z and solves the
resulting problem as a 2-D lightsim simulation (see
[2-D, 2.5-D and 3-D](../guide/dimensions.md)). It plays the
role of Lumerical's varFDTD (MODE "2.5D FDTD") for quick layout-level
simulations of gdsfactory components, at a small fraction of the 3-D cost.

## Method

1. **Vertical reduction.** For every patterned level of the layer stack
   (core, partially etched slab, ...) the vertical stack "wafer layers + that
   level" is a 1-D multilayer. Its guided slab mode gives one effective index
   per level. Where no patterned level is present the index is that of the
   wafer material at the height of the patterned levels (the lateral
   cladding, e.g. SiO2), since an unpatterned stack guides nothing.
2. **Slab solver.** `slab_modes` solves the exact dispersion relation of a
   lossless isotropic multilayer by a transfer-matrix characteristic
   function: start from the solution decaying into the bottom half-space,
   propagate the pair (F, F'/rho) through the layers (F = E_y, rho = 1 for
   TE; F = H_y, rho = n^2 for TM) and require decay in the top half-space.
   The function is continuous in n_eff, so its zeros are bracketed on a
   sampled grid and refined with Brent's method. No discretization error.
3. **2-D model.** Each polygon becomes a z-invariant region with its level's
   effective index; where levels overlap, the higher effective index wins
   (a rib over its slab). The model is a **2-D simulation invariant along
   z** (`dimensions=2`): one cell along z, PML in x and y, the 2-D Courant
   limit, the usual automatic graded mesh in the plane. `z_cell` (1 um by
   default) is the thickness of the slab the model stands for. Fields do not
   depend on z, so it only scales fluxes and mode powers (checked: identical
   S-parameters to 4 digits with a thin and a thick cell).
4. **Polarization.** The slab TE family (E in the chip plane) is the 2-D
   polarization `"te"`, fields (Ex, Ey, Hz); the slab TM family is `"tm"`,
   fields (Ez, Hx, Hy). The simulation is restricted to the set of the chosen
   slab mode, so the solver advances three field components, not six.
5. **Port modes.** The full-vectorial solver uses closed-wall 2-D difference
   operators and cannot represent a single periodic cell, so
   `solve_port_mode` solves the 1-D lateral eigenproblem directly on the Yee
   positions of the port line, with the sub-pixel-smoothed permittivity the
   engine uses (from `plane_info`). For TE it solves
   `beta^2 e = (k0^2 + D_f eps_n^-1 D_b) eps_t e` with `H_z = eps_t e / n_eff`,
   and for TM `beta^2 e = (k0^2 eps_z + D_b D_f) e` with `H_t = -n_eff e`.
   It returns a `lightsim.Mode` whose `profile` feeds `ModeSource` /
   `ModeMonitor` unchanged. It works for any 2-D simulation, whatever its
   invariant axis, and `lightsim.solve_modes` calls it for one.
6. **S-parameters.** `write_sparameters_2d` follows `sparams.write_sparameters`
   step by step (extended ports, mode source on the input, mode monitors on
   every port, `S[j, i] = a_out_j / a_in_i`).

This is the **plain effective-index method**, not the variational reduction
of varFDTD: one non-dispersive effective index per region, evaluated at the
mean wavelength, with an isotropic effective medium.

## API

```python
import gdsfactory as gf, numpy as np
from lightsim import eim

gf.gpdk.PDK.activate()
c = gf.components.mmi1x2()
sp = eim.write_sparameters_2d(c, wavelengths=np.linspace(1.5e-6, 1.6e-6, 11),
                              input_ports=["o1"], device="cpu")
abs(sp["o2@0,o1@0"]) ** 2
```

| function | purpose |
| --- | --- |
| `slab_modes(indices, thicknesses, wavelength, polarization="te")` | guided n_eff of a multilayer, highest first; `indices = [n_bottom, n_1..n_m, n_top]`, `thicknesses = [d_1..d_m]` |
| `slab_neff(n_core, thickness, wavelength, n_substrate, n_cladding=None, polarization="te", mode=0)` | three-layer slab; raises if the mode is cut off |
| `effective_indices(component, wavelength, layer_stack=None, material_index=None, polarization="te", mode=0, background="sio2")` | `(neff per level, n_background, layers)` |
| `effective_index_simulation(component, wavelengths, layer_stack=None, material_index=None, polarization="te", slab_mode=0, steps_per_wavelength=20, xy_margin=1um, port_extension=2um, background="sio2", run_time_factor=6, shutoff=1e-5, z_cell=1um, pml_layers=12, level_indices=None, background_index=None)` | `(sim, ports, info)`: 2-D `Simulation` without sources / monitors; `level_indices` / `background_index` replace the slab indices (with bulk indices: the plain 2-D cross-section of the layout) |
| `solve_port_mode(sim, axis, position, min, max, freq, polarization="te", mode_index=0, num_modes=None)` | lateral mode(s) on a port line of a 2-D simulation |
| `write_sparameters_2d(component, wavelengths=None, layer_stack=None, material_index=None, polarization="te", steps_per_wavelength=20, port_offset=0.5um, port_lateral_margin=1um, mode_index=0, device="auto", input_ports=None, return_results=False, **kwargs)` | S-matrix dict, same keys as `sparams.write_sparameters`, plus `slab_neff` and `n_background` |

Material indices come from `lightsim.gds.MATERIAL_INDEX` (override with
`material_index`). The module is imported as `lightsim.eim`; it is not
re-exported from the package root.

## Validation

`python/tests/test_eim.py` (CPU, about 1 min):

| check | result |
| --- | --- |
| slab n_eff vs analytic symmetric-slab relation `kappa tan(kappa t/2) = r gamma`, Si in SiO2, t = 90 / 150 / 220 / 400 nm, TE and TM | agreement to better than 1e-6 (test bound 1e-4); 220 nm: TE 2.847782, TM 2.053320 |
| multilayer consistency (split core, redundant cladding layer) | n_eff unchanged to 1e-9 |
| rib layer stack (generic PDK `rib`) | core 2.8478, slab90 2.1033, background 1.444; rib drawn over slab |
| straight 500 nm waveguide, 1.5 / 1.55 / 1.6 um, 16 cells per wavelength | TE: \|S21\|^2 = 0.9996 / 1.0000 / 0.9995, \|S11\|^2 <= 2e-4; TM: \|S21\|^2 >= 0.9993 |
| port mode n_eff vs analytic lateral slab | TE 2.4916 vs 2.4914, TM 1.8483 vs 1.8470 |
| gdsfactory `mmi1x2`, generic PDK, 1.55 um | see below |

MMI 1x2 (220 nm Si in SiO2, input `o1`, 11 wavelengths 1.5-1.6 um, 18 cells
per wavelength, CPU, 10 threads, same machine and same script):

| | 2.5-D EIM | full 3-D |
| --- | --- | --- |
| \|S21\|^2 at 1.50 / 1.55 / 1.60 um | 0.4926 / 0.4964 / 0.4926 | 0.4933 / 0.4950 / 0.4837 |
| \|S31\|^2 at 1.50 / 1.55 / 1.60 um | 0.4926 / 0.4964 / 0.4926 | 0.4931 / 0.4948 / 0.4836 |
| \|S11\|^2 | <= 2e-4 | <= 1e-4 |
| port n_eff (o1) | 2.497 | 2.453 |
| grid | 132 628 cells x 7 000 steps | 11 389 896 cells x 12 200 steps |
| wall time | 6.3 s | 435 s |

At 1.55 um the 2.5-D split differs from 3-D by 0.3 %; over the band the
largest difference is 1.8 %, at 1.60 um, where the 3-D transmission rolls
off and the non-dispersive 2.5-D model does not follow. The test asserts 7 %
against the recorded 3-D value 0.49 (set `LIGHTSIM_EIM_3D=1` to rerun the
3-D reference inside the test).

**Speed-up: about 70x in wall time** (435 s / 6.3 s), 150x in cell-steps; the
difference is the lower CPU throughput on the small 2-D grid (149 vs 320
Mcells/s). Other jobs were running on the machine during both runs, so the
wall times are indicative.

The two outputs o2 and o3 agree to 6e-5 in 2.5-D and 2e-4 in 3-D: the
automatic mesh is mirror symmetric about the MMI axis, because it is built
outwards from the centre of the region.

## Limitations

- **Plain EIM, not variational.** Accuracy degrades when the vertical mode
  profile changes strongly between regions: deep-etched narrow waveguides
  (the 2.5-D port n_eff of a 500 x 220 nm strip is about 2.49, the 3-D value
  about 2.44), small bend radii, and anything that scatters out of plane
  (grating couplers, vertical tapers, slab-mode leakage) is not modelled:
  out-of-plane radiation loss is absent by construction.
- **No dispersion of the effective index.** The slab n_eff is evaluated at
  the mean wavelength and used as a constant, so the group index and the
  wavelength dependence of interferometric devices are off away from the
  centre. For wide bands run several narrow bands.
- **One slab mode per run**, no TE / TM coupling, no polarization
  conversion; sidewall angles are ignored.
- **Layer handling.** Each patterned level is reduced on its own ("wafer
  layers + that level"); overlapping levels resolve to the higher effective
  index. That is right for same-material levels (rib over slab) but not for
  stacked levels of different materials (e.g. nitride above silicon), which
  need a combined stack; build that case with `slab_modes` and a custom
  `material_index` / layer stack.
- **Lossless, isotropic layers** in the slab solver; a level whose slab mode
  is cut off falls back to the background index.
- Ports must be Manhattan (as in `sparams`).
- Not run on GPU in validation; the 2-D grids are small enough (about 1e5
  cells) that the CPU engine is not at its best throughput either.
