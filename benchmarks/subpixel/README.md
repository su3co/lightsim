# Sub-pixel weighting: cell average against tent

`subpixel_weighting="tent"` (see [Simulation setup](../../docs/guide/simulation.md))
replaces the average over a field sample's own cell by a tent-weighted average
that reaches to the neighbouring samples. These scripts measure what that
changes, on bodies and quantities with a converged or analytic reference, with
the body moved through a cell (shift 0, 1/8, 1/4, 3/8, 1/2 of a cell along
every transverse axis). The setting is off by default; with it off the
rasteriser gives the same bits as before the setting existed.

| script | what it measures | reference | result file |
|---|---|---|---|
| `placement_study.py`, `placement_table.py` | effective index of the port mode on the FDTD grid: silicon wire (TE0, TM0), nitride wire, two coupled wires (splitting), a silicon wire with a fitted Lorentz pole, a slab in a 2-D simulation; cells of 40 to 10 nm | finite-difference mode solver on conforming meshes of 20 / 10 / 5 nm, Richardson limit; the slab: its dispersion relation | `results/placement.jsonl` |
| `disc_resonance.py`, `disc_table.py` | resonance frequency of a silicon disc in oxide in a 2-D FDTD run (a curved surface, drawn as a `Cylinder`), E along the axis and E in the plane | complex root of the Bessel / Hankel matching condition (the fitted quality factors, 307 and 205 to 210 at 20 nm, agree with the analytic 302 and 215) | `results/disc.jsonl` |
| `wire_fdtd.py`, `wire_fdtd_table.py` | index of the silicon wire's TE0 from the phase between two mode ports of a 3-D FDTD run, cells of 60 to 10 nm, wire moved along z | mode solver limit 2.43750 | `results/wire_fdtd_gb10.jsonl` (one GB10) |
| `validate_compare.py` | two outputs of `lightsim validate`: which values move, error of each check against its own reference | the validation cases' analytic references | `results/validation_gb10_cell.txt`, `results/validation_gb10_tent.txt` |
| `setup_cost.py` | wall time of a one-step `lightsim run` on the CPU with each weighting (grid, rasterisation, coefficients), five scenes of 13.2 M cells | – | `results/setup_cost_*.jsonl` |

## Results

Error ranges over the five placements (index error; the disc: relative error of
the resonance frequency).

| case | cell size | `"cell"` | `"tent"` |
|---|---|---|---|
| slab in 2-D, E parallel to the faces | 40 nm | -5.0e-3 to +3.3e-3 | -1.0e-2 to -8.6e-3 |
| | 20 nm | -2.5e-3 to +4.1e-3 | -2.5e-3 to -2.4e-3 |
| | 10 nm | -6.4e-4 to +1.0e-3 | -6.4e-4 to -6.2e-4 |
| slab in 2-D, E normal to the faces | 40 nm | -1.2e-2 to +2.1e-2 | -1.6e-2 to +7.6e-4 |
| | 20 nm | +2.7e-4 to +7.7e-3 | -2.1e-3 to -9.7e-4 |
| | 10 nm | +7.4e-5 to +1.9e-3 | -4.0e-4 to -2.5e-4 |
| silicon wire, TE0 | 40 nm | -1.4e-3 to +2.0e-3 | -1.3e-2 to -1.0e-2 |
| | 20 nm | -2.0e-3 to +5.1e-3 | -3.5e-3 to -3.0e-3 |
| | 10 nm | -4.8e-4 to +1.0e-3 | -9.7e-4 to -6.9e-4 |
| silicon wire, TM0 | 40 nm | -1.1e-2 to +2.3e-2 | -1.3e-2 to -8.2e-4 |
| | 20 nm | -6.5e-4 to +8.0e-3 | -2.1e-3 to -7.3e-4 |
| | 10 nm | -1.3e-4 to +1.8e-3 | -4.8e-4 to -2.6e-5 |
| nitride wire, TE0 | 40 nm | -6.9e-4 to +1.3e-3 | -7.9e-4 to -6.9e-4 |
| | 20 nm | -1.0e-4 to +2.0e-4 | -2.0e-4 to -1.7e-4 |
| | 10 nm | -2.5e-5 to +4.9e-5 | -5.1e-5 to -4.3e-5 |
| nitride wire, TM0 | 40 nm | -5.0e-4 to +1.4e-3 | -6.7e-4 to -4.7e-4 |
| | 20 nm | -8.6e-5 to +3.1e-4 | -1.7e-4 to -1.3e-4 |
| | 10 nm | -2.3e-5 to +8.0e-5 | -4.3e-5 to -3.0e-5 |
| coupled wires, splitting (0.0229) | 40 nm | -4.7e-4 to +7.5e-4 | +1.2e-3 to +2.3e-3 |
| | 20 nm | -2.1e-4 to +3.4e-4 | +1.1e-4 to +4.1e-4 |
| | 10 nm | -1.7e-4 to +1.2e-4 | +1.8e-5 to +1.1e-4 |
| disc, E along the axis | 40 nm | -8.1e-3 to -7.0e-3 | -6.2e-3 to -6.1e-3 |
| | 20 nm | -2.0e-3 to -1.5e-3 | -1.49e-3 to -1.49e-3 |
| | 10 nm | -5.5e-4 to -2.8e-4 | -3.7e-4 to -3.7e-4 |
| disc, E in the plane | 40 nm | -5.3e-3 to -4.0e-3 | -2.5e-3 to -1.7e-3 |
| | 20 nm | -5.9e-4 to -3.2e-4 | +5.8e-4 to +6.0e-4 |
| | 10 nm | +2.1e-4 to +2.8e-4 | +6.6e-4 to +6.6e-4 |

The silicon wire with a fitted Lorentz pole (a transparent dispersive
dielectric, smoothed through the mixed voxels) gives the plain silicon wire's
values at every cell size, weighting and placement.

Index of the silicon wire's TE0 in a 3-D FDTD run (one GB10, single precision),
error against 2.43750 at z shifts of 0, 1/4 and 1/2 cell:

| cell size | `"cell"` | `"tent"` |
|---|---|---|
| 60 nm | +2.9e-02, +4.2e-02, +4.1e-02 | +1.5e-02, +1.8e-03, +1.4e-04 |
| 40 nm | +1.6e-02, +2.2e-02, +1.3e-02 | +4.1e-03, -3.2e-04, -1.3e-03 |
| 30 nm | +1.2e-02, +9.7e-03, +5.6e-03 | +1.6e-03, +1.2e-03, -9.4e-04 |
| 20 nm | +8.7e-03, +3.6e-03, +2.1e-03 | +4.2e-04, +3.1e-04, -4.3e-04 |
| 15 nm | +2.7e-03, +2.5e-03, +1.5e-03 | +3.1e-05, +8.3e-05, +4.8e-04 |
| 10 nm | +5.6e-04, +2.1e-03 | -7.1e-05, +1.4e-04 |

(10 nm: shifts 0 and 1/2 only.) Largest error over the placements: 2.2e-2, 1.2e-2, 8.7e-3, 2.7e-3 and 2.2e-3 at 40, 30, 20, 15 and 10 nm with the cell average; 4.1e-3, 1.7e-3, 4.3e-4, 4.8e-4 and 1.4e-4 with the tent.

Here the tent's error, which is negative in the cross-section, cancels most of
the grid dispersion of the propagation, which is positive. That cancellation
belongs to high-contrast guided modes; it is not a property of the weight.

Validation suite on one GB10 (`lightsim validate --device cuda:0`, then with
`--subpixel-weighting tent`): 70 of 70 cases pass with either; 500 of 764
measured values are identical; 264 values in 19 cases move, 141 of them
towards their reference and 106 away from it, none beyond its tolerance. The
cases that move by more than a tenth of their tolerance:

| case | error with `"cell"` | error with `"tent"` |
|---|---|---|
| sphere, scattered power against Mie (worst of 5 frequencies) | 0.7 % | 1.8 % |
| cylinder in 2-D, E in the plane, scattered power against Mie (worst) | 0.9 % | 1.6 % |
| grating in 2-D, diffraction efficiencies (sum of the errors of 16 checks, in tolerances) | 3.2 | 1.1 |
| Brewster angle, largest departure of the transmittance from 1 over the band | 5.6e-5 | 8.0e-6 |
| slab at oblique incidence, transmittance against Airy | 2.2e-3 | 1.7e-3 |
| Fresnel reflection at 30 degrees (Bloch boundaries) | 3.0e-2 | 2.1e-2 |
| conducting sheet between nodes of an explicit mesh, reflectance | 1.2e-2 | 1.7e-2 |

Set-up cost: wall time of `lightsim run` for one time step on the CPU (process start, grid, rasterisation of the six
components, coefficients, one step), the two weightings alternating, median and range of five runs:

| scene (400 x 300 x 110 cells) | Apple M1 Pro, 10 threads: `"cell"` | `"tent"` | GB10 CPU, 10 threads: `"cell"` | `"tent"` |
|---|---|---|---|---|
| one wire (box) | 0.19 s (0.19-0.47) | 0.21 s (0.20-0.26) | 0.81 s (0.76-0.83) | 0.82 s (0.77-0.84) |
| bus and ring (720-vertex polygons) | 0.21 s (0.20-0.23) | 0.28 s (0.27-0.29) | 0.84 s (0.80-0.85) | 0.87 s (0.86-0.89) |
| four layers that share faces over the whole plane | 0.23 s (0.23-0.24) | 0.42 s (0.40-0.45) | 0.86 s (0.80-0.87) | 1.02 s (0.99-1.02) |
| sphere | 0.35 s (0.34-0.39) | 0.30 s (0.28-0.31) | 0.96 s (0.93-0.96) | 0.99 s (0.97-1.01) |
| 96 cylinders | 0.30 s (0.29-0.32) | 0.24 s (0.23-0.25) | 0.90 s (0.88-0.94) | 0.86 s (0.82-0.89) |

The tent costs up to 0.2 s more on 13.2 M cells where every cell next to a shared face is composited from the
geometry, and is within 0.07 s of the cell average elsewhere (its sphere and cylinder paths skip the cells that lie wholly
inside or outside). The time-stepping kernels are the same for both.
