# Eigenmode expansion

`lightsim.eme` computes the S-matrix of a device that changes along its
propagation axis (tapers, MMIs, directional couplers, gratings) from the
modes of its cross-sections. It is the counterpart of meow
(`gplugins.meow`) and runs on a CPU in seconds to minutes where a 3-D FDTD
run needs a GPU.

## Method

The device is a sequence of **sections**, each a
[cross-section](mode-solver.md) and a length, all on one transverse grid.

1. In each section the field is a sum of the section's modes (forward and
   backward), from the finite-difference [mode solver](mode-solver.md).
   Identical cross-sections are solved once.
2. A mode crosses a section as `exp(i beta L)`.
3. At the junction of two sections the transverse fields are matched with
   the unconjugated overlaps `O_ij = <E_i^L, H_j^R> = ∫ (E_i x H_j) · n dA`
   (modes normalised to `<E_i, H_i> = 1`). Testing E continuity with the
   modes of the far side and H continuity with those of the near side
   gives, for light from the left,
   `T = 2 O^T (I + O O^T)^-1`, `R = (I + O O^T)^-1 (I - O O^T)`
   (and the same with `<E_i^R, H_j^L>` from the right). For lossless
   propagating modes this junction conserves power exactly for any number
   of modes, and with one mode per side it reduces to the Fresnel
   coefficients. (The two-overlap formula of CAMFR and meow,
   `T = 2 (O_LR + O_RL^T)^-1`, gave up to 0.6 % more power out than in at a
   small width step with up to 16 modes in our tests.)
4. Junctions and sections are combined with the Redheffer star product;
   `N` identical periods by repeated squaring (`log2 N` products).
   Reflections are kept everywhere.

The S-matrix is first obtained in the mode bases of the two end sections.
For named ports, `sparameters` projects it onto port modes: the end cut
restricted to each port's guide (the shapes within half the distance to
the neighbouring port), with the projection
`C[p, m] = (<E_m, H_p> + <E_p, H_m>) / 2`.

## From a gdsfactory component

```python
import numpy as np
import gdsfactory as gf
from lightsim import eme

gf.gpdk.PDK.activate()
taper = gf.components.taper(length=4, width1=0.5, width2=1.0)
sp = eme.sparameters(taper, wavelengths=[1.55e-6], num_modes=6, cell_length=0.5e-6,
                     step=40e-9, margin=0.8e-6, v_margin=0.6e-6,
                     material_index={"si": 3.476, "sio2": 1.444})
print(f"|S21|^2 {abs(sp['o2@0,o1@0'][0])**2:.4f}  |S11|^2 {abs(sp['o1@0,o1@0'][0])**2:.1e}")
print(sp["sections"], "sections,", sp["distinct_cross_sections"], "distinct")
```

```text
|S21|^2 0.9996  |S11|^2 5.6e-06
10 sections, 8 distinct
```

`component_sections` cuts the component along x: the extent between the
west and east ports is split at every polygon edge normal to x (grating
teeth, the ends of an MMI body) and each interval into equal cells of at
most `cell_length`; each cell takes the cut at its centre (a staircase
for tapers and bends), and identical neighbours are merged. Ports must face
west (180 degrees) or east (0 degrees). The keys follow lightsim's
S-parameter files: `"o2@0,o1@0"` is the amplitude leaving `o2` for unit
amplitude entering `o1`.

`EMEComponent(component, layer_stack, wavelength=1.55, num_modes=4,
cell_length=0.5, ...)` has the interface of `gplugins.meow.MEOW`
(micrometres, `compute_sparameters()` returning `"o1@i,o2@j"` keys for the
modes of the two end cross-sections, ordered in, out as meow and SAX).

## By hand, and periodic sections

```python
from lightsim.xsection import CrossSection
from lightsim.eme import EME

def strip(w):
    xs = CrossSection(background=1.444)
    xs.add_rect(-w / 2, 0, w / 2, 0.22e-6, 3.476)
    xs.set_mesh(np.arange(-1.2e-6, 1.2e-6 + 1e-12, 40e-9), np.arange(-0.6e-6, 0.82e-6 + 1e-12, 40e-9))
    return xs

lead, narrow, wide = strip(0.5e-6), strip(0.45e-6), strip(0.55e-6)
grating = EME([(lead, 0.0), ("period", [(narrow, 0.159e-6), (wide, 0.159e-6)], 200), (lead, 0.0)], num_modes=4)
for lam in (1.50e-6, 1.55e-6):
    S = grating.solve(lam)
    print(f"{lam * 1e9:.0f} nm: R {abs(S.s11[0, 0])**2:.3f}  T {abs(S.s21[0, 0])**2:.3f}")
```

```text
1500 nm: R 0.000  T 0.999
1550 nm: R 0.818  T 0.179
```

A section `("period", sections, N)` repeats a sub-list `N` times: one
unit is built, closed with the junction from its last section to its first,
and raised to the power `N - 1` by squaring. `EME.solve` returns an
`SMatrix` with blocks `s11`, `s21`, `s12`, `s22` (outgoing mode index
first) and fills `timings` (seconds in mode solves, overlaps, cascade).

## Accuracy

- **1-D stack**: homogeneous sections (one plane wave per polarisation)
  reproduce the exact transfer matrix of a multilayer, including 37
  quarter-wave periods by squaring, to 1e-13 (`test_eme.py`).
- **Convergence** of the 10 µm silicon taper from 0.5 to 2 µm
  (`gf.components.taper`, 1.55 µm, n 3.476 / 1.444, 25 nm grid): |S21|²
  0.9737–0.9892 with 1 µm cells (8–32 modes), 0.9948–0.9961 with 0.5 µm,
  0.9961–0.9964 with 0.25 µm and 0.9960–0.9963 with 0.125 µm cells. The
  cell length is the parameter to check; with cells of 0.25 µm, 8, 16 and
  32 modes agree to 3e-4.
- **Junction**: power is conserved to 1e-15 for propagating modes;
  reciprocity (`S21 = S12^T`) holds to the truncation of the mode sets,
  1e-4 for the fundamental mode of a 0.5 to 0.65 µm step with 6 modes.

## Devices and run times

EME of gdsfactory components of the generic PDK (220 nm silicon in oxide,
n 3.476 / 1.444), 25 nm grid, cells of at most 0.25 µm, five wavelengths
from 1.50 to 1.60 µm (51 for the grating), one laptop CPU process:

| Device | Sections (distinct) | Modes | Run time | Result |
|---|---|---|---|---|
| `taper(length=10, width1=0.5, width2=2.0)` | 42 (40) | 16 | 101 s | abs(S21)^2 0.9956–0.9966 |
| `mmi1x2()` | 83 (81) | 24 | 321 s | abs(S21)^2, abs(S31)^2 0.484–0.497 each |
| `coupler(gap=0.2, length=10)` | 83 (41) | 16 | 498 s | bar 0.784 to 0.567, cross 0.179 to 0.378 from 1.50 to 1.60 µm |
| Bragg grating, 60 periods of 0.45/0.55 µm x 159 nm | 124 (3) | 8 | 50 s | peak reflection 0.456 near 1.55 µm |

Almost all the time goes into mode solves (one per distinct cross-section
and wavelength); overlaps and cascading take under 1 %.

## Compared with meow and with 3-D FDTD

meow 0.15 through `gplugins.meow.MEOW` (its defaults: 100 x 100 points
over the window, i.e. 40 nm laterally and 80 nm vertically for the generic
stack) on the 10 µm taper above gives abs(S21)^2 0.962 (4 modes, 0.5 µm
cells, 4 s), 0.944 (8 modes, 0.5 µm, 11 s) and 0.943 (8 modes, 0.25 µm,
20 s); lightsim gives 0.996. On gplugins' own test case
(`taper_cross_section_linear`, stack slab90/core/box/clad) meow gives
0.966 (4 and 8 modes, 0.5 µm cells) and 0.980 (8 and 16 modes, 0.25 µm
cells). meow is faster per run at its default resolution; its coarse
vertical grid (three or four points across the core) is the likely cause
of the difference, which we have not resolved further.

The comparison with lightsim's 3-D FDTD (`lightsim.sparams.write_sparameters`
on the same components and indices) has not been run yet: the job is
queued for a GPU.

## When to use FDTD instead

EME assumes the device can be cut into sections whose modes describe the
field. It handles slowly varying and abruptly stepped devices along one
axis, with reflections, and periodic devices cheaply. Use 3-D FDTD for
devices that radiate strongly or scatter out of the plane (grating
couplers, sharp corners of photonic crystals), for devices without a
propagation axis (rings as a whole, crossings, Y-junctions with ports on
several sides), for broadband results with strong dispersion (EME solves
the modes again at every wavelength), and to check a final design. Bends
in couplers are staircased into thin straight cells, which is accurate only
for gentle curvature.

## Limits

- Ports must face west or east; the device runs along x.
- Every section uses the same grid; very wide devices need many cells.
- The modes come from closed windows (no PML unless `pml=` is given), so
  radiation is represented by box modes; light scattered to large angles
  is poorly described.
- Mode sets are truncated: check convergence in `num_modes` and
  `cell_length`.
- No sidewall angles in the cuts (vertical walls, as the 3-D import).
