# Imported and graded index structures

`IndexGrid` is the equivalent of Lumerical's "import (n,k)" object and
index-perturbation grid attribute: a structure whose relative permittivity
is given by samples on a regular grid instead of by a single medium. Typical
uses are graded-index layers and lenses, permittivity maps exported from another tool, and
carrier-induced index perturbations of silicon modulators (`lightsim.eo`).

## What it does

`Shape::IndexGrid { min, max, dims, eps, eps_axes }` (JSON `"type": "index_grid"`)
holds `dims = [nx, ny, nz]` samples, x-major `eps[(i*ny + j)*nz + k]`, whose
nodes span the box `[min, max]` inclusive. A dimension of 1 means "uniform
along that axis". Optional `eps_axes = [eps_x, eps_y, eps_z]` (same layout)
gives a diagonal-anisotropic grid and is used instead of `eps`.

In the rasterizer (`raster.rs`) the grid is a special structure kind. For each
E component, every voxel whose **component position** lies inside the box
(half-open `[min, max)`, the same ownership rule as a staircased `Box`) gets

- `value` = trilinear interpolation of the samples at that position (each Yee
  component is sampled at its own staggered location),
- `medium` = the structure's medium index.

Outside the box nothing changes. Because the same rasterizer feeds
`plane_info`, the mode solver sees the identical permittivity.

Rules and limits:

- **The grid replaces the medium's permittivity.** The medium supplies only
  its name, conductivity `sigma` (loss), `chi2`/`chi3` and permeability. Its
  own `eps` is still what the automatic mesher uses to choose the cell size
  and what sources placed inside the box use for normalisation, so set it to
  the largest permittivity of the grid and keep sources outside the grid.
- Dispersive (pole), PEC and off-diagonal-tensor media are **rejected** for an
  index grid: the plan compiler takes `eps_inf` of those from the medium, so a
  grid permittivity would be applied inconsistently. Loss is one uniform
  `sigma` per grid (no per-voxel `k`).
- No sub-pixel smoothing of the grid's own box faces (it is already a sampled
  field): the faces are staircased to the Yee positions. To embed a grid in
  another material, add that structure first and let the grid overlap it; do
  not abut a `Box` against a grid face.
- Painter's rule as usual: structures added later cover the grid (with their
  normal sub-pixel smoothing against the grid value underneath).
- H components treat the shape as a `Box` of its medium (permeability only).
- Samples must be finite and > 0, `max > min` on every axis, and the sample
  count must equal `nx*ny*nz`; violations are reported when the simulation is
  compiled.

## API

Rust (`lightsim_core::geometry`):

```rust
Shape::IndexGrid { min: [f64; 3], max: [f64; 3], dims: [usize; 3],
                   eps: Vec<f64>, eps_axes: Option<[Vec<f64>; 3]> }
shape.grid_eps(axis, p) -> Option<f64>   // interpolated eps along `axis` at p
shape.validate() -> Result<(), String>
```

`bbox` and `contains` behave as for `Box`.

Python (`lightsim.spec`):

```text
IndexGrid(min, max, eps=None, *, n=None, axes=None, eps_axes=None, n_axes=None)
```

`eps` or `n` (refractive index, squared internally) is a real scalar or a
1-D/2-D/3-D numpy array, padded to 3-D with uniform axes. By default a 1-D
array varies along x and a 2-D array along (x, y); `axes` selects the
simulation axes, e.g. `axes=(1, 2)` for a (y, z) cross-section map.

```python
import numpy as np, lightsim as ls
sim = ls.Simulation(min=[0, 0, 0], max=[5e-6, 0.5e-6, 0.5e-6], mesh=ls.UniformMesh(25e-9), run_time=50e-15)
x = np.linspace(0, 1, 41)
ramp = ls.IndexGrid([2e-6, -1, -1], [3e-6, 1, 1], 1 + 3 * x)        # eps 1 -> 4 along x
sim.add_structure(ramp, ls.Medium("graded", eps=4.0))
```

Python (`lightsim.eo`), free-carrier plasma dispersion in silicon:

```python
from lightsim import eo
dN = dP = np.full((21, 11), 1e17)                # carrier densities on a (y, z) map, cm^-3
lo, hi = [3.5e-6, 0.1e-6, 0.1e-6], [4.5e-6, 0.4e-6, 0.3e-6]   # box of the map
dn, dalpha = eo.soref_bennett(dN, dP, wavelength=1550e-9, model="soref1987")  # cm^-3 -> (-, cm^-1)
grid = eo.carrier_index_grid(lo, hi, dN, dP, wavelength=1550e-9, n_base=None,
                             axes=(1, 2), model="soref1987", mask=None)        # IndexGrid of (n_base + dn)^2
med  = eo.carrier_medium("si_doped", dN, dP, wavelength=1550e-9, weights=None) # sigma from <dalpha>
sim.add_structure(grid, med)
eo.alpha_to_sigma(100.0, 3.476)     # sigma = n * alpha * eps0 * c, alpha in 1/m
```

Coefficients (`dn = -(A_e dN^p_e + A_h dP^p_h)`, `dalpha = B_e dN^q_e + B_h dP^q_h`,
densities in cm^-3, absorption in cm^-1):

| model | wavelength | A_e, p_e | A_h, p_h | B_e, q_e | B_h, q_h |
|---|---|---|---|---|---|
| `soref1987` | 1550 nm | 8.8e-22, 1 | 8.5e-18, 0.8 | 8.5e-18, 1 | 6.0e-18, 1 |
| `soref1987` | 1310 nm | 6.2e-22, 1 | 6.0e-18, 0.8 | 6.0e-18, 1 | 4.0e-18, 1 |
| `nedeljkovic2011` | 1550 nm | 5.40e-22, 1.011 | 1.53e-18, 0.838 | 8.88e-21, 1.167 | 5.84e-20, 1.109 |
| `nedeljkovic2011` | 1310 nm | 2.98e-22, 1.016 | 1.25e-18, 0.835 | 3.48e-22, 1.229 | 1.02e-19, 1.089 |

Sources: R. A. Soref and B. R. Bennett, IEEE J. Quantum Electron. 23, 123
(1987) (fitted form as in Reed et al., Nature Photon. 4, 518 (2010));
M. Nedeljkovic, R. Soref and G. Z. Mashanovich, IEEE Photon. J. 3, 1171 (2011).
The grid carries the real index change only; the free-carrier absorption is
represented by one conductivity per grid (average of `dalpha`, optionally
weighted by the mode intensity).

## Validation

Rust case `index_grid` (`validation.rs`, test in `tests/physics.rs`, registered
in `all_cases()`), CPU engine, plane-wave column with periodic transverse
boundaries, lambda0 = 1 um, band 0.85-1.15 f0:

| check | measured | reference | error |
|---|---|---|---|
| constant grid (eps 4, 3x1x2 samples) vs `Box`, faces off-grid, `subpixel = false`: max dE / peak of the DFT field | 0 | 0 | exactly 0 (tol 1e-6) |
| same, transmitted flux max dT/T | 0 | 0 | exactly 0 (tol 1e-6) |
| linear eps ramp 1 -> 4 over 1 um into an eps = 4 half-space, dx = lambda/60: T at 5 frequencies | 0.9947 ... 0.9963 | transfer matrix (4000 sub-layers) 0.9957 ... 0.9975 | 0.10 - 0.18 % rel (tol 1 %) |
| same, R | 2.48e-3 ... 4.33e-3 | 2.45e-3 ... 4.31e-3 | <= 4.8e-5 abs (tol 2e-4) |
| free-standing ramp 1 -> 4 with abrupt back face: T | 0.847 ... 0.914 | 0.849 ... 0.916 | 0.19 - 0.46 % rel (tol 1 %) |
| same, R | 0.085 ... 0.152 | 0.084 ... 0.151 | <= 2.1e-3 abs (tol 3e-3) |

Unit tests: `geometry::tests::index_grid_interpolation` (trilinear exactness,
serde round trip, size validation) and
`raster::tests::index_grid_matches_staircased_box_and_interpolates` (bit-identical
`value`/`medium` to a staircased `Box` for Ex/Ey/Ez, per-component sampling of a
ramp, grid value kept under a lossy medium, later structures painted on top,
half-open upper face).

Python `python/tests/test_index_grid.py`: builder padding / axis mapping /
errors; a bilinear (y, z) map read back through the core's `plane_info` to
1e-5; constant grid vs `Box` flux through the FDTD run (1e-6); Soref-Bennett
at 1e18 cm^-3 (1550 nm: electrons dn = -8.8e-4, dalpha = 8.5 /cm; holes
dn = -2.1e-3, dalpha = 6.0 /cm; 1310 nm holes dn = -1.5e-3); carrier map ->
grid values and absorption recovered from the core's complex permittivity
(14.5 /cm for 1e18 / 1e18).

Not covered: GPU run of the case (the kernels are untouched; the grid only
changes the rasterized coefficients, so the case is backend-independent and is
picked up by any runner iterating `all_cases()`).
