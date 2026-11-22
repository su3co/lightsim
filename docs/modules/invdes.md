# Inverse design

Gradient-based topology optimisation with adjoint back-propagation through the
S-matrix (cf. Lumerical lumopt, Meep adjoint, FDTDX, Tidy3D inverse design).
The module answers one question efficiently: *how does every entry of a
multi-port, multi-frequency S-matrix change with every design voxel?* Any real
figure of merit of the S-matrix can then be back-propagated to the design
density, by hand, with the bundled objectives, or by PyTorch / JAX.

## What it does

### S-matrix and its vector-Jacobian product

`S[m, n, k]` is the complex amplitude of output channel `m` (a waveguide mode
leaving through a port) per unit excitation `n` (a mode launched into a port,
or any source) at frequency `k`.

- **N forward runs**, one per excitation: design-region field at all K
  frequencies (running DFT) and the mode amplitudes at every output port.
- **M adjoint runs**, one per output channel: that mode launched *backwards*
  into the device with a broadband pulse, same design-region DFT.

By Lorentz reciprocity (the formula of `lightsim.adjoint`, whose
`amplitude_sensitivity` kernel is shared), for each Yee E component `c`

    dS[m, n, k] / d eps_c(r) = (i w_k eps0 / 4) E_adj,m(r) E_fwd,n(r) dV_c(r)

with `E_fwd,n` per unit excitation and `E_adj,m` per unit launched amplitude.
N + M simulations give the whole Jacobian, so the gradient of any `L(S)` costs
no extra run. The N + M runs are independent and are dispatched concurrently
over a device list (`devices=["cuda:0", "cuda:1"]`, one run at a time per
entry; threads, the core releases the GIL).

**Normalisation.** With `normalize="monitor"` (default) every mode source is
calibrated in its own run: two mode monitors 2 and 4 cells downstream measure
the launched amplitude `a1`, `a2`; the amplitude at the source plane is
`a1 * (a1 / a2)` (geometric extrapolation, numerical dispersion included). `S`
is then per unit *incident mode amplitude*, as in `lightsim.sparams`, and the
adjoint field is per unit launched amplitude referred to the output-monitor
plane, which is what reciprocity needs. `normalize="source"` divides by the
source-signal spectrum instead (the convention of
`lightsim.adjoint.mode_power_gradient`; the amplitude is then low by what
the mode source loses on the mesh, about 1 % at 12 cells per wavelength). A port that is both excited and observed (a
reflection entry) is read 2 cells inside and referred back to the port plane.

**Complex-gradient convention.** For a real loss `L` the upstream gradient is

    g = dL/dRe(S) + i dL/dIm(S)        so that     dL = Re(conj(g) dS).

This is what `torch.Tensor.grad` holds for a complex tensor. JAX cotangents
are `conj(g)`; `lightsim.invdes.jax` conjugates so each framework sees its own
convention. `SMatrixSolution.vjp(g)` returns `dL/d eps` on the grid samples,
`Re sum_mnk conj(g_mnk) dS_mnk/d eps`.

### Parametrisation

    rho in [0,1] --conic/Gaussian filter--> rho~ --tanh projection(beta, eta)--> rho^
    eps = eps_base + sum_l coeff_l u_l,    u_0 = rho^_0,  u_1 = rho^_0 rho^_1

- The density lives on a 2-D grid: one etch layer extruded through the slab.
- `Design.slab` (3-D, full etch): one index grid over `[zmin, zmax)`.
- `Design.partial_etch` (3-D, two levels): two stacked index grids; field 0 is
  the lower part of the slab, field 1 the upper part, and `u_1 = rho^_0 rho^_1`
  keeps the upper level from overhanging.
- `Design.effective_index` (2.5-D): one z-invariant grid from `n_lo^2` to
  `n_hi^2` (slab effective indices from `lightsim.eim.slab_neff`), optionally a
  third level `n_partial`.
- Filter: normalised convolution `(K * rho) / (K * 1)`; the minimum feature is
  about twice the radius. Projection: `(tanh(b e) + tanh(b (x - e))) /
  (tanh(b e) + tanh(b (1 - e)))`; beta is raised during the optimisation
  (continuation). `eta +- d` gives eroded / dilated variants.
- `fixed_solid` / `fixed_void` masks pin pixels (waveguide stubs); `symmetry`
  mirrors the density.

**Shapes.** `ShapeDesign` (see [Shape optimisation](shapes.md)) replaces
the density by the parameters of polygons with movable vertices or of level
sets, drawn onto the same grid with an exact transpose; the rest of the
chain, the engines and the optimiser are unchanged.

**Injection and its exact transpose.** The permittivity enters the solver as
`IndexGrid` structures appended after the base structures. The rasteriser
(`raster.rs`) gives every Yee E component whose *own staggered position* lies
in the half-open box `[min, max)` the clamped trilinear interpolation of the
samples (`Shape::grid_eps`); nothing else changes. `grid_weights` builds the
same interpolation as separable matrices per component and the gradient is
pulled back with their transpose, then through projection and filter
(`Design.vjp`). `python/tests/test_invdes.py` checks the sampling against the
core's `plane_info` and the whole chain against finite differences.

### Engines

`SMatrixProblem(..., engine=...)` selects how S and its vector-Jacobian
product are computed. The API (`solve`, `SMatrixSolution.S` and `vjp`,
`InverseDesign`, `ShapeDesign`) and S are the same for all three;
`LIGHTSIM_INVDES_ENGINE` sets the default (`"python"`).

- `"python"` drives every run from Python: one forward run per input and one
  adjoint run per output, each a fresh simulation; the design-region fields
  come back to the host and the product `(i w eps0 / 4) E_adj E_fwd dV` and
  the transposed sampling are NumPy.
- `"native"` computes the same S and the same product in the persistent Rust
  session of [`lightsim.autodiff`](autodiff.md): the scene is rasterised
  once, a design update uploads only the design box, the design-region
  fields stay on the device, where the product is formed, and the
  transposed sampling is the rasteriser's own stencil. It agrees with
  `"python"` to rounding (`python/tests/test_autodiff.py`).
- `"reverse"` runs one forward simulation per input and, for `vjp`, one
  reverse pass per input through the time loop
  (`lightsim.autodiff.ReverseSMatrix`). The objective of each pass is S
  itself: the DFT of the fields on the port planes, the mode overlap and the
  normalisation by the launched amplitude (`normalize="monitor"`), with the
  cotangent `Re sum conj(g) dS` fed back into the recorded planes at every
  step. The gradient is the exact derivative of the discrete scheme,
  including the dependence of the launched amplitude on the design. The
  forward run stops on the energy criterion of an ordinary run (the
  `shutoff` of `sim`), so S is the S of the other engines (same mode
  overlap, unit-power ports, normalisation and DFT; 4e-8 against `"native"`
  in 3-D, the round-off of single precision).

`"python"` and `"native"` use the continuous adjoint formula on the Yee
grid. Its error against the discrete scheme is a smooth pattern of a few
per cent of the local value (see
[accuracy at interfaces](autodiff.md#accuracy-at-interfaces)): about 1 % on
the derivatives along a random density direction, but it does not cancel
where many samples move together, i.e. on a boundary. `"reverse"` has no
such error.

| | `python` | `native` | `reverse` |
| --- | --- | --- | --- |
| runs per gradient (N inputs, M outputs) | N forward + M adjoint | N forward + M adjoint | N forward + N reverse passes |
| time per gradient, 3-D partial etch (1 x 1, 2 frequencies, 4 x 2.4 x 1.8 um at 40 nm), RTX PRO 6000 | 0.59 s | 0.25 s | 1.11 s |
| peak GPU memory of the process, same scene | 642 MiB | 642 MiB | 2440 MiB |
| 3-D partial etch, density, along the gradient | 4.0 % | 4.0 % | 4e-5 |
| same, along a random density direction | 25 % (a small derivative) | 25 % | 2.5e-3 (FD limit) |
| 3-D polygon vertices (`test_shapes.py`) | 0.2 - 4.0 % | same as `python` | 0.00 - 0.10 % |
| 2.5-D polygon vertices | 0.09 - 0.67 % | same | 0.01 - 0.11 % |
| several GPUs per run (`"cuda:0,1"`) | yes | no | no |
| `yee_vjp`, `jacobian`, extra field monitors | yes | yes (no monitors) | no |

Errors are against central finite differences of the simulated S, relative
to the largest derivative. A reverse pass costs about 3 forward runs (the
transposed step with CPML runs on the split kernels, see the
[reverse-mode limits](autodiff.md#limits)), and the engine keeps the
recorded absorbing shell and port planes of every step on the device; the
memory grows with the number of steps and the area of the PML and the port
planes, not with the design. With a list of devices the inputs of
`"reverse"` run concurrently, one per entry.

When to switch: use `"native"` while the design is grey and the steps are
large, where a 1 % gradient error does not matter and its runs are 4 times
cheaper; use `"reverse"` once the design is binary, for the final stages of
a projection schedule, and for shape parameters (`ShapeDesign`), whose
derivatives are sums along a boundary where the continuous formula is off
by a few per cent. Use `"python"` when each run must be split over several
GPUs or when field monitors are needed.

`"reverse"` refuses, with an error that names the engine to use instead:
media with off-diagonal permittivity or oriented poles and Bloch boundaries
(`"native"` or `"python"`), a device entry that names several GPUs
(`"python"`), design grids inside a PML, and absorbers too strong to be
stepped backwards (the reconstruction check of reverse mode: a conductor of
3e4 S/m, long runs through lossy media; use `"native"`). 2-D and 2.5-D
simulations are supported.

### Optimiser

`optimize(fn, rho0, iterations, beta=..., method="adam" | "lbfgs", ...)`:
projected Adam (moments restart when beta changes; `reset_on_beta=False`
keeps them, which is gentler on narrow-band designs) or scipy L-BFGS-B per beta
stage; `beta_schedule(start, factor, every, maximum)`; binarisation metric
`1 - mean(4 rho^ (1 - rho^))`; npz checkpoint / resume with the history (FOM,
beta, binarisation, seconds per iteration, simulation seconds).
`InverseDesign(problem, design, objective, robust=d)` adds eroded and dilated
variants (three times the runs, mean objective).

Objectives (`lightsim.invdes.objectives`, each returns `(value, g)`):

- `matrix_fidelity(S, T)`: `mean_k |<T, S_k>|^2 / ||T||^4`, equal to 1 exactly
  when `S_k = exp(i phi_k) T` (normalised fidelity times throughput);
  `matrix_metrics` reports the two factors separately.
- `band_transmission(S, want, crosstalk, alpha)`: in-band transmission to the
  right port (mean or soft minimum over pass entries) minus a crosstalk
  penalty on the stop entries.
- `mean_coupling(S, weights)`: coupled power averaged over sources, outputs
  and frequencies.

## Example

A waveguide with a design patch beside it, in the 2.5-D model on the CPU: one
forward and one adjoint run give `S` and the gradient of `|S|^2` with respect
to every density pixel.

```python
import numpy as np
import lightsim as ls
from lightsim import eim, invdes
from lightsim.spec import C0, Box, Medium, um

lam = 1.55 * um
n_bg, n_slab = 1.444, eim.slab_neff(3.476, 220e-9, lam, 1.444)
sim = invdes.planar_simulation([0, -1.5 * um], [5 * um, 1.5 * um], lam, n_bg, n_slab,
                               steps_per_wavelength=12, run_time=300e-15)
sim.add_medium(Medium("core", n=n_slab))
sim.add_structure(Box([-1, -0.25 * um, -1], [1, 0.25 * um, 1]), "core")

def port(name, x, direction):
    return invdes.port_mode(sim, name, 0, x, [x, -1 * um, 0], [x, 1 * um, sim.max[2]],
                            C0 / lam, direction, "te")

design = invdes.Design.effective_index([2 * um, 0.3 * um], [3 * um, 0.9 * um], n_bg, n_slab,
                                       pixel=50e-9, filter_radius=100e-9)
problem = invdes.SMatrixProblem(sim, design.grids(), [port("in", 1 * um, "+")],
                                [port("out", 4 * um, "-")], [C0 / lam], devices="cpu")
inv = invdes.InverseDesign(problem, design, objective=invdes.mean_coupling)
rho = design.random(seed=0, mean=0.5, amplitude=0.3)
value, grad, info = inv.value_and_grad(rho, beta=4.0)
print(info["S"].shape, grad.shape, info["simulations"])
```

```text
(1, 1, 1) (21, 13) 2
```

## API

```text
from lightsim import invdes

# ports: mode planes with the direction pointing INTO the device
p = invdes.port_mode(sim, "in", axis, position, min, max, freq, "+", polarization="te")  # 2.5-D
p = invdes.port_mode(sim, "in", axis, position, min, max, freq, "+")                     # 3-D
invdes.Port(name, axis, position, min, max, profiles, direction)       # any profile dict(s)
invdes.Excitation(name, source, norm=None)                             # non-modal source

design = invdes.Design.slab(min_xy, max_xy, (z0, z1), eps_lo, eps_hi, pixel=..., filter_radius=...)
design = invdes.Design.partial_etch(min_xy, max_xy, (z0, zmid, z1), eps_lo, eps_hi, ...)
design = invdes.Design.effective_index(min_xy, max_xy, n_lo, n_hi, n_partial=None, ...)
design.eps(rho, beta, eta=None)            # list of grid permittivities
design.vjp(rho, beta, grads, eta=None)     # dL/d rho
design.projected(rho, beta), design.binarization(rho, beta), design.random(seed, mean, amplitude)

problem = invdes.SMatrixProblem(sim, design.grids(), inputs, outputs, freqs,
                                devices=["cuda:0", "cuda:1"], normalize="monitor",
                                engine="python")   # or "native": lightsim.autodiff session
problem = invdes.SMatrixProblem(..., monitors=[ls.DFTMonitor("map", ...)])   # optional field maps
sol = problem.solve(eps, adjoint=True)     # sol.S [M, N, K], sol.stats, sol.fields[n]["map"]["Ey"]
sol.vjp(g)                                 # dL/d eps per grid
sol.yee_vjp(g), sol.jacobian(m, n, k)      # per Yee voxel / one complex entry

inv = invdes.InverseDesign(problem, design, objective=lambda S: invdes.matrix_fidelity(S, T))
S, pullback = inv.forward(rho, beta)       # the differentiable layer
value, grad, info = inv.value_and_grad(rho, beta)
rho, history = invdes.optimize(inv, rho0, 200, beta=invdes.beta_schedule(2, 2, 40, 64),
                               lr=0.02, checkpoint="run.npz")

sim2d = invdes.planar_simulation(min_xy, max_xy, wavelength, n_background, n_max)  # empty 2.5-D model
polys = invdes.density_polygons(rho_hat, min_xy, max_xy, level=0.5)   # [(exterior, holes)] for GDS
```

Back-propagation wrappers (imported on demand; neither framework is a
dependency):

```text
import torch, lightsim.invdes.torch as lt
rho = torch.tensor(rho0, requires_grad=True)
S = lt.s_matrix(inv, rho, beta=8.0)        # torch.autograd.Function, complex [M, N, K]
loss = (S.abs() ** 2)[0, 0].mean(); loss.backward()

import jax, lightsim.invdes.jax as lj
g = jax.grad(lambda r: jnp.mean(jnp.abs(lj.s_matrix(inv, r, beta=8.0)) ** 2))(rho0)   # jax.custom_vjp
```

`lightsim.adjoint.mode_power_gradient` is unchanged in behaviour and now calls
the shared kernel `lightsim.adjoint.amplitude_sensitivity`.

## Validation

`python/tests/test_invdes.py` (CPU by default, `LIGHTSIM_DEVICE=cuda:0` for a
GPU). Finite differences are central differences along a random density
direction (2 extra S-matrix evaluations), compared with the adjoint value for
the real *and* imaginary part of each entry.

| check | result |
| --- | --- |
| parametrisation chain (filter, projection, levels, masks, symmetry; 4 variants) vs FD | relative 1e-5 |
| grid sampling vs the core rasteriser (`plane_info`), 3 E components | 2e-6 (f32 storage); transpose dot-product test 1e-10 |
| single port, 3-D: Yee-voxel gradient of abs(S)^2 vs `mode_power_gradient` | same arrays within tolerance; S within 0.2 % |
| single port, 3-D: complex dS vs FD | 0.13 % |
| 2 inputs x 2 outputs x 3 frequencies, 2.5-D, full chain: Re and Im of all 12 entries | 0.1 - 1.0 % per entry |
| matrix-fidelity objective through the chain, 2.5-D | 2.2 % |
| partial etch, 3-D, 2 frequencies: complex dS | 1.2 % |
| `engine="reverse"`, partial etch, 3-D, 2 frequencies: S against `engine="native"` | 3.8e-8 (CPU), 4.5e-8 (RTX PRO 6000) |
| same: derivative of the mean coupling along the gradient (central FD, step 0.01) | 4.1e-5 (CPU), 2.0e-4 (RTX PRO 6000; `native`: 4.0 %) |
| same: along a random density direction (Richardson FD, steps 0.01 and 0.02) | 2.5e-3 on both (`native`: 25 %; the derivative is 16 times smaller) |
| `engine="reverse"` with an oriented medium | refused, the error names `"native"` |
| torch `autograd.Function` and JAX `custom_vjp` vs native autodiff of an analytic layer | 1e-9; on the FDTD layer equal to the hand pullback |
| optimiser: checkpoint / resume reproduces an uninterrupted run; L-BFGS-B | exact |

The same file passes on `cuda:0` (RTX PRO 6000 and GB10) with the same
errors to three significant digits (2.5-D entries 0.1 - 1.0 %, 3-D 0.15 %,
partial etch 1.2 %); the whole file takes 16 s on an RTX PRO 6000 against
about 200 s on CPU.

## Scope and limits

- The design grids are appended last: no base structure may be meant to cover
  the design region (put fixed features into the density with `fixed_solid`).
- Index grids are staircased at their box faces and interpolate linearly in
  permittivity between samples; the final binarised design should be
  re-simulated as polygons (`density_polygons` -> GDS -> `lightsim.sparams`),
  where sub-pixel smoothing applies. The linear (arithmetic) average across a
  silicon edge is larger than the smoothed permittivity seen by a field
  normal to that edge, so a grid-optimised design behaves like slightly
  dilated polygons: trace the contour at a calibrated level below 0.5 (scan
  `density_polygons(level=...)` against a 3-D run) rather than at 0.5.
- Mode profiles are solved at the centre frequency and used over the band (as
  in `lightsim.sparams`); the cells around a mode source must be uniform over
  4 cells along the propagation axis for the two-plane calibration.
- Lossless, non-dispersive design media only (an `IndexGrid` restriction; a
  uniform conductivity can be set with `Design(medium_sigma=...)`).
- The JAX wrapper runs the simulations outside the trace: `jax.grad` /
  `jax.vjp` work, `jit` / `vmap` do not.
- The 2.5-D path inherits the limits of the effective-index method
  (`docs/modules/eim.md`): designs optimised there must be verified in 3-D.
