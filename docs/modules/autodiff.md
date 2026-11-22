# Gradients inside the engine

Inverse design needs `dL/d eps` for every voxel of a design region, thousands
of times. `lightsim.invdes` computes it by driving every simulation from
Python and multiplying the adjoint fields in NumPy. This module moves the
whole gradient into the Rust engine, on the CPU and on the GPU:

| | what it gives | cost per gradient |
|---|---|---|
| `GradientSession` | S-matrix `S[M, N, K]` and its vector-Jacobian product for any upstream gradient (continuous adjoint, same formula as `lightsim.invdes`) | N forward + M adjoint runs |
| `ReverseSession`, `value_and_grad` | gradient of a time-domain objective by reverse-mode differentiation through the time loop (discrete adjoint) | 1 forward run + 1 reverse pass (the pass costs 2.6 to 3.5 forward runs on the benchmark scenes) |

The design is a set of `lightsim.IndexGrid` structures
(`lightsim.invdes.DesignGrid`) painted after every other structure; gradients
are returned on the grid samples.

## Python API

### Persistent S-matrix session

The samples of this page run on the CPU in a few seconds; put `"cuda:0"`, or
a list such as `["cuda:0", "cuda:1"]`, in place of `"cpu"` to run them on
GPUs. A 2.5-D waveguide with a design region between two ports:

```python
import numpy as np
import lightsim as ls
from lightsim import autodiff, eim, invdes
from lightsim.spec import C0, Box, Medium, um

lam = 1.55 * um
f0 = C0 / lam
n_bg, n_si = 1.444, eim.slab_neff(3.476, 220e-9, lam, 1.444)
sim = invdes.planar_simulation([0, -1.5 * um], [5.4 * um, 1.5 * um], lam, n_bg, n_si,
                               steps_per_wavelength=12, run_time=300e-15, shutoff=1e-6)
sim.add_medium(Medium("core", n=n_si))
sim.add_structure(Box([-5 * um, -0.25 * um, -1], [12 * um, 0.25 * um, 1]), "core")


def port(name, x, direction):
    return invdes.port_mode(sim, name, 0, x, [x, -0.75 * um, 0], [x, 0.75 * um, 1 * um], f0, direction, "te")


p_in, p_out = port("in", 1 * um, "+"), port("out", 4.4 * um, "-")
des = invdes.Design.effective_index([2 * um, -1.0 * um], [3.4 * um, 1.0 * um], n_bg, n_si,
                                    pixel=50e-9, filter_radius=120e-9)
grids = des.grids()
freqs = C0 / (np.array([1.53, 1.57]) * um)
eps = des.eps(des.random(seed=1, mean=0.5, amplitude=0.4), 4.0)   # one array per grid

sess = autodiff.GradientSession(sim, grids, inputs=[p_in], outputs=[p_out], freqs=freqs,
                                devices="cpu")                    # set up once
for it in range(2):
    sess.set_design(eps)             # only the design box is re-evaluated and uploaded
    S = sess.forward()               # complex [M, N, K]; design-region fields stay on the device
    L, g = float(np.sum(np.abs(S) ** 2)), 2 * S      # g = dL/dRe(S) + i dL/dIm(S)
    grads = sess.backward(g)         # list of dL/d(eps), one array per grid (shape grid.dims)
    print(f"design {it}: sum |S|^2 = {L:.3f}, S {S.shape}, gradient {grads[0].shape}")
    eps = [e + 0.3 * dg / np.abs(dg).max() for e, dg in zip(eps, grads)]
```

```text
design 0: sum |S|^2 = 0.727, S (1, 1, 2), gradient (29, 41, 1)
design 1: sum |S|^2 = 0.820, S (1, 1, 2), gradient (29, 41, 1)
```

- **Set-up (once).** The scene is rasterised, the sources and monitors of the
  N forward and M adjoint runs are compiled and uploaded, and every device of
  the list gets one persistent engine (fields, CPML state, monitors,
  coefficients). Port modes are solved before, in Python
  (`invdes.port_mode`), and passed in with the ports.
- **`set_design(eps)`.** The index-grid samples are evaluated on the Yee
  voxels of the design box only, through the rasteriser's own functions, and
  the coefficient sub-blocks are uploaded (in fused mode the cells of the
  design box own rows of the material table, so only those rows change).
- **`forward(adjoint=False)`.** Runs the N excitations, one at a time per
  device, and returns `S`. With `adjoint=True` the M adjoint runs go into the
  same dispatch.
- **`backward(g)`.** Runs the adjoint simulations `g` needs (outputs whose
  rows of `g` are zero are skipped; runs already done for this design are
  reused), forms `Re[conj(g) (i w eps0 / 4) E_adj E_fwd] dV` on the first
  device and applies the transpose of the rasterisation. `backward_yee(g)`
  returns the per-voxel gradient before that last step.
- **Sampling.** The design-region fields and the port monitors of a run are
  accumulated together, every `sess.dft_stride` time steps: the stride of
  the simulation (`Simulation(dft_stride=n)`), by default the automatic one
  that the bandwidth of the source allows (see
  [Monitors](../guide/monitors.md)), typically 20 to 50 steps. Every monitor
  of a session keeps its own sums, also when two ports lie on one plane,
  because the adjoint product reads them on the device.

`lightsim.invdes` uses it through one keyword:

```python
prob = invdes.SMatrixProblem(sim, grids, [p_in], [p_out], freqs, devices="cpu", engine="native")
sol = prob.solve(eps)
sess.set_design(eps)
print(type(sol).__name__, np.abs(sol.S - sess.forward()).max())
```

```text
NativeSolution 0.0
```

`engine="python"` (the default; `LIGHTSIM_INVDES_ENGINE` overrides it) keeps
the NumPy path. Everything else is unchanged: `solve`, `SMatrixSolution.S`,
`vjp`, `yee_vjp`, `jacobian`, `InverseDesign`, `optimize`, the torch and JAX
wrappers. The session holds one design at a time; a solution used after the
problem was solved for another design re-runs its own design first, so old
solutions stay valid (the usual solve, back-propagate, next-design loop never
triggers this).

### Reverse mode through the time loop

```python
from lightsim.spec import GaussianPulse, PointDipole

dx = 50e-9
sim = ls.Simulation(min=[0, 0, 0], max=[3.2 * um, 2.0 * um, dx], mesh=ls.UniformMesh(dx),
                    run_time=70e-15, background="ox",
                    boundaries={"x": [ls.PML(), ls.PML()], "y": [ls.PML(), ls.PML()], "z": [ls.PERIODIC, ls.PERIODIC]})
sim.add_medium(Medium("ox", n=1.444))
sim.add_medium(Medium("core", n=2.6))
sim.add_structure(Box([-1, 0.8 * um, -1], [10 * um, 1.2 * um, 1]), "core")
sim.add_source(PointDipole([0.5 * um, 1.0 * um, 0], "Ez", GaussianPulse(f0, 0.25 * f0), amplitude=1e-15))
grid = invdes.DesignGrid([1.21 * um, 0.59 * um, -1], [2.03 * um, 1.43 * um, 1], (7, 9), Medium("design", eps=2.6**2))
x_out, mn, mx = 2.7 * um, [2.7 * um, 0.4 * um, 0], [2.7 * um, 1.6 * um, dx]

obj = autodiff.flux_energy(axis=0, position=x_out, min=mn, max=mx, t_start=20e-15, t_end=60e-15, taper=7e-15)
eps = np.random.default_rng(4).uniform(2.5, 6.5, grid.dims)
L, grad, info = autodiff.value_and_grad(sim, grid, eps, obj, device="cpu")
# next design on the same buffers:
step = 0.2 * grad / np.abs(grad).max()
L2, grad2, info = autodiff.value_and_grad(sim, grid, eps + step, obj, session=info["session"])
print(grad.shape, info["steps"], f"predicted change {np.sum(grad * step) / L:+.3f}, actual {(L2 - L) / L:+.3f}")


def loss(series):                                    # {comp: array [step, i, j, k]}
    e = series["Ez"].astype(np.float64)
    return float(np.sum(e * e)), {"Ez": 2 * e}       # L and dL/d(sample)


samples = autodiff.field_samples(mn, mx, components=("Ez",))
L3, grad3, _ = autodiff.value_and_grad(sim, grid, eps, samples, device="cpu", fn=loss)
print(grad3.shape, L3 > 0)
```

```text
(7, 9, 1) 735 predicted change +0.032, actual +0.031
(7, 9, 1) True
```

`sim` keeps its own sources (any lightsim source). Objectives:

- `flux_energy(axis, position, min, max, t_start, t_end, taper)`: the energy
  (J) crossing a plane towards `+axis` in a time window,
  `L = sum_n w(t_n+1/2) dt sum_cells dA (E x H)_axis`, with E averaged over the
  two time levels around the H sample and H over the two planes around the E
  plane (the discrete Poynting flux).
- `field_samples(min, max, components)` with a callable: any function of the
  recorded samples. `fn(series) -> (L, cotangents)` receives
  `{comp: array [step, i, j, k]}` and returns `dL/d(sample)` in the same
  layout (write it by hand, or let torch / JAX differentiate it).

`ReverseSession` exposes the steps (`set_design`, `forward`, `series`,
`backward`).

## Permeability designs and magnetic loss

Media have a magnetic conductivity `sigma_m` (Ohm/m; `Medium(sigma_m=...)`
in Python), the magnetic current `sigma_m H` in Faraday's law. The H update
becomes `H~' = da H~ - ch curl E` with `f = sigma_m dt / (2 mu0 mu)`,
`da = (1 - f) / (1 + f)`, `ch = c dt / mu / (1 + f)`. The engines keep the
list of lossy H cells and scale them by `da` before the regular H update
(CPU; on the GPU a separate kernel, `hloss_scale`, launched only for plans
with magnetic loss, which also run the split kernels, as all magnetic
plans do; the 2-D update is not used for them). Validation case
`magnetic_loss`: a half-space with matched loss (`sigma_m / (mu0 mu) =
sigma / (eps0 eps)`) reflects 1e-3 (the same `sigma` alone: 2e-2) and
decays at the analytic rate to 0.3 %, and a half-space with magnetic loss
only has the analytic reflectance (1–2 %), decay rate and impedance.

Time reversal divides by `da` (`H(n-1/2) = (H(n+1/2) + ch curl E) / da`).
The transposed step scales H *after* the E update (`b~ = b - ch curl a`,
`a' = ca a + cb curl b~`, `b' = da b~`), where the forward step scales it
before its curl update; with that order `<y, T^k x> = <(T^k)^T y, x>` holds
to 1e-6 with a magnetically lossy block in the scene of the transposition
test.

A design grid can carry the permeability instead of the permittivity:

```python
mug = invdes.DesignGrid(grid.min, grid.max, (7, 9), Medium("mudesign", n=1.444, mu=4.0),
                        name="mu", permeability=True)
mu = np.random.default_rng(5).uniform(1.0, 4.0, mug.dims)
Lm, gm, info = autodiff.value_and_grad(sim, mug, mu, obj, device="cpu")
print(gm.shape, f"reconstruction residual {info['reconstruction_residual']:.0e}")
```

```text
(7, 9, 1) reconstruction residual 5e-07
```

The grid owns the H voxels whose positions it contains (the rasteriser's
`Shape::IndexGrid` with `mu` samples), its E voxels take the medium's
`eps`, and the medium's `mu` must be the largest value the grid takes
(above 1; no magnetic conductivity in the design medium). Reverse mode
accumulates `q1 = sum_n b^ (H'(n+1/2) - H(n-1/2))` over the H boxes of the
grids, with `b^` the scaled adjoint of `H'(n+1/2)`, and returns
`dL/d mu = -V_H / ch^2 q1 d ch / d mu` pulled back to the samples (on the
GPU with the existing `ad_accumulate` kernel). Permittivity and permeability
grids can be used together if they do not overlap. The S-matrix session
refuses permeability grids (its continuous adjoint formula is written for
the permittivity).

Measured (reverse mode against central differences along a random
direction; CPU / RTX PRO 6000; GPU against CPU 1e-7):

| case | CPU | RTX PRO 6000 |
|---|---|---|
| permeability grid behind a magnetic slab (mu = 1.5), flux / phasor objective | 4.1e-4 / 3.8e-5 | 5.0e-5 / 4.9e-5 |
| same, slab with magnetic conductivity 3e8 Ohm/m | 1.5e-4 / 3.7e-5 | 4.3e-4 / 5.5e-5 |
| permittivity grid through a magnetically lossy block | 6.8e-6 | 9.6e-5 |
| permeability and permittivity grids side by side | 1.2e-5 | |

## Gradients with respect to dispersive poles

`ReverseSession.set_pole_gradients(True)` adds to every reverse pass the
gradient with respect to the poles of the scene's dispersive media, per
medium, pole and axis (the sum over the cells of that E component):

```python
from lightsim.spec import Lorentz

psim = ls.Simulation(min=[0, 0, 0], max=[1.6 * um, 1.2 * um, 1.0 * um], mesh=ls.UniformMesh(dx),
                     run_time=28e-15, background="ox")
psim.add_medium(Medium("ox", n=1.45))
psim.add_structure(Box([-1, 0.4 * um, 0.3 * um], [10 * um, 0.8 * um, 0.7 * um]), Medium("core", n=2.45))
psim.add_structure(Box([1.05 * um, 0, 0], [1.15 * um, 1.2 * um, 1.0 * um]),
                   Medium("pole", eps=(2.0, 2.5, 3.0), poles=[Lorentz(1.5, 2 * np.pi * 1.4 * f0, 2e14)],
                          pole_axes=(True, False, True)))
psim.add_source(PointDipole([0.3 * um, 0.6 * um, 0.525 * um], "Ez", GaussianPulse(f0, 0.3 * f0), amplitude=1e-15))
pgrid = invdes.DesignGrid([0.52 * um, 0.33 * um, 0.3 * um], [0.93 * um, 0.91 * um, 0.7 * um], (5, 6, 2),
                          Medium("design", eps=6.0))
pobj = autodiff.flux_energy(0, 1.35 * um, [1.35 * um, 0.2 * um, 0.1 * um], [1.35 * um, 1.0 * um, 0.9 * um])
psess = autodiff.ReverseSession(psim, pgrid, pobj, device="cpu")
psess.set_pole_gradients(True)
Lp, gp, _ = autodiff.value_and_grad(psim, pgrid, np.full(pgrid.dims, 4.0), pobj, session=psess)
for e in psess.pole_gradients():
    print(e["medium"], "pole", e["pole"], "axis", "xyz"[e["axis"]], "dL/d(delta_eps, omega0, gamma) / L =",
          ", ".join(f"{v / Lp:.3e}" for v in e["params"]))
```

```text
pole pole 0 axis x dL/d(delta_eps, omega0, gamma) / L = 1.494e-02, 6.844e-17, -7.593e-17
pole pole 0 axis z dL/d(delta_eps, omega0, gamma) / L = -2.122e-01, 1.379e-15, -2.414e-16
```

`pole_gradients()` returns one entry per (medium, pole, axis) with
`dL/d(c0, c1, c2)` of the ADE step `P(n+1) = c0 P(n) + c1 P(n-1) + c2 E(n)`
and `dL/d` of the pole's parameters: Lorentz (delta_eps, omega0, gamma),
Drude (omega_p, gamma), Debye (delta_eps, tau). A per-axis medium
(`pole_axes`) gives entries for its axes only, so a different pole per axis
is a separate medium per axis.

How: the adjoint E field of the reverse pass is the cotangent of E(n+1)
(the forward step on the scaled adjoints is the transposed step on the
fields; for the pole states it is a different but equivalent realisation,
so their cotangents are not available from it). An extra recursion per
dispersive cell carries the true cotangents of P(n+1) and P(n) backwards,
`w = g_P - s a`, `dL/dc += w (P(n), P(n-1), E(n))`,
`g_P <- c0 w + s a + g_PP`, `g_PP <- c1 w` (`a` the cotangent of E(n+1),
`s` the cell's polarisation scale), with the reconstructed pole states; on
the GPU one kernel (`ad_pole_grad`) per dispersive component and step, only
when pole gradients are asked for. The parameter derivatives follow from
the ADE coefficients by central differences in double precision.

Limits: dispersive structures must stay out of the absorbing layers (the
reverse pass does not reconstruct the fields there; such a scene is
refused), and 2-D / 2.5-D simulations are refused (the 2-D update keeps its
own pole state).

Measured (reverse mode against central differences with Richardson
extrapolation; per-axis media: a Lorentz pole on x, another Lorentz pole and
a Debye pole on z; GPU against CPU 1e-6):

| parameter | CPU |
|---|---|
| x-only Lorentz: delta_eps / omega0 / gamma | 2.2e-5 / 1.3e-4 / 8.7e-5 |
| z-only Lorentz: delta_eps / omega0 / gamma | 1.9e-5 / 4.0e-4 / 1.1e-4 |
| z-only Debye: delta_eps / tau | 1.6e-4 / 4.6e-4 |
| RTX PRO 6000: x-only delta_eps (GPU runs) | 9.0e-5 |

The damping needs a larger finite-difference step (10 %): at a 1 % step its
effect on the single-precision coefficients `c0 ~ 2`, `c1 ~ -1` is a few
ulp, and the finite differences are off by up to 6e-3.

## Off-diagonal and oriented-pole media in the S-matrix session

The S-matrix session accepts media with off-diagonal permittivity
(`eps_off`, `sigma_off`, rotated crystals) and oriented dispersive poles
(`oriented`) in the background, outside the design. These media are
reciprocal (symmetric tensors), so the adjoint run is an ordinary run of
the same scene and the adjoint formula is unchanged; the engines run them
forward with the tensor update of the plain solver (the GPU engine resets the
tensor and oriented-pole state between runs). The Python engine of
`lightsim.invdes` drives ordinary runs and accepts them too. Reverse mode
keeps refusing them: its time reversal of the tensor update would need a
coupled solve over neighbouring cells at every step.

Measured (FDTDX `TestOrientedDispersionGradients`: a slab with an oriented
Lorentz pole, u along (1, 1, 0), next to the design; the gradient along its
own direction against central differences):

| scene | CPU | RTX PRO 6000 |
|---|---|---|
| plane-wave ports, oriented slab behind the design (`session_through_an_oriented_dispersive_medium`) | 3.4e-3 | 3.5e-3 (GPU vs CPU: S 4e-8, gradient 2e-6) |
| same slab, isotropic (no pole) | 7.1e-3 | |
| 3-D partial etch, oriented slab, both engines (`test_session_through_an_oriented_dispersive_medium`) | 4.6 % (Python engine; native 5.1 %; 3.5 % without the slab) | |

With the pole closer to the band (1.4 f0) the slab reflects more, the
derivative is five times smaller and the same absolute error is 3.5 % of
it.

## One gradient simulation over several GPUs

With `lightsim.invdes` and `engine="python"`, every forward and adjoint run
of the S-matrix problem is an ordinary simulation, so a device string such
as `"cuda:0,1"` splits each of them over two GPUs (domain decomposition
along x, as for any run): `SMatrixProblem(..., devices="cuda:0,1",
engine="python")`. On two RTX PRO 6000 the 3-D partial-etch problem gives
the one-GPU S and gradient bit for bit
(`test_python_engine_runs_split_over_two_gpus`). The native session and
reverse mode run each simulation on one device (a device list spreads their
runs over several); decomposing their runs would need halo exchanges in the
reverse and adjoint steps and design boxes spanning slabs.

## Accuracy at interfaces

The S-matrix gradient (session and the Python engine of `lightsim.invdes`)
uses the continuous adjoint formula `(i w eps0 / 4) E_adj E_fwd dV` per Yee
voxel; reverse mode differentiates the discrete scheme exactly. Comparing
the two on the plane-wave-port scene of `reverse_mode_and_session_agree_on_one_objective`
(reverse mode is within 9e-5 of central differences there):

| design | session against reverse mode |
|---|---|
| random samples (2.5 ... 6) | 1.4e-3 along a random direction; 7.7e-3 of the largest entry |
| binary: a block of eps 6 in eps 2.1, 12 x 8 x 4 samples | per sample 0.96 ... 1.05 of the exact value; mean deviation 5.2e-3 of the largest entry on the 144 samples next to the block's faces, 2.0e-3 on the others |
| same, summed over the 144 edge samples (moving the faces outwards) | 1.216e-3 against 8.257e-4: 47 % |

The deviation is not concentrated at the faces: it varies smoothly across
the design (a standing-wave-like pattern of a few per cent of the local
value, with the sign changing along the propagation axis). Derivatives of
shape parameters are sums over the samples along a boundary in which these
errors do not cancel while the values largely do, hence 2–4 % of the
largest derivative for polygon vertices in 3-D (`python/tests/test_shapes.py`,
Python engine, against central differences: 2.5-D polygon 0.1–0.7 %, 3-D
slab 0.4 / 0.6 / 2.3 %, 3-D partial etch 1.8 / 1.3 / 3.5 %; tolerance 4 %).

What was tried and not kept:

- The D-normal / E-parallel form (`Delta eps E_par . E_par - Delta(1/eps)
  D_perp . D_perp`) is the continuum limit for a moving boundary in a
  sub-pixel-smoothed model. lightsim's design grids are not smoothed: every
  Yee component takes the interpolated sample at its own position, and the
  derivative of that discrete model with respect to a voxel permittivity is
  the per-component product the formula already uses. The form would not be
  the derivative of what is simulated.
- The discrete-time factor of the leapfrog (`e^{+-i w dt / 2} sinc(w dt / 2)`
  on the product, `w dt = 0.12` here): the per-sample ratios spread from
  0.96 ... 1.05 to 0.65 ... 1.58 (or worse with the other sign).
- Correcting the port overlap for the averaging of H over the two planes
  around the port (`1 / cos(beta d)`): a pure port mode then shows no
  spurious reflection (`|S11|` without a design from 0.4–0.65 % to 1e-4), but
  the session moves away from reverse mode (1.4e-3 to 7.1e-3 along a random
  direction, 7.7e-3 to 1.3e-2 of the largest entry).

Where the 1 % level matters for boundary parameters, reverse mode gives the
exact derivative of the discrete scheme (the port objective of
`reverse_mode_and_session_agree_on_one_objective` evaluated on the samples
of the output plane).

## Gradient conventions

- Upstream gradient of a real loss with respect to a complex S entry:
  `g = dL/dRe(S) + i dL/dIm(S)`, so that `dL = Re(conj(g) dS)`. This is the
  convention of `lightsim.invdes` and of `torch.Tensor.grad`; JAX cotangents
  are the conjugate.
- Returned gradients are real arrays `dL/d(eps)` with the shape of the
  index-grid samples (`grid.dims`), one per grid.
- S, its normalisation (`"monitor"`: two mode monitors behind every mode
  source; `"source"`: source spectrum), the placement of the monitors and
  the adjoint formula `dS = (i w eps0 / 4) E_adj E_fwd dV` are those of
  `SMatrixProblem`; the two engines agree to rounding (see the numbers
  below).
- Found while comparing the two engines: with `normalize="source"` (or an
  `Excitation`) and frequencies not given in ascending order, the Python
  path paired the source spectrum, which the core returned in ascending
  order, with the wrong frequencies (S off by its phase). Both engines now
  evaluate the spectrum at the frequencies as given (the Python path with
  `Results.source_spectrum(freqs=...)`), and the two agree to 2e-16 for
  either order.

## How it works

### Shared with the forward path

`Shape::grid_eps` is now evaluated through `Shape::grid_stencil` (same
arithmetic, same order), `plan::e_update_coefficients` is the function the
plan compiler uses for `(ca, cb)`, and `raster::center_inside` decides which
voxels a grid owns. `DesignMap` calls exactly these, so a design update is
bit-identical to a re-compilation (`design_update_matches_recompilation`) and
the pullback scatters with the weights the forward sampling uses
(`pullback_is_the_transpose_of_the_sampling`).

### Engine

Both drivers are written against `lightsim_core::autodiff::Engine`: a
persistent backend for one device with attachable run configurations (sources
and monitors), partial coefficient updates, DFT fields that stay on the
device, the adjoint product and the primitives of the reverse pass.
`CpuEngine` (lightsim-core) and `CudaEngine` (lightsim-cuda) implement it.
The GPU engine steps with the plain solver's kernels (fused when the plan
allows; the 2-D step for 2-D and 2.5-D simulations, see
[2-D, 2.5-D and 3-D](../guide/dimensions.md)) and adds a few of its own, in
`kernels/src/autodiff.rs`:

| kernel | purpose |
|---|---|
| `ad_dft_multi` | running DFT of every monitor box of a run in one launch per step |
| `ad_gather_multi`, `ad_scatter_multi` | record / replay all the boxes of a time monitor in one launch (objective samples, PML shell) |
| `ad_scatter_box` | write a box of a field: design coefficient updates |
| `ad_product` | adjoint product of two DFT fields |
| `ad_inject_scaled`, `ad_inject_complex_scaled` | remove a step's sources during time reversal |
| `ad_accumulate` | per-step sums for `dL/d cb`, `dL/d ca` (compensated) |
| `ad_disp_pre`, `ad_disp_post` | time reversal of dispersive cells |
| `ad_nl_post`, `ad_nl_adj_pre`, `ad_nl_adj_post` | chi2 / chi3 cells: time reversal of the implicit step, the two adjoint factors |
| `ad_pml_phi`, `ad_pml_h`, `ad_pml_e` | transposed CPML of the adjoint step (below) |

They live in their own `#[cuda_module]` and are compiled into the same PTX;
the stepping kernels of `kernels/src/main.rs` are untouched (their PTX text is
identical with and without this module). A plain solver run launches one
`dft_accumulate` per monitor box at every sampled step, the session one for
all boxes (see `docs/benchmarks.md`).

### The discrete adjoint

With `H' = H - ch Cp E` and `E' = ca E + cb Cd H'` (`Cp`, `Cd` the curls with
primal / dual spacings), the Yee curls satisfy `Cd^T = V_H Cp V_E^-1`, where
`V_E`, `V_H` are the diagonal "update volumes" (products of the spacings the
update divides by, the dual cell halved at a PMC low wall). The transposed
step is therefore the forward step itself on the scaled adjoints
`a^ = V_E^-1 cb a`, `b^ = -V_H^-1 ch b`, run backwards in time with the
cotangents of the objective as sources. So the interior of the adjoint runs
with the same kernels as the forward simulation. The same holds for
conductive media and for dispersive poles (the pole recursion is diagonal in
space; `cell::tests::dispersive_cell_adjoint`). For chi2 / chi3 cells the
adjoint is the linear step with two per-cell factors from the forward state,
`dE(n+1)/dE_lin` and `dE(n+1)/dE(n)`.

**CPML is the exception.** The forward CPML of a slab convolves the
*difference* of the other field with the stretching coefficients of the cell
being updated. Its transpose convolves the *field* with the coefficients of
the cell it comes from and differences afterwards; with `(b, c)` the two
components tangential to the slab, in the scaled variables:

```text
phi_b   <- bE phi_b + e_b        w_b = (1/kappaE - 1) e_b + aE phi_b       (E cells of the slab)
h_c     -= ch d(w_b)             h_b += ch d(w_c)
theta_b <- bH theta_b + h_b      v_b = (1/kappaH - 1) h_b + aH theta_b     (H cells of the slab)
e_c     += cb d(v_b)             e_b -= cb d(v_c)
```

The two coincide only where the profile is flat. `autodiff::cpu::adjoint_step`
and the `ad_pml_*` kernels implement the transpose; with absorbing layers the
GPU adjoint therefore steps with the split update kernels plus these, and
with the fused kernel only when the scene has none. Using the forward CPML
for the adjoint (the first version of this module did) leaves an error that
depends on how much of the adjoint field reaches the layers: 1.4 % on the
6 M-cell benchmark gradient and 3 % with design voxels next to a layer,
against 5e-7 ... 6e-5 with the transpose (Verification).

The reverse pass, per step n (from the last to the first):

1. replay the recorded shell (H), remove the E sources of step n, undo the E
   update with `-cb/ca`, `1/ca`: `E(n+1) -> E(n)`;
2. add the cotangent `dL/dE(n+1)` to the adjoint; accumulate
   `sum a^(n+1) (E'(n+1) - ca E(n))` and `sum a^(n+1) E(n)` over the design box;
3. replay the shell (E), remove the H sources, undo the H update;
4. add `dL/dH(n+1/2)`; advance the adjoint by one transposed step.

At the end `dL/d eps = dL/d cb d cb/d eps + dL/d ca d ca/d eps` on the owned
voxels, then the pullback to the grid samples.

**Memory.** Two field sets (forward and adjoint, each with its CPML
memory), plus per step the one-cell shell of PML cells
next to the interior (two tangential components per face: what the reversed
interior update reads from the layers) and the objective monitor. Nothing of
the volume is stored. For the 6 M-cell benchmark scene the shell is 2.0 MB per
step (5.2 GB for 2 623 steps), for the 32 M-cell one 7.0 MB per step
(27.7 GB for 3 934 steps), kept on the device.

### What "exact" covers

- PEC, PMC and periodic boundaries, graded meshes, conductive and magnetic
  media: the forward step on the scaled adjoints *is* the transpose
  (`transposed_step_is_the_forward_step_on_scaled_adjoints`: `<y, T^k x>` and
  `<(T^k)^T y, x>` agree to 1e-6 or better).
- CPML layers, with the transposed recursion above: for random fields
  filling the layers, `<y, T^k x>` and `<(T^k)^T y, x>` agree to
  1e-8 ... 2e-6 (with the forward CPML in its place: 1e-2 ... 0.3).
- Forward fields come back by time reversal in single precision: to 1e-6 of
  the peak over a whole run in the tests. Absorption is amplified on the way
  back, by `exp(loss)` over the run: conductors, Drude / Lorentz damping and
  above all Debye relaxation limit how long a run can be reversed. Measured
  with a Debye block: `t / tau` = 3.5 gives a gradient good to 3e-5,
  `t / tau` = 14 (round-off amplified a million times) 3e-3.

## Rust's own autodiff (`std::autodiff`, Enzyme)

The question was whether the adjoint kernels could be generated by
`#[autodiff_reverse]` instead of written by hand. Findings (2026-11-21):

**Pinned toolchain (Rust 1.99.1): not usable.** A small test crate with the
E-cell update `ca * e + cb * curl - scale * dp`:

```text
$ cargo build --release
error[E0554]: `#![feature]` may not be used on the stable release channel

$ RUSTC_BOOTSTRAP=1 cargo build --release
error: using the autodiff feature requires -Z autodiff=Enable

$ RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zautodiff=Enable" cargo build --release
error: autodiff backend not found in the sysroot: failed to find a `libEnzyme-23` folder in the sysroot candidates:
       * ~/.rustup/toolchains/1.99.1-aarch64-apple-darwin/lib
  = note: it will be distributed via rustup in the future
```

The channel manifests list an `enzyme-preview` component for 1.99.1 (and for
beta 1.101) with `available = false` for every target; only nightly ships it
(aarch64 and x86_64, macOS and Linux).

**Nightly + Enzyme on the CPU: works for per-cell code.**

```text
$ rustup toolchain install nightly --profile minimal --component enzyme-preview
$ RUSTFLAGS="-Zautodiff=Enable" cargo +nightly build --release      # with lto = "fat"
$ ./target/release/adtest
(0.646, 0.98, 0.21, 0.3, 1.7, -0.01, -0.5)      # value, d/de, d/dcurl, d/dca, d/dcb, d/dscale, d/ddp
```

This is the optional cargo feature `enzyme` of lightsim-core
(`scripts/test_enzyme.sh`; nothing else in the project uses nightly).
`autodiff::enzyme` generates the adjoints of the two material updates that
are tedious by hand and compares them with the hand-written ones of
`autodiff::cell`:

| update | hand-written | Enzyme | difference |
|---|---|---|---|
| chi2 / chi3 step (`nl_solve`, 4 Newton iterations): `dF/dE(n)`, `dF/dE_lin`, `dF/dchi2`, `dF/dchi3` | implicit function theorem on `D(E') = D(E) + E_lin - E` | reverse mode through the iterations | 1.2e-7 (relative) |
| two-pole dispersive cell: vector-Jacobian product of the 6 -> 5 step | transposed recursion | reverse mode | 3.0e-8 |

**Limit found on the CPU.** The feature builds and passes as lightsim-core's
own unit-test binary only. As soon as the reverse engine is alive in the same
LTO unit (lightsim-core built as a dependency for the integration tests or
the Python extension, or a unit test that instantiates the engine), rustc
aborts inside Enzyme, even when the engine does not call the generated
functions:

```text
inserting into : {[0]:Pointer, [0,-1]:Pointer} with [0,-1] of Float@double
illegal insertion
UNREACHABLE executed at .../enzyme/Enzyme/TypeAnalysis/TypeTree.h:364!
error: could not compile `lightsim-core` (lib)       (signal: 6, SIGABRT)
```

(nightly 1.102.0, 2026-11-20; a two-crate reduction of the same function
builds and runs.) The engine therefore uses the hand-written adjoints, and
Enzyme serves as their check.

**GPU kernels (cuda-oxide): not possible.** cuda-oxide is a custom rustc
codegen backend (Rust MIR -> Pliron IR -> LLVM IR -> PTX); Enzyme runs inside
rustc's LLVM backend, which is never invoked. Its repository and book do not
mention autodiff or Enzyme. A minimal kernel calling an
`#[autodiff_reverse]` function:

```text
$ cargo oxide build --arch sm_80
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed: Translation failed:
       adkernel::d_pole: ["src/main.rs": line: 8, column: 1] Compilation error: invalid input program.
       Unsupported construct: rustc intrinsic `std::intrinsics::autodiff` is not yet supported on the device
```

Declaring Enzyme's entry point as an extern, to run Enzyme's LLVM pass on
the exported IR, fails earlier:

```text
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed: Verification failed for lowered LLVM module:
       Symbol __enzyme_autodiff not found
```

So the GPU adjoint kernels are written by hand. That turned out to be little
code, because the interior of the transposed step is the forward step
(above): the new device code is the bookkeeping kernels and the transposed
CPML of the table, and the scalar formulas of the material updates
(`autodiff::cell`) are the ones Enzyme checks on the CPU.

## Verification

Run on the CPU (`cargo test -p lightsim-core`, `pytest python/tests/test_autodiff.py`)
and on an RTX PRO 6000 (`cargo test -p lightsim-cuda --test autodiff_gpu`,
`LIGHTSIM_DEVICE=cuda:0 pytest ...`).

**Transposed step** (`<y, T^k x>` against `<(T^k)^T y, x>` on a graded mesh
with a conductive and a magnetic block). PEC, periodic, PMC and two mixed
sets of walls, k = 1 and 30: 6e-9 ... 9e-7. CPML (on x and y; on all sides;
on z with PMC / PEC in x), random fields inside the layers, k = 1, 30 and
120: 1e-8 ... 1.9e-6 with the transposed CPML, 7e-3 ... 0.28 with the forward
CPML in its place.

**Time reversal**: fields at mid-run and the initial zero state reconstructed
to 3e-7 and 1e-6 of the peak (CPML on all sides, shell replayed).

**Reverse mode against central finite differences** (energy through a plane
in a time window; 3-D scene with a dipole, a waveguide and a 5 x 6 x 2 design
grid, directional derivative along a random direction; relative difference):

| case | CPU | RTX PRO 6000 | max abs(grad GPU - grad CPU) / max |
|---|---|---|---|
| PEC / periodic / PMC walls | 1.9e-6 | 5.0e-6 | 2.8e-7 |
| CPML on all sides | 1.6e-5 | 1.2e-4 | 5.6e-8 |
| periodic x, PMC y, CPML z | | 2.8e-5 | 1.3e-7 |
| CPML, conductive design medium (`ca`) | 7.8e-5 | 9.8e-8 | 1.8e-7 |
| CPML, Lorentz + Drude block | 1.7e-4 | 2.2e-4 | 2.1e-7 |
| walls, Debye block (`t / tau` = 3.5) | 2.9e-5 | 5.9e-5 | 3.4e-7 |
| CPML, design voxels up to the layers | 5.1e-7 | 3.1e-5 | 1.4e-7 |
| walls, chi2 + chi3 block | 4.1e-5 | 6.6e-6 | 2.1e-7 |
| CPML, chi2 + chi3 block | | 4.2e-5 | 1.3e-7 |
| CPML, split kernels for the forward run too | | 6.1e-5 | 9.6e-8 |

The finite differences themselves are good to about 1e-4 in single
precision, so these numbers are at the noise floor of the check; the last
column compares the two engines directly. In the chi2 + chi3 case the
nonlinearity changes the objective by 5 % and its gradient by 20 %.

At benchmark scale (`benchmarks/lightsim_autodiff.py`, RTX PRO 6000,
derivative along `sign(grad)`, step chosen for a given change of the
objective):

| scene | change of the objective, dL / L | relative difference |
|---|---|---|
| 6.0 M cells x 2 623 steps, random design | 4 % / 1 % / 0.25 % | 6.0e-5 / 5.3e-6 / 4.6e-7 |
| same, smooth design | 4 % / 1 % / 0.25 % | 3.3e-5 / 8.0e-6 / 8.5e-6 |
| same, uniform design | 4 % / 1 % / 0.25 % | 1.1e-4 / 6.0e-5 / 6.1e-5 |
| 32 M cells x 3 934 steps, random design | 1.7 % | 3.1e-4 |
| 2.5-D, 23 k cells, mode source, 1 658 steps | 0.7 % | 3.0e-5 |

The same 6 M check with the forward CPML as adjoint gave 1.4e-2 whatever the
step, which is how the difference was found.

Python (`test_autodiff.py`), reverse mode against central differences along
a random direction, CPU / RTX PRO 6000: 2.5-D scene 1.4e-4 / 1.5e-4, 3-D scene
4.8e-6 / 6.4e-5, caller-defined objective on recorded samples
2.3e-5 / 2.8e-6.

**S-matrix session.**

| check | result |
|---|---|
| native against Python path, same design (2 x 2 x 3 problem): S | 5e-16 (CPU), 3e-16 (GPU) |
| same: vector-Jacobian product for a random complex upstream gradient | 6.1e-8 (CPU), 6.6e-8 (GPU) |
| GPU session against CPU session (Rust test): S / gradient | 1.3e-7 / 2.2e-8 |
| device lists: GPU + CPU, two engines on one GPU | same S and gradient as one engine (1.3e-7 / 3.0e-8) |
| device list of two GPUs | same (checked once, on the build before the merge of the kernel work; not repeated on the final one) |
| split kernels against fused | 1.8e-7 / 2.3e-8 |
| design update against a fresh session; back to an earlier design | bit-identical S |
| all 2 x 2 x 3 entries, Re and Im, against central differences | worst entry 1.0 % of the largest (Python path: 0.97 %), CPU and GPU |
| `test_invdes.py` with `LIGHTSIM_INVDES_ENGINE=native` (2.5-D, 3-D slab, 3-D partial etch, torch and JAX wrappers) | passes on the CPU and the GPU; finite differences 1 %, 0.15 %, 1.2 % as with the Python engine |

The 1 % level of the finite-difference rows is the discretisation error of
the continuous adjoint formula, identical in the two engines.

**Whole suite** on the current tree (gradient engine, fixes, the 2-D mode and the tests that mirror FDTDX's gradient suite).

| suite | CPU | RTX PRO 6000 |
|---|---|---|
| `cargo test` (lightsim-core and lightsim-cuda) | all pass | all pass |
| `lightsim validate` | 51 / 51 | 51 / 51 fused, 51 / 51 with `LIGHTSIM_FUSED=0`; 51 / 51 on two GPUs, fused and split |
| `pytest python/tests` (engine `python`) | 200 passed | 200 passed |
| `test_invdes.py` + `test_autodiff.py` with `LIGHTSIM_INVDES_ENGINE=native` | 30 passed | 30 passed |
| `scripts/test_enzyme.sh` (nightly, feature `enzyme`) | 7 passed before the fixes branch was merged; not run again (the generated adjoints are those of `cell.rs`, which the merge did not touch) | |


## Limits

- One GPU per engine (`"cuda:N"`); several devices run independent
  simulations, not one decomposed simulation. No Bloch boundaries.
  Off-diagonal permittivity and oriented poles: S-matrix session only
  (background media), not reverse mode.
- Design media: plain or conductive dielectrics (the index-grid rule); no
  chi2 / chi3 in the design itself. Sources must not sit on design voxels
  (their injection coefficients depend on the local permittivity, which the
  gradient does not follow); the session checks this.
- Reverse mode covers linear, conductive, dispersive (Lorentz, Drude, Debye)
  and chi2 / chi3 media on both engines. On the GPU a scene that would use
  the dense dispersive form of the fused step runs on the split kernels
  instead (the reverse pass needs the per-cell pole lists). A cell that is
  both nonlinear and a first-order (Debye) pole is reversed approximately.
- Reverse mode reverses absorption (see above) and records the PML shell and
  the objective monitor every step, on the device.
- Reverse mode checks its own time reversal. After the reverse pass the
  fields left outside the absorbing layers must be below 0.1 of the peak field
  energy of the forward run, and the sums and the gradient must be finite;
  otherwise it returns an error instead of a gradient. Strong absorbers end
  there (a 3e4 S/m conductor, long runs through lossy media): use the
  S-matrix session for those. Measured residuals: 3e-7 to 9e-7 for exact
  reversals, 2e-3 for a 1.5e4 S/m slab whose gradient is still right to
  1e-6, and 1e3 or more when the reversal has diverged.
- Design grids must stay out of the absorbing layers. A grid whose voxels
  overlap a PML is refused when the session is built (reverse mode and the
  S-matrix session); a grid that reaches up to the layers is accepted.
- Both S-matrix engines (the session and the Python engine of
  `lightsim.invdes`) scale every port profile to unit power as the mode
  overlap measures it (`lightsim._core.port_power`), so S and its gradient do
  not depend on the power a profile carries and the two engines return the
  same gradient (3-D: to 4e-7; before, the Python engine used the profiles as
  given, and a 3-D mode-solver profile, unit power in the solver's own
  measure, carries 0.9953 W in the overlap's, which made the gradients differ
  by 0.47 %). The overlap measure is the consistent one: S is the overlap
  amplitude `(I1 + I2) / (4 Pm)` with `Pm` computed on the overlap's surface
  elements, and the adjoint formula takes the modes to have unit power in
  that same inner product. (On the 3-D partial-etch test scene the change
  moves the gradient from 1.24 / 1.07 % to 1.49 / 1.23 % of the finite
  differences, within the discretisation error of the continuous formula;
  in `python/tests/test_shapes.py` the 3-D vertex derivatives go from slab
  0.4 / 0.6 / 2.3 %, partial etch 1.8 / 1.3 / 3.5 % to 0.2 / 1.1 / 2.3 % and
  1.5 / 1.2 / 4.0 %, and that test's 3-D tolerance is now 4.5 %.)
  The adjoint formula assumes that a profile is an eigenmode of its port.
- A design is a permittivity or (reverse mode only) a permeability;
  reverse mode also returns the gradient with respect to the dispersive
  poles of the media (3-D, outside the PML). Reverse mode refuses oriented
  poles, and both engines are single precision.
- In a 2-D or 2.5-D simulation the S-matrix session and the forward steps
  of reverse mode run the 2-D update (three field components per
  polarization). On the GPU the reverse pass itself, i.e. the time reversal
  and, with CPML, the adjoint step, uses the general split kernels on the
  one-cell grid, all six components; on the CPU it uses the 2-D update.
- With CPML the adjoint step of the reverse pass runs on the split update
  kernels and the transposed-CPML kernels, not on the fused kernel: the
  reverse pass costs 2.6 (6 M cells) to 3.5 (32 M cells) forward runs. A
  fused adjoint kernel with the transposed CPML inside would bring it back
  towards 2; it is not written.
- The S-matrix session uses the continuous adjoint formula, like
  `lightsim.invdes`: its gradient differs from the finite differences of the
  discrete S by the discretisation error of that formula (about 1 % in the
  tests). Reverse mode differentiates the discrete scheme itself.
- Set-up compiles the scene once per run configuration (N + M
  rasterisations), then never again.
