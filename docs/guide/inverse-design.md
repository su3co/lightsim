# Inverse design

After this page you can compute the gradient of a device's response with
respect to the permittivity of every cell in a design region, describe a
design by a density with a minimum feature size, and run a gradient-based
optimisation of a multi-port, multi-frequency S-matrix, directly or through
PyTorch or JAX.

> **Note:** This page has two parts. `lightsim.adjoint` is part of every
> version. `lightsim.invdes` is described as of commit `8a8bb73` of the
> `invdes` development branch (package version 0.1.0); it is not yet in the
> main branch, and its interface may still change. The samples that import
> `invdes` need a build that contains it.

## The adjoint method

Inverse design treats the permittivity $\varepsilon(\mathbf{r})$ in a design
region as the unknown and improves it by following the gradient of a figure
of merit. Computing that gradient by changing one cell at a time would take
one simulation per cell. The adjoint method obtains it for all cells from two
simulations.

Consider the amplitude $a$ of a waveguide mode leaving the device. A small
permittivity change $\delta\varepsilon$ in a volume $dV$ at $\mathbf{r}$
changes it by

$$
\delta a = \frac{i\,\omega\,\varepsilon_0}{4}\;
\mathbf{E}_{\text{adj}}(\mathbf{r}) \cdot \mathbf{E}_{\text{fwd}}(\mathbf{r})\;
\delta\varepsilon\, dV ,
$$

where $\mathbf{E}_{\text{fwd}}$ is the field of the ordinary (forward)
simulation and $\mathbf{E}_{\text{adj}}$ is the field of an **adjoint**
simulation, in which the output mode is launched backwards into the device.
This follows from Lorentz reciprocity. One forward and one adjoint run
therefore give the derivative of $a$ with respect to the permittivity of
every cell at once.

For a device with $N$ inputs and $M$ output modes, $N$ forward runs and $M$
adjoint runs give the derivative of every entry of the S-matrix at every
frequency, and with it the gradient of any figure of merit that is a function
of the S-matrix.

## One output: `mode_power_gradient`

`lightsim.adjoint.mode_power_gradient(sim, design_min, design_max, output, freq)`
handles the simplest case: one source, one output mode, one frequency, and
the figure of merit $F = |a|^2$, the power in the output mode per unit source
spectrum. `sim` contains the source. `output` is a dictionary that describes
the output plane: `axis`, `position`, `min`, `max`, the mode `profile` and
the `direction` in which the mode leaves.

The example is a silicon strip waveguide. The design region is a box around a
stretch of it, and the question is how the transmitted amplitude changes when
the permittivity of a small block of cladding next to the core is raised.

```python
import numpy as np
import lightsim as ls
from lightsim.adjoint import mode_power_gradient
from lightsim.spec import um

wavelength = 1.55 * um
f0 = ls.C0 / wavelength
x_src, x_out = 0.5 * um, 2.5 * um
window_min, window_max = [x_src, -1.0 * um, -0.7 * um], [x_src, 1.0 * um, 0.7 * um]
block = ([1.4 * um, 0.33 * um, -0.11 * um], [1.6 * um, 0.48 * um, 0.11 * um])


def scene(delta_eps=0.0, mesh=None):
    sim = ls.Simulation(min=[0, -1.2 * um, -0.9 * um], max=[3 * um, 1.2 * um, 0.9 * um],
                        mesh=mesh or ls.AutoMesh(wavelength, steps_per_wavelength=12),
                        run_time=150e-15, background="sio2", shutoff=1e-5)
    sim.add_structure(ls.Box([-1, -0.25 * um, -0.11 * um], [1, 0.25 * um, 0.11 * um]), "si")
    if delta_eps:
        sim.add_structure(ls.Box(*block), ls.Medium("perturbed", eps=1.444**2 + delta_eps))
    return sim


mode = ls.solve_modes(scene(), 0, x_src, window_min, window_max, f0, num_modes=1)[0]
output = {"axis": 0, "position": x_out, "min": [x_out, window_min[1], window_min[2]],
          "max": [x_out, window_max[1], window_max[2]], "profile": mode.profile, "direction": "+"}


def with_source(sim):
    return sim.add_source(ls.ModeSource(0, x_src, "+", window_min, window_max, mode.profile,
                                        ls.GaussianPulse(f0, 0.1 * f0), name="src"))


sim = with_source(scene())
result = mode_power_gradient(sim, [1.2 * um, -0.6 * um, -0.3 * um], [1.8 * um, 0.6 * um, 0.3 * um],
                             output, f0, device="cpu")
print(round(result.fom, 3), {c: g.shape for c, (origin, g) in result.grad.items()})
```

```text
0.983 {'Ex': (16, 25, 13), 'Ey': (17, 24, 13), 'Ez': (17, 25, 12)}
```

The result holds:

| Attribute | Meaning |
|---|---|
| `fom` | $F = \lvert a \rvert^2$ |
| `a_plus` | The complex output amplitude $a$ |
| `grad` | Per electric component, the array origin and $\partial F / \partial\varepsilon$ for every cell of the design box |
| `da` | The same for the complex derivative $\partial a / \partial\varepsilon$ |
| `total_gradient_in_box(sim, min, max)` | $\partial F/\partial\varepsilon$ for a uniform change in a box |
| `amplitude_derivative_in_box(sim, min, max)` | $\partial a/\partial\varepsilon$ for a uniform change in a box |

There is one array per electric field component because each component has
its own grid positions and its own permittivity value.

A finite-difference check changes the permittivity of the block by
$\pm 0.1$ and runs two more simulations. The automatic mesh follows the
materials, so a block of another index moves grid lines, and the difference
of the two runs would then contain the change of the mesh. Both runs
therefore use the grid of the unperturbed device, taken with `frozen_mesh`
(see [Simulation](simulation.md)):

```python
mesh = sim.frozen_mesh()


def amplitude(delta_eps):
    s = with_source(scene(delta_eps, mesh))
    s.add_monitor(ls.ModeMonitor("out", 0, x_out, output["min"], output["max"], [f0], [mode.profile]))
    r = s.run(device="cpu")
    return r.mode_amplitudes("out")[0][0] / r.source_spectrum("src", monitor="out")[1][0]


da_adjoint = result.amplitude_derivative_in_box(sim, *block)
da_difference = (amplitude(+0.1) - amplitude(-0.1)) / 0.2
print(f"magnitude ratio {abs(da_adjoint) / abs(da_difference):.3f}, "
      f"phase difference {np.angle(da_adjoint / da_difference):.2f} rad")
```

```text
magnitude ratio 0.991, phase difference 0.00 rad
```

The two derivatives agree within 1 % in magnitude and 0.001 rad in phase. The
difference in magnitude is the amplitude that the mode source launches on
this coarse mesh, $\sqrt{0.983} = 0.992$ of the ideal one: this function
normalises the adjoint field by the source spectrum, which takes the launched
amplitude to be 1. The S-matrix machinery below calibrates the launched
amplitude with monitors and does not have this factor.

Limits of `mode_power_gradient`:

- One source, one output mode and one frequency per call.
- It works in 3-D and in 2-D or 2.5-D simulations (see
  [2-D, 2.5-D and 3-D](dimensions.md)); there the gradient is per cell of the
  plane, and the port modes come from `lightsim.eim.solve_port_mode`.
- The gradient is with respect to the permittivity of each grid cell. Next to
  an interface with sub-pixel smoothing, the cell permittivity is not a linear
  function of the material permittivity, so a uniform material change is
  predicted less accurately there.

## S-matrix gradients

`lightsim.invdes` generalises this to many ports and frequencies and adds
the pieces needed for optimisation.

`S[m, n, k]` is the amplitude of output channel `m` per unit amplitude of
input `n` at frequency `k`. The design region is described by one or more
grids of permittivity samples, which enter the simulation as `IndexGrid`
structures (see [Materials](materials.md)) placed on top of everything else.

| Object | Role |
|---|---|
| `Port(name, axis, position, min, max, profiles, direction)` | A mode plane. `direction` points **into** the device |
| `port_mode(sim, name, axis, position, min, max, freq, direction, polarization=None)` | Solves the port modes and returns a `Port`. `polarization="te"` or `"tm"` selects the 2.5-D solver, `None` the 3-D one |
| `Excitation(name, source, norm=None)` | An input that is not a port mode: any source, such as a beam |
| `SMatrixProblem(sim, grids, inputs, outputs, freqs, devices="cpu")` | Ties a base simulation to design grids and channels |
| `problem.solve(eps, adjoint=True)` | Runs the simulations for the given grid permittivities and returns an `SMatrixSolution` |
| `solution.S` | Complex array `[M, N, K]` |
| `solution.vjp(g)` | Gradient of a real figure of merit with respect to the grid samples |

The gradient of a real figure of merit $L$ with respect to a complex entry is
passed as

$$
g = \frac{\partial L}{\partial \operatorname{Re} S} + i\,\frac{\partial L}{\partial \operatorname{Im} S},
\qquad\text{so that}\qquad \delta L = \operatorname{Re}\big( g^*\, \delta S \big) .
$$

For $L = |S|^2$ this is $g = 2S$. `solution.vjp(g)` then returns
$\partial L/\partial\varepsilon$ on the samples of every design grid.

Points to know:

- `devices` is a device name or a list. The forward and adjoint runs of one
  `solve` are independent and are distributed over the list, one run at a
  time per entry, for example `devices=["cuda:0", "cuda:1"]`.
- With the default `normalize="monitor"`, every mode source is calibrated by
  two mode monitors just after it, so that `S` is per unit incident mode
  amplitude. The mesh must be uniform along the propagation axis for four
  cells after each port plane.
- Mode profiles are solved at one frequency and used over the band.
- `solve(eps, adjoint=False)` runs only the forward simulations, when the
  S-matrix is needed without a gradient.
- Extra `DFTMonitor` dictionaries passed as `monitors=[...]` are recorded in
  every forward run and returned in `solution.fields`, per unit input.

### Choosing an engine

`SMatrixProblem(..., engine=...)` selects how S and `vjp` are computed. All
three return the same S (to single-precision round-off) and take the same
upstream gradient `g`.

| Engine | Runs per gradient | Gradient | Time, memory (3-D partial etch, RTX PRO 6000) |
|---|---|---|---|
| `"python"` (default) | N forward + M adjoint | continuous adjoint formula: about 1 % in general, a few per cent on boundary parameters | 0.59 s, 642 MiB |
| `"native"` | N forward + M adjoint, buffers kept on the device | same as `"python"` to rounding | 0.25 s, 642 MiB |
| `"reverse"` | N forward + N reverse passes through the time loop | exact derivative of the discrete scheme (finite differences agree to 1e-4 ... 1e-3) | 1.11 s, 2440 MiB |

The adjoint formula `(i w eps0/4) E_adj E_fwd dV` of `"python"` and
`"native"` is the continuum result evaluated on the grid; its error is a
smooth pattern of a few per cent of the local value. It averages out on a
smooth density direction but not on a boundary, where many samples move
together: polygon vertices in 3-D are off by up to 4 %, the sum over the
faces of a binary block by 47 %. `"reverse"` differentiates S as it is
computed, DFT, mode overlap and normalisation included; it costs about
4 times the time and memory of `"native"`, because each reverse pass costs
about three forward runs and the absorbing layers and port planes are
recorded at every step.

A plain recommendation: `"native"` while the design is grey, `"reverse"`
once it is binary (the late, high-`beta` stages) and for shape
optimisation. `"python"` is needed only to split each run over several GPUs
(`devices="cuda:0,1"`) or to record field monitors. `"reverse"` refuses
off-diagonal and oriented media, Bloch boundaries and absorbers it cannot
step backwards, naming the engine to use instead. See the
[engine table](../modules/invdes.md#engines) for the measurements.

### Many ports: combined runs and random probes

An exact S-matrix gradient costs one forward run per input and one adjoint
run per output, 2N runs for an N × N device. For an objective that is a sum
over inputs, such as the distance to a target matrix up to a global phase,

$$
L = \frac{\lVert S - e^{i\varphi} T\rVert_F^2}{\lVert T\rVert_F^2}
  = \frac{\mathbb E_x\,\lVert (S - e^{i\varphi} T)\,x\rVert^2}{\lVert T\rVert_F^2},
\qquad \mathbb E[x x^\dagger] = I ,
$$

(`invdes.matrix_distance`), one random complex input vector $x$ gives an
unbiased estimate of $L$ and of its gradient from two runs: a forward run
with all inputs excited at once with amplitudes $x$, which measures
$y = Sx$, and an adjoint run with the outputs launched together with the
amplitudes $\overline{r}$ of the residual $r = y - e^{i\varphi} T x$.

| Object | Role |
|---|---|
| `problem.solve_combined(eps, weights)` | `P` forward runs, run `p` exciting every input `n` with complex amplitude `weights[p, n]`; returns a `CombinedSolution` |
| `combined.X`, `combined.Y` | the input vectors actually launched `[P, N, K]` and the outputs `[M, P, K]`, with `Y[:, p, k] = S[:, :, k] @ X[p, :, k]` |
| `combined.vjp(g)` | gradient for an upstream `g = dL/dRe Y + i dL/dIm Y`: one adjoint run per probe and frequency |
| `ProbeDesign(problem, design, T, probes=P)` | `value_and_grad(rho, beta)`: the estimated distance and its estimated gradient, `P (1 + K)` runs |

The launched amplitudes are measured at the source planes, so `Y = S X` and
the combined gradient are exact for the vectors that were launched (they
differ from the requested ones by the port-to-port spread of the mode
sources, 0.2 % in the tests; the adjoint weights are corrected for it from
the previous call). The tests compare the combined gradient with the exact
`vjp` of the equivalent upstream gradient on `S` (agreement to 1e-3). The
estimate is noisy: Adam averages it over iterations, and the cost per step,
`2P` runs, does not grow with N. Combined runs need the Python engine and
distinct input and output ports (they do not separate reflections), and one
adjoint run per frequency, so they suit single-wavelength objectives best.
Measured on an 8 × 8 unitary in the 2.5-D model (the matrix-multiplier
scaling example of the examples handbook): at the same number of
simulations, two or four probes per step reach about the same distance as
exact gradients (0.014 and 0.011 against 0.010 after 2 400 to 2 560 runs).
Each combined run excites every input, and a mode source costs a little time
every step, so a combined run of N sources is slower than a one-source run
on a small grid (about three times on a 4 M-cell 2.5-D grid with 64
sources); on 3-D grids the difference is small.

## Parametrisation

Optimising the permittivity of every cell freely produces designs that cannot
be made. `Design` describes the design by a density $\rho \in [0, 1]$ on a
2-D grid, one value per pixel of an etch mask, and maps it to permittivity in
three steps:

$$
\rho \;\xrightarrow{\text{filter}}\; \tilde\rho \;\xrightarrow{\text{projection}}\; \hat\rho
\;\longrightarrow\; \varepsilon = \varepsilon_{\text{lo}} + (\varepsilon_{\text{hi}} - \varepsilon_{\text{lo}})\,\hat\rho .
$$

- The **filter** is a convolution with a cone of radius `filter_radius`. It
  removes features smaller than about twice the radius.
- The **projection** pushes the filtered density towards 0 or 1:
  $\hat\rho = \big(\tanh\beta\eta + \tanh\beta(\tilde\rho - \eta)\big) / \big(\tanh\beta\eta + \tanh\beta(1 - \eta)\big)$.
  A small $\beta$ leaves the design grey and smooth to optimise; $\beta$ is
  raised during the optimisation until the design is nearly binary. $\eta$ is
  the threshold, 0.5 by default.

| Constructor | Design |
|---|---|
| `Design.slab(min, max, (z0, z1), eps_lo, eps_hi, pixel=...)` | A fully etched layer between two heights, for 3-D |
| `Design.partial_etch(min, max, (z0, zmid, z1), eps_lo, eps_hi, pixel=...)` | Two etch depths, for 3-D. Two density fields; the upper level exists only where the lower one does |
| `Design.effective_index(min, max, n_lo, n_hi, pixel=...)` | One layer of a 2.5-D model, between two effective indices |

All take `filter_radius`, `eta`, boolean masks `fixed_solid` and `fixed_void`
for pixels that must stay filled or empty (waveguide stubs), and `symmetry`
(`("x",)`, `("y",)` or both) to mirror the density.

| Method | Returns |
|---|---|
| `design.grids()` | The design grids for `SMatrixProblem` |
| `design.eps(rho, beta)` | The permittivity arrays for `problem.solve` |
| `design.vjp(rho, beta, grads)` | The gradient with respect to `rho`, from the gradient with respect to the permittivity |
| `design.projected(rho, beta)` | $\hat\rho$ |
| `design.binarization(rho, beta)` | 0 for uniform grey, 1 for a binary design |
| `design.random(seed, mean, amplitude)` | A starting density |

## Worked example: a waveguide offset

Two silicon waveguides are offset sideways by 1.2 µm. The task is to design
the 2.4 µm square between them so that light from one enters the other, over
a 40 nm band around 1.55 µm. The 2.5-D effective-index model (see
[Modes and S-parameters](modes-sparams.md)) keeps each simulation to about a
second on a CPU.

```python
from lightsim import eim, invdes

freqs = ls.C0 / (np.array([1.53, 1.55, 1.57]) * um)
n_clad = 1.444
n_core = eim.slab_neff(3.476, 220e-9, wavelength, n_clad)     # slab index of 220 nm silicon

base = invdes.planar_simulation([0, -2.0 * um], [5.6 * um, 2.0 * um], wavelength, n_clad, n_core,
                                steps_per_wavelength=14, run_time=400e-15, shutoff=1e-6)
base.add_medium(ls.Medium("core", n=n_core))
y_in, y_out = -0.6 * um, 0.6 * um
base.add_structure(ls.Box([-5 * um, y_in - 0.25 * um, -1], [1.6 * um, y_in + 0.25 * um, 1]), "core")
base.add_structure(ls.Box([4.0 * um, y_out - 0.25 * um, -1], [12 * um, y_out + 0.25 * um, 1]), "core")


def port(name, x, y, direction):
    return invdes.port_mode(base, name, 0, x, [x, y - 0.8 * um, 0], [x, y + 0.8 * um, 1 * um],
                            f0, direction, polarization="te")


p_in = port("in", 0.8 * um, y_in, "+")          # light enters towards +x
p_out = port("out", 4.8 * um, y_out, "-")       # into the device is towards -x here

design = invdes.Design.effective_index([1.6 * um, -1.2 * um], [4.0 * um, 1.2 * um], n_clad, n_core,
                                       pixel=50e-9, filter_radius=100e-9)
problem = invdes.SMatrixProblem(base, design.grids(), [p_in], [p_out], freqs, devices="cpu")
inverse = invdes.InverseDesign(problem, design, objective=invdes.mean_coupling)

rho0 = design.random(seed=0, mean=0.5)          # uniform grey
value, gradient, info = inverse.value_and_grad(rho0, 2.0)
print(design.shape, info["S"].shape, info["simulations"], round(value, 3))
```

```text
(49, 49) (1, 1, 3) 2 0.05
```

`InverseDesign` joins the problem, the parametrisation and an objective.
`value_and_grad(rho, beta)` runs one forward and one adjoint simulation and
returns the figure of merit, its gradient with respect to the density and
some diagnostics. The objective `mean_coupling` is the transmitted power
averaged over the frequencies. The grey starting design transmits 5 %.

The gradient can be checked against a finite difference along a smooth
random direction, which costs two more forward runs:

```python
rng = np.random.default_rng(0)
direction = invdes.filter_density(rng.standard_normal(rho0.shape), invdes.conic_kernel(4))
direction /= np.abs(direction).max()
h = 0.01


def merit(rho):
    return invdes.mean_coupling(inverse.s_matrix(rho, 2.0))[0]


from_adjoint = float(np.sum(gradient * direction))
from_difference = (merit(rho0 + h * direction) - merit(rho0 - h * direction)) / (2 * h)
print(f"{from_adjoint / from_difference:.2f}")
```

```text
0.98
```

The two agree within 2 %.

A plain central difference is enough at this small $\beta$. For a tighter
check of the whole chain (filter, projection, S-matrix, objective), and at
larger $\beta$, where the projection makes the figure of merit strongly
curved in the density, combine two steps by Richardson extrapolation:
`invdes.directional_derivative(merit, rho0, direction, h)` returns
$(4 D(h) - D(2h))/3$ from the central differences $D$ with steps $h$ and
$2h$, whose error is of order $h^4$ instead of $h^2$, for four evaluations
of the figure of merit. Keep $h$ well above the single-precision noise of
the runs (two runs of one design differ in their last bits): about 0.01 of
the density range, 0.001 in the double-precision build.

The arrays of `S` have the axes (outputs, inputs, frequencies): `S[m, n, k]`
is the amplitude in output `m` for a unit excitation of input `n` at the
frequency `freqs[k]`.

`optimize` runs the loop. Here it takes ten steps of the Adam method and
doubles $\beta$ after five:

```python
rho, history = invdes.optimize(inverse, rho0, 10, beta=invdes.beta_schedule(2, 2, 5, 8),
                               lr=0.1, verbose=False)
print(np.round(history["fom"], 2))
print(round(float(history["binarization"][-1]), 2))
```

```text
[0.05 0.3  0.51 0.69 0.83 0.88 0.74 0.92 0.9  0.89]
0.64
```

The transmission rises from 5 % to about 90 % in ten iterations. The dip
after the fifth is where $\beta$ doubles and the optimiser restarts. A real
design run continues for a few hundred iterations while $\beta$ increases
further, until the binarisation is close to 1.

The optimised density is turned into polygons for a layout with
`density_polygons`. It needs matplotlib's `contourpy` package.

```python
shapes = invdes.density_polygons(design.projected(rho, 8.0)[0], design.min, design.max, level=0.5)
exterior, holes = shapes[0]
print(len(shapes) >= 1, exterior.shape[1])
```

```text
True 2
```

Each entry is an outline, as an array of (x, y) points in metres, and a list
of holes.

To optimise a few shape parameters instead of every pixel (polygon
vertices, the width profile of a taper), wrap the design in
`invdes.ShapeDesign`; see [Shape optimisation](../modules/shapes.md).

> **Note:** A design optimised on index grids should be simulated again as
> polygons, where sub-pixel smoothing applies, and a design optimised in
> 2.5-D must be verified in 3-D.

## Objectives

An objective is a function of `S` that returns the value and the gradient
`g` in the convention above. The module provides:

| Function | Figure of merit |
|---|---|
| `mean_coupling(S, weights=None)` | $\lvert S \rvert^2$ averaged over inputs, outputs and frequencies |
| `band_transmission(S, want, crosstalk=1.0, alpha=None)` | Transmission where `want` is true minus a penalty where it is false; with `alpha`, a smooth minimum over the wanted entries instead of their mean |
| `matrix_fidelity(S, target)` | Overlap of `S` with a target matrix, 1 when they agree up to a phase per frequency |
| `matrix_metrics(S, target)` | Diagnostics for the same comparison: fidelity, throughput, phase |

All are written to be maximised. Your own objective is any function with the
same return values, for example `lambda S: (float(np.sum(np.abs(S) ** 2)), 2 * S)`.

### A quantum figure of merit

The objective does not have to be a classical power. What a device does to
quantum states of light is also a function of its `S`: photons interfere
through the same matrix, so photon statistics, heralding probabilities and
state fidelities are functions of its entries, and their gradients go back to
the design through the same adjoint runs.

The simplest case is two identical photons entering the two inputs of a
2 × 2 device. The amplitude for one photon to leave by each output is the
permanent of the matrix, $S_{00}S_{11} + S_{01}S_{10}$; in a balanced coupler
the two terms cancel (Hong-Ou-Mandel interference). The coincidence
probability and its gradient in the convention above:

```python
import numpy as np


def coincidence(S):
    """Two identical photons into ports 0 and 1: probability that one leaves
    by each output, averaged over the frequencies; to be minimised."""
    c = S[0, 0] * S[1, 1] + S[0, 1] * S[1, 0]  # permanent, one value per frequency
    K = S.shape[2]
    g = np.zeros_like(S)
    g[0, 0], g[1, 1] = 2 * c * np.conj(S[1, 1]) / K, 2 * c * np.conj(S[0, 0]) / K
    g[0, 1], g[1, 0] = 2 * c * np.conj(S[1, 0]) / K, 2 * c * np.conj(S[0, 1]) / K
    return float(np.mean(np.abs(c) ** 2)), g


rng = np.random.default_rng(0)
S = (rng.standard_normal((2, 2, 3)) + 1j * rng.standard_normal((2, 2, 3))) / 2
dS = 1e-6 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
value, g = coincidence(S)
print(f"{np.sum(np.real(np.conj(g) * dS)):.6e} {coincidence(S + dS)[0] - value:.6e}")
```

```text
2.359899e-07 2.359903e-07
```

Passed as `InverseDesign(problem, design, objective=coincidence)` and
minimised (`optimize(..., maximize=False)`), it designs a coupler for
two-photon interference without ever naming a splitting ratio. The test suite
checks this gradient through the whole chain against finite differences: the
default adjoint engine is within 6 % (the objective is a product of entries of
`S`, each good to a few per cent), and `engine="reverse"` gives it to 1e-4,
which is the engine to use when a figure of merit depends on the phases
between entries.
Larger figures of merit (photon-number distributions, Gaussian states,
heralded gates) have the same form: a value and its gradient with respect to
every entry of `S`.

## The optimiser

`optimize(fn, rho0, iterations, beta=..., method="adam", lr=0.02, checkpoint=None)`

| Parameter | Meaning |
|---|---|
| `fn` | An `InverseDesign`, or a function `(rho, beta)` returning `(value, gradient, info)` |
| `beta` | A number, or a function of the iteration such as `beta_schedule(start, factor, every, maximum)` |
| `method` | `"adam"` (projected Adam, restarted when `beta` changes) or `"lbfgs"` (L-BFGS-B from scipy, one stage per `beta`) |
| `lr` | Step size of Adam |
| `checkpoint` | Path of an `.npz` file. The state is saved every iteration and the run resumes from the file if it exists |
| `maximize` | Maximise (default) or minimise |
| `callback` | Called as `callback(iteration, rho, value, beta, info)` |

It returns the final density and a history with `fom`, `beta`,
`binarization`, `seconds`, `sim_seconds`, and the best design found as
`best_rho` and `best_fom`.

`InverseDesign(problem, design, objective, robust=d)` evaluates two more
variants of the design at thresholds $\eta \pm d$, which correspond to
over- and under-etched devices, and averages the objective over the three.
This triples the number of simulations.

## PyTorch and JAX

The S-matrix can be used as a differentiable function inside a PyTorch or JAX
computation, so that the figure of merit is written in the framework and
differentiated automatically. Neither framework is a dependency of lightsim;
the wrappers are imported on demand.

```python
import torch
import lightsim.invdes.torch as lt

rho_t = torch.tensor(rho0, dtype=torch.float64, requires_grad=True)
S = lt.s_matrix(inverse, rho_t, beta=2.0)          # complex tensor [M, N, K]
loss = (S.abs() ** 2).mean()
loss.backward()
print(round(loss.item(), 3), bool(np.allclose(rho_t.grad.numpy(), gradient, rtol=1e-6, atol=1e-12)))
```

```text
0.05 True
```

The gradient that PyTorch returns equals the one from `value_and_grad`
above.

```python
import jax
import jax.numpy as jnp
import lightsim.invdes.jax as lj

jax.config.update("jax_enable_x64", True)
loss_fn = lambda r: jnp.mean(jnp.abs(lj.s_matrix(inverse, r, beta=2.0)) ** 2)
value_j, grad_j = jax.value_and_grad(loss_fn)(jnp.asarray(rho0))
print(round(float(value_j), 3), bool(np.allclose(np.asarray(grad_j), gradient, rtol=1e-6, atol=1e-12)))
```

```text
0.05 True
```

The simulations run outside the framework. With JAX, `grad` and `vjp` work;
`jit` and `vmap` do not.

## Limits

- The design grids are painted last. A fixed feature inside the design region
  must be part of the density, through `fixed_solid`, not a structure of the
  base simulation.
- Design media are lossless and non-dispersive, because index grids are. A
  uniform conductivity can be set with `Design(..., medium_sigma=...)`.
- Index grids are not sub-pixel smoothed at the faces of their box and
  interpolate permittivity linearly between samples.
- The gradient does not include a change of the sources with the design.
  Keep ports and sources outside the design region.
- `engine="reverse"` keeps the absorbing shell and the port planes of every
  time step in memory, so its memory grows with the run length; it has no
  per-voxel gradient (`yee_vjp`) and no field monitors.
- The 2.5-D model inherits the limits of the effective-index method (see the
  [effective-index notes](../modules/eim.md)).
