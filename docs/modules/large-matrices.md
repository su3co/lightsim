# Very large matrix multipliers: physics, bounds and solvers

This page asks what it takes to inverse-design a passive photonic chip whose
transmission matrix is a prescribed $N \times N$ matrix when $N$ is in the
thousands or, as an extrapolation, $150\,000$; which simulation and gradient
methods scale to that, under which approximations; and what was measured.
The solver that came out of it is described in
[Composite S-matrix problems](composite.md).

## What an N-port matrix device has to be

The numbers below use silicon (220 nm) in oxide at 1550 nm: slab index
$n \approx 2.85$, single-mode wires 0.5 µm wide at a pitch $p$ of
1 to 1.5 µm so that neighbouring wires do not couple over the device.

**Width.** $W = N p$: 1.5 mm for $N = 1000$, 15 to 22 cm for
$N = 150\,000$. A lithographic reticle field is about $26 \times 33$ mm, so a
single device of $150\,000$ ports in one row does not fit on one die.

**Degrees of freedom and area.** A general complex $N \times N$ matrix has
$2N^2$ real parameters (a unitary $N^2$). A free-form region carries about
one independent real parameter per resolvable pixel of side
$\lambda / 2n \approx 0.27$ µm, so its area must be at least
$A_{\min} \approx 2N^2 (\lambda/2n)^2 \approx 0.15\,N^2$ µm². At
$N = 150\,000$ this is $3.3 \times 10^{9}$ µm², 33 cm²: four reticles of
perfectly efficient design, before any margin.

**Depth.** For every input to reach every output the light has to spread
over the whole width. Two bounds apply. Geometrically, light confined to
angles $|\theta| < \theta_{\max}$ needs a length
$L \gtrsim W / (2\tan\theta_{\max})$, of order $W$. From the counting
argument of D. A. B. Miller ("How complicated must an optical component
be?", J. Opt. Soc. Am. A 30, 238 (2013)), a cut along the propagation
direction through the middle of the device must carry as many independent
channels as the rank of the coupling between the upper inputs and the lower
outputs, about $N/2$ for a general matrix, and a cut of length $L$ carries
about $2nL/\lambda$ channels, so $L \gtrsim N\lambda/(4n) \approx 0.14\,N$
µm. Both grow linearly with $N$; with the area bound, a general free-form
device is roughly square, $W \times L \sim N p \times N p/2$.

**Loss.** A free-form scatterer of depth $L$ loses light out of the plane at
every scattering event and through sidewall roughness. Optimised 3-D designs
of a few wavelengths lose about 1 dB (the 4 × 4 example in
lightsim-examples: 83 % throughput). Over a depth of millimetres, as a
general $N = 1000$ device needs, the loss of the order of a few dB per
millimetre of a strongly scattering silicon region is not established; at
$N = 150\,000$ the required depth is centimetres and the loss is far beyond
any useful budget.

**What this means.** A general $150\,000 \times 150\,000$ matrix
($2.25 \times 10^{10}$ complex entries) is not one free-form device. It is
plausible only as an architecture of smaller blocks:

| Architecture | Parameters | Optical depth | Loss grows with |
|---|---|---|---|
| One free-form device | $2N^2$ | $\propto N$ | $N$ (depth) |
| Mesh of $n$-port blocks (brick wall, universal for unitaries) | $\ge N^2$ | $\ge N/n$ layers | number of layers, $\propto N/n$ |
| Tiles: $(N/n)^2$ independent $n \times n$ devices, inputs re-modulated or fanned out, outputs summed (electronically after coherent detection, or optically) | $2N^2$ | one block | fan-out/fan-in loss $(N/n)^2$ if done optically; none if inputs are re-modulated per tile |

The requirements relax for structured matrices. A unitary needs half the
parameters. A rank-$r$ matrix factors as $U \Sigma V^\dagger$ with
$2Nr$ parameters and depth $\propto r$. A banded or convolutional matrix of
bandwidth $b$ needs only a depth $\propto b$, so the light spreads over $b$
ports, not $N$. Products of $O(\log N)$ sparse layers (Fourier, Hadamard,
butterfly matrices) need $O(N \log N)$ parameters.

## Lower bounds on any algorithm

Every method must at least read and update the design parameters, $\Theta(N^2)$
for a general matrix by the area bound, and, for an objective that compares
with an explicit target, touch its $N^2$ entries. One gradient step therefore
costs $\Omega(N^2)$. Monolithic FDTD costs, in 2-D (and in a 3-D slab of
fixed thickness), $N^2$ cells times $\propto N$ time steps per run, and an
exact gradient of all $N^2$ matrix entries needs $N$ forward and $N$ adjoint
runs: $\Theta(N^4)$. A random-probe gradient (all inputs at once, the adjoint
driven by the output residual) removes one factor $N$: $\Theta(N^3)$ per
stochastic gradient. "Much better scaling" therefore means moving from $N^4$
(or $N^3$) towards $N^2$.

## Methods examined

Each method below was judged on its theory (what it approximates, with what
error, when it is valid), on a prototype and on measurements. The
prototypes are in `benchmarks/scaling/` of the repository; the 2-D ones use
the effective-index model of 220 nm silicon (slab index 2.85, oxide 1.444)
at 1550 nm on a 33 nm grid. "Contrast" below scales every permittivity
difference from the slab value: 1 is silicon and oxide, 0.1 a shallow etch
or a silicon nitride platform.

### 1. Activity-windowed FDTD (only where the wave is)

**Idea.** The domain is cut into tiles; a tile is updated only while its
energy density is above a fraction $\tau$ of the largest seen so far,
dilated by one tile so that the wave cannot cross an inactive tile between
two checks (C. R. Doerr, IEEE Photon. Technol. Lett. 25, 2259 (2013), for
waveguide circuits). Frequency-domain accumulators for the gradient live
only where the design is.

**Error certificate.** The Yee scheme conserves a discrete energy in a
lossless medium and absorbing layers only remove energy. Dropping the
fields of a tile, or the field that would have entered an inactive tile,
injects their negative into the exact scheme; by linearity the difference
between the windowed and the exact run is the response to these
injections, and its discrete energy norm is at all times at most the sum of
the energy norms of everything dropped. The prototype accumulates that sum
during the run (`bound` below, relative to the peak field energy norm): a
computable a-posteriori bound on the field error in the energy norm. It is
not a bound on a port amplitude relative to that amplitude: when little
light reaches a port, the relative error there can exceed the bound.

**Measured** (NumPy prototype, tiles of 30 cells, checked every 8 steps;
"speed-up" counts updated cells; "equal cost" is the error of the ordinary
run stopped after the same number of cell updates):

| Device | $\tau$ | Speed-up | Port error | Equal-cost error | Bound |
|---|---|---|---|---|---|
| 4 × 4 optimised multiplier (lightsim-examples i1), all 4 inputs | 1e-4 | 3.5 | 1.1e-2 | 0.17 | 0.46 |
| | 1e-6 | 2.2 | 9.4e-4 | 7.9e-3 | 0.044 |
| | 1e-8 | 1.6 | 1.0e-4 | 4.6e-4 | 0.004 |
| Random binary 8-port, depth 6 µm (silicon) | 1e-6 | 2.7 | 2.1e-2 | 6.6e-2 | 0.31 |
| | 1e-8 | 1.8 | 2.3e-3 | 5.9e-3 | 0.023 |
| Beam through a random region, contrast 0.1, depth 10 / 20 / 40 µm | 1e-4 | 4.5 / 7.8 / 14.2 | 3.9e-3 / 4.3e-3 / 6.7e-3 | 1 / 1 / 1 | |
| Same, contrast 1 (silicon), depth 10 / 20 / 40 µm | 1e-6 | 2.4 / 3.1 / 5.0 | 9e-3 / 0.15 / 1.0 | | |
| | 1e-8 | 1.7 / 1.9 / – | 7e-4 / 1e-2 / – | | |

**Verdict.** Where the light passes through once (weak contrast, feed-forward
layouts) the window works as hoped: the speed-up grows in proportion to the
depth (the active band has a fixed width) at a fixed error of a few 1e-3,
where the ordinary run at the same cost has not yet reached the output. In
strongly scattering silicon designs, optimised or not, it gives a factor
1.5 to 3 at errors of 1e-3 to 1e-2: light reverberates over the whole
region until it leaves, and what reaches the outputs is attenuated so much
that a threshold relative to the peak must fall with the depth. That is a
constant factor, not a better scaling. On a GPU, where irregular tiles cost
more per cell than the dense fused update, the gain would be smaller still.
The forward and adjoint windows are at different places at different times,
but that does not matter for the gradient, which needs only the
frequency-domain fields in the design region (accumulated where a tile is
active, zero elsewhere).

### 2. Frequency-domain solution with one factorisation for all ports

**Idea.** At one frequency the field solves a sparse linear system
$A(\varepsilon) u = b$. In 2-D a nested-dissection ordering factorises
$A$ in $O(P^{3/2})$ operations and $O(P \log P)$ memory for $P$ cells
(A. George, SIAM J. Numer. Anal. 10, 345 (1973)), after which every
right-hand side costs $O(P \log P)$. All $N$ forward and $N$ adjoint
problems of an exact gradient share the factors (for reciprocal media the
adjoint system is the transpose). With $P \propto N^2$ the exact gradient
costs $O(N^3)$ instead of FDTD's $O(N^4)$; the frequency count multiplies
it, where FDTD gets every frequency from one run.

**Measured** (scipy SuperLU, one thread, geometric nested dissection;
random binary N-port device of depth equal to its width; the operator
agrees with the FDTD prototype to 2 % on the optimised 4 × 4 device):

| N | Cells | Factor (s) | Solve per right-hand side (s) | Factor memory (GB) |
|---|---|---|---|---|
| 8 | 2.1e5 | 2.7 | 0.16 | 0.30 |
| 16 | 6.6e5 | 13.0 | 0.50 | 1.1 |
| 32 | 2.4e6 | 74 | 2.5 | 4.8 |

Fitted against the number of cells: factorisation $\propto P^{1.36}$, solve
$\propto P^{1.12}$, memory $\propto P^{1.14}$, consistent with the theory.
One exact gradient at N = 32 and one frequency: 74 s + 64 × 2.5 s = 234 s
on one core, against 64 FDTD runs on 32 cores (see the scaling section).
SuperLU's 32-bit indices stop it at N ≈ 40 here; a parallel multifrontal
solver would go further, but the memory, $\approx 5\,(N/32)^{2.3}$ GB, reaches
the 246 GB of the host near N = 180. It applies to 2-D and 2.5-D models
only: in 3-D the separators are surfaces and the factorisation of a slab
device grows as $(N h)^3$ with $h$ cells through the stack.

### 3. One-way propagation and the Bremmer series

**Idea.** March the field through the device column by column with the
square-root propagator $\exp(i\Delta x\sqrt{\partial_y^2 + k_0^2\varepsilon})$,
at the cost of one cross-section per step, differentiable by reversing the
march. This is the leading term of the Bremmer series, which expands the
full solution in the number of reflections (H. Bremmer, Commun. Pure Appl.
Math. 4, 105 (1951); the series converges for weakly reflecting media).

**Measured** against the frequency-domain solution (TM, a beam launched in
the slab, the field compared on a plane behind the design, so only the
design scatters; the propagator is exact for each column, the error is
the reflections left out):

| Contrast | Free-form, depth 3 µm | 6 µm | 12 µm | Thin layers (metalines), 12 µm |
|---|---|---|---|---|
| 0.003 | 7e-4 | 9e-4 | 8e-4 | 1e-4 |
| 0.01 | 2.2e-3 | 3.1e-3 | 2.6e-3 | 5e-4 |
| 0.03 | 6.6e-3 | 9.2e-3 | 7.7e-3 | 1.4e-3 |
| 0.1 | 2.3e-2 | 3.1e-2 | 2.6e-2 | 4.9e-3 |
| 0.3 | 8.4e-2 | 0.11 | 0.14 | 2.4e-2 |
| 1 (silicon) | 0.83 | 0.97 | – | 0.32 |

(relative error of the output field, mean over four input positions). The
error grows linearly with the contrast and tracks the reflected amplitude
(about a third of it). **Verdict:** one-way methods are valid, at the 1 %
level, for free-form designs of contrast below about 0.03 (index
modulation $\Delta n \lesssim 0.03$) and for stacks of thin scattering
layers up to contrast 0.1 to 0.2: on-chip diffractive layers and metalines
in a slab, not binary silicon scatterers, where they fail completely. In
the silicon effective-index model even the facets between the slab and the
waveguide array reflect 11 % of the power, so a waveguide-coupled device
fails for the same reason.

### 4. Sweeping and domain decomposition in the frequency domain

Not prototyped. Sweeping preconditioners (B. Engquist and L. Ying, Commun.
Pure Appl. Math. 64, 697 (2011)), polarized traces (L. Zepeda-Núñez and L.
Demanet, J. Comput. Phys. 308, 347 (2016)) and optimized Schwarz methods
with PML transmission conditions (M. J. Gander and H. Zhang, SIAM Rev. 61,
3 (2019), a review) solve slab by slab with absorbing transmission
conditions. Each sweep is exact for waves travelling one way and leaves out
the reflections that come back across the slab interfaces, which is the
one-way approximation above applied as a preconditioner; the iteration
count is then governed by the reflected amplitude (0.3 to 0.7 for silicon
designs in the table above) and is known to grow for high-contrast and
resonant media. Memory is that of one slab, but the exact gradient still
needs 2N solves, each $O(P \log P)$ times the iteration count: no better
than method 2 in time, better in memory.

### 5. Fewer simulations per gradient

A random-probe gradient (all inputs at once with random amplitudes, the
adjoint driven by the residual) removes the factor N in the number of runs
for a squared-error objective; it is being measured on the `invdes-probes`
branch of lightsim. It combines with every method here.

### 6. Composition of blocks by scattering matrices

Solve small blocks exactly (FDTD with its adjoint), compose them by network
theory, carry the gradient through the composition. Exact when all interface
modes are kept; the approximation is keeping only the guided modes of the
waveguides that join blocks. This is the method that was built into
lightsim: [Composite S-matrix problems](composite.md) gives the theory, the
interface and the validation; the measurements are summarised below.

<!-- composite measurements -->

## The road to 150 000 × 150 000

This section is an extrapolation from the measurements, with its
assumptions stated; nothing at this size was run.

**Physics first.** From the first section: one free-form device needs
≥ 33 cm² of perfectly used design area and a depth of centimetres; a mesh of
blocks universal for unitaries needs ≥ $N/n$ layers (37 500 layers of
4-port blocks), whose loss at even 0.1 dB per layer is thousands of dB. The
only architecture that makes physical sense is **tiling**: $(N/n)^2$
independent $n \times n$ devices, each computing one block $W_{ij}$ of the
matrix, with the inputs re-modulated (or fanned out) per tile row and the
outputs summed after coherent detection. With $n = 64$ that is 5.5 million
tiles, $N^2/n = 3.5 \times 10^8$ modulators and as many detectors; with
$n = 1024$, 21 500 tiles and $2.2 \times 10^7$. Larger tiles save
electronics, which is the reason to want large $n$.

**Simulation cost has an $N^2$ floor.** Whatever the solver, every matrix
entry is carried by some area of design that must be simulated. The
composite method reaches that floor: its cost per gradient is proportional
to the number of blocks. Measured on the 2.5-D model, one exact gradient of
one 4-port block (4 forward and 4 adjoint runs, 7.3e4 cells each) is
$2.1 \times 10^9$ cell updates; a mesh holds about $N^2/8$ such blocks
for depth $N/2$. In 3-D the same block has about 40 times more cells
(the film and 0.7 µm of cladding on each side at 44 nm), about
$8 \times 10^{10}$ cell updates per block gradient, or 7 s at the
12 Gcells/s lightsim reaches on one RTX PRO 6000 for large 3-D grids
(`docs/benchmarks.md`; small blocks batched into one layer simulation are
needed to approach that rate).

| Quantity | General 150 000², every block designed | Library of 100 block designs, tiles assembled from them |
|---|---|---|
| Blocks to simulate per iteration | $N^2/8 = 2.8 \times 10^9$ | 100 |
| Full-wave cost per iteration (3-D) | $2.3 \times 10^{20}$ cell updates, ≈ 600 GPU-years | 700 GPU-s |
| Full-wave cost of 300 iterations | ≈ 180 000 GPU-years | ≈ 60 GPU-hours |
| Network optimisation (phases of the waveguide sections between blocks, analytic) | – | $2.25 \times 10^{10}$ real parameters (90 GB in single precision); one gradient by random probes $O(N^2 K)$, a few seconds on a few GPUs; an explicit $T$ (180 GB in complex64) costs $O(N^3) \approx 10^{16}$ flops |
| Storage | target matrix 180 GB (complex64), designs $\approx 10^{12}$ pixels | target 180 GB, 100 block designs, phase table 90 GB |

So: designing every block of a general $150\,000 \times 150\,000$
matrix by full-wave simulation is out of reach of any cluster, even with a
solver at the $N^2$ lower bound (which the composite method is), because
the constant is the cost of simulating a few µm² of silicon per 16 matrix
entries. What is computable is the architecture the physics asks for anyway:
tiles built from a library of full-wave designed blocks (tens of GPU-hours
with the composite solver or plain lightsim), with the per-tile variety in
the waveguide phase sections between blocks, which are computed analytically
or from a 1-D mode model, and the network optimisation done as distributed
linear algebra.

On the owner's GPUs (two RTX PRO 6000, three GB10), with every block
designed:

| Device | Blocks | Composite cost per iteration (2.5-D) | Monolithic exact gradient |
|---|---|---|---|
| Mesh, $N = 1024$, depth 8 (banded or structured matrix) | 2 048 | $4.3 \times 10^{12}$ cell updates, minutes | $\approx 3 \times 10^{15}$ (2 048 runs of a 1.5 mm × 55 µm domain), days |
| Mesh, $N = 1024$, depth 512 (universal) | 131 000 | $2.8 \times 10^{14}$, hours | weeks per iteration |
| Mesh, $N = 128$, depth 64 (universal) | 2 000 | $4 \times 10^{12}$, minutes | ≈ $4 \times 10^{15}$, a day per iteration |

"Thousands of ports" with every block designed is therefore within reach
for structured matrices (shallow meshes) and, for universal meshes, at the
low thousands only at hours per iteration in 2.5-D.
