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

<!-- measurements and per-method sections follow -->
