# Composite S-matrix problems

`lightsim.invdes.CompositeProblem` computes the S-matrix of a large device
and its gradient by splitting the device into blocks that meet only at
waveguide ports. Every block is simulated on its own by FDTD, forward and
adjoint, as `SMatrixProblem` does for a whole device; the blocks are joined
by scattering-matrix composition; and the gradient of the figure of merit is
carried back through the composition to the design of every block. A
`CompositeProblem` has the interface of an `SMatrixProblem` (`solve`,
`S`, `vjp`), so `InverseDesign` and `optimize` run on it unchanged.

It is not a replacement for FDTD of the whole device: it is an approximation
whose error is stated below, measured against monolithic FDTD, and can be
checked with it on any design small enough.

## What is approximated

A block is a design region with single-mode waveguides on its ports. The
port planes lie on the connecting waveguides, and two blocks joined at a
port share that plane and its mode profile, so what leaves one block in the
guided mode enters the other. With all block S-matrices gathered into one
block-diagonal matrix $S$, the internal ports $I$, the external ports $E$
and the connection matrix $C$,

$$
T = S_{EE} + S_{EI}\, C\, (1 - S_{II} C)^{-1} S_{IE} .
$$

This is exact for the block S-matrices it is given (network theory; the
geometric series of $(1 - S_{II}C)^{-1}$ orders it by the number of
internal reflections, R. Redheffer, J. Math. Phys. 41, 1 (1962)). What a
block S-matrix leaves out, and therefore the whole approximation, is:

1. **Non-guided field at the port planes.** The field a block scatters
   includes evanescent and radiation modes of the connecting waveguide. At
   a distance $d$ from the design region the evanescent part has decayed as
   $e^{-\kappa d}$, and the radiation has spread out of the window; the
   port overlap measures only the guided mode. Mode-matching theory makes
   this the truncation of the interface basis to one mode, with an error
   that falls with $d$ (the `stub` of the examples).
2. **Light that one block scatters out of its ports and another picks up**,
   through the cladding (radiation) or by evanescent coupling between
   neighbouring waveguides and design regions. Each block is simulated with
   absorbing boundaries where its neighbours would be. A bound follows from
   energy conservation and reciprocity: the amplitude that block $A$ sends
   into port $q$ of block $B$ through non-guided paths is at most
   $\sqrt{\ell_A\,\ell_{B,q}}$, where $\ell_A$ is the fraction of power $A$
   scatters out of its ports and $\ell_{B,q}$ the fraction $B$ scatters
   when driven backwards from $q$ (Cauchy–Schwarz on the reciprocal
   overlap integral of the two radiated fields). Low-loss blocks therefore
   couple little; lossy, strongly scattering blocks can couple a lot.

The gradient is exact for the composed model: with
$R = [1,\; S_{EI} C (1 - S_{II} C)^{-1}]$ and
$B = [1;\; C (1 - S_{II} C)^{-1} S_{IE}]$, $\delta T = R\,\delta S\,B$, so
the upstream gradient $g$ of $T$ becomes $R^H g B^H$ on the block
S-matrices, and each block's adjoint simulation turns its part into a
gradient on its design. The adjoint formula inside each block is the one of
`SMatrixProblem` (see [Inverse design](../guide/inverse-design.md)).

## Usage

The building blocks:

| Object | Role |
|---|---|
| `Network(ports, connections, external, sides=None)` | Topology: ports per block, joined port pairs `((b, p), (b2, p2))`, external ports in order |
| `network.solve(S_blocks, reflections=True)` | `T[e, e2, k]` and `vjp(g)` to the block S-matrices |
| `chain([T1, T2, ...])` | Feed-forward cascade of transmission matrices and its pullback |
| `Block(problem, in_ports, out_ports, ports)` | A block: an `SMatrixProblem` of the block alone (or a fixed S-matrix array), and which block ports its inputs and outputs are |
| `CompositeProblem(blocks, network, inputs, outputs, devices, reflections)` | The device: `solve(eps)` returns `S` between the chosen external ports and `vjp(g)` |
| `CompositeDesign(designs)` | The blocks' `Design`s as one design for `InverseDesign` |

A block's `SMatrixProblem` lists the block's ports as inputs and outputs.
For the full model it lists all ports on both sides as both inputs and
outputs ($2n$ forward and $2n$ adjoint runs for $n$ ports per side); for the
feed-forward model only the left ports as inputs and the right ports as
outputs ($n + n$ runs), and `CompositeProblem(..., reflections=False)` drops
the reflections between blocks, the zeroth order of the multiple-reflection
series.

Port profiles must be identical arrays with the same sign at the two sides
of a connection, and the grids of the block and of any monolithic check
must be aligned: the examples put every coordinate on a multiple of the
cell size and solve the waveguide mode once. Waveguides should run through
the block under the design grid rather than end at its edge, so that the
block and the assembled device have the same permittivity at the design
boundary.

**Relation to `lightsim.circuit`.** The network is the netlist of a
circuit: blocks are instances, connections are pairs of instance ports and
the external ports are the circuit's ports, with the conventions of
[Circuits](circuit.md) (`S[out, in]`; a block entry `S[p, q, k]` is the
SDict entry `(q, p)`). The forward solve is `lightsim.circuit.solve_matrix`,
the solver of `circuit`; `Network.netlist(S_blocks)` returns the same
network as `(instances, connections, ports)` for
`lightsim.circuit.solve_network`, which gives the same $T$ (a test checks
this on a mesh). What `Network` adds is the gradient. Ports are numbered
here, not named, and every connected pair must share its reference plane,
because a block S-matrix computed by FDTD is referred to its own port planes
(no waveguide segment is inserted between blocks).

A worked example (a mesh of 4-port blocks in the 2.5-D model, optimised to
the 8-point DFT) is in `benchmarks/scaling/` of the repository:
`mesh2d.py` builds the blocks, the layers and the monolithic device;
`optimize_mesh.py` optimises; `validate_mesh.py`, `grad_check.py` and
`compare_layers.py` compare with monolithic FDTD.

```python
import numpy as np
from lightsim import invdes

# a two-block chain: 4 ports per block, left ports 0..1, right ports 2..3
net = invdes.Network([4, 4], [((0, 2), (1, 0)), ((0, 3), (1, 1))],
                     external=[(0, 0), (0, 1), (1, 2), (1, 3)],
                     sides=[["in", "in", "out", "out"]] * 2)
rng = np.random.default_rng(0)
S = [0.5 * (rng.standard_normal((4, 4, 1)) + 1j * rng.standard_normal((4, 4, 1))) for _ in range(2)]
sol = net.solve(S)
g = np.zeros_like(sol.T)
g[2:, :2] = 2 * sol.T[2:, :2]            # d|T_out,in|^2
grads = sol.vjp(g)                        # one [4, 4, 1] array per block
print(sol.T.shape, [x.shape for x in grads])
```

```text
(4, 4, 1) [(4, 4, 1), (4, 4, 1)]
```

## How to check a result with FDTD

Build the assembled device as one `SMatrixProblem` whose design grids are
the blocks' grids at their positions (`mesh2d.monolithic_problem` does this
for meshes), solve it once with the final design, and compare its S-matrix
with `CompositeProblem.solve(...).S`. On a device too large for that, check
a sub-mesh (a few layers, a few blocks wide) monolithically, the energy
balance of every block ($\sum_p |S_{pq}|^2 \le 1$ and its loss
$\ell$, which bounds the inter-block coupling as above), reciprocity of
every block ($S = S^T$ for reciprocal media), and convergence in the port
distance `stub` and in the separation of neighbouring blocks.

## When not to use it

- Free-form devices whose regions interact through the field everywhere,
  without waveguides between them: cut them into blocks and the
  interaction through the cut is lost. For those, use monolithic FDTD.
- Blocks that scatter a large fraction of the light out of their ports
  (random or early grey designs: 60 to 80 % loss in the measurements): the
  composite S-matrix is then off by tens of per cent. It is accurate for
  blocks that guide most of the light, which is what an optimised design
  does; check the final design with FDTD.
- Closely spaced waveguides of neighbouring blocks: their mutual coupling is
  a non-port path. Keep them apart (1 µm of oxide between 0.5 µm silicon
  wires in the 2.5-D model gives 1e-3 over the 13 µm of a two-layer mesh;
  0.5 µm gives 4e-2).

## Measured accuracy

Brick-wall meshes of 4-port free-form blocks in the 2.5-D model (220 nm
silicon, 1550 nm, 33 nm cells, ports at 1.5 µm pitch, 1 µm of oxide
between the design regions of neighbouring blocks, port planes 1.2 µm from
the design regions), composite against the monolithic FDTD of the same
layout and permittivity, relative Frobenius norm of the difference of the
$N \times N$ transmission matrices:

| Mesh | Blocks | Mesh loss $1 - \lVert T \rVert^2/N$ | Composite error |
|---|---|---|---|
| 8 ports, 2 layers, straight waveguides through the blocks | 3 | 0.004 | 9e-4 |
| same, port pitch 1.0 µm (0.5 µm between wires) | 3 | 0.003 | 3.1e-2 (lateral coupling of neighbouring wires) |
| 8 ports, 2 layers, random binary blocks | 3 | 0.83 | 0.57 |
| 8 ports, 2 layers, random low-contrast blocks | 3 | 0.57 | 0.41 |
| 8 ports, 4 layers, partly optimised to the 8-point DFT (fidelity 0.76) | 6 | 0.34 | 0.099 |
| same, port planes 0.6 / 2.4 µm from the designs | 6 | 0.34 | 0.102 / 0.088 |

The error follows the light the blocks lose, as the bound predicts, and
hardly depends on the port distance: in these meshes it is the radiation of
one block reaching another, not the evanescent field at the port planes.
The feed-forward model (no reflections between blocks) and the full model
differ by less than 1e-3 in every case: these blocks reflect little.

Layer-batched runs (above) against one simulation per block, 8-port 4-layer mesh with
random blocks: S-matrices within 4 %, gradients with cosine similarity 0.994.

**3-D.** A two-layer mesh of 2-port blocks (4 ports) in 220 nm silicon with
oxide cladding (40 nm cells, 5 through the film), against the monolithic
3-D FDTD: blocks that guide most of the light (waveguides with random
bumps, mesh loss 0.17) 1.3e-2; random binary blocks (loss 0.96) 0.27.

**Gradient.** On the 8-port, 4-layer mesh at an intermediate design
(fidelity 0.76, loss 0.34), along three smooth random directions of the
density: composite adjoint against central differences of the composite
objective (h = 0.02) −0.154 / −0.158, 0.0185 / 0.0217, −0.061 / −0.056 (the
few-per-cent error of the continuous adjoint formula of the `"python"`
engine at $\beta = 8$, see [Inverse design](../guide/inverse-design.md));
composite against the monolithic adjoint gradient: cosine similarity 0.81,
objective 0.494 against 0.480. The gradient of the network itself is exact
(tested to 1e-9 against finite differences in `python/tests/test_composite.py`).

**An optimisation run to the end.** The 8-port, 4-layer mesh (six designed
4-port blocks, 4 × 5 µm each) optimised to the 8-point DFT with the
composite solver alone (`optimize_mesh.py`: Adam from a random start, 60
iterations at $\beta = 8$, then 250 more at $\beta = 8$ from the best
design; 48 block runs per iteration, 13 s per iteration on the CPU),
then simulated as one device by monolithic FDTD:

| | Composite (as optimised) | Monolithic FDTD |
|---|---|---|
| Figure of merit (fidelity × throughput) | 0.825 | 0.825 |
| Normalised fidelity | 0.927 | 0.927 |
| Throughput | 0.890 | 0.889 |
| S-matrix difference (relative Frobenius) | | 2.4e-2 |
| Same design projected at $\beta = 32$ (binarisation 0.97): fidelity / throughput | 0.866 / 0.811 | 0.864 / 0.818 (difference 3.5e-2) |

That design is grey (binarisation 0.87 at $\beta = 8$). A gentle
continuation makes it binary without losing fidelity: from it, $\beta$ was
raised by $\sqrt 2$ every 30 iterations (8, 11, 16, 23, 32, 45, 64) with
Adam (step 0.01) re-optimising at each value, 210 iterations in all. Each
step costs a few per cent of figure of merit that the next iterations
recover:

| | Composite | Monolithic FDTD |
|---|---|---|
| Binarised design ($\beta = 64$, binarisation 0.993): figure of merit | 0.831 | 0.832 |
| Normalised fidelity | 0.928 | 0.928 |
| Throughput | 0.896 | 0.897 |
| S-matrix difference (relative Frobenius) | | 2.1e-2 |

So the 8-port design is manufacturable as far as binarisation goes (the
minimum feature size is that of the 80 nm filter radius). What does not
work is a large jump early in the run: steps of 1.5 to 2 in $\beta$
taken from a design still far from converged threw it far back (figure of
merit 0.33 → 0.002 at $\beta = 2 \to 4$, 0.41 → 0.11 at $8 \to 12$),
because the phase errors of the blocks multiply through the layers.

## Measured cost

One exact gradient of every matrix entry (feed-forward blocks: 4 forward
and 4 adjoint runs per block) of a brick-wall mesh with $D = N/2$ layers,
against the $N$ forward and $N$ adjoint runs of the monolithic problem, on
the CPU (32 threads per run, shared host), in cell updates (cells × time
steps, summed over runs) and wall time:

| N | Layers | Blocks | Composite: cell updates, wall | Monolithic: cell updates, wall | Ratio |
|---|---|---|---|---|---|
| 8 | 4 | 6 | 1.24e10, 17 s | 8.5e10, 114 s | 6.8 |
| 16 | 8 | 28 | 5.7e10, 101 s | 1.38e12, 2 279 s | 24 |
| 32 | 16 | 120 | 2.46e11, 296 s | ≈ 2.2e13 (extrapolated) | ≈ 90 |

On one GB10 (two runs at a time on the GPU), the same gradients take 4.3 /
13.7 / 58.7 s with the composite solver against 13.0 / 282 s monolithic
(N = 8 / 16 / 32); 0.06 s per block run.

Running all the blocks of a layer in one simulation (one run per port and
colour, research code in `benchmarks/scaling/layer_problem.py`) was measured
too and is not part of the interface: at N = 32 it needs 256 runs instead of
960 but 8.4 times the cell updates (359 s against 59 s on the GB10), because
a strip holding a whole layer takes far longer to decay below the shutoff
than one block does. No case was found where it is faster.

Fitted exponents: composite $\propto N^{2.15}$ (the number of blocks,
$\propto N^2$, at constant cost per block), monolithic $\propto N^{4.0}$
($N^2$ cells, $\propto N$ steps, $2N$ runs). The cost per block is
independent of $N$, and the memory is that of one block instead of the
whole device.
