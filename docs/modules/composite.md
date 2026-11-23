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
| `LayerProblem(sim, members, freqs, devices)` | Several non-interacting blocks in one simulation: each run excites the same port of every block of one colour (see below) |
| `LayerMember(layer, index)` | A member of a `LayerProblem`, as the `problem` of a `Block` |

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

`LayerProblem` reduces the number of simulations of a layer of $B$ blocks
from $B (n_{in} + n_{out})$ to $c\,(n_{in} + n_{out})$ for $c$ colours: it
excites input $i$ of every block of one colour in the same run and attributes
the fields in each block's design grid and the amplitudes at its ports to
that block alone. This is the same approximation as above, made inside one
simulation; with two colours no two neighbours are driven together.

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

<!-- filled in from benchmarks/scaling results -->
