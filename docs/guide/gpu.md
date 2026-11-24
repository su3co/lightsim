# GPUs and scaling

After this page you can choose where a simulation runs, estimate whether it
fits in the memory of a GPU, split it over several GPUs or several machines,
and you know which features change how fast it runs.

GPU execution needs a build with GPU support (see [Install](install.md)).

## Choosing a device

`Simulation.run(device=...)` and the `--device` option of the command-line
program take the same names:

| Name | Runs on |
|---|---|
| `"cpu"` | The multithreaded CPU engine |
| `"cuda"`, `"cuda:0"` | The first GPU |
| `"cuda:1"` | The second GPU |
| `"cuda:0,1"` | Several GPUs, the domain split between them |
| `"cuda:all"` | All GPUs of the machine |
| `"auto"` (the default) | The first GPU if the build supports GPUs and one is present, otherwise the CPU |

`lightsim.devices()` lists what is available, and `Results.stats["backend"]`
records what a run actually used.

The CPU engine uses one thread per logical CPU, or `RAYON_NUM_THREADS` if
that is set, and worker processes started with `multiprocessing` (forked or
spawned) can run simulations on the CPU whether or not the parent ran one
first; for GPU work in workers of a process that has used a GPU, start them
with the `"spawn"` method, since a forked child cannot use the GPU.

All simulation features run on both the CPU and the GPU, and the validation
suite passes on both (see [Validation](../validation.md)). The CPU engine is
the reference implementation; it is practical for small problems, tests and
small 2-D runs. The worked example runs one simulation on both and compares:

```python
import lightsim as ls
from lightsim.spec import um

f0 = ls.C0 / (1.0 * um)
sim = ls.Simulation(min=[0, 0, 0], max=[2 * um, 2 * um, 2 * um], mesh=ls.UniformMesh(25e-9), run_time=20e-15)
sim.add_source(ls.PointDipole([1 * um] * 3, "Ez", ls.GaussianPulse(f0, 0.3 * f0)))
sim.add_monitor(ls.FluxBoxMonitor("power", [0.5 * um] * 3, [1.5 * um] * 3, [f0]))

on_cpu = sim.run(device="cpu")
print(on_cpu.stats["backend"], on_cpu.stats["cells"])
```

```text
cpu (10 threads) 1124864
```

```python
on_gpu = sim.run(device="cuda:0")
print(on_gpu.stats["backend"])
print(f"{abs(on_gpu.flux('power')[0] / on_cpu.flux('power')[0] - 1):.1e}")
```

```text
cuda:0 (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, fused)
1.6e-07
```

The two engines agree to about seven digits here. Asking for a GPU index that does
not exist raises an error.

The throughput in `stats["mcells_per_s"]` is the number of cell updates per
second in millions: cells times steps divided by the time spent stepping. It
does not include the set-up time. FDTD is limited by memory bandwidth, so
throughput is highest for large grids; a small grid such as the one above
does not load a GPU fully. Measured figures for large grids are on the
[Benchmarks](../benchmarks.md) page.

## Precision

Fields, material coefficients and monitor accumulators are stored in single
precision (32-bit floats), on the CPU and on the GPU. There is no
double-precision mode. Geometry, meshing and coefficients are computed in
double precision before being stored.

Single precision limits the dynamic range of a result to roughly six to
seven digits. In practice this matters for very small signals next to large
ones, for example a reflection below $10^{-6}$ of the incident power read
from the same monitor.

## Fused and split kernels

On a GPU a time step is normally one **fused** kernel: a single pass over the
grid updates the magnetic and the electric field together. This moves less
data than two passes and is the faster path. The **split** kernels update the
two fields in separate passes; they support every feature.

The fused kernel is used automatically unless the simulation contains:

- a magnetic medium (`mu` different from 1),
- Bloch boundaries with a non-zero wavevector,
- a medium with off-diagonal tensor terms or oriented poles,
- an absorbing layer along x that is cut by a boundary between two GPUs,
- a source that injects magnetic field inside an absorbing layer, or on the
  last plane before a boundary between two GPUs.

When the fused kernel is in use, the backend name in the statistics ends in
`fused`, as in the output above. Dispersive and nonlinear media,
conductivity, PEC and PMC walls and all monitors work with the fused kernel.

In one measurement on an RTX PRO 6000, a vacuum grid of 64 million cells, the
split kernels reached 79 % of the throughput of the fused kernel.

Three environment variables report or override the automatic choices:

| Variable | Effect |
|---|---|
| `LIGHTSIM_VERBOSE=1` | Print the reason when the fused kernel is not used |
| `LIGHTSIM_FUSED=0` | Use the split kernels |
| `LIGHTSIM_DISP_DENSE=1` or `0` | Force or forbid the dense storage of dispersive poles (default: dense when at least a quarter of the cells are dispersive) |

## 2-D simulations

A 2-D simulation (see [2-D, 2.5-D and 3-D](dimensions.md)) has its own step
kernel. A grid one cell thick is a poor fit for the fused 3-D kernel, whose
blocks are 32 cells wide along z: one thread in 32 has a cell to work on and
all six components are updated. The 2-D kernel gives every thread a cell of
the plane and updates the three components of each polarization in use,
magnetic and electric field in one pass. The backend name then ends in
`2-D te`, `2-D tm` or `2-D te+tm`:

```python
flat = ls.Simulation(min=[0, 0], max=[20 * um, 20 * um], mesh=ls.UniformMesh(25e-9), run_time=20e-15,
                     polarization="tm")
flat.add_source(ls.PointDipole([10 * um, 10 * um], "Ez", ls.GaussianPulse(f0, 0.3 * f0)))
on_gpu = flat.run(device="cuda:0")
print(on_gpu.stats["backend"], on_gpu.stats["cells"], round(on_gpu.stats["mcells_per_s"] / 1e3, 1))
```

```text
cuda:0 (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 2-D tm) 678976 56.1
```

Points to know:

- **Small grids are limited by the number of kernel launches per step**, not
  by the cells: a launch costs a few microseconds whatever it does. The 2-D
  path therefore accumulates all the frequency-domain monitors that share
  their frequencies in one launch, and applies the source terms in one or
  two. Below about TBD cells the CPU engine is faster than a GPU; the
  [Benchmarks](../benchmarks.md) page has the figures.
- **Memory** is that of a one-cell 3-D grid with the fused kernel (64 bytes
  per cell and up): all six components are allocated even when three are
  updated.
- **Several GPUs** split a 2-D grid along x, as in 3-D, when x is one of the
  in-plane axes. It pays off only for grids of tens of millions of cells.
- The features that make a 2-D simulation run on the general kernels are
  listed in [2-D, 2.5-D and 3-D](dimensions.md); the backend name then lacks
  `2-D`. `LIGHTSIM_2D=0` forces the general kernels, and
  `LIGHTSIM_2D_BATCH=0` launches every monitor box and source list
  separately.

## Memory

Memory use is proportional to the number of cells, absorbing layers
included. `sim.grid()` gives the cell counts before running anything.

| Configuration | Bytes per cell, from the arrays | Measured |
|---|---|---|
| Fused kernel | 64: two copies of the six fields, three coefficient arrays, a material index | 71 |
| Split kernels | 36: six fields and three coefficient arrays | 39 |
| Bloch boundaries (split kernels) | 60: a real and an imaginary set of fields | 62 |
| Conductive media anywhere in the domain | 12 more | |
| Magnetic media anywhere in the domain | 12 more | |
| Dispersive media, dense storage, $P$ poles | $24P + 3$ more | 98 in total for one pole everywhere |
| Dispersive media, sparse storage | $24P + 60$ per dispersive cell | |
| Off-diagonal media | | 159 in total for a rotated crystal everywhere |

The measured column is the peak GPU memory of `lightsim bench` on a grid of
400 by 400 by 400 cells (64 million) on an RTX PRO 6000, divided by the
number of cells, after subtracting 0.6 GB that does not depend on the grid
size (estimated from three grid sizes). It includes the absorbing layers, 12
cells thick, which is why it is a little above the count of the arrays.

On top of the per-cell cost:

- A frequency-domain monitor stores 8 bytes per recorded sample, per
  component and per frequency. A flux or mode monitor records six field
  planes; a volume `DFTMonitor` with many frequencies can exceed the fields.
- Time-domain monitors keep every sample until the end of the run: 4 bytes
  per recorded value. Their buffers grow with the steps actually taken, in
  chunks that start at 1 MiB and double up to 256 MiB, so a recording holds
  at most about twice what it has recorded (on a small scene 100 to 140 MiB
  more than an exact fit), and a run that stops early never holds its whole
  run-time cap.
- The absorbing layers add 16 bytes per cell inside them, per axis, with the
  split kernels and 40 with the fused kernel.

As an example, a vacuum grid of 640 by 640 by 512 cells (210 million) used
15.1 GB. By the same count a GPU with 96 GB of memory holds about 1.3 billion
cells with the fused kernel, before monitors.

A process that has run on a GPU keeps that GPU's context, about 550 MiB of
its memory, until it exits, which saves about 120 ms on every later run.
`lightsim.release_gpus()` lets the contexts go earlier (for a long-lived
process that has finished its GPU work); the next run creates them again
and gives the same results.

## Several GPUs in one machine

With `device="cuda:0,1"` the grid is cut into slabs along **x**, one slab per
GPU, of equal thickness. At every step the GPUs exchange the fields on the
cut planes, directly between the cards when the machine supports
peer-to-peer copies and through host memory otherwise. `lightsim info` lists
the GPUs and shows which pairs can copy directly:

```text
CUDA devices: 2
  cuda:0  NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  cuda:1  NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  P2P 0->1: true
  P2P 1->0: true
```

The results are assembled as if the run had been done on one device.
Periodic and Bloch boundaries along x, dispersive, nonlinear and tensor media
all work across the cut.

Points to know:

- Orient the simulation so that x is its longest axis. The cost of the
  exchange is proportional to the area of the cut, and slabs along the
  longest axis have the smallest area per cell.
- The split is into equal parts. GPUs of different speed are held back by
  the slowest one. The command `lightsim rank` accepts weights (see below).
- A slab must be thicker than the absorbing layer along x for the fused
  kernel to be used. This only matters for very short domains.
- Measured speed-ups on two GPUs are on the [Benchmarks](../benchmarks.md)
  page. Splitting pays off for grids that load one GPU fully; a small grid
  runs no faster on two.

## Several machines

A run can also be split over several machines, each with one or more GPUs.
This is done with the command-line program, not from Python: every machine
runs one process (a "rank") that owns a slab along x, and neighbouring ranks
exchange the cut planes over TCP. The commands and their options are
described in [Command line](cli.md); the script `scripts/multi_node.sh`
starts one rank per machine:

```bash
HOSTS=node0,node1,node2 scripts/multi_node.sh sim.json --out results.json
```

| Variable | Meaning |
|---|---|
| `HOSTS` | Addresses of the machines, rank 0 first. Each rank is started over ssh |
| `CHAIN` | Optional addresses of dedicated links between neighbours: the address of rank 1 on its link to rank 0, of rank 2 on its link to rank 1, and so on |
| `LIGHTSIM_BIN` | Path of the program on the machines (default `lightsim`) |

Further arguments are passed to every rank, for example `--gpus 0,1` to use
two GPUs per machine or `--weights 2,1,1` for machines of unequal speed.

Points to know:

- At every step each rank sends two field components on the cut plane to
  each neighbour, 8 bytes per cell of the plane (twice that in a Bloch run).
  For large grids the link between neighbours should be fast; a dedicated
  direct link between neighbouring machines is the usual set-up. The
  [Benchmarks](../benchmarks.md) page reports what was measured over 10 and
  200 gigabit links.
- The JSON file and the program must be at the same path on every machine.
- Rank 0 gathers the monitor data of all ranks and writes the results file.
- The early-stop test (`shutoff`) uses the energy summed over all ranks.

## Practical advice

- Check the size first: `sim.grid()` and `sim.time_info()` cost nothing.
  The run time is the number of cells times the number of steps divided by
  the throughput.
- Keep frequency-domain monitors to planes unless you need a volume.
- Avoid a single very small cell: it shortens the time step of the whole run
  (see [Simulation setup](simulation.md)).
- Several independent simulations, such as the ports of an S-parameter
  calculation or a parameter sweep, are better run one per GPU than split
  across GPUs.
