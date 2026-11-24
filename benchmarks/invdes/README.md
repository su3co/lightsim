# Inverse design: time per gradient against other tools

One number per tool and scene: the wall time to go from a design permittivity
to the gradient of the objective with respect to every design sample. The
results are in `benchmarks/invdes/results/invdes_*.json` (raw timings, hardware,
finite-difference checks) with the gradient arrays next to them
(`invdes_*.npz`); the tables are in `docs/benchmarks.md` and
`benchmarks/invdes/results/invdes_summary.md`, the figure is
`docs/img/invdes_speed.svg`.

## Files

| file | purpose |
|---|---|
| `problems.py` | the scenes, shared by every tool: domain, cell size, materials, absorbing layers, ports, pulse, simulated time, design, objective, finite-difference patches (NumPy only) |
| `lightsim_bench.py` | lightsim: Python engine, native engine, reverse mode; one or two GPUs |
| `fdtdx_bench.py` | FDTDX (JAX, GPU): reversible autodiff |
| `meep_bench.py` | Meep adjoint solver (CPU, MPI) |
| `meep_rate.py` | Meep time-stepping rate against the process count |
| `meep_extrapolate.py` | Meep gradients too slow to run in full: the measured gradient of a smaller scene scaled by cells × steps and the measured stepping rate |
| `ceviche_bench.py` | ceviche (2-D FDFD, CPU): as shipped, and with one factorisation per wavelength |
| `fdtdz_probe.py` | fdtd-z: whether its kernel runs on the GPU |
| `convergence.py` | how the simulated time of each scene was chosen |
| `summarize.py` | tables from the result files, gradient maps compared across tools |
| `common.py` | timing, memory, result files |
| `plot_invdes.py` | the figure |

## Scenes

Silicon (permittivity 12.25) in oxide (1.444²), wavelength 1.55 µm. Every tool
gets the same domain, cell size, absorbing-layer thickness (inside the
domain), port planes, pulse (Gaussian, spectral standard deviation 0.1 f0),
simulated time and design.

| scene | what | grid | absorbing layers | simulated time |
|---|---|---|---|---|
| `wg2d_45nm`, `_15nm`, `_5nm` | 2-D, E out of plane. Two 225 nm waveguides offset by 0.81 µm, a 2.43 µm square design between them. Gradient of the power transmitted into the output mode | 144×108, 432×324, 1296×972 | 0.54 µm (12, 36, 108 cells) | 500 fs |
| `wg3d_6M`, `_32M`, `_86M` | 3-D. A 480×220 nm wire through a 1.6×1.6×0.22 µm design region, 20 nm cells; the domain grows, the design does not. Gradient of the power transmitted into the fundamental mode | 400×200×76, 800×400×100, 1200×600×120 | 10 cells | 220, 340, 460 fs |
| `mm2d_45nm`, `_15nm`, `_5nm` | 2-D. Four inputs, four outputs, a 4.32 µm square design, three wavelengths (1.54, 1.55, 1.56 µm). Gradient of the fidelity of the 4×4 transmission matrix to the discrete Fourier transform | 196×144, 588×432, 1764×1296 | 0.54 µm | 600 fs |
| `mm3d_44nm`, `_22nm` | the same device in 220 nm silicon, cubic cells | 196×144×54, 392×288×108 | 12, 24 cells | 400 fs |

The design is a permittivity map on the nodes of the design rectangle
(55², 163², 487² samples for `wg2d`; 81² for `wg3d`; 97² ... 865² for the
matrix scenes), extruded through the film in 3-D, linearly interpolated
between nodes. It is a fixed smooth pseudo-random pattern with density between
0.3 and 0.7 (a grey design, as early in an optimisation), the same physical
pattern at every cell size. Its perimeter carries the surrounding scene.

**Simulated time.** A run that stops while the fields still ring gives a
different derivative in every tool: adjoint tools assume the steady state,
autodiff tools differentiate the truncated run. `convergence.py` scans the
simulated time with lightsim; the time used is the shortest after which the
objective is stable to 1 % and the adjoint and finite-difference derivatives
agree (`invdes_convergence_*.json`). The earlier single-GPU gradient rows of
`docs/benchmarks.md` used 100 and 150 fs on a similar scene, which ends
before the pulse reaches the output; they measure steps per second, not a
converged gradient, and are not comparable with these rows.

**What differs between tools, and cannot be made the same.**

- Where the permittivity lives. lightsim and Meep interpolate the node samples
  at each Yee component (their lattices coincide on these grids). FDTDX has
  one value per cell: it gets the mean of the four surrounding samples, and
  its 220 nm film is half a cell lower. ceviche has the permittivity on the
  E_z nodes, which is the same discrete map as lightsim and Meep in 2-D.
- Interfaces. lightsim and Meep smooth the waveguide boundaries sub-pixel,
  FDTDX staircases them. In 2-D every interface is half a cell from the
  nodes, so there is nothing to smooth.
- Pulse. FDTDX's Gaussian pulse peaks at 6 σ instead of 5 σ. Meep's adjoint
  runs: see below. ceviche has no time stepping.
- Time step. lightsim and FDTDX step at 0.99 of the Courant limit; Meep is
  set to the same time step (its default is 0.5 Δx / c, 14 % shorter in 3-D).
- Mode solvers, mode sources and monitors are each tool's own.

Because of this the objective values agree to a few per cent between tools on
the single-port scenes (see `invdes_summary.md`), not exactly. The fidelity of
the matrix scenes is the squared sum of 16 phasors that nearly cancel for a
random design, and is much more sensitive to these differences.

## What one "gradient" includes

| tool | included | not included |
|---|---|---|
| lightsim, Python and native engines | N forward + M adjoint simulations (all wavelengths in each), S-matrix, objective, product of the fields, pull-back onto the samples | mode solves; native: session set-up (in `first_call_seconds`) |
| lightsim, reverse mode | one forward simulation, one reverse pass; the objective is the energy through the output plane (a time-domain objective) | mode solve, session set-up |
| FDTDX | N forward runs with interface recording + N backward runs, inside one compiled function called again | XLA compilation (`compile_seconds`), object placement, mode solves of sources and detectors |
| Meep | N forward + N adjoint runs; the structure is re-initialised in every forward run, as `OptimizationProblem` does | construction of the `Simulation` objects |
| ceviche | assembly, factorisations and substitutions for every input and wavelength | mode profiles, calibration solves |

N inputs, M outputs. lightsim's adjoint runs are one per *output* and serve
any objective; Meep's and FDTDX's backward runs are one per *input* and
belong to the objective.

**Meep's adjoint pulse.** `meep.adjoint` launches the adjoint sources with a
pulse 100 optical periods long (517 fs), whatever the forward pulse. The
benchmark gives the mode monitors the bandwidth of the forward pulse instead
(a five-line subclass in `meep_bench.py`), so the single-wavelength adjoint
run takes the scene's simulated time; `--stock-pulse` measures Meep as
shipped. With several wavelengths Meep's adjoint source is a filtered source
whose length is the inverse of the wavelength spacing (805 fs here), longer
than the scene's simulated time; that cannot be shortened and is in the
number. The step counts of both runs are in the result files.

## Finite-difference check

For every tool and scene the gradient is compared with central differences of
the same tool's objective (permittivity step 0.4):

- on a few patches of samples at fixed physical places, each patch one
  45 nm (2-D), 40 nm (`wg3d`) or 44 nm (`mm3d`) square, so that the change of
  the objective stays above single-precision noise on fine grids. Reported:
  the worst patch, as the difference divided by the largest finite difference
  of the set;
- along a smooth direction over the whole design.

On the largest scenes the number of patches is reduced or only the direction
is checked; the result files say which. `summarize.py` also compares the
gradient maps of different tools on the same scene (cosine similarity).

## Reproducing

The scripts take `--scene`, `--reps`, `--fd` (number of patches) and write
`benchmarks/invdes/results/invdes_<tool>_<variant>_<scene>_<tag>.json`.

**lightsim** (`maturin develop --release --features cuda`). The 2-D rows were
measured with lightsim 0.1.0 (commit 54e0973), which steps one-cell-thick
grids with its 2-D update; the 3-D rows with lightsim 0.1.0 at commit 37601a2.
The result files carry the build in `build` (absent in the earlier build's
files, whose 3-D results are kept but not in the tables). `LIGHTSIM_BUILD` sets
that label; the result-file tag `v010_1gpu` / `v010_2gpu` marks the 0.1.0 runs.
On the earlier build the 2-D scenes ran through the 3-D kernels, where
`LIGHTSIM_FUSED=0` (separate E and H kernels, "split kernels") was 1.4 to 11
times faster than the default fused kernel; those results are kept in the
result files but not in the tables:

```sh
cd benchmarks/invdes
export LIGHTSIM_BUILD="lightsim 0.1.0 (commit 37601a2)"
CUDA_VISIBLE_DEVICES=0 python lightsim_bench.py --scene wg3d_6M --devices cuda:0 --engines python,native,reverse --tag v010_1gpu
python lightsim_bench.py --scene mm3d_44nm --devices cuda:0+cuda:1 --engines python,native --fd 0 --direction 0 --tag v010_2gpu
LIGHTSIM_BUILD="lightsim 0.1.0 (commit 54e0973)" CUDA_VISIBLE_DEVICES=0 python lightsim_bench.py --scene wg2d_5nm --devices cuda:0 --engines python,native,reverse --fd 2 --tag v010_1gpu
python convergence.py --scene wg3d_6M --times 160e-15,220e-15,330e-15
```

**FDTDX** 0.6.3 (`pip install fdtdx "jax[cuda13]"`); pin one GPU, JAX
otherwise takes all of them. `--recorder` chooses what the forward run
stores for the time-reversed backward run: `example` (FDTDX's own
optimisation examples: every 5th step, 8-bit floats), `fp16` (every 5th step,
16-bit floats) or `exact` (every step, 32-bit). The exact recorder needs 66 GB
at 6 M cells and does not fit in 96 GB above that; the 8-bit one fails the
finite-difference check (see `docs/benchmarks.md`). The 2-D scenes run as one
cell with periodic boundaries in z:

```sh
CUDA_VISIBLE_DEVICES=0 python fdtdx_bench.py --scene wg3d_6M --recorder exact --fd 4
CUDA_VISIBLE_DEVICES=0 python fdtdx_bench.py --scene wg3d_32M --recorder fp16 --warmup 0 --fd 0 --direction 1
CUDA_VISIBLE_DEVICES=0 python fdtdx_bench.py --scene wg2d_15nm --recorder exact
```

**Meep** 1.35.1 with MPI (`micromamba create -n meep -c conda-forge "pymeep=*=mpi_mpich_*" nlopt autograd`):

```sh
OMP_NUM_THREADS=1 mpirun -np 64 python meep_rate.py --scene wg3d_6M      # pick the process count
OMP_NUM_THREADS=1 mpirun -np 64 python meep_bench.py --scene wg3d_6M --no-forward --fd 2
OMP_NUM_THREADS=1 mpirun -np 64 python meep_bench.py --scene wg2d_15nm --stock-pulse
```

Scenes whose Meep gradient takes much longer than 15 minutes (`wg3d_32M`,
`wg3d_86M`, `mm3d_22nm`) are extrapolated, not measured. Shorter runs are not
an option: `meep.adjoint` runs every simulation until its sources have ended
(2158 steps for the forward pulse on a 20 nm grid, 805 fs for the
multi-wavelength adjoint source), whatever run time is asked for. Instead
`meep_extrapolate.py` scales the full, measured gradient of the smallest scene
of the same family (`wg3d_6M`, `mm3d_44nm`) by cells × steps and by Meep's
stepping rate measured on both scenes with `meep_rate.py` at 100 and 200
steps (agreement of the two is the check that the cost per step is
constant). As a check of the method, one `wg3d_32M` gradient with every run
stopped at the end of its source (`--timing-steps`, 2158 + 2158 steps) was
measured and is compared with the formula's prediction. The tables mark these
numbers as extrapolated and give the measured cells per second.

```sh
for s in wg3d_6M wg3d_32M wg3d_86M mm3d_44nm mm3d_22nm; do for n in 100 200; do
  OMP_NUM_THREADS=1 mpirun -np 64 python meep_rate.py --scene $s --steps $n | grep '^{' >> results/invdes_meep_rate_scaling.jsonl
done; done
OMP_NUM_THREADS=1 mpirun -np 64 python meep_bench.py --scene wg3d_32M --reps 1 --no-forward --timing-steps 200 --tag 64p_clipped
python meep_extrapolate.py
```

**ceviche** 0.1.3 (`pip install ceviche autograd pyMKL mkl`; pyMKL looks for
`libmkl_rt.so`, so link the versioned library of the `mkl` wheel to that name
and put its directory on `LD_LIBRARY_PATH`; without it ceviche falls back to
SciPy's SuperLU):

```sh
MKL_NUM_THREADS=8 python ceviche_bench.py --scene wg2d_15nm --variant stock
MKL_NUM_THREADS=32 python ceviche_bench.py --scene mm2d_5nm --variant lu
```

**fdtd-z** 1.1.3 with pjz 0.0.8 (`pip install fdtdz pjz "jax[cuda13]"`, with
the CUDA compiler on the path: the wheel is built from source):

```sh
CUDA_VISIBLE_DEVICES=0 python fdtdz_probe.py
```

**Hardware of the published numbers:** one or two NVIDIA RTX PRO 6000
Blackwell Max-Q (96 GB) for the GPU tools; one AMD Threadripper 9980X
(64 cores, 128 threads, one NUMA node) for the CPU tools. The tables give the
process or thread count that was fastest; every count tried is in the result
files.

**Tables and figure:**

```sh
python benchmarks/invdes/summarize.py
python benchmarks/invdes/plot_invdes.py
```

## Rules followed

- Single-GPU timings with one visible GPU, on an otherwise idle GPU; the
  utilisation of the visible GPUs before and after is in every result file
  (`gpu_state_before`, `gpu_state_after`).
- CPU timings with nothing else running on the machine; Meep and ceviche at
  the process or thread count that gave their best time (all tried counts are
  in the result files).
- Best of at least three evaluations, except where a result says
  otherwise (`seconds_all` holds every one).
- A gradient that fails the finite-difference check is reported with its
  error, not left out.
