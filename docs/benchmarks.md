# Benchmarks

![Throughput and speed-up against problem size](img/speed.svg)

Vacuum box with 12-cell CPML on all faces and a dipole, f32, best of runs
(`benchmarks/sweep.py`, which calls `lightsim bench` and
`benchmarks/fdtdx_box.py`). Each run takes 200 steps, or more on small grids so
that it does at least 2×10¹⁰ cell updates. Gcells/s = 10⁹ cell updates per
second. In the figure lightsim is solid and FDTDX dashed, on the same GPUs; the
star is a published Lumerical figure on a different GPU, not a measurement
of ours.

**NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120)**

| grid | cells | lightsim 1 GPU | FDTDX 1 GPU | speed-up | lightsim 2 GPUs | FDTDX 2 GPUs | speed-up |
|---|---|---|---|---|---|---|---|
| 128³ | 2.1 M | 22.3 | 21.9 | 1.01× | 23.8 | 4.33 | 5.49× |
| 192³ | 7.1 M | 19.9 | 13.7 | 1.45× | 26.4 | 7.64 | 3.45× |
| 256³ | 16.8 M | 23.7 | 10.9 | 2.17× | 36.6 | 9.81 | 3.73× |
| 320³ | 32.8 M | 24.3 | 10.9 | 2.23× | 40.6 | 12.1 | 3.36× |
| 400³ | 64 M | 24.9 | 11.1 | 2.24× | 44.7 | 14.4 | 3.11× |
| 512³ | 134 M | 26.4 | 11.3 | 2.35× | 48.6 | 17.0 | 2.86× |
| 640×640×512 | 210 M | 26.6 | 11.3 | 2.35× | 49.9 | 19.1 | 2.61× |
| 1024×640×640 | 419 M | 26.5 | 11.5 | 2.30× | 51.0 | 23.9 | 2.13× |
| 1600×800×800 | 1.02 B | | | | 50.5 | 27.8 | 1.82× |

On one GPU FDTDX is as fast as lightsim on the smallest box and falls to about
11 Gcells/s on larger ones, where lightsim holds 24–27. With two GPUs visible
FDTDX is slow on small grids and keeps gaining with size, so lightsim's lead
narrows from 5.5× to 1.8×.

**NVIDIA GB10 (sm_121)**

| grid | cells | lightsim | FDTDX | speed-up |
|---|---|---|---|---|
| 128³ | 2.1 M | 2.94 | 1.56 | 1.88× |
| 192³ | 7.1 M | 3.28 | 1.58 | 2.08× |
| 256³ | 16.8 M | 3.51 | 1.60 | 2.19× |
| 320³ | 32.8 M | 3.43 | 1.66 | 2.07× |
| 400³ | 64 M | 3.58 | 1.69 | 2.12× |
| 512³ | 134 M | 3.73 | 1.71 | 2.19× |
| 640×640×512 | 210 M | 3.78 | 1.71 | 2.21× |

**Several GB10 nodes** (one GB10 per node, direct 200 GbE QSFP links between neighbours). lightsim
exchanges halos over TCP; FDTDX runs one JAX process per node
(`benchmarks/fdtdx_multinode.py`) with NCCL over RDMA (RoCE on both halves of
the ConnectX-7 link) — its fastest transport — or over TCP sockets:

| nodes | grid | cells | lightsim | FDTDX (NCCL RDMA) | FDTDX (NCCL sockets) | speed-up |
|---|---|---|---|---|---|---|
| 2 | 1200×640×512 | 393 M | 5.90 | 4.57 | 2.52 | 1.29× (2.34× vs sockets) |
| 2 | 1800×800×800 | 1.15 B | 6.25 | 4.36 | | 1.43× |
| 3 | 1200×640×512 | 393 M | 8.41 (200 GbE) / 8.15 (10 GbE) | — | 0.79 (10 GbE) | 10.3× (same 10 GbE network) |
| 3 | 1800×800×800 | 1.15 B | 9.13 (200 GbE) / 8.82 (10 GbE) | — | 1.13 (10 GbE) | 7.8× (same 10 GbE network) |

lightsim scales at 94–97 % parallel efficiency on two nodes and 88–96 % on three (2.65× and 2.87× one
node). FDTDX scales superlinearly from one to two nodes (1.73 → 4.57 Gcells/s
on the same 393 M-cell grid; the final fields agree to f32 rounding, Σ E² =
2.482678 in both runs), which narrows the gap to 1.3–1.4×. On three nodes FDTDX
runs over the 10 GbE Ethernet network with NCCL sockets: the nodes are cabled as a
triangle with a separate subnet per 200 GbE link, and NCCL needs every rank
reachable at one address, so it cannot use those links. The 3-node FDTDX run
reproduces its single-node final field (Σ E² = 2.482678) but is slower than one
node, because every step's halo crosses 10 GbE. lightsim reaches 8.2–8.8
Gcells/s on that same network, and 8.4–9.1 over the 200 GbE links (its
`--chain-addrs` gives each link its own address).

**Feature scenes** (`lightsim bench --scene ...`, `benchmarks/fdtdx_scenes.py`;
same grid, materials, boundaries and source in both codes, 200 steps, Gcells/s):

| scene | GPU(s) | grid | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|
| whole domain Lorentz-dispersive | 1× RTX PRO 6000 | 400³ | 8.92 | 5.23 | 1.71× |
| | 1× RTX PRO 6000 | 640×640×512 | 9.50 | 5.29 | 1.80× |
| | 2× RTX PRO 6000 | 640×640×512 | 18.46 | 9.72 | 1.90× |
| | GB10 | 400³ | 1.40 | 0.80 | 1.75× |
| Bloch k ≠ 0 along x and y (complex fields) | 1× RTX PRO 6000 | 400³ | 7.82 | 3.92 | 2.00× |
| | 1× RTX PRO 6000 | 640×640×512 | 8.34 | 3.94 | 2.12× |
| | 2× RTX PRO 6000 | 640×640×512 | 16.09 | 6.56 | 2.45× |
| | GB10 | 400³ | 1.20 | 0.55 | 2.18× |
| rotated crystal (off-diagonal ε everywhere) | 1× RTX PRO 6000 | 400³ | 5.19 | 0.85 | 6.1× |
| | 1× RTX PRO 6000 | 640×640×512 | 5.40 | 0.73 | 7.4× |
| | 2× RTX PRO 6000 | 640×640×512 | 10.48 | 0.69 | 15.2× |
| | GB10 | 400³ | 0.81 | 0.14 | 5.7× |

Dispersive media use a dense per-cell pole state when most of a slab is
dispersive (one pass after the fused step, no index lists). Small dispersive
objects keep sparse lists. All FDTDX runs ended with finite fields.

The fused kernel computes H and E of a step in one pass (tile-marching blocks,
shared-memory H halo, double-buffered fields and CPML state, u32 material index)
and moves ~52 bytes per cell-update instead of ~84; on the RTX PRO 6000 it
reaches ~75 % of the measured copy bandwidth (1.49 TB/s).

**Inverse-design gradients** (`benchmarks/lightsim_gradient.py`,
`benchmarks/fdtdx_gradient.py`; one RTX PRO 6000). The scene is a straight
silicon waveguide through a 1.6 × 1.6 × 0.4 µm design region on a 20 nm grid,
with identical cell and step counts in both codes. FDTDX differentiates
`run_fdtd` with `jax.value_and_grad`: reversible time stepping with the
recorder settings from its own optimization example. lightsim uses its adjoint
method: a forward run plus an adjoint run.

| cells × steps | lightsim forward | FDTDX forward | lightsim gradient | FDTDX gradient | gradient speed-up |
|---|---|---|---|---|---|
| 6.0 M × 2 623 | 1.40 s | 3.01 s | 2.93 s | 10.74 s | 3.7× |
| 32 M × 3 934 | 9.60 s | 34.34 s | 19.47 s | 102.57 s | 5.3× |

lightsim's time reversal (`lightsim_core::reverse`, CPU and GPU) steps a
16.8 M-cell box backwards at 13.2 Gcells/s on one RTX PRO 6000 (forward
18.2), reconstructing the fields to 8e-7.

**Gradients inside the engine** (`benchmarks/lightsim_autodiff.py`, one RTX
PRO 6000; module notes in [docs/modules/autodiff.md](modules/autodiff.md)).
Time for one gradient with respect to the samples of an index-grid design,
three ways: `lightsim.invdes` driving every run from Python
(`engine="python"`), the same S-matrix gradient in the persistent Rust session
(`engine="native"`: scene rasterised once, design sub-block uploaded, adjoint
product on the GPU), and reverse-mode differentiation through the time loop
of a time-domain objective (energy through the output plane). The two
S-matrix engines return the same numbers (S to 3e-16, gradient to 1.2e-7).

| scene | Python path | native session | speed-up | reverse mode (forward + reverse pass) |
|---|---|---|---|---|
| 2.5-D, 23 k cells, 2 inputs × 2 outputs × 3 frequencies (4 runs × 2 000 steps) | 1.28 s | 0.29 s | 4.5× | |
| gradient scene, 6.0 M × 2 623, 137 781 design samples | 2.97 s | 2.12 s | 1.40× | 4.29 s (1.15 + 3.04) |
| gradient scene, 32 M × 3 934, 137 781 design samples | 15.75 s | 12.40 s | 1.27× | 29.56 s (6.54 + 22.86) |

Where the time goes. In the small problem the step rate is bounded by kernel
launches, not by cells: the plain solver launches one DFT kernel per monitor
box per step (27 here), the session one for all of them (78 → 35 µs per
step), and the Python path also pays a set-up per run (spec, rasterisation,
device allocation, PTX load): 0.66 s of its 1.28 s. In the large scenes the
simulations themselves dominate (2 × 1.05 s of the 2.12 s at 6 M) and the
session saves the per-run set-up and the host-side product. Session set-up,
once: 0.22 s, 0.33 s and 2.2 s for the three scenes.

Reverse mode costs a forward run plus a reverse pass of 2.6 (6 M) to 3.5
(32 M) times that. Per step the pass reverses the forward fields and
advances the adjoint, both with the split kernels (the adjoint needs the
transposed CPML, which the fused kernel does not contain), and accumulates
the coefficient sums. It records the one-cell PML shell on the device:
5.2 GB at 6 M, 27.7 GB at 32 M. Its gradient agrees with central finite
differences of the objective to 6e-5 (6 M) and 3e-4 (32 M). With the adjoint
on the fused kernel and the forward CPML in place of its transpose the pass
took 2.41 s and 16.0 s, and the 6 M gradient was off by 1.4 %; that version
was dropped. On the same two scenes FDTDX's reversible `jax.value_and_grad`
takes 10.74 s and 102.57 s (`benchmarks/results/fdtdx_gradient_*.json`), and
`lightsim.adjoint` (uniform design block, gradient per Yee cell, two
Python-driven runs) 2.5 ... 3.3 s (two batches) and 14.2 s on this build.

**FDTDX's own speed benchmark** (SOI directional coupler from FDTDX's
`performance/coupler.gds`, non-uniform grid, mode source and three mode
detectors; same grid and number of steps in both codes,
`benchmarks/lightsim_coupler.py`):

| GPU | cells/λ | grid | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|
| RTX PRO 6000 | 10 | 948×145×65 | 9.2 s | 24.4 s | 2.64× |
| RTX PRO 6000 | 20 | 1897×291×128 | 91.8 s | 425.8 s | 4.64× |
| RTX PRO 6000 | 30 (4000 steps) | 2845×436×193 | 52.6 s | 246.9 s | 4.70× |
| GB10 | 10 | 948×145×65 | 46.0 s | 200.6 s | 4.36× |
| GB10 | 20 (4000 steps) | 1897×291×128 | 100.4 s | 482.8 s | 4.81× |

lightsim's coupler transmissions are converged to ~2 % between 10 and 20
cells/λ (T_thru/T_cross 0.571/0.425 → 0.555/0.442), whereas FDTDX's swing from
0.25/0.72 to 0.92/0.07 at these meshes. Full tables: `benchmarks/results/summary.md` in the repository.

## 2-D simulations

Square 2-D grids of 10⁴ to 10⁸ cells (12-cell CPML included), 25 nm cells,
one electric dipole per field set in use, f32, best of three runs
(`benchmarks/bench_2d.py`, which calls `lightsim bench --n N,N` and
`benchmarks/fdtdx_2d.py`). "2-D" is the specialised update of a
`dimensions=2` simulation; "one-cell 3-D route" is the same grid run through
the general 3-D update (`--kernels 3d`), which is how a 2-D problem ran
before the 2-D mode: a 3-D grid one cell thick with periodic boundaries.
"TE" updates one field set (three components), "both" the two sets (six).
Gcells/s = 10⁹ cell updates per second, whatever the number of components.

**GPU: one NVIDIA RTX PRO 6000 Blackwell Max-Q** (FDTDX 0.6.3 has no 2-D
solver; its column is the same grid as a one-cell 3-D grid with periodic
boundaries, all six components updated, best of two runs)

| cells | 2-D, TE | 2-D, both | one-cell 3-D route, TE | one-cell 3-D route, both | FDTDX (one-cell 3-D grid) |
|---|---|---|---|---|---|
| 10 000 | 1.82 | 1.30 | 0.38 | 0.36 | 0.81 |
| 99 856 | 15.9 | 11.5 | 1.57 | 1.54 | 6.95 |
| 1 000 000 | 50.1 | 31.5 | 2.27 | 2.27 | 32.5 |
| 9 998 244 | 41.7 | 25.2 | 2.19 | 2.19 | 12.3 |
| 100 000 000 | 44.4 | 27.7 | 2.05 | 2.05 | 11.9 |

On the GPU the 2-D update is 5–22× faster than the one-cell 3-D route with
one polarization and 4–14× with both: the fused 3-D kernel's blocks are 32
cells wide along the thin axis, so most of its threads idle. Against FDTDX on
the same one-cell grid lightsim is 2.2–3.7× faster with TE and 1.6–2.3× with
both polarizations, except at 10⁶ cells, where FDTDX is at its best: 1.5×
slower than lightsim with TE and level with it with both polarizations.

**CPU: AMD Ryzen Threadripper 9980X, 64 cores / 128 threads**

| cells | 2-D, TE | 2-D, both | one-cell 3-D route, TE | one-cell 3-D route, both |
|---|---|---|---|---|
| 10 000 | 0.29 | 0.22 | 0.01 | 0.01 |
| 99 856 | 0.57 | 0.28 | 0.11 | 0.11 |
| 1 000 000 | 1.04 | 0.49 | 0.67 | 0.66 |
| 9 998 244 | 1.43 | 0.78 | 1.04 | 1.03 |
| 100 000 000 | 2.94 | 1.54 | 1.63 | 1.63 |

On this CPU one polarization runs 1.4–2× faster than the one-cell route on
grids of 10⁶ cells and more, and 5–25× faster on small grids, where the
3-D route is dominated by per-step overhead. With both polarizations the
2-D update is faster below 10⁶ cells but 25 % *slower* than the 3-D route
at 10⁶–10⁷ cells and level with it at 10⁸: on 128 threads the two sets'
row loops do not split as well as the 3-D update. (On a 10-core laptop
CPU the 2-D update with both sets is 2.3× faster than the 3-D route at
10⁶ cells.) The large-grid CPU figures use 40 steps and vary by about
10 % between runs.

**Applications, before and after the 2-D path.** "Before" is the tree
without the 2-D mode (a one-cell 3-D grid on the general update), "after"
the current one; "after, 3-D kernels" is the current tree with
`LIGHTSIM_2D=0`. Time per run, best of three after a warm-up
(`benchmarks/bench_2d_apps.py`): the effective-index S-parameters of the
gdsfactory 1x2 MMI (132 628 cells, 7 000 steps, 11 wavelengths, one input
port) and one planar inverse-design gradient (solve + vector-Jacobian
product, two inputs, two outputs, three frequencies, about 2×10⁴ cells).

| | before | after | after, 3-D kernels | speed-up |
|---|---|---|---|---|
| 2.5-D MMI S-parameters, one RTX PRO 6000 | 0.668 s | 0.202 s | 0.672 s | 3.3× |
| 2.5-D MMI S-parameters, CPU (128 threads) | 8.86 s | 2.70 s | 8.18 s | 3.3× |
| planar gradient, native engine, one RTX PRO 6000 | 0.258 s | 0.069 s | 0.259 s | 3.8× |
| planar gradient, Python engine, one RTX PRO 6000 | 0.332 s | 0.125 s | 0.336 s | 2.7× |
| planar gradient, native engine, CPU (128 threads) | 8.01 s | 0.420 s | 7.34 s | 19× |
| planar gradient, Python engine, CPU (128 threads) | 8.11 s | 0.460 s | 7.01 s | 18× |

The S-parameters and the gradient's figure of merit agree between the
three columns to 4×10⁻⁶. On the CPU the 3-D update scales poorly to 128
threads on a grid this small, hence the large gain of the planar gradient.
Result files: `benchmarks/results/twod_*.json` and
`benchmarks/results/twod_apps/`.

## Lumerical (published figures)

Lumerical FDTD needs a commercial license, and none is installed on this
hardware, so there is no head-to-head run. As an indirect comparison, here are
Ansys's and its partners' published GPU throughput figures (cells per second;
Ansys writes Mn/s or Mnodes/s), normalized by each GPU's memory bandwidth.
FDTD is bandwidth-bound, so throughput per GB/s compares codes across GPUs:

| code | GPU (bandwidth) | scene | throughput | per GB/s of bandwidth |
|---|---|---|---|---|
| lightsim (measured) | RTX PRO 6000 Max-Q (1.79 TB/s) | vacuum box, 210–419 M cells | 21.1–21.2 G/s | 11.8 M/s |
| lightsim (measured) | RTX PRO 6000 Max-Q (1.79 TB/s) | FDTDX's SOI coupler, 71–239 M cells | 17.9–18.2 G/s | 10.0–10.2 M/s |
| lightsim (measured) | GB10 (273 GB/s) | vacuum box | 3.0–3.1 G/s | 11.0–11.5 M/s |
| Lumerical (Ozen Engineering) | A100 80 GB (1.94 TB/s) | metalens, 0.85 B cells | 13.2 G/s | 6.8 M/s |
| Lumerical (Ansys) | V100 16 GB (0.90 TB/s) | grating coupler | ~6.5 G/s | ~7.2 M/s |
| Lumerical (Ansys) | A40 (0.70 TB/s) / RTX PRO 5000 Laptop (0.90 TB/s) | CMOS image sensor | 6.1–8.7 G/s | 6.8–12.5 M/s* |

\* Ansys gives the range for the two GPUs without saying which number belongs
to which card.

On the data-center cards with unambiguous figures (A100, V100), lightsim
updates **1.4–1.7× more cells per GB/s of bandwidth** than the published
Lumerical numbers. At that efficiency an A100 80 GB would run lightsim at
roughly 19–23 Gcells/s, against the 13.2 Gcells/s reported for Lumerical.

`benchmarks/lumerical_bench.py` builds the same box and Lorentz scenes through
Lumerical's Python API (`lumapi`), with matching cells, time step and step
count, for a measured head-to-head on a machine with a Lumerical license. It
has not been run: no license is available on this hardware.

This is an estimate from published numbers, not a measurement. Scenes,
precision and software versions differ, and a direct run would need a
Lumerical license. Sources: Ansys "FDTD-GPU Performance Benchmarks" and
"GPU Acceleration for Grating Coupler Optimizations" (as summarized by
LightBridge, lightbridge.co.jp/en/lumerical-fdtd-gpu-performance-and-selection),
and Ozen Engineering's Lumerical cloud-burst benchmark.
