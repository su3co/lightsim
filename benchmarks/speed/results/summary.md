# Benchmark results (lightsim 0.1.0, commit 37601a2; measured 2026-11-22)

Hardware: a workstation with 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 96 GB,
CUDA 13.4, driver 615.71) and nodes with one NVIDIA GB10 each (sm_121, 121 GB unified,
CUDA 13.0, driver 580.178).
lightsim 0.1.0 at commit 37601a2 (fresh clone, `cargo build --release -p lightsim-cli --features cuda`,
`maturin develop --release --features cuda`), fused single-kernel step, f32.
FDTDX 0.6.3 with JAX 0.12.1 (CUDA 13); FDTDX rows are re-measured where marked (R), otherwise carried over
from the 2026-11-21 measurements.
Gcells/s = 10⁹ cell updates per second (MCUPS / 1000). Every timing: best of at least three runs, each in a
new process, on GPUs that were idle just before the run; every lightsim run ended with finite fields
(the runner fails a run whose field energy is not finite). Previous-build values in brackets.

## Vacuum box, 12-cell CPML, dipole (`lightsim bench`, `fdtdx_box.py`), `sweep_*.json`

200 steps, or more so that a run does at least 2×10¹⁰ cell updates.

| GPU(s) | grid | lightsim Gcells/s | FDTDX Gcells/s | speed-up |
|---|---|---|---|---|
| 1× RTX PRO 6000 | 128³ | 23.43 [22.26] | 21.74 (R) | 1.08× |
| 1× RTX PRO 6000 | 192³ | 19.93 [19.90] | 13.70 | 1.46× |
| 1× RTX PRO 6000 | 256³ | 23.65 [23.67] | 10.90 | 2.17× |
| 1× RTX PRO 6000 | 320³ | 24.33 [24.34] | 10.90 | 2.23× |
| 1× RTX PRO 6000 | 400³ | 24.89 [24.85] | 11.09 | 2.25× |
| 1× RTX PRO 6000 | 512³ | 26.39 [26.44] | 11.25 (R) | 2.35× |
| 1× RTX PRO 6000 | 640×640×512 | 26.57 [26.63] | 11.32 | 2.35× |
| 1× RTX PRO 6000 | 1024×640×640 | 26.40 [26.45] | 11.49 (R) | 2.30× |
| 2× RTX PRO 6000 | 128³ | 23.17 [23.80], spread 9.6 % | 4.33 | 5.35× |
| 2× RTX PRO 6000 | 192³ | 26.63 [26.37], spread 3.1 % | 7.64 | 3.49× |
| 2× RTX PRO 6000 | 256³ | 36.83 [36.64] | 9.81 | 3.75× |
| 2× RTX PRO 6000 | 320³ | 40.69 [40.62] | 12.10 | 3.36× |
| 2× RTX PRO 6000 | 400³ | 44.16 [44.69] | 14.39 | 3.07× |
| 2× RTX PRO 6000 | 512³ | 48.65 [48.64] | 17.01 (R) | 2.86× |
| 2× RTX PRO 6000 | 640×640×512 | 49.87 [49.90] | 19.10 | 2.61× |
| 2× RTX PRO 6000 | 1024×640×640 | 50.77 [50.98] | 23.90 | 2.12× |
| 2× RTX PRO 6000 | 1600×800×800 | 50.31 [50.53] | 27.75 | 1.81× |
| GB10 | 128³ | 2.95 [2.94] | 1.56 | 1.89× |
| GB10 | 192³ | 3.27 [3.28] | 1.58 | 2.07× |
| GB10 | 256³ | 3.50 [3.51] | 1.60 | 2.19× |
| GB10 | 320³ | 3.42 [3.43] | 1.66 | 2.06× |
| GB10 | 400³ | 3.56 [3.58] | 1.69 | 2.11× |
| GB10 | 512³ | 3.73 [3.73] | 1.70 (R) | 2.20× |
| GB10 | 640×640×512 | 3.77 [3.78] | 1.71 | 2.21× |

Single-GB10 references for the multi-node rows (`box_gb10_single_extra.json`):
400×640×512 3.73, 600×640×512 3.77, 1200×640×512 3.82, 600×800×800 3.77, 900×800×800 3.80, 1800×800×800 3.77 Gcells/s.

## Several GB10 nodes (`lightsim rank`, `fdtdx_multinode.py`), `lightsim_box_*xgb10_*.json`

| nodes, network | grid | lightsim | FDTDX | speed-up | lightsim parallel efficiency |
|---|---|---|---|---|---|
| 2× GB10, 200 GbE links | 1200×640×512 (393 M) | 6.95 [5.90] | 4.57 (NCCL RDMA), 2.52 (NCCL sockets) | 1.52× | 92 % (1.82× one GB10 on the same grid) |
| 2× GB10, 200 GbE links | 1800×800×800 (1.15 B) | 7.31 [6.25] | 4.36 (NCCL RDMA) | 1.68× | 96 % (1.94× one GB10 on the same grid) |
| 3× GB10, 200 GbE links | 1200×640×512 (393 M) | 9.80 [8.41] | — |  | 88 % (2.57× one GB10 on the same grid) |
| 3× GB10, 10 GbE | 1200×640×512 (393 M) | 9.54 [8.15] | 0.79 (NCCL sockets) | 12.08× | 85 % (2.50× one GB10 on the same grid) |
| 3× GB10, 200 GbE links | 1800×800×800 (1.15 B) | 10.64 [9.13] | — |  | 94 % (2.82× one GB10 on the same grid) |
| 3× GB10, 10 GbE | 1800×800×800 (1.15 B) | 10.20 [8.82] | 1.13 (NCCL sockets) | 9.03× | 90 % (2.70× one GB10 on the same grid) |

lightsim ranks exchange halos over TCP, over the direct QSFP links with `--chain-addrs` or over the
10 GbE network. Parallel efficiency: rate / (nodes × one GB10 on one node's slab). FDTDX uses
`jax.distributed` with one process per node, NCCL over RoCE on both ConnectX-7 halves of the link
or over TCP sockets. FDTDX re-check of the 2-node 393 M-cell rows (`fdtdx_box_2xgb10_1200_*_recheck.json`):
RDMA best of two launches 4.55 Gcells/s (first launch 4.15), sockets 2.48; Σ E² 2.482678 as before.
The values in the table are the earlier ones (within 5 %); the 1.15 B-cell and 3-node FDTDX rows are
carried over. On 3 GB10 nodes FDTDX runs over the 10 GbE network, because NCCL cannot use the per-link
subnets of the 200 GbE point-to-point links; its 3-node final field matches the single-node run.

## Feature scenes (`lightsim bench --scene`, `fdtdx_scenes.py`), 200 steps, Gcells/s

| scene | GPU(s) | grid | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|
| Lorentz everywhere | 1× RTX PRO 6000 | 400³ | 13.79 [8.92] | 5.23 (R) | 2.64× |
| Lorentz everywhere | 1× RTX PRO 6000 | 640×640×512 | 15.49 [9.50] | 5.29 | 2.93× |
| Lorentz everywhere | 2× RTX PRO 6000 | 640×640×512 | 29.88 [18.46] | 9.72 | 3.07× |
| Lorentz everywhere | GB10 | 400³ | 2.09 [1.40] | 0.801 | 2.61× |
| Bloch k≠0 (x, y) | 1× RTX PRO 6000 | 400³ | 9.36 [7.82] | 3.92 (R) | 2.39× |
| Bloch k≠0 (x, y) | 1× RTX PRO 6000 | 640×640×512 | 11.21 [8.34] | 3.94 | 2.85× |
| Bloch k≠0 (x, y) | 2× RTX PRO 6000 | 640×640×512 | 21.67 [16.09] | 6.55 | 3.31× |
| Bloch k≠0 (x, y) | GB10 | 400³ | 1.71 [1.20] | 0.548 | 3.12× |
| rotated crystal (off-diagonal ε) | 1× RTX PRO 6000 | 400³ | 6.14 [5.19] | 0.852 (R) | 7.20× |
| rotated crystal (off-diagonal ε) | 1× RTX PRO 6000 | 640×640×512 | 6.29 [5.40] | 0.731 | 8.61× |
| rotated crystal (off-diagonal ε) | 2× RTX PRO 6000 | 640×640×512 | 12.19 [10.48] | 0.688 | 17.71× |
| rotated crystal (off-diagonal ε) | GB10 | 400³ | 0.95 [0.81] | 0.142 | 6.70× |

FDTDX was pinned with `CUDA_VISIBLE_DEVICES`: without it, JAX shards over every visible GPU.

## Inverse-design gradient (`lightsim_gradient.py`, `fdtdx_gradient.py`), 1x RTX PRO 6000

| cells x steps | lightsim forward | FDTDX forward | lightsim gradient (adjoint) | FDTDX gradient (jax.value_and_grad, reversible) | speed-up |
|---|---|---|---|---|---|
| 6.0 M x 2623 | 1.18 s [1.40] | 3.01 s | 2.41 s [2.93] | 10.74 s | 4.5x |
| 32 M x 3934 | 7.05 s [9.60] | 34.34 s | 14.16 s [19.47] | 102.57 s | 7.2x |

lightsim: best of three in one process (`--reps 3`). FDTDX carried over (not re-measured). The objective
value of this short-time scene differs from the earlier build (fom 2.95e-6 against 4.18e-6 at 6 M,
6.97e-11 against 1.68e-11 at 32 M); the scene stops before the pulse reaches the output, so the value is
small and sensitive.

GPU time reversal (`time_reversal_gpu.rs`, ignored test `gpu_reverse_throughput`, `time_reversal_gpu_rtx6000.json`):
256^3 periodic box, 200 steps, forward 20.75 Gcells/s [18.24], reverse 12.60 Gcells/s [13.19]
(runs: 12.6, 12.27, 12.33), reconstruction error 8.0e-7.

## Gradients inside the engine (`lightsim_autodiff.py`), 1x RTX PRO 6000

Time per gradient with respect to an index-grid design (`lightsim_autodiff_*_rtx6000.json`).
"python" / "native": `invdes.SMatrixProblem(engine=...)`, solve + vjp of L = sum |S|^2;
"reverse": `lightsim.autodiff` reverse mode, energy through the output plane.

| scene | python | native | speed-up | reverse (fwd + rev) | lightsim.adjoint | FDTDX |
|---|---|---|---|---|---|---|
| 2.5-D 23 k cells, 2 x 2 x 3, 4 runs x 2000 steps | 0.622 s [1.283] | 0.067 s [0.288] | 9.26x | | | |
| 6.0 M x 2623, 137 781 samples | 2.773 s [2.968] | 2.035 s [2.12] | 1.36x | 4.275 s (1.150 + 3.025) [4.286] | 2.418 s [3.256] | 10.74 s |
| 32 M x 3934, 137 781 samples | 15.164 s [15.748] | 12.239 s [12.399] | 1.24x | 29.384 s (6.516 + 22.728) [29.561] | 14.021 s [14.152] | 102.57 s |

The planar scene is one cell thick and runs on the 2-D update in 0.1.0 (`2-D te` backend).
Native session set-up (once): 0.142 s, 0.312 s, 1.745 s. Reverse mode records the PML shell
on the device: 5.25 GB (6 M), 27.7 GB (32 M). Python and native agree: S to 5e-16, gradient
to 1.0e-7. Reverse-mode gradient against central differences of the objective along
sign(grad): 4.1e-05 (6 M, dL/L = 4 %), 2.7e-04 (32 M, dL/L = 2 %) [6.0e-5, 3.1e-4].

## FDTDX's own benchmark: SOI directional coupler (`coupler.gds`, non-uniform grid, mode source + 3 mode detectors)

Same grid (built with FDTDX's node construction, PML inside the domain), same
number of time steps; times are the time-stepping loop (FDTDX: after JIT
compilation; lightsim: setup/mode solve reported separately, < 4 s). The whole FDTDX column was
re-measured (`fdtdx_coupler_20261122*.json`): the GB10 run at 10 cells/λ was 5.2 % faster than before.

| GPU | cells/λ | grid | steps | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|---|
| RTX PRO 6000 | 10 | 948×145×65 | 11 586 | 7.06 s (14.66 Gc/s) [9.23] | 25.07 s (4.13 Gc/s) [24.37] | **3.55×** |
| RTX PRO 6000 | 20 | 1897×291×128 | 23 204 | 73.19 s (22.40 Gc/s) [91.8] | 426.02 s (3.85 Gc/s) [425.8] | **5.82×** |
| RTX PRO 6000 | 30 (4000 steps) | 2845×436×193 | 4 000 | 43.60 s (21.97 Gc/s) [52.6] | 247.03 s (3.88 Gc/s) [246.9] | **5.67×** |
| GB10 | 10 | 948×145×65 | 11 586 | 39.42 s (2.63 Gc/s) [46.0] | 190.15 s (0.54 Gc/s) [200.6] | **4.82×** |
| GB10 | 20 (4000 steps) | 1897×291×128 | 4 000 | 82.76 s (3.42 Gc/s) [100.4] | 470.10 s (0.60 Gc/s) [482.8] | **5.68×** |
| GB10 | 20 (full) | 1897×291×128 | 23 204 | 480 s (one run, 3.41 Gc/s) [582] | — | |

Transmissions (normalised to the source-port detector):

| solver | cells/λ | T_thru | T_cross |
|---|---|---|---|
| lightsim | 10 | 0.5715 | 0.4249 |
| lightsim | 20 | 0.5551 | 0.4422 |
| FDTDX | 10 | 0.2513 | 0.7178 |
| FDTDX | 20 | 0.9229 | 0.0691 |

lightsim's coupler response is converged to ~2 % between 10 and 20 cells/λ
(sub-pixel smoothing of the 200 nm gap and the waveguide sidewalls), while
FDTDX's staircased geometry swings strongly with resolution at these meshes.
lightsim's transmissions agree between the RTX PRO 6000 and the GB10 to 1e-6 and are unchanged
from the earlier build; FDTDX's are identical to its earlier runs.
