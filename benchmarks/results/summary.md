# Benchmark results (2026-11-21)

Hardware: a workstation with 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 96 GB,
CUDA 13.4, driver 615.71) and nodes with one NVIDIA GB10 each (sm_121, 121 GB unified,
CUDA 13.0, driver 580.178).
FDTDX 0.6.3 with JAX 0.12.1 (CUDA 13). lightsim fused single-kernel step, f32.
Gcells/s = 10⁹ cell updates per second (MCUPS / 1000).

## Vacuum box, 12-cell CPML, dipole, 200 steps (`lightsim bench`, `fdtdx_box.py`)

| GPU(s) | grid | lightsim Gcells/s | FDTDX Gcells/s | speed-up |
|---|---|---|---|---|
| 1× RTX PRO 6000 | 256³ | 18.8 | 10.9 | 1.72× |
| 1× RTX PRO 6000 | 400³ | 18.4 | 11.1 | 1.66× |
| 1× RTX PRO 6000 | 640×640×512 | 21.2 | 11.2 | 1.90× |
| 1× RTX PRO 6000 | 1024×640×640 | 21.1 | 11.3 | 1.87× |
| 2× RTX PRO 6000 | 640×640×512 | 39.6 | 18.9 | 2.10× |
| 2× RTX PRO 6000 | 1024×640×640 | 40.8 | 23.7 | 1.72× |
| 2× RTX PRO 6000 | 1600×800×800 (1.02 B) | 43.6 | 27.7 | 1.58× |
| GB10 | 128³ | 2.75 | 1.40 | 1.96× |
| GB10 | 256³ | 3.01 | 1.52 | 1.98× |
| GB10 | 400³ | 3.03 | 1.64 | 1.85× |
| GB10 | 600×640×512 | 3.13 | 1.74 | 1.80× |
| GB10 | 640×640×512 | 3.14 | 1.71 | 1.83× |
| GB10 | 900×800×800 | 3.21 | 1.60 | 2.01× |
| GB10 | 1200×640×512 | — | 1.73 | |
| 2× GB10 (200 GbE link; FDTDX NCCL RDMA) | 1200×640×512 (393 M) | 5.90 | 4.57 | 1.29× |
| 2× GB10 (200 GbE link; FDTDX NCCL sockets) | 1200×640×512 (393 M) | 5.90 | 2.52 | 2.34× |
| 2× GB10 (200 GbE link; FDTDX NCCL RDMA) | 1800×800×800 (1.15 B) | 6.25 | 4.36 | 1.43× |
| 3× GB10 (TCP, 200 GbE) | 1200×640×512 (393 M) | 8.41 | — | 2.65× one GB10 |
| 3× GB10 (TCP, 200 GbE) | 1800×800×800 (1.15 B) | 9.13 | — | 2.87× one GB10 |
| 3× GB10 (10 GbE; FDTDX NCCL sockets) | 1200×640×512 (393 M) | 8.15 | 0.79 | 10.3× |
| 3× GB10 (10 GbE; FDTDX NCCL sockets) | 1800×800×800 (1.15 B) | 8.82 | 1.13 | 7.8× |

Multi-node: lightsim ranks exchange halos over TCP on the direct QSFP links
(`--chain-addrs`); FDTDX uses `jax.distributed` with one process per node
(`fdtdx_multinode.py`), NCCL over RoCE on both ConnectX-7 halves of the link
(`NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0` / `rocep1s0f1,roceP2p1s0f1`) or over
TCP sockets. The 2-node FDTDX run reproduces the single-node final field
(Σ E² 2.482678 vs 2.482678, max |E| identical), so its superlinear speed-up
(1.73 → 4.57 Gcells/s) is genuine. The 2-node grids split into 600×640×512 and
900×800×800 slabs per GPU, which is why the single-GB10 rows above include those
sizes. On 3 GB10 nodes FDTDX runs over the 10 GbE Ethernet network (coordinator port inside
the open ephemeral range 32768-60999; NCCL_SOCKET_IFNAME=enP7s7), because NCCL
cannot use the per-link subnets of the 200 GbE point-to-point links. Its 3-node final field
matches the single-node run (Σ E² 2.482678).

Split (non-fused) lightsim kernels for reference: 1× RTX PRO 6000 14.3–16.0, 2× 30.0–32.4, GB10 1.61–2.29 Gcells/s.

## Feature scenes (`lightsim bench --scene`, `fdtdx_scenes.py`), 200 steps, Gcells/s

| scene | GPU(s) | grid | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|
| Lorentz everywhere | 1× RTX PRO 6000 | 400³ | 8.92 | 5.23 | 1.71× |
| Lorentz everywhere | 1× RTX PRO 6000 | 640×640×512 | 9.50 | 5.29 | 1.80× |
| Lorentz everywhere | 2× RTX PRO 6000 | 640×640×512 | 18.46 | 9.72 | 1.90× |
| Lorentz everywhere | GB10 | 400³ | 1.40 | 0.80 | 1.75× |
| Bloch k≠0 (x, y) | 1× RTX PRO 6000 | 400³ | 7.82 | 3.92 | 2.00× |
| Bloch k≠0 (x, y) | 1× RTX PRO 6000 | 640×640×512 | 8.34 | 3.94 | 2.12× |
| Bloch k≠0 (x, y) | 2× RTX PRO 6000 | 640×640×512 | 16.09 | 6.56 | 2.45× |
| Bloch k≠0 (x, y) | GB10 | 400³ | 1.20 | 0.55 | 2.18× |
| rotated crystal (off-diagonal ε) | 1× RTX PRO 6000 | 400³ | 5.19 | 0.853 | 6.1× |
| rotated crystal (off-diagonal ε) | 1× RTX PRO 6000 | 640×640×512 | 5.40 | 0.731 | 7.4× |
| rotated crystal (off-diagonal ε) | 2× RTX PRO 6000 | 640×640×512 | 10.48 | 0.688 | 15.2× |
| rotated crystal (off-diagonal ε) | GB10 | 400³ | 0.81 | 0.142 | 5.7× |

FDTDX was pinned with `CUDA_VISIBLE_DEVICES`: without it, JAX shards over
every visible GPU. Before the dense dispersive mode, lightsim's Lorentz scene
ran at 5.85/6.10 Gcells/s (1 GPU) with sparse per-cell lists.

## Inverse-design gradient (`lightsim_gradient.py`, `fdtdx_gradient.py`), 1x RTX PRO 6000

| cells x steps | lightsim forward | FDTDX forward | lightsim gradient (adjoint) | FDTDX gradient (jax.value_and_grad, reversible) | speed-up |
|---|---|---|---|---|---|
| 6.0 M x 2623 | 1.40 s | 3.01 s | 2.93 s | 10.74 s | 3.7x |
| 32 M x 3934 | 9.60 s | 34.34 s | 19.47 s | 102.57 s | 5.3x |

GPU time reversal (`time_reversal_gpu.rs`, ignored test `gpu_reverse_throughput`):
256^3 periodic box, 200 steps, forward 18.24 Gcells/s, reverse 13.19 Gcells/s,
reconstruction error 8.0e-7.

## Gradients inside the engine (`lightsim_autodiff.py`), 1x RTX PRO 6000

Time per gradient with respect to an index-grid design (`lightsim_autodiff_*_rtx6000.json`).
"python" / "native": `invdes.SMatrixProblem(engine=...)`, solve + vjp of L = sum |S|^2;
"reverse": `lightsim.autodiff` reverse mode, energy through the output plane.

| scene | python | native | speed-up | reverse (fwd + rev) | lightsim.adjoint | FDTDX |
|---|---|---|---|---|---|---|
| 2.5-D 23 k cells, 2 x 2 x 3, 4 runs x 2000 steps | 1.283 s | 0.288 s | 4.46x | | | |
| 6.0 M x 2623, 137 781 samples | 2.968 s | 2.120 s | 1.40x | 4.286 s (1.149 + 3.039) | 3.256 s | 10.74 s |
| 32 M x 3934, 137 781 samples | 15.748 s | 12.399 s | 1.27x | 29.561 s (6.539 + 22.861) | 14.152 s | 102.57 s |

Native session set-up (once): 0.221 s, 0.327 s, 2.154 s. Reverse mode records the PML shell
on the device: 5.25 GB (6 M), 27.7 GB (32 M). Python and native agree: S to 3e-16, gradient
to 1.2e-7. Reverse-mode gradient against central differences of the objective along
sign(grad): 6.0e-5 (6 M, dL/L = 4 %), 3.1e-4 (32 M, dL/L = 1.7 %).

Earlier batch of the same build, with the adjoint of the reverse pass on the fused kernel
and the forward CPML in place of its transpose (dropped: 1.4 % gradient error at 6 M):
python 1.173 / 3.015 / 15.539 s, native 0.283 / 2.102 / 12.383 s, reverse 3.653 s
(1.145 + 2.410) and 22.681 s (6.597 + 16.025), lightsim.adjoint 2.538 / 14.289 s.

## FDTDX's own benchmark: SOI directional coupler (`coupler.gds`, non-uniform grid, mode source + 3 mode detectors)

Same grid (built with FDTDX's node construction, PML inside the domain), same
number of time steps; times are the time-stepping loop (FDTDX: after JIT
compilation; lightsim: setup/mode solve reported separately, < 4 s).

| GPU | cells/λ | grid | steps | lightsim | FDTDX | speed-up |
|---|---|---|---|---|---|---|
| RTX PRO 6000 | 10 | 948×145×65 | 11 586 | 9.23 s (11.2 Gc/s) | 24.37 s (4.25 Gc/s) | **2.64×** |
| RTX PRO 6000 | 20 | 1897×291×128 | 23 204 | 91.8 s (17.9 Gc/s) | 425.8 s (3.85 Gc/s) | **4.64×** |
| RTX PRO 6000 | 30 | 2845×436×193 | 4 000 | 52.6 s (18.2 Gc/s) | 246.9 s (3.88 Gc/s) | **4.70×** |
| GB10 | 10 | 948×145×65 | 11 586 | 46.0 s (2.25 Gc/s) | 200.6 s (0.52 Gc/s) | **4.36×** |
| GB10 | 20 (4000 steps) | 1897×291×128 | 4 000 | 100.4 s (2.81 Gc/s) | 482.8 s (0.59 Gc/s) | **4.81×** |
| GB10 | 20 (full) | 1897×291×128 | 23 204 | 582 s (2.82 Gc/s) | — | |

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
lightsim produces bit-identical transmissions on the RTX PRO 6000 and the GB10.
