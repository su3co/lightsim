### 2-D, one input and one output, one wavelength

| tool | hardware | precision | 15.6 k cells x 4759 steps | 140 k cells x 14276 steps | 1.26 M cells x 42826 steps |
|---|---|---|---|---|---|
| lightsim, native engine | 1x RTX PRO 6000 | single | 0.069 s | 0.228 s | 2.27 s |
| lightsim, Python engine | 1x RTX PRO 6000 | single | 0.340 s | 0.538 s | 2.70 s |
| lightsim, reverse mode | 1x RTX PRO 6000 | single | 0.367 s | 1.21 s | 7.51 s |
| lightsim, native engine | 2x RTX PRO 6000 | single | 0.035 s | 0.117 s | 1.14 s |
| lightsim, Python engine | 2x RTX PRO 6000 | single | 0.248 s | 0.279 s | 1.41 s |
| Meep adjoint | 64-core CPU (Threadripper 9980X), MPI | double | 0.293 s, 4 proc. | 1.62 s, 16 proc. | 13.2 s, 64 proc. |
| ceviche (autograd, as shipped) | 64-core CPU (Threadripper 9980X), MKL | double | 0.050 s, 8 thr. | 0.423 s, 32 thr. | 3.97 s, 32 thr. |
| ceviche operators, one LU per wavelength | 64-core CPU (Threadripper 9980X), MKL | double | 0.027 s, 32 thr. | 0.233 s, 32 thr. | 2.13 s, 32 thr. |

Gradient against central finite differences (worst patch, relative to the largest / derivative along a smooth direction), and against lightsim's gradient map (cosine similarity of the interior samples):

| tool | wg2d_45nm | wg2d_15nm | wg2d_5nm |
|---|---|---|---|
| lightsim, native engine | 5.6 % / 0.13 % | 0.12 % / 2e-04 | 0.78 % / 0.17 % |
| lightsim, Python engine | 5.6 % / 0.13 %; cos 1.0000 | 0.12 % / 2e-04; cos 1.0000 | 0.78 % / 0.17 % |
| lightsim, reverse mode | 3e-05 / 0.13 % | 8e-06 / 0.12 % | 9e-06 / 0.12 % |
| Meep adjoint | 0.16 % / 0.45 %; cos 0.9997 | 6e-04 / 2e-04; cos 1.0000 | 0.83 % / 0.17 % |
| ceviche (autograd, as shipped) | 4e-05 / 0.59 %; cos 0.9959 | 3e-05 / 0.11 %; cos 1.0000 | 6e-05 / 8e-04 |
| ceviche operators, one LU per wavelength | 4e-05 / 0.59 %; cos 0.9959 | 3e-05 / 0.11 %; cos 1.0000 | 3e-05 / 8e-04 |

Objective value (same design in every tool); for the matrix scenes also the power reaching the four outputs per unit input power, averaged over inputs and wavelengths:

| tool | wg2d_45nm | wg2d_15nm | wg2d_5nm |
|---|---|---|---|
| lightsim, native engine | 0.03586 | 0.03173 | 0.04247 |
| lightsim, Python engine | 0.03586 | 0.03173 | 0.04247 |
| Meep adjoint | 0.03497 | 0.03169 | 0.04248 |
| ceviche (autograd, as shipped) | 0.03412 | 0.03203 | 0.04247 |
| ceviche operators, one LU per wavelength | 0.03412 | 0.03203 | 0.04247 |

### 3-D, one input and one output, one wavelength

| tool | hardware | precision | 6.08 M cells x 5770 steps | 32 M cells x 8917 steps | 86.4 M cells x 12064 steps |
|---|---|---|---|---|---|
| lightsim, native engine | 1x RTX PRO 6000 | single | 4.45 s | 27.8 s | 92.6 s |
| lightsim, Python engine | 1x RTX PRO 6000 | single | 5.03 s | 29.7 s | 96.9 s |
| lightsim, reverse mode | 1x RTX PRO 6000 | single | 10.2 s | 72.5 s |  |
| lightsim, native engine | 2x RTX PRO 6000 | single | 2.22 s | 13.9 s | 46.3 s |
| lightsim, Python engine | 2x RTX PRO 6000 | single | 2.48 s | 15.0 s | 49.2 s |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 1x RTX PRO 6000 | single | 13.6 s | 142 s |  |
| FDTDX, reversible autodiff (16-bit recorder, every 10th step) | 1x RTX PRO 6000 | single | 13.6 s |  | 509 s (1 run) |
| FDTDX, reversible autodiff (16-bit recorder) | 1x RTX PRO 6000 | single | 13.4 s | 141 s |  |
| FDTDX, reversible autodiff (exact recorder) | 1x RTX PRO 6000 | single | 13.8 s |  |  |
| Meep adjoint | 64-core CPU (Threadripper 9980X), MPI | double | 295 s, 64 proc. | ~2554 s, 64 proc., extrapolated from 100- and 200-step runs (417 M cells/s) and the full wg3d_6M gradient | ~9153 s, 64 proc., extrapolated from 100- and 200-step runs (425 M cells/s) and the full wg3d_6M gradient |

Gradient against central finite differences (worst patch, relative to the largest / derivative along a smooth direction), and against lightsim's gradient map (cosine similarity of the interior samples):

| tool | wg3d_6M | wg3d_32M | wg3d_86M |
|---|---|---|---|
| lightsim, native engine | 0.33 % / 0.27 % | 3e-04 / 0.17 % | - / 0.22 % |
| lightsim, Python engine | 0.33 % / 0.27 %; cos 1.0000 | ; cos 1.0000 | ; cos 1.0000 |
| lightsim, reverse mode | 5e-04 / 0.14 % | 5e-04 / 0.15 % |  |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 7.7 % / 6.9 %; cos 0.9849 | - / 23 %; cos 0.9742 |  |
| FDTDX, reversible autodiff (16-bit recorder, every 10th step) | 2.2 % / 1.5 %; cos 0.9874 |  | ; cos 0.9850 |
| FDTDX, reversible autodiff (16-bit recorder) | 0.55 % / 0.29 %; cos 0.9874 | - / 0.33 %; cos 0.9880 |  |
| FDTDX, reversible autodiff (exact recorder) | 7e-04 / 9e-04; cos 0.9874 |  |  |
| Meep adjoint | 0.11 % / 7e-04; cos 0.9999 |  |  |

Objective value (same design in every tool); for the matrix scenes also the power reaching the four outputs per unit input power, averaged over inputs and wavelengths:

| tool | wg3d_6M | wg3d_32M | wg3d_86M |
|---|---|---|---|
| lightsim, native engine | 0.2094 | 0.2126 | 0.2132 |
| lightsim, Python engine | 0.2094 | 0.2126 | 0.2132 |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 0.2886 | 0.2942 |  |
| FDTDX, reversible autodiff (16-bit recorder, every 10th step) | 0.2886 |  | 0.293 |
| FDTDX, reversible autodiff (16-bit recorder) | 0.2886 | 0.2942 |  |
| FDTDX, reversible autodiff (exact recorder) | 0.2886 |  |  |
| Meep adjoint | 0.2171 |  |  |

### 2-D, 4 inputs x 4 outputs x 3 wavelengths

| tool | hardware | precision | 28.2 k cells x 5711 steps | 254 k cells x 17131 steps | 2.29 M cells x 51391 steps |
|---|---|---|---|---|---|
| lightsim, native engine | 1x RTX PRO 6000 | single | 0.345 s | 1.43 s | 18.5 s |
| lightsim, Python engine | 1x RTX PRO 6000 | single | 1.52 s | 2.81 s | 23.7 s |
| lightsim, native engine | 2x RTX PRO 6000 | single | 0.175 s | 0.729 s | 14.1 s |
| lightsim, Python engine | 2x RTX PRO 6000 | single | 0.975 s | 1.66 s | 12.1 s |
| Meep adjoint | 64-core CPU (Threadripper 9980X), MPI | double | 7.56 s, 16 proc. | 40.9 s, 64 proc. | 199 s, 64 proc. (1 run) |
| ceviche (autograd, as shipped) | 64-core CPU (Threadripper 9980X), MKL | double | 0.926 s, 32 thr. | 7.68 s, 32 thr. | 92.3 s, 32 thr. (2 runs) |
| ceviche operators, one LU per wavelength | 64-core CPU (Threadripper 9980X), MKL | double | 0.174 s, 32 thr. | 1.79 s, 32 thr. | 19.9 s, 64 thr. |

Gradient against central finite differences (worst patch, relative to the largest / derivative along a smooth direction), and against lightsim's gradient map (cosine similarity of the interior samples):

| tool | mm2d_45nm | mm2d_15nm | mm2d_5nm |
|---|---|---|---|
| lightsim, native engine | 2.1 % / 5.8 % | 0.69 % / 4.1 % | 2e-04 / 3.5 % |
| lightsim, Python engine | 2.1 % / 5.8 %; cos 1.0000 | 0.69 % / 4.1 %; cos 1.0000 |  |
| Meep adjoint | 0.41 % / 4.4 %; cos 0.9994 | 0.74 % / 4.4 %; cos 0.9999 | - / 3.5 % |
| ceviche (autograd, as shipped) | 5e-05 / 3.7 %; cos 0.9994 | 6e-05 / 3.4 %; cos 0.9999 |  |
| ceviche operators, one LU per wavelength | 5e-05 / 3.7 %; cos 0.9994 | 6e-05 / 3.4 %; cos 0.9999 | 7e-05 / 2.8 % |

Objective value (same design in every tool); for the matrix scenes also the power reaching the four outputs per unit input power, averaged over inputs and wavelengths:

| tool | mm2d_45nm | mm2d_15nm | mm2d_5nm |
|---|---|---|---|
| lightsim, native engine | 0.005211 (throughput 0.135) | 0.005033 (throughput 0.136) | 0.004507 (throughput 0.137) |
| lightsim, Python engine | 0.005211 (throughput 0.135) | 0.005033 (throughput 0.136) | 0.004507 (throughput 0.137) |
| Meep adjoint | 0.005046 (throughput 0.130) | 0.005133 (throughput 0.135) | 0.004599 (throughput 0.137) |
| ceviche (autograd, as shipped) | 0.005165 (throughput 0.135) | 0.005106 (throughput 0.135) | 0.004595 (throughput 0.137) |
| ceviche operators, one LU per wavelength | 0.005165 (throughput 0.135) | 0.005106 (throughput 0.135) | 0.004595 (throughput 0.137) |

### 3-D, 4 inputs x 4 outputs x 3 wavelengths

| tool | hardware | precision | 1.52 M cells x 4769 steps | 12.2 M cells x 9537 steps |
|---|---|---|---|---|
| lightsim, native engine | 1x RTX PRO 6000 | single | 3.28 s | 57.6 s |
| lightsim, Python engine | 1x RTX PRO 6000 | single | 4.44 s | 60.8 s |
| lightsim, native engine | 2x RTX PRO 6000 | single | 1.63 s | 29.1 s |
| lightsim, Python engine | 2x RTX PRO 6000 | single | 2.49 s | 31.0 s |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 1x RTX PRO 6000 | single | 22.8 s |  |
| FDTDX, reversible autodiff (exact recorder) | 1x RTX PRO 6000 | single | 22.4 s |  |
| Meep adjoint | 64-core CPU (Threadripper 9980X), MPI | double | 188 s, 64 proc. | ~10628 s, 64 proc., extrapolated from 100- and 200-step runs (389 M cells/s) and the full mm3d_44nm gradient |

Gradient against central finite differences (worst patch, relative to the largest / derivative along a smooth direction), and against lightsim's gradient map (cosine similarity of the interior samples):

| tool | mm3d_44nm | mm3d_22nm |
|---|---|---|
| lightsim, native engine | 1 % / 1.9 % | 0.17 % / 0.22 % |
| lightsim, Python engine | 1 % / 1.9 %; cos 1.0000 | ; cos 1.0000 |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 2.6 % / 5.4 %; cos 0.7562 |  |
| FDTDX, reversible autodiff (exact recorder) | 3e-04 / 0.49 %; cos 0.7559 |  |
| Meep adjoint | 1e-04 / 0.19 %; cos 0.9955 |  |

Objective value (same design in every tool); for the matrix scenes also the power reaching the four outputs per unit input power, averaged over inputs and wavelengths:

| tool | mm3d_44nm | mm3d_22nm |
|---|---|---|
| lightsim, native engine | 0.00249 (throughput 0.441) | 0.002865 (throughput 0.448) |
| lightsim, Python engine | 0.00249 (throughput 0.441) | 0.002865 (throughput 0.448) |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | 0.09786 (throughput 1.435) |  |
| FDTDX, reversible autodiff (exact recorder) | 0.09786 (throughput 1.435) |  |
| Meep adjoint | 0.002428 (throughput 0.456) |  |

### Cost per gradient in forward runs, one-time costs, memory

| tool | scene | hardware | runs per gradient | gradient / one forward run | first call | one-time set-up | peak memory |
|---|---|---|---|---|---|---|---|
| lightsim, native engine | mm2d_15nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.92 | 1.66 s | 0.211 s | 0.6 GB (GPU) |
| lightsim, Python engine | mm2d_15nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 8.28 | 2.77 s | 0.023 s | 0.6 GB (GPU) |
| lightsim, native engine | mm2d_15nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.95 | 1.09 s | 0.323 s | 1.2 GB (GPU) |
| lightsim, Python engine | mm2d_15nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.55 | 1.73 s | 0.024 s | 1.1 GB (GPU) |
| Meep adjoint | mm2d_15nm | 64-core CPU (Threadripper 9980X), MPI | 4 forward, 4 adjoint | 30.77 | 41.0 s | 0.002 s | 17.4 GB (host) |
| ceviche (autograd, as shipped) | mm2d_15nm | 64-core CPU (Threadripper 9980X), MKL | 12 forward factorisations, 12 adjoint factorisations | 21.80 | 7.78 s | 4.69 s | 2.5 GB (host) |
| ceviche operators, one LU per wavelength | mm2d_15nm | 64-core CPU (Threadripper 9980X), MKL | 3 factorisations, 12 forward substitutions, 12 adjoint substitutions |  | 1.99 s | 4.78 s | 3.4 GB (host) |
| lightsim, native engine | mm2d_45nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.83 | 0.505 s | 0.150 s | 0.5 GB (GPU) |
| lightsim, Python engine | mm2d_45nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.67 | 1.60 s | 0.017 s | 0.5 GB (GPU) |
| lightsim, native engine | mm2d_45nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.84 | 0.413 s | 0.222 s | 1.1 GB (GPU) |
| lightsim, Python engine | mm2d_45nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.64 | 0.825 s | 0.016 s | 0.6 GB (GPU) |
| Meep adjoint | mm2d_45nm | 64-core CPU (Threadripper 9980X), MPI | 4 forward, 4 adjoint | 29.69 | 7.67 s | 0.001 s | 3.2 GB (host) |
| ceviche (autograd, as shipped) | mm2d_45nm | 64-core CPU (Threadripper 9980X), MKL | 12 forward factorisations, 12 adjoint factorisations | 21.25 | 0.917 s | 0.597 s | 0.4 GB (host) |
| ceviche operators, one LU per wavelength | mm2d_45nm | 64-core CPU (Threadripper 9980X), MKL | 3 factorisations, 12 forward substitutions, 12 adjoint substitutions |  | 0.183 s | 0.659 s | 0.5 GB (host) |
| lightsim, native engine | mm2d_5nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.70 | 19.3 s | 0.888 s | 1.2 GB (GPU) |
| lightsim, Python engine | mm2d_5nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.98 | 24.0 s | 0.186 s | 0.8 GB (GPU) |
| lightsim, native engine | mm2d_5nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 6.90 | 11.9 s | 0.678 s | 2.3 GB (GPU) |
| lightsim, Python engine | mm2d_5nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.94 | 12.2 s | 0.158 s | 1.7 GB (GPU) |
| Meep adjoint | mm2d_5nm | 64-core CPU (Threadripper 9980X), MPI | 4 forward, 4 adjoint | 19.46 | 199 s | 0.039 s | 26.3 GB (host) |
| ceviche (autograd, as shipped) | mm2d_5nm | 64-core CPU (Threadripper 9980X), MKL | 12 forward factorisations, 12 adjoint factorisations | 23.15 | 95.9 s | 52.9 s | 20.5 GB (host) |
| ceviche operators, one LU per wavelength | mm2d_5nm | 64-core CPU (Threadripper 9980X), MKL | 3 factorisations, 12 forward substitutions, 12 adjoint substitutions |  | 20.9 s | 50.1 s | 32.5 GB (host) |
| lightsim, native engine | mm3d_22nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.99 | 59.4 s | 1.62 s | 2.0 GB (GPU) |
| lightsim, Python engine | mm3d_22nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 8.04 | 60.9 s | 0.732 s | 1.7 GB (GPU) |
| lightsim, native engine | mm3d_22nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 8.00 | 31.5 s | 2.37 s | 3.9 GB (GPU) |
| lightsim, Python engine | mm3d_22nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 8.01 | 31.1 s | 0.643 s | 3.4 GB (GPU) |
| Meep adjoint | mm3d_22nm | 64-core CPU (Threadripper 9980X), MPI | 4 forward, 4 adjoint |  |  |  |  |
| lightsim, native engine | mm3d_44nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.97 | 3.70 s | 0.403 s | 0.7 GB (GPU) |
| lightsim, Python engine | mm3d_44nm | 1x RTX PRO 6000 | 4 forward, 4 adjoint | 7.81 | 4.68 s | 0.191 s | 0.7 GB (GPU) |
| lightsim, native engine | mm3d_44nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 7.97 | 2.21 s | 0.509 s | 1.5 GB (GPU) |
| lightsim, Python engine | mm3d_44nm | 2x RTX PRO 6000 | 4 forward, 4 adjoint | 8.09 | 2.63 s | 0.182 s | 1.4 GB (GPU) |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | mm3d_44nm | 1x RTX PRO 6000 | 4 forward, 4 reverse pass | 10.73 | 22.6 s | 26.6 s (compile 16.0 s) | 2.9 GB (GPU) |
| FDTDX, reversible autodiff (exact recorder) | mm3d_44nm | 1x RTX PRO 6000 | 4 forward, 4 reverse pass | 10.54 | 22.2 s | 33.9 s (compile 22.4 s) | 50.1 GB (GPU) |
| Meep adjoint | mm3d_44nm | 64-core CPU (Threadripper 9980X), MPI | 4 forward, 4 adjoint | 8.97 | 315 s | 0.001 s | 20.4 GB (host) |
| lightsim, native engine | wg2d_15nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.97 | 0.352 s | 0.112 s | 0.6 GB (GPU) |
| lightsim, Python engine | wg2d_15nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.12 | 0.555 s | 0.028 s | 0.6 GB (GPU) |
| lightsim, reverse mode | wg2d_15nm | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 5.64 | 1.25 s | 0.018 s | 0.8 GB (GPU) |
| lightsim, native engine | wg2d_15nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.01 | 0.303 s | 0.161 s | 1.1 GB (GPU) |
| lightsim, Python engine | wg2d_15nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.08 | 0.307 s | 0.019 s | 1.1 GB (GPU) |
| Meep adjoint | wg2d_15nm | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint | 2.02 | 1.66 s | 0.000 s | 3.0 GB (host) |
| ceviche (autograd, as shipped) | wg2d_15nm | 64-core CPU (Threadripper 9980X), MKL | 1 forward factorisations, 1 adjoint factorisations | 1.81 | 0.421 s | 0.279 s | 0.8 GB (host) |
| ceviche operators, one LU per wavelength | wg2d_15nm | 64-core CPU (Threadripper 9980X), MKL | 1 factorisations, 1 forward substitutions, 1 adjoint substitutions |  | 0.247 s | 0.278 s | 0.7 GB (host) |
| lightsim, native engine | wg2d_45nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.98 | 0.195 s | 0.120 s | 0.5 GB (GPU) |
| lightsim, Python engine | wg2d_45nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.00 | 0.358 s | 0.009 s | 0.5 GB (GPU) |
| lightsim, reverse mode | wg2d_45nm | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 6.04 | 0.380 s | 0.010 s | 0.6 GB (GPU) |
| lightsim, native engine | wg2d_45nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.01 | 0.203 s | 0.160 s | 1.1 GB (GPU) |
| lightsim, Python engine | wg2d_45nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.56 | 0.288 s | 0.008 s | 0.5 GB (GPU) |
| Meep adjoint | wg2d_45nm | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint | 2.01 | 0.323 s | 0.000 s | 0.7 GB (host) |
| ceviche (autograd, as shipped) | wg2d_45nm | 64-core CPU (Threadripper 9980X), MKL | 1 forward factorisations, 1 adjoint factorisations | 1.77 | 0.051 s | 0.082 s | 0.2 GB (host) |
| ceviche operators, one LU per wavelength | wg2d_45nm | 64-core CPU (Threadripper 9980X), MKL | 1 factorisations, 1 forward substitutions, 1 adjoint substitutions |  | 0.029 s | 0.101 s | 0.2 GB (host) |
| lightsim, native engine | wg2d_5nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.98 | 2.51 s | 0.191 s | 0.7 GB (GPU) |
| lightsim, Python engine | wg2d_5nm | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.02 | 2.77 s | 0.134 s | 0.7 GB (GPU) |
| lightsim, reverse mode | wg2d_5nm | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 4.51 | 7.55 s | 0.051 s | 3.1 GB (GPU) |
| lightsim, native engine | wg2d_5nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.01 | 1.45 s | 0.234 s | 1.4 GB (GPU) |
| lightsim, Python engine | wg2d_5nm | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.07 | 1.45 s | 0.145 s | 1.3 GB (GPU) |
| Meep adjoint | wg2d_5nm | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint | 1.95 | 13.6 s | 0.003 s | 13.7 GB (host) |
| ceviche (autograd, as shipped) | wg2d_5nm | 64-core CPU (Threadripper 9980X), MKL | 1 forward factorisations, 1 adjoint factorisations | 1.78 | 4.28 s | 2.42 s | 6.3 GB (host) |
| ceviche operators, one LU per wavelength | wg2d_5nm | 64-core CPU (Threadripper 9980X), MKL | 1 factorisations, 1 forward substitutions, 1 adjoint substitutions |  | 2.27 s | 2.48 s | 5.8 GB (host) |
| lightsim, native engine | wg3d_32M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.00 | 29.7 s | 1.71 s | 2.9 GB (GPU) |
| lightsim, Python engine | wg3d_32M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.99 | 29.7 s | 4.52 s | 2.8 GB (GPU) |
| lightsim, reverse mode | wg3d_32M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 3.94 | 72.5 s | 1.14 s | 76.4 GB (GPU) |
| lightsim, native engine | wg3d_32M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.00 | 16.4 s | 2.23 s | 5.7 GB (GPU) |
| lightsim, Python engine | wg3d_32M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.01 | 15.2 s | 3.94 s | 5.6 GB (GPU) |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | wg3d_32M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 2.33 |  | 18.5 s (compile 2.5 s) | 22.1 GB (GPU) |
| FDTDX, reversible autodiff (16-bit recorder) | wg3d_32M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 2.32 |  | 15.8 s (compile 2.1 s) | 39.7 GB (GPU) |
| Meep adjoint | wg3d_32M | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint |  |  |  |  |
| lightsim, native engine | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.99 | 4.82 s | 0.310 s | 1.0 GB (GPU) |
| lightsim, Python engine | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.04 | 5.11 s | 0.718 s | 1.0 GB (GPU) |
| lightsim, reverse mode | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 3.17 | 10.3 s | 0.146 s | 15.0 GB (GPU) |
| lightsim, native engine | wg3d_6M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.00 | 2.78 s | 0.454 s | 2.1 GB (GPU) |
| lightsim, Python engine | wg3d_6M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.00 | 2.60 s | 0.699 s | 2.1 GB (GPU) |
| FDTDX, reversible autodiff (8-bit recorder of its examples) | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 2.98 | 13.4 s | 11.7 s (compile 2.7 s) | 4.0 GB (GPU) |
| FDTDX, reversible autodiff (16-bit recorder, every 10th step) | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 3.00 | 13.5 s | 9.63 s (compile 2.5 s) | 4.0 GB (GPU) |
| FDTDX, reversible autodiff (16-bit recorder) | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 3.00 | 13.3 s | 10.8 s (compile 2.6 s) | 7.3 GB (GPU) |
| FDTDX, reversible autodiff (exact recorder) | wg3d_6M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 3.02 | 13.7 s | 10.9 s (compile 2.4 s) | 65.6 GB (GPU) |
| Meep adjoint | wg3d_6M | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint | 1.66 | 295 s | 0.000 s | 15.8 GB (host) |
| lightsim, native engine | wg3d_86M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 2.00 | 97.6 s | 4.43 s | 6.5 GB (GPU) |
| lightsim, Python engine | wg3d_86M | 1x RTX PRO 6000 | 1 forward, 1 adjoint | 1.99 | 96.5 s | 10.8 s | 6.3 GB (GPU) |
| lightsim, native engine | wg3d_86M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.00 | 52.5 s | 5.55 s | 13.0 GB (GPU) |
| lightsim, Python engine | wg3d_86M | 2x RTX PRO 6000 | 1 forward, 1 adjoint | 1.01 | 48.8 s | 11.0 s | 12.6 GB (GPU) |
| FDTDX, reversible autodiff (16-bit recorder, every 10th step) | wg3d_86M | 1x RTX PRO 6000 | 1 forward, 1 reverse pass | 2.31 |  | 33.9 s (compile 2.6 s) | 61.4 GB (GPU) |
| Meep adjoint | wg3d_86M | 64-core CPU (Threadripper 9980X), MPI | 1 forward, 1 adjoint |  |  |  |  |
