# Install

After this page you have the `lightsim` command-line program and the `lightsim`
Python package built from source, for CPU only or with GPU support, and you
have checked the build against the physics validation suite.

lightsim is built from its source repository,
[github.com/su3co/lightsim](https://github.com/su3co/lightsim) (MIT licence).
There are no pre-built packages yet.

```bash
git clone https://github.com/su3co/lightsim
cd lightsim
```

## Requirements

| Component | Version | Needed for |
|---|---|---|
| Rust toolchain | 1.99.1 | Everything. The file `rust-toolchain.toml` pins it; `rustup` installs it on first use. |
| Python | 3.10 or later | The Python package. |
| maturin | 1.7 or later, below 2 | Building the Python package. |
| numpy, scipy | numpy 1.24, scipy 1.10 or later | Installed with the package. |
| gdsfactory | 8 or later | Optional: importing layouts and port-driven S-parameters. |
| NVIDIA GPU, CUDA toolkit | CUDA 13, driver 580 or later | Optional: GPU execution. |

A CPU-only build needs nothing beyond Rust and Python and works on Linux and
macOS. The samples in this documentation were run with a CPU-only build on
macOS (Apple silicon) and with a CUDA 13.4 build on Linux.

The GPU kernels are written in Rust and compiled to PTX, NVIDIA's portable GPU
assembly, with NVIDIA's cuda-oxide compiler. The compiled PTX file is part of
the repository, so a GPU build does **not** need cuda-oxide. The file targets
compute capability 8.0, so the driver can compile it for Ampere and later
GPUs. It has been tested on the RTX PRO 6000 and the GB10.

## Build the command-line program

From the repository root:

```bash
cargo build --release -p lightsim-cli                    # CPU only
cargo build --release -p lightsim-cli --features cuda    # with GPU support
```

The program is `target/release/lightsim`. Put it on your `PATH` or call it by
its path.

A build with `--features cuda` downloads NVIDIA's `cuda-core` crate, which
generates its bindings from the CUDA headers. It needs to know where the
toolkit and `libclang` are:

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13      # your CUDA 13 installation
export LIBCLANG_PATH=/usr/lib/llvm-21/lib        # directory that holds libclang.so
```

## Build the Python package

The package is a Rust extension module plus Python sources. maturin builds
the extension and installs the package into the active environment in
editable mode, with its numpy and scipy dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release                     # CPU only
maturin develop --release --features cuda     # with GPU support
```

For layout import install gdsfactory as well:

```bash
pip install "gdsfactory>=8"
```

> **Note:** A build without `--features cuda` runs on the CPU only. Asking it
> for a GPU raises an error rather than falling back silently; `device="auto"`
> selects the CPU.

## Check the installation

`lightsim info` prints the version, the number of CPU threads and the GPUs it
can use. A CPU-only build prints:

```text
lightsim 0.1.0 (10 threads)
(built without CUDA)
```

From Python, `lightsim.devices()` lists the same devices:

```python
import lightsim as ls

print(ls.__version__)
print(ls.devices())
```

```text
0.1.0
['cpu']
```

On a GPU build each GPU appears as an extra entry, `cuda:0`, `cuda:1` and so
on, followed by its model name, for example
`'cuda:0 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'`.

`lightsim validate` runs the physics validation suite: 39 simulations whose
results are compared with analytic references, from Fresnel reflection to Mie
scattering. Each case prints the measured and expected values and whether it
passed, and the last line gives the total:

```bash
lightsim validate --device cpu
lightsim validate --device cuda:0
lightsim validate --device cpu --case fresnel    # only cases whose name contains "fresnel"
```

```text
fresnel_normal_n2 [PASS]
    R @ 0.850 f0                             measured  1.101268e-1  expected  1.111111e-1  rel err  8.859e-3 (tol 2.0e-2)
    T @ 0.850 f0                             measured  8.888792e-1  expected  8.888889e-1  rel err  1.095e-5 (tol 1.0e-2)
    ...

47 / 47 cases passed on Cpu
```

The whole suite takes about a minute and a half on a 10-core laptop CPU and
15 seconds on an RTX PRO 6000. The exit status is 1 if any case fails. [Validation](../validation.md) lists the
cases and their accuracy.

The Python test suite covers the Python layer (mode solver, layout import,
S-parameters, gradients):

```bash
pip install pytest
pytest python/tests
```

It uses the CPU unless the environment variable `LIGHTSIM_DEVICE` names
another device, for example `LIGHTSIM_DEVICE=cuda:0`.

## Rebuilding the GPU kernels

This is only needed after editing the kernel source `kernels/src/main.rs`.
`scripts/build_kernels.sh` compiles it with the cuda-oxide toolchain and
copies the new PTX file into `crates/lightsim-cuda/ptx/`. Rebuild the
command-line program and the Python package afterwards.
