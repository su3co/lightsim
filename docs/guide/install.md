# Install

After this page you have the `lightsim` Python package, installed from PyPI or
built from source, for CPU only or with GPU support, and you have checked it
against the physics validation suite. The command-line program `lightsim`
is built from source.

## Install from PyPI

**Not on PyPI yet.** The first release, 0.1.0, has not been published.
Until it is, build from source as described [below](#build-from-source);
the wheels described here are built and tested on every change.

```bash
pip install lightsim
```

pip picks the wheel for your platform. One wheel serves every Python from 3.10
on:

| Platform | Wheel | Runs on |
|---|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64`, with GPU support | CPU; NVIDIA GPUs |
| Linux aarch64 (Grace, GB10) | `manylinux_2_28_aarch64`, with GPU support | CPU; NVIDIA GPUs |
| macOS, Apple silicon | `macosx_11_0_arm64` | CPU |
| macOS, Intel | `macosx_10_13_x86_64` | CPU |

The Linux wheels need glibc 2.28 or later (RHEL 8, Ubuntu 20.04, Debian 10 and
later). They contain the GPU backend and its kernels, and nothing from the
CUDA toolkit: lightsim loads the NVIDIA driver when it starts and uses it if
there is one. To run on a GPU you need only

* an NVIDIA GPU of compute capability 8.0 or later (Ampere, Ada, Hopper,
  Blackwell), and
* the NVIDIA driver, version 580 or later (the first with CUDA 13).

No CUDA toolkit, no `nvcc`, no `LD_LIBRARY_PATH`. On a machine without a
driver the same wheel runs on the CPU: `lightsim.devices()` lists only
`'cpu'`, `device="auto"` selects the CPU, and asking for `device="cuda:0"`
raises an error saying that no GPU is available.

numpy and scipy are installed with the package. Optional extras:

```bash
pip install "lightsim[gds]"     # gdsfactory: layout import, port-driven S-parameters
pip install "lightsim[plot]"    # matplotlib
```

Other platforms (Windows, Linux on other architectures, older glibc) have no
wheel; pip then builds the package from the source distribution, which
needs the Rust toolchain (see below) and gives a CPU-only build.

## Build from source

Building from source gives the command-line program, a GPU build on a
platform without a GPU wheel, the double-precision build (below), and the
development version. The source repository is
[github.com/su3co/lightsim](https://github.com/su3co/lightsim) (MIT licence).

```bash
git clone https://github.com/su3co/lightsim
cd lightsim
```

### Requirements

| Component | Version | Needed for |
|---|---|---|
| Rust toolchain | 1.99.1 | Everything. The file `rust-toolchain.toml` pins it; `rustup` installs it on first use. |
| Python | 3.10 or later | The Python package. |
| maturin | 1.7 or later, below 2 | Building the Python package. |
| numpy, scipy | numpy 1.24, scipy 1.10 or later | Installed with the package. |
| gdsfactory | 8 or later | Optional: importing layouts and port-driven S-parameters. |
| NVIDIA GPU, CUDA toolkit | CUDA 13, driver 580 or later | Optional: GPU execution. |

A CPU-only build needs nothing beyond Rust and Python and works on Linux and
macOS. `pip install lightsim` on a platform without a wheel does such a build
by itself. A GPU build needs the CUDA 13 headers (the driver API and cuRAND)
and `libclang` at build time, and only the driver at run time. The samples in this documentation were run with a CPU-only build on
macOS (Apple silicon) and with a CUDA 13.4 build on Linux.

The GPU kernels are written in Rust and compiled to PTX, NVIDIA's portable GPU
assembly, with NVIDIA's cuda-oxide compiler. The compiled PTX file is part of
the repository, so a GPU build does **not** need cuda-oxide. The file targets
compute capability 8.0, so the driver can compile it for Ampere and later
GPUs. It has been tested on the RTX PRO 6000 and the GB10.

### Build the command-line program

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

### Build the Python package

This replaces a package installed from PyPI. The package is a Rust extension
module plus Python sources. maturin builds
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

> **Note:** A build without `--features cuda` (and the macOS wheels) runs on the CPU only. Asking it
> for a GPU raises an error rather than falling back silently; `device="auto"`
> selects the CPU.

## Precision

The default build stores fields, update coefficients and recorded data in
single precision (`f32`), on the CPU and on the GPU. A CPU-only build in
double precision (`f64`) is available with the cargo feature `f64`:

```bash
cargo build --release -p lightsim-cli --features f64     # command-line program
maturin develop --release --features f64                 # Python package
```

In that build the CPU engine, the plan, results and monitors, time reversal
and both gradient engines (`lightsim.autodiff`, all engines of
`lightsim.invdes`) work in `f64`, and the Python package returns `float64`
arrays (`complex128` design fields). `lightsim info` and
`lightsim.precision` say which build is installed (`"f32"` or `"f64"`).
Parameters, geometry, frequencies and post-processing are `f64` in both
builds; the default build compiles to the same machine code as before the
feature existed (the feature only switches the type alias `Real`).

The GPU kernels are single precision, so `f64` cannot be combined with
`cuda`: the build stops with a message, and asking an `f64` build for a GPU
raises an error naming the precision.

What it costs, `lightsim bench --n 256,256,256 --steps 50 --device cpu`
(best of three):

| build | 10-core laptop CPU | peak memory (Linux workstation, same grid) |
|---|---|---|
| default (`f32`) | 416 Mcells/s | 0.83 GB |
| `--features f64` | 307 Mcells/s (0.74 x) | 1.69 GB (2.0 x) |

The time step is memory-bound, so the cost follows the doubled data; on a
workstation CPU shared with other jobs the same build ran at 0.52 x. Use it
where single-precision round-off limits a result: finite-difference checks
of gradients (reverse mode agrees with central differences to 1e-11 ... 1e-10
in `f64`, against 1e-4 ... 1e-3 in `f32`), long runs that accumulate small
differences, or quantities near 1e-7 of the field scale. The `lightsim
validate` tolerances are unchanged; 33 of the 512 measured values improve by
more than ten times (those set by round-off, such as TFSF leakage, 1e-7 to
1e-17), the others are set by the discretisation and stay the same.

## Check the installation

The command-line program, if you built it, prints the version, the number of
CPU threads and the GPUs it can use with `lightsim info`. A CPU-only build
prints:

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

On a GPU build (the Linux wheels, or `--features cuda`) on a machine with an
NVIDIA driver, each GPU appears as an extra entry, `cuda:0`, `cuda:1` and so
on, followed by its model name, for example
`'cuda:0 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'`.

`lightsim validate` (command-line program) runs the physics validation suite: 68 cases whose
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

68 / 68 cases passed on Cpu
```

The whole suite takes about a minute and a half on a 10-core laptop CPU and
15 seconds on an RTX PRO 6000. The exit status is 1 if any case fails. [Validation](../validation.md) lists the
cases and their accuracy.

The Python test suite, in the source repository, covers the Python layer
(mode solver, layout import, S-parameters, gradients). It runs against the
installed package, from PyPI or built from source:

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
