#!/usr/bin/env bash
# Compile the cuda-oxide kernels to PTX and install them into lightsim-cuda.
# Requires the cuda-oxide toolchain (cargo-oxide, Rust 1.99.1 with rustc-dev,
# CUDA 13). PTX targets sm_80 so one artifact serves Ampere through Blackwell
# (RTX PRO 6000 = sm_120, GB10 = sm_121); the driver JITs it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HOME/.config/lightsim/env.sh" ] && source "$HOME/.config/lightsim/env.sh"
cd "$ROOT/kernels"
cargo oxide build --arch "${LIGHTSIM_PTX_ARCH:-sm_80}"
cp lightsim_kernels.ptx "$ROOT/crates/lightsim-cuda/ptx/lightsim_kernels.ptx"
echo "installed $(wc -c < lightsim_kernels.ptx) bytes of PTX"
