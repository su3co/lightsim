#!/usr/bin/env bash
# Install, in a manylinux_2_28 (AlmaLinux 8) build container, what the
# `cuda` feature needs to compile: the CUDA 13 headers that cuda-core's
# bindings generator reads (cuda.h, curand.h and what they include) and
# libclang for bindgen. Nothing from here ends up in the wheel: cuda-core
# loads the driver library (libcuda.so.1) at run time.
#
# Prints the environment to export afterwards, for example:
#   eval "$(bash .github/scripts/install-cuda-headers.sh)"
set -euo pipefail

CUDA_PKG_VERSION="${CUDA_PKG_VERSION:-13-0}"   # CUDA 13.0: runs on every driver >= 580

case "$(uname -m)" in
    x86_64) repo_arch=x86_64 ;;
    aarch64) repo_arch=sbsa ;;
    *) echo "unsupported architecture $(uname -m)" >&2; exit 1 ;;
esac

{
    curl -fsSL -o /etc/yum.repos.d/cuda-rhel8.repo \
        "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${repo_arch}/cuda-rhel8.repo"
    dnf install -y -q --setopt=install_weak_deps=False \
        "cuda-driver-devel-${CUDA_PKG_VERSION}" \
        "cuda-cudart-devel-${CUDA_PKG_VERSION}" \
        "cuda-crt-${CUDA_PKG_VERSION}" \
        "libcurand-devel-${CUDA_PKG_VERSION}" \
        clang clang-devel
} >&2

toolkit="/usr/local/cuda-${CUDA_PKG_VERSION/-/.}"
test -f "${toolkit}/include/cuda.h" || test -n "$(ls "${toolkit}"/targets/*/include/cuda.h 2>/dev/null)" || {
    echo "cuda.h not found under ${toolkit}" >&2; exit 1; }
libclang_dir="$(dirname "$(ls /usr/lib64/libclang.so* | head -1)")"

echo "export CUDA_TOOLKIT_PATH=${toolkit}"
echo "export LIBCLANG_PATH=${libclang_dir}"
echo "export BINDGEN_EXTRA_CLANG_ARGS=-I$(clang -print-resource-dir)/include"
