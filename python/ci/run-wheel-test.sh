#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run-wheel-test.sh <full|smoke> <project-dir>}
project_dir=${2:?usage: run-wheel-test.sh <full|smoke> <project-dir>}

# CUDA-enabled wheels link the driver SONAME libcuda.so.1. Toolkit-only CI
# images ship a linkable stub named libcuda.so, so expose an exact SONAME link
# in a temporary directory for the wheel test. Production discovery never
# searches or preloads this stub.
cuda_root=${CUDA_HOME:-/usr/local/cuda}
stub_source=$(find -L "$cuda_root" -path '*/stubs/libcuda.so' -print -quit)
if [[ -z "$stub_source" ]]; then
  echo "CUDA driver stub libcuda.so was not found under $cuda_root" >&2
  exit 1
fi

stub_dir=$(mktemp -d)
trap 'rm -rf "$stub_dir"' EXIT
ln -s "$stub_source" "$stub_dir/libcuda.so.1"
export LD_LIBRARY_PATH="$stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "$mode" in
  full)
    python -m pytest "$project_dir/python/tests" -q
    ;;
  smoke)
    python -c 'import gpuxtb; print(gpuxtb.library.get_version())'
    ;;
  *)
    echo "unknown wheel-test mode: $mode" >&2
    exit 2
    ;;
esac
