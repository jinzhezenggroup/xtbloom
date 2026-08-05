#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run-wheel-test.sh <full|cpu|smoke> <project-dir>}
project_dir=${2:?usage: run-wheel-test.sh <full|cpu|smoke> <project-dir>}

# Toolkit-only CI images have no kernel driver but do ship a libcuda stub. Make
# that stub visible under the production SONAME so the dynamic driver resolver
# exercises a present-library/no-device path during wheel tests. Production
# discovery never searches or preloads toolkit stubs.
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
  cpu)
    # Force a real installed-wheel CPU inference. Import-only smoke tests do
    # not exercise the eigensolver because its BLAS runtime is loaded lazily.
    python - <<'PY'
import numpy as np
from gpuxtb import Calculator

calculator = Calculator(
    "GFN2-xTB",
    np.array([8, 1, 1]),
    np.array(
        [
            [0.0, 0.0, -0.73578586109551],
            [1.44183152868459, 0.0, 0.36789293054775],
            [-1.44183152868459, 0.0, 0.36789293054775],
        ]
    ),
    backend="cpu",
)
result = calculator.singlepoint()
assert result.scc_converged
assert np.isfinite(result.energy)
assert np.isfinite(result.forces).all()
print(f"gpuxtb CPU wheel smoke energy: {result.energy:.16g}")
PY
    ;;
  smoke)
    python -c 'import gpuxtb; print(gpuxtb.library.get_version())'
    ;;
  *)
    echo "unknown wheel-test mode: $mode" >&2
    exit 2
    ;;
esac
