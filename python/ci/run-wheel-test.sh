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
    python -c 'import torch; print(f"wheel test torch: {torch.__version__}")'
    python -m pytest "$project_dir/python/tests" -q
    ;;
  cpu)
    # Force a real installed-wheel CPU inference. Import-only smoke tests do
    # not exercise the eigensolver because its BLAS runtime is loaded lazily.
    python - <<'PY'
import numpy as np
import torch
from gpuxtb import Calculator, gpuxtb_torch

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

# Exercise the repaired wheel's stable-ABI extension and its external
# libtorch_cpu.so resolution on native aarch64 as well as x86_64.
torch_positions = torch.tensor(
    calculator.positions, dtype=torch.float64, requires_grad=True
)
torch_energies, torch_forces = gpuxtb_torch(
    torch_positions,
    torch.tensor(calculator.numbers, dtype=torch.int32),
    torch.tensor([0, len(calculator.numbers)], dtype=torch.int64),
    torch.zeros(1, dtype=torch.float64),
    torch.zeros(1, dtype=torch.int32),
    torch.ones(1, dtype=torch.int32),
    backend="cpu",
)
torch_energies.sum().backward()
torch.testing.assert_close(
    torch_energies,
    torch.tensor([result.energy], dtype=torch.float64),
    atol=1.0e-12,
    rtol=1.0e-12,
)
torch.testing.assert_close(
    torch_forces,
    torch.from_numpy(np.ascontiguousarray(result.forces)),
    atol=1.0e-12,
    rtol=1.0e-12,
)
torch.testing.assert_close(torch_positions.grad, -torch_forces, atol=0.0, rtol=0.0)
print(f"gpuxtb Torch wheel smoke: torch {torch.__version__}")
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
