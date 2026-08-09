#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: repair-wheel.sh <destination> <wheel>}
wheel=${2:?usage: repair-wheel.sh <destination> <wheel>}

# Resolve by distribution metadata only. Importing scipy_openblas32 would load
# its provider into the process-global namespace, which is precisely the host
# interaction xTBloom's private shim is designed to avoid.
provider_dir=$(python - <<'PY'
import importlib.metadata
from pathlib import Path

distribution = importlib.metadata.distribution("scipy-openblas32")
provider = distribution.locate_file("scipy_openblas32/lib/libscipy_openblas.so")
print(Path(provider).resolve().parent)
PY
)
export LD_LIBRARY_PATH="$provider_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec auditwheel repair -w "$destination" "$wheel" \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcusolver.so.11 \
  --exclude libcusparse.so.12 \
  --exclude libcudart.so.12 \
  --exclude libnvJitLink.so.12 \
  --exclude libcufft.so.11 \
  --exclude libcurand.so.10 \
  --exclude libcuda.so.1
