#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: repair-wheel.sh <destination> <wheel>}
wheel=${2:?usage: repair-wheel.sh <destination> <wheel>}
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
provider_python="$project_dir/build/wheel-openblas-provider/bin/python"

# Resolve by distribution metadata only. Importing scipy_openblas32 would load
# its provider into the process-global namespace, which is precisely the host
# interaction xTBloom's private shim is designed to avoid.
provider_path=$(
  "$provider_python" "$project_dir/python/ci/resolve-openblas-wheel.py" \
    --manifest "$project_dir/cmake/3rdparty/scipy_openblas32_manifest.json" | \
    "$provider_python" -c \
      'import json, sys; print(json.load(sys.stdin)["provider_path"])'
)
test -f "$provider_path"
provider_dir=$(dirname -- "$provider_path")
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
  --exclude libcuda.so.1 \
  --exclude libtorch_cpu.so
