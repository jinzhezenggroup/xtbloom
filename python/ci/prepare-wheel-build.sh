#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
bash "$project_dir/python/ci/install-ccache.sh"

# Keep the provenance-reviewed provider available after PEP 517's temporary
# build environment is removed: auditwheel follows the shim's DT_NEEDED edge
# during repair and must resolve the external DSO before it can vendor it.
provider_env="$project_dir/build/wheel-openblas-provider"
UV_DEFAULT_INDEX=https://pypi.org/simple \
UV_PROJECT_ENVIRONMENT="$provider_env" \
UV_PYTHON="$(command -v python)" \
  uv sync --project "$project_dir" --locked --no-install-project \
    --only-group wheel-build

# Fail before the expensive native build if the locked environment differs
# from the reviewed version, architecture, ELF inventory, hashes, or license.
"$provider_env/bin/python" \
  "$project_dir/python/ci/resolve-openblas-wheel.py" \
  --manifest "$project_dir/cmake/3rdparty/scipy_openblas32_manifest.json" \
  >/dev/null
