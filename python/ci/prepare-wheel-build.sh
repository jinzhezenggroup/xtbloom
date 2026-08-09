#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
bash "$project_dir/python/ci/install-ccache.sh"

# Keep the provenance-reviewed provider available after PEP 517's temporary
# build environment is removed: auditwheel follows the shim's DT_NEEDED edge
# during repair and must resolve the external DSO before it can vendor it.
python -m pip install \
  --index-url https://pypi.org/simple \
  --only-binary=:all: \
  'scipy-openblas32==0.3.34.0.0'
