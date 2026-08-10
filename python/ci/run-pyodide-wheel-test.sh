#!/usr/bin/env bash
set -euo pipefail

project_dir=${1:?usage: run-pyodide-wheel-test.sh <project-dir>}
runner="$project_dir/python/ci/run-pyodide-wheel-test.py"

# scipy-openblas32 is a native build input and must never leak into this
# WebAssembly test environment or published metadata.
python - <<'PY'
import importlib.metadata

try:
    importlib.metadata.distribution("scipy-openblas32")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("native scipy-openblas32 leaked into the Pyodide environment")
PY

mapfile -t private_paths < <(
  python "$runner" --source-root "$project_dir" --mode locate
)
if [[ ${#private_paths[@]} -ne 2 ]]; then
  echo "failed to locate the private Pyodide provider cohort" >&2
  exit 1
fi
adapter=${private_paths[0]}
provider=${private_paths[1]}

# Prove that neither a missing adapter nor a missing/corrupt private provider
# can silently reuse SciPy's globally loaded OpenBLAS module.
mv "$adapter" "$adapter.disabled"
python "$runner" --source-root "$project_dir" --mode expect-unavailable
mv "$adapter.disabled" "$adapter"

mv "$provider" "$provider.disabled"
python "$runner" --source-root "$project_dir" --mode expect-unavailable
mv "$provider.disabled" "$provider"

cp "$provider" "$provider.pristine"
printf 'corrupt-provider\n' > "$provider"
python "$runner" --source-root "$project_dir" --mode expect-unavailable
mv "$provider.pristine" "$provider"

# Fresh Pyodide/Node processes are required because the native backend factory
# and Emscripten dynamic-linker module table are process-lifetime state.
python "$runner" --source-root "$project_dir" --mode scipy-first --full
python "$runner" --source-root "$project_dir" --mode xtbloom-first
