#!/usr/bin/env bash
# Run the gpuxtb conformance suite inside a cibuildwheel test environment.
#
# This is invoked from [tool.cibuildwheel.linux] test-command (cibuildwheel's
# own test feature). It adds two runtime library locations to LD_LIBRARY_PATH
# before running pytest:
#
#  - the MKL runtime (libmkl_rt) installed by `mkl` on PyPI, which the CPU
#    eigensolver dlopens; and
#  - the CUDA runtime libraries in the manylinux CUDA container
#    (/usr/local/cuda/lib64), which a CUDA-enabled libgpuxtb is linked against.
set -euo pipefail

cd "$(dirname "$0")/../.."  # repository root

mkl_lib="$(python - <<'PY'
import os
import site
import glob

sp = site.getsitepackages()[0]
candidate = os.path.join(sp, "mkl", "lib")
if os.path.isdir(candidate):
    print(candidate)
else:
    matches = sorted(glob.glob(os.path.join(sp, "**", "libmkl_rt.so*"), recursive=True))
    if matches:
        print(os.path.dirname(matches[0]))
PY
)"

export LD_LIBRARY_PATH="${mkl_lib:+${mkl_lib}:}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

exec python -m pytest python/tests -q