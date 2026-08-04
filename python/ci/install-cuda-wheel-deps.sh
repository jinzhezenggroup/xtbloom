#!/usr/bin/env bash
# Install the CUDA compute libraries that the PyPA manylinux CUDA images do not
# ship, so the gpuxtb CUDA backend (which links cuBLAS/cuSOLVER/cuDRT) can be
# compiled inside the container.
#
# The quay.io/manylinux_cuda images ship nvcc, headers, cuBLAS, and cuDRT but
# not libcusolver. FindCUDAToolkit therefore cannot create the CUDA::cusolver
# imported target. We pull the missing .so files from PyPI and drop them into
# the toolkit's lib directory that CMake already searches.
#
# The resulting wheels still exclude these libraries (see the
# repair-wheel-command in pyproject.toml); at runtime a CUDA-enabled wheel uses
# the system CUDA driver plus the same PyPI nvidia-* packages.
set -euo pipefail

python -m pip install --quiet --no-cache-dir nvidia-cusolver-cu12 nvidia-nvjitlink-cu12

python - <<'PY'
import glob
import os
import shutil
import site

dest = "/usr/local/cuda/lib64"
os.makedirs(dest, exist_ok=True)

pattern = os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib", "*.so*")
installed = []
for source in glob.glob(pattern):
    name = os.path.basename(source)
    target = os.path.join(dest, name)
    # The same .so appears under multiple nvidia/*/lib dirs; keep the first.
    if os.path.isfile(target):
        continue
    shutil.copy2(source, target)
    installed.append(name)

if not installed:
    raise SystemExit("no CUDA runtime libraries were installed into " + dest)
print("installed CUDA runtime libraries:", ", ".join(sorted(installed)))
PY