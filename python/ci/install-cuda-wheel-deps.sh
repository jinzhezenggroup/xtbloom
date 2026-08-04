#!/usr/bin/env bash
# Install the CUDA compute libraries that the PyPA manylinux CUDA images do not
# ship, so the gpuxtb CUDA backend (which links cuBLAS/cuSOLVER/cuDRT) can be
# compiled inside the container.
#
# The quay.io/manylinux_cuda images ship nvcc, headers, cuBLAS, and cuDRT but
# not cuSOLVER. FindCUDAToolkit therefore cannot create the CUDA::cusolver
# imported target. We pull the missing .so files from PyPI and drop them into
# the toolkit's lib directory that CMake already searches, and create the
# unversioned `lib<name>.so` symlinks that CMake's `find_library(NAMES ...)`
# relies on (the packages only ship soname-suffixed files, e.g.
# libcusolver.so.11).
#
# The resulting wheels still exclude these libraries (see the
# repair-wheel-command in pyproject.toml); at runtime a CUDA-enabled wheel uses
# the system CUDA driver plus the same PyPI nvidia-* packages.
set -euo pipefail

python -m pip install --quiet --no-cache-dir nvidia-cusolver-cu12 nvidia-nvjitlink-cu12

python - <<'PY'
import glob
import os
import re
import shutil
import site

dest = "/usr/local/cuda/lib64"
os.makedirs(dest, exist_ok=True)

copied = set()
for source in glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib", "*.so*")):
    if os.path.islink(source):
        continue  # copy the real files; symlinks are recreated below
    name = os.path.basename(source)
    if name in copied:
        continue
    copied.add(name)
    shutil.copy2(source, os.path.join(dest, name))

# Provide the unversioned names that CMake's find_library(NAMES <lib>) matches.
for name in sorted(copied):
    match = re.match(r"^(lib[\w.-]+)\.so\.\d+", name)
    if match:
        link = os.path.join(dest, match.group(1) + ".so")
        if not os.path.lexists(link):
            os.symlink(name, link)

# The image also lacks the cuSOLVER headers (cusolverDn.h, cusolver_common.h,
# ...); copy them into every CUDA include directory on the compile path.
header_source = os.path.join(site.getsitepackages()[0], "nvidia", "*", "include", "*.h*")
headers = [path for path in glob.glob(header_source) if path.endswith((".h", ".hpp"))]
include_dirs = [
    "/usr/local/cuda/include",
    "/usr/local/cuda/targets/x86_64-linux/include",
]
copied_headers = set()
for include_dir in include_dirs:
    if not os.path.isdir(include_dir):
        continue
    for source in headers:
        name = os.path.basename(source)
        if name in copied_headers:
            continue
        copied_headers.add(name)
        shutil.copy2(source, os.path.join(include_dir, name))

if not copied:
    raise SystemExit("no CUDA runtime libraries were installed into " + dest)
print("installed CUDA runtime libraries:", ", ".join(sorted(copied)))
print("installed CUDA headers:", ", ".join(sorted(copied_headers)) or "(none found)")
PY