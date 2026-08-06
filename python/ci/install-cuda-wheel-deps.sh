#!/usr/bin/env bash
# Install the CUDA compute libraries that the PyPA manylinux CUDA images do not
# ship. The build needs their headers and ELF symbol tables to compile the CUDA
# backend and generate the host-API trampolines inside the container.
#
# The quay.io/manylinux_cuda images ship nvcc, headers, cuBLAS, and cuDRT but
# not cuSOLVER (libcudart/.hx absent), so FindCUDAToolkit cannot create the
# CUDA::cusolver target and the compiler cannot find cusolverDn.h. We download
# the small nvidia-cusolver / nvidia-nvjitlink wheels from PyPI with --no-deps,
# extract only the needed ELF libraries and headers into the toolkit dirs that
# CMake and nvcc already search, then delete every downloaded/extracted file so
# the build container never accumulates gigabytes of nvidia packages.
#
# The resulting wheels exclude these host shared libraries (see the
# repair-wheel-command in pyproject.toml); at runtime the trampolines resolve
# the system driver and the same PyPI nvidia-* packages dynamically. gpuxtb's
# device link disables cudadevrt; compiler-inserted libdevice code may remain.
set -euo pipefail

# The container persists across the per-Python-Version before-build calls, so
# install the deps only once.
if [ -f /usr/local/cuda/lib64/libcusolver.so ]; then
  echo "cuSOLVER already installed; skipping"
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# --no-deps keeps this to the cuSOLVER/nvJitLink wheels (pull in neither cuBLAS
# nor cuSPARSE; the image already provides cuBLAS and the linker defers the
# rest, which we exclude from the wheel anyway). uv has no `pip download`
# equivalent, so install the three wheels with `--target` (extracted, not
# zip-archived) into a scratch directory and let the walk below collect the
# ELF libraries and headers from the installed tree.
uv pip install --quiet --no-deps --no-cache \
  nvidia-cusolver-cu12==11.7.5.82 \
  nvidia-nvjitlink-cu12==12.9.86 \
  nvidia-cusparse-cu12==12.5.10.65 \
  --target "$work"/wheels

python - "$work" <<'PY'
import os
import re
import shutil
import sys

work = sys.argv[1]
# `uv pip install --target` has already extracted each wheel into this tree.
extract_dir = os.path.join(work, "wheels")
lib_dir = "/usr/local/cuda/lib64"
include_dirs = [
    "/usr/local/cuda/include",
    "/usr/local/cuda/targets/x86_64-linux/include",
]
os.makedirs(lib_dir, exist_ok=True)

copied_libs = set()
for root, _dirs, files in os.walk(extract_dir):
    for name in files:
        source = os.path.join(root, name)
        with open(source, "rb") as handle:
            magic = handle.read(4)
        if name.startswith("lib") and ".so" in name and magic == b"\x7fELF":
            target = os.path.join(lib_dir, name)
            if name not in copied_libs:
                copied_libs.add(name)
                shutil.copy2(source, target)
        elif name.endswith(".h") or name.endswith(".hpp"):
            for include_dir in include_dirs:
                if os.path.isdir(include_dir):
                    shutil.copy2(source, os.path.join(include_dir, name))

# Provide the unversioned names that CMake's find_library(NAMES <lib>) matches.
for name in sorted(copied_libs):
    match = re.match(r"^(lib[\w.-]+)\.so\.\d+", name)
    if match:
        link = os.path.join(lib_dir, match.group(1) + ".so")
        if not os.path.lexists(link):
            os.symlink(name, link)

if not copied_libs:
    raise SystemExit("no CUDA runtime libraries were installed into " + lib_dir)
print("installed CUDA runtime libraries:", ", ".join(sorted(copied_libs)))
PY

# All temporary downloads and extractions are removed by the EXIT trap above.
