"""Verify that a CUDA-enabled shared gpuxtb library has no DT_NEEDED entry on
a GPL-incompatible NVIDIA library.

libgpuxtb.so resolves cudart/cuBLAS/cuSOLVER/libcuda lazily through dlopen
trampolines (see src/runtime/cuda_dlopen.c); a hard DT_NEEDED entry on one of
these proprietary libraries would change the GPL compatibility analysis of the
distribution (Issue #162) and break the no-CUDA-machine load fallback.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FORBIDDEN_NEEDED_PARTS = (
    "libcudart",
    "libcublas",
    "libcusolver",
    "libcuda",
    "libnv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readelf", required=True, help="readelf-compatible executable"
    )
    parser.add_argument(
        "--library", required=True, type=Path, help="shared library to inspect"
    )
    args = parser.parse_args()

    completed = subprocess.run(
        [args.readelf, "-d", str(args.library)],
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden: list[str] = []
    for line in completed.stdout.splitlines():
        if "(NEEDED)" not in line:
            continue
        name = line.rsplit("[", 1)[-1].rstrip("]")
        if any(part in name for part in FORBIDDEN_NEEDED_PARTS):
            forbidden.append(name)

    if forbidden:
        print(
            "CUDA-enabled libgpuxtb must not carry a DT_NEEDED entry on an NVIDIA "
            "library; found: " + ", ".join(sorted(forbidden))
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())