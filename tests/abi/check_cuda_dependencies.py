"""Verify that a CUDA-enabled shared gpuxtb library is loader-closed.

libgpuxtb.so resolves NVIDIA APIs through hidden, pre-resolved trampolines.  A
hard NVIDIA DT_NEEDED, an unresolved CUDA-facing symbol, or an exported loader
implementation symbol would break the no-runtime load contract and ABI gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

FORBIDDEN_NEEDED_PREFIXES = (
    "libcudart",
    "libcudnn",
    "libcufile",
    "libcublas",
    "libcutensor",
    "libcusolver",
    "libcusparse",
    "libcurand",
    "libcufft",
    "libcupti",
    "libcuda",
    "libnccl",
    "libnv",
    "libnpp",
    "libnvidia",
    "libnvshmem",
)
CUDA_SYMBOL_RE = re.compile(r"^(?:__cuda|cuda|cublas|cusolver|cu[A-Z])")
LOADER_SYMBOL_RE = re.compile(
    r"^(?:gpuxtb_cuda_|gpu_xtb_cuda_|_lib(?:cuda|cudart|cublas|cusolver)_so_tramp_)"
)


def find_forbidden_needed(dynamic_output: str) -> list[str]:
    """Return NVIDIA shared objects present in the ELF dependency list."""
    forbidden: set[str] = set()
    for line in dynamic_output.splitlines():
        if "(NEEDED)" not in line or "[" not in line:
            continue
        name = line.rsplit("[", 1)[-1].rstrip("]").lower()
        if name.startswith(FORBIDDEN_NEEDED_PREFIXES):
            forbidden.add(name)
    return sorted(forbidden)


def find_cuda_symbol_leaks(dynamic_symbols_output: str) -> tuple[list[str], list[str]]:
    """Return unresolved CUDA names and defined loader/CUDA exports."""
    unresolved: set[str] = set()
    exported: set[str] = set()
    for line in dynamic_symbols_output.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":"):
            continue
        bind, visibility, section, versioned_name = (
            fields[4],
            fields[5],
            fields[6],
            fields[7],
        )
        if bind not in {"GLOBAL", "WEAK"}:
            continue
        name = versioned_name.split("@", 1)[0]
        if section == "UND" and CUDA_SYMBOL_RE.match(name):
            unresolved.add(name)
        elif (
            section != "UND"
            and visibility == "DEFAULT"
            and (CUDA_SYMBOL_RE.match(name) or LOADER_SYMBOL_RE.match(name))
        ):
            exported.add(name)
    return sorted(unresolved), sorted(exported)


def main() -> int:
    """Inspect one shared library for forbidden CUDA loader dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readelf", required=True, help="readelf-compatible executable"
    )
    parser.add_argument(
        "--library", required=True, type=Path, help="shared library to inspect"
    )
    args = parser.parse_args()

    dynamic = subprocess.run(
        [args.readelf, "-dW", str(args.library)],
        check=True,
        capture_output=True,
        text=True,
    )
    dynamic_symbols = subprocess.run(
        [args.readelf, "--dyn-syms", "-W", str(args.library)],
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = find_forbidden_needed(dynamic.stdout)
    unresolved, exported = find_cuda_symbol_leaks(dynamic_symbols.stdout)

    failed = False
    if forbidden:
        print(  # noqa: T201 - CLI validation report
            "CUDA-enabled libgpuxtb must not carry a DT_NEEDED entry on an NVIDIA "
            "library; found: " + ", ".join(forbidden)
        )
        failed = True
    if unresolved:
        print(  # noqa: T201 - CLI validation report
            "unresolved CUDA-facing symbols: " + ", ".join(unresolved)
        )
        failed = True
    if exported:
        print(  # noqa: T201 - CLI validation report
            "exported CUDA loader/shim symbols: " + ", ".join(exported)
        )
        failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
