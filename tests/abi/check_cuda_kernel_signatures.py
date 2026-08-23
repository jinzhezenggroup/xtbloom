"""Reject CUDA kernels whose parameters are passed by reference.

CUDA lowers a kernel reference parameter to a pointer to the launcher's host
object instead of copying the descriptor into launch argument storage.  Such a
kernel can appear to work on a full-HMM host while faulting on a GPU that cannot
access ordinary host stack memory.  The CUDA programming guide therefore
forbids pass-by-reference parameters on ``__global__`` functions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

KERNEL_DECLARATION = re.compile(
    r"\b__global__\b(?P<declaration>.*?)(?:\{|;)", re.DOTALL
)
CUDA_SOURCE_SUFFIXES = frozenset({".cu", ".cuh"})


def find_reference_kernel_parameters(source_root: Path) -> list[tuple[Path, int]]:
    """Return source locations of kernel declarations containing references."""
    violations: list[tuple[Path, int]] = []
    for subtree in (Path("src/backends/cuda"), Path("src/runtime")):
        for path in sorted((source_root / subtree).rglob("*")):
            if path.suffix not in CUDA_SOURCE_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for match in KERNEL_DECLARATION.finditer(text):
                if "&" not in match.group("declaration"):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                violations.append((path.relative_to(source_root), line))
    return violations


def main() -> int:
    """Audit production CUDA kernel signatures below one source root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    violations = find_reference_kernel_parameters(args.source_root.resolve())
    for path, line in violations:
        print(  # noqa: T201 - command-line validation report
            f"{path}:{line}: __global__ parameters must be passed by value or pointer, "
            "never by reference"
        )
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
