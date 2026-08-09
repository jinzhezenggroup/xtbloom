#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify Torch extension build sources and binaries stay in the right archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_SUFFIX = "python/src/gpuxtb_torch_ext.cpp"
WHEEL_EXTENSION_PATH = "gpuxtb/lib/libgpuxtb_torch_ext.so"


def _wheel_names(path: Path) -> list[str]:
    """Return regular-file names from one wheel."""
    with zipfile.ZipFile(path) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


def _sdist_names(path: Path) -> list[str]:
    """Return regular-file names from one source distribution."""
    with tarfile.open(path, "r:*") as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]


def check_wheel(path: Path) -> None:
    """Require the compiled extension and reject leaked C++ build inputs."""
    names = _wheel_names(path)
    leaked = [name for name in names if name.endswith("gpuxtb_torch_ext.cpp")]
    if leaked:
        raise RuntimeError(f"{path} leaks Torch extension source: {leaked[0]}")
    extensions = sorted(
        name
        for name in names
        if PurePosixPath(name).name.startswith("libgpuxtb_torch_ext")
    )
    if extensions != [WHEEL_EXTENSION_PATH]:
        raise RuntimeError(
            f"{path} must contain only {WHEEL_EXTENSION_PATH}; found {extensions}"
        )


def check_sdist(path: Path) -> None:
    """Require the non-package C++ source needed by downstream wheel builds."""
    names = _sdist_names(path)
    roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(roots) != 1:
        raise RuntimeError(
            f"{path} must contain exactly one archive root; found {sorted(roots)}"
        )
    expected = f"{next(iter(roots))}/{SOURCE_SUFFIX}"
    sources = sorted(
        name
        for name in names
        if PurePosixPath(name).name == "gpuxtb_torch_ext.cpp"
    )
    if sources != [expected]:
        raise RuntimeError(f"{path} must contain only {expected}; found {sources}")


def main() -> int:
    """Check all requested source or wheel distribution archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wheel", nargs="+", type=Path)
    group.add_argument("--sdist", nargs="+", type=Path)
    args = parser.parse_args()

    paths = args.wheel if args.wheel is not None else args.sdist
    check = check_wheel if args.wheel is not None else check_sdist
    for path in paths:
        check(path)
        print(f"Torch distribution payload is valid: {path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
