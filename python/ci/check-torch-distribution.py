#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify Torch extension build sources and binaries stay in the right archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_SUFFIX = "src/bindings/torch/xtbloom_torch_ext.cpp"
WHEEL_EXTENSION_PATHS = {
    "linux": "xtbloom/lib/libxtbloom_torch_ext.so",
    "macos": "xtbloom/lib/libxtbloom_torch_ext.dylib",
    "windows": "xtbloom/bin/xtbloom_torch_ext.dll",
}
FORBIDDEN_TORCH_RUNTIME_NAMES = {
    "c10.dll",
    "c10_cuda.dll",
    "libc10.dylib",
    "libc10.so",
    "libc10_cuda.dylib",
    "libc10_cuda.so",
    "libtorch.dylib",
    "libtorch.so",
    "libtorch_cpu.dylib",
    "libtorch_cpu.so",
    "libtorch_cuda.dylib",
    "libtorch_cuda.so",
    "libtorch_global_deps.dylib",
    "libtorch_global_deps.so",
    "libtorch_python.dylib",
    "libtorch_python.so",
    "torch.dll",
    "torch_cpu.dll",
    "torch_cpu.exp",
    "torch_cpu.lib",
    "torch_cpu.pdb",
    "torch_cpu_stub.c",
    "torch_cpu_stub.def",
    "torch_cuda.dll",
}


def _wheel_names(path: Path) -> list[str]:
    """Return regular-file names from one wheel."""
    with zipfile.ZipFile(path) as archive:
        return [info.filename for info in archive.infolist() if not info.is_dir()]


def _sdist_names(path: Path) -> list[str]:
    """Return regular-file names from one source distribution."""
    with tarfile.open(path, "r:*") as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]


def _wheel_platform(path: Path, names: list[str]) -> str | None:
    """Infer the native wheel platform from its tag or extension payload."""
    filename = path.name.lower()
    if "manylinux" in filename or "musllinux" in filename:
        return "linux"
    if "macosx" in filename:
        return "macos"
    if "win_" in filename:
        return "windows"
    if "pyemscripten" in filename or "emscripten" in filename:
        return "pyodide"
    for platform, expected in WHEEL_EXTENSION_PATHS.items():
        if expected in names:
            return platform
    for name in _extension_names(names):
        basename = PurePosixPath(name).name.lower()
        if ".dylib" in basename:
            return "macos"
        if basename.endswith(".dll") or ".dll." in basename:
            return "windows"
        if ".so" in basename:
            return "linux"
    return None


def _extension_names(names: list[str]) -> list[str]:
    """Return every file whose basename resembles the xTBloom Torch plugin."""
    return sorted(
        name
        for name in names
        if PurePosixPath(name).name.startswith(
            ("libxtbloom_torch_ext", "xtbloom_torch_ext")
        )
    )


def check_wheel(path: Path, *, expect: str = "present") -> None:
    """Enforce the platform extension and reject build/runtime Torch payloads."""
    names = _wheel_names(path)
    leaked = [name for name in names if name.endswith("xtbloom_torch_ext.cpp")]
    if leaked:
        raise RuntimeError(f"{path} leaks Torch extension source: {leaked[0]}")
    leaked_runtime = sorted(
        name
        for name in names
        if PurePosixPath(name).name.lower() in FORBIDDEN_TORCH_RUNTIME_NAMES
    )
    if leaked_runtime:
        raise RuntimeError(
            f"{path} bundles a forbidden PyTorch runtime/stub: {leaked_runtime[0]}"
        )

    extensions = _extension_names(names)
    if expect == "absent":
        if extensions:
            raise RuntimeError(
                f"{path} must not contain a Torch extension; found {extensions}"
            )
        return
    if expect != "present":
        raise ValueError(f"unknown Torch wheel expectation: {expect}")

    platform = _wheel_platform(path, names)
    expected = WHEEL_EXTENSION_PATHS.get(platform or "")
    if expected is None:
        raise RuntimeError(f"cannot infer a supported Torch wheel platform for {path}")
    if extensions != [expected]:
        raise RuntimeError(f"{path} must contain only {expected}; found {extensions}")


def check_sdist(path: Path) -> None:
    """Require source inputs while rejecting generated Torch build artifacts."""
    names = _sdist_names(path)
    leaked_runtime = sorted(
        name
        for name in names
        if PurePosixPath(name).name.lower() in FORBIDDEN_TORCH_RUNTIME_NAMES
    )
    if leaked_runtime:
        raise RuntimeError(
            f"{path} bundles a generated PyTorch runtime/stub: {leaked_runtime[0]}"
        )
    roots = {
        PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts
    }
    if len(roots) != 1:
        raise RuntimeError(
            f"{path} must contain exactly one archive root; found {sorted(roots)}"
        )
    expected = f"{next(iter(roots))}/{SOURCE_SUFFIX}"
    sources = sorted(
        name for name in names if PurePosixPath(name).name == "xtbloom_torch_ext.cpp"
    )
    if sources != [expected]:
        raise RuntimeError(f"{path} must contain only {expected}; found {sources}")


def main() -> int:
    """Check all requested source or wheel distribution archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wheel", nargs="+", type=Path)
    group.add_argument("--sdist", nargs="+", type=Path)
    parser.add_argument(
        "--expect",
        choices=("present", "absent"),
        default="present",
        help="whether wheel archives must contain the platform Torch extension",
    )
    args = parser.parse_args()

    paths = args.wheel if args.wheel is not None else args.sdist
    for path in paths:
        if args.wheel is not None:
            check_wheel(path, expect=args.expect)
        else:
            check_sdist(path)
        print(f"Torch distribution payload is valid: {path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
