#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify Torch extension build sources and binaries stay in the right archives."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_SUFFIX = "src/bindings/torch/xtbloom_torch_ext.cpp"
WHEEL_EXTENSION_PATHS = {
    "linux": "xtbloom/lib/libxtbloom_torch_ext.so",
    "macos": "xtbloom/lib/libxtbloom_torch_ext.dylib",
    "windows": "xtbloom/bin/xtbloom_torch_ext.dll",
}
NATIVE_BUILD_SUFFIX = re.compile(
    r"(?:\.so(?:\.[^.]+)*|\.dylib(?:\.[^.]+)*|\.dll(?:\.[^.]+)*|"
    r"\.lib|\.exp|\.pdb|\.def|\.a)$"
)


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


def _torch_native_artifacts(
    names: list[str], *, allowed_extension: str | None
) -> list[str]:
    """Find Torch/c10 native or build artifacts outside the one plugin path."""
    artifacts = []
    for name in names:
        basename = PurePosixPath(name).name.lower()
        torch_related = "torch" in basename or basename.startswith("c10")
        if (
            name != allowed_extension
            and torch_related
            and NATIVE_BUILD_SUFFIX.search(basename)
        ):
            artifacts.append(name)
    return sorted(artifacts)


def _generated_stub_sources(names: list[str]) -> list[str]:
    """Find configure-generated C stubs that no distribution may retain."""
    return sorted(
        name
        for name in names
        if re.fullmatch(
            r"torch_cpu_stub(?:\.[^.]+)*\.c",
            PurePosixPath(name).name.lower(),
        )
    )


def check_wheel(path: Path, *, expect: str = "present") -> None:
    """Enforce the platform extension and reject build/runtime Torch payloads."""
    names = _wheel_names(path)
    leaked = [name for name in names if name.endswith("xtbloom_torch_ext.cpp")]
    if leaked:
        raise RuntimeError(f"{path} leaks Torch extension source: {leaked[0]}")
    platform = _wheel_platform(path, names)
    extensions = _extension_names(names)
    if expect == "absent":
        if extensions:
            raise RuntimeError(
                f"{path} must not contain a Torch extension; found {extensions}"
            )
        leaked_runtime = _torch_native_artifacts(
            names, allowed_extension=None
        ) + _generated_stub_sources(names)
        if leaked_runtime:
            raise RuntimeError(
                f"{path} bundles a forbidden PyTorch runtime/stub: {leaked_runtime[0]}"
            )
        return
    if expect != "present":
        raise ValueError(f"unknown Torch wheel expectation: {expect}")

    expected = WHEEL_EXTENSION_PATHS.get(platform or "")
    if expected is None:
        raise RuntimeError(f"cannot infer a supported Torch wheel platform for {path}")
    if extensions != [expected]:
        raise RuntimeError(f"{path} must contain only {expected}; found {extensions}")

    leaked_runtime = _torch_native_artifacts(
        names, allowed_extension=expected
    ) + _generated_stub_sources(names)
    if leaked_runtime:
        raise RuntimeError(
            f"{path} bundles a forbidden PyTorch runtime/stub: {leaked_runtime[0]}"
        )


def check_sdist(path: Path) -> None:
    """Require source inputs while rejecting generated Torch build artifacts."""
    names = _sdist_names(path)
    leaked_runtime = _torch_native_artifacts(
        names, allowed_extension=None
    ) + _generated_stub_sources(names)
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
