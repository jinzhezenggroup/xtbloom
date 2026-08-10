#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify Python SCM metadata and the embedded native release contract."""

from __future__ import annotations

import argparse
import ctypes
import email.policy
import os
import re
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

VERSION_STRING_RE = re.compile(r'^#define XTBLOOM_VERSION_STRING "([^"]+)"$', re.M)
VERSION_COMPONENT_RE = re.compile(
    r"^#define XTBLOOM_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)$", re.M
)
_NUMBER = r"(?:0|[1-9][0-9]*)"
RELEASE_RE = re.compile(rf"^({_NUMBER})\.({_NUMBER})\.({_NUMBER})$")
DEVELOPMENT_RE = re.compile(
    rf"^({_NUMBER})\.({_NUMBER})\.({_NUMBER})\.post1\.dev({_NUMBER})"
    r"\+g[0-9a-f]+(?:\.d[0-9]{8})?$"
)


def _native_version_from_distribution(distribution_version: str) -> str:
    """Return the native release tag represented by Python SCM metadata."""
    release = RELEASE_RE.fullmatch(distribution_version)
    if release is not None:
        return distribution_version
    development = DEVELOPMENT_RE.fullmatch(distribution_version)
    if development is None:
        raise RuntimeError(
            f"unsupported Python distribution version {distribution_version}"
        )
    major, minor, patch, _distance = development.groups()
    return f"{major}.{minor}.{patch}"


def _is_native_library(path: PurePosixPath) -> bool:
    """Accept exactly the installed xTBloom library names on native platforms."""
    return (
        re.fullmatch(r"libxtbloom\.so(?:\.[0-9]+)*", path.name) is not None
        or path.name == "libxtbloom.dylib"
        or path.name == "xtbloom.dll"
    )


def _single_member(
    archive: zipfile.ZipFile,
    predicate: Callable[[PurePosixPath], bool],
    label: str,
) -> str:
    """Return the one archive member accepted by ``predicate``."""
    candidates = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and predicate(PurePosixPath(info.filename))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"wheel must contain exactly one {label}; found {len(candidates)}"
        )
    return candidates[0]


def _release_native_library(library: ctypes.CDLL) -> None:
    """Release a loaded DLL before deleting its temporary extraction tree."""
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    free_library = kernel32.FreeLibrary
    free_library.argtypes = [ctypes.c_void_p]
    free_library.restype = ctypes.c_int
    handle = ctypes.c_void_p(library._handle)
    if free_library(handle) == 0:
        raise RuntimeError("failed to release the extracted native wheel DLL")


def inspect_wheel(
    wheel: Path,
    temporary_root: Path,
    *,
    metadata_only: bool = False,
    expected_version: str | None = None,
) -> None:
    """Compare Python SCM metadata with the embedded native release."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = _single_member(
            archive,
            lambda path: (
                path.name == "METADATA" and path.parent.name.endswith(".dist-info")
            ),
            "dist-info/METADATA",
        )
        header_name = _single_member(
            archive,
            lambda path: path.as_posix().endswith("xtbloom/include/xtbloom/version.h"),
            "public version.h",
        )
        library_name = _single_member(
            archive,
            _is_native_library,
            "native libxtbloom",
        )

        metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(metadata_name)
        )
        metadata_version = str(metadata["Version"])
        expected_native_version = _native_version_from_distribution(metadata_version)
        # CMake emits generated headers with the host platform's newline
        # convention. Normalize Windows CRLF before applying line-anchored
        # release-macro checks so equivalent wheel metadata is portable.
        header = archive.read(header_name).decode("utf-8").replace("\r\n", "\n")
        string_match = VERSION_STRING_RE.search(header)
        components = dict(VERSION_COMPONENT_RE.findall(header))
        native_release_match = RELEASE_RE.fullmatch(expected_native_version)
        if string_match is None or native_release_match is None or len(components) != 3:
            raise RuntimeError(f"{wheel} contains malformed version metadata")

        header_version = string_match.group(1)
        native_release = native_release_match.groups()
        header_release = (
            components["MAJOR"],
            components["MINOR"],
            components["PATCH"],
        )
        if expected_version is not None and metadata_version != expected_version:
            raise RuntimeError(
                f"{wheel} version {metadata_version} does not match expected "
                f"Python distribution version {expected_version}"
            )
        if expected_native_version != header_version:
            raise RuntimeError(
                f"{wheel} native release mismatch: Python metadata={metadata_version}, "
                f"expected native={expected_native_version}, header={header_version}"
            )
        if native_release != header_release:
            raise RuntimeError(
                f"{wheel} numeric native version mismatch: expected={native_release}, "
                f"header={header_release}"
            )
        if metadata_only:
            print(  # noqa: T201 - CI validation report
                f"{wheel.name}: python={metadata_version}; "
                f"native/header={expected_native_version}"
            )
            return

        extracted = temporary_root / wheel.stem / PurePosixPath(library_name).name
        extracted.parent.mkdir(parents=True)
        extracted.write_bytes(archive.read(library_name))

    library = ctypes.CDLL(str(extracted))
    try:
        library.xtbloom_version_string.argtypes = []
        library.xtbloom_version_string.restype = ctypes.c_char_p
        native_value = library.xtbloom_version_string()
        if native_value is None:
            raise RuntimeError(f"{wheel} native version function returned NULL")
        native_version = native_value.decode("utf-8")
        if native_version != expected_native_version:
            raise RuntimeError(
                f"{wheel} native release mismatch: Python metadata={metadata_version}, "
                f"header={expected_native_version}, native={native_version}"
            )
    finally:
        # Windows denies unlinking a loaded DLL, so balance LoadLibrary before
        # TemporaryDirectory attempts to remove the extracted wheel payload.
        _release_native_library(library)
    print(  # noqa: T201 - CI validation report
        f"{wheel.name}: python={metadata_version}; "
        f"native/header/C-API={expected_native_version}"
    )


def main() -> int:
    """Inspect every wheel passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="compare wheel metadata and headers without loading the target DSO",
    )
    parser.add_argument(
        "--expected-version",
        help="require wheel metadata to match this Python distribution version",
    )
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="xtbloom-wheel-version-") as directory:
        root = Path(directory)
        for wheel in args.wheels:
            inspect_wheel(
                wheel,
                root,
                metadata_only=args.metadata_only,
                expected_version=args.expected_version,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
