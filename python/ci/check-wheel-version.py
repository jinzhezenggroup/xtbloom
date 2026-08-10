#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Require wheel metadata, public headers, and the native C API to agree."""

from __future__ import annotations

import argparse
import ctypes
import email.policy
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


def _is_native_library(path: PurePosixPath) -> bool:
    """Accept exactly the installed xTBloom library names on native platforms."""
    return (
        path.name.startswith("libxtbloom.so")
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


def inspect_wheel(wheel: Path, temporary_root: Path) -> None:
    """Compare every installed product-version representation in one wheel."""
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
        header = archive.read(header_name).decode("utf-8")
        string_match = VERSION_STRING_RE.search(header)
        components = dict(VERSION_COMPONENT_RE.findall(header))
        release_match = RELEASE_RE.match(metadata_version)
        if string_match is None or release_match is None or len(components) != 3:
            raise RuntimeError(f"{wheel} contains malformed version metadata")

        extracted = temporary_root / wheel.stem / PurePosixPath(library_name).name
        extracted.parent.mkdir(parents=True)
        extracted.write_bytes(archive.read(library_name))

    library = ctypes.CDLL(str(extracted))
    library.xtbloom_version_string.argtypes = []
    library.xtbloom_version_string.restype = ctypes.c_char_p
    native_value = library.xtbloom_version_string()
    if native_value is None:
        raise RuntimeError(f"{wheel} native version function returned NULL")
    native_version = native_value.decode("utf-8")
    header_version = string_match.group(1)
    release = release_match.groups()
    header_release = (
        components["MAJOR"],
        components["MINOR"],
        components["PATCH"],
    )
    if not metadata_version == header_version == native_version:
        raise RuntimeError(
            f"{wheel} version mismatch: metadata={metadata_version}, "
            f"header={header_version}, native={native_version}"
        )
    if release != header_release:
        raise RuntimeError(
            f"{wheel} numeric version mismatch: metadata={release}, "
            f"header={header_release}"
        )
    print(  # noqa: T201 - CI validation report
        f"{wheel.name}: metadata/header/native={metadata_version}; "
        f"release={'.'.join(release)}"
    )


def main() -> int:
    """Inspect every wheel passed on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="xtbloom-wheel-version-") as directory:
        root = Path(directory)
        for wheel in args.wheels:
            inspect_wheel(wheel, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
