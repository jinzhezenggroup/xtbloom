#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate source, installed, and archived gpuxtb legal material.

The release workflows invoke this script at each distribution boundary. A
single validator keeps the required filenames and provenance invariants from
drifting between CMake installs, source distributions, and Python wheels.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path

import tomllib

PROJECT_LICENSE = "GPL-3.0-or-later"
SOURCE_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "data/parameters/d4.NOTICE",
    "data/parameters/licenses/dftd4-COPYING",
    "data/parameters/licenses/dftd4-COPYING.LESSER",
    "data/parameters/licenses/mctc-lib-LICENSE",
    "data/parameters/manifest.json",
    "data/parameters/spin_manifest.json",
    "data/parameters/d4_manifest.json",
    "data/parameters/mctc_manifest.json",
)
COMMON_ARCHIVE_SUFFIXES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
)
SDIST_ARCHIVE_SUFFIXES = (
    "data/parameters/d4.NOTICE",
    "data/parameters/licenses/dftd4-COPYING",
    "data/parameters/licenses/dftd4-COPYING.LESSER",
    "data/parameters/licenses/mctc-lib-LICENSE",
    "data/parameters/manifest.json",
    "data/parameters/spin_manifest.json",
    "data/parameters/d4_manifest.json",
    "data/parameters/mctc_manifest.json",
)
WHEEL_ARCHIVE_SUFFIXES = (
    "share/licenses/gpuxtb/THIRD_PARTY_NOTICES.md",
    "share/licenses/gpuxtb/provenance/manifest.json",
    "share/licenses/gpuxtb/provenance/spin_manifest.json",
    "share/licenses/gpuxtb/provenance/d4_manifest.json",
    "share/licenses/gpuxtb/provenance/mctc_manifest.json",
    "share/licenses/gpuxtb/third-party/d4/d4.NOTICE",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE",
)
FORBIDDEN_ARCHIVE_PARTS = ("/build/", "/.cache/", "/.claude/", "/.ruff_cache/")
INSTALL_FILES = (
    "share/licenses/gpuxtb/LICENSE",
    "share/licenses/gpuxtb/THIRD_PARTY_NOTICES.md",
    "share/licenses/gpuxtb/third-party/LGPL-3.0-or-later.txt",
    "share/licenses/gpuxtb/third-party/Apache-2.0.txt",
    "share/licenses/gpuxtb/provenance/manifest.json",
    "share/licenses/gpuxtb/provenance/spin_manifest.json",
    "share/licenses/gpuxtb/provenance/d4_manifest.json",
    "share/licenses/gpuxtb/provenance/mctc_manifest.json",
    "share/licenses/gpuxtb/third-party/d4/d4.NOTICE",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE",
)
SPDX_FILES = {
    "data/parameters/gfn2.hpp": "LGPL-3.0-or-later",
    "data/parameters/d4.hpp": "LGPL-3.0-or-later",
    "src/model/gfn2/basis.cpp": "GPL-3.0-or-later",
    "src/model/gfn2/spin.cpp": "GPL-3.0-or-later",
    "src/model/gfn2/coordination.cpp": "GPL-3.0-or-later",
    "src/model/gfn2/h0.cpp": "GPL-3.0-or-later",
}
NOTICE_TOKENS = (
    "fa8a4416e8fe093d0075bc10ac875494c2a449a9",
    "6e1f59c3f39d919a2dbef0601d2576727c8b30e8",
    "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d",
    "edcfbbe39d411edc225e27315fbda3a204ddb023",
    "9ab8ca565e0f71d967587e0bca2015f7d689f19f",
    "No LAMMPS source code",
)


class LicenseCheckError(RuntimeError):
    """Report incomplete or internally inconsistent legal material."""


def _require_files(root: Path, relative_paths: tuple[str, ...], context: str) -> None:
    missing = [path for path in relative_paths if not (root / path).is_file()]
    if missing:
        raise LicenseCheckError(f"{context} is missing: {', '.join(missing)}")


def check_source(root: Path) -> None:
    """Validate project metadata, provenance, and derived-file SPDX tags."""

    _require_files(root, SOURCE_FILES, "source tree")
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if (
        "GNU GENERAL PUBLIC LICENSE" not in license_text
        or "Version 3" not in license_text
    ):
        raise LicenseCheckError("LICENSE is not the complete GNU GPL version 3 text")

    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata.get("project", {})
    if project.get("license") != PROJECT_LICENSE:
        raise LicenseCheckError("pyproject project.license must be GPL-3.0-or-later")
    declared = set(project.get("license-files", ()))
    for required in ("LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/*.txt"):
        if required not in declared:
            raise LicenseCheckError(f"pyproject license-files omits {required}")

    notice = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in NOTICE_TOKENS:
        if token not in notice:
            raise LicenseCheckError(f"THIRD_PARTY_NOTICES.md omits {token}")

    gfn2 = json.loads(
        (root / "data/parameters/manifest.json").read_text(encoding="utf-8")
    )
    spin = json.loads(
        (root / "data/parameters/spin_manifest.json").read_text(encoding="utf-8")
    )
    d4 = json.loads(
        (root / "data/parameters/d4_manifest.json").read_text(encoding="utf-8")
    )
    mctc = json.loads(
        (root / "data/parameters/mctc_manifest.json").read_text(encoding="utf-8")
    )
    if gfn2["source"]["license"]["spdx"] != "LGPL-3.0-or-later":
        raise LicenseCheckError("GFN2 parameter manifest has the wrong SPDX license")
    if spin["source"]["license"] != "LGPL-3.0-or-later":
        raise LicenseCheckError("spin manifest has the wrong SPDX license")
    if d4["license"] != "LGPL-3.0-or-later":
        raise LicenseCheckError("D4 manifest has the wrong SPDX license")
    if (
        mctc["license"] != "Apache-2.0"
        or mctc["revision"] != "e9de066d89f250d1cfb6de3a33f0c27c0e2f855d"
    ):
        raise LicenseCheckError("mctc-lib manifest has the wrong license or revision")
    expected_mctc_sources = {
        "src/mctc/data/atomicrad.f90": (
            "f72b137328f24dc2abb0b709f8803be0cb616fef",
            "2c25924d89215c1810e61a1f38f67781e5083d15678de4bf326a071bc27d827c",
        ),
        "src/mctc/data/covrad.f90": (
            "b3cb9cefd702169d6be662eb438932525990a1ac",
            "fdbd599664a7f113633d96110531d810fdc8e54b6db26d4f584120d9c7cec314",
        ),
        "src/mctc/io/constants.f90": (
            "2fd35c66ce80a47aa88d12952f0cce2886cd753f",
            "6ee2b599fc2d338f1a7e07b5b46e984e2612b4cfc58110d1203ecf19ebde2385",
        ),
        "src/mctc/ncoord/dexp.f90": (
            "307e84898387fcddf77f54daecf24b6ce28a27b1",
            "c35fc4694621dd86c016cc4496d87e43ed32bd5539d2700aea5b42cb698396dc",
        ),
        "src/mctc/ncoord/erf/en.f90": (
            "80d2ea15a262731c395b62d388601520a9db8547",
            "5435fd884bc28418f1917e5b6a47e685c4f4ae4fe4ddab55ed7eb1abf739339e",
        ),
    }
    observed_mctc_sources = {
        entry.get("path"): (entry.get("git_blob"), entry.get("sha256"))
        for entry in mctc.get("consumers", ())
    }
    if observed_mctc_sources != expected_mctc_sources:
        raise LicenseCheckError("mctc-lib manifest has incomplete source coverage")

    for relative, identifier in SPDX_FILES.items():
        prefix = (root / relative).read_text(encoding="utf-8")[:4096]
        if f"SPDX-License-Identifier: {identifier}" not in prefix:
            raise LicenseCheckError(f"{relative} omits SPDX identifier {identifier}")


def check_install(prefix: Path) -> None:
    """Validate the legal payload installed by CMake."""

    _require_files(prefix, INSTALL_FILES, "install tree")


def _archive_names(path: Path) -> set[str]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            return set(archive.getnames())
    raise LicenseCheckError(f"unsupported distribution archive: {path}")


def check_archive(path: Path) -> None:
    """Require every distribution archive to retain the common legal set."""

    names = _archive_names(path)
    required = COMMON_ARCHIVE_SUFFIXES + (
        WHEEL_ARCHIVE_SUFFIXES if path.suffix == ".whl" else SDIST_ARCHIVE_SUFFIXES
    )
    missing = [
        suffix
        for suffix in required
        if not any(name == suffix or name.endswith(f"/{suffix}") for name in names)
    ]
    if missing:
        raise LicenseCheckError(
            f"{path} is missing archived legal files: {', '.join(missing)}"
        )
    leaked = sorted(
        name
        for name in names
        if any(component in f"/{name}" for component in FORBIDDEN_ARCHIVE_PARTS)
    )
    if leaked:
        raise LicenseCheckError(
            f"{path} contains excluded local artifacts: {leaked[0]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()

    try:
        check_source(args.source_root.resolve())
        if args.install_prefix is not None:
            check_install(args.install_prefix.resolve())
        for archive in args.archives:
            check_archive(archive.resolve())
    except (LicenseCheckError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"license check failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
