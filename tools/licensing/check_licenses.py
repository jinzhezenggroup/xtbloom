#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate source, installed, and archived gpuxtb legal material.

The release workflows invoke this script at each distribution boundary. A
single validator keeps the required filenames and provenance invariants from
drifting between CMake installs, source distributions, and Python wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import tarfile
import zipfile
from pathlib import Path

import tomllib

PROJECT_LICENSE = "GPL-3.0-or-later"
EXCEPTION_FILE = "CUDA_MKL_LINKING_EXCEPTION"
IMPLIB_MANIFEST_PATH = "cmake/3rdparty/implib_manifest.json"
IMPLIB_VENDOR_PATH = "cmake/3rdparty/implib"
IMPLIB_REVISION = "6f4fc02ae058ef11848046af01a1a756f3229c29"
IMPLIB_TREE = "5fbe7e9f2c4efe0c2be4d2eed409e81f35458ba4"
SOURCE_FILES = (
    "LICENSE",
    EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT.txt",
    "data/parameters/d4.NOTICE",
    "data/parameters/tblite_sto.hpp",
    "data/parameters/tblite_spin.hpp",
    "data/parameters/licenses/dftd4-COPYING",
    "data/parameters/licenses/dftd4-COPYING.LESSER",
    "data/parameters/licenses/mctc-lib-LICENSE",
    "data/parameters/manifest.json",
    "data/parameters/sto_manifest.json",
    "data/parameters/spin_manifest.json",
    "data/parameters/d4_manifest.json",
    "data/parameters/mctc_manifest.json",
    IMPLIB_MANIFEST_PATH,
)
COMMON_ARCHIVE_SUFFIXES = (
    "LICENSE",
    EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT.txt",
)
SDIST_ARCHIVE_SUFFIXES = (
    "data/parameters/d4.NOTICE",
    "data/parameters/tblite_sto.hpp",
    "data/parameters/tblite_spin.hpp",
    "data/parameters/licenses/dftd4-COPYING",
    "data/parameters/licenses/dftd4-COPYING.LESSER",
    "data/parameters/licenses/mctc-lib-LICENSE",
    "data/parameters/manifest.json",
    "data/parameters/sto_manifest.json",
    "data/parameters/spin_manifest.json",
    "data/parameters/d4_manifest.json",
    "data/parameters/mctc_manifest.json",
    IMPLIB_MANIFEST_PATH,
)
WHEEL_ARCHIVE_SUFFIXES = (
    "share/licenses/gpuxtb/THIRD_PARTY_NOTICES.md",
    f"share/licenses/gpuxtb/{EXCEPTION_FILE}",
    "share/licenses/gpuxtb/provenance/manifest.json",
    "share/licenses/gpuxtb/provenance/sto_manifest.json",
    "share/licenses/gpuxtb/provenance/spin_manifest.json",
    "share/licenses/gpuxtb/provenance/d4_manifest.json",
    "share/licenses/gpuxtb/provenance/mctc_manifest.json",
    "share/licenses/gpuxtb/provenance/implib_manifest.json",
    "share/licenses/gpuxtb/third-party/MIT.txt",
    "share/licenses/gpuxtb/third-party/d4/d4.NOTICE",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE",
)
FORBIDDEN_ARCHIVE_PARTS = ("/build/", "/.cache/", "/.claude/", "/.ruff_cache/")
INSTALL_FILES = (
    "share/licenses/gpuxtb/LICENSE",
    f"share/licenses/gpuxtb/{EXCEPTION_FILE}",
    "share/licenses/gpuxtb/THIRD_PARTY_NOTICES.md",
    "share/licenses/gpuxtb/third-party/LGPL-3.0-or-later.txt",
    "share/licenses/gpuxtb/third-party/Apache-2.0.txt",
    "share/licenses/gpuxtb/third-party/MIT.txt",
    "share/licenses/gpuxtb/provenance/manifest.json",
    "share/licenses/gpuxtb/provenance/sto_manifest.json",
    "share/licenses/gpuxtb/provenance/spin_manifest.json",
    "share/licenses/gpuxtb/provenance/d4_manifest.json",
    "share/licenses/gpuxtb/provenance/mctc_manifest.json",
    "share/licenses/gpuxtb/provenance/implib_manifest.json",
    "share/licenses/gpuxtb/third-party/d4/d4.NOTICE",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING",
    "share/licenses/gpuxtb/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/gpuxtb/third-party/d4/mctc-lib-LICENSE",
)
SPDX_FILES = {
    "data/parameters/gfn2.hpp": "LGPL-3.0-or-later",
    "data/parameters/d4.hpp": "LGPL-3.0-or-later",
    "data/parameters/tblite_sto.hpp": "LGPL-3.0-or-later",
    "data/parameters/tblite_spin.hpp": "LGPL-3.0-or-later",
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
    "6f4fc02ae058ef11848046af01a1a756f3229c29",
    "No LAMMPS source code",
    "scipy-openblas32",
    EXCEPTION_FILE,
)
EXCEPTION_TOKENS = (
    "Copyright (C) 2026 Jinzhe Zeng",
    "section 7",
    "libcuda",
    "libcudart",
    "cuBLAS",
    "cuSOLVER",
    "cuSPARSE",
    "nvJitLink",
    "libdevice",
    "libmkl_rt",
    "LP64 CBLAS and LAPACKE",
    "does not apply to third-party material",
)
EXCEPTION_NOTICE = (
    "gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION."
)
NVIDIA_DEPENDENCIES = {
    "nvidia-cublas-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-nvjitlink-cu12",
}
SOURCE_NOTICE_SUFFIXES = {".c", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
FORBIDDEN_VENDOR_LIBRARY_RE = re.compile(
    r"(?:^|/)(?:lib(?:cuda|cudart|cublas|cusolver|cusparse|nvjitlink|mkl)[^/]*)"
    r"(?:\.a|\.so(?:\.[0-9]+)*)$",
    re.IGNORECASE,
)


class LicenseCheckError(RuntimeError):
    """Report incomplete or internally inconsistent legal material."""


def _require_files(root: Path, relative_paths: tuple[str, ...], context: str) -> None:
    missing = [path for path in relative_paths if not (root / path).is_file()]
    if missing:
        raise LicenseCheckError(f"{context} is missing: {', '.join(missing)}")


def _require_exception_policy(root: Path) -> None:
    """Validate the grant and source notices required by GPLv3 section 7."""

    exception = (root / EXCEPTION_FILE).read_text(encoding="utf-8")
    for token in EXCEPTION_TOKENS:
        if token not in exception:
            raise LicenseCheckError(f"{EXCEPTION_FILE} omits {token}")
    if "cudadevrt" in exception:
        raise LicenseCheckError(
            f"{EXCEPTION_FILE} must not cover cudadevrt without renewed review"
        )

    for directory in (root / "src", root / "include"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_NOTICE_SUFFIXES:
                continue
            prefix = path.read_text(encoding="utf-8")[:4096]
            if EXCEPTION_NOTICE not in prefix:
                relative = path.relative_to(root).as_posix()
                raise LicenseCheckError(
                    f"{relative} omits the additional-permission notice"
                )


def _requirement_name(requirement: str) -> str:
    """Return a normalized distribution name from a PEP 508 requirement."""

    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    if match is None:
        raise LicenseCheckError(f"invalid dependency requirement: {requirement}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _require_dependency_policy(project: object) -> None:
    """Keep proprietary providers out of the mandatory Python dependency set."""

    if not isinstance(project, dict):
        raise LicenseCheckError("pyproject project metadata must be a table")
    dependencies = project.get("dependencies")
    extras = project.get("optional-dependencies")
    if not isinstance(dependencies, list) or not isinstance(extras, dict):
        raise LicenseCheckError("pyproject dependency metadata is incomplete")

    mandatory = {_requirement_name(item): item for item in dependencies}
    if "scipy-openblas32" not in mandatory:
        raise LicenseCheckError("Linux CPU installs must require scipy-openblas32")
    openblas = mandatory["scipy-openblas32"]
    for token in (
        "scipy-openblas32>=0.3.34.0.0",
        "sys_platform == 'linux'",
        "x86_64",
        "aarch64",
    ):
        if token not in openblas:
            raise LicenseCheckError(
                "scipy-openblas32 must use the reviewed minimum and cover Linux x86_64 and aarch64"
            )
    if "mkl" in mandatory:
        raise LicenseCheckError("mkl must not be a mandatory Python dependency")
    unexpected_nvidia = NVIDIA_DEPENDENCIES.intersection(mandatory)
    if unexpected_nvidia:
        raise LicenseCheckError("NVIDIA packages must be confined to the cuda12 extra")

    cuda_requirements = extras.get("cuda12")
    if not isinstance(cuda_requirements, list):
        raise LicenseCheckError("pyproject must define the cuda12 optional dependency")
    cuda_names = {_requirement_name(item) for item in cuda_requirements}
    if cuda_names != NVIDIA_DEPENDENCIES:
        raise LicenseCheckError(
            "cuda12 must contain the reviewed NVIDIA dependency set"
        )
    for extra, requirements in extras.items():
        if extra == "cuda12" or not isinstance(requirements, list):
            continue
        names = {_requirement_name(item) for item in requirements}
        if names.intersection(NVIDIA_DEPENDENCIES) or "mkl" in names:
            raise LicenseCheckError(
                f"proprietary providers must not be included in the {extra} extra"
            )


def _find_bundled_vendor_libraries(names: set[str]) -> list[str]:
    """Return separately packaged CUDA/MKL library files in an artifact."""

    return sorted(name for name in names if FORBIDDEN_VENDOR_LIBRARY_RE.search(name))


def _git_object_id(kind: str, data: bytes) -> str:
    """Return the Git SHA-1 object ID for canonical object bytes."""

    header = f"{kind} {len(data)}\0".encode()
    # SHA-1 is part of the pinned Git object format here, not a security check.
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _git_tree_id(entries: dict[str, tuple[str, str]]) -> str:
    """Reconstruct a Git tree ID from relative paths and (mode, blob) pairs."""

    root: dict[str, object] = {}
    for path, leaf in entries.items():
        node = root
        parts = path.split("/")
        for component in parts[:-1]:
            child = node.setdefault(component, {})
            if not isinstance(child, dict):
                raise LicenseCheckError(f"implib manifest path collision at {path}")
            node = child
        if parts[-1] in node:
            raise LicenseCheckError(f"implib manifest duplicates {path}")
        node[parts[-1]] = leaf

    def digest_tree(node: dict[str, object]) -> str:
        records: list[tuple[bytes, bytes]] = []
        for name, value in node.items():
            encoded_name = name.encode()
            if isinstance(value, dict):
                mode = "40000"
                object_id = digest_tree(value)
                sort_key = encoded_name + b"/"
            else:
                mode, object_id = value
                sort_key = encoded_name
            record = (
                f"{mode} ".encode() + encoded_name + b"\0" + bytes.fromhex(object_id)
            )
            records.append((sort_key, record))
        payload = b"".join(record for _key, record in sorted(records))
        return _git_object_id("tree", payload)

    return digest_tree(root)


def _check_implib_manifest(manifest: object) -> dict[str, tuple[str, str, str]]:
    """Validate pinned implib metadata and return its declared file mapping."""

    if not isinstance(manifest, dict):
        raise LicenseCheckError("implib manifest root must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("license") != "MIT"
        or manifest.get("repository") != "https://github.com/deepmodeling/deepmd-kit"
        or manifest.get("revision") != IMPLIB_REVISION
        or manifest.get("source_path") != "source/3rdparty/implib"
        or manifest.get("tree") != IMPLIB_TREE
        or manifest.get("upstream_repository") != "https://github.com/yugr/Implib.so"
    ):
        raise LicenseCheckError("implib manifest has incorrect pinned provenance")

    declared: dict[str, tuple[str, str, str]] = {}
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 24:
        raise LicenseCheckError("implib manifest must describe exactly 24 files")
    for entry in files:
        if not isinstance(entry, dict):
            raise LicenseCheckError("implib manifest contains a non-object file entry")
        path = entry.get("path")
        mode = entry.get("mode")
        blob = entry.get("git_blob")
        sha256 = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or mode not in ("100644", "100755")
            or not isinstance(blob, str)
            or len(blob) != 40
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise LicenseCheckError("implib manifest contains an invalid file entry")
        if path in declared:
            raise LicenseCheckError(f"implib manifest duplicates {path}")
        declared[path] = (mode, blob, sha256)

    declared_tree = _git_tree_id(
        {path: (mode, blob) for path, (mode, blob, _sha256) in declared.items()}
    )
    if declared_tree != IMPLIB_TREE:
        raise LicenseCheckError(
            "implib manifest file entries do not match the pinned tree"
        )
    return declared


def _check_implib_provenance(root: Path) -> None:
    """Verify the vendored implib tree is exactly the pinned DeepMD copy."""

    manifest = json.loads((root / IMPLIB_MANIFEST_PATH).read_text(encoding="utf-8"))
    declared = _check_implib_manifest(manifest)

    vendor_root = root / IMPLIB_VENDOR_PATH
    observed_paths = {
        path.relative_to(vendor_root).as_posix()
        for path in vendor_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_paths != set(declared):
        missing = sorted(set(declared) - observed_paths)
        unexpected = sorted(observed_paths - set(declared))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise LicenseCheckError(
            "implib vendored file set differs: " + "; ".join(details)
        )

    observed_tree: dict[str, tuple[str, str]] = {}
    for relative, (expected_mode, expected_blob, expected_sha256) in declared.items():
        path = vendor_root / relative
        if path.is_symlink():
            raise LicenseCheckError(
                f"implib vendored file must not be a symlink: {relative}"
            )
        data = path.read_bytes()
        observed_mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        observed_blob = _git_object_id("blob", data)
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if (
            observed_mode != expected_mode
            or observed_blob != expected_blob
            or observed_sha256 != expected_sha256
        ):
            raise LicenseCheckError(
                f"implib vendored file differs from pinned bytes: {relative}"
            )
        observed_tree[relative] = (observed_mode, observed_blob)
    if _git_tree_id(observed_tree) != IMPLIB_TREE:
        raise LicenseCheckError("implib vendored tree does not match the pinned tree")


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
    for required in (
        "LICENSE",
        EXCEPTION_FILE,
        "THIRD_PARTY_NOTICES.md",
        "LICENSES/*.txt",
    ):
        if required not in declared:
            raise LicenseCheckError(f"pyproject license-files omits {required}")
    _require_dependency_policy(project)
    _require_exception_policy(root)

    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    if '"$<DEVICE_LINK:--cudadevrt=none>"' not in cmake:
        raise LicenseCheckError("CUDA device link must pass --cudadevrt=none")

    notice = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in NOTICE_TOKENS:
        if token not in notice:
            raise LicenseCheckError(f"THIRD_PARTY_NOTICES.md omits {token}")

    _check_implib_provenance(root)

    gfn2 = json.loads(
        (root / "data/parameters/manifest.json").read_text(encoding="utf-8")
    )
    spin = json.loads(
        (root / "data/parameters/spin_manifest.json").read_text(encoding="utf-8")
    )
    sto = json.loads(
        (root / "data/parameters/sto_manifest.json").read_text(encoding="utf-8")
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
    if (
        sto["source"]["license"] != "LGPL-3.0-or-later"
        or sto["source"]["revision"] != "fa8a4416e8fe093d0075bc10ac875494c2a449a9"
        or sto["source"]["sha256"]
        != "8a3df2db076469b0e22c02af9dfadf9880932fc241b82d5802ebb268d002773c"
        or sto["consumer"] != "data/parameters/tblite_sto.hpp"
    ):
        raise LicenseCheckError("STO manifest has incorrect LGPL provenance")
    if spin["consumer"] != "data/parameters/tblite_spin.hpp":
        raise LicenseCheckError("spin manifest must identify the LGPL data header")
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
    bundled = _find_bundled_vendor_libraries(
        {
            path.relative_to(prefix).as_posix()
            for path in prefix.rglob("*")
            if path.is_file()
        }
    )
    if bundled:
        raise LicenseCheckError(
            "install tree bundles a CUDA/MKL provider library: " + bundled[0]
        )


def _archive_names(path: Path) -> set[str]:
    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {info.filename for info in archive.infolist() if not info.is_dir()}
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            return {info.name for info in archive.getmembers() if info.isfile()}
    raise LicenseCheckError(f"unsupported distribution archive: {path}")


def _read_archive_members(path: Path, names: set[str]) -> dict[str, bytes]:
    """Read only selected legal/provenance payloads, not a wheel's large DSO."""

    if path.suffix == ".whl" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in names}
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            payloads: dict[str, bytes] = {}
            for name in names:
                info = archive.getmember(name)
                extracted = archive.extractfile(info)
                if extracted is None:
                    raise LicenseCheckError(f"cannot read archived file: {name}")
                payloads[name] = extracted.read()
            return payloads
    raise LicenseCheckError(f"unsupported distribution archive: {path}")


def _find_archive_name(names: set[str], suffix: str) -> str:
    matches = [name for name in names if name == suffix or name.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise LicenseCheckError(
            f"archive must contain exactly one {suffix}; found {len(matches)}"
        )
    return matches[0]


def _check_archived_implib(path: Path, names: set[str], wheel: bool) -> None:
    """Validate the installed manifest and the complete sdist vendor payload."""

    manifest_suffix = (
        "share/licenses/gpuxtb/provenance/implib_manifest.json"
        if wheel
        else IMPLIB_MANIFEST_PATH
    )
    manifest_name = _find_archive_name(names, manifest_suffix)
    manifest_bytes = _read_archive_members(path, {manifest_name})[manifest_name]
    declared = _check_implib_manifest(json.loads(manifest_bytes.decode("utf-8")))
    if wheel:
        return

    archive_root = manifest_name[: -len(IMPLIB_MANIFEST_PATH)]
    vendor_prefix = archive_root + IMPLIB_VENDOR_PATH + "/"
    archived_vendor = {
        name.removeprefix(vendor_prefix): name
        for name in names
        if name.startswith(vendor_prefix)
    }
    if set(archived_vendor) != set(declared):
        missing = sorted(set(declared) - set(archived_vendor))
        unexpected = sorted(set(archived_vendor) - set(declared))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise LicenseCheckError(
            "sdist implib vendored file set differs: " + "; ".join(details)
        )
    vendor_payloads = _read_archive_members(path, set(archived_vendor.values()))
    for relative, (_mode, expected_blob, expected_sha256) in declared.items():
        data = vendor_payloads[archived_vendor[relative]]
        if (
            _git_object_id("blob", data) != expected_blob
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise LicenseCheckError(
                f"sdist implib vendored file differs from pinned bytes: {relative}"
            )


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
    bundled = _find_bundled_vendor_libraries(names)
    if bundled:
        raise LicenseCheckError(
            f"{path} bundles a CUDA/MKL provider library: {bundled[0]}"
        )
    _check_archived_implib(path, names, wheel=path.suffix == ".whl")
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
