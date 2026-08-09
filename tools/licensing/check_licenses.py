#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate source, installed, and archived xtbloom legal material.

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
REVIEWED_BUILD_REQUIREMENTS = {
    "scikit-build-core==1.0.3",
    "setuptools-scm==10.2.1",
    "vcs-versioning==2.2.4",
    "packaging==26.3",
    "pathspec==1.1.1",
    "setuptools==84.0.0",
    "exceptiongroup==1.3.1; python_version < '3.11'",
    "tomli==2.4.1; python_version < '3.11'",
    "typing-extensions==4.16.0; python_version < '3.11'",
}
EXCEPTION_FILE = "CUDA_MKL_LINKING_EXCEPTION"
IMPLIB_MANIFEST_PATH = "cmake/3rdparty/implib_manifest.json"
IMPLIB_VENDOR_PATH = "cmake/3rdparty/implib"
IMPLIB_REVISION = "6f4fc02ae058ef11848046af01a1a756f3229c29"
IMPLIB_TREE = "5fbe7e9f2c4efe0c2be4d2eed409e81f35458ba4"
ARRAY_API_COMPAT_LICENSE = "LICENSES/array-api-compat-MIT.txt"
MPL_LICENSE = "LICENSES/MPL-2.0.txt"
MPL_LICENSE_SHA256 = "fab3dd6bdab226f1c08630b1dd917e11fcb4ec5e1e020e2c16f83a0a13863e85"
OPEN_CHEMLIB_LICENSE = "LICENSES/openchemlib-BSD-3-Clause.txt"
OPEN_CHEMLIB_MANIFEST = "web/openchemlib_manifest.json"
OPEN_CHEMLIB_VERSION = "9.21.0"
OPEN_CHEMLIB_MODULE_URL = (
    "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/openchemlib.js"
)
OPEN_CHEMLIB_RESOURCES_URL = (
    "https://cdn.jsdelivr.net/npm/openchemlib@9.21.0/dist/resources.json"
)
WEB_LICENSE_FILES = (
    "LICENSES/3Dmol.js-BSD-3-Clause.txt",
    OPEN_CHEMLIB_LICENSE,
    "LICENSES/iobuffer-MIT.txt",
    "LICENSES/netcdfjs-MIT.txt",
    "LICENSES/pako-MIT.txt",
    "LICENSES/pako-Zlib.txt",
    "LICENSES/upng-js-MIT.txt",
)
WEB_SOURCE_FILES = (
    *WEB_LICENSE_FILES,
    "web/package.json",
    "web/package-lock.json",
    OPEN_CHEMLIB_MANIFEST,
)
SOURCE_FILES = (
    "LICENSE",
    EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT.txt",
    MPL_LICENSE,
    ARRAY_API_COMPAT_LICENSE,
    *WEB_SOURCE_FILES,
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
    MPL_LICENSE,
    ARRAY_API_COMPAT_LICENSE,
)
SDIST_ARCHIVE_SUFFIXES = (
    *WEB_SOURCE_FILES,
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
    "share/licenses/xtbloom/THIRD_PARTY_NOTICES.md",
    f"share/licenses/xtbloom/{EXCEPTION_FILE}",
    "share/licenses/xtbloom/provenance/manifest.json",
    "share/licenses/xtbloom/provenance/sto_manifest.json",
    "share/licenses/xtbloom/provenance/spin_manifest.json",
    "share/licenses/xtbloom/provenance/d4_manifest.json",
    "share/licenses/xtbloom/provenance/mctc_manifest.json",
    "share/licenses/xtbloom/provenance/implib_manifest.json",
    "share/licenses/xtbloom/third-party/MIT.txt",
    "share/licenses/xtbloom/third-party/MPL-2.0.txt",
    "share/licenses/xtbloom/third-party/array-api-compat-MIT.txt",
    "share/licenses/xtbloom/third-party/d4/d4.NOTICE",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE",
)
FORBIDDEN_ARCHIVE_PARTS = ("/build/", "/.cache/", "/.claude/", "/.ruff_cache/")
INSTALL_FILES = (
    "share/licenses/xtbloom/LICENSE",
    f"share/licenses/xtbloom/{EXCEPTION_FILE}",
    "share/licenses/xtbloom/THIRD_PARTY_NOTICES.md",
    "share/licenses/xtbloom/third-party/LGPL-3.0-or-later.txt",
    "share/licenses/xtbloom/third-party/Apache-2.0.txt",
    "share/licenses/xtbloom/third-party/MIT.txt",
    "share/licenses/xtbloom/third-party/MPL-2.0.txt",
    "share/licenses/xtbloom/third-party/array-api-compat-MIT.txt",
    "share/licenses/xtbloom/provenance/manifest.json",
    "share/licenses/xtbloom/provenance/sto_manifest.json",
    "share/licenses/xtbloom/provenance/spin_manifest.json",
    "share/licenses/xtbloom/provenance/d4_manifest.json",
    "share/licenses/xtbloom/provenance/mctc_manifest.json",
    "share/licenses/xtbloom/provenance/implib_manifest.json",
    "share/licenses/xtbloom/third-party/d4/d4.NOTICE",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE",
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
    "array-api-compat",
    "scikit-build-core 1.0.3",
    "setuptools-scm 10.2.1",
    "vcs-versioning 2.2.4",
    "packaging 26.3",
    "pathspec 1.1.1",
    "setuptools 84.0.0",
    "a4d7a05978ee37975c37743510c8991e2debce7ef83afb0a07c0c576fd4f16e8",
    "ae95427b7d3c14a6cf8bbfd4d901f6138ab64c99e20cbe8ea7d75cd26093f085",
    "4fa7dd82cf8c800df59c9a288c90299b1657ff1ecfc3f5cc00287c5dbf5e27a9",
    "b7c82f4102d389ee57dc66ccdb4f9b4bca3c40ba83b43f1f63d68ccd72db2580",
    "ed718fdee42170e128a8add6f23f53aa64dc7d9ab2de87d3083a691df881a809",
    "48ffea8a6a175c37a718c7ee2e5f508d0e500c2cf3371791f7948bccb2b44628",
    "94edc256424af38762eb31306eed28beb9f0efc50a8837492c9d6fd6004aed79",
    "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
    "17db5ecd524104a120e173814c90367a96a98d07c45b2e10c2f3919fff91bf5a",
    "a00ce642f577bf7f473932318056212bc4f8bfdf53128c78bbd5af0b9b20b189",
    "f4695c21257f0d9b537ec2692c941d02ee143b7cc1276941349a546573b2ef73",
    "51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670",
    "8b412432c6055b0b7d14c310000ae93352ed6754f70fa8f7c34141f91c4e3219",
    "a7a39a3bd276781e98394987d3a5701d0c4edffb633bb7a5144577f82c773598",
    "7c7e1a961a0b2f2472c1ac5b69affa0ae1132c39adcb67aba98568702b9cc23f",
    "0d85819802132122da43cb86656f8d1f8c6587d54ae7dcaf30e90533028b49fe",
    "dc983d19a509c94dba722ee6abd33940f7c05a89e243c47e907eb4db6f1a43e5",
    "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
    "076218e4f5aa18578418c7d04fad9ab581a16bb8",
    "Copyright (c) 2022 Consortium for Python Data API Standards",
    "3dmol@2.5.5",
    "iobuffer 5.4.0",
    "netcdfjs 3.0.0",
    "UPNG.js 2.1.0",
    "pako 2.2.0 and pako 1.0.11",
    "475e2213ac02fbf2d4a8c4fc287b570fc476da2fda9de3f5a72a2554b5716e71",
    "OpenChemLib 9.21.0",
    "36aec7791ac38e7fdc23a37ba07e19514eb1e5c9",
    "27d2b2fe2195ec0b159c3aa2cae3bc1464b41daf",
    "5978967b12e938208e8d36222370f88fd615a2b5ec83f02e435caab26f3f4cb3",
    "d2741130d5a5546aeebebc43eb3dac937881b04755fefe5925e4b228a56bee14",
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
    "xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION."
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
WEB_SITE_SOURCE_MAP = {
    "LICENSE": "LICENSE",
    EXCEPTION_FILE: EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
    "LICENSES/Apache-2.0.txt": "LICENSES/Apache-2.0.txt",
    "LICENSES/LGPL-3.0-or-later.txt": "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/MIT.txt": "LICENSES/MIT.txt",
    "LICENSES/MPL-2.0.txt": MPL_LICENSE,
    "LICENSES/array-api-compat-MIT.txt": ARRAY_API_COMPAT_LICENSE,
    **{path: path for path in WEB_LICENSE_FILES},
    "LICENSES/parameters/d4.NOTICE": "data/parameters/d4.NOTICE",
    "LICENSES/parameters/dftd4-COPYING": ("data/parameters/licenses/dftd4-COPYING"),
    "LICENSES/parameters/dftd4-COPYING.LESSER": (
        "data/parameters/licenses/dftd4-COPYING.LESSER"
    ),
    "LICENSES/parameters/mctc-lib-LICENSE": (
        "data/parameters/licenses/mctc-lib-LICENSE"
    ),
    "provenance/parameters/manifest.json": "data/parameters/manifest.json",
    "provenance/parameters/sto_manifest.json": "data/parameters/sto_manifest.json",
    "provenance/parameters/spin_manifest.json": "data/parameters/spin_manifest.json",
    "provenance/parameters/d4_manifest.json": "data/parameters/d4_manifest.json",
    "provenance/parameters/mctc_manifest.json": "data/parameters/mctc_manifest.json",
    "provenance/openchemlib_manifest.json": OPEN_CHEMLIB_MANIFEST,
}
WEB_SITE_RUNTIME_FILES = (
    "index.html",
    "style.css",
    "app.js",
    "app_helpers.js",
    "worker.js",
    "smiles_helpers.js",
    "smiles_worker.js",
    "xtbloom-mark.svg",
    "xtbloom_web.js",
    "xtbloom_web.wasm",
    "xtbloom_web.data",
    "vendor/3Dmol-min.js",
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
    array_compat = mandatory.get("array-api-compat")
    if array_compat is None or ">=1.15,<2" not in array_compat:
        raise LicenseCheckError(
            "array-api-compat must use the reviewed >=1.15,<2 runtime range"
        )
    if "scipy-openblas32" not in mandatory:
        raise LicenseCheckError("Linux CPU installs must require scipy-openblas32")
    openblas = mandatory["scipy-openblas32"]
    for token in (
        "scipy-openblas32==0.3.34.0.0",
        "sys_platform == 'linux'",
        "x86_64",
        "aarch64",
    ):
        if token not in openblas:
            raise LicenseCheckError(
                "scipy-openblas32 must use the reviewed exact version and cover Linux "
                "x86_64 and aarch64"
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


def _require_build_dependency_policy(build_system: object) -> None:
    """Require one reproducible, provenance-reviewed PEP 517 environment."""
    if not isinstance(build_system, dict):
        raise LicenseCheckError("pyproject build-system metadata must be a table")
    requirements = build_system.get("requires")
    if (
        not isinstance(requirements, list)
        or set(requirements) != REVIEWED_BUILD_REQUIREMENTS
    ):
        raise LicenseCheckError(
            "build-system.requires must equal the reviewed exact build dependency set"
        )
    if build_system.get("build-backend") != "scikit_build_core.build":
        raise LicenseCheckError("pyproject uses an unreviewed build backend")


def _require_version_metadata_policy(metadata: object) -> None:
    """Require strict tag-derived metadata with no usable fallback version."""
    if not isinstance(metadata, dict):
        raise LicenseCheckError("pyproject metadata must be a table")
    project = metadata.get("project")
    tool = metadata.get("tool")
    if not isinstance(project, dict) or not isinstance(tool, dict):
        raise LicenseCheckError("pyproject project/tool metadata must be tables")
    if project.get("dynamic") != ["version"] or "version" in project:
        raise LicenseCheckError("project version must be exclusively dynamic")
    if tool.get("dynamic-metadata") != [
        {
            "provider": {
                "path": "python/ci",
                "module": "xtbloom_version_provider",
            }
        }
    ]:
        raise LicenseCheckError("project uses an unreviewed dynamic version provider")
    expected_setuptools_scm = {
        "version_scheme": "only-version",
        "local_scheme": "no-local-version",
        "tag": {
            "regex": (
                r"^v(?P<version>(?:0|[1-9][0-9]*)\."
                r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))$"
            ),
        },
        "scm": {
            "git": {
                "pre_parse": "fail_on_shallow",
                "describe_command": [
                    "git",
                    "describe",
                    "--dirty",
                    "--tags",
                    "--long",
                    "--abbrev=40",
                    "--match",
                    "v*",
                ],
            }
        },
    }
    if tool.get("setuptools_scm") != expected_setuptools_scm:
        raise LicenseCheckError(
            "setuptools-scm must use the reviewed strict Git-tag policy"
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
    observed_mpl_sha256 = hashlib.sha256((root / MPL_LICENSE).read_bytes()).hexdigest()
    if observed_mpl_sha256 != MPL_LICENSE_SHA256:
        raise LicenseCheckError(
            f"{MPL_LICENSE} differs from the reviewed pathspec text"
        )

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
    _require_build_dependency_policy(metadata.get("build-system"))
    _require_version_metadata_policy(metadata)
    _require_exception_policy(root)

    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    if '"$<DEVICE_LINK:--cudadevrt=none>"' not in cmake:
        raise LicenseCheckError("CUDA device link must pass --cudadevrt=none")

    notice = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in NOTICE_TOKENS:
        if token not in notice:
            raise LicenseCheckError(f"THIRD_PARTY_NOTICES.md omits {token}")

    web_lock = json.loads((root / "web/package-lock.json").read_text(encoding="utf-8"))
    web_packages = web_lock.get("packages", {})
    expected_web_packages = {
        "node_modules/3dmol": (
            "2.5.5",
            "sha512-kqNHouGqq3YfW58174tdERvm0XYTmP0tavQKOqIw1ouc2OJ7epkXEFrtEkVXV0clBZT2Ze2xHRC/qxX0u0qCdw==",
        ),
        "node_modules/iobuffer": (
            "5.4.0",
            "sha512-DRebOWuqDvxunfkNJAlc3IzWIPD5xVxwUNbHr7xKB8E6aLJxIPfNX3CoMJghcFjpv6RWQsrcJbghtEwSPoJqMA==",
        ),
        "node_modules/netcdfjs": (
            "3.0.0",
            "sha512-LOvT8KkC308qtpUkcBPiCMBtii7ZQCN6LxcVheWgyUeZ6DQWcpSRFV9dcVXLj/2eHZ/bre9tV5HTH4Sf93vrFw==",
        ),
        "node_modules/pako": (
            "2.2.0",
            "sha512-zJq6RP/5q+TO2OpFV3FHzlPnFjmkb7Nc99a5SNjJE+uu/PkpChs+NIZSSzbBoD+6kjiISXjfYdwj1ZRQ81dz/w==",
        ),
        "node_modules/upng-js": (
            "2.1.0",
            "sha512-d3xzZzpMP64YkjP5pr8gNyvBt7dLk/uGI67EctzDuVp4lCZyVMo0aJO6l/VDlgbInJYDY6cnClLoBp29eKWI6g==",
        ),
        "node_modules/upng-js/node_modules/pako": (
            "1.0.11",
            "sha512-4hLB8Py4zZce5s4yd9XzopqwVv/yGNhV1Bl8NTmCq1763HeK2+EwVTv+leGeL13Dnh2wfbqowVPXCIO0z4taYw==",
        ),
    }
    for package_path, (version, integrity) in expected_web_packages.items():
        package = web_packages.get(package_path)
        if not isinstance(package, dict) or (
            package.get("version") != version or package.get("integrity") != integrity
        ):
            raise LicenseCheckError(
                f"web/package-lock.json has unreviewed {package_path} resolution"
            )

    web_license_tokens = {
        "LICENSES/3Dmol.js-BSD-3-Clause.txt": (
            "University of Pittsburgh",
            "Redistribution and use in source and binary forms",
        ),
        "LICENSES/iobuffer-MIT.txt": ("Copyright (c) 2015 Michaël Zasso",),
        "LICENSES/netcdfjs-MIT.txt": ("Copyright (c) 2016 cheminfo",),
        "LICENSES/upng-js-MIT.txt": ("Copyright (c) 2017 Photopea",),
        "LICENSES/pako-MIT.txt": (
            "Copyright (C) 2014-2017 by Vitaly Puzrin and Andrei Tuputcyn",
        ),
        "LICENSES/pako-Zlib.txt": (
            "Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler",
            "This notice may not be removed or altered",
        ),
        OPEN_CHEMLIB_LICENSE: (
            "Copyright (c) 2015-2017, cheminfo",
            "Redistribution and use in source and binary forms",
        ),
    }
    for relative, tokens in web_license_tokens.items():
        text = (root / relative).read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                raise LicenseCheckError(f"{relative} omits upstream text: {token}")

    openchemlib = json.loads((root / OPEN_CHEMLIB_MANIFEST).read_text(encoding="utf-8"))
    dependency = openchemlib.get("dependency", {})
    source = openchemlib.get("source", {})
    license_info = openchemlib.get("license", {})
    if (
        openchemlib.get("schema_version") != 1
        or dependency.get("npm_package") != "openchemlib"
        or dependency.get("version") != OPEN_CHEMLIB_VERSION
        or dependency.get("classification") != "runtime-provided browser dependency"
        or source.get("release_commit") != "36aec7791ac38e7fdc23a37ba07e19514eb1e5c9"
        or source.get("openchemlib_java_submodule_commit")
        != "27d2b2fe2195ec0b159c3aa2cae3bc1464b41daf"
        or license_info.get("spdx") != "BSD-3-Clause"
        or license_info.get("local_copy") != OPEN_CHEMLIB_LICENSE
        or license_info.get("sha256")
        != "38dc3aed3def8cc4dd15ac879daa4af9b0d71af86fef82611ca1752497c6f464"
    ):
        raise LicenseCheckError("OpenChemLib manifest has unreviewed provenance")
    if (
        hashlib.sha256((root / OPEN_CHEMLIB_LICENSE).read_bytes()).hexdigest()
        != (license_info["sha256"])
    ):
        raise LicenseCheckError(
            "OpenChemLib license differs from pinned upstream bytes"
        )

    artifacts = {
        artifact.get("url"): artifact
        for artifact in openchemlib.get("cdn_artifacts", [])
        if isinstance(artifact, dict)
    }
    expected_artifacts = {
        OPEN_CHEMLIB_MODULE_URL: (
            "5978967b12e938208e8d36222370f88fd615a2b5ec83f02e435caab26f3f4cb3",
            1097449,
        ),
        OPEN_CHEMLIB_RESOURCES_URL: (
            "d2741130d5a5546aeebebc43eb3dac937881b04755fefe5925e4b228a56bee14",
            1351963,
        ),
    }
    if set(artifacts) != set(expected_artifacts):
        raise LicenseCheckError("OpenChemLib manifest has unreviewed CDN URLs")
    for url, (digest, size) in expected_artifacts.items():
        artifact = artifacts[url]
        if (
            artifact.get("sha256") != digest
            or artifact.get("size_bytes") != size
            or artifact.get("redistributed_by_xtbloom") is not False
        ):
            raise LicenseCheckError(f"OpenChemLib manifest has unreviewed bytes: {url}")

    resource_payload = openchemlib.get("resource_payload", {})
    groups = resource_payload.get("groups", [])
    resource_paths = [
        path
        for group in groups
        if isinstance(group, dict)
        for path in group.get("paths", [])
    ]
    if (
        resource_payload.get("entry_count") != 35
        or len(resource_paths) != 35
        or len(set(resource_paths)) != 35
        or not any("toxpredictor" in path for path in resource_paths)
        or not any("druglikeness" in path for path in resource_paths)
        or not any("forcefield/mmff94" in path for path in resource_paths)
        or not any("/cod/" in path for path in resource_paths)
    ):
        raise LicenseCheckError("OpenChemLib resource inventory is incomplete")

    smiles_helpers = (root / "web/smiles_helpers.js").read_text(encoding="utf-8")
    smiles_worker = (root / "web/smiles_worker.js").read_text(encoding="utf-8")
    for token in (
        OPEN_CHEMLIB_VERSION,
        OPEN_CHEMLIB_MODULE_URL,
        OPEN_CHEMLIB_RESOURCES_URL,
    ):
        if token not in smiles_helpers:
            raise LicenseCheckError(
                f"SMILES helper omits pinned OpenChemLib token: {token}"
            )
    if "Resources.registerFromUrl(OPEN_CHEMLIB_RESOURCES_URL)" not in smiles_worker:
        raise LicenseCheckError(
            "SMILES worker does not register the pinned resources URL"
        )
    if re.search(r"openchemlib@(?:latest|[^\"'`]*\+esm)", smiles_helpers):
        raise LicenseCheckError("SMILES helper uses a floating/transformed CDN URL")

    array_compat_license = (root / ARRAY_API_COMPAT_LICENSE).read_text(encoding="utf-8")
    for token in (
        "MIT License",
        "Copyright (c) 2022 Consortium for Python Data API Standards",
        "Permission is hereby granted, free of charge",
    ):
        if token not in array_compat_license:
            raise LicenseCheckError(
                f"{ARRAY_API_COMPAT_LICENSE} omits required upstream text: {token}"
            )

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
    installed_mpl = prefix / "share/licenses/xtbloom/third-party/MPL-2.0.txt"
    if hashlib.sha256(installed_mpl.read_bytes()).hexdigest() != MPL_LICENSE_SHA256:
        raise LicenseCheckError("install tree contains modified MPL text")
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


def check_web_site(site: Path, source_root: Path | None = None) -> None:
    """Validate the legal payload conveyed beside the browser binaries."""
    _require_files(
        site,
        WEB_SITE_RUNTIME_FILES + tuple(WEB_SITE_SOURCE_MAP),
        "web site",
    )
    if (site / "libscipy_openblas.so").exists():
        raise LicenseCheckError(
            "web site contains the raw LAPACK side module; it must only be "
            "conveyed inside xtbloom_web.data"
        )
    # Pages uploads the complete site directory, so accepting arbitrary extra
    # files would allow an obsolete JS/WASM variant or unreviewed payload to be
    # redistributed even when every required file is present.
    expected_files = set(WEB_SITE_RUNTIME_FILES) | set(WEB_SITE_SOURCE_MAP)
    observed_files = {
        path.relative_to(site).as_posix() for path in site.rglob("*") if path.is_file()
    }
    unexpected_files = sorted(observed_files - expected_files)
    if unexpected_files:
        raise LicenseCheckError(
            "web site contains unexpected or orphaned files: "
            + ", ".join(unexpected_files)
        )
    index = (site / "index.html").read_text(encoding="utf-8")
    for token in (
        'href="LICENSE"',
        'href="THIRD_PARTY_NOTICES.md"',
        f'href="{EXCEPTION_FILE}"',
        "https://xtbloom.jinzhezeng.group",
        "https://github.com/jinzhezenggroup/xtbloom",
        'href="LICENSES/openchemlib-BSD-3-Clause.txt"',
    ):
        if token not in index:
            raise LicenseCheckError(f"web site index does not expose {token}")

    if source_root is not None:
        for site_relative, source_relative in WEB_SITE_SOURCE_MAP.items():
            if (site / site_relative).read_bytes() != (
                source_root / source_relative
            ).read_bytes():
                raise LicenseCheckError(
                    f"web site legal file differs from source: {site_relative}"
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
        "share/licenses/xtbloom/provenance/implib_manifest.json"
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
    mpl_names = {
        name
        for name in names
        if name.endswith(f"/{MPL_LICENSE}") or name.endswith("/third-party/MPL-2.0.txt")
    }
    mpl_payloads = _read_archive_members(path, mpl_names)
    for name, payload in mpl_payloads.items():
        if hashlib.sha256(payload).hexdigest() != MPL_LICENSE_SHA256:
            raise LicenseCheckError(f"{path} contains modified MPL text at {name}")
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
    """Validate source, install, and archive licensing payloads from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--install-prefix", type=Path)
    parser.add_argument("--web-site", type=Path)
    parser.add_argument("archives", nargs="*", type=Path)
    args = parser.parse_args()

    try:
        check_source(args.source_root.resolve())
        if args.install_prefix is not None:
            check_install(args.install_prefix.resolve())
        if args.web_site is not None:
            check_web_site(args.web_site.resolve(), args.source_root.resolve())
        for archive in args.archives:
            check_archive(archive.resolve())
    except (LicenseCheckError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"license check failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
