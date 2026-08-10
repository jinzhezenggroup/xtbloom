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
import os
import re
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePath, PurePosixPath

import tomllib

PROJECT_LICENSE = "GPL-3.0-or-later"
REVIEWED_BUILD_REQUIREMENTS = {
    "scikit-build-core>=1.0.3",
    "setuptools-scm>=10.2.1",
}
EXCEPTION_FILE = "CUDA_MKL_LINKING_EXCEPTION"
IMPLIB_MANIFEST_PATH = "cmake/3rdparty/implib_manifest.json"
IMPLIB_VENDOR_PATH = "cmake/3rdparty/implib"
IMPLIB_REVISION = "6f4fc02ae058ef11848046af01a1a756f3229c29"
IMPLIB_TREE = "5fbe7e9f2c4efe0c2be4d2eed409e81f35458ba4"
TORCH_STABLE_MANIFEST_PATH = "cmake/3rdparty/torch-stable/manifest.json"
TORCH_STABLE_VENDOR_PATH = "cmake/3rdparty/torch-stable"
TORCH_STABLE_INCLUDE_SUBDIR = "include"
TORCH_STABLE_REVISION = "2.12.1"
TORCH_STABLE_TREE = "e2df0197562bc2b0f55ee910d9899ecaac465e78"
EIGEN_MANIFEST_PATH = "cmake/3rdparty/eigen_manifest.json"
EIGEN_LEGAL_ROOT = "LICENSES/eigen"
EIGEN_VERSION = "5.0.1"
EIGEN_REVISION = "bc3b39870ecb690a623a3f49149a358b95c5781d"
EIGEN_ARCHIVE_URL = (
    "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz"
)
EIGEN_ARCHIVE_SHA256 = (
    "e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec"
)
EIGEN_ARCHIVE_SIZE = 2_967_272
EIGEN_ACQUISITION = {
    "method": "CMake FetchContent for WebAssembly builds only",
    "offline_archive_cache_variable": "XTBLOOM_WEB_EIGEN_ARCHIVE",
    "policy": (
        "The exact official archive is SHA-256 verified before extraction. "
        "Native builds, installs, sdists, and wheels do not fetch or bundle Eigen."
    ),
}
EIGEN_DISTRIBUTION = (
    "Eigen is a downloaded WebAssembly build input. The repository and sdist "
    "retain provenance and exact legal records but not the header tree or archive. "
    "Pages carry compiled code, the legal records, provenance, and an exact source "
    "URL/hash. Native installs and Python wheels exclude all Eigen material."
)
EIGEN_LICENSE_RECORDS = (
    (
        "COPYING.MPL2",
        f"{EIGEN_LEGAL_ROOT}/COPYING.MPL2",
        "MPL-2.0",
        "Primary license for the Eigen source tree.",
        16727,
        "66a3107d5ad6a058aab753eaac2047ccb2ed0e39465dd0fe5844da3e300d5172",
    ),
    (
        "COPYING.BSD",
        f"{EIGEN_LEGAL_ROOT}/COPYING.BSD",
        "BSD-3-Clause",
        "BSD and other permissive files identified by upstream.",
        1517,
        "51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb",
    ),
    (
        "COPYING.APACHE",
        f"{EIGEN_LEGAL_ROOT}/COPYING.APACHE",
        "Apache-2.0",
        "Apache-licensed files identified by upstream.",
        11362,
        "03379001a7b12a2ec997a25554247d985270b353c10d5bafee9ac8d6519820b7",
    ),
    (
        "COPYING.MINPACK",
        f"{EIGEN_LEGAL_ROOT}/COPYING.MINPACK",
        "Minpack",
        "Retained upstream legal record; unsupported/ contains the "
        "MINPACK-derived sources and is not compiled by xTBloom.",
        2193,
        "c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0",
    ),
    (
        "COPYING.README",
        f"{EIGEN_LEGAL_ROOT}/COPYING.README",
        "NOASSERTION",
        "Upstream explanation of the accompanying license records.",
        264,
        "db640ff2bd90c6abd6a4d3fbb351e0ee4d555417cf840492054d1cbb2ea85644",
    ),
)
EIGEN_EMBEDDED_NOTICE_RECORDS = (
    (
        "Eigen/src/Core/arch/Default/Half.h",
        f"{EIGEN_LEGAL_ROOT}/notices/Half.h",
        47293,
        "6aef8811305cb4ed25860e3e403f885a11b340b6b73b150db59cb0502b7283f6",
    ),
    (
        "Eigen/src/Core/arch/Default/BFloat16.h",
        f"{EIGEN_LEGAL_ROOT}/notices/BFloat16.h",
        36700,
        "6e1983203d85f4874cb11431b5b41357236ea806fd188cca80135043a9718ebc",
    ),
    (
        "Eigen/src/Geometry/AlignedBox.h",
        f"{EIGEN_LEGAL_ROOT}/notices/AlignedBox.h",
        19095,
        "1249e7e037e5c849eb1a4197c1f051e02dc09be8ce8377db2b5471a16c5e2036",
    ),
    (
        "Eigen/src/LU/arch/InverseSize4.h",
        f"{EIGEN_LEGAL_ROOT}/notices/InverseSize4.h",
        13886,
        "619dfeab1fb47d95a647d54f2e187d3aa92cc06f97b1e6a475f72f5f1987e68c",
    ),
)
EIGEN_RETAINED_FILES = tuple(
    record[1] for record in (*EIGEN_LICENSE_RECORDS, *EIGEN_EMBEDDED_NOTICE_RECORDS)
)
ARRAY_API_COMPAT_LICENSE = "LICENSES/array-api-compat-MIT.txt"
OPENBLAS_LICENSE = "LICENSES/scipy-openblas32-0.3.34.0.0.txt"
OPENBLAS_WINDOWS_LICENSE = "LICENSES/scipy-openblas32-tools-LICENSE_win32.txt"
OPENBLAS_EXACT_PACKAGED_LICENSES = (
    "LICENSES/scipy-openblas32-0.3.34.0.0-macos.txt",
    "LICENSES/scipy-openblas32-0.3.34.0.0-windows-amd64.txt",
    "LICENSES/scipy-openblas32-0.3.34.0.0-windows-arm64.txt",
)
OPENBLAS_MANIFEST_PATH = "cmake/3rdparty/scipy_openblas32_manifest.json"
OPENBLAS_MANIFEST_CANONICAL_SHA256 = (
    "738b14d01a73ae38bce1cc36b47b7034e1932bc35a9a508e86c8dfc8deb02d1b"
)
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
    "web/c60_case.js",
    "web/wasm/linalg_eigen.cpp",
    OPEN_CHEMLIB_MANIFEST,
)
SOURCE_FILES = (
    "LICENSE",
    EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT.txt",
    "LICENSES/BSD-3-Clause.txt",
    ARRAY_API_COMPAT_LICENSE,
    OPENBLAS_LICENSE,
    OPENBLAS_WINDOWS_LICENSE,
    *OPENBLAS_EXACT_PACKAGED_LICENSES,
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
    TORCH_STABLE_MANIFEST_PATH,
    EIGEN_MANIFEST_PATH,
    *EIGEN_RETAINED_FILES,
    "tools/eigen_dependency.py",
    OPENBLAS_MANIFEST_PATH,
)
COMMON_ARCHIVE_SUFFIXES = (
    "LICENSE",
    EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/MIT.txt",
    "LICENSES/BSD-3-Clause.txt",
    ARRAY_API_COMPAT_LICENSE,
    OPENBLAS_LICENSE,
    OPENBLAS_WINDOWS_LICENSE,
    *OPENBLAS_EXACT_PACKAGED_LICENSES,
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
    TORCH_STABLE_MANIFEST_PATH,
    EIGEN_MANIFEST_PATH,
    *EIGEN_RETAINED_FILES,
    "tools/eigen_dependency.py",
    OPENBLAS_MANIFEST_PATH,
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
    "share/licenses/xtbloom/provenance/torch_stable_manifest.json",
    "share/licenses/xtbloom/provenance/scipy_openblas32_manifest.json",
    "share/licenses/xtbloom/third-party/MIT.txt",
    "share/licenses/xtbloom/third-party/BSD-3-Clause.txt",
    "share/licenses/xtbloom/third-party/array-api-compat-MIT.txt",
    "share/licenses/xtbloom/third-party/scipy-openblas32-0.3.34.0.0.txt",
    "share/licenses/xtbloom/third-party/scipy-openblas32-tools-LICENSE_win32.txt",
    *(
        "share/licenses/xtbloom/third-party/" + PurePath(path).name
        for path in OPENBLAS_EXACT_PACKAGED_LICENSES
    ),
    "share/licenses/xtbloom/third-party/d4/d4.NOTICE",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING",
    "share/licenses/xtbloom/third-party/d4/dftd4-COPYING.LESSER",
    "share/licenses/xtbloom/third-party/d4/mctc-lib-LICENSE",
)
FORBIDDEN_ARCHIVE_PARTS = ("/build/", "/.cache/", "/.claude/", "/.ruff_cache/")
SDIST_INSTALLATION_FILES = (
    ".git_archival.txt",
    "CMakeLists.txt",
    "CUDA_MKL_LINKING_EXCEPTION",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "python/README.md",
    "python/ci/resolve-openblas-wheel.py",
    "tools/eigen_dependency.py",
    "tools/implib_stubgen.py",
    "tools/torch_stable_vendor.py",
)
SDIST_INSTALLATION_PREFIXES = (
    "LICENSES/",
    "cmake/",
    "data/parameters/",
    "include/",
    "python/xtbloom/",
    "src/",
    "tools/parameters/",
)
SDIST_INSTALLATION_EXCLUDED_FILES = (
    # This white-box helper is included only by a repository test translation
    # unit; the similarly named SCC helper remains because production CUDA
    # compilation includes its macro-gated declarations.
    "src/backends/cuda/gfn2_energy_force_execution_test.cuh",
)
SDIST_INSTALLATION_EXCLUDED_PREFIXES: tuple[str, ...] = ()
INSTALL_FILES = (
    "share/licenses/xtbloom/LICENSE",
    f"share/licenses/xtbloom/{EXCEPTION_FILE}",
    "share/licenses/xtbloom/THIRD_PARTY_NOTICES.md",
    "share/licenses/xtbloom/third-party/LGPL-3.0-or-later.txt",
    "share/licenses/xtbloom/third-party/Apache-2.0.txt",
    "share/licenses/xtbloom/third-party/MIT.txt",
    "share/licenses/xtbloom/third-party/BSD-3-Clause.txt",
    "share/licenses/xtbloom/third-party/array-api-compat-MIT.txt",
    "share/licenses/xtbloom/third-party/scipy-openblas32-0.3.34.0.0.txt",
    "share/licenses/xtbloom/third-party/scipy-openblas32-tools-LICENSE_win32.txt",
    *(
        "share/licenses/xtbloom/third-party/" + PurePath(path).name
        for path in OPENBLAS_EXACT_PACKAGED_LICENSES
    ),
    "share/licenses/xtbloom/provenance/manifest.json",
    "share/licenses/xtbloom/provenance/sto_manifest.json",
    "share/licenses/xtbloom/provenance/spin_manifest.json",
    "share/licenses/xtbloom/provenance/d4_manifest.json",
    "share/licenses/xtbloom/provenance/mctc_manifest.json",
    "share/licenses/xtbloom/provenance/implib_manifest.json",
    "share/licenses/xtbloom/provenance/torch_stable_manifest.json",
    "share/licenses/xtbloom/provenance/scipy_openblas32_manifest.json",
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
    TORCH_STABLE_TREE,
    "No LAMMPS source code",
    "scipy-openblas32",
    "scipy-openblas32-tools-LICENSE_win32.txt",
    "1ce4c83d89bc30a0a97d4bc18d72ccaa9d3cb7c90ba1408c6b3e29ebf0c5a71c",
    "array-api-compat",
    "scikit-build-core >=1.0.3",
    "setuptools-scm >=10.2.1",
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
    "Eigen 5.0.1",
    EIGEN_REVISION,
    EIGEN_ARCHIVE_SHA256,
    EIGEN_MANIFEST_PATH,
    "unsupported/",
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
OPENBLAS_BINARY_RE = re.compile(
    r"(?:^|/)(?:libxtbloom_openblas_lp64_shim\.so|"
    r"libscipy_openblas(?:[^/]*\.so(?:\.[0-9]+)*|\.dylib|\.dll)|"
    r"scipy_openblas\.dll|"
    r"libgfortran-[^/]*\.so(?:\.[0-9]+)*|libquadmath-[^/]*\.so(?:\.[0-9]+)*|"
    r"lib(?:gfortran|quadmath|gcc_s)[^/]*\.dylib|"
    r"libxtbloom_blas-[0-9a-f]{8}\.dylib|libxb(?:gf|qm|gcc)-[0-9a-f]{8}\.dylib|"
    r"xtbloom_openblas-[0-9a-f]{8}\.dll)$"
)
WEB_SITE_SOURCE_MAP = {
    "LICENSE": "LICENSE",
    EXCEPTION_FILE: EXCEPTION_FILE,
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
    "LICENSES/Apache-2.0.txt": "LICENSES/Apache-2.0.txt",
    "LICENSES/BSD-3-Clause.txt": "LICENSES/BSD-3-Clause.txt",
    "LICENSES/LGPL-3.0-or-later.txt": "LICENSES/LGPL-3.0-or-later.txt",
    "LICENSES/MIT.txt": "LICENSES/MIT.txt",
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
    **{path: path for path in EIGEN_RETAINED_FILES},
    "provenance/eigen_manifest.json": EIGEN_MANIFEST_PATH,
}
WEB_SITE_RUNTIME_FILES = (
    "index.html",
    "style.css",
    "bootstrap.js",
    "app.js",
    "c60_case.js",
    "app_helpers.js",
    "worker.js",
    "smiles_helpers.js",
    "smiles_worker.js",
    "xtbloom-mark.svg",
    "engine-manifest.json",
    "xtbloom_web.js",
    "xtbloom_web.wasm",
    "xtbloom_web.data",
    "vendor/3Dmol-min.js",
)
WEB_VERSIONED_ASSETS = (
    ("app", "app.js"),
    ("c60", "c60_case.js"),
    ("worker", "worker.js"),
    ("helpers", "app_helpers.js"),
    ("module", "xtbloom_web.js"),
    ("wasm", "xtbloom_web.wasm"),
    ("data", "xtbloom_web.data"),
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
    if "scipy-openblas32" in mandatory:
        raise LicenseCheckError(
            "scipy-openblas32 is a wheel-build input and must not be a "
            "runtime dependency"
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
        if (
            names.intersection(NVIDIA_DEPENDENCIES)
            or "mkl" in names
            or "scipy-openblas32" in names
        ):
            raise LicenseCheckError(
                "build-only or proprietary providers must not be included in "
                f"the {extra} extra"
            )


def _require_build_dependency_policy(build_system: object) -> None:
    """Require the reviewed direct PEP 517 tools with compatible lower bounds."""
    if not isinstance(build_system, dict):
        raise LicenseCheckError("pyproject build-system metadata must be a table")
    requirements = build_system.get("requires")
    if (
        not isinstance(requirements, list)
        or set(requirements) != REVIEWED_BUILD_REQUIREMENTS
    ):
        raise LicenseCheckError(
            "build-system.requires must equal the reviewed direct compatible "
            "requirements"
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
        {"provider": "scikit_build_core.metadata.setuptools_scm"}
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


def _require_openblas_build_policy(metadata: object) -> None:
    """Require an exact build-only pin without publishing a runtime edge."""
    if not isinstance(metadata, dict):
        raise LicenseCheckError("pyproject metadata must be a table")
    dependency_groups = metadata.get("dependency-groups", {})
    wheel_group = dependency_groups.get("wheel-build", [])
    if not isinstance(wheel_group, list) or len(wheel_group) != 1:
        raise LicenseCheckError(
            "wheel-build must lock one reviewed scipy-openblas32 input"
        )
    requirement = wheel_group[0]
    for token in (
        "scipy-openblas32==0.3.34.0.0",
        "sys_platform == 'linux'",
        "sys_platform == 'darwin'",
        "sys_platform == 'win32'",
        "x86_64",
        "aarch64",
        "arm64",
        "AMD64",
        "ARM64",
    ):
        if token not in requirement:
            raise LicenseCheckError(
                "wheel-build must lock the reviewed scipy-openblas32 version "
                "and architectures"
            )

    scikit_build = metadata.get("tool", {}).get("scikit-build", {})
    overrides = scikit_build.get("overrides", [])
    matching = [
        override
        for override in overrides
        if override.get("build", {}).get("requires") == ["scipy-openblas32==0.3.34.0.0"]
    ]
    if len(matching) != 1:
        raise LicenseCheckError(
            "scikit-build must request exactly one reviewed scipy-openblas32 "
            "wheel build input"
        )
    conditions = matching[0].get("if", {})
    if conditions != {
        "state": "^wheel$",
        "platform-system": "(?i)^(linux|darwin|macos|windows|win32)$",
        "platform-machine": "(?i)^(x86_64|aarch64|arm64|amd64)$",
        "env": {"XTBLOOM_BUNDLE_WHEEL_OPENBLAS": "^ON$"},
    }:
        raise LicenseCheckError(
            "scipy-openblas32 build input must be confined to reviewed native "
            "Linux, macOS, and Windows wheels"
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


def _git_index_modes(root: Path, relative_paths: set[str]) -> dict[str, str] | None:
    """Return tracked Git modes when ``root`` is an actual checkout.

    Windows filesystems do not reliably expose Git's executable bit through
    ``stat``.  The index remains authoritative in a checkout, while extracted
    source archives intentionally fall back to filesystem capabilities.
    """
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--stage",
                "--",
                *sorted(relative_paths),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise LicenseCheckError("cannot inspect vendored Git index modes") from error
    if result.returncode != 0:
        raise LicenseCheckError(
            "cannot inspect vendored Git index modes: " + result.stderr.strip()
        )

    observed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        try:
            metadata, relative = line.split("\t", 1)
            mode, _object_id, stage = metadata.split()
        except ValueError as error:
            raise LicenseCheckError(
                "Git returned an invalid index-mode record"
            ) from error
        if stage != "0" or mode not in ("100644", "100755"):
            raise LicenseCheckError(f"Git returned an invalid mode for {relative}")
        if relative in observed:
            raise LicenseCheckError(f"Git returned duplicate modes for {relative}")
        observed[relative] = mode
    if set(observed) != relative_paths:
        missing = sorted(relative_paths - set(observed))
        unexpected = sorted(set(observed) - relative_paths)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise LicenseCheckError(
            "vendored Git index paths differ: " + "; ".join(details)
        )
    return observed


def _filesystem_git_mode(path: Path) -> str | None:
    """Map a filesystem mode to Git semantics when the platform supports it."""
    if os.name == "nt":
        return None
    return "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"


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

    index_paths = {f"{IMPLIB_VENDOR_PATH}/{relative}" for relative in declared}
    index_modes = _git_index_modes(root, index_paths)
    observed_tree: dict[str, tuple[str, str]] = {}
    for relative, (expected_mode, expected_blob, expected_sha256) in declared.items():
        path = vendor_root / relative
        if path.is_symlink():
            raise LicenseCheckError(
                f"implib vendored file must not be a symlink: {relative}"
            )
        data = path.read_bytes()
        index_path = f"{IMPLIB_VENDOR_PATH}/{relative}"
        available_modes: list[tuple[str, str]] = []
        if index_modes is not None:
            available_modes.append(("Git index", index_modes[index_path]))
        filesystem_mode = _filesystem_git_mode(path)
        if filesystem_mode is not None:
            available_modes.append(("filesystem", filesystem_mode))
        for source, observed_mode in available_modes:
            if observed_mode != expected_mode:
                raise LicenseCheckError(
                    "implib vendored file mode differs from pinned Git mode: "
                    f"{relative} ({source}: expected {expected_mode}, "
                    f"observed {observed_mode})"
                )
        observed_blob = _git_object_id("blob", data)
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if observed_blob != expected_blob or observed_sha256 != expected_sha256:
            raise LicenseCheckError(
                f"implib vendored file differs from pinned bytes: {relative}"
            )
        observed_tree[relative] = (expected_mode, observed_blob)
    if _git_tree_id(observed_tree) != IMPLIB_TREE:
        raise LicenseCheckError("implib vendored tree does not match the pinned tree")


def _check_torch_stable_manifest(manifest: object) -> dict[str, tuple[str, str, str]]:
    """Validate pinned LibTorch Stable ABI metadata and its file mapping."""
    if not isinstance(manifest, dict):
        raise LicenseCheckError("torch-stable manifest root must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("license") != "BSD-3-Clause"
        or manifest.get("upstream_repository") != "https://github.com/pytorch/pytorch"
        or manifest.get("upstream_release") != TORCH_STABLE_REVISION
        or manifest.get("source_path") != "torch/include"
        or manifest.get("tree") != TORCH_STABLE_TREE
    ):
        raise LicenseCheckError("torch-stable manifest has incorrect pinned provenance")

    declared: dict[str, tuple[str, str, str]] = {}
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 49:
        raise LicenseCheckError("torch-stable manifest must describe exactly 49 files")
    for entry in files:
        if not isinstance(entry, dict):
            raise LicenseCheckError("torch-stable manifest has a non-object file entry")
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
            raise LicenseCheckError(
                "torch-stable manifest contains an invalid file entry"
            )
        if path in declared:
            raise LicenseCheckError(f"torch-stable manifest duplicates {path}")
        declared[path] = (mode, blob, sha256)

    declared_tree = _git_tree_id(
        {path: (mode, blob) for path, (mode, blob, _sha256) in declared.items()}
    )
    if declared_tree != TORCH_STABLE_TREE:
        raise LicenseCheckError(
            "torch-stable manifest file entries do not match the pinned tree"
        )
    return declared


def _check_torch_stable_provenance(root: Path) -> None:
    """Verify the vendored torch-stable tree is exactly the pinned torch copy."""
    manifest = json.loads(
        (root / TORCH_STABLE_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    declared = _check_torch_stable_manifest(manifest)

    vendor_root = root / TORCH_STABLE_VENDOR_PATH
    observed_paths = {
        path.relative_to(vendor_root).as_posix()
        for path in vendor_root.glob(f"{TORCH_STABLE_INCLUDE_SUBDIR}/**/*")
        if path.is_file() or path.is_symlink()
    }
    expected = {f"{TORCH_STABLE_INCLUDE_SUBDIR}/{relative}" for relative in declared}
    if observed_paths != expected:
        missing = sorted(expected - observed_paths)
        unexpected = sorted(observed_paths - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise LicenseCheckError(
            "torch-stable vendored file set differs: " + "; ".join(details)
        )

    observed_tree: dict[str, tuple[str, str]] = {}
    for relative, (expected_mode, expected_blob, expected_sha256) in declared.items():
        path = vendor_root / TORCH_STABLE_INCLUDE_SUBDIR / relative
        if path.is_symlink():
            raise LicenseCheckError(
                f"torch-stable vendored file must not be a symlink: {relative}"
            )
        data = path.read_bytes()
        observed_blob = _git_object_id("blob", data)
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if observed_blob != expected_blob or observed_sha256 != expected_sha256:
            raise LicenseCheckError(
                f"torch-stable vendored file differs from pinned bytes: {relative}"
            )
        observed_tree[relative] = (expected_mode, observed_blob)
    if _git_tree_id(observed_tree) != TORCH_STABLE_TREE:
        raise LicenseCheckError(
            "torch-stable vendored tree does not match the pinned tree"
        )


def _check_eigen_manifest(manifest: object) -> dict[str, tuple[int, str]]:
    """Validate Eigen release metadata and return retained legal-file hashes."""
    if not isinstance(manifest, dict):
        raise LicenseCheckError("Eigen manifest root must be an object")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dependency") != "Eigen"
        or manifest.get("version") != EIGEN_VERSION
        or manifest.get("revision") != EIGEN_REVISION
        or manifest.get("upstream_repository") != "https://gitlab.com/libeigen/eigen"
        or manifest.get("release_archive")
        != {
            "url": EIGEN_ARCHIVE_URL,
            "size_bytes": EIGEN_ARCHIVE_SIZE,
            "sha256": EIGEN_ARCHIVE_SHA256,
        }
        or manifest.get("acquisition") != EIGEN_ACQUISITION
    ):
        raise LicenseCheckError("Eigen manifest has incorrect pinned provenance")

    expected_records = [
        {
            "upstream_path": upstream_path,
            "local_path": local_path,
            "spdx": spdx,
            "scope": scope,
            "size_bytes": size,
            "sha256": sha256,
        }
        for upstream_path, local_path, spdx, scope, size, sha256 in (
            EIGEN_LICENSE_RECORDS
        )
    ]
    expected_notices = [
        {
            "upstream_path": upstream_path,
            "local_path": local_path,
            "size_bytes": size,
            "sha256": sha256,
        }
        for upstream_path, local_path, size, sha256 in EIGEN_EMBEDDED_NOTICE_RECORDS
    ]
    if manifest.get("license") != {
        "primary_spdx": "MPL-2.0",
        "records": expected_records,
        "embedded_notices": expected_notices,
    }:
        raise LicenseCheckError("Eigen manifest has incorrect license records")
    if manifest.get("distribution") != EIGEN_DISTRIBUTION:
        raise LicenseCheckError("Eigen manifest has incorrect distribution policy")

    declared: dict[str, tuple[int, str]] = {}
    for entry in (*expected_records, *expected_notices):
        local_path = entry["local_path"]
        if local_path in declared:
            raise LicenseCheckError(f"Eigen manifest duplicates {local_path}")
        declared[local_path] = (entry["size_bytes"], entry["sha256"])
    if set(declared) != set(EIGEN_RETAINED_FILES):
        raise LicenseCheckError("Eigen manifest has incomplete legal-file coverage")
    return declared


def _check_eigen_provenance(root: Path) -> None:
    """Verify retained Eigen legal bytes and reject a repository-side source tree."""
    manifest = json.loads((root / EIGEN_MANIFEST_PATH).read_text(encoding="utf-8"))
    declared = _check_eigen_manifest(manifest)
    for relative, (expected_size, expected_sha256) in declared.items():
        path = root / relative
        if not path.is_file():
            raise LicenseCheckError(f"retained Eigen legal file is missing: {relative}")
        if path.is_symlink():
            raise LicenseCheckError(
                f"retained Eigen legal file is a symlink: {relative}"
            )
        data = path.read_bytes()
        if (
            len(data) != expected_size
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise LicenseCheckError(
                f"retained Eigen legal file differs from pinned bytes: {relative}"
            )

    allowed_payloads = {EIGEN_MANIFEST_PATH, *declared}
    unexpected_payloads = sorted(
        relative
        for path in root.rglob("*")
        if path.is_file()
        and (relative := path.relative_to(root).as_posix()) not in allowed_payloads
        and not _is_ignored_source_artifact(relative)
        and _is_eigen_payload_name(relative)
    )
    if unexpected_payloads:
        raise LicenseCheckError(
            "repository must not vendor Eigen source or archives: "
            + unexpected_payloads[0]
        )

    attributes = (root / ".gitattributes").read_text(encoding="utf-8")
    if "LICENSES/eigen/** -text" not in attributes:
        raise LicenseCheckError(
            ".gitattributes does not preserve retained Eigen legal bytes"
        )
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    sdist_include = (
        metadata.get("tool", {})
        .get("scikit-build", {})
        .get("sdist", {})
        .get("include", [])
    )
    for required in (
        EIGEN_MANIFEST_PATH,
        "LICENSES/eigen/**",
        "tools/eigen_dependency.py",
    ):
        if required not in sdist_include:
            raise LicenseCheckError(f"sdist.include omits {required}")
    for forbidden in ("cmake/3rdparty/eigen/**", "tools/eigen_vendor.py"):
        if forbidden in sdist_include:
            raise LicenseCheckError(f"sdist.include retains obsolete {forbidden}")


def _is_ignored_source_artifact(name: str) -> bool:
    """Ignore local build/cache products when auditing repository dependencies."""
    parts = tuple(part.lower() for part in PurePath(name).parts)
    if not parts:
        return False
    return (
        parts[0].startswith("build")
        or parts[0] in {".cache", ".git", ".ruff_cache", ".venv", "dist"}
        or "node_modules" in parts
    )


def _is_eigen_payload_name(name: str) -> bool:
    """Recognize Eigen material even after source/install-path relocation."""
    path = PurePath(name)
    parts = {part.lower() for part in path.parts}
    archive = re.fullmatch(
        r"eigen(?:-[0-9a-z.+_-]+)?\.(?:tar(?:\.(?:gz|bz2|xz|zst))?|tgz|zip)",
        path.name.lower(),
    )
    return (
        "eigen_manifest.json" in parts
        or bool(parts & {"eigen", "eigen3"})
        or archive is not None
    )


def _check_openblas_manifest(manifest: object) -> dict[str, dict[str, object]]:
    """Validate every reviewed native wheel and redistributed binary cohort."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise LicenseCheckError("scipy-openblas32 manifest has an unsupported schema")
    dependency = manifest.get("dependency", {})
    source = manifest.get("source", {})
    if dependency != {
        "name": "scipy-openblas32",
        "version": "0.3.34.0.0",
        "classification": "wheel build input and redistributed private binary provider",
        "runtime_dependency": False,
    }:
        raise LicenseCheckError(
            "scipy-openblas32 manifest has unreviewed dependency metadata"
        )
    expected_source = {
        "repository": "https://github.com/MacPython/openblas-libs",
        "release_tag": "v0.3.34.0.0",
        "release_commit": "7e5538356afac3934e872b8b572799b875900657",
        "openblas_repository": "https://github.com/OpenMathLib/OpenBLAS",
        "openblas_tag": "v0.3.34",
        "openblas_commit": "e0166008be8e466242aa76b2ff75ce3f0fbf574a",
        "local_license": OPENBLAS_LICENSE,
        "local_license_sha256": (
            "51b0d449fdce3b1fabfead1af1cc6eff1df46a70b378c27dc0f5663afc6cc66a"
        ),
        "windows_license_source": "tools/LICENSE_win32.txt",
        "windows_license_url": (
            "https://github.com/MacPython/openblas-libs/blob/"
            "7e5538356afac3934e872b8b572799b875900657/tools/LICENSE_win32.txt"
        ),
        "windows_license_sha256": (
            "1ce4c83d89bc30a0a97d4bc18d72ccaa9d3cb7c90ba1408c6b3e29ebf0c5a71c"
        ),
        "local_windows_license": OPENBLAS_WINDOWS_LICENSE,
    }
    for key, value in expected_source.items():
        if source.get(key) != value:
            raise LicenseCheckError(f"scipy-openblas32 manifest has unreviewed {key}")
    if set(source) != set(expected_source):
        raise LicenseCheckError("scipy-openblas32 manifest source fields differ")

    targets = manifest.get("targets", {})
    expected_targets = {
        "linux-x86_64",
        "linux-aarch64",
        "macos-x86_64",
        "macos-arm64",
        "windows-amd64",
        "windows-arm64",
    }
    if not isinstance(targets, dict) or set(targets) != expected_targets:
        raise LicenseCheckError("scipy-openblas32 manifest target set differs")
    for target, record in targets.items():
        if not isinstance(record, dict):
            raise LicenseCheckError(f"scipy-openblas32 {target} record is invalid")
        platform_name, architecture = target.split("-", 1)
        if (
            record.get("platform") != platform_name
            or record.get("architecture") != architecture
        ):
            raise LicenseCheckError(f"scipy-openblas32 {target} identity differs")
        linux = platform_name == "linux"
        if record.get("bundle_strategy") != (
            "auditwheel-shim" if linux else "renamed-direct-provider"
        ) or record.get("thread_control") != ("local" if linux else "private-global"):
            raise LicenseCheckError(f"scipy-openblas32 {target} runtime policy differs")
        config_prefix = record.get("expected_config_prefix")
        if config_prefix not in {"OpenBLAS 0.3.34", "OpenBLAS 0.3.34.0.0"}:
            raise LicenseCheckError(f"scipy-openblas32 {target} config prefix differs")
        wheel = record.get("wheel", {})
        license_record = record.get("license", {})
        expected_local_license = {
            "linux-x86_64": OPENBLAS_LICENSE,
            "linux-aarch64": OPENBLAS_LICENSE,
            "macos-x86_64": OPENBLAS_EXACT_PACKAGED_LICENSES[0],
            "macos-arm64": OPENBLAS_EXACT_PACKAGED_LICENSES[0],
            "windows-amd64": OPENBLAS_EXACT_PACKAGED_LICENSES[1],
            "windows-arm64": OPENBLAS_EXACT_PACKAGED_LICENSES[2],
        }[target]
        if (
            not isinstance(wheel, dict)
            or not str(wheel.get("filename", "")).startswith(
                "scipy_openblas32-0.3.34.0.0-py3-none-"
            )
            or not str(wheel.get("url", "")).startswith(
                "https://files.pythonhosted.org/packages/"
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(wheel.get("sha256", "")))
            or not isinstance(wheel.get("size"), int)
            or wheel["size"] <= 0
            or not isinstance(license_record, dict)
            or license_record.get("source")
            != "scipy_openblas32-0.3.34.0.0.dist-info/licenses/LICENSE.txt"
            or license_record.get("local") != expected_local_license
            or not re.fullmatch(r"[0-9a-f]{64}", str(license_record.get("sha256", "")))
        ):
            raise LicenseCheckError(f"scipy-openblas32 {target} wheel differs")
        files = record.get("files", [])
        if not isinstance(files, list) or not files:
            raise LicenseCheckError(f"scipy-openblas32 {target} cohort is empty")
        sources: set[str] = set()
        provider_count = 0
        provider_item: dict[str, object] | None = None
        for item in files:
            if not isinstance(item, dict):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} cohort entry is invalid"
                )
            item_source = item.get("source")
            if (
                not isinstance(item_source, str)
                or item_source in sources
                or item.get("role") not in {"provider", "support"}
                or not isinstance(item.get("size"), int)
                or item["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            ):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} cohort entry differs"
                )
            sources.add(item_source)
            provider_count += item.get("role") == "provider"
            if item.get("role") == "provider":
                provider_item = item
            if not linux and (
                not isinstance(item.get("install_destination"), str)
                or not isinstance(item.get("install_name"), str)
            ):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} install mapping differs"
                )
            if platform_name == "macos":
                install_name = item["install_name"]
                install_id = item.get("install_id")
                rewrites = item.get("load_rewrites")
                if (
                    not install_name.startswith("libx")
                    or item["sha256"][:8] not in install_name
                    or install_id != f"@rpath/{install_name}"
                    or not isinstance(rewrites, list)
                    or any(
                        not isinstance(rewrite, dict)
                        or set(rewrite) != {"from", "to"}
                        or not isinstance(rewrite["from"], str)
                        or not isinstance(rewrite["to"], str)
                        for rewrite in rewrites
                    )
                ):
                    raise LicenseCheckError(
                        f"scipy-openblas32 {target} Mach-O private mapping differs"
                    )
            elif platform_name == "windows" and (
                not re.fullmatch(
                    r"xtbloom_openblas-[0-9a-f]{8}\.dll", item["install_name"]
                )
                or item["sha256"][:8] not in item["install_name"]
            ):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} PE private mapping differs"
                )
        if (
            provider_count != 1
            or record.get("provider_source") not in sources
            or (
                platform_name == "windows"
                and record.get("supplemental_license") != "tools/LICENSE_win32.txt"
            )
        ):
            raise LicenseCheckError(f"scipy-openblas32 {target} provider differs")
        if platform_name == "macos":
            assert provider_item is not None
            installed_names = {item["install_name"] for item in files}
            for item in files:
                for rewrite in item["load_rewrites"]:
                    if PurePath(rewrite["to"]).name not in installed_names:
                        raise LicenseCheckError(
                            f"scipy-openblas32 {target} Mach-O dependency "
                            "escapes cohort"
                        )
            installed_provider = (
                f"{provider_item['install_destination']}/"
                f"{provider_item['install_name']}"
            )
            if (
                record.get("installed_provider") != installed_provider
                or record.get("installed_provider_id") != provider_item["install_id"]
            ):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} installed provider differs"
                )
        elif platform_name == "windows":
            assert provider_item is not None
            if record.get("installed_provider") != (
                f"{provider_item['install_destination']}/{provider_item['install_name']}"
            ):
                raise LicenseCheckError(
                    f"scipy-openblas32 {target} installed provider differs"
                )

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != OPENBLAS_MANIFEST_CANONICAL_SHA256:
        raise LicenseCheckError(
            "scipy-openblas32 manifest differs from reviewed canonical provenance"
        )
    return targets


def _check_openblas_provenance(root: Path) -> None:
    manifest = json.loads((root / OPENBLAS_MANIFEST_PATH).read_text(encoding="utf-8"))
    _check_openblas_manifest(manifest)
    source = manifest["source"]
    if (
        hashlib.sha256((root / OPENBLAS_LICENSE).read_bytes()).hexdigest()
        != source["local_license_sha256"]
    ):
        raise LicenseCheckError(
            "scipy-openblas32 license differs from pinned upstream bytes"
        )
    if (
        hashlib.sha256((root / OPENBLAS_WINDOWS_LICENSE).read_bytes()).hexdigest()
        != source["windows_license_sha256"]
    ):
        raise LicenseCheckError(
            "scipy-openblas32 Windows license differs from pinned upstream bytes"
        )
    local_license_hashes: dict[str, str] = {}
    for target in manifest["targets"].values():
        license_record = target["license"]
        local = license_record["local"]
        expected = license_record["sha256"]
        previous = local_license_hashes.setdefault(local, expected)
        if previous != expected:
            raise LicenseCheckError(
                "scipy-openblas32 local license maps to inconsistent upstream bytes"
            )
    for local, expected in local_license_hashes.items():
        if hashlib.sha256((root / local).read_bytes()).hexdigest() != expected:
            raise LicenseCheckError(
                f"scipy-openblas32 exact packaged license differs: {local}"
            )


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
    _require_build_dependency_policy(metadata.get("build-system"))
    _require_version_metadata_policy(metadata)
    _require_openblas_build_policy(metadata)
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
    _check_torch_stable_provenance(root)
    _check_eigen_provenance(root)
    _check_openblas_provenance(root)

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
    openblas_binaries = sorted(
        path.relative_to(prefix).as_posix()
        for path in prefix.rglob("*")
        if path.is_file()
        and OPENBLAS_BINARY_RE.search(path.relative_to(prefix).as_posix())
    )
    if openblas_binaries:
        raise LicenseCheckError(
            "native install bundles a wheel-only OpenBLAS binary: "
            + openblas_binaries[0]
        )
    eigen_payloads = sorted(
        path.relative_to(prefix).as_posix()
        for path in prefix.rglob("*")
        if path.is_file()
        and _is_eigen_payload_name(path.relative_to(prefix).as_posix())
    )
    if eigen_payloads:
        raise LicenseCheckError(
            "native install bundles Web-only Eigen material: " + eigen_payloads[0]
        )


def _check_web_engine_manifest(site: Path) -> None:
    """Require the browser cache version to describe exact deployed bytes."""
    manifest_path = site / "engine-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise LicenseCheckError("web engine manifest has an unsupported schema")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) != len(WEB_VERSIONED_ASSETS):
        raise LicenseCheckError("web engine manifest has incomplete asset coverage")

    expected_entries: list[dict[str, object]] = []
    version_material = ""
    for asset_id, relative in WEB_VERSIONED_ASSETS:
        path = site / relative
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        size = len(payload)
        expected_entries.append(
            {"id": asset_id, "path": relative, "bytes": size, "sha256": digest}
        )
        version_material += f"{asset_id}:{relative}:{size}:{digest}\n"
    if assets != expected_entries:
        raise LicenseCheckError("web engine manifest does not match deployed assets")
    expected_version = hashlib.sha256(version_material.encode()).hexdigest()
    if manifest.get("version") != expected_version:
        raise LicenseCheckError("web engine manifest has an invalid content version")


def check_web_site(site: Path, source_root: Path | None = None) -> None:
    """Validate the legal payload conveyed beside the browser binaries."""
    _require_files(
        site,
        WEB_SITE_RUNTIME_FILES + tuple(WEB_SITE_SOURCE_MAP),
        "web site",
    )
    if (site / "libscipy_openblas.so").exists():
        raise LicenseCheckError(
            "web site contains the raw Eigen LAPACKE/CBLAS side module; "
            "it must only be conveyed inside xtbloom_web.data"
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
    _check_web_engine_manifest(site)
    index = (site / "index.html").read_text(encoding="utf-8")
    for token in (
        'href="LICENSE"',
        'href="THIRD_PARTY_NOTICES.md"',
        f'href="{EXCEPTION_FILE}"',
        "https://xtbloom.jinzhezeng.group",
        "https://github.com/jinzhezenggroup/xtbloom",
        'href="LICENSES/openchemlib-BSD-3-Clause.txt"',
        'href="LICENSES/eigen/COPYING.MPL2"',
        'href="provenance/eigen_manifest.json"',
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


def _check_archived_torch_stable(path: Path, names: set[str]) -> None:
    """Validate the sdist carries the exact vendored LibTorch header tree."""
    manifest_suffix = TORCH_STABLE_MANIFEST_PATH
    manifest_name = _find_archive_name(names, manifest_suffix)
    manifest_bytes = _read_archive_members(path, {manifest_name})[manifest_name]
    declared = _check_torch_stable_manifest(json.loads(manifest_bytes.decode("utf-8")))

    archive_root = manifest_name[: -len(manifest_suffix)]
    vendor_prefix = (
        archive_root
        + TORCH_STABLE_VENDOR_PATH
        + "/"
        + TORCH_STABLE_INCLUDE_SUBDIR
        + "/"
    )
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
            "sdist torch-stable vendored file set differs: " + "; ".join(details)
        )
    vendor_payloads = _read_archive_members(path, set(archived_vendor.values()))
    for relative, (_mode, expected_blob, expected_sha256) in declared.items():
        data = vendor_payloads[archived_vendor[relative]]
        if (
            _git_object_id("blob", data) != expected_blob
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise LicenseCheckError(
                "sdist torch-stable vendored file differs from pinned bytes: "
                f"{relative}"
            )


def _check_archived_eigen(path: Path, names: set[str]) -> None:
    """Validate sdist provenance/legal bytes and reject Eigen source or archives."""
    manifest_name = _find_archive_name(names, EIGEN_MANIFEST_PATH)
    manifest_bytes = _read_archive_members(path, {manifest_name})[manifest_name]
    declared = _check_eigen_manifest(json.loads(manifest_bytes.decode("utf-8")))

    archive_root = manifest_name[: -len(EIGEN_MANIFEST_PATH)]
    archived_legal = {
        relative: archive_root + relative
        for relative in declared
        if archive_root + relative in names
    }
    missing = sorted(set(declared) - set(archived_legal))
    if missing:
        raise LicenseCheckError(
            "sdist is missing retained Eigen legal files: " + ", ".join(missing)
        )
    payloads = _read_archive_members(path, set(archived_legal.values()))
    for relative, (expected_size, expected_sha256) in declared.items():
        data = payloads[archived_legal[relative]]
        if (
            len(data) != expected_size
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise LicenseCheckError(
                f"sdist retained Eigen legal file differs from pinned bytes: {relative}"
            )

    allowed_payloads = {EIGEN_MANIFEST_PATH, *declared}
    unexpected = sorted(
        relative
        for name in names
        if name.startswith(archive_root)
        and (relative := name.removeprefix(archive_root)) not in allowed_payloads
        and _is_eigen_payload_name(relative)
    )
    if unexpected:
        raise LicenseCheckError(
            "sdist must not bundle Eigen source or archives: " + unexpected[0]
        )


def _auditwheel_name(source_name: str, sha256: str) -> str:
    index = source_name.find(".so")
    if index < 0:
        raise LicenseCheckError(
            f"OpenBLAS manifest payload is not a DSO: {source_name}"
        )
    return source_name[:index] + f"-{sha256[:8]}" + source_name[index:]


def _openblas_target_for_wheel(path: Path) -> str | None:
    """Map one native wheel tag to its reviewed private-provider target."""
    name = path.name.lower()
    if "manylinux" in name and name.endswith("_x86_64.whl"):
        return "linux-x86_64"
    if "manylinux" in name and name.endswith("_aarch64.whl"):
        return "linux-aarch64"
    if "macosx" in name and name.endswith("_x86_64.whl"):
        return "macos-x86_64"
    if "macosx" in name and name.endswith("_arm64.whl"):
        return "macos-arm64"
    if name.endswith("-win_amd64.whl"):
        return "windows-amd64"
    if name.endswith("-win_arm64.whl"):
        return "windows-arm64"
    return None


def _check_archived_openblas(path: Path, names: set[str], wheel: bool) -> None:
    manifest_suffix = (
        "share/licenses/xtbloom/provenance/scipy_openblas32_manifest.json"
        if wheel
        else OPENBLAS_MANIFEST_PATH
    )
    manifest_name = _find_archive_name(names, manifest_suffix)
    license_suffix = (
        "share/licenses/xtbloom/third-party/scipy-openblas32-0.3.34.0.0.txt"
        if wheel
        else OPENBLAS_LICENSE
    )
    windows_license_suffix = (
        "share/licenses/xtbloom/third-party/scipy-openblas32-tools-LICENSE_win32.txt"
        if wheel
        else OPENBLAS_WINDOWS_LICENSE
    )
    license_name = _find_archive_name(names, license_suffix)
    windows_license_name = _find_archive_name(names, windows_license_suffix)
    payloads = _read_archive_members(
        path, {manifest_name, license_name, windows_license_name}
    )
    manifest = json.loads(payloads[manifest_name].decode("utf-8"))
    targets = _check_openblas_manifest(manifest)
    source = manifest["source"]
    if (
        hashlib.sha256(payloads[license_name]).hexdigest()
        != source["local_license_sha256"]
    ):
        raise LicenseCheckError("archived scipy-openblas32 license bytes differ")
    if (
        hashlib.sha256(payloads[windows_license_name]).hexdigest()
        != source["windows_license_sha256"]
    ):
        raise LicenseCheckError(
            "archived scipy-openblas32 Windows license bytes differ"
        )

    exact_expected: dict[str, str] = {}
    for target_record in targets.values():
        license_record = target_record["license"]
        local = license_record["local"]
        expected = license_record["sha256"]
        previous = exact_expected.setdefault(local, expected)
        if previous != expected:
            raise LicenseCheckError(
                "archived scipy-openblas32 local license mapping is inconsistent"
            )
    exact_members = {
        local: _find_archive_name(
            names,
            (
                "share/licenses/xtbloom/third-party/" + PurePath(local).name
                if wheel
                else local
            ),
        )
        for local in exact_expected
    }
    exact_payloads = _read_archive_members(path, set(exact_members.values()))
    for local, expected in exact_expected.items():
        if hashlib.sha256(exact_payloads[exact_members[local]]).hexdigest() != expected:
            raise LicenseCheckError(
                f"archived scipy-openblas32 exact packaged license differs: {local}"
            )

    openblas_binaries = sorted(
        name for name in names if OPENBLAS_BINARY_RE.search(name)
    )
    if not wheel:
        if openblas_binaries:
            raise LicenseCheckError(
                "sdist must not bundle wheel-only OpenBLAS binary: "
                f"{openblas_binaries[0]}"
            )
        return

    target = _openblas_target_for_wheel(path)
    if target is None:
        if openblas_binaries:
            raise LicenseCheckError(
                "wheel without a reviewed OpenBLAS target bundles private binaries"
            )
        return

    target_record = targets[target]
    if target.startswith("linux-"):
        expected = {"libxtbloom_openblas_lp64_shim.so"}
        for record in target_record["files"]:
            expected.add(
                _auditwheel_name(PurePath(record["source"]).name, record["sha256"])
            )
    else:
        expected = {record["install_name"] for record in target_record["files"]}
    observed = {PurePath(name).name for name in openblas_binaries}
    if observed != expected:
        raise LicenseCheckError(
            "wheel OpenBLAS binary cohort differs: expected "
            f"{sorted(expected)}, found {sorted(observed)}"
        )

    if target.startswith("windows-"):
        members_by_basename = {PurePath(name).name: name for name in openblas_binaries}
        exact_records = target_record["files"]
        exact_payloads = _read_archive_members(
            path,
            {members_by_basename[record["install_name"]] for record in exact_records},
        )
        for record in exact_records:
            member = members_by_basename[record["install_name"]]
            data = exact_payloads[member]
            if (
                len(data) != record["size"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]
            ):
                raise LicenseCheckError(
                    f"wheel OpenBLAS payload differs: {record['install_name']}"
                )


def _tracked_sdist_installation_manifest(source_root: Path) -> dict[str, int]:
    """Select allowed tracked files and their executable-bit contract."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "ls-files", "--stage", "-z"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise LicenseCheckError(
            "cannot enumerate tracked files for the sdist payload audit"
        ) from exc
    tracked: dict[str, str] = {}
    for record in result.stdout.split("\0"):
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        tracked[name] = metadata.split(" ", 1)[0]
    policy_files = set(SDIST_INSTALLATION_FILES) | set(
        SDIST_INSTALLATION_EXCLUDED_FILES
    )
    missing_policy_files = sorted(policy_files - set(tracked))
    if missing_policy_files:
        raise LicenseCheckError(
            "sdist installation policy names untracked files: "
            + ", ".join(missing_policy_files)
        )
    missing_excluded_prefixes = [
        prefix
        for prefix in SDIST_INSTALLATION_EXCLUDED_PREFIXES
        if not any(name.startswith(prefix) for name in tracked)
    ]
    if missing_excluded_prefixes:
        raise LicenseCheckError(
            "sdist installation policy names empty tracked prefixes: "
            + ", ".join(missing_excluded_prefixes)
        )
    selected: dict[str, int] = {}
    for name, git_mode in tracked.items():
        if name in SDIST_INSTALLATION_EXCLUDED_FILES or any(
            name.startswith(prefix) for prefix in SDIST_INSTALLATION_EXCLUDED_PREFIXES
        ):
            continue
        if not (
            name in SDIST_INSTALLATION_FILES
            or any(name.startswith(prefix) for prefix in SDIST_INSTALLATION_PREFIXES)
        ):
            continue
        if git_mode == "100644":
            selected[name] = 0o644
        elif git_mode == "100755":
            selected[name] = 0o755
        else:
            raise LicenseCheckError(
                f"sdist installation file has unsupported Git mode {git_mode}: {name}"
            )
    # scikit-build-core freezes the setuptools-scm result into this generated
    # metadata file so builds from an unpacked archive do not need Git history.
    selected["PKG-INFO"] = 0o644
    return selected


def _tracked_sdist_installation_files(source_root: Path) -> set[str]:
    """Return the exact relative file set allowed in a PyPI sdist."""
    return set(_tracked_sdist_installation_manifest(source_root))


def _check_sdist_archive_against_manifest(
    path: Path, source_root: Path, expected: dict[str, int]
) -> None:
    """Reject archive-shape surprises and require expected bytes and modes."""
    file_members: dict[str, tarfile.TarInfo] = {}
    seen: set[str] = set()
    with tarfile.open(path, "r:*") as archive:
        for info in archive.getmembers():
            member_path = PurePosixPath(info.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise LicenseCheckError(
                    f"sdist contains unsafe archive path: {info.name}"
                )
            # Tar member names are POSIX paths even when this checker runs on
            # Windows. Compare their canonical spelling so aliases such as
            # ``root/file`` and ``root/./file`` cannot hide duplicate payload.
            canonical_name = member_path.as_posix()
            if canonical_name in seen:
                raise LicenseCheckError(
                    f"sdist contains duplicate archive member: {info.name}"
                )
            seen.add(canonical_name)
            if info.isdir():
                continue
            if not info.isfile():
                raise LicenseCheckError(
                    "sdist contains non-regular archive member: " + info.name
                )
            file_members[canonical_name] = info

        _check_sdist_installation_payload(set(file_members), set(expected))
        roots = {PurePosixPath(name).parts[0] for name in file_members}
        root = next(iter(roots))
        for relative, expected_mode in expected.items():
            archive_name = f"{root}/{relative}"
            info = file_members[archive_name]
            observed_mode = info.mode & 0o777
            if observed_mode != expected_mode:
                raise LicenseCheckError(
                    "sdist file mode differs: "
                    f"{relative} expected {expected_mode:o}, found {observed_mode:o}"
                )
            if relative == "PKG-INFO":
                continue
            extracted = archive.extractfile(info)
            if extracted is None:
                raise LicenseCheckError(f"cannot read archived file: {archive_name}")
            if extracted.read() != (source_root / relative).read_bytes():
                raise LicenseCheckError(
                    "sdist tracked file bytes differ from source: " + relative
                )


def _check_sdist_archive_payload(path: Path, source_root: Path) -> None:
    """Audit an sdist against the exact tracked installation manifest."""
    _check_sdist_archive_against_manifest(
        path, source_root, _tracked_sdist_installation_manifest(source_root)
    )


def _check_sdist_installation_payload(
    names: set[str], expected_relative_names: set[str]
) -> None:
    """Require the sdist to match the tracked installation allow-surface."""
    paths = [PurePosixPath(name) for name in names]
    roots = {path.parts[0] for path in paths if path.parts}
    if len(roots) != 1:
        raise LicenseCheckError(
            f"sdist must contain exactly one archive root; found {sorted(roots)}"
        )
    root = next(iter(roots))
    relative_names = {
        PurePosixPath(*path.parts[1:]).as_posix()
        for path in paths
        if path.parts[0] == root and len(path.parts) > 1
    }
    missing = sorted(expected_relative_names - relative_names)
    unexpected = sorted(relative_names - expected_relative_names)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append(
                "unexpected repository-only or generated payload "
                + ", ".join(unexpected)
            )
        raise LicenseCheckError(
            "sdist installation payload differs: " + "; ".join(details)
        )


def check_archive(path: Path, source_root: Path | None = None) -> None:
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
    if path.suffix != ".whl":
        if source_root is None:
            raise LicenseCheckError(
                "sdist payload audit requires the tracked source root"
            )
        _check_sdist_archive_payload(path, source_root)
        _check_archived_torch_stable(path, names)
        _check_archived_eigen(path, names)
    elif any(_is_eigen_payload_name(name) for name in names):
        raise LicenseCheckError("wheel must not bundle Web-only Eigen material")
    _check_archived_openblas(path, names, wheel=path.suffix == ".whl")
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
            check_archive(archive.resolve(), source_root=args.source_root.resolve())
    except (LicenseCheckError, OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"license check failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
