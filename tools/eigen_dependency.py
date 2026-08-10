#!/usr/bin/env python3
"""Verify the pinned Eigen WebAssembly dependency and retained legal records.

The Web build downloads Eigen only when ``XTBLOOM_BUILD_WEB_DEMO=ON``. CMake
verifies the official archive SHA-256 before extraction; offline builds may
provide the same archive through ``XTBLOOM_WEB_EIGEN_ARCHIVE``. This checker
keeps the small repository-side provenance and legal payload byte-exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

MANIFEST_RELPATH = Path("cmake/3rdparty/eigen_manifest.json")
VERSION = "5.0.1"
REVISION = "bc3b39870ecb690a623a3f49149a358b95c5781d"
UPSTREAM_REPOSITORY = "https://gitlab.com/libeigen/eigen"
ARCHIVE_URL = "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz"
ARCHIVE_SIZE = 2_967_272
ARCHIVE_SHA256 = "e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec"
ACQUISITION = {
    "method": "CMake FetchContent for WebAssembly builds only",
    "offline_archive_cache_variable": "XTBLOOM_WEB_EIGEN_ARCHIVE",
    "policy": (
        "The exact official archive is SHA-256 verified before extraction. "
        "Native builds, installs, sdists, and wheels do not fetch or bundle Eigen."
    ),
}
DISTRIBUTION = (
    "Eigen is a downloaded WebAssembly build input. The repository and sdist "
    "retain provenance and exact legal records but not the header tree or archive. "
    "Pages carry compiled code, the legal records, provenance, and an exact source "
    "URL/hash. Native installs and Python wheels exclude all Eigen material."
)
RETAINED_FILES = {
    "LICENSES/eigen/COPYING.MPL2": (
        16727,
        "66a3107d5ad6a058aab753eaac2047ccb2ed0e39465dd0fe5844da3e300d5172",
    ),
    "LICENSES/eigen/COPYING.BSD": (
        1517,
        "51928dce36213c5333ba3172e847d735d4c6e9b7ff2722a326c49067155b82eb",
    ),
    "LICENSES/eigen/COPYING.APACHE": (
        11362,
        "03379001a7b12a2ec997a25554247d985270b353c10d5bafee9ac8d6519820b7",
    ),
    "LICENSES/eigen/COPYING.MINPACK": (
        2193,
        "c87b7f8ee88f6195e91743820c00354833583aef091b72e2d4a49c8e28e798a0",
    ),
    "LICENSES/eigen/COPYING.README": (
        264,
        "db640ff2bd90c6abd6a4d3fbb351e0ee4d555417cf840492054d1cbb2ea85644",
    ),
    "LICENSES/eigen/notices/Half.h": (
        47293,
        "6aef8811305cb4ed25860e3e403f885a11b340b6b73b150db59cb0502b7283f6",
    ),
    "LICENSES/eigen/notices/BFloat16.h": (
        36700,
        "6e1983203d85f4874cb11431b5b41357236ea806fd188cca80135043a9718ebc",
    ),
    "LICENSES/eigen/notices/AlignedBox.h": (
        19095,
        "1249e7e037e5c849eb1a4197c1f051e02dc09be8ce8377db2b5471a16c5e2036",
    ),
    "LICENSES/eigen/notices/InverseSize4.h": (
        13886,
        "619dfeab1fb47d95a647d54f2e187d3aa92cc06f97b1e6a475f72f5f1987e68c",
    ),
}


def _sha256(data: bytes) -> str:
    """Return one lowercase SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def _load_manifest(root: Path) -> dict[str, object]:
    """Load the repository-side dependency manifest."""
    payload = json.loads((root / MANIFEST_RELPATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Eigen manifest root must be an object")
    return payload


def check(args: argparse.Namespace) -> int:
    """Verify pinned metadata, legal bytes, and absence of vendored headers."""
    root = args.source_root.resolve()
    manifest = _load_manifest(root)
    archive = manifest.get("release_archive")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dependency") != "Eigen"
        or manifest.get("version") != VERSION
        or manifest.get("revision") != REVISION
        or manifest.get("upstream_repository") != UPSTREAM_REPOSITORY
        or archive
        != {
            "url": ARCHIVE_URL,
            "size_bytes": ARCHIVE_SIZE,
            "sha256": ARCHIVE_SHA256,
        }
        or manifest.get("acquisition") != ACQUISITION
        or manifest.get("distribution") != DISTRIBUTION
    ):
        raise ValueError("Eigen manifest pinned provenance differs")

    license_info = manifest.get("license")
    if (
        not isinstance(license_info, dict)
        or license_info.get("primary_spdx") != "MPL-2.0"
    ):
        raise ValueError("Eigen manifest license data is missing")
    retained = [
        *license_info.get("records", []),
        *license_info.get("embedded_notices", []),
    ]
    if len(retained) != len(RETAINED_FILES):
        raise ValueError("Eigen manifest must describe nine retained legal files")
    observed_paths: set[str] = set()
    for entry in retained:
        if not isinstance(entry, dict):
            raise ValueError("Eigen manifest has a non-object legal entry")
        relative = entry.get("local_path")
        size = entry.get("size_bytes")
        expected = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative.startswith("LICENSES/eigen/")
            or relative in observed_paths
            or not isinstance(size, int)
            or not isinstance(expected, str)
        ):
            raise ValueError("Eigen manifest has an invalid legal entry")
        observed_paths.add(relative)
        pinned = RETAINED_FILES.get(relative)
        if pinned != (size, expected):
            raise ValueError(f"Eigen manifest legal record differs: {relative}")
        path = root / relative
        data = path.read_bytes()
        if len(data) != size or _sha256(data) != expected:
            raise ValueError(f"retained Eigen legal bytes differ: {relative}")

    if observed_paths != set(RETAINED_FILES):
        raise ValueError("Eigen manifest retained legal-file set differs")

    vendored_root = root / "cmake/3rdparty/eigen"
    if vendored_root.exists() and any(
        path.is_file() for path in vendored_root.rglob("*")
    ):
        raise ValueError("repository must not vendor the Eigen header tree")
    print("Eigen dependency OK: official 5.0.1 archive + 9 retained legal files")  # noqa: T201
    return 0


def verify_archive(args: argparse.Namespace) -> int:
    """Verify a downloaded archive before using it as the offline CMake input."""
    archive = args.archive.resolve()
    data = archive.read_bytes()
    if len(data) != ARCHIVE_SIZE or _sha256(data) != ARCHIVE_SHA256:
        raise ValueError("Eigen archive size or SHA-256 differs")
    with tarfile.open(archive, "r:gz") as payload:
        names = set(payload.getnames())
    root = f"eigen-{VERSION}"
    required = {
        f"{root}/Eigen/Core",
        f"{root}/Eigen/Cholesky",
        f"{root}/Eigen/Eigenvalues",
        *(
            f"{root}/{name}"
            for name in (
                "COPYING.MPL2",
                "COPYING.BSD",
                "COPYING.APACHE",
                "COPYING.MINPACK",
                "COPYING.README",
            )
        ),
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError("Eigen archive is missing " + ", ".join(missing))
    print(f"Eigen archive OK: {archive} ({len(data)} bytes)")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch dependency metadata or archive verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--source-root", type=Path, default=Path("."))
    check_parser.set_defaults(func=check)

    archive_parser = subparsers.add_parser("verify-archive")
    archive_parser.add_argument("archive", type=Path)
    archive_parser.set_defaults(func=verify_archive)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"Eigen dependency check failed: {error}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
