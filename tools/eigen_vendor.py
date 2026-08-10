#!/usr/bin/env python3
"""Vendor and verify the pinned Eigen headers used by the WebAssembly build.

Eigen is a header-only build dependency for ``web/wasm/linalg_eigen.cpp``.
The complete upstream ``Eigen/`` include tree is retained so Emscripten and
native test builds select the same architecture headers without downloading
anything at configure or build time. Regenerate only from the exact extracted
official release archive recorded below::

    python3 tools/eigen_vendor.py generate \
        --source-dir /path/to/extracted/eigen-5.0.1
    python3 tools/eigen_vendor.py check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

VERSION = "5.0.1"
REVISION = "bc3b39870ecb690a623a3f49149a358b95c5781d"
ARCHIVE_URL = "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz"
ARCHIVE_SHA256 = "e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec"
VENDOR_RELPATH = Path("cmake/3rdparty/eigen")
MANIFEST_NAME = "manifest.json"
LICENSE_NAMES = (
    "COPYING.README",
    "COPYING.MPL2",
    "COPYING.BSD",
    "COPYING.APACHE",
    "COPYING.MINPACK",
)
LICENSE_RECORDS = (
    {
        "path": "COPYING.MPL2",
        "spdx": "MPL-2.0",
        "scope": "Primary license for the Eigen source tree.",
    },
    {
        "path": "COPYING.BSD",
        "spdx": "BSD-3-Clause",
        "scope": "BSD and other permissive files identified by upstream.",
    },
    {
        "path": "COPYING.APACHE",
        "spdx": "Apache-2.0",
        "scope": "Apache-licensed files identified by upstream.",
    },
    {
        "path": "COPYING.MINPACK",
        "spdx": "Minpack",
        "scope": (
            "Retained upstream legal record; unsupported/ and its MINPACK "
            "sources are excluded from this vendor tree."
        ),
    },
    {
        "path": "COPYING.README",
        "spdx": "NOASSERTION",
        "scope": "Upstream explanation of the accompanying license records.",
    },
)


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one vendored file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_files(root: Path) -> list[dict[str, object]]:
    """Describe every regular file below the vendor root deterministically."""
    entries: list[dict[str, object]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        if path.name == MANIFEST_NAME:
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def generate(args: argparse.Namespace) -> int:
    """Copy the reviewed Eigen include tree and emit byte-exact provenance."""
    source = args.source_dir.resolve()
    out = args.out.resolve()
    if not (source / "Eigen/Core").is_file():
        raise FileNotFoundError(f"{source} is not an extracted Eigen {VERSION} tree")
    for name in LICENSE_NAMES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Eigen source tree is missing {name}")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source / "Eigen", out / "Eigen")
    for name in LICENSE_NAMES:
        shutil.copy2(source / name, out / name)

    files = _declared_files(out)
    manifest = {
        "schema_version": 1,
        "dependency": "Eigen",
        "version": VERSION,
        "revision": REVISION,
        "upstream_repository": "https://gitlab.com/libeigen/eigen",
        "release_archive": {
            "url": ARCHIVE_URL,
            "sha256": ARCHIVE_SHA256,
        },
        "license": {
            "primary_spdx": "MPL-2.0",
            "records": list(LICENSE_RECORDS),
        },
        "distribution": (
            "Header-only WebAssembly build input. The source tree is retained "
            "in source distributions, is not installed by native CMake, and "
            "is not bundled in Python wheels. The Pages artifact contains "
            "only compiled code plus the applicable license/provenance files."
        ),
        "files": files,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"vendored Eigen {VERSION}: {len(files)} files -> {out}")  # noqa: T201
    return 0


def check(args: argparse.Namespace) -> int:
    """Verify metadata and every vendored byte without network access."""
    out = args.out.resolve()
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"missing {manifest_path}", file=sys.stderr)  # noqa: T201
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": 1,
        "dependency": "Eigen",
        "version": VERSION,
        "revision": REVISION,
        "upstream_repository": "https://gitlab.com/libeigen/eigen",
    }
    errors = [
        f"manifest {key} differs"
        for key, value in expected_header.items()
        if manifest.get(key) != value
    ]
    archive = manifest.get("release_archive", {})
    if archive != {"url": ARCHIVE_URL, "sha256": ARCHIVE_SHA256}:
        errors.append("release archive provenance differs")
    license_info = manifest.get("license", {})
    if license_info != {
        "primary_spdx": "MPL-2.0",
        "records": list(LICENSE_RECORDS),
    }:
        errors.append("license declaration differs")

    declared_entries = manifest.get("files", [])
    declared = {
        entry.get("path"): entry
        for entry in declared_entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    observed = {
        path.relative_to(out).as_posix()
        for path in out.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    errors.extend(f"missing {path}" for path in sorted(set(declared) - observed))
    errors.extend(f"unexpected {path}" for path in sorted(observed - set(declared)))
    for relative in sorted(observed & set(declared)):
        path = out / relative
        entry = declared[relative]
        if entry.get("size_bytes") != path.stat().st_size:
            errors.append(f"size mismatch {relative}")
        if entry.get("sha256") != _sha256(path):
            errors.append(f"hash mismatch {relative}")
    if errors:
        print("Eigen vendor check failed:", file=sys.stderr)  # noqa: T201
        for error in errors:
            print(f"  {error}", file=sys.stderr)  # noqa: T201
        return 1
    print(f"Eigen vendor OK: {len(observed)} files")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch generation or offline verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--source-dir", required=True, type=Path)
    generate_parser.add_argument("--out", default=VENDOR_RELPATH, type=Path)
    generate_parser.set_defaults(func=generate)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--out", default=VENDOR_RELPATH, type=Path)
    check_parser.set_defaults(func=check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
