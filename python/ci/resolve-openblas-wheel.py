#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve and verify the scipy-openblas32 input used by Linux wheel builds.

The upstream distribution is a build input, not a runtime dependency. This
tool deliberately uses importlib.metadata instead of importing
``scipy_openblas32`` because that module loads OpenBLAS into the process-global
namespace as an import side effect. CMake links a private shim to the verified
provider; auditwheel then vendors and collision-renames its complete ELF
dependency closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path, PurePosixPath
from typing import Any


class ResolveError(RuntimeError):
    """Report a build input that differs from the reviewed provider cohort."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_architecture(value: str) -> str:
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    normalized = aliases.get(value.lower(), value.lower())
    if normalized not in {"x86_64", "aarch64"}:
        raise ResolveError(f"unsupported scipy-openblas32 wheel architecture: {value}")
    return normalized


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the reviewed provenance record and reject unknown schemas."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ResolveError("unsupported scipy-openblas32 manifest schema")
    return manifest


def _distribution_elf_files(distribution: importlib.metadata.Distribution) -> set[str]:
    files = distribution.files
    if files is None:
        raise ResolveError(
            "scipy-openblas32 distribution has no installed file inventory"
        )
    return {
        PurePosixPath(str(entry)).as_posix()
        for entry in files
        if PurePosixPath(str(entry)).parent.as_posix() == "scipy_openblas32/lib"
        and ".so" in PurePosixPath(str(entry)).name
    }


def resolve_provider(
    manifest: dict[str, Any],
    architecture: str,
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any]:
    """Verify one installed architecture and return its absolute provider path."""
    dependency = manifest["dependency"]
    if distribution.version != dependency["version"]:
        raise ResolveError(
            "scipy-openblas32 version differs from reviewed build input: "
            f"expected {dependency['version']}, found {distribution.version}"
        )

    architecture = _normalize_architecture(architecture)
    arch_manifest = manifest["architectures"][architecture]
    records = arch_manifest["files"]
    expected_sources = {record["source"] for record in records}
    observed_sources = _distribution_elf_files(distribution)
    if observed_sources != expected_sources:
        missing = sorted(expected_sources - observed_sources)
        unexpected = sorted(observed_sources - expected_sources)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ResolveError(
            "scipy-openblas32 ELF inventory differs: " + "; ".join(details)
        )

    source = manifest["source"]
    license_path = Path(distribution.locate_file(source["license_source"]))
    if _sha256(license_path) != source["license_sha256"]:
        raise ResolveError(
            "scipy-openblas32 license differs from reviewed upstream bytes"
        )

    resolved_files: list[dict[str, Any]] = []
    for record in records:
        path = Path(distribution.locate_file(record["source"])).resolve()
        if not path.is_file() or path.is_symlink():
            raise ResolveError(
                f"scipy-openblas32 payload is not a regular file: {record['source']}"
            )
        if path.stat().st_size != record["size"] or _sha256(path) != record["sha256"]:
            raise ResolveError(
                "scipy-openblas32 payload differs from reviewed bytes: "
                f"{record['source']}"
            )
        resolved_files.append(
            {
                "source": record["source"],
                "path": str(path),
                "sha256": record["sha256"],
            }
        )

    provider = next(
        item
        for item in resolved_files
        if item["source"].endswith("/libscipy_openblas.so")
    )
    return {
        "schema_version": 1,
        "architecture": architecture,
        "source_distribution": dependency["name"],
        "source_version": dependency["version"],
        "provider_path": provider["path"],
        "files": resolved_files,
    }


def main() -> int:
    """Resolve the reviewed provider from the current build environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--architecture", default=platform.machine())
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        distribution = importlib.metadata.distribution(manifest["dependency"]["name"])
        resolved = resolve_provider(manifest, args.architecture, distribution)
    except (
        ResolveError,
        importlib.metadata.PackageNotFoundError,
        OSError,
        KeyError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(resolved, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
