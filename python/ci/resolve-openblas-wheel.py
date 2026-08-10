#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resolve and verify the scipy-openblas32 input used by native wheel builds.

The upstream distribution is a build input, not a runtime dependency. This
tool deliberately uses importlib metadata instead of importing
``scipy_openblas32`` because that module loads OpenBLAS into the process-global
namespace as an import side effect. Linux builds expose the verified provider
to auditwheel through a private shim; macOS and Windows copy only the reviewed
dynamic-library cohort into the xTBloom wheel under private names.
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
    """Report a build input that differs from reviewed provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_platform(value: str) -> str:
    aliases = {
        "darwin": "macos",
        "linux": "linux",
        "macos": "macos",
        "win32": "windows",
        "windows": "windows",
    }
    normalized = aliases.get(value.lower())
    if normalized is None:
        raise ResolveError(f"unsupported scipy-openblas32 wheel platform: {value}")
    return normalized


def _normalize_architecture(platform_name: str, value: str) -> str:
    architecture = value.lower()
    if architecture in {"x86_64", "amd64"}:
        return "amd64" if platform_name == "windows" else "x86_64"
    if architecture in {"aarch64", "arm64"}:
        return "aarch64" if platform_name == "linux" else "arm64"
    raise ResolveError(f"unsupported scipy-openblas32 wheel architecture: {value}")


def target_name(platform_name: str, architecture: str) -> str:
    """Return the manifest key for one normalized native wheel target."""
    normalized_platform = _normalize_platform(platform_name)
    normalized_architecture = _normalize_architecture(normalized_platform, architecture)
    return f"{normalized_platform}-{normalized_architecture}"


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the reviewed provenance record and reject unknown schemas."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ResolveError("unsupported scipy-openblas32 manifest schema")
    return manifest


def _distribution_binary_files(
    distribution: importlib.metadata.Distribution,
) -> set[str]:
    files = distribution.files
    if files is None:
        raise ResolveError(
            "scipy-openblas32 distribution has no installed file inventory"
        )

    binaries: set[str] = set()
    for entry in files:
        path = PurePosixPath(str(entry))
        parent = path.parent.as_posix()
        name = path.name.lower()
        if (
            parent == "scipy_openblas32/lib"
            and (".so" in name or name.endswith((".dylib", ".dll")))
        ) or (parent == "scipy_openblas32/.dylibs" and name.endswith(".dylib")):
            binaries.add(path.as_posix())
    return binaries


def resolve_provider(
    manifest: dict[str, Any],
    platform_name: str,
    architecture: str,
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any]:
    """Verify one installed target and return its private provider payload."""
    dependency = manifest["dependency"]
    if distribution.version != dependency["version"]:
        raise ResolveError(
            "scipy-openblas32 version differs from reviewed build input: "
            f"expected {dependency['version']}, found {distribution.version}"
        )

    normalized_platform = _normalize_platform(platform_name)
    normalized_architecture = _normalize_architecture(normalized_platform, architecture)
    target = f"{normalized_platform}-{normalized_architecture}"
    try:
        target_manifest = manifest["targets"][target]
    except KeyError as error:
        raise ResolveError(
            f"unsupported scipy-openblas32 wheel target: {target}"
        ) from error

    records = target_manifest["files"]
    expected_sources = {record["source"] for record in records}
    observed_sources = _distribution_binary_files(distribution)
    if observed_sources != expected_sources:
        missing = sorted(expected_sources - observed_sources)
        unexpected = sorted(observed_sources - expected_sources)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ResolveError(
            "scipy-openblas32 dynamic-library inventory differs: " + "; ".join(details)
        )

    license_record = target_manifest["license"]
    license_path = Path(distribution.locate_file(license_record["source"]))
    if _sha256(license_path) != license_record["sha256"]:
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
        resolved = dict(record)
        resolved["path"] = str(path)
        resolved_files.append(resolved)

    provider_source = target_manifest["provider_source"]
    provider = next(
        (item for item in resolved_files if item["source"] == provider_source), None
    )
    if provider is None or provider.get("role") != "provider":
        raise ResolveError("scipy-openblas32 manifest does not identify one provider")

    return {
        "schema_version": 2,
        "target": target,
        "platform": normalized_platform,
        "architecture": normalized_architecture,
        "bundle_strategy": target_manifest["bundle_strategy"],
        "expected_config_prefix": target_manifest["expected_config_prefix"],
        "source_distribution": dependency["name"],
        "source_version": dependency["version"],
        "provider_path": provider["path"],
        "provider_install_name": provider.get("install_name", ""),
        "files": resolved_files,
    }


def main() -> int:
    """Resolve the reviewed provider from the current build environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--platform", default=platform.system())
    parser.add_argument("--architecture", default=platform.machine())
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        distribution = importlib.metadata.distribution(manifest["dependency"]["name"])
        resolved = resolve_provider(
            manifest, args.platform, args.architecture, distribution
        )
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
