#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare the provenance-pinned Pyodide OpenBLAS wheel build input.

The official Pyodide package archive contains only ``libopenblas.so``. This
resolver verifies the release lock, archive, payload, retained recipe closure,
and license bytes before copying that payload to a content-qualified private
filename. The ZIP and WebAssembly binary remain build outputs; source and
sdist artifacts carry only the manifest, recipe source, and legal records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PYODIDE_OPENBLAS_RECIPE_ROOT = Path("cmake/3rdparty/pyodide-openblas/recipe")


class ResolutionError(RuntimeError):
    """Report an unreviewed or corrupted Pyodide provider input."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_local_files(root: Path, manifest: dict[str, object]) -> None:
    """Require every retained recipe and notice to match pinned upstream bytes."""
    expected_recipe_files: set[str] = set()
    for group in ("recipe_files", "licenses"):
        records = manifest.get(group)
        if not isinstance(records, list) or not records:
            raise ResolutionError(f"manifest {group} must be a nonempty list")
        for record in records:
            if not isinstance(record, dict):
                raise ResolutionError(f"manifest {group} entry is invalid")
            relative = record.get("local")
            expected = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ResolutionError(f"manifest {group} entry is incomplete")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ResolutionError(
                    f"retained Pyodide provenance is not a regular file: {relative}"
                )
            if _sha256(path.read_bytes()) != expected:
                raise ResolutionError(
                    f"retained Pyodide provenance differs: {relative}"
                )
            if group == "recipe_files":
                expected_recipe_files.add(Path(relative).as_posix())

    recipe_root = root / PYODIDE_OPENBLAS_RECIPE_ROOT
    if recipe_root.is_symlink() or not recipe_root.is_dir():
        raise ResolutionError("retained Pyodide recipe root is not a directory")
    observed_recipe_files: set[str] = set()
    for path in recipe_root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ResolutionError(
                f"retained Pyodide recipe entry is a symlink: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ResolutionError(
                f"retained Pyodide recipe entry is not a regular file: {relative}"
            )
        observed_recipe_files.add(relative)
    if observed_recipe_files != expected_recipe_files:
        missing = sorted(expected_recipe_files - observed_recipe_files)
        unexpected = sorted(observed_recipe_files - expected_recipe_files)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ResolutionError(
            "retained Pyodide recipe file set differs: " + "; ".join(details)
        )


def _download_verified(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    require_existing: bool,
) -> bytes:
    """Return exact cached bytes, downloading atomically when permitted."""
    if destination.is_file():
        data = destination.read_bytes()
        if _sha256(data) == expected_sha256:
            return data
        if require_existing:
            raise ResolutionError(f"cached file hash differs: {destination}")
    elif require_existing:
        raise ResolutionError(f"required cached file is missing: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", delete=False
        ) as temporary,
    ):
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    data = temporary_path.read_bytes()
    if _sha256(data) != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise ResolutionError(f"downloaded file hash differs: {url}")
    temporary_path.replace(destination)
    return data


def resolve(
    manifest_path: Path,
    cache_dir: Path,
    *,
    require_existing: bool = False,
) -> dict[str, object]:
    """Verify and materialize the reviewed private WebAssembly provider."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ResolutionError("Pyodide OpenBLAS manifest schema is unsupported")
    root = manifest_path.parents[2]
    _verified_local_files(root, manifest)

    lock = manifest.get("lock")
    artifact = manifest.get("artifact")
    dependency = manifest.get("dependency")
    if not isinstance(lock, dict) or not isinstance(artifact, dict):
        raise ResolutionError("Pyodide OpenBLAS manifest omits lock or artifact data")
    if (
        not isinstance(dependency, dict)
        or dependency.get("runtime_dependency") is not False
    ):
        raise ResolutionError("Pyodide OpenBLAS must remain a build-only dependency")

    lock_data = _download_verified(
        str(lock["url"]),
        cache_dir / "pyodide-lock.json",
        str(lock["sha256"]),
        require_existing=require_existing,
    )
    lock_document = json.loads(lock_data)
    package = lock_document.get("packages", {}).get("libopenblas")
    expected_lock_record = {
        "depends": [],
        "file_name": artifact["filename"],
        "imports": [],
        "install_dir": "dynlib",
        "name": "libopenblas",
        "package_type": "shared_library",
        "sha256": artifact["sha256"],
        "unvendored_tests": False,
        "version": dependency["version"],
    }
    if package != expected_lock_record:
        raise ResolutionError(
            "Pyodide lock libopenblas record differs from the manifest"
        )

    archive_data = _download_verified(
        str(artifact["url"]),
        cache_dir / str(artifact["filename"]),
        str(artifact["sha256"]),
        require_existing=require_existing,
    )
    archive_path = cache_dir / str(artifact["filename"])
    with zipfile.ZipFile(archive_path) as archive:
        members = [entry for entry in archive.infolist() if not entry.is_dir()]
        if [entry.filename for entry in members] != [artifact["member"]]:
            raise ResolutionError("Pyodide OpenBLAS archive member set differs")
        payload = archive.read(str(artifact["member"]))
    if len(payload) != artifact["member_size"]:
        raise ResolutionError("Pyodide OpenBLAS payload size differs")
    if _sha256(payload) != artifact["member_sha256"]:
        raise ResolutionError("Pyodide OpenBLAS payload hash differs")
    if _sha256(archive_data) != artifact["sha256"]:
        raise ResolutionError(
            "Pyodide OpenBLAS cached archive changed during resolution"
        )

    install_name = str(artifact["private_install_name"])
    if artifact["member_sha256"][:8] not in install_name:
        raise ResolutionError("private provider name is not content-qualified")
    provider_path = cache_dir / install_name
    provider_exists = provider_path.is_file()
    provider_matches = provider_exists and provider_path.read_bytes() == payload
    if provider_exists and not provider_matches and require_existing:
        raise ResolutionError("cached private provider payload differs")
    if not provider_matches:
        provider_path.write_bytes(payload)

    return {
        "provider_path": str(provider_path.resolve()),
        "provider_install_name": install_name,
        "adapter_install_name": artifact["adapter_install_name"],
        "expected_config_prefix": artifact["expected_config_prefix"],
        "payload_sha256": artifact["member_sha256"],
        "payload_size": artifact["member_size"],
    }


def main() -> int:
    """Resolve the reviewed provider and print its build metadata as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="verify the prepared cache without performing network downloads",
    )
    args = parser.parse_args()
    try:
        result = resolve(
            args.manifest,
            args.cache_dir.resolve(),
            require_existing=args.require_existing,
        )
    except (
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        ResolutionError,
    ) as error:
        parser.error(str(error))
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
