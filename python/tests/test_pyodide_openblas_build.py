"""Tests for the provenance-pinned Pyodide OpenBLAS resolver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RESOLVER_PATH = REPOSITORY / "python" / "ci" / "resolve-pyodide-openblas.py"
SPEC = importlib.util.spec_from_file_location(
    "xtbloom_pyodide_openblas_resolver", RESOLVER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVER)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """Create a compact local release lock/archive with the production schema."""
    root = tmp_path / "source"
    manifest_path = root / "cmake" / "3rdparty" / "manifest.json"
    recipe_relative = RESOLVER.PYODIDE_OPENBLAS_RECIPE_ROOT / "recipe.txt"
    recipe = root / recipe_relative
    license_path = root / "license.txt"
    recipe.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_bytes(b"recipe\n")
    license_path.write_bytes(b"license\n")

    payload = b"\0asmprivate-openblas"
    payload_hash = _sha256(payload)
    archive_path = tmp_path / "upstream.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("libopenblas.so", payload)
    archive_data = archive_path.read_bytes()
    lock_record = {
        "depends": [],
        "file_name": "libopenblas.zip",
        "imports": [],
        "install_dir": "dynlib",
        "name": "libopenblas",
        "package_type": "shared_library",
        "sha256": _sha256(archive_data),
        "unvendored_tests": False,
        "version": "0.3.28",
    }
    lock_path = tmp_path / "pyodide-lock.json"
    lock_path.write_text(json.dumps({"packages": {"libopenblas": lock_record}}))
    manifest = {
        "schema_version": 1,
        "dependency": {"version": "0.3.28", "runtime_dependency": False},
        "lock": {"url": lock_path.as_uri(), "sha256": _sha256(lock_path.read_bytes())},
        "artifact": {
            "filename": "libopenblas.zip",
            "url": archive_path.as_uri(),
            "sha256": _sha256(archive_data),
            "member": "libopenblas.so",
            "member_size": len(payload),
            "member_sha256": payload_hash,
            "private_install_name": f"libxtbloom_openblas-{payload_hash[:8]}.so",
            "adapter_install_name": "libxtbloom_pyodide_lapacke.so",
            "expected_config_prefix": "OpenBLAS 0.3.28",
        },
        "recipe_files": [
            {
                "local": recipe_relative.as_posix(),
                "sha256": _sha256(recipe.read_bytes()),
            }
        ],
        "licenses": [
            {"local": "license.txt", "sha256": _sha256(license_path.read_bytes())}
        ],
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, tmp_path / "cache", payload


def test_resolver_materializes_content_qualified_provider(tmp_path: Path) -> None:
    """Verify the release lock, archive, payload, recipe, and legal records."""
    manifest, cache, payload = _fixture(tmp_path)
    result = RESOLVER.resolve(manifest, cache)
    provider = Path(result["provider_path"])
    assert provider.name.startswith("libxtbloom_openblas-")
    assert provider.read_bytes() == payload

    # The repair step is network-independent and rejects a changed prepared file.
    provider.write_bytes(b"corrupt")
    with pytest.raises(RESOLVER.ResolutionError, match="provider payload differs"):
        RESOLVER.resolve(manifest, cache, require_existing=True)


def test_resolver_rejects_changed_retained_recipe(tmp_path: Path) -> None:
    """Treat the retained Pyodide build recipe as audited source input."""
    manifest, cache, _ = _fixture(tmp_path)
    manifest.parents[2].joinpath(
        RESOLVER.PYODIDE_OPENBLAS_RECIPE_ROOT, "recipe.txt"
    ).write_text("changed\n")
    with pytest.raises(RESOLVER.ResolutionError, match="provenance differs"):
        RESOLVER.resolve(manifest, cache)


def test_resolver_rejects_unreviewed_retained_recipe(tmp_path: Path) -> None:
    """Reject files captured by the sdist wildcard but absent from the manifest."""
    manifest, cache, _ = _fixture(tmp_path)
    manifest.parents[2].joinpath(
        RESOLVER.PYODIDE_OPENBLAS_RECIPE_ROOT, "unexpected.patch"
    ).write_text("unexpected\n")
    with pytest.raises(RESOLVER.ResolutionError, match=r"unexpected.*unexpected.patch"):
        RESOLVER.resolve(manifest, cache)
