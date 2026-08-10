"""Tests for the build-only scipy-openblas32 resolver."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RESOLVER_PATH = REPOSITORY / "python" / "ci" / "resolve-openblas-wheel.py"
SPEC = importlib.util.spec_from_file_location(
    "xtbloom_openblas_resolver", RESOLVER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVER)


class FakeDistribution:
    """Minimal importlib.metadata distribution used by the resolver."""

    def __init__(self, root: Path, version: str, files: list[str]) -> None:
        self.root = root
        self.version = version
        self.files = files

    def locate_file(self, relative: str) -> Path:
        """Resolve one distribution-relative test payload."""
        return self.root / relative


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(root: Path) -> tuple[dict, FakeDistribution]:
    license_bytes = b"reviewed license\n"
    provider_bytes = b"\x7fELFprovider"
    support_bytes = b"\x7fELFsupport"
    payloads = {
        "scipy_openblas32/lib/libscipy_openblas.so": provider_bytes,
        "scipy_openblas32/lib/libgfortran-test.so.5": support_bytes,
        "scipy_openblas32-1.2.3.dist-info/licenses/LICENSE.txt": license_bytes,
    }
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    files = list(payloads)
    manifest = {
        "schema_version": 2,
        "dependency": {"name": "scipy-openblas32", "version": "1.2.3"},
        "targets": {
            "linux-x86_64": {
                "platform": "linux",
                "architecture": "x86_64",
                "bundle_strategy": "auditwheel-shim",
                "expected_config_prefix": "OpenBLAS 1.2.3",
                "provider_source": "scipy_openblas32/lib/libscipy_openblas.so",
                "license": {
                    "source": "scipy_openblas32-1.2.3.dist-info/licenses/LICENSE.txt",
                    "sha256": _sha256(license_bytes),
                },
                "files": [
                    {
                        "source": "scipy_openblas32/lib/libscipy_openblas.so",
                        "role": "provider",
                        "size": len(provider_bytes),
                        "sha256": _sha256(provider_bytes),
                    },
                    {
                        "source": "scipy_openblas32/lib/libgfortran-test.so.5",
                        "role": "support",
                        "size": len(support_bytes),
                        "sha256": _sha256(support_bytes),
                    },
                ],
            }
        },
    }
    return manifest, FakeDistribution(root, "1.2.3", files)


def test_resolver_accepts_exact_payload_without_importing_package() -> None:
    """Return the absolute provider path after validating every ELF byte."""
    with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-resolver-") as directory:
        root = Path(directory)
        manifest, distribution = _fixture(root)
        result = RESOLVER.resolve_provider(manifest, "linux", "amd64", distribution)
        assert result["architecture"] == "x86_64"
        assert result["target"] == "linux-x86_64"
        assert result["provider_path"] == str(
            (root / "scipy_openblas32/lib/libscipy_openblas.so").resolve()
        )


def test_resolver_rejects_an_unreviewed_extra_elf() -> None:
    """Do not let a changed upstream wheel silently expand redistribution."""
    with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-resolver-") as directory:
        root = Path(directory)
        manifest, distribution = _fixture(root)
        extra = "scipy_openblas32/lib/libunexpected.so"
        (root / extra).write_bytes(b"\x7fELFextra")
        distribution.files.append(extra)
        with pytest.raises(RESOLVER.ResolveError, match="unexpected"):
            RESOLVER.resolve_provider(manifest, "linux", "x86_64", distribution)


def test_resolver_rejects_a_changed_version() -> None:
    """Pin the reviewed symbol and local-thread-control ABI exactly."""
    with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-resolver-") as directory:
        root = Path(directory)
        manifest, distribution = _fixture(root)
        distribution.version = "1.2.4"
        with pytest.raises(RESOLVER.ResolveError, match="version differs"):
            RESOLVER.resolve_provider(manifest, "linux", "x86_64", distribution)


@pytest.mark.parametrize(
    ("platform_name", "architecture", "expected"),
    [
        ("Linux", "AMD64", "linux-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
        ("Windows", "AMD64", "windows-amd64"),
        ("win32", "aarch64", "windows-arm64"),
    ],
)
def test_target_name_normalizes_native_platform_aliases(
    platform_name: str, architecture: str, expected: str
) -> None:
    """Select the same manifest target spellings CMake uses on hosted runners."""
    assert RESOLVER.target_name(platform_name, architecture) == expected
