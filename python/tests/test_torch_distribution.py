"""Tests for the Torch extension source/binary distribution boundary."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "ci" / "check-torch-distribution.py"
_SPEC = importlib.util.spec_from_file_location("check_torch_distribution", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)


def test_wheel_keeps_binary_but_excludes_cpp(tmp_path: Path) -> None:
    """Accept the runtime extension without its build source."""
    wheel = tmp_path / "gpuxtb.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gpuxtb/lib/libgpuxtb_torch_ext.so", b"ELF")
    _CHECKER.check_wheel(wheel)


def test_wheel_rejects_packaged_cpp(tmp_path: Path) -> None:
    """Reject a recurrence of copying the extension source into the package."""
    wheel = tmp_path / "gpuxtb.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gpuxtb/lib/libgpuxtb_torch_ext.so", b"ELF")
        archive.writestr("gpuxtb/_torch_ext/gpuxtb_torch_ext.cpp", b"source")
    with pytest.raises(RuntimeError, match="leaks Torch extension source"):
        _CHECKER.check_wheel(wheel)


@pytest.mark.parametrize(
    "extension_path",
    [
        "junk/libgpuxtb_torch_ext.so",
        "gpuxtb/lib/libgpuxtb_torch_ext.so.backup",
    ],
)
def test_wheel_rejects_misplaced_or_backup_binary(
    tmp_path: Path, extension_path: str
) -> None:
    """Reject extension-like payloads that the runtime loader cannot use."""
    wheel = tmp_path / "gpuxtb.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(extension_path, b"ELF")
    with pytest.raises(RuntimeError, match="must contain only"):
        _CHECKER.check_wheel(wheel)


def test_sdist_keeps_nonpackage_cpp(tmp_path: Path) -> None:
    """Accept the build source at its explicit non-package location."""
    sdist = tmp_path / "gpuxtb.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"source"
        info = tarfile.TarInfo("gpuxtb-0.1.0/python/src/gpuxtb_torch_ext.cpp")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    _CHECKER.check_sdist(sdist)


@pytest.mark.parametrize(
    "wrong_path",
    [
        "gpuxtb-0.1.0/python/gpuxtb/_torch_ext/gpuxtb_torch_ext.cpp",
        "gpuxtb-0.1.0/junk/python/src/gpuxtb_torch_ext.cpp",
    ],
)
def test_sdist_rejects_misplaced_cpp(tmp_path: Path, wrong_path: str) -> None:
    """Reject sdists that retain a second or decoy extension source."""
    sdist = tmp_path / "gpuxtb.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "gpuxtb-0.1.0/python/src/gpuxtb_torch_ext.cpp",
            wrong_path,
        ):
            payload = b"source"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="must contain only"):
        _CHECKER.check_sdist(sdist)
