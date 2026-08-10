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


@pytest.mark.parametrize(
    ("tag", "extension_path"),
    [
        ("manylinux_2_28_x86_64", "xtbloom/lib/libxtbloom_torch_ext.so"),
        ("macosx_14_0_arm64", "xtbloom/lib/libxtbloom_torch_ext.dylib"),
        ("win_amd64", "xtbloom/bin/xtbloom_torch_ext.dll"),
    ],
)
def test_wheel_keeps_platform_binary_but_excludes_cpp(
    tmp_path: Path, tag: str, extension_path: str
) -> None:
    """Accept each supported runtime extension without its build source."""
    wheel = tmp_path / f"xtbloom-0.1.0-py3-none-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(extension_path, b"native")
    _CHECKER.check_wheel(wheel)


def test_wheel_rejects_packaged_cpp(tmp_path: Path) -> None:
    """Reject a recurrence of copying the extension source into the package."""
    wheel = tmp_path / "xtbloom.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xtbloom/lib/libxtbloom_torch_ext.so", b"ELF")
        archive.writestr("xtbloom/_torch_ext/xtbloom_torch_ext.cpp", b"source")
    with pytest.raises(RuntimeError, match="leaks Torch extension source"):
        _CHECKER.check_wheel(wheel)


@pytest.mark.parametrize(
    "runtime_path",
    [
        "xtbloom/lib/libtorch_cpu.so",
        "xtbloom/lib/libtorch_cpu.dylib",
        "xtbloom/bin/torch_cpu.dll",
        "xtbloom/bin/torch_cpu.lib",
        "xtbloom/bin/torch_cpu_stub.def",
        "xtbloom/bin/torch_cuda.dll",
        "xtbloom/bin/torch_python.dll",
        "xtbloom/lib/libtorch_cpu.so.2.13",
        "xtbloom/lib/librenamed_torch_runtime.so.2",
        "xtbloom/bin/c10_custom.dll",
        "xtbloom/lib/libc10.so",
        "xtbloom/lib/libc10.dylib",
        "xtbloom/lib/libc10_cuda.a",
        "xtbloom/bin/libc10.dll",
        "xtbloom/lib/libtorch_cpu.o",
        "xtbloom/bin/torch_cpu.obj",
        "xtbloom/bin/torch_cpu.ilk",
    ],
)
def test_wheel_rejects_torch_runtime_or_build_stub(
    tmp_path: Path, runtime_path: str
) -> None:
    """Never redistribute the separately installed Torch runtime or stub."""
    wheel = tmp_path / "xtbloom-0.1.0-py3-none-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xtbloom/lib/libxtbloom_torch_ext.so", b"ELF")
        archive.writestr(runtime_path, b"forbidden")
    with pytest.raises(RuntimeError, match="forbidden PyTorch runtime/stub"):
        _CHECKER.check_wheel(wheel)


@pytest.mark.parametrize(
    "tag", ["macosx_10_15_x86_64", "win_arm64", "pyemscripten_2026_0_wasm32"]
)
def test_wheel_accepts_explicitly_absent_extension(tmp_path: Path, tag: str) -> None:
    """Unsupported upstream Torch platforms must ship no untested plugin."""
    wheel = tmp_path / f"xtbloom-0.1.0-py3-none-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xtbloom/__init__.py", b"")
    _CHECKER.check_wheel(wheel, expect="absent")


def test_wheel_rejects_extension_when_absence_is_required(tmp_path: Path) -> None:
    """Keep unsupported desktop/Pyodide wheels honest about Torch support."""
    wheel = tmp_path / "xtbloom-0.1.0-py3-none-win_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("xtbloom/bin/xtbloom_torch_ext.dll", b"PE")
    with pytest.raises(RuntimeError, match="must not contain a Torch extension"):
        _CHECKER.check_wheel(wheel, expect="absent")


@pytest.mark.parametrize(
    "extension_path",
    [
        "junk/libxtbloom_torch_ext.so",
        "xtbloom/lib/libxtbloom_torch_ext.so.backup",
    ],
)
def test_wheel_rejects_misplaced_or_backup_binary(
    tmp_path: Path, extension_path: str
) -> None:
    """Reject extension-like payloads that the runtime loader cannot use."""
    wheel = tmp_path / "xtbloom.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(extension_path, b"ELF")
    with pytest.raises(RuntimeError, match="must contain only"):
        _CHECKER.check_wheel(wheel)


def test_sdist_keeps_nonpackage_cpp(tmp_path: Path) -> None:
    """Accept the build source at its explicit non-package location."""
    sdist = tmp_path / "xtbloom.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"source"
        info = tarfile.TarInfo("xtbloom-0.1.0/src/bindings/torch/xtbloom_torch_ext.cpp")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    _CHECKER.check_sdist(sdist)


@pytest.mark.parametrize(
    "stub_path",
    [
        "xtbloom-0.1.0/generated/torch_cpu_stub.c",
        "xtbloom-0.1.0/generated/torch_cpu_stub.generated.c",
    ],
)
def test_sdist_rejects_generated_torch_stub(tmp_path: Path, stub_path: str) -> None:
    """Source archives retain inputs, never generated platform stub bytes."""
    sdist = tmp_path / "xtbloom.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        source = b"// source"
        source_info = tarfile.TarInfo(
            "xtbloom-0.1.0/src/bindings/torch/xtbloom_torch_ext.cpp"
        )
        source_info.size = len(source)
        archive.addfile(source_info, io.BytesIO(source))
        stub = b"void aoti_torch_stub(void) {}"
        stub_info = tarfile.TarInfo(stub_path)
        stub_info.size = len(stub)
        archive.addfile(stub_info, io.BytesIO(stub))
    with pytest.raises(RuntimeError, match="generated PyTorch runtime/stub"):
        _CHECKER.check_sdist(sdist)


@pytest.mark.parametrize(
    "artifact_path",
    [
        "xtbloom-0.1.0/generated/libc10.so",
        "xtbloom-0.1.0/generated/torch_cpu.obj",
    ],
)
def test_sdist_rejects_native_torch_build_artifact(
    tmp_path: Path, artifact_path: str
) -> None:
    """Source archives must not retain compiled Torch or c10 build outputs."""
    sdist = tmp_path / "xtbloom.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        source = b"// source"
        source_info = tarfile.TarInfo(
            "xtbloom-0.1.0/src/bindings/torch/xtbloom_torch_ext.cpp"
        )
        source_info.size = len(source)
        archive.addfile(source_info, io.BytesIO(source))
        artifact = b"native"
        artifact_info = tarfile.TarInfo(artifact_path)
        artifact_info.size = len(artifact)
        archive.addfile(artifact_info, io.BytesIO(artifact))
    with pytest.raises(RuntimeError, match="generated PyTorch runtime/stub"):
        _CHECKER.check_sdist(sdist)


@pytest.mark.parametrize(
    "wrong_path",
    [
        "xtbloom-0.1.0/python/xtbloom/_torch_ext/xtbloom_torch_ext.cpp",
        "xtbloom-0.1.0/junk/src/bindings/torch/xtbloom_torch_ext.cpp",
    ],
)
def test_sdist_rejects_misplaced_cpp(tmp_path: Path, wrong_path: str) -> None:
    """Reject sdists that retain a second or decoy extension source."""
    sdist = tmp_path / "xtbloom.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "xtbloom-0.1.0/src/bindings/torch/xtbloom_torch_ext.cpp",
            wrong_path,
        ):
            payload = b"source"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="must contain only"):
        _CHECKER.check_sdist(sdist)
