"""Unit tests for the dependency-free WebAssembly wheel inspector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
INSPECTOR_PATH = REPOSITORY / "python" / "ci" / "inspect-pyodide-openblas-wheel.py"
SPEC = importlib.util.spec_from_file_location(
    "xtbloom_pyodide_openblas_inspector", INSPECTOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)


def _u32(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _name(value: str) -> bytes:
    data = value.encode()
    return _u32(len(data)) + data


def _section(section_id: int, payload: bytes) -> bytes:
    return bytes([section_id]) + _u32(len(payload)) + payload


def _module(
    *,
    needed: list[str],
    runtime_paths: list[str],
    exports: list[str],
    export_signatures: dict[str, tuple[list[str], list[str]]] | None = None,
    memory_info: bytes = b"\xe8\xa5\x04\x04\x18\x00",
    duplicate_needed: bool = False,
) -> bytes:
    value_types = {"i32": b"\x7f", "i64": b"\x7e", "f32": b"\x7d", "f64": b"\x7c"}

    def string_list(values: list[str]) -> bytes:
        return _u32(len(values)) + b"".join(_name(value) for value in values)

    def function_type(name: str) -> bytes:
        parameters, results = (export_signatures or {}).get(name, ([], []))
        return (
            b"\x60"
            + _u32(len(parameters))
            + b"".join(value_types[value] for value in parameters)
            + _u32(len(results))
            + b"".join(value_types[value] for value in results)
        )

    dylink = _name("dylink.0")
    dylink += _u32(1) + _u32(len(memory_info)) + memory_info
    for subsection, values in ((2, needed), (5, runtime_paths)):
        body = string_list(values)
        dylink += _u32(subsection) + _u32(len(body)) + body
    if duplicate_needed:
        body = string_list(["libopenblas-shadow.so"])
        dylink += _u32(2) + _u32(len(body)) + body
    export_section = _u32(len(exports)) + b"".join(
        _name(name) + b"\x00" + _u32(index) for index, name in enumerate(exports)
    )
    type_section = _u32(len(exports)) + b"".join(
        function_type(name) for name in exports
    )
    function_section = _u32(len(exports)) + b"".join(
        _u32(index) for index in range(len(exports))
    )
    return (
        b"\0asm\x01\0\0\0"
        + _section(0, dylink)
        + _section(1, type_section)
        + _section(3, function_section)
        + _section(7, export_section)
    )


def test_wasm_inspector_reads_dylink_and_exports() -> None:
    """Read exact NEEDED names, runtime paths, and exported functions."""
    module = _module(
        needed=["libxtbloom_openblas-deadbeef.so"],
        runtime_paths=["$ORIGIN/../../xtbloom.libs"],
        exports=["dpotrf_", "cblas_dgemm"],
    )
    assert INSPECTOR._dylink(module) == (
        ["libxtbloom_openblas-deadbeef.so"],
        ["$ORIGIN/../../xtbloom.libs"],
    )
    assert INSPECTOR._exports(module) == {"dpotrf_", "cblas_dgemm"}
    assert INSPECTOR._function_export_signatures(module) == {
        "dpotrf_": ((), ()),
        "cblas_dgemm": ((), ()),
    }

    changed_paths = _module(
        needed=[], runtime_paths=["$ORIGIN"], exports=["dpotrf_", "cblas_dgemm"]
    )
    assert INSPECTOR._repair_stable_sha256(module) == INSPECTOR._repair_stable_sha256(
        changed_paths
    )
    changed_memory = _module(
        needed=["libxtbloom_openblas-deadbeef.so"],
        runtime_paths=["$ORIGIN/../../xtbloom.libs"],
        exports=["dpotrf_", "cblas_dgemm"],
        memory_info=b"\x00",
    )
    assert INSPECTOR._repair_stable_sha256(module) != INSPECTOR._repair_stable_sha256(
        changed_memory
    )


def test_wasm_inspector_rejects_truncation() -> None:
    """Fail closed when a repaired module has an invalid section extent."""
    with pytest.raises(INSPECTOR.InspectionError, match="truncated"):
        INSPECTOR._sections(b"\0asm\x01\0\0\0\x07\x10\x00")


def test_wasm_inspector_rejects_duplicate_dylink_lists() -> None:
    """Do not let a later NEEDED list hide an earlier dependency edge."""
    module = _module(
        needed=["libxtbloom_openblas-deadbeef.so"],
        runtime_paths=["$ORIGIN"],
        exports=["dpotrf_"],
        duplicate_needed=True,
    )
    with pytest.raises(INSPECTOR.InspectionError, match=r"duplicate.*subsection 2"):
        INSPECTOR._dylink(module)


def _inspection_fixture(
    tmp_path: Path,
    *,
    extra_openblas: bool = False,
    main_needed: list[str] | None = None,
    main_runtime_paths: list[str] | None = None,
    cblas_result: str = "i32",
) -> tuple[Path, Path]:
    """Build one compact repaired-wheel fixture with real legal filenames."""
    root = tmp_path / "source"
    manifest_path = root / "cmake" / "3rdparty" / "manifest.json"
    license_path = root / "LICENSES" / "OpenBLAS-0.3.28-BSD-3-Clause.txt"
    manifest_path.parent.mkdir(parents=True)
    license_path.parent.mkdir(parents=True)
    license_path.write_bytes(b"OpenBLAS license\n")

    provider_name = "libxtbloom_openblas-deadbeef.so"
    adapter_name = "libxtbloom_pyodide_lapacke.so"
    provider_exports = [
        "dpotrf_",
        "dpocon_",
        "dsyevd_",
        "cblas_dgemm",
        "cblas_dtrsm",
        "openblas_get_config",
        "openblas_set_num_threads_local",
    ]
    provider_signatures = {
        "dpotrf_": (["i32"] * 5, ["i32"]),
        "dpocon_": (["i32"] * 9, ["i32"]),
        "dsyevd_": (["i32"] * 11, ["i32"]),
        "cblas_dgemm": (
            [
                "i32",
                "i32",
                "i32",
                "i32",
                "i32",
                "i32",
                "f64",
                "i32",
                "i32",
                "i32",
                "i32",
                "f64",
                "i32",
                "i32",
            ],
            [cblas_result] if cblas_result else [],
        ),
        "cblas_dtrsm": (
            [
                "i32",
                "i32",
                "i32",
                "i32",
                "i32",
                "i32",
                "i32",
                "f64",
                "i32",
                "i32",
                "i32",
                "i32",
            ],
            [cblas_result] if cblas_result else [],
        ),
        "openblas_get_config": ([], ["i32"]),
        "openblas_set_num_threads_local": (["i32"], ["i32"]),
    }
    adapter_exports = [
        "xtbloom_pyodide_LAPACKE_dpotrf_work",
        "xtbloom_pyodide_LAPACKE_dpocon_work",
        "xtbloom_pyodide_LAPACKE_dsyevd_work",
        "xtbloom_pyodide_cblas_dtrsm",
        "xtbloom_pyodide_cblas_dgemm",
        "xtbloom_pyodide_openblas_get_config",
        "xtbloom_pyodide_openblas_set_num_threads_local",
        "xtbloom_pyodide_openblas_dependency_anchor",
    ]
    main = _module(
        needed=main_needed or [],
        runtime_paths=(
            ["$ORIGIN/../../xtbloom.libs"]
            if main_runtime_paths is None
            else main_runtime_paths
        ),
        exports=["xtbloom_version_string"],
    )
    adapter = _module(
        needed=[provider_name],
        runtime_paths=["$ORIGIN/../../xtbloom.libs"],
        exports=adapter_exports,
    )
    provider = _module(
        needed=[],
        runtime_paths=["$ORIGIN"],
        exports=provider_exports,
        export_signatures=provider_signatures,
    )
    manifest = {
        "artifact": {
            "filename": "libopenblas-0.3.28.zip",
            "private_install_name": provider_name,
            "adapter_install_name": adapter_name,
            "member_repair_stable_sha256": INSPECTOR._repair_stable_sha256(provider),
            "required_exports": provider_exports,
            "required_export_signatures": {
                name: {
                    "parameters": parameters,
                    "results": (
                        ["i32"] if name in {"cblas_dgemm", "cblas_dtrsm"} else results
                    ),
                }
                for name, (parameters, results) in provider_signatures.items()
            },
        },
        "licenses": [
            {
                "name": "OpenBLAS BSD-3-Clause",
                "local": "LICENSES/OpenBLAS-0.3.28-BSD-3-Clause.txt",
                "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wheel = tmp_path / "xtbloom-0-py3-none-pyodide_2026_0_wasm32.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "xtbloom-0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: xtbloom\nVersion: 0\n",
        )
        archive.writestr(
            "xtbloom-0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-pyodide_2026_0_wasm32\n",
        )
        archive.writestr("xtbloom/lib/libxtbloom.so", main)
        archive.writestr(f"xtbloom/lib/{adapter_name}", adapter)
        archive.writestr(f"xtbloom.libs/{provider_name}", provider)
        archive.writestr(
            "xtbloom/share/licenses/xtbloom/provenance/pyodide_openblas_manifest.json",
            manifest_path.read_bytes(),
        )
        archive.writestr(
            "xtbloom/share/licenses/xtbloom/third-party/OpenBLAS-0.3.28-BSD-3-Clause.txt",
            license_path.read_bytes(),
        )
        if extra_openblas:
            archive.writestr("xtbloom.libs/libopenblas.so", provider)
    return wheel, manifest_path


def test_full_wheel_inspection_allows_openblas_legal_records(tmp_path: Path) -> None:
    """Legal/provenance filenames containing OpenBLAS are not binary payloads."""
    wheel, manifest = _inspection_fixture(tmp_path)
    INSPECTOR.inspect(wheel, manifest)


def test_full_wheel_inspection_rejects_extra_openblas_module(tmp_path: Path) -> None:
    """Reject an additional generic provider even when the private one is intact."""
    wheel, manifest = _inspection_fixture(tmp_path, extra_openblas=True)
    with pytest.raises(INSPECTOR.InspectionError, match="forbidden payloads"):
        INSPECTOR.inspect(wheel, manifest)


@pytest.mark.parametrize(
    "dependency",
    [
        "libblas.so",
        "liblapack.so",
        "libscipy_linalg.so",
        "libopenblas.so",
        "libhostmath.so",
    ],
)
def test_full_wheel_inspection_rejects_main_provider_dependency(
    tmp_path: Path, dependency: str
) -> None:
    """Keep libxtbloom independent from generic or host linear algebra DSOs."""
    wheel, manifest = _inspection_fixture(tmp_path, main_needed=[dependency])
    with pytest.raises(INSPECTOR.InspectionError, match="libxtbloom linkage differs"):
        INSPECTOR.inspect(wheel, manifest)


@pytest.mark.parametrize(
    "runtime_paths",
    [[], ["$ORIGIN"], ["$ORIGIN/../../xtbloom.libs", "$ORIGIN"]],
)
def test_full_wheel_inspection_rejects_main_runtime_path(
    tmp_path: Path, runtime_paths: list[str]
) -> None:
    """Require auditwheel's exact private-cohort path on libxtbloom."""
    wheel, manifest = _inspection_fixture(tmp_path, main_runtime_paths=runtime_paths)
    with pytest.raises(INSPECTOR.InspectionError, match="libxtbloom linkage differs"):
        INSPECTOR.inspect(wheel, manifest)


def test_full_wheel_inspection_rejects_provider_function_type(tmp_path: Path) -> None:
    """Reject the native-void CBLAS signature that traps through WASM dispatch."""
    wheel, manifest = _inspection_fixture(tmp_path, cblas_result="")
    with pytest.raises(
        INSPECTOR.InspectionError, match="provider function signatures differ"
    ):
        INSPECTOR.inspect(wheel, manifest)
