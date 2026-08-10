"""Regression tests for native OpenBLAS wheel inspection helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
INSPECTOR_PATH = REPOSITORY / "python" / "ci" / "inspect-openblas-wheel.py"
SPEC = importlib.util.spec_from_file_location(
    "xtbloom_openblas_wheel_inspector", INSPECTOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
INSPECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECTOR)

_REQUIRED_PROVIDER_SYMBOLS = {
    "scipy_LAPACKE_dpotrf_work",
    "scipy_LAPACKE_dpocon_work",
    "scipy_LAPACKE_dsyevd_work",
    "scipy_cblas_dtrsm",
    "scipy_cblas_dgemm",
    "scipy_openblas_get_config",
    "scipy_openblas_set_num_threads",
    "scipy_openblas_get_num_threads",
}


class FakeArchive:
    """Provide the small ``ZipFile.read`` surface used by platform inspectors."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def read(self, name: str) -> bytes:
        """Return one wheel-member payload."""
        return self.payloads[name]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_pe(*, machine: int, imports: set[str], exports: set[str]) -> bytes:
    """Build a minimal PE32+ image containing named import/export tables.

    The inspector intentionally parses these tables without a third-party PE
    package, so this compact fixture protects the RVA-to-file-offset logic and
    avoids committing upstream binary artifacts solely for unit tests.
    """
    data = bytearray(0x1000)
    pe_offset = 0x80
    optional_size = 0xF0
    optional = pe_offset + 24
    section_table = optional + optional_size
    section_rva = 0x1000
    section_offset = 0x200

    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe_offset)
    data[pe_offset : pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<H", data, pe_offset + 4, machine)
    struct.pack_into("<H", data, pe_offset + 6, 1)
    struct.pack_into("<H", data, pe_offset + 20, optional_size)
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into(
        "<IIII",
        data,
        section_table + 8,
        0x800,
        section_rva,
        0x800,
        section_offset,
    )

    def file_offset(rva: int) -> int:
        return section_offset + rva - section_rva

    export_rva = 0x1000
    export_names_rva = 0x1100
    export_strings_rva = 0x1200
    struct.pack_into("<I", data, optional + 112, export_rva)
    export_offset = file_offset(export_rva)
    ordered_exports = sorted(exports)
    struct.pack_into("<I", data, export_offset + 24, len(ordered_exports))
    struct.pack_into("<I", data, export_offset + 32, export_names_rva)
    cursor_rva = export_strings_rva
    for index, name in enumerate(ordered_exports):
        struct.pack_into(
            "<I", data, file_offset(export_names_rva) + 4 * index, cursor_rva
        )
        encoded = name.encode("ascii") + b"\0"
        cursor = file_offset(cursor_rva)
        data[cursor : cursor + len(encoded)] = encoded
        cursor_rva += len(encoded)

    import_rva = 0x1400
    import_strings_rva = 0x1500
    struct.pack_into("<I", data, optional + 120, import_rva)
    cursor_rva = import_strings_rva
    for index, name in enumerate(sorted(imports)):
        descriptor = file_offset(import_rva) + 20 * index
        struct.pack_into("<IIIII", data, descriptor, 0, 0, 0, cursor_rva, 0)
        encoded = name.encode("ascii") + b"\0"
        cursor = file_offset(cursor_rva)
        data[cursor : cursor + len(encoded)] = encoded
        cursor_rva += len(encoded)
    return bytes(data)


def test_pe_helpers_read_machine_imports_and_exports() -> None:
    """Read the exact architecture and named dependency ABI from PE32+."""
    image = _minimal_pe(
        machine=0x8664,
        imports={"KERNEL32.dll", "api-ms-win-crt-math-l1-1-0.dll"},
        exports={"scipy_cblas_dgemm", "scipy_openblas_get_config"},
    )
    assert INSPECTOR._pe_machine(image) == 0x8664
    assert INSPECTOR._pe_imports(image) == {
        "KERNEL32.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
    }
    assert INSPECTOR._pe_exports(image) == {
        "scipy_cblas_dgemm",
        "scipy_openblas_get_config",
    }


def test_windows_inspector_enforces_exports_and_import_closure() -> None:
    """Accept only the reviewed x64 provider architecture and dependency set."""
    provider = _minimal_pe(
        machine=0x8664,
        imports={"KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"},
        exports=_REQUIRED_PROVIDER_SYMBOLS,
    )
    libxtbloom = _minimal_pe(
        machine=0x8664,
        imports={"KERNEL32.dll"},
        exports=set(),
    )
    provider_name = "xtbloom/bin/xtbloom_openblas-deadbeef.dll"
    archive = FakeArchive(
        {
            "xtbloom/bin/xtbloom.dll": libxtbloom,
            provider_name: provider,
        }
    )
    names = set(archive.payloads)
    target = {
        "architecture": "amd64",
        "files": [
            {
                "role": "provider",
                "install_destination": "bin",
                "install_name": "xtbloom_openblas-deadbeef.dll",
                "size": len(provider),
                "sha256": _sha256(provider),
            }
        ],
    }
    INSPECTOR._inspect_windows(archive, names, target)

    unreviewed = _minimal_pe(
        machine=0x8664,
        imports={"KERNEL32.dll", "USER32.dll"},
        exports=_REQUIRED_PROVIDER_SYMBOLS,
    )
    archive.payloads[provider_name] = unreviewed
    target["files"][0]["size"] = len(unreviewed)
    target["files"][0]["sha256"] = _sha256(unreviewed)
    with pytest.raises(INSPECTOR.InspectionError, match="unreviewed imports"):
        INSPECTOR._inspect_windows(archive, names, target)


def _macos_fixture() -> tuple[FakeArchive, set[str], dict[str, object]]:
    provider_name = "libxtbloom_blas-deadbeef.dylib"
    support_name = "libxbgf-feedface.dylib"
    payloads = {
        "xtbloom/lib/libxtbloom.dylib": b"xtbloom",
        f"xtbloom/lib/{provider_name}": b"rewritten provider",
        f"xtbloom/lib/{support_name}": b"rewritten support",
    }
    target: dict[str, object] = {
        "installed_provider": f"lib/{provider_name}",
        "installed_provider_id": f"@rpath/{provider_name}",
        "files": [
            {
                "role": "provider",
                "install_destination": "lib",
                "install_name": provider_name,
                "install_id": f"@rpath/{provider_name}",
                "sha256": _sha256(b"upstream provider"),
                "load_rewrites": [
                    {
                        "from": "@rpath/libgfortran.5.dylib",
                        "to": f"@rpath/{support_name}",
                    }
                ],
            },
            {
                "role": "support",
                "install_destination": "lib",
                "install_name": support_name,
                "install_id": f"@rpath/{support_name}",
                "sha256": _sha256(b"upstream support"),
                "load_rewrites": [],
            },
        ],
    }
    archive = FakeArchive(payloads)
    return archive, set(payloads), target


def test_macos_inspector_enforces_private_ids_and_complete_rewrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate every renamed image, not only the top-level OpenBLAS dylib."""
    archive, names, target = _macos_fixture()
    records = target["files"]
    ids = {record["install_name"]: record["install_id"] for record in records}
    dependencies = {
        records[0]["install_name"]: [
            records[0]["install_id"],
            records[0]["load_rewrites"][0]["to"],
            "/usr/lib/libSystem.B.dylib",
        ],
        records[1]["install_name"]: [
            records[1]["install_id"],
            "/usr/lib/libSystem.B.dylib",
        ],
        "libxtbloom.dylib": ["/usr/lib/libSystem.B.dylib"],
    }

    def fake_run(command: list[str], description: str) -> str:
        del description
        path = Path(command[-1])
        if command[0] == "nm":
            return "\n".join(sorted(_REQUIRED_PROVIDER_SYMBOLS))
        if command[:2] == ["otool", "-D"]:
            return f"{path}:\n{ids[path.name]}\n"
        if command[:2] == ["otool", "-L"]:
            rendered = "\n".join(
                f"\t{dependency} (compatibility version 1.0.0, current version 1.0.0)"
                for dependency in dependencies[path.name]
            )
            return f"{path}:\n{rendered}\n"
        if command[0] == "codesign":
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(INSPECTOR, "_run", fake_run)
    INSPECTOR._inspect_macos(archive, names, target, "otool", "nm", "codesign")

    dependencies[records[0]["install_name"]].append("@rpath/libgfortran.5.dylib")
    with pytest.raises(INSPECTOR.InspectionError, match="dependency closure differs"):
        INSPECTOR._inspect_macos(archive, names, target, "otool", "nm", "codesign")
