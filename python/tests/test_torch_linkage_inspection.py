"""Unit tests for cross-platform Torch extension linkage inspection."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).parents[2] / "tests" / "abi" / "check_torch_extension_linkage.py"
)
_SPEC = importlib.util.spec_from_file_location("check_torch_extension_linkage", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKER)


def test_elf_checker_accepts_standard_math_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow the separate libm edge retained by hosted manylinux compilers."""
    library = tmp_path / "libxtbloom_torch_ext.so"
    outputs = iter(
        [
            "\n".join(
                f" (NEEDED) Shared library: [{name}]"
                for name in (
                    "libtorch_cpu.so",
                    "libstdc++.so.6",
                    "libgcc_s.so.1",
                    "libm.so.6",
                    "libc.so.6",
                )
            ),
            "                 U aoti_torch_get_data_ptr\n",
        ]
    )
    monkeypatch.setattr(_CHECKER, "_run", lambda command: next(outputs))
    _CHECKER._check_elf(
        library,
        {"aoti_torch_get_data_ptr"},
        readelf="readelf",
        nm="nm",
    )


def test_stable_symbol_parser_accepts_macho_leading_underscore() -> None:
    """Normalize the underscore that Mach-O tooling adds to C symbol names."""
    assert _CHECKER._stable_symbols(
        "_aoti_torch_get_data_ptr\n_torch_library_impl\n___gxx_personality_v0\n"
    ) == {"aoti_torch_get_data_ptr", "torch_library_impl"}


def test_macho_checker_removes_verified_self_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat LC_ID_DYLIB as identity, not as a provider dependency."""
    library = tmp_path / "libxtbloom_torch_ext.dylib"
    outputs = iter(
        [
            f"{library}:\n@rpath/{library.name}\n",
            (
                f"{library}:\n"
                f"\t@rpath/{library.name} (compatibility version 0.0.0, "
                "current version 0.0.0)\n"
                "\t@rpath/libtorch_cpu.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
                "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
                "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
            ),
            "Load command 0\n      cmd LC_SEGMENT_64\n",
            "_aoti_torch_get_data_ptr\n",
        ]
    )
    monkeypatch.setattr(_CHECKER, "_run", lambda command: next(outputs))
    _CHECKER._check_macho(
        library,
        {"aoti_torch_get_data_ptr"},
        otool="otool",
        nm="nm",
    )


def test_macho_checker_rejects_unreviewed_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject even non-Torch dylibs outside the complete reviewed load set."""
    library = tmp_path / "libxtbloom_torch_ext.dylib"
    outputs = iter(
        [
            f"{library}:\n@rpath/{library.name}\n",
            (
                f"{library}:\n"
                f"\t@rpath/{library.name} (compatibility version 0.0.0, "
                "current version 0.0.0)\n"
                "\t@rpath/libtorch_cpu.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
                "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
                "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
                "\t@rpath/libevil.dylib (compatibility version 1.0.0, "
                "current version 1.0.0)\n"
            ),
        ]
    )
    monkeypatch.setattr(_CHECKER, "_run", lambda command: next(outputs))
    with pytest.raises(SystemExit, match="unexpected Mach-O dependencies"):
        _CHECKER._check_macho(
            library,
            {"aoti_torch_get_data_ptr"},
            otool="otool",
            nm="nm",
        )


def _minimal_pe_imports(
    dll: str,
    symbols: list[str],
    *,
    machine: int = 0x8664,
    ordinal: bool = False,
) -> bytes:
    """Build a minimal PE32+ image with one named import descriptor."""
    data = bytearray(0x800)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"

    coff = 0x84
    struct.pack_into("<H", data, coff, machine)
    struct.pack_into("<H", data, coff + 2, 1)  # one section
    struct.pack_into("<H", data, coff + 16, 0xF0)
    optional = coff + 20
    struct.pack_into("<H", data, optional, 0x20B)  # PE32+
    struct.pack_into("<II", data, optional + 112 + 8, 0x1000, 0x200)

    section = optional + 0xF0
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x500, 0x1000, 0x500, 0x200)

    def offset(rva: int) -> int:
        return 0x200 + rva - 0x1000

    dll_rva = 0x1080
    thunk_rva = 0x1100
    struct.pack_into("<IIIII", data, offset(0x1000), thunk_rva, 0, 0, dll_rva, 0x1180)
    encoded_dll = dll.encode("ascii") + b"\0"
    data[offset(dll_rva) : offset(dll_rva) + len(encoded_dll)] = encoded_dll

    for index, symbol in enumerate(symbols):
        name_rva = 0x1200 + index * 0x60
        thunk = (1 << 63) | index if ordinal else name_rva
        struct.pack_into("<Q", data, offset(thunk_rva) + index * 8, thunk)
        encoded = b"\0\0" + symbol.encode("ascii") + b"\0"
        data[offset(name_rva) : offset(name_rva) + len(encoded)] = encoded
    return bytes(data)


def test_pe_parser_returns_named_stable_imports(tmp_path: Path) -> None:
    """Decode the DLL name and exact imported symbol names without dumpbin."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    symbols = ["aoti_torch_get_data_ptr", "torch_library_impl"]
    library.write_bytes(_minimal_pe_imports("torch_cpu.dll", symbols))
    assert _CHECKER._pe_imports(library) == {"torch_cpu.dll": set(symbols)}


def test_pe_checker_accepts_exact_reviewed_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accept the hosted MSVC/UCRT runtime closure and stable Torch imports."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    symbols = {"aoti_torch_get_data_ptr", "torch_library_impl"}
    assert "api-ms-win-crt-math-l1-1-0.dll" in _CHECKER.PE_DEPENDENCIES_EXPECTED
    imports = {name: set() for name in _CHECKER.PE_DEPENDENCIES_EXPECTED}
    imports["torch_cpu.dll"] = symbols
    monkeypatch.setattr(_CHECKER, "_pe_machine", lambda path: 0x8664)
    monkeypatch.setattr(_CHECKER, "_pe_imports", lambda path: imports)
    _CHECKER._check_pe(library, symbols)


def test_pe_checker_rejects_unreviewed_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject ordinary DLL imports outside the complete reviewed load set."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    imports = {name: set() for name in _CHECKER.PE_DEPENDENCIES_EXPECTED}
    imports["torch_cpu.dll"] = {"aoti_torch_get_data_ptr"}
    imports["evil.dll"] = {"evil"}
    monkeypatch.setattr(_CHECKER, "_pe_machine", lambda path: 0x8664)
    monkeypatch.setattr(_CHECKER, "_pe_imports", lambda path: imports)
    with pytest.raises(SystemExit, match="unexpected PE dependencies"):
        _CHECKER._check_pe(library, {"aoti_torch_get_data_ptr"})


def test_pe_checker_rejects_direct_nonstable_torch_dependency(tmp_path: Path) -> None:
    """A plugin may import stable symbols only from torch_cpu.dll."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    library.write_bytes(_minimal_pe_imports("c10.dll", ["aoti_torch_get_data_ptr"]))
    with pytest.raises(SystemExit, match="unexpected PE dependencies"):
        _CHECKER._check_pe(library, {"aoti_torch_get_data_ptr"})


def test_pe_checker_rejects_nonstable_torch_cpu_symbol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one Torch DLL edge may contain only stable-C imports."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    imports = {name: set() for name in _CHECKER.PE_DEPENDENCIES_EXPECTED}
    imports["torch_cpu.dll"] = {"at_tensor_new"}
    monkeypatch.setattr(_CHECKER, "_pe_machine", lambda path: 0x8664)
    monkeypatch.setattr(_CHECKER, "_pe_imports", lambda path: imports)
    with pytest.raises(SystemExit, match="non-stable symbols"):
        _CHECKER._check_pe(library, {"aoti_torch_get_data_ptr"})


def test_pe_checker_rejects_torch_cpu_ordinal_import(tmp_path: Path) -> None:
    """Stable ABI symbols are name-based and must never bind by ordinal."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    library.write_bytes(
        _minimal_pe_imports("torch_cpu.dll", ["aoti_torch_get_data_ptr"], ordinal=True)
    )
    with pytest.raises(SystemExit, match="must use named imports"):
        _CHECKER._check_pe(library, {"aoti_torch_get_data_ptr"})


def test_pe_checker_rejects_non_amd64_extension(tmp_path: Path) -> None:
    """Windows ARM64 wheels deliberately carry no untested Torch plugin."""
    library = tmp_path / "xtbloom_torch_ext.dll"
    library.write_bytes(
        _minimal_pe_imports(
            "torch_cpu.dll", ["aoti_torch_get_data_ptr"], machine=0xAA64
        )
    )
    with pytest.raises(SystemExit, match="must be AMD64"):
        _CHECKER._check_pe(library, {"aoti_torch_get_data_ptr"})
