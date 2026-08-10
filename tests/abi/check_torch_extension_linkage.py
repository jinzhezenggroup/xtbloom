"""Verify the LibTorch Stable ABI extension's platform linkage contract.

The extension is linked against a build-only stub that has the same runtime
identity as the real Torch CPU library.  The shipped plugin must therefore:

- depend on exactly one platform Torch CPU runtime name;
- avoid c10, torch_python, CUDA, or other provider-library dependencies;
- carry no ELF RPATH/RUNPATH or Mach-O LC_RPATH;
- import only the manifest-pinned stable C symbols from that runtime; and
- never retain a build-tree path or a bundled stub.

Linux uses readelf/nm, macOS uses otool/nm, and Windows parses the PE import
table directly so hosted wheel validation does not depend on an activated
Visual Studio command prompt.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ELF_NEEDED_EXPECTED = {
    "libtorch_cpu.so",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    # Some manylinux compilers retain a separate edge for standard math calls.
    "libm.so.6",
    # glibc < 2.34 still carries dlopen/dlsym/dladdr in a separate library.
    "libdl.so.2",
    # manylinux_2_28 keeps std::thread support in a separate system library.
    "libpthread.so.0",
}
MACHO_DEPENDENCIES_EXPECTED = {
    "@rpath/libtorch_cpu.dylib",
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libc++.1.dylib",
}
PE_DEPENDENCIES_EXPECTED = {
    "api-ms-win-crt-environment-l1-1-0.dll",
    "api-ms-win-crt-heap-l1-1-0.dll",
    "api-ms-win-crt-runtime-l1-1-0.dll",
    "api-ms-win-crt-stdio-l1-1-0.dll",
    "api-ms-win-crt-string-l1-1-0.dll",
    "kernel32.dll",
    "msvcp140.dll",
    "torch_cpu.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}
PE_MACHINE_AMD64 = 0x8664
WHEEL_EXTENSION_PATHS = {
    "linux": "xtbloom/lib/libxtbloom_torch_ext.so",
    "macos": "xtbloom/lib/libxtbloom_torch_ext.dylib",
    "windows": "xtbloom/bin/xtbloom_torch_ext.dll",
}


def _run(command: list[str]) -> str:
    """Run one inspection command and return its standard output."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _stable_symbols(text: str) -> set[str]:
    """Extract stable Torch C names, tolerating Mach-O's leading underscore."""
    return set(
        re.findall(
            r"(?<![A-Za-z0-9_])_?(aoti_torch_[A-Za-z0-9_]+|torch_[A-Za-z0-9_]+)\b",
            text,
        )
    )


def _check_stable_symbols(actual: set[str], pinned: set[str]) -> None:
    """Require a nonempty subset of the manifest-pinned stable symbol set."""
    if not actual:
        raise SystemExit("extension does not import any stable Torch C symbols")
    stale = actual - pinned
    if stale:
        raise SystemExit(
            "extension references Torch symbols absent from aoti_symbols.txt: "
            + ", ".join(sorted(stale))
        )


def _check_elf(
    library: Path, pinned: set[str], *, readelf: str | None, nm: str | None
) -> None:
    """Validate the Linux ELF dependency and undefined-symbol tables."""
    readelf_command = readelf or shutil.which("readelf")
    nm_command = nm or shutil.which("nm")
    if readelf_command is None or nm_command is None:
        raise SystemExit("Linux Torch linkage inspection requires readelf and nm")

    dynamic = _run([readelf_command, "-dW", str(library)])
    needed = set(re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic))
    unexpected = needed - ELF_NEEDED_EXPECTED
    if unexpected:
        raise SystemExit(f"unexpected DT_NEEDED entries: {sorted(unexpected)}")
    if "libtorch_cpu.so" not in needed:
        raise SystemExit("extension must carry DT_NEEDED libtorch_cpu.so")
    if re.search(r"\(RPATH\)|\(RUNPATH\)", dynamic):
        raise SystemExit("extension must not carry RPATH/RUNPATH")

    undefined = _run([nm_command, "-D", "--undefined-only", str(library)])
    _check_stable_symbols(_stable_symbols(undefined), pinned)
    print(  # noqa: T201 - CLI validation output
        f"Torch ELF linkage OK ({len(needed)} DT_NEEDED, no RPATH/RUNPATH)"
    )


def _check_macho(
    library: Path, pinned: set[str], *, otool: str | None, nm: str | None
) -> None:
    """Validate the macOS load commands and undefined stable symbols."""
    otool_command = otool or shutil.which("otool")
    nm_command = nm or shutil.which("nm")
    if otool_command is None or nm_command is None:
        raise SystemExit("macOS Torch linkage inspection requires otool and nm")

    identity_output = _run([otool_command, "-D", str(library)])
    identities = {
        line.strip() for line in identity_output.splitlines()[1:] if line.strip()
    }
    expected_identity = f"@rpath/{library.name}"
    if identities != {expected_identity}:
        raise SystemExit(
            f"extension must have LC_ID_DYLIB {expected_identity}; found "
            + ", ".join(sorted(identities))
        )

    linked = _run([otool_command, "-L", str(library)])
    dependencies = {
        line.strip().split(" (compatibility", 1)[0]
        for line in linked.splitlines()[1:]
        if line.strip()
    }
    # otool -L includes a dylib's own LC_ID_DYLIB before its true dependency
    # load commands, so remove the separately verified self-identity.
    dependencies.discard(expected_identity)
    if dependencies != MACHO_DEPENDENCIES_EXPECTED:
        raise SystemExit(
            "unexpected Mach-O dependencies: expected "
            f"{sorted(MACHO_DEPENDENCIES_EXPECTED)}, found {sorted(dependencies)}"
        )

    load_commands = _run([otool_command, "-l", str(library)])
    if re.search(r"\bcmd LC_RPATH\b", load_commands):
        raise SystemExit("extension must not carry LC_RPATH")

    undefined = _run([nm_command, "-u", "-j", str(library)])
    _check_stable_symbols(_stable_symbols(undefined), pinned)
    print(  # noqa: T201 - CLI validation output
        f"Torch Mach-O linkage OK ({len(dependencies)} loads, no LC_RPATH)"
    )


def _read_c_string(data: bytes, offset: int) -> str:
    """Read one NUL-terminated ASCII string from a binary image."""
    if offset < 0 or offset >= len(data):
        raise SystemExit("PE import table contains an out-of-range string")
    end = data.find(b"\0", offset)
    if end < 0:
        raise SystemExit("PE import table contains an unterminated string")
    return data[offset:end].decode("ascii", "strict")


def _pe_imports(path: Path) -> dict[str, set[str]]:
    """Return DLL-to-symbol imports from a PE32/PE32+ image."""
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise SystemExit(f"{path} is not a PE image")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise SystemExit(f"{path} has an invalid PE header")

    coff = pe_offset + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    if optional + optional_size > len(data):
        raise SystemExit("PE optional header extends beyond the file")
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        pointer_size = 8
        data_directory = optional + 112
    elif magic == 0x10B:
        pointer_size = 4
        data_directory = optional + 96
    else:
        raise SystemExit(f"unsupported PE optional-header magic: {magic:#x}")
    if data_directory + 16 > optional + optional_size:
        raise SystemExit("PE optional header has no complete import directory")
    import_rva, _ = struct.unpack_from("<II", data, data_directory + 8)
    if import_rva == 0:
        raise SystemExit("PE extension has no import directory")

    section_table = optional + optional_size
    sections: list[tuple[int, int, int]] = []
    for index in range(section_count):
        offset = section_table + index * 40
        if offset + 40 > len(data):
            raise SystemExit("PE section table extends beyond the file")
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def rva_to_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                result = raw_offset + (rva - virtual_address)
                if result >= len(data):
                    break
                return result
        raise SystemExit(f"PE import RVA {rva:#x} is outside mapped sections")

    imports: dict[str, set[str]] = {}
    descriptor_offset = rva_to_offset(import_rva)
    while True:
        if descriptor_offset + 20 > len(data):
            raise SystemExit("PE import descriptor extends beyond the file")
        descriptor = struct.unpack_from("<IIIII", data, descriptor_offset)
        original_thunk, timestamp, forwarder, name_rva, first_thunk = descriptor
        if not any((original_thunk, timestamp, forwarder, name_rva, first_thunk)):
            break
        dll_name = _read_c_string(data, rva_to_offset(name_rva)).lower()
        thunk_rva = original_thunk or first_thunk
        thunk_offset = rva_to_offset(thunk_rva)
        symbols: set[str] = set()
        ordinal_mask = 1 << (pointer_size * 8 - 1)
        value_format = "<Q" if pointer_size == 8 else "<I"
        while True:
            if thunk_offset + pointer_size > len(data):
                raise SystemExit("PE import thunk extends beyond the file")
            thunk = struct.unpack_from(value_format, data, thunk_offset)[0]
            if thunk == 0:
                break
            if thunk & ordinal_mask:
                if dll_name == "torch_cpu.dll":
                    raise SystemExit(
                        "torch_cpu.dll stable ABI symbols must use named imports"
                    )
            else:
                hint_name = rva_to_offset(thunk)
                symbols.add(_read_c_string(data, hint_name + 2))
            thunk_offset += pointer_size
        imports.setdefault(dll_name, set()).update(symbols)
        descriptor_offset += 20
    return imports


def _pe_machine(path: Path) -> int:
    """Return the COFF machine identifier from one PE image."""
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise SystemExit(f"{path} is not a PE image")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 6 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise SystemExit(f"{path} has an invalid PE header")
    return struct.unpack_from("<H", data, pe_offset + 4)[0]


def _check_pe(library: Path, pinned: set[str]) -> None:
    """Validate Windows DLL imports without external PE inspection tools."""
    machine = _pe_machine(library)
    if machine != PE_MACHINE_AMD64:
        raise SystemExit(
            f"Windows Torch extension must be AMD64 ({PE_MACHINE_AMD64:#x}); "
            f"found {machine:#x}"
        )
    imports = _pe_imports(library)
    dependencies = set(imports)
    if dependencies != PE_DEPENDENCIES_EXPECTED:
        raise SystemExit(
            "unexpected PE dependencies: expected "
            f"{sorted(PE_DEPENDENCIES_EXPECTED)}, found {sorted(dependencies)}"
        )
    actual = imports["torch_cpu.dll"]
    nonstable = {
        symbol for symbol in actual if not symbol.startswith(("aoti_torch_", "torch_"))
    }
    if nonstable:
        raise SystemExit(
            "extension imports non-stable symbols from torch_cpu.dll: "
            + ", ".join(sorted(nonstable))
        )
    _check_stable_symbols(actual, pinned)
    print(  # noqa: T201 - CLI validation output
        f"Torch PE linkage OK ({len(dependencies)} imported DLLs)"
    )


def _platform_from_wheel(path: Path) -> str:
    """Map a wheel filename to the native binary format it contains."""
    name = path.name.lower()
    if "manylinux" in name or "musllinux" in name or "-linux_" in name:
        return "linux"
    if "macosx" in name:
        return "macos"
    if "win_" in name:
        return "windows"
    raise SystemExit(f"cannot infer native Torch platform from wheel name: {path}")


def _extract_wheel_extension(path: Path, platform: str, directory: Path) -> Path:
    """Extract the one expected extension path from a wheel archive."""
    expected = WHEEL_EXTENSION_PATHS[platform]
    with zipfile.ZipFile(path) as archive:
        regular = {info.filename for info in archive.infolist() if not info.is_dir()}
        matches = sorted(
            name
            for name in regular
            if PurePosixPath(name).name.startswith(
                ("libxtbloom_torch_ext", "xtbloom_torch_ext")
            )
        )
        if matches != [expected]:
            raise SystemExit(f"{path} must contain only {expected}; found {matches}")
        archive.extract(expected, directory)
    return directory / expected


def main() -> int:
    """Validate one built extension or one platform wheel archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readelf", help="readelf-compatible executable")
    parser.add_argument("--nm", help="nm-compatible executable")
    parser.add_argument("--otool", help="otool-compatible executable")
    parser.add_argument(
        "--symbol-list",
        required=True,
        type=Path,
        help="pinned stable-C symbol set (aoti_symbols.txt)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--library", type=Path)
    source.add_argument("--wheel", type=Path)
    parser.add_argument("--platform", choices=("linux", "macos", "windows"))
    args = parser.parse_args()

    pinned = set(args.symbol_list.read_text(encoding="utf-8").split())
    if not pinned:
        raise SystemExit("stable Torch symbol list is empty")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.wheel is not None:
            platform = args.platform or _platform_from_wheel(args.wheel)
            temporary = tempfile.TemporaryDirectory(prefix="xtbloom-torch-linkage-")
            library = _extract_wheel_extension(
                args.wheel, platform, Path(temporary.name)
            )
        else:
            assert args.library is not None
            library = args.library
            if args.platform is not None:
                platform = args.platform
            elif sys.platform == "darwin":
                platform = "macos"
            elif sys.platform == "win32":
                platform = "windows"
            else:
                platform = "linux"

        if platform == "linux":
            _check_elf(library, pinned, readelf=args.readelf, nm=args.nm)
        elif platform == "macos":
            _check_macho(library, pinned, otool=args.otool, nm=args.nm)
        else:
            _check_pe(library, pinned)
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
