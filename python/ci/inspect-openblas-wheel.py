#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify xTBloom's private OpenBLAS provider cohort in native wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class InspectionError(RuntimeError):
    """Report wheel metadata or native structure outside reviewed policy."""


MACHO_LIBXTBLOOM_DEPENDENCIES = {
    "/usr/lib/libSystem.B.dylib",
    "/usr/lib/libc++.1.dylib",
}
MACHO_OPENBLAS_SYSTEM_DEPENDENCIES = {"/usr/lib/libSystem.B.dylib"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _one(names: set[str], pattern: str, description: str) -> str:
    matches = sorted(name for name in names if re.fullmatch(pattern, name))
    if len(matches) != 1:
        raise InspectionError(
            f"wheel must contain exactly one {description}; found {len(matches)}"
        )
    return matches[0]


def _run(command: list[str], description: str) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise InspectionError(f"{description} failed: {detail}")
    return completed.stdout


def _run_readelf(readelf: str, option: str, path: Path) -> str:
    return _run(
        [readelf, option, "--wide", str(path)],
        f"readelf {option} for {path.name}",
    )


def _elf_dynamic(readelf: str, path: Path) -> tuple[set[str], str | None, str | None]:
    output = _run_readelf(readelf, "--dynamic", path)
    needed = set(re.findall(r"Shared library: \[([^]]+)]", output))
    sonames = re.findall(r"Library soname: \[([^]]+)]", output)
    rpaths = re.findall(r"Library rpath: \[([^]]+)]", output)
    runpaths = re.findall(r"Library runpath: \[([^]]+)]", output)
    if len(sonames) > 1 or len(rpaths) > 1 or len(runpaths) > 1:
        raise InspectionError(f"ambiguous dynamic tags in {path.name}")
    if rpaths and runpaths:
        raise InspectionError(f"{path.name} carries both DT_RPATH and DT_RUNPATH")
    search_path = rpaths[0] if rpaths else (runpaths[0] if runpaths else None)
    return needed, sonames[0] if sonames else None, search_path


def _elf_machine(readelf: str, path: Path) -> str:
    output = _run_readelf(readelf, "--file-header", path)
    match = re.search(r"^\s*Machine:\s*(.+?)\s*$", output, re.MULTILINE)
    if match is None:
        raise InspectionError(f"cannot determine ELF machine for {path.name}")
    return match.group(1)


def _expected_vendored_name(source_name: str, source_sha256: str) -> str:
    index = source_name.find(".so")
    if index < 0:
        raise InspectionError(f"manifest payload is not an ELF DSO: {source_name}")
    return source_name[:index] + f"-{source_sha256[:8]}" + source_name[index:]


def _wheel_target(name: str) -> str:
    lowered = name.lower()
    if "manylinux" in lowered and lowered.endswith("_x86_64.whl"):
        return "linux-x86_64"
    if "manylinux" in lowered and lowered.endswith("_aarch64.whl"):
        return "linux-aarch64"
    if "macosx" in lowered and lowered.endswith("_x86_64.whl"):
        return "macos-x86_64"
    if "macosx" in lowered and lowered.endswith("_arm64.whl"):
        return "macos-arm64"
    if lowered.endswith("-win_amd64.whl"):
        return "windows-amd64"
    if lowered.endswith("-win_arm64.whl"):
        return "windows-arm64"
    raise InspectionError(f"unsupported native wheel target: {name}")


def _common_policy(
    archive: zipfile.ZipFile,
    wheel: Path,
    manifest_path: Path,
) -> tuple[set[str], str, dict[str, object]]:
    names = {info.filename for info in archive.infolist() if not info.is_dir()}
    metadata_name = _one(names, r"[^/]+\.dist-info/METADATA", "METADATA")
    metadata = BytesParser().parsebytes(archive.read(metadata_name))
    requirements = [value.lower() for value in metadata.get_all("Requires-Dist", [])]
    if any(value.startswith("scipy-openblas32") for value in requirements):
        raise InspectionError("wheel publishes scipy-openblas32 as a dependency")
    if any(name.startswith("scipy_openblas32/") for name in names):
        raise InspectionError("wheel contains the upstream scipy_openblas32 package")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = _wheel_target(wheel.name)
    target_record = manifest["targets"][target]
    provenance_name = _one(
        names,
        r"xtbloom/share/licenses/xtbloom/provenance/scipy_openblas32_manifest\.json",
        "installed scipy-openblas32 provenance manifest",
    )
    if archive.read(provenance_name) != manifest_path.read_bytes():
        raise InspectionError("wheel OpenBLAS provenance differs from source bytes")

    source_root = manifest_path.parents[2]
    source = manifest["source"]
    for source_key, pattern in (
        (
            "local_license",
            r"xtbloom/share/licenses/xtbloom/third-party/"
            r"scipy-openblas32-0\.3\.34\.0\.0\.txt",
        ),
        (
            "local_windows_license",
            r"xtbloom/share/licenses/xtbloom/third-party/"
            r"scipy-openblas32-tools-LICENSE_win32\.txt",
        ),
    ):
        wheel_license = _one(names, pattern, f"installed {source_key}")
        expected = (source_root / source[source_key]).read_bytes()
        if archive.read(wheel_license) != expected:
            raise InspectionError(f"wheel {source_key} differs from reviewed bytes")
    exact_license = target_record["license"]["local"]
    exact_license_name = _one(
        names,
        r"xtbloom/share/licenses/xtbloom/third-party/"
        + re.escape(PurePosixPath(exact_license).name),
        "target-specific scipy-openblas32 packaged license",
    )
    if archive.read(exact_license_name) != (source_root / exact_license).read_bytes():
        raise InspectionError(
            "wheel target-specific scipy-openblas32 license differs from reviewed bytes"
        )
    return names, target, target_record


def _inspect_linux(
    archive: zipfile.ZipFile,
    names: set[str],
    target_record: dict[str, object],
    readelf: str,
) -> None:
    records = target_record["files"]
    expected_machine = target_record["elf_machine"]
    expected_vendor_names = {
        _expected_vendored_name(PurePosixPath(record["source"]).name, record["sha256"])
        for record in records
    }
    if any(PurePosixPath(name).name == "libscipy_openblas.so" for name in names):
        raise InspectionError("wheel retains the upstream generic OpenBLAS SONAME")

    libxtbloom_name = _one(names, r"xtbloom/lib(?:64)?/libxtbloom\.so", "libxtbloom")
    shim_name = _one(
        names,
        r"xtbloom/lib(?:64)?/libxtbloom_openblas_lp64_shim\.so",
        "private OpenBLAS shim",
    )
    vendor_members = {
        name
        for name in names
        if ".libs/" in name and (name.endswith(".so") or ".so." in name)
    }
    observed_vendor_names = {PurePosixPath(name).name for name in vendor_members}
    if observed_vendor_names != expected_vendor_names:
        raise InspectionError(
            "auditwheel OpenBLAS cohort differs: expected "
            f"{sorted(expected_vendor_names)}, found {sorted(observed_vendor_names)}"
        )
    vendor_dirs = {str(PurePosixPath(name).parent) for name in vendor_members}
    if len(vendor_dirs) != 1:
        raise InspectionError("auditwheel provider cohort is split across directories")
    vendor_dir = next(iter(vendor_dirs))

    with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-elf-") as directory:
        root = Path(directory)
        selected = {libxtbloom_name, shim_name, *vendor_members}
        extracted: dict[str, Path] = {}
        for name in selected:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
            extracted[name] = destination

        for name, path in extracted.items():
            if _elf_machine(readelf, path) != expected_machine:
                raise InspectionError(f"{name} has the wrong ELF machine")

        lib_needed, _, _ = _elf_dynamic(readelf, extracted[libxtbloom_name])
        forbidden_direct = {
            name
            for name in lib_needed
            if re.search(r"openblas|gfortran|quadmath|xtbloom_openblas", name)
        }
        if forbidden_direct:
            raise InspectionError(
                "libxtbloom has a hard private-provider dependency: "
                + ", ".join(sorted(forbidden_direct))
            )

        provider_record = next(
            record for record in records if record["role"] == "provider"
        )
        provider_name = _expected_vendored_name(
            PurePosixPath(provider_record["source"]).name,
            provider_record["sha256"],
        )
        shim_needed, shim_soname, shim_rpath = _elf_dynamic(
            readelf, extracted[shim_name]
        )
        if (
            provider_name not in shim_needed
            or shim_soname != PurePosixPath(shim_name).name
        ):
            raise InspectionError("private shim does not bind the vendored provider")
        relative_vendor = os.path.relpath(
            vendor_dir, str(PurePosixPath(shim_name).parent)
        )
        if shim_rpath != f"$ORIGIN/{relative_vendor}":
            raise InspectionError(
                "private shim does not use auditwheel's relative RPATH"
            )

        provider_member = next(
            name for name in vendor_members if PurePosixPath(name).name == provider_name
        )
        provider_needed, provider_soname, provider_rpath = _elf_dynamic(
            readelf, extracted[provider_member]
        )
        if provider_soname != provider_name or provider_rpath != "$ORIGIN":
            raise InspectionError("vendored OpenBLAS SONAME/RPATH is not private")
        gfortran_name = next(
            name for name in expected_vendor_names if "gfortran" in name
        )
        if gfortran_name not in provider_needed:
            raise InspectionError(
                "vendored provider does not bind reviewed libgfortran"
            )
        symbols = _run_readelf(readelf, "--dyn-syms", extracted[provider_member])
        for symbol in (
            "scipy_LAPACKE_dpotrf_work",
            "scipy_LAPACKE_dpocon_work",
            "scipy_LAPACKE_dsyevd_work",
            "scipy_cblas_dtrsm",
            "scipy_cblas_dgemm",
            "scipy_openblas_get_config",
            "openblas_set_num_threads_local",
        ):
            if symbol not in symbols:
                raise InspectionError(f"vendored OpenBLAS omits {symbol}")

        gfortran_member = next(
            name for name in vendor_members if PurePosixPath(name).name == gfortran_name
        )
        gfortran_needed, gfortran_soname, _ = _elf_dynamic(
            readelf, extracted[gfortran_member]
        )
        if gfortran_soname != gfortran_name:
            raise InspectionError("vendored libgfortran SONAME is not private")
        quadmath_names = {name for name in expected_vendor_names if "quadmath" in name}
        if quadmath_names and not quadmath_names.intersection(gfortran_needed):
            raise InspectionError("vendored libgfortran does not bind libquadmath")


def _macho_dependencies(otool: str, path: Path) -> list[str]:
    output = _run([otool, "-L", str(path)], f"otool -L for {path.name}")
    dependencies = []
    for line in output.splitlines()[1:]:
        stripped = line.strip()
        if stripped:
            dependencies.append(stripped.split(" (", 1)[0])
    return dependencies


def _inspect_macos(
    archive: zipfile.ZipFile,
    names: set[str],
    target_record: dict[str, object],
    otool: str,
    nm: str,
    codesign: str,
) -> None:
    libxtbloom_name = _one(names, r"xtbloom/lib/libxtbloom\.dylib", "libxtbloom")
    records = target_record["files"]
    expected_members = {
        f"xtbloom/{record['install_destination']}/{record['install_name']}".replace(
            "//", "/"
        )
        for record in records
    }
    observed_members = {
        name
        for name in names
        if PurePosixPath(name).name in {record["install_name"] for record in records}
    }
    if observed_members != expected_members:
        raise InspectionError(
            f"macOS OpenBLAS cohort differs: expected {sorted(expected_members)}, "
            f"found {sorted(observed_members)}"
        )
    provider_record = next(record for record in records if record["role"] == "provider")
    provider_member = f"xtbloom/{target_record['installed_provider']}"
    with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-macho-") as directory:
        root = Path(directory)
        extracted: dict[str, Path] = {}
        for name in {libxtbloom_name, *expected_members}:
            destination = root / PurePosixPath(name).name
            destination.write_bytes(archive.read(name))
            extracted[name] = destination

        provider = extracted[provider_member]
        symbols = _run([nm, "-gU", str(provider)], "nm provider symbols")
        for symbol in (
            "scipy_LAPACKE_dpotrf_work",
            "scipy_LAPACKE_dpocon_work",
            "scipy_LAPACKE_dsyevd_work",
            "scipy_cblas_dtrsm",
            "scipy_cblas_dgemm",
            "scipy_openblas_get_config",
            "scipy_openblas_set_num_threads",
            "scipy_openblas_get_num_threads",
        ):
            if symbol not in symbols:
                raise InspectionError(f"macOS provider omits {symbol}")

        for record in records:
            member = f"xtbloom/{record['install_destination']}/{record['install_name']}"
            binary = extracted[member]
            if _sha256(archive.read(member)) == record["sha256"]:
                raise InspectionError(
                    f"macOS image retained generic upstream load commands: "
                    f"{record['install_name']}"
                )
            install_ids = _run(
                [otool, "-D", str(binary)],
                f"otool ID for {record['install_name']}",
            ).splitlines()
            if len(install_ids) < 2 or install_ids[1].strip() != record["install_id"]:
                raise InspectionError(
                    "macOS image does not use its private LC_ID: "
                    f"{record['install_name']}"
                )
            _run(
                [codesign, "--verify", "--strict", str(binary)],
                f"codesign verification for {record['install_name']}",
            )
            observed_dependencies = set(_macho_dependencies(otool, binary))
            observed_dependencies.discard(record["install_id"])
            expected_dependencies = MACHO_OPENBLAS_SYSTEM_DEPENDENCIES | {
                rewrite["to"] for rewrite in record["load_rewrites"]
            }
            if observed_dependencies != expected_dependencies:
                raise InspectionError(
                    f"{record['install_name']} dependency closure differs: "
                    f"expected {sorted(expected_dependencies)}, "
                    f"found {sorted(observed_dependencies)}"
                )
        libxtbloom = extracted[libxtbloom_name]
        expected_id = "@rpath/libxtbloom.dylib"
        lib_ids = {
            line.strip()
            for line in _run(
                [otool, "-D", str(libxtbloom)],
                "otool ID for libxtbloom.dylib",
            ).splitlines()[1:]
            if line.strip()
        }
        if lib_ids != {expected_id}:
            raise InspectionError(
                f"libxtbloom LC_ID differs: expected {expected_id}, "
                f"found {sorted(lib_ids)}"
            )
        lib_dependencies = set(_macho_dependencies(otool, libxtbloom))
        lib_dependencies.discard(expected_id)
        if lib_dependencies != MACHO_LIBXTBLOOM_DEPENDENCIES:
            raise InspectionError(
                "libxtbloom dependency closure differs: expected "
                f"{sorted(MACHO_LIBXTBLOOM_DEPENDENCIES)}, "
                f"found {sorted(lib_dependencies)}"
            )

        if target_record["installed_provider_id"] != provider_record["install_id"]:
            raise InspectionError("macOS provider manifest IDs disagree")


def _pe_imports(data: bytes) -> set[str]:
    """Return IMAGE_IMPORT_DESCRIPTOR names without third-party PE tooling."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise InspectionError("provider is not a PE image")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise InspectionError("provider has an invalid PE signature")
    number_of_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    optional = pe_offset + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    directory = optional + (112 if magic == 0x20B else 96 if magic == 0x10B else -1)
    if directory < optional:
        raise InspectionError("provider has an unsupported PE optional header")
    import_rva = struct.unpack_from("<I", data, directory + 8)[0]
    section_table = optional + optional_size
    sections = []
    for index in range(number_of_sections):
        offset = section_table + 40 * index
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def rva_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        raise InspectionError(f"PE RVA 0x{rva:x} is outside mapped sections")

    if import_rva == 0:
        return set()
    cursor = rva_offset(import_rva)
    imports: set[str] = set()
    while True:
        descriptor = struct.unpack_from("<IIIII", data, cursor)
        if descriptor == (0, 0, 0, 0, 0):
            break
        name_offset = rva_offset(descriptor[3])
        end = data.find(b"\0", name_offset)
        if end < 0:
            raise InspectionError("PE import name is unterminated")
        imports.add(data[name_offset:end].decode("ascii"))
        cursor += 20
    return imports


def _pe_machine(data: bytes) -> int:
    """Return the COFF machine identifier from one PE image."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise InspectionError("provider is not a PE image")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise InspectionError("provider has an invalid PE signature")
    return struct.unpack_from("<H", data, pe_offset + 4)[0]


def _pe_exports(data: bytes) -> set[str]:
    """Return named exports from one PE image without third-party tooling."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise InspectionError("provider is not a PE image")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise InspectionError("provider has an invalid PE signature")
    number_of_sections = struct.unpack_from("<H", data, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe_offset + 20)[0]
    optional = pe_offset + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    directory = optional + (112 if magic == 0x20B else 96 if magic == 0x10B else -1)
    if directory < optional:
        raise InspectionError("provider has an unsupported PE optional header")
    export_rva = struct.unpack_from("<I", data, directory)[0]
    section_table = optional + optional_size
    sections = []
    for index in range(number_of_sections):
        offset = section_table + 40 * index
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def rva_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                return raw_offset + rva - virtual_address
        raise InspectionError(f"PE RVA 0x{rva:x} is outside mapped sections")

    if export_rva == 0:
        return set()
    export_offset = rva_offset(export_rva)
    number_of_names = struct.unpack_from("<I", data, export_offset + 24)[0]
    names_rva = struct.unpack_from("<I", data, export_offset + 32)[0]
    names_offset = rva_offset(names_rva)
    exports: set[str] = set()
    for index in range(number_of_names):
        name_rva = struct.unpack_from("<I", data, names_offset + 4 * index)[0]
        name_offset = rva_offset(name_rva)
        end = data.find(b"\0", name_offset)
        if end < 0:
            raise InspectionError("PE export name is unterminated")
        exports.add(data[name_offset:end].decode("ascii"))
    return exports


def _inspect_windows(
    archive: zipfile.ZipFile,
    names: set[str],
    target_record: dict[str, object],
) -> None:
    libxtbloom_name = _one(names, r"xtbloom/bin/xtbloom\.dll", "xtbloom.dll")
    records = target_record["files"]
    if len(records) != 1 or records[0]["role"] != "provider":
        raise InspectionError("Windows provider cohort must contain one DLL")
    record = records[0]
    provider_name = f"xtbloom/{record['install_destination']}/{record['install_name']}"
    if provider_name not in names:
        raise InspectionError(f"Windows wheel is missing {provider_name}")
    provider = archive.read(provider_name)
    if len(provider) != record["size"] or _sha256(provider) != record["sha256"]:
        raise InspectionError("Windows private provider differs from reviewed bytes")
    expected_machine = 0xAA64 if target_record["architecture"] == "arm64" else 0x8664
    if _pe_machine(provider) != expected_machine:
        raise InspectionError("Windows private provider has the wrong architecture")
    exports = _pe_exports(provider)
    required_exports = {
        "scipy_LAPACKE_dpotrf_work",
        "scipy_LAPACKE_dpocon_work",
        "scipy_LAPACKE_dsyevd_work",
        "scipy_cblas_dtrsm",
        "scipy_cblas_dgemm",
        "scipy_openblas_get_config",
        "scipy_openblas_set_num_threads",
        "scipy_openblas_get_num_threads",
    }
    missing_exports = required_exports - exports
    if missing_exports:
        raise InspectionError(
            f"Windows provider omits required exports: {sorted(missing_exports)}"
        )
    provider_imports = {name.lower() for name in _pe_imports(provider)}
    allowed = {
        name
        for name in provider_imports
        if name == "kernel32.dll"
        or name == "vcruntime140.dll"
        or name.startswith("api-ms-win-crt-")
    }
    if provider_imports != allowed:
        unexpected = sorted(provider_imports - allowed)
        raise InspectionError(f"Windows provider has unreviewed imports: {unexpected}")
    if (
        target_record["architecture"] == "arm64"
        and "vcruntime140.dll" not in provider_imports
    ):
        raise InspectionError("Windows ARM64 provider is missing VCRUNTIME140.dll")
    lib_imports = {name.lower() for name in _pe_imports(archive.read(libxtbloom_name))}
    if any("openblas" in name or "xtbloom_scipy" in name for name in lib_imports):
        raise InspectionError("xtbloom.dll has a hard private-provider dependency")


def inspect_wheel(
    wheel: Path,
    manifest_path: Path,
    readelf: str,
    otool: str,
    nm: str,
    codesign: str,
) -> None:
    """Validate metadata, provenance, payload, and native dependency policy."""
    with zipfile.ZipFile(wheel) as archive:
        names, target, target_record = _common_policy(archive, wheel, manifest_path)
        if target.startswith("linux-"):
            _inspect_linux(archive, names, target_record, readelf)
        elif target.startswith("macos-"):
            _inspect_macos(archive, names, target_record, otool, nm, codesign)
        else:
            _inspect_windows(archive, names, target_record)


def main() -> int:
    """Inspect one repaired or directly bundled native wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("--otool", default="otool")
    parser.add_argument("--nm", default="nm")
    parser.add_argument("--codesign", default="codesign")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        inspect_wheel(
            args.wheel.resolve(),
            args.manifest.resolve(),
            args.readelf,
            args.otool,
            args.nm,
            args.codesign,
        )
    except (
        InspectionError,
        OSError,
        KeyError,
        ValueError,
        struct.error,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ) as error:
        parser.error(str(error))
    print(f"OpenBLAS wheel inspection passed: {args.wheel}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
