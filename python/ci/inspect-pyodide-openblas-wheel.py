#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect the repaired Pyodide wheel's private OpenBLAS dependency boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class InspectionError(RuntimeError):
    """Report a wheel payload or WebAssembly linkage policy violation."""


_VALUE_TYPES = {
    0x7F: "i32",
    0x7E: "i64",
    0x7D: "f32",
    0x7C: "f64",
    0x7B: "v128",
    0x70: "funcref",
    0x6F: "externref",
}


def _read_u32(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 35:
            raise InspectionError("invalid WebAssembly LEB128 integer")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, offset
        shift += 7


def _encode_u32(value: int) -> bytes:
    """Encode one non-negative WebAssembly LEB128 integer canonically."""
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _read_name(data: bytes, offset: int) -> tuple[str, int]:
    size, offset = _read_u32(data, offset)
    end = offset + size
    if end > len(data):
        raise InspectionError("truncated WebAssembly name")
    try:
        return data[offset:end].decode("utf-8"), end
    except UnicodeDecodeError as error:
        raise InspectionError("invalid UTF-8 WebAssembly name") from error


def _encode_name(value: str) -> bytes:
    """Encode one WebAssembly UTF-8 name with a canonical length prefix."""
    data = value.encode("utf-8")
    return _encode_u32(len(data)) + data


def _sections(data: bytes) -> list[tuple[int, str | None, bytes, bytes]]:
    """Return section id/name/payload/raw records with strict bounds checks."""
    if data[:8] != b"\0asm\x01\0\0\0":
        raise InspectionError("wheel binary is not a WebAssembly v1 module")
    records = []
    offset = 8
    while offset < len(data):
        start = offset
        section_id = data[offset]
        offset += 1
        size, payload_offset = _read_u32(data, offset)
        end = payload_offset + size
        if end > len(data):
            raise InspectionError("truncated WebAssembly section")
        payload = data[payload_offset:end]
        name = None
        if section_id == 0:
            name, _ = _read_name(payload, 0)
        records.append((section_id, name, payload, data[start:end]))
        offset = end
    return records


def _dylink(data: bytes) -> tuple[list[str], list[str]]:
    records = [
        record for record in _sections(data) if record[1] in {"dylink", "dylink.0"}
    ]
    if len(records) != 1:
        raise InspectionError("WebAssembly side module must contain one dylink section")
    payload = records[0][2]
    _, offset = _read_name(payload, 0)
    needed: list[str] = []
    runtime_paths: list[str] = []
    seen_string_lists: set[int] = set()
    while offset < len(payload):
        subsection, offset = _read_u32(payload, offset)
        size, body_offset = _read_u32(payload, offset)
        end = body_offset + size
        if end > len(payload):
            raise InspectionError("truncated WebAssembly dylink subsection")
        body = payload[body_offset:end]
        if subsection in {2, 5}:
            if subsection in seen_string_lists:
                raise InspectionError(
                    f"duplicate WebAssembly dylink subsection {subsection}"
                )
            seen_string_lists.add(subsection)
            count, position = _read_u32(body, 0)
            values = []
            for _ in range(count):
                value, position = _read_name(body, position)
                values.append(value)
            if position != len(body):
                raise InspectionError("unexpected bytes in dylink string list")
            if subsection == 2:
                needed = values
            else:
                runtime_paths = values
        offset = end
    return needed, runtime_paths


def _exports(data: bytes) -> set[str]:
    sections = [
        payload for section_id, _, payload, _ in _sections(data) if section_id == 7
    ]
    if len(sections) != 1:
        raise InspectionError("WebAssembly module must contain one export section")
    payload = sections[0]
    count, offset = _read_u32(payload, 0)
    names = set()
    for _ in range(count):
        name, offset = _read_name(payload, offset)
        if offset >= len(payload):
            raise InspectionError("truncated WebAssembly export")
        offset += 1  # external kind
        _, offset = _read_u32(payload, offset)
        names.add(name)
    if offset != len(payload):
        raise InspectionError("unexpected bytes in WebAssembly export section")
    return names


def _read_limits(data: bytes, offset: int) -> int:
    """Skip one WebAssembly table or memory limits record."""
    flags, offset = _read_u32(data, offset)
    _, offset = _read_u32(data, offset)
    if flags & 0x01:
        _, offset = _read_u32(data, offset)
    return offset


def _function_export_signatures(
    data: bytes,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return exact parameter/result types for every exported function.

    Native C permits callers to ignore a scalar return from some ABIs, but a
    WebAssembly ``call_indirect`` traps when the complete function type differs.
    The production provider therefore needs type-level inspection, not merely
    an export-name allowlist.
    """
    type_sections = [
        payload for section_id, _, payload, _ in _sections(data) if section_id == 1
    ]
    if len(type_sections) != 1:
        raise InspectionError("WebAssembly module must contain one type section")
    payload = type_sections[0]
    count, offset = _read_u32(payload, 0)
    function_types: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for _ in range(count):
        if offset >= len(payload) or payload[offset] != 0x60:
            raise InspectionError("unsupported WebAssembly function type")
        offset += 1
        parameter_count, offset = _read_u32(payload, offset)
        parameters = []
        for _ in range(parameter_count):
            if offset >= len(payload) or payload[offset] not in _VALUE_TYPES:
                raise InspectionError("unsupported WebAssembly parameter type")
            parameters.append(_VALUE_TYPES[payload[offset]])
            offset += 1
        result_count, offset = _read_u32(payload, offset)
        results = []
        for _ in range(result_count):
            if offset >= len(payload) or payload[offset] not in _VALUE_TYPES:
                raise InspectionError("unsupported WebAssembly result type")
            results.append(_VALUE_TYPES[payload[offset]])
            offset += 1
        function_types.append((tuple(parameters), tuple(results)))
    if offset != len(payload):
        raise InspectionError("unexpected bytes in WebAssembly type section")

    function_type_indices: list[int] = []
    import_sections = [
        payload for section_id, _, payload, _ in _sections(data) if section_id == 2
    ]
    if len(import_sections) > 1:
        raise InspectionError("WebAssembly module contains duplicate import sections")
    if import_sections:
        payload = import_sections[0]
        count, offset = _read_u32(payload, 0)
        for _ in range(count):
            _, offset = _read_name(payload, offset)
            _, offset = _read_name(payload, offset)
            if offset >= len(payload):
                raise InspectionError("truncated WebAssembly import")
            kind = payload[offset]
            offset += 1
            if kind == 0:
                type_index, offset = _read_u32(payload, offset)
                function_type_indices.append(type_index)
            elif kind == 1:
                if offset >= len(payload):
                    raise InspectionError("truncated WebAssembly table import")
                offset += 1
                offset = _read_limits(payload, offset)
            elif kind == 2:
                offset = _read_limits(payload, offset)
            elif kind == 3:
                if offset + 2 > len(payload):
                    raise InspectionError("truncated WebAssembly global import")
                offset += 2
            elif kind == 4:
                _, offset = _read_u32(payload, offset)
                _, offset = _read_u32(payload, offset)
            else:
                raise InspectionError(f"unsupported WebAssembly import kind {kind}")
        if offset != len(payload):
            raise InspectionError("unexpected bytes in WebAssembly import section")

    function_sections = [
        payload for section_id, _, payload, _ in _sections(data) if section_id == 3
    ]
    if len(function_sections) > 1:
        raise InspectionError("WebAssembly module contains duplicate function sections")
    if function_sections:
        payload = function_sections[0]
        count, offset = _read_u32(payload, 0)
        for _ in range(count):
            type_index, offset = _read_u32(payload, offset)
            function_type_indices.append(type_index)
        if offset != len(payload):
            raise InspectionError("unexpected bytes in WebAssembly function section")

    export_sections = [
        payload for section_id, _, payload, _ in _sections(data) if section_id == 7
    ]
    if len(export_sections) != 1:
        raise InspectionError("WebAssembly module must contain one export section")
    payload = export_sections[0]
    count, offset = _read_u32(payload, 0)
    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for _ in range(count):
        name, offset = _read_name(payload, offset)
        if offset >= len(payload):
            raise InspectionError("truncated WebAssembly export")
        kind = payload[offset]
        offset += 1
        index, offset = _read_u32(payload, offset)
        if kind != 0:
            continue
        if index >= len(function_type_indices):
            raise InspectionError(
                f"WebAssembly function export index is invalid: {name}"
            )
        type_index = function_type_indices[index]
        if type_index >= len(function_types):
            raise InspectionError(f"WebAssembly function type index is invalid: {name}")
        signatures[name] = function_types[type_index]
    if offset != len(payload):
        raise InspectionError("unexpected bytes in WebAssembly export section")
    return signatures


def _repair_stable_sha256(data: bytes) -> str:
    """Hash every module byte except exact repair-owned dylink rewrites.

    ``auditwheel-emscripten`` rewrites NEEDED and RUNTIME_PATH, and its dylink
    encoder materializes absent EXPORT_INFO and IMPORT_INFO subsections as
    canonical empty lists.  Only those exact changes are ignored.  Non-empty
    symbol metadata, memory/table metadata, and every other custom or core
    section remain part of this digest.
    """
    canonical = bytearray(data[:8])
    for section_id, name, payload, raw in _sections(data):
        if section_id != 0 or name not in {"dylink", "dylink.0"}:
            canonical.extend(raw)
            continue
        _, offset = _read_name(payload, 0)
        stable_payload = bytearray(_encode_name(name))
        seen_subsections: set[int] = set()
        while offset < len(payload):
            subsection, offset = _read_u32(payload, offset)
            if subsection in seen_subsections:
                raise InspectionError(
                    f"duplicate WebAssembly dylink subsection {subsection}"
                )
            seen_subsections.add(subsection)
            size, body_offset = _read_u32(payload, offset)
            end = body_offset + size
            if end > len(payload):
                raise InspectionError("truncated WebAssembly dylink subsection")
            body = payload[body_offset:end]
            if subsection not in {2, 5}:
                # auditwheel-emscripten 0.2.5 always emits these two lists,
                # even when the input dylink section omitted them.  Treat only
                # the canonical zero-count encoding as repair-owned so a real
                # symbol-policy change cannot hide behind normalization.
                if subsection in {3, 4} and body == b"\0":
                    offset = end
                    continue
                stable_payload.extend(_encode_u32(subsection))
                stable_payload.extend(_encode_u32(len(body)))
                stable_payload.extend(body)
            offset = end
        canonical.append(0)
        canonical.extend(_encode_u32(len(stable_payload)))
        canonical.extend(stable_payload)
    return hashlib.sha256(canonical).hexdigest()


def _one(names: set[str], pattern: str, description: str) -> str:
    matches = sorted(name for name in names if re.fullmatch(pattern, name))
    if len(matches) != 1:
        raise InspectionError(f"expected one {description}, found {matches}")
    return matches[0]


def inspect(wheel: Path, manifest_path: Path) -> None:
    """Require the exact reviewed provider, adapter ABI, metadata, and notices."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    artifact = manifest["artifact"]
    with zipfile.ZipFile(wheel) as archive:
        names = {entry.filename for entry in archive.infolist() if not entry.is_dir()}
        metadata_name = _one(names, r"[^/]+\.dist-info/METADATA", "METADATA")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        requirements = [
            value.lower() for value in metadata.get_all("Requires-Dist", [])
        ]
        if any("scipy" in value or "openblas" in value for value in requirements):
            raise InspectionError(
                "Pyodide wheel publishes SciPy/OpenBLAS as a dependency"
            )
        wheel_name = _one(names, r"[^/]+\.dist-info/WHEEL", "WHEEL metadata")
        wheel_metadata = archive.read(wheel_name).decode("utf-8")
        if "pyodide" not in wheel_metadata and "pyemscripten" not in wheel_metadata:
            raise InspectionError("wheel does not publish a Pyodide/PyEmscripten tag")

        provenance = _one(
            names,
            r"xtbloom/share/licenses/xtbloom/provenance/pyodide_openblas_manifest\.json",
            "installed Pyodide OpenBLAS manifest",
        )
        if archive.read(provenance) != manifest_bytes:
            raise InspectionError("wheel Pyodide OpenBLAS manifest differs from source")
        root = manifest_path.parents[2]
        for record in manifest["licenses"]:
            local = Path(record["local"])
            installed = _one(
                names,
                r"xtbloom/share/licenses/xtbloom/third-party/" + re.escape(local.name),
                f"installed {record['name']} license",
            )
            if archive.read(installed) != (root / local).read_bytes():
                raise InspectionError(f"wheel license differs: {record['name']}")

        main_name = _one(names, r"xtbloom/lib/libxtbloom\.so", "libxtbloom")
        adapter_name = _one(
            names,
            r"xtbloom/lib/" + re.escape(artifact["adapter_install_name"]),
            "Pyodide LAPACKE adapter",
        )
        provider_name = _one(
            names,
            r"xtbloom\.libs/" + re.escape(artifact["private_install_name"]),
            "private Pyodide OpenBLAS provider",
        )
        expected_shared_modules = {main_name, adapter_name, provider_name}
        shared_modules = {name for name in names if name.endswith(".so")}
        forbidden = sorted(shared_modules - expected_shared_modules)
        forbidden.extend(
            sorted(
                name
                for name in names
                if name.endswith((".dll", ".dylib"))
                or PurePosixPath(name).name == artifact["filename"]
                or (
                    "openblas" in PurePosixPath(name).name.lower()
                    and PurePosixPath(name)
                    .name.lower()
                    .endswith(
                        (".a", ".wasm", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz")
                    )
                )
            )
        )
        if forbidden:
            raise InspectionError(
                f"Pyodide wheel contains forbidden payloads: {forbidden}"
            )

        main = archive.read(main_name)
        adapter = archive.read(adapter_name)
        provider = archive.read(provider_name)
        main_needed, main_rpath = _dylink(main)
        adapter_needed, adapter_rpath = _dylink(adapter)
        provider_needed, provider_rpath = _dylink(provider)
        # auditwheel-emscripten applies the vendored-library search path to
        # every repaired side module, including libxtbloom even though its
        # NEEDED list is intentionally empty. Pin that exact repair output so
        # an additional or more permissive runtime path remains a failure.
        if main_needed or main_rpath != ["$ORIGIN/../../xtbloom.libs"]:
            raise InspectionError(
                "libxtbloom linkage differs: "
                f"needed={main_needed}, runtime_paths={main_rpath}"
            )
        if adapter_needed != [artifact["private_install_name"]]:
            raise InspectionError(f"adapter dependency differs: {adapter_needed}")
        if adapter_rpath != ["$ORIGIN/../../xtbloom.libs"]:
            raise InspectionError(f"adapter runtime path differs: {adapter_rpath}")
        if provider_needed:
            raise InspectionError(
                f"private provider has dependencies: {provider_needed}"
            )
        if provider_rpath != ["$ORIGIN"]:
            raise InspectionError(
                f"private provider runtime path differs: {provider_rpath}"
            )
        if _repair_stable_sha256(provider) != artifact["member_repair_stable_sha256"]:
            raise InspectionError(
                "repaired provider differs outside exact repair-owned dylink rewrites"
            )

        provider_exports = _exports(provider)
        missing_provider = set(artifact["required_exports"]) - provider_exports
        if missing_provider:
            raise InspectionError(
                f"private provider omits exports: {sorted(missing_provider)}"
            )
        provider_signatures = _function_export_signatures(provider)
        expected_signatures = {
            name: (tuple(record["parameters"]), tuple(record["results"]))
            for name, record in artifact["required_export_signatures"].items()
        }
        observed_signatures = {
            name: provider_signatures.get(name) for name in expected_signatures
        }
        if observed_signatures != expected_signatures:
            raise InspectionError(
                "private provider function signatures differ: "
                f"expected {expected_signatures}, found {observed_signatures}"
            )
        required_adapter = {
            "xtbloom_pyodide_LAPACKE_dpotrf_work",
            "xtbloom_pyodide_LAPACKE_dpocon_work",
            "xtbloom_pyodide_LAPACKE_dsyevd_work",
            "xtbloom_pyodide_cblas_dtrsm",
            "xtbloom_pyodide_cblas_dgemm",
            "xtbloom_pyodide_openblas_get_config",
            "xtbloom_pyodide_openblas_set_num_threads_local",
            "xtbloom_pyodide_openblas_dependency_anchor",
        }
        missing_adapter = required_adapter - _exports(adapter)
        if missing_adapter:
            raise InspectionError(
                f"LAPACKE adapter omits exports: {sorted(missing_adapter)}"
            )


def main() -> int:
    """Parse command-line arguments and inspect one repaired wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        inspect(args.wheel, args.manifest)
    except (
        InspectionError,
        KeyError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        parser.error(str(error))
    sys.stdout.write(f"Pyodide OpenBLAS wheel inspection passed: {args.wheel}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
