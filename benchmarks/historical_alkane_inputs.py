#!/usr/bin/env python3
"""Materialize hash-pinned historical alkane arrays as assembler inputs.

This utility creates only coordinate files, an assembler-compatible manifest,
explicit empty golden placeholders, and a checksum list. It never invokes a
model or implies that the placeholders are scientific references.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SOURCE_FIELDS = {
    "resource_stub",
    "source_commit",
    "source_hashes",
    "workloads",
}
WORKLOAD_FIELDS = {
    "name",
    "natoms",
    "batch_size",
    "seed",
    "perturb_sigma_bohr",
    "atom_offsets",
    "atomic_numbers",
    "positions_bohr",
    "molecular_charges",
    "unpaired_electrons",
    "spin_channels",
}
EXPECTED_WORKLOADS = (
    ("alkane32-b256", 32),
    ("alkane62-b256", 62),
)
ELEMENT_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
)
UNITS = {
    "coordinates": "bohr",
    "energy": "hartree",
    "forces": "hartree/bohr",
    "gradient": "hartree/bohr",
    "molecular_charge": "elementary_charge",
}
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class HistoricalInputError(ValueError):
    """An invalid, changed, or unsupported historical input corpus."""


def select_workload_cases(
    manifest: dict[str, Any], workload_name: str
) -> tuple[dict[str, Any], ...]:
    """Return one workload's validated source-slot order for collector adapters.

    The returned cases are input-assembly records only. Their empty golden
    placeholders must not be passed to correctness qualification or used to
    make performance evidence eligible.
    """
    eligibility = manifest.get("eligibility")
    if (
        manifest.get("schema") != "xtbloom-historical-alkane-assembler-inputs-v1"
        or manifest.get("diagnostic_only") is not True
        or not isinstance(eligibility, dict)
        or eligibility.get("correctness") is not False
        or eligibility.get("performance") is not False
    ):
        raise HistoricalInputError(
            "manifest is not an explicitly ineligible historical input corpus"
        )
    workload_records = manifest.get("workloads")
    cases = manifest.get("cases")
    if not isinstance(workload_records, list) or not isinstance(cases, list):
        raise HistoricalInputError("manifest must contain workload and case arrays")
    matching = [
        record
        for record in workload_records
        if isinstance(record, dict) and record.get("name") == workload_name
    ]
    if len(matching) != 1:
        raise HistoricalInputError(
            f"manifest must contain exactly one workload {workload_name!r}"
        )
    record = matching[0]
    batch_size = record.get("batch_size")
    natoms = record.get("natoms")
    identifiers = record.get("case_ids")
    if (
        type(batch_size) is not int
        or batch_size != 256
        or type(natoms) is not int
        or natoms != dict(EXPECTED_WORKLOADS).get(workload_name)
        or not isinstance(identifiers, list)
        or len(identifiers) != batch_size
        or any(not isinstance(identifier, str) for identifier in identifiers)
        or len(set(identifiers)) != batch_size
    ):
        raise HistoricalInputError(
            f"workload {workload_name!r} has invalid saved slot identity"
        )
    cases_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise HistoricalInputError("manifest cases must have string IDs")
        identifier = case["id"]
        if identifier in cases_by_id:
            raise HistoricalInputError(f"duplicate case ID in manifest: {identifier}")
        cases_by_id[identifier] = case
    selected: list[dict[str, Any]] = []
    for slot, identifier in enumerate(identifiers):
        case = cases_by_id.get(identifier)
        if (
            case is None
            or case.get("source_workload") != workload_name
            or case.get("source_slot") != slot
            or case.get("atom_count") != natoms
            or case.get("independent_reference_available") is not False
            or case.get("correctness_eligible") is not False
            or case.get("performance_eligible") is not False
        ):
            raise HistoricalInputError(
                f"case {identifier!r} does not match "
                f"workload {workload_name!r} slot {slot}"
            )
        selected.append(case)
    return tuple(selected)


def _sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact input or output bytes."""
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    """Serialize numeric identities deterministically for stable geometry hashes."""
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HistoricalInputError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise HistoricalInputError(f"non-finite JSON value is not permitted: {value}")


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise HistoricalInputError(
            f"{label} fields differ from the supported schema: {actual}"
        )
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HistoricalInputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise HistoricalInputError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise HistoricalInputError(f"{label} is outside binary64 range") from exc
    if not math.isfinite(result):
        raise HistoricalInputError(f"{label} must be finite")
    return result


def _numeric_array(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise HistoricalInputError(f"{label} must contain exactly {length} values")
    return [
        _finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def _number_array(value: object, length: int, label: str) -> list[int | float]:
    if not isinstance(value, list) or len(value) != length:
        raise HistoricalInputError(f"{label} must contain exactly {length} values")
    for index, item in enumerate(value):
        _finite_number(item, f"{label}[{index}]")
    return value


def _load_source(
    input_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str, Path]:
    """Read only bytes matching the caller's mandatory SHA-256 pin."""
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise HistoricalInputError(
            "--expected-sha256 must be 64 hexadecimal characters"
        )
    source_path = input_path.expanduser().resolve()
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        raise HistoricalInputError(f"cannot read input {source_path}: {exc}") from exc
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256.lower():
        raise HistoricalInputError(
            f"input SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"actual {actual_sha256}"
        )
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalInputError(f"input is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise HistoricalInputError("input root must be a JSON object")
    return document, actual_sha256, source_path


def _validate_source(
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate source arrays and return normalized workload views."""
    source = _require_keys(document, SOURCE_FIELDS, "input")
    if not isinstance(source["resource_stub"], str) or not source["resource_stub"]:
        raise HistoricalInputError("resource_stub must be a nonempty string")
    if not isinstance(source["source_commit"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", source["source_commit"]
    ):
        raise HistoricalInputError("source_commit must be a 40-character Git hash")
    source_hashes = source["source_hashes"]
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise HistoricalInputError("source_hashes must be a nonempty object")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str):
            raise HistoricalInputError(
                "source_hashes must map paths to SHA-256 strings"
            )
        if not SHA256_PATTERN.fullmatch(digest):
            raise HistoricalInputError(
                f"source_hashes[{name!r}] is not a SHA-256 digest"
            )

    workloads = source["workloads"]
    if not isinstance(workloads, list) or len(workloads) != len(EXPECTED_WORKLOADS):
        raise HistoricalInputError(
            "workloads must contain the exact 32- and 62-atom arrays"
        )
    names = [item.get("name") if isinstance(item, dict) else None for item in workloads]
    expected_names = [name for name, _ in EXPECTED_WORKLOADS]
    if names != expected_names:
        raise HistoricalInputError(
            f"workload names/order must be {expected_names}; got {names}"
        )

    normalized: list[dict[str, Any]] = []
    for raw_workload, (expected_name, expected_natoms) in zip(
        workloads, EXPECTED_WORKLOADS, strict=True
    ):
        workload = _require_keys(raw_workload, WORKLOAD_FIELDS, expected_name)
        if workload["name"] != expected_name:
            raise HistoricalInputError(f"unexpected workload name {workload['name']!r}")
        natoms = _integer(workload["natoms"], f"{expected_name}.natoms", 1)
        if natoms != expected_natoms:
            raise HistoricalInputError(
                f"{expected_name}.natoms must be {expected_natoms}, got {natoms}"
            )
        batch_size = _integer(workload["batch_size"], f"{expected_name}.batch_size", 1)
        if batch_size != 256:
            raise HistoricalInputError(f"{expected_name}.batch_size must be 256")
        _integer(workload["seed"], f"{expected_name}.seed")
        perturb_sigma = _finite_number(
            workload["perturb_sigma_bohr"], f"{expected_name}.perturb_sigma_bohr"
        )
        if perturb_sigma < 0:
            raise HistoricalInputError(
                f"{expected_name}.perturb_sigma_bohr must be nonnegative"
            )

        atom_offsets = workload["atom_offsets"]
        expected_offsets = [index * natoms for index in range(batch_size + 1)]
        if atom_offsets != expected_offsets or any(
            type(value) is not int for value in atom_offsets
        ):
            raise HistoricalInputError(
                f"{expected_name}.atom_offsets must be "
                f"the exact regular {natoms}-atom layout"
            )
        atom_count = batch_size * natoms
        atomic_numbers = workload["atomic_numbers"]
        if (
            not isinstance(atomic_numbers, list)
            or len(atomic_numbers) != atom_count
            or any(
                type(value) is not int or not 1 <= value < len(ELEMENT_SYMBOLS)
                for value in atomic_numbers
            )
        ):
            raise HistoricalInputError(
                f"{expected_name}.atomic_numbers must contain "
                f"{atom_count} supported atomic-number tags"
            )
        positions = _numeric_array(
            workload["positions_bohr"],
            3 * atom_count,
            f"{expected_name}.positions_bohr",
        )
        charges = _number_array(
            workload["molecular_charges"],
            batch_size,
            f"{expected_name}.molecular_charges",
        )
        unpaired = workload["unpaired_electrons"]
        if (
            not isinstance(unpaired, list)
            or len(unpaired) != batch_size
            or any(type(value) is not int or value < 0 for value in unpaired)
        ):
            raise HistoricalInputError(
                f"{expected_name}.unpaired_electrons must contain "
                f"{batch_size} nonnegative integer tags"
            )
        spin_channels = workload["spin_channels"]
        if (
            not isinstance(spin_channels, list)
            or len(spin_channels) != batch_size
            or any(
                type(value) is not int or value not in (1, 2) for value in spin_channels
            )
        ):
            raise HistoricalInputError(
                f"{expected_name}.spin_channels must contain "
                f"{batch_size} integer tags in {{1, 2}}"
            )
        normalized.append(
            {
                "name": expected_name,
                "natoms": natoms,
                "batch_size": batch_size,
                "seed": workload["seed"],
                "perturb_sigma_bohr": perturb_sigma,
                "atom_offsets": atom_offsets,
                "atomic_numbers": atomic_numbers,
                "positions_bohr": positions,
                "molecular_charges": charges,
                "unpaired_electrons": unpaired,
                "spin_channels": spin_channels,
            }
        )
    return normalized, source


def _case_id(workload_name: str, slot: int) -> str:
    """Build a stable unique identifier without collapsing repeated geometries."""
    return f"historical-{workload_name}-slot-{slot:04d}"


def _geometry_sha256(atomic_numbers: list[int], positions: list[float]) -> str:
    geometry = {
        "atomic_numbers": atomic_numbers,
        "coordinates": "bohr",
        "positions_bohr": positions,
    }
    return _sha256(_canonical_json_bytes(geometry))


def _coordinate_bytes(workload: dict[str, Any], slot: int) -> bytes:
    """Serialize one $coord file with round-trip-safe binary64 decimals."""
    natoms = workload["natoms"]
    begin = workload["atom_offsets"][slot]
    lines = ["$coord"]
    for atom_index in range(begin, begin + natoms):
        number = workload["atomic_numbers"][atom_index]
        offset = 3 * atom_index
        xyz = workload["positions_bohr"][offset : offset + 3]
        lines.append(
            " ".join(repr(value) for value in xyz)
            + f" {ELEMENT_SYMBOLS[number].lower()}"
        )
    lines.append("$end")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _rename_linux_noreplace(stage: Path, output: Path) -> None:
    """Use the kernel's atomic no-replace operation, never a check/rename pair."""
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise HistoricalInputError(
            "atomic no-replace publication requires libc renameat2 support"
        ) from exc
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    # Linux UAPI: AT_FDCWD=-100 and RENAME_NOREPLACE=1. Missing kernel or
    # filesystem support must fail closed, not fall back to a clobbering rename.
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(output))


def _publish_stage_directory(stage: Path, output: Path) -> None:
    """Publish without replacing an output path won by another process."""
    if os.name == "nt":
        os.rename(stage, output)
        return
    if sys.platform == "linux":
        _rename_linux_noreplace(stage, output)
        return
    raise HistoricalInputError(
        "atomic no-replace publication is supported only on Linux and Windows"
    )


def _make_manifest(
    stage: Path,
    workloads: list[dict[str, Any]],
    source: dict[str, Any],
    source_path: Path,
    source_sha256: str,
    published_dir: Path,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    workload_records: list[dict[str, Any]] = []
    all_case_ids: set[str] = set()
    for workload in workloads:
        workload_name = workload["name"]
        workload_case_ids: list[str] = []
        geometry_hashes: list[str] = []
        for slot in range(workload["batch_size"]):
            identifier = _case_id(workload_name, slot)
            if identifier in all_case_ids:
                raise HistoricalInputError(f"duplicate generated case ID: {identifier}")
            all_case_ids.add(identifier)
            workload_case_ids.append(identifier)
            begin = workload["atom_offsets"][slot]
            stop = workload["atom_offsets"][slot + 1]
            numbers = workload["atomic_numbers"][begin:stop]
            positions = workload["positions_bohr"][3 * begin : 3 * stop]
            geometry_hash = _geometry_sha256(numbers, positions)
            geometry_hashes.append(geometry_hash)
            relative_input = f"inputs/{workload_name}/{identifier}.coord"
            coordinate_bytes = _coordinate_bytes(workload, slot)
            _write_bytes(stage / relative_input, coordinate_bytes)
            relative_golden = f"golden-placeholders/{identifier}.json"
            placeholder = {
                "case_id": identifier,
                "method": "GFN2-xTB",
                "units": UNITS,
                "molecular_charge": workload["molecular_charges"][slot],
                "unpaired_electrons": workload["unpaired_electrons"][slot],
                "properties": {},
                "provenance": {
                    "status": "placeholder_not_oracle",
                    "engine": None,
                    "note": (
                        "No model output or independent reference is represented "
                        "by this empty placeholder."
                    ),
                },
            }
            golden_bytes = (
                json.dumps(placeholder, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            _write_bytes(stage / relative_golden, golden_bytes)
            cases.append(
                {
                    "id": identifier,
                    "input": str(published_dir / relative_input),
                    "input_sha256": _sha256(coordinate_bytes),
                    "golden": str(published_dir / relative_golden),
                    "golden_sha256": _sha256(golden_bytes),
                    "atom_count": workload["natoms"],
                    "molecular_charge": workload["molecular_charges"][slot],
                    "unpaired_electrons": workload["unpaired_electrons"][slot],
                    "spin_channels": workload["spin_channels"][slot],
                    "source_workload": workload_name,
                    "source_slot": slot,
                    "geometry_sha256": geometry_hash,
                    "independent_reference_available": False,
                    "correctness_eligible": False,
                    "performance_eligible": False,
                }
            )
        layout = {
            "atom_offsets": workload["atom_offsets"],
            "case_ids": workload_case_ids,
            "geometry_sha256": geometry_hashes,
            "molecular_charges": workload["molecular_charges"],
            "natoms": workload["natoms"],
            "name": workload_name,
            "spin_channels": workload["spin_channels"],
            "unpaired_electrons": workload["unpaired_electrons"],
        }
        workload_records.append(
            {
                "name": workload_name,
                "natoms": workload["natoms"],
                "batch_size": workload["batch_size"],
                "seed": workload["seed"],
                "perturb_sigma_bohr": workload["perturb_sigma_bohr"],
                "case_ids": workload_case_ids,
                "source_layout_sha256": _sha256(_canonical_json_bytes(layout)),
                "geometry_sha256_by_slot": geometry_hashes,
            }
        )
    manifest = {
        "schema": "xtbloom-historical-alkane-assembler-inputs-v1",
        "diagnostic_only": True,
        "method": "GFN2-xTB",
        "units": UNITS,
        "eligibility": {
            "correctness": False,
            "performance": False,
            "reason": "empty golden placeholders and no independent reference",
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "source_commit": source["source_commit"],
            "source_hashes": source["source_hashes"],
            "resource_stub": source["resource_stub"],
            "workload_order": [item["name"] for item in workloads],
            "input_coordinate_field": "positions_bohr",
            "coordinate_unit_conversion": "none; source and output are bohr",
            "coordinate_serialization": (
                "Python binary64 repr; decimal parses back to the exact source float"
            ),
            "atom_order_conversion": "none",
            "atomic_number_conversion": (
                "element symbols only for $coord syntax; order and tags are retained"
            ),
            "charge_and_spin_conversion": (
                "none; molecular charge, unpaired electrons, "
                "and spin_channels copied by source slot"
            ),
            "slot_deduplication": (
                "none; every source slot has a distinct ID and payload, "
                "including repeated geometries"
            ),
        },
        "goldens": (
            "empty properties placeholders; not oracle data "
            "and not suitable for correctness qualification"
        ),
        "workloads": workload_records,
        "cases": cases,
    }
    return manifest


def materialize(
    input_path: Path, output_dir: Path, expected_sha256: str
) -> dict[str, Any]:
    """Create new assembler inputs, refusing replacement or partial output."""
    document, source_sha256, source_path = _load_source(input_path, expected_sha256)
    workloads, source = _validate_source(document)
    requested_output = output_dir.expanduser()
    if (
        not requested_output.name
        or requested_output.exists()
        or requested_output.is_symlink()
    ):
        raise HistoricalInputError(f"output directory must be new: {requested_output}")
    output = requested_output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise HistoricalInputError(f"output directory must be new: {output}")
    if not output.parent.is_dir():
        raise HistoricalInputError(f"output parent must already exist: {output.parent}")

    stage_root: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temporary:
            stage = Path(temporary)
            stage_root = stage
            manifest = _make_manifest(
                stage, workloads, source, source_path, source_sha256, output
            )
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            _write_bytes(stage / "manifest.json", manifest_bytes)
            digest_lines = []
            for artifact in sorted(path for path in stage.rglob("*") if path.is_file()):
                relative = artifact.relative_to(stage).as_posix()
                digest_lines.append(f"{_sha256(artifact.read_bytes())}  {relative}")
            checksum_bytes = ("\n".join(digest_lines) + "\n").encode("utf-8")
            _write_bytes(stage / "SHA256SUMS", checksum_bytes)
            _publish_stage_directory(stage, output)
            stage_root = None
    except OSError as exc:
        raise HistoricalInputError(f"cannot publish generated inputs: {exc}") from exc
    finally:
        if stage_root is not None and stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)

    checksum_path = output / "SHA256SUMS"
    return {
        "output_dir": str(output),
        "manifest_path": str(output / "manifest.json"),
        "source_sha256": source_sha256,
        "checksum_list_sha256": _sha256(checksum_path.read_bytes()),
        "case_count": len(manifest["cases"]),
        "workloads": [record["name"] for record in manifest["workloads"]],
    }


def build_parser() -> argparse.ArgumentParser:
    """Define an explicit CLI; source bytes are never accepted without a hash pin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="historical-inputs.json"
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="required SHA-256 of the exact input file bytes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to create; existing paths are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize inputs only and print paths and hashes for the capture owner."""
    args = build_parser().parse_args(argv)
    try:
        report = materialize(args.input, args.output_dir, args.expected_sha256)
    except HistoricalInputError as exc:
        sys.stderr.write(f"historical input conversion failed: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
