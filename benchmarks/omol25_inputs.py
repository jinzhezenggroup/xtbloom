#!/usr/bin/env python3
"""Convert pinned OMol25 CSV/NPZ inputs into an external diagnostic corpus.

The input contract is the canonical OMol25 ragged archive
``xtbloom-omol25-ragged-v1`` plus its ordered CSV and source manifest v2. All
three source hashes are required from the caller. Spin channels are likewise
an explicit CLI choice; multiplicity and unpaired-electron counts are retained
as source metadata but never used to infer that choice.

The output is shaped for the existing finite-list public benchmark assembler,
not for scientific conformance: each case points to a shared empty
``reference-unavailable.json`` and has the diagnostic-only oracle role. The
converter does not calculate, copy, or invent model outputs. It writes no
input payload into the repository and refuses output roots inside any Git
worktree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

ao_grouping = importlib.import_module("benchmarks.ao_grouping")
conformance = importlib.import_module("tools.conformance.xtbloom_conformance")
ELEMENT_SYMBOLS = conformance.ELEMENT_SYMBOLS

GFN2_PARAMETERS = REPOSITORY_ROOT / "data" / "parameters" / "gfn2.json"
NPZ_FORMAT_VERSION = "xtbloom-omol25-ragged-v1"
SOURCE_MANIFEST_VERSION = 2
CONVERSION_SCHEMA = "xtbloom-omol25-inputs-diagnostic-v1"
UNITS = {
    "coordinates": "bohr",
    "energy": "hartree",
    "forces": "hartree/bohr",
    "gradient": "hartree/bohr",
}
ARRAY_NAMES = {
    "format_version",
    "sample_ids",
    "offsets",
    "atomic_numbers",
    "positions_bohr",
    "charges",
    "multiplicities",
    "unpaired_electrons",
    "natoms",
    "gfn2_n_ao",
    "input_sha256",
}
INTEGER_CSV_FIELDS = (
    "charge",
    "multiplicity",
    "unpaired_electrons",
    "natoms",
    "gfn2_n_ao",
)
HASH_CSV_FIELDS = (
    "xtbloom_input_sha256",
    "atomic_numbers_sha256",
    "positions_bohr_sha256",
)
PRESERVED_SOURCE_IDS = (
    "molecule_id",
    "molecule_group_id",
    "source_group_id",
    "source_configuration_hash",
    "source_structure_hash",
)
SAFE_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class OMol25InputError(ValueError):
    """A pinned source corpus or output destination violates the contract."""


@dataclass(frozen=True)
class SourceRecord:
    """One canonical row after CSV, NPZ, hashes, and exact AO agree."""

    case_id: str
    canonical_index: int
    atomic_numbers: tuple[int, ...]
    positions_bohr: tuple[tuple[float, float, float], ...]
    molecular_charge: int
    multiplicity: int
    unpaired_electrons: int
    ao_count: int
    source_input_sha256: str
    source_ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ConversionResult:
    """Stable locations and identities emitted by one conversion."""

    output_root: Path
    manifest_path: Path
    provenance_path: Path
    case_count: int
    manifest_sha256: str


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for an immutable byte sequence."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a source or converted file without loading it as one byte string."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read JSON while rejecting duplicate keys that could obscure metadata."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise OMol25InputError(f"duplicate JSON key {key!r} in {path}")
            document[key] = value
        return document

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OMol25InputError(
            f"cannot read JSON source manifest {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise OMol25InputError(f"source manifest {path} must contain a JSON object")
    return document


def _check_pinned_file(path: Path, expected: str, label: str) -> str:
    """Require the caller's SHA-256 pin to identify the exact source bytes."""
    if not SHA256_PATTERN.fullmatch(expected):
        raise OMol25InputError(f"{label} SHA-256 pin must contain 64 hex digits")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise OMol25InputError(f"cannot hash {label} {path}: {exc}") from exc
    if actual != expected.lower():
        raise OMol25InputError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )
    return actual


def _integer_field(row: dict[str, str], field: str, case_id: str) -> int:
    """Parse a CSV integer without accepting fractional or blank metadata."""
    value = row.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise OMol25InputError(f"case {case_id} CSV field {field!r} must be an integer")
    return int(value.strip(), 10)


def _validate_sha256(value: str, description: str) -> str:
    """Validate a data-carried digest before comparing it with source bytes."""
    if not SHA256_PATTERN.fullmatch(value):
        raise OMol25InputError(f"{description} must be a 64-digit SHA-256")
    return value.lower()


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load ordered source rows and reject ambiguous or incomplete CSV schema."""
    required = {
        "sample_set",
        "configuration_id",
        "charge",
        "multiplicity",
        "unpaired_electrons",
        "natoms",
        "gfn2_n_ao",
        "coordinate_output_unit",
        *HASH_CSV_FIELDS,
        "atomic_numbers_sha256",
        "positions_bohr_sha256",
    }
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            if not headers or len(set(headers)) != len(headers):
                raise OMol25InputError("CSV header must be present and unique")
            missing = sorted(required - set(headers))
            if missing:
                raise OMol25InputError(f"CSV is missing required columns: {missing}")
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise OMol25InputError(
                        f"CSV row {line_number} has a different field count "
                        "than its header"
                    )
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise OMol25InputError(f"cannot read source CSV {path}: {exc}") from exc
    if not rows:
        raise OMol25InputError("canonical CSV roster must not be empty")
    return rows, list(headers)


def _validate_source_manifest(
    document: dict[str, Any], row_count: int, rows: list[dict[str, str]]
) -> int:
    """Validate the canonical v2 source-order contract without guessing IDs."""
    if document.get("schema_version") != SOURCE_MANIFEST_VERSION:
        raise OMol25InputError(
            f"source manifest schema_version must be {SOURCE_MANIFEST_VERSION}"
        )
    sampling = document.get("sampling")
    if not isinstance(sampling, dict):
        raise OMol25InputError("source manifest must contain a sampling object")
    target_count = sampling.get("target_count")
    max_ao = sampling.get("max_gfn2_n_ao")
    sample_set_order = sampling.get("order")
    if type(target_count) is not int or target_count <= 0:
        raise OMol25InputError("source manifest sampling.target_count must be positive")
    if target_count != row_count:
        raise OMol25InputError(
            f"CSV roster has {row_count} rows but source manifest pins {target_count}"
        )
    if type(max_ao) is not int or max_ao <= 0:
        raise OMol25InputError(
            "source manifest sampling.max_gfn2_n_ao must be a positive integer"
        )
    if (
        not isinstance(sample_set_order, list)
        or not sample_set_order
        or any(not isinstance(value, str) or not value for value in sample_set_order)
        or len(set(sample_set_order)) != len(sample_set_order)
    ):
        raise OMol25InputError("source manifest sampling.order must be unique strings")
    rank = {name: index for index, name in enumerate(sample_set_order)}
    previous_rank = -1
    for index, row in enumerate(rows):
        sample_set = row.get("sample_set", "")
        if sample_set not in rank:
            raise OMol25InputError(
                f"CSV row {index} sample_set {sample_set!r} is absent "
                "from manifest order"
            )
        current_rank = rank[sample_set]
        if current_rank < previous_rank:
            raise OMol25InputError(
                "CSV row order disagrees with source manifest sampling.order"
            )
        previous_rank = current_rank
    return max_ao


def _load_npz(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load only the known non-object arrays from the pinned ragged NPZ."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise OMol25InputError("NPZ archive contains duplicate member names")
            if any(not name.endswith(".npy") for name in names):
                raise OMol25InputError("NPZ archive contains non-NPY members")
            invalid_member = archive.testzip()
            if invalid_member is not None:
                raise OMol25InputError(
                    f"NPZ member failed CRC validation: {invalid_member}"
                )
            npz_names = {name[:-4] for name in names}
            if npz_names != ARRAY_NAMES:
                raise OMol25InputError(
                    "NPZ array names differ from the canonical contract: "
                    f"missing={sorted(ARRAY_NAMES - npz_names)}, "
                    f"unexpected={sorted(npz_names - ARRAY_NAMES)}"
                )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except OMol25InputError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OMol25InputError(f"cannot read canonical NPZ {path}: {exc}") from exc
    return arrays


def _require_dtype(
    arrays: dict[str, np.ndarray[Any, Any]], name: str, expected: str
) -> None:
    """Require exact integer/float widths and Unicode-only text arrays."""
    dtype = arrays[name].dtype
    valid = dtype.kind == "U" if expected == "unicode" else dtype == np.dtype(expected)
    if not valid:
        raise OMol25InputError(f"NPZ {name} has dtype {dtype.str}; expected {expected}")


def _validated_records(
    rows: list[dict[str, str]],
    csv_headers: list[str],
    arrays: dict[str, np.ndarray[Any, Any]],
    max_ao: int,
    basis_counts: dict[int, int],
) -> list[SourceRecord]:
    """Validate roster order, ragged extents, atom metadata, and per-ID hashes."""
    count = len(rows)
    for name in ("format_version", "sample_ids", "input_sha256"):
        _require_dtype(arrays, name, "unicode")
    for name in ("offsets",):
        _require_dtype(arrays, name, "<i8")
    for name in (
        "atomic_numbers",
        "charges",
        "multiplicities",
        "unpaired_electrons",
        "natoms",
        "gfn2_n_ao",
    ):
        _require_dtype(arrays, name, "<i4")
    _require_dtype(arrays, "positions_bohr", "<f8")

    if (
        arrays["format_version"].shape != ()
        or str(arrays["format_version"].item()) != NPZ_FORMAT_VERSION
    ):
        raise OMol25InputError(f"NPZ format_version must be {NPZ_FORMAT_VERSION!r}")
    per_record_shapes = {
        "sample_ids": (count,),
        "charges": (count,),
        "multiplicities": (count,),
        "unpaired_electrons": (count,),
        "natoms": (count,),
        "gfn2_n_ao": (count,),
        "input_sha256": (count,),
    }
    for name, shape in per_record_shapes.items():
        if arrays[name].shape != shape:
            raise OMol25InputError(
                f"NPZ {name} has shape {arrays[name].shape}; expected {shape}"
            )
    offsets = arrays["offsets"]
    numbers = arrays["atomic_numbers"]
    positions = arrays["positions_bohr"]
    if offsets.shape != (count + 1,):
        raise OMol25InputError(
            f"NPZ offsets has shape {offsets.shape}; expected {(count + 1,)}"
        )
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise OMol25InputError("NPZ positions_bohr must have shape (total_atoms, 3)")
    if numbers.ndim != 1 or positions.shape[0] != numbers.shape[0]:
        raise OMol25InputError("NPZ atom and coordinate extents do not match")
    if offsets[0] != 0 or offsets[-1] != numbers.size:
        raise OMol25InputError("NPZ offsets must span the full flattened atom arrays")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise OMol25InputError(
            "NPZ offsets must strictly increase for nonempty molecules"
        )
    if not np.isfinite(positions).all():
        raise OMol25InputError("NPZ positions_bohr contains non-finite coordinates")
    if np.any(numbers <= 0) or np.any(numbers >= len(ELEMENT_SYMBOLS)):
        raise OMol25InputError("NPZ atomic_numbers contains unsupported elements")
    if np.any(arrays["multiplicities"] <= 0):
        raise OMol25InputError("NPZ multiplicities must be positive integers")
    if np.any(arrays["unpaired_electrons"] < 0):
        raise OMol25InputError("NPZ unpaired_electrons must be nonnegative integers")
    if np.any(arrays["natoms"] <= 0) or np.any(arrays["gfn2_n_ao"] <= 0):
        raise OMol25InputError("NPZ atom and AO counts must be positive")

    ids = [row.get("configuration_id", "") for row in rows]
    for index, case_id in enumerate(ids):
        if not SAFE_CASE_ID.fullmatch(case_id):
            raise OMol25InputError(
                f"CSV row {index} has unsafe configuration_id {case_id!r}"
            )
    if len(set(ids)) != len(ids):
        raise OMol25InputError("canonical CSV configuration_id values must be unique")
    npz_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    if npz_ids != ids:
        raise OMol25InputError("NPZ sample_ids do not match full CSV roster order")

    records: list[SourceRecord] = []
    for index, (row, case_id) in enumerate(zip(rows, ids, strict=True)):
        csv_values = {
            name: _integer_field(row, name, case_id) for name in INTEGER_CSV_FIELDS
        }
        charge = csv_values["charge"]
        multiplicity = csv_values["multiplicity"]
        unpaired = csv_values["unpaired_electrons"]
        atom_count = csv_values["natoms"]
        csv_ao_count = csv_values["gfn2_n_ao"]
        if multiplicity <= 0 or unpaired < 0 or atom_count <= 0 or csv_ao_count <= 0:
            raise OMol25InputError(f"case {case_id} has invalid integer metadata")
        if row.get("coordinate_output_unit", "").strip().lower() != "bohr":
            raise OMol25InputError(
                f"case {case_id} coordinate_output_unit must be 'bohr'"
            )
        source_hashes = {
            field: _validate_sha256(row[field].strip(), f"case {case_id} {field}")
            for field in HASH_CSV_FIELDS
        }
        if int(arrays["charges"][index]) != charge:
            raise OMol25InputError(f"case {case_id} CSV charge disagrees with NPZ")
        if int(arrays["multiplicities"][index]) != multiplicity:
            raise OMol25InputError(
                f"case {case_id} CSV multiplicity disagrees with NPZ"
            )
        if int(arrays["unpaired_electrons"][index]) != unpaired:
            raise OMol25InputError(
                f"case {case_id} CSV unpaired_electrons disagrees with NPZ"
            )
        if int(arrays["natoms"][index]) != atom_count:
            raise OMol25InputError(f"case {case_id} CSV natoms disagrees with NPZ")

        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop - start != atom_count:
            raise OMol25InputError(f"case {case_id} offsets disagree with atom count")
        case_numbers = numbers[start:stop]
        case_positions = positions[start:stop]
        if not np.isfinite(case_positions).all():
            raise OMol25InputError(f"case {case_id} contains non-finite coordinates")
        number_hash = sha256_bytes(
            np.asarray(case_numbers, dtype="<i4").tobytes(order="C")
        )
        position_hash = sha256_bytes(
            np.asarray(case_positions, dtype="<f8").tobytes(order="C")
        )
        if number_hash != source_hashes["atomic_numbers_sha256"]:
            raise OMol25InputError(f"case {case_id} atomic_numbers SHA-256 mismatch")
        if position_hash != source_hashes["positions_bohr_sha256"]:
            raise OMol25InputError(f"case {case_id} positions_bohr SHA-256 mismatch")
        npz_input_hash = _validate_sha256(
            str(arrays["input_sha256"][index]), f"case {case_id} NPZ input_sha256"
        )
        if npz_input_hash != source_hashes["xtbloom_input_sha256"]:
            raise OMol25InputError(f"case {case_id} CSV/NPZ input SHA-256 mismatch")

        exact_ao_count = ao_grouping.count_gfn2_aos(
            [int(value) for value in case_numbers], basis_counts
        )
        if exact_ao_count > max_ao:
            raise OMol25InputError(
                f"case {case_id} exact AO count {exact_ao_count} exceeds "
                f"source manifest limit {max_ao}"
            )
        if exact_ao_count != csv_ao_count or exact_ao_count != int(
            arrays["gfn2_n_ao"][index]
        ):
            raise OMol25InputError(
                f"case {case_id} exact GFN2 AO count disagrees with CSV/NPZ metadata"
            )
        source_ids = tuple(
            (name, row[name])
            for name in PRESERVED_SOURCE_IDS
            if name in csv_headers and row.get(name, "") != ""
        )
        records.append(
            SourceRecord(
                case_id=case_id,
                canonical_index=index,
                atomic_numbers=tuple(int(value) for value in case_numbers),
                positions_bohr=tuple(
                    (float(x), float(y), float(z)) for x, y, z in case_positions
                ),
                molecular_charge=charge,
                multiplicity=multiplicity,
                unpaired_electrons=unpaired,
                ao_count=exact_ao_count,
                source_input_sha256=npz_input_hash,
                source_ids=source_ids,
            )
        )
    return records


def _json_bytes(document: dict[str, Any]) -> bytes:
    """Serialize stable UTF-8 JSON with deterministic key order and LF ending."""
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _check_external_output_root(output_root: Path) -> tuple[Path, Path]:
    """Require a fresh destination outside the repository and all Git roots."""
    resolved = output_root.expanduser().resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise OMol25InputError("output root must be outside the xTBloom repository")
    if resolved.exists():
        raise OMol25InputError(
            f"output root already exists; refusing overwrite: {resolved}"
        )
    parent = resolved.parent
    existing_parent = parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    try:
        existing_parent = existing_parent.resolve(strict=True)
    except OSError as exc:
        raise OMol25InputError(f"cannot resolve output parent {parent}: {exc}") from exc
    for ancestor in (existing_parent, *existing_parent.parents):
        if (ancestor / ".git").exists():
            raise OMol25InputError(f"output root is inside a Git worktree: {resolved}")
    return resolved, parent


def _coord_bytes(record: SourceRecord) -> bytes:
    """Serialize coordinates in the assembler's atomic-unit $coord syntax."""
    lines = ["$coord"]
    for atomic_number, position in zip(
        record.atomic_numbers, record.positions_bohr, strict=True
    ):
        symbol = ELEMENT_SYMBOLS[atomic_number].lower()
        lines.append(" ".join([*(format(value, ".17g") for value in position), symbol]))
    lines.append("$end")
    return ("\n".join(lines) + "\n").encode("ascii")


def _id_list_bytes(case_ids: Sequence[str]) -> bytes:
    """Serialize one finite ordered selection as one ID per line."""
    return ("".join(f"{case_id}\n" for case_id in case_ids)).encode("ascii")


def convert_dataset(
    csv_path: Path,
    npz_path: Path,
    source_manifest_path: Path,
    output_root: Path,
    *,
    csv_sha256: str,
    npz_sha256: str,
    source_manifest_sha256: str,
    spin_channels: int,
) -> ConversionResult:
    """Validate pinned sources and atomically emit an external diagnostic input set.

    ``spin_channels`` is intentionally a required keyword. The NPZ format has
    no channel field, and multiplicity must not be used as a proxy. Exact AO
    counts are recomputed from this checkout's generated GFN2 parameter JSON.
    The output directory must not already exist so this call cannot overwrite
    another agent's or user's artifacts.
    """
    if type(spin_channels) is not int or spin_channels not in (1, 2):
        raise OMol25InputError("spin_channels must be explicitly set to 1 or 2")
    csv_path = csv_path.expanduser().resolve()
    npz_path = npz_path.expanduser().resolve()
    source_manifest_path = source_manifest_path.expanduser().resolve()
    csv_digest = _check_pinned_file(csv_path, csv_sha256, "CSV")
    npz_digest = _check_pinned_file(npz_path, npz_sha256, "NPZ")
    source_digest = _check_pinned_file(
        source_manifest_path, source_manifest_sha256, "source manifest"
    )
    rows, csv_headers = _load_csv(csv_path)
    source_document = _read_json_object(source_manifest_path)
    max_ao = _validate_source_manifest(source_document, len(rows), rows)
    arrays = _load_npz(npz_path)
    basis_counts, basis_digest = ao_grouping.load_gfn2_basis_ao_counts(GFN2_PARAMETERS)
    try:
        records = _validated_records(rows, csv_headers, arrays, max_ao, basis_counts)
    except ao_grouping.AOGroupingError as exc:
        raise OMol25InputError(
            f"cannot derive exact AO counts from GFN2 parameters: {exc}"
        ) from exc
    resolved_root, parent = _check_external_output_root(output_root)
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OMol25InputError(f"cannot create output parent {parent}: {exc}") from exc

    ids = [record.case_id for record in records]
    ao_by_id = {record.case_id: record.ao_count for record in records}
    id_views = {
        "all": ids,
        "small": [case_id for case_id in ids if 1 <= ao_by_id[case_id] <= 38],
        "medium": [case_id for case_id in ids if 39 <= ao_by_id[case_id] <= 69],
        "large": [case_id for case_id in ids if 70 <= ao_by_id[case_id] <= 180],
        "first256": ids[:256],
        "last256": ids[-256:],
    }
    unknown_ao = sorted(case_id for case_id in ids if not 1 <= ao_by_id[case_id] <= 180)
    if unknown_ao:
        raise OMol25InputError(
            "exact AO count falls outside the requested 1-180 views: "
            + ", ".join(unknown_ao[:8])
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{resolved_root.name}.staging-", dir=parent)
    )
    try:
        inputs_dir = staging_root / "inputs"
        ids_dir = staging_root / "id-lists"
        inputs_dir.mkdir()
        ids_dir.mkdir()
        manifest_cases: list[dict[str, Any]] = []
        converted_files: dict[str, str] = {}
        final_inputs_dir = resolved_root / "inputs"
        final_reference_path = resolved_root / "reference-unavailable.json"
        for record in records:
            stem = f"{record.canonical_index:04d}_{record.case_id}"
            relative_input = Path("inputs") / f"{stem}.coord"
            coord_payload = _coord_bytes(record)
            (staging_root / relative_input).write_bytes(coord_payload)
            converted_files[relative_input.as_posix()] = sha256_bytes(coord_payload)
            case: dict[str, Any] = {
                "id": record.case_id,
                "canonical_index": record.canonical_index,
                "input": str(final_inputs_dir / relative_input.name),
                "input_sha256": sha256_bytes(coord_payload),
                "golden": str(final_reference_path),
                "golden_sha256": "",
                "atom_count": len(record.atomic_numbers),
                "gfn2_n_ao": record.ao_count,
                "molecular_charge": record.molecular_charge,
                "multiplicity": record.multiplicity,
                "unpaired_electrons": record.unpaired_electrons,
                "spin_channels": spin_channels,
                "spin_channels_source": "explicit --spin-channels CLI argument",
                "source_xtbloom_input_sha256": record.source_input_sha256,
                "oracle_role": "diagnostic-no-independent-reference",
                "xtbloom_oracle_properties": [],
                "xtbloom_backends": ["cpu", "cuda"],
                "independent_reference_available": False,
                "qualification": {
                    "scientific_correctness_qualified": False,
                    "performance_claim_eligible": False,
                    "oracle_comparison_available": False,
                },
            }
            case.update(dict(record.source_ids))
            manifest_cases.append(case)

        reference_unavailable = {
            "schema_version": 1,
            "method": "GFN2-xTB",
            "units": UNITS,
            "properties": {},
            "independent_reference_available": False,
            "qualification": {
                "scientific_correctness_qualified": False,
                "performance_claim_eligible": False,
            },
            "provenance": {
                "status": "reference-unavailable",
                "note": (
                    "Input conversion only; no scientific reference "
                    "or model result was created."
                ),
            },
        }
        golden_payload = _json_bytes(reference_unavailable)
        (staging_root / "reference-unavailable.json").write_bytes(golden_payload)
        golden_digest = sha256_bytes(golden_payload)
        converted_files["reference-unavailable.json"] = golden_digest
        for case in manifest_cases:
            case["golden_sha256"] = golden_digest

        for view_name, view_ids in id_views.items():
            relative_list = Path("id-lists") / f"{view_name}.txt"
            payload = _id_list_bytes(view_ids)
            (staging_root / relative_list).write_bytes(payload)
            converted_files[relative_list.as_posix()] = sha256_bytes(payload)

        manifest = {
            "schema": CONVERSION_SCHEMA,
            "schema_version": 1,
            "golden_schema_version": 1,
            "method": "GFN2-xTB",
            "units": UNITS,
            "diagnostic_only": True,
            "independent_reference_available": False,
            "qualification": {
                "scientific_correctness_qualified": False,
                "performance_claim_eligible": False,
                "oracle_comparison_available": False,
            },
            "source_csv_sha256": csv_digest,
            "source_npz_sha256": npz_digest,
            "source_manifest_sha256": source_digest,
            "gfn2_parameters_sha256": basis_digest,
            "spin_channels": spin_channels,
            "spin_channels_source": (
                "explicit --spin-channels CLI argument; not inferred from multiplicity"
            ),
            "case_id_order": (
                "canonical CSV row order, cross-checked against NPZ sample_ids"
            ),
            "cases": manifest_cases,
        }
        manifest_payload = _json_bytes(manifest)
        (staging_root / "manifest.json").write_bytes(manifest_payload)
        converted_files["manifest.json"] = sha256_bytes(manifest_payload)

        provenance = {
            "schema": "xtbloom-omol25-conversion-provenance-v1",
            "conversion_schema": CONVERSION_SCHEMA,
            "source_inputs": {
                "csv": {
                    "path": str(csv_path),
                    "size_bytes": csv_path.stat().st_size,
                    "sha256": csv_digest,
                },
                "npz": {
                    "path": str(npz_path),
                    "size_bytes": npz_path.stat().st_size,
                    "sha256": npz_digest,
                    "format_version": NPZ_FORMAT_VERSION,
                    "arrays": sorted(ARRAY_NAMES),
                },
                "source_manifest": {
                    "path": str(source_manifest_path),
                    "size_bytes": source_manifest_path.stat().st_size,
                    "sha256": source_digest,
                    "schema_version": SOURCE_MANIFEST_VERSION,
                },
            },
            "gfn2_parameters": {
                "path": str(GFN2_PARAMETERS),
                "sha256": basis_digest,
            },
            "spin_channels": {
                "value": spin_channels,
                "source": "explicit CLI argument",
            },
            "canonical_roster": {
                "row_count": len(records),
                "first_id": ids[0],
                "last_id": ids[-1],
                "ordered_ids_sha256": sha256_bytes(_id_list_bytes(ids)),
                "retained_source_id_fields": [
                    name for name in PRESERVED_SOURCE_IDS if name in csv_headers
                ],
                "no_inferred_molecule_or_near_duplicate_groups": True,
            },
            "id_lists": {
                name: {
                    "path": f"id-lists/{name}.txt",
                    "count": len(view_ids),
                    "sha256": converted_files[f"id-lists/{name}.txt"],
                }
                for name, view_ids in id_views.items()
            },
            "converted_files_sha256": dict(sorted(converted_files.items())),
            "provenance_file_self_hash_excluded": True,
            "independent_reference_available": False,
            "qualification": {
                "scientific_correctness_qualified": False,
                "performance_claim_eligible": False,
            },
        }
        (staging_root / "provenance.json").write_bytes(_json_bytes(provenance))

        if resolved_root.exists():
            raise OMol25InputError(
                "output root appeared during conversion; "
                f"refusing overwrite: {resolved_root}"
            )
        os.rename(staging_root, resolved_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    manifest_path = resolved_root / "manifest.json"
    return ConversionResult(
        output_root=resolved_root,
        manifest_path=manifest_path,
        provenance_path=resolved_root / "provenance.json",
        case_count=len(records),
        manifest_sha256=sha256_file(manifest_path),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the source-pinned CLI; none of the identity pins has a default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="canonical ordered CSV")
    parser.add_argument("--npz", type=Path, required=True, help="canonical ragged NPZ")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="canonical source manifest JSON",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="new root outside Git"
    )
    parser.add_argument(
        "--csv-sha256", required=True, help="expected canonical CSV SHA-256"
    )
    parser.add_argument(
        "--npz-sha256", required=True, help="expected canonical NPZ SHA-256"
    )
    parser.add_argument(
        "--source-manifest-sha256",
        required=True,
        help="expected source manifest SHA-256",
    )
    parser.add_argument(
        "--spin-channels",
        type=int,
        choices=(1, 2),
        required=True,
        help="explicitly selected spin-channel count; never inferred from multiplicity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the conversion and report only paths and frozen input identities."""
    args = build_parser().parse_args(argv)
    try:
        result = convert_dataset(
            args.csv,
            args.npz,
            args.source_manifest,
            args.output_root,
            csv_sha256=args.csv_sha256,
            npz_sha256=args.npz_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            spin_channels=args.spin_channels,
        )
    except (OMol25InputError, ao_grouping.AOGroupingError, OSError, ValueError) as exc:
        print(  # noqa: T201 - CLI failure diagnostic
            f"omol25 input conversion failed: {exc}", file=sys.stderr
        )
        return 2
    print(  # noqa: T201 - machine-readable CLI result
        json.dumps(
            {
                "status": "diagnostic-input-conversion-only",
                "case_count": result.case_count,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "provenance": str(result.provenance_path),
                "independent_reference_available": False,
                "qualification": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
