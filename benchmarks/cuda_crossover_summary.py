"""Strict offline component JSONL -> compact JSON and CSV, without GPU access.

Run ``python3 -m benchmarks.cuda_crossover_summary --input RAW --json-output
JSON --csv-output CSV``. Defaults require the complete 90-coordinate grid;
subsets must be specified explicitly with the original grid selectors because
the raw metadata records only the requested count, not the requested axes.
Producing provenance is copied from RAW, never replaced by the analyzer's HEAD.
Quantiles describe measurement pairs only and are not efficacy confidence bounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__:
    from . import cuda_crossover
else:
    import cuda_crossover

FIXTURE_SOURCES = (
    "tests/support/gfn2_scc_test_case.cpp",
    "tests/support/gfn2_scc_test_case.hpp",
    "tests/cuda_scc_iteration_production_test.cu",
)
SUMMARY_FIELDS = (
    "setup_ms",
    "fixture_arena_bytes",
    "fresh_checkpoint_bytes",
    "fixture_device_bytes",
    "chain_known_control_table_bytes",
    "tail_known_control_table_bytes",
    "retained_device_bytes",
    "retained_device_bytes_scope",
    "execution_counts",
    "graph_executable_counts",
)
LIMITATIONS = (
    "Component-only forced chain/tail CUDA event intervals, not full E2E latency.",
    "FRESH replay and one SCC iteration with a static deterministic active mask; "
    "not a declining-activity or converging SCC trajectory.",
    "Synthetic AO40/41/180 fixtures are not qualified healthy molecular endpoints.",
    "Matching unconverged ledgers establish replay parity, not SCC convergence.",
    "No public endpoint, force, independent oracle, threshold acceptance, benefit, "
    "or no-go decision follows from these component statistics.",
    "Native CUDA device/NVML inventory association remains UNVERIFIED.",
    "Retained bytes are a known subtotal excluding opaque CUDA graph/executable "
    "and provider allocations.",
    "Input geometry is identified by producing fixture source hashes; the raw "
    "harness does not record a materialized geometry hash.",
)
CSV_COLUMNS = (
    "ao_count",
    "batch_size",
    "active_denominator",
    "active_fraction_requested",
    "requested_active_system_count",
    "requested_mask_fraction",
    "active_system_count",
    "effective_active_fraction",
    "fixture",
    "status",
    "reason",
    "state_parity",
    "native_validation_status",
    "native_validation_error",
    "warmup_pairs_observed",
    "measurement_pairs_observed",
    "warmup_pairs_expected",
    "measurement_pairs_expected",
    "statistics_pairs_used",
    "statistics_pairs_unvalidated",
    "warmup_parity_pass_pairs",
    "warmup_parity_fail_pairs",
    "measurement_parity_pass_pairs",
    "measurement_parity_fail_pairs",
    "chain_ms_median",
    "chain_ms_p95",
    "tail_ms_median",
    "tail_ms_p95",
    "pair_ratio_median",
    "pair_ratio_p05",
    "pair_ratio_p95",
    "pair_ratio_defined_count",
    "pair_ratio_undefined_count",
    *SUMMARY_FIELDS[:-2],
    "chain_calls",
    "tail_calls",
    "total_calls",
    "chain_dispatch_family_executables",
    "chain_total_executables",
    "tail_total_executables",
    "measurement_ledger",
    "input_identity_sha256",
    "raw_sha256",
    "raw_byte_count",
    "producing_source_revision",
    "producing_binary_sha256",
    "producing_cache_sha256",
    "analyzer_source_revision",
    "analyzer_working_tree_dirty",
    "qualification",
    "claim_eligible",
    "native_selection_association",
)
TABLE_CSV_MAX_BYTES = 20_000
TABLE_CSV_COLUMNS = (
    "ao_count",
    "batch_size",
    "active_denominator",
    "status",
    "state_parity",
    "native_validation_status",
    "warmup_pairs_observed",
    "measurement_pairs_observed",
    "statistics_pairs_used",
    "statistics_pairs_unvalidated",
    "requested_active_system_count",
    "active_system_count",
    "pair_ratio_defined_count",
    "pair_ratio_undefined_count",
    "pair_ratio_median",
    "pair_ratio_p05",
    "pair_ratio_p95",
    "chain_ms_median",
    "tail_ms_median",
    "setup_ms",
    "fixture_device_bytes",
    "chain_known_control_table_bytes",
    "tail_known_control_table_bytes",
    "retained_device_bytes",
    "chain_calls",
    "tail_calls",
    "total_calls",
    "chain_total_executables",
    "tail_total_executables",
    "qualification",
    "claim_eligible",
)


def canonical_json(value: object) -> str:
    """Use stable key order and finite JSON numbers, without timestamps."""
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _coordinate(value: object) -> cuda_crossover.Coordinate:
    if not isinstance(value, dict) or set(value) != {
        "ao_count",
        "batch_size",
        "active_denominator",
    }:
        raise ValueError("coordinate must contain exactly the three grid dimensions")
    dimensions = [
        _integer(value[key], key, 1, 256)
        for key in ("ao_count", "batch_size", "active_denominator")
    ]
    return cuda_crossover.build_grid(
        (dimensions[0],), (dimensions[1],), (dimensions[2],)
    )[0]


def _check_metadata(metadata: dict[str, Any], coordinate_count: int) -> None:
    if (
        metadata.get("record_type") != "matrix_metadata"
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("benchmark") != "restricted_scc_component_crossover"
        or metadata.get("qualification") != "component_only"
    ):
        raise ValueError("expected version-1 component matrix metadata first")
    _integer(metadata.get("warmups"), "warmups", 0, cuda_crossover.MAX_WARMUPS)
    _integer(
        metadata.get("paired_samples"), "paired_samples", 1, cuda_crossover.MAX_SAMPLES
    )
    _integer(metadata.get("seed"), "seed", 0, (1 << 63) - 1)
    requested = _integer(
        metadata.get("coordinates_requested"), "coordinates_requested", 1, 90
    )
    if requested != coordinate_count:
        raise ValueError("requested coordinate count differs from the explicit grid")
    for key in ("timing_protocol", "native_command_template"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"missing producing protocol field {key}")
    source = metadata.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError("missing producing source identity")
    if "source_revision" not in source or (
        source["source_revision"] is not None
        and not isinstance(source["source_revision"], str)
    ):
        raise ValueError("producing source revision must be a string or null")
    if "working_tree_dirty" not in source or (
        source["working_tree_dirty"] is not None
        and type(source["working_tree_dirty"]) is not bool
    ):
        raise ValueError("producing dirty state must be boolean or null (unknown)")
    hashes = source.get("source_file_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("missing producing source hashes")
    for key in FIXTURE_SOURCES:
        _check_hash(hashes.get(key), f"producing fixture source {key}")
    _check_hash(source.get("binary_sha256"), "producing binary")
    if "cmake_cache_sha256" not in source:
        raise ValueError("missing producing cache identity (null is allowed)")
    if source["cmake_cache_sha256"] is not None:
        _check_hash(source["cmake_cache_sha256"], "producing cache")


def _check_hash(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must have a lowercase SHA-256 identity")


def analyzer_identity(root: Path) -> dict[str, Any]:
    """Identify this analyzer separately; unavailable Git never means clean."""
    status = cuda_crossover._git_value(
        root, "status", "--porcelain", "--untracked-files=all"
    )
    paths = (
        "benchmarks/cuda_crossover_summary.py",
        "benchmarks/test_cuda_crossover_summary.py",
        "benchmarks/cuda_crossover.py",
    )
    return {
        "source_revision": cuda_crossover._git_value(root, "rev-parse", "HEAD"),
        "source_branch": cuda_crossover._git_value(root, "branch", "--show-current"),
        "working_tree_dirty": None if status is None else bool(status),
        "source_file_sha256": {
            path: cuda_crossover._sha256(root / path)
            for path in paths
            if (root / path).is_file()
        },
        "python_version": sys.version.split()[0],
    }


def quantile(values: list[float], probability: float) -> float | None:
    """Interpolate sorted values at (n-1)*p; empty series has no quantile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _native_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross-check redundant raw copies rather than trusting the convenient one."""
    samples = result.get("samples")
    summary = result.get("summary")
    if (
        not isinstance(samples, list)
        or any(not isinstance(sample, dict) for sample in samples)
        or (summary is not None and not isinstance(summary, dict))
    ):
        raise ValueError("malformed wrapper samples or summary")
    records = result.get("native_records", [*samples, *([summary] if summary else [])])
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("malformed native_records")
    native_samples = [
        record for record in records if record.get("record_type") == "sample"
    ]
    native_summaries = [
        record
        for record in records
        if record.get("record_type") == "coordinate_summary"
    ]
    if canonical_json(native_samples) != canonical_json(samples) or (
        canonical_json(native_summaries) != canonical_json([summary] if summary else [])
    ):
        raise ValueError("wrapper/native record copies disagree")
    stdout = result.get("stdout")
    if stdout is not None:
        if not isinstance(stdout, str):
            raise ValueError("native stdout must be a string or null")
        try:
            parsed = cuda_crossover.parse_native_records(stdout)
        except cuda_crossover.NativeRecordParseError as error:
            if result.get("status") != "failure":
                raise ValueError(
                    "non-failed coordinate has truncated native stdout"
                ) from error
            parsed = error.records
        if canonical_json(parsed) != canonical_json(records):
            raise ValueError("native stdout/record copies disagree")
    return records


def _ledger(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in ("chain", "tail"):
        for suffix in ("statuses", "iterations", "converged"):
            counts: dict[str, int] = {}
            for sample in samples:
                for value in sample[f"{mode}_{suffix}"]:
                    key = str(value)
                    counts[key] = counts.get(key, 0) + 1
            result[f"{mode}_{suffix}_counts"] = counts
    return result


def _check_native_contract(
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    coordinate: cuda_crossover.Coordinate,
) -> None:
    """Reject numeric aliases and ledgers inconsistent with a one-body replay.

    The shared validator checks equality with the finite plan, but Python makes
    booleans and integral floats equal to integers. Keep the native integer
    schema strict before any records become usable for component statistics.
    """
    for field in ("ao_count", "batch_size"):
        _integer(summary.get(field), f"native {field}", 1, 256)
    if summary["coordinate_status"] == "unavailable":
        return
    for field, minimum, maximum in (
        ("active_denominator", 1, 4),
        ("active_system_count", 0, coordinate.batch_size),
        ("warmup_pairs", 0, cuda_crossover.MAX_WARMUPS),
        ("measured_pairs", 1, cuda_crossover.MAX_SAMPLES),
        ("maximum_iterations", 1, 1),
    ):
        _integer(summary.get(field), f"native {field}", minimum, maximum)
    if summary.get("fixture") != coordinate.fixture:
        raise ValueError("native input is not the declared one-body component fixture")
    for value in summary["active_mask"]:
        _integer(value, "native active_mask entry", 0, 1)
    for field in ("execution_counts", "graph_executable_counts"):
        for member, value in summary[field].items():
            _integer(value, f"native {field}.{member}", 0, (1 << 64) - 1)
    for sample in samples:
        for mode in ("chain", "tail"):
            for value in sample[f"{mode}_iterations"]:
                _integer(
                    value,
                    f"native {mode}_iterations entry",
                    0,
                    summary["maximum_iterations"],
                )


def _summarize_coordinate(
    result: dict[str, Any],
    coordinate: cuda_crossover.Coordinate,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    status = result.get("status")
    if status not in {"pass", "failure", "unavailable"}:
        raise ValueError("unknown wrapper coordinate status")
    if result.get("fixture") != coordinate.fixture:
        raise ValueError("wrapper fixture differs from its exact AO coordinate")
    if status != "pass" and (
        not isinstance(result.get("reason"), str) or not result["reason"]
    ):
        raise ValueError("failed/unavailable coordinate must retain its reason")
    records = _native_records(result)
    validation_error = None
    native_status = None
    summary = result.get("summary")
    samples = result["samples"]
    if not records and status == "unavailable":
        validation_status = "unavailable_without_native_measurements"
    else:
        try:
            native_status, _, summary, samples = cuda_crossover.validate_native_records(
                records,
                coordinate,
                metadata["warmups"],
                metadata["paired_samples"],
                metadata["seed"],
            )
            _check_native_contract(summary, samples, coordinate)
            validation_status = "complete"
        except ValueError as error:
            if status != "failure":
                raise ValueError(
                    f"{coordinate}: native validation failed: {error}"
                ) from error
            validation_status = "incomplete_or_invalid_failure"
            validation_error = str(error)
    if status == "pass" and (
        native_status != "pass"
        or type(result.get("subprocess_return_code")) is not int
        or result["subprocess_return_code"] != 0
        or result.get("timed_out") is not False
        or result.get("parse_diagnostic") is not None
        or result.get("reason") is not None
    ):
        raise ValueError("false passed wrapper ledger or subprocess outcome")
    if status == "unavailable" and (
        samples or native_status not in {None, "unavailable"}
    ):
        raise ValueError("unavailable coordinate contains measured results")
    validated = validation_status == "complete" and native_status != "unavailable"
    measurements = [
        sample for sample in samples if sample.get("sample_kind") == "measurement"
    ]
    warmups = [sample for sample in samples if sample.get("sample_kind") == "warmup"]
    row: dict[str, Any] = {
        **asdict(coordinate),
        "active_fraction_requested": coordinate.active_fraction,
        "fixture": coordinate.fixture,
        "status": status,
        "reason": result.get("reason"),
        "native_status": native_status,
        "native_validation_status": validation_status,
        "native_validation_error": validation_error,
        "parse_diagnostic": result.get("parse_diagnostic"),
        "subprocess_return_code": result.get("subprocess_return_code"),
        "timed_out": result.get("timed_out"),
        "state_parity": summary.get("state_parity") if validated else None,
        "warmup_pairs_observed": len(warmups),
        "measurement_pairs_observed": len(measurements),
        "warmup_pairs_expected": metadata["warmups"],
        "measurement_pairs_expected": metadata["paired_samples"],
        "statistics_pairs_used": len(measurements) if validated else 0,
        "statistics_pairs_unvalidated": len(measurements) if not validated else 0,
        "measurement_ledger": _ledger(measurements) if validated else None,
        "unvalidated_native_summary": summary if validation_error else None,
    }
    for phase, phase_samples in (("warmup", warmups), ("measurement", measurements)):
        for passed in (True, False):
            outcome = "pass" if passed else "fail"
            row[f"{phase}_parity_{outcome}_pairs"] = (
                sum(sample["state_parity"] is passed for sample in phase_samples)
                if validated
                else None
            )
    mask = cuda_crossover.deterministic_active_mask(
        coordinate.batch_size, coordinate.active_denominator, metadata["seed"]
    )
    input_identity = {
        "coordinate": asdict(coordinate),
        "fixture": coordinate.fixture,
        "seed": metadata["seed"],
        "requested_active_mask": mask,
        "molecular_charge_e": 1 if coordinate.ao_count == 41 else 0,
        "spin_channels": 1,
        "unpaired_electrons": 0,
        "fixture_source_sha256": {
            key: metadata["source_identity"]["source_file_sha256"][key]
            for key in FIXTURE_SOURCES
        },
        "maximum_iterations": 1,
    }
    row["input_identity_sha256"] = hashlib.sha256(
        canonical_json(input_identity).encode("utf-8")
    ).hexdigest()
    row["requested_active_system_count"] = sum(mask)
    row["requested_mask_fraction"] = sum(mask) / coordinate.batch_size
    row["active_system_count"] = (
        summary.get("active_system_count") if validated else None
    )
    row["effective_active_fraction"] = (
        summary.get("effective_active_fraction") if validated else None
    )
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field) if validated else None
    for mode in ("chain", "tail"):
        times = [sample[f"{mode}_ms"] for sample in measurements] if validated else []
        row[f"{mode}_ms_median"] = quantile(times, 0.5)
        row[f"{mode}_ms_p95"] = quantile(times, 0.95)
    ratios = []
    undefined = 0
    if validated:
        for sample in measurements:
            ratio = (
                sample["chain_ms"] / sample["tail_ms"]
                if sample["tail_ms"] > 0
                else math.inf
            )
            if math.isfinite(ratio):
                ratios.append(ratio)
            else:
                undefined += 1
    row.update(
        {
            "pair_ratio_median": quantile(ratios, 0.5),
            "pair_ratio_p05": quantile(ratios, 0.05),
            "pair_ratio_p95": quantile(ratios, 0.95),
            "pair_ratio_defined_count": len(ratios),
            "pair_ratio_undefined_count": undefined,
        }
    )
    return row


def summarize(
    raw_path: Path,
    coordinates: list[cuda_crossover.Coordinate] | None = None,
    *,
    analyzer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the entire requested grid; failures remain rows, never omissions.

    Incomplete failed native replays remain diagnostic rows without timing
    statistics. Incomplete outer JSONL or a passed-but-invalid ledger is rejected.
    The original metadata is retained verbatim, including its clean/dirty bit.
    """
    raw = raw_path.read_bytes()
    records = cuda_crossover.parse_native_records(raw.decode("utf-8"))
    canonical_json(records)
    if not records:
        raise ValueError("empty matrix input")
    planned = cuda_crossover.build_grid() if coordinates is None else coordinates
    if not planned or len(set(planned)) != len(planned):
        raise ValueError("requested coordinates must be nonempty and unique")
    for coordinate in planned:
        _coordinate(asdict(coordinate))
    metadata = records[0]
    _check_metadata(metadata, len(planned))
    observed: dict[cuda_crossover.Coordinate, dict[str, Any]] = {}
    for result in records[1:]:
        if result.get("record_type") != "coordinate_result":
            raise ValueError("duplicate metadata or unknown matrix record type")
        coordinate = _coordinate(result.get("coordinate"))
        if coordinate in observed:
            raise ValueError(f"duplicate coordinate: {coordinate}")
        if coordinate not in planned:
            raise ValueError(f"unrequested coordinate: {coordinate}")
        observed[coordinate] = result
    missing = set(planned) - set(observed)
    if missing:
        raise ValueError(f"missing requested coordinates: {sorted(map(str, missing))}")
    rows = [
        _summarize_coordinate(observed[coordinate], coordinate, metadata)
        for coordinate in sorted(
            planned,
            key=lambda item: (item.ao_count, item.batch_size, item.active_denominator),
        )
    ]
    identity = (
        analyzer
        if analyzer is not None
        else analyzer_identity(Path(__file__).resolve().parents[1])
    )
    return {
        "schema_version": 1,
        "benchmark": "restricted_scc_component_crossover_summary",
        "qualification": "component_only",
        "claim_eligible": False,
        "claim_decision": "not_available_without_required_public_endpoints",
        "native_selection_association": "UNVERIFIED",
        "limitations": LIMITATIONS,
        "raw_identity": {
            "path": str(raw_path.resolve()),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "producing_metadata": metadata,
        "analyzer_identity": identity,
        "analysis_qualification": (
            "clean_analyzer"
            if identity.get("working_tree_dirty") is False
            else "diagnostic_dirty_or_unknown_analyzer"
        ),
        "input_identity_protocol": (
            "SHA-256 of canonical JSON: coordinate, fixture, seed, requested "
            "SplitMix64/Fisher-Yates mask, charge/spin, producing fixture source "
            "hashes, and maximum_iterations=1; no materialized geometry identity "
            "is inferred"
        ),
        "statistics_protocol": {
            "samples": "measurement pairs only; warmups excluded from every statistic",
            "quantile": "linear interpolation of sorted values at (n-1)*p",
            "paired_ratio": (
                "chain_ms/tail_ms from the same measured pair; p05, median, p95 "
                "are descriptive quantiles, NOT confidence intervals or efficacy claims"
            ),
            "undefined_ratios": (
                "zero tail times or nonfinite ratios are counted explicitly"
            ),
            "uncertainty": "no bootstrap or statistical efficacy inference",
            "timing_boundary": metadata["timing_protocol"],
        },
        "coordinate_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("pass", "failure", "unavailable")
        },
        "coordinates": rows,
    }


def _csv_cell(value: object, *, compact: bool) -> object:
    """Keep full precision by default; round only compact display floats."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, bool)):
        return canonical_json(value)
    if compact and isinstance(value, float):
        return format(value, ".6g")
    return value


def csv_table(document: dict[str, Any], *, compact: bool = False) -> str:
    """Write stable CSV, optionally projecting a bounded display-only table.

    Compact floats use six significant digits; the external full summary keeps
    exact values, diagnostics, limitations, and producing/analyzer/raw identity.
    An oversized projection is rejected rather than dropping coordinates.
    """
    output = io.StringIO(newline="")
    columns = TABLE_CSV_COLUMNS if compact else CSV_COLUMNS
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    source = document["producing_metadata"]["source_identity"]
    for row in document["coordinates"]:
        calls = row["execution_counts"] or {}
        executables = row["graph_executable_counts"] or {}
        flattened = {
            **row,
            **calls,
            "chain_dispatch_family_executables": executables.get(
                "chain_dispatch_family"
            ),
            "chain_total_executables": executables.get("chain_total"),
            "tail_total_executables": executables.get("tail_total"),
            "raw_sha256": document["raw_identity"]["sha256"],
            "raw_byte_count": document["raw_identity"]["byte_count"],
            "producing_source_revision": source.get("source_revision"),
            "producing_binary_sha256": source["binary_sha256"],
            "producing_cache_sha256": source["cmake_cache_sha256"],
            "analyzer_source_revision": document["analyzer_identity"].get(
                "source_revision"
            ),
            "analyzer_working_tree_dirty": document["analyzer_identity"].get(
                "working_tree_dirty"
            ),
            "qualification": document["qualification"],
            "claim_eligible": document["claim_eligible"],
            "native_selection_association": document["native_selection_association"],
        }
        writer.writerow(
            {key: _csv_cell(flattened.get(key), compact=compact) for key in columns}
        )
    table = output.getvalue()
    if compact and len(table.encode("utf-8")) > TABLE_CSV_MAX_BYTES:
        raise ValueError(
            f"compact table exceeds {TABLE_CSV_MAX_BYTES}-byte limit; "
            "keep full outputs external instead"
        )
    return table


def main(argv: list[str] | None = None) -> int:
    """Validate before writing either output; reject raw/output path collisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument(
        "--table-csv-output",
        type=Path,
        help=(
            "optional <=20000-byte table; six-digit display floats, "
            "identity in full JSON"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ao-counts",
        type=lambda value: cuda_crossover._parse_integer_list(
            value, cuda_crossover.AO_COUNTS, "AO counts"
        ),
        default=cuda_crossover.AO_COUNTS,
    )
    parser.add_argument(
        "--batches",
        type=lambda value: cuda_crossover._parse_integer_list(
            value, cuda_crossover.BATCH_SIZES, "batch sizes"
        ),
        default=cuda_crossover.BATCH_SIZES,
    )
    parser.add_argument(
        "--active-fractions",
        type=cuda_crossover._parse_fractions,
        default=cuda_crossover.ACTIVE_DENOMINATORS,
    )
    args = parser.parse_args(argv)
    outputs = [args.json_output.resolve(), args.csv_output.resolve()]
    if args.table_csv_output is not None:
        outputs.append(args.table_csv_output.resolve())
    if len({args.input.resolve(), *outputs}) != len(outputs) + 1:
        parser.error("input and all output paths must be distinct")
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("output exists; select new paths or pass --overwrite")
    try:
        document = summarize(
            args.input,
            cuda_crossover.build_grid(
                args.ao_counts, args.batches, args.active_fractions
            ),
        )
        table = csv_table(document)
        compact_table = None
        if args.table_csv_output is not None:
            compact_table = csv_table(document, compact=True)
            payload = compact_table.encode("utf-8")
            document["table_csv_projection"] = {
                "path": str(outputs[2]),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "columns": TABLE_CSV_COLUMNS,
                "floating_point_format": ".6g",
                "maximum_bytes": TABLE_CSV_MAX_BYTES,
                "coordinate_count": len(document["coordinates"]),
                "authoritative_provenance_and_precision": "external full summary JSON",
            }
        serialized = canonical_json(document) + "\n"
        payloads = [serialized, table]
        if compact_table is not None:
            payloads.append(compact_table)
        for path, contents in zip(outputs, payloads, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    result = {
        "json_output": str(outputs[0]),
        "csv_output": str(outputs[1]),
        "coordinate_counts": document["coordinate_counts"],
        "claim_eligible": False,
    }
    if compact_table is not None:
        result["table_csv_output"] = str(outputs[2])
    sys.stdout.write(canonical_json(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
