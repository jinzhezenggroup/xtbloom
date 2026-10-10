"""Validate and summarize instrumented CUDA SCC diagnostic JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TextIO


SCHEMA_NAME = "xtbloom.cuda.scc_diagnostics.v1"
REPORT_SCHEMA_NAME = "xtbloom.cuda.scc_diagnostics.report.v1"
EXECUTION_MODES = frozenset(
    {"device_dispatch_chain", "device_tail_graph", "bounded_fallback"}
)
FIXED_CAPACITY_MODES = frozenset({"device_tail_graph", "bounded_fallback"})
OPTIONAL_CALL_FIELDS = frozenset(
    {
        "plan_token",
        "wavefunction_layout_fingerprint",
        "start_policy",
        "graph_preference",
        "chain_measured_ao_bound",
        "diagnostic_device_bytes",
        "chain_selection_constraints",
        "terminal_buckets",
        "terminal_systems",
    }
)
NUMERIC_METADATA_FIELDS = frozenset(
    {
        "plan_token",
        "wavefunction_layout_fingerprint",
        "chain_measured_ao_bound",
        "diagnostic_device_bytes",
    }
)
TERMINAL_COUNT_FIELDS = (
    "converged_systems",
    "failed_systems",
    "exhausted_systems",
    "unfinished_systems",
)
SUCCESS_STATUS = 0
SCC_NOT_CONVERGED_STATUS = 7


class DiagnosticError(ValueError):
    """An invalid diagnostic JSONL record or schema value."""


def _object(
    value: object,
    field: str,
    expected_keys: set[str],
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise DiagnosticError(f"{field} must be an object")
    allowed_keys = expected_keys | (optional_keys or set())
    missing = expected_keys.difference(value)
    unexpected = set(value).difference(allowed_keys)
    if missing:
        raise DiagnosticError(
            f"{field} is missing fields: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise DiagnosticError(
            f"{field} has unknown fields: {', '.join(sorted(unexpected))}"
        )
    return value


def _array(value: object, field: str) -> list[Any]:
    if type(value) is not list:
        raise DiagnosticError(f"{field} must be an array")
    return value


def _integer(value: object, field: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DiagnosticError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise DiagnosticError(f"{field} must be at least {minimum}")
    return value


def _count(value: object, field: str) -> int:
    return _integer(value, field, minimum=0)


def _slot_count(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _count(value, field)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DiagnosticError(f"non-standard JSON constant {value!r} is not allowed")


def _terminal_class(status: int, converged: int) -> str:
    """Classify public SCC snapshots without treating unfinished as failure."""
    if converged == 1:
        return "converged_systems"
    if status == SCC_NOT_CONVERGED_STATUS:
        return "exhausted_systems"
    if status != SUCCESS_STATUS:
        return "failed_systems"
    return "unfinished_systems"


def validate_call(value: object) -> dict[str, Any]:
    """Validate one v1 call object and return it without coercing its values."""
    call = _object(
        value,
        "call",
        {
            "schema",
            "instrumented",
            "execution_mode",
            "fallback_reason",
            "batch_size",
            "maximum_iterations",
            "buckets",
            "iterations",
        },
        optional_keys=set(OPTIONAL_CALL_FIELDS),
    )
    if call["schema"] != SCHEMA_NAME:
        raise DiagnosticError(f"schema must equal {SCHEMA_NAME!r}")
    if call["instrumented"] is not True:
        raise DiagnosticError("instrumented must be true")

    execution_mode = call["execution_mode"]
    if type(execution_mode) is not str or execution_mode not in EXECUTION_MODES:
        choices = ", ".join(sorted(EXECUTION_MODES))
        raise DiagnosticError(f"execution_mode must be one of: {choices}")

    _integer(call["fallback_reason"], "fallback_reason")
    batch_size = _count(call["batch_size"], "batch_size")
    maximum_iterations = _integer(
        call["maximum_iterations"], "maximum_iterations", minimum=1
    )
    for field in NUMERIC_METADATA_FIELDS.intersection(call):
        _count(call[field], field)
    for field, allowed_values in (
        ("start_policy", frozenset({0, 1, 2})),
        ("graph_preference", frozenset({0, 1, 2})),
    ):
        if field in call:
            value = _integer(call[field], field)
            if value not in allowed_values:
                choices = ", ".join(str(choice) for choice in sorted(allowed_values))
                raise DiagnosticError(f"{field} must be one of: {choices}")
    if "chain_selection_constraints" in call:
        constraints = _array(
            call["chain_selection_constraints"], "chain_selection_constraints"
        )
        for index, constraint in enumerate(constraints):
            if type(constraint) is not str:
                raise DiagnosticError(
                    f"chain_selection_constraints[{index}] must be a string"
                )

    bucket_values = _array(call["buckets"], "buckets")
    bucket_metadata: list[dict[str, int]] = []
    for position, value in enumerate(bucket_values):
        field = f"buckets[{position}]"
        bucket = _object(
            value,
            field,
            {"bucket_index", "ao", "system_capacity", "channel_capacity"},
        )
        bucket_index = _count(bucket["bucket_index"], f"{field}.bucket_index")
        ao = _integer(bucket["ao"], f"{field}.ao", minimum=1)
        system_capacity = _integer(
            bucket["system_capacity"], f"{field}.system_capacity", minimum=1
        )
        channel_capacity = _integer(
            bucket["channel_capacity"], f"{field}.channel_capacity", minimum=1
        )
        if not system_capacity <= channel_capacity <= 2 * system_capacity:
            raise DiagnosticError(
                f"{field}.channel_capacity must be between system_capacity and "
                "twice system_capacity"
            )
        if bucket_index != position:
            raise DiagnosticError(
                "bucket metadata indices must be unique and contiguous from zero"
            )
        bucket_metadata.append(
            {
                "bucket_index": bucket_index,
                "ao": ao,
                "system_capacity": system_capacity,
                "channel_capacity": channel_capacity,
            }
        )

    if sum(bucket["system_capacity"] for bucket in bucket_metadata) != batch_size:
        raise DiagnosticError("bucket system capacities must sum to batch_size")
    if batch_size == 0 and bucket_metadata:
        raise DiagnosticError("an empty batch must not declare buckets")

    terminal_bucket_values: list[dict[str, int]] | None = None
    if "terminal_buckets" in call:
        values = _array(call["terminal_buckets"], "terminal_buckets")
        if len(values) != len(bucket_metadata):
            raise DiagnosticError(
                "terminal_buckets must contain every metadata bucket exactly once"
            )
        terminal_bucket_values = []
        for position, value in enumerate(values):
            field = f"terminal_buckets[{position}]"
            terminal_bucket = _object(
                value,
                field,
                {"bucket_index", *TERMINAL_COUNT_FIELDS},
            )
            bucket_index = _count(
                terminal_bucket["bucket_index"], f"{field}.bucket_index"
            )
            if bucket_index != position:
                raise DiagnosticError(
                    "terminal bucket indices must match contiguous metadata indices"
                )
            counts = {
                name: _count(terminal_bucket[name], f"{field}.{name}")
                for name in TERMINAL_COUNT_FIELDS
            }
            if sum(counts.values()) != bucket_metadata[position]["system_capacity"]:
                raise DiagnosticError(
                    f"{field} terminal classes must sum to system_capacity"
                )
            terminal_bucket_values.append({"bucket_index": bucket_index, **counts})

    terminal_system_values: list[dict[str, int]] | None = None
    terminal_system_totals = dict.fromkeys(TERMINAL_COUNT_FIELDS, 0)
    if "terminal_systems" in call:
        values = _array(call["terminal_systems"], "terminal_systems")
        if len(values) != batch_size:
            raise DiagnosticError("terminal_systems must contain every system once")
        terminal_system_values = []
        for position, value in enumerate(values):
            field = f"terminal_systems[{position}]"
            terminal_system = _object(
                value,
                field,
                {"system_index", "iterations", "status", "converged"},
            )
            system_index = _count(
                terminal_system["system_index"], f"{field}.system_index"
            )
            if system_index != position:
                raise DiagnosticError(
                    "terminal system indices must be unique and contiguous from zero"
                )
            system_iterations = _count(
                terminal_system["iterations"], f"{field}.iterations"
            )
            if system_iterations > maximum_iterations:
                raise DiagnosticError(
                    f"{field}.iterations cannot exceed maximum_iterations"
                )
            status = _count(terminal_system["status"], f"{field}.status")
            converged = _integer(terminal_system["converged"], f"{field}.converged")
            if converged not in (0, 1):
                raise DiagnosticError(f"{field}.converged must be 0 or 1")
            if converged == 1 and status != SUCCESS_STATUS:
                raise DiagnosticError(
                    f"{field}.converged must be zero unless status is success"
                )
            category = _terminal_class(status, converged)
            terminal_system_totals[category] += 1
            terminal_system_values.append(
                {
                    "system_index": system_index,
                    "iterations": system_iterations,
                    "status": status,
                    "converged": converged,
                }
            )

    if terminal_bucket_values is not None and terminal_system_values is not None:
        terminal_bucket_totals = {
            name: sum(bucket[name] for bucket in terminal_bucket_values)
            for name in TERMINAL_COUNT_FIELDS
        }
        if terminal_bucket_totals != terminal_system_totals:
            raise DiagnosticError(
                "terminal_buckets totals must match terminal_systems classifications"
            )

    iteration_values = _array(call["iterations"], "iterations")
    if len(iteration_values) > maximum_iterations:
        raise DiagnosticError("iterations cannot exceed maximum_iterations")
    if (
        execution_mode == "bounded_fallback"
        and batch_size > 0
        and len(iteration_values) != maximum_iterations
    ):
        raise DiagnosticError(
            "bounded_fallback must record every maximum_iterations body"
        )
    previous_end_ns: int | None = None
    previous_terminal: list[tuple[int, int, int] | None] = [
        None for _ in bucket_metadata
    ]
    previous_activity: list[tuple[int, int] | None] = [None for _ in bucket_metadata]

    for position, value in enumerate(iteration_values):
        field = f"iterations[{position}]"
        iteration = _object(
            value,
            field,
            {
                "iteration",
                "start_ns",
                "end_ns",
                "plan_failure_record",
                "buckets",
            },
        )
        iteration_index = _count(iteration["iteration"], f"{field}.iteration")
        if iteration_index != position:
            raise DiagnosticError(
                "iteration indices must be unique and contiguous from zero"
            )
        start_ns = _count(iteration["start_ns"], f"{field}.start_ns")
        end_ns = _count(iteration["end_ns"], f"{field}.end_ns")
        if end_ns < start_ns:
            raise DiagnosticError(f"{field}.end_ns must not precede start_ns")
        if previous_end_ns is not None and start_ns < previous_end_ns:
            raise DiagnosticError("iteration timestamp intervals must not overlap")
        previous_end_ns = end_ns
        plan_failure_record = _integer(
            iteration["plan_failure_record"], f"{field}.plan_failure_record"
        )

        iteration_buckets = _array(iteration["buckets"], f"{field}.buckets")
        if len(iteration_buckets) != len(bucket_metadata):
            raise DiagnosticError(
                f"{field}.buckets must contain every metadata bucket exactly once"
            )
        for bucket_position, bucket_value in enumerate(iteration_buckets):
            bucket_field = f"{field}.buckets[{bucket_position}]"
            bucket = _object(
                bucket_value,
                bucket_field,
                {
                    "bucket_index",
                    "active_systems",
                    "active_channels",
                    "submitted_solver_slots",
                    "submitted_backtransform_slots",
                    "converged_systems",
                    "failed_systems",
                    "exhausted_systems",
                },
            )
            bucket_index = _count(
                bucket["bucket_index"], f"{bucket_field}.bucket_index"
            )
            if bucket_index != bucket_position:
                raise DiagnosticError(
                    f"{field}.buckets indices must match contiguous metadata indices"
                )

            metadata = bucket_metadata[bucket_position]
            system_capacity = metadata["system_capacity"]
            channel_capacity = metadata["channel_capacity"]
            active_systems = _count(
                bucket["active_systems"], f"{bucket_field}.active_systems"
            )
            active_channels = _count(
                bucket["active_channels"], f"{bucket_field}.active_channels"
            )
            if active_systems > system_capacity:
                raise DiagnosticError(
                    f"{bucket_field}.active_systems exceeds system_capacity"
                )
            if (
                not active_systems
                <= active_channels
                <= min(channel_capacity, 2 * active_systems)
            ):
                raise DiagnosticError(
                    f"{bucket_field}.active_channels must be between active_systems "
                    "and twice active_systems, within channel_capacity"
                )

            solver_slots = _slot_count(
                bucket["submitted_solver_slots"],
                f"{bucket_field}.submitted_solver_slots",
            )
            backtransform_slots = _slot_count(
                bucket["submitted_backtransform_slots"],
                f"{bucket_field}.submitted_backtransform_slots",
            )
            if plan_failure_record != 0 and (
                solver_slots is not None or backtransform_slots is not None
            ):
                raise DiagnosticError(
                    f"{bucket_field} submitted-slot counts must be null when "
                    "plan_failure_record is nonzero"
                )
            if execution_mode in FIXED_CAPACITY_MODES:
                for name, submitted in (
                    ("submitted_solver_slots", solver_slots),
                    ("submitted_backtransform_slots", backtransform_slots),
                ):
                    if submitted is not None and submitted != channel_capacity:
                        raise DiagnosticError(
                            f"{bucket_field}.{name} must equal channel_capacity "
                            f"in {execution_mode} mode"
                        )
            else:
                for name, submitted in (
                    ("submitted_solver_slots", solver_slots),
                    ("submitted_backtransform_slots", backtransform_slots),
                ):
                    if submitted is not None and submitted > active_channels:
                        raise DiagnosticError(
                            f"{bucket_field}.{name} cannot exceed active_channels "
                            "in device_dispatch_chain mode"
                        )

            converged = _count(
                bucket["converged_systems"], f"{bucket_field}.converged_systems"
            )
            failed = _count(bucket["failed_systems"], f"{bucket_field}.failed_systems")
            exhausted = _count(
                bucket["exhausted_systems"], f"{bucket_field}.exhausted_systems"
            )
            terminal = (converged, failed, exhausted)
            terminal_total = sum(terminal)
            if terminal_total > system_capacity:
                raise DiagnosticError(
                    f"{bucket_field} terminal counts exceed system_capacity"
                )
            if terminal_total < system_capacity - active_systems:
                raise DiagnosticError(
                    f"{bucket_field} terminal counts omit systems that were already "
                    "inactive before the numerical body"
                )

            previous = previous_terminal[bucket_position]
            previous_counts = previous_activity[bucket_position]
            if previous is not None:
                if any(
                    current < prior
                    for current, prior in zip(terminal, previous, strict=True)
                ):
                    raise DiagnosticError(
                        f"{bucket_field} cumulative terminal counts must not decrease"
                    )
                expected_active = system_capacity - sum(previous)
                if active_systems != expected_active:
                    raise DiagnosticError(
                        f"{bucket_field}.active_systems must equal the previous "
                        "iteration's remaining active systems"
                    )
                if terminal_total - sum(previous) > active_systems:
                    raise DiagnosticError(
                        f"{bucket_field} newly terminal systems exceed active_systems"
                    )
                if active_systems > previous_counts[0]:
                    raise DiagnosticError(
                        f"{bucket_field}.active_systems must not increase"
                    )
                if active_channels > previous_counts[1]:
                    raise DiagnosticError(
                        f"{bucket_field}.active_channels must not increase"
                    )
            previous_terminal[bucket_position] = terminal
            previous_activity[bucket_position] = (active_systems, active_channels)

    if batch_size == 0 and iteration_values:
        raise DiagnosticError("an empty batch must not contain numerical iterations")
    return call


def iter_jsonl(stream: TextIO) -> Iterator[dict[str, Any]]:
    """Yield validated call objects, reporting malformed input by line number."""
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            raise DiagnosticError(
                f"line {line_number}: blank lines are not valid JSONL"
            )
        try:
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            yield validate_call(value)
        except json.JSONDecodeError as exc:
            raise DiagnosticError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        except DiagnosticError as exc:
            raise DiagnosticError(f"line {line_number}: {exc}") from exc


def _summarize_call(call: dict[str, Any], call_index: int) -> dict[str, Any]:
    terminal_bucket_by_index = {
        item["bucket_index"]: item for item in call.get("terminal_buckets", [])
    }
    bucket_summaries: list[dict[str, Any]] = []
    for metadata in call["buckets"]:
        bucket_index = metadata["bucket_index"]
        activity: list[dict[str, Any]] = []
        for iteration in call["iterations"]:
            record = iteration["buckets"][bucket_index]
            active_channels = record["active_channels"]
            solver_slots = record["submitted_solver_slots"]
            backtransform_slots = record["submitted_backtransform_slots"]
            terminal_total = (
                record["converged_systems"]
                + record["failed_systems"]
                + record["exhausted_systems"]
            )
            activity.append(
                {
                    "iteration": iteration["iteration"],
                    "active_systems": record["active_systems"],
                    "active_channels": active_channels,
                    "submitted_solver_slots": solver_slots,
                    "submitted_solver_slack": (
                        None if solver_slots is None else active_channels - solver_slots
                    ),
                    "submitted_backtransform_slots": backtransform_slots,
                    "submitted_backtransform_slack": (
                        None
                        if backtransform_slots is None
                        else active_channels - backtransform_slots
                    ),
                    "converged_systems": record["converged_systems"],
                    "failed_systems": record["failed_systems"],
                    "exhausted_systems": record["exhausted_systems"],
                    "remaining_active_systems": (
                        metadata["system_capacity"] - terminal_total
                    ),
                }
            )
        bucket_summary = {**metadata, "activity": activity}
        if bucket_index in terminal_bucket_by_index:
            bucket_summary["terminal"] = {
                name: terminal_bucket_by_index[bucket_index][name]
                for name in TERMINAL_COUNT_FIELDS
            }
        bucket_summaries.append(bucket_summary)

    metadata = {
        field: call[field]
        for field in OPTIONAL_CALL_FIELDS.difference(
            {"terminal_buckets", "terminal_systems"}
        )
        if field in call
    }
    summary = {
        "call_index": call_index,
        "execution_mode": call["execution_mode"],
        "fallback_reason": call["fallback_reason"],
        "batch_size": call["batch_size"],
        "maximum_iterations": call["maximum_iterations"],
        **metadata,
        "iteration_count": len(call["iterations"]),
        "buckets": bucket_summaries,
    }
    if "terminal_buckets" in call:
        summary["terminal_totals"] = {
            name: sum(bucket[name] for bucket in call["terminal_buckets"])
            for name in TERMINAL_COUNT_FIELDS
        }
    elif "terminal_systems" in call:
        terminal_totals = dict.fromkeys(TERMINAL_COUNT_FIELDS, 0)
        for system in call["terminal_systems"]:
            category = _terminal_class(system["status"], system["converged"])
            terminal_totals[category] += 1
        summary["terminal_totals"] = terminal_totals
    if "terminal_systems" in call:
        summary["terminal_systems"] = [dict(item) for item in call["terminal_systems"]]
    return summary


def build_report(calls: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact count report without deriving performance claims."""
    validated_calls = [validate_call(call) for call in calls]
    return {
        "schema": REPORT_SCHEMA_NAME,
        "source_schema": SCHEMA_NAME,
        "call_count": len(validated_calls),
        "timing_policy": {
            "interval": "full_numerical_body",
            "stage_timing_status": "unknown",
            "instrumentation_perturbs_timing": True,
            "eligible_for_primary_performance_tables": False,
        },
        "calls": [
            _summarize_call(call, call_index)
            for call_index, call in enumerate(validated_calls, start=1)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the JSONL report command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate CUDA SCC diagnostic JSONL and emit a compact activity report. "
            "The report is diagnostic only and makes no performance claim."
        )
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        default="-",
        help="input JSONL path, or - for standard input (default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate input JSONL and write its compact diagnostic report."""
    args = build_parser().parse_args(argv)
    try:
        if args.jsonl == "-":
            report = build_report(iter_jsonl(sys.stdin))
        else:
            with Path(args.jsonl).open("r", encoding="utf-8") as stream:
                report = build_report(iter_jsonl(stream))
    except (DiagnosticError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 2

    json.dump(report, sys.stdout, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
