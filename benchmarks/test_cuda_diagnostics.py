"""Tests for the CUDA SCC diagnostic JSONL schema and report CLI."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmarks import cuda_diagnostics as diagnostics

SCHEMA_PATH = Path(__file__).with_name("cuda-scc-diagnostics.schema.json")


def schema_errors(
    schema: dict[str, object],
    value: object,
    *,
    root: dict[str, object] | None = None,
    path: str = "$",
) -> list[str]:
    """Apply the schema keywords used by this checked-in artifact."""
    if root is None:
        root = schema
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return [f"{path}: unsupported schema reference {reference!r}"]
        target: object = root
        for component in reference[2:].split("/"):
            if not isinstance(target, dict) or component not in target:
                return [f"{path}: unresolved schema reference {reference!r}"]
            target = target[component]
        if not isinstance(target, dict):
            return [f"{path}: schema reference is not an object"]
        return schema_errors(target, value, root=root, path=path)

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        type_matches = {
            "object": type(value) is dict,
            "array": type(value) is list,
            "string": type(value) is str,
            "integer": type(value) is int,
            "number": type(value) in (int, float),
            "boolean": type(value) is bool,
            "null": value is None,
        }
        if not any(type_matches.get(item, False) for item in types):
            return [f"{path}: does not match type {expected_type!r}"]

    if "const" in schema and (
        type(value) is not type(schema["const"]) or value != schema["const"]
    ):
        errors.append(f"{path}: does not match const")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: is outside enum")
    if (
        "minimum" in schema
        and type(value) in (int, float)
        and value < schema["minimum"]
    ):
        errors.append(f"{path}: is below minimum")

    if type(value) is dict:
        required = schema.get("required", [])
        errors.extend(
            f"{path}.{field}: is required" for field in required if field not in value
        )
        properties = schema.get("properties", {})
        for field, field_value in value.items():
            if field in properties:
                errors.extend(
                    schema_errors(
                        properties[field],
                        field_value,
                        root=root,
                        path=f"{path}.{field}",
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{field}: is not allowed")
    elif type(value) is list and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(
                schema_errors(schema["items"], item, root=root, path=f"{path}[{index}]")
            )
    return errors


def make_metadata(
    bucket_index: int = 0,
    *,
    ao: int = 2,
    system_capacity: int = 2,
    channel_capacity: int = 3,
) -> dict[str, int]:
    """Build exact-AO bucket metadata for a test call."""
    return {
        "bucket_index": bucket_index,
        "ao": ao,
        "system_capacity": system_capacity,
        "channel_capacity": channel_capacity,
    }


def make_terminal_bucket(
    bucket_index: int = 0,
    *,
    converged_systems: int = 0,
    failed_systems: int = 0,
    exhausted_systems: int = 0,
    unfinished_systems: int = 0,
) -> dict[str, int]:
    """Build final disjoint outcome counts for one metadata bucket."""
    return {
        "bucket_index": bucket_index,
        "converged_systems": converged_systems,
        "failed_systems": failed_systems,
        "exhausted_systems": exhausted_systems,
        "unfinished_systems": unfinished_systems,
    }


def make_terminal_system(
    system_index: int,
    *,
    iterations: int = 0,
    status: int = 0,
    converged: int = 0,
) -> dict[str, int]:
    """Build one final per-system SCC snapshot."""
    return {
        "system_index": system_index,
        "iterations": iterations,
        "status": status,
        "converged": converged,
    }


def make_bucket(
    bucket_index: int = 0,
    *,
    active_systems: int = 2,
    active_channels: int = 3,
    submitted_solver_slots: int | None = 3,
    submitted_backtransform_slots: int | None = 3,
    converged_systems: int = 0,
    failed_systems: int = 0,
    exhausted_systems: int = 0,
) -> dict[str, int | None]:
    """Build one bucket's activity, submissions, and terminal counts."""
    return {
        "bucket_index": bucket_index,
        "active_systems": active_systems,
        "active_channels": active_channels,
        "submitted_solver_slots": submitted_solver_slots,
        "submitted_backtransform_slots": submitted_backtransform_slots,
        "converged_systems": converged_systems,
        "failed_systems": failed_systems,
        "exhausted_systems": exhausted_systems,
    }


def make_iteration(
    iteration: int,
    buckets: list[dict[str, int | None]],
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
    plan_failure_record: int = 0,
) -> dict[str, object]:
    """Build one numerical-body interval with its bucket records."""
    if start_ns is None:
        start_ns = iteration * 10
    if end_ns is None:
        end_ns = start_ns + 5
    return {
        "iteration": iteration,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "plan_failure_record": plan_failure_record,
        "buckets": buckets,
    }


def make_call(
    *,
    batch_size: int = 2,
    buckets: list[dict[str, int]] | None = None,
    iterations: list[dict[str, object]] | None = None,
    execution_mode: str = "device_dispatch_chain",
    fallback_reason: int = 0,
    maximum_iterations: int = 4,
    terminal_buckets: list[dict[str, int]] | None = None,
    terminal_systems: list[dict[str, int]] | None = None,
) -> dict[str, object]:
    """Build a top-level v1 call with schema-valid test defaults."""
    if buckets is None:
        buckets = [make_metadata()]
    if iterations is None:
        iterations = []
    call = {
        "schema": diagnostics.SCHEMA_NAME,
        "instrumented": True,
        "execution_mode": execution_mode,
        "fallback_reason": fallback_reason,
        "batch_size": batch_size,
        "maximum_iterations": maximum_iterations,
        "diagnostic_device_bytes": 4096,
        "buckets": buckets,
        "iterations": iterations,
    }
    if terminal_buckets is not None:
        call["terminal_buckets"] = terminal_buckets
    if terminal_systems is not None:
        call["terminal_systems"] = terminal_systems
    return call


class CudaDiagnosticsTests(unittest.TestCase):
    """Exercise schema invariants, JSONL handling, and compact reporting."""

    def test_public_capture_cli_imports_from_repository_root(self) -> None:
        """Import the registered CTest entry point without inherited tool paths."""
        process = subprocess.run(
            [sys.executable, "-m", "benchmarks.check_cuda_diagnostics", "--help"],
            cwd=SCHEMA_PATH.parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--memory-mode", process.stdout)

    def test_empty_batch_and_empty_iterations_are_valid(self) -> None:
        """Allow an empty request and a nonempty request without body records."""
        empty_batch = make_call(batch_size=0, buckets=[], iterations=[])
        self.assertIs(diagnostics.validate_call(empty_batch), empty_batch)

        no_numerical_body = make_call(
            buckets=[make_metadata(system_capacity=2, channel_capacity=3)]
        )
        self.assertIs(diagnostics.validate_call(no_numerical_body), no_numerical_body)

    def test_terminal_snapshot_reports_initial_failures_without_body_rows(self) -> None:
        """Use post-call state to report terminal systems when graphs skip SCC."""
        call = make_call(
            execution_mode="device_tail_graph",
            iterations=[],
            terminal_buckets=[make_terminal_bucket(failed_systems=2)],
            terminal_systems=[
                make_terminal_system(0, status=8),
                make_terminal_system(1, status=8),
            ],
        )
        diagnostics.validate_call(call)
        summarized = diagnostics.build_report([call])["calls"][0]
        self.assertEqual(summarized["iteration_count"], 0)
        self.assertEqual(summarized["buckets"][0]["activity"], [])
        self.assertEqual(
            summarized["buckets"][0]["terminal"],
            {
                "converged_systems": 0,
                "failed_systems": 2,
                "exhausted_systems": 0,
                "unfinished_systems": 0,
            },
        )
        self.assertEqual(
            summarized["terminal_totals"],
            {
                "converged_systems": 0,
                "failed_systems": 2,
                "exhausted_systems": 0,
                "unfinished_systems": 0,
            },
        )
        self.assertEqual(len(summarized["terminal_systems"]), 2)

    def test_terminal_snapshot_checks_disjoint_counts_and_system_mapping(self) -> None:
        """Cross-check grouped outcomes against every ordinal system result."""
        valid_call = make_call(
            batch_size=4,
            buckets=[make_metadata(system_capacity=4, channel_capacity=6)],
            terminal_buckets=[
                make_terminal_bucket(
                    converged_systems=1,
                    failed_systems=1,
                    exhausted_systems=1,
                    unfinished_systems=1,
                )
            ],
            terminal_systems=[
                make_terminal_system(0, status=0, converged=1),
                make_terminal_system(1, status=8),
                make_terminal_system(2, status=7),
                make_terminal_system(3),
            ],
        )
        summarized = diagnostics.build_report([valid_call])["calls"][0]
        self.assertEqual(
            summarized["terminal_totals"],
            {
                "converged_systems": 1,
                "failed_systems": 1,
                "exhausted_systems": 1,
                "unfinished_systems": 1,
            },
        )

        invalid_calls = []
        bad_capacity = json.loads(json.dumps(valid_call))
        bad_capacity["terminal_buckets"][0]["unfinished_systems"] = 0
        invalid_calls.append((bad_capacity, "terminal classes must sum"))

        bad_index = json.loads(json.dumps(valid_call))
        bad_index["terminal_systems"][1]["system_index"] = 0
        invalid_calls.append((bad_index, "terminal system indices"))

        bad_totals = json.loads(json.dumps(valid_call))
        bad_totals["terminal_buckets"][0]["failed_systems"] = 2
        bad_totals["terminal_buckets"][0]["unfinished_systems"] = 0
        invalid_calls.append(
            (bad_totals, "must match terminal_systems classifications")
        )

        bad_convergence = json.loads(json.dumps(valid_call))
        bad_convergence["terminal_systems"][1]["converged"] = True
        invalid_calls.append((bad_convergence, "converged must be an integer"))

        bad_status = json.loads(json.dumps(valid_call))
        bad_status["terminal_systems"][2]["converged"] = 1
        invalid_calls.append((bad_status, "unless status is success"))

        for call, message in invalid_calls:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(diagnostics.DiagnosticError, message),
            ):
                diagnostics.validate_call(call)

    def test_terminal_snapshot_arrays_are_independently_optional(self) -> None:
        """Summarize either final snapshot form while preserving legacy records."""
        bucket_only = make_call(
            terminal_buckets=[make_terminal_bucket(unfinished_systems=2)]
        )
        system_only = make_call(
            terminal_systems=[make_terminal_system(0), make_terminal_system(1)]
        )
        for call in (bucket_only, system_only):
            with self.subTest(fields=tuple(sorted(call))):
                summary = diagnostics.build_report([call])["calls"][0]
                self.assertEqual(summary["terminal_totals"]["unfinished_systems"], 2)

    def test_graph_modes_may_skip_body_but_bounded_mode_keeps_all_bodies(self) -> None:
        """Distinguish graph root exits from bounded provider body submission."""
        for execution_mode in ("device_dispatch_chain", "device_tail_graph"):
            call = make_call(execution_mode=execution_mode, iterations=[])
            self.assertIs(diagnostics.validate_call(call), call)

        bounded = make_call(execution_mode="bounded_fallback", iterations=[])
        with self.assertRaisesRegex(
            diagnostics.DiagnosticError,
            "bounded_fallback must record every maximum_iterations body",
        ):
            diagnostics.validate_call(bounded)

    def test_all_active_bucket_counts_and_chain_submissions(self) -> None:
        """Report active counts and chain submission slack independently."""
        call = make_call(
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=3,
                        )
                    ],
                )
            ]
        )
        report = diagnostics.build_report([call])
        self.assertEqual(report["calls"][0]["diagnostic_device_bytes"], 4096)
        bucket = report["calls"][0]["buckets"][0]
        activity = bucket["activity"][0]
        self.assertEqual(activity["active_systems"], 2)
        self.assertEqual(activity["active_channels"], 3)
        self.assertEqual(activity["submitted_solver_slack"], 1)
        self.assertEqual(activity["submitted_backtransform_slack"], 0)
        self.assertEqual(activity["remaining_active_systems"], 2)

    def test_partial_and_all_convergence_preserve_channel_drops(self) -> None:
        """Track cumulative convergence while mixed-spin channels drop."""
        partial = make_call(
            batch_size=4,
            buckets=[make_metadata(system_capacity=4, channel_capacity=7)],
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            active_systems=4,
                            active_channels=7,
                            submitted_solver_slots=7,
                            submitted_backtransform_slots=7,
                            converged_systems=2,
                        )
                    ],
                )
            ],
        )
        partial_activity = diagnostics.build_report([partial])["calls"][0]["buckets"][
            0
        ]["activity"][0]
        self.assertEqual(partial_activity["remaining_active_systems"], 2)

        all_converged = make_call(
            batch_size=2,
            buckets=[make_metadata(system_capacity=2, channel_capacity=3)],
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            submitted_solver_slots=3,
                            submitted_backtransform_slots=3,
                            converged_systems=1,
                        )
                    ],
                ),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            active_systems=1,
                            active_channels=1,
                            submitted_solver_slots=1,
                            submitted_backtransform_slots=1,
                            converged_systems=2,
                        )
                    ],
                    start_ns=5,
                    end_ns=9,
                ),
            ],
        )
        all_activity = diagnostics.build_report([all_converged])["calls"][0]["buckets"][
            0
        ]["activity"]
        self.assertEqual(all_activity[-1]["remaining_active_systems"], 0)
        self.assertEqual(all_activity[1]["active_channels"], 1)

    def test_peer_failure_does_not_hide_other_bucket_and_fixed_slots(self) -> None:
        """Retain peer progress and fixed-capacity slots for inactive buckets."""
        call = make_call(
            batch_size=2,
            execution_mode="device_tail_graph",
            buckets=[
                make_metadata(0, ao=2, system_capacity=1, channel_capacity=2),
                make_metadata(1, ao=6, system_capacity=1, channel_capacity=2),
            ],
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            0,
                            active_systems=1,
                            active_channels=2,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                            failed_systems=1,
                        ),
                        make_bucket(
                            1,
                            active_systems=1,
                            active_channels=2,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                        ),
                    ],
                ),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            0,
                            active_systems=0,
                            active_channels=0,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                            failed_systems=1,
                        ),
                        make_bucket(
                            1,
                            active_systems=1,
                            active_channels=1,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                            converged_systems=1,
                        ),
                    ],
                    start_ns=5,
                    end_ns=9,
                ),
            ],
        )
        report = diagnostics.build_report([call])["calls"][0]
        failed_bucket = report["buckets"][0]["activity"]
        peer_bucket = report["buckets"][1]["activity"]
        self.assertEqual(failed_bucket[1]["active_systems"], 0)
        self.assertEqual(failed_bucket[1]["submitted_solver_slots"], 2)
        self.assertEqual(failed_bucket[1]["submitted_solver_slack"], -2)
        self.assertEqual(peer_bucket[1]["converged_systems"], 1)

    def test_mixed_spin_channel_counts_are_independent_of_system_count(self) -> None:
        """Allow channel reductions to differ from system reductions."""
        call = make_call(
            batch_size=3,
            buckets=[make_metadata(system_capacity=3, channel_capacity=5)],
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            active_systems=3,
                            active_channels=5,
                            submitted_solver_slots=5,
                            submitted_backtransform_slots=5,
                            converged_systems=1,
                        )
                    ],
                ),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            active_systems=2,
                            active_channels=2,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                            converged_systems=3,
                        )
                    ],
                    start_ns=5,
                    end_ns=8,
                ),
            ],
        )
        activity = diagnostics.build_report([call])["calls"][0]["buckets"][0][
            "activity"
        ]
        self.assertEqual(activity[1]["active_systems"], 2)
        self.assertEqual(activity[1]["active_channels"], 2)

    def test_unknown_codes_and_unavailable_counts_remain_explicit(self) -> None:
        """Preserve unknown integer codes and unavailable submission counts."""
        call = make_call(
            fallback_reason=987654,
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            submitted_solver_slots=None,
                            submitted_backtransform_slots=None,
                        )
                    ],
                    plan_failure_record=123456,
                )
            ],
        )
        report = diagnostics.build_report([call])
        activity = report["calls"][0]["buckets"][0]["activity"][0]
        self.assertIsNone(activity["submitted_solver_slots"])
        self.assertIsNone(activity["submitted_solver_slack"])
        self.assertIsNone(activity["submitted_backtransform_slots"])
        self.assertEqual(report["timing_policy"]["stage_timing_status"], "unknown")
        self.assertFalse(
            report["timing_policy"]["eligible_for_primary_performance_tables"]
        )

    def test_optional_metadata_round_trips_without_selection_inference(self) -> None:
        """Preserve preference metadata separately from actual mode and fallback."""
        call = make_call(execution_mode="device_tail_graph", fallback_reason=9)
        call.update(
            plan_token=123,
            wavefunction_layout_fingerprint=456,
            start_policy=2,
            graph_preference=2,
            chain_measured_ao_bound=40,
            diagnostic_device_bytes=8192,
            chain_selection_constraints=["mixed_spin", "unsupported_layout"],
        )
        summarized = diagnostics.build_report([call])["calls"][0]
        self.assertEqual(summarized["execution_mode"], "device_tail_graph")
        self.assertEqual(summarized["fallback_reason"], 9)
        self.assertEqual(summarized["graph_preference"], 2)
        self.assertEqual(summarized["chain_measured_ao_bound"], 40)
        self.assertEqual(
            summarized["chain_selection_constraints"],
            ["mixed_spin", "unsupported_layout"],
        )

    def test_all_optional_metadata_fields_may_be_absent(self) -> None:
        """Accept older or white-box records without optional metadata."""
        call = make_call()
        for field in (
            "plan_token",
            "wavefunction_layout_fingerprint",
            "start_policy",
            "graph_preference",
            "chain_measured_ao_bound",
            "diagnostic_device_bytes",
            "chain_selection_constraints",
            "terminal_buckets",
            "terminal_systems",
        ):
            call.pop(field, None)
        diagnostics.validate_call(call)
        summarized = diagnostics.build_report([call])["calls"][0]
        self.assertNotIn("diagnostic_device_bytes", summarized)

    def test_optional_metadata_types_and_enum_values_are_strict(self) -> None:
        """Reject booleans, invalid policy codes, and non-string constraints."""
        invalid_calls = (
            ("plan_token", True, "plan_token must be an integer"),
            ("start_policy", 3, "start_policy must be one of"),
            ("graph_preference", 3, "graph_preference must be one of"),
            (
                "chain_selection_constraints",
                ["valid", 1],
                "chain_selection_constraints\\[1\\] must be a string",
            ),
        )
        for field, value, message in invalid_calls:
            with self.subTest(field=field):
                call = make_call()
                call[field] = value
                with self.assertRaisesRegex(diagnostics.DiagnosticError, message):
                    diagnostics.validate_call(call)

    def test_plan_failure_requires_null_submission_counts(self) -> None:
        """Reject stale slot counts when a plan failure interrupts telemetry."""
        call = make_call(
            iterations=[
                make_iteration(
                    0,
                    [make_bucket()],
                    plan_failure_record=17,
                )
            ]
        )
        with self.assertRaisesRegex(
            diagnostics.DiagnosticError, "must be null when plan_failure_record"
        ):
            diagnostics.validate_call(call)

    def test_bounded_fallback_retains_full_slots_after_all_systems_terminate(
        self,
    ) -> None:
        """Keep every bounded body, including full slots at zero activity."""
        call = make_call(
            execution_mode="bounded_fallback",
            fallback_reason=15,
            maximum_iterations=3,
            iterations=[
                make_iteration(
                    0,
                    [
                        make_bucket(
                            converged_systems=1,
                            failed_systems=1,
                        )
                    ],
                ),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            active_systems=0,
                            active_channels=0,
                            converged_systems=1,
                            failed_systems=1,
                            submitted_solver_slots=3,
                            submitted_backtransform_slots=3,
                        )
                    ],
                    start_ns=5,
                    end_ns=9,
                ),
                make_iteration(
                    2,
                    [
                        make_bucket(
                            active_systems=0,
                            active_channels=0,
                            converged_systems=1,
                            failed_systems=1,
                            submitted_solver_slots=3,
                            submitted_backtransform_slots=3,
                        )
                    ],
                    start_ns=10,
                    end_ns=14,
                ),
            ],
        )
        self.assertIs(diagnostics.validate_call(call), call)
        summarized = diagnostics.build_report([call])["calls"][0]
        activity = summarized["buckets"][0]["activity"]
        self.assertEqual(summarized["iteration_count"], 3)
        self.assertEqual(summarized["fallback_reason"], 15)
        self.assertEqual(len(activity), 3)
        self.assertEqual([row["active_systems"] for row in activity], [2, 0, 0])
        self.assertEqual([row["submitted_solver_slots"] for row in activity], [3, 3, 3])
        self.assertEqual(
            [row["submitted_solver_slack"] for row in activity], [0, -3, -3]
        )

    def test_repeated_calls_are_independent_jsonl_records(self) -> None:
        """Validate repeated call records without sharing per-call state."""
        call = make_call(
            iterations=[make_iteration(0, [make_bucket()])],
        )
        payload = f"{json.dumps(call)}\n{json.dumps(call)}\n"
        calls = list(diagnostics.iter_jsonl(io.StringIO(payload)))
        report = diagnostics.build_report(calls)
        self.assertEqual(report["call_count"], 2)
        self.assertEqual([item["call_index"] for item in report["calls"]], [1, 2])

    def test_schema_artifact_structure_and_execution_modes_match_parser(self) -> None:
        """Keep the published schema shape and actual-mode enum in sync."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        properties = schema["properties"]
        self.assertEqual(
            set(properties),
            {
                "schema",
                "instrumented",
                "execution_mode",
                "fallback_reason",
                "batch_size",
                "maximum_iterations",
                "plan_token",
                "wavefunction_layout_fingerprint",
                "start_policy",
                "graph_preference",
                "chain_measured_ao_bound",
                "diagnostic_device_bytes",
                "chain_selection_constraints",
                "terminal_buckets",
                "terminal_systems",
                "buckets",
                "iterations",
            },
        )
        execution_modes = properties["execution_mode"]["enum"]
        self.assertEqual(len(execution_modes), 3)
        self.assertEqual(set(execution_modes), diagnostics.EXECUTION_MODES)
        self.assertEqual(
            set(schema["required"]),
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
        )
        for definition in schema["$defs"].values():
            self.assertEqual(definition["type"], "object")
            self.assertIs(definition["additionalProperties"], False)

    def test_schema_accepts_emitted_call_contracts_and_rejects_bad_types(self) -> None:
        """Validate producer-shaped JSON with the checked-in schema, stdlib only."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        calls = [
            make_call(batch_size=0, buckets=[], iterations=[]),
            make_call(
                execution_mode="device_dispatch_chain",
                iterations=[make_iteration(0, [make_bucket()])],
            ),
            make_call(
                execution_mode="device_tail_graph",
                iterations=[make_iteration(0, [make_bucket()])],
            ),
            make_call(
                execution_mode="bounded_fallback",
                maximum_iterations=1,
                iterations=[make_iteration(0, [make_bucket()])],
            ),
            make_call(
                terminal_buckets=[make_terminal_bucket(unfinished_systems=2)],
                terminal_systems=[
                    make_terminal_system(0),
                    make_terminal_system(1),
                ],
            ),
            make_call(
                iterations=[
                    make_iteration(
                        0,
                        [
                            make_bucket(
                                submitted_solver_slots=None,
                                submitted_backtransform_slots=None,
                            )
                        ],
                        plan_failure_record=17,
                    )
                ]
            ),
        ]
        calls[-1].update(
            plan_token=1,
            wavefunction_layout_fingerprint=2,
            start_policy=0,
            graph_preference=0,
            chain_measured_ao_bound=40,
            chain_selection_constraints=["periodic_request"],
        )
        for call in calls:
            with self.subTest(execution_mode=call["execution_mode"]):
                emitted_json = json.loads(json.dumps(call))
                self.assertEqual(schema_errors(schema, emitted_json), [])
                diagnostics.validate_call(emitted_json)

        invalid = dict(calls[1], batch_size=True)
        self.assertTrue(schema_errors(schema, invalid))
        invalid = dict(calls[1], instrumented=1)
        self.assertTrue(schema_errors(schema, invalid))
        invalid = dict(calls[1], unexpected_metadata=1)
        self.assertTrue(schema_errors(schema, invalid))
        invalid_terminal = json.loads(json.dumps(calls[4]))
        invalid_terminal["terminal_systems"][0]["converged"] = True
        self.assertTrue(schema_errors(schema, invalid_terminal))

    def test_cli_reads_jsonl_and_emits_compact_json_report(self) -> None:
        """Read a JSONL file and emit a compact machine-readable report."""
        call = make_call(batch_size=0, buckets=[], iterations=[])
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "diagnostics.jsonl"
            input_path.write_text(json.dumps(call) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = diagnostics.main([str(input_path)])
        self.assertEqual(result, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["call_count"], 1)
        self.assertEqual(report["calls"][0]["buckets"], [])

    def test_cli_reports_invalid_jsonl_without_a_report(self) -> None:
        """Return an error status without writing a partial report."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("sys.stdin", io.StringIO("\n")),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = diagnostics.main(["-"])
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("line 1", stderr.getvalue())

    def test_rejects_invalid_counts_capacities_and_types(self) -> None:
        """Reject booleans, unknown fields, and inconsistent capacities."""
        mutations = (
            (
                lambda call: call.update(batch_size=True),
                "batch_size must be an integer",
            ),
            (
                lambda call: call.update(diagnostic_device_bytes=True),
                "diagnostic_device_bytes must be an integer",
            ),
            (lambda call: call.update(extra=True), "unknown fields"),
            (
                lambda call: call["buckets"][0].update(channel_capacity=5),
                "channel_capacity must be between",
            ),
            (
                lambda call: call["buckets"][0].update(bucket_index=True),
                "bucket_index must be an integer",
            ),
            (
                lambda call: call.update(batch_size=3),
                "capacities must sum to batch_size",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                call = make_call()
                mutate(call)
                with self.assertRaisesRegex(diagnostics.DiagnosticError, message):
                    diagnostics.validate_call(call)

    def test_rejects_duplicate_or_gapped_bucket_and_iteration_indices(self) -> None:
        """Require unique contiguous zero-based bucket and iteration indices."""
        bucket_cases = (
            [make_metadata(1)],
            [
                make_metadata(0, system_capacity=1, channel_capacity=1),
                make_metadata(0, ao=3, system_capacity=1, channel_capacity=1),
            ],
        )
        for buckets in bucket_cases:
            with self.subTest(buckets=buckets):
                call = make_call(
                    batch_size=sum(item["system_capacity"] for item in buckets),
                    buckets=buckets,
                )
                with self.assertRaisesRegex(
                    diagnostics.DiagnosticError, "bucket metadata indices"
                ):
                    diagnostics.validate_call(call)

        for indices in ([0, 2], [0, 0]):
            with self.subTest(indices=indices):
                call = make_call(
                    iterations=[
                        make_iteration(index, [make_bucket()]) for index in indices
                    ]
                )
                with self.assertRaisesRegex(
                    diagnostics.DiagnosticError, "iteration indices"
                ):
                    diagnostics.validate_call(call)

        call = make_call(iterations=[make_iteration(0, [make_bucket(1)])])
        with self.assertRaisesRegex(diagnostics.DiagnosticError, "indices must match"):
            diagnostics.validate_call(call)

    def test_rejects_timestamp_and_activity_inconsistencies(self) -> None:
        """Reject reversed or overlapping times and impossible channel bounds."""
        invalid_calls = (
            make_call(
                iterations=[make_iteration(0, [make_bucket()], start_ns=5, end_ns=4)]
            ),
            make_call(
                iterations=[
                    make_iteration(0, [make_bucket()], start_ns=0, end_ns=5),
                    make_iteration(1, [make_bucket()], start_ns=4, end_ns=8),
                ]
            ),
            make_call(
                iterations=[
                    make_iteration(
                        0,
                        [make_bucket(active_systems=1, active_channels=3)],
                    )
                ]
            ),
        )
        expected_errors = (
            "must not precede start_ns",
            "must not overlap",
            "active_channels must be between",
        )
        for call, message in zip(invalid_calls, expected_errors, strict=True):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(diagnostics.DiagnosticError, message),
            ):
                diagnostics.validate_call(call)

    def test_rejects_mode_specific_submission_bounds(self) -> None:
        """Enforce active-channel chain limits and fixed-capacity mode counts."""
        chain = make_call(
            iterations=[
                make_iteration(
                    0,
                    [make_bucket(active_channels=2, submitted_solver_slots=3)],
                )
            ]
        )
        with self.assertRaisesRegex(
            diagnostics.DiagnosticError, "cannot exceed active_channels"
        ):
            diagnostics.validate_call(chain)

        fixed = make_call(
            execution_mode="device_tail_graph",
            iterations=[
                make_iteration(
                    0,
                    [make_bucket(submitted_backtransform_slots=2)],
                )
            ],
        )
        with self.assertRaisesRegex(
            diagnostics.DiagnosticError, "must equal channel_capacity"
        ):
            diagnostics.validate_call(fixed)

    def test_rejects_terminal_count_overlap_or_inconsistent_next_activity(self) -> None:
        """Reject overlapping terminal classes and mismatched next activity."""
        too_many_terminal = make_call(
            iterations=[
                make_iteration(
                    0,
                    [make_bucket(converged_systems=2, failed_systems=1)],
                )
            ]
        )
        decreasing_terminal = make_call(
            iterations=[
                make_iteration(0, [make_bucket(converged_systems=1)]),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            active_systems=1,
                            active_channels=1,
                            submitted_solver_slots=1,
                            submitted_backtransform_slots=1,
                            failed_systems=1,
                            converged_systems=0,
                        )
                    ],
                    start_ns=5,
                    end_ns=8,
                ),
            ]
        )
        wrong_next_active = make_call(
            iterations=[
                make_iteration(0, [make_bucket(converged_systems=1)]),
                make_iteration(
                    1,
                    [
                        make_bucket(
                            active_systems=2,
                            active_channels=2,
                            submitted_solver_slots=2,
                            submitted_backtransform_slots=2,
                            converged_systems=1,
                        )
                    ],
                    start_ns=5,
                    end_ns=8,
                ),
            ]
        )
        for call, message in (
            (too_many_terminal, "terminal counts exceed system_capacity"),
            (decreasing_terminal, "cumulative terminal counts must not decrease"),
            (wrong_next_active, "must equal the previous iteration's remaining"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(diagnostics.DiagnosticError, message),
            ):
                diagnostics.validate_call(call)

    def test_rejects_duplicate_json_keys_nonstandard_constants_and_blank_lines(
        self,
    ) -> None:
        """Reject duplicate keys, non-standard constants, and blank lines."""
        invalid_records = (
            '{"schema":"xtbloom.cuda.scc_diagnostics.v1","schema":"other"}',
            '{"schema": NaN}',
            "\n",
        )
        for record in invalid_records:
            with (
                self.subTest(record=record),
                self.assertRaises(diagnostics.DiagnosticError),
            ):
                list(diagnostics.iter_jsonl(io.StringIO(record)))


if __name__ == "__main__":
    unittest.main()
