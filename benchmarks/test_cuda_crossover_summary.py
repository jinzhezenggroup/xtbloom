"""Hardware-free integrity and reproducibility tests for compact component tables."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks import cuda_crossover
from benchmarks import cuda_crossover_summary as summary_tool


class CudaCrossoverSummaryTests(unittest.TestCase):
    """Use complete fake native ledgers, not real GPU timing or oracle evidence."""

    def setUp(self) -> None:
        """Prepare a temporary raw fixture and deliberately dirty analyzer identity."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.raw_path = self.directory / "raw.jsonl"
        self.coordinate = cuda_crossover.Coordinate(40, 1, 1)
        self.analyzer = {
            "source_revision": "e" * 40,
            "working_tree_dirty": True,
            "source_file_sha256": {"benchmarks/cuda_crossover_summary.py": "f" * 64},
        }

    def _metadata(self, count: int = 1) -> dict[str, Any]:
        return {
            "record_type": "matrix_metadata",
            "schema_version": 1,
            "benchmark": "restricted_scc_component_crossover",
            "qualification": "component_only",
            "coordinates_requested": count,
            "warmups": 1,
            "paired_samples": 3,
            "seed": 515,
            "timing_protocol": "CUDA events exclude FRESH upload and state download",
            "native_command_template": (
                "native --benchmark-crossover-one AO B DENOM W S SEED"
            ),
            "source_identity": {
                "source_revision": "d" * 40,
                "working_tree_dirty": False,
                "binary_sha256": "b" * 64,
                "cmake_cache_sha256": "c" * 64,
                "source_file_sha256": dict.fromkeys(
                    summary_tool.FIXTURE_SOURCES, "a" * 64
                ),
            },
            "gpu_inventory": {"native_selection_association": "UNVERIFIED"},
        }

    def _result(
        self, coordinate: cuda_crossover.Coordinate | None = None
    ) -> dict[str, Any]:
        coordinate = coordinate or self.coordinate
        samples = []
        for index, (chain_ms, tail_ms) in enumerate(((100, 1), (2, 4), (4, 4), (6, 3))):
            samples.append(
                {
                    "record_type": "sample",
                    "sample_kind": "warmup" if index == 0 else "measurement",
                    "pair_index": 0 if index == 0 else index - 1,
                    "execution_order": "chain_then_tail"
                    if index % 2 == 0
                    else "tail_then_chain",
                    "chain_ms": chain_ms,
                    "tail_ms": tail_ms,
                    "state_parity": True,
                    "maximum_energy_delta_hartree": 0.0,
                    "maximum_charge_delta_e": 0.0,
                    "maximum_state_delta": 0.0,
                    **{
                        f"{mode}_{suffix}": [value] * coordinate.batch_size
                        for mode in ("chain", "tail")
                        for suffix, value in (
                            ("iterations", 1),
                            ("statuses", 7),
                            ("converged", 0),
                        )
                    },
                }
            )
        mask = cuda_crossover.deterministic_active_mask(
            coordinate.batch_size, coordinate.active_denominator, 515
        )
        native_summary = {
            "record_type": "coordinate_summary",
            "coordinate_status": "pass",
            "qualification": "component_only",
            "fixture": coordinate.fixture,
            "ao_count": coordinate.ao_count,
            "batch_size": coordinate.batch_size,
            "active_denominator": coordinate.active_denominator,
            "active_mask": mask,
            "active_system_count": sum(mask),
            "effective_active_fraction": sum(mask) / coordinate.batch_size,
            "maximum_iterations": 1,
            "warmup_pairs": 1,
            "measured_pairs": 3,
            "execution_counts": {"chain_calls": 4, "tail_calls": 4, "total_calls": 8},
            "graph_executable_counts": {
                "chain_dispatch_family": 5,
                "chain_total": 6,
                "tail_total": 2,
            },
            "setup_ms": 12.0,
            "fixture_arena_bytes": 100,
            "fresh_checkpoint_bytes": 20,
            "fixture_device_bytes": 120,
            "chain_known_control_table_bytes": 72,
            "tail_known_control_table_bytes": 32,
            "retained_device_bytes": 224,
            "retained_device_bytes_scope": cuda_crossover.RETAINED_DEVICE_BYTES_SCOPE,
            "state_parity": True,
        }
        result = {
            "record_type": "coordinate_result",
            "coordinate": {
                "ao_count": coordinate.ao_count,
                "batch_size": coordinate.batch_size,
                "active_denominator": coordinate.active_denominator,
            },
            "fixture": coordinate.fixture,
            "status": "pass",
            "reason": None,
            "samples": samples,
            "summary": native_summary,
            "subprocess_return_code": 0,
            "timed_out": False,
            "parse_diagnostic": None,
        }
        self._replay_copies(result)
        return result

    def _replay_copies(self, result: dict[str, Any]) -> None:
        records = [*result["samples"]]
        if result["summary"] is not None:
            records.append(result["summary"])
        result["native_records"] = copy.deepcopy(records)
        result["stdout"] = "\n".join(
            summary_tool.canonical_json(record) for record in records
        )

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.raw_path.write_text(
            "\n".join(summary_tool.canonical_json(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def _summarize(
        self, coordinates: list[cuda_crossover.Coordinate] | None = None
    ) -> dict[str, Any]:
        return summary_tool.summarize(
            self.raw_path, coordinates or [self.coordinate], analyzer=self.analyzer
        )

    def test_measurement_quantiles_do_not_merge_warmups(self) -> None:
        """The 100x warmup ratio cannot contaminate the three measured pairs."""
        self._write([self._metadata(), self._result()])
        document = self._summarize()
        row = document["coordinates"][0]
        self.assertEqual(row["chain_ms_median"], 4)
        self.assertAlmostEqual(row["chain_ms_p95"], 5.8)
        self.assertEqual(row["tail_ms_median"], 4)
        self.assertEqual(row["pair_ratio_median"], 1)
        self.assertAlmostEqual(row["pair_ratio_p05"], 0.55)
        self.assertAlmostEqual(row["pair_ratio_p95"], 1.9)
        self.assertEqual(row["pair_ratio_defined_count"], 3)
        self.assertEqual(row["warmup_pairs_observed"], 1)
        self.assertEqual(row["measurement_pairs_observed"], 3)
        self.assertEqual(row["warmup_parity_pass_pairs"], 1)
        self.assertEqual(row["measurement_parity_pass_pairs"], 3)
        self.assertEqual(row["statistics_pairs_used"], 3)
        self.assertEqual(row["measurement_ledger"]["chain_statuses_counts"], {"7": 3})
        self.assertTrue(row["state_parity"])
        self.assertFalse(document["claim_eligible"])

    def test_existing_native_validator_is_used_without_gpu_access(self) -> None:
        """The compact tool delegates qualification instead of weakening its gate."""
        self._write([self._metadata(), self._result()])
        with (
            patch.object(
                cuda_crossover,
                "validate_native_records",
                wraps=cuda_crossover.validate_native_records,
            ) as validate,
            patch.object(cuda_crossover, "gpu_inventory", side_effect=AssertionError),
            patch.object(cuda_crossover, "run_coordinate", side_effect=AssertionError),
        ):
            self._summarize()
        validate.assert_called_once()

    def test_repeated_json_and_csv_are_byte_deterministic_and_provenance_separated(
        self,
    ) -> None:
        """Preserve producing clean HEAD even when the analyzer is dirty/newer."""
        metadata = self._metadata()
        self._write([metadata, self._result()])
        first = self._summarize()
        second = self._summarize()
        self.assertEqual(
            summary_tool.canonical_json(first), summary_tool.canonical_json(second)
        )
        self.assertEqual(summary_tool.csv_table(first), summary_tool.csv_table(second))
        self.assertEqual(first["producing_metadata"], metadata)
        self.assertEqual(first["analyzer_identity"], self.analyzer)
        self.assertEqual(
            first["analysis_qualification"], "diagnostic_dirty_or_unknown_analyzer"
        )
        raw = self.raw_path.read_bytes()
        self.assertEqual(
            first["raw_identity"]["sha256"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(first["raw_identity"]["byte_count"], len(raw))
        self.assertEqual(first["native_selection_association"], "UNVERIFIED")
        table = list(csv.DictReader(StringIO(summary_tool.csv_table(first))))
        self.assertEqual(table[0]["claim_eligible"], "false")
        self.assertEqual(table[0]["producing_source_revision"], "d" * 40)
        self.assertEqual(table[0]["analyzer_source_revision"], "e" * 40)
        self.assertEqual(table[0]["fresh_checkpoint_bytes"], "20")
        self.assertEqual(table[0]["chain_total_executables"], "6")
        self.assertIn(
            "opaque CUDA graph/executable", table[0]["retained_device_bytes_scope"]
        )

    def test_unknown_analyzer_git_state_is_not_clean(self) -> None:
        """Missing Git tooling gives unknown, not clean-source evidence."""
        with patch.object(cuda_crossover, "_git_value", return_value=None):
            identity = summary_tool.analyzer_identity(self.directory)
        self.assertIsNone(identity["working_tree_dirty"])
        self._write([self._metadata(), self._result()])
        document = summary_tool.summarize(
            self.raw_path, [self.coordinate], analyzer=identity
        )
        self.assertEqual(
            document["analysis_qualification"], "diagnostic_dirty_or_unknown_analyzer"
        )

    def test_missing_and_duplicate_requested_coordinates_are_rejected(self) -> None:
        """The requested grid must not shrink to surviving coordinates."""
        other = cuda_crossover.Coordinate(40, 8, 1)
        self._write([self._metadata(2), self._result()])
        with self.assertRaisesRegex(ValueError, "missing requested"):
            self._summarize([self.coordinate, other])
        self._write([self._metadata(), self._result(), self._result()])
        with self.assertRaisesRegex(ValueError, "duplicate coordinate"):
            self._summarize()

    def test_subset_must_be_explicit_not_inferred_from_surviving_rows(self) -> None:
        """Explicit axes, not surviving results, define subset completeness."""
        self._write([self._metadata(), self._result()])
        with self.assertRaisesRegex(ValueError, "explicit grid"):
            summary_tool.summarize(self.raw_path, analyzer=self.analyzer)
        self._write(
            [self._metadata(), self._result(cuda_crossover.Coordinate(40, 8, 1))]
        )
        with self.assertRaisesRegex(ValueError, "unrequested coordinate"):
            self._summarize()

    def test_strict_malformed_truncated_and_nonfinite_input_is_rejected(self) -> None:
        """Reject ambiguous JSON objects and truncated outer matrix streams."""
        for tail in ('{"record_type":', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            self._write([self._metadata(), self._result()])
            with self.raw_path.open("a", encoding="utf-8") as stream:
                stream.write(tail)
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                self._summarize()

    def test_false_passed_parity_and_subprocess_ledgers_are_rejected(self) -> None:
        """Use native validation to reject false replay and subprocess passes."""
        for change in (
            "contradictory_iterations",
            "missing_sample",
            "negative_time",
            "duplicate_sample",
            "bad_memory",
            "failed_process",
            "boolean_process",
            "boolean_iteration_bound",
            "parity_false",
        ):
            result = self._result()
            if change == "contradictory_iterations":
                result["samples"][1]["tail_iterations"] = [2]
            elif change == "missing_sample":
                result["samples"].pop()
            elif change == "negative_time":
                result["samples"][1]["chain_ms"] = -1
            elif change == "duplicate_sample":
                result["samples"][2] = copy.deepcopy(result["samples"][1])
            elif change == "bad_memory":
                result["summary"]["fixture_device_bytes"] = 100
            elif change == "failed_process":
                result["subprocess_return_code"] = 1
            elif change == "boolean_process":
                result["subprocess_return_code"] = False
            elif change == "boolean_iteration_bound":
                result["summary"]["maximum_iterations"] = True
            else:
                result["samples"][1]["state_parity"] = False
            self._replay_copies(result)
            self._write([self._metadata(), result])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self._summarize()

    def test_redundant_raw_copies_cannot_disagree(self) -> None:
        """Modifying a convenient copy cannot hide raw-record tampering."""
        for copy_name in ("samples", "stdout"):
            result = self._result()
            if copy_name == "samples":
                result["samples"][1]["chain_ms"] = 9
            else:
                result["stdout"] = ""
            self._write([self._metadata(), result])
            with (
                self.subTest(copy_name=copy_name),
                self.assertRaisesRegex(ValueError, "copies disagree"),
            ):
                self._summarize()

    def test_native_integer_fields_reject_boolean_and_float_aliases(self) -> None:
        """Consistent raw copies still require integers, not equal-valued aliases."""
        for field, value in (
            ("ao_count", 40.0),
            ("batch_size", True),
            ("active_denominator", True),
            ("active_system_count", True),
            ("warmup_pairs", True),
            ("measured_pairs", 3.0),
            ("active_mask", [True]),
            ("active_mask", [1.0]),
            ("execution_counts.chain_calls", 4.0),
            ("execution_counts.tail_calls", 4.0),
            ("execution_counts.total_calls", 8.0),
            ("graph_executable_counts.chain_dispatch_family", 5.0),
            ("graph_executable_counts.chain_total", 6.0),
            ("graph_executable_counts.tail_total", 2.0),
        ):
            result = self._result()
            members = field.split(".")
            target = result["summary"]
            for member in members[:-1]:
                target = target[member]
            target[members[-1]] = value
            self._replay_copies(result)
            self._write([self._metadata(), result])
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self._summarize()

    def test_matching_iteration_ledgers_cannot_exceed_declared_bound(self) -> None:
        """Reject matching modes exceeding one iteration in either sample phase."""
        for phase in ("warmup", "measurement"):
            result = self._result()
            sample = next(
                sample for sample in result["samples"] if sample["sample_kind"] == phase
            )
            sample.update(chain_iterations=[2], tail_iterations=[2])
            self._replay_copies(result)
            self._write([self._metadata(), result])
            with (
                self.subTest(phase=phase),
                self.assertRaisesRegex(ValueError, r"iterations.*0\.\.1"),
            ):
                self._summarize()

    def test_invalid_failed_native_contract_retains_diagnostics_without_statistics(
        self,
    ) -> None:
        """Retain schema and iteration failures without usable timing pairs."""
        for fault in ("integer_alias", "iteration_bound"):
            result = self._result()
            result.update(
                status="failure",
                reason="invalid native contract",
                subprocess_return_code=1,
            )
            result["summary"]["coordinate_status"] = "failure"
            if fault == "integer_alias":
                result["summary"]["batch_size"] = True
            else:
                result["samples"][1].update(chain_iterations=[2], tail_iterations=[2])
            self._replay_copies(result)
            self._write([self._metadata(), result])
            with self.subTest(fault=fault):
                document = self._summarize()
                row = document["coordinates"][0]
                self.assertEqual(row["status"], "failure")
                self.assertEqual(
                    row["native_validation_status"], "incomplete_or_invalid_failure"
                )
                self.assertEqual(row["statistics_pairs_used"], 0)
                self.assertEqual(row["statistics_pairs_unvalidated"], 3)
                self.assertIsNone(row["chain_ms_median"])
                self.assertIsNone(row["pair_ratio_median"])
                self.assertIsNone(row["measurement_ledger"])
                self.assertIsNotNone(row["native_validation_error"])
                self.assertEqual(row["unvalidated_native_summary"], result["summary"])
                self.assertFalse(document["claim_eligible"])

    def test_unavailable_native_dimensions_require_integer_types(self) -> None:
        """Keep the smaller unavailable schema while rejecting numeric aliases."""
        for field, value in (("ao_count", 40.0), ("batch_size", True)):
            result = self._result()
            result.update(
                status="unavailable",
                reason="no CUDA device available",
                samples=[],
                summary={
                    "record_type": "coordinate_summary",
                    "coordinate_status": "unavailable",
                    "ao_count": 40,
                    "batch_size": 1,
                    "reason": "no CUDA device available",
                },
            )
            result["summary"][field] = value
            self._replay_copies(result)
            self._write([self._metadata(), result])
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._summarize()

    def test_failed_and_unavailable_coordinates_are_retained(self) -> None:
        """Non-passing coordinates must not disappear from the compact grid."""
        other = cuda_crossover.Coordinate(40, 8, 1)
        failed = self._result()
        failed["status"] = "failure"
        failed["reason"] = "chain and tail differ"
        failed["samples"][1]["state_parity"] = False
        failed["summary"]["state_parity"] = False
        failed["summary"]["coordinate_status"] = "failure"
        failed["subprocess_return_code"] = 1
        self._replay_copies(failed)
        unavailable = self._result(other)
        unavailable.update(
            {
                "status": "unavailable",
                "reason": "forced graph unavailable",
                "samples": [],
                "summary": {
                    "record_type": "coordinate_summary",
                    "coordinate_status": "unavailable",
                    "ao_count": 40,
                    "batch_size": 8,
                    "reason": "forced graph unavailable",
                },
            }
        )
        self._replay_copies(unavailable)
        self._write([self._metadata(2), unavailable, failed])
        document = self._summarize([other, self.coordinate])
        self.assertEqual(
            document["coordinate_counts"], {"pass": 0, "failure": 1, "unavailable": 1}
        )
        failed_row, unavailable_row = document["coordinates"]
        self.assertEqual(failed_row["status"], "failure")
        self.assertFalse(failed_row["state_parity"])
        self.assertEqual(failed_row["chain_ms_median"], 4)
        self.assertIsNone(unavailable_row["chain_ms_median"])
        self.assertEqual(unavailable_row["reason"], "forced graph unavailable")

    def test_failed_partial_and_launchless_unavailable_rows_keep_diagnostics(
        self,
    ) -> None:
        """Retain failure diagnostics without timing unvalidated partial samples."""
        result = self._result()
        result.update(
            {
                "status": "failure",
                "reason": "native crash",
                "summary": None,
                "samples": result["samples"][:2],
                "subprocess_return_code": 1,
            }
        )
        self._replay_copies(result)
        result["stdout"] += '\n{"record_type":'
        result["parse_diagnostic"] = "truncated native summary"
        self._write([self._metadata(), result])
        row = self._summarize()["coordinates"][0]
        self.assertEqual(row["status"], "failure")
        self.assertEqual(row["measurement_pairs_observed"], 1)
        self.assertIsNone(row["chain_ms_median"])
        self.assertEqual(row["statistics_pairs_unvalidated"], 1)
        self.assertIn("coordinate summary", row["native_validation_error"])
        self.assertEqual(row["parse_diagnostic"], "truncated native summary")
        result.update(
            {
                "status": "unavailable",
                "reason": "memory preflight",
                "samples": [],
                "summary": None,
                "subprocess_return_code": None,
                "parse_diagnostic": None,
            }
        )
        self._replay_copies(result)
        self._write([self._metadata(), result])
        self.assertEqual(self._summarize()["coordinates"][0]["status"], "unavailable")

    def test_zero_tail_ratio_is_explicit_not_dropped_silently(self) -> None:
        """Count undefined ratios without hiding the measurement count."""
        result = self._result()
        result["samples"][1]["tail_ms"] = 0
        self._replay_copies(result)
        self._write([self._metadata(), result])
        row = self._summarize()["coordinates"][0]
        self.assertEqual(row["measurement_pairs_observed"], 3)
        self.assertEqual(row["pair_ratio_defined_count"], 2)
        self.assertEqual(row["pair_ratio_undefined_count"], 1)
        self.assertEqual(row["pair_ratio_median"], 1.5)

    def test_rounded_static_mask_fraction_is_explicit_in_both_tables(self) -> None:
        """B1 at requested quarter activity measures a zero-active mask, not 25%."""
        coordinate = cuda_crossover.Coordinate(40, 1, 4)
        self._write([self._metadata(), self._result(coordinate)])
        document = self._summarize([coordinate])
        row = document["coordinates"][0]
        self.assertEqual(row["active_fraction_requested"], 0.25)
        self.assertEqual(row["effective_active_fraction"], 0)
        self.assertEqual(row["active_system_count"], 0)
        table = list(csv.DictReader(StringIO(summary_tool.csv_table(document))))
        self.assertEqual(table[0]["effective_active_fraction"], "0.0")

    def test_negative_and_boolean_metadata_counts_are_rejected(self) -> None:
        """Reject booleans and negative values as hostile protocol counts."""
        for key, value in (
            ("warmups", -1),
            ("paired_samples", -3),
            ("seed", -515),
            ("coordinates_requested", True),
        ):
            metadata = self._metadata()
            metadata[key] = value
            self._write([metadata, self._result()])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._summarize()

    def test_compact_projection_keeps_complete_grid_and_nonpassing_outcomes(
        self,
    ) -> None:
        """Keep all 90 coordinates and both non-passing outcomes under the byte cap."""
        coordinates = cuda_crossover.build_grid()
        results = [self._result(coordinate) for coordinate in coordinates]
        failed = results[0]
        failed.update(
            status="failure", reason="state parity failed", subprocess_return_code=1
        )
        failed["samples"][1]["state_parity"] = False
        failed["summary"].update(coordinate_status="failure", state_parity=False)
        self._replay_copies(failed)
        unavailable = results[-1]
        unavailable.update(
            status="unavailable",
            reason="no CUDA device available",
            samples=[],
            summary={
                "record_type": "coordinate_summary",
                "coordinate_status": "unavailable",
                "ao_count": coordinates[-1].ao_count,
                "batch_size": coordinates[-1].batch_size,
                "reason": "no CUDA device available",
            },
        )
        self._replay_copies(unavailable)
        self._write([self._metadata(len(coordinates)), *results])
        document = self._summarize(coordinates)
        table = summary_tool.csv_table(document, compact=True)
        rows = list(csv.DictReader(StringIO(table)))
        self.assertLessEqual(
            len(table.encode("utf-8")), summary_tool.TABLE_CSV_MAX_BYTES
        )
        self.assertEqual(len(rows), 90)
        self.assertEqual(
            {
                (
                    int(row["ao_count"]),
                    int(row["batch_size"]),
                    int(row["active_denominator"]),
                )
                for row in rows
            },
            {
                (
                    coordinate.ao_count,
                    coordinate.batch_size,
                    coordinate.active_denominator,
                )
                for coordinate in coordinates
            },
        )
        self.assertEqual(
            {
                status: sum(row["status"] == status for row in rows)
                for status in ("pass", "failure", "unavailable")
            },
            {"pass": 88, "failure": 1, "unavailable": 1},
        )
        self.assertEqual(rows[0]["state_parity"], "false")
        self.assertEqual(rows[-1]["pair_ratio_median"], "")
        self.assertTrue(all(row["claim_eligible"] == "false" for row in rows))

    def test_compact_display_rounding_does_not_modify_full_summary(self) -> None:
        """Only the projection rounds floats; full JSON and CSV stay authoritative."""
        result = self._result()
        for sample in result["samples"]:
            if sample["sample_kind"] == "measurement":
                sample["chain_ms"] = 1.234567891234
        self._replay_copies(result)
        self._write([self._metadata(), result])
        document = self._summarize()
        original = summary_tool.canonical_json(document)
        full = list(csv.DictReader(StringIO(summary_tool.csv_table(document))))
        compact = list(
            csv.DictReader(StringIO(summary_tool.csv_table(document, compact=True)))
        )
        self.assertEqual(full[0]["chain_ms_median"], "1.234567891234")
        self.assertEqual(compact[0]["chain_ms_median"], "1.23457")
        self.assertEqual(summary_tool.canonical_json(document), original)
        self.assertEqual(
            summary_tool.csv_table(document, compact=True),
            summary_tool.csv_table(document, compact=True),
        )

    def test_cli_optional_projection_records_identity_and_checks_collisions(
        self,
    ) -> None:
        """Link projection bytes to external full provenance without overwriting raw."""
        self._write([self._metadata(), self._result()])
        json_path = self.directory / "summary.json"
        csv_path = self.directory / "summary.csv"
        table_path = self.directory / "table.csv"
        arguments = [
            "--input",
            str(self.raw_path),
            "--json-output",
            str(json_path),
            "--csv-output",
            str(csv_path),
            "--table-csv-output",
            str(table_path),
            "--ao-counts",
            "40",
            "--batches",
            "1",
            "--active-fractions",
            "1",
        ]
        with (
            redirect_stdout(StringIO()),
            patch.object(summary_tool, "analyzer_identity", return_value=self.analyzer),
        ):
            self.assertEqual(summary_tool.main(arguments), 0)
        document = json.loads(json_path.read_text())
        projection = document["table_csv_projection"]
        payload = table_path.read_bytes()
        self.assertEqual(projection["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(projection["byte_count"], len(payload))
        self.assertEqual(projection["coordinate_count"], 1)
        self.assertEqual(projection["floating_point_format"], ".6g")
        self.assertEqual(document["producing_metadata"], self._metadata())
        self.assertEqual(document["analyzer_identity"], self.analyzer)
        self.assertEqual(
            document["raw_identity"]["sha256"],
            hashlib.sha256(self.raw_path.read_bytes()).hexdigest(),
        )
        self.assertFalse(document["claim_eligible"])
        snapshots = {
            path: path.read_bytes()
            for path in (self.raw_path, json_path, csv_path, table_path)
        }
        for collision in (self.raw_path, json_path, csv_path):
            collided = list(arguments)
            collided[collided.index("--table-csv-output") + 1] = str(collision)
            with (
                redirect_stderr(StringIO()),
                self.subTest(collision=collision),
                self.assertRaises(SystemExit),
            ):
                summary_tool.main([*collided, "--overwrite"])
            self.assertEqual({path: path.read_bytes() for path in snapshots}, snapshots)

    def test_oversized_projection_is_rejected_before_any_outputs(self) -> None:
        """Reject over-budget tables without truncating rows or writing outputs."""
        self._write([self._metadata(), self._result()])
        paths = [
            self.directory / name
            for name in ("summary.json", "summary.csv", "table.csv")
        ]
        arguments = [
            "--input",
            str(self.raw_path),
            "--json-output",
            str(paths[0]),
            "--csv-output",
            str(paths[1]),
            "--table-csv-output",
            str(paths[2]),
            "--ao-counts",
            "40",
            "--batches",
            "1",
            "--active-fractions",
            "1",
        ]
        with (
            patch.object(summary_tool, "TABLE_CSV_MAX_BYTES", 1),
            patch.object(summary_tool, "analyzer_identity", return_value=self.analyzer),
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit),
        ):
            summary_tool.main(arguments)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_cli_generates_both_outputs_and_rejects_collision_and_truncation(
        self,
    ) -> None:
        """Never overwrite raw artifacts or summarize truncated outer JSONL."""
        self._write([self._metadata(), self._result()])
        json_path = self.directory / "compact.json"
        csv_path = self.directory / "compact.csv"
        arguments = [
            "--input",
            str(self.raw_path),
            "--json-output",
            str(json_path),
            "--csv-output",
            str(csv_path),
            "--ao-counts",
            "40",
            "--batches",
            "1",
            "--active-fractions",
            "1",
        ]
        with (
            redirect_stdout(StringIO()),
            patch.object(summary_tool, "analyzer_identity", return_value=self.analyzer),
        ):
            self.assertEqual(summary_tool.main(arguments), 0)
        self.assertFalse(json.loads(json_path.read_text())["claim_eligible"])
        self.assertEqual(len(list(csv.DictReader(StringIO(csv_path.read_text())))), 1)
        raw_before = self.raw_path.read_bytes()
        arguments[arguments.index("--json-output") + 1] = str(self.raw_path)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            summary_tool.main([*arguments, "--overwrite"])
        self.assertEqual(self.raw_path.read_bytes(), raw_before)
        json_path.unlink()
        csv_path.unlink()
        arguments[arguments.index("--json-output") + 1] = str(json_path)
        self.raw_path.write_text('{"record_type":', encoding="utf-8")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            summary_tool.main(arguments)
        self.assertFalse(json_path.exists())
        self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
