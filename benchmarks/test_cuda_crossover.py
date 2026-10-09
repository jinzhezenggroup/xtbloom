"""Hardware-independent tests for the bounded CUDA crossover harness."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks import cuda_crossover


class CudaCrossoverGridTests(unittest.TestCase):
    """Check fixed matrix dimensions, active masks, and opt-in CLI behavior."""

    def test_default_grid_contains_all_ninety_requested_coordinates(self) -> None:
        """Build every requested AO, batch, and activity combination."""
        grid = cuda_crossover.build_grid()
        self.assertEqual(len(grid), 90)
        self.assertEqual(
            {
                (item.ao_count, item.batch_size, item.active_denominator)
                for item in grid
            },
            {
                (ao_count, batch_size, denominator)
                for ao_count in cuda_crossover.AO_COUNTS
                for batch_size in cuda_crossover.BATCH_SIZES
                for denominator in cuda_crossover.ACTIVE_DENOMINATORS
            },
        )
        self.assertEqual(
            cuda_crossover.Coordinate(62, 8, 2).fixture, "C10H22_all_trans"
        )
        self.assertEqual(
            cuda_crossover.Coordinate(122, 32, 1).fixture, "C20H42_straight_chain"
        )

    def test_grid_rejects_empty_duplicate_and_unlisted_values(self) -> None:
        """Reject matrix dimensions outside the published finite grid."""
        for arguments in (
            ((), cuda_crossover.BATCH_SIZES, cuda_crossover.ACTIVE_DENOMINATORS),
            ((40, 40), cuda_crossover.BATCH_SIZES, cuda_crossover.ACTIVE_DENOMINATORS),
            ((40, 99), cuda_crossover.BATCH_SIZES, cuda_crossover.ACTIVE_DENOMINATORS),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                cuda_crossover.build_grid(*arguments)

    def test_active_mask_is_repeatable_and_rounds_to_a_reported_count(self) -> None:
        """Use a reproducible mask while reporting rounded fractions honestly."""
        mask = cuda_crossover.deterministic_active_mask(8, 2, 515)
        self.assertEqual(mask, cuda_crossover.deterministic_active_mask(8, 2, 515))
        self.assertEqual(sum(mask), 4)
        self.assertEqual(sum(cuda_crossover.deterministic_active_mask(1, 4, 515)), 0)
        self.assertEqual(sum(cuda_crossover.deterministic_active_mask(1, 2, 515)), 1)

    def test_cli_lists_grid_without_running_a_binary(self) -> None:
        """List defaults without needing a compiled executable or CUDA device."""
        output = StringIO()
        with redirect_stdout(output):
            result = cuda_crossover.main(["--list-grid"])
        self.assertEqual(result, 0)
        document = json.loads(output.getvalue())
        self.assertEqual(document["coordinate_count"], 90)
        self.assertEqual(document["warmups"], 5)
        self.assertEqual(document["paired_samples"], 30)

    def test_cli_requires_an_explicit_mode(self) -> None:
        """Require users to request listing or execution explicitly."""
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            cuda_crossover.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_sample_counts_outside_bounds_including_huge_negatives(
        self,
    ) -> None:
        """Reject hostile integer counts without wrapping them into a valid plan."""
        for option, invalid_counts in (
            ("--warmups", (-1, 51, -(1 << 63), -(1 << 64), 1 << 64)),
            ("--samples", (0, -1, 201, -(1 << 63), -(1 << 64), 1 << 64)),
        ):
            for count in invalid_counts:
                with (
                    self.subTest(option=option, count=count),
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    cuda_crossover.main(["--list-grid", f"{option}={count}"])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_parses_a_finite_subset_and_rejects_unknown_fraction(self) -> None:
        """Parse selected grid dimensions without invoking CUDA code."""
        parser = cuda_crossover.build_parser()
        arguments = parser.parse_args(
            [
                "--list-grid",
                "--ao-counts",
                "41,180",
                "--batches",
                "1,256",
                "--active-fractions",
                "1,0.25",
            ]
        )
        self.assertEqual(
            len(
                cuda_crossover.build_grid(
                    arguments.ao_counts,
                    arguments.batches,
                    arguments.active_fractions,
                )
            ),
            8,
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--list-grid", "--active-fractions", "0.2"])


class CudaCrossoverRecordTests(unittest.TestCase):
    """Exercise strict native JSONL parsing and complete sample-ledger checks."""

    def setUp(self) -> None:
        """Prepare one compact coordinate and its deterministic mask."""
        self.coordinate = cuda_crossover.Coordinate(40, 8, 2)
        self.seed = 515
        self.mask = cuda_crossover.deterministic_active_mask(8, 2, self.seed)

    def _sample(self, kind: str, pair_index: int) -> dict[str, Any]:
        values = [1] * self.coordinate.batch_size
        statuses = [0] * self.coordinate.batch_size
        converged = [1] * self.coordinate.batch_size
        global_pair_index = pair_index + (1 if kind == "measurement" else 0)
        return {
            "record_type": "sample",
            "pair_index": pair_index,
            "sample_kind": kind,
            "execution_order": (
                "chain_then_tail" if global_pair_index % 2 == 0 else "tail_then_chain"
            ),
            "chain_ms": 1.25,
            "tail_ms": 1.5,
            "state_parity": True,
            "maximum_energy_delta_hartree": 0.0,
            "maximum_charge_delta_e": 0.0,
            "maximum_state_delta": 0.0,
            "chain_iterations": values.copy(),
            "chain_statuses": statuses.copy(),
            "chain_converged": converged.copy(),
            "tail_iterations": values.copy(),
            "tail_statuses": statuses.copy(),
            "tail_converged": converged.copy(),
        }

    def _summary(self) -> dict[str, Any]:
        calls = 2
        return {
            "record_type": "coordinate_summary",
            "coordinate_status": "pass",
            "qualification": "component_only",
            "ao_count": self.coordinate.ao_count,
            "batch_size": self.coordinate.batch_size,
            "active_denominator": self.coordinate.active_denominator,
            "active_mask": self.mask,
            "active_system_count": sum(self.mask),
            "effective_active_fraction": sum(self.mask) / self.coordinate.batch_size,
            "warmup_pairs": 1,
            "measured_pairs": 1,
            "execution_counts": {
                "chain_calls": calls,
                "tail_calls": calls,
                "total_calls": 2 * calls,
            },
            "graph_executable_counts": {
                "chain_dispatch_family": 4,
                "chain_total": 5,
                "tail_total": 2,
            },
            "setup_ms": 12.5,
            "fixture_arena_bytes": 80,
            "fresh_checkpoint_bytes": 20,
            "fixture_device_bytes": 100,
            "chain_known_control_table_bytes": 20,
            "tail_known_control_table_bytes": 30,
            "retained_device_bytes": 150,
            "retained_device_bytes_scope": (
                "fixture arenas, FRESH checkpoint, and known graph control/table "
                "bytes; opaque CUDA graph/executable and provider allocations excluded"
            ),
            "state_parity": True,
        }

    def test_complete_records_preserve_every_per_system_status_and_iteration(
        self,
    ) -> None:
        """Retain the native per-system ledger unchanged after validation."""
        samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
        status, reason, summary, observed = cuda_crossover.validate_native_records(
            [*samples, self._summary()], self.coordinate, 1, 1, self.seed
        )
        self.assertEqual((status, reason), ("pass", None))
        self.assertIsNotNone(summary)
        self.assertEqual(observed, samples)
        self.assertEqual(observed[1]["chain_iterations"], [1] * 8)
        self.assertEqual(observed[1]["tail_statuses"], [0] * 8)

    def test_state_parity_cannot_override_contradictory_ledgers(self) -> None:
        """Independently compare iteration, status, and convergence vectors."""
        for ledger, contradictory in (
            ("iterations", 2),
            ("statuses", 1),
            ("converged", 0),
        ):
            samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
            samples[1][f"tail_{ledger}"][0] = contradictory
            with (
                self.subTest(ledger=ledger),
                self.assertRaisesRegex(
                    ValueError, f"state_parity contradicts chain/tail {ledger}"
                ),
            ):
                cuda_crossover.validate_native_records(
                    [*samples, self._summary()], self.coordinate, 1, 1, self.seed
                )

    def test_matching_unconverged_one_body_ledgers_are_valid(self) -> None:
        """A single-iteration replay checks parity, not endpoint convergence."""
        samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
        for sample in samples:
            for mode in ("chain", "tail"):
                sample[f"{mode}_converged"] = [0] * self.coordinate.batch_size
                sample[f"{mode}_statuses"] = [1] * self.coordinate.batch_size
        status, reason, _, observed = cuda_crossover.validate_native_records(
            [*samples, self._summary()], self.coordinate, 1, 1, self.seed
        )
        self.assertEqual((status, reason), ("pass", None))
        self.assertEqual(observed, samples)

    def test_missing_sample_and_short_status_ledger_are_rejected(self) -> None:
        """Reject omitted paired samples and truncated system vectors."""
        with self.assertRaisesRegex(ValueError, "expected 2 sample records"):
            cuda_crossover.validate_native_records(
                [self._sample("warmup", 0), self._summary()],
                self.coordinate,
                1,
                1,
                self.seed,
            )
        broken = self._sample("measurement", 0)
        broken["chain_statuses"] = [0]
        with self.assertRaisesRegex(ValueError, "chain_statuses"):
            cuda_crossover.validate_native_records(
                [self._sample("warmup", 0), broken, self._summary()],
                self.coordinate,
                1,
                1,
                self.seed,
            )

    def test_nonalternating_pair_order_is_rejected(self) -> None:
        """Require the pair order to follow the native deterministic AB/BA plan."""
        sample_records = [self._sample("warmup", 0), self._sample("measurement", 0)]
        sample_records[1]["execution_order"] = "chain_then_tail"
        with self.assertRaisesRegex(ValueError, "alternating"):
            cuda_crossover.validate_native_records(
                [*sample_records, self._summary()], self.coordinate, 1, 1, self.seed
            )

    def test_coordinate_mask_parity_and_memory_accounting_are_checked(self) -> None:
        """Cross-check native mask, parity summary, and retained-byte totals."""
        sample_records = [self._sample("warmup", 0), self._sample("measurement", 0)]
        summary = self._summary()
        summary["active_mask"] = [0] * 8
        with self.assertRaisesRegex(ValueError, "active mask"):
            cuda_crossover.validate_native_records(
                [*sample_records, summary], self.coordinate, 1, 1, self.seed
            )
        summary = self._summary()
        summary["retained_device_bytes"] = 149
        with self.assertRaisesRegex(ValueError, "accounting"):
            cuda_crossover.validate_native_records(
                [*sample_records, summary], self.coordinate, 1, 1, self.seed
            )
        summary = self._summary()
        sample_records[1]["state_parity"] = False
        summary["state_parity"] = False
        summary["coordinate_status"] = "failure"
        status, _, _, _ = cuda_crossover.validate_native_records(
            [*sample_records, summary], self.coordinate, 1, 1, self.seed
        )
        self.assertEqual(status, "failure")

    def test_checkpoint_and_opaque_memory_scope_cannot_be_misreported(self) -> None:
        """Count the FRESH checkpoint once and reject an overstated memory scope."""
        samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
        for key, value, diagnostic in (
            ("fixture_device_bytes", 80, "checkpoint byte accounting"),
            ("fixture_device_bytes", 120, "checkpoint byte accounting"),
            ("retained_device_bytes", 170, "retained-device byte accounting"),
            ("retained_device_bytes_scope", "all graph bytes", "accounting scope"),
        ):
            summary = self._summary()
            summary[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, diagnostic),
            ):
                cuda_crossover.validate_native_records(
                    [*samples, summary], self.coordinate, 1, 1, self.seed
                )

    def test_graph_executable_counts_include_roots_and_bodies(self) -> None:
        """Do not confuse per-sample launch counts with setup-owned executables."""
        samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
        summary = self._summary()
        summary["graph_executable_counts"]["chain_total"] = 4
        with self.assertRaisesRegex(ValueError, "executable counts"):
            cuda_crossover.validate_native_records(
                [*samples, summary], self.coordinate, 1, 1, self.seed
            )

    def test_unavailable_coordinate_is_retained_with_reason(self) -> None:
        """Keep unsupported forced graphs as explicit unavailable coordinates."""
        record = {
            "record_type": "coordinate_summary",
            "coordinate_status": "unavailable",
            "ao_count": self.coordinate.ao_count,
            "batch_size": self.coordinate.batch_size,
            "reason": "forced graph is unavailable",
        }
        status, reason, _, samples = cuda_crossover.validate_native_records(
            [record], self.coordinate, 1, 1, self.seed
        )
        self.assertEqual(status, "unavailable")
        self.assertEqual(reason, "forced graph is unavailable")
        self.assertEqual(samples, [])

    def test_source_identity_hashes_executable_and_build_cache(self) -> None:
        """Record immutable binary and CMake-cache identities for reproduction."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            binary = Path(directory) / "benchmark"
            binary.write_bytes(b"native-test-binary")
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\n"
                "CMAKE_CUDA_COMPILER_VERSION:INTERNAL=12.9.86\n",
                encoding="utf-8",
            )
            identity = cuda_crossover.source_identity(root, binary, cache)
        self.assertEqual(len(identity["binary_sha256"]), 64)
        self.assertEqual(len(identity["cmake_cache_sha256"]), 64)
        self.assertIsNone(identity["working_tree_dirty"])
        self.assertIsNone(identity["source_revision"])
        self.assertEqual(
            identity["cmake_build_identity"]["CMAKE_CUDA_COMPILER_VERSION:INTERNAL"],
            "12.9.86",
        )

    def test_strict_jsonl_rejects_duplicate_keys_and_nonfinite_tokens(self) -> None:
        """Reject JSON extensions that would make raw sample data ambiguous."""
        for line in ('{"x":1,"x":2}', '{"elapsed":NaN}'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                cuda_crossover.strict_json_loads(line)
        with self.assertRaisesRegex(
            cuda_crossover.NativeRecordParseError, "line 2"
        ) as raised:
            cuda_crossover.parse_native_records('{"ok":true}\n{"value":Infinity}')
        self.assertEqual(raised.exception.records, [{"ok": True}])

    def test_coordinate_runner_keeps_unavailable_and_partial_failure_records(
        self,
    ) -> None:
        """Preserve missing-device rows and partial per-system ledgers."""
        coordinate = cuda_crossover.Coordinate(40, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "no-cuda"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            unavailable = cuda_crossover.run_coordinate(
                binary, coordinate, 0, 1, self.seed, timeout=5
            )
            self.assertEqual(unavailable["status"], "unavailable")

            partial = self._sample("measurement", 0)
            partial["chain_iterations"] = [1]
            partial["chain_statuses"] = [0]
            partial["chain_converged"] = [1]
            partial["tail_iterations"] = [1]
            partial["tail_statuses"] = [0]
            partial["tail_converged"] = [1]
            partial["execution_order"] = "chain_then_tail"
            script = (
                "#!/usr/bin/env python3\n"
                "import json\n"
                f"record = {partial!r}\n"
                "print(json.dumps(record))\n"
                "raise SystemExit(1)\n"
            )
            binary.write_text(script, encoding="utf-8")
            binary.chmod(0o755)
            failed = cuda_crossover.run_coordinate(
                binary, coordinate, 0, 1, self.seed, timeout=5
            )
            self.assertEqual(failed["status"], "failure")
            self.assertEqual(len(failed["samples"]), 1)
            self.assertEqual(failed["samples"][0]["chain_iterations"], [1])

    def test_malformed_tail_preserves_prefix_raw_stdout_and_parse_diagnostic(
        self,
    ) -> None:
        """Keep earlier ledgers even when a later summary is truncated or invalid."""
        samples = [self._sample("warmup", 0), self._sample("measurement", 0)]
        prefix = "\n".join(json.dumps(sample) for sample in samples) + "\n"
        for tail in ('{"record_type":"coordinate_summary"', '{"value":NaN}'):
            stdout = prefix + tail
            completed = subprocess.CompletedProcess([], 1, stdout, "native failure")
            with (
                self.subTest(tail=tail),
                patch.object(cuda_crossover.subprocess, "run", return_value=completed),
            ):
                failed = cuda_crossover.run_coordinate(
                    Path("fake-native"), self.coordinate, 1, 1, self.seed, timeout=5
                )
            self.assertEqual(failed["status"], "failure")
            self.assertEqual(failed["samples"], samples)
            self.assertEqual(failed["native_records"], samples)
            self.assertEqual(failed["stdout"], stdout)
            self.assertEqual(failed["stderr"], "native failure")
            self.assertIn("line 3", failed["parse_diagnostic"])
            self.assertEqual(failed["subprocess_return_code"], 1)
            self.assertEqual(json.loads(cuda_crossover._json_line(failed)), failed)

    def test_os_launch_error_is_an_explicit_coordinate_failure(self) -> None:
        """Execution-format and other OS errors must not abort the matrix."""
        with patch.object(
            cuda_crossover.subprocess, "run", side_effect=OSError("exec format error")
        ):
            failed = cuda_crossover.run_coordinate(
                Path("fake-native"), self.coordinate, 1, 1, self.seed, timeout=5
            )
        self.assertEqual(failed["status"], "failure")
        self.assertIn("OSError: exec format error", failed["reason"])
        self.assertIsNone(failed["subprocess_return_code"])
        self.assertEqual(failed["samples"], [])

    def test_timeout_preserves_prefix_and_truncated_stdout(self) -> None:
        """TimeoutExpired carries bytes, which must remain available for diagnosis."""
        sample = self._sample("warmup", 0)
        stdout = json.dumps(sample) + '\n{"record_type":'
        timeout = subprocess.TimeoutExpired(
            [], 5, output=stdout.encode(), stderr=b"interrupted summary"
        )
        with patch.object(cuda_crossover.subprocess, "run", side_effect=timeout):
            failed = cuda_crossover.run_coordinate(
                Path("fake-native"), self.coordinate, 1, 1, self.seed, timeout=5
            )
        self.assertEqual(failed["status"], "failure")
        self.assertTrue(failed["timed_out"])
        self.assertEqual(failed["samples"], [sample])
        self.assertEqual(failed["stdout"], stdout)
        self.assertEqual(failed["stderr"], "interrupted summary")
        self.assertIn("line 2", failed["parse_diagnostic"])
        self.assertIsNone(failed["subprocess_return_code"])

    def test_nonexecutable_binary_keeps_all_requested_matrix_coordinates(self) -> None:
        """A real permission error must produce each row, not terminate iteration."""
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "not-executable"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o644)
            output = Path(directory) / "matrix.jsonl"
            with (
                redirect_stdout(StringIO()),
                patch.object(cuda_crossover, "gpu_inventory", return_value={}),
            ):
                result = cuda_crossover.main(
                    [
                        "--run",
                        "--binary",
                        str(binary),
                        "--output",
                        str(output),
                        "--ao-counts",
                        "40",
                        "--batches",
                        "1,8",
                        "--active-fractions",
                        "1",
                        "--warmups",
                        "0",
                        "--samples",
                        "1",
                    ]
                )
            records = cuda_crossover.parse_native_records(
                output.read_text(encoding="utf-8")
            )
        self.assertEqual(result, 1)
        self.assertEqual(records[0]["coordinates_requested"], 2)
        self.assertEqual(len(records[1:]), 2)
        for record in records[1:]:
            self.assertEqual(record["status"], "failure")
            self.assertIn("PermissionError", record["reason"])
            self.assertEqual(record["samples"], [])

    def test_inventory_is_not_native_selected_device_qualification(self) -> None:
        """NVML success cannot imply CUDA ordinal/visibility mapping was checked."""
        completed = subprocess.CompletedProcess(
            [], 0, "GPU model, 100 MiB, driver, GPU-uuid\n", ""
        )
        for outcome in (completed, OSError("nvidia-smi unavailable")):
            with (
                self.subTest(outcome=outcome),
                patch.object(cuda_crossover.subprocess, "run") as run,
            ):
                if isinstance(outcome, OSError):
                    run.side_effect = outcome
                else:
                    run.return_value = outcome
                inventory = cuda_crossover.gpu_inventory()
            self.assertEqual(inventory["native_selected_device_ordinal"], 0)
            self.assertEqual(inventory["native_selection_association"], "UNVERIFIED")
            self.assertEqual(inventory["qualification"], "inventory_only")


if __name__ == "__main__":
    unittest.main()
