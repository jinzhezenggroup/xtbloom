"""Hardware-free tests for the public Hessian benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from benchmarks import hessian as hb
from benchmarks.natoms_scaling import make_alkane


class FakeEngine(hb.HessianEngine):
    """Return a deterministic translationally invariant quadratic Hessian."""

    def __init__(self, batch_size: int = 1, nthreads: int = 16) -> None:
        self.preparations = 0
        self.invocations = 0
        self.batch_size = batch_size
        self.nthreads = nthreads

    def prepare_sample(self) -> None:
        """Count one untimed fresh-state preparation."""
        self.preparations += 1

    def invoke(self) -> hb.EngineResult:
        """Return the deterministic test Hessian."""
        self.invocations += 1
        projector = np.eye(hb.NATOMS) - np.ones((hb.NATOMS, hb.NATOMS)) / hb.NATOMS
        matrix = np.kron(projector, np.eye(3))
        return hb.EngineResult(
            [np.array(matrix, copy=True) for _ in range(self.batch_size)],
            {
                "fake": True,
                "invocation": self.invocations,
            },
        )


def fake_args() -> argparse.Namespace:
    """Return the subset of parsed options consumed by :func:`run_row`."""
    return argparse.Namespace(
        warmups=1,
        repetitions=3,
        hessian_atol=1.0e-12,
        symmetry_atol=1.0e-12,
        acoustic_atol=1.0e-12,
        repeatability_atol=1.0e-12,
        make_reference=False,
        cpu_threads=16,
        max_serial_hessian_batch_size=1,
    )


class HessianBenchmarkTest(unittest.TestCase):
    """Exercise batch semantics, artifact encoding, gates, and failures."""

    def test_timing_summary_uses_complete_hessian_batch_size(self) -> None:
        """Report wall time, amortized latency, and true Hessian throughput."""
        summary = hb.timing_summary([1000.0, 1200.0, 1400.0], 128)
        self.assertEqual(summary["median_ms"], 1200.0)
        self.assertEqual(summary["amortized_ms_per_hessian_at_median"], 9.375)
        self.assertEqual(summary["hessians_per_hour_at_median"], 384000.0)
        with self.assertRaises(hb.BenchmarkError):
            hb.timing_summary([1.0], 0)

    def test_child_coordinates_keep_nthreads_independent_of_batch_size(self) -> None:
        """Batch size changes sampling cost, never the worker budget."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = hb.build_parser().parse_args(
                [
                    "--engines",
                    "xtb",
                    "--xtb-library",
                    str(Path(__file__)),
                    "--nthreads",
                    "16",
                    "--output-json",
                    str(root / "parent.json"),
                    "--output-csv",
                    str(root / "parent.csv"),
                ]
            )
            commands = [
                hb.coordinate_command(
                    args,
                    engine="xtb",
                    hessian_batch_size=batch_size,
                    output_json=root / f"child-{batch_size}.json",
                    output_csv=root / f"child-{batch_size}.csv",
                )
                for batch_size in (1, 128)
            ]

        for batch_size, command in zip((1, 128), commands, strict=True):
            self.assertEqual(
                command[command.index("--batch-sizes") + 1], str(batch_size)
            )
            self.assertEqual(command[command.index("--nthreads") + 1], "16")
        self.assertEqual(commands[0][commands[0].index("--warmups") + 1], "1")
        self.assertEqual(commands[0][commands[0].index("--repetitions") + 1], "3")
        self.assertEqual(commands[1][commands[1].index("--warmups") + 1], "0")
        self.assertEqual(commands[1][commands[1].index("--repetitions") + 1], "1")

    def test_explicit_sampling_policy_overrides_bounded_defaults(self) -> None:
        """Publication runs can deliberately request larger distributions."""
        args = fake_args()
        args.warmups = 2
        args.repetitions = 5
        self.assertEqual(hb.coordinate_sample_policy(args, 128), (2, 5))

    def test_default_coordinate_timeout_bounds_slow_engines(self) -> None:
        """A forgotten timeout must not recreate a tens-of-minutes run."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = hb.build_parser().parse_args(
                [
                    "--output-json",
                    str(root / "result.json"),
                    "--output-csv",
                    str(root / "result.csv"),
                ]
            )
        self.assertEqual(args.coordinate_timeout_seconds, 300.0)

    def test_run_row_rejects_mismatched_effective_nthreads(self) -> None:
        """Do not time a row whose engine changed the fixed worker budget."""
        row = hb.run_row(
            "xtbloom-cpu",
            args=fake_args(),
            molecule=make_alkane(62),
            hessian_batch_size=1,
            references=None,
            factory=lambda *_args, **_kwargs: FakeEngine(nthreads=1),
        )
        self.assertEqual(row["availability"], "unavailable")
        self.assertIn("requested 16, effective 1", row["unavailable_reason"])

    def test_serial_only_engine_skips_impractical_complete_batch(self) -> None:
        """Do not spend tens of minutes repeating a single-system Hessian API."""
        factory = mock.Mock(side_effect=AssertionError("engine must not be created"))
        row = hb.run_row(
            "xtb",
            args=fake_args(),
            molecule=make_alkane(62),
            hessian_batch_size=128,
            references=None,
            factory=factory,
        )
        factory.assert_not_called()
        self.assertEqual(row["availability"], "unavailable")
        self.assertIn("no complete-Hessian batch API", row["unavailable_reason"])
        self.assertEqual(row["nthreads"], 16)

    def test_hessian_round_trip_authenticates_exact_binary64_payload(self) -> None:
        """Retain every Hessian element and reject a forged payload digest."""
        matrix = np.arange(hb.COORDINATE_COUNT**2, dtype=np.float64).reshape(
            hb.COORDINATE_COUNT, hb.COORDINATE_COUNT
        )
        encoded = hb.encode_hessian(matrix)
        np.testing.assert_array_equal(hb.decode_hessian(encoded), matrix)
        self.assertEqual(
            encoded["sha256"],
            hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(),
        )
        encoded["sha256"] = "0" * 64
        with self.assertRaises(hb.BenchmarkError):
            hb.decode_hessian(encoded)

        batch = hb.encode_hessian_batch([matrix, matrix + 1.0])
        decoded = hb.decode_hessian_batch(batch)
        np.testing.assert_array_equal(decoded[0], matrix)
        np.testing.assert_array_equal(decoded[1], matrix + 1.0)

    def test_diagnostics_and_comparison_use_symmetric_views(self) -> None:
        """Retain raw antisymmetry but compare the common symmetric matrix."""
        projector = np.eye(hb.NATOMS) - np.ones((hb.NATOMS, hb.NATOMS)) / hb.NATOMS
        reference = np.kron(projector, np.eye(3))
        actual = reference.copy()
        actual[0, 1] += 2.0e-4
        actual[1, 0] -= 2.0e-4
        diagnostics = hb.hessian_diagnostics(actual)
        self.assertAlmostEqual(
            diagnostics["max_abs_antisymmetry_hartree_per_bohr2"], 4.0e-4
        )
        self.assertLess(
            diagnostics["max_abs_acoustic_row_residual_hartree_per_bohr2"],
            3.0e-4,
        )
        comparison = hb.compare_hessians(actual, reference)
        self.assertLess(comparison["max_abs_delta_hartree_per_bohr2"], 1.0e-15)

    def test_single_reference_qualifies_slot_zero_of_complete_batch(self) -> None:
        """Avoid a 128-call xTB loop while retaining independent correctness."""
        matrix = np.zeros((hb.COORDINATE_COUNT, hb.COORDINATE_COUNT))
        correctness = hb.evaluate_correctness(
            [[matrix, matrix]],
            references=[matrix],
            hessian_atol=1.0e-12,
            symmetry_atol=1.0e-12,
            acoustic_atol=1.0e-12,
            repeatability_atol=1.0e-12,
            is_reference=False,
        )
        comparison = correctness["cross_engine"]
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(comparison["reference_scope"], "slot_zero")
        self.assertEqual(comparison["compared_hessian_indices"], [0])

    def test_run_row_retains_all_samples_and_correctness(self) -> None:
        """Time every requested sample and publish one qualified Hessian."""
        fake = FakeEngine(batch_size=128)

        def factory(*_args: object, **_kwargs: object) -> hb.HessianEngine:
            return fake

        row = hb.run_row(
            "xtbloom-cpu",
            args=fake_args(),
            molecule=make_alkane(62),
            hessian_batch_size=128,
            references=None,
            factory=factory,
        )
        self.assertEqual(row["availability"], "available")
        self.assertEqual(row["timing"]["sample_count"], 3)
        self.assertEqual(len(row["timing"]["samples_ms"]), 3)
        self.assertEqual(row["correctness"]["status"], "pass")
        self.assertEqual(fake.preparations, 4)
        self.assertEqual(fake.invocations, 4)
        self.assertEqual(row["hessian_batch_size"], 128)
        self.assertEqual(row["nthreads"], 16)
        self.assertEqual(row["requested_cpu_threads"], 16)
        self.assertEqual(
            row["timing"]["amortized_ms_per_hessian_at_median"],
            row["timing"]["median_ms"] / 128,
        )
        self.assertEqual(len(row["final_hessians_binary64_le_zlib_base64"]), 128)

    def test_run_row_preserves_unavailable_reason_and_completed_samples(self) -> None:
        """Keep partial timings and the exact exception when a coordinate fails."""

        class FailingEngine(FakeEngine):
            def invoke(self) -> hb.EngineResult:
                """Fail after one successful retained sample."""
                if self.invocations == 1:
                    raise MemoryError("synthetic OOM")
                return super().invoke()

        failing = FailingEngine()

        def factory(*_args: object, **_kwargs: object) -> hb.HessianEngine:
            return failing

        args = fake_args()
        args.warmups = 0
        row = hb.run_row(
            "dxtb-cuda-ad",
            args=args,
            molecule=make_alkane(62),
            hessian_batch_size=1,
            references=None,
            factory=factory,
        )
        self.assertEqual(row["availability"], "unavailable")
        self.assertIn("MemoryError: synthetic OOM", row["unavailable_reason"])
        self.assertEqual(len(row["completed_samples_ms"]), 1)

    def test_json_and_csv_retain_failure_and_hessian_payload(self) -> None:
        """Publish compact summaries without dropping exact outputs or failures."""
        matrix = np.eye(hb.COORDINATE_COUNT)
        rows = [
            {
                "engine": "xtb",
                "natoms": 62,
                "hessian_batch_size": 1,
                "requested_cpu_threads": 16,
                "availability": "available",
                "timing": hb.timing_summary([1.0, 2.0, 3.0]),
                "correctness": {
                    "status": "pass",
                    "diagnostics": {
                        "max_abs_antisymmetry_hartree_per_bohr2": 0.0,
                        "max_abs_acoustic_residual_hartree_per_bohr2": 0.0,
                    },
                    "cross_engine": {"status": "reference"},
                },
                "final_hessians_binary64_le_zlib_base64": (
                    hb.encode_hessian_batch([matrix])
                ),
            },
            {
                "engine": "dxtb-cuda-ad",
                "natoms": 62,
                "hessian_batch_size": 128,
                "requested_cpu_threads": 16,
                "availability": "unavailable",
                "unavailable_reason": "RuntimeError: OOM",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            hb.write_json(json_path, {"rows": rows})
            hb.write_csv(csv_path, rows)
            self.assertIn('"zlib_base64"', json_path.read_text())
            csv_text = csv_path.read_text()
            self.assertIn("dxtb-cuda-ad", csv_text)
            self.assertIn("RuntimeError: OOM", csv_text)

    def test_compact_projection_keeps_every_hessian_identity(self) -> None:
        """Omit dense bytes mechanically while authenticating all batch members."""
        matrix = np.eye(hb.COORDINATE_COUNT)
        document = {
            "metadata": {"runner": "/work/xtbloom/benchmarks/hessian.py"},
            "rows": [
                {
                    "engine": "xtbloom-cpu",
                    "hessian_batch_size": 2,
                    "final_hessians_binary64_le_zlib_base64": (
                        hb.encode_hessian_batch([matrix, matrix + 1.0])
                    ),
                }
            ],
        }
        compact = hb.compact_hessian_document(
            document,
            raw_filename="raw.json",
            raw_byte_count=123,
            raw_sha256="a" * 64,
            path_replacements=(("/work/xtbloom", "${XTBLOOM_SOURCE_ROOT}"),),
        )
        row = compact["rows"][0]
        self.assertNotIn("final_hessians_binary64_le_zlib_base64", row)
        self.assertEqual(len(row["final_hessian_identities"]), 2)
        self.assertEqual(
            row["final_hessian_identities"][0]["byte_count"],
            hb.COORDINATE_COUNT**2 * 8,
        )
        self.assertEqual(
            compact["metadata"]["runner"],
            "${XTBLOOM_SOURCE_ROOT}/benchmarks/hessian.py",
        )
        self.assertEqual(
            compact["metadata"]["compact_projection"]["omitted_hessian_payload_count"],
            2,
        )

    def test_isolated_coordinate_rejects_artifact_from_crashed_child(self) -> None:
        """Do not publish timings when native teardown terminates the process."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coordinate.json"
            hb.write_json(
                output,
                {
                    "rows": [
                        {
                            "engine": "xtbloom-cpu",
                            "availability": "available",
                            "timing": hb.timing_summary([12.0]),
                            "correctness": {"status": "pass"},
                            "final_hessians_binary64_le_zlib_base64": [
                                {"payload": True}
                            ],
                        }
                    ]
                },
            )
            completed = mock.Mock(returncode=-11, stdout="", stderr="native crash")
            with mock.patch.object(hb.subprocess, "run", return_value=completed):
                row = hb.run_isolated_coordinate(
                    ["python", "coordinate"],
                    output_json=output,
                )
            self.assertEqual(row["availability"], "unavailable")
            self.assertIn("SIGSEGV", row["unavailable_reason"])
            self.assertEqual(row["completed_samples_ms"], [12.0])
            self.assertNotIn("timing", row)
            self.assertNotIn("correctness", row)
            self.assertNotIn("final_hessians_binary64_le_zlib_base64", row)

    def test_isolated_coordinate_contains_truncated_child_json(self) -> None:
        """Treat invalid child output as one unavailable coordinate."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coordinate.json"
            output.write_text('{"rows": [')
            completed = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(hb.subprocess, "run", return_value=completed):
                row = hb.run_isolated_coordinate(
                    ["python", "coordinate"], output_json=output
                )
        self.assertEqual(row["availability"], "unavailable")
        self.assertIn(
            "successful child produced no usable coordinate row",
            row["unavailable_reason"],
        )
        self.assertIn("JSONDecodeError", row["unavailable_reason"])

    def test_isolated_coordinate_retains_timeout_as_unavailable(self) -> None:
        """Bound impractical engines without losing the requested matrix row."""
        expired = hb.subprocess.TimeoutExpired(
            ["python", "coordinate"], 120.0, stderr="partial diagnostic"
        )
        with mock.patch.object(hb.subprocess, "run", side_effect=expired):
            row = hb.run_isolated_coordinate(
                ["python", "coordinate"],
                output_json=Path("unused.json"),
                timeout_seconds=120.0,
            )
        self.assertEqual(row["availability"], "unavailable")
        self.assertIn("timeout of 120 seconds", row["unavailable_reason"])
        self.assertIn("partial diagnostic", row["unavailable_reason"])
        self.assertEqual(row["completed_samples_ms"], [])

    def test_validate_args_refuses_existing_artifact(self) -> None:
        """Never replace a prior raw timing artifact implicitly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_json = root / "result.json"
            output_json.write_text("stale")
            args = hb.build_parser().parse_args(
                [
                    "--engines",
                    "xtb",
                    "--xtb-library",
                    str(Path(__file__)),
                    "--make-reference",
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(root / "result.csv"),
                ]
            )
            with self.assertRaisesRegex(hb.BenchmarkError, "refusing to overwrite"):
                hb.validate_args(args)

    def test_validate_args_rejects_negative_coordinate_timeout(self) -> None:
        """A disabled timeout is zero; negative or non-finite values are invalid."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = hb.build_parser().parse_args(
                [
                    "--engines",
                    "xtb",
                    "--xtb-library",
                    str(Path(__file__)),
                    "--coordinate-timeout-seconds",
                    "-1",
                    "--output-json",
                    str(root / "result.json"),
                    "--output-csv",
                    str(root / "result.csv"),
                ]
            )
            with self.assertRaisesRegex(hb.BenchmarkError, "finite and nonnegative"):
                hb.validate_args(args)

    def test_nonfinite_hessian_reports_failure_without_acoustic_exception(self) -> None:
        """Retain the primary non-finite reason when diagnostics are unavailable."""
        matrix = np.zeros((hb.COORDINATE_COUNT, hb.COORDINATE_COUNT))
        matrix[0, 0] = np.nan
        correctness = hb.evaluate_correctness(
            [[matrix]],
            references=None,
            hessian_atol=1.0e-3,
            symmetry_atol=1.0e-3,
            acoustic_atol=1.0e-3,
            repeatability_atol=1.0e-8,
            is_reference=False,
        )
        self.assertEqual(correctness["status"], "fail")
        self.assertIn("non_finite_hessian", correctness["reasons"])

    def test_nonfinite_reference_comparison_stays_json_serializable(self) -> None:
        """Skip deltas that would otherwise publish forbidden JSON NaN values."""
        matrix = np.zeros((hb.COORDINATE_COUNT, hb.COORDINATE_COUNT))
        matrix[0, 0] = np.nan
        correctness = hb.evaluate_correctness(
            [[matrix]],
            references=[np.zeros_like(matrix)],
            hessian_atol=1.0e-3,
            symmetry_atol=1.0e-3,
            acoustic_atol=1.0e-3,
            repeatability_atol=1.0e-8,
            is_reference=False,
        )
        self.assertEqual(
            correctness["cross_engine"]["status"],
            "not_comparable_non_finite",
        )
        with tempfile.TemporaryDirectory() as directory:
            hb.write_json(Path(directory) / "result.json", correctness)

    def test_nonfinite_earlier_sample_cannot_pass_on_finite_final_sample(self) -> None:
        """Every retained sample, not only the final payload, must be finite."""
        nonfinite = np.zeros((hb.COORDINATE_COUNT, hb.COORDINATE_COUNT))
        nonfinite[0, 0] = np.nan
        finite = np.zeros_like(nonfinite)
        correctness = hb.evaluate_correctness(
            [[nonfinite], [finite]],
            references=None,
            hessian_atol=1.0e-3,
            symmetry_atol=1.0e-3,
            acoustic_atol=1.0e-3,
            repeatability_atol=1.0e-8,
            is_reference=False,
        )
        self.assertEqual(correctness["status"], "fail")
        self.assertIsNone(correctness["max_abs_repeatability_delta_hartree_per_bohr2"])
        self.assertEqual(
            correctness["diagnostics"]["nonfinite_samples"],
            [{"sample_index": 0, "hessian_indices": [0]}],
        )


if __name__ == "__main__":
    unittest.main()
