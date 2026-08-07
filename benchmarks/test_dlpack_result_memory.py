"""Hardware-free tests for the issue #214 arena-vs-out benchmark protocol."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import dlpack_result_memory as benchmark


class DlpackResultMemoryProtocolTest(unittest.TestCase):
    """Protocol checks that never create a real CUDA context."""

    def test_packed_water_topology_is_consistent(self) -> None:
        """The packed water workload has one system with three atoms."""
        packed = benchmark._packed_water()
        self.assertEqual(packed["atom_offsets"].tolist(), [0, 3])
        self.assertEqual(packed["atomic_numbers"].tolist(), [8, 1, 1])
        self.assertEqual(packed["positions"].shape, (3, 3))
        self.assertEqual(packed["molecular_charges"].tolist(), [0.0])
        self.assertEqual(packed["unpaired_electrons"].tolist(), [0])
        self.assertEqual(packed["spin_channels"].tolist(), [1])

    def test_summary_statistics_are_exact(self) -> None:
        """Latency summary uses the documented distribution statistics."""
        summary = benchmark._summary([1.0, 2.0, 2.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean_ms"], 2.25)
        self.assertAlmostEqual(summary["median_ms"], 2.0)
        self.assertAlmostEqual(summary["min_ms"], 1.0)
        self.assertAlmostEqual(summary["max_ms"], 4.0)

    def test_output_refuses_overwrite_before_gpu_work(self) -> None:
        """The CLI rejects an existing JSON path before any GPU work."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            output.write_text("stale", encoding="utf-8")
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "dlpack_result_memory.py",
                        "--library",
                        str(Path(tmp) / "libgpuxtb.so"),
                        "--output",
                        str(output),
                    ],
                ),
                mock.patch.object(
                    benchmark,
                    "_require_gpu",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(SystemExit),
            ):
                benchmark.main()

    def test_output_refuses_existing_sibling_csv(self) -> None:
        """A stale CSV cannot be overwritten by a fresh JSON request."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            output.with_suffix(".csv").write_text("stale", encoding="utf-8")
            with self.assertRaises(SystemExit):
                benchmark._artifact_paths(output)

    def test_requires_gpu_raises_when_backend_absent(self) -> None:
        """The production probe translates an unavailable native context."""
        from gpuxtb.exceptions import GPUxtbRuntimeError

        with (
            mock.patch(
                "gpuxtb.interface.Context",
                side_effect=GPUxtbRuntimeError("injected unavailable backend"),
            ),
            self.assertRaisesRegex(SystemExit, "CUDA backend is not usable"),
        ):
            benchmark._require_gpu()

    def test_dirty_source_is_rejected_before_gpu_work(self) -> None:
        """Final evidence cannot be emitted from unrecoverable source bytes."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "dlpack_result_memory.py",
                        "--library",
                        str(Path(tmp) / "libgpuxtb.so"),
                        "--output",
                        str(output),
                    ],
                ),
                mock.patch.object(
                    benchmark, "_git_revision", return_value=("a" * 40, True)
                ),
                mock.patch.object(
                    benchmark,
                    "_require_gpu",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaisesRegex(SystemExit, "dirty sources"),
            ):
                benchmark.main()

    def test_invalid_repetitions_precede_identity_probe(self) -> None:
        """The CLI rejects an empty statistical protocol immediately."""
        with (
            mock.patch(
                "sys.argv",
                [
                    "dlpack_result_memory.py",
                    "--library",
                    "/tmp/libgpuxtb.so",
                    "--output",
                    "/tmp/unused-evidence.json",
                    "--repetitions",
                    "1",
                ],
            ),
            mock.patch.object(
                benchmark,
                "_git_revision",
                side_effect=AssertionError("must not run"),
            ),
            self.assertRaisesRegex(SystemExit, "at least 2"),
        ):
            benchmark.main()

    def test_paired_gate_has_explicit_five_percent_margin(self) -> None:
        """The conclusion follows the documented mean-overhead gate."""
        passing = benchmark._paired_summary([1.04, 1.04], [1.0, 1.0])
        failing = benchmark._paired_summary([1.06, 1.06], [1.0, 1.0])
        self.assertTrue(passing["passes_mean_overhead_gate"])
        self.assertFalse(failing["passes_mean_overhead_gate"])
        self.assertEqual(passing["max_mean_overhead_fraction"], 0.05)

    def test_publication_writes_json_and_lf_only_csv(self) -> None:
        """Production publication writes the documented schema and line endings."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "x.json"
            mode_records = {
                "arena": {"raw_latency_ms": [1.0, 2.0]},
                "out": {"raw_latency_ms": [3.0]},
            }
            json_path, csv_path = benchmark._write_artifacts(
                {"schema_version": 2}, output, mode_records
            )
            self.assertEqual(json.loads(json_path.read_text()), {"schema_version": 2})
            self.assertNotIn(b"\r", csv_path.read_bytes())
            with csv_path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["mode", "sample", "latency_ms"])
            self.assertEqual(rows[1], ["arena", "0", "1.000000"])
            self.assertEqual(rows[-1], ["out", "0", "3.000000"])


if __name__ == "__main__":
    unittest.main()
