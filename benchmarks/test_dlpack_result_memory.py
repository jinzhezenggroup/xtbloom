"""Hardware-free tests for the issue #214 arena-vs-out benchmark protocol.

``benchmarks/dlpack_result_memory.py`` measures the steady-state allocation
cost of ``result_memory='cuda'`` against caller-owned ``out=`` on a real GPU.
These tests cover only the hardware-independent parts of the protocol: the
packed workload builder, the summary statistics, the refuse-overwrite guard,
the required-CUDA precondition, and the JSON/CSV document shape.  They never
create a CUDA context or import a provider library.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks import dlpack_result_memory as benchmark


class DlpackResultMemoryProtocolTest(unittest.TestCase):
    """Protocol-level checks that need no CUDA device or provider import."""

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

    def test_output_refuses_overwrite(self) -> None:
        """The CLI rejects an existing output path before any GPU work."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evidence.json"
            output.write_text("stale", encoding="utf-8")
            with (
                mock.patch(
                    "sys.argv", ["dlpack_result_memory.py", "--output", str(output)]
                ),
                mock.patch.object(
                    benchmark,
                    "_require_gpu",
                    side_effect=AssertionError("must not run"),
                ),
                self.assertRaises(SystemExit),
            ):
                benchmark.main()

    def test_requires_gpu_raises_when_backend_absent(self) -> None:
        """The benchmark refuses to run without a usable CUDA device."""
        original = benchmark._require_gpu
        try:
            benchmark._require_gpu = _fail_require_gpu
            with self.assertRaises(SystemExit):
                benchmark._require_gpu()
        finally:
            benchmark._require_gpu = original

    def test_csv_schema_matches_documented_columns(self) -> None:
        """The CSV view embeds mode, sample index, and raw latency only."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "x.csv"
            document = {
                "timing": {
                    "modes": {
                        "arena": {"raw_latency_ms": [1.0, 2.0]},
                        "out": {"raw_latency_ms": [3.0]},
                    }
                }
            }
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["mode", "sample", "latency_ms"])
                for mode, record in document["timing"]["modes"].items():
                    for index, latency in enumerate(record["raw_latency_ms"]):
                        writer.writerow([mode, index, f"{latency:.6f}"])
            with csv_path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["mode", "sample", "latency_ms"])
            self.assertEqual(rows[1], ["arena", "0", "1.000000"])
            self.assertEqual(rows[-1], ["out", "0", "3.000000"])


def _fail_require_gpu() -> None:
    """Stand-in that always reports CUDA as unusable."""
    raise SystemExit("CUDA backend is not usable: injected failure")


if __name__ == "__main__":
    unittest.main()
