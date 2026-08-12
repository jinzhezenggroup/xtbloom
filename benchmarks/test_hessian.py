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

    def __init__(self) -> None:
        self.preparations = 0
        self.invocations = 0

    def prepare_sample(self) -> None:
        """Count one untimed fresh-state preparation."""
        self.preparations += 1

    def invoke(self) -> hb.EngineResult:
        """Return the deterministic test Hessian."""
        self.invocations += 1
        projector = np.eye(hb.NATOMS) - np.ones((hb.NATOMS, hb.NATOMS)) / hb.NATOMS
        matrix = np.kron(projector, np.eye(3))
        return hb.EngineResult(
            matrix,
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
    )


class HessianBenchmarkTest(unittest.TestCase):
    """Exercise batch semantics, artifact encoding, gates, and failures."""

    def test_displacement_batch_size_maps_to_atom_limit_and_exact_chunks(self) -> None:
        """Expose displacement-system batch sizes without leaking atom limits."""
        self.assertEqual(hb.displacement_atom_limit(1), 62)
        self.assertEqual(hb.displacement_atom_limit(128), 7936)
        self.assertEqual(hb.displacement_chunks(1), [1] * 372)
        self.assertEqual(hb.displacement_chunks(128), [128, 128, 116])
        with self.assertRaises(hb.BenchmarkError):
            hb.displacement_atom_limit(0)

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

    def test_run_row_retains_all_samples_and_correctness(self) -> None:
        """Time every requested sample and publish one qualified Hessian."""
        fake = FakeEngine()

        def factory(*_args: object, **_kwargs: object) -> hb.HessianEngine:
            return fake

        row = hb.run_row(
            "xtbloom-cpu",
            args=fake_args(),
            molecule=make_alkane(62),
            displacement_batch_size=128,
            reference=None,
            factory=factory,
        )
        self.assertEqual(row["availability"], "available")
        self.assertEqual(row["timing"]["sample_count"], 3)
        self.assertEqual(len(row["timing"]["samples_ms"]), 3)
        self.assertEqual(row["correctness"]["status"], "pass")
        self.assertEqual(fake.preparations, 4)
        self.assertEqual(fake.invocations, 4)
        self.assertEqual(row["displacement_batch_size"], 128)

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
            displacement_batch_size=None,
            reference=None,
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
                "displacement_batch_size": None,
                "availability": "available",
                "timing": hb.timing_summary([1.0, 2.0, 3.0]),
                "correctness": {
                    "status": "pass",
                    "diagnostics": hb.hessian_diagnostics(matrix),
                    "cross_engine": {"status": "reference"},
                },
                "final_hessian_binary64_le_zlib_base64": hb.encode_hessian(matrix),
            },
            {
                "engine": "dxtb-cuda-ad",
                "natoms": 62,
                "displacement_batch_size": None,
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
                            "final_hessian_binary64_le_zlib_base64": {"payload": True},
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
            self.assertNotIn("final_hessian_binary64_le_zlib_base64", row)

    def test_validate_args_refuses_existing_artifact(self) -> None:
        """Never replace a prior raw timing artifact implicitly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_json = root / "result.json"
            output_json.write_text("stale")
            args = argparse.Namespace(
                engines=("xtb",),
                displacement_batch_sizes=(1, 128),
                make_reference=True,
                reference_json=None,
                output_json=output_json,
                output_csv=root / "result.csv",
                cpu_threads=16,
                warmups=1,
                repetitions=5,
                scc_max_iterations=500,
                step=0.005,
                scc_charge_tolerance=1.0e-4,
                scc_energy_tolerance=1.0e-6,
                hessian_atol=2.0e-3,
                symmetry_atol=2.0e-3,
                acoustic_atol=2.0e-3,
                repeatability_atol=1.0e-8,
                library=None,
                xtb_library=Path(__file__),
            )
            with self.assertRaisesRegex(hb.BenchmarkError, "refusing to overwrite"):
                hb.validate_args(args)


if __name__ == "__main__":
    unittest.main()
