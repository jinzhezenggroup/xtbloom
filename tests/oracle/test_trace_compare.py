"""Offline tests for the ``gpuxtb-scc-trace-v1`` comparison foundation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace" / "gpuxtb_scc_compare.py"
)
SPEC = importlib.util.spec_from_file_location("gpuxtb_scc_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("gpuxtb_scc_compare", COMPARE)
SPEC.loader.exec_module(COMPARE)

REVISION = "e9abc395b122018ed688aecb1c3a65cecaf97beb"
PATCH_SHA256 = "d6a51afc4b3c56d6589a2b5b115ea8b4891600c1161c525939ca3cc16e2b4954"


def _iteration(index: int, converged: bool) -> dict:
    mixed_qsh = [[0.1]]
    raw_qsh = [[0.09]]
    mixed_dipoles = [[[0.0, 0.0, 0.0]]]
    raw_dipoles = [[[0.01, -0.01, 0.02]]]
    mixed_quadrupoles = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    raw_quadrupoles = [[[0.01, -0.02, 0.03, -0.04, 0.05, -0.06]]]
    mixed = (
        mixed_qsh[0]
        + [value for atom in mixed_dipoles[0] for value in atom]
        + [value for atom in mixed_quadrupoles[0] for value in atom]
    )
    raw = (
        raw_qsh[0]
        + [value for atom in raw_dipoles[0] for value in atom]
        + [value for atom in raw_quadrupoles[0] for value in atom]
    )
    residual = [raw_value - mixed_value for raw_value, mixed_value in zip(raw, mixed)]
    residual_rms = math.sqrt(sum(value * value / len(residual) for value in residual))
    return {
        "index": index,
        "hamiltonian": [[[-1.0]]],
        "density": [[[0.5]]],
        "eigenvalues": [[-1.0]],
        "occupations": [[0.5], [0.5]],
        "mixed_qsh": mixed_qsh,
        "raw_qsh": raw_qsh,
        "mixed_qat": [[0.1]],
        "raw_qat": [[0.09]],
        "mixed_dipoles": mixed_dipoles,
        "raw_dipoles": raw_dipoles,
        "mixed_quadrupoles": mixed_quadrupoles,
        "raw_quadrupoles": raw_quadrupoles,
        "residual": residual,
        "residual_rms": residual_rms,
        "energy": -2.5 + 0.01 * index,
        "energy_delta": -0.01,
        "convergence": {
            "energy": converged,
            "population": converged,
            "temperature": True,
            "overall": converged,
        },
    }


def _trace(iterations: int = 1) -> dict:
    """Build a valid multi-iteration trace (only the last iteration converges)."""

    all_iterations = [
        _iteration(index + 1, index + 1 == iterations) for index in range(iterations)
    ]
    return {
        "format": COMPARE.FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": PATCH_SHA256,
        },
        "input": {
            "atomic_numbers": [1],
            "positions": [0.0, 0.0, 0.0],
            "molecular_charge": 1.0,
            "unpaired_electrons": 0,
            "spin_channels": 1,
            "temperature": 300.0,
        },
        "basis": {
            "nao": 1,
            "n_shells": 1,
            "n_atoms": 1,
            "atom_to_shell_count": [1],
        },
        "statics": {"overlap": [[[1.0]]], "core_hamiltonian": [[[-1.0]]]},
        "residual_layout": {
            "shell_charges": 1,
            "atomic_dipoles": 3,
            "atomic_quadrupoles": 6,
        },
        "iterations": all_iterations,
        "terminal": {
            "status": COMPARE.TRACE.STATUS_CONVERGED,
            "converged": True,
            "iterations": len(all_iterations),
        },
    }


def _nonconverged_trace(status: int) -> dict:
    """Build one valid completed-but-nonconverged lifecycle."""

    trace = _trace(1)
    trace["iterations"][0]["convergence"] = {
        "energy": False,
        "population": False,
        "temperature": True,
        "overall": False,
    }
    trace["terminal"] = {"status": status, "converged": False, "iterations": 1}
    return trace


def _setup_failure() -> dict:
    """Build a valid failure before any SCC iteration completes."""

    trace = _trace(1)
    trace["iterations"] = []
    trace["terminal"] = {
        "status": COMPARE.TRACE.STATUS_FAILED,
        "converged": False,
        "iterations": 0,
    }
    return trace


def _run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), *(str(argument) for argument in arguments)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class TraceCompareTest(unittest.TestCase):
    """Exercise numeric, exact, metadata, and iteration comparison behavior."""

    def test_identical_trace_matches(self) -> None:
        result = COMPARE.compare_trace(_trace(2), _trace(2))
        self.assertTrue(result.matches, msg=result.render())
        self.assertEqual(result.mismatches, ())
        self.assertEqual(result.profile, "cpu_closed_loop_v1")

    def test_one_value_perturbation_fails_at_expected_path(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["iterations"][0]["density"][0][0][0] = 0.5 + 1.0e-5
        result = COMPARE.compare_trace(actual, golden)
        self.assertFalse(result.matches)
        self.assertEqual(len(result.mismatches), 1)
        mismatch = result.mismatches[0]
        self.assertEqual(mismatch.path, "iterations[0].density[0][0][0]")
        self.assertAlmostEqual(mismatch.absolute_error, 1.0e-5, delta=1.0e-12)
        self.assertIn("iterations[0].density[0][0][0]", result.render())

    def test_later_iteration_error_is_isolated(self) -> None:
        golden = _trace(2)
        actual = deepcopy(golden)
        actual["iterations"][1]["energy"] = golden["iterations"][1]["energy"] + 1.0e-4
        result = COMPARE.compare_trace(actual, golden)
        self.assertFalse(result.matches)
        self.assertEqual(len(result.mismatches), 1)
        self.assertEqual(result.mismatches[0].path, "iterations[1].energy")

    def test_exact_fields_are_compared_exactly(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["provenance"]["oracle_patch_sha256"] = "a" * 64
        result = COMPARE.compare_trace(actual, golden)
        self.assertFalse(result.matches)
        self.assertEqual(result.mismatches[0].path, "provenance.oracle_patch_sha256")

    def test_numeric_integer_and_float_compare_numerically(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        golden["iterations"][0]["energy"] = 0.0
        actual["iterations"][0]["energy"] = 0
        self.assertTrue(COMPARE.compare_trace(actual, golden).matches)

    def test_metadata_only_ignores_numerics(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["iterations"][0]["energy"] += 1.0e-3
        actual["input"]["positions"][0] += 0.25
        result = COMPARE.compare_trace(actual, golden, identical_metadata_only=True)
        self.assertTrue(result.matches, msg=result.render())

    def test_metadata_only_catches_iteration_convergence_flag(self) -> None:
        golden = _trace(2)
        actual = deepcopy(golden)
        actual["iterations"][0]["convergence"]["temperature"] = False
        result = COMPARE.compare_trace(actual, golden, identical_metadata_only=True)
        self.assertFalse(result.matches)
        self.assertEqual(
            result.mismatches[0].path, "iterations[0].convergence.temperature"
        )

    def test_metadata_only_catches_iteration_count(self) -> None:
        result = COMPARE.compare_trace(
            _trace(1), _trace(2), identical_metadata_only=True
        )
        self.assertFalse(result.matches)
        self.assertTrue(
            any(
                mismatch.path in ("iteration_count", "iterations")
                for mismatch in result.mismatches
            )
        )

    def test_metadata_only_catches_terminal_status(self) -> None:
        actual = _nonconverged_trace(COMPARE.TRACE.STATUS_FAILED)
        golden = _nonconverged_trace(COMPARE.TRACE.STATUS_MAX_ITERATIONS)
        result = COMPARE.compare_trace(actual, golden, identical_metadata_only=True)
        self.assertFalse(result.matches)
        self.assertEqual(result.mismatches[0].path, "terminal.status")

    def test_empty_setup_failure_trace_compares(self) -> None:
        result = COMPARE.compare_trace(_setup_failure(), _setup_failure())
        self.assertTrue(result.matches, msg=result.render())

    def test_metadata_only_catches_point_charge_layout(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["input"]["point_charges"] = {
            "positions": [0.0, 0.0, 2.0],
            "charges": [1.0],
            "hardnesses": [999.0],
        }
        result = COMPARE.compare_trace(actual, golden, identical_metadata_only=True)
        self.assertFalse(result.matches)
        self.assertTrue(
            all(
                mismatch.path.startswith("input.point_charges.")
                for mismatch in result.mismatches
            )
        )

    def test_cuda_replay_profile_is_versioned(self) -> None:
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["iterations"][0]["eigenvalues"] = [[-1.0 + 1.0e-10]]
        result = COMPARE.compare_trace(actual, golden, profile="cuda_replay_v1")
        self.assertTrue(result.matches)
        self.assertEqual(result.profile, "cuda_replay_v1")

    def test_unknown_profile_is_actionable(self) -> None:
        with self.assertRaisesRegex(COMPARE.TraceCompareError, "choose one of"):
            COMPARE.compare_trace(_trace(1), _trace(1), profile="cuda_replay")

    def test_malformed_profile_configuration_is_actionable(self) -> None:
        malformed = (
            {"version": True},
            {"atol": "small"},
            {"per_field": []},
            {"per_field": {1: (0.0, 0.0)}},
            {"per_field": {"energy": (0.0,)}},
            {"per_field": {"energy": (False, 0.0)}},
        )
        for overrides in malformed:
            with self.subTest(overrides=overrides):
                arguments = {"name": "invalid", "version": 1, **overrides}
                with self.assertRaises(COMPARE.TraceCompareError):
                    COMPARE.CompareProfile(**arguments)

    def test_tolerance_boundary_is_inclusive(self) -> None:
        profile = COMPARE.CompareProfile("boundary", 1, atol=1.0e-3)
        golden = _trace(1)
        golden["iterations"][0]["energy"] = 0.0
        actual = deepcopy(golden)
        actual["iterations"][0]["energy"] = 1.0e-3
        self.assertTrue(COMPARE.compare_trace(actual, golden, profile=profile).matches)

        actual["iterations"][0]["energy"] = math.nextafter(1.0e-3, math.inf)
        self.assertFalse(COMPARE.compare_trace(actual, golden, profile=profile).matches)

    def test_relative_tolerance_uses_symmetric_max_scale(self) -> None:
        profile = COMPARE.CompareProfile("relative", 1, rtol=0.1)
        first = _trace(1)
        second = deepcopy(first)
        first["iterations"][0]["energy"] = 11.0
        second["iterations"][0]["energy"] = 10.0
        self.assertTrue(COMPARE.compare_trace(first, second, profile=profile).matches)
        self.assertTrue(COMPARE.compare_trace(second, first, profile=profile).matches)

    def test_per_field_override_wins(self) -> None:
        profile = COMPARE.CompareProfile(
            "override", 1, per_field={"energy": (1.0e-3, 0.0)}
        )
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["iterations"][0]["energy"] += 5.0e-4
        self.assertTrue(COMPARE.compare_trace(actual, golden, profile=profile).matches)
        actual["iterations"][0]["density"][0][0][0] += 5.0e-4
        result = COMPARE.compare_trace(actual, golden, profile=profile)
        self.assertFalse(result.matches)
        self.assertEqual(result.mismatches[0].path, "iterations[0].density[0][0][0]")

    def test_array_field_override_reaches_scalar_leaves(self) -> None:
        profile = COMPARE.CompareProfile(
            "array_override", 1, per_field={"density": (1.0e-3, 0.0)}
        )
        golden = _trace(1)
        actual = deepcopy(golden)
        actual["iterations"][0]["density"][0][0][0] += 5.0e-4

        result = COMPARE.compare_trace(actual, golden, profile=profile)

        self.assertTrue(result.matches)
        self.assertEqual(
            COMPARE.CPU_CLOSED_LOOP_V1.tolerance_for("iterations[0].residual[3]"),
            (1.0e-7, 1.0e-7),
        )

    def test_iteration_comparison_reports_logical_array_position(self) -> None:
        golden = _trace(2)
        actual_iteration = deepcopy(golden["iterations"][1])
        actual_iteration["density"] = [[[0.5 + 1.0e-7]]]
        result = COMPARE.compare_iteration(actual_iteration, golden, 2)
        self.assertFalse(result.matches)
        self.assertEqual(result.mismatches[0].path, "iterations[1].density[0][0][0]")

    def test_iteration_index_is_schema_validated_exactly(self) -> None:
        golden = _trace(2)
        actual_iteration = deepcopy(golden["iterations"][1])
        actual_iteration["index"] = 3
        with self.assertRaisesRegex(COMPARE.TRACE.TraceError, "out-of-order index"):
            COMPARE.compare_iteration(actual_iteration, golden, 2)

    def test_iteration_validation_rejects_numeric_string(self) -> None:
        golden = _trace(1)
        actual_iteration = deepcopy(golden["iterations"][0])
        actual_iteration["density"][0][0][0] = "0.5"
        with self.assertRaisesRegex(COMPARE.TRACE.TraceError, "must be a number"):
            COMPARE.compare_iteration(actual_iteration, golden, 1)

    def test_iteration_validation_rejects_missing_field(self) -> None:
        golden = _trace(1)
        actual_iteration = deepcopy(golden["iterations"][0])
        del actual_iteration["density"]
        with self.assertRaisesRegex(COMPARE.TRACE.TraceError, "missing required field"):
            COMPARE.compare_iteration(actual_iteration, golden, 1)

    def test_iteration_validation_rejects_nonfinite_value(self) -> None:
        golden = _trace(1)
        actual_iteration = deepcopy(golden["iterations"][0])
        actual_iteration["energy"] = float("nan")
        with self.assertRaisesRegex(COMPARE.TRACE.TraceError, "must be finite"):
            COMPARE.compare_iteration(actual_iteration, golden, 1)

    def test_iteration_validation_uses_full_trace_shape(self) -> None:
        golden = _trace(1)
        actual_iteration = deepcopy(golden["iterations"][0])
        actual_iteration["residual"] = actual_iteration["residual"][:-1]
        with self.assertRaisesRegex(COMPARE.TRACE.TraceError, "residual needs 10"):
            COMPARE.compare_iteration(actual_iteration, golden, 1)

    def test_iteration_comparison_accepts_valid_tuple_sequence(self) -> None:
        golden = _trace(1)
        golden["iterations"] = tuple(golden["iterations"])
        actual_iteration = deepcopy(golden["iterations"][0])

        result = COMPARE.compare_iteration(actual_iteration, golden, 1)

        self.assertTrue(result.matches)

    def test_iteration_convergence_difference_is_scientific_mismatch(self) -> None:
        golden = _trace(1)
        actual_iteration = deepcopy(golden["iterations"][0])
        actual_iteration["convergence"] = {
            "energy": False,
            "population": False,
            "temperature": True,
            "overall": False,
        }

        result = COMPARE.compare_iteration(actual_iteration, golden, 1)

        self.assertFalse(result.matches)
        self.assertEqual(
            {mismatch.path for mismatch in result.mismatches},
            {
                "iterations[0].convergence.energy",
                "iterations[0].convergence.overall",
                "iterations[0].convergence.population",
            },
        )

    def test_rejects_malformed_actual_trace(self) -> None:
        with self.assertRaises(COMPARE.TRACE.TraceError):
            COMPARE.compare_trace({"format": "gpuxtb-scc-trace-v1"}, _trace(1))

    def test_json_roundtrip_then_compare_identical(self) -> None:
        text = COMPARE.TRACE.dumps(_trace(2))
        actual = json.loads(text)
        result = COMPARE.compare_trace(actual, _trace(2))
        self.assertTrue(result.matches)


class TraceCompareCliTest(unittest.TestCase):
    """Exercise exit codes and the canonical golden read-only boundary."""

    def test_trace_match_is_zero_and_golden_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            golden_bytes = COMPARE.TRACE.dumps(_trace(2)).encode("utf-8")
            golden_path.write_bytes(golden_bytes)
            actual_path.write_bytes(golden_bytes)
            golden_path.chmod(0o444)
            before_stat = golden_path.stat()
            digest = hashlib.sha256(golden_bytes).hexdigest()

            completed = _run_cli(
                "trace",
                actual_path,
                golden_path,
                "--golden-sha256",
                digest,
            )

            self.assertEqual(completed.returncode, COMPARE.EXIT_MATCH, completed.stderr)
            self.assertIn("cpu_closed_loop_v1", completed.stdout)
            self.assertEqual(golden_path.read_bytes(), golden_bytes)
            self.assertEqual(golden_path.stat().st_mtime_ns, before_stat.st_mtime_ns)

    def test_trace_mismatch_is_one_and_reports_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            golden_path.write_text(COMPARE.TRACE.dumps(_trace(1)), encoding="utf-8")
            actual = _trace(1)
            actual["iterations"][0]["energy"] += 0.1
            actual_path.write_text(COMPARE.TRACE.dumps(actual), encoding="utf-8")

            completed = _run_cli("trace", actual_path, golden_path)

            self.assertEqual(completed.returncode, COMPARE.EXIT_MISMATCH)
            self.assertIn("iterations[0].energy", completed.stdout)
            self.assertEqual(completed.stderr, "")

    def test_iteration_cli_selects_and_validates_golden_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual-iteration.json"
            golden = _trace(2)
            golden_path.write_text(COMPARE.TRACE.dumps(golden), encoding="utf-8")
            actual_path.write_text(
                json.dumps(golden["iterations"][1]), encoding="utf-8"
            )

            completed = _run_cli(
                "iteration", actual_path, golden_path, "--iteration", 2
            )

            self.assertEqual(completed.returncode, COMPARE.EXIT_MATCH, completed.stderr)
            self.assertIn("cuda_replay_v1", completed.stdout)

    def test_malformed_json_is_input_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            golden_path.write_text(COMPARE.TRACE.dumps(_trace(1)), encoding="utf-8")
            actual_path.write_text("{broken", encoding="utf-8")

            completed = _run_cli("trace", actual_path, golden_path)

            self.assertEqual(completed.returncode, COMPARE.EXIT_INPUT_ERROR)
            self.assertIn("not valid UTF-8 JSON", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)

    def test_out_of_range_json_integer_is_input_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            golden = _trace(1)
            golden_path.write_text(COMPARE.TRACE.dumps(golden), encoding="utf-8")
            actual = deepcopy(golden)
            actual["iterations"][0]["energy"] = 10**400
            actual_path.write_text(json.dumps(actual), encoding="utf-8")

            completed = _run_cli("trace", actual_path, golden_path)

            self.assertEqual(completed.returncode, COMPARE.EXIT_INPUT_ERROR)
            self.assertIn("too large to convert to float", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)

    def test_wrong_golden_hash_is_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            text = COMPARE.TRACE.dumps(_trace(1))
            golden_path.write_text(text, encoding="utf-8")
            actual_path.write_text(text, encoding="utf-8")

            completed = _run_cli(
                "trace", actual_path, golden_path, "--golden-sha256", "0" * 64
            )

            self.assertEqual(completed.returncode, COMPARE.EXIT_INPUT_ERROR)
            self.assertIn("SHA-256 mismatch", completed.stderr)

    def test_noncanonical_golden_is_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden_path = root / "golden.json"
            actual_path = root / "actual.json"
            golden_path.write_text(json.dumps(_trace(1)), encoding="utf-8")
            actual_path.write_text(COMPARE.TRACE.dumps(_trace(1)), encoding="utf-8")

            completed = _run_cli("trace", actual_path, golden_path)

            self.assertEqual(completed.returncode, COMPARE.EXIT_INPUT_ERROR)
            self.assertIn("not canonical", completed.stderr)

    def test_unknown_cli_profile_is_argparse_error(self) -> None:
        completed = _run_cli(
            "trace", "actual.json", "golden.json", "--profile", "cuda_replay"
        )
        self.assertEqual(completed.returncode, COMPARE.EXIT_INPUT_ERROR)
        self.assertIn("invalid choice", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
