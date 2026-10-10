"""Offline structural and numerical tests for two-origin replay evidence."""

from __future__ import annotations

import importlib.util
import io
import math
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "tools" / "oracle" / "check_spin_two_origin_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "check_spin_two_origin_replay", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("check_spin_two_origin_replay", CHECKER)
SPEC.loader.exec_module(CHECKER)

CHECKPOINT_SCHEMA = CHECKER._checkpoint_schema()
SCALAR_FIELDS = CHECKER._scalar_fields()
INPUTS = {
    "positions": ("f64", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
    "molecular_charges": ("f64", (3.0,)),
    "atomic_numbers": ("i32", (39, 1)),
    "unpaired_electrons": ("i32", (1,)),
    "spin_channels": ("i32", (2,)),
    "atom_offsets": ("i64", (0, 2)),
}
FIELD_COUNTS = {
    "scc_shell_inputs": 4,
    "scc_dipole_inputs": 12,
    "scc_quadrupole_inputs": 24,
    "mixer_current_inputs": 40,
    "mixer_previous_inputs": 40,
    "mixer_previous_residuals": 40,
    "mixer_df_history": 320,
    "mixer_u_history": 320,
    "mixer_omega": 8,
    "published_shell_population": 4,
    "published_atomic_population": 4,
    "published_dipole": 12,
    "published_quadrupole": 24,
    "hamiltonian": 8,
    "eigenvalues": 4,
    "coefficients": 8,
    "occupations": 4,
    "density": 8,
    "weighted_density": 8,
    "raw_shell_population": 4,
    "raw_atomic_population": 4,
    "raw_dipole": 12,
    "raw_quadrupole": 24,
}


def _default_values() -> dict[tuple[str, str], tuple[int | float, ...]]:
    """Build a compact, schema-complete synthetic replay with matching outputs."""
    values = {("input", name): payload for name, (_dtype, payload) in INPUTS.items()}
    for prefix in CHECKER.CHECKPOINTS:
        if prefix.endswith("_pre"):
            base = 0.25 if prefix == "cpu_pre" else 0.5
            step = 19
        else:
            base = 100.0 if prefix.startswith("cpu_origin_") else 200.0
            step = 20
        for field, dtype in CHECKPOINT_SCHEMA:
            count = FIELD_COUNTS.get(field, 1)
            if dtype == "f64":
                if field == "mixer_residual_rms":
                    payload = (0.1,)
                elif field == "mixer_residual_maximum":
                    payload = (0.2,)
                elif prefix == "cpu_pre" and field == "scc_free_energy":
                    payload = (-1.19,)
                else:
                    payload = tuple(base + index / 1000 for index in range(count))
            elif dtype == "u64":
                payload = (
                    {
                        "mixer_iterations": step,
                        "mixer_restarts": 0,
                        "scc_iterations": step,
                    }[field],
                )
            elif dtype == "i32":
                payload = (0,)
            else:
                payload = (1 if field == "mixer_initialized" else 0,)
            values[(prefix, field)] = payload
        values[(prefix, "mixer_current_inputs")] = tuple(
            value
            for field in (
                "scc_shell_inputs",
                "scc_dipole_inputs",
                "scc_quadrupole_inputs",
            )
            for value in values[(prefix, field)]
        )
    return values


def _dtype_for(prefix: str, name: str) -> str:
    if prefix == "input":
        return dict(CHECKER.INPUT_SCHEMA)[name]
    return dict(CHECKPOINT_SCHEMA)[name]


def _write_snapshots(
    directory: Path, values: dict[tuple[str, str], tuple[int | float, ...]]
) -> None:
    """Write native-contract little-endian bytes for synthetic snapshots."""
    for (prefix, name), payload in values.items():
        dtype = _dtype_for(prefix, name)
        code, _width = CHECKER.DTYPES[dtype]
        (directory / f"{prefix}_{name}.bin").write_bytes(
            struct.pack(f"{code[0]}{len(payload)}{code[1:]}", *payload)
        )


def _snapshot_line(prefix: str, name: str, payload: tuple[int | float, ...]) -> str:
    return (
        f"SNAPSHOT {prefix} {_dtype_for(prefix, name)} {len(payload)} "
        f"{prefix}_{name}.bin"
    )


def _record(
    origin: str, field: str, values: dict[tuple[str, str], tuple[int | float, ...]]
) -> tuple[str, bool]:
    left = values[(f"{origin}_origin_cpu_post", field)]
    right = values[(f"{origin}_origin_cuda_post", field)]
    tolerance = CHECKER.REAL_TOLERANCES.get(field, 0.0)
    floating = field in CHECKER.REAL_TOLERANCES
    maximum, index, passed = CHECKER.compare_values(
        left, right, floating=floating, tolerance=tolerance
    )
    line = (
        f"REPLAY origin={origin} field={field} count={len(left)} max={maximum:.17g} "
        f"index={index} limit={tolerance:.17g} pass={int(passed)}"
    )
    return line, passed


def _stdout(
    directory: Path, values: dict[tuple[str, str], tuple[int | float, ...]]
) -> str:
    """Render all C++ ledger records in their production emission order."""
    lines = [
        f"TWO_ORIGIN directory={directory} transition=20 prefix_steps=19 "
        "expected_replays=4"
    ]
    input_order = [name for name, _dtype in CHECKER.INPUT_SCHEMA]
    lines.extend(
        _snapshot_line("input", name, values[("input", name)]) for name in input_order
    )
    temperature = 300.0 * 3.166808578545117e-6
    lines.append(
        "POLICY maximum_iterations=500 history=8 energy_tolerance=1e-10 "
        f"residual_tolerance=1e-08 electronic_temperature_hartree={temperature:.17g} "
        "spin_channels=2"
    )
    for transition in range(1, 20):
        energy = -1.0 - transition / 100
        lines.append(
            f"PREFIX transition={transition} cpu_iterations={transition} "
            f"cpu_energy={energy:.17g}"
        )
    values[("cpu_pre", "scc_free_energy")] = (-1.19,)
    lines.append(
        "PRESTATE actual_independent_cuda_origin=1 transition_inputs_distinct=1"
    )
    for checkpoint in CHECKER.CHECKPOINTS[:2]:
        lines.extend(
            _snapshot_line(checkpoint, name, values[(checkpoint, name)])
            for name, _dtype in CHECKPOINT_SCHEMA
        )
    for origin, cpu_prefix, cuda_prefix in (
        ("cpu", "cpu_origin_cpu_post", "cpu_origin_cuda_post"),
        ("cuda", "cuda_origin_cpu_post", "cuda_origin_cuda_post"),
    ):
        lines.append(f"RESTORED origin={origin} byte_exact=1")
        for checkpoint in (cpu_prefix, cuda_prefix):
            lines.extend(
                _snapshot_line(checkpoint, name, values[(checkpoint, name)])
                for name, _dtype in CHECKPOINT_SCHEMA
            )
        origin_pass = True
        for field, tolerance in CHECKER.REAL_TOLERANCES.items():
            if tolerance >= 0:
                line, passed = _record(origin, field, values)
                lines.append(line)
                origin_pass = origin_pass and passed
        lines.append(
            f"GAUGE origin={origin} spectrum_limit=1e-12 subspace_limit=3e-08 pass=1"
        )
        for field in (
            *CHECKER.COUNTER_FIELDS,
            *CHECKER.STATUS_FIELDS,
            *CHECKER.FLAG_FIELDS,
        ):
            line, passed = _record(origin, field, values)
            lines.append(line)
            origin_pass = origin_pass and passed
        lines.append(f"ORIGIN origin={origin} replays=2 pass={int(origin_pass)}")
    passed = all(
        record.endswith("pass=1") for record in lines if record.startswith("ORIGIN ")
    )
    lines.append(
        f"TWO_ORIGIN completed_replays=4 diagnostic_pass={int(passed)} "
        "public_parity_unresolved=1"
    )
    return "\n".join(lines) + "\n"


def _fixture(
    directory: Path,
    mutate: Callable[[dict[tuple[str, str], tuple[int | float, ...]]], None]
    | None = None,
) -> tuple[str, dict[tuple[str, str], tuple[int | float, ...]]]:
    values = _default_values()
    if mutate is not None:
        mutate(values)
    _write_snapshots(directory, values)
    return _stdout(directory, values), values


class SpinTwoOriginReplayTest(unittest.TestCase):
    """Prove exit outcomes and reject incomplete or contradictory evidence."""

    def _check(
        self,
        mutate: Callable[[dict[tuple[str, str], tuple[int | float, ...]]], None]
        | None = None,
    ) -> tuple[Path, str, dict[tuple[str, str], tuple[int | float, ...]]]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        stdout, values = _fixture(directory, mutate)
        return directory, stdout, values

    def test_complete_matching_replay_passes_without_resolving_public_parity(
        self,
    ) -> None:
        """Accept all 294 snapshots and retain the unresolved parity label."""
        directory, stdout, _values = self._check()
        result = CHECKER.validate_evidence(stdout, directory)
        self.assertTrue(result.passed)
        self.assertEqual(
            sum(line.startswith("SNAPSHOT ") for line in stdout.splitlines()), 294
        )
        self.assertEqual(
            sum(line.startswith("PREFIX ") for line in stdout.splitlines()), 19
        )
        self.assertEqual(
            result.final_line,
            "TWO_ORIGIN completed_replays=4 diagnostic_pass=1 "
            "public_parity_unresolved=1",
        )

    def test_numerical_failure_retains_both_origin_failures(self) -> None:
        """Keep CPU- and CUDA-origin numerical failures independently visible."""

        def perturb(values: dict[tuple[str, str], tuple[int | float, ...]]) -> None:
            for origin in ("cpu", "cuda"):
                key = (f"{origin}_origin_cpu_post", "density")
                values[key] = (values[key][0] + 1e-4, *values[key][1:])

        directory, stdout, _values = self._check(perturb)
        result = CHECKER.validate_evidence(stdout, directory)
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_origins, ("cpu", "cuda"))
        self.assertIn("cpu/density", result.failures)
        self.assertIn("cuda/density", result.failures)
        self.assertTrue(
            result.final_line.endswith("diagnostic_pass=0 public_parity_unresolved=1")
        )

    def test_terminal_publication_may_differ_from_private_scc_inputs(self) -> None:
        """Compare terminal fields independently without conflating their sources."""

        def terminal_publication(
            values: dict[tuple[str, str], tuple[int | float, ...]],
        ) -> None:
            for prefix in CHECKER.CHECKPOINTS[2:]:
                for inputs in (
                    "scc_shell_inputs",
                    "scc_dipole_inputs",
                    "scc_quadrupole_inputs",
                ):
                    values[(prefix, inputs)] = tuple(
                        value + 0.25 for value in values[(prefix, inputs)]
                    )
                values[(prefix, "mixer_current_inputs")] = tuple(
                    value
                    for field in (
                        "scc_shell_inputs",
                        "scc_dipole_inputs",
                        "scc_quadrupole_inputs",
                    )
                    for value in values[(prefix, field)]
                )
                values[(prefix, "mixer_residual_rms")] = (1e-12,)
                values[(prefix, "mixer_residual_maximum")] = (1e-9,)
                values[(prefix, "mixer_residual_converged")] = (1,)
                values[(prefix, "scc_terminal_converged")] = (1,)
                values[(prefix, "cpu_mixer_terminal_converged")] = (1,)

        directory, stdout, values = self._check(terminal_publication)
        for prefix in CHECKER.CHECKPOINTS[2:]:
            self.assertNotEqual(
                values[(prefix, "scc_shell_inputs")],
                values[(prefix, "published_shell_population")],
            )
        self.assertTrue(CHECKER.validate_evidence(stdout, directory).passed)

    def test_residual_flag_uses_the_frozen_rms_and_maximum_limits(self) -> None:
        """Accept residual convergence when both values are below 1e-8."""

        def converged_residuals(
            values: dict[tuple[str, str], tuple[int | float, ...]],
        ) -> None:
            for prefix in CHECKER.CHECKPOINTS:
                values[(prefix, "mixer_residual_rms")] = (5e-9,)
                values[(prefix, "mixer_residual_maximum")] = (5e-9,)
                values[(prefix, "mixer_residual_converged")] = (1,)

        directory, stdout, _values = self._check(converged_residuals)
        self.assertTrue(CHECKER.validate_evidence(stdout, directory).passed)

    def test_precheck_failure_with_only_saved_origins_is_malformed(self) -> None:
        """Do not turn saved prestate-only evidence into a replay pass."""
        directory, stdout, _values = self._check()
        retained = [
            line
            for line in stdout.splitlines()
            if line.startswith(
                (
                    "TWO_ORIGIN directory=",
                    "SNAPSHOT input ",
                    "POLICY ",
                    "PREFIX ",
                    "PRESTATE ",
                    "SNAPSHOT cpu_pre ",
                    "SNAPSHOT cuda_pre ",
                )
            )
        ]
        with self.assertRaises(CHECKER.EvidenceError):
            CHECKER.validate_evidence("\n".join(retained), directory)

    def test_active_prestate_inputs_match_published_population(self) -> None:
        """Reject active prestate q/d/Q that differ from its publication."""

        def mismatched_publication(
            values: dict[tuple[str, str], tuple[int | float, ...]],
        ) -> None:
            values[("cpu_pre", "scc_shell_inputs")] = (9.0, 9.0, 9.0, 9.0)
            values[("cpu_pre", "mixer_current_inputs")] = tuple(
                value
                for field in (
                    "scc_shell_inputs",
                    "scc_dipole_inputs",
                    "scc_quadrupole_inputs",
                )
                for value in values[("cpu_pre", field)]
            )

        directory, stdout, _values = self._check(mismatched_publication)
        with self.assertRaisesRegex(CHECKER.EvidenceError, "active SCC inputs differ"):
            CHECKER.validate_evidence(stdout, directory)

    def test_every_checkpoint_mixer_current_matches_concatenated_scc_inputs(
        self,
    ) -> None:
        """Reject split mixer vectors inconsistent with their checkpoint q/d/Q."""
        for prefix in CHECKER.CHECKPOINTS:
            with self.subTest(prefix=prefix):

                def mismatched_mixer_current(
                    values: dict[tuple[str, str], tuple[int | float, ...]],
                    checkpoint: str = prefix,
                ) -> None:
                    current = values[(checkpoint, "mixer_current_inputs")]
                    values[(checkpoint, "mixer_current_inputs")] = (
                        current[0] + 1.0,
                        *current[1:],
                    )

                directory, stdout, _values = self._check(mismatched_mixer_current)
                with self.assertRaisesRegex(
                    CHECKER.EvidenceError, "mixer_current_inputs does not match"
                ):
                    CHECKER.validate_evidence(stdout, directory)

    def test_non_distinct_prestate_origins_are_malformed(self) -> None:
        """Reject copied q/d/Q and mixer inputs presented as two origins."""

        def copy_transition_inputs(
            values: dict[tuple[str, str], tuple[int | float, ...]],
        ) -> None:
            for field in tuple(CHECKER.REAL_TOLERANCES)[:9]:
                values[("cuda_pre", field)] = values[("cpu_pre", field)]
            for field in (
                "published_shell_population",
                "published_dipole",
                "published_quadrupole",
            ):
                values[("cuda_pre", field)] = values[("cpu_pre", field)]

        directory, stdout, _values = self._check(copy_transition_inputs)
        with self.assertRaisesRegex(CHECKER.EvidenceError, "not distinct"):
            CHECKER.validate_evidence(stdout, directory)

    def test_missing_prefix_record_is_malformed(self) -> None:
        """Require every transition in the fixed nineteen-step prefix."""
        directory, stdout, _values = self._check()
        prefix = next(
            line
            for line in stdout.splitlines()
            if line.startswith("PREFIX transition=7 ")
        )
        with self.assertRaisesRegex(CHECKER.EvidenceError, "19 ordered PREFIX"):
            CHECKER.validate_evidence(stdout.replace(prefix + "\n", "", 1), directory)

    def test_truncated_snapshot_is_malformed(self) -> None:
        """Reject bytes that stop before the ledger-declared element count."""
        directory, stdout, _values = self._check()
        path = directory / "cpu_pre_hamiltonian.bin"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(CHECKER.EvidenceError, "truncated or oversized"):
            CHECKER.validate_evidence(stdout, directory)

    def test_nonfinite_snapshot_value_is_malformed(self) -> None:
        """Reject non-finite binary64 inputs and checkpoint values."""
        directory, stdout, _values = self._check()
        (directory / "input_positions.bin").write_bytes(
            struct.pack("<6d", math.nan, 1, 2, 3, 4, 5)
        )
        with self.assertRaisesRegex(CHECKER.EvidenceError, "non-finite"):
            CHECKER.validate_evidence(stdout, directory)

    def test_missing_and_duplicate_snapshot_ledger_entries_are_malformed(self) -> None:
        """Reject absent and duplicate records in the stdout snapshot ledger."""
        directory, stdout, _values = self._check()
        snapshot = next(
            line for line in stdout.splitlines() if line.startswith("SNAPSHOT cpu_pre ")
        )
        with self.assertRaises(CHECKER.EvidenceError):
            CHECKER.validate_evidence(stdout.replace(snapshot + "\n", "", 1), directory)
        with self.assertRaises(CHECKER.EvidenceError):
            CHECKER.validate_evidence(
                stdout.replace(snapshot + "\n", snapshot + "\n" + snapshot + "\n", 1),
                directory,
            )

    def test_wrong_count_type_and_limit_are_malformed(self) -> None:
        """Enforce scalar counts, exact dtypes, and frozen replay tolerances."""
        directory, stdout, _values = self._check(
            lambda values: values.__setitem__(
                ("cpu_pre", "mixer_residual_rms"), (0.1, 0.2)
            )
        )
        with self.assertRaisesRegex(CHECKER.EvidenceError, "must be scalar"):
            CHECKER.validate_evidence(stdout, directory)
        directory, stdout, _values = self._check()
        line = "SNAPSHOT cpu_pre f64 4 cpu_pre_scc_shell_inputs.bin"
        with self.assertRaises(CHECKER.EvidenceError):
            CHECKER.validate_evidence(
                stdout.replace(line, line.replace(" f64 ", " u64 ")), directory
            )
        directory, stdout, _values = self._check()
        line = next(
            line
            for line in stdout.splitlines()
            if line.startswith("REPLAY origin=cpu field=density ")
        )
        with self.assertRaisesRegex(CHECKER.EvidenceError, "limit mismatch"):
            CHECKER.validate_evidence(
                stdout.replace(line, line.replace("limit=1e-10", "limit=1e-09")),
                directory,
            )

    def test_integer_mismatch_above_binary64_exact_range_is_detected(self) -> None:
        """Use exact integer equality even when float conversion hides a unit."""
        left = (2**53,)
        right = (2**53 + 1,)
        maximum, index, passed = CHECKER.compare_values(
            left, right, floating=False, tolerance=0.0
        )
        self.assertEqual((maximum, index), (0.0, 0))
        self.assertFalse(passed)

    def test_unsafe_snapshot_path_is_malformed(self) -> None:
        """Reject traversal paths before opening a snapshot file."""
        directory, stdout, _values = self._check()
        line = "SNAPSHOT cpu_pre f64 4 cpu_pre_scc_shell_inputs.bin"
        unsafe = line.replace(
            "cpu_pre_scc_shell_inputs.bin", "../cpu_pre_scc_shell_inputs.bin"
        )
        with self.assertRaisesRegex(CHECKER.EvidenceError, "unsafe snapshot path"):
            CHECKER.validate_evidence(stdout.replace(line, unsafe), directory)

    def test_extra_snapshot_file_is_malformed(self) -> None:
        """Require the snapshot directory to contain exactly ledger files."""
        directory, stdout, _values = self._check()
        (directory / "unexpected.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(CHECKER.EvidenceError, "missing or extra files"):
            CHECKER.validate_evidence(stdout, directory)

    def test_incomplete_coordinate_array_is_malformed(self) -> None:
        """Require three saved coordinates for every input atom."""

        def truncate_coordinate(
            values: dict[tuple[str, str], tuple[int | float, ...]],
        ) -> None:
            values[("input", "positions")] = (0.0, 1.0, 2.0, 3.0, 4.0)

        directory, stdout, _values = self._check(truncate_coordinate)
        with self.assertRaisesRegex(
            CHECKER.EvidenceError, "three coordinates per atom"
        ):
            CHECKER.validate_evidence(stdout, directory)

    def test_invalid_prestate_counter_status_and_flag_are_malformed(self) -> None:
        """Reject impossible active-state counters, statuses, and flag bytes."""
        mutations = (
            ("scc_iterations", (18,)),
            ("mixer_status", (6,)),
            ("mixer_initialized", (2,)),
        )
        for field, payload in mutations:
            with self.subTest(field=field):
                directory, stdout, _values = self._check(
                    lambda values, field=field, payload=payload: values.__setitem__(
                        ("cpu_pre", field), payload
                    )
                )
                with self.assertRaises(CHECKER.EvidenceError):
                    CHECKER.validate_evidence(stdout, directory)

    def test_cli_exit_codes_distinguish_pass_fail_and_malformed(self) -> None:
        """Map complete PASS, complete FAIL, and malformed logs to 0, 1, 2."""
        scenarios = (
            (None, False, 0),
            (
                lambda values: values.__setitem__(
                    ("cpu_origin_cpu_post", "density"),
                    (100.0001, *values[("cpu_origin_cpu_post", "density")][1:]),
                ),
                False,
                1,
            ),
            (None, True, 2),
        )
        for mutate, malformed, expected_code in scenarios:
            with self.subTest(expected_code=expected_code):
                directory, stdout, _values = self._check(mutate)
                if malformed:
                    stdout = stdout.rsplit("TWO_ORIGIN completed_replays=", maxsplit=1)[
                        0
                    ]
                log_path = directory.parent / f"{directory.name}.stdout"
                self.addCleanup(lambda path=log_path: path.unlink(missing_ok=True))
                log_path.write_text(stdout, encoding="utf-8")
                captured_out = io.StringIO()
                captured_err = io.StringIO()
                with redirect_stdout(captured_out), redirect_stderr(captured_err):
                    code = CHECKER.main([str(log_path), str(directory)])
                self.assertEqual(code, expected_code)
                if expected_code == 1:
                    self.assertIn("FAIL cpu/density", captured_out.getvalue())
                if expected_code == 2:
                    self.assertIn("MALFORMED evidence:", captured_err.getvalue())


if __name__ == "__main__":
    unittest.main()
