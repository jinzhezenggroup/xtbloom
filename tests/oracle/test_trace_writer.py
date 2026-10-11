"""Offline tests for the canonical ``xtbloom-scc-trace-v1`` writer."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover - runtime writer remains standard-library only.
    jsonschema = None

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace" / "xtbloom_scc_trace.py"
)
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "oracle"
    / "tblite_scc_trace"
    / "xtbloom-scc-trace-v1.schema.json"
)
SCHEMA_V2_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "oracle"
    / "tblite_scc_trace"
    / "xtbloom-scc-trace-v2.schema.json"
)
SPEC = importlib.util.spec_from_file_location("xtbloom_scc_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("xtbloom_scc_trace", TRACE)
SPEC.loader.exec_module(TRACE)

REVISION = "e9abc395b122018ed688aecb1c3a65cecaf97beb"
PATCH_SHA256 = "d6a51afc4b3c56d6589a2b5b115ea8b4891600c1161c525939ca3cc16e2b4954"


def _iteration(index: int = 1, *, converged: bool = True) -> dict:
    """Build one completed two-atom restricted observer iteration."""
    mixed_qsh = [[0.10, -0.20, 0.30]]
    raw_qsh = [[0.12, -0.19, 0.27]]
    mixed_dipoles = [[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]]
    raw_dipoles = [[[0.015, 0.018, 0.035], [0.03, 0.07, 0.055]]]
    mixed_quadrupoles = [
        [
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
        ]
    ]
    raw_quadrupoles = [
        [
            [0.02, 0.01, 0.05, 0.03, 0.08, 0.04],
            [0.06, 0.10, 0.08, 0.13, 0.09, 0.15],
        ]
    ]
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
    residual = [
        raw_value - mixed_value
        for raw_value, mixed_value in zip(raw, mixed, strict=True)
    ]
    residual_rms = math.sqrt(sum(value * value / len(residual) for value in residual))
    return {
        "index": index,
        "hamiltonian": [[[-0.7, 0.3], [0.3, -0.2]]],
        "eigenvalues": [[-0.9, -0.3]],
        # Restricted tblite still exposes focc[nao,2].
        "occupations": [[1.0, 0.0], [1.0, 0.0]],
        "density": [[[1.0, 0.05], [0.05, 1.0]]],
        "mixed_qsh": mixed_qsh,
        "raw_qsh": raw_qsh,
        "mixed_qat": [[0.10, 0.10]],
        "raw_qat": [[0.12, 0.08]],
        "mixed_dipoles": mixed_dipoles,
        "raw_dipoles": raw_dipoles,
        "mixed_quadrupoles": mixed_quadrupoles,
        "raw_quadrupoles": raw_quadrupoles,
        "residual": residual,
        "residual_rms": residual_rms,
        "energy": -1.25,
        "energy_delta": -0.02,
        "convergence": {
            "energy": converged,
            "population": converged,
            "temperature": True,
            "overall": converged,
        },
    }


def _pre_solve_attempt(iteration: dict, index: int) -> dict:
    """Extract exactly the payload available before an eigensolver failure."""
    fields = (
        "hamiltonian",
        "mixed_qsh",
        "mixed_qat",
        "mixed_dipoles",
        "mixed_quadrupoles",
    )
    return {
        "index": index,
        **{field: copy.deepcopy(iteration[field]) for field in fields},
    }


def _fixture() -> dict:
    """Build a schema-valid one-iteration restricted trace."""
    return {
        "format": TRACE.FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": PATCH_SHA256,
        },
        "input": {
            "atomic_numbers": [1, 8],
            "positions": [0.0, 0.0, 0.0, 0.0, 0.0, 1.8],
            "molecular_charge": 0.0,
            "unpaired_electrons": 0,
            "spin_channels": 1,
            "temperature": 300.0,
        },
        "basis": {
            "n_atoms": 2,
            "n_shells": 3,
            "nao": 2,
            "atom_to_shell_count": [1, 2],
            "shell_offsets": [0, 1],
        },
        "statics": {
            "overlap": [[[1.0, 0.1], [0.1, 1.0]]],
            "core_hamiltonian": [[[-0.8, 0.2], [0.2, -0.4]]],
        },
        "residual_layout": {
            "shell_charges": 3,
            "atomic_dipoles": 6,
            "atomic_quadrupoles": 12,
        },
        "iterations": [_iteration()],
        "terminal": {
            "status": TRACE.STATUS_CONVERGED,
            "converged": True,
            "iterations": 1,
        },
    }


def _unrestricted_fixture() -> dict:
    """Build a schema-valid v2 trace with distinct population/orbital channels."""
    fixture = _fixture()
    entry = fixture["iterations"][0]
    assembled = [
        [[-0.35, 0.15], [0.15, -0.10]],
        [[-0.32, 0.12], [0.12, -0.08]],
    ]
    entry.pop("hamiltonian")
    entry["assembled_hamiltonian"] = assembled
    entry["solver_hamiltonian"] = [
        [[2.0 * value for value in row] for row in channel] for channel in assembled
    ]
    entry["eigenvalues"] = [[-0.9, -0.3], [-0.8, -0.2]]
    entry["density"] = [entry["density"][0], [[0.8, 0.04], [0.04, 0.1]]]
    entry["mixed_qsh"] = [entry["mixed_qsh"][0], [0.03, 0.02, -0.01]]
    entry["raw_qsh"] = [entry["raw_qsh"][0], [0.02, 0.04, -0.02]]
    entry["mixed_qat"] = [entry["mixed_qat"][0], [0.03, 0.01]]
    entry["raw_qat"] = [entry["raw_qat"][0], [0.02, 0.02]]
    for field in ("mixed_dipoles", "raw_dipoles"):
        entry[field] = [entry[field][0], copy.deepcopy(entry[field][0])]
    for field in ("mixed_quadrupoles", "raw_quadrupoles"):
        entry[field] = [entry[field][0], copy.deepcopy(entry[field][0])]
    mixed = TRACE._flatten_population(entry, "mixed", 2)
    raw = TRACE._flatten_population(entry, "raw", 2)
    entry["residual"] = [
        raw_value - mixed_value
        for raw_value, mixed_value in zip(raw, mixed, strict=True)
    ]
    entry["residual_rms"] = math.sqrt(
        sum(value * value / len(entry["residual"]) for value in entry["residual"])
    )
    fixture["format"] = TRACE.FORMAT_V2
    fixture["input"]["unpaired_electrons"] = 1
    fixture["input"]["spin_channels"] = 2
    fixture["residual_layout"] = {
        "shell_charges": 6,
        "atomic_dipoles": 12,
        "atomic_quadrupoles": 24,
    }
    return fixture


def _setup_failure() -> dict:
    fixture = _fixture()
    fixture["iterations"] = []
    fixture["terminal"] = {
        "status": TRACE.STATUS_FAILED,
        "converged": False,
        "iterations": 0,
    }
    return fixture


def _eigensolver_failure() -> dict:
    fixture = _fixture()
    fixture["failed_attempt"] = _pre_solve_attempt(fixture["iterations"][0], 1)
    fixture["iterations"] = []
    fixture["terminal"] = {
        "status": TRACE.STATUS_FAILED,
        "converged": False,
        "iterations": 1,
    }
    return fixture


def _unrestricted_eigensolver_failure() -> dict:
    """Build a v2 failed-attempt fixture with only pre-solve state."""
    fixture = _unrestricted_fixture()
    iteration = fixture["iterations"][0]
    fields = (
        "assembled_hamiltonian",
        "solver_hamiltonian",
        "mixed_qsh",
        "mixed_qat",
        "mixed_dipoles",
        "mixed_quadrupoles",
    )
    fixture["failed_attempt"] = {
        "index": 1,
        **{field: copy.deepcopy(iteration[field]) for field in fields},
    }
    fixture["iterations"] = []
    fixture["terminal"] = {
        "status": TRACE.STATUS_FAILED,
        "converged": False,
        "iterations": 1,
    }
    return fixture


class TraceWriterTest(unittest.TestCase):
    """Exercise canonical emission, schema alignment, and observer semantics."""

    def test_fixture_records_restricted_channels_and_nested_multipoles(self) -> None:
        """Retain restricted channels and nested multipoles in the fixture."""
        loaded = json.loads(TRACE.dumps(_fixture()))
        iteration = loaded["iterations"][0]
        self.assertEqual(len(iteration["occupations"]), 2)
        self.assertEqual(len(iteration["mixed_dipoles"][0]), 2)
        self.assertEqual(len(iteration["mixed_dipoles"][0][0]), 3)
        self.assertEqual(len(iteration["mixed_quadrupoles"][0][1]), 6)

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_all_writer_lifecycle_outputs_validate_against_schema(self) -> None:
        """Validate every writer lifecycle fixture against the schema."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft7Validator.check_schema(schema)
        validator = jsonschema.Draft7Validator(schema)
        point_charge = _fixture()
        point_charge["input"]["point_charges"] = {
            "positions": [0.0, 0.0, 4.0],
            "charges": [-0.5],
            "hardnesses": [999.0],
        }
        for fixture in (
            _fixture(),
            self._restricted_open_shell_fixture(),
            point_charge,
            _setup_failure(),
            _eigensolver_failure(),
        ):
            with self.subTest(status=fixture["terminal"]["status"]):
                validator.validate(json.loads(TRACE.dumps(fixture)))

    def test_roundtrip_is_byte_identical(self) -> None:
        """Round-trip canonical trace bytes without changes."""
        text = TRACE.dumps(_fixture())
        self.assertEqual(TRACE.dumps(json.loads(text)), text)

    def test_v2_roundtrip_preserves_two_distinct_representations(self) -> None:
        """Keep charge/magnetization separate from alpha/beta orbital fields."""
        fixture = _unrestricted_fixture()
        text = TRACE.dumps(fixture)
        loaded = json.loads(text)
        self.assertEqual(loaded["input"]["spin_channels"], 2)
        self.assertEqual(len(loaded["iterations"][0]["mixed_qsh"]), 2)
        self.assertEqual(len(loaded["iterations"][0]["eigenvalues"]), 2)
        self.assertEqual(TRACE.dumps(loaded), text)

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_v2_writer_output_validates_against_v2_schema(self) -> None:
        """Keep runtime and Draft 7 unrestricted contracts synchronized."""
        schema = json.loads(SCHEMA_V2_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft7Validator.check_schema(schema)
        jsonschema.Draft7Validator(schema).validate(
            json.loads(TRACE.dumps(_unrestricted_fixture()))
        )

    def test_v2_rejects_mixed_channel_reduction_and_hamiltonian_scaling(self) -> None:
        """Reject representation drift before comparison or replay."""
        fixture = _unrestricted_fixture()
        fixture["iterations"][0]["mixed_qat"][1][0] += 1.0
        with self.assertRaisesRegex(TRACE.TraceError, "reduced mixed_qsh"):
            TRACE.validate(fixture)
        fixture = _unrestricted_fixture()
        fixture["iterations"][0]["solver_hamiltonian"][1][0][0] += 1.0e-6
        with self.assertRaisesRegex(TRACE.TraceError, "twice assembled_hamiltonian"):
            TRACE.validate(fixture)

    def test_v2_failed_attempt_preserves_only_spin_resolved_pre_solve_state(
        self,
    ) -> None:
        """Keep both Hamiltonians and mixed q/d/Q, but no post-solve payload."""
        fixture = _unrestricted_eigensolver_failure()
        attempt = fixture["failed_attempt"]
        self.assertEqual(len(attempt["assembled_hamiltonian"]), 2)
        self.assertEqual(len(attempt["mixed_qsh"]), 2)
        for field in ("density", "eigenvalues", "occupations", "raw_qsh", "residual"):
            self.assertNotIn(field, attempt)
        TRACE.validate(fixture)

    def test_v2_rejects_legacy_or_missing_hamiltonian_fields(self) -> None:
        """Require the explicit assembled/solver distinction in every v2 attempt."""
        fixture = _unrestricted_fixture()
        fixture["iterations"][0]["hamiltonian"] = copy.deepcopy(
            fixture["iterations"][0]["assembled_hamiltonian"]
        )
        with self.assertRaisesRegex(TRACE.TraceError, "unsupported field"):
            TRACE.validate(fixture)
        for field in ("assembled_hamiltonian", "solver_hamiltonian"):
            fixture = _unrestricted_fixture()
            del fixture["iterations"][0][field]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(TRACE.TraceError, "missing required field"),
            ):
                TRACE.validate(fixture)

    def test_v2_rejects_orbital_and_population_channel_shape_drift(self) -> None:
        """Enforce alpha/beta orbitals and charge/magnetization q/d/Q shapes."""
        mutations = {
            "eigenvalues": lambda entry: entry["eigenvalues"].pop(),
            "occupations": lambda entry: entry["occupations"].pop(),
            "density": lambda entry: entry["density"].pop(),
            "qsh": lambda entry: entry["raw_qsh"].pop(),
            "dipoles": lambda entry: entry["raw_dipoles"][1].pop(),
            "quadrupoles": lambda entry: entry["raw_quadrupoles"][1][0].pop(),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                fixture = _unrestricted_fixture()
                mutation(fixture["iterations"][0])
                with self.assertRaises(TRACE.TraceError):
                    TRACE.validate(fixture)

    def test_v2_residual_uses_both_population_channels(self) -> None:
        """Reject second-channel corruption and a one-channel residual layout."""
        fixture = _unrestricted_fixture()
        fixture["iterations"][0]["residual"][-1] += 1.0e-6
        with self.assertRaisesRegex(TRACE.TraceError, "raw q/d/Q minus mixed"):
            TRACE.validate(fixture)
        fixture = _unrestricted_fixture()
        fixture["residual_layout"]["shell_charges"] = 3
        with self.assertRaisesRegex(TRACE.TraceError, "must be 6"):
            TRACE.validate(fixture)

    def test_unicode_and_extensible_provenance_roundtrip(self) -> None:
        """Preserve Unicode and extensible provenance through serialization."""
        fixture = _fixture()
        fixture["provenance"]["oracle_command"] = "trace --molecule H₃⁺"
        fixture["provenance"]["toolchain"] = {"compiler": "gfortran", "flags": []}
        text = TRACE.dumps(fixture)
        self.assertIn("H₃⁺", text)
        self.assertEqual(TRACE.dumps(json.loads(text)), text)

    def test_floats_keep_17_significant_digits(self) -> None:
        """Serialize binary64 values with 17 significant digits."""
        text = TRACE.dumps(_fixture())
        self.assertIn("0.10000000000000001", text)
        fixture = _fixture()
        fixture["input"]["molecular_charge"] = 2.0
        text = TRACE.dumps(fixture)
        self.assertIn('"molecular_charge": 2.0', text)
        self.assertIsInstance(json.loads(text)["input"]["molecular_charge"], float)

    def test_quadrupole_and_residual_layout_is_independent_of_fortran(self) -> None:
        """Validate quadrupole and residual layouts independently of Fortran."""
        fixture = _fixture()
        iteration = fixture["iterations"][0]
        mixed = (
            iteration["mixed_qsh"][0]
            + [value for atom in iteration["mixed_dipoles"][0] for value in atom]
            + [value for atom in iteration["mixed_quadrupoles"][0] for value in atom]
        )
        raw = (
            iteration["raw_qsh"][0]
            + [value for atom in iteration["raw_dipoles"][0] for value in atom]
            + [value for atom in iteration["raw_quadrupoles"][0] for value in atom]
        )
        self.assertEqual(
            TRACE.QUADRUPOLE_COMPONENTS, ("xx", "xy", "yy", "xz", "yz", "zz")
        )
        self.assertEqual(
            iteration["residual"],
            [
                raw_value - mixed_value
                for raw_value, mixed_value in zip(raw, mixed, strict=True)
            ],
        )
        self.assertEqual(len(iteration["residual"]), 3 + 2 * 3 + 2 * 6)
        TRACE.validate(fixture)

    def test_multiple_completed_iterations(self) -> None:
        """Accept a lifecycle with multiple completed iterations."""
        fixture = _fixture()
        fixture["iterations"] = [_iteration(converged=False), _iteration(index=2)]
        fixture["terminal"]["iterations"] = 2
        TRACE.validate(fixture)

    def test_setup_failure_has_zero_iterations(self) -> None:
        """Require setup failures to contain zero completed iterations."""
        TRACE.validate(_setup_failure())

    def test_eigensolver_failure_preserves_only_pre_solve_state(self) -> None:
        """Retain only pre-solve state after an eigensolver failure."""
        fixture = _eigensolver_failure()
        self.assertNotIn("density", fixture["failed_attempt"])
        self.assertNotIn("raw_qsh", fixture["failed_attempt"])
        TRACE.validate(fixture)

    def test_later_mixer_failure_has_no_failed_attempt(self) -> None:
        """Omit failed-attempt payloads for later mixer failures."""
        fixture = _fixture()
        fixture["iterations"] = [_iteration(converged=False)]
        fixture["terminal"] = {
            "status": TRACE.STATUS_FAILED,
            "converged": False,
            "iterations": 1,
        }
        TRACE.validate(fixture)

    def test_max_iterations_uses_terminal_status_only(self) -> None:
        """Represent an iteration limit through terminal status only."""
        fixture = _fixture()
        fixture["iterations"] = [_iteration(converged=False)]
        fixture["terminal"] = {
            "status": TRACE.STATUS_MAX_ITERATIONS,
            "converged": False,
            "iterations": 1,
        }
        self.assertNotIn("status", fixture["iterations"][0])
        TRACE.validate(fixture)

    def test_rejects_unsupported_format_version(self) -> None:
        """Reject unsupported trace format versions."""
        fixture = _fixture()
        fixture["format"] = "xtbloom-scc-trace-v0"
        with self.assertRaisesRegex(TRACE.TraceError, "unsupported trace format"):
            TRACE.validate(fixture)

    @staticmethod
    def _restricted_open_shell_fixture() -> dict:
        """Build a shared-orbital open-shell trace with one spin channel."""
        fixture = _fixture()
        fixture["input"]["unpaired_electrons"] = 1
        return fixture

    def test_accepts_restricted_open_shell_inputs(self) -> None:
        """Accept restricted open-shell inputs represented by the v1 schema."""
        fixture = self._restricted_open_shell_fixture()
        TRACE.validate(fixture)

    def test_rejects_unrestricted_v1_inputs(self) -> None:
        """Reject unrestricted inputs unsupported by the v1 schema."""
        fixture = _fixture()
        fixture["input"]["spin_channels"] = 2
        with self.assertRaisesRegex(TRACE.TraceError, "restricted-only"):
            TRACE.validate(fixture)

    def test_rejects_negative_unpaired_electron_count(self) -> None:
        """Reject negative unpaired-electron counts."""
        fixture = _fixture()
        fixture["input"]["unpaired_electrons"] = -1
        with self.assertRaisesRegex(TRACE.TraceError, "must be nonnegative"):
            TRACE.validate(fixture)

    def test_rejects_malformed_molecular_input(self) -> None:
        """Reject malformed molecular descriptors."""
        fixture = _fixture()
        fixture["input"]["positions"] = [0.0, 0.0]
        with self.assertRaisesRegex(TRACE.TraceError, "positions must hold"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["input"]["atomic_numbers"][0] = 0
        with self.assertRaisesRegex(TRACE.TraceError, "between 1 and 118"):
            TRACE.validate(fixture)
        fixture = _fixture()
        del fixture["input"]["temperature"]
        with self.assertRaisesRegex(TRACE.TraceError, "temperature"):
            TRACE.validate(fixture)

    def test_point_charge_dimensions_values_and_hardness(self) -> None:
        """Validate point-charge dimensions, values, and hardness data."""
        fixture = _fixture()
        fixture["input"]["point_charges"] = {
            "positions": [0.0, 0.0, 4.0],
            "charges": [-0.5],
            "hardnesses": [999.0],
        }
        TRACE.validate(fixture)
        fixture["input"]["point_charges"]["charges"].append(0.2)
        with self.assertRaisesRegex(TRACE.TraceError, "3 values per point charge"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["input"]["point_charges"] = {
            "positions": [0.0, 0.0, 4.0],
            "charges": [-0.5],
            "hardnesses": [0.0],
        }
        with self.assertRaisesRegex(TRACE.TraceError, "must be positive"):
            TRACE.validate(fixture)

    def test_rejects_missing_or_inconsistent_basis_mapping(self) -> None:
        """Reject missing or inconsistent atom-to-basis mappings."""
        fixture = _fixture()
        del fixture["basis"]["atom_to_shell_count"]
        with self.assertRaisesRegex(TRACE.TraceError, "atom_to_shell_count"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["basis"]["atom_to_shell_count"] = [1, 1]
        with self.assertRaisesRegex(TRACE.TraceError, "must equal basis.n_shells"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["basis"]["shell_offsets"] = [0, 2]
        with self.assertRaisesRegex(TRACE.TraceError, "must be 1"):
            TRACE.validate(fixture)

    def test_rejects_wrong_matrix_multipole_and_occupation_shapes(self) -> None:
        """Reject invalid matrix, multipole, and occupation shapes."""
        fixture = _fixture()
        fixture["statics"]["overlap"][0][0].append(0.0)
        with self.assertRaisesRegex(TRACE.TraceError, "columns"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"][0]["mixed_dipoles"][0][0].append(0.0)
        with self.assertRaisesRegex(TRACE.TraceError, "3 components"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"][0]["occupations"] = [[1.0, 0.0]]
        with self.assertRaisesRegex(TRACE.TraceError, "needs 2 channels"):
            TRACE.validate(fixture)

    def test_rejects_wrong_residual_layout_value_and_rms(self) -> None:
        """Reject invalid residual layouts, values, and RMS summaries."""
        fixture = _fixture()
        fixture["residual_layout"]["atomic_dipoles"] = 5
        with self.assertRaisesRegex(TRACE.TraceError, "must be 6"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"][0]["residual"][0] = 99.0
        with self.assertRaisesRegex(TRACE.TraceError, "raw q/d/Q minus mixed"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"][0]["residual_rms"] = 1.0
        with self.assertRaisesRegex(TRACE.TraceError, "unweighted numerical RMS"):
            TRACE.validate(fixture)

    def test_rejects_qat_inconsistent_with_qsh_reduction(self) -> None:
        """Reject atomic charges inconsistent with shell-charge reduction."""
        fixture = _fixture()
        fixture["iterations"][0]["raw_qat"][0][1] = 7.0
        with self.assertRaisesRegex(TRACE.TraceError, "reduced raw_qsh"):
            TRACE.validate(fixture)

    def test_rejects_inconsistent_convergence_flags(self) -> None:
        """Reject mutually inconsistent convergence flags."""
        fixture = _fixture()
        fixture["iterations"][0]["convergence"]["energy"] = False
        with self.assertRaisesRegex(TRACE.TraceError, "must be the conjunction"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"] = [_iteration(), _iteration(index=2)]
        fixture["terminal"]["iterations"] = 2
        with self.assertRaisesRegex(TRACE.TraceError, "before the final"):
            TRACE.validate(fixture)

    def test_rejects_out_of_order_iteration_or_failed_attempt_index(self) -> None:
        """Reject out-of-order iteration and failed-attempt indices."""
        fixture = _fixture()
        fixture["iterations"] = [_iteration(index=2)]
        with self.assertRaisesRegex(TRACE.TraceError, "out-of-order index"):
            TRACE.validate(fixture)
        fixture = _eigensolver_failure()
        fixture["failed_attempt"]["index"] = 2
        with self.assertRaisesRegex(TRACE.TraceError, "must be 1"):
            TRACE.validate(fixture)

    def test_terminal_count_status_and_convergence_are_consistent(self) -> None:
        """Keep terminal count, status, and convergence mutually consistent."""
        fixture = _fixture()
        fixture["terminal"]["iterations"] = 999
        with self.assertRaisesRegex(TRACE.TraceError, "must be 1"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["terminal"]["status"] = 9
        with self.assertRaisesRegex(TRACE.TraceError, "status must be 1"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["terminal"]["converged"] = False
        with self.assertRaisesRegex(TRACE.TraceError, "true exactly"):
            TRACE.validate(fixture)
        fixture = _eigensolver_failure()
        fixture["terminal"]["status"] = TRACE.STATUS_MAX_ITERATIONS
        with self.assertRaisesRegex(TRACE.TraceError, "only valid"):
            TRACE.validate(fixture)

    def test_rejects_nonfinite_values_in_core_or_extensible_provenance(self) -> None:
        """Reject nonfinite values in core data and extensible provenance."""
        fixture = _fixture()
        fixture["iterations"][0]["energy"] = float("nan")
        with self.assertRaisesRegex(TRACE.TraceError, "finite"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["provenance"]["toolchain"] = {"clock": float("inf")}
        with self.assertRaisesRegex(TRACE.TraceError, "finite"):
            TRACE.validate(fixture)

    def test_provenance_is_lowercase_hex(self) -> None:
        """Require provenance digests and revisions to use lowercase hex."""
        fixture = _fixture()
        fixture["provenance"]["tblite_revision"] = "z" * 40
        with self.assertRaisesRegex(TRACE.TraceError, "40-hex"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["provenance"]["oracle_patch_sha256"] = "A" * 64
        with self.assertRaisesRegex(TRACE.TraceError, "64-hex"):
            TRACE.validate(fixture)

    def test_rejects_schema_forbidden_extra_fields(self) -> None:
        """Reject extra fields outside explicitly extensible objects."""
        fixture = _fixture()
        fixture["unexpected"] = True
        with self.assertRaisesRegex(TRACE.TraceError, "unsupported field"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["iterations"][0]["status"] = TRACE.STATUS_CONVERGED
        with self.assertRaisesRegex(TRACE.TraceError, "unsupported field"):
            TRACE.validate(fixture)
        fixture = _fixture()
        fixture["provenance"][1] = "non-string key"
        with self.assertRaisesRegex(TRACE.TraceError, "keys must be strings"):
            TRACE.validate(fixture)

    def test_schema_document_is_valid_and_versioned(self) -> None:
        """Keep the checked-in JSON schema valid and explicitly versioned."""
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["format"]["const"], TRACE.FORMAT)
        self.assertEqual(
            schema["properties"]["input"]["properties"]["spin_channels"]["const"], 1
        )
        self.assertEqual(
            schema["properties"]["input"]["properties"]["unpaired_electrons"][
                "minimum"
            ],
            0,
        )
        self.assertEqual(schema["definitions"]["restricted_occupations"]["minItems"], 2)


if __name__ == "__main__":
    unittest.main()
