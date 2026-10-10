"""Synthetic tests for opt-in offline split-source provenance checking."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools/conformance/xtbloom_conformance.py"
CANONICAL_MANIFEST_PATH = ROOT / "data/conformance/manifest.json"
CANONICAL_GOLDEN_PATH = ROOT / "data/conformance/golden/ketene.json"
SPEC = importlib.util.spec_from_file_location("split_reference_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)

CANONICAL_MANIFEST = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
CANONICAL_GOLDEN = json.loads(CANONICAL_GOLDEN_PATH.read_text(encoding="utf-8"))
PINNED_TBLITE = CANONICAL_MANIFEST["reference_engines"]["tblite"]
PINNED_CLI_PROVENANCE = CANONICAL_GOLDEN["provenance"]
UNITS = copy.deepcopy(CANONICAL_MANIFEST["units"])
METHOD = CANONICAL_MANIFEST["method"]
PRODUCER_ID = "synthetic_h2_producer"
PROSPECTIVE_ID = "synthetic_h2_prospective"
COORD_BYTES = (
    b"$coord\n"
    b"0.0000000000000000 0.0000000000000000 0.0000000000000000 h\n"
    b"0.0000000000000000 0.0000000000000000 1.4000000000000000 h\n"
    b"$end\n"
)
ENERGY = -1.1
FORCES = [0.01, 0.0, 0.02, -0.01, 0.0, -0.02]
GRADIENT = [-value for value in FORCES]
CHARGES = [-0.05, 0.05]
API_SETTINGS = {
    "accuracy": 1.0e-4,
    "charge": 0,
    "unpaired": 0,
    "nspin": "default1",
    "initial_guess": "defaultSAD",
    "max_iterations": "default250",
    "temperature_hartree": "default300 * 3.166808578545117e-06",
    "configuration": "NULL optional config; builtin pinned GFN2 parameters",
}
API_STAGES = [
    "setVerbosity",
    "newStructure",
    "newCalculatorAndResult",
    "setAccuracy",
    "singlepoint",
    "get_energy",
    "get_gradient",
    "get_charges",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _set_nested(document: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = copy.deepcopy(value)


class SyntheticSplitReference:
    """Build a complete bundle from temporary, non-executed synthetic records."""

    def __init__(self, temporary_root: Path) -> None:
        self.root = temporary_root / "bundle"
        self.root.mkdir()
        self.input_path = self.root / "inputs/synthetic-h2.coord"
        self.golden_path = self.root / "golden/synthetic-h2.json"
        self.cli_path = self.root / "frozen-sources/producer-cli.json"
        self.api_path = self.root / "frozen-sources/producer-api.json"
        self.driver_path = self.root / "frozen-sources/api-producer.py"
        self.captured_input_path = (
            self.root / f"frozen-sources/{PRODUCER_ID}-api-captured-input.coord"
        )
        self.manifest_path = self.root / "manifest.json"
        self.input_path.parent.mkdir(parents=True)
        self.cli_path.parent.mkdir(parents=True, exist_ok=True)
        self.input_path.write_bytes(COORD_BYTES)
        self.driver_path.write_text(
            "Synthetic fixture marker. This file is never executed.\n",
            encoding="utf-8",
        )
        self.captured_input_path.write_bytes(COORD_BYTES)
        input_sha256 = _sha256(COORD_BYTES)
        positions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.4]
        positions_sha256 = _sha256(struct.pack("<6d", *positions))
        properties = {
            "energy_hartree": ENERGY,
            "forces_hartree_per_bohr": copy.deepcopy(FORCES),
            "partial_charges_e": copy.deepcopy(CHARGES),
        }
        cli_properties = {
            "energy_hartree": ENERGY,
            "forces_hartree_per_bohr": copy.deepcopy(FORCES),
            "gradient_hartree_per_bohr": copy.deepcopy(GRADIENT),
            "virial_hartree": [0.0] * 9,
        }
        cli_template = copy.deepcopy(PINNED_TBLITE["cli_command_template"])
        self.cli = {
            "case_id": PRODUCER_ID,
            "schema_version": 1,
            "method": METHOD,
            "units": copy.deepcopy(UNITS),
            "molecular_charge": 0,
            "unpaired_electrons": 0,
            "properties": cli_properties,
            "provenance": {
                "accuracy": PINNED_TBLITE["accuracy"],
                "command": copy.deepcopy(cli_template),
                "command_template": copy.deepcopy(cli_template),
                "engine": "tblite",
                "environment": copy.deepcopy(PINNED_CLI_PROVENANCE["environment"]),
                "executable_sha256": PINNED_CLI_PROVENANCE["executable_sha256"],
                "executable_version": PINNED_CLI_PROVENANCE["executable_version"],
                "generation_mode": "live-cli",
                "input": f"/synthetic/{PRODUCER_ID}.coord",
                "runtime": copy.deepcopy(PINNED_CLI_PROVENANCE["runtime"]),
                "source_output_sha256": CONFORMANCE.sha256_json(cli_properties),
                "source_revision": PINNED_TBLITE["revision"],
            },
        }
        self.api = {
            "case_id": PRODUCER_ID,
            "schema_version": 1,
            "method": METHOD,
            "units": copy.deepcopy(UNITS),
            "molecular_charge": 0,
            "unpaired_electrons": 0,
            "spin_channels": 1,
            "input_sha256": input_sha256,
            "positions_binary64_le_sha256": positions_sha256,
            "numerical_settings": copy.deepcopy(API_SETTINGS),
            "api_checks": [
                {
                    "stage": stage,
                    "error_status": 0,
                    "context_status": 0,
                }
                for stage in API_STAGES
            ],
            "deleted_objects_null": True,
            "attempted_singlepoints": 1,
            "version_api": 700,
            "pinned_artifacts": {
                "/synthetic/runtime/libtblite.so.0.7.0": PINNED_TBLITE[
                    "runtime_artifacts"
                ]["libtblite_sha256"]
            },
            "loaded_library_hashes": {
                "/synthetic/runtime/libtblite.so.0.7.0": PINNED_TBLITE[
                    "runtime_artifacts"
                ]["libtblite_sha256"]
            },
            "environment": {
                "LC_ALL": "C",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "LD_PRELOAD": None,
                "LD_AUDIT": None,
            },
            "properties": {
                "energy_hartree": ENERGY,
                "forces_hartree_per_bohr": copy.deepcopy(FORCES),
                "partial_charges_e": copy.deepcopy(CHARGES),
            },
            "claim_eligible": False,
        }
        self.case = {
            "id": PROSPECTIVE_ID,
            "atom_count": 2,
            "input": str(self.input_path),
            "input_sha256": input_sha256,
            "golden": str(self.golden_path),
            "golden_sha256": "",
            "reference_engine": "tblite",
            "reference_output_sha256": "",
            "molecular_charge": 0,
            "unpaired_electrons": 0,
            "spin_channels": 1,
            "positions_binary64_le_sha256": positions_sha256,
            "qualification": {
                "performance_claim_eligible": False,
                "scientific_correctness_qualified": False,
                "native_implementation_validated": False,
            },
        }
        self.golden = {
            "case_id": PROSPECTIVE_ID,
            "method": METHOD,
            "molecular_charge": 0,
            "properties": properties,
            "provenance": {
                "engine": "tblite",
                "generation_mode": "derived-live-cli-and-live-c-api",
                "source_revision": PINNED_TBLITE["revision"],
                "source_output_sha256": "",
                "accuracy": PINNED_TBLITE["accuracy"],
                "claim_eligible": False,
                "cli_aggregate_literal_exit": 1,
                "cli_aggregate_failure": "synthetic declared aggregate exit",
                "property_sources": {
                    "energy_hartree": {
                        "relative_path": "frozen-sources/producer-cli.json",
                        "sha256": "",
                        "bytes": 0,
                        "source_path": "/historical/producer/producer-cli.json",
                    },
                    "forces_hartree_per_bohr": {
                        "relative_path": "frozen-sources/producer-cli.json",
                        "sha256": "",
                        "bytes": 0,
                        "source_path": "/historical/producer/producer-cli.json",
                    },
                    "partial_charges_e": {
                        "relative_path": "frozen-sources/producer-api.json",
                        "sha256": "",
                        "bytes": 0,
                        "source_path": "/historical/producer/producer-api.json",
                    },
                },
                "label_mapping": {
                    "producer_case_id": PRODUCER_ID,
                    "prospective_case_id": PROSPECTIVE_ID,
                },
                "cli_provenance": copy.deepcopy(self.cli["provenance"]),
                "api_settings": copy.deepcopy(API_SETTINGS),
                "api_driver_sha256": _sha256(self.driver_path.read_bytes()),
            },
            "schema_version": 1,
            "units": copy.deepcopy(UNITS),
            "unpaired_electrons": 0,
        }
        self.manifest = {
            "schema_version": 1,
            "golden_schema_version": 1,
            "method": METHOD,
            "units": copy.deepcopy(UNITS),
            "claim_eligible": False,
            "reference_engines": copy.deepcopy(CANONICAL_MANIFEST["reference_engines"]),
            "cases": [self.case],
        }
        self.persist(refresh_cli_output_digest=True)

    def persist(
        self,
        *,
        refresh_source_pins: bool = True,
        refresh_cli_output_digest: bool = False,
        sync_cli_provenance: bool = True,
        sync_api_settings: bool = True,
        refresh_composite_output_digest: bool = True,
    ) -> None:
        """Write the bundle and refresh only the pins selected by each test."""
        if refresh_cli_output_digest:
            self.cli["provenance"]["source_output_sha256"] = CONFORMANCE.sha256_json(
                self.cli["properties"]
            )
        _write_json(self.cli_path, self.cli)
        _write_json(self.api_path, self.api)
        if refresh_source_pins:
            for name, path in (
                ("energy_hartree", self.cli_path),
                ("forces_hartree_per_bohr", self.cli_path),
                ("partial_charges_e", self.api_path),
            ):
                descriptor = self.golden["provenance"]["property_sources"][name]
                contents = path.read_bytes()
                descriptor["sha256"] = _sha256(contents)
                descriptor["bytes"] = len(contents)
        if sync_cli_provenance:
            self.golden["provenance"]["cli_provenance"] = copy.deepcopy(
                self.cli["provenance"]
            )
        if sync_api_settings:
            self.golden["provenance"]["api_settings"] = copy.deepcopy(
                self.api["numerical_settings"]
            )
        if refresh_composite_output_digest:
            output_digest = CONFORMANCE.sha256_json(self.golden["properties"])
            self.golden["provenance"]["source_output_sha256"] = output_digest
            self.case["reference_output_sha256"] = output_digest
        _write_json(self.golden_path, self.golden)
        self.case["golden_sha256"] = _sha256(self.golden_path.read_bytes())
        _write_json(self.manifest_path, self.manifest)


Mutation = Callable[[SyntheticSplitReference], None]


class SplitReferenceTest(unittest.TestCase):
    """Cover the opt-in boundary and hostile split-source records offline."""

    def assert_rejected(
        self,
        name: str,
        mutate: Mutation,
        *,
        persist_options: dict[str, bool] | None = None,
    ) -> None:
        """Assert that one independently pinned synthetic mutation is rejected."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            mutate(fixture)
            fixture.persist(**(persist_options or {}))
            with (
                self.subTest(record=name),
                redirect_stdout(StringIO()),
                self.assertRaises(CONFORMANCE.ConformanceError),
            ):
                CONFORMANCE.check_manifest(
                    fixture.manifest_path, allow_derived_tblite=True
                )

    def test_opt_in_accepts_synthetic_cli_api_composite(self) -> None:
        """Only the explicit option admits the synthetic split-source record."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            with redirect_stdout(StringIO()):
                CONFORMANCE.check_manifest(
                    fixture.manifest_path, allow_derived_tblite=True
                )
            self.assertEqual(
                set(fixture.golden["properties"]),
                {"energy_hartree", "forces_hartree_per_bohr", "partial_charges_e"},
            )
            self.assertIn("gradient_hartree_per_bohr", fixture.cli["properties"])
            self.assertIn("virial_hartree", fixture.cli["properties"])
            self.assertNotIn("gradient_hartree_per_bohr", fixture.golden["properties"])
            self.assertEqual(
                fixture.golden["provenance"]["property_sources"]["energy_hartree"],
                fixture.golden["provenance"]["property_sources"][
                    "forces_hartree_per_bohr"
                ],
            )
            self.assertFalse(
                Path(
                    fixture.golden["provenance"]["property_sources"]["energy_hartree"][
                        "source_path"
                    ]
                ).exists()
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--manifest",
                    str(fixture.manifest_path),
                    "check",
                    "--allow-derived-tblite",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertIn("source consistency only", completed.stdout.lower())
            self.assertIn(
                "input-byte link remains unverified", completed.stdout.lower()
            )

    def test_zero_aggregate_exit_requires_an_empty_failure(self) -> None:
        """A declared successful aggregate has no failure description."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            fixture.golden["provenance"]["cli_aggregate_literal_exit"] = 0
            fixture.golden["provenance"]["cli_aggregate_failure"] = ""
            fixture.persist()
            with redirect_stdout(StringIO()):
                CONFORMANCE.check_manifest(
                    fixture.manifest_path, allow_derived_tblite=True
                )

    def test_explicit_full_oracle_property_list_is_permitted(self) -> None:
        """A declared oracle subset must retain energy, forces, and charges."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            fixture.case["xtbloom_oracle_properties"] = [
                "energy_hartree",
                "forces_hartree_per_bohr",
                "partial_charges_e",
            ]
            fixture.persist()
            with redirect_stdout(StringIO()):
                CONFORMANCE.check_manifest(
                    fixture.manifest_path, allow_derived_tblite=True
                )

    def test_rejects_source_bytes_changed_after_pinning(self) -> None:
        """A matching JSON shape cannot replace the frozen source bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            fixture.cli_path.write_bytes(fixture.cli_path.read_bytes() + b" ")
            with (
                redirect_stdout(StringIO()),
                self.assertRaises(CONFORMANCE.ConformanceError),
            ):
                CONFORMANCE.check_manifest(
                    fixture.manifest_path, allow_derived_tblite=True
                )

    def test_default_check_rejects_split_source_and_cli_flag_is_opt_in(self) -> None:
        """The canonical default refuses derived records without the flag."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticSplitReference(Path(temporary))
            with (
                redirect_stdout(StringIO()),
                self.assertRaisesRegex(
                    CONFORMANCE.ConformanceError, "allow-derived-tblite"
                ),
            ):
                CONFORMANCE.check_manifest(fixture.manifest_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_PATH),
                    "--manifest",
                    str(fixture.manifest_path),
                    "check",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("allow-derived-tblite", completed.stderr)

    def test_rejects_path_pins_extents_and_source_mapping_tampering(self) -> None:
        """Byte pins and source roles are checked independently of JSON validity."""
        cases: list[tuple[str, Mutation, dict[str, bool]]] = []

        def add(
            name: str,
            mutate: Mutation,
            **persist_options: bool,
        ) -> None:
            cases.append((name, mutate, persist_options))

        add(
            "source digest",
            lambda fixture: _set_nested(
                fixture.golden,
                ("provenance", "property_sources", "energy_hartree", "sha256"),
                "0" * 64,
            ),
            refresh_source_pins=False,
        )
        add(
            "source byte count",
            lambda fixture: _set_nested(
                fixture.golden,
                ("provenance", "property_sources", "partial_charges_e", "bytes"),
                fixture.golden["provenance"]["property_sources"]["partial_charges_e"][
                    "bytes"
                ]
                + 1,
            ),
            refresh_source_pins=False,
        )
        add(
            "parent traversal",
            lambda fixture: _set_nested(
                fixture.golden,
                (
                    "provenance",
                    "property_sources",
                    "energy_hartree",
                    "relative_path",
                ),
                "../outside.json",
            ),
            refresh_source_pins=False,
        )
        add(
            "absolute source path",
            lambda fixture: _set_nested(
                fixture.golden,
                (
                    "provenance",
                    "property_sources",
                    "energy_hartree",
                    "relative_path",
                ),
                str(fixture.cli_path),
            ),
            refresh_source_pins=False,
        )

        def symlink_escape(fixture: SyntheticSplitReference) -> None:
            outside = fixture.root.parent / "outside-source.json"
            outside.write_bytes(fixture.cli_path.read_bytes())
            link = fixture.root / "frozen-sources/escaped-source.json"
            link.symlink_to(outside)
            descriptor = fixture.golden["provenance"]["property_sources"][
                "energy_hartree"
            ]
            descriptor["relative_path"] = "frozen-sources/escaped-source.json"
            descriptor["sha256"] = _sha256(outside.read_bytes())
            descriptor["bytes"] = len(outside.read_bytes())

        add("symlink escape", symlink_escape, refresh_source_pins=False)

        add(
            "wrong composite force extent",
            lambda fixture: fixture.golden["properties"][
                "forces_hartree_per_bohr"
            ].pop(),
        )
        add(
            "wrong CLI gradient extent",
            lambda fixture: fixture.cli["properties"][
                "gradient_hartree_per_bohr"
            ].pop(),
            refresh_cli_output_digest=True,
        )
        add(
            "E/F descriptor split",
            lambda fixture: _set_nested(
                fixture.golden,
                (
                    "provenance",
                    "property_sources",
                    "forces_hartree_per_bohr",
                ),
                copy.deepcopy(
                    fixture.golden["provenance"]["property_sources"][
                        "partial_charges_e"
                    ]
                ),
            ),
            refresh_source_pins=False,
        )
        add(
            "charge source mapped to CLI",
            lambda fixture: _set_nested(
                fixture.golden,
                ("provenance", "property_sources", "partial_charges_e"),
                copy.deepcopy(
                    fixture.golden["provenance"]["property_sources"]["energy_hartree"]
                ),
            ),
            refresh_source_pins=False,
        )
        add(
            "invalid producer label",
            lambda fixture: _set_nested(
                fixture.golden,
                ("provenance", "label_mapping", "producer_case_id"),
                "producer/with/slash",
            ),
        )
        add(
            "prospective label mismatch",
            lambda fixture: _set_nested(
                fixture.golden,
                ("provenance", "label_mapping", "prospective_case_id"),
                "another_case",
            ),
        )

        def wrong_producer_case(fixture: SyntheticSplitReference) -> None:
            fixture.api["case_id"] = "different_producer"

        add("API producer case mismatch", wrong_producer_case)

        def wrong_case_label(fixture: SyntheticSplitReference) -> None:
            fixture.case["id"] = "renamed_prospective"
            fixture.golden["case_id"] = "renamed_prospective"

        add("case ID differs from prospective label", wrong_case_label)
        add(
            "wrong API input digest",
            lambda fixture: fixture.api.__setitem__("input_sha256", "f" * 64),
        )

        def wrong_captured_input(fixture: SyntheticSplitReference) -> None:
            fixture.captured_input_path.write_bytes(COORD_BYTES + b"# altered\n")

        add("captured input digest", wrong_captured_input)

        def wrong_positions_digest(fixture: SyntheticSplitReference) -> None:
            wrong = "e" * 64
            fixture.case["positions_binary64_le_sha256"] = wrong
            fixture.api["positions_binary64_le_sha256"] = wrong

        add("recomputed positions digest", wrong_positions_digest)
        add(
            "driver digest",
            lambda fixture: _set_nested(
                fixture.golden, ("provenance", "api_driver_sha256"), "d" * 64
            ),
        )
        add(
            "source output digest",
            lambda fixture: _set_nested(
                fixture.cli, ("provenance", "source_output_sha256"), "a" * 64
            ),
        )
        add(
            "CLI input producer basename mismatch",
            lambda fixture: fixture.cli["provenance"].__setitem__(
                "input", "/synthetic/other.coord"
            ),
        )
        add(
            "aggregate exit is text",
            lambda fixture: fixture.golden["provenance"].__setitem__(
                "cli_aggregate_literal_exit", "1"
            ),
        )
        add(
            "aggregate exit is boolean",
            lambda fixture: fixture.golden["provenance"].__setitem__(
                "cli_aggregate_literal_exit", True
            ),
        )
        add(
            "aggregate exit out of range",
            lambda fixture: fixture.golden["provenance"].__setitem__(
                "cli_aggregate_literal_exit", 256
            ),
        )
        add(
            "nonzero aggregate with empty failure",
            lambda fixture: fixture.golden["provenance"].__setitem__(
                "cli_aggregate_failure", ""
            ),
        )
        add(
            "zero aggregate with failure text",
            lambda fixture: (
                fixture.golden["provenance"].__setitem__(
                    "cli_aggregate_literal_exit", 0
                ),
                fixture.golden["provenance"].__setitem__(
                    "cli_aggregate_failure", "contradictory failure"
                ),
            ),
        )

        for name, mutate, options in cases:
            with self.subTest(record=name):
                self.assert_rejected(name, mutate, persist_options=options)

    def test_rejects_scope_and_eligibility_escalation(self) -> None:
        """The opt-in remains restricted-neutral and never grants eligibility."""
        cases: list[tuple[str, Mutation]] = [
            (
                "manifest claim eligibility",
                lambda fixture: fixture.manifest.__setitem__("claim_eligible", True),
            ),
            (
                "composite claim eligibility",
                lambda fixture: fixture.golden["provenance"].__setitem__(
                    "claim_eligible", True
                ),
            ),
            (
                "API claim eligibility",
                lambda fixture: fixture.api.__setitem__("claim_eligible", True),
            ),
            (
                "performance qualification",
                lambda fixture: fixture.case["qualification"].__setitem__(
                    "performance_claim_eligible", True
                ),
            ),
            (
                "science qualification",
                lambda fixture: fixture.case["qualification"].__setitem__(
                    "scientific_correctness_qualified", True
                ),
            ),
            (
                "native implementation qualification",
                lambda fixture: fixture.case["qualification"].__setitem__(
                    "native_implementation_validated", True
                ),
            ),
            (
                "missing native implementation qualification",
                lambda fixture: fixture.case["qualification"].pop(
                    "native_implementation_validated"
                ),
            ),
            (
                "unsupported tblite version",
                lambda fixture: fixture.manifest["reference_engines"][
                    "tblite"
                ].__setitem__("version", "0.7.1"),
            ),
            (
                "electric field attachment",
                lambda fixture: fixture.case.__setitem__("efield", [0.0, 0.0, 0.1]),
            ),
            (
                "point-charge count metadata",
                lambda fixture: fixture.case.__setitem__("point_charge_count", 1),
            ),
            (
                "QMMM input metadata",
                lambda fixture: fixture.case.__setitem__("input_schema", "qmmm-v1"),
            ),
            (
                "oracle properties omit charges",
                lambda fixture: fixture.case.__setitem__(
                    "xtbloom_oracle_properties",
                    ["energy_hartree", "forces_hartree_per_bohr"],
                ),
            ),
            (
                "charged case",
                lambda fixture: (
                    fixture.case.__setitem__("molecular_charge", 1),
                    fixture.golden.__setitem__("molecular_charge", 1),
                ),
            ),
            (
                "open-shell case",
                lambda fixture: (
                    fixture.case.__setitem__("unpaired_electrons", 1),
                    fixture.golden.__setitem__("unpaired_electrons", 1),
                ),
            ),
            (
                "spin polarized case",
                lambda fixture: fixture.case.__setitem__("spin_channels", 2),
            ),
            (
                "boolean false charge is not integer zero",
                lambda fixture: (
                    fixture.case.__setitem__("molecular_charge", False),
                    fixture.golden.__setitem__("molecular_charge", False),
                ),
            ),
            (
                "boolean false unpaired count is not integer zero",
                lambda fixture: (
                    fixture.case.__setitem__("unpaired_electrons", False),
                    fixture.golden.__setitem__("unpaired_electrons", False),
                ),
            ),
            (
                "boolean true spin count is not integer one",
                lambda fixture: fixture.case.__setitem__("spin_channels", True),
            ),
        ]
        for name, mutate in cases:
            with self.subTest(record=name):
                self.assert_rejected(name, mutate)

    def test_rejects_settings_lifecycle_status_and_runtime_mismatches(self) -> None:
        """Recorded defaults, API ordering and runtime identity stay exact."""
        cases: list[tuple[str, Mutation, dict[str, bool]]] = []

        def add(name: str, mutate: Mutation, **options: bool) -> None:
            cases.append((name, mutate, options))

        add(
            "API max iteration setting",
            lambda fixture: fixture.api["numerical_settings"].__setitem__(
                "max_iterations", "default251"
            ),
        )
        add(
            "API charge false is not integer zero",
            lambda fixture: fixture.api["numerical_settings"].__setitem__(
                "charge", False
            ),
        )
        add(
            "API unpaired false is not integer zero",
            lambda fixture: fixture.api["numerical_settings"].__setitem__(
                "unpaired", False
            ),
        )
        add(
            "embedded API charge false is not integer zero",
            lambda fixture: _set_nested(
                fixture.golden, ("provenance", "api_settings", "charge"), False
            ),
            sync_api_settings=False,
        )
        add(
            "embedded API unpaired false is not integer zero",
            lambda fixture: _set_nested(
                fixture.golden, ("provenance", "api_settings", "unpaired"), False
            ),
            sync_api_settings=False,
        )
        add(
            "API singlepoint count bool",
            lambda fixture: fixture.api.__setitem__("attempted_singlepoints", True),
        )
        add(
            "API objects not deleted",
            lambda fixture: fixture.api.__setitem__("deleted_objects_null", False),
        )
        add(
            "API version bool",
            lambda fixture: fixture.api.__setitem__("version_api", True),
        )
        add(
            "wrong API version",
            lambda fixture: fixture.api.__setitem__("version_api", 701),
        )

        def boolean_status(fixture: SyntheticSplitReference) -> None:
            fixture.api["api_checks"][0]["error_status"] = False

        add("boolean status is not integer zero", boolean_status)

        def boolean_context_status(fixture: SyntheticSplitReference) -> None:
            fixture.api["api_checks"][1]["context_status"] = False

        add("boolean context status is not integer zero", boolean_context_status)

        def reordered_stages(fixture: SyntheticSplitReference) -> None:
            checks = fixture.api["api_checks"]
            checks[6], checks[7] = checks[7], checks[6]

        add("API stage order", reordered_stages)
        add(
            "API pinned library hash",
            lambda fixture: fixture.api["pinned_artifacts"].__setitem__(
                "/synthetic/runtime/libtblite.so.0.7.0", "0" * 64
            ),
        )

        def different_loaded_library_path(fixture: SyntheticSplitReference) -> None:
            digest = PINNED_TBLITE["runtime_artifacts"]["libtblite_sha256"]
            fixture.api["loaded_library_hashes"] = {
                "/another/runtime/libtblite.so.0.7.0": digest
            }

        add("API loaded library path mismatch", different_loaded_library_path)
        add(
            "API preload is absent",
            lambda fixture: fixture.api["environment"].__setitem__(
                "LD_PRELOAD", "unexpected.so"
            ),
        )
        add(
            "API audit library is absent",
            lambda fixture: fixture.api["environment"].__setitem__(
                "LD_AUDIT", "unexpected.so"
            ),
        )

        def missing_preload_key(fixture: SyntheticSplitReference) -> None:
            del fixture.api["environment"]["LD_PRELOAD"]

        add("missing API preload declaration", missing_preload_key)

        def missing_audit_key(fixture: SyntheticSplitReference) -> None:
            del fixture.api["environment"]["LD_AUDIT"]

        add("missing API audit declaration", missing_audit_key)
        add(
            "API thread environment",
            lambda fixture: fixture.api["environment"].__setitem__(
                "OMP_NUM_THREADS", "2"
            ),
        )
        add(
            "CLI accuracy",
            lambda fixture: fixture.cli["provenance"].__setitem__("accuracy", 1.0e-3),
        )
        add(
            "CLI revision",
            lambda fixture: fixture.cli["provenance"].__setitem__(
                "source_revision", "0" * 40
            ),
        )
        add(
            "CLI executable version",
            lambda fixture: fixture.cli["provenance"].__setitem__(
                "executable_version", "tblite version 0.7.1"
            ),
        )
        add(
            "CLI command template",
            lambda fixture: fixture.cli["provenance"].__setitem__(
                "command_template", ["tblite"]
            ),
        )
        add(
            "CLI runtime library hash",
            lambda fixture: fixture.cli["provenance"]["runtime"][
                "libtblite"
            ].__setitem__("sha256", "9" * 64),
        )

        def malformed_cli_runtime(fixture: SyntheticSplitReference) -> None:
            fixture.cli["provenance"]["runtime"] = {
                "libtblite": {"filename": "libtblite.so.0.7.0"}
            }

        add("malformed CLI runtime nesting", malformed_cli_runtime)

        def missing_cli_runtime(fixture: SyntheticSplitReference) -> None:
            fixture.cli["provenance"]["runtime"] = None

        add("missing CLI runtime object", missing_cli_runtime)

        def malformed_cli_environment(fixture: SyntheticSplitReference) -> None:
            fixture.cli["provenance"]["environment"] = {"set": None}

        add("malformed CLI environment nesting", malformed_cli_environment)

        def missing_cli_environment(fixture: SyntheticSplitReference) -> None:
            fixture.cli["provenance"]["environment"] = None

        add("missing CLI environment object", missing_cli_environment)

        def cli_provenance_not_embedded(fixture: SyntheticSplitReference) -> None:
            fixture.golden["provenance"]["cli_provenance"]["accuracy"] = 2.0e-4

        add(
            "embedded CLI provenance differs",
            cli_provenance_not_embedded,
            sync_cli_provenance=False,
        )

        for name, mutate, options in cases:
            with self.subTest(record=name):
                self.assert_rejected(name, mutate, persist_options=options)

    def test_rejects_numeric_coercions_charge_drift_and_force_sign_changes(
        self,
    ) -> None:
        """JSON numbers must be finite real scalars with exact source agreement."""
        cases: list[tuple[str, Mutation, dict[str, bool]]] = []

        def add(name: str, mutate: Mutation, **options: bool) -> None:
            cases.append((name, mutate, options))

        add(
            "boolean composite energy",
            lambda fixture: fixture.golden["properties"].__setitem__(
                "energy_hartree", True
            ),
        )
        add(
            "string composite energy",
            lambda fixture: fixture.golden["properties"].__setitem__(
                "energy_hartree", "-1.1"
            ),
        )
        add(
            "huge integer composite energy",
            lambda fixture: fixture.golden["properties"].__setitem__(
                "energy_hartree", 10**400
            ),
        )

        def nonfinite_api_charge(fixture: SyntheticSplitReference) -> None:
            fixture.api["properties"]["partial_charges_e"][0] = float("nan")

        add("nonfinite API charge", nonfinite_api_charge)

        def source_string_energy(fixture: SyntheticSplitReference) -> None:
            fixture.cli["properties"]["energy_hartree"] = "-1.1"

        add("string CLI energy", source_string_energy, refresh_cli_output_digest=True)

        def source_boolean_force(fixture: SyntheticSplitReference) -> None:
            fixture.api["properties"]["forces_hartree_per_bohr"][0] = False

        add("boolean API force", source_boolean_force)

        def charge_mismatch(fixture: SyntheticSplitReference) -> None:
            fixture.api["properties"]["partial_charges_e"][0] = -0.04

        add("API charge differs from composite", charge_mismatch)

        def api_energy_mismatch(fixture: SyntheticSplitReference) -> None:
            fixture.api["properties"]["energy_hartree"] = ENERGY + 0.01

        add("API energy differs from CLI", api_energy_mismatch)

        def api_force_mismatch(fixture: SyntheticSplitReference) -> None:
            fixture.api["properties"]["forces_hartree_per_bohr"][0] += 0.01

        add("API force differs from CLI", api_force_mismatch)

        def force_sign(fixture: SyntheticSplitReference) -> None:
            fixture.cli["properties"]["gradient_hartree_per_bohr"] = copy.deepcopy(
                FORCES
            )

        add("CLI force sign", force_sign, refresh_cli_output_digest=True)
        add(
            "CLI source contains charges",
            lambda fixture: fixture.cli["properties"].__setitem__(
                "partial_charges_e", copy.deepcopy(CHARGES)
            ),
            refresh_cli_output_digest=True,
        )
        add(
            "composite contains validation gradient",
            lambda fixture: fixture.golden["properties"].__setitem__(
                "gradient_hartree_per_bohr", copy.deepcopy(GRADIENT)
            ),
        )

        for name, mutate, options in cases:
            with self.subTest(record=name):
                self.assert_rejected(name, mutate, persist_options=options)


if __name__ == "__main__":
    unittest.main()
