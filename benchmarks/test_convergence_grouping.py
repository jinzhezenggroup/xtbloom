"""Leakage, split, annotation, and freeze-plan tests for issue #514."""

from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks import convergence_grouping as grouping


def _case(
    case_id: str,
    group_id: str,
    *,
    atomic_numbers: list[int] | None = None,
    ao_count: int = 4,
    charge: float = 0,
    spin_channels: int | None = 1,
    unpaired_electrons: int | None = 0,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "atomic_numbers": atomic_numbers
        if atomic_numbers is not None
        else [6, 1, 1, 1, 1],
        "exact_ao_count": ao_count,
        "molecular_charge": charge,
    }
    if spin_channels is not None:
        metadata["spin_channels"] = spin_channels
    if unpaired_electrons is not None:
        metadata["unpaired_electrons"] = unpaired_electrons
    return {
        "case_id": case_id,
        "molecule_group_id": group_id,
        "scheduling_metadata": metadata,
    }


def _manifest(systems: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "systems": systems
        if systems is not None
        else [
            _case("alkane-a-conf-1", "alkane-a", ao_count=18),
            _case("alkane-a-conf-2", "alkane-a", ao_count=18),
            _case("ion-b", "ion-b", atomic_numbers=[8, 1], charge=-1, ao_count=7),
            _case(
                "radical-c",
                "radical-c",
                atomic_numbers=[6, 18],
                charge=1,
                spin_channels=2,
                unpaired_electrons=1,
                ao_count=12,
            ),
            _case("control-d", "control-d", ao_count=6),
            _case("control-e", "control-e", ao_count=9),
        ],
    }


class SchedulingContractTests(unittest.TestCase):
    """Keep the candidate input-only, explicit, complete, and explainable."""

    def test_risk_annotations_use_only_allowlisted_input_fields(self) -> None:
        """Map declared charge, spin, element, and AO inputs to fixed bands."""
        annotations = grouping.annotate_scheduling_inputs(_manifest())
        neutral = annotations["alkane-a-conf-1"]
        ionic = annotations["ion-b"]
        difficult = annotations["radical-c"]
        self.assertEqual((neutral.risk_score, neutral.risk_band), (0, "low"))
        self.assertEqual(ionic.risk_score, 1)
        self.assertEqual(ionic.risk_band, "guarded")
        self.assertEqual(difficult.risk_score, 4)
        self.assertEqual(difficult.risk_band, "high")
        self.assertIn("exact_ao_count", grouping.policy_document()["allowed_features"])
        self.assertIn(
            "molecular_charge", grouping.policy_document()["allowed_features"]
        )

    def test_difficult_candidate_inputs_are_never_filtered(self) -> None:
        """Retain every requested ID even when explicit risk is high."""
        systems = _manifest()["systems"]
        self.assertIsInstance(systems, list)
        difficult = _case(
            "known-hard-radical",
            "hard-family",
            atomic_numbers=[26, 8, 1],
            charge=-1,
            spin_channels=2,
            unpaired_electrons=3,
            ao_count=24,
        )
        systems.append(difficult)
        annotations = grouping.annotate_scheduling_inputs(_manifest(systems))
        self.assertEqual(set(annotations), {item["case_id"] for item in systems})
        self.assertEqual(annotations["known-hard-radical"].risk_band, "high")
        self.assertEqual(len(annotations), len(systems))

    def test_case_and_group_ids_are_mapping_fields_not_risk_features(self) -> None:
        """Keep identifiers out of feature values and risk explanations."""
        annotation = grouping.annotate_scheduling_inputs(_manifest())["radical-c"]
        self.assertNotIn("radical-c", annotation.risk_reasons)
        self.assertNotIn("radical-c", grouping.policy_document()["allowed_features"])
        self.assertNotIn(
            "molecule_group_id", grouping.policy_document()["allowed_features"]
        )
        self.assertEqual(annotation.planner_bucket_key, (12, 0))

    def test_baselines_are_distinct_and_original_remains_default(self) -> None:
        """Preserve original and AO-only as separate comparison policies."""
        policy = grouping.policy_document()
        self.assertEqual(grouping.BASELINE_STRATEGIES, ("original", "exact-ao"))
        self.assertEqual(policy["candidate_strategy"], "ao-risk")
        self.assertEqual(policy["baseline_strategies"], ["original", "exact-ao"])
        self.assertEqual(policy["default_strategy"], "original")
        self.assertFalse(policy["production_or_default_change"])
        self.assertFalse(policy["deployment_claim"])

    def test_unknown_scheduling_metadata_is_rejected(self) -> None:
        """Reject fields outside the preregistered feature allowlist."""
        document = _manifest()
        systems = document["systems"]
        systems[0]["scheduling_metadata"]["temperature"] = 300
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "unknown"):
            grouping.parse_scheduling_manifest(document)

    def test_outcome_fields_are_rejected_inside_scheduling_metadata(self) -> None:
        """Reject outcome-shaped keys before constructing risk annotations."""
        for field in (
            "scc_iterations",
            "status",
            "scc_converged",
            "energy_hartree",
            "outcome_label",
        ):
            with self.subTest(field=field):
                document = _manifest()
                document["systems"][0]["scheduling_metadata"][field] = 1
                with self.assertRaisesRegex(
                    grouping.ConvergenceGroupingError, "outcome field"
                ):
                    grouping.parse_scheduling_manifest(document)

    def test_outcome_fields_are_rejected_outside_scheduling_metadata_too(self) -> None:
        """Reject measured outcomes anywhere in the scheduling manifest."""
        document = _manifest()
        document["systems"][0]["status"] = "converged"
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "outcome field"):
            grouping.parse_scheduling_manifest(document)

    def test_manifest_requires_molecule_group_identity(self) -> None:
        """Require the manifest identity needed for group-held-out splitting."""
        document = _manifest()
        del document["systems"][0]["molecule_group_id"]
        with self.assertRaisesRegex(
            grouping.ConvergenceGroupingError, "molecule_group_id"
        ):
            grouping.parse_scheduling_manifest(document)

    def test_manifest_requires_explicit_spin_metadata(self) -> None:
        """Refuse to infer spin metadata from atoms or molecular charge."""
        document = _manifest()
        metadata = document["systems"][0]["scheduling_metadata"]
        metadata.pop("spin_channels")
        metadata.pop("unpaired_electrons")
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "explicit spin"):
            grouping.parse_scheduling_manifest(document)

    def test_manifest_rejects_duplicate_case_ids(self) -> None:
        """Require unique case IDs for unambiguous downstream mapping."""
        duplicate = _case("same", "g1")
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "unique"):
            grouping.parse_scheduling_manifest(
                _manifest([duplicate, _case("same", "g2")])
            )

    def test_manifest_requires_a_finite_positive_exact_ao_count(self) -> None:
        """Reject nonintegral, nonpositive, boolean, and infinite AO counts."""
        for ao_count in (0, -1, 2.5, True, math.inf):
            with self.subTest(ao_count=ao_count):
                document = _manifest()
                document["systems"][0]["scheduling_metadata"]["exact_ao_count"] = (
                    ao_count
                )
                with self.assertRaisesRegex(
                    grouping.ConvergenceGroupingError, "exact_ao_count"
                ):
                    grouping.parse_scheduling_manifest(document)

    def test_manifest_rejects_nonfinite_charge_and_boolean_charge(self) -> None:
        """Reject non-finite charge and booleans masquerading as numbers."""
        for charge in (math.nan, math.inf, -math.inf, True):
            with self.subTest(charge=charge):
                document = _manifest()
                document["systems"][0]["scheduling_metadata"]["molecular_charge"] = (
                    charge
                )
                with self.assertRaisesRegex(
                    grouping.ConvergenceGroupingError, "molecular_charge"
                ):
                    grouping.parse_scheduling_manifest(document)

    def test_manifest_rejects_invalid_atomic_numbers_and_spin_fields(self) -> None:
        """Validate element IDs and explicit nonnegative spin values."""
        for atomic_numbers in ([], [0], [119], [6, True], [6, 1.0]):
            with self.subTest(atomic_numbers=atomic_numbers):
                document = _manifest()
                document["systems"][0]["scheduling_metadata"]["atomic_numbers"] = (
                    atomic_numbers
                )
                with self.assertRaisesRegex(
                    grouping.ConvergenceGroupingError, "atomic_numbers"
                ):
                    grouping.parse_scheduling_manifest(document)
        for field, value in (
            ("spin_channels", True),
            ("spin_channels", 0),
            ("unpaired_electrons", -1),
        ):
            with self.subTest(field=field, value=value):
                document = _manifest()
                document["systems"][0]["scheduling_metadata"][field] = value
                with self.assertRaisesRegex(grouping.ConvergenceGroupingError, field):
                    grouping.parse_scheduling_manifest(document)


class FrozenSplitTests(unittest.TestCase):
    """Prove deterministic group-held-out assignment and stable hash identity."""

    def test_conformers_and_near_duplicates_share_one_group_partition(self) -> None:
        """Assign all conformers represented by one group ID together."""
        plan = grouping.build_freeze_plan(_manifest())
        group_entries = {
            item["molecule_group_id"]: item for item in plan["split"]["groups"]
        }
        self.assertEqual(
            group_entries["alkane-a"]["case_ids"],
            ["alkane-a-conf-1", "alkane-a-conf-2"],
        )
        case_partitions = {
            case_id: group["partition"]
            for group in group_entries.values()
            for case_id in group["case_ids"]
        }
        self.assertEqual(
            case_partitions["alkane-a-conf-1"],
            case_partitions["alkane-a-conf-2"],
        )
        partitions = {item["partition"] for item in group_entries.values()}
        self.assertEqual(partitions, {"calibration", "holdout"})

    def test_split_and_hashes_are_stable_under_manifest_row_reordering(self) -> None:
        """Keep frozen identities unchanged when manifest rows are reordered."""
        manifest = _manifest()
        first = grouping.build_freeze_plan(manifest)
        reordered = dict(manifest, systems=list(reversed(manifest["systems"])))
        second = grouping.build_freeze_plan(reordered)
        self.assertEqual(first["policy_sha256"], second["policy_sha256"])
        self.assertEqual(first["split_sha256"], second["split_sha256"])
        self.assertEqual(
            first["input_identity_sha256"], second["input_identity_sha256"]
        )
        self.assertEqual(first["freeze_plan_sha256"], second["freeze_plan_sha256"])
        first_annotations = grouping.annotate_scheduling_inputs(manifest)
        second_annotations = grouping.annotate_scheduling_inputs(reordered)
        self.assertEqual(tuple(first_annotations), tuple(second_annotations))

    def test_policy_and_split_are_fixed_while_input_identity_tracks_input_changes(
        self,
    ) -> None:
        """Separate policy/split identities from changed scheduling input."""
        original = grouping.build_freeze_plan(_manifest())
        changed_manifest = _manifest()
        changed_manifest["systems"][0]["scheduling_metadata"]["molecular_charge"] = 1
        changed = grouping.build_freeze_plan(changed_manifest)
        self.assertEqual(original["policy_sha256"], changed["policy_sha256"])
        self.assertEqual(original["split_sha256"], changed["split_sha256"])
        self.assertNotEqual(
            original["input_identity_sha256"], changed["input_identity_sha256"]
        )

    def test_split_uses_fixed_sha256_seed_and_contains_no_measurement_fields(
        self,
    ) -> None:
        """Serialize fixed seed hashes without training or holdout results."""
        plan = grouping.build_freeze_plan(_manifest())
        self.assertEqual(plan["split"]["algorithm"], "sha256-group-rank-v1")
        self.assertEqual(plan["split"]["seed"], grouping.SPLIT_SEED)
        self.assertEqual(len(plan["split_sha256"]), 64)
        self.assertEqual(len(plan["policy_sha256"]), 64)
        self.assertEqual(len(plan["input_identity_sha256"]), 64)
        self.assertNotIn("training_results", plan)
        self.assertNotIn("holdout_measurements", plan)
        self.assertEqual(plan["status"], "preregistered_only")

    def test_split_rejects_single_group_and_empty_input(self) -> None:
        """Require enough independent molecule groups for a holdout split."""
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "at least two"):
            grouping.build_freeze_plan(
                _manifest([_case("one-a", "same"), _case("one-b", "same")])
            )
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "nonempty"):
            grouping.parse_scheduling_manifest({"schema_version": 1, "systems": []})

    def test_calibration_only_input_accepts_calibration_groups_and_labels(self) -> None:
        """Accept labels only through the purpose-tagged calibration boundary."""
        plan = grouping.build_freeze_plan(_manifest())
        calibration_group = next(
            group
            for group in plan["split"]["groups"]
            if group["partition"] == "calibration"
        )
        case_id = calibration_group["case_ids"][0]
        result = grouping.validate_calibration_input(
            {
                "schema_version": 1,
                "purpose": "calibration-only",
                "records": [
                    {
                        "case_id": case_id,
                        "molecule_group_id": calibration_group["molecule_group_id"],
                        "scc_iterations": 17,
                        "scc_converged": False,
                        "status": 5,
                    }
                ],
            },
            plan,
        )
        self.assertEqual(result[0]["case_id"], case_id)

    def test_calibration_only_input_rejects_holdout_group_ids(self) -> None:
        """Prevent holdout labels from entering a calibration dataset."""
        plan = grouping.build_freeze_plan(_manifest())
        holdout_group = next(
            group
            for group in plan["split"]["groups"]
            if group["partition"] == "holdout"
        )
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "holdout"):
            grouping.validate_calibration_input(
                {
                    "schema_version": 1,
                    "purpose": "calibration-only",
                    "records": [
                        {
                            "case_id": holdout_group["case_ids"][0],
                            "molecule_group_id": holdout_group["molecule_group_id"],
                            "scc_iterations": 12,
                            "scc_converged": False,
                            "status": 5,
                        }
                    ],
                },
                plan,
            )

    def test_calibration_input_rejects_wrong_purpose_unknown_case_and_bad_outcomes(
        self,
    ) -> None:
        """Reject mislabeled, mismapped, or non-finite calibration records."""
        plan = grouping.build_freeze_plan(_manifest())
        calibration_group = next(
            group
            for group in plan["split"]["groups"]
            if group["partition"] == "calibration"
        )
        valid = {
            "schema_version": 1,
            "purpose": "calibration-only",
            "records": [
                {
                    "case_id": calibration_group["case_ids"][0],
                    "molecule_group_id": calibration_group["molecule_group_id"],
                    "scc_iterations": 4,
                    "scc_converged": True,
                    "status": 0,
                }
            ],
        }
        wrong_purpose = dict(valid, purpose="scheduling")
        with self.assertRaisesRegex(
            grouping.ConvergenceGroupingError, "calibration-only"
        ):
            grouping.validate_calibration_input(wrong_purpose, plan)
        mismatched = json.loads(json.dumps(valid))
        mismatched["records"][0]["case_id"] = "not-in-frozen-input"
        with self.assertRaisesRegex(
            grouping.ConvergenceGroupingError, "frozen group mapping"
        ):
            grouping.validate_calibration_input(mismatched, plan)
        nonfinite = json.loads(json.dumps(valid))
        nonfinite["records"][0]["scc_iterations"] = math.nan
        with self.assertRaisesRegex(
            grouping.ConvergenceGroupingError, "scc_iterations"
        ):
            grouping.validate_calibration_input(nonfinite, plan)
        invalid_boolean = json.loads(json.dumps(valid))
        invalid_boolean["records"][0]["scc_converged"] = 1
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "scc_converged"):
            grouping.validate_calibration_input(invalid_boolean, plan)
        hidden_group = json.loads(json.dumps(valid))
        hidden_group["records"][0]["holdout_group_id"] = "holdout-family"
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "unknown"):
            grouping.validate_calibration_input(hidden_group, plan)

    def test_calibration_guard_rejects_tampered_serialized_hashes(self) -> None:
        """Verify frozen hashes before trusting a calibration partition."""
        plan = grouping.build_freeze_plan(_manifest())
        plan["split"]["groups"][0]["partition"] = "holdout"
        calibration_input = {
            "schema_version": 1,
            "purpose": "calibration-only",
            "records": [
                {
                    "case_id": "alkane-a-conf-1",
                    "molecule_group_id": "alkane-a",
                    "scc_iterations": 5,
                    "scc_converged": False,
                    "status": 5,
                }
            ],
        }
        with self.assertRaisesRegex(grouping.ConvergenceGroupingError, "split hash"):
            grouping.validate_calibration_input(calibration_input, plan)


class FreezePlanCliTests(unittest.TestCase):
    """Exercise the serialized preregistration command without measurements."""

    def test_missing_prerequisites_do_not_become_empirical_no_go(self) -> None:
        """Keep environmental blockers separate from a completed policy result."""
        protocol = grouping.build_freeze_plan(_manifest())["evaluation_protocol"]
        self.assertIn("unverified_or_blocked", protocol)
        self.assertIn("completed eligible evaluation", " ".join(protocol["no_go"]))
        self.assertNotIn("unavailable", " ".join(protocol["no_go"]))
        self.assertIn(
            "do not permit issue closure", " ".join(protocol["unverified_or_blocked"])
        )

    def test_freeze_plan_cli_serializes_hashes_and_refuses_overwrite(self) -> None:
        """Write preregistration hashes once without attaching outcomes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "input.json"
            output_path = root / "freeze-plan.json"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            self.assertEqual(
                grouping.main(
                    [
                        "freeze-plan",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            plan = json.loads(output_path.read_text(encoding="utf-8"))
            for field in ("policy_sha256", "split_sha256", "input_identity_sha256"):
                self.assertEqual(len(plan[field]), 64)
            self.assertNotIn("training_results", plan)
            self.assertNotIn("holdout_measurements", plan)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                grouping.main(
                    [
                        "freeze-plan",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertEqual(error.exception.code, 2)

    def test_strict_json_loader_rejects_duplicate_keys_and_nan(self) -> None:
        """Reject duplicate object keys and nonstandard NaN tokens on read."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            for payload in (
                '{"schema_version":1,"schema_version":1,"systems":[]}',
                '{"schema_version":1,"systems":[],"value":NaN}',
            ):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(grouping.ConvergenceGroupingError):
                    grouping.load_json_document(path)


if __name__ == "__main__":
    unittest.main()
