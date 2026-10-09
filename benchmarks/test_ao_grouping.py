"""Focused tests for finite-list exact-AO planning and output restoration."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from benchmarks import ao_grouping, run
from tools.conformance import xtbloom_conformance as conformance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GFN2_PARAMETERS = REPOSITORY_ROOT / "data" / "parameters" / "gfn2.json"


class AOPlanTests(unittest.TestCase):
    """Cover exact shell semantics, stable grouping, and canonical restoration."""

    def _make_risk_plan(
        self,
        case_ids: tuple[str, ...],
        max_batch_size: int,
        counts: dict[str, int],
        bands: dict[str, str],
        *,
        basis_sha256: str = "a" * 64,
        risk_policy_sha256: str = "b" * 64,
        risk_freeze_plan_sha256: str = "c" * 64,
    ) -> ao_grouping.AOGroupingPlan:
        """Build a risk plan with valid, fixed provenance defaults."""
        return ao_grouping.make_plan(
            case_ids,
            max_batch_size,
            strategy="ao-risk",
            ao_counts_by_case_id=counts,
            basis_sha256=basis_sha256,
            risk_bands_by_case_id=bands,
            risk_policy_sha256=risk_policy_sha256,
            risk_freeze_plan_sha256=risk_freeze_plan_sha256,
        )

    def test_generated_gfn2_shells_define_exact_ao_degeneracies(self) -> None:
        """Use generated shell angular momenta for spatial AO counts."""
        basis_counts, basis_sha256 = ao_grouping.load_gfn2_basis_ao_counts(
            GFN2_PARAMETERS
        )
        self.assertEqual(basis_counts[1], 1)
        self.assertEqual(basis_counts[2], 4)
        self.assertEqual(basis_counts[6], 4)
        self.assertEqual(basis_counts[26], 9)
        self.assertEqual(ao_grouping.count_gfn2_aos([1, 6, 8], basis_counts), 9)
        self.assertEqual(len(basis_sha256), 64)

    def test_empty_and_single_case_plans(self) -> None:
        """Handle empty input and retain the identity of a singleton."""
        empty = ao_grouping.make_plan((), 64)
        self.assertEqual(empty.ordered_case_ids, ())
        self.assertEqual(empty.batches, ())
        single = ao_grouping.make_plan(("one",), 64, "exact-ao", {"one": 4}, "a" * 64)
        self.assertEqual(single.batches[0].case_ids, ("one",))
        self.assertEqual(single.canonical_index_by_case_id, {"one": 0})

        risk_empty = self._make_risk_plan((), 64, {}, {})
        self.assertEqual(risk_empty.ordered_case_ids, ())
        self.assertEqual(risk_empty.batches, ())
        self.assertEqual(risk_empty.risk_bands_by_case_id, ())
        risk_single = self._make_risk_plan(
            ("one",), 64, {"one": 4}, {"one": "elevated"}
        )
        self.assertEqual(risk_single.batches[0].case_ids, ("one",))
        self.assertEqual(risk_single.batches[0].risk_band, "elevated")
        self.assertEqual(risk_single.risk_bands_by_case_id, (("one", "elevated"),))

    def test_risk_buckets_keep_stable_ties_caps_tails_and_rare_groups(self) -> None:
        """Keep every case in stable AO/risk buckets, including short tails."""
        case_ids = (
            "guarded-a",
            "low-a",
            "high-a",
            "elevated-a",
            "guarded-b",
            "high-b",
            "low-b",
            "high-c",
        )
        plan = self._make_risk_plan(
            case_ids,
            2,
            dict.fromkeys(case_ids, 8),
            {
                "guarded-a": "guarded",
                "low-a": "low",
                "high-a": "high",
                "elevated-a": "elevated",
                "guarded-b": "guarded",
                "high-b": "high",
                "low-b": "low",
                "high-c": "high",
            },
        )
        self.assertEqual(
            plan.ordered_case_ids,
            (
                "high-a",
                "high-b",
                "high-c",
                "elevated-a",
                "guarded-a",
                "guarded-b",
                "low-a",
                "low-b",
            ),
        )
        self.assertEqual(
            [batch.case_ids for batch in plan.batches],
            [
                ("high-a", "high-b"),
                ("high-c",),
                ("elevated-a",),
                ("guarded-a", "guarded-b"),
                ("low-a", "low-b"),
            ],
        )
        self.assertEqual(
            [batch.risk_band for batch in plan.batches],
            ["high", "high", "elevated", "guarded", "low"],
        )
        self.assertEqual(
            [index for batch in plan.batches for index in batch.canonical_indices],
            [2, 5, 7, 3, 0, 4, 1, 6],
        )
        self.assertEqual(
            tuple(case_id for batch in plan.batches for case_id in batch.case_ids),
            plan.ordered_case_ids,
        )
        self.assertTrue(all(len(batch.case_ids) <= 2 for batch in plan.batches))

    def test_risk_sort_uses_ao_before_band_and_splits_each_ao_bucket(self) -> None:
        """Order distinct AO buckets first even when risk bands reverse them."""
        case_ids = ("largest-low", "smallest-high", "middle-guarded", "next-elevated")
        plan = self._make_risk_plan(
            case_ids,
            8,
            {
                "largest-low": 12,
                "smallest-high": 3,
                "middle-guarded": 9,
                "next-elevated": 6,
            },
            {
                "largest-low": "low",
                "smallest-high": "high",
                "middle-guarded": "guarded",
                "next-elevated": "elevated",
            },
        )
        self.assertEqual(
            plan.ordered_case_ids,
            ("smallest-high", "next-elevated", "middle-guarded", "largest-low"),
        )
        self.assertEqual(
            [(batch.ao_count, batch.risk_band) for batch in plan.batches],
            [(3, "high"), (6, "elevated"), (9, "guarded"), (12, "low")],
        )
        self.assertTrue(all(len(batch.case_ids) == 1 for batch in plan.batches))

    def test_risk_plan_hash_uses_schema_two_and_records_provenance(self) -> None:
        """Include ordered annotations and both frozen hashes in plan identity."""
        case_ids = ("guarded", "high", "low")
        counts = dict.fromkeys(case_ids, 4)
        bands = {"guarded": "guarded", "high": "high", "low": "low"}
        plan = self._make_risk_plan(case_ids, 2, counts, bands)
        expected_document = {
            "schema_version": 2,
            "strategy": "ao-risk",
            "max_batch_size": 2,
            "original_case_ids": ["guarded", "high", "low"],
            "ao_counts_by_case_id": [["guarded", 4], ["high", 4], ["low", 4]],
            "basis_sha256": "a" * 64,
            "batches": [
                {
                    "case_ids": ["high"],
                    "canonical_indices": [1],
                    "ao_count": 4,
                    "risk_band": "high",
                },
                {
                    "case_ids": ["guarded"],
                    "canonical_indices": [0],
                    "ao_count": 4,
                    "risk_band": "guarded",
                },
                {
                    "case_ids": ["low"],
                    "canonical_indices": [2],
                    "ao_count": 4,
                    "risk_band": "low",
                },
            ],
            "risk_bands_by_case_id": [
                ["guarded", "guarded"],
                ["high", "high"],
                ["low", "low"],
            ],
            "risk_policy_sha256": "b" * 64,
            "risk_freeze_plan_sha256": "c" * 64,
        }
        expected_hash = hashlib.sha256(
            json.dumps(
                expected_document, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.assertEqual(plan.plan_sha256, expected_hash)
        self.assertEqual(plan.risk_bands_by_case_id, tuple(bands.items()))
        self.assertEqual(plan.risk_policy_sha256, "b" * 64)
        self.assertEqual(plan.risk_freeze_plan_sha256, "c" * 64)
        self.assertNotEqual(
            plan.plan_sha256,
            self._make_risk_plan(
                case_ids,
                2,
                counts,
                bands,
                risk_policy_sha256="d" * 64,
            ).plan_sha256,
        )
        self.assertNotEqual(
            plan.plan_sha256,
            self._make_risk_plan(
                case_ids,
                2,
                counts,
                {"guarded": "low", "high": "high", "low": "guarded"},
            ).plan_sha256,
        )
        self.assertNotEqual(
            plan.plan_sha256,
            self._make_risk_plan(
                case_ids,
                2,
                counts,
                bands,
                risk_freeze_plan_sha256="d" * 64,
            ).plan_sha256,
        )

    def test_risk_metadata_is_required_exact_and_strictly_typed(self) -> None:
        """Reject incomplete IDs, invalid labels, malformed hashes, and bool counts."""
        base = {
            "case_ids": ("a", "b"),
            "max_batch_size": 2,
            "strategy": "ao-risk",
            "ao_counts_by_case_id": {"a": 2, "b": 4},
            "basis_sha256": "a" * 64,
            "risk_bands_by_case_id": {"a": "high", "b": "low"},
            "risk_policy_sha256": "b" * 64,
            "risk_freeze_plan_sha256": "c" * 64,
        }
        invalid_updates = (
            ("missing counts", {"ao_counts_by_case_id": None}),
            ("missing count ID", {"ao_counts_by_case_id": {"a": 2}}),
            ("boolean count", {"ao_counts_by_case_id": {"a": True, "b": 4}}),
            ("missing basis hash", {"basis_sha256": None}),
            ("uppercase basis hash", {"basis_sha256": "A" * 64}),
            ("missing risk bands", {"risk_bands_by_case_id": None}),
            ("missing risk ID", {"risk_bands_by_case_id": {"a": "high"}}),
            (
                "unexpected risk ID",
                {"risk_bands_by_case_id": {"a": "high", "extra": "low"}},
            ),
            (
                "invalid risk label",
                {"risk_bands_by_case_id": {"a": "HIGH", "b": "low"}},
            ),
            ("boolean risk label", {"risk_bands_by_case_id": {"a": True, "b": "low"}}),
            ("missing policy hash", {"risk_policy_sha256": None}),
            ("uppercase policy hash", {"risk_policy_sha256": "B" * 64}),
            ("short policy hash", {"risk_policy_sha256": "b" * 63}),
            ("boolean policy hash", {"risk_policy_sha256": True}),
            ("missing freeze hash", {"risk_freeze_plan_sha256": None}),
            ("nonhex freeze hash", {"risk_freeze_plan_sha256": "g" * 64}),
            ("short freeze hash", {"risk_freeze_plan_sha256": "c" * 63}),
            ("boolean freeze hash", {"risk_freeze_plan_sha256": False}),
            ("boolean cap", {"max_batch_size": True}),
        )
        for label, updates in invalid_updates:
            with (
                self.subTest(label=label),
                self.assertRaises(ao_grouping.AOGroupingError),
            ):
                ao_grouping.make_plan(**(base | updates))

    def test_legacy_strategies_reject_risk_metadata(self) -> None:
        """Prevent risk inputs from being silently ignored by legacy plans."""
        metadata = (
            {"risk_bands_by_case_id": {}},
            {"risk_policy_sha256": "a" * 64},
            {"risk_freeze_plan_sha256": "b" * 64},
        )
        for strategy in ("original", "exact-ao"):
            for risk_metadata in metadata:
                arguments: dict[str, Any] = {
                    "case_ids": ("case",),
                    "max_batch_size": 1,
                    "strategy": strategy,
                    **risk_metadata,
                }
                if strategy == "exact-ao":
                    arguments.update(
                        ao_counts_by_case_id={"case": 3}, basis_sha256="c" * 64
                    )
                with (
                    self.subTest(strategy=strategy, metadata=risk_metadata),
                    self.assertRaisesRegex(
                        ao_grouping.AOGroupingError, "risk metadata"
                    ),
                ):
                    ao_grouping.make_plan(**arguments)

    def test_ao_risk_scatter_keeps_every_peer_and_failed_nan_in_input_order(
        self,
    ) -> None:
        """Restore the full risk-sorted batch result set through the shared scatter."""
        case_ids = ("failed", "peer-b", "peer-a")
        plan = self._make_risk_plan(
            case_ids,
            1,
            {"failed": 7, "peer-b": 3, "peer-a": 3},
            {"failed": "high", "peer-b": "low", "peer-a": "high"},
        )
        records = []
        for batch in plan.batches:
            for case_id in batch.case_ids:
                is_failed = case_id == "failed"
                records.append(
                    {
                        "case_id": case_id,
                        "energy_hartree": float("nan") if is_failed else 1.25,
                        "scc_converged": 0 if is_failed else 1,
                        "status": 5 if is_failed else 0,
                    }
                )
        restored = ao_grouping.scatter_case_results(plan, records)
        self.assertEqual([record["case_id"] for record in restored], list(case_ids))
        self.assertEqual([record["original_index"] for record in restored], [0, 1, 2])
        self.assertTrue(math.isnan(restored[0]["energy_hartree"]))
        self.assertEqual(restored[0]["status"], 5)
        self.assertEqual(
            [record["energy_hartree"] for record in restored[1:]], [1.25, 1.25]
        )

    def test_legacy_exact_ao_hash_remains_schema_one_regression(self) -> None:
        """Keep the pre-risk exact-AO hash bytes and default risk fields stable."""
        plan = ao_grouping.make_plan(
            ("large", "small-b", "medium", "small-a"),
            2,
            "exact-ao",
            {"large": 9, "small-b": 3, "medium": 6, "small-a": 3},
            "c" * 64,
        )
        self.assertEqual(
            plan.plan_sha256,
            "01d38a643fb5d4701d5bd3d65a2797b181e0ad817cf3ebae41f418622f8d1e37",
        )
        self.assertEqual(plan.risk_bands_by_case_id, ())
        self.assertIsNone(plan.risk_policy_sha256)
        self.assertIsNone(plan.risk_freeze_plan_sha256)

        original = ao_grouping.make_plan(("second", "first"), 1)
        self.assertEqual(
            original.plan_sha256,
            "8418ecc0c59b8b366533ca685bfb686063c7e0e45c8074baee4efeeedebcccaf",
        )
        self.assertIsNone(original.batches[0].risk_band)

    def test_same_and_different_ao_plans_are_stable_and_cap_bounded(self) -> None:
        """Keep exact-AO groups deterministic and split them at the cap."""
        same = ao_grouping.make_plan(
            ("a", "b", "c", "d", "e"),
            2,
            "exact-ao",
            dict.fromkeys(("a", "b", "c", "d", "e"), 8),
            "b" * 64,
        )
        self.assertEqual(
            [batch.case_ids for batch in same.batches],
            [("a", "b"), ("c", "d"), ("e",)],
        )
        self.assertEqual([len(batch.case_ids) for batch in same.batches], [2, 2, 1])

        mixed = ao_grouping.make_plan(
            ("large", "small-b", "medium", "small-a"),
            2,
            "exact-ao",
            {"large": 9, "small-b": 3, "medium": 6, "small-a": 3},
            "c" * 64,
        )
        self.assertEqual(
            mixed.ordered_case_ids, ("small-b", "small-a", "medium", "large")
        )
        self.assertEqual([batch.ao_count for batch in mixed.batches], [3, 6, 9])
        self.assertEqual(
            ao_grouping.make_plan(
                ("large", "small-b", "medium", "small-a"),
                2,
                "exact-ao",
                {"large": 9, "small-b": 3, "medium": 6, "small-a": 3},
                "c" * 64,
            ).plan_sha256,
            mixed.plan_sha256,
        )
        self.assertNotEqual(
            mixed.plan_sha256,
            ao_grouping.make_plan(
                ("large", "small-b", "medium", "small-a"),
                3,
                "exact-ao",
                {"large": 9, "small-b": 3, "medium": 6, "small-a": 3},
                "c" * 64,
            ).plan_sha256,
        )

    def test_duplicate_and_incomplete_ids_are_rejected(self) -> None:
        """Reject non-unique case identities and incomplete AO metadata."""
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "unique"):
            ao_grouping.make_plan(("same", "same"), 2)
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "match"):
            ao_grouping.make_plan(("a", "b"), 2, "exact-ao", {"a": 3}, "d" * 64)
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "basis SHA"):
            ao_grouping.make_plan(("a",), 2, "exact-ao", {"a": 3})

    def test_batch_slices_keep_status_nan_and_variable_output_lengths(self) -> None:
        """Slice ragged properties and preserve status and NaN values."""
        records = ao_grouping.split_batch_results(
            ("a", "b"),
            [0, 1, 3],
            [0, 2, 3],
            {
                "energies_hartree": [1.0, 2.0],
                "scc_iterations": [4, 7],
                "scc_converged": [1, 0],
                "per_system_status": [0, 5],
                "forces_hartree_per_bohr": list(range(9)),
                "atomic_charges_e": [0.1, 0.2, 0.3],
                "point_charge_forces_hartree_per_bohr": list(range(9)),
            },
        )
        self.assertEqual(records[0]["forces_hartree_per_bohr"], [0.0, 1.0, 2.0])
        self.assertEqual(
            records[1]["forces_hartree_per_bohr"], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        )
        self.assertEqual(records[0]["atomic_charges_e"], [0.1])
        self.assertEqual(records[1]["atomic_charges_e"], [0.2, 0.3])
        self.assertEqual(
            records[0]["point_charge_forces_hartree_per_bohr"],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        self.assertEqual(
            records[1]["point_charge_forces_hartree_per_bohr"], [6.0, 7.0, 8.0]
        )
        self.assertEqual(records[1]["status"], 5)
        self.assertEqual(records[1]["scc_converged"], 0)
        with self.assertRaisesRegex(
            ao_grouping.AOGroupingError, "required result field"
        ):
            ao_grouping.split_batch_results(
                ("a",),
                [0, 1],
                [0, 0],
                {
                    "energies_hartree": [1.0],
                    "scc_iterations": [1],
                    "scc_converged": [1],
                    "per_system_status": [0],
                },
                required_outputs=("atomic_charges_e",),
            )

        failing = dict(records[1], energy_hartree=float("nan"))
        plan = ao_grouping.make_plan(("a", "b"), 2)
        restored = ao_grouping.scatter_case_results(plan, (failing, records[0]))
        self.assertEqual([item["case_id"] for item in restored], ["a", "b"])
        self.assertEqual([item["original_index"] for item in restored], [0, 1])
        self.assertTrue(math.isnan(restored[1]["energy_hartree"]))

    def test_scatter_rejects_missing_duplicate_and_unexpected_results(self) -> None:
        """Require exactly one result for every planned case ID."""
        plan = ao_grouping.make_plan(("a", "b"), 1)
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "missing"):
            ao_grouping.scatter_case_results(plan, [{"case_id": "a"}])
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "duplicate"):
            ao_grouping.scatter_case_results(
                plan, [{"case_id": "a"}, {"case_id": "a"}, {"case_id": "b"}]
            )
        with self.assertRaisesRegex(ao_grouping.AOGroupingError, "unexpected"):
            ao_grouping.scatter_case_results(plan, [{"case_id": "a"}, {"case_id": "x"}])


class FiniteRunnerTests(unittest.TestCase):
    """Exercise CLI constraints and a mocked public-runner grouping sweep."""

    def _run_main_with_finite_row(
        self, row: dict[str, Any], flags: tuple[str, ...]
    ) -> tuple[int, dict[str, Any]]:
        """Run the finite CLI with a mocked cell and real temporary provenance."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "case.coord"
            input_bytes = b"canonical input fixture\n"
            input_path.write_bytes(input_bytes)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"cases": [{"id": "case", "input": str(input_path)}]}),
                encoding="utf-8",
            )
            plan = ao_grouping.make_plan(("case",), 1)
            documents: list[dict[str, Any]] = []
            patches = mock.patch.multiple(
                run,
                environment_metadata=mock.DEFAULT,
                finite_case_plan=mock.DEFAULT,
                benchmark_finite_xtbloom_cell=mock.DEFAULT,
                write_json=mock.DEFAULT,
                write_csv=mock.DEFAULT,
            )
            with patches as mocked:
                mocked["environment_metadata"].return_value = {"test": True}
                mocked["finite_case_plan"].return_value = (plan, 0.0)
                mocked["benchmark_finite_xtbloom_cell"].return_value = row
                mocked["write_json"].side_effect = lambda _path, document: (
                    documents.append(document)
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = run.main(
                        (
                            "--library",
                            str(root / "libxtbloom.so"),
                            "--manifest",
                            str(manifest_path),
                            "--case-ids",
                            "case",
                            "--engines",
                            "xtbloom",
                            "--backends",
                            "cpu",
                            "--ao-grouping",
                            "original",
                            "--batch-sizes",
                            "1",
                            "--properties",
                            "energy",
                            "--warmups",
                            "0",
                            "--repetitions",
                            "1",
                            "--output-json",
                            str(root / "report.json"),
                            "--output-csv",
                            str(root / "report.csv"),
                            *flags,
                        )
                    )
            self.assertEqual(len(documents), 1)
            provenance = documents[0]["provenance"]
            self.assertEqual(
                provenance["selected_inputs"][0]["sha256"],
                hashlib.sha256(input_bytes).hexdigest(),
            )
            self.assertEqual(
                provenance["manifest"]["sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertNotIn(input_bytes.decode().strip(), json.dumps(provenance))
            return exit_code, documents[0]

    def test_json_writer_tags_nonfinite_values_for_strict_reconstruction(self) -> None:
        """Write strict JSON while preserving reconstructable NaN semantics."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            run.write_json(
                path,
                {
                    "case_results": [
                        {
                            "case_id": "failed",
                            "energy_hartree": float("nan"),
                            "forces_hartree_per_bohr": [float("inf"), float("-inf")],
                            "status": 5,
                        }
                    ],
                },
            )

            def reject_nonstandard_constant(value: str) -> None:
                raise ValueError(f"non-standard JSON constant: {value}")

            encoded = path.read_text(encoding="utf-8")
            decoded = json.loads(
                encoded,
                parse_constant=reject_nonstandard_constant,
                object_hook=run.restore_json_safe_value,
            )
        self.assertIn(run.NONFINITE_JSON_TAG, encoded)
        failed_result = decoded["case_results"][0]
        self.assertEqual(failed_result["status"], 5)
        self.assertTrue(math.isnan(failed_result["energy_hartree"]))
        self.assertEqual(
            failed_result["forces_hartree_per_bohr"],
            [float("inf"), float("-inf")],
        )

    def test_require_available_turns_unavailable_finite_row_nonzero(self) -> None:
        """Keep unavailable rows but fail qualification when explicitly required."""
        row = {
            "availability": "unavailable",
            "unavailable_reason": "test backend unavailable",
        }
        exit_code, report = self._run_main_with_finite_row(
            row, ("--require-available",)
        )
        self.assertEqual(exit_code, 3)
        self.assertTrue(report["protocol"]["require_available"])
        self.assertEqual(report["rows"][0]["availability"], "unavailable")
        default_exit, default_report = self._run_main_with_finite_row(row, ())
        self.assertEqual(default_exit, 0)
        self.assertFalse(default_report["protocol"]["require_available"])

    def test_finite_runner_requires_cuda_host_and_explicit_xtbloom(self) -> None:
        """Limit finite grouping to one xTBloom backend and host CUDA mode."""
        arguments = run.build_parser().parse_args(
            [
                "--library",
                "/tmp/libxtbloom.so",
                "--case-ids",
                "h3_plus,ketene",
                "--engines",
                "xtbloom",
                "--backends",
                "cuda",
                "--cuda-memory-modes",
                "host",
                "--ao-grouping",
                "exact-ao",
            ]
        )
        run.validate_args(arguments)
        self.assertEqual(arguments.ao_grouping, "exact-ao")

        arguments.cuda_memory_modes = ("device",)
        with self.assertRaisesRegex(run.BenchmarkError, "host descriptors"):
            run.validate_args(arguments)
        arguments.backends = ("cpu",)
        run.validate_args(arguments)

    def test_case_id_file_preserves_order_and_ignores_comments(self) -> None:
        """Parse ordered finite IDs while skipping comments and blank lines."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.txt"
            path.write_text("# frozen selection\nalpha\n\nbeta\n", encoding="utf-8")
            self.assertEqual(run.read_case_id_file(path), ("alpha", "beta"))

    def test_exact_plan_uses_publicly_parsed_manifest_atoms(self) -> None:
        """Derive AO counts from the actual manifest input structures."""
        manifest = conformance.load_json(conformance.DEFAULT_MANIFEST)
        cases = {
            case["id"]: case for case in conformance.selected_cases(manifest, None)
        }
        arguments = SimpleNamespace(
            ao_grouping="exact-ao", manifest=conformance.DEFAULT_MANIFEST
        )
        plan, planning_ms = run.finite_case_plan(
            arguments, manifest, cases, ("ketene", "h3_plus"), 4
        )
        counts = dict(plan.ao_counts_by_case_id)
        self.assertEqual(counts["h3_plus"], 3)
        self.assertGreater(counts["ketene"], counts["h3_plus"])
        self.assertEqual(plan.ordered_case_ids, ("h3_plus", "ketene"))
        self.assertGreaterEqual(planning_ms, 0.0)

    def test_finite_sweep_restores_peer_failure_outputs_and_uses_strict_fresh(
        self,
    ) -> None:
        """Retain failed-peer NaNs and use strict FRESH without filtering."""
        expected_bad = {
            "energy_hartree": 2.0,
            "forces_hartree_per_bohr": [0.0] * 6,
            "partial_charges_e": [0.0, 0.0],
            "point_charge_forces_hartree_per_bohr": [0.0] * 3,
        }
        expected_good = {
            "energy_hartree": 1.0,
            "forces_hartree_per_bohr": [0.0] * 3,
            "partial_charges_e": [0.0],
        }
        cases = {
            "bad": {
                "id": "bad",
                "atom_count": 2,
                "point_count": 1,
                "expected": expected_bad,
                "mock": {
                    "energy": float("nan"),
                    "forces": [float("nan")] * 6,
                    "charges": [float("nan")] * 2,
                    "point_forces": [float("nan")] * 3,
                    "status": 5,
                    "converged": 0,
                    "iterations": 8,
                },
            },
            "good": {
                "id": "good",
                "atom_count": 1,
                "point_count": 0,
                "expected": expected_good,
                "mock": {
                    "energy": 1.0,
                    "forces": [0.0] * 3,
                    "charges": [0.0],
                    "point_forces": [],
                    "status": 0,
                    "converged": 1,
                    "iterations": 3,
                },
            },
        }
        plan = ao_grouping.make_plan(
            ("bad", "good"), 1, "exact-ao", {"bad": 6, "good": 3}, "e" * 64
        )
        args = SimpleNamespace(
            library=Path("/tmp/fake-libxtbloom.so"),
            manifest=Path("/tmp/fake-manifest.json"),
            backends=("cpu",),
            device_id=0,
            cpu_threads=1,
            warmups=0,
            repetitions=2,
        )
        manifest = {
            "tolerances": {
                "energy": {"atol": 1.0e-7},
                "forces": {"atol": 1.0e-7},
                "charges": {"atol": 1.0e-7},
                "point_charge_forces": {"atol": 1.0e-7},
            }
        }
        constructor_calls: list[dict[str, object]] = []

        class FakeAdapter:
            def __init__(
                self,
                _library_path: Path,
                _manifest_path: Path,
                _manifest: dict[str, Any],
                case_sequence: tuple[dict[str, Any], ...],
                _cell: run.Cell,
                _device_id: int,
                _cpu_threads: int,
                **options: object,
            ) -> None:
                self.sweep_index = len(constructor_calls) // len(plan.batches)
                constructor_calls.append(options)
                offsets = [0]
                point_offsets = [0]
                for case in case_sequence:
                    offsets.append(offsets[-1] + case["atom_count"])
                    point_offsets.append(point_offsets[-1] + case["point_count"])
                self.storage = SimpleNamespace(
                    atom_offsets=offsets,
                    point_charge_offsets=point_offsets,
                    point_charge_values=[
                        0.0
                        for case in case_sequence
                        for _ in range(case["point_count"])
                    ],
                    slices=[
                        SimpleNamespace(case=case, expected=case["expected"])
                        for case in case_sequence
                    ],
                )
                self.case_sequence = case_sequence

            def invoke(self) -> None:
                return None

            def synchronize(self) -> None:
                return None

            def results(self) -> dict[str, list[Any]]:
                output = {
                    "energies_hartree": [],
                    "scc_iterations": [],
                    "scc_converged": [],
                    "per_system_status": [],
                    "forces_hartree_per_bohr": [],
                    "atomic_charges_e": [],
                    "point_charge_forces_hartree_per_bohr": [],
                }
                for case in self.case_sequence:
                    values = case["mock"]
                    if self.sweep_index > 0 and case["id"] == "bad":
                        values = {
                            "energy": 2.0,
                            "forces": [0.0] * 6,
                            "charges": [0.0, 0.0],
                            "point_forces": [0.0] * 3,
                            "status": 0,
                            "converged": 1,
                            "iterations": 9,
                        }
                    output["energies_hartree"].append(values["energy"])
                    output["scc_iterations"].append(values["iterations"])
                    output["scc_converged"].append(values["converged"])
                    output["per_system_status"].append(values["status"])
                    output["forces_hartree_per_bohr"].extend(values["forces"])
                    output["atomic_charges_e"].extend(values["charges"])
                    output["point_charge_forces_hartree_per_bohr"].extend(
                        values["point_forces"]
                    )
                return output

            def memory_snapshot(self) -> dict[str, int]:
                return {"host_process_hwm_bytes": 9 if self.sweep_index == 0 else 1}

            def close(self) -> None:
                return None

        with mock.patch.object(run, "XTBloomAdapter", FakeAdapter):
            row = run.benchmark_finite_xtbloom_cell(
                args, manifest, cases, plan, 0.25, "force"
            )

        self.assertEqual(row["availability"], "available")
        self.assertEqual(row["planned_case_order"], ["good", "bad"])
        self.assertEqual(
            [result["case_id"] for result in row["case_results"]], ["bad", "good"]
        )
        self.assertEqual(row["case_results"][0]["status"], 0)
        self.assertEqual(row["case_results"][0]["correctness"]["status"], "pass")
        self.assertEqual(row["case_results"][0]["scc_iterations"], 9)
        self.assertEqual(row["case_results"][0]["scc_converged"], 1)
        self.assertEqual(row["case_results"][0]["original_index"], 0)
        self.assertEqual(row["case_results"][1]["original_index"], 1)
        self.assertEqual(
            [sweep["correctness"]["status"] for sweep in row["measurement_sweeps"]],
            ["fail", "pass"],
        )
        self.assertEqual(
            [sweep["case_results"][0]["status"] for sweep in row["measurement_sweeps"]],
            [5, 0],
        )
        self.assertEqual(
            [
                sweep["case_results"][0]["scc_converged"]
                for sweep in row["measurement_sweeps"]
            ],
            [0, 1],
        )
        self.assertEqual(
            [
                sweep["case_results"][0]["scc_iterations"]
                for sweep in row["measurement_sweeps"]
            ],
            [8, 9],
        )
        self.assertEqual(row["correctness"]["status"], "fail")
        self.assertEqual(row["correctness"]["successful_system_ids"], ["good"])
        self.assertEqual(row["correctness"]["failed_system_ids"], ["bad"])
        self.assertEqual(row["correctness"]["correctness_failure_ids"], ["bad"])
        self.assertEqual(row["memory"]["host_peak_rss_bytes"], 9)
        self.assertEqual(len(row["memory"]["batch_context_snapshots"]), 4)
        self.assertEqual(
            [len(sweep["memory_snapshots"]) for sweep in row["measurement_sweeps"]],
            [2, 2],
        )
        self.assertIn("correctness_validation_ms_median", row["timing"])
        fail_exit, fail_report = self._run_main_with_finite_row(
            row, ("--fail-on-correctness",)
        )
        self.assertEqual(fail_exit, 2)
        self.assertTrue(fail_report["protocol"]["fail_on_correctness"])
        self.assertTrue(
            all(
                call
                == {
                    "strict_fresh": True,
                    "request_charges": True,
                    "allow_system_failures": True,
                }
                for call in constructor_calls
            )
        )


class ReferenceQualificationTests(unittest.TestCase):
    """Distinguish numerical execution from independent property qualification."""

    def test_missing_reference_is_diagnostic_not_scientifically_qualified(self) -> None:
        """Finite converged outputs do not replace an independent E/F/q oracle."""
        result = {
            "status": 0,
            "scc_converged": 1,
            "energy_hartree": 1.0,
            "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
            "atomic_charges_e": [0.0],
            "point_charge_forces_hartree_per_bohr": [],
        }
        check = run.finite_case_correctness(
            {"oracle_role": "diagnostic-no-independent-reference"},
            {},
            result,
            {},
            "force",
        )
        self.assertEqual(check["status"], "pass")
        self.assertFalse(check["independent_reference_pass"])
        self.assertEqual(
            check["missing_reference_properties"],
            ["energy_hartree", "forces_hartree_per_bohr", "partial_charges_e"],
        )
        self.assertIn("no independent reference", check["reference_validation"])

    def test_energy_reference_cannot_qualify_force_and_charge_outputs(self) -> None:
        """Requested-property qualification requires all relevant reference slices."""
        result = {
            "status": 0,
            "scc_converged": 1,
            "energy_hartree": 1.0,
            "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
            "atomic_charges_e": [0.0],
            "point_charge_forces_hartree_per_bohr": [],
        }
        manifest = {
            "tolerances": {
                "energy": {"atol": 1.0e-7},
                "forces": {"atol": 1.0e-7},
                "charges": {"atol": 1.0e-7},
            }
        }
        expected = {"energy_hartree": 1.0}
        check = run.finite_case_correctness({}, expected, result, manifest, "force")
        self.assertEqual(check["status"], "pass")
        self.assertFalse(check["independent_reference_pass"])
        expected.update(
            forces_hartree_per_bohr=[0.0, 0.0, 0.0], partial_charges_e=[0.0]
        )
        check = run.finite_case_correctness({}, expected, result, manifest, "force")
        self.assertTrue(check["independent_reference_pass"])
        self.assertEqual(check["missing_reference_properties"], [])


if __name__ == "__main__":
    unittest.main()
