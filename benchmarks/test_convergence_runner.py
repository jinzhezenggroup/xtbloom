"""Frozen-input binding and finite-list AO/risk runner regression tests."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from benchmarks import ao_grouping, convergence_grouping, run


def _json_hash(value: object) -> str:
    """Independently reproduce the documented logical JSON hash encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrozenRunnerTests(unittest.TestCase):
    """Use real input parsers and a pinned workload without running inference."""

    def setUp(self) -> None:
        """Freeze four complete systems in two explicitly separate families."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cases: dict[str, dict[str, Any]] = {}
        systems = []
        basis, _ = ao_grouping.load_gfn2_basis_ao_counts(
            run.REPOSITORY_ROOT / "data" / "parameters" / "gfn2.json"
        )
        for case_id, numbers, charge, spin, unpaired in (
            ("a", [6, 1, 1, 1, 1], 0, 1, 0),
            ("b", [6, 1, 1, 1, 1], 1, 1, 0),
            ("c", [6, 1, 1, 1, 1], 0, 2, 1),
            ("d", [8, 1], 0, 1, 0),
        ):
            input_path = self.root / f"{case_id}.coord"
            input_path.write_text(
                "$coord\n"
                + "".join(
                    f"{index} 0 0 {run.conformance.ELEMENT_SYMBOLS[number].lower()}\n"
                    for index, number in enumerate(numbers)
                )
                + "$end\n",
                encoding="utf-8",
            )
            metadata = {
                "atomic_numbers": numbers,
                "exact_ao_count": ao_grouping.count_gfn2_aos(numbers, basis),
                "molecular_charge": charge,
                "spin_channels": spin,
                "unpaired_electrons": unpaired,
            }
            self.cases[case_id] = {
                "id": case_id,
                "input": str(input_path),
                "input_sha256": run.sha256_file(input_path),
                "atom_count": len(numbers),
                "molecular_charge": charge,
                "spin_channels": spin,
                "unpaired_electrons": unpaired,
            }
            systems.append(
                {
                    "case_id": case_id,
                    "molecule_group_id": "carbon-family"
                    if case_id != "d"
                    else "oxygen-family",
                    "scheduling_metadata": metadata,
                }
            )
        self.manifest = {"method": "GFN2-xTB", "cases": list(self.cases.values())}
        self.document = {"schema_version": 1, "systems": systems}
        self.freeze = convergence_grouping.build_freeze_plan(self.document)
        self.manifest_path = self.root / "manifest.json"
        self.document_path = self.root / "scheduling.json"
        self.freeze_path = self.root / "freeze.json"
        for path, document in (
            (self.manifest_path, self.manifest),
            (self.document_path, self.document),
            (self.freeze_path, self.freeze),
        ):
            path.write_text(json.dumps(document), encoding="utf-8")
        self.args = SimpleNamespace(
            ao_grouping="ao-risk",
            manifest=self.manifest_path,
            convergence_manifest=self.document_path,
            convergence_plan=self.freeze_path,
            convergence_freeze_sha256=self.freeze["freeze_plan_sha256"],
            convergence_workload_sha256=run.sha256_file(self.manifest_path),
            convergence_cohort_sha256=convergence_grouping.ordered_case_ids_sha256(
                ("a", "b", "c", "d")
            ),
            convergence_partition="all",
        )
        self.addCleanup(self._close_capture)

    def _close_capture(self) -> None:
        """Release private input snapshots created by a fixture's direct planner."""
        captured = getattr(self.args, "finite_frozen_workload", None)
        if captured is not None:
            captured.close()

    def plan(
        self, ids: tuple[str, ...] = ("a", "b", "c", "d"), *, pin_selected: bool = True
    ) -> tuple[ao_grouping.AOGroupingPlan, float]:
        """Exercise the public-runner planning boundary with complete fixtures."""
        if pin_selected:
            self.args.convergence_cohort_sha256 = (
                convergence_grouping.ordered_case_ids_sha256(ids)
            )
        return run.finite_case_plan(self.args, self.manifest, self.cases, ids, 2)

    def test_candidate_uses_single_planner_and_complete_annotations(self) -> None:
        """Preserve every identity while adding only the frozen risk bucket key."""
        plan, elapsed = self.plan()
        annotations = convergence_grouping.annotate_scheduling_inputs(self.document)
        self.assertEqual(plan.ordered_case_ids, ("d", "c", "b", "a"))
        self.assertEqual(
            dict(plan.risk_bands_by_case_id),
            {
                case_id: annotation.risk_band
                for case_id, annotation in annotations.items()
            },
        )
        self.assertEqual(
            plan.risk_freeze_plan_sha256, self.args.convergence_freeze_sha256
        )
        self.assertEqual(self.args.finite_convergence_binding["frozen_case_count"], 4)
        self.assertGreaterEqual(elapsed, 0)

    def test_selected_subset_validates_whole_frozen_roster(self) -> None:
        """Reject changed metadata even for an unselected frozen peer."""
        self.cases["d"]["spin_channels"] = 2
        with self.assertRaisesRegex(run.BenchmarkError, "does not match workload"):
            self.plan(("a",))

    def test_missing_full_roster_peer_is_rejected(self) -> None:
        """Reject a workload that cannot supply the complete frozen roster."""
        del self.cases["d"]
        with self.assertRaisesRegex(run.BenchmarkError, "absent from the workload"):
            self.plan(("a",))

    def test_metadata_mismatches_never_schedule(self) -> None:
        """Keep charge and explicit spin metadata bound to actual inputs."""
        for field, value in (
            ("molecular_charge", 2),
            ("spin_channels", 2),
            ("unpaired_electrons", 1),
        ):
            with (
                self.subTest(field=field),
                mock.patch.dict(self.cases["a"], {field: value}),
                self.assertRaisesRegex(run.BenchmarkError, "does not match workload"),
            ):
                self.plan()

    def test_changed_geometry_and_missing_hash_are_rejected(self) -> None:
        """Bind geometry bytes without exposing them to risk annotations."""
        input_path = Path(self.cases["d"]["input"])
        original = input_path.read_bytes()
        input_path.write_bytes(original.replace(b"0 0 0", b"0.1 0 0"))
        with self.assertRaisesRegex(run.BenchmarkError, "input SHA-256 mismatch"):
            self.plan(("a",))
        input_path.write_bytes(original)
        del self.cases["d"]["input_sha256"]
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.args.convergence_workload_sha256 = run.sha256_file(self.manifest_path)
        with self.assertRaisesRegex(run.BenchmarkError, "input SHA-256.*missing"):
            self.plan(("a",))

    def test_expected_workload_and_freeze_identities_are_external_pins(self) -> None:
        """Reject self-consistent artifacts from another experiment identity."""
        for field in ("convergence_workload_sha256", "convergence_freeze_sha256"):
            with (
                self.subTest(field=field),
                mock.patch.object(self.args, field, "0" * 64),
                self.assertRaisesRegex(
                    run.BenchmarkError, "expected (experiment identity|SHA-256)"
                ),
            ):
                self.plan()

    def test_cohort_pin_rejects_omission_addition_and_order_change(self) -> None:
        """Require the exact preregistered view, not an arbitrary valid subset."""
        for ids in (("a",), ("a", "b", "c", "d", "a"), ("d", "c", "b", "a")):
            with self.subTest(ids=ids), self.assertRaises(run.BenchmarkError):
                self.plan(ids, pin_selected=False)

    def test_partition_is_a_validator_not_a_filter(self) -> None:
        """Keep submitted view order and reject rather than remove wrong peers."""
        for group in self.freeze["split"]["groups"]:
            self.args.convergence_partition = group["partition"]
            ids = tuple(reversed(group["case_ids"]))
            plan, _ = self.plan(ids)
            self.assertEqual(plan.original_case_ids, ids)
            self.assertEqual(set(plan.ordered_case_ids), set(ids))
            with self.assertRaisesRegex(run.BenchmarkError, "cross.*partition"):
                self.plan()

    def test_default_strategies_keep_legacy_plan_hashes(self) -> None:
        """Retain legacy defaults and schema-1 hash encoding without freeze flags."""
        ids = ("a", "b", "c", "d")
        for strategy in ("original", "exact-ao"):
            self.args.ao_grouping = strategy
            self.args.convergence_plan = None
            legacy, _ = self.plan(ids)
            counts = (
                dict(legacy.ao_counts_by_case_id) if strategy == "exact-ao" else None
            )
            expected = ao_grouping.make_plan(
                ids, 2, strategy, counts, legacy.basis_sha256
            )
            self.assertEqual(legacy.plan_sha256, expected.plan_sha256)
            self.assertEqual(legacy.risk_bands_by_case_id, ())

    def test_frozen_baselines_do_not_run_risk_annotation(self) -> None:
        """Validate shared bindings without charging baseline for candidate work."""
        for strategy in ("original", "exact-ao"):
            self.args.ao_grouping = strategy
            with mock.patch.object(
                convergence_grouping,
                "annotate_scheduling_inputs",
                side_effect=AssertionError("candidate-only"),
            ):
                plan, _ = self.plan()
            self.assertEqual(plan.strategy, strategy)
            self.assertEqual(plan.risk_bands_by_case_id, ())

    def test_frozen_original_keeps_its_legacy_hash_and_csv_columns(self) -> None:
        """Keep new binding opt-in without changing the original permutation hash."""
        self.args.ao_grouping = "original"
        plan, _ = self.plan()
        expected = ao_grouping.make_plan(("a", "b", "c", "d"), 2)
        self.assertEqual(plan.plan_sha256, expected.plan_sha256)
        self.assertEqual(plan.ao_counts_by_case_id, ())
        self.assertIsNone(plan.basis_sha256)
        csv_path = self.root / "legacy.csv"
        run.write_csv(
            csv_path,
            [
                dict(
                    run.base_row(
                        run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1, ("a",))
                    ),
                    availability="available",
                )
            ],
        )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle))
        self.assertNotIn("convergence_binding_json", fields)
        self.assertNotIn("planning_inclusive_end_to_end_median_ms", fields)

    def test_finite_row_restores_failures_and_retains_binding_in_csv(self) -> None:
        """Publish all failed slices and keep planning-inclusive provenance."""
        plan, _ = self.plan()
        expected_inputs = {
            case_id: Path(case["input"]).read_bytes()
            for case_id, case in self.cases.items()
        }
        for case in self.cases.values():
            Path(case["input"]).write_text("changed source after capture\n")
        self.manifest_path.write_text('{"method": "changed after capture"}')
        self.args.library = Path("/tmp/mock-library.so")
        self.args.backends = ("cpu",)
        self.args.device_id = 0
        self.args.cpu_threads = 1
        self.args.warmups = 0
        self.args.repetitions = 2
        failed_case_id: str | None = "b"

        class FakeAdapter:
            """Publish controlled peer outcomes without a native backend."""

            def __init__(
                self,
                _library: Path,
                _path: Path,
                _manifest: object,
                cases: tuple[dict[str, Any], ...],
                *_args: object,
                **options: object,
            ) -> None:
                self.cases = cases
                for case in cases:
                    if Path(case["input"]).read_bytes() != expected_inputs[case["id"]]:
                        raise AssertionError(
                            "native assembly did not receive frozen input bytes"
                        )
                if options != {
                    "strict_fresh": True,
                    "request_charges": True,
                    "allow_system_failures": True,
                }:
                    raise AssertionError(options)
                offsets = [0]
                for case in cases:
                    offsets.append(offsets[-1] + case["atom_count"])
                self.storage = SimpleNamespace(
                    atom_offsets=offsets,
                    point_charge_offsets=[0] * (len(cases) + 1),
                    point_charge_values=[],
                    slices=[SimpleNamespace(case=case, expected={}) for case in cases],
                )

            def results(self) -> dict[str, list[Any]]:
                output = {
                    field: []
                    for field in (
                        "energies_hartree",
                        "scc_iterations",
                        "scc_converged",
                        "per_system_status",
                        "forces_hartree_per_bohr",
                        "atomic_charges_e",
                    )
                }
                for case in self.cases:
                    failed = case["id"] == failed_case_id
                    value = float("nan") if failed else 0.0
                    output["energies_hartree"].append(value)
                    output["scc_iterations"].append(8 if failed else 3)
                    output["scc_converged"].append(0 if failed else 1)
                    output["per_system_status"].append(5 if failed else 0)
                    output["forces_hartree_per_bohr"].extend(
                        [value] * (3 * case["atom_count"])
                    )
                    output["atomic_charges_e"].extend([value] * case["atom_count"])
                return output

            def memory_snapshot(self) -> dict[str, int]:
                return {"host_process_hwm_bytes": 1}

            def close(self) -> None:
                return None

        with (
            mock.patch.object(run, "XTBloomAdapter", FakeAdapter),
            mock.patch.object(run, "timed_invoke", return_value=0.1),
        ):
            row = run.benchmark_finite_xtbloom_cell(
                self.args, self.manifest, self.cases, plan, 7.5, "force"
            )
        self.assertEqual(
            [result["case_id"] for result in row["case_results"]],
            list(plan.original_case_ids),
        )
        failed = row["case_results"][1]
        self.assertEqual((failed["status"], failed["scc_converged"]), (5, 0))
        self.assertTrue(
            all(math.isnan(value) for value in failed["forces_hartree_per_bohr"])
        )
        self.assertFalse(row["claim_eligible"])
        self.assertEqual(
            row["convergence_binding"], self.args.finite_convergence_binding
        )
        self.assertEqual(row["risk_band_by_case_id"], dict(plan.risk_bands_by_case_id))
        for batch_record, batch in zip(row["ao_batches"], plan.batches, strict=True):
            self.assertEqual(batch_record["risk_band"], batch.risk_band)
        timing = row["timing"]
        self.assertAlmostEqual(
            timing["planning_inclusive_end_to_end_ms"]["median_ms"],
            7.5 + timing["end_to_end_ms"]["median_ms"],
        )
        csv_path = self.root / "results.csv"
        run.write_csv(csv_path, [row])
        with csv_path.open(newline="", encoding="utf-8") as handle:
            csv_row = next(csv.DictReader(handle))
        self.assertEqual(
            json.loads(csv_row["convergence_binding_json"]), row["convergence_binding"]
        )
        self.assertEqual(
            json.loads(csv_row["risk_band_by_case_id_json"]),
            row["risk_band_by_case_id"],
        )
        failed_case_id = None
        self.args.repetitions = 1
        with (
            mock.patch.object(run, "XTBloomAdapter", FakeAdapter),
            mock.patch.object(run, "timed_invoke", return_value=0.1),
            mock.patch.object(
                run,
                "finite_case_correctness",
                return_value={"status": "pass", "independent_reference_pass": True},
            ),
        ):
            qualified = run.benchmark_finite_xtbloom_cell(
                self.args, self.manifest, self.cases, plan, 7.5, "force"
            )
        self.assertTrue(qualified["independent_reference_qualified"])
        self.assertFalse(qualified["claim_eligible"])
        self.assertIn(
            "paired holdout decision incomplete", qualified["claim_eligibility_scope"]
        )
        before_cleanup = qualified["timing"]["one_shot_total_ms"]
        complete_before_cleanup = qualified["timing"][
            "planning_inclusive_end_to_end_ms"
        ]["median_ms"]
        with mock.patch.object(
            run.time, "perf_counter_ns", side_effect=[10_000_000, 14_000_000]
        ):
            run.finish_frozen_run(self.args, qualified)
        self.assertIsNone(self.args.finite_frozen_workload)
        self.assertEqual(qualified["timing"]["input_snapshot_release_ms"], 4.0)
        self.assertAlmostEqual(
            qualified["timing"]["one_shot_total_ms"], before_cleanup + 4.0
        )
        self.assertAlmostEqual(
            qualified["timing"]["planning_inclusive_end_to_end_ms"]["median_ms"],
            complete_before_cleanup + 4.0,
        )

    def test_metadata_parsing_uses_the_verified_input_snapshot(self) -> None:
        """Keep a source-file mutation between capture and parsing out of the plan."""
        parser = run.case_atomic_numbers
        original = Path(self.cases["a"]["input"]).read_bytes()

        def mutate_source_then_parse(
            path: Path, manifest: dict[str, Any], case: dict[str, Any]
        ) -> tuple[int, ...]:
            Path(self.cases["a"]["input"]).write_text("source changed after capture\n")
            return parser(path, manifest, case)

        with mock.patch.object(
            run, "case_atomic_numbers", side_effect=mutate_source_then_parse
        ):
            plan, _ = self.plan()
        self.assertEqual(dict(plan.ao_counts_by_case_id)["a"], 8)
        captured = self.args.finite_frozen_workload
        self.assertEqual(
            Path(captured.cases_by_id["a"]["input"]).read_bytes(), original
        )

    def test_validate_freeze_rejects_rehashed_edits_and_numeric_aliases(self) -> None:
        """Reconstruct the protocol instead of trusting newly self-consistent hashes."""
        for value in (False, 5.0, 6):
            changed = copy.deepcopy(self.freeze)
            changed["evaluation_protocol"][
                "warmup_runs_per_strategy_and_coordinate"
            ] = value
            del changed["freeze_plan_sha256"]
            changed["freeze_plan_sha256"] = _json_hash(changed)
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    convergence_grouping.ConvergenceGroupingError, "preregistered split"
                ),
            ):
                convergence_grouping.validate_freeze_plan(self.document, changed)
        changed_document = copy.deepcopy(self.document)
        changed_document["systems"][0]["molecule_group_id"] = "rewritten-family"
        changed_freeze = convergence_grouping.build_freeze_plan(changed_document)
        with self.assertRaisesRegex(
            convergence_grouping.ConvergenceGroupingError,
            "expected experiment identity",
        ):
            convergence_grouping.validate_freeze_plan(
                changed_document,
                changed_freeze,
                expected_freeze_plan_sha256=self.args.convergence_freeze_sha256,
            )


class ConvergenceCLITests(unittest.TestCase):
    """Keep the prototype opt-in and direct/module entry points consistent."""

    def test_freeze_flags_are_complete_and_finite_list_only(self) -> None:
        """Reject partial identities and implicit finite-list experiments."""
        parser = run.build_parser()
        base = [
            "--library",
            "/tmp/mock-library.so",
            "--engines",
            "xtbloom",
            "--backends",
            "cpu",
            "--case-ids",
            "a",
        ]
        flags = [
            "--ao-grouping",
            "ao-risk",
            "--convergence-manifest",
            "/tmp/scheduling.json",
            "--convergence-plan",
            "/tmp/freeze.json",
            "--convergence-freeze-sha256",
            "a" * 64,
            "--convergence-workload-sha256",
            "b" * 64,
            "--convergence-cohort-sha256",
            "c" * 64,
        ]
        run.validate_args(parser.parse_args(base + flags))
        for flag in (
            "--convergence-manifest",
            "--convergence-plan",
            "--convergence-freeze-sha256",
            "--convergence-workload-sha256",
            "--convergence-cohort-sha256",
        ):
            partial = flags.copy()
            offset = partial.index(flag)
            del partial[offset : offset + 2]
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(run.BenchmarkError, "required together"),
            ):
                run.validate_args(parser.parse_args(base + partial))
        with self.assertRaisesRegex(run.BenchmarkError, "explicit finite case list"):
            run.validate_args(parser.parse_args(base[:-2] + flags))
        with self.assertRaisesRegex(run.BenchmarkError, "pinned freeze plan"):
            run.validate_args(
                parser.parse_args([*base, "--convergence-partition", "holdout"])
            )

    def test_direct_script_help_remains_available(self) -> None:
        """Support the documented direct-script fallback as well as module imports."""
        result = subprocess.run(
            [
                sys.executable,
                str(run.REPOSITORY_ROOT / "benchmarks" / "run.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ao-risk", result.stdout)


if __name__ == "__main__":
    unittest.main()
