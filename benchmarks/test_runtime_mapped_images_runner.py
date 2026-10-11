"""Endpoint instrumentation tests without loading a native library or model."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks import run
from benchmarks import test_build_receipt_runner as receipt_tests
from benchmarks import test_run as paired_tests


def observation_check(inode: int = 17) -> dict[str, object]:
    """Represent a synthetic endpoint result, never a real mapped-image pass."""
    return {
        "status": "PASS",
        "error": None,
        "validation_ms": 2.0,
        "observer_source": {"sha256": "b" * 64},
        "observation": {
            "selected_library": {"inode": inode, "sha256": "a" * 64},
            "entrypoints": {"owner-0:xtbloom_compute": 4096},
            "complete_runtime_dependency_closure": "NOT_ESTABLISHED",
            "performance_admission": False,
        },
    }


class RuntimePairObservationTests(unittest.TestCase):
    """Preserve the existing paired protocol and numerical failure isolation."""

    def setUp(self) -> None:
        """Reuse the established fake protocol without requiring native handles."""
        self.harness = paired_tests.PairedRunnerTest(
            "test_alternates_all_phases_and_reuses_each_layout_owner"
        )

    def enabled_pair(
        self, checks: list[dict[str, object]], **kwargs: object
    ) -> tuple[
        dict[str, object], list[tuple[object, ...]], list[object], mock.MagicMock
    ]:
        """Reuse the established fake adapter protocol, changing only the opt-in."""
        original = run.benchmark_finite_xtbloom_pair

        def enable_observation(
            args: SimpleNamespace, *remaining: object
        ) -> dict[str, object]:
            args.runtime_mapped_images = True
            args.runtime_images_expected_sha256 = "a" * 64
            return original(args, *remaining)

        with (
            mock.patch.object(
                run, "benchmark_finite_xtbloom_pair", side_effect=enable_observation
            ),
            mock.patch.object(run, "runtime_image_check", side_effect=checks) as check,
        ):
            result = self.harness._run_pair(**kwargs)
        return (*result, check)

    def test_default_does_not_observe_fake_adapters_or_maps(self) -> None:
        """The existing default does not need a CDLL or perform observation I/O."""
        with mock.patch.object(
            run, "runtime_image_check", side_effect=AssertionError("unexpected I/O")
        ) as check:
            row, _, _ = self.harness._run_pair()
        check.assert_not_called()
        self.assertNotIn("runtime_mapped_images", row["producer_provenance"])
        self.assertFalse(row["claim_eligible"])

    def test_success_preserves_all_rounds_and_never_admits_performance(self) -> None:
        """Image matches cannot independently qualify science or timing."""
        row, events, created, check = self.enabled_pair(
            [observation_check(), observation_check()]
        )
        self.assertEqual(check.call_count, 2)
        self.assertEqual(sum(adapter.calls for adapter in created), 16)
        self.assertEqual(sum(event[0] == "close" for event in events), len(created))
        evidence = row["producer_provenance"]["runtime_mapped_images"]
        self.assertTrue(evidence["integrity_passed"])
        self.assertFalse(evidence["performance_admission"])
        self.assertFalse(row["claim_eligible"])
        self.assertTrue(row["timing_excludes_runtime_image_observation"])
        self.assertTrue(
            all(
                not pair["performance_comparison_eligible"]
                for pair in row["paired_rounds"]
            )
        )

    def test_failed_preflight_starts_no_sweeps_and_still_closes_all_owners(
        self,
    ) -> None:
        """Refuse both arms' calls while releasing every constructed context."""
        failure = {
            "status": "FAIL",
            "error": "wrong mapped inode",
            "validation_ms": 2.0,
        }
        row, events, created, check = self.enabled_pair([failure])
        self.assertEqual(check.call_count, 1)
        self.assertFalse(any(event[0] == "invoke" for event in events))
        self.assertTrue(all(adapter.calls == 0 for adapter in created))
        self.assertEqual(sum(event[0] == "close" for event in events), len(created))
        evidence = row["producer_provenance"]["runtime_mapped_images"]
        self.assertEqual(evidence["postflight"]["status"], "NOT_RUN")
        self.assertFalse(evidence["integrity_passed"])
        self.assertEqual(row["availability"], "error")

    def test_failed_postflight_retains_successful_outputs_and_failed_peer_nans(
        self,
    ) -> None:
        """Keep physical failure NaNs separate from tooling integrity errors."""
        baseline, _, _ = self.harness._run_pair(behavior={"failed_ids": {"b"}})
        failure = {"status": "FAIL", "error": "changed image", "validation_ms": 2.0}
        row, events, created, check = self.enabled_pair(
            [observation_check(), failure], behavior={"failed_ids": {"b"}}
        )
        self.assertEqual(check.call_count, 2)
        for before, after in zip(
            baseline["paired_rounds"], row["paired_rounds"], strict=True
        ):
            for layout in ("original", "exact-ao"):
                self.assertEqual(
                    run.json_safe_value(before["arms"][layout]["case_results"]),
                    run.json_safe_value(after["arms"][layout]["case_results"]),
                )
        self.assertEqual(sum(event[0] == "close" for event in events), len(created))
        self.assertFalse(
            row["producer_provenance"]["runtime_mapped_images"]["integrity_passed"]
        )

    def test_between_endpoint_identity_change_fails_without_replacing_postflight(
        self,
    ) -> None:
        """Keep the failed image record to diagnose endpoint replacement."""
        row, _, _, _ = self.enabled_pair([observation_check(), observation_check(18)])
        evidence = row["producer_provenance"]["runtime_mapped_images"]
        self.assertEqual(evidence["postflight"]["status"], "FAIL")
        self.assertEqual(
            evidence["postflight"]["observation"]["selected_library"]["inode"], 18
        )
        self.assertFalse(evidence["integrity_passed"])

    def test_observer_source_change_does_not_rebind_an_existing_run(self) -> None:
        """Reject changed observer bytes without erasing completed coordinates."""
        after = observation_check()
        after["observer_source"] = {"sha256": "c" * 64}
        row, _, _, _ = self.enabled_pair([observation_check(), after])
        evidence = row["producer_provenance"]["runtime_mapped_images"]
        self.assertFalse(evidence["integrity_passed"])
        self.assertIn("observer source changed", evidence["postflight"]["error"])

    def test_observation_helper_reports_costs_and_preserves_tool_failure(self) -> None:
        """Tool failures are structured evidence, not unhandled exceptions."""
        args = SimpleNamespace(
            library=Path("fake.so"), runtime_images_expected_sha256="a" * 64
        )
        owners = {
            layout: [
                {"setup_state": "ready", "adapter": SimpleNamespace(library=object())}
            ]
            for layout in ("original", "exact-ao")
        }
        with (
            mock.patch.object(
                run.runtime_mapped_images,
                "library_entrypoints",
                return_value={"compute": 4096},
            ),
            mock.patch.object(
                run.runtime_mapped_images,
                "capture",
                side_effect=RuntimeError("maps denied"),
            ),
        ):
            result = run.runtime_image_check(args, owners)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["error"], "maps denied")
        self.assertGreaterEqual(result["validation_ms"], 0.0)

    def test_incomplete_owner_setup_is_not_a_partial_mapping_pass(self) -> None:
        """An unavailable owner cannot borrow another owner's valid mapping."""
        args = SimpleNamespace(
            library=Path("fake.so"), runtime_images_expected_sha256="a" * 64
        )
        owners = {
            "original": [{"setup_state": "unavailable", "adapter": None}],
            "exact-ao": [],
        }
        with mock.patch.object(run.runtime_mapped_images, "capture") as capture:
            result = run.runtime_image_check(args, owners)
        self.assertEqual(result["status"], "FAIL")
        capture.assert_not_called()


class RuntimeObservationCLITests(unittest.TestCase):
    """Integrity errors return nonzero after retaining existing numerical rows."""

    def setUp(self) -> None:
        """Attach required receipt pins to one fake paired cell."""
        fixture = receipt_tests.ReceiptRunnerTests()
        fixture.setUp()
        self.base = fixture.base
        self.flags = [*fixture.flags, "--runtime-mapped-images"]
        self.verified = dict(fixture.verified, library={"sha256": "a" * 64})

    def test_constraints_fail_before_receipt_or_workload_io(self) -> None:
        """Reject unsupported scope before workload or provenance I/O."""
        for flags, platform_name in (
            (["--runtime-mapped-images"], "linux"),
            (self.flags, "darwin"),
        ):
            with (
                self.subTest(platform=platform_name),
                mock.patch.object(run.sys, "platform", platform_name),
                mock.patch.object(run.conformance, "load_json") as load,
                mock.patch.object(run.build_receipt, "verify_receipt") as verify,
                mock.patch("sys.stderr"),
            ):
                self.assertEqual(run.main(self.base + flags), 1)
                load.assert_not_called()
                verify.assert_not_called()

    def test_convergence_options_fail_before_policy_workload_or_native_io(
        self,
    ) -> None:
        """Mapping pins never authorize convergence inputs or holdout access."""
        conflicting_flags = [
            ["--convergence-manifest", "/tmp/unread-scheduling.json"],
            ["--convergence-plan", "/tmp/unread-freeze.json"],
            ["--convergence-freeze-sha256", "a" * 64],
            ["--convergence-workload-sha256", "b" * 64],
            ["--convergence-cohort-sha256", "c" * 64],
            ["--convergence-partition", "calibration"],
            ["--convergence-partition", "holdout"],
        ]
        conflicting_flags.append(
            [value for option in conflicting_flags[:5] for value in option]
            + ["--convergence-partition", "holdout"]
        )
        for case_flag, case_value in (
            ("--case-ids", "a"),
            ("--case-ids-file", "/tmp/unread-case-ids.txt"),
        ):
            base = self.base.copy()
            offset = base.index("--case-ids")
            base[offset : offset + 2] = [case_flag, case_value]
            for flags in conflicting_flags:
                with (
                    self.subTest(case_flag=case_flag, flags=flags),
                    mock.patch.object(run.sys, "platform", "linux"),
                    mock.patch.object(run, "read_case_id_file") as read_ids,
                    mock.patch.object(run, "controlled_receipt_check") as receipt,
                    mock.patch.object(run.conformance, "load_json") as workload,
                    mock.patch.object(
                        run.convergence_grouping, "load_json_document"
                    ) as policy,
                    mock.patch.object(run.frozen_inputs, "capture_workload") as capture,
                    mock.patch.object(run, "XTBloomAdapter") as adapter,
                    mock.patch.object(run, "runtime_image_check") as observe,
                    mock.patch("sys.stderr") as stderr,
                ):
                    self.assertEqual(run.main(base + self.flags + flags), 1)
                    stderr.write.assert_any_call(
                        "error: paired mode is AO-only and cannot use "
                        "convergence inputs or partitions"
                    )
                    for boundary in (
                        read_ids,
                        receipt,
                        workload,
                        policy,
                        capture,
                        adapter,
                        observe,
                    ):
                        boundary.assert_not_called()

    def test_case_file_contents_are_revalidated_before_setup(self) -> None:
        """Early option validation must not bypass empty or duplicate ID checks."""
        base = self.base.copy()
        offset = base.index("--case-ids")
        base[offset : offset + 2] = ["--case-ids-file", "/tmp/fake-case-ids.txt"]
        for case_ids in ((), ("a", "a")):
            with (
                self.subTest(case_ids=case_ids),
                mock.patch.object(run.sys, "platform", "linux"),
                mock.patch.object(
                    run, "read_case_id_file", return_value=case_ids
                ) as read_ids,
                mock.patch.object(run, "controlled_receipt_check") as receipt,
                mock.patch.object(run.conformance, "load_json") as workload,
                mock.patch.object(run, "XTBloomAdapter") as adapter,
                mock.patch("sys.stderr"),
            ):
                self.assertEqual(run.main(base + self.flags), 1)
                read_ids.assert_called_once_with(Path("/tmp/fake-case-ids.txt"))
                receipt.assert_not_called()
                workload.assert_not_called()
                adapter.assert_not_called()

    def main_with_observation(
        self, integrity_passed: bool
    ) -> tuple[int, dict[str, object]]:
        """Retain the fake CLI document actually sent to the output writer."""
        row = run.base_row(
            run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1, ("a",))
        )
        row.update(
            availability="available",
            correctness={"status": "UNQUALIFIED"},
            producer_provenance={
                "runtime_mapped_images": {"integrity_passed": integrity_passed}
            },
            claim_ineligibility_reasons=[],
            claim_eligible=False,
            case_results=[{"case_id": "a", "energy_hartree": -1.0, "system_status": 0}],
        )
        manifest = {"method": "GFN2-xTB", "cases": [{"id": "a"}]}
        plan = run.ao_grouping.make_plan(("a",), 1)
        arguments = self.base + self.flags
        with (
            mock.patch.object(run.sys, "platform", "linux"),
            mock.patch.object(
                run.build_receipt,
                "verify_receipt",
                return_value=copy.deepcopy(self.verified),
            ),
            mock.patch.object(run.conformance, "load_json", return_value=manifest),
            mock.patch.object(run, "environment_metadata", return_value={}),
            mock.patch.object(run, "input_provenance", return_value={}),
            mock.patch.object(run, "finite_case_plan", return_value=(plan, 0.0)),
            mock.patch.object(run, "benchmark_finite_xtbloom_pair", return_value=row),
            mock.patch.object(run, "write_json") as write_json,
            mock.patch.object(run, "write_csv") as write_csv,
            mock.patch("sys.stdout"),
        ):
            status = run.main(arguments)
        write_csv.assert_called_once()
        return status, write_json.call_args.args[1]

    def test_failed_observation_preserves_available_native_row_and_writes_outputs(
        self,
    ) -> None:
        """A tooling failure must not erase published native coordinates."""
        status, document = self.main_with_observation(False)
        self.assertEqual(status, 1)
        self.assertEqual(document["rows"][0]["availability"], "available")
        self.assertEqual(document["rows"][0]["case_results"][0]["energy_hartree"], -1.0)
        self.assertFalse(
            document["metadata"]["runtime_mapped_image_observation"]["integrity_passed"]
        )

    def test_success_does_not_promote_diagnostic_numerics_or_performance(self) -> None:
        """Keep a mapping match diagnostic when science remains unqualified."""
        status, document = self.main_with_observation(True)
        self.assertEqual(status, 0)
        self.assertFalse(document["rows"][0]["claim_eligible"])
        self.assertFalse(
            document["metadata"]["runtime_mapped_image_observation"][
                "performance_admission"
            ]
        )

    def test_no_observed_rows_is_not_a_successful_observation(self) -> None:
        """An empty matrix cannot imply a successful requested observation."""
        with (
            mock.patch("sys.stderr"),
            mock.patch.object(run.conformance, "load_json") as load,
            self.assertRaises(SystemExit) as exit_code,
        ):
            run.main(self.base + self.flags + ["--properties", ""])
        self.assertEqual(exit_code.exception.code, 2)
        load.assert_not_called()

    def test_csv_appends_integrity_status_only_for_observed_rows(self) -> None:
        """Keep default columns unchanged and disclose failed opted-in checks."""
        row = run.base_row(run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1))
        row["availability"] = "available"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            run.write_csv(path, [row])
            with path.open(newline="") as handle:
                baseline_fields = csv.DictReader(handle).fieldnames
            row["producer_provenance"] = {
                "runtime_mapped_images": {
                    "integrity_passed": False,
                    "preflight": {"status": "PASS", "validation_ms": 2.0},
                    "postflight": {"status": "FAIL", "validation_ms": 3.0},
                }
            }
            run.write_csv(path, [row])
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames[:-4], baseline_fields)
                output = next(reader)
            self.assertEqual(output["runtime_image_integrity_passed"], "False")
            self.assertEqual(output["runtime_image_postflight_status"], "FAIL")
            self.assertEqual(output["runtime_image_validation_ms"], "5.0")

    def test_mixed_csv_preserves_convergence_columns_and_appends_mapping(self) -> None:
        """Synthetic frozen and observed rows retain distinct boolean encodings."""
        base = run.base_row(run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1))
        base["availability"] = "available"
        frozen = dict(
            base,
            ao_grouping="ao-risk",
            convergence_binding={"freeze_plan_sha256": "c" * 64},
            risk_band_by_case_id={"a": "high"},
            timing={
                "input_snapshot_release_ms": 1.0,
                "planning_inclusive_end_to_end_ms": {"median_ms": 7.0},
            },
            claim_eligible=False,
            independent_reference_qualified=False,
        )
        observed = dict(
            base,
            ao_grouping="paired",
            claim_eligible=False,
            independent_reference_qualified=False,
            producer_provenance={
                "runtime_mapped_images": {
                    "integrity_passed": False,
                    "preflight": {"status": "PASS", "validation_ms": 2.0},
                    "postflight": {"status": "FAIL", "validation_ms": 3.0},
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            run.write_csv(path, [base, frozen])
            with path.open(newline="", encoding="utf-8") as handle:
                convergence_fields = csv.DictReader(handle).fieldnames
            run.write_csv(path, [base, observed, frozen])
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                records = list(reader)
        self.assertEqual(
            convergence_fields[-5:],
            [
                "planning_inclusive_end_to_end_median_ms",
                "convergence_binding_json",
                "risk_band_by_case_id_json",
                "input_snapshot_release_ms",
                "claim_eligibility_scope",
            ],
        )
        self.assertEqual(fields[:-4], convergence_fields)
        self.assertEqual(
            fields[-4:],
            [
                "runtime_image_integrity_passed",
                "runtime_image_preflight_status",
                "runtime_image_postflight_status",
                "runtime_image_validation_ms",
            ],
        )
        self.assertEqual(len(fields), len(set(fields)))
        self.assertEqual(records[0]["claim_eligible"], "")
        self.assertEqual(records[1]["claim_eligible"], "False")
        self.assertEqual(records[1]["independent_reference_qualified"], "False")
        self.assertEqual(records[1]["runtime_image_postflight_status"], "FAIL")
        self.assertEqual(records[1]["runtime_image_validation_ms"], "5.0")
        self.assertEqual(records[2]["claim_eligible"], "false")
        self.assertEqual(records[2]["independent_reference_qualified"], "false")
        self.assertEqual(
            json.loads(records[2]["convergence_binding_json"]),
            frozen["convergence_binding"],
        )
        self.assertEqual(
            json.loads(records[2]["risk_band_by_case_id_json"]),
            frozen["risk_band_by_case_id"],
        )
        self.assertEqual(records[2]["input_snapshot_release_ms"], "1.0")
        self.assertEqual(records[2]["planning_inclusive_end_to_end_median_ms"], "7.0")


if __name__ == "__main__":
    unittest.main()
