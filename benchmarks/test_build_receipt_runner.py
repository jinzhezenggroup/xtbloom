"""Record-only receipt integration tests using a fake native benchmark path."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from benchmarks import run


class ReceiptRunnerTests(unittest.TestCase):
    """No source receipt may independently qualify numerical or performance data."""

    def setUp(self) -> None:
        """Keep all native setup and output writes behind explicit test doubles."""
        self.base = [
            "--library",
            "/tmp/library-fixture.so",
            "--case-ids",
            "a",
            "--engines",
            "xtbloom",
            "--backends",
            "cpu",
            "--cuda-memory-modes",
            "host",
            "--ao-grouping",
            "paired",
            "--batch-sizes",
            "1",
            "--properties",
            "energy",
        ]
        self.flags = [
            "--build-receipt",
            "/tmp/receipt-fixture.json",
            "--build-receipt-sha256",
            "a" * 64,
            "--build-source-revision",
            "b" * 40,
        ]
        self.verified = {
            "backend": "cpu",
            "status": "LOCAL_RECEIPT_MATCHES",
            "performance_claim_eligible": False,
            "receipt": {"sha256": "a" * 64},
            "source_revision": "b" * 40,
        }

    def test_partial_or_malformed_pins_and_nonpaired_modes_fail_before_setup(
        self,
    ) -> None:
        """Argument-level evidence failures never enter native or workload setup."""
        bad_flags = [self.flags[:2], self.flags[2:4], self.flags[4:]]
        for field in ("--build-receipt-sha256", "--build-source-revision"):
            malformed = self.flags.copy()
            malformed[malformed.index(field) + 1] = "bad-pin"
            bad_flags.append(malformed)
        for flags in bad_flags:
            with (
                self.subTest(flags=flags),
                mock.patch.object(run.conformance, "load_json") as load,
                mock.patch.object(run, "XTBloomAdapter") as adapter,
                mock.patch.object(run.build_receipt, "verify_receipt") as verify,
                mock.patch("sys.stderr"),
            ):
                self.assertEqual(run.main(self.base + flags), 1)
                load.assert_not_called()
                adapter.assert_not_called()
                verify.assert_not_called()
        for strategy in ("original", "exact-ao"):
            arguments = self.base.copy()
            arguments[arguments.index("--ao-grouping") + 1] = strategy
            with self.assertRaisesRegex(
                run.BenchmarkError, "record-only in paired mode"
            ):
                run.validate_args(run.build_parser().parse_args(arguments + self.flags))

    def test_default_has_no_receipt_io_and_backend_mismatch_is_rejected(self) -> None:
        """Preserve the old unbound diagnostic mode; CPU receipts cannot claim CUDA."""
        args = run.build_parser().parse_args(self.base)
        with mock.patch.object(run.build_receipt, "verify_receipt") as verify:
            self.assertIsNone(run.controlled_receipt_check(args))
            verify.assert_not_called()
        args = run.build_parser().parse_args(self.base + self.flags)
        with (
            mock.patch.object(
                run.build_receipt,
                "verify_receipt",
                return_value=dict(self.verified, backend="cuda"),
            ),
            self.assertRaisesRegex(run.BenchmarkError, "backend differs"),
        ):
            run.controlled_receipt_check(args)

    def test_matching_receipt_is_record_only_and_costs_are_retained(self) -> None:
        """Separate selected-file/source integrity from loaded images and science."""
        args = run.build_parser().parse_args(self.base + self.flags)
        with mock.patch.object(
            run.build_receipt, "verify_receipt", return_value=self.verified
        ):
            preflight = run.controlled_receipt_check(args)
            row = {
                "producer_provenance": {"status": "UNVERIFIED"},
                "claim_ineligibility_reasons": [],
                "claim_eligible": False,
            }
            metadata = {}
            self.assertTrue(run.receipt_postflight(args, preflight, metadata, [row]))
        evidence = metadata["controlled_build_receipt"]
        self.assertEqual(evidence["status"], "LOCAL_RECEIPT_MATCHES")
        self.assertFalse(evidence["performance_admission"])
        self.assertGreaterEqual(preflight["validation_ms"], 0.0)
        self.assertEqual(preflight["runtime_loaded_image_binding"], "NOT_ESTABLISHED")
        self.assertFalse(row["claim_eligible"])
        self.assertEqual(row["producer_provenance"]["status"], "UNVERIFIED")

    def test_postflight_failure_retains_all_computed_rows_and_publishes_artifacts(
        self,
    ) -> None:
        """Integrity failures return nonzero without erasing successful peer data."""
        manifest = {"method": "GFN2-xTB", "cases": [{"id": "a"}]}
        row = run.base_row(
            run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1, ("a",))
        )
        row.update(
            availability="available",
            correctness={"status": "UNQUALIFIED"},
            producer_provenance={"status": "UNVERIFIED"},
            claim_ineligibility_reasons=[],
            claim_eligible=False,
            case_results=[{"case_id": "a", "energy_hartree": -1.0, "system_status": 0}],
        )
        plan = run.ao_grouping.make_plan(("a",), 1)
        with (
            mock.patch.object(
                run.build_receipt,
                "verify_receipt",
                side_effect=[
                    self.verified,
                    run.build_receipt.ReceiptError("library changed"),
                ],
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
            self.assertEqual(run.main(self.base + self.flags), 1)
        document = write_json.call_args.args[1]
        self.assertEqual(document["rows"][0]["case_results"], row["case_results"])
        self.assertEqual(document["rows"][0]["availability"], "available")
        self.assertEqual(
            document["metadata"]["controlled_build_receipt"]["status"],
            "POSTFLIGHT_FAILED",
        )
        self.assertFalse(document["rows"][0]["claim_eligible"])
        write_csv.assert_called_once()

    def test_changed_postflight_identity_is_not_silently_rebound(self) -> None:
        """Even newly re-pinned metadata cannot replace the initial experiment."""
        args = run.build_parser().parse_args(self.base + self.flags)
        with mock.patch.object(
            run.build_receipt, "verify_receipt", return_value=self.verified
        ):
            preflight = run.controlled_receipt_check(args)
        changed = copy.deepcopy(self.verified)
        changed["receipt"]["sha256"] = "c" * 64
        row = {
            "producer_provenance": {"status": "UNVERIFIED"},
            "claim_ineligibility_reasons": [],
        }
        metadata = {}
        with mock.patch.object(
            run.build_receipt, "verify_receipt", return_value=changed
        ):
            self.assertFalse(run.receipt_postflight(args, preflight, metadata, [row]))
        postflight = metadata["controlled_build_receipt"]["postflight"]
        self.assertEqual(postflight["verification"]["receipt"]["sha256"], "c" * 64)
        self.assertIn("identity changed", postflight["error"])

    def test_preflight_receipt_failure_leaves_workload_and_native_not_run(self) -> None:
        """An invalid requested receipt fails before any coordinate is attempted."""
        with (
            mock.patch.object(
                run.build_receipt,
                "verify_receipt",
                side_effect=run.build_receipt.ReceiptError("wrong receipt"),
            ),
            mock.patch.object(run.conformance, "load_json") as load,
            mock.patch.object(run, "XTBloomAdapter") as adapter,
            mock.patch.object(run, "write_json") as write,
            mock.patch("sys.stderr"),
        ):
            self.assertEqual(run.main(self.base + self.flags), 1)
        load.assert_not_called()
        adapter.assert_not_called()
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
