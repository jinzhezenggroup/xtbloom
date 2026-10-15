"""Tests for the benchmark runner's default finite-queue policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from benchmarks import ao_grouping, run


class AutoGroupingRunnerTests(unittest.TestCase):
    """Keep automatic grouping within its validated workload boundary."""

    def _plan(
        self,
        ao_counts: tuple[int, ...],
        *,
        backend: str = "cuda",
        memory_mode: str = "host",
        model_tag: int | None = None,
    ) -> tuple[ao_grouping.AOGroupingPlan, SimpleNamespace]:
        case_ids = tuple(chr(ord("a") + index) for index in range(len(ao_counts)))
        count_by_id = dict(zip(case_ids, ao_counts, strict=True))
        cases = {case_id: {"id": case_id} for case_id in case_ids}
        args = SimpleNamespace(
            ao_grouping="auto",
            manifest=Path("fixture-manifest.json"),
            backends=(backend,),
            convergence_plan=None,
        )
        actual_model_tag = (
            run.public_api.XTBLOOM_MODEL_GFN2_XTB if model_tag is None else model_tag
        )

        def atomic_numbers(
            _manifest_path: Path,
            _manifest: dict[str, Any],
            case: dict[str, Any],
        ) -> tuple[int, ...]:
            return (1,) if count_by_id[case["id"]] == 1 else (6,)

        with (
            mock.patch.object(
                run.public_api, "model_tag", return_value=actual_model_tag
            ),
            mock.patch.object(
                ao_grouping,
                "load_gfn2_basis_ao_counts",
                return_value=({1: 1, 6: 4}, "a" * 64),
            ) as load_basis,
            mock.patch.object(run, "case_atomic_numbers", side_effect=atomic_numbers),
        ):
            plan, _planning_ms = run.finite_case_plan(
                args,
                {"model": "fixture"},
                cases,
                case_ids,
                2,
                memory_mode=memory_mode,
            )
        if (
            backend != "cuda"
            or memory_mode != "host"
            or actual_model_tag != (run.public_api.XTBLOOM_MODEL_GFN2_XTB)
        ):
            load_basis.assert_not_called()
        return plan, args

    def test_groups_only_when_equal_ao_cases_cross_original_batch_boundaries(
        self,
    ) -> None:
        """Resolve to exact-AO only when it creates shared-AO batches."""
        plan, args = self._plan((4, 1, 4, 1))
        self.assertEqual(plan.strategy, "exact-ao")
        self.assertEqual(plan.ordered_case_ids, ("b", "d", "a", "c"))
        self.assertEqual(args.finite_ao_grouping_resolution["requested"], "auto")
        self.assertEqual(args.finite_ao_grouping_resolution["resolved"], "exact-ao")
        self.assertTrue(args.finite_ao_grouping_resolution["reason"])
        self.assertEqual(len(plan.plan_sha256), 64)
        args.library = Path("fixture-library.so")
        args.device_id = 0
        args.cpu_threads = 1
        args.warmups = 0
        args.repetitions = 1
        cases = {case_id: {"id": case_id} for case_id in plan.original_case_ids}
        with mock.patch.object(
            run,
            "XTBloomAdapter",
            side_effect=run.public_api.BackendUnavailable("mock unavailable"),
        ):
            row = run.benchmark_finite_xtbloom_cell(
                args, {"model": "fixture"}, cases, plan, 0.0, "energy"
            )
        self.assertEqual(row["ao_grouping"], "exact-ao")
        self.assertEqual(row["ao_grouping_requested"], "auto")
        self.assertEqual(row["ao_grouping_resolved"], "exact-ao")
        self.assertEqual(
            row["ao_grouping_reason"],
            args.finite_ao_grouping_resolution["reason"],
        )
        self.assertEqual(row["plan_sha256"], plan.plan_sha256)

    def test_singleton_homogeneous_and_already_grouped_queues_stay_original(
        self,
    ) -> None:
        """Avoid sorting when exact-AO cannot consolidate original batches."""
        for ao_counts in ((4,), (4, 4, 4, 4), (4, 4, 1, 1)):
            with self.subTest(ao_counts=ao_counts):
                plan, args = self._plan(ao_counts)
                self.assertEqual(plan.strategy, "original")
                self.assertEqual(
                    args.finite_ao_grouping_resolution["resolved"], "original"
                )
                self.assertTrue(args.finite_ao_grouping_resolution["reason"])
                self.assertEqual(len(plan.plan_sha256), 64)
                self.assertEqual(plan.ao_counts_by_case_id, ())
                self.assertIsNone(plan.basis_sha256)
                self.assertEqual(
                    plan.plan_sha256,
                    ao_grouping.make_plan(
                        tuple(chr(ord("a") + index) for index in range(len(ao_counts))),
                        2,
                        strategy="original",
                    ).plan_sha256,
                )

    def test_unsupported_auto_coordinates_keep_original_and_explain_why(
        self,
    ) -> None:
        """Preserve input order for CPU, GFN1, and non-host CUDA descriptors."""
        cases = (
            ("cpu", "host", run.public_api.XTBLOOM_MODEL_GFN2_XTB, "CUDA backend"),
            ("cuda", "host", run.public_api.XTBLOOM_MODEL_GFN2_XTB + 1, "GFN2"),
            ("cuda", "device", run.public_api.XTBLOOM_MODEL_GFN2_XTB, "host"),
            ("cuda", "mixed", run.public_api.XTBLOOM_MODEL_GFN2_XTB, "host"),
        )
        for backend, memory_mode, model_tag, reason_fragment in cases:
            with self.subTest(backend=backend, mode=memory_mode, model=model_tag):
                plan, args = self._plan(
                    (4, 1, 4, 1),
                    backend=backend,
                    memory_mode=memory_mode,
                    model_tag=model_tag,
                )
                self.assertEqual(plan.strategy, "original")
                self.assertIn(
                    reason_fragment,
                    args.finite_ao_grouping_resolution["reason"],
                )

    def test_explicit_exact_ao_rejects_non_gfn2_manifests(self) -> None:
        """Keep unsupported explicit exact-AO requests as clear errors."""
        args = SimpleNamespace(
            ao_grouping="exact-ao",
            manifest=Path("fixture-manifest.json"),
            backends=("cpu",),
            convergence_plan=None,
        )
        with (
            mock.patch.object(
                run.public_api,
                "model_tag",
                return_value=run.public_api.XTBLOOM_MODEL_GFN2_XTB + 1,
            ),
            self.assertRaisesRegex(run.BenchmarkError, "requires a GFN2"),
        ):
            run.finite_case_plan(args, {}, {"a": {"id": "a"}}, ("a",), 1)

    def test_auto_keeps_legacy_host_only_validation_for_finite_cuda_lists(
        self,
    ) -> None:
        """Do not expose device or mixed finite-run execution through auto."""
        for memory_mode in ("device", "mixed"):
            with self.subTest(memory_mode=memory_mode):
                args = run.build_parser().parse_args(
                    [
                        "--library",
                        "lib.so",
                        "--case-ids",
                        "a,b",
                        "--engines",
                        "xtbloom",
                        "--backends",
                        "cuda",
                        "--cuda-memory-modes",
                        memory_mode,
                    ]
                )
                with self.assertRaisesRegex(
                    run.BenchmarkError, "requires host descriptors"
                ):
                    run.validate_args(args)

    def test_nonfinite_auto_is_recorded_as_original_without_finite_planning(
        self,
    ) -> None:
        """Keep the ordinary matrix path and expose its automatic fallback."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = run.Cell("xtbloom", "cpu", "host", "gas", "energy", 1)
            row = run.base_row(cell)
            row.update({"availability": "available", "correctness": {"status": "pass"}})
            documents: list[dict[str, Any]] = []
            with (
                mock.patch.object(run.conformance, "load_json", return_value={}),
                mock.patch.object(
                    run.conformance,
                    "selected_cases",
                    return_value=({"id": "ketene"},),
                ),
                mock.patch.object(run, "environment_metadata", return_value={}),
                mock.patch.object(run, "input_provenance", return_value={}),
                mock.patch.object(run, "xtbloom_cells", return_value=(cell,)),
                mock.patch.object(run, "workload_case_sequence", return_value=()),
                mock.patch.object(run, "benchmark_xtbloom_cell", return_value=row),
                mock.patch.object(
                    run, "finite_case_plan", side_effect=AssertionError("finite plan")
                ),
                mock.patch.object(
                    run,
                    "write_json",
                    side_effect=lambda _path, document: documents.append(document),
                ),
                mock.patch.object(run, "write_csv"),
            ):
                status = run.main(
                    [
                        "--library",
                        "lib.so",
                        "--engines",
                        "xtbloom",
                        "--backends",
                        "cpu",
                        "--cuda-memory-modes",
                        "host",
                        "--workloads",
                        "gas",
                        "--properties",
                        "energy",
                        "--batch-sizes",
                        "1",
                        "--warmups",
                        "0",
                        "--repetitions",
                        "1",
                        "--output-json",
                        str(root / "report.json"),
                        "--output-csv",
                        str(root / "report.csv"),
                    ]
                )
        self.assertEqual(status, 0)
        protocol = documents[0]["protocol"]
        self.assertEqual(
            protocol["ao_grouping_resolution"],
            {
                "requested": "auto",
                "resolved": "original",
                "reason": "auto AO grouping requires an explicit finite case list",
            },
        )
        self.assertEqual(documents[0]["rows"][0]["ao_grouping_resolved"], "original")
        self.assertEqual(
            documents[0]["rows"][0]["ao_grouping_reason"],
            protocol["ao_grouping_resolution"]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
