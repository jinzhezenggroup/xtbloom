"""Fast self-checks for benchmark matrix construction and serialization."""

from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from benchmarks import run, tblite_adapter, xtb_adapter
from benchmarks.tblite_adapter import TbliteAdapter, TbliteError, TbliteState
from benchmarks.xtb_adapter import XtbAdapter, XtbError, XtbState


class HarnessTest(unittest.TestCase):
    """Exercise non-hardware protocol logic without loading xtbloom or CUDA."""

    def test_timing_summary_retains_samples_and_batch_throughput(self) -> None:
        """Retain raw samples and derive batch throughput from their median."""
        summary = run.timing_summary([3.0, 1.0, 2.0], batch_size=8)
        self.assertEqual(summary["samples_ms"], [3.0, 1.0, 2.0])
        self.assertEqual(summary["median_ms"], 2.0)
        self.assertEqual(summary["systems_per_second_at_median"], 4000.0)

    def test_environment_metadata_records_cuda_shell_pair_schedule(self) -> None:
        """Retain the same-binary shell-pair schedule used by CUDA evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libxtbloom.so"
            library.write_bytes(b"xtbloom")
            args = SimpleNamespace(
                cuda_root=root / "cuda",
                tblite_executable=None,
                xtb_executable=None,
                dxtb_executable=None,
                xtb_library=None,
                tblite_library=None,
                dxtb_source=None,
                dxtb_backends=("cpu", "cuda"),
                dxtb_cpu_threads=1,
                library=library,
            )
            with (
                mock.patch.dict(
                    os.environ, {"XTBLOOM_CUDA_SHELL_PAIR_SCHEDULE": "compact"}
                ),
                mock.patch.object(run, "discover_reference", return_value={}),
                mock.patch.object(run, "git_state", return_value={"dirty": False}),
                mock.patch.object(run, "run_text", return_value=None),
            ):
                metadata = run.environment_metadata(args)
        self.assertEqual(
            metadata["environment"]["XTBLOOM_CUDA_SHELL_PAIR_SCHEDULE"],
            "compact",
        )

    def test_reference_thread_budget_keeps_blas_single_threaded(self) -> None:
        """Do not multiply the declared CPU budget through nested BLAS workers."""
        for module, loader_name in (
            (xtb_adapter, "ctypes.CDLL"),
            (tblite_adapter, "_load_first"),
        ):
            openmp = SimpleNamespace(
                omp_set_dynamic=mock.Mock(), omp_set_num_threads=mock.Mock()
            )
            blas = SimpleNamespace(openblas_set_num_threads=mock.Mock())
            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch(
                    f"benchmarks.{module.__name__.split('.')[-1]}.{loader_name}",
                    side_effect=(openmp, blas),
                ),
            ):
                controls = module._configure_runtime_threads(Path("/tmp"), 16)
                openmp.omp_set_num_threads.assert_called_once_with(16)
                blas.openblas_set_num_threads.assert_called_once_with(1)
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "16")
                self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
                self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
                self.assertEqual(controls["openmp_threads"], 16)
                self.assertEqual(controls["blas_threads"], 1)

    def test_xtbloom_cell_matrix_contains_cpu_and_three_cuda_placements(self) -> None:
        """Cover CPU host and all supported CUDA descriptor placements."""
        args = SimpleNamespace(
            backends=("cpu", "cuda"),
            cuda_memory_modes=("host", "device", "mixed"),
            workloads=("gas", "qmmm"),
            properties=("energy", "force"),
            batch_sizes=(1, 8, 32, 128),
        )
        cells = list(run.xtbloom_cells(args))
        self.assertEqual(len(cells), 64)
        placements = {(cell.backend, cell.memory_mode) for cell in cells}
        self.assertEqual(
            placements,
            {("cpu", "host"), ("cuda", "host"), ("cuda", "device"), ("cuda", "mixed")},
        )

    def test_heterogeneous_case_sequences_cycle_by_workload_class(self) -> None:
        """Cycle committed gas and QM/MM cases deterministically for large B."""
        gas = run.workload_case_ids("heterogeneous-gas", 8)
        qmmm = run.workload_case_ids("heterogeneous-qmmm", 8)
        self.assertEqual(
            gas,
            (
                *run.HETEROGENEOUS_WORKLOAD_CASES["heterogeneous-gas"],
                "h3_plus",
                "ketene",
                "nenacl",
                "sif5_minus",
            ),
        )
        self.assertEqual(
            qmmm,
            (
                *run.HETEROGENEOUS_WORKLOAD_CASES["heterogeneous-qmmm"],
                "water_one_pc_gamma999",
                "water_dimer_6pc_hardness",
                "water_dimer_6pc_gamma999",
                "water_one_pc_gamma999",
                "water_dimer_6pc_hardness",
            ),
        )
        self.assertTrue(all("water" not in case_id for case_id in gas))
        self.assertTrue(all(case_id.startswith("water_") for case_id in qmmm))

    def test_homogeneous_defaults_and_row_identity_remain_unchanged(self) -> None:
        """Retain scalar case IDs for the original default matrix coordinates."""
        self.assertEqual(run.DEFAULT_WORKLOADS, ("gas", "qmmm"))
        self.assertEqual(run.workload_case_ids("gas", 3), ("ketene",) * 3)
        row = run.base_row(run.Cell("xtbloom", "cpu", "host", "gas", "force", 3))
        self.assertEqual(row["case_id"], "ketene")
        self.assertNotIn("case_ids", row)

    def test_heterogeneous_rows_and_csv_preserve_every_case_id(self) -> None:
        """Serialize the exact ragged case sequence instead of one misleading ID."""
        cell = run.Cell("xtbloom", "cuda", "host", "heterogeneous-gas", "energy", 8)
        row = run.unavailable_row(cell, "test")
        self.assertNotIn("case_id", row)
        self.assertEqual(row["case_ids"], list(run.workload_case_ids(cell.workload, 8)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            run.write_csv(path, [row])
            with path.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))
        self.assertEqual(json.loads(written["case_ids"]), row["case_ids"])

    def test_xtbloom_adapter_receives_exact_case_sequence(self) -> None:
        """Pass heterogeneous cases to the public batch assembler without repetition."""
        sequence = tuple({"id": value} for value in ("a", "b", "c"))
        cell = run.Cell("xtbloom", "cpu", "host", "heterogeneous-gas", "energy", 3)
        fake_library = SimpleNamespace()
        with (
            mock.patch.object(
                run.public_api, "_configure_library", return_value=fake_library
            ),
            mock.patch.object(
                run.public_api, "assemble_batch", side_effect=RuntimeError("stop")
            ) as assemble,
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
            run.XTBloomAdapter(
                Path("lib.so"), Path("manifest.json"), {}, sequence, cell, 0, 1
            )
        self.assertEqual(assemble.call_args.args[2], sequence)

    def test_public_timing_semantics_label_is_exact(self) -> None:
        """Use the agreed repeated-compute term and avoid a list-cache claim."""
        self.assertEqual(run.REPEATED_CALL_SEMANTICS, "same_geometry_repeated_compute")
        self.assertNotIn("reuse", run.REPEATED_CALL_SEMANTICS)

    def test_xtb_matrix_is_serial_cpu_host_and_templates_are_strict(self) -> None:
        """Keep reference matrices serial and command templates reproducible."""
        args = SimpleNamespace(
            workloads=("gas", "qmmm"),
            properties=("energy", "force"),
            batch_sizes=(1, 8, 32, 128),
        )
        cells = list(run.xtb_cells(args))
        self.assertEqual(len(cells), 16)
        self.assertEqual(
            {(cell.backend, cell.memory_mode) for cell in cells},
            {("cpu", "host")},
        )
        for engine in ("tblite", "xtb"):
            template = run.REFERENCE_COMMANDS[engine]
            self.assertIn("0.0001", template)
            self.assertIn("OMP_NUM_THREADS=1", template)
            self.assertIn("OPENBLAS_NUM_THREADS=1", template)
        self.assertEqual(run.REFERENCE_COMMANDS["xtb"][-1], "1")

        tblite_cells = list(run.tblite_cells(args))
        self.assertEqual(len(tblite_cells), 16)
        self.assertEqual(
            {(cell.backend, cell.memory_mode) for cell in tblite_cells},
            {("cpu", "host")},
        )

    def test_dxtb_matrix_contains_persistent_cpu_and_cuda_rows(self) -> None:
        """Emit persistent dxtb rows for both supported compute devices."""
        args = SimpleNamespace(
            workloads=("gas", "qmmm"),
            properties=("energy", "force"),
            batch_sizes=(1, 8, 32, 128),
            dxtb_backends=("cpu", "cuda"),
        )
        cells = list(run.dxtb_cells(args))
        self.assertEqual(len(cells), 32)
        self.assertEqual(
            {(cell.backend, cell.memory_mode) for cell in cells},
            {("cpu", "host"), ("cuda", "device")},
        )

    def test_xtb_result_normalization_uses_force_sign_and_optional_pc_output(
        self,
    ) -> None:
        """Convert xTB gradients to QM and point-charge force conventions."""
        adapter = object.__new__(XtbAdapter)
        adapter.property_name = "force"
        adapter.states = [
            XtbState(
                environment=None,
                molecule=None,
                calculator=None,
                result=None,
                positions=None,
                energy=SimpleNamespace(value=-2.0),
                gradient=[1.0, -2.0, 3.0],
                point_gradient=[-4.0, 5.0, -6.0],
                has_external_charges=True,
                point_count=None,
                point_numbers=None,
                point_charges=None,
                point_positions=None,
                keepalive=(),
            )
        ]
        output = adapter.results()
        self.assertEqual(output["energies_hartree"], [-2.0])
        self.assertEqual(output["forces_hartree_per_bohr"], [-1.0, 2.0, -3.0])
        self.assertEqual(
            output["point_charge_forces_hartree_per_bohr"], [4.0, -5.0, 6.0]
        )

    def test_tblite_result_normalization_uses_force_sign_and_charges(self) -> None:
        """Convert tblite gradients while preserving requested atomic charges."""
        adapter = object.__new__(TbliteAdapter)
        adapter.property_name = "force"
        adapter.states = [
            TbliteState(
                error=None,
                context=None,
                structure=None,
                calculator=None,
                result=None,
                positions=None,
                energy=SimpleNamespace(value=-3.0),
                gradient=[1.0, -2.0, 3.0],
                charges=[0.25, -0.25],
                keepalive=(),
            )
        ]
        output = adapter.results()
        self.assertEqual(output["energies_hartree"], [-3.0])
        self.assertEqual(output["forces_hartree_per_bohr"], [-1.0, 2.0, -3.0])
        self.assertEqual(output["atomic_charges_e"], [0.25, -0.25])

    def test_xtb_restart_failure_clears_owned_handles(self) -> None:
        """A failed cold rebuild must not leave double-freeable xTB handles."""
        adapter = object.__new__(XtbAdapter)
        adapter.accuracy = 1.0e-4
        adapter.max_iterations = 500
        adapter.electronic_temperature_kelvin = 300.0
        adapter.library = SimpleNamespace(
            xtb_delResults=mock.Mock(),
            xtb_delCalculator=mock.Mock(),
            xtb_delMolecule=mock.Mock(),
            xtb_delEnvironment=mock.Mock(),
            xtb_newCalculator=mock.Mock(return_value=101),
            xtb_newResults=mock.Mock(return_value=0),
        )
        adapter.states = [
            XtbState(
                environment=ctypes.c_void_p(1),
                molecule=ctypes.c_void_p(2),
                calculator=ctypes.c_void_p(3),
                result=ctypes.c_void_p(4),
                positions=None,
                energy=ctypes.c_double(),
                gradient=None,
                point_gradient=None,
                has_external_charges=False,
                point_count=None,
                point_numbers=None,
                point_charges=None,
                point_positions=None,
                keepalive=(),
            )
        ]
        state = adapter.states[0]
        with self.assertRaisesRegex(XtbError, "allocation returned NULL"):
            adapter.restart_scc()
        self.assertFalse(state.calculator)
        self.assertFalse(state.result)
        self.assertEqual(adapter.library.xtb_delCalculator.call_count, 2)
        self.assertEqual(adapter.library.xtb_delResults.call_count, 1)
        adapter.close()
        self.assertEqual(adapter.library.xtb_delCalculator.call_count, 2)
        self.assertEqual(adapter.library.xtb_delResults.call_count, 1)

    def test_tblite_restart_failure_clears_owned_handles(self) -> None:
        """A failed cold rebuild must not leave double-freeable tblite handles."""
        adapter = object.__new__(TbliteAdapter)
        adapter.library = SimpleNamespace(
            tblite_delete_result=mock.Mock(),
            tblite_delete_calculator=mock.Mock(),
            tblite_delete_structure=mock.Mock(),
            tblite_delete_context=mock.Mock(),
            tblite_delete_error=mock.Mock(),
            tblite_new_gfn2_calculator=mock.Mock(return_value=101),
            tblite_new_result=mock.Mock(return_value=0),
        )
        adapter._check_context = mock.Mock()
        adapter._configure_calculator = mock.Mock()
        adapter.states = [
            TbliteState(
                error=ctypes.c_void_p(1),
                context=ctypes.c_void_p(2),
                structure=ctypes.c_void_p(3),
                calculator=ctypes.c_void_p(4),
                result=ctypes.c_void_p(5),
                positions=None,
                energy=ctypes.c_double(),
                gradient=None,
                charges=None,
                keepalive=(),
            )
        ]
        state = adapter.states[0]
        with self.assertRaisesRegex(TbliteError, "allocation returned NULL"):
            adapter.restart_scc()
        self.assertFalse(state.calculator)
        self.assertFalse(state.result)
        self.assertEqual(adapter.library.tblite_delete_calculator.call_count, 2)
        self.assertEqual(adapter.library.tblite_delete_result.call_count, 1)
        adapter.close()
        self.assertEqual(adapter.library.tblite_delete_calculator.call_count, 2)
        self.assertEqual(adapter.library.tblite_delete_result.call_count, 1)

    def test_correctness_includes_qm_and_point_charge_forces(self) -> None:
        """Gate both QM and point-charge force vectors for QMMM workloads."""
        expected = {
            "energy_hartree": -1.0,
            "forces_hartree_per_bohr": [0.1, 0.2, 0.3],
            "point_charge_forces_hartree_per_bohr": [-0.1, -0.2, -0.3],
        }
        storage = SimpleNamespace(
            slices=[
                SimpleNamespace(
                    atom_begin=0,
                    atom_end=1,
                    point_begin=0,
                    point_end=1,
                    expected=expected,
                )
            ]
        )
        manifest = {
            "tolerances": {
                "energy": {"atol": 1.0e-6},
                "forces": {"atol": 1.0e-6},
                "point_charge_forces": {"atol": 1.0e-6},
            }
        }
        output = {
            "energies_hartree": [-1.0],
            "forces_hartree_per_bohr": [0.1, 0.2, 0.3],
            "point_charge_forces_hartree_per_bohr": [-0.1, -0.2, -0.3],
        }
        cell = run.Cell("xtbloom", "cpu", "host", "qmmm", "force", 1)
        result = run.correctness(cell, storage, output, manifest)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["max_abs_point_charge_force_error_hartree_per_bohr"], 0.0
        )

    def test_json_and_csv_preserve_unavailable_rows(self) -> None:
        """Preserve unavailable benchmark cells in both artifact formats."""
        cell = run.Cell("tblite", "cpu", "host", "gas", "force", 8)
        row = run.unavailable_row(cell, "missing library")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "matrix.json"
            csv_path = root / "matrix.csv"
            run.write_json(json_path, {"schema_version": 1, "rows": [row]})
            run.write_csv(csv_path, [row])
            self.assertEqual(json.loads(json_path.read_text())["rows"][0], row)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                written = next(csv.DictReader(handle))
            self.assertEqual(written["availability"], "unavailable")
        self.assertEqual(written["unavailable_reason"], "missing library")


class PairedRunnerTest(unittest.TestCase):
    """Exercise interleaved finite-list execution with persistent fake owners."""

    def _record(self, case_id: str) -> dict[str, object]:
        index = "abcd".index(case_id)
        case = self.cases[case_id]
        atom_count = case["atom_count"]
        point_count = case["point_charge_count"]
        record: dict[str, object] = {
            "energy_hartree": -1.0 - index,
            "partial_charges_e": [
                index + atom_index / 10.0 for atom_index in range(atom_count)
            ],
            "forces_hartree_per_bohr": [
                index + component / 100.0 for component in range(3 * atom_count)
            ],
            "status": run.public_api.XTBLOOM_STATUS_SUCCESS,
            "scc_converged": 1,
            "scc_iterations": 3,
        }
        if point_count:
            record["point_charge_forces_hartree_per_bohr"] = [
                index + component / 100.0 for component in range(3 * point_count)
            ]
        return record

    def _run_pair(
        self,
        *,
        property_name: str = "force",
        warmups: int = 1,
        repetitions: int = 2,
        missing_reference_ids: tuple[str, ...] = (),
        behavior: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], list[tuple[object, ...]], list[object]]:
        behavior = behavior or {}
        ids = ("a", "b", "c", "d")
        atom_counts = {"a": 1, "b": 2, "c": 1, "d": 3}
        point_counts = {"a": 1, "b": 0, "c": 2, "d": 1}
        ao_counts = {"a": 2, "b": 1, "c": 2, "d": 1}
        self.cases = {
            case_id: {
                "id": case_id,
                "atom_count": atom_counts[case_id],
                "point_charge_count": point_counts[case_id],
                **(
                    {"oracle_role": "diagnostic-no-independent-reference"}
                    if case_id in missing_reference_ids
                    else {}
                ),
            }
            for case_id in ids
        }
        expected = {
            case_id: ({} if case_id in missing_reference_ids else self._record(case_id))
            for case_id in ids
        }
        manifest = {
            "tolerances": {
                "energy": {"atol": 1.0e-12},
                "charges": {"atol": 1.0e-12},
                "forces": {"atol": 1.0e-12},
                "point_charge_forces": {"atol": 1.0e-12},
            }
        }
        basis_hash = "a" * 64
        plans = {
            "original": run.ao_grouping.make_plan(
                ids, 2, "original", ao_counts, basis_hash
            ),
            "exact-ao": run.ao_grouping.make_plan(
                ids, 2, "exact-ao", ao_counts, basis_hash
            ),
        }
        original_batches = {batch.case_ids for batch in plans["original"].batches}
        events: list[tuple[object, ...]] = []
        created: list[object] = []
        active: list[object] = []

        class FakeAdapter:
            def __init__(
                self,
                library_path: Path,
                manifest_path: Path,
                loaded_manifest: dict[str, object],
                case_sequence: tuple[dict[str, object], ...],
                cell: run.Cell,
                device_id: int,
                cpu_threads: int,
                **options: object,
            ) -> None:
                self.cell = cell
                self.case_ids = tuple(case["id"] for case in case_sequence)
                self.layout = (
                    "original" if self.case_ids in original_batches else "exact-ao"
                )
                events.append(("setup", self.layout, self.case_ids, options))
                failure = behavior.get("setup_failures", {}).get(
                    (self.layout, self.case_ids)
                )
                if failure is not None:
                    raise failure
                self.options = run.public_api.ComputeOptions()
                self.options.model = run.public_api.XTBLOOM_MODEL_GFN2_XTB
                self.options.flags = (
                    run.public_api.XTBLOOM_COMPUTE_ENERGY
                    | run.public_api.XTBLOOM_COMPUTE_ATOMIC_CHARGES
                )
                if property_name == "force":
                    self.options.flags |= run.public_api.XTBLOOM_COMPUTE_FORCES
                    if any(case["point_charge_count"] for case in case_sequence):
                        self.options.flags |= (
                            run.public_api.XTBLOOM_COMPUTE_POINT_CHARGE_FORCES
                        )
                self.options.scc_start_mode = run.public_api.XTBLOOM_SCC_START_FRESH
                self.options.max_scc_iterations = behavior.get(
                    "max_iterations_by_layout", {}
                ).get(self.layout, 321)
                self.options.charge_tolerance = 2.0e-9
                self.options.energy_tolerance = 3.0e-11
                self.options.electronic_temperature = 0.007
                self.calls = 0
                atom_offsets = [0]
                point_offsets = [0]
                slices = []
                for case in case_sequence:
                    atom_offsets.append(atom_offsets[-1] + case["atom_count"])
                    point_offsets.append(point_offsets[-1] + case["point_charge_count"])
                    slices.append(
                        SimpleNamespace(case=case, expected=expected[case["id"]])
                    )
                self.storage = SimpleNamespace(
                    slices=slices,
                    atom_offsets=atom_offsets,
                    point_charge_offsets=point_offsets,
                    point_charge_values=[0.0] * point_offsets[-1],
                )
                created.append(self)
                active.append(self)

            def results(self) -> dict[str, object]:
                events.append(("results", self.layout, self.case_ids, self.calls))
                failure = behavior.get("result_failures", {}).get(
                    (self.layout, self.case_ids)
                )
                if failure is not None:
                    raise failure
                energies = []
                charges = []
                forces = []
                point_forces = []
                iterations = []
                converged = []
                statuses = []
                failed_ids = behavior.get("failed_ids", set())
                iteration_overrides = behavior.get("iteration_overrides", {})
                for case_id in self.case_ids:
                    record = self_outer._record(case_id)
                    failure_status = case_id in failed_ids
                    energies.append(
                        math.nan if failure_status else record["energy_hartree"]
                    )
                    charges.extend(
                        [math.nan] * self_outer.cases[case_id]["atom_count"]
                        if failure_status
                        else record["partial_charges_e"]
                    )
                    forces.extend(
                        [math.nan] * (3 * self_outer.cases[case_id]["atom_count"])
                        if failure_status
                        else record["forces_hartree_per_bohr"]
                    )
                    point_count = self_outer.cases[case_id]["point_charge_count"]
                    if point_count:
                        point_forces.extend(
                            [math.nan] * (3 * point_count)
                            if failure_status
                            else record["point_charge_forces_hartree_per_bohr"]
                        )
                    iterations.append(
                        iteration_overrides.get(
                            (self.layout, case_id), record["scc_iterations"]
                        )
                    )
                    converged.append(0 if failure_status else 1)
                    statuses.append(
                        1 if failure_status else run.public_api.XTBLOOM_STATUS_SUCCESS
                    )
                output: dict[str, object] = {
                    "energies_hartree": energies,
                    "atomic_charges_e": charges,
                    "scc_iterations": iterations,
                    "scc_converged": converged,
                    "per_system_status": statuses,
                }
                if self.storage.point_charge_values:
                    output["point_charge_forces_hartree_per_bohr"] = point_forces
                if property_name == "force":
                    output["forces_hartree_per_bohr"] = forces
                return output

            def memory_snapshot(self) -> dict[str, int]:
                events.append(("memory", self.layout, self.case_ids))
                return {"active_owner_count": len(active), "host_process_hwm_bytes": 10}

            def close(self) -> None:
                events.append(("close", self.layout, self.case_ids))
                if self in active:
                    active.remove(self)
                if (self.layout, self.case_ids) in behavior.get(
                    "close_failures", set()
                ):
                    raise RuntimeError("fake close failure")

        self_outer = self
        args = SimpleNamespace(
            backends=("cpu",),
            library=Path("fake-library.so"),
            manifest=Path("fake-manifest.json"),
            device_id=0,
            cpu_threads=1,
            warmups=warmups,
            repetitions=repetitions,
        )

        def timed_fake_invoke(adapter: FakeAdapter) -> float:
            events.append(("invoke", adapter.layout, adapter.case_ids))
            failure = behavior.get("invoke_failures", {}).get(
                (adapter.layout, adapter.case_ids)
            )
            if failure is not None:
                run.time.sleep(0.002)
                raise failure
            adapter.calls += 1
            return 0.25

        split_batch_results = run.ao_grouping.split_batch_results

        def split_fake_batch(
            case_ids: tuple[str, ...],
            atom_offsets: list[int],
            point_offsets: list[int],
            output: dict[str, object],
            required_outputs: tuple[str, ...],
        ) -> tuple[dict[str, object], ...]:
            failure = behavior.get("publication_failures", {}).get(tuple(case_ids))
            if failure is not None:
                raise failure
            return split_batch_results(
                case_ids,
                atom_offsets,
                point_offsets,
                output,
                required_outputs,
            )

        with (
            mock.patch.object(run, "XTBloomAdapter", FakeAdapter),
            mock.patch.object(run, "timed_invoke", side_effect=timed_fake_invoke),
            mock.patch.object(
                run.ao_grouping,
                "split_batch_results",
                side_effect=split_fake_batch,
            ),
            mock.patch.object(run, "current_rss_bytes", return_value=10),
            mock.patch.object(run, "process_hwm_bytes", return_value=10),
        ):
            row = run.benchmark_finite_xtbloom_pair(
                args,
                manifest,
                self.cases,
                plans,
                {"original": 1.0, "exact-ao": 2.0},
                property_name,
            )
        return row, events, created

    def test_alternates_all_phases_and_reuses_each_layout_owner(self) -> None:
        """Use AB/BA order continuously and keep each batch adapter alive."""
        row, events, created = self._run_pair(
            property_name="energy", warmups=2, repetitions=3
        )
        rounds = row["paired_rounds"]
        self.assertEqual(
            [item["phase"] for item in rounds],
            ["cold", "warmup", "warmup", "measured", "measured", "measured"],
        )
        self.assertEqual(
            [item["execution_order"] for item in rounds],
            [
                ["original", "exact-ao"],
                ["exact-ao", "original"],
                ["original", "exact-ao"],
                ["exact-ao", "original"],
                ["original", "exact-ao"],
                ["exact-ao", "original"],
            ],
        )
        self.assertEqual(len(created), 4)
        self.assertTrue(all(adapter.calls == 6 for adapter in created))
        self.assertEqual(sum(event[0] == "setup" for event in events), 4)
        self.assertEqual(sum(event[0] == "close" for event in events), 4)
        setup_options = [event[3] for event in events if event[0] == "setup"]
        self.assertTrue(all(options["strict_fresh"] for options in setup_options))
        self.assertTrue(all(options["request_charges"] for options in setup_options))
        self.assertEqual(row["paired_timing"]["requested_round_count"], 6)
        self.assertEqual(row["paired_timing"]["measured_round_count"], 3)
        self.assertEqual(
            row["paired_timing"]["layout_samples"]["original"]["measured_end_to_end"][
                "requested_count"
            ],
            3,
        )
        self.assertEqual(
            [
                item["original_index"]
                for item in rounds[0]["arms"]["original"]["case_results"]
            ],
            [0, 1, 2, 3],
        )
        self.assertTrue(
            all(
                snapshot["snapshot"]["active_owner_count"] == 4
                for item in rounds
                for arm in item["arms"].values()
                for snapshot in arm["memory_snapshots"]
                if "snapshot" in snapshot
            )
        )

    def test_scatters_scalar_and_all_ragged_force_outputs(self) -> None:
        """Restore energies, charges, QM forces, and point-charge force slices."""
        energy_row, _, _ = self._run_pair(
            property_name="energy", warmups=0, repetitions=1
        )
        self.assertEqual(energy_row["correctness"]["status"], "pass")
        self.assertTrue(energy_row["independent_reference_qualified"])
        energy_results = energy_row["paired_rounds"][0]["arms"]["exact-ao"][
            "case_results"
        ]
        for case_id, result in zip(("a", "b", "c", "d"), energy_results, strict=True):
            self.assertEqual(result["case_id"], case_id)
            self.assertIsInstance(result["energy_hartree"], float)
            self.assertEqual(
                result["atomic_charges_e"], self._record(case_id)["partial_charges_e"]
            )

        force_row, _, _ = self._run_pair(
            property_name="force", warmups=0, repetitions=1
        )
        force_results = force_row["paired_rounds"][0]["arms"]["exact-ao"][
            "case_results"
        ]
        for case_id, result in zip(("a", "b", "c", "d"), force_results, strict=True):
            expected = self._record(case_id)
            self.assertEqual(
                result["forces_hartree_per_bohr"], expected["forces_hartree_per_bohr"]
            )
            self.assertEqual(
                len(result["forces_hartree_per_bohr"]),
                3 * self.cases[case_id]["atom_count"],
            )
            if self.cases[case_id]["point_charge_count"]:
                self.assertEqual(
                    result["point_charge_forces_hartree_per_bohr"],
                    expected["point_charge_forces_hartree_per_bohr"],
                )
            self.assertTrue(result["correctness"]["independent_reference_pass"])

    def test_missing_reference_and_branch_differences_never_qualify_pair(self) -> None:
        """Keep diagnostic equality separate from scientific qualification."""
        row, _, _ = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            missing_reference_ids=("a", "b", "c", "d"),
        )
        measured = row["paired_rounds"][-1]
        self.assertEqual(row["correctness"]["finite_output_status"], "pass")
        self.assertEqual(row["correctness"]["status"], "unqualified")
        self.assertEqual(measured["layout_equivalence"]["status"], "pass")
        self.assertFalse(measured["performance_comparison_eligible"])
        self.assertFalse(row["claim_eligible"])

        branch_row, _, _ = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            behavior={"iteration_overrides": {("exact-ao", "c"): 4}},
        )
        self.assertEqual(
            branch_row["branch_differences"][0]["differences"]["scc_iterations"],
            {"original": 3, "exact_ao": 4},
        )
        self.assertFalse(branch_row["claim_eligible"])

    def test_nan_system_and_batch_exception_preserve_peers_and_cleanup(self) -> None:
        """Retain peer-local NaNs, continue other batches, and report close errors."""
        failed_row, _, _ = self._run_pair(
            property_name="force",
            warmups=0,
            repetitions=1,
            behavior={"failed_ids": {"b"}},
        )
        failed_results = failed_row["paired_rounds"][-1]["arms"]["original"][
            "case_results"
        ]
        self.assertEqual(failed_results[0]["execution_state"], "completed")
        self.assertEqual(failed_results[1]["execution_state"], "system-failure")
        self.assertTrue(math.isnan(failed_results[1]["energy_hartree"]))
        self.assertTrue(
            all(
                math.isnan(value)
                for value in failed_results[1]["forces_hartree_per_bohr"]
            )
        )
        self.assertFalse(math.isnan(failed_results[0]["energy_hartree"]))

        failure = RuntimeError("fake batch result failure")
        row, events, created = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            behavior={
                "result_failures": {("original", ("a", "b")): failure},
                "close_failures": {("original", ("c", "d"))},
            },
        )
        rounds = row["paired_rounds"]
        errored = rounds[0]["arms"]["original"]["case_results"]
        self.assertEqual(
            [item["execution_state"] for item in errored],
            ["error", "error", "completed", "completed"],
        )
        self.assertTrue(
            any(
                event[0] == "invoke"
                and event[1] == "original"
                and event[2] == ("c", "d")
                for event in events
            )
        )
        self.assertEqual(len(row["cleanup_errors"]), 1)
        self.assertEqual(sum(event[0] == "close" for event in events), len(created))
        self.assertEqual(row["availability"], "error")
        self.assertFalse(row["claim_eligible"])
        original_timing = rounds[0]["arms"]["original"]["timing"]
        self.assertTrue(original_timing["compute_complete"])
        self.assertIsNone(original_timing["download_ms"])
        self.assertFalse(original_timing["download_complete"])
        self.assertEqual(original_timing["download_attempt_count"], 2)
        self.assertEqual(original_timing["download_success_count"], 1)
        self.assertGreater(original_timing["download_attempted_time_ms"], 0.0)

        invoke_error_row, _, _ = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            behavior={
                "invoke_failures": {
                    ("original", ("a", "b")): RuntimeError("fake invoke failure")
                }
            },
        )
        invoke_arm = invoke_error_row["paired_rounds"][0]["arms"]["original"]
        self.assertEqual(
            [item["execution_state"] for item in invoke_arm["case_results"]],
            ["error", "error", "completed", "completed"],
        )
        self.assertIsNone(invoke_arm["timing"]["compute_ms"])
        self.assertFalse(invoke_arm["timing"]["compute_complete"])
        self.assertEqual(invoke_arm["timing"]["compute_attempt_count"], 2)
        self.assertEqual(invoke_arm["timing"]["compute_success_count"], 1)
        self.assertGreater(invoke_arm["timing"]["compute_attempted_time_ms"], 0.0)
        self.assertIsNotNone(invoke_arm["timing"]["end_to_end_ms"])
        self.assertEqual(
            invoke_error_row["paired_timing"]["layout_samples"]["original"][
                "measured_end_to_end"
            ]["requested_count"],
            1,
        )

        publication_error_row, _, _ = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            behavior={
                "publication_failures": {
                    ("a", "b"): RuntimeError("fake publication failure")
                }
            },
        )
        publication_timing = publication_error_row["paired_rounds"][0]["arms"][
            "original"
        ]["timing"]
        self.assertTrue(publication_timing["download_complete"])
        self.assertIsNone(publication_timing["publication_ms"])
        self.assertFalse(publication_timing["publication_complete"])
        self.assertEqual(publication_timing["publication_attempt_count"], 2)
        self.assertEqual(publication_timing["publication_success_count"], 1)
        self.assertGreater(publication_timing["publication_attempted_time_ms"], 0.0)

    def test_scatter_failure_retains_ids_without_qualifying_fallback(self) -> None:
        """Fallback archival must not promote a failed canonical mapping to PASS."""
        with mock.patch.object(
            run.ao_grouping,
            "scatter_case_results",
            side_effect=run.BenchmarkError("fake canonical mapping failure"),
        ):
            row, _, _ = self._run_pair(warmups=0, repetitions=1)
        for pair_round in row["paired_rounds"]:
            for arm in pair_round["arms"].values():
                self.assertEqual(arm["status"], "fail")
                self.assertTrue(arm["sweep_errors"])
                self.assertEqual(
                    [result["case_id"] for result in arm["case_results"]],
                    list(self.cases),
                )
            self.assertFalse(pair_round["performance_comparison_eligible"])
        self.assertFalse(row["claim_eligible"])

    def test_cold_and_warmup_errors_block_claim_despite_passing_measured_rounds(
        self,
    ) -> None:
        """Every phase remains a gate even when later strict-FRESH calls recover."""
        actual_sweep = run._finite_pair_arm_sweep
        for failed_arm_index in (0, 2):
            with self.subTest(failed_arm_index=failed_arm_index):
                arm_index = 0

                def inject_phase_error(
                    manifest: dict[str, Any],
                    cases: dict[str, dict[str, Any]],
                    expected_by_id: dict[str, dict[str, Any]],
                    plan: run.ao_grouping.AOGroupingPlan,
                    owners: list[dict[str, Any]],
                    property_name: str,
                    failure_index: int = failed_arm_index,
                ) -> dict[str, Any]:
                    nonlocal arm_index
                    arm = actual_sweep(
                        manifest, cases, expected_by_id, plan, owners, property_name
                    )
                    if arm_index == failure_index:
                        arm["status"] = "fail"
                        arm["sweep_errors"].append("fake pre-measurement error")
                    arm_index += 1
                    return arm

                with mock.patch.object(
                    run, "_finite_pair_arm_sweep", side_effect=inject_phase_error
                ):
                    row, _, _ = self._run_pair(warmups=1, repetitions=1)
                self.assertTrue(
                    row["paired_rounds"][-1]["numerical_comparison_eligible"]
                )
                self.assertFalse(row["numerical_comparison_eligible"])
                self.assertFalse(row["claim_eligible"])
                self.assertIn(
                    "cold or warmup qualification failed or was incomplete",
                    row["claim_ineligibility_reasons"],
                )

    def test_options_are_actual_snapshots_and_provenance_stays_unverified(self) -> None:
        """Correct outputs and matching options cannot prove a binary's producer."""
        row, _, _ = self._run_pair(warmups=0, repetitions=1)
        self.assertTrue(row["compute_options_match"])
        self.assertTrue(row["numerical_comparison_eligible"])
        self.assertFalse(row["claim_eligible"])
        self.assertEqual(row["producer_provenance"]["status"], "UNVERIFIED")
        for owners in row["owner_engine_options"].values():
            for owner in owners:
                options = owner["engine_options"]["compute_options"]
                self.assertEqual(options["max_scc_iterations"], 321)
                self.assertEqual(options["electronic_temperature"], 0.007)
        self.assertTrue(
            all(
                not pair_round["performance_comparison_eligible"]
                for pair_round in row["paired_rounds"]
            )
        )
        mismatched_row, _, _ = self._run_pair(
            warmups=0,
            repetitions=1,
            behavior={"max_iterations_by_layout": {"exact-ao": 322}},
        )
        self.assertFalse(mismatched_row["compute_options_match"])
        self.assertFalse(mismatched_row["numerical_comparison_eligible"])
        self.assertFalse(mismatched_row["claim_eligible"])

    def test_constructor_rolls_back_partial_context_and_control_acquisition(
        self,
    ) -> None:
        """A constructor that never returns still releases every acquired owner."""
        for failed_stage in ("after-context", "after-control"):
            with self.subTest(failed_stage=failed_stage):
                releases = []
                context = ctypes.c_void_p(123)
                library = SimpleNamespace(
                    xtbloom_context_destroy=mock.Mock(
                        side_effect=lambda _context, captured=releases: captured.append(
                            "context"
                        )
                    )
                )
                memory = SimpleNamespace(
                    cuda=None,
                    close=mock.Mock(
                        side_effect=lambda captured=releases: captured.append("memory")
                    ),
                )
                control = SimpleNamespace(
                    close=mock.Mock(
                        side_effect=lambda captured=releases: captured.append("control")
                    )
                )
                cell = run.Cell("xtbloom", "cuda", "host", "test", "energy", 1)
                with (
                    mock.patch.object(
                        run.public_api, "_configure_library", return_value=library
                    ),
                    mock.patch.object(
                        run.public_api, "assemble_batch", return_value=SimpleNamespace()
                    ),
                    mock.patch.object(
                        run.public_api, "_make_context", return_value=context
                    ),
                    mock.patch.object(
                        run.public_api,
                        "DescriptorMemory",
                        return_value=memory,
                        side_effect=(
                            run.BenchmarkError("post-context failure")
                            if failed_stage == "after-context"
                            else None
                        ),
                    ),
                    mock.patch.object(
                        run.public_api, "CudaRuntime", return_value=control
                    ),
                    mock.patch.object(run, "configure_cuda_runtime"),
                    mock.patch.object(
                        run.XTBloomAdapter,
                        "_initialize_batch_and_results",
                        side_effect=run.BenchmarkError("post-control failure"),
                    ),
                    self.assertRaises(run.BenchmarkError),
                ):
                    run.XTBloomAdapter(
                        Path("fake.so"),
                        Path("manifest.json"),
                        {},
                        ({"id": "a"},),
                        cell,
                        0,
                        1,
                    )
                self.assertEqual(
                    releases,
                    ["context"]
                    if failed_stage == "after-context"
                    else ["control", "memory", "context"],
                )

    def test_unavailable_setup_keeps_not_run_coordinates_and_closes_peers(self) -> None:
        """Keep unavailable setup coordinates explicit without skipping other owners."""
        row, events, created = self._run_pair(
            property_name="energy",
            warmups=0,
            repetitions=1,
            behavior={
                "setup_failures": {
                    ("original", ("a", "b")): run.public_api.BackendUnavailable(
                        "fake backend unavailable"
                    )
                }
            },
        )
        arm = row["paired_rounds"][0]["arms"]["original"]
        self.assertEqual(
            [item["execution_state"] for item in arm["case_results"]],
            ["not_run", "not_run", "completed", "completed"],
        )
        self.assertEqual(row["availability"], "unavailable")
        self.assertEqual(len(created), 3)
        self.assertEqual(sum(event[0] == "close" for event in events), 3)
        self.assertEqual(
            row["correctness"]["coordinate_states_by_layout"]["original"]["not_run"],
            ["a", "b"],
        )

    def test_cli_defaults_and_paired_option_validation_remain_explicit(self) -> None:
        """Leave original as the default and reject incompatible pair matrices."""
        parser = run.build_parser()
        defaults = parser.parse_args(["--library", "lib.so", "--case-ids", "a"])
        self.assertEqual(defaults.ao_grouping, "original")
        defaults.engines = ("xtbloom",)
        defaults.backends = ("cpu",)
        defaults.cuda_memory_modes = ("host",)
        run.validate_args(defaults)
        exact = parser.parse_args(
            [
                "--library",
                "lib.so",
                "--case-ids",
                "a",
                "--ao-grouping",
                "exact-ao",
                "--engines",
                "xtbloom",
                "--backends",
                "cpu",
            ]
        )
        exact.cuda_memory_modes = ("host",)
        run.validate_args(exact)

        for options in (
            {"engines": ("xtbloom", "tblite")},
            {"backends": ("cpu", "cuda")},
            {"cuda_memory_modes": ("host", "device")},
        ):
            paired = parser.parse_args(
                [
                    "--library",
                    "lib.so",
                    "--case-ids",
                    "a",
                    "--ao-grouping",
                    "paired",
                    "--engines",
                    "xtbloom",
                    "--backends",
                    "cpu",
                    "--cuda-memory-modes",
                    "host",
                ]
            )
            for name, value in options.items():
                setattr(paired, name, value)
            with self.assertRaises(run.BenchmarkError):
                run.validate_args(paired)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run.conformance, "load_json", return_value={}),
                mock.patch.object(
                    run.public_api,
                    "model_tag",
                    return_value=run.public_api.XTBLOOM_MODEL_GFN2_XTB + 1,
                ),
                mock.patch.object(run.conformance, "selected_cases") as selected,
                mock.patch.object(run, "environment_metadata") as metadata,
                mock.patch.object(run, "finite_case_plan") as planning,
            ):
                status = run.main(
                    [
                        "--library",
                        "lib.so",
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
                        "--output-json",
                        str(root / "row.json"),
                        "--output-csv",
                        str(root / "row.csv"),
                    ]
                )
            self.assertEqual(status, 1)
            selected.assert_not_called()
            metadata.assert_not_called()
            planning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
