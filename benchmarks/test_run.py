"""Fast self-checks for benchmark matrix construction and serialization."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_xtbloom_raw_results_preserve_peer_statuses_and_optional_charges(
        self,
    ) -> None:
        """Expose dataset diagnostics without weakening strict matrix results."""
        adapter = object.__new__(run.XTBloomAdapter)
        adapter.systems = 2
        adapter.energies = (ctypes.c_double * 2)(-1.0, float("nan"))
        adapter.iterations = (ctypes.c_int32 * 2)(12, 500)
        adapter.converged = (ctypes.c_uint8 * 2)(1, 0)
        adapter.statuses = (ctypes.c_int32 * 2)(0, 7)
        adapter.forces = (ctypes.c_double * 6)(*range(6))
        adapter.charges = (ctypes.c_double * 2)(0.1, float("nan"))
        adapter.point_forces = None
        adapter.synchronize = mock.Mock()
        adapter.memory = SimpleNamespace(download_outputs=mock.Mock())

        raw = adapter.raw_results()

        self.assertEqual(raw["per_system_status"], [0, 7])
        self.assertEqual(raw["scc_converged"], [1, 0])
        self.assertEqual(raw["atomic_charges_e"][0], 0.1)
        with self.assertRaisesRegex(run.BenchmarkError, "system 1"):
            adapter.results()

    def test_xtbloom_from_storage_validates_before_loading_resources(self) -> None:
        """Reject invalid SCC limits before opening the library or a context."""
        storage = SimpleNamespace(slices=[SimpleNamespace()])
        cell = run.Cell("xtbloom", "cpu", "host", "dataset", "force", 1)
        with (
            mock.patch.object(run.public_api, "_configure_library") as configure,
            self.assertRaisesRegex(run.BenchmarkError, "max SCC iterations"),
        ):
            run.XTBloomAdapter.from_storage(
                Path("lib.so"),
                storage,
                cell,
                0,
                1,
                max_scc_iterations=0,
            )
        configure.assert_not_called()

        with (
            mock.patch.object(run.public_api, "_configure_library") as configure,
            self.assertRaisesRegex(run.BenchmarkError, "electronic temperature"),
        ):
            run.XTBloomAdapter.from_storage(
                Path("lib.so"),
                storage,
                cell,
                0,
                1,
                electronic_temperature_hartree=float("nan"),
            )
        configure.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
