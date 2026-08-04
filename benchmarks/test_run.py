"""Fast self-checks for benchmark matrix construction and serialization."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks import run
from benchmarks.tblite_adapter import TbliteAdapter, TbliteState
from benchmarks.xtb_adapter import XtbAdapter, XtbState


class HarnessTest(unittest.TestCase):
    """Exercise non-hardware protocol logic without loading gpuxtb or CUDA."""

    def test_timing_summary_retains_samples_and_batch_throughput(self) -> None:
        summary = run.timing_summary([3.0, 1.0, 2.0], batch_size=8)
        self.assertEqual(summary["samples_ms"], [3.0, 1.0, 2.0])
        self.assertEqual(summary["median_ms"], 2.0)
        self.assertEqual(summary["systems_per_second_at_median"], 4000.0)

    def test_gpuxtb_cell_matrix_contains_cpu_and_three_cuda_placements(self) -> None:
        args = SimpleNamespace(
            backends=("cpu", "cuda"),
            cuda_memory_modes=("host", "device", "mixed"),
            workloads=("gas", "qmmm"),
            properties=("energy", "force"),
            batch_sizes=(1, 8, 32, 128),
        )
        cells = list(run.gpuxtb_cells(args))
        self.assertEqual(len(cells), 64)
        placements = {(cell.backend, cell.memory_mode) for cell in cells}
        self.assertEqual(
            placements,
            {("cpu", "host"), ("cuda", "host"), ("cuda", "device"), ("cuda", "mixed")},
        )

    def test_xtb_matrix_is_serial_cpu_host_and_templates_are_strict(self) -> None:
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

    def test_correctness_includes_qm_and_point_charge_forces(self) -> None:
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
        cell = run.Cell("gpuxtb", "cpu", "host", "qmmm", "force", 1)
        result = run.correctness(cell, storage, output, manifest)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["max_abs_point_charge_force_error_hartree_per_bohr"], 0.0
        )

    def test_json_and_csv_preserve_unavailable_rows(self) -> None:
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
