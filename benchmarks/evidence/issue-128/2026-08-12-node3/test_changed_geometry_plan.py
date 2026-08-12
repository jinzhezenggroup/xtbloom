"""Hardware-independent tests for the issue #128 changed-geometry reproducer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).with_name("changed-geometry-plan.py")
SPEC = importlib.util.spec_from_file_location("issue128_changed_geometry", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ChangedGeometryPlanTest(unittest.TestCase):
    """Exercise clean-source, geometry, correctness, and exit-status helpers."""

    def test_dirty_source_is_rejected(self) -> None:
        """A dirty measured worktree must never produce final evidence."""
        with (
            mock.patch.object(
                MODULE, "git_text", side_effect=["deadbeef", " M tracked.txt"]
            ),
            self.assertRaisesRegex(RuntimeError, "must be clean"),
        ):
            MODULE.require_clean_source(Path("/source"))

    def test_changed_geometry_sequence_is_distinct(self) -> None:
        """Twenty sampling steps change values without changing extents."""
        original = [float(index) for index in range(15)]
        geometries = [
            MODULE.changed_positions(original, 5, step) for step in range(3, 23)
        ]
        self.assertEqual(len({tuple(values) for values in geometries}), 20)
        for values in geometries:
            self.assertEqual(len(values), len(original))
            changed = sum(a != b for a, b in zip(values, original, strict=True))
            self.assertEqual(changed, 1)

    def test_sample_mismatch_is_reported_and_returns_failure(self) -> None:
        """A numerical mismatch remains visible and makes automation fail."""
        cuda_output = {
            "energies_hartree": [0.0],
            "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
        }
        cpu_output = SimpleNamespace(energies=[1.0e-3], forces=[0.0, 0.0, 0.0])
        comparison = MODULE.compare_sample(cuda_output, cpu_output, 1)
        sample = {
            "correctness_status": comparison["status"],
            "max_abs_energy_error_hartree": comparison["max_abs_energy_error_hartree"],
            "max_abs_force_error_hartree_per_bohr": comparison[
                "max_abs_force_error_hartree_per_bohr"
            ],
        }
        correctness = MODULE.summarize_correctness([sample])
        document = {"correctness": correctness}
        self.assertEqual(correctness["status"], "fail")
        self.assertEqual(MODULE.result_exit_status(document), 1)

    def test_successful_samples_return_zero(self) -> None:
        """A completely qualified sample set remains a successful command."""
        sample = {
            "correctness_status": "pass",
            "max_abs_energy_error_hartree": 0.0,
            "max_abs_force_error_hartree_per_bohr": 0.0,
        }
        document = {"correctness": MODULE.summarize_correctness([sample])}
        self.assertEqual(MODULE.result_exit_status(document), 0)


if __name__ == "__main__":
    unittest.main()
