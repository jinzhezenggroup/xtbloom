"""Offline integrity tests for the native periodic GFN2 oracle corpus."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/conformance/periodic_gfn2.py"
SPEC = importlib.util.spec_from_file_location("periodic_gfn2", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
periodic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(periodic)


class PeriodicGfn2OracleTests(unittest.TestCase):
    """Exercise canonical validation without needing a tblite executable."""

    def setUp(self) -> None:
        self.manifest_path = periodic.DEFAULT_MANIFEST
        self.manifest = periodic.load_json(self.manifest_path)

    def test_committed_corpus_passes(self) -> None:
        periodic.check(self.manifest_path)

    def test_affine_deformation_matches_public_row_major_convention(self) -> None:
        structure = {
            "positions_bohr": [2.0, 3.0, 5.0],
            "symbols": ["He"],
            "cell_matrix_row_major_bohr": [
                7.0,
                11.0,
                13.0,
                17.0,
                19.0,
                23.0,
                29.0,
                31.0,
                37.0,
            ],
        }
        positions, cell = periodic.affine_deformation(structure, 0, 1, 0.25)
        self.assertEqual(positions, [2.75, 3.0, 5.0])
        # H' = H (I+epsilon)^T: every lattice row's x component gains 0.25*y.
        self.assertEqual(
            cell,
            [9.75, 11.0, 13.0, 21.75, 19.0, 23.0, 36.75, 31.0, 37.0],
        )

    def test_tblite_column_major_virial_is_normalized_to_row_major(self) -> None:
        raw = {
            "energy": -1.0,
            "gradient": [1.0, 2.0, 3.0],
            "virial": [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
        }
        normalized = periodic.normalize_tblite(raw, 1)
        self.assertEqual(
            normalized["strain_derivatives_hartree"],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        self.assertEqual(normalized["forces_hartree_per_bohr"], [-1.0, -2.0, -3.0])

    def test_background_reconstruction_rejects_energy_drift(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["analytic_background_cases"][0]["energy_hartree"] += 1.0e-12
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "energy drifted"):
            periodic.check_analytic_background(manifest)

    def test_background_derivative_matches_volume_finite_difference(self) -> None:
        case = self.manifest["analytic_background_cases"][0]
        charge = case["charge_e"]
        alpha = case["alpha_bohr_inverse"]
        volume = case["volume_bohr3"]
        step = 1.0e-6

        def energy(strain: float) -> float:
            strained_volume = volume * (1.0 + strain)
            return -math.pi * charge * charge / (
                2.0 * alpha * alpha * strained_volume
            )

        numerical = (energy(step) - energy(-step)) / (2.0 * step)
        self.assertAlmostEqual(
            numerical, case["strain_derivatives_hartree"][0], delta=2.0e-12
        )

    def test_manifest_input_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = copy.deepcopy(self.manifest)
            source = periodic.repository_path(manifest["cases"][0]["input"])
            copied = root / "input.tmol"
            copied.write_bytes(source.read_bytes() + b"\n")
            manifest["cases"][0]["input"] = str(copied)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            # Absolute paths outside the repository are rejected before a
            # misleading digest can assert corpus identity.
            with self.assertRaisesRegex(periodic.PeriodicOracleError, "escapes repository"):
                periodic.check(manifest_path)

    def test_strain_richardson_value_is_recomputed(self) -> None:
        case = next(
            item for item in self.manifest["cases"] if item["strain_finite_difference"]
        )
        golden = periodic.load_json(periodic.repository_path(case["golden"]))
        strain = golden["properties"]["strain_derivatives_hartree"]
        changed = copy.deepcopy(golden)
        changed["strain_finite_difference"]["modes"]["xx"][
            "richardson_hartree"
        ] += 1.0e-10
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "Richardson drifted"):
            periodic.check_strain_evidence(changed, strain)


if __name__ == "__main__":
    unittest.main()
