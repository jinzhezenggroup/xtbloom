"""Offline integrity tests for the native periodic GFN2 oracle corpus."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/conformance/periodic_gfn2.py"
SPEC = importlib.util.spec_from_file_location("periodic_gfn2", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
periodic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(periodic)


class PeriodicGfn2OracleTests(unittest.TestCase):
    """Exercise canonical validation without needing a tblite executable."""

    def setUp(self) -> None:
        """Load the committed manifest used by every test."""
        self.manifest_path = periodic.DEFAULT_MANIFEST
        self.manifest = periodic.load_json(self.manifest_path)

    def _case_bundle(
        self, predicate: Callable[[dict[str, object]], bool]
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """Load one case, its golden, and the independently authored input."""
        case = next(item for item in self.manifest["cases"] if predicate(item))
        golden = periodic.load_json(periodic.repository_path(case["golden"]))
        structure = periodic.parse_turbomole(periodic.repository_path(case["input"]))
        return case, golden, structure

    def test_committed_corpus_passes(self) -> None:
        """Accept the complete committed periodic corpus."""
        periodic.check(self.manifest_path)

    def test_affine_deformation_matches_public_row_major_convention(self) -> None:
        """Apply strain using the public row-major affine convention."""
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
        """Transpose tblite's Fortran virial into the public matrix order."""
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
        """Reject a charged-background energy that no longer matches its equation."""
        manifest = copy.deepcopy(self.manifest)
        manifest["analytic_background_cases"][0]["energy_hartree"] += 1.0e-12
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "energy drifted"):
            periodic.check_analytic_background(manifest)

    def test_background_reconstruction_accepts_one_ulp_rounding(self) -> None:
        """Accept equivalent closed-form evaluations that differ by one ULP."""
        manifest = copy.deepcopy(self.manifest)
        case = manifest["analytic_background_cases"][0]
        for key in ("energy_hartree", "potential_hartree_per_e"):
            case[key] = math.nextafter(case[key], math.inf)
        for diagonal in (0, 4, 8):
            case["strain_derivatives_hartree"][diagonal] = math.nextafter(
                case["strain_derivatives_hartree"][diagonal], math.inf
            )
        periodic.check_analytic_background(manifest)

    def test_background_derivative_matches_volume_finite_difference(self) -> None:
        """Match the analytic background strain term to a volume derivative."""
        case = self.manifest["analytic_background_cases"][0]
        charge = case["charge_e"]
        alpha = case["alpha_bohr_inverse"]
        volume = case["volume_bohr3"]
        step = 1.0e-6

        def energy(strain: float) -> float:
            strained_volume = volume * (1.0 + strain)
            return -math.pi * charge * charge / (2.0 * alpha * alpha * strained_volume)

        numerical = (energy(step) - energy(-step)) / (2.0 * step)
        self.assertAlmostEqual(
            numerical, case["strain_derivatives_hartree"][0], delta=2.0e-12
        )

    def test_manifest_input_hash_is_enforced(self) -> None:
        """Reject corpus paths that escape the repository provenance boundary."""
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
            with self.assertRaisesRegex(
                periodic.PeriodicOracleError, "escapes repository"
            ):
                periodic.check(manifest_path)

    def test_process_launch_failure_is_actionable(self) -> None:
        """Convert a missing oracle executable into the CLI error contract."""
        with (
            mock.patch.object(
                periodic.subprocess,
                "run",
                side_effect=FileNotFoundError("missing executable"),
            ),
            self.assertRaisesRegex(
                periodic.PeriodicOracleError,
                "cannot launch.*/missing/tblite.*querying the tblite version",
            ),
        ):
            periodic.executable_version(Path("/missing/tblite"), {})

    def test_libtblite_discovery_requires_ldd(self) -> None:
        """Explain that live runtime discovery requires a Linux-like loader."""
        with (
            mock.patch.object(periodic.shutil, "which", return_value=None),
            self.assertRaisesRegex(periodic.PeriodicOracleError, "Linux-like.*ldd"),
        ):
            periodic.discover_libtblite(Path("/missing/tblite"), {}, "0" * 64)

    def test_strain_richardson_value_is_recomputed(self) -> None:
        """Reject stored Richardson values that are inconsistent with raw steps."""
        case, golden, structure = self._case_bundle(
            lambda item: item["strain_finite_difference"]
        )
        strain = golden["properties"]["strain_derivatives_hartree"]
        changed = copy.deepcopy(golden)
        changed["strain_finite_difference"]["modes"]["xx"]["richardson_hartree"] += (
            1.0e-10
        )
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "Richardson drifted"):
            periodic.check_strain_evidence(changed, strain, structure, case)

    def test_strain_central_difference_is_recomputed_from_energies(self) -> None:
        """Reject retained derivatives that no longer match their energy pairs."""
        case, golden, structure = self._case_bundle(
            lambda item: item["strain_finite_difference"]
        )
        changed = copy.deepcopy(golden)
        changed["strain_finite_difference"]["modes"]["xx"]["evaluations"][0]["plus"][
            "energy_hartree"
        ] += 1.0e-8
        with self.assertRaisesRegex(
            periodic.PeriodicOracleError, "central differences drifted"
        ):
            periodic.check_strain_evidence(
                changed,
                changed["properties"]["strain_derivatives_hartree"],
                structure,
                case,
            )

    def test_strain_materialized_input_hash_is_recomputed(self) -> None:
        """Bind every strain energy to its deterministic deformed input."""
        case, golden, structure = self._case_bundle(
            lambda item: item["strain_finite_difference"]
        )
        changed = copy.deepcopy(golden)
        changed["strain_finite_difference"]["modes"]["xx"]["evaluations"][0]["minus"][
            "input_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "input SHA-256"):
            periodic.check_strain_evidence(
                changed,
                changed["properties"]["strain_derivatives_hartree"],
                structure,
                case,
            )

    def test_compare_rejects_changed_strain_run_hash(self) -> None:
        """Require live strain runs to retain the reviewed raw-output identity."""
        case, golden, _structure = self._case_bundle(
            lambda item: item["strain_finite_difference"]
        )
        changed = copy.deepcopy(golden)
        changed["strain_finite_difference"]["modes"]["xx"]["evaluations"][0]["minus"][
            "raw_output_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            periodic.dump_json(Path(directory) / f"{case['id']}.json", changed)
            with self.assertRaisesRegex(
                periodic.PeriodicOracleError, "raw_output_sha256 mismatch"
            ):
                periodic.compare(self.manifest_path, Path(directory), [case["id"]])

    def test_compare_rejects_missing_strain_evidence(self) -> None:
        """Reject a live result that omits a required finite-difference ledger."""
        case, golden, _structure = self._case_bundle(
            lambda item: item["strain_finite_difference"]
        )
        changed = copy.deepcopy(golden)
        del changed["strain_finite_difference"]
        with tempfile.TemporaryDirectory() as directory:
            periodic.dump_json(Path(directory) / f"{case['id']}.json", changed)
            with self.assertRaisesRegex(
                periodic.PeriodicOracleError, "lacks canonical strain steps"
            ):
                periodic.compare(self.manifest_path, Path(directory), [case["id"]])

    def test_cartesian_energy_and_richardson_values_are_recomputed(self) -> None:
        """Reject Cartesian evidence detached from its energies or extrapolation."""
        case, golden, structure = self._case_bundle(
            lambda item: item.get("cartesian_finite_difference", False)
        )
        forces = golden["properties"]["forces_hartree_per_bohr"]
        tolerance = case["tolerances"]["cartesian_finite_difference"]
        changed_energy = copy.deepcopy(golden)
        changed_energy["cartesian_finite_difference"]["coordinates"][0]["evaluations"][
            0
        ]["plus"]["energy_hartree"] += 1.0e-8
        with self.assertRaisesRegex(
            periodic.PeriodicOracleError, "central forces drifted"
        ):
            periodic.check_cartesian_evidence(
                changed_energy, forces, structure, tolerance
            )
        changed_richardson = copy.deepcopy(golden)
        changed_richardson["cartesian_finite_difference"]["coordinates"][0][
            "richardson_force_hartree_per_bohr"
        ] += 1.0e-8
        with self.assertRaisesRegex(
            periodic.PeriodicOracleError, "Richardson force drifted"
        ):
            periodic.check_cartesian_evidence(
                changed_richardson, forces, structure, tolerance
            )

    def test_full_model_net_force_and_strain_symmetry_are_gated(self) -> None:
        """Reject broken translation and rotational invariants in base results."""
        case, golden, structure = self._case_bundle(
            lambda item: (
                item.get("invariant_variants", False) and int(item["atom_count"]) > 1
            )
        )
        changed_force = copy.deepcopy(golden)
        changed_force["properties"]["forces_hartree_per_bohr"][0] += 1.0e-6
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "net force"):
            periodic.check_invariant_evidence(
                changed_force, case, structure, changed_force["properties"]
            )
        changed_strain = copy.deepcopy(golden)
        changed_strain["properties"]["strain_derivatives_hartree"][1] += 1.0e-6
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "not symmetric"):
            periodic.check_invariant_evidence(
                changed_strain, case, structure, changed_strain["properties"]
            )

    def test_invariant_input_and_property_drift_are_rejected(self) -> None:
        """Bind symmetry variants to exact inputs and complete properties."""
        case, golden, structure = self._case_bundle(
            lambda item: item.get("invariant_variants", False)
        )
        changed_input = copy.deepcopy(golden)
        changed_input["invariant_evidence"]["variants"]["translation"][
            "input_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "input hash drifted"):
            periodic.check_invariant_evidence(
                changed_input, case, structure, changed_input["properties"]
            )
        changed_property = copy.deepcopy(golden)
        variant = changed_property["invariant_evidence"]["variants"]["translation"]
        variant["properties"]["energy_hartree"] += 1.0e-6
        variant["source_output_sha256"] = periodic.sha256_json(variant["properties"])
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "invariant error"):
            periodic.check_invariant_evidence(
                changed_property, case, structure, changed_property["properties"]
            )
        changed_wrapping = copy.deepcopy(golden)
        changed_wrapping["invariant_evidence"]["wrapping"][
            "wrapped_positions_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "hash drifted"):
            periodic.check_invariant_evidence(
                changed_wrapping, case, structure, changed_wrapping["properties"]
            )

    def test_build_attestation_and_runtime_closure_corruption_are_rejected(
        self,
    ) -> None:
        """Bind every live result to the exact source build and loader closure."""
        reference = copy.deepcopy(self.manifest["reference_engine"])
        changed_reference = copy.deepcopy(reference)
        changed_reference["build_attestation"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "SHA-256 mismatch"):
            periodic.load_build_attestation(changed_reference)

        attestation = periodic.load_json(
            periodic.repository_path(reference["build_attestation"]["path"])
        )
        attestation["runtime"]["non_system_libraries"][0]["sha256"] = "0" * 64
        with (
            mock.patch.object(periodic, "load_json", return_value=attestation),
            self.assertRaisesRegex(
                periodic.PeriodicOracleError, "runtime closure digest drifted"
            ),
        ):
            periodic.load_build_attestation(reference, verify_file_hash=False)

    def test_ewald_alpha_and_cutoff_contract_drift_is_rejected(self) -> None:
        """Recompute the pinned orthogonal and skew alpha/cutoff decisions."""
        changed = copy.deepcopy(self.manifest)
        changed["ewald_numerics_contract"]["cases"][0]["expected"]["monopole"][
            "alpha_bohr_inverse"
        ] += 0.03125
        with self.assertRaisesRegex(periodic.PeriodicOracleError, "alpha/cutoff"):
            periodic.check_ewald_numerics_contract(changed)

    def test_ewald_background_omission_and_component_drift_are_rejected(self) -> None:
        """Keep the charged reconstruction complete and component resolved."""
        identity = self.manifest["ewald_reconstruction"]
        reconstruction = periodic.load_json(periodic.repository_path(identity["path"]))
        for field, replacement in (
            ("background_energy_hartree", 0.0),
            ("reciprocal_energy_hartree", 1.0),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(reconstruction)
                changed["alphas"][0]["base"][field] = replacement
                with (
                    mock.patch.object(
                        periodic, "sha256_file", return_value=identity["sha256"]
                    ),
                    mock.patch.object(periodic, "load_json", return_value=changed),
                    self.assertRaisesRegex(
                        periodic.PeriodicOracleError, "numerical content drifted"
                    ),
                ):
                    periodic.check_ewald_reconstruction(self.manifest)


if __name__ == "__main__":
    unittest.main()
