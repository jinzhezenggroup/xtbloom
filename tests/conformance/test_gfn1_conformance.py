"""Focused standard-library tests for the independent GFN1 oracle corpus."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools/conformance/gfn1_conformance.py"
MANIFEST = ROOT / "data/conformance/gfn1/manifest.json"
SPEC = importlib.util.spec_from_file_location("gfn1_conformance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class Gfn1ConformanceTest(unittest.TestCase):
    """Protect oracle independence, model identity, hashes, and force signs."""

    def run_tool(
        self, *arguments: str, status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        """Run the standalone CLI and retain output in assertion diagnostics."""
        completed = subprocess.run(
            [sys.executable, str(TOOL_PATH), "--manifest", str(MANIFEST), *arguments],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            status,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return completed

    def test_committed_manifest_and_eight_cases_are_valid(self) -> None:
        """The offline check verifies all reviewed inputs, goldens, and provenance."""
        completed = self.run_tool("check")
        self.assertIn("8 cases", completed.stdout)

    def test_model_and_oracle_routing_are_explicitly_gfn1(self) -> None:
        """No GFN2 command or runtime support claim can enter this corpus silently."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["method"], "GFN1-xTB")
        tblite_command = manifest["reference_engines"]["tblite"]["cli_command_template"]
        self.assertEqual(tblite_command[tblite_command.index("--method") + 1], "gfn1")
        for key in ("cli_command_template", "qmmm_cli_command_template"):
            command = manifest["reference_engines"]["xtb"][key]
            self.assertEqual(command[command.index("--gfn") + 1], "1")
        self.assertNotIn("xtbloom_backends", json.dumps(manifest))
        for case in manifest["cases"][:4]:
            self.assertRegex(case["upstream_input_git_blob"], r"^[0-9a-f]{40}$")
            self.assertEqual(case["upstream_input_sha256"], case["input_sha256"])

    def test_specialized_cases_pin_spin_screening_and_halogen_fixtures(self) -> None:
        """The xTB subset covers the three GFN1-only scientific risk areas."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        self.assertEqual(cases["gfn1_oh_radical"]["unpaired_electrons"], 1)
        qmmm = TOOL.load_qmmm(
            ROOT / cases["gfn1_water_dimer_6pc_hardness"]["input"],
            cases["gfn1_water_dimer_6pc_hardness"],
            manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"],
        )
        self.assertEqual(
            qmmm["external_point_charges"]["gammas_hartree"],
            [0.583349, 0.470099, 0.470099, 0.583349, 0.470099, 0.470099],
        )
        symbols, _ = TOOL.load_coord(
            ROOT / cases["gfn1_halogen_bond"]["input"],
            cases["gfn1_halogen_bond"]["atom_count"],
        )
        self.assertEqual(symbols, ["Br", "Br", "O", "C", "H", "H"])

    def test_xtb_goldens_match_independent_unit_fixture_values(self) -> None:
        """Live strict-oracle results stay near the pinned xTB unit-test constants."""
        expected = {
            "gfn1_halogen_bond": -15.606233877972,
            "gfn1_water_dimer_6pc_hardness": -11.559896105984,
            "gfn1_water_dimer_6pc_gamma999": -11.565012263827,
        }
        for case_id, energy in expected.items():
            with self.subTest(case_id=case_id):
                golden = json.loads(
                    (ROOT / f"data/conformance/gfn1/golden/{case_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertAlmostEqual(
                    golden["properties"]["energy_hartree"], energy, delta=1.0e-9
                )

    def test_compare_rejects_missing_point_charge_forces(self) -> None:
        """A QM-only result cannot satisfy the GFN1 PCEM oracle."""
        case_id = "gfn1_water_dimer_6pc_hardness"
        source = ROOT / f"data/conformance/gfn1/golden/{case_id}.json"
        with tempfile.TemporaryDirectory() as temporary:
            actual = json.loads(source.read_text(encoding="utf-8"))
            del actual["properties"]["point_charge_forces_hartree_per_bohr"]
            actual["provenance"]["source_output_sha256"] = TOOL.sha256_json(
                actual["properties"]
            )
            Path(temporary, f"{case_id}.json").write_text(
                json.dumps(actual), encoding="utf-8"
            )
            completed = self.run_tool(
                "compare", "--actual-dir", temporary, "--case", case_id, status=1
            )
            self.assertIn("missing point_charge_forces", completed.stderr)

    def test_compare_rejects_mismatched_oracle_identity_before_numbers(self) -> None:
        """Numerically identical output cannot be credited to another oracle."""
        case_id = "gfn1_h3_plus"
        source = ROOT / f"data/conformance/gfn1/golden/{case_id}.json"
        mutations = (
            (("case_id",), "gfn1_ketene", "case_id"),
            (("method",), "GFN2-xTB", "method"),
            (("provenance", "engine"), "xtb", "reference engine"),
            (("provenance", "source_revision"), "0" * 40, "source revision"),
            (("provenance", "input"), "unrelated.coord", "provenance input"),
            (
                ("provenance", "executable_sha256"),
                "0" * 64,
                "provenance identity",
            ),
            (
                ("provenance", "source_output_sha256"),
                "0" * 64,
                "does not bind the actual properties",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            actual_path = Path(temporary, f"{case_id}.json")
            for keys, replacement, message in mutations:
                with self.subTest(field=".".join(keys)):
                    actual = json.loads(source.read_text(encoding="utf-8"))
                    target = actual
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    actual_path.write_text(json.dumps(actual), encoding="utf-8")
                    completed = self.run_tool(
                        "compare",
                        "--actual-dir",
                        temporary,
                        "--case",
                        case_id,
                        status=1,
                    )
                    self.assertIn("identity mismatch", completed.stderr)
                    self.assertIn(message, completed.stderr)

    def test_qmmm_rejects_invalid_units_elements_shapes_and_numbers(self) -> None:
        """PCEM materialization accepts only one finite, typed physical schema."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        case = cases["gfn1_water_dimer_6pc_hardness"]
        hardness = manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"]
        source = json.loads((ROOT / case["input"]).read_text(encoding="utf-8"))
        mutations = (
            (("units", "qm_positions"), "angstrom", "inconsistent units"),
            (("qm", "symbols", 0), "H", "symbols do not match"),
            (("qm", "atomic_numbers", 0), 0, "unsupported atomic number"),
            (("qm", "positions_bohr", 0), [0.0, 0.0], "positions_bohr shape"),
            (("qm", "positions_bohr", 0, 0), "0.0", "invalid numeric value"),
            (("qm", "positions_bohr", 0, 0), math.inf, "invalid numeric value"),
            (
                ("external_point_charges", "positions_bohr", 0),
                [0.0, 0.0],
                "positions_bohr shape",
            ),
            (
                ("external_point_charges", "positions_bohr", 0, 0),
                True,
                "invalid numeric value",
            ),
            (
                ("external_point_charges", "positions_bohr", 0, 0),
                math.nan,
                "invalid numeric value",
            ),
            (
                ("external_point_charges", "charges_e"),
                [0.0],
                "charges_e shape",
            ),
            (
                ("external_point_charges", "charges_e", 0),
                "-0.5",
                "invalid numeric value",
            ),
            (
                ("external_point_charges", "charges_e", 0),
                math.inf,
                "invalid numeric value",
            ),
            (
                ("external_point_charges", "gammas_hartree"),
                [1.0],
                "gammas_hartree shape",
            ),
            (
                ("external_point_charges", "gammas_hartree", 0),
                "0.583349",
                "invalid numeric value",
            ),
            (
                ("external_point_charges", "gammas_hartree", 0),
                math.nan,
                "invalid numeric value",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (keys, replacement, message) in enumerate(mutations):
                with self.subTest(field=".".join(str(key) for key in keys)):
                    invalid = json.loads(json.dumps(source))
                    target = invalid
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                    path = Path(temporary, f"invalid-{index}.json")
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(TOOL.ConformanceError, message):
                        TOOL.load_qmmm(path, case, hardness)

    def test_hash_mutation_is_rejected(self) -> None:
        """Changing a canonical input without updating live evidence fails offline."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "manifest.json"
            manifest["cases"][0]["input_sha256"] = "0" * 64
            invalid.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(TOOL.ConformanceError, "input hash mismatch"):
                TOOL.check_manifest(invalid)

    def test_upstream_git_blob_must_hash_retained_input_bytes(self) -> None:
        """A plausible but unrelated SHA-1 cannot assert copied-source identity."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "manifest.json"
            manifest["cases"][0]["upstream_input_git_blob"] = "0" * 40
            invalid.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                TOOL.ConformanceError, "upstream input Git blob mismatch"
            ):
                TOOL.check_manifest(invalid)


if __name__ == "__main__":
    unittest.main()
