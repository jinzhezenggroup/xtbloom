"""Self-contained tests for the reference corpus tooling.

These tests use only Python's standard library so they can run before the
gpuxtb physics implementation or either Fortran reference package is built.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "gpuxtb_conformance.py"
MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
SPEC = importlib.util.spec_from_file_location("gpuxtb_conformance_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)


class ConformanceToolTest(unittest.TestCase):
    """Exercise integrity checks, supported result schemas, and live generation."""

    def run_tool(
        self, *arguments: str, expected_status: int = 0
    ) -> subprocess.CompletedProcess[str]:
        """Run the public CLI and include captured output in assertion failures."""
        completed = subprocess.run(
            [sys.executable, str(TOOL), "--manifest", str(MANIFEST), *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            completed.returncode,
            expected_status,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return completed

    def test_manifest_and_committed_corpus_are_internally_consistent(self) -> None:
        """Hashes, units, array shapes, and the force sign convention are checked."""
        completed = self.run_tool("check")
        self.assertIn("8 cases", completed.stdout)

    def test_qmmm_goldens_embed_inputs_and_both_force_domains(self) -> None:
        """QM/MM cases retain exact PC inputs and use force=-gradient twice."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        qmmm_cases = [
            case for case in manifest["cases"] if case.get("input_schema") == "qmmm-v1"
        ]
        self.assertEqual(len(qmmm_cases), 3)
        for case in qmmm_cases:
            qmmm_input = json.loads(
                (REPOSITORY_ROOT / case["input"]).read_text(encoding="utf-8")
            )
            golden = json.loads(
                (REPOSITORY_ROOT / case["golden"]).read_text(encoding="utf-8")
            )
            self.assertEqual(golden["qmmm_input"], qmmm_input)
            self.assertEqual(
                golden["provenance"]["command_template"],
                manifest["reference_engines"]["xtb"]["qmmm_cli_command_template"],
            )
            self.assertEqual(
                golden["provenance"]["command"],
                CONFORMANCE.xtb_command(Path("{executable}"), case),
            )
            self.assertEqual(
                golden["provenance"]["materialized_input"]["schema"],
                "gpuxtb-xtb-pcem-cli-v1",
            )
            self.assertEqual(
                set(golden["provenance"]["materialized_input"]["files_sha256"]),
                {"coord", "pcharge", "xcontrol"},
            )
            self.assertEqual(
                golden["provenance"]["runtime"]["libxtb"]["sha256"],
                manifest["reference_engines"]["xtb"]["runtime_artifacts"][
                    "libxtb_sha256"
                ],
            )
            properties = golden["properties"]
            self.assertEqual(
                properties["forces_hartree_per_bohr"],
                [-value for value in properties["gradient_hartree_per_bohr"]],
            )
            self.assertEqual(
                properties["point_charge_forces_hartree_per_bohr"],
                [
                    -value
                    for value in properties["point_charge_gradient_hartree_per_bohr"]
                ],
            )

        by_id = {case["id"]: case for case in qmmm_cases}
        for case_id in (
            "water_one_pc_gamma999",
            "water_dimer_6pc_gamma999",
        ):
            qmmm_input = json.loads(
                (REPOSITORY_ROOT / by_id[case_id]["input"]).read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    gamma == 999.0
                    for gamma in qmmm_input["external_point_charges"]["gammas_hartree"]
                )
            )
            self.assertEqual(
                qmmm_input["external_point_charges"]["gamma_mode"], "explicit"
            )
            self.assertNotIn(
                "source_atomic_numbers", qmmm_input["external_point_charges"]
            )

        hardness_input = json.loads(
            (REPOSITORY_ROOT / by_id["water_dimer_6pc_hardness"]["input"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            hardness_input["external_point_charges"]["gamma_mode"],
            "element_hardness",
        )

        # Independent constants copied from xTB's own GFN2 PCEM unit test.
        hardness = json.loads(
            (REPOSITORY_ROOT / by_id["water_dimer_6pc_hardness"]["golden"]).read_text(
                encoding="utf-8"
            )
        )["properties"]
        self.assertAlmostEqual(
            hardness["energy_hartree"], -10.160927754235, delta=1e-10
        )
        self.assertAlmostEqual(
            hardness["gradient_hartree_per_bohr"][12],
            -2.0326750053811e-3,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            hardness["point_charge_gradient_hartree_per_bohr"][12],
            -2.4831958012524e-4,
            delta=5e-9,
        )
        gamma999 = json.loads(
            (REPOSITORY_ROOT / by_id["water_dimer_6pc_gamma999"]["golden"]).read_text(
                encoding="utf-8"
            )
        )["properties"]
        self.assertAlmostEqual(
            gamma999["energy_hartree"], -10.168788268962, delta=1e-10
        )

    def test_qmmm_input_rejects_element_and_gamma_semantic_mismatches(self) -> None:
        """The oracle cannot calculate a different element/gamma model than declared."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        hardness = manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"]
        source_path = REPOSITORY_ROOT / cases["water_dimer_6pc_hardness"]["input"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        invalid_documents = []

        wrong_symbol = json.loads(json.dumps(source))
        wrong_symbol["qm"]["symbols"][0] = "H"
        invalid_documents.append((wrong_symbol, "symbols do not match"))

        wrong_gamma = json.loads(json.dumps(source))
        wrong_gamma["external_point_charges"]["gammas_hartree"][0] += 0.01
        invalid_documents.append((wrong_gamma, "gammas do not match"))

        wrong_source_element = json.loads(json.dumps(source))
        wrong_source_element["external_point_charges"]["source_atomic_numbers"][0] = 1
        invalid_documents.append((wrong_source_element, "gammas do not match"))

        explicit_with_source = json.loads(json.dumps(source))
        explicit_with_source["external_point_charges"]["gamma_mode"] = "explicit"
        invalid_documents.append((explicit_with_source, "must not provide source"))

        with tempfile.TemporaryDirectory() as temporary:
            for index, (document, message) in enumerate(invalid_documents):
                with self.subTest(message=message):
                    path = Path(temporary) / f"invalid-{index}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(CONFORMANCE.ConformanceError, message):
                        CONFORMANCE.load_qmmm_input(
                            path,
                            cases["water_dimer_6pc_hardness"],
                            hardness,
                        )

    def test_compare_accepts_canonical_golden_files(self) -> None:
        """A generated canonical directory can be compared without translation."""
        golden = REPOSITORY_ROOT / "data" / "conformance" / "golden"
        self.run_tool("compare", "--actual-dir", str(golden))

    def test_compare_accepts_raw_tblite_energy_and_gradient(self) -> None:
        """tblite gradients are negated exactly once before force comparison."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            actual_dir = Path(temporary)
            tblite_cases = [
                case
                for case in manifest["cases"]
                if case.get("reference_engine", "tblite") == "tblite"
            ]
            for case in tblite_cases:
                golden = json.loads(
                    (REPOSITORY_ROOT / case["golden"]).read_text(encoding="utf-8")
                )
                raw = {
                    "energy": golden["properties"]["energy_hartree"],
                    "gradient": golden["properties"]["gradient_hartree_per_bohr"],
                }
                (actual_dir / f"{case['id']}.json").write_text(
                    json.dumps(raw), encoding="utf-8"
                )
            selected: list[str] = []
            for case in tblite_cases:
                selected.extend(["--case", case["id"]])
            self.run_tool("compare", "--actual-dir", str(actual_dir), *selected)

    def test_compare_rejects_an_energy_outside_tolerance(self) -> None:
        """A regression larger than the manifest threshold produces a nonzero exit."""
        source = REPOSITORY_ROOT / "data" / "conformance" / "golden" / "h3_plus.json"
        with tempfile.TemporaryDirectory() as temporary:
            actual_dir = Path(temporary)
            actual = json.loads(source.read_text(encoding="utf-8"))
            actual["properties"]["energy_hartree"] += 1.0e-4
            (actual_dir / "h3_plus.json").write_text(
                json.dumps(actual), encoding="utf-8"
            )
            completed = self.run_tool(
                "compare",
                "--actual-dir",
                str(actual_dir),
                "--case",
                "h3_plus",
                expected_status=1,
            )
            self.assertIn("FAIL", completed.stdout)

    def test_generate_runs_tblite_command_and_normalizes_gradient(self) -> None:
        """A tiny fake executable verifies argv construction without a Fortran toolchain."""
        expected = json.loads(
            (
                REPOSITORY_ROOT / "data" / "conformance" / "golden" / "h3_plus.json"
            ).read_text(encoding="utf-8")
        )["properties"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake-tblite"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('tblite fake-reference 1')\n"
                "    raise SystemExit(0)\n"
                "assert sys.argv[sys.argv.index('--method') + 1] == 'gfn2'\n"
                "assert sys.argv[sys.argv.index('--charge') + 1] == '+1'\n"
                "assert '--grad' in sys.argv and '--no-restart' in sys.argv\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--json') + 1])\n"
                f"raw = {{'energy': {expected['energy_hartree']!r}, "
                f"'gradient': {expected['gradient_hartree_per_bohr']!r}, "
                f"'virial': {expected['virial_hartree']!r}}}\n"
                "out.write_text(json.dumps(raw), encoding='utf-8')\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            output_dir = root / "generated"
            self.run_tool(
                "generate",
                "--executable",
                str(executable),
                "--output-dir",
                str(output_dir),
                "--case",
                "h3_plus",
            )
            generated = json.loads(
                (output_dir / "h3_plus.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generated["properties"], expected)
            self.assertEqual(generated["provenance"]["generation_mode"], "live-cli")
            self.run_tool(
                "compare",
                "--actual-dir",
                str(output_dir),
                "--case",
                "h3_plus",
            )

    def test_generate_xtb_normalizes_gradient_and_scc_multipoles(self) -> None:
        """The xtb adapter joins its gradient artifact with atom-resolved JSON."""
        expected = json.loads(
            (
                REPOSITORY_ROOT / "data" / "conformance" / "golden" / "oh_radical.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake-xtb"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('xtb version 6.7.1 (edcfbbe)')\n"
                "    raise SystemExit(0)\n"
                "assert sys.argv[1] == 'coord'\n"
                "assert sys.argv[sys.argv.index('--gfn') + 1] == '2'\n"
                "assert sys.argv[sys.argv.index('--chrg') + 1] == '0'\n"
                "assert sys.argv[sys.argv.index('--uhf') + 1] == '1'\n"
                "assert sys.argv[sys.argv.index('-P') + 1] == '1'\n"
                "assert '--grad' in sys.argv and '--json' in sys.argv\n"
                "assert pathlib.Path('coord').read_text().startswith('$coord')\n"
                f"properties = {expected['properties']!r}\n"
                "raw = {\n"
                "  'partial charges': properties['partial_charges_e'],\n"
                "  'atomic dipole moments': properties['atomic_dipoles_e_bohr'],\n"
                "  'atomic quadrupole moments': properties['atomic_quadrupoles_e_bohr2'],\n"
                "  'number of unpaired electrons': 1,\n"
                "}\n"
                "pathlib.Path('xtbout.json').write_text(json.dumps(raw))\n"
                "gradient = properties['gradient_hartree_per_bohr']\n"
                "rows = [' '.join(map(str, gradient[i:i+3])) for i in range(0, 6, 3)]\n"
                "pathlib.Path('gradient').write_text(\n"
                "  '$grad\\n  cycle = 1 SCF energy = -4.42833345932 |dE/dxyz| = 0.0\\n'\n"
                "  ' 0 0 0 o\\n 0 0 1.834 h\\n' + '\\n'.join(rows) + '\\n$end\\n'\n"
                ")\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            output_dir = root / "generated"
            self.run_tool(
                "generate-xtb",
                "--executable",
                str(executable),
                "--output-dir",
                str(output_dir),
                "--case",
                "oh_radical",
            )
            generated = json.loads(
                (output_dir / "oh_radical.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generated["properties"], expected["properties"])
            self.assertEqual(generated["provenance"]["engine"], "xtb")
            self.assertEqual(
                generated["provenance"]["source_output_sha256"],
                expected["provenance"]["source_output_sha256"],
            )
            self.run_tool(
                "compare",
                "--actual-dir",
                str(output_dir),
                "--case",
                "oh_radical",
            )

    def test_generate_xtb_materializes_qmmm_input_and_pcgradient(self) -> None:
        """The xTB adapter creates explicit gamma input and negates pcgrad."""
        expected = json.loads(
            (
                REPOSITORY_ROOT
                / "data"
                / "conformance"
                / "golden"
                / "water_one_pc_gamma999.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "fake-xtb"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "if '--version' in sys.argv:\n"
                "    print('xtb version 6.7.1 (edcfbbe)')\n"
                "    raise SystemExit(0)\n"
                "assert sys.argv[sys.argv.index('--input') + 1] == 'xcontrol'\n"
                "assert 'input=pcharge' in pathlib.Path('xcontrol').read_text()\n"
                "pcharge = pathlib.Path('pcharge').read_text().splitlines()\n"
                "assert pcharge[0] == '1' and float(pcharge[1].split()[4]) == 999.0\n"
                f"properties = {expected['properties']!r}\n"
                "raw = {\n"
                "  'partial charges': properties['partial_charges_e'],\n"
                "  'atomic dipole moments': properties['atomic_dipoles_e_bohr'],\n"
                "  'atomic quadrupole moments': properties['atomic_quadrupoles_e_bohr2'],\n"
                "  'number of unpaired electrons': 0,\n"
                "}\n"
                "pathlib.Path('xtbout.json').write_text(json.dumps(raw))\n"
                "gradient = properties['gradient_hartree_per_bohr']\n"
                "rows = [' '.join(map(str, gradient[i:i+3])) for i in range(0, 9, 3)]\n"
                "coords = pathlib.Path('coord').read_text().splitlines()[1:-1]\n"
                f"energy = {expected['properties']['energy_hartree']!r}\n"
                "pathlib.Path('gradient').write_text(\n"
                "  '$grad\\n  cycle = 1 SCF energy = ' + str(energy) + ' |dE/dxyz| = 0.0\\n'\n"
                "  + '\\n'.join(coords) + '\\n' + '\\n'.join(rows) + '\\n$end\\n'\n"
                ")\n"
                "pcgradient = properties['point_charge_gradient_hartree_per_bohr']\n"
                "pathlib.Path('pcgrad').write_text(' '.join(map(str, pcgradient)) + '\\n')\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            output_dir = root / "generated"
            self.run_tool(
                "generate-xtb",
                "--executable",
                str(executable),
                "--output-dir",
                str(output_dir),
                "--case",
                "water_one_pc_gamma999",
            )
            generated = json.loads(
                (output_dir / "water_one_pc_gamma999.json").read_text(encoding="utf-8")
            )
            self.assertEqual(generated["qmmm_input"], expected["qmmm_input"])
            self.assertEqual(generated["properties"], expected["properties"])
            self.assertEqual(
                generated["provenance"]["command"], expected["provenance"]["command"]
            )
            self.assertEqual(
                generated["provenance"]["materialized_input"],
                expected["provenance"]["materialized_input"],
            )
            self.run_tool(
                "compare",
                "--actual-dir",
                str(output_dir),
                "--case",
                "water_one_pc_gamma999",
            )

    def test_open_shell_comparison_requires_scc_state(self) -> None:
        """The open-shell gate cannot silently pass an energy/force-only result."""
        source = REPOSITORY_ROOT / "data" / "conformance" / "golden" / "oh_radical.json"
        with tempfile.TemporaryDirectory() as temporary:
            actual_dir = Path(temporary)
            actual = json.loads(source.read_text(encoding="utf-8"))
            del actual["properties"]["partial_charges_e"]
            (actual_dir / "oh_radical.json").write_text(
                json.dumps(actual), encoding="utf-8"
            )
            completed = self.run_tool(
                "compare",
                "--actual-dir",
                str(actual_dir),
                "--case",
                "oh_radical",
                expected_status=1,
            )
            self.assertIn("missing partial_charges_e", completed.stdout)

    def test_qmmm_comparison_requires_point_charge_forces(self) -> None:
        """A QM-only force result cannot silently satisfy a QM/MM golden."""
        source = (
            REPOSITORY_ROOT
            / "data"
            / "conformance"
            / "golden"
            / "water_one_pc_gamma999.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            actual_dir = Path(temporary)
            actual = json.loads(source.read_text(encoding="utf-8"))
            del actual["properties"]["point_charge_forces_hartree_per_bohr"]
            (actual_dir / "water_one_pc_gamma999.json").write_text(
                json.dumps(actual), encoding="utf-8"
            )
            completed = self.run_tool(
                "compare",
                "--actual-dir",
                str(actual_dir),
                "--case",
                "water_one_pc_gamma999",
                expected_status=1,
            )
            self.assertIn(
                "missing point_charge_forces_hartree_per_bohr", completed.stdout
            )


if __name__ == "__main__":
    unittest.main()
