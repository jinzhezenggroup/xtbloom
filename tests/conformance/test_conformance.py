"""Self-contained tests for the reference corpus tooling.

These tests use only Python's standard library so they can run before the
gpuxtb physics implementation or either Fortran reference package is built.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "gpuxtb_conformance.py"
PUBLIC_API_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "gpuxtb_public_api.py"
INVARIANTS_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "gpuxtb_invariants.py"
MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
SPEC = importlib.util.spec_from_file_location("gpuxtb_conformance_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)
sys.modules.setdefault("gpuxtb_conformance", CONFORMANCE)
PUBLIC_SPEC = importlib.util.spec_from_file_location(
    "gpuxtb_public_api_tool", PUBLIC_API_TOOL
)
assert PUBLIC_SPEC is not None and PUBLIC_SPEC.loader is not None
PUBLIC_API = importlib.util.module_from_spec(PUBLIC_SPEC)
sys.modules[PUBLIC_SPEC.name] = PUBLIC_API
PUBLIC_SPEC.loader.exec_module(PUBLIC_API)
# The invariants tool imports the ABI mirror by its real module name; alias the
# already-loaded module so both tools share one ctypes mirror in-process.
sys.modules.setdefault("gpuxtb_public_api", PUBLIC_API)
INVARIANTS_SPEC = importlib.util.spec_from_file_location(
    "gpuxtb_invariants_tool", INVARIANTS_TOOL
)
assert INVARIANTS_SPEC is not None and INVARIANTS_SPEC.loader is not None
INVARIANTS = importlib.util.module_from_spec(INVARIANTS_SPEC)
sys.modules[INVARIANTS_SPEC.name] = INVARIANTS
INVARIANTS_SPEC.loader.exec_module(INVARIANTS)


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

    def test_primary_oracles_pin_the_reviewed_accuracy(self) -> None:
        """Neither live reference command may regress to its loose CLI default."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for engine in ("tblite", "xtb"):
            reference = manifest["reference_engines"][engine]
            self.assertEqual(reference["accuracy"], 0.0001)
            templates = [reference["cli_command_template"]]
            if engine == "xtb":
                templates.append(reference["qmmm_cli_command_template"])
            for template in templates:
                index = template.index("--acc")
                self.assertEqual(template[index + 1], "0.0001")

        with tempfile.TemporaryDirectory() as temporary:
            invalid = json.loads(json.dumps(manifest))
            invalid["reference_engines"]["tblite"]["accuracy"] = 1.0
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(
                CONFORMANCE.ConformanceError, "tblite accuracy must be 0.0001"
            ):
                CONFORMANCE.check_manifest(path)

    def test_gas_phase_coord_inputs_are_parsed_for_public_consumers(self) -> None:
        """The shared parser handles element case and stops at later directives."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        gas_cases = [
            case for case in manifest["cases"] if case.get("input_schema") is None
        ]
        by_id = {case["id"]: case for case in gas_cases}
        for case in gas_cases:
            parsed = CONFORMANCE.load_turbomole_coord(
                REPOSITORY_ROOT / case["input"], case
            )
            self.assertEqual(len(parsed["atomic_numbers"]), case["atom_count"])
            self.assertEqual(len(parsed["positions_bohr"]), case["atom_count"])
        self.assertEqual(by_id["h3_plus"]["molecular_charge"], 1)
        self.assertEqual(
            CONFORMANCE.load_turbomole_coord(
                REPOSITORY_ROOT / by_id["h3_plus"]["input"], by_id["h3_plus"]
            )["atomic_numbers"],
            [1, 1, 1],
        )

    def test_coord_parser_rejects_malformed_nonfinite_input(self) -> None:
        """Public inference must not turn malformed corpus data into ABI calls."""
        case = {"id": "invalid", "atom_count": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.coord"
            path.write_text("$coord\n nan 0 0 h\n$end\n", encoding="utf-8")
            with self.assertRaisesRegex(CONFORMANCE.ConformanceError, "non-finite"):
                CONFORMANCE.load_turbomole_coord(path, case)

    def test_public_runner_preserves_open_shell_channel_metadata(self) -> None:
        """Unpaired electrons do not implicitly enable spin polarization."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = PUBLIC_API.supported_cases(manifest, ["oh_radical"])
        self.assertEqual([case["id"] for case in cases], ["oh_radical"])
        storage = PUBLIC_API.assemble_batch(MANIFEST, manifest, cases)
        self.assertEqual(storage.unpaired_electrons, [1])
        self.assertEqual(storage.spin_channels, [1])
        self.assertEqual(PUBLIC_API.Batch._fields_[-1][0], "spin_channels")

        completed = subprocess.run(
            [
                sys.executable,
                str(PUBLIC_API_TOOL),
                "--library",
                "/does/not/exist/libgpuxtb.so",
                "--backend",
                "cpu",
                "--case",
                "oh_radical",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("shared library is missing", completed.stderr)
        self.assertNotIn("SKIP oh_radical", completed.stdout)

    def test_public_open_shell_comparison_requires_force_energy_and_charge(
        self,
    ) -> None:
        """The standard xTB OH golden gates every public property it contains."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = PUBLIC_API.supported_cases(manifest, ["oh_radical"])
        case_slice = PUBLIC_API.assemble_batch(MANIFEST, manifest, cases).slices[0]
        actual = {
            "energy_hartree": case_slice.expected["energy_hartree"],
            "forces_hartree_per_bohr": case_slice.expected["forces_hartree_per_bohr"],
            "partial_charges_e": case_slice.expected["partial_charges_e"],
        }
        unsupported: dict[str, str] = {}
        self.assertEqual(
            PUBLIC_API._compare_case(manifest, case_slice, actual, unsupported), []
        )
        actual["energy_hartree"] += 1.0e-3
        failures = PUBLIC_API._compare_case(manifest, case_slice, actual, unsupported)
        self.assertEqual(len(failures), 1)
        self.assertIn("energy_hartree", failures[0])

    def test_public_runner_rejects_cuda_memory_for_cpu_before_loading(self) -> None:
        """Device descriptors cannot accidentally be routed through the CPU backend."""
        for memory_mode in ("device", "mixed"):
            with self.subTest(memory_mode=memory_mode):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(PUBLIC_API_TOOL),
                        "--library",
                        "/does/not/exist/libgpuxtb.so",
                        "--backend",
                        "cpu",
                        "--memory-mode",
                        memory_mode,
                    ],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(completed.returncode, 1)
                self.assertIn("CPU backend only supports", completed.stderr)

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
            self.assertEqual(golden["provenance"]["accuracy"], 0.0001)
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

        # The strict live oracle remains close to the independent constants in
        # xTB's PCEM unit test while avoiding its default-accuracy SCC residue.
        hardness = json.loads(
            (REPOSITORY_ROOT / by_id["water_dimer_6pc_hardness"]["golden"]).read_text(
                encoding="utf-8"
            )
        )["properties"]
        self.assertAlmostEqual(hardness["energy_hartree"], -10.16092775478, delta=1e-11)
        self.assertAlmostEqual(
            hardness["gradient_hartree_per_bohr"][12],
            -2.0334108000991e-3,
            delta=1e-14,
        )
        self.assertAlmostEqual(
            hardness["point_charge_gradient_hartree_per_bohr"][12],
            -2.4831e-4,
            delta=5e-9,
        )
        gamma999 = json.loads(
            (REPOSITORY_ROOT / by_id["water_dimer_6pc_gamma999"]["golden"]).read_text(
                encoding="utf-8"
            )
        )["properties"]
        self.assertAlmostEqual(gamma999["energy_hartree"], -10.16878826896, delta=1e-11)

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
                "    print('tblite version 0.7.0')\n"
                "    raise SystemExit(0)\n"
                "assert sys.argv[sys.argv.index('--method') + 1] == 'gfn2'\n"
                "assert sys.argv[sys.argv.index('--acc') + 1] == '0.0001'\n"
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
            self.assertEqual(generated["provenance"]["accuracy"], 0.0001)
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
                "assert sys.argv[sys.argv.index('--acc') + 1] == '0.0001'\n"
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
                "assert sys.argv[sys.argv.index('--acc') + 1] == '0.0001'\n"
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

    def test_manifest_justifies_each_tolerance_separately(self) -> None:
        """Every gate tolerance records a unit and a property-specific rationale."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for property_name, tolerance in manifest["tolerances"].items():
            if property_name == "rationale":
                continue
            with self.subTest(property_name=property_name):
                self.assertIsInstance(tolerance["atol"], (int, float))
                self.assertIsInstance(tolerance["rtol"], (int, float))
                self.assertIn("unit", tolerance)
                justification = tolerance["justification"]
                self.assertIsInstance(justification, str)
                self.assertGreater(len(justification.strip()), 20)


def _zero_invariant_result(
    geometry: INVARIANTS.Geometry,
) -> INVARIANTS.InvariantResult:
    """Build an all-zero result sized to one geometry."""
    return INVARIANTS.InvariantResult(
        case_id=geometry.case_id,
        molecular_charge=geometry.molecular_charge,
        energy=0.0,
        forces=[0.0] * (3 * len(geometry.atomic_numbers)),
        charges=[0.0] * len(geometry.atomic_numbers),
        point_forces=[0.0] * (3 * len(geometry.point_values)),
    )


def _invariant_geometries() -> list[INVARIANTS.Geometry]:
    """Two small geometries: a neutral gas pair and one QM atom plus a point."""
    return [
        INVARIANTS.Geometry(
            case_id="gas_pair",
            atomic_numbers=[1, 1],
            positions=[0.0, 0.0, 0.0, 0.0, 0.0, 1.4],
            molecular_charge=0,
            unpaired_electrons=0,
            spin_channels=1,
        ),
        INVARIANTS.Geometry(
            case_id="qm_point",
            atomic_numbers=[8],
            positions=[0.0, 0.0, 0.0],
            molecular_charge=0,
            unpaired_electrons=0,
            spin_channels=1,
            point_positions=[3.0, 0.0, 0.0],
            point_values=[-0.5],
            point_gammas=[0.405771],
        ),
    ]


class InvarianceToolTest(unittest.TestCase):
    """Exercise the automated symmetry, conservation, and batch gates."""

    def run_invariant_checks(self, solver, geometries) -> list[str]:
        with redirect_stdout(io.StringIO()):
            return INVARIANTS.run_invariant_checks(solver, geometries, ())

    def test_rotation_matrix_is_proper_and_orthogonal(self) -> None:
        """The Rodrigues rotation used by the gate is deterministic and valid."""
        matrix = INVARIANTS.rotation_matrix((1.0, 1.0, 1.0), 37.0)
        for row in matrix:
            self.assertAlmostEqual(sum(component**2 for component in row), 1.0)
        columns = [[matrix[row][column] for row in range(3)] for column in range(3)]
        for column in columns:
            self.assertAlmostEqual(sum(component**2 for component in column), 1.0)
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        self.assertAlmostEqual(determinant, 1.0, places=12)

    def test_transforms_preserve_pairwise_distances(self) -> None:
        """Translations and rotations are rigid-body geometry transforms."""
        geometry = _invariant_geometries()[0]
        baseline_distance = 1.4
        transforms = [
            *(
                INVARIANTS.translated(geometry, delta)
                for delta in INVARIANTS.TRANSLATION_DELTAS
            ),
            INVARIANTS.rotated(
                geometry, INVARIANTS.rotation_matrix((1.0, 1.0, 1.0), 37.0)
            ),
            INVARIANTS.rotated(
                geometry, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        ]
        for transformed in transforms:
            vector = [
                transformed.positions[3] - transformed.positions[0],
                transformed.positions[4] - transformed.positions[1],
                transformed.positions[5] - transformed.positions[2],
            ]
            self.assertAlmostEqual(
                sum(component**2 for component in vector) ** 0.5,
                baseline_distance,
                places=12,
            )

    def test_zero_solver_passes_every_gate(self) -> None:
        """A trivially symmetric solver satisfies all self-consistency gates."""
        geometries = _invariant_geometries()

        def solver(items):
            return [_zero_invariant_result(item) for item in items]

        failures = self.run_invariant_checks(solver, geometries)
        self.assertEqual(failures, [])

    def test_translation_break_is_detected(self) -> None:
        """A position-dependent energy cannot pass the translation gate."""
        geometries = _invariant_geometries()

        def solver(items):
            results = []
            for item in items:
                result = _zero_invariant_result(item)
                result.energy = 0.5 * item.positions[0]
                results.append(result)
            return results

        failures = self.run_invariant_checks(solver, geometries)
        translation_failures = [
            failure for failure in failures if "translation_invariant" in failure
        ]
        self.assertTrue(translation_failures)
        self.assertTrue(any("energy_hartree" in item for item in translation_failures))

    def test_rotation_break_is_detected(self) -> None:
        """A lab-frame force cannot pass the rotation-covariance gate."""
        geometries = _invariant_geometries()

        def solver(items):
            results = []
            for item in items:
                result = _zero_invariant_result(item)
                if len(item.atomic_numbers) == 2:
                    result.forces[0] = 1.0
                    result.forces[3] = -1.0
                results.append(result)
            return results

        failures = self.run_invariant_checks(solver, geometries)
        rotation_failures = [
            failure for failure in failures if "rotation_covariant" in failure
        ]
        self.assertTrue(rotation_failures)
        self.assertFalse(
            any("total_force" in failure for failure in failures),
            "symmetric constant forces must still conserve net force",
        )

    def test_force_conservation_break_is_detected(self) -> None:
        """A nonzero net force cannot pass the conservation gate."""
        geometries = _invariant_geometries()

        def solver(items):
            results = []
            for item in items:
                result = _zero_invariant_result(item)
                for atom in range(len(item.atomic_numbers)):
                    result.forces[3 * atom] = 1.0
                results.append(result)
            return results

        failures = self.run_invariant_checks(solver, geometries)
        self.assertTrue(any("total_force" in failure for failure in failures))

    def test_batch_dependent_results_are_detected(self) -> None:
        """Per-call state must not make ragged batches differ from sequential runs."""
        geometries = _invariant_geometries()
        invocation = {"count": 0}

        def solver(items):
            invocation["count"] += 1
            return [
                INVARIANTS.InvariantResult(
                    case_id=item.case_id,
                    molecular_charge=item.molecular_charge,
                    energy=float(invocation["count"]) * 1.0e-6,
                    forces=[0.0] * (3 * len(item.atomic_numbers)),
                    charges=[0.0] * len(item.atomic_numbers),
                    point_forces=[0.0] * (3 * len(item.point_values)),
                )
                for item in items
            ]

        failures = self.run_invariant_checks(solver, geometries)
        self.assertTrue(any("batch_vs_sequential" in failure for failure in failures))

    def test_cli_rejects_cuda_memory_for_cpu_backend(self) -> None:
        """The invariance CLI enforces the same placement rule as the golden runner."""
        for memory_mode in ("device", "mixed"):
            with self.subTest(memory_mode=memory_mode):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(INVARIANTS_TOOL),
                        "--library",
                        "/does/not/exist/libgpuxtb.so",
                        "--backend",
                        "cpu",
                        "--memory-mode",
                        memory_mode,
                    ],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(completed.returncode, 1)
                self.assertIn("CPU backend only supports", completed.stderr)

    def test_cli_requires_an_existing_library(self) -> None:
        """A missing shared library is a hard error, not a skip."""
        completed = subprocess.run(
            [
                sys.executable,
                str(INVARIANTS_TOOL),
                "--library",
                "/does/not/exist/libgpuxtb.so",
                "--backend",
                "cpu",
                "--case",
                "ketene",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("shared library is missing", completed.stderr)


if __name__ == "__main__":
    unittest.main()
