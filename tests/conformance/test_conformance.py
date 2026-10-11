"""Self-contained tests for the reference corpus tooling.

These tests use only Python's standard library so they can run before the
xtbloom physics implementation or either Fortran reference package is built.
"""

from __future__ import annotations

import ctypes
import importlib.util
import io
import itertools
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "xtbloom_conformance.py"
PUBLIC_API_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "xtbloom_public_api.py"
INVARIANTS_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "xtbloom_invariants.py"
MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
GFN1_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "gfn1_conformance.py"
GFN1_MANIFEST = REPOSITORY_ROOT / "data" / "conformance" / "gfn1" / "manifest.json"
SPEC = importlib.util.spec_from_file_location("xtbloom_conformance_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)
sys.modules.setdefault("xtbloom_conformance", CONFORMANCE)
GFN1_SPEC = importlib.util.spec_from_file_location("gfn1_conformance_tool", GFN1_TOOL)
assert GFN1_SPEC is not None and GFN1_SPEC.loader is not None
GFN1_CONFORMANCE = importlib.util.module_from_spec(GFN1_SPEC)
GFN1_SPEC.loader.exec_module(GFN1_CONFORMANCE)
sys.modules.setdefault("gfn1_conformance", GFN1_CONFORMANCE)
PERIODIC_TOOL = REPOSITORY_ROOT / "tools" / "conformance" / "periodic_gfn2.py"
PERIODIC_SPEC = importlib.util.spec_from_file_location(
    "periodic_gfn2_tool", PERIODIC_TOOL
)
assert PERIODIC_SPEC is not None and PERIODIC_SPEC.loader is not None
PERIODIC = importlib.util.module_from_spec(PERIODIC_SPEC)
PERIODIC_SPEC.loader.exec_module(PERIODIC)
sys.modules.setdefault("periodic_gfn2", PERIODIC)
PUBLIC_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_public_api_tool", PUBLIC_API_TOOL
)
assert PUBLIC_SPEC is not None and PUBLIC_SPEC.loader is not None
PUBLIC_API = importlib.util.module_from_spec(PUBLIC_SPEC)
sys.modules[PUBLIC_SPEC.name] = PUBLIC_API
PUBLIC_SPEC.loader.exec_module(PUBLIC_API)
# The invariants tool imports the ABI mirror by its real module name; alias the
# already-loaded module so both tools share one ctypes mirror in-process.
sys.modules.setdefault("xtbloom_public_api", PUBLIC_API)
INVARIANTS_SPEC = importlib.util.spec_from_file_location(
    "xtbloom_invariants_tool", INVARIANTS_TOOL
)
assert INVARIANTS_SPEC is not None and INVARIANTS_SPEC.loader is not None
INVARIANTS = importlib.util.module_from_spec(INVARIANTS_SPEC)
sys.modules[INVARIANTS_SPEC.name] = INVARIANTS
INVARIANTS_SPEC.loader.exec_module(INVARIANTS)


class ConformanceToolTest(unittest.TestCase):
    """Exercise integrity checks, supported result schemas, and live generation."""

    def test_public_compute_options_mirror_includes_abi_v3_policy(self) -> None:
        """The standalone conformance runner must pass the complete native ABI image."""
        self.assertEqual(PUBLIC_API.ctypes.sizeof(PUBLIC_API.ComputeOptions), 80)
        self.assertEqual(PUBLIC_API.ComputeOptions.scc_start_mode.offset, 48)
        self.assertEqual(PUBLIC_API.ComputeOptions.scc_mixer.offset, 56)
        self.assertEqual(PUBLIC_API.ComputeOptions.scc_mixer_history.offset, 60)
        self.assertEqual(PUBLIC_API.ComputeOptions.scc_mixer_damping.offset, 64)
        self.assertEqual(PUBLIC_API.ComputeOptions.determinism.offset, 72)
        self.assertEqual(PUBLIC_API.ComputeOptions.reserved_v3.offset, 76)

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
        self.assertIn("14 cases", completed.stdout)

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
        # The public mirror keeps the ABI-v3 interaction and ABI-v4 lattice
        # suffixes, matching python/xtbloom/library.py.
        self.assertEqual(
            [name for name, _ in PUBLIC_API.Batch._fields_[-6:]],
            [
                "spin_channels",
                "total_interactions",
                "interaction_descriptors",
                "interaction_payload",
                "cell_matrices",
                "periodic_axes",
            ],
        )
        self.assertEqual(
            [name for name, _ in PUBLIC_API.BatchResult._fields_[-5:]],
            [
                "dipole_moments",
                "quadrupole_moments",
                "wiberg_orders",
                "spin_populations",
                "strain_derivatives",
            ],
        )
        self.assertEqual(ctypes.sizeof(PUBLIC_API.BatchResult), 304)
        self.assertEqual(PUBLIC_API.BatchResult.dipole_moments.offset, 184)

        completed = subprocess.run(
            [
                sys.executable,
                str(PUBLIC_API_TOOL),
                "--library",
                "/does/not/exist/libxtbloom.so",
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

    def test_public_runner_selects_gfn1_cpu_and_cuda(self) -> None:
        """GFN1 corpus publishes both backends and preserves open-shell input."""
        manifest = json.loads(GFN1_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            PUBLIC_API.model_tag(manifest), PUBLIC_API.XTBLOOM_MODEL_GFN1_XTB
        )
        self.assertEqual(len(PUBLIC_API.supported_cases(manifest, None, "cpu")), 8)
        self.assertEqual(len(PUBLIC_API.supported_cases(manifest, None, "cuda")), 8)
        cases = PUBLIC_API.supported_cases(manifest, ["gfn1_oh_radical"])
        storage = PUBLIC_API.assemble_batch(GFN1_MANIFEST, manifest, cases)
        self.assertEqual(storage.unpaired_electrons, [1])
        self.assertEqual(storage.spin_channels, [1])

    def test_gfn1_invariant_loader_uses_model_specific_inputs(self) -> None:
        """GFN1 invariants load both coord and PCEM inputs without GFN2 parsers."""
        manifest = json.loads(GFN1_MANIFEST.read_text(encoding="utf-8"))
        cases = PUBLIC_API.supported_cases(
            manifest,
            ["gfn1_h3_plus", "gfn1_water_dimer_6pc_hardness"],
            "cpu",
        )
        geometries = INVARIANTS.load_geometries(GFN1_MANIFEST, manifest, cases)
        self.assertEqual(
            [geometry.case_id for geometry in geometries],
            ["gfn1_h3_plus", "gfn1_water_dimer_6pc_hardness"],
        )
        self.assertEqual(geometries[0].atomic_numbers, [1, 1, 1])
        self.assertEqual(len(geometries[1].point_values), 6)

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

    def test_periodic_charged_diagnostic_is_not_an_acceptance_failure(self) -> None:
        """The no-background charged-cell probe remains executable evidence only."""
        manifest_path = (
            REPOSITORY_ROOT / "data" / "conformance" / "periodic" / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        case = next(
            item
            for item in manifest["cases"]
            if item["oracle_role"] == "diagnostic-unbackgrounded-charged"
        )
        case_slice = PUBLIC_API.assemble_batch(manifest_path, manifest, [case]).slices[
            0
        ]
        # Deliberately use values that would fail every numerical tolerance if
        # this diagnostic were accidentally treated as a primary oracle.
        actual = {
            "energy_hartree": 123.0,
            "forces_hartree_per_bohr": [123.0] * (3 * case["atom_count"]),
            "partial_charges_e": [123.0] * case["atom_count"],
            "strain_derivatives_hartree": [123.0] * 9,
        }
        with redirect_stdout(io.StringIO()) as captured:
            failures = PUBLIC_API._compare_case(manifest, case_slice, actual, {})
        self.assertEqual(failures, [])
        self.assertIn("SKIP periodic_lithium_cation_diagnostic", captured.getvalue())
        self.assertIn("diagnostic-only", captured.getvalue())

    def test_public_runner_rejects_cuda_memory_for_cpu_before_loading(self) -> None:
        """Device descriptors cannot accidentally be routed through the CPU backend."""
        for memory_mode in ("device", "mixed"):
            with self.subTest(memory_mode=memory_mode):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(PUBLIC_API_TOOL),
                        "--library",
                        "/does/not/exist/libxtbloom.so",
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
        self.assertEqual(len(qmmm_cases), 5)
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
                "xtbloom-xtb-pcem-cli-v1",
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

    def test_public_ragged_batch_covers_changing_point_charge_counts(self) -> None:
        """The expected-data batch must retain gas, one-PC, and six-PC members."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = PUBLIC_API.supported_cases(manifest, None, "cpu")
        storage = PUBLIC_API.assemble_batch(MANIFEST, manifest, cases)
        counts = [
            storage.point_charge_offsets[index + 1]
            - storage.point_charge_offsets[index]
            for index in range(len(cases))
        ]
        self.assertEqual(set(counts), {0, 1, 6})
        self.assertTrue(
            any(left != right for left, right in itertools.pairwise(counts))
        )
        for case_slice, count in zip(storage.slices, counts, strict=True):
            self.assertEqual(case_slice.point_end - case_slice.point_begin, count)

    def test_close_and_coincident_finite_hardness_sites_are_pinned(self) -> None:
        """Short-range QMMM rows use finite O hardness and exact reviewed geometry."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        expected_separations = {
            "water_one_pc_close_hardness": 0.25,
            "water_one_pc_coincident_hardness": 0.0,
        }
        for case_id, expected_separation in expected_separations.items():
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                document = CONFORMANCE.load_qmmm_input(
                    REPOSITORY_ROOT / case["input"],
                    case,
                    manifest["reference_engines"]["xtb"][
                        "point_charge_hardness_hartree"
                    ],
                )
                points = document["external_point_charges"]
                self.assertEqual(points["gamma_mode"], "element_hardness")
                self.assertEqual(points["source_atomic_numbers"], [8])
                self.assertEqual(points["gammas_hartree"], [0.451896])
                oxygen = document["qm"]["positions_bohr"][0]
                point = points["positions_bohr"][0]
                separation = math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(oxygen, point, strict=True)
                    )
                )
                self.assertAlmostEqual(separation, expected_separation, places=14)
                golden = json.loads(
                    (REPOSITORY_ROOT / case["golden"]).read_text(encoding="utf-8")
                )
                properties = golden["properties"]
                self.assertTrue(math.isfinite(properties["energy_hartree"]))
                self.assertTrue(
                    all(
                        math.isfinite(value)
                        for value in properties["point_charge_forces_hartree_per_bohr"]
                    )
                )

    def test_minimal_element_and_ion_rows_cover_transition_heavy_and_boundary(
        self,
    ) -> None:
        """Independent atom/ion rows isolate Z=30, Z=53 anion, and Z=86."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        expected = {
            "zn_atom": ([30], 0),
            "i_anion": ([53], -1),
            "rn_atom": ([86], 0),
        }
        for case_id, (atomic_numbers, molecular_charge) in expected.items():
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                parsed = CONFORMANCE.load_turbomole_coord(
                    REPOSITORY_ROOT / case["input"], case
                )
                self.assertEqual(parsed["atomic_numbers"], atomic_numbers)
                self.assertEqual(case["molecular_charge"], molecular_charge)
                self.assertEqual(case["reference_engine"], "tblite")
                golden = json.loads(
                    (REPOSITORY_ROOT / case["golden"]).read_text(encoding="utf-8")
                )
                self.assertTrue(math.isfinite(golden["properties"]["energy_hartree"]))

    def test_difficult_scc_status_ledger_retains_default_nonconvergence(self) -> None:
        """The conformance ledger records status evidence without choosing a policy."""
        evidence_root = (
            REPOSITORY_ROOT / "data/conformance/evidence/tmacl-temperature-continuation"
        )
        manifest = json.loads(
            (evidence_root / "manifest.json").read_text(encoding="utf-8")
        )
        trace_lines = (
            (evidence_root / "tmacl_trace_300K.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        terminal = next(line for line in reversed(trace_lines) if line.strip())
        readme = (evidence_root / "README.md").read_text(encoding="utf-8")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(terminal.split()[-2:], ["7", "0"])
        self.assertIn(
            "does not select temperature\ncontinuation over a deterministic "
            "mixer-policy change",
            readme,
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
        """Tblite gradients are negated exactly once before force comparison."""
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
        """Verify command construction and normalization with a fake executable."""
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

    def test_electric_field_case_packs_payload_and_matches_golden(self) -> None:
        """The field pilot retains energy provenance and explicit force evidence."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cases = {case["id"]: case for case in manifest["cases"]}
        case = cases["water_efield"]
        self.assertEqual(case["efield"], [0.003, -0.004, 0.005])
        self.assertNotIn("xtbloom_backends", case)
        self.assertEqual(case["xtbloom_oracle_properties"], ["energy_hartree"])
        self.assertEqual(
            case["xtbloom_force_evidence"], "public_energy_finite_difference"
        )
        self.assertEqual(
            [item["id"] for item in PUBLIC_API.supported_cases(manifest, None, "cuda")],
            [item["id"] for item in manifest["cases"]],
        )
        self.assertIn(case, PUBLIC_API.supported_cases(manifest, None, "cpu"))

        command = CONFORMANCE.tblite_command(
            Path("{executable}"), case, Path("{input}"), Path("{json_output}")
        )
        efield_index = command.index("--efield")
        expected_vperangstrom = ",".join(
            f"{component / 0.019446903964791384:.12g}"
            for component in (0.003, -0.004, 0.005)
        )
        self.assertEqual(command[efield_index + 1], expected_vperangstrom)

        storage = PUBLIC_API.assemble_batch(MANIFEST, manifest, [case])
        self.assertEqual(storage.efields, [[0.003, -0.004, 0.005]])
        descriptors, payload = PUBLIC_API.pack_efield_interactions(storage.efields)
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(
            descriptors[0].type, PUBLIC_API.XTBLOOM_INTERACTION_ELECTRIC_FIELD
        )
        self.assertEqual(descriptors[0].flags, 0)
        self.assertEqual(descriptors[0].system_index, 0)
        self.assertEqual(descriptors[0].payload_offset, 0)
        self.assertEqual(descriptors[0].payload_size, 32)
        self.assertEqual(len(payload), 32)
        self.assertEqual(struct.unpack("<2i", payload[:8]), (1, 0))
        self.assertEqual(struct.unpack("<3d", payload[8:]), tuple(case["efield"]))
        zero_descriptors, zero_payload = PUBLIC_API.pack_efield_interactions(
            [[0.0, 0.0, 0.0]]
        )
        self.assertEqual(len(zero_descriptors), 1)
        self.assertEqual(len(zero_payload), 32)

        golden = json.loads(
            (REPOSITORY_ROOT / case["golden"]).read_text(encoding="utf-8")
        )
        properties = golden["properties"]
        self.assertTrue(
            all(math.isfinite(value) for value in properties["forces_hartree_per_bohr"])
        )
        self.assertEqual(
            properties["forces_hartree_per_bohr"],
            [-value for value in properties["gradient_hartree_per_bohr"]],
        )
        # The pinned tblite 0.7.0 energy remains authoritative. Its analytic
        # field gradient is retained byte-for-byte for provenance diagnostics,
        # but xtbloom forces use public energy finite differences instead.
        self.assertAlmostEqual(
            properties["energy_hartree"], -4.772344124360096, delta=5e-7
        )
        for axis, expected in enumerate(
            (
                -0.002908945025103449,
                0.00481382916179114,
                -0.09951281990856943,
                -0.003040513847078862,
                0.05991143495260846,
                0.041857108572417936,
                -0.0030505410569194313,
                -0.05272526420893061,
                0.04265571145431526,
            )
        ):
            self.assertAlmostEqual(
                properties["gradient_hartree_per_bohr"][axis],
                expected,
                delta=5e-7,
            )

        storage = PUBLIC_API.assemble_batch(MANIFEST, manifest, [case])
        case_slice = storage.slices[0]
        actual = {
            "energy_hartree": properties["energy_hartree"],
            "forces_hartree_per_bohr": [99.0] * 9,
        }
        with redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(
                PUBLIC_API._compare_case(manifest, case_slice, actual, {}), []
            )
        self.assertIn("diagnostic-only", captured.getvalue())

        # The standalone directory comparator accepts both canonical public
        # artifacts and its documented minimal xtbloom result shape. Neither
        # may re-enable the diagnostic tblite force comparison.
        with tempfile.TemporaryDirectory() as temporary:
            actual_dir = Path(temporary)
            public_document = {
                "properties": actual,
                "provenance": {"engine": "xtbloom"},
            }
            (actual_dir / "water_efield.json").write_text(
                json.dumps(public_document), encoding="utf-8"
            )
            completed = self.run_tool(
                "compare",
                "--actual-dir",
                str(actual_dir),
                "--case",
                "water_efield",
            )
            self.assertIn("diagnostic-only", completed.stdout)

            (actual_dir / "water_efield.json").write_text(
                json.dumps({"energy_hartree": properties["energy_hartree"]}),
                encoding="utf-8",
            )
            self.run_tool(
                "compare",
                "--actual-dir",
                str(actual_dir),
                "--case",
                "water_efield",
            )

    def test_field_provenance_requires_the_efield_template_token(self) -> None:
        """Legacy tblite templates remain valid only for field-free cases."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        reference = manifest["reference_engines"]["tblite"]
        cases = {case["id"]: case for case in manifest["cases"]}
        field_templates = CONFORMANCE.accepted_tblite_command_templates(
            reference, cases["water_efield"]
        )
        plain_templates = CONFORMANCE.accepted_tblite_command_templates(
            reference, cases["h3_plus"]
        )
        self.assertEqual(len(field_templates), 1)
        self.assertIn(CONFORMANCE.EFIELD_COMMAND_TOKEN, field_templates[0])
        self.assertEqual(len(plain_templates), 2)
        self.assertNotIn(CONFORMANCE.EFIELD_COMMAND_TOKEN, plain_templates[1])

    def test_field_free_batch_does_not_bind_empty_interaction_owners(self) -> None:
        """A focused device-style plain batch never uploads zero-byte owners."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        case = next(case for case in manifest["cases"] if case["id"] == "h3_plus")
        storage = PUBLIC_API.assemble_batch(MANIFEST, manifest, [case])
        self.assertEqual(storage.efields, [None])

        class FakeLibrary:
            @staticmethod
            def xtbloom_batch_init(_batch: object, _size: int) -> int:
                return PUBLIC_API.XTBLOOM_STATUS_SUCCESS

        class RejectEmptyOwnerMemory:
            @staticmethod
            def input(*_args: object) -> PUBLIC_API.ConstBuffer:
                return PUBLIC_API.ConstBuffer(
                    None, 0, PUBLIC_API.XTBLOOM_MEMORY_HOST, 0
                )

            @staticmethod
            def input_owner(*_args: object) -> PUBLIC_API.ConstBuffer:
                raise AssertionError(
                    "field-free batches must not bind interaction owners"
                )

        batch = PUBLIC_API._make_batch(
            FakeLibrary(), storage, RejectEmptyOwnerMemory(), include_spin_channels=True
        )
        self.assertEqual(batch.total_interactions, 0)

    def test_cuda_field_selection_reaches_the_public_runner(self) -> None:
        """The released CUDA field case proceeds to shared-library loading."""
        completed = subprocess.run(
            [
                sys.executable,
                str(PUBLIC_API_TOOL),
                "--library",
                "/does/not/exist/libxtbloom.so",
                "--backend",
                "cuda",
                "--case",
                "water_efield",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("SKIP water_efield", completed.stdout)
        self.assertIn("shared library is missing", completed.stderr)

    def test_mixed_mode_splits_interactions_and_publishes_dipoles(self) -> None:
        """Mixed conformance covers independent attachment and result placement."""
        memory = object.__new__(PUBLIC_API.DescriptorMemory)
        memory.mode = "mixed"
        memory.cuda = object()
        memory.device_outputs = []
        self.assertTrue(memory._is_device("interaction_descriptors"))
        self.assertFalse(memory._is_device("interaction_payload"))
        self.assertTrue(memory._is_device("dipole_moments"))

        class FakeLibrary:
            @staticmethod
            def xtbloom_compute_options_init(_options: object, _size: int) -> int:
                return PUBLIC_API.XTBLOOM_STATUS_SUCCESS

        options = PUBLIC_API.pinned_compute_options(
            FakeLibrary(),
            PUBLIC_API.XTBLOOM_MODEL_GFN2_XTB,
            request_forces=True,
            request_charges=True,
            request_point_forces=False,
        )
        self.assertTrue(options.flags & PUBLIC_API.XTBLOOM_COMPUTE_DIPOLE_MOMENTS)

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
                "  'atomic quadrupole moments': "
                "properties['atomic_quadrupoles_e_bohr2'],\n"
                "  'number of unpaired electrons': 1,\n"
                "}\n"
                "pathlib.Path('xtbout.json').write_text(json.dumps(raw))\n"
                "gradient = properties['gradient_hartree_per_bohr']\n"
                "rows = [' '.join(map(str, gradient[i:i+3])) for i in range(0, 6, 3)]\n"
                "pathlib.Path('gradient').write_text(\n"
                "  '$grad\\n  cycle = 1 SCF energy = -4.42833345932 "
                "|dE/dxyz| = 0.0\\n'\n"
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
                "  'atomic quadrupole moments': "
                "properties['atomic_quadrupoles_e_bohr2'],\n"
                "  'number of unpaired electrons': 0,\n"
                "}\n"
                "pathlib.Path('xtbout.json').write_text(json.dumps(raw))\n"
                "gradient = properties['gradient_hartree_per_bohr']\n"
                "rows = [' '.join(map(str, gradient[i:i+3])) for i in range(0, 9, 3)]\n"
                "coords = pathlib.Path('coord').read_text().splitlines()[1:-1]\n"
                f"energy = {expected['properties']['energy_hartree']!r}\n"
                "pathlib.Path('gradient').write_text(\n"
                "  '$grad\\n  cycle = 1 SCF energy = ' + str(energy) +"
                " ' |dE/dxyz| = 0.0\\n'\n"
                "  + '\\n'.join(coords) + '\\n' + '\\n'.join(rows) + '\\n$end\\n'\n"
                ")\n"
                "pcgradient = properties['point_charge_gradient_hartree_per_bohr']\n"
                "pathlib.Path('pcgrad').write_text("
                "' '.join(map(str, pcgradient)) + '\\n')\n",
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
        dipoles=[0.0, 0.0, 0.0],
        efield=None if geometry.efield is None else list(geometry.efield),
    )


def force_for(case_id: str, atom: int, axis: int) -> float:
    """Deterministic per-(case, atom, axis) force for the finite-difference tests."""
    return 0.01 * ((ord(case_id[0]) + 3 * atom + 5 * axis) % 9 - 4)


def linear_energy_solver() -> Callable[
    [list[INVARIANTS.Geometry]], list[INVARIANTS.InvariantResult]
]:
    """Return a solver whose energy is the linear form of consistent forces.

    The energy equals ``-sum(force * coordinate)`` (so ``-dE/dx`` reproduces
    ``force``), and the published analytic force equals the same ``force``
    vector, so every central finite difference matches the analytic force.
    """

    def solver(
        items: list[INVARIANTS.Geometry],
    ) -> list[INVARIANTS.InvariantResult]:
        results: list[INVARIANTS.InvariantResult] = []
        for item in items:
            atom_count = len(item.atomic_numbers)
            forces = [
                force_for(item.case_id, atom, axis)
                for atom in range(atom_count)
                for axis in range(3)
            ]
            energy = -sum(
                force * coordinate
                for force, coordinate in zip(forces, item.positions, strict=True)
            )
            results.append(
                INVARIANTS.InvariantResult(
                    case_id=item.case_id,
                    molecular_charge=item.molecular_charge,
                    energy=energy,
                    forces=forces,
                    charges=[0.0] * atom_count,
                    point_forces=[0.0] * (3 * len(item.point_values)),
                    dipoles=[0.0, 0.0, 0.0],
                    efield=None if item.efield is None else list(item.efield),
                )
            )
        return results

    return solver


def wrong_first_component_baseline(
    solver: Callable[[list[INVARIANTS.Geometry]], list[INVARIANTS.InvariantResult]],
    geometries: list[INVARIANTS.Geometry],
) -> list[INVARIANTS.InvariantResult]:
    """Return consistent baselines except for one deliberately wrong force."""
    baseline = solver(geometries)
    baseline[0].forces[0] += 1.0  # analytic force no longer matches dE/dx
    return baseline


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

    def run_invariant_checks(
        self,
        solver: Callable[[list[INVARIANTS.Geometry]], list[INVARIANTS.InvariantResult]],
        geometries: list[INVARIANTS.Geometry],
    ) -> list[str]:
        """Run every invariant while suppressing expected progress output."""
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

    def test_field_geometry_is_preserved_and_rotated_with_the_system(self) -> None:
        """Invariant batches carry the field and rotate it as a Cartesian vector."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        case = next(case for case in manifest["cases"] if case["id"] == "water_efield")
        geometry = INVARIANTS.load_geometries(MANIFEST, manifest, [case])[0]
        self.assertEqual(geometry.efield, [0.003, -0.004, 0.005])
        self.assertEqual(
            INVARIANTS.geometry_storage([geometry]).efields,
            [[0.003, -0.004, 0.005]],
        )
        self.assertEqual(
            INVARIANTS.translated(geometry, (1.0, 2.0, 3.0)).efield,
            geometry.efield,
        )
        self.assertEqual(
            INVARIANTS.displaced_atom(geometry, 0, 1, 0.1).efield,
            geometry.efield,
        )
        rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertEqual(
            INVARIANTS.rotated(geometry, rotation).efield,
            [0.004, 0.003, 0.005],
        )

    def test_charged_field_probe_is_derived_from_committed_h3_plus(self) -> None:
        """The charged public probe changes only the diagnostic ID and field."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        case = next(case for case in manifest["cases"] if case["id"] == "h3_plus")
        source = INVARIANTS.load_geometries(MANIFEST, manifest, [case])[0]
        probe = INVARIANTS.charged_field_probe([source])
        self.assertIsNotNone(probe)
        assert probe is not None
        self.assertEqual(probe.case_id, INVARIANTS.CHARGED_FIELD_PROBE_ID)
        self.assertEqual(probe.atomic_numbers, source.atomic_numbers)
        self.assertEqual(probe.positions, source.positions)
        self.assertEqual(probe.molecular_charge, 1)
        self.assertEqual(probe.unpaired_electrons, 0)
        self.assertEqual(probe.spin_channels, 1)
        self.assertEqual(probe.point_positions, source.point_positions)
        self.assertEqual(probe.point_values, source.point_values)
        self.assertEqual(probe.point_gammas, source.point_gammas)
        self.assertEqual(probe.efield, [0.003, -0.004, 0.005])
        self.assertIsNone(source.efield)
        self.assertAlmostEqual(
            sum(
                field * delta
                for field, delta in zip(
                    INVARIANTS.CHARGED_FIELD_PROBE_EFIELD,
                    INVARIANTS.CHARGED_FIELD_PROBE_DELTA,
                    strict=True,
                )
            ),
            0.043,
        )
        self.assertIsNone(INVARIANTS.charged_field_probe(_invariant_geometries()))

    def test_charged_field_probe_public_singleton_laws_and_failures(self) -> None:
        """The charged H3+ gate makes two singleton calls and diagnoses each law."""
        probe = INVARIANTS.Geometry(
            case_id=INVARIANTS.CHARGED_FIELD_PROBE_ID,
            atomic_numbers=[1, 1, 1],
            positions=[
                -0.47073898552969,
                0.81534384004086,
                0.0,
                -0.47073898552969,
                -0.81534384004086,
                0.0,
                0.94147797105939,
                0.0,
                0.0,
            ],
            molecular_charge=1,
            unpaired_electrons=0,
            spin_channels=1,
            efield=list(INVARIANTS.CHARGED_FIELD_PROBE_EFIELD),
        )

        def run(perturbation: str | None) -> tuple[list[str], list[list[float]]]:
            calls: list[list[float]] = []

            def solver(
                items: list[INVARIANTS.Geometry],
            ) -> list[INVARIANTS.InvariantResult]:
                self.assertEqual(len(items), 1)
                item = items[0]
                calls.append(list(item.positions))
                translated_call = len(calls) == 2
                energy = 1.25 - (0.043 if translated_call else 0.0)
                dipole = [0.4, -0.2, 0.1]
                if translated_call:
                    dipole = [
                        dipole[axis] + INVARIANTS.CHARGED_FIELD_PROBE_DELTA[axis]
                        for axis in range(3)
                    ]
                force = [component / 3.0 for component in item.efield or ()] * 3
                if perturbation == "energy" and translated_call:
                    energy += 0.5
                if perturbation == "dipole" and translated_call:
                    dipole[0] += 0.5
                if perturbation == "force":
                    force[0] += 0.5
                return [
                    INVARIANTS.InvariantResult(
                        case_id=item.case_id,
                        molecular_charge=item.molecular_charge,
                        energy=energy,
                        forces=force,
                        charges=[1.0 / 3.0] * 3,
                        point_forces=[],
                        dipoles=dipole,
                        efield=list(item.efield or ()),
                    )
                ]

            with redirect_stdout(io.StringIO()):
                failures: list[str] = []
                INVARIANTS.gate_charged_field_probe(solver, probe, failures)
            return failures, calls

        failures, calls = run(None)
        self.assertEqual(failures, [])
        self.assertEqual(calls[0], probe.positions)
        self.assertEqual(
            calls[1],
            INVARIANTS.translated(
                probe, INVARIANTS.CHARGED_FIELD_PROBE_DELTA
            ).positions,
        )
        expected_labels = {
            "energy": "charged_field_translation_shift",
            "dipole": "charged_field_origin_shift",
            "force": "total_force charged_field_",
        }
        for perturbation, label in expected_labels.items():
            with self.subTest(perturbation=perturbation):
                failures, _ = run(perturbation)
                self.assertTrue(any(label in failure for failure in failures))

    def test_zero_solver_passes_every_gate(self) -> None:
        """A trivially symmetric solver satisfies all self-consistency gates."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            return [_zero_invariant_result(item) for item in items]

        failures = self.run_invariant_checks(solver, geometries)
        self.assertEqual(failures, [])

    def test_translation_break_is_detected(self) -> None:
        """A position-dependent energy cannot pass the translation gate."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            results = []
            for item in items:
                result = _zero_invariant_result(item)
                result.energy = 0.5 * item.positions[0]
                results.append(result)
            return results

        failures = self.run_invariant_checks(solver, geometries)
        translation_failures = [
            failure
            for failure in failures
            if "translation_invariant" in failure or "translation_covariant" in failure
        ]
        self.assertTrue(translation_failures)
        self.assertTrue(any("energy_hartree" in item for item in translation_failures))

    def test_charged_field_translation_laws_pass(self) -> None:
        """Charged field energy and dipole acquire their exact origin shifts."""
        delta = (2.0, -3.0, 5.0)
        field = [0.1, -0.2, 0.3]
        baseline = INVARIANTS.InvariantResult(
            case_id="charged_field",
            molecular_charge=-1,
            energy=4.0,
            forces=[-0.1, 0.2, -0.3],
            charges=[-1.0],
            point_forces=[],
            dipoles=[0.4, 0.5, 0.6],
            efield=field,
        )
        shifted = INVARIANTS.InvariantResult(
            case_id="charged_field",
            molecular_charge=-1,
            energy=4.0 + sum(field[axis] * delta[axis] for axis in range(3)),
            forces=list(baseline.forces),
            charges=list(baseline.charges),
            point_forces=[],
            dipoles=[baseline.dipoles[axis] - delta[axis] for axis in range(3)],
            efield=field,
        )
        with redirect_stdout(io.StringIO()):
            failures: list[str] = []
            INVARIANTS.gate_translation_invariance(
                [baseline],
                [shifted],
                delta,
                INVARIANTS.INVARIANT_ENERGY_ATOL,
                INVARIANTS.INVARIANT_FORCE_ATOL,
                INVARIANTS.INVARIANT_CHARGE_ATOL,
                INVARIANTS.INVARIANT_DIPOLE_ATOL,
                failures,
            )
        self.assertEqual(failures, [])

        shifted.dipoles[0] += 0.5
        with redirect_stdout(io.StringIO()):
            failures = []
            INVARIANTS.gate_translation_invariance(
                [baseline],
                [shifted],
                delta,
                INVARIANTS.INVARIANT_ENERGY_ATOL,
                INVARIANTS.INVARIANT_FORCE_ATOL,
                INVARIANTS.INVARIANT_CHARGE_ATOL,
                INVARIANTS.INVARIANT_DIPOLE_ATOL,
                failures,
            )
        self.assertTrue(any("translation_origin_shift" in item for item in failures))

    def test_rotation_break_is_detected(self) -> None:
        """A lab-frame force cannot pass the rotation-covariance gate."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
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

    def test_dipole_rotation_break_is_detected(self) -> None:
        """A lab-frame molecular dipole cannot pass rotation covariance."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            results = [_zero_invariant_result(item) for item in items]
            for result in results:
                result.dipoles = [1.0, 0.0, 0.0]
            return results

        failures = self.run_invariant_checks(solver, geometries)
        self.assertTrue(
            any(
                "molecular_dipole_e_bohr rotation_covariant" in failure
                for failure in failures
            )
        )

    def test_force_conservation_break_is_detected(self) -> None:
        """A nonzero net force cannot pass the conservation gate."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            results = []
            for item in items:
                result = _zero_invariant_result(item)
                for atom in range(len(item.atomic_numbers)):
                    result.forces[3 * atom] = 1.0
                results.append(result)
            return results

        failures = self.run_invariant_checks(solver, geometries)
        self.assertTrue(any("total_force" in failure for failure in failures))

    def test_field_force_balance_includes_q_times_e_and_point_forces(self) -> None:
        """The combined QM/point force balances the external ``Q E`` force."""
        result = INVARIANTS.InvariantResult(
            case_id="charged_field_with_point",
            molecular_charge=2,
            energy=0.0,
            forces=[0.25, -0.30, 0.50],
            charges=[2.0],
            point_forces=[-0.05, 0.10, 0.10],
            dipoles=[0.0, 0.0, 0.0],
            efield=[0.10, -0.10, 0.30],
        )
        with redirect_stdout(io.StringIO()):
            failures: list[str] = []
            INVARIANTS.gate_force_conservation(
                [result], INVARIANTS.INVARIANT_NET_FORCE_ATOL, failures
            )
        self.assertEqual(failures, [])

        result.forces[0] -= 0.2
        with redirect_stdout(io.StringIO()):
            failures = []
            INVARIANTS.gate_force_conservation(
                [result], INVARIANTS.INVARIANT_NET_FORCE_ATOL, failures
            )
        self.assertTrue(any("total_force_axis_0" in failure for failure in failures))

    def test_nonfinite_array_component_is_detected(self) -> None:
        """A non-leading NaN cannot be hidden by maximum-error reduction."""
        passed, message = INVARIANTS._compare(
            "gas_pair",
            "forces_hartree_per_bohr",
            [0.0, 0.0],
            [0.0, float("nan")],
            1.0e-12,
        )
        self.assertFalse(passed)
        self.assertIn("non-finite component 1", message)

    def test_batch_dependent_results_are_detected(self) -> None:
        """Batch-size-dependent results must differ from one-system solves."""
        geometries = _invariant_geometries()

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            return [
                INVARIANTS.InvariantResult(
                    case_id=item.case_id,
                    molecular_charge=item.molecular_charge,
                    energy=1.0e-6 if len(items) > 1 else 0.0,
                    forces=[0.0] * (3 * len(item.atomic_numbers)),
                    charges=[0.0] * len(item.atomic_numbers),
                    point_forces=[0.0] * (3 * len(item.point_values)),
                    dipoles=[1.0e-6, 0.0, 0.0] if len(items) > 1 else [0.0, 0.0, 0.0],
                    efield=None if item.efield is None else list(item.efield),
                )
                for item in items
            ]

        failures = self.run_invariant_checks(solver, geometries)
        self.assertTrue(any("batch_vs_sequential" in failure for failure in failures))

    def test_dipole_batch_and_homogeneous_mismatches_are_detected(self) -> None:
        """Both ragged consistency gates compare the molecular dipole outlet."""
        geometry = _invariant_geometries()[0]
        sequential = _zero_invariant_result(geometry)
        mismatch = _zero_invariant_result(geometry)
        mismatch.dipoles[2] = 1.0e-6
        with redirect_stdout(io.StringIO()):
            failures: list[str] = []
            INVARIANTS.gate_batch_versus_sequential(
                [sequential], [mismatch], INVARIANTS.INVARIANT_EXACT_ATOL, failures
            )
            INVARIANTS.gate_homogeneous_replicates(
                sequential, [mismatch], INVARIANTS.INVARIANT_EXACT_ATOL, failures
            )
        self.assertTrue(
            any(
                "molecular_dipole_e_bohr batch_vs_sequential" in item
                for item in failures
            )
        )
        self.assertTrue(
            any(
                "molecular_dipole_e_bohr homogeneous_replica" in item
                for item in failures
            )
        )

    def test_sequential_baseline_uses_single_system_calls(self) -> None:
        """Each baseline result comes from its own public-style solver call."""
        geometries = _invariant_geometries()
        invocation_sizes: list[int] = []

        def solver(
            items: list[INVARIANTS.Geometry],
        ) -> list[INVARIANTS.InvariantResult]:
            invocation_sizes.append(len(items))
            return [_zero_invariant_result(item) for item in items]

        failures = self.run_invariant_checks(solver, geometries)
        self.assertEqual(failures, [])
        self.assertEqual(invocation_sizes[:3], [1, 1, 2])

    def test_homogeneous_selection_respects_focused_case_sets(self) -> None:
        """Arbitrary gas or point-charge selections retain a homogeneous gate."""
        gas, qm_point = _invariant_geometries()
        self.assertEqual(
            INVARIANTS.select_homogeneous_case_ids([gas, qm_point]),
            ("gas_pair", "qm_point"),
        )
        self.assertEqual(INVARIANTS.select_homogeneous_case_ids([gas]), ("gas_pair",))
        self.assertEqual(
            INVARIANTS.select_homogeneous_case_ids([qm_point]), ("qm_point",)
        )

    def test_displacement_helpers_shift_one_coordinate(self) -> None:
        """A displaced copy changes exactly one coordinate by the requested delta."""
        geometry = _invariant_geometries()[0]
        shifted_atom = INVARIANTS.displaced_atom(geometry, 0, 2, 0.1)
        expected = list(geometry.positions)
        expected[2] = (
            expected[2] + 0.1
        )  # atom 0 axis 2 flat index into the 2-atom/6-float array
        self.assertEqual(shifted_atom.positions, expected)
        self.assertEqual(shifted_atom.point_positions, geometry.point_positions)
        shifted_point = INVARIANTS.displaced_point(
            _invariant_geometries()[1], 0, 1, 0.25
        )
        self.assertEqual(
            shifted_point.point_positions[1],
            _invariant_geometries()[1].point_positions[1] + 0.25,
        )
        self.assertEqual(shifted_point.positions, _invariant_geometries()[1].positions)

    def test_finite_difference_gate_passes_exact_forces(self) -> None:
        """A solver whose energy gradient matches its analytic force passes."""
        geometries = _invariant_geometries()
        solver = linear_energy_solver()
        with redirect_stdout(io.StringIO()):
            baseline = solver(geometries)
            failures: list[str] = []
            INVARIANTS.gate_central_finite_difference(
                solver,
                baseline,
                geometries,
                INVARIANTS.FINITE_DIFFERENCE_STEP,
                INVARIANTS.FINITE_DIFFERENCE_FORCE_ATOL,
                INVARIANTS.FINITE_DIFFERENCE_POINT_FORCE_ATOL,
                failures,
            )
        self.assertEqual(failures, [])

    def test_finite_difference_gate_detects_wrong_force(self) -> None:
        """An analytic force that does not match dE/dx fails the gate."""
        geometries = _invariant_geometries()
        solver = linear_energy_solver()
        baseline = wrong_first_component_baseline(solver, geometries)
        with redirect_stdout(io.StringIO()):
            failures: list[str] = []
            INVARIANTS.gate_central_finite_difference(
                solver,
                baseline,
                geometries,
                INVARIANTS.FINITE_DIFFERENCE_STEP,
                INVARIANTS.FINITE_DIFFERENCE_FORCE_ATOL,
                INVARIANTS.FINITE_DIFFERENCE_POINT_FORCE_ATOL,
                failures,
            )
        self.assertTrue(failures)
        self.assertTrue(any("atom0_axis0" in failure for failure in failures))

    def test_cli_rejects_cuda_memory_for_cpu_backend(self) -> None:
        """The invariance CLI enforces the same placement rule as the golden runner."""
        for memory_mode in ("device", "mixed"):
            with self.subTest(memory_mode=memory_mode):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(INVARIANTS_TOOL),
                        "--library",
                        "/does/not/exist/libxtbloom.so",
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
                "/does/not/exist/libxtbloom.so",
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
