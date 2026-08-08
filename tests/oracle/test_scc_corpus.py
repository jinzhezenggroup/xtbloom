"""Offline tests for the pinned gpuxtb-scc-trace-v1 restricted corpus.

These tests run without the Fortran oracle: they check that the committed
goldens validate against the versioned schema, match their manifest hashes,
compare byte-identically through the canonical writer, and diagnose an un-scalar
perturbation at the documented scalar path.  Deterministic regeneration of the
raw-stream parser uses a committed fixture derived from a real oracle run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace"
CORPUS_DIR = REPOSITORY_ROOT / "data" / "conformance" / "scc-traces"

SPEC = importlib.util.spec_from_file_location(
    "gpuxtb_scc_trace", TOOL_DIR / "gpuxtb_scc_trace.py"
)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("gpuxtb_scc_trace", TRACE)
SPEC.loader.exec_module(TRACE)

SPEC2 = importlib.util.spec_from_file_location(
    "generate_scc_corpus", TOOL_DIR / "generate_scc_corpus.py"
)
assert SPEC2 is not None and SPEC2.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC2)
sys.modules.setdefault("generate_scc_corpus", GENERATOR)
SPEC2.loader.exec_module(GENERATOR)

SPEC3 = importlib.util.spec_from_file_location(
    "gpuxtb_scc_cpu_trace", TOOL_DIR / "gpuxtb_scc_cpu_trace.py"
)
assert SPEC3 is not None and SPEC3.loader is not None
CPU_TRACE = importlib.util.module_from_spec(SPEC3)
sys.modules.setdefault("gpuxtb_scc_cpu_trace", CPU_TRACE)
SPEC3.loader.exec_module(CPU_TRACE)

CASES = sorted(GENERATOR.CASES)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RestrictedCorpusTest(unittest.TestCase):
    """Offline validation of the committed restricted SCC trace corpus."""

    def test_corpus_files_and_manifest_are_present(self) -> None:
        """Every case must have a manifest entry and an existing golden file."""
        manifest_path = CORPUS_DIR / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "corpus manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case_id in CASES:
            entry = manifest["cases"].get(case_id)
            self.assertIsNotNone(entry, f"manifest misses {case_id}")
            golden = CORPUS_DIR / entry["path"]
            self.assertTrue(golden.is_file(), f"golden {entry['path']} is missing")

    def test_every_golden_validates_against_the_v1_schema(self) -> None:
        """Each golden document must satisfy the runtime v1 schema validator."""
        for case_id in CASES:
            golden = CORPUS_DIR / f"{case_id}.json"
            with self.subTest(case=case_id):
                document = json.loads(golden.read_text(encoding="utf-8"))
                TRACE.validate(document)

    def test_every_golden_hash_matches_the_manifest(self) -> None:
        """Each golden's SHA-256 must match the pinned manifest entry."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        for case_id in CASES:
            entry = manifest["cases"][case_id]
            with self.subTest(case=case_id):
                digest = sha256_file(CORPUS_DIR / entry["path"])
                self.assertEqual(digest, entry["sha256"])

    def test_canonical_rewrite_is_byte_identical(self) -> None:
        """Canonical dumps of each golden must reproduce its exact bytes."""
        for case_id in CASES:
            golden = CORPUS_DIR / f"{case_id}.json"
            with self.subTest(case=case_id):
                original = golden.read_bytes()
                document = json.loads(original.decode("utf-8"))
                canonical = TRACE.dumps(document)
                self.assertEqual(canonical.encode("utf-8"), original)

    def test_manifest_is_fully_pinned(self) -> None:
        """Provenance, revision, digests, and dependency pins must be complete."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["format"], "gpuxtb-scc-trace-v1")
        self.assertEqual(
            manifest["revision"], "e9abc395b122018ed688aecb1c3a65cecaf97beb"
        )
        self.assertEqual(
            manifest["oracle_patch_sha256"],
            sha256_file(TOOL_DIR / "tblite-e9abc395-scc-observer.patch"),
        )
        self.assertEqual(
            manifest["oracle_sources"],
            {
                "scc_trace_main.f90": sha256_file(TOOL_DIR / "scc_trace_main.f90"),
                "scc_trace_recorder.f90": sha256_file(
                    TOOL_DIR / "scc_trace_recorder.f90"
                ),
            },
        )
        self.assertTrue(manifest["command"], "manifest command must not be empty")
        self.assertEqual(manifest["dependencies"], GENERATOR.PINNED_DEPENDENCIES)
        self.assertEqual(manifest["environment"], GENERATOR.DETERMINISTIC_ENVIRONMENT)
        for dependency, commit in manifest["dependencies"].items():
            self.assertEqual(
                len(commit), 40, f"dependency {dependency} is not a full commit"
            )
        self.assertTrue(manifest["toolchain"]["compiler"]["executables"])
        self.assertTrue(manifest["toolchain"]["blas_lapack"]["libraries"])

    def test_point_charge_cases_match_canonical_qmmm_inputs(self) -> None:
        """Duplicated recorder literals must equal the canonical QMMM inputs."""
        inputs = REPOSITORY_ROOT / "data" / "conformance" / "inputs"
        for case_id in ("water_one_pc_gamma999", "water_dimer_6pc_hardness"):
            with self.subTest(case=case_id):
                source = json.loads(
                    (inputs / f"{case_id}.qmmm.json").read_text(encoding="utf-8")
                )
                spec = GENERATOR.CASES[case_id]
                self.assertEqual(spec["atomic_numbers"], source["qm"]["atomic_numbers"])
                self.assertEqual(spec["positions"], source["qm"]["positions_bohr"])
                self.assertEqual(
                    spec["molecular_charge"], source["qm"]["molecular_charge"]
                )
                self.assertEqual(
                    spec["unpaired_electrons"], source["qm"]["unpaired_electrons"]
                )
                point_charges = spec["point_charges"]
                source_charges = source["external_point_charges"]
                self.assertEqual(
                    point_charges["positions"], source_charges["positions_bohr"]
                )
                self.assertEqual(point_charges["charges"], source_charges["charges_e"])
                self.assertEqual(
                    point_charges["gammas"], source_charges["gammas_hartree"]
                )

    def test_point_charge_shell_potentials_follow_gfn2_parameters(self) -> None:
        """Independently reconstruct Vpc from canonical GFN2 shell hardnesses."""
        parameters = json.loads(
            (REPOSITORY_ROOT / "data" / "parameters" / "gfn2.json").read_text(
                encoding="utf-8"
            )
        )
        elements = {
            element["atomic_number"]: element for element in parameters["elements"]
        }
        for case_id in ("water_one_pc_gamma999", "water_dimer_6pc_hardness"):
            with self.subTest(case=case_id):
                trace = json.loads(
                    (CORPUS_DIR / f"{case_id}.json").read_text(encoding="utf-8")
                )
                molecule = trace["input"]
                positions = [
                    molecule["positions"][index : index + 3]
                    for index in range(0, len(molecule["positions"]), 3)
                ]
                point_charges = molecule["point_charges"]
                pc_positions = [
                    point_charges["positions"][index : index + 3]
                    for index in range(0, len(point_charges["positions"]), 3)
                ]
                expected = []
                for atom, atomic_number in enumerate(molecule["atomic_numbers"]):
                    element = elements[atomic_number]
                    self.assertEqual(
                        len(element["shells"]),
                        trace["basis"]["atom_to_shell_count"][atom],
                    )
                    for shell in element["shells"]:
                        shell_gamma = element["gam"] * shell["shell_hubbard_scale"]
                        potential = 0.0
                        for pc_position, charge, pc_gamma in zip(
                            pc_positions,
                            point_charges["charges"],
                            point_charges["hardnesses"],
                            strict=True,
                        ):
                            distance_squared = sum(
                                (positions[atom][axis] - pc_position[axis]) ** 2
                                for axis in range(3)
                            )
                            screening = 2.0 / (shell_gamma + pc_gamma)
                            potential += charge / math.sqrt(
                                distance_squared + screening * screening
                            )
                        expected.append(potential)
                for iteration in trace["iterations"]:
                    observed = iteration["point_charge_shell_potential"][0]
                    self.assertEqual(len(observed), len(expected))
                    for shell, (actual, reference) in enumerate(
                        zip(observed, expected, strict=True)
                    ):
                        self.assertAlmostEqual(
                            actual,
                            reference,
                            delta=5.0e-15,
                            msg=f"{case_id} shell {shell}",
                        )

    def test_point_charge_cases_record_pcem_primary_fields(self) -> None:
        """QM/MM-like goldens must record the per-iteration PCEM primary fields."""
        for case_id in ("water_one_pc_gamma999", "water_dimer_6pc_hardness"):
            document = json.loads(
                (CORPUS_DIR / f"{case_id}.json").read_text(encoding="utf-8")
            )
            self.assertIn("point_charges", document["input"])
            self.assertEqual(
                len(document["input"]["point_charges"]["positions"]) % 3,
                0,
                "point-charge positions must be per-point-major",
            )
            self.assertTrue(document["iterations"])
            for entry in document["iterations"]:
                self.assertIn("point_charge_shell_potential", entry)
                self.assertIn("point_charge_energy", entry)
                self.assertEqual(
                    len(entry["point_charge_shell_potential"][0]),
                    document["basis"]["n_shells"],
                )
                self.assertEqual(
                    len(entry["point_charge_energy"][0]), document["basis"]["n_shells"]
                )

    def test_plain_cases_do_not_record_pcem_fields(self) -> None:
        """Non-point-charge goldens must omit the optional PCEM fields."""
        for case_id in ("h3_plus", "ketene", "nenacl"):
            document = json.loads(
                (CORPUS_DIR / f"{case_id}.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("point_charges", document["input"])
            for entry in document["iterations"]:
                self.assertNotIn("point_charge_shell_potential", entry)
                self.assertNotIn("point_charge_energy", entry)

    def test_corpus_check_command_passes(self) -> None:
        """The generator --check path must verify the committed corpus offline."""
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL_DIR / "generate_scc_corpus.py"),
                "--source-root",
                "/nonexistent",
                "--corpus-dir",
                str(CORPUS_DIR),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # --check only needs the corpus directory and the module metadata; it
        # must not require a source checkout or network.
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_corpus_check_rejects_stale_or_incomplete_provenance(self) -> None:
        """Offline checking must reject repository and toolchain provenance drift."""
        original = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        mutations = {
            "recorder": lambda manifest: manifest["oracle_sources"].__setitem__(
                "scc_trace_recorder.f90", "0" * 64
            ),
            "main": lambda manifest: manifest["oracle_sources"].pop(
                "scc_trace_main.f90"
            ),
            "dependency": lambda manifest: manifest["dependencies"].__setitem__(
                "dftd4", "0" * 40
            ),
            "provider": lambda manifest: manifest["toolchain"][
                "blas_lapack"
            ].__setitem__("libraries", []),
            "requested_provider_type": lambda manifest: manifest["toolchain"][
                "blas_lapack"
            ].__setitem__("requested", []),
            "resolved_provider_type": lambda manifest: manifest["toolchain"][
                "blas_lapack"
            ].__setitem__("resolved_provider", []),
            "path": lambda manifest: manifest["cases"]["h3_plus"].__setitem__(
                "path", "../h3_plus.json"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(field=label), tempfile.TemporaryDirectory() as directory:
                corpus = Path(directory) / "corpus"
                shutil.copytree(CORPUS_DIR, corpus)
                manifest = copy.deepcopy(original)
                mutate(manifest)
                (corpus / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(TOOL_DIR / "generate_scc_corpus.py"),
                        "--source-root",
                        "/nonexistent",
                        "--corpus-dir",
                        str(corpus),
                        "--check",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)

        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            shutil.copytree(CORPUS_DIR, corpus)
            (corpus / "manifest.json").write_bytes(b"\xff")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_DIR / "generate_scc_corpus.py"),
                    "--source-root",
                    "/nonexistent",
                    "--corpus-dir",
                    str(corpus),
                    "--check",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_dependency_binding_checks_out_the_reviewed_commit(self) -> None:
        """Binding must use the pin even when the local dependency HEAD moved."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "subprojects" / "sample"
            source.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "config", "user.name", "Corpus Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "config",
                    "user.email",
                    "corpus@example.invalid",
                ],
                check=True,
            )
            payload = source / "payload.txt"
            payload.write_text("pinned\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(source), "add", "payload.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(source), "commit", "-q", "-m", "pinned"],
                check=True,
            )
            pinned = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            payload.write_text("moved\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(source), "commit", "-qam", "moved"], check=True
            )

            outer = root / "outer"
            (outer / "subprojects").mkdir(parents=True)
            GENERATOR.clone_pinned_dependencies(
                outer, root / "source", {"sample": pinned}
            )
            bound = outer / "subprojects" / "sample"
            head = subprocess.run(
                ["git", "-C", str(bound), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            self.assertEqual(head, pinned)
            self.assertEqual(
                (bound / "payload.txt").read_text(encoding="utf-8"), "pinned\n"
            )

    def test_cpu_wrapper_passes_the_manifest_hash_to_comparator(self) -> None:
        """The evidence wrapper must reject a canonical golden with drifted bytes."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        document = json.loads((CORPUS_DIR / "h3_plus.json").read_text(encoding="utf-8"))
        document["iterations"][-1]["energy"] += 1.0e-3
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / "h3_plus.json"
            drifted.write_text(TRACE.dumps(document), encoding="utf-8")
            return_code, report = CPU_TRACE.compare_with_comparator(
                drifted,
                drifted,
                "cpu_closed_loop_v1",
                manifest["cases"]["h3_plus"]["sha256"],
            )
        self.assertEqual(return_code, 2, report)
        self.assertIn("SHA-256 mismatch", report)

    def test_comparator_self_compare_passes_and_scalar_mismatch_is_localized(
        self,
    ) -> None:
        """Self-comparison passes and a scalar perturbation is localized."""
        compare = TOOL_DIR / "gpuxtb_scc_compare.py"
        golden = CORPUS_DIR / "h3_plus.json"
        self_compare = subprocess.run(
            [
                sys.executable,
                str(compare),
                "trace",
                str(golden),
                str(golden),
                "--profile",
                "cpu_closed_loop_v1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            self_compare.returncode, 0, self_compare.stdout + self_compare.stderr
        )
        document = json.loads(golden.read_text(encoding="utf-8"))
        document["iterations"][-1]["energy"] += 1.0e-3
        perturbed_path = REPOSITORY_ROOT / "build" / "scc_trace_perturbed.json"
        perturbed_path.parent.mkdir(parents=True, exist_ok=True)
        perturbed_path.write_text(TRACE.dumps(document), encoding="utf-8")
        try:
            mismatch = subprocess.run(
                [
                    sys.executable,
                    str(compare),
                    "trace",
                    str(perturbed_path),
                    str(golden),
                    "--profile",
                    "cpu_closed_loop_v1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("iterations[2].energy", mismatch.stdout)
        finally:
            perturbed_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
