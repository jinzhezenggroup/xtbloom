"""Offline tests for the pinned xtbloom-scc-trace-v1 restricted corpus.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import jsonschema
except ImportError:  # pragma: no cover - runtime checks remain standard-library only.
    jsonschema = None

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace"
CORPUS_DIR = REPOSITORY_ROOT / "data" / "conformance" / "scc-traces"

SPEC = importlib.util.spec_from_file_location(
    "xtbloom_scc_trace", TOOL_DIR / "xtbloom_scc_trace.py"
)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("xtbloom_scc_trace", TRACE)
SPEC.loader.exec_module(TRACE)

SPEC2 = importlib.util.spec_from_file_location(
    "generate_scc_corpus", TOOL_DIR / "generate_scc_corpus.py"
)
assert SPEC2 is not None and SPEC2.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC2)
sys.modules.setdefault("generate_scc_corpus", GENERATOR)
SPEC2.loader.exec_module(GENERATOR)

SPEC3 = importlib.util.spec_from_file_location(
    "xtbloom_scc_cpu_trace", TOOL_DIR / "xtbloom_scc_cpu_trace.py"
)
assert SPEC3 is not None and SPEC3.loader is not None
CPU_TRACE = importlib.util.module_from_spec(SPEC3)
sys.modules.setdefault("xtbloom_scc_cpu_trace", CPU_TRACE)
SPEC3.loader.exec_module(CPU_TRACE)

SPEC4 = importlib.util.spec_from_file_location(
    "xtbloom_scc_compare", TOOL_DIR / "xtbloom_scc_compare.py"
)
assert SPEC4 is not None and SPEC4.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC4)
sys.modules.setdefault("xtbloom_scc_compare", COMPARE)
SPEC4.loader.exec_module(COMPARE)

SPEC5 = importlib.util.spec_from_file_location(
    "generate_unrestricted_scc_corpus",
    TOOL_DIR / "generate_unrestricted_scc_corpus.py",
)
assert SPEC5 is not None and SPEC5.loader is not None
UNRESTRICTED_GENERATOR = importlib.util.module_from_spec(SPEC5)
sys.modules.setdefault("generate_unrestricted_scc_corpus", UNRESTRICTED_GENERATOR)
SPEC5.loader.exec_module(UNRESTRICTED_GENERATOR)


def cpu_trace_safe_compare(actual: dict, golden: dict) -> object:
    """Compare two in-memory trace documents with the shared comparator."""
    return COMPARE.compare_trace(actual, golden)


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

    def test_v1_corpus_bytes_remain_frozen(self) -> None:
        """Issue #51 must not rewrite or reinterpret any restricted evidence."""
        expected = {
            "h3_plus.json": (
                "31a3b1c3c6aa90a9fae1870d08fcfc2e88ecac1e8eb6c4e0f108bfed984b7af2"
            ),
            "ketene.json": (
                "bc591e1b6d865e6d8944fbcd278d3407e2883aba803edf268e5c32f4cb5ad639"
            ),
            "nenacl.json": (
                "e9de43561798ff4ba054da9ae1e4959b8adea59b2e2c52d7388c14ef920d311a"
            ),
            "water_dimer_6pc_hardness.json": (
                "4f321b3a12992e8eb8498c4c9b0524815f6691b1e1bdafb4cf95cc68d37c4de5"
            ),
            "water_one_pc_gamma999.json": (
                "dafec557110075ec45acb752fccf916a24d8c5537193866980888da8344e8c1a"
            ),
            "manifest.json": (
                "7d84420c8304a68ca629fb005acb8a5cb4385d6968515ddcdbd9f7f00dba4fad"
            ),
        }
        self.assertEqual(
            {name: sha256_file(CORPUS_DIR / name) for name in expected}, expected
        )

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
        self.assertEqual(manifest["format"], "xtbloom-scc-trace-v1")
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
        compare = TOOL_DIR / "xtbloom_scc_compare.py"
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

    def test_cpu_capture_provenance_command_does_not_fail_golden_compare(
        self,
    ) -> None:
        """A CPU capture (different oracle_command) compares against the golden."""
        golden = json.loads((CORPUS_DIR / "h3_plus.json").read_text(encoding="utf-8"))
        actual = copy.deepcopy(golden)
        actual["provenance"]["oracle_command"] = "xtbloom_scc_cpu_trace.py (capture)"
        result = cpu_trace_safe_compare(actual, golden)
        self.assertTrue(result.matches, msg=result.render())

    def test_batch_lane_splitting_parses_status_and_bodies(self) -> None:
        """Split a synthetic batch raw stream into lanes and summaries."""
        synthetic = (
            "batch_system 0\n"
            "status 0 iterations 3\n"
            "nat 3 nsh 3 nao 3 niterations 3 terminal 1\n"
            "atomic_numbers\n1\n1\n1\n"
            "batch_system 1\n"
            "status -1 iterations 0\n"
            "nat 3 nsh 3 nao 3 niterations 0 terminal 3\n"
        )
        lanes = CPU_TRACE.split_batch_lanes(synthetic)
        self.assertEqual(len(lanes), 2)
        self.assertEqual(lanes[0][0], 0)
        self.assertEqual(CPU_TRACE.status_bits(lanes[0][1]), (0, 3))
        self.assertEqual(CPU_TRACE.status_bits(lanes[1][1]), (-1, 0))
        self.assertIn("niterations 0 terminal 3", lanes[1][2])

    def test_cpu_raw_parser_preserves_eigensolver_failed_attempt(self) -> None:
        """Canonicalize only the pre-solve payload of a failed eigensolve."""
        raw = """nat 1 nsh 1 nao 1 niterations 0 terminal 3 failed_attempt 1
atomic_numbers
1
positions
0
0
0
molecular_charge
0
unpaired_electrons
0
temperature
300
n_point_charges
0
atom_to_shell_count
1
overlap
1
core_hamiltonian
-0.5
failed_attempt
1
hamiltonian
-0.4
mixed_qsh
0
mixed_qat
0
mixed_dipoles
0
0
0
mixed_quadrupoles
0
0
0
0
0
0
"""
        trace = GENERATOR.canonicalize(raw, {}, "synthetic eigensolver failure")
        TRACE.validate(trace)
        self.assertEqual(trace["iterations"], [])
        self.assertEqual(
            trace["terminal"], {"status": 3, "converged": False, "iterations": 1}
        )
        attempt = trace["failed_attempt"]
        self.assertEqual(attempt["index"], 1)
        self.assertEqual(attempt["hamiltonian"], [[[-0.4]]])
        for forbidden in (
            "eigenvalues",
            "occupations",
            "density",
            "raw_qsh",
            "residual",
            "energy",
            "convergence",
        ):
            self.assertNotIn(forbidden, attempt)

    def test_replay_lifecycle_metadata_is_checked_before_snapshot_compare(self) -> None:
        """Do not discard a replay artifact's terminal lifecycle metadata."""
        expected_iteration = {"convergence": {"overall": False}}
        trace = {"terminal": {"status": 2, "converged": False, "iterations": 1}}
        CPU_TRACE.validate_replay_lifecycle(trace, expected_iteration)
        trace["terminal"] = {"status": 1, "converged": True, "iterations": 1}
        with self.assertRaisesRegex(GENERATOR.CorpusError, "lifecycle mismatch"):
            CPU_TRACE.validate_replay_lifecycle(trace, expected_iteration)

    def test_mixer_flatten_helpers_match_residual_layout(self) -> None:
        """Flatten mixed/raw multipoles in the canonical residual order."""
        golden = json.loads((CORPUS_DIR / "h3_plus.json").read_text(encoding="utf-8"))
        iteration = golden["iterations"][1]
        flat = CPU_TRACE.flatten_multipoles(iteration)
        expected = (
            iteration["mixed_qsh"][0]
            + [value for atom in iteration["mixed_dipoles"][0] for value in atom]
            + [value for atom in iteration["mixed_quadrupoles"][0] for value in atom]
        )
        self.assertEqual(flat, expected)
        self.assertEqual(
            len(flat), golden["basis"]["n_shells"] + 9 * golden["basis"]["n_atoms"]
        )
        raw = CPU_TRACE.flatten_raw(iteration)
        self.assertEqual(len(raw), len(flat))

    def test_mixer_wrapper_requires_mixer_cases(self) -> None:
        """Reject the mixer mode without a case list."""
        wrapper = TOOL_DIR / "xtbloom_scc_cpu_trace.py"
        result = subprocess.run(
            [sys.executable, str(wrapper), "--mixer", "mixer"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--mixer requires --mixer-cases", result.stdout)

    def test_load_pinned_golden_accepts_manifest_matching_canonical_bytes(self) -> None:
        """Evidence goldens must match the manifest hash and canonical bytes."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        entry = manifest["cases"]["h3_plus"]
        golden_path = CORPUS_DIR / entry["path"]
        document = CPU_TRACE.load_pinned_golden(golden_path, entry)
        self.assertEqual(document["format"], "xtbloom-scc-trace-v1")

    def test_load_pinned_golden_rejects_drifted_bytes(self) -> None:
        """A drifted golden must be rejected before replay evidence is built."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        entry = dict(manifest["cases"]["h3_plus"])
        original = CORPUS_DIR / entry["path"]
        with tempfile.TemporaryDirectory() as directory:
            drifted = Path(directory) / "h3_plus.json"
            document = json.loads(original.read_text(encoding="utf-8"))
            document["iterations"][-1]["energy"] += 1.0e-3
            drifted.write_text(TRACE.dumps(document), encoding="utf-8")
            with self.assertRaises(CPU_TRACE.generator.CorpusError):
                CPU_TRACE.load_pinned_golden(drifted, entry)

    def test_load_pinned_golden_rejects_noncanonical_bytes(self) -> None:
        """Noncanonical serialization must be rejected for evidence goldens."""
        manifest = json.loads(
            (CORPUS_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        entry = dict(manifest["cases"]["h3_plus"])
        with tempfile.TemporaryDirectory() as directory:
            noncanonical = Path(directory) / "h3_plus.json"
            document = json.loads(
                (CORPUS_DIR / entry["path"]).read_text(encoding="utf-8")
            )
            compressed = json.dumps(document, indent=2, sort_keys=True)
            noncanonical.write_text(compressed + "  \n", encoding="utf-8")
            # Align the manifest pin with the changed bytes so only the
            # canonical-serialization check can reject this file.
            entry["sha256"] = CPU_TRACE.sha256_file(noncanonical)
            with self.assertRaisesRegex(
                CPU_TRACE.generator.CorpusError, "not canonical"
            ):
                CPU_TRACE.load_pinned_golden(noncanonical, entry)

    def test_wrapper_requires_exactly_one_mode(self) -> None:
        """Reject wrappers that select zero or multiple capture modes."""
        wrapper = TOOL_DIR / "xtbloom_scc_cpu_trace.py"
        without_mode = subprocess.run(
            [sys.executable, str(wrapper)], capture_output=True, text=True, check=False
        )
        self.assertNotEqual(without_mode.returncode, 0)
        self.assertIn("exactly one of", without_mode.stdout)
        with_mode = subprocess.run(
            [
                sys.executable,
                str(wrapper),
                "--capture",
                "capture",
                "--batch-capture",
                "batch",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(with_mode.returncode, 0)
        self.assertIn("exactly one of", with_mode.stdout)


class UnrestrictedCorpusTest(unittest.TestCase):
    """Independent offline validation of the pinned v2 OH radical evidence."""

    def setUp(self) -> None:
        """Load the committed manifest entry and canonical OH trace."""
        self.manifest = json.loads(
            (CORPUS_DIR / "manifest-v2.json").read_text(encoding="utf-8")
        )
        self.entry = self.manifest["cases"]["oh_radical"]
        self.trace = json.loads(
            (CORPUS_DIR / self.entry["path"]).read_text(encoding="utf-8")
        )

    def test_v2_manifest_schema_hashes_and_canonical_bytes(self) -> None:
        """Pin the complete v2 source bundle and committed evidence bytes."""
        self.assertEqual(self.manifest["format"], TRACE.FORMAT_V2)
        self.assertEqual(self.manifest["revision"], GENERATOR.REVISION)
        self.assertEqual(
            self.manifest["oracle_patch_sha256"],
            sha256_file(TOOL_DIR / "tblite-e9abc395-scc-observer-v2.patch"),
        )
        self.assertEqual(
            self.manifest["oracle_sources"],
            {
                "scc_trace_main_v2.f90": sha256_file(
                    TOOL_DIR / "scc_trace_main_v2.f90"
                ),
                "scc_trace_recorder_v2.f90": sha256_file(
                    TOOL_DIR / "scc_trace_recorder_v2.f90"
                ),
            },
        )
        golden_path = CORPUS_DIR / self.entry["path"]
        self.assertEqual(
            self.entry["sha256"],
            "18526bd79fb2c598af93a2bc1385344ef0bc83eaac6d966c585ed814838bf1dd",
        )
        self.assertEqual(
            self.entry["spec_sha256"],
            "1d65d28476a281e3e03c9480e3210f3b030f0c851f3392d15b50cc551b10698b",
        )
        self.assertEqual(sha256_file(golden_path), self.entry["sha256"])
        self.assertEqual(TRACE.dumps(self.trace).encode(), golden_path.read_bytes())
        TRACE.validate(self.trace)

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_committed_oh_golden_validates_against_draft7_schema(self) -> None:
        """Validate the real pinned evidence, not only a synthetic v2 fixture."""
        schema = json.loads(
            (TOOL_DIR / "xtbloom-scc-trace-v2.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft7Validator.check_schema(schema)
        jsonschema.Draft7Validator(schema).validate(self.trace)

    def test_v2_offline_check_passes(self) -> None:
        """Verify the committed v2 manifest without rebuilding tblite."""
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL_DIR / "generate_unrestricted_scc_corpus.py"),
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
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def _run_mutated_manifest(
        self, mutation: Callable[[dict[str, object]], object]
    ) -> subprocess.CompletedProcess[str]:
        """Run the public offline checker against one isolated manifest drift."""
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "scc-traces"
            shutil.copytree(CORPUS_DIR, corpus)
            manifest_path = corpus / "manifest-v2.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutation(manifest)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    str(TOOL_DIR / "generate_unrestricted_scc_corpus.py"),
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

    def test_v2_manifest_rejects_provenance_mutations(self) -> None:
        """Reject stale, missing, extra, and malformed v2 provenance records."""
        mutations = {
            "source hash": lambda value: value["oracle_sources"].__setitem__(
                "scc_trace_recorder_v2.f90", "0" * 64
            ),
            "source set": lambda value: value["oracle_sources"].__setitem__(
                "extra.f90", "0" * 64
            ),
            "dependency": lambda value: value["dependencies"].__setitem__(
                next(iter(value["dependencies"])), "0" * 40
            ),
            "compiler hash": lambda value: value["toolchain"]["compiler"][
                "executables"
            ][0].__setitem__("sha256", "not-a-digest"),
            "compiler path": lambda value: value["toolchain"]["compiler"][
                "executables"
            ][0].__setitem__("path", "relative/gfortran"),
            "Meson": lambda value: value["toolchain"]["meson"].__setitem__(
                "version", ""
            ),
            "provider": lambda value: value["toolchain"]["blas_lapack"].__setitem__(
                "resolved_provider", "ilp64"
            ),
            "library hash": lambda value: value["toolchain"]["blas_lapack"][
                "libraries"
            ][0].__setitem__("sha256", "0" * 63),
            "library SONAME": lambda value: value["toolchain"]["blas_lapack"][
                "libraries"
            ][0].__setitem__("soname", ""),
            "case path": lambda value: value["cases"]["oh_radical"].__setitem__(
                "path", "../oh_radical.json"
            ),
            "case hash": lambda value: value["cases"]["oh_radical"].__setitem__(
                "sha256", "0" * 64
            ),
            "spec hash": lambda value: value["cases"]["oh_radical"].__setitem__(
                "spec_sha256", "0" * 64
            ),
            "command": lambda value: value.__setitem__("command", ""),
            "extra field": lambda value: value.__setitem__("unexpected", True),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                result = self._run_mutated_manifest(mutation)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(
                    "ERROR: invalid unrestricted corpus manifest", result.stderr
                )
                self.assertNotIn("Traceback", result.stderr)

    def test_oh_geometry_matches_the_canonical_conformance_input(self) -> None:
        """Bind the OH trace geometry and elements to the public corpus input."""
        coord = (
            REPOSITORY_ROOT / "data" / "conformance" / "inputs" / "oh_radical.coord"
        ).read_text(encoding="utf-8")
        rows = [line.split() for line in coord.splitlines() if len(line.split()) == 4]
        positions = [[float(value) for value in row[:3]] for row in rows]
        elements = [row[3].lower() for row in rows]
        spec = UNRESTRICTED_GENERATOR.CASES["oh_radical"]
        self.assertEqual(elements, ["o", "h"])
        self.assertEqual(spec["atomic_numbers"], [8, 1])
        self.assertEqual(positions, spec["positions"])
        self.assertEqual(
            self.trace["input"]["positions"],
            [value for atom in positions for value in atom],
        )
        self.assertEqual(self.trace["input"]["unpaired_electrons"], 1)
        self.assertEqual(self.trace["input"]["spin_channels"], 2)

    def test_first_iteration_contains_the_independent_spin_energy(self) -> None:
        """Guard against a two-channel recorder which omits spin polarization."""
        first = self.trace["iterations"][0]
        magnetization = first["raw_qsh"][1]
        # O owns the first two shells (s,p); H owns the final s shell.  These
        # pinned tblite constants are independently retained in tblite_spin.hpp.
        w_ss, w_sp, w_pp = -0.035075, -0.029538, -0.027850
        spin_energy = 0.5 * (
            magnetization[0] * (w_ss * magnetization[0] + w_sp * magnetization[1])
            + magnetization[1] * (w_sp * magnetization[0] + w_pp * magnetization[1])
        )
        # The first-step magnetization is entirely the O p-shell unpaired
        # electron, so the independently reconstructable spin contribution is
        # exactly one half of the pinned O p-p constant.
        self.assertAlmostEqual(spin_energy, 0.5 * w_pp, delta=2.0e-15)
        self.assertNotEqual(spin_energy, 0.0)
        second = self.trace["iterations"][1]
        self.assertGreater(
            max(
                abs(alpha - beta)
                for alpha_row, beta_row in zip(
                    second["solver_hamiltonian"][0],
                    second["solver_hamiltonian"][1],
                    strict=True,
                )
                for alpha, beta in zip(alpha_row, beta_row, strict=True)
            ),
            1.0e-3,
        )

    def test_cpu_helpers_preserve_both_population_channels(self) -> None:
        """Keep charge and magnetization channels through replay helpers."""
        iteration = self.trace["iterations"][1]
        expected = self.trace["input"]["spin_channels"] * (
            self.trace["basis"]["n_shells"] + 9 * self.trace["basis"]["n_atoms"]
        )
        self.assertEqual(len(CPU_TRACE.flatten_multipoles(iteration)), expected)
        self.assertEqual(len(CPU_TRACE.flatten_raw(iteration)), expected)
        catalog = CPU_TRACE.load_golden_catalog(CORPUS_DIR)
        self.assertIn("h3_plus", catalog)
        self.assertIn("oh_radical", catalog)

    def test_v2_raw_parser_preserves_unrestricted_eigensolver_failed_attempt(
        self,
    ) -> None:
        """Canonicalize the complete spin-resolved pre-solve failure payload."""
        raw = """nat 1 nsh 1 nao 1 niterations 0 terminal 3 failed_attempt 1
atomic_numbers
8
positions
0
0
0
molecular_charge
0
unpaired_electrons
1
spin_channels
2
temperature
300
n_point_charges
0
atom_to_shell_count
1
overlap
1
core_hamiltonian
-0.5
failed_attempt
1
assembled_hamiltonian
-0.2
-0.1
solver_hamiltonian
-0.4
-0.2
mixed_qsh
0.25
0.5
mixed_qat
0.25
0.5
mixed_dipoles
0
0
0
0
0
0
mixed_quadrupoles
0
0
0
0
0
0
0
0
0
0
0
0
"""
        trace = UNRESTRICTED_GENERATOR.canonicalize(
            raw, {}, "synthetic unrestricted eigensolver failure"
        )
        TRACE.validate(trace)
        self.assertEqual(trace["iterations"], [])
        self.assertEqual(
            trace["terminal"], {"status": 3, "converged": False, "iterations": 1}
        )
        attempt = trace["failed_attempt"]
        self.assertEqual(attempt["index"], 1)
        self.assertEqual(attempt["assembled_hamiltonian"], [[[-0.2]], [[-0.1]]])
        self.assertEqual(attempt["solver_hamiltonian"], [[[-0.4]], [[-0.2]]])
        self.assertEqual(attempt["mixed_qsh"], [[0.25], [0.5]])
        self.assertEqual(attempt["mixed_qat"], [[0.25], [0.5]])
        for forbidden in (
            "eigenvalues",
            "occupations",
            "density",
            "raw_qsh",
            "residual",
            "energy",
            "convergence",
        ):
            self.assertNotIn(forbidden, attempt)


if __name__ == "__main__":
    unittest.main()
