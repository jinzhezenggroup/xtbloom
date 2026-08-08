"""Offline tests for the pinned gpuxtb-scc-trace-v1 restricted corpus.

These tests run without the Fortran oracle: they check that the committed
goldens validate against the versioned schema, match their manifest hashes,
compare byte-identically through the canonical writer, and diagnose an un-scalar
perturbation at the documented scalar path.  Deterministic regeneration of the
raw-stream parser uses a committed fixture derived from a real oracle run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
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
        self.assertEqual(len(manifest["oracle_patch_sha256"]), 64)
        self.assertEqual(len(manifest["recorder_sha256"]), 64)
        self.assertTrue(manifest["command"], "manifest command must not be empty")
        self.assertIn("dependencies", manifest)
        for dependency, commit in manifest["dependencies"].items():
            self.assertEqual(
                len(commit), 40, f"dependency {dependency} is not a full commit"
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
