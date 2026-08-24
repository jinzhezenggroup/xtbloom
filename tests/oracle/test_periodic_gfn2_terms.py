"""Offline tests for the pinned periodic GFN2 term fixture corpus."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools/oracle/periodic_gfn2_terms/periodic_gfn2_terms.py"
SPEC = importlib.util.spec_from_file_location("periodic_gfn2_terms", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
terms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(terms)


class PeriodicGfn2TermFixtureTests(unittest.TestCase):
    """Exercise the complete corpus without loading tblite or xTBloom."""

    def setUp(self) -> None:
        """Load the manifest and canonical term documents."""
        self.root = terms.corpus_root()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.documents = {
            fixture["mode"]: json.loads(
                (REPOSITORY_ROOT / fixture["golden"]).read_text(encoding="utf-8")
            )
            for fixture in self.manifest["fixtures"]
        }

    def test_committed_corpus_passes_offline(self) -> None:
        """Verify hashes, recompositions, finite differences, and invariants."""
        terms.check()

    def test_corpus_covers_every_required_term_family(self) -> None:
        """Keep issue #472's six term families explicit and complete."""
        self.assertEqual(set(self.documents), set(terms.MODES))

    def test_upstream_source_digest_format_is_enforced(self) -> None:
        """Reject a source manifest whose copied SHA-256 has the wrong width."""
        changed = copy.deepcopy(terms.SOURCE_RECORDS)
        changed["tblite"]["files"]["src/tblite/disp/d4.f90"]["sha256"] += "b"
        with self.assertRaisesRegex(terms.FixtureError, "malformed SHA-256"):
            terms.check_source_record_digests(changed)

    def test_raw_outputs_are_lossless_golden_sources(self) -> None:
        """Require every normalized array to come directly from pinned raw output."""
        for fixture in self.manifest["fixtures"]:
            parsed = terms.parse_probe_output(
                (REPOSITORY_ROOT / fixture["raw"]).read_bytes()
            )
            self.assertEqual(
                parsed["arrays"], self.documents[fixture["mode"]]["arrays"]
            )

    def test_primary_charge_and_quadrupole_inputs_are_physical(self) -> None:
        """Use neutral charges and explicitly traceless packed quadrupoles."""
        data = terms.parse_input(REPOSITORY_ROOT / self.manifest["input"]["path"])
        self.assertAlmostEqual(sum(data["qat"]), 0.0, delta=1.0e-14)
        self.assertAlmostEqual(sum(data["qsh"]), 0.0, delta=1.0e-14)
        for quadrupole in data["qpat"]:
            self.assertAlmostEqual(
                quadrupole[0] + quadrupole[2] + quadrupole[5],
                0.0,
                delta=1.0e-14,
            )

    def test_shell_ewald_matrix_recomposition_is_enforced(self) -> None:
        """Reject a shell-resolved Ewald energy detached from its matrix fixture."""
        changed = copy.deepcopy(self.documents)
        changed["charge-ewald"]["arrays"]["total_energy_hartree"]["values"][0] += 1.0e-8
        with self.assertRaisesRegex(
            terms.FixtureError, "shell Ewald matrix contraction"
        ):
            terms.check_invariants(changed, self.manifest["tolerances"])

    def test_h0_matrix_symmetry_is_enforced(self) -> None:
        """Reject an AO-ordering drift that breaks the pinned H0 matrix symmetry."""
        changed = copy.deepcopy(self.documents)
        h0 = changed["integrals-h0"]["arrays"]["h0_matrix_hartree"]
        index = terms.flat_index(h0["shape"], (0, 1))
        h0["values"][index] += 1.0e-7
        with self.assertRaisesRegex(terms.FixtureError, "H0 symmetry"):
            terms.check_invariants(changed, self.manifest["tolerances"])

    def test_finite_difference_evidence_is_recomputed(self) -> None:
        """Reject stored derivative estimates not supported by plus/minus values."""
        changed = copy.deepcopy(self.documents["repulsion"])
        changed["finite_difference"][0]["samples"][0]["estimate"] += 1.0e-6
        with self.assertRaisesRegex(terms.FixtureError, "stored finite difference"):
            terms.check_finite_differences(changed, self.manifest["tolerances"])

    def test_d4_coordination_covers_cartesian_and_affine_derivatives(self) -> None:
        """Require independent D4-CN Cartesian and affine-strain evidence."""
        d4_cn_kinds = {
            evidence["kind"]
            for evidence in self.documents["d4"]["finite_difference"]
            if evidence["kind"].startswith("d4_cn_")
        }
        self.assertEqual(
            d4_cn_kinds,
            {"d4_cn_cartesian", "d4_cn_affine_strain"},
        )


if __name__ == "__main__":
    unittest.main()
