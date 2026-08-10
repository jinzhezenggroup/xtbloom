"""Integrity tests for the difficult-SCC fixture and generated diagnostics."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "conformance" / "tmacl_evidence.py"
MANIFEST = (
    REPOSITORY_ROOT
    / "data"
    / "conformance"
    / "evidence"
    / "tmacl-temperature-continuation"
    / "manifest.json"
)
SPEC = importlib.util.spec_from_file_location("xtbloom_tmacl_evidence", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


class TmaclEvidenceTest(unittest.TestCase):
    """Keep provenance and generated evidence synchronized with the harness."""

    def test_committed_manifest_pins_fixture_generator_and_evidence(self) -> None:
        """The complete committed artifact set must match reviewed SHA-256 values."""
        EVIDENCE.check_manifest(MANIFEST)

    def test_evidence_hash_drift_is_rejected(self) -> None:
        """A hand edit cannot silently turn archived diagnostics into new evidence."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["evidence_files"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(EVIDENCE.EvidenceError, "SHA-256 mismatch"):
                EVIDENCE.check_manifest(path)

    def test_distribution_boundary_drift_is_rejected(self) -> None:
        """The copied fixture's repository-only treatment stays explicit."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["distribution"]["source_distribution"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                EVIDENCE.EvidenceError, "distribution boundaries"
            ):
                EVIDENCE.check_manifest(path)

    def test_distribution_documentation_drift_is_rejected(self) -> None:
        """Human-readable packaging claims must match the canonical manifest."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "README.md"
            path.write_text(
                "This repository-only fixture is included in source distributions.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                EVIDENCE.EvidenceError, "missing required distribution text"
            ):
                EVIDENCE.require_document_text(
                    path,
                    (
                        "repository-only validation data",
                        "excluded from installation-focused PyPI source distributions",
                    ),
                    "tmacl evidence README",
                )

    def test_evidence_path_alias_is_rejected(self) -> None:
        """A digest match cannot redirect one entry away from its canonical file."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        original = manifest["evidence_files"][0]["path"]
        manifest["evidence_files"][0]["path"] = original.replace(
            "tmacl-temperature-continuation/",
            "tmacl-temperature-continuation/../tmacl-temperature-continuation/",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                EVIDENCE.EvidenceError, "canonical evidence path"
            ):
                EVIDENCE.check_manifest(path)

    def test_provider_metadata_drift_is_rejected(self) -> None:
        """Provider identity is part of the provenance contract, not commentary."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["generator"]["runtime"]["interface"] = "ILP64"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(EVIDENCE.EvidenceError, "runtime.interface"):
                EVIDENCE.check_manifest(path)

    def test_xyz_validation_rejects_nonfinite_and_trailing_rows(self) -> None:
        """Reject a non-finite, truncated, or extended scientific input."""
        fixture = (
            REPOSITORY_ROOT / "data" / "conformance" / "inputs" / "tmacl.xyz"
        ).read_text(encoding="ascii")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.xyz"
            path.write_text(fixture.replace("-5.12996550", "nan", 1), encoding="ascii")
            with self.assertRaisesRegex(EVIDENCE.EvidenceError, "non-finite"):
                EVIDENCE.validate_xyz(path)
            path.write_text(fixture + "H 0 0 0\n", encoding="ascii")
            with self.assertRaisesRegex(EVIDENCE.EvidenceError, "exact 18-row"):
                EVIDENCE.validate_xyz(path)


if __name__ == "__main__":
    unittest.main()
