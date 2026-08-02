"""Offline integrity tests for the pinned tblite SCC observer bundle."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace"
VALIDATOR_PATH = TOOL_DIR / "validate_observer_patch.py"
SPEC = importlib.util.spec_from_file_location("validate_observer_patch", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class TbliteObserverPatchTest(unittest.TestCase):
    """Keep the patch, metadata, documentation, and probe synchronized."""

    def setUp(self) -> None:
        self.metadata = VALIDATOR.load_metadata()
        patch = self.metadata["patch"]
        assert isinstance(patch, dict)
        self.patch_path = TOOL_DIR / str(patch["file"])
        self.patch_text = self.patch_path.read_text(encoding="utf-8")
        self.probe_text = (TOOL_DIR / "observer_probe.f90").read_text(encoding="utf-8")
        self.readme_text = (TOOL_DIR / "README.md").read_text(encoding="utf-8")

    def test_patch_hash_and_pinned_revision_are_consistent(self) -> None:
        """The reviewed patch bytes and full upstream commit are immutable."""
        digest = VALIDATOR.validate_bundle(self.metadata)
        upstream = self.metadata["upstream"]
        patch = self.metadata["patch"]
        assert isinstance(upstream, dict) and isinstance(patch, dict)
        self.assertEqual(
            upstream["revision"], "e9abc395b122018ed688aecb1c3a65cecaf97beb"
        )
        self.assertEqual(digest, patch["sha256"])
        self.assertIn(digest, self.readme_text)
        self.assertRegex(str(upstream["revision"]), r"^[0-9a-f]{40}$")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        for entry in self.metadata["license_files"]:
            license_path = TOOL_DIR / entry["file"]
            self.assertEqual(VALIDATOR.sha256_file(license_path), entry["sha256"])
            self.assertIn(entry["file"], self.readme_text)

    def test_patch_defines_read_only_no_op_hooks_and_terminal_status(self) -> None:
        """The seam stays concrete, borrowed, and separate from failed payloads."""
        self.assertIn("type, public :: scf_observer", self.patch_text)
        self.assertNotIn("abstract :: scf_observer", self.patch_text)
        self.assertNotIn("deferred", self.patch_text)
        self.assertEqual(
            self.patch_text.count("type(wavefunction_type), intent(in) :: wfn"), 2
        )
        self.assertIn(
            "if (.not.allocated(error) .and. present(observer)) then",
            self.patch_text,
        )
        self.assertIn("procedure :: finished => no_op_finished", self.patch_text)
        self.assertIn(
            "call observer%finished(iscf, scf_observer_status_failed)",
            self.patch_text,
        )
        self.assertIn("No wavefunction payload is exposed on failure", self.patch_text)

    def test_probe_covers_numerics_max_iterations_and_setup_failure(self) -> None:
        """The standalone probe exercises the scientific and terminal contracts."""
        required = (
            "observer changed the energy bits",
            "observer changed density",
            "mixed/raw snapshots do not reconstruct the mixer RMS",
            "before_solve did not capture the effective Hamiltonian",
            "limited_calc%max_iter = 1",
            "invalid_calc%mixer_input%scf = 0",
            "invalid mixer emitted iteration callbacks",
            "scf_observer_status_failed",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.probe_text)
        self.assertRegex(
            self.probe_text,
            re.compile(r"lhs\(:, iorb\).*first_hamiltonian", re.DOTALL),
        )

    def test_metadata_status_values_match_the_fortran_contract(self) -> None:
        """Consumers can interpret terminal status without parsing Fortran."""
        observer = self.metadata["observer"]
        assert isinstance(observer, dict)
        self.assertEqual(
            observer["callbacks"], ["before_solve", "after_iteration", "finished"]
        )
        self.assertEqual(
            observer["status_values"],
            {"converged": 1, "max_iterations": 2, "failed": 3},
        )
        parsed = json.loads((TOOL_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(parsed, self.metadata)

    def test_retained_checkout_cannot_dirty_the_source_repository(self) -> None:
        """An output clone inside source-root is rejected before it is created."""
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "tblite"
            subprocess.run(
                ["git", "init", "--quiet", str(source_root)],
                check=True,
            )
            nested_checkout = source_root / "build" / "oracle" / "tblite"
            with self.assertRaisesRegex(
                VALIDATOR.ObserverPatchError,
                "outside the source tblite checkout",
            ):
                VALIDATOR.validate_output_location(source_root, nested_checkout)
            self.assertFalse(nested_checkout.exists())


if __name__ == "__main__":
    unittest.main()
