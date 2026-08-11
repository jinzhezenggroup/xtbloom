"""Tests for the staged benchmark-evidence storage policy."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPOSITORY_ROOT / "tools" / "check_benchmark_evidence_size.py"


class EvidenceSizePolicyTests(unittest.TestCase):
    """Exercise the policy against real temporary Git indexes."""

    def setUp(self) -> None:
        """Create one isolated repository for each policy scenario."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary_directory.name)
        self._git("init", "-q")

    def tearDown(self) -> None:
        """Remove the temporary repository and all staged test blobs."""
        self.temporary_directory.cleanup()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.repository), *arguments],
            check=True,
            text=True,
            capture_output=True,
        )

    def _write(self, relative_path: str, payload: bytes) -> Path:
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _check(
        self, *, max_file_bytes: int = 8, max_total_bytes: int = 16
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(CHECKER),
                "--repository",
                str(self.repository),
                "--max-file-bytes",
                str(max_file_bytes),
                "--max-total-bytes",
                str(max_total_bytes),
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_under_limit_index_passes(self) -> None:
        """Small committed summaries fit both policy dimensions."""
        self._write("benchmarks/evidence/issue-1/run/summary.csv", b"12345678")
        self._git("add", "benchmarks/evidence/issue-1/run/summary.csv")
        result = self._check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 tracked files", result.stdout)

    def test_oversized_file_fails(self) -> None:
        """One staged file cannot consume more than the per-file budget."""
        path = "benchmarks/evidence/issue-1/run/raw.json"
        self._write(path, b"123456789")
        self._git("add", path)
        result = self._check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("exceed 8 bytes", result.stderr)
        self.assertIn(path, result.stderr)

    def test_aggregate_budget_fails(self) -> None:
        """Splitting evidence cannot bypass the tracked-directory cap."""
        for name in ("first.csv", "second.csv"):
            path = f"benchmarks/evidence/issue-1/run/{name}"
            self._write(path, b"123456")
            self._git("add", path)
        result = self._check(max_total_bytes=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("totals 12 bytes", result.stderr)
        self.assertIn("exceeding 10 bytes", result.stderr)

    def test_forced_ignored_file_is_still_measured(self) -> None:
        """The staged index closes the ``git add -f`` escape hatch."""
        self._write(".gitignore", b"benchmarks/evidence/*.json\n")
        path = "benchmarks/evidence/raw.json"
        self._write(path, b"123456789")
        self._git("add", ".gitignore")
        self._git("add", "-f", path)
        result = self._check()
        self.assertEqual(result.returncode, 1)
        self.assertIn(path, result.stderr)

    def test_replace_ref_cannot_hide_staged_blob_size(self) -> None:
        """Git replacement refs cannot substitute a smaller diagnostic blob."""
        path = "benchmarks/evidence/issue-1/run/raw.json"
        self._write(path, b"123456789")
        self._git("add", path)
        large_object = self._git("rev-parse", f":{path}").stdout.strip()
        self._write("small-replacement.txt", b"x")
        small_object = self._git(
            "hash-object", "-w", "small-replacement.txt"
        ).stdout.strip()
        self._git("replace", large_object, small_object)
        result = self._check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("9 bytes", result.stderr)
        self.assertIn(path, result.stderr)

    def test_untracked_measurement_output_is_ignored(self) -> None:
        """Large local measurements remain possible until explicitly staged."""
        self._write("benchmarks/evidence/local-raw.json", b"x" * 100)
        result = self._check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0 tracked files", result.stdout)

    def test_staged_bytes_take_precedence_over_worktree_bytes(self) -> None:
        """The checker evaluates the exact blob selected for the commit."""
        path = "benchmarks/evidence/issue-1/run/summary.json"
        self._write(path, b"small")
        self._git("add", path)
        self._write(path, b"this worktree copy is deliberately oversized")
        result = self._check()
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
