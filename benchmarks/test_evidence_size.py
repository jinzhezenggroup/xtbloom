"""Tests for the staged benchmark-evidence storage policy."""

from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.check_benchmark_evidence_size as checker

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
        self.assertIn("omit reproducible oversized raw samples", result.stderr)

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


class EvidenceSizePolicyUnitTests(unittest.TestCase):
    """Cover malformed Git states and command-line helper boundaries."""

    def test_run_git_reports_failures_and_disables_replace_refs(self) -> None:
        """Git diagnostics stay concise and replacement objects stay disabled."""
        scenarios = [
            (b"fatal: not a repository\n", "fatal: not a repository"),
            (b"", "git status failed"),
        ]
        for stderr, expected in scenarios:
            with self.subTest(stderr=stderr):
                completed = subprocess.CompletedProcess(
                    args=[], returncode=1, stdout=b"", stderr=stderr
                )
                with mock.patch.object(
                    checker.subprocess, "run", return_value=completed
                ) as run:
                    with self.assertRaisesRegex(checker.EvidencePolicyError, expected):
                        checker._run_git(
                            Path("repository"), ["status"], input_bytes=b"input"
                        )
                self.assertEqual(run.call_args.args[0][:3], ["git", "-C", "repository"])
                self.assertEqual(run.call_args.kwargs["input"], b"input")
                self.assertEqual(
                    run.call_args.kwargs["env"]["GIT_NO_REPLACE_OBJECTS"], "1"
                )

    def test_indexed_evidence_rejects_malformed_or_unmerged_entries(self) -> None:
        """Corrupt and conflicted index records fail before blob measurement."""
        scenarios = [
            (b"broken-record\0", "cannot parse git ls-files output"),
            (
                b"100644 deadbeef 2\tbenchmarks/evidence/conflict.json\0",
                "unmerged index entry.*stage 2",
            ),
        ]
        for index_output, expected in scenarios:
            with self.subTest(index_output=index_output):
                with mock.patch.object(checker, "_run_git", return_value=index_output):
                    with self.assertRaisesRegex(checker.EvidencePolicyError, expected):
                        checker.indexed_evidence(Path("repository"))

    def test_indexed_evidence_rejects_invalid_cat_file_records(self) -> None:
        """Only well-formed blob metadata can supply staged evidence sizes."""
        index_output = (
            b"100644 deadbeef 0\tbenchmarks/evidence/summary.json\0"
        )
        scenarios = [
            (b"deadbeef blob\n", checker.EvidencePolicyError, "cannot parse git cat-file"),
            (
                b"deadbeef tree 9\n",
                checker.EvidencePolicyError,
                "deadbeef is tree, not a blob",
            ),
            (b"deadbeef blob invalid\n", ValueError, "invalid literal"),
        ]
        for cat_output, error_type, expected in scenarios:
            with self.subTest(cat_output=cat_output):
                with mock.patch.object(
                    checker, "_run_git", side_effect=[index_output, cat_output]
                ):
                    with self.assertRaisesRegex(error_type, expected):
                        checker.indexed_evidence(Path("repository"))

    def test_indexed_evidence_deduplicates_blob_size_queries(self) -> None:
        """Several staged paths sharing one blob request its size only once."""
        index_output = (
            b"100644 deadbeef 0\tbenchmarks/evidence/first.json\0"
            b"100644 deadbeef 0\tbenchmarks/evidence/second.json\0"
        )
        with mock.patch.object(
            checker,
            "_run_git",
            side_effect=[index_output, b"deadbeef blob 7\n"],
        ) as run_git:
            entries = checker.indexed_evidence(Path("repository"))
        self.assertEqual([entry.size for entry in entries], [7, 7])
        self.assertEqual(run_git.call_args_list[1].kwargs["input_bytes"], b"deadbeef\n")

    def test_byte_formatting_covers_binary_units(self) -> None:
        """Diagnostics retain exact byte counts across byte, KiB, and MiB ranges."""
        self.assertEqual(checker._format_bytes(1023), "1023 bytes")
        self.assertEqual(checker._format_bytes(1536), "1.50 KiB (1536 bytes)")
        self.assertEqual(
            checker._format_bytes(2 * 1024 * 1024),
            "2.00 MiB (2097152 bytes)",
        )

    def test_positive_integer_parser_rejects_zero_and_negative_values(self) -> None:
        """Command-line limit overrides must be strictly positive."""
        self.assertEqual(checker._positive_integer("7"), 7)
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError, "limit must be a positive integer"
                ):
                    checker._positive_integer(value)

    def test_main_reports_operational_errors_without_tracebacks(self) -> None:
        """Expected repository failures become one-line hook diagnostics."""
        stderr = io.StringIO()
        with (
            mock.patch.object(checker, "check_repository", side_effect=OSError("disk")),
            contextlib.redirect_stderr(stderr),
        ):
            status = checker.main(["--repository", "repository"])
        self.assertEqual(status, 1)
        self.assertEqual(stderr.getvalue(), "FAIL benchmark evidence size: disk\n")


if __name__ == "__main__":
    unittest.main()
