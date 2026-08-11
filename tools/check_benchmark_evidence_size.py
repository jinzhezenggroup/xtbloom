#!/usr/bin/env python3
"""Enforce bounded Git storage for committed benchmark evidence.

The check reads the Git index instead of the working tree. That makes it match
the bytes that a commit would publish and prevents ``git add -f`` from
bypassing the policy while still allowing large untracked measurement outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EVIDENCE_DIRECTORY = "benchmarks/evidence"
DEFAULT_MAX_FILE_BYTES = 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 16 * 1024 * 1024


class EvidencePolicyError(RuntimeError):
    """Report an invalid repository or index state without a traceback."""


@dataclass(frozen=True)
class EvidenceEntry:
    """Describe one stage-zero evidence blob selected for the next commit."""

    path: str
    object_id: str
    size: int


def _run_git(
    repository: Path, arguments: list[str], *, input_bytes: bytes | None = None
) -> bytes:
    """Run Git in ``repository`` and turn command failures into policy errors."""
    environment = os.environ.copy()
    # Replacement refs change what ``cat-file`` reads without changing the
    # object ID staged in the index. Ignore them so the measured bytes are the
    # exact blob a commit will reference.
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    completed = subprocess.run(
        ["git", "-C", os.fspath(repository), *arguments],
        input=input_bytes,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode(errors="replace").strip()
        raise EvidencePolicyError(diagnostic or f"git {' '.join(arguments)} failed")
    return completed.stdout


def indexed_evidence(repository: Path) -> list[EvidenceEntry]:
    """Return tracked evidence and blob sizes from the current Git index."""
    output = _run_git(
        repository,
        ["ls-files", "--stage", "-z", "--", EVIDENCE_DIRECTORY],
    )
    indexed: list[tuple[str, str]] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            _mode, raw_object_id, raw_stage = metadata.split()
        except ValueError as exc:
            raise EvidencePolicyError("cannot parse git ls-files output") from exc
        path = os.fsdecode(raw_path)
        stage = raw_stage.decode()
        if stage != "0":
            raise EvidencePolicyError(
                f"unmerged index entry under {EVIDENCE_DIRECTORY}: "
                f"{path} (stage {stage})"
            )
        indexed.append((path, raw_object_id.decode()))

    if not indexed:
        return []

    object_ids = sorted({object_id for _path, object_id in indexed})
    size_output = _run_git(
        repository,
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input_bytes=("\n".join(object_ids) + "\n").encode(),
    )
    sizes: dict[str, int] = {}
    for line in size_output.decode().splitlines():
        try:
            object_id, object_type, raw_size = line.split()
        except ValueError as exc:
            raise EvidencePolicyError("cannot parse git cat-file output") from exc
        if object_type != "blob":
            raise EvidencePolicyError(
                f"tracked evidence object {object_id} is {object_type}, not a blob"
            )
        sizes[object_id] = int(raw_size)

    return [
        EvidenceEntry(path=path, object_id=object_id, size=sizes[object_id])
        for path, object_id in indexed
    ]


def _format_bytes(size: int) -> str:
    """Return an exact byte count with a compact binary-unit rendering."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MiB ({size} bytes)"
    if size >= 1024:
        return f"{size / 1024:.2f} KiB ({size} bytes)"
    return f"{size} bytes"


def check_repository(
    repository: Path, *, max_file_bytes: int, max_total_bytes: int
) -> tuple[list[EvidenceEntry], int]:
    """Validate per-file and aggregate limits and return measured entries."""
    entries = indexed_evidence(repository)
    oversized = sorted(
        (entry for entry in entries if entry.size > max_file_bytes),
        key=lambda entry: (-entry.size, entry.path),
    )
    total = sum(entry.size for entry in entries)
    failures: list[str] = []
    if oversized:
        failures.append(
            f"{len(oversized)} tracked benchmark evidence file(s) exceed "
            f"{_format_bytes(max_file_bytes)}:"
        )
        failures.extend(
            f"  {_format_bytes(entry.size)}  {entry.path}" for entry in oversized
        )
    if total > max_total_bytes:
        failures.append(
            "tracked benchmark evidence totals "
            f"{_format_bytes(total)}, exceeding {_format_bytes(max_total_bytes)}"
        )
    if failures:
        failures.append(
            "Keep compact summaries in Git and archive oversized raw samples "
            "externally with an immutable URL, byte count, and SHA-256."
        )
        raise EvidencePolicyError("\n".join(failures))
    return entries, total


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return parsed


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line limits; overrides exist for focused unit tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--max-file-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_FILE_BYTES,
    )
    parser.add_argument(
        "--max-total-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_TOTAL_BYTES,
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Check the selected repository and emit one concise hook diagnostic."""
    options = parse_arguments(arguments)
    try:
        entries, total = check_repository(
            options.repository,
            max_file_bytes=options.max_file_bytes,
            max_total_bytes=options.max_total_bytes,
        )
    except (EvidencePolicyError, OSError, ValueError) as exc:
        sys.stderr.write(f"FAIL benchmark evidence size: {exc}\n")
        return 1
    sys.stdout.write(
        "PASS benchmark evidence size: "
        f"{len(entries)} tracked files, {_format_bytes(total)} total\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
