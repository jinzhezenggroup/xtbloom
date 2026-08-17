#!/usr/bin/env python3
"""Verify the omitted raw issue-429 artifacts and emit compact distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


def _percentile(samples: list[float], fraction: float) -> float:
    """Return a linearly interpolated inclusive percentile."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_manifest(manifest: Path) -> dict[str, str]:
    """Load the exact raw-artifact hashes retained by the evidence bundle."""
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[filename] = digest
    return entries


def _verified_documents(
    raw_directory: Path, manifest: Path
) -> list[tuple[str, dict[str, Any]]]:
    """Return every manifest-listed JSON document after byte verification."""
    expected = _load_manifest(manifest)
    actual = {path.name for path in raw_directory.glob("*.json")}
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise ValueError(
            f"raw artifact set differs: missing={missing}, unexpected={unexpected}"
        )

    documents: list[tuple[str, dict[str, Any]]] = []
    for filename, expected_digest in sorted(expected.items()):
        path = raw_directory / filename
        payload = path.read_bytes()
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"raw artifact hash differs for {filename}: "
                f"expected={expected_digest}, actual={actual_digest}"
            )
        documents.append((filename, json.loads(payload)))
    return documents


def _process_round(filename: str) -> int:
    """Recover the explicit process round, treating an unsuffixed run as round one."""
    match = re.search(r"-r([0-9]+)\.json$", filename)
    return int(match.group(1)) if match is not None else 1


def main() -> int:
    """Verify raw inputs and write one compact distribution row per benchmark cell."""
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    arguments = parser.parse_args()

    manifest = Path(__file__).with_name("RAW_SHA256SUMS")
    documents = _verified_documents(arguments.raw_directory, manifest)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "artifact",
            "generated_at_utc",
            "isa",
            "process_round",
            "start_mode",
            "batch_size",
            "natoms",
            "cpu_threads",
            "affinity",
            "samples",
            "min_ms",
            "q1_ms",
            "median_ms",
            "q3_ms",
            "max_ms",
            "mean_ms",
            "p95_ms",
            "scc_iterations_median",
            "correctness_status",
        )
    )
    for filename, document in documents:
        identity = document["run_identity"]
        isa = identity["cpu_dispatch_environment"]["XTBLOOM_CPU_ISA"]
        affinity = ";".join(
            str(cpu) for cpu in identity["hardware"]["process_affinity"]
        )
        for row in document["rows"]:
            samples = [float(value) for value in row["timing"]["samples_ms"]]
            writer.writerow(
                (
                    filename,
                    identity["generated_at_utc"],
                    isa,
                    _process_round(filename),
                    row["start_mode"],
                    row["batch_size"],
                    row["natoms"],
                    row["compute_options"]["cpu_threads"],
                    affinity,
                    len(samples),
                    f"{min(samples):.6f}",
                    f"{_percentile(samples, 0.25):.6f}",
                    f"{statistics.median(samples):.6f}",
                    f"{_percentile(samples, 0.75):.6f}",
                    f"{max(samples):.6f}",
                    f"{statistics.fmean(samples):.6f}",
                    f"{row['timing']['p95_ms']:.6f}",
                    row["iteration_summary"]["median"],
                    row["correctness"]["status"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
