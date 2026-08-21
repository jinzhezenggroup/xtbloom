#!/usr/bin/env python3
"""Create compact paper table/figure inputs and verify evidence boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROHIBITED = {".nsys-rep", ".ncu-rep", ".qdstrm", ".sqlite", ".db", ".prof"}
REQUIRED_EXPERIMENTS = {
    "freeze-manifests",
    "p0a-canonical-cpu",
    "p0a-canonical-gpu",
    "p0b-dataset-cpu",
    "p0b-dataset-gpu",
    "p0c-fd-cpu",
    "p0c-fd-gpu",
    "p0d-failure-cpu",
    "p0d-failure-gpu",
    "p0e-degeneracy",
    "p0-gate",
    "performance-reference",
    "exp1-cpu-native",
    "si-cpu-process-pool",
    "exp2-gpu-crossover",
    "exp2-gpu-capacity",
    "exp2-gpu-profiler",
    "exp3a-convergence",
    "exp3b-ragged-cpu",
    "exp3b-ragged-gpu",
    "si-gfn1-cpu",
    "si-qmmm",
    "si-cuda-mixed",
    "si-energy-only",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_stage_checksum(stage: Path) -> list[str]:
    manifest = stage / "SHA256SUMS"
    if not manifest.is_file():
        return ["missing SHA256SUMS"]
    failures = []
    environment = stage / "logs" / "environment.txt"
    environment_text = (
        environment.read_text(encoding="utf-8", errors="replace")
        if environment.is_file()
        else ""
    )
    if (
        "phase=formal\n" not in environment_text
        or "eligibility=eligible\n" not in environment_text
    ):
        failures.append("stage is not marked formal and eligible")
    listed = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            expected, name = line.split("  ", 1)
        except ValueError:
            failures.append(f"SHA256SUMS:{line_number}: malformed")
            continue
        path = Path(name)
        lexical_path = Path(os.path.abspath(path))
        listed.add(lexical_path)
        try:
            lexical_path.relative_to(Path(os.path.abspath(stage)))
        except ValueError:
            failures.append(f"SHA256SUMS:{line_number}: path escapes stage")
            continue
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"SHA256SUMS:{line_number}: missing or changed: {path}")
    actual = {
        Path(os.path.abspath(path))
        for path in stage.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.name not in {"SHA256SUMS", ".complete"}
    }
    if listed != actual:
        failures.append("stage file inventory differs from SHA256SUMS")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ignore-experiment", action="append", default=[])
    parser.add_argument("--ignore-relative", action="append", default=[])
    parser.add_argument("--require-experiment", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    ignored = set(args.ignore_relative)
    output_stage = args.output.parent.parent.resolve()
    files = sorted(
        path
        for path in args.run_root.rglob("*")
        if path.is_file()
        and not path.resolve().is_relative_to(output_stage)
        and str(path.relative_to(args.run_root)) not in ignored
    )
    prohibited = [
        str(path)
        for path in files
        if any(str(path).endswith(suffix) for suffix in PROHIBITED)
        and "raw-profiler" not in path.parts
    ]
    if prohibited:
        raise RuntimeError(f"raw profiler capture escaped raw-profiler: {prohibited}")
    experiments = {}
    failed_gates = []
    for path in files:
        if path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failed_gates.append(
                {
                    "path": str(path.relative_to(args.run_root)),
                    "reasons": [f"invalid-json: {exc}"],
                }
            )
            continue
        reasons = []
        if payload.get("formal_gate", {}).get("status") == "fail":
            reasons.append("formal_gate")
        if payload.get("summary", {}).get("failed", 0):
            reasons.append("summary.failed")
        if payload.get("passed") is False:
            reasons.append("passed=false")
        if payload.get("outlier_count", 0):
            reasons.append("equivalence_outliers")
        if reasons:
            failed_gates.append(
                {"path": str(path.relative_to(args.run_root)), "reasons": reasons}
            )
    for directory in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        complete = (directory / ".complete").is_file()
        experiments[directory.name] = {
            "complete": complete,
            "files": sum(path.is_file() for path in directory.rglob("*")),
            "bytes": sum(
                path.stat().st_size for path in directory.rglob("*") if path.is_file()
            ),
        }
    required = REQUIRED_EXPERIMENTS | set(args.require_experiment)
    ignored_experiments = set(args.ignore_experiment)
    incomplete = sorted(
        name
        for name in required
        if name not in experiments or not experiments[name]["complete"]
    )
    incomplete.extend(
        sorted(
            name
            for name, row in experiments.items()
            if not row["complete"]
            and name not in ignored_experiments
            and name not in required
        )
    )
    checksum_failures = {}
    for name in sorted(required):
        stage = args.run_root / name
        if (stage / ".complete").is_file():
            failures = verify_stage_checksum(stage)
            if failures:
                checksum_failures[name] = failures
    document = {
        "schema_version": 1,
        "run_root": str(args.run_root.resolve()),
        "experiments": experiments,
        "files": [
            {
                "path": str(path.relative_to(args.run_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
        "required_experiments": sorted(required),
        "incomplete_experiments": sorted(set(incomplete)),
        "checksum_failures": checksum_failures,
        "failed_row_level_gates": failed_gates,
        "ignored_live_files": sorted(ignored),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return (
        1
        if document["incomplete_experiments"] or checksum_failures or failed_gates
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
