#!/usr/bin/env python3
"""Select three profiler candidates without assigning unmeasured bottlenecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for root in args.input_root:
        for path in sorted(root.rglob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(
                row
                for row in document.get("rows", [])
                if row.get("suite") == "gpu-crossover"
                and row.get("availability") == "available"
                and row.get("engine") == "xtbloom-ragged"
                and row.get("backend") == "cuda"
                and row.get("memory_mode") == "device"
                and row.get("correctness", {}).get("status") == "pass"
            )
    if not rows:
        raise RuntimeError("no correctness-qualified CUDA device rows are available")
    launch = min(
        (row for row in rows if row["batch_size"] == 1),
        key=lambda row: (max(row["ao_counts"]), row["ao_bin"]),
        default=None,
    )
    peak_throughput = max(
        rows, key=lambda row: row["steady_fresh"]["median_systems_per_second"]
    )
    large = [row for row in rows if max(row["ao_counts"]) >= 513]
    eigensolver = max(
        large,
        key=lambda row: (
            max(row["ao_counts"]),
            row["batch_size"] == 16,
            -abs(row["batch_size"] - 16),
        ),
        default=None,
    )
    if launch is None or eigensolver is None:
        raise RuntimeError(
            "measured matrix lacks the batch-1 or large-AO profiler candidate coordinate"
        )
    selected = []
    candidates = (
        ("batch1-smallest-ao-candidate", "possible launch-overhead regime", launch),
        (
            "peak-throughput-candidate",
            "measured crossover-matrix peak throughput",
            peak_throughput,
        ),
        (
            "large-ao-near-b16-candidate",
            "possible eigensolver-heavy regime",
            eigensolver,
        ),
    )
    for regime, target_interpretation, row in candidates:
        selected.append(
            {
                "regime": regime,
                "dataset": row["dataset"],
                "ao_bin": row["ao_bin"],
                "batch_size": row["batch_size"],
                "system_ids": row["system_ids"],
                "selection_basis": "correctness-qualified operational coordinate from the measured CUDA device crossover matrix",
                "target_hypothesis": target_interpretation,
                "claim_status": "candidate-needs-profiler-evidence",
            }
        )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps({"schema_version": 1, "points": selected}, indent=2, sort_keys=True)
        + "\n"
    )
    args.tsv_output.write_text(
        "".join(
            f"{row['regime']}\t{row['dataset']}\t{row['ao_bin']}\t{row['batch_size']}\n"
            for row in selected
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
