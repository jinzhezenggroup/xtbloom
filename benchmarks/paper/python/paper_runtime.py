#!/usr/bin/env python3
"""Install the benchmark modules from one exact xTBloom checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def comparison_workload(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Bind a ratio to the same ordered inputs and successful peer mask."""
    ids = row.get("system_ids")
    outcomes = row.get("correctness", {}).get("system_outcomes")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != row.get("batch_size")
        or not all(isinstance(item, str) and item for item in ids)
        or len(set(ids)) != len(ids)
        or not isinstance(outcomes, list)
        or len(outcomes) != len(ids)
    ):
        return None
    mask = []
    for system_id, outcome in zip(ids, outcomes, strict=True):
        if (
            not isinstance(outcome, dict)
            or outcome.get("system_id") != system_id
            or not isinstance(outcome.get("successful"), bool)
        ):
            return None
        mask.append(outcome["successful"])
    return (tuple(ids), tuple(mask))


def install(repository_root: Path) -> Path:
    repo = repository_root.resolve()
    benchmark_root = repo / "benchmarks"
    for required in (benchmark_root / "dataset_runner.py", benchmark_root / "run.py"):
        if not required.is_file():
            raise RuntimeError(f"paper benchmark runtime is missing: {required}")
    os.environ["PAPER_REPO_ROOT"] = str(repo)
    # The clean, commit-pinned checkout owns the runner and every adapter.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(benchmark_root))
    return benchmark_root


def physical_cpu_ids() -> list[int]:
    """Return one Slurm-allowed logical CPU for each physical core."""
    if not hasattr(os, "sched_getaffinity"):
        return []
    selected: dict[tuple[str, str], int] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError:
            key = ("unknown", str(cpu))
        selected.setdefault(key, cpu)
    return list(selected.values())
