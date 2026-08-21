#!/usr/bin/env python3
"""Resolve and validate the Slurm CPU affinity, NUMA and governor policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-governor")
    args = parser.parse_args()
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux sched_getaffinity is required")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("Slurm CPU affinity is empty")
    physical: dict[tuple[str, str], int] = {}
    nodes: set[int] = set()
    governors: set[str] = set()
    drivers: set[str] = set()
    minimums: set[str] = set()
    maximums: set[str] = set()
    for cpu in allowed:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        package = read(root / "topology/physical_package_id") or "unknown"
        core = read(root / "topology/core_id") or str(cpu)
        physical.setdefault((package, core), cpu)
        node_links = sorted(root.glob("node[0-9]*"))
        if node_links:
            nodes.update(int(path.name.removeprefix("node")) for path in node_links)
        governor = read(root / "cpufreq/scaling_governor")
        driver = read(root / "cpufreq/scaling_driver")
        minimum = read(root / "cpufreq/scaling_min_freq")
        maximum = read(root / "cpufreq/scaling_max_freq")
        if governor is not None:
            governors.add(governor)
        if driver is not None:
            drivers.add(driver)
        if minimum is not None:
            minimums.add(minimum)
        if maximum is not None:
            maximums.add(maximum)
    if not nodes:
        nodes.add(0)
    if args.expected_governor is not None and governors != {args.expected_governor}:
        raise RuntimeError(
            f"assigned CPU governors {sorted(governors)} do not equal {args.expected_governor!r}"
        )
    print(",".join(str(cpu) for cpu in physical.values()))
    print(",".join(str(node) for node in sorted(nodes)))
    print(",".join(sorted(governors)) or "unavailable")
    print(",".join(sorted(drivers)) or "unavailable")
    print(",".join(sorted(minimums)) or "unavailable")
    print(",".join(sorted(maximums)) or "unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
