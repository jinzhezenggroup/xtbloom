#!/usr/bin/env python3
"""Conditional SI: paired FRESH/WARM checks on a real fixed-topology trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="JSON with atomic_numbers, charge, uhf and frames_bohr",
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    parser.add_argument("--energy-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=5e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    os.environ["XTBLOOM_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(args.repo / "python"))
    import numpy as np
    from xtbloom import Calculator

    trajectory = json.loads(args.trajectory.read_text())
    frames = trajectory["frames_bohr"]
    if len(frames) < 3:
        raise RuntimeError("trajectory needs at least three real frames")

    def run(warm: bool) -> list[dict[str, object]]:
        calculator = Calculator(
            "GFN2-xTB",
            trajectory["atomic_numbers"],
            np.asarray(frames[0]),
            charge=trajectory.get("charge", 0.0),
            uhf=trajectory.get("uhf", 0),
            backend=args.backend,
            device_id=args.device_id if args.backend == "cuda" else None,
            cpu_threads=args.cpu_threads,
            max_scc_iterations=args.max_scc_iterations,
            electronic_temperature=args.electronic_temperature_kelvin,
            warm_start=warm,
        )
        rows = []
        try:
            for index, frame in enumerate(frames):
                calculator.update(positions=np.asarray(frame))
                start = time.perf_counter_ns()
                result = calculator.singlepoint()
                elapsed = (time.perf_counter_ns() - start) * 1e-6
                rows.append(
                    {
                        "frame": index,
                        "elapsed_ms": elapsed,
                        "energy_hartree": float(result.energy),
                        "forces_hartree_per_bohr": np.asarray(result.forces)
                        .reshape(-1)
                        .tolist(),
                        "scc_iterations": int(result.scc_iterations),
                        "scc_status": str(result.scc_status),
                    }
                )
        finally:
            calculator.close()
        return rows

    fresh, warm = run(False), run(True)
    energy_errors = [
        abs(a["energy_hartree"] - b["energy_hartree"])
        for a, b in zip(fresh, warm, strict=True)
    ]
    force_errors = [
        max(
            abs(float(x) - float(y))
            for x, y in zip(
                a["forces_hartree_per_bohr"], b["forces_hartree_per_bohr"], strict=True
            )
        )
        for a, b in zip(fresh, warm, strict=True)
    ]
    generator = random.Random(hashlib.sha256(args.trajectory.read_bytes()).hexdigest())
    speedups = []
    for _ in range(args.bootstrap_samples):
        selected = [generator.randrange(len(frames)) for _ in frames]
        fresh_median = statistics.median(
            float(fresh[index]["elapsed_ms"]) for index in selected
        )
        warm_median = statistics.median(
            float(warm[index]["elapsed_ms"]) for index in selected
        )
        speedups.append(fresh_median / warm_median)
    speedups.sort()
    passed = (
        max(energy_errors) <= args.energy_atol and max(force_errors) <= args.force_atol
    )
    document = {
        "schema_version": 1,
        "backend": args.backend,
        "device_id": args.device_id if args.backend == "cuda" else None,
        "max_scc_iterations": args.max_scc_iterations,
        "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
        "electronic_temperature_hartree": (
            args.electronic_temperature_kelvin * 3.166808578545117e-6
        ),
        "xtbloom_module": str(Path(sys.modules["xtbloom"].__file__).resolve()),
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "fresh": fresh,
        "warm": warm,
        "summary": {
            "frames": len(frames),
            "max_abs_energy_difference_hartree": max(energy_errors),
            "energy_atol_hartree": args.energy_atol,
            "max_abs_force_difference_hartree_per_bohr": max(force_errors),
            "force_atol_hartree_per_bohr": args.force_atol,
            "fresh_median_ms": statistics.median(row["elapsed_ms"] for row in fresh),
            "warm_median_ms": statistics.median(row["elapsed_ms"] for row in warm),
            "warm_speedup_bootstrap_95_ci": [
                speedups[int(0.025 * (len(speedups) - 1))],
                speedups[int(0.975 * (len(speedups) - 1))],
            ],
        },
        "passed": passed,
        "claim_scope": "conditional SI sanity only; not part of the main performance matrix",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
