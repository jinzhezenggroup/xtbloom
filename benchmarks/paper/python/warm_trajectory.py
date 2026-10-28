#!/usr/bin/env python3
"""Conditional SI: paired FRESH/WARM checks on a real fixed-topology trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def paired_trajectories(
    run: Callable[[bool], list[dict[str, Any]]], warmups: int, repetitions: int
) -> list[dict[str, Any]]:
    """Repeat complete trajectories with fresh contexts and balanced mode order."""
    if warmups < 0 or repetitions < 2:
        raise ValueError("need nonnegative warmups and at least two paired repetitions")
    pairs = []
    for index in range(warmups + repetitions):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair: dict[str, Any] = {
            "order": ["warm" if warm else "fresh" for warm in order]
        }
        for warm in order:
            pair["warm" if warm else "fresh"] = run(warm)
        if index >= warmups:
            pair["repetition"] = index - warmups
            pairs.append(pair)
    return pairs


def paired_speedup_interval(
    pairs: list[dict[str, Any]], draws: int, seed: str
) -> dict[str, Any]:
    """Resample independent paired trajectory totals, never correlated frames."""
    if draws < 1 or len(pairs) < 2:
        raise ValueError("need positive bootstrap samples and two paired repetitions")
    totals = {
        mode: [sum(float(row["elapsed_ms"]) for row in pair[mode]) for pair in pairs]
        for mode in ("fresh", "warm")
    }
    if any(
        not math.isfinite(value) or value <= 0
        for values in totals.values()
        for value in values
    ):
        raise ValueError("trajectory timings must be finite and positive")
    generator = random.Random(seed)
    speedups = []
    for _ in range(draws):
        selected = [generator.randrange(len(pairs)) for _ in pairs]
        speedups.append(
            statistics.median(totals["fresh"][i] for i in selected)
            / statistics.median(totals["warm"][i] for i in selected)
        )
    speedups.sort()
    fresh = statistics.median(totals["fresh"])
    warm = statistics.median(totals["warm"])
    return {
        "fresh_median_trajectory_ms": fresh,
        "warm_median_trajectory_ms": warm,
        "warm_speedup": fresh / warm,
        "warm_speedup_bootstrap_95_ci": [
            speedups[int(0.025 * (len(speedups) - 1))],
            speedups[int(0.975 * (len(speedups) - 1))],
        ],
        "sampling_unit": "independent paired complete trajectory; initial frame included",
    }


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
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.warmups < 0 or args.repetitions < 2:
        parser.error("need nonnegative --warmups and at least two --repetitions")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    os.environ["XTBLOOM_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(args.repo / "python"))
    import numpy as np
    from xtbloom import Calculator

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frames = trajectory["frames_bohr"]
    if len(frames) < 3:
        raise RuntimeError("trajectory needs at least three real frames")

    def run(warm: bool) -> list[dict[str, Any]]:
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
                        "scc_status": int(result.scc_status),
                        "scc_converged": bool(result.scc_converged),
                    }
                )
        finally:
            calculator.close()
        return rows

    pairs = paired_trajectories(run, args.warmups, args.repetitions)
    fresh = [row for pair in pairs for row in pair["fresh"]]
    warm = [row for pair in pairs for row in pair["warm"]]
    timing = paired_speedup_interval(
        pairs,
        args.bootstrap_samples,
        hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
    )
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
    passed = (
        all(
            math.isfinite(float(value))
            for row in (*fresh, *warm)
            for value in (row["energy_hartree"], *row["forces_hartree_per_bohr"])
        )
        and all(math.isfinite(value) for value in (*energy_errors, *force_errors))
        and all(
            row["scc_status"] == 0 and row["scc_converged"] for row in (*fresh, *warm)
        )
        and max(energy_errors) <= args.energy_atol
        and max(force_errors) <= args.force_atol
    )
    document = {
        "schema_version": 2,
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
        "paired_repetitions": pairs,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "bootstrap_samples": args.bootstrap_samples,
        "summary": {
            "frames": len(frames),
            "max_abs_energy_difference_hartree": max(energy_errors),
            "energy_atol_hartree": args.energy_atol,
            "max_abs_force_difference_hartree_per_bohr": max(force_errors),
            "force_atol_hartree_per_bohr": args.force_atol,
            **timing,
        },
        "passed": passed,
        "claim_scope": "conditional SI sanity only; not part of the main performance matrix",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
