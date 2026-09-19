#!/usr/bin/env python3
"""P0-D real OMol25 failure plus healthy-peer publication/isolation probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def choose_ids(results: Path, backend: str) -> tuple[str, str]:
    healthy = None
    failed = None
    with results.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row["program"]["engine"] != "xtbloom"
                or row["program"]["backend"] != backend
            ):
                continue
            category = row["status"]["category"]
            if category == "success" and healthy is None:
                healthy = row["input"]["system_id"]
            elif (
                category
                in {"scc_not_converged", "eigensolver_failed", "internal_error"}
                and failed is None
            ):
                failed = row["input"]["system_id"]
            if healthy and failed:
                return healthy, failed
    raise RuntimeError(
        f"source evidence lacks a healthy/failing xTBloom {backend} pair"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--memory-mode", choices=("host", "device", "mixed"), default="host"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-scc-iterations", type=int, default=500)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    if args.backend == "cpu" and args.memory_mode != "host":
        raise RuntimeError("CPU probe requires host descriptors")

    from paper_runtime import install

    install(args.repo)
    import dataset_runner  # type: ignore
    import run as benchmark_run  # type: ignore

    healthy_id, failed_id = choose_ids(args.source_results, args.backend)
    loaded = {
        item.system_id: item.system
        for item in dataset_runner.load_manifest(args.manifest, "omol25")
        if item.system is not None and item.system_id in {healthy_id, failed_id}
    }
    if set(loaded) != {healthy_id, failed_id}:
        raise RuntimeError("selected real peer IDs are absent from the frozen manifest")

    def execute(systems: list[Any]) -> dict[str, Any]:
        storage = dataset_runner.storage_from_systems(systems)
        cell = benchmark_run.Cell(
            "xtbloom",
            args.backend,
            args.memory_mode,
            "p0d-real-peer",
            "force",
            len(systems),
        )
        adapter = benchmark_run.XTBloomAdapter.from_storage(
            args.library,
            storage,
            cell,
            args.device_id,
            args.cpu_threads,
            collect_atomic_charges=True,
            max_scc_iterations=args.max_scc_iterations,
        )
        try:
            adapter.invoke()
            return adapter.raw_results()
        finally:
            adapter.close()

    healthy = loaded[healthy_id]
    failed = loaded[failed_id]
    sequential_healthy = execute([healthy])
    sequential_failed = execute([failed])
    ragged = execute([healthy, failed])
    healthy_atoms = len(healthy.atomic_numbers)
    failed_atoms = len(failed.atomic_numbers)
    checks: dict[str, bool] = {
        "healthy_sequential_success": sequential_healthy["per_system_status"][0] == 0
        and sequential_healthy["scc_converged"][0] == 1,
        "healthy_ragged_success": ragged["per_system_status"][0] == 0
        and ragged["scc_converged"][0] == 1,
        "failed_sequential_failed": sequential_failed["per_system_status"][0] != 0
        or sequential_failed["scc_converged"][0] != 1,
        "failed_ragged_failed": ragged["per_system_status"][1] != 0
        or ragged["scc_converged"][1] != 1,
        "healthy_energy_unchanged": math.isclose(
            sequential_healthy["energies_hartree"][0],
            ragged["energies_hartree"][0],
            rel_tol=0.0,
            abs_tol=args.atol,
        ),
        "healthy_forces_unchanged": all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=args.atol)
            for a, b in zip(
                sequential_healthy["forces_hartree_per_bohr"],
                ragged["forces_hartree_per_bohr"][: 3 * healthy_atoms],
                strict=True,
            )
        ),
        "healthy_charges_unchanged": all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=args.atol)
            for a, b in zip(
                sequential_healthy["atomic_charges_e"],
                ragged["atomic_charges_e"][:healthy_atoms],
                strict=True,
            )
        ),
        "failed_ragged_energy_nan": math.isnan(ragged["energies_hartree"][1]),
        "failed_ragged_forces_all_nan": all(
            math.isnan(value)
            for value in ragged["forces_hartree_per_bohr"][
                3 * healthy_atoms : 3 * (healthy_atoms + failed_atoms)
            ]
        ),
        "failed_ragged_charges_all_nan": all(
            math.isnan(value)
            for value in ragged["atomic_charges_e"][
                healthy_atoms : healthy_atoms + failed_atoms
            ]
        ),
    }
    document = {
        "schema_version": 1,
        "backend": args.backend,
        "memory_mode": args.memory_mode,
        "healthy_system_id": healthy_id,
        "failed_system_id": failed_id,
        "checks": checks,
        "passed": all(checks.values()),
        "sequential_healthy": sequential_healthy,
        "sequential_failed": sequential_failed,
        "ragged": ragged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
