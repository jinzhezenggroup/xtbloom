#!/usr/bin/env python3
"""Profile only synchronized steady-state CUDA calls for one qualified cell.

The CUDA profiler API starts after adapter setup and warm-up and stops before
output publication.  ``nsys --capture-range=cudaProfilerApi`` can therefore
attribute kernels to the same FRESH energy+force invocation used for timing.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

from performance_matrix import parse_bins, select_systems
from resolve_visible_gpu import resolve_cudart


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qm9", "omol25"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cudart-library", type=Path)
    parser.add_argument("--ao-bin", type=parse_bins, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.ao_bin) != 1:
        raise RuntimeError("profile_cell requires exactly one AO bin")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    from paper_runtime import install

    install(args.repo)
    import dataset_runner  # type: ignore
    import run as benchmark_run  # type: ignore

    items = [
        item
        for item in dataset_runner.load_manifest(args.manifest, args.dataset)
        if item.system is not None and item.subset == "performance"
    ]
    selected = select_systems(items, args.ao_bin[0], args.batch_size, args.seed)
    storage = dataset_runner.storage_from_systems([item.system for item in selected])
    cell = benchmark_run.Cell(
        "xtbloom", "cuda", "device", "paper-real", "force", args.batch_size
    )
    adapter = benchmark_run.XTBloomAdapter.from_storage(
        args.library,
        storage,
        cell,
        args.device_id,
        1,
        collect_atomic_charges=False,
        max_scc_iterations=args.max_scc_iterations,
        electronic_temperature_hartree=(
            args.electronic_temperature_kelvin * 3.166808578545117e-6
        ),
    )
    cudart = ctypes.CDLL(str(resolve_cudart(args.cuda_root, args.cudart_library)))
    cudart.cudaProfilerStart.restype = ctypes.c_int
    cudart.cudaProfilerStop.restype = ctypes.c_int
    try:
        invoke = lambda: (adapter.invoke(), adapter.synchronize())
        for _ in range(args.warmups):
            invoke()
        status = int(cudart.cudaProfilerStart())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStart failed with status {status}")
        samples = []
        try:
            for _ in range(args.repetitions):
                start = time.perf_counter_ns()
                invoke()
                samples.append((time.perf_counter_ns() - start) * 1e-6)
        finally:
            stop_status = int(cudart.cudaProfilerStop())
        if stop_status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {stop_status}")
        outputs = adapter.raw_results()
        finite = all(math.isfinite(value) for value in outputs["energies_hartree"])
        finite = finite and all(
            math.isfinite(value) for value in outputs["forces_hartree_per_bohr"]
        )
        success = all(value == 0 for value in outputs["per_system_status"])
        success = success and all(value == 1 for value in outputs["scc_converged"])
        failures = []
        if not finite:
            failures.append("non-finite energy/force publication")
        if not success:
            failures.append("profiled coordinate did not converge for every system")
        document = {
            "schema_version": 1,
            "dataset": args.dataset,
            "ao_bin": args.ao_bin[0][2],
            "batch_size": args.batch_size,
            "system_ids": [item.system_id for item in selected],
            "warmups_before_capture": args.warmups,
            "profiled_repetitions": args.repetitions,
            "raw_profiled_wall_ms": samples,
            "timed_output_contract": "energy+analytic-forces",
            "capture_contract": "CUDA profiler API range contains only synchronized FRESH invokes",
            "formal_gate": {
                "status": "fail" if failures else "pass",
                "failures": failures,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        return 1 if failures else 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
