#!/usr/bin/env python3
"""Measure one correctness-qualified fixed-plan changed-geometry CUDA coordinate."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ENERGY_ATOL_HARTREE = 5.0e-7
FORCE_ATOL_HARTREE_PER_BOHR = 5.0e-7


def parse_args() -> argparse.Namespace:
    """Return the explicit source, library, and output identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Hash a regular file without retaining its contents twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(source_root: Path, *arguments: str) -> str:
    """Run one required read-only Git query for the measured source tree."""
    return subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_source(source_root: Path) -> str:
    """Return the measured revision or reject a dirty source worktree."""
    revision = git_text(source_root, "rev-parse", "HEAD")
    if git_text(source_root, "status", "--porcelain"):
        raise RuntimeError("measured source worktree must be clean")
    return revision


def changed_positions(
    original_positions: list[float], atoms: int, step: int
) -> list[float]:
    """Return one deterministic, topology-preserving changed geometry."""
    if atoms <= 0 or len(original_positions) != 3 * atoms:
        raise ValueError("positions must contain exactly three values per atom")
    values = list(original_positions)
    atom = step % atoms
    axis = (step // atoms) % 3
    direction = -1.0 if step % 2 else 1.0
    values[3 * atom + axis] += direction * (0.0005 + 0.00001 * step)
    return values


def compare_sample(
    cuda_output: dict[str, object], cpu_output: object, atoms: int
) -> dict[str, float | str]:
    """Compare one changed-geometry CUDA result with its public CPU result."""
    cuda_energies = cuda_output["energies_hartree"]
    cuda_forces = cuda_output["forces_hartree_per_bohr"]
    if not isinstance(cuda_energies, list) or not isinstance(cuda_forces, list):
        raise TypeError("CUDA output must contain materialized energy and force lists")
    cpu_energies = cpu_output.energies
    cpu_forces = cpu_output.forces
    energy_errors = [
        abs(float(cuda_energies[index]) - float(cpu_energies[index]))
        for index in range(len(cuda_energies))
    ]
    force_errors = [
        abs(float(cuda_forces[index]) - float(cpu_forces[index]))
        for index in range(3 * atoms)
    ]
    max_energy_error = max(energy_errors)
    max_force_error = max(force_errors)
    passed = (
        max_energy_error <= ENERGY_ATOL_HARTREE
        and max_force_error <= FORCE_ATOL_HARTREE_PER_BOHR
    )
    return {
        "status": "pass" if passed else "fail",
        "max_abs_energy_error_hartree": max_energy_error,
        "max_abs_force_error_hartree_per_bohr": max_force_error,
    }


def summarize_correctness(
    samples: list[dict[str, int | float | str]],
) -> dict[str, int | float | str]:
    """Aggregate per-geometry correctness without hiding a failed sample."""
    if not samples:
        raise ValueError("at least one changed-geometry sample is required")
    passed = all(sample["correctness_status"] == "pass" for sample in samples)
    return {
        "reference": "same-library public CPU execution at every changed geometry",
        "validated_samples": len(samples),
        "max_abs_energy_error_hartree": max(
            float(sample["max_abs_energy_error_hartree"]) for sample in samples
        ),
        "max_abs_force_error_hartree_per_bohr": max(
            float(sample["max_abs_force_error_hartree_per_bohr"]) for sample in samples
        ),
        "energy_atol_hartree": ENERGY_ATOL_HARTREE,
        "force_atol_hartree_per_bohr": FORCE_ATOL_HARTREE_PER_BOHR,
        "status": "pass" if passed else "fail",
    }


def result_exit_status(document: dict[str, object]) -> int:
    """Return nonzero when the published correctness aggregate failed."""
    correctness = document.get("correctness")
    if not isinstance(correctness, dict):
        raise TypeError("result document is missing a correctness object")
    return 0 if correctness.get("status") == "pass" else 1


def main() -> int:
    """Measure and CPU-qualify 20 changed geometries on one fixed CUDA plan."""
    args = parse_args()
    source_root = args.source_root.resolve()
    library_path = args.library.resolve()
    source_revision = require_clean_source(source_root)

    sys.path.insert(0, str(source_root))
    from benchmarks import run as benchmark

    public_api = benchmark.public_api
    manifest_path = source_root / "data/conformance/manifest.json"
    manifest = benchmark.conformance.load_json(manifest_path)
    cases = {
        case["id"]: case
        for case in benchmark.conformance.selected_cases(manifest, None)
    }
    cell = benchmark.Cell("xtbloom", "cuda", "device", "gas", "force", 4)
    case_sequence = benchmark.workload_case_sequence("gas", 4, cases)

    setup_start = time.perf_counter_ns()
    adapter = benchmark.XTBloomAdapter(
        library_path,
        manifest_path,
        manifest,
        case_sequence,
        cell,
        args.device_id,
        1,
    )
    adapter_setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
    plan = ctypes.c_void_p()
    try:
        library = adapter.library
        library.xtbloom_plan_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(public_api.Batch),
            ctypes.POINTER(public_api.ComputeOptions),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.xtbloom_plan_create.restype = ctypes.c_int32
        library.xtbloom_plan_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(public_api.Batch),
            ctypes.POINTER(public_api.ComputeOptions),
            ctypes.POINTER(public_api.BatchResult),
        ]
        library.xtbloom_plan_compute.restype = ctypes.c_int32
        library.xtbloom_plan_destroy.argtypes = [ctypes.c_void_p]
        library.xtbloom_plan_destroy.restype = None

        plan_start = time.perf_counter_ns()
        public_api._call_ok(
            library,
            library.xtbloom_plan_create(
                adapter.context,
                ctypes.byref(adapter.batch),
                ctypes.byref(adapter.options),
                ctypes.byref(plan),
            ),
            "xtbloom_plan_create",
        )
        plan_create_ms = (time.perf_counter_ns() - plan_start) * 1.0e-6

        def plan_timing() -> float:
            start = time.perf_counter_ns()
            public_api._call_ok(
                library,
                library.xtbloom_plan_compute(
                    plan,
                    ctypes.byref(adapter.batch),
                    ctypes.byref(adapter.options),
                    ctypes.byref(adapter.result),
                ),
                "xtbloom_plan_compute",
            )
            adapter.synchronize()
            return (time.perf_counter_ns() - start) * 1.0e-6

        first_original_geometry_ms = plan_timing()
        original_positions = list(adapter.storage.positions)
        cuda = adapter.memory.cuda
        if cuda is None:
            raise RuntimeError("device descriptors require a CUDA allocation owner")
        position_pointer = ctypes.c_void_p(adapter.batch.positions.data)

        def upload_geometry(step: int) -> list[float]:
            values = changed_positions(original_positions, adapter.atoms, step)
            owner = (ctypes.c_double * len(values))(*values)
            # The public descriptor already names device memory. The caller's
            # geometry update therefore precedes, and is excluded from, the
            # timed plan_compute boundary just as it would in an MD integrator.
            cuda._check(
                cuda.runtime.cudaMemcpy(
                    position_pointer,
                    ctypes.cast(owner, ctypes.c_void_p),
                    ctypes.sizeof(owner),
                    public_api.CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "cudaMemcpy changed positions H2D",
            )
            return values

        for step in range(3):
            upload_geometry(step)
            plan_timing()

        samples: list[dict[str, int | float | str]] = []
        cpu_storage = public_api.assemble_batch(manifest_path, manifest, case_sequence)
        for sample in range(20):
            step = sample + 3
            positions = upload_geometry(step)
            latency_ms = plan_timing()
            cuda_output = adapter.results()
            statuses = [int(value) for value in adapter.statuses]
            converged = [int(value) for value in adapter.converged]
            iterations = [int(value) for value in adapter.iterations]
            if statuses != [public_api.XTBLOOM_STATUS_SUCCESS] * adapter.systems:
                raise RuntimeError(f"failed statuses at sample {sample}: {statuses}")
            if converged != [1] * adapter.systems:
                raise RuntimeError(f"non-converged sample {sample}: {converged}")

            # CPU qualification deliberately follows the synchronized timed
            # interval. Every retained geometry is checked independently so a
            # geometry-specific CUDA error cannot hide behind the final sample.
            cpu_storage.positions[:] = positions
            cpu_output = public_api.run_compute(
                library,
                cpu_storage,
                adapter.options,
                "cpu",
                args.device_id,
                1,
                "host",
            )
            correctness = compare_sample(cuda_output, cpu_output, adapter.atoms)
            samples.append(
                {
                    "sample": sample,
                    "geometry_step": step,
                    "latency_ms": latency_ms,
                    "scc_iterations_min": min(iterations),
                    "scc_iterations_max": max(iterations),
                    "correctness_status": correctness["status"],
                    "max_abs_energy_error_hartree": correctness[
                        "max_abs_energy_error_hartree"
                    ],
                    "max_abs_force_error_hartree_per_bohr": correctness[
                        "max_abs_force_error_hartree_per_bohr"
                    ],
                }
            )

        latencies = [float(sample["latency_ms"]) for sample in samples]
        correctness = summarize_correctness(samples)
        document = {
            "schema_version": 1,
            "source_revision": source_revision,
            "source_clean": True,
            # Preserve the caller-facing path from the exact command while the
            # resolved file and its hash above remain the binary identity.
            "library": str(args.library),
            "library_sha256": sha256_file(library_path),
            "backend": "cuda",
            "memory_mode": "device",
            "workload": "gas/ketene",
            "property": "force",
            "batch_size": 4,
            "scc_start": "FRESH",
            "setup": {
                "adapter_setup_ms": adapter_setup_ms,
                "plan_create_ms": plan_create_ms,
                "first_original_geometry_plan_compute_ms": (first_original_geometry_ms),
            },
            "changed_geometry": {
                "warmups": 3,
                "samples": samples,
                "median_ms": statistics.median(latencies),
                "p95_ms": benchmark.percentile(latencies, 0.95),
                "systems_per_second_at_median": (4000.0 / statistics.median(latencies)),
                "geometry_upload_scope": (
                    "synchronous H2D position update before each timed "
                    "plan_compute; excluded from latency"
                ),
                "timing_scope": (
                    "xtbloom_plan_compute plus explicit cudaDeviceSynchronize"
                ),
                "descriptor_address": int(position_pointer.value or 0),
                "all_system_status": "SUCCESS",
                "all_scc_converged": True,
                "scc_iterations_min": min(
                    int(sample["scc_iterations_min"]) for sample in samples
                ),
                "scc_iterations_max": max(
                    int(sample["scc_iterations_max"]) for sample in samples
                ),
            },
            "correctness": correctness,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        if plan.value:
            adapter.library.xtbloom_plan_destroy(plan)
        adapter.close()
    return result_exit_status(document)


if __name__ == "__main__":
    raise SystemExit(main())
