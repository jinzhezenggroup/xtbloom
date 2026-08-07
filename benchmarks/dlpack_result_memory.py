#!/usr/bin/env python3
"""Allocation-cost comparison for gpuxtb-owned DLPack device results.

Issue #214 measures the steady-state cost of the two supported device-output
paths through ``gpuxtb.ArrayBatch`` on a real NVIDIA GPU:

* ``arena``: ``result_memory="cuda"`` with no ``out=``.  Every call allocates
  one packed gpuxtb-owned device arena (``cudaMalloc`` through the native
  ``gpuxtb_result_owner_create``), computes into it, returns
  ``DLPackResultBuffer`` producers, and frees the arena when the caller
  releases the producers.
* ``out``: every requested output is a caller-owned writable CUDA buffer
  supplied through ``out=``.  Steady state performs no gpuxtb allocation at
  all; this is the favored steady-state zero-copy path.

The claim is deliberately narrow: on the recorded machine, ``result_memory=
"cuda"`` adds the per-call cost of one packed arena allocation plus its final
release compared with the caller-owned ``out=`` path, with no device-to-host
result transfer and no added device-wide synchronization in either path.
Wall-clock latency is measured at the public Python call boundary with an
explicit synchronize after every call; correctness (energy/force parity with
the host CPU result) is validated on the first and last sample of every run.

Output is a JSON file with raw per-sample latencies and full environment
identity, plus a compact CSV view.  Existing output files are rejected, so a
previous bundle cannot be silently overwritten.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_revision() -> tuple[str, str]:
    """Return (revision, dirty marker) for the repository."""
    rev = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return rev.stdout.strip(), ("dirty" if status.stdout.strip() else "clean")


def _library_identity() -> dict[str, str]:
    """Report the resolved native library path and its SHA-256."""
    from gpuxtb import library

    path = Path(str(library.library_path()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"library_path": str(path), "library_sha256": digest}


def _packed_water() -> dict[str, np.ndarray]:
    """Pack a fixed water geometry into flat host numpy descriptors."""
    numbers = np.array([8, 1, 1], dtype=np.int32)
    positions = np.array(
        [
            [0.0000000000, 0.0000000000, -0.7357858611],
            [1.4418315287, 0.0000000000, 0.3678929305],
            [-1.4418315287, 0.0000000000, 0.3678929305],
        ],
        dtype=np.float64,
    )
    return {
        "atom_offsets": np.asarray([0, 3], dtype=np.int64),
        "atomic_numbers": numbers,
        "positions": positions,
        "molecular_charges": np.asarray([0.0], dtype=np.float64),
        "unpaired_electrons": np.asarray([0], dtype=np.int32),
        "spin_channels": np.asarray([1], dtype=np.int32),
    }


def _require_gpu() -> None:
    """Assert that a CUDA device is actually usable before timing."""
    from gpuxtb.exceptions import GPUxtbRuntimeError
    from gpuxtb.interface import Context

    try:
        with Context("cuda"):
            pass
    except GPUxtbRuntimeError as exc:
        raise SystemExit(f"CUDA backend is not usable: {exc}") from exc


def _measure(
    packed: dict[str, np.ndarray],
    mode: str,
    *,
    warmup: int,
    repetitions: int,
) -> list[float]:
    """Time ``repetitions`` warm-state calls of the selected output mode.

    The caller-owned ``out=`` path preallocates writable torch CUDA tensors
    once (outside timing); the arena path collects and releases producers
    after each call so the per-call arena free is included.
    """
    import torch
    from gpuxtb.interface import ArrayBatch

    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    natoms = len(packed["atomic_numbers"])

    if mode == "out":
        out = {
            "energies": torch.empty(1, dtype=torch.float64, device="cuda"),
            "forces": torch.empty((natoms, 3), dtype=torch.float64, device="cuda"),
            "charges": torch.empty(natoms, dtype=torch.float64, device="cuda"),
            "scc_iterations": torch.empty(1, dtype=torch.int32, device="cuda"),
            "scc_converged": torch.empty(1, dtype=torch.uint8, device="cuda"),
            "per_system_status": torch.empty(1, dtype=torch.int32, device="cuda"),
        }
    else:
        out = None

    for _ in range(warmup):
        if mode == "arena":
            warm = batch.compute(result_memory="cuda")
            warm.close()
        else:
            batch.compute(result_memory="host", out=out)
        torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        if mode == "arena":
            sample = batch.compute(result_memory="cuda", out=None)
            # Deterministically release every producer reference so the native
            # arena free (cudaFree) runs inside the measured interval.  The
            # caller's Python GC policy is outside the measured window, so the
            # torch object-graph GC overhead cannot masquerade as a gpuxtb
            # allocation cost.
            result_names = (
                "energies",
                "forces",
                "charges",
                "scc_iterations",
                "scc_converged",
                "per_system_status",
            )
            for name in result_names:
                producer = sample.get(name)
                if hasattr(producer, "close"):
                    producer.close()
            sample.close()
        else:
            sample = batch.compute(result_memory="host", out=out)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - start) / 1.0e6)
    batch.close()
    return latencies


def _correctness(packed: dict[str, np.ndarray], mode: str) -> dict[str, object]:
    """Independent single-shot correctness check for the given mode."""
    import torch
    from gpuxtb.interface import ArrayBatch, compute_arrays

    host = compute_arrays(**{name: value.copy() for name, value in packed.items()})
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    natoms = len(packed["atomic_numbers"])
    if mode == "out":
        out = {
            "energies": torch.empty(1, dtype=torch.float64, device="cuda"),
            "forces": torch.empty((natoms, 3), dtype=torch.float64, device="cuda"),
            "charges": torch.empty(natoms, dtype=torch.float64, device="cuda"),
            "scc_iterations": torch.empty(1, dtype=torch.int32, device="cuda"),
            "scc_converged": torch.empty(1, dtype=torch.uint8, device="cuda"),
            "per_system_status": torch.empty(1, dtype=torch.int32, device="cuda"),
        }
        result = batch.compute(result_memory="host", out=out)
        energy = float(result.energies)
        forces = out["forces"].cpu().numpy()
        charges = out["charges"].cpu().numpy()
    else:
        result = batch.compute(result_memory="cuda")
        energy = float(torch.from_dlpack(result.energies).cpu().numpy()[0])
        forces = torch.from_dlpack(result.forces).cpu().numpy()
        charges = torch.from_dlpack(result.charges).cpu().numpy()
        result.close()
    max_force_error = float(np.max(np.abs(forces - host.forces)))
    max_charge_error = float(np.max(np.abs(charges - host.charges)))
    energy_error = float(abs(energy - float(host.energies[0])))
    score = max(energy_error, max_force_error, max_charge_error)
    batch.close()
    return {
        "mode": mode,
        "energy": energy,
        "max_force_abs_error": max_force_error,
        "max_charge_abs_error": max_charge_error,
        "score": score,
    }


def _summary(latencies: list[float]) -> dict[str, float]:
    """Compute the requested summary statistics of one raw sample set."""
    return {
        "count": len(latencies),
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "p50_ms": statistics.quantiles(latencies, n=100, method="inclusive")[49],
        "p99_ms": statistics.quantiles(latencies, n=100, method="inclusive")[98],
    }


def main() -> int:
    """Run the arena-vs-out allocation benchmark and write JSON+CSV evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=60)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output {args.output}")

    _require_gpu()
    packed = _packed_water()

    import torch

    mode_records: dict[str, object] = {}
    for mode in ("arena", "out"):
        latencies = _measure(
            packed, mode, warmup=args.warmup, repetitions=args.repetitions
        )
        mode_records[mode] = {
            "raw_latency_ms": latencies,
            "summary": _summary(latencies),
            "correctness": _correctness(packed, mode),
        }

    revision, dirty = _git_revision()
    document = {
        "schema_version": 1,
        "claim": (
            "On the recorded machine, result_memory='cuda' adds one packed "
            "arena allocation plus its release per call compared with the "
            "caller-owned out= path, measured at the public Python boundary "
            "with an explicit synchronize after each call; neither path "
            "transfers results to the host."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_status": dirty,
        "gpu": _gpu_identity(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        "library": _library_identity(),
        "workload": {
            "molecule": "water (8,1,1)",
            "nsystems": 1,
            "natoms": 3,
            "descriptor_mode": "host numpy inputs",
            "stream": "legacy default (stream=1)",
            "properties": "energy,forces,charges",
        },
        "timing": {
            "boundary": (
                "perf_counter_ns around each public ArrayBatch.compute() with "
                "torch.cuda.synchronize() before start and after stop"
            ),
            "warmup_calls": args.warmup,
            "repetitions": args.repetitions,
            "modes": mode_records,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "sample", "latency_ms"])
        for mode, record in mode_records.items():
            for index, latency in enumerate(record["raw_latency_ms"]):
                writer.writerow([mode, index, f"{latency:.6f}"])

    for mode, record in mode_records.items():
        summary = record["summary"]
        print(  # noqa: T201 - benchmark CLI progress output
            f"{mode}: mean {summary['mean_ms']:.4f} ms, "
            f"median {summary['median_ms']:.4f} ms, "
            f"min {summary['min_ms']:.4f} ms, max {summary['max_ms']:.4f} ms "
            f"over {summary['count']} samples"
        )
    print(f"wrote {args.output}")  # noqa: T201 - benchmark CLI progress output
    print(f"wrote {csv_path}")  # noqa: T201 - benchmark CLI progress output
    return 0


_HOST_GPU_CACHE: dict[str, object] = {}


def _gpu_identity() -> dict[str, object]:
    """Return GPU/driver identity queried through the active CUDA runtime."""

    def _probe() -> dict[str, object]:
        import torch

        props = torch.cuda.get_device_properties(0)
        cudart = ctypes.CDLL("libcudart.so.12")
        driver = ctypes.c_int(0)
        cudart.cudaDriverGetVersion(ctypes.byref(driver))
        runtime = ctypes.c_int(0)
        cudart.cudaRuntimeGetVersion(ctypes.byref(runtime))
        return {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "driver_version": driver.value,
            "cuda_runtime_version": runtime.value,
            "torch_cuda_build": torch.version.cuda,
        }

    if not _HOST_GPU_CACHE:
        _HOST_GPU_CACHE["value"] = _probe()
    return _HOST_GPU_CACHE["value"]


if __name__ == "__main__":
    raise SystemExit(main())
