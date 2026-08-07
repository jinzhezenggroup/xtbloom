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
explicit synchronize after every call. Arena/out calls use a counterbalanced
AB/BA order, and the final-evidence gate requires arena mean latency to be no
more than five percent above ``out=``. Finite energy/force/charge parity against
an explicit CPU backend, SCC status, and convergence are validated before and
after timing.

Output is a JSON file with raw per-sample latencies and full environment
identity, plus a compact LF-terminated CSV view. Dirty source trees and either
pre-existing output file are rejected, so final evidence is tied to recoverable
committed code and a previous bundle cannot be silently overwritten.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_NAMES = (
    "energies",
    "forces",
    "charges",
    "scc_iterations",
    "scc_converged",
    "per_system_status",
)
CORRECTNESS_ATOL = {
    "energy_hartree": 1.0e-10,
    "force_hartree_per_bohr": 1.0e-9,
    "charge_electron": 1.0e-9,
}
MAX_MEAN_OVERHEAD_FRACTION = 0.05
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "MKL_INTERFACE_LAYER",
    "MKL_THREADING_LAYER",
    "CUDA_VISIBLE_DEVICES",
)


def _git_revision() -> tuple[str, bool]:
    """Return the exact repository revision and dirty bit."""
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
    if rev.returncode != 0 or status.returncode != 0 or not rev.stdout.strip():
        raise SystemExit("cannot resolve the benchmark repository Git identity")
    return rev.stdout.strip(), bool(status.stdout.strip())


def _library_identity() -> dict[str, object]:
    """Report the selected native library and adjacent CMake build identity."""
    import gpuxtb
    from gpuxtb import library

    path = Path(str(library.library_path())).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity: dict[str, object] = {
        "library_path": str(path),
        "library_sha256": digest,
        "python_package": str(Path(gpuxtb.__file__).resolve()),
    }
    cache = path.parent / "CMakeCache.txt"
    if cache.is_file():
        cache_text = cache.read_text(encoding="utf-8", errors="replace")
        keys = (
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER",
            "CMAKE_CXX_COMPILER_VERSION",
            "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_COMPILER_VERSION",
            "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_GENERATOR",
            "GPUXTB_ENABLE_CUDA",
        )
        values: dict[str, str] = {}
        for line in cache_text.splitlines():
            if "=" not in line or ":" not in line:
                continue
            key_with_type, value = line.split("=", 1)
            key = key_with_type.split(":", 1)[0]
            if key in keys:
                values[key] = value
        identity["cmake"] = {
            "cache_path": str(cache),
            "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "values": values,
        }
    else:
        identity["cmake"] = {"status": "not_adjacent_to_library"}
    return identity


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


def _mode_state(
    packed: dict[str, np.ndarray], mode: str
) -> tuple[object, dict[str, object] | None]:
    """Create one persistent batch and its optional caller-owned outputs."""
    import torch
    from gpuxtb.interface import ArrayBatch

    if mode not in ("arena", "out"):
        raise ValueError(f"unknown output mode {mode!r}")
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    if mode == "arena":
        return batch, None
    natoms = len(packed["atomic_numbers"])
    return batch, {
        "energies": torch.empty(1, dtype=torch.float64, device="cuda"),
        "forces": torch.empty((natoms, 3), dtype=torch.float64, device="cuda"),
        "charges": torch.empty(natoms, dtype=torch.float64, device="cuda"),
        "scc_iterations": torch.empty(1, dtype=torch.int32, device="cuda"),
        "scc_converged": torch.empty(1, dtype=torch.uint8, device="cuda"),
        "per_system_status": torch.empty(1, dtype=torch.int32, device="cuda"),
    }


def _release_arena_result(result: object) -> None:
    """Release every producer retain before closing its shared arena."""
    for name in RESULT_NAMES:
        producer = result.get(name)
        if hasattr(producer, "close"):
            producer.close()
    result.close()


def _invoke_mode(batch: object, mode: str, out: dict[str, object] | None) -> None:
    """Run one public call and deterministically release transient results."""
    if mode == "arena":
        result = batch.compute(result_memory="cuda")
        _release_arena_result(result)
    else:
        result = batch.compute(result_memory="host", out=out)
        result.close()


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

    batch, out = _mode_state(packed, mode)
    try:
        for _ in range(warmup):
            _invoke_mode(batch, mode, out)
            torch.cuda.synchronize()

        latencies: list[float] = []
        for _ in range(repetitions):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            _invoke_mode(batch, mode, out)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter_ns() - start) / 1.0e6)
        return latencies
    finally:
        batch.close()


def _measure_pair(
    packed: dict[str, np.ndarray], *, warmup: int, repetitions: int
) -> dict[str, list[float]]:
    """Counterbalance paired arena/out calls to limit temporal order bias."""
    import torch

    states = {mode: _mode_state(packed, mode) for mode in ("arena", "out")}
    try:
        for _ in range(warmup):
            for mode in ("arena", "out"):
                batch, out = states[mode]
                _invoke_mode(batch, mode, out)
                torch.cuda.synchronize()

        samples = {"arena": [], "out": []}
        for index in range(repetitions):
            order = ("arena", "out") if index % 2 == 0 else ("out", "arena")
            for mode in order:
                batch, out = states[mode]
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                _invoke_mode(batch, mode, out)
                torch.cuda.synchronize()
                samples[mode].append((time.perf_counter_ns() - start) / 1.0e6)
        return samples
    finally:
        for batch, _ in states.values():
            batch.close()


def _correctness(packed: dict[str, np.ndarray], mode: str) -> dict[str, object]:
    """Require finite, converged CUDA results within the explicit CPU gate."""
    import torch
    from gpuxtb.interface import ArrayBatch, compute_arrays

    host = compute_arrays(
        **{name: value.copy() for name, value in packed.items()}, backend="cpu"
    )
    host.raise_for_status()
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    natoms = len(packed["atomic_numbers"])
    result = None
    try:
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
            arrays = out
        else:
            result = batch.compute(result_memory="cuda")
            arrays = {
                name: torch.from_dlpack(result.get(name)) for name in RESULT_NAMES
            }

        energies = arrays["energies"].cpu().numpy()
        forces = arrays["forces"].cpu().numpy()
        charges = arrays["charges"].cpu().numpy()
        iterations = arrays["scc_iterations"].cpu().numpy()
        converged = arrays["scc_converged"].cpu().numpy()
        status = arrays["per_system_status"].cpu().numpy()
    finally:
        if result is not None:
            if mode == "arena":
                _release_arena_result(result)
            else:
                result.close()
        batch.close()

    max_force_error = float(np.max(np.abs(forces - host.forces)))
    max_charge_error = float(np.max(np.abs(charges - host.charges)))
    energy_error = float(np.max(np.abs(energies - host.energies)))
    score = max(energy_error, max_force_error, max_charge_error)
    finite = bool(
        np.isfinite(energies).all()
        and np.isfinite(forces).all()
        and np.isfinite(charges).all()
    )
    status_ok = bool(np.all(status == 0) and np.all(converged == 1))
    tolerance_ok = bool(
        energy_error <= CORRECTNESS_ATOL["energy_hartree"]
        and max_force_error <= CORRECTNESS_ATOL["force_hartree_per_bohr"]
        and max_charge_error <= CORRECTNESS_ATOL["charge_electron"]
    )
    if not finite or not status_ok or not tolerance_ok:
        raise RuntimeError(
            f"{mode} correctness gate failed: finite={finite}, "
            f"status={status.tolist()}, converged={converged.tolist()}, "
            f"energy_error={energy_error:.3e}, force_error={max_force_error:.3e}, "
            f"charge_error={max_charge_error:.3e}"
        )
    return {
        "mode": mode,
        "passed": True,
        "finite": finite,
        "per_system_status": status.tolist(),
        "scc_converged": converged.tolist(),
        "scc_iterations": iterations.tolist(),
        "energy_hartree": float(energies[0]),
        "max_force_abs_error": max_force_error,
        "max_charge_abs_error": max_charge_error,
        "energy_abs_error": energy_error,
        "score": score,
        "tolerances": dict(CORRECTNESS_ATOL),
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


def _paired_summary(arena: list[float], out: list[float]) -> dict[str, float | bool]:
    """Summarize counterbalanced per-repetition arena-minus-out deltas."""
    if len(arena) != len(out) or not arena:
        raise ValueError("arena and out samples must be nonempty paired sequences")
    deltas = [
        arena_value - out_value
        for arena_value, out_value in zip(arena, out, strict=True)
    ]
    mean_delta = statistics.fmean(deltas)
    stderr = (
        statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    )
    mean_out = statistics.fmean(out)
    mean_overhead = statistics.fmean(arena) / mean_out - 1.0
    return {
        "count": len(deltas),
        "mean_delta_ms": mean_delta,
        "mean_overhead_fraction": mean_overhead,
        "ci95_mean_delta_ms": 1.96 * stderr,
        "max_mean_overhead_fraction": MAX_MEAN_OVERHEAD_FRACTION,
        "passes_mean_overhead_gate": mean_overhead <= MAX_MEAN_OVERHEAD_FRACTION,
    }


def _cpu_model() -> str | None:
    """Read the host CPU model without relying on a platform-specific command."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _artifact_paths(output: Path) -> tuple[Path, Path]:
    """Return the paired JSON/CSV paths and reject either existing artifact."""
    json_path = output.resolve()
    csv_path = json_path.with_suffix(".csv")
    for path in (json_path, csv_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output {path}")
    return json_path, csv_path


def _write_artifacts(
    document: dict[str, object], output: Path, mode_records: dict[str, object]
) -> tuple[Path, Path]:
    """Exclusively publish paired JSON/CSV evidence without overwriting either."""
    json_path, csv_path = _artifact_paths(output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        created.append(json_path)
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["mode", "sample", "latency_ms"])
            for mode, record in mode_records.items():
                for index, latency in enumerate(record["raw_latency_ms"]):
                    writer.writerow([mode, index, f"{latency:.6f}"])
        created.append(csv_path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return json_path, csv_path


def main() -> int:
    """Run the arena-vs-out allocation benchmark and write JSON+CSV evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=60)
    parser.add_argument("--profile-mode", choices=("arena", "out"))
    args = parser.parse_args()

    if args.warmup < 0:
        raise SystemExit("--warmup must be nonnegative")
    if args.repetitions < 2:
        raise SystemExit("--repetitions must be at least 2")
    if args.profile_mode is None and args.output is None:
        raise SystemExit("--output is required unless --profile-mode is selected")
    if args.profile_mode is not None and args.output is not None:
        raise SystemExit("--output is not used with --profile-mode")
    if args.output is not None:
        _artifact_paths(args.output)

    revision, dirty = _git_revision()
    if dirty:
        raise SystemExit("dirty sources cannot produce final benchmark evidence")
    library_path = args.library.resolve()
    if not library_path.is_file():
        raise SystemExit(f"--library does not name a regular file: {library_path}")
    # The package loads lazily, so the explicit benchmark library is selected
    # before any gpuxtb import can cache a different shared object.
    os.environ["GPUXTB_LIBRARY"] = str(library_path)

    _require_gpu()
    packed = _packed_water()

    if args.profile_mode is not None:
        latencies = _measure(
            packed,
            args.profile_mode,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        summary = _summary(latencies)
        print(  # noqa: T201 - benchmark CLI progress output
            f"{args.profile_mode}: mean {summary['mean_ms']:.4f} ms, "
            f"median {summary['median_ms']:.4f} ms over {summary['count']} samples"
        )
        return 0

    import torch

    correctness_before = {mode: _correctness(packed, mode) for mode in ("arena", "out")}
    samples = _measure_pair(packed, warmup=args.warmup, repetitions=args.repetitions)
    correctness_after = {mode: _correctness(packed, mode) for mode in ("arena", "out")}
    mode_records: dict[str, object] = {}
    for mode in ("arena", "out"):
        mode_records[mode] = {
            "raw_latency_ms": samples[mode],
            "summary": _summary(samples[mode]),
            "correctness_before": correctness_before[mode],
            "correctness_after": correctness_after[mode],
        }
    paired = _paired_summary(samples["arena"], samples["out"])

    document = {
        "schema_version": 2,
        "claim": (
            "On the recorded machine and workload, the mean public-call latency "
            "of result_memory='cuda' is at most 5 percent above caller-owned out= "
            "under a counterbalanced paired protocol."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_status": "clean",
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "gpu": _gpu_identity(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "environment": {
            key: os.environ.get(key, "") for key in THREAD_ENVIRONMENT_NAMES
        },
        "library": _library_identity(),
        "workload": {
            "molecule": "water (8,1,1)",
            "nsystems": 1,
            "natoms": 3,
            "descriptor_mode": "host numpy inputs",
            "stream": "legacy default (stream=1)",
            "properties": "energy,forces,charges,SCC diagnostics",
            "compute_options": {
                "max_scc_iterations": 250,
                "charge_tolerance": 1.0e-6,
                "energy_tolerance": 1.0e-8,
                "electronic_temperature_kelvin": 300.0,
            },
        },
        "timing": {
            "boundary": (
                "perf_counter_ns around each public ArrayBatch.compute() with "
                "torch.cuda.synchronize() before start and after stop"
            ),
            "warmup_calls": args.warmup,
            "repetitions": args.repetitions,
            "ordering": "counterbalanced AB/BA per paired repetition",
            "paired_comparison": paired,
            "modes": mode_records,
        },
    }

    json_path, csv_path = _write_artifacts(document, args.output, mode_records)

    for mode, record in mode_records.items():
        summary = record["summary"]
        print(  # noqa: T201 - benchmark CLI progress output
            f"{mode}: mean {summary['mean_ms']:.4f} ms, "
            f"median {summary['median_ms']:.4f} ms, "
            f"min {summary['min_ms']:.4f} ms, max {summary['max_ms']:.4f} ms "
            f"over {summary['count']} samples"
        )
    print(f"wrote {json_path}")  # noqa: T201 - benchmark CLI progress output
    print(f"wrote {csv_path}")  # noqa: T201 - benchmark CLI progress output
    return 0 if paired["passes_mean_overhead_gate"] else 2


_HOST_GPU_CACHE: dict[str, object] = {}


def _gpu_identity() -> dict[str, object]:
    """Return GPU/driver identity queried through the active CUDA runtime."""

    def _probe() -> dict[str, object]:
        import torch

        candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
        major = (torch.version.cuda or "").split(".", 1)[0]
        if major:
            candidates.append(f"libcudart.so.{major}")
        cudart = None
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                cudart = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if cudart is None:
            raise RuntimeError(
                "cannot load the active CUDA runtime for identity capture"
            )
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
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
            "logical_device_id": device_id,
        }

    if not _HOST_GPU_CACHE:
        _HOST_GPU_CACHE["value"] = _probe()
    return _HOST_GPU_CACHE["value"]


if __name__ == "__main__":
    raise SystemExit(main())
