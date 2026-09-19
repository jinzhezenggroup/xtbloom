#!/usr/bin/env python3
"""Correctness-qualified real-structure timing for Experiments 1, 2 and 3-B.

The runner reuses xTBloom's benchmark adapters.  Inputs are distinct systems
from a frozen performance manifest; setup stays outside samples; every xTBloom
sample uses public FRESH semantics; CUDA timings end at an explicit synchronize.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

try:
    import resource
except (
    ImportError
):  # CLI/help remains inspectable on Windows; experiments are Linux-only.
    resource = None  # type: ignore[assignment]


class PerformanceError(RuntimeError):
    pass


def exception_category(exc: BaseException) -> str:
    """Classify capacity failures without depending on one backend's wording."""
    current: BaseException | None = exc
    seen: set[int] = set()
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}".lower())
        if (
            isinstance(current, MemoryError)
            or "outofmemory" in type(current).__name__.lower()
        ):
            return "oom"
        current = current.__cause__ or current.__context__
    combined = " | ".join(messages)
    oom_markers = (
        "out of memory",
        "allocation failed",
        "xtbloom_status_allocation_failed",
        "cuda_error_out_of_memory",
        "cublas_status_alloc_failed",
    )
    if any(marker in combined for marker in oom_markers):
        return "oom"
    return "unavailable" if "unavailable" in combined else "error"


def csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def parse_bins(value: str) -> tuple[tuple[int, int | None, str], ...]:
    bins = []
    for token in value.split(","):
        low_text, high_text = token.split("-", 1)
        low = int(low_text)
        high = None if high_text == "inf" else int(high_text)
        if low < 1 or (high is not None and high < low):
            raise argparse.ArgumentTypeError(f"invalid AO bin: {token}")
        bins.append((low, high, token))
    return tuple(bins)


def ao_count(item: Any) -> int:
    value = item.system.manifest.get("gfn2_n_ao") or item.system.strata.get("gfn2_n_ao")
    if value in (None, ""):
        raise PerformanceError(f"{item.system_id} lacks frozen gfn2_n_ao")
    return int(value)


def geometric_pair_counts(items: list[Any]) -> dict[str, Any]:
    """Count distinct atom pairs at the production GFN2 neighbor cutoffs."""
    cutoffs = (25.0, 30.0, 50.0)
    values: dict[str, list[int]] = {f"{cutoff:g}": [] for cutoff in cutoffs}
    for item in items:
        positions = item.system.positions_bohr
        counts = dict.fromkeys(cutoffs, 0)
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                distance2 = sum(
                    (float(positions[left][axis]) - float(positions[right][axis])) ** 2
                    for axis in range(3)
                )
                for cutoff in cutoffs:
                    if distance2 <= cutoff * cutoff:
                        counts[cutoff] += 1
        for cutoff in cutoffs:
            values[f"{cutoff:g}"].append(counts[cutoff])
    return {
        "definition": "distinct i<j geometric pairs with distance <= cutoff; nonperiodic molecular inputs; cutoffs in bohr",
        "per_system_by_cutoff_bohr": values,
        "batch_total_by_cutoff_bohr": {
            cutoff: sum(counts) for cutoff, counts in values.items()
        },
    }


def select_systems(
    items: list[Any], ao_bin: tuple[int, int | None, str], count: int, seed: str
) -> list[Any]:
    low, high, _ = ao_bin
    eligible = [
        item
        for item in items
        if item.system is not None
        and ao_count(item) >= low
        and (high is None or ao_count(item) <= high)
    ]
    eligible.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}\0{item.system_id}".encode()
        ).hexdigest()
    )
    if len(eligible) < count:
        raise PerformanceError(
            f"AO bin {ao_bin[2]} has {len(eligible)} systems, needs {count}"
        )
    return eligible[:count]


def controlled_wide(
    items: list[Any],
    bins: tuple[tuple[int, int | None, str], ...],
    count: int,
    seed: str,
) -> list[Any]:
    buckets = []
    for ao_bin in bins:
        low, high, _ = ao_bin
        bucket = [
            item
            for item in items
            if item.system is not None
            and ao_count(item) >= low
            and (high is None or ao_count(item) <= high)
        ]
        bucket.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}\0wide\0{item.system_id}".encode()
            ).hexdigest()
        )
        buckets.append(bucket)
    result = []
    cursor = 0
    while len(result) < count:
        progressed = False
        for bucket in buckets:
            if cursor < len(bucket):
                result.append(bucket[cursor])
                progressed = True
                if len(result) == count:
                    break
        if not progressed:
            raise PerformanceError(
                f"controlled-wide selection has only {len(result)} systems, needs {count}"
            )
        cursor += 1
    return result


def rss_bytes() -> int:
    return (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if resource is not None
        else 0
    )


def current_rss_bytes() -> int:
    """Read current RSS, not the inherited process-lifetime ru_maxrss."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class NvmlProcessMemory:
    """Minimal NVML binding for per-process resident GPU memory."""

    class ProcessInfo(ctypes.Structure):
        _fields_ = [
            ("pid", ctypes.c_uint),
            ("used_gpu_memory", ctypes.c_ulonglong),
            ("gpu_instance_id", ctypes.c_uint),
            ("compute_instance_id", ctypes.c_uint),
        ]

    def __init__(self, visible_cuda_ordinal: int):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        self.library.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.library.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByPciBusId_v2.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByPciBusId_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetComputeRunningProcesses_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(self.ProcessInfo),
        ]
        self.library.nvmlDeviceGetComputeRunningProcesses_v3.restype = ctypes.c_int
        if self.library.nvmlInit_v2() != 0:
            raise PerformanceError("NVML initialization failed")
        self.visible_cuda_ordinal = visible_cuda_ordinal
        resolved_bus = os.environ.get("PAPER_RESOLVED_GPU_PCI_BUS_ID")
        if resolved_bus:
            handle = ctypes.c_void_p()
            if (
                self.library.nvmlDeviceGetHandleByPciBusId_v2(
                    resolved_bus.encode("ascii"), ctypes.byref(handle)
                )
                != 0
            ):
                self.library.nvmlShutdown()
                raise PerformanceError(
                    f"NVML cannot open resolved CUDA PCI device {resolved_bus}"
                )
            self.handles = [handle]
            return
        count = ctypes.c_uint()
        if self.library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            self.library.nvmlShutdown()
            raise PerformanceError("NVML cannot enumerate physical devices")
        self.handles = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if (
                self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))
                != 0
            ):
                self.library.nvmlShutdown()
                raise PerformanceError(f"NVML cannot open physical device {index}")
            self.handles.append(handle)

    def used_bytes(self, pid: int) -> int | None:
        values = []
        for handle in self.handles:
            count = ctypes.c_uint(0)
            status = self.library.nvmlDeviceGetComputeRunningProcesses_v3(
                handle, ctypes.byref(count), None
            )
            if status not in {0, 7}:  # success or NVML_ERROR_INSUFFICIENT_SIZE
                raise PerformanceError(
                    f"NVML process query failed with status {status}"
                )
            capacity = max(8, int(count.value) + 8)
            entries = (self.ProcessInfo * capacity)()
            count = ctypes.c_uint(capacity)
            status = self.library.nvmlDeviceGetComputeRunningProcesses_v3(
                handle, ctypes.byref(count), entries
            )
            if status != 0:
                raise PerformanceError(
                    f"NVML process query failed with status {status}"
                )
            for entry in entries[: count.value]:
                if int(entry.pid) == pid:
                    value = int(entry.used_gpu_memory)
                    if value != (1 << 64) - 1:
                        values.append(value)
        return sum(values) if values else None

    def close(self) -> None:
        self.library.nvmlShutdown()


class ResourceProbe:
    """Sample row-local current RSS and per-process VRAM outside timings."""

    def __init__(self, device_id: int | None, interval_seconds: float = 0.002):
        self.visible_cuda_ordinal = device_id
        self.interval_seconds = interval_seconds
        self.started_ns = 0
        self.samples: list[dict[str, int | None]] = []
        self.sample_count_total = 0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.nvml = None
        if device_id is not None:
            try:
                self.nvml = NvmlProcessMemory(device_id)
            except Exception as exc:  # noqa: BLE001 - unavailable must be recorded
                self.error = f"{type(exc).__name__}: {exc}"
        self.thread = threading.Thread(
            target=self._loop, name="paper-resource-probe", daemon=True
        )

    def _sample(self) -> None:
        try:
            gpu = self.nvml.used_bytes(os.getpid()) if self.nvml is not None else None
            row = {
                "elapsed_ns": time.perf_counter_ns() - self.started_ns,
                "current_rss_bytes": current_rss_bytes(),
                "process_gpu_memory_bytes": gpu,
            }
            self.sample_count_total += 1
            if (
                not self.samples
                or row["current_rss_bytes"] != self.samples[-1]["current_rss_bytes"]
                or row["process_gpu_memory_bytes"]
                != self.samples[-1]["process_gpu_memory_bytes"]
                or self.sample_count_total % 100 == 0
            ):
                self.samples.append(row)
        except Exception as exc:  # noqa: BLE001 - preserve monitoring failure
            self.error = f"{type(exc).__name__}: {exc}"

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self.started_ns = time.perf_counter_ns()
        self._sample()
        self.thread.start()

    def sample_now(self) -> None:
        self._sample()

    def stop(self, backend: str, invocations: int) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join()
        self._sample()
        if self.nvml is not None:
            self.nvml.close()
        rss_values = [int(row["current_rss_bytes"] or 0) for row in self.samples]
        gpu_values = [
            int(row["process_gpu_memory_bytes"])
            for row in self.samples
            if row["process_gpu_memory_bytes"] is not None
        ]
        status = "available"
        reason = None
        if not rss_values or max(rss_values) == 0:
            status, reason = "unavailable", "current RSS sampling failed"
        elif backend == "cuda" and (self.error is not None or not gpu_values):
            status, reason = (
                "unavailable",
                self.error or "NVML did not report this process",
            )
        return {
            "status": status,
            "reason": reason,
            "scope": "separate untimed adapter setup plus steady-state probe for this row and process",
            "probe_invocations": invocations,
            "sampling_interval_seconds": self.interval_seconds,
            "visible_cuda_ordinal": self.visible_cuda_ordinal,
            "sample_count": self.sample_count_total,
            "retained_raw_sample_count": len(self.samples),
            "baseline_current_rss_bytes": rss_values[0] if rss_values else None,
            "sampled_peak_current_rss_bytes": max(rss_values) if rss_values else None,
            "sampled_incremental_rss_bytes": max(rss_values) - rss_values[0]
            if rss_values
            else None,
            "sampled_peak_process_gpu_memory_bytes": max(gpu_values)
            if gpu_values
            else None,
            "monitor_error": self.error,
            "raw_samples": self.samples,
        }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return (
        ordered[lo]
        if lo == hi
        else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
    )


def bootstrap_median_ci(samples: list[float], draws: int, seed: str) -> list[float]:
    generator = random.Random(seed)
    medians = []
    for _ in range(draws):
        medians.append(statistics.median(generator.choice(samples) for _ in samples))
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def summary(
    samples: list[float], systems: int, draws: int, seed: str
) -> dict[str, Any]:
    median_ms = statistics.median(samples)
    return {
        "raw_ms": samples,
        "count": len(samples),
        "median_ms": median_ms,
        "bootstrap_95_ci_median_ms": bootstrap_median_ci(samples, draws, seed),
        "bootstrap_draws": draws,
        "iqr_ms": percentile(samples, 0.75) - percentile(samples, 0.25),
        "p95_ms": percentile(samples, 0.95),
        "median_systems_per_second": systems / (median_ms * 1e-3),
    }


def load_reference(root: Path | None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if root is None:
        return {}
    references: dict[tuple[str, str, str], dict[str, Any]] = {}
    paths = [root] if root.is_file() else sorted(root.rglob("results.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["status"]["category"] != "success":
                    continue
                key = (
                    row["input"]["dataset"],
                    row["input"]["system_id"],
                    row["program"]["engine"],
                )
                if key in references:
                    raise PerformanceError(f"duplicate correctness reference: {key}")
                references[key] = row
    return references


def field(row: dict[str, Any], name: str) -> Any | None:
    value = row["results"][name]
    return value.get("value") if value.get("availability") == "available" else None


def flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for child in value:
            result.extend(flatten(child))
        return result
    return [float(value)]


def qualify(
    selected: list[Any],
    outputs: dict[str, Any],
    references: dict[tuple[str, str, str], dict[str, Any]],
    engine: str,
    energy_atol: float,
    force_atol: float,
    charge_atol: float,
    property_name: str,
    require: bool,
    allow_system_failures: bool = False,
    require_charges: bool = False,
) -> dict[str, Any]:
    total_atoms = sum(len(item.system.atomic_numbers) for item in selected)
    energies = outputs.get("energies_hartree")
    forces = outputs.get("forces_hartree_per_bohr")
    charges = outputs.get("atomic_charges_e")
    statuses = outputs.get("per_system_status")
    converged = outputs.get("scc_converged")
    contract_failures: list[str] = []

    def require_vector(
        name: str, values: Any, size: int, *, require_finite: bool = True
    ) -> None:
        if not isinstance(values, (list, tuple)):
            contract_failures.append(f"{name}: missing or not a sequence")
            return
        if len(values) != size:
            contract_failures.append(f"{name}: length={len(values)} expected={size}")
            return
        if require_finite:
            try:
                finite = all(math.isfinite(float(value)) for value in values)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                contract_failures.append(
                    f"{name}: contains non-finite or non-numeric values"
                )

    # A failed xTBloom ragged slice is required by the public ABI to contain
    # NaNs.  The ragged experiment therefore validates finiteness per system;
    # strict all-success suites retain the cheaper whole-vector check.
    require_vector(
        "energies_hartree",
        energies,
        len(selected),
        require_finite=not allow_system_failures,
    )
    if property_name == "force":
        require_vector(
            "forces_hartree_per_bohr",
            forces,
            3 * total_atoms,
            require_finite=not allow_system_failures,
        )
    if require_charges:
        require_vector(
            "atomic_charges_e",
            charges,
            total_atoms,
            require_finite=not allow_system_failures,
        )
    if engine.startswith("xtbloom"):
        require_vector("per_system_status", statuses, len(selected))
        require_vector("scc_converged", converged, len(selected))
    if contract_failures:
        message = "; ".join(contract_failures)
        if require:
            raise PerformanceError(f"output contract failed: {message}")
        return {
            "status": "unavailable",
            "paired_systems": 0,
            "missing_reference_ids": [],
            "successful_systems": 0,
            "failed_systems": len(selected),
            "system_outcomes": [],
            "failure_aware": allow_system_failures,
            "performance_claim_eligible": False,
            "output_contract_failures": contract_failures,
            "max_abs_energy_error_hartree": None,
            "energy_atol_hartree": energy_atol,
            "max_abs_force_error_hartree_per_bohr": None,
            "force_atol_hartree_per_bohr": force_atol
            if property_name == "force"
            else None,
            "max_abs_charge_error_e": None,
            "charge_atol_e": charge_atol if require_charges else None,
        }
    assert energies is not None
    statuses = statuses if statuses is not None else [0] * len(selected)
    converged = converged if converged is not None else [1] * len(selected)
    max_energy = 0.0
    max_force = 0.0
    max_charge = 0.0
    paired = 0
    successful = 0
    atom_cursor = 0
    missing: list[str] = []
    outcomes: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        atoms = len(item.system.atomic_numbers)
        ok = int(statuses[index]) == 0 and int(converged[index]) == 1
        successful += int(ok)
        force_slice = (
            list(forces[3 * atom_cursor : 3 * (atom_cursor + atoms)])
            if forces is not None
            else []
        )
        charge_slice = (
            list(charges[atom_cursor : atom_cursor + atoms])
            if charges is not None
            else []
        )
        outcomes.append(
            {
                "system_id": item.system_id,
                "status": int(statuses[index]),
                "scc_converged": int(converged[index]),
                "successful": ok,
            }
        )
        if allow_system_failures and not ok:
            failed_values: list[tuple[str, list[Any]]] = [("energy", [energies[index]])]
            if property_name == "force":
                failed_values.append(("forces", force_slice))
            if require_charges:
                failed_values.append(("atomic_charges", charge_slice))
            for quantity, values in failed_values:
                try:
                    valid_nan_slice = bool(values) and all(
                        math.isnan(float(value)) for value in values
                    )
                except (TypeError, ValueError):
                    valid_nan_slice = False
                if not valid_nan_slice:
                    contract_failures.append(
                        f"{item.system_id}:{quantity}: failed-system slice is not entirely NaN"
                    )
            atom_cursor += atoms
            continue
        reference_engine = (
            "xtb" if engine.startswith("xtbloom") or engine == "xtb" else engine
        )
        reference = references.get((item.dataset, item.system_id, reference_engine))
        if reference is None:
            missing.append(f"{item.system_id}:{reference_engine}")
        elif ok:
            ref_energy = field(reference, "energy")
            ref_forces = field(reference, "forces")
            if ref_energy is not None:
                values = (float(energies[index]), float(ref_energy))
                if not all(math.isfinite(value) for value in values):
                    raise PerformanceError(f"non-finite energy for {item.system_id}")
                max_energy = max(max_energy, abs(values[0] - values[1]))
            else:
                missing.append(f"{item.system_id}:{reference_engine}:energy")
            if (
                property_name == "force"
                and forces is not None
                and ref_forces is not None
            ):
                actual = force_slice
                expected = flatten(ref_forces)
                if len(actual) != 3 * atoms or len(expected) != 3 * atoms:
                    raise PerformanceError(f"force shape mismatch for {item.system_id}")
                if not all(math.isfinite(float(value)) for value in actual + expected):
                    raise PerformanceError(f"non-finite force for {item.system_id}")
                max_force = max(
                    max_force,
                    max(abs(a - b) for a, b in zip(actual, expected, strict=True)),
                )
            elif property_name == "force":
                missing.append(f"{item.system_id}:{reference_engine}:forces")
            if require_charges and charges is not None:
                charge_reference = references.get(
                    (item.dataset, item.system_id, "tblite")
                )
                ref_charges = (
                    field(charge_reference, "atomic_charges")
                    if charge_reference is not None
                    else None
                )
                if ref_charges is None:
                    missing.append(f"{item.system_id}:tblite:atomic_charges")
                else:
                    actual_charges = charge_slice
                    expected_charges = flatten(ref_charges)
                    if len(actual_charges) != atoms or len(expected_charges) != atoms:
                        raise PerformanceError(
                            f"charge shape mismatch for {item.system_id}"
                        )
                    if not all(
                        math.isfinite(float(value))
                        for value in actual_charges + expected_charges
                    ):
                        raise PerformanceError(
                            f"non-finite charge for {item.system_id}"
                        )
                    max_charge = max(
                        max_charge,
                        max(
                            abs(a - b)
                            for a, b in zip(
                                actual_charges, expected_charges, strict=True
                            )
                        ),
                    )
            if not any(entry.startswith(f"{item.system_id}:") for entry in missing):
                paired += 1
        atom_cursor += atoms
    if contract_failures:
        message = "; ".join(contract_failures)
        if require:
            raise PerformanceError(f"output contract failed: {message}")
    numerical_passed = (
        max_energy <= energy_atol
        and (property_name != "force" or max_force <= force_atol)
        and (not require_charges or max_charge <= charge_atol)
    )
    success_requirement = (
        paired == successful if allow_system_failures else successful == len(selected)
    )
    passed = not contract_failures and success_requirement and numerical_passed
    claim_eligible = passed and successful > 0
    if require and (missing or not passed):
        raise PerformanceError(
            f"correctness qualification failed: missing={len(missing)} successful={successful}/{len(selected)} "
            f"energy={max_energy:g} force={max_force:g} charge={max_charge:g}"
        )
    return {
        "status": "pass"
        if passed and not missing
        else "unavailable"
        if missing
        else "fail",
        "paired_systems": paired,
        "missing_reference_ids": missing,
        "successful_systems": successful,
        "failed_systems": len(selected) - successful,
        "system_outcomes": outcomes,
        "failure_aware": allow_system_failures,
        "performance_claim_eligible": claim_eligible,
        "output_contract_failures": contract_failures,
        "max_abs_energy_error_hartree": max_energy,
        "energy_atol_hartree": energy_atol,
        "max_abs_force_error_hartree_per_bohr": max_force
        if property_name == "force"
        else None,
        "force_atol_hartree_per_bohr": force_atol if property_name == "force" else None,
        "max_abs_charge_error_e": max_charge if require_charges else None,
        "charge_atol_e": charge_atol if require_charges else None,
    }


class SharedContextSequential:
    """Prebuild batch-1 descriptors while sharing exactly one native context."""

    def __init__(
        self,
        run: Any,
        dataset_runner: Any,
        systems: list[Any],
        library: Path,
        property_name: str,
        cpu_threads: int,
        max_scc_iterations: int,
        electronic_temperature_hartree: float,
        collect_atomic_charges: bool = False,
    ):
        self.adapters = []
        self.shared_context = None
        self.options = None
        for item in systems:
            storage = dataset_runner.storage_from_systems([item.system])
            cell = run.Cell("xtbloom", "cpu", "host", "paper-real", property_name, 1)
            adapter = run.XTBloomAdapter.from_storage(
                library,
                storage,
                cell,
                0,
                cpu_threads,
                collect_atomic_charges=collect_atomic_charges,
                max_scc_iterations=max_scc_iterations,
                electronic_temperature_hartree=electronic_temperature_hartree,
            )
            if self.shared_context is None:
                self.shared_context = adapter.context
                self.options = adapter.options
            else:
                adapter.library.xtbloom_context_destroy(adapter.context)
                adapter.context = self.shared_context
            self.adapters.append(adapter)

    def invoke(self) -> None:
        for adapter in self.adapters:
            adapter.invoke()
            adapter.synchronize()

    def results(self) -> dict[str, Any]:
        energies: list[float] = []
        forces: list[float] = []
        iterations: list[int] = []
        converged: list[int] = []
        statuses: list[int] = []
        charges: list[float] = []
        for adapter in self.adapters:
            result = adapter.raw_results()
            energies.extend(result["energies_hartree"])
            forces.extend(result.get("forces_hartree_per_bohr", []))
            iterations.extend(result["scc_iterations"])
            converged.extend(result["scc_converged"])
            statuses.extend(result["per_system_status"])
            charges.extend(result.get("atomic_charges_e", []))
        output = {
            "energies_hartree": energies,
            "scc_iterations": iterations,
            "scc_converged": converged,
            "per_system_status": statuses,
        }
        if forces:
            output["forces_hartree_per_bohr"] = forces
        if charges:
            output["atomic_charges_e"] = charges
        return output

    def memory_snapshot(self) -> dict[str, Any]:
        return {"host_process_hwm_bytes": rss_bytes()}

    def close(self) -> None:
        for adapter in self.adapters:
            adapter.memory.close()
            if adapter.owns_cuda_control and adapter.cuda_control is not None:
                adapter.cuda_control.close()
        if self.adapters and self.shared_context is not None:
            self.adapters[0].library.xtbloom_context_destroy(self.shared_context)


def memory_probe_worker(connection: Any, request: dict[str, Any]) -> None:
    """Measure one cell in a fresh process so allocators cannot leak across rows."""
    adapter: Any | None = None
    monitor: ResourceProbe | None = None
    try:
        if request["cpu_affinity"] and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(request["cpu_affinity"]))
        from paper_runtime import install

        install(Path(request["repo"]))
        import dataset_runner  # type: ignore
        import run as benchmark_run  # type: ignore
        from dxtb_adapter import DxtbAdapter  # type: ignore
        from tblite_adapter import TbliteAdapter  # type: ignore
        from xtb_adapter import XtbAdapter  # type: ignore

        items = [
            item
            for item in dataset_runner.load_manifest(
                Path(request["manifest"]), request["dataset"]
            )
            if item.system is not None and item.subset == request["subset"]
        ]
        lookup = {item.system_id: item for item in items}
        if len(lookup) != len(items):
            raise PerformanceError(
                "memory probe manifest contains duplicate system IDs"
            )
        missing = [
            system_id for system_id in request["system_ids"] if system_id not in lookup
        ]
        if missing:
            raise PerformanceError(
                f"memory probe cannot reload selected IDs: {missing[:8]}"
            )
        selected = [lookup[system_id] for system_id in request["system_ids"]]
        storage = dataset_runner.storage_from_systems(
            [item.system for item in selected]
        )
        engine = request["engine"]
        backend = request["backend"]
        memory = request["memory"]
        threads = request["threads"]
        property_name = request["property"]
        max_iterations = request["max_scc_iterations"]
        temperature_k = request["electronic_temperature_kelvin"]
        temperature_hartree = temperature_k * 3.166808578545117e-6
        batch_size = len(selected)

        monitor = ResourceProbe(request["device_id"] if backend == "cuda" else None)
        monitor.start()
        setup_start = time.perf_counter_ns()
        if engine == "xtbloom-sequential":
            adapter = SharedContextSequential(
                benchmark_run,
                dataset_runner,
                selected,
                Path(request["cpu_library"]),
                property_name,
                threads,
                max_iterations,
                temperature_hartree,
                False,
            )
            invoke = adapter.invoke
        elif engine == "xtbloom-ragged":
            library = (
                request["cpu_library"] if backend == "cpu" else request["cuda_library"]
            )
            cell = benchmark_run.Cell(
                "xtbloom", backend, memory, "paper-real", property_name, batch_size
            )
            adapter = benchmark_run.XTBloomAdapter.from_storage(
                Path(library),
                storage,
                cell,
                request["device_id"],
                threads,
                collect_atomic_charges=False,
                max_scc_iterations=max_iterations,
                electronic_temperature_hartree=temperature_hartree,
            )
            invoke = lambda adapter=adapter: (
                adapter.invoke(),
                adapter.synchronize(),
            )
        elif engine == "xtb":
            adapter = XtbAdapter(
                Path(request["xtb_library"]),
                storage,
                property_name,
                None,
                accuracy=request["accuracy"],
                max_iterations=max_iterations,
                electronic_temperature_kelvin=temperature_k,
                threads=threads,
            )
            invoke = lambda adapter=adapter: (
                adapter.restart_scc(),
                adapter.invoke(),
            )
        elif engine == "tblite":
            adapter = TbliteAdapter(
                Path(request["tblite_library"]),
                storage,
                property_name,
                accuracy=request["accuracy"],
                max_iterations=max_iterations,
                electronic_temperature_hartree=temperature_hartree,
                collect_atomic_charges=False,
                threads=threads,
            )
            invoke = lambda adapter=adapter: (
                adapter.restart_scc(),
                adapter.invoke(),
            )
        else:
            adapter = DxtbAdapter(
                storage,
                property_name,
                backend,
                request["device_id"],
                1,
                Path(request["dxtb_source"]),
                force_convergence=True,
                accuracy=request["accuracy"],
                max_iterations=max_iterations,
            )
            invoke = lambda adapter=adapter: (
                adapter.invoke(),
                adapter.synchronize(),
            )
        setup_ms = (time.perf_counter_ns() - setup_start) * 1e-6
        for _ in range(request["probe_invocations"]):
            invoke()
        monitor.sample_now()
        adapter.close()
        adapter = None
        memory_result = monitor.stop(backend, request["probe_invocations"])
        monitor = None
        memory_result["probe_setup_ms"] = setup_ms
        memory_result["isolation"] = "fresh spawned process per performance cell"
        connection.send({"state": "ok", "memory": memory_result})
    except BaseException as exc:  # noqa: BLE001 - child must preserve every failure
        connection.send({"state": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            if monitor is not None:
                monitor.stop(
                    request.get("backend", "cpu"), request.get("probe_invocations", 0)
                )
            connection.close()


def isolated_memory_probe(
    request: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    """Run the memory-only reconstruction with a bounded spawn lifecycle."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=memory_probe_worker, args=(child, request))
    process.start()
    child.close()
    if not parent.poll(timeout_seconds):
        process.terminate()
        process.join()
        raise PerformanceError(
            f"isolated memory probe timed out after {timeout_seconds} seconds"
        )
    try:
        message = parent.recv()
    except EOFError as exc:
        process.join()
        raise PerformanceError(
            f"isolated memory probe exited without a record: {process.exitcode}"
        ) from exc
    finally:
        parent.close()
    process.join()
    if process.exitcode != 0:
        raise PerformanceError(f"isolated memory probe exited with {process.exitcode}")
    if message.get("state") != "ok":
        raise PerformanceError(f"isolated memory probe failed: {message.get('error')}")
    return message["memory"]


def timed(call: Callable[[], None]) -> tuple[float, float]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    call()
    return (
        (time.perf_counter_ns() - wall_start) * 1e-6,
        (time.process_time_ns() - cpu_start) * 1e-6,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qm9", "omol25"), required=True)
    parser.add_argument("--subset", default="performance")
    parser.add_argument("--cpu-library", type=Path)
    parser.add_argument("--cuda-library", type=Path)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument(
        "--suite",
        choices=(
            "cpu-native",
            "pool-baseline",
            "gpu-crossover",
            "capacity",
            "ragged",
            "energy-only",
            "second-hardware",
            "mixed-spot",
        ),
        required=True,
    )
    parser.add_argument("--ao-bins", type=parse_bins, required=True)
    parser.add_argument("--batch-sizes", type=csv_ints, required=True)
    parser.add_argument("--cpu-threads", type=csv_ints, default=(1,))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--property", choices=("energy", "force"), default="force")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--selection-mode",
        choices=("by-bin", "controlled-wide", "real-order"),
        default="by-bin",
    )
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--require-correctness", action="store_true")
    parser.add_argument("--energy-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=5e-6)
    parser.add_argument("--charge-atol", type=float, default=5e-7)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    parser.add_argument("--memory-probe-timeout-seconds", type=int, default=7200)
    parser.add_argument("--require-dxtb", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.energy_atol > 5e-7 or args.force_atol > 5e-6 or args.charge_atol > 5e-7:
        raise PerformanceError("requested correctness gate widens the paper contract")
    if args.max_scc_iterations <= 0:
        raise PerformanceError("max SCC iterations must be positive")
    if not math.isfinite(args.accuracy) or args.accuracy <= 0.0:
        raise PerformanceError("accuracy must be finite and positive")
    if (
        not math.isfinite(args.electronic_temperature_kelvin)
        or args.electronic_temperature_kelvin < 0.0
    ):
        raise PerformanceError("electronic temperature must be finite and nonnegative")
    if args.memory_probe_timeout_seconds <= 0:
        raise PerformanceError("memory probe timeout must be positive")
    electronic_temperature_hartree = (
        args.electronic_temperature_kelvin * 3.166808578545117e-6
    )
    if args.require_dxtb and args.suite not in {"gpu-crossover", "second-hardware"}:
        raise PerformanceError(
            "--require-dxtb is only valid for suites that contain dxtb"
        )
    if args.output.exists():
        raise PerformanceError(f"refusing to overwrite: {args.output}")
    for path in (args.repo, args.manifest):
        if not path.exists():
            raise PerformanceError(f"required path is missing: {path}")
    if args.cpu_library is not None and not args.cpu_library.is_file():
        raise PerformanceError(f"CPU library is missing: {args.cpu_library}")
    if args.cuda_library is not None and not args.cuda_library.is_file():
        raise PerformanceError(f"CUDA library is missing: {args.cuda_library}")

    from paper_runtime import install, physical_cpu_ids

    install(args.repo)
    import dataset_runner  # type: ignore
    import run as benchmark_run  # type: ignore
    from dxtb_adapter import DxtbAdapter  # type: ignore
    from tblite_adapter import TbliteAdapter  # type: ignore
    from xtb_adapter import XtbAdapter  # type: ignore

    all_items = [
        item
        for item in dataset_runner.load_manifest(args.manifest, args.dataset)
        if item.system is not None and item.subset == args.subset
    ]
    references = load_reference(args.reference_root)
    rows = []
    allowed_cpus = physical_cpu_ids()

    def variants(threads: int) -> Iterable[tuple[str, str, str]]:
        if args.suite == "cpu-native":
            yield from (
                ("xtbloom-sequential", "cpu", "host"),
                ("xtbloom-ragged", "cpu", "host"),
                ("xtb", "cpu", "host"),
                ("tblite", "cpu", "host"),
            )
        elif args.suite == "pool-baseline":
            yield ("xtbloom-ragged", "cpu", "host")
        elif args.suite == "gpu-crossover":
            yield from (
                ("xtbloom-ragged", "cpu", "host"),
                ("xtbloom-ragged", "cuda", "host"),
                ("xtbloom-ragged", "cuda", "device"),
                ("dxtb", "cuda", "device"),
            )
        elif args.suite == "second-hardware":
            yield from (
                ("xtbloom-ragged", "cuda", "host"),
                ("xtbloom-ragged", "cuda", "device"),
                ("dxtb", "cuda", "device"),
            )
        elif args.suite == "capacity":
            yield ("xtbloom-ragged", "cuda", "device")
        elif args.suite == "ragged" or args.suite == "energy-only":
            if args.cpu_library is not None:
                yield from (
                    ("xtbloom-sequential", "cpu", "host"),
                    ("xtbloom-ragged", "cpu", "host"),
                )
            if args.cuda_library is not None:
                yield from (
                    ("xtbloom-ragged", "cuda", "host"),
                    ("xtbloom-ragged", "cuda", "device"),
                )
        elif args.suite == "mixed-spot":
            yield ("xtbloom-ragged", "cuda", "mixed")

    def construct_adapter(
        engine: str,
        backend: str,
        memory: str,
        threads: int,
        selected: list[Any],
        storage: Any,
        batch_size: int,
        collect_atomic_charges: bool = False,
    ) -> tuple[Any, Callable[[], None]]:
        if engine == "xtbloom-sequential":
            if args.cpu_library is None:
                raise PerformanceError("CPU xTBloom library unavailable")
            adapter = SharedContextSequential(
                benchmark_run,
                dataset_runner,
                selected,
                args.cpu_library,
                args.property,
                threads,
                args.max_scc_iterations,
                electronic_temperature_hartree,
                collect_atomic_charges,
            )
            return adapter, adapter.invoke
        if engine == "xtbloom-ragged":
            native_library = args.cpu_library if backend == "cpu" else args.cuda_library
            if native_library is None:
                raise PerformanceError(f"{backend} xTBloom library unavailable")
            cell = benchmark_run.Cell(
                "xtbloom", backend, memory, "paper-real", args.property, batch_size
            )
            adapter = benchmark_run.XTBloomAdapter.from_storage(
                native_library,
                storage,
                cell,
                args.device_id,
                threads,
                collect_atomic_charges=collect_atomic_charges,
                max_scc_iterations=args.max_scc_iterations,
                electronic_temperature_hartree=electronic_temperature_hartree,
            )
            return adapter, lambda adapter=adapter: (
                adapter.invoke(),
                adapter.synchronize(),
            )
        if engine == "xtb":
            if args.xtb_library is None or not args.xtb_library.is_file():
                raise PerformanceError("xTB library unavailable")
            adapter = XtbAdapter(
                args.xtb_library,
                storage,
                args.property,
                None,
                accuracy=args.accuracy,
                max_iterations=args.max_scc_iterations,
                electronic_temperature_kelvin=args.electronic_temperature_kelvin,
                threads=threads,
            )
            return adapter, lambda adapter=adapter: (
                adapter.restart_scc(),
                adapter.invoke(),
            )
        if engine == "tblite":
            if args.tblite_library is None or not args.tblite_library.is_file():
                raise PerformanceError("tblite library unavailable")
            adapter = TbliteAdapter(
                args.tblite_library,
                storage,
                args.property,
                accuracy=args.accuracy,
                max_iterations=args.max_scc_iterations,
                electronic_temperature_hartree=electronic_temperature_hartree,
                collect_atomic_charges=False,
                threads=threads,
            )
            return adapter, lambda adapter=adapter: (
                adapter.restart_scc(),
                adapter.invoke(),
            )
        if args.dxtb_source is None or not args.dxtb_source.is_dir():
            raise PerformanceError("dxtb source/runtime unavailable")
        adapter = DxtbAdapter(
            storage,
            args.property,
            backend,
            args.device_id,
            1,
            args.dxtb_source,
            force_convergence=True,
            accuracy=args.accuracy,
            max_iterations=args.max_scc_iterations,
        )
        return adapter, lambda adapter=adapter: (
            adapter.invoke(),
            adapter.synchronize(),
        )

    iteration_bins = (
        args.ao_bins
        if args.selection_mode == "by-bin"
        else ((1, None, args.selection_mode),)
    )
    for ao_bin in iteration_bins:
        for batch_size in args.batch_sizes:
            try:
                if args.selection_mode == "controlled-wide":
                    selected = controlled_wide(
                        all_items, args.ao_bins, batch_size, args.seed
                    )
                elif args.selection_mode == "real-order":
                    if len(all_items) < batch_size:
                        raise PerformanceError(
                            f"real-order selection has {len(all_items)} systems, needs {batch_size}"
                        )
                    selected = all_items[:batch_size]
                else:
                    selected = select_systems(all_items, ao_bin, batch_size, args.seed)
            except PerformanceError as exc:
                reason = str(exc)
                not_applicable = "has 0 systems" in reason
                censored = (
                    "systems, needs" in reason
                    or "selection has only" in reason
                    or "real-order selection has" in reason
                )
                availability = (
                    "not-applicable"
                    if not_applicable
                    else "sample-censored"
                    if censored
                    else "unavailable"
                )
                rows.append(
                    {
                        "suite": args.suite,
                        "hardware_id": os.environ.get("PAPER_HARDWARE_ID"),
                        "dataset": args.dataset,
                        "ao_bin": ao_bin[2],
                        "batch_size": batch_size,
                        "max_scc_iterations": args.max_scc_iterations,
                        "accuracy": args.accuracy,
                        "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                        "availability": availability,
                        "claim_eligible": False,
                        "reason": reason,
                    }
                )
                continue
            storage = dataset_runner.storage_from_systems(
                [item.system for item in selected]
            )
            selected_pair_counts = geometric_pair_counts(selected)
            for threads in args.cpu_threads:
                for engine, backend, memory in variants(threads):
                    if backend == "cuda" and threads != args.cpu_threads[0]:
                        continue
                    if (
                        backend == "cpu"
                        and allowed_cpus
                        and threads > len(allowed_cpus)
                    ):
                        rows.append(
                            {
                                "suite": args.suite,
                                "hardware_id": os.environ.get("PAPER_HARDWARE_ID"),
                                "dataset": args.dataset,
                                "engine": engine,
                                "backend": backend,
                                "memory_mode": memory,
                                "ao_bin": ao_bin[2],
                                "batch_size": batch_size,
                                "cpu_threads": threads,
                                "max_scc_iterations": args.max_scc_iterations,
                                "availability": "unavailable",
                                "reason": f"requested {threads} CPU threads but Slurm affinity exposes {len(allowed_cpus)} CPUs",
                            }
                        )
                        continue
                    adapter: Any | None = None
                    setup_start = time.perf_counter_ns()
                    original_affinity = (
                        set(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity")
                        else set()
                    )
                    adapter_constructed = False
                    compute_attempted = False
                    try:
                        # CPU cells with budget N are pinned to the same first N
                        # Slurm-assigned logical CPUs.  This makes the budget an
                        # enforced affinity contract instead of a thread-count hint.
                        if backend == "cpu" and original_affinity:
                            os.sched_setaffinity(0, set(allowed_cpus[:threads]))
                        adapter, invoke = construct_adapter(
                            engine,
                            backend,
                            memory,
                            threads,
                            selected,
                            storage,
                            batch_size,
                        )
                        adapter_constructed = True
                        setup_ms = (time.perf_counter_ns() - setup_start) * 1e-6
                        compute_attempted = True
                        cold_ms, cold_cpu_ms = timed(invoke)
                        for _ in range(args.warmups):
                            invoke()
                        measured = [timed(invoke) for _ in range(args.repetitions)]
                        samples = [wall_ms for wall_ms, _ in measured]
                        cpu_samples = [cpu_ms for _, cpu_ms in measured]
                        outputs = (
                            adapter.results()
                            if engine in {"xtb", "tblite", "dxtb"}
                            else adapter.results()
                            if engine == "xtbloom-sequential"
                            else adapter.raw_results()
                        )
                        timed_correctness = qualify(
                            selected,
                            outputs,
                            references,
                            engine,
                            args.energy_atol,
                            args.force_atol,
                            args.charge_atol,
                            args.property,
                            args.require_correctness,
                            allow_system_failures=args.suite == "ragged",
                        )
                        engine_options = {
                            "accuracy": getattr(adapter, "accuracy", None),
                            "electronic_temperature_kelvin": getattr(
                                adapter, "electronic_temperature_kelvin", None
                            ),
                            "electronic_temperature_hartree": getattr(
                                getattr(adapter, "options", None),
                                "electronic_temperature",
                                getattr(
                                    adapter, "electronic_temperature_hartree", None
                                ),
                            ),
                            "xtbloom_charge_tolerance": getattr(
                                getattr(adapter, "options", None),
                                "charge_tolerance",
                                None,
                            ),
                            "xtbloom_energy_tolerance": getattr(
                                getattr(adapter, "options", None),
                                "energy_tolerance",
                                None,
                            ),
                        }
                        adapter_identity = {
                            "python_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                            "engine_version": getattr(
                                adapter,
                                "version",
                                getattr(adapter, "api_version", None),
                            ),
                            "torch_version": getattr(adapter, "torch_version", None),
                            "engine_module_path": getattr(adapter, "module_path", None),
                        }
                        adapter.close()
                        adapter = None
                        del invoke
                        correctness = timed_correctness
                        qualification_output_contract = (
                            "energy+analytic-forces+atomic-charges"
                            if engine.startswith("xtbloom")
                            and args.property == "force"
                            and args.suite != "capacity"
                            else "energy+analytic-forces"
                            if args.property == "force"
                            else "energy"
                        )
                        if (
                            engine.startswith("xtbloom")
                            and args.property == "force"
                            and args.suite != "capacity"
                        ):
                            # Charge publication is qualified in a separate,
                            # untimed invocation.  The timing and memory rows
                            # remain the common energy+force contract used by
                            # xTB, tblite and dxtb.
                            adapter, qualification_invoke = construct_adapter(
                                engine,
                                backend,
                                memory,
                                threads,
                                selected,
                                storage,
                                batch_size,
                                collect_atomic_charges=True,
                            )
                            qualification_invoke()
                            qualification_outputs = (
                                adapter.results()
                                if engine == "xtbloom-sequential"
                                else adapter.raw_results()
                            )
                            correctness = qualify(
                                selected,
                                qualification_outputs,
                                references,
                                engine,
                                args.energy_atol,
                                args.force_atol,
                                args.charge_atol,
                                args.property,
                                args.require_correctness,
                                allow_system_failures=args.suite == "ragged",
                                require_charges=True,
                            )
                            adapter.close()
                            adapter = None
                            del qualification_invoke
                        probe_invocations = 3
                        # Reconstruct the complete cell in a fresh spawned
                        # process. Process exit clears Python, BLAS, CUDA and
                        # PyTorch allocators, preventing cross-row memory carry.
                        memory_probe = isolated_memory_probe(
                            {
                                "repo": str(args.repo),
                                "manifest": str(args.manifest),
                                "dataset": args.dataset,
                                "subset": args.subset,
                                "system_ids": [item.system_id for item in selected],
                                "engine": engine,
                                "backend": backend,
                                "memory": memory,
                                "threads": threads,
                                "property": args.property,
                                "device_id": args.device_id,
                                "max_scc_iterations": args.max_scc_iterations,
                                "accuracy": args.accuracy,
                                "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                                "cpu_library": str(args.cpu_library)
                                if args.cpu_library is not None
                                else None,
                                "cuda_library": str(args.cuda_library)
                                if args.cuda_library is not None
                                else None,
                                "xtb_library": str(args.xtb_library)
                                if args.xtb_library is not None
                                else None,
                                "tblite_library": str(args.tblite_library)
                                if args.tblite_library is not None
                                else None,
                                "dxtb_source": str(args.dxtb_source)
                                if args.dxtb_source is not None
                                else None,
                                "cpu_affinity": allowed_cpus[:threads]
                                if backend == "cpu"
                                else allowed_cpus,
                                "probe_invocations": probe_invocations,
                            },
                            args.memory_probe_timeout_seconds,
                        )
                        if (
                            args.require_correctness
                            and memory_probe["status"] != "available"
                        ):
                            raise PerformanceError(
                                f"row-local peak-memory probe failed: {memory_probe['reason']}"
                            )
                        timing = summary(
                            samples,
                            batch_size,
                            args.bootstrap_samples,
                            f"{args.seed}|{args.suite}|{dataset_runner.__name__}|{engine}|{backend}|{memory}|{ao_bin[2]}|{batch_size}|{threads}",
                        )
                        median_seconds = timing["median_ms"] * 1e-3
                        timing["raw_process_cpu_ms"] = cpu_samples
                        timing["median_process_cpu_ms"] = statistics.median(cpu_samples)
                        timing["median_cpu_utilization_cores"] = (
                            timing["median_process_cpu_ms"] / timing["median_ms"]
                        )
                        timing["all_input_time_to_solution_ms"] = timing["median_ms"]
                        timing["successful_systems_per_second"] = (
                            correctness["successful_systems"] / median_seconds
                        )
                        timing["matched_success_systems_per_second"] = (
                            correctness["paired_systems"] / median_seconds
                        )
                        timing["success_rate"] = (
                            correctness["successful_systems"] / batch_size
                        )
                        iteration_values = [
                            int(value) for value in outputs.get("scc_iterations", [])
                        ]
                        iteration_summary = (
                            {
                                "availability": "available",
                                "values": iteration_values,
                                "median": statistics.median(iteration_values),
                                "p95": percentile(
                                    [float(value) for value in iteration_values], 0.95
                                ),
                            }
                            if iteration_values
                            else {
                                "availability": "unavailable",
                                "reason": "adapter does not publish SCC iterations",
                            }
                        )
                        rows.append(
                            {
                                "suite": args.suite,
                                "hardware_id": os.environ.get("PAPER_HARDWARE_ID"),
                                "dataset": args.dataset,
                                "subset": args.subset,
                                "engine": engine,
                                "backend": backend,
                                "memory_mode": memory,
                                "ao_bin": ao_bin[2],
                                "ao_counts": [ao_count(item) for item in selected],
                                "atom_counts": [
                                    len(item.system.atomic_numbers) for item in selected
                                ],
                                "geometric_pair_counts": selected_pair_counts,
                                "batch_size": batch_size,
                                "cpu_threads": threads,
                                "max_scc_iterations": args.max_scc_iterations,
                                "accuracy": args.accuracy,
                                "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                                "engine_options": engine_options,
                                "adapter_identity": adapter_identity,
                                "system_ids": [item.system_id for item in selected],
                                "availability": "available",
                                "adapter_constructed": adapter_constructed,
                                "compute_attempted": compute_attempted,
                                "setup_ms": setup_ms,
                                "cold_ms": cold_ms,
                                "cold_process_cpu_ms": cold_cpu_ms,
                                "cpu_affinity": sorted(os.sched_getaffinity(0))
                                if hasattr(os, "sched_getaffinity")
                                else [],
                                "steady_fresh": timing,
                                "scc_iterations": iteration_summary,
                                "correctness": correctness,
                                "timed_output_correctness": timed_correctness,
                                "timed_output_contract": (
                                    "energy+analytic-forces"
                                    if args.property == "force"
                                    else "energy"
                                ),
                                "qualification_output_contract": qualification_output_contract,
                                "memory": memory_probe,
                                "timing_contract": "public FRESH compute including required reset and explicit CUDA synchronization; setup and correctness publication excluded",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - retain every unavailable/error cell
                        category = exception_category(exc)
                        rows.append(
                            {
                                "suite": args.suite,
                                "hardware_id": os.environ.get("PAPER_HARDWARE_ID"),
                                "dataset": args.dataset,
                                "engine": engine,
                                "backend": backend,
                                "memory_mode": memory,
                                "ao_bin": ao_bin[2],
                                "batch_size": batch_size,
                                "cpu_threads": threads,
                                "max_scc_iterations": args.max_scc_iterations,
                                "accuracy": args.accuracy,
                                "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                                "availability": category,
                                "adapter_constructed": adapter_constructed,
                                "compute_attempted": compute_attempted,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    finally:
                        if adapter is not None:
                            try:
                                adapter.close()
                            except Exception:  # noqa: BLE001 - primary error is already retained
                                pass
                        if original_affinity:
                            os.sched_setaffinity(0, original_affinity)

    # Derive worker scaling only when the exact dataset/bin/batch/engine/memory
    # cell also has a one-thread observation.  Missing baselines remain null.
    one_thread = {
        (
            row.get("dataset"),
            row.get("engine"),
            row.get("backend"),
            row.get("memory_mode"),
            row.get("ao_bin"),
            row.get("batch_size"),
        ): row
        for row in rows
        if row.get("availability") == "available" and row.get("cpu_threads") == 1
    }
    for row in rows:
        if row.get("availability") != "available" or row.get("backend") != "cpu":
            continue
        baseline = one_thread.get(
            (
                row.get("dataset"),
                row.get("engine"),
                row.get("backend"),
                row.get("memory_mode"),
                row.get("ao_bin"),
                row.get("batch_size"),
            )
        )
        if baseline is None:
            row["worker_scaling"] = {
                "availability": "unavailable",
                "reason": "one-thread baseline missing",
            }
            continue
        speedup = (
            row["steady_fresh"]["median_systems_per_second"]
            / baseline["steady_fresh"]["median_systems_per_second"]
        )
        row["worker_scaling"] = {
            "availability": "available",
            "speedup_vs_one_thread": speedup,
            "parallel_efficiency": speedup / int(row["cpu_threads"]),
        }

    capacity_boundaries: dict[str, Any] = {}
    if args.suite == "capacity":
        for _, _, label in iteration_bins:
            cells = sorted(
                (
                    row
                    for row in rows
                    if row.get("ao_bin") == label
                    and row.get("engine") == "xtbloom-ragged"
                ),
                key=lambda row: row.get("batch_size", 0),
            )
            omitted_empty_bin = any(
                row.get("ao_bin") == label
                and row.get("availability") == "not-applicable"
                for row in rows
            )
            available = [row for row in cells if row.get("availability") == "available"]
            oom = [row for row in cells if row.get("availability") == "oom"]
            saturated_at = None
            if len(available) >= 3:
                rates = [
                    row["steady_fresh"]["median_systems_per_second"]
                    for row in available
                ]
                for index in range(2, len(rates)):
                    if (
                        rates[index] <= 1.05 * rates[index - 1]
                        and rates[index - 1] <= 1.05 * rates[index - 2]
                    ):
                        saturated_at = available[index]["batch_size"]
                        break
            status = (
                "not-applicable"
                if omitted_empty_bin and not cells
                else "oom"
                if oom
                else "saturated"
                if saturated_at is not None
                else "sample-censored"
            )
            capacity_boundaries[label] = {
                "status": status,
                "largest_available_batch": max(
                    (row["batch_size"] for row in available), default=None
                ),
                "first_oom_batch": min(
                    (row["batch_size"] for row in oom), default=None
                ),
                "saturated_at_batch": saturated_at,
                "requested_batches": list(args.batch_sizes),
                "note": "sample-censored means the frozen distinct-system set ended before measured saturation/OOM",
                "claim_eligible": status in {"oom", "saturated"},
            }
    fatal_rows: list[int] = []
    fatal_reasons: list[str] = []
    if args.require_correctness:
        for index, row in enumerate(rows):
            if row.get("availability") in {"not-applicable", "sample-censored"}:
                continue
            if row.get("engine") == "dxtb":
                if not args.require_dxtb:
                    continue
                # dxtb is a pinned secondary diagnostic.  Formal evidence
                # requires the runtime to construct and every coordinate to
                # be attempted, while scientific failures stay explicit and
                # simply remain ineligible for performance figures.
                if row.get("adapter_constructed") and row.get("compute_attempted"):
                    continue
                fatal_rows.append(index)
                continue
            availability = row.get("availability")
            correctness = row.get("correctness", {}).get("status")
            if args.suite == "capacity" and availability == "oom":
                continue
            if availability != "available" or correctness != "pass":
                fatal_rows.append(index)
                continue
            if args.suite == "ragged" and not row.get("correctness", {}).get(
                "performance_claim_eligible"
            ):
                fatal_rows.append(index)
        if args.suite == "capacity":
            for label, boundary in capacity_boundaries.items():
                if (
                    boundary["largest_available_batch"] is None
                    and boundary["status"] != "not-applicable"
                ):
                    fatal_reasons.append(
                        f"{label}: no correctness-qualified capacity cell"
                    )
        if args.suite != "pool-baseline" and not any(
            row.get("availability") == "available" and row.get("engine") != "dxtb"
            for row in rows
        ):
            fatal_reasons.append("no correctness-qualified non-dxtb performance cell")
    document = {
        "schema_version": 1,
        "command": sys.argv,
        "suite": args.suite,
        "hardware_id": os.environ.get("PAPER_HARDWARE_ID"),
        "max_scc_iterations": args.max_scc_iterations,
        "accuracy": args.accuracy,
        "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
        "formal_gate": {
            "required": args.require_correctness,
            "status": "fail" if fatal_rows or fatal_reasons else "pass",
            "fatal_row_indices": fatal_rows,
            "fatal_reasons": fatal_reasons,
        },
        "capacity_boundaries": capacity_boundaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 1 if fatal_rows or fatal_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
