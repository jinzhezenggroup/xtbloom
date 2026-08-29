#!/usr/bin/env python3
"""Reproducible end-to-end GFN2-xTB benchmark matrix.

The xtbloom adapter calls only the public C ABI through ``ctypes``.  Contexts,
ragged descriptors, and caller-owned output buffers persist across measured
calls.  CUDA timings end with an explicit ``cudaDeviceSynchronize``; device
outputs are copied back only after timing for correctness validation.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Paper and ordinary benchmark callers share this adapter surface. Physics and
# canonical conformance helpers come from the same exact clean xTBloom source
# selected by PAPER_REPO_ROOT.
REPOSITORY_ROOT = Path(
    os.environ.get("PAPER_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
sys.path.insert(0, str(CONFORMANCE_TOOLS))

import xtbloom_conformance as conformance
import xtbloom_public_api as public_api
from xtbloom_public_api import PublicBatchStorage

try:
    from .xtb_adapter import XtbAdapter, XtbError, XtbState
except ImportError:  # Direct ``python benchmarks/run.py`` execution.
    from xtb_adapter import XtbAdapter, XtbError, XtbState

try:
    from .tblite_adapter import TbliteAdapter, TbliteError
except ImportError:  # Direct ``python benchmarks/run.py`` execution.
    from tblite_adapter import TbliteAdapter, TbliteError

try:
    from .dxtb_adapter import DxtbAdapter, DxtbError
    from .dxtb_adapter import timed_invoke as timed_dxtb_invoke
except ImportError:  # Direct ``python benchmarks/run.py`` execution.
    from dxtb_adapter import DxtbAdapter, DxtbError
    from dxtb_adapter import timed_invoke as timed_dxtb_invoke

SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES = (1, 8, 32, 128)
DEFAULT_PROPERTIES = ("energy", "force")
DEFAULT_WORKLOADS = ("gas", "qmmm")
DEFAULT_REFERENCE_ENV = Path("/tmp/xtbloom-reference-env.E0KcEA")
REPEATED_CALL_SEMANTICS = "same_geometry_repeated_compute"
WORKLOAD_CASES = {
    "gas": "ketene",
    "qmmm": "water_dimer_6pc_hardness",
}
HETEROGENEOUS_WORKLOAD_CASES = {
    "heterogeneous-gas": (
        "h3_plus",
        "ketene",
        "nenacl",
        "sif5_minus",
    ),
    "heterogeneous-qmmm": (
        "water_one_pc_gamma999",
        "water_dimer_6pc_hardness",
        "water_dimer_6pc_gamma999",
    ),
}
REFERENCE_REPOSITORIES = {
    "tblite": Path.home() / "codes" / "tblite",
    "xtb": Path.home() / "codes" / "xtb",
    "dxtb": Path.home() / "codes" / "dxtb",
}
REFERENCE_COMMANDS = {
    "tblite": [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "{tblite}",
        "{input}",
        "--no-restart",
        "--method",
        "gfn2",
        "--acc",
        "0.0001",
        "[--grad gradient]",
        "--json",
        "result.json",
    ],
    "xtb": [
        "env",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "{xtb}",
        "{input}",
        "--gfn",
        "2",
        "--acc",
        "0.0001",
        "[--grad]",
        "--json",
        "--norestart",
        "--chrg",
        "{charge}",
        "--uhf",
        "{uhf}",
        "-P",
        "1",
    ],
    "dxtb": [
        "{python}",
        "-m",
        "dxtb",
        "{input}",
        "--method",
        "gfn2",
        "[--forces]",
        "--dtype",
        "float64",
        "--device",
        "{device}",
    ],
}


class BenchmarkError(RuntimeError):
    """An actionable adapter, timing, or result-publication failure."""


class ReferenceUnavailable(BenchmarkError):
    """A requested reference coordinate the selected public API cannot express."""


def parse_csv_values(value: str, converter: type = str) -> tuple[Any, ...]:
    """Parse one nonempty comma-separated CLI selection."""
    try:
        values = tuple(
            converter(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("selection must not be empty")
    return values


def sha256_file(path: Path) -> str | None:
    """Hash a regular file without loading it into memory."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: Sequence[str]) -> str | None:
    """Return stripped stdout from a diagnostic command, or ``None``."""
    try:
        completed = subprocess.run(
            list(command), check=False, text=True, capture_output=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state(path: Path) -> dict[str, Any]:
    """Capture an exact revision and dirty bit for a source checkout."""
    revision = run_text(("git", "-C", str(path), "rev-parse", "HEAD"))
    status = run_text(("git", "-C", str(path), "status", "--porcelain"))
    return {
        "path": str(path),
        "revision": revision,
        "dirty": None if revision is None else bool(status),
    }


def current_rss_bytes() -> int | None:
    """Read Linux resident memory without changing the process high-water mark."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_hwm_bytes() -> int:
    """Return the process-wide maximum RSS reported by getrusage."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Compute a deterministic nearest-rank percentile for small sample sets."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def timing_summary(samples_ms: Sequence[float], batch_size: int) -> dict[str, Any]:
    """Summarize raw samples without discarding the evidence used by the CSV."""
    median = statistics.median(samples_ms)
    return {
        "samples_ms": list(samples_ms),
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": median,
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "systems_per_second_at_median": 1000.0 * batch_size / median,
    }


def configure_cuda_runtime(runtime: public_api.CudaRuntime) -> None:
    """Declare CUDA calls needed for explicit timing and memory boundaries."""
    runtime.runtime.cudaDeviceSynchronize.argtypes = []
    runtime.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    runtime.runtime.cudaMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    runtime.runtime.cudaMemGetInfo.restype = ctypes.c_int


def cuda_synchronize(runtime: public_api.CudaRuntime) -> None:
    """Make every CUDA timing encompass completion, not just submission."""
    status = runtime.runtime.cudaDeviceSynchronize()
    runtime._check(status, "cudaDeviceSynchronize")


def cuda_memory(runtime: public_api.CudaRuntime) -> dict[str, int]:
    """Sample device-global free/total memory at a documented sync point."""
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    runtime._check(
        runtime.runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
        "cudaMemGetInfo",
    )
    return {
        "device_global_free_bytes": int(free.value),
        "device_global_total_bytes": int(total.value),
        "device_global_used_bytes": int(total.value - free.value),
    }


@dataclass(frozen=True)
class Cell:
    """One independently constructed matrix cell."""

    engine: str
    backend: str
    memory_mode: str
    workload: str
    property: str
    batch_size: int


def workload_case_ids(workload: str, batch_size: int) -> tuple[str, ...]:
    """Return the exact deterministic corpus sequence for one matrix row."""
    if batch_size <= 0:
        raise BenchmarkError("batch size must be positive")
    if workload in WORKLOAD_CASES:
        return (WORKLOAD_CASES[workload],) * batch_size
    try:
        candidates = HETEROGENEOUS_WORKLOAD_CASES[workload]
    except KeyError as exc:
        raise BenchmarkError(f"unknown workload: {workload}") from exc
    return tuple(candidates[index % len(candidates)] for index in range(batch_size))


def workload_case_sequence(
    workload: str,
    batch_size: int,
    cases: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Resolve one row's exact case IDs without silently dropping a coordinate."""
    identifiers = workload_case_ids(workload, batch_size)
    missing = sorted(set(identifiers) - cases.keys())
    if missing:
        raise BenchmarkError(
            "workload references missing conformance cases: " + ", ".join(missing)
        )
    return tuple(cases[identifier] for identifier in identifiers)


class XTBloomAdapter:
    """Persistent public-C-API adapter for one matrix cell."""

    def __init__(
        self,
        library_path: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        case_sequence: Sequence[dict[str, Any]],
        cell: Cell,
        device_id: int,
        cpu_threads: int,
    ) -> None:
        storage = public_api.assemble_batch(manifest_path, manifest, case_sequence)
        self._initialize(
            library_path,
            storage,
            cell,
            device_id,
            cpu_threads,
            collect_atomic_charges=False,
            max_scc_iterations=None,
            electronic_temperature_hartree=None,
        )

    @classmethod
    def from_storage(
        cls,
        library_path: Path,
        storage: PublicBatchStorage,
        cell: Cell,
        device_id: int,
        cpu_threads: int,
        *,
        collect_atomic_charges: bool = False,
        max_scc_iterations: int | None = None,
        electronic_temperature_hartree: float | None = None,
    ) -> XTBloomAdapter:
        """Construct the shared adapter from already assembled ragged storage.

        The publication benchmark keeps using the manifest constructor above.
        Dataset consumers can use this entry point without reproducing the
        public-C-ABI setup, output ownership, or synchronization logic.
        """
        adapter = cls.__new__(cls)
        adapter._initialize(
            library_path,
            storage,
            cell,
            device_id,
            cpu_threads,
            collect_atomic_charges=collect_atomic_charges,
            max_scc_iterations=max_scc_iterations,
            electronic_temperature_hartree=electronic_temperature_hartree,
        )
        return adapter

    def _initialize(
        self,
        library_path: Path,
        storage: PublicBatchStorage,
        cell: Cell,
        device_id: int,
        cpu_threads: int,
        *,
        collect_atomic_charges: bool,
        max_scc_iterations: int | None,
        electronic_temperature_hartree: float | None,
    ) -> None:
        """Bind one preassembled batch while preserving the legacy defaults."""
        if max_scc_iterations is not None and max_scc_iterations <= 0:
            raise BenchmarkError("max SCC iterations must be positive")
        if electronic_temperature_hartree is not None and (
            not math.isfinite(electronic_temperature_hartree)
            or electronic_temperature_hartree < 0.0
        ):
            raise BenchmarkError(
                "electronic temperature must be finite and nonnegative"
            )
        if len(storage.slices) != cell.batch_size:
            raise BenchmarkError("storage size must equal the requested batch")
        self.cell = cell
        self.library_path = library_path
        self.library = public_api._configure_library(library_path)
        self.storage = storage
        self.context = public_api._make_context(
            self.library, cell.backend, device_id, cpu_threads
        )
        self.memory = public_api.DescriptorMemory(cell.memory_mode, device_id)
        # Host descriptors still execute on CUDA.  Keep a control runtime even
        # when DescriptorMemory did not need one so every CUDA timing has an
        # explicit completion boundary and device-memory sample.
        self.cuda_control = self.memory.cuda
        self.owns_cuda_control = False
        if cell.backend == "cuda" and self.cuda_control is None:
            self.cuda_control = public_api.CudaRuntime(device_id)
            self.owns_cuda_control = True
        if self.cuda_control is not None:
            configure_cuda_runtime(self.cuda_control)
        # Both backends consume the same explicit ABI-v2 one/two-channel
        # selection; benchmark cells therefore exercise the production suffix.
        self.batch = public_api._make_batch(
            self.library,
            self.storage,
            self.memory,
            include_spin_channels=True,
        )
        self.options = public_api.ComputeOptions()
        public_api._call_ok(
            self.library,
            self.library.xtbloom_compute_options_init(
                ctypes.byref(self.options), ctypes.sizeof(self.options)
            ),
            "xtbloom_compute_options_init",
        )
        self.options.model = public_api.XTBLOOM_MODEL_GFN2_XTB
        self.options.flags = public_api.XTBLOOM_COMPUTE_ENERGY
        if cell.property == "force":
            self.options.flags |= public_api.XTBLOOM_COMPUTE_FORCES
            if self.storage.point_charge_values:
                # A QM/MM force workload covers the complete public force
                # contract: both QM atoms and caller-owned external sites.
                self.options.flags |= public_api.XTBLOOM_COMPUTE_POINT_CHARGE_FORCES
        if collect_atomic_charges:
            self.options.flags |= public_api.XTBLOOM_COMPUTE_ATOMIC_CHARGES
        if max_scc_iterations is not None:
            self.options.max_scc_iterations = max_scc_iterations
        if electronic_temperature_hartree is not None:
            self.options.electronic_temperature = electronic_temperature_hartree
        # These public defaults are recorded in every row and match normal API use.
        self.systems = cell.batch_size
        self.atoms = len(self.storage.atomic_numbers)
        self.energies = (ctypes.c_double * self.systems)()
        self.forces = (
            (ctypes.c_double * (3 * self.atoms))() if cell.property == "force" else None
        )
        self.charges = (
            (ctypes.c_double * self.atoms)() if collect_atomic_charges else None
        )
        self.point_forces = (
            (ctypes.c_double * (3 * len(self.storage.point_charge_values)))()
            if cell.property == "force" and self.storage.point_charge_values
            else None
        )
        self.iterations = (ctypes.c_int32 * self.systems)()
        self.converged = (ctypes.c_uint8 * self.systems)()
        self.statuses = (ctypes.c_int32 * self.systems)()
        self.result = public_api.BatchResult()
        public_api._call_ok(
            self.library,
            self.library.xtbloom_batch_result_init(
                ctypes.byref(self.result), ctypes.sizeof(self.result)
            ),
            "xtbloom_batch_result_init",
        )
        self.result.energies = self.memory.output(self.energies, "energies")
        if self.forces is not None:
            self.result.forces = self.memory.output(self.forces, "forces")
        if self.charges is not None:
            self.result.atomic_charges = self.memory.output(
                self.charges, "atomic_charges"
            )
        if self.point_forces is not None:
            self.result.point_charge_forces = self.memory.output(
                self.point_forces, "point_charge_forces"
            )
        self.result.scc_iterations = self.memory.output(
            self.iterations, "scc_iterations"
        )
        self.result.scc_converged = self.memory.output(self.converged, "scc_converged")
        self.result.per_system_status = self.memory.output(
            self.statuses, "per_system_status"
        )

    def invoke(self) -> None:
        """Submit one inference without allocating or publishing to Python."""
        public_api._call_ok(
            self.library,
            self.library.xtbloom_compute(
                self.context,
                ctypes.byref(self.batch),
                ctypes.byref(self.options),
                ctypes.byref(self.result),
            ),
            f"xtbloom {self.cell.backend}/{self.cell.memory_mode} inference",
        )

    def synchronize(self) -> None:
        """Explicitly complete CUDA work; CPU execution is synchronous."""
        if self.cuda_control is not None:
            cuda_synchronize(self.cuda_control)

    def memory_snapshot(self) -> dict[str, Any]:
        """Return process and CUDA memory sampled at a synchronized boundary."""
        snapshot: dict[str, Any] = {
            "host_rss_bytes": current_rss_bytes(),
            "host_process_hwm_bytes": process_hwm_bytes(),
            "host_hwm_scope": "entire benchmark runner process",
        }
        if self.cuda_control is not None:
            snapshot.update(cuda_memory(self.cuda_control))
            snapshot["device_memory_scope"] = "cudaMemGetInfo device-global sample"
        return snapshot

    def raw_results(self) -> dict[str, Any]:
        """Download every per-system output without aggregating peer failures."""
        self.synchronize()
        self.memory.download_outputs()
        output: dict[str, Any] = {
            "energies_hartree": [float(value) for value in self.energies],
            "scc_iterations": [int(value) for value in self.iterations],
            "scc_converged": [int(value) for value in self.converged],
            "per_system_status": [int(value) for value in self.statuses],
        }
        if self.forces is not None:
            output["forces_hartree_per_bohr"] = [float(value) for value in self.forces]
        if self.charges is not None:
            output["atomic_charges_e"] = [float(value) for value in self.charges]
        if self.point_forces is not None:
            output["point_charge_forces_hartree_per_bohr"] = [
                float(value) for value in self.point_forces
            ]
        return output

    def results(self) -> dict[str, Any]:
        """Preserve the benchmark's strict all-systems-success publication."""
        raw = self.raw_results()
        failures = [
            "system "
            f"{index}: status={raw['per_system_status'][index]}, "
            f"converged={raw['scc_converged'][index]}, "
            f"iterations={raw['scc_iterations'][index]}"
            for index in range(self.systems)
            if raw["per_system_status"][index] != public_api.XTBLOOM_STATUS_SUCCESS
            or raw["scc_converged"][index] != 1
        ]
        if failures:
            raise BenchmarkError("; ".join(failures))
        output: dict[str, Any] = {
            "energies_hartree": raw["energies_hartree"],
            "scc_iterations": raw["scc_iterations"],
        }
        if self.forces is not None:
            output["forces_hartree_per_bohr"] = raw["forces_hartree_per_bohr"]
        if self.charges is not None:
            output["atomic_charges_e"] = raw["atomic_charges_e"]
        if self.point_forces is not None:
            output["point_charge_forces_hartree_per_bohr"] = raw[
                "point_charge_forces_hartree_per_bohr"
            ]
        return output

    def close(self) -> None:
        """Release descriptors before destroying their backend context."""
        try:
            self.memory.close()
        finally:
            try:
                if self.owns_cuda_control and self.cuda_control is not None:
                    self.cuda_control.close()
            finally:
                self.library.xtbloom_context_destroy(self.context)


def timed_invoke(adapter: XTBloomAdapter) -> float:
    """Measure one public inference through an explicit completion boundary."""
    start = time.perf_counter_ns()
    adapter.invoke()
    adapter.synchronize()
    return (time.perf_counter_ns() - start) * 1.0e-6


def correctness(
    cell: Cell,
    storage: public_api.PublicBatchStorage,
    output: dict[str, Any],
    manifest: dict[str, Any],
    tolerance_profile: str = "tolerances",
) -> dict[str, Any]:
    """Compare each repeated system with the committed independent golden."""
    energy_errors: list[float] = []
    force_errors: list[float] = []
    point_force_errors: list[float] = []
    energies = output["energies_hartree"]
    forces = output.get("forces_hartree_per_bohr")
    point_forces = output.get("point_charge_forces_hartree_per_bohr")
    for index, item in enumerate(storage.slices):
        energy_errors.append(
            abs(energies[index] - float(item.expected["energy_hartree"]))
        )
        if forces is not None:
            actual = forces[3 * item.atom_begin : 3 * item.atom_end]
            expected = [
                float(value) for value in item.expected["forces_hartree_per_bohr"]
            ]
            force_errors.append(
                max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            )
        if point_forces is not None:
            actual_points = point_forces[3 * item.point_begin : 3 * item.point_end]
            expected_points = [
                float(value)
                for value in item.expected["point_charge_forces_hartree_per_bohr"]
            ]
            point_force_errors.append(
                max(
                    abs(a - b)
                    for a, b in zip(actual_points, expected_points, strict=True)
                )
            )
    tolerances = manifest[tolerance_profile]
    energy_limit = float(tolerances["energy"]["atol"])
    force_limit = float(tolerances["forces"]["atol"])
    passed = max(energy_errors) <= energy_limit
    if force_errors:
        passed = passed and max(force_errors) <= force_limit
    point_force_limit = float(manifest["tolerances"]["point_charge_forces"]["atol"])
    if point_force_errors:
        passed = passed and max(point_force_errors) <= point_force_limit
    return {
        "status": "pass" if passed else "fail",
        "reference": "committed independent conformance golden",
        "tolerance_profile": tolerance_profile,
        "max_abs_energy_error_hartree": max(energy_errors),
        "energy_atol_hartree": energy_limit,
        "max_abs_force_error_hartree_per_bohr": max(force_errors)
        if force_errors
        else None,
        "force_atol_hartree_per_bohr": force_limit if force_errors else None,
        "max_abs_point_charge_force_error_hartree_per_bohr": (
            max(point_force_errors) if point_force_errors else None
        ),
        "point_charge_force_atol_hartree_per_bohr": (
            point_force_limit if point_force_errors else None
        ),
    }


def benchmark_xtbloom_cell(
    cell: Cell,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case_sequence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Construct, cold-run, warm up, sample, validate, and destroy one cell."""
    setup_start = time.perf_counter_ns()
    adapter: XTBloomAdapter | None = None
    rss_before = current_rss_bytes()
    try:
        adapter = XTBloomAdapter(
            args.library,
            args.manifest,
            manifest,
            case_sequence,
            cell,
            args.device_id,
            args.cpu_threads,
        )
        setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        memory_after_setup = adapter.memory_snapshot()
        cold_ms = timed_invoke(adapter)
        for _ in range(args.warmups):
            adapter.invoke()
            adapter.synchronize()
        samples = [timed_invoke(adapter) for _ in range(args.repetitions)]
        output = adapter.results()
        memory_after_measurement = adapter.memory_snapshot()
        row = base_row(cell)
        row.update(
            {
                "availability": "available",
                "setup_ms": setup_ms,
                "cold_latency_ms": cold_ms,
                "warm": timing_summary(samples, cell.batch_size),
                "correctness": correctness(cell, adapter.storage, output, manifest),
                "diagnostics": {
                    "scc_iterations_min": min(output["scc_iterations"]),
                    "scc_iterations_max": max(output["scc_iterations"]),
                },
                "memory": {
                    "host_rss_before_setup_bytes": rss_before,
                    "after_setup": memory_after_setup,
                    "after_measurement": memory_after_measurement,
                },
                "timing_scope": {
                    "setup": (
                        "shared-library load, context create, descriptor "
                        "allocation/upload"
                    ),
                    "cold": "first xtbloom_compute plus explicit CUDA synchronize",
                    "warm": "xtbloom_compute plus explicit CUDA synchronize",
                    "repeated_call_semantics": REPEATED_CALL_SEMANTICS,
                    "repeated_call_note": (
                        "unchanged coordinates still execute the full public compute "
                        "path; this is not proof of pair-list no-refresh reuse"
                    ),
                    "excluded": (
                        "post-timing device-to-host download and correctness comparison"
                    ),
                },
                "engine_options": {
                    "model": "GFN2-xTB",
                    "max_scc_iterations": int(adapter.options.max_scc_iterations),
                    "charge_tolerance": float(adapter.options.charge_tolerance),
                    "energy_tolerance": float(adapter.options.energy_tolerance),
                    "electronic_temperature_hartree": float(
                        adapter.options.electronic_temperature
                    ),
                    "cpu_threads": args.cpu_threads,
                    "device_id": args.device_id,
                },
            }
        )
        return row
    except public_api.BackendUnavailable as exc:
        return unavailable_row(cell, str(exc))
    except (BenchmarkError, conformance.ConformanceError, OSError) as exc:
        row = base_row(cell)
        row.update({"availability": "error", "error": str(exc)})
        return row
    finally:
        if adapter is not None:
            adapter.close()


def point_source_atomic_numbers(
    manifest_path: Path,
    manifest: dict[str, Any],
    case_sequence: Sequence[dict[str, Any]],
) -> list[list[int] | None] | None:
    """Read per-system xTB element-hardness identifiers for a ragged batch."""
    per_system: list[list[int] | None] = []
    for case in case_sequence:
        if case.get("input_schema") != "qmmm-v1":
            per_system.append(None)
            continue
        input_path = conformance.resolve_manifest_path(manifest_path, case["input"])
        hardness = manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"]
        document = conformance.load_qmmm_input(input_path, case, hardness)
        points = document["external_point_charges"]
        if points["gamma_mode"] != "element_hardness":
            raise ReferenceUnavailable(
                "xTB C API baseline supports the selected QM/MM workload only when "
                "gammas are represented by source atomic numbers"
            )
        per_system.append([int(value) for value in points["source_atomic_numbers"]])
    return per_system if any(numbers is not None for numbers in per_system) else None


class RaggedXtbAdapter(XtbAdapter):
    """Supply xTB's per-calculator point-source numbers for ragged QM/MM rows.

    The upstream adapter builds all calculator states through ``_create_state``
    and historically accepted one homogeneous source-number vector. This
    benchmark-only specialization selects the vector belonging to each storage
    slice while retaining the same persistent library and serial execution path.
    """

    def __init__(
        self,
        library_path: Path,
        storage: PublicBatchStorage,
        property_name: str,
        per_system_source_numbers: list[list[int] | None] | None,
        *,
        accuracy: float,
        max_iterations: int,
        electronic_temperature_kelvin: float,
    ) -> None:
        self._per_system_source_numbers = per_system_source_numbers or [None] * len(
            storage.slices
        )
        super().__init__(
            library_path,
            storage,
            property_name,
            None,
            accuracy=accuracy,
            max_iterations=max_iterations,
            electronic_temperature_kelvin=electronic_temperature_kelvin,
        )

    def _create_state(self, index: int) -> XtbState:
        self.point_source_atomic_numbers = self._per_system_source_numbers[index]
        return super()._create_state(index)


def timed_xtb_invoke(adapter: XtbAdapter) -> float:
    """Measure one in-process serial xTB logical batch."""
    start = time.perf_counter_ns()
    adapter.invoke()
    return (time.perf_counter_ns() - start) * 1.0e-6


def benchmark_xtb_cell(
    cell: Cell,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case_sequence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Measure a persistent xTB 6.7.1 C API baseline without process startup."""
    if args.xtb_library is None or not args.xtb_library.is_file():
        return unavailable_row(
            cell, f"xTB shared library unavailable: {args.xtb_library}"
        )
    setup_start = time.perf_counter_ns()
    rss_before = current_rss_bytes()
    adapter: XtbAdapter | None = None
    try:
        storage = public_api.assemble_batch(args.manifest, manifest, case_sequence)
        source_numbers = point_source_atomic_numbers(
            args.manifest, manifest, case_sequence
        )
        adapter = RaggedXtbAdapter(
            args.xtb_library,
            storage,
            cell.property,
            source_numbers,
            accuracy=1.0e-4,
            max_iterations=500,
            electronic_temperature_kelvin=300.0,
        )
        setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        memory_after_setup = {
            "host_rss_bytes": current_rss_bytes(),
            "host_process_hwm_bytes": process_hwm_bytes(),
            "host_hwm_scope": "entire benchmark runner process",
        }
        cold_ms = timed_xtb_invoke(adapter)
        for _ in range(args.warmups):
            adapter.invoke()
        samples = [timed_xtb_invoke(adapter) for _ in range(args.repetitions)]
        output = adapter.results()
        row = base_row(cell)
        row.update(
            {
                "availability": "available",
                "setup_ms": setup_ms,
                "cold_latency_ms": cold_ms,
                "warm": timing_summary(samples, cell.batch_size),
                "correctness": correctness(
                    cell,
                    storage,
                    output,
                    manifest,
                    tolerance_profile="tolerances",
                ),
                "memory": {
                    "host_rss_before_setup_bytes": rss_before,
                    "after_setup": memory_after_setup,
                    "after_measurement": {
                        "host_rss_bytes": current_rss_bytes(),
                        "host_process_hwm_bytes": process_hwm_bytes(),
                        "host_hwm_scope": "entire benchmark runner process",
                    },
                },
                "timing_scope": {
                    "setup": (
                        "libxtb load plus one persistent environment/molecule/"
                        "calculator/result per logical batch system"
                    ),
                    "cold": (
                        "serial loop of xtb_updateMolecule, xtb_singlepoint, and "
                        "requested result getters"
                    ),
                    "warm": (
                        "same in-process serial C API loop; no process startup or "
                        "handle allocation"
                    ),
                    "repeated_call_semantics": REPEATED_CALL_SEMANTICS,
                    "excluded": "setup, cleanup, and correctness comparison",
                },
                "engine_options": {
                    "model": "GFN2-xTB",
                    "api_version": adapter.api_version,
                    "accuracy": adapter.accuracy,
                    "max_scc_iterations": adapter.max_iterations,
                    "electronic_temperature_kelvin": (
                        adapter.electronic_temperature_kelvin
                    ),
                    "logical_batch_execution": "serial C API loop",
                    "persistent_states": len(adapter.states),
                    "thread_control": adapter.thread_control,
                    "energy_property_note": (
                        "xtb_singlepoint computes its native full single-point result; "
                        "the C API exposes no energy-only execution flag"
                    ),
                },
            }
        )
        return row
    except ReferenceUnavailable as exc:
        return unavailable_row(cell, str(exc))
    except (XtbError, BenchmarkError, conformance.ConformanceError, OSError) as exc:
        row = base_row(cell)
        row.update({"availability": "error", "error": str(exc)})
        return row
    finally:
        if adapter is not None:
            adapter.close()


def xtb_cells(args: argparse.Namespace) -> Iterable[Cell]:
    """Yield the persistent CPU xTB baseline coordinates."""
    for workload in args.workloads:
        for property_name in args.properties:
            for batch_size in args.batch_sizes:
                yield Cell("xtb", "cpu", "host", workload, property_name, batch_size)


def timed_tblite_invoke(adapter: TbliteAdapter) -> float:
    """Measure one in-process serial tblite logical batch."""
    start = time.perf_counter_ns()
    adapter.invoke()
    return (time.perf_counter_ns() - start) * 1.0e-6


def benchmark_tblite_cell(
    cell: Cell,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case_sequence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Measure a persistent tblite public-C-API baseline without startup cost."""
    if args.tblite_library is None or not args.tblite_library.is_file():
        return unavailable_row(
            cell, f"tblite shared library unavailable: {args.tblite_library}"
        )
    setup_start = time.perf_counter_ns()
    rss_before = current_rss_bytes()
    adapter: TbliteAdapter | None = None
    try:
        storage = public_api.assemble_batch(args.manifest, manifest, case_sequence)
        if storage.point_charge_values:
            return unavailable_row(cell, TbliteAdapter.external_point_charge_reason)
        adapter = TbliteAdapter(
            args.tblite_library,
            storage,
            cell.property,
            accuracy=1.0e-4,
            max_iterations=500,
        )
        setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        memory_after_setup = {
            "host_rss_bytes": current_rss_bytes(),
            "host_process_hwm_bytes": process_hwm_bytes(),
            "host_hwm_scope": "entire benchmark runner process",
        }
        cold_ms = timed_tblite_invoke(adapter)
        for _ in range(args.warmups):
            adapter.invoke()
        samples = [timed_tblite_invoke(adapter) for _ in range(args.repetitions)]
        output = adapter.results()
        row = base_row(cell)
        row.update(
            {
                "availability": "available",
                "setup_ms": setup_ms,
                "cold_latency_ms": cold_ms,
                "warm": timing_summary(samples, cell.batch_size),
                "correctness": correctness(cell, storage, output, manifest),
                "memory": {
                    "host_rss_before_setup_bytes": rss_before,
                    "after_setup": memory_after_setup,
                    "after_measurement": {
                        "host_rss_bytes": current_rss_bytes(),
                        "host_process_hwm_bytes": process_hwm_bytes(),
                        "host_hwm_scope": "entire benchmark runner process",
                    },
                },
                "timing_scope": {
                    "setup": (
                        "libtblite load plus one persistent context/structure/"
                        "calculator/result per logical batch system"
                    ),
                    "cold": (
                        "serial loop of tblite_update_structure_geometry, "
                        "tblite_get_singlepoint, and requested result getters"
                    ),
                    "warm": (
                        "same in-process serial public C API loop; no process startup "
                        "or handle allocation"
                    ),
                    "repeated_call_semantics": REPEATED_CALL_SEMANTICS,
                    "excluded": "setup, cleanup, and correctness comparison",
                },
                "engine_options": {
                    "model": "GFN2-xTB",
                    "api_version": adapter.version,
                    "accuracy": adapter.accuracy,
                    "max_scc_iterations": adapter.max_iterations,
                    "electronic_temperature_hartree": (
                        adapter.electronic_temperature_hartree
                    ),
                    "logical_batch_execution": "serial C API loop",
                    "persistent_states": len(adapter.states),
                    "thread_control": adapter.thread_control,
                    "energy_property_note": (
                        "tblite_get_singlepoint computes its native full single-point "
                        "result; the C API exposes no energy-only execution flag"
                    ),
                },
            }
        )
        return row
    except (TbliteError, BenchmarkError, conformance.ConformanceError, OSError) as exc:
        row = base_row(cell)
        row.update({"availability": "error", "error": str(exc)})
        return row
    finally:
        if adapter is not None:
            adapter.close()


def tblite_cells(args: argparse.Namespace) -> Iterable[Cell]:
    """Yield the persistent CPU tblite baseline coordinates."""
    for workload in args.workloads:
        for property_name in args.properties:
            for batch_size in args.batch_sizes:
                yield Cell("tblite", "cpu", "host", workload, property_name, batch_size)


def benchmark_dxtb_cell(
    cell: Cell,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case_sequence: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Measure persistent in-process dxtb without calculator reconstruction."""
    setup_start = time.perf_counter_ns()
    rss_before = current_rss_bytes()
    adapter: DxtbAdapter | None = None
    try:
        storage = public_api.assemble_batch(args.manifest, manifest, case_sequence)
        if storage.point_charge_values:
            return unavailable_row(cell, DxtbAdapter.external_point_charge_reason)
        adapter = DxtbAdapter(
            storage,
            cell.property,
            cell.backend,
            device_id=args.device_id,
            cpu_threads=args.dxtb_cpu_threads,
            source_root=args.dxtb_source,
            accuracy=1.0e-4,
            max_iterations=500,
        )
        setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        memory_after_setup = {
            "host_rss_bytes": current_rss_bytes(),
            "host_process_hwm_bytes": process_hwm_bytes(),
            "host_hwm_scope": "entire benchmark runner process",
        }
        cold_ms = timed_dxtb_invoke(adapter)
        for _ in range(args.warmups):
            adapter.invoke()
            adapter.synchronize()
        samples = [timed_dxtb_invoke(adapter) for _ in range(args.repetitions)]
        output = adapter.results()
        row = base_row(cell)
        row.update(
            {
                "availability": "available",
                "setup_ms": setup_ms,
                "cold_latency_ms": cold_ms,
                "warm": timing_summary(samples, cell.batch_size),
                "correctness": correctness(cell, storage, output, manifest),
                "memory": {
                    "host_rss_before_setup_bytes": rss_before,
                    "after_setup": memory_after_setup,
                    "after_measurement": {
                        "host_rss_bytes": current_rss_bytes(),
                        "host_process_hwm_bytes": process_hwm_bytes(),
                        "host_hwm_scope": "entire benchmark runner process",
                    },
                },
                "timing_scope": {
                    "setup": (
                        "PyTorch/dxtb import, persistent Calculator construction, "
                        "parameter materialization, and input tensor allocation"
                    ),
                    "cold": (
                        "Calculator.reset plus one GFN2 singlepoint, optional "
                        "autograd force, and explicit CUDA synchronize"
                    ),
                    "warm": (
                        "same persistent Calculator path; identity-keyed dxtb caches "
                        "are reset inside every measured call"
                    ),
                    "repeated_call_semantics": REPEATED_CALL_SEMANTICS,
                    "excluded": "setup, cleanup, host result download, and correctness",
                },
                "engine_options": {
                    "model": "GFN2-xTB",
                    "dxtb_version": adapter.version,
                    "torch_version": adapter.torch_version,
                    "module_path": adapter.module_path,
                    "accuracy": adapter.accuracy,
                    "max_scc_iterations": adapter.max_iterations,
                    "batch_mode": adapter.batch_mode,
                    "backend": adapter.backend,
                    "device_id": adapter.device_id,
                    "thread_control": adapter.thread_control,
                    "cache_policy": "Calculator.reset inside each measured call",
                },
            }
        )
        return row
    except DxtbError as exc:
        if adapter is None:
            return unavailable_row(cell, str(exc))
        row = base_row(cell)
        row.update({"availability": "error", "error": str(exc)})
        return row
    except (BenchmarkError, conformance.ConformanceError, OSError) as exc:
        row = base_row(cell)
        row.update({"availability": "error", "error": str(exc)})
        return row
    finally:
        if adapter is not None:
            adapter.close()


def dxtb_cells(args: argparse.Namespace) -> Iterable[Cell]:
    """Yield requested persistent dxtb CPU/CUDA baseline coordinates."""
    for workload in args.workloads:
        for property_name in args.properties:
            for batch_size in args.batch_sizes:
                for backend in args.dxtb_backends:
                    memory_mode = "host" if backend == "cpu" else "device"
                    yield Cell(
                        "dxtb",
                        backend,
                        memory_mode,
                        workload,
                        property_name,
                        batch_size,
                    )


def base_row(cell: Cell) -> dict[str, Any]:
    """Create stable identity fields shared by available and unavailable rows."""
    identifiers = workload_case_ids(cell.workload, cell.batch_size)
    row = {
        "engine": cell.engine,
        "backend": cell.backend,
        "memory_mode": cell.memory_mode,
        "workload": cell.workload,
        "property": cell.property,
        "batch_size": cell.batch_size,
    }
    if cell.workload in WORKLOAD_CASES:
        row["case_id"] = identifiers[0]
    else:
        row["case_ids"] = list(identifiers)
    return row


def unavailable_row(cell: Cell, reason: str) -> dict[str, Any]:
    """Preserve a requested matrix coordinate instead of silently dropping it."""
    row = base_row(cell)
    row.update({"availability": "unavailable", "unavailable_reason": reason})
    return row


def discover_reference(engine: str, explicit: Path | None) -> dict[str, Any]:
    """Record source revision, executable discovery, and honest adapter scope."""
    executable = explicit or (Path(value) if (value := shutil.which(engine)) else None)
    version = None
    if executable is not None and executable.is_file():
        version = run_text((str(executable), "--version"))
    return {
        "source": git_state(REFERENCE_REPOSITORIES[engine]),
        "executable": str(executable) if executable is not None else None,
        "executable_sha256": sha256_file(executable)
        if executable is not None
        else None,
        "version_output": version,
        "command_template": REFERENCE_COMMANDS[engine],
        "process_model": (
            "CLI template; process startup must be included in any timing generated by "
            "this provisional adapter"
        ),
    }


def reference_rows(
    engine: str, args: argparse.Namespace, metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Emit provisional rows until persistent baseline adapters are available."""
    reason = (
        f"provisional {engine} CLI adapter is not timed yet; command template and "
        "executable/version discovery are recorded in metadata.references"
    )
    executable = metadata["references"][engine]["executable"]
    if executable is None:
        reason = f"{engine} executable unavailable; " + reason
    rows = [
        unavailable_row(
            Cell(engine, "cpu", "host", workload, property_name, batch_size),
            reason,
        )
        for workload in args.workloads
        for property_name in args.properties
        for batch_size in args.batch_sizes
    ]
    return rows


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Capture revisions, hardware, runtime versions, and relevant environment."""
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    cuda_root = Path(args.cuda_root)
    nvcc = cuda_root / "bin" / "nvcc"
    references = {
        "tblite": discover_reference("tblite", args.tblite_executable),
        "xtb": discover_reference("xtb", args.xtb_executable),
        "dxtb": discover_reference("dxtb", args.dxtb_executable),
    }
    references["xtb"].update(
        {
            "library": str(args.xtb_library.resolve())
            if args.xtb_library is not None and args.xtb_library.is_file()
            else None,
            "library_sha256": sha256_file(args.xtb_library)
            if args.xtb_library is not None
            else None,
            "adapter": "persistent public C API",
            "process_model": (
                "persistent in-process public C API; CLI template is provenance only"
            ),
            "thread_contract": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
            },
        }
    )
    references["tblite"].update(
        {
            "library": str(args.tblite_library.resolve())
            if args.tblite_library is not None and args.tblite_library.is_file()
            else None,
            "library_sha256": sha256_file(args.tblite_library)
            if args.tblite_library is not None
            else None,
            "adapter": "persistent public C API",
            "process_model": (
                "persistent in-process public C API; CLI template is provenance only"
            ),
            "thread_contract": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
            },
        }
    )
    references["dxtb"].update(
        {
            "source": git_state(args.dxtb_source)
            if args.dxtb_source is not None
            else None,
            "adapter": "persistent in-process PyTorch API",
            "process_model": (
                "one runner process with persistent Calculator/tensors; CLI template "
                "is provenance only"
            ),
            "requested_backends": list(args.dxtb_backends),
            "thread_contract": {
                "torch_threads": args.dxtb_cpu_threads,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "runner": {
            "python": sys.version,
            "platform": platform.platform(),
            "xtbloom_source": git_state(REPOSITORY_ROOT),
            "xtbloom_library": str(args.library.resolve()),
            "xtbloom_library_sha256": sha256_file(args.library),
        },
        "hardware": {
            "hostname": platform.node(),
            "cpu_model": cpu_model,
            "logical_cpu_count": os.cpu_count(),
            "nvidia_smi": run_text(("nvidia-smi", "-L")),
        },
        "cuda": {
            "root": str(cuda_root),
            "nvcc_version": run_text((str(nvcc), "--version"))
            if nvcc.is_file()
            else None,
            "driver": run_text(
                ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
            ),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "XTBLOOM_CUDA_SHELL_PAIR_SCHEDULE",
                "MKL_INTERFACE_LAYER",
                "MKL_THREADING_LAYER",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "references": references,
    }


def xtbloom_cells(args: argparse.Namespace) -> Iterable[Cell]:
    """Yield CPU host and CUDA host/device/mixed public-API coordinates."""
    placements = []
    if "cpu" in args.backends:
        placements.append(("cpu", "host"))
    if "cuda" in args.backends:
        placements.extend(("cuda", mode) for mode in args.cuda_memory_modes)
    for workload in args.workloads:
        for property_name in args.properties:
            for batch_size in args.batch_sizes:
                for backend, memory_mode in placements:
                    yield Cell(
                        "xtbloom",
                        backend,
                        memory_mode,
                        workload,
                        property_name,
                        batch_size,
                    )


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace one reviewable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Flatten the main metrics while retaining nested evidence as JSON columns."""
    fields = [
        "engine",
        "backend",
        "memory_mode",
        "workload",
        "case_id",
        "case_ids",
        "property",
        "batch_size",
        "availability",
        "unavailable_reason",
        "error",
        "setup_ms",
        "cold_latency_ms",
        "warm_median_ms",
        "warm_p95_ms",
        "systems_per_second",
        "correctness_status",
        "max_abs_energy_error_hartree",
        "max_abs_force_error_hartree_per_bohr",
        "max_abs_point_charge_force_error_hartree_per_bohr",
        "warm_samples_ms",
        "memory_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            warm = row.get("warm", {})
            correct = row.get("correctness", {})
            writer.writerow(
                {
                    "engine": row["engine"],
                    "backend": row["backend"],
                    "memory_mode": row["memory_mode"],
                    "workload": row["workload"],
                    "case_id": row.get("case_id"),
                    "case_ids": (
                        json.dumps(row["case_ids"]) if "case_ids" in row else None
                    ),
                    "property": row["property"],
                    "batch_size": row["batch_size"],
                    "availability": row["availability"],
                    "unavailable_reason": row.get("unavailable_reason"),
                    "error": row.get("error"),
                    "setup_ms": row.get("setup_ms"),
                    "cold_latency_ms": row.get("cold_latency_ms"),
                    "warm_median_ms": warm.get("median_ms"),
                    "warm_p95_ms": warm.get("p95_ms"),
                    "systems_per_second": warm.get("systems_per_second_at_median"),
                    "correctness_status": correct.get("status"),
                    "max_abs_energy_error_hartree": correct.get(
                        "max_abs_energy_error_hartree"
                    ),
                    "max_abs_force_error_hartree_per_bohr": correct.get(
                        "max_abs_force_error_hartree_per_bohr"
                    ),
                    "max_abs_point_charge_force_error_hartree_per_bohr": correct.get(
                        "max_abs_point_charge_force_error_hartree_per_bohr"
                    ),
                    "warm_samples_ms": json.dumps(warm.get("samples_ms")),
                    "memory_json": json.dumps(row.get("memory"), sort_keys=True),
                }
            )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """Define one command that can expand from smoke tests to the full matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=conformance.DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPOSITORY_ROOT / "build" / "benchmarks" / "matrix.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPOSITORY_ROOT / "build" / "benchmarks" / "matrix.csv",
    )
    parser.add_argument(
        "--engines",
        type=lambda value: parse_csv_values(value),
        default=("xtbloom", "tblite", "xtb", "dxtb"),
    )
    parser.add_argument(
        "--backends",
        type=lambda value: parse_csv_values(value),
        default=("cpu", "cuda"),
    )
    parser.add_argument(
        "--cuda-memory-modes",
        type=lambda value: parse_csv_values(value),
        default=("host", "device", "mixed"),
    )
    parser.add_argument(
        "--workloads",
        type=lambda value: parse_csv_values(value),
        default=DEFAULT_WORKLOADS,
    )
    parser.add_argument(
        "--properties",
        type=lambda value: parse_csv_values(value),
        default=DEFAULT_PROPERTIES,
    )
    parser.add_argument(
        "--batch-sizes",
        type=lambda value: parse_csv_values(value, int),
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--cuda-root", default=os.environ.get("CUDA_HOME", "/usr/local/cuda")
    )
    parser.add_argument(
        "--tblite-executable",
        type=Path,
        default=(
            DEFAULT_REFERENCE_ENV / "bin" / "tblite"
            if (DEFAULT_REFERENCE_ENV / "bin" / "tblite").is_file()
            else None
        ),
    )
    parser.add_argument(
        "--tblite-library",
        type=Path,
        help=(
            "path to a validated libtblite shared library; omitted rows remain "
            "explicitly unavailable"
        ),
    )
    parser.add_argument(
        "--xtb-executable",
        type=Path,
        default=(
            DEFAULT_REFERENCE_ENV / "bin" / "xtb"
            if (DEFAULT_REFERENCE_ENV / "bin" / "xtb").is_file()
            else None
        ),
    )
    parser.add_argument(
        "--xtb-library",
        type=Path,
        default=(
            DEFAULT_REFERENCE_ENV / "lib" / "libxtb.so"
            if (DEFAULT_REFERENCE_ENV / "lib" / "libxtb.so").is_file()
            else None
        ),
    )
    parser.add_argument("--dxtb-executable", type=Path)
    parser.add_argument(
        "--dxtb-source",
        type=Path,
        default=(
            REFERENCE_REPOSITORIES["dxtb"]
            if REFERENCE_REPOSITORIES["dxtb"].is_dir()
            else None
        ),
    )
    parser.add_argument(
        "--dxtb-backends",
        type=lambda value: parse_csv_values(value),
        default=("cpu", "cuda"),
    )
    parser.add_argument("--dxtb-cpu-threads", type=int, default=1)
    parser.add_argument("--fail-on-correctness", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject typo-driven partial matrices before any expensive inference."""
    allowed = {
        "engines": ({"xtbloom", "tblite", "xtb", "dxtb"}, args.engines),
        "backends": ({"cpu", "cuda"}, args.backends),
        "CUDA memory modes": ({"host", "device", "mixed"}, args.cuda_memory_modes),
        "workloads": (
            set(WORKLOAD_CASES) | set(HETEROGENEOUS_WORKLOAD_CASES),
            args.workloads,
        ),
        "properties": ({"energy", "force"}, args.properties),
        "dxtb backends": ({"cpu", "cuda"}, args.dxtb_backends),
    }
    for label, (choices, selected) in allowed.items():
        unknown = set(selected) - choices
        if unknown:
            raise BenchmarkError(f"unknown {label}: {', '.join(sorted(unknown))}")
    if args.warmups < 0 or args.repetitions <= 0:
        raise BenchmarkError("warmups must be nonnegative and repetitions positive")
    if any(value <= 0 for value in args.batch_sizes):
        raise BenchmarkError("batch sizes must be positive")
    if args.dxtb_cpu_threads <= 0:
        raise BenchmarkError("dxtb CPU threads must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested cells and always retain both machine-readable artifacts."""
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        manifest = conformance.load_json(args.manifest)
        cases = {
            case["id"]: case for case in conformance.selected_cases(manifest, None)
        }
        metadata = environment_metadata(args)
        rows: list[dict[str, Any]] = []
        if "xtbloom" in args.engines:
            for cell in xtbloom_cells(args):
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"RUN {cell.engine} {cell.backend}/{cell.memory_mode} "
                    f"{cell.workload} {cell.property} batch={cell.batch_size}",
                    flush=True,
                )
                row = benchmark_xtbloom_cell(
                    cell,
                    args,
                    manifest,
                    workload_case_sequence(cell.workload, cell.batch_size, cases),
                )
                rows.append(row)
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"  {row['availability']}", flush=True
                )
        if "xtb" in args.engines:
            for cell in xtb_cells(args):
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"RUN {cell.engine} {cell.backend}/{cell.memory_mode} "
                    f"{cell.workload} {cell.property} batch={cell.batch_size}",
                    flush=True,
                )
                row = benchmark_xtb_cell(
                    cell,
                    args,
                    manifest,
                    workload_case_sequence(cell.workload, cell.batch_size, cases),
                )
                rows.append(row)
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"  {row['availability']}", flush=True
                )
        if "tblite" in args.engines:
            for cell in tblite_cells(args):
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"RUN {cell.engine} {cell.backend}/{cell.memory_mode} "
                    f"{cell.workload} {cell.property} batch={cell.batch_size}",
                    flush=True,
                )
                row = benchmark_tblite_cell(
                    cell,
                    args,
                    manifest,
                    workload_case_sequence(cell.workload, cell.batch_size, cases),
                )
                rows.append(row)
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"  {row['availability']}", flush=True
                )
        if "dxtb" in args.engines:
            for cell in dxtb_cells(args):
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"RUN {cell.engine} {cell.backend}/{cell.memory_mode} "
                    f"{cell.workload} {cell.property} batch={cell.batch_size}",
                    flush=True,
                )
                row = benchmark_dxtb_cell(
                    cell,
                    args,
                    manifest,
                    workload_case_sequence(cell.workload, cell.batch_size, cases),
                )
                rows.append(row)
                print(  # noqa: T201 - preserve benchmark CLI progress output
                    f"  {row['availability']}", flush=True
                )
        document = {
            "schema_version": SCHEMA_VERSION,
            "metadata": metadata,
            "protocol": {
                "batch_sizes": list(args.batch_sizes),
                "properties": list(args.properties),
                "workloads": {
                    name: (
                        WORKLOAD_CASES[name]
                        if name in WORKLOAD_CASES
                        else list(HETEROGENEOUS_WORKLOAD_CASES[name])
                    )
                    for name in args.workloads
                },
                "repeated_call_semantics": REPEATED_CALL_SEMANTICS,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "units": {"latency": "ms", "throughput": "systems/s"},
            },
            "rows": rows,
        }
        write_json(args.output_json, document)
        write_csv(args.output_csv, rows)
        print(  # noqa: T201 - preserve benchmark CLI completion output
            f"wrote {args.output_json} and {args.output_csv}"
        )
        errors = [row for row in rows if row["availability"] == "error"]
        failed = [
            row for row in rows if row.get("correctness", {}).get("status") == "fail"
        ]
        if errors:
            return 1
        if args.fail_on_correctness and failed:
            return 2
        return 0
    except (BenchmarkError, conformance.ConformanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
