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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
sys.path.insert(0, str(CONFORMANCE_TOOLS))

import xtbloom_conformance as conformance
import xtbloom_public_api as public_api
from xtbloom_public_api import PublicBatchStorage

try:
    from . import ao_grouping
except ImportError:  # Direct ``python benchmarks/run.py`` execution.
    import ao_grouping

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
NONFINITE_JSON_TAG = "__xtbloom_nonfinite_float__"
NONFINITE_JSON_VALUES = ("NaN", "Infinity", "-Infinity")
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
    case_ids: tuple[str, ...] | None = None


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
        *,
        strict_fresh: bool = False,
        request_charges: bool = False,
        allow_system_failures: bool = False,
    ) -> None:
        self.cell = cell
        self.library_path = library_path
        self.library = public_api._configure_library(library_path)
        if cell.case_ids is None and len(case_sequence) != cell.batch_size:
            raise BenchmarkError("case sequence length must equal the requested batch")
        if cell.case_ids is not None and (
            len(case_sequence) != len(cell.case_ids)
            or len(case_sequence) > cell.batch_size
            or tuple(case["id"] for case in case_sequence) != cell.case_ids
        ):
            raise BenchmarkError("finite case sequence does not match its batch plan")
        self.storage = public_api.assemble_batch(manifest_path, manifest, case_sequence)
        self.allow_system_failures = allow_system_failures
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
        if request_charges:
            self.options.flags |= public_api.XTBLOOM_COMPUTE_ATOMIC_CHARGES
        if strict_fresh:
            self.options.scc_start_mode = public_api.XTBLOOM_SCC_START_FRESH
            self.options.max_scc_iterations = 500
            self.options.charge_tolerance = 1.0e-10
            self.options.energy_tolerance = 1.0e-12
            self.options.electronic_temperature = (
                300.0 * public_api.XTBLOOM_KELVIN_TO_HARTREE
            )
        # These public defaults are recorded in every row and match normal API use.
        self.systems = len(case_sequence)
        self.atoms = len(self.storage.atomic_numbers)
        self.energies = (ctypes.c_double * self.systems)()
        self.forces = (
            (ctypes.c_double * (3 * self.atoms))() if cell.property == "force" else None
        )
        self.charges = (ctypes.c_double * self.atoms)() if request_charges else None
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

    def results(self) -> dict[str, Any]:
        """Download device outputs after timing and validate SCC publication."""
        self.synchronize()
        self.memory.download_outputs()
        failures = [
            "system "
            f"{index}: status={self.statuses[index]}, "
            f"converged={self.converged[index]}, "
            f"iterations={self.iterations[index]}"
            for index in range(self.systems)
            if self.statuses[index] != public_api.XTBLOOM_STATUS_SUCCESS
            or self.converged[index] != 1
        ]
        if failures and not self.allow_system_failures:
            raise BenchmarkError("; ".join(failures))
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


def case_atomic_numbers(
    manifest_path: Path, manifest: dict[str, Any], case: dict[str, Any]
) -> tuple[int, ...]:
    """Read QM atom identities using the same input parsers as the public runner."""
    input_path = canonical_case_input_path(manifest_path, case)
    if public_api.is_periodic_manifest(manifest):
        document = public_api.periodic_gfn2.parse_turbomole(input_path)
        symbol_numbers = {
            symbol.lower(): number
            for number, symbol in enumerate(conformance.ELEMENT_SYMBOLS)
            if symbol
        }
        try:
            return tuple(
                symbol_numbers[symbol.lower()] for symbol in document["symbols"]
            )
        except KeyError as exc:
            raise BenchmarkError(
                f"case {case['id']} has an element absent from the GFN2 basis"
            ) from exc
    if case.get("input_schema") == "qmmm-v1":
        hardness = (
            manifest.get("reference_engines", {})
            .get("xtb", {})
            .get("point_charge_hardness_hartree")
        )
        if hardness is None:
            raise BenchmarkError(
                "point-charge case lacks the pinned GFN2 hardness table"
            )
        document = conformance.load_qmmm_input(input_path, case, hardness)
        return tuple(int(number) for number in document["qm"]["atomic_numbers"])
    document = conformance.load_turbomole_coord(input_path, case)
    return tuple(int(number) for number in document["atomic_numbers"])


def canonical_resolved_path(path: Path, description: str) -> Path:
    """Resolve one provenance/input path to an existing canonical file path."""
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BenchmarkError(f"cannot resolve {description} {path}: {exc}") from exc


def canonical_case_input_path(manifest_path: Path, case: dict[str, Any]) -> Path:
    """Resolve a manifest input with the same repository-root rule as adapters."""
    return canonical_resolved_path(
        conformance.resolve_manifest_path(manifest_path, case["input"]),
        f"input for case {case['id']}",
    )


def canonical_path_label(path: Path) -> str:
    """Use a repository-relative provenance label when the file is in-tree."""
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def provenance_file_record(path: Path, description: str) -> dict[str, Any]:
    """Hash a resolved file incrementally without retaining its contents."""
    resolved = canonical_resolved_path(path, description)
    try:
        digest = sha256_file(resolved)
        if digest is None:
            raise BenchmarkError(f"{description} is not a regular file: {resolved}")
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        raise BenchmarkError(
            f"cannot hash or stat {description} {resolved}: {exc}"
        ) from exc
    return {
        "path": canonical_path_label(resolved),
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def provenance_case_ids(args: argparse.Namespace) -> tuple[str, ...]:
    """Return each input identity selected by the finite list or matrix once."""
    if args.case_ids is not None:
        return tuple(dict.fromkeys(args.case_ids))
    return tuple(
        dict.fromkeys(
            case_id
            for workload in args.workloads
            for batch_size in args.batch_sizes
            for case_id in workload_case_ids(workload, batch_size)
        )
    )


def input_provenance(
    manifest_path: Path,
    case_ids: Sequence[str],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build archival manifest/input digests outside all measured intervals."""
    manifest_record = provenance_file_record(manifest_path, "manifest")
    selected_inputs = []
    for case_id in case_ids:
        case = cases.get(case_id)
        if case is None:
            raise BenchmarkError(
                f"selected case {case_id!r} is absent from the manifest"
            )
        input_path = canonical_case_input_path(manifest_path, case)
        selected_inputs.append(
            {
                "case_id": case_id,
                "manifest_input": case["input"],
                **provenance_file_record(input_path, f"input for case {case_id}"),
            }
        )
    return {
        "manifest": manifest_record,
        "selected_inputs": selected_inputs,
        "hashing_scope": (
            "SHA-256 provenance only; computed before benchmark cells and excluded "
            "from planning_ms and all sweep timings"
        ),
    }


def finite_case_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    case_ids: tuple[str, ...],
    batch_cap: int,
) -> tuple[ao_grouping.AOGroupingPlan, float]:
    """Plan one finite list and account for input inspection and AO analysis."""
    started = time.perf_counter_ns()
    basis_sha256 = None
    counts: dict[str, int] | None = None
    if args.ao_grouping == "exact-ao":
        if public_api.model_tag(manifest) != public_api.XTBLOOM_MODEL_GFN2_XTB:
            raise BenchmarkError("exact-AO grouping requires a GFN2 manifest")
        basis_counts, basis_sha256 = ao_grouping.load_gfn2_basis_ao_counts(
            REPOSITORY_ROOT / "data" / "parameters" / "gfn2.json"
        )
        counts = {
            case_id: ao_grouping.count_gfn2_aos(
                case_atomic_numbers(args.manifest, manifest, cases[case_id]),
                basis_counts,
            )
            for case_id in case_ids
        }
    plan = ao_grouping.make_plan(
        case_ids,
        batch_cap,
        strategy=args.ao_grouping,
        ao_counts_by_case_id=counts,
        basis_sha256=basis_sha256,
    )
    planning_ms = (time.perf_counter_ns() - started) * 1.0e-6
    return plan, planning_ms


def _max_abs_error(actual: object, expected: object) -> float | None:
    """Return a finite vector error, or ``None`` when shape/data are unusable."""
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return None
        values = list(zip(actual, expected, strict=True))
    else:
        values = [(actual, expected)]
    errors = []
    for actual_value, expected_value in values:
        try:
            actual_float = float(actual_value)
            expected_float = float(expected_value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return None
        errors.append(abs(actual_float - expected_float))
    return max(errors, default=0.0)


def finite_case_correctness(
    case: dict[str, Any],
    expected: dict[str, Any],
    result: dict[str, Any],
    manifest: dict[str, Any],
    property_name: str,
) -> dict[str, Any]:
    """Compare available per-ID outputs with the pinned public conformance oracle."""
    comparisons = {
        "energy_hartree": ("energy_hartree", "energy"),
        "forces_hartree_per_bohr": (
            "forces_hartree_per_bohr",
            "forces",
        ),
        "atomic_charges_e": ("partial_charges_e", "charges"),
        "point_charge_forces_hartree_per_bohr": (
            "point_charge_forces_hartree_per_bohr",
            "point_charge_forces",
        ),
    }
    tolerances = manifest.get("tolerances", {})
    errors: dict[str, float | None] = {}
    skipped: list[str] = []
    passed = (
        result["status"] == public_api.XTBLOOM_STATUS_SUCCESS
        and result["scc_converged"] == 1
    )
    for key in (
        "energy_hartree",
        "forces_hartree_per_bohr",
        "atomic_charges_e",
        "point_charge_forces_hartree_per_bohr",
    ):
        value = result.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if any(not math.isfinite(float(component)) for component in values):
            passed = False
    diagnostic_only = case.get("oracle_role") in {
        "diagnostic-unbackgrounded-charged",
        "diagnostic-no-independent-reference",
    }
    required_reference_properties = {
        "energy_hartree",
        "partial_charges_e",
    }
    if property_name == "force":
        required_reference_properties.add("forces_hartree_per_bohr")
        if result.get("point_charge_forces_hartree_per_bohr"):
            required_reference_properties.add("point_charge_forces_hartree_per_bohr")
    oracle_properties = case.get("xtbloom_oracle_properties")
    for actual_key, (expected_key, tolerance_key) in comparisons.items():
        requested = actual_key == "energy_hartree" or actual_key == "atomic_charges_e"
        requested = requested or (
            property_name == "force"
            and actual_key
            in {
                "forces_hartree_per_bohr",
                "point_charge_forces_hartree_per_bohr",
            }
        )
        if not requested or expected_key not in expected:
            continue
        if diagnostic_only or (
            oracle_properties is not None and expected_key not in oracle_properties
        ):
            skipped.append(expected_key)
            continue
        actual_value = result.get(actual_key)
        error = _max_abs_error(actual_value, expected[expected_key])
        errors[expected_key] = error
        tolerance = case.get("tolerances", {}).get(expected_key)
        if tolerance is None:
            tolerance = tolerances[tolerance_key]["atol"]
        if error is None or error > float(tolerance):
            passed = False

    if case.get("oracle_role") == "diagnostic-no-independent-reference":
        reference_validation = "no independent reference; status/finite outputs checked"
    elif diagnostic_only:
        reference_validation = "diagnostic-only oracle; finite status checked"
    elif errors:
        reference_validation = "committed independent conformance golden"
    else:
        reference_validation = "no applicable independent reference property"
    return {
        "status": "pass" if passed else "fail",
        "reference_validation": reference_validation,
        "independent_reference_pass": (
            passed and required_reference_properties.issubset(errors)
        ),
        "missing_reference_properties": sorted(
            required_reference_properties - errors.keys()
        ),
        "max_abs_errors": errors,
        "skipped_oracle_properties": skipped,
    }


def finite_sweep_outcome(
    sweep_index: int, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Retain per-ID convergence and correctness for one measured sweep."""
    case_results = []
    successful_ids = []
    failed_system_ids = []
    correctness_failure_ids = []
    for result in results:
        case_id = result["case_id"]
        system_failed = (
            result["status"] != public_api.XTBLOOM_STATUS_SUCCESS
            or result["scc_converged"] != 1
        )
        correctness = result["correctness"]
        if system_failed:
            failed_system_ids.append(case_id)
        else:
            successful_ids.append(case_id)
        if correctness["status"] != "pass":
            correctness_failure_ids.append(case_id)
        case_results.append(
            {
                "case_id": case_id,
                "original_index": result["original_index"],
                "status": result["status"],
                "scc_converged": result["scc_converged"],
                "scc_iterations": result["scc_iterations"],
                "correctness": correctness,
            }
        )
    correctness_status = (
        "fail" if failed_system_ids or correctness_failure_ids else "pass"
    )
    return {
        "sweep_index": sweep_index,
        "case_results": case_results,
        "correctness": {
            "status": correctness_status,
            "case_count": len(case_results),
            "successful_system_ids": successful_ids,
            "failed_system_ids": failed_system_ids,
            "correctness_failure_ids": correctness_failure_ids,
        },
    }


def benchmark_finite_xtbloom_cell(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    plan: ao_grouping.AOGroupingPlan,
    planning_ms: float,
    property_name: str,
) -> dict[str, Any]:
    """Measure complete strict-FRESH host-descriptor sweeps over planned batches."""
    cell = Cell(
        "xtbloom",
        args.backends[0],
        "host",
        "finite-list",
        property_name,
        plan.max_batch_size,
        plan.original_case_ids,
    )
    row = base_row(cell)
    row.update(
        {
            "availability": "available",
            "ao_grouping": plan.strategy,
            "plan_sha256": plan.plan_sha256,
            "planning_ms": planning_ms,
            "total_systems": len(plan.original_case_ids),
            "batch_count": len(plan.batches),
            "actual_batch_sizes": [len(batch.case_ids) for batch in plan.batches],
            "planned_case_order": list(plan.ordered_case_ids),
            "canonical_index_by_case_id": plan.canonical_index_by_case_id,
            "basis_sha256": plan.basis_sha256,
            "ao_count_by_case_id": dict(plan.ao_counts_by_case_id),
            "ao_batches": [
                {
                    "ao_count": batch.ao_count,
                    "case_ids": list(batch.case_ids),
                    "canonical_indices": list(batch.canonical_indices),
                    "size": len(batch.case_ids),
                }
                for batch in plan.batches
            ],
            "grouping_contract": {
                "key": "exact GFN2 spatial AO count"
                if plan.strategy == "exact-ao"
                else None,
                "tie_break": "canonical input index",
                "spin_policy": (
                    "whole systems retain manifest spin metadata; "
                    "spin is not a grouping key"
                ),
                "outcome_policy": (
                    "planning does not read SCC results or convergence outcomes"
                ),
                "tail_policy": (
                    "retain a final batch smaller than the cap; never pad or drop cases"
                ),
                "batch_size_cap": plan.max_batch_size,
            },
        }
    )
    sweep_samples: list[dict[str, Any]] = []
    warmup_samples: list[dict[str, Any]] = []
    memory_snapshots: list[dict[str, Any]] = []
    measurement_sweeps: list[dict[str, Any]] = []
    row["measurement_sweeps"] = measurement_sweeps
    final_results: tuple[dict[str, Any], ...] = ()
    rss_before = current_rss_bytes()

    def run_sweep(
        capture_results: bool,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        sweep_start = time.perf_counter_ns()
        preparation_ms = 0.0
        compute_ms = 0.0
        publication_ms = 0.0
        batch_outputs: list[dict[str, Any]] = []
        batch_case_correctness: dict[str, Any] = {}
        sweep_memory: list[dict[str, Any]] = []
        for batch in plan.batches:
            prep_start = time.perf_counter_ns()
            batch_cell = Cell(
                cell.engine,
                cell.backend,
                cell.memory_mode,
                cell.workload,
                cell.property,
                plan.max_batch_size,
                batch.case_ids,
            )
            case_sequence = tuple(cases[case_id] for case_id in batch.case_ids)
            adapter = XTBloomAdapter(
                args.library,
                args.manifest,
                manifest,
                case_sequence,
                batch_cell,
                args.device_id,
                args.cpu_threads,
                strict_fresh=True,
                request_charges=True,
                allow_system_failures=True,
            )
            preparation_ms += (time.perf_counter_ns() - prep_start) * 1.0e-6
            try:
                compute_ms += timed_invoke(adapter)
                publish_start = time.perf_counter_ns()
                output = adapter.results()
                required_outputs = ["atomic_charges_e"]
                if property_name == "force":
                    required_outputs.append("forces_hartree_per_bohr")
                    if adapter.storage.point_charge_values:
                        required_outputs.append("point_charge_forces_hartree_per_bohr")
                batch_outputs.extend(
                    ao_grouping.split_batch_results(
                        batch.case_ids,
                        adapter.storage.atom_offsets,
                        adapter.storage.point_charge_offsets,
                        output,
                        tuple(required_outputs),
                    )
                )
                for case_slice in adapter.storage.slices:
                    batch_case_correctness[case_slice.case["id"]] = (
                        case_slice.case,
                        case_slice.expected,
                    )
                publication_ms += (time.perf_counter_ns() - publish_start) * 1.0e-6
                sweep_memory.append(
                    {
                        "case_ids": list(batch.case_ids),
                        "ao_count": batch.ao_count,
                        "snapshot": adapter.memory_snapshot(),
                    }
                )
            finally:
                adapter.close()

        scatter_start = time.perf_counter_ns()
        restored = ao_grouping.scatter_case_results(plan, batch_outputs)
        scatter_ms = (time.perf_counter_ns() - scatter_start) * 1.0e-6
        sweep_ms = (time.perf_counter_ns() - sweep_start) * 1.0e-6
        correctness_validation_ms = 0.0
        if capture_results:
            correctness_start = time.perf_counter_ns()
            for result in restored:
                case, expected = batch_case_correctness[result["case_id"]]
                result["correctness"] = finite_case_correctness(
                    case, expected, result, manifest, property_name
                )
            correctness_validation_ms = (
                time.perf_counter_ns() - correctness_start
            ) * 1.0e-6
        return (
            {
                "preparation_ms": preparation_ms,
                "compute_ms": compute_ms,
                "publication_ms": publication_ms,
                "scatter_ms": scatter_ms,
                "correctness_validation_ms": correctness_validation_ms,
                "end_to_end_ms": sweep_ms,
                "memory_snapshots": sweep_memory,
            },
            restored if capture_results else (),
        )

    try:
        for _ in range(args.warmups):
            timing, _ = run_sweep(capture_results=False)
            timing.pop("memory_snapshots", None)
            warmup_samples.append(timing)
        for sweep_index in range(1, args.repetitions + 1):
            timing, results = run_sweep(capture_results=True)
            sweep_memory = timing.pop("memory_snapshots", [])
            memory_snapshots.extend(sweep_memory)
            sweep_samples.append(timing)
            final_results = results
            sweep_outcome = finite_sweep_outcome(sweep_index, results)
            sweep_outcome["memory_snapshots"] = sweep_memory
            measurement_sweeps.append(sweep_outcome)
            row["case_results"] = list(final_results)
    except public_api.BackendUnavailable as exc:
        row.update({"availability": "unavailable", "unavailable_reason": str(exc)})
        return row
    except (
        BenchmarkError,
        ao_grouping.AOGroupingError,
        conformance.ConformanceError,
        OSError,
    ) as exc:
        row.update({"availability": "error", "error": str(exc)})
        return row

    # Keep output slices from the final measured sweep, but retain context-held
    # memory snapshots from every measured sweep before context destruction.
    row["case_results"] = list(final_results)
    row["case_results_scope"] = "full output slices from the final measured sweep"
    row["memory"] = {
        "host_rss_before_setup_bytes": rss_before,
        "host_peak_rss_bytes": max(
            (entry["snapshot"]["host_process_hwm_bytes"] for entry in memory_snapshots),
            default=process_hwm_bytes(),
        ),
        "batch_context_snapshots": memory_snapshots,
        "scope": (
            "all measured sweeps, one batch context at a time; "
            "samples taken before context destruction"
        ),
    }
    row["timing"] = {
        "warmups": warmup_samples,
        "samples": sweep_samples,
        "preparation_ms_median": statistics.median(
            item["preparation_ms"] for item in sweep_samples
        ),
        "compute_ms_median": statistics.median(
            item["compute_ms"] for item in sweep_samples
        ),
        "publication_ms_median": statistics.median(
            item["publication_ms"] for item in sweep_samples
        ),
        "scatter_ms_median": statistics.median(
            item["scatter_ms"] for item in sweep_samples
        ),
        "correctness_validation_ms_median": statistics.median(
            item["correctness_validation_ms"] for item in sweep_samples
        ),
        "end_to_end_ms": timing_summary(
            [item["end_to_end_ms"] for item in sweep_samples],
            len(plan.original_case_ids),
        ),
        "one_shot_total_ms": planning_ms + sweep_samples[0]["end_to_end_ms"],
        "reusable_plan_mean_total_ms_per_sweep": (
            planning_ms / args.repetitions
            + statistics.mean(item["end_to_end_ms"] for item in sweep_samples)
        ),
        "plan_amortization": (
            "plan_ms / number_of_reuses is added per sweep only when the same ordered "
            "case IDs, input bytes, GFN2 parameter hash, and batch cap remain valid"
        ),
        "scope": (
            "end-to-end includes per-batch input assembly, context and descriptor "
            "setup, synchronous strict-FRESH compute with CUDA completion sync "
            "when applicable, output publication, context-held memory sampling, "
            "context destruction, and canonical scatter; oracle comparisons are "
            "reported separately"
        ),
    }
    correctness_failures = {
        case_id
        for sweep in measurement_sweeps
        for case_id in sweep["correctness"]["correctness_failure_ids"]
    }
    failed_systems = {
        case_id
        for sweep in measurement_sweeps
        for case_id in sweep["correctness"]["failed_system_ids"]
    }
    successful_every_sweep = set(plan.original_case_ids)
    for sweep in measurement_sweeps:
        successful_every_sweep.intersection_update(
            sweep["correctness"]["successful_system_ids"]
        )
    ordered_correctness_failures = [
        case_id for case_id in plan.original_case_ids if case_id in correctness_failures
    ]
    ordered_failed_systems = [
        case_id for case_id in plan.original_case_ids if case_id in failed_systems
    ]
    row["correctness"] = {
        "status": (
            "fail"
            if any(
                sweep["correctness"]["status"] != "pass" for sweep in measurement_sweeps
            )
            else "pass"
        ),
        "reference": "per-ID independent property comparisons where applicable",
        "case_count": len(final_results),
        "sweep_count": len(measurement_sweeps),
        "successful_system_ids": [
            case_id
            for case_id in plan.original_case_ids
            if case_id in successful_every_sweep
        ],
        "failed_system_ids": ordered_failed_systems,
        "correctness_failure_ids": ordered_correctness_failures,
    }
    unqualified_reference_ids = {
        result["case_id"]
        for sweep in measurement_sweeps
        for result in sweep["case_results"]
        if not result["correctness"].get("independent_reference_pass", False)
    }
    row["correctness"]["unqualified_reference_ids"] = [
        case_id
        for case_id in plan.original_case_ids
        if case_id in unqualified_reference_ids
    ]
    row["independent_reference_qualified"] = (
        row["correctness"]["status"] == "pass" and not unqualified_reference_ids
    )
    if not row["independent_reference_qualified"]:
        row["claim_eligible"] = False
    return row


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
    row = {
        "engine": cell.engine,
        "backend": cell.backend,
        "memory_mode": cell.memory_mode,
        "workload": cell.workload,
        "property": cell.property,
        "batch_size": cell.batch_size,
    }
    if cell.case_ids is not None:
        row["case_ids"] = list(cell.case_ids)
        row["case_count"] = len(cell.case_ids)
    elif cell.workload in WORKLOAD_CASES:
        identifiers = workload_case_ids(cell.workload, cell.batch_size)
        row["case_id"] = identifiers[0]
    else:
        identifiers = workload_case_ids(cell.workload, cell.batch_size)
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


def json_safe_value(value: object) -> object:
    """Encode non-finite floats as tagged objects accepted by strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        marker = (
            "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        )
        return {NONFINITE_JSON_TAG: marker}
    if isinstance(value, dict):
        return {key: json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def restore_json_safe_value(value: dict[str, object]) -> object:
    """Decode a tagged non-finite float object created by :func:`json_safe_value`."""
    if len(value) == 1 and value.get(NONFINITE_JSON_TAG) in NONFINITE_JSON_VALUES:
        marker = value[NONFINITE_JSON_TAG]
        if marker == "NaN":
            return float("nan")
        if marker == "Infinity":
            return float("inf")
        return float("-inf")
    return value


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace one standards-compliant JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_safe_value(document),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
        "ao_grouping",
        "plan_sha256",
        "planning_ms",
        "batch_count",
        "preparation_ms_median",
        "compute_ms_median",
        "publication_ms_median",
        "scatter_ms_median",
        "end_to_end_median_ms",
        "end_to_end_samples_ms",
        "one_shot_total_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            warm = row.get("warm", {})
            correct = row.get("correctness", {})
            finite_timing = row.get("timing", {})
            end_to_end = finite_timing.get("end_to_end_ms", {})
            writer.writerow(
                {
                    "engine": row["engine"],
                    "backend": row["backend"],
                    "memory_mode": row["memory_mode"],
                    "workload": row["workload"],
                    "case_id": row.get("case_id"),
                    "case_ids": (
                        json.dumps(json_safe_value(row["case_ids"]), allow_nan=False)
                        if "case_ids" in row
                        else None
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
                    "warm_samples_ms": json.dumps(
                        json_safe_value(warm.get("samples_ms")), allow_nan=False
                    ),
                    "memory_json": json.dumps(
                        json_safe_value(row.get("memory")),
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    "ao_grouping": row.get("ao_grouping"),
                    "plan_sha256": row.get("plan_sha256"),
                    "planning_ms": row.get("planning_ms"),
                    "batch_count": row.get("batch_count"),
                    "preparation_ms_median": finite_timing.get("preparation_ms_median"),
                    "compute_ms_median": finite_timing.get("compute_ms_median"),
                    "publication_ms_median": finite_timing.get("publication_ms_median"),
                    "scatter_ms_median": finite_timing.get("scatter_ms_median"),
                    "end_to_end_median_ms": end_to_end.get("median_ms"),
                    "end_to_end_samples_ms": json.dumps(
                        json_safe_value(end_to_end.get("samples_ms")),
                        allow_nan=False,
                    ),
                    "one_shot_total_ms": finite_timing.get("one_shot_total_ms"),
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
    finite_cases = parser.add_mutually_exclusive_group()
    finite_cases.add_argument(
        "--case-ids",
        type=lambda value: parse_csv_values(value),
        help="finite manifest case IDs in canonical input order",
    )
    finite_cases.add_argument(
        "--case-ids-file",
        type=Path,
        help="UTF-8 file with one manifest case ID per line",
    )
    parser.add_argument(
        "--ao-grouping",
        choices=("original", "exact-ao"),
        default="original",
        help="finite-list strategy; exact-ao is opt-in and original remains default",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cuda-root", default="/group/software/cuda-12.9.1")
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
    parser.add_argument(
        "--require-available",
        action="store_true",
        help="exit nonzero if any requested benchmark coordinate is unavailable",
    )
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
    finite_list_requested = args.case_ids is not None or args.case_ids_file is not None
    if not finite_list_requested and args.ao_grouping != "original":
        raise BenchmarkError("--ao-grouping requires --case-ids or --case-ids-file")
    if finite_list_requested:
        if args.case_ids is not None:
            if not args.case_ids:
                raise BenchmarkError("finite case list must not be empty")
            if len(set(args.case_ids)) != len(args.case_ids):
                raise BenchmarkError("finite case IDs must be unique")
        if args.engines != ("xtbloom",):
            raise BenchmarkError("finite-list grouping requires --engines xtbloom")
        if len(args.backends) != 1:
            raise BenchmarkError(
                "finite-list grouping requires exactly one backend: cpu or cuda"
            )
        if args.backends == ("cuda",) and args.cuda_memory_modes != ("host",):
            raise BenchmarkError("finite-list CUDA grouping requires host descriptors")


def read_case_id_file(path: Path) -> tuple[str, ...]:
    """Read a finite ordered manifest selection, ignoring blank and comment lines."""
    try:
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as exc:
        raise BenchmarkError(f"cannot read finite case ID file {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run requested cells and always retain both machine-readable artifacts."""
    args = build_parser().parse_args(argv)
    try:
        if args.case_ids_file is not None:
            args.case_ids = read_case_id_file(args.case_ids_file)
        validate_args(args)
        manifest = conformance.load_json(args.manifest)
        manifest_cases = conformance.selected_cases(manifest, None)
        cases: dict[str, dict[str, Any]] = {}
        for case in manifest_cases:
            case_id = case["id"]
            if case_id in cases:
                raise BenchmarkError(f"manifest contains duplicate case ID {case_id!r}")
            cases[case_id] = case
        if args.case_ids is not None:
            missing = [case_id for case_id in args.case_ids if case_id not in cases]
            if missing:
                raise BenchmarkError(
                    "finite case IDs are absent from the manifest: "
                    + ", ".join(missing)
                )
        metadata = environment_metadata(args)
        provenance = input_provenance(args.manifest, provenance_case_ids(args), cases)
        rows: list[dict[str, Any]] = []
        if args.case_ids is not None:
            for property_name in args.properties:
                for batch_cap in args.batch_sizes:
                    plan, planning_ms = finite_case_plan(
                        args, manifest, cases, args.case_ids, batch_cap
                    )
                    print(  # noqa: T201 - preserve benchmark CLI progress output
                        "RUN xtbloom "
                        f"{args.backends[0]}/host finite-list {plan.strategy} "
                        f"{property_name} cap={batch_cap} systems={len(args.case_ids)}",
                        flush=True,
                    )
                    row = benchmark_finite_xtbloom_cell(
                        args,
                        manifest,
                        cases,
                        plan,
                        planning_ms,
                        property_name,
                    )
                    rows.append(row)
                    print(  # noqa: T201 - preserve benchmark CLI progress output
                        f"  {row['availability']}", flush=True
                    )
        if args.case_ids is None and "xtbloom" in args.engines:
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
        if args.case_ids is None and "xtb" in args.engines:
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
        if args.case_ids is None and "tblite" in args.engines:
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
        if args.case_ids is None and "dxtb" in args.engines:
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
            "provenance": provenance,
            "nonfinite_json_encoding": {
                "tag": NONFINITE_JSON_TAG,
                "values": list(NONFINITE_JSON_VALUES),
            },
            "protocol": {
                "batch_sizes": list(args.batch_sizes),
                "finite_case_ids": (
                    list(args.case_ids) if args.case_ids is not None else None
                ),
                "ao_grouping": args.ao_grouping,
                "strict_fresh_host_descriptors": args.case_ids is not None,
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
                "fail_on_correctness": args.fail_on_correctness,
                "require_available": args.require_available,
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
        unavailable = [row for row in rows if row["availability"] == "unavailable"]
        if errors:
            return 1
        if args.fail_on_correctness and failed:
            return 2
        if args.require_available and unavailable:
            return 3
        return 0
    except (BenchmarkError, conformance.ConformanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
