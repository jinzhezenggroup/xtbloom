#!/usr/bin/env python3
"""Cross-engine GFN2-xTB atom-count scaling benchmark with distinct systems.

This runner measures public single-point GFN2-xTB inference latency. Context,
calculator, descriptor, and cold-reset construction happen outside the timed
region; the timed boundary begins immediately before the public inference call
and ends after synchronous host-visible energy and force publication:

- ``gpuxtb`` CPU and CUDA through the committed ctypes conformance adapter
  (``gpuxtb_public_api``), exactly like ``natoms_scaling.py``;
- ``xtb`` and ``tblite`` through their persistent public C API adapters;
- ``dxtb`` through the persistent in-process PyTorch adapter.

For every molecule size and batch size the batch is built from *distinct*
seeded thermal-like conformers of the same alkane stoichiometry (identical
atomic numbers, slightly different coordinates), so an engine cannot win a
batch row by reusing one identical geometry.  At batch size one the first slot
keeps the clean ideal alkane geometry. The sweep therefore reports comparable
per-call latency rather than process or calculator setup time. ``auto-warm``
rows are WARM steady state after an untimed cold seed; ``cold`` rows clear
electronic state before every timed inference call.

An optional MD-trajectory mode measures per-frame latency over a sequence of
nearly identical frames (positions mutated in place through the persistent
host descriptors), exercising gpuxtb's sequential geometry path while the
reference engines update their persistent structures per frame.

Artifacts are two explicit JSON/CSV paths in the same style as the other
benchmark harnesses. JSON is authoritative and retains raw timing samples,
complete final energy/force vectors, force digests for every repetition, run
identity, hardware, threads, correctness, and per-row diagnostics; CSV is the
compact row summary. Final publication evidence rejects a dirty repository and
can compare every row with a clean independent xTB reference artifact.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import csv
import ctypes
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = REPOSITORY_ROOT / "tools" / "conformance"
if str(CONFORMANCE_TOOLS) not in sys.path:
    sys.path.insert(0, str(CONFORMANCE_TOOLS))
public_api = importlib.import_module("gpuxtb_public_api")

try:
    from .natoms_scaling import Molecule, SystemSlice, make_alkane
except ImportError:  # Direct ``python benchmarks/natoms_cross_engine.py`` execution.
    from natoms_scaling import Molecule, SystemSlice, make_alkane

try:
    from .xtb_adapter import XtbAdapter, XtbError
except ImportError:
    from xtb_adapter import XtbAdapter, XtbError

try:
    from .tblite_adapter import TbliteAdapter, TbliteError
except ImportError:
    from tblite_adapter import TbliteAdapter, TbliteError

try:
    from .dxtb_adapter import DxtbAdapter, DxtbError
except ImportError:
    from dxtb_adapter import DxtbAdapter, DxtbError

SCHEMA_VERSION = 2
DEFAULT_NATOMS = (5, 32, 122, 362, 602, 962)
DEFAULT_BATCH_SIZES = (1, 128)
SUPPORTED_ENGINES = (
    "gpuxtb-cpu",
    "gpuxtb-cuda",
    "xtb",
    "tblite",
    "dxtb-cpu",
    "dxtb-cuda",
)
DEFAULT_ENGINES = ("gpuxtb-cpu", "gpuxtb-cuda", "xtb", "tblite", "dxtb-cpu")
CROSS_ENGINE_ENERGY_ATOL_HARTREE = 2.0e-3
CROSS_ENGINE_FORCE_ATOL_HARTREE_PER_BOHR = 2.0e-3
PERTURB_SIGMA_BOHR = 0.02
TRAJECTORY_STEP_SIGMA_BOHR = 0.01


class BenchmarkError(RuntimeError):
    """An actionable benchmark request, adapter, or publication failure."""


def parse_csv_values(value: str) -> tuple[Any, ...]:
    """Parse one nonempty comma-separated CLI selection of integers."""
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parts:
        raise BenchmarkError("empty comma-separated selection")
    return tuple(parts)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse one nonempty comma-separated integer selection."""
    try:
        parsed = tuple(int(part.strip()) for part in parse_csv_values(value))
    except ValueError as exc:
        raise BenchmarkError(f"expected integers, got {value!r}") from exc
    if not parsed:
        raise BenchmarkError("empty integer selection")
    return parsed


def sha256_file(path: Path | None) -> str | None:
    """Return the lowercase SHA-256 of a file, or None when it is absent."""
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict[str, Any]:
    """Return clean/dirty and HEAD revision of a git checkout."""

    def run(command: Sequence[str]) -> str | None:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return completed.stdout.strip() or None

    revision = run(("git", "-C", str(path), "rev-parse", "HEAD"))
    status = run(("git", "-C", str(path), "status", "--porcelain"))
    return {
        "head": revision,
        "dirty": bool(status),
    }


def run_text(command: Sequence[str]) -> str | None:
    """Run one diagnostic command and return stripped stdout or None."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() or None


def cpu_model() -> str | None:
    """Read the CPU model name when /proc/cpuinfo is available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def current_rss_bytes() -> int | None:
    """Return current host RSS for this process, or None."""
    try:
        pages = int(Path(f"/proc/{os.getpid()}/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def log(message: str) -> None:
    """Emit one flushable, timestamped progress line on stdout."""
    stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    print(  # noqa: T201 - benchmark CLI progress output
        f"[{stamp}] {message}", flush=True
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the requested nearest-rank percentile of a sample set."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def timing_summary(samples_ms: Sequence[float], batch_size: int) -> dict[str, Any]:
    """Summarize raw latencies without discarding the samples."""
    return {
        "samples_ms": list(samples_ms),
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "systems_per_second_at_median": 1000.0
        * batch_size
        / statistics.median(samples_ms),
    }


@dataclass(frozen=True)
class Cell:
    """One engine/natoms/batch-size benchmark coordinate."""

    engine: str
    natoms: int
    batch_size: int
    cpu_threads: int
    device_id: int


@dataclass(frozen=True)
class ReferenceArtifact:
    """One clean xTB artifact used to qualify dependent engine rows."""

    path: Path
    sha256: str
    metadata: dict[str, Any]
    rows: dict[tuple[int, int], dict[str, Any]]


@dataclass
class BatchStorage:
    """Duck-typed storage accepted by ``gpuxtb_public_api._make_batch``."""

    atom_offsets: list[int]
    atomic_numbers: list[int]
    positions: list[float]
    molecular_charges: list[float]
    unpaired_electrons: list[int]
    spin_channels: list[int]
    point_charge_offsets: list[int]
    point_charge_positions: list[float]
    point_charge_values: list[float]
    point_charge_gammas: list[float]
    slices: list[SystemSlice]
    keepalive: list[Any]


def _perturbed_positions(
    base: Sequence[float],
    rng: random.Random,
    sigma_bohr: float,
) -> tuple[float, ...]:
    """Apply an independent Gaussian displacement to every coordinate."""
    return tuple(float(value) + rng.gauss(0.0, sigma_bohr) for value in base)


def build_batch(
    base_molecule: Molecule,
    batch_size: int,
    seed: int,
    perturb_sigma_bohr: float = PERTURB_SIGMA_BOHR,
) -> BatchStorage:
    """Build a ragged batch of *distinct* conformers of one alkane.

    Slot zero keeps the ideal geometry; every later slot receives a seeded
    independent thermal-like perturbation so the batch never contains two
    identical systems, which would permit an engine to reuse one geometry.
    """
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    slices: list[SystemSlice] = []
    rng = random.Random(seed)
    for slot in range(batch_size):
        if slot == 0:
            slot_positions = tuple(base_molecule.positions_bohr)
        else:
            slot_positions = _perturbed_positions(
                base_molecule.positions_bohr, rng, perturb_sigma_bohr
            )
        begin = len(atomic_numbers)
        atomic_numbers.extend(base_molecule.atomic_numbers)
        positions.extend(slot_positions)
        atom_offsets.append(len(atomic_numbers))
        slices.append(SystemSlice(begin, len(atomic_numbers)))
    return BatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=[0.0] * batch_size,
        unpaired_electrons=[0] * batch_size,
        spin_channels=[1] * batch_size,
        point_charge_offsets=[0] * (batch_size + 1),
        point_charge_positions=[],
        point_charge_values=[],
        point_charge_gammas=[],
        slices=slices,
        keepalive=[],
    )


def build_trajectory(
    base_molecule: Molecule,
    frames: int,
    seed: int,
    step_sigma_bohr: float = TRAJECTORY_STEP_SIGMA_BOHR,
) -> list[tuple[float, ...]]:
    """Return frames of one alkane mutated by a seeded small random walk.

    Successive frames deliberately keep their coordinates close (like an MD
    trajectory), which is exactly the sequential-geometry regime the benchmark
    wants to isolate.
    """
    rng = random.Random(seed)
    frames_out: list[tuple[float, ...]] = []
    current = list(base_molecule.positions_bohr)
    for _ in range(frames):
        current = list(_perturbed_positions(current, rng, step_sigma_bohr))
        frames_out.append(tuple(current))
    return frames_out


def configure_gpuxtb_scc(
    options: Any,  # noqa: ANN401 - ctypes compute-options mirror
    charge_tolerance: float = 1.0e-10,
    energy_tolerance: float = 1.0e-12,
    max_iterations: int = 500,
) -> None:
    """Pin the SCC convergence controls on a gpuxtb options object.

    Defaults match the conformance oracle.  A tolerance-matched benchmark
    (e.g. ``--scc-charge-tolerance 1e-4 --scc-energy-tolerance 1e-4``) lets
    gpuxtb stop at the same accuracy the reference engines use (their
    ``accuracy=1e-4``), removing the strict-tolerance iteration gap.
    """
    options.max_scc_iterations = max_iterations
    options.charge_tolerance = charge_tolerance
    options.energy_tolerance = energy_tolerance


def cuda_synchronize(
    control: Any,  # noqa: ANN401 - ctypes CUDA runtime control mirror
) -> None:
    """Complete all CUDA work at the documented timing boundary."""
    control.runtime.cudaDeviceSynchronize.argtypes = []
    control.runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    status = control.runtime.cudaDeviceSynchronize()
    control._check(status, "cudaDeviceSynchronize")


class GpuxtbRunner:
    """Persistent public-C-ABI runner with in-place position mutation."""

    def __init__(
        self,
        library_path: Path,
        storage: BatchStorage,
        backend: str,
        cpu_threads: int,
        device_id: int,
        scc_charge_tolerance: float = 1.0e-10,
        scc_energy_tolerance: float = 1.0e-12,
        scc_max_iterations: int = 500,
    ) -> None:
        self.library = public_api._configure_library(library_path)
        self.storage = storage
        self.backend = backend
        self.context = public_api._make_context(
            self.library, backend, device_id, cpu_threads
        )
        self.memory = public_api.DescriptorMemory("host", device_id)
        self.has_cuda = backend == "cuda"
        self.cuda_control = None
        if self.has_cuda:
            self.cuda_control = public_api.CudaRuntime(device_id)
        self.batch = public_api._make_batch(
            self.library,
            storage,
            self.memory,
            include_spin_channels=True,
        )
        self.options = public_api.ComputeOptions()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute_options_init(
                ctypes.byref(self.options), ctypes.sizeof(self.options)
            ),
            "gpuxtb_compute_options_init",
        )
        self.options.model = public_api.GPUXTB_MODEL_GFN2_XTB
        self.options.flags = public_api.GPUXTB_COMPUTE_ENERGY
        self.options.flags |= public_api.GPUXTB_COMPUTE_FORCES
        configure_gpuxtb_scc(
            self.options,
            charge_tolerance=scc_charge_tolerance,
            energy_tolerance=scc_energy_tolerance,
            max_iterations=scc_max_iterations,
        )
        systems = len(storage.slices)
        atoms = len(storage.atomic_numbers)
        self.energies = (ctypes.c_double * systems)()
        self.forces = (ctypes.c_double * (3 * atoms))()
        self.iterations = (ctypes.c_int32 * systems)()
        self.converged = (ctypes.c_uint8 * systems)()
        self.statuses = (ctypes.c_int32 * systems)()
        self.result = public_api.BatchResult()
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_batch_result_init(
                ctypes.byref(self.result), ctypes.sizeof(self.result)
            ),
            "gpuxtb_batch_result_init",
        )
        self.result.energies = self.memory.output(self.energies, "energies")
        self.result.forces = self.memory.output(self.forces, "forces")
        self.result.scc_iterations = self.memory.output(
            self.iterations, "scc_iterations"
        )
        self.result.scc_converged = self.memory.output(self.converged, "scc_converged")
        self.result.per_system_status = self.memory.output(
            self.statuses, "per_system_status"
        )
        # Retain the caller-owned positions owner array for in-place mutation
        # so repeated calls can stream updated trajectories through the same
        # host descriptors without rebuilding them. ``input`` appended the
        # owner to ``storage.keepalive``; the position owner is the only
        # c_double array covering the full flattened coordinate set.
        self._position_owner = None
        for owner in storage.keepalive:
            if (
                isinstance(owner, ctypes.Array)
                and getattr(owner, "_type_", None) is ctypes.c_double
                and len(owner) == len(storage.positions)
            ):
                self._position_owner = owner
                break
        if self._position_owner is None:
            raise BenchmarkError("gpuxtb positions owner array is missing")
        self.closed = False

    def set_start_mode(self, mode: str) -> None:
        """Select FRESH or WARM SCC continuation without rebuilding descriptors."""
        if mode not in ("fresh", "warm"):
            raise BenchmarkError(f"unsupported gpuxtb start mode: {mode}")
        self.options.scc_start_mode = (
            public_api.GPUXTB_SCC_START_WARM
            if mode == "warm"
            else public_api.GPUXTB_SCC_START_FRESH
        )

    def set_positions(self, positions: Sequence[float]) -> None:
        """Write a new flattened position vector into the persistent owner."""
        if len(positions) != len(self.storage.positions):
            raise BenchmarkError("trajectory frame length changed")
        for index, value in enumerate(positions):
            self._position_owner[index] = value

    def invoke(self) -> None:
        """Execute one synchronous public batch inference."""
        public_api._call_ok(
            self.library,
            self.library.gpuxtb_compute(
                self.context,
                ctypes.byref(self.batch),
                ctypes.byref(self.options),
                ctypes.byref(self.result),
            ),
            f"gpuxtb {self.backend} inference",
        )
        if self.has_cuda:
            cuda_synchronize(self.cuda_control)

    def snapshot(self) -> dict[str, Any]:
        """Download outputs once and normalize observables."""
        self.memory.download_outputs()
        return {
            "energies_hartree": [float(value) for value in self.energies],
            "forces_hartree_per_bohr": [float(value) for value in self.forces],
            "scc_iterations": [int(value) for value in self.iterations],
            "scc_converged": [int(value) for value in self.converged],
            "per_system_status": [int(value) for value in self.statuses],
        }

    def close(self) -> None:
        """Release descriptors before destroying the backend context."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.cuda_control is not None:
                self.cuda_control.close()
        finally:
            self.library.gpuxtb_context_destroy(self.context)


class ReferenceRunner:
    """Persistent adapter for xtb/tblite/dxtb logical batches."""

    def __init__(
        self,
        engine: str,
        library_path: Path,
        storage: BatchStorage,
        cpu_threads: int,
        device_id: int,
        dxtb_source: Path | None,
    ) -> None:
        if engine == "xtb":
            self.adapter = XtbAdapter(
                library_path,
                storage,
                "force",
                None,
                accuracy=1.0e-4,
                max_iterations=500,
                threads=cpu_threads,
            )
            self.engine = "xtb"
        elif engine == "tblite":
            self.adapter = TbliteAdapter(
                library_path,
                storage,
                "force",
                accuracy=1.0e-4,
                max_iterations=500,
                collect_atomic_charges=False,
                threads=cpu_threads,
            )
            self.engine = "tblite"
        elif engine in ("dxtb-cpu", "dxtb-cuda"):
            backend = "cpu" if engine == "dxtb-cpu" else "cuda"
            self.adapter = DxtbAdapter(
                storage,
                "force",
                backend,
                device_id=device_id,
                cpu_threads=cpu_threads,
                source_root=dxtb_source,
                accuracy=1.0e-4,
                max_iterations=500,
            )
            self.engine = "dxtb"
        else:
            raise BenchmarkError(f"unsupported reference engine: {engine}")
        self._dxtb_adapter = self.adapter if engine.startswith("dxtb") else None
        states = getattr(self.adapter, "states", ())
        if self._dxtb_adapter is not None or not states:
            self._position_slices = []
        else:
            self._position_slices = [
                (state, slice_.atom_begin, slice_.atom_end)
                for state, slice_ in zip(states, storage.slices, strict=True)
            ]

    def set_positions(self, positions: Sequence[float]) -> None:
        """Stream one frame into every persistent system without rebuilding."""
        if self._dxtb_adapter is not None:
            # dxtb holds one torch tensor on the selected device; write each
            # system's atom range into the persistent tensor in place.
            import torch

            flat = torch.tensor(
                list(positions),
                dtype=torch.float64,
                device=self._dxtb_adapter.device,
            )
            tensor = self._dxtb_adapter.positions
            with torch.no_grad():
                tensor.view(-1).copy_(flat)
            return
        if not self._position_slices:
            raise BenchmarkError(f"{self.engine} adapter has no position stream")
        for state, atom_begin, atom_end in self._position_slices:
            state_length = len(state.positions)
            expected = 3 * (atom_end - atom_begin)
            if state_length != expected:
                raise BenchmarkError(
                    f"{self.engine} state positions length {state_length} "
                    f"does not match slice {expected}"
                )
            for index in range(expected):
                state.positions[index] = positions[3 * atom_begin + index]

    def invoke(self) -> None:
        """Run one persistent logical batch inference."""
        self.adapter.invoke()
        if getattr(self.adapter, "backend", None) == "cuda":
            self.adapter.synchronize()

    def restart_scc(self) -> None:
        """Drop convergence state so the next sample is a genuine cold solve.

        Only engines whose persistent adapter can rebuild the SCC state support
        this; the runner no-ops otherwise (gpuxtb FRESH and dxtb reset already
        cold-start every measured call).
        """
        restart = getattr(self.adapter, "restart_scc", None)
        if restart is not None:
            restart()

    def snapshot(self) -> dict[str, Any]:
        """Normalize persistent reference results."""
        output = self.adapter.results()
        return {
            "energies_hartree": list(output["energies_hartree"]),
            "forces_hartree_per_bohr": (
                list(output["forces_hartree_per_bohr"])
                if "forces_hartree_per_bohr" in output
                else None
            ),
            "scc_iterations": None,
            "scc_converged": None,
            "per_system_status": None,
        }

    def close(self) -> None:
        """Release all persistent reference state exactly once."""
        self.adapter.close()


def _force_digest(values: Sequence[float]) -> str:
    """Hash one complete binary64 force vector in canonical little endian."""
    packed = array.array("d", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the maximum absolute difference between equal-length vectors."""
    if len(left) != len(right):
        raise BenchmarkError(
            f"vector length mismatch: observed {len(left)}, expected {len(right)}"
        )
    return max(
        (abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)),
        default=0.0,
    )


def measure_cell(
    runner: Any,  # noqa: ANN401 - gpuxtb/xtb/tblite/dxtb adapter union
    protocol: tuple[int, int],
    cell: Cell,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
    start_policy: str = "auto-warm",
) -> dict[str, Any]:
    """Run warmups and measured samples and return a normalized row fragment.

    ``start_policy`` controls SCC restart semantics for every engine:

    - ``auto-warm``: one untimed cold seed establishes state, then every timed
      sample continues from that converged state (gpuxtb strict WARM;
      xTB/tblite persistent warm state).
    - ``cold``: electronic state is cleared before every timed inference call
      (gpuxtb FRESH; xTB/tblite calculator rebuild; dxtb reset). The reset or
      rebuild itself is deliberately outside the timer, matching gpuxtb's
      persistent-context boundary rather than claiming process/setup latency.
    """
    warmups, repetitions = protocol
    gpuxtb_runner = hasattr(runner, "set_start_mode")
    restart = getattr(runner, "restart_scc", None)

    def cold_start() -> None:
        """Force a genuine cold start for whichever runner variant is active."""
        if gpuxtb_runner:
            runner.set_start_mode("fresh")
        elif restart is not None:
            restart()

    for warmup_index in range(warmups):
        if start_policy == "auto-warm" and gpuxtb_runner:
            runner.set_start_mode("fresh" if warmup_index == 0 else "warm")
        elif start_policy == "cold":
            cold_start()
        runner.invoke()
    raw_samples: list[dict[str, Any]] = []
    force_samples: list[list[float]] = []
    expected_energy_count = cell.batch_size
    expected_force_count = 3 * cell.natoms * cell.batch_size
    for sample_index in range(repetitions):
        if start_policy == "auto-warm":
            if gpuxtb_runner:
                runner.set_start_mode("warm")
        else:
            cold_start()
        start = time.perf_counter_ns()
        runner.invoke()
        elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
        snapshot = runner.snapshot()
        energies = snapshot.get("energies_hartree")
        forces = snapshot.get("forces_hartree_per_bohr")
        if not isinstance(energies, list) or len(energies) != expected_energy_count:
            actual = len(energies) if isinstance(energies, list) else None
            raise BenchmarkError(
                f"inference returned {actual} energies; expected "
                f"{expected_energy_count}"
            )
        if not isinstance(forces, list) or len(forces) != expected_force_count:
            actual = len(forces) if isinstance(forces, list) else None
            raise BenchmarkError(
                f"inference returned {actual} force values; expected "
                f"{expected_force_count}"
            )
        if not all(math.isfinite(float(value)) for value in energies):
            raise BenchmarkError("inference returned non-finite energies")
        if not all(math.isfinite(float(value)) for value in forces):
            raise BenchmarkError("inference returned non-finite forces")
        normalized_forces = [float(value) for value in forces]
        force_samples.append(normalized_forces)
        raw_samples.append(
            {
                "sample_index": sample_index,
                "latency_ms": elapsed_ms,
                "energies_hartree": [float(value) for value in energies],
                "force_count": len(normalized_forces),
                "forces_sha256_binary64_le": _force_digest(normalized_forces),
                "scc_iterations": snapshot["scc_iterations"],
                "scc_converged": snapshot["scc_converged"],
                "per_system_status": snapshot["per_system_status"],
            }
        )
    status_ok = all(
        sample.get("per_system_status") is None
        or all(
            status == public_api.GPUXTB_STATUS_SUCCESS
            for status in sample["per_system_status"]
        )
        for sample in raw_samples
    )
    converged_ok = all(
        sample.get("scc_converged") is None
        or all(value == 1 for value in sample["scc_converged"])
        for sample in raw_samples
    )
    energy_reference = raw_samples[0]["energies_hartree"]
    force_reference = force_samples[0]
    energy_drift = max(
        _max_abs_delta(sample["energies_hartree"], energy_reference)
        for sample in raw_samples
    )
    force_drift = max(
        _max_abs_delta(sample, force_reference) for sample in force_samples
    )
    repeatability_ok = (
        energy_drift <= energy_atol_hartree
        and force_drift <= force_atol_hartree_per_bohr
    )
    latencies = [sample["latency_ms"] for sample in raw_samples]
    iteration_min = iteration_max = None
    iterations = [
        value
        for sample in raw_samples
        if sample["scc_iterations"] is not None
        for value in sample["scc_iterations"]
    ]
    if iterations:
        iteration_min = min(iterations)
        iteration_max = max(iterations)
    fragment = {
        "raw_samples": raw_samples,
        "timing": timing_summary(latencies, cell.batch_size),
        "energies_hartree": raw_samples[-1]["energies_hartree"],
        "forces_hartree_per_bohr": force_samples[-1],
        "iteration_summary": {"min": iteration_min, "max": iteration_max},
        "correctness": {
            "status": (
                "pass" if (status_ok and converged_ok and repeatability_ok) else "fail"
            ),
            "finite_energies": True,
            "finite_forces": True,
            "force_value_count": expected_force_count,
            "scc_converged_ok": converged_ok,
            "scc_status_ok": status_ok,
            "repeatability": {
                "energy_atol_hartree": energy_atol_hartree,
                "max_abs_energy_drift_hartree": energy_drift,
                "force_atol_hartree_per_bohr": force_atol_hartree_per_bohr,
                "max_abs_force_drift_hartree_per_bohr": force_drift,
            },
            "cross_engine": {"status": "not_requested"},
        },
    }
    return fragment


def base_row(cell: Cell) -> dict[str, Any]:
    """Create stable identity fields shared by every row type."""
    return {
        "engine": cell.engine,
        "natoms": cell.natoms,
        "batch_size": cell.batch_size,
        "total_atoms_in_batch": cell.natoms * cell.batch_size,
        "cpu_threads": cell.cpu_threads,
        "device_id": cell.device_id,
    }


def unavailable_row(cell: Cell, reason: str) -> dict[str, Any]:
    """Create an explicitly unavailable row for a missing engine."""
    row = base_row(cell)
    row.update({"availability": "unavailable", "reason": reason})
    return row


def error_row(cell: Cell, error: str) -> dict[str, Any]:
    """Create an error row without inventing unavailable reasons."""
    row = base_row(cell)
    row.update({"availability": "error", "error": error})
    return row


def load_reference_artifact(
    path: Path, allow_dirty_evidence: bool = False
) -> ReferenceArtifact:
    """Load and validate one xTB JSON artifact for dependent runs."""
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
        document = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"cannot read reference artifact {resolved}: {exc}"
        ) from exc
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise BenchmarkError("reference artifact has no metadata object")
    commit = metadata.get("commit") or {}
    if not commit.get("head") or (
        commit.get("dirty") is not False and not allow_dirty_evidence
    ):
        raise BenchmarkError("reference artifact must come from a verified clean HEAD")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(document.get("rows") or []):
        if row.get("availability") != "available":
            continue
        if row.get("engine") != "xtb":
            raise BenchmarkError(
                f"reference row {index} uses {row.get('engine')!r}, expected 'xtb'"
            )
        correctness = row.get("correctness") or {}
        cross_engine = correctness.get("cross_engine") or {}
        if (
            correctness.get("status") != "pass"
            or cross_engine.get("status") != "reference"
        ):
            raise BenchmarkError(f"reference row {index} is not qualified")
        natoms = row.get("natoms")
        batch_size = row.get("batch_size")
        if type(natoms) is not int or type(batch_size) is not int:
            raise BenchmarkError(f"reference row {index} has invalid coordinate")
        energies = row.get("energies_hartree")
        forces = row.get("forces_hartree_per_bohr")
        if not isinstance(energies, list) or len(energies) != batch_size:
            raise BenchmarkError(f"reference row {index} has invalid energies")
        if not isinstance(forces, list) or len(forces) != 3 * natoms * batch_size:
            raise BenchmarkError(f"reference row {index} has invalid forces")
        if not all(math.isfinite(float(value)) for value in energies) or not all(
            math.isfinite(float(value)) for value in forces
        ):
            raise BenchmarkError(f"reference row {index} has non-finite observables")
        key = (natoms, batch_size)
        if key in rows:
            raise BenchmarkError(f"reference artifact duplicates coordinate {key}")
        rows[key] = row
    if not rows:
        raise BenchmarkError("reference artifact has no qualified available rows")
    return ReferenceArtifact(
        resolved,
        hashlib.sha256(payload).hexdigest(),
        metadata,
        rows,
    )


def apply_cross_engine_reference(
    row: dict[str, Any],
    reference: ReferenceArtifact,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
) -> bool:
    """Compare one complete final result with the matching independent xTB row."""
    if row.get("availability") != "available":
        return False
    correctness = row.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("status") != "pass":
        return True
    key = (row["natoms"], row["batch_size"])
    expected = reference.rows.get(key)
    comparison: dict[str, Any] = {
        "status": "missing_reference",
        "reference_engine": "xtb",
        "artifact": str(reference.path),
        "artifact_sha256": reference.sha256,
        "energy_atol_hartree": energy_atol_hartree,
        "force_atol_hartree_per_bohr": force_atol_hartree_per_bohr,
        "max_abs_energy_delta_hartree": None,
        "max_abs_force_delta_hartree_per_bohr": None,
    }
    failed = expected is None
    if expected is not None:
        try:
            energy_delta = _max_abs_delta(
                row["energies_hartree"], expected["energies_hartree"]
            )
            force_delta = _max_abs_delta(
                row["forces_hartree_per_bohr"],
                expected["forces_hartree_per_bohr"],
            )
        except (BenchmarkError, KeyError, TypeError) as exc:
            comparison["status"] = "fail"
            comparison["error"] = str(exc)
            failed = True
        else:
            comparison["max_abs_energy_delta_hartree"] = energy_delta
            comparison["max_abs_force_delta_hartree_per_bohr"] = force_delta
            failed = (
                energy_delta > energy_atol_hartree
                or force_delta > force_atol_hartree_per_bohr
            )
            comparison["status"] = "fail" if failed else "pass"
    correctness["cross_engine"] = comparison
    if failed:
        correctness["status"] = "fail"
    return failed


def run_cell(
    cell: Cell,
    library: Path,
    xtb_library: Path | None,
    tblite_library: Path | None,
    dxtb_source: Path | None,
    warmups: int,
    repetitions: int,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
    start_policy: str = "auto-warm",
    scc_charge_tolerance: float = 1.0e-10,
    scc_energy_tolerance: float = 1.0e-12,
    scc_max_iterations: int = 500,
) -> dict[str, Any]:
    """Measure one cell and return a complete row."""
    base = base_row(cell)
    try:
        molecule = make_alkane(cell.natoms)
    except Exception as exc:  # noqa: BLE001 - input validation diagnostics
        row = error_row(cell, f"molecule builder failed: {exc}")
        row.update(base)
        return row
    storage = build_batch(
        molecule, cell.batch_size, seed=cell.natoms * 1000 + cell.batch_size
    )
    runner: Any = None
    try:
        if cell.engine in ("gpuxtb-cpu", "gpuxtb-cuda"):
            backend = "cpu" if cell.engine == "gpuxtb-cpu" else "cuda"
            runner = GpuxtbRunner(
                library,
                storage,
                backend,
                cell.cpu_threads,
                cell.device_id,
                scc_charge_tolerance=scc_charge_tolerance,
                scc_energy_tolerance=scc_energy_tolerance,
                scc_max_iterations=scc_max_iterations,
            )
        elif cell.engine == "xtb":
            if xtb_library is None:
                row = unavailable_row(cell, "no --xtb-library supplied")
                row.update(base)
                return row
            runner = ReferenceRunner(
                cell.engine,
                xtb_library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
            )
        elif cell.engine == "tblite":
            if tblite_library is None:
                row = unavailable_row(cell, "no --tblite-library supplied")
                row.update(base)
                return row
            runner = ReferenceRunner(
                cell.engine,
                tblite_library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
            )
        else:
            runner = ReferenceRunner(
                cell.engine,
                library,
                storage,
                cell.cpu_threads,
                cell.device_id,
                dxtb_source,
            )
        fragment = measure_cell(
            runner,
            (warmups, repetitions),
            cell,
            energy_atol_hartree,
            force_atol_hartree_per_bohr,
            start_policy,
        )
        row = base_row(cell)
        row.update(fragment)
        row["availability"] = "available"
        return row
    except (BenchmarkError, XtbError, TbliteError, DxtbError, OSError) as exc:
        row = error_row(cell, str(exc))
        row.update(base)
        return row
    except Exception as exc:  # noqa: BLE001 - cell-level isolation
        row = error_row(cell, f"{type(exc).__name__}: {exc}")
        row.update(base)
        return row
    finally:
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()


def run_trajectory(
    engine: str,
    natoms: int,
    frames: int,
    seed: int,
    library: Path,
    xtb_library: Path | None,
    tblite_library: Path | None,
    dxtb_source: Path | None,
    cpu_threads: int,
    device_id: int,
    warmups: int,
    repetitions: int,
    energy_atol_hartree: float,
    force_atol_hartree_per_bohr: float,
    scc_charge_tolerance: float = 1.0e-10,
    scc_energy_tolerance: float = 1.0e-12,
    scc_max_iterations: int = 500,
) -> dict[str, Any]:
    """Measure per-frame latency over one nearly identical MD-style trajectory."""
    base = {
        "engine": engine,
        "natoms": natoms,
        "batch_size": 1,
        "total_atoms_in_batch": natoms,
        "cpu_threads": cpu_threads,
        "device_id": device_id,
        "job": "trajectory",
        "frames": frames,
    }
    try:
        molecule = make_alkane(natoms)
        frames_list = build_trajectory(molecule, frames, seed)
        storage = build_batch(molecule, 1, seed)
        runner: Any = None
        try:
            if engine in ("gpuxtb-cpu", "gpuxtb-cuda"):
                backend = "cpu" if engine == "gpuxtb-cpu" else "cuda"
                runner = GpuxtbRunner(
                    library,
                    storage,
                    backend,
                    cpu_threads,
                    device_id,
                    scc_charge_tolerance=scc_charge_tolerance,
                    scc_energy_tolerance=scc_energy_tolerance,
                    scc_max_iterations=scc_max_iterations,
                )
            elif engine == "xtb":
                if xtb_library is None:
                    row = dict(base)
                    row.update(
                        {
                            "availability": "unavailable",
                            "reason": "no --xtb-library supplied",
                        }
                    )
                    return row
                runner = ReferenceRunner(
                    engine, xtb_library, storage, cpu_threads, device_id, dxtb_source
                )
            elif engine == "tblite":
                if tblite_library is None:
                    row = dict(base)
                    row.update(
                        {
                            "availability": "unavailable",
                            "reason": "no --tblite-library supplied",
                        }
                    )
                    return row
                runner = ReferenceRunner(
                    engine, tblite_library, storage, cpu_threads, device_id, dxtb_source
                )
            else:
                runner = ReferenceRunner(
                    engine, library, storage, cpu_threads, device_id, dxtb_source
                )
            is_gpuxtb = engine in ("gpuxtb-cpu", "gpuxtb-cuda")
            if is_gpuxtb:
                # MD-style workflow: seed SCC once on frame zero, then continue
                # in WARM mode for every later (nearly identical) frame.
                runner.set_start_mode("fresh")
                runner.set_positions(frames_list[0])
                runner.invoke()
            for _ in range(warmups):
                runner.invoke()
            sample_latencies: list[float] = []
            energies: list[float] = []
            force_digests: list[str] = []
            final_forces: list[float] | None = None
            frame_count = 0
            for _ in range(repetitions):
                for frame in frames_list:
                    runner.set_positions(frame)
                    if is_gpuxtb:
                        runner.set_start_mode("warm")
                    start = time.perf_counter_ns()
                    runner.invoke()
                    sample_latencies.append((time.perf_counter_ns() - start) * 1.0e-6)
                    snapshot = runner.snapshot()
                    snapshot_energies = snapshot.get("energies_hartree")
                    snapshot_forces = snapshot.get("forces_hartree_per_bohr")
                    if (
                        not isinstance(snapshot_energies, list)
                        or len(snapshot_energies) != 1
                    ):
                        raise BenchmarkError("trajectory frame returned invalid energy")
                    if (
                        not isinstance(snapshot_forces, list)
                        or len(snapshot_forces) != 3 * natoms
                    ):
                        raise BenchmarkError("trajectory frame returned invalid forces")
                    if not all(
                        math.isfinite(float(value))
                        for value in (*snapshot_energies, *snapshot_forces)
                    ):
                        raise BenchmarkError(
                            "trajectory frame returned non-finite observables"
                        )
                    energies.extend(float(value) for value in snapshot_energies)
                    final_forces = [float(value) for value in snapshot_forces]
                    force_digests.append(_force_digest(final_forces))
                    frame_count += 1
                    if frame_count % 8 == 0:
                        log(
                            f"trajectory {engine}: {frame_count}/"
                            f"{frames * repetitions} frames"
                        )
            row = dict(base)
            row.update(
                {
                    "availability": "available",
                    "timing": timing_summary(sample_latencies, 1),
                    "per_frame_samples_ms": sample_latencies,
                    "energies_hartree": [energies[-1]],
                    "forces_hartree_per_bohr": final_forces,
                    "energies_hartree_min": min(energies),
                    "energies_hartree_max": max(energies),
                    "force_sample_digests_binary64_le": force_digests,
                    "correctness": {
                        "status": "pass",
                        "finite_energies": True,
                        "finite_forces": True,
                        "force_value_count": 3 * natoms,
                        "energy_atol_hartree": energy_atol_hartree,
                        "force_atol_hartree_per_bohr": (force_atol_hartree_per_bohr),
                        "cross_engine": {"status": "not_requested"},
                    },
                }
            )
            return row
        finally:
            if runner is not None:
                with contextlib.suppress(Exception):
                    runner.close()
    except (BenchmarkError, XtbError, TbliteError, DxtbError, OSError) as exc:
        row = dict(base)
        row.update({"availability": "error", "error": str(exc)})
        return row
    except Exception as exc:  # noqa: BLE001 - cell-level isolation
        row = dict(base)
        row.update({"availability": "error", "error": f"{type(exc).__name__}: {exc}"})
        return row


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Capture the exact revisions, hardware, runtime, and thread environment."""
    repository_state = git_state(REPOSITORY_ROOT)
    dxtb_threads = args.dxtb_cpu_threads or args.cpu_threads
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "commit": repository_state,
        "evidence_eligibility": {
            "status": (
                "diagnostic_dirty"
                if repository_state.get("dirty")
                else "eligible_clean_head"
            ),
            "allow_dirty_evidence": bool(args.allow_dirty_evidence),
        },
        "runner": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpuxtb_library": str(args.library.resolve()),
            "gpuxtb_library_sha256": sha256_file(args.library),
            "xtb_library": str(args.xtb_library.resolve())
            if args.xtb_library
            else None,
            "xtb_library_sha256": sha256_file(args.xtb_library),
            "tblite_library": str(args.tblite_library.resolve())
            if args.tblite_library
            else None,
            "tblite_library_sha256": sha256_file(args.tblite_library),
            "dxtb_source": git_state(args.dxtb_source) if args.dxtb_source else None,
            "reference_json": (
                str(args.reference_json.resolve()) if args.reference_json else None
            ),
            "reference_json_sha256": sha256_file(args.reference_json),
        },
        "hardware": {
            "hostname": platform.node(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "process_rss_bytes": current_rss_bytes(),
            "nvidia_smi": run_text(("nvidia-smi", "-L")),
            "gpu_memory_mib": run_text(
                (
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                )
            ),
        },
        "threads": {
            "cpu_threads": args.cpu_threads,
            "dxtb_cpu_threads": dxtb_threads,
            "reference_threads": args.cpu_threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "MKL_INTERFACE_LAYER",
                "MKL_THREADING_LAYER",
            )
        },
        "protocol": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "start_policy": args.start_policy,
            "cross_engine_energy_atol_hartree": args.energy_atol,
            "cross_engine_force_atol_hartree_per_bohr": args.force_atol,
            "perturb_sigma_bohr": PERTURB_SIGMA_BOHR,
            "trajectory_step_sigma_bohr": TRAJECTORY_STEP_SIGMA_BOHR,
            "scc_max_iterations": args.scc_max_iterations,
            "scc_charge_tolerance": args.scc_charge_tolerance,
            "scc_energy_tolerance": args.scc_energy_tolerance,
        },
    }


def write_json(
    path: Path,
    document: Any,  # noqa: ANN401 - JSON-serializable benchmark document
) -> None:
    """Write one JSON artifact with a trailing newline."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one compact CSV row summary with LF line endings."""
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    columns = [
        "engine",
        "natoms",
        "batch_size",
        "total_atoms_in_batch",
        "cpu_threads",
        "device_id",
        "job",
        "availability",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "systems_per_second_at_median",
        "correctness_status",
        "cross_engine_status",
        "max_abs_energy_delta_hartree",
        "max_abs_force_delta_hartree_per_bohr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            timing = row.get("timing") or {}
            flat["median_ms"] = timing.get("median_ms")
            flat["mean_ms"] = timing.get("mean_ms")
            flat["p95_ms"] = timing.get("p95_ms")
            flat["min_ms"] = timing.get("min_ms")
            flat["systems_per_second_at_median"] = timing.get(
                "systems_per_second_at_median"
            )
            correctness = row.get("correctness") or {}
            cross_engine = correctness.get("cross_engine") or {}
            flat["correctness_status"] = correctness.get("status")
            flat["cross_engine_status"] = cross_engine.get("status")
            flat["max_abs_energy_delta_hartree"] = cross_engine.get(
                "max_abs_energy_delta_hartree"
            )
            flat["max_abs_force_delta_hartree_per_bohr"] = cross_engine.get(
                "max_abs_force_delta_hartree_per_bohr"
            )
            writer.writerow(flat)


def build_parser() -> argparse.ArgumentParser:
    """Configure the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Cross-engine GFN2-xTB atom-count scaling benchmark with distinct "
            "per-slot systems and an optional MD-trajectory mode."
        )
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="clean xTB artifact used for complete energy/force qualification",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--engines", type=parse_csv_values, default=DEFAULT_ENGINES)
    parser.add_argument("--natoms", type=parse_csv_ints, default=DEFAULT_NATOMS)
    parser.add_argument(
        "--natoms-large-batch",
        type=parse_csv_ints,
        help=(
            "molecule sizes used when batch_size > 1; defaults to --natoms. "
            "Use a smaller set for large batches so serial reference engines "
            "remain practical."
        ),
    )
    parser.add_argument(
        "--batch-sizes", type=parse_csv_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument(
        "--start-policy",
        choices=("auto-warm", "cold"),
        default="auto-warm",
        help=(
            "SCC restart policy for the batch matrices. auto-warm (default): "
            "one untimed cold seed, then WARM measured samples. cold: clear "
            "electronic state before every timed inference call; reset/setup "
            "itself is excluded. --cold-samples is an alias."
        ),
    )
    parser.add_argument(
        "--cold-samples",
        dest="start_policy",
        action="store_const",
        const="cold",
        help="alias for --start-policy cold (every sample cold-start).",
    )
    parser.add_argument(
        "--trajectory-natoms",
        type=parse_csv_ints,
        default=(32, 62, 122, 242),
        help=(
            "molecule sizes measured in MD-trajectory mode; each size yields "
            "one per-frame latency row per engine so the figure can sweep "
            "atom count on its x-axis."
        ),
    )
    parser.add_argument("--trajectory-frames", type=int, default=20)
    parser.add_argument(
        "--energy-atol", type=float, default=CROSS_ENGINE_ENERGY_ATOL_HARTREE
    )
    parser.add_argument(
        "--force-atol",
        type=float,
        default=CROSS_ENGINE_FORCE_ATOL_HARTREE_PER_BOHR,
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--dxtb-cpu-threads",
        type=int,
        default=None,
        help="optional dxtb override; defaults to --cpu-threads",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--allow-dirty-evidence",
        action="store_true",
        help="diagnostic only: permit a dirty repository and mark it in metadata",
    )
    parser.add_argument(
        "--scc-charge-tolerance",
        type=float,
        default=1.0e-10,
        help="gpuxtb SCC charge convergence tolerance",
    )
    parser.add_argument(
        "--scc-energy-tolerance",
        type=float,
        default=1.0e-12,
        help="gpuxtb SCC energy convergence tolerance",
    )
    parser.add_argument(
        "--scc-max-iterations",
        type=int,
        default=500,
        help="gpuxtb SCC iteration cap",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject unsupported engine sets and nonpositive controls."""
    for engine in args.engines:
        if engine not in SUPPORTED_ENGINES:
            raise BenchmarkError(f"unsupported engine: {engine}")
    if args.warmups < 0 or args.repetitions <= 0:
        raise BenchmarkError("warmups must be >= 0 and repetitions must be > 0")
    if args.cpu_threads <= 0 or (
        args.dxtb_cpu_threads is not None and args.dxtb_cpu_threads <= 0
    ):
        raise BenchmarkError("thread counts must be positive")
    for name, value in (
        ("energy tolerance", args.energy_atol),
        ("force tolerance", args.force_atol),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise BenchmarkError(f"{name} must be finite and nonnegative")
    if args.device_id < 0:
        raise BenchmarkError("device id must be nonnegative")
    if args.trajectory_frames <= 0:
        raise BenchmarkError("trajectory frames must be positive")
    for natoms in args.trajectory_natoms:
        try:
            make_alkane(natoms)
        except Exception as exc:  # noqa: PERF203 - per-size validation before timing
            raise BenchmarkError(
                f"unsupported trajectory natoms {natoms}: {exc}"
            ) from exc
    if args.output_json == args.output_csv:
        raise BenchmarkError("JSON and CSV output paths must be distinct")
    for natoms in args.natoms:
        try:
            make_alkane(natoms)
        except Exception as exc:  # noqa: PERF203 - per-size validation before timing
            raise BenchmarkError(f"unsupported natoms {natoms}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested matrix and leave complete artifacts on any failure path."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    failed = False
    rows: list[dict[str, Any]] = []
    try:
        validate_arguments(args)
        repository_state = git_state(REPOSITORY_ROOT)
        if not repository_state.get("head"):
            raise BenchmarkError("cannot verify the benchmark repository revision")
        if repository_state.get("dirty") and not args.allow_dirty_evidence:
            raise BenchmarkError(
                "publication evidence requires a clean repository; use "
                "--allow-dirty-evidence only for diagnostics"
            )
        reference = (
            load_reference_artifact(args.reference_json, args.allow_dirty_evidence)
            if args.reference_json is not None
            else None
        )
        if reference is not None:
            reference_commit = (reference.metadata.get("commit") or {}).get("head")
            if reference_commit != repository_state["head"]:
                raise BenchmarkError(
                    "reference artifact and current runner must use the same clean HEAD"
                )
            reference_protocol = reference.metadata.get("protocol") or {}
            for name, expected in (
                ("warmups", args.warmups),
                ("repetitions", args.repetitions),
                ("start_policy", args.start_policy),
            ):
                if reference_protocol.get(name) != expected:
                    raise BenchmarkError(
                        f"reference protocol {name}={reference_protocol.get(name)!r} "
                        f"does not match requested {expected!r}"
                    )
            reference_threads = reference.metadata.get("threads") or {}
            if reference_threads.get("reference_threads") != args.cpu_threads:
                raise BenchmarkError(
                    "reference artifact uses a different thread budget"
                )
        library = args.library.resolve()
        natoms_large_batch = args.natoms_large_batch or args.natoms
        dxtb_threads = args.dxtb_cpu_threads or args.cpu_threads
        cells = [
            Cell(
                engine,
                natoms,
                batch_size,
                dxtb_threads if engine.startswith("dxtb") else args.cpu_threads,
                args.device_id,
            )
            for engine in args.engines
            for batch_size in args.batch_sizes
            for natoms in (args.natoms if batch_size == 1 else natoms_large_batch)
        ]
        total_cells = len(cells)
        log(
            f"matrix: {total_cells} cells from engines={args.engines} "
            f"batch={args.batch_sizes} natoms={args.natoms} "
            f"natoms_large_batch={natoms_large_batch}"
        )
        for cell_index, cell in enumerate(cells, start=1):
            log_start = time.perf_counter()
            row = run_cell(
                cell,
                library,
                args.xtb_library,
                args.tblite_library,
                args.dxtb_source,
                args.warmups,
                args.repetitions,
                args.energy_atol,
                args.force_atol,
                start_policy=args.start_policy,
                scc_charge_tolerance=args.scc_charge_tolerance,
                scc_energy_tolerance=args.scc_energy_tolerance,
                scc_max_iterations=args.scc_max_iterations,
            )
            # Record the SCC start policy on the cell row itself.  A
            # trajectory-invoked matrix cell (for example a batch=1 sample
            # measured with ``--start-policy auto-warm``) is then
            # unambiguously distinguishable from a genuine cold-start row by
            # downstream consumers such as the plotter.
            row["start_policy"] = args.start_policy
            if row.get("availability") == "available":
                if (row.get("correctness") or {}).get("status") != "pass":
                    failed = True
                if row.get("engine") == "xtb" and reference is None:
                    row["correctness"]["cross_engine"] = {
                        "status": "reference",
                        "role": "independent_xTB_baseline",
                    }
                elif reference is not None:
                    failed = (
                        apply_cross_engine_reference(
                            row,
                            reference,
                            args.energy_atol,
                            args.force_atol,
                        )
                        or failed
                    )
            elapsed_s = time.perf_counter() - log_start
            rows.append(row)
            if row["availability"] == "available":
                log(
                    f"[{cell_index}/{total_cells}] {row['engine']} "
                    f"natoms={row['natoms']} batch={row['batch_size']} "
                    f"median={row['timing']['median_ms']:.6f} ms "
                    f"({elapsed_s:.1f} s)"
                )
            else:
                log(
                    f"[{cell_index}/{total_cells}] {row['engine']} "
                    f"natoms={row['natoms']} batch={row['batch_size']} "
                    f"UNAVAILABLE: {row.get('reason', row.get('error', 'unknown'))}"
                )
                failed = True
        if args.trajectory:
            for engine in args.engines:
                for trajectory_natoms in args.trajectory_natoms:
                    log(
                        f"trajectory engine={engine} natoms={trajectory_natoms} "
                        f"frames={args.trajectory_frames}"
                    )
                    row = run_trajectory(
                        engine,
                        trajectory_natoms,
                        args.trajectory_frames,
                        seed=trajectory_natoms * 7 + 42,
                        library=library,
                        xtb_library=args.xtb_library,
                        tblite_library=args.tblite_library,
                        dxtb_source=args.dxtb_source,
                        cpu_threads=args.cpu_threads,
                        device_id=args.device_id,
                        warmups=args.warmups,
                        repetitions=args.repetitions,
                        energy_atol_hartree=args.energy_atol,
                        force_atol_hartree_per_bohr=args.force_atol,
                        scc_charge_tolerance=args.scc_charge_tolerance,
                        scc_energy_tolerance=args.scc_energy_tolerance,
                        scc_max_iterations=args.scc_max_iterations,
                    )
                    rows.append(row)
                    if row["availability"] == "available":
                        log(
                            f"trajectory {row['engine']} natoms={row['natoms']} "
                            f"median={row['timing']['median_ms']:.6f} ms/frame"
                        )
                    else:
                        log(
                            f"trajectory {row['engine']} UNAVAILABLE: "
                            f"{row.get('reason', row.get('error', 'unknown'))}"
                        )
                        failed = True
        document = {
            "schema_version": SCHEMA_VERSION,
            "metadata": environment_metadata(args),
            "rows": rows,
        }
        write_json(args.output_json, document)
        write_csv(args.output_csv, rows)
    except (BenchmarkError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 2
    print(f"wrote {args.output_json} and {args.output_csv}")  # noqa: T201
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
