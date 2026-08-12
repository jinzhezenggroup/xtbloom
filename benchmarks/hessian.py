#!/usr/bin/env python3
"""Correctness-qualified 62-atom GFN2-xTB Hessian benchmark.

The xTBloom batch coordinate is the number of finite-difference displacement
systems submitted per native force call.  It is deliberately not a batch of
independent Hessians: the current public API computes one Hessian for one
``Calculator``.  A 62-atom Hessian contains 372 displaced force systems, so a
displacement batch of 128 is executed as native chunks of 128, 128, and 116.

Every timed interval ends with a complete host-visible dense Hessian.  Engine
and calculator construction is outside timing; reset operations required by a
public call remain inside timing unless the protocol explicitly identifies a
fresh result object prepared before the interval.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import hashlib
import importlib
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from . import natoms_cross_engine as nce
    from .natoms_scaling import (
        Molecule,
        _cmake_build_metadata,
        cpu_model,
        make_alkane,
        process_affinity,
    )
    from .xtb_adapter import XtbAdapter, XtbError, _delete
except ImportError:  # Direct ``python benchmarks/hessian.py`` execution.
    import natoms_cross_engine as nce
    from natoms_scaling import (  # type: ignore[no-redef]
        Molecule,
        _cmake_build_metadata,
        cpu_model,
        make_alkane,
        process_affinity,
    )
    from xtb_adapter import XtbAdapter, XtbError, _delete

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import TracebackType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
NATOMS = 62
COORDINATE_COUNT = 3 * NATOMS
DISPLACEMENT_COUNT = 2 * COORDINATE_COUNT
DEFAULT_ENGINES = ("xtbloom-cpu", "xtbloom-cuda", "xtb")
SUPPORTED_ENGINES = (
    "xtbloom-cpu",
    "xtbloom-cuda",
    "xtb",
    "dxtb-cpu-ad",
    "dxtb-cuda-ad",
    "dxtb-cpu-numerical",
    "dxtb-cuda-numerical",
)


class BenchmarkError(RuntimeError):
    """An invalid benchmark request or publication state."""


def parse_csv_values(value: str) -> tuple[str, ...]:
    """Parse a nonempty comma-separated selection."""
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("selection must not be empty")
    return values


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a nonempty comma-separated positive-integer selection."""
    try:
        values = tuple(int(part) for part in parse_csv_values(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return values


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for retained raw samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BenchmarkError("cannot summarize an empty sample set")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Summarize all retained end-to-end timing samples."""
    median = statistics.median(samples_ms)
    return {
        "samples_ms": [float(value) for value in samples_ms],
        "sample_count": len(samples_ms),
        "median_ms": median,
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "hessians_per_hour_at_median": 3_600_000.0 / median,
    }


def encode_hessian(matrix: np.ndarray) -> dict[str, Any]:
    """Encode one exact dense Hessian as compressed little-endian binary64."""
    canonical = np.asarray(matrix, dtype="<f8", order="C")
    raw = canonical.tobytes(order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": "float64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "zlib_base64": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def decode_hessian(encoded: dict[str, Any]) -> np.ndarray:
    """Decode and authenticate one Hessian stored by :func:`encode_hessian`."""
    if encoded.get("dtype") != "float64-le":
        raise BenchmarkError("reference Hessian has unsupported dtype")
    shape = tuple(int(value) for value in encoded.get("shape", ()))
    if shape != (COORDINATE_COUNT, COORDINATE_COUNT):
        raise BenchmarkError(f"reference Hessian has unexpected shape {shape}")
    try:
        raw = zlib.decompress(base64.b64decode(encoded["zlib_base64"], validate=True))
    except (KeyError, ValueError, zlib.error) as exc:
        raise BenchmarkError("reference Hessian payload is invalid") from exc
    if hashlib.sha256(raw).hexdigest() != encoded.get("sha256"):
        raise BenchmarkError("reference Hessian SHA-256 mismatch")
    expected = COORDINATE_COUNT * COORDINATE_COUNT * 8
    if len(raw) != expected:
        raise BenchmarkError("reference Hessian byte count is invalid")
    return np.frombuffer(raw, dtype="<f8").reshape(shape).astype(np.float64, copy=True)


def hessian_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    """Return finite, symmetry, translation, norm, and range diagnostics."""
    hessian = np.asarray(matrix, dtype=np.float64)
    if hessian.shape != (COORDINATE_COUNT, COORDINATE_COUNT):
        raise BenchmarkError(f"engine returned Hessian shape {hessian.shape}")
    finite = bool(np.isfinite(hessian).all())
    if not finite:
        return {
            "finite": False,
            "max_abs_element_hartree_per_bohr2": None,
            "frobenius_norm_hartree_per_bohr2": None,
            "max_abs_antisymmetry_hartree_per_bohr2": None,
            "max_abs_acoustic_row_residual_hartree_per_bohr2": None,
            "max_abs_acoustic_column_residual_hartree_per_bohr2": None,
        }
    tensor = hessian.reshape(NATOMS, 3, NATOMS, 3)
    return {
        "finite": True,
        "max_abs_element_hartree_per_bohr2": float(np.max(np.abs(hessian))),
        "frobenius_norm_hartree_per_bohr2": float(np.linalg.norm(hessian)),
        "max_abs_antisymmetry_hartree_per_bohr2": float(
            np.max(np.abs(hessian - hessian.T))
        ),
        "max_abs_acoustic_row_residual_hartree_per_bohr2": float(
            np.max(np.abs(tensor.sum(axis=0)))
        ),
        "max_abs_acoustic_column_residual_hartree_per_bohr2": float(
            np.max(np.abs(tensor.sum(axis=2)))
        ),
    }


def compare_hessians(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare symmetrized Hessians because xTB symmetrizes in its C API."""
    actual_symmetric = 0.5 * (actual + actual.T)
    reference_symmetric = 0.5 * (reference + reference.T)
    delta = actual_symmetric - reference_symmetric
    return {
        "max_abs_delta_hartree_per_bohr2": float(np.max(np.abs(delta))),
        "rms_delta_hartree_per_bohr2": float(np.sqrt(np.mean(delta * delta))),
    }


def displacement_atom_limit(displacement_batch_size: int) -> int:
    """Translate user-facing displacement systems to xTBloom's atom limit."""
    if displacement_batch_size <= 0:
        raise BenchmarkError("displacement batch size must be positive")
    return NATOMS * displacement_batch_size


def displacement_chunks(displacement_batch_size: int) -> list[int]:
    """Return the exact native system counts for one xTBloom Hessian."""
    chunks = []
    remaining = DISPLACEMENT_COUNT
    while remaining:
        chunk = min(displacement_batch_size, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


@dataclass
class EngineResult:
    """One completed Hessian and engine-specific timing metadata."""

    hessian: np.ndarray
    metadata: dict[str, Any]


class HessianEngine:
    """Small protocol implemented by the real and fake benchmark engines."""

    def prepare_sample(self) -> None:
        """Prepare fresh state outside the measured interval."""

    def invoke(self) -> EngineResult:
        """Return one complete host-visible Hessian."""
        raise NotImplementedError

    def close(self) -> None:
        """Release engine resources."""

    def __enter__(self) -> HessianEngine:  # noqa: PYI034 - Python 3.10 support.
        """Return the live engine for one scoped benchmark coordinate."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        """Release resources and propagate any benchmark exception."""
        self.close()
        return False


def run_isolated_coordinate(
    command: Sequence[str],
    *,
    output_json: Path,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one engine coordinate in a subprocess and retain hard failures.

    Native numerical backends may abort the interpreter during teardown or on
    device OOM, which cannot be converted into an unavailable row from inside
    that process.  The parent therefore owns the final artifact and imports a
    completed child row only after the child exits successfully.  A signal or
    nonzero status remains explicit evidence instead of losing the rest of the
    requested matrix.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        diagnostic = exc.stderr or exc.stdout or ""
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode("utf-8", errors="replace")
        reason = f"coordinate exceeded timeout of {timeout_seconds:g} seconds"
        if diagnostic.strip():
            reason += f": {diagnostic.strip()[-4000:]}"
        return {
            "availability": "unavailable",
            "unavailable_reason": reason,
            "completed_samples_ms": [],
            "isolated_command": list(command),
        }
    child_row: dict[str, Any] | None = None
    if output_json.is_file():
        document = json.loads(output_json.read_text(encoding="utf-8"))
        rows = document.get("rows") or []
        if len(rows) != 1:
            raise BenchmarkError("isolated coordinate produced an invalid row count")
        child_row = rows[0]
    if completed.returncode == 0 and child_row is not None:
        child_row["isolated_command"] = list(command)
        return child_row

    if completed.returncode < 0:
        signal_number = -completed.returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal_{signal_number}"
        reason = f"child terminated by {signal_name} ({signal_number})"
    else:
        reason = f"child exited with status {completed.returncode}"
    diagnostic = completed.stderr.strip() or completed.stdout.strip()
    if diagnostic:
        reason += f": {diagnostic[-4000:]}"
    row = dict(child_row or {})
    timing = row.pop("timing", None) or {}
    row.pop("correctness", None)
    row.pop("final_hessian_binary64_le_zlib_base64", None)
    row["availability"] = "unavailable"
    row["unavailable_reason"] = reason
    row["completed_samples_ms"] = timing.get("samples_ms", [])
    row["isolated_command"] = list(command)
    return row


def coordinate_command(
    args: argparse.Namespace,
    *,
    engine: str,
    displacement_batch_size: int | None,
    output_json: Path,
    output_csv: Path,
) -> list[str]:
    """Reconstruct one exact child command from validated public options."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engines",
        engine,
        "--device-id",
        str(args.device_id),
        "--cpu-threads",
        str(args.cpu_threads),
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--step",
        repr(args.step),
        "--scc-max-iterations",
        str(args.scc_max_iterations),
        "--scc-charge-tolerance",
        repr(args.scc_charge_tolerance),
        "--scc-energy-tolerance",
        repr(args.scc_energy_tolerance),
        "--hessian-atol",
        repr(args.hessian_atol),
        "--symmetry-atol",
        repr(args.symmetry_atol),
        "--acoustic-atol",
        repr(args.acoustic_atol),
        "--repeatability-atol",
        repr(args.repeatability_atol),
        "--coordinate-timeout-seconds",
        repr(args.coordinate_timeout_seconds),
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--coordinate-child",
    ]
    if displacement_batch_size is not None:
        command.extend(["--displacement-batch-sizes", str(displacement_batch_size)])
    for option, value in (
        ("--library", args.library),
        ("--xtb-library", args.xtb_library),
        ("--xtb-source", args.xtb_source),
        ("--dxtb-source", args.dxtb_source),
        ("--reference-json", args.reference_json),
    ):
        if value is not None:
            command.extend([option, str(value.resolve())])
    if args.make_reference:
        command.append("--make-reference")
    if args.allow_dirty_evidence:
        command.append("--allow-dirty-evidence")
    return command


class XTBloomHessianEngine(HessianEngine):
    """Use the public Python Hessian convenience method and native C ABI."""

    def __init__(
        self,
        *,
        backend: str,
        library_path: Path,
        numbers: np.ndarray,
        positions: np.ndarray,
        displacement_batch_size: int,
        step: float,
        cpu_threads: int,
        device_id: int,
        max_iterations: int,
        charge_tolerance: float,
        energy_tolerance: float,
    ) -> None:
        os.environ["XTBLOOM_LIBRARY"] = str(library_path.resolve())
        xtbloom = importlib.import_module("xtbloom")
        self.calculator = xtbloom.Calculator(
            "GFN2-xTB",
            numbers,
            positions,
            backend=backend,
            device_id=device_id if backend == "cuda" else None,
            cpu_threads=cpu_threads,
            max_scc_iterations=max_iterations,
            charge_tolerance=charge_tolerance,
            energy_tolerance=energy_tolerance,
            electronic_temperature=300.0,
            warm_start=False,
        )
        self.backend = backend
        self.displacement_batch_size = displacement_batch_size
        self.atom_limit = displacement_atom_limit(displacement_batch_size)
        self.step = step

    def invoke(self) -> EngineResult:
        """Compute one raw xTBloom Hessian through the public Python API."""
        hessian = self.calculator.hessian(
            step=self.step,
            symmetrize=False,
            auto_batch_size=self.atom_limit,
        )
        return EngineResult(
            np.asarray(hessian, dtype=np.float64),
            {
                "raw_symmetrize": False,
                "public_auto_batch_size_atom_limit": self.atom_limit,
                "native_displacement_chunks": displacement_chunks(
                    self.displacement_batch_size
                ),
                "fresh_temporary_context_per_hessian": True,
            },
        )

    def close(self) -> None:
        """Release the persistent outer calculator context."""
        self.calculator.close()


class XtbHessianEngine(HessianEngine):
    """Call xTB 6.7.1's native, OpenMP-parallel ``xtb_hessian`` C API."""

    def __init__(
        self,
        *,
        library_path: Path,
        molecule: Molecule,
        step: float,
        cpu_threads: int,
        max_iterations: int,
    ) -> None:
        if step != 0.005:
            raise BenchmarkError(
                "xTB adapter currently requires its C API default 0.005 bohr step"
            )
        storage = nce.build_batch(molecule, 1, seed=0)
        self.adapter = XtbAdapter(
            library_path,
            storage,
            "force",
            None,
            accuracy=1.0,
            max_iterations=max_iterations,
            electronic_temperature_kelvin=300.0,
            threads=cpu_threads,
        )
        self.library = self.adapter.library
        # The released xTB header names the two optional pointers in the
        # opposite order/types from the Fortran bind(C) implementation.  NULL
        # for both selects all atoms and the exact desired default step without
        # depending on the mismatched declarations.
        self.library.xtb_hessian.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.xtb_hessian.restype = None
        self.hessian = (ctypes.c_double * (COORDINATE_COUNT**2))()

    def prepare_sample(self) -> None:
        """Replace xTB's result checkpoint so every Hessian starts fresh."""
        state = self.adapter.states[0]
        old_result = state.result
        state.result = ctypes.c_void_p(self.library.xtb_newResults())
        _delete(self.library, "xtb_delResults", old_result)
        if not state.result:
            raise XtbError("xtb_newResults returned NULL before Hessian")
        self.adapter._check(state.environment, "xtb_newResults")

    def invoke(self) -> EngineResult:
        """Compute one native xTB Hessian and publish its host matrix."""
        state = self.adapter.states[0]
        ctypes.memset(ctypes.addressof(self.hessian), 0, ctypes.sizeof(self.hessian))
        self.library.xtb_hessian(
            state.environment,
            state.molecule,
            state.calculator,
            state.result,
            self.hessian,
            None,
            None,
            None,
            None,
        )
        self.adapter._check(state.environment, "xtb_hessian")
        matrix = np.ctypeslib.as_array(self.hessian).reshape(
            (COORDINATE_COUNT, COORDINATE_COUNT), order="F"
        )
        return EngineResult(
            np.array(matrix, copy=True),
            {
                "raw_symmetrize": True,
                "fresh_result_checkpoint_prepared_outside_timing": True,
                "accuracy_factor": 1.0,
                "optional_step_and_atom_list": "NULL/default 0.005 bohr/all atoms",
                "native_openmp_displacement_parallelism": True,
                "unavoidable_auxiliary_dipole_gradient": True,
            },
        )

    def close(self) -> None:
        """Release all persistent xTB C API handles."""
        self.adapter.close()


class DxtbHessianEngine(HessianEngine):
    """Use dxtb's public AD or numerical single-system Hessian method."""

    def __init__(
        self,
        *,
        backend: str,
        mode: str,
        source_root: Path,
        numbers: np.ndarray,
        positions: np.ndarray,
        step: float,
        cpu_threads: int,
        device_id: int,
        max_iterations: int,
    ) -> None:
        source_directory = source_root.resolve() / "src"
        if str(source_directory) not in sys.path:
            sys.path.insert(0, str(source_directory))
        self.torch = importlib.import_module("torch")
        self.dxtb = importlib.import_module("dxtb")
        self.backend = backend
        self.mode = mode
        self.step = step
        self.previous_threads = int(self.torch.get_num_threads())
        self.torch.set_num_threads(cpu_threads)
        self.device = self.torch.device(
            f"cuda:{device_id}" if backend == "cuda" else "cpu"
        )
        if backend == "cuda" and not self.torch.cuda.is_available():
            raise BenchmarkError("dxtb CUDA requested but torch.cuda is unavailable")
        self.numbers = self.torch.tensor(
            numbers, dtype=self.torch.long, device=self.device
        )
        self.positions = self.torch.tensor(
            positions, dtype=self.torch.float64, device=self.device
        )
        self.positions.requires_grad_(True)
        self.charge = self.torch.tensor(
            0.0, dtype=self.torch.float64, device=self.device
        )
        options = {
            "verbosity": 0,
            "batch_mode": 0,
            "maxiter": max_iterations,
            "x_atol": 1.0e-4,
            "x_atol_max": 1.0e-5,
            "f_atol": 1.0e-4,
            "force_convergence": True,
        }
        self.calculator = self.dxtb.Calculator(
            self.numbers,
            self.dxtb.GFN2_XTB,
            opts=options,
            device=self.device,
            dtype=self.torch.float64,
        )

    @property
    def live_calculator(self) -> Any:  # noqa: ANN401 - third-party dynamic API.
        """Return the live dxtb calculator and reject use after cleanup."""
        if self.calculator is None:
            raise BenchmarkError("dxtb Hessian engine is closed")
        return self.calculator

    def invoke(self) -> EngineResult:
        """Compute one dxtb AD or numerical Hessian and publish it to host."""
        # dxtb caches derivative results by tensor identity, so reset is part
        # of every timed public-call workflow rather than a setup exclusion.
        calculator = self.live_calculator
        calculator.reset()
        if self.mode == "ad":
            result = calculator.hessian(
                self.positions,
                self.charge,
                use_functorch=False,
                derived_quantity="forces",
                matrix=True,
            )
        else:
            result = calculator.hessian_numerical(
                self.positions,
                self.charge,
                step_size=self.step,
                matrix=True,
            )
        if self.backend == "cuda":
            self.torch.cuda.synchronize(self.device)
        host = result.detach().to(device="cpu").contiguous().numpy().copy()
        return EngineResult(
            np.asarray(host, dtype=np.float64),
            {
                "raw_symmetrize": False,
                "mode": self.mode,
                "calculator_reset_inside_timing": True,
                "batch_mode": 0,
                "batch_hessian_supported": False,
                "host_publication_inside_timing": True,
                "torch_version": getattr(self.torch, "__version__", None),
                "dxtb_version": getattr(self.dxtb, "__version__", None),
            },
        )

    def close(self) -> None:
        """Release dxtb tensors and restore the process PyTorch thread count."""
        self.calculator = None
        if self.backend == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.torch.set_num_threads(self.previous_threads)


def create_engine(
    engine: str,
    *,
    args: argparse.Namespace,
    molecule: Molecule,
    displacement_batch_size: int | None,
) -> HessianEngine:
    """Construct one selected public Hessian engine outside timing."""
    numbers = np.asarray(molecule.atomic_numbers, dtype=np.int64)
    positions = np.asarray(molecule.positions_bohr, dtype=np.float64).reshape(-1, 3)
    if engine.startswith("xtbloom-"):
        if args.library is None or displacement_batch_size is None:
            raise BenchmarkError("xTBloom engine requires a library and batch size")
        return XTBloomHessianEngine(
            backend=engine.removeprefix("xtbloom-"),
            library_path=args.library,
            numbers=numbers,
            positions=positions,
            displacement_batch_size=displacement_batch_size,
            step=args.step,
            cpu_threads=args.cpu_threads,
            device_id=args.device_id,
            max_iterations=args.scc_max_iterations,
            charge_tolerance=args.scc_charge_tolerance,
            energy_tolerance=args.scc_energy_tolerance,
        )
    if engine == "xtb":
        if args.xtb_library is None:
            raise BenchmarkError("xTB engine requires --xtb-library")
        return XtbHessianEngine(
            library_path=args.xtb_library,
            molecule=molecule,
            step=args.step,
            cpu_threads=args.cpu_threads,
            max_iterations=args.scc_max_iterations,
        )
    if engine.startswith("dxtb-"):
        if args.dxtb_source is None:
            raise BenchmarkError("dxtb engine requires --dxtb-source")
        _, backend, mode = engine.split("-", maxsplit=2)
        return DxtbHessianEngine(
            backend=backend,
            mode=mode,
            source_root=args.dxtb_source,
            numbers=numbers,
            positions=positions,
            step=args.step,
            cpu_threads=args.cpu_threads,
            device_id=args.device_id,
            max_iterations=args.scc_max_iterations,
        )
    raise BenchmarkError(f"unsupported engine {engine!r}")


def load_reference(path: Path | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load the one reference Hessian and its authenticated artifact identity."""
    if path is None:
        return None, {"designation": "none"}
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    rows = [
        row
        for row in document.get("rows", [])
        if row.get("availability") == "available"
    ]
    if len(rows) != 1:
        raise BenchmarkError("reference artifact must contain one available row")
    encoded = rows[0].get("final_hessian_binary64_le_zlib_base64")
    if not isinstance(encoded, dict):
        raise BenchmarkError("reference artifact has no final Hessian")
    return decode_hessian(encoded), {
        "designation": "independent_baseline",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "engine": rows[0].get("engine"),
    }


def evaluate_correctness(
    matrices: Sequence[np.ndarray],
    *,
    reference: np.ndarray | None,
    hessian_atol: float,
    symmetry_atol: float,
    acoustic_atol: float,
    repeatability_atol: float,
    is_reference: bool,
) -> dict[str, Any]:
    """Apply finite, diagnostic, repeatability, and cross-engine gates."""
    diagnostics = hessian_diagnostics(matrices[-1])
    repeatability = (
        max(float(np.max(np.abs(matrix - matrices[0]))) for matrix in matrices[1:])
        if len(matrices) > 1
        else 0.0
    )
    reasons = []
    if not diagnostics["finite"]:
        reasons.append("non_finite_hessian")
    if (
        diagnostics["max_abs_antisymmetry_hartree_per_bohr2"] is not None
        and diagnostics["max_abs_antisymmetry_hartree_per_bohr2"] > symmetry_atol
    ):
        reasons.append("antisymmetry_exceeds_tolerance")
    acoustic = max(
        value
        for value in (
            diagnostics["max_abs_acoustic_row_residual_hartree_per_bohr2"],
            diagnostics["max_abs_acoustic_column_residual_hartree_per_bohr2"],
        )
        if value is not None
    )
    if acoustic > acoustic_atol:
        reasons.append("acoustic_residual_exceeds_tolerance")
    if repeatability > repeatability_atol:
        reasons.append("repeatability_exceeds_tolerance")
    if is_reference:
        comparison: dict[str, Any] = {"status": "reference"}
    elif reference is None:
        comparison = {"status": "not_requested"}
    else:
        comparison = dict(compare_hessians(matrices[-1], reference))
        comparison["status"] = (
            "pass"
            if comparison["max_abs_delta_hartree_per_bohr2"] <= hessian_atol
            else "fail"
        )
        if comparison["status"] == "fail":
            reasons.append("cross_engine_delta_exceeds_tolerance")
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "diagnostics": diagnostics,
        "max_abs_repeatability_delta_hartree_per_bohr2": repeatability,
        "cross_engine": comparison,
    }


def run_row(
    engine_name: str,
    *,
    args: argparse.Namespace,
    molecule: Molecule,
    displacement_batch_size: int | None,
    reference: np.ndarray | None,
    factory: Callable[..., HessianEngine] = create_engine,
) -> dict[str, Any]:
    """Run one engine coordinate while preserving actionable failures."""
    row: dict[str, Any] = {
        "engine": engine_name,
        "natoms": NATOMS,
        "molecule": molecule.name,
        "coordinate_count": COORDINATE_COUNT,
        "displacement_count": DISPLACEMENT_COUNT,
        "displacement_batch_size": displacement_batch_size,
        "public_batch_semantics": (
            "finite_difference_displacement_systems_per_native_force_call"
            if engine_name.startswith("xtbloom-")
            else "single_system_hessian; xTBloom displacement batch not applicable"
        ),
        "availability": "unavailable",
    }
    matrices: list[np.ndarray] = []
    samples_ms: list[float] = []
    engine_metadata: dict[str, Any] = {}
    try:
        with factory(
            engine_name,
            args=args,
            molecule=molecule,
            displacement_batch_size=displacement_batch_size,
        ) as engine:
            for _ in range(args.warmups):
                engine.prepare_sample()
                engine.invoke()
            for _ in range(args.repetitions):
                engine.prepare_sample()
                start = time.perf_counter_ns()
                result = engine.invoke()
                elapsed_ms = (time.perf_counter_ns() - start) * 1.0e-6
                matrices.append(np.array(result.hessian, copy=True))
                samples_ms.append(elapsed_ms)
                engine_metadata = result.metadata
    except Exception as exc:  # noqa: BLE001 - unavailable rows retain engine errors.
        row["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        row["completed_samples_ms"] = samples_ms
        return row

    is_reference = bool(args.make_reference)
    correctness = evaluate_correctness(
        matrices,
        reference=reference,
        hessian_atol=args.hessian_atol,
        symmetry_atol=args.symmetry_atol,
        acoustic_atol=args.acoustic_atol,
        repeatability_atol=args.repeatability_atol,
        is_reference=is_reference,
    )
    row.update(
        {
            "availability": "available",
            "timing": timing_summary(samples_ms),
            "correctness": correctness,
            "engine_metadata": engine_metadata,
            "final_hessian_binary64_le_zlib_base64": encode_hessian(matrices[-1]),
        }
    )
    return row


def git_state(path: Path | None) -> dict[str, Any] | None:
    """Capture one source checkout using the existing evidence helper."""
    if path is None:
        return None
    return nce.git_state(path.resolve())


def build_metadata(library: Path | None) -> dict[str, Any] | None:
    """Capture an xTBloom CMake build when the adjacent cache is available."""
    if library is None:
        return None
    cache = library.resolve().parent / "CMakeCache.txt"
    if not cache.is_file():
        return None
    return _cmake_build_metadata(library.resolve(), cache)


def runner_metadata(
    args: argparse.Namespace, reference_identity: dict[str, Any]
) -> dict[str, Any]:
    """Record source, binary, runtime, hardware, and exact protocol identity."""
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": nce.sha256_file(Path(__file__).resolve()),
            },
            "python": sys.version,
            "argv": sys.argv,
            "xtbloom_library": str(args.library.resolve()) if args.library else None,
            "xtbloom_native_identity": (
                nce.native_library_identity(str(args.library.resolve()))
                if args.library
                else None
            ),
            "xtbloom_build": build_metadata(args.library),
            "xtb_library": (
                str(args.xtb_library.resolve()) if args.xtb_library else None
            ),
            "xtb_native_identity": (
                nce.native_library_identity(str(args.xtb_library.resolve()))
                if args.xtb_library
                else None
            ),
            "xtb_source": git_state(args.xtb_source),
            "dxtb_source": git_state(args.dxtb_source),
            "python_distributions": {
                name: nce.installed_distribution_identity(name)
                for name in ("dxtb", "torch", "tad-libcint")
            },
        },
        "commit": git_state(REPOSITORY_ROOT),
        "evidence_eligibility": {
            "allow_dirty_evidence": args.allow_dirty_evidence,
            "status": (
                "diagnostic_dirty_allowed"
                if args.allow_dirty_evidence
                else "eligible_clean_head"
            ),
        },
        "hardware": {
            "hostname": platform.node(),
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "process_cpu_affinity": process_affinity(),
            "nvidia_smi": nce.run_text(("nvidia-smi", "-L")),
            "nvidia_smi_runtime": nce.run_text(
                (
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,driver_version,memory.total,pstate,"
                    "power.limit,clocks.current.sm,clocks.current.memory",
                    "--format=csv,noheader",
                )
            ),
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        },
        "threads": {
            "cpu_threads": args.cpu_threads,
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
            "workload": "neutral singlet C20H42 generated by make_alkane(62)",
            "position_units": "bohr",
            "hessian_units": "Hartree/bohr^2",
            "step_bohr": args.step,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "timing_boundary": (
                "complete public Hessian call through host-visible dense matrix"
            ),
            "scc_max_iterations": args.scc_max_iterations,
            "scc_charge_tolerance": args.scc_charge_tolerance,
            "scc_energy_tolerance": args.scc_energy_tolerance,
            "hessian_atol_hartree_per_bohr2": args.hessian_atol,
            "symmetry_atol_hartree_per_bohr2": args.symmetry_atol,
            "acoustic_atol_hartree_per_bohr2": args.acoustic_atol,
            "repeatability_atol_hartree_per_bohr2": args.repeatability_atol,
            "coordinate_timeout_seconds": args.coordinate_timeout_seconds,
            "reference": reference_identity,
        },
    }


def validate_clean_sources(args: argparse.Namespace) -> None:
    """Reject dirty or mismatched publication sources unless diagnostic mode."""
    if args.allow_dirty_evidence:
        return
    sources = [("xTBloom", REPOSITORY_ROOT)]
    if "xtb" in args.engines and args.xtb_source is not None:
        sources.append(("xTB", args.xtb_source))
    if any(engine.startswith("dxtb-") for engine in args.engines) and args.dxtb_source:
        sources.append(("dxtb", args.dxtb_source))
    for name, source in sources:
        state = nce.git_state(source.resolve())
        if state["dirty"]:
            raise BenchmarkError(f"{name} source checkout is dirty: {source}")
    if args.library is not None:
        build = build_metadata(args.library)
        if build is None:
            raise BenchmarkError(
                "xTBloom publication library lacks adjacent CMakeCache.txt"
            )
        source_git = build["source"]["git"]
        repository_git = nce.git_state(REPOSITORY_ROOT)
        if source_git["revision"] != repository_git["head"] or source_git["dirty"]:
            raise BenchmarkError(
                "xTBloom library is not built from the clean runner HEAD"
            )


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Write one authoritative, finite-valued JSON artifact."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a compact human-readable row summary."""
    columns = [
        "engine",
        "natoms",
        "displacement_batch_size",
        "availability",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "hessians_per_hour_at_median",
        "correctness_status",
        "max_abs_delta_hartree_per_bohr2",
        "rms_delta_hartree_per_bohr2",
        "max_abs_antisymmetry_hartree_per_bohr2",
        "max_abs_acoustic_residual_hartree_per_bohr2",
        "unavailable_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            timing = row.get("timing") or {}
            correctness = row.get("correctness") or {}
            diagnostics = correctness.get("diagnostics") or {}
            comparison = correctness.get("cross_engine") or {}
            acoustic_values = [
                diagnostics.get("max_abs_acoustic_row_residual_hartree_per_bohr2"),
                diagnostics.get("max_abs_acoustic_column_residual_hartree_per_bohr2"),
            ]
            acoustic = (
                max(value for value in acoustic_values if value is not None)
                if any(value is not None for value in acoustic_values)
                else None
            )
            writer.writerow(
                {
                    "engine": row.get("engine"),
                    "natoms": row.get("natoms"),
                    "displacement_batch_size": row.get("displacement_batch_size"),
                    "availability": row.get("availability"),
                    "median_ms": timing.get("median_ms"),
                    "mean_ms": timing.get("mean_ms"),
                    "p95_ms": timing.get("p95_ms"),
                    "min_ms": timing.get("min_ms"),
                    "max_ms": timing.get("max_ms"),
                    "hessians_per_hour_at_median": timing.get(
                        "hessians_per_hour_at_median"
                    ),
                    "correctness_status": correctness.get("status"),
                    "max_abs_delta_hartree_per_bohr2": comparison.get(
                        "max_abs_delta_hartree_per_bohr2"
                    ),
                    "rms_delta_hartree_per_bohr2": comparison.get(
                        "rms_delta_hartree_per_bohr2"
                    ),
                    "max_abs_antisymmetry_hartree_per_bohr2": diagnostics.get(
                        "max_abs_antisymmetry_hartree_per_bohr2"
                    ),
                    "max_abs_acoustic_residual_hartree_per_bohr2": acoustic,
                    "unavailable_reason": row.get("unavailable_reason"),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line protocol for reproducible Hessian evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", type=parse_csv_values, default=DEFAULT_ENGINES)
    parser.add_argument(
        "--displacement-batch-sizes", type=parse_csv_ints, default=(1, 128)
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--xtb-source", type=Path, default=Path.home() / "codes/xtb")
    parser.add_argument("--dxtb-source", type=Path, default=Path.home() / "codes/dxtb")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--scc-max-iterations", type=int, default=500)
    parser.add_argument("--scc-charge-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--scc-energy-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--hessian-atol", type=float, default=2.0e-3)
    parser.add_argument("--symmetry-atol", type=float, default=2.0e-3)
    parser.add_argument("--acoustic-atol", type=float, default=2.0e-3)
    parser.add_argument("--repeatability-atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--coordinate-timeout-seconds",
        type=float,
        default=0.0,
        help="parent timeout per engine coordinate; zero disables the limit",
    )
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--make-reference", action="store_true")
    parser.add_argument("--allow-dirty-evidence", action="store_true")
    parser.add_argument("--fail-on-correctness", action="store_true")
    parser.add_argument(
        "--coordinate-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or incomplete benchmark requests before execution."""
    unknown = sorted(set(args.engines) - set(SUPPORTED_ENGINES))
    if unknown:
        raise BenchmarkError(f"unsupported engines: {', '.join(unknown)}")
    if args.make_reference and (args.engines != ("xtb",) or args.reference_json):
        raise BenchmarkError("--make-reference requires only --engines xtb")
    if args.output_json.resolve() == args.output_csv.resolve():
        raise BenchmarkError("JSON and CSV outputs must be distinct")
    if args.output_json.parent.resolve() != args.output_csv.parent.resolve():
        raise BenchmarkError("JSON and CSV outputs must share a directory")
    for output in (args.output_json, args.output_csv):
        if output.exists():
            raise BenchmarkError(f"refusing to overwrite existing artifact: {output}")
    for name in (
        "cpu_threads",
        "warmups",
        "repetitions",
        "scc_max_iterations",
    ):
        value = getattr(args, name)
        if value < (0 if name == "warmups" else 1):
            raise BenchmarkError(f"--{name.replace('_', '-')} has invalid value")
    for name in (
        "step",
        "scc_charge_tolerance",
        "scc_energy_tolerance",
        "hessian_atol",
        "symmetry_atol",
        "acoustic_atol",
        "repeatability_atol",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise BenchmarkError(f"--{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(args.coordinate_timeout_seconds)
        or args.coordinate_timeout_seconds < 0.0
    ):
        raise BenchmarkError("--coordinate-timeout-seconds must be finite and nonnegative")
    if any(engine.startswith("xtbloom-") for engine in args.engines) and (
        args.library is None or not args.library.is_file()
    ):
        raise BenchmarkError("xTBloom engines require an existing --library")
    if "xtb" in args.engines and (
        args.xtb_library is None or not args.xtb_library.is_file()
    ):
        raise BenchmarkError("xTB requires an existing --xtb-library")


def main(argv: Sequence[str] | None = None) -> int:
    """Execute selected coordinates and publish JSON/CSV even for failures."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        validate_clean_sources(args)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        reference, reference_identity = load_reference(args.reference_json)
        molecule = make_alkane(NATOMS)
        rows = []
        for engine in args.engines:
            batches: Sequence[int | None] = (
                args.displacement_batch_sizes
                if engine.startswith("xtbloom-")
                else (None,)
            )
            for batch_size in batches:
                if args.coordinate_child:
                    row = run_row(
                        engine,
                        args=args,
                        molecule=molecule,
                        displacement_batch_size=batch_size,
                        reference=reference,
                    )
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="xtbloom-hessian-coordinate-"
                    ) as directory:
                        child_json = Path(directory) / "coordinate.json"
                        child_csv = Path(directory) / "coordinate.csv"
                        command = coordinate_command(
                            args,
                            engine=engine,
                            displacement_batch_size=batch_size,
                            output_json=child_json,
                            output_csv=child_csv,
                        )
                        row = run_isolated_coordinate(
                            command,
                            output_json=child_json,
                            timeout_seconds=(
                                args.coordinate_timeout_seconds
                                if args.coordinate_timeout_seconds > 0.0
                                else None
                            ),
                        )
                        row.setdefault("engine", engine)
                        row.setdefault("natoms", NATOMS)
                        row.setdefault("molecule", molecule.name)
                        row.setdefault("coordinate_count", COORDINATE_COUNT)
                        row.setdefault("displacement_count", DISPLACEMENT_COUNT)
                        row.setdefault("displacement_batch_size", batch_size)
                rows.append(row)
                availability = row["availability"]
                timing = row.get("timing") or {}
                print(  # noqa: T201 - benchmark progress belongs on stdout.
                    f"{engine} displacement_batch={batch_size}: {availability} "
                    f"median_ms={timing.get('median_ms')}"
                )
        positions = np.asarray(molecule.positions_bohr, dtype=np.float64)
        numbers = np.asarray(molecule.atomic_numbers, dtype=np.int64)
        document = {
            "metadata": runner_metadata(args, reference_identity),
            "workload": {
                "name": molecule.name,
                "natoms": NATOMS,
                "atomic_numbers_sha256": hashlib.sha256(
                    numbers.astype("<i8").tobytes()
                ).hexdigest(),
                "positions_bohr_sha256": hashlib.sha256(
                    positions.astype("<f8").tobytes()
                ).hexdigest(),
                "molecular_charge_e": 0.0,
                "unpaired_electrons": 0,
            },
            "rows": rows,
        }
        write_json(args.output_json, document)
        write_csv(args.output_csv, rows)
        print(f"wrote {args.output_json} and {args.output_csv}")  # noqa: T201
        if args.fail_on_correctness and any(
            row.get("availability") != "available"
            or (row.get("correctness") or {}).get("status") != "pass"
            for row in rows
        ):
            return 2
        return 0
    except (BenchmarkError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
