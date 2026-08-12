#!/usr/bin/env python3
"""Run committed GFN2 conformance cases through the public xtbloom C ABI.

The runner deliberately uses :mod:`ctypes`: it validates the installed/shared-
library surface rather than linking to implementation details. Cases are
submitted in property-compatible ragged batches so the same gate also exercises
public batch inference. CUDA runs can place descriptors in host, device, or
mixed memory.
Comparison uses the primary tolerances from ``data/conformance/manifest.json``
without reference-engine relaxation.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import TracebackType

import xtbloom_conformance as conformance

XTBLOOM_STATUS_SUCCESS = 0
XTBLOOM_STATUS_BACKEND_UNAVAILABLE = 2
XTBLOOM_BACKEND_CPU = 1
XTBLOOM_BACKEND_CUDA = 2
XTBLOOM_MEMORY_HOST = 0
XTBLOOM_MEMORY_CUDA_DEVICE = 1
XTBLOOM_MODEL_GFN2_XTB = 2
XTBLOOM_SCC_START_FRESH = 1
XTBLOOM_SCC_START_WARM = 2
XTBLOOM_COMPUTE_ENERGY = 1 << 0
XTBLOOM_COMPUTE_FORCES = 1 << 1
XTBLOOM_COMPUTE_ATOMIC_CHARGES = 1 << 2
XTBLOOM_COMPUTE_POINT_CHARGE_FORCES = 1 << 3
XTBLOOM_KELVIN_TO_HARTREE = 3.166808578545117e-6

XTBLOOM_INTERACTION_ELECTRIC_FIELD = 0x0101

CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class ContextOptions(ctypes.Structure):
    """ctypes mirror of ``xtbloom_context_options_t`` ABI version 1."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("backend", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("cpu_threads", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
    ]


class ConstBuffer(ctypes.Structure):
    """ctypes mirror of ``xtbloom_const_buffer_t``."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Buffer(ctypes.Structure):
    """ctypes mirror of ``xtbloom_buffer_t``."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
        ("memory_space", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class Batch(ctypes.Structure):
    """ctypes mirror of ``xtbloom_batch_t`` including its ABI-v2 suffix."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("batch_size", ctypes.c_int64),
        ("total_atoms", ctypes.c_int64),
        ("total_point_charges", ctypes.c_int64),
        ("total_charge_response_elements", ctypes.c_int64),
        ("atom_offsets", ConstBuffer),
        ("atomic_numbers", ConstBuffer),
        ("positions", ConstBuffer),
        ("molecular_charges", ConstBuffer),
        ("unpaired_electrons", ConstBuffer),
        ("point_charge_offsets", ConstBuffer),
        ("point_charge_positions", ConstBuffer),
        ("point_charge_values", ConstBuffer),
        ("point_charge_gammas", ConstBuffer),
        ("atomic_potential_shifts", ConstBuffer),
        ("charge_response_offsets", ConstBuffer),
        ("charge_response_matrix", ConstBuffer),
        ("spin_channels", ConstBuffer),
        ("total_interactions", ctypes.c_int64),
        ("interaction_descriptors", ConstBuffer),
        ("interaction_payload", ConstBuffer),
    ]


class Interaction(ctypes.Structure):
    """ctypes mirror of ``xtbloom_interaction_t`` (one attachment entry)."""

    _fields_ = [
        ("type", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("system_index", ctypes.c_int64),
        ("payload_offset", ctypes.c_uint64),
        ("payload_size", ctypes.c_uint64),
    ]


class ComputeOptions(ctypes.Structure):
    """ctypes mirror of ``xtbloom_compute_options_t`` through ABI version 3."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("model", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("max_scc_iterations", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("charge_tolerance", ctypes.c_double),
        ("energy_tolerance", ctypes.c_double),
        ("electronic_temperature", ctypes.c_double),
        ("scc_start_mode", ctypes.c_int32),
        ("reserved_v2", ctypes.c_uint32),
        ("scc_mixer", ctypes.c_int32),
        ("scc_mixer_history", ctypes.c_int32),
        ("scc_mixer_damping", ctypes.c_double),
        ("determinism", ctypes.c_int32),
        ("reserved_v3", ctypes.c_uint32),
    ]


class BatchResult(ctypes.Structure):
    """ctypes mirror of ``xtbloom_batch_result_t`` ABI version 1."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("energies", Buffer),
        ("forces", Buffer),
        ("atomic_charges", Buffer),
        ("point_charge_forces", Buffer),
        ("scc_iterations", Buffer),
        ("scc_converged", Buffer),
        ("per_system_status", Buffer),
    ]


class BackendUnavailable(conformance.ConformanceError):
    """Requested compiled backend cannot create a context on this host."""


class CudaRuntime:
    """Own CUDA allocations made through a dynamically loaded libcudart.

    Allocations are always released in reverse order. Cleanup attempts every
    ``cudaFree`` and restores the caller's current device even when allocation,
    transfer, or xtbloom execution raises an exception.
    """

    def __init__(self, device_id: int) -> None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so.12",
            "libcudart.so",
        ]
        runtime = None
        load_errors: list[str] = []
        for candidate in dict.fromkeys(value for value in candidates if value):
            try:
                runtime = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                load_errors.append(f"{candidate}: {exc}")
        if runtime is None:
            raise conformance.ConformanceError(
                "cannot dynamically load libcudart: " + "; ".join(load_errors)
            )

        self.runtime = runtime
        self.runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.runtime.cudaGetErrorString.restype = ctypes.c_char_p
        self.runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.runtime.cudaGetDevice.restype = ctypes.c_int
        self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        self.runtime.cudaSetDevice.restype = ctypes.c_int
        self.runtime.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.runtime.cudaMalloc.restype = ctypes.c_int
        self.runtime.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.runtime.cudaMemcpy.restype = ctypes.c_int
        self.runtime.cudaFree.argtypes = [ctypes.c_void_p]
        self.runtime.cudaFree.restype = ctypes.c_int

        self.allocations: list[ctypes.c_void_p] = []
        self.device_id = device_id
        self.original_device = ctypes.c_int()
        self.closed = False
        self._check(
            self.runtime.cudaGetDevice(ctypes.byref(self.original_device)),
            "cudaGetDevice",
        )
        self._check(
            self.runtime.cudaSetDevice(device_id), f"cudaSetDevice({device_id})"
        )

    def _error_name(self, status: int) -> str:
        return _decode(self.runtime.cudaGetErrorString(status))

    def _check(self, status: int, operation: str) -> None:
        if status != CUDA_SUCCESS:
            raise conformance.ConformanceError(
                f"{operation} failed with {self._error_name(status)}"
            )

    def allocate(self, size_bytes: int) -> ctypes.c_void_p:
        """Allocate nonempty device storage and register it for cleanup."""
        if size_bytes <= 0:
            raise conformance.ConformanceError(
                f"cannot allocate a non-positive CUDA buffer ({size_bytes} bytes)"
            )
        pointer = ctypes.c_void_p()
        self._check(
            self.runtime.cudaMalloc(ctypes.byref(pointer), size_bytes),
            f"cudaMalloc({size_bytes})",
        )
        self.allocations.append(pointer)
        return pointer

    def upload(self, owner: object) -> ctypes.c_void_p:
        """Allocate a device buffer and synchronously copy one ctypes owner."""
        size_bytes = ctypes.sizeof(owner)
        pointer = self.allocate(size_bytes)
        self._check(
            self.runtime.cudaMemcpy(
                pointer,
                ctypes.cast(owner, ctypes.c_void_p),
                size_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            f"cudaMemcpy(H2D, {size_bytes})",
        )
        return pointer

    def download(self, pointer: ctypes.c_void_p, owner: object) -> None:
        """Copy one device output synchronously into its ctypes host mirror."""
        size_bytes = ctypes.sizeof(owner)
        self._check(
            self.runtime.cudaMemcpy(
                ctypes.cast(owner, ctypes.c_void_p),
                pointer,
                size_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            f"cudaMemcpy(D2H, {size_bytes})",
        )

    def close(self) -> None:
        """Free all allocations and restore the entry device, attempting all steps."""
        if self.closed:
            return
        self.closed = True
        failures: list[str] = []
        status = self.runtime.cudaSetDevice(self.device_id)
        if status != CUDA_SUCCESS:
            failures.append(
                f"select allocation device {self.device_id} for cleanup: "
                f"{self._error_name(status)}"
            )
        while self.allocations:
            pointer = self.allocations.pop()
            status = self.runtime.cudaFree(pointer)
            if status != CUDA_SUCCESS:
                failures.append(f"cudaFree: {self._error_name(status)}")
        status = self.runtime.cudaSetDevice(self.original_device.value)
        if status != CUDA_SUCCESS:
            failures.append(
                f"restore cuda device {self.original_device.value}: "
                f"{self._error_name(status)}"
            )
        if failures:
            raise conformance.ConformanceError("; ".join(failures))

    def __enter__(self) -> CudaRuntime:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this allocation owner for a managed CUDA operation."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release CUDA resources while preserving an active exception."""
        try:
            self.close()
        except conformance.ConformanceError as cleanup_error:
            if exc_type is None:
                raise
            print(  # noqa: T201 - CLI validation report
                f"warning: CUDA cleanup failed: {cleanup_error}", file=sys.stderr
            )
        return False


@dataclass
class CaseSlice:
    """Offsets needed to split one public ragged-batch result into case JSON."""

    case: dict[str, Any]
    atom_begin: int
    atom_end: int
    point_begin: int
    point_end: int
    expected: dict[str, Any]


@dataclass
class PublicBatchStorage:
    """Python values and ctypes owners whose addresses are passed to xtbloom."""

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
    slices: list[CaseSlice]
    keepalive: list[Any]
    # Per-system electric field in atomic units aligned with ``slices``
    # (index i belongs to slice i); ``None`` marks a system without a field.
    efields: list[list[float] | None] = field(default_factory=list)


MIXED_DEVICE_ROLES = {
    # Keep topology identifiers/offsets on the host while numerical inputs are
    # device-resident. This mirrors a useful application staging boundary.
    "positions",
    "molecular_charges",
    "unpaired_electrons",
    "point_charge_positions",
    "point_charge_values",
    "point_charge_gammas",
    # Exercise mixed output publication too: large Cartesian results and one
    # diagnostic use device pointers while scalar/state outputs remain host.
    "forces",
    "point_charge_forces",
    "scc_converged",
}


class DescriptorMemory:
    """Bind public descriptors according to one explicit memory placement mode."""

    def __init__(self, mode: str, device_id: int) -> None:
        self.mode = mode
        self.cuda = CudaRuntime(device_id) if mode != "host" else None
        self.device_outputs: list[tuple[ctypes.c_void_p, Any]] = []

    def _is_device(self, role: str) -> bool:
        return self.mode == "device" or (
            self.mode == "mixed" and role in MIXED_DEVICE_ROLES
        )

    def input(
        self,
        storage: PublicBatchStorage,
        values: Sequence[int | float],
        scalar: type[ctypes._SimpleCData],
        role: str,
    ) -> ConstBuffer:
        """Create one host or CUDA input descriptor and retain its owner."""
        if not values:
            return ConstBuffer(None, 0, XTBLOOM_MEMORY_HOST, 0)
        owner = (scalar * len(values))(*values)
        storage.keepalive.append(owner)
        if self._is_device(role):
            assert self.cuda is not None
            pointer = self.cuda.upload(owner)
            return ConstBuffer(
                pointer,
                ctypes.sizeof(owner),
                XTBLOOM_MEMORY_CUDA_DEVICE,
                0,
            )
        return ConstBuffer(
            ctypes.cast(owner, ctypes.c_void_p),
            ctypes.sizeof(owner),
            XTBLOOM_MEMORY_HOST,
            0,
        )

    def input_owner(
        self,
        storage: PublicBatchStorage,
        owner: object,
        role: str,
    ) -> ConstBuffer:
        """Create one host or CUDA input descriptor for an existing owner."""
        storage.keepalive.append(owner)
        if self._is_device(role):
            assert self.cuda is not None
            pointer = self.cuda.upload(owner)
            return ConstBuffer(
                pointer,
                ctypes.sizeof(owner),
                XTBLOOM_MEMORY_CUDA_DEVICE,
                0,
            )
        return ConstBuffer(
            ctypes.cast(owner, ctypes.c_void_p),
            ctypes.sizeof(owner),
            XTBLOOM_MEMORY_HOST,
            0,
        )

    def output(self, owner: object, role: str) -> Buffer:
        """Create one output descriptor and retain device-to-host download state."""
        if self._is_device(role):
            assert self.cuda is not None
            pointer = self.cuda.allocate(ctypes.sizeof(owner))
            self.device_outputs.append((pointer, owner))
            return Buffer(
                pointer,
                ctypes.sizeof(owner),
                XTBLOOM_MEMORY_CUDA_DEVICE,
                0,
            )
        return Buffer(
            ctypes.cast(owner, ctypes.c_void_p),
            ctypes.sizeof(owner),
            XTBLOOM_MEMORY_HOST,
            0,
        )

    def download_outputs(self) -> None:
        """Materialize all device result buffers before Python inspects them."""
        if self.cuda is None:
            return
        for pointer, owner in self.device_outputs:
            self.cuda.download(pointer, owner)

    def close(self) -> None:
        """Release device storage when this descriptor set owns any."""
        if self.cuda is not None:
            self.cuda.close()

    def __enter__(self) -> DescriptorMemory:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this descriptor owner for a managed compute operation."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release any device storage while propagating active exceptions."""
        if self.cuda is None:
            return False
        return self.cuda.__exit__(exc_type, exc, traceback)


def _configure_library(path: Path) -> ctypes.CDLL:
    """Load one shared xtbloom library and declare every called C signature."""
    if not path.is_file():
        raise conformance.ConformanceError(f"xtbloom shared library is missing: {path}")
    try:
        library = ctypes.CDLL(str(path.resolve()))
    except OSError as exc:
        raise conformance.ConformanceError(
            f"cannot load xtbloom shared library {path}: {exc}"
        ) from exc

    library.xtbloom_get_last_error.argtypes = []
    library.xtbloom_get_last_error.restype = ctypes.c_char_p
    library.xtbloom_status_string.argtypes = [ctypes.c_int32]
    library.xtbloom_status_string.restype = ctypes.c_char_p
    library.xtbloom_context_options_init.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.c_size_t,
    ]
    library.xtbloom_context_options_init.restype = ctypes.c_int32
    library.xtbloom_batch_init.argtypes = [ctypes.POINTER(Batch), ctypes.c_size_t]
    library.xtbloom_batch_init.restype = ctypes.c_int32
    library.xtbloom_compute_options_init.argtypes = [
        ctypes.POINTER(ComputeOptions),
        ctypes.c_size_t,
    ]
    library.xtbloom_compute_options_init.restype = ctypes.c_int32
    library.xtbloom_batch_result_init.argtypes = [
        ctypes.POINTER(BatchResult),
        ctypes.c_size_t,
    ]
    library.xtbloom_batch_result_init.restype = ctypes.c_int32
    library.xtbloom_context_create.argtypes = [
        ctypes.POINTER(ContextOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.xtbloom_context_create.restype = ctypes.c_int32
    library.xtbloom_context_destroy.argtypes = [ctypes.c_void_p]
    library.xtbloom_context_destroy.restype = None
    library.xtbloom_context_get_backend.argtypes = [ctypes.c_void_p]
    library.xtbloom_context_get_backend.restype = ctypes.c_int32
    library.xtbloom_compute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Batch),
        ctypes.POINTER(ComputeOptions),
        ctypes.POINTER(BatchResult),
    ]
    library.xtbloom_compute.restype = ctypes.c_int32
    return library


def _decode(value: bytes | None) -> str:
    """Decode a public diagnostic string without hiding a NULL return."""
    return "<null>" if value is None else value.decode("utf-8", errors="replace")


def _call_ok(library: ctypes.CDLL, status: int, operation: str) -> None:
    """Turn a failing C initializer or compute call into an actionable error."""
    if status == XTBLOOM_STATUS_SUCCESS:
        return
    raise conformance.ConformanceError(
        f"{operation} failed with {_decode(library.xtbloom_status_string(status))}: "
        f"{_decode(library.xtbloom_get_last_error())}"
    )


def supported_cases(
    manifest: dict[str, Any], names: list[str] | None, backend: str | None = None
) -> list[dict[str, Any]]:
    """Return selected cases supported by ``backend`` (or by either backend)."""
    cases = conformance.selected_cases(manifest, names)
    if backend is None:
        return cases
    return [
        case
        for case in cases
        if backend in case.get("xtbloom_backends", ["cpu", "cuda"])
    ]


def _flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Flatten a validated atom-major or point-major Cartesian matrix."""
    return [float(component) for row in matrix for component in row]


def pack_efield_interactions(
    efields: Sequence[list[float] | None],
) -> tuple[list[Interaction], bytes]:
    """Build electric-field interaction descriptors and one byte payload.

    Each finite three-component field (atomic units) becomes one 32-byte
    block_version-1 payload (int32 1, int32 0, three little-endian doubles)
    attached to the batch system whose slice index equals the descriptor's
    index. ``None`` systems produce no descriptor.
    """
    descriptors: list[Interaction] = []
    payload = bytearray()
    for index, components in enumerate(efields):
        if components is None:
            continue
        values = [float(value) for value in components]
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise conformance.ConformanceError(
                f"system {index} efield must be a finite three-component list"
            )
        descriptors.append(
            Interaction(
                XTBLOOM_INTERACTION_ELECTRIC_FIELD,
                0,
                index,
                len(payload),
                32,
            )
        )
        payload.extend(struct.pack("<i", 1))  # block_version
        payload.extend(struct.pack("<i", 0))  # reserved
        payload.extend(struct.pack("<3d", *values))
    return descriptors, bytes(payload)


def assemble_batch(
    manifest_path: Path,
    manifest: dict[str, Any],
    cases: Sequence[dict[str, Any]],
) -> PublicBatchStorage:
    """Load supported corpus inputs and concatenate them into one ragged batch."""
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    molecular_charges: list[float] = []
    unpaired_electrons: list[int] = []
    spin_channels: list[int] = []
    point_charge_offsets = [0]
    point_charge_positions: list[float] = []
    point_charge_values: list[float] = []
    point_charge_gammas: list[float] = []
    slices: list[CaseSlice] = []
    efields: list[list[float] | None] = []
    hardness = manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"]

    for case in cases:
        input_path = conformance.resolve_manifest_path(manifest_path, case["input"])
        atom_begin = len(atomic_numbers)
        point_begin = len(point_charge_values)
        efield = case.get("efield")
        if efield is None:
            efields.append(None)
        else:
            if not isinstance(efield, list) or len(efield) != 3:
                raise conformance.ConformanceError(
                    f"case {case['id']} efield must be a three-component list "
                    "in atomic units"
                )
            efields.append([float(component) for component in efield])
        if case.get("input_schema") == "qmmm-v1":
            document = conformance.load_qmmm_input(input_path, case, hardness)
            qm = document["qm"]
            points = document["external_point_charges"]
            atomic_numbers.extend(int(number) for number in qm["atomic_numbers"])
            positions.extend(_flatten(qm["positions_bohr"]))
            point_charge_positions.extend(_flatten(points["positions_bohr"]))
            point_charge_values.extend(float(value) for value in points["charges_e"])
            point_charge_gammas.extend(
                float(value) for value in points["gammas_hartree"]
            )
        else:
            document = conformance.load_turbomole_coord(input_path, case)
            atomic_numbers.extend(document["atomic_numbers"])
            positions.extend(_flatten(document["positions_bohr"]))

        atom_offsets.append(len(atomic_numbers))
        point_charge_offsets.append(len(point_charge_values))
        molecular_charges.append(float(case["molecular_charge"]))
        unpaired = int(case["unpaired_electrons"])
        unpaired_electrons.append(unpaired)
        spin_channel_count = case.get("spin_channels", 1)
        if type(spin_channel_count) is not int or spin_channel_count not in (1, 2):
            raise conformance.ConformanceError(
                f"case {case['id']} spin_channels must be one or two"
            )
        spin_channels.append(spin_channel_count)
        golden = conformance.load_json(
            conformance.resolve_manifest_path(manifest_path, case["golden"])
        )
        slices.append(
            CaseSlice(
                case=case,
                atom_begin=atom_begin,
                atom_end=len(atomic_numbers),
                point_begin=point_begin,
                point_end=len(point_charge_values),
                expected=golden["properties"],
            )
        )

    return PublicBatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=molecular_charges,
        unpaired_electrons=unpaired_electrons,
        spin_channels=spin_channels,
        point_charge_offsets=point_charge_offsets,
        point_charge_positions=point_charge_positions,
        point_charge_values=point_charge_values,
        point_charge_gammas=point_charge_gammas,
        slices=slices,
        keepalive=[],
        efields=efields,
    )


def _make_batch(
    library: ctypes.CDLL,
    storage: PublicBatchStorage,
    memory: DescriptorMemory,
    include_spin_channels: bool,
) -> Batch:
    """Initialize and bind a public batch using the selected placement policy.

    ABI-v2 calls explicitly bind the spin suffix on both backends, including
    channel value one for restricted systems. ``include_spin_channels=False``
    remains available only for explicit ABI-v1 compatibility checks. When any
    system carries a uniform electric field, the ABI-v3 ``total_interactions``
    and its descriptor/payload buffers are bound as well.
    """
    batch = Batch()
    _call_ok(
        library,
        library.xtbloom_batch_init(ctypes.byref(batch), ctypes.sizeof(batch)),
        "xtbloom_batch_init",
    )
    batch.batch_size = len(storage.slices)
    batch.total_atoms = len(storage.atomic_numbers)
    batch.total_point_charges = len(storage.point_charge_values)
    batch.total_charge_response_elements = 0
    batch.atom_offsets = memory.input(
        storage, storage.atom_offsets, ctypes.c_int64, "atom_offsets"
    )
    batch.atomic_numbers = memory.input(
        storage, storage.atomic_numbers, ctypes.c_int32, "atomic_numbers"
    )
    batch.positions = memory.input(
        storage, storage.positions, ctypes.c_double, "positions"
    )
    batch.molecular_charges = memory.input(
        storage, storage.molecular_charges, ctypes.c_double, "molecular_charges"
    )
    batch.unpaired_electrons = memory.input(
        storage, storage.unpaired_electrons, ctypes.c_int32, "unpaired_electrons"
    )
    if include_spin_channels:
        batch.spin_channels = memory.input(
            storage, storage.spin_channels, ctypes.c_int32, "spin_channels"
        )
    if any(efield is not None for efield in storage.efields):
        descriptors, payload = pack_efield_interactions(storage.efields)
        batch.total_interactions = len(descriptors)
        descriptor_owner = (Interaction * len(descriptors))(*descriptors)
        batch.interaction_descriptors = memory.input_owner(
            storage, descriptor_owner, "interaction_descriptors"
        )
        payload_owner = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        batch.interaction_payload = memory.input_owner(
            storage, payload_owner, "interaction_payload"
        )
    if storage.point_charge_values:
        batch.point_charge_offsets = memory.input(
            storage,
            storage.point_charge_offsets,
            ctypes.c_int64,
            "point_charge_offsets",
        )
        batch.point_charge_positions = memory.input(
            storage,
            storage.point_charge_positions,
            ctypes.c_double,
            "point_charge_positions",
        )
        batch.point_charge_values = memory.input(
            storage,
            storage.point_charge_values,
            ctypes.c_double,
            "point_charge_values",
        )
        batch.point_charge_gammas = memory.input(
            storage,
            storage.point_charge_gammas,
            ctypes.c_double,
            "point_charge_gammas",
        )
    return batch


def _make_context(
    library: ctypes.CDLL, backend: str, device_id: int, cpu_threads: int
) -> ctypes.c_void_p:
    """Create one explicitly selected backend context with no fallback."""
    backend_tag = XTBLOOM_BACKEND_CPU if backend == "cpu" else XTBLOOM_BACKEND_CUDA
    options = ContextOptions()
    _call_ok(
        library,
        library.xtbloom_context_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
        "xtbloom_context_options_init",
    )
    options.backend = backend_tag
    options.device_id = device_id
    options.cpu_threads = cpu_threads
    context = ctypes.c_void_p()
    status = library.xtbloom_context_create(
        ctypes.byref(options), ctypes.byref(context)
    )
    if status == XTBLOOM_STATUS_BACKEND_UNAVAILABLE:
        diagnostic = _decode(library.xtbloom_get_last_error())
        raise BackendUnavailable(f"{backend} backend unavailable: {diagnostic}")
    _call_ok(library, status, f"create {backend} context")
    if library.xtbloom_context_get_backend(context) != backend_tag:
        library.xtbloom_context_destroy(context)
        raise conformance.ConformanceError(
            f"requested explicit {backend} backend but context selected another backend"
        )
    return context


def _compare_case(
    manifest: dict[str, Any],
    case_slice: CaseSlice,
    actual: dict[str, Any],
    unsupported_properties: dict[str, str],
) -> list[str]:
    """Compare exactly the properties exposed by the current public C result ABI."""
    mappings = [
        ("energy_hartree", "energy"),
        ("forces_hartree_per_bohr", "forces"),
        ("partial_charges_e", "charges"),
        ("point_charge_forces_hartree_per_bohr", "point_charge_forces"),
    ]
    failures: list[str] = []
    oracle_properties = case_slice.case.get("xtbloom_oracle_properties")
    for property_name, tolerance_name in mappings:
        if property_name not in case_slice.expected:
            continue
        if oracle_properties is not None and property_name not in oracle_properties:
            print(  # noqa: T201 - CLI validation report
                f"INFO {case_slice.case['id']} {property_name}: pinned reference "
                "is diagnostic-only; public energy finite differences are the "
                "force authority"
            )
            continue
        if property_name in unsupported_properties:
            print(  # noqa: T201 - CLI validation report
                f"SKIP {case_slice.case['id']} {property_name}: "
                f"{unsupported_properties[property_name]}"
            )
            continue
        if property_name not in actual:
            message = f"{case_slice.case['id']} is missing {property_name}"
            print(f"FAIL {message}")  # noqa: T201 - CLI validation report
            failures.append(message)
            continue
        tolerance = manifest["tolerances"][tolerance_name]
        passed, message = conformance.compare_values(
            case_slice.case["id"],
            property_name,
            case_slice.expected[property_name],
            actual[property_name],
            float(tolerance["atol"]),
            float(tolerance["rtol"]),
        )
        print(  # noqa: T201 - CLI validation report
            ("PASS " if passed else "FAIL ") + message
        )
        if not passed:
            failures.append(message)
    return failures


def pinned_compute_options(
    library: ctypes.CDLL,
    request_forces: bool,
    request_charges: bool,
    request_point_forces: bool,
) -> ComputeOptions:
    """Build the strict single-shot GFN2 options shared by every conformance run.

    Conformance cases must remain independent so reference comparisons never
    depend on execution order or an earlier checkpoint. The SCC solve is pinned
    stricter than the public convenience defaults so the gates measure
    model/property agreement rather than loose convergence.
    """
    options = ComputeOptions()
    _call_ok(
        library,
        library.xtbloom_compute_options_init(
            ctypes.byref(options), ctypes.sizeof(options)
        ),
        "xtbloom_compute_options_init",
    )
    options.scc_start_mode = XTBLOOM_SCC_START_FRESH
    options.model = XTBLOOM_MODEL_GFN2_XTB
    options.flags = XTBLOOM_COMPUTE_ENERGY
    if request_forces:
        options.flags |= XTBLOOM_COMPUTE_FORCES
    if request_charges:
        options.flags |= XTBLOOM_COMPUTE_ATOMIC_CHARGES
    if request_forces and request_point_forces:
        options.flags |= XTBLOOM_COMPUTE_POINT_CHARGE_FORCES
    options.max_scc_iterations = 500
    options.charge_tolerance = 1.0e-10
    options.energy_tolerance = 1.0e-12
    options.electronic_temperature = 300.0 * XTBLOOM_KELVIN_TO_HARTREE
    return options


@dataclass
class RawBatchOutputs:
    """Raw public batch outputs shared by golden comparison and invariance gates."""

    energies: Any
    forces: Any | None
    charges: Any | None
    point_forces: Any | None
    iterations: Any
    converged: Any
    statuses: Any
    flags: int


def run_compute(
    library: ctypes.CDLL,
    storage: PublicBatchStorage,
    options: ComputeOptions,
    backend: str,
    device_id: int,
    cpu_threads: int,
    memory_mode: str,
) -> RawBatchOutputs:
    """Execute one public ragged batch and materialize every requested output.

    This is the single ABI execution path shared by the golden comparison runner
    and the invariance checks, so both gates necessarily exercise the identical
    descriptor binding, publication, and failure semantics.
    """
    systems = len(storage.slices)
    atoms = len(storage.atomic_numbers)
    points = len(storage.point_charge_values)
    request_forces = bool(options.flags & XTBLOOM_COMPUTE_FORCES)
    request_charges = bool(options.flags & XTBLOOM_COMPUTE_ATOMIC_CHARGES)
    request_point_forces = bool(options.flags & XTBLOOM_COMPUTE_POINT_CHARGE_FORCES)
    energies = (ctypes.c_double * systems)()
    forces = (ctypes.c_double * (3 * atoms))() if request_forces else None
    charges = (ctypes.c_double * atoms)() if request_charges else None
    point_forces = (
        (ctypes.c_double * (3 * points))() if request_point_forces and points else None
    )
    iterations = (ctypes.c_int32 * systems)()
    converged = (ctypes.c_uint8 * systems)()
    statuses = (ctypes.c_int32 * systems)()
    context = _make_context(library, backend, device_id, cpu_threads)
    try:
        with DescriptorMemory(memory_mode, device_id) as memory:
            batch = _make_batch(
                library,
                storage,
                memory,
                include_spin_channels=True,
            )
            result = BatchResult()
            _call_ok(
                library,
                library.xtbloom_batch_result_init(
                    ctypes.byref(result), ctypes.sizeof(result)
                ),
                "xtbloom_batch_result_init",
            )
            result.energies = memory.output(energies, "energies")
            if forces is not None:
                result.forces = memory.output(forces, "forces")
            if charges is not None:
                result.atomic_charges = memory.output(charges, "atomic_charges")
            if point_forces is not None:
                result.point_charge_forces = memory.output(
                    point_forces, "point_charge_forces"
                )
            result.scc_iterations = memory.output(iterations, "scc_iterations")
            result.scc_converged = memory.output(converged, "scc_converged")
            result.per_system_status = memory.output(statuses, "per_system_status")
            _call_ok(
                library,
                library.xtbloom_compute(
                    context,
                    ctypes.byref(batch),
                    ctypes.byref(options),
                    ctypes.byref(result),
                ),
                f"{backend}/{memory_mode} public batch inference",
            )
            memory.download_outputs()
    finally:
        library.xtbloom_context_destroy(context)

    for index, item in enumerate(storage.slices):
        if statuses[index] != XTBLOOM_STATUS_SUCCESS or converged[index] != 1:
            raise conformance.ConformanceError(
                f"{backend}/{memory_mode} public inference produced a failed "
                f"system {item.case['id']}: per_system_status="
                f"{_decode(library.xtbloom_status_string(statuses[index]))}, "
                f"scc_converged={converged[index]}, iterations={iterations[index]}"
            )
    return RawBatchOutputs(
        energies=energies,
        forces=forces,
        charges=charges,
        point_forces=point_forces,
        iterations=iterations,
        converged=converged,
        statuses=statuses,
        flags=int(result.flags),
    )


def run_backend(
    library: ctypes.CDLL,
    manifest_path: Path,
    manifest: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    backend: str,
    actual_root: Path,
    device_id: int,
    cpu_threads: int,
    memory_mode: str,
    request_forces: bool,
) -> None:
    """Execute and compare one property-compatible public ragged batch."""
    storage = assemble_batch(manifest_path, manifest, cases)
    request_charges = any(
        "partial_charges_e" in item.expected for item in storage.slices
    )
    options = pinned_compute_options(
        library,
        request_forces,
        request_charges,
        request_point_forces=bool(storage.point_charge_values),
    )
    outputs = run_compute(
        library, storage, options, backend, device_id, cpu_threads, memory_mode
    )
    energies, forces, charges, point_forces = (
        outputs.energies,
        outputs.forces,
        outputs.charges,
        outputs.point_forces,
    )

    artifact_name = backend if memory_mode == "host" else f"{backend}-{memory_mode}"
    backend_dir = actual_root / artifact_name
    failures: list[str] = []
    unsupported_properties: dict[str, str] = {}
    for index, item in enumerate(storage.slices):
        properties: dict[str, Any] = {
            "energy_hartree": float(energies[index]),
        }
        if forces is not None:
            properties["forces_hartree_per_bohr"] = [
                float(value)
                for value in forces[3 * item.atom_begin : 3 * item.atom_end]
            ]
        if "partial_charges_e" in item.expected:
            properties["partial_charges_e"] = [
                float(value) for value in charges[item.atom_begin : item.atom_end]
            ]
        if "point_charge_forces_hartree_per_bohr" in item.expected:
            assert point_forces is not None
            properties["point_charge_forces_hartree_per_bohr"] = [
                float(value)
                for value in point_forces[3 * item.point_begin : 3 * item.point_end]
            ]
        document = {
            "backend": backend,
            "case_id": item.case["id"],
            "diagnostics": {
                "per_system_status": int(outputs.statuses[index]),
                "scc_converged": int(outputs.converged[index]),
                "scc_iterations": int(outputs.iterations[index]),
            },
            "memory_mode": memory_mode,
            "method": manifest["method"],
            "properties": properties,
            "provenance": {
                "backend": backend,
                "engine": "xtbloom",
                "memory_mode": memory_mode,
                "spin_channels": storage.spin_channels[index],
            },
            "result_flags": int(outputs.flags),
            "schema_version": manifest["golden_schema_version"],
            "unsupported_properties": unsupported_properties,
            "units": manifest["units"],
        }
        conformance.dump_json(backend_dir / f"{item.case['id']}.json", document)
        failures.extend(
            _compare_case(
                manifest,
                item,
                properties,
                unsupported_properties,
            )
        )
    if failures:
        raise conformance.ConformanceError(
            f"{backend}/{memory_mode}: {len(failures)} public C API conformance "
            f"comparison(s) failed; actual JSON retained in {backend_dir}"
        )
    print(  # noqa: T201 - CLI validation report
        f"public C API conformance OK: backend={backend}, "
        f"memory_mode={memory_mode}, cases={len(storage.slices)}, "
        f"forces={request_forces}, actual_dir={backend_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Define the focused local and CTest invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=conformance.DEFAULT_MANIFEST)
    parser.add_argument("--backend", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument(
        "--memory-mode",
        choices=("host", "device", "mixed"),
        default="host",
        help=(
            "descriptor placement; device and mixed require --backend cuda "
            "and dynamically load libcudart"
        ),
    )
    parser.add_argument(
        "--actual-dir",
        type=Path,
        default=conformance.REPOSITORY_ROOT
        / "build"
        / "conformance"
        / "xtbloom-public",
    )
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--skip-backend-unavailable",
        action="store_true",
        help="return 77 when an explicitly requested backend cannot create a context",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run selected supported cases and preserve emitted values on failure."""
    args = build_parser().parse_args(argv)
    try:
        if args.memory_mode != "host" and args.backend != "cuda":
            raise conformance.ConformanceError(
                "CPU backend only supports --memory-mode host; select "
                "--backend cuda for device or mixed descriptors"
            )
        manifest = conformance.load_json(args.manifest)
        selected = supported_cases(manifest, args.cases)
        backends = ("cpu", "cuda") if args.backend == "all" else (args.backend,)
        cases_by_backend = {
            backend: supported_cases(manifest, args.cases, backend)
            for backend in backends
        }
        if not selected:
            print(  # noqa: T201 - CLI validation report
                "no GFN2 conformance cases selected"
            )
            return 0
        if not any(cases_by_backend.values()):
            for backend in backends:
                for case in selected:
                    print(  # noqa: T201 - CLI validation report
                        f"SKIP {case['id']}: public backend {backend} is not "
                        "released for this case"
                    )
            return 0
        library = _configure_library(args.library)
        for backend in backends:
            cases = cases_by_backend[backend]
            unsupported = [case for case in selected if case not in cases]
            for case in unsupported:
                print(  # noqa: T201 - CLI validation report
                    f"SKIP {case['id']}: public backend {backend} is not released "
                    "for this case"
                )
            if not cases:
                continue
            shared_orbital = [
                case for case in cases if int(case.get("spin_channels", 1)) == 1
            ]
            spin_polarized = [
                case for case in cases if int(case.get("spin_channels", 1)) == 2
            ]
            if shared_orbital:
                run_backend(
                    library,
                    args.manifest,
                    manifest,
                    shared_orbital,
                    backend,
                    args.actual_dir,
                    args.device_id,
                    args.cpu_threads,
                    args.memory_mode,
                    request_forces=True,
                )
            if spin_polarized:
                run_backend(
                    library,
                    args.manifest,
                    manifest,
                    spin_polarized,
                    backend,
                    args.actual_dir,
                    args.device_id,
                    args.cpu_threads,
                    args.memory_mode,
                    request_forces=True,
                )
    except BackendUnavailable as exc:
        print(f"SKIP {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 77 if args.skip_backend_unavailable else 1
    except conformance.ConformanceError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
