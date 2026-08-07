"""Minimal DLPack consumer bridge for the gpuxtb Python interface.

The public C ABI accepts borrowed host/CUDA buffers.  Dense eager arrays from
NumPy, CuPy, JAX, and PyTorch all expose the DLPack producer protocol
(``__dlpack__`` / ``__dlpack_device__``), so this module turns such an array
into a validated ``gpuxtb`` descriptor view without importing any backend
package and without copying the array's data.

The consumer follows the DLPack 1.0 Python specification:

* DLPack 1.0 is negotiated with ``max_version=(1, 0)``; a producer whose
  signature rejects that keyword is retried without it while preserving the
  CUDA consumer stream and any explicit copy request.
* Both capsule forms (``dltensor`` and ``dltensor_versioned``) are accepted;
  unsupported major versions are rejected.
* The producer object and the consumed capsule are kept alive through the
  caller's synchronous ``gpuxtb_compute`` return.  A consumed capsule is
  renamed to ``used_dltensor[_versioned]`` and its destructor is detached,
  so the bridge owns exactly one producer-deleter call even on error paths.

The bridge deliberately does not import ``array_api_compat`` or any array
library; :mod:`gpuxtb._array` owns the backend-neutral probing and this module
owns only the raw capsule contract.

No public C ABI is used here and nothing is exported from the package: this
module is an implementation detail of :class:`gpuxtb.interface.ArrayBatch`.
"""

from __future__ import annotations

import contextlib
import ctypes
import math
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from . import library
from .exceptions import GPUxtbNotSupportedError, GPUxtbRuntimeError, GPUxtbValueError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# DLPack 1.0 device-kind constants (DLDeviceType from the DLPack C header).
_DLPACK_DEVICE_CPU = 1
_DLPACK_DEVICE_CUDA = 2
_DLPACK_DEVICE_CUDA_HOST = 3
_DLPACK_DEVICE_ROCM = 10
_DLPACK_DEVICE_ROCM_HOST = 11
_DLPACK_DEVICE_CUDA_MANAGED = 13

# DLDataType code constants (DLDataTypeCode from the DLPack C header).
_DLPACK_CODE_INT = 0
_DLPACK_CODE_UINT = 1
_DLPACK_CODE_FLOAT = 2
_DLPACK_CODE_BFLOAT = 4
_DLPACK_CODE_COMPLEX = 5
_DLPACK_CODE_BOOL = 6

# Flags carried by the versioned managed tensor (DLManagedTensorVersioned).
_DLPACK_FLAG_READ_ONLY = 1 << 0
_DLPACK_FLAG_IS_COPIED = 1 << 1

# Capsule names mandated by the DLPack specification.  A consumed capsule is
# renamed from the producer name to the matching ``used_*`` name so a stray
# capsule can never be mistaken for (or double-deleted as) a fresh transfer.
_CAPSULE_NAME = b"dltensor"
_CAPSULE_NAME_VERSIONED = b"dltensor_versioned"
_USED_CAPSULE_NAME = b"used_dltensor"
_USED_CAPSULE_NAME_VERSIONED = b"used_dltensor_versioned"

# The highest DLPack (major, minor) pair this consumer understands.  Version
# 1.0 introduced the versioned capsule and stream negotiation.
_DLPACK_MAX_VERSION = (1, 0)

# DLPack stream convention: when gpuxtb's context uses the native legacy CUDA
# stream (``context_options.stream == NULL``) producers receive stream value
# ``1`` (cudaStreamLegacy).  Custom ``CUstream`` handles pass their raw
# pointer value.  ``-1`` means "do not synchronize" and is never used without
# an explicit, proven-safe design.
_DLPACK_LEGACY_STREAM = 1

# ``gpuxtb_buffer_t`` stores byte sizes as ``size_t`` and native validation
# performs pointer-range arithmetic in ``uintptr_t``.  CPython exposes both
# with pointer width on supported platforms, so this is the exact representable
# upper bound for descriptors crossing the ctypes boundary.
_POINTER_MAX = ctypes.c_size_t(-1).value

# --- ctypes mirrors of the DLPack 1.0 C structs -------------------------------

# All struct layouts below were verified byte-for-byte against NumPy's
# ``dltensor_versioned`` capsule and the DLPack 1.0 header that NumPy vendors
# as ``numpy/_core/src/common/dlpack/dlpack.h``.


class _DLDataType(ctypes.Structure):
    """Mirror of ``DLDataType``: type code, bit width, and vector lanes."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLDevice(ctypes.Structure):
    """Mirror of ``DLDevice``: the device-kind enum and its ordinal."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLTensor(ctypes.Structure):
    """Mirror of ``DLTensor``: 48-byte plain tensor descriptor."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    """Mirror of the legacy ``DLManagedTensor`` (64 bytes, unversioned)."""

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    """Mirror of ``DLManagedTensorVersioned`` (80 bytes, DLPack 1.0).

    Layout: ``DLPackVersion`` (major+minor), ``manager_ctx``, ``deleter``,
    ``flags``, then the 48-byte ``DLTensor``.
    """

    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("version_major", ctypes.c_uint32),
        ("version_minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


# --- CPython capsule protocol -------------------------------------------------

_pyapi = ctypes.pythonapi
_pyapi.PyCapsule_IsValid.restype = ctypes.c_int
_pyapi.PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pyapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
_pyapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pyapi.PyCapsule_SetName.restype = ctypes.c_int
_pyapi.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pyapi.PyCapsule_SetDestructor.restype = ctypes.c_int
_pyapi.PyCapsule_SetDestructor.argtypes = [ctypes.py_object, ctypes.c_void_p]
_pyapi.PyCapsule_New.restype = ctypes.py_object
_pyapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_pyapi.PyCapsule_GetName.restype = ctypes.c_char_p
_pyapi.PyCapsule_GetName.argtypes = [ctypes.py_object]

_NP_DTYPE_TO_DLPACK: dict[np.dtype, tuple[int, int]] = {
    np.dtype(np.int64): (_DLPACK_CODE_INT, 64),
    np.dtype(np.int32): (_DLPACK_CODE_INT, 32),
    np.dtype(np.int16): (_DLPACK_CODE_INT, 16),
    np.dtype(np.int8): (_DLPACK_CODE_INT, 8),
    np.dtype(np.uint8): (_DLPACK_CODE_UINT, 8),
    np.dtype(np.float64): (_DLPACK_CODE_FLOAT, 64),
    np.dtype(np.float32): (_DLPACK_CODE_FLOAT, 32),
}

# Expected descriptor names/types used by the ABI batch, mirrored here so
# diagnostic messages are stable.  Keys are the batch field names.
EXPECTED_INPUT_DTYPES: dict[str, np.dtype] = {
    "atom_offsets": np.dtype(np.int64),
    "atomic_numbers": np.dtype(np.int32),
    "positions": np.dtype(np.float64),
    "molecular_charges": np.dtype(np.float64),
    "unpaired_electrons": np.dtype(np.int32),
    "spin_channels": np.dtype(np.int32),
    "point_charge_offsets": np.dtype(np.int64),
    "point_charge_positions": np.dtype(np.float64),
    "point_charge_values": np.dtype(np.float64),
    "point_charge_gammas": np.dtype(np.float64),
    "atomic_potential_shifts": np.dtype(np.float64),
    "charge_response_offsets": np.dtype(np.int64),
    "charge_response_matrix": np.dtype(np.float64),
}

EXPECTED_OUTPUT_DTYPES: dict[str, np.dtype] = {
    "energies": np.dtype(np.float64),
    "forces": np.dtype(np.float64),
    "charges": np.dtype(np.float64),
    "point_charge_forces": np.dtype(np.float64),
    "scc_iterations": np.dtype(np.int32),
    "scc_converged": np.dtype(np.uint8),
    "per_system_status": np.dtype(np.int32),
}


def _dlpack_bytes(item_count: Sequence[int], itemsize: int) -> int:
    """Return the exact byte extent of ``item_count`` items without overflow."""
    return int(math.prod(item_count)) * itemsize


@dataclass
class DLPackView:
    """A validated borrowed DLPack tensor usable as one C-ABI buffer.

    The view owns the consumed capsule: :meth:`release` invokes the producer's
    deleter exactly once and must be called by the consumer after the
    synchronous compute call (including on every error path after the view was
    committed).  Until then the view keeps the producer object alive.
    """

    pointer: int
    size_bytes: int
    memory_space: int
    device_type: int
    device_id: int
    writable: bool
    is_copy: bool
    shape: tuple[int, ...]
    _producer: object | None
    _capsule: object | None
    _deleter: Callable[[int], None] | None
    _managed_pointer: int
    _released: bool = False

    # public ---

    def release(self) -> None:
        """Invoke the producer deleter once and drop the capsule reference.

        Idempotent: the second call is a no-op so a shared release pass on
        both the success and the exception path cannot double-delete.
        """
        if self._released:
            return
        self._released = True
        deleter = self._deleter
        if deleter is not None and self._managed_pointer:
            deleter(self._managed_pointer)
        self._producer = None
        self._capsule = None
        self._deleter = None

    @property
    def descriptor(self) -> library.ConstBuffer:
        """The borrowed ``gpuxtb_const_buffer_t`` for this view."""
        return library.ConstBuffer(
            ctypes.c_void_p(self.pointer), self.size_bytes, self.memory_space, 0
        )

    def as_output_buffer(self) -> library.Buffer:
        """Return the writable ``gpuxtb_buffer_t`` for this ``out=`` view."""
        return library.Buffer(
            ctypes.c_void_p(self.pointer), self.size_bytes, self.memory_space, 0
        )


def _release_view(view: DLPackView) -> Exception | None:
    """Release one view, returning any deleter failure instead of raising."""
    try:
        view.release()
    except Exception as exc:  # noqa: BLE001 - a failed deleter must not prevent
        # the remaining views from being released
        return exc
    return None


def release_all(views: list[DLPackView]) -> None:
    """Release every committed view, tolerating individual failures."""
    failures = [error for error in (_release_view(view) for view in views) if error]
    if failures:
        raise GPUxtbRuntimeError(
            f"failed to release a DLPack view: {failures[0]}"
        ) from failures[0]


def dlpack_device(producer: object) -> tuple[int, int]:
    """Return the ``(device_type, device_id)`` pair of a DLPack producer.

    The probe calls only ``__dlpack_device__``; it never evaluates a lazy
    object or touches the raw data.
    """
    if not hasattr(producer, "__dlpack_device__"):
        raise GPUxtbValueError(
            "expected an Array API array implementing __dlpack__/__dlpack_device__; "
            f"got {type(producer).__name__}"
        )
    try:
        device = producer.__dlpack_device__()
    except Exception as exc:
        raise GPUxtbValueError(
            f"__dlpack_device__() failed for {type(producer).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(device, tuple)
        or len(device) != 2
        or not all(isinstance(value, int) for value in device)
    ):
        raise GPUxtbValueError(
            "__dlpack_device__() must return a (device_type, device_id) tuple of "
            f"integers, got {device!r}"
        )
    device_type, device_id = device
    if device_type == _DLPACK_DEVICE_CPU and device_id != 0:
        raise GPUxtbValueError(
            f"CPU DLPack producers must report device_id 0, got {device_id}"
        )
    return int(device_type), int(device_id)


def memory_space_for_device(device_type: int) -> int | None:
    """Map a DLPack device kind to a ``gpuxtb`` memory-space tag (or ``None``).

    ``None`` means the device kind is reserved by gpuxtb (ROCm) or otherwise
    unsupported by the native execution path (CUDA-managed memory is rejected
    because gpuxtb deliberately uses ``CudaManagedMemoryPolicy::kReject``).
    """
    if device_type in (_DLPACK_DEVICE_CPU, _DLPACK_DEVICE_CUDA_HOST):
        return library.MEMORY_HOST
    if device_type == _DLPACK_DEVICE_CUDA:
        return library.MEMORY_CUDA_DEVICE
    return None


def consume_from_dlpack(
    producer: object,
    *,
    expected_dtype: np.dtype,
    expected_shape: Sequence[int],
    stream: int | None,
    copy: bool = False,
    writable_required: bool = False,
    writable_hint: bool | None = None,
) -> DLPackView:
    """Produce a validated, committed ``DLPackView`` for one array.

    Parameters
    ----------
    producer
        The array object implementing ``__dlpack__``/``__dlpack_device__``.
    expected_dtype
        The exact numpy dtype the C ABI requires for this descriptor.
    expected_shape
        The exact logical shape the C ABI requires for this descriptor.
    stream
        The gpuxtb context's native CUDA stream pointer, or ``None`` for the
        legacy default stream.  Only meaningful for CUDA producers.
    copy
        When ``True`` the producer may create a contiguous copy; otherwise a
        non-contiguous/unaligned view raises :class:`BufferError` instead of
        silently copying.
    writable_required
        When ``True`` the view must be writable (used for ``out=`` buffers);
        read-only producers raise :class:`BufferError`.
    writable_hint
        Backend-neutral mutability result supplied by the array adapter. This
        is used only for legacy capsules, which predate DLPack's read-only
        flag; versioned capsule flags remain authoritative.

    Returns
    -------
    DLPackView
        A committed view whose :meth:`DLPackView.release` must be called
        exactly once by the caller after the synchronous compute call.

    Notes
    -----
    The capsule is only renamed and disarmed at the very end of this function.
    Any validation failure before that point leaves the capsule untouched so
    its own destructor cleanly releases the producer when it is garbage
    collected.
    """
    export_producer = producer
    copied_by_consumer = False
    if copy and isinstance(producer, np.ndarray):
        # NumPy releases predating the DLPack 1.0 keyword surface cannot honor
        # ``copy=True`` themselves. NumPy is already a required gpuxtb runtime
        # dependency, so provide the compact copy locally while preserving the
        # producer dtype; exact-dtype validation below remains authoritative.
        export_producer = np.array(producer, order="C", copy=True)
        copied_by_consumer = True

    device_type, device_id = dlpack_device(export_producer)
    memory_space = memory_space_for_device(device_type)
    if memory_space is None:
        raise GPUxtbNotSupportedError(
            f"DLPack device type {device_type} is not supported by gpuxtb "
            "(only CPU, CPU-pinned host memory, and CUDA device memory are usable)"
        )

    stream_value = _stream_argument(stream, device_type)
    capsule = _produce_capsule(
        export_producer, stream_value, copy and not copied_by_consumer
    )
    versioned = bool(_pyapi.PyCapsule_IsValid(capsule, _CAPSULE_NAME_VERSIONED))
    legacy = bool(_pyapi.PyCapsule_IsValid(capsule, _CAPSULE_NAME))
    if not versioned and not legacy:
        raise BufferError(
            f"{type(export_producer).__name__} returned a capsule that is neither "
            "dltensor_versioned nor dltensor"
        )

    try:
        managed_pointer = _pyapi.PyCapsule_GetPointer(
            capsule, _CAPSULE_NAME_VERSIONED if versioned else _CAPSULE_NAME
        )
    except ValueError:
        raise BufferError(
            f"{type(export_producer).__name__} returned a malformed DLPack capsule"
        ) from None
    if not managed_pointer:
        raise BufferError(
            f"{type(export_producer).__name__} returned a NULL DLPack managed tensor"
        )

    parsed = _parse_managed_tensor(
        managed_pointer, versioned, expected_ndim=len(expected_shape)
    )
    if versioned and parsed.version_major != 1:
        raise BufferError(
            f"{type(export_producer).__name__} exported DLPack major version "
            f"{parsed.version_major}, which this consumer cannot read"
        )
    if parsed.lanes != 1:
        raise GPUxtbValueError(
            f"DLPack vector lanes {parsed.lanes} are not supported; "
            "gpuxtb requires lane-1 (scalar) descriptors"
        )
    if (parsed.device_type, parsed.device_id) != (device_type, device_id):
        raise BufferError(
            f"{type(export_producer).__name__} reported DLPack device "
            f"({device_type}, {device_id}) but exported capsule device "
            f"({parsed.device_type}, {parsed.device_id})"
        )
    if parsed.is_copy and not copy:
        raise BufferError(
            f"{type(export_producer).__name__} returned a copied DLPack tensor "
            "when copy=False; a borrowed zero-copy view is required"
        )

    _validated_dtype(parsed, expected_dtype)
    pointer = _validated_pointer(parsed, expected_shape)
    size_bytes = _validated_extent(parsed, pointer, expected_shape, expected_dtype)
    if size_bytes == 0:
        # Empty logical vectors use the ABI's null-buffer convention even when
        # the producer's data pointer is a valid zero-sized allocation.
        pointer = 0
    _validated_layout(parsed, size_bytes, expected_dtype, copy)
    if writable_required:
        _validated_writable(parsed, export_producer, writable_hint)

    writable = parsed.writable or (parsed.version_major == 0 and writable_hint is True)

    if versioned:
        _pyapi.PyCapsule_SetName(capsule, _USED_CAPSULE_NAME_VERSIONED)
    else:
        _pyapi.PyCapsule_SetName(capsule, _USED_CAPSULE_NAME)
    _pyapi.PyCapsule_SetDestructor(capsule, None)

    deleter: Callable[[int], None] | None = None
    deleter_address = parsed.deleter
    if deleter_address:
        deleter = _pyapi_void_deleter(deleter_address)

    return DLPackView(
        pointer=pointer,
        size_bytes=size_bytes,
        memory_space=memory_space,
        device_type=device_type,
        device_id=device_id,
        writable=writable,
        is_copy=parsed.is_copy or copied_by_consumer,
        shape=tuple(int(value) for value in expected_shape),
        _producer=export_producer,
        _capsule=capsule,
        _deleter=deleter,
        _managed_pointer=int(managed_pointer),
    )


# --- internals -----------------------------------------------------------------


@dataclass
class _ParsedTensor:
    """Validated raw fields of one consumed ``DLManagedTensor``."""

    data: int
    device_type: int
    device_id: int
    ndim: int
    code: int
    bits: int
    lanes: int
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    byte_offset: int
    writable: bool
    is_copy: bool
    version_major: int
    deleter: int


def _stream_argument(stream: int | None, device_type: int) -> int | None:
    """Translate the gpuxtb context stream into the DLPack stream argument.

    Only CUDA producers receive a stream: host memory has no ordering
    contract.  ``None`` (native legacy default stream) maps to DLPack value
    ``1``; a configured ``CUstream`` passes its raw handle.
    """
    if device_type not in (_DLPACK_DEVICE_CUDA, _DLPACK_DEVICE_CUDA_MANAGED):
        return None
    if stream is None:
        return _DLPACK_LEGACY_STREAM
    return int(stream)


def _produce_capsule(producer: object, stream_value: int | None, copy: bool) -> object:
    """Call ``__dlpack__`` and return the producer's capsule.

    DLPack 1.0 is negotiated with ``max_version=(1, 0)`` and an explicit copy
    policy. A legacy producer is retried without unsupported keywords while
    retaining the CUDA consumer stream; versioned copied flags are rejected
    whenever the caller requires a borrowed zero-copy view.
    """
    full_kwargs: dict[str, object] = {
        "max_version": tuple(_DLPACK_MAX_VERSION),
        # ``None`` permits a producer copy. Always state the caller's policy so
        # compliant modern producers can enforce zero-copy before exporting.
        "copy": copy,
    }
    if stream_value is not None:
        full_kwargs["stream"] = stream_value
    try:
        return producer.__dlpack__(**full_kwargs)
    except TypeError:
        # A legacy producer may reject max_version, but CUDA synchronization is
        # still mandatory. Retry without only that keyword so the consumer
        # stream (and any explicit copy request) cannot be silently discarded.
        legacy_kwargs = dict(full_kwargs)
        legacy_kwargs.pop("max_version")
        try:
            return producer.__dlpack__(**legacy_kwargs)
        except TypeError:
            if copy:
                raise GPUxtbNotSupportedError(
                    f"{type(producer).__name__}.__dlpack__ does not accept a copy "
                    "request; supply an array matching the required dtype and layout "
                    "instead"
                ) from None
            # Pre-copy-keyword producers cannot negotiate ``copy=False``. The
            # legacy protocol historically exported borrowed views, so retry
            # without only that keyword while retaining CUDA synchronization.
            no_copy_kwargs = dict(legacy_kwargs)
            no_copy_kwargs.pop("copy")
            try:
                return producer.__dlpack__(**no_copy_kwargs)
            except TypeError as no_copy_exc:
                exc = no_copy_exc
            if stream_value is not None:
                raise BufferError(
                    f"{type(producer).__name__}.__dlpack__ does not accept the "
                    "consumer CUDA stream; a safe zero-copy transfer cannot be made"
                ) from exc
            raise BufferError(
                f"{type(producer).__name__}.__dlpack__ is not callable without "
                f"arguments and did not accept max_version: {exc}"
            ) from exc


def _parse_managed_tensor(
    managed_pointer: int, versioned: bool, expected_ndim: int
) -> _ParsedTensor:
    """Copy and structurally validate one managed tensor from raw memory."""
    try:
        if versioned:
            struct = ctypes.cast(
                managed_pointer, ctypes.POINTER(_DLManagedTensorVersioned)
            ).contents
            tensor = struct.dl_tensor
            version_major = int(struct.version_major)
            writable = not bool(struct.flags & _DLPACK_FLAG_READ_ONLY)
            is_copy = bool(struct.flags & _DLPACK_FLAG_IS_COPIED)
            deleter = int(struct.deleter or 0)
        else:
            struct = ctypes.cast(
                managed_pointer, ctypes.POINTER(_DLManagedTensor)
            ).contents
            tensor = struct.dl_tensor
            # Legacy capsules cannot carry read-only/read-write metadata; NumPy
            # treats them as read-only, so mutable-output requests reject them.
            version_major = 0
            writable = False
            is_copy = False
            deleter = int(struct.deleter or 0)
    except (ValueError, OSError) as exc:
        raise BufferError(f"could not read the DLPack managed tensor: {exc}") from exc

    device_type = int(tensor.device.device_type)
    device_id = int(tensor.device.device_id)
    ndim = int(tensor.ndim)
    if ndim < 0:
        raise BufferError(f"DLPack tensor has a negative ndim ({ndim})")
    if ndim != expected_ndim:
        raise GPUxtbValueError(
            f"DLPack ndim {ndim} does not match the expected ndim {expected_ndim}"
        )
    if tensor.shape and ndim:
        shape = tuple(int(tensor.shape[index]) for index in range(ndim))
    else:
        shape = ()
    if any(value < 0 for value in shape):
        raise BufferError(f"DLPack tensor has a negative shape {shape}")
    if tensor.strides and ndim:
        strides = tuple(int(tensor.strides[index]) for index in range(ndim))
    else:
        strides = None
    return _ParsedTensor(
        data=int(tensor.data or 0),
        device_type=device_type,
        device_id=device_id,
        ndim=ndim,
        code=int(tensor.dtype.code),
        bits=int(tensor.dtype.bits),
        lanes=int(tensor.dtype.lanes),
        shape=shape,
        strides=strides,
        byte_offset=int(tensor.byte_offset),
        writable=writable,
        is_copy=is_copy,
        version_major=version_major,
        deleter=deleter,
    )


def _validated_dtype(parsed: _ParsedTensor, expected_dtype: np.dtype) -> None:
    """Require the exact scalar dtype (code/bits) the C ABI expects."""
    expected = np.dtype(expected_dtype)
    try:
        expected_code, expected_bits = _NP_DTYPE_TO_DLPACK[expected]
    except KeyError:
        raise GPUxtbValueError(
            f"nothing maps numpy dtype {expected} to a DLPack scalar dtype"
        ) from None
    if parsed.code != expected_code or parsed.bits != expected_bits:
        raise GPUxtbValueError(
            f"DLPack dtype ({parsed.code}, {parsed.bits} bits) does not match "
            f"the required {expected} ({expected_code}, {expected_bits} bits); "
            f"provide an array with dtype {expected}"
        )


def _validated_pointer(parsed: _ParsedTensor, expected_shape: Sequence[int]) -> int:
    """Return the absolute element pointer after applying ``byte_offset``."""
    if len(parsed.shape) != len(expected_shape):
        raise GPUxtbValueError(
            f"DLPack ndim {len(parsed.shape)} does not match the expected "
            f"ndim {len(expected_shape)}"
        )
    if parsed.shape != tuple(int(value) for value in expected_shape):
        raise GPUxtbValueError(
            f"DLPack shape {parsed.shape} does not match the required shape "
            f"{tuple(expected_shape)}"
        )
    base = parsed.data
    offset = parsed.byte_offset
    if base == 0:
        if _dlpack_bytes(expected_shape, 1) != 0:
            raise BufferError("DLPack tensor has a NULL data pointer")
        return 0
    if offset > _POINTER_MAX - base:
        raise BufferError("DLPack data pointer plus byte_offset overflows uintptr_t")
    pointer = base + offset
    return pointer


def _validated_extent(
    parsed: _ParsedTensor,
    pointer: int,
    expected_shape: Sequence[int],
    expected_dtype: np.dtype,
) -> int:
    """Compute the byte extent of the logical tensor with overflow checks."""
    itemsize = _itemsize(parsed)
    size_bytes = _dlpack_bytes(expected_shape, itemsize)
    if size_bytes == 0:
        return 0
    alignment = np.dtype(expected_dtype).alignment
    if pointer and pointer % alignment != 0:
        raise BufferError(
            f"DLPack data pointer 0x{pointer:x} is not aligned to "
            f"{alignment} bytes for {np.dtype(expected_dtype)}"
        )
    if size_bytes > _POINTER_MAX:
        raise BufferError(f"DLPack logical extent {size_bytes} bytes exceeds size_t")
    if pointer and size_bytes > _POINTER_MAX - pointer:
        raise BufferError("DLPack data pointer plus logical extent overflows uintptr_t")
    return size_bytes


def _itemsize(parsed: _ParsedTensor) -> int:
    """Byte width of one element of the DLPack scalar type."""
    if parsed.bits % 8 != 0:
        raise GPUxtbValueError(
            f"DLPack dtype uses {parsed.bits} bits, which is not byte-aligned"
        )
    return (parsed.bits // 8) * parsed.lanes


def _validated_layout(
    parsed: _ParsedTensor,
    size_bytes: int,
    expected_dtype: np.dtype,
    copy: bool,
) -> None:
    """Require a compact C-contiguous view unless a copy was requested.

    ``strides == NULL`` means compact row-major by spec.  Explicit strides
    must match C-contiguous element strides exactly; negative or overlapping
    strides are rejected.  Zero-size vectors need no strides.
    """
    if size_bytes == 0:
        return
    if parsed.strides is None:
        return
    expected: list[int] = []
    cumulative = 1
    for extent in reversed(parsed.shape):
        expected.append(cumulative)
        cumulative *= extent
    expected.reverse()
    if parsed.strides != tuple(expected):
        if copy:
            raise GPUxtbNotSupportedError(
                "a copy was requested but the producer returned a "
                "non-contiguous tensor; supply an already-contiguous array"
            )
        raise BufferError(
            f"DLPack tensor with strides {parsed.strides} is not C-contiguous; "
            "pass an already-contiguous array or request copy=True"
        )


def _validated_writable(
    parsed: _ParsedTensor, producer: object, writable_hint: bool | None
) -> None:
    """Reject read-only buffers requested as mutable ``out=`` targets.

    Legacy managed tensors have no flags field. They are accepted only when
    the backend-neutral adapter already established that the source array is
    writable; versioned DLPack flags never use that compatibility hint.
    """
    legacy_writable = parsed.version_major == 0 and writable_hint is True
    if not parsed.writable and not legacy_writable:
        raise BufferError(
            f"{type(producer).__name__} exported a read-only DLPack tensor and "
            "cannot be used as a mutable output buffer"
        )


def _pyapi_void_deleter(address: int) -> Callable[[int], None]:
    """Wrap a producer deleter function pointer so it can be called once."""
    prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    @prototype
    def deleter(managed_pointer: int) -> None:  # pragma: no cover - trivial bridge
        """Forward the recorded address through the raw DLPack signature."""
        ctypes.cast(address, prototype)(managed_pointer)

    return deleter


# --- gpuxtb-owned result producer --------------------------------------------


class _ResultArena:
    """Internal native arena wrapper shared by result-slice producers.

    The native arena is ref-counted.  ``create`` leaves one producer reference
    owned by this object, released exactly once when the arena is closed or
    garbage-collected.  Every :class:`DLPackResultBuffer` retains one
    additional reference on construction and releases its own reference on
    close/GC; each exported capsule retains independently through its native
    managed-tensor deleter.  The allocation is freed by the native side
    exactly when the last reference (producer, buffers, or capsules) drops, so
    it can never be freed while an imported array can still observe it.
    """

    __slots__ = ("__weakref__", "_finalizer", "_handle")

    def __init__(self, handle: object) -> None:
        self._handle = handle
        self._finalizer = weakref.finalize(
            self, _release_result_owner, library.load_library(), handle
        )

    @property
    def handle(self) -> object:
        """The native ``gpuxtb_result_owner_t`` handle (kept for exports)."""
        return self._handle

    def base_pointer(self) -> int:
        """Return the arena base data pointer for slice arithmetic."""
        buffer = library.Buffer()
        library._check_init(
            "gpuxtb_result_owner_buffer",
            library.load_library().gpuxtb_result_owner_buffer(
                self._handle, ctypes.byref(buffer)
            ),
        )
        return int(buffer.data or 0)

    def retain(self) -> None:
        """Retain one arena reference for one producer slice."""
        library.load_library().gpuxtb_result_owner_retain(self._handle)

    def release(self) -> None:
        """Release one arena reference held by one producer slice."""
        library.load_library().gpuxtb_result_owner_release(self._handle)

    def close(self) -> None:
        """Release the arena's own producer reference (idempotent)."""
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer()

    def __del__(self) -> None:
        """Best-effort native cleanup on garbage collection."""
        with contextlib.suppress(Exception):
            self.close()


def _release_result_owner(library_instance: object, handle: object) -> None:
    """Release the arena producer reference (the finalizer target)."""
    library_instance.gpuxtb_result_owner_release(handle)


@dataclass
class DLPackResultBuffer:
    """A DLPack producer over one gpuxtb-owned result arena.

    Wraps a native result arena and one compact C-contiguous slice of it, so
    ``cupy.from_dlpack``, ``torch.from_dlpack``, ``jax.dlpack.from_dlpack``,
    or a NumPy-conformant ``from_dlpack`` can import the finished result
    without a host copy.

    Lifecycle contract
    ------------------
    * The arena is ref-counted natively and outlives the compute context that
      filled it.  Each exported capsule independently retains the arena; the
      managed-tensor deleter is a plain native function that importing
      frameworks may call from any thread after this Python wrapper is gone.
    * :meth:`close` (aliased as :meth:`delete`) releases this producer's
      reference and is idempotent; exports after close raise
      :class:`GPUxtbValueError`.
    * Repeated exports are supported: every :meth:`__dlpack__` call creates a
      fresh single-use capsule that retains the shared arena, exactly as the
      DLPack 1.0 Python protocol requires.
    """

    arena: _ResultArena
    byte_offset: int
    size_bytes: int
    shape: tuple[int, ...]
    dtype: np.dtype
    memory_space: int
    device_id: int
    stream: int | None
    _closed: bool = False

    def __post_init__(self) -> None:
        """Retain the shared arena for the lifetime of this slice."""
        self.arena.retain()

    @property
    def device_type(self) -> int:
        """The DLPack device kind this slice lives on."""
        if self.memory_space == library.MEMORY_CUDA_DEVICE:
            return _DLPACK_DEVICE_CUDA
        return _DLPACK_DEVICE_CPU

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the ``(device_type, device_id)`` pair of this slice."""
        self._ensure_open()
        return (self.device_type, int(self.device_id))

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        """Report a namespace so Array-API probing sees a concrete eager array."""
        self._ensure_open()
        return np

    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        copy: bool | None = None,
        dl_device: tuple[int, int] | None = None,
    ) -> object:
        """Export one gpuxtb-owned slice as a ``dltensor`` PyCapsule.

        ``max_version >= (1, 0)`` negotiates the versioned capsule; otherwise a
        legacy ``dltensor`` capsule is produced.  ``copy`` never causes a copy:
        the slice is already compact C-contiguous, and returning the same
        allocation for ``copy=True`` is explicitly permitted by the DLPack
        specification (the copied flag is therefore never set).  A foreign
        ``dl_device`` is rejected before any capsule is created.
        """
        self._ensure_open()
        if dl_device is not None and (
            not isinstance(dl_device, tuple)
            or len(dl_device) != 2
            or tuple(int(value) for value in dl_device) != self.__dlpack_device__()
        ):
            raise BufferError(
                f"producer device {self.__dlpack_device__()} does not match the "
                f"requested dl_device {dl_device!r}"
            )
        versioned = max_version is not None and tuple(max_version) >= (1, 0)
        managed_pointer = _export_native_slice(self, versioned)
        capsule_name = _CAPSULE_NAME_VERSIONED if versioned else _CAPSULE_NAME
        destructor = _PRODUCER_CAPSULE_DESTRUCTORS[1 if versioned else 0]
        capsule = _pyapi.PyCapsule_New(
            ctypes.c_void_p(managed_pointer), capsule_name, destructor
        )
        return capsule

    def close(self) -> None:
        """Release this producer's arena reference (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.arena.release()

    delete = close

    def __del__(self) -> None:
        """Best-effort native cleanup when the wrapper is garbage-collected."""
        with contextlib.suppress(Exception):
            self.close()

    def __enter__(self) -> DLPackResultBuffer:  # noqa: PYI034 - 3.10 lacks Self
        """Use as a context manager so the arena reference is released."""
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Release the arena reference when leaving a ``with`` block."""
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise GPUxtbValueError(
                "this gpuxtb-owned result buffer is closed and cannot be used"
            )


def _export_native_slice(buffer: DLPackResultBuffer, versioned: bool) -> int:
    """Call ``gpuxtb_result_owner_export_dltensor`` for one slice."""
    library_instance = library.load_library()
    view = library.DlpackView()
    code, bits = _producer_dtype_fields(buffer.dtype)
    ndim = len(buffer.shape)
    if ndim > library.DLPACK_MAX_NDIM:
        raise GPUxtbValueError(
            f"cannot export a {ndim}-dimensional result slice; the DLPack "
            f"producer supports at most {library.DLPACK_MAX_NDIM} dimensions"
        )
    view.struct_size = ctypes.sizeof(library.DlpackView)
    view.api_version = library.API_VERSION
    view.byte_offset = int(buffer.byte_offset)
    view.dtype_code = code
    view.dtype_bits = bits
    view.dtype_lanes = 1
    view.ndim = ndim
    view.reserved = 0
    shape_storage: list[np.ndarray] = []
    if ndim:
        owner = np.ascontiguousarray(np.asarray(buffer.shape, dtype=np.int64))
        shape_storage.append(owner)
        view.shape = ctypes.cast(owner.ctypes.data, ctypes.POINTER(ctypes.c_int64))
    else:
        view.shape = None
    managed_pointer = ctypes.c_void_p()
    status = library_instance.gpuxtb_result_owner_export_dltensor(
        buffer.arena.handle,
        ctypes.byref(view),
        int(versioned),
        ctypes.byref(managed_pointer),
    )
    if status != library.STATUS_SUCCESS:
        raise GPUxtbRuntimeError(
            "gpuxtb_result_owner_export_dltensor failed with "
            f"{library.status_string(status)}: {library.get_last_error()}",
            status,
        )
    return int(managed_pointer.value or managed_pointer)


def _producer_dtype_fields(dtype: np.dtype) -> tuple[int, int]:
    """Map a numpy dtype to its (DLPack code, bits) pair for export."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
        return library.DLPACK_DTYPE_FLOAT, 64
    if dtype == np.dtype(np.float32):
        return library.DLPACK_DTYPE_FLOAT, 32
    if dtype == np.dtype(np.int64):
        return library.DLPACK_DTYPE_INT, 64
    if dtype == np.dtype(np.int32):
        return library.DLPACK_DTYPE_INT, 32
    if dtype == np.dtype(np.int16):
        return library.DLPACK_DTYPE_INT, 16
    if dtype == np.dtype(np.int8):
        return library.DLPACK_DTYPE_INT, 8
    if dtype == np.dtype(np.uint8):
        return library.DLPACK_DTYPE_UINT, 8
    raise GPUxtbValueError(
        f"cannot export numpy dtype {dtype} through the gpuxtb DLPack producer"
    )


@dataclass
class _ProducerCapsuleDestructor:
    """CPython capsule destructor forwarding to a native managed-tensor deleter.

    The destructor is only invoked by CPython when a producer capsule is
    garbage-collected *without* having been consumed (a consumer renames the
    capsule to ``used_dltensor[_versioned]`` and detaches this destructor, then
    invokes the managed tensor's own native deleter).  It therefore forwards to
    the exact same native deleter, so an unconsumed capsule never leaks and no
    allocation is ever freed twice.

    The callback signature is ``void (*)(PyObject*)`` but the capsule is passed
    as a raw address and every PyCapsule_* helper is declared with
    ``c_void_p``/``c_char_p`` arguments: creating a ``py_object`` inside
    ``capsule_dealloc`` would bump the dying capsule's refcount and
    recursively re-enter deallocation.
    """

    versioned: bool

    def __post_init__(self) -> None:
        self._callback = self._build()

    def __call__(self, capsule_pointer: int) -> None:
        try:
            name = _pyapi_raw.capsule_name(capsule_pointer)
        except (ValueError, TypeError):
            return
        expected = _CAPSULE_NAME_VERSIONED if self.versioned else _CAPSULE_NAME
        if name not in (expected,):
            # Already consumed and renamed: the importing framework now owns
            # the managed tensor and will invoke its native deleter itself.
            return
        pointer = _pyapi_raw.capsule_pointer(capsule_pointer, name)
        if not pointer:
            return
        try:
            if self.versioned:
                struct = ctypes.cast(
                    pointer, ctypes.POINTER(_DLManagedTensorVersioned)
                ).contents
                deleter = int(struct.deleter or 0)
            else:
                struct = ctypes.cast(pointer, ctypes.POINTER(_DLManagedTensor)).contents
                deleter = int(struct.deleter or 0)
        except (ValueError, OSError):
            return
        if deleter:
            _pyapi_void_deleter(deleter)(pointer)

    def _build(self) -> object:
        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)  # type: ignore[arg-type]
        def destructor(capsule_pointer: int) -> None:
            self(capsule_pointer)

        return destructor


def _producer_capsule_destructors() -> list[object]:
    """One persistent CPython capsule destructor per capsule flavor."""
    legacy = _ProducerCapsuleDestructor(versioned=False)
    versioned = _ProducerCapsuleDestructor(versioned=True)
    return [legacy._callback, versioned._callback]


# Kept alive for the process lifetime: a produced capsule may outlive every
# Python reference, and a dangling capsule destructor would crash CPython GC.
_PRODUCER_CAPSULE_DESTRUCTORS = _producer_capsule_destructors()


class _RawPyApi:
    """Raw-address capsule access used only inside the producer destructor.

    Accessing a dying capsule through ``ctypes.py_object`` would increment its
    refcount during ``capsule_dealloc`` and recursively re-enter deallocation.
    Passing the raw ``PyObject*`` as ``c_void_p`` and reading with
    ``c_char_p`` avoids every refcount mutation.  Own CFUNCTYPE prototypes are
    used so the shared ``ctypes.pythonapi`` attribute signatures used by the
    consumer bridge are never mutated.
    """

    def __init__(self) -> None:
        api = ctypes.pythonapi
        get_name_address = ctypes.cast(api.PyCapsule_GetName, ctypes.c_void_p).value
        get_pointer_address = ctypes.cast(
            api.PyCapsule_GetPointer, ctypes.c_void_p
        ).value
        # Build dedicated function-pointer prototypes from the raw C addresses
        # so the consumer-side py_object signatures on pythonapi are never
        # consulted (and never mutated) for these destructive accesses.
        self._get_name = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(
            get_name_address
        )
        self._get_pointer = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
        )(get_pointer_address)
        self._callbacks: list[object] = [self._get_name, self._get_pointer]

    def capsule_name(self, capsule_pointer: int) -> bytes | None:
        """Return the capsule's name, or None when it cannot be read."""
        if not capsule_pointer:
            return None
        try:
            return self._get_name(ctypes.c_void_p(capsule_pointer))
        except (ValueError, TypeError):
            return None

    def capsule_pointer(self, capsule_pointer: int, name: bytes) -> int:
        """Return the payload pointer of a raw capsule address."""
        try:
            return int(self._get_pointer(ctypes.c_void_p(capsule_pointer), name) or 0)
        except (ValueError, TypeError):
            return 0


_pyapi_raw = _RawPyApi()
