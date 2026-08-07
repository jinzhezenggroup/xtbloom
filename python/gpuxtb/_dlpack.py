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

import ctypes
import math
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
