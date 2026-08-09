"""Deterministic DLPack producer fakes for the Python test suite.

The production DLPack consumer (:mod:`xtbloom._dlpack`) is tested against real
NumPy arrays for the happy path and against these handcrafted producers for
the protocol edge cases (legacy capsules, read-only flags, producer
exceptions, stream/copy keyword negotiation, and capsule-lifetime accounting)
that cannot be produced with NumPy alone.  No third-party array library is
required to import or run the fakes.

Every fake records deleter invocations in the module-level ``DELETED``
registry so tests can assert the "deleter runs exactly once, including on
error paths" invariant.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import xtbloom._dlpack as dlpack

# --- raw DLPack 1.0 struct mirrors (same layout as xtbloom._dlpack) -------------


class _DLDataType(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLDevice(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLTensor(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("version_major", ctypes.c_uint32),
        ("version_minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def _setup_pyapi() -> None:
    """Declare the CPython capsule/malloc ABI exactly once for the fakes."""
    api = ctypes.pythonapi
    if not getattr(_setup_pyapi, "configured", False):
        api.PyCapsule_New.restype = ctypes.py_object
        api.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        api.PyMem_Malloc.restype = ctypes.c_void_p
        api.PyMem_Malloc.argtypes = [ctypes.c_size_t]
        api.PyMem_Free.restype = None
        api.PyMem_Free.argtypes = [ctypes.c_void_p]
        _setup_pyapi.configured = True


_setup_pyapi()

# Unique producer id -> deleter invocation count.
DELETED: dict[int, int] = {}

_CALLBACKS: list[Callable[[int], None]] = []


def _deleter_for(producer_id: int) -> Callable[[int], None]:
    """Build (and keep alive) the C deleter callback for one producer."""

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def deleter(managed_pointer: int) -> None:
        DELETED[producer_id] = DELETED.get(producer_id, 0) + 1
        ctypes.pythonapi.PyMem_Free(managed_pointer)

    _CALLBACKS.append(deleter)
    return deleter


def _dtype_fields(dtype: np.dtype) -> tuple[int, int]:
    """Map a numpy scalar dtype to its (DLPack code, bits) pair."""
    mapping = {
        np.dtype(np.int64): (0, 64),
        np.dtype(np.int32): (0, 32),
        np.dtype(np.float64): (2, 64),
        np.dtype(np.uint8): (1, 8),
    }
    return mapping[np.dtype(dtype)]


class FakeArray:
    """A DLPack producer wrapping a caller-owned numpy buffer.

    Parameters
    ----------
    data
        The backing numpy array (any dtype/layout).
    device
        Device kind reported by ``__dlpack_device__`` (``1`` CPU, ``2`` CUDA).
    device_id
        Ordinal reported alongside ``device``.
    capsule_device, capsule_device_id
        Optional device metadata embedded in the exported capsule. Defaults to
        the values reported by ``__dlpack_device__``; differing values model a
        malformed producer contract.
    capsule_ndim
        Optional rank embedded in the capsule, for malformed-rank tests.
    versioned
        Produce the DLPack 1.0 ``dltensor_versioned`` capsule (default) or the
        legacy ``dltensor`` capsule.
    readonly
        Set the versioned read-only flag (ignored for legacy capsules).
    copied
        Set the versioned copied-tensor flag (ignored for legacy capsules).
    stream_support
        Whether ``__dlpack__`` accepts the ``stream`` keyword.
    max_version_support
        Whether ``__dlpack__`` accepts the ``max_version`` keyword.
    copy_support
        Whether ``__dlpack__`` accepts (and honors) the ``copy`` keyword.
    stream_record
        Optional list that captures every ``stream`` value the consumer
        passes to ``__dlpack__``.
    byte_offset
        Payload byte offset reported by the managed tensor.
    force_error
        Optional exception raised by ``__dlpack_device__``/``__dlpack__``.
    major_version
        ``DLPackVersion.major`` reported in a versioned capsule (default 1).
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        device: int = 1,
        device_id: int = 0,
        capsule_device: int | None = None,
        capsule_device_id: int | None = None,
        capsule_ndim: int | None = None,
        versioned: bool = True,
        readonly: bool = False,
        copied: bool = False,
        stream_support: bool = True,
        max_version_support: bool = True,
        copy_support: bool = True,
        stream_record: list[object] | None = None,
        byte_offset: int = 0,
        force_error: Exception | None = None,
        major_version: int = 1,
    ) -> None:
        self._data = data
        self._device = (int(device), int(device_id))
        self._capsule_device = (
            int(device if capsule_device is None else capsule_device),
            int(device_id if capsule_device_id is None else capsule_device_id),
        )
        self._capsule_ndim = capsule_ndim
        self._versioned = versioned
        self._readonly = readonly
        self._copied = copied
        self._stream_support = stream_support
        self._max_version_support = max_version_support
        self._copy_support = copy_support
        self._stream_record = stream_record
        self._byte_offset = byte_offset
        self._force_error = force_error
        self._major_version = major_version
        self._id = id(self)
        self._deleter = _deleter_for(self._id)
        DELETED[self._id] = 0

    # --- Array-API-like metadata ------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """The logical shape of the wrapped buffer."""
        return tuple(int(value) for value in self._data.shape)

    @property
    def dtype(self) -> np.dtype:
        """The numpy dtype of the wrapped buffer."""
        return self._data.dtype

    # --- DLPack producer protocol -------------------------------------------------

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the fake ``(device_type, device_id)`` pair."""
        if self._force_error is not None:
            raise self._force_error
        return self._device

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        """Mark the fake as an eager writable Array API object for output tests."""
        return np

    def __dlpack__(self, **kwargs: object) -> object:
        """Build a fresh managed-tensor capsule for the wrapped buffer.

        Mirrors the Array API 2025.12 keyword surface (``stream``,
        ``max_version``, ``copy``) so the consumer's negotiation and fallback
        paths can be exercised deterministically.
        """
        if self._stream_record is not None:
            self._stream_record.append(kwargs.get("stream"))
        if not self._max_version_support and "max_version" in kwargs:
            raise TypeError(
                "__dlpack__() got an unexpected keyword argument 'max_version'"
            )
        if not self._copy_support and "copy" in kwargs:
            raise TypeError("__dlpack__() got an unexpected keyword argument 'copy'")
        if not self._stream_support and "stream" in kwargs:
            raise TypeError("__dlpack__() got an unexpected keyword argument 'stream'")

        data = self._data
        code, bits = _dtype_fields(data.dtype)
        ndim = data.ndim
        struct_class = (
            _DLManagedTensorVersioned if self._versioned else _DLManagedTensor
        )
        capsule_name = b"dltensor_versioned" if self._versioned else b"dltensor"
        pyt_exact = ctypes.pythonapi.PyMem_Malloc
        pointer = pyt_exact(ctypes.sizeof(struct_class))
        struct = ctypes.cast(pointer, ctypes.POINTER(struct_class)).contents
        if self._versioned:
            struct.version_major = self._major_version
            struct.version_minor = 0
            struct.flags = 0
            if self._readonly:
                struct.flags |= dlpack._DLPACK_FLAG_READ_ONLY
            if self._copied:
                struct.flags |= dlpack._DLPACK_FLAG_IS_COPIED
        tensor = struct.dl_tensor
        tensor.data = ctypes.c_void_p(data.ctypes.data)
        tensor.device.device_type = self._capsule_device[0]
        tensor.device.device_id = self._capsule_device[1]
        tensor.ndim = ndim if self._capsule_ndim is None else self._capsule_ndim
        tensor.dtype.code = code
        tensor.dtype.bits = bits
        tensor.dtype.lanes = 1
        if ndim:
            element_strides = tuple(s // data.itemsize for s in data.strides)
            shape = (ctypes.c_int64 * ndim)(*data.shape)
            strides = (ctypes.c_int64 * ndim)(*element_strides)
            tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
            tensor.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))
        else:
            tensor.shape = None
            tensor.strides = None
        tensor.byte_offset = self._byte_offset
        struct.manager_ctx = None
        struct.deleter = ctypes.cast(self._deleter, ctypes.c_void_p).value
        return ctypes.pythonapi.PyCapsule_New(pointer, capsule_name, None)

    # --- test bookkeeping ---------------------------------------------------------

    def deleted_count(self) -> int:
        """Return how many times this producer's deleter was invoked."""
        return DELETED.get(self._id, 0)
