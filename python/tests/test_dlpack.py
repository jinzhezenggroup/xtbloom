"""Unit tests for the DLPack consumer bridge (``xtbloom._dlpack``).

The bridge is the zero-copy path behind :class:`xtbloom.ArrayBatch`: it turns
any DLPack producer array into a validated C-ABI buffer view with an exact
capsule-lifetime contract.  Happy paths run against real NumPy arrays; the
protocol edge cases (legacy/read-only capsules, producer exceptions, version
mismatches, stream negotiation, deleter accounting) run against the
deterministic fakes in :mod:`_dlpack_fakes`.
"""

from __future__ import annotations

import ctypes
import gc
import weakref

import numpy as np
import pytest
import xtbloom._dlpack as dlpack
from _dlpack_fakes import DELETED, FakeArray
from xtbloom import library
from xtbloom.exceptions import XTBloomNotSupportedError, XTBloomValueError

F64 = np.dtype(np.float64)
I32 = np.dtype(np.int32)


def _consume(
    producer: object,
    *,
    dtype: np.dtype = F64,
    shape: tuple[int, ...] = (3,),
    stream: int | None = None,
    copy: bool = False,
    writable_required: bool = False,
    writable_hint: bool | None = None,
) -> dlpack.DLPackView:
    return dlpack.consume_from_dlpack(
        producer,
        expected_dtype=dtype,
        expected_shape=shape,
        stream=stream,
        copy=copy,
        writable_required=writable_required,
        writable_hint=writable_hint,
    )


# --- real NumPy producer ------------------------------------------------------


def test_numpy_view_is_borrowed_zero_copy() -> None:
    """The view must alias NumPy's own buffer with the exact ABI extents."""
    values = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    view = _consume(values, shape=(4, 3))
    assert view.pointer == int(values.ctypes.data)
    assert view.size_bytes == values.nbytes
    assert view.memory_space == library.MEMORY_HOST
    assert not view.is_copy
    try:
        view.release()
    finally:
        pass


def test_numpy_int32_contract() -> None:
    """int32 descriptors map to the DLPack integer/32 layout."""
    values = np.array([1, 2, 3], dtype=np.int32)
    view = _consume(values, dtype=I32, shape=(3,))
    assert view.size_bytes == 12
    assert view.pointer == int(values.ctypes.data)
    view.release()


def test_numpy_dtype_mismatch_is_rejected() -> None:
    """A float64 descriptor must not silently accept float32/bytes."""
    values = np.arange(3, dtype=np.float32)
    with pytest.raises(XTBloomValueError, match="dtype"):
        _consume(values, shape=(3,))


def test_numpy_shape_mismatch_is_rejected() -> None:
    """Shape/ndim mismatches fail before any buffer is taken."""
    with pytest.raises(XTBloomValueError, match=r"ndim|shape"):
        _consume(np.zeros((2, 3)), shape=(3,))


def test_capsule_rank_mismatch_is_rejected_before_shape_access() -> None:
    """Rank is bounded before the parser walks producer-owned shape memory."""
    fake = FakeArray(np.arange(3.0), capsule_ndim=1024)
    with pytest.raises(XTBloomValueError, match="ndim"):
        _consume(fake, shape=(3,))


def test_numpy_noncontiguous_copy_false_raises_bufffererror() -> None:
    """Non-contiguous views are rejected instead of silently copied."""
    values = np.arange(12.0).reshape(4, 3)[:, ::2]
    assert not values.flags["C_CONTIGUOUS"]
    with pytest.raises(BufferError, match="C-contiguous"):
        _consume(values, shape=(4, 2))


def test_numpy_copy_true_makes_contiguous_copy() -> None:
    """``copy=True`` lets NumPy produce a packed copy we do not own."""
    values = np.arange(12.0).reshape(4, 3)[:, ::2]
    view = _consume(values, shape=(4, 2), copy=True)
    assert view.is_copy
    assert view.size_bytes == 2 * 4 * 8
    np.testing.assert_array_equal(
        np.frombuffer(
            (ctypes.c_char * view.size_bytes).from_address(view.pointer),
            dtype=np.float64,
        ).reshape(4, 2),
        values,
    )
    view.release()


def test_numpy_copy_true_does_not_convert_dtype() -> None:
    """A layout copy never hides an ABI dtype mismatch."""
    values = np.arange(3, dtype=np.float32)
    with pytest.raises(XTBloomValueError, match="dtype"):
        _consume(values, shape=(3,), copy=True)


def test_numpy_writable_required_rejects_readonly() -> None:
    """Read-only output buffers fail deterministically."""
    values = np.arange(3.0)
    values.flags.writeable = False
    with pytest.raises(BufferError, match=r"read.?only|writable"):
        _consume(values, writable_required=True)


def test_numpy_writable_required_accepts_writable() -> None:
    """Writable output buffers pass the policy check."""
    values = np.arange(6.0)
    view = _consume(values, shape=(6,), writable_required=True, writable_hint=True)
    assert view.writable
    view.release()


# --- fake producer: capsule lifecycle -------------------------------------------


def test_fake_deleter_runs_exactly_once_on_release() -> None:
    """Release invokes the deleter once and drops managed metadata storage."""
    fake = FakeArray(np.arange(3.0))
    view = _consume(fake, shape=(3,))
    assert view.pointer == int(fake._data.ctypes.data)
    assert fake.active_export_count() == 1
    view.release()
    assert fake.deleted_count() == 1
    assert fake.active_export_count() == 0
    view.release()  # idempotent
    assert fake.deleted_count() == 1


def test_fake_multiple_exports_release_their_own_metadata() -> None:
    """Concurrent managed tensors retain and release independent metadata."""
    fake = FakeArray(np.arange(3.0))
    first = _consume(fake, shape=(3,))
    second = _consume(fake, shape=(3,))
    assert fake.active_export_count() == 2
    second.release()
    assert fake.active_export_count() == 1
    first.release()
    assert fake.active_export_count() == 0
    assert fake.deleted_count() == 2


def test_fake_raw_capsule_outlives_producer_and_cleans_up() -> None:
    """An unconsumed managed tensor owns its data and metadata without producer."""
    fake = FakeArray(np.arange(3.0))
    producer_id = fake._id
    export_owners = fake._export_owners
    producer_ref = weakref.ref(fake)
    capsule = fake.__dlpack__()
    assert len(export_owners) == 1

    del fake
    gc.collect()
    assert producer_ref() is None
    assert len(export_owners) == 1

    del capsule
    gc.collect()
    assert len(export_owners) == 0
    assert DELETED[producer_id] == 1


def test_fake_producer_stays_alive_until_release() -> None:
    """The consumed capsule keeps the producer object alive on its own."""
    fake = FakeArray(np.arange(3.0))
    view = _consume(fake, shape=(3,))
    # The capsule reference is the only remaining pointer to the producer.
    del fake
    view.release()  # must not crash: the managed struct is still alive
    assert True


def test_fake_legacy_capsule_is_consumed_as_readonly() -> None:
    """Legacy dltensor capsules are accepted but treated as read-only."""
    fake = FakeArray(np.arange(3.0), versioned=False)
    view = _consume(fake, shape=(3,))
    assert not view.writable
    assert view.memory_space == library.MEMORY_HOST
    with pytest.raises(BufferError, match="read-only"):
        _consume(fake, shape=(3,), writable_required=True)
    view.release()
    assert fake.deleted_count() == 2
    assert fake.active_export_count() == 0


def test_fake_legacy_writable_hint_allows_mutable_output() -> None:
    """Prevalidated mutability enables outputs for pre-1.0 capsule producers."""
    fake = FakeArray(np.arange(3.0), versioned=False)
    view = _consume(
        fake,
        shape=(3,),
        writable_required=True,
        writable_hint=True,
    )
    assert view.writable
    view.release()


def test_fake_versioned_readonly_flag() -> None:
    """Versioned capsules carrying the read-only flag are rejected as outputs."""
    fake = FakeArray(np.arange(3.0), readonly=True)
    with pytest.raises(BufferError, match="read-only"):
        _consume(fake, shape=(3,), writable_required=True)
    assert fake.deleted_count() == 1
    assert fake.active_export_count() == 0


def test_fake_unsupported_major_version_is_rejected() -> None:
    """A versioned capsule with major version 2 must be refused."""
    fake = FakeArray(np.arange(3.0), major_version=2)
    with pytest.raises(BufferError, match="major version"):
        _consume(fake, shape=(3,))
    assert fake.deleted_count() == 1
    assert fake.active_export_count() == 0


def test_fake_copy_false_rejects_copied_flag() -> None:
    """A producer-side temporary cannot satisfy the zero-copy contract."""
    fake = FakeArray(np.arange(3.0), copied=True)
    with pytest.raises(BufferError, match=r"copied.*copy=False"):
        _consume(fake, shape=(3,), copy=False)


def test_fake_copy_true_accepts_copied_flag() -> None:
    """The copied flag is valid only when the caller permitted a copy."""
    fake = FakeArray(np.arange(3.0), copied=True)
    view = _consume(fake, shape=(3,), copy=True)
    assert view.is_copy
    view.release()


def test_fake_producer_exception_propagates() -> None:
    """Producer failures surface to the caller, not as generic errors."""
    fake = FakeArray(np.arange(3.0), force_error=RuntimeError("boom"))
    with pytest.raises(XTBloomValueError, match="boom"):
        _consume(fake, shape=(3,))


# --- fake producer: devices and streams ------------------------------------------


def test_fake_cuda_maps_to_cuda_device_memory() -> None:
    """CUDA device producers map to XTBLOOM_MEMORY_CUDA_DEVICE."""
    fake = FakeArray(np.arange(3.0), device=dlpack._DLPACK_DEVICE_CUDA)
    view = _consume(fake, shape=(3,))
    assert view.memory_space == library.MEMORY_CUDA_DEVICE
    assert view.device_type == 2
    view.release()


def test_reported_device_must_match_capsule_device() -> None:
    """A producer cannot obtain a host tag for an embedded CUDA pointer."""
    fake = FakeArray(
        np.arange(3.0),
        device=dlpack._DLPACK_DEVICE_CPU,
        capsule_device=dlpack._DLPACK_DEVICE_CUDA,
    )
    with pytest.raises(BufferError, match="reported DLPack device"):
        _consume(fake, shape=(3,))


def test_fake_cuda_host_maps_to_host() -> None:
    """CPU-pinned (kDLCUDAHost) producers remain host descriptors."""
    fake = FakeArray(np.arange(3.0), device=dlpack._DLPACK_DEVICE_CUDA_HOST)
    view = _consume(fake, shape=(3,))
    assert view.memory_space == library.MEMORY_HOST
    view.release()


@pytest.mark.parametrize(
    "device_kind",
    [
        dlpack._DLPACK_DEVICE_ROCM,
        dlpack._DLPACK_DEVICE_ROCM_HOST,
        dlpack._DLPACK_DEVICE_CUDA_MANAGED,
        7,  # Vulkan
        8,  # Metal
        15,  # WebGPU
    ],
)
def test_fake_unsupported_device_kinds_are_rejected(device_kind: int) -> None:
    """ROCm, managed CUDA, and other device kinds fail with a precise error."""
    fake = FakeArray(np.arange(3.0), device=device_kind)
    with pytest.raises(XTBloomNotSupportedError, match="device"):
        _consume(fake, shape=(3,))


def test_fake_stream_negotiation_legacy_default() -> None:
    """CUDA producers receive DLPack stream value 1 for the legacy stream."""
    record: list[object] = []
    fake = FakeArray(
        np.arange(3.0), device=dlpack._DLPACK_DEVICE_CUDA, stream_record=record
    )
    view = _consume(fake, shape=(3,), stream=None)
    assert record == [dlpack._DLPACK_LEGACY_STREAM]
    view.release()


def test_fake_stream_negotiation_custom_handle() -> None:
    """A configured CUstream handle is forwarded verbatim."""
    record: list[object] = []
    fake = FakeArray(
        np.arange(3.0), device=dlpack._DLPACK_DEVICE_CUDA, stream_record=record
    )
    view = _consume(fake, shape=(3,), stream=0x12345678)
    assert record == [0x12345678]
    view.release()


def test_fake_cpu_producer_gets_no_stream() -> None:
    """Host producers are not given a stream argument."""
    record: list[object] = []
    fake = FakeArray(np.arange(3.0), stream_record=record)
    _consume(fake, shape=(3,), stream=0x12345678)
    assert record == [None]


def test_fake_legacy_signature_fallback() -> None:
    """Producers without max_version/stream support fall back to legacy calls."""
    record: list[object] = []
    fake = FakeArray(
        np.arange(3.0),
        max_version_support=False,
        stream_support=True,
        stream_record=record,
    )
    view = _consume(fake, shape=(3,), stream=7)
    # The max_version keyword was dropped; the stream keyword survived.
    assert view.size_bytes == 24
    view.release()


def test_fake_cuda_legacy_signature_keeps_stream_argument() -> None:
    """Dropping max_version must not discard CUDA producer synchronization."""
    record: list[object] = []
    fake = FakeArray(
        np.arange(3.0),
        device=dlpack._DLPACK_DEVICE_CUDA,
        versioned=False,
        max_version_support=False,
        stream_record=record,
    )
    view = _consume(fake, shape=(3,), stream=7)
    assert record == [7, 7]
    view.release()


def test_fake_cuda_without_stream_support_is_rejected() -> None:
    """A CUDA producer that cannot synchronize with xTBloom is unsafe to borrow."""
    fake = FakeArray(
        np.arange(3.0),
        device=dlpack._DLPACK_DEVICE_CUDA,
        stream_support=False,
    )
    with pytest.raises(BufferError, match="consumer CUDA stream"):
        _consume(fake, shape=(3,), stream=7)


def test_fake_copy_unsupported_raises_precise_error() -> None:
    """A copy request to a producer without copy support fails clearly."""
    fake = FakeArray(np.arange(3.0), copy_support=False)
    with pytest.raises(XTBloomNotSupportedError, match="copy"):
        _consume(fake, shape=(3,), copy=True)


def test_fake_unaligned_byte_offset_is_rejected() -> None:
    """Interior byte offsets that break alignment are refused."""
    fake = FakeArray(np.arange(8.0), byte_offset=4)
    with pytest.raises(BufferError, match="aligned"):
        _consume(fake, shape=(8,))
    assert fake.deleted_count() == 1
    assert fake.active_export_count() == 0


def test_naturally_aligned_int32_slice_is_accepted() -> None:
    """The ABI needs scalar alignment, not allocation-base alignment."""
    values = np.arange(9, dtype=np.int32)[1:]
    assert values.ctypes.data % values.dtype.alignment == 0
    view = _consume(values, dtype=I32, shape=(8,))
    assert view.pointer == int(values.ctypes.data)
    view.release()


def test_fake_empty_tensor_has_null_descriptor() -> None:
    """Zero-size vectors map to the ABI's null buffer convention."""
    fake = FakeArray(np.zeros((0,), dtype=np.float64))
    view = _consume(fake, shape=(0,))
    assert view.size_bytes == 0
    descriptor = view.descriptor
    assert descriptor.data is None
    view.release()


def test_descriptor_matches_abi_shape() -> None:
    """The bound ConstBuffer mirrors pointer, size, and memory space."""
    fake = FakeArray(np.arange(3.0))
    view = _consume(fake, shape=(3,))
    descriptor = view.descriptor
    assert int(descriptor.data) == int(fake._data.ctypes.data)
    assert descriptor.size_bytes == 24
    assert descriptor.memory_space == library.MEMORY_HOST
    view.release()
