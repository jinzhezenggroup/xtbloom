"""Unit tests for the DLPack consumer bridge (``gpuxtb._dlpack``).

The bridge is the zero-copy path behind :class:`gpuxtb.ArrayBatch`: it turns
any DLPack producer array into a validated C-ABI buffer view with an exact
capsule-lifetime contract.  Happy paths run against real NumPy arrays; the
protocol edge cases (legacy/read-only capsules, producer exceptions, version
mismatches, stream negotiation, deleter accounting) run against the
deterministic fakes in :mod:`_dlpack_fakes`.
"""

from __future__ import annotations

import ctypes

import gpuxtb._dlpack as dlpack
import numpy as np
import pytest
from _dlpack_fakes import FakeArray
from gpuxtb import library
from gpuxtb.exceptions import GPUxtbNotSupportedError, GPUxtbValueError

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
) -> dlpack.DLPackView:
    return dlpack.consume_from_dlpack(
        producer,
        expected_dtype=dtype,
        expected_shape=shape,
        stream=stream,
        copy=copy,
        writable_required=writable_required,
    )


# --- real NumPy producer ------------------------------------------------------


def test_numpy_versioned_view_is_borrowed_zero_copy() -> None:
    """The view must alias NumPy's own buffer with the exact ABI extents."""
    values = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    view = _consume(values, shape=(4, 3))
    assert view.pointer == int(values.ctypes.data)
    assert view.size_bytes == values.nbytes
    assert view.memory_space == library.MEMORY_HOST
    assert view.writable
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
    with pytest.raises(GPUxtbValueError, match="dtype"):
        _consume(values, shape=(3,))


def test_numpy_shape_mismatch_is_rejected() -> None:
    """Shape/ndim mismatches fail before any buffer is taken."""
    with pytest.raises(GPUxtbValueError, match=r"ndim|shape"):
        _consume(np.zeros((2, 3)), shape=(3,))


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


def test_numpy_writable_required_rejects_readonly() -> None:
    """Read-only output buffers fail deterministically."""
    values = np.arange(3.0)
    values.flags.writeable = False
    with pytest.raises(BufferError, match=r"read-only|writable"):
        _consume(values, writable_required=True)


def test_numpy_writable_required_accepts_writable() -> None:
    """Writable output buffers pass the policy check."""
    values = np.arange(6.0)
    view = _consume(values, shape=(6,), writable_required=True)
    assert view.writable
    view.release()


# --- fake producer: capsule lifecycle -------------------------------------------


def test_fake_deleter_runs_exactly_once_on_release() -> None:
    """Releasing a committed view invokes the producer deleter once."""
    fake = FakeArray(np.arange(3.0))
    view = _consume(fake, shape=(3,))
    assert view.pointer == int(fake._data.ctypes.data)
    view.release()
    assert fake.deleted_count() == 1
    view.release()  # idempotent
    assert fake.deleted_count() == 1


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
    assert fake.deleted_count() == 1


def test_fake_versioned_readonly_flag() -> None:
    """Versioned capsules carrying the read-only flag are rejected as outputs."""
    fake = FakeArray(np.arange(3.0), readonly=True)
    with pytest.raises(BufferError, match="read-only"):
        _consume(fake, shape=(3,), writable_required=True)
    assert fake.deleted_count() == 0  # never committed


def test_fake_unsupported_major_version_is_rejected() -> None:
    """A versioned capsule with major version 2 must be refused."""
    fake = FakeArray(np.arange(3.0), major_version=2)
    with pytest.raises(BufferError, match="major version"):
        _consume(fake, shape=(3,))
    assert fake.deleted_count() == 0


def test_fake_producer_exception_propagates() -> None:
    """Producer failures surface to the caller, not as generic errors."""
    fake = FakeArray(np.arange(3.0), force_error=RuntimeError("boom"))
    with pytest.raises(GPUxtbValueError, match="boom"):
        _consume(fake, shape=(3,))


# --- fake producer: devices and streams ------------------------------------------


def test_fake_cuda_maps_to_cuda_device_memory() -> None:
    """CUDA device producers map to GPUXTB_MEMORY_CUDA_DEVICE."""
    fake = FakeArray(np.arange(3.0), device=dlpack._DLPACK_DEVICE_CUDA)
    view = _consume(fake, shape=(3,))
    assert view.memory_space == library.MEMORY_CUDA_DEVICE
    assert view.device_type == 2
    view.release()


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
    with pytest.raises(GPUxtbNotSupportedError, match="device"):
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


def test_fake_copy_unsupported_raises_precise_error() -> None:
    """A copy request to a producer without copy support fails clearly."""
    fake = FakeArray(np.arange(3.0), copy_support=False)
    with pytest.raises(GPUxtbNotSupportedError, match="copy"):
        _consume(fake, shape=(3,), copy=True)


def test_fake_unaligned_byte_offset_is_rejected() -> None:
    """Interior byte offsets that break alignment are refused."""
    fake = FakeArray(np.arange(8.0), byte_offset=4)
    with pytest.raises(BufferError, match="aligned"):
        _consume(fake, shape=(8,))
    assert fake.deleted_count() == 0


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
