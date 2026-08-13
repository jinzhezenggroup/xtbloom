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
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xtbloom._dlpack as dlpack
import xtbloom.interface as interface
from _dlpack_fakes import DELETED, FakeArray
from xtbloom import library
from xtbloom.exceptions import (
    XTBloomNotSupportedError,
    XTBloomRuntimeError,
    XTBloomValueError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F64 = np.dtype(np.float64)
I32 = np.dtype(np.int32)


class _FakeCudaFunction:
    """ctypes-like callable used to exercise device selection without CUDA."""

    def __init__(self, callback: Callable[..., int]) -> None:
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: object) -> int:
        return self.callback(*args)


class _FakeCudaDriver:
    """Record CUDA driver context-stack changes around DLPack export."""

    def __init__(
        self,
        *,
        init_status: int = 0,
        device_status: int = 0,
        retain_status: int = 0,
        push_status: int = 0,
        restore_status: int = 0,
        release_status: int = 0,
    ) -> None:
        self.current_device = 1
        self.init_status = init_status
        self.device_status = device_status
        self.retain_status = retain_status
        self.push_status = push_status
        self.restore_status = restore_status
        self.release_status = release_status
        self.push_calls: list[int] = []
        self.pop_calls = 0
        self.release_calls: list[int] = []
        self._saved_device = self.current_device
        self.cuInit = _FakeCudaFunction(lambda _flags: self.init_status)
        self.cuDeviceGet = _FakeCudaFunction(self._device_get)
        self.cuDevicePrimaryCtxRetain = _FakeCudaFunction(self._retain)
        self.cuDevicePrimaryCtxRelease = _FakeCudaFunction(self._release)
        self.cuCtxPushCurrent_v2 = _FakeCudaFunction(self._push)
        self.cuCtxPopCurrent_v2 = _FakeCudaFunction(self._pop)

    def _device_get(self, pointer: object, ordinal: object) -> int:
        if self.device_status != 0:
            return self.device_status
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = int(ordinal)
        return 0

    def _retain(self, pointer: object, device: object) -> int:
        if self.retain_status != 0:
            return self.retain_status
        value = int(device)
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = value + 100
        return 0

    def _release(self, device: object) -> int:
        self.release_calls.append(int(device))
        return self.release_status

    def _push(self, context: object) -> int:
        value = int(ctypes.cast(context, ctypes.c_void_p).value or 0)
        self.push_calls.append(value)
        if self.push_status != 0:
            return self.push_status
        self._saved_device = self.current_device
        self.current_device = value - 100
        return 0

    def _pop(self, pointer: object) -> int:
        self.pop_calls += 1
        if self.restore_status != 0:
            return self.restore_status
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            self.current_device + 100
        )
        self.current_device = self._saved_device
        return 0


class _DeviceCheckingFakeArray(FakeArray):
    """Require the fake CUDA device to be current during capsule export."""

    def __init__(
        self,
        data: np.ndarray,
        runtime: _FakeCudaDriver,
        *,
        export_error: Exception | None = None,
    ) -> None:
        super().__init__(
            data,
            device=dlpack._DLPACK_DEVICE_CUDA,
            device_id=0,
        )
        self.runtime = runtime
        self.export_error = export_error
        self.export_devices: list[int] = []

    def __dlpack__(self, **kwargs: object) -> object:
        self.export_devices.append(self.runtime.current_device)
        if self.runtime.current_device != 0:
            raise BufferError("producer device is not current")
        if self.export_error is not None:
            raise self.export_error
        return super().__dlpack__(**kwargs)


class _FakeContext:
    """Minimal resolved context metadata for device-preflight unit tests."""

    def __init__(self, backend: int, device_id: int) -> None:
        self.backend = backend
        self.device_id = device_id


def _consume(
    producer: object,
    *,
    dtype: np.dtype = F64,
    shape: tuple[int, ...] = (3,),
    stream: int | None = None,
    expected_cuda_device: int | None = None,
    copy: bool = False,
    writable_required: bool = False,
    writable_hint: bool | None = None,
) -> dlpack.DLPackView:
    return dlpack.consume_from_dlpack(
        producer,
        expected_dtype=dtype,
        expected_shape=shape,
        stream=stream,
        expected_cuda_device=expected_cuda_device,
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


def test_cuda_export_uses_producer_device_and_restores_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DLPack export temporarily selects the array device, then restores it."""
    runtime = _FakeCudaDriver()
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(np.arange(3.0), runtime)

    view = _consume(fake, shape=(3,))

    assert fake.export_devices == [0]
    assert runtime.push_calls == [100]
    assert runtime.pop_calls == 1
    assert runtime.release_calls == [0]
    assert runtime.current_device == 1
    view.release()


def test_cuda_export_restores_caller_after_producer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer exception cannot strand the caller on the array device."""
    runtime = _FakeCudaDriver()
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(
        np.arange(3.0), runtime, export_error=RuntimeError("boom")
    )

    with pytest.raises(RuntimeError, match="boom"):
        _consume(fake, shape=(3,))

    assert fake.export_devices == [0]
    assert runtime.push_calls == [100]
    assert runtime.pop_calls == 1
    assert runtime.release_calls == [0]
    assert runtime.current_device == 1


def test_cuda_export_reports_device_restore_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed caller-device restoration is explicit instead of silent."""
    runtime = _FakeCudaDriver(restore_status=7)
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(np.arange(3.0), runtime)

    with pytest.raises(XTBloomRuntimeError, match=r"restore.*CUDA context"):
        _consume(fake, shape=(3,))

    assert fake.export_devices == [0]
    assert runtime.push_calls == [100]
    assert runtime.pop_calls == 1
    assert runtime.release_calls == []
    assert runtime.current_device == 0


def test_cuda_export_combines_producer_and_restore_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep both diagnostics when export and caller restoration both fail."""
    runtime = _FakeCudaDriver(restore_status=7)
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(
        np.arange(3.0), runtime, export_error=RuntimeError("producer boom")
    )

    with pytest.raises(
        XTBloomRuntimeError,
        match=r"DLPack export failed \(producer boom\).*restore.*CUDA context",
    ) as error:
        _consume(fake, shape=(3,))

    assert isinstance(error.value.__cause__, RuntimeError)
    assert fake.export_devices == [0]
    assert runtime.push_calls == [100]
    assert runtime.pop_calls == 1
    assert runtime.release_calls == []
    assert runtime.current_device == 0


@pytest.mark.parametrize(
    ("runtime", "message", "release_calls"),
    [
        (_FakeCudaDriver(device_status=2), r"resolve CUDA device", []),
        (_FakeCudaDriver(retain_status=3), r"retain CUDA device", []),
        (_FakeCudaDriver(push_status=4), r"make CUDA device", [0]),
        (
            _FakeCudaDriver(push_status=4, release_status=5),
            r"primary-context release status 5",
            [0],
        ),
        (_FakeCudaDriver(release_status=6), r"restore.*CUDA context", [0]),
    ],
)
def test_cuda_export_reports_driver_failures(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeCudaDriver,
    message: str,
    release_calls: list[int],
) -> None:
    """Every driver failure phase produces an explicit xTBloom diagnostic."""
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(np.arange(3.0), runtime)

    with pytest.raises(XTBloomRuntimeError, match=message):
        _consume(fake, shape=(3,))

    assert runtime.release_calls == release_calls


@pytest.mark.parametrize("driver", [None, object(), _FakeCudaDriver(init_status=1)])
def test_cuda_export_falls_back_to_producer_without_usable_driver(
    monkeypatch: pytest.MonkeyPatch, driver: object | None
) -> None:
    """Missing driver support leaves the producer responsible for diagnostics."""
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: driver)
    fake = FakeArray(
        np.arange(3.0),
        device=dlpack._DLPACK_DEVICE_CUDA,
        force_error=RuntimeError("producer diagnostic"),
    )

    with pytest.raises(XTBloomValueError, match="producer diagnostic"):
        _consume(fake, shape=(3,))


@pytest.mark.parametrize("driver", [None, object()])
def test_cuda_scope_without_usable_driver_is_noop(
    monkeypatch: pytest.MonkeyPatch, driver: object | None
) -> None:
    """A missing driver or symbol table leaves the context scope untouched."""
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: driver)

    with library._cuda_device_scope(0):
        pass


def test_cuda_driver_loader_caches_and_reports_missing_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The driver loader caches success and cleanly reports loader failure."""
    previous = library._cuda_driver_handle
    try:
        library._cuda_driver_handle = None
        loaded = object()
        calls: list[str] = []

        def load(name: str) -> object:
            calls.append(name)
            return loaded

        monkeypatch.setattr(library.ctypes, "CDLL", load)
        assert library._load_cuda_driver() is loaded
        assert library._load_cuda_driver() is loaded
        assert calls == ["libcuda.so.1"]

        library._cuda_driver_handle = None

        def missing(_name: str) -> object:
            raise OSError("missing")

        monkeypatch.setattr(library.ctypes, "CDLL", missing)
        assert library._load_cuda_driver() is None
    finally:
        library._cuda_driver_handle = previous


@pytest.mark.parametrize(
    ("device_type", "backend", "device_id", "message"),
    [
        (dlpack._DLPACK_DEVICE_ROCM, library.BACKEND_CUDA, 0, "not supported"),
        (dlpack._DLPACK_DEVICE_CUDA, library.BACKEND_CPU, 0, "CUDA backend"),
        (dlpack._DLPACK_DEVICE_CUDA, library.BACKEND_CUDA, 1, "device 0.*device 1"),
    ],
)
def test_array_device_preflight_rejects_invalid_requests(
    device_type: int, backend: int, device_id: int, message: str
) -> None:
    """Metadata-only preflight rejects unsupported and foreign devices."""
    array = FakeArray(np.arange(3.0), device=device_type, device_id=0)
    context = _FakeContext(backend, device_id)

    with pytest.raises(XTBloomNotSupportedError, match=message):
        interface._validate_array_devices_before_export([array], context)  # type: ignore[arg-type]


def test_array_device_preflight_accepts_host_array() -> None:
    """Host arrays remain valid for any resolved backend."""
    array = FakeArray(np.arange(3.0))
    context = _FakeContext(library.BACKEND_CUDA, 0)

    interface._validate_array_devices_before_export([array], context)  # type: ignore[arg-type]


def test_foreign_cuda_device_is_rejected_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never pass a context stream to a producer on a foreign CUDA device."""
    runtime = _FakeCudaDriver()
    monkeypatch.setattr(library, "_load_cuda_driver", lambda: runtime)
    fake = _DeviceCheckingFakeArray(np.arange(3.0), runtime)

    with pytest.raises(XTBloomNotSupportedError, match=r"device 0.*device 1"):
        _consume(
            fake,
            shape=(3,),
            stream=12345,
            expected_cuda_device=1,
        )

    assert fake.export_devices == []
    assert runtime.push_calls == []
    assert runtime.pop_calls == 0
    assert runtime.release_calls == []
    assert runtime.current_device == 1


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
