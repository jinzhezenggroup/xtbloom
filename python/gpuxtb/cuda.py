"""Optional CUDA device-memory runtime for the Python interface.

The public C ABI accepts ``GPUXTB_MEMORY_CUDA_DEVICE`` descriptors on the
CUDA backend (:mod:`gpuxtb.library`), which lets gpuxtb consume and publish
buffers that already live on the device instead of staging them through the
host. This module provides the CUDA allocation and transfer calls needed to
build those device buffers from Python.

The calls go directly through ``ctypes`` to ``libcudart``, the same provider
interface enumerated by ``CUDA_MKL_LINKING_EXCEPTION`` that
:func:`gpuxtb.library.device_memory_info` and the conformance runner already
use. No proprietary Python package is imported, so there is no new license
boundary and no new dependency.

Everything here is optional: when ``libcudart`` cannot be loaded, host buffers
remain the default and the rest of the package is unaffected. Device memory
placement is selected per calculator through the ``memory_space`` option of
:class:`gpuxtb.interface.Calculator` and
:class:`gpuxtb.interface.BatchCalculator`.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from . import library
from .exceptions import GPUxtbNotSupportedError, GPUxtbRuntimeError

if TYPE_CHECKING:
    from types import TracebackType

    import numpy as np
    from numpy.typing import NDArray

_CUDA_SUCCESS = 0
_CUDA_ERROR_MEMORY_ALLOCATION = 2
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

_CUDART_SONAME = "libcudart.so.12"


def device_memory_available() -> bool:
    """Return whether the CUDA device-memory runtime can be loaded.

    The probe is lazy and never raises: it checks whether a compatible
    ``libcudart`` can be resolved, so importing the package never forces
    CUDA onto a CPU-only environment.
    """
    try:
        _load_cudart()
        return True
    except (GPUxtbNotSupportedError, OSError):
        return False


def _load_cudart() -> ctypes.CDLL:
    """Load the exact CUDA-12 runtime cohort and declare its signatures.

    Discovery mirrors :func:`gpuxtb.library.device_memory_info`: the loader
    name first, then the runtime directories the package already registers
    for the ``nvidia-*`` CPU-side providers. The runtime dependencies are
    preloaded first so the exact CUDA-12 cohort the package ships is bound
    under the SONAME instead of an older ldconfig-registered toolkit; a
    CUDA-less host, a loader stub without a real driver context, or an old
    runtime with missing symbols all raise a helpful
    ``GPUxtbNotSupportedError``.
    """
    library._preload_runtime_libraries()
    try:
        cudart = ctypes.CDLL(_CUDART_SONAME)
    except OSError:
        for directory in library._runtime_search_dirs():
            candidate = directory / _CUDART_SONAME
            if not candidate.is_file():
                continue
            try:
                cudart = ctypes.CDLL(str(candidate))
                break
            except OSError:
                continue
        else:
            raise GPUxtbNotSupportedError(
                "device-resident CUDA memory requires a loadable CUDA runtime "
                "library; install the `gpuxtb[cuda12]` extra or a compatible "
                "system toolkit so libcudart can be discovered"
            ) from None

    try:
        cuda_get_device = cudart.cudaGetDevice
        cuda_set_device = cudart.cudaSetDevice
        cuda_malloc = cudart.cudaMalloc
        cuda_free = cudart.cudaFree
        cuda_memcpy = cudart.cudaMemcpy
        cuda_get_error_string = cudart.cudaGetErrorString
    except AttributeError as exc:
        raise GPUxtbNotSupportedError(
            f"the CUDA runtime at {getattr(cudart, '_name', '?')} is missing "
            "required device-memory symbols"
        ) from exc

    cuda_get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuda_get_device.restype = ctypes.c_int
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = ctypes.c_int
    cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda_malloc.restype = ctypes.c_int
    cuda_free.argtypes = [ctypes.c_void_p]
    cuda_free.restype = ctypes.c_int
    cuda_memcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cuda_memcpy.restype = ctypes.c_int
    cuda_get_error_string.argtypes = [ctypes.c_int]
    cuda_get_error_string.restype = ctypes.c_char_p
    cuda_get_last_error = cudart.cudaGetLastError
    cuda_get_last_error.argtypes = []
    cuda_get_last_error.restype = ctypes.c_int

    return cudart


def _host_address(host: NDArray[np.generic]) -> int:
    """Return the integer address of a contiguous host buffer."""
    return int(host.ctypes.data)


class CudaDeviceContext:
    """Own CUDA device buffers for one gpuxtb compute call.

    The context selects ``device_id`` for the allocation thread, tracks every
    ``cudaMalloc`` so it can be released on :meth:`close`, and restores the
    caller's previous CUDA device on close. It is deliberately scoped to a
    single compute call: gpuxtb results are synchronously available when the
    call returns, so device buffers are uploaded, computed, downloaded, and
    freed together.

    Decision note (PR #207): device buffers are allocated and released per
    call, mirroring the per-call host numpy owners the host path already
    allocates and the conformance runner's device buffers. Reusing
    context-scoped device buffers across repeated calls is a documented
    follow-up that needs profiling evidence before changing the steady-state
    allocation behavior.

    Parameters
    ----------
    device_id : int
        CUDA device on which the caller's gpuxtb context runs.
    """

    def __init__(self, device_id: int) -> None:
        self._runtime = _load_cudart()
        self._device_id = int(device_id)
        self._allocations: list[int] = []
        self._closed = False

        current = ctypes.c_int()
        status = self._runtime.cudaGetDevice(ctypes.byref(current))
        self._check(status, "cudaGetDevice")
        self._original_device = int(current.value)
        if self._original_device != self._device_id:
            self._check(
                self._runtime.cudaSetDevice(self._device_id),
                f"cudaSetDevice({self._device_id})",
            )

    @property
    def device_id(self) -> int:
        """The CUDA device this context allocates on."""
        return self._device_id

    def _check(self, status: int, operation: str, *, allocation: bool = False) -> None:
        """Raise a runtime error unless the CUDA call reported success.

        ``allocation`` maps the CUDA out-of-memory error to the ABI's
        ``STATUS_ALLOCATION_FAILED`` so automatic batch recovery
        (:meth:`gpuxtb.interface.BatchCalculator.compute` with
        ``auto_batch_size=True``) can retry at smaller sizes.
        """
        if status == _CUDA_SUCCESS:
            return
        message = self._error_string(status)
        # A failed CUDA call leaves a sticky per-thread error that the next
        # ``cudaGetLastError`` consumer (including other CUDA code in this
        # process, such as a later gpuxtb compute) would otherwise observe as
        # its own failure. Consume it so a caught failure here cannot poison
        # subsequent work.
        get_last_error = getattr(self._runtime, "cudaGetLastError", None)
        if get_last_error is not None:
            get_last_error()
        raise GPUxtbRuntimeError(
            f"{operation} failed: {message}",
            status=library.STATUS_ALLOCATION_FAILED
            if allocation and status == _CUDA_ERROR_MEMORY_ALLOCATION
            else None,
        )

    def _error_string(self, status: int) -> str:
        """Decode a CUDA error code into a human-readable message."""
        encoded = self._runtime.cudaGetErrorString(status)
        if encoded is None:
            return f"CUDA error {status}"
        return encoded.decode("utf-8", errors="replace")

    def allocate(self, size_bytes: int) -> int:
        """Allocate a device buffer and register it for cleanup.

        Zero-byte views are represented by the null address and are not
        allocated, mirroring the empty-descriptor convention of the C ABI.
        """
        if self._closed:
            raise GPUxtbRuntimeError("CudaDeviceContext is already closed")
        if size_bytes < 0:
            raise GPUxtbRuntimeError(
                f"cannot allocate a negative CUDA buffer size ({size_bytes} bytes)"
            )
        if size_bytes == 0:
            return 0
        pointer = ctypes.c_void_p()
        status = self._runtime.cudaMalloc(ctypes.byref(pointer), size_bytes)
        self._check(status, f"cudaMalloc({size_bytes})", allocation=True)
        address = int(pointer.value or 0)
        self._allocations.append(address)
        return address

    def upload(self, host: NDArray[np.generic]) -> int:
        """Copy a contiguous host array into freshly allocated device memory."""
        if host.nbytes == 0:
            return 0
        address = self.allocate(host.nbytes)
        status = self._runtime.cudaMemcpy(
            address,
            _host_address(host),
            host.nbytes,
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        self._check(status, f"cudaMemcpy(H2D, {host.nbytes})")
        return address

    def download(self, address: int, host: NDArray[np.generic]) -> None:
        """Copy a device buffer synchronously into a contiguous host array."""
        if host.nbytes == 0:
            return
        status = self._runtime.cudaMemcpy(
            _host_address(host), address, host.nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST
        )
        self._check(status, f"cudaMemcpy(D2H, {host.nbytes})")

    def download_all(self, outputs: list[tuple[int, NDArray[np.generic]]]) -> None:
        """Materialize every registered device output into its host owner."""
        for address, host in outputs:
            self.download(address, host)

    def close(self) -> None:
        """Free all device buffers and restore the caller's CUDA device.

        Every release is attempted even if an earlier one fails; failures are
        collected and raised only after the device is restored.
        """
        if self._closed:
            return
        self._closed = True
        failures: list[str] = []
        if self._original_device != self._device_id:
            status = self._runtime.cudaSetDevice(self._device_id)
            if status != _CUDA_SUCCESS:
                failures.append(
                    f"cudaSetDevice({self._device_id}) for cleanup: "
                    f"{self._error_string(status)}"
                )
        for address in reversed(self._allocations):
            status = self._runtime.cudaFree(address)
            if status != _CUDA_SUCCESS:
                failures.append(f"cudaFree: {self._error_string(status)}")
        self._allocations.clear()
        if self._original_device != self._device_id:
            status = self._runtime.cudaSetDevice(self._original_device)
            if status != _CUDA_SUCCESS:
                failures.append(
                    f"restore cuda device {self._original_device}: "
                    f"{self._error_string(status)}"
                )
        if failures:
            raise GPUxtbRuntimeError(
                "CUDA device-memory cleanup failed: " + "; ".join(failures)
            )

    def __enter__(self) -> CudaDeviceContext:  # noqa: PYI034 - Python 3.10 lacks typing.Self
        """Return this allocation owner for a managed device-memory operation."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release device memory without masking an in-flight exception."""
        if exc_type is None:
            self.close()
            return False
        try:
            self.close()
        except GPUxtbRuntimeError as cleanup_error:  # pragma: no cover - defensive
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(f"CUDA device-memory cleanup also failed: {cleanup_error}")
        return False


__all__ = ["CudaDeviceContext", "device_memory_available"]
