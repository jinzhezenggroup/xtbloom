"""PyTorch autograd integration for gpuxtb through a compiled stable-ABI op.

:func:`gpuxtb_torch` runs packed gpuxtb inference on PyTorch tensors (host CPU
or CUDA device) with zero copy, and exposes exactly one analytic gradient:
``dE/dR = -F`` with respect to the atomic positions, which the native library
already evaluates.

The native data plane lives in a compiled torch extension,
``libgpuxtb_torch_ext`` (built from ``python/gpuxtb/_torch_ext``), which is
written against the **LibTorch Stable ABI** (torch >= 2.10): it binds torch
tensor data pointers directly to the public gpuxtb C ABI descriptors and runs
one synchronous ``gpuxtb_compute`` per call, so the results (and gpuxtb's
failure semantics) are identical to the rest of the package.  One binary works
across torch releases and is loaded lazily through ``torch.ops.load_library``;
the Python module below only supplies the thin autograd ``Function``, the
CUDA async worker, and the ``torch.compile`` graph-break shim.

The autograd contract is intentionally narrow.

* Only ``positions`` may require gradient.  Requesting autograd for any other
  tensor input (atomic numbers, molecular charge, ``uhf``, spin channels, and
  the optional point-charge/response groups) raises
  :class:`GPUxtbNotSupportedError` eagerly at forward time, because gpuxtb
  does not compute those derivatives.
* Gradient flow through the ``forces`` output (the force Hessian ``dF/dR``)
  raises :class:`GPUxtbNotSupportedError` during backward.
* Higher-order differentiation is rejected explicitly instead of returning a
  partial or zero Hessian, because the native force derivative is unavailable.
* Every tensor input is detached before the native call, so calling
  :func:`gpuxtb_torch` never participates in or mutates an existing autograd
  graph beyond the one this op creates.

PyTorch is never imported by the rest of gpuxtb, and this module imports it
only lazily inside :func:`gpuxtb_torch` (through ``importlib``), preserving the
package's "no runtime torch import" guarantee.  The compiled extension is a
plain shared library loaded by torch, so ``import gpuxtb`` likewise does not
load either torch or the extension.
"""

from __future__ import annotations

import contextlib
import importlib
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from . import library
from .exceptions import GPUxtbNotSupportedError, GPUxtbRuntimeError, GPUxtbValueError

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_FUNCTION_CLASS: object | None = None
_ASYNC_FUNCTION_CLASS: object | None = None


class _Tensor(Protocol):
    """Structural type for the torch.Tensor members the op actually uses.

    The protocol is type-checking only; gpuxtb never imports torch at runtime
    unless :func:`gpuxtb_torch` is called.  Only the small, dtype/device-safe
    surface below is declared.
    """

    dtype: object
    shape: tuple[int, ...]
    device: object
    is_cuda: bool

    def is_floating_point(self) -> bool: ...

    def fill_(self, value: object) -> _Tensor: ...

    def dim(self) -> int: ...

    def detach(self) -> _Tensor: ...

    def clone(self) -> _Tensor: ...

    def contiguous(self) -> _Tensor: ...

    def to(self, *, device: object = ..., dtype: object = ...) -> _Tensor: ...

    def index_select(self, dim: int, index: _Tensor) -> _Tensor: ...

    def unsqueeze(self, dim: int) -> _Tensor: ...

    def __neg__(self) -> _Tensor: ...

    def __mul__(self, other: object) -> _Tensor: ...

    def __sub__(self, other: object) -> _Tensor: ...

    def __getitem__(self, index: object) -> _Tensor: ...


class _AutogradFunction(Protocol):
    """Structural type for the ``torch.autograd.Function.apply`` entry point."""

    def apply(self, *args: object) -> tuple[object, object]: ...


class _FunctionCtx(Protocol):
    """Structural type for the autograd context attributes saved by forward."""

    _forces: _Tensor
    _atom_offsets: _Tensor
    _handle: _AsyncHandle

    def set_materialize_grads(self, value: bool) -> None: ...


class _CudaEvent(Protocol):
    """Structural type for the completion event the worker records."""

    def record(self, stream: object) -> None: ...

    def synchronize(self) -> None: ...


class _CudaStream(Protocol):
    """Structural type for the torch stream the job orders against."""

    cuda_stream: int

    def wait_event(self, event: object) -> None: ...


class _CudaDevice(Protocol):
    """Structural type for the CUDA device of the positions tensor."""

    index: int


def _torch() -> ModuleType:
    """Import PyTorch on first use; the rest of gpuxtb never imports it."""
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise GPUxtbNotSupportedError(
            "gpuxtb_torch requires PyTorch, which is an optional integration "
            "and not a gpuxtb dependency"
        ) from exc


def _normalize_layout(value: object) -> object:
    """Return a detached, compact C-contiguous array of the same dtype.

    The DLPack bridge only ever exports borrowed views (torch's ``copy=True``
    does not actually pack strided views into a contiguous copy), so the op
    itself produces the contiguous copy for non-contiguous torch tensors and
    numpy arrays before the native descriptors are bound.  Contiguous inputs
    pass through without a copy; scalar types are never coerced.
    """
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value).detach().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return value


def _to_tensor(value: object) -> _Tensor:
    """Import one finished result array as a PyTorch tensor without a copy."""
    torch = _torch()
    if torch.is_tensor(value):
        return cast("_Tensor", value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return torch.from_dlpack(value)


# --- compiled torch extension -------------------------------------------------
#
# The native data plane is libgpuxtb_torch_ext (LibTorch Stable ABI), a plain
# shared library shipped next to libgpuxtb and loaded once per process through
# torch.ops.load_library.  It is not a CPython module, so the wheel remains a
# single pure-``py3`` archive and no torch headers are needed beyond building
# the extension itself.

_TORCH_EXT_LOADED = False


def _torch_extension_path() -> Path | None:
    """Locate ``libgpuxtb_torch_ext`` next to the resolved ``libgpuxtb``.

    CMake installs the extension into the same ``lib`` directory as
    ``libgpuxtb``.  Mirror ``library.library_path`` resolution so an explicit
    ``GPUXTB_LIBRARY`` (or an installed wheel whose native libraries sit next
    to the Python package) still yields the extension when the Python package
    itself is imported from a source tree on ``PYTHONPATH``.
    """
    runtime_dirs: list[Path] = []
    with contextlib.suppress(library.GPUxtbRuntimeError):
        runtime_dirs.append(Path(library.library_path()).resolve().parent)
    package_dir = Path(__file__).resolve().parent
    for runtime_dir in (
        package_dir / "lib",
        package_dir / "lib64",
        package_dir / "bin",
        package_dir,
    ):
        if runtime_dir not in runtime_dirs:
            runtime_dirs.append(runtime_dir)
    for runtime_dir in runtime_dirs:
        for pattern in (
            "libgpuxtb_torch_ext*.so*",
            "libgpuxtb_torch_ext*.dylib*",
            "libgpuxtb_torch_ext*.dll",
        ):
            matches = sorted(runtime_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _gpuxtb_torch_op() -> Callable[..., tuple[object, object]]:
    """Return (and load on first use) the compiled ``gpuxtb_torch_forward`` op."""
    global _TORCH_EXT_LOADED
    torch = _torch()
    if not _TORCH_EXT_LOADED:
        path = _torch_extension_path()
        if path is None:
            raise GPUxtbNotSupportedError(
                "gpuxtb_torch requires the compiled torch extension "
                "(libgpuxtb_torch_ext), which is not installed next to this "
                "gpuxtb package; rebuild/install gpuxtb with CMake support for "
                "Torch >= 2.10 so the extension is bundled"
            )
        try:
            torch.ops.load_library(str(path))
        except OSError as exc:
            raise GPUxtbRuntimeError(
                "gpuxtb_torch could not load the torch extension: this build "
                "of libgpuxtb_torch_ext may be incompatible with the installed "
                "torch"
            ) from exc
        _TORCH_EXT_LOADED = True
    return cast(
        "Callable[..., tuple[object, object]]", torch.ops.gpuxtb.gpuxtb_torch_forward
    )


_BACKEND_ALIASES = {
    "auto": library.BACKEND_AUTO,
    "cpu": library.BACKEND_CPU,
    "cuda": library.BACKEND_CUDA,
}


def _resolve_backend(backend: str | int) -> int:
    """Validate the backend selector and return its int32 ABI value."""
    if isinstance(backend, str):
        try:
            return _BACKEND_ALIASES[backend]
        except KeyError:
            raise GPUxtbValueError(f"unknown backend {backend!r}") from None
    value = int(backend)
    if value not in (
        library.BACKEND_AUTO,
        library.BACKEND_CPU,
        library.BACKEND_CUDA,
    ):
        raise GPUxtbValueError(f"unknown backend {backend!r}")
    return value


def _resolve_context_scalars(
    backend: int,
    device_id: int | None,
    cpu_threads: int,
    stream: int | None,
) -> tuple[int, int, int]:
    """Resolve the context scalar arguments exactly like ``interface.Context``."""
    resolved_device = -1 if device_id is None else int(device_id)
    resolved_threads = int(cpu_threads)
    if resolved_threads < 0:
        raise GPUxtbValueError("cpu_threads must be nonnegative")
    resolved_stream = 0
    if stream is not None:
        resolved_stream = int(stream)
        if resolved_stream <= 0:
            raise GPUxtbValueError("stream must be a positive CUstream handle")
        if backend == library.BACKEND_CPU:
            raise GPUxtbValueError(
                "a native GPU stream cannot be attached to the CPU backend"
            )
    return resolved_device, resolved_threads, resolved_stream


def _native_forward(
    *,
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object,
    out_energies: object,
    out_forces: object,
    backend: int,
    device_id: int,
    cpu_threads: int,
    stream: int,
    max_scc_iterations: int,
    charge_tolerance: float,
    energy_tolerance: float,
    electronic_temperature: float,
) -> tuple[object, object]:
    """Run the compiled stable-ABI op: tensor data plane + one gpuxtb_compute.

    This is the only native call site of the module (used by both the
    synchronous and asynchronous paths), so tests can substitute it to inject
    failures.  ``electronic_temperature`` is in kelvin; the op converts it to
    the native k_B*T scale.
    """
    return _gpuxtb_torch_op()(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        out_energies,
        out_forces,
        backend,
        device_id,
        cpu_threads,
        stream,
        int(max_scc_iterations),
        float(charge_tolerance),
        float(energy_tolerance),
        float(electronic_temperature),
    )


def _allocate_outputs(
    torch: ModuleType, device: object, nsystems: int, natoms: int
) -> tuple[_Tensor, _Tensor]:
    """Allocate the caller-returned energies/forces tensors on ``device``."""
    return (
        torch.empty((nsystems,), dtype=torch.float64, device=device),
        torch.empty((natoms, 3), dtype=torch.float64, device=device),
    )


def _preflight_positions(torch: ModuleType, positions: object) -> None:
    """Validate the autograd-relevant contract of the positions tensor."""
    positions_t = cast("_Tensor", positions)
    if not (torch.is_tensor(positions_t) and positions_t.is_floating_point()):
        raise GPUxtbValueError("positions must be a floating-point PyTorch tensor")
    if positions_t.dtype != torch.float64:
        raise GPUxtbValueError(
            f"gpuxtb_torch requires float64 positions, got {positions_t.dtype}"
        )
    if positions_t.dim() != 2 or positions_t.shape[1] != 3:
        raise GPUxtbValueError("positions must have shape (natoms, 3)")


def _reject_nonposition_grads(torch: ModuleType, values: dict[str, object]) -> None:
    """Reject autograd on any non-positions input eagerly."""
    for name, value in values.items():
        value_t = cast("_Tensor", value)
        if torch.is_tensor(value_t) and bool(getattr(value, "requires_grad", False)):
            raise GPUxtbNotSupportedError(
                "gpuxtb_torch supports autograd only for positions; "
                f"gradients w.r.t. {name} are not computed"
            )


def _function() -> _AutogradFunction:
    """Return the cached ``torch.autograd.Function`` subclass for this op.

    The subclass is defined after PyTorch is imported so the module keeps its
    lazy-import guarantee; only the first call pays for the definition.
    """
    global _FUNCTION_CLASS
    if _FUNCTION_CLASS is not None:
        return cast("_AutogradFunction", _FUNCTION_CLASS)

    torch = _torch()

    class _GPUxtbTorchFunction(torch.autograd.Function):
        """One gpuxtb forward/backward pair restricted to the dR gradient."""

        @staticmethod
        def forward(
            ctx: _FunctionCtx,
            positions: _Tensor,
            atomic_numbers: object,
            atom_offsets: object,
            molecular_charges: object,
            unpaired_electrons: object,
            spin_channels: object,
            backend: str | int,
            device_id: int | None,
            cpu_threads: int,
            stream: int | None,
            max_scc_iterations: int,
            charge_tolerance: float,
            energy_tolerance: float,
            electronic_temperature: float,
        ) -> tuple[_Tensor, _Tensor]:
            # The C ABI and the DLPack bridge are deliberately strict about
            # dtype/layout, so validate the couple of autograd-relevant facts
            # here and let the native path reject everything else.
            _preflight_positions(torch, positions)
            _reject_nonposition_grads(
                torch,
                {
                    "atomic_numbers": atomic_numbers,
                    "atom_offsets": atom_offsets,
                    "molecular_charges": molecular_charges,
                    "unpaired_electrons": unpaired_electrons,
                    "spin_channels": spin_channels,
                },
            )

            # Normalize layout and detach everything before the native call so
            # the op never participates in an outside autograd graph and the
            # strict zero-copy tensor contract is always satisfied.  Import
            # offsets through DLPack once and pass the resulting Torch tensor
            # to both native inference and backward.  In particular, CUDA
            # producers such as CuPy intentionally reject np.asarray.
            normalized_atom_offsets = _to_tensor(_normalize_layout(atom_offsets))
            nsystems = int(normalized_atom_offsets.shape[0]) - 1
            if spin_channels is None:
                spin_channels = torch.ones(
                    nsystems, dtype=torch.int32, device=positions.device
                )
            resolved_backend = _resolve_backend(backend)
            resolved_device, resolved_threads, resolved_stream = (
                _resolve_context_scalars(
                    resolved_backend, device_id, cpu_threads, stream
                )
            )
            out_energies, out_forces = _allocate_outputs(
                torch,
                positions.device,
                nsystems,
                int(positions.shape[0]),
            )
            _native_result = _native_forward(
                positions=_normalize_layout(positions),
                atomic_numbers=_normalize_layout(atomic_numbers),
                atom_offsets=normalized_atom_offsets,
                molecular_charges=_normalize_layout(molecular_charges),
                unpaired_electrons=_normalize_layout(unpaired_electrons),
                spin_channels=_normalize_layout(spin_channels),
                out_energies=out_energies,
                out_forces=out_forces,
                backend=resolved_backend,
                device_id=resolved_device,
                cpu_threads=resolved_threads,
                stream=resolved_stream,
                max_scc_iterations=max_scc_iterations,
                charge_tolerance=charge_tolerance,
                energy_tolerance=energy_tolerance,
                electronic_temperature=electronic_temperature,
            )
            energies, forces = cast("tuple[_Tensor, _Tensor]", _native_result)
            # Backward needs its own private snapshot of positions' gradient
            # source (-forces) so later in-place user edits of the returned
            # tensors cannot corrupt the gradient.
            ctx._forces = forces.detach().clone()
            ctx._atom_offsets = normalized_atom_offsets.detach().clone()
            # An energy-only loss has no gradient for the forces output.  Keep
            # that state as None so CUDA backward avoids materializing and
            # scanning a full zero tensor solely to distinguish an unused
            # output from a real force-gradient request.
            ctx.set_materialize_grads(False)
            return energies, forces

        @staticmethod
        def backward(
            ctx: _FunctionCtx,
            grad_energies: _Tensor | None,
            grad_forces: _Tensor | None,
        ) -> tuple[_Tensor | None, ...]:
            # create_graph=True enables grad mode while custom backward runs.
            # Reject it immediately: ctx._forces is a detached native result,
            # so allowing this path would silently omit dF/dR and report a
            # partial or all-zero Hessian.
            if torch.is_grad_enabled():
                raise GPUxtbNotSupportedError(
                    "gpuxtb_torch does not support higher-order "
                    "differentiation because dF/dR is not computed"
                )
            if grad_forces is not None:
                raise GPUxtbNotSupportedError(
                    "gpuxtb_torch does not support gradients through the forces "
                    "output (the force Hessian dF/dR is not computed); only the "
                    "energy gradient dE/dR = -F is available"
                )
            if grad_energies is None:
                return (None,) * 14
            # dE/dR = -F, block-diagonal over the ragged batch: atom a of
            # system i receives -grad_energy[i] * F_a.
            offsets = ctx._atom_offsets.to(device=ctx._forces.device, dtype=torch.int64)
            counts = offsets[1:] - offsets[:-1]
            system_of_atom = torch.repeat_interleave(
                torch.arange(offsets.shape[0] - 1, device=offsets.device), counts
            )
            per_atom = (
                grad_energies.to(device=ctx._forces.device, dtype=ctx._forces.dtype)
                .index_select(0, system_of_atom)
                .unsqueeze(1)
            )
            grad_positions = -per_atom * ctx._forces
            return (
                grad_positions,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    _GPUxtbTorchFunction.__name__ = "GPUxtbTorchFunction"
    _FUNCTION_CLASS = _GPUxtbTorchFunction
    return cast("_AutogradFunction", _FUNCTION_CLASS)


# --- asynchronous execution (CUDA-only) -----------------------------------------
#
# gpuxtb's public C ABI is synchronous, so true GPU-side asynchrony would need
# a native asynchronous entry point.  This module-side engine instead delivers
# the caller-visible host asynchrony for the torch op through the internal
# (not public API) entry :func:`gpuxtb.torch._gpuxtb_torch_async`:
#
# * ``_gpuxtb_torch_async`` returns before the native compute finishes: the
#   output tensors are allocated and NaN-prefilled immediately, a background
#   worker fills them with real values, and the host only ever blocks at a real
#   synchronization point (``backward`` or an explicit engine drain).
# * Compute runs on a single daemon worker thread, one job at a time.  Each
#   job owns its own native context (created inside the compiled op), so
#   the per-context CUDA caches and workspaces are never shared between threads
#   and successive calls never race each other.
# * A job always records its completion event, even on failure.  The outputs
#   are pre-filled with NaN while they are still plain tensors (before
#   ``Function.apply`` tracks them), so a failed job simply leaves them as NaN
#   - the library's data-level failure convention - without any post-return
#   torch operation that would bump the autograd version counters of the
#   returned tensors.  The error itself is raised at the next synchronization
#   point: the backward pass of the failing op, or the next async call on this
#   engine.
#
# Contract for eager reads: with a host-synchronous native call, the worker
# thread's completion is the only data boundary, so a read that overtakes it
# sees the NaN prefill, never stale memory.  ``backward`` is the natural
# synchronization point (it awaits the worker and the completion event); a
# plain ``torch.cuda.synchronize()`` cannot cover host work that has not been
# enqueued to the device yet.
#
# Input tensors are only read by the worker after submission, so the caller
# must not mutate an input between enqueueing it and consuming the results.


class _AsyncOutcome:
    """The finished state of one asynchronous job (filled on the worker)."""

    __slots__ = ("error", "result")

    def __init__(self, result: object | None, error: BaseException | None) -> None:
        self.result = result
        self.error = error

    def raise_if_failed(self) -> None:
        """Re-raise the job's error on the waiting caller thread, if any."""
        if self.error is not None:
            raise self.error


class _AsyncJob:
    """One queued compute job: a closure plus the caller's completion event."""

    __slots__ = (
        "_device_index",
        "_fn",
        "done",
        "event",
        "outcome",
        "stream",
    )

    def __init__(
        self,
        fn: Callable[[], object],
        device_index: int,
        event: _CudaEvent,
        stream: _CudaStream,
    ) -> None:
        self._fn = fn
        self._device_index = device_index
        self.event = event
        self.stream = stream
        self.done = threading.Event()
        self.outcome: _AsyncOutcome | None = None

    def run(self) -> None:
        """Execute the job on the worker thread and finalize its outcome."""
        torch = _torch()
        torch.cuda.set_device(self._device_index)
        result = None
        error: BaseException | None = None
        try:
            result = self._fn()
        except BaseException as exc:  # noqa: BLE001 - must deliver any failure
            error = exc
        self.outcome = _AsyncOutcome(result=result, error=error)
        self.done.set()
        # Always record completion so a caller's stream barrier cannot hang,
        # even when the job failed before any output was committed.
        self.event.record(self.stream)


class _AsyncHandle:
    """Non-blocking handle to one queued job; resolves to its outcome later."""

    __slots__ = ("_job",)

    def __init__(self, job: _AsyncJob) -> None:
        self._job = job

    def await_outcome(self) -> _AsyncOutcome:
        """Block until the worker finished this job, then return its outcome."""
        outcome = self._job.outcome
        while outcome is None:
            self._job.done.wait()
            outcome = self._job.outcome
        return outcome

    def synchronize(self) -> None:
        """Wait for completion on host and device and surface any job error."""
        outcome = self.await_outcome()
        self._job.event.synchronize()
        outcome.raise_if_failed()


class _AsyncComputeEngine:
    """Single-daemon-worker FIFO executor for the op's async compute jobs.

    A job always runs to completion on the worker, and the engine never touches
    a live job's data from the submitting thread, so jobs are safe to leave
    queued while the caller continues enqueueing other torch work.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[_AsyncJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._pending_error: BaseException | None = None

    def submit(self, job: _AsyncJob) -> _AsyncHandle:
        """Enqueue one job and return a non-blocking handle to it.

        Raises the most recent failed job's error (and clears it) before
        accepting new work, so a failure an inference-only caller never
        synchronizes on still surfaces at the next async call.
        """
        with self._lock:
            if self._pending_error is not None:
                error, self._pending_error = self._pending_error, None
                raise error
            if not self._started:
                self._thread = threading.Thread(
                    target=self._run, name="gpuxtb-async", daemon=True
                )
                self._thread.start()
                self._started = True
        self._queue.put(job)
        return _AsyncHandle(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            job.run()
            outcome = job.outcome
            if outcome is not None and outcome.error is not None:
                with self._lock:
                    if self._pending_error is None:
                        self._pending_error = outcome.error

    def close(self) -> None:
        """Drain pending jobs, stop the worker, and join it."""
        with self._lock:
            if not self._started:
                return
            self._queue.put(None)
            thread = self._thread
            if thread is not None:
                thread.join(timeout=120.0)
            self._started = False
            self._thread = None


_ASYNC_ENGINE: _AsyncComputeEngine | None = None


def _async_engine() -> _AsyncComputeEngine:
    """Return the process-wide async compute engine (created on first use)."""
    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        _ASYNC_ENGINE = _AsyncComputeEngine()
    return _ASYNC_ENGINE


def _close_async_engine() -> None:
    """Drain and stop the process-wide engine; used by tests and shutdown."""
    global _ASYNC_ENGINE
    engine, _ASYNC_ENGINE = _ASYNC_ENGINE, None
    if engine is not None:
        engine.close()


def _async_function() -> _AutogradFunction:
    """Return the cached asynchronous ``torch.autograd.Function`` subclass."""
    global _ASYNC_FUNCTION_CLASS
    if _ASYNC_FUNCTION_CLASS is not None:
        return cast("_AutogradFunction", _ASYNC_FUNCTION_CLASS)

    torch = _torch()

    class _GPUxtbTorchAsyncFunction(torch.autograd.Function):
        """One gpuxtb forward/backward pair with a host/device-async forward.

        Forward allocates the result tensors, enqueues the native compute on
        the process-wide async engine, and returns immediately; the caller's
        stream is ordered to wait on the worker's completion event.  Backward
        is the synchronization point: it waits for completion, surfaces any
        deferred failure, and only then reads the forces.
        """

        @staticmethod
        def forward(
            ctx: _FunctionCtx,
            positions: _Tensor,
            atomic_numbers: object,
            atom_offsets: object,
            molecular_charges: object,
            unpaired_electrons: object,
            spin_channels: object,
            backend: str | int,
            device_id: int | None,
            cpu_threads: int,
            stream: int | None,
            max_scc_iterations: int,
            charge_tolerance: float,
            energy_tolerance: float,
            electronic_temperature: float,
        ) -> tuple[_Tensor, _Tensor]:
            _preflight_positions(torch, positions)
            if not bool(positions.is_cuda):
                raise GPUxtbNotSupportedError(
                    "async_exec=True requires CUDA tensors: gpuxtb's public ABI "
                    "is synchronous, so the async engine exists only for "
                    "device-resident work; call gpuxtb_torch without "
                    "async_exec=True for the host path"
                )
            if backend == "cpu":
                raise GPUxtbNotSupportedError(
                    "async_exec=True requires the CUDA backend, not the CPU backend"
                )
            _reject_nonposition_grads(
                torch,
                {
                    "atomic_numbers": atomic_numbers,
                    "atom_offsets": atom_offsets,
                    "molecular_charges": molecular_charges,
                    "unpaired_electrons": unpaired_electrons,
                    "spin_channels": spin_channels,
                },
            )

            # Normalize layout and detach everything before the native call so
            # the op never participates in an outside autograd graph and the
            # strict zero-copy tensor contract is always satisfied.
            normalized_atom_offsets = _to_tensor(_normalize_layout(atom_offsets))
            normalized_atomic_numbers = _normalize_layout(atomic_numbers)
            normalized_positions = _normalize_layout(positions)
            normalized_molecular_charges = _normalize_layout(molecular_charges)
            normalized_unpaired_electrons = _normalize_layout(unpaired_electrons)
            resolved_backend = _resolve_backend(backend)
            if resolved_backend == library.BACKEND_CPU:
                raise GPUxtbNotSupportedError(
                    "async_exec=True requires the CUDA backend, not the CPU backend"
                )
            device = positions.device
            if spin_channels is None:
                spin_channels = torch.ones(
                    int(normalized_atom_offsets.shape[0]) - 1,
                    dtype=torch.int32,
                    device=device,
                )
            normalized_spin_channels = _normalize_layout(spin_channels)

            nsystems = int(normalized_atom_offsets.shape[0]) - 1
            natoms = int(positions.shape[0])
            outputs = {
                "energies": torch.empty(
                    (nsystems,), dtype=torch.float64, device=device
                ),
                "forces": torch.empty((natoms, 3), dtype=torch.float64, device=device),
            }
            # Pre-fill the floating-point outputs with NaN while they are still
            # plain tensors (before ``Function.apply`` tracks them), so a job
            # that fails keeps the library's data-level NaN failure convention
            # without any post-return torch operation on the returned tensors.
            for name in ("energies", "forces"):
                outputs[name].fill_(float("nan"))
            # Bind the worker to detached views: they share the same storage, so
            # the returned (differentiable) tensors alias the finished bytes,
            # while the worker's in-place writes never bump the version counter
            # of the autograd-tracked outputs.
            out_views = {name: tensor.detach() for name, tensor in outputs.items()}
            # The whole point of the async path is stream-native ordering with
            # the caller's surrounding torch work, so the gpuxtb context runs
            # on torch's current stream for the positions device.  torch
            # represents its legacy default stream with handle 0, which gpuxtb
            # reads as "no explicit stream" (the same legacy default stream);
            # custom streams pass their real handle.
            stream = torch.cuda.current_stream(device)
            raw_handle = int(stream.cuda_stream)
            stream_handle = raw_handle if raw_handle > 0 else 0
            resolved_device, resolved_threads, _ = _resolve_context_scalars(
                resolved_backend, device_id, cpu_threads, stream=None
            )
            event = torch.cuda.Event()

            def run() -> object:
                return _native_forward(
                    positions=normalized_positions,
                    atomic_numbers=normalized_atomic_numbers,
                    atom_offsets=normalized_atom_offsets,
                    molecular_charges=normalized_molecular_charges,
                    unpaired_electrons=normalized_unpaired_electrons,
                    spin_channels=normalized_spin_channels,
                    out_energies=cast("_Tensor", out_views["energies"]),
                    out_forces=cast("_Tensor", out_views["forces"]),
                    backend=resolved_backend,
                    device_id=resolved_device,
                    cpu_threads=resolved_threads,
                    stream=stream_handle,
                    max_scc_iterations=max_scc_iterations,
                    charge_tolerance=charge_tolerance,
                    energy_tolerance=energy_tolerance,
                    electronic_temperature=electronic_temperature,
                )

            handle = _async_engine().submit(
                _AsyncJob(
                    run,
                    int(cast("_CudaDevice", positions.device).index),
                    event,
                    stream,
                )
            )
            # Keep the handle on the autograd context: backward waits on it
            # (``await_outcome`` plus the completion event) before reading the
            # forces.  A ``cudaStreamWaitEvent`` enqueued here would be a no-op
            # because the event is not recorded until the worker finishes, so
            # eager reads are ordered by the documented barrier points instead
            # (``backward`` / ``torch.cuda.synchronize()``).
            ctx._handle = handle
            ctx._atom_offsets = normalized_atom_offsets.detach().clone()
            ctx.set_materialize_grads(False)
            return outputs["energies"], outputs["forces"]

        @staticmethod
        def backward(
            ctx: _FunctionCtx,
            grad_energies: _Tensor | None,
            grad_forces: _Tensor | None,
        ) -> tuple[_Tensor | None, ...]:
            if torch.is_grad_enabled():
                raise GPUxtbNotSupportedError(
                    "gpuxtb_torch does not support higher-order "
                    "differentiation because dF/dR is not computed"
                )
            if grad_forces is not None:
                raise GPUxtbNotSupportedError(
                    "gpuxtb_torch does not support gradients through the forces "
                    "output (the force Hessian dF/dR is not computed); only the "
                    "energy gradient dE/dR = -F is available"
                )
            if grad_energies is None:
                return (None,) * 14
            # Backward is the synchronization point: the worker fills the
            # outputs asynchronously, so wait for completion and surface any
            # deferred failure before touching the forces.  The event wait is a
            # host-blocking device wait, exactly where the user needs results.
            handle = ctx._handle
            outcome = handle.await_outcome()
            outcome.raise_if_failed()
            handle.synchronize()
            result = cast("tuple[object, object] | None", outcome.result)
            if result is None:
                raise GPUxtbRuntimeError(
                    "async gpuxtb job produced no result without an error"
                )
            forces = cast("_Tensor", result[1]).detach().clone()
            # dE/dR = -F, block-diagonal over the ragged batch: atom a of
            # system i receives -grad_energy[i] * F_a.
            offsets = ctx._atom_offsets.to(device=forces.device, dtype=torch.int64)
            counts = offsets[1:] - offsets[:-1]
            system_of_atom = torch.repeat_interleave(
                torch.arange(offsets.shape[0] - 1, device=offsets.device), counts
            )
            per_atom = (
                grad_energies.to(device=forces.device, dtype=forces.dtype)
                .index_select(0, system_of_atom)
                .unsqueeze(1)
            )
            grad_positions = -per_atom * forces
            return (
                grad_positions,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    _GPUxtbTorchAsyncFunction.__name__ = "GPUxtbTorchAsyncFunction"
    _ASYNC_FUNCTION_CLASS = _GPUxtbTorchAsyncFunction
    return cast("_AutogradFunction", _ASYNC_FUNCTION_CLASS)


def _gpuxtb_torch_impl(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    async_exec: bool = False,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
) -> tuple[object, object]:
    """Execute one packed gpuxtb inference (the traceable-unsafe core).

    Kept private so it can be exposed through a ``torch.compile``-safe wrapper:
    Dynamo must never trace the ctypes/DLPack/worker-thread internals below.

    The deferred-result contract of ``async_exec`` cannot be expressed inside a
    compiled graph (a fused region consumes results immediately), so an async
    request that was submitted from a ``torch.compile`` trace degrades to the
    synchronous path here: correct values, no compilation speedup, and no
    NaN-read hazard.  The trace-time ``gpuxtb_torch`` entry increments a
    thread-local counter so the eager replay of that call can be told apart
    from an ordinary eager async call.
    """
    global _compile_degraded_async
    if async_exec and getattr(_compile_degraded_async, "count", 0) > 0:
        _compile_degraded_async.count -= 1
        async_exec = False
    _torch()
    if async_exec:
        if stream is not None:
            raise GPUxtbValueError(
                "async_exec=True runs on torch.cuda.current_stream(), so an "
                "explicit stream argument is not supported"
            )
        function = _async_function()
    else:
        function = _function()
    return function.apply(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        backend,
        device_id,
        int(cpu_threads),
        stream,
        int(max_scc_iterations),
        float(charge_tolerance),
        float(energy_tolerance),
        float(electronic_temperature),
    )


_GPU_XTB_TORCH_IMPL: Callable[..., tuple[object, object]] | None = None
# Thread-local count of async requests whose trace-time ``gpuxtb_torch`` call
# detected an active ``torch.compile`` trace (see ``_gpuxtb_torch_impl``).
_compile_degraded_async = threading.local()


def _disabled_torch_impl(torch: ModuleType) -> Callable[..., tuple[object, object]]:
    """Mark ``_gpuxtb_torch_impl`` as opaque so Dynamo graph-breaks on it.

    Dynamo cannot trace gpuxtb's ctypes/DLPack/worker-thread internals (tracing
    raises, e.g. an unsupported ``DLDeviceType`` constant), so the op is
    deliberately excluded from the compiler: the recursive form of
    ``torch._dynamo.disable`` uninstalls Dynamo's frame interception for the
    duration of the call and marks the callable opaque, so calling it inside
    ``torch.compile`` graph-breaks and executes it eagerly.  Correct results,
    no trace-time error, and no compilation speedup for the gpuxtb call itself.
    """
    return cast(
        "Callable[..., tuple[object, object]]",
        torch._dynamo.disable(_gpuxtb_torch_impl),
    )


def _is_compiling(torch: ModuleType) -> bool:
    """Return whether the call runs inside a ``torch.compile`` trace."""
    for module_name in ("compiler", "_dynamo"):
        probe = getattr(getattr(torch, module_name, None), "is_compiling", None)
        if probe is not None and callable(probe) and bool(probe()):
            return True
    return False


def gpuxtb_torch(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
) -> tuple[object, object]:
    """Run gpuxtb inference on PyTorch tensors with a ``dR``-only autograd op.

    The inputs mirror the packed ragged-batch descriptors of
    :class:`gpuxtb.ArrayBatch`; ``positions`` is the only differentiable
    tensor and the only argument that may set ``requires_grad=True``.  The
    returned ``(energies, forces)`` pair follows the same units as the rest of
    gpuxtb (Hartree and Hartree/bohr).  This public entry point is always
    synchronous; an experimental (internal, not public API) host-asynchronous
    CUDA variant lives at :func:`gpuxtb.torch._gpuxtb_torch_async`.

    ``gpuxtb_torch`` is eager-only by design: it drives the native library
    through ctypes/DLPack, which Dynamo cannot trace.  Calling it inside
    ``torch.compile`` therefore inserts a graph break and executes it eagerly
    (correct results, no compilation speedup for the gpuxtb call itself),
    instead of failing at trace time.

    Parameters
    ----------
    positions : (natoms, 3) float64 torch.Tensor
        Cartesian coordinates in bohr.  The only input supporting autograd;
        its analytic gradient is ``dE/dR = -F``.
    atomic_numbers : (natoms,) int32
        Atomic numbers; a torch tensor or any DLPack producer (for example a
        numpy array).
    atom_offsets : (nsystems + 1,) int64
        Ragged atom offsets; ``offsets[-1]`` is the total atom count.
    molecular_charges : (nsystems,) float64
        Total molecular charge of each system.
    unpaired_electrons : (nsystems,) int32
        Number of unpaired electrons of each system.
    spin_channels : (nsystems,) int32, optional
        Orbital channels (1 restricted / 2 unrestricted); defaults to all
        restricted ``1``, exactly like :class:`gpuxtb.ArrayBatch`.
    backend, device_id, cpu_threads, stream
        Same context selection as :class:`gpuxtb.ArrayBatch`.
    max_scc_iterations, charge_tolerance, energy_tolerance, electronic_temperature
        Same SCC options as :class:`gpuxtb.ArrayBatch`.  ``electronic_temperature``
        is given in kelvin.

    Returns
    -------
    energies : (nsystems,) float64 torch.Tensor
        Per-system energies in Hartree.
    forces : (natoms, 3) float64 torch.Tensor
        Per-atom forces in Hartree/bohr.

    Raises
    ------
    GPUxtbNotSupportedError
        If PyTorch is unavailable, if a non-``positions`` tensor requests
        autograd, if gradient flows through the ``forces`` output, or if
        higher-order differentiation is requested.
    GPUxtbValueError
        If ``positions`` is not a ``float64`` tensor of shape ``(natoms, 3)``.
    """
    global _GPU_XTB_TORCH_IMPL
    torch = _torch()
    if _GPU_XTB_TORCH_IMPL is None:
        _GPU_XTB_TORCH_IMPL = _disabled_torch_impl(torch)
    return _GPU_XTB_TORCH_IMPL(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        backend=backend,
        device_id=device_id,
        cpu_threads=cpu_threads,
        stream=stream,
        async_exec=False,
        max_scc_iterations=max_scc_iterations,
        charge_tolerance=charge_tolerance,
        energy_tolerance=energy_tolerance,
        electronic_temperature=electronic_temperature,
    )


def _gpuxtb_torch_async(
    positions: object,
    atomic_numbers: object,
    atom_offsets: object,
    molecular_charges: object,
    unpaired_electrons: object,
    spin_channels: object | None = None,
    *,
    backend: str | int = "auto",
    device_id: int | None = None,
    cpu_threads: int = 1,
    stream: int | None = None,
    max_scc_iterations: int = 250,
    charge_tolerance: float = 1.0e-6,
    energy_tolerance: float = 1.0e-8,
    electronic_temperature: float = 300.0,
) -> tuple[object, object]:
    """Experimental host-asynchronous CUDA variant (internal, not public API).

    Same arguments as :func:`gpuxtb_torch`.  The call returns before the native
    compute finishes and a background worker fills the returned tensors; this
    is kept private because its contract is intrusive: with the host-synchronous
    public C ABI the worker's completion is the only data boundary, the tensors
    read as NaN until then, ``backward`` is the synchronization point, and eager
    reads must wait on the worker host-side.  Requires CUDA ``positions`` and
    the CUDA backend; the ``stream`` argument is refused because the op runs on
    ``torch.cuda.current_stream()``.  Inside ``torch.compile`` it degrades to
    the synchronous path (a compiled graph cannot express deferred results).
    """
    global _GPU_XTB_TORCH_IMPL
    torch = _torch()
    if _GPU_XTB_TORCH_IMPL is None:
        _GPU_XTB_TORCH_IMPL = _disabled_torch_impl(torch)
    if _is_compiling(torch):
        # An async request inside torch.compile: mark it so the eager replay
        # (which cannot see the compiler) degrades to the synchronous path in
        # ``_gpuxtb_torch_impl``.
        _compile_degraded_async.count = getattr(_compile_degraded_async, "count", 0) + 1
    return _GPU_XTB_TORCH_IMPL(
        positions,
        atomic_numbers,
        atom_offsets,
        molecular_charges,
        unpaired_electrons,
        spin_channels,
        backend=backend,
        device_id=device_id,
        cpu_threads=cpu_threads,
        stream=stream,
        async_exec=True,
        max_scc_iterations=max_scc_iterations,
        charge_tolerance=charge_tolerance,
        energy_tolerance=energy_tolerance,
        electronic_temperature=electronic_temperature,
    )
