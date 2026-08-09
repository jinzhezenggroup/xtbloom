"""Tests for the asynchronous execution path of :func:`gpuxtb.gpuxtb_torch`.

The asynchronous path (internal entry ``gpuxtb.torch._gpuxtb_torch_async``,
not part of the public ``gpuxtb_torch`` signature) is the torch integration's
host/device asynchrony layer: forward returns before the native compute
finishes, a background worker fills the returned tensors, and the returned
tensors read as NaN until the worker completes (never as stale allocation
contents).  Because
the public C ABI is host-synchronous, the worker thread's completion is the
only guaranteed data boundary, so *eager* reads of the results require a
host-side wait on the worker (the ``backward`` pass is such a point; an engine
drain is used in these tests), not merely a device barrier.

These tests cover

* the engine/job/handle mechanics white-box (CPU-only, no GPU required);
* the eager guards that reject the async entry on host tensors or with an
  explicit stream;
* the CUDA functional behaviour (results match the synchronous op after a
  barrier, backward is ``dE/dR = -F``, pipelining, NaN-before-completion,
  deferred-failure propagation, and poisoning), gated on a real GPU plus a
  torch CUDA build.
"""

from __future__ import annotations

import importlib
import threading

import numpy as np
import pytest
from gpuxtb import gpuxtb_torch
from gpuxtb.exceptions import (
    GPUxtbNotSupportedError,
    GPUxtbRuntimeError,
    GPUxtbValueError,
)
from gpuxtb.torch import _gpuxtb_torch_async

_TORCH = importlib.util.find_spec("torch")


def _skip_reason() -> str | None:
    """Return a skip reason when the torch tests cannot run."""
    return None if _TORCH is not None else "torch is not installed"


def _library_has_cuda() -> bool:
    """Check whether a CUDA context can actually be created on this host."""
    from gpuxtb.exceptions import GPUxtbRuntimeError
    from gpuxtb.interface import Context

    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


def _cuda_ready() -> str | None:
    """Return a skip reason when CUDA torch tests cannot run."""
    reason = _skip_reason()
    if reason:
        return reason
    import torch

    if not torch.cuda.is_available():
        return "torch has no usable CUDA device"
    if not _library_has_cuda():
        return "CUDA backend is not available on this host"
    return None


# --- helpers ----------------------------------------------------------------------


class _FakeEvent:
    """Minimal completion-event stand-in for engine white-box tests."""

    def __init__(self) -> None:
        self.recorded: object | None = None
        self.syncs = 0

    def record(self, stream: object) -> None:
        """Record the (fake) completion on the given stream."""
        self.recorded = stream

    def synchronize(self) -> None:
        """Count a (fake) device-side synchronization."""
        self.syncs += 1


class _FakeStream:
    """Minimal torch-stream stand-in for engine white-box tests."""

    cuda_stream = 0

    def __init__(self) -> None:
        self.waits: list[object] = []

    def wait_event(self, event: object) -> None:
        """Record a (fake) stream barrier on this stream."""
        self.waits.append(event)


def _engine_modules() -> tuple[object, object]:
    """Return the process-wide async engine and its handle class."""
    from gpuxtb import torch as torch_module

    return torch_module, torch_module._async_engine()


@pytest.fixture(autouse=True)
def _clean_engine() -> object:
    """Drain the process-wide async engine before and after each test."""
    from gpuxtb import torch as torch_module

    yield torch_module._close_async_engine()
    torch_module._close_async_engine()


def _make_job(torch_module: object, fn: object, monkeypatch: object) -> object:
    """Build a job with no-op torch CUDA setup and a fake event/stream."""
    import torch

    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    return torch_module._AsyncJob(fn, 0, _FakeEvent(), _FakeStream())


# --- engine mechanics (CPU-only white-box) ---------------------------------------


def test_submit_does_not_block_on_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """An async submission returns before the queued job has run."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    torch_module, engine = _engine_modules()
    started = threading.Event()
    release = threading.Event()

    def slow_fn() -> object:
        started.set()
        release.wait(timeout=30.0)
        return "done"

    job = _make_job(torch_module, slow_fn, monkeypatch)
    handle = engine.submit(job)
    # Submit must return while the worker is (or will be) running slow_fn;
    # a blocking submit would wait forever on ``release`` and time out.
    assert started.wait(timeout=30.0)
    release.set()
    assert handle.await_outcome().result == "done"


def test_await_outcome_delivers_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """await_outcome resolves to the worker's finished outcome."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    torch_module, engine = _engine_modules()
    handle = engine.submit(_make_job(torch_module, lambda: "value", monkeypatch))
    outcome = handle.await_outcome()
    assert outcome.result == "value"
    assert outcome.error is None


def test_close_drains_and_recreates_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """close() finishes queued jobs, then a fresh engine can run more."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    torch_module, engine = _engine_modules()
    handle = engine.submit(_make_job(torch_module, lambda: "first", monkeypatch))
    torch_module._close_async_engine()
    assert handle.await_outcome().result == "first"

    # A later submission lazily recreates a brand-new worker.
    fresh_engine = torch_module._async_engine()
    handle2 = fresh_engine.submit(
        _make_job(torch_module, lambda: "second", monkeypatch)
    )
    assert handle2.await_outcome().result == "second"


def test_failed_job_surfaces_at_next_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job failure poisons the next submission with the deferred error."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    torch_module, engine = _engine_modules()

    def failing() -> object:
        raise GPUxtbRuntimeError("boom")

    handle = engine.submit(_make_job(torch_module, failing, monkeypatch))
    outcome = handle.await_outcome()
    with pytest.raises(GPUxtbRuntimeError, match="boom"):
        outcome.raise_if_failed()

    # The next submission re-raises (and clears) the poisoned error.
    with pytest.raises(GPUxtbRuntimeError, match="boom"):
        engine.submit(_make_job(torch_module, lambda: "never queued", monkeypatch))

    # After the poison was consumed, new work proceeds normally.
    handle2 = engine.submit(_make_job(torch_module, lambda: "ok", monkeypatch))
    assert handle2.await_outcome().result == "ok"


def test_handle_synchronize_surfaces_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """handle.synchronize() waits and re-raises the job's error."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    torch_module, engine = _engine_modules()

    def failing() -> object:
        raise ValueError("sync-boom")

    handle = engine.submit(_make_job(torch_module, failing, monkeypatch))
    with pytest.raises(ValueError, match="sync-boom"):
        handle.synchronize()


# --- eager guards (run everywhere) -----------------------------------------------


def test_async_entry_rejects_host_positions() -> None:
    """The async entry without a CUDA tensor is refused eagerly."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        [
            [0.0, 0.0, -0.73],
            [1.44, 0.0, 0.36],
            [-1.44, 0.0, 0.36],
        ],
        dtype=torch.float64,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 3], dtype=torch.int64)
    charges = torch.zeros(1, dtype=torch.float64)
    uhf = torch.zeros(1, dtype=torch.int32)
    spins = torch.ones(1, dtype=torch.int32)
    with pytest.raises(GPUxtbNotSupportedError, match="CUDA"):
        _gpuxtb_torch_async(
            positions,
            numbers,
            offsets,
            charges,
            uhf,
            spins,
            backend="cpu",
        )


def test_async_entry_rejects_explicit_stream() -> None:
    """The async entry cannot be combined with a caller-provided stream."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        [
            [0.0, 0.0, -0.73],
            [1.44, 0.0, 0.36],
            [-1.44, 0.0, 0.36],
        ],
        dtype=torch.float64,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 3], dtype=torch.int64)
    charges = torch.zeros(1, dtype=torch.float64)
    uhf = torch.zeros(1, dtype=torch.int32)
    spins = torch.ones(1, dtype=torch.int32)
    with pytest.raises(GPUxtbValueError, match="stream"):
        _gpuxtb_torch_async(
            positions,
            numbers,
            offsets,
            charges,
            uhf,
            spins,
            backend="cpu",
            stream=0x1234,
        )


# --- CUDA functional tests -------------------------------------------------------


def _packed(
    numbers: list[np.ndarray], positions: list[np.ndarray], torch: object
) -> dict[str, object]:
    """Pack a ragged batch into descriptor arrays as CUDA torch tensors."""
    offsets = [0]
    all_numbers: list[int] = []
    all_positions: list[float] = []
    for numbers_i, positions_i in zip(numbers, positions, strict=True):
        all_numbers.extend(int(value) for value in numbers_i)
        all_positions.extend(float(value) for value in positions_i.ravel())
        offsets.append(len(all_numbers))
    nsystems = len(numbers)
    arrays = {
        "atom_offsets": np.asarray(offsets, dtype=np.int64),
        "atomic_numbers": np.asarray(all_numbers, dtype=np.int32),
        "positions": np.asarray(all_positions, dtype=np.float64).reshape(-1, 3),
        "molecular_charges": np.zeros(nsystems, dtype=np.float64),
        "unpaired_electrons": np.zeros(nsystems, dtype=np.int32),
        "spin_channels": np.ones(nsystems, dtype=np.int32),
    }
    return {
        name: torch.tensor(
            value.tolist(),
            dtype=torch.int64
            if value.dtype == np.int64
            else (torch.int32 if value.dtype == np.int32 else torch.float64),
            device="cuda",
        )
        for name, value in arrays.items()
    }


WATER_NUMBERS = np.array([8, 1, 1], dtype=np.int32)
WATER_POSITIONS = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ],
    dtype=np.float64,
)


@pytest.mark.cuda
def test_async_cuda_matches_sync() -> None:
    """Async results equal the synchronous CUDA op after a host wait."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    args = (
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
    )

    sync_energies, sync_forces = gpuxtb_torch(*args, backend="cuda")
    async_energies, async_forces = _gpuxtb_torch_async(*args, backend="cuda")
    # Force a full host/device barrier before comparing.
    async_energies.sum().backward()
    torch.cuda.synchronize()
    assert torch.allclose(async_energies.cpu(), sync_energies.cpu(), atol=0.0, rtol=0.0)
    assert torch.allclose(async_forces.cpu(), sync_forces.cpu(), atol=0.0, rtol=0.0)
    assert torch.allclose(positions.grad, -async_forces, atol=0.0, rtol=0.0)


@pytest.mark.cuda
def test_async_backward_grad_equals_neg_forces() -> None:
    """The async analytic gradient is exactly ``-F`` after backward."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = _gpuxtb_torch_async(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cuda",
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert positions.grad.device.type == "cuda"
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


@pytest.mark.cuda
def test_async_pipelines_multiple_forward_calls() -> None:
    """Several async forwards can be enqueued before any synchronization."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch
    from gpuxtb import torch as torch_module

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    args = (
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
    )

    sync_energy, _ = gpuxtb_torch(*args, backend="cuda")

    # Submit several async runs back to back; submission is host-non-blocking.
    # With a host-synchronous native ABI a device barrier alone cannot wait for
    # a worker thread that has not yet enqueued its calls, so the test waits
    # host-side (draining the engine) before consuming, exactly like backward.
    energies_list = [_gpuxtb_torch_async(*args, backend="cuda")[0] for _ in range(3)]
    torch_module._close_async_engine()
    energy_sum = energies_list[0].sum()
    for energies in energies_list[1:]:
        energy_sum = energy_sum + energies.sum()
    energy_sum.backward()
    assert torch.allclose(energy_sum.cpu(), 3.0 * sync_energy.cpu(), atol=1.0e-12)


@pytest.mark.cuda
def test_async_deferred_failure_nan_fills_and_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native failure NaN-fills floats, raises in backward, poisons submit."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch
    from gpuxtb import torch as torch_module

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise GPUxtbRuntimeError("async-native-boom")

    monkeypatch.setattr(torch_module, "_native_forward", explode)

    energies, forces = _gpuxtb_torch_async(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cuda",
    )
    # Inference reads only after a device barrier: failed floats are NaNs.
    torch.cuda.synchronize()
    assert torch.isnan(energies.cpu()).all()
    assert torch.isnan(forces.cpu()).all()

    # Backward is the synchronization point and surfaces the deferred error.
    with pytest.raises(GPUxtbRuntimeError, match="async-native-boom"):
        energies.sum().backward()

    # The next async submission re-raises the same deferred error.
    with pytest.raises(GPUxtbRuntimeError, match="async-native-boom"):
        _gpuxtb_torch_async(
            positions,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cuda",
        )


@pytest.mark.cuda
def test_async_custom_stream_ordering_is_preserved() -> None:
    """A custom torch stream submits async work whose results are correct."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch
    from gpuxtb import torch as torch_module

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda"
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    args = (
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
    )

    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        energies, forces = _gpuxtb_torch_async(*args, backend="cuda")
    # Wait host-side for the worker (the site backward would wait), then
    # consume the finished results on a different stream.
    torch_module._close_async_engine()
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        consumed = energies.sum()
    torch.cuda.synchronize()
    ref_energy, ref_forces = gpuxtb_torch(*args, backend="cuda")
    assert torch.allclose(consumed.cpu(), ref_energy.cpu(), atol=1.0e-12)
    assert torch.allclose(forces.cpu(), ref_forces.cpu(), atol=1.0e-12)


@pytest.mark.cuda
def test_async_read_before_completion_is_nan() -> None:
    """Not-yet-finished async results read as NaN, never as stale numbers."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda"
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    args = (
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
    )

    # Block the worker with a slow fake job so the real job is still pending.
    from gpuxtb import torch as async_module

    real_run = async_module._AsyncJob.run

    def slow_run(self: object) -> None:
        import time

        time.sleep(1.5)
        real_run(self)  # type: ignore[arg-type]

    async_module._AsyncJob.run = slow_run
    try:
        energies, _ = _gpuxtb_torch_async(*args, backend="cuda")
        # Read immediately (before the worker finishes): the NaN prefill must
        # be visible instead of stale or temporary allocation contents.
        assert torch.isnan(energies.cpu()).all()
    finally:
        async_module._AsyncJob.run = real_run
    # Wait host-side for the worker, then the finished values are non-NaN.
    async_module._close_async_engine()
    torch.cuda.synchronize()
    assert not torch.isnan(energies.cpu()).any()


@pytest.mark.cuda
def test_async_unsupported_autograd_paths_still_raise() -> None:
    """Forces-gradient and higher-order requests stay rejected in async mode."""
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    args = (
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
    )
    _, forces = _gpuxtb_torch_async(*args, backend="cuda")
    with pytest.raises(GPUxtbNotSupportedError, match="forces"):
        (forces**2).sum().backward()

    def energy_sum(values: torch.Tensor) -> torch.Tensor:
        energies, _ = _gpuxtb_torch_async(
            values,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cuda",
        )
        return energies.sum()

    fresh = positions.detach().clone().requires_grad_(True)
    with pytest.raises(GPUxtbNotSupportedError, match="higher-order"):
        torch.autograd.grad(energy_sum(fresh), fresh, create_graph=True)


@pytest.mark.cuda
def test_torch_compile_graph_breaks_for_async_op() -> None:
    """torch.compile around the async op graph-breaks and stays correct.

    The deferred-result contract of ``async_exec`` cannot live inside a fused
    graph, so an async request submitted from a compiled region degrades to the
    synchronous path: correct results, a clean graph break, and no error.
    """
    reason = _cuda_ready()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)

    def loss(p: torch.Tensor, async_exec: bool) -> torch.Tensor:
        entry = _gpuxtb_torch_async if async_exec else gpuxtb_torch
        energies, _ = entry(
            p,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cuda",
        )
        return energies.sum() + (p * 1e-9).sum()

    # Synchronous eager reference; the compiled region submits async_exec=True
    # and must degrade to the same numbers.
    ref = loss(positions, async_exec=False)
    ref_grad = torch.autograd.grad(ref, positions)[0]

    compiled = torch.compile(lambda p: loss(p, async_exec=True))
    out = compiled(positions)
    assert torch.allclose(out, ref, atol=1.0e-12, rtol=1.0e-12)
    out.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, ref_grad, atol=1.0e-12, rtol=1.0e-12)

    # Two async requests in one compiled region: the degrade counter must stay
    # balanced and every call must stay correct.
    def two_losses(p: torch.Tensor) -> torch.Tensor:
        e1, _ = _gpuxtb_torch_async(
            p,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cuda",
        )
        e2, _ = _gpuxtb_torch_async(
            p,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cuda",
        )
        return e1.sum() + e2.sum()

    compiled_two = torch.compile(two_losses)
    assert torch.allclose(
        compiled_two(positions), 2.0 * ref, atol=1.0e-12, rtol=1.0e-12
    )

    # An ordinary eager async call after compilation is still fully async.
    from gpuxtb import torch as torch_module

    q = positions.detach().clone().requires_grad_(True)
    energies, forces = _gpuxtb_torch_async(
        q,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cuda",
    )
    energies.sum().backward()
    assert q.grad is not None
    assert torch.allclose(q.grad, -forces, atol=0.0, rtol=0.0)
    torch_module._close_async_engine()
