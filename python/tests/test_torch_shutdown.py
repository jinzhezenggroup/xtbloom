"""Test lazy cleanup registration without shutting down the pytest process pool.

Loader tests replace every native entry point and restore the existing module's
globals. Real CUDA cleanup is terminal, so its tests run only in watchdog-bound
children; the parent never initializes Torch or CUDA for those tests.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import xtbloom
from xtbloom.exceptions import XTBloomNotSupportedError, XTBloomRuntimeError

_PACKAGE_ROOT = str(Path(xtbloom.__file__).resolve().parents[1])


@dataclass
class _LoaderHarness:
    """Keep fake dispatchers and loader dependencies together for each test."""

    module: ModuleType
    torch: SimpleNamespace
    forward: Mock
    shutdown: Mock
    runtime_load: Mock
    extension_path: Mock
    extension_load: Mock
    register: Mock


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> _LoaderHarness:
    """Isolate loader state without reloading a module with an active pool."""
    module = importlib.import_module("xtbloom.torch")
    forward = Mock(name="forward")
    shutdown = Mock(name="shutdown")
    extension_load = Mock(name="extension_load")
    torch = SimpleNamespace(
        ops=SimpleNamespace(
            load_library=extension_load,
            xtbloom=SimpleNamespace(
                _xtbloom_torch_forward=forward,
                _xtbloom_torch_shutdown=shutdown,
            ),
        )
    )
    runtime_load = Mock(name="runtime_load")
    extension_path = Mock(return_value=Path("/fake/libxtbloom_torch_ext.so"))
    register = Mock(name="atexit_register")
    monkeypatch.setattr(module, "_TORCH_EXT_LOADED", False)
    monkeypatch.setattr(module, "_TORCH_EXT_LOAD_LOCK", threading.Lock())
    monkeypatch.setattr(module, "_TORCH_EXT_LOAD_PID", os.getpid())
    monkeypatch.setattr(module, "_torch", Mock(return_value=torch))
    monkeypatch.setattr(module.library, "load_library", runtime_load)
    monkeypatch.setattr(module, "_torch_extension_path", extension_path)
    monkeypatch.setattr(atexit, "register", register)
    return _LoaderHarness(
        module,
        torch,
        forward,
        shutdown,
        runtime_load,
        extension_path,
        extension_load,
        register,
    )


def test_import_does_not_load_torch_or_native_cuda() -> None:
    """Import succeeds with Torch and native CUDA library loads forbidden."""
    script = r"""
import ctypes
import importlib.abc
import sys

sys.path.insert(0, sys.argv[1])

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("eager Torch import: " + fullname)
        return None

original_cdll = ctypes.CDLL
def guarded_cdll(name, *args, **kwargs):
    lowered = str(name).lower()
    if any(part in lowered for part in ("libcuda", "libxtbloom", "xtbloom_torch_ext")):
        raise AssertionError("eager native load: " + lowered)
    return original_cdll(name, *args, **kwargs)

sys.meta_path.insert(0, RejectTorch())
ctypes.CDLL = guarded_cdll
import xtbloom
import xtbloom.torch as module
assert "torch" not in sys.modules
assert not module._TORCH_EXT_LOADED
assert xtbloom.library._lib is None
print("LAZY_IMPORT_OK", flush=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, _PACKAGE_ROOT],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LAZY_IMPORT_OK" in completed.stdout


def test_lazy_load_registers_captured_shutdown_once(loader: _LoaderHarness) -> None:
    """Publish loaded state only after capturing cleanup for the loaded DSO."""

    def check_not_published(_path: str) -> None:
        assert loader.module._TORCH_EXT_LOADED is False
        loader.runtime_load.assert_called_once_with()
        loader.register.assert_not_called()

    loader.extension_load.side_effect = check_not_published
    for _attempt in range(3):
        assert loader.module._xtbloom_torch_op() is loader.forward

    assert loader.module._TORCH_EXT_LOADED is True
    loader.runtime_load.assert_called_once_with()
    loader.extension_path.assert_called_once_with()
    loader.extension_load.assert_called_once_with("/fake/libxtbloom_torch_ext.so")
    loader.register.assert_called_once_with(loader.shutdown)
    loader.shutdown.assert_not_called()


@pytest.mark.parametrize("failure", ["runtime", "path", "extension", "shutdown"])
def test_failed_load_does_not_publish_or_register(
    loader: _LoaderHarness, failure: str
) -> None:
    """Failed loading remains retryable and never installs a partial cleanup."""
    if failure == "runtime":
        loader.runtime_load.side_effect = XTBloomRuntimeError("runtime unavailable")
    elif failure == "path":
        loader.extension_path.return_value = None
    elif failure == "extension":
        loader.extension_load.side_effect = OSError("incompatible extension")
    else:
        del loader.torch.ops.xtbloom._xtbloom_torch_shutdown

    with pytest.raises((XTBloomRuntimeError, XTBloomNotSupportedError)):
        loader.module._xtbloom_torch_op()
    assert loader.module._TORCH_EXT_LOADED is False
    loader.register.assert_not_called()
    loader.shutdown.assert_not_called()

    loader.runtime_load.side_effect = None
    loader.extension_path.return_value = Path("/fake/libxtbloom_torch_ext.so")
    loader.extension_load.side_effect = None
    loader.torch.ops.xtbloom._xtbloom_torch_shutdown = loader.shutdown
    assert loader.module._xtbloom_torch_op() is loader.forward
    assert loader.module._TORCH_EXT_LOADED is True
    loader.register.assert_called_once_with(loader.shutdown)


def test_failed_registration_does_not_mark_loaded(loader: _LoaderHarness) -> None:
    """An atexit failure cannot publish an extension with no early cleanup."""
    loader.register.side_effect = RuntimeError("registration failed")
    with pytest.raises(RuntimeError, match="registration failed"):
        loader.module._xtbloom_torch_op()
    assert loader.module._TORCH_EXT_LOADED is False
    loader.shutdown.assert_not_called()

    loader.register.side_effect = None
    loader.register.reset_mock()
    assert loader.module._xtbloom_torch_op() is loader.forward
    loader.register.assert_called_once_with(loader.shutdown)


def test_cleanup_uses_captured_native_op_without_reimport(
    loader: _LoaderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown needs neither live module attributes nor another Torch import."""
    loader.module._xtbloom_torch_op()
    captured = loader.register.call_args.args[0]
    unavailable = Mock(side_effect=AssertionError("late loader lookup"))
    monkeypatch.setattr(loader.module, "_torch", unavailable)
    monkeypatch.setattr(loader.module.library, "load_library", unavailable)
    monkeypatch.setattr(loader.module, "_torch_extension_path", unavailable)
    loader.torch.ops.xtbloom._xtbloom_torch_shutdown = unavailable
    captured()
    captured()
    assert loader.shutdown.call_count == 2
    unavailable.assert_not_called()
    loader.extension_load.assert_called_once()
    loader.register.assert_called_once_with(loader.shutdown)


def test_unloaded_child_replaces_inherited_loader_lock(
    loader: _LoaderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulated fork before load never enters a potentially orphaned lock."""
    child_pid = os.getpid() + 1
    inherited_lock = MagicMock()
    inherited_lock.__enter__.side_effect = AssertionError("inherited loader lock")
    monkeypatch.setattr(loader.module, "_TORCH_EXT_LOAD_LOCK", inherited_lock)
    monkeypatch.setattr(loader.module, "os", SimpleNamespace(getpid=lambda: child_pid))

    assert loader.module._xtbloom_torch_op() is loader.forward
    assert child_pid == loader.module._TORCH_EXT_LOAD_PID
    assert loader.module._TORCH_EXT_LOAD_LOCK is not inherited_lock
    inherited_lock.__enter__.assert_not_called()
    loader.register.assert_called_once_with(loader.shutdown)


def test_cpu_preload_does_not_freeze_cleanup_to_loader_pid(
    loader: _LoaderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU-loaded child still invokes native cleanup, which owns PID checks."""
    loader.module._xtbloom_torch_op()
    captured = loader.register.call_args.args[0]
    child_pid = os.getpid() + 1
    cleanup_pids: list[int] = []
    loader.shutdown.side_effect = lambda: cleanup_pids.append(loader.module.os.getpid())
    monkeypatch.setattr(loader.module, "os", SimpleNamespace(getpid=lambda: child_pid))
    inherited_lock = MagicMock()
    inherited_lock.__enter__.side_effect = AssertionError("inherited loader lock")
    monkeypatch.setattr(loader.module, "_TORCH_EXT_LOAD_LOCK", inherited_lock)

    assert loader.module._xtbloom_torch_op() is loader.forward
    captured()
    assert cleanup_pids == [child_pid]
    inherited_lock.__enter__.assert_not_called()
    loader.extension_load.assert_called_once()
    loader.register.assert_called_once_with(loader.shutdown)


class _ObservedLoadLock:
    """Expose actual first-load contention without relying on thread timing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._attempts = 0
        self.contended = threading.Event()

    def __enter__(self) -> None:
        """Signal a second entrant before it waits for the first loader."""
        with self._counter_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.contended.set()
        if not self._lock.acquire(timeout=5):
            raise TimeoutError("loader did not release its lock")

    def __exit__(self, *_exception: object) -> None:
        """Release the loader even when its fake extension raises."""
        self._lock.release()


def test_concurrent_first_load_registers_once(
    loader: _LoaderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent first callers share one load and one cleanup callback."""
    load_started = threading.Event()
    release_load = threading.Event()
    observed_lock = _ObservedLoadLock()
    monkeypatch.setattr(loader.module, "_TORCH_EXT_LOAD_LOCK", observed_lock)

    def wait_for_release(_path: str) -> None:
        load_started.set()
        assert release_load.wait(timeout=5), "test did not release extension load"

    loader.extension_load.side_effect = wait_for_release
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(loader.module._xtbloom_torch_op)
        try:
            assert load_started.wait(timeout=5), "first loader did not start"
            second = executor.submit(loader.module._xtbloom_torch_op)
            assert observed_lock.contended.wait(timeout=5), "second loader did not wait"
        finally:
            release_load.set()
        assert first.result(timeout=5) is loader.forward
        assert second.result(timeout=5) is loader.forward

    loader.runtime_load.assert_called_once_with()
    loader.extension_load.assert_called_once()
    loader.register.assert_called_once_with(loader.shutdown)


_CUDA_SHUTDOWN_SCRIPT = r"""
import atexit
import ctypes
import gc
import importlib
import os
import sys
import threading

sys.path.insert(0, sys.argv[1])
scenario = sys.argv[2]

def unavailable(reason):
    print("SHUTDOWN_SKIP: " + reason, flush=True)
    raise SystemExit(77)

try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    unavailable("optional Torch dependency is not installed")
import numpy as np
from xtbloom import Context, library, xtbloom_torch
from xtbloom import _dlpack
from xtbloom.exceptions import XTBloomRuntimeError

main_thread = threading.get_ident()
automatic_cleanup = {}

def require_cuda():
    if not torch.cuda.is_available():
        unavailable("Torch has no usable CUDA device")
    try:
        with Context("cuda"):
            pass
    except XTBloomRuntimeError as error:
        if (error.status == library.STATUS_BACKEND_UNAVAILABLE
                and "built without CUDA support" in error.message):
            unavailable("xTBloom was built without CUDA support")
        raise

def inputs():
    positions = torch.tensor([
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ], dtype=torch.float64, device="cuda")
    return (
        positions,
        torch.from_numpy(np.array([8, 1, 1], dtype=np.int32)),
        torch.from_numpy(np.array([0, 3], dtype=np.int64)),
        torch.from_numpy(np.array([0.0], dtype=np.float64)),
        torch.from_numpy(np.array([0], dtype=np.int32)),
        torch.from_numpy(np.array([1], dtype=np.int32)),
    )

def warm():
    stream = torch.cuda.Stream()
    warm_inputs = inputs()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        energies, forces = xtbloom_torch(*warm_inputs, backend="cuda")
    stream.synchronize()
    reference = (energies.cpu().clone(), forces.cpu().clone())
    assert all(torch.isfinite(value).all().item() for value in reference)
    return stream, warm_inputs, (energies, forces), reference

def check_outputs(outputs, reference):
    assert torch.allclose(outputs[0].cpu(), reference[0], atol=1e-9, rtol=1e-9)
    assert torch.allclose(outputs[1].cpu(), reference[1], atol=1e-8, rtol=1e-8)

def check_terminal():
    try:
        xtbloom_torch(*inputs(), backend="cuda")
    except RuntimeError as error:
        assert "shutting down" in str(error), str(error)
    else:
        raise AssertionError("a terminal pool accepted another CUDA submission")

class DeleterProbe:
    '''Keep callback code alive, but never retain the adopted consumer tensor.'''

    def __init__(self, block=False):
        self.calls = []
        self.errors = []
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event() if block else None

    def adopt(self, positions):
        '''Wrap one legacy capsule and preserve its original native deleter.'''
        self.capsule = positions.__dlpack__()
        assert _dlpack._pyapi.PyCapsule_IsValid(self.capsule, _dlpack._CAPSULE_NAME)
        pointer = _dlpack._pyapi.PyCapsule_GetPointer(
            self.capsule, _dlpack._CAPSULE_NAME
        )
        managed = ctypes.cast(pointer, ctypes.POINTER(_dlpack._DLManagedTensor))
        assert managed.contents.deleter
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        self.original = prototype(managed.contents.deleter)

        @prototype
        def callback(managed_pointer):
            if self.calls:
                self.errors.append("duplicate DLPack deleter invocation")
                return
            self.calls.append(threading.get_ident())
            self.entered.set()
            try:
                if self.release is not None:
                    assert self.release.wait(10), "deleter release gate timed out"
            except BaseException as error:
                self.errors.append(repr(error))
            finally:
                try:
                    self.original(managed_pointer)
                except BaseException as error:
                    self.errors.append(repr(error))
                self.finished.set()

        self.callback = callback
        managed.contents.deleter = ctypes.cast(callback, ctypes.c_void_p).value
        return torch.from_dlpack(self.capsule)

    def check_drained(self):
        assert self.finished.is_set(), "shutdown returned before the Python deleter"
        assert not self.errors, self.errors
        assert len(self.calls) == 1, self.calls
        assert self.calls[0] != main_thread, "deleter did not run on the reaper"

def pending(stream, warm_inputs, probe=None, sleep_cycles=500_000_000):
    '''Reuse every topology identity so first-plan creation cannot drain sleep.'''
    cuda_sleep = getattr(torch.cuda, "_sleep", None)
    if cuda_sleep is None:
        unavailable("Torch CUDA sleep primitive is unavailable")
    positions = inputs()[0]
    if probe is not None:
        positions = probe.adopt(positions)
    pending_inputs = (positions, *warm_inputs[1:])
    torch.cuda.synchronize()
    completion = torch.cuda.Event()
    with torch.cuda.stream(stream):
        cuda_sleep(sleep_cycles)
        outputs = xtbloom_torch(*pending_inputs, backend="cuda")
        completion.record()
    return pending_inputs, outputs, completion

def check_pending(completion, probe):
    gc.collect()
    assert not completion.query(), "pending-cleanup precondition was not exercised"
    if probe is not None:
        assert not probe.finished.is_set(), "deleter ran before the cleanup call"

def after_automatic_cleanup():
    if not automatic_cleanup:
        return
    automatic_cleanup["probe"].check_drained()
    check_outputs(automatic_cleanup["outputs"], automatic_cleanup["reference"])
    check_terminal()
    print("ATEXIT_DRAIN_OK:" + scenario, flush=True)

def concurrent_cleanup(shutdown, probe):
    start = threading.Barrier(3)
    counter_lock = threading.Lock()
    both_started = threading.Event()
    returned = threading.Event()
    started = []
    errors = []

    def close():
        try:
            start.wait(timeout=5)
            with counter_lock:
                started.append(threading.get_ident())
                if len(started) == 2:
                    both_started.set()
            shutdown()
            probe.check_drained()
        except BaseException as error:
            errors.append(repr(error))
        finally:
            returned.set()

    workers = [threading.Thread(target=close) for _attempt in range(2)]
    for worker in workers:
        worker.start()
    try:
        start.wait(timeout=5)
        assert both_started.wait(5), "cleanup callers did not start"
        assert probe.entered.wait(10), "reaper did not enter the Python deleter"
        returned.wait(timeout=0.25)
    finally:
        probe.release.set()
        for worker in workers:
            worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers), "cleanup did not join"
    assert not errors, errors
    probe.check_drained()
    assert probe.calls[0] not in started, "deleter ran on a cleanup caller"
    print("CONCURRENT_DRAIN_OK", flush=True)

class DLPackOnly:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def __dlpack__(self, *args, **kwargs):
        return self.tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __array__(self, *args, **kwargs):
        raise AssertionError("external device topology must not be copied to NumPy")

def backward_history():
    stream = torch.cuda.Stream()
    packed = inputs()
    positions = packed[0].requires_grad_(True)
    owners = tuple(tensor.to("cuda") for tensor in packed[1:])
    wrappers = tuple(DLPackOnly(tensor) for tensor in owners)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        good_energy, good_forces = xtbloom_torch(positions, *wrappers, backend="cuda")
        good_loss = good_energy.sum()
        good_loss.backward(retain_graph=True)
    stream.synchronize()
    reference_forces = good_forces.detach().cpu().clone()
    positions.grad = None
    good_token = good_energy.grad_fn._xtbloom_submission_id
    changed_numbers = torch.tensor([1, 8, 1], dtype=torch.int32, device="cuda")
    failed_positions = positions.detach().requires_grad_(True)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        owners[0].copy_(changed_numbers)
        bad_energy, _forces = xtbloom_torch(
            failed_positions, *wrappers, backend="cuda"
        )
    bad_token = bad_energy.grad_fn._xtbloom_submission_id
    assert good_token > 0 and bad_token > 0 and good_token != bad_token
    shutdown = torch.ops.xtbloom._xtbloom_torch_shutdown
    shutdown()
    shutdown()

    with torch.cuda.stream(stream):
        good_loss.backward()
    stream.synchronize()
    assert torch.allclose(positions.grad.cpu(), -reference_forces, atol=1e-8, rtol=1e-8)
    diagnostics = []
    for _attempt in range(2):
        try:
            with torch.cuda.stream(stream):
                bad_energy.sum().backward(retain_graph=True)
        except RuntimeError as error:
            diagnostic = str(error)
            assert "CUDA submission failed" in diagnostic, diagnostic
            assert "fixed CUDA plan topology" in diagnostic, diagnostic
            diagnostics.append(diagnostic)
        else:
            raise AssertionError("post-close backward lost its exact failure")
    assert diagnostics[0] == diagnostics[1], diagnostics
    torch.ops.xtbloom._xtbloom_torch_wait(good_token)
    check_terminal()
    print("BACKWARD_HISTORY_OK", flush=True)
    print("SHUTDOWN_OK:" + scenario, flush=True)

def wait_child(child):
    waited, status = os.waitpid(child, 0)
    assert waited == child
    code = os.waitstatus_to_exitcode(status)
    if code == 77:
        unavailable("fork child lacks a required CUDA dependency; see child output")
    assert code == 0, "fork child process status: " + str(code)

def cpu_preload_fork():
    module = importlib.import_module("xtbloom.torch")
    assert not torch.cuda.is_initialized()
    module._xtbloom_torch_op()
    assert module._CUDA_PROCESS_ID is None
    assert not torch.cuda.is_initialized(), "CPU preload initialized Torch CUDA"
    print("CPU_PRELOAD_BEFORE_FORK", flush=True)
    child = os.fork()
    if child == 0:
        require_cuda()
        stream, warm_inputs, _outputs, reference = warm()
        probe = DeleterProbe()
        pending_inputs, outputs, completion = pending(stream, warm_inputs, probe)
        del pending_inputs, warm_inputs
        check_pending(completion, probe)
        automatic_cleanup.update(probe=probe, outputs=outputs, reference=reference)
        print("RETURNING_WITH_PENDING:" + scenario, flush=True)
        sys.exit(0)
    wait_child(child)
    assert not torch.cuda.is_initialized(), "child initialization escaped into parent"
    require_cuda()
    stream, warm_inputs, outputs, reference = warm()
    check_outputs(outputs, reference)
    shutdown = torch.ops.xtbloom._xtbloom_torch_shutdown
    shutdown()
    shutdown()
    check_terminal()
    print("PARENT_USABLE:" + scenario, flush=True)
    print("SHUTDOWN_OK:" + scenario, flush=True)

def main():
    if scenario.startswith("fork-") and not hasattr(os, "fork"):
        unavailable("real fork regression requires os.fork")
    if scenario == "fork-cpu-preload":
        cpu_preload_fork()
        return
    require_cuda()
    if scenario == "backward-history":
        backward_history()
        return
    stream, warm_inputs, outputs, reference = warm()
    shutdown = torch.ops.xtbloom._xtbloom_torch_shutdown
    if scenario == "fork-inherited":
        child = os.fork()
        if child == 0:
            shutdown()
            shutdown()
            print("INHERITED_CHILD_SHUTDOWN_OK", flush=True)
            sys.exit(0)
        wait_child(child)
        with torch.cuda.stream(stream):
            outputs = xtbloom_torch(*warm_inputs, backend="cuda")
        stream.synchronize()
        check_outputs(outputs, reference)
        print("PARENT_USABLE:" + scenario, flush=True)
    probe = None
    if scenario in ("dlpack-deleter", "concurrent", "atexit-pending"):
        probe = DeleterProbe(block=scenario == "concurrent")
    if scenario in ("pending", "dlpack-deleter", "concurrent", "atexit-pending"):
        cycles = 2_000_000_000 if scenario == "concurrent" else 500_000_000
        pending_inputs, outputs, completion = pending(
            stream, warm_inputs, probe, cycles
        )
        del pending_inputs, warm_inputs
        check_pending(completion, probe)
    if scenario == "atexit-pending":
        automatic_cleanup.update(probe=probe, outputs=outputs, reference=reference)
        print("RETURNING_WITH_PENDING:" + scenario, flush=True)
        return

    print("BEFORE_SHUTDOWN:" + scenario, flush=True)
    if scenario == "concurrent":
        concurrent_cleanup(shutdown, probe)
    else:
        shutdown()
        shutdown()
    print("AFTER_SHUTDOWN:" + scenario, flush=True)
    if probe is not None:
        probe.check_drained()
    check_outputs(outputs, reference)
    check_terminal()
    print("SHUTDOWN_OK:" + scenario, flush=True)

atexit.register(after_automatic_cleanup)
main()
"""


def test_lifecycle_subprocess_script_compiles() -> None:
    """Check every subprocess branch's syntax without importing Torch or CUDA."""
    compile(_CUDA_SHUTDOWN_SCRIPT, "<torch-shutdown-child>", "exec")


def _run_lifecycle_subprocess(scenario: str) -> subprocess.CompletedProcess[str]:
    """Bound the complete process tree, including normal-exit fork children."""
    arguments = [
        sys.executable,
        "-c",
        _CUDA_SHUTDOWN_SCRIPT,
        _PACKAGE_ROOT,
        scenario,
    ]
    with subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            stdout, stderr = child.communicate(timeout=10)
            pytest.fail(f"shutdown scenario {scenario} timed out:\n{stdout}{stderr}")
        return subprocess.CompletedProcess(arguments, child.returncode, stdout, stderr)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "scenario",
    [
        "idle",
        "pending",
        "dlpack-deleter",
        "concurrent",
        "backward-history",
        "fork-inherited",
        "fork-cpu-preload",
        "atexit-pending",
    ],
)
def test_cuda_shutdown_in_subprocess(scenario: str) -> None:
    """Exercise native drain, history, fork, GIL release, and actual atexit."""
    completed = _run_lifecycle_subprocess(scenario)
    output = completed.stdout + completed.stderr
    if completed.returncode == 77 and "SHUTDOWN_SKIP: " in completed.stdout:
        pytest.skip(completed.stdout.strip())
    assert completed.returncode == 0, output
    if scenario in {"fork-cpu-preload", "atexit-pending"}:
        assert f"RETURNING_WITH_PENDING:{scenario}" in completed.stdout, output
        assert f"ATEXIT_DRAIN_OK:{scenario}" in completed.stdout, output
    if scenario != "atexit-pending":
        assert f"SHUTDOWN_OK:{scenario}" in completed.stdout, output
    if scenario.startswith("fork-"):
        assert f"PARENT_USABLE:{scenario}" in completed.stdout, output
    if scenario == "fork-inherited":
        assert "INHERITED_CHILD_SHUTDOWN_OK" in completed.stdout, output
    if scenario == "fork-cpu-preload":
        assert "CPU_PRELOAD_BEFORE_FORK" in completed.stdout, output
    if scenario == "backward-history":
        assert "BACKWARD_HISTORY_OK" in completed.stdout, output
    if scenario == "concurrent":
        assert "CONCURRENT_DRAIN_OK" in completed.stdout, output
