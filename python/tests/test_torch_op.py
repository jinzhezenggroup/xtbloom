"""Tests for the PyTorch autograd op :func:`gpuxtb.gpuxtb_torch`.

The op runs packed inference on PyTorch tensors through the compiled stable-ABI
extension and
exposes a single analytic gradient, ``dE/dR = -F``.  These tests verify the
host (CPU) path, the exact gradient identity against the returned forces, the
ragged-batch gradient slicing, a finite-difference cross-check of the energy
gradient, and the hard errors for every unsupported autograd direction.
CUDA coverage is gated on a real GPU plus a torch CUDA build, mirroring
``test_array_batch_cuda.py``.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from gpuxtb import Calculator, gpuxtb_torch
from gpuxtb.exceptions import GPUxtbNotSupportedError, GPUxtbValueError

_TORCH = importlib.util.find_spec("torch")


def test_torch_public_signature_hides_execution_details() -> None:
    """Users select Torch execution context without raw stream/async controls."""
    import gpuxtb.torch as torch_module

    parameters = inspect.signature(gpuxtb_torch).parameters
    assert "stream" not in parameters
    assert "async_exec" not in parameters
    assert not hasattr(torch_module, "_gpuxtb_torch_async")


def test_output_allocation_is_failure_safe() -> None:
    """Caller-visible outputs start as NaN until native publication succeeds."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch
    from gpuxtb import torch as torch_module

    energies, forces = torch_module._allocate_outputs(torch, "cpu", 2, 3)
    assert torch.isnan(energies).all()
    assert torch.isnan(forces).all()


def test_compiled_schema_marks_outputs_mutable() -> None:
    """The dispatcher must know that native execution writes both outputs."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    if sys.platform != "linux":
        pytest.skip("the vendored stable-ABI extension is currently Linux-only")
    from gpuxtb import torch as torch_module

    schema = str(torch_module._gpuxtb_torch_op().default._schema)
    assert "Tensor(a!) out_energies" in schema
    assert "Tensor(b!) out_forces" in schema
    assert "-> (Tensor(a!), Tensor(b!))" in schema


class _DLPackOnly:
    """Expose a tensor through DLPack while forbidding NumPy conversion."""

    def __init__(self, tensor: object) -> None:
        self._tensor = tensor
        self.shape = tensor.shape

    def __dlpack__(self, *args: object, **kwargs: object) -> object:
        """Delegate DLPack export to the wrapped tensor."""
        return self._tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        """Delegate DLPack device discovery to the wrapped tensor."""
        return self._tensor.__dlpack_device__()

    def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
        """Reject the implicit host conversion used by the original bug."""
        del dtype, copy
        raise AssertionError("DLPack-only input must not be converted by NumPy")


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


WATER_NUMBERS = np.array([8, 1, 1], dtype=np.int32)
WATER_POSITIONS = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ],
    dtype=np.float64,
)

# Carbon at the origin, four tetrahedral hydrogens at ~2.06 bohr.
_METHANE_D = 2.06 / np.sqrt(3.0)
METHANE_NUMBERS = np.array([6, 1, 1, 1, 1], dtype=np.int32)
METHANE_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [_METHANE_D, _METHANE_D, _METHANE_D],
        [_METHANE_D, -_METHANE_D, -_METHANE_D],
        [-_METHANE_D, _METHANE_D, -_METHANE_D],
        [-_METHANE_D, -_METHANE_D, _METHANE_D],
    ],
    dtype=np.float64,
)


def _packed(
    numbers: list[np.ndarray], positions: list[np.ndarray], torch: object
) -> dict[str, object]:
    """Pack a ragged batch into descriptor arrays as torch tensors."""
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
        )
        for name, value in arrays.items()
    }


def test_forward_matches_calculator_host() -> None:
    """The op's energies and forces must reproduce ``Calculator`` exactly."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    assert energies.shape == (1,)
    assert forces.shape == (3, 3)
    ref = Calculator("GFN2-xTB", WATER_NUMBERS, WATER_POSITIONS).singlepoint()
    assert torch.allclose(
        energies,
        torch.tensor([ref.energy], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        forces,
        torch.from_numpy(np.ascontiguousarray(ref.forces)),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_backward_grad_equals_neg_forces() -> None:
    """The analytic gradient of the summed energy must be exactly ``-F``."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


def test_energy_backward_does_not_scan_unused_force_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An energy-only loss must not materialize and scan a zero force grad."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, _ = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )

    def reject_any(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("energy backward must not scan an unused force grad")

    monkeypatch.setattr(torch.Tensor, "any", reject_any)
    energies.sum().backward()
    assert positions.grad is not None


def test_numpy_auxiliary_arrays_dispatch_as_tensors() -> None:
    """Every documented NumPy auxiliary reaches the compiled Tensor schema."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    auxiliary = {
        name: value.detach().cpu().numpy()
        for name, value in arrays.items()
        if name != "positions"
    }
    energies, forces = gpuxtb_torch(
        positions,
        auxiliary["atomic_numbers"],
        auxiliary["atom_offsets"],
        auxiliary["molecular_charges"],
        auxiliary["unpaired_electrons"],
        auxiliary["spin_channels"],
        backend="cpu",
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


def test_dlpack_only_auxiliaries_survive_backward() -> None:
    """Every non-Torch DLPack auxiliary stays native through dispatch."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = gpuxtb_torch(
        positions,
        _DLPackOnly(arrays["atomic_numbers"]),
        _DLPackOnly(arrays["atom_offsets"]),
        _DLPackOnly(arrays["molecular_charges"]),
        _DLPackOnly(arrays["unpaired_electrons"]),
        _DLPackOnly(arrays["spin_channels"]),
        backend="cpu",
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


def test_native_loader_honors_gpuxtb_library_override() -> None:
    """The extension must not replace an explicit native runtime with its neighbor."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    if sys.platform != "linux":
        pytest.skip("the vendored stable-ABI extension is currently Linux-only")

    import torch
    from gpuxtb import torch as torch_module

    extension = torch_module._torch_extension_path()
    assert extension is not None, "compiled Torch extension is missing"
    torch_cpu = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
    if not torch_cpu.is_file():
        pytest.skip("installed torch does not expose libtorch_cpu.so")

    # Load the extension directly in a fresh process so both its once-only API
    # table and torch's operator registry start clean. libtorch_cpu is a valid
    # DSO but not a gpuxtb ABI implementation: honoring the override must fail
    # closed instead of silently opening the extension-adjacent libgpuxtb.
    script = """
import sys
import torch

torch.ops.load_library(sys.argv[1])
positions = torch.zeros((1, 3), dtype=torch.float64)
try:
    torch.ops.gpuxtb.gpuxtb_torch_forward(
        positions,
        torch.ones(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.empty(1, dtype=torch.float64),
        torch.empty((1, 3), dtype=torch.float64),
        1, -1, 0, 0, 50, 1.0e-6, 1.0e-8, 300.0,
    )
except RuntimeError as exc:
    if "cannot load libgpuxtb" not in str(exc):
        raise
else:
    raise AssertionError("GPUXTB_LIBRARY was ignored by the torch extension")
"""
    env = os.environ.copy()
    env["GPUXTB_LIBRARY"] = str(torch_cpu)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(extension)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_torch_entrypoint_preloads_native_runtime_providers() -> None:
    """A fresh process may call gpuxtb_torch before any ctypes-backed API."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    script = """
import numpy as np
import torch
from gpuxtb import gpuxtb_torch

positions = torch.tensor(
    [[0.0, 0.0, -0.73578586109551],
     [1.44183152868459, 0.0, 0.36789293054775],
     [-1.44183152868459, 0.0, 0.36789293054775]],
    dtype=torch.float64,
)
energies, forces = gpuxtb_torch(
    positions,
    np.array([8, 1, 1], dtype=np.int32),
    np.array([0, 3], dtype=np.int64),
    np.zeros(1, dtype=np.float64),
    np.zeros(1, dtype=np.int32),
    np.ones(1, dtype=np.int32),
    backend="cpu",
)
assert torch.isfinite(energies).all()
assert torch.isfinite(forces).all()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stderr


def test_energy_gradient_finite_difference() -> None:
    """A central-difference dE/dR must match the op's analytic gradient."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    h = 1.0e-3
    positions = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)

    def energy_at(p: torch.Tensor) -> torch.Tensor:
        values, _ = gpuxtb_torch(
            p,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
            charge_tolerance=1.0e-8,
            energy_tolerance=1.0e-12,
        )
        return values[0]

    analytic_positions = positions.detach().clone().requires_grad_(True)
    analytic = torch.autograd.grad(energy_at(analytic_positions), analytic_positions)[0]
    numerical = torch.empty_like(positions)
    for index in itertools.product(range(positions.shape[0]), range(3)):
        plus = positions.clone()
        minus = positions.clone()
        plus[index] += h
        minus[index] -= h
        numerical[index] = (energy_at(plus) - energy_at(minus)) / (2.0 * h)
    assert torch.allclose(numerical, analytic, atol=2.0e-3, rtol=1.0e-2)


def test_batch_gradient_selects_its_system() -> None:
    """Backprop through one system's energy must not move the other system."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        np.concatenate([WATER_POSITIONS, METHANE_POSITIONS], axis=0).tolist(),
        dtype=torch.float64,
        requires_grad=True,
    )
    arrays = _packed(
        [WATER_NUMBERS, METHANE_NUMBERS],
        [WATER_POSITIONS, METHANE_POSITIONS],
        torch,
    )
    energies, _ = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    energies[0].backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad[3:], torch.zeros_like(positions.grad[3:]))
    assert not torch.allclose(positions.grad[:3], torch.zeros_like(positions.grad[:3]))


def test_nonposition_requires_grad_raises() -> None:
    """Autograd on any non-positions input must fail eagerly."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    arrays["molecular_charges"] = torch.zeros(
        1, dtype=torch.float64, requires_grad=True
    )
    with pytest.raises(GPUxtbNotSupportedError, match="molecular_charges"):
        gpuxtb_torch(
            positions,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
        )


def test_grad_through_forces_raises() -> None:
    """A loss depending on the forces output needs the Hessian and must fail."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    _, forces = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    with pytest.raises(GPUxtbNotSupportedError, match="forces"):
        (forces**2).sum().backward()


def test_zero_grad_through_forces_raises() -> None:
    """A real force-output gradient is rejected even when its value is zero."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    _, forces = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    with pytest.raises(GPUxtbNotSupportedError, match="forces"):
        forces.backward(torch.zeros_like(forces))


def test_higher_order_gradient_raises() -> None:
    """Hessian requests fail instead of silently returning zero curvature."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)

    def total_energy(values: torch.Tensor) -> torch.Tensor:
        energies, _ = gpuxtb_torch(
            values,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
        )
        return energies.sum()

    direct_positions = positions.clone().requires_grad_(True)
    with pytest.raises(GPUxtbNotSupportedError, match="higher-order"):
        torch.autograd.grad(
            total_energy(direct_positions), direct_positions, create_graph=True
        )
    with pytest.raises(GPUxtbNotSupportedError, match="higher-order"):
        torch.autograd.functional.hessian(total_energy, positions)


def test_noncontiguous_positions_copied() -> None:
    """A non-contiguous view of positions is packed, not read wrongly."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    water = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    noncontiguous = water.t()
    assert not noncontiguous.is_contiguous()
    positions = noncontiguous.requires_grad_(True)
    transposed_geometry = noncontiguous.detach().numpy()
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = gpuxtb_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    ref = Calculator("GFN2-xTB", WATER_NUMBERS, transposed_geometry).singlepoint()
    assert torch.allclose(
        energies,
        torch.tensor([ref.energy], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=1.0e-10, rtol=1.0e-10)


def test_positions_must_be_float64() -> None:
    """Non-float64 or non-tensor positions are rejected up front."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    positions32 = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float32)
    with pytest.raises(GPUxtbValueError, match="float64"):
        gpuxtb_torch(
            positions32,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
        )


def test_torch_compile_graph_breaks_for_sync_op() -> None:
    """torch.compile around the op graph-breaks and stays correct (no error).

    gpuxtb_torch is eager-only: it drives the native library through a compiled
    stable-ABI custom op, which Dynamo cannot trace, so the op is marked opaque
    and ``torch.compile`` inserts a graph break. There is no compilation
    speedup for the gpuxtb call, but compilation must never fail at trace time.
    """
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)

    def loss(p: torch.Tensor) -> torch.Tensor:
        energies, _ = gpuxtb_torch(
            p,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
        )
        return energies.sum() + (p * 1e-9).sum()

    eager = loss(positions)
    eager_grad = torch.autograd.grad(eager, positions)[0]
    compiled = torch.compile(loss)
    out = compiled(positions)
    assert torch.allclose(out, eager, atol=1.0e-12, rtol=1.0e-12)
    compiled(positions).sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, eager_grad, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(compiled(positions), eager, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.cuda
def test_torch_auto_all_host_passes_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AUTO forwards the active Torch stream even when every tensor is host."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import gpuxtb.torch as torch_module
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")

    recorded: list[int] = []

    def fake_native_forward(**kwargs: object) -> tuple[object, object]:
        recorded.append(int(kwargs["stream"]))
        return kwargs["out_energies"], kwargs["out_forces"]

    monkeypatch.setattr(torch_module, "_native_forward", fake_native_forward)
    host_arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpuxtb_torch(
            torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64),
            host_arrays["atomic_numbers"],
            host_arrays["atom_offsets"],
            host_arrays["molecular_charges"],
            host_arrays["unpaired_electrons"],
            host_arrays["spin_channels"],
            backend="auto",
        )
    assert recorded == [int(stream.cuda_stream)]


@pytest.mark.cuda
def test_torch_auto_all_host_keeps_cpu_only_fallback() -> None:
    """A candidate Torch stream does not disable native AUTO-to-CPU fallback."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")
    if _library_has_cuda():
        pytest.skip("requires a CPU-only gpuxtb build")

    host_arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        energies, forces = gpuxtb_torch(
            torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64),
            host_arrays["atomic_numbers"],
            host_arrays["atom_offsets"],
            host_arrays["molecular_charges"],
            host_arrays["unpaired_electrons"],
            host_arrays["spin_channels"],
            backend="auto",
        )
    assert torch.isfinite(energies).all()
    assert torch.isfinite(forces).all()


@pytest.mark.cuda
def test_torch_cuda_matches_host() -> None:
    """CUDA tensors through the op give the same energies, forces, and dR."""
    reason = _skip_reason() if _library_has_cuda() else "CUDA backend unavailable"
    if reason:
        pytest.skip(reason)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")

    host_positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    host_arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    host_energies, host_forces = gpuxtb_torch(
        host_positions,
        host_arrays["atomic_numbers"],
        host_arrays["atom_offsets"],
        host_arrays["molecular_charges"],
        host_arrays["unpaired_electrons"],
        host_arrays["spin_channels"],
        backend="cpu",
    )

    cuda_positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    cuda_arrays = {
        name: value.to("cuda")
        for name, value in _packed([WATER_NUMBERS], [WATER_POSITIONS], torch).items()
    }
    cuda_energies, cuda_forces = gpuxtb_torch(
        cuda_positions,
        cuda_arrays["atomic_numbers"],
        cuda_arrays["atom_offsets"],
        cuda_arrays["molecular_charges"],
        cuda_arrays["unpaired_electrons"],
        cuda_arrays["spin_channels"],
        backend="cuda",
    )
    cuda_energies.sum().backward()
    assert cuda_positions.grad is not None
    assert cuda_positions.grad.device.type == "cuda"
    assert torch.allclose(cuda_energies.cpu(), host_energies, atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(cuda_forces.cpu(), host_forces, atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(cuda_positions.grad, -cuda_forces, atol=0.0, rtol=0.0)


@pytest.mark.cuda
def test_torch_cuda_accepts_host_auxiliary_descriptors() -> None:
    """CUDA positions may retain topology and charge tensors on the host."""
    reason = _skip_reason() if _library_has_cuda() else "CUDA backend unavailable"
    if reason:
        pytest.skip(reason)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")

    host_positions = torch.tensor(WATER_POSITIONS.tolist(), dtype=torch.float64)
    host_arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    host_energies, host_forces = gpuxtb_torch(
        host_positions,
        host_arrays["atomic_numbers"],
        host_arrays["atom_offsets"],
        host_arrays["molecular_charges"],
        host_arrays["unpaired_electrons"],
        host_arrays["spin_channels"],
        backend="cpu",
    )

    mixed_positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda", requires_grad=True
    )
    mixed_energies, mixed_forces = gpuxtb_torch(
        mixed_positions,
        host_arrays["atomic_numbers"],
        host_arrays["atom_offsets"],
        host_arrays["molecular_charges"],
        host_arrays["unpaired_electrons"],
        host_arrays["spin_channels"],
        backend="cuda",
    )
    mixed_energies.sum().backward()
    assert mixed_positions.grad is not None
    assert torch.allclose(mixed_energies.cpu(), host_energies, atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(mixed_forces.cpu(), host_forces, atol=1.0e-9, rtol=1.0e-9)
    assert torch.allclose(mixed_positions.grad, -mixed_forces, atol=0.0, rtol=0.0)


@pytest.mark.cuda
def test_torch_cuda_uses_current_stream() -> None:
    """Raw tensor pointers remain ordered on Torch's active custom stream."""
    reason = _skip_reason() if _library_has_cuda() else "CUDA backend unavailable"
    if reason:
        pytest.skip(reason)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")
    cuda_sleep = getattr(torch.cuda, "_sleep", None)
    if cuda_sleep is None:
        pytest.skip("torch CUDA sleep primitive is unavailable")

    host_arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    geometry_b = WATER_POSITIONS * 1.15
    reference_energies, _ = gpuxtb_torch(
        torch.tensor(geometry_b.tolist(), dtype=torch.float64),
        host_arrays["atomic_numbers"],
        host_arrays["atom_offsets"],
        host_arrays["molecular_charges"],
        host_arrays["unpaired_electrons"],
        host_arrays["spin_channels"],
        backend="cpu",
    )

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, device="cuda"
    )
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        cuda_sleep(100_000_000)
        positions.copy_(
            torch.tensor(geometry_b.tolist(), dtype=torch.float64, device="cuda")
        )
        energies, _ = gpuxtb_torch(
            positions,
            host_arrays["atomic_numbers"],
            host_arrays["atom_offsets"],
            host_arrays["molecular_charges"],
            host_arrays["unpaired_electrons"],
            host_arrays["spin_channels"],
            backend="cuda",
        )
    producer.synchronize()
    assert torch.allclose(
        energies.cpu(),
        reference_energies,
        atol=1.0e-9,
        rtol=1.0e-9,
    )
