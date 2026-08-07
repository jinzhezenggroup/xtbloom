"""CUDA device-array tests for :class:`gpuxtb.ArrayBatch`.

These tests exercise the zero-copy DLPack path with real CUDA device buffers
from PyTorch/CuPy when both the CUDA backend and a provider library are
available; they skip otherwise so CPU-only installations stay green.  The
provider libraries are opt-in test dependencies and are never imported by
gpuxtb runtime code.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from _cases import case_by_id, structure_inputs
from gpuxtb import ArrayBatch, Calculator
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

# --- environment probes ---------------------------------------------------------

_TORCH = importlib.util.find_spec("torch")
_CUPY = importlib.util.find_spec("cupy")
_JAX = importlib.util.find_spec("jax")


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


def _provider() -> str | None:
    """Return the first available device-array provider name (or ``None``)."""
    if _TORCH is not None:
        return "torch"
    if _CUPY is not None:
        return "cupy"
    return None


def _device_ready() -> str | None:
    """Return a skip reason when CUDA device-array tests cannot run."""
    if not _library_has_cuda():
        return "CUDA backend is not available on this host"
    if _provider() is None:
        return "no device-array provider (torch/cupy) is installed"
    return None


def _wrap(torch: object, name: str, value: np.ndarray) -> object:
    """Convert a host numpy array to a CUDA tensor."""
    mapping = {
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.uint8): torch.uint8,
    }
    dtype = mapping[value.dtype]
    return torch.tensor(value.tolist(), dtype=dtype, device="cuda")


def _water_arrays(torch: object) -> dict[str, object]:
    """Packed water descriptors, all on the CUDA device."""
    numbers = np.array([8, 1, 1], dtype=np.int32)
    positions = np.array(
        [
            [0.0000000000, 0.0000000000, -0.7357858611],
            [1.4418315287, 0.0000000000, 0.3678929305],
            [-1.4418315287, 0.0000000000, 0.3678929305],
        ],
        dtype=np.float64,
    )
    return {
        "atom_offsets": _wrap(torch, "atom_offsets", np.array([0, 3], np.int64)),
        "atomic_numbers": _wrap(torch, "atomic_numbers", numbers),
        "positions": _wrap(torch, "positions", positions),
        "molecular_charges": _wrap(torch, "molecular_charges", np.array([0.0])),
        "unpaired_electrons": _wrap(
            torch, "unpaired_electrons", np.array([0], np.int32)
        ),
    }


def _water_host_arrays() -> dict[str, np.ndarray]:
    """Packed host descriptors for CUDA stream-state tests."""
    return {
        "atom_offsets": np.array([0, 3], dtype=np.int64),
        "atomic_numbers": np.array([8, 1, 1], dtype=np.int32),
        "positions": np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ],
            dtype=np.float64,
        ),
        "molecular_charges": np.array([0.0], dtype=np.float64),
        "unpaired_electrons": np.array([0], dtype=np.int32),
    }


@pytest.mark.cuda
def test_torch_device_arrays_match_host_cuda() -> None:
    """Native torch CUDA tensors must reproduce the host-descriptor CUDA run."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    reference = Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ]
        ),
        backend="cuda",
    ).singlepoint()
    arrays = _water_arrays(torch)
    result = ArrayBatch(**arrays, backend="cuda").compute()
    assert abs(float(result.energies.item()) - reference.energy) < 1.0e-10
    # Default output policy: results are host numpy arrays, even for device
    # inputs.
    assert isinstance(result.forces, np.ndarray)
    assert np.allclose(result.forces, reference.forces, atol=1.0e-9)


@pytest.mark.cuda
def test_mixed_host_and_device_arrays() -> None:
    """Host and device descriptors may share one CUDA context."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    reference = Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ]
        ),
        backend="cuda",
    ).singlepoint()
    arrays = _water_arrays(torch)
    arrays["positions"] = np.asarray(
        [
            [0.0000000000, 0.0000000000, -0.7357858611],
            [1.4418315287, 0.0000000000, 0.3678929305],
            [-1.4418315287, 0.0000000000, 0.3678929305],
        ],
        dtype=np.float64,
    )
    result = ArrayBatch(**arrays, backend="cuda").compute()
    assert abs(float(result.energies.item()) - reference.energy) < 1.0e-10


@pytest.mark.cuda
def test_torch_device_out_buffers() -> None:
    """out= may be CUDA device buffers; results reference them in place."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    reference = Calculator(
        "GFN2-xTB",
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0000000000, 0.0000000000, -0.7357858611],
                [1.4418315287, 0.0000000000, 0.3678929305],
                [-1.4418315287, 0.0000000000, 0.3678929305],
            ]
        ),
        backend="cuda",
    ).singlepoint()
    arrays = _water_arrays(torch)
    out_forces = torch.zeros(3, 3, dtype=torch.float64, device="cuda")
    out_energies = torch.zeros(1, dtype=torch.float64, device="cuda")
    result = ArrayBatch(**arrays, backend="cuda").compute(
        out={"forces": out_forces, "energies": out_energies}
    )
    assert result.forces is out_forces
    assert result.energies is out_energies
    assert torch.allclose(
        out_forces, torch.tensor(reference.forces, device="cuda"), atol=1.0e-9
    )


@pytest.mark.cuda
def test_torch_noncontiguous_cuda_rejected() -> None:
    """Non-contiguous CUDA tensors are refused under the zero-copy contract."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    arrays = _water_arrays(torch)
    arrays["positions"] = arrays["positions"].as_strided((3, 3), (1, 3))
    assert not arrays["positions"].is_contiguous()
    with pytest.raises(BufferError, match="C-contiguous"):
        ArrayBatch(**arrays, backend="cuda").compute()


@pytest.mark.cuda
def test_torch_dtype_mismatch_cuda_rejected() -> None:
    """float32 CUDA tensors do not silently become float64 descriptors."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    arrays = _water_arrays(torch)
    arrays["positions"] = arrays["positions"].float()
    with pytest.raises(GPUxtbValueError, match="dtype"):
        ArrayBatch(**arrays, backend="cuda").compute()


@pytest.mark.cuda
def test_requires_grad_output_rejected() -> None:
    """Autograd leaf tensors cannot be used as gpuxtb output buffers."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    arrays = _water_arrays(torch)
    out_forces = torch.zeros(3, 3, dtype=torch.float64, device="cuda").requires_grad_(
        True
    )
    with pytest.raises(BufferError):
        ArrayBatch(**arrays, backend="cuda").compute(out={"forces": out_forces})


@pytest.mark.cuda
def test_torch_cuda_batch_with_spin() -> None:
    """A ragged unrestricted batch runs on pure torch device arrays."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    case = case_by_id("h3_plus")
    numbers, positions, charge, _uhf, _spin = structure_inputs(case)
    spin_channels = 2
    arrays = {
        "atom_offsets": _wrap(torch, "atom_offsets", np.array([0, 3], np.int64)),
        "atomic_numbers": _wrap(torch, "atomic_numbers", numbers.astype(np.int32)),
        "positions": _wrap(torch, "positions", positions),
        "molecular_charges": _wrap(torch, "molecular_charges", np.array([1.0])),
        "unpaired_electrons": _wrap(
            torch, "unpaired_electrons", np.array([0], np.int32)
        ),
        "spin_channels": _wrap(torch, "spin_channels", np.array([2], np.int32)),
    }
    reference = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=0,
        spin_channels=spin_channels,
        backend="cuda",
    ).singlepoint()
    result = ArrayBatch(**arrays, backend="cuda").compute()
    assert abs(float(result.energies.item()) - reference.energy) < 1.0e-10
    assert np.allclose(result.forces, reference.forces, atol=1.0e-9)


@pytest.mark.cuda
def test_torch_custom_stream_and_changed_geometry() -> None:
    """Repeated calls honor a custom stream and newly produced input data."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    arrays = _water_arrays(torch)
    stream = torch.cuda.Stream()
    out_energies = torch.full((1,), 123.0, dtype=torch.float64, device="cuda")
    with ArrayBatch(**arrays, backend="cuda", stream=int(stream.cuda_stream)) as batch:
        first = batch.compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": out_energies},
        )
        first_energy = float(first.energies.item())
        arrays["positions"][0, 2] += 0.2
        second = batch.compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": out_energies},
        )
        second_energy = float(second.energies.item())

    assert batch.context.stream == int(stream.cuda_stream)
    assert first.energies is out_energies
    assert second.energies is out_energies
    assert first_energy != pytest.approx(second_energy, abs=1.0e-8)


@pytest.mark.cuda
def test_cuda_validation_failure_leaves_device_output_unchanged() -> None:
    """A pre-execution descriptor failure must not publish device outputs."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    arrays = _water_arrays(torch)
    arrays["atom_offsets"] = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    sentinel = -456.25
    out_energies = torch.full((1,), sentinel, dtype=torch.float64, device="cuda")
    with pytest.raises(GPUxtbRuntimeError, match="atom_offsets"):
        ArrayBatch(**arrays, backend="cuda").compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": out_energies},
        )
    assert out_energies.item() == sentinel


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty.*")
def test_cuda_active_stream_capture_is_rejected() -> None:
    """The synchronous public API refuses a stream already under capture."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    stream = torch.cuda.Stream()
    batch = ArrayBatch(
        **_water_host_arrays(), backend="cuda", stream=int(stream.cuda_stream)
    )
    # Create the native context before capture so the assertion targets the
    # compute contract rather than context-initialization side effects.
    _ = batch.context.backend
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        try:
            with pytest.raises(GPUxtbRuntimeError, match="capture"):
                batch.compute()
        finally:
            graph.capture_end()
    batch.close()


@pytest.mark.cuda
def test_cuda_compute_restores_callers_current_device() -> None:
    """Normal and validation-failure exits preserve the caller's device."""
    reason = _device_ready()
    if reason:
        pytest.skip(reason)
    if _TORCH is None:
        pytest.skip("torch not installed")
    import torch

    context_device = 0
    caller_device = 1 if torch.cuda.device_count() > 1 else 0
    with torch.cuda.device(context_device):
        good = _water_arrays(torch)
        bad = _water_arrays(torch)
        bad["atom_offsets"] = torch.tensor(
            [1, 3], dtype=torch.int64, device=f"cuda:{context_device}"
        )
    with torch.cuda.device(caller_device):
        ArrayBatch(**good, backend="cuda", device_id=context_device).compute(
            compute_forces=False, compute_charges=False
        )
        assert torch.cuda.current_device() == caller_device

        with pytest.raises(GPUxtbRuntimeError, match="atom_offsets"):
            ArrayBatch(**bad, backend="cuda", device_id=context_device).compute()
        assert torch.cuda.current_device() == caller_device


@pytest.mark.cuda
def test_jax_eager_cuda_arrays_match_host_cuda() -> None:
    """Concrete JAX CUDA arrays are valid immutable DLPack inputs."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    if _JAX is None:
        pytest.skip("JAX is not installed")
    import jax
    import jax.numpy as jnp

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    jax.config.update("jax_enable_x64", True)
    host = _water_host_arrays()
    arrays = {name: jax.device_put(jnp.asarray(value)) for name, value in host.items()}
    reference = Calculator(
        "GFN2-xTB",
        host["atomic_numbers"],
        host["positions"],
        backend="cuda",
    ).singlepoint()
    result = ArrayBatch(**arrays, backend="cuda").compute()
    assert result.energies == pytest.approx([reference.energy], abs=1.0e-10)
    assert np.allclose(result.forces, reference.forces, atol=1.0e-9)


@pytest.mark.cuda
def test_jax_array_is_rejected_as_mutable_output() -> None:
    """JAX arrays remain immutable even when their capsule is writable."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    if _JAX is None:
        pytest.skip("JAX is not installed")
    import jax
    import jax.numpy as jnp

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX has no CUDA device")
    jax.config.update("jax_enable_x64", True)
    out = jax.device_put(jnp.empty((1,), dtype=jnp.float64))
    with pytest.raises(BufferError, match="not writable"):
        ArrayBatch(**_water_host_arrays(), backend="cuda").compute(
            compute_forces=False,
            compute_charges=False,
            out={"energies": out},
        )


@pytest.mark.cuda
def test_cupy_device_arrays_and_outputs_match_host_cuda() -> None:
    """CuPy inputs and mutable outputs stay device-resident and in place."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    if _CUPY is None:
        pytest.skip("CuPy is not installed")
    import cupy as cp

    host = _water_host_arrays()
    arrays = {name: cp.asarray(value) for name, value in host.items()}
    reference = Calculator(
        "GFN2-xTB",
        host["atomic_numbers"],
        host["positions"],
        backend="cuda",
    ).singlepoint()
    out_energies = cp.full((1,), 123.0, dtype=cp.float64)
    out_forces = cp.empty((3, 3), dtype=cp.float64)
    result = ArrayBatch(**arrays, backend="cuda").compute(
        out={"energies": out_energies, "forces": out_forces}
    )
    assert result.energies is out_energies
    assert result.forces is out_forces
    assert cp.asnumpy(out_energies) == pytest.approx([reference.energy], abs=1.0e-10)
    assert np.allclose(cp.asnumpy(out_forces), reference.forces, atol=1.0e-9)
