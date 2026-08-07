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
from gpuxtb.exceptions import GPUxtbValueError

# --- environment probes ---------------------------------------------------------

_TORCH = importlib.util.find_spec("torch")
_CUPY = importlib.util.find_spec("cupy")


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
