"""GFN model-selection coverage for the PyTorch autograd adapter."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from xtbloom import Calculator, xtbloom_torch
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError
from xtbloom.torch import _resolve_method

_TORCH = importlib.util.find_spec("torch")

WATER_NUMBERS = np.array([8, 1, 1], dtype=np.int32)
WATER_POSITIONS = np.array(
    [
        [0.0000000000, 0.0000000000, -0.7357858611],
        [1.4418315287, 0.0000000000, 0.3678929305],
        [-1.4418315287, 0.0000000000, 0.3678929305],
    ],
    dtype=np.float64,
)


def _skip_reason() -> str | None:
    return None if _TORCH is not None else "torch is not installed"


def _library_has_cuda() -> bool:
    from xtbloom.interface import Context

    try:
        with Context("cuda"):
            pass
        return True
    except XTBloomRuntimeError:
        return False


def _packed(torch: object, *, device: str = "cpu") -> dict[str, object]:
    return {
        "positions": torch.tensor(WATER_POSITIONS, dtype=torch.float64, device=device),
        "atomic_numbers": torch.tensor(WATER_NUMBERS, dtype=torch.int32, device=device),
        "atom_offsets": torch.tensor([0, 3], dtype=torch.int64, device=device),
        "molecular_charges": torch.zeros(1, dtype=torch.float64, device=device),
        "unpaired_electrons": torch.zeros(1, dtype=torch.int32, device=device),
        "spin_channels": torch.ones(1, dtype=torch.int32, device=device),
    }


def _run(torch: object, arrays: dict[str, object], *, method: str, backend: str):
    return xtbloom_torch(
        arrays["positions"],
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        method=method,
        backend=backend,
    )


def test_torch_method_resolver_matches_public_model_tags() -> None:
    import xtbloom.library as library

    assert _resolve_method("GFN1-xTB") == library.MODEL_GFN1_XTB
    assert _resolve_method("GFN1") == library.MODEL_GFN1_XTB
    assert _resolve_method("GFN2-xTB") == library.MODEL_GFN2_XTB
    assert _resolve_method("GFN2") == library.MODEL_GFN2_XTB
    with pytest.raises(XTBloomValueError, match="unknown method"):
        _resolve_method("GFN0-xTB")


@pytest.mark.parametrize("method", ["GFN1-xTB", "GFN1"])
def test_torch_gfn1_cpu_matches_calculator(method: str) -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    arrays = _packed(torch)
    energies, forces = _run(torch, arrays, method=method, backend="cpu")
    reference = Calculator(
        "GFN1-xTB", WATER_NUMBERS, WATER_POSITIONS, backend="cpu"
    ).singlepoint()
    assert torch.allclose(
        energies,
        torch.tensor([reference.energy], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert torch.allclose(
        forces,
        torch.from_numpy(np.ascontiguousarray(reference.forces)),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_torch_gfn1_autograd_is_negative_force() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    arrays = _packed(torch)
    positions = arrays["positions"].requires_grad_(True)
    energies, forces = _run(torch, arrays, method="GFN1-xTB", backend="cpu")
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


def test_private_torch_op_rejects_unknown_model() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch
    from xtbloom import torch as torch_module

    arrays = _packed(torch)
    with pytest.raises(RuntimeError, match="model"):
        torch_module._xtbloom_torch_op()(
            arrays["positions"],
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            0,
            0,
            0,
            0,
            0,
            torch.empty(1, dtype=torch.float64),
            torch.empty((3, 3), dtype=torch.float64),
            999,
            1,
            -1,
            1,
            0,
            250,
            1.0e-6,
            1.0e-8,
            300.0,
            1,
            8,
            0.4,
            0,
        )


@pytest.mark.cuda
def test_torch_cuda_cache_separates_gfn1_and_gfn2() -> None:
    reason = _skip_reason() if _library_has_cuda() else "CUDA backend unavailable"
    if reason:
        pytest.skip(reason)
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch has no usable CUDA device")

    references = {
        method: Calculator(
            method, WATER_NUMBERS, WATER_POSITIONS, backend="cpu"
        ).singlepoint()
        for method in ("GFN1-xTB", "GFN2-xTB")
    }
    arrays = _packed(torch, device="cuda")
    for method in ("GFN1-xTB", "GFN2-xTB", "GFN1-xTB"):
        positions = arrays["positions"].detach().clone().requires_grad_(True)
        call_arrays = {**arrays, "positions": positions}
        energies, forces = _run(torch, call_arrays, method=method, backend="cuda")
        energies.sum().backward()
        reference = references[method]
        assert positions.grad is not None
        assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)
        assert torch.allclose(
            energies.cpu(),
            torch.tensor([reference.energy], dtype=torch.float64),
            atol=1.0e-9,
            rtol=1.0e-9,
        )
        assert torch.allclose(
            forces.cpu(),
            torch.from_numpy(np.ascontiguousarray(reference.forces)),
            atol=1.0e-9,
            rtol=1.0e-9,
        )
