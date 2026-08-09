"""Tests for the PyTorch autograd op :func:`xtbloom.xtbloom_torch`.

The op runs packed inference on PyTorch tensors through the DLPack bridge and
exposes a single analytic gradient, ``dE/dR = -F``.  These tests verify the
host (CPU) path, the exact gradient identity against the returned forces, the
ragged-batch gradient slicing, a finite-difference cross-check of the energy
gradient, and the hard errors for every unsupported autograd direction.
CUDA coverage is gated on a real GPU plus a torch CUDA build, mirroring
``test_array_batch_cuda.py``.
"""

from __future__ import annotations

import importlib
import itertools

import numpy as np
import pytest
from xtbloom import Calculator, xtbloom_torch
from xtbloom.exceptions import XTBloomNotSupportedError, XTBloomValueError

_TORCH = importlib.util.find_spec("torch")


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
        raise AssertionError("DLPack-only offsets must not be converted by NumPy")


def _skip_reason() -> str | None:
    """Return a skip reason when the torch tests cannot run."""
    return None if _TORCH is not None else "torch is not installed"


def _library_has_cuda() -> bool:
    """Check whether a CUDA context can actually be created on this host."""
    from xtbloom.exceptions import XTBloomRuntimeError
    from xtbloom.interface import Context

    try:
        with Context("cuda"):
            pass
        return True
    except XTBloomRuntimeError:
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
    energies, forces = xtbloom_torch(
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
    energies, forces = xtbloom_torch(
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
    energies, _ = xtbloom_torch(
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


def test_dlpack_only_offsets_survive_backward() -> None:
    """Non-Torch offsets remain DLPack-native through gradient construction."""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    import torch

    positions = torch.tensor(
        WATER_POSITIONS.tolist(), dtype=torch.float64, requires_grad=True
    )
    arrays = _packed([WATER_NUMBERS], [WATER_POSITIONS], torch)
    energies, forces = xtbloom_torch(
        positions,
        arrays["atomic_numbers"],
        _DLPackOnly(arrays["atom_offsets"]),
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    energies.sum().backward()
    assert positions.grad is not None
    assert torch.allclose(positions.grad, -forces, atol=0.0, rtol=0.0)


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
        values, _ = xtbloom_torch(
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
    energies, _ = xtbloom_torch(
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
    with pytest.raises(XTBloomNotSupportedError, match="molecular_charges"):
        xtbloom_torch(
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
    _, forces = xtbloom_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    with pytest.raises(XTBloomNotSupportedError, match="forces"):
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
    _, forces = xtbloom_torch(
        positions,
        arrays["atomic_numbers"],
        arrays["atom_offsets"],
        arrays["molecular_charges"],
        arrays["unpaired_electrons"],
        arrays["spin_channels"],
        backend="cpu",
    )
    with pytest.raises(XTBloomNotSupportedError, match="forces"):
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
        energies, _ = xtbloom_torch(
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
    with pytest.raises(XTBloomNotSupportedError, match="higher-order"):
        torch.autograd.grad(
            total_energy(direct_positions), direct_positions, create_graph=True
        )
    with pytest.raises(XTBloomNotSupportedError, match="higher-order"):
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
    energies, forces = xtbloom_torch(
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
    with pytest.raises(XTBloomValueError, match="float64"):
        xtbloom_torch(
            positions32,
            arrays["atomic_numbers"],
            arrays["atom_offsets"],
            arrays["molecular_charges"],
            arrays["unpaired_electrons"],
            arrays["spin_channels"],
            backend="cpu",
        )


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
    host_energies, host_forces = xtbloom_torch(
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
    cuda_energies, cuda_forces = xtbloom_torch(
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
