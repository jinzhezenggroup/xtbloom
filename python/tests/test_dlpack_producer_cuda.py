"""CUDA device-result producer tests for :class:`gpuxtb.ArrayBatch`.

These tests run ``result_memory="cuda"``, which allocates a gpuxtb-owned CUDA
device arena, runs the real native compute into it, and hands the finished
slices to importing frameworks through the DLPack producer protocol
(``torch.from_dlpack`` / ``cupy.from_dlpack`` / ``jax.dlpack.from_dlpack``)
without a host round trip.  The provider libraries are opt-in test
dependencies and are never imported by gpuxtb runtime code; the whole file
skips when no real CUDA device is available.
"""

from __future__ import annotations

import gc
import importlib

import gpuxtb._dlpack as dlpack
import numpy as np
import pytest
from gpuxtb import library
from gpuxtb.exceptions import GPUxtbNotSupportedError
from gpuxtb.interface import ArrayBatch

from _cases import case_by_id, structure_inputs

_TORCH = importlib.util.find_spec("torch")
_CUPY = importlib.util.find_spec("cupy")
_JAX = importlib.util.find_spec("jax")


def _library_has_cuda() -> bool:
    from gpuxtb.exceptions import GPUxtbRuntimeError
    from gpuxtb.interface import Context

    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


def _device_ready() -> str | None:
    if not _library_has_cuda():
        return "CUDA backend is not available on this host"
    if _TORCH is None:
        return "torch is required for CUDA producer tests"
    return None


def _packed(case_id: str) -> dict[str, np.ndarray]:
    """Pack one conformance structure into flat host numpy descriptors."""
    numbers, positions, charge, uhf, spin = structure_inputs(case_by_id(case_id))
    n = len(numbers)
    return {
        "atom_offsets": np.asarray([0, n], dtype=np.int64),
        "atomic_numbers": np.asarray(numbers, dtype=np.int32),
        "positions": np.asarray(positions, dtype=np.float64),
        "molecular_charges": np.asarray([charge], dtype=np.float64),
        "unpaired_electrons": np.asarray([uhf], dtype=np.int32),
        "spin_channels": np.asarray([spin], dtype=np.int32),
    }


@pytest.mark.cuda
def test_device_producer_imports_with_torch(tmp_path: object) -> None:
    """result_memory='cuda' exports the finished device bytes to torch."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()}
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")

    energies = result.energies
    assert isinstance(energies, dlpack.DLPackResultBuffer), type(energies)
    assert energies.__dlpack_device__() == (dlpack._DLPACK_DEVICE_CUDA, 0)

    t_energy = torch.from_dlpack(energies)
    assert t_energy.is_cuda
    assert t_energy.dtype == torch.float64
    t_forces = torch.from_dlpack(result.forces)
    t_charges = torch.from_dlpack(result.charges)
    assert t_forces.shape == (len(_packed("h3_plus")["atomic_numbers"]), 3)

    # Values must match the host CPU result (parity).
    from gpuxtb.interface import compute_arrays

    host = compute_arrays(**{name: np.ascontiguousarray(value) for name, value in packed.items()})
    np.testing.assert_allclose(t_energy.cpu().numpy(), host.energies, rtol=1e-10)
    np.testing.assert_allclose(t_forces.cpu().numpy(), host.forces, atol=1e-9)
    np.testing.assert_allclose(t_charges.cpu().numpy(), host.charges, atol=1e-9)

    # A second independent export reads the same arena bytes.
    t_energy2 = torch.from_dlpack(result.energies)
    np.testing.assert_array_equal(t_energy2.cpu().numpy(), t_energy.cpu().numpy())

    batch.close()


@pytest.mark.cuda
def test_device_producer_out_precedence_mixed(tmp_path: object) -> None:
    """out= buffers always win; omitted outputs go to the device arena."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {name: np.ascontiguousarray(value) for name, value in _packed("ketene").items()}
    host_out = np.zeros(1, dtype=np.float64)
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(
        result_memory="cuda",
        out={"energies": host_out},
    )
    assert isinstance(result.energies, np.ndarray)  # caller-owned host buffer
    assert isinstance(result.forces, dlpack.DLPackResultBuffer)  # device producer
    t_forces = torch.from_dlpack(result.forces)
    assert t_forces.is_cuda
    batch.close()


@pytest.mark.cuda
def test_device_producer_close_releases_reference(tmp_path: object) -> None:
    """Result and producer close must be idempotent and safe on device arenas."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()}
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")
    t_energy = torch.from_dlpack(result.energies)
    energy = t_energy.cpu().numpy().copy()
    # Closing the batch context and the result releases producer references;
    # the imported torch tensor owns its own native retain.
    batch.close()
    del result
    gc.collect()
    np.testing.assert_array_equal(t_energy.cpu().numpy(), energy)
    del t_energy
    gc.collect()


@pytest.mark.cuda
def test_device_producer_failed_indices_mode_is_host_only(tmp_path: object) -> None:
    """Device-resident diagnostics make failed_indices raise precisely."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)

    packed = {name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()}
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")
    with pytest.raises(GPUxtbNotSupportedError, match="host numpy status"):
        _ = result.failed_indices
    batch.close()