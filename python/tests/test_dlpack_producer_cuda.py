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
from _cases import case_by_id, structure_inputs
from gpuxtb import library
from gpuxtb.exceptions import GPUxtbNotSupportedError
from gpuxtb.interface import ArrayBatch

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


def _device_ready(*providers: str) -> str | None:
    """Return a skip reason unless gpuxtb and every named provider use CUDA."""
    if not _library_has_cuda():
        return "CUDA backend is not available on this host"
    for provider in providers or ("torch",):
        if provider == "torch":
            if _TORCH is None:
                return "torch is required for CUDA producer tests"
            import torch

            if not torch.cuda.is_available():
                return "torch does not have a usable CUDA device"
        elif provider == "cupy":
            if _CUPY is None:
                return "cupy is required for CUDA producer tests"
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                return "cupy does not have a usable CUDA device"
        elif provider == "jax":
            if _JAX is None:
                return "jax is required for CUDA producer tests"
            import jax

            if not any(device.platform in ("cuda", "gpu") for device in jax.devices()):
                return "jax does not have a usable CUDA device"
        else:
            raise ValueError(f"unknown CUDA provider {provider!r}")
    return None


def _producer_base_pointer(producer: object) -> int:
    """Return this producer's slice pointer inside the native arena."""
    import ctypes

    buffer = library.Buffer()
    library._check_init(
        "gpuxtb_result_owner_buffer",
        library.load_library().gpuxtb_result_owner_buffer(
            producer.arena.handle, ctypes.byref(buffer)
        ),
    )
    return int(buffer.data) + producer.byte_offset


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

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")

    energies = result.energies
    assert isinstance(energies, dlpack.DLPackResultBuffer), type(energies)
    assert energies.__dlpack_device__() == (
        dlpack._DLPACK_DEVICE_CUDA,
        batch.context.device_id,
    )

    t_energy = torch.from_dlpack(energies)
    assert t_energy.is_cuda
    assert t_energy.dtype == torch.float64
    t_forces = torch.from_dlpack(result.forces)
    t_charges = torch.from_dlpack(result.charges)
    assert t_forces.shape == (len(_packed("h3_plus")["atomic_numbers"]), 3)

    # Values must match the host CPU result (parity).
    from gpuxtb.interface import compute_arrays

    host = compute_arrays(
        **{name: np.ascontiguousarray(value) for name, value in packed.items()},
        backend="cpu",
    )
    np.testing.assert_allclose(t_energy.cpu().numpy(), host.energies, rtol=1e-10)
    np.testing.assert_allclose(t_forces.cpu().numpy(), host.forces, atol=1e-9)
    np.testing.assert_allclose(t_charges.cpu().numpy(), host.charges, atol=1e-9)

    # A second independent export reads the same arena bytes.
    t_energy2 = torch.from_dlpack(result.energies)
    np.testing.assert_array_equal(t_energy2.cpu().numpy(), t_energy.cpu().numpy())

    batch.close()


@pytest.mark.cuda
def test_device_producer_imports_with_cupy(tmp_path: object) -> None:
    """``cupy.from_dlpack`` imports the finished device bytes zero-copy."""
    skip = _device_ready("cupy")
    if skip is not None:
        pytest.skip(skip)
    import cupy as cp

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")

    energies = result.energies
    assert isinstance(energies, dlpack.DLPackResultBuffer)
    assert energies.__dlpack_device__() == (
        dlpack._DLPACK_DEVICE_CUDA,
        batch.context.device_id,
    )

    cp_energy = cp.from_dlpack(energies)
    assert cp_energy.device.id == batch.context.device_id
    assert cp_energy.dtype == cp.float64
    cp_forces = cp.from_dlpack(result.forces)
    cp_charges = cp.from_dlpack(result.charges)
    assert cp_forces.shape == (len(_packed("h3_plus")["atomic_numbers"]), 3)

    # Values must match the host CPU result (parity).
    from gpuxtb.interface import compute_arrays

    host = compute_arrays(
        **{name: np.ascontiguousarray(value) for name, value in packed.items()},
        backend="cpu",
    )
    np.testing.assert_allclose(cp_energy.get(), host.energies, rtol=1e-10)
    np.testing.assert_allclose(cp_forces.get(), host.forces, atol=1e-9)
    np.testing.assert_allclose(cp_charges.get(), host.charges, atol=1e-9)

    # Zero-copy device import: the imported array aliases the arena bytes, so
    # the CuPy data pointer equals the gpuxtb-owned slice pointer and no host
    # transfer happened.
    expected_pointer = _producer_base_pointer(energies)
    assert cp_energy.data.ptr == expected_pointer

    # A second independent export reads the same arena bytes.
    cp_energy2 = cp.from_dlpack(result.energies)
    np.testing.assert_array_equal(cp_energy2.get(), cp_energy.get())
    assert cp_energy2.data.ptr == expected_pointer

    # The arena stays alive through the imported array's native retain even
    # after every producer and unrelated imported array has been released.
    energy = cp_energy.get().copy()
    for name in (
        "energies",
        "forces",
        "charges",
        "scc_iterations",
        "scc_converged",
        "per_system_status",
    ):
        result.get(name).close()
    result.close()
    batch.close()
    del result, energies, cp_energy2, cp_forces, cp_charges
    import gc

    gc.collect()
    np.testing.assert_array_equal(cp_energy.get(), energy)
    del cp_energy
    gc.collect()


@pytest.mark.cuda
def test_device_producer_imports_with_jax(tmp_path: object) -> None:
    """``jax.dlpack.from_dlpack`` imports the finished device bytes."""
    skip = _device_ready("jax")
    if skip is not None:
        pytest.skip(skip)
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("ketene").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")

    energies = result.energies
    assert isinstance(energies, dlpack.DLPackResultBuffer)
    assert energies.__dlpack_device__() == (
        dlpack._DLPACK_DEVICE_CUDA,
        batch.context.device_id,
    )

    jax_energy = jax.dlpack.from_dlpack(energies, copy=False)
    assert jax_energy.dtype == jnp.float64
    assert {device.id for device in jax_energy.devices()} == {batch.context.device_id}
    jax_forces = jax.dlpack.from_dlpack(result.forces)
    jax_charges = jax.dlpack.from_dlpack(result.charges)
    assert jax_forces.shape == (len(_packed("ketene")["atomic_numbers"]), 3)

    # Values must match the host CPU result (parity), read back through JAX.
    from gpuxtb.interface import compute_arrays

    host = compute_arrays(
        **{name: np.ascontiguousarray(value) for name, value in packed.items()},
        backend="cpu",
    )
    np.testing.assert_allclose(np.asarray(jax_energy), host.energies, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(jax_forces), host.forces, atol=1e-9)
    np.testing.assert_allclose(np.asarray(jax_charges), host.charges, atol=1e-9)
    assert jax_energy.unsafe_buffer_pointer() == _producer_base_pointer(energies)

    # A second independent export reads the same arena bytes.
    jax_energy2 = jax.dlpack.from_dlpack(result.energies)
    np.testing.assert_array_equal(np.asarray(jax_energy2), np.asarray(jax_energy))

    result.close()
    batch.close()


@pytest.mark.cuda
def test_device_producer_providers_alias_same_device_bytes(tmp_path: object) -> None:
    """torch, CuPy, and JAX all import the same arena without a host copy."""
    skip = _device_ready("torch", "cupy", "jax")
    if skip is not None:
        pytest.skip(skip)
    import cupy as cp
    import jax
    import torch

    jax.config.update("jax_enable_x64", True)

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")
    energies = result.energies
    expected_pointer = _producer_base_pointer(energies)

    t_energy = torch.from_dlpack(energies)
    cp_energy = cp.from_dlpack(energies)
    jax_energy = jax.dlpack.from_dlpack(energies, copy=False)

    # Every provider observes the exact same device bytes by pointer.
    assert (
        t_energy.data_ptr()
        == cp_energy.data.ptr
        == jax_energy.unsafe_buffer_pointer()
        == expected_pointer
    )
    np.testing.assert_array_equal(t_energy.cpu().numpy(), cp_energy.get())
    np.testing.assert_array_equal(t_energy.cpu().numpy(), np.asarray(jax_energy))
    batch.close()


@pytest.mark.cuda
def test_device_producer_imports_after_custom_stream_compute(tmp_path: object) -> None:
    """A framework imports results computed on a non-default native stream."""
    skip = _device_ready("torch")
    if skip is not None:
        pytest.skip(skip)
    import torch
    from gpuxtb.interface import compute_arrays

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    batch = ArrayBatch(
        **packed,
        backend="cuda",
        stream=int(producer_stream.cuda_stream),
    )
    result = batch.compute(result_memory="cuda")
    with torch.cuda.stream(consumer_stream):
        imported = torch.from_dlpack(result.energies)
        observed = imported.clone()
    consumer_stream.synchronize()

    host = compute_arrays(
        **{name: np.ascontiguousarray(value) for name, value in packed.items()},
        backend="cpu",
    )
    np.testing.assert_allclose(observed.cpu().numpy(), host.energies, rtol=1e-10)
    assert imported.data_ptr() == _producer_base_pointer(result.energies)
    result.close()
    batch.close()


@pytest.mark.cuda
def test_device_producer_out_precedence_mixed(tmp_path: object) -> None:
    """out= buffers always win; omitted outputs go to the device arena."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("ketene").items()
    }
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
def test_device_producer_all_outputs_supplied_does_not_allocate_arena(
    tmp_path: object,
) -> None:
    """An all-``out=`` CUDA call needs no empty gpuxtb-owned arena."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    natoms = len(packed["atomic_numbers"])
    out = {
        "energies": torch.empty(1, dtype=torch.float64, device="cuda"),
        "forces": torch.empty((natoms, 3), dtype=torch.float64, device="cuda"),
        "charges": torch.empty(natoms, dtype=torch.float64, device="cuda"),
        "scc_iterations": torch.empty(1, dtype=torch.int32, device="cuda"),
        "scc_converged": torch.empty(1, dtype=torch.uint8, device="cuda"),
        "per_system_status": torch.empty(1, dtype=torch.int32, device="cuda"),
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda", out=out)
    for name, expected in out.items():
        assert getattr(result, name) is expected
    batch.close()


@pytest.mark.cuda
def test_device_producer_close_releases_reference(tmp_path: object) -> None:
    """Closing the result releases only the producer reference.

    Returned DLPackResultBuffer producers retain the arena independently, so
    they keep the finished device bytes alive after the result is closed and
    can still be exported; their own close (or GC) releases those references.
    """
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")
    t_energy = torch.from_dlpack(result.energies)
    energy = t_energy.cpu().numpy().copy()
    # Importing again after the result is closed must still work: the producer
    # holds its own retained reference.
    result.close()
    t_energy_again = torch.from_dlpack(result.energies)
    np.testing.assert_array_equal(t_energy_again.cpu().numpy(), energy)
    # Closing the batch context and result scope then GC keeps the imported
    # tensor readable through its native retain.
    batch.close()
    del result
    gc.collect()
    np.testing.assert_array_equal(t_energy.cpu().numpy(), energy)
    del t_energy
    del t_energy_again
    gc.collect()


@pytest.mark.cuda
def test_device_producer_failed_indices_mode_is_host_only(tmp_path: object) -> None:
    """Device-resident diagnostics make failed_indices raise precisely."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda")
    with pytest.raises(GPUxtbNotSupportedError, match="host numpy status"):
        _ = result.failed_indices
    batch.close()


@pytest.mark.cuda
def test_device_producer_preserves_nan_publication_on_failure(tmp_path: object) -> None:
    """Per-system SCC failure fills the failed slice with NaNs on the device.

    The native NaN/status publication semantics must survive the
    gpuxtb-owned device arena path.
    """
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch

    packed = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**packed, backend="cuda", stream=1)
    result = batch.compute(result_memory="cuda", max_scc_iterations=1)
    t_energy = torch.from_dlpack(result.energies)
    t_status = torch.from_dlpack(result.per_system_status)
    t_converged = torch.from_dlpack(result.scc_converged)
    assert t_converged.cpu().numpy().tolist() == [0]
    assert t_status.cpu().numpy().tolist() == [library.STATUS_SCC_NOT_CONVERGED]
    assert not np.isfinite(t_energy.cpu().numpy()).all()
    batch.close()


@pytest.mark.cuda
def test_device_producer_repeated_and_changed_geometry(tmp_path: object) -> None:
    """Repeated calls and changed geometry re-export fresh, correct arenas."""
    skip = _device_ready()
    if skip is not None:
        pytest.skip(skip)
    import torch
    from gpuxtb.interface import compute_arrays

    base = {
        name: np.ascontiguousarray(value) for name, value in _packed("h3_plus").items()
    }
    batch = ArrayBatch(**base, backend="cuda", stream=1)
    result1 = batch.compute(result_memory="cuda")
    e1 = torch.from_dlpack(result1.energies).cpu().numpy().copy()

    result2 = batch.compute(result_memory="cuda")
    e2 = torch.from_dlpack(result2.energies).cpu().numpy()
    np.testing.assert_array_equal(e2, e1)

    changed = {name: np.ascontiguousarray(value.copy()) for name, value in base.items()}
    changed["positions"] = np.ascontiguousarray(base["positions"].copy())
    changed["positions"][1, 0] += 0.1
    host_changed = compute_arrays(**dict(changed.items()), backend="cpu")
    batch2 = ArrayBatch(**changed, backend="cuda", stream=1)
    result3 = batch2.compute(result_memory="cuda")
    e3 = torch.from_dlpack(result3.energies).cpu().numpy()
    np.testing.assert_allclose(e3, host_changed.energies, rtol=1e-10)
    batch.close()
    batch2.close()
