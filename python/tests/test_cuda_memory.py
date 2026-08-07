"""Tests for device-resident CUDA memory support in the Python interface.

The public C ABI accepts ``GPUXTB_MEMORY_CUDA_DEVICE`` input and output
descriptors on the CUDA backend; ``memory_space="device"`` and
``"mixed"`` place those descriptors in CUDA device memory through the
CUDA runtime (libcudart) instead of staging through the host.

Device-mode tests require a real CUDA device, a CUDA-enabled library, and
a loadable CUDA runtime (libcudart); they skip otherwise so CPU-only
installations remain green. Error-handling tests run everywhere.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
from _cases import case_by_id, qmmm_points, structure_inputs
from gpuxtb import (
    BatchCalculator,
    Calculator,
    ChargeResponse,
    PointCharge,
    Structure,
    library,
)
from gpuxtb import cuda as gpuxtb_cuda
from gpuxtb.exceptions import (
    GPUxtbNotSupportedError,
    GPUxtbRuntimeError,
    GPUxtbValueError,
)
from gpuxtb.interface import Context

H2_POSITIONS = np.array(
    [
        [-0.71, 0.0, 0.0],
        [0.71, 0.0, 0.0],
    ]
)


def _library_has_cuda() -> bool:
    """Check whether a CUDA context can actually be created on this host."""
    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


def _has_cuda_runtime() -> bool:
    """Check whether the CUDA device-memory runtime is importable."""
    return gpuxtb_cuda.device_memory_available()


def _device_memory_ready() -> str | None:
    """Return a skip reason when device-resident tests cannot run, else ``None``.

    Device placement needs both a working CUDA backend and a loadable CUDA
    runtime (libcudart). When libcudart cannot be discovered the placement
    calls would raise instead of running, so such hosts must skip too.
    """
    if not _library_has_cuda():
        return "CUDA backend is not available on this host"
    if not _has_cuda_runtime():
        return "CUDA device-memory runtime is not available"
    return None


def _structure(case_id: str) -> Structure:
    """Build a conformance-golden structure with its documented spin state."""
    numbers, positions, charge, uhf, spin = structure_inputs(case_by_id(case_id))
    return Structure(numbers, positions, charge=charge, uhf=uhf, spin_channels=spin)


# --- error handling (no CUDA device required) ---------------------------------


def test_unknown_memory_space_is_rejected() -> None:
    """Reject memory placements outside the documented host/mixed/device set."""
    with pytest.raises(GPUxtbValueError, match="memory_space"):
        Calculator(
            "GFN2-xTB",
            numbers=[1, 1],
            positions=H2_POSITIONS,
            memory_space="gpu",
        )


def test_device_memory_requires_cuda_backend() -> None:
    """Device placement is a CUDA-backend feature; CPU must refuse it."""
    for memory_space in ("device", "mixed"):
        calculator = Calculator(
            "GFN2-xTB",
            numbers=[1, 1],
            positions=H2_POSITIONS,
            backend="cpu",
            memory_space=memory_space,
        )
        with pytest.raises(GPUxtbNotSupportedError, match="CUDA backend"):
            calculator.singlepoint()


def test_host_memory_space_is_an_acceptable_explicit_choice() -> None:
    """''host'' is the documented default and stays an explicit option."""
    explicit = Calculator(
        "GFN2-xTB",
        numbers=[1, 1],
        positions=H2_POSITIONS,
        backend="cpu",
        memory_space="host",
    ).singlepoint()
    default = Calculator(
        "GFN2-xTB", numbers=[1, 1], positions=H2_POSITIONS, backend="cpu"
    ).singlepoint()
    assert explicit.energy == pytest.approx(default.energy, rel=1.0e-12, abs=1.0e-11)


def test_cuda_module_availability_probe_is_boolean() -> None:
    """The lazy device-memory probe never raises and returns a bool."""
    assert gpuxtb_cuda.device_memory_available() in (True, False)


# --- device-resident execution (real CUDA device required) --------------------


@pytest.mark.cuda
@pytest.mark.parametrize("memory_space", ["device", "mixed"])
def test_device_memory_single_matches_host(memory_space: str) -> None:
    """Placement must not change single-system results."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    case = case_by_id("h3_plus")
    numbers, positions, charge, uhf, spin = structure_inputs(case)
    host_result = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cuda",
        memory_space="host",
    ).singlepoint()
    device_result = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cuda",
        memory_space=memory_space,
    ).singlepoint()
    assert device_result.energy == pytest.approx(host_result.energy, rel=1.0e-10)
    assert device_result.forces == pytest.approx(host_result.forces, abs=1.0e-9)
    assert device_result.charges == pytest.approx(host_result.charges, abs=1.0e-9)


@pytest.mark.cuda
@pytest.mark.parametrize("memory_space", ["device", "mixed"])
def test_device_memory_batch_matches_host(memory_space: str) -> None:
    """Ragged batches must be placement-invariant, including per-system results."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    structures = [_structure("ketene"), _structure("h3_plus")]
    host = BatchCalculator(structures, backend="cuda", memory_space="host").compute()
    device = BatchCalculator(
        structures, backend="cuda", memory_space=memory_space
    ).compute()
    assert device.energies == pytest.approx(host.energies, rel=1.0e-10)
    assert device.forces == pytest.approx(host.forces, abs=1.0e-9)
    assert device.charges == pytest.approx(host.charges, abs=1.0e-9)
    for index in range(len(structures)):
        assert device[index].energy == pytest.approx(host[index].energy, rel=1.0e-10)


@pytest.mark.cuda
def test_device_memory_sliced_batch_matches_unsliced() -> None:
    """Auto-slicing must keep device results bit-comparable to an unsliced run."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    structures = [_structure("ketene"), _structure("h3_plus")]
    unsliced = BatchCalculator(
        structures, backend="cuda", memory_space="device"
    ).compute()
    sliced = BatchCalculator(structures, backend="cuda", memory_space="device").compute(
        auto_batch_size=4
    )
    assert sliced.energies == pytest.approx(unsliced.energies, rel=1.0e-10)
    assert sliced.forces == pytest.approx(unsliced.forces, abs=1.0e-9)


@pytest.mark.cuda
def test_device_memory_point_charges_match_host() -> None:
    """Point-charge inputs and outputs must survive device placement."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    case = case_by_id("water_one_pc_gamma999")
    numbers, positions, charge, uhf, spin = structure_inputs(case)
    points = qmmm_points(case)
    assert points is not None
    point_charges = PointCharge(*points)
    host = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=point_charges,
        backend="cuda",
        memory_space="host",
    ).singlepoint()
    device = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=point_charges,
        backend="cuda",
        memory_space="device",
    ).singlepoint()
    assert device.energy == pytest.approx(host.energy, rel=1.0e-10)
    assert device.point_charge_forces is not None
    assert device.point_charge_forces == pytest.approx(
        host.point_charge_forces, abs=1.0e-9
    )


@pytest.mark.cuda
@pytest.mark.parametrize("memory_space", ["device", "mixed"])
def test_device_memory_charge_response_matches_host(memory_space: str) -> None:
    """Charge-response descriptors must survive device and mixed placement."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    response = ChargeResponse(
        shifts=[0.003, -0.002, 0.001],
        matrix=np.array([[0.02, 0.001, 0.0], [0.001, 0.018, 0.0], [0.0, 0.0, 0.015]]),
    )
    structure = _structure("h3_plus")
    host_result = Calculator(
        "GFN2-xTB",
        structure.numbers,
        structure.positions,
        charge=structure.charge,
        uhf=structure.uhf,
        spin_channels=structure.spin_channels,
        charge_response=response,
        backend="cuda",
        memory_space="host",
    ).singlepoint()
    device_result = Calculator(
        "GFN2-xTB",
        structure.numbers,
        structure.positions,
        charge=structure.charge,
        uhf=structure.uhf,
        spin_channels=structure.spin_channels,
        charge_response=response,
        backend="cuda",
        memory_space=memory_space,
    ).singlepoint()
    assert device_result.energy == pytest.approx(host_result.energy, rel=1.0e-10)
    assert device_result.charges == pytest.approx(host_result.charges, abs=1.0e-9)


@pytest.mark.cuda
def test_cuda_device_context_round_trip_and_cleanup() -> None:
    """The device-memory owner must upload, read back, and free device memory."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    if not _has_cuda_runtime():
        pytest.skip("CUDA device-memory runtime is not available")
    context = gpuxtb_cuda.CudaDeviceContext(0)
    try:
        host = np.arange(16, dtype=np.float64)
        address = context.upload(host)
        assert address != 0
        owner = np.empty_like(host)
        context.download(address, owner)
        np.testing.assert_array_equal(owner, host)
        assert len(context._allocations) == 1
    finally:
        context.close()
    assert not context._allocations


@pytest.mark.cuda
def test_cuda_device_allocation_failure_reports_abi_status() -> None:
    """Map CUDA OOM to ``STATUS_ALLOCATION_FAILED`` and recover cleanly."""
    reason = _device_memory_ready()
    if reason:
        pytest.skip(reason)
    context = gpuxtb_cuda.CudaDeviceContext(0)
    try:
        with pytest.raises(GPUxtbRuntimeError) as caught:
            # Far larger than any available device: cudaMalloc must fail OOM.
            context.allocate(1 << 42)
        assert caught.value.status == library.STATUS_ALLOCATION_FAILED
        # A caught CUDA failure must not leave a sticky error that poisons
        # subsequent CUDA work in the same process (gpuxtb reads the error
        # state on its next launch).
        assert context.allocate(64) != 0
    finally:
        context.close()


def test_descriptor_helpers_remain_host_by_default() -> None:
    """Library descriptor helpers keep their historical host placement."""
    buffer_descriptor = library.host_const([1.0, 2.0], ctypes.c_double, np.float64)[0]
    assert buffer_descriptor.memory_space == library.MEMORY_HOST
