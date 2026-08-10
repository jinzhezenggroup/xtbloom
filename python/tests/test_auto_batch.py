"""Tests for bounded batch slicing in ``BatchCalculator.compute``."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, NoReturn

import _cases
import numpy as np
import pytest
from xtbloom import (
    BatchCalculator,
    BatchResult,
    ChargeResponse,
    Context,
    PointCharge,
    Structure,
    library,
)
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError
from xtbloom.interface import (
    _AUTO_BATCH_BYTES_PER_ATOM,
    _AUTO_BATCH_FALLBACK_MAX_ATOMS,
    _AUTO_BATCH_MAX_ATOMS,
    _AUTO_BATCH_MEMORY_FRACTION,
    _AUTO_BATCH_RESERVE_BYTES,
    _ComputedBatch,
    _merge_computed,
    _resolve_auto_batch_limit,
    _slice_by_total_atoms,
    _split_chunk_near_half,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _make_structures(
    case_ids: Sequence[str], repeats: int = 1, **kwargs: object
) -> list[Structure]:
    """Build repeated structures from molecular conformance cases."""
    structures: list[Structure] = []
    for _ in range(repeats):
        for case_id in case_ids:
            case = _cases.case_by_id(case_id)
            numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
            structures.append(
                Structure(
                    numbers,
                    positions,
                    charge=charge,
                    uhf=uhf,
                    spin_channels=spin,
                    **kwargs,
                )
            )
    return structures


def _make_point_charge_structures(case_ids: Sequence[str]) -> list[Structure]:
    """Build QM/MM structures from point-charge conformance cases."""
    structures: list[Structure] = []
    for case_id in case_ids:
        case = _cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
        point_positions, point_values, point_gammas = _cases.qmmm_points(case)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin,
                point_charges=PointCharge(point_positions, point_values, point_gammas),
            )
        )
    return structures


def _library_has_cuda() -> bool:
    """Return whether this host can create a CUDA context."""
    try:
        with Context("cuda"):
            pass
        return True
    except XTBloomRuntimeError:
        return False


def _fake_computed(
    structures: Sequence[Structure], *, result_flags: int = 0
) -> _ComputedBatch:
    """Build deterministic native-shaped output for batch-control tests."""
    atom_offsets = np.cumsum([0, *(len(structure) for structure in structures)])
    total_atoms = int(atom_offsets[-1])
    return _ComputedBatch(
        energies=np.arange(len(structures), dtype=np.float64),
        forces=np.zeros((total_atoms, 3), dtype=np.float64),
        charges=np.zeros(total_atoms, dtype=np.float64),
        point_charge_forces=None,
        dipole_moments=None,
        scc_iterations=np.ones(len(structures), dtype=np.int32),
        scc_converged=np.ones(len(structures), dtype=np.uint8),
        per_system_status=np.zeros(len(structures), dtype=np.int32),
        result_flags=result_flags,
        atom_offsets=atom_offsets,
        point_offsets=None,
        keepalive=[],
    )


def _assert_cpu_batches_identical(actual: BatchResult, expected: BatchResult) -> None:
    """Require bit-identical public arrays from two CPU batch strategies."""
    np.testing.assert_array_equal(actual.energies, expected.energies)
    np.testing.assert_array_equal(actual.forces, expected.forces)
    np.testing.assert_array_equal(actual.charges, expected.charges)
    np.testing.assert_array_equal(actual.scc_iterations, expected.scc_iterations)
    np.testing.assert_array_equal(actual.scc_converged, expected.scc_converged)
    np.testing.assert_array_equal(actual.per_system_status, expected.per_system_status)
    if expected.point_charge_forces is None:
        assert actual.point_charge_forces is None
    else:
        np.testing.assert_array_equal(
            actual.point_charge_forces, expected.point_charge_forces
        )


def test_slice_keeps_systems_whole_and_documents_soft_limit() -> None:
    """Keep systems indivisible even when one exceeds the soft atom limit."""
    structures = _make_structures(["ketene"], repeats=3)
    chunks = _slice_by_total_atoms(structures, max_total_atoms=7)
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]

    chunks = _slice_by_total_atoms(structures, max_total_atoms=2)
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]
    assert all(sum(len(structure) for structure in chunk) == 5 for chunk in chunks)

    with pytest.raises(XTBloomValueError):
        _slice_by_total_atoms(structures, max_total_atoms=0)


def test_split_chunk_balances_atoms_and_preserves_order() -> None:
    """Split near half the atoms without changing structure order."""
    structures = _make_structures(["h3_plus", "ketene", "sif5_minus", "ketene"])
    left, right = _split_chunk_near_half(structures)
    assert [*left, *right] == structures
    assert left and right
    with pytest.raises(XTBloomValueError):
        _split_chunk_near_half(structures[:1])


class _FakeContext:
    def __init__(self, backend: int, device_id: int = 0) -> None:
        self.backend = backend
        self.device_id = device_id


def test_auto_limit_requeries_current_cuda_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requery free CUDA memory for each automatic limit decision."""
    structures = [[None] * 10] * 1_000
    memory = [(5_000_000_000, 32_000_000_000), (3_000_000_000, 32_000_000_000)]

    def next_memory(_device_id: int) -> tuple[int, int]:
        return memory.pop(0)

    monkeypatch.setattr(library, "device_memory_info", next_memory)
    context = _FakeContext(library.BACKEND_CUDA)
    first = _resolve_auto_batch_limit(context, structures)
    second = _resolve_auto_batch_limit(context, structures)
    assert first == 3_750
    assert second == 1_250
    assert not memory


def test_auto_limit_is_bounded_and_has_query_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cap automatic limits and use the fallback when memory is unknown."""
    structures = [[None] * 10] * 10_000
    context = _FakeContext(library.BACKEND_CUDA)
    monkeypatch.setattr(
        library,
        "device_memory_info",
        lambda _device_id: (100_000_000_000, 100_000_000_000),
    )
    assert _resolve_auto_batch_limit(context, structures) == _AUTO_BATCH_MAX_ATOMS

    monkeypatch.setattr(library, "device_memory_info", lambda _device_id: None)
    assert (
        _resolve_auto_batch_limit(context, structures) == _AUTO_BATCH_FALLBACK_MAX_ATOMS
    )
    assert (
        _resolve_auto_batch_limit(_FakeContext(library.BACKEND_CPU), structures)
        == _AUTO_BATCH_FALLBACK_MAX_ATOMS
    )


def test_auto_limit_formula_uses_reserved_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive the atom limit from reserved and fractional CUDA memory."""
    structures = [[None]] * 100
    target_atoms = 17
    free_bytes = int(
        (_AUTO_BATCH_RESERVE_BYTES + target_atoms * _AUTO_BATCH_BYTES_PER_ATOM)
        / _AUTO_BATCH_MEMORY_FRACTION
    )
    monkeypatch.setattr(
        library,
        "device_memory_info",
        lambda _device_id: (free_bytes, free_bytes),
    )
    assert (
        _resolve_auto_batch_limit(_FakeContext(library.BACKEND_CUDA), structures)
        == target_atoms
    )


def test_auto_retries_only_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bisect automatic chunks only after native allocation failures."""
    structures = _make_structures(["ketene"], repeats=4)
    calls = []

    def fail_large(
        _context: Context,
        chunk: Sequence[Structure],
        **_kwargs: object,
    ) -> _ComputedBatch:
        calls.append(len(chunk))
        if len(chunk) > 1:
            raise XTBloomRuntimeError("synthetic OOM", library.STATUS_ALLOCATION_FAILED)
        return _fake_computed(chunk)

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail_large)
    monkeypatch.setattr(
        "xtbloom.interface._resolve_auto_batch_limit",
        lambda _context, _structures: 1_000,
    )
    result = BatchCalculator(structures).compute(auto_batch_size=True)
    assert len(result) == len(structures)
    assert calls == [4, 2, 1, 1, 2, 1, 1]


@pytest.mark.parametrize(
    "status", [library.STATUS_INTERNAL_ERROR, library.STATUS_ALLOCATION_FAILED]
)
def test_auto_propagates_internal_and_indivisible_failures(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Propagate internal errors and failures of indivisible systems."""
    structures = _make_structures(["ketene"])
    calls = []

    def fail(
        _context: Context,
        chunk: Sequence[Structure],
        **_kwargs: object,
    ) -> NoReturn:
        calls.append(len(chunk))
        raise XTBloomRuntimeError("synthetic failure", status)

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail)
    monkeypatch.setattr(
        "xtbloom.interface._resolve_auto_batch_limit",
        lambda _context, _structures: 1_000,
    )
    with pytest.raises(XTBloomRuntimeError) as caught:
        BatchCalculator(structures).compute(auto_batch_size=True)
    assert caught.value.status == status
    assert calls == [1]


def test_explicit_limit_does_not_override_allocation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid automatic retry when the caller supplies an explicit limit."""
    structures = _make_structures(["ketene"], repeats=2)

    def fail(
        _context: Context,
        _chunk: Sequence[Structure],
        **_kwargs: object,
    ) -> NoReturn:
        raise XTBloomRuntimeError("synthetic OOM", library.STATUS_ALLOCATION_FAILED)

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail)
    with pytest.raises(XTBloomRuntimeError):
        BatchCalculator(structures).compute(auto_batch_size=100)


def test_chunk_output_flags_follow_local_point_charge_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request point-force output only for chunks containing point charges."""
    structures = [
        *_make_structures(["ketene"]),
        *_make_point_charge_structures(["water_one_pc_gamma999"]),
    ]
    flags = []

    def record_flags(
        _context: Context,
        chunk: Sequence[Structure],
        **kwargs: int,
    ) -> _ComputedBatch:
        flags.append(kwargs["flags"])
        return _fake_computed(chunk)

    monkeypatch.setattr("xtbloom.interface._compute_batch", record_flags)
    result = BatchCalculator(structures).compute(auto_batch_size=1)
    assert len(result) == 2
    base_flags = (
        library.COMPUTE_ENERGY | library.COMPUTE_FORCES | library.COMPUTE_ATOMIC_CHARGES
    )
    assert flags == [base_flags, base_flags | library.COMPUTE_POINT_CHARGE_FORCES]


def test_merged_result_flags_include_every_chunk() -> None:
    """Combine every result qualifier emitted by constituent chunks."""
    structures = _make_structures(["ketene"], repeats=2)
    merged = _merge_computed(
        [
            _fake_computed(structures[:1], result_flags=1),
            _fake_computed(structures[1:], result_flags=2),
        ],
        structures,
    )
    assert merged.result_flags == 3


def test_auto_batch_int_is_bit_identical_on_cpu() -> None:
    """Keep explicitly sliced molecular CPU batches bit-identical."""
    structures = _make_structures(
        ["ketene", "h3_plus", "oh_radical", "sif5_minus"], repeats=3
    )
    unsliced = BatchCalculator(structures, backend="cpu").compute()
    sliced = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=6)
    _assert_cpu_batches_identical(sliced, unsliced)
    for index in range(len(unsliced)):
        np.testing.assert_array_equal(sliced[index].forces, unsliced[index].forces)


def test_auto_batch_point_charges_are_bit_identical_on_cpu() -> None:
    """Keep sliced point-charge CPU batches bit-identical."""
    case_ids = [
        "water_one_pc_gamma999",
        "water_dimer_6pc_hardness",
        "water_dimer_6pc_gamma999",
    ]
    structures = _make_point_charge_structures(case_ids) * 2
    unsliced = BatchCalculator(structures, backend="cpu").compute()
    sliced = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=4)
    _assert_cpu_batches_identical(sliced, unsliced)
    for index in range(len(unsliced)):
        np.testing.assert_array_equal(
            sliced[index].point_charge_forces,
            unsliced[index].point_charge_forces,
        )


def test_auto_batch_charge_response_is_bit_identical_on_cpu() -> None:
    """Keep sliced charge-response CPU batches bit-identical."""
    positions = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    structures = [
        Structure([1, 1], positions),
        Structure([1, 1], positions, charge_response=response),
        Structure([1, 1], positions + np.array([0.0, 0.1, 0.0])),
    ]
    unsliced = BatchCalculator(structures, backend="cpu").compute()
    sliced = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=1)
    _assert_cpu_batches_identical(sliced, unsliced)


def test_auto_batch_preserves_peer_local_failure_on_cpu() -> None:
    """Preserve peer-local CPU failure status across sliced execution."""
    structures = _make_structures(["h3_plus", "nenacl", "ketene"])
    unsliced = BatchCalculator(
        structures, backend="cpu", max_scc_iterations=4
    ).compute()
    sliced = BatchCalculator(structures, backend="cpu", max_scc_iterations=4).compute(
        auto_batch_size=3
    )
    _assert_cpu_batches_identical(sliced, unsliced)
    assert sliced.failed_indices.tolist() == [1, 2]
    assert np.isfinite(sliced.energies[0])
    assert np.isnan(sliced.energies[1:]).all()


def test_auto_batch_rejects_invalid_limit() -> None:
    """Reject nonpositive and nonintegral explicit atom limits."""
    structures = _make_structures(["ketene"])
    for invalid in (0, -3, 1.5):
        with pytest.raises(XTBloomValueError):
            BatchCalculator(structures).compute(auto_batch_size=invalid)


class _FakeCudaFunction:
    def __init__(self, callback: Callable[..., int]) -> None:
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args: object) -> int:
        return self.callback(*args)


class _FakeCudaRuntime:
    def __init__(self, *, query_status: int = 0, restore_status: int = 0) -> None:
        self.set_calls: list[int] = []
        self.query_status = query_status
        self.restore_status = restore_status
        self.cudaGetDevice = _FakeCudaFunction(self._get_device)
        self.cudaSetDevice = _FakeCudaFunction(self._set_device)
        self.cudaMemGetInfo = _FakeCudaFunction(self._memory_info)

    @staticmethod
    def _get_device(pointer: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
        return 0

    def _set_device(self, device: object) -> int:
        value = int(device)
        self.set_calls.append(value)
        return self.restore_status if value == 1 else 0

    def _memory_info(self, free_pointer: object, total_pointer: object) -> int:
        ctypes.cast(free_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 4_000
        ctypes.cast(total_pointer, ctypes.POINTER(ctypes.c_size_t))[0] = 8_000
        return self.query_status


@pytest.mark.parametrize(
    ("query_status", "restore_status", "expected"),
    [(0, 0, (4_000, 8_000)), (1, 0, None), (0, 1, None)],
)
def test_device_memory_info_uses_exact_runtime_and_restores_device(
    monkeypatch: pytest.MonkeyPatch,
    query_status: int,
    restore_status: int,
    expected: tuple[int, int] | None,
) -> None:
    """Use the exact CUDA runtime and restore the caller's current device."""
    runtime = _FakeCudaRuntime(query_status=query_status, restore_status=restore_status)
    loaded = []

    def load_runtime(name: str) -> _FakeCudaRuntime:
        loaded.append(name)
        return runtime

    monkeypatch.setattr(library.ctypes, "CDLL", load_runtime)
    assert library.device_memory_info(0) == expected
    assert loaded == ["libcudart.so.12"]
    assert runtime.set_calls == [0, 1]


@pytest.mark.cuda
def test_device_memory_info_on_available_cuda_runtime() -> None:
    """Return a valid memory pair from an available real CUDA runtime."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    info = library.device_memory_info(0)
    assert info is not None
    free_bytes, total_bytes = info
    assert total_bytes > 0
    assert 0 <= free_bytes <= total_bytes


@pytest.mark.cuda
def test_auto_batch_cuda_matches_unsliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match explicit and automatic CUDA slicing to unsliced execution."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    structures = [
        *_make_structures(["ketene", "h3_plus"]),
        *_make_point_charge_structures(["water_dimer_6pc_hardness"]),
    ] * 2
    unsliced = BatchCalculator(structures, backend="cuda").compute()
    explicit = BatchCalculator(structures, backend="cuda").compute(auto_batch_size=4)

    free_bytes = int(
        (_AUTO_BATCH_RESERVE_BYTES + 4 * _AUTO_BATCH_BYTES_PER_ATOM)
        / _AUTO_BATCH_MEMORY_FRACTION
    )
    monkeypatch.setattr(
        library,
        "device_memory_info",
        lambda _device_id: (free_bytes, free_bytes),
    )
    auto = BatchCalculator(structures, backend="cuda").compute(auto_batch_size=True)

    for result in (explicit, auto):
        np.testing.assert_allclose(
            result.energies, unsliced.energies, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(result.forces, unsliced.forces, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.charges, unsliced.charges, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            result.point_charge_forces,
            unsliced.point_charge_forces,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            result.per_system_status, unsliced.per_system_status
        )
