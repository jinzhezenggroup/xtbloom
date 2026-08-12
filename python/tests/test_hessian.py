"""Tests for batched numerical Hessians from analytic xTBloom forces."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import _cases
import numpy as np
import pytest
from xtbloom import Calculator, ChargeResponse, Context, PointCharge, Structure, library
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError
from xtbloom.interface import _ComputedBatch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _h2_calculator(
    *,
    warm_start: bool = False,
    charge_response: ChargeResponse | None = None,
    efield: np.ndarray | list[float] | None = None,
) -> Calculator:
    """Return a small calculator suitable for control-flow and CPU tests."""
    return Calculator(
        "GFN2-xTB",
        [1, 1],
        np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]]),
        backend="cpu",
        warm_start=warm_start,
        charge_response=charge_response,
        efield=efield,
    )


def _fake_computed(
    structures: Sequence[Structure],
    force: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    failed_index: int | None = None,
) -> _ComputedBatch:
    """Build native-shaped forces and diagnostics for Hessian control tests."""
    atom_offsets = np.cumsum([0, *(len(structure) for structure in structures)])
    total_atoms = int(atom_offsets[-1])
    forces = np.zeros((total_atoms, 3), dtype=np.float64)
    if force is not None:
        for index, structure in enumerate(structures):
            begin = int(atom_offsets[index])
            end = int(atom_offsets[index + 1])
            forces[begin:end] = force(structure.positions).reshape(-1, 3)

    statuses = np.full(len(structures), library.STATUS_SUCCESS, dtype=np.int32)
    converged = np.ones(len(structures), dtype=np.uint8)
    iterations = np.full(len(structures), 7, dtype=np.int32)
    if failed_index is not None:
        statuses[failed_index] = library.STATUS_SCC_NOT_CONVERGED
        converged[failed_index] = 0
        iterations[failed_index] = 11
        begin = int(atom_offsets[failed_index])
        end = int(atom_offsets[failed_index + 1])
        forces[begin:end] = np.nan

    return _ComputedBatch(
        energies=np.empty(len(structures), dtype=np.float64),
        forces=forces,
        charges=np.empty(total_atoms, dtype=np.float64),
        point_charge_forces=None,
        dipole_moments=None,
        scc_iterations=iterations,
        scc_converged=converged,
        per_system_status=statuses,
        result_flags=0,
        atom_offsets=atom_offsets,
        point_offsets=None,
        keepalive=[],
    )


@pytest.mark.parametrize("step", [True, np.bool_(False), 0.0, -0.1, np.nan, np.inf])
def test_hessian_rejects_invalid_steps(step: object) -> None:
    """Require a finite positive Cartesian displacement."""
    with _h2_calculator() as calculator, pytest.raises(XTBloomValueError):
        calculator.hessian(step=step)  # type: ignore[arg-type]


@pytest.mark.parametrize("auto_batch_size", [0, -3, 1.5, True + 0j])
def test_hessian_rejects_invalid_batch_limits_before_native_execution(
    monkeypatch: pytest.MonkeyPatch,
    auto_batch_size: object,
) -> None:
    """Reject invalid atom limits before resolving or creating a native context."""

    def unexpected_create(_context: Context) -> NoReturn:
        raise AssertionError("invalid Hessian input reached native context creation")

    monkeypatch.setattr(Context, "_create", unexpected_create)
    with _h2_calculator() as calculator, pytest.raises(XTBloomValueError):
        calculator.hessian(auto_batch_size=auto_batch_size)  # type: ignore[arg-type]
    assert calculator._context._handle is None


def test_hessian_sign_layout_displacement_order_and_forces_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover ``F=-Kx`` exactly with flattened ``+,-`` coordinate ordering."""
    matrix = (np.arange(36, dtype=np.float64).reshape(6, 6) + 1.0) / 7.0
    original = np.array([[-0.71, 0.1, -0.2], [0.71, -0.3, 0.4]])
    calls: list[list[Structure]] = []
    keyword_calls: list[dict[str, object]] = []

    def compute(
        _context: Context,
        structures: Sequence[Structure],
        **kwargs: object,
    ) -> _ComputedBatch:
        calls.append(list(structures))
        keyword_calls.append(kwargs)
        return _fake_computed(
            structures,
            lambda positions: -(matrix @ positions.reshape(-1)),
        )

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    with Calculator("GFN2-xTB", [1, 1], original, backend="cpu") as calculator:
        actual = calculator.hessian(step=0.01, auto_batch_size=None)
        np.testing.assert_array_equal(calculator.positions, original)

    np.testing.assert_allclose(actual, matrix, atol=3.0e-14, rtol=0.0)
    assert [len(chunk) for chunk in calls] == [12]
    assert keyword_calls[0]["flags"] == library.COMPUTE_FORCES
    assert keyword_calls[0]["warm_start"] is False

    deltas = [
        structure.positions.reshape(-1) - original.reshape(-1) for structure in calls[0]
    ]
    expected = []
    for coordinate in range(6):
        plus = np.zeros(6)
        plus[coordinate] = 0.01
        expected.extend([plus, -plus])
    np.testing.assert_allclose(deltas, expected, atol=1.0e-17, rtol=0.0)


def test_hessian_displacements_preserve_fixed_external_attachments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy every fixed external attachment to each displaced structure."""
    points = PointCharge(
        positions=np.array([[4.0, 0.0, 0.0]]),
        charges=np.array([0.5]),
        gammas=np.array([0.405771]),
    )
    response = ChargeResponse(
        shifts=np.array([0.003, -0.002]),
        matrix=np.array([[0.02, 0.001], [0.001, 0.018]]),
    )
    field = np.array([0.004, -0.003, 0.002])
    calls: list[dict[str, object]] = []

    def compute(
        _context: Context,
        structures: Sequence[Structure],
        **kwargs: object,
    ) -> _ComputedBatch:
        for structure in structures:
            assert structure.point_charges is points
            assert structure.charge_response is response
            np.testing.assert_array_equal(structure.efield, field)
        calls.append(kwargs)
        return _fake_computed(structures)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    with Calculator(
        "GFN2-xTB",
        [1, 1],
        np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]]),
        point_charges=points,
        charge_response=response,
        efield=field,
        backend="cpu",
    ) as calculator:
        calculator.hessian(auto_batch_size=None)

    assert len(calls) == 1
    assert calls[0]["flags"] == (
        library.COMPUTE_FORCES | library.COMPUTE_POINT_CHARGE_FORCES
    )


@pytest.mark.parametrize(
    ("auto_batch_size", "automatic_limit", "expected_calls"),
    [
        (4, None, [2, 2, 2, 2, 2, 2]),
        (True, 6, [3, 3, 3, 3]),
    ],
)
def test_hessian_materializes_only_one_displacement_chunk(
    monkeypatch: pytest.MonkeyPatch,
    auto_batch_size: bool | int,
    automatic_limit: int | None,
    expected_calls: list[int],
) -> None:
    """Translate atom limits to lazy equal-topology displacement chunks."""
    calls = []

    def compute(
        _context: Context,
        structures: Sequence[Structure],
        **_kwargs: object,
    ) -> _ComputedBatch:
        calls.append(len(structures))
        return _fake_computed(structures)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    if automatic_limit is not None:
        monkeypatch.setattr(
            "xtbloom.interface._resolve_auto_batch_limit_for_total_atoms",
            lambda _context, _total_atoms: automatic_limit,
        )
    with _h2_calculator() as calculator:
        calculator.hessian(auto_batch_size=auto_batch_size)
    assert calls == expected_calls


def test_hessian_auto_retries_allocation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bisect only automatic displacement chunks after a native allocation error."""
    calls = []

    def compute(
        _context: Context,
        structures: Sequence[Structure],
        **_kwargs: object,
    ) -> _ComputedBatch:
        calls.append(len(structures))
        if len(structures) > 3:
            raise XTBloomRuntimeError(
                "synthetic allocation failure", library.STATUS_ALLOCATION_FAILED
            )
        return _fake_computed(structures)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    monkeypatch.setattr(
        "xtbloom.interface._resolve_auto_batch_limit_for_total_atoms",
        lambda _context, _total_atoms: 1_000,
    )
    with _h2_calculator() as calculator:
        calculator.hessian(auto_batch_size=True)
    assert calls == [12, 6, 3, 3, 6, 3, 3]


@pytest.mark.parametrize("auto_batch_size", [None, 100])
def test_hessian_nonautomatic_allocation_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    auto_batch_size: int | None,
) -> None:
    """Do not silently override an explicit caller batching decision."""

    def fail(
        _context: Context,
        _structures: Sequence[Structure],
        **_kwargs: object,
    ) -> NoReturn:
        raise XTBloomRuntimeError(
            "synthetic allocation failure", library.STATUS_ALLOCATION_FAILED
        )

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail)
    original = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    with _h2_calculator() as calculator:
        with pytest.raises(XTBloomRuntimeError):
            calculator.hessian(auto_batch_size=auto_batch_size)
        np.testing.assert_array_equal(calculator.positions, original)


def test_hessian_failure_identifies_displacement_and_preserves_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a batch-local failure to its global atom, Cartesian axis, and sign."""
    calls = 0

    def compute(
        _context: Context,
        structures: Sequence[Structure],
        **_kwargs: object,
    ) -> _ComputedBatch:
        nonlocal calls
        calls += 1
        # An atom limit of four gives two H2 displacements per native call;
        # global displacement 9 is local index 1 of the fifth chunk.
        return _fake_computed(structures, failed_index=1 if calls == 5 else None)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    original = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    with _h2_calculator() as calculator:
        with pytest.raises(XTBloomRuntimeError) as caught:
            calculator.hessian(auto_batch_size=4)
        np.testing.assert_array_equal(calculator.positions, original)
    assert caught.value.status == library.STATUS_SCC_NOT_CONVERGED
    assert "atom 1, axis y, displacement -step" in str(caught.value)
    assert "scc_converged=0, iterations=11" in str(caught.value)


def test_hessian_does_not_consume_original_warm_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use an independent context and leave the original warm state untouched."""
    contexts = []

    def compute(
        context: Context,
        structures: Sequence[Structure],
        **_kwargs: object,
    ) -> _ComputedBatch:
        contexts.append(context)
        context._warm_ready = True
        return _fake_computed(structures)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    with _h2_calculator(warm_start=True) as calculator:
        calculator._context._warm_ready = True
        calculator.hessian(auto_batch_size=2)
        assert calculator._context._warm_ready
        assert contexts
        assert all(context is not calculator._context for context in contexts)


def test_h2_hessian_is_finite_symmetric_and_translationally_invariant() -> None:
    """Exercise the real CPU force path on a symmetry-constrained molecule."""
    with _h2_calculator() as calculator:
        hessian = calculator.hessian(step=0.005, auto_batch_size=2)
    assert hessian.shape == (6, 6)
    assert hessian.dtype == np.float64
    assert hessian.flags.c_contiguous
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-12, rtol=0.0)
    blocks = hessian.reshape(2, 3, 2, 3)
    np.testing.assert_allclose(blocks.sum(axis=2), 0.0, atol=1.0e-11, rtol=0.0)


def test_h2_hessian_directional_curvature_matches_energy_second_difference() -> None:
    """Validate a Hessian projection against independent energy differences."""
    reference = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    direction = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    with Calculator(
        "GFN2-xTB",
        [1, 1],
        reference,
        backend="cpu",
        max_scc_iterations=500,
        charge_tolerance=1.0e-8,
        energy_tolerance=1.0e-11,
    ) as calculator:
        hessian = calculator.hessian(step=0.0025, auto_batch_size=2)

        def energy_at(displacement: float) -> float:
            positions = reference.reshape(-1) + displacement * direction
            calculator.update(positions=positions.reshape(reference.shape))
            return calculator.singlepoint().energy

        energy_zero = energy_at(0.0)
        curvatures = [
            (energy_at(delta) - 2.0 * energy_zero + energy_at(-delta)) / delta**2
            for delta in (0.01, 0.005)
        ]
        calculator.update(positions=reference)

    projected = float(direction @ hessian @ direction)
    coarse_error = abs(curvatures[0] - projected)
    fine_error = abs(curvatures[1] - projected)
    assert fine_error < 0.3 * coarse_error
    np.testing.assert_allclose(curvatures[1], projected, atol=5.0e-5, rtol=0.0)


def _calculator_from_case(case_id: str) -> Calculator:
    """Construct a calculator from one canonical conformance input."""
    case = _cases.case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    efield = case.get("efield")
    return Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        efield=efield,
        backend="cpu",
    )


def test_water_central_difference_convergence_symmetrization_and_chunking() -> None:
    """Show second-order matrix convergence on a real non-linear molecule."""
    case = _cases.case_by_id("water_one_pc_gamma999")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cpu",
    ) as calculator:
        coarse = calculator.hessian(step=0.01, auto_batch_size=None)
        fine = calculator.hessian(step=0.005, auto_batch_size=None)
        finest = calculator.hessian(step=0.0025, auto_batch_size=None)
        chunked = calculator.hessian(step=0.005, auto_batch_size=6)
        symmetric = calculator.hessian(step=0.005, auto_batch_size=6, symmetrize=True)

    coarse_difference = np.max(np.abs(coarse - fine))
    fine_difference = np.max(np.abs(fine - finest))
    assert fine_difference < 0.35 * coarse_difference
    coarse_antisymmetry = np.max(np.abs(coarse - coarse.T))
    fine_antisymmetry = np.max(np.abs(fine - fine.T))
    assert fine_antisymmetry < 0.35 * coarse_antisymmetry
    np.testing.assert_array_equal(chunked, fine)
    np.testing.assert_array_equal(symmetric, 0.5 * (fine + fine.T))


def test_open_shell_and_fixed_charge_response_hessians_are_finite() -> None:
    """Cover restricted open-shell SCC and fixed caller-owned response data."""
    with _calculator_from_case("oh_radical") as calculator:
        open_shell = calculator.hessian(auto_batch_size=2)

    response = ChargeResponse(
        shifts=np.array([0.003, -0.002]),
        matrix=np.array([[0.02, 0.001], [0.001, 0.018]]),
    )
    with _h2_calculator(charge_response=response) as calculator:
        periodic = calculator.hessian(auto_batch_size=2)

    assert np.isfinite(open_shell).all()
    assert np.isfinite(periodic).all()


def test_fixed_electric_field_hessian_leaves_field_unchanged() -> None:
    """Keep the CPU electric-field attachment fixed during QM displacements."""
    field = np.array([0.003, -0.004, 0.005])
    original = np.array(field, copy=True)
    with _h2_calculator(efield=field) as calculator:
        hessian = calculator.hessian(auto_batch_size=2)
        np.testing.assert_array_equal(calculator.efield, original)
    assert np.isfinite(hessian).all()


def test_fixed_point_charge_hessian_leaves_embedding_unchanged() -> None:
    """Displace only QM coordinates while retaining external point-charge data."""
    case = _cases.case_by_id("water_one_pc_gamma999")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    point_inputs = _cases.qmmm_points(case)
    assert point_inputs is not None
    point_positions, point_values, point_gammas = point_inputs
    points = PointCharge(point_positions, point_values, point_gammas)
    original_points = np.array(points.positions, copy=True)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=points,
        backend="cpu",
    ) as calculator:
        hessian = calculator.hessian(auto_batch_size=6)
    assert np.isfinite(hessian).all()
    np.testing.assert_array_equal(points.positions, original_points)


def _library_has_cuda() -> bool:
    """Return whether this test environment can create a CUDA context."""
    try:
        with Context("cuda"):
            pass
        return True
    except XTBloomRuntimeError:
        return False


@pytest.mark.cuda
def test_cuda_hessian_matches_cpu_and_explicit_chunking() -> None:
    """Require CPU/CUDA parity and stable CUDA displacement chunking."""
    if not _library_has_cuda():
        pytest.skip("xTBloom CUDA backend is unavailable")
    positions = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    with Calculator("GFN2-xTB", [1, 1], positions, backend="cpu") as calculator:
        cpu = calculator.hessian(auto_batch_size=None)
    with Calculator("GFN2-xTB", [1, 1], positions, backend="cuda") as calculator:
        cuda = calculator.hessian(auto_batch_size=None)
        chunked = calculator.hessian(auto_batch_size=2)
    np.testing.assert_allclose(cuda, cpu, atol=1.0e-8, rtol=1.0e-8)
    np.testing.assert_allclose(chunked, cuda, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.cuda
def test_cuda_open_shell_and_point_charge_hessians_match_cpu() -> None:
    """Cover CUDA unrestricted/embedded force publication used by Hessians."""
    if not _library_has_cuda():
        pytest.skip("xTBloom CUDA backend is unavailable")

    case = _cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, _spin = _cases.structure_inputs(case)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=2,
        backend="cpu",
    ) as calculator:
        cpu_open_shell = calculator.hessian(auto_batch_size=2)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=2,
        backend="cuda",
    ) as calculator:
        cuda_open_shell = calculator.hessian(auto_batch_size=2)
    np.testing.assert_allclose(
        cuda_open_shell, cpu_open_shell, atol=1.0e-8, rtol=1.0e-8
    )

    case = _cases.case_by_id("water_one_pc_gamma999")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    point_inputs = _cases.qmmm_points(case)
    assert point_inputs is not None
    point_positions, point_values, point_gammas = point_inputs
    points = PointCharge(point_positions, point_values, point_gammas)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=points,
        backend="cpu",
    ) as calculator:
        cpu_embedded = calculator.hessian(auto_batch_size=6)
    with Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=points,
        backend="cuda",
    ) as calculator:
        cuda_embedded = calculator.hessian(auto_batch_size=6)
    np.testing.assert_allclose(cuda_embedded, cpu_embedded, atol=1.0e-8, rtol=1.0e-8)
