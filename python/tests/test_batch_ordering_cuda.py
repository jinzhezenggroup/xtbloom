"""Real-CUDA regressions for finite GFN2 exact-AO batch ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import _cases
import numpy as np
import pytest
from xtbloom import BatchCalculator, Context, PointCharge, Structure, library
from xtbloom.exceptions import XTBloomRuntimeError

if TYPE_CHECKING:
    from xtbloom import BatchResult, Result


def _library_has_cuda() -> bool:
    """Return whether this host can create the requested CUDA backend."""
    try:
        with Context("cuda"):
            pass
        return True
    except XTBloomRuntimeError:
        return False


def _conformance_structures(
    case_ids: tuple[str, ...], *, include_efield: bool = True
) -> list[Structure]:
    """Build GFN2 host descriptors from the pinned molecular/QM-MM inputs."""
    structures = []
    for case_id in case_ids:
        case = _cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin_channels = _cases.structure_inputs(case)
        point_charges = None
        if case.get("input_schema") == "qmmm-v1":
            point_data = _cases.qmmm_points(case)
            assert point_data is not None
            point_positions, point_values, point_gammas = point_data
            point_charges = PointCharge(point_positions, point_values, point_gammas)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin_channels,
                point_charges=point_charges,
                efield=[0.001, 0.0, 0.0] if include_efield else None,
            )
        )
    return structures


def _assert_batch_outputs_equal(actual: BatchResult, expected: BatchResult) -> None:
    """Compare every requested public molecular output at layout tolerance."""
    for name in (
        "energies",
        "forces",
        "charges",
        "point_charge_forces",
        "dipole_moments",
    ):
        actual_values = getattr(actual, name)
        expected_values = getattr(expected, name)
        if expected_values is None:
            assert actual_values is None
        else:
            assert actual_values is not None
            np.testing.assert_allclose(
                actual_values, expected_values, rtol=0.0, atol=1.0e-12
            )
    for name in ("per_system_status", "scc_iterations", "scc_converged"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def _assert_single_outputs_equal(actual: Result, expected: Result) -> None:
    """Compare one public system result with an independent single-system call."""
    for name in (
        "energy",
        "forces",
        "charges",
        "point_charge_forces",
        "dipole_moments",
    ):
        actual_values = actual.get(name)
        expected_values = expected.get(name)
        if expected_values is None:
            if name == "point_charge_forces" and actual_values is not None:
                # A mixed QM/MM batch retains empty point slices for QM-only peers.
                assert np.asarray(actual_values).shape == (0, 3)
            else:
                assert actual_values is None
        else:
            assert actual_values is not None
            np.testing.assert_allclose(
                actual_values, expected_values, rtol=0.0, atol=1.0e-12
            )
    for name in ("scc_status", "scc_iterations", "scc_converged"):
        assert actual.get(name) == expected.get(name)


def _input_arrays(structure: Structure) -> tuple[np.ndarray, ...]:
    """Return the caller arrays exercised by these molecular/QM-MM cases."""
    arrays = [structure.numbers, structure.positions]
    if structure.efield is not None:
        arrays.append(structure.efield)
    if structure.point_charges is not None:
        arrays.extend(
            array
            for array in (
                structure.point_charges.positions,
                structure.point_charges.charges,
                structure.point_charges.gammas,
            )
            if array is not None
        )
    return tuple(arrays)


def _assert_input_arrays_unchanged(
    structures: list[Structure], snapshots: list[tuple[np.ndarray, ...]]
) -> None:
    """Ensure native batching did not mutate caller-owned host arrays."""
    for structure, snapshot in zip(structures, snapshots, strict=True):
        arrays = _input_arrays(structure)
        assert len(arrays) == len(snapshot)
        for actual, expected in zip(arrays, snapshot, strict=True):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.cuda
def test_cuda_auto_grouping_restores_conformance_outputs() -> None:
    """Group repeated AO classes and restore all outputs to caller order."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")

    structures = _conformance_structures(
        ("ketene", "h3_plus", "water_dimer_6pc_hardness") * 2
    )
    input_snapshots = [
        tuple(array.copy() for array in _input_arrays(structure))
        for structure in structures
    ]

    with BatchCalculator(structures, backend="cuda", warm_start=False) as calculator:
        automatic = calculator.compute(auto_batch_size=7)
        original = calculator.compute(auto_batch_size=7, batch_ordering="original")

    plan = automatic.execution_plan
    assert plan is not None
    assert plan.requested_policy == "auto"
    assert plan.resolved_policy == "exact-ao"
    assert plan.reason == "cross_batch_exact_ao"
    assert plan.max_total_atoms == 7
    assert plan.atom_counts == (5, 3, 6, 5, 3, 6)
    assert plan.ao_counts == (14, 3, 12, 14, 3, 12)
    assert plan.canonical_indices == (1, 4, 2, 5, 0, 3)
    assert plan.inverse_indices == (4, 0, 2, 5, 1, 3)
    assert plan.planned_batches == ((1, 4), (2,), (5,), (0,), (3,))
    assert plan.batches == plan.planned_batches
    assert all(
        sum(plan.atom_counts[index] for index in batch) <= plan.max_total_atoms
        for batch in plan.batches
    )
    assert automatic.failed_indices.size == 0
    assert automatic.point_charge_forces is not None
    assert automatic.dipole_moments is not None
    _assert_batch_outputs_equal(automatic, original)

    for index, structure in enumerate(structures):
        with BatchCalculator(
            [structure], backend="cuda", warm_start=False
        ) as calculator:
            individual = calculator.compute()
        _assert_single_outputs_equal(automatic[index], individual[0])

    _assert_input_arrays_unchanged(structures, input_snapshots)


@pytest.mark.cuda
def test_cuda_auto_grouping_preserves_peer_local_failures() -> None:
    """Restore failed-system diagnostics and NaN slices after CUDA ordering."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")

    structures = _conformance_structures(
        ("h3_plus", "nenacl", "ketene", "h3_plus"), include_efield=False
    )
    with BatchCalculator(
        structures,
        backend="cuda",
        max_scc_iterations=4,
        warm_start=False,
    ) as calculator:
        automatic = calculator.compute(auto_batch_size=7)
        original = calculator.compute(auto_batch_size=7, batch_ordering="original")

    plan = automatic.execution_plan
    assert plan is not None
    assert plan.resolved_policy == "exact-ao"
    assert plan.reason == "cross_batch_exact_ao"
    assert plan.max_total_atoms == 7
    assert plan.canonical_indices == (0, 3, 2, 1)
    assert plan.inverse_indices == (0, 3, 2, 1)
    assert plan.planned_batches == ((0, 3), (2,), (1,))
    assert plan.batches == plan.planned_batches
    np.testing.assert_array_equal(automatic.failed_indices, original.failed_indices)
    assert automatic.failed_indices.size > 0
    assert automatic.failed_indices.size < len(structures)
    for name in ("energies", "forces", "charges", "dipole_moments"):
        actual_values = getattr(automatic, name)
        expected_values = getattr(original, name)
        if expected_values is None:
            assert actual_values is None
        else:
            assert actual_values is not None
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
    _assert_batch_outputs_equal(automatic, original)

    for index in range(len(structures)):
        ordered_result = automatic[index]
        baseline_result = original[index]
        if index in automatic.failed_indices:
            assert np.isnan(ordered_result.energy)
            assert np.isnan(ordered_result.forces).all()
            assert np.isnan(ordered_result.charges).all()
            assert (
                ordered_result.scc_status != library.STATUS_SUCCESS
                or not ordered_result.scc_converged
            )
            assert not ordered_result.scc_converged
        else:
            assert np.isfinite(ordered_result.energy)
            assert np.isfinite(ordered_result.forces).all()
            assert np.isfinite(ordered_result.charges).all()
            assert ordered_result.scc_status == library.STATUS_SUCCESS
            assert ordered_result.scc_converged
        assert ordered_result.scc_status == baseline_result.scc_status
        assert ordered_result.scc_iterations == baseline_result.scc_iterations
        assert ordered_result.scc_converged == baseline_result.scc_converged


@pytest.mark.cuda
def test_cuda_repeated_fresh_calls_rebuild_from_public_updates() -> None:
    """Recompute fresh results after geometry changes and a new atom budget."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")

    structures = _conformance_structures(
        ("ketene", "h3_plus", "water_dimer_6pc_hardness") * 2
    )
    with BatchCalculator(structures, backend="cuda", warm_start=False) as calculator:
        first = calculator.compute(auto_batch_size=7)
        repeated = calculator.compute(auto_batch_size=7)
        assert repeated.execution_plan is not first.execution_plan
        assert repeated.execution_plan == first.execution_plan
        _assert_batch_outputs_equal(repeated, first)

        moved_positions = structures[1].positions.copy()
        moved_positions[0, 0] += 0.1
        structures[1].update(positions=moved_positions)
        moved = calculator.compute(auto_batch_size=7)
        moved_original = calculator.compute(
            auto_batch_size=7, batch_ordering="original"
        )
        assert moved.execution_plan is not first.execution_plan
        assert moved.execution_plan == first.execution_plan
        _assert_batch_outputs_equal(moved, moved_original)
        assert not np.isclose(moved[1].energy, first[1].energy, rtol=0.0, atol=1.0e-12)
        np.testing.assert_array_equal(structures[1].positions, moved_positions)
        np.testing.assert_array_equal(moved_positions[0], structures[1].positions[0])

        larger_budget = calculator.compute(auto_batch_size=10)
        larger_plan = larger_budget.execution_plan
        moved_plan = moved.execution_plan
        assert larger_plan is not None
        assert moved_plan is not None
        assert larger_plan is not moved_plan
        assert larger_plan.max_total_atoms == 10
        assert larger_plan.plan_sha256 != moved_plan.plan_sha256
        assert larger_plan.planned_batches != moved_plan.planned_batches
        assert larger_budget.failed_indices.size == 0
        assert larger_budget.point_charge_forces is not None
        assert larger_budget.dipole_moments is not None
        _assert_batch_outputs_equal(larger_budget, moved_original)
