"""Finite host-queue scheduling contracts, independent of GPU availability."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from xtbloom import BatchCalculator, PointCharge, Structure, library
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError
from xtbloom.interface import _ComputedBatch, _restore_computed_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xtbloom import BatchResult, Context


def _structures() -> list[Structure]:
    """Alternate two repeated AO sizes and retain a rare oversized tail."""
    numbers = [[8, 1, 1], [1, 1], [8, 1, 1], [1, 1], [6, 1, 1, 1, 1]]
    structures = []
    for index, atoms in enumerate(numbers):
        positions = np.zeros((len(atoms), 3))
        positions[:, 0] = index + np.arange(len(atoms)) * 0.1
        structures.append(
            Structure(
                atoms,
                positions,
                point_charges=(
                    PointCharge(
                        np.ones((index, 3)), np.full(index, 0.1), np.ones(index)
                    )
                    if index % 2
                    else None
                ),
                efield=[0.01, 0.0, 0.0] if index == 0 else None,
            )
        )
    return structures


def _computed(structures: Sequence[Structure], *, failed: int = -1) -> _ComputedBatch:
    """Encode canonical identity in every output, with complete peer-local NaNs."""
    ids = np.asarray([int(structure.positions[0, 0]) for structure in structures])
    atoms = np.cumsum([0, *(len(structure) for structure in structures)])
    points = np.cumsum(
        [
            0,
            *(
                len(structure.point_charges.charges)
                if structure.point_charges is not None
                else 0
                for structure in structures
            ),
        ]
    )
    energies = ids.astype(np.float64)
    forces = np.repeat(ids, np.diff(atoms))[:, None] * np.ones((1, 3))
    charges = np.repeat(ids, np.diff(atoms)).astype(np.float64)
    point_forces = (
        np.repeat(ids, np.diff(points))[:, None] * np.ones((1, 3))
        if points[-1]
        else None
    )
    dipoles = ids[:, None] * np.ones((1, 3))
    strain = ids[:, None] * np.ones((1, 9))
    for local_index in np.flatnonzero(ids == failed):
        energies[local_index] = np.nan
        forces[atoms[local_index] : atoms[local_index + 1]] = np.nan
        charges[atoms[local_index] : atoms[local_index + 1]] = np.nan
        if point_forces is not None:
            point_forces[points[local_index] : points[local_index + 1]] = np.nan
        dipoles[local_index] = np.nan
        strain[local_index] = np.nan
    return _ComputedBatch(
        energies=energies,
        forces=forces,
        charges=charges,
        point_charge_forces=point_forces,
        dipole_moments=dipoles,
        strain_derivatives=strain,
        scc_iterations=np.asarray(ids + 10, dtype=np.int32),
        scc_converged=np.asarray(ids != failed, dtype=np.uint8),
        per_system_status=np.where(
            ids == failed, library.STATUS_SCC_NOT_CONVERGED, library.STATUS_SUCCESS
        ).astype(np.int32),
        result_flags=8,
        atom_offsets=atoms,
        point_offsets=points,
        keepalive=[],
    )


def _fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
    structures: list[Structure],
    *,
    failed: int = -1,
) -> tuple[BatchCalculator, list[tuple[int, ...]]]:
    calculator = BatchCalculator(structures, backend="cpu")
    monkeypatch.setattr(
        calculator,
        "_context",
        SimpleNamespace(backend=library.BACKEND_CUDA, device_id=0),
    )
    calls: list[tuple[int, ...]] = []

    def compute(
        _context: Context, chunk: Sequence[Structure], **_kwargs: object
    ) -> _ComputedBatch:
        calls.append(tuple(int(structure.positions[0, 0]) for structure in chunk))
        return _computed(chunk, failed=failed)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    return calculator, calls


def _assert_outputs(actual: BatchResult, expected: _ComputedBatch) -> None:
    for name in (
        "energies",
        "forces",
        "charges",
        "point_charge_forces",
        "dipole_moments",
        "strain_derivatives",
        "scc_iterations",
        "scc_converged",
        "per_system_status",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_auto_groups_globally_and_restores_every_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scatter every ragged output and failed peer without modifying inputs."""
    structures = _structures()
    before = [
        (structure.numbers.copy(), structure.positions.copy())
        for structure in structures
    ]
    calculator, calls = _fake_cuda(monkeypatch, structures, failed=3)
    result = calculator.compute(auto_batch_size=5)
    assert calls == [(1, 3), (0,), (2,), (4,)]
    _assert_outputs(result, _computed(structures, failed=3))
    assert result.failed_indices.tolist() == [3]
    with pytest.raises(XTBloomRuntimeError, match="system 3:"):
        result.raise_for_status()
    for index, structure in enumerate(structures):
        np.testing.assert_array_equal(structure.numbers, before[index][0])
        np.testing.assert_array_equal(structure.positions, before[index][1])
        np.testing.assert_array_equal(
            result[index].get("forces"),
            result.forces[
                result._atom_offsets[index] : result._atom_offsets[index + 1]
            ],
        )
    plan = result.execution_plan
    assert plan is calculator.last_batch_plan
    assert plan is not None
    assert plan.resolved_policy == "exact-ao"
    assert plan.canonical_indices == (1, 3, 0, 2, 4)
    assert plan.inverse_indices == (2, 0, 3, 1, 4)
    assert len(plan.plan_sha256) == 64
    assert plan.grouping_plan_sha256 is not None
    detached = plan.to_dict()
    detached["reason"] = "changed"
    assert plan.to_dict()["reason"] != "changed"


@pytest.mark.parametrize("ordering", ["original", "off"])
def test_original_switch_keeps_original_chunks(
    monkeypatch: pytest.MonkeyPatch, ordering: str
) -> None:
    """Both diagnostic opt-out spellings retain the historical atom slicing."""
    structures = _structures()
    calculator, calls = _fake_cuda(monkeypatch, structures)
    result = calculator.compute(auto_batch_size=5, batch_ordering=ordering)
    assert calls == [(0, 1), (2, 3), (4,)]
    _assert_outputs(result, _computed(structures))
    assert result.execution_plan.resolved_policy == "original"


@pytest.mark.parametrize("size", [None, False])
def test_single_native_contract_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, size: bool | None
) -> None:
    """Default auto must not convert an unsliced request into several calls."""
    calculator, calls = _fake_cuda(monkeypatch, _structures())
    result = calculator.compute(auto_batch_size=size)
    assert calls == [(0, 1, 2, 3, 4)]
    assert result.execution_plan.reason == "single_native_call"
    with pytest.raises(XTBloomValueError, match="single_native_call"):
        calculator.compute(auto_batch_size=size, batch_ordering="exact-ao")
    assert len(calls) == 1


def test_adaptive_oom_preserves_ordinals_and_actual_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record adaptive successful calls separately from the initial AO plan."""
    structures = _structures()
    calculator, _ = _fake_cuda(monkeypatch, structures)
    calls = []

    def compute(
        _context: Context, chunk: Sequence[Structure], **_kwargs: object
    ) -> _ComputedBatch:
        calls.append(tuple(int(structure.positions[0, 0]) for structure in chunk))
        if len(chunk) > 1:
            raise XTBloomRuntimeError("OOM", library.STATUS_ALLOCATION_FAILED)
        return _computed(chunk)

    monkeypatch.setattr("xtbloom.interface._compute_batch", compute)
    monkeypatch.setattr("xtbloom.interface._resolve_auto_batch_limit", lambda *_args: 6)
    result = calculator.compute(auto_batch_size=True)
    _assert_outputs(result, _computed(structures))
    plan = result.execution_plan
    assert plan.planned_batches == ((1, 3), (0, 2), (4,))
    assert plan.batches == ((1,), (3,), (0,), (2,), (4,))
    assert plan.allocation_retries == ((1, 3), (0, 2))
    assert calls == [(1, 3), (1,), (3,), (0, 2), (0,), (2,), (4,)]


def test_rebuild_on_budget_geometry_elements_and_member_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild scheduling dependencies while still submitting current geometry."""
    structures = _structures()
    calculator, _ = _fake_cuda(monkeypatch, structures)
    first = calculator.compute(auto_batch_size=5).execution_plan
    assert first == calculator.compute(auto_batch_size=5).execution_plan
    structures[0].update(
        positions=structures[0].positions + np.asarray([0.0, 0.1, 0.0])
    )
    assert first == calculator.compute(auto_batch_size=5).execution_plan
    calculator.set("max_scc_iterations", 10)
    assert first == calculator.compute(auto_batch_size=5).execution_plan
    assert (
        first.plan_sha256
        != calculator.compute(auto_batch_size=6).execution_plan.plan_sha256
    )
    structures[0].numbers[0] = 1
    changed = calculator.compute(auto_batch_size=5).execution_plan
    assert changed.ao_counts != first.ao_counts
    assert changed.plan_sha256 != first.plan_sha256
    calculator._structures.reverse()
    assert (
        changed.plan_sha256
        != calculator.compute(auto_batch_size=5).execution_plan.plan_sha256
    )


def test_unqualified_paths_and_duplicate_objects_keep_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate references are ordinals, and CPU/GFN1/WARM stay ungrouped."""
    structures = _structures()
    calculator, calls = _fake_cuda(monkeypatch, [structures[0]] * 3)
    homogeneous = calculator.compute(auto_batch_size=3).execution_plan
    assert homogeneous.resolved_policy == "original"
    assert homogeneous.reason == "homogeneous_ao"
    assert calls == [(0,), (0,), (0,)]
    monkeypatch.setattr(calculator._context, "backend", library.BACKEND_CPU)
    assert (
        calculator.compute(auto_batch_size=3).execution_plan.reason
        == "non_cuda_backend"
    )
    with pytest.raises(XTBloomValueError, match="non_cuda_backend"):
        calculator.compute(auto_batch_size=3, batch_ordering="exact-ao")
    monkeypatch.setattr(calculator._context, "backend", library.BACKEND_CUDA)
    calculator._settings.model = library.MODEL_GFN1_XTB
    assert (
        calculator.compute(auto_batch_size=3).execution_plan.reason
        == "unsupported_model"
    )
    calculator._warm_start = True
    assert calculator.compute().execution_plan.reason == "warm_start"
    with pytest.raises(XTBloomValueError, match="warm_start=True"):
        calculator.compute(auto_batch_size=3)


def test_unknown_policy_and_empty_batch_reject_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid policy and the existing empty-batch rule cannot launch work."""
    calculator, calls = _fake_cuda(monkeypatch, _structures())
    with pytest.raises(XTBloomValueError, match="unknown batch ordering"):
        calculator.compute(batch_ordering="invalid")
    with pytest.raises(XTBloomValueError, match="at least one"):
        BatchCalculator([])
    assert not calls


def test_call_error_has_canonical_chunk_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the native status and make chunk-local error indices actionable."""
    calculator, _ = _fake_cuda(monkeypatch, _structures())

    def fail(*_args: object, **_kwargs: object) -> _ComputedBatch:
        raise XTBloomRuntimeError("system 1: invalid", library.STATUS_INVALID_ARGUMENT)

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail)
    with pytest.raises(
        XTBloomRuntimeError, match=r"canonical ordinals \(1, 3\)"
    ) as caught:
        calculator.compute(auto_batch_size=5)
    assert caught.value.status == library.STATUS_INVALID_ARGUMENT


def test_scatter_retains_result_qualifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reordering cannot drop batch-wide force/publication qualifiers."""
    structures = _structures()
    calculator, _ = _fake_cuda(monkeypatch, structures)
    plan = calculator.compute(auto_batch_size=5).execution_plan
    ordered = [structures[index] for index in plan.canonical_indices]
    assert (
        _restore_computed_order(_computed(ordered), plan, structures).result_flags == 8
    )


@pytest.mark.parametrize("indices,limit", [([0], 1), ([0, 1, 4], 3), ([0, 1, 2], 100)])
def test_auto_no_op_controls_preserve_original_calls(
    monkeypatch: pytest.MonkeyPatch, indices: list[int], limit: int
) -> None:
    """Singleton, unique-AO and single-chunk controls never get fragmented."""
    source = _structures()
    structures = [source[index] for index in indices]
    calculator, calls = _fake_cuda(monkeypatch, structures)
    result = calculator.compute(auto_batch_size=limit)
    assert result.execution_plan.resolved_policy == "original"
    assert tuple(index for chunk in calls for index in chunk) == tuple(indices)


def test_nonfinite_input_is_not_repaired_or_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native validator still receives nonfinite input without filtering."""
    structures = _structures()
    calculator, _ = _fake_cuda(monkeypatch, structures)
    structures[3].positions[0, 1] = np.nan
    submitted = []

    def validate(
        _context: Context, chunk: Sequence[Structure], **_kwargs: object
    ) -> _ComputedBatch:
        submitted.extend(chunk)
        if any(not np.isfinite(structure.positions).all() for structure in chunk):
            raise XTBloomRuntimeError(
                "nonfinite positions", library.STATUS_INVALID_ARGUMENT
            )
        return _computed(chunk)

    monkeypatch.setattr("xtbloom.interface._compute_batch", validate)
    with pytest.raises(XTBloomRuntimeError, match="nonfinite positions"):
        calculator.compute(auto_batch_size=5)
    assert structures[3] in submitted
    assert np.isnan(structures[3].positions[0, 1])


def test_explicit_exact_ao_atom_budget_and_strain_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit grouping keeps indivisible systems and rejects the host bridge."""
    structures = _structures()
    calculator, calls = _fake_cuda(monkeypatch, structures)
    result = calculator.compute(auto_batch_size=1, batch_ordering="exact-ao")
    assert len(calls) == len(structures)
    assert all(len(batch) == 1 for batch in result.execution_plan.batches)
    with pytest.raises(XTBloomValueError, match="strain_host_bridge"):
        calculator.compute(
            auto_batch_size=5, batch_ordering="exact-ao", compute_strain=True
        )


def test_invalid_updated_element_is_revalidated_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No cached AO plan can conceal an unsupported mutated atomic number."""
    structures = _structures()
    calculator, calls = _fake_cuda(monkeypatch, structures)
    structures[-1].numbers[0] = 999
    with pytest.raises(XTBloomValueError, match="absent from GFN2"):
        calculator.compute(auto_batch_size=5)
    assert not calls


def test_original_oom_child_error_reports_canonical_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later original-order retry children also need a local-to-input index map."""
    calculator, _ = _fake_cuda(monkeypatch, _structures())

    def fail_later(
        _context: Context, chunk: Sequence[Structure], **_kwargs: object
    ) -> _ComputedBatch:
        if len(chunk) > 1:
            raise XTBloomRuntimeError("OOM", library.STATUS_ALLOCATION_FAILED)
        if int(chunk[0].positions[0, 0]) == 1:
            raise XTBloomRuntimeError("system 0 failed", library.STATUS_INTERNAL_ERROR)
        return _computed(chunk)

    monkeypatch.setattr("xtbloom.interface._compute_batch", fail_later)
    monkeypatch.setattr("xtbloom.interface._resolve_auto_batch_limit", lambda *_args: 5)
    with pytest.raises(XTBloomRuntimeError, match=r"canonical ordinals \(1,\)"):
        calculator.compute(auto_batch_size=True, batch_ordering="original")


@pytest.mark.parametrize("backend,method", [("cpu", "GFN2-xTB"), ("cuda", "GFN1-xTB")])
def test_unqualified_auto_calls_keep_original_order_and_counts(
    monkeypatch: pytest.MonkeyPatch, backend: str, method: str
) -> None:
    """Changing only auto policy cannot merge, reorder or add fallback calls."""
    structures = _structures()
    calculator, calls = _fake_cuda(monkeypatch, structures)
    monkeypatch.setattr(
        calculator._context,
        "backend",
        library.BACKEND_CPU if backend == "cpu" else library.BACKEND_CUDA,
    )
    calculator._settings.model = (
        library.MODEL_GFN1_XTB if method == "GFN1-xTB" else library.MODEL_GFN2_XTB
    )
    monkeypatch.setattr("xtbloom.interface._resolve_auto_batch_limit", lambda *_args: 5)
    calculator.compute(auto_batch_size=True)
    assert calls == [(0, 1), (2, 3), (4,)]
    calls.clear()
    calculator._warm_start = True
    calculator.compute()
    assert calls == [(0, 1, 2, 3, 4)]


def test_grouped_host_publication_against_real_cpu_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real native ragged outputs behind simulated CUDA eligibility.

    Only the eligibility metadata is simulated. The solver is explicitly CPU,
    so this verifies host packing/scatter, not CUDA parity or performance.
    """
    import _cases
    from xtbloom import Context
    from xtbloom.interface import _compute_batch

    structures = []
    for case_id in ["ketene", "h3_plus", "water_dimer_6pc_hardness"] * 2:
        case = _cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
        point_charges = None
        if case_id == "water_dimer_6pc_hardness":
            point_positions, values, gammas = _cases.qmmm_points(case)
            point_charges = PointCharge(point_positions, values, gammas)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin,
                point_charges=point_charges,
                efield=[0.001, 0.0, 0.0],
            )
        )
    expected = BatchCalculator(structures, backend="cpu").compute(auto_batch_size=8)
    calculator, _ = _fake_cuda(monkeypatch, structures)
    with Context("cpu") as context:

        def native_cpu(
            _context: Context, chunk: Sequence[Structure], **kwargs: object
        ) -> _ComputedBatch:
            return _compute_batch(context, chunk, **kwargs)

        monkeypatch.setattr("xtbloom.interface._compute_batch", native_cpu)
        actual = calculator.compute(auto_batch_size=8)
    assert actual.execution_plan.resolved_policy == "exact-ao"
    assert not actual.failed_indices.size
    for name in (
        "energies",
        "forces",
        "charges",
        "point_charge_forces",
        "dipole_moments",
        "scc_iterations",
        "scc_converged",
        "per_system_status",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
