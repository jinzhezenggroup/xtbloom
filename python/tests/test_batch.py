"""Tests for the native ragged-batch interface (``gpuxtb.BatchCalculator``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import _cases
import numpy as np
import pytest
from gpuxtb import BatchCalculator, Calculator, PointCharge, Structure
from gpuxtb.exceptions import GPUxtbRuntimeError

if TYPE_CHECKING:
    from collections.abc import Sequence

MOLECULAR_CASES = [
    "ketene",
    "nenacl",
    "h3_plus",
    "sif5_minus",
    "oh_radical",
]


def _make_structures(case_ids: Sequence[str], **kwargs: object) -> list[Structure]:
    """Build structures from named molecular conformance cases."""
    structures: list[Structure] = []
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


@pytest.mark.parametrize("case_id", MOLECULAR_CASES)
def test_batch_single_matches_serial(case_id: str) -> None:
    """Match a one-system batch to serial calculation output."""
    case = _cases.case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)

    single = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    serial = single.singlepoint()

    batch = BatchCalculator(_make_structures([case_id]))
    result = batch.compute()
    assert result.energies[0] == pytest.approx(serial.energy, abs=1e-12)
    assert result.forces == pytest.approx(serial.forces, abs=1e-12)
    assert result[0].energy == pytest.approx(serial.energy, abs=1e-12)


def test_batch_matches_goldens_all_molecular_cases() -> None:
    """Match every molecular batch slice to its conformance golden."""
    structures = _make_structures(MOLECULAR_CASES)
    batch = BatchCalculator(structures)
    result = batch.compute()
    tolerances = _cases.tolerances()
    for index, case_id in enumerate(MOLECULAR_CASES):
        golden = _cases.golden(_cases.case_by_id(case_id))
        item = result[index]
        assert item.energy == pytest.approx(
            golden["energy_hartree"], abs=tolerances["energy"]["atol"]
        ), case_id
        assert item.forces == pytest.approx(
            np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3),
            abs=tolerances["forces"]["atol"],
        ), case_id
    assert result[-1].energy == pytest.approx(result[len(result) - 1].energy, abs=0.0)
    assert result[-1].forces == pytest.approx(result[len(result) - 1].forces, abs=0.0)


def test_batch_preserves_healthy_peer_when_another_fails() -> None:
    """Preserve a successful peer when another system does not converge."""
    # h3+ converges in four iterations while NeNaCl does not.  The high-level
    # batch API must retain the healthy result and expose peer-local status.
    calculator = BatchCalculator(
        _make_structures(["h3_plus", "nenacl"]),
        backend="cpu",
        max_scc_iterations=4,
    )
    result = calculator.compute()
    assert result.failed_indices.tolist() == [1]
    assert result.per_system_status[0] == 0
    assert np.isfinite(result.energies[0])
    assert np.isnan(result.energies[1])
    with pytest.raises(GPUxtbRuntimeError):
        result.raise_for_status()
    with pytest.raises(GPUxtbRuntimeError):
        calculator.compute(raise_on_failure=True)


def test_batch_mixed_charge_and_spin_diagnostics() -> None:
    """Publish diagnostics for a batch with mixed charges and spin states."""
    charges = {
        "ketene": (0, 0),
        "h3_plus": (1, 0),
        "sif5_minus": (-1, 0),
        "oh_radical": (0, 1),
    }
    structures = []
    for case_id, (charge, uhf) in charges.items():
        numbers, positions, _, _, _ = _cases.structure_inputs(
            _cases.case_by_id(case_id)
        )
        structures.append(Structure(numbers, positions, charge=charge, uhf=uhf))
    batch = BatchCalculator(structures)
    result = batch.compute()
    assert len(result) == 4
    assert result.scc_converged.shape == (4,)
    assert (result.scc_converged == 1).all()
    assert result.per_system_status.shape == (4,)
    assert (result.per_system_status == 0).all()
    assert (result.scc_iterations > 0).all()


def test_batch_point_charges() -> None:
    """Match point-charge batch results to QM/MM conformance goldens."""
    case_ids = [
        "water_one_pc_gamma999",
        "water_dimer_6pc_hardness",
        "water_dimer_6pc_gamma999",
    ]
    structures = []
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
    batch = BatchCalculator(structures)
    result = batch.compute()
    assert result.point_charge_forces is not None
    tolerances = _cases.tolerances()
    for index, case_id in enumerate(case_ids):
        golden = _cases.golden(_cases.case_by_id(case_id))
        item = result[index]
        assert item.energy == pytest.approx(
            golden["energy_hartree"], abs=tolerances["energy"]["atol"]
        ), case_id
        assert item.forces == pytest.approx(
            np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3),
            abs=tolerances["forces"]["atol"],
        ), case_id


def test_batch_empty_rejected() -> None:
    """Reject a batch that contains no structures."""
    from gpuxtb.exceptions import GPUxtbValueError

    with pytest.raises(GPUxtbValueError):
        BatchCalculator([])
