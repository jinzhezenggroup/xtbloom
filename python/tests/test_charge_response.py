"""Tests for the periodic charge-response (b + A q) Python exposure."""

from __future__ import annotations

import numpy as np
import pytest
from gpuxtb import BatchCalculator, Calculator, ChargeResponse, Structure
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError
from gpuxtb.interface import _pack_charge_responses

H2_POSITIONS = np.array(
    [
        [-0.71, 0.0, 0.0],
        [0.71, 0.0, 0.0],
    ]
)


def _h2(charge_response=None):
    return Calculator(
        "GFN2-xTB",
        numbers=[1, 1],
        positions=H2_POSITIONS,
        charge_response=charge_response,
    )


def test_zero_charge_response_matches_baseline():
    baseline = _h2().singlepoint()
    zero = _h2(
        ChargeResponse(shifts=[0.0, 0.0], matrix=[[0.0, 0.0], [0.0, 0.0]])
    ).singlepoint()
    assert zero.energy == pytest.approx(baseline.energy, rel=1.0e-12, abs=1.0e-10)
    assert zero.forces == pytest.approx(baseline.forces, abs=1.0e-9)
    assert zero.charges == pytest.approx(baseline.charges, abs=1.0e-9)


def test_charge_response_changes_energy():
    baseline = _h2().singlepoint()
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    result = _h2(response).singlepoint()
    assert result.scc_converged
    assert result.scc_status == 0
    assert np.isfinite(result.energy)
    assert not np.isclose(result.energy, baseline.energy, rtol=0.0, atol=1.0e-10)
    assert not np.allclose(result.charges, baseline.charges, rtol=0.0, atol=1.0e-10)


def test_single_matches_batch_and_sequential():
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    single = _h2(response).singlepoint()

    structure = Structure([1, 1], H2_POSITIONS, charge_response=response)
    batch = BatchCalculator([structure]).compute(raise_on_failure=True)
    assert batch.energies[0] == pytest.approx(single.energy, rel=1.0e-12, abs=1.0e-12)
    assert batch.charges == pytest.approx(single.charges, abs=1.0e-9)
    assert batch.forces == pytest.approx(single.forces, abs=1.0e-9)

    mixed = BatchCalculator(
        [
            Structure([1, 1], H2_POSITIONS, charge_response=response),
            Structure([1, 1], H2_POSITIONS),
        ]
    ).compute(raise_on_failure=True)
    assert mixed.energies[0] == pytest.approx(single.energy, rel=1.0e-12, abs=1.0e-12)
    assert mixed.energies[1] == pytest.approx(
        _h2().singlepoint().energy, rel=1.0e-12, abs=1.0e-12
    )


def test_response_packing_skips_dense_zeros_for_ordinary_batches():
    structures = [
        Structure([2], [[0.0, 0.0, 0.0]]),
        Structure([1, 1], H2_POSITIONS),
        Structure(
            [8, 1, 1],
            [[0.0, 0.0, 0.0], [1.4, 0.0, 1.1], [-1.4, 0.0, 1.1]],
        ),
    ]
    assert _pack_charge_responses(structures) is None


@pytest.mark.parametrize("response_index", [1, 2])
def test_ragged_response_packing_zero_fills_only_mixed_batches(response_index):
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    structures = [
        Structure([2], [[0.0, 0.0, 0.0]]),
        Structure(
            [1, 1],
            H2_POSITIONS,
            charge_response=response if response_index == 1 else None,
        ),
        Structure(
            [1, 1],
            H2_POSITIONS + np.array([0.0, 0.1, 0.0]),
            charge_response=response if response_index == 2 else None,
        ),
    ]

    offsets, shifts, matrix = _pack_charge_responses(structures)
    assert offsets == [0, 1, 5, 9]
    assert len(shifts) == 5
    assert len(matrix) == 9
    response_shift = slice(1, 3) if response_index == 1 else slice(3, 5)
    response_matrix = slice(1, 5) if response_index == 1 else slice(5, 9)
    expected_shifts = np.zeros(5)
    expected_shifts[response_shift] = response.shifts
    expected_matrix = np.zeros(9)
    expected_matrix[response_matrix] = response.matrix.ravel()
    assert shifts == pytest.approx(expected_shifts)
    assert matrix == pytest.approx(expected_matrix)


def test_charge_response_shape_validation():
    with pytest.raises(GPUxtbValueError):
        ChargeResponse(shifts=[0.0, 0.0], matrix=[0.0, 0.0, 0.0, 0.0])
    with pytest.raises(GPUxtbValueError):
        ChargeResponse(shifts=[0.0, 0.0], matrix=[[0.0], [0.0]])
    with pytest.raises(GPUxtbValueError):
        ChargeResponse(shifts=[0.0, 0.0, 0.0], matrix=[[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(GPUxtbValueError):
        ChargeResponse(shifts=[float("nan"), 0.0], matrix=[[0.0, 0.0], [0.0, 0.0]])
    calc = _h2(ChargeResponse(shifts=[0.0], matrix=[[0.0]]))
    with pytest.raises(GPUxtbValueError):
        calc.singlepoint()


def test_nonsymmetric_matrix_rejected_by_native():
    calc = _h2(ChargeResponse(shifts=[0.0, 0.0], matrix=[[0.0, 0.001], [0.0, 0.0]]))
    with pytest.raises(GPUxtbRuntimeError):
        calc.singlepoint()
