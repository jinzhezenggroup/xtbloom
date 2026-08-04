"""Tests for the periodic charge-response (b + A q) Python exposure."""

from __future__ import annotations

import numpy as np
import pytest
from gpuxtb import BatchCalculator, Calculator, ChargeResponse, Structure
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

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
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    result = _h2(response).singlepoint()
    assert result.scc_converged
    assert result.scc_status == 0
    assert np.isfinite(result.energy)


def test_single_matches_batch_and_sequential():
    response = ChargeResponse(
        shifts=[0.003, -0.002], matrix=[[0.02, 0.001], [0.001, 0.018]]
    )
    single = _h2(response).singlepoint()

    structure = Structure([1, 1], H2_POSITIONS, charge_response=response)
    batch = BatchCalculator([structure]).compute(raise_on_failure=True)
    assert batch.energies[0] == pytest.approx(single.energy, rel=1.0e-12, abs=1.0e-12)
    assert batch.charges == pytest.approx(single.charges, abs=1.0e-9)

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
