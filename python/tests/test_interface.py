"""High-level interface tests against the committed conformance goldens.

The conformance manifest declares the reference tolerances (atol only, rtol 0)
for energy, forces, and charges, so the goldens act as the authoritative
ground truth for the Python bindings.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuxtb import Calculator, PointCharge
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

import _cases

# Cases whose goldens are pure molecular (no external point charges).
MOLECULAR_CASES = [
    "ketene",
    "nenacl",
    "h3_plus",
    "sif5_minus",
    "oh_radical",
]


def _assert_matches_golden(result, golden_values, tolerances):
    assert result.energy == pytest.approx(golden_values["energy_hartree"], abs=tolerances["energy"]["atol"])
    assert result.forces == pytest.approx(
        np.asarray(golden_values["forces_hartree_per_bohr"]).reshape(-1, 3),
        abs=tolerances["forces"]["atol"],
    )
    if "partial_charges_e" in golden_values:
        assert result.charges == pytest.approx(
            golden_values["partial_charges_e"], abs=tolerances["charges"]["atol"]
        )


@pytest.mark.parametrize("case_id", MOLECULAR_CASES)
def test_singlepoint_matches_golden(case_id):
    case = _cases.case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
    )
    result = calc.singlepoint()
    assert result.scc_converged
    assert result.scc_status == 0
    assert result.scc_iterations > 0
    _assert_matches_golden(result, _cases.golden(case), _cases.tolerances())


def test_h2o_singlepoint_smoke():
    calc = Calculator(
        "GFN2-xTB",
        np.array([8, 1, 1]),
        np.array(
            [
                [0.0, 0.0, -0.73578586109551],
                [1.44183152868459, 0.0, 0.36789293054775],
                [-1.44183152868459, 0.0, 0.36789293054775],
            ]
        ),
    )
    result = calc.singlepoint()
    assert result.energy < 0.0
    # H2O is asymmetric enough to have a nonzero force amplitude.
    assert np.linalg.norm(result.forces) > 1e-6
    assert result.charges.shape == (3,)
    # Mulliken charges are still atomic units for this method.
    assert result.charges.sum() == pytest.approx(0.0, abs=1e-12)


def test_update_positions_reuses_calculator():
    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin)
    first = calc.singlepoint().energy
    distorted = np.array(positions, copy=True)
    distorted[0, 0] += 0.5
    calc.update(positions=distorted)
    second = calc.singlepoint().energy
    assert second != pytest.approx(first, abs=1e-6)


def test_charge_and_multiplicity_are_consistent():
    case = _cases.case_by_id("h3_plus")
    numbers, positions, charge, _, _ = _cases.structure_inputs(case)
    calc = Calculator("GFN2-xTB", numbers, positions, charge=charge, multiplicity=1)
    assert calc.uhf == 0
    with pytest.raises(GPUxtbValueError):
        Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=1, multiplicity=1)


def test_open_shell_spin_polarized_differs_from_restricted():
    case = _cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, _ = _cases.structure_inputs(case)
    restricted = Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=1)
    unrestricted = Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=2)
    e_restricted = restricted.singlepoint().energy
    e_unrestricted = unrestricted.singlepoint().energy
    assert e_restricted != pytest.approx(e_unrestricted, abs=1e-5)
    # The unrestricted (spin-polarized) solution is variationally lower.
    assert e_unrestricted < e_restricted


def test_point_charge_singlepoint():
    case = _cases.case_by_id("water_one_pc_gamma999")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    point_positions, point_values, point_gammas = _cases.qmmm_points(case)
    points = PointCharge(point_positions, point_values, point_gammas)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=points,
    )
    result = calc.singlepoint()
    tol = _cases.tolerances()
    assert result.energy == pytest.approx(
        _cases.golden(case)["energy_hartree"], abs=tol["energy"]["atol"]
    )
    assert result.forces == pytest.approx(
        np.asarray(_cases.golden(case)["forces_hartree_per_bohr"]).reshape(-1, 3),
        abs=tol["forces"]["atol"],
    )


def test_scc_failure_raises():
    # A loose convergence ceiling on a charged system should fail cleanly.
    case = _cases.case_by_id("sif5_minus")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        max_scc_iterations=1,
    )
    with pytest.raises(GPUxtbRuntimeError):
        calc.singlepoint()