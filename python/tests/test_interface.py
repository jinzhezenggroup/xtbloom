"""High-level interface tests against the committed conformance goldens.

The conformance manifest declares the reference tolerances (atol only, rtol 0)
for energy, forces, and charges, so the goldens act as the authoritative
ground truth for the Python bindings.
"""

from __future__ import annotations

import _cases
import gpuxtb.library as _library
import numpy as np
import pytest
from gpuxtb import Calculator, PointCharge, Result, Structure
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

# Cases whose goldens are pure molecular (no external point charges).
MOLECULAR_CASES = [
    "ketene",
    "nenacl",
    "h3_plus",
    "sif5_minus",
    "oh_radical",
]


def _assert_matches_golden(
    result: Result, golden_values: dict, tolerances: dict
) -> None:
    """Compare one result with the declared conformance tolerances."""
    assert result.energy == pytest.approx(
        golden_values["energy_hartree"], abs=tolerances["energy"]["atol"]
    )
    assert result.forces == pytest.approx(
        np.asarray(golden_values["forces_hartree_per_bohr"]).reshape(-1, 3),
        abs=tolerances["forces"]["atol"],
    )
    if "partial_charges_e" in golden_values:
        assert result.charges == pytest.approx(
            golden_values["partial_charges_e"], abs=tolerances["charges"]["atol"]
        )


@pytest.mark.parametrize("case_id", MOLECULAR_CASES)
def test_singlepoint_matches_golden(case_id: str) -> None:
    """Match each molecular single-point result to its golden data."""
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


def test_h2o_singlepoint_smoke() -> None:
    """Produce finite nontrivial water outputs through the high-level API."""
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
    assert result["gradient"] == pytest.approx(-result.forces, abs=0.0)


def test_update_positions_reuses_calculator() -> None:
    """Reuse a calculator after replacing its positions."""
    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    first = calc.singlepoint().energy
    distorted = np.array(positions, copy=True)
    distorted[0, 0] += 0.5
    calc.update(positions=distorted)
    second = calc.singlepoint().energy
    assert second != pytest.approx(first, abs=1e-6)


def test_charge_and_multiplicity_are_consistent() -> None:
    """Resolve consistent multiplicity and reject conflicting UHF input."""
    case = _cases.case_by_id("h3_plus")
    numbers, positions, charge, _, _ = _cases.structure_inputs(case)
    calc = Calculator("GFN2-xTB", numbers, positions, charge=charge, multiplicity=1)
    assert calc.uhf == 0
    with pytest.raises(GPUxtbValueError):
        Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=1, multiplicity=1)


def test_structure_update_is_transactional() -> None:
    """Leave a structure unchanged when a multi-field update is invalid."""
    structure = Structure(np.array([1, 1]), np.zeros((2, 3)))
    original = structure.positions.copy()
    with pytest.raises(GPUxtbValueError):
        structure.update(positions=np.ones((2, 3)), spin_channels=3)
    assert structure.positions == pytest.approx(original, abs=0.0)
    assert structure.spin_channels == 1


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("max_scc_iterations", 0),
        ("max_scc_iterations", 1.5),
        ("charge_tolerance", float("nan")),
        ("energy_tolerance", -1.0),
        ("electronic_temperature", float("inf")),
    ],
)
def test_invalid_compute_settings_are_rejected(setting: str, value: object) -> None:
    """Reject nonfinite, nonpositive, or nonintegral compute settings."""
    calc = Calculator("GFN2-xTB", np.array([1, 1]), np.zeros((2, 3)))
    with pytest.raises(GPUxtbValueError):
        calc.set(setting, value)


def test_open_shell_spin_polarized_differs_from_restricted() -> None:
    """Distinguish unrestricted open-shell energy from restricted energy."""
    case = _cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, _ = _cases.structure_inputs(case)
    restricted = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=1
    )
    unrestricted = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=2
    )
    e_restricted = restricted.singlepoint().energy
    e_unrestricted = unrestricted.singlepoint().energy
    assert e_restricted != pytest.approx(e_unrestricted, abs=1e-5)
    # The unrestricted (spin-polarized) solution is variationally lower.
    assert e_unrestricted < e_restricted


def test_point_charge_singlepoint() -> None:
    """Match single-point QM/MM energy and forces to golden values."""
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


def test_scc_failure_raises() -> None:
    """Raise a runtime error when SCC exhausts its iteration budget."""
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


def test_calculator_warm_start_reuses_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Calculator(warm_start=True)`` reconverges from the previous state."""
    modes: list[int] = []
    original = _library.compute_checked

    def recording(
        context: object, batch: object, options: object, result: object
    ) -> None:
        modes.append(int(options.scc_start_mode))  # type: ignore[attr-defined]
        return original(context, batch, options, result)  # type: ignore[arg-type]

    monkeypatch.setattr(_library, "compute_checked", recording)

    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    positions = np.asarray(positions, dtype=np.float64)
    warm = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        warm_start=True,
    )
    warm_energies: list[float] = []
    for step in range(3):
        displacement = np.zeros((len(numbers), 3))
        displacement[step % len(numbers), 0] = 0.02
        warm.update(positions=positions + displacement)
        warm_energies.append(warm.singlepoint().energy)
    assert modes == [
        _library.SCC_START_FRESH,
        *([_library.SCC_START_WARM] * 2),
    ]

    modes.clear()
    fresh = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    fresh_energies: list[float] = []
    for step in range(3):
        displacement = np.zeros((len(numbers), 3))
        displacement[step % len(numbers), 0] = 0.02
        fresh.update(positions=positions + displacement)
        fresh_energies.append(fresh.singlepoint().energy)
    assert modes == [_library.SCC_START_FRESH] * 3
    assert warm_energies == pytest.approx(fresh_energies, abs=1e-8)


def test_calculator_warm_start_default_stays_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``warm_start`` defaults off, preserving reproducible fresh solves."""
    modes: list[int] = []
    original = _library.compute_checked

    def recording(
        context: object, batch: object, options: object, result: object
    ) -> None:
        modes.append(int(options.scc_start_mode))  # type: ignore[attr-defined]
        return original(context, batch, options, result)  # type: ignore[arg-type]

    monkeypatch.setattr(_library, "compute_checked", recording)

    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
    )
    for step in range(2):
        displacement = np.zeros((len(numbers), 3))
        displacement[step % len(numbers), 0] = 0.02
        calc.update(positions=np.asarray(positions) + displacement)
        calc.singlepoint()
    assert modes == [_library.SCC_START_FRESH] * 2
