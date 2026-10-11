"""High-level interface tests against the committed conformance goldens.

The conformance manifest declares the reference tolerances (atol only, rtol 0)
for energy, forces, and charges, so the goldens act as the authoritative
ground truth for the Python bindings.
"""

from __future__ import annotations

import _cases
import numpy as np
import pytest
import xtbloom.library as _library
from xtbloom import BatchCalculator, Calculator, PointCharge, Result, Structure
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError

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


def test_periodic_calculator_publishes_strain_and_accepts_cell_updates() -> None:
    """Exercise the high-level native XYZ cell and ABI-v3 strain outlet."""
    numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    cell = np.diag([8.0, 8.0, 8.0])
    calculator = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        cell=cell,
        pbc=True,
        backend="cpu",
    )

    first = calculator.singlepoint(compute_strain=True)
    assert first.strain_derivatives is not None
    assert first.strain_derivatives.shape == (9,)
    assert np.isfinite(first.strain_derivatives).all()
    assert calculator.cell is not None

    # Cell changes are explicit topology changes.  The Python object updates
    # transactionally and the native context rebuilds its fixed-cell plan.
    calculator.update(cell=np.diag([8.5, 8.0, 8.0]), pbc=True)
    second = calculator.singlepoint(compute_strain=True)
    assert second.strain_derivatives is not None
    assert np.isfinite(second.strain_derivatives).all()
    assert second.energy != pytest.approx(first.energy, abs=1.0e-10)


def test_periodic_calculator_rejects_unreleased_model_and_attachment() -> None:
    """Keep GFN1 and external field combinations fail-closed for native PBC."""
    numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    cell = np.diag([8.0, 8.0, 8.0])
    with pytest.raises(XTBloomRuntimeError, match="not implemented"):
        Calculator(
            "GFN1-xTB",
            numbers,
            positions,
            cell=cell,
            pbc=True,
            backend="cpu",
        ).singlepoint()

    with pytest.raises(XTBloomRuntimeError, match="periodic"):
        Calculator(
            "GFN2-xTB",
            numbers,
            positions,
            cell=cell,
            pbc=True,
            efield=[0.1, 0.0, 0.0],
            backend="cpu",
        ).singlepoint()


@pytest.mark.parametrize("case_id", ["gfn1_ketene", "gfn1_oh_radical"])
def test_gfn1_singlepoint_matches_independent_golden(case_id: str) -> None:
    """Match closed-shell and shared-orbital open-shell GFN1 to oracles."""
    case = _cases.gfn1_case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.gfn1_structure_inputs(case)
    result = Calculator(
        "GFN1-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cpu",
    ).singlepoint()
    assert result.scc_converged
    _assert_matches_golden(result, _cases.gfn1_golden(case), _cases.gfn1_tolerances())


def test_gfn1_two_channel_open_shell_public_smoke() -> None:
    """Exercise public two-channel GFN1 without claiming a new golden."""
    result = Calculator(
        "GFN1-xTB",
        np.array([8, 1]),
        np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
        uhf=1,
        spin_channels=2,
        backend="cpu",
    ).singlepoint()
    assert result.scc_converged
    assert np.isfinite(result.energy)
    assert np.isfinite(result.forces).all()
    assert np.isfinite(result.charges).all()


def test_gfn1_auto_and_explicit_cuda_policy() -> None:
    """Use shared AUTO routing and qualify explicit CUDA when available."""
    case = _cases.gfn1_case_by_id("gfn1_ketene")
    numbers, positions, charge, uhf, spin = _cases.gfn1_structure_inputs(case)
    calculator = Calculator(
        "GFN1",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
    )
    auto_result = calculator.singlepoint()
    assert calculator.backend in (_library.BACKEND_CPU, _library.BACKEND_CUDA)
    assert auto_result.scc_converged

    cpu_result = Calculator(
        "GFN1-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cpu",
    ).singlepoint()

    explicit_cuda = Calculator(
        "GFN1-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cuda",
    )
    try:
        cuda_result = explicit_cuda.singlepoint()
    except XTBloomRuntimeError as caught:
        assert caught.status == _library.STATUS_BACKEND_UNAVAILABLE
    else:
        assert cuda_result.scc_converged
        assert cuda_result.energy == pytest.approx(cpu_result.energy, abs=3.0e-8)
        assert cuda_result.forces == pytest.approx(cpu_result.forces, abs=3.0e-7)
        assert cuda_result.charges == pytest.approx(cpu_result.charges, abs=1.0e-7)


def test_gfn1_point_charges_match_independent_golden() -> None:
    """Exercise the GFN1-specific harmonic-hardness point-charge path."""
    case = _cases.gfn1_case_by_id("gfn1_water_dimer_6pc_hardness")
    numbers, positions, charge, uhf, spin = _cases.gfn1_structure_inputs(case)
    point_data = _cases.gfn1_qmmm_points(case)
    assert point_data is not None
    result = Calculator(
        "GFN1-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        point_charges=PointCharge(*point_data),
        backend="cpu",
    ).singlepoint()
    golden = _cases.gfn1_golden(case)
    tolerances = _cases.gfn1_tolerances()
    _assert_matches_golden(result, golden, tolerances)
    assert result.point_charge_forces == pytest.approx(
        np.asarray(golden["point_charge_forces_hartree_per_bohr"]).reshape(-1, 3),
        abs=tolerances["point_charge_forces"]["atol"],
    )


def test_gfn1_ragged_batch_matches_independent_goldens() -> None:
    """Keep GFN1 model identity across differently sized batched systems."""
    case_ids = ["gfn1_oh_radical", "gfn1_ketene"]
    structures = []
    for case_id in case_ids:
        case = _cases.gfn1_case_by_id(case_id)
        numbers, positions, charge, uhf, spin = _cases.gfn1_structure_inputs(case)
        structures.append(
            Structure(
                numbers,
                positions,
                charge=charge,
                uhf=uhf,
                spin_channels=spin,
            )
        )
    result = BatchCalculator(structures, method="GFN1", backend="cpu").compute(
        raise_on_failure=True
    )
    for index, case_id in enumerate(case_ids):
        _assert_matches_golden(
            result[index],
            _cases.gfn1_golden(_cases.gfn1_case_by_id(case_id)),
            _cases.gfn1_tolerances(),
        )


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
    with pytest.raises(XTBloomValueError):
        Calculator("GFN2-xTB", numbers, positions, charge=charge, uhf=1, multiplicity=1)


def test_structure_update_is_transactional() -> None:
    """Leave a structure unchanged when a multi-field update is invalid."""
    structure = Structure(np.array([1, 1]), np.zeros((2, 3)))
    original = structure.positions.copy()
    with pytest.raises(XTBloomValueError):
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
        ("scc_mixer", "linear"),
        ("scc_mixer", 2),
        ("scc_mixer_history", 0),
        ("scc_mixer_history", 65),
        ("scc_mixer_damping", 0.0),
        ("scc_mixer_damping", float("nan")),
        ("determinism", "portable"),
        ("determinism", 2),
    ],
)
def test_invalid_compute_settings_are_rejected(setting: str, value: object) -> None:
    """Reject nonfinite, nonpositive, or nonintegral compute settings."""
    calc = Calculator("GFN2-xTB", np.array([1, 1]), np.zeros((2, 3)))
    with pytest.raises(XTBloomValueError):
        calc.set(setting, value)


def test_compute_policy_aliases_and_tags_are_normalized() -> None:
    """Accept frozen names or int32 tags and retain their exact ABI values."""
    calc = Calculator(
        "GFN2-xTB",
        np.array([1, 1]),
        np.zeros((2, 3)),
        scc_mixer="modified_broyden",
        scc_mixer_history=16,
        scc_mixer_damping=0.25,
        determinism="reproducible",
    )
    assert calc._settings.scc_mixer == _library.SCC_MIXER_MODIFIED_BROYDEN
    assert calc._settings.scc_mixer_history == 16
    assert calc._settings.scc_mixer_damping == 0.25
    assert calc._settings.determinism == _library.DETERMINISM_REPRODUCIBLE

    calc.set("scc_mixer", _library.SCC_MIXER_MODIFIED_BROYDEN)
    calc.set("determinism", _library.DETERMINISM_DEFAULT)
    assert calc._settings.determinism == _library.DETERMINISM_DEFAULT


def test_high_level_policy_updates_fail_closed_on_legacy_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calculator and BatchCalculator reject policies an old core would ignore."""
    monkeypatch.setattr(
        _library, "compute_options_v3_available", lambda _lib=None: False
    )
    calc = Calculator("GFN2-xTB", np.array([1, 1]), np.zeros((2, 3)))
    calc.set("scc_mixer_history", 8)
    with pytest.raises(XTBloomRuntimeError, match=r"does not support.*ABI v3"):
        calc.set("scc_mixer_history", 16)


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
    with pytest.raises(XTBloomRuntimeError):
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


def _water_calculator(**kwargs: object) -> Calculator:
    """Water in the tblite 0.7.0 h2o.xyz orientation (angstrom -> bohr)."""
    angstrom_per_bohr = 1.8897261246257702
    xyz = [
        [0.0, 0.0, -0.2358784530],
        [0.0, 1.4270063049, 1.0081495306],
        [0.0, -1.4270063049, 1.0081495306],
    ]
    return Calculator(
        "GFN2-xTB",
        np.array([8, 1, 1]),
        np.array([[c * angstrom_per_bohr for c in atom] for atom in xyz]),
        **kwargs,
    )


def test_efield_matches_tblite_energy_and_force_derivative() -> None:
    """Match the tblite energy while forces follow public energy derivatives.

    The oracle ran with ``--efield 0.0514221,0.1028442,-0.0771332`` (V/A),
    which is (0.001, 0.002, -0.0015) atomic units. Its analytic gradient uses
    the nonvariational ``+E``-per-atom term, so it remains diagnostic only.
    """
    calc = _water_calculator(
        efield=[0.001, 0.002, -0.0015],
        max_scc_iterations=500,
        charge_tolerance=1.0e-10,
        energy_tolerance=1.0e-12,
    )
    result = calc.singlepoint()
    assert result.energy == pytest.approx(-4.7652477392228, abs=1e-7)

    reference_positions = calc.positions.copy()
    analytic_forces = result.forces.reshape(-1).copy()
    for step in (2.0e-3, 1.0e-3, 5.0e-4):
        for coordinate in range(reference_positions.size):
            displaced = reference_positions.copy().reshape(-1)
            displaced[coordinate] += step
            calc.update(positions=displaced.reshape(reference_positions.shape))
            energy_plus = calc.singlepoint().energy

            displaced[coordinate] -= 2.0 * step
            calc.update(positions=displaced.reshape(reference_positions.shape))
            energy_minus = calc.singlepoint().energy

            numerical_force = -(energy_plus - energy_minus) / (2.0 * step)
            assert numerical_force == pytest.approx(
                analytic_forces[coordinate], abs=1.0e-5
            )

    dipole = result.get("dipole_moments")
    assert dipole is not None
    assert np.isfinite(dipole).all()
    assert dipole.shape == (3,)


def test_efield_changes_energy_forces_charges() -> None:
    """The field self-consistently polarizes water versus the field-free run."""
    plain = _water_calculator().singlepoint()
    field = _water_calculator(efield=[0.001, 0.002, -0.0015]).singlepoint()
    assert field.energy != pytest.approx(plain.energy, abs=1e-8)
    assert np.linalg.norm(field.forces - plain.forces) > 1e-4
    # Mulliken charges conserve the molecular charge under the field.
    assert field.charges.sum() == pytest.approx(0.0, abs=1e-12)
    assert field.charges[0] != pytest.approx(plain.charges[0], abs=1e-6)


def test_efield_dipole_publication_flag() -> None:
    """Dipole moments are published only when a field is present (or requested)."""
    # A field enables the dipole publication flag through the low-level compute.

    calc = _water_calculator(efield=[0.001, 0.0, 0.0])
    result = calc.singlepoint()
    assert result.get("dipole_moments") is not None


def test_efield_invalid_input_is_rejected() -> None:
    """A malformed electric field is rejected eagerly by the high-level API."""
    with pytest.raises(XTBloomValueError):
        _water_calculator(efield=[0.001, 0.002])
    with pytest.raises(XTBloomValueError):
        _water_calculator(efield=[0.001, 0.002, float("nan")])


def test_efield_zero_equivalent_to_none() -> None:
    """A zero field normalizes to no attachment and preserves results."""
    zero = _water_calculator(efield=[0.0, 0.0, 0.0]).singlepoint()
    plain = _water_calculator().singlepoint()
    assert zero.energy == pytest.approx(plain.energy, abs=0.0)
    assert zero.get("dipole_moments") is None
