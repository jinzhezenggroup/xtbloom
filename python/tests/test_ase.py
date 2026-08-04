"""Tests for the ASE calculator (``gpuxtb.ase.GPUxtb``)."""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
import _cases  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import InputError  # noqa: E402
from gpuxtb.ase import GPUxtb  # noqa: E402
from gpuxtb.exceptions import GPUxtbValueError  # noqa: E402

_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


@pytest.fixture(scope="module")
def ketene_atoms():
    case = _cases.case_by_id("ketene")
    numbers, positions_bohr, _, _, _ = _cases.structure_inputs(case)
    atoms = Atoms(
        numbers=numbers, positions=np.asarray(positions_bohr) * _BOHR, pbc=False
    )
    return atoms


def test_ase_energy_matches_golden(ketene_atoms):
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = GPUxtb(method="GFN2-xTB")
    energy = ketene_atoms.get_potential_energy()
    assert energy == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )
    assert ketene_atoms.get_potential_energy(force_consistent=True) == pytest.approx(
        energy, abs=0.0
    )


def test_ase_forces_and_charges(ketene_atoms):
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = GPUxtb(method="GFN2-xTB")
    forces = ketene_atoms.get_forces()
    assert forces == pytest.approx(
        np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3)
        * _HARTREE_TO_EV
        / _BOHR,
        abs=tolerance["forces"]["atol"] * _HARTREE_TO_EV / _BOHR,
    )
    if "partial_charges_e" in golden:
        charges = ketene_atoms.get_charges()
        assert charges == pytest.approx(
            golden["partial_charges_e"], abs=tolerance["charges"]["atol"]
        )


def test_ase_charge_from_atoms(ketene_atoms):
    case = _cases.case_by_id("h3_plus")
    numbers, positions_bohr, _, _, _ = _cases.structure_inputs(case)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions_bohr) * _BOHR)
    atoms.set_initial_charges([0, 0, 1])
    atoms.calc = GPUxtb(method="GFN2-xTB")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    assert atoms.get_potential_energy() == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_ase_set_resets_results(ketene_atoms):
    ketene_atoms.calc = GPUxtb(method="GFN2-xTB")
    first = ketene_atoms.get_potential_energy()
    ketene_atoms.calc.set(electronic_temperature=5000.0)
    second = ketene_atoms.get_potential_energy()
    assert second != pytest.approx(first, abs=1e-2)


def test_ase_set_rejects_invalid_settings_transactionally(ketene_atoms):
    calculator = GPUxtb(method="GFN2-xTB")
    ketene_atoms.calc = calculator
    ketene_atoms.get_potential_energy()

    with pytest.raises(GPUxtbValueError):
        calculator.set(max_scc_iterations=0)
    assert calculator.parameters.max_scc_iterations == 250


def test_ase_rejects_fractional_multiplicity():
    with pytest.raises(GPUxtbValueError):
        GPUxtb(method="GFN2-xTB", multiplicity=1.5)


def test_ase_rebuilds_when_atomic_numbers_change():
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.calc = GPUxtb(method="GFN2-xTB", backend="cpu")
    atoms.get_potential_energy()
    atoms.numbers[:] = [2, 2]
    reused = atoms.get_potential_energy()

    reference = Atoms(numbers=[2, 2], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    reference.calc = GPUxtb(method="GFN2-xTB", backend="cpu")
    assert reused == pytest.approx(reference.get_potential_energy(), abs=1e-10)


def test_ase_rejects_periodic_systems():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.calc = GPUxtb(method="GFN2-xTB")
    with pytest.raises(InputError):
        atoms.get_potential_energy()


def test_ase_registered_class():
    from ase.calculators.calculator import external_calculators

    assert "gpuxtb" in external_calculators
