"""Tests for the ASE calculator (``gpuxtb.ase.GPUxtb``)."""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

from gpuxtb.ase import GPUxtb  # noqa: E402

import _cases  # noqa: E402

_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


@pytest.fixture(scope="module")
def ketene_atoms():
    case = _cases.case_by_id("ketene")
    numbers, positions_bohr, _, _, _ = _cases.structure_inputs(case)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions_bohr) * _BOHR, pbc=False)
    return atoms


def test_ase_energy_matches_golden(ketene_atoms):
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = GPUxtb(method="GFN2-xTB")
    energy = ketene_atoms.get_potential_energy()
    assert energy == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV, abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV
    )


def test_ase_forces_and_charges(ketene_atoms):
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = GPUxtb(method="GFN2-xTB")
    forces = ketene_atoms.get_forces()
    assert forces == pytest.approx(
        np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3) * _HARTREE_TO_EV / _BOHR,
        abs=tolerance["forces"]["atol"] * _HARTREE_TO_EV / _BOHR,
    )
    if "partial_charges_e" in golden:
        charges = ketene_atoms.get_charges()
        assert charges == pytest.approx(golden["partial_charges_e"], abs=tolerance["charges"]["atol"])


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


def test_ase_registered_class():
    from ase.calculators.calculator import external_calculators

    assert "gpuxtb" in external_calculators