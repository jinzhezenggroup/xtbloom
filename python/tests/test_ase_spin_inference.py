"""Focused tests for ASE magnetic-moment to UHF inference."""

from __future__ import annotations

import pytest

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import InputError  # noqa: E402
from xtbloom.ase import XTBloom, _get_uhf  # noqa: E402


@pytest.mark.parametrize("total_moment", [1.0, -1.0])
def test_signed_initial_magnetic_moments_map_to_same_uhf(total_moment: float) -> None:
    """Spin orientation must not change the scalar unpaired-electron count."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    atoms.set_initial_magnetic_moments([total_moment, 0.0])
    calculator = XTBloom(method="GFN2-xTB")

    assert _get_uhf(atoms, calculator._api_parameters()) == 1


@pytest.mark.parametrize("total_moment", [0.5, -0.5, float("nan")])
def test_nonintegral_or_nonfinite_inferred_spin_is_rejected(total_moment: float) -> None:
    """Do not silently round ambiguous ASE magnetic moments into a multiplicity."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    atoms.set_initial_magnetic_moments([total_moment, 0.0])
    calculator = XTBloom(method="GFN2-xTB")

    with pytest.raises(InputError, match="integer number of unpaired electrons"):
        _get_uhf(atoms, calculator._api_parameters())


def test_explicit_multiplicity_overrides_initial_magnetic_moments() -> None:
    """Keep the explicit ASE multiplicity authoritative over inferred spin metadata."""
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    atoms.set_initial_magnetic_moments([-0.5, 0.0])
    calculator = XTBloom(method="GFN2-xTB", multiplicity=3)

    assert _get_uhf(atoms, calculator._api_parameters()) == 2
