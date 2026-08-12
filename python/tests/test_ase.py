"""Tests for the ASE calculator (``xtbloom.ase.XTBloom``)."""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")
import _cases  # noqa: E402
import xtbloom.library as _library  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import InputError  # noqa: E402
from xtbloom.ase import XTBloom  # noqa: E402
from xtbloom.exceptions import XTBloomValueError  # noqa: E402

_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


def _record_scc_start_modes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap ``xtbloom_compute`` to record the per-call ``scc_start_mode``."""
    modes: list[int] = []
    original = _library.compute_checked

    def recording(
        context: object, batch: object, options: object, result: object
    ) -> None:
        modes.append(int(options.scc_start_mode))  # type: ignore[attr-defined]
        return original(context, batch, options, result)  # type: ignore[arg-type]

    monkeypatch.setattr(_library, "compute_checked", recording)
    return modes


_MD_DISPLACEMENTS = [
    [0.02, 0.0, 0.0],
    [-0.01, 0.015, 0.0],
    [0.0, -0.01, 0.02],
    [0.01, 0.0, -0.015],
]


@pytest.fixture(scope="module")
def ketene_atoms() -> Atoms:
    """Build the ketene conformance structure in ASE units."""
    case = _cases.case_by_id("ketene")
    numbers, positions_bohr, _, _, _ = _cases.structure_inputs(case)
    atoms = Atoms(
        numbers=numbers, positions=np.asarray(positions_bohr) * _BOHR, pbc=False
    )
    return atoms


def test_ase_energy_matches_golden(ketene_atoms: Atoms) -> None:
    """Match the ASE energy and free-energy alias to the golden value."""
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = XTBloom(method="GFN2-xTB")
    energy = ketene_atoms.get_potential_energy()
    assert energy == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )
    assert ketene_atoms.get_potential_energy(force_consistent=True) == pytest.approx(
        energy, abs=0.0
    )


def test_ase_forces_and_charges(ketene_atoms: Atoms) -> None:
    """Match ASE forces and charges to the conformance golden values."""
    case = _cases.case_by_id("ketene")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    ketene_atoms.calc = XTBloom(method="GFN2-xTB")
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


def test_ase_charge_from_atoms(ketene_atoms: Atoms) -> None:
    """Read the molecular charge from ASE initial atomic charges."""
    case = _cases.case_by_id("h3_plus")
    numbers, positions_bohr, _, _, _ = _cases.structure_inputs(case)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions_bohr) * _BOHR)
    atoms.set_initial_charges([0, 0, 1])
    atoms.calc = XTBloom(method="GFN2-xTB")
    golden = _cases.golden(case)
    tolerance = _cases.tolerances()
    assert atoms.get_potential_energy() == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_ase_set_resets_results(ketene_atoms: Atoms) -> None:
    """Invalidate cached ASE results after changing a compute setting."""
    ketene_atoms.calc = XTBloom(method="GFN2-xTB")
    first = ketene_atoms.get_potential_energy()
    ketene_atoms.calc.set(electronic_temperature=5000.0)
    second = ketene_atoms.get_potential_energy()
    assert second != pytest.approx(first, abs=1e-2)


def test_ase_set_rejects_invalid_settings_transactionally(
    ketene_atoms: Atoms,
) -> None:
    """Keep ASE parameters unchanged when validation rejects an update."""
    calculator = XTBloom(method="GFN2-xTB")
    ketene_atoms.calc = calculator
    ketene_atoms.get_potential_energy()

    with pytest.raises(XTBloomValueError):
        calculator.set(max_scc_iterations=0)
    assert calculator.parameters.max_scc_iterations == 250


def test_ase_scc_policy_parameters_are_validated() -> None:
    """Expose the frozen ABI-v3 controls through ASE's parameter mapping."""
    calculator = XTBloom(
        method="GFN2-xTB",
        scc_mixer="modified_broyden",
        scc_mixer_history=16,
        scc_mixer_damping=0.25,
        determinism="reproducible",
    )
    assert calculator.parameters.scc_mixer_history == 16
    with pytest.raises(XTBloomValueError, match="scc_mixer_history"):
        calculator.set(scc_mixer_history=65)


def test_ase_updates_cached_scc_policy_in_place() -> None:
    """Push all numerical V3 policy changes into an existing API calculator."""

    class FakeCalculator:
        def __init__(self) -> None:
            self.updates: list[tuple[str, object]] = []

        def set(self, name: str, value: object) -> None:
            self.updates.append((name, value))

        def close(self) -> None:
            pass

    calculator = XTBloom(method="GFN2-xTB")
    fake = FakeCalculator()
    calculator._xtb = fake  # type: ignore[assignment]
    calculator.set(
        scc_mixer=_library.SCC_MIXER_MODIFIED_BROYDEN,
        scc_mixer_history=16,
        scc_mixer_damping=0.25,
        determinism="reproducible",
    )

    assert fake.updates == [
        ("scc_mixer", _library.SCC_MIXER_MODIFIED_BROYDEN),
        ("scc_mixer_history", 16),
        ("scc_mixer_damping", 0.25),
        ("determinism", "reproducible"),
    ]
    calculator.close()


def test_ase_rejects_fractional_multiplicity() -> None:
    """Reject nonintegral spin multiplicities through the ASE interface."""
    with pytest.raises(XTBloomValueError):
        XTBloom(method="GFN2-xTB", multiplicity=1.5)


def test_ase_rebuilds_when_atomic_numbers_change() -> None:
    """Rebuild fixed-topology native state after ASE changes atom species."""
    atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    atoms.calc = XTBloom(method="GFN2-xTB", backend="cpu")
    atoms.get_potential_energy()
    atoms.numbers[:] = [2, 2]
    reused = atoms.get_potential_energy()

    reference = Atoms(numbers=[2, 2], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    reference.calc = XTBloom(method="GFN2-xTB", backend="cpu")
    assert reused == pytest.approx(reference.get_potential_energy(), abs=1e-10)


def test_ase_rejects_periodic_systems() -> None:
    """Reject periodic ASE structures unsupported by the molecular ABI."""
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.calc = XTBloom(method="GFN2-xTB")
    with pytest.raises(InputError):
        atoms.get_potential_energy()


def test_ase_registered_class() -> None:
    """Register the calculator for ASE's name-based construction path."""
    from ase.calculators.calculator import external_calculators

    assert "xtbloom" in external_calculators


def test_ase_warm_start_reuses_state_across_md_steps(
    ketene_atoms: Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default warm start issues FRESH then WARM on a steady MD-like trajectory."""
    modes = _record_scc_start_modes(monkeypatch)
    atoms = ketene_atoms.copy()
    atoms.calc = XTBloom(method="GFN2-xTB")
    atoms.get_potential_energy()
    assert modes == [_library.SCC_START_FRESH]
    for step in range(len(_MD_DISPLACEMENTS)):
        atoms.positions[step % len(atoms)] += _MD_DISPLACEMENTS[step]
        atoms.get_potential_energy()
    assert modes == [
        _library.SCC_START_FRESH,
        *([_library.SCC_START_WARM] * len(_MD_DISPLACEMENTS)),
    ]


def test_ase_warm_start_matches_fresh_trajectory(ketene_atoms: Atoms) -> None:
    """Warm-started energies equal independent FRESH solves along the same path."""
    warm_atoms = ketene_atoms.copy()
    warm_atoms.calc = XTBloom(method="GFN2-xTB", warm_start=True)
    fresh_atoms = ketene_atoms.copy()
    fresh_atoms.calc = XTBloom(method="GFN2-xTB", warm_start=False)
    warm_energies: list[float] = []
    fresh_energies: list[float] = []
    for step, displacement in enumerate(_MD_DISPLACEMENTS):
        warm_atoms.positions[step % len(warm_atoms)] += displacement
        fresh_atoms.positions[step % len(fresh_atoms)] += displacement
        warm_energies.append(warm_atoms.get_potential_energy())
        fresh_energies.append(fresh_atoms.get_potential_energy())
    assert warm_energies == pytest.approx(fresh_energies, abs=1e-6)


def test_ase_warm_start_falls_back_on_identity_change(
    ketene_atoms: Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compute-policy change invalidates the checkpoint and falls back."""
    modes = _record_scc_start_modes(monkeypatch)
    atoms = ketene_atoms.copy()
    atoms.calc = XTBloom(method="GFN2-xTB")
    atoms.get_potential_energy()
    atoms.calc.set(electronic_temperature=5000.0)
    energy = atoms.get_potential_energy()
    assert modes == [
        _library.SCC_START_FRESH,
        _library.SCC_START_WARM,
        _library.SCC_START_FRESH,
    ]
    reference = ketene_atoms.copy()
    reference.calc = XTBloom(
        method="GFN2-xTB", electronic_temperature=5000.0, warm_start=False
    )
    assert energy == pytest.approx(reference.get_potential_energy(), abs=1e-9)


def test_ase_warm_start_disabled_stays_fresh(
    ketene_atoms: Atoms, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``warm_start=False`` keeps independent reproducible FRESH solves."""
    modes = _record_scc_start_modes(monkeypatch)
    atoms = ketene_atoms.copy()
    atoms.calc = XTBloom(method="GFN2-xTB", warm_start=False)
    for step in range(len(_MD_DISPLACEMENTS)):
        atoms.positions[step % len(atoms)] += _MD_DISPLACEMENTS[step]
        atoms.get_potential_energy()
    assert modes == [_library.SCC_START_FRESH] * len(_MD_DISPLACEMENTS)
