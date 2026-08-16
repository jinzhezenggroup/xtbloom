"""Tests for the dpdata driver plugin (``xtbloom.dpdata``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

dpdata = pytest.importorskip("dpdata")

import _cases  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from xtbloom.interface import Structure

_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


def _case_data_dict(
    case_id: str, nframes: int = 1, distort: bool = False
) -> dict[str, object]:
    """Build a dpdata System data dict (coords in Angstrom) from a conformance case."""
    case = _cases.case_by_id(case_id)
    numbers, positions, _, _, _ = _cases.structure_inputs(case)
    coords = np.stack([np.asarray(positions) for _ in range(nframes)])
    if distort:
        coords[1] = coords[1].copy()
        coords[1, 0, 0] += 0.5
    coords = coords * _BOHR  # bohr -> Angstrom (dpdata convention)
    atom_names = _cases.numbers_to_symbols(sorted({int(z) for z in numbers}))
    type_map = {
        number: index for index, number in enumerate(sorted({int(z) for z in numbers}))
    }
    atom_types = np.array([type_map[int(z)] for z in numbers], dtype=np.int64)
    atom_numbs = [int(sum(int(z) == number for z in numbers)) for number in type_map]
    return {
        "atom_names": atom_names,
        "atom_numbs": atom_numbs,
        "atom_types": atom_types,
        "orig": np.zeros(3),
        "cells": np.eye(3)[None, ...] * np.ones((nframes, 1, 1)),
        "coords": coords,
        "nopbc": True,
    }


def _gfn1_case_data_dict(case_id: str) -> dict[str, object]:
    """Build a dpdata data dict from the independent GFN1 oracle manifest."""
    case = _cases.gfn1_case_by_id(case_id)
    numbers, positions, _, _, _ = _cases.gfn1_structure_inputs(case)
    atom_names = _cases.numbers_to_symbols(sorted({int(z) for z in numbers}))
    type_map = {
        number: index for index, number in enumerate(sorted({int(z) for z in numbers}))
    }
    return {
        "atom_names": atom_names,
        "atom_numbs": [
            int(sum(int(z) == number for z in numbers)) for number in type_map
        ],
        "atom_types": np.array([type_map[int(z)] for z in numbers], dtype=np.int64),
        "orig": np.zeros(3),
        "cells": np.eye(3)[None, ...],
        "coords": np.asarray(positions)[None, ...] * _BOHR,
        "nopbc": True,
    }


def _ensure_driver_registered() -> type:
    """Load and return the registered xTBloom dpdata driver class."""
    # The entry point is registered after a normal wheel install; for a source
    # checkout we register the module explicitly, mirroring dpdata's loader.
    import xtbloom.dpdata as _  # noqa: F401
    from dpdata.driver import Driver

    assert "xtbloom" in Driver.get_drivers()
    return Driver.get_driver("xtbloom")


def test_driver_registered() -> None:
    """Expose the plugin through dpdata's driver registry."""
    driver_class = _ensure_driver_registered()
    assert driver_class.__module__ == "xtbloom.dpdata"


def test_driver_forwards_scc_policy_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dpdata's generic keyword bridge aligned with BatchCalculator."""
    from xtbloom.dpdata import XTBloomDriver

    captured: dict[str, object] = {}

    class FakeBatchCalculator:
        """Record constructor settings without requiring a numerical runtime."""

        def __init__(
            self, structures: list[Structure], method: str, **kwargs: object
        ) -> None:
            captured["structures"] = structures
            captured["method"] = method
            captured.update(kwargs)

        def compute(self, *, raise_on_failure: bool) -> SimpleNamespace:
            assert raise_on_failure
            structures = captured["structures"]
            assert isinstance(structures, list)
            return SimpleNamespace(
                energies=np.zeros(len(structures), dtype=np.float64),
                forces=np.concatenate(
                    [np.zeros_like(structure.positions) for structure in structures]
                ),
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("xtbloom.dpdata.BatchCalculator", FakeBatchCalculator)
    XTBloomDriver(
        scc_mixer="modified_broyden",
        scc_mixer_history=16,
        scc_mixer_damping=0.25,
        determinism="reproducible",
    ).label(_case_data_dict("ketene"))
    assert captured["scc_mixer"] == "modified_broyden"
    assert captured["scc_mixer_history"] == 16
    assert captured["scc_mixer_damping"] == 0.25
    assert captured["determinism"] == "reproducible"


def test_driver_forwards_gfn1_method_and_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep model identity explicit across the dpdata generic keyword bridge."""
    from xtbloom.dpdata import XTBloomDriver

    captured: dict[str, object] = {}

    class FakeBatchCalculator:
        def __init__(
            self, structures: list[Structure], method: str, **kwargs: object
        ) -> None:
            captured["method"] = method
            captured["backend"] = kwargs.get("backend")
            captured["structures"] = structures

        def compute(self, *, raise_on_failure: bool) -> SimpleNamespace:
            assert raise_on_failure
            structures = captured["structures"]
            assert isinstance(structures, list)
            return SimpleNamespace(
                energies=np.zeros(len(structures), dtype=np.float64),
                forces=np.concatenate(
                    [np.zeros_like(structure.positions) for structure in structures]
                ),
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("xtbloom.dpdata.BatchCalculator", FakeBatchCalculator)
    XTBloomDriver(method="GFN1", backend="cpu").label(
        _gfn1_case_data_dict("gfn1_ketene")
    )
    assert captured["method"] == "GFN1"
    assert captured["backend"] == "cpu"
    assert isinstance(captured["structures"], list)


def test_label_energies_match_golden() -> None:
    """Match dpdata labels to golden energies and forces in dpdata units."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("ketene"))
    labeled = system.predict(driver="xtbloom")
    golden = _cases.golden(_cases.case_by_id("ketene"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )
    assert labeled.data["forces"][0] == pytest.approx(
        np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3)
        * _HARTREE_TO_EV
        / _BOHR,
        abs=tolerance["forces"]["atol"] * _HARTREE_TO_EV / _BOHR,
    )


def test_label_gfn1_cpu_matches_independent_golden() -> None:
    """Label dpdata frames with GFN1 without substituting the default GFN2 model."""
    _ensure_driver_registered()
    case = _cases.gfn1_case_by_id("gfn1_ketene")
    system = dpdata.System(data=_gfn1_case_data_dict("gfn1_ketene"))
    labeled = system.predict(driver="xtbloom", method="GFN1-xTB")
    golden = _cases.gfn1_golden(case)
    tolerance = _cases.gfn1_tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )
    assert labeled.data["forces"][0] == pytest.approx(
        np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3)
        * _HARTREE_TO_EV
        / _BOHR,
        abs=tolerance["forces"]["atol"] * _HARTREE_TO_EV / _BOHR,
    )


def test_label_distinct_frames() -> None:
    """Label every distinct frame in one dpdata system."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("ketene", nframes=2, distort=True))
    labeled = system.predict(driver="xtbloom")
    assert labeled.get_nframes() == 2
    assert labeled.data["energies"].shape == (2,)
    assert labeled.data["forces"].shape == (2, 5, 3)
    assert labeled.data["energies"][0] != pytest.approx(labeled.data["energies"][1])


def test_driver_charge() -> None:
    """Forward a fixed charged-system state through the dpdata driver."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("h3_plus"))
    labeled = system.predict(driver="xtbloom", charge=1)
    golden = _cases.golden(_cases.case_by_id("h3_plus"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_driver_multiplicity() -> None:
    """Forward spin multiplicity through the dpdata driver."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("oh_radical"))
    labeled = system.predict(driver="xtbloom", multiplicity=2, spin_channels=1)
    golden = _cases.golden(_cases.case_by_id("oh_radical"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_driver_reads_per_frame_multiplicity_without_forcing_uhf_zero() -> None:
    """Honor per-frame multiplicity when no fixed UHF value is supplied."""
    _ensure_driver_registered()
    data = _case_data_dict("oh_radical")
    data["multiplicity"] = np.array([2], dtype=np.int32)
    system = dpdata.System(data=data)
    labeled = system.predict(driver="xtbloom", spin_channels=1)
    golden = _cases.golden(_cases.case_by_id("oh_radical"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("charge", np.array([0.0, 0.0])),
        ("charge", np.zeros((3, 1))),
        ("uhf", np.array([0, 0, 0, 0], dtype=np.int32)),
        ("multiplicity", np.array([[1, 1, 1]], dtype=np.int32)),
    ],
)
def test_driver_rejects_malformed_per_frame_metadata(
    key: str, value: np.ndarray
) -> None:
    """Reject present charge/spin metadata whose shape does not match the frames."""
    from xtbloom.dpdata import XTBloomDriver
    from xtbloom.exceptions import XTBloomValueError

    data = _case_data_dict("ketene", nframes=3)
    data[key] = value
    with pytest.raises(XTBloomValueError, match=key):
        XTBloomDriver().label(data)


def test_driver_fixed_metadata_overrides_malformed_per_frame_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit constructor values authoritative over ignored data metadata."""
    from xtbloom.dpdata import XTBloomDriver

    captured: dict[str, object] = {}

    class FakeBatchCalculator:
        def __init__(
            self, structures: list[Structure], _method: str, **_kwargs: object
        ) -> None:
            captured["structures"] = structures

        def compute(self, *, raise_on_failure: bool) -> SimpleNamespace:
            assert raise_on_failure
            structures = captured["structures"]
            assert isinstance(structures, list)
            return SimpleNamespace(
                energies=np.zeros(len(structures), dtype=np.float64),
                forces=np.concatenate(
                    [np.zeros_like(structure.positions) for structure in structures]
                ),
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr("xtbloom.dpdata.BatchCalculator", FakeBatchCalculator)
    data = _case_data_dict("ketene", nframes=3)
    data["charge"] = np.array([99.0, 99.0])
    XTBloomDriver(charge=1.0).label(data)
    structures = captured["structures"]
    assert isinstance(structures, list)
    assert [structure.charge for structure in structures] == [1.0, 1.0, 1.0]


def test_driver_raises_instead_of_publishing_failed_frame_nans() -> None:
    """Raise instead of publishing NaNs for a failed dpdata frame."""
    _ensure_driver_registered()
    from xtbloom.exceptions import XTBloomRuntimeError

    system = dpdata.System(data=_case_data_dict("sif5_minus"))
    with pytest.raises(XTBloomRuntimeError, match="failed systems"):
        system.predict(driver="xtbloom", charge=-1, backend="cpu", max_scc_iterations=1)


def test_driver_rejects_periodic() -> None:
    """Reject periodic dpdata systems unsupported by the molecular ABI."""
    _ensure_driver_registered()
    from xtbloom.exceptions import XTBloomNotSupportedError

    data = _case_data_dict("ketene")
    data["nopbc"] = False
    system = dpdata.System(data=data)
    with pytest.raises(XTBloomNotSupportedError):
        system.predict(driver="xtbloom")


@pytest.mark.parametrize("coords", [None, np.array(1.0)])
def test_driver_rejects_malformed_scalar_coordinates(coords: object) -> None:
    """Report the public coordinate-shape error for scalar-like inputs."""
    from xtbloom.dpdata import XTBloomDriver
    from xtbloom.exceptions import XTBloomValueError

    data = _case_data_dict("ketene")
    data["coords"] = coords
    with pytest.raises(
        XTBloomValueError, match=r"coords must have shape \(nframes, natoms, 3\)"
    ):
        XTBloomDriver().label(data)


@pytest.mark.parametrize(
    ("atom_types", "message"),
    [
        (np.array([0.5, 1.0, 1.0, 0.0, 2.0]), "exact integer"),
        (np.array([True, False, False, True, False]), "exact integer"),
        (np.array([-1, 1, 1, 0, 2], dtype=np.int64), "outside atom_names"),
        (np.array([3, 1, 1, 0, 2], dtype=np.int64), "outside atom_names"),
        (np.array([[0], [1], [1], [0], [2]], dtype=np.int64), "one-dimensional"),
        (np.array([], dtype=np.int64), "nonempty one-dimensional"),
    ],
)
def test_driver_rejects_malformed_atom_types(
    atom_types: np.ndarray, message: str
) -> None:
    """Reject atom type metadata before it can change species by coercion/indexing."""
    from xtbloom.dpdata import XTBloomDriver
    from xtbloom.exceptions import XTBloomValueError

    data = _case_data_dict("ketene")
    data["atom_types"] = atom_types
    with pytest.raises(XTBloomValueError, match=message):
        XTBloomDriver().label(data)


def _ensure_minimizer_registered() -> type:
    """Load and return the registered xTBloom dpdata minimizer class."""
    import xtbloom.dpdata as _  # noqa: F401
    from dpdata.driver import Minimizer

    assert "xtbloom" in Minimizer.get_minimizers()
    return Minimizer.get_minimizer("xtbloom")


def _distorted_data(case_id: str, nframes: int = 1, displacement: float = 0.3) -> dict:
    """Return a system whose frames are displaced along the first atom x axis."""
    data = _case_data_dict(case_id, nframes=nframes)
    coords = np.asarray(data["coords"]).copy()
    for frame in range(nframes):
        coords[frame, 0, 0] += displacement * (frame + 1)
    data["coords"] = coords
    return data


def _patch_minimizer_calculator(
    monkeypatch: pytest.MonkeyPatch,
    evaluator: Callable[[int, np.ndarray], tuple[float, float]],
) -> list[list[np.ndarray]]:
    """Replace the native batch with a deterministic position-aware evaluator."""
    calls: list[list[np.ndarray]] = []

    class FakeBatchCalculator:
        """Minimal BatchCalculator test double retaining live structures."""

        def __init__(
            self,
            structures: list[Structure],
            _method: str,
            **_kwargs: object,
        ) -> None:
            self._structures = structures

        def compute(self) -> list[SimpleNamespace]:
            call = len(calls)
            positions = [structure.positions.copy() for structure in self._structures]
            calls.append(positions)
            outputs = []
            for frame_positions in positions:
                energy, force = evaluator(call, frame_positions)
                outputs.append(
                    SimpleNamespace(
                        energy=energy,
                        forces=np.full_like(frame_positions, force),
                    )
                )
            return outputs

        def close(self) -> None:
            pass

    monkeypatch.setattr("xtbloom.dpdata.BatchCalculator", FakeBatchCalculator)
    return calls


def test_minimizer_registered() -> None:
    """Expose the minimizer through dpdata's minimizer registry."""
    minimizer_class = _ensure_minimizer_registered()
    assert minimizer_class.__module__ == "xtbloom.dpdata"


def test_minimizer_applies_bounded_first_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluate the bounded initial move instead of repeating the input geometry."""
    import xtbloom.dpdata as dpdata_module

    calls = _patch_minimizer_calculator(
        monkeypatch, lambda call, _positions: (1.0 - 0.5 * call, 1.0)
    )
    direction = Mock(wraps=dpdata_module.lbfgs_direction)
    monkeypatch.setattr(dpdata_module, "lbfgs_direction", direction)
    labeled = dpdata_module.XTBloomMinimizer(max_steps=1).minimize(
        _case_data_dict("oh_radical")
    )

    assert len(calls) == 2
    assert direction.call_count == 1
    # A unit force gives alpha=0.1 and the fresh L-BFGS direction is +force.
    np.testing.assert_allclose(calls[1][0], calls[0][0] + 0.1)
    np.testing.assert_allclose(labeled["coords"][0] / _BOHR, calls[1][0])


def test_minimizer_uses_per_atom_force_norm_for_fmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a frame active when components pass fmax but the vector norm does not."""
    from xtbloom.dpdata import XTBloomMinimizer

    component = 0.004 / (_HARTREE_TO_EV / _BOHR)

    def diagonal_force(call: int, _positions: np.ndarray) -> tuple[float, float]:
        return (0.0, component) if call == 0 else (-1.0, 0.0)

    calls = _patch_minimizer_calculator(monkeypatch, diagonal_force)
    XTBloomMinimizer(fmax=0.005, max_steps=1).minimize(_case_data_dict("oh_radical"))

    assert len(calls) == 2


def test_minimizer_reports_accepted_state_at_step_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not publish a rejected final trial, even if its force is converged."""
    from xtbloom.dpdata import XTBloomMinimizer

    data = _case_data_dict("oh_radical")

    def reject_trial(call: int, _positions: np.ndarray) -> tuple[float, float]:
        return (0.0, 1.0) if call == 0 else (1.0, 0.0)

    calls = _patch_minimizer_calculator(monkeypatch, reject_trial)
    labeled = XTBloomMinimizer(max_steps=1).minimize(data)

    assert len(calls) == 2
    np.testing.assert_allclose(labeled["coords"], data["coords"])
    np.testing.assert_allclose(labeled["energies"], [0.0])
    np.testing.assert_allclose(
        labeled["forces"], np.full_like(data["coords"], _HARTREE_TO_EV / _BOHR)
    )


def test_minimizer_stops_after_rejection_at_minimum_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn rejection at the alpha floor into an error instead of an infinite loop."""
    from xtbloom.dpdata import XTBloomMinimizer
    from xtbloom.exceptions import XTBloomRuntimeError

    def reject_every_trial(call: int, _positions: np.ndarray) -> tuple[float, float]:
        if call >= 20:
            raise AssertionError("line search did not terminate at the minimum step")
        return (0.0 if call == 0 else 1.0, 1.0)

    calls = _patch_minimizer_calculator(monkeypatch, reject_every_trial)
    with pytest.raises(XTBloomRuntimeError, match="line search stalled"):
        XTBloomMinimizer().minimize(_case_data_dict("oh_radical"))

    assert 2 < len(calls) < 20


def test_system_minimize_converges_and_lowers_energy() -> None:
    """Relax a distorted frame below fmax while lowering the energy."""
    from xtbloom.dpdata import XTBloomDriver

    _ensure_minimizer_registered()
    system = dpdata.System(data=_distorted_data("ketene"))
    initial = system.predict(driver="xtbloom", backend="cpu")
    labeled = system.minimize(
        minimizer="xtbloom",
        driver=XTBloomDriver(backend="cpu"),
        fmax=5e-3,
        max_steps=200,
    )
    forces = np.asarray(labeled.data["forces"])
    energies = np.asarray(labeled.data["energies"])
    coords = np.asarray(labeled.data["coords"])
    assert labeled.get_nframes() == 1
    assert forces.shape == (1, 5, 3)
    assert float(np.max(np.linalg.norm(forces, axis=2))) <= 5e-3
    assert energies[0] <= initial.data["energies"][0]
    assert not np.allclose(coords[0], initial.data["coords"][0], atol=1e-3)


def test_minimize_relaxes_every_frame_in_one_batch() -> None:
    """All frames of a multi-frame system reach fmax in one batch run."""
    from xtbloom.dpdata import XTBloomDriver

    _ensure_minimizer_registered()
    system = dpdata.System(data=_distorted_data("ketene", nframes=3))
    labeled = system.minimize(
        minimizer="xtbloom",
        driver=XTBloomDriver(backend="cpu"),
        fmax=5e-3,
        max_steps=300,
    )
    forces = np.asarray(labeled.data["forces"])
    energies = np.asarray(labeled.data["energies"])
    coords = np.asarray(labeled.data["coords"])
    assert labeled.get_nframes() == 3
    assert forces.shape == (3, 5, 3)
    assert energies.shape == (3,)
    assert float(np.max(np.linalg.norm(forces, axis=2))) <= 5e-3
    # Each frame converged to a different (energy-lowered) geometry.
    assert len({float(e) for e in energies}) == 3
    assert not np.allclose(coords[0], coords[1])


def test_minimizer_raises_for_failed_frames() -> None:
    """A frame that fails SCC raises instead of publishing bogus labels."""
    from xtbloom.dpdata import XTBloomDriver
    from xtbloom.exceptions import XTBloomRuntimeError

    _ensure_minimizer_registered()
    system = dpdata.System(data=_case_data_dict("sif5_minus"))
    with pytest.raises(XTBloomRuntimeError, match="failed systems"):
        system.minimize(
            minimizer="xtbloom",
            driver=XTBloomDriver(backend="cpu", charge=-1, max_scc_iterations=1),
            max_steps=1,
        )


def test_minimizer_rejects_periodic() -> None:
    """Reject periodic dpdata systems unsupported by the molecular ABI."""
    from xtbloom.dpdata import XTBloomDriver
    from xtbloom.exceptions import XTBloomNotSupportedError

    _ensure_minimizer_registered()
    data = _distorted_data("ketene")
    data["nopbc"] = False
    system = dpdata.System(data=data)
    with pytest.raises(XTBloomNotSupportedError):
        system.minimize(minimizer="xtbloom", driver=XTBloomDriver(backend="cpu"))
