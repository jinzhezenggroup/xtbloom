"""Tests for the dpdata driver plugin (``gpuxtb.dpdata``)."""

from __future__ import annotations

import numpy as np
import pytest

dpdata = pytest.importorskip("dpdata")

import _cases  # noqa: E402

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


def _ensure_driver_registered() -> type:
    """Load and return the registered gpuxtb dpdata driver class."""
    # The entry point is registered after a normal wheel install; for a source
    # checkout we register the module explicitly, mirroring dpdata's loader.
    import gpuxtb.dpdata as _  # noqa: F401
    from dpdata.driver import Driver

    assert "gpuxtb" in Driver.get_drivers()
    return Driver.get_driver("gpuxtb")


def test_driver_registered() -> None:
    """Expose the plugin through dpdata's driver registry."""
    driver_class = _ensure_driver_registered()
    assert driver_class.__module__ == "gpuxtb.dpdata"


def test_label_energies_match_golden() -> None:
    """Match dpdata labels to golden energies and forces in dpdata units."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("ketene"))
    labeled = system.predict(driver="gpuxtb")
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


def test_label_distinct_frames() -> None:
    """Label every distinct frame in one dpdata system."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("ketene", nframes=2, distort=True))
    labeled = system.predict(driver="gpuxtb")
    assert labeled.get_nframes() == 2
    assert labeled.data["energies"].shape == (2,)
    assert labeled.data["forces"].shape == (2, 5, 3)
    assert labeled.data["energies"][0] != pytest.approx(labeled.data["energies"][1])


def test_driver_charge() -> None:
    """Forward a fixed charged-system state through the dpdata driver."""
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("h3_plus"))
    labeled = system.predict(driver="gpuxtb", charge=1)
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
    labeled = system.predict(driver="gpuxtb", multiplicity=2, spin_channels=1)
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
    labeled = system.predict(driver="gpuxtb", spin_channels=1)
    golden = _cases.golden(_cases.case_by_id("oh_radical"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_driver_raises_instead_of_publishing_failed_frame_nans() -> None:
    """Raise instead of publishing NaNs for a failed dpdata frame."""
    _ensure_driver_registered()
    from gpuxtb.exceptions import GPUxtbRuntimeError

    system = dpdata.System(data=_case_data_dict("sif5_minus"))
    with pytest.raises(GPUxtbRuntimeError, match="failed systems"):
        system.predict(driver="gpuxtb", charge=-1, backend="cpu", max_scc_iterations=1)


def test_driver_rejects_periodic() -> None:
    """Reject periodic dpdata systems unsupported by the molecular ABI."""
    _ensure_driver_registered()
    from gpuxtb.exceptions import GPUxtbNotSupportedError

    data = _case_data_dict("ketene")
    data["nopbc"] = False
    system = dpdata.System(data=data)
    with pytest.raises(GPUxtbNotSupportedError):
        system.predict(driver="gpuxtb")


def _ensure_minimizer_registered() -> type:
    """Load and return the registered gpuxtb dpdata minimizer class."""
    import gpuxtb.dpdata as _  # noqa: F401
    from dpdata.driver import Minimizer

    assert "gpuxtb" in Minimizer.get_minimizers()
    return Minimizer.get_minimizer("gpuxtb")


def _distorted_data(case_id: str, nframes: int = 1, displacement: float = 0.3) -> dict:
    """Return a system whose frames are displaced along the first atom x axis."""
    data = _case_data_dict(case_id, nframes=nframes)
    coords = np.asarray(data["coords"]).copy()
    for frame in range(nframes):
        coords[frame, 0, 0] += displacement * (frame + 1)
    data["coords"] = coords
    return data


def test_minimizer_registered() -> None:
    """Expose the minimizer through dpdata's minimizer registry."""
    minimizer_class = _ensure_minimizer_registered()
    assert minimizer_class.__module__ == "gpuxtb.dpdata"


def test_system_minimize_converges_and_lowers_energy() -> None:
    """Relax a distorted frame below fmax while lowering the energy."""
    from gpuxtb.dpdata import GPUxtbDriver

    _ensure_minimizer_registered()
    system = dpdata.System(data=_distorted_data("ketene"))
    initial = system.predict(driver="gpuxtb", backend="cpu")
    labeled = system.minimize(
        minimizer="gpuxtb",
        driver=GPUxtbDriver(backend="cpu"),
        fmax=5e-3,
        max_steps=200,
    )
    forces = np.asarray(labeled.data["forces"])
    energies = np.asarray(labeled.data["energies"])
    coords = np.asarray(labeled.data["coords"])
    assert labeled.get_nframes() == 1
    assert forces.shape == (1, 5, 3)
    assert float(np.max(np.abs(forces))) <= 5e-3
    assert energies[0] <= initial.data["energies"][0]
    assert not np.allclose(coords[0], initial.data["coords"][0], atol=1e-3)


def test_minimize_relaxes_every_frame_in_one_batch() -> None:
    """All frames of a multi-frame system reach fmax in one batch run."""
    from gpuxtb.dpdata import GPUxtbDriver

    _ensure_minimizer_registered()
    system = dpdata.System(data=_distorted_data("ketene", nframes=3))
    labeled = system.minimize(
        minimizer="gpuxtb",
        driver=GPUxtbDriver(backend="cpu"),
        fmax=5e-3,
        max_steps=300,
    )
    forces = np.asarray(labeled.data["forces"])
    energies = np.asarray(labeled.data["energies"])
    coords = np.asarray(labeled.data["coords"])
    assert labeled.get_nframes() == 3
    assert forces.shape == (3, 5, 3)
    assert energies.shape == (3,)
    assert float(np.max(np.abs(forces))) <= 5e-3
    # Each frame converged to a different (energy-lowered) geometry.
    assert len({float(e) for e in energies}) == 3
    assert not np.allclose(coords[0], coords[1])


def test_minimizer_raises_for_failed_frames() -> None:
    """A frame that fails SCC raises instead of publishing bogus labels."""
    from gpuxtb.dpdata import GPUxtbDriver
    from gpuxtb.exceptions import GPUxtbRuntimeError

    _ensure_minimizer_registered()
    system = dpdata.System(data=_case_data_dict("sif5_minus"))
    with pytest.raises(GPUxtbRuntimeError, match="failed systems"):
        system.minimize(
            minimizer="gpuxtb",
            driver=GPUxtbDriver(backend="cpu", charge=-1, max_scc_iterations=1),
            max_steps=1,
        )


def test_minimizer_rejects_periodic() -> None:
    """Reject periodic dpdata systems unsupported by the molecular ABI."""
    from gpuxtb.dpdata import GPUxtbDriver
    from gpuxtb.exceptions import GPUxtbNotSupportedError

    _ensure_minimizer_registered()
    data = _distorted_data("ketene")
    data["nopbc"] = False
    system = dpdata.System(data=data)
    with pytest.raises(GPUxtbNotSupportedError):
        system.minimize(minimizer="gpuxtb", driver=GPUxtbDriver(backend="cpu"))
