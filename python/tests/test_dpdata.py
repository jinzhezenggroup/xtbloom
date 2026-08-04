"""Tests for the dpdata driver plugin (``gpuxtb.dpdata``)."""

from __future__ import annotations

import numpy as np
import pytest

dpdata = pytest.importorskip("dpdata")

import _cases  # noqa: E402

_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


def _case_data_dict(case_id, nframes=1, distort=False):
    """Build a dpdata System data dict (coords in Angstrom) from a conformance case."""
    case = _cases.case_by_id(case_id)
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    coords = np.stack([np.asarray(positions) for _ in range(nframes)])
    if distort:
        coords[1] = coords[1].copy()
        coords[1, 0, 0] += 0.5
    coords = coords * _BOHR  # bohr -> Angstrom (dpdata convention)
    atom_names = _cases.numbers_to_symbols(sorted(set(int(z) for z in numbers)))
    type_map = {
        number: index
        for index, number in enumerate(sorted(set(int(z) for z in numbers)))
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


def _ensure_driver_registered():
    # The entry point is registered after a normal wheel install; for a source
    # checkout we register the module explicitly, mirroring dpdata's loader.
    import gpuxtb.dpdata as _  # noqa: F401
    from dpdata.driver import Driver

    assert "gpuxtb" in Driver.get_drivers()
    return Driver.get_driver("gpuxtb")


def test_driver_registered():
    driver_class = _ensure_driver_registered()
    assert driver_class.__module__ == "gpuxtb.dpdata"


def test_label_energies_match_golden():
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


def test_label_distinct_frames():
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("ketene", nframes=2, distort=True))
    labeled = system.predict(driver="gpuxtb")
    assert labeled.get_nframes() == 2
    assert labeled.data["energies"].shape == (2,)
    assert labeled.data["forces"].shape == (2, 5, 3)
    assert labeled.data["energies"][0] != pytest.approx(labeled.data["energies"][1])


def test_driver_charge():
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("h3_plus"))
    labeled = system.predict(driver="gpuxtb", charge=1)
    golden = _cases.golden(_cases.case_by_id("h3_plus"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_driver_multiplicity():
    _ensure_driver_registered()
    system = dpdata.System(data=_case_data_dict("oh_radical"))
    labeled = system.predict(driver="gpuxtb", multiplicity=2, spin_channels=1)
    golden = _cases.golden(_cases.case_by_id("oh_radical"))
    tolerance = _cases.tolerances()
    assert labeled.data["energies"][0] == pytest.approx(
        golden["energy_hartree"] * _HARTREE_TO_EV,
        abs=tolerance["energy"]["atol"] * _HARTREE_TO_EV,
    )


def test_driver_reads_per_frame_multiplicity_without_forcing_uhf_zero():
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


def test_driver_raises_instead_of_publishing_failed_frame_nans():
    _ensure_driver_registered()
    from gpuxtb.exceptions import GPUxtbRuntimeError

    system = dpdata.System(data=_case_data_dict("sif5_minus"))
    with pytest.raises(GPUxtbRuntimeError, match="failed systems"):
        system.predict(
            driver="gpuxtb", charge=-1, backend="cpu", max_scc_iterations=1
        )


def test_driver_rejects_periodic():
    _ensure_driver_registered()
    from gpuxtb.exceptions import GPUxtbNotSupportedError

    data = _case_data_dict("ketene")
    data["nopbc"] = False
    system = dpdata.System(data=data)
    with pytest.raises(GPUxtbNotSupportedError):
        system.predict(driver="gpuxtb")
