"""Tests for charge, multiplicity, and orbital-channel (spin) handling."""

from __future__ import annotations

import _cases
import numpy as np
import pytest
from gpuxtb import Calculator, Context, Structure, library, numbers_to_symbols
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError


def test_multiplicity_maps_to_unpaired_electrons():
    calc = Calculator(
        "GFN2-xTB",
        np.array([1]),
        np.zeros((1, 3)),
        multiplicity=2,
    )
    assert calc.uhf == 1
    assert calc.multiplicity == 2
    calc2 = Calculator("GFN2-xTB", np.array([1]), np.zeros((1, 3)), multiplicity=3)
    assert calc2.uhf == 2


def test_default_spin_channels_follow_open_shell():
    closed = Calculator("GFN2-xTB", np.array([1, 1]), np.zeros((2, 3)))
    assert closed.spin_channels == 1
    open_shell = Calculator("GFN2-xTB", np.array([1, 1]), np.zeros((2, 3)), uhf=1)
    assert open_shell.spin_channels == 2


def test_structure_rejects_invalid_inputs():
    with pytest.raises(GPUxtbValueError):
        Structure(np.array([1]), np.zeros((1, 3)), spin_channels=3)
    with pytest.raises(GPUxtbValueError):
        Structure(np.array([1]), np.zeros((1, 3)), multiplicity=0)
    with pytest.raises(GPUxtbValueError):
        Structure(np.array([1]), np.zeros((1, 3)), uhf=1, multiplicity=1)
    with pytest.raises(GPUxtbValueError, match="exact integers"):
        Structure(np.array([1.9]), np.zeros((1, 3)))
    with pytest.raises(GPUxtbValueError, match="exact integers"):
        Structure(np.array([True]), np.zeros((1, 3)))
    with pytest.raises(GPUxtbValueError, match="exact integers"):
        Structure([True, 1], np.zeros((2, 3)))


@pytest.mark.parametrize("numbers", [[6.9], [0], [-1], [119], [True]])
def test_numbers_to_symbols_rejects_invalid_atomic_numbers(numbers):
    with pytest.raises(GPUxtbValueError):
        numbers_to_symbols(numbers)


def _library_has_cuda() -> bool:
    """Check whether a CUDA context can actually be created on this host."""
    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


@pytest.mark.cuda
def test_cuda_unrestricted_open_shell_is_supported():
    """The public Python CUDA path accepts ABI-v2 unrestricted radicals."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    case = _cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, _ = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=2,
        backend="cuda",
    )
    result = calc.singlepoint()
    assert calc.backend == library.BACKEND_CUDA
    assert np.isfinite(result.energy)
    assert np.all(np.isfinite(result.forces))
    assert np.all(np.isfinite(result.charges))

    step = 1.0e-4
    analytic_force = result.forces[1, 2]
    displaced = np.array(positions, copy=True)
    displaced[1, 2] += step
    calc.update(positions=displaced)
    energy_plus = calc.singlepoint().energy
    displaced[1, 2] -= 2.0 * step
    calc.update(positions=displaced)
    energy_minus = calc.singlepoint().energy
    finite_difference = -(energy_plus - energy_minus) / (2.0 * step)
    assert analytic_force == pytest.approx(finite_difference, abs=2.0e-5)


@pytest.mark.cuda
def test_auto_keeps_cuda_for_spin_updates_on_gpu_hosts():
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")

    radical = _cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(radical)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="auto",
    )
    assert calc.backend == library.BACKEND_CUDA
    assert np.isfinite(calc.singlepoint().energy)

    closed_case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(closed_case)
    changed = Calculator(
        "GFN2-xTB", numbers, positions, charge=charge, uhf=uhf, spin_channels=spin
    )
    assert changed.backend == library.BACKEND_CUDA
    changed.update(spin_channels=2)  # unrestricted singlet after CUDA creation
    assert changed.backend == library.BACKEND_CUDA
    assert np.isfinite(changed.singlepoint().energy)
    changed.update(spin_channels=1)
    assert changed.backend == library.BACKEND_CUDA
    assert np.isfinite(changed.singlepoint().energy)


@pytest.mark.cuda
def test_cuda_closed_shell_matches_golden():
    """A restricted closed-shell CUDA run must reproduce the golden numbers."""
    if not _library_has_cuda():
        pytest.skip("CUDA backend is not available on this host")
    case = _cases.case_by_id("ketene")
    numbers, positions, charge, uhf, spin = _cases.structure_inputs(case)
    calc = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cuda",
    )
    result = calc.singlepoint()
    tolerance = _cases.tolerances()
    assert result.energy == pytest.approx(
        _cases.golden(case)["energy_hartree"], abs=tolerance["energy"]["atol"]
    )
