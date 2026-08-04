"""Tests for charge, multiplicity, and orbital-channel (spin) handling."""

from __future__ import annotations

import numpy as np
import pytest

from gpuxtb import Calculator, Context, Structure
from gpuxtb.exceptions import GPUxtbNotSupportedError, GPUxtbRuntimeError, GPUxtbValueError

import _cases


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


def _library_has_cuda() -> bool:
    """Check whether a CUDA context can actually be created on this host."""
    try:
        with Context("cuda"):
            pass
        return True
    except GPUxtbRuntimeError:
        return False


def test_cuda_open_shell_scope_is_rejected():
    """Open-shell systems must never reach a CUDA node.

    With a CUDA-capable library this is a scope error; otherwise the explicit
    CUDA backend is unavailable, which is also a clean failure.
    """
    has_cuda = _library_has_cuda()
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
    with pytest.raises(GPUxtbNotSupportedError if has_cuda else GPUxtbRuntimeError):
        calc.singlepoint()


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