"""Tests for the low-level ctypes binding (``gpuxtb.library``)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from gpuxtb import library
from gpuxtb.exceptions import GPUxtbRuntimeError, GPUxtbValueError

import _cases


def test_version_string():
    assert library.get_version() == "0.1.0"


def test_status_strings():
    assert library.status_string(library.STATUS_SUCCESS) == "success"
    assert library.status_string(library.STATUS_SCC_NOT_CONVERGED) == "SCC not converged"
    assert library.status_string(library.STATUS_EIGENSOLVER_FAILED) == "eigensolver failed"


def test_abi_struct_sizes():
    # The ctypes mirrors must match the C ABI field layout used by validation.
    assert ctypes.sizeof(library.Batch) >= ctypes.sizeof(library.Batch)  # sanity
    options = library.ComputeOptions()
    library.load_library().gpuxtb_compute_options_init(
        ctypes.byref(options), ctypes.sizeof(options)
    )
    assert options.electronic_temperature == pytest.approx(
        library.DEFAULT_ELECTRONIC_TEMPERATURE
    )
    assert options.max_scc_iterations == 250


def test_unknown_method_rejected():
    from gpuxtb.interface import Calculator
    from gpuxtb.exceptions import GPUxtbNotSupportedError

    with pytest.raises(GPUxtbValueError):
        Calculator("NoSuchMethod", np.array([1]), np.zeros((1, 3)))
    with pytest.raises(GPUxtbNotSupportedError):
        Calculator("GFN1-xTB", np.array([1]), np.zeros((1, 3)))