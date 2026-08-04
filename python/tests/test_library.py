"""Tests for the low-level ctypes binding (``gpuxtb.library``)."""

from __future__ import annotations

import ctypes
import site

import numpy as np
import pytest
from gpuxtb import library
from gpuxtb.exceptions import GPUxtbValueError


def test_version_string():
    assert library.get_version() == "0.1.0"


def test_runtime_search_includes_user_site_packages(monkeypatch, tmp_path):
    user_site = tmp_path / "site-packages"
    runtime_dir = user_site / "nvidia" / "cublas" / "lib"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", lambda: [])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)

    assert runtime_dir in library._runtime_search_dirs()


def test_runtime_search_and_preload_match_scipy_openblas32(monkeypatch, tmp_path):
    """Keep runtime discovery aligned with scipy-openblas32's wheel layout."""
    user_site = tmp_path / "site-packages"
    runtime_dir = user_site / "scipy_openblas32" / "lib"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(site, "getsitepackages", list)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)

    assert runtime_dir in library._runtime_search_dirs()
    assert any(
        "libscipy_openblas.so" in alternatives
        for alternatives in library._RUNTIME_LIBRARY_GROUPS
    )


def test_status_strings():
    assert library.status_string(library.STATUS_SUCCESS) == "success"
    assert (
        library.status_string(library.STATUS_SCC_NOT_CONVERGED) == "SCC not converged"
    )
    assert (
        library.status_string(library.STATUS_EIGENSOLVER_FAILED) == "eigensolver failed"
    )


def test_abi_struct_sizes():
    # The ctypes mirrors must match the C ABI field layout used by validation.
    assert ctypes.sizeof(ctypes.c_void_p) == 8  # supported wheel platforms are 64-bit
    assert ctypes.sizeof(library.ContextOptions) == 32
    assert ctypes.sizeof(library.ConstBuffer) == 24
    assert ctypes.sizeof(library.Buffer) == 24
    assert ctypes.sizeof(library.Batch) == 352
    assert ctypes.sizeof(library.ComputeOptions) == 56
    assert library.ComputeOptions.scc_start_mode.offset == 48
    assert library.ComputeOptions.reserved_v2.offset == 52
    assert ctypes.sizeof(library.BatchResult) == 184
    options = library.ComputeOptions()
    library.load_library().gpuxtb_compute_options_init(
        ctypes.byref(options), ctypes.sizeof(options)
    )
    assert options.electronic_temperature == pytest.approx(
        library.DEFAULT_ELECTRONIC_TEMPERATURE
    )
    assert options.max_scc_iterations == 250
    assert options.scc_start_mode == library.SCC_START_FRESH
    assert options.reserved_v2 == 0
    assert library.SCC_START_WARM == 2


def test_unknown_method_rejected():
    from gpuxtb.exceptions import GPUxtbNotSupportedError
    from gpuxtb.interface import Calculator

    with pytest.raises(GPUxtbValueError):
        Calculator("NoSuchMethod", np.array([1]), np.zeros((1, 3)))
    with pytest.raises(GPUxtbNotSupportedError):
        Calculator("GFN1-xTB", np.array([1]), np.zeros((1, 3)))
