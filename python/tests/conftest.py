"""Shared pytest fixtures for the gpuxtb Python tests."""

from __future__ import annotations

import os

import pytest

import gpuxtb
from gpuxtb import library


def pytest_configure(config) -> None:
    marker = "cuda: requires a gpuxtb library built with CUDA and a device"
    config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session", autouse=True)
def _ensure_library():
    """Resolve the gpuxtb shared library once for the whole session.

    If no library can be found, the suite is skipped instead of failing so a
    source checkout without a built/installed library still reports cleanly.
    """
    try:
        path = library.library_path()
    except gpuxtb.GPUxtbRuntimeError as exc:
        pytest.skip(f"gpuxtb shared library unavailable: {exc}")
    assert path.is_file(), f"resolved library path is not a file: {path}"
    return path