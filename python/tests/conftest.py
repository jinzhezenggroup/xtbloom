"""Shared pytest fixtures for the xTBloom Python tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import xtbloom
from xtbloom import library


def pytest_configure(config: pytest.Config) -> None:
    """Register the CUDA availability marker."""
    marker = "cuda: requires an xTBloom library built with CUDA and a device"
    config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session", autouse=True)
def _ensure_library() -> str | Path:
    """Resolve the xTBloom shared library once for the whole session.

    Missing or unloadable native code is a packaging/test failure.  In
    particular, wheel CI must never turn a misplaced shared library into an
    all-skipped green job.
    """
    try:
        path = library.library_path()
        library.load_library()
    except xtbloom.XTBloomRuntimeError as exc:
        pytest.fail(f"xTBloom shared library unavailable: {exc}")
    if isinstance(path, Path):
        assert path.is_file(), f"resolved library path is not a file: {path}"
    return path
