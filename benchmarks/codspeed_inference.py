"""Small CPU regression benchmarks for CodSpeed simulation.

These cases are deliberately much smaller than the publication-grade benchmark
protocols in this directory. They exercise the public Python layer and native
C ABI end to end so pull requests get a stable regression signal without
turning CodSpeed numbers into hardware performance claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
from xtbloom import BatchCalculator, BatchResult, Calculator, Result, Structure

from benchmarks.natoms_scaling import make_alkane

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")

_WATER_NUMBERS = np.array([8, 1, 1], dtype=np.int32)
_WATER_POSITIONS = np.array(
    [
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ],
    dtype=np.float64,
)
# Reuse the deterministic 32-atom C10H22 fixture from the audit-ready scaling
# protocol. Water-only timings are dominated by fixed Python/C-ABI and tiny-
# matrix costs; this size exposes model and eigensolver regressions while still
# keeping instruction-simulation CI bounded.
_ALKANE = make_alkane(32)
_ALKANE_NUMBERS = np.asarray(_ALKANE.atomic_numbers, dtype=np.int32)
_ALKANE_POSITIONS = np.asarray(_ALKANE.positions_bohr, dtype=np.float64).reshape(-1, 3)


class _BenchmarkFixture(Protocol):
    """Minimal callable surface used from pytest-codspeed."""

    def __call__(self, target: Callable[[], _T]) -> _T:
        """Measure one target callable and return its final result."""
        ...


def _assert_single_result(result: Result) -> None:
    """Require a complete successful public single-system result."""
    assert result.scc_converged
    assert result.scc_status == 0
    assert np.isfinite(result.energy)
    assert np.isfinite(result.forces).all()
    assert np.isfinite(result.charges).all()


def _assert_batch_result(result: BatchResult) -> None:
    """Require every system in a public ragged batch to succeed."""
    result.raise_for_status()
    assert result.failed_indices.size == 0
    assert np.isfinite(result.energies).all()
    assert np.isfinite(result.forces).all()
    assert np.isfinite(result.charges).all()


def test_gfn2_c10h22_fresh(benchmark: _BenchmarkFixture) -> None:
    """Measure repeated fresh-SCC GFN2 32-atom inference on one CPU worker."""
    with Calculator(
        "GFN2-xTB",
        _ALKANE_NUMBERS,
        _ALKANE_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn2_c10h22_warm(benchmark: _BenchmarkFixture) -> None:
    """Measure strict warm-SCC 32-atom GFN2 reuse after an untimed fresh seed."""
    with Calculator(
        "GFN2-xTB",
        _ALKANE_NUMBERS,
        _ALKANE_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=True,
    ) as calculator:
        seed = calculator.singlepoint()
        _assert_single_result(seed)
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn1_c10h22_fresh(benchmark: _BenchmarkFixture) -> None:
    """Measure repeated fresh-SCC GFN1 32-atom inference on one CPU worker."""
    with Calculator(
        "GFN1-xTB",
        _ALKANE_NUMBERS,
        _ALKANE_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn2_ragged_batch_fresh(benchmark: _BenchmarkFixture) -> None:
    """Measure a mixed-size ragged GFN2 batch through the public Python API."""
    structures = [
        Structure(_WATER_NUMBERS, _WATER_POSITIONS),
        Structure(_ALKANE_NUMBERS, _ALKANE_POSITIONS),
    ]
    with BatchCalculator(
        structures,
        method="GFN2-xTB",
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(
            lambda: calculator.compute(
                raise_on_failure=True,
                auto_batch_size=False,
            )
        )
    _assert_batch_result(result)
