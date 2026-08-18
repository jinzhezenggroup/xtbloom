"""Small CPU regression benchmarks for CodSpeed simulation.

These cases are deliberately much smaller than the publication-grade benchmark
protocols in this directory.  They exercise the public Python layer and native
C ABI end to end so pull requests get a stable regression signal without
turning CodSpeed numbers into hardware performance claims.
"""

from __future__ import annotations

import numpy as np
from xtbloom import BatchCalculator, BatchResult, Calculator, Result, Structure

_WATER_NUMBERS = np.array([8, 1, 1], dtype=np.int32)
_WATER_POSITIONS = np.array(
    [
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ],
    dtype=np.float64,
)
_H2_NUMBERS = np.array([1, 1], dtype=np.int32)
_H2_POSITIONS = np.array(
    [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]],
    dtype=np.float64,
)


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


def test_gfn2_water_fresh(benchmark) -> None:
    """Measure repeated fresh-SCC GFN2 water inference on one CPU worker."""
    with Calculator(
        "GFN2-xTB",
        _WATER_NUMBERS,
        _WATER_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn2_water_warm(benchmark) -> None:
    """Measure strict warm-SCC GFN2 reuse after an untimed fresh seed."""
    with Calculator(
        "GFN2-xTB",
        _WATER_NUMBERS,
        _WATER_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=True,
    ) as calculator:
        seed = calculator.singlepoint()
        _assert_single_result(seed)
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn1_water_fresh(benchmark) -> None:
    """Measure repeated fresh-SCC GFN1 water inference on one CPU worker."""
    with Calculator(
        "GFN1-xTB",
        _WATER_NUMBERS,
        _WATER_POSITIONS,
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(calculator.singlepoint)
    _assert_single_result(result)


def test_gfn2_ragged_batch_fresh(benchmark) -> None:
    """Measure one small ragged GFN2 batch through the public Python API."""
    structures = [
        Structure(_H2_NUMBERS, _H2_POSITIONS),
        Structure(_WATER_NUMBERS, _WATER_POSITIONS),
    ]
    with BatchCalculator(
        structures,
        method="GFN2-xTB",
        backend="cpu",
        cpu_threads=1,
        warm_start=False,
    ) as calculator:
        result = benchmark(
            calculator.compute,
            raise_on_failure=True,
            auto_batch_size=False,
        )
    _assert_batch_result(result)
