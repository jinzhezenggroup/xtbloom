"""Tests for direct Python geometry optimization."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from xtbloom import optimize, optimize_batch
from xtbloom.exceptions import XTBloomValueError

_optimize_module = importlib.import_module("xtbloom.optimize")


class FakeStructure:
    def __init__(self, positions: list[list[float]]) -> None:
        self._positions = np.asarray(positions, dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    def update(self, *, positions: np.ndarray) -> None:
        self._positions = np.asarray(positions, dtype=np.float64).copy()


class FakeSingleResult:
    def __init__(self, structure: FakeStructure) -> None:
        self.energy = 0.5 * float(np.sum(structure.positions**2))
        self.forces = -structure.positions.copy()


class FakeCalculator(FakeStructure):
    def singlepoint(self) -> FakeSingleResult:
        return FakeSingleResult(self)


class FakeBatchResult:
    def __init__(self, structures: list[FakeStructure]) -> None:
        self._results = [FakeSingleResult(structure) for structure in structures]

    def __getitem__(self, index: int) -> FakeSingleResult:
        return self._results[index]


class FakeBatchCalculator:
    closed = False

    def __init__(
        self,
        structures: list[FakeStructure],
        _method: str,
        **_kwargs: object,
    ) -> None:
        self.structures = structures

    def compute(self) -> FakeBatchResult:
        return FakeBatchResult(self.structures)

    def close(self) -> None:
        self.closed = True


def test_optimize_converges_quadratic_and_updates_calculator() -> None:
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    result = optimize(calculator, fmax=1.0e-10, max_steps=10)

    assert result.all_converged
    assert result.steps.tolist() == [2]
    np.testing.assert_allclose(result.positions[0], 0.0, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(calculator.positions, result.positions[0])
    np.testing.assert_allclose(result.forces[0], 0.0, atol=1.0e-14, rtol=0.0)


def test_max_steps_returns_last_energy_accepted_geometry() -> None:
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    result = optimize(calculator, fmax=1.0e-12, max_steps=1)

    assert not result.all_converged
    assert result.steps.tolist() == [1]
    np.testing.assert_allclose(result.positions[0], [[0.9, 0.0, 0.0]])
    np.testing.assert_allclose(calculator.positions, result.positions[0])


def test_optimize_batch_uses_one_reusable_ragged_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_optimize_module, "BatchCalculator", FakeBatchCalculator)
    structures = [
        FakeStructure([[1.0, 0.0, 0.0]]),
        FakeStructure([[0.0, -2.0, 0.0], [0.0, 0.5, 0.0]]),
    ]

    result = optimize_batch(
        structures,  # type: ignore[arg-type]
        fmax=1.0e-10,
        max_steps=10,
        backend="cpu",
    )

    assert result.all_converged
    assert result.positions[0].shape == (1, 3)
    assert result.positions[1].shape == (2, 3)
    for position in result.positions:
        np.testing.assert_allclose(position, 0.0, atol=1.0e-14, rtol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fmax": 0.0},
        {"fmax": np.nan},
        {"max_steps": 0},
        {"max_steps": 1.5},
        {"memory": 0},
        {"memory": False},
    ],
)
def test_optimize_rejects_invalid_controls(kwargs: dict[str, object]) -> None:
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    with pytest.raises(XTBloomValueError):
        optimize(calculator, **kwargs)  # type: ignore[arg-type]
