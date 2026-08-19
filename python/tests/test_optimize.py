"""Tests for direct Python geometry optimization."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from xtbloom import optimize, optimize_batch
from xtbloom.exceptions import XTBloomValueError

_optimize_module = importlib.import_module("xtbloom.optimize")


class FakeStructure:
    """Minimal mutable structure for optimizer controller tests."""

    def __init__(self, positions: list[list[float]]) -> None:
        """Store one Cartesian coordinate array."""
        self._positions = np.asarray(positions, dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        """Return the current Cartesian coordinates."""
        return self._positions

    def update(self, *, positions: np.ndarray) -> None:
        """Replace the current Cartesian coordinates."""
        self._positions = np.asarray(positions, dtype=np.float64).copy()


class FakeSingleResult:
    """Quadratic energy/force result at one fake structure."""

    def __init__(self, structure: FakeStructure) -> None:
        """Evaluate E=x^2/2 and F=-x."""
        self.energy = 0.5 * float(np.sum(structure.positions**2))
        self.forces = -structure.positions.copy()


class FakeCalculator(FakeStructure):
    """Single-system quadratic calculator for deterministic optimization."""

    def singlepoint(self) -> FakeSingleResult:
        """Return the quadratic energy and analytic force."""
        return FakeSingleResult(self)


class FakeBatchResult:
    """Indexable collection of quadratic single-system results."""

    def __init__(self, structures: list[FakeStructure]) -> None:
        """Evaluate every fake structure in input order."""
        self._results = [FakeSingleResult(structure) for structure in structures]

    def __getitem__(self, index: int) -> FakeSingleResult:
        """Return one fake single-system result."""
        return self._results[index]


class FakeBatchCalculator:
    """Reusable ragged calculator stand-in for controller tests."""

    closed = False

    def __init__(
        self,
        structures: list[FakeStructure],
        _method: str,
        **_kwargs: object,
    ) -> None:
        """Retain the caller-owned fake structures."""
        self.structures = structures

    def compute(self) -> FakeBatchResult:
        """Evaluate the current coordinates of every fake structure."""
        return FakeBatchResult(self.structures)

    def close(self) -> None:
        """Record resource closure."""
        self.closed = True


def test_optimize_converges_quadratic_and_updates_calculator() -> None:
    """Converge a one-dimensional quadratic and leave its accepted geometry."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    result = optimize(calculator, fmax=1.0e-10, max_steps=10)

    assert result.all_converged
    assert result.steps.tolist() == [2]
    np.testing.assert_allclose(result.positions[0], 0.0, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(calculator.positions, result.positions[0])
    np.testing.assert_allclose(result.forces[0], 0.0, atol=1.0e-14, rtol=0.0)


def test_max_steps_returns_last_energy_accepted_geometry() -> None:
    """Publish the last accepted state rather than an unevaluated next trial."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    result = optimize(calculator, fmax=1.0e-12, max_steps=1)

    assert not result.all_converged
    assert result.steps.tolist() == [1]
    np.testing.assert_allclose(result.positions[0], [[0.9, 0.0, 0.0]])
    np.testing.assert_allclose(calculator.positions, result.positions[0])


def test_optimize_batch_uses_one_reusable_ragged_calculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converge differently shaped systems through one reusable batch object."""
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
    """Reject invalid force thresholds, step limits, and history sizes."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    with pytest.raises(XTBloomValueError):
        optimize(calculator, **kwargs)  # type: ignore[arg-type]
