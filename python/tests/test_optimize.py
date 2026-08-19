"""Tests for direct Python geometry optimization."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from xtbloom import Calculator, Structure, library, optimize, optimize_batch
from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError

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
        self.scc_status = library.STATUS_SUCCESS
        self.scc_converged = True
        self.scc_iterations = 4


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
    assert not result.failed.any()
    assert result.failed_indices.tolist() == []
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


def test_rejected_trial_at_step_limit_restores_accepted_geometry() -> None:
    """Do not publish or leave the calculator at a rejected final trial."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    calls = 0

    def singlepoint() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                energy=0.0,
                forces=np.array([[-1.0, 0.0, 0.0]]),
            )
        return SimpleNamespace(energy=1.0, forces=np.zeros((1, 3)))

    calculator.singlepoint = singlepoint  # type: ignore[method-assign]
    original = calculator.positions.copy()
    result = optimize(calculator, fmax=1.0e-12, max_steps=1)

    assert calls == 2
    assert result.steps.tolist() == [0]
    assert not result.converged[0]
    np.testing.assert_allclose(result.positions[0], original)
    np.testing.assert_allclose(calculator.positions, original)


def test_line_search_stall_raises_and_restores_accepted_geometry() -> None:
    """Rejecting the alpha floor is finite and leaves the strict input accepted."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    calls = 0

    def singlepoint() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls > 20:
            raise AssertionError("line search did not terminate")
        return SimpleNamespace(
            energy=0.0 if calls == 1 else 1.0,
            forces=np.array([[-1.0, 0.0, 0.0]]),
        )

    calculator.singlepoint = singlepoint  # type: ignore[method-assign]
    original = calculator.positions.copy()
    with pytest.raises(XTBloomRuntimeError, match="line search stalled"):
        optimize(calculator, fmax=1.0e-12, max_steps=None)

    assert 2 < calls < 20
    np.testing.assert_allclose(calculator.positions, original)


def test_evaluator_exception_restores_last_accepted_geometry() -> None:
    """Restore accepted coordinates when a later native evaluation raises."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    calls = 0

    def singlepoint() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic evaluator failure")
        return SimpleNamespace(
            energy=0.5 * float(np.sum(calculator.positions**2)),
            forces=-calculator.positions.copy(),
        )

    calculator.singlepoint = singlepoint  # type: ignore[method-assign]
    original = calculator.positions.copy()
    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        optimize(calculator, fmax=1.0e-12, max_steps=2)

    np.testing.assert_allclose(calculator.positions, [[0.9, 0.0, 0.0]])
    assert not np.array_equal(calculator.positions, original)


@pytest.mark.parametrize("malformed", ["count", "shape"])
def test_malformed_evaluator_output_restores_original_geometry(malformed: str) -> None:
    """Treat evaluator contract violations as call-level errors."""
    structure = FakeStructure([[1.0, 0.0, 0.0]])
    original = structure.positions.copy()

    def evaluate() -> list[object]:
        if malformed == "count":
            return []
        return [_optimize_module._Evaluation(0.0, np.zeros((2, 3)))]

    with pytest.raises(XTBloomRuntimeError, match="evaluator returned"):
        _optimize_module._optimize_structures(
            [structure],
            evaluate,
            fmax=1.0e-4,
            max_steps=1,
            memory=5,
            peer_local_failures=False,
        )

    np.testing.assert_allclose(structure.positions, original)


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


def test_optimize_batch_keeps_numerical_failure_peer_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop one failed peer while another continues to an accepted minimum."""

    class PeerFailureBatchCalculator(FakeBatchCalculator):
        def __init__(
            self,
            structures: list[FakeStructure],
            method: str,
            **kwargs: object,
        ) -> None:
            super().__init__(structures, method, **kwargs)
            self.calls = 0

        def compute(self) -> FakeBatchResult:
            result = super().compute()
            if self.calls == 2:
                failed = result[0]
                failed.energy = np.nan
                failed.forces.fill(np.nan)
                failed.scc_status = library.STATUS_SCC_NOT_CONVERGED
                failed.scc_converged = False
                failed.scc_iterations = 9
            self.calls += 1
            return result

    monkeypatch.setattr(_optimize_module, "BatchCalculator", PeerFailureBatchCalculator)
    monkeypatch.setattr(
        _optimize_module.library,
        "status_string",
        lambda status: f"status {status}",
    )
    structures = [
        FakeStructure([[1.0, 0.0, 0.0]]),
        FakeStructure([[2.0, 0.0, 0.0]]),
    ]
    accepted_failed = np.array([[0.9, 0.0, 0.0]])

    result = optimize_batch(
        structures,  # type: ignore[arg-type]
        fmax=1.0e-10,
        max_steps=10,
        backend="cpu",
    )

    assert not result.all_converged
    assert result.failed.tolist() == [True, False]
    assert result.failed_indices.tolist() == [0]
    assert result.failure_messages[0] is not None
    assert "scc_converged=0" in result.failure_messages[0]
    assert result.failure_messages[1] is None
    np.testing.assert_allclose(result.positions[0], accepted_failed)
    np.testing.assert_allclose(structures[0].positions, accepted_failed)
    np.testing.assert_allclose(result.positions[1], 0.0, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(structures[1].positions, result.positions[1])
    with pytest.raises(XTBloomRuntimeError, match="system 0"):
        result.raise_for_status()
    np.testing.assert_allclose(result.positions[1], 0.0, atol=1.0e-14, rtol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fmax": True},
        {"fmax": np.bool_(False)},
        {"fmax": 0.0},
        {"fmax": -1.0},
        {"fmax": np.nan},
        {"fmax": np.inf},
        {"max_steps": 0},
        {"max_steps": -1},
        {"max_steps": True},
        {"max_steps": np.bool_(False)},
        {"max_steps": 1.0},
        {"max_steps": 1.5},
        {"max_steps": np.nan},
        {"max_steps": np.inf},
        {"memory": 0},
        {"memory": -1},
        {"memory": False},
        {"memory": np.bool_(True)},
        {"memory": 2.0},
        {"memory": np.nan},
        {"memory": np.inf},
    ],
)
def test_optimize_rejects_invalid_controls(kwargs: dict[str, object]) -> None:
    """Reject invalid force thresholds, step limits, and history sizes."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    with pytest.raises(XTBloomValueError):
        optimize(calculator, **kwargs)  # type: ignore[arg-type]


def test_optimize_accepts_numpy_integer_controls() -> None:
    """Accept exact NumPy integer scalars through the index protocol."""
    calculator = FakeCalculator([[1.0, 0.0, 0.0]])
    result = optimize(
        calculator,
        fmax=1.0e-12,
        max_steps=np.int64(1),  # type: ignore[arg-type]
        memory=np.int64(2),  # type: ignore[arg-type]
    )
    assert result.evaluations == 2


def test_optimize_batch_validates_controls_before_calculator_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optimizer controls before acquiring a native batch context."""

    def unexpected_calculator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid controls reached BatchCalculator")

    monkeypatch.setattr(_optimize_module, "BatchCalculator", unexpected_calculator)
    with pytest.raises(XTBloomValueError):
        optimize_batch(
            [FakeStructure([[1.0, 0.0, 0.0]])],  # type: ignore[list-item]
            max_steps=1.0,  # type: ignore[arg-type]
        )


def test_optimize_batch_rejects_aliased_structures_before_calculator_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject one mutable structure occupying multiple controller slots."""

    def unexpected_calculator(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("aliased structures reached BatchCalculator")

    monkeypatch.setattr(_optimize_module, "BatchCalculator", unexpected_calculator)
    structure = FakeStructure([[1.0, 0.0, 0.0]])
    original = structure.positions.copy()

    with pytest.raises(XTBloomValueError, match="distinct mutable objects"):
        optimize_batch([structure, structure])  # type: ignore[list-item]

    np.testing.assert_array_equal(structure.positions, original)


def test_optimize_executes_native_cpu_and_restores_returned_state() -> None:
    """Exercise the direct optimizer through real CPU SCC and force calls."""
    positions = np.array([[-0.80, 0.0, 0.0], [0.80, 0.0, 0.0]])
    with Calculator(
        "GFN2-xTB", [1, 1], positions, backend="cpu", warm_start=True
    ) as calculator:
        result = optimize(calculator, fmax=1.0e-12, max_steps=1)
        assert result.evaluations == 2
        assert not result.failed.any()
        assert np.isfinite(result.energies).all()
        assert np.isfinite(result.forces[0]).all()
        np.testing.assert_allclose(calculator.positions, result.positions[0])


def test_optimize_batch_executes_native_ragged_cpu() -> None:
    """Exercise real CPU evaluations for differently sized molecular peers."""
    structures = [
        Structure([1, 1], np.array([[-0.80, 0.0, 0.0], [0.80, 0.0, 0.0]])),
        Structure(
            [8, 1, 1],
            np.array(
                [
                    [0.0, 0.0, -0.45],
                    [0.0, 1.40, 0.55],
                    [0.0, -1.40, 0.55],
                ]
            ),
        ),
    ]
    result = optimize_batch(
        structures,
        backend="cpu",
        warm_start=True,
        fmax=1.0e-12,
        max_steps=1,
    )

    assert result.evaluations == 2
    assert not result.failed.any()
    assert np.isfinite(result.energies).all()
    assert all(np.isfinite(force).all() for force in result.forces)
    for structure, position in zip(structures, result.positions, strict=True):
        np.testing.assert_allclose(structure.positions, position)
