"""Tests for the opt-in AUTO_SAFE SCC recovery portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from xtbloom.exceptions import XTBloomRuntimeError, XTBloomValueError
from xtbloom.interface import Calculator
from xtbloom.scc_recovery import AUTO_SAFE_POLICIES, SccMixerPolicy, singlepoint_auto_safe


@dataclass
class _FakeSettings:
    model: int = 2
    scc_mixer_history: int = 11
    scc_mixer_damping: float = 0.7


class _FakeCalculator:
    method = "GFN2-xTB"

    def __init__(self, outcomes: list[object], *, warm_start: bool = False) -> None:
        self._warm_start = warm_start
        self._settings = _FakeSettings()
        self._outcomes = iter(outcomes)
        self.attempts: list[tuple[int, float]] = []

    def set(self, attribute: str, value: object) -> None:
        setattr(self._settings, attribute, value)

    def singlepoint(self) -> object:
        self.attempts.append(
            (self._settings.scc_mixer_history, self._settings.scc_mixer_damping)
        )
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _scc_failure() -> XTBloomRuntimeError:
    return XTBloomRuntimeError(
        "xTBloom batch inference produced failed systems: system 0: SCC not converged, "
        "scc_converged=0, iterations=250"
    )


def test_auto_safe_uses_reviewed_order_and_restores_settings() -> None:
    calculator = _FakeCalculator([_scc_failure(), "recovered"])

    assert singlepoint_auto_safe(calculator) == "recovered"
    assert calculator.attempts == [(8, 0.4), (2, 0.2)]
    assert calculator._settings.scc_mixer_history == 11
    assert calculator._settings.scc_mixer_damping == pytest.approx(0.7)


def test_auto_safe_stops_after_first_success() -> None:
    calculator = _FakeCalculator(["default-converged"])

    assert singlepoint_auto_safe(calculator) == "default-converged"
    assert calculator.attempts == [(8, 0.4)]


def test_auto_safe_exhaustion_reraises_final_scc_failure() -> None:
    failures = [_scc_failure(), _scc_failure(), _scc_failure()]
    calculator = _FakeCalculator(failures)

    with pytest.raises(XTBloomRuntimeError) as caught:
        singlepoint_auto_safe(calculator)

    assert caught.value is failures[-1]
    assert calculator.attempts == [(8, 0.4), (2, 0.2), (16, 0.4)]
    assert calculator._settings.scc_mixer_history == 11
    assert calculator._settings.scc_mixer_damping == pytest.approx(0.7)


def test_auto_safe_does_not_retry_eigensolver_failure() -> None:
    error = XTBloomRuntimeError(
        "xTBloom batch inference produced failed systems: system 0: eigensolver failed, "
        "scc_converged=0, iterations=1"
    )
    calculator = _FakeCalculator([error])

    with pytest.raises(XTBloomRuntimeError) as caught:
        singlepoint_auto_safe(calculator)

    assert caught.value is error
    assert calculator.attempts == [(8, 0.4)]


def test_auto_safe_rejects_warm_start() -> None:
    calculator = _FakeCalculator([], warm_start=True)
    with pytest.raises(XTBloomValueError, match="warm_start=False"):
        singlepoint_auto_safe(calculator)
    assert calculator.attempts == []


def test_policy_bounds_and_empty_portfolio() -> None:
    with pytest.raises(XTBloomValueError):
        SccMixerPolicy(history=0, damping=0.4)
    with pytest.raises(XTBloomValueError):
        SccMixerPolicy(history=65, damping=0.4)
    with pytest.raises(XTBloomValueError):
        SccMixerPolicy(history=8, damping=0.0)

    calculator = _FakeCalculator([])
    with pytest.raises(XTBloomValueError, match="at least one"):
        singlepoint_auto_safe(calculator, policies=())


def _load_tmacl() -> tuple[list[str], np.ndarray]:
    fixture = Path(__file__).resolve().parents[2] / "data/conformance/inputs/tmacl.xyz"
    lines = fixture.read_text(encoding="utf-8").splitlines()
    assert int(lines[0]) == 18
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in lines[2:20]:
        symbol, x, y, z = line.split()
        symbols.append(symbol)
        positions.append([float(x), float(y), float(z)])
    return symbols, np.asarray(positions, dtype=float) / 0.529177210903


def test_auto_safe_recovers_issue_217_tmacl_at_300k() -> None:
    symbols, positions = _load_tmacl()
    calculator = Calculator(
        "GFN2-xTB",
        symbols,
        positions,
        backend="cpu",
        max_scc_iterations=250,
        electronic_temperature=300.0,
        warm_start=False,
    )

    result = singlepoint_auto_safe(calculator)

    assert result.scc_converged
    assert result.scc_status == 0
    assert np.isfinite(result.energy)
    assert np.isfinite(result.forces).all()
    assert float(np.sum(result.charges[:17])) == pytest.approx(0.8285, abs=2.0e-3)
    assert float(result.charges[17]) == pytest.approx(-0.8285, abs=2.0e-3)
    # The helper is transactional with respect to the caller's compute settings.
    assert calculator._settings.scc_mixer_history == 8
    assert calculator._settings.scc_mixer_damping == pytest.approx(0.4)


def test_default_portfolio_is_stable_contract() -> None:
    assert AUTO_SAFE_POLICIES == (
        SccMixerPolicy(8, 0.4),
        SccMixerPolicy(2, 0.2),
        SccMixerPolicy(16, 0.4),
    )
