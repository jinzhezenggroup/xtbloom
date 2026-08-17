"""Opt-in deterministic recovery for difficult GFN2-xTB SCC solves.

``singlepoint_auto_safe`` preserves the caller's physical problem and retries
only the modified-Broyden numerical policy.  Every attempt is a normal FRESH
single-point calculation at the calculator's requested electronic temperature,
Hamiltonian, charge/energy tolerances, and iteration ceiling.  The helper does
not weaken convergence criteria, change temperature, or reuse a failed
attempt's electronic state.

The initial portfolio is deliberately small and evidence-backed by issue #217:
the public default ``(history=8, damping=0.4)`` followed by two independently
reviewed 300 K recovery policies, ``(2, 0.2)`` and ``(16, 0.4)``.  This module
is a high-level opt-in prototype; it does not change the C ABI or the default
fixed-policy behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from .exceptions import XTBloomNotSupportedError, XTBloomRuntimeError, XTBloomValueError

if TYPE_CHECKING:
    from collections.abc import Sequence

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class SccMixerPolicy:
    """One bounded modified-Broyden policy in the AUTO_SAFE portfolio."""

    history: int
    damping: float

    def __post_init__(self) -> None:
        """Validate and normalize one recovery policy."""
        if isinstance(self.history, bool) or not isinstance(self.history, int):
            raise XTBloomValueError("SCC recovery history must be an integer")
        if self.history < 1 or self.history > 64:
            raise XTBloomValueError("SCC recovery history must lie in [1, 64]")
        damping = float(self.damping)
        if not 0.0 < damping <= 1.0:
            raise XTBloomValueError("SCC recovery damping must lie in (0, 1]")
        object.__setattr__(self, "damping", damping)


AUTO_SAFE_POLICIES: tuple[SccMixerPolicy, ...] = (
    SccMixerPolicy(history=8, damping=0.4),
    SccMixerPolicy(history=2, damping=0.2),
    SccMixerPolicy(history=16, damping=0.4),
)


class _Settings(Protocol):
    model: int
    scc_mixer_history: int
    scc_mixer_damping: float


class _CalculatorLike(Protocol[_ResultT]):
    _warm_start: bool
    _settings: _Settings

    @property
    def method(self) -> str: ...

    def set(self, attribute: str, value: object) -> None: ...

    def singlepoint(self) -> _ResultT: ...


def _is_retryable_scc_failure(error: XTBloomRuntimeError) -> bool:
    """Return whether a high-level failure contains only SCC nonconvergence.

    ``Calculator.singlepoint`` raises one batch-level diagnostic without
    attaching the per-system status as ``error.status``.  Keep the recognition
    intentionally narrow: a mixed diagnostic containing an eigensolver failure
    is not safe to hide behind another mixer retry.
    """
    message = error.message.lower()
    return "scc not converged" in message and "eigensolver failed" not in message


def singlepoint_auto_safe(
    calculator: _CalculatorLike[_ResultT],
    *,
    policies: Sequence[SccMixerPolicy] = AUTO_SAFE_POLICIES,
) -> _ResultT:
    """Run a GFN2 single point through a bounded deterministic SCC portfolio.

    Parameters
    ----------
    calculator
        A :class:`~xtbloom.interface.Calculator`. ``warm_start`` must be false
        because every retry intentionally starts from the immutable FRESH/SAD
        electronic state.
    policies
        Ordered modified-Broyden policies.  The default sequence is the
        evidence-backed AUTO_SAFE portfolio from issue #217.

    Returns
    -------
    Result
        The first converged single-point result.

    Raises
    ------
    XTBloomNotSupportedError
        If used with a method other than GFN2-xTB.
    XTBloomValueError
        If warm-start semantics or an empty portfolio are requested.
    XTBloomRuntimeError
        The first non-SCC numerical/runtime failure, or the final SCC
        nonconvergence after all bounded policies are exhausted.

    Notes
    -----
    The calculator's original mixer history and damping are restored before
    this function returns or raises.  Other compute settings are never changed.
    """
    if calculator.method not in {"GFN2", "GFN2-xTB"}:
        raise XTBloomNotSupportedError(
            "AUTO_SAFE SCC recovery is currently validated only for GFN2-xTB"
        )
    if bool(calculator._warm_start):
        raise XTBloomValueError(
            "AUTO_SAFE SCC recovery requires warm_start=False so every retry is FRESH"
        )

    ordered = tuple(policies)
    if not ordered:
        raise XTBloomValueError(
            "AUTO_SAFE SCC recovery requires at least one mixer policy"
        )
    if not all(isinstance(policy, SccMixerPolicy) for policy in ordered):
        raise XTBloomValueError("AUTO_SAFE policies must be SccMixerPolicy instances")

    original_history = int(calculator._settings.scc_mixer_history)
    original_damping = float(calculator._settings.scc_mixer_damping)
    last_scc_error: XTBloomRuntimeError | None = None

    try:
        for policy in ordered:
            calculator.set("scc_mixer_history", policy.history)
            calculator.set("scc_mixer_damping", policy.damping)
            try:
                return calculator.singlepoint()
            except XTBloomRuntimeError as error:
                if not _is_retryable_scc_failure(error):
                    raise
                last_scc_error = error

        assert last_scc_error is not None
        raise last_scc_error
    finally:
        calculator.set("scc_mixer_history", original_history)
        calculator.set("scc_mixer_damping", original_damping)


__all__ = ["AUTO_SAFE_POLICIES", "SccMixerPolicy", "singlepoint_auto_safe"]
