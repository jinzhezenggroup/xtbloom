"""Batch-native L-BFGS geometry stepping on the Array API.

The dpdata minimizer (:mod:`xtbloom.dpdata`) moves every frame of a system in
lockstep and evaluates energies and forces for all active frames in one xTBloom
ragged-batch call per iteration.  The per-frame stepping mathematics lives here
and is written against the Array API namespace (via ``array_api_compat``)
instead of NumPy directly, so the same kernels can be reused later with any
Array API backend that supplies gradients.  xTBloom backends currently return
NumPy arrays, so callers pass plain ``numpy.ndarray`` objects; the code never
mutates its inputs and returns fresh arrays in the input namespace.

Only elementwise operations and reductions are used, so every function is also
valid on device-resident arrays (PyTorch, CuPy, JAX) that implement the Array
API standard.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

from ._array import _compat
from .exceptions import XTBloomValueError

#: Guard against dividing by an exactly-zero curvature or gradient norm.
_EPS = 1.0e-30

# Shared energy-acceptance and line-search policy. The direct Python optimizer
# and the dpdata minimizer use the same per-system controller while retaining
# their different outer batching schedules and public unit conventions.
_ACCEPT_TOLERANCE = 1.0e-8
_ALPHA_DECAY = 0.5
_ALPHA_MIN = 1.0e-4


def _validated_controls(
    fmax: object, max_steps: object | None, memory: object
) -> tuple[float, int | None, int]:
    """Normalize public optimizer controls without accepting booleans/floats as ints."""
    if isinstance(fmax, bool | np.bool_):
        raise XTBloomValueError("fmax must be finite and positive")
    try:
        threshold = float(cast("SupportsFloat | SupportsIndex", fmax))
    except (TypeError, ValueError, OverflowError):
        raise XTBloomValueError("fmax must be finite and positive") from None
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise XTBloomValueError("fmax must be finite and positive")

    if max_steps is None:
        steps = None
    else:
        if isinstance(max_steps, bool | np.bool_):
            raise XTBloomValueError("max_steps must be a positive integer or None")
        try:
            steps = operator.index(cast("SupportsIndex", max_steps))
        except (TypeError, ValueError, OverflowError):
            raise XTBloomValueError(
                "max_steps must be a positive integer or None"
            ) from None
        if steps < 1:
            raise XTBloomValueError("max_steps must be a positive integer or None")

    if isinstance(memory, bool | np.bool_):
        raise XTBloomValueError("memory must be a positive integer")
    try:
        memory_value = operator.index(cast("SupportsIndex", memory))
    except (TypeError, ValueError, OverflowError):
        raise XTBloomValueError("memory must be a positive integer") from None
    if memory_value < 1:
        raise XTBloomValueError("memory must be a positive integer")
    return threshold, steps, memory_value


def lbfgs_direction(
    gradient: np.ndarray,
    s_history: list[np.ndarray],
    y_history: list[np.ndarray],
    rho_history: list[float],
) -> np.ndarray:
    """Return the two-loop-recursion L-BFGS search direction ``-H g``.

    Parameters
    ----------
    gradient : (..., 3) array
        Energy gradient dE/dR at the current position (any consistent units,
        e.g. Hartree/bohr).
    s_history : list
        Position differences ``x_new - x_old`` of accepted steps, oldest
        first.  Empty for a fresh frame.
    y_history : list
        Gradient differences ``g_new - g_old`` of the same accepted steps.
    rho_history : list
        ``1 / (s . y)`` for each stored pair.

    Returns
    -------
    (..., 3) array
        Approximate Newton step.  With an empty history the method degenerates
        to steepest descent ``-g``.
    """
    xp = _compat().array_namespace(gradient)
    q = -gradient
    if not s_history:
        return q

    alphas: list[float] = []
    for s, y, rho in zip(
        reversed(s_history), reversed(y_history), reversed(rho_history), strict=True
    ):
        alpha = rho * float(xp.sum(s * q))
        alphas.append(alpha)
        q = q - alpha * y

    # Barzilai-Borwein-like initial inverse-Hessian scale from the newest pair.
    s_last, y_last = s_history[-1], y_history[-1]
    gamma = float(xp.sum(s_last * y_last)) / max(float(xp.sum(y_last * y_last)), _EPS)
    result = gamma * q

    for s, y, rho, alpha in zip(
        s_history, y_history, rho_history, reversed(alphas), strict=True
    ):
        beta = rho * float(xp.sum(y * result))
        result = result + (alpha - beta) * s
    return result


def initial_step_size(
    gradient: np.ndarray,
    *,
    scale: float = 0.1,
    maximum: float = 0.5,
) -> float:
    """Return a conservative first gradient-descent step length.

    The first L-BFGS direction of a fresh frame is ``-g`` (no curvature
    information yet), whose magnitude is the raw gradient.  Scale it so the
    first trial move is a small fraction of the steepest-descent distance,
    clamped to ``maximum`` for near-converged starting geometries.
    """
    xp = _compat().array_namespace(gradient)
    inf_norm = float(xp.max(xp.abs(gradient)))
    return min(maximum, scale / max(inf_norm, _EPS))


def curvature(s: np.ndarray, y: np.ndarray) -> float:
    """Return the curvature ``s . y`` of an accepted step pair."""
    xp = _compat().array_namespace(s)
    return float(xp.sum(s * y))


@dataclass
class _LBFGSState:
    """Last accepted state and L-BFGS history for one NumPy structure.

    Outer controllers decide which systems remain active and how evaluations
    are batched. This object owns the shared per-system invariant: only an
    energy-accepted evaluation replaces the published position, energy, force,
    gradient, or curvature history.
    """

    position: np.ndarray
    energy: float
    force: np.ndarray
    gradient: np.ndarray
    alpha: float
    memory: int
    s_history: list[np.ndarray]
    y_history: list[np.ndarray]
    rho_history: list[float]
    steps: int = 0

    @classmethod
    def from_evaluation(
        cls,
        position: np.ndarray,
        energy: float,
        force: np.ndarray,
        memory: int,
    ) -> _LBFGSState:
        """Create the accepted baseline and its conservative first step size."""
        accepted_position = np.asarray(position, dtype=np.float64).copy()
        accepted_force = np.asarray(force, dtype=np.float64).copy()
        gradient = -accepted_force
        return cls(
            position=accepted_position,
            energy=float(energy),
            force=accepted_force,
            gradient=gradient,
            alpha=initial_step_size(gradient),
            memory=memory,
            s_history=[],
            y_history=[],
            rho_history=[],
        )

    def accepts(self, energy: float) -> bool:
        """Return whether a trial energy is within the accepted noise envelope."""
        limit = self.energy + _ACCEPT_TOLERANCE * max(1.0, abs(self.energy))
        return energy <= limit

    def _trial(self) -> np.ndarray:
        direction = lbfgs_direction(
            self.gradient,
            self.s_history,
            self.y_history,
            self.rho_history,
        )
        return self.position + self.alpha * direction

    def initial_trial(self) -> np.ndarray:
        """Return the first bounded steepest-descent trial."""
        return self._trial()

    def backoff_trial(self) -> np.ndarray | None:
        """Return a shorter trial, or ``None`` after rejecting the alpha floor."""
        if self.alpha <= _ALPHA_MIN:
            return None
        self.alpha = max(self.alpha * _ALPHA_DECAY, _ALPHA_MIN)
        return self._trial()

    def accept_trial(
        self, position: np.ndarray, energy: float, force: np.ndarray
    ) -> None:
        """Commit one accepted trial and retain only positive-curvature history."""
        accepted_position = np.asarray(position, dtype=np.float64).copy()
        accepted_force = np.asarray(force, dtype=np.float64).copy()
        gradient = -accepted_force
        s_step = accepted_position - self.position
        y_step = gradient - self.gradient
        pair_curvature = curvature(s_step, y_step)
        if pair_curvature > 0.0:
            self.s_history.append(s_step)
            self.y_history.append(y_step)
            self.rho_history.append(1.0 / pair_curvature)
            if len(self.s_history) > self.memory:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)

        self.position = accepted_position
        self.energy = float(energy)
        self.force = accepted_force
        self.gradient = gradient
        self.alpha = 1.0
        self.steps += 1

    def next_trial(self) -> np.ndarray:
        """Return the full L-BFGS trial after an accepted step."""
        return self._trial()


__all__ = ["curvature", "initial_step_size", "lbfgs_direction"]
