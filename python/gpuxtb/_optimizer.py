"""Batch-native L-BFGS geometry stepping on the Array API.

The dpdata minimizer (:mod:`gpuxtb.dpdata`) moves every frame of a system in
lockstep and evaluates energies and forces for all active frames in one gpuxtb
ragged-batch call per iteration.  The per-frame stepping mathematics lives here
and is written against the Array API namespace (via ``array_api_compat``)
instead of NumPy directly, so the same kernels can be reused later with any
Array API backend that supplies gradients.  gpuxtb backends currently return
NumPy arrays, so callers pass plain ``numpy.ndarray`` objects; the code never
mutates its inputs and returns fresh arrays in the input namespace.

Only elementwise operations and reductions are used, so every function is also
valid on device-resident arrays (PyTorch, CuPy, JAX) that implement the Array
API standard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._array import _compat

if TYPE_CHECKING:
    import numpy as np

#: Guard against dividing by an exactly-zero curvature or gradient norm.
_EPS = 1.0e-30


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


__all__ = ["curvature", "initial_step_size", "lbfgs_direction"]
