"""Tests for the Array-API L-BFGS stepping kernel (:mod:`xtbloom._optimizer`)."""

from __future__ import annotations

import numpy as np
import pytest
from xtbloom._optimizer import curvature, initial_step_size, lbfgs_direction


def test_lbfgs_direction_with_empty_history_is_steepest_descent() -> None:
    """An empty history must reduce to ``-g``."""
    gradient = np.array([0.3, -1.2, 0.0], dtype=np.float64)
    direction = lbfgs_direction(gradient, [], [], [])
    assert np.array_equal(direction, -gradient)


def test_lbfgs_direction_api_consistency() -> None:
    """A populated history must produce an array in the input namespace."""
    gradient = np.array([0.2, 0.5], dtype=np.float64)
    s = [np.array([0.1, 0.0]), np.array([0.0, 0.2])]
    y = [np.array([0.05, 0.0]), np.array([0.0, 0.1])]
    rho = [float(1.0 / curvature(s_, y_)) for s_, y_ in zip(s, y, strict=True)]
    direction = lbfgs_direction(gradient, s, y, rho)
    assert isinstance(direction, np.ndarray)
    assert direction.shape == gradient.shape
    assert bool(np.isfinite(direction).all())


def test_lbfgs_exact_quadratic_convergence() -> None:
    """L-BFGS converges an SPD quadratic in at most n exact-line-search steps."""
    rng = np.random.default_rng(7)
    size = 3
    a = rng.normal(size=size)
    x0 = rng.normal(size=size) * 4.0
    # SPD matrix A = B^T B + I.
    raw = rng.normal(size=(size, size))
    matrix = raw.T @ raw + np.eye(size)

    x = x0
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    rho_history: list[float] = []
    converged = False
    for _ in range(size + 2):
        gradient = matrix @ (x - a)
        direction = lbfgs_direction(gradient, s_history, y_history, rho_history)
        a_d = matrix @ direction
        step = -float(direction @ gradient) / float(direction @ a_d)
        x_prev, g_prev = x, gradient
        x = x + step * direction
        s = x - x_prev
        y = matrix @ (x - a) - g_prev
        curv = curvature(s, y)
        if curv > 0.0:
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / curv)
        if np.linalg.norm(x - a) < 1e-10:
            converged = True
            break
    assert converged, (
        f"L-BFGS did not converge; final error {np.linalg.norm(x - a):.3e}"
    )


def test_initial_step_size_positive_and_bounded() -> None:
    """The first-step length is finite, positive, and clamped by ``maximum``."""
    assert initial_step_size(np.array([1.0, 1.0])) == pytest.approx(0.1)  # 0.1/1
    tiny = np.array([1e-12, 1e-12])
    assert initial_step_size(tiny) == 0.5  # clamped by maximum
    strong = np.array([10.0, 10.0])
    assert initial_step_size(strong) == pytest.approx(0.01)


def test_curvature_sign_and_value() -> None:
    """Curvature is the plain ``s . y`` dot product."""
    s = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    assert curvature(s, y) == pytest.approx(11.0)
