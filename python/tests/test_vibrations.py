"""Tests for Python-layer vibrational analysis."""

from __future__ import annotations

import numpy as np
import pytest
from xtbloom import analyze_vibrations, vibrations
from xtbloom.exceptions import XTBloomValueError
from xtbloom.vibrations import _HESSIAN_AMU_TO_WAVENUMBER, _rigid_basis


def _hessian_with_vibrational_eigenvalues(
    positions: np.ndarray, masses: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    """Construct a Cartesian Hessian with a prescribed rigid-free spectrum."""
    rigid, rank = _rigid_basis(positions, masses)
    complete, _, _ = np.linalg.svd(rigid, full_matrices=True)
    vibrational = complete[:, rank:]
    assert vibrational.shape[1] == len(eigenvalues)
    mass_weighted = vibrational @ np.diag(eigenvalues) @ vibrational.T
    sqrt_mass = np.repeat(np.sqrt(masses), 3)
    return mass_weighted * sqrt_mass[:, None] * sqrt_mass[None, :]


def test_nonlinear_projection_recovers_signed_spectrum() -> None:
    """Recover real and imaginary modes after nonlinear rigid projection."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.4, 1.3, 0.0]],
        dtype=np.float64,
    )
    masses = np.array([15.999, 1.008, 1.008])
    eigenvalues = np.array([-1.0e-4, 2.0e-4, 3.0e-4])
    hessian = _hessian_with_vibrational_eigenvalues(positions, masses, eigenvalues)

    result = analyze_vibrations(hessian, positions, masses)

    assert result.rigid_rank == 6
    np.testing.assert_allclose(result.eigenvalues, eigenvalues, atol=2.0e-18, rtol=0.0)
    expected = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    expected *= _HESSIAN_AMU_TO_WAVENUMBER
    np.testing.assert_allclose(result.frequencies_cm1, expected, atol=2.0e-11, rtol=0.0)
    np.testing.assert_array_equal(result.imaginary, [True, False, False])
    assert result.modes.shape == (3, 3, 3)
    assert result.mass_weighted_modes.shape == (3, 3, 3)
    np.testing.assert_allclose(
        np.linalg.norm(result.modes.reshape(3, -1), axis=1),
        1.0,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_linear_and_single_atom_rigid_rank() -> None:
    """Detect five rigid modes for linear H2 and three for one atom."""
    h2_positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    h2_masses = np.array([1.008, 1.008])
    h2 = analyze_vibrations(np.zeros((6, 6)), h2_positions, h2_masses)
    assert h2.rigid_rank == 5
    assert h2.frequencies_cm1.shape == (1,)

    atom = analyze_vibrations(np.zeros((3, 3)), np.zeros((1, 3)), np.array([4.002602]))
    assert atom.rigid_rank == 3
    assert atom.frequencies_cm1.size == 0
    assert atom.modes.shape == (0, 1, 3)


def test_projection_can_be_disabled() -> None:
    """Retain every Cartesian mode when rigid projection is disabled."""
    result = analyze_vibrations(
        np.eye(6),
        np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        project_rigid=False,
    )
    assert result.rigid_rank == 0
    assert result.frequencies_cm1.shape == (6,)


@pytest.mark.parametrize(
    ("hessian", "positions", "masses"),
    [
        (np.eye(3), np.zeros((0, 3)), np.empty(0)),
        (np.eye(2), np.zeros((1, 3)), np.ones(1)),
        (np.eye(3), np.zeros((1, 3)), np.ones(2)),
        (np.eye(3), np.zeros((1, 3)), np.array([0.0])),
        (np.eye(3), np.zeros((1, 3)), np.array([np.nan])),
    ],
)
def test_invalid_inputs_are_rejected(
    hessian: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> None:
    """Reject malformed, nonfinite, and nonpositive vibrational inputs."""
    with pytest.raises(XTBloomValueError):
        analyze_vibrations(hessian, positions, masses)


def test_vibrations_wrapper_preserves_raw_hessian_diagnostic_path() -> None:
    """Request the raw numerical Hessian before vibrational post-processing."""

    class FakeCalculator:
        """Minimal calculator stand-in that records Hessian options."""

        positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])

        def __init__(self) -> None:
            """Initialize the Hessian-call ledger."""
            self.calls: list[dict[str, object]] = []

        def hessian(self, **kwargs: object) -> np.ndarray:
            """Record Hessian options and return a zero Cartesian matrix."""
            self.calls.append(kwargs)
            return np.zeros((6, 6))

    calculator = FakeCalculator()
    result = vibrations(
        calculator,  # type: ignore[arg-type]
        [1.008, 1.008],
        step=0.01,
        auto_batch_size=17,
    )
    assert calculator.calls == [
        {"step": 0.01, "symmetrize": False, "auto_batch_size": 17}
    ]
    assert result.rigid_rank == 5
    assert result.frequencies_cm1.shape == (1,)
