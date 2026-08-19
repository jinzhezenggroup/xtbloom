"""Tests for Python-layer vibrational analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from xtbloom import Calculator, analyze_vibrations, vibrations
from xtbloom.exceptions import XTBloomValueError


def _codata_2018_hessian_factor() -> float:
    """Independently derive the Eh/(bohr^2 u) to cm^-1 conversion."""
    hartree_joule = 4.3597447222071e-18
    bohr_metre = 5.29177210903e-11
    unified_atomic_mass_kg = 1.66053906660e-27
    speed_of_light_metre_per_second = 299792458.0
    angular_frequency = math.sqrt(
        hartree_joule / (bohr_metre**2 * unified_atomic_mass_kg)
    )
    return angular_frequency / (2.0 * math.pi * speed_of_light_metre_per_second * 100.0)


def _independent_nonlinear_bases() -> tuple[np.ndarray, np.ndarray]:
    """Return hand-derived rigid and internal bases for a right-triangle molecule."""
    # Coordinates are flattened as (x1, y1, z1, ...). The first six columns
    # are three translations and rotations about the center of mass; the three
    # internal columns independently solve zero translation and rotation.
    rigid_candidates = np.column_stack(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0, 2.0, 0.0, -2.0, -1.0, 0.0],
        ]
    )
    internal_candidates = np.column_stack(
        [
            [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0],
        ]
    )
    rigid, _ = np.linalg.qr(rigid_candidates)
    vibrational, _ = np.linalg.qr(internal_candidates)
    return rigid, vibrational


def _hessian_with_vibrational_eigenvalues(
    vibrational: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    """Construct a Cartesian Hessian from an independent internal basis."""
    mass_weighted = vibrational @ np.diag(eigenvalues) @ vibrational.T
    return np.ascontiguousarray(mass_weighted)


def test_nonlinear_projection_recovers_signed_spectrum() -> None:
    """Recover an independently constructed nonlinear signed spectrum."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    masses = np.ones(3)
    eigenvalues = np.array([-1.0e-4, 2.0e-4, 3.0e-4])
    rigid, vibrational = _independent_nonlinear_bases()
    hessian = _hessian_with_vibrational_eigenvalues(vibrational, eigenvalues)

    result = analyze_vibrations(hessian, positions, masses)

    assert result.rigid_rank == 6
    np.testing.assert_allclose(result.eigenvalues, eigenvalues, atol=2.0e-18, rtol=0.0)
    expected = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    expected *= _codata_2018_hessian_factor()
    np.testing.assert_allclose(result.frequencies_cm1, expected, atol=3.0e-11, rtol=0.0)
    np.testing.assert_array_equal(result.imaginary, [True, False, False])
    assert result.modes.shape == (3, 3, 3)
    assert result.mass_weighted_modes.shape == (3, 3, 3)
    mass_weighted_modes = result.mass_weighted_modes.reshape(3, -1)
    np.testing.assert_allclose(
        np.linalg.norm(result.modes.reshape(3, -1), axis=1),
        1.0,
        atol=2.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(hessian @ rigid, 0.0, atol=2.0e-19, rtol=0.0)
    np.testing.assert_allclose(mass_weighted_modes @ rigid, 0.0, atol=2.0e-15, rtol=0.0)
    np.testing.assert_allclose(
        np.abs(mass_weighted_modes @ vibrational),
        np.eye(3),
        atol=2.0e-15,
        rtol=0.0,
    )


def test_mass_weighting_and_frequency_conversion_match_codata_2018() -> None:
    """Check mass division and wavenumbers against independent SI constants."""
    result = analyze_vibrations(
        np.diag([4.0, 9.0, 16.0]),
        np.zeros((1, 3)),
        np.array([4.0]),
        project_rigid=False,
    )
    np.testing.assert_array_equal(result.eigenvalues, [1.0, 2.25, 4.0])
    factor = _codata_2018_hessian_factor()
    np.testing.assert_allclose(
        result.frequencies_cm1,
        [factor, 1.5 * factor, 2.0 * factor],
        atol=3.0e-10,
        rtol=0.0,
    )


def test_analysis_always_symmetrizes_an_asymmetric_input() -> None:
    """Diagonalize the symmetric part while leaving the caller input unchanged."""
    hessian = np.array([[4.0, 2.0, 0.0], [0.0, 9.0, 3.0], [0.0, 1.0, 16.0]])
    original = hessian.copy()
    result = analyze_vibrations(
        hessian,
        np.zeros((1, 3)),
        np.ones(1),
        project_rigid=False,
    )
    np.testing.assert_array_equal(hessian, original)
    np.testing.assert_allclose(
        result.eigenvalues,
        np.linalg.eigvalsh(0.5 * (original + original.T)),
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


def test_vibrations_runs_real_cpu_calculator_hessian() -> None:
    """Exercise the complete public wrapper through native CPU forces."""
    positions = np.array([[-0.71, 0.0, 0.0], [0.71, 0.0, 0.0]])
    with Calculator("GFN2-xTB", [1, 1], positions, backend="cpu") as calculator:
        result = vibrations(
            calculator,
            [1.00784, 1.00784],
            step=0.005,
            auto_batch_size=2,
        )

    assert result.rigid_rank == 5
    assert result.frequencies_cm1.shape == (1,)
    assert result.modes.shape == (1, 2, 3)
    assert result.mass_weighted_modes.shape == (1, 2, 3)
    assert np.isfinite(result.frequencies_cm1).all()
    assert result.frequencies_cm1[0] > 0.0
