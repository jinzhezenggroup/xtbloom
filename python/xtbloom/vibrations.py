"""Vibrational analysis for numerical xTBloom Cartesian Hessians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import XTBloomValueError

if TYPE_CHECKING:
    from .interface import Calculator

# sqrt(Eh / (bohr^2 * unified-atomic-mass-unit)) / (2*pi*c), in cm^-1.
_HESSIAN_AMU_TO_WAVENUMBER = 5140.487143715828


@dataclass(frozen=True)
class VibrationalResult:
    """Normal modes derived from a Cartesian energy Hessian."""

    frequencies_cm1: np.ndarray
    eigenvalues: np.ndarray
    modes: np.ndarray
    mass_weighted_modes: np.ndarray
    rigid_rank: int

    @property
    def imaginary(self) -> np.ndarray:
        """Boolean mask for imaginary modes (reported as negative frequencies)."""
        return self.frequencies_cm1 < 0.0


def _validated_inputs(
    hessian: object, positions: object, masses: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(hessian, dtype=np.float64)
    r = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] == 0:
        raise XTBloomValueError("positions must have shape (natoms, 3) with natoms > 0")
    natoms = r.shape[0]
    if h.shape != (3 * natoms, 3 * natoms):
        raise XTBloomValueError(f"hessian must have shape ({3 * natoms}, {3 * natoms})")
    if m.shape != (natoms,):
        raise XTBloomValueError(f"masses must have shape ({natoms},)")
    if not (np.isfinite(h).all() and np.isfinite(r).all() and np.isfinite(m).all()):
        raise XTBloomValueError("hessian, positions, and masses must be finite")
    if np.any(m <= 0.0):
        raise XTBloomValueError(
            "masses must be positive atomic masses in unified atomic mass units"
        )
    return h, r, m


def _rigid_basis(positions: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an orthonormal mass-weighted translation/rotation basis."""
    natoms = len(masses)
    sqrt_m = np.sqrt(masses)
    center = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    centered = positions - center
    vectors: list[np.ndarray] = []
    for axis in range(3):
        vector = np.zeros((natoms, 3), dtype=np.float64)
        vector[:, axis] = sqrt_m
        vectors.append(vector.reshape(-1))
    for axis in np.eye(3):
        # The sign convention is irrelevant; only the rigid subspace matters.
        vector = sqrt_m[:, None] * np.cross(axis, centered)
        vectors.append(vector.reshape(-1))
    candidates = np.column_stack(vectors)
    u, singular, _ = np.linalg.svd(candidates, full_matrices=True)
    if singular.size == 0:
        return np.empty((3 * natoms, 0), dtype=np.float64), 0
    tolerance = max(candidates.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return u[:, :rank], rank


def analyze_vibrations(
    hessian: object,
    positions: object,
    masses: object,
    *,
    project_rigid: bool = True,
    symmetrize: bool = True,
) -> VibrationalResult:
    """Diagonalize a Cartesian Hessian into molecular vibrational modes.

    Parameters
    ----------
    hessian
        Cartesian energy Hessian in Hartree/bohr^2, shape ``(3N, 3N)``.
    positions
        Cartesian coordinates in bohr, shape ``(N, 3)``.
    masses
        Atomic masses in unified atomic mass units (u), shape ``(N,)``.
    project_rigid
        Remove the numerically detected mass-weighted translation/rotation
        subspace before diagonalization. Rank detection naturally gives three
        rigid directions for one atom, five for a linear molecule, and six for
        a non-linear molecule.
    symmetrize
        Diagonalize ``0.5 * (H + H.T)``. This is recommended for xTBloom's
        finite-difference Hessian while the raw Hessian remains available to
        the caller for antisymmetry diagnostics.

    Returns
    -------
    VibrationalResult
        Frequencies are signed cm^-1: negative values denote imaginary modes.
        ``modes`` contains unit-norm Cartesian displacement vectors with shape
        ``(nmodes, N, 3)``; ``mass_weighted_modes`` contains the corresponding
        orthonormal mass-weighted eigenvectors.
    """
    h, positions_array, masses_array = _validated_inputs(hessian, positions, masses)
    if symmetrize:
        h = 0.5 * (h + h.T)
    inv_sqrt_mass = np.repeat(1.0 / np.sqrt(masses_array), 3)
    mass_weighted = h * inv_sqrt_mass[:, None] * inv_sqrt_mass[None, :]

    if project_rigid:
        rigid, rigid_rank = _rigid_basis(positions_array, masses_array)
        if rigid_rank:
            # Complete SVD supplies a numerically orthonormal complement without
            # forming I - Q Q^T and filtering artificial near-zero modes later.
            complete, _, _ = np.linalg.svd(rigid, full_matrices=True)
            vibrational_basis = complete[:, rigid_rank:]
        else:
            vibrational_basis = np.eye(3 * len(masses_array), dtype=np.float64)
    else:
        rigid_rank = 0
        vibrational_basis = np.eye(3 * len(masses_array), dtype=np.float64)

    if vibrational_basis.shape[1] == 0:
        empty_modes = np.empty((0, len(masses_array), 3), dtype=np.float64)
        return VibrationalResult(
            frequencies_cm1=np.empty(0, dtype=np.float64),
            eigenvalues=np.empty(0, dtype=np.float64),
            modes=empty_modes.copy(),
            mass_weighted_modes=empty_modes,
            rigid_rank=rigid_rank,
        )

    reduced = vibrational_basis.T @ mass_weighted @ vibrational_basis
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_modes = np.linalg.eigh(reduced)
    mass_weighted_columns = vibrational_basis @ reduced_modes

    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    frequencies *= _HESSIAN_AMU_TO_WAVENUMBER

    cartesian_columns = mass_weighted_columns * inv_sqrt_mass[:, None]
    norms = np.linalg.norm(cartesian_columns, axis=0)
    nonzero = norms > 0.0
    cartesian_columns[:, nonzero] /= norms[nonzero]

    return VibrationalResult(
        frequencies_cm1=np.ascontiguousarray(frequencies),
        eigenvalues=np.ascontiguousarray(eigenvalues),
        modes=np.ascontiguousarray(
            cartesian_columns.T.reshape(-1, len(masses_array), 3)
        ),
        mass_weighted_modes=np.ascontiguousarray(
            mass_weighted_columns.T.reshape(-1, len(masses_array), 3)
        ),
        rigid_rank=rigid_rank,
    )


def vibrations(
    calculator: Calculator,
    masses: object,
    *,
    step: float = 5.0e-3,
    auto_batch_size: bool | int | None = True,
    project_rigid: bool = True,
) -> VibrationalResult:
    """Compute a numerical xTBloom Hessian and analyze its vibrational modes."""
    hessian = calculator.hessian(
        step=step,
        symmetrize=False,
        auto_batch_size=auto_batch_size,
    )
    return analyze_vibrations(
        hessian,
        calculator.positions,
        masses,
        project_rigid=project_rigid,
        symmetrize=True,
    )


__all__ = ["VibrationalResult", "analyze_vibrations", "vibrations"]
