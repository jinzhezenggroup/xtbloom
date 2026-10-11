#!/usr/bin/env python3
"""Prepare deterministic case specifications for issue #343 diagnostics.

The five conformance-corpus specifications are consumed directly. This
generator creates only the additional repository-local diagnostic inputs:
simple analytic rings, a validated all-trans alkane, the committed TMAC/Cl
difficult-SCC fixture, and a deterministic two-geometry trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

BOHR = 1.8897261254578281
_CC_BOND_BOHR = 1.53 * BOHR
_CH_BOND_BOHR = 1.09 * BOHR
_CCC_ANGLE_RAD = np.deg2rad(112.0)
_TETRAHEDRAL_COS = -1.0 / 3.0

FloatArray = NDArray[np.float64]


def write_spec(
    path: Path,
    atomic_numbers: Sequence[int],
    positions_bohr: FloatArray,
    *,
    charge: float = 0.0,
    unpaired: int = 0,
    temperature: float = 300.0,
    mixer_memory: int = 8,
    mixer_damping: float = 0.4,
    maximum_iterations: int = 60,
    point_charges: Sequence[Sequence[float]] | None = None,
) -> None:
    """Write one corpus-style SCC diagnostic specification."""
    lines = [str(len(atomic_numbers))]
    lines.append(" ".join(str(number) for number in atomic_numbers))
    lines.extend(f"{xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}" for xyz in positions_bohr)
    lines.extend(
        [
            f"{charge:.17g}",
            str(unpaired),
            f"{temperature:.17g}",
            str(mixer_memory),
            f"{mixer_damping:.17g}",
            str(maximum_iterations),
        ]
    )
    rows = point_charges or []
    lines.append(str(len(rows)))
    lines.extend(
        f"{row[0]:.17g} {row[1]:.17g} {row[2]:.17g} {row[3]:.17g} {row[4]:.17g}"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    sys.stdout.write(f"wrote {path}: {len(atomic_numbers)} atoms\n")


def planar_ring(*, pyridine: bool) -> tuple[list[int], FloatArray]:
    """Build an analytic regular benzene or pyridine diagnostic geometry."""
    carbon_radius = 1.397 * BOHR
    hydrogen_radius = carbon_radius + 1.09 * BOHR
    ring_positions = [
        np.array(
            [carbon_radius * np.cos(angle), carbon_radius * np.sin(angle), 0.0],
            dtype=np.float64,
        )
        for angle in (index * np.pi / 3.0 for index in range(6))
    ]
    numbers = [7 if pyridine and index == 0 else 6 for index in range(6)]
    positions = [position.copy() for position in ring_positions]
    hydrogen_sites = range(1, 6) if pyridine else range(6)
    for index in hydrogen_sites:
        direction = ring_positions[index] / np.linalg.norm(ring_positions[index])
        positions.append(hydrogen_radius * direction)
        numbers.append(1)
    return numbers, np.asarray(positions, dtype=np.float64)


def _orthogonal_unit(vector: FloatArray) -> FloatArray:
    """Return a deterministic unit vector perpendicular to ``vector``."""
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(vector, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    perpendicular = np.cross(vector, reference)
    return perpendicular / np.linalg.norm(perpendicular)


def trans_planar_alkane(n_carbon: int) -> tuple[list[int], FloatArray]:
    """Build a non-self-intersecting all-trans ``C_n H_(2n+2)`` geometry.

    Carbon bond vectors alternate above and below the chain axis, so their
    x-components are always positive.  Hydrogen directions are constructed
    from tetrahedral dot products instead of a distance-based carbon-neighbor
    search; this prevents folded chains from silently losing hydrogens.
    """
    if n_carbon < 2:
        raise ValueError("an alkane diagnostic requires at least two carbons")

    half_turn = 0.5 * (np.pi - _CCC_ANGLE_RAD)
    carbon_positions = [np.zeros(3, dtype=np.float64)]
    for bond_index in range(n_carbon - 1):
        sign = 1.0 if bond_index % 2 == 0 else -1.0
        step = _CC_BOND_BOHR * np.array(
            [np.cos(half_turn), sign * np.sin(half_turn), 0.0],
            dtype=np.float64,
        )
        carbon_positions.append(carbon_positions[-1] + step)
    carbons = np.asarray(carbon_positions, dtype=np.float64)
    carbons -= np.mean(carbons, axis=0)

    positions = [position.copy() for position in carbons]
    numbers = [6] * n_carbon
    for carbon_index, carbon in enumerate(carbons):
        neighbor_indices = []
        if carbon_index > 0:
            neighbor_indices.append(carbon_index - 1)
        if carbon_index + 1 < n_carbon:
            neighbor_indices.append(carbon_index + 1)
        neighbor_vectors = [
            (carbons[index] - carbon) / np.linalg.norm(carbons[index] - carbon)
            for index in neighbor_indices
        ]

        if len(neighbor_vectors) == 1:
            carbon_axis = neighbor_vectors[0]
            radial_one = _orthogonal_unit(carbon_axis)
            radial_two = np.cross(carbon_axis, radial_one)
            radial_length = np.sqrt(1.0 - _TETRAHEDRAL_COS**2)
            for phase in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
                direction = _TETRAHEDRAL_COS * carbon_axis + radial_length * (
                    np.cos(phase) * radial_one + np.sin(phase) * radial_two
                )
                positions.append(carbon + _CH_BOND_BOHR * direction)
                numbers.append(1)
        else:
            first, second = neighbor_vectors
            bisector = (first + second) / np.linalg.norm(first + second)
            normal = np.cross(first, second)
            normal /= np.linalg.norm(normal)
            bisector_projection = _TETRAHEDRAL_COS / float(np.dot(bisector, first))
            normal_projection = np.sqrt(1.0 - bisector_projection**2)
            for sign in (-1.0, 1.0):
                direction = (
                    bisector_projection * bisector + sign * normal_projection * normal
                )
                positions.append(carbon + _CH_BOND_BOHR * direction)
                numbers.append(1)

    result = np.asarray(positions, dtype=np.float64)
    validate_alkane(numbers, result, n_carbon)
    return numbers, result


def validate_alkane(
    atomic_numbers: Sequence[int], positions_bohr: FloatArray, n_carbon: int
) -> None:
    """Validate formula and nearest-neighbor connectivity of an alkane case."""
    expected_hydrogen = 2 * n_carbon + 2
    if list(atomic_numbers).count(6) != n_carbon:
        raise ValueError(f"expected {n_carbon} carbon atoms")
    if list(atomic_numbers).count(1) != expected_hydrogen:
        raise ValueError(f"expected {expected_hydrogen} hydrogen atoms")
    if positions_bohr.shape != (n_carbon + expected_hydrogen, 3):
        raise ValueError("alkane coordinate shape does not match its formula")

    carbons = positions_bohr[:n_carbon]
    carbon_pairs = {
        (left, right)
        for left in range(n_carbon)
        for right in range(left + 1, n_carbon)
        if np.linalg.norm(carbons[left] - carbons[right]) < 1.8 * BOHR
    }
    expected_pairs = {(index, index + 1) for index in range(n_carbon - 1)}
    if carbon_pairs != expected_pairs:
        raise ValueError("alkane carbon connectivity is not one linear chain")

    for hydrogen in positions_bohr[n_carbon:]:
        attached = np.count_nonzero(
            np.linalg.norm(carbons - hydrogen, axis=1) < 1.25 * BOHR
        )
        if attached != 1:
            raise ValueError("each hydrogen must attach to exactly one carbon")


def _load_tmacl() -> tuple[list[int], FloatArray]:
    """Load the committed TMAC/Cl geometry and convert Angstrom to bohr."""
    xyz = Path(__file__).resolve().parents[2] / "data/conformance/inputs/tmacl.xyz"
    lines = xyz.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    element_numbers = {"C": 6, "H": 1, "N": 7, "O": 8, "F": 9, "Cl": 17}
    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + atom_count]:
        tokens = line.split()
        numbers.append(element_numbers[tokens[0]])
        positions.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    return numbers, np.asarray(positions, dtype=np.float64) * BOHR


def main(arguments: list[str] | None = None) -> int:
    """Generate every repository-local issue #343 diagnostic input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "cases")
    options = parser.parse_args(arguments)
    options.out.mkdir(parents=True, exist_ok=True)

    for label, is_pyridine in (("benzene", False), ("pyridine", True)):
        numbers, positions = planar_ring(pyridine=is_pyridine)
        write_spec(options.out / f"{label}.spec", numbers, positions)

    numbers, positions = trans_planar_alkane(12)
    write_spec(options.out / "dodecane.spec", numbers, positions)

    numbers, positions = _load_tmacl()
    write_spec(
        options.out / "tmacl.spec",
        numbers,
        positions,
        maximum_iterations=100,
    )

    numbers, positions = trans_planar_alkane(12)
    random_generator = np.random.default_rng(343)
    displaced = positions + random_generator.normal(0.0, 0.015, size=positions.shape)
    validate_alkane(numbers, displaced, 12)
    write_spec(options.out / "dodecane_traj1.spec", numbers, positions)
    write_spec(options.out / "dodecane_traj2.spec", numbers, displaced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
