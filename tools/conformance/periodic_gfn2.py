#!/usr/bin/env python3
"""Generate and verify independent native-3D-periodic GFN2 evidence.

Neutral full-model values come from a hash-pinned tblite executable. Charged
uniform-background values are reconstructed analytically because the reviewed
tblite 0.7.0 monopole matrix does not contain the background constant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data/conformance/periodic/manifest.json"
ACCURACY = 1.0e-4
ACCURACY_TEXT = "0.0001"
SCHEMA = "xtbloom-periodic-gfn2-conformance-v2"
GOLDEN_SCHEMA = "xtbloom-periodic-gfn2-golden-v2"
BUILD_SCHEMA = "xtbloom-tblite-build-attestation-v1"
STRAIN_STEPS = (2.0e-4, 1.0e-4, 5.0e-5)
CARTESIAN_STEPS = (2.0e-4, 1.0e-4, 5.0e-5)
STRAIN_MODES = (
    ("xx", 0, 0),
    ("yy", 1, 1),
    ("zz", 2, 2),
    ("xy", 0, 1),
    ("xz", 0, 2),
    ("yz", 1, 2),
)


class PeriodicOracleError(RuntimeError):
    """An actionable periodic-corpus or oracle failure."""


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and retain the path in malformed-input errors."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeriodicOracleError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PeriodicOracleError(f"expected a JSON object in {path}")
    return value


def dump_json(path: Path, value: dict[str, Any]) -> None:
    """Write the canonical, review-friendly representation of generated data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without assuming it fits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Hash one value using the corpus canonical JSON serialization."""
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validated_sha256(value: object, label: str) -> str:
    """Return one canonical lowercase SHA-256 identity."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PeriodicOracleError(f"{label} must be a lowercase SHA-256")
    return value


def repository_path(relative: str) -> Path:
    """Resolve a manifest path while forbidding escape from the repository."""
    candidate = (REPOSITORY_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise PeriodicOracleError(
            f"manifest path escapes repository: {relative}"
        ) from exc
    return candidate


def finite_number(value: object, label: str) -> float:
    """Return a finite binary64-compatible numeric value."""
    if type(value) not in (int, float):
        raise PeriodicOracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PeriodicOracleError(f"{label} must be finite")
    return result


def finite_text_number(value: str, label: str) -> float:
    """Parse one finite number from the textual Turbomole grammar."""
    try:
        result = float(value)
    except ValueError as exc:
        raise PeriodicOracleError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PeriodicOracleError(f"{label} must be finite")
    return result


def finite_vector(value: object, count: int, label: str) -> list[float]:
    """Validate a fixed-length numeric vector."""
    if not isinstance(value, list) or len(value) != count:
        raise PeriodicOracleError(f"{label} must contain {count} values")
    return [
        finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)
    ]


def determinant(cell: list[float]) -> float:
    """Return the determinant of a row-major 3-by-3 matrix."""
    return (
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6])
    )


def parse_turbomole(path: Path) -> dict[str, Any]:
    """Parse the deliberately narrow periodic Turbomole fixture grammar."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    try:
        coord_begin = lines.index("$coord") + 1
        periodic_line = lines.index("$periodic 3")
        lattice_begin = lines.index("$lattice") + 1
        end_line = lines.index("$end")
    except ValueError as exc:
        raise PeriodicOracleError(
            f"{path} must contain $coord, $periodic 3, $lattice, and $end"
        ) from exc
    if not (coord_begin <= periodic_line < lattice_begin <= end_line):
        raise PeriodicOracleError(f"{path} has malformed section ordering")
    coord_lines = [line for line in lines[coord_begin:periodic_line] if line]
    lattice_lines = [line for line in lines[lattice_begin:end_line] if line]
    if not coord_lines or len(lattice_lines) != 3:
        raise PeriodicOracleError(f"{path} must contain atoms and three lattice rows")

    positions: list[float] = []
    symbols: list[str] = []
    for index, line in enumerate(coord_lines):
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r"[A-Za-z]{1,2}", fields[3]):
            raise PeriodicOracleError(f"{path} coordinate row {index} is malformed")
        positions.extend(
            finite_text_number(value, f"{path} coordinate") for value in fields[:3]
        )
        symbols.append(fields[3][0].upper() + fields[3][1:].lower())

    cell: list[float] = []
    for index, line in enumerate(lattice_lines):
        fields = line.split()
        if len(fields) != 3:
            raise PeriodicOracleError(f"{path} lattice row {index} is malformed")
        cell.extend(finite_text_number(value, f"{path} lattice") for value in fields)
    volume = determinant(cell)
    if not (volume > 0.0) or not math.isfinite(volume):
        raise PeriodicOracleError(f"{path} cell must be finite and right-handed")
    return {
        "cell_matrix_row_major_bohr": cell,
        "positions_bohr": positions,
        "symbols": symbols,
        "volume_bohr3": volume,
    }


def format_turbomole(
    structure: dict[str, Any], positions: list[float], cell: list[float]
) -> str:
    """Materialize one affine-deformed periodic fixture for finite differences."""
    lines = ["$coord"]
    for atom, symbol in enumerate(structure["symbols"]):
        xyz = positions[3 * atom : 3 * atom + 3]
        lines.append(f"  {xyz[0]:.17g}  {xyz[1]:.17g}  {xyz[2]:.17g}  {symbol}")
    lines.extend(["$periodic 3", "$lattice"])
    for row in range(3):
        values = cell[3 * row : 3 * row + 3]
        lines.append(f"  {values[0]:.17g}  {values[1]:.17g}  {values[2]:.17g}")
    lines.extend(["$end", ""])
    return "\n".join(lines)


def affine_deformation(
    structure: dict[str, Any], row: int, column: int, step: float
) -> tuple[list[float], list[float]]:
    """Apply r'=(I+eps)r and H'=H(I+eps)^T in row-major storage."""
    deformation = [0.0] * 9
    deformation[0] = deformation[4] = deformation[8] = 1.0
    deformation[3 * row + column] += step

    positions = structure["positions_bohr"]
    deformed_positions: list[float] = []
    for atom in range(len(positions) // 3):
        vector = positions[3 * atom : 3 * atom + 3]
        deformed_positions.extend(
            sum(
                deformation[3 * component + source] * vector[source]
                for source in range(3)
            )
            for component in range(3)
        )

    cell = structure["cell_matrix_row_major_bohr"]
    deformed_cell = [0.0] * 9
    for lattice_row in range(3):
        for cartesian_column in range(3):
            deformed_cell[3 * lattice_row + cartesian_column] = sum(
                cell[3 * lattice_row + source]
                * deformation[3 * cartesian_column + source]
                for source in range(3)
            )
    return deformed_positions, deformed_cell


def cartesian_displacement(
    structure: dict[str, Any], atom: int, axis: int, step: float
) -> tuple[list[float], list[float]]:
    """Displace one Cartesian coordinate while retaining the direct cell."""
    positions = list(structure["positions_bohr"])
    positions[3 * atom + axis] += step
    return positions, list(structure["cell_matrix_row_major_bohr"])


def invariant_variant(
    structure: dict[str, Any], name: str
) -> tuple[list[float], list[float], str]:
    """Materialize one exact periodic symmetry transform."""
    positions = list(structure["positions_bohr"])
    cell = list(structure["cell_matrix_row_major_bohr"])
    if name == "translation":
        shift = (0.37, -0.23, 0.19)
        for atom in range(len(positions) // 3):
            for axis in range(3):
                positions[3 * atom + axis] += shift[axis]
        return positions, cell, "all positions shifted by (0.37,-0.23,0.19) bohr"
    if name == "cell_basis":
        # U=[[1,1,0],[0,1,0],[0,0,1]] is unimodular with det(U)=+1,
        # so replacing H by U*H preserves the complete Bravais lattice.
        transformed = list(cell)
        for axis in range(3):
            transformed[axis] = cell[axis] + cell[3 + axis]
        return (
            positions,
            transformed,
            "unimodular direct basis U=[[1,1,0],[0,1,0],[0,0,1]]",
        )
    raise PeriodicOracleError(f"unknown periodic invariant variant: {name}")


def wrapping_probe(structure: dict[str, Any]) -> tuple[list[float], list[float], str]:
    """Build outside-cell coordinates and wrap them into the half-open cell."""
    cell = structure["cell_matrix_row_major_bohr"]
    positions = list(structure["positions_bohr"])
    translations = ((1, 0, 0), (0, -1, 0), (0, 0, 1), (-1, 1, 0))
    for atom in range(len(positions) // 3):
        coefficients = translations[atom % len(translations)]
        for axis in range(3):
            positions[3 * atom + axis] += sum(
                coefficients[row] * cell[3 * row + axis] for row in range(3)
            )
    volume = determinant(cell)
    inverse = [
        (cell[4] * cell[8] - cell[5] * cell[7]) / volume,
        (cell[2] * cell[7] - cell[1] * cell[8]) / volume,
        (cell[1] * cell[5] - cell[2] * cell[4]) / volume,
        (cell[5] * cell[6] - cell[3] * cell[8]) / volume,
        (cell[0] * cell[8] - cell[2] * cell[6]) / volume,
        (cell[2] * cell[3] - cell[0] * cell[5]) / volume,
        (cell[3] * cell[7] - cell[4] * cell[6]) / volume,
        (cell[1] * cell[6] - cell[0] * cell[7]) / volume,
        (cell[0] * cell[4] - cell[1] * cell[3]) / volume,
    ]
    wrapped: list[float] = []
    for atom in range(len(positions) // 3):
        vector = positions[3 * atom : 3 * atom + 3]
        fractional = [
            sum(vector[source] * inverse[3 * source + axis] for source in range(3))
            for axis in range(3)
        ]
        fractional = [value - math.floor(value) for value in fractional]
        wrapped.extend(
            sum(fractional[row] * cell[3 * row + axis] for row in range(3))
            for axis in range(3)
        )
    return (
        positions,
        wrapped,
        "u=r*H^-1; u_wrapped=u-floor(u); r_wrapped=u_wrapped*H",
    )


def reciprocal_identity_error(cell: list[float]) -> float:
    """Return max|H B^T-2*pi*I| for B=2*pi*H^-T."""
    volume = determinant(cell)
    inverse = [
        (cell[4] * cell[8] - cell[5] * cell[7]) / volume,
        (cell[2] * cell[7] - cell[1] * cell[8]) / volume,
        (cell[1] * cell[5] - cell[2] * cell[4]) / volume,
        (cell[5] * cell[6] - cell[3] * cell[8]) / volume,
        (cell[0] * cell[8] - cell[2] * cell[6]) / volume,
        (cell[2] * cell[3] - cell[0] * cell[5]) / volume,
        (cell[3] * cell[7] - cell[4] * cell[6]) / volume,
        (cell[1] * cell[6] - cell[0] * cell[7]) / volume,
        (cell[0] * cell[4] - cell[1] * cell[3]) / volume,
    ]
    reciprocal = [
        2.0 * math.pi * inverse[3 * column + row]
        for row in range(3)
        for column in range(3)
    ]
    return max(
        abs(
            sum(
                cell[3 * row + axis] * reciprocal[3 * column + axis]
                for axis in range(3)
            )
            - (2.0 * math.pi if row == column else 0.0)
        )
        for row in range(3)
        for column in range(3)
    )


def reciprocal_cell(cell: list[float]) -> list[float]:
    """Return row-major B=2*pi*H^-T for a row-major direct cell H."""
    volume = determinant(cell)
    inverse = [
        (cell[4] * cell[8] - cell[5] * cell[7]) / volume,
        (cell[2] * cell[7] - cell[1] * cell[8]) / volume,
        (cell[1] * cell[5] - cell[2] * cell[4]) / volume,
        (cell[5] * cell[6] - cell[3] * cell[8]) / volume,
        (cell[0] * cell[8] - cell[2] * cell[6]) / volume,
        (cell[2] * cell[3] - cell[0] * cell[5]) / volume,
        (cell[3] * cell[7] - cell[4] * cell[6]) / volume,
        (cell[1] * cell[6] - cell[0] * cell[7]) / volume,
        (cell[0] * cell[4] - cell[1] * cell[3]) / volume,
    ]
    return [
        2.0 * math.pi * inverse[3 * column + row]
        for row in range(3)
        for column in range(3)
    ]


def vector_norm(vector: list[float]) -> float:
    """Return the Euclidean norm of one three-vector."""
    return math.sqrt(sum(value * value for value in vector))


def ewald_decay(
    space: str, multipole: bool, distance: float, alpha: float, volume: float
) -> float:
    """Evaluate the reviewed tblite Ewald cutoff/search envelope."""
    if space == "direct":
        argument = alpha * distance
        if multipole:
            return (
                math.erfc(argument)
                + 2.0 / math.sqrt(math.pi) * argument * math.exp(-argument * argument)
            ) / (distance**3)
        return math.erfc(argument) / distance
    exponent = math.exp(-0.25 * distance * distance / (alpha * alpha))
    if multipole:
        return 4.0 * math.pi * exponent / volume
    return 4.0 * math.pi * exponent / (volume * distance * distance)


def select_ewald_alpha(cell: list[float], multipole: bool) -> float:
    """Reproduce tblite 133f91e's bracketed alpha search."""
    volume = determinant(cell)
    reciprocal = reciprocal_cell(cell)
    direct_min = min(vector_norm(cell[3 * row : 3 * row + 3]) for row in range(3))
    reciprocal_min = min(
        vector_norm(reciprocal[3 * row : 3 * row + 3]) for row in range(3)
    )
    tolerance = math.sqrt(sys.float_info.epsilon)

    def difference(alpha: float) -> float:
        return (
            ewald_decay("reciprocal", multipole, 4.0 * reciprocal_min, alpha, volume)
            - ewald_decay("reciprocal", multipole, 5.0 * reciprocal_min, alpha, volume)
            - ewald_decay("direct", multipole, 2.0 * direct_min, alpha, volume)
            + ewald_decay("direct", multipole, 3.0 * direct_min, alpha, volume)
        )

    alpha_initial = tolerance
    alpha = alpha_initial
    diff = difference(alpha)
    while diff < -tolerance and math.isfinite(alpha):
        alpha *= 2.0
        diff = difference(alpha)
    if not math.isfinite(alpha) or alpha == alpha_initial:
        return 0.25
    left = 0.5 * alpha
    while diff < tolerance and math.isfinite(alpha):
        alpha *= 2.0
        diff = difference(alpha)
    if not math.isfinite(alpha):
        return 0.25
    right = alpha
    alpha = 0.5 * (left + right)
    diff = difference(alpha)
    iterations = 0
    while abs(diff) > tolerance and iterations <= 30:
        if diff < 0.0:
            left = alpha
        else:
            right = alpha
        alpha = 0.5 * (left + right)
        diff = difference(alpha)
        iterations += 1
    return 0.25 if iterations > 30 else alpha


def search_ewald_cutoff(
    space: str, multipole: bool, alpha: float, volume: float, convergence: float
) -> float:
    """Reproduce the reviewed doubling/bisection cutoff search."""
    cutoff = math.sqrt(sys.float_info.epsilon)
    value = ewald_decay(space, multipole, cutoff, alpha, volume)
    while value > convergence and math.isfinite(cutoff):
        cutoff *= 2.0
        value = ewald_decay(space, multipole, cutoff, alpha, volume)
    left = 0.5 * cutoff
    left_value = ewald_decay(space, multipole, left, alpha, volume)
    right = cutoff
    right_value = value
    for _ in range(30):
        if left_value - right_value <= convergence:
            break
        cutoff = 0.5 * (left + right)
        value = ewald_decay(space, multipole, cutoff, alpha, volume)
        if value >= convergence:
            left = cutoff
            left_value = value
        else:
            right = cutoff
            right_value = value
    return cutoff


def cross_product(lhs: list[float], rhs: list[float]) -> list[float]:
    """Return one Cartesian cross product."""
    return [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]


def lattice_repeats(cell: list[float], cutoff: float) -> list[int]:
    """Return tblite-compatible inclusive rectangular lattice repeat counts."""
    rows = [cell[3 * row : 3 * row + 3] for row in range(3)]
    repeats: list[int] = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        normal = cross_product(rows[other[0]], rows[other[1]])
        normal_length = vector_norm(normal)
        plane_spacing = abs(
            sum(normal[component] * rows[axis][component] for component in range(3))
            / normal_length
        )
        repeats.append(math.ceil(abs(cutoff / plane_spacing)))
    return repeats


def lattice_vectors(cell: list[float], repeats: list[int]) -> list[list[float]]:
    """Enumerate the complete inclusive rectangular translation box."""
    rows = [cell[3 * row : 3 * row + 3] for row in range(3)]
    vectors: list[list[float]] = []
    for first in range(-repeats[0], repeats[0] + 1):
        for second in range(-repeats[1], repeats[1] + 1):
            for third in range(-repeats[2], repeats[2] + 1):
                coefficients = (first, second, third)
                vectors.append(
                    [
                        sum(coefficients[row] * rows[row][axis] for row in range(3))
                        for axis in range(3)
                    ]
                )
    return vectors


def charged_ewald_components(
    cell: list[float], charge: float, alpha: float, margin: int = 0
) -> dict[str, Any]:
    """Independently reconstruct one-charge 3D Ewald component energies."""
    volume = determinant(cell)
    convergence = sys.float_info.epsilon
    direct_cutoff = search_ewald_cutoff("direct", False, alpha, volume, convergence)
    reciprocal_cutoff = search_ewald_cutoff(
        "reciprocal", False, alpha, volume, convergence
    )
    direct_repeats = [value + margin for value in lattice_repeats(cell, direct_cutoff)]
    reciprocal = reciprocal_cell(cell)
    reciprocal_repeats = [
        value + margin for value in lattice_repeats(reciprocal, reciprocal_cutoff)
    ]
    direct_kernel = sum(
        math.erfc(alpha * distance) / distance
        for vector in lattice_vectors(cell, direct_repeats)
        if (distance := vector_norm(vector)) > math.sqrt(sys.float_info.epsilon)
    )
    reciprocal_kernel = sum(
        4.0
        * math.pi
        / volume
        * math.exp(-distance * distance / (4.0 * alpha * alpha))
        / (distance * distance)
        for vector in lattice_vectors(reciprocal, reciprocal_repeats)
        if (distance := vector_norm(vector)) > math.sqrt(sys.float_info.epsilon)
    )
    scale = 0.5 * charge * charge
    components = {
        "alpha_bohr_inverse": alpha,
        "background_energy_hartree": scale * (-math.pi / (alpha * alpha * volume)),
        "direct_cutoff_bohr": direct_cutoff,
        "direct_energy_hartree": scale * direct_kernel,
        "direct_repeats": direct_repeats,
        "reciprocal_cutoff_bohr_inverse": reciprocal_cutoff,
        "reciprocal_energy_hartree": scale * reciprocal_kernel,
        "reciprocal_repeats": reciprocal_repeats,
        "self_energy_hartree": scale * (-2.0 * alpha / math.sqrt(math.pi)),
    }
    components["total_energy_hartree"] = sum(
        components[name]
        for name in (
            "direct_energy_hartree",
            "reciprocal_energy_hartree",
            "self_energy_hartree",
            "background_energy_hartree",
        )
    )
    components["unbackgrounded_energy_hartree"] = (
        components["total_energy_hartree"] - components["background_energy_hartree"]
    )
    components["potential_hartree_per_e"] = (
        2.0 * components["total_energy_hartree"] / charge
    )
    return components


def ewald_contract_values(cell: list[float]) -> dict[str, Any]:
    """Evaluate both reviewed alpha/cutoff policies for one direct cell."""
    volume = determinant(cell)
    monopole_alpha = select_ewald_alpha(cell, False)
    multipole_alpha = select_ewald_alpha(cell, True)
    monopole_direct = search_ewald_cutoff(
        "direct", False, monopole_alpha, volume, sys.float_info.epsilon
    )
    monopole_reciprocal = search_ewald_cutoff(
        "reciprocal", False, monopole_alpha, volume, sys.float_info.epsilon
    )
    multipole_reciprocal = search_ewald_cutoff(
        "reciprocal",
        True,
        multipole_alpha,
        volume,
        100.0 * math.sqrt(sys.float_info.epsilon),
    )
    reciprocal = reciprocal_cell(cell)
    return {
        "monopole": {
            "alpha_bohr_inverse": monopole_alpha,
            "direct_cutoff_bohr": monopole_direct,
            "direct_repeats": lattice_repeats(cell, monopole_direct),
            "reciprocal_cutoff_bohr_inverse": monopole_reciprocal,
            "reciprocal_repeats": lattice_repeats(reciprocal, monopole_reciprocal),
        },
        "multipole": {
            "alpha_bohr_inverse": multipole_alpha,
            "direct_cutoff_bohr": 100.0,
            "direct_repeats": lattice_repeats(cell, 100.0),
            "reciprocal_cutoff_bohr_inverse": multipole_reciprocal,
            "reciprocal_repeats": lattice_repeats(reciprocal, multipole_reciprocal),
        },
    }


def check_ewald_numerics_contract(manifest: dict[str, Any]) -> None:
    """Freeze the exact binary64 alpha and cutoff selection algorithm."""
    contract = manifest.get("ewald_numerics_contract")
    if not isinstance(contract, dict):
        raise PeriodicOracleError("Ewald numerics contract must be an object")
    if contract.get("constants") != {
        "alpha_fallback_bohr_inverse": 0.25,
        "alpha_initial": math.sqrt(sys.float_info.epsilon),
        "alpha_max_bisections": 30,
        "alpha_tolerance": math.sqrt(sys.float_info.epsilon),
        "binary64_epsilon": sys.float_info.epsilon,
        "monopole_convergence": sys.float_info.epsilon,
        "multipole_direct_cutoff_bohr": 100.0,
        "multipole_reciprocal_convergence": 100.0 * math.sqrt(sys.float_info.epsilon),
    }:
        raise PeriodicOracleError("Ewald binary64 constants drifted")
    if contract.get("algorithm") != {
        "alpha": (
            "balance [R(4*g_min)-R(5*g_min)]-[D(2*h_min)-D(3*h_min)]; "
            "double from sqrt(epsilon), bracket, then at most 30 bisections"
        ),
        "cutoff": (
            "double from sqrt(epsilon), then at most 30 bisections; enumerate "
            "the complete inclusive rectangular repeat box"
        ),
        "derivatives": "hold alpha fixed; exclude d(alpha)/d(epsilon)",
    }:
        raise PeriodicOracleError("Ewald algorithm description drifted")
    cases = contract.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise PeriodicOracleError(
            "Ewald numerics contract needs orthogonal and skew cells"
        )
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise PeriodicOracleError("Ewald numerics case is malformed")
        cell = finite_vector(case.get("cell_matrix_row_major_bohr"), 9, "Ewald cell")
        expected = ewald_contract_values(cell)
        if case.get("expected") != expected:
            raise PeriodicOracleError(f"{case['id']} Ewald alpha/cutoff values drifted")


def build_ewald_reconstruction(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the independent multi-alpha charged Ewald reconstruction."""
    identity = manifest.get("ewald_reconstruction")
    if not isinstance(identity, dict):
        raise PeriodicOracleError("Ewald reconstruction identity must be an object")
    cell = finite_vector(identity.get("cell_matrix_row_major_bohr"), 9, "Ewald cell")
    charge = finite_number(identity.get("charge_e"), "Ewald reconstruction charge")
    alphas = finite_vector(identity.get("alphas_bohr_inverse"), 4, "Ewald alphas")
    transformed_cell = list(cell)
    for axis in range(3):
        transformed_cell[axis] = cell[axis] + cell[3 + axis]
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        base = charged_ewald_components(cell, charge, alpha)
        expanded = charged_ewald_components(cell, charge, alpha, margin=1)
        transformed = charged_ewald_components(
            transformed_cell, charge, alpha, margin=1
        )
        rows.append(
            {
                "base": base,
                "expanded_box_total_energy_hartree": expanded["total_energy_hartree"],
                "unimodular_basis_total_energy_hartree": transformed[
                    "total_energy_hartree"
                ],
            }
        )
    return {
        "alphas": rows,
        "cell_matrix_row_major_bohr": cell,
        "charge_e": charge,
        "force_hartree_per_bohr": [0.0, 0.0, 0.0],
        "schema": "xtbloom-periodic-ewald-reconstruction-v1",
    }


def generate_ewald_reconstruction(manifest_path: Path, output: Path) -> None:
    """Write the canonical independent charged-Ewald reconstruction."""
    document = build_ewald_reconstruction(load_json(manifest_path))
    dump_json(output, document)
    print(f"generated {output}")  # noqa: T201 - CLI progress


def check_numeric_document(
    committed: object, expected: object, tolerance: float, label: str
) -> None:
    """Compare a structured analytic reconstruction across Python/libm builds."""
    if isinstance(committed, bool) or isinstance(expected, bool):
        if committed is not expected:
            raise PeriodicOracleError(f"{label} numerical content drifted")
        return
    if isinstance(committed, (int, float)) and isinstance(expected, (int, float)):
        if not math.isfinite(float(committed)) or abs(committed - expected) > tolerance:
            raise PeriodicOracleError(f"{label} numerical content drifted")
        return
    if isinstance(committed, dict) and isinstance(expected, dict):
        if set(committed) != set(expected):
            raise PeriodicOracleError(f"{label} numerical content drifted")
        for key in committed:
            check_numeric_document(
                committed[key], expected[key], tolerance, f"{label}.{key}"
            )
        return
    if isinstance(committed, list) and isinstance(expected, list):
        if len(committed) != len(expected):
            raise PeriodicOracleError(f"{label} numerical content drifted")
        for index, (lhs, rhs) in enumerate(zip(committed, expected, strict=True)):
            check_numeric_document(lhs, rhs, tolerance, f"{label}[{index}]")
        return
    if committed != expected:
        raise PeriodicOracleError(f"{label} numerical content drifted")


def check_ewald_reconstruction(manifest: dict[str, Any]) -> None:
    """Recompute and gate multi-alpha charged Ewald convergence and invariance."""
    identity = manifest.get("ewald_reconstruction")
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
        raise PeriodicOracleError("Ewald reconstruction identity is malformed")
    path = repository_path(identity["path"])
    if sha256_file(path) != validated_sha256(
        identity.get("sha256"), "Ewald reconstruction SHA-256"
    ):
        raise PeriodicOracleError("Ewald reconstruction file hash mismatch")
    committed = load_json(path)
    expected = build_ewald_reconstruction(manifest)
    recomputation_tolerance = finite_number(
        identity.get("recomputation_tolerance_hartree"),
        "Ewald reconstruction recomputation tolerance",
    )
    check_numeric_document(
        committed, expected, recomputation_tolerance, "Ewald reconstruction"
    )
    tolerance = finite_number(
        identity.get("total_tolerance_hartree"), "Ewald total tolerance"
    )
    totals = [row["base"]["total_energy_hartree"] for row in committed["alphas"]]
    if max(totals) - min(totals) > tolerance:
        raise PeriodicOracleError("charged Ewald total is not alpha invariant")
    unbackgrounded = [
        row["base"]["unbackgrounded_energy_hartree"] for row in committed["alphas"]
    ]
    if max(unbackgrounded) - min(unbackgrounded) < 1.0e-3:
        raise PeriodicOracleError("charged Ewald missing-background diagnostic is weak")
    for component in (
        "direct_energy_hartree",
        "reciprocal_energy_hartree",
        "self_energy_hartree",
        "background_energy_hartree",
    ):
        values = [row["base"][component] for row in committed["alphas"]]
        if max(values) - min(values) < 1.0e-4:
            raise PeriodicOracleError(
                f"charged Ewald {component} does not vary with alpha"
            )
    for row in committed["alphas"]:
        total = row["base"]["total_energy_hartree"]
        if abs(row["expanded_box_total_energy_hartree"] - total) > tolerance:
            raise PeriodicOracleError("charged Ewald enlarged-box convergence drifted")
        if abs(row["unimodular_basis_total_energy_hartree"] - total) > tolerance:
            raise PeriodicOracleError("charged Ewald cell-basis invariance drifted")
        charge = committed["charge_e"]
        expected_potential = 2.0 * total / charge
        if not math.isclose(
            row["base"]["potential_hartree_per_e"],
            expected_potential,
            rel_tol=1.0e-14,
            abs_tol=0.0,
        ):
            raise PeriodicOracleError("charged Ewald potential reconstruction drifted")
    if committed.get("force_hartree_per_bohr") != [0.0, 0.0, 0.0]:
        raise PeriodicOracleError("one-charge Ewald force invariant drifted")


def command_template() -> list[str]:
    """Return the path-independent reviewed tblite CLI command."""
    return [
        "{executable}",
        "run",
        "--method",
        "gfn2",
        "--acc",
        ACCURACY_TEXT,
        "--no-restart",
        "--grad",
        "{gradient}",
        "--json",
        "{json}",
        "[--charge {charge}]",
        "[--spin {unpaired_electrons}]",
        "{input}",
    ]


def tblite_command(
    executable: Path, case: dict[str, Any], input_path: Path, output_path: Path
) -> list[str]:
    """Build the exact single-point invocation for one periodic case."""
    command = [
        str(executable),
        "run",
        "--method",
        "gfn2",
        "--acc",
        ACCURACY_TEXT,
        "--no-restart",
        "--grad",
        str(output_path.with_suffix(".gradient")),
        "--json",
        str(output_path),
    ]
    charge = int(case["molecular_charge"])
    if charge:
        command.extend(["--charge", str(charge)])
    unpaired = int(case["unpaired_electrons"])
    if unpaired:
        command.extend(["--spin", str(unpaired)])
    command.append(str(input_path))
    return command


def oracle_environment() -> tuple[dict[str, str], dict[str, Any]]:
    """Force the deterministic thread and restart environment boundary."""
    environment = os.environ.copy()
    removed = sorted(name for name in environment if name.upper().startswith("XTB"))
    for name in removed:
        environment.pop(name, None)
    fixed = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    environment.update(fixed)
    return environment, {
        "inherited_environment_boundary": (
            "All non-XTB variables except the recorded thread controls are inherited; "
            "runtime identity is pinned independently."
        ),
        "removed_variables": removed,
        "set": fixed,
    }


def run_process(
    command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path | None = None,
    context: str,
) -> subprocess.CompletedProcess[str]:
    """Run one argv-only child process with actionable launch diagnostics."""
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        raise PeriodicOracleError(
            f"cannot launch {command[0]} while {context}: {exc}"
        ) from exc


def executable_version(executable: Path, environment: dict[str, str]) -> str:
    """Read the exact CLI version text."""
    completed = run_process(
        [str(executable), "--version"],
        environment,
        context="querying the tblite version",
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"cannot query tblite version ({completed.returncode}):\n{completed.stdout}"
        )
    return completed.stdout.strip()


def discover_libtblite(
    executable: Path, environment: dict[str, str], expected_sha256: str
) -> dict[str, str]:
    """Resolve and hash the libtblite selected by the executable loader."""
    loader_inspector = shutil.which("ldd")
    if loader_inspector is None:
        raise PeriodicOracleError(
            "cannot resolve libtblite: oracle generation requires a Linux-like "
            "loader environment with ldd"
        )
    completed = run_process(
        [loader_inspector, str(executable)],
        environment,
        context="resolving libtblite with ldd on a Linux-like loader",
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"ldd failed for {executable}; oracle generation requires a "
            f"Linux-like loader:\n{completed.stdout}"
        )
    match = re.search(r"\blibtblite\.so\S*\s+=>\s+(\S+)", completed.stdout)
    if match is None:
        raise PeriodicOracleError("cannot resolve libtblite through ldd")
    library = Path(match.group(1)).resolve()
    digest = sha256_file(library)
    if digest != expected_sha256:
        raise PeriodicOracleError(
            f"libtblite SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return {
        "discovery": "ldd",
        "filename": library.name,
        "sha256": digest,
        "status": "resolved",
    }


def inspect_runtime_closure(
    executable: Path, environment: dict[str, str]
) -> dict[str, Any]:
    """Hash every non-system shared object selected by the Linux loader."""
    loader_inspector = shutil.which("ldd")
    if loader_inspector is None:
        raise PeriodicOracleError(
            "cannot inspect tblite runtime closure: oracle generation requires "
            "a Linux-like loader environment with ldd"
        )
    completed = run_process(
        [loader_inspector, str(executable)],
        environment,
        context="inspecting the tblite runtime closure with ldd",
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"ldd failed for {executable}; oracle generation requires a "
            f"Linux-like loader:\n{completed.stdout}"
        )
    libraries: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
        if match is None or match.group(2) == "not":
            continue
        soname, resolved_text = match.groups()
        resolved = Path(resolved_text).resolve()
        if not resolved.is_file():
            continue
        if any(
            resolved.is_relative_to(system_root)
            for system_root in (Path("/lib"), Path("/usr/lib"))
        ):
            continue
        libraries.append(
            {
                "filename": resolved.name,
                "sha256": sha256_file(resolved),
                "soname": soname,
            }
        )
    libraries.sort(key=lambda item: item["soname"])
    if not any(item["soname"].startswith("libtblite.so") for item in libraries):
        raise PeriodicOracleError("tblite runtime closure does not contain libtblite")
    return {
        "discovery": "ldd",
        "non_system_libraries": libraries,
        "sha256": sha256_json(libraries),
    }


def load_build_attestation(
    reference: dict[str, Any], *, verify_file_hash: bool = True
) -> tuple[dict[str, Any], str]:
    """Load and structurally verify the source-build attestation."""
    identity = reference.get("build_attestation")
    if not isinstance(identity, dict):
        raise PeriodicOracleError("reference build_attestation must be an object")
    path_text = identity.get("path")
    if not isinstance(path_text, str):
        raise PeriodicOracleError("reference build attestation path is invalid")
    expected_digest = validated_sha256(
        identity.get("sha256"), "reference build attestation SHA-256"
    )
    path = repository_path(path_text)
    if verify_file_hash and sha256_file(path) != expected_digest:
        raise PeriodicOracleError("tblite build attestation SHA-256 mismatch")
    attestation = load_json(path)
    if attestation.get("schema") != BUILD_SCHEMA:
        raise PeriodicOracleError(
            f"tblite build attestation schema must be {BUILD_SCHEMA}"
        )
    if attestation.get("build_id") != identity.get("build_id"):
        raise PeriodicOracleError("tblite build identity drifted")
    source = attestation.get("source")
    if (
        not isinstance(source, dict)
        or source.get("repository") != reference.get("repository")
        or source.get("revision") != reference.get("revision")
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("tree", ""))) is None
        or source.get("tracked_files_clean") is not True
    ):
        raise PeriodicOracleError("tblite build source identity is invalid")
    artifacts = attestation.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PeriodicOracleError("tblite build artifacts must be an object")
    for name in ("executable", "libtblite"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            raise PeriodicOracleError(f"tblite build artifact {name} is malformed")
        validated_sha256(artifact.get("sha256"), f"tblite build {name} SHA-256")
        if not isinstance(artifact.get("filename"), str) or not artifact["filename"]:
            raise PeriodicOracleError(f"tblite build {name} filename is invalid")
    if artifacts["executable"]["sha256"] != reference.get("executable_sha256"):
        raise PeriodicOracleError("tblite build executable identity drifted")
    runtime_artifacts = reference.get("runtime_artifacts")
    if not isinstance(runtime_artifacts, dict) or artifacts["libtblite"][
        "sha256"
    ] != runtime_artifacts.get("libtblite_sha256"):
        raise PeriodicOracleError("tblite build libtblite identity drifted")
    runtime = attestation.get("runtime")
    if not isinstance(runtime, dict):
        raise PeriodicOracleError("tblite build runtime closure is malformed")
    libraries = runtime.get("non_system_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise PeriodicOracleError("tblite build runtime closure is empty")
    for index, library in enumerate(libraries):
        if not isinstance(library, dict) or set(library) != {
            "filename",
            "sha256",
            "soname",
        }:
            raise PeriodicOracleError(
                f"tblite build runtime library {index} is malformed"
            )
        validated_sha256(
            library.get("sha256"), f"tblite build runtime library {index} SHA-256"
        )
    if runtime.get("sha256") != sha256_json(libraries):
        raise PeriodicOracleError("tblite build runtime closure digest drifted")
    environment = attestation.get("environment")
    packages = (
        environment.get("conda_packages") if isinstance(environment, dict) else None
    )
    if not isinstance(packages, list) or not packages:
        raise PeriodicOracleError("tblite build environment lock is missing")
    for index, package in enumerate(packages):
        if not isinstance(package, dict) or set(package) != {
            "build",
            "name",
            "sha256",
            "subdir",
            "url",
            "version",
        }:
            raise PeriodicOracleError(f"tblite build package {index} is malformed")
        validated_sha256(package.get("sha256"), f"tblite build package {index} SHA-256")
        if not all(
            isinstance(package.get(field), str) and package[field]
            for field in ("build", "name", "subdir", "url", "version")
        ):
            raise PeriodicOracleError(f"tblite build package {index} is incomplete")
    if attestation.get("redistribution") != {
        "binary_payload": "not-redistributed",
        "repository_payload": "attestation-and-normalized-numerical-results-only",
    }:
        raise PeriodicOracleError("tblite build redistribution boundary drifted")
    return attestation, expected_digest


def checked_process_output(
    command: list[str], environment: dict[str, str], *, cwd: Path, context: str
) -> str:
    """Run one metadata command and return its stripped standard output."""
    completed = run_process(command, environment, cwd=cwd, context=context)
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"{context} failed ({completed.returncode}):\n{completed.stdout}"
        )
    return completed.stdout.strip()


def capture_build_attestation(
    source_root: Path,
    build_dir: Path,
    environment_prefix: Path,
    install_prefix: Path,
    output: Path,
) -> None:
    """Capture a path-independent attestation for one reviewed tblite build."""
    source_root = source_root.resolve()
    build_dir = build_dir.resolve()
    environment_prefix = environment_prefix.resolve()
    install_prefix = install_prefix.resolve()
    executable = install_prefix / "bin/tblite"
    libraries = sorted(install_prefix.glob("lib*/**/libtblite.so.*.*.*"))
    if not executable.is_file() or len(libraries) != 1:
        raise PeriodicOracleError(
            "capture-build requires one installed tblite executable and versioned "
            "libtblite shared library"
        )
    libtblite = libraries[0].resolve()
    environment = os.environ.copy()
    library_dirs = sorted({str(libtblite.parent), str(environment_prefix / "lib")})
    inherited_loader_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = ":".join(
        library_dirs + ([inherited_loader_path] if inherited_loader_path else [])
    )

    revision = checked_process_output(
        ["git", "rev-parse", "HEAD"],
        environment,
        cwd=source_root,
        context="reading the tblite source revision",
    )
    tree = checked_process_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        environment,
        cwd=source_root,
        context="reading the tblite source tree",
    )
    tracked_status = checked_process_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        environment,
        cwd=source_root,
        context="checking the tblite tracked source state",
    )
    if tracked_status:
        raise PeriodicOracleError("tblite tracked source files are not clean")

    compiler = environment_prefix / "bin/x86_64-conda-linux-gnu-gfortran"
    compiler_version = checked_process_output(
        [str(compiler), "--version"],
        environment,
        cwd=source_root,
        context="reading the Fortran compiler version",
    ).splitlines()[0]
    compiler_target = checked_process_output(
        [str(compiler), "-dumpmachine"],
        environment,
        cwd=source_root,
        context="reading the Fortran compiler target",
    )
    meson = environment_prefix / "bin/meson"
    ninja = environment_prefix / "bin/ninja"
    meson_version = checked_process_output(
        [str(meson), "--version"],
        environment,
        cwd=source_root,
        context="reading the Meson version",
    )
    ninja_version = checked_process_output(
        [str(ninja), "--version"],
        environment,
        cwd=source_root,
        context="reading the Ninja version",
    )

    options_path = build_dir / "meson-info/intro-buildoptions.json"
    dependencies_path = build_dir / "meson-info/intro-dependencies.json"
    try:
        option_rows = json.loads(options_path.read_text(encoding="utf-8"))
        dependency_rows = json.loads(dependencies_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeriodicOracleError("cannot read Meson build introspection") from exc
    selected_options = {
        row["name"]: row["value"]
        for row in option_rows
        if isinstance(row, dict)
        and row.get("name")
        in {
            "b_lto",
            "buildtype",
            "ddx",
            "debug",
            "default_library",
            "fortran_std",
            "hdf5",
            "lapack",
            "openmp",
            "optimization",
            "python",
            "trexio",
        }
    }
    direct_dependencies = sorted(
        (
            {
                "name": str(row.get("name")),
                "type": str(row.get("type")),
                "version": str(row.get("version")),
            }
            for row in dependency_rows
            if isinstance(row, dict)
        ),
        key=lambda item: item["name"],
    )

    conda_packages: list[dict[str, str]] = []
    for metadata_path in sorted((environment_prefix / "conda-meta").glob("*.json")):
        metadata = load_json(metadata_path)
        record = {
            "build": str(metadata.get("build", "")),
            "name": str(metadata.get("name", "")),
            "sha256": str(metadata.get("sha256", "")),
            "subdir": str(metadata.get("subdir", "")),
            "url": str(metadata.get("url", "")),
            "version": str(metadata.get("version", "")),
        }
        validated_sha256(record["sha256"], f"conda package {record['name']} SHA-256")
        if not all(record.values()):
            raise PeriodicOracleError(
                f"conda package metadata is incomplete: {metadata_path.name}"
            )
        conda_packages.append(record)
    conda_packages.sort(key=lambda item: (item["name"], item["version"], item["build"]))

    subprojects: list[dict[str, Any]] = []
    mstore = source_root / "subprojects/mstore"
    if (mstore / ".git").exists():
        subprojects.append(
            {
                "linked_into_oracle": False,
                "repository": checked_process_output(
                    ["git", "remote", "get-url", "origin"],
                    environment,
                    cwd=mstore,
                    context="reading the mstore repository",
                ),
                "revision": checked_process_output(
                    ["git", "rev-parse", "HEAD"],
                    environment,
                    cwd=mstore,
                    context="reading the mstore revision",
                ),
                "role": "configured test-only Meson subproject",
                "tree": checked_process_output(
                    ["git", "rev-parse", "HEAD^{tree}"],
                    environment,
                    cwd=mstore,
                    context="reading the mstore tree",
                ),
            }
        )

    test_counts: dict[str, int] = {}
    test_log = build_dir / "meson-logs/testlog.json"
    if test_log.is_file():
        for line in test_log.read_text(encoding="utf-8").splitlines():
            result = str(json.loads(line).get("result", "UNKNOWN")).lower()
            test_counts[result] = test_counts.get(result, 0) + 1

    runtime = inspect_runtime_closure(executable, environment)
    executable_digest = sha256_file(executable)
    attestation = {
        "artifacts": {
            "executable": {"filename": executable.name, "sha256": executable_digest},
            "libtblite": {
                "filename": libtblite.name,
                "sha256": sha256_file(libtblite),
            },
        },
        "build": {
            "compiler": {
                "language": "Fortran",
                "target": compiler_target,
                "version": compiler_version,
            },
            "direct_dependencies": direct_dependencies,
            "meson_options": selected_options,
            "tools": {"meson": meson_version, "ninja": ninja_version},
        },
        "build_id": f"tblite-{revision[:8]}-gfortran14-netlib-{executable_digest[:8]}",
        "environment": {"conda_packages": conda_packages},
        "generation": {
            "command_template": [
                "periodic_gfn2.py",
                "capture-build",
                "--source-root",
                "{source_root}",
                "--build-dir",
                "{build_dir}",
                "--environment-prefix",
                "{environment_prefix}",
                "--install-prefix",
                "{install_prefix}",
                "--output",
                "{output}",
            ]
        },
        "redistribution": {
            "binary_payload": "not-redistributed",
            "repository_payload": "attestation-and-normalized-numerical-results-only",
        },
        "runtime": runtime,
        "schema": BUILD_SCHEMA,
        "source": {
            "repository": "https://github.com/tblite/tblite",
            "revision": revision,
            "tracked_files_clean": True,
            "tree": tree,
        },
        "subprojects": subprojects,
        "system": {
            "architecture": platform.machine(),
            "libc": " ".join(part for part in platform.libc_ver() if part),
            "operating_system": platform.system(),
        },
        "upstream_tests": {
            "complete": False,
            "counts": dict(sorted(test_counts.items())),
            "qualification": (
                "The captured Meson run was interrupted after slow netlib timeouts; "
                "it is build provenance, not an upstream test-suite pass."
            ),
        },
    }
    dump_json(output, attestation)
    print(f"captured {output}")  # noqa: T201 - CLI progress


def normalize_tblite(raw: dict[str, Any], atom_count: int) -> dict[str, Any]:
    """Normalize tblite JSON into xTBloom units and row-major strain order."""
    energy = finite_number(raw.get("energy"), "tblite energy")
    gradient = finite_vector(raw.get("gradient"), 3 * atom_count, "tblite gradient")
    # tblite reshapes its Fortran [3,3] array in column-major order.
    virial_column_major = finite_vector(raw.get("virial"), 9, "tblite virial")
    strain_row_major = [
        virial_column_major[3 * column + row] for row in range(3) for column in range(3)
    ]
    return {
        "energy_hartree": energy,
        "forces_hartree_per_bohr": [-value for value in gradient],
        "gradient_hartree_per_bohr": gradient,
        "strain_derivatives_hartree": strain_row_major,
    }


def run_tblite(
    executable: Path,
    case: dict[str, Any],
    input_path: Path,
    output_path: Path,
    environment: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Execute one isolated oracle calculation and normalize its output."""
    command = tblite_command(executable, case, input_path, output_path)
    completed = run_process(
        command,
        environment,
        cwd=output_path.parent,
        context=f"running periodic oracle case {case['id']}",
    )
    if completed.returncode != 0:
        raise PeriodicOracleError(
            f"tblite failed for {case['id']} ({completed.returncode}):\n"
            f"{completed.stdout}"
        )
    raw = load_json(output_path)
    return normalize_tblite(raw, int(case["atom_count"])), sha256_file(output_path)


def evaluation_record(
    properties: dict[str, Any], input_path: Path, raw_digest: str
) -> dict[str, Any]:
    """Retain one normalized run and the identities needed to reproduce it."""
    return {
        "input_sha256": sha256_file(input_path),
        "properties": properties,
        "raw_output_sha256": raw_digest,
        "source_output_sha256": sha256_json(properties),
    }


def strain_finite_differences(
    executable: Path,
    case: dict[str, Any],
    structure: dict[str, Any],
    work: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Generate three-step central differences for six ordered strain modes."""
    evidence: dict[str, Any] = {
        "affine_convention": "r'=(I+epsilon)r; H'=H(I+epsilon)^T",
        "modes": {},
        "steps": list(STRAIN_STEPS),
    }
    for mode, row, column in STRAIN_MODES:
        estimates: list[float] = []
        evaluations: list[dict[str, Any]] = []
        for step_index, step in enumerate(STRAIN_STEPS):
            energy_pair: dict[int, float] = {}
            evaluation: dict[str, Any] = {"step": step}
            for sign, label in ((-1, "minus"), (1, "plus")):
                positions, cell = affine_deformation(
                    structure, row, column, sign * step
                )
                input_path = work / f"{case['id']}-{mode}-{step_index}-{sign:+d}.tmol"
                input_path.write_text(
                    format_turbomole(structure, positions, cell), encoding="utf-8"
                )
                output_path = input_path.with_suffix(".json")
                properties, raw_digest = run_tblite(
                    executable, case, input_path, output_path, environment
                )
                energy = float(properties["energy_hartree"])
                energy_pair[sign] = energy
                evaluation[label] = {
                    "energy_hartree": energy,
                    "input_sha256": sha256_file(input_path),
                    "raw_output_sha256": raw_digest,
                    "source_output_sha256": sha256_json(properties),
                }
            evaluations.append(evaluation)
            estimates.append((energy_pair[1] - energy_pair[-1]) / (2.0 * step))
        # The two finest estimates give the ordinary O(h^2) Richardson value.
        richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
        evidence["modes"][mode] = {
            "component": [row, column],
            "central_differences_hartree": estimates,
            "evaluations": evaluations,
            "richardson_hartree": richardson,
        }
    return evidence


def cartesian_finite_differences(
    executable: Path,
    case: dict[str, Any],
    structure: dict[str, Any],
    work: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Generate three-step central differences for every Cartesian force."""
    evidence: dict[str, Any] = {
        "coordinates": [],
        "definition": "F_Ak=-(E(R_Ak+h)-E(R_Ak-h))/(2h)",
        "steps_bohr": list(CARTESIAN_STEPS),
    }
    for atom in range(int(case["atom_count"])):
        for axis in range(3):
            estimates: list[float] = []
            evaluations: list[dict[str, Any]] = []
            for step_index, step in enumerate(CARTESIAN_STEPS):
                energy_pair: dict[int, float] = {}
                evaluation: dict[str, Any] = {"step_bohr": step}
                for sign, label in ((-1, "minus"), (1, "plus")):
                    positions, cell = cartesian_displacement(
                        structure, atom, axis, sign * step
                    )
                    input_path = work / (
                        f"{case['id']}-cart-{atom}-{axis}-{step_index}-{sign:+d}.tmol"
                    )
                    input_path.write_text(
                        format_turbomole(structure, positions, cell), encoding="utf-8"
                    )
                    output_path = input_path.with_suffix(".json")
                    properties, raw_digest = run_tblite(
                        executable, case, input_path, output_path, environment
                    )
                    energy_pair[sign] = float(properties["energy_hartree"])
                    record = evaluation_record(properties, input_path, raw_digest)
                    evaluation[label] = {
                        "energy_hartree": properties["energy_hartree"],
                        "input_sha256": record["input_sha256"],
                        "raw_output_sha256": record["raw_output_sha256"],
                        "source_output_sha256": record["source_output_sha256"],
                    }
                evaluations.append(evaluation)
                estimates.append((energy_pair[-1] - energy_pair[1]) / (2.0 * step))
            richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
            evidence["coordinates"].append(
                {
                    "atom": atom,
                    "axis": axis,
                    "central_forces_hartree_per_bohr": estimates,
                    "evaluations": evaluations,
                    "richardson_force_hartree_per_bohr": richardson,
                }
            )
    return evidence


def invariant_evidence(
    executable: Path,
    case: dict[str, Any],
    structure: dict[str, Any],
    work: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Generate translation, wrapping, and cell-basis oracle variants."""
    variants: dict[str, Any] = {}
    for name in ("translation", "cell_basis"):
        positions, cell, definition = invariant_variant(structure, name)
        input_path = work / f"{case['id']}-invariant-{name}.tmol"
        input_path.write_text(
            format_turbomole(structure, positions, cell), encoding="utf-8"
        )
        output_path = input_path.with_suffix(".json")
        properties, raw_digest = run_tblite(
            executable, case, input_path, output_path, environment
        )
        variants[name] = {
            "definition": definition,
            **evaluation_record(properties, input_path, raw_digest),
        }
    outside, wrapped, wrapping_definition = wrapping_probe(structure)
    return {
        "reciprocal_identity_max_error": reciprocal_identity_error(
            structure["cell_matrix_row_major_bohr"]
        ),
        "variants": variants,
        "wrapping": {
            "definition": wrapping_definition,
            "outside_positions_bohr": outside,
            "outside_positions_sha256": sha256_json(outside),
            "wrapped_positions_bohr": wrapped,
            "wrapped_positions_sha256": sha256_json(wrapped),
        },
    }


def selected_cases(
    manifest: dict[str, Any], names: list[str] | None
) -> list[dict[str, Any]]:
    """Select unique case IDs and reject command-line typos."""
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise PeriodicOracleError("manifest cases must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise PeriodicOracleError("each case must have a string ID")
        if case["id"] in by_id:
            raise PeriodicOracleError(f"duplicate case ID: {case['id']}")
        by_id[case["id"]] = case
    if not names:
        return list(by_id.values())
    unknown = sorted(set(names) - set(by_id))
    if unknown:
        raise PeriodicOracleError(f"unknown case ID(s): {', '.join(unknown)}")
    return [by_id[name] for name in names]


def generate(
    manifest_path: Path, executable_arg: Path, output_dir: Path, names: list[str] | None
) -> None:
    """Generate neutral primary and charged diagnostic tblite results."""
    manifest = load_json(manifest_path)
    reference = manifest["reference_engine"]
    attestation, attestation_digest = load_build_attestation(reference)
    executable = Path(shutil.which(str(executable_arg)) or executable_arg).resolve()
    if not executable.is_file():
        raise PeriodicOracleError(f"tblite executable does not exist: {executable_arg}")
    if sha256_file(executable) != reference["executable_sha256"]:
        raise PeriodicOracleError("tblite executable SHA-256 does not match manifest")
    environment, environment_provenance = oracle_environment()
    version = executable_version(executable, environment)
    if f"tblite version {reference['version']}" not in version:
        raise PeriodicOracleError(f"unexpected tblite version:\n{version}")
    runtime_closure = inspect_runtime_closure(executable, environment)
    if runtime_closure != attestation["runtime"]:
        raise PeriodicOracleError(
            "tblite runtime closure does not match the reviewed build attestation"
        )
    libtblite = next(
        item
        for item in runtime_closure["non_system_libraries"]
        if item["soname"].startswith("libtblite.so")
    )
    runtime = {
        "closure_sha256": runtime_closure["sha256"],
        "libtblite": {
            "discovery": runtime_closure["discovery"],
            "filename": libtblite["filename"],
            "sha256": libtblite["sha256"],
            "status": "resolved",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="xtbloom-periodic-oracle-") as temporary:
        work = Path(temporary)
        for case in selected_cases(manifest, names):
            if case["oracle_role"] == "analytic-background":
                continue
            input_path = repository_path(case["input"])
            raw_path = work / f"{case['id']}.json"
            properties, raw_digest = run_tblite(
                executable, case, input_path, raw_path, environment
            )
            structure = parse_turbomole(input_path)
            finite_difference = None
            if case.get("strain_finite_difference", False):
                finite_difference = strain_finite_differences(
                    executable, case, structure, work, environment
                )
            cartesian_evidence = None
            if case.get("cartesian_finite_difference", False):
                cartesian_evidence = cartesian_finite_differences(
                    executable, case, structure, work, environment
                )
            invariants = None
            if case.get("invariant_variants", False):
                invariants = invariant_evidence(
                    executable, case, structure, work, environment
                )
            provenance = {
                "accuracy": ACCURACY,
                "build_attestation_sha256": attestation_digest,
                "build_id": attestation["build_id"],
                "command_template": command_template(),
                "engine": "tblite",
                "environment": environment_provenance,
                "executable_sha256": sha256_file(executable),
                "executable_version": version,
                "generation_mode": "live-cli",
                "input": case["input"],
                "input_sha256": sha256_file(input_path),
                "raw_output_sha256": raw_digest,
                "runtime": runtime,
                "source_revision": reference["revision"],
            }
            golden: dict[str, Any] = {
                "case_id": case["id"],
                "oracle_role": case["oracle_role"],
                "properties": properties,
                "provenance": provenance,
                "schema": GOLDEN_SCHEMA,
                "source_output_sha256": sha256_json(properties),
            }
            if finite_difference is not None:
                golden["strain_finite_difference"] = finite_difference
            if cartesian_evidence is not None:
                golden["cartesian_finite_difference"] = cartesian_evidence
            if invariants is not None:
                golden["invariant_evidence"] = invariants
            destination = output_dir / f"{case['id']}.json"
            dump_json(destination, golden)
            print(f"generated {destination}")  # noqa: T201 - CLI progress


def check_source_modules(reference: dict[str, Any]) -> None:
    """Validate immutable source identifiers without requiring a checkout."""
    modules = reference.get("source_modules")
    if not isinstance(modules, dict) or not modules:
        raise PeriodicOracleError("reference source_modules must be a nonempty object")
    for path, identity in modules.items():
        if not isinstance(path, str) or not isinstance(identity, dict):
            raise PeriodicOracleError("source module entries are malformed")
        if re.fullmatch(r"[0-9a-f]{40}", str(identity.get("git_blob", ""))) is None:
            raise PeriodicOracleError(f"{path} has an invalid Git blob ID")
        if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))) is None:
            raise PeriodicOracleError(f"{path} has an invalid SHA-256")
        if not isinstance(identity.get("role"), str) or not identity["role"]:
            raise PeriodicOracleError(f"{path} has no reviewed role")


def check_strain_evidence(
    golden: dict[str, Any],
    strain: list[float],
    structure: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, tuple[list[float], float]]:
    """Require convergent multi-step evidence for all six released modes."""
    evidence = golden.get("strain_finite_difference")
    if not isinstance(evidence, dict) or evidence.get("steps") != list(STRAIN_STEPS):
        raise PeriodicOracleError(f"{golden['case_id']} lacks canonical strain steps")
    if evidence.get("affine_convention") != "r'=(I+epsilon)r; H'=H(I+epsilon)^T":
        raise PeriodicOracleError(
            f"{golden['case_id']} affine strain convention drifted"
        )
    modes = evidence.get("modes")
    if not isinstance(modes, dict) or set(modes) != {mode[0] for mode in STRAIN_MODES}:
        raise PeriodicOracleError(f"{golden['case_id']} lacks all six strain modes")
    recomputed: dict[str, tuple[list[float], float]] = {}
    for name, row, column in STRAIN_MODES:
        mode = modes[name]
        if not isinstance(mode, dict):
            raise PeriodicOracleError(f"{golden['case_id']} {name} mode is malformed")
        if mode.get("component") != [row, column]:
            raise PeriodicOracleError(f"{golden['case_id']} {name} component drifted")
        stored_estimates = finite_vector(
            mode.get("central_differences_hartree"),
            len(STRAIN_STEPS),
            f"{golden['case_id']} {name} strain differences",
        )
        evaluations = mode.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(STRAIN_STEPS):
            raise PeriodicOracleError(
                f"{golden['case_id']} {name} lacks complete strain energy pairs"
            )
        estimates: list[float] = []
        for index, (step, evaluation) in enumerate(
            zip(STRAIN_STEPS, evaluations, strict=True)
        ):
            if not isinstance(evaluation, dict) or evaluation.get("step") != step:
                raise PeriodicOracleError(
                    f"{golden['case_id']} {name} strain step {index} drifted"
                )
            energies: dict[str, float] = {}
            for sign in ("minus", "plus"):
                run = evaluation.get(sign)
                if not isinstance(run, dict) or set(run) != {
                    "energy_hartree",
                    "input_sha256",
                    "raw_output_sha256",
                    "source_output_sha256",
                }:
                    raise PeriodicOracleError(
                        f"{golden['case_id']} {name} {sign} run is malformed"
                    )
                energies[sign] = finite_number(
                    run.get("energy_hartree"),
                    f"{golden['case_id']} {name} {sign} energy",
                )
                for hash_name in (
                    "input_sha256",
                    "raw_output_sha256",
                    "source_output_sha256",
                ):
                    validated_sha256(
                        run.get(hash_name),
                        f"{golden['case_id']} {name} {sign} {hash_name}",
                    )
                signed_step = (-step) if sign == "minus" else step
                positions, cell = affine_deformation(
                    structure, row, column, signed_step
                )
                expected_input_sha256 = sha256_bytes(
                    format_turbomole(structure, positions, cell).encode("utf-8")
                )
                if run["input_sha256"] != expected_input_sha256:
                    raise PeriodicOracleError(
                        f"{golden['case_id']} {name} {sign} input SHA-256 drifted"
                    )
            estimates.append((energies["plus"] - energies["minus"]) / (2.0 * step))
        if estimates != stored_estimates:
            raise PeriodicOracleError(
                f"{golden['case_id']} {name} central differences drifted"
            )
        stability_tolerance = finite_number(
            case["tolerances"]["strain_finite_difference_stability_hartree"],
            f"{golden['case_id']} strain stability tolerance",
        )
        coarse_change = abs(estimates[1] - estimates[0])
        fine_change = abs(estimates[2] - estimates[1])
        if fine_change > coarse_change + stability_tolerance:
            raise PeriodicOracleError(
                f"{golden['case_id']} {name} strain finite differences are unstable"
            )
        richardson = finite_number(
            mode.get("richardson_hartree"), f"{golden['case_id']} {name} Richardson"
        )
        expected_richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
        if richardson != expected_richardson:
            raise PeriodicOracleError(f"{golden['case_id']} {name} Richardson drifted")
        analytic = strain[3 * row + column]
        derivative_tolerance = finite_number(
            case["tolerances"]["strain_derivatives_hartree"],
            f"{golden['case_id']} strain derivative tolerance",
        )
        if abs(richardson - analytic) > derivative_tolerance:
            raise PeriodicOracleError(
                f"{golden['case_id']} {name} strain mismatch: "
                f"analytic {analytic}, finite difference {richardson}"
            )
        recomputed[name] = (estimates, richardson)
    return recomputed


def compare_strain_evidence(
    case: dict[str, Any],
    expected: dict[str, Any],
    actual: dict[str, Any],
    structure: dict[str, Any],
) -> None:
    """Compare every retained strain run, derivative, and provenance identity."""
    expected_evidence = expected.get("strain_finite_difference")
    actual_evidence = actual.get("strain_finite_difference")
    if not isinstance(expected_evidence, dict) or not isinstance(actual_evidence, dict):
        raise PeriodicOracleError(f"{case['id']} lacks comparable strain evidence")
    expected_summary = check_strain_evidence(
        expected,
        expected["properties"]["strain_derivatives_hartree"],
        structure,
        case,
    )
    actual_summary = check_strain_evidence(
        actual,
        actual["properties"]["strain_derivatives_hartree"],
        structure,
        case,
    )
    energy_tolerance = finite_number(
        case["tolerances"]["energy_hartree"], f"{case['id']} energy tolerance"
    )
    strain_tolerance = finite_number(
        case["tolerances"]["strain_derivatives_hartree"],
        f"{case['id']} strain tolerance",
    )
    if actual_evidence.get("steps") != expected_evidence.get(
        "steps"
    ) or actual_evidence.get("affine_convention") != expected_evidence.get(
        "affine_convention"
    ):
        raise PeriodicOracleError(f"{case['id']} strain contract mismatch")
    for name, _row, _column in STRAIN_MODES:
        expected_mode = expected_evidence["modes"][name]
        actual_mode = actual_evidence["modes"][name]
        if actual_mode.get("component") != expected_mode.get("component"):
            raise PeriodicOracleError(f"{case['id']} {name} component mismatch")
        for index, (expected_run, actual_run) in enumerate(
            zip(
                expected_mode["evaluations"],
                actual_mode["evaluations"],
                strict=True,
            )
        ):
            for sign in ("minus", "plus"):
                expected_point = expected_run[sign]
                actual_point = actual_run[sign]
                for hash_name in (
                    "input_sha256",
                    "raw_output_sha256",
                    "source_output_sha256",
                ):
                    if actual_point[hash_name] != expected_point[hash_name]:
                        raise PeriodicOracleError(
                            f"{case['id']} {name} step {index} {sign} "
                            f"{hash_name} mismatch"
                        )
                energy_error = abs(
                    actual_point["energy_hartree"] - expected_point["energy_hartree"]
                )
                if energy_error > energy_tolerance:
                    raise PeriodicOracleError(
                        f"{case['id']} {name} step {index} {sign} energy error "
                        f"{energy_error} > {energy_tolerance}"
                    )
        for index, (expected_value, actual_value) in enumerate(
            zip(expected_summary[name][0], actual_summary[name][0], strict=True)
        ):
            difference_error = abs(actual_value - expected_value)
            if difference_error > strain_tolerance:
                raise PeriodicOracleError(
                    f"{case['id']} {name} step {index} central-difference error "
                    f"{difference_error} > {strain_tolerance}"
                )
        richardson_error = abs(actual_summary[name][1] - expected_summary[name][1])
        if richardson_error > strain_tolerance:
            raise PeriodicOracleError(
                f"{case['id']} {name} Richardson error "
                f"{richardson_error} > {strain_tolerance}"
            )


def check_cartesian_evidence(
    golden: dict[str, Any],
    forces: list[float],
    structure: dict[str, Any],
    tolerance: float,
) -> None:
    """Recompute every retained Cartesian central difference from energy pairs."""
    evidence = golden.get("cartesian_finite_difference")
    if (
        not isinstance(evidence, dict)
        or evidence.get("steps_bohr") != list(CARTESIAN_STEPS)
        or evidence.get("definition") != "F_Ak=-(E(R_Ak+h)-E(R_Ak-h))/(2h)"
    ):
        raise PeriodicOracleError(
            f"{golden['case_id']} lacks canonical Cartesian finite differences"
        )
    coordinates = evidence.get("coordinates")
    expected_count = len(forces)
    if not isinstance(coordinates, list) or len(coordinates) != expected_count:
        raise PeriodicOracleError(
            f"{golden['case_id']} Cartesian finite-difference extent drifted"
        )
    for coordinate_index, coordinate in enumerate(coordinates):
        atom, axis = divmod(coordinate_index, 3)
        if (
            not isinstance(coordinate, dict)
            or coordinate.get("atom") != atom
            or coordinate.get("axis") != axis
        ):
            raise PeriodicOracleError(
                f"{golden['case_id']} Cartesian coordinate order drifted"
            )
        stored_estimates = finite_vector(
            coordinate.get("central_forces_hartree_per_bohr"),
            len(CARTESIAN_STEPS),
            f"{golden['case_id']} Cartesian force differences",
        )
        evaluations = coordinate.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != len(
            CARTESIAN_STEPS
        ):
            raise PeriodicOracleError(
                f"{golden['case_id']} Cartesian energy pairs are incomplete"
            )
        estimates: list[float] = []
        for step_index, (step, evaluation) in enumerate(
            zip(CARTESIAN_STEPS, evaluations, strict=True)
        ):
            if not isinstance(evaluation, dict) or evaluation.get("step_bohr") != step:
                raise PeriodicOracleError(
                    f"{golden['case_id']} Cartesian step {step_index} drifted"
                )
            energies: dict[str, float] = {}
            for sign in ("minus", "plus"):
                run = evaluation.get(sign)
                if not isinstance(run, dict) or set(run) != {
                    "energy_hartree",
                    "input_sha256",
                    "raw_output_sha256",
                    "source_output_sha256",
                }:
                    raise PeriodicOracleError(
                        f"{golden['case_id']} Cartesian {sign} run is malformed"
                    )
                energies[sign] = finite_number(
                    run.get("energy_hartree"),
                    f"{golden['case_id']} Cartesian {sign} energy",
                )
                for hash_name in (
                    "input_sha256",
                    "raw_output_sha256",
                    "source_output_sha256",
                ):
                    validated_sha256(
                        run.get(hash_name),
                        f"{golden['case_id']} Cartesian {sign} {hash_name}",
                    )
                signed_step = (-step) if sign == "minus" else step
                positions, cell = cartesian_displacement(
                    structure, atom, axis, signed_step
                )
                expected_input_sha256 = sha256_bytes(
                    format_turbomole(structure, positions, cell).encode("utf-8")
                )
                if run["input_sha256"] != expected_input_sha256:
                    raise PeriodicOracleError(
                        f"{golden['case_id']} Cartesian {sign} input hash drifted"
                    )
            estimates.append((energies["minus"] - energies["plus"]) / (2.0 * step))
        if estimates != stored_estimates:
            raise PeriodicOracleError(
                f"{golden['case_id']} Cartesian central forces drifted"
            )
        richardson = finite_number(
            coordinate.get("richardson_force_hartree_per_bohr"),
            f"{golden['case_id']} Cartesian Richardson force",
        )
        expected_richardson = (4.0 * estimates[-1] - estimates[-2]) / 3.0
        if richardson != expected_richardson:
            raise PeriodicOracleError(
                f"{golden['case_id']} Cartesian Richardson force drifted"
            )
        if abs(richardson - forces[coordinate_index]) > tolerance:
            raise PeriodicOracleError(
                f"{golden['case_id']} Cartesian force mismatch at atom {atom}, "
                f"axis {axis}: analytic {forces[coordinate_index]}, "
                f"finite difference {richardson}"
            )


def property_error(expected: object, actual: object, label: str) -> float:
    """Return a maximum absolute scalar/vector property error."""
    expected_values = expected if isinstance(expected, list) else [expected]
    actual_values = actual if isinstance(actual, list) else [actual]
    if len(expected_values) != len(actual_values):
        raise PeriodicOracleError(f"{label} extent mismatch")
    return max(
        abs(finite_number(lhs, label) - finite_number(rhs, label))
        for lhs, rhs in zip(expected_values, actual_values, strict=True)
    )


def check_invariant_evidence(
    golden: dict[str, Any],
    case: dict[str, Any],
    structure: dict[str, Any],
    properties: dict[str, Any],
) -> None:
    """Gate reciprocal, conservation, wrapping, translation, and basis invariants."""
    evidence = golden.get("invariant_evidence")
    if not isinstance(evidence, dict):
        raise PeriodicOracleError(f"{case['id']} lacks periodic invariant evidence")
    reciprocal_error = finite_number(
        evidence.get("reciprocal_identity_max_error"),
        f"{case['id']} reciprocal identity error",
    )
    expected_reciprocal_error = reciprocal_identity_error(
        structure["cell_matrix_row_major_bohr"]
    )
    if reciprocal_error != expected_reciprocal_error or reciprocal_error > 1.0e-13:
        raise PeriodicOracleError(f"{case['id']} reciprocal identity drifted")
    forces = finite_vector(
        properties["forces_hartree_per_bohr"],
        3 * int(case["atom_count"]),
        f"{case['id']} invariant forces",
    )
    net_force = [
        sum(forces[3 * atom + axis] for atom in range(int(case["atom_count"])))
        for axis in range(3)
    ]
    net_force_tolerance = finite_number(
        case["tolerances"]["net_force_hartree_per_bohr"],
        f"{case['id']} net-force tolerance",
    )
    if max(abs(value) for value in net_force) > net_force_tolerance:
        raise PeriodicOracleError(f"{case['id']} net force is not zero")
    strain = finite_vector(
        properties["strain_derivatives_hartree"], 9, f"{case['id']} invariant strain"
    )
    symmetry_error = max(
        abs(strain[3 * row + column] - strain[3 * column + row])
        for row in range(3)
        for column in range(row + 1, 3)
    )
    symmetry_tolerance = finite_number(
        case["tolerances"]["strain_symmetry_hartree"],
        f"{case['id']} strain-symmetry tolerance",
    )
    if symmetry_error > symmetry_tolerance:
        raise PeriodicOracleError(f"{case['id']} strain derivative is not symmetric")
    variants = evidence.get("variants")
    if not isinstance(variants, dict) or set(variants) != {
        "translation",
        "cell_basis",
    }:
        raise PeriodicOracleError(f"{case['id']} invariant variant set drifted")
    for name, variant in variants.items():
        positions, cell, definition = invariant_variant(structure, name)
        if not isinstance(variant, dict) or variant.get("definition") != definition:
            raise PeriodicOracleError(f"{case['id']} {name} invariant is malformed")
        expected_input_sha256 = sha256_bytes(
            format_turbomole(structure, positions, cell).encode("utf-8")
        )
        if variant.get("input_sha256") != expected_input_sha256:
            raise PeriodicOracleError(f"{case['id']} {name} input hash drifted")
        validated_sha256(
            variant.get("raw_output_sha256"), f"{case['id']} {name} raw output"
        )
        variant_properties = variant.get("properties")
        if not isinstance(variant_properties, dict):
            raise PeriodicOracleError(f"{case['id']} {name} properties are malformed")
        if variant.get("source_output_sha256") != sha256_json(variant_properties):
            raise PeriodicOracleError(f"{case['id']} {name} output hash drifted")
        for property_name in (
            "energy_hartree",
            "forces_hartree_per_bohr",
            "strain_derivatives_hartree",
        ):
            tolerance = finite_number(
                case["tolerances"][property_name],
                f"{case['id']} {property_name} tolerance",
            )
            maximum = property_error(
                properties[property_name],
                variant_properties.get(property_name),
                f"{case['id']} {name} {property_name}",
            )
            if maximum > tolerance:
                raise PeriodicOracleError(
                    f"{case['id']} {name} {property_name} invariant error "
                    f"{maximum} > {tolerance}"
                )
    outside, wrapped, wrapping_definition = wrapping_probe(structure)
    wrapping = evidence.get("wrapping")
    if (
        not isinstance(wrapping, dict)
        or wrapping.get("definition") != wrapping_definition
    ):
        raise PeriodicOracleError(f"{case['id']} wrapping evidence drifted")
    stored_outside = finite_vector(
        wrapping.get("outside_positions_bohr"),
        len(outside),
        f"{case['id']} outside-cell positions",
    )
    stored_wrapped = finite_vector(
        wrapping.get("wrapped_positions_bohr"),
        len(wrapped),
        f"{case['id']} wrapped positions",
    )
    if wrapping.get("outside_positions_sha256") != sha256_json(
        stored_outside
    ) or wrapping.get("wrapped_positions_sha256") != sha256_json(stored_wrapped):
        raise PeriodicOracleError(f"{case['id']} wrapping evidence hash drifted")
    # Different supported Python/libm builds can round the independently
    # inverted skew cell by one ULP. Preserve the committed values by hash,
    # then compare the recomputed mathematical contract with an explicit bound.
    wrapping_error = max(
        max(abs(lhs - rhs) for lhs, rhs in zip(stored_outside, outside, strict=True)),
        max(abs(lhs - rhs) for lhs, rhs in zip(stored_wrapped, wrapped, strict=True)),
        max(
            abs(lhs - rhs)
            for lhs, rhs in zip(
                stored_wrapped, structure["positions_bohr"], strict=True
            )
        ),
    )
    if wrapping_error > 5.0e-14:
        raise PeriodicOracleError(
            f"{case['id']} wrapped coordinates do not recover the canonical cell"
        )


def check_analytic_background(manifest: dict[str, Any]) -> None:
    """Reconstruct the charged-cell background energy, potential, and strain."""
    cases = manifest.get("analytic_background_cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise PeriodicOracleError("at least two analytic background cases are required")
    for case in cases:
        charge = finite_number(case.get("charge_e"), "background charge")
        alpha = finite_number(case.get("alpha_bohr_inverse"), "background alpha")
        volume = finite_number(case.get("volume_bohr3"), "background volume")
        if not alpha > 0.0 or not volume > 0.0 or charge == 0.0:
            raise PeriodicOracleError("background cases require Q!=0, alpha>0, V>0")
        factor = math.pi / (alpha * alpha * volume)
        expected_energy = -0.5 * factor * charge * charge
        expected_potential = -factor * charge
        expected_strain = [0.0] * 9
        for diagonal in (0, 4, 8):
            expected_strain[diagonal] = -expected_energy
        energy = finite_number(case.get("energy_hartree"), "background energy")
        if not math.isclose(energy, expected_energy, rel_tol=1.0e-14, abs_tol=0.0):
            raise PeriodicOracleError(f"{case.get('id')} background energy drifted")
        potential = finite_number(
            case.get("potential_hartree_per_e"), "background potential"
        )
        if not math.isclose(
            potential, expected_potential, rel_tol=1.0e-14, abs_tol=0.0
        ):
            raise PeriodicOracleError(f"{case.get('id')} background potential drifted")
        strain = finite_vector(
            case.get("strain_derivatives_hartree"), 9, "background strain"
        )
        if any(
            not math.isclose(lhs, rhs, rel_tol=1.0e-14, abs_tol=1.0e-300)
            for lhs, rhs in zip(strain, expected_strain, strict=True)
        ):
            raise PeriodicOracleError(f"{case.get('id')} background strain drifted")


def validate_golden_document(
    case: dict[str, Any],
    golden: dict[str, Any],
    reference: dict[str, Any],
    attestation: dict[str, Any],
    attestation_digest: str,
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Validate one complete generated result before trusting its numbers."""
    role = case.get("oracle_role")
    if (
        golden.get("schema") != GOLDEN_SCHEMA
        or golden.get("case_id") != case["id"]
        or golden.get("oracle_role") != role
    ):
        raise PeriodicOracleError(f"{case['id']} golden identity mismatch")
    properties = golden.get("properties")
    if not isinstance(properties, dict):
        raise PeriodicOracleError(f"{case['id']} properties must be an object")
    finite_number(properties.get("energy_hartree"), f"{case['id']} energy")
    gradient = finite_vector(
        properties.get("gradient_hartree_per_bohr"),
        3 * int(case["atom_count"]),
        f"{case['id']} gradient",
    )
    forces = finite_vector(
        properties.get("forces_hartree_per_bohr"),
        len(gradient),
        f"{case['id']} forces",
    )
    if forces != [-value for value in gradient]:
        raise PeriodicOracleError(f"{case['id']} force sign drifted")
    strain = finite_vector(
        properties.get("strain_derivatives_hartree"), 9, f"{case['id']} strain"
    )
    if golden.get("source_output_sha256") != sha256_json(properties):
        raise PeriodicOracleError(f"{case['id']} normalized output hash mismatch")
    provenance = golden.get("provenance")
    if not isinstance(provenance, dict):
        raise PeriodicOracleError(f"{case['id']} provenance must be an object")
    runtime = provenance.get("runtime")
    libtblite = runtime.get("libtblite") if isinstance(runtime, dict) else None
    if (
        provenance.get("accuracy") != ACCURACY
        or provenance.get("build_attestation_sha256") != attestation_digest
        or provenance.get("build_id") != attestation["build_id"]
        or provenance.get("command_template") != command_template()
        or provenance.get("engine") != "tblite"
        or provenance.get("executable_sha256") != reference["executable_sha256"]
        or provenance.get("generation_mode") != "live-cli"
        or provenance.get("input") != case["input"]
        or provenance.get("input_sha256") != case["input_sha256"]
        or provenance.get("source_revision") != reference["revision"]
        or not isinstance(runtime, dict)
        or runtime.get("closure_sha256") != attestation["runtime"]["sha256"]
        or not isinstance(libtblite, dict)
        or libtblite.get("sha256") != reference["runtime_artifacts"]["libtblite_sha256"]
        or libtblite.get("filename")
        != attestation["artifacts"]["libtblite"]["filename"]
        or libtblite.get("discovery") != "ldd"
        or libtblite.get("status") != "resolved"
    ):
        raise PeriodicOracleError(f"{case['id']} oracle provenance mismatch")
    validated_sha256(
        provenance.get("raw_output_sha256"), f"{case['id']} raw output SHA-256"
    )
    version = provenance.get("executable_version")
    if (
        not isinstance(version, str)
        or f"tblite version {reference['version']}" not in version
    ):
        raise PeriodicOracleError(f"{case['id']} tblite version provenance drifted")
    environment = provenance.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("set")
        != {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        or not isinstance(environment.get("removed_variables"), list)
        or not isinstance(environment.get("inherited_environment_boundary"), str)
    ):
        raise PeriodicOracleError(f"{case['id']} oracle environment drifted")
    if case.get("strain_finite_difference", False):
        check_strain_evidence(golden, strain, structure, case)
    elif "strain_finite_difference" in golden:
        raise PeriodicOracleError(f"{case['id']} has unrequested strain evidence")
    if case.get("cartesian_finite_difference", False):
        check_cartesian_evidence(
            golden,
            forces,
            structure,
            finite_number(
                case["tolerances"]["cartesian_finite_difference"],
                f"{case['id']} Cartesian finite-difference tolerance",
            ),
        )
    elif "cartesian_finite_difference" in golden:
        raise PeriodicOracleError(
            f"{case['id']} has unrequested Cartesian finite differences"
        )
    if case.get("invariant_variants", False):
        check_invariant_evidence(golden, case, structure, properties)
    elif "invariant_evidence" in golden:
        raise PeriodicOracleError(f"{case['id']} has unrequested invariant evidence")
    return properties


def check(manifest_path: Path) -> None:
    """Verify all committed periodic evidence without running tblite."""
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise PeriodicOracleError(f"manifest schema must be {SCHEMA}")
    if manifest.get("units") != {
        "cell": "bohr",
        "energy": "hartree",
        "forces": "hartree_per_bohr",
        "positions": "bohr",
        "strain_derivative": "hartree",
    }:
        raise PeriodicOracleError("manifest atomic-unit contract drifted")
    reference = manifest.get("reference_engine")
    if not isinstance(reference, dict):
        raise PeriodicOracleError("manifest reference_engine must be an object")
    if reference.get("revision") != "133f91efb94b47f05848e1f86832f40a1accc385":
        raise PeriodicOracleError("tblite source revision drifted")
    if reference.get("version") != "0.7.0" or reference.get("accuracy") != ACCURACY:
        raise PeriodicOracleError("tblite version or accuracy drifted")
    if reference.get("cli_command_template") != command_template():
        raise PeriodicOracleError("tblite command template drifted")
    for name in ("executable_sha256",):
        if re.fullmatch(r"[0-9a-f]{64}", str(reference.get(name, ""))) is None:
            raise PeriodicOracleError(f"reference {name} is invalid")
    runtime = reference.get("runtime_artifacts")
    if (
        not isinstance(runtime, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("libtblite_sha256", "")))
        is None
    ):
        raise PeriodicOracleError("reference libtblite hash is invalid")
    attestation, attestation_digest = load_build_attestation(reference)
    check_source_modules(reference)

    primary_count = 0
    diagnostic_count = 0
    for case in selected_cases(manifest, None):
        if case.get("periodic_axes") != 7:
            raise PeriodicOracleError(f"{case['id']} must use XYZ periodicity")
        input_path = repository_path(case["input"])
        if sha256_file(input_path) != case.get("input_sha256"):
            raise PeriodicOracleError(f"{case['id']} input SHA-256 mismatch")
        structure = parse_turbomole(input_path)
        if len(structure["symbols"]) != int(case["atom_count"]):
            raise PeriodicOracleError(f"{case['id']} atom count mismatch")
        if structure["cell_matrix_row_major_bohr"] != finite_vector(
            case.get("cell_matrix_row_major_bohr"), 9, f"{case['id']} cell"
        ):
            raise PeriodicOracleError(f"{case['id']} cell metadata mismatch")
        role = case.get("oracle_role")
        if role == "primary-neutral-full-model":
            primary_count += 1
            if int(case["molecular_charge"]) != 0:
                raise PeriodicOracleError(
                    f"{case['id']} primary tblite cell must be neutral"
                )
        elif role == "diagnostic-unbackgrounded-charged":
            diagnostic_count += 1
            if int(case["molecular_charge"]) == 0:
                raise PeriodicOracleError(
                    f"{case['id']} charged diagnostic must be charged"
                )
        else:
            raise PeriodicOracleError(f"{case['id']} has unknown oracle role")

        golden_path = repository_path(case["golden"])
        if sha256_file(golden_path) != case.get("golden_sha256"):
            raise PeriodicOracleError(f"{case['id']} golden SHA-256 mismatch")
        golden = load_json(golden_path)
        validate_golden_document(
            case, golden, reference, attestation, attestation_digest, structure
        )
    if primary_count < 4 or diagnostic_count < 1:
        raise PeriodicOracleError("periodic corpus lacks the reviewed case roles")
    check_analytic_background(manifest)
    check_ewald_numerics_contract(manifest)
    check_ewald_reconstruction(manifest)
    sys.stdout.write(
        f"periodic GFN2 corpus check passed: {primary_count} primary, "
        f"{diagnostic_count} diagnostic, "
        f"{len(manifest['analytic_background_cases'])} analytic background\n"
    )


def compare(manifest_path: Path, actual_dir: Path, names: list[str] | None) -> None:
    """Compare independently generated normalized results with committed goldens."""
    check(manifest_path)
    manifest = load_json(manifest_path)
    reference = manifest["reference_engine"]
    attestation, attestation_digest = load_build_attestation(reference)
    compared = 0
    for case in selected_cases(manifest, names):
        if case["oracle_role"] == "analytic-background":
            continue
        expected = load_json(repository_path(case["golden"]))
        structure = parse_turbomole(repository_path(case["input"]))
        actual_path = actual_dir / f"{case['id']}.json"
        actual = load_json(actual_path)
        validate_golden_document(
            case, actual, reference, attestation, attestation_digest, structure
        )
        expected_provenance = expected["provenance"]
        actual_provenance = actual["provenance"]
        for field in (
            "build_attestation_sha256",
            "build_id",
            "command_template",
            "engine",
            "executable_sha256",
            "executable_version",
            "generation_mode",
            "input",
            "input_sha256",
            "raw_output_sha256",
            "runtime",
            "source_revision",
        ):
            if actual_provenance.get(field) != expected_provenance.get(field):
                raise PeriodicOracleError(
                    f"{case['id']} generated provenance field {field} mismatch"
                )
        tolerances = case["tolerances"]
        for property_name in (
            "energy_hartree",
            "forces_hartree_per_bohr",
            "strain_derivatives_hartree",
        ):
            expected_value = expected["properties"][property_name]
            actual_value = actual["properties"][property_name]
            expected_values = (
                expected_value if isinstance(expected_value, list) else [expected_value]
            )
            actual_values = (
                actual_value if isinstance(actual_value, list) else [actual_value]
            )
            if len(expected_values) != len(actual_values):
                raise PeriodicOracleError(
                    f"{case['id']} {property_name} extent mismatch"
                )
            tolerance = finite_number(
                tolerances[property_name], f"{case['id']} tolerance"
            )
            maximum = max(
                abs(
                    finite_number(lhs, property_name)
                    - finite_number(rhs, property_name)
                )
                for lhs, rhs in zip(expected_values, actual_values, strict=True)
            )
            if maximum > tolerance:
                raise PeriodicOracleError(
                    f"{case['id']} {property_name} maximum error "
                    f"{maximum} > {tolerance}"
                )
        if case.get("strain_finite_difference", False):
            compare_strain_evidence(case, expected, actual, structure)
        elif "strain_finite_difference" in actual:
            raise PeriodicOracleError(
                f"{case['id']} generated unexpected strain evidence"
            )
        if case.get("cartesian_finite_difference", False):
            if actual.get("cartesian_finite_difference") != expected.get(
                "cartesian_finite_difference"
            ):
                raise PeriodicOracleError(
                    f"{case['id']} Cartesian finite-difference evidence mismatch"
                )
        elif "cartesian_finite_difference" in actual:
            raise PeriodicOracleError(
                f"{case['id']} generated unexpected Cartesian evidence"
            )
        if case.get("invariant_variants", False):
            if actual.get("invariant_evidence") != expected.get("invariant_evidence"):
                raise PeriodicOracleError(
                    f"{case['id']} periodic invariant evidence mismatch"
                )
        elif "invariant_evidence" in actual:
            raise PeriodicOracleError(
                f"{case['id']} generated unexpected invariant evidence"
            )
        compared += 1
    sys.stdout.write(f"periodic GFN2 comparison passed: {compared} cases\n")


def parser() -> argparse.ArgumentParser:
    """Build the standalone periodic oracle CLI."""
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subcommands = command.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check", help="verify committed evidence offline")
    generate_parser = subcommands.add_parser(
        "generate", help="generate tblite results into a separate directory"
    )
    generate_parser.add_argument("--executable", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--case", action="append", dest="cases")
    compare_parser = subcommands.add_parser(
        "compare", help="compare a generated directory with committed goldens"
    )
    compare_parser.add_argument("--actual-dir", type=Path, required=True)
    compare_parser.add_argument("--case", action="append", dest="cases")
    capture_parser = subcommands.add_parser(
        "capture-build", help="capture one exact-source tblite build attestation"
    )
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--build-dir", type=Path, required=True)
    capture_parser.add_argument("--environment-prefix", type=Path, required=True)
    capture_parser.add_argument("--install-prefix", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    ewald_parser = subcommands.add_parser(
        "generate-ewald-reconstruction",
        help="generate the independent charged Ewald reconstruction",
    )
    ewald_parser.add_argument("--output", type=Path, required=True)
    return command


def main() -> int:
    """Run the requested periodic conformance operation."""
    arguments = parser().parse_args()
    try:
        if arguments.command == "check":
            check(arguments.manifest)
        elif arguments.command == "generate":
            generate(
                arguments.manifest,
                arguments.executable,
                arguments.output_dir,
                arguments.cases,
            )
        elif arguments.command == "compare":
            compare(arguments.manifest, arguments.actual_dir, arguments.cases)
        elif arguments.command == "capture-build":
            capture_build_attestation(
                arguments.source_root,
                arguments.build_dir,
                arguments.environment_prefix,
                arguments.install_prefix,
                arguments.output,
            )
        elif arguments.command == "generate-ewald-reconstruction":
            generate_ewald_reconstruction(arguments.manifest, arguments.output)
        else:  # pragma: no cover - argparse enforces the command set.
            raise PeriodicOracleError(f"unknown command: {arguments.command}")
    except PeriodicOracleError as exc:
        sys.stderr.write(f"periodic GFN2 oracle error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
