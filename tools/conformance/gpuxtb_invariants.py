#!/usr/bin/env python3
"""Automated invariance, conservation, and batch-consistency gates for gpuxtb.

These self-consistency checks complement, and never replace, the committed
golden comparisons: agreement with an independently generated pinned oracle
remains the acceptance test (no optimized implementation is accepted solely on
agreement with itself). The gates here instead reject physically impossible
implementations that could happen to match a single reference geometry, by
checking that the public C ABI reproduces the exact symmetries of an isolated
finite GFN2-xTB system:

* homogeneous/heterogeneous ragged-batch results match sequential solves;
* total energy, atomic charges, and forces are invariant under a rigid
  translation of the whole system (QM atoms and point charges together);
* energy and atomic charges are invariant under a proper rotation while QM and
  point-charge forces transform covariantly;
* the total force vanishes for an isolated system and the net atomic charge
  equals the declared molecular charge.

Every gate executes through the same public C ABI path as the golden runner
(shared descriptor binding, memory placement, and failure semantics), and each
backend is only ever compared with itself at transformed geometries. The
tolerances therefore measure numerical reproducibility of one backend rather
than cross-engine physics differences; the measured margins recorded below were
taken with the strict SCC solve (charge tolerance 1e-10) on the committed
8-case corpus.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Iterable, Sequence

import gpuxtb_conformance as conformance
import gpuxtb_public_api as public_api

# Measured CPU margins on the committed 8-case corpus (fresh SCC, charge
# tolerance 1e-10, 300 K electronic temperature): batch-vs-sequential and
# homogeneous replicates are bit-identical; translation changes energy by at
# most ~1.3e-14 Ha and forces by at most ~6e-15 Ha/bohr; rotation covariance
# leaves charges within ~8e-11 e and rotated forces within ~1.2e-11 Ha/bohr of
# the baseline; net forces are ~1e-17 Ha/bohr and net charges ~2e-14 e.
# Each absolute tolerance below sits 2-4 orders of magnitude above the observed
# CPU reproducibility so it also holds on CUDA (whose FMAs differ at the
# ~1e-11..1e-9 level on identical solves) while still failing a genuine physics
# error: a translation break produces errors the size of the forces themselves
# (~1e-2 Ha/bohr in the current corpus) and a rotation break produces the same
# order before covariance is applied.
INVARIANT_EXACT_ATOL = 1.0e-12  # batch/sequential and homogeneous replicate agreement.
INVARIANT_ENERGY_ATOL = 1.0e-9  # translation and rotation energy invariance (hartree).
INVARIANT_FORCE_ATOL = 1.0e-7  # translation force invariance and rotation covariance.
INVARIANT_CHARGE_ATOL = 1.0e-7  # charge invariance under rotation (elementary charge).
INVARIANT_NET_FORCE_ATOL = 1.0e-9  # isolated-system total-force conservation.
INVARIANT_NET_CHARGE_ATOL = 1.0e-9  # net charge versus declared molecular charge.

# Corpus-wide analytic-vs-numeric force gate. ``FINITE_DIFFERENCE_STEP`` is the
# central-difference displacement in bohr applied to one Cartesian coordinate
# of one atom (or one external point charge) at a time.  Measured on the
# committed 8-case corpus through the public C ABI (fresh SCC, charge tolerance
# 1e-10, 300 K): the numeric central difference agrees with the analytic force
# to about 1e-7 Ha/bohr for QM atoms and about 1e-11 Ha/bohr for point charges
# at this step, where the QM residual is dominated by SCC convergence noise
# rather than truncation error (steps of 2e-3 and 5e-3 increase it to ~1e-6).
# The tolerances below sit two to four orders above those margins so a genuine
# analytic-force defect (which is the size of the force itself, ~1e-2 Ha/bohr)
# still fails decisively on both CPU and CUDA.
FINITE_DIFFERENCE_STEP = 1.0e-3  # bohr
FINITE_DIFFERENCE_FORCE_ATOL = 1.0e-5  # QM analytic vs numeric force (Ha/bohr).
FINITE_DIFFERENCE_POINT_FORCE_ATOL = 1.0e-7  # point-charge analytic vs numeric force.

# Deterministic rigid transforms shared by every backend and run so failures are
# reproducible and comparable across machines. Rotations are proper (det = +1)
# and deliberately mix all Cartesian components; the 90-degree z rotation has
# exact 0/1 entries and exercises integer-exact covariance as well.
TRANSLATION_DELTAS = [(10.0, -7.0, 3.0), (-5.0, 2.5, 11.0)]

HOMOGENEOUS_REPLICAS = 3


@dataclass
class Geometry:
    """An in-memory copy of one corpus case with transformable positions."""

    case_id: str
    atomic_numbers: list[int]
    positions: list[float]  # flat atom-major Cartesian coordinates, bohr
    molecular_charge: int
    unpaired_electrons: int
    spin_channels: int
    point_positions: list[float] = field(default_factory=list)  # flat, bohr
    point_values: list[float] = field(default_factory=list)  # elementary charge
    point_gammas: list[float] = field(default_factory=list)  # hartree


@dataclass
class InvariantResult:
    """Per-system public outputs used by the invariance gates."""

    case_id: str
    molecular_charge: int
    energy: float
    forces: list[float]  # flat atom-major, hartree/bohr
    charges: list[float]  # elementary charge
    point_forces: list[float]  # flat point-major, hartree/bohr; empty for gas


def select_homogeneous_case_ids(
    geometries: Sequence[Geometry],
) -> tuple[str, ...]:
    """Select one gas and one point-charge case from the active case filter.

    Using the selected inputs rather than fixed corpus IDs keeps the homogeneous
    gate active for focused ``--case`` runs while bounding the full gate's cost.
    """
    selected: list[str] = []
    for has_point_charges in (False, True):
        case_id = next(
            (
                geometry.case_id
                for geometry in geometries
                if bool(geometry.point_values) == has_point_charges
            ),
            None,
        )
        if case_id is not None:
            selected.append(case_id)
    return tuple(selected)


def load_geometries(
    manifest_path: Path, manifest: dict[str, Any], cases: Sequence[dict[str, Any]]
) -> list[Geometry]:
    """Load selected corpus inputs into transformable in-memory geometries."""
    hardness = manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"]
    geometries: list[Geometry] = []
    for case in cases:
        input_path = conformance.resolve_manifest_path(manifest_path, case["input"])
        gy: dict[str, Any] = {
            "case_id": case["id"],
            "atomic_numbers": [],
            "positions": [],
            "molecular_charge": int(case["molecular_charge"]),
            "unpaired_electrons": int(case["unpaired_electrons"]),
            "spin_channels": int(case.get("spin_channels", 1)),
            "point_positions": [],
            "point_values": [],
            "point_gammas": [],
        }
        if case.get("input_schema") == "qmmm-v1":
            document = conformance.load_qmmm_input(input_path, case, hardness)
            qm = document["qm"]
            points = document["external_point_charges"]
            gy["atomic_numbers"] = [int(number) for number in qm["atomic_numbers"]]
            gy["positions"] = [
                float(value) for row in qm["positions_bohr"] for value in row
            ]
            gy["point_positions"] = [
                float(value) for row in points["positions_bohr"] for value in row
            ]
            gy["point_values"] = [float(value) for value in points["charges_e"]]
            gy["point_gammas"] = [float(value) for value in points["gammas_hartree"]]
        else:
            document = conformance.load_turbomole_coord(input_path, case)
            gy["atomic_numbers"] = document["atomic_numbers"]
            gy["positions"] = [
                float(value) for row in document["positions_bohr"] for value in row
            ]
        geometries.append(Geometry(**gy))
    return geometries


def rotate_vector(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> list[float]:
    """Apply a 3x3 matrix to a length-3 Cartesian vector."""
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def rotation_matrix(
    axis: Sequence[float] = (1.0, 1.0, 1.0), angle_degrees: float = 37.0
) -> list[list[float]]:
    """Return a deterministic proper rotation using the Rodrigues formula."""
    norm = math.sqrt(sum(component * component for component in axis))
    normalized = [component / norm for component in axis]
    ax, ay, az = normalized
    theta = math.radians(angle_degrees)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    versine = 1.0 - cosine
    return [
        [
            cosine + ax * ax * versine,
            ax * ay * versine - az * sine,
            ax * az * versine + ay * sine,
        ],
        [
            ay * ax * versine + az * sine,
            cosine + ay * ay * versine,
            ay * az * versine - ax * sine,
        ],
        [
            az * ax * versine - ay * sine,
            az * ay * versine + ax * sine,
            cosine + az * az * versine,
        ],
    ]


def _apply_to_positions(
    positions: list[float],
    vertex_count: int,
    function: Callable[[list[float]], list[float]],
) -> list[float]:
    """Apply a per-vertex Cartesian function to a flat position array."""
    transformed: list[float] = []
    for vertex in range(vertex_count):
        transformed.extend(function(positions[3 * vertex : 3 * vertex + 3]))
    return transformed


def translated(geometry: Geometry, delta: Sequence[float]) -> Geometry:
    """Return a copy displaced by one constant vector, QM atoms and points together."""
    result = copy.deepcopy(geometry)
    atom_count = len(geometry.atomic_numbers)
    point_count = len(geometry.point_values)
    result.positions = _apply_to_positions(
        geometry.positions,
        atom_count,
        lambda vector: [vector[axis] + delta[axis] for axis in range(3)],
    )
    result.point_positions = _apply_to_positions(
        geometry.point_positions,
        point_count,
        lambda vector: [vector[axis] + delta[axis] for axis in range(3)],
    )
    return result


def rotated(geometry: Geometry, matrix: Sequence[Sequence[float]]) -> Geometry:
    """Return a copy with every Cartesian coordinate (and point) rotated."""
    result = copy.deepcopy(geometry)
    result.positions = _apply_to_positions(
        geometry.positions,
        len(geometry.atomic_numbers),
        lambda vector: rotate_vector(matrix, vector),
    )
    result.point_positions = _apply_to_positions(
        geometry.point_positions,
        len(geometry.point_values),
        lambda vector: rotate_vector(matrix, vector),
    )
    return result


def _displaced_copy(
    geometry: Geometry, vertex_index: int, axis: int, delta: float, point: bool
) -> Geometry:
    """Return a copy with one atom or point-charge coordinate shifted by delta."""
    result = copy.deepcopy(geometry)
    target = result.point_positions if point else result.positions
    coordinate = 3 * vertex_index + axis
    target[coordinate] += delta
    return result


def displaced_atom(
    geometry: Geometry, atom_index: int, axis: int, delta: float
) -> Geometry:
    """Return a copy with one atom's Cartesian coordinate shifted by delta (bohr)."""
    return _displaced_copy(geometry, atom_index, axis, delta, point=False)


def displaced_point(
    geometry: Geometry, point_index: int, axis: int, delta: float
) -> Geometry:
    """Return a copy with one point charge's coordinate shifted by delta (bohr)."""
    return _displaced_copy(geometry, point_index, axis, delta, point=True)


def geometry_storage(geometries: Sequence[Geometry]) -> public_api.PublicBatchStorage:
    """Concatenate in-memory geometries into one public ragged batch storage."""
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    molecular_charges: list[float] = []
    unpaired_electrons: list[int] = []
    spin_channels: list[int] = []
    point_charge_offsets = [0]
    point_charge_positions: list[float] = []
    point_charge_values: list[float] = []
    point_charge_gammas: list[float] = []
    slices: list[public_api.CaseSlice] = []
    for geometry in geometries:
        atom_begin = len(atomic_numbers)
        point_begin = len(point_charge_values)
        atomic_numbers.extend(geometry.atomic_numbers)
        positions.extend(geometry.positions)
        molecular_charges.append(float(geometry.molecular_charge))
        unpaired_electrons.append(geometry.unpaired_electrons)
        spin_channels.append(geometry.spin_channels)
        point_charge_positions.extend(geometry.point_positions)
        point_charge_values.extend(geometry.point_values)
        point_charge_gammas.extend(geometry.point_gammas)
        atom_offsets.append(len(atomic_numbers))
        point_charge_offsets.append(len(point_charge_values))
        slices.append(
            public_api.CaseSlice(
                case={"id": geometry.case_id},
                atom_begin=atom_begin,
                atom_end=len(atomic_numbers),
                point_begin=point_begin,
                point_end=len(point_charge_values),
                expected={},
            )
        )
    return public_api.PublicBatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=molecular_charges,
        unpaired_electrons=unpaired_electrons,
        spin_channels=spin_channels,
        point_charge_offsets=point_charge_offsets,
        point_charge_positions=point_charge_positions,
        point_charge_values=point_charge_values,
        point_charge_gammas=point_charge_gammas,
        slices=slices,
        keepalive=[],
    )


def gpuxtb_solver(
    library: ctypes.CDLL,
    backend: str,
    device_id: int,
    cpu_threads: int,
    memory_mode: str,
) -> Callable[[Sequence[Geometry]], list[InvariantResult]]:
    """Return a solver that runs geometries through the public C ABI."""

    def solve(geometries: Sequence[Geometry]) -> list[InvariantResult]:
        storage = geometry_storage(geometries)
        options = public_api.pinned_compute_options(
            library,
            request_forces=True,
            request_charges=True,
            # Match the golden runner: a gas-only sequential call has no
            # point-force property or destination to request, while mixed and
            # QM/MM batches publish the complete nonempty point-force extent.
            request_point_forces=bool(storage.point_charge_values),
        )
        outputs = public_api.run_compute(
            library, storage, options, backend, device_id, cpu_threads, memory_mode
        )
        results: list[InvariantResult] = []
        for index, geometry in enumerate(geometries):
            begin = storage.slices[index].atom_begin
            end = storage.slices[index].atom_end
            point_begin = storage.slices[index].point_begin
            point_end = storage.slices[index].point_end
            assert outputs.forces is not None and outputs.charges is not None
            results.append(
                InvariantResult(
                    case_id=geometry.case_id,
                    molecular_charge=geometry.molecular_charge,
                    energy=float(outputs.energies[index]),
                    forces=[
                        float(value) for value in outputs.forces[3 * begin : 3 * end]
                    ],
                    charges=[float(value) for value in outputs.charges[begin:end]],
                    point_forces=(
                        [
                            float(value)
                            for value in outputs.point_forces[
                                3 * point_begin : 3 * point_end
                            ]
                        ]
                        if outputs.point_forces is not None
                        else []
                    ),
                )
            )
        return results

    return solve


def _compare(
    case_id: str, label: str, expected: object, actual: object, atol: float
) -> tuple[bool, str]:
    """Compare a scalar or equal-length numeric arrays and report the worst error."""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return (
                False,
                f"{case_id} {label}: shape {len(actual)} != {len(expected)}",
            )
        expected_values = [float(value) for value in expected]
        actual_values = [float(value) for value in actual]
        for index, (left, right) in enumerate(
            zip(expected_values, actual_values, strict=True)
        ):
            if not math.isfinite(left) or not math.isfinite(right):
                return (
                    False,
                    f"{case_id} {label}: non-finite component {index} "
                    f"expected={left} actual={right}",
                )
        error = max(
            (
                abs(left - right)
                for left, right in zip(expected_values, actual_values, strict=True)
            ),
            default=0.0,
        )
        location = "component"
    else:
        expected_value = float(expected)
        actual_value = float(actual)
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            return (
                False,
                f"{case_id} {label}: non-finite scalar "
                f"expected={expected_value} actual={actual_value}",
            )
        error = abs(expected_value - actual_value)
        location = "scalar"
    passed = error <= atol
    return passed, (
        f"{case_id} {label}: max_abs_error={error:.6e} limit={atol:.6e} at {location}"
    )


def _report(failures: list[str], passed: bool, message: str) -> None:
    """Emit one reproducible PASS/FAIL line and accumulate any failure."""
    print(  # noqa: T201 - CLI validation report
        ("PASS " if passed else "FAIL ") + message
    )
    if not passed:
        failures.append(message)


def gate_batch_versus_sequential(
    sequential: list[InvariantResult],
    batched: list[InvariantResult],
    atol: float,
    failures: list[str],
) -> None:
    """Require a ragged batch to preserve every per-system property."""
    for expected, actual in zip(sequential, batched, strict=True):
        for label, expected_value, actual_value in (
            ("energy_hartree", expected.energy, actual.energy),
            ("forces_hartree_per_bohr", expected.forces, actual.forces),
            ("partial_charges_e", expected.charges, actual.charges),
            (
                "point_charge_forces_hartree_per_bohr",
                expected.point_forces,
                actual.point_forces,
            ),
        ):
            passed, message = _compare(
                actual.case_id,
                f"{label} batch_vs_sequential",
                expected_value,
                actual_value,
                atol,
            )
            _report(failures, passed, message)


def gate_homogeneous_replicates(
    sequential: InvariantResult,
    replicas: list[InvariantResult],
    atol: float,
    failures: list[str],
) -> None:
    """Identical systems in one batch must reproduce the sequential result."""
    for index, replica in enumerate(replicas):
        for label, expected_value, actual_value in (
            ("energy_hartree", sequential.energy, replica.energy),
            ("forces_hartree_per_bohr", sequential.forces, replica.forces),
            ("partial_charges_e", sequential.charges, replica.charges),
            (
                "point_charge_forces_hartree_per_bohr",
                sequential.point_forces,
                replica.point_forces,
            ),
        ):
            passed, message = _compare(
                replica.case_id,
                f"{label} homogeneous_replica_{index}",
                expected_value,
                actual_value,
                atol,
            )
            _report(failures, passed, message)


def gate_translation_invariance(
    baseline: list[InvariantResult],
    translated_results: list[InvariantResult],
    atol_energy: float,
    atol_force: float,
    atol_charge: float,
    failures: list[str],
) -> None:
    """Energy, forces, and charges must not change under a rigid translation."""
    for expected, actual in zip(baseline, translated_results, strict=True):
        for label, expected_value, actual_value, tolerance in (
            ("energy_hartree", expected.energy, actual.energy, atol_energy),
            ("forces_hartree_per_bohr", expected.forces, actual.forces, atol_force),
            ("partial_charges_e", expected.charges, actual.charges, atol_charge),
            (
                "point_charge_forces_hartree_per_bohr",
                expected.point_forces,
                actual.point_forces,
                atol_force,
            ),
        ):
            passed, message = _compare(
                actual.case_id,
                f"{label} translation_invariant",
                expected_value,
                actual_value,
                tolerance,
            )
            _report(failures, passed, message)


def gate_rotation_covariance(
    baseline: list[InvariantResult],
    rotated_results: list[InvariantResult],
    matrix: Sequence[Sequence[float]],
    atol_energy: float,
    atol_force: float,
    atol_charge: float,
    label_suffix: str,
    failures: list[str],
) -> None:
    """Energy and charges are invariant; forces rotate with the structure."""
    for expected, actual in zip(baseline, rotated_results, strict=True):
        passed, message = _compare(
            actual.case_id,
            f"energy_hartree rotation_{label_suffix}",
            expected.energy,
            actual.energy,
            atol_energy,
        )
        _report(failures, passed, message)
        passed, message = _compare(
            actual.case_id,
            f"partial_charges_e rotation_{label_suffix}",
            expected.charges,
            actual.charges,
            atol_charge,
        )
        _report(failures, passed, message)
        atom_count = len(expected.charges)
        rotated_forces: list[float] = []
        for atom in range(atom_count):
            rotated_forces.extend(
                rotate_vector(matrix, expected.forces[3 * atom : 3 * atom + 3])
            )
        passed, message = _compare(
            actual.case_id,
            f"forces_hartree_per_bohr rotation_covariant_{label_suffix}",
            rotated_forces,
            actual.forces,
            atol_force,
        )
        _report(failures, passed, message)
        if expected.point_forces:
            point_count = len(expected.point_forces) // 3
            rotated_point_forces: list[float] = []
            for point in range(point_count):
                rotated_point_forces.extend(
                    rotate_vector(
                        matrix, expected.point_forces[3 * point : 3 * point + 3]
                    )
                )
            passed, message = _compare(
                actual.case_id,
                "point_charge_forces_hartree_per_bohr "
                f"rotation_covariant_{label_suffix}",
                rotated_point_forces,
                actual.point_forces,
                atol_force,
            )
            _report(failures, passed, message)


def gate_force_conservation(
    results: Sequence[InvariantResult], atol: float, failures: list[str]
) -> None:
    """Require total isolated-system force to vanish componentwise."""
    for result in results:
        axis_net: list[float] = [0.0, 0.0, 0.0]
        for atom in range(len(result.charges)):
            for axis in range(3):
                axis_net[axis] += result.forces[3 * atom + axis]
        for point in range(len(result.point_forces) // 3):
            for axis in range(3):
                axis_net[axis] += result.point_forces[3 * point + axis]
        for axis, net in enumerate(axis_net):
            passed = abs(net) <= atol
            _report(
                failures,
                passed,
                f"{result.case_id} total_force_axis_{axis}: "
                f"net={net:.6e} limit={atol:.6e}",
            )


def gate_charge_conservation(
    results: Sequence[InvariantResult], atol: float, failures: list[str]
) -> None:
    """Require summed atomic charges to reproduce the molecular charge."""
    for result in results:
        net = sum(result.charges)
        passed = abs(net - result.molecular_charge) <= atol
        _report(
            failures,
            passed,
            f"{result.case_id} net_charge: sum={net:.6e} "
            f"molecular_charge={result.molecular_charge} limit={atol:.6e}",
        )


def gate_central_finite_difference(
    solver: Callable[[Sequence[Geometry]], list[InvariantResult]],
    baseline: Sequence[InvariantResult],
    geometries: Sequence[Geometry],
    step: float,
    atol_force: float,
    atol_point_force: float,
    failures: list[str],
) -> None:
    """Compare analytic forces with central finite differences of energy.

    For every corpus case, every QM atom axis and (when present) every external
    point-charge axis is displaced by ``+-step`` in isolation and the singleton
    energy evaluates are combined into the numerical force
    ``-(E(+)-E(-)) / (2*step)``, which must match the analytic force published
    by the public C ABI for the undisplaced geometry.  This exercises the whole
    corpus (including QM/MM coupling and point-charge force signs) through the
    same descriptor path as the golden runner, complementing the analytic golden
    and self-consistency gates with a direct force-definition check.
    """
    by_id = {result.case_id: result for result in baseline}
    for geometry in geometries:
        analytic = by_id[geometry.case_id]
        for atom in range(len(geometry.atomic_numbers)):
            for axis in range(3):
                energies: list[float] = []
                for delta in (-step, step):
                    displaced = displaced_atom(geometry, atom, axis, delta)
                    single = solver([displaced])
                    if len(single) != 1 or single[0].case_id != geometry.case_id:
                        raise conformance.ConformanceError(
                            "finite-difference solve returned an unexpected "
                            f"result set for {geometry.case_id} atom {atom} axis {axis}"
                        )
                    energies.append(single[0].energy)
                numerical = -(energies[1] - energies[0]) / (2.0 * step)
                analytic_force = analytic.forces[3 * atom + axis]
                passed, message = _compare(
                    geometry.case_id,
                    f"forces_hartree_per_bohr finite_difference atom{atom}_axis{axis}",
                    [analytic_force],
                    [numerical],
                    atol_force,
                )
                _report(failures, passed, message)
        for point in range(len(geometry.point_values)):
            for axis in range(3):
                energies: list[float] = []
                for delta in (-step, step):
                    displaced = displaced_point(geometry, point, axis, delta)
                    single = solver([displaced])
                    if len(single) != 1 or single[0].case_id != geometry.case_id:
                        raise conformance.ConformanceError(
                            "finite-difference solve returned an unexpected "
                            f"result set for {geometry.case_id} point {point} "
                            f"axis {axis}"
                        )
                    energies.append(single[0].energy)
                numerical = -(energies[1] - energies[0]) / (2.0 * step)
                analytic_force = analytic.point_forces[3 * point + axis]
                label = (
                    f"point_charge_forces_hartree_per_bohr finite_difference "
                    f"point{point}_axis{axis}"
                )
                if not analytic.point_forces:
                    raise conformance.ConformanceError(
                        f"{geometry.case_id} requests point-charge forces but the "
                        "baseline result has none"
                    )
                passed, message = _compare(
                    geometry.case_id,
                    label,
                    [analytic_force],
                    [numerical],
                    atol_point_force,
                )
                _report(failures, passed, message)


def run_invariant_checks(
    solver: Callable[[Sequence[Geometry]], list[InvariantResult]],
    geometries: Sequence[Geometry],
    homogeneous_case_ids: Sequence[str],
) -> list[str]:
    """Run every symmetry gate and return the collected failure messages."""
    failures: list[str] = []
    by_id: dict[str, Geometry] = {geometry.case_id: geometry for geometry in geometries}

    print(  # noqa: T201 - CLI validation report
        f"sequential baseline: {len(geometries)} case(s)"
    )
    sequential: list[InvariantResult] = []
    for geometry in geometries:
        # A true one-system-at-a-time baseline is essential here. Repeating the
        # same ragged call would only test determinism and could miss shared
        # workspace, offset, or peer-isolation defects that appear in batches.
        single_result = solver([geometry])
        if len(single_result) != 1 or single_result[0].case_id != geometry.case_id:
            raise conformance.ConformanceError(
                "sequential invariance solve returned an unexpected result set "
                f"for {geometry.case_id}"
            )
        sequential.append(single_result[0])
    baseline_by_id = {result.case_id: result for result in sequential}

    print(  # noqa: T201 - CLI validation report
        f"heterogeneous ragged batch: {len(geometries)} case(s)"
    )
    batched = solver([by_id[result.case_id] for result in sequential])
    gate_batch_versus_sequential(sequential, batched, INVARIANT_EXACT_ATOL, failures)

    for case_id in homogeneous_case_ids:
        if case_id not in by_id:
            continue
        print(  # noqa: T201 - CLI validation report
            f"homogeneous ragged batch: {case_id} x{HOMOGENEOUS_REPLICAS}"
        )
        replicas = solver([by_id[case_id]] * HOMOGENEOUS_REPLICAS)
        gate_homogeneous_replicates(
            baseline_by_id[case_id], replicas, INVARIANT_EXACT_ATOL, failures
        )

    for delta in TRANSLATION_DELTAS:
        print(f"translation: delta={delta}")  # noqa: T201 - CLI validation report
        translated_results = solver(
            [translated(by_id[r.case_id], delta) for r in sequential]
        )
        gate_translation_invariance(
            sequential,
            translated_results,
            INVARIANT_ENERGY_ATOL,
            INVARIANT_FORCE_ATOL,
            INVARIANT_CHARGE_ATOL,
            failures,
        )

    rotations = [
        ("r37_111", rotation_matrix((1.0, 1.0, 1.0), 37.0)),
        ("r90_z", [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]
    for label, matrix in rotations:
        print(f"rotation: {label}")  # noqa: T201 - CLI validation report
        rotated_results = solver(
            [rotated(by_id[r.case_id], matrix) for r in sequential]
        )
        gate_rotation_covariance(
            sequential,
            rotated_results,
            matrix,
            INVARIANT_ENERGY_ATOL,
            INVARIANT_FORCE_ATOL,
            INVARIANT_CHARGE_ATOL,
            label,
            failures,
        )

    print("force conservation")  # noqa: T201 - CLI validation report
    gate_force_conservation(sequential, INVARIANT_NET_FORCE_ATOL, failures)
    print("charge conservation")  # noqa: T201 - CLI validation report
    gate_charge_conservation(sequential, INVARIANT_NET_CHARGE_ATOL, failures)
    print("central finite differences")  # noqa: T201 - CLI validation report
    gate_central_finite_difference(
        solver,
        sequential,
        geometries,
        FINITE_DIFFERENCE_STEP,
        FINITE_DIFFERENCE_FORCE_ATOL,
        FINITE_DIFFERENCE_POINT_FORCE_ATOL,
        failures,
    )
    return failures


def build_parser() -> argparse.ArgumentParser:
    """Define the focused local and CTest invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=conformance.DEFAULT_MANIFEST)
    parser.add_argument("--backend", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument(
        "--memory-mode",
        choices=("host", "device", "mixed"),
        default="host",
        help=(
            "descriptor placement; device and mixed require --backend cuda "
            "and dynamically load libcudart"
        ),
    )
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--skip-backend-unavailable",
        action="store_true",
        help="return 77 when an explicitly requested backend cannot create a context",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the selected cases through every invariance gate on every backend."""
    args = build_parser().parse_args(argv)
    try:
        if args.memory_mode != "host" and args.backend != "cuda":
            raise conformance.ConformanceError(
                "CPU backend only supports --memory-mode host; select "
                "--backend cuda for device or mixed descriptors"
            )
        manifest = conformance.load_json(args.manifest)
        cases = conformance.selected_cases(manifest, args.cases)
        if not cases:
            print(  # noqa: T201 - CLI validation report
                "no GFN2 conformance cases selected"
            )
            return 0
        geometries = load_geometries(args.manifest, manifest, cases)
        homogeneous_case_ids = select_homogeneous_case_ids(geometries)
        library = public_api._configure_library(args.library)
        backends = ("cpu", "cuda") if args.backend == "all" else (args.backend,)
        overall_failures: list[str] = []
        for backend in backends:
            solver = gpuxtb_solver(
                library,
                backend,
                args.device_id,
                args.cpu_threads,
                args.memory_mode,
            )
            print(  # noqa: T201 - CLI validation report
                f"invariance checks: backend={backend}, memory_mode={args.memory_mode}"
            )
            failures = run_invariant_checks(solver, geometries, homogeneous_case_ids)
            print(  # noqa: T201 - CLI validation report
                f"invariance {'OK' if not failures else 'FAILED'}: "
                f"backend={backend}, memory_mode={args.memory_mode}, "
                f"cases={len(cases)}, failures={len(failures)}"
            )
            overall_failures.extend(
                f"{backend}/{args.memory_mode}: {failure}" for failure in failures
            )
        if overall_failures:
            raise conformance.ConformanceError(
                f"{len(overall_failures)} invariance check(s) failed"
            )
    except public_api.BackendUnavailable as exc:
        print(f"SKIP {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 77 if args.skip_backend_unavailable else 1
    except conformance.ConformanceError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
