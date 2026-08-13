#!/usr/bin/env python3
"""Compare the hidden CPU GFN1 composition with the immutable oracle corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Case:
    """One normalized GFN1 request plus its independent expected properties."""

    case_id: str
    atomic_numbers: list[int]
    positions: list[float]
    charge: int
    unpaired: int
    spin_channels: int
    point_positions: list[float]
    point_charges: list[float]
    point_gammas: list[float]
    periodic_shifts: list[float]
    response_matrix: list[float]
    expected: dict[str, Any]


def flatten(values: Iterable[Any]) -> list[float]:
    """Flatten the corpus' nested Cartesian arrays without external packages."""

    flattened: list[float] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(flatten(value))
        else:
            flattened.append(float(value))
    return flattened


def load_tool(source_root: Path) -> Any:
    """Reuse the canonical corpus parser so input validation remains single-sourced."""

    path = source_root / "tools/conformance/gfn1_conformance.py"
    spec = importlib.util.spec_from_file_location("gfn1_conformance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(source_root: Path) -> tuple[list[Case], dict[str, Any]]:
    """Load all reviewed cases without copying canonical coordinates into tests."""

    tool = load_tool(source_root)
    manifest_path = source_root / "data/conformance/gfn1/manifest.json"
    tool.check_manifest(manifest_path)
    manifest = tool.load_json(manifest_path)
    symbol_to_number = {symbol: number for number, symbol in enumerate(tool.ELEMENT_SYMBOLS)}
    cases: list[Case] = []
    for record in tool.selected_cases(manifest, None):
        input_path = source_root / record["input"]
        if record.get("input_schema") == "qmmm-v1":
            document = tool.load_qmmm(
                input_path,
                record,
                manifest["reference_engines"]["xtb"]["point_charge_hardness_hartree"],
            )
            qm = document["qm"]
            point = document["external_point_charges"]
            atomic_numbers = [int(value) for value in qm["atomic_numbers"]]
            positions = flatten(qm["positions_bohr"])
            point_positions = flatten(point["positions_bohr"])
            point_charges = [float(value) for value in point["charges_e"]]
            point_gammas = [float(value) for value in point["gammas_hartree"]]
        else:
            symbols, nested_positions = tool.load_coord(input_path, record["atom_count"])
            atomic_numbers = [symbol_to_number[symbol] for symbol in symbols]
            positions = flatten(nested_positions)
            point_positions = []
            point_charges = []
            point_gammas = []
        golden = tool.load_json(source_root / record["golden"])
        cases.append(
            Case(
                case_id=record["id"],
                atomic_numbers=atomic_numbers,
                positions=positions,
                charge=int(record["molecular_charge"]),
                unpaired=int(record["unpaired_electrons"]),
                # The pinned xTB OH command uses shared orbitals; a separate
                # spin-polarized fixture exercises the two-channel path.
                spin_channels=int(record.get("spin_channels", 1)),
                point_positions=point_positions,
                point_charges=point_charges,
                point_gammas=point_gammas,
                periodic_shifts=[],
                response_matrix=[],
                expected=golden["properties"],
            )
        )
    return cases, manifest


# XTBLOOM_GFN1_FIXTURE_BEGIN gfn1-spin2-p10-tblite
def p10_case() -> Case:
    """Return tblite's independent unrestricted RSE43 P10 energy/force fixture."""

    gradient = [
        4.6250747795231898e-3,
        3.0613008404354290e-3,
        -4.1003763872489694e-17,
        -5.8127688918931083e-3,
        7.0212976481872583e-3,
        7.4006869452289192e-17,
        8.4737687859435182e-3,
        -7.8529734509620933e-3,
        1.4182424493755559e-17,
        1.6732818000371087e-4,
        -2.6987199782906278e-3,
        8.6784398428552196e-18,
        -2.4727166634656915e-3,
        1.1335882595829142e-3,
        1.1165068841281504e-17,
        -1.2010615143533908e-3,
        -5.3284767411932365e-4,
        2.1499927548673772e-3,
        -1.2010615143534362e-3,
        -5.3284767411941928e-4,
        -2.1499927548674475e-3,
        -2.5785631614047800e-3,
        4.0120202928582558e-4,
        -2.7346723399901103e-18,
    ]
    return Case(
        case_id="gfn1_spin2_p10",
        atomic_numbers=[6, 6, 8, 1, 1, 1, 1, 1],
        positions=[
            -1.97051959765227,
            -0.865723337874754,
            0.0,
            0.350984622791913,
            0.686290619844032,
            0.0,
            2.50609985217434,
            -0.934496149122418,
            0.0,
            -1.83649606109455,
            -2.90299181092583,
            0.0,
            -3.80466245712260,
            0.0349832428602470,
            0.0,
            0.373555581511497,
            1.94431040908594,
            -1.66596178649581,
            0.373555581511497,
            1.94431040908594,
            1.66596178649581,
            4.00748247788016,
            0.0933166170468600,
            0.0,
        ],
        charge=0,
        unpaired=1,
        spin_channels=2,
        point_positions=[],
        point_charges=[],
        point_gammas=[],
        periodic_shifts=[],
        response_matrix=[],
        expected={
            "energy_hartree": -11.539671328635730,
            "forces_hartree_per_bohr": [-value for value in gradient],
        },
    )
# XTBLOOM_GFN1_FIXTURE_END gfn1-spin2-p10-tblite


def encode(requests: list[list[Case]]) -> str:
    """Serialize deterministic request batches understood by the C++ probe."""

    fields = ["XTBLOOM_GFN1_PROBE_V2", str(len(requests))]
    for cases in requests:
        fields.append(str(len(cases)))
        for case in cases:
            atoms = len(case.atomic_numbers)
            periodic = bool(case.periodic_shifts or case.response_matrix)
            if periodic and (
                len(case.periodic_shifts) != atoms
                or len(case.response_matrix) != atoms * atoms
            ):
                raise ValueError(f"{case.case_id}: invalid periodic operator shape")
            fields.extend(
                (
                    str(atoms),
                    str(len(case.point_charges)),
                    str(case.charge),
                    str(case.unpaired),
                    str(case.spin_channels),
                    "1" if periodic else "0",
                )
            )
            fields.extend(str(value) for value in case.atomic_numbers)
            fields.extend(format(value, ".17g") for value in case.positions)
            fields.extend(format(value, ".17g") for value in case.point_positions)
            fields.extend(format(value, ".17g") for value in case.point_charges)
            fields.extend(format(value, ".17g") for value in case.point_gammas)
            if periodic:
                fields.extend(format(value, ".17g") for value in case.periodic_shifts)
                fields.extend(format(value, ".17g") for value in case.response_matrix)
    return "\n".join(fields) + "\n"


def run_requests(probe: Path, requests: list[list[Case]]) -> list[dict[str, Any]]:
    """Execute request batches in one process while preserving FRESH SCC state."""

    completed = subprocess.run(
        [str(probe)],
        input=encode(requests),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GFN1 probe exited {completed.returncode}: {completed.stderr.strip()}"
        )
    results = json.loads(completed.stdout)
    if not isinstance(results, list) or len(results) != len(requests):
        raise RuntimeError("GFN1 probe returned an unexpected request count")
    for cases, result in zip(requests, results, strict=True):
        if result["statuses"] != [0] * len(cases) or result["converged"] != [1] * len(cases):
            raise RuntimeError(
                f"GFN1 systems did not converge: statuses={result['statuses']} "
                f"converged={result['converged']} iterations={result['iterations']}"
            )
    return results


def run_probe(probe: Path, cases: list[Case]) -> dict[str, Any]:
    """Execute one request batch."""

    return run_requests(probe, [cases])[0]


def maximum_error(actual: Any, expected: Any) -> float:
    """Return an absolute maximum error while rejecting non-finite results."""

    actual_flat = [float(actual)] if not isinstance(actual, list) else flatten(actual)
    expected_flat = [float(expected)] if not isinstance(expected, list) else flatten(expected)
    if len(actual_flat) != len(expected_flat):
        return math.inf
    return max(
        (
            abs(left - right)
            if math.isfinite(left) and math.isfinite(right)
            else math.inf
            for left, right in zip(actual_flat, expected_flat, strict=True)
        ),
        default=0.0,
    )


def slices(cases: list[Case], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one ragged result into the property shapes used by each golden."""

    split: list[dict[str, Any]] = []
    atom_begin = 0
    point_begin = 0
    for index, case in enumerate(cases):
        atoms = len(case.atomic_numbers)
        points = len(case.point_charges)
        split.append(
            {
                "energy_hartree": result["energies"][index],
                "forces_hartree_per_bohr": result["forces"][
                    3 * atom_begin : 3 * (atom_begin + atoms)
                ],
                "partial_charges_e": result["atomic_charges"][
                    atom_begin : atom_begin + atoms
                ],
                "point_charge_forces_hartree_per_bohr": result[
                    "point_charge_forces"
                ][3 * point_begin : 3 * (point_begin + points)],
            }
        )
        atom_begin += atoms
        point_begin += points
    return split


def compare_oracles(
    cases: list[Case], actual: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[str]:
    """Apply the immutable manifest's primary-oracle tolerances unchanged."""

    failures: list[str] = []
    tolerance_names = {
        "energy_hartree": "energy",
        "forces_hartree_per_bohr": "forces",
        "partial_charges_e": "charges",
        "point_charge_forces_hartree_per_bohr": "point_charge_forces",
    }
    for case, properties in zip(cases, actual, strict=True):
        for name, expected in case.expected.items():
            if name not in tolerance_names:
                continue
            error = maximum_error(properties[name], expected)
            limit = float(manifest["tolerances"][tolerance_names[name]]["atol"])
            if error > limit:
                failures.append(f"{case.case_id}: {name} error {error:.3e} > {limit:.3e}")
    return failures


def add_request(
    requests: list[list[Case]], metadata: list[tuple[str, int, int, float]], case: Case,
    coordinate_kind: int, coordinate: int, step: float
) -> None:
    """Append the plus/minus pair for one Cartesian central difference."""

    values = case.positions if coordinate_kind == 0 else case.point_positions
    for sign in (1.0, -1.0):
        displaced = list(values)
        displaced[coordinate] += sign * step
        requests.append(
            [
                replace(
                    case,
                    positions=displaced if coordinate_kind == 0 else case.positions,
                    point_positions=(
                        displaced if coordinate_kind == 1 else case.point_positions
                    ),
                )
            ]
        )
    metadata.append((case.case_id, coordinate_kind, coordinate, step))


def finite_difference_checks(
    probe: Path, cases: list[Case], tolerance: float = 5.0e-7
) -> list[str]:
    """Check total SCC forces at one full and two representative step sizes."""

    requests: list[list[Case]] = []
    metadata: list[tuple[str, int, int, float]] = []
    for case in cases:
        for coordinate in range(len(case.positions)):
            for step in (2.0e-4, 1.0e-4, 5.0e-5):
                add_request(requests, metadata, case, 0, coordinate, step)
        if case.point_positions:
            for coordinate in range(len(case.point_positions)):
                for step in (2.0e-4, 1.0e-4, 5.0e-5):
                    add_request(requests, metadata, case, 1, coordinate, step)

    results = run_requests(probe, requests)
    failures: list[str] = []
    maximum = 0.0
    for pair, (case_id, coordinate_kind, coordinate, step) in enumerate(metadata):
        plus = results[2 * pair]
        minus = results[2 * pair + 1]
        numerical = -(plus["energies"][0] - minus["energies"][0]) / (2.0 * step)
        field = "forces" if coordinate_kind == 0 else "point_charge_forces"
        # Compare at the central geometry. Averaging the analytic forces from
        # the symmetric endpoints cancels their first-order displacement.
        analytic = 0.5 * (plus[field][coordinate] + minus[field][coordinate])
        error = abs(numerical - analytic)
        maximum = max(maximum, error)
        if not math.isfinite(error) or error > tolerance:
            kind = "QM" if coordinate_kind == 0 else "point"
            failures.append(
                f"{case_id}: {kind} coordinate {coordinate} h={step:.1e} "
                f"total-force FD error {error:.3e} > {tolerance:.3e}"
            )
    print(f"GFN1 total-SCC finite differences: max error {maximum:.3e} Eh/bohr")
    return failures


def rotate_vectors(values: list[float], matrix: list[list[float]]) -> list[float]:
    """Apply one proper Cartesian rotation to a flat vector list."""

    rotated: list[float] = []
    for begin in range(0, len(values), 3):
        vector = values[begin : begin + 3]
        rotated.extend(
            sum(matrix[row][column] * vector[column] for column in range(3))
            for row in range(3)
        )
    return rotated


def rotation_matrix() -> list[list[float]]:
    """Return the fixed 37-degree proper rotation used by covariance checks."""

    axis = [1.0, 2.0, -1.0]
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = (value / norm for value in axis)
    angle = math.radians(37.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return [
        [cosine + x * x * complement, x * y * complement - z * sine,
         x * z * complement + y * sine],
        [y * x * complement + z * sine, cosine + y * y * complement,
         y * z * complement - x * sine],
        [z * x * complement - y * sine, z * y * complement + x * sine,
         cosine + z * z * complement],
    ]


def covariance_and_conservation_checks(probe: Path, cases: list[Case]) -> list[str]:
    """Check rigid covariance plus isolated-system force and torque conservation."""

    translation = [7.25, -3.5, 2.125]
    rotation = rotation_matrix()
    requests: list[list[Case]] = []
    for case in cases:
        translated_positions = [
            value + translation[index % 3] for index, value in enumerate(case.positions)
        ]
        translated_points = [
            value + translation[index % 3]
            for index, value in enumerate(case.point_positions)
        ]
        requests.extend(
            [
                [case],
                [replace(case, positions=translated_positions,
                         point_positions=translated_points)],
                [replace(case, positions=rotate_vectors(case.positions, rotation),
                         point_positions=rotate_vectors(case.point_positions, rotation))],
            ]
        )
    results = run_requests(probe, requests)
    failures: list[str] = []
    for index, case in enumerate(cases):
        original, translated, rotated = results[3 * index : 3 * index + 3]
        if abs(original["energies"][0] - translated["energies"][0]) > 1.0e-9:
            failures.append(f"{case.case_id}: translation changed energy")
        if maximum_error(original["forces"], translated["forces"]) > 1.0e-7:
            failures.append(f"{case.case_id}: translation changed QM forces")
        if maximum_error(
            original["point_charge_forces"], translated["point_charge_forces"]
        ) > 1.0e-7:
            failures.append(f"{case.case_id}: translation changed point forces")
        if maximum_error(original["atomic_charges"], translated["atomic_charges"]) > 1.0e-7:
            failures.append(f"{case.case_id}: translation changed charges")
        if abs(original["energies"][0] - rotated["energies"][0]) > 1.0e-9:
            failures.append(f"{case.case_id}: rotation changed energy")
        if maximum_error(rotate_vectors(original["forces"], rotation), rotated["forces"]) > 1.0e-7:
            failures.append(f"{case.case_id}: QM-force rotation covariance failed")
        if maximum_error(
            rotate_vectors(original["point_charge_forces"], rotation),
            rotated["point_charge_forces"],
        ) > 1.0e-7:
            failures.append(f"{case.case_id}: point-force rotation covariance failed")
        if maximum_error(original["atomic_charges"], rotated["atomic_charges"]) > 1.0e-7:
            failures.append(f"{case.case_id}: rotation changed charges")

        total_force = [0.0, 0.0, 0.0]
        torque = [0.0, 0.0, 0.0]
        for positions, forces in (
            (case.positions, original["forces"]),
            (case.point_positions, original["point_charge_forces"]),
        ):
            for begin in range(0, len(positions), 3):
                x, y, z = positions[begin : begin + 3]
                fx, fy, fz = forces[begin : begin + 3]
                total_force[0] += fx
                total_force[1] += fy
                total_force[2] += fz
                torque[0] += y * fz - z * fy
                torque[1] += z * fx - x * fz
                torque[2] += x * fy - y * fx
        if max(abs(value) for value in total_force) > 1.0e-9:
            failures.append(f"{case.case_id}: total force is not conserved")
        if max(abs(value) for value in torque) > 1.0e-7:
            failures.append(f"{case.case_id}: total torque is not conserved")
        charge_error = abs(sum(original["atomic_charges"]) - case.charge)
        if charge_error > 5.0e-7:
            failures.append(f"{case.case_id}: atomic charge is not conserved")
    return failures


def periodic_case(case: Case) -> Case:
    """Attach a fixed, symmetric b+A*q operator without coordinate derivatives."""

    atoms = len(case.atomic_numbers)
    shifts = [0.013 * ((index % 3) - 1) for index in range(atoms)]
    response = [0.0] * (atoms * atoms)
    for row in range(atoms):
        response[row * atoms + row] = 0.02 + 0.003 * row
        for column in range(row):
            value = 0.001 * (1 + ((row + column) % 3))
            response[row * atoms + column] = value
            response[column * atoms + row] = value
    return replace(case, case_id=f"{case.case_id}_periodic", periodic_shifts=shifts,
                   response_matrix=response)


def ragged_replication_checks(probe: Path, cases: list[Case]) -> list[str]:
    """Compare true homogeneous replicated batches with independent singleton results."""

    results = run_requests(probe, [[case] for case in cases] + [[case, case, case] for case in cases])
    failures: list[str] = []
    for index, case in enumerate(cases):
        singleton = slices([case], results[index])[0]
        replicated = slices([case, case, case], results[len(cases) + index])
        for replica in replicated:
            for name, expected in singleton.items():
                if maximum_error(replica[name], expected) > 1.0e-12:
                    failures.append(f"{case.case_id}: homogeneous ragged {name} differs")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    cases, manifest = load_cases(args.source_root.resolve())
    cases.append(p10_case())
    singleton_results = [slices([case], run_probe(args.probe, [case]))[0] for case in cases]
    ragged_results = slices(cases, run_probe(args.probe, cases))
    failures = compare_oracles(cases, singleton_results, manifest)
    failures.extend(compare_oracles(cases, ragged_results, manifest))

    # A ragged batch owns independent one-system executors.  Comparing it to
    # separately initialized processes catches topology slicing and peer-state
    # contamination independently of the external oracle gate.
    for case, singleton, ragged in zip(cases, singleton_results, ragged_results, strict=True):
        for name in singleton:
            error = maximum_error(ragged[name], singleton[name])
            if error > 1.0e-12:
                failures.append(
                    f"{case.case_id}: ragged/sequential {name} error {error:.3e} > 1.000e-12"
                )
    selected = {
        case.case_id: case
        for case in cases
        if case.case_id
        in {
            "gfn1_ketene",
            "gfn1_oh_radical",
            "gfn1_spin2_p10",
            "gfn1_halogen_bond",
            "gfn1_water_dimer_6pc_hardness",
            "gfn1_water_dimer_6pc_gamma999",
        }
    }
    scientific_cases = list(selected.values())
    failures.extend(finite_difference_checks(args.probe, scientific_cases))
    failures.extend(covariance_and_conservation_checks(args.probe, scientific_cases))
    failures.extend(
        ragged_replication_checks(
            args.probe,
            [selected["gfn1_ketene"], selected["gfn1_water_dimer_6pc_hardness"]],
        )
    )
    periodic = periodic_case(selected["gfn1_ketene"])
    periodic_result = run_probe(args.probe, [periodic])
    if periodic_result["flags"] != 1:
        failures.append("fixed periodic operator did not publish the external-derivative flag")
    failures.extend(finite_difference_checks(args.probe, [periodic]))
    if failures:
        print("GFN1 hidden CPU conformance failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        f"GFN1 hidden CPU scientific acceptance OK: {len(cases)} singleton, "
        "heterogeneous/homogeneous ragged, unrestricted, FD, covariance, "
        "conservation, point-charge, and fixed-periodic gates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
