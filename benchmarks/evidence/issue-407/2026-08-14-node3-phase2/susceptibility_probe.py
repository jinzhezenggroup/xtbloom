#!/usr/bin/env python3
"""Compare converged public dq/db with the Phase-2 bare QEq response proxy.

The finite differences perturb the public caller-owned atomic potential ``b``
with a zero response matrix. The proxy is the exact atom projection of the
prototype's frozen shell-diagonal map

    C = D - D 1 1^T D / (1^T D 1),  D_ss = alpha / gamma_s,

and reports ``-T C T^T`` beside the fully reconverged public SCC derivative.
The latter also contains self-consistent ES2/AES2/band response and is not the
bare electronic susceptibility ``-T C T^T``. The implemented pair operator is
screened as well, so this comparison is descriptive and non-decisive: a
mismatch neither identifies alpha nor validates/invalidates this candidate or
the wider shell-response family. The measurement uses the established SCC
fixed point with acceleration disabled, so it does not compare the prototype
against itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PAIR_RESPONSE_SCALE = 0.08


@dataclass(frozen=True)
class Case:
    """One public GFN2 response probe and its atom-space tangent directions."""

    case_id: str
    numbers: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    directions: tuple[tuple[float, ...], ...]
    charge: float = 0.0
    uhf: int = 0
    spin_channels: int = 1


CASES = (
    Case(
        "h2",
        (1, 1),
        ((-0.71, 0.0, 0.0), (0.71, 0.0, 0.0)),
        ((-1.0, 1.0),),
    ),
    Case(
        "co",
        (6, 8),
        ((-1.1, 0.0, 0.0), (1.1, 0.0, 0.0)),
        ((-1.0, 1.0),),
    ),
    Case(
        "water",
        (8, 1, 1),
        ((0.0, 0.0, 0.0), (1.4, 0.0, 1.1), (-1.4, 0.0, 1.1)),
        ((2.0, -1.0, -1.0), (0.0, 1.0, -1.0)),
    ),
    Case(
        "oxygen_triplet",
        (8, 8),
        ((-1.15, 0.0, 0.0), (1.15, 0.0, 0.0)),
        ((-1.0, 1.0),),
        uhf=2,
        spin_channels=2,
    ),
)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repository: Path, *args: str) -> str:
    """Run one read-only Git query."""
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def normalized_tangent(values: tuple[float, ...]) -> np.ndarray:
    """Project one atom-space vector to zero mean and unit Euclidean norm."""
    direction = np.asarray(values, dtype=np.float64)
    direction -= np.mean(direction)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("response direction must have a finite nonzero tangent norm")
    return direction / norm


def parameter_index(parameter_path: Path) -> dict[int, dict[str, Any]]:
    """Index the canonical generated GFN2 JSON by atomic number."""
    document = json.loads(parameter_path.read_text(encoding="utf-8"))
    return {int(element["atomic_number"]): element for element in document["elements"]}


def predicted_atomic_response(
    numbers: tuple[int, ...],
    direction: np.ndarray,
    parameters: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Project the frozen shell-diagonal QEq proxy to atomic charges."""
    shell_atoms: list[int] = []
    shell_response: list[float] = []
    for atom, atomic_number in enumerate(numbers):
        element = parameters[atomic_number]
        element_hardness = float(element["gam"])
        for shell in element["shells"]:
            hardness = element_hardness * float(shell["shell_hubbard_scale"])
            shell_atoms.append(atom)
            shell_response.append(PAIR_RESPONSE_SCALE / hardness)
    atom_index = np.asarray(shell_atoms, dtype=np.int64)
    diagonal = np.asarray(shell_response, dtype=np.float64)
    shell_potential = direction[atom_index]
    weighted_mean = float(np.dot(diagonal, shell_potential) / np.sum(diagonal))
    shell_charge_response = -diagonal * (shell_potential - weighted_mean)
    result = np.zeros(len(numbers), dtype=np.float64)
    np.add.at(result, atom_index, shell_charge_response)
    return result


def response_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float] | None:
    """Return scale and direction diagnostics for two tangent responses."""
    actual_norm = float(np.linalg.norm(actual))
    predicted_norm = float(np.linalg.norm(predicted))
    if (
        not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(predicted))
        or not math.isfinite(actual_norm)
        or not math.isfinite(predicted_norm)
        or actual_norm == 0.0
        or predicted_norm == 0.0
    ):
        return None
    cosine = float(np.dot(actual, predicted) / (actual_norm * predicted_norm))
    if not math.isfinite(cosine):
        return None
    return {
        "actual_norm_e_per_hartree": actual_norm,
        "predicted_norm_e_per_hartree": predicted_norm,
        "direction_cosine": cosine,
        "actual_to_predicted_norm_ratio": actual_norm / predicted_norm,
        "relative_vector_error": float(
            np.linalg.norm(actual - predicted) / actual_norm
        ),
        "charge_sum_e_per_hartree": float(np.sum(actual)),
    }


def main() -> int:
    """Run the finite-difference corpus and write a reproducible JSON record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", type=float, nargs="+", default=(1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
    )
    args = parser.parse_args()

    repository = args.repository.resolve()
    library = args.library.resolve()
    output = args.output.resolve()
    steps = tuple(float(value) for value in args.steps)
    if not steps or any(not math.isfinite(value) or value <= 0.0 for value in steps):
        parser.error("--steps must contain finite positive Hartree values")

    os.environ["XTBLOOM_LIBRARY"] = str(library)
    os.environ["XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC"] = "off"
    from xtbloom import BatchCalculator, ChargeResponse, Structure, __version__

    parameter_path = repository / "data/parameters/gfn2.json"
    parameters = parameter_index(parameter_path)
    rows: list[dict[str, Any]] = []
    for case in CASES:
        numbers = np.asarray(case.numbers, dtype=np.int32)
        positions = np.asarray(case.positions, dtype=np.float64)
        for direction_index, raw_direction in enumerate(case.directions):
            direction = normalized_tangent(raw_direction)
            predicted = predicted_atomic_response(case.numbers, direction, parameters)
            samples: list[dict[str, Any]] = []
            actual_vectors: list[np.ndarray | None] = []
            for step in steps:
                charges: list[np.ndarray] = []
                iterations: list[int] = []
                statuses: list[int] = []
                converged: list[bool] = []
                for sign in (1.0, -1.0):
                    response = ChargeResponse(
                        shifts=sign * step * direction,
                        matrix=np.zeros((numbers.size, numbers.size), dtype=np.float64),
                    )
                    structure = Structure(
                        numbers,
                        positions,
                        charge=case.charge,
                        uhf=case.uhf,
                        spin_channels=case.spin_channels,
                        charge_response=response,
                    )
                    calculator = BatchCalculator(
                        [structure],
                        method="GFN2-xTB",
                        backend="cpu",
                        cpu_threads=1,
                        max_scc_iterations=500,
                        charge_tolerance=1.0e-10,
                        energy_tolerance=1.0e-12,
                        electronic_temperature=300.0,
                        warm_start=False,
                    )
                    result = calculator.compute(raise_on_failure=False)
                    charges.append(np.asarray(result.charges, dtype=np.float64).copy())
                    iterations.append(int(result.scc_iterations[0]))
                    statuses.append(int(result.per_system_status[0]))
                    converged.append(bool(result.scc_converged[0]))
                sample_is_valid = (
                    all(converged)
                    and all(status == 0 for status in statuses)
                    and all(np.all(np.isfinite(values)) for values in charges)
                )
                actual = (
                    (charges[0] - charges[1]) / (2.0 * step)
                    if sample_is_valid
                    else None
                )
                metrics = (
                    response_metrics(actual, predicted) if actual is not None else None
                )
                if metrics is None:
                    actual = None
                actual_vectors.append(actual)
                samples.append(
                    {
                        "step_hartree": step,
                        "converged": converged,
                        "per_system_status": statuses,
                        "scc_iterations": iterations,
                        "eligible_for_descriptive_metrics": actual is not None,
                        "actual_atomic_response_e_per_hartree": (
                            actual.tolist() if actual is not None else None
                        ),
                        "metrics": metrics,
                    }
                )
            valid_indices = [
                index
                for index, actual in enumerate(actual_vectors)
                if actual is not None
            ]
            reference_index = (
                min(valid_indices, key=lambda index: steps[index])
                if valid_indices
                else None
            )
            reference = (
                actual_vectors[reference_index] if reference_index is not None else None
            )
            for sample, actual in zip(samples, actual_vectors, strict=True):
                sample["difference_from_smallest_valid_step_norm_e_per_hartree"] = (
                    float(np.linalg.norm(actual - reference))
                    if actual is not None and reference is not None
                    else None
                )
            rows.append(
                {
                    "case_id": case.case_id,
                    "direction_index": direction_index,
                    "numbers": list(case.numbers),
                    "positions_bohr": [list(row) for row in case.positions],
                    "charge": case.charge,
                    "uhf": case.uhf,
                    "spin_channels": case.spin_channels,
                    "normalized_atomic_potential_direction": direction.tolist(),
                    "predicted_atomic_response_e_per_hartree": predicted.tolist(),
                    "smallest_valid_step_index": reference_index,
                    "samples": samples,
                }
            )

    smallest_step_metrics = [
        row["samples"][row["smallest_valid_step_index"]]["metrics"]
        for row in rows
        if row["smallest_valid_step_index"] is not None
    ]
    document = {
        "schema_version": 1,
        "claim": (
            "Descriptively compare the frozen bare QEq proxy with fully reconverged public "
            "dq/db; the mismatch is non-decisive and is not a susceptibility fit"
        ),
        "model": {
            "alpha": PAIR_RESPONSE_SCALE,
            "bare_qeq_proxy": "D - D11^TD/(1^TD1), D_ss=alpha/gamma_s",
            "measured_policy": "off",
            "interpretation_limit": (
                "public dq/db includes self-consistent ES2/AES2/band response and is not "
                "the bare -TCT^T electronic susceptibility; this script also does not "
                "evaluate the implemented screened pair operator"
            ),
        },
        "metadata": {
            "repository": str(repository),
            "git_revision": git_output(repository, "rev-parse", "HEAD"),
            "git_status_porcelain": git_output(repository, "status", "--porcelain"),
            "library": str(library),
            "library_sha256": sha256(library),
            "parameter_file": str(parameter_path.relative_to(repository)),
            "parameter_sha256": sha256(parameter_path),
            "xtbloom_python_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "steps_hartree": list(steps),
        },
        "summary_at_smallest_step": {
            "eligible_row_count": len(smallest_step_metrics),
            "total_row_count": len(rows),
            "minimum_direction_cosine": (
                min(
                    float(metrics["direction_cosine"])
                    for metrics in smallest_step_metrics
                )
                if smallest_step_metrics
                else None
            ),
            "minimum_actual_to_predicted_norm_ratio": (
                min(
                    float(metrics["actual_to_predicted_norm_ratio"])
                    for metrics in smallest_step_metrics
                )
                if smallest_step_metrics
                else None
            ),
            "maximum_actual_to_predicted_norm_ratio": (
                max(
                    float(metrics["actual_to_predicted_norm_ratio"])
                    for metrics in smallest_step_metrics
                )
                if smallest_step_metrics
                else None
            ),
            "maximum_relative_vector_error": (
                max(
                    float(metrics["relative_vector_error"])
                    for metrics in smallest_step_metrics
                )
                if smallest_step_metrics
                else None
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
