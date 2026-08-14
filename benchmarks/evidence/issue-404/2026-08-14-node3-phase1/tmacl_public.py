#!/usr/bin/env python3
"""Run the issue #404 TMAC/Cl case through xTBloom's public Python/C ABI.

The experimental SCC policy is process-global and frozen when the CPU context
is created, so each invocation evaluates exactly one policy/iteration ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

ANGSTROM_PER_BOHR = 0.529177210903


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one evidence input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(repository: Path, *args: str) -> str:
    """Run a read-only Git query against the measured source worktree."""
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def parse_xyz(path: Path, symbol_to_number: dict[str, int]) -> tuple[Any, Any]:
    """Load and validate the exact issue #404 TMAC/Cl fixture."""
    import numpy as np

    lines = path.read_text(encoding="ascii").splitlines()
    natoms = int(lines[0])
    if natoms != 18 or lines[1] != "tmacl" or len(lines) != natoms + 2:
        raise ValueError("expected the exact 18-atom tmacl XYZ envelope")
    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4 or fields[0] not in symbol_to_number:
            raise ValueError(f"malformed XYZ row: {line!r}")
        numbers.append(symbol_to_number[fields[0]])
        positions.append([float(value) / ANGSTROM_PER_BOHR for value in fields[1:]])
    return np.asarray(numbers, dtype=np.int32), np.asarray(positions, dtype=np.float64)


def finite_or_none(value: float) -> float | None:
    """Convert non-finite publication values to strict JSON nulls."""
    return float(value) if math.isfinite(float(value)) else None


def vector_or_none(values: Iterable[float]) -> list[float | None]:
    """Convert a public floating-point vector to strict JSON values."""
    return [finite_or_none(value) for value in values]


def main() -> int:
    """Execute one frozen-policy public-ABI calculation and write JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("off", "controller", "local-v1"), required=True
    )
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--finite-difference-step", type=float)
    parser.add_argument("--finite-difference-max-iterations", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    library_path = args.library.resolve()
    input_path = args.input.resolve()
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")

    os.environ["XTBLOOM_LIBRARY"] = str(library_path)
    os.environ["XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC"] = args.policy

    # Import only after the exact native library and frozen SCC policy are set.
    from xtbloom import library as xtbloom_library
    from xtbloom.interface import (
        SYMBOL_TO_NUMBER,
        BatchCalculator,
        Structure,
    )

    numbers, positions = parse_xyz(input_path, SYMBOL_TO_NUMBER)
    structure = Structure(
        numbers,
        positions,
        charge=0.0,
        uhf=0,
        spin_channels=1,
    )
    calculator = BatchCalculator(
        [structure],
        method="GFN2-xTB",
        backend="cpu",
        cpu_threads=1,
        max_scc_iterations=args.max_iterations,
        charge_tolerance=1.0e-6,
        energy_tolerance=1.0e-8,
        electronic_temperature=300.0,
        warm_start=False,
    )
    result = calculator.compute(raise_on_failure=False)
    atomic_charges = [finite_or_none(value) for value in result.charges]
    forces = vector_or_none(result.forces.reshape(-1))
    converged = bool(result.scc_converged[0])
    status = int(result.per_system_status[0])
    finite_difference = None
    if args.finite_difference_step is not None:
        import numpy as np

        if not converged:
            raise RuntimeError("finite differences require a converged base solve")
        step = float(args.finite_difference_step)
        if not math.isfinite(step) or step <= 0.0:
            parser.error("--finite-difference-step must be finite and positive")
        fd_max_iterations = (
            args.finite_difference_max_iterations
            if args.finite_difference_max_iterations is not None
            else args.max_iterations
        )
        if fd_max_iterations <= 0:
            parser.error("--finite-difference-max-iterations must be positive")
        fd_structure = Structure(
            numbers,
            positions.copy(),
            charge=0.0,
            uhf=0,
            spin_channels=1,
        )
        fd_calculator = BatchCalculator(
            [fd_structure],
            method="GFN2-xTB",
            backend="cpu",
            cpu_threads=1,
            max_scc_iterations=fd_max_iterations,
            charge_tolerance=1.0e-6,
            energy_tolerance=1.0e-8,
            electronic_temperature=300.0,
            warm_start=False,
        )
        numerical = np.empty(positions.size, dtype=np.float64)
        failure = None
        for coordinate in range(positions.size):
            plus = positions.copy().reshape(-1)
            plus[coordinate] += step
            fd_structure.update(positions=plus.reshape(positions.shape))
            plus_result = fd_calculator.compute(raise_on_failure=False)
            if not bool(plus_result.scc_converged[0]):
                failure = {
                    "coordinate": coordinate,
                    "displacement": "plus",
                    "per_system_status": int(plus_result.per_system_status[0]),
                    "per_system_status_name": xtbloom_library.status_string(
                        int(plus_result.per_system_status[0])
                    ),
                    "scc_iterations": int(plus_result.scc_iterations[0]),
                }
                break
            energy_plus = float(plus_result.energies[0])
            minus = positions.copy().reshape(-1)
            minus[coordinate] -= step
            fd_structure.update(positions=minus.reshape(positions.shape))
            minus_result = fd_calculator.compute(raise_on_failure=False)
            if not bool(minus_result.scc_converged[0]):
                failure = {
                    "coordinate": coordinate,
                    "displacement": "minus",
                    "per_system_status": int(minus_result.per_system_status[0]),
                    "per_system_status_name": xtbloom_library.status_string(
                        int(minus_result.per_system_status[0])
                    ),
                    "scc_iterations": int(minus_result.scc_iterations[0]),
                }
                break
            energy_minus = float(minus_result.energies[0])
            numerical[coordinate] = -(energy_plus - energy_minus) / (2.0 * step)
        net_force = np.sum(result.forces, axis=0)
        torque = np.sum(np.cross(positions, result.forces), axis=0)
        finite_difference = {
            "step_bohr": step,
            "coordinates": int(positions.size),
            "max_scc_iterations": fd_max_iterations,
            "complete": failure is None,
            "failure": failure,
            "net_force_hartree_per_bohr": [float(value) for value in net_force],
            "net_force_norm_hartree_per_bohr": float(np.linalg.norm(net_force)),
            "torque_hartree": [float(value) for value in torque],
            "torque_norm_hartree": float(np.linalg.norm(torque)),
        }
        if failure is None:
            analytic = result.forces.reshape(-1)
            absolute_error = np.abs(analytic - numerical)
            finite_difference.update(
                {
                    "maximum_absolute_error_hartree_per_bohr": float(
                        np.max(absolute_error)
                    ),
                    "rms_error_hartree_per_bohr": float(
                        np.sqrt(np.mean(absolute_error**2))
                    ),
                }
            )
    document = {
        "schema_version": 1,
        "case": {
            "id": "tmacl",
            "input": str(input_path.relative_to(repository)),
            "input_sha256": sha256(input_path),
            "natoms": int(numbers.size),
            "charge": 0.0,
            "unpaired_electrons": 0,
            "spin_channels": 1,
        },
        "source": {
            "revision": git_output(repository, "rev-parse", "HEAD"),
            "dirty": bool(git_output(repository, "status", "--porcelain")),
        },
        "library": {
            "path": str(library_path),
            "sha256": sha256(library_path.resolve()),
        },
        "protocol": {
            "backend": "cpu",
            "memory_mode": "host",
            "start_mode": "fresh",
            "policy": args.policy,
            "max_scc_iterations": args.max_iterations,
            "charge_tolerance": 1.0e-6,
            "energy_tolerance_hartree": 1.0e-8,
            "electronic_temperature_kelvin": 300.0,
            "cpu_threads": 1,
            "process_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "GOMP_CPU_AFFINITY",
                )
            },
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "result": {
            "per_system_status": status,
            "per_system_status_name": xtbloom_library.status_string(status),
            "scc_converged": converged,
            "scc_iterations": int(result.scc_iterations[0]),
            "energy_hartree": finite_or_none(result.energies[0]),
            "atomic_charges_e": atomic_charges,
            "forces_hartree_per_bohr": forces,
            "me4n_fragment_charge_e": (
                finite_or_none(
                    sum(value for value in atomic_charges[:-1] if value is not None)
                )
                if all(value is not None for value in atomic_charges)
                else None
            ),
            "chloride_charge_e": atomic_charges[-1],
            "finite_difference": finite_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document["result"], sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
