#!/usr/bin/env python3
"""Run the issue #407 policy on the public TMAC/Cl difficult SCC case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

ANGSTROM_PER_BOHR = 0.529177210903


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


def finite_or_none(value: float) -> float | None:
    """Convert non-finite public values to strict JSON nulls."""
    return float(value) if math.isfinite(float(value)) else None


def main() -> int:
    """Execute one frozen-policy public calculation and write JSON evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("off", "controller", "local-v1", "pair-response-v1"),
        required=True,
    )
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.resolve()
    library = args.library.resolve()
    input_path = args.input.resolve()
    output = args.output.resolve()
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")

    os.environ["XTBLOOM_LIBRARY"] = str(library)
    os.environ["XTBLOOM_EXPERIMENTAL_GFN2_PAIRS_SCC"] = args.policy
    from xtbloom import BatchCalculator, Structure, __version__
    from xtbloom.interface import SYMBOL_TO_NUMBER

    lines = input_path.read_text(encoding="ascii").splitlines()
    natoms = int(lines[0])
    if natoms != 18 or lines[1] != "tmacl" or len(lines) != natoms + 2:
        raise ValueError("expected the exact 18-atom tmacl XYZ envelope")
    numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2:]:
        symbol, x, y, z = line.split()
        numbers.append(SYMBOL_TO_NUMBER[symbol])
        positions.append(
            [
                float(x) / ANGSTROM_PER_BOHR,
                float(y) / ANGSTROM_PER_BOHR,
                float(z) / ANGSTROM_PER_BOHR,
            ]
        )
    structure = Structure(
        np.asarray(numbers, dtype=np.int32),
        np.asarray(positions, dtype=np.float64),
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
    charges = np.asarray(result.charges, dtype=np.float64)
    forces = np.asarray(result.forces, dtype=np.float64)
    document = {
        "schema_version": 1,
        "case": {
            "id": "tmacl",
            "input": str(input_path.relative_to(repository)),
            "input_sha256": sha256(input_path),
            "natoms": natoms,
            "charge": 0.0,
            "uhf": 0,
            "spin_channels": 1,
            "electronic_temperature_kelvin": 300.0,
            "charge_tolerance": 1.0e-6,
            "energy_tolerance_hartree": 1.0e-8,
        },
        "policy": args.policy,
        "max_scc_iterations": args.max_iterations,
        "result": {
            "scc_converged": bool(result.scc_converged[0]),
            "per_system_status": int(result.per_system_status[0]),
            "scc_iterations": int(result.scc_iterations[0]),
            "energy_hartree": finite_or_none(float(result.energies[0])),
            "atomic_charges_e": [finite_or_none(value) for value in charges],
            "forces_hartree_per_bohr": [
                finite_or_none(value) for value in forces.reshape(-1)
            ],
        },
        "metadata": {
            "repository": str(repository),
            "git_revision": git_output(repository, "rev-parse", "HEAD"),
            "git_status_porcelain": git_output(repository, "status", "--porcelain"),
            "library": str(library),
            "library_sha256": sha256(library),
            "xtbloom_python_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
