"""Helpers to load xTBloom conformance cases for the Python test suite.

Reads ``data/conformance/manifest.json`` and the committed golden files so the
Python bindings are validated against the same trusted reference data as the C
tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from xtbloom import (  # noqa: F401 - re-exported test helper
    numbers_to_symbols,
    symbols_to_numbers,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "conformance" / "manifest.json"
GOLDEN_DIR = REPOSITORY_ROOT / "data" / "conformance" / "golden"
INPUT_DIR = REPOSITORY_ROOT / "data" / "conformance" / "inputs"

_UNITS_ANGSTROM_PER_BOHR = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def manifest() -> dict:
    return load_json(MANIFEST_PATH)


def case_ids() -> list[str]:
    return [case["id"] for case in manifest()["cases"]]


def case_by_id(case_id: str) -> dict:
    for case in manifest()["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def golden(case: dict) -> dict:
    return load_json(REPOSITORY_ROOT / case["golden"])["properties"]


def _parse_coord(path: Path) -> tuple[list[int], np.ndarray]:
    """Parse a Turbomole ``$coord`` file into numbers and bohr positions."""
    numbers: list[int] = []
    positions: list[float] = []
    in_coord = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "$coord":
            in_coord = True
            continue
        if line.startswith("$"):
            in_coord = False
            continue
        if not in_coord or not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        # Turbomole writes element labels in lowercase, while the public
        # helper intentionally accepts canonical chemical symbols.
        numbers.append(symbols_to_numbers([parts[3].capitalize()])[0])
        positions.append(float(parts[0]))
        positions.append(float(parts[1]))
        positions.append(float(parts[2]))
    return numbers, np.asarray(positions, dtype=np.float64).reshape(-1, 3)


def structure_inputs(case: dict) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Return per-case numbers, positions (bohr), charge, uhf, spin_channels.

    A missing ``spin_channels`` key in the manifest means restricted (1); the
    caller should still pass it explicitly so open-shell cases stay restricted
    rather than adopting the Python-interface unrestricted default.
    """
    input_path = REPOSITORY_ROOT / case["input"]
    if case.get("input_schema") == "qmmm-v1":
        document = load_json(input_path)
        numbers = np.asarray(document["qm"]["atomic_numbers"], dtype=np.int64)
        positions = np.asarray(document["qm"]["positions_bohr"], dtype=np.float64)
    else:
        numbers, positions = _parse_coord(input_path)
        numbers = np.asarray(numbers, dtype=np.int64)
    return (
        numbers,
        positions,
        float(case["molecular_charge"]),
        int(case["unpaired_electrons"]),
        int(case.get("spin_channels", 1)),
    )


def qmmm_points(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return external point charge (positions, values, gammas) for QM/MM cases."""
    if case.get("input_schema") != "qmmm-v1":
        return None
    document = load_json(REPOSITORY_ROOT / case["input"])
    points = document["external_point_charges"]
    return (
        np.asarray(points["positions_bohr"], dtype=np.float64),
        np.asarray(points["charges_e"], dtype=np.float64),
        np.asarray(points["gammas_hartree"], dtype=np.float64),
    )


def tolerances() -> dict:
    return manifest()["tolerances"]
