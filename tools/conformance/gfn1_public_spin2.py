#!/usr/bin/env python3
"""Gate the public CPU GFN1 two-channel path against the pinned P10 oracle."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

import xtbloom_conformance as conformance
import xtbloom_invariants as invariants
import xtbloom_public_api as public_api

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GFN1_MANIFEST = REPOSITORY_ROOT / "data/conformance/gfn1/manifest.json"
P10_FIXTURE = REPOSITORY_ROOT / "tests/gfn1_cpu_conformance.py"


class P10Case(Protocol):
    """Structural view of the pinned fixture returned by the existing tool."""

    case_id: str
    atomic_numbers: list[int]
    positions: list[float]
    charge: int
    unpaired: int
    spin_channels: int
    expected: dict[str, object]


def load_p10() -> P10Case:
    """Load the already licensed/hash-bound P10 fixture without copying it."""
    spec = importlib.util.spec_from_file_location("gfn1_cpu_conformance", P10_FIXTURE)
    if spec is None or spec.loader is None:
        raise conformance.ConformanceError(
            f"cannot load P10 fixture from {P10_FIXTURE}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("P10Case", module.p10_case())


def maximum_error(expected: object, actual: object) -> float:
    """Return the maximum absolute scalar/array error."""
    if isinstance(expected, list):
        actual_values = list(actual)
        if len(expected) != len(actual_values):
            return math.inf
        return max(
            (
                abs(float(left) - float(right))
                for left, right in zip(expected, actual_values, strict=True)
            ),
            default=0.0,
        )
    return abs(float(expected) - float(actual))


def main(argv: Iterable[str] | None = None) -> int:
    """Run singleton and heterogeneous-ragged public P10 comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        manifest = conformance.load_json(GFN1_MANIFEST)
        p10 = load_p10()
        p10_geometry = invariants.Geometry(
            case_id=p10.case_id,
            atomic_numbers=p10.atomic_numbers,
            positions=p10.positions,
            molecular_charge=p10.charge,
            unpaired_electrons=p10.unpaired,
            spin_channels=p10.spin_channels,
        )
        ketene_case = public_api.supported_cases(manifest, ["gfn1_ketene"], "cpu")
        ketene = invariants.load_geometries(GFN1_MANIFEST, manifest, ketene_case)[0]
        library = public_api._configure_library(args.library)
        solve = invariants.xtbloom_solver(
            library,
            public_api.XTBLOOM_MODEL_GFN1_XTB,
            "cpu",
            0,
            args.cpu_threads,
            "host",
        )
        singleton = solve([p10_geometry])[0]
        ragged = solve([ketene, p10_geometry])[1]
        tolerances = manifest["tolerances"]
        expected_energy = p10.expected["energy_hartree"]
        expected_forces = p10.expected["forces_hartree_per_bohr"]
        failures: list[str] = []
        for label, expected, actual, limit in (
            (
                "energy_hartree",
                expected_energy,
                singleton.energy,
                float(tolerances["energy"]["atol"]),
            ),
            (
                "forces_hartree_per_bohr",
                expected_forces,
                singleton.forces,
                float(tolerances["forces"]["atol"]),
            ),
            ("ragged_energy", singleton.energy, ragged.energy, 1.0e-12),
            ("ragged_forces", singleton.forces, ragged.forces, 1.0e-12),
            ("ragged_charges", singleton.charges, ragged.charges, 1.0e-12),
        ):
            error = maximum_error(expected, actual)
            print(  # noqa: T201 - CLI validation report
                f"{'PASS' if error <= limit else 'FAIL'} P10 {label}: "
                f"max_abs_error={error:.6e} limit={limit:.6e}"
            )
            if error > limit:
                failures.append(label)
        charge_error = abs(sum(singleton.charges) - p10.charge)
        if charge_error > 1.0e-9:
            failures.append("net_charge")
        print(  # noqa: T201 - CLI validation report
            f"{'PASS' if charge_error <= 1.0e-9 else 'FAIL'} P10 net_charge: "
            f"error={charge_error:.6e} limit=1.000000e-09"
        )
        if failures:
            raise conformance.ConformanceError(
                "public CPU GFN1 P10 failures: " + ", ".join(failures)
            )
    except conformance.ConformanceError as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
