#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise production GFN1/GFN2 inference from an installed Pyodide wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


def _installed_private_paths() -> tuple[Path, Path]:
    """Locate the repaired adapter and unique content-qualified provider."""
    spec = importlib.util.find_spec("xtbloom")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed xtbloom package is unavailable")
    package = Path(spec.origin).resolve().parent
    adapter = package / "lib" / "libxtbloom_pyodide_lapacke.so"
    providers = sorted(
        (package.parent / "xtbloom.libs").glob("libxtbloom_openblas-*.so")
    )
    if len(providers) != 1:
        raise RuntimeError(f"expected one private Pyodide provider, found {providers}")
    return adapter, providers[0]


def _expect_unavailable() -> None:
    """Require a missing or corrupt private payload to fail without SciPy fallback."""
    import numpy as np
    import scipy.linalg
    from xtbloom import Calculator

    # Load SciPy's own side-module cohort first. A broken xTBloom private
    # provider must still fail rather than resolving generic symbols here.
    scipy.linalg.eigh(np.array([[2.0, 0.5], [0.5, 1.0]]), eigvals_only=True)
    try:
        Calculator(
            "GFN2-xTB",
            np.array([8, 1, 1]),
            np.array(
                [
                    [0.0, 0.0, -0.73578586109551],
                    [1.44183152868459, 0.0, 0.36789293054775],
                    [-1.44183152868459, 0.0, 0.36789293054775],
                ]
            ),
            backend="cpu",
        ).singlepoint()
    except Exception as error:
        message = str(error).lower()
        if not any(
            token in message for token in ("pyodide", "openblas", "lapacke", "backend")
        ):
            raise RuntimeError(
                f"unexpected private-provider failure: {error}"
            ) from error
        sys.stdout.write(f"expected private-provider failure: {error}\n")
        return
    raise RuntimeError(
        "GFN2 inference unexpectedly used a non-private provider fallback"
    )


def _load_cases(source_root: Path) -> ModuleType:
    """Load the repository's canonical case parser without importing source xtbloom."""
    helper_path = source_root / "python" / "tests" / "_cases.py"
    spec = importlib.util.spec_from_file_location("xtbloom_pyodide_cases", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load conformance helper {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_scc_options() -> dict[str, float | int]:
    """Match the established public conformance solver settings exactly."""
    return {
        "max_scc_iterations": 500,
        "charge_tolerance": 1.0e-10,
        "energy_tolerance": 1.0e-12,
    }


def _load_invariants(source_root: Path) -> ModuleType:
    """Load the authoritative invariant gates against the installed package."""
    conformance_dir = source_root / "tools" / "conformance"
    helper_path = conformance_dir / "xtbloom_invariants.py"
    module_name = "xtbloom_pyodide_invariants"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load invariant helper {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(conformance_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(conformance_dir))
    return module


def _run_complete_invariants(source_root: Path) -> None:
    """Run the complete corpus-wide public invariance/FD/ragged gates."""
    import numpy as np
    from xtbloom import Context, PointCharge, Structure
    from xtbloom import interface as xtbloom_interface
    from xtbloom import library as xtbloom_library

    invariants = _load_invariants(source_root)
    manifest_path = source_root / "data" / "conformance" / "manifest.json"
    manifest = invariants.conformance.load_json(manifest_path)
    selected = invariants.public_api.supported_cases(manifest, None, "cpu")
    geometries = invariants.load_geometries(manifest_path, manifest, selected)

    def solve(input_geometries: Sequence[Any]) -> list[Any]:
        structures = []
        for geometry in input_geometries:
            points = None
            if geometry.point_values:
                points = PointCharge(
                    np.asarray(geometry.point_positions, dtype=float).reshape(-1, 3),
                    np.asarray(geometry.point_values, dtype=float),
                    np.asarray(geometry.point_gammas, dtype=float),
                )
            structures.append(
                Structure(
                    np.asarray(geometry.atomic_numbers, dtype=np.int64),
                    np.asarray(geometry.positions, dtype=float).reshape(-1, 3),
                    charge=geometry.molecular_charge,
                    uhf=geometry.unpaired_electrons,
                    spin_channels=geometry.spin_channels,
                    point_charges=points,
                    efield=geometry.efield,
                )
            )

        # The invariant contract requires molecular dipoles for field-free as
        # well as field-attached systems. Request that ABI outlet explicitly:
        # the high-level Structure API intentionally normalizes an all-zero
        # field to no attachment and therefore cannot be used as a proxy for
        # an output-only dipole request.
        flags = (
            xtbloom_library.COMPUTE_ENERGY
            | xtbloom_library.COMPUTE_FORCES
            | xtbloom_library.COMPUTE_ATOMIC_CHARGES
            | xtbloom_library.COMPUTE_DIPOLE_MOMENTS
        )
        if any(structure.point_charges is not None for structure in structures):
            flags |= xtbloom_library.COMPUTE_POINT_CHARGE_FORCES
        strict = _strict_scc_options()
        with Context("cpu") as context:
            computed = xtbloom_interface._compute_batch(
                context,
                structures,
                model=xtbloom_library.MODEL_GFN2_XTB,
                max_scc_iterations=int(strict["max_scc_iterations"]),
                charge_tolerance=float(strict["charge_tolerance"]),
                energy_tolerance=float(strict["energy_tolerance"]),
                electronic_temperature=300.0,
                scc_mixer=xtbloom_library.SCC_MIXER_MODIFIED_BROYDEN,
                scc_mixer_history=xtbloom_library.DEFAULT_SCC_MIXER_HISTORY,
                scc_mixer_damping=xtbloom_library.DEFAULT_SCC_MIXER_DAMPING,
                determinism=xtbloom_library.DETERMINISM_DEFAULT,
                flags=flags,
            )
        failed = np.flatnonzero(
            (computed.per_system_status != xtbloom_library.STATUS_SUCCESS)
            | (computed.scc_converged != 1)
        )
        if failed.size:
            raise RuntimeError(
                f"Pyodide invariant solve failed for systems {failed.tolist()}"
            )
        if computed.dipole_moments is None:
            raise RuntimeError("Pyodide invariant solve omitted dipoles")

        results = []
        for index, geometry in enumerate(input_geometries):
            atom_begin = int(computed.atom_offsets[index])
            atom_end = int(computed.atom_offsets[index + 1])
            point_begin = (
                int(computed.point_offsets[index])
                if computed.point_offsets is not None
                else 0
            )
            point_end = (
                int(computed.point_offsets[index + 1])
                if computed.point_offsets is not None
                else 0
            )
            point_forces = (
                []
                if computed.point_charge_forces is None
                else np.asarray(computed.point_charge_forces[point_begin:point_end])
                .reshape(-1)
                .tolist()
            )
            results.append(
                invariants.InvariantResult(
                    case_id=geometry.case_id,
                    molecular_charge=geometry.molecular_charge,
                    energy=float(computed.energies[index]),
                    forces=np.asarray(computed.forces[atom_begin:atom_end])
                    .reshape(-1)
                    .tolist(),
                    charges=np.asarray(computed.charges[atom_begin:atom_end]).tolist(),
                    point_forces=point_forces,
                    dipoles=np.asarray(computed.dipole_moments[index]).tolist(),
                    efield=None if geometry.efield is None else list(geometry.efield),
                )
            )
        return results

    failures = invariants.run_invariant_checks(
        solve,
        geometries,
        invariants.select_homogeneous_case_ids(geometries),
    )
    if failures:
        raise RuntimeError(
            f"Pyodide complete invariant suite reported {len(failures)} failure(s)"
        )


def _calculator_for_case(case: dict[str, object], cases: ModuleType) -> object:
    """Build one public calculator from a committed conformance case."""
    from xtbloom import Calculator, PointCharge

    numbers, positions, charge, uhf, spin = cases.structure_inputs(case)
    kwargs: dict[str, object] = {}
    points = cases.qmmm_points(case)
    if points is not None:
        kwargs["point_charges"] = PointCharge(*points)
    if "efield" in case:
        kwargs["efield"] = case["efield"]
    return Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=spin,
        backend="cpu",
        **_strict_scc_options(),
        **kwargs,
    )


def _run_full_conformance(source_root: Path) -> tuple[float, object]:
    """Run primary goldens, invariance, finite differences, and ragged failure."""
    import numpy as np
    from xtbloom import BatchCalculator, Calculator, Structure

    cases = _load_cases(source_root)
    tolerances = cases.tolerances()
    results: dict[str, object] = {}
    for case_id in cases.case_ids():
        case = cases.case_by_id(case_id)
        result = _calculator_for_case(case, cases).singlepoint()
        if not result.scc_converged:
            raise RuntimeError(f"Pyodide conformance did not converge: {case_id}")
        golden = cases.golden(case)
        oracle_properties = case.get("xtbloom_oracle_properties")
        primary_properties = (
            None if oracle_properties is None else set(oracle_properties)
        )

        np.testing.assert_allclose(
            result.energy,
            golden["energy_hartree"],
            rtol=0.0,
            atol=tolerances["energy"]["atol"],
            err_msg=case_id,
        )
        if "forces_hartree_per_bohr" in golden and (
            primary_properties is None
            or "forces_hartree_per_bohr" in primary_properties
        ):
            np.testing.assert_allclose(
                result.forces,
                np.asarray(golden["forces_hartree_per_bohr"]).reshape(-1, 3),
                rtol=0.0,
                atol=tolerances["forces"]["atol"],
                err_msg=case_id,
            )
        if "partial_charges_e" in golden and (
            primary_properties is None or "partial_charges_e" in primary_properties
        ):
            np.testing.assert_allclose(
                result.charges,
                golden["partial_charges_e"],
                rtol=0.0,
                atol=tolerances["charges"]["atol"],
                err_msg=case_id,
            )
        if "point_charge_forces_hartree_per_bohr" in golden and (
            primary_properties is None
            or "point_charge_forces_hartree_per_bohr" in primary_properties
        ):
            if result.point_charge_forces is None:
                raise RuntimeError(f"Pyodide omitted point-charge forces: {case_id}")
            np.testing.assert_allclose(
                result.point_charge_forces,
                np.asarray(golden["point_charge_forces_hartree_per_bohr"]).reshape(
                    -1, 3
                ),
                rtol=0.0,
                atol=tolerances["point_charge_forces"]["atol"],
                err_msg=case_id,
            )
        results[case_id] = result

    # Restricted and unrestricted OH must both work; the latter is variationally lower.
    radical = cases.case_by_id("oh_radical")
    numbers, positions, charge, uhf, _ = cases.structure_inputs(radical)
    unrestricted = Calculator(
        "GFN2-xTB",
        numbers,
        positions,
        charge=charge,
        uhf=uhf,
        spin_channels=2,
        backend="cpu",
        **_strict_scc_options(),
    ).singlepoint()
    if unrestricted.energy >= results["oh_radical"].energy:
        raise RuntimeError("unrestricted OH is not below the restricted solution")

    # Use the established public unrestricted-force finite-difference gate.
    step = 1.0e-4
    displaced_plus = np.array(positions, copy=True)
    displaced_minus = np.array(positions, copy=True)
    displaced_plus[1, 0] += step
    displaced_minus[1, 0] -= step
    energy_plus = (
        Calculator(
            "GFN2-xTB",
            numbers,
            displaced_plus,
            charge=charge,
            uhf=uhf,
            spin_channels=2,
            **_strict_scc_options(),
        )
        .singlepoint()
        .energy
    )
    energy_minus = (
        Calculator(
            "GFN2-xTB",
            numbers,
            displaced_minus,
            charge=charge,
            uhf=uhf,
            spin_channels=2,
            **_strict_scc_options(),
        )
        .singlepoint()
        .energy
    )
    finite_difference = -(energy_plus - energy_minus) / (2.0 * step)
    np.testing.assert_allclose(
        unrestricted.forces[1, 0], finite_difference, rtol=0.0, atol=2.0e-5
    )

    _run_complete_invariants(source_root)

    def structure(case_id: str) -> Structure:
        case = cases.case_by_id(case_id)
        numbers, positions, charge, uhf, spin = cases.structure_inputs(case)
        return Structure(numbers, positions, charge=charge, uhf=uhf, spin_channels=spin)

    failing = BatchCalculator(
        [structure("h3_plus"), structure("nenacl")],
        backend="cpu",
        max_scc_iterations=4,
    ).compute()
    successful_peer = failing[0]
    failed_peer = failing[1]
    if failing.failed_indices.tolist() != [1] or not (
        np.isfinite(successful_peer.energy)
        and np.isfinite(successful_peer.forces).all()
        and np.isfinite(successful_peer.charges).all()
    ):
        raise RuntimeError("Pyodide ragged peer-local failure semantics differ")
    if not (
        np.isnan(failed_peer.energy)
        and np.isnan(failed_peer.forces).all()
        and np.isnan(failed_peer.charges).all()
    ):
        raise RuntimeError("failed Pyodide batch peer did not fully publish NaNs")
    return float(results["ketene"].energy), results["ketene"].forces


def _run_load_order(source_root: Path, load_order: str, full: bool) -> None:
    """Prove both SciPy/xTBloom load orders and repeat inference afterward."""
    import numpy as np

    water_numbers = np.array([8, 1, 1])
    water_positions = np.array(
        [
            [0.0, 0.0, -0.73578586109551],
            [1.44183152868459, 0.0, 0.36789293054775],
            [-1.44183152868459, 0.0, 0.36789293054775],
        ]
    )
    if load_order == "scipy-first":
        import scipy.linalg

        host_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
        host_before = scipy.linalg.eigh(host_matrix, eigvals_only=True)
        from xtbloom import Calculator

        first = Calculator(
            "GFN2-xTB", water_numbers, water_positions, backend="cpu"
        ).singlepoint()
    else:
        from xtbloom import Calculator

        # Trigger the complete xTBloom adapter/provider factory before SciPy's
        # separately installed OpenBLAS side module enters the process.
        first = Calculator(
            "GFN2-xTB", water_numbers, water_positions, backend="cpu"
        ).singlepoint()
        import scipy.linalg

        host_before = None
    if full:
        reference_energy, reference_forces = _run_full_conformance(source_root)
    else:
        reference_energy, reference_forces = first.energy, first.forces
    host_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    host_after = scipy.linalg.eigh(host_matrix, eigvals_only=True)
    if host_before is not None:
        np.testing.assert_allclose(host_after, host_before, rtol=0.0, atol=0.0)
    repeated = Calculator(
        "GFN2-xTB", water_numbers, water_positions, backend="cpu"
    ).singlepoint()
    np.testing.assert_allclose(repeated.energy, first.energy, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(repeated.forces, first.forces, rtol=0.0, atol=1.0e-12)
    # The Pyodide package contains the same CPU model registry and generated
    # parameter payload as native wheels. Exercise GFN1 explicitly so a wheel
    # cannot pass solely through the default GFN2 selector.
    gfn1 = Calculator(
        "GFN1-xTB", water_numbers, water_positions, backend="cpu"
    ).singlepoint()
    if not gfn1.scc_converged or not np.isfinite(gfn1.energy):
        raise RuntimeError(
            "installed Pyodide wheel did not execute finite GFN1 inference"
        )
    if not np.isfinite(gfn1.forces).all():
        raise RuntimeError("installed Pyodide wheel returned non-finite GFN1 forces")
    adapter, provider = _installed_private_paths()
    if Path(os.environ.get("XTBLOOM_PYODIDE_LAPACKE_SHIM", "")) != adapter:
        raise RuntimeError("native loader did not retain the exact adapter path")
    if Path(os.environ.get("XTBLOOM_PYODIDE_OPENBLAS", "")) != provider:
        raise RuntimeError("native loader did not retain the exact provider path")
    sys.stdout.write(
        "Pyodide GFN1/GFN2 wheel passed: "
        f"order={load_order}; full={full}; energy={reference_energy:.16g}; "
        f"gfn1_energy={gfn1.energy:.16g}; "
        f"force_norm={np.linalg.norm(reference_forces):.16g}; "
        f"numpy={np.__version__}; scipy={importlib.metadata.version('scipy')}; "
        f"provider={provider.name}\n"
    )


def main() -> int:
    """Dispatch fresh-process load-order and negative-provider checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("locate", "expect-unavailable", "scipy-first", "xtbloom-first"),
        required=True,
    )
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.mode == "locate":
        adapter, provider = _installed_private_paths()
        sys.stdout.write(f"{adapter}\n{provider}\n")
    elif args.mode == "expect-unavailable":
        _expect_unavailable()
    else:
        _run_load_order(args.source_root.resolve(), args.mode, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
