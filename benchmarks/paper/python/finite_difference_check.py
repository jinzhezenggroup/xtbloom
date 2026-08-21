#!/usr/bin/env python3
"""P0-C central-difference checks on the pre-frozen real-system coordinates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


class FiniteDifferenceError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset", choices=("qm9", "omol25"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-scc-iterations", type=int, default=500)
    parser.add_argument("--steps", default="0.001,0.0005,0.002")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.atol > 1e-5:
        raise FiniteDifferenceError("finite-difference tolerance may not exceed 1e-5")
    steps = tuple(float(value) for value in args.steps.split(","))
    if not steps or any(value <= 0 for value in steps):
        raise FiniteDifferenceError("steps must be positive")
    for path in (args.manifest, args.selection, args.library):
        if not path.is_file():
            raise FiniteDifferenceError(f"required file is missing: {path}")
    if args.output.exists():
        raise FiniteDifferenceError(f"refusing to overwrite: {args.output}")

    from paper_runtime import install

    install(args.repo)
    sys.path.insert(0, str(args.repo / "python"))
    import dataset_runner  # type: ignore

    os.environ["XTBLOOM_LIBRARY"] = str(args.library.resolve())
    import numpy as np
    from xtbloom import library
    from xtbloom.interface import Context, Structure, _compute_batch

    frozen = json.loads(args.selection.read_text())
    chosen = {
        row["system_id"]: row
        for row in frozen["finite_difference_selection"]
        if row["dataset"] == args.dataset
    }
    if len(chosen) != 16:
        raise FiniteDifferenceError(f"expected 16 frozen {args.dataset} systems")
    loaded = {
        item.system_id: item.system
        for item in dataset_runner.load_manifest(args.manifest, args.dataset)
        if item.system is not None and item.system_id in chosen
    }
    if set(loaded) != set(chosen):
        raise FiniteDifferenceError("selection IDs are missing from the manifest")

    def structure(system: Any, positions: Any | None = None) -> Structure:
        return Structure(
            list(system.atomic_numbers),
            np.asarray(system.positions_bohr if positions is None else positions),
            charge=system.charge,
            uhf=system.unpaired_electrons,
        )

    def compute(context: Context, one: Structure, flags: int) -> Any:
        result = _compute_batch(
            context,
            [one],
            model=library.MODEL_GFN2_XTB,
            max_scc_iterations=args.max_scc_iterations,
            charge_tolerance=1e-6,
            energy_tolerance=1e-8,
            electronic_temperature=300.0,
            scc_mixer=library.SCC_MIXER_MODIFIED_BROYDEN,
            scc_mixer_history=library.DEFAULT_SCC_MIXER_HISTORY,
            scc_mixer_damping=library.DEFAULT_SCC_MIXER_DAMPING,
            determinism=library.DETERMINISM_DEFAULT,
            flags=flags,
            warm_start=False,
        )
        if (
            int(result.per_system_status[0]) != library.STATUS_SUCCESS
            or int(result.scc_converged[0]) != 1
        ):
            raise FiniteDifferenceError(
                f"native failure status={int(result.per_system_status[0])} "
                f"converged={int(result.scc_converged[0])}"
            )
        return result

    rows = []
    flags = library.COMPUTE_ENERGY | library.COMPUTE_FORCES
    with Context(
        args.backend,
        args.device_id if args.backend == "cuda" else None,
        args.cpu_threads,
    ) as context:
        for system_id, frozen_row in sorted(chosen.items()):
            system = loaded[system_id]
            if (
                frozen_row.get("input_sha256")
                and frozen_row["input_sha256"] != system.input_sha256
            ):
                raise FiniteDifferenceError(f"input hash mismatch for {system_id}")
            base = compute(context, structure(system), flags)
            base_forces = np.asarray(base.forces).reshape((-1, 3))
            original = np.asarray(system.positions_bohr, dtype=float)
            for coordinate in frozen_row["coordinate_indices"]:
                atom, axis = divmod(int(coordinate), 3)
                for step in steps:
                    plus, minus = original.copy(), original.copy()
                    plus[atom, axis] += step
                    minus[atom, axis] -= step
                    e_plus = float(
                        compute(
                            context, structure(system, plus), library.COMPUTE_ENERGY
                        ).energies[0]
                    )
                    e_minus = float(
                        compute(
                            context, structure(system, minus), library.COMPUTE_ENERGY
                        ).energies[0]
                    )
                    numerical_force = -(e_plus - e_minus) / (2 * step)
                    analytic_force = float(base_forces[atom, axis])
                    error = abs(numerical_force - analytic_force)
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "system_id": system_id,
                            "backend": args.backend,
                            "coordinate_index": int(coordinate),
                            "atom": atom,
                            "axis": axis,
                            "step_bohr": step,
                            "analytic_force_hartree_per_bohr": analytic_force,
                            "numerical_force_hartree_per_bohr": numerical_force,
                            "absolute_error_hartree_per_bohr": error,
                            "passed": error <= args.atol,
                        }
                    )

    document = {
        "schema_version": 1,
        "dataset": args.dataset,
        "backend": args.backend,
        "atol_hartree_per_bohr": args.atol,
        "rows": rows,
        "summary": {
            "count": len(rows),
            "failed": sum(not row["passed"] for row in rows),
            "maximum_error_hartree_per_bohr": max(
                row["absolute_error_hartree_per_bohr"] for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 1 if document["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
