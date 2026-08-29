#!/usr/bin/env python3
"""Isolated one-engine/one-case worker for the P0-E stress probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

FE_ALL_EXCEPT = 0x3F
POSITIONS = ((0.0, 0.0, 0.0), (1e20, 0.0, 0.0), (2e20, 0.0, 0.0))


def emit(document: dict[str, Any]) -> None:
    print("PAPER_CHILD_JSON=" + json.dumps(document, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=("xtbloom", "xtb", "tblite", "dxtb"), required=True
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--charge", type=float, required=True)
    parser.add_argument("--uhf", type=int, required=True)
    parser.add_argument("--xtbloom-library", type=Path, required=True)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    args = parser.parse_args()

    if args.engine == "xtb" and args.case == "fractional-binary64":
        emit(
            {
                "availability": "unavailable",
                "reason": "integer-charge xTB public surface cannot express this binary64 coordinate; no failure claim",
            }
        )
        return 0
    required = {
        "xtbloom": args.xtbloom_library,
        "xtb": args.xtb_library,
        "tblite": args.tblite_library,
    }.get(args.engine)
    if required is not None and not required.is_file():
        emit({"availability": "unavailable", "reason": f"library missing: {required}"})
        return 0
    if args.engine == "dxtb" and (
        args.dxtb_source is None or not args.dxtb_source.is_dir()
    ):
        emit({"availability": "unavailable", "reason": "dxtb source/runtime missing"})
        return 0

    from paper_runtime import install

    install(args.repo)
    sys.path.insert(0, str(args.repo / "python"))
    if args.engine == "xtbloom":
        os.environ["XTBLOOM_LIBRARY"] = str(args.xtbloom_library.resolve())
    import dataset_runner  # type: ignore
    import numpy as np
    from dxtb_adapter import DxtbAdapter  # type: ignore
    from tblite_adapter import TbliteAdapter  # type: ignore
    from xtb_adapter import XtbAdapter  # type: ignore
    from xtbloom import Calculator

    libc = ctypes.CDLL(None)
    libc.feclearexcept.argtypes = [ctypes.c_int]
    libc.fetestexcept.argtypes = [ctypes.c_int]
    libc.feclearexcept(FE_ALL_EXCEPT)
    try:
        if args.engine == "xtbloom":
            with Calculator(
                "GFN2-xTB",
                [1, 1, 1],
                np.asarray(POSITIONS),
                charge=args.charge,
                uhf=args.uhf,
                backend="cpu",
                max_scc_iterations=args.max_scc_iterations,
                electronic_temperature=args.electronic_temperature_kelvin,
            ) as calculator:
                result = calculator.singlepoint()
            payload: dict[str, Any] = {
                "energy_hartree": float(result.energy),
                "forces_hartree_per_bohr": np.asarray(result.forces)
                .reshape(-1)
                .tolist(),
                "atomic_charges_e": np.asarray(result.charges).reshape(-1).tolist(),
                "scc_status": int(result.scc_status),
                "scc_converged": bool(result.scc_converged),
                "scc_iterations": int(result.scc_iterations),
            }
            published = [
                payload["energy_hartree"],
                *payload["forces_hartree_per_bohr"],
                *payload["atomic_charges_e"],
            ]
            if not all(math.isfinite(float(value)) for value in published):
                raise RuntimeError(
                    "xTBloom stress probe published a non-finite requested value"
                )
            if payload["scc_status"] != 0 or not payload["scc_converged"]:
                raise RuntimeError(
                    f"xTBloom stress probe did not converge successfully: {payload['scc_status']}"
                )
            charge_span = max(payload["atomic_charges_e"]) - min(
                payload["atomic_charges_e"]
            )
            payload["symmetric_charge_span_e"] = charge_span
            if charge_span > 1.0e-10:
                raise RuntimeError(
                    f"xTBloom symmetric sites published asymmetric charges: span={charge_span:g}"
                )
        else:
            system = dataset_runner.DatasetSystem(
                "probe",
                "p0e",
                args.case,
                0,
                (1, 1, 1),
                POSITIONS,
                args.charge,
                args.uhf + 1,
                args.uhf,
                None,
                {},
                {},
            )
            storage = dataset_runner.storage_from_systems([system])
            if args.engine == "xtb":
                adapter = XtbAdapter(
                    args.xtb_library,
                    storage,
                    "force",
                    None,
                    accuracy=args.accuracy,
                    max_iterations=args.max_scc_iterations,
                    electronic_temperature_kelvin=args.electronic_temperature_kelvin,
                    threads=1,
                )
            elif args.engine == "tblite":
                adapter = TbliteAdapter(
                    args.tblite_library,
                    storage,
                    "force",
                    accuracy=args.accuracy,
                    max_iterations=args.max_scc_iterations,
                    electronic_temperature_hartree=(
                        args.electronic_temperature_kelvin * 3.166808578545117e-6
                    ),
                    threads=1,
                )
            else:
                adapter = DxtbAdapter(
                    storage,
                    "force",
                    "cpu",
                    device_id=0,
                    cpu_threads=1,
                    source_root=args.dxtb_source,
                    accuracy=args.accuracy,
                    force_convergence=True,
                    max_iterations=args.max_scc_iterations,
                )
            try:
                adapter.invoke()
                payload = adapter.results()
            finally:
                adapter.close()
        emit(
            {
                "availability": "available",
                "status": "success",
                "result": payload,
                "floating_point_flags": libc.fetestexcept(FE_ALL_EXCEPT),
                "runtime_identity": {
                    "xtbloom_module": str(
                        Path(sys.modules["xtbloom"].__file__).resolve()
                    ),
                    "max_scc_iterations": args.max_scc_iterations,
                    "accuracy": args.accuracy,
                    "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
                },
            }
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - child must serialize every Python exit
        emit(
            {
                "availability": "available",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "floating_point_flags": libc.fetestexcept(FE_ALL_EXCEPT),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
