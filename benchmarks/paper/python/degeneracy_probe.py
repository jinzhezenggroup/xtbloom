#!/usr/bin/env python3
"""P0-E supervisor: isolate every engine/case and retain signals and output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def digest(path: Path | None) -> str | None:
    return (
        hashlib.sha256(path.read_bytes()).hexdigest()
        if path and path.is_file()
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument("--max-scc-iterations", type=int, required=True)
    parser.add_argument("--accuracy", type=float, required=True)
    parser.add_argument("--electronic-temperature-kelvin", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    worker = Path(__file__).with_name("degeneracy_probe_child.py")
    fractional = 3.0 - 2.0 * math.nextafter(3.0, 0.0)
    cases = (
        ("neutral-open-shell", 0.0, 1),
        ("anion-exact-block", -3.0, 0),
        ("fractional-binary64", fractional, 0),
    )
    rows = []
    for case, charge, uhf in cases:
        for engine in ("xtbloom", "xtb", "tblite", "dxtb"):
            command = [
                sys.executable,
                str(worker),
                "--repo",
                str(args.repo),
                "--engine",
                engine,
                "--case",
                case,
                "--charge",
                repr(charge),
                "--uhf",
                str(uhf),
                "--xtbloom-library",
                str(args.library),
                "--max-scc-iterations",
                str(args.max_scc_iterations),
                "--accuracy",
                repr(args.accuracy),
                "--electronic-temperature-kelvin",
                repr(args.electronic_temperature_kelvin),
            ]
            for flag, value in (
                ("--xtb-library", args.xtb_library),
                ("--tblite-library", args.tblite_library),
                ("--dxtb-source", args.dxtb_source),
            ):
                if value is not None:
                    command.extend((flag, str(value)))
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                marker = next(
                    (
                        line.removeprefix("PAPER_CHILD_JSON=")
                        for line in reversed(completed.stdout.splitlines())
                        if line.startswith("PAPER_CHILD_JSON=")
                    ),
                    None,
                )
                child = (
                    json.loads(marker)
                    if marker is not None
                    else {
                        "availability": "unavailable",
                        "status": "no-record",
                        "reason": "child emitted no structured record",
                    }
                )
                rows.append(
                    {
                        "engine": engine,
                        "case": case,
                        "command": command,
                        "returncode": completed.returncode,
                        "signal": -completed.returncode
                        if completed.returncode < 0
                        else None,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        **child,
                    }
                )
            except subprocess.TimeoutExpired as exc:
                rows.append(
                    {
                        "engine": engine,
                        "case": case,
                        "command": command,
                        "availability": "unavailable",
                        "status": "timeout",
                        "timeout_seconds": args.timeout_seconds,
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    }
                )

    expected_coordinates = {
        (case, engine)
        for case, _, _ in cases
        for engine in ("xtbloom", "xtb", "tblite", "dxtb")
    }
    actual_coordinates = {(row["case"], row["engine"]) for row in rows}
    failures = []
    if actual_coordinates != expected_coordinates or len(rows) != len(
        expected_coordinates
    ):
        failures.append("engine/case record matrix is incomplete or duplicated")
    for row in rows:
        semantic_unavailable = (
            row["engine"] == "xtb"
            and row["case"] == "fractional-binary64"
            and row.get("availability") == "unavailable"
            and "integer-charge xTB public surface" in row.get("reason", "")
            and row.get("returncode") == 0
        )
        available_result = (
            row.get("availability") == "available"
            and row.get("status") in {"success", "error"}
            and row.get("returncode") in {0, 1}
            and row.get("signal") is None
        )
        if not semantic_unavailable and not available_result:
            failures.append(
                f"{row['engine']}/{row['case']}: missing runtime, timeout, signal, or malformed child record"
            )
        if row["engine"] == "xtbloom" and (
            row.get("availability") != "available"
            or row.get("status") != "success"
            or row.get("returncode") != 0
        ):
            failures.append(
                f"xtbloom/{row['case']}: production stress coordinate failed"
            )
    document = {
        "schema_version": 1,
        "geometry": {
            "atomic_numbers": [1, 1, 1],
            "positions_bohr": [[0.0, 0.0, 0.0], [1e20, 0.0, 0.0], [2e20, 0.0, 0.0]],
        },
        "fractional_charge": fractional,
        "max_scc_iterations": args.max_scc_iterations,
        "accuracy": args.accuracy,
        "electronic_temperature_kelvin": args.electronic_temperature_kelvin,
        "programs": {
            "xtbloom_sha256": digest(args.library),
            "xtb_sha256": digest(args.xtb_library),
            "tblite_sha256": digest(args.tblite_library),
            "dxtb_source": str(args.dxtb_source) if args.dxtb_source else None,
        },
        "scope_warning": "This fixed coordinate is not a general convergence ranking.",
        "formal_gate": {
            "status": "fail" if failures else "pass",
            "failures": failures,
            "allowed_unavailable": "xtb/fractional-binary64 integer-charge surface only",
            "reference_error_semantics": "a structured available/error result is evidence; timeout, signal, missing runtime, and no-record are fatal",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
