#!/usr/bin/env python3
"""Exercise issue #343 capture, analysis, and defensive input checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from prepare_cases import trans_planar_alkane
from scc_reuse_analyze import parse_document


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one diagnostic subprocess without raising on its exit status."""
    return subprocess.run(arguments, capture_output=True, text=True, check=False)


def _modified_spec(
    source: Path,
    destination: Path,
    *,
    displacement: float = 0.0,
    unpaired: int | None = None,
    maximum_iterations: int | None = None,
) -> None:
    """Write a token-equivalent spec with selected fields changed."""
    tokens = source.read_text(encoding="ascii").split()
    atom_count = int(tokens[0])
    position_begin = 1 + atom_count
    charge_index = position_begin + 3 * atom_count
    if displacement:
        tokens[position_begin] = f"{float(tokens[position_begin]) + displacement:.17g}"
    if unpaired is not None:
        tokens[charge_index + 1] = str(unpaired)
    if maximum_iterations is not None:
        tokens[charge_index + 5] = str(maximum_iterations)
    destination.write_text("\n".join(tokens) + "\n", encoding="ascii")


def _require(condition: bool, message: str) -> None:
    """Raise one concise assertion used by the CTest wrapper."""
    if not condition:
        raise AssertionError(message)


def main(arguments: list[str] | None = None) -> int:
    """Run the focused native/Python diagnostic smoke matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    options = parser.parse_args(arguments)

    numbers, positions = trans_planar_alkane(12)
    _require(len(numbers) == 38, "dodecane must contain 38 atoms")
    _require(numbers.count(6) == 12, "dodecane must contain 12 carbons")
    _require(numbers.count(1) == 26, "dodecane must contain 26 hydrogens")
    _require(positions.shape == (38, 3), "dodecane coordinate shape is invalid")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        single_diagnostic = temporary / "h3_plus.diag"
        single_report = temporary / "h3_plus.json"
        result = run(
            [
                str(options.capture),
                "single",
                str(options.spec),
                str(single_diagnostic),
            ]
        )
        _require(result.returncode == 0, result.stdout + result.stderr)
        result = run(
            [
                sys.executable,
                str(options.analyzer),
                str(single_diagnostic),
                "--report",
                str(single_report),
                "--quiet",
            ]
        )
        _require(result.returncode == 0, result.stdout + result.stderr)
        with single_report.open(encoding="utf-8") as handle:
            single = json.load(handle)
        geometry = single["geometries"][0]
        _require(geometry["converged_state"], "expected h3_plus to converge")
        _require(geometry["nao"] == 3, "unexpected h3_plus basis size")
        _require(bool(geometry["iterations"]), "h3_plus recorded no iterations")
        for row in geometry["iterations"]:
            _require(
                row["validation_ctsc_max"] <= 1.0e-9,
                "C^T S C = I validation failed",
            )
            _require(
                row["validation_he_max"] <= 1.0e-9,
                "generalized eigenpair validation failed",
            )

        target_spec = temporary / "h3_target.spec"
        _modified_spec(options.spec, target_spec, displacement=0.01)
        trajectory_diagnostic = temporary / "h3_trajectory.diag"
        trajectory_report = temporary / "h3_trajectory.json"
        result = run(
            [
                str(options.capture),
                "traj",
                str(options.spec),
                str(target_spec),
                str(trajectory_diagnostic),
            ]
        )
        _require(result.returncode == 0, result.stdout + result.stderr)
        result = run(
            [
                sys.executable,
                str(options.analyzer),
                str(trajectory_diagnostic),
                "--report",
                str(trajectory_report),
                "--quiet",
            ]
        )
        _require(result.returncode == 0, result.stdout + result.stderr)
        with trajectory_report.open(encoding="utf-8") as handle:
            trajectory = json.load(handle)
        roles = {entry["role"] for entry in trajectory["geometries"]}
        _require(
            roles == {"source", "target_warm", "target_fresh"},
            "trajectory is missing its same-geometry FRESH control",
        )
        physical_capture = trajectory["trajectory"]["subspace_capture_fraction"]
        _require(0.0 <= physical_capture <= 1.0, "physical capture is out of range")
        rr_residual = trajectory["trajectory"]["target_metric_reuse"][
            "rr_rel_residual_occupied"
        ]
        _require(np.isfinite(rr_residual), "target effective-H residual is non-finite")
        control = trajectory["warm_start_control"]
        _require(
            control["final_density_rel_difference"] < 1.0e-5,
            "WARM and FRESH target solutions disagree",
        )

        identical_diagnostic = temporary / "h3_identical.diag"
        result = run(
            [
                str(options.capture),
                "traj",
                str(options.spec),
                str(options.spec),
                str(identical_diagnostic),
            ]
        )
        _require(result.returncode == 0, result.stdout + result.stderr)
        parsed = parse_document(identical_diagnostic)
        source = parsed["geometries"][0]
        warm = parsed["geometries"][1]
        _require(
            np.allclose(
                warm["cross_overlap_from_source"],
                source["overlap"],
                rtol=0.0,
                atol=2.0e-14,
            ),
            "identical-geometry cross overlap does not reproduce S",
        )

        unrestricted_spec = temporary / "h3_unrestricted.spec"
        _modified_spec(options.spec, unrestricted_spec, unpaired=1)
        result = run([str(options.capture), "single", str(unrestricted_spec)])
        _require(result.returncode == 2, "unrestricted input was not rejected")
        _require(
            "restricted SCC only" in result.stderr,
            "unrestricted rejection lacks a useful diagnostic",
        )

        mismatched_spec = temporary / "h3_policy_mismatch.spec"
        _modified_spec(options.spec, mismatched_spec, maximum_iterations=99)
        result = run(
            [
                str(options.capture),
                "traj",
                str(options.spec),
                str(mismatched_spec),
            ]
        )
        _require(result.returncode == 2, "trajectory SCC-policy mismatch was accepted")
        _require(
            "identical SCC policy" in result.stderr,
            "policy mismatch rejection lacks a useful diagnostic",
        )

    sys.stdout.write("issue #343 SCC reuse smoke matrix passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
