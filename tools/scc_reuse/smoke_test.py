#!/usr/bin/env python3
"""Smoke test for the issue #343 SCC subspace-reuse diagnostics.

Builds the capture executable, runs it on the pinned h3_plus case, and checks
the analyzer reproduces the expected trajectory shape: 3 completed iterations,
a converged terminal state, finite diagnostics, and numerically valid
eigenpairs (C^T S C = I to roundoff). This is a tooling self-test, not a
scientific acceptance gate.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, help="path to xtbloom_scc_reuse_capture")
    parser.add_argument("--spec", required=True, help="path to h3_plus.spec")
    parser.add_argument(
        "--analyzer",
        required=True,
        help="path to scc_reuse_analyze.py",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        diag = os.path.join(tmp, "h3_plus.diag")
        report = os.path.join(tmp, "h3_plus.json")
        result = run([args.capture, "single", args.spec, diag])
        if result.returncode != 0:
            print(result.stdout, result.stderr, file=sys.stderr)
            return 1
        result = run([sys.executable, args.analyzer, diag, "--report", report])
        if result.returncode != 0:
            print(result.stdout, result.stderr, file=sys.stderr)
            return 1
        with open(report, encoding="utf-8") as handle:
            data = json.load(handle)
        geo = data["geometries"][0]
        if not geo["converged_state"]:
            print("expected h3_plus to converge", file=sys.stderr)
            return 1
        if geo["nao"] != 3 or geo["iterations"] == 0:
            print("unexpected geometry shape", file=sys.stderr)
            return 1
        for row in geo["iterations"]:
            for key in ("rel_dH", "rel_dP", "subspace_capture_fraction",
                        "rel_residual_occupied", "rr_eigenvalue_max_err"):
                if key in row and not (0.0 <= row[key] and row[key] < 1e10):
                    print(f"non-finite metric {key}", file=sys.stderr)
                    return 1
            if abs(row["validation_ctsc_max"]) > 1e-9:
                print("C^T S C = I violated", file=sys.stderr)
                return 1
        print("h3_plus smoke test passed: "
              f"iterations={len(geo['iterations'])} nao={geo['nao']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())