#!/usr/bin/env python3
"""Run the pinned CPU corpus through the production SCC driver and compare.

Issue #50: prove that the gpuxtb CPU SCC driver follows the pinned tblite
iteration traces both one system at a time and (with --ragged) in a homogeneous
build of the corpus.  This tool:

  * runs the ``gpuxtb_scc_trace_capture`` binary on each case.spec;
  * canonically converts the captured raw stream with the same generator
    pipeline that produced the goldens;
  * compares the complete closed-loop trace against the pinned golden with the
    documented ``cpu_closed_loop_v1`` profile (field-level
    ``abs <= atol + rtol*max`` rule);
  * prints one ``PASS``/``FAIL`` line per case naming the first divergent
    iteration and field, and exits non-zero when any case diverges.

The capture uses the internal GFN2 driver with the same mixer history, damping,
tolerances, temperature, and initial SAD guess as the fortran oracle where the
production path permits; any divergence is a real scientific finding, not a
tolerance change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scc_corpus as generator
import gpuxtb_scc_trace as writer

TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[2]
CORPUS_DIR = REPOSITORY_ROOT / "data" / "conformance" / "scc-traces"
COMPARE = TOOL_DIR / "gpuxtb_scc_compare.py"


def canonicalize_capture(raw: str, case_id: str) -> dict:
    """Convert one captured CPU raw stream into a canonical trace document."""
    spec = generator.CASES[case_id]
    return generator.canonicalize(raw, spec, "gpuxtb_scc_cpu_trace.py (capture)")


def compare_with_comparator(
    actual: Path, golden: Path, profile: str
) -> tuple[int, str]:
    """Compare one captured trace with the golden via the comparator CLI."""
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "trace",
            str(actual),
            str(golden),
            "--profile",
            profile,
            "--max-reported",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the CPU trace comparison tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="path to the gpuxtb_scc_trace_capture driver executable",
    )
    parser.add_argument(
        "--profile",
        default="cpu_closed_loop_v1",
        choices=("cpu_closed_loop_v1", "cuda_replay_v1"),
        help="comparator tolerance profile",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help="directory containing the pinned goldens and specs",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("."),
        help="scratch directory for captured traces",
    )
    parser.add_argument(
        "--cases",
        default=sorted(generator.CASES),
        nargs="*",
        help="restricted cases to compare (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every requested corpus case through the CPU driver and compare."""
    arguments = build_parser().parse_args(argv)

    corpus_dir = arguments.corpus_dir
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    goldens = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))[
        "cases"
    ]

    failures = 0
    for case_id in arguments.cases:
        spec_path = corpus_dir / "specs" / f"{case_id}.spec"
        golden_path = corpus_dir / goldens[case_id]["path"]
        actual_path = work_dir / f"{case_id}_cpu.json"
        raw_path = work_dir / f"{case_id}_cpu.raw"

        with raw_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [str(arguments.capture), str(spec_path)],
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            print(  # noqa: T201 - CLI diagnostics
                f"{case_id}: FAIL capture died with "
                f"{result.returncode}: {result.stderr}"
            )
            failures += 1
            continue

        raw = raw_path.read_text(encoding="utf-8")
        try:
            trace = canonicalize_capture(raw, case_id)
            canonical = writer.dumps(trace)
        except (
            writer.TraceError,
            generator.CorpusError,
            ValueError,
            AssertionError,
        ) as error:
            print(  # noqa: T201 - CLI diagnostics
                f"{case_id}: FAIL cannot canonicalize capture: {error}"
            )
            failures += 1
            continue
        actual_path.write_text(canonical, encoding="utf-8")

        return_code, report = compare_with_comparator(
            actual_path, golden_path, arguments.profile
        )
        if return_code == 0:
            print(f"{case_id}: PASS ({arguments.profile})")  # noqa: T201
        else:
            failures += 1
            summary = report.strip().splitlines()
            print(f"{case_id}: FAIL ({arguments.profile})")  # noqa: T201
            for line in summary[:6]:
                print(f"  {line}")  # noqa: T201

    if failures:
        print(f"{failures}/{len(arguments.cases)} CPU trace comparisons failed")  # noqa: T201
    else:
        print(f"all {len(arguments.cases)} CPU trace comparisons passed")  # noqa: T201
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
