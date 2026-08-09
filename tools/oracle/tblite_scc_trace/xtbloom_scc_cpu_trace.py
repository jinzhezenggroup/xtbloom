#!/usr/bin/env python3
"""Run the pinned CPU corpus through the production SCC driver and compare.

Issues #42/#49/#50: prove that the xtbloom CPU SCC driver follows the pinned
tblite iteration traces one system at a time, in heterogeneous ragged batches,
and for independently replayed iterations.  This tool:

  * ``--capture`` runs the ``xtbloom_scc_trace_capture`` binary on each case.spec
    and compares each complete closed-loop trajectory against the pinned golden
    with the documented ``cpu_closed_loop_v1`` profile;
  * ``--batch-capture`` runs ``xtbloom_scc_trace_batch_capture`` with several
    cases in one ragged driver batch (plus a controlled failing lane), proves
    each healthy lane equals its pinned sequential trajectory, and checks that
    a per-system failure neither corrupts nor suppresses the peers;
  * ``--replay`` runs ``xtbloom_scc_trace_replay`` for every golden iteration:
    the golden mixed q/d/Q state is injected, one driver iteration is executed,
    and the snapshot is compared with the ``cpu_replay_v1`` single-iteration
    profile, so a divergence is assigned to the exact iteration where it
    originates without inheriting Broyden drift.

Every comparison names the case, batch lane (where relevant), SCC iteration,
and first mismatching trace field, and exits non-zero when any case diverges.
The capture uses the internal GFN2 driver with the same mixer history, damping,
tolerances, temperature, and zero-charge initial guess as the fortran oracle
where the production path permits; any divergence is a real scientific finding,
not a tolerance change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scc_corpus as generator
import xtbloom_scc_trace as writer

TOOL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOL_DIR.parents[2]
CORPUS_DIR = REPOSITORY_ROOT / "data" / "conformance" / "scc-traces"
COMPARE = TOOL_DIR / "xtbloom_scc_compare.py"

# xtbloom_status_t ABI values (mirrored from include/xtbloom/xtbloom.h) used for
# the batch failure-lane assertions.
STATUS_SUCCESS = 0
STATUS_INTERNAL_ERROR = 6
STATUS_SCC_NOT_CONVERGED = 7


def canonicalize_capture(raw: str, case_id: str) -> dict:
    """Convert one captured CPU raw stream into a canonical trace document."""
    spec = generator.CASES[case_id]
    return generator.canonicalize(raw, spec, "xtbloom_scc_cpu_trace.py (capture)")


def validate_replay_lifecycle(trace: dict, expected_iteration: dict) -> None:
    """Require the single-step replay artifact to carry exact terminal state."""
    converged = bool(expected_iteration["convergence"]["overall"])
    expected_terminal = {
        "status": writer.STATUS_CONVERGED
        if converged
        else writer.STATUS_MAX_ITERATIONS,
        "converged": converged,
        "iterations": 1,
    }
    if "failed_attempt" in trace or trace["terminal"] != expected_terminal:
        raise generator.CorpusError(
            "replay lifecycle mismatch: expected "
            f"{expected_terminal}, got terminal={trace['terminal']} "
            f"failed_attempt={'failed_attempt' in trace}"
        )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_golden(golden_path: Path, golden_entry: dict) -> dict:
    """Load, pin, and canonically validate a golden before evidence use.

    Every mode that derives expected values directly from a golden (replay and
    mixer) must first prove the file matches its manifest SHA-256 and that the
    bytes are canonical version-1 JSON, so a drifted or unpinned golden cannot
    silently report PASS against itself.
    """
    content = golden_path.read_bytes()
    if sha256_file(golden_path) != golden_entry["sha256"]:
        raise generator.CorpusError(
            f"golden {golden_path} SHA-256 does not match the manifest"
        )
    document = json.loads(content.decode("utf-8"))
    writer.validate(document)
    if writer.dumps(document).encode("utf-8") != content:
        raise generator.CorpusError(f"golden {golden_path} is not canonical")
    return document


def compare_with_comparator(
    actual: Path,
    golden: Path,
    profile: str,
    golden_sha256: str,
    *,
    metadata_only: bool = False,
) -> tuple[int, str]:
    """Compare one captured trace with the golden via the comparator CLI."""
    command = [
        sys.executable,
        str(COMPARE),
        "trace",
        str(actual),
        str(golden),
        "--profile",
        profile,
        "--golden-sha256",
        golden_sha256,
        "--max-reported",
        "3",
    ]
    if metadata_only:
        command.append("--metadata-only")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr)


def compare_iteration_with_comparator(
    actual_iteration: Path,
    golden: Path,
    logical_index: int,
    golden_sha256: str,
) -> tuple[int, str]:
    """Compare one replayed iteration snapshot with the golden via the CLI."""
    command = [
        sys.executable,
        str(COMPARE),
        "iteration",
        str(actual_iteration),
        str(golden),
        "--iteration",
        str(logical_index),
        "--profile",
        "cpu_replay_v1",
        "--golden-sha256",
        golden_sha256,
        "--max-reported",
        "3",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout + result.stderr)


def run_capture(arguments: argparse.Namespace, goldens: dict[str, dict]) -> int:
    """Run every requested case sequentially through the CPU driver."""
    failures = 0
    corpus_dir = arguments.corpus_dir
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    for case_id in arguments.cases:
        golden_entry = goldens[case_id]
        spec_path = corpus_dir / "specs" / f"{case_id}.spec"
        golden_path = corpus_dir / golden_entry["path"]
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
            actual_path,
            golden_path,
            arguments.profile,
            golden_entry["sha256"],
            metadata_only=arguments.metadata_only,
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


def split_batch_lanes(stdout: str) -> list[tuple[int, str, str]]:
    """Split the batch raw output into (lane index, status line, raw body)."""
    lanes: list[tuple[int, str, str]] = []
    current_raw: list[str] = []
    current_status = ""
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if line.startswith("batch_system "):
            if current_status or current_raw:
                lanes.append((len(lanes), current_status, "\n".join(current_raw)))
            current_raw = []
            current_status = ""
        elif line.startswith("status "):
            current_status = line
        else:
            current_raw.append(line)
    if current_status or current_raw:
        lanes.append((len(lanes), current_status, "\n".join(current_raw)))
    return lanes


def status_bits(status: str) -> tuple[int, int]:
    """Parse the batch "status <int> iterations <int>" summary line."""
    parts = status.split()
    return int(parts[1]), int(parts[3])


def run_batch(arguments: argparse.Namespace, goldens: dict[str, dict]) -> int:
    """Run the ragged heterogeneous batch and compare every lane."""
    failures = 0
    corpus_dir = arguments.corpus_dir
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    cases = list(arguments.batch_cases)
    if len(cases) < 2:
        print("--batch-cases needs at least two cases")  # noqa: T201
        return 1
    spec_paths = [str(corpus_dir / "specs" / f"{case_id}.spec") for case_id in cases]
    # Controlled per-system failure lane: a copy of the first case whose H0 is
    # poisoned with NaN.  It must fail exactly its first preparation and leave
    # the healthy peers fully converging.
    poison_index = len(cases)
    spec_paths.append(spec_paths[0])

    result = subprocess.run(
        [str(arguments.batch_capture), *spec_paths, "--poison", str(poison_index)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(  # noqa: T201 - CLI diagnostics
            f"batch capture died with {result.returncode}: {result.stderr}"
        )
        return 1

    lanes = split_batch_lanes(result.stdout)
    if len(lanes) != len(cases) + 1:
        print(  # noqa: T201
            f"batch emitted {len(lanes)} lanes, expected {len(cases) + 1}"
        )
        return 1

    for index, (_, status_line, raw_body) in enumerate(lanes):
        status_value, banner_iterations = status_bits(status_line)
        if index == poison_index:
            # The poisoned lane must fail exactly its first preparation (both
            # INTERNAL_ERROR status and zero completed iterations) without
            # corrupting the peers.  Its emitted raw stream is deliberately a
            # degenerate recorder stream: the poisoned H0 is NaN by
            # construction, so the payload cannot form a canonical JSON trace
            # document and is validated only through the banner plus a
            # zero-iteration, terminal-failed header instead of a golden
            # comparison.
            if status_value != STATUS_INTERNAL_ERROR:
                failures += 1
                print(  # noqa: T201
                    f"batch lane {index}: FAIL expected INTERNAL_ERROR "
                    f"(status {STATUS_INTERNAL_ERROR}), got {status_value} "
                    f"iterations {banner_iterations}"
                )
                continue
            if banner_iterations != 0:
                failures += 1
                print(  # noqa: T201
                    f"batch lane {index}: FAIL failing lane advanced "
                    f"{banner_iterations} iterations"
                )
                continue
            header = raw_body.splitlines()[0].split() if raw_body.splitlines() else []
            if (
                len(header) < 10
                or header[0] != "nat"
                or header[6] != "niterations"
                or header[7] != "0"
                or header[8] != "terminal"
                or header[9] != "3"
            ):
                failures += 1
                first_line = raw_body.splitlines()[0] if raw_body.splitlines() else ""
                print(  # noqa: T201
                    f"batch lane {index}: FAIL expected a zero-iteration "
                    f"terminal-failed recorder header, got {first_line}"
                )
                continue
            print(f"batch lane {index} (failure isolation): PASS")  # noqa: T201
            continue

        case_id = cases[index]
        golden_entry = goldens[case_id]
        golden_path = corpus_dir / golden_entry["path"]
        actual_path = work_dir / f"batch_lane{index}_{case_id}.json"
        try:
            trace = canonicalize_capture(raw_body, case_id)
            actual_path.write_text(writer.dumps(trace), encoding="utf-8")
        except (
            writer.TraceError,
            generator.CorpusError,
            ValueError,
            AssertionError,
            IndexError,
        ) as error:
            failures += 1
            print(f"batch lane {index} {case_id}: FAIL cannot canonicalize: {error}")  # noqa: T201
            continue

        return_code, report = compare_with_comparator(
            actual_path,
            golden_path,
            arguments.profile,
            golden_entry["sha256"],
        )
        if return_code == 0:
            print(  # noqa: T201
                f"batch lane {index} {case_id}: PASS ({arguments.profile}, "
                f"{banner_iterations} iterations)"
            )
        else:
            failures += 1
            summary = report.strip().splitlines()
            print(  # noqa: T201
                f"batch lane {index} {case_id}: FAIL ({arguments.profile})"
            )
            for line in summary[:6]:
                print(f"  {line}")  # noqa: T201
    return 1 if failures else 0


def write_mixed_state(state_path: Path, iteration: dict, nat: int, nsh: int) -> None:
    """Write one golden iteration's mixed q/d/Q in the replay state layout."""
    lines = [f"qsh {nsh}"]
    lines.extend(repr(value) for value in iteration["mixed_qsh"][0])
    lines.append(f"qat {nat}")
    lines.extend(repr(value) for value in iteration["mixed_qat"][0])
    lines.append(f"dipoles {3 * nat}")
    for atom in iteration["mixed_dipoles"][0]:
        lines.extend(repr(value) for value in atom)
    lines.append(f"quadrupoles {6 * nat}")
    for atom in iteration["mixed_quadrupoles"][0]:
        lines.extend(repr(value) for value in atom)
    state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_replay(arguments: argparse.Namespace, goldens: dict[str, dict]) -> int:
    """Replay every golden iteration from its injected mixed state."""
    failures = 0
    corpus_dir = arguments.corpus_dir
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    total_iterations = 0
    for case_id in arguments.replay_cases:
        golden_entry = goldens[case_id]
        golden_path = corpus_dir / golden_entry["path"]
        spec_path = corpus_dir / "specs" / f"{case_id}.spec"
        try:
            golden = load_pinned_golden(golden_path, golden_entry)
        except (generator.CorpusError, writer.TraceError, ValueError) as error:
            failures += 1
            print(  # noqa: T201
                f"{case_id}: FAIL golden not pinned/canonical: {error}"
            )
            continue
        nat = golden["basis"]["n_atoms"]
        nsh = golden["basis"]["n_shells"]
        iterations = golden["iterations"]
        for logical_index in range(1, len(iterations) + 1):
            total_iterations += 1
            state_path = work_dir / f"{case_id}_it{logical_index}.state"
            snapshot_path = work_dir / f"{case_id}_it{logical_index}_cpu.json"
            previous_energy = (
                iterations[logical_index - 2]["energy"] if logical_index > 1 else 0.0
            )
            write_mixed_state(state_path, iterations[logical_index - 1], nat, nsh)
            result = subprocess.run(
                [
                    str(arguments.replay),
                    str(spec_path),
                    str(state_path),
                    str(logical_index),
                    repr(previous_energy),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: FAIL replay died with "
                    f"{result.returncode}: {result.stderr}"
                )
                continue
            try:
                trace = canonicalize_capture(result.stdout, case_id)
            except (
                writer.TraceError,
                generator.CorpusError,
                ValueError,
                AssertionError,
            ) as error:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: FAIL cannot parse "
                    f"replay: {error}"
                )
                continue
            # compare_iteration validates the snapshot against the golden
            # iteration at logical_index; canonicalize always emits index 1 for
            # its single iteration, so fix the logical index first.
            if not trace["iterations"]:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: FAIL replay produced "
                    f"no completed iteration (terminal {trace['terminal']}); "
                    "a data-level eigensolver/preparation/mixer failure left "
                    "the replayed step uncommitted"
                )
                continue
            try:
                validate_replay_lifecycle(trace, iterations[logical_index - 1])
            except generator.CorpusError as error:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: FAIL invalid replay "
                    f"lifecycle: {error}"
                )
                continue
            snapshot = trace["iterations"][0]
            snapshot["index"] = logical_index
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            return_code, report = compare_iteration_with_comparator(
                snapshot_path,
                golden_path,
                logical_index,
                golden_entry["sha256"],
            )
            if return_code == 0:
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: PASS (cpu_replay_v1)"
                )
            else:
                failures += 1
                summary = report.strip().splitlines()
                print(  # noqa: T201
                    f"{case_id} iteration {logical_index}: FAIL (cpu_replay_v1)"
                )
                for line in summary[:6]:
                    print(f"  {line}")  # noqa: T201

    if failures:
        print(f"{failures} replayed comparisons failed")  # noqa: T201
    else:
        print(f"all {total_iterations} replayed iterations passed")  # noqa: T201
    return 1 if failures else 0


def flatten_multipoles(iteration: dict) -> list[float]:
    """Flatten one iteration's multipoles in the canonical residual order."""
    return [
        *iteration["mixed_qsh"][0],
        *[value for atom in iteration["mixed_dipoles"][0] for value in atom],
        *[value for atom in iteration["mixed_quadrupoles"][0] for value in atom],
    ]


def flatten_raw(iteration: dict) -> list[float]:
    """Flatten one iteration's raw multipoles in the canonical residual order."""
    return [
        *iteration["raw_qsh"][0],
        *[value for atom in iteration["raw_dipoles"][0] for value in atom],
        *[value for atom in iteration["raw_quadrupoles"][0] for value in atom],
    ]


def run_mixer(arguments: argparse.Namespace, goldens: dict[str, dict]) -> int:
    """Replay the pinned golden residual sequence through xtbloom's mixer."""
    failures = 0
    corpus_dir = arguments.corpus_dir
    work_dir = arguments.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    for case_id in arguments.mixer_cases:
        golden_entry = goldens[case_id]
        golden_path = corpus_dir / golden_entry["path"]
        spec_path = corpus_dir / "specs" / f"{case_id}.spec"
        try:
            golden = load_pinned_golden(golden_path, golden_entry)
        except (generator.CorpusError, writer.TraceError, ValueError) as error:
            failures += 1
            print(  # noqa: T201
                f"{case_id}: FAIL golden not pinned/canonical: {error}"
            )
            continue
        iterations = golden["iterations"]
        if not iterations:
            continue
        nat = golden["basis"]["n_atoms"]
        nsh = golden["basis"]["n_shells"]
        lines = [f"nat {nat} nsh {nsh} steps {len(iterations)}"]
        dimension = nsh + 9 * nat
        for logical_index, iteration in enumerate(iterations, start=1):
            lines.append(f"step {logical_index}")
            lines.append(
                "mixed "
                + " ".join(repr(value) for value in flatten_multipoles(iteration))
            )
            lines.append(
                "raw " + " ".join(repr(value) for value in flatten_raw(iteration))
            )
        sequence_path = work_dir / f"{case_id}_sequence.txt"
        sequence_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = subprocess.run(
            [str(arguments.mixer), str(spec_path), str(sequence_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures += 1
            print(  # noqa: T201
                f"{case_id}: FAIL mixer replay died with "
                f"{result.returncode}: {result.stderr}"
            )
            continue

        # Parse the predicted next-mixed lines and compare to the golden.
        predicted: dict[int, list[float]] = {}
        for line in result.stdout.splitlines():
            if line.startswith("predicted "):
                parts = line.split()
                predicted[int(parts[1])] = [float(value) for value in parts[2:]]
        for logical_index in range(2, len(iterations) + 1):
            total_steps += 1
            if logical_index not in predicted:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} transition {logical_index - 1}->{logical_index}: "
                    "FAIL no prediction emitted"
                )
                continue
            expected = flatten_multipoles(iterations[logical_index - 1])
            actual = predicted[logical_index]
            if len(actual) != dimension:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} transition {logical_index - 1}->{logical_index}: "
                    f"FAIL predicted length {len(actual)} != {dimension}"
                )
                continue
            if any(not math.isfinite(value) for value in actual) or any(
                not math.isfinite(value) for value in expected
            ):
                failures += 1
                print(  # noqa: T201
                    f"{case_id} transition {logical_index - 1}->{logical_index}: "
                    "FAIL predicted or golden mixed state is not finite"
                )
                continue
            worst_absolute = -1.0
            worst_path = ""
            for component, (predicted_value, golden_value) in enumerate(
                zip(actual, expected, strict=True)
            ):
                scale = max(abs(predicted_value), abs(golden_value))
                tolerance = 1.0e-8 + 1.0e-9 * scale
                absolute = abs(predicted_value - golden_value)
                if absolute <= tolerance:
                    continue
                # Report the first logical component above tolerance so the
                # diagnostic localizes the earliest diverging mixed element.
                worst_absolute = absolute
                worst_path = (
                    f"mixed[{component}] (abs={absolute:.6e} "
                    f"rel={absolute / scale if scale else 0.0:.6e} "
                    f"actual={predicted_value:.17e} expected={golden_value:.17e} "
                    f"tol={tolerance:.3e})"
                )
                break
            if worst_absolute < 0.0:
                print(  # noqa: T201
                    f"{case_id} transition {logical_index - 1}->{logical_index}: "
                    "PASS (cpu_replay_v1)"
                )
            else:
                failures += 1
                print(  # noqa: T201
                    f"{case_id} transition {logical_index - 1}->{logical_index}: "
                    f"FAIL first above tolerance at {worst_path} "
                    f"(abs_err={worst_absolute:.6e})"
                )
    if failures:
        print(f"{failures} mixer transition comparisons failed")  # noqa: T201
    else:
        print(f"all {total_steps} mixer transition comparisons passed")  # noqa: T201
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the CPU trace comparison tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        help="path to the xtbloom_scc_trace_capture driver executable",
    )
    parser.add_argument(
        "--batch-capture",
        type=Path,
        help="path to the xtbloom_scc_trace_batch_capture executable",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="path to the xtbloom_scc_trace_replay executable",
    )
    parser.add_argument(
        "--mixer",
        type=Path,
        help="path to the xtbloom_scc_trace_mixer executable",
    )
    parser.add_argument(
        "--profile",
        default="cpu_closed_loop_v1",
        choices=("cpu_closed_loop_v1", "cpu_replay_v1", "cuda_replay_v1"),
        help="comparator tolerance profile for closed-loop trace comparison",
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
        "--metadata-only",
        action="store_true",
        help="compare exact dimensions, lifecycle, and convergence flags only",
    )
    parser.add_argument(
        "--cases",
        default=sorted(generator.CASES),
        nargs="*",
        help="restricted cases to compare sequentially (default: all)",
    )
    parser.add_argument(
        "--batch-cases",
        nargs="*",
        help="restricted cases to run in one ragged batch (default: none)",
    )
    parser.add_argument(
        "--replay-cases",
        nargs="*",
        help="restricted cases whose every iteration is replayed (default: none)",
    )
    parser.add_argument(
        "--mixer-cases",
        nargs="*",
        help="restricted cases whose golden residual sequence is mixer-replayed "
        "(default: none)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested capture/batch/replay/mixer modes; non-zero on fail."""
    arguments = build_parser().parse_args(argv)

    modes = [
        arguments.capture is not None,
        arguments.batch_capture is not None,
        arguments.replay is not None,
        arguments.mixer is not None,
    ]
    if sum(modes) != 1:
        print(  # noqa: T201
            "exactly one of --capture, --batch-capture, --replay, --mixer is required"
        )
        return 2

    corpus_dir = arguments.corpus_dir
    goldens = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))[
        "cases"
    ]
    if arguments.capture is not None:
        return run_capture(arguments, goldens)
    if arguments.batch_capture is not None:
        if not arguments.batch_cases:
            print("--batch-capture requires --batch-cases")  # noqa: T201
            return 2
        return run_batch(arguments, goldens)
    if arguments.replay is not None:
        if not arguments.replay_cases:
            print("--replay requires --replay-cases")  # noqa: T201
            return 2
        return run_replay(arguments, goldens)
    if not arguments.mixer_cases:
        print("--mixer requires --mixer-cases")  # noqa: T201
        return 2
    return run_mixer(arguments, goldens)


if __name__ == "__main__":
    sys.exit(main())
