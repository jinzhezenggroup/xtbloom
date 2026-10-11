"""Run the bounded, component-only CUDA SCC scheduler crossover matrix.

The harness is intentionally opt-in because each coordinate creates both
forced CUDA graph variants and executes paired FRESH replays. Its output keeps
all per-sample SCC ledgers so later endpoint qualification can inspect the
measurements without treating synthetic components as molecular evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

AO_COUNTS = (40, 41, 62, 122, 180)
BATCH_SIZES = (1, 8, 32, 64, 128, 256)
ACTIVE_DENOMINATORS = (1, 2, 4)
MAX_WARMUPS = 50
MAX_SAMPLES = 200
DEFAULT_HOST_BYTE_BUDGET = 2 * 1024**3
RETAINED_DEVICE_BYTES_SCOPE = (
    "fixture arenas, FRESH checkpoint, and known graph control/table bytes; "
    "opaque CUDA graph/executable and provider allocations excluded"
)
SOURCE_FILES = (
    "tests/cuda_scc_iteration_production_test.cu",
    "tests/support/gfn2_scc_test_case.cpp",
    "tests/support/gfn2_scc_test_case.hpp",
    "benchmarks/cuda_crossover.py",
    "benchmarks/test_cuda_crossover.py",
)


@dataclass(frozen=True, slots=True)
class Coordinate:
    """One exact AO, batch, and requested active-fraction measurement."""

    ao_count: int
    batch_size: int
    active_denominator: int

    @property
    def active_fraction(self) -> float:
        """Return the requested fraction as a numeric value."""
        return 1.0 / self.active_denominator

    @property
    def fixture(self) -> str:
        """Name the component fixture selected by the native executable."""
        return {
            40: "synthetic_C10_cluster",
            41: "synthetic_C10H_cation",
            62: "C10H22_all_trans",
            122: "C20H42_straight_chain",
            180: "synthetic_C45_cluster",
        }[self.ao_count]


def build_grid(
    ao_counts: tuple[int, ...] = AO_COUNTS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    active_denominators: tuple[int, ...] = ACTIVE_DENOMINATORS,
) -> list[Coordinate]:
    """Return the requested subset of the fixed, bounded component matrix."""
    for name, values, allowed in (
        ("AO counts", ao_counts, AO_COUNTS),
        ("batch sizes", batch_sizes, BATCH_SIZES),
        ("active denominators", active_denominators, ACTIVE_DENOMINATORS),
    ):
        if (
            not values
            or len(values) != len(set(values))
            or any(value not in allowed for value in values)
        ):
            raise ValueError(f"{name} must be a nonempty unique subset of {allowed}")
    return [
        Coordinate(ao_count, batch_size, denominator)
        for ao_count in ao_counts
        for batch_size in batch_sizes
        for denominator in active_denominators
    ]


def deterministic_active_mask(
    batch_size: int, denominator: int, seed: int
) -> list[int]:
    """Mirror the native SplitMix64/Fisher-Yates mask, including rounded counts."""
    if (
        batch_size not in BATCH_SIZES
        or denominator not in ACTIVE_DENOMINATORS
        or seed < 0
    ):
        raise ValueError("unsupported batch, active denominator, or seed")
    mask64 = (1 << 64) - 1
    state = seed

    def next_random() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask64
        return value ^ (value >> 31)

    permutation = list(range(batch_size))
    for index in range(batch_size - 1, 0, -1):
        other = next_random() % (index + 1)
        permutation[index], permutation[other] = permutation[other], permutation[index]
    active_count = (batch_size + denominator // 2) // denominator
    active = [0] * batch_size
    for system in permutation[:active_count]:
        active[system] = 1
    return active


def estimate_host_bytes(coordinate: Coordinate) -> int:
    """Conservatively preflight quadratic SCC storage for one isolated process."""
    return 16 * coordinate.batch_size * coordinate.ao_count**2 * 8


def strict_json_loads(line: str) -> dict[str, Any]:
    """Parse one strict JSON object, rejecting duplicate keys and NaN tokens."""

    def pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {value}")

    value = json.loads(
        line,
        object_pairs_hook=pairs_without_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("native output line must be a JSON object")
    return value


class NativeRecordParseError(ValueError):
    """Keep complete prefix records when a crashed process truncates later JSONL."""

    def __init__(self, diagnostic: str, records: list[dict[str, Any]]) -> None:
        super().__init__(diagnostic)
        self.records = records


def parse_native_records(stdout: str) -> list[dict[str, Any]]:
    """Parse strict JSONL, retaining the valid prefix on a malformed later line."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(strict_json_loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise NativeRecordParseError(
                f"invalid native JSONL at line {line_number}: {error}", records
            ) from error
    return records


def _integer_array(record: dict[str, Any], key: str, batch_size: int) -> list[int]:
    values = record.get(key)
    if (
        not isinstance(values, list)
        or len(values) != batch_size
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        )
    ):
        raise ValueError(f"sample field {key!r} must contain {batch_size} integers")
    return values


def validate_native_records(
    records: list[dict[str, Any]],
    coordinate: Coordinate,
    warmups: int,
    samples: int,
    seed: int,
) -> tuple[str, str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate sample completeness and return status, reason, summary, and samples."""
    summaries = [
        record
        for record in records
        if record.get("record_type") == "coordinate_summary"
    ]
    sample_records = [
        record for record in records if record.get("record_type") == "sample"
    ]
    if any(
        record.get("record_type") not in {"coordinate_summary", "sample"}
        for record in records
    ):
        raise ValueError("native output contains an unknown record type")
    if len(summaries) != 1:
        raise ValueError("native output must contain exactly one coordinate summary")
    summary = summaries[0]
    if (
        summary.get("ao_count") != coordinate.ao_count
        or summary.get("batch_size") != coordinate.batch_size
    ):
        raise ValueError(
            "native summary does not match the requested AO and batch coordinate"
        )
    status = summary.get("coordinate_status")
    if status not in {"pass", "failure", "unavailable"}:
        raise ValueError("native summary has an unknown coordinate status")
    if status == "unavailable":
        if sample_records:
            raise ValueError(
                "unavailable coordinate unexpectedly contains measurements"
            )
        reason = summary.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("unavailable coordinate must provide a reason")
        return status, reason, summary, sample_records

    expected_count = warmups + samples
    if len(sample_records) != expected_count:
        raise ValueError(
            f"expected {expected_count} sample records, found {len(sample_records)}"
        )
    observed: dict[str, set[int]] = {"warmup": set(), "measurement": set()}
    all_parity = True
    for record in sample_records:
        kind = record.get("sample_kind")
        pair_index = record.get("pair_index")
        if (
            kind not in observed
            or not isinstance(pair_index, int)
            or isinstance(pair_index, bool)
        ):
            raise ValueError("sample kind or pair index is malformed")
        if pair_index < 0 or pair_index >= (warmups if kind == "warmup" else samples):
            raise ValueError("sample pair index is out of range")
        if pair_index in observed[kind]:
            raise ValueError("duplicate sample pair index")
        observed[kind].add(pair_index)
        if record.get("execution_order") not in {"chain_then_tail", "tail_then_chain"}:
            raise ValueError("sample execution order is missing or unsupported")
        global_pair_index = pair_index + (warmups if kind == "measurement" else 0)
        expected_order = (
            "chain_then_tail" if global_pair_index % 2 == 0 else "tail_then_chain"
        )
        if record["execution_order"] != expected_order:
            raise ValueError(
                "paired execution order is not alternating deterministically"
            )
        for key in ("chain_ms", "tail_ms"):
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"sample timing {key!r} must be a nonnegative number")
        parity = record.get("state_parity")
        if not isinstance(parity, bool):
            raise ValueError("sample state_parity must be boolean")
        all_parity = all_parity and parity
        for key in (
            "maximum_energy_delta_hartree",
            "maximum_charge_delta_e",
            "maximum_state_delta",
        ):
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"sample delta {key!r} must be finite and nonnegative")
        for key in (
            "chain_iterations",
            "chain_statuses",
            "chain_converged",
            "tail_iterations",
            "tail_statuses",
            "tail_converged",
        ):
            values = _integer_array(record, key, coordinate.batch_size)
            if key.endswith("_iterations") and any(value < 0 for value in values):
                raise ValueError(
                    f"sample field {key!r} contains a negative iteration count"
                )
            if key.endswith("_converged") and any(
                value not in {0, 1} for value in values
            ):
                raise ValueError(
                    f"sample field {key!r} contains an invalid convergence flag"
                )
        for ledger in ("iterations", "statuses", "converged"):
            if parity and record[f"chain_{ledger}"] != record[f"tail_{ledger}"]:
                raise ValueError(
                    f"sample state_parity contradicts chain/tail {ledger} vectors"
                )
    if observed["warmup"] != set(range(warmups)) or observed["measurement"] != set(
        range(samples)
    ):
        raise ValueError("native sample indices are incomplete")
    expected_mask = deterministic_active_mask(
        coordinate.batch_size, coordinate.active_denominator, seed
    )
    if summary.get("active_mask") != expected_mask:
        raise ValueError(
            "native active mask differs from the reproducible requested mask"
        )
    if summary.get("active_denominator") != coordinate.active_denominator:
        raise ValueError("native summary active fraction does not match the request")
    active_count = sum(expected_mask)
    if summary.get("active_system_count") != active_count:
        raise ValueError(
            "native active system count differs from the reproducible mask"
        )
    effective_fraction = summary.get("effective_active_fraction")
    if (
        not isinstance(effective_fraction, (int, float))
        or isinstance(effective_fraction, bool)
        or not math.isfinite(effective_fraction)
        or not math.isclose(effective_fraction, active_count / coordinate.batch_size)
    ):
        raise ValueError("native effective active fraction is inconsistent")
    if summary.get("qualification") != "component_only":
        raise ValueError(
            "native summary does not carry the component-only qualification"
        )
    if (
        summary.get("warmup_pairs") != warmups
        or summary.get("measured_pairs") != samples
    ):
        raise ValueError(
            "native warmup or measured-pair count differs from the finite plan"
        )
    if summary.get("state_parity") is not all_parity:
        raise ValueError("coordinate parity summary disagrees with its sample records")
    if status == "pass" and not all_parity:
        raise ValueError(
            "native summary reports pass despite a failed state-parity sample"
        )
    setup_ms = summary.get("setup_ms")
    if (
        not isinstance(setup_ms, (int, float))
        or isinstance(setup_ms, bool)
        or not math.isfinite(setup_ms)
        or setup_ms < 0
    ):
        raise ValueError("native setup_ms is missing or invalid")
    byte_fields = (
        "fixture_arena_bytes",
        "fresh_checkpoint_bytes",
        "fixture_device_bytes",
        "chain_known_control_table_bytes",
        "tail_known_control_table_bytes",
        "retained_device_bytes",
    )
    for key in byte_fields:
        value = summary.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"native summary byte field {key!r} is missing or invalid")
    if summary.get("retained_device_bytes_scope") != RETAINED_DEVICE_BYTES_SCOPE:
        raise ValueError(
            "native summary does not define retained-byte accounting scope"
        )
    if summary["fixture_device_bytes"] != (
        summary["fixture_arena_bytes"] + summary["fresh_checkpoint_bytes"]
    ):
        raise ValueError("fixture checkpoint byte accounting is inconsistent")
    retained = (
        summary["fixture_device_bytes"]
        + summary["chain_known_control_table_bytes"]
        + summary["tail_known_control_table_bytes"]
    )
    if retained != summary["retained_device_bytes"]:
        raise ValueError("retained-device byte accounting is inconsistent")
    counts = summary.get("execution_counts")
    expected_calls = warmups + samples
    if counts != {
        "chain_calls": expected_calls,
        "tail_calls": expected_calls,
        "total_calls": 2 * expected_calls,
    }:
        raise ValueError(
            "native execution counts do not match the requested finite sample plan"
        )
    executable_counts = summary.get("graph_executable_counts")
    if not isinstance(executable_counts, dict):
        raise ValueError("native graph executable counts are missing")
    chain_family = executable_counts.get("chain_dispatch_family")
    if (
        not isinstance(chain_family, int)
        or isinstance(chain_family, bool)
        or chain_family <= 0
        or executable_counts
        != {
            "chain_dispatch_family": chain_family,
            "chain_total": chain_family + 1,
            "tail_total": 2,
        }
    ):
        raise ValueError("native graph executable counts are inconsistent")
    return status, None, summary, sample_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def source_identity(
    root: Path, binary: Path, cmake_cache: Path | None
) -> dict[str, Any]:
    """Hash the executable and the exact source/build descriptors used by the run."""
    source_hashes = {
        relative: _sha256(root / relative)
        for relative in SOURCE_FILES
        if (root / relative).is_file()
    }
    status = _git_value(root, "status", "--porcelain", "--untracked-files=all")
    cache_text: dict[str, str] = {}
    cache_hash = None
    if cmake_cache is not None:
        cache_hash = _sha256(cmake_cache)
        for line in cmake_cache.read_text(encoding="utf-8").splitlines():
            if line.startswith(
                (
                    "CMAKE_BUILD_TYPE:",
                    "CMAKE_CXX_COMPILER:",
                    "CMAKE_CUDA_COMPILER:",
                    "CMAKE_CUDA_COMPILER_VERSION:",
                    "CMAKE_CUDA_ARCHITECTURES:",
                    "CMAKE_CUDA_FLAGS:",
                    "CMAKE_CXX_FLAGS:",
                    "CMAKE_CUDA_COMPILER_LAUNCHER:",
                    "CUDAToolkit_VERSION:",
                )
            ):
                key, _, value = line.partition("=")
                cache_text[key] = value
    return {
        "source_revision": _git_value(root, "rev-parse", "HEAD"),
        "source_branch": _git_value(root, "branch", "--show-current"),
        "working_tree_dirty": None if status is None else bool(status),
        "source_file_sha256": source_hashes,
        "binary_path": str(binary.resolve()),
        "binary_sha256": _sha256(binary),
        "cmake_cache_path": str(cmake_cache.resolve())
        if cmake_cache is not None
        else None,
        "cmake_cache_sha256": cache_hash,
        "cmake_build_identity": cache_text,
    }


def _parse_integer_list(
    value: str, allowed: tuple[int, ...], label: str
) -> tuple[int, ...]:
    try:
        values = tuple(int(item, 10) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if (
        not values
        or len(values) != len(set(values))
        or any(item not in allowed for item in values)
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must be a unique subset of {allowed}"
        )
    return values


def _parse_fractions(value: str) -> tuple[int, ...]:
    mapping = {"1": 1, "0.5": 2, "0.25": 4}
    items = tuple(value.split(","))
    if (
        not items
        or len(items) != len(set(items))
        or any(item not in mapping for item in items)
    ):
        raise argparse.ArgumentTypeError(
            "active fractions must be selected from 1,0.5,0.25"
        )
    return tuple(mapping[item] for item in items)


def build_parser() -> argparse.ArgumentParser:
    """Create the explicit opt-in command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--run", action="store_true", help="execute the selected CUDA matrix"
    )
    action.add_argument(
        "--list-grid", action="store_true", help="print coordinates without execution"
    )
    parser.add_argument("--binary", type=Path, help="CUDA production-test executable")
    parser.add_argument(
        "--cmake-cache", type=Path, help="CMakeCache.txt for compiler/build provenance"
    )
    parser.add_argument("--output", type=Path, help="new JSONL output path")
    parser.add_argument(
        "--ao-counts",
        type=lambda value: _parse_integer_list(value, AO_COUNTS, "AO counts"),
        default=AO_COUNTS,
    )
    parser.add_argument(
        "--batches",
        type=lambda value: _parse_integer_list(value, BATCH_SIZES, "batches"),
        default=BATCH_SIZES,
    )
    parser.add_argument(
        "--active-fractions", type=_parse_fractions, default=ACTIVE_DENOMINATORS
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="finite timeout for each coordinate, seconds",
    )
    parser.add_argument(
        "--max-estimated-host-bytes", type=int, default=DEFAULT_HOST_BYTE_BUDGET
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    return parser


def _decode_partial(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


def gpu_inventory() -> dict[str, Any]:
    """Record NVML inventory, not a verified mapping to native CUDA device zero."""
    argv = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,uuid",
        "--format=csv,noheader",
    ]
    association = {
        "native_selected_device_ordinal": 0,
        "native_selection_association": "UNVERIFIED",
        "qualification": "inventory_only",
        "selection_association_reason": (
            "NVML inventory is not matched to the native CUDA device ordinal; "
            "CUDA visibility can remap devices"
        ),
    }
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            **association,
            "status": "unavailable",
            "reason": str(error),
            "command": argv,
        }
    return {
        **association,
        "status": "available" if result.returncode == 0 else "unavailable",
        "command": argv,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def execution_environment() -> dict[str, Any]:
    """Record CPU affinity, BLAS thread controls, and inherited Slurm identity."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    tracked_environment = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "cpu_affinity": affinity,
        "environment": {name: os.environ.get(name) for name in tracked_environment},
    }


def run_coordinate(
    binary: Path,
    coordinate: Coordinate,
    warmups: int,
    samples: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    """Run one fixed coordinate and retain every native record or its failure."""
    argv = [
        str(binary.resolve()),
        "--benchmark-crossover-one",
        str(coordinate.ao_count),
        str(coordinate.batch_size),
        str(coordinate.active_denominator),
        str(warmups),
        str(samples),
        str(seed),
    ]
    launch_error = None
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = _decode_partial(error.stdout)
        stderr = _decode_partial(error.stderr)
        return_code = None
        timed_out = True
    except OSError as error:
        stdout = ""
        stderr = ""
        return_code = None
        launch_error = (
            f"native executable launch failed: {type(error).__name__}: {error}"
        )
    parse_diagnostic = None
    try:
        records = parse_native_records(stdout)
    except NativeRecordParseError as error:
        records = error.records
        parse_diagnostic = str(error)
    try:
        status, reason, summary, sample_records = validate_native_records(
            records, coordinate, warmups, samples, seed
        )
    except ValueError as error:
        status = "failure"
        reason = str(error)
        summary = next(
            (
                record
                for record in records
                if record.get("record_type") == "coordinate_summary"
            ),
            None,
        )
        sample_records = [
            record for record in records if record.get("record_type") == "sample"
        ]
    if launch_error is not None or parse_diagnostic is not None:
        status = "failure"
        reason = launch_error or parse_diagnostic
    elif timed_out:
        status = "failure"
        reason = f"coordinate exceeded its {timeout}s timeout"
    elif return_code not in (0, None):
        status = "failure"
        reason = f"native executable exited with status {return_code}"
    elif not records:
        status = "unavailable"
        reason = (
            "native executable emitted no CUDA records; device/runtime availability "
            "is unverified"
        )
    elif status == "pass" and summary is not None and not summary.get("state_parity"):
        status = "failure"
        reason = "chain and monolithic-tail states differ"
    return {
        "coordinate": asdict(coordinate),
        "fixture": coordinate.fixture,
        "estimated_host_bytes": estimate_host_bytes(coordinate),
        "status": status,
        "reason": reason,
        "subprocess_return_code": return_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "parse_diagnostic": parse_diagnostic,
        "native_records": records,
        "summary": summary,
        "samples": sample_records,
    }


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    """List the bounded matrix or execute it with per-coordinate time limits."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        grid = build_grid(args.ao_counts, args.batches, args.active_fractions)
    except ValueError as error:
        parser.error(str(error))
    if args.warmups < 0 or args.warmups > MAX_WARMUPS:
        parser.error(f"--warmups must be between 0 and {MAX_WARMUPS}")
    if args.samples <= 0 or args.samples > MAX_SAMPLES:
        parser.error(f"--samples must be between 1 and {MAX_SAMPLES}")
    if args.seed < 0 or args.seed > (1 << 63) - 1:
        parser.error("--seed must be a nonnegative signed 64-bit integer")
    if args.timeout <= 0 or args.timeout > 3600:
        parser.error("--timeout must be between 1 and 3600 seconds")
    if args.max_estimated_host_bytes <= 0:
        parser.error("--max-estimated-host-bytes must be positive")
    if args.list_grid:
        sys.stdout.write(
            _json_line(
                {
                    "schema_version": 1,
                    "matrix": "restricted_scc_component_crossover",
                    "coordinate_count": len(grid),
                    "warmups": args.warmups,
                    "paired_samples": args.samples,
                    "coordinates": [
                        {
                            **asdict(coordinate),
                            "active_fraction": coordinate.active_fraction,
                            "fixture": coordinate.fixture,
                            "estimated_host_bytes": estimate_host_bytes(coordinate),
                        }
                        for coordinate in grid
                    ],
                }
            )
            + "\n"
        )
        return 0

    if args.binary is None or not args.binary.is_file():
        parser.error("--run requires an existing --binary")
    if args.output is None:
        parser.error("--run requires --output")
    if args.cmake_cache is not None and not args.cmake_cache.is_file():
        parser.error("--cmake-cache must name an existing CMakeCache.txt")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error("output already exists; select a new path or pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    identity = source_identity(root, args.binary, args.cmake_cache)
    metadata = {
        "record_type": "matrix_metadata",
        "schema_version": 1,
        "benchmark": "restricted_scc_component_crossover",
        "qualification": "component_only",
        "claim_decision": "not_available_without_required_public_endpoints",
        "warmups": args.warmups,
        "paired_samples": args.samples,
        "seed": args.seed,
        "timeout_seconds_per_coordinate": args.timeout,
        "estimated_host_byte_budget": args.max_estimated_host_bytes,
        "coordinates_requested": len(grid),
        "source_identity": identity,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "execution_environment": execution_environment(),
        "gpu_inventory": gpu_inventory(),
        "timing_protocol": (
            "CUDA events surround the forced SCC-loop launch. FRESH initialization is "
            "queued before event start; state downloads follow event stop"
        ),
        "native_command_template": shlex.join(
            [
                str(args.binary.resolve()),
                "--benchmark-crossover-one",
                "AO",
                "B",
                "DENOM",
                "WARMUPS",
                "SAMPLES",
                "SEED",
            ]
        ),
    }
    counts = {"pass": 0, "failure": 0, "unavailable": 0}
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_json_line(metadata) + "\n")
        stream.flush()
        for coordinate in grid:
            estimate = estimate_host_bytes(coordinate)
            if estimate > args.max_estimated_host_bytes:
                result: dict[str, Any] = {
                    "coordinate": asdict(coordinate),
                    "fixture": coordinate.fixture,
                    "estimated_host_bytes": estimate,
                    "status": "unavailable",
                    "reason": (
                        "quadratic host-memory preflight exceeds the byte budget"
                    ),
                    "summary": None,
                    "samples": [],
                }
            else:
                result = run_coordinate(
                    args.binary,
                    coordinate,
                    args.warmups,
                    args.samples,
                    args.seed,
                    args.timeout,
                )
                result["estimated_host_bytes"] = estimate
            counts[result["status"]] += 1
            stream.write(
                _json_line({"record_type": "coordinate_result", **result}) + "\n"
            )
            stream.flush()
    sys.stdout.write(
        _json_line({"output": str(output), "coordinate_counts": counts}) + "\n"
    )
    return 0 if counts["failure"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
