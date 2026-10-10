"""Validate saved evidence from the diagnostic-only two-origin SCC replay.

The checker consumes the replay test's stdout and flat snapshot directory. It
does not run xTBloom or independently evaluate the coefficient gauge: GAUGE
records are required and treated as assertions from the production comparator.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

_REAL_FIELD_GROUPS = (
    (
        3e-9,
        "scc_shell_inputs scc_dipole_inputs scc_quadrupole_inputs mixer_current_inputs "
        "mixer_previous_inputs mixer_previous_residuals mixer_df_history "
        "mixer_u_history "
        "mixer_omega mixer_residual_rms mixer_residual_maximum",
    ),
    (1e-10, "scc_free_energy scc_previous_free_energy scc_free_energy_change"),
    (
        3e-9,
        "scc_residual_rms published_shell_population published_atomic_population "
        "published_dipole published_quadrupole",
    ),
    (1e-12, "hamiltonian eigenvalues"),
    (-1.0, "coefficients"),
    (
        1e-10,
        "occupations density weighted_density raw_shell_population "
        "raw_atomic_population "
        "raw_dipole raw_quadrupole core_energy es2_energy es3_energy aes2_energy "
        "d4_pair_energy spin_energy entropy internal_energy band_energy "
        "free_energy_output",
    ),
)
REAL_TOLERANCES = {
    name: tolerance
    for tolerance, fields in _REAL_FIELD_GROUPS
    for name in fields.split()
}
COUNTER_FIELDS = ("mixer_iterations", "mixer_restarts", "scc_iterations")
STATUS_FIELDS = ("mixer_status", "scc_status")
FLAG_FIELDS = (
    "mixer_initialized",
    "mixer_residual_converged",
    "scc_terminal_converged",
    "cpu_mixer_terminal_converged",
)
CHECKPOINTS = (
    "cpu_pre",
    "cuda_pre",
    "cpu_origin_cpu_post",
    "cpu_origin_cuda_post",
    "cuda_origin_cpu_post",
    "cuda_origin_cuda_post",
)
INPUT_SCHEMA = (
    ("positions", "f64"),
    ("molecular_charges", "f64"),
    ("atomic_numbers", "i32"),
    ("unpaired_electrons", "i32"),
    ("spin_channels", "i32"),
    ("atom_offsets", "i64"),
)
DTYPES = {
    "f64": ("<d", 8),
    "u64": ("<Q", 8),
    "i32": ("<i", 4),
    "i64": ("<q", 8),
    "u8": ("<B", 1),
}
MAX_ELEMENTS = 10_000_000
MAX_TOTAL_BYTES = 512 * 1024 * 1024
FLOAT_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


class EvidenceError(ValueError):
    """Raised when the stdout ledger or snapshot set is incomplete or invalid."""


@dataclass(frozen=True)
class Snapshot:
    """One validated binary field, retaining integers without float conversion."""

    prefix: str
    name: str
    dtype: str
    values: tuple[int | float, ...]
    filename: str


@dataclass(frozen=True)
class ReplayRecord:
    """One stdout comparison result emitted by the production replay test."""

    origin: str
    field: str
    count: int
    maximum: float
    index: int
    limit: float
    passed: bool


@dataclass(frozen=True)
class ValidationResult:
    """Numerical outcome after the complete evidence set passes validation."""

    passed: bool
    final_line: str
    failed_origins: tuple[str, ...]
    failures: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def compare_values(
    left: tuple[int | float, ...],
    right: tuple[int | float, ...],
    *,
    floating: bool,
    tolerance: float,
) -> tuple[float, int, bool]:
    """Reproduce the logged maximum/index and the comparator's pass decision."""
    _require(
        len(left) == len(right) and bool(left),
        "replay arrays must have equal nonzero size",
    )
    maximum = 0.0
    worst = 0
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        difference = abs(float(left_value) - float(right_value))
        if difference > maximum:
            maximum = difference
            worst = index
    passed = maximum <= tolerance if floating else left == right
    return maximum, worst, passed


def _checkpoint_schema() -> tuple[tuple[str, str], ...]:
    return (
        *((name, "f64") for name in REAL_TOLERANCES),
        *((name, "u64") for name in COUNTER_FIELDS),
        *((name, "i32") for name in STATUS_FIELDS),
        *((name, "u8") for name in FLAG_FIELDS),
    )


def _parse_stdout(text: str) -> dict[str, object]:
    """Parse only the reserved replay ledger records, rejecting malformed ones."""
    parsed: dict[str, object] = {
        "snapshots": [],
        "prefixes": [],
        "restored": [],
        "replays": [],
        "gauges": [],
        "origins": [],
    }
    patterns = {
        "snapshot": re.compile(r"SNAPSHOT (\w+) (\w+) (\d+) (\S+)"),
        "prefix": re.compile(
            rf"PREFIX transition=(\d+) cpu_iterations=(\d+) cpu_energy=({FLOAT_TOKEN})"
        ),
        "start": re.compile(
            r"TWO_ORIGIN directory=(.+) transition=20 prefix_steps=19 "
            r"expected_replays=4"
        ),
        "policy": re.compile(
            rf"POLICY maximum_iterations=(\d+) history=(\d+) "
            rf"energy_tolerance=({FLOAT_TOKEN}) residual_tolerance=({FLOAT_TOKEN}) "
            rf"electronic_temperature_hartree=({FLOAT_TOKEN}) spin_channels=(\d+)"
        ),
        "prestate": re.compile(
            r"PRESTATE actual_independent_cuda_origin=([01]) "
            r"transition_inputs_distinct=([01])"
        ),
        "restored": re.compile(r"RESTORED origin=(cpu|cuda) byte_exact=([01])"),
        "replay": re.compile(
            rf"REPLAY origin=(cpu|cuda) field=([a-z0-9_]+) count=(\d+) "
            rf"max=({FLOAT_TOKEN}) index=(\d+) limit=({FLOAT_TOKEN}) pass=([01])"
        ),
        "gauge": re.compile(
            rf"GAUGE origin=(cpu|cuda) spectrum_limit=({FLOAT_TOKEN}) "
            rf"subspace_limit=({FLOAT_TOKEN}) pass=([01])"
        ),
        "origin": re.compile(r"ORIGIN origin=(cpu|cuda) replays=(\d+) pass=([01])"),
        "final": re.compile(
            r"TWO_ORIGIN completed_replays=4 diagnostic_pass=([01]) "
            r"public_parity_unresolved=1"
        ),
    }
    tags = {
        "SNAPSHOT": "snapshot",
        "PREFIX": "prefix",
        "TWO_ORIGIN directory=": "start",
        "POLICY": "policy",
        "PRESTATE": "prestate",
        "RESTORED": "restored",
        "REPLAY": "replay",
        "GAUGE": "gauge",
        "ORIGIN": "origin",
        "TWO_ORIGIN completed_replays=": "final",
    }
    for line in text.splitlines():
        if line.startswith("TWO_ORIGIN ") and not line.startswith(
            ("TWO_ORIGIN directory=", "TWO_ORIGIN completed_replays=")
        ):
            raise EvidenceError(f"unexpected TWO_ORIGIN record: {line!r}")
        kind = next(
            (value for tag, value in tags.items() if line.startswith(tag)), None
        )
        if kind is None:
            continue
        match = patterns[kind].fullmatch(line)
        _require(match is not None, f"malformed {kind} record: {line!r}")
        groups = match.groups()
        if kind == "snapshot":
            prefix, dtype, raw_count, filename = groups
            count = int(raw_count)
            _require(
                0 < count <= MAX_ELEMENTS, f"invalid snapshot count for {filename}"
            )
            parsed["snapshots"].append((prefix, dtype, count, filename))
        elif kind == "prefix":
            transition, iterations, energy = groups
            value = float(energy)
            _require(math.isfinite(value), "PREFIX energy must be finite")
            parsed["prefixes"].append((int(transition), int(iterations), value))
        elif kind == "start":
            _require("start" not in parsed, "duplicate TWO_ORIGIN start record")
            parsed["start"] = groups[0]
        elif kind == "policy":
            _require("policy" not in parsed, "duplicate POLICY record")
            parsed["policy"] = (
                int(groups[0]),
                int(groups[1]),
                *(float(value) for value in groups[2:5]),
                int(groups[5]),
            )
        elif kind == "prestate":
            _require("prestate" not in parsed, "duplicate PRESTATE record")
            parsed["prestate"] = tuple(int(value) for value in groups)
        elif kind == "restored":
            parsed["restored"].append((groups[0], int(groups[1])))
        elif kind == "replay":
            origin, field, count, maximum, index, limit, passed = groups
            parsed["replays"].append(
                ReplayRecord(
                    origin,
                    field,
                    int(count),
                    float(maximum),
                    int(index),
                    float(limit),
                    passed == "1",
                )
            )
        elif kind == "gauge":
            origin, spectrum, subspace, passed = groups
            parsed["gauges"].append(
                (origin, float(spectrum), float(subspace), passed == "1")
            )
        elif kind == "origin":
            parsed["origins"].append((groups[0], int(groups[1]), int(groups[2])))
        else:
            _require("final" not in parsed, "duplicate final TWO_ORIGIN record")
            parsed["final"] = (int(groups[0]), line)
    for required in ("start", "policy", "prestate", "final"):
        _require(required in parsed, f"missing {required} record")
    return parsed


def _validate_policy(parsed: dict[str, object]) -> None:
    (
        maximum_iterations,
        history,
        energy_tolerance,
        residual_tolerance,
        temperature,
        channels,
    ) = parsed["policy"]
    expected_temperature = 300.0 * 3.166808578545117e-6
    _require(
        (
            maximum_iterations,
            history,
            energy_tolerance,
            residual_tolerance,
            temperature,
            channels,
        )
        == (500, 8, 1e-10, 1e-8, expected_temperature, 2),
        "POLICY values do not match the frozen two-origin replay setup",
    )


def _load_snapshots(
    directory: Path, ledger: list[tuple[str, str, int, str]]
) -> dict[tuple[str, str], Snapshot]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        "snapshot directory must be a real directory",
    )
    expected_names: set[str] = set()
    for _prefix, dtype, _count, filename in ledger:
        _require(dtype in DTYPES, f"unsupported dtype {dtype!r}")
        _require(
            re.fullmatch(r"[A-Za-z0-9_.-]+\.bin", filename) is not None
            and Path(filename).name == filename
            and "/" not in filename
            and "\\" not in filename
            and filename not in (".", ".."),
            f"unsafe snapshot path {filename!r}",
        )
        expected_names.add(filename)
    entries = list(directory.iterdir())
    _require(
        all(not item.is_symlink() and item.is_file() for item in entries),
        "snapshot directory may contain only regular, non-symlink files",
    )
    actual_names = {item.name for item in entries}
    _require(
        actual_names == expected_names, "snapshot directory has missing or extra files"
    )
    total_bytes = 0
    snapshots: dict[tuple[str, str], Snapshot] = {}
    for prefix, dtype, count, filename in ledger:
        key = (prefix, filename.removeprefix(prefix + "_").removesuffix(".bin"))
        _require(key not in snapshots, f"duplicate snapshot field {prefix}/{key[1]}")
        format_code, width = DTYPES[dtype]
        byte_count = count * width
        total_bytes += byte_count
        _require(
            total_bytes <= MAX_TOTAL_BYTES, "snapshot payload exceeds the safety limit"
        )
        path = directory / filename
        _require(
            path.stat().st_size == byte_count,
            f"truncated or oversized snapshot {filename}",
        )
        raw = path.read_bytes()
        values = tuple(item[0] for item in struct.iter_unpack(format_code, raw))
        if dtype == "f64":
            _require(
                all(math.isfinite(value) for value in values),
                f"non-finite value in {filename}",
            )
        snapshots[key] = Snapshot(prefix, key[1], dtype, values, filename)
    return snapshots


def _validate_schema(
    ledger: list[tuple[str, str, int, str]], snapshots: dict[tuple[str, str], Snapshot]
) -> None:
    expected = [("input", name, dtype) for name, dtype in INPUT_SCHEMA]
    checkpoint_schema = _checkpoint_schema()
    for prefix in CHECKPOINTS:
        expected.extend((prefix, name, dtype) for name, dtype in checkpoint_schema)
    _require(
        len(ledger) == len(expected), "snapshot ledger has missing or extra entries"
    )
    for record, (prefix, name, dtype) in zip(ledger, expected, strict=True):
        actual_prefix, actual_dtype, _count, filename = record
        _require(
            (actual_prefix, filename, actual_dtype)
            == (prefix, f"{prefix}_{name}.bin", dtype),
            f"snapshot schema mismatch at {filename}",
        )
    input_values = {
        name: snapshots[("input", name)].values for name, _dtype in INPUT_SCHEMA
    }
    atoms = input_values["atomic_numbers"]
    _require(
        bool(atoms) and all(1 <= number <= 118 for number in atoms),
        "invalid atomic_numbers",
    )
    _require(
        len(input_values["positions"]) == 3 * len(atoms),
        "positions must contain exactly three coordinates per atom",
    )
    _require(
        input_values["atom_offsets"] == (0, len(atoms)),
        "atom_offsets must describe one complete system",
    )
    _require(input_values["molecular_charges"] == (3.0,), "unexpected molecular charge")
    _require(
        input_values["unpaired_electrons"] == (1,), "unexpected unpaired-electron count"
    )
    _require(input_values["spin_channels"] == (2,), "spin replay requires two channels")
    counts: dict[str, int] = {}
    for name, dtype in checkpoint_schema:
        count = len(snapshots[("cpu_pre", name)].values)
        _require(count > 0, f"empty checkpoint field {name}")
        if name in _scalar_fields() or name in (
            *COUNTER_FIELDS,
            *STATUS_FIELDS,
            *FLAG_FIELDS,
        ):
            _require(count == 1, f"{name} must be scalar")
        counts[name] = count
        _require(
            all(snapshots[(prefix, name)].dtype == dtype for prefix in CHECKPOINTS),
            f"wrong dtype for {name}",
        )
        _require(
            all(
                len(snapshots[(prefix, name)].values) == count for prefix in CHECKPOINTS
            ),
            f"inconsistent count for {name}",
        )
    vector_count = sum(
        counts[name]
        for name in ("scc_shell_inputs", "scc_dipole_inputs", "scc_quadrupole_inputs")
    )
    _require(
        counts["mixer_current_inputs"] == vector_count,
        "mixer vector count does not match q/d/Q inputs",
    )
    for prefix in CHECKPOINTS:
        mixer_values = snapshots[(prefix, "mixer_current_inputs")].values
        offset = 0
        for name in ("scc_shell_inputs", "scc_dipole_inputs", "scc_quadrupole_inputs"):
            for index, value in enumerate(snapshots[(prefix, name)].values):
                _require(
                    struct.pack("<d", value)
                    == struct.pack("<d", mixer_values[offset + index]),
                    f"{prefix} mixer_current_inputs does not match "
                    "concatenated SCC q/d/Q",
                )
            offset += counts[name]
    _require(
        counts["mixer_previous_inputs"] == vector_count
        and counts["mixer_previous_residuals"] == vector_count,
        "mixer input counts are inconsistent",
    )
    _require(
        counts["mixer_df_history"] == vector_count * 8
        and counts["mixer_u_history"] == vector_count * 8,
        "mixer history count does not match policy",
    )
    _require(counts["mixer_omega"] == 8, "mixer_omega count does not match policy")
    for stem, source in (
        ("published_shell_population", "scc_shell_inputs"),
        ("raw_shell_population", "scc_shell_inputs"),
        ("published_dipole", "scc_dipole_inputs"),
        ("raw_dipole", "scc_dipole_inputs"),
        ("published_quadrupole", "scc_quadrupole_inputs"),
        ("raw_quadrupole", "scc_quadrupole_inputs"),
    ):
        _require(
            counts[stem] == counts[source], f"{stem} count does not match its SCC input"
        )
    _require(
        counts["raw_atomic_population"] == counts["published_atomic_population"],
        "atomic population counts differ",
    )


def _scalar_fields() -> set[str]:
    return {
        "mixer_residual_rms",
        "mixer_residual_maximum",
        "scc_free_energy",
        "scc_previous_free_energy",
        "scc_free_energy_change",
        "scc_residual_rms",
        "core_energy",
        "es2_energy",
        "es3_energy",
        "aes2_energy",
        "d4_pair_energy",
        "spin_energy",
        "entropy",
        "internal_energy",
        "band_energy",
        "free_energy_output",
    }


def _field_map(
    snapshots: dict[tuple[str, str], Snapshot], prefix: str
) -> dict[str, tuple[int | float, ...]]:
    return {
        name: snapshots[(prefix, name)].values
        for name in (*REAL_TOLERANCES, *COUNTER_FIELDS, *STATUS_FIELDS, *FLAG_FIELDS)
    }


def _validate_state(
    snapshots: dict[tuple[str, str], Snapshot], prefix: str, step: int
) -> None:
    fields = _field_map(snapshots, prefix)
    counters = {
        name: values[0] for name, values in fields.items() if name in COUNTER_FIELDS
    }
    statuses = {
        name: values[0] for name, values in fields.items() if name in STATUS_FIELDS
    }
    flags = {name: values[0] for name, values in fields.items() if name in FLAG_FIELDS}
    _require(counters["scc_iterations"] == step, f"{prefix} has invalid scc_iterations")
    _require(
        0 < counters["mixer_iterations"] <= step,
        f"{prefix} has invalid mixer_iterations",
    )
    _require(
        counters["mixer_restarts"] <= counters["mixer_iterations"],
        f"{prefix} has invalid mixer_restarts",
    )
    _require(
        all(status in (0, 7) for status in statuses.values()),
        f"{prefix} has invalid status",
    )
    _require(
        all(flag in (0, 1) for flag in flags.values()), f"{prefix} has invalid flag"
    )
    _require(flags["mixer_initialized"] == 1, f"{prefix} mixer is not initialized")
    expected_residual_flag = int(
        counters["scc_iterations"] > 0
        and fields["mixer_residual_rms"][0] < 1e-8
        and fields["mixer_residual_maximum"][0] < 1e-8
    )
    _require(
        flags["mixer_residual_converged"] == expected_residual_flag,
        f"{prefix} residual flag is inconsistent",
    )
    if step == 19:
        _require(
            all(status == 0 for status in statuses.values()),
            f"{prefix} prestate status must be SUCCESS",
        )
        _require(
            flags["scc_terminal_converged"] == 0
            and flags["cpu_mixer_terminal_converged"] == 0,
            f"{prefix} prestate is already terminal",
        )


def _expected_replay_order() -> list[tuple[str, str]]:
    compared_real = [
        name for name, tolerance in REAL_TOLERANCES.items() if tolerance >= 0.0
    ]
    fields = [*compared_real, *COUNTER_FIELDS, *STATUS_FIELDS, *FLAG_FIELDS]
    return [(origin, name) for origin in ("cpu", "cuda") for name in fields]


def validate_evidence(stdout: str, snapshot_directory: Path) -> ValidationResult:
    """Validate complete stdout and binary evidence without running the model."""
    parsed = _parse_stdout(stdout)
    _validate_policy(parsed)
    prefixes = parsed["prefixes"]
    _require(
        len(prefixes) == 19 and [row[0] for row in prefixes] == list(range(1, 20)),
        "stdout must contain all 19 ordered PREFIX records",
    )
    _require(
        all(row[1] == row[0] for row in prefixes),
        "PREFIX cpu_iterations must match each transition",
    )
    ledger = parsed["snapshots"]
    _require(
        len(ledger) == len(INPUT_SCHEMA) + len(CHECKPOINTS) * len(_checkpoint_schema()),
        "snapshot ledger is incomplete or has extra entries",
    )
    snapshots = _load_snapshots(snapshot_directory, ledger)
    _validate_schema(ledger, snapshots)
    logged_directory = Path(parsed["start"])
    _require(
        logged_directory.resolve() == snapshot_directory.resolve(),
        "TWO_ORIGIN directory does not match the supplied snapshot directory",
    )
    _require(
        parsed["prestate"] == (1, 1),
        "PRESTATE must retain an independent, distinct transition state",
    )
    _validate_state(snapshots, "cpu_pre", 19)
    _validate_state(snapshots, "cuda_pre", 19)
    for prefix in CHECKPOINTS[:2]:
        for inputs, published in (
            ("scc_shell_inputs", "published_shell_population"),
            ("scc_dipole_inputs", "published_dipole"),
            ("scc_quadrupole_inputs", "published_quadrupole"),
        ):
            _require(
                snapshots[(prefix, inputs)].values
                == snapshots[(prefix, published)].values,
                f"{prefix} active SCC inputs differ from {published}",
            )
    _require(
        snapshots[("cpu_pre", "scc_free_energy")].values[0] == prefixes[-1][2],
        "last PREFIX energy does not match cpu_pre scc_free_energy",
    )
    distinct = any(
        snapshots[("cpu_pre", name)].values != snapshots[("cuda_pre", name)].values
        for name in tuple(REAL_TOLERANCES)[:9]
    )
    _require(distinct, "CPU and CUDA prestate q/d/Q/mixer inputs are not distinct")
    _require(
        parsed["restored"] == [("cpu", 1), ("cuda", 1)],
        "both byte-exact RESTORED origins are required",
    )
    for prefix in CHECKPOINTS[2:]:
        _validate_state(snapshots, prefix, 20)

    expected_order = _expected_replay_order()
    replay_records = parsed["replays"]
    _require(
        [(record.origin, record.field) for record in replay_records] == expected_order,
        "REPLAY ledger has missing, duplicate, extra, or reordered fields",
    )
    gauges = parsed["gauges"]
    _require(
        [(origin, spectrum, subspace) for origin, spectrum, subspace, _passed in gauges]
        == [("cpu", 1e-12, 3e-8), ("cuda", 1e-12, 3e-8)],
        "both GAUGE records with the production limits are required",
    )
    origins = parsed["origins"]
    _require(
        [(origin, replays) for origin, replays, _passed in origins]
        == [("cpu", 2), ("cuda", 2)],
        "both ORIGIN records are required",
    )
    failed_fields: list[str] = []
    replay_passes: dict[str, bool] = {"cpu": True, "cuda": True}
    for record in replay_records:
        cpu_prefix = f"{record.origin}_origin_cpu_post"
        cuda_prefix = f"{record.origin}_origin_cuda_post"
        left = snapshots[(cpu_prefix, record.field)].values
        right = snapshots[(cuda_prefix, record.field)].values
        tolerance = REAL_TOLERANCES.get(record.field, 0.0)
        floating = record.field in REAL_TOLERANCES
        _require(
            record.count == len(left),
            f"REPLAY count mismatch for {record.origin}/{record.field}",
        )
        _require(
            record.limit == tolerance,
            f"REPLAY limit mismatch for {record.origin}/{record.field}",
        )
        maximum, index, passed = compare_values(
            left, right, floating=floating, tolerance=tolerance
        )
        _require(
            record.maximum == maximum and record.index == index,
            f"REPLAY difference mismatch for {record.origin}/{record.field}",
        )
        _require(
            record.passed == passed,
            f"REPLAY pass flag mismatch for {record.origin}/{record.field}",
        )
        replay_passes[record.origin] = replay_passes[record.origin] and passed
        if not passed:
            failed_fields.append(f"{record.origin}/{record.field}")
    gauge_passes = {origin: passed for origin, _spectrum, _subspace, passed in gauges}
    expected_origin_passes = {
        origin: replay_passes[origin] and gauge_passes[origin]
        for origin in ("cpu", "cuda")
    }
    _require(
        [(origin, replays, passed) for origin, replays, passed in origins]
        == [
            (origin, 2, int(expected_origin_passes[origin]))
            for origin in ("cpu", "cuda")
        ],
        "ORIGIN outcomes do not match their REPLAY and GAUGE records",
    )
    final_pass = all(expected_origin_passes.values())
    final_value, final_line = parsed["final"]
    _require(
        final_value == int(final_pass),
        "final diagnostic outcome does not match both origins",
    )
    failed_origins = tuple(
        origin for origin in ("cpu", "cuda") if not expected_origin_passes[origin]
    )
    failures = tuple(failed_fields) + tuple(
        f"{origin}/GAUGE" for origin in failed_origins if not gauge_passes[origin]
    )
    return ValidationResult(final_pass, final_line, failed_origins, failures)


def main(argv: list[str] | None = None) -> int:
    """Run the offline checker; return 0=pass, 1=numerical fail, 2=malformed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stdout_log", type=Path, help="captured two-origin replay stdout"
    )
    parser.add_argument(
        "snapshot_directory", type=Path, help="flat directory named by TWO_ORIGIN"
    )
    arguments = parser.parse_args(argv)
    try:
        stdout = arguments.stdout_log.read_text(encoding="utf-8")
        result = validate_evidence(stdout, arguments.snapshot_directory)
    except (EvidenceError, OSError, UnicodeError) as error:
        sys.stderr.write(f"MALFORMED evidence: {error}\n")
        return 2
    for failure in result.failures:
        sys.stdout.write(f"FAIL {failure}\n")
    sys.stdout.write(f"{result.final_line}\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
