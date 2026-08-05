"""Compare ``gpuxtb-scc-trace-v1`` traces and validated iteration snapshots.

This module is the comparison foundation for issue #49.  It deliberately does
not execute gpuxtb, inject an SCC state, or replay a mixer history; backend
replay harnesses provide actual snapshots and call this comparator.

Numeric leaves use
``abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))``.
Descriptor fields, iteration indices/convergence flags, terminal status, and all
integer/boolean/string leaves compare exactly.  The command-line interface is
read-only: it validates a canonical pinned golden and never updates it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MODULE_PATH = Path(__file__).resolve().parent / "gpuxtb_scc_trace.py"
_SPEC = importlib.util.spec_from_file_location("gpuxtb_scc_trace", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
TRACE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(TRACE)

FORMAT = TRACE.FORMAT
EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_INPUT_ERROR = 2


class TraceCompareError(ValueError):
    """Raised for invalid comparison configuration or CLI input."""


@dataclass(frozen=True)
class CompareProfile:
    """Versioned tolerance policy for one comparison mode."""

    name: str
    version: int
    atol: float = 0.0
    rtol: float = 0.0
    #: Optional per-path-suffix ``(atol, rtol)`` overrides.  Longer suffixes
    #: win, so a future specific field cannot be shadowed by a shorter suffix.
    per_field: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or type(self.version) is not int
            or self.version <= 0
        ):
            raise TraceCompareError(
                "comparison profile needs a name and positive version"
            )
        if self.per_field is not None and not isinstance(self.per_field, Mapping):
            raise TraceCompareError(
                f"comparison profile {self.identifier} per_field must be a mapping"
            )

        tolerances: list[tuple[str, Any]] = [("default", (self.atol, self.rtol))]
        for key, value in (self.per_field or {}).items():
            if not isinstance(key, str) or not key:
                raise TraceCompareError(
                    f"comparison profile {self.identifier} has an invalid field suffix"
                )
            tolerances.append((key, value))

        for label, tolerance in tolerances:
            if (
                not isinstance(tolerance, Sequence)
                or isinstance(tolerance, (str, bytes))
                or len(tolerance) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                    for value in tolerance
                )
            ):
                raise TraceCompareError(
                    f"comparison profile {self.identifier} has invalid {label} tolerance"
                )

    @property
    def identifier(self) -> str:
        """Stable CLI/diagnostic identifier for this profile version."""

        return f"{self.name}_v{self.version}"

    def tolerance_for(self, path: str) -> tuple[float, float]:
        """Return the most-specific tolerance override for ``path``.

        A field override applies to both a scalar field and every numerical
        leaf below an array-valued field.  Comparator paths include concrete
        sequence indices (for example ``iterations[0].residual[3]``), so the
        trailing indices must not hide the owning ``residual`` field.
        """

        overrides = sorted(
            (self.per_field or {}).items(), key=lambda item: len(item[0]), reverse=True
        )
        normalized_path = _normalized_path(path)
        field_path = re.sub(r"(?:\[\*\])+$", "", normalized_path)
        for suffix, tolerance in overrides:
            normalized_suffix = _normalized_path(suffix)
            if normalized_path.endswith(normalized_suffix) or field_path.endswith(
                normalized_suffix
            ):
                return tolerance
        return self.atol, self.rtol


#: Version 1 full CPU trajectory comparison from the same initial guess.
CPU_CLOSED_LOOP_V1 = CompareProfile(
    name="cpu_closed_loop",
    version=1,
    atol=1.0e-8,
    rtol=1.0e-9,
    per_field={
        # Residuals carry mixed units and are the loosest numerical field.
        "residual": (1.0e-7, 1.0e-7),
        "residual_rms": (1.0e-7, 1.0e-7),
        # Energies use the repository conformance magnitude.
        "energy": (1.0e-8, 1.0e-8),
    },
)

#: Version 1 single-iteration comparison for a separately executed replay.
CUDA_REPLAY_V1 = CompareProfile(
    name="cuda_replay",
    version=1,
    atol=1.0e-9,
    rtol=1.0e-10,
    per_field={
        "residual": (1.0e-8, 1.0e-8),
        "residual_rms": (1.0e-8, 1.0e-8),
    },
)

# Compatibility constants retain the original import names while diagnostics
# and CLI arguments always expose the versioned identifiers.
CPU_CLOSED_LOOP = CPU_CLOSED_LOOP_V1
CUDA_REPLAY = CUDA_REPLAY_V1

_PROFILES = {
    CPU_CLOSED_LOOP_V1.identifier: CPU_CLOSED_LOOP_V1,
    CUDA_REPLAY_V1.identifier: CUDA_REPLAY_V1,
}

# Normalized dotted paths whose complete subtrees are exact.  Sequence indices
# are normalized to ``[*]`` before matching so iteration metadata is explicit
# rather than relying only on the Python runtime type of its leaf.
EXACT_PATHS = (
    "format",
    "provenance.tblite_revision",
    "provenance.oracle_patch_sha256",
    "input.atomic_numbers",
    "input.unpaired_electrons",
    "input.spin_channels",
    "basis",
    "residual_layout",
    "iterations[*].index",
    "iterations[*].convergence",
    "failed_attempt.index",
    "terminal.iterations",
    "terminal.status",
    "terminal.converged",
)


@dataclass(frozen=True)
class Mismatch:
    """One divergent scalar location in a trace comparison."""

    path: str
    actual: Any
    expected: Any
    absolute_error: float
    relative_error: float
    tolerance: tuple[float, float]

    def render(self) -> str:
        """Render an actionable scalar diagnostic for CTest output."""

        atol, rtol = self.tolerance
        return (
            f"  {self.path}\n"
            f"      actual={self.actual!r} expected={self.expected!r}\n"
            f"      abs_err={self.absolute_error:.6e} rel_err={self.relative_error:.6e} "
            f"(tol atol={atol:.1e} rtol={rtol:.1e})"
        )


@dataclass(frozen=True)
class TraceCompareResult:
    """Outcome of comparing an actual trace to a golden trace."""

    mismatches: tuple[Mismatch, ...]
    profile: str

    @property
    def matches(self) -> bool:
        return not self.mismatches

    def render(self, max_reported: int = 5) -> str:
        """Render a deterministic summary and the first ``max_reported`` paths."""

        if max_reported <= 0:
            raise TraceCompareError("max_reported must be positive")
        header = (
            f"{FORMAT} comparison ({self.profile}): {len(self.mismatches)} "
            f"mismatch(es){', all reported' if len(self.mismatches) <= max_reported else ''}"
        )
        if not self.mismatches:
            return header
        lines = [header]
        lines.extend(mismatch.render() for mismatch in self.mismatches[:max_reported])
        if len(self.mismatches) > max_reported:
            lines.append(f"  ... and {len(self.mismatches) - max_reported} more")
        return "\n".join(lines)


def _scalar_path(parent: str, key: Any) -> str:
    return f"{parent}.{key}" if parent else str(key)


def _normalized_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def _under_exact_path(path: str) -> bool:
    normalized = _normalized_path(path)
    return any(
        normalized == prefix or normalized.startswith((prefix + ".", prefix + "["))
        for prefix in EXACT_PATHS
    )


def _is_exact(path: str, actual: Any, expected: Any) -> bool:
    if _under_exact_path(path):
        return True
    if isinstance(actual, (bool, str)) or isinstance(expected, (bool, str)):
        return True
    return isinstance(actual, int) and isinstance(expected, int)


def _metadata_array_length(value: Any) -> int | str:
    """Return a deterministic descriptor even before #157's strict validation."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return f"invalid-{type(value).__name__}"


class _Comparator:
    def __init__(self, profile: CompareProfile) -> None:
        self._profile = profile

    def _compare_scalar(self, path: str, actual: Any, expected: Any) -> Mismatch | None:
        if _is_exact(path, actual, expected):
            if type(actual) is type(expected) and actual == expected:
                return None
            return Mismatch(
                path, actual, expected, float("nan"), float("nan"), (0.0, 0.0)
            )
        try:
            actual_value = float(actual)
            expected_value = float(expected)
        except (TypeError, ValueError):
            if type(actual) is type(expected) and actual == expected:
                return None
            return Mismatch(
                path, actual, expected, float("nan"), float("nan"), (0.0, 0.0)
            )
        if not math.isfinite(actual_value) or not math.isfinite(expected_value):
            return Mismatch(
                path, actual, expected, float("nan"), float("nan"), (0.0, 0.0)
            )
        scale = max(abs(actual_value), abs(expected_value))
        atol, rtol = self._profile.tolerance_for(path)
        tolerance = atol + rtol * scale
        absolute_error = abs(actual_value - expected_value)
        relative_error = 0.0 if scale == 0.0 else absolute_error / scale
        if absolute_error <= tolerance:
            return None
        return Mismatch(
            path, actual, expected, absolute_error, relative_error, (atol, rtol)
        )

    def compare(
        self, parent: str, actual: Any, expected: Any, mismatches: list[Mismatch]
    ) -> None:
        if isinstance(actual, Mapping) and isinstance(expected, Mapping):
            for key in sorted(set(actual) | set(expected)):
                if key not in actual:
                    mismatches.append(
                        Mismatch(
                            _scalar_path(parent, key),
                            "<missing>",
                            expected[key],
                            float("nan"),
                            float("nan"),
                            (0.0, 0.0),
                        )
                    )
                    continue
                if key not in expected:
                    mismatches.append(
                        Mismatch(
                            _scalar_path(parent, key),
                            actual[key],
                            "<missing>",
                            float("nan"),
                            float("nan"),
                            (0.0, 0.0),
                        )
                    )
                    continue
                self.compare(
                    _scalar_path(parent, key), actual[key], expected[key], mismatches
                )
            return
        if (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
        ):
            if len(actual) != len(expected):
                mismatches.append(
                    Mismatch(
                        parent,
                        len(actual),
                        len(expected),
                        float("nan"),
                        float("nan"),
                        (0.0, 0.0),
                    )
                )
                return
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                self.compare(
                    f"{parent}[{index}]", actual_item, expected_item, mismatches
                )
            return
        mismatch = self._compare_scalar(parent, actual, expected)
        if mismatch is not None:
            mismatches.append(mismatch)


def _resolve_profile(profile: CompareProfile | str) -> CompareProfile:
    if isinstance(profile, CompareProfile):
        return profile
    try:
        return _PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise TraceCompareError(
            f"unknown comparison profile {profile!r}; choose one of: {choices}"
        ) from exc


def _point_charge_metadata(molecule: Mapping[str, Any]) -> Mapping[str, Any]:
    point_charges = molecule.get("point_charges")
    if point_charges is None:
        return {"present": False}
    if not isinstance(point_charges, Mapping):
        # Strict trace validation normally rejects this first.  Retain an
        # explicit projection so the helper is robust to a future validator.
        return {"present": True, "invalid": True}
    return {
        "present": True,
        "position_elements": _metadata_array_length(point_charges.get("positions", ())),
        "charge_elements": _metadata_array_length(point_charges.get("charges", ())),
        "hardness_elements": _metadata_array_length(
            point_charges.get("hardnesses", ())
        ),
    }


def _metadata_projection(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract only exact descriptors used to gate a numerical comparison."""

    molecule = trace["input"]
    iterations = trace["iterations"]
    return {
        "format": trace["format"],
        "provenance": {
            "tblite_revision": trace["provenance"]["tblite_revision"],
            "oracle_patch_sha256": trace["provenance"]["oracle_patch_sha256"],
        },
        "input": {
            "atomic_numbers": molecule["atomic_numbers"],
            "unpaired_electrons": molecule["unpaired_electrons"],
            "spin_channels": molecule["spin_channels"],
            "point_charges": _point_charge_metadata(molecule),
        },
        "basis": trace["basis"],
        "residual_layout": trace["residual_layout"],
        "iteration_count": len(iterations),
        "iterations": [
            {
                "index": iteration["index"],
                "convergence": iteration["convergence"],
            }
            for iteration in iterations
        ],
        "failed_attempt": (
            None
            if "failed_attempt" not in trace
            else {"index": trace["failed_attempt"]["index"]}
        ),
        "terminal": {
            "iterations": trace["terminal"]["iterations"],
            "status": trace["terminal"]["status"],
            "converged": trace["terminal"]["converged"],
        },
    }


def compare_trace(
    actual: Any,
    expected: Any,
    profile: CompareProfile | str = CPU_CLOSED_LOOP_V1.identifier,
    *,
    identical_metadata_only: bool = False,
) -> TraceCompareResult:
    """Compare an actual trace to a validated golden trace.

    ``identical_metadata_only`` compares an explicit metadata projection,
    including every iteration index/convergence flag, failed-attempt presence,
    and terminal status/count before a caller performs a numerical comparison.
    """

    resolved = _resolve_profile(profile)
    TRACE.validate(actual)
    TRACE.validate(expected)
    if identical_metadata_only:
        actual = _metadata_projection(actual)
        expected = _metadata_projection(expected)
    mismatches: list[Mismatch] = []
    _Comparator(resolved).compare("", actual, expected, mismatches)
    return TraceCompareResult(tuple(mismatches), resolved.identifier)


def compare_iteration(
    actual: Mapping[str, Any],
    expected_trace: Mapping[str, Any],
    logical_index: int,
    profile: CompareProfile | str = CUDA_REPLAY_V1.identifier,
) -> TraceCompareResult:
    """Compare one externally executed iteration snapshot to its golden entry.

    The actual snapshot is inserted into a deep copy of ``expected_trace`` and
    validated by the strict version-1 trace validator before comparison.  This
    supplies every dimensional and lifecycle invariant without accepting two
    equally malformed standalone dictionaries.  The function does not inject
    state or execute gpuxtb; a backend replay harness must do that work.
    """

    resolved = _resolve_profile(profile)
    position, expected = _validated_iteration_context(
        actual, expected_trace, logical_index
    )
    mismatches: list[Mismatch] = []
    _Comparator(resolved).compare(
        f"iterations[{position}]", dict(actual), dict(expected), mismatches
    )
    return TraceCompareResult(tuple(mismatches), resolved.identifier)


def _read_json(path: Path, label: str) -> tuple[bytes, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise TraceCompareError(f"cannot read {label} {path}: {exc}") from exc
    try:
        return content, json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceCompareError(
            f"{label} {path} is not valid UTF-8 JSON: {exc}"
        ) from exc


def _validate_expected_sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise TraceCompareError(
            "--golden-sha256 must be exactly 64 hexadecimal characters"
        )
    return normalized


def _load_golden(path: Path, expected_sha256: str | None) -> Mapping[str, Any]:
    content, golden = _read_json(path, "golden")
    if expected_sha256 is not None:
        actual_sha256 = hashlib.sha256(content).hexdigest()
        pinned_sha256 = _validate_expected_sha256(expected_sha256)
        if actual_sha256 != pinned_sha256:
            raise TraceCompareError(
                f"golden {path} SHA-256 mismatch: got {actual_sha256}, expected {pinned_sha256}"
            )
    TRACE.validate(golden)
    canonical = TRACE.dumps(golden).encode("utf-8")
    if content != canonical:
        raise TraceCompareError(
            f"golden {path} is valid but not canonical {FORMAT} JSON"
        )
    return golden


def _select_iteration(
    trace: Mapping[str, Any], logical_index: int
) -> tuple[int, Mapping[str, Any]]:
    matches = [
        (position, iteration)
        for position, iteration in enumerate(trace["iterations"])
        if iteration.get("index") == logical_index
    ]
    if len(matches) != 1:
        raise TraceCompareError(
            f"golden trace has no unique logical iteration {logical_index}"
        )
    return matches[0]


def _validated_iteration_context(
    iteration: Mapping[str, Any], expected_trace: Mapping[str, Any], logical_index: int
) -> tuple[int, Mapping[str, Any]]:
    """Validate a snapshot in a self-consistent prefix of its golden context.

    Golden terminal state and later iterations are intentionally excluded.
    A replay may legitimately converge differently from the golden; that is a
    scientific mismatch to report, not a malformed-input error.  The prefix
    still supplies all dimensions and the logical index needed by strict trace
    validation.
    """

    TRACE.validate(expected_trace)
    position, expected = _select_iteration(expected_trace, logical_index)
    candidate = dict(deepcopy(expected_trace))
    candidate["iterations"] = [
        *list(candidate["iterations"][:position]),
        deepcopy(iteration),
    ]
    candidate.pop("failed_attempt", None)
    convergence = (
        iteration.get("convergence") if isinstance(iteration, Mapping) else None
    )
    converged = isinstance(convergence, Mapping) and convergence.get("overall") is True
    candidate["terminal"] = {
        "status": (
            TRACE.STATUS_CONVERGED if converged else TRACE.STATUS_MAX_ITERATIONS
        ),
        "converged": converged,
        "iterations": position + 1,
    }
    TRACE.validate(candidate)
    return position, expected


def validate_iteration(
    iteration: Mapping[str, Any], expected_trace: Mapping[str, Any], logical_index: int
) -> None:
    """Validate one standalone snapshot in its complete golden trace context."""

    _validated_iteration_context(iteration, expected_trace, logical_index)


def _positive_integer(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return integer


def build_parser() -> argparse.ArgumentParser:
    """Build the read-only trace-comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser("trace", help="compare two complete traces")
    trace_parser.add_argument("actual", type=Path, help="actual trace JSON")
    trace_parser.add_argument("golden", type=Path, help="canonical pinned golden JSON")
    trace_parser.add_argument(
        "--profile",
        choices=sorted(_PROFILES),
        default=CPU_CLOSED_LOOP_V1.identifier,
    )
    trace_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="compare only exact descriptors and per-iteration convergence metadata",
    )
    trace_parser.add_argument(
        "--golden-sha256",
        help="optional SHA-256 pin for the exact canonical golden bytes",
    )
    trace_parser.add_argument("--max-reported", type=_positive_integer, default=5)

    iteration_parser = subparsers.add_parser(
        "iteration", help="compare one iteration snapshot to a complete golden trace"
    )
    iteration_parser.add_argument(
        "actual", type=Path, help="actual iteration JSON object"
    )
    iteration_parser.add_argument(
        "golden", type=Path, help="canonical pinned golden trace"
    )
    iteration_parser.add_argument(
        "--iteration",
        type=_positive_integer,
        required=True,
        help="one-based logical iteration index in the golden",
    )
    iteration_parser.add_argument(
        "--profile",
        choices=sorted(_PROFILES),
        default=CUDA_REPLAY_V1.identifier,
    )
    iteration_parser.add_argument(
        "--golden-sha256",
        help="optional SHA-256 pin for the exact canonical golden bytes",
    )
    iteration_parser.add_argument("--max-reported", type=_positive_integer, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return 0 match, 1 mismatch, or 2 input/tool error."""

    arguments = build_parser().parse_args(argv)
    try:
        golden = _load_golden(arguments.golden, arguments.golden_sha256)
        if arguments.command == "trace":
            _, actual = _read_json(arguments.actual, "actual trace")
            result = compare_trace(
                actual,
                golden,
                profile=arguments.profile,
                identical_metadata_only=arguments.metadata_only,
            )
        else:
            _, actual = _read_json(arguments.actual, "actual iteration")
            result = compare_iteration(
                actual,
                golden,
                arguments.iteration,
                profile=arguments.profile,
            )
    except (TraceCompareError, TRACE.TraceError, OverflowError) as exc:
        # The trace validator converts every numerical leaf to ``float``.
        # JSON integers outside the platform float range raise OverflowError;
        # classify those documents as input errors instead of leaking a
        # traceback and accidentally using the scientific-mismatch exit code.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print(result.render(arguments.max_reported))
    return EXIT_MATCH if result.matches else EXIT_MISMATCH


__all__ = [
    "CPU_CLOSED_LOOP",
    "CPU_CLOSED_LOOP_V1",
    "CUDA_REPLAY",
    "CUDA_REPLAY_V1",
    "EXACT_PATHS",
    "EXIT_INPUT_ERROR",
    "EXIT_MATCH",
    "EXIT_MISMATCH",
    "FORMAT",
    "CompareProfile",
    "Mismatch",
    "TraceCompareError",
    "TraceCompareResult",
    "build_parser",
    "compare_iteration",
    "compare_trace",
    "main",
    "validate_iteration",
]


if __name__ == "__main__":
    raise SystemExit(main())
