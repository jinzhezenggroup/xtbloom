"""Canonical ``gpuxtb-scc-trace-v1`` writer and structural validation.

The format is the restricted GFN2 SCC interchange contract defined by issue
#47.  The writer is deliberately standard-library only so it can run before
either Fortran reference is built.

Version 1 invariants:

- ``format`` is exactly ``gpuxtb-scc-trace-v1``.
- Restricted matrices use logical ``[spin=1][row][column]`` order, while
  occupations retain tblite's two alpha/beta channels.
- Multipoles use ``[spin=1][atom][component]`` order; quadrupoles pack
  ``xx, xy, yy, xz, yz, zz``.
- Mixer residuals flatten raw-minus-mixed shell charges, atomic dipoles, then
  atomic quadrupoles in tblite order.
- Only attempts that reach ``after_iteration`` appear in ``iterations``.  A
  solve that fails after ``before_solve`` is represented by ``failed_attempt``.
- All floats are finite and emitted with at least 17 significant digits.
- Object keys are recursively sorted, making a read/rewrite byte-identical.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

FORMAT = "gpuxtb-scc-trace-v1"

# tblite qpat uses six symmetric Cartesian components in this exact order.
QUADRUPOLE_COMPONENTS: tuple[str, ...] = ("xx", "xy", "yy", "xz", "yz", "zz")

# tblite status values surfaced by the observer's finished callback.
STATUS_CONVERGED = 1
STATUS_MAX_ITERATIONS = 2
STATUS_FAILED = 3


class TraceError(ValueError):
    """Raised when a trace is malformed or cannot be emitted canonically."""


# --- canonical JSON emission -----------------------------------------------------


def _float_text(value: float) -> str:
    """Serialize a finite float with at least 17 significant digits."""
    if not math.isfinite(value):
        raise TraceError(f"trace floats must be finite (got {value!r})")
    text = format(value, ".17g")
    if not any(character in text for character in ".eEnN"):
        # Keep an integral binary64 value on the JSON floating-point path.
        text += ".0"
    return text


def _iter_items(value: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield mapping items while rejecting non-string JSON object keys."""
    for key, item in value.items():
        if not isinstance(key, str):
            raise TraceError(f"trace object keys must be strings (got {key!r})")
        yield key, item


def _emit(value: Any, out: list[str], level: int, indent: str) -> None:
    """Append the canonical JSON representation of ``value`` to ``out``."""
    if isinstance(value, Mapping):
        items = sorted(_iter_items(value), key=lambda item: item[0])
        if not items:
            out.append("{}")
            return
        out.append("{")
        for position, (key, item) in enumerate(items):
            out.append("\n" + indent * (level + 1))
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(": ")
            _emit(item, out, level + 1, indent)
            if position + 1 < len(items):
                out.append(",")
        out.append("\n" + indent * level + "}")
        return
    if isinstance(value, (list, tuple)):
        if not value:
            out.append("[]")
            return
        out.append("[")
        for position, item in enumerate(value):
            out.append("\n" + indent * (level + 1))
            _emit(item, out, level + 1, indent)
            if position + 1 < len(value):
                out.append(",")
        out.append("\n" + indent * level + "]")
        return
    if isinstance(value, bool):
        out.append("true" if value else "false")
        return
    if isinstance(value, int):
        out.append(str(int(value)))
        return
    if isinstance(value, float):
        out.append(_float_text(value))
        return
    if value is None:
        out.append("null")
        return
    if isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
        return
    raise TraceError(f"non-JSON value {value!r}")


# --- runtime schema validation ---------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceError(message)


def _as_dict(value: Any, path: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return dict(value)


def _as_list(value: Any, path: str) -> list[Any]:
    _require(isinstance(value, (list, tuple)), f"{path} must be an array")
    return list(value)


def _as_float(value: Any, path: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be a number",
    )
    number = float(value)
    _require(math.isfinite(number), f"{path} must be finite")
    return number


def _as_int(value: Any, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    return int(value)


def _as_bool(value: Any, path: str) -> bool:
    _require(isinstance(value, bool), f"{path} must be a boolean")
    return bool(value)


def _object_fields(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
    allow_extra: bool = False,
) -> dict[str, Any]:
    """Validate one object's required and supported field names."""
    result = _as_dict(value, path)
    non_string_keys = [repr(key) for key in result if not isinstance(key, str)]
    _require(
        not non_string_keys,
        f"{path} object keys must be strings (got {', '.join(non_string_keys)})",
    )
    missing = sorted(required - result.keys())
    _require(not missing, f"{path} is missing required field(s): {', '.join(missing)}")
    if not allow_extra:
        allowed = required | (optional or set())
        extra = sorted(result.keys() - allowed)
        _require(not extra, f"{path} has unsupported field(s): {', '.join(extra)}")
    return result


def _json_value(value: Any, path: str) -> None:
    """Validate extensible provenance values as finite JSON data."""
    if isinstance(value, Mapping):
        for key, item in _iter_items(value):
            _json_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, float):
        _as_float(value, path)
        return
    _require(
        value is None or isinstance(value, (bool, int, str)),
        f"{path} must be JSON-compatible",
    )


def _matrix(path: str, value: Any, channels: int, rows: int, columns: int) -> None:
    spin_values = _as_list(value, path)
    _require(
        len(spin_values) == channels,
        f"{path} needs {channels} channels, got {len(spin_values)}",
    )
    for channel_index, channel in enumerate(spin_values):
        channel_path = f"{path}[{channel_index}]"
        data = _as_list(channel, channel_path)
        _require(
            len(data) == rows,
            f"{channel_path} needs {rows} rows, got {len(data)}",
        )
        for row_index, row in enumerate(data):
            row_path = f"{channel_path}[{row_index}]"
            row_values = _as_list(row, row_path)
            _require(
                len(row_values) == columns,
                f"{row_path} needs {columns} columns, got {len(row_values)}",
            )
            for column_index, element in enumerate(row_values):
                _as_float(element, f"{row_path}[{column_index}]")


def _spectrum(path: str, value: Any, channels: int, length: int) -> None:
    channel_values = _as_list(value, path)
    _require(
        len(channel_values) == channels,
        f"{path} needs {channels} channels, got {len(channel_values)}",
    )
    for channel_index, channel in enumerate(channel_values):
        channel_path = f"{path}[{channel_index}]"
        values = _as_list(channel, channel_path)
        _require(
            len(values) == length,
            f"{channel_path} needs {length} values, got {len(values)}",
        )
        for element_index, element in enumerate(values):
            _as_float(element, f"{channel_path}[{element_index}]")


def _multipoles(path: str, value: Any, atoms: int, components: int) -> None:
    """Validate restricted ``[spin=1][atom][component]`` multipoles."""
    spin_values = _as_list(value, path)
    _require(
        len(spin_values) == 1,
        f"{path} needs 1 spin channel, got {len(spin_values)}",
    )
    atom_values = _as_list(spin_values[0], path + "[0]")
    _require(
        len(atom_values) == atoms,
        f"{path}[0] needs {atoms} atoms, got {len(atom_values)}",
    )
    for atom_index, atom in enumerate(atom_values):
        atom_path = f"{path}[0][{atom_index}]"
        values = _as_list(atom, atom_path)
        _require(
            len(values) == components,
            f"{atom_path} needs {components} components, got {len(values)}",
        )
        for component_index, element in enumerate(values):
            _as_float(element, f"{atom_path}[{component_index}]")


def _pre_solve_state(
    entry: dict[str, Any], path: str, nao: int, n_shells: int, n_atoms: int
) -> None:
    """Validate fields captured by the observer's ``before_solve`` hook."""
    _matrix(path + ".hamiltonian", entry["hamiltonian"], 1, nao, nao)
    _spectrum(path + ".mixed_qsh", entry["mixed_qsh"], 1, n_shells)
    _spectrum(path + ".mixed_qat", entry["mixed_qat"], 1, n_atoms)
    _multipoles(path + ".mixed_dipoles", entry["mixed_dipoles"], n_atoms, 3)
    _multipoles(path + ".mixed_quadrupoles", entry["mixed_quadrupoles"], n_atoms, 6)


def _flatten_population(entry: dict[str, Any], prefix: str) -> list[float]:
    """Flatten restricted q/d/Q into the exact tblite mixer order."""
    values = [float(value) for value in entry[prefix + "_qsh"][0]]
    for atom in entry[prefix + "_dipoles"][0]:
        values.extend(float(value) for value in atom)
    for atom in entry[prefix + "_quadrupoles"][0]:
        values.extend(float(value) for value in atom)
    return values


def _within_roundoff(actual: float, expected: float, ulps: int = 8) -> bool:
    """Allow only arithmetic roundoff, never a scientific comparison tolerance."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    if actual == expected:
        return True
    tolerance = ulps * max(math.ulp(actual), math.ulp(expected))
    return abs(actual - expected) <= tolerance


def _validate_atomic_charges(
    entry: dict[str, Any], prefix: str, shell_counts: list[int], path: str
) -> None:
    """Check that qat is the per-atom reduction of contiguous qsh values."""
    shell_charges = entry[prefix + "_qsh"][0]
    atomic_charges = entry[prefix + "_qat"][0]
    shell_offset = 0
    for atom_index, shell_count in enumerate(shell_counts):
        derived = 0.0
        for shell_index in range(shell_offset, shell_offset + shell_count):
            derived += float(shell_charges[shell_index])
        actual = float(atomic_charges[atom_index])
        _require(
            _within_roundoff(actual, derived),
            f"{path}.{prefix}_qat[0][{atom_index}] must equal the atom's reduced "
            f"{prefix}_qsh (expected {derived!r}, got {actual!r})",
        )
        shell_offset += shell_count


def validate(trace: Any) -> None:
    """Validate a candidate restricted ``gpuxtb-scc-trace-v1`` document.

    The JSON Schema carries the machine-readable static contract; this runtime
    validation additionally enforces dynamic dimensions and observer lifecycle
    invariants that Draft 7 cannot express.  :class:`TraceError` messages name
    the malformed path and expected relationship.
    """
    root = _object_fields(
        trace,
        "trace",
        required={
            "format",
            "provenance",
            "input",
            "basis",
            "statics",
            "residual_layout",
            "iterations",
            "terminal",
        },
        optional={"failed_attempt"},
    )
    _require(
        root["format"] == FORMAT,
        f"unsupported trace format {root['format']!r} (expected {FORMAT!r})",
    )

    provenance = _object_fields(
        root["provenance"],
        "provenance",
        required={"tblite_revision", "oracle_patch_sha256"},
        optional={"oracle_command"},
        # Corpus generation may attach pinned compiler/toolchain metadata.
        allow_extra=True,
    )
    revision = provenance["tblite_revision"]
    _require(
        isinstance(revision, str)
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "provenance.tblite_revision must be a pinned lowercase 40-hex commit",
    )
    digest = provenance["oracle_patch_sha256"]
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "provenance.oracle_patch_sha256 must be a lowercase 64-hex digest",
    )
    if "oracle_command" in provenance:
        _require(
            isinstance(provenance["oracle_command"], str),
            "provenance.oracle_command must be a string",
        )
    for key, value in provenance.items():
        _json_value(value, f"provenance.{key}")

    molecule = _object_fields(
        root["input"],
        "input",
        required={
            "atomic_numbers",
            "positions",
            "molecular_charge",
            "unpaired_electrons",
            "spin_channels",
            "temperature",
        },
        optional={"point_charges"},
    )
    atomic_numbers = _as_list(molecule["atomic_numbers"], "input.atomic_numbers")
    _require(atomic_numbers, "input.atomic_numbers must be nonempty")
    for atom_index, number in enumerate(atomic_numbers):
        atomic_number = _as_int(number, f"input.atomic_numbers[{atom_index}]")
        _require(
            1 <= atomic_number <= 118,
            f"input.atomic_numbers[{atom_index}] must be between 1 and 118",
        )
    positions = _as_list(molecule["positions"], "input.positions")
    _require(
        len(positions) == len(atomic_numbers) * 3,
        f"input.positions must hold {len(atomic_numbers) * 3} values",
    )
    for index, value in enumerate(positions):
        _as_float(value, f"input.positions[{index}]")
    _as_float(molecule["molecular_charge"], "input.molecular_charge")
    unpaired = _as_int(molecule["unpaired_electrons"], "input.unpaired_electrons")
    _require(
        unpaired == 0,
        "gpuxtb-scc-trace-v1 is restricted-only; input.unpaired_electrons must be 0",
    )
    spin_channels = _as_int(molecule["spin_channels"], "input.spin_channels")
    _require(
        spin_channels == 1,
        "gpuxtb-scc-trace-v1 is restricted-only; input.spin_channels must be 1",
    )
    temperature = _as_float(molecule["temperature"], "input.temperature")
    _require(temperature >= 0.0, "input.temperature must be nonnegative")

    if "point_charges" in molecule:
        point_charges = _object_fields(
            molecule["point_charges"],
            "input.point_charges",
            required={"positions", "charges", "hardnesses"},
        )
        charges = _as_list(point_charges["charges"], "input.point_charges.charges")
        _require(charges, "input.point_charges.charges must be nonempty")
        point_positions = _as_list(
            point_charges["positions"], "input.point_charges.positions"
        )
        hardnesses = _as_list(
            point_charges["hardnesses"], "input.point_charges.hardnesses"
        )
        _require(
            len(point_positions) == 3 * len(charges),
            "input.point_charges.positions must hold 3 values per point charge",
        )
        _require(
            len(hardnesses) == len(charges),
            "input.point_charges.hardnesses must match the point-charge count",
        )
        for index, value in enumerate(point_positions):
            _as_float(value, f"input.point_charges.positions[{index}]")
        for index, value in enumerate(charges):
            _as_float(value, f"input.point_charges.charges[{index}]")
        for index, value in enumerate(hardnesses):
            hardness = _as_float(value, f"input.point_charges.hardnesses[{index}]")
            _require(
                hardness > 0.0,
                f"input.point_charges.hardnesses[{index}] must be positive",
            )

    basis = _object_fields(
        root["basis"],
        "basis",
        required={"n_atoms", "n_shells", "nao", "atom_to_shell_count"},
        optional={"shell_offsets"},
    )
    n_atoms = _as_int(basis["n_atoms"], "basis.n_atoms")
    _require(n_atoms > 0, "basis.n_atoms must be positive")
    _require(
        n_atoms == len(atomic_numbers),
        "basis.n_atoms must match input.atomic_numbers",
    )
    n_shells = _as_int(basis["n_shells"], "basis.n_shells")
    _require(n_shells > 0, "basis.n_shells must be positive")
    nao = _as_int(basis["nao"], "basis.nao")
    _require(nao > 0, "basis.nao must be positive")
    shell_counts = _as_list(basis["atom_to_shell_count"], "basis.atom_to_shell_count")
    _require(
        len(shell_counts) == n_atoms,
        f"basis.atom_to_shell_count needs {n_atoms} entries, got {len(shell_counts)}",
    )
    validated_shell_counts: list[int] = []
    for atom_index, value in enumerate(shell_counts):
        count = _as_int(value, f"basis.atom_to_shell_count[{atom_index}]")
        _require(
            count > 0,
            f"basis.atom_to_shell_count[{atom_index}] must be positive",
        )
        validated_shell_counts.append(count)
    _require(
        sum(validated_shell_counts) == n_shells,
        "sum(basis.atom_to_shell_count) must equal basis.n_shells",
    )
    if "shell_offsets" in basis:
        offsets = _as_list(basis["shell_offsets"], "basis.shell_offsets")
        _require(
            len(offsets) == n_atoms,
            f"basis.shell_offsets needs {n_atoms} entries, got {len(offsets)}",
        )
        expected_offset = 0
        for atom_index, value in enumerate(offsets):
            offset = _as_int(value, f"basis.shell_offsets[{atom_index}]")
            _require(
                offset == expected_offset,
                f"basis.shell_offsets[{atom_index}] must be {expected_offset}, got {offset}",
            )
            expected_offset += validated_shell_counts[atom_index]

    statics = _object_fields(
        root["statics"],
        "statics",
        required={"overlap", "core_hamiltonian"},
    )
    _matrix("statics.overlap", statics["overlap"], 1, nao, nao)
    _matrix("statics.core_hamiltonian", statics["core_hamiltonian"], 1, nao, nao)

    residual_layout = _object_fields(
        root["residual_layout"],
        "residual_layout",
        required={"shell_charges", "atomic_dipoles", "atomic_quadrupoles"},
    )
    expected_layout = {
        "shell_charges": n_shells,
        "atomic_dipoles": 3 * n_atoms,
        "atomic_quadrupoles": 6 * n_atoms,
    }
    for field, expected in expected_layout.items():
        actual = _as_int(residual_layout[field], f"residual_layout.{field}")
        _require(
            actual == expected,
            f"residual_layout.{field} must be {expected}, got {actual}",
        )
    residual_elements = sum(expected_layout.values())

    iteration_required = {
        "index",
        "hamiltonian",
        "eigenvalues",
        "occupations",
        "density",
        "mixed_qsh",
        "raw_qsh",
        "mixed_qat",
        "raw_qat",
        "mixed_dipoles",
        "raw_dipoles",
        "mixed_quadrupoles",
        "raw_quadrupoles",
        "residual",
        "residual_rms",
        "energy",
        "energy_delta",
        "convergence",
    }
    iterations = _as_list(root["iterations"], "iterations")
    for iteration_offset, iteration in enumerate(iterations):
        path = f"iterations[{iteration_offset}]"
        entry = _object_fields(iteration, path, required=iteration_required)
        expected_index = iteration_offset + 1
        index = _as_int(entry["index"], path + ".index")
        _require(
            index == expected_index,
            f"{path} has out-of-order index {index} (expected {expected_index})",
        )
        _pre_solve_state(entry, path, nao, n_shells, n_atoms)
        _validate_atomic_charges(entry, "mixed", validated_shell_counts, path)
        _matrix(path + ".density", entry["density"], 1, nao, nao)
        _spectrum(path + ".eigenvalues", entry["eigenvalues"], 1, nao)
        # tblite focc is [nao,max(2,nspin)]; restricted traces retain both.
        _spectrum(path + ".occupations", entry["occupations"], 2, nao)
        _spectrum(path + ".raw_qsh", entry["raw_qsh"], 1, n_shells)
        _spectrum(path + ".raw_qat", entry["raw_qat"], 1, n_atoms)
        _multipoles(path + ".raw_dipoles", entry["raw_dipoles"], n_atoms, 3)
        _multipoles(path + ".raw_quadrupoles", entry["raw_quadrupoles"], n_atoms, 6)
        _validate_atomic_charges(entry, "raw", validated_shell_counts, path)

        residual = _as_list(entry["residual"], path + ".residual")
        _require(
            len(residual) == residual_elements,
            f"{path}.residual needs {residual_elements} elements, got {len(residual)}",
        )
        validated_residual = [
            _as_float(value, f"{path}.residual[{index}]")
            for index, value in enumerate(residual)
        ]
        mixed = _flatten_population(entry, "mixed")
        raw = _flatten_population(entry, "raw")
        reconstructed = [
            raw_value - mixed_value for raw_value, mixed_value in zip(raw, mixed)
        ]
        for residual_index, (actual, expected) in enumerate(
            zip(validated_residual, reconstructed)
        ):
            _require(
                actual == expected,
                f"{path}.residual[{residual_index}] must equal raw q/d/Q minus mixed q/d/Q "
                f"(expected {expected!r}, got {actual!r})",
            )
        residual_rms = _as_float(entry["residual_rms"], path + ".residual_rms")
        reconstructed_rms = math.sqrt(
            sum(value * value / residual_elements for value in reconstructed)
        )
        _require(
            math.isfinite(reconstructed_rms),
            f"{path}.residual magnitude overflows finite binary64 RMS",
        )
        _require(
            _within_roundoff(residual_rms, reconstructed_rms),
            f"{path}.residual_rms must be the unweighted numerical RMS of residual",
        )
        _as_float(entry["energy"], path + ".energy")
        _as_float(entry["energy_delta"], path + ".energy_delta")

        convergence = _object_fields(
            entry["convergence"],
            path + ".convergence",
            required={"energy", "population", "temperature", "overall"},
        )
        energy_converged = _as_bool(convergence["energy"], path + ".convergence.energy")
        population_converged = _as_bool(
            convergence["population"], path + ".convergence.population"
        )
        temperature_converged = _as_bool(
            convergence["temperature"], path + ".convergence.temperature"
        )
        overall_converged = _as_bool(
            convergence["overall"], path + ".convergence.overall"
        )
        _require(
            overall_converged
            == (energy_converged and population_converged and temperature_converged),
            f"{path}.convergence.overall must be the conjunction of energy, population, and temperature",
        )
        _require(
            not overall_converged or iteration_offset + 1 == len(iterations),
            f"{path}.convergence.overall cannot be true before the final completed iteration",
        )

    failed_attempt: dict[str, Any] | None = None
    if "failed_attempt" in root:
        failed_attempt = _object_fields(
            root["failed_attempt"],
            "failed_attempt",
            required={
                "index",
                "hamiltonian",
                "mixed_qsh",
                "mixed_qat",
                "mixed_dipoles",
                "mixed_quadrupoles",
            },
        )
        expected_failed_index = len(iterations) + 1
        failed_index = _as_int(failed_attempt["index"], "failed_attempt.index")
        _require(
            failed_index == expected_failed_index,
            f"failed_attempt.index must be {expected_failed_index}, got {failed_index}",
        )
        _pre_solve_state(failed_attempt, "failed_attempt", nao, n_shells, n_atoms)
        _validate_atomic_charges(
            failed_attempt, "mixed", validated_shell_counts, "failed_attempt"
        )

    terminal = _object_fields(
        root["terminal"],
        "terminal",
        required={"status", "converged", "iterations"},
    )
    terminal_status = _as_int(terminal["status"], "terminal.status")
    _require(
        terminal_status in (STATUS_CONVERGED, STATUS_MAX_ITERATIONS, STATUS_FAILED),
        "terminal.status must be 1 (converged), 2 (max iterations), or 3 (failed)",
    )
    terminal_converged = _as_bool(terminal["converged"], "terminal.converged")
    _require(
        terminal_converged == (terminal_status == STATUS_CONVERGED),
        "terminal.converged must be true exactly when terminal.status is 1",
    )
    terminal_iterations = _as_int(terminal["iterations"], "terminal.iterations")
    _require(terminal_iterations >= 0, "terminal.iterations must be nonnegative")
    expected_attempts = len(iterations) + (1 if failed_attempt else 0)
    _require(
        terminal_iterations == expected_attempts,
        f"terminal.iterations must be {expected_attempts}, got {terminal_iterations}",
    )
    _require(
        failed_attempt is None or terminal_status == STATUS_FAILED,
        "failed_attempt is only valid when terminal.status is 3",
    )
    if terminal_status == STATUS_CONVERGED:
        _require(iterations, "a converged terminal state needs a completed iteration")
        _require(
            bool(iterations[-1]["convergence"]["overall"]),
            "the last completed iteration must be converged when terminal.status is 1",
        )
    elif iterations:
        _require(
            not bool(iterations[-1]["convergence"]["overall"]),
            "the last completed iteration cannot be converged unless terminal.status is 1",
        )


def dumps(trace: Any) -> str:
    """Serialize a validated trace to canonical JSON with a trailing newline."""
    validate(trace)
    out: list[str] = []
    _emit(trace, out, 0, "  ")
    return "".join(out) + "\n"


__all__ = [
    "FORMAT",
    "QUADRUPOLE_COMPONENTS",
    "STATUS_CONVERGED",
    "STATUS_FAILED",
    "STATUS_MAX_ITERATIONS",
    "TraceError",
    "dumps",
    "validate",
]
