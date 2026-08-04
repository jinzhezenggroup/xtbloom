"""Canonical ``gpuxtb-scc-trace-v1`` writer and schema validation.

This module implements the versioned, deterministic interchange format for
tblite GFN2 SCC iteration goldens defined by issue #47.  It has no third-party
dependencies (standard library only) so it can run before either Fortran
reference package is built.

Format contract (see :file:`gpuxtb-scc-trace-v1.schema.json` for the machine
readable schema):

- ``format`` must be ``gpuxtb-scc-trace-v1``; every other schema version is
  rejected with an actionable error.
- Matrices are stored in logical ``[spin][row][column]`` order and multipoles
  in ``[spin][atom][component]`` order, never raw Fortran memory layout.
- Quadrupoles use the packing order ``xx, xy, yy, xz, yz, zz``.
- The mixer residual vector is shell charges followed by atomic dipoles and
  atomic quadrupoles in the exact tblite flattening order.
- All floats are finite and serialized with at least 17 significant decimal
  digits so every binary64 value round-trips exactly.
- ``dumps()`` is canonical: object keys are emitted in sorted order, so
  reading and re-writing a trace yields byte-identical JSON for the same
  logical state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from typing import Any

FORMAT = "gpuxtb-scc-trace-v1"

# Quadrupole component packing order (xxxx -> 0 ... zzzz -> 5 in tblite qpat).
QUADRUPOLE_COMPONENTS: tuple[str, ...] = ("xx", "xy", "yy", "xz", "yz", "zz")

# tblite status values surfaced by the observer callbacks.
STATUS_CONVERGED = 1
STATUS_MAX_ITERATIONS = 2
STATUS_FAILED = 3


class TraceError(ValueError):
    """Raised when a trace is malformed or cannot be serialized canonically."""


# --- canonical JSON emission --------------------------------------------------------


def _float_text(value: float) -> str:
    """Serialize a finite float with at least 17 significant digits."""
    if not math.isfinite(value):
        raise TraceError(f"trace floats must be finite (got {value!r})")
    text = format(value, ".17g")
    if not any(character in text for character in ".eEnN"):
        # An integral float would otherwise be re-parsed as a JSON integer;
        # force a decimal point so the element round-trips as a float.
        text += ".0"
    return text


def _iter_items(value: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield key/value pairs, rejecting non-str keys so canonicalization works."""
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


# --- schema validation --------------------------------------------------------------


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


def _matrix(path: str, value: Any, spin: int, rows: int, columns: int) -> None:
    channels = _as_list(value, path)
    _require(
        len(channels) == spin,
        f"{path} needs {spin} spin channels, got {len(channels)}",
    )
    for spin_index, channel in enumerate(channels):
        channel_path = f"{path}[{spin_index}]"
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
            for element in row_values:
                _as_float(element, row_path)


def _spectrum(path: str, value: Any, spin: int, length: int) -> None:
    channels = _as_list(value, path)
    _require(
        len(channels) == spin,
        f"{path} needs {spin} spin channels, got {len(channels)}",
    )
    for spin_index, channel in enumerate(channels):
        channel_path = f"{path}[{spin_index}]"
        values = _as_list(channel, channel_path)
        _require(
            len(values) == length,
            f"{channel_path} needs {length} values, got {len(values)}",
        )
        for element in values:
            _as_float(element, channel_path)


def validate(trace: Any) -> None:
    """Validate a candidate trace against the ``gpuxtb-scc-trace-v1`` schema.

    Raises :class:`TraceError` with an actionable message on any structural,
    dimensional, or versioning violation.
    """
    root = _as_dict(trace, "trace")
    _require(
        root.get("format") == FORMAT,
        "unsupported trace format {!r} (expected {!r})".format(
            root.get("format"), FORMAT
        ),
    )

    provenance = _as_dict(root.get("provenance"), "provenance")
    revision = provenance.get("tblite_revision")
    _require(
        isinstance(revision, str) and len(revision) == 40,
        "provenance.tblite_revision must be a pinned 40-hex commit",
    )
    digest = provenance.get("oracle_patch_sha256")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        "provenance.oracle_patch_sha256 must be a 64-hex digest",
    )

    molecule = _as_dict(root.get("input"), "input")
    atomic_numbers = _as_list(molecule.get("atomic_numbers"), "input.atomic_numbers")
    _require(len(atomic_numbers) > 0, "input.atomic_numbers must be nonempty")
    for number in atomic_numbers:
        _as_int(number, "input.atomic_numbers")
    positions = _as_list(molecule.get("positions"), "input.positions")
    _require(
        len(positions) == len(atomic_numbers) * 3,
        f"input.positions must hold {len(atomic_numbers) * 3} doubles",
    )
    for value in positions:
        _as_float(value, "input.positions")
    _as_float(molecule.get("molecular_charge"), "input.molecular_charge")
    _as_int(molecule.get("unpaired_electrons"), "input.unpaired_electrons")
    spin = _as_int(molecule.get("spin_channels"), "input.spin_channels")
    _require(spin in (1, 2), "input.spin_channels must be 1 or 2")
    if "temperature" in molecule:
        _as_float(molecule.get("temperature"), "input.temperature")

    basis = _as_dict(root.get("basis"), "basis")
    nao = _as_int(basis.get("nao"), "basis.nao")
    _require(nao > 0, "basis.nao must be positive")
    n_shells = _as_int(basis.get("n_shells"), "basis.n_shells")
    _require(n_shells > 0, "basis.n_shells must be positive")
    n_atoms = _as_int(basis.get("n_atoms"), "basis.n_atoms")
    _require(
        n_atoms == len(atomic_numbers),
        "basis.n_atoms must match input.atomic_numbers",
    )

    statics = _as_dict(root.get("statics"), "statics")
    _matrix("statics.overlap", statics.get("overlap"), 1, nao, nao)
    _matrix("statics.core_hamiltonian", statics.get("core_hamiltonian"), 1, nao, nao)

    residual_elements = 0
    for key, length in _iter_items(
        _as_dict(root.get("residual_layout"), "residual_layout")
    ):
        residual_elements += _as_int(length, f"residual_layout.{key}")

    iterations = _as_list(root.get("iterations"), "iterations")
    _require(len(iterations) > 0, "iterations must not be empty")
    previous_index = 0
    for iteration_index, iteration in enumerate(iterations):
        path = f"iterations[{iteration_index}]"
        entry = _as_dict(iteration, path)
        index = _as_int(entry.get("index"), path + ".index")
        _require(
            index == previous_index + 1,
            f"{path} has out-of-order index {index} (expected {previous_index + 1})",
        )
        previous_index = index
        _matrix(path + ".hamiltonian", entry.get("hamiltonian"), spin, nao, nao)
        _matrix(path + ".density", entry.get("density"), spin, nao, nao)
        _spectrum(path + ".eigenvalues", entry.get("eigenvalues"), spin, nao)
        _spectrum(path + ".occupations", entry.get("occupations"), spin, nao)
        for field in ("mixed_qsh", "raw_qsh"):
            _spectrum(path + "." + field, entry.get(field), spin, n_shells)
        for field in ("mixed_qat", "raw_qat"):
            _spectrum(path + "." + field, entry.get(field), spin, n_atoms)
        residual = _as_list(entry.get("residual"), path + ".residual")
        _require(
            len(residual) == residual_elements,
            f"{path} residual needs {residual_elements} elements, got {len(residual)}",
        )
        for value in residual:
            _as_float(value, path + ".residual")
        _as_float(entry.get("residual_rms"), path + ".residual_rms")
        _as_float(entry.get("energy"), path + ".energy")
        _as_bool(entry.get("converged"), path + ".converged")
        _as_int(entry.get("status"), path + ".status")

    terminal = _as_dict(root.get("terminal"), "terminal")
    _as_int(terminal.get("iterations"), "terminal.iterations")
    terminal_status = _as_int(terminal.get("status"), "terminal.status")
    _as_bool(terminal.get("converged"), "terminal.converged")
    _require(
        terminal_status == iterations[-1].get("status"),
        "terminal.status must match the last iteration status",
    )
    _require(
        bool(terminal.get("converged")) == bool(iterations[-1].get("converged")),
        "terminal.converged must match the last iteration",
    )


def dumps(trace: Any) -> str:
    """Serialize a validated trace to canonical ``gpuxtb-scc-trace-v1`` JSON.

    The output is deterministic: object keys are sorted recursively, floats
    keep at least 17 significant digits, and the result ends with a newline.
    """
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
