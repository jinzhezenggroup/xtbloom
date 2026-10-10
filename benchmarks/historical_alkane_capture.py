#!/usr/bin/env python3
"""Capture bounded native replays of the saved 32/62-atom alkane batches.

This diagnostic-only runner preserves every public result and failure. It
never qualifies the empty historical golden placeholders or emits an eligible
performance claim. The normal entry point is ``python -m
benchmarks.historical_alkane_capture``; ``--describe`` validates pinned input
metadata without loading a native library.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import cuda_diagnostics, historical_alkane_inputs, run

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


PIN_SCHEMA = "xtbloom.historical-alkane-library-pins.v1"
CALL_SCHEMA = "xtbloom.historical-alkane-capture.call.v1"
MANIFEST_SCHEMA = "xtbloom-historical-alkane-assembler-inputs-v1"
DIAGNOSTIC_SINK_ENV = "XTBLOOM_CUDA_SCC_DIAGNOSTICS_FILE"
HISTORICAL_BRIDGE_RAW_SHA256 = (
    "a864292f533bd3316e127a3db8be9f8f3ba09073cabd2107b026c056c4ca5ba8"
)
HISTORICAL_REQUEST = {
    "source_artifact": "n1 bridge-raw",
    "source_sha256": HISTORICAL_BRIDGE_RAW_SHA256,
    "batch_size": 256,
    "backend": "cuda",
    "loose": False,
    "repetitions": 0,
    "property_set": "EF",
    "compute_flags": 3,
    "max_scc_iterations": 500,
    "charge_tolerance": 1.0e-6,
    "energy_tolerance": 1.0e-8,
    "electronic_temperature_hartree": 0.000950042573563535,
    "scc_start_mode": 1,
}
CAPTURE_WORKLOADS = historical_alkane_inputs.EXPECTED_WORKLOADS
CAPTURE_PROPERTIES = (("EF", 3), ("EFq", 7))
CAPTURE_VARIANTS = (
    ("baseline", "baseline"),
    ("off", "off"),
    ("on-no-sink", "on"),
    ("on-sink", "on"),
)
COLD_CALLS = 1
EXCLUDED_WARMUPS = 5
MEASURED_CALLS = 30
SCC_MIXER_MODIFIED_BROYDEN = 1
SCC_START_FRESH = 1
DETERMINISM_DEFAULT = 0
RESULT_FLAGS_SENTINEL = 0xA5A55A5A
ENERGY_SENTINEL = -9.87654321e290
FORCE_SENTINEL = -8.76543210e290
CHARGE_SENTINEL = -7.65432109e290
ITERATIONS_SENTINEL = -0x5A5A
STATUS_SENTINEL = -0x4B4B
CONVERGED_SENTINEL = 0xA5
STATUS_INTERNAL_ERROR = 6
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class CaptureError(ValueError):
    """Invalid capture inputs, pins, or output-directory state."""


class NativeAdapter(run.XTBloomAdapter):
    """Persistent CUDA/host owner that keeps raw public buffers on failures."""

    def __init__(
        self,
        library_path: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        case_sequence: Sequence[dict[str, Any]],
        workload: str,
        property_set: str,
    ) -> None:
        if property_set not in {name for name, _ in CAPTURE_PROPERTIES}:
            raise CaptureError(f"unsupported property set {property_set!r}")
        self.property_set = property_set
        self.property_flags = dict(CAPTURE_PROPERTIES)[property_set]
        cell = run.Cell("xtbloom", "cuda", "host", workload, "force", 256)
        try:
            super().__init__(
                library_path,
                manifest_path,
                manifest,
                case_sequence,
                cell,
                device_id=0,
                cpu_threads=1,
            )
            self.options.model = run.public_api.XTBLOOM_MODEL_GFN2_XTB
            self.options.flags = self.property_flags
            self.options.max_scc_iterations = 500
            self.options.charge_tolerance = 1.0e-6
            self.options.energy_tolerance = 1.0e-8
            self.options.electronic_temperature = (
                300.0 * run.public_api.XTBLOOM_KELVIN_TO_HARTREE
            )
            self.options.scc_start_mode = SCC_START_FRESH
            self.options.scc_mixer = SCC_MIXER_MODIFIED_BROYDEN
            self.options.scc_mixer_history = 8
            self.options.scc_mixer_damping = 0.4
            self.options.determinism = DETERMINISM_DEFAULT
            self.charges = (
                (run.public_api.ctypes.c_double * self.atoms)()
                if property_set == "EFq"
                else None
            )
            if self.charges is not None:
                self.result.atomic_charges = self.memory.output(
                    self.charges, "atomic_charges"
                )
            self.last_call_status: int | None = None
            self.last_call_error: str | None = None
        except BaseException as error:
            cleanup_error = self._cleanup_partial_construction()
            if cleanup_error is not None:
                error.add_note(f"partial adapter cleanup also failed: {cleanup_error}")
            raise

    def _cleanup_partial_construction(self) -> BaseException | None:
        """Release resources if base initialization failed after context creation."""
        cleanup_errors: list[BaseException] = []
        memory = getattr(self, "memory", None)
        if memory is not None:
            try:
                memory.close()
            except BaseException as error:  # noqa: BLE001 - continue partial cleanup
                cleanup_errors.append(error)
        if getattr(self, "owns_cuda_control", False):
            cuda_control = getattr(self, "cuda_control", None)
            if cuda_control is not None:
                try:
                    cuda_control.close()
                except BaseException as error:  # noqa: BLE001 - continue partial cleanup
                    cleanup_errors.append(error)
        library = getattr(self, "library", None)
        context = getattr(self, "context", None)
        if library is not None and context is not None:
            try:
                library.xtbloom_context_destroy(context)
            except BaseException as error:  # noqa: BLE001 - continue partial cleanup
                cleanup_errors.append(error)
        return cleanup_errors[0] if cleanup_errors else None

    def invoke(self) -> None:
        """Call the ABI without raising away status or caller-owned raw buffers."""
        status = self.library.xtbloom_compute(
            self.context,
            run.public_api.ctypes.byref(self.batch),
            run.public_api.ctypes.byref(self.options),
            run.public_api.ctypes.byref(self.result),
        )
        self.last_call_status = int(status)
        self.last_call_error = (
            _decode_c_string(self.library.xtbloom_get_last_error())
            if self.last_call_status != run.public_api.XTBLOOM_STATUS_SUCCESS
            else None
        )

    def reset_outputs(self) -> None:
        """Poison caller-owned output sentinels before each timed ABI call."""
        self.result.flags = RESULT_FLAGS_SENTINEL
        for index in range(len(self.energies)):
            self.energies[index] = ENERGY_SENTINEL
        if self.forces is not None:
            for index in range(len(self.forces)):
                self.forces[index] = FORCE_SENTINEL
        if self.charges is not None:
            for index in range(len(self.charges)):
                self.charges[index] = CHARGE_SENTINEL
        for index in range(len(self.iterations)):
            self.iterations[index] = ITERATIONS_SENTINEL
            self.statuses[index] = STATUS_SENTINEL
            self.converged[index] = CONVERGED_SENTINEL


def _decode_c_string(value: bytes | None) -> str | None:
    """Decode a native status string while preserving null returns explicitly."""
    return None if value is None else value.decode("utf-8", errors="replace")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous pin JSON instead of silently taking the last key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load one strict UTF-8 JSON object and include its path on errors."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CaptureError) as error:
        raise CaptureError(f"cannot load {label} {path}: {error}") from error
    if type(value) is not dict:
        raise CaptureError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    """Hash a file incrementally so large input coordinate assets stay bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bridge_raw_sha256(value: str) -> None:
    """Require the independently retrieved bridge-raw provenance pin."""
    if not SHA256_RE.fullmatch(value) or value != HISTORICAL_BRIDGE_RAW_SHA256:
        raise CaptureError(
            "--historical-bridge-raw-sha256 must equal the pinned n1 "
            "bridge-raw SHA-256 "
            f"{HISTORICAL_BRIDGE_RAW_SHA256}"
        )


def _load_manifest_bundle(
    manifest_path: Path, manifest_sha256: str, bridge_raw_sha256: str
) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], ...]], str]:
    """Pin manifest bytes and every input/golden before assembling saved cases."""
    _validate_bridge_raw_sha256(bridge_raw_sha256)
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise CaptureError("--manifest-sha256 must be 64 lowercase hex characters")
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != manifest_sha256:
        raise CaptureError(
            f"manifest SHA-256 mismatch: expected {manifest_sha256}, "
            f"got {actual_manifest_sha256}"
        )
    manifest = run.conformance.load_json(manifest_path)
    source = manifest.get("source")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("diagnostic_only") is not True
        or not isinstance(source, dict)
        or not SHA256_RE.fullmatch(str(source.get("sha256", "")))
        or not COMMIT_RE.fullmatch(str(source.get("source_commit", "")))
        or not isinstance(source.get("source_hashes"), dict)
    ):
        raise CaptureError("manifest is not a hash-pinned historical input corpus")
    if manifest.get("eligibility") != {
        "correctness": False,
        "performance": False,
        "reason": "empty golden placeholders and no independent reference",
    }:
        raise CaptureError("historical manifest eligibility must remain false")

    workload_records = manifest.get("workloads")
    if (
        not isinstance(workload_records, list)
        or len(workload_records) != len(CAPTURE_WORKLOADS)
        or any(not isinstance(record, dict) for record in workload_records)
        or [
            (record.get("name"), record.get("natoms"), record.get("batch_size"))
            for record in workload_records
        ]
        != [(name, natoms, 256) for name, natoms in CAPTURE_WORKLOADS]
    ):
        raise CaptureError("manifest must contain the two pinned 256-system workloads")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 512:
        raise CaptureError("manifest must contain exactly 512 original case slots")

    selected: dict[str, tuple[dict[str, Any], ...]] = {}
    for workload, _ in CAPTURE_WORKLOADS:
        sequence = historical_alkane_inputs.select_workload_cases(manifest, workload)
        if len(sequence) != 256:
            raise CaptureError(f"workload {workload} must contain 256 cases")
        selected[workload] = sequence

    for case in cases:
        for path_key, digest_key in (
            ("input", "input_sha256"),
            ("golden", "golden_sha256"),
        ):
            expected = case.get(digest_key)
            if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
                raise CaptureError(
                    f"case {case.get('id')!r} has an invalid {digest_key}"
                )
            relative_path = case.get(path_key)
            if not isinstance(relative_path, str):
                raise CaptureError(f"case {case.get('id')!r} lacks {path_key} path")
            path = run.conformance.resolve_manifest_path(manifest_path, relative_path)
            if not path.is_file():
                raise CaptureError(f"case {case.get('id')!r} {path_key} is missing")
            actual = _sha256_file(path)
            if actual != expected:
                raise CaptureError(
                    f"case {case.get('id')!r} {path_key} SHA-256 mismatch"
                )
    return manifest, selected, actual_manifest_sha256


def _validate_library_pins(path: Path) -> dict[str, dict[str, str]]:
    """Read the three exact build identities without loading any shared object."""
    document = _load_json_object(path.expanduser().resolve(strict=True), "library pins")
    if set(document) != {"schema", "builds"} or document.get("schema") != PIN_SCHEMA:
        raise CaptureError(f"library pins must use schema {PIN_SCHEMA!r}")
    builds = document.get("builds")
    if not isinstance(builds, list) or len(builds) != 3:
        raise CaptureError("library pins must define baseline, off, and on builds")
    pins: dict[str, dict[str, str]] = {}
    expected_keys = {
        "role",
        "dso_path",
        "dso_sha256",
        "cache_path",
        "cache_sha256",
        "source_dir",
        "source_commit",
        "source_sha256",
    }
    for build in builds:
        if not isinstance(build, dict) or set(build) != expected_keys:
            raise CaptureError("each build pin has an unsupported field set")
        role = build["role"]
        if not isinstance(role, str) or role not in {"baseline", "off", "on"}:
            raise CaptureError("build roles must be unique baseline, off, and on")
        if role in pins:
            raise CaptureError("build roles must be unique baseline, off, and on")
        for field in ("dso_path", "cache_path", "source_dir"):
            value = build[field]
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise CaptureError(f"{role}.{field} must be an absolute path")
        for field in ("dso_sha256", "cache_sha256", "source_sha256"):
            value = build[field]
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise CaptureError(f"{role}.{field} must be a lowercase SHA-256")
        if not isinstance(build["source_commit"], str) or not COMMIT_RE.fullmatch(
            build["source_commit"]
        ):
            raise CaptureError(f"{role}.source_commit must be a full Git commit")
        pins[role] = {key: str(value) for key, value in build.items()}
    if set(pins) != {"baseline", "off", "on"}:
        raise CaptureError("build roles must include baseline, off, and on")
    return pins


def _git_output(source_dir: Path, *arguments: str) -> str:
    """Read one Git identity value without exposing the process environment."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CaptureError(f"Git source check failed for {source_dir}") from error
    return result.stdout.strip()


def _git_archive_sha256(source_dir: Path) -> str:
    """Hash the exact deterministic tar stream produced by ``git archive HEAD``."""
    try:
        process = subprocess.Popen(
            ["git", "-C", str(source_dir), "archive", "--format=tar", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise CaptureError(f"cannot archive pinned source tree {source_dir}") from error
    assert process.stdout is not None
    digest = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise CaptureError(
            f"cannot hash pinned source tree {source_dir}: {detail or return_code}"
        )
    return digest.hexdigest()


def _verify_one_build(role: str, pin: dict[str, str]) -> dict[str, Any]:
    """Verify source cleanliness, current commit, archive, DSO, and cache pins."""
    dso = Path(pin["dso_path"])
    cache = Path(pin["cache_path"])
    source_dir = Path(pin["source_dir"])
    if not dso.is_file() or not cache.is_file() or not source_dir.is_dir():
        raise CaptureError(f"{role} pin path is missing or has the wrong type")
    actual_commit = _git_output(source_dir, "rev-parse", "HEAD")
    dirty_state = _git_output(
        source_dir, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if dirty_state:
        raise CaptureError(f"{role} source checkout is dirty")
    if actual_commit != pin["source_commit"]:
        raise CaptureError(f"{role} source commit does not match its pin")
    source_sha256 = _git_archive_sha256(source_dir)
    if source_sha256 != pin["source_sha256"]:
        raise CaptureError(f"{role} source archive SHA-256 does not match its pin")
    dso_sha256 = _sha256_file(dso)
    cache_sha256 = _sha256_file(cache)
    if dso_sha256 != pin["dso_sha256"]:
        raise CaptureError(f"{role} DSO SHA-256 does not match its pin")
    if cache_sha256 != pin["cache_sha256"]:
        raise CaptureError(f"{role} cache SHA-256 does not match its pin")
    return {
        "source_commit": actual_commit,
        "source_clean": True,
        "source_sha256": source_sha256,
        "dso_sha256": dso_sha256,
        "cache_sha256": cache_sha256,
    }


def _verify_all_builds(pins: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Verify every requested build before loading and after cleanup."""
    return {
        role: _verify_one_build(role, pins[role]) for role in ("baseline", "off", "on")
    }


def _case_metadata(case: dict[str, Any]) -> dict[str, Any]:
    """Retain original slot, charge/UHF, spin tags, and hash identities."""
    return {
        "id": case["id"],
        "source_slot": case["source_slot"],
        "atom_count": case["atom_count"],
        "molecular_charge": case["molecular_charge"],
        "unpaired_electrons": case["unpaired_electrons"],
        "spin_channels": case["spin_channels"],
        "input_sha256": case["input_sha256"],
        "golden_sha256": case["golden_sha256"],
    }


def _input_archive(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: dict[str, tuple[dict[str, Any], ...]],
    workloads: Sequence[tuple[str, int]] = CAPTURE_WORKLOADS,
) -> dict[str, Any]:
    """Preserve original case order and complete assembled numerical arrays."""
    archived: list[dict[str, Any]] = []
    for workload, _ in workloads:
        cases = selected[workload]
        storage = run.public_api.assemble_batch(manifest_path, manifest, cases)
        archived.append(
            {
                "name": workload,
                "case_ids": [case["id"] for case in cases],
                "source_slots": [case["source_slot"] for case in cases],
                "cases": [_case_metadata(case) for case in cases],
                "atom_offsets": storage.atom_offsets,
                "atomic_numbers": storage.atomic_numbers,
                "positions_bohr": storage.positions,
                "molecular_charges": [case["molecular_charge"] for case in cases],
                "unpaired_electrons": [case["unpaired_electrons"] for case in cases],
                "spin_channels": [case["spin_channels"] for case in cases],
            }
        )
    return {
        "schema": "xtbloom.historical-alkane-capture.inputs.v1",
        "goldens_used_as_oracles": False,
        "workloads": archived,
    }


def _json_number(value: float) -> float | dict[str, str]:
    """Represent all non-finite native values as strict-JSON tagged objects."""
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return {"nonfinite": "nan"}
    return {"nonfinite": "+inf" if value > 0 else "-inf"}


def _json_float_array(values: object) -> list[float | dict[str, str]]:
    """Convert one caller buffer to strict-JSON values without dropping NaNs."""
    return [_json_number(float(value)) for value in values]  # type: ignore[union-attr]


def _quiet_nan_at(owner: object, index: int) -> bool:
    """Inspect the stored binary64 quiet bit without normalizing NaN payloads."""
    address = run.public_api.ctypes.addressof(
        owner
    ) + index * run.public_api.ctypes.sizeof(run.public_api.ctypes.c_double)
    payload = run.public_api.ctypes.string_at(address, 8)
    bits = int.from_bytes(payload, byteorder=sys.byteorder, signed=False)
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    return exponent == 0x7FF and fraction != 0 and bool(fraction & (1 << 51))


def _status_name(adapter: Any, status: int) -> str | None:  # noqa: ANN401
    """Return the native symbolic status when a loaded adapter exposes it."""
    library = getattr(adapter, "library", None)
    status_string = getattr(library, "xtbloom_status_string", None)
    if status_string is None:
        return None
    return _decode_c_string(status_string(status))


def _sentinels_unchanged(adapter: Any) -> bool:  # noqa: ANN401
    """Prove whether a rejected call left all caller outputs and flags untouched."""
    try:
        return (
            adapter.result.flags == RESULT_FLAGS_SENTINEL
            and all(value == ENERGY_SENTINEL for value in adapter.energies)
            and (
                adapter.forces is None
                or all(value == FORCE_SENTINEL for value in adapter.forces)
            )
            and (
                adapter.charges is None
                or all(value == CHARGE_SENTINEL for value in adapter.charges)
            )
            and all(
                adapter.iterations[index] == ITERATIONS_SENTINEL
                and adapter.statuses[index] == STATUS_SENTINEL
                and adapter.converged[index] == CONVERGED_SENTINEL
                for index in range(adapter.systems)
            )
        )
    except (IndexError, TypeError):
        return False


def _peer_results(adapter: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Keep the typed per-system status, convergence, and SCC iteration data."""
    peers: list[dict[str, Any]] = []
    for index in range(adapter.systems):
        case = adapter.storage.slices[index].case
        try:
            status = int(adapter.statuses[index])
        except (IndexError, TypeError, ValueError):
            status = None
        try:
            converged = int(adapter.converged[index])
        except (IndexError, TypeError, ValueError):
            converged = None
        try:
            iterations = int(adapter.iterations[index])
        except (IndexError, TypeError, ValueError):
            iterations = None
        peers.append(
            {
                "system_index": index,
                "case_id": case["id"],
                "source_slot": case["source_slot"],
                "status": status,
                "status_name": _status_name(adapter, status)
                if status is not None
                else None,
                "converged": converged,
                "iterations": iterations,
            }
        )
    return peers


def _typed_array_has_shape(values: object, element_type: type, size: int) -> bool:
    """Require exact ctypes element types as well as the full public extent."""
    ctypes = run.public_api.ctypes
    return (
        isinstance(values, ctypes.Array)
        and type(values)._type_ is element_type
        and len(values) == size
    )


def _validate_published_outputs(
    adapter: Any,  # noqa: ANN401
    call_status: int,
) -> list[str]:
    """Check public shapes, peer-local NaNs, and finite successful peer slices."""
    if call_status != run.public_api.XTBLOOM_STATUS_SUCCESS:
        return []
    issues: list[str] = []
    systems = adapter.systems
    offsets = adapter.storage.atom_offsets
    if len(offsets) != systems + 1 or offsets[0] != 0:
        return ["assembled atom offsets do not cover the complete batch"]
    if any(right < left for left, right in pairwise(offsets)):
        return ["assembled atom offsets are not monotonic"]
    expected_atoms = offsets[-1]
    ctypes = run.public_api.ctypes
    typed_buffers = (
        ("energy", adapter.energies, ctypes.c_double, systems),
        ("force", adapter.forces, ctypes.c_double, 3 * expected_atoms),
        ("iterations", adapter.iterations, ctypes.c_int32, systems),
        ("status", adapter.statuses, ctypes.c_int32, systems),
        ("convergence", adapter.converged, ctypes.c_uint8, systems),
    )
    for label, values, element_type, size in typed_buffers:
        if not _typed_array_has_shape(values, element_type, size):
            issues.append(f"{label} output has an invalid ctypes type or extent")
    if adapter.property_set == "EFq" and not _typed_array_has_shape(
        adapter.charges, ctypes.c_double, expected_atoms
    ):
        issues.append("charge output has an invalid ctypes type or extent")
    elif adapter.property_set == "EF" and adapter.charges is not None:
        issues.append("unrequested atomic-charge output was allocated")
    elif adapter.property_set not in {"EF", "EFq"}:
        issues.append("adapter has an unsupported property set")
    if issues:
        return issues
    if adapter.result.flags == RESULT_FLAGS_SENTINEL:
        issues.append("successful call left result flags untouched")
    if any(value == ENERGY_SENTINEL for value in adapter.energies):
        issues.append("successful call left an energy output sentinel untouched")
    if any(value == FORCE_SENTINEL for value in adapter.forces):
        issues.append("successful call left a force output sentinel untouched")
    if adapter.charges is not None and any(
        value == CHARGE_SENTINEL for value in adapter.charges
    ):
        issues.append("successful call left a charge output sentinel untouched")

    for index in range(systems):
        status = int(adapter.statuses[index])
        converged = int(adapter.converged[index])
        iterations = int(adapter.iterations[index])
        if iterations == ITERATIONS_SENTINEL:
            issues.append(f"system {index} retained a per-system iteration sentinel")
        if status == STATUS_SENTINEL or converged == CONVERGED_SENTINEL:
            issues.append(f"system {index} retained a per-system status sentinel")
            continue
        if converged not in (0, 1):
            issues.append(f"system {index} has invalid convergence tag {converged}")
            continue
        failed = status != run.public_api.XTBLOOM_STATUS_SUCCESS or converged != 1
        atom_begin, atom_end = offsets[index], offsets[index + 1]
        floating_slices: list[tuple[str, object, int, int]] = [
            ("energy", adapter.energies, index, index + 1)
        ]
        if adapter.forces is not None:
            floating_slices.append(
                ("force", adapter.forces, 3 * atom_begin, 3 * atom_end)
            )
        if adapter.charges is not None:
            floating_slices.append(("charge", adapter.charges, atom_begin, atom_end))
        for label, values, begin, end in floating_slices:
            current = [float(values[offset]) for offset in range(begin, end)]
            if failed:
                if not all(
                    _quiet_nan_at(values, offset) for offset in range(begin, end)
                ):
                    issues.append(
                        f"failed system {index} {label} slice is not all quiet NaNs"
                    )
            elif not all(math.isfinite(value) for value in current):
                issues.append(f"successful system {index} {label} slice is non-finite")
        if not failed and iterations <= 0:
            issues.append(
                f"successful system {index} has no positive SCC iteration count"
            )
    return issues


def _trace_delta(
    path: Path,
    offset: int,
    expected_record: bool,
    systems: int,
    peers: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], int, list[str]]:
    """Parse one newly appended diagnostic segment through the shared schema."""
    issues: list[str] = []
    original_offset = offset
    try:
        raw = path.read_bytes()
    except OSError as error:
        return (
            {"path": str(path), "offset_before": offset},
            offset,
            [f"cannot read diagnostic sink: {error}"],
        )
    if len(raw) < offset:
        issues.append("diagnostic sink was truncated between calls")
        offset = len(raw)
    delta = raw[offset:]
    new_offset = len(raw)
    records: list[dict[str, Any]] = []
    try:
        text = delta.decode("utf-8", errors="strict")
        records = list(cuda_diagnostics.iter_jsonl(io.StringIO(text))) if delta else []
    except (UnicodeError, cuda_diagnostics.DiagnosticError) as error:
        issues.append(f"diagnostic JSONL is malformed: {error}")

    expected_count = 1 if expected_record else 0
    if len(records) != expected_count:
        issues.append(
            f"diagnostic append count is {len(records)}, expected {expected_count}"
        )
    report_summary: dict[str, Any] | None = None
    if len(records) == 1:
        record = records[0]
        try:
            report = cuda_diagnostics.build_report(records)
            report_summary = {
                "schema": report["schema"],
                "call_count": report["call_count"],
                "instrumentation_perturbs_timing": report["timing_policy"][
                    "instrumentation_perturbs_timing"
                ],
                "eligible_for_primary_performance_tables": report["timing_policy"][
                    "eligible_for_primary_performance_tables"
                ],
            }
        except Exception as error:  # noqa: BLE001 - keep raw invalid trace for diagnosis
            issues.append(
                "diagnostic report construction failed: "
                f"{type(error).__name__}: {error}"
            )
        if not expected_record:
            issues.append("diagnostic record appeared after a failed native call")
        if record["batch_size"] != systems:
            issues.append("diagnostic batch size does not match the captured owner")
        if record["maximum_iterations"] != 500:
            issues.append("diagnostic maximum_iterations differs from the request")
        if record.get("start_policy") != SCC_START_FRESH:
            issues.append("diagnostic start_policy does not record FRESH")
        terminal = record.get("terminal_systems")
        if not isinstance(terminal, list) or len(terminal) != systems or peers is None:
            issues.append("diagnostic record lacks one terminal row per captured peer")
        elif any(
            row.get("system_index") != peer["system_index"]
            or row.get("status") != peer["status"]
            or row.get("converged") != peer["converged"]
            or row.get("iterations") != peer["iterations"]
            for row, peer in zip(terminal, peers, strict=True)
        ):
            issues.append("diagnostic terminal rows differ from public peer outputs")

    return (
        {
            "path": str(path),
            "offset_before": original_offset,
            "offset_after": new_offset,
            "new_bytes": len(delta),
            "parsed_records": len(records),
            "report": report_summary,
        },
        new_offset,
        issues,
    )


def _sink_environment(path: Path | None) -> dict[str, str | None]:
    """Set one owner sink and remember the already-validated prior state."""
    previous = os.environ.get(DIAGNOSTIC_SINK_ENV)
    if path is None:
        os.environ.pop(DIAGNOSTIC_SINK_ENV, None)
    else:
        os.environ[DIAGNOSTIC_SINK_ENV] = str(path.resolve())
    return {DIAGNOSTIC_SINK_ENV: previous}


def _restore_sink_environment(previous: dict[str, str | None]) -> None:
    """Restore the already-validated absent inherited sink state."""
    value = previous[DIAGNOSTIC_SINK_ENV]
    if value is None:
        os.environ.pop(DIAGNOSTIC_SINK_ENV, None)
    else:
        os.environ[DIAGNOSTIC_SINK_ENV] = value


def _call_record_base(
    ordinal: int,
    workload: str,
    property_set: str,
    flags: int,
    variant: str,
    role: str,
    phase: str,
    round_index: int,
    variant_order: int,
    setup_seconds: float | None,
    case_ids: Sequence[str],
) -> dict[str, Any]:
    """Create stable identity fields shared by executed and NOT_RUN rows."""
    return {
        "schema": CALL_SCHEMA,
        "ordinal": ordinal,
        "workload": workload,
        "property_set": property_set,
        "compute_flags": flags,
        "variant": variant,
        "build_role": role,
        "phase": phase,
        "round": round_index,
        "variant_order": variant_order,
        "setup_seconds": setup_seconds,
        "case_ids": list(case_ids),
    }


def _write_json_line(handle: Any, value: dict[str, Any]) -> None:  # noqa: ANN401
    """Append one strict JSON record and flush it before the next native call."""
    handle.write(json.dumps(value, allow_nan=False, separators=(",", ":")))
    handle.write("\n")
    handle.flush()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write or replace one JSON artifact owned by the fresh output directory."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _protocol_metadata(
    manifest: dict[str, Any],
    manifest_sha256: str,
    pins: dict[str, dict[str, str]],
    workloads: Sequence[tuple[str, int]],
    properties: Sequence[tuple[str, int]],
    variants: Sequence[tuple[str, str]],
    warmups: int,
    measured: int,
) -> dict[str, Any]:
    """Describe timing and option provenance without claiming historical replay."""
    source = manifest["source"]
    return {
        "schema": "xtbloom.historical-alkane-capture.metadata.v1",
        "claim_eligible": False,
        "performance_claim_eligible": False,
        "correctness_oracle_eligible": False,
        "manifest_sha256": manifest_sha256,
        "historical_input_source_full_pin": {
            "source_commit": source["source_commit"],
            "source_sha256": source["sha256"],
            "source_hashes": source["source_hashes"],
        },
        "historical_bridge_raw_request": HISTORICAL_REQUEST,
        "historical_bridge_raw_source_sha256": HISTORICAL_BRIDGE_RAW_SHA256,
        "capture_request": {
            "backend": "cuda",
            "device_ordinal": 0,
            "cpu_threads": 1,
            "descriptor_memory": "host",
            "batch_sizes": [256 for _ in workloads],
            "workloads": [name for name, _ in workloads],
            "property_sets": [
                {"name": name, "compute_flags": flags} for name, flags in properties
            ],
            "cold_calls_per_owner": COLD_CALLS,
            "excluded_warmups_per_owner": warmups,
            "measured_calls_per_owner": measured,
            "owners": len(workloads) * len(properties) * len(variants),
            "planned_calls": len(workloads)
            * len(properties)
            * len(variants)
            * (COLD_CALLS + warmups + measured),
            "planned_peer_observations": 256
            * len(workloads)
            * len(properties)
            * len(variants)
            * (COLD_CALLS + warmups + measured),
            "on_sink_records_if_every_call_succeeds": len(workloads)
            * len(properties)
            * (COLD_CALLS + warmups + measured),
            "timed_boundary": "run.timed_invoke synchronized public invoke",
            "setup_timing": "owner construction only; outside native invoke timing",
            "warmup_semantics": "excluded FRESH calls; no SCC_START_WARM request",
            "historical_timing_replay": False,
        },
        "capture_options": {
            "model": run.public_api.XTBLOOM_MODEL_GFN2_XTB,
            "model_name": "GFN2-xTB",
            "max_scc_iterations": 500,
            "energy_tolerance": 1.0e-8,
            "charge_tolerance": 1.0e-6,
            "temperature_kelvin": 300.0,
            "electronic_temperature_hartree": (
                300.0 * run.public_api.XTBLOOM_KELVIN_TO_HARTREE
            ),
            "scc_start_mode": "FRESH",
        },
        "capture_options_not_emitted_by_historical_bridge_raw": {
            "scc_mixer": "modified_broyden",
            "scc_mixer_history": 8,
            "scc_mixer_damping": 0.4,
            "determinism": 0,
            "historically_unemitted_fields": ["scc_mixer", "determinism", "CPU ISA"],
        },
        "timing_variants": [
            {"name": name, "build_role": role, "diagnostic_sink": sink}
            for name, role, sink in (
                (name, role, name == "on-sink") for name, role in variants
            )
        ],
        "library_pins": {
            role: {
                "dso_path": pin["dso_path"],
                "dso_sha256": pin["dso_sha256"],
                "cache_path": pin["cache_path"],
                "cache_sha256": pin["cache_sha256"],
                "source_dir": pin["source_dir"],
                "source_commit": pin["source_commit"],
                "source_sha256": pin["source_sha256"],
            }
            for role, pin in pins.items()
        },
        "diagnostic_sink_env_name": DIAGNOSTIC_SINK_ENV,
    }


def _planned_rounds(
    warmups: int = EXCLUDED_WARMUPS,
    measured: int = MEASURED_CALLS,
) -> list[tuple[str, int]]:
    """Return the immutable one-cold, excluded-warmup, measured call plan."""
    return [
        ("cold", 0),
        *(("warmup", index + 1) for index in range(warmups)),
        *(("measured", index + 1) for index in range(measured)),
    ]


def _capture_one(
    adapter: Any,  # noqa: ANN401 - native and fake adapters share this protocol
    *,
    ordinal: int,
    workload: str,
    property_set: str,
    flags: int,
    variant: str,
    role: str,
    phase: str,
    round_index: int,
    variant_order: int,
    setup_seconds: float,
    case_ids: Sequence[str],
    trace_path: Path | None,
    trace_offset: int,
) -> tuple[dict[str, Any], bool, int]:
    """Run one synchronized public call and preserve raw buffers on all exits."""
    base = _call_record_base(
        ordinal,
        workload,
        property_set,
        flags,
        variant,
        role,
        phase,
        round_index,
        variant_order,
        setup_seconds,
        case_ids,
    )
    reset_error: str | None = None
    timing_error: str | None = None
    elapsed_ms: float | None = None
    invoked = False
    try:
        adapter.reset_outputs()
        adapter.last_call_status = None
        adapter.last_call_error = None
    except Exception as error:  # noqa: BLE001 - adapter implementations are injectable
        reset_error = f"{type(error).__name__}: {error}"

    if reset_error is None:
        previous = _sink_environment(trace_path)
        invoked = True
        try:
            elapsed_ms = run.timed_invoke(adapter)
        except Exception as error:  # noqa: BLE001 - includes ctypes/runtime failures
            timing_error = f"{type(error).__name__}: {error}"
        finally:
            _restore_sink_environment(previous)

    download_error: str | None = None
    if invoked:
        try:
            adapter.memory.download_outputs()
        except Exception as error:  # noqa: BLE001 - preserve partial host buffers
            download_error = f"{type(error).__name__}: {error}"

    call_status = getattr(adapter, "last_call_status", None) if invoked else None
    if type(call_status) is not int:
        call_status = None
    call_error = getattr(adapter, "last_call_error", None) if invoked else None
    if reset_error is not None:
        call_error = f"output reset failed: {reset_error}"
    elif timing_error is not None and call_error is None:
        call_error = timing_error

    result_flags = int(adapter.result.flags)
    peers = _peer_results(adapter)
    issues: list[str] = []
    if call_status is not None:
        issues.extend(_validate_published_outputs(adapter, call_status))
    if reset_error is not None:
        issues.append("owner cannot be safely called after output reset failure")
    if timing_error is not None:
        issues.append("native timing or synchronization failed; owner poisoned")
    if download_error is not None:
        issues.append(f"output download failed: {download_error}")
    if call_status is None:
        issues.append("native call status was not recorded")

    peer_failures = (
        [peer for peer in peers if peer["status"] != 0 or peer["converged"] != 1]
        if call_status == run.public_api.XTBLOOM_STATUS_SUCCESS
        else []
    )
    if call_status == run.public_api.XTBLOOM_STATUS_SUCCESS:
        unchanged = None
        owner_poisoned = bool(issues)
    else:
        unchanged = _sentinels_unchanged(adapter)
        if call_status == STATUS_INTERNAL_ERROR and not unchanged:
            issues.append(
                "INTERNAL_ERROR modified caller outputs after commit; owner poisoned"
            )
        elif call_status is not None and not unchanged:
            issues.append("failed call modified caller outputs; raw buffers retained")
        owner_poisoned = True

    trace_info: dict[str, Any] | None = None
    next_trace_offset = trace_offset
    if trace_path is not None:
        expected_trace = (
            call_status == run.public_api.XTBLOOM_STATUS_SUCCESS and adapter.systems > 0
        )
        try:
            trace_info, next_trace_offset, trace_issues = _trace_delta(
                trace_path,
                trace_offset,
                expected_trace,
                adapter.systems,
                peers if expected_trace else None,
            )
        except Exception as error:  # noqa: BLE001 - retain trace and continue matrix
            trace_issues = [
                f"diagnostic trace validation failed: {type(error).__name__}: {error}"
            ]
        issues.extend(trace_issues)

    record = {
        **base,
        "state": (
            "CAPTURED"
            if call_status == run.public_api.XTBLOOM_STATUS_SUCCESS
            and timing_error is None
            and reset_error is None
            and download_error is None
            else "ERROR"
        ),
        "elapsed_synchronized_public_invoke_ms": (
            None if elapsed_ms is None else float(elapsed_ms)
        ),
        "call_status": call_status,
        "call_status_name": _status_name(adapter, call_status)
        if call_status is not None
        else None,
        "call_error": call_error,
        "timing_error": timing_error,
        "output_download_error": download_error,
        "compute_options": {
            "model": int(adapter.options.model),
            "max_scc_iterations": int(adapter.options.max_scc_iterations),
            "energy_tolerance": float(adapter.options.energy_tolerance),
            "charge_tolerance": float(adapter.options.charge_tolerance),
            "electronic_temperature_hartree": float(
                adapter.options.electronic_temperature
            ),
            "scc_start_mode": int(adapter.options.scc_start_mode),
            "scc_mixer": int(adapter.options.scc_mixer),
            "scc_mixer_history": int(adapter.options.scc_mixer_history),
            "scc_mixer_damping": float(adapter.options.scc_mixer_damping),
            "determinism": int(adapter.options.determinism),
        },
        "result_flags_before": RESULT_FLAGS_SENTINEL,
        "result_flags_after": result_flags,
        "outputs_untouched_after_call_level_failure": unchanged,
        "peer_results": peers,
        "peer_failures": peer_failures,
        "energies_hartree": _json_float_array(adapter.energies),
        "forces_hartree_per_bohr": (
            _json_float_array(adapter.forces)
            if adapter.forces is not None
            else {"state": "not_requested"}
        ),
        "atomic_charges_e": (
            _json_float_array(adapter.charges)
            if adapter.charges is not None
            else {"state": "not_requested"}
        ),
        "diagnostic_trace": trace_info,
        "capture_issues": issues,
    }
    return record, owner_poisoned, next_trace_offset


def _not_run_record(
    ordinal: int,
    workload: str,
    property_set: str,
    flags: int,
    variant: str,
    role: str,
    phase: str,
    round_index: int,
    variant_order: int,
    setup_seconds: float | None,
    case_ids: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    """Represent every unavailable matrix coordinate instead of dropping it."""
    return {
        **_call_record_base(
            ordinal,
            workload,
            property_set,
            flags,
            variant,
            role,
            phase,
            round_index,
            variant_order,
            setup_seconds,
            case_ids,
        ),
        "state": "NOT_RUN",
        "elapsed_synchronized_public_invoke_ms": None,
        "reason": reason,
    }


def _new_output_directory(path: Path) -> Path:
    """Reserve a fresh output directory atomically without replacing user data."""
    requested = path.expanduser()
    if requested.exists() or requested.is_symlink():
        raise CaptureError(f"output directory already exists: {requested}")
    try:
        requested.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise CaptureError(
            f"cannot create fresh output directory {requested}"
        ) from error
    return requested.resolve(strict=True)


def _execute_matrix(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: dict[str, tuple[dict[str, Any], ...]],
    pins: dict[str, dict[str, str]],
    output_directory: Path,
    diagnostics_dir: Path,
    metadata: dict[str, Any],
    owners: dict[tuple[str, str, str], Any],
    owner_states: dict[tuple[str, str, str], dict[str, Any]],
    sink_paths: dict[tuple[str, str, str], Path | None],
    trace_offsets: dict[tuple[str, str, str], int],
    capture_issues: list[str],
    progress: dict[str, int],
    *,
    adapter_factory: Callable[..., Any],
    clock_ns: Callable[[], int],
    warmups: int,
    measured: int,
    workloads: Sequence[tuple[str, int]],
    properties: Sequence[tuple[str, int]],
    variants: Sequence[tuple[str, str]],
) -> None:
    """Create every owner, then append each executed or unavailable matrix row."""
    for workload, _ in workloads:
        case_sequence = selected[workload]
        for property_set, _ in properties:
            for variant, role in variants:
                key = (workload, property_set, variant)
                setup_start = clock_ns()
                trace_path: Path | None = None
                try:
                    if variant == "on-sink":
                        trace_path = diagnostics_dir / (
                            f"{workload}-{property_set}-{variant}.jsonl"
                        )
                        with trace_path.open("x", encoding="utf-8"):
                            pass
                    adapter = adapter_factory(
                        Path(pins[role]["dso_path"]),
                        manifest_path,
                        manifest,
                        case_sequence,
                        workload,
                        property_set,
                    )
                    owners[key] = adapter
                    setup_seconds = (clock_ns() - setup_start) * 1.0e-9
                    owner_states[key] = {
                        "poisoned": False,
                        "setup_state": "READY",
                        "setup_seconds": setup_seconds,
                        "error": None,
                    }
                    sink_paths[key] = trace_path
                    trace_offsets[key] = 0
                except Exception as error:  # noqa: BLE001 - owner initialization boundary
                    setup_seconds = (clock_ns() - setup_start) * 1.0e-9
                    message = f"{type(error).__name__}: {error}"
                    owner_states[key] = {
                        "poisoned": True,
                        "setup_state": "ERROR",
                        "setup_seconds": setup_seconds,
                        "error": message,
                    }
                    sink_paths[key] = trace_path
                    trace_offsets[key] = 0
                    capture_issues.append(
                        f"owner setup failed for {workload}/{property_set}/{variant}: "
                        f"{message}"
                    )
                metadata["owner_setup"].append(
                    {
                        "workload": workload,
                        "property_set": property_set,
                        "variant": variant,
                        "build_role": role,
                        **owner_states[key],
                        "diagnostic_trace_path": (
                            str(trace_path.relative_to(output_directory))
                            if trace_path is not None
                            else None
                        ),
                    }
                )
                _write_json(output_directory / "metadata.json", metadata)

    calls_path = output_directory / "calls.jsonl"
    with calls_path.open("x", encoding="utf-8") as calls_output:
        for workload_index, (workload, _) in enumerate(workloads):
            case_ids = [case["id"] for case in selected[workload]]
            for property_set, flags in properties:
                for round_position, (phase, round_index) in enumerate(
                    _planned_rounds(warmups, measured)
                ):
                    variant_start = (workload_index + round_position) % len(variants)
                    variant_ordering = [
                        *variants[variant_start:],
                        *variants[:variant_start],
                    ]
                    for order_index, (variant, role) in enumerate(variant_ordering):
                        ordinal = progress["ordinal"] + 1
                        key = (workload, property_set, variant)
                        state = owner_states[key]
                        if key not in owners:
                            row = _not_run_record(
                                ordinal,
                                workload,
                                property_set,
                                flags,
                                variant,
                                role,
                                phase,
                                round_index,
                                order_index,
                                state["setup_seconds"],
                                case_ids,
                                f"owner setup failed: {state['error']}",
                            )
                        elif state["poisoned"]:
                            row = _not_run_record(
                                ordinal,
                                workload,
                                property_set,
                                flags,
                                variant,
                                role,
                                phase,
                                round_index,
                                order_index,
                                state["setup_seconds"],
                                case_ids,
                                "owner was poisoned by an earlier call-level or "
                                "output failure",
                            )
                        else:
                            row, poisoned, next_offset = _capture_one(
                                owners[key],
                                ordinal=ordinal,
                                workload=workload,
                                property_set=property_set,
                                flags=flags,
                                variant=variant,
                                role=role,
                                phase=phase,
                                round_index=round_index,
                                variant_order=order_index,
                                setup_seconds=state["setup_seconds"],
                                case_ids=case_ids,
                                trace_path=sink_paths[key],
                                trace_offset=trace_offsets[key],
                            )
                            trace_offsets[key] = next_offset
                            state["poisoned"] = poisoned
                            row_issues = row.get("capture_issues", [])
                            capture_issues.extend(
                                f"call {ordinal}: {issue}" for issue in row_issues
                            )
                            if row.get("call_status") not in (0, None):
                                capture_issues.append(
                                    f"call {ordinal}: native status "
                                    f"{row['call_status']}"
                                )
                            peer_failures = row.get("peer_failures", [])
                            capture_issues.extend(
                                f"call {ordinal}: peer failure "
                                f"{failure['system_index']} "
                                f"status={failure['status']} "
                                f"converged={failure['converged']}"
                                for failure in peer_failures
                            )
                            trace = row.get("diagnostic_trace")
                            if isinstance(trace, dict):
                                progress["diagnostic_trace_records"] += int(
                                    trace["parsed_records"]
                                )
                            if row.get("state") == "ERROR":
                                capture_issues.append(
                                    f"call {ordinal}: {row.get('call_error')}"
                                )
                        _write_json_line(calls_output, row)
                        progress["ordinal"] = ordinal
                        if row["state"] == "NOT_RUN":
                            progress["not_run_rows"] += 1


def _close_owner(
    key: tuple[str, str, str],
    adapter: Any,  # noqa: ANN401
) -> str | None:
    """Close one persistent native owner and return a safe failure summary."""
    try:
        adapter.close()
    except Exception as error:  # noqa: BLE001 - cleanup must continue for all owners
        return f"{key}: {type(error).__name__}: {error}"
    return None


def run_capture(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected: dict[str, tuple[dict[str, Any], ...]],
    manifest_sha256: str,
    pins: dict[str, dict[str, str]],
    output_directory: Path,
    *,
    adapter_factory: Callable[..., Any] = NativeAdapter,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    warmups: int = EXCLUDED_WARMUPS,
    measured: int = MEASURED_CALLS,
    workloads: Sequence[tuple[str, int]] = CAPTURE_WORKLOADS,
    properties: Sequence[tuple[str, int]] = CAPTURE_PROPERTIES,
    variants: Sequence[tuple[str, str]] = CAPTURE_VARIANTS,
) -> int:
    """Execute the fixed persistent-owner schedule with injectable test seams."""
    if DIAGNOSTIC_SINK_ENV in os.environ:
        raise CaptureError(
            f"inherited {DIAGNOSTIC_SINK_ENV} is not permitted for this capture"
        )
    before_pins = _verify_all_builds(pins)
    output_directory = _new_output_directory(output_directory)
    diagnostics_dir = output_directory / "diagnostics"
    diagnostics_dir.mkdir()
    inputs_path = output_directory / "inputs.json"
    with inputs_path.open("x", encoding="utf-8") as output:
        json.dump(
            _input_archive(manifest_path, manifest, selected, workloads),
            output,
            allow_nan=False,
            separators=(",", ":"),
        )
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())

    metadata = _protocol_metadata(
        manifest,
        manifest_sha256,
        pins,
        workloads,
        properties,
        variants,
        warmups,
        measured,
    )
    metadata.update(
        {
            "output_state": "running",
            "inputs_artifact": inputs_path.name,
            "pre_run_pins": before_pins,
            "post_run_pins": None,
            "owner_setup": [],
            "cleanup_failures": [],
            "capture_issues": [],
        }
    )
    _write_json(output_directory / "metadata.json", metadata)

    if not workloads or not properties or not variants:
        raise CaptureError("capture matrix dimensions must be nonempty")

    owners: dict[tuple[str, str, str], Any] = {}
    owner_states: dict[tuple[str, str, str], dict[str, Any]] = {}
    sink_paths: dict[tuple[str, str, str], Path | None] = {}
    trace_offsets: dict[tuple[str, str, str], int] = {}
    capture_issues: list[str] = []
    cleanup_failures: list[str] = []
    progress = {"ordinal": 0, "not_run_rows": 0, "diagnostic_trace_records": 0}
    matrix_error: str | None = None
    try:
        _execute_matrix(
            manifest_path,
            manifest,
            selected,
            pins,
            output_directory,
            diagnostics_dir,
            metadata,
            owners,
            owner_states,
            sink_paths,
            trace_offsets,
            capture_issues,
            progress,
            adapter_factory=adapter_factory,
            clock_ns=clock_ns,
            warmups=warmups,
            measured=measured,
            workloads=workloads,
            properties=properties,
            variants=variants,
        )
    except Exception as error:  # noqa: BLE001 - finalize and close after any run failure
        matrix_error = f"{type(error).__name__}: {error}"
        capture_issues.append(f"matrix execution failed: {matrix_error}")

    for key, adapter in owners.items():
        message = _close_owner(key, adapter)
        if message is not None:
            cleanup_failures.append(message)
            capture_issues.append(f"owner cleanup failed: {message}")

    try:
        after_pins = _verify_all_builds(pins)
    except Exception as error:  # noqa: BLE001 - retain failed post-run verification
        after_pins = None
        capture_issues.append(f"post-run pin verification failed: {error}")
    else:
        capture_issues.extend(
            f"{role} build identity changed during capture"
            for role in before_pins
            if before_pins[role] != after_pins[role]
        )

    planned_rows = (
        len(workloads)
        * len(properties)
        * len(variants)
        * len(_planned_rounds(warmups, measured))
    )

    metadata.update(
        {
            "output_state": "complete" if not capture_issues else "invalid",
            "post_run_pins": after_pins,
            "cleanup_failures": cleanup_failures,
            "capture_issues": capture_issues,
            "matrix_execution_error": matrix_error,
            "planned_call_rows": planned_rows,
            "captured_call_rows": progress["ordinal"],
            "explicit_not_run_rows": progress["not_run_rows"],
            "unrecorded_call_rows": planned_rows - progress["ordinal"],
            "unrecorded_rows_state": "NOT_RUN" if matrix_error else None,
            "diagnostic_trace_records": progress["diagnostic_trace_records"],
            "capture_valid": (
                not capture_issues and not cleanup_failures and matrix_error is None
            ),
            "claim_eligible": False,
            "performance_claim_eligible": False,
            "correctness_oracle_eligible": False,
        }
    )
    _write_json(output_directory / "metadata.json", metadata)
    return 0 if metadata["capture_valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Define mandatory corpus/build pins and the optional offline description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--historical-bridge-raw-sha256",
        required=True,
        help="SHA-256 of the historical request-protocol JSONL, not the input JSON",
    )
    parser.add_argument("--library-pins", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="validate pinned corpus metadata without loading a native library",
    )
    return parser


def _describe(
    manifest: dict[str, Any],
    selected: dict[str, tuple[dict[str, Any], ...]],
    manifest_sha256: str,
    pins: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Return offline workload/variant protocol facts without touching DSOs."""
    return {
        "schema": "xtbloom.historical-alkane-capture.describe.v1",
        "manifest_sha256": manifest_sha256,
        "historical_bridge_raw_source_sha256": HISTORICAL_BRIDGE_RAW_SHA256,
        "manifest_source_sha256": manifest["source"]["sha256"],
        "workloads": [
            {
                "name": workload,
                "atom_count": natoms,
                "case_count": len(selected[workload]),
                "first_case_id": selected[workload][0]["id"],
                "last_case_id": selected[workload][-1]["id"],
            }
            for workload, natoms in CAPTURE_WORKLOADS
        ],
        "variants": [
            {"name": name, "build_role": role} for name, role in CAPTURE_VARIANTS
        ],
        "build_pins": {
            role: {
                "dso_path": pin["dso_path"],
                "dso_sha256": pin["dso_sha256"],
                "cache_path": pin["cache_path"],
                "cache_sha256": pin["cache_sha256"],
                "source_dir": pin["source_dir"],
                "source_commit": pin["source_commit"],
                "source_sha256": pin["source_sha256"],
            }
            for role, pin in pins.items()
        },
        "native_library_loaded": False,
        "claim_eligible": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate required pins, then describe offline or execute the capture."""
    args = build_parser().parse_args(argv)
    try:
        if not args.describe and args.output_dir is None:
            raise CaptureError("--output-dir is required unless --describe is used")
        manifest, selected, actual_manifest_sha256 = _load_manifest_bundle(
            args.manifest,
            args.manifest_sha256,
            args.historical_bridge_raw_sha256,
        )
        pins = _validate_library_pins(args.library_pins)
        if args.describe:
            print(  # noqa: T201 - describe-mode CLI output
                json.dumps(
                    _describe(manifest, selected, actual_manifest_sha256, pins),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return run_capture(
            args.manifest.expanduser().resolve(strict=True),
            manifest,
            selected,
            actual_manifest_sha256,
            pins,
            args.output_dir,
        )
    except (CaptureError, run.conformance.ConformanceError) as error:
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 - CLI diagnostics
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
