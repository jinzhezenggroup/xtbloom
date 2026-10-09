"""Capture hash-pinned workload inputs into private read-only snapshots.

The manifest and each distinct source input are read once. Hashing, manifest
parsing, and snapshot creation all use those captured bytes, so later changes to
the original paths cannot change the data consumed by parsers or native setup.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class FrozenInputError(ValueError):
    """A pinned workload manifest or one of its frozen inputs is invalid."""


@dataclass
class FrozenWorkload:
    """Captured manifest metadata and snapshot paths owned until :meth:`close`.

    ``manifest`` is parsed from the exact bytes named by ``manifest_record``.
    Its case entries retain their source ``input`` values; callers must use
    ``cases_by_id`` for input parsing and native assembly because those copies
    point only to the immutable-by-convention private snapshots. Snapshot files
    are mode 0400, and ``captured_input_bytes`` counts each unique source inode
    once, even when several case IDs resolve to aliases of that input.
    """

    manifest: dict[str, Any]
    cases_by_id: dict[str, dict[str, Any]]
    manifest_record: dict[str, Any]
    input_records_by_case_id: dict[str, dict[str, Any]]
    captured_input_bytes: int
    _temporary_directory: Any = field(repr=False, compare=False)

    def close(self) -> None:
        """Remove private snapshots; repeated calls are harmless."""
        temporary_directory = self._temporary_directory
        if temporary_directory is None:
            return
        temporary_directory.cleanup()
        self._temporary_directory = None


def _is_lowercase_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise FrozenInputError(f"non-finite JSON number is forbidden: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FrozenInputError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _parse_manifest(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
    except FrozenInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise FrozenInputError(
            f"cannot parse captured workload manifest {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise FrozenInputError("captured workload manifest must be a JSON object")
    return document


def _regular_file_identity(
    path: Path, description: str
) -> tuple[Path, tuple[object, ...]]:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except (OSError, RuntimeError, ValueError) as exc:
        raise FrozenInputError(f"cannot resolve {description} {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FrozenInputError(f"{description} is not a regular file: {path}")
    if metadata.st_ino:
        identity = (metadata.st_dev, metadata.st_ino)
    else:
        identity = (str(resolved),)
    return resolved, identity


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FrozenInputError(f"cannot capture {description} {path}: {exc}") from exc


def _write_snapshot(directory: Path, index: int, payload: bytes) -> Path:
    snapshot_path = directory / f"input-{index:06d}"
    try:
        with snapshot_path.open("xb") as handle:
            handle.write(payload)
        snapshot_path.chmod(0o400)
    except OSError as exc:
        raise FrozenInputError(
            f"cannot write private frozen input snapshot {snapshot_path}: {exc}"
        ) from exc
    return snapshot_path


def capture_workload(
    manifest_path: Path,
    expected_manifest_sha256: str,
    frozen_case_ids: tuple[str, ...],
    *,
    input_resolver: Callable[[dict[str, Any]], Path],
) -> FrozenWorkload:
    """Capture and verify the complete frozen roster from one manifest read.

    Every source input is read once, hashed against the matching manifest field,
    then copied from those same captured bytes to a private read-only file. File
    IDs never appear in snapshot names. Resolved hardlink/symlink/path aliases
    share one capture and snapshot, but each case's expected hash is checked
    independently. This does not defend against a process with the same UID
    deliberately changing the private snapshot after capture.
    """
    if not _is_lowercase_sha256(expected_manifest_sha256):
        raise FrozenInputError(
            "expected workload manifest identity must be a lowercase SHA-256"
        )
    if not isinstance(manifest_path, Path):
        raise FrozenInputError("workload manifest path must be a Path")
    if not isinstance(frozen_case_ids, tuple) or not frozen_case_ids:
        raise FrozenInputError("frozen case IDs must be a nonempty tuple")
    if any(
        not isinstance(case_id, str) or not case_id or case_id.strip() != case_id
        for case_id in frozen_case_ids
    ):
        raise FrozenInputError("frozen case IDs must be nonempty trimmed strings")
    if len(set(frozen_case_ids)) != len(frozen_case_ids):
        raise FrozenInputError("frozen case IDs must be unique")
    if not callable(input_resolver):
        raise FrozenInputError("input_resolver must be callable")

    resolved_manifest_path, _ = _regular_file_identity(
        manifest_path, "workload manifest"
    )
    captured_manifest = _read_bytes(resolved_manifest_path, "workload manifest")
    manifest_sha256 = _sha256(captured_manifest)
    if manifest_sha256 != expected_manifest_sha256:
        raise FrozenInputError(
            "captured workload manifest does not match the expected SHA-256"
        )
    manifest = _parse_manifest(captured_manifest, manifest_path)
    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise FrozenInputError("captured workload manifest 'cases' must be an array")

    cases_by_id_from_manifest: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(manifest_cases):
        if not isinstance(case, dict):
            raise FrozenInputError(f"captured workload case {index} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise FrozenInputError(
                f"captured workload case {index} must have a nonempty string ID"
            )
        if case_id in cases_by_id_from_manifest:
            raise FrozenInputError(f"duplicate workload case ID: {case_id!r}")
        cases_by_id_from_manifest[case_id] = case

    missing_case_ids = [
        case_id
        for case_id in frozen_case_ids
        if case_id not in cases_by_id_from_manifest
    ]
    if missing_case_ids:
        raise FrozenInputError(
            "frozen case IDs are absent from the captured workload manifest: "
            + ", ".join(missing_case_ids)
        )

    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="xtbloom-frozen-inputs-"
        )
    except OSError as exc:
        raise FrozenInputError(
            f"cannot create private input snapshot directory: {exc}"
        ) from exc

    directory = Path(temporary_directory.name)
    snapshot_by_identity: dict[tuple[object, ...], tuple[Path, str, int]] = {}
    snapshot_cases: dict[str, dict[str, Any]] = {}
    input_records: dict[str, dict[str, Any]] = {}
    captured_input_bytes = 0
    try:
        for case_id in frozen_case_ids:
            source_case = cases_by_id_from_manifest[case_id]
            expected_input_sha256 = source_case.get("input_sha256")
            if not _is_lowercase_sha256(expected_input_sha256):
                raise FrozenInputError(
                    f"input SHA-256 is missing or malformed for case {case_id!r}"
                )
            try:
                source_path = input_resolver(source_case)
            except Exception as exc:
                raise FrozenInputError(
                    f"cannot resolve input for case {case_id!r}: {exc}"
                ) from exc
            if not isinstance(source_path, Path):
                raise FrozenInputError(
                    f"input resolver must return a Path for case {case_id!r}"
                )
            resolved_path, identity = _regular_file_identity(
                source_path, f"input for case {case_id!r}"
            )
            captured = snapshot_by_identity.get(identity)
            if captured is None:
                payload = _read_bytes(resolved_path, f"input for case {case_id!r}")
                actual_input_sha256 = _sha256(payload)
                if actual_input_sha256 != expected_input_sha256:
                    raise FrozenInputError(
                        f"input SHA-256 mismatch for frozen case {case_id!r}"
                    )
                snapshot_path = _write_snapshot(
                    directory, len(snapshot_by_identity), payload
                )
                captured = (snapshot_path, actual_input_sha256, len(payload))
                snapshot_by_identity[identity] = captured
                captured_input_bytes += len(payload)
            snapshot_path, actual_input_sha256, input_size_bytes = captured
            if actual_input_sha256 != expected_input_sha256:
                raise FrozenInputError(
                    "aliased frozen inputs have inconsistent expected SHA-256 values "
                    f"for case {case_id!r}"
                )

            snapshot_case = copy.deepcopy(source_case)
            snapshot_case["input"] = str(snapshot_path)
            snapshot_cases[case_id] = snapshot_case
            input_records[case_id] = {
                "path": str(source_path),
                "sha256": actual_input_sha256,
                "size_bytes": input_size_bytes,
            }

        return FrozenWorkload(
            manifest=manifest,
            cases_by_id=snapshot_cases,
            manifest_record={
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "size_bytes": len(captured_manifest),
            },
            input_records_by_case_id=input_records,
            captured_input_bytes=captured_input_bytes,
            _temporary_directory=temporary_directory,
        )
    except Exception as exc:
        with suppress(OSError):
            temporary_directory.cleanup()
        if isinstance(exc, FrozenInputError):
            raise
        raise FrozenInputError(f"cannot capture frozen workload inputs: {exc}") from exc
