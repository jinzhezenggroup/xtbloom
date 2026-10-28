"""Bounded, endpoint-only observation of file-backed process mappings.

This module does not load a library or establish a runtime dependency closure.
It records the file-backed mappings visible in two Linux ``/proc/self/maps``
snapshots around file hashing; images loaded and unloaded between snapshots are
not observed, and anonymous or pseudo-path mappings are only counted.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class MappingError(RuntimeError):
    """Raised when an endpoint mapping observation cannot be trusted."""


_MAX_MAPS_BYTES = 16 * 1024 * 1024
_MAX_ADDRESS = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_MAX_UINT64 = (1 << 64) - 1
_MAPS_LINE_RE = re.compile(
    r"^(\S+)[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)(?:[ \t]+(.*))?$"
)
_ENTRYPOINT_KEY_RE = re.compile(
    r"^owner-(0|[1-9][0-9]*):(xtbloom_compute|xtbloom_context_create)$"
)
_ENTRYPOINT_SYMBOLS = ("xtbloom_compute", "xtbloom_context_create")
SCHEMA_VERSION = 1
OBSERVATION_KIND = "xtbloom-linux-mapped-image-endpoint"


def library_entrypoints(libraries: Sequence[ctypes.CDLL]) -> dict[str, int]:
    """Return addresses of the two public functions for each loaded CDLL.

    The function pointers are inspected but never called. Callers should pass
    the actual CDLL objects whose mappings they intend to observe.
    """
    try:
        iterator = iter(libraries)
    except TypeError as exc:
        raise MappingError(
            "libraries must be a sequence of ctypes.CDLL objects"
        ) from exc

    result: dict[str, int] = {}
    for owner_index, library in enumerate(iterator):
        if not isinstance(library, ctypes.CDLL):
            raise MappingError(f"library owner-{owner_index} is not a ctypes.CDLL")
        for symbol in _ENTRYPOINT_SYMBOLS:
            try:
                function = getattr(library, symbol)
                address = ctypes.cast(function, ctypes.c_void_p).value
            except (AttributeError, TypeError, ValueError) as exc:
                raise MappingError(
                    f"cannot obtain {symbol} from library owner-{owner_index}"
                ) from exc
            if (
                not isinstance(address, int)
                or isinstance(address, bool)
                or address <= 0
            ):
                raise MappingError(
                    f"{symbol} from library owner-{owner_index} has no valid address"
                )
            result[f"owner-{owner_index}:{symbol}"] = address
    return result


def _parse_hex(value: str, field: str, line_number: int) -> int:
    if not re.fullmatch(r"[0-9a-fA-F]+", value):
        raise MappingError(f"invalid {field} on maps line {line_number}")
    return int(value, 16)


def _classify_path(path: str, line_number: int) -> str:
    if not path:
        return "anonymous"
    try:
        path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MappingError(f"invalid Unicode path on maps line {line_number}") from exc
    if any(unicodedata.category(character) == "Cc" for character in path):
        raise MappingError(f"control character in maps path on line {line_number}")
    if "\\" in path:
        raise MappingError(f"ambiguous backslash in maps path on line {line_number}")
    if path.startswith("["):
        if not (path.startswith("[") and path.endswith("]")):
            raise MappingError(f"malformed pseudo-path on maps line {line_number}")
        return "pseudopath"
    if path.endswith(" (deleted)"):
        raise MappingError(f"deleted file mapping on maps line {line_number}")
    if not path.startswith("/"):
        raise MappingError(f"non-absolute file path on maps line {line_number}")
    return "file"


def parse_maps(text: str) -> list[dict[str, int | str]]:
    """Parse complete Linux maps lines and reject ambiguous mapping records.

    Returned addresses and offsets are integers. ``kind`` is ``file`` for an
    absolute file-backed path, ``pseudopath`` for bracketed kernel labels, and
    ``anonymous`` when no path is present. A nonempty input must end at a line
    boundary so a truncated final record cannot be mistaken for a snapshot.
    """
    if not isinstance(text, str):
        raise MappingError("maps snapshot must be text")
    if not text:
        return []
    if not text.endswith("\n"):
        raise MappingError("maps snapshot ends with a truncated line")

    mappings: list[dict[str, int | str]] = []
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        match = _MAPS_LINE_RE.fullmatch(line)
        if match is None:
            raise MappingError(f"malformed maps line {line_number}")
        address_range, permissions, offset_text, device_text, inode_text, path = (
            match.groups()
        )
        range_match = re.fullmatch(r"([0-9a-fA-F]+)-([0-9a-fA-F]+)", address_range)
        if range_match is None:
            raise MappingError(f"invalid address range on maps line {line_number}")
        start = int(range_match.group(1), 16)
        end = int(range_match.group(2), 16)
        if start >= end or end > _MAX_ADDRESS:
            raise MappingError(f"invalid address range on maps line {line_number}")
        if not re.fullmatch(r"[r-][w-][x-][ps]", permissions):
            raise MappingError(f"invalid permissions on maps line {line_number}")

        offset = _parse_hex(offset_text, "offset", line_number)
        if offset > _MAX_UINT64:
            raise MappingError(f"offset is out of range on maps line {line_number}")
        device_match = re.fullmatch(r"([0-9a-fA-F]+):([0-9a-fA-F]+)", device_text)
        if device_match is None:
            raise MappingError(f"invalid device on maps line {line_number}")
        device_major = int(device_match.group(1), 16)
        device_minor = int(device_match.group(2), 16)
        if device_major > 0xFFFFFFFF or device_minor > 0xFFFFFFFF:
            raise MappingError(f"device is out of range on maps line {line_number}")
        if not re.fullmatch(r"[0-9]+", inode_text):
            raise MappingError(f"invalid inode on maps line {line_number}")
        inode = int(inode_text, 10)
        if inode > _MAX_UINT64:
            raise MappingError(f"inode is out of range on maps line {line_number}")

        pathname = path or ""
        kind = _classify_path(pathname, line_number)
        if kind == "anonymous" and inode:
            raise MappingError(f"unnamed file inode on maps line {line_number}")
        mappings.append(
            {
                "start": start,
                "end": end,
                "permissions": permissions,
                "offset": offset,
                "device_major": device_major,
                "device_minor": device_minor,
                "inode": inode,
                "path": pathname,
                "kind": kind,
            }
        )

    mappings.sort(key=lambda mapping: (int(mapping["start"]), int(mapping["end"])))
    previous_end = -1
    for mapping in mappings:
        start = int(mapping["start"])
        if start < previous_end:
            raise MappingError("overlapping or duplicate maps ranges")
        previous_end = int(mapping["end"])
    return mappings


def _read_maps() -> str:
    """Read a complete bounded snapshot to EOF without retrying the capture.

    The kernel can return a short read at a complete record boundary. Only EOF
    establishes completeness; the total byte cap also bounds nonempty reads.
    """
    if not sys.platform.startswith("linux"):
        raise MappingError("/proc/self/maps observation is supported only on Linux")

    descriptor: int | None = None
    try:
        descriptor = os.open(
            "/proc/self/maps", os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        )
        raw = bytearray()
        while True:
            block = os.read(descriptor, min(64 * 1024, _MAX_MAPS_BYTES + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
            if len(raw) > _MAX_MAPS_BYTES:
                raise MappingError("/proc/self/maps exceeds the observation size limit")
        if raw and not raw.endswith(b"\n"):
            raise MappingError("/proc/self/maps read was truncated")
    except OSError as exc:
        raise MappingError(f"cannot read /proc/self/maps: {exc}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise MappingError(f"cannot close /proc/self/maps: {exc}") from exc

    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MappingError("/proc/self/maps is not valid UTF-8") from exc


def _file_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _check_file_metadata(
    metadata: os.stat_result,
    path: str,
    expected_identity: tuple[int, int, int] | None,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise MappingError(f"mapped path is not a regular file: {path!r}")
    if expected_identity is not None:
        expected_major, expected_minor, expected_inode = expected_identity
        actual = (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)
        if actual != (expected_major, expected_minor, expected_inode):
            raise MappingError(f"mapped path identity changed: {path!r}")


def _hash_descriptor(descriptor: int, size_bytes: int) -> tuple[str, bool]:
    """Hash a fixed extent so a growing file cannot extend the observation."""
    digest = hashlib.sha256()
    prefix = bytearray()
    remaining = size_bytes
    while remaining:
        block = os.read(descriptor, min(remaining, 1024 * 1024))
        if not block:
            raise MappingError("mapped file length changed while hashing")
        remaining -= len(block)
        if len(prefix) < 4:
            prefix.extend(block[: 4 - len(prefix)])
        digest.update(block)
    if os.read(descriptor, 1):
        raise MappingError("mapped file length changed while hashing")
    return digest.hexdigest(), bytes(prefix) == b"\x7fELF"


def _observe_file(
    path: str,
    expected_identity: tuple[int, int, int],
    *,
    nofollow: bool = False,
) -> tuple[dict[str, int | str | bool], tuple[int, int, int, int, int, int]]:
    """Pin descriptor identity and reread bytes despite coarse inode timestamps."""
    descriptor: int | None = None
    try:
        before_path = os.stat(path)
        _check_file_metadata(before_path, path, expected_identity)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        if nofollow:
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before_fd = os.fstat(descriptor)
        _check_file_metadata(before_fd, path, expected_identity)
        if _file_fingerprint(before_path) != _file_fingerprint(before_fd):
            raise MappingError(f"mapped path changed while opening: {path!r}")

        digest, is_elf = _hash_descriptor(descriptor, before_fd.st_size)
        os.lseek(descriptor, 0, os.SEEK_SET)
        repeated_digest, _ = _hash_descriptor(descriptor, before_fd.st_size)
        if digest != repeated_digest:
            raise MappingError(f"mapped file bytes changed while hashing: {path!r}")

        after_fd = os.fstat(descriptor)
        after_path = os.stat(path)
        _check_file_metadata(after_fd, path, expected_identity)
        _check_file_metadata(after_path, path, expected_identity)
        fingerprint = _file_fingerprint(before_fd)
        if fingerprint != _file_fingerprint(
            after_fd
        ) or fingerprint != _file_fingerprint(after_path):
            raise MappingError(f"mapped file changed while hashing: {path!r}")

        record: dict[str, int | str | bool] = {
            "path": path,
            "size_bytes": after_fd.st_size,
            "sha256": digest,
            "device_major": os.major(after_fd.st_dev),
            "device_minor": os.minor(after_fd.st_dev),
            "inode": after_fd.st_ino,
            "is_elf": is_elf,
        }
        return record, fingerprint
    except MappingError:
        raise
    except OSError as exc:
        raise MappingError(f"cannot inspect mapped file {path!r}: {exc}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise MappingError(f"cannot close mapped file {path!r}: {exc}") from exc


def _validate_entrypoints(entrypoints: dict[str, int]) -> None:
    if not isinstance(entrypoints, dict) or not entrypoints:
        raise MappingError("at least one complete owner entrypoint pair is required")

    owners: dict[int, set[str]] = {}
    for key, address in entrypoints.items():
        if not isinstance(key, str):
            raise MappingError("entrypoint keys must be strings")
        match = _ENTRYPOINT_KEY_RE.fullmatch(key)
        if match is None:
            raise MappingError(f"invalid entrypoint key: {key!r}")
        if not isinstance(address, int) or isinstance(address, bool) or address <= 0:
            raise MappingError(
                f"entrypoint {key!r} must have a positive integer address"
            )
        owner = int(match.group(1), 10)
        owners.setdefault(owner, set()).add(match.group(2))

    if sorted(owners) != list(range(len(owners))):
        raise MappingError("entrypoint owners must be contiguous from owner-0")
    for owner, symbols in owners.items():
        if symbols != set(_ENTRYPOINT_SYMBOLS):
            raise MappingError(f"owner-{owner} is missing a required public entrypoint")


def _file_mappings(mappings: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    return [mapping for mapping in mappings if mapping["kind"] == "file"]


def _mapping_identity(mapping: dict[str, int | str]) -> tuple[int, int, int]:
    identity = (
        int(mapping["device_major"]),
        int(mapping["device_minor"]),
        int(mapping["inode"]),
    )
    if identity[2] <= 0:
        raise MappingError(f"file mapping has no inode: {mapping['path']!r}")
    return identity


def _stable_file_mappings(
    first: list[dict[str, int | str]], second: list[dict[str, int | str]]
) -> bool:
    return first == second


def _mapping_pin(mapping: dict[str, int | str]) -> dict[str, int | str]:
    return {
        "start": int(mapping["start"]),
        "end": int(mapping["end"]),
        "permissions": str(mapping["permissions"]),
        "offset": int(mapping["offset"]),
        "device_major": int(mapping["device_major"]),
        "device_minor": int(mapping["device_minor"]),
        "inode": int(mapping["inode"]),
        "path": str(mapping["path"]),
    }


def _counts(mappings: list[dict[str, int | str]]) -> dict[str, int]:
    return {
        "anonymous": sum(mapping["kind"] == "anonymous" for mapping in mappings),
        "pseudopath": sum(mapping["kind"] == "pseudopath" for mapping in mappings),
    }


def capture(
    library_path: Path,
    entrypoints: dict[str, int],
    expected_sha256: str,
) -> dict:
    """Capture a bounded endpoint observation without loading ``library_path``.

    Two real maps reads bracket descriptor hashing. Every file-backed map is
    opened by its kernel-reported path and checked against its mapped device
    and inode. This endpoint evidence does not establish transient images, a
    complete dependency closure, resident memory, or performance admission.
    """
    if not sys.platform.startswith("linux"):
        raise MappingError("/proc/self/maps observation is supported only on Linux")
    _validate_entrypoints(entrypoints)
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise MappingError("expected_sha256 must contain exactly 64 hexadecimal digits")

    try:
        resolved_path = Path(library_path).resolve(strict=True)
        resolved_name = str(resolved_path)
        before_mappings = parse_maps(_read_maps())
        before_file_mappings = _file_mappings(before_mappings)
        selected_path_stat = os.stat(resolved_name)
    except MappingError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise MappingError(
            f"cannot resolve selected library {library_path!r}: {exc}"
        ) from exc

    _check_file_metadata(selected_path_stat, resolved_name, None)
    selected_identity = (
        os.major(selected_path_stat.st_dev),
        os.minor(selected_path_stat.st_dev),
        selected_path_stat.st_ino,
    )
    selected_mappings = [
        mapping
        for mapping in before_file_mappings
        if _mapping_identity(mapping) == selected_identity
    ]
    if not selected_mappings:
        raise MappingError("selected library inode is not present in /proc/self/maps")

    selected_record, selected_fingerprint = _observe_file(
        resolved_name, selected_identity, nofollow=True
    )
    if selected_record["is_elf"] is not True:
        raise MappingError("selected library does not have ELF magic")
    selected_digest = str(selected_record["sha256"])
    if selected_digest.lower() != expected_sha256.lower():
        raise MappingError("selected library SHA-256 does not match the expected pin")

    image_observations: dict[
        tuple[int, int, int, str],
        tuple[dict[str, int | str | bool], tuple[int, int, int, int, int, int]],
    ] = {}
    for mapping in before_file_mappings:
        identity = _mapping_identity(mapping)
        mapped_path = str(mapping["path"])
        image_key = (*identity, mapped_path)
        if image_key in image_observations:
            continue
        if identity == selected_identity and mapped_path == resolved_name:
            image_observations[image_key] = (selected_record, selected_fingerprint)
        else:
            image_observations[image_key] = _observe_file(mapped_path, identity)

    try:
        after_mappings = parse_maps(_read_maps())
    except MappingError:
        raise
    after_file_mappings = _file_mappings(after_mappings)
    if not _stable_file_mappings(before_file_mappings, after_file_mappings):
        raise MappingError("file-backed maps entries changed during the observation")

    for (_, _, _, mapped_path), (_, fingerprint) in image_observations.items():
        try:
            path_stat = os.stat(mapped_path)
        except OSError as exc:
            raise MappingError(
                f"mapped path disappeared after hashing: {mapped_path!r}"
            ) from exc
        if _file_fingerprint(path_stat) != fingerprint:
            raise MappingError(f"mapped path changed after hashing: {mapped_path!r}")
    try:
        selected_path_after = os.stat(resolved_name)
    except OSError as exc:
        raise MappingError(
            f"selected library path disappeared: {resolved_name!r}"
        ) from exc
    if _file_fingerprint(selected_path_after) != selected_fingerprint:
        raise MappingError("selected library path changed after hashing")

    selected_images = [
        record
        for (identity_major, identity_minor, identity_inode, _mapped_path), (
            record,
            _fingerprint,
        ) in image_observations.items()
        if (identity_major, identity_minor, identity_inode) == selected_identity
    ]
    if not selected_images or any(
        record["sha256"] != selected_digest or record["is_elf"] is not True
        for record in selected_images
    ):
        raise MappingError(
            "mapped selected image does not match the selected library pin"
        )

    for address_key, address in entrypoints.items():
        if not any(
            _mapping_identity(mapping) == selected_identity
            and "x" in str(mapping["permissions"])
            and int(mapping["start"]) <= address < int(mapping["end"])
            for mapping in before_file_mappings
        ):
            raise MappingError(
                f"entrypoint {address_key!r} is outside selected executable mappings"
            )

    selected_library = {
        "resolved_path": resolved_name,
        "size_bytes": int(selected_record["size_bytes"]),
        "sha256": selected_digest,
        "device_major": selected_identity[0],
        "device_minor": selected_identity[1],
        "inode": selected_identity[2],
        "mtime_ns": selected_path_after.st_mtime_ns,
        "ctime_ns": selected_path_after.st_ctime_ns,
    }
    images = [dict(record) for record, _ in image_observations.values()]
    maps_pins = [_mapping_pin(mapping) for mapping in before_file_mappings]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": OBSERVATION_KIND,
        "selected_library": selected_library,
        "entrypoints": dict(entrypoints),
        "images": images,
        "maps_pins": maps_pins,
        "maps_observation": {
            "file_backed_entries_stable": True,
            "anonymous_mapping_count_before": _counts(before_mappings)["anonymous"],
            "anonymous_mapping_count_after": _counts(after_mappings)["anonymous"],
            "pseudopath_mapping_count_before": _counts(before_mappings)["pseudopath"],
            "pseudopath_mapping_count_after": _counts(after_mappings)["pseudopath"],
            "non_file_backed_limitations": (
                "Anonymous and pseudo-path mappings are counted only; they are not "
                "file-backed image inventory or dependency-closure evidence."
            ),
        },
        "pid": os.getpid(),
        "scope": (
            "endpoint-only /proc/self/maps observation; only file-backed mappings "
            "present in both snapshots are pinned"
        ),
        "complete_runtime_dependency_closure": "NOT_ESTABLISHED",
        "transient_images": "NOT_OBSERVED",
        "resident_memory_bytes": "NOT_ATTESTED",
        "performance_admission": False,
    }
