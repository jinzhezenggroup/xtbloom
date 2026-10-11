"""Read-only integrity checks for locally controlled xTBloom build receipts.

These receipts trust the reviewed recorder and its local execution environment.
They are neither signed remote attestations nor scientific/performance gates.
The selected commit, complete archive, source tree and output bytes are bound
independently; current checkout or adjacent CMake state is never a substitute.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
import sys
import tarfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
RECEIPT_KIND = "xtbloom-controlled-build-receipt"
TRUST_SCOPE = "trusted-local-execution; no remote attestation"
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CUDA_ARCHITECTURES_PATTERN = re.compile(
    r"[0-9]+(?:-real|-virtual)?(?:;[0-9]+(?:-real|-virtual)?)*\Z"
)
ENVIRONMENT_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CCACHE_DIR",
        "CCACHE_BASEDIR",
        "CCACHE_COMPILERCHECK",
        "CCACHE_SLOPPINESS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)


class ReceiptError(RuntimeError):
    """The requested receipt is incomplete, inconsistent or no longer pinned."""


def canonical_json_bytes(value: object) -> bytes:
    """Give the external receipt pin one unambiguous, finite serialization."""
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptError(f"noncanonical receipt value: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Hash bytes in bounded chunks without executing an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Record a canonical regular file, rejecting mutation during the read."""
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ReceiptError(f"not a regular file: {resolved}")
    digest = sha256_file(resolved)
    after = resolved.stat()
    identities = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identities):
        raise ReceiptError(f"file changed while hashing: {resolved}")
    return {"path": str(resolved), "sha256": digest, "size_bytes": after.st_size}


def git_object_oid(kind: str, data: bytes, object_format: str) -> str:
    """Hash the actual Git object framing, not a JSON description of it."""
    if object_format not in {"sha1", "sha256"}:
        raise ReceiptError("unsupported Git object format")
    digest = hashlib.new(object_format)
    digest.update(f"{kind} {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ReceiptError("invalid source-relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {".", "..", ".git"} for part in parsed.parts)
    ):
        raise ReceiptError(f"unsafe source-relative path: {value!r}")
    return value


def git_tree_oid(manifest: list[dict[str, Any]], object_format: str) -> str:
    """Reconstruct hierarchical Git trees, including Git's directory ordering."""
    root: dict[str, Any] = {}
    for record in manifest:
        parts = PurePosixPath(_relative_path(record.get("path"))).parts
        current = root
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            if not isinstance(child, dict):
                raise ReceiptError("source paths collide with a file")
            current = child
        if parts[-1] in current:
            raise ReceiptError("duplicate or colliding source path")
        mode = record.get("mode")
        object_id = record.get("git_blob_oid")
        expected_length = 40 if object_format == "sha1" else 64
        if (
            mode not in {"100644", "100755"}
            or not isinstance(object_id, str)
            or len(object_id) != expected_length
            or not REVISION_PATTERN.fullmatch(object_id)
        ):
            raise ReceiptError("invalid source Git mode or blob identity")
        current[parts[-1]] = (mode, object_id)

    def encode_tree(entries: dict[str, Any]) -> str:
        encoded = bytearray()
        names = sorted(
            entries,
            key=lambda name: (
                name.encode("utf-8")
                + (b"/" if isinstance(entries[name], dict) else b"")
            ),
        )
        for name in names:
            entry = entries[name]
            if isinstance(entry, dict):
                mode, object_id = "40000", encode_tree(entry)
            else:
                mode, object_id = entry
            encoded.extend(mode.encode("ascii") + b" " + name.encode("utf-8") + b"\0")
            encoded.extend(bytes.fromhex(object_id))
        return git_object_oid("tree", bytes(encoded), object_format)

    return encode_tree(root)


def archive_manifest(
    archive_path: Path, object_format: str, destination: Path | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Inspect or safely unpack a Git archive; links and traversal are rejected.

    Only regular files/directories are supported. The complete payload is
    checked, not only files that happened to compile in this configuration.
    Extraction never delegates untrusted paths to tar's filesystem routines.
    """
    if object_format not in {"sha1", "sha256"}:
        raise ReceiptError("unsupported Git object format")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ReceiptError("source archive exceeds the bounded reader limit")
    if destination is not None:
        if destination.is_symlink() or (
            destination.exists() and any(destination.iterdir())
        ):
            raise ReceiptError("archive destination must be fresh and empty")
        destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen: set[str] = set()
    total_bytes = 0
    with tarfile.open(archive_path, "r:") as archive:
        commit_id = archive.pax_headers.get("comment", "").strip()
        if not REVISION_PATTERN.fullmatch(commit_id):
            raise ReceiptError("archive lacks its full Git commit identity")
        for member in archive:
            name = member.name.rstrip("/") if member.isdir() else member.name
            relative = _relative_path(name)
            if relative in seen:
                raise ReceiptError("duplicate archive member")
            seen.add(relative)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0:
                raise ReceiptError("source archive links/special files are unsupported")
            total_bytes += member.size
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise ReceiptError("expanded archive exceeds the bounded reader limit")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ReceiptError("archive member bytes are missing")
            digest = hashlib.sha256()
            blob_digest = hashlib.new(object_format)
            blob_digest.update(f"blob {member.size}\0".encode("ascii"))
            observed_bytes = 0
            with ExitStack() as cleanup:
                cleanup.enter_context(extracted)
                output = None
                if destination is not None:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = cleanup.enter_context(target.open("xb"))
                while chunk := extracted.read(1024 * 1024):
                    observed_bytes += len(chunk)
                    digest.update(chunk)
                    blob_digest.update(chunk)
                    if output is not None:
                        output.write(chunk)
            if observed_bytes != member.size:
                raise ReceiptError("truncated archive member")
            mode = "100755" if member.mode & stat.S_IXUSR else "100644"
            manifest.append(
                {
                    "path": relative,
                    "mode": mode,
                    "sha256": digest.hexdigest(),
                    "size_bytes": observed_bytes,
                    "git_blob_oid": blob_digest.hexdigest(),
                }
            )
            if destination is not None:
                target.chmod(0o755 if mode == "100755" else 0o644)
    return commit_id, sorted(manifest, key=lambda record: record["path"])


def configure_argv(receipt: dict[str, Any]) -> list[str]:
    """Reconstruct the recorder's fixed configure operation without a shell."""
    tools = receipt["tools"]
    options = receipt["options"]
    arguments = [
        tools["cmake"]["path"],
        "-S",
        receipt["source"]["snapshot_path"],
        "-B",
        receipt["build"]["root"],
        "-G",
        "Ninja",
        "-DCMAKE_MAKE_PROGRAM=" + tools["ninja"]["path"],
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DXTBLOOM_BUILD_TESTS=OFF",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DXTBLOOM_ENABLE_CUDA=" + ("ON" if options["backend"] == "cuda" else "OFF"),
        "-DCMAKE_C_COMPILER=" + tools["cc"]["path"],
        "-DCMAKE_CXX_COMPILER=" + tools["cxx"]["path"],
        "-DXTBLOOM_CPU_LINALG_LIBRARY=" + receipt["provider"]["path"],
    ]
    if options["ccache"]:
        arguments.extend(
            (f"-DCMAKE_{language}_COMPILER_LAUNCHER=" + tools["ccache"]["path"])
            for language in ("C", "CXX")
        )
    if options["backend"] == "cuda":
        arguments.extend(
            [
                "-DCMAKE_CUDA_COMPILER=" + tools["nvcc"]["path"],
                "-DCMAKE_CUDA_HOST_COMPILER=" + tools["cxx"]["path"],
                "-DCMAKE_CUDA_ARCHITECTURES=" + options["cuda_architectures"],
            ]
        )
        if options["ccache"]:
            arguments.append(
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=" + tools["ccache"]["path"]
            )
    return arguments


def validate_cuda_architectures(value: object) -> None:
    """Keep recorder and consumer architecture semantics identical.

    Real/virtual variants of one architecture are distinct requests. Numeric
    aliases such as 90 and 090 are duplicates when their suffixes also match.
    """
    if not isinstance(value, str) or not CUDA_ARCHITECTURES_PATTERN.fullmatch(value):
        raise ReceiptError("CUDA requires explicit numeric architectures")
    try:
        architectures = [
            (int(item.partition("-")[0]), item.partition("-")[2])
            for item in value.split(";")
        ]
    except ValueError as exc:
        raise ReceiptError("CUDA architecture number is unsupported") from exc
    if any(number <= 0 for number, _ in architectures) or len(
        set(architectures)
    ) != len(architectures):
        raise ReceiptError("CUDA architectures must be positive and unique")


def archive_argv(
    git: Path, source_root: Path, revision: str, archive_path: Path
) -> list[str]:
    """Describe the source-capture command independently of CMake execution."""
    return [
        str(git),
        "-C",
        str(source_root),
        "archive",
        "--format=tar",
        "--output",
        str(archive_path),
        revision,
    ]


def build_argv(receipt: dict[str, Any]) -> list[str]:
    """Build only the library and its declared dependencies, never GPU tests."""
    return [
        receipt["tools"]["cmake"]["path"],
        "--build",
        receipt["build"]["root"],
        "--target",
        "xtbloom",
        "--parallel",
        str(receipt["options"]["jobs"]),
    ]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReceiptError(f"nonfinite JSON value: {value}")


def _verify_file(record: object, label: str, root: Path | None = None) -> Path:
    if not isinstance(record, dict):
        raise ReceiptError(f"missing {label} file record")
    path_text = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if (
        not isinstance(path_text, str)
        or not Path(path_text).is_absolute()
        or not isinstance(digest, str)
        or not SHA256_PATTERN.fullmatch(digest)
        or type(size) is not int
        or size < 0
    ):
        raise ReceiptError(f"invalid {label} file identity")
    path = Path(path_text)
    actual = file_record(path)
    if actual["path"] != path_text:
        raise ReceiptError(f"noncanonical {label} path")
    if root is not None and not path.is_relative_to(root):
        raise ReceiptError(f"{label} escapes its receipt bundle")
    if any(actual[key] != record[key] for key in ("sha256", "size_bytes")):
        raise ReceiptError(f"{label} bytes changed")
    return path


def _verify_source(source: dict[str, Any], bundle: Path) -> None:
    revision = source["revision"]
    object_format = source["object_format"]
    if object_format not in {"sha1", "sha256"}:
        raise ReceiptError("unsupported Git object format")
    commit_bytes = base64.b64decode(source["commit_object_base64"], validate=True)
    if git_object_oid("commit", commit_bytes, object_format) != revision:
        raise ReceiptError("source commit object does not match revision")
    if commit_bytes.split(b"\n", 1)[0] != b"tree " + source["tree"].encode("ascii"):
        raise ReceiptError("source commit does not bind the claimed Git tree")
    manifest = source["manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ReceiptError("source manifest must be complete and nonempty")
    if git_tree_oid(manifest, object_format) != source["tree"]:
        raise ReceiptError("source manifest does not match the immutable Git tree")
    if (
        hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        != source["manifest_sha256"]
    ):
        raise ReceiptError("source manifest digest changed")
    archive = _verify_file(source["archive"], "source archive", bundle)
    archive_commit, archived = archive_manifest(archive, object_format)
    if archive_commit != revision:
        raise ReceiptError("archive and source commit differ")
    original = source["export_subst_original_base64"]
    if original is not None:
        original_bytes = base64.b64decode(original, validate=True)
        for record in archived:
            if record["path"] == ".git_archival.txt":
                record["git_blob_oid"] = git_object_oid(
                    "blob", original_bytes, object_format
                )
                break
        else:
            raise ReceiptError("export-subst metadata is absent from archive")
    if archived != manifest:
        raise ReceiptError(
            "archive bytes or paths do not match the complete Git manifest"
        )
    snapshot = Path(source["snapshot_path"])
    if snapshot != bundle / "source" or not snapshot.is_dir() or snapshot.is_symlink():
        raise ReceiptError("source snapshot is not the private bundle directory")
    expected = {record["path"]: record for record in manifest}
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    observed = set()
    for path in [snapshot, *snapshot.rglob("*")]:
        if path.is_symlink():
            raise ReceiptError("source snapshot contains a symlink")
        if path.stat().st_mode & 0o222:
            raise ReceiptError("source snapshot is writable")
        if path.is_dir():
            if (
                path != snapshot
                and path.relative_to(snapshot).as_posix() not in expected_directories
            ):
                raise ReceiptError("source snapshot has an unrecorded directory")
            continue
        relative = path.relative_to(snapshot).as_posix()
        if relative not in expected:
            raise ReceiptError("source snapshot has an unrecorded file")
        observed.add(relative)
        record = expected[relative]
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ReceiptError("invalid source file extent")
        actual = file_record(path)
        mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        if (
            actual["sha256"] != record["sha256"]
            or actual["size_bytes"] != record["size_bytes"]
            or mode != record["mode"]
        ):
            raise ReceiptError(f"source snapshot bytes/mode changed: {relative}")
    if observed != expected.keys():
        raise ReceiptError("source snapshot lost a recorded file")


def _verify_source_capture(receipt: dict[str, Any], bundle: Path) -> None:
    capture = receipt["source_capture"]
    source_root = Path(capture["source_root"])
    if not source_root.is_absolute() or ".." in source_root.parts:
        raise ReceiptError("invalid source-capture checkout path")
    command = capture["command"]
    expected = archive_argv(
        Path(receipt["tools"]["git"]["path"]),
        source_root,
        receipt["source"]["revision"],
        bundle / "source.tar",
    )
    if (
        command["phase"] != "archive"
        or command["argv"] != expected
        or command["started"] is not True
        or type(command["exit_code"]) is not int
        or command["exit_code"] != 0
        or command["timed_out"] is not False
        or command["launch_error"] is not None
        or receipt["source"]["archive"]["path"] != str(bundle / "source.tar")
    ):
        raise ReceiptError("source-capture command or exit status differs")
    log = _verify_file(command["log"], "archive log", bundle)
    if log != bundle / "source-archive.log":
        raise ReceiptError("archive log is not the retained source-capture log")


def _verify_build(receipt: dict[str, Any], bundle: Path) -> Path:
    environment = receipt["environment"]
    if not isinstance(environment, dict) or any(
        key not in ENVIRONMENT_KEYS or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ReceiptError("unsupported or malformed build environment")
    options = receipt["options"]
    if (
        options["backend"] not in {"cpu", "cuda"}
        or options["build_type"] != "Release"
        or options["tests"] is not False
        or type(options["jobs"]) is not int
        or options["jobs"] <= 0
        or type(options["ccache"]) is not bool
    ):
        raise ReceiptError("invalid controlled build options")
    architectures = options["cuda_architectures"]
    if options["backend"] == "cuda":
        validate_cuda_architectures(architectures)
    elif architectures is not None or "nvcc" in receipt["tools"]:
        raise ReceiptError("CPU receipt contains a CUDA configuration")
    for name in ("git", "cmake", "ninja", "cc", "cxx"):
        tool = receipt["tools"][name]
        _verify_file(tool, f"{name} tool")
        if not isinstance(tool.get("version"), str) or not tool["version"].strip():
            raise ReceiptError(f"missing {name} tool version")
    for name in ("ccache", "nvcc"):
        required = (
            options["ccache"] if name == "ccache" else options["backend"] == "cuda"
        )
        if required:
            tool = receipt["tools"][name]
            _verify_file(tool, f"{name} tool")
            if not isinstance(tool.get("version"), str) or not tool["version"].strip():
                raise ReceiptError(f"missing {name} tool version")
        elif name in receipt["tools"]:
            raise ReceiptError(f"unexpected {name} tool")
    _verify_file(receipt["provider"], "selected LP64 provider")
    build = receipt["build"]
    root = Path(build["root"])
    if root != bundle / "build" or not root.is_dir() or root.is_symlink():
        raise ReceiptError("build directory is outside the private bundle")
    commands = build["commands"]
    expected_commands = [
        ("configure", configure_argv(receipt)),
        ("build", build_argv(receipt)),
    ]
    if not isinstance(commands, list) or len(commands) != len(expected_commands):
        raise ReceiptError("controlled build command history is incomplete")
    for command, (phase, arguments) in zip(commands, expected_commands, strict=True):
        if (
            command["phase"] != phase
            or command["argv"] != arguments
            or type(command["exit_code"]) is not int
            or command["exit_code"] != 0
        ):
            raise ReceiptError("controlled build command or exit status differs")
        _verify_file(command["log"], f"{phase} log", bundle)
    cache_path = _verify_file(build["cache"], "CMake cache", root)
    cache = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if (
            line
            and not line.startswith(("#", "//"))
            and "=" in line
            and ":" in line.split("=", 1)[0]
        ):
            declaration, value = line.split("=", 1)
            key = declaration.rsplit(":", 1)[0]
            if key in cache:
                raise ReceiptError("duplicate CMake cache assignment")
            cache[key] = value
    expected_cache = {
        "CMAKE_HOME_DIRECTORY": receipt["source"]["snapshot_path"],
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "ON",
        "XTBLOOM_BUILD_TESTS": "OFF",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "XTBLOOM_ENABLE_CUDA": "ON" if options["backend"] == "cuda" else "OFF",
        "CMAKE_C_COMPILER": receipt["tools"]["cc"]["path"],
        "CMAKE_CXX_COMPILER": receipt["tools"]["cxx"]["path"],
        "CMAKE_MAKE_PROGRAM": receipt["tools"]["ninja"]["path"],
        "XTBLOOM_CPU_LINALG_LIBRARY": receipt["provider"]["path"],
    }
    if options["ccache"]:
        for language in ("C", "CXX"):
            expected_cache[f"CMAKE_{language}_COMPILER_LAUNCHER"] = receipt["tools"][
                "ccache"
            ]["path"]
    if options["backend"] == "cuda":
        expected_cache["CMAKE_CUDA_COMPILER"] = receipt["tools"]["nvcc"]["path"]
        expected_cache["CMAKE_CUDA_HOST_COMPILER"] = receipt["tools"]["cxx"]["path"]
        expected_cache["CMAKE_CUDA_ARCHITECTURES"] = architectures
        if options["ccache"]:
            expected_cache["CMAKE_CUDA_COMPILER_LAUNCHER"] = receipt["tools"]["ccache"][
                "path"
            ]
    if any(cache.get(key) != value for key, value in expected_cache.items()):
        raise ReceiptError(
            "actual CMake configuration differs from the recorded operation"
        )
    manifest = build["manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ReceiptError("missing generated/build artifact manifest")
    paths = set()
    for record in manifest:
        path = _verify_file(record, "generated/build artifact", root)
        if path in paths:
            raise ReceiptError("duplicate build artifact identity")
        paths.add(path)
    for required in (
        root / "CMakeCache.txt",
        root / "build.ninja",
        root / "compile_commands.json",
    ):
        if required not in paths:
            raise ReceiptError("required build configuration artifact is missing")
    library = _verify_file(receipt["artifacts"]["library"], "produced library", root)
    if library not in paths:
        raise ReceiptError("produced library is absent from the build manifest")
    return library


def verify_receipt(
    receipt_path: Path,
    expected_sha256: str,
    library_path: Path,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    """Verify retained producer artifacts and exact selected bytes, read-only.

    The independently supplied receipt pin must not be derived from untrusted
    input in this function. A successful local check still leaves performance,
    runtime, transitive dependency closure and independent science unqualified.
    """
    try:
        if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
            expected_sha256
        ):
            raise ReceiptError("expected receipt SHA-256 is required")
        receipt_file = file_record(receipt_path)
        if receipt_file["sha256"] != expected_sha256:
            raise ReceiptError("receipt digest does not match its external pin")
        if receipt_file["size_bytes"] > MAX_RECEIPT_BYTES:
            raise ReceiptError("receipt exceeds the bounded reader limit")
        raw = receipt_path.read_bytes()
        document = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
            raise ReceiptError("receipt serialization is not canonical")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != SCHEMA_VERSION
        ):
            raise ReceiptError("unsupported receipt schema")
        if (
            document["kind"] != RECEIPT_KIND
            or document["status"] != "complete"
            or document["failure"] is not None
        ):
            raise ReceiptError("receipt is not a complete successful controlled build")
        if document["performance_claim_eligible"] is not False:
            raise ReceiptError("a build receipt cannot admit a performance claim")
        producer = document["producer"]
        if (
            producer["trust_scope"] != TRUST_SCOPE
            or producer["git"]["dirty"] is not False
            or not REVISION_PATTERN.fullmatch(producer["git"]["revision"])
        ):
            raise ReceiptError("producer identity/trust boundary is incomplete")
        for name, module in (
            ("recorder", "record_build.py"),
            ("verifier", "build_receipt.py"),
        ):
            _verify_file(producer[name], f"producer {name}")
            if producer[name]["sha256"] != sha256_file(
                Path(__file__).with_name(module)
            ):
                raise ReceiptError(f"unknown producer {name} implementation")
        source = document["source"]
        if not REVISION_PATTERN.fullmatch(source["revision"]):
            raise ReceiptError("source requires its full immutable revision")
        if (
            expected_source_revision is not None
            and source["revision"] != expected_source_revision
        ):
            raise ReceiptError("source revision differs from the requested producer")
        bundle = receipt_path.resolve(strict=True).parent
        _verify_source(source, bundle)
        _verify_source_capture(document, bundle)
        produced_library = _verify_build(document, bundle)
        selected = file_record(library_path)
        produced = file_record(produced_library)
        if any(selected[key] != produced[key] for key in ("sha256", "size_bytes")):
            raise ReceiptError("selected library is not the controlled build output")
        if file_record(receipt_path) != receipt_file:
            raise ReceiptError("receipt changed during verification")
        return {
            "status": "LOCAL_RECEIPT_MATCHES",
            "receipt": receipt_file,
            "source_revision": source["revision"],
            "source_tree": source["tree"],
            "source_manifest_sha256": source["manifest_sha256"],
            "producer_revision": producer["git"]["revision"],
            "library": selected,
            "backend": document["options"]["backend"],
            "trust_scope": TRUST_SCOPE,
            "performance_claim_eligible": False,
            "not_qualified": [
                "remote attestation",
                "transitive build/runtime dependency closure",
                "live backend execution",
                "independent scientific correctness",
                "performance or adoption",
            ],
        }
    except ReceiptError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        tarfile.TarError,
        RecursionError,
    ) as exc:
        raise ReceiptError(f"invalid controlled build receipt: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """Expose an offline verifier that never loads or executes the library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(
            args.receipt,
            args.expected_sha256,
            args.library,
            args.expected_source_revision,
        )
    except ReceiptError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 - explicit verifier diagnostics
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))  # noqa: T201 - verifier result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
