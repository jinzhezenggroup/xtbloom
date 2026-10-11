"""Produce a trusted-local receipt for one controlled xTBloom build.

The recorder archives a clean immutable Git commit, configures one fresh
Release shared build, and records the resulting local bytes. A receipt is
execution provenance only; it is not remote attestation or scientific or
performance evidence.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks import build_receipt

RECEIPT_KIND = "xtbloom-controlled-build-receipt"
TRUST_SCOPE = "trusted-local-execution; no remote attestation"
REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
TOOL_TIMEOUT_SECONDS = 30
GIT_TIMEOUT_SECONDS = 60
ARCHIVE_TIMEOUT_SECONDS = 300
CONFIGURE_TIMEOUT_SECONDS = 1800
BUILD_TIMEOUT_SECONDS = 7200
TERMINATE_GRACE_SECONDS = 5


class RecorderError(RuntimeError):
    """An input, controlled command, or postflight check failed."""

    def __init__(self, phase: str, reason: str) -> None:
        super().__init__(reason)
        self.phase = phase


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Define the complete, non-shell controlled-build command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cpu-linalg-library", type=Path, required=True)
    parser.add_argument("--jobs", type=_positive_int, required=True)
    parser.add_argument("--cc")
    parser.add_argument("--cxx")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--cuda-architectures")
    parser.add_argument("--no-ccache", action="store_true")
    return parser


def _validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if not REVISION_PATTERN.fullmatch(args.revision):
        parser.error("--revision must be a full lowercase SHA-1 or SHA-256 commit ID")
    for name in ("source_root", "output", "cpu_linalg_library"):
        if not getattr(args, name).is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be absolute")
    if args.backend == "cuda":
        if args.nvcc is None or args.cuda_architectures is None:
            parser.error("CUDA requires both --nvcc and --cuda-architectures")
        if not args.nvcc.is_absolute():
            parser.error("--nvcc must be absolute")
        try:
            build_receipt.validate_cuda_architectures(args.cuda_architectures)
        except build_receipt.ReceiptError as exc:
            parser.error(str(exc))
    elif args.nvcc is not None or args.cuda_architectures is not None:
        parser.error("--nvcc and --cuda-architectures are only valid for CUDA")


def _output_directory(path: Path) -> Path:
    if not path.is_absolute():
        raise RecorderError("arguments", "--output must be absolute")
    if os.path.lexists(path):
        raise RecorderError("arguments", "output path already exists")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise RecorderError(
            "arguments", f"output parent is unavailable: {exc}"
        ) from exc
    if not parent.is_dir():
        raise RecorderError("arguments", "output parent must be a directory")
    return parent / path.name


def _minimal_environment() -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in build_receipt.ENVIRONMENT_KEYS
        if name in os.environ
    }
    environment.setdefault("PATH", os.defpath)
    environment["LANG"] = "C"
    environment["LC_ALL"] = "C"
    return environment


def _git_environment() -> dict[str, str]:
    environment = _minimal_environment()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: int = TOOL_TIMEOUT_SECONDS,
) -> bytes:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecorderError(
            "preflight", f"command could not complete: {argv[0]}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise RecorderError(
            "preflight",
            f"command exited {result.returncode}: {argv[0]}: {detail.strip()[:1000]}",
        )
    return result.stdout


def _resolve_executable(value: str | None, default: str, label: str) -> Path:
    requested = value or default
    if os.path.isabs(requested) or os.sep in requested:
        candidate = Path(requested)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise RecorderError(
                "preflight", f"{label} is unavailable: {requested}"
            ) from exc
    else:
        found = shutil.which(requested)
        if found is None:
            raise RecorderError("preflight", f"{label} was not found: {requested}")
        resolved = Path(found).resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RecorderError(
            "preflight", f"{label} is not an executable file: {resolved}"
        )
    return resolved


def _tool_record(
    path: Path, version_args: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    output = _capture(
        [str(path), *version_args],
        environment=environment,
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    version = output.decode("utf-8", errors="replace").strip()
    if not version:
        raise RecorderError("preflight", f"tool returned an empty version: {path}")
    return {**build_receipt.file_record(path), "version": version[:16000]}


def _resolve_tools(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    environment = _minimal_environment()
    paths = {
        "git": _resolve_executable(None, "git", "Git"),
        "cmake": _resolve_executable(None, "cmake", "CMake"),
        "ninja": _resolve_executable(None, "ninja", "Ninja"),
        "cc": _resolve_executable(args.cc, "cc", "C compiler"),
        "cxx": _resolve_executable(args.cxx, "c++", "C++ compiler"),
    }
    use_ccache = not args.no_ccache and shutil.which("ccache") is not None
    if use_ccache:
        paths["ccache"] = _resolve_executable(None, "ccache", "ccache")
    if args.backend == "cuda":
        paths["nvcc"] = _resolve_executable(str(args.nvcc), "", "NVCC")

    tools = {
        "git": _tool_record(paths["git"], ["--version"], environment),
        "cmake": _tool_record(paths["cmake"], ["--version"], environment),
        "ninja": _tool_record(paths["ninja"], ["--version"], environment),
        "cc": _tool_record(paths["cc"], ["--version"], environment),
        "cxx": _tool_record(paths["cxx"], ["--version"], environment),
    }
    if use_ccache:
        tools["ccache"] = _tool_record(paths["ccache"], ["--version"], environment)
    if args.backend == "cuda":
        tools["nvcc"] = _tool_record(paths["nvcc"], ["--version"], environment)
    return tools, environment


def _git_bytes(
    git: Path, root: Path, *arguments: str, timeout: int = GIT_TIMEOUT_SECONDS
) -> bytes:
    return _capture(
        [str(git), "-C", str(root), *arguments],
        environment=_git_environment(),
        timeout=timeout,
    )


def _git_text(git: Path, root: Path, *arguments: str) -> str:
    return _git_bytes(git, root, *arguments).decode("utf-8", errors="strict").strip()


def _source_tree_entries(
    git: Path, root: Path, revision: str
) -> tuple[list[dict[str, str]], dict[str, bytes]]:
    raw = _git_bytes(git, root, "ls-tree", "-r", "-z", "--full-tree", revision)
    entries: list[dict[str, str]] = []
    blobs: dict[str, bytes] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode_bytes, object_type, object_id_bytes = metadata.split(b" ")
            path = raw_path.decode("utf-8", errors="strict")
            mode = mode_bytes.decode("ascii")
            object_type_text = object_type.decode("ascii")
            object_id = object_id_bytes.decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise RecorderError(
                "source", "Git tree has an unrepresentable path"
            ) from exc
        if (
            object_type_text != "blob"
            or mode not in {"100644", "100755"}
            or not path
            or "\\" in path
            or PurePosixPath(path).is_absolute()
            or any(
                part in {"", ".", "..", ".git"} for part in PurePosixPath(path).parts
            )
        ):
            raise RecorderError(
                "source", f"unsupported source entry: {path!r} ({mode})"
            )
        entries.append({"path": path, "mode": mode, "git_blob_oid": object_id})
        if path == ".git_archival.txt":
            if mode != "100644":
                raise RecorderError(
                    "source", "export-subst metadata must be a regular file"
                )
            blobs[path] = _git_bytes(git, root, "cat-file", "blob", object_id)
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise RecorderError("source", "selected Git tree is empty")
    return entries, blobs


def _producer_identity(git: Path) -> dict[str, Any]:
    producer_root = Path(__file__).resolve().parents[1]
    top = Path(_git_text(git, producer_root, "rev-parse", "--show-toplevel")).resolve()
    if top != producer_root:
        raise RecorderError("producer", "recorder is not inside its Git worktree root")
    head = _git_text(git, producer_root, "rev-parse", "HEAD")
    status = _git_bytes(
        git, producer_root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if not REVISION_PATTERN.fullmatch(head) or status:
        raise RecorderError(
            "producer", "recorder and verifier must come from a clean full Git commit"
        )
    files = {}
    for key, relative in (
        ("recorder", "benchmarks/record_build.py"),
        ("verifier", "benchmarks/build_receipt.py"),
    ):
        path = producer_root / relative
        tracked = _git_bytes(
            git, producer_root, "cat-file", "blob", f"{head}:{relative}"
        )
        if not path.is_file() or path.read_bytes() != tracked:
            raise RecorderError(
                "producer",
                f"producer file differs from selected Git commit: {relative}",
            )
        files[key] = build_receipt.file_record(path)
    return {
        **files,
        "git": {"revision": head, "dirty": False},
        "trust_scope": TRUST_SCOPE,
    }


def producer_identity(git: Path) -> dict[str, Any]:
    """Bind both executing Python modules to a clean full producer commit."""
    return _producer_identity(git)


def _check_source_root(
    args: argparse.Namespace,
    git: Path,
) -> tuple[Path, str, bytes, str, list[dict[str, str]], dict[str, bytes], str]:
    try:
        root = args.source_root.resolve(strict=True)
    except OSError as exc:
        raise RecorderError("source", f"source root is unavailable: {exc}") from exc
    if not root.is_dir():
        raise RecorderError("source", "source root must be a directory")
    top = Path(_git_text(git, root, "rev-parse", "--show-toplevel")).resolve()
    if root != top:
        raise RecorderError("source", "--source-root must name the Git worktree root")
    object_format = _git_text(git, root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise RecorderError("source", f"unsupported Git object format: {object_format}")
    expected_length = 40 if object_format == "sha1" else 64
    if len(args.revision) != expected_length:
        raise RecorderError(
            "source", "revision length does not match the Git object format"
        )
    status = _git_bytes(
        git, root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    if status:
        raise RecorderError(
            "source", "source root contains tracked or untracked changes"
        )
    head = _git_text(git, root, "rev-parse", "HEAD")
    resolved = _git_text(
        git, root, "rev-parse", "--verify", f"{args.revision}^{{commit}}"
    )
    if resolved != args.revision:
        raise RecorderError(
            "source", "revision is not the selected full commit object ID"
        )
    commit = _git_bytes(git, root, "cat-file", "commit", args.revision)
    tree_line = commit.split(b"\n", 1)[0]
    if not tree_line.startswith(b"tree "):
        raise RecorderError("source", "selected commit has no canonical tree header")
    try:
        tree = tree_line[5:].decode("ascii")
    except UnicodeError as exc:
        raise RecorderError("source", "commit tree ID is malformed") from exc
    entries, special_blobs = _source_tree_entries(git, root, args.revision)
    if build_receipt.git_tree_oid(entries, object_format) != tree:
        raise RecorderError(
            "source", "Git tree listing does not reconstruct the commit tree"
        )
    return root, object_format, commit, tree, entries, special_blobs, head


def _create_archive(
    git: Path, root: Path, revision: str, output: Path, command: dict[str, Any]
) -> Path:
    """Retain literal source-capture history even when Git fails to produce a tar."""
    archive_path = output / "source.tar"
    log_path = output / "source-archive.log"
    argv = build_receipt.archive_argv(git, root, revision, archive_path)
    command.update(
        phase="archive",
        argv=argv,
        started=False,
        exit_code=None,
        timed_out=False,
        launch_error=None,
        log=None,
    )
    process = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                argv,
                env=_git_environment(),
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=(os.name == "posix"),
            )
            command["started"] = True
            try:
                process.wait(timeout=ARCHIVE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                command["timed_out"] = True
                _terminate_process(process)
            log.flush()
            os.fsync(log.fileno())
    except OSError as exc:
        command["launch_error"] = str(exc)
        raise RecorderError(
            "archive", f"could not write source archive: {exc}"
        ) from exc
    finally:
        if process is not None:
            command["exit_code"] = process.returncode
        if log_path.is_file():
            command["log"] = build_receipt.file_record(log_path)
    if command["timed_out"]:
        raise RecorderError("archive", "git archive exceeded its finite timeout")
    if process.returncode != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[:1000]
        raise RecorderError(
            "archive", f"git archive exited {process.returncode}: {detail.strip()}"
        )
    archive_path.chmod(0o444)
    return archive_path


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait()


def _archive_source(
    *,
    git: Path,
    root: Path,
    revision: str,
    object_format: str,
    commit: bytes,
    tree: str,
    git_entries: list[dict[str, str]],
    special_blobs: dict[str, bytes],
    output: Path,
    archive_command: dict[str, Any],
) -> dict[str, Any]:
    archive_path = _create_archive(git, root, revision, output, archive_command)
    snapshot = output / "source"
    try:
        archive_commit, manifest = build_receipt.archive_manifest(
            archive_path, object_format, destination=snapshot
        )
    except build_receipt.ReceiptError as exc:
        raise RecorderError("archive", str(exc)) from exc
    if archive_commit != revision:
        raise RecorderError(
            "archive", "Git archive PAX commit differs from the selected revision"
        )
    expected = {entry["path"]: entry for entry in git_entries}
    observed = {entry["path"]: entry for entry in manifest}
    if expected.keys() != observed.keys():
        raise RecorderError(
            "archive", "Git archive does not contain the complete committed tree"
        )
    original_subst = None
    for path, expected_record in expected.items():
        actual = observed[path]
        if actual["mode"] != expected_record["mode"]:
            raise RecorderError("archive", f"Git archive changed source mode: {path}")
        if path == ".git_archival.txt":
            original_bytes = special_blobs.get(path)
            if original_bytes is None:
                raise RecorderError(
                    "archive", "export-subst original blob is unavailable"
                )
            original_subst = base64.b64encode(original_bytes).decode("ascii")
            actual["git_blob_oid"] = expected_record["git_blob_oid"]
        elif actual["git_blob_oid"] != expected_record["git_blob_oid"]:
            raise RecorderError(
                "archive", f"Git archive changed committed source bytes: {path}"
            )
    manifest = sorted(manifest, key=lambda record: record["path"])
    if build_receipt.git_tree_oid(manifest, object_format) != tree:
        raise RecorderError(
            "archive", "source manifest does not reconstruct the committed tree"
        )
    _make_snapshot_read_only(snapshot)
    _assert_snapshot_matches(snapshot, manifest)
    return {
        "revision": revision,
        "tree": tree,
        "object_format": object_format,
        "commit_object_base64": base64.b64encode(commit).decode("ascii"),
        "snapshot_path": str(snapshot.resolve(strict=True)),
        "archive": build_receipt.file_record(archive_path),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(
            build_receipt.canonical_json_bytes(manifest)
        ).hexdigest(),
        "export_subst_original_base64": original_subst,
    }


def _make_snapshot_read_only(snapshot: Path) -> None:
    for current, directories, files in os.walk(
        snapshot, topdown=False, followlinks=False
    ):
        directory = Path(current)
        for name in files:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise RecorderError(
                    "archive", f"unsupported source snapshot entry: {path}"
                )
            mode = path.stat().st_mode
            path.chmod(0o555 if mode & 0o111 else 0o444)
        for name in directories:
            path = directory / name
            if path.is_symlink() or not path.is_dir():
                raise RecorderError(
                    "archive", f"unsupported source snapshot directory: {path}"
                )
            path.chmod(0o555)
        directory.chmod(0o555)


def _assert_snapshot_matches(snapshot: Path, manifest: list[dict[str, Any]]) -> None:
    expected = {entry["path"]: entry for entry in manifest}
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    observed: set[str] = set()
    for current, directories, files in os.walk(
        snapshot, topdown=True, followlinks=False
    ):
        directory = Path(current)
        if directory.stat().st_mode & 0o222:
            raise RecorderError("postflight", "source snapshot directory is writable")
        for name in directories:
            path = directory / name
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise RecorderError(
                    "postflight", "source snapshot has a writable directory or symlink"
                )
            if path.relative_to(snapshot).as_posix() not in expected_directories:
                raise RecorderError(
                    "postflight", "source snapshot has an unrecorded directory"
                )
        for name in files:
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
                raise RecorderError(
                    "postflight", "source snapshot has a writable or nonregular file"
                )
            relative = path.relative_to(snapshot).as_posix()
            record = expected.get(relative)
            if record is None:
                raise RecorderError(
                    "postflight", f"source snapshot has an unrecorded file: {relative}"
                )
            actual = build_receipt.file_record(path)
            mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
            if (
                actual["sha256"] != record["sha256"]
                or actual["size_bytes"] != record["size_bytes"]
                or mode != record["mode"]
            ):
                raise RecorderError(
                    "postflight", f"source snapshot changed: {relative}"
                )
            observed.add(relative)
    if observed != expected.keys():
        raise RecorderError("postflight", "source snapshot is missing committed files")


def _assert_original_source_unchanged(
    git: Path,
    root: Path,
    revision: str,
    initial_head: str,
    initial_commit: bytes,
    initial_entries: list[dict[str, str]],
) -> None:
    status = _git_bytes(
        git, root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    head = _git_text(git, root, "rev-parse", "HEAD")
    commit = _git_bytes(git, root, "cat-file", "commit", revision)
    entries, _ = _source_tree_entries(git, root, revision)
    if (
        status
        or head != initial_head
        or commit != initial_commit
        or entries != initial_entries
    ):
        raise RecorderError(
            "postflight", "source root or selected Git commit changed during build"
        )


def _build_environment(
    tools: dict[str, Any], initial: dict[str, str]
) -> dict[str, str]:
    del tools
    return dict(sorted(initial.items()))


def _run_logged(
    argv: list[str],
    *,
    phase: str,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout: float,
) -> tuple[int, dict[str, Any], bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        with log_path.open("xb") as log:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                log.write(f"recorder could not start command: {exc}\n".encode())
                exit_code = 127
            else:
                try:
                    exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process(process)
                    exit_code = process.returncode
            log.flush()
            os.fsync(log.fileno())
    except OSError as exc:
        raise RecorderError(phase, f"could not record {phase} output: {exc}") from exc
    log_record = build_receipt.file_record(log_path)
    return exit_code, log_record, timed_out


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_file_records(build_root: Path) -> list[dict[str, Any]]:
    root = build_root.resolve(strict=True)
    resolved_files: set[Path] = set()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        for name in list(directories):
            path = directory / name
            if path.is_symlink():
                try:
                    target = path.resolve(strict=True)
                except OSError as exc:
                    raise RecorderError(
                        "postflight", f"broken build symlink: {path}"
                    ) from exc
                if not _inside(root, target):
                    raise RecorderError(
                        "postflight", f"build symlink escapes build root: {path}"
                    )
                if not target.is_dir():
                    raise RecorderError(
                        "postflight",
                        f"build directory symlink is not a directory: {path}",
                    )
                directories.remove(name)
                continue
            if not path.is_dir():
                raise RecorderError(
                    "postflight", f"non-directory in build tree: {path}"
                )
        for name in files:
            path = directory / name
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise RecorderError(
                    "postflight", f"broken build symlink: {path}"
                ) from exc
            if not _inside(root, resolved):
                raise RecorderError(
                    "postflight", f"build file escapes build root: {path}"
                )
            if not resolved.is_file():
                raise RecorderError("postflight", f"nonregular build artifact: {path}")
            resolved_files.add(resolved)
    records = [build_receipt.file_record(path) for path in sorted(resolved_files)]
    return sorted(records, key=lambda record: record["path"])


def _find_library(manifest: list[dict[str, Any]]) -> Path:
    prefix = "libxtbloom.so."
    candidates = {
        Path(record["path"])
        for record in manifest
        if Path(record["path"]).name.startswith(prefix)
        and Path(record["path"]).is_file()
    }
    if len(candidates) != 1:
        raise RecorderError(
            "postflight", "build must contain one canonical versioned libxtbloom.so"
        )
    return next(iter(candidates))


def _validate_cmake_cache(receipt: dict[str, Any]) -> None:
    cache_path = Path(receipt["build"]["root"]) / "CMakeCache.txt"
    observed: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        if ":" not in declaration:
            continue
        key = declaration.rsplit(":", 1)[0]
        if key in observed:
            raise RecorderError("postflight", f"duplicate CMake cache entry: {key}")
        observed[key] = value
    expected = {
        "CMAKE_HOME_DIRECTORY": receipt["source"]["snapshot_path"],
        "CMAKE_GENERATOR": "Ninja",
    }
    for argument in build_receipt.configure_argv(receipt):
        if argument.startswith("-D") and "=" in argument:
            key, value = argument[2:].split("=", 1)
            expected[key] = value
    mismatches = [
        f"{key}={observed.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    if mismatches:
        raise RecorderError(
            "postflight",
            "CMake cache differs from configure argv: " + "; ".join(mismatches),
        )


def _verify_tool_postflight(tools: dict[str, Any], environment: dict[str, str]) -> None:
    version_arguments = {
        "git": ["--version"],
        "cmake": ["--version"],
        "ninja": ["--version"],
        "cc": ["--version"],
        "cxx": ["--version"],
        "ccache": ["--version"],
        "nvcc": ["--version"],
    }
    for name, original in tools.items():
        current = _tool_record(
            Path(original["path"]), version_arguments[name], environment
        )
        if current != original:
            raise RecorderError(
                "postflight", f"build tool changed during execution: {name}"
            )


def _atomic_receipt(path: Path, receipt: dict[str, Any]) -> tuple[bytes, str]:
    contents = build_receipt.canonical_json_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".receipt.json.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise
    return contents, hashlib.sha256(contents).hexdigest()


def _execute(args: argparse.Namespace, output: Path) -> tuple[dict[str, Any], str]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "status": "failed",
        "performance_claim_eligible": False,
        "failure": {"phase": "initialization", "reason": "recorder did not complete"},
        "source_capture": None,
    }
    initial_phase = "preflight"
    failures: list[tuple[str, str]] = []
    source_context: dict[str, Any] = {}
    tools: dict[str, Any] = {}
    environment: dict[str, str] = {}
    build_root = output / "build"
    receipt_path = output / "receipt.json"

    try:
        tools, initial_environment = _resolve_tools(args)
        git = Path(tools["git"]["path"])
        producer = producer_identity(git)
        receipt["producer"] = producer
        receipt["tools"] = tools

        provider_path = args.cpu_linalg_library.resolve(strict=True)
        if not provider_path.is_file():
            raise RecorderError(
                "preflight", "CPU linear algebra provider must be a regular file"
            )
        provider = build_receipt.file_record(provider_path)
        root, object_format, commit, tree, entries, special_blobs, head = (
            _check_source_root(args, git)
        )
        receipt["options"] = {
            "backend": args.backend,
            "build_type": "Release",
            "tests": False,
            "jobs": args.jobs,
            "cuda_architectures": args.cuda_architectures,
            "ccache": "ccache" in tools,
        }
        receipt["provider"] = provider
        environment = _build_environment(tools, initial_environment)
        receipt["environment"] = environment
        receipt["build"] = {"root": str(build_root), "commands": []}

        initial_phase = "archive"
        receipt["source_capture"] = {"source_root": str(root), "command": {}}
        source = _archive_source(
            git=git,
            root=root,
            revision=args.revision,
            object_format=object_format,
            commit=commit,
            tree=tree,
            git_entries=entries,
            special_blobs=special_blobs,
            output=output,
            archive_command=receipt["source_capture"]["command"],
        )
        receipt["source"] = source
        source_context = {
            "git": git,
            "root": root,
            "revision": args.revision,
            "head": head,
            "commit": commit,
            "entries": entries,
            "manifest": source["manifest"],
            "archive": source["archive"],
        }
        build_root.mkdir(mode=0o700)

        initial_phase = "configure"
        configure_argv = build_receipt.configure_argv(receipt)
        exit_code, log, timed_out = _run_logged(
            configure_argv,
            phase="configure",
            cwd=output,
            environment=environment,
            log_path=output / "logs" / "configure.log",
            timeout=CONFIGURE_TIMEOUT_SECONDS,
        )
        receipt["build"]["commands"].append(
            {
                "phase": "configure",
                "argv": configure_argv,
                "exit_code": exit_code,
                "log": log,
            }
        )
        if timed_out:
            raise RecorderError(
                "configure", f"configure timed out with exit code {exit_code}"
            )
        if exit_code != 0:
            raise RecorderError(
                "configure", f"configure exited with status {exit_code}"
            )

        initial_phase = "build"
        build_argv = build_receipt.build_argv(receipt)
        exit_code, log, timed_out = _run_logged(
            build_argv,
            phase="build",
            cwd=output,
            environment=environment,
            log_path=output / "logs" / "build.log",
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        receipt["build"]["commands"].append(
            {"phase": "build", "argv": build_argv, "exit_code": exit_code, "log": log}
        )
        if timed_out:
            raise RecorderError("build", f"build timed out with exit code {exit_code}")
        if exit_code != 0:
            raise RecorderError("build", f"build exited with status {exit_code}")
    except RecorderError as exc:
        failures.append((exc.phase, str(exc)))
    except Exception as exc:  # noqa: BLE001 - preserve all build failures in the receipt.
        failures.append((initial_phase, f"{type(exc).__name__}: {exc}"))

    if source_context:
        try:
            initial_phase = "postflight"
            _assert_original_source_unchanged(
                source_context["git"],
                source_context["root"],
                source_context["revision"],
                source_context["head"],
                source_context["commit"],
                source_context["entries"],
            )
            _assert_snapshot_matches(output / "source", source_context["manifest"])
            current_archive = build_receipt.file_record(output / "source.tar")
            if current_archive != source_context["archive"]:
                raise RecorderError(
                    "postflight", "source archive changed during the build"
                )
            _verify_tool_postflight(tools, environment)
            current_provider = build_receipt.file_record(
                Path(receipt["provider"]["path"])
            )
            if current_provider != receipt["provider"]:
                raise RecorderError(
                    "postflight", "selected CPU linear algebra provider changed"
                )
            current_producer = producer_identity(Path(tools["git"]["path"]))
            if current_producer != receipt["producer"]:
                raise RecorderError(
                    "postflight", "producer identity changed during the build"
                )
        except RecorderError as exc:
            failures.append((exc.phase, str(exc)))
        except Exception as exc:  # noqa: BLE001 - postflight errors must fail the receipt.
            failures.append(("postflight", f"{type(exc).__name__}: {exc}"))

    if build_root.is_dir() and not build_root.is_symlink():
        try:
            manifest = _build_file_records(build_root)
            receipt.setdefault("build", {"root": str(build_root), "commands": []})
            receipt["build"]["manifest"] = manifest
            cache_path = build_root / "CMakeCache.txt"
            if cache_path.is_file() and not cache_path.is_symlink():
                receipt["build"]["cache"] = build_receipt.file_record(cache_path)
            if not failures:
                for required_name in (
                    "CMakeCache.txt",
                    "build.ninja",
                    "compile_commands.json",
                ):
                    if not (build_root / required_name).is_file():
                        raise RecorderError(
                            "postflight",
                            f"required build file is missing: {required_name}",
                        )
                _validate_cmake_cache(receipt)
                library = _find_library(manifest)
                library_record = build_receipt.file_record(library)
                matching_records = [
                    item for item in manifest if item["path"] == library_record["path"]
                ]
                if len(matching_records) != 1 or matching_records[0] != library_record:
                    raise RecorderError(
                        "postflight", "produced library differs from its build manifest"
                    )
                receipt["artifacts"] = {"library": library_record}
                for item in manifest:
                    current = build_receipt.file_record(Path(item["path"]))
                    if current != item:
                        raise RecorderError(
                            "postflight",
                            "build output changed while recording the manifest",
                        )
        except RecorderError as exc:
            failures.append((exc.phase, str(exc)))
        except Exception as exc:  # noqa: BLE001 - incomplete manifests cannot be trusted.
            failures.append(("postflight", f"{type(exc).__name__}: {exc}"))

    if failures:
        phase = failures[0][0]
        reason = "; ".join(
            message if index == 0 else f"{later_phase}: {message}"
            for index, (later_phase, message) in enumerate(failures)
        )
        receipt["status"] = "failed"
        receipt["failure"] = {"phase": phase, "reason": reason}
    else:
        receipt["status"] = "complete"
        receipt["failure"] = None

        candidate = output / ".receipt-candidate.json"
        try:
            _, candidate_digest = _atomic_receipt(candidate, receipt)
            build_receipt.verify_receipt(
                candidate,
                candidate_digest,
                Path(receipt["artifacts"]["library"]["path"]),
                args.revision,
            )
        except Exception as exc:  # noqa: BLE001 - never publish an unchecked success.
            receipt["status"] = "failed"
            receipt["failure"] = {
                "phase": "consumer-check",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            with contextlib.suppress(FileNotFoundError):
                candidate.unlink()

    _, digest = _atomic_receipt(receipt_path, receipt)
    return receipt, digest


def main(argv: list[str] | None = None) -> int:
    """Run the controlled build and print one canonical JSON result."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args, parser)
    try:
        output = _output_directory(args.output)
        output.mkdir(mode=0o700)
    except RecorderError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"could not create fresh output directory: {exc}\n")
        return 2

    receipt_path = output / "receipt.json"
    initial = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "status": "failed",
        "performance_claim_eligible": False,
        "failure": {"phase": "initialization", "reason": "recorder did not complete"},
    }
    try:
        _atomic_receipt(receipt_path, initial)
        receipt, digest = _execute(args, output)
    except Exception as exc:  # noqa: BLE001 - keep CLI failures machine-readable.
        if not receipt_path.exists():
            sys.stderr.write(
                f"could not write receipt bundle: {type(exc).__name__}: {exc}\n"
            )
            return 2
        receipt = initial
        receipt["failure"] = {
            "phase": "bundle",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        try:
            _, digest = _atomic_receipt(receipt_path, receipt)
        except Exception as write_exc:  # noqa: BLE001 - surface failed publication.
            sys.stderr.write(f"could not finalize failed receipt: {write_exc}\n")
            return 2

    result = {
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "receipt_sha256": digest,
    }
    if receipt["status"] == "failed":
        result["failure"] = receipt["failure"]
    sys.stdout.write(build_receipt.canonical_json_bytes(result).decode("utf-8"))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
