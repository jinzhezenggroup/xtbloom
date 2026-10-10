"""Fake-tool tests for controlled build receipts; no native code is executed."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import build_receipt, record_build

FAKE_CMAKE = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
root = Path(__file__).resolve().parent
if arguments == ["--version"]:
    print("cmake version fake-1")
    raise SystemExit(0)
if arguments and arguments[0] == "--build":
    build = Path(arguments[1])
    if (root / "omit-library").exists():
        raise SystemExit(0)
    library = build / "libxtbloom.so.1.2.3"
    library.write_bytes(b"fake shared library")
    (build / "libxtbloom.so.1").symlink_to(library.name)
    (build / "libxtbloom.so").symlink_to("libxtbloom.so.1")
    objects = build / "CMakeFiles" / "xtbloom.dir"
    objects.mkdir(parents=True, exist_ok=True)
    (objects / "fake.o").write_bytes(b"fake object")
    if (root / "escape-build-link").exists():
        outside = Path((root / "escape-build-link").read_text())
        (build / "outside-link").symlink_to(outside)
    raise SystemExit(29 if (root / "fail-build").exists() else 0)

source = Path(arguments[arguments.index("-S") + 1])
build = Path(arguments[arguments.index("-B") + 1])
build.mkdir(parents=True, exist_ok=True)
options = {}
for argument in arguments:
    if argument.startswith("-D") and "=" in argument:
        key, value = argument[2:].split("=", 1)
        options[key] = value
options["CMAKE_HOME_DIRECTORY"] = str(source)
options["CMAKE_GENERATOR"] = "Ninja"
defaults = {
    "CMAKE_BUILD_TYPE": "Release",
    "BUILD_SHARED_LIBS": "ON",
    "XTBLOOM_BUILD_TESTS": "OFF",
    "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
}
defaults.update(options)
if (root / "wrong-cache").exists():
    options["XTBLOOM_BUILD_TESTS"] = "ON"
    defaults["XTBLOOM_BUILD_TESTS"] = "ON"
(build / "CMakeCache.txt").write_text(
    "".join(f"{key}:STRING={value}\n" for key, value in sorted(defaults.items()))
)
(build / "build.ninja").write_text("# fake generated build graph\n")
(build / "compile_commands.json").write_text("[]\n")
if (root / "mutate-source").exists():
    source_root = Path((root / "mutate-source").read_text())
    (source_root / "tracked.txt").write_text("mutated source root\n")
if (root / "mutate-snapshot").exists():
    tracked = source / "tracked.txt"
    tracked.chmod(0o644)
    tracked.write_text("mutated source snapshot\n")
if (root / "change-cc-version").exists():
    (root / "change-cc").touch()
raise SystemExit(23 if (root / "fail-configure").exists() else 0)
"""


FAKE_TOOL = r"""#!/usr/bin/env python3
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
if name == "cc" and (Path(__file__).resolve().parent / "change-cc").exists():
    print("cc compiler fake-2")
elif name == "nvcc":
    print("Cuda compilation tools, release fake-1")
else:
    print(f"{name} tool fake-1")
"""


def _pin_fixture_tree(root: Path) -> str:
    """Create inert fixture objects without authoring project Git history."""
    tree = subprocess.check_output(
        ["git", "-C", str(root), "write-tree"], text=True
    ).strip()
    contents = (
        f"tree {tree}\n"
        "author Synthetic Fixture <fixture@example.invalid> 0 +0000\n"
        "committer Synthetic Fixture <fixture@example.invalid> 0 +0000\n\n"
        "Synthetic test input object; not a source-work commit.\n"
    ).encode()
    result = subprocess.run(
        ["git", "-C", str(root), "hash-object", "-w", "-t", "commit", "--stdin"],
        input=contents,
        check=True,
        capture_output=True,
    )
    revision = result.stdout.decode().strip()
    (root / ".git" / "HEAD").write_text(revision + "\n")
    return revision


class RecordBuildFakeToolTests(unittest.TestCase):
    """Exercise the CLI and verifier against temporary Git data and fake tools."""

    def setUp(self) -> None:
        """Create isolated Git data and executable fake build tools."""
        self.temporary = tempfile.TemporaryDirectory(prefix="xtbloom-record-build-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source-root"
        self.source.mkdir()
        self._git("init", "-q")
        (self.source / ".gitattributes").write_text(".git_archival.txt export-subst\n")
        (self.source / ".git_archival.txt").write_text(
            "node: $Format:%H$\ndate: $Format:%cI$\n"
        )
        (self.source / "tracked.txt").write_text("committed source\n")
        nested = self.source / "src" / "nested"
        nested.mkdir(parents=True)
        (nested / "input.cpp").write_text("int input = 1;\n")
        self._git("add", ".")
        self.revision = _pin_fixture_tree(self.source)

        self.provider = self.root / "libprovider.so"
        self.provider.write_bytes(b"fake provider")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self._write_tool("cmake", FAKE_CMAKE)
        for name in ("ninja", "cc", "c++", "ccache", "nvcc"):
            self._write_tool(name, FAKE_TOOL)
        self.previous_path = os.environ.get("PATH", os.defpath)
        self.environment = {
            "PATH": f"{self.bin}:{self.previous_path}",
            "CUDA_VISIBLE_DEVICES": "3,7",
            "CFLAGS": "-DUNRECORDED_COMPILER_INPUT=1",
            "CXXFLAGS": "-funsafe-math-optimizations",
            "LDFLAGS": "-Wl,unrecorded",
            "CPATH": "/unrecorded/include",
            "LD_PRELOAD": "/unrecorded/preload.so",
            "CC": "/unrecorded/cc",
            "CXX": "/unrecorded/cxx",
            "OMP_NUM_THREADS": "6",
            "CCACHE_SLOPPINESS": "pch_defines",
        }
        self._identity_patch = mock.patch.object(
            record_build, "producer_identity", side_effect=self._fake_producer_identity
        )
        self._identity_patch.start()
        self.addCleanup(self._identity_patch.stop)

    def tearDown(self) -> None:
        """Remove temporary repositories and tool fixtures."""
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.source), *arguments],
            check=True,
            text=True,
            capture_output=True,
        )

    def _write_tool(self, name: str, contents: str) -> Path:
        path = self.bin / name
        path.write_text(contents)
        path.chmod(0o755)
        return path

    def _fake_producer_identity(self, git: Path) -> dict[str, object]:
        del git
        return {
            "recorder": build_receipt.file_record(Path(record_build.__file__)),
            "verifier": build_receipt.file_record(Path(build_receipt.__file__)),
            "git": {"revision": "f" * 40, "dirty": False},
            "trust_scope": build_receipt.TRUST_SCOPE,
        }

    def _arguments(
        self,
        output: Path,
        *,
        backend: str = "cpu",
        no_ccache: bool = False,
        cuda_architectures: str = "80;90",
    ) -> list[str]:
        arguments = [
            "--source-root",
            str(self.source),
            "--revision",
            self.revision,
            "--output",
            str(output),
            "--backend",
            backend,
            "--cpu-linalg-library",
            str(self.provider),
            "--jobs",
            "3",
            "--cc",
            str(self.bin / "cc"),
            "--cxx",
            str(self.bin / "c++"),
        ]
        if backend == "cuda":
            arguments.extend(
                (
                    "--nvcc",
                    str(self.bin / "nvcc"),
                    "--cuda-architectures",
                    cuda_architectures,
                )
            )
        if no_ccache:
            arguments.append("--no-ccache")
        return arguments

    def _invoke(
        self, output: Path, **kwargs: object
    ) -> tuple[int, dict[str, object], str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = record_build.main(self._arguments(output, **kwargs))
        output_text = stdout.getvalue()
        if status == 2:
            raise AssertionError(f"argument/output error: {stderr.getvalue()}")
        return status, json.loads(output_text), output_text

    def _write_marker(self, name: str, contents: str = "") -> Path:
        marker = self.bin / name
        marker.write_text(contents)
        return marker

    def _receipt(self, output: Path) -> dict[str, object]:
        return json.loads((output / "receipt.json").read_text())

    def test_consumer_rejection_never_publishes_complete_success(self) -> None:
        """A successful fake build still needs the full consumer integrity check."""
        output = self.root / "consumer-rejected"
        with mock.patch.object(
            build_receipt,
            "verify_receipt",
            side_effect=build_receipt.ReceiptError("consumer rejection"),
        ):
            status, result, _ = self._invoke(output)
        self.assertEqual(status, 1)
        self.assertEqual(result["status"], "failed")
        failed = self._receipt(output)
        self.assertEqual(failed["failure"]["phase"], "consumer-check")
        self.assertIn("consumer rejection", failed["failure"]["reason"])
        self.assertFalse((output / ".receipt-candidate.json").exists())
        self.assertTrue((output / "build" / "libxtbloom.so").exists())

    def test_logged_timeout_is_terminal_failed_with_literal_exit_and_log(self) -> None:
        """Kill only the controlled process group and retain its timeout outcome."""
        log_path = self.root / "timeout.log"
        exit_code, log_record, timed_out = record_build._run_logged(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            phase="build",
            cwd=self.root,
            environment=record_build._minimal_environment(),
            log_path=log_path,
            timeout=0.1,
        )
        self.assertTrue(timed_out)
        self.assertLess(exit_code, 0)
        self.assertEqual(log_record["sha256"], build_receipt.sha256_file(log_path))

    def test_cpu_receipt_is_canonical_and_verifies_without_loading_library(
        self,
    ) -> None:
        """Produce and verify canonical CPU evidence using only fake binaries."""
        output = self.root / "cpu-bundle"
        with mock.patch.dict(os.environ, self.environment, clear=True):
            status, result, stdout = self._invoke(output)
        self.assertEqual(status, 0)
        receipt_path = output / "receipt.json"
        raw = receipt_path.read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(raw, build_receipt.canonical_json_bytes(receipt))
        self.assertEqual(result["receipt_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(stdout, build_receipt.canonical_json_bytes(result).decode())
        self.assertEqual(receipt["status"], "complete")
        self.assertFalse(receipt["performance_claim_eligible"])
        self.assertIsNone(receipt["failure"])
        self.assertNotEqual(
            receipt["producer"]["git"]["revision"], receipt["source"]["revision"]
        )
        self.assertIsNotNone(receipt["source"]["export_subst_original_base64"])
        source_paths = [record["path"] for record in receipt["source"]["manifest"]]
        self.assertEqual(source_paths, sorted(source_paths))
        self.assertEqual(receipt["source"]["snapshot_path"], str(output / "source"))
        self.assertEqual(receipt["build"]["root"], str(output / "build"))
        self.assertTrue(receipt["options"]["ccache"])
        capture = receipt["source_capture"]["command"]
        self.assertEqual(capture["phase"], "archive")
        self.assertEqual(
            capture["argv"],
            build_receipt.archive_argv(
                Path(receipt["tools"]["git"]["path"]),
                self.source,
                self.revision,
                output / "source.tar",
            ),
        )
        self.assertTrue(capture["started"])
        self.assertEqual(capture["exit_code"], 0)
        self.assertFalse(capture["timed_out"])
        self.assertIsNone(capture["launch_error"])
        self.assertEqual(
            capture["log"], build_receipt.file_record(output / "source-archive.log")
        )
        self.assertEqual(
            [command["argv"] for command in receipt["build"]["commands"]],
            [
                build_receipt.configure_argv(receipt),
                build_receipt.build_argv(receipt),
            ],
        )
        self.assertEqual(receipt["environment"]["CUDA_VISIBLE_DEVICES"], "3,7")
        self.assertEqual(receipt["environment"]["OMP_NUM_THREADS"], "6")
        self.assertEqual(receipt["environment"]["CCACHE_SLOPPINESS"], "pch_defines")
        for blocked in (
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
            "CPATH",
            "LD_PRELOAD",
        ):
            self.assertNotIn(blocked, receipt["environment"])
        self.assertTrue(
            all(key in build_receipt.ENVIRONMENT_KEYS for key in receipt["environment"])
        )
        self.assertTrue(
            all(
                Path(record["path"]).is_relative_to(output.resolve())
                for record in (
                    receipt["source"]["archive"],
                    capture["log"],
                    *[command["log"] for command in receipt["build"]["commands"]],
                )
            )
        )
        self.assertIn(
            "fake.o",
            {Path(item["path"]).name for item in receipt["build"]["manifest"]},
        )
        library = Path(receipt["artifacts"]["library"]["path"])
        self.assertTrue(library.name.startswith("libxtbloom.so."))
        verified = build_receipt.verify_receipt(
            receipt_path,
            result["receipt_sha256"],
            library,
            expected_source_revision=self.revision,
        )
        self.assertEqual(verified["status"], "LOCAL_RECEIPT_MATCHES")
        self.assertFalse(verified["performance_claim_eligible"])

    def test_cuda_argv_and_visible_devices_are_recorded(self) -> None:
        """Record explicit CUDA options without probing or running a GPU."""
        output = self.root / "cuda-bundle"
        status, result, _ = self._invoke(output, backend="cuda")
        receipt = self._receipt(output)
        self.assertEqual(status, 0, result)
        self.assertEqual(receipt["options"]["cuda_architectures"], "80;90")
        self.assertEqual(receipt["environment"]["CUDA_VISIBLE_DEVICES"], "3,7")
        configure = receipt["build"]["commands"][0]["argv"]
        self.assertIn("-DCMAKE_CUDA_COMPILER=" + str(self.bin / "nvcc"), configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80;90", configure)
        self.assertIn(
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=" + str(self.bin / "ccache"), configure
        )

    def test_cuda_real_virtual_suffixes_reach_the_fake_build(self) -> None:
        """Supported CMake suffixes must not crash before receipt creation."""
        output = self.root / "cuda-suffix-bundle"
        architectures = "80;90-real;90-virtual"
        status, result, _ = self._invoke(
            output, backend="cuda", cuda_architectures=architectures
        )
        self.assertEqual(status, 0, result)
        receipt = self._receipt(output)
        self.assertEqual(receipt["options"]["cuda_architectures"], architectures)
        self.assertIn(
            "-DCMAKE_CUDA_ARCHITECTURES=" + architectures,
            receipt["build"]["commands"][0]["argv"],
        )

    def test_cuda_invalid_architectures_fail_as_cli_errors(self) -> None:
        """Reject bad numeric components consistently without uncaught exceptions."""
        for index, architectures in enumerate(
            ("0-real", "90;090", "90-real;090-real", "native")
        ):
            with self.subTest(architectures=architectures):
                output = self.root / f"invalid-architecture-{index}"
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as failure,
                ):
                    record_build.main(
                        self._arguments(
                            output,
                            backend="cuda",
                            cuda_architectures=architectures,
                        )
                    )
                self.assertEqual(failure.exception.code, 2)
                self.assertFalse(output.exists())

    def test_archive_failure_and_timeout_preserve_literal_command_logs(self) -> None:
        """Failed source capture keeps its exit and log, not just a reason string."""
        real_git = shutil.which("git", path=self.previous_path)
        self.assertIsNotNone(real_git)
        for index, timed_out in enumerate((False, True)):
            with self.subTest(timed_out=timed_out):
                action = "time.sleep(10)" if timed_out else "sys.exit(37)"
                self._write_tool(
                    "git",
                    f"""#!/usr/bin/env python3
import os
import sys
import time
if "archive" in sys.argv[1:]:
    print("synthetic archive failure", file=sys.stderr, flush=True)
    {action}
os.execv({real_git!r}, [{real_git!r}, *sys.argv[1:]])
""",
                )
                output = self.root / f"archive-failure-{index}"
                with mock.patch.object(record_build, "ARCHIVE_TIMEOUT_SECONDS", 0.1):
                    status, result, _ = self._invoke(output)
                self.assertEqual(status, 1, result)
                receipt = self._receipt(output)
                self.assertEqual(receipt["failure"]["phase"], "archive")
                command = receipt["source_capture"]["command"]
                self.assertTrue(command["started"])
                self.assertEqual(command["timed_out"], timed_out)
                self.assertIsNone(command["launch_error"])
                self.assertEqual(command["argv"][-1], self.revision)
                self.assertNotEqual(command["exit_code"], 0)
                if not timed_out:
                    self.assertEqual(command["exit_code"], 37)
                log = output / "source-archive.log"
                self.assertEqual(command["log"], build_receipt.file_record(log))
                self.assertIn("synthetic archive failure", log.read_text())
                self.assertEqual(receipt["build"]["commands"], [])

    def test_archive_launch_error_retains_unstarted_history(self) -> None:
        """An OS launch failure must not invent a successful command exit."""
        output = self.root / "archive-launch-failure"
        output.mkdir()
        command = {}
        with (
            mock.patch.object(
                record_build.subprocess, "Popen", side_effect=OSError("launch failed")
            ),
            self.assertRaisesRegex(record_build.RecorderError, "launch failed"),
        ):
            record_build._create_archive(
                Path("/fake/git"), self.source, self.revision, output, command
            )
        self.assertFalse(command["started"])
        self.assertIsNone(command["exit_code"])
        self.assertFalse(command["timed_out"])
        self.assertEqual(command["launch_error"], "launch failed")
        self.assertEqual(
            command["log"], build_receipt.file_record(output / "source-archive.log")
        )

    def test_no_ccache_uses_no_compiler_launchers(self) -> None:
        """Honor the opt-out for compiler launchers."""
        output = self.root / "no-cache-bundle"
        status, result, _ = self._invoke(output, no_ccache=True)
        receipt = self._receipt(output)
        self.assertEqual(status, 0, result)
        self.assertFalse(receipt["options"]["ccache"])
        self.assertNotIn("ccache", receipt["tools"])
        self.assertFalse(
            any(
                "COMPILER_LAUNCHER" in item
                for item in receipt["build"]["commands"][0]["argv"]
            )
        )

    def test_cuda_requires_both_explicit_compiler_arguments(self) -> None:
        """Reject incomplete CUDA configuration before creating a bundle."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing-cuda"
            stdout = io.StringIO()
            stderr = io.StringIO()
            arguments = self._arguments(output, backend="cuda")
            arguments.remove("--nvcc")
            arguments.remove(str(self.bin / "nvcc"))
            arguments.remove("--cuda-architectures")
            arguments.remove("80;90")
            with (
                mock.patch.dict(os.environ, self.environment, clear=True),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit),
            ):
                record_build.main(arguments)
            self.assertFalse(output.exists())
            self.assertIn(
                "requires both --nvcc and --cuda-architectures", stderr.getvalue()
            )

    def test_existing_output_directory_is_never_modified(self) -> None:
        """Refuse to reuse any existing output path."""
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("preserve")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = record_build.main(self._arguments(output))
        self.assertEqual(status, 2)
        self.assertEqual(sentinel.read_text(), "preserve")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_dirty_source_and_git_symlinks_emit_failed_receipts(self) -> None:
        """Reject dirty trees and unsupported symlink blobs as failed evidence."""
        (self.source / "tracked.txt").write_text("dirty\n")
        output = self.root / "dirty-bundle"
        status, result, _ = self._invoke(output)
        self.assertEqual(status, 1, result)
        self.assertEqual(self._receipt(output)["failure"]["phase"], "source")
        self.assertEqual(self._receipt(output)["status"], "failed")

        symlink_source = self.root / "symlink-source"
        symlink_source.mkdir()
        self._git("reset", "--hard", "-q", self.revision)
        self._git("clean", "-fdq")
        (symlink_source / "link").symlink_to(self.source / "tracked.txt")
        subprocess.run(["git", "-C", str(symlink_source), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(symlink_source), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(symlink_source),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(symlink_source), "add", "link"], check=True)
        symlink_revision = _pin_fixture_tree(symlink_source)
        symlink_output = self.root / "symlink-bundle"
        with mock.patch.object(
            record_build,
            "producer_identity",
            side_effect=self._fake_producer_identity,
        ):
            status, result, _ = self._invoke_for_source(
                symlink_source, symlink_revision, symlink_output
            )
        self.assertEqual(status, 1, result)
        self.assertEqual(self._receipt(symlink_output)["failure"]["phase"], "source")

    def _invoke_for_source(
        self, source: Path, revision: str, output: Path
    ) -> tuple[int, dict[str, object], str]:
        arguments = self._arguments(output)
        arguments[arguments.index("--source-root") + 1] = str(source)
        arguments[arguments.index("--revision") + 1] = revision
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = record_build.main(arguments)
        if status == 2:
            raise AssertionError(stderr.getvalue())
        return status, json.loads(stdout.getvalue()), stdout.getvalue()

    def test_configure_failure_is_recorded_and_never_complete(self) -> None:
        """Record a configure exit failure and its exact log."""
        self._write_marker("fail-configure")
        output = self.root / "configure-failure"
        status, result, _ = self._invoke(output)
        receipt = self._receipt(output)
        self.assertEqual(status, 1, result)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["failure"]["phase"], "configure")
        self.assertEqual(receipt["build"]["commands"][0]["exit_code"], 23)
        self.assertTrue(Path(receipt["build"]["commands"][0]["log"]["path"]).is_file())

    def test_build_failure_and_missing_library_are_failed_receipts(self) -> None:
        """Keep build failures and missing outputs ineligible."""
        self._write_marker("fail-build")
        output = self.root / "build-failure"
        status, result, _ = self._invoke(output)
        receipt = self._receipt(output)
        self.assertEqual(status, 1, result)
        self.assertEqual(receipt["failure"]["phase"], "build")
        self.assertEqual(receipt["build"]["commands"][1]["exit_code"], 29)

        (self.bin / "fail-build").unlink()
        self._write_marker("omit-library")
        missing_output = self.root / "missing-library"
        status, result, _ = self._invoke(missing_output)
        self.assertEqual(status, 1, result)
        self.assertIn(
            "one canonical versioned libxtbloom.so",
            self._receipt(missing_output)["failure"]["reason"],
        )

    def test_source_snapshot_and_original_tree_mutation_fail_postflight(self) -> None:
        """Reject changed snapshots and source worktrees after the build."""
        marker = self._write_marker("mutate-snapshot")
        output = self.root / "snapshot-mutation"
        status, result, _ = self._invoke(output)
        self.assertEqual(status, 1, result)
        self.assertIn("source snapshot", self._receipt(output)["failure"]["reason"])
        marker.unlink()

        self._write_marker("mutate-source", str(self.source))
        source_output = self.root / "source-mutation"
        status, result, _ = self._invoke(source_output)
        self.assertEqual(status, 1, result)
        self.assertIn(
            "source root or selected Git commit changed",
            self._receipt(source_output)["failure"]["reason"],
        )

    def test_tool_postflight_mismatch_and_escaping_build_symlink_fail(self) -> None:
        """Reject replaced tools and build artifacts that resolve outside."""
        self._write_marker("change-cc-version")
        output = self.root / "tool-mutation"
        status, result, _ = self._invoke(output)
        self.assertEqual(status, 1, result)
        self.assertIn("build tool changed", self._receipt(output)["failure"]["reason"])
        (self.bin / "change-cc-version").unlink()

        outside = self.root / "outside.txt"
        outside.write_text("outside")
        self._write_marker("escape-build-link", str(outside))
        escaped_output = self.root / "escaped-build"
        status, result, _ = self._invoke(escaped_output)
        self.assertEqual(status, 1, result)
        self.assertIn(
            "escapes build root", self._receipt(escaped_output)["failure"]["reason"]
        )

    def test_cmake_cache_must_match_the_reconstructed_configure_argv(self) -> None:
        """Reject cache drift from the exact recorded configure arguments."""
        self._write_marker("wrong-cache")
        output = self.root / "wrong-cache-bundle"
        status, result, _ = self._invoke(output)
        receipt = self._receipt(output)
        self.assertEqual(status, 1, result)
        self.assertEqual(receipt["build"]["commands"][0]["exit_code"], 0)
        self.assertEqual(receipt["build"]["commands"][1]["exit_code"], 0)
        self.assertIn(
            "CMake cache differs from configure argv", receipt["failure"]["reason"]
        )


if __name__ == "__main__":
    unittest.main()
