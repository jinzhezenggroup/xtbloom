"""Synthetic integrity and hostile-input tests; no native library is executed."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmarks import build_receipt as receipt


def _archive(path: Path, revision: str, files: dict[str, bytes]) -> None:
    """Create fixture payloads, not actual Git commits or native build evidence."""
    with tarfile.open(
        path, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": revision}
    ) as archive:
        for name, data in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))


def _freeze(snapshot: Path) -> None:
    """Match the recorder's private read-only source boundary."""
    for path in snapshot.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot.chmod(0o555)


class ReceiptFixture:
    """Build synthetic pinned data; successful verification is not runtime proof."""

    def __init__(self, root: Path, *, export_subst: bool = False) -> None:
        root.mkdir()
        self.root = root
        self.snapshot = root / "source"
        self.build = root / "build"
        self.build.mkdir()
        files = {
            "CMakeLists.txt": b"fixture source\n",
            "src/model.cpp": b"fixture model\n",
        }
        original = b"node: $Format:%H$\n" if export_subst else None
        if original is not None:
            files[".git_archival.txt"] = original
        manifest = [
            {
                "path": name,
                "mode": "100644",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "git_blob_oid": receipt.git_object_oid("blob", data, "sha1"),
            }
            for name, data in sorted(files.items())
        ]
        tree = receipt.git_tree_oid(manifest, "sha1")
        commit = (
            f"tree {tree}\nauthor Fixture <fixture@example.invalid> 0 +0000\n"
            "committer Fixture <fixture@example.invalid> 0 +0000\n\n"
            "synthetic object bytes\n"
        ).encode()
        self.revision = receipt.git_object_oid("commit", commit, "sha1")
        if original is not None:
            files[".git_archival.txt"] = f"node: {self.revision}\n".encode()
            manifest[0]["sha256"] = hashlib.sha256(
                files[".git_archival.txt"]
            ).hexdigest()
            manifest[0]["size_bytes"] = len(files[".git_archival.txt"])
        archive_path = root / "source.tar"
        _archive(archive_path, self.revision, files)
        receipt.archive_manifest(archive_path, "sha1", self.snapshot)
        _freeze(self.snapshot)
        tool_path = root / "tool-fixture"
        tool_path.write_bytes(b"fake tool, never executed\n")
        tool = dict(receipt.file_record(tool_path), version="fixture version")
        self.library = self.build / "libxtbloom.so.0"
        self.library.write_bytes(b"fake library, never loaded\n")
        provider = root / "provider-fixture.so"
        provider.write_bytes(b"fake LP64 provider, never loaded\n")
        archive_log = root / "source-archive.log"
        archive_log.write_bytes(b"synthetic source-capture log; not executed\n")
        self.path = root / "receipt.json"
        self.document: dict[str, Any] = {
            "schema_version": 1,
            "kind": receipt.RECEIPT_KIND,
            "status": "complete",
            "performance_claim_eligible": False,
            "failure": None,
            "producer": {
                "recorder": receipt.file_record(
                    Path(receipt.__file__).with_name("record_build.py")
                ),
                "verifier": receipt.file_record(Path(receipt.__file__)),
                "git": {"revision": "a" * 40, "dirty": False},
                "trust_scope": receipt.TRUST_SCOPE,
            },
            "source": {
                "revision": self.revision,
                "tree": tree,
                "object_format": "sha1",
                "commit_object_base64": base64.b64encode(commit).decode(),
                "snapshot_path": str(self.snapshot),
                "archive": receipt.file_record(archive_path),
                "manifest": manifest,
                "manifest_sha256": hashlib.sha256(
                    receipt.canonical_json_bytes(manifest)
                ).hexdigest(),
                "export_subst_original_base64": base64.b64encode(original).decode()
                if original is not None
                else None,
            },
            "options": {
                "backend": "cpu",
                "build_type": "Release",
                "tests": False,
                "jobs": 2,
                "cuda_architectures": None,
                "ccache": False,
            },
            "tools": {
                name: copy.deepcopy(tool)
                for name in ("git", "cmake", "ninja", "cc", "cxx")
            },
            "provider": receipt.file_record(provider),
            "source_capture": {
                "source_root": str(root),
                "command": {
                    "phase": "archive",
                    "argv": receipt.archive_argv(
                        Path(tool["path"]), root, self.revision, archive_path
                    ),
                    "started": True,
                    "exit_code": 0,
                    "timed_out": False,
                    "launch_error": None,
                    "log": receipt.file_record(archive_log),
                },
            },
            "environment": {"CUDA_VISIBLE_DEVICES": "1"},
            "build": {"root": str(self.build), "commands": []},
            "artifacts": {"library": receipt.file_record(self.library)},
        }
        self.write_build_configuration()
        self.sign()

    def write_build_configuration(self) -> None:
        """Emit matching fake cache/logs instead of calling CMake or a compiler."""
        commands = self.document["build"]["commands"] = []
        configure = receipt.configure_argv(self.document)
        cache_values = {
            "CMAKE_HOME_DIRECTORY": str(self.snapshot),
            "CMAKE_GENERATOR": "Ninja",
        }
        for argument in configure:
            if argument.startswith("-D"):
                key, value = argument[2:].split("=", 1)
                cache_values[key] = value
        cache = self.build / "CMakeCache.txt"
        cache.write_text(
            "".join(f"{key}:STRING={value}\n" for key, value in cache_values.items())
        )
        (self.build / "build.ninja").write_bytes(b"fake build graph\n")
        (self.build / "compile_commands.json").write_bytes(b"[]\n")
        for phase, arguments in (
            ("configure", configure),
            ("build", receipt.build_argv(self.document)),
        ):
            log = self.root / f"{phase}.log"
            log.write_bytes(b"synthetic successful command record; not executed\n")
            commands.append(
                {
                    "phase": phase,
                    "argv": arguments,
                    "exit_code": 0,
                    "log": receipt.file_record(log),
                }
            )
        self.document["build"]["cache"] = receipt.file_record(cache)
        self.document["build"]["manifest"] = [
            receipt.file_record(path) for path in sorted(self.build.iterdir())
        ]

    def sign(self) -> str:
        """Pin canonical fixture bytes externally; this is not an attestation."""
        self.path.write_bytes(receipt.canonical_json_bytes(self.document))
        self.digest = receipt.sha256_file(self.path)
        return self.digest

    def verify(self) -> dict[str, Any]:
        """Exercise the complete read-only checker at the expected source pin."""
        return receipt.verify_receipt(
            self.path, self.digest, self.library, self.revision
        )


class ObjectAndArchiveTests(unittest.TestCase):
    """Reject hostile archives and verify object framing without a Git process."""

    def setUp(self) -> None:
        """Keep archive extraction inside one disposable fixture directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_git_objects_and_directory_sorting(self) -> None:
        """Directory names sort as slash-terminated names in a Git tree."""
        empty_blob = receipt.git_object_oid("blob", b"", "sha1")
        self.assertEqual(empty_blob, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        records = [
            {"path": path, "mode": "100644", "git_blob_oid": empty_blob}
            for path in ("a/z", "a.txt")
        ]
        child = receipt.git_object_oid(
            "tree", b"100644 z\0" + bytes.fromhex(empty_blob), "sha1"
        )
        expected = receipt.git_object_oid(
            "tree",
            b"100644 a.txt\0"
            + bytes.fromhex(empty_blob)
            + b"40000 a\0"
            + bytes.fromhex(child),
            "sha1",
        )
        self.assertEqual(receipt.git_tree_oid(records, "sha1"), expected)

    def test_archive_rejects_traversal_links_and_duplicates(self) -> None:
        """No rejected archive may publish a path outside its destination."""
        for name, kind in (
            ("../escape", tarfile.REGTYPE),
            ("/escape", tarfile.REGTYPE),
            ("link", tarfile.SYMTYPE),
            ("link", tarfile.LNKTYPE),
        ):
            with self.subTest(name=name, kind=kind):
                archive_path = self.root / "hostile.tar"
                with tarfile.open(
                    archive_path,
                    "w",
                    format=tarfile.PAX_FORMAT,
                    pax_headers={"comment": "a" * 40},
                ) as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.linkname = "../escape"
                    archive.addfile(member)
                with self.assertRaises(receipt.ReceiptError):
                    receipt.archive_manifest(archive_path, "sha1")
        archive_path = self.root / "duplicate.tar"
        with tarfile.open(
            archive_path,
            "w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": "a" * 40},
        ) as archive:
            archive.addfile(tarfile.TarInfo("same"))
            archive.addfile(tarfile.TarInfo("same"))
        with self.assertRaisesRegex(receipt.ReceiptError, "duplicate"):
            receipt.archive_manifest(archive_path, "sha1")

    def test_extraction_rejects_existing_payload_and_missing_commit(self) -> None:
        """Fresh-destination rules prevent overwriting or following old files."""
        archive_path = self.root / "source.tar"
        _archive(archive_path, "a" * 40, {"source.cpp": b"data"})
        destination = self.root / "snapshot"
        destination.mkdir()
        (destination / "keep").write_text("user data")
        with self.assertRaisesRegex(receipt.ReceiptError, "fresh"):
            receipt.archive_manifest(archive_path, "sha1", destination)
        self.assertEqual((destination / "keep").read_text(), "user data")
        _archive(archive_path, "", {"source.cpp": b"data"})
        with self.assertRaisesRegex(receipt.ReceiptError, "commit"):
            receipt.archive_manifest(archive_path, "sha1")


class ReceiptIntegrityTests(unittest.TestCase):
    """Synthetic data proves integrity predicates, never actual build execution."""

    def setUp(self) -> None:
        """Create synthetic source/artifact pins without invoking a compiler."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = ReceiptFixture(self.root / "bundle")

    def test_valid_receipt_and_equal_byte_copy_do_not_qualify_performance(self) -> None:
        """Require the retained producer bundle even when selecting a binary copy."""
        result = self.fixture.verify()
        self.assertEqual(result["status"], "LOCAL_RECEIPT_MATCHES")
        self.assertFalse(result["performance_claim_eligible"])
        self.assertIn("independent scientific correctness", result["not_qualified"])
        copied = self.root / "copied.so"
        copied.write_bytes(self.fixture.library.read_bytes())
        result = receipt.verify_receipt(
            self.fixture.path, self.fixture.digest, copied, self.fixture.revision
        )
        self.assertEqual(result["library"]["path"], str(copied))

    def test_external_receipt_source_and_library_pins_are_independent(self) -> None:
        """Current source state or a different binary cannot repair a wrong pin."""
        with self.assertRaisesRegex(receipt.ReceiptError, "external pin"):
            receipt.verify_receipt(self.fixture.path, "0" * 64, self.fixture.library)
        with self.assertRaisesRegex(receipt.ReceiptError, "source revision"):
            receipt.verify_receipt(
                self.fixture.path, self.fixture.digest, self.fixture.library, "0" * 40
            )
        wrong = self.root / "wrong.so"
        wrong.write_bytes(b"different code")
        with self.assertRaisesRegex(receipt.ReceiptError, "selected library"):
            receipt.verify_receipt(self.fixture.path, self.fixture.digest, wrong)

    def test_malformed_schema_and_nonfinite_or_duplicate_json_fail_closed(self) -> None:
        """Bool aliases, unsupported schemas and self-declared eligibility fail."""
        original = copy.deepcopy(self.fixture.document)
        for field, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("status", "failed"),
            ("performance_claim_eligible", True),
            ("failure", {"phase": "build", "reason": "failed"}),
        ):
            with self.subTest(field=field, value=value):
                self.fixture.document = dict(original, **{field: value})
                self.fixture.sign()
                with self.assertRaises(receipt.ReceiptError):
                    self.fixture.verify()
        for raw in (
            b'{"schema_version":1,"schema_version":1}\n',
            b'{"value":NaN}\n',
            b'{ "schema_version": 1 }\n',
        ):
            self.fixture.path.write_bytes(raw)
            self.fixture.digest = receipt.sha256_file(self.fixture.path)
            with self.assertRaises(receipt.ReceiptError):
                self.fixture.verify()

    def test_source_commit_tree_archive_and_snapshot_are_all_bound(self) -> None:
        """Re-pinning damaged metadata does not change the underlying Git object."""
        original = copy.deepcopy(self.fixture.document)
        for field, value in (
            ("tree", "0" * 40),
            ("commit_object_base64", base64.b64encode(b"wrong commit").decode()),
            ("manifest_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                self.fixture.document = copy.deepcopy(original)
                self.fixture.document["source"][field] = value
                self.fixture.sign()
                with self.assertRaises(receipt.ReceiptError):
                    self.fixture.verify()
        self.fixture.document = original
        self.fixture.sign()
        path = self.fixture.snapshot / "src/model.cpp"
        path.chmod(0o644)
        path.write_bytes(b"changed model")
        path.chmod(0o444)
        with self.assertRaisesRegex(receipt.ReceiptError, "snapshot"):
            self.fixture.verify()

    def test_only_git_archival_export_substitution_is_supported(self) -> None:
        """Bind original Git metadata and expanded archive bytes separately."""
        fixture = ReceiptFixture(self.root / "exported", export_subst=True)
        self.assertEqual(fixture.verify()["status"], "LOCAL_RECEIPT_MATCHES")
        fixture.document["source"]["export_subst_original_base64"] = base64.b64encode(
            b"wrong original metadata"
        ).decode()
        fixture.sign()
        with self.assertRaisesRegex(receipt.ReceiptError, "archive bytes"):
            fixture.verify()

    def test_actual_cache_command_exits_and_build_files_are_required(self) -> None:
        """A successful-looking declaration cannot omit or contradict execution."""
        original = copy.deepcopy(self.fixture.document)
        for change in ("command", "exit", "manifest", "backend", "jobs"):
            with self.subTest(change=change):
                self.fixture.document = copy.deepcopy(original)
                if change == "command":
                    self.fixture.document["build"]["commands"][0]["argv"].append(
                        "-DXTBLOOM_BUILD_TESTS=ON"
                    )
                elif change == "exit":
                    self.fixture.document["build"]["commands"][1]["exit_code"] = False
                elif change == "manifest":
                    self.fixture.document["build"]["manifest"] = []
                elif change == "backend":
                    self.fixture.document["options"]["backend"] = "AUTO"
                else:
                    self.fixture.document["options"]["jobs"] = True
                self.fixture.sign()
                with self.assertRaises(receipt.ReceiptError):
                    self.fixture.verify()

    def test_archive_command_exit_timeout_and_log_are_bound(self) -> None:
        """Successful source bytes cannot replace the source-capture operation."""
        original = copy.deepcopy(self.fixture.document)
        for change in ("missing", "argv", "exit", "timeout", "started", "launch"):
            with self.subTest(change=change):
                self.fixture.document = copy.deepcopy(original)
                command = self.fixture.document["source_capture"]["command"]
                if change == "missing":
                    del self.fixture.document["source_capture"]
                elif change == "argv":
                    command["argv"][-1] = "b" * 40
                elif change == "exit":
                    command["exit_code"] = False
                elif change == "timeout":
                    command["timed_out"] = True
                elif change == "started":
                    command["started"] = False
                else:
                    command["launch_error"] = "failed to launch"
                self.fixture.sign()
                with self.assertRaises(receipt.ReceiptError):
                    self.fixture.verify()
        self.fixture.document = original
        self.fixture.sign()
        log = self.fixture.document["source_capture"]["command"]["log"]
        Path(log["path"]).write_bytes(b"changed archive log")
        with self.assertRaisesRegex(receipt.ReceiptError, "archive log bytes changed"):
            self.fixture.verify()

    def test_cuda_architectures_reject_nonpositive_and_normalized_duplicates(
        self,
    ) -> None:
        """Both sides reject aliases while keeping real/virtual variants legal."""
        for invalid in ("0", "0-real", "90;90", "090-real;90-real", "-90", "native"):
            with (
                self.subTest(architectures=invalid),
                self.assertRaises(receipt.ReceiptError),
            ):
                receipt.validate_cuda_architectures(invalid)
        for valid in ("90-real", "90-virtual", "90-real;90-virtual", "80;90-real"):
            with self.subTest(architectures=valid):
                receipt.validate_cuda_architectures(valid)
        self.fixture.document["options"].update(
            backend="cuda", cuda_architectures="90-real;90-virtual"
        )
        self.fixture.document["tools"]["nvcc"] = copy.deepcopy(
            self.fixture.document["tools"]["cxx"]
        )
        for invalid in ("0-real", "90;090", "90-virtual;090-virtual"):
            with self.subTest(receipt_architectures=invalid):
                self.fixture.document["options"]["cuda_architectures"] = invalid
                self.fixture.write_build_configuration()
                self.fixture.sign()
                with self.assertRaisesRegex(
                    receipt.ReceiptError, "positive and unique"
                ):
                    self.fixture.verify()

    def test_postflight_tool_log_provider_and_library_mutations_are_rejected(
        self,
    ) -> None:
        """Keep failed integrity coordinates explicit instead of accepting a tail."""
        for index, role in enumerate(("tool", "log", "provider", "library")):
            with self.subTest(role=role):
                fixture = ReceiptFixture(self.root / f"mutation-{index}")
                choices = {
                    "tool": fixture.document["tools"]["cmake"],
                    "log": fixture.document["build"]["commands"][0]["log"],
                    "provider": fixture.document["provider"],
                    "library": fixture.document["artifacts"]["library"],
                }
                Path(choices[role]["path"]).write_bytes(b"changed bytes")
                with self.assertRaises(receipt.ReceiptError):
                    fixture.verify()

    def test_explicit_cuda_configuration_and_visibility_are_metadata_only(self) -> None:
        """CUDA receipts require nvcc/architecture pins without probing a GPU."""
        self.fixture.document["options"].update(
            backend="cuda", cuda_architectures="120;90-real"
        )
        self.fixture.document["tools"]["nvcc"] = copy.deepcopy(
            self.fixture.document["tools"]["cxx"]
        )
        self.fixture.write_build_configuration()
        self.fixture.sign()
        result = self.fixture.verify()
        self.assertEqual(result["backend"], "cuda")
        self.assertEqual(
            self.fixture.document["environment"]["CUDA_VISIBLE_DEVICES"], "1"
        )
        self.fixture.document["options"]["cuda_architectures"] = "native"
        self.fixture.sign()
        with self.assertRaisesRegex(receipt.ReceiptError, "explicit numeric"):
            self.fixture.verify()


if __name__ == "__main__":
    unittest.main()
