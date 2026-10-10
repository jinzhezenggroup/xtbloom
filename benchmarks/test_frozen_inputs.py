"""Regression tests for hash-pinned workload byte snapshots."""

from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import frozen_inputs


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FrozenWorkloadTests(unittest.TestCase):
    """Exercise manifest/input capture without launching a model backend."""

    def setUp(self) -> None:
        """Create a manifest with three independently hash-pinned inputs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "workload.json"
        self.input_payloads = {
            case_id: f"$coord\n{index} 0 0 h\n$end\n".encode() + b"\xff"
            for index, case_id in enumerate(("a", "b", "outside"))
        }
        self.source_paths = {
            case_id: self.root / f"{case_id}.coord" for case_id in self.input_payloads
        }
        for case_id, path in self.source_paths.items():
            path.write_bytes(self.input_payloads[case_id])
        self.document = {
            "schema_version": 1,
            "method": "GFN2-xTB",
            "cases": [
                {
                    "id": case_id,
                    "input": f"{case_id}.coord",
                    "input_sha256": _sha256(self.input_payloads[case_id]),
                    "atom_count": 1,
                }
                for case_id in self.input_payloads
            ],
        }
        self._write_manifest()

    def _write_manifest(self) -> bytes:
        payload = json.dumps(self.document, sort_keys=True).encode("utf-8")
        self.manifest_path.write_bytes(payload)
        self.manifest_payload = payload
        self.manifest_sha256 = _sha256(payload)
        return payload

    def _resolver(self, case: dict[str, object]) -> Path:
        return self.root / str(case["input"])

    def capture(
        self,
        case_ids: tuple[str, ...] = ("a", "b"),
        *,
        expected_manifest_sha256: str | None = None,
    ) -> frozen_inputs.FrozenWorkload:
        """Capture selected fixture cases using their manifest-relative paths."""
        return frozen_inputs.capture_workload(
            self.manifest_path,
            (
                self.manifest_sha256
                if expected_manifest_sha256 is None
                else expected_manifest_sha256
            ),
            case_ids,
            input_resolver=self._resolver,
        )

    def test_copies_only_frozen_cases_to_read_only_snapshots(self) -> None:
        """Expose read-only copies only for IDs in the requested frozen roster."""
        frozen = self.capture()
        self.addCleanup(frozen.close)

        self.assertEqual(frozen.manifest, self.document)
        self.assertEqual(set(frozen.cases_by_id), {"a", "b"})
        self.assertEqual(set(frozen.input_records_by_case_id), {"a", "b"})
        self.assertEqual(
            frozen.manifest_record,
            {
                "path": str(self.manifest_path),
                "sha256": self.manifest_sha256,
                "size_bytes": len(self.manifest_payload),
            },
        )
        self.assertEqual(
            frozen.captured_input_bytes,
            len(self.input_payloads["a"]) + len(self.input_payloads["b"]),
        )
        for case_id in ("a", "b"):
            snapshot_path = Path(frozen.cases_by_id[case_id]["input"])
            self.assertTrue(snapshot_path.is_absolute())
            self.assertEqual(snapshot_path.read_bytes(), self.input_payloads[case_id])
            self.assertEqual(stat.S_IMODE(snapshot_path.stat().st_mode), 0o400)
            self.assertEqual(
                frozen.input_records_by_case_id[case_id],
                {
                    "path": str(self.source_paths[case_id]),
                    "sha256": _sha256(self.input_payloads[case_id]),
                    "size_bytes": len(self.input_payloads[case_id]),
                },
            )
            self.assertEqual(
                self.document["cases"][ord(case_id) - ord("a")]["input"],
                f"{case_id}.coord",
            )

        snapshot_paths = [Path(case["input"]) for case in frozen.cases_by_id.values()]
        frozen.close()
        frozen.close()
        self.assertTrue(all(not path.exists() for path in snapshot_paths))

    def test_manifest_pin_requires_lowercase_sha256_and_exact_bytes(self) -> None:
        """Reject malformed or mismatched external manifest identities."""
        for invalid_pin in ("", "A" * 64, "a" * 63, "z" * 64):
            with (
                self.subTest(pin=invalid_pin),
                self.assertRaises(frozen_inputs.FrozenInputError),
            ):
                self.capture(expected_manifest_sha256=invalid_pin)
        with self.assertRaisesRegex(frozen_inputs.FrozenInputError, "does not match"):
            self.capture(expected_manifest_sha256="0" * 64)

    def test_rejects_duplicate_manifest_and_frozen_case_ids(self) -> None:
        """Reject duplicate IDs on either side of the manifest lookup."""
        duplicate = self.document["cases"][0].copy()
        self.document["cases"].append(duplicate)
        self._write_manifest()
        with self.assertRaisesRegex(
            frozen_inputs.FrozenInputError, "duplicate workload"
        ):
            self.capture()

        self.document["cases"].pop()
        self._write_manifest()
        with self.assertRaisesRegex(frozen_inputs.FrozenInputError, "must be unique"):
            self.capture(("a", "a"))

    def test_rejects_missing_frozen_peer(self) -> None:
        """Require every frozen roster ID to exist in the captured manifest."""
        with self.assertRaisesRegex(frozen_inputs.FrozenInputError, "absent"):
            self.capture(("a", "missing"))

    def test_hashes_and_parses_same_captured_manifest_and_input_bytes(self) -> None:
        """Keep the snapshot stable when source files change after read_bytes."""
        original_manifest = self.manifest_payload
        original_input = self.input_payloads["a"]
        changed_manifest = b'{"cases": [], "schema_version": 1}'
        changed_input = b"different geometry bytes"
        original_read_bytes = Path.read_bytes

        def read_then_change(path: Path) -> bytes:
            payload = original_read_bytes(path)
            resolved_path = path.resolve()
            if resolved_path == self.manifest_path.resolve():
                self.manifest_path.write_bytes(changed_manifest)
            elif resolved_path == self.source_paths["a"].resolve():
                self.source_paths["a"].write_bytes(changed_input)
            return payload

        with mock.patch.object(Path, "read_bytes", read_then_change):
            frozen = self.capture(("a",))
        self.addCleanup(frozen.close)

        self.assertEqual(frozen.manifest, json.loads(original_manifest))
        self.assertEqual(frozen.manifest_record["sha256"], self.manifest_sha256)
        snapshot_path = Path(frozen.cases_by_id["a"]["input"])
        self.assertEqual(snapshot_path.read_bytes(), original_input)
        self.assertEqual(self.manifest_path.read_bytes(), changed_manifest)
        self.assertEqual(self.source_paths["a"].read_bytes(), changed_input)

    def test_rejects_nonfinite_and_duplicate_key_json(self) -> None:
        """Reject non-standard constants, overflowing floats, and duplicate keys."""
        malformed_documents = (
            b'{"cases": [], "value": NaN}',
            b'{"cases": [], "value": 1e999}',
            b'{"cases": [], "cases": []}',
        )
        original_payload = self.manifest_payload
        for payload in malformed_documents:
            with self.subTest(payload=payload):
                self.manifest_path.write_bytes(payload)
                with self.assertRaises(frozen_inputs.FrozenInputError):
                    self.capture(expected_manifest_sha256=_sha256(payload))
        self.manifest_path.write_bytes(original_payload)

    def test_aliases_share_one_capture_and_check_each_expected_hash(self) -> None:
        """Capture aliased source files once and require matching case hashes."""
        alias_directory = self.root / "alias"
        alias_directory.mkdir()
        alias_path = alias_directory / ".." / self.source_paths["a"].name
        self.document["cases"][1]["input"] = str(alias_path)
        self.document["cases"][1]["input_sha256"] = _sha256(self.input_payloads["a"])
        self._write_manifest()

        read_count = 0
        original_read_bytes = Path.read_bytes

        def count_source_reads(path: Path) -> bytes:
            nonlocal read_count
            if path.resolve() == self.source_paths["a"].resolve():
                read_count += 1
            return original_read_bytes(path)

        with mock.patch.object(Path, "read_bytes", count_source_reads):
            frozen = self.capture(("a", "b"))
        self.addCleanup(frozen.close)
        self.assertEqual(read_count, 1)
        self.assertEqual(
            frozen.cases_by_id["a"]["input"], frozen.cases_by_id["b"]["input"]
        )
        self.assertEqual(frozen.captured_input_bytes, len(self.input_payloads["a"]))

        self.document["cases"][1]["input_sha256"] = _sha256(b"another input")
        self._write_manifest()
        with self.assertRaisesRegex(frozen_inputs.FrozenInputError, "inconsistent"):
            self.capture(("a", "b"))

    def test_input_hash_format_and_mismatch_fail_before_snapshot_escape(self) -> None:
        """Clean partial snapshots on mismatch and reject malformed input hashes."""
        self.document["cases"][0]["input_sha256"] = "0" * 64
        self._write_manifest()
        real_temporary_directory = tempfile.TemporaryDirectory
        created_directories: list[Path] = []

        def track_temporary_directory(*args: object, **kwargs: object) -> object:
            temporary_directory = real_temporary_directory(*args, **kwargs)
            created_directories.append(Path(temporary_directory.name))
            return temporary_directory

        with (
            mock.patch.object(
                frozen_inputs.tempfile,
                "TemporaryDirectory",
                side_effect=track_temporary_directory,
            ),
            self.assertRaisesRegex(frozen_inputs.FrozenInputError, "mismatch"),
        ):
            self.capture(("a",))
        self.assertEqual(len(created_directories), 1)
        self.assertFalse(created_directories[0].exists())

        self.document["cases"][0]["input_sha256"] = "bad"
        self._write_manifest()
        with self.assertRaisesRegex(frozen_inputs.FrozenInputError, "malformed"):
            self.capture(("a",))


if __name__ == "__main__":
    unittest.main()
