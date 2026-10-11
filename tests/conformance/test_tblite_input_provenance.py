"""Offline tests for hash-pinned tblite input execution and provenance."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools/conformance/xtbloom_conformance.py"
MANIFEST_PATH = ROOT / "data/conformance/manifest.json"
SPEC = importlib.util.spec_from_file_location("tblite_input_conformance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
CONFORMANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFORMANCE)

RunSideEffect = Callable[..., subprocess.CompletedProcess[str]]


class TbliteInputProvenanceTests(unittest.TestCase):
    """Exercise input-byte gates without starting a reference implementation."""

    def setUp(self) -> None:
        """Copy canonical case metadata into a private temporary test bundle."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output_dir = self.root / "generated"
        self.executable = self.root / "fake-tblite"
        self.executable.write_bytes(b"fake executable identity\n")
        self.executable.chmod(0o755)

        canonical = CONFORMANCE.load_json(MANIFEST_PATH)
        cases_by_id = {case["id"]: case for case in canonical["cases"]}
        self.input_paths: dict[str, Path] = {}
        selected_cases = []
        for case_id in ("h3_plus", "ketene"):
            case = copy.deepcopy(cases_by_id[case_id])
            source_path = ROOT / case["input"]
            input_bytes = source_path.read_bytes()
            input_path = self.root / "inputs" / source_path.name
            input_path.parent.mkdir(exist_ok=True)
            input_path.write_bytes(input_bytes)
            self.input_paths[case_id] = input_path
            case["input"] = str(input_path)
            case["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
            selected_cases.append(case)

        self.manifest = copy.deepcopy(canonical)
        self.manifest["cases"] = selected_cases
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest()

    def write_manifest(self) -> None:
        """Persist the fixture manifest after an individual test mutation."""
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @contextmanager
    def patched_runner(
        self, run_side_effect: RunSideEffect | None = None
    ) -> Iterator[dict[str, mock.MagicMock]]:
        """Stub version/runtime discovery and record every would-be CLI call."""
        with (
            mock.patch.object(
                CONFORMANCE.shutil, "which", return_value=str(self.executable)
            ),
            mock.patch.object(
                CONFORMANCE,
                "executable_version",
                return_value="tblite version 0.7.0\n",
            ) as version,
            mock.patch.object(
                CONFORMANCE,
                "discover_tblite_runtime",
                return_value={"libtblite": {"status": "unresolved"}},
            ) as runtime,
            mock.patch.object(
                CONFORMANCE.subprocess, "run", side_effect=run_side_effect
            ) as run,
        ):
            yield {"version": version, "runtime": runtime, "run": run}

    def fake_cli(
        self,
        captured: list[dict[str, Any]],
        *,
        mutate: str | None = None,
        write_output: bool = True,
    ) -> RunSideEffect:
        """Return a fake successful tblite call that can mutate either input."""

        def invoke(
            command: list[str], *, cwd: Path, **options: object
        ) -> subprocess.CompletedProcess[str]:
            input_path = Path(command[1])
            case_id = input_path.stem
            original_path = self.input_paths[case_id]
            input_bytes = input_path.read_bytes()
            captured.append(
                {
                    "command": list(command),
                    "cwd": cwd,
                    "input_path": input_path,
                    "input_bytes": input_bytes,
                    "original_path": original_path,
                }
            )

            output_path = Path(command[command.index("--json") + 1])
            if write_output:
                atom_count = next(
                    case["atom_count"]
                    for case in self.manifest["cases"]
                    if case["id"] == case_id
                )
                raw_output = {
                    "energy": -1.2345,
                    "gradient": [0.01 * (index + 1) for index in range(3 * atom_count)],
                }
                output_path.write_text(json.dumps(raw_output), encoding="utf-8")

            if mutate == "original":
                original_path.write_bytes(input_bytes + b"source mutation\n")
            elif mutate == "copy":
                input_path.write_bytes(input_bytes + b"private-copy mutation\n")
            elif mutate == "remove_original":
                original_path.unlink()
            elif mutate == "remove_copy":
                input_path.unlink()

            return subprocess.CompletedProcess(command, 0, stdout="fake success")

        return invoke

    def generate(self, case_ids: list[str] | None = None) -> None:
        """Run only the generator orchestration against the fake CLI boundary."""
        CONFORMANCE.generate_with_tblite(
            self.manifest_path,
            self.executable,
            self.output_dir,
            case_ids,
        )

    def test_generates_from_private_byte_identical_input_and_records_provenance(
        self,
    ) -> None:
        """The fake CLI sees captured bytes and output pins describe those bytes."""
        captured: list[dict[str, Any]] = []
        expected_bytes = self.input_paths["h3_plus"].read_bytes()
        with (
            self.patched_runner(self.fake_cli(captured)),
            redirect_stdout(StringIO()),
        ):
            self.generate(["h3_plus"])

        self.assertEqual(len(captured), 1)
        invocation = captured[0]
        original_path = self.input_paths["h3_plus"]
        private_path = invocation["input_path"]
        command = invocation["command"]
        self.assertNotEqual(private_path, original_path)
        self.assertEqual(private_path.name, original_path.name)
        self.assertEqual(private_path.parent, invocation["cwd"])
        self.assertEqual(invocation["input_bytes"], expected_bytes)
        self.assertNotIn(str(original_path), command)
        self.assertEqual(
            command[2:7], ["--no-restart", "--method", "gfn2", "--acc", "0.0001"]
        )
        self.assertEqual(command[command.index("--charge") + 1], "+1")
        self.assertNotIn("--spin", command)

        generated = json.loads(
            (self.output_dir / "h3_plus.json").read_text(encoding="utf-8")
        )
        provenance = generated["provenance"]
        self.assertEqual(provenance["input"], str(original_path))
        self.assertEqual(
            provenance["input_sha256"], hashlib.sha256(expected_bytes).hexdigest()
        )
        self.assertEqual(provenance["input_byte_count"], len(expected_bytes))

    def test_selected_cases_use_isolated_working_directories(self) -> None:
        """Each successful case gets its own cwd and raw tblite output path."""
        captured: list[dict[str, Any]] = []
        with (
            self.patched_runner(self.fake_cli(captured)),
            redirect_stdout(StringIO()),
        ):
            self.generate(["h3_plus", "ketene"])

        self.assertEqual(len(captured), 2)
        working_directories = [invocation["cwd"] for invocation in captured]
        self.assertEqual(len(set(working_directories)), 2)
        for invocation in captured:
            case_id = invocation["input_path"].stem
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    invocation["input_bytes"], self.input_paths[case_id].read_bytes()
                )
                self.assertEqual(invocation["input_path"].parent, invocation["cwd"])
                output_path = Path(
                    invocation["command"][invocation["command"].index("--json") + 1]
                )
                self.assertEqual(output_path.parent, invocation["cwd"])
                self.assertEqual(output_path.name, "tblite.json")
                self.assertTrue((self.output_dir / f"{case_id}.json").is_file())

    def test_invalid_input_pin_fails_before_any_subprocess_or_probe(self) -> None:
        """A bad selected input pin is rejected before version/runtime access."""
        self.manifest["cases"][0]["input_sha256"] = "0" * 64
        self.write_manifest()
        with (
            self.patched_runner() as patched,
            self.assertRaisesRegex(CONFORMANCE.ConformanceError, "h3_plus"),
        ):
            self.generate(["h3_plus"])

        patched["version"].assert_not_called()
        patched["runtime"].assert_not_called()
        patched["run"].assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_wrong_later_selected_case_prevents_every_subprocess(self) -> None:
        """Validate the whole selected cohort before probing or running case one."""
        self.manifest["cases"][1]["input_sha256"] = "f" * 64
        self.write_manifest()
        with (
            self.patched_runner() as patched,
            self.assertRaisesRegex(CONFORMANCE.ConformanceError, "ketene"),
        ):
            self.generate(["h3_plus", "ketene"])

        patched["version"].assert_not_called()
        patched["runtime"].assert_not_called()
        patched["run"].assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def assert_mutation_is_not_published(self, mutation: str) -> None:
        """Require post-success input mutation checks before golden publication."""
        captured: list[dict[str, Any]] = []
        with (
            self.patched_runner(self.fake_cli(captured, mutate=mutation)),
            redirect_stdout(StringIO()),
            self.assertRaisesRegex(
                CONFORMANCE.ConformanceError, "input bytes changed during generation"
            ),
        ):
            self.generate(["h3_plus"])

        self.assertEqual(len(captured), 1)
        self.assertFalse((self.output_dir / "h3_plus.json").exists())

    def test_source_input_mutation_during_successful_subprocess_is_rejected(
        self,
    ) -> None:
        """A source edit concurrent with tblite cannot publish that case."""
        self.assert_mutation_is_not_published("original")

    def test_private_copy_mutation_during_successful_subprocess_is_rejected(
        self,
    ) -> None:
        """A modified materialized input cannot publish a golden either."""
        self.assert_mutation_is_not_published("copy")

    def assert_removed_input_is_not_published(self, mutation: str) -> None:
        """Preserve an existing golden if an input disappears during execution."""
        self.output_dir.mkdir()
        destination = self.output_dir / "h3_plus.json"
        previous_golden = b'{"case_id":"previous golden"}\n'
        destination.write_bytes(previous_golden)
        captured: list[dict[str, Any]] = []
        with (
            self.patched_runner(self.fake_cli(captured, mutate=mutation)),
            redirect_stdout(StringIO()),
            self.assertRaises(OSError),
        ):
            self.generate(["h3_plus"])

        self.assertEqual(len(captured), 1)
        self.assertEqual(destination.read_bytes(), previous_golden)

    def test_removed_source_input_during_subprocess_does_not_overwrite_golden(
        self,
    ) -> None:
        """A disappeared original input is rejected before golden publication."""
        self.assert_removed_input_is_not_published("remove_original")

    def test_removed_private_copy_during_subprocess_does_not_overwrite_golden(
        self,
    ) -> None:
        """A disappeared private input is rejected before golden publication."""
        self.assert_removed_input_is_not_published("remove_copy")

    def test_missing_source_input_does_not_overwrite_existing_golden(self) -> None:
        """Preflight I/O failure leaves prior output untouched and runs no CLI."""
        self.output_dir.mkdir()
        destination = self.output_dir / "h3_plus.json"
        previous_golden = b'{"case_id":"previous golden"}\n'
        destination.write_bytes(previous_golden)
        self.input_paths["h3_plus"].unlink()

        with self.patched_runner() as patched, self.assertRaises(OSError):
            self.generate(["h3_plus"])

        patched["version"].assert_not_called()
        patched["runtime"].assert_not_called()
        patched["run"].assert_not_called()
        self.assertEqual(destination.read_bytes(), previous_golden)

    def test_stale_output_is_never_used_when_fake_cli_writes_no_result(self) -> None:
        """A pre-existing golden cannot stand in for this invocation's raw output."""
        self.output_dir.mkdir()
        stale_bytes = b'{"case_id":"stale result"}\n'
        stale_path = self.output_dir / "h3_plus.json"
        stale_path.write_bytes(stale_bytes)
        captured: list[dict[str, Any]] = []
        with (
            self.patched_runner(self.fake_cli(captured, write_output=False)),
            redirect_stdout(StringIO()),
            self.assertRaises(CONFORMANCE.ConformanceError),
        ):
            self.generate(["h3_plus"])

        self.assertEqual(len(captured), 1)
        self.assertEqual(stale_path.read_bytes(), stale_bytes)


if __name__ == "__main__":
    unittest.main()
