"""Offline checks for hash-pinned historical alkane input materialization."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from benchmarks import historical_alkane_inputs as historical
from benchmarks import run

if TYPE_CHECKING:
    from typing import Any


class HistoricalAlkaneInputsTest(unittest.TestCase):
    """Exercise validation and assembler construction without loading a model."""

    def setUp(self) -> None:
        """Isolate input, output, and staging paths for every test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_path = self.root / "historical-inputs.json"
        self.output_path = self.root / "materialized"
        self.document = self.make_document()
        self.write_document(self.document)

    def tearDown(self) -> None:
        """Remove only the test-owned temporary directory."""
        self.temporary.cleanup()

    @staticmethod
    def make_document() -> dict[str, Any]:
        """Build synthetic repeated slots with one distinct charge/spin case."""
        workloads = []
        for name, natoms in historical.EXPECTED_WORKLOADS:
            numbers_per_slot = [6, *([1] * (natoms - 1))]
            atomic_numbers = numbers_per_slot * 256
            one_geometry = [0.5606069543851774, -1.982044895130482, 1.25] * natoms
            positions = one_geometry * 256
            charges = [0] * 256
            unpaired = [0] * 256
            spin_channels = [1] * 256
            charges[7] = -1
            unpaired[7] = 1
            spin_channels[7] = 2
            workloads.append(
                {
                    "name": name,
                    "natoms": natoms,
                    "batch_size": 256,
                    "seed": natoms * 1000,
                    "perturb_sigma_bohr": 0.02,
                    "atom_offsets": [slot * natoms for slot in range(257)],
                    "atomic_numbers": atomic_numbers,
                    "positions_bohr": positions,
                    "molecular_charges": charges,
                    "unpaired_electrons": unpaired,
                    "spin_channels": spin_channels,
                }
            )
        return {
            "resource_stub": "synthetic input fixture; not a historical corpus",
            "source_commit": "c9c0a432947f122d25cb91d0a4624af0a3e761ad",
            "source_hashes": {"fixture.py": "a" * 64},
            "workloads": workloads,
        }

    def write_document(self, document: object) -> str:
        """Write exact fixture bytes and return their mandatory input pin."""
        raw = json.dumps(document, allow_nan=True, separators=(",", ":")).encode(
            "utf-8"
        )
        self.source_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def materialize(self, expected_sha256: str | None = None) -> dict[str, object]:
        """Invoke the offline producer with an explicit or current fixture pin."""
        if expected_sha256 is None:
            expected_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        return historical.materialize(
            self.source_path, self.output_path, expected_sha256
        )

    @unittest.skipUnless(
        sys.platform == "linux" or os.name == "nt",
        "atomic no-replace publication requires Linux or Windows",
    )
    def test_valid_32_and_62_workloads_preserve_slots_identity_and_values(self) -> None:
        """Verify all slots and assembled arrays without treating goldens as data."""
        report = self.materialize()
        manifest_path = self.output_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(report["case_count"], 512)
        self.assertEqual(manifest["source"]["sha256"], report["source_sha256"])
        self.assertEqual(
            manifest["source"]["coordinate_unit_conversion"],
            "none; source and output are bohr",
        )
        self.assertEqual(manifest["source"]["atom_order_conversion"], "none")
        self.assertIn("slot_deduplication", manifest["source"])
        self.assertEqual(manifest["eligibility"]["correctness"], False)
        self.assertEqual(manifest["eligibility"]["performance"], False)
        self.assertFalse(
            any(case["independent_reference_available"] for case in manifest["cases"])
        )
        self.assertFalse(
            any(case["correctness_eligible"] for case in manifest["cases"])
        )
        self.assertFalse(
            any(case["performance_eligible"] for case in manifest["cases"])
        )
        self.assertTrue(all(case["geometry_sha256"] for case in manifest["cases"]))

        self.assertEqual(len({case["id"] for case in manifest["cases"]}), 512)
        for workload, natoms in historical.EXPECTED_WORKLOADS:
            cases = [
                case
                for case in manifest["cases"]
                if case["source_workload"] == workload
            ]
            self.assertEqual(len(cases), 256)
            self.assertEqual(cases[0]["id"], f"historical-{workload}-slot-0000")
            self.assertEqual(cases[-1]["id"], f"historical-{workload}-slot-0255")
            self.assertEqual([case["source_slot"] for case in cases], list(range(256)))
            self.assertEqual(cases[0]["geometry_sha256"], cases[-1]["geometry_sha256"])
            self.assertEqual(cases[7]["molecular_charge"], -1)
            self.assertIs(type(cases[7]["molecular_charge"]), int)
            self.assertEqual(cases[7]["unpaired_electrons"], 1)
            self.assertEqual(cases[7]["spin_channels"], 2)
            self.assertEqual(cases[7]["atom_count"], natoms)

            original = next(
                item for item in self.document["workloads"] if item["name"] == workload
            )
            coord_path = self.output_path / cases[7]["input"]
            parsed = run.conformance.load_turbomole_coord(coord_path, cases[7])
            begin = 7 * natoms
            expected_numbers = original["atomic_numbers"][begin : begin + natoms]
            expected_positions = original["positions_bohr"][
                3 * begin : 3 * (begin + natoms)
            ]
            self.assertEqual(parsed["atomic_numbers"], expected_numbers)
            self.assertEqual(
                [value for xyz in parsed["positions_bohr"] for value in xyz],
                expected_positions,
            )
            self.assertEqual(parsed["positions_bohr"][0][2], 1.25)

        for workload_index, (workload_name, natoms) in enumerate(
            historical.EXPECTED_WORKLOADS
        ):
            ordered_cases = historical.select_workload_cases(manifest, workload_name)
            self.assertEqual(len(ordered_cases), 256)
            self.assertEqual(
                [case["source_slot"] for case in ordered_cases], list(range(256))
            )
            self.assertEqual(
                [case["id"] for case in ordered_cases],
                [f"historical-{workload_name}-slot-{slot:04d}" for slot in range(256)],
            )
            source_workload = self.document["workloads"][workload_index]
            storage = run.public_api.assemble_batch(
                manifest_path, manifest, ordered_cases
            )
            self.assertEqual(
                storage.atom_offsets, [slot * natoms for slot in range(257)]
            )
            self.assertEqual(storage.atomic_numbers, source_workload["atomic_numbers"])
            self.assertEqual(
                storage.positions,
                [float(value) for value in source_workload["positions_bohr"]],
            )
            self.assertEqual(
                storage.molecular_charges,
                [float(value) for value in source_workload["molecular_charges"]],
            )
            self.assertEqual(
                storage.unpaired_electrons, source_workload["unpaired_electrons"]
            )
            self.assertEqual(storage.spin_channels, source_workload["spin_channels"])
            self.assertTrue(all(item.expected == {} for item in storage.slices))

        duplicate_manifest = dict(manifest)
        duplicate_manifest["cases"] = [*manifest["cases"], manifest["cases"][0]]
        with self.assertRaisesRegex(
            historical.HistoricalInputError, "duplicate case ID"
        ):
            historical.select_workload_cases(duplicate_manifest, "alkane32-b256")

        checksum_rows = (
            (self.output_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        )
        checksum_map = {
            line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in checksum_rows
        }
        self.assertEqual(
            checksum_map["manifest.json"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        for case in manifest["cases"]:
            input_relative = (
                Path(case["input"]).relative_to(self.output_path).as_posix()
            )
            golden_relative = (
                Path(case["golden"]).relative_to(self.output_path).as_posix()
            )
            self.assertEqual(checksum_map[input_relative], case["input_sha256"])
            self.assertEqual(checksum_map[golden_relative], case["golden_sha256"])

    def test_bad_hash_fails_before_creating_output(self) -> None:
        """A mismatched pin must not create an output path."""
        with self.assertRaisesRegex(
            historical.HistoricalInputError, "SHA-256 mismatch"
        ):
            self.materialize("0" * 64)
        self.assertFalse(self.output_path.exists())

    def test_malformed_sizes_nonfinite_values_and_invalid_tags_fail_closed(
        self,
    ) -> None:
        """Reject hostile descriptors before staging any output."""
        mutations = (
            (
                "atom offsets",
                lambda doc: doc["workloads"][0]["atom_offsets"].__setitem__(1, 31),
            ),
            ("position size", lambda doc: doc["workloads"][0]["positions_bohr"].pop()),
            (
                "position finite",
                lambda doc: doc["workloads"][0]["positions_bohr"].__setitem__(
                    0, math.inf
                ),
            ),
            (
                "charge finite",
                lambda doc: doc["workloads"][0]["molecular_charges"].__setitem__(
                    0, math.nan
                ),
            ),
            (
                "spin tag",
                lambda doc: doc["workloads"][0]["spin_channels"].__setitem__(0, 3),
            ),
            (
                "unpaired tag",
                lambda doc: doc["workloads"][0]["unpaired_electrons"].__setitem__(
                    0, True
                ),
            ),
            (
                "atomic tag",
                lambda doc: doc["workloads"][0]["atomic_numbers"].__setitem__(0, 0),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                self.output_path = self.root / f"output-{label.replace(' ', '-')}"
                document = self.make_document()
                mutate(document)
                expected_sha256 = self.write_document(document)
                with self.assertRaises(historical.HistoricalInputError):
                    self.materialize(expected_sha256)
                self.assertFalse(self.output_path.exists())

    def test_duplicate_workload_ids_are_rejected(self) -> None:
        """Do not collapse or relabel repeated workload names."""
        document = self.make_document()
        document["workloads"][1]["name"] = document["workloads"][0]["name"]
        expected_sha256 = self.write_document(document)
        with self.assertRaisesRegex(historical.HistoricalInputError, "names/order"):
            self.materialize(expected_sha256)
        self.assertFalse(self.output_path.exists())

    def test_duplicate_json_keys_are_rejected(self) -> None:
        """Ambiguous JSON must not silently replace the earlier field."""
        raw = b'{"resource_stub":"a","resource_stub":"b"}'
        self.source_path.write_bytes(raw)
        expected_sha256 = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(historical.HistoricalInputError, "duplicate JSON"):
            self.materialize(expected_sha256)
        self.assertFalse(self.output_path.exists())

    def test_failed_staging_leaves_no_partial_output_or_staging_directory(self) -> None:
        """A write failure leaves neither publication nor owned staging behind."""
        expected_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        with (
            mock.patch.object(
                historical, "_write_bytes", side_effect=OSError("injected failure")
            ),
            self.assertRaisesRegex(
                historical.HistoricalInputError, "publish generated inputs"
            ),
        ):
            self.materialize(expected_sha256)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])

    def test_existing_output_is_never_overwritten(self) -> None:
        """Preserve user files at an existing output path."""
        self.output_path.mkdir()
        sentinel = self.output_path / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(historical.HistoricalInputError, "must be new"):
            self.materialize()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    @unittest.skipUnless(sys.platform == "linux", "Linux renameat2 publication path")
    def test_empty_output_created_in_publication_race_is_not_replaced(self) -> None:
        """An empty destination arriving immediately before rename is preserved."""
        expected_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        rename = historical._rename_linux_noreplace

        def create_racing_directory(stage: Path, output: Path) -> None:
            output.mkdir()
            rename(stage, output)

        with (
            mock.patch.object(
                historical,
                "_rename_linux_noreplace",
                side_effect=create_racing_directory,
            ),
            self.assertRaises(historical.HistoricalInputError),
        ):
            self.materialize(expected_sha256)
        self.assertTrue(self.output_path.is_dir())
        self.assertEqual(list(self.output_path.iterdir()), [])
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])

    @unittest.skipUnless(sys.platform == "linux", "Linux renameat2 publication path")
    def test_user_directory_at_publication_survives_failure(self) -> None:
        """Publication and cleanup never remove a competing directory's contents."""
        expected_sha256 = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        rename = historical._rename_linux_noreplace

        def add_user_content_then_rename(stage: Path, output: Path) -> None:
            output.mkdir()
            (output / "user-data.txt").write_text("keep", encoding="utf-8")
            rename(stage, output)

        with (
            mock.patch.object(
                historical,
                "_rename_linux_noreplace",
                side_effect=add_user_content_then_rename,
            ),
            self.assertRaises(historical.HistoricalInputError),
        ):
            self.materialize(expected_sha256)
        self.assertEqual(
            (self.output_path / "user-data.txt").read_text(encoding="utf-8"),
            "keep",
        )
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])

    @unittest.skipUnless(sys.platform == "linux", "Linux renameat2 publication path")
    def test_missing_atomic_primitive_fails_closed(self) -> None:
        """An old libc must not trigger a check-then-rename fallback."""
        with (
            mock.patch.object(historical.ctypes, "CDLL", return_value=object()),
            self.assertRaisesRegex(historical.HistoricalInputError, "libc renameat2"),
        ):
            self.materialize()
        self.assertFalse(self.output_path.exists())
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])

    @unittest.skipUnless(sys.platform == "linux", "Linux renameat2 publication path")
    def test_unsupported_filesystem_fails_closed(self) -> None:
        """A rejected no-replace flag leaves no published or reserved output."""
        rename = mock.Mock(return_value=-1)
        library = mock.Mock(renameat2=rename)
        with (
            mock.patch.object(historical.ctypes, "CDLL", return_value=library),
            mock.patch.object(
                historical.ctypes, "get_errno", return_value=errno.EINVAL
            ),
            self.assertRaises(historical.HistoricalInputError),
        ):
            self.materialize()
        self.assertEqual(rename.call_args.args[-1], 1)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])

    @unittest.skipIf(os.name == "nt", "Windows uses its native no-replace rename")
    def test_unsupported_platform_fails_closed(self) -> None:
        """Other POSIX platforms require a reviewed atomic primitive first."""
        with (
            mock.patch.object(historical.sys, "platform", "unsupported"),
            self.assertRaisesRegex(
                historical.HistoricalInputError, "Linux and Windows"
            ),
        ):
            self.materialize()
        self.assertFalse(self.output_path.exists())
        self.assertEqual(list(self.root.glob(".materialized.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
