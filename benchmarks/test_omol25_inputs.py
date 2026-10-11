"""Hardware-free source validation and conversion tests for OMol25 inputs."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import omol25_inputs
from tools.conformance import xtbloom_conformance as conformance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE_TOOLS = str(REPOSITORY_ROOT / "tools" / "conformance")
if CONFORMANCE_TOOLS not in sys.path:
    sys.path.insert(0, CONFORMANCE_TOOLS)

public_api = importlib.import_module("tools.conformance.xtbloom_public_api")


class OMol25InputConversionTests(unittest.TestCase):
    """Exercise strict source identity, order, units, and diagnostic-only output."""

    def test_direct_script_help_does_not_require_pythonpath(self) -> None:
        """Support the documented direct CLI as well as package module execution."""
        completed = subprocess.run(
            [sys.executable, str(Path(omol25_inputs.__file__)), "--help"],
            capture_output=True,
            text=True,
            check=False,
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--spin-channels", completed.stdout)

    def setUp(self) -> None:
        """Create fresh pinned input fixtures outside all repository worktrees."""
        self.temporary = tempfile.TemporaryDirectory(prefix="xtbloom-omol25-inputs-")
        self.root = Path(self.temporary.name)
        self.source_dir = self.root / "source"
        self.source_dir.mkdir()
        self.output_root = self.root / "converted"
        self.csv_path = self.source_dir / "canonical.csv"
        self.npz_path = self.source_dir / "canonical.npz"
        self.source_manifest_path = self.source_dir / "source_manifest.json"
        self.rows, self.arrays = self._fixture()
        self._write_sources(self.rows, self.arrays)

    def tearDown(self) -> None:
        """Remove only this test's temporary inputs and generated artifacts."""
        self.temporary.cleanup()

    @staticmethod
    def _sha(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _fixture(self) -> tuple[list[dict[str, str]], dict[str, np.ndarray[Any, Any]]]:
        """Create a small canonical-order ragged corpus spanning AO views."""
        atoms = [
            np.asarray([1], dtype="<i4"),
            np.full(39, 1, dtype="<i4"),
            np.full(70, 1, dtype="<i4"),
        ]
        positions = [
            np.asarray([[0.12345678901234567, -0.0, 1.25]], dtype="<f8"),
            np.asarray(
                [[index / 10, 0.0, -index / 7] for index in range(39)], dtype="<f8"
            ),
            np.asarray(
                [[index / 13, -index / 11, 0.5] for index in range(70)], dtype="<f8"
            ),
        ]
        ids = ["CO_small_fixture", "CO_medium_fixture", "CO_large_fixture"]
        charges = [0, 0, 0]
        multiplicities = [1, 2, 1]
        unpaired = [0, 1, 0]
        offsets = [0]
        flat_numbers: list[int] = []
        flat_positions: list[list[float]] = []
        rows: list[dict[str, str]] = []
        input_hashes = []
        basis_counts, _ = omol25_inputs.ao_grouping.load_gfn2_basis_ao_counts(
            omol25_inputs.GFN2_PARAMETERS
        )
        for index, case_id in enumerate(ids):
            atom_numbers = atoms[index]
            coordinates = positions[index]
            ao_count = omol25_inputs.ao_grouping.count_gfn2_aos(
                [int(value) for value in atom_numbers], basis_counts
            )
            atomic_hash = self._sha(atom_numbers.tobytes(order="C"))
            position_hash = self._sha(coordinates.tobytes(order="C"))
            input_hash = self._sha(f"fixture-input-{case_id}".encode("ascii"))
            input_hashes.append(input_hash)
            rows.append(
                {
                    "sample_set": "performance",
                    "configuration_id": case_id,
                    "charge": str(charges[index]),
                    "multiplicity": str(multiplicities[index]),
                    "unpaired_electrons": str(unpaired[index]),
                    "natoms": str(len(atom_numbers)),
                    "gfn2_n_ao": str(ao_count),
                    "coordinate_output_unit": "bohr",
                    "xtbloom_input_sha256": input_hash,
                    "atomic_numbers_sha256": atomic_hash,
                    "positions_bohr_sha256": position_hash,
                }
            )
            flat_numbers.extend(int(value) for value in atom_numbers)
            flat_positions.extend(coordinates.tolist())
            offsets.append(len(flat_numbers))
        arrays: dict[str, np.ndarray[Any, Any]] = {
            "format_version": np.asarray(omol25_inputs.NPZ_FORMAT_VERSION),
            "sample_ids": np.asarray(ids, dtype="<U32"),
            "offsets": np.asarray(offsets, dtype="<i8"),
            "atomic_numbers": np.asarray(flat_numbers, dtype="<i4"),
            "positions_bohr": np.asarray(flat_positions, dtype="<f8"),
            "charges": np.asarray(charges, dtype="<i4"),
            "multiplicities": np.asarray(multiplicities, dtype="<i4"),
            "unpaired_electrons": np.asarray(unpaired, dtype="<i4"),
            "natoms": np.asarray([len(value) for value in atoms], dtype="<i4"),
            "gfn2_n_ao": np.asarray([1, 39, 70], dtype="<i4"),
            "input_sha256": np.asarray(input_hashes, dtype="<U64"),
        }
        return rows, arrays

    def _write_sources(
        self,
        rows: list[dict[str, str]],
        arrays: dict[str, np.ndarray[Any, Any]],
        *,
        source_manifest: dict[str, Any] | None = None,
    ) -> None:
        """Write repeatable CSV, NPZ, and source-manifest test inputs."""
        columns = list(rows[0])
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(self.npz_path, **arrays)
        document = source_manifest or {
            "schema_version": omol25_inputs.SOURCE_MANIFEST_VERSION,
            "sampling": {
                "target_count": len(rows),
                "max_gfn2_n_ao": 180,
                "order": ["performance"],
            },
        }
        self.source_manifest_path.write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _convert(self, *, spin_channels: int = 2) -> omol25_inputs.ConversionResult:
        """Convert using freshly calculated caller pins for this fixture."""
        return omol25_inputs.convert_dataset(
            self.csv_path,
            self.npz_path,
            self.source_manifest_path,
            self.output_root,
            csv_sha256=omol25_inputs.sha256_file(self.csv_path),
            npz_sha256=omol25_inputs.sha256_file(self.npz_path),
            source_manifest_sha256=omol25_inputs.sha256_file(self.source_manifest_path),
            spin_channels=spin_channels,
        )

    def test_roundtrip_uses_full_precision_and_explicit_spin_not_multiplicity(
        self,
    ) -> None:
        """Keep exact bohr coordinates and apply the caller's common spin choice."""
        result = self._convert(spin_channels=2)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        cases = manifest["cases"]
        assembled = public_api.assemble_batch(result.manifest_path, manifest, cases)
        self.assertEqual(
            [case["id"] for case in cases],
            [row["configuration_id"] for row in self.rows],
        )
        self.assertEqual([case["spin_channels"] for case in cases], [2, 2, 2])
        self.assertEqual(cases[0]["multiplicity"], 1)
        self.assertEqual(cases[0]["spin_channels"], 2)
        self.assertEqual(
            assembled.atomic_numbers, self.arrays["atomic_numbers"].tolist()
        )
        self.assertEqual(
            assembled.positions, self.arrays["positions_bohr"].ravel().tolist()
        )
        self.assertEqual(assembled.spin_channels, [2, 2, 2])
        self.assertTrue(all(item.expected == {} for item in assembled.slices))
        for index, case in enumerate(cases):
            input_path = Path(case["input"])
            loaded = conformance.load_turbomole_coord(
                input_path, {"id": case["id"], "atom_count": case["atom_count"]}
            )
            self.assertEqual(
                loaded["atomic_numbers"],
                self.arrays["atomic_numbers"][
                    sum(self.arrays["natoms"][:index]) : sum(
                        self.arrays["natoms"][: index + 1]
                    )
                ].tolist(),
            )
            begin, end = self.arrays["offsets"][index : index + 2]
            expected_positions = self.arrays["positions_bohr"][begin:end].tolist()
            self.assertEqual(loaded["positions_bohr"], expected_positions)
        coord_bytes = Path(cases[0]["input"]).read_bytes()
        self.assertIn(b"0.12345678901234566 -0 1.25 h", coord_bytes)

    def test_ordered_canonical_ao_views_and_edge_lists(self) -> None:
        """Emit stable original-order IDs and exact generated-parameter AO bins."""
        result = self._convert()
        root = result.output_root
        self.assertEqual(
            (root / "id-lists/all.txt").read_text().splitlines(),
            [row["configuration_id"] for row in self.rows],
        )
        self.assertEqual(
            (root / "id-lists/small.txt").read_text().splitlines(), ["CO_small_fixture"]
        )
        self.assertEqual(
            (root / "id-lists/medium.txt").read_text().splitlines(),
            ["CO_medium_fixture"],
        )
        self.assertEqual(
            (root / "id-lists/large.txt").read_text().splitlines(), ["CO_large_fixture"]
        )
        self.assertEqual(
            (root / "id-lists/first256.txt").read_text().splitlines(),
            [row["configuration_id"] for row in self.rows],
        )
        self.assertEqual(
            (root / "id-lists/last256.txt").read_text().splitlines(),
            [row["configuration_id"] for row in self.rows],
        )

    def test_provenance_freezes_all_sources_and_converted_bytes_without_oracle(
        self,
    ) -> None:
        """Hash every converted artifact and label all scientific gates unavailable."""
        result = self._convert()
        root = result.output_root
        provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            provenance["source_inputs"]["csv"]["sha256"],
            omol25_inputs.sha256_file(self.csv_path),
        )
        self.assertEqual(
            provenance["source_inputs"]["npz"]["sha256"],
            omol25_inputs.sha256_file(self.npz_path),
        )
        self.assertEqual(
            provenance["source_inputs"]["source_manifest"]["sha256"],
            omol25_inputs.sha256_file(self.source_manifest_path),
        )
        for relative_path, expected_hash in provenance[
            "converted_files_sha256"
        ].items():
            self.assertEqual(
                omol25_inputs.sha256_file(root / relative_path), expected_hash
            )
        self.assertFalse(provenance["independent_reference_available"])
        self.assertFalse(
            provenance["qualification"]["scientific_correctness_qualified"]
        )
        self.assertFalse(manifest["qualification"]["performance_claim_eligible"])
        self.assertFalse((root / "canonical.npz").exists())
        placeholder_path = root / "reference-unavailable.json"
        placeholder = json.loads(placeholder_path.read_text(encoding="utf-8"))
        self.assertEqual(placeholder["properties"], {})
        self.assertFalse(placeholder["independent_reference_available"])
        self.assertTrue(
            all(
                case["oracle_role"] == "diagnostic-no-independent-reference"
                for case in manifest["cases"]
            )
        )
        self.assertTrue(
            all(
                not case["qualification"]["performance_claim_eligible"]
                for case in manifest["cases"]
            )
        )

    def test_source_molecule_and_group_ids_are_only_retained_when_present(self) -> None:
        """Copy exact source ID columns, never invent a molecular grouping."""
        result = self._convert()
        cases = json.loads(result.manifest_path.read_text(encoding="utf-8"))["cases"]
        self.assertNotIn("molecule_group_id", cases[0])
        self.assertNotIn("molecule_id", cases[0])

        self.output_root = self.root / "converted-with-groups"
        grouped_rows = [
            dict(
                row, molecule_group_id=f"group-{index // 2}", molecule_id=f"mol-{index}"
            )
            for index, row in enumerate(self.rows)
        ]
        self._write_sources(grouped_rows, self.arrays)
        grouped = self._convert()
        grouped_cases = json.loads(grouped.manifest_path.read_text(encoding="utf-8"))[
            "cases"
        ]
        self.assertEqual(
            [case["molecule_group_id"] for case in grouped_cases],
            ["group-0", "group-0", "group-1"],
        )
        self.assertEqual(
            [case["molecule_id"] for case in grouped_cases], ["mol-0", "mol-1", "mol-2"]
        )

    def test_missing_explicit_spin_option_is_a_cli_error_and_invalid_values_fail(
        self,
    ) -> None:
        """Require explicit one/two-channel selection without multiplicity inference."""
        parser = omol25_inputs.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--csv",
                    str(self.csv_path),
                    "--npz",
                    str(self.npz_path),
                    "--source-manifest",
                    str(self.source_manifest_path),
                    "--output-root",
                    str(self.output_root),
                    "--csv-sha256",
                    "a" * 64,
                    "--npz-sha256",
                    "b" * 64,
                    "--source-manifest-sha256",
                    "c" * 64,
                ]
            )
        with self.assertRaisesRegex(omol25_inputs.OMol25InputError, "spin_channels"):
            self._convert(spin_channels=0)

    def test_source_hash_pin_mismatch_fails_before_output(self) -> None:
        """Reject a changed or incorrectly pinned source byte stream."""
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "CSV SHA-256 mismatch"
        ):
            omol25_inputs.convert_dataset(
                self.csv_path,
                self.npz_path,
                self.source_manifest_path,
                self.output_root,
                csv_sha256="0" * 64,
                npz_sha256=omol25_inputs.sha256_file(self.npz_path),
                source_manifest_sha256=omol25_inputs.sha256_file(
                    self.source_manifest_path
                ),
                spin_channels=2,
            )
        self.assertFalse(self.output_root.exists())

    def test_csv_npz_per_id_coordinate_hash_mismatch_is_rejected(self) -> None:
        """Check source row hashes against actual little-endian NPZ contents."""
        arrays = dict(self.arrays)
        arrays["positions_bohr"] = arrays["positions_bohr"].copy()
        arrays["positions_bohr"][0, 0] += 0.01
        self._write_sources(self.rows, arrays)
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "positions_bohr SHA-256 mismatch"
        ):
            self._convert()

    def test_duplicate_and_unsafe_case_ids_are_rejected(self) -> None:
        """Prevent duplicate roster keys and path components in output names."""
        duplicate_rows = [dict(row) for row in self.rows]
        duplicate_rows[1]["configuration_id"] = duplicate_rows[0]["configuration_id"]
        self._write_sources(duplicate_rows, self.arrays)
        with self.assertRaisesRegex(omol25_inputs.OMol25InputError, "must be unique"):
            self._convert()

        self.output_root = self.root / "converted-unsafe"
        unsafe_rows = [dict(row) for row in self.rows]
        unsafe_rows[0]["configuration_id"] = "../escape"
        self._write_sources(unsafe_rows, self.arrays)
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "unsafe configuration_id"
        ):
            self._convert()

    def test_nonfinite_coordinates_and_bad_offset_extent_are_rejected(self) -> None:
        """Reject NaN geometry and ragged offsets that do not span the atoms."""
        arrays = dict(self.arrays)
        arrays["positions_bohr"] = arrays["positions_bohr"].copy()
        arrays["positions_bohr"][0, 0] = np.nan
        self._write_sources(self.rows, arrays)
        with self.assertRaisesRegex(omol25_inputs.OMol25InputError, "non-finite"):
            self._convert()

        self.output_root = self.root / "converted-bad-offset"
        arrays = dict(self.arrays)
        arrays["offsets"] = arrays["offsets"].copy()
        arrays["offsets"][-1] -= 1
        self._write_sources(self.rows, arrays)
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "offsets must span"
        ):
            self._convert()

    def test_npz_dtype_mismatch_and_csv_charge_disagreement_are_rejected(self) -> None:
        """Enforce the fixed-width archive schema and per-row integer identity."""
        arrays = dict(self.arrays)
        arrays["atomic_numbers"] = arrays["atomic_numbers"].astype("<i8")
        self._write_sources(self.rows, arrays)
        with self.assertRaisesRegex(omol25_inputs.OMol25InputError, "dtype"):
            self._convert()

        self.output_root = self.root / "converted-charge-mismatch"
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["charge"] = "1"
        self._write_sources(changed_rows, self.arrays)
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "CSV charge disagrees"
        ):
            self._convert()

    def test_source_manifest_order_and_roster_count_are_enforced(self) -> None:
        """Refuse incomplete rosters or order contradicting source metadata."""
        manifest = {
            "schema_version": 2,
            "sampling": {
                "target_count": 4,
                "max_gfn2_n_ao": 180,
                "order": ["performance"],
            },
        }
        self._write_sources(self.rows, self.arrays, source_manifest=manifest)
        with self.assertRaisesRegex(omol25_inputs.OMol25InputError, "pins 4"):
            self._convert()

        self.output_root = self.root / "converted-order"
        manifest = {
            "schema_version": 2,
            "sampling": {
                "target_count": 3,
                "max_gfn2_n_ao": 180,
                "order": ["performance", "holdout"],
            },
        }
        changed_rows = [dict(row) for row in self.rows]
        changed_rows[0]["sample_set"] = "holdout"
        changed_rows[1]["sample_set"] = "performance"
        self._write_sources(changed_rows, self.arrays, source_manifest=manifest)
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "disagrees with source manifest"
        ):
            self._convert()

    def test_output_inside_repository_is_rejected(self) -> None:
        """Protect the source checkout from generated input and placeholder bytes."""
        with self.assertRaisesRegex(
            omol25_inputs.OMol25InputError, "outside the xTBloom repository"
        ):
            omol25_inputs._check_external_output_root(
                REPOSITORY_ROOT / "build" / "omol25-test"
            )


if __name__ == "__main__":
    unittest.main()
