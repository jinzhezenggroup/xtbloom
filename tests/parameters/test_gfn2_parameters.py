"""Regression tests for the generated GFN2-xTB parameter bundle."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import tomllib

REPOSITORY = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY / "data" / "parameters"
GENERATOR_PATH = REPOSITORY / "tools" / "parameters" / "generate_gfn2.py"

SPEC = importlib.util.spec_from_file_location("gpuxtb_generate_gfn2", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class Gfn2ParameterTests(unittest.TestCase):
    """Verify canonical GFN2 artifacts and their regeneration tooling."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the canonical source, normalized data, and manifest once."""
        cls.raw_bytes = (DATA_DIR / GENERATOR.RAW_FILENAME).read_bytes()
        cls.raw = tomllib.loads(cls.raw_bytes.decode("utf-8"))
        cls.normalized = json.loads(
            (DATA_DIR / GENERATOR.JSON_FILENAME).read_text(encoding="utf-8")
        )
        cls.manifest = json.loads(
            (DATA_DIR / GENERATOR.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )

    def test_normalized_data_round_trips_entire_tblite_export(self) -> None:
        """Preserve every supported value from the tblite export."""
        # normalize_export validates every supported table and rejects unknown
        # fields, so equality proves no accepted tblite value was dropped.
        self.assertEqual(GENERATOR.normalize_export(self.raw), self.normalized)
        self.assertEqual(self.normalized["schema_version"], 1)
        self.assertEqual(self.normalized["method"], "gfn2-xtb")

    def test_all_parameter_families_are_present(self) -> None:
        """Retain every required GFN2 parameter family and default."""
        self.assertEqual(
            set(self.normalized),
            {
                "schema_version",
                "method",
                "meta",
                "hamiltonian",
                "dispersion",
                "repulsion",
                "charge",
                "thirdorder",
                "multipole",
                "elements",
            },
        )
        self.assertEqual(self.normalized["hamiltonian"]["pair_scale_default"], 1.0)
        # GFN2 initializes all 86x86 pair scales to one; the export therefore
        # correctly contains no non-default pair overrides.
        self.assertEqual(self.normalized["hamiltonian"]["pair_scale_overrides"], [])

    def test_shell_pair_scales_follow_tblite_load_defaults(self) -> None:
        """Reproduce tblite defaults for omitted shell-pair scales."""
        shell_scales = {
            tuple(entry["angular_momenta"]): entry["value"]
            for entry in self.normalized["hamiltonian"]["shell_pair_scale"]
        }
        # tblite omits sp from its export because it equals the loader's
        # arithmetic-mean default: (ss + pp) / 2 = (1.85 + 2.23) / 2.
        self.assertNotIn("sp", self.raw["hamiltonian"]["xtb"]["shell"])
        self.assertEqual(shell_scales[(0, 1)], 2.04)
        # GFN2 explicitly overrides the other two cross terms to 2.0.
        self.assertEqual(shell_scales[(0, 2)], 2.0)
        self.assertEqual(shell_scales[(1, 2)], 2.0)

        defaulted = copy.deepcopy(self.raw)
        del defaulted["hamiltonian"]["xtb"]["shell"]["sd"]
        defaulted_scales = {
            tuple(entry["angular_momenta"]): entry["value"]
            for entry in GENERATOR.normalize_export(defaulted)["hamiltonian"][
                "shell_pair_scale"
            ]
        }
        self.assertEqual(defaulted_scales[(0, 2)], 2.04)

    def test_noncanonical_shell_pair_name_is_rejected(self) -> None:
        """Reject reversed or otherwise noncanonical shell-pair names."""
        invalid = copy.deepcopy(self.raw)
        invalid["hamiltonian"]["xtb"]["shell"]["ps"] = 9.0
        with self.assertRaisesRegex(GENERATOR.ParameterError, "unsupported fields"):
            GENERATOR.normalize_export(invalid)

    def test_supported_element_and_shell_boundaries(self) -> None:
        """Cover supported element and shell boundaries in normalized data."""
        elements = self.normalized["elements"]
        self.assertEqual(len(elements), 86)
        self.assertEqual(
            (elements[0]["atomic_number"], elements[0]["symbol"]), (1, "H")
        )
        self.assertEqual(
            (elements[-1]["atomic_number"], elements[-1]["symbol"]), (86, "Rn")
        )
        self.assertEqual(elements[0]["shells"][0]["principal_quantum_number"], 1)
        self.assertEqual(elements[0]["shells"][0]["angular_momentum"], 0)
        self.assertEqual(elements[-1]["shells"][-1]["angular_momentum"], 2)
        self.assertTrue(all(1 <= len(element["shells"]) <= 3 for element in elements))

    def test_manifest_records_and_verifies_provenance(self) -> None:
        """Match manifest provenance and hashes to the canonical artifacts."""
        self.assertEqual(
            self.manifest["source"]["license"]["spdx"], "LGPL-3.0-or-later"
        )
        self.assertRegex(self.manifest["source"]["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(
            self.manifest["source"]["parameter_sources_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertIn("tblite version 0.7.0", self.manifest["exporter"]["version"])
        for filename, metadata in self.manifest["outputs"].items():
            content = (DATA_DIR / filename).read_bytes()
            self.assertEqual(len(content), metadata["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), metadata["sha256"])

    def test_offline_regeneration_is_byte_deterministic(self) -> None:
        """Regenerate every parameter artifact byte-for-byte offline."""
        provenance = {
            "source": self.manifest["source"],
            "exporter": self.manifest["exporter"],
        }
        generated = GENERATOR.build_artifacts(self.raw_bytes, provenance)
        for filename, content in generated.items():
            self.assertEqual(content, (DATA_DIR / filename).read_bytes(), filename)

    def test_check_mode_detects_a_stale_generated_file(self) -> None:
        """Make check mode reject a stale generated header."""
        provenance = {
            "source": self.manifest["source"],
            "exporter": self.manifest["exporter"],
        }
        generated = GENERATOR.build_artifacts(self.raw_bytes, provenance)
        with tempfile.TemporaryDirectory(
            prefix="gpuxtb-stale-parameter-test-"
        ) as directory:
            output_dir = Path(directory)
            GENERATOR.write_or_check(output_dir, generated, check=False)
            (output_dir / GENERATOR.HEADER_FILENAME).write_bytes(b"stale\n")
            with self.assertRaisesRegex(GENERATOR.ParameterError, "gfn2.hpp"):
                GENERATOR.write_or_check(output_dir, generated, check=True)

    def test_unknown_element_is_rejected(self) -> None:
        """Reject elements beyond the supported GFN2 range."""
        invalid = copy.deepcopy(self.raw)
        invalid["element"]["Og"] = copy.deepcopy(invalid["element"]["Rn"])
        with self.assertRaisesRegex(GENERATOR.ParameterError, "unsupported fields"):
            GENERATOR.normalize_export(invalid)

    def test_missing_element_is_rejected(self) -> None:
        """Reject an export missing a required supported element."""
        invalid = copy.deepcopy(self.raw)
        del invalid["element"]["H"]
        with self.assertRaisesRegex(GENERATOR.ParameterError, "missing fields"):
            GENERATOR.normalize_export(invalid)

    def test_shell_vector_length_mismatch_is_rejected(self) -> None:
        """Reject inconsistent per-shell vector lengths."""
        invalid = copy.deepcopy(self.raw)
        invalid["element"]["C"]["levels"].pop()
        with self.assertRaisesRegex(GENERATOR.ParameterError, "exactly 2 values"):
            GENERATOR.normalize_export(invalid)

    def test_unknown_parameter_is_rejected_instead_of_dropped(self) -> None:
        """Reject unknown parameters instead of silently dropping them."""
        invalid = copy.deepcopy(self.raw)
        invalid["multipole"]["damped"]["future_parameter"] = 1.0
        with self.assertRaisesRegex(GENERATOR.ParameterError, "unsupported fields"):
            GENERATOR.normalize_export(invalid)

    def test_unsupported_algorithm_selector_is_rejected(self) -> None:
        """Reject algorithm selectors unsupported by gpuxtb."""
        invalid = copy.deepcopy(self.raw)
        invalid["charge"]["effective"]["average"] = "geometric"
        with self.assertRaisesRegex(
            GENERATOR.ParameterError, "unsupported GFN2 charge"
        ):
            GENERATOR.normalize_export(invalid)

    def test_generated_header_compiles_and_checks_access_boundaries(self) -> None:
        """Compile the generated header and exercise accessor boundaries."""
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")
        with tempfile.TemporaryDirectory(prefix="gpuxtb-parameter-test-") as directory:
            executable = Path(directory) / "header_test"
            subprocess.run(
                (
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(REPOSITORY),
                    str(Path(__file__).with_name("gfn2_header_test.cpp")),
                    "-o",
                    str(executable),
                ),
                check=True,
            )
            subprocess.run((str(executable),), check=True)


if __name__ == "__main__":
    unittest.main()
