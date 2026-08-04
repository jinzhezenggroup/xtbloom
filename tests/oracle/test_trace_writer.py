"""Offline tests for the canonical ``gpuxtb-scc-trace-v1`` writer."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPOSITORY_ROOT / "tools" / "oracle" / "tblite_scc_trace" / "gpuxtb_scc_trace.py"
)
SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "oracle"
    / "tblite_scc_trace"
    / "gpuxtb-scc-trace-v1.schema.json"
)
SPEC = importlib.util.spec_from_file_location("gpuxtb_scc_trace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)

REVISION = "e9abc395b122018ed688aecb1c3a65cecaf97beb"
PATCH_SHA256 = "d6a51afc4b3c56d6589a2b5b115ea8b4891600c1161c525939ca3cc16e2b4954"


def _fixture(spin: int = 1) -> dict:
    """Build a minimal but schema-valid one-iteration trace."""
    nao = 1
    n_shells = 1
    n_atoms = 1

    def matrices(count: int) -> list:
        return [[[0.1]] for _ in range(count)]

    def spectra() -> list:
        return [[-0.5] for _ in range(spin)]

    residual = [0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
    return {
        "format": TRACE.FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": PATCH_SHA256,
        },
        "input": {
            "atomic_numbers": [1],
            "positions": [0.0, 0.0, 0.0],
            "molecular_charge": 1.0,
            "unpaired_electrons": 0,
            "spin_channels": spin,
            "temperature": 300.0,
        },
        "basis": {
            "nao": nao,
            "n_shells": n_shells,
            "n_atoms": n_atoms,
            "atom_to_shell_count": [n_shells],
        },
        "statics": {"overlap": matrices(1), "core_hamiltonian": matrices(1)},
        # Ignored: kept for clarity that the writer needs no tblite oracle.
        "residual_layout": {
            "shell_charges": 1,
            "atomic_dipoles": 3,
            "atomic_quadrupoles": 6,
        },
        "iterations": [
            {
                "index": 1,
                "hamiltonian": matrices(spin),
                "density": matrices(spin),
                "eigenvalues": spectra(),
                "occupations": spectra(),
                "mixed_qsh": spectra(),
                "raw_qsh": spectra(),
                "mixed_qat": spectra(),
                "raw_qat": spectra(),
                "residual": residual,
                "residual_rms": 0.141,
                "energy": -1.25,
                "converged": True,
                "status": TRACE.STATUS_CONVERGED,
            }
        ],
        "terminal": {
            "status": TRACE.STATUS_CONVERGED,
            "converged": True,
            "iterations": 1,
        },
    }


class TraceWriterTest(unittest.TestCase):
    """Exercise canonical emission, schema validation, and round-tripping."""

    def test_fixture_is_independently_valid(self) -> None:
        # The layout invariants must hold without any Fortran reference build.
        text = TRACE.dumps(_fixture())
        loaded = json.loads(text)
        self.assertEqual(loaded["format"], TRACE.FORMAT)
        self.assertEqual(loaded["terminal"]["iterations"], 1)
        self.assertTrue(loaded["iterations"][0]["converged"])

    def test_roundtrip_is_byte_identical(self) -> None:
        text = TRACE.dumps(_fixture())
        loaded = json.loads(text)
        self.assertEqual(TRACE.dumps(loaded), text)

    def test_unicode_and_string_roundtrip(self) -> None:
        fixture = _fixture()
        fixture["provenance"]["oracle_command"] = "tblite-scc-trace --molecule h3_plus"
        text = TRACE.dumps(fixture)
        self.assertIn("oracle_command", text)
        self.assertEqual(TRACE.dumps(json.loads(text)), text)

    def test_floats_keep_17_significant_digits(self) -> None:
        text = TRACE.dumps(_fixture())
        loaded = json.loads(text)
        # 0.1 is not exactly representable; 17 significant digits expose it.
        self.assertIn("0.10000000000000001", text)
        self.assertEqual(loaded["iterations"][0]["residual"][1], 0.1)
        # Integral floats must stay on the JSON number path as floats.
        fixture = _fixture()
        fixture["input"]["molecular_charge"] = 2.0
        text = TRACE.dumps(fixture)
        self.assertIn('"molecular_charge": 2.0', text)
        self.assertIsInstance(json.loads(text)["input"]["molecular_charge"], float)

    def test_quadrupole_packing_order_constant(self) -> None:
        self.assertEqual(
            TRACE.QUADRUPOLE_COMPONENTS, ("xx", "xy", "yy", "xz", "yz", "zz")
        )

    def test_two_spin_channels_are_nested(self) -> None:
        fixture = _fixture(spin=2)
        text = TRACE.dumps(fixture)
        loaded = json.loads(text)
        self.assertEqual(loaded["input"]["spin_channels"], 2)
        self.assertEqual(len(loaded["iterations"][0]["eigenvalues"]), 2)

    def test_rejects_unsupported_format_version(self) -> None:
        fixture = _fixture()
        fixture["format"] = "gpuxtb-scc-trace-v0"
        with self.assertRaisesRegex(TRACE.TraceError, "unsupported trace format"):
            TRACE.validate(fixture)

    def test_rejects_malformed_positions(self) -> None:
        fixture = _fixture()
        fixture["input"] = dict(fixture["input"], positions=[0.0, 0.0])
        with self.assertRaisesRegex(TRACE.TraceError, "positions must hold"):
            TRACE.validate(fixture)

    def test_rejects_oversized_matrix(self) -> None:
        fixture = _fixture()
        fixture["statics"] = dict(fixture["statics"], overlap=[[[0.1, 0.0]]])
        with self.assertRaisesRegex(TRACE.TraceError, "columns"):
            TRACE.validate(fixture)

    def test_rejects_wrong_residual_length(self) -> None:
        fixture = _fixture()
        fixture["iterations"][0] = dict(fixture["iterations"][0], residual=[0.0, 0.0])
        with self.assertRaisesRegex(TRACE.TraceError, "residual needs 10 elements"):
            TRACE.validate(fixture)

    def test_rejects_out_of_order_iteration_index(self) -> None:
        fixture = _fixture()
        fixture["iterations"] = [dict(fixture["iterations"][0], index=2)]
        with self.assertRaisesRegex(TRACE.TraceError, "out-of-order index"):
            TRACE.validate(fixture)

    def test_terminal_must_match_last_iteration(self) -> None:
        fixture = _fixture()
        fixture["terminal"] = {
            "status": TRACE.STATUS_MAX_ITERATIONS,
            "converged": False,
            "iterations": 1,
        }
        with self.assertRaisesRegex(TRACE.TraceError, "terminal.status"):
            TRACE.validate(fixture)

    def test_rejects_nonfinite_floats(self) -> None:
        fixture = _fixture()
        fixture["iterations"][0] = dict(fixture["iterations"][0], energy=float("nan"))
        with self.assertRaisesRegex(TRACE.TraceError, "finite"):
            TRACE.validate(fixture)

    def test_provenance_is_pinned_and_hashed(self) -> None:
        fixture = _fixture()
        with self.assertRaisesRegex(TRACE.TraceError, "tblite_revision"):
            TRACE.validate(dict(fixture, provenance={"tblite_revision": "deadbeef"}))
        with self.assertRaisesRegex(TRACE.TraceError, "oracle_patch_sha256"):
            TRACE.validate(dict(fixture, provenance={"tblite_revision": REVISION}))

    def test_schema_json_document_is_valid_and_versioned(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["format"]["const"], TRACE.FORMAT)
        self.assertIn("scc-trace-v1", str(SCHEMA_PATH))


if __name__ == "__main__":
    unittest.main()
