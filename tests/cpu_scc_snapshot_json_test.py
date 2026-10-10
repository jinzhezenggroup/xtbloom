"""Qualify snapshot JSON against a separate production shared-library producer."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/conformance"))
api = importlib.import_module("xtbloom_public_api")


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys instead of silently replacing captured evidence."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> NoReturn:
    """Nonfinite tokens are not part of this diagnostic's JSON contract."""
    raise ValueError(f"nonfinite JSON token: {value}")


class SnapshotJsonTest(unittest.TestCase):
    """Run the real diagnostic executable and an independent public C ABI handle."""

    executable: Path
    library_path: Path

    def setUp(self) -> None:
        """Allocate an isolated input path without touching retained evidence."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.input_path = Path(self.temporary.name) / "input.txt"

    def capture(self, text: str) -> subprocess.CompletedProcess[bytes]:
        """Keep the literal native exit and stderr visible in assertion failures."""
        self.input_path.write_text(text, encoding="ascii")
        return subprocess.run(
            [str(self.executable), "--capture", str(self.input_path)],
            capture_output=True,
            check=False,
            timeout=30,
        )

    def test_shared_public_outputs_and_round_trip_json(self) -> None:
        """Compare all three E/F/q receipts bitwise across the distinct producers."""
        for atoms, unpaired, spins, positions in (
            (2, 0, 1, [-0.0, 0.0, 0.0, 1.4, 0.0, 0.0]),
            (3, 1, 2, [0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 3.1, 0.2, 0.0]),
        ):
            with self.subTest(spins=spins):
                text = f"{atoms} 0 {unpaired} {spins}\n" + "".join(
                    "1 "
                    + " ".join(
                        format(value, ".17g")
                        for value in positions[3 * atom : 3 * atom + 3]
                    )
                    + "\n"
                    for atom in range(atoms)
                )
                completed = self.capture(text)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                document = json.loads(
                    completed.stdout,
                    object_pairs_hook=strict_object,
                    parse_constant=reject_constant,
                )
                self.assertEqual(document["schema"], "xtbloom.cpu.scc_snapshot.v1")
                self.assertEqual(
                    document["producer"], "standalone-public-api-diagnostic"
                )
                self.assertEqual(len(document["captures"]), 3)
                expected_options = {
                    "cpu_threads": 1,
                    "model": api.XTBLOOM_MODEL_GFN2_XTB,
                    "flags": (
                        api.XTBLOOM_COMPUTE_ENERGY
                        | api.XTBLOOM_COMPUTE_FORCES
                        | api.XTBLOOM_COMPUTE_ATOMIC_CHARGES
                    ),
                    "max_scc_iterations": 500,
                    "electronic_temperature": 300.0 * api.XTBLOOM_KELVIN_TO_HARTREE,
                    "energy_tolerance": 1e-10,
                    "charge_tolerance": 1e-8,
                    "scc_mixer": 1,
                    "scc_mixer_history": 8,
                    "scc_mixer_damping": 0.4,
                    "determinism": 0,
                }
                self.assertEqual(document["options"], expected_options)
                library = api._configure_library(self.library_path)
                context = api._make_context(library, "cpu", 0, 1)
                try:
                    storage = api.PublicBatchStorage(
                        atom_offsets=[0, atoms],
                        atomic_numbers=[1] * atoms,
                        positions=positions,
                        molecular_charges=[0.0],
                        unpaired_electrons=[unpaired],
                        spin_channels=[spins],
                        point_charge_offsets=[0, 0],
                        point_charge_positions=[],
                        point_charge_values=[],
                        point_charge_gammas=[],
                        cell_matrices=[],
                        periodic_axes=[],
                        slices=[
                            api.CaseSlice({"id": "snapshot-json"}, 0, atoms, 0, 0, {})
                        ],
                        keepalive=[],
                    )
                    with api.DescriptorMemory("host", 0) as memory:
                        batch = api._make_batch(library, storage, memory, True)
                        result = api.BatchResult()
                        api._call_ok(
                            library,
                            library.xtbloom_batch_result_init(
                                ctypes.byref(result), ctypes.sizeof(result)
                            ),
                            "result init",
                        )
                        owners = {
                            "energies": (ctypes.c_double * 1)(),
                            "forces": (ctypes.c_double * (3 * atoms))(),
                            "atomic_charges": (ctypes.c_double * atoms)(),
                            "scc_iterations": (ctypes.c_int32 * 1)(),
                            "scc_converged": (ctypes.c_uint8 * 1)(),
                            "per_system_status": (ctypes.c_int32 * 1)(),
                        }
                        for role, owner in owners.items():
                            setattr(result, role, memory.output(owner, role))
                        options = api.ComputeOptions()
                        api._call_ok(
                            library,
                            library.xtbloom_compute_options_init(
                                ctypes.byref(options), ctypes.sizeof(options)
                            ),
                            "options init",
                        )
                        for name, value in expected_options.items():
                            if name != "cpu_threads":
                                setattr(options, name, value)
                        for step, capture in enumerate(document["captures"]):
                            options.scc_start_mode = (
                                api.XTBLOOM_SCC_START_FRESH
                                if step == 0
                                else api.XTBLOOM_SCC_START_WARM
                            )
                            api._call_ok(
                                library,
                                library.xtbloom_compute(
                                    context,
                                    ctypes.byref(batch),
                                    ctypes.byref(options),
                                    ctypes.byref(result),
                                ),
                                "shared public compute",
                            )
                            self.assertEqual(
                                capture["public_compute_sequence"], step + 1
                            )
                            self.assertEqual(
                                capture["status"], owners["per_system_status"][0]
                            )
                            self.assertEqual(
                                capture["iterations"], owners["scc_iterations"][0]
                            )
                            self.assertEqual(owners["scc_converged"][0], 1)
                            for role, values in (
                                ("energies", [capture["public_energy"]]),
                                ("forces", capture["public_forces"]),
                                ("atomic_charges", capture["public_atomic_charges"]),
                            ):
                                self.assertTrue(
                                    all(
                                        type(value) is float and math.isfinite(value)
                                        for value in values
                                    )
                                )
                                self.assertEqual(
                                    struct.pack(f"={len(values)}d", *values),
                                    bytes(owners[role]),
                                )
                            self.assertEqual(
                                len(capture["occupations"]), 2 * capture["orbitals"]
                            )
                            self.assertEqual(
                                len(capture["density"]),
                                spins * capture["orbitals"] ** 2,
                            )
                finally:
                    library.xtbloom_context_destroy(context)

    def test_bad_input_has_nonzero_exit_and_no_json(self) -> None:
        """Reject bad inputs rather than accepting an empty evidence document."""
        for text in (
            "",
            "0 0 0 1\n",
            "2 0 0 3\n1 0 0 0\n1 1.4 0 0\n",
            "2 0 0 1\n1 0 0 0\n1 1.4 0 0\ntrailing\n",
        ):
            with self.subTest(text=text):
                completed = self.capture(text)
                self.assertEqual(completed.returncode, 1)
                self.assertEqual(completed.stdout, b"")

    def test_usage_exit(self) -> None:
        """Retain the distinct usage exit for unknown arguments."""
        completed = subprocess.run(
            [str(self.executable), "--unknown"],
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 2)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires a POSIX full device")
    def test_stdout_failure_is_not_success(self) -> None:
        """Report a failed write even after successful native inference."""
        self.input_path.write_text("2 0 0 1\n1 0 0 0\n1 1.4 0 0\n", encoding="ascii")
        with Path("/dev/full").open("wb") as output:
            completed = subprocess.run(
                [str(self.executable), "--capture", str(self.input_path)],
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 1, completed.stderr.decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments = parser.parse_args()
    SnapshotJsonTest.executable = arguments.executable.resolve(strict=True)
    SnapshotJsonTest.library_path = arguments.library.resolve(strict=True)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
