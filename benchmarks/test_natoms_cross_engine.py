"""Hardware-free tests for the cross-engine natoms benchmark and plotter."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks import natoms_cross_engine as nce
from benchmarks import plot_natoms_cross_engine as plotters


class StorageCheck:
    """Minimal storage view to spot-check distinct conformer batches."""

    def __init__(self, storage: nce.BatchStorage) -> None:
        self.storage = storage

    def system_positions(self, index: int) -> list[float]:
        item = self.storage.slices[index]
        return self.storage.positions[3 * item.atom_begin : 3 * item.atom_end]


class NatomsCrossEngineTest(unittest.TestCase):
    """Exercise the deterministic builders and plotting data selection."""

    def test_distinct_conformer_batch_never_reuses_a_geometry(self) -> None:
        """Every slot above zero must differ from slot zero and each other."""
        from benchmarks.natoms_scaling import make_alkane

        for natoms in (5, 14, 32):
            storage = nce.build_batch(make_alkane(natoms), 8, seed=100 * natoms)
            self.assertEqual(len(storage.slices), 8)
            positions = [
                tuple(storage.positions[3 * it.atom_begin : 3 * it.atom_end])
                for it in storage.slices
            ]
            self.assertEqual(len(set(positions)), 8)
            for slot in range(8):
                self.assertEqual(
                    len(positions[slot]), 3 * natoms, msg=f"slot {slot} size"
                )
                self.assertTrue(
                    all(math.isfinite(value) for value in positions[slot]),
                    msg=f"slot {slot} finite",
                )

    def test_trajectory_frames_are_close_but_distinct(self) -> None:
        """MD frames stay near one another while never being identical."""
        from benchmarks.natoms_scaling import make_alkane

        base = make_alkane(14)
        frames = nce.build_trajectory(base, 6, seed=42)
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(frames[0]), 3 * base.natoms)
        self.assertEqual(len(set(frames)), 6)
        reference = frames[0]
        for frame in frames[1:]:
            drift = max(abs(a - b) for a, b in zip(frame, reference, strict=True))
            self.assertLess(drift, 0.2)
            self.assertGreater(drift, 0.0)

    def test_regular_batch_size_one_keeps_ideal_geometry(self) -> None:
        """batch size one must reproduce the exact input molecule."""
        from benchmarks.natoms_scaling import make_alkane

        molecule = make_alkane(14)
        storage = nce.build_batch(molecule, 1, seed=7)
        self.assertEqual(len(storage.slices), 1)
        self.assertEqual(
            tuple(storage.positions), tuple(molecule.positions_bohr)
        )

    def test_eligibility_rejects_nonpass_and_nonfinite_rows(self) -> None:
        """Plot selection keeps only successful finite median rows."""
        good = {
            "availability": "available",
            "timing": {"median_ms": 1.25},
            "correctness": {"status": "pass"},
        }
        self.assertTrue(plotters._is_eligible(good))
        bad_finite = dict(good)
        bad_finite["timing"] = {"median_ms": float("nan")}
        self.assertFalse(plotters._is_eligible(bad_finite))
        bad_correctness = dict(good)
        bad_correctness["correctness"] = {"status": "fail"}
        self.assertFalse(plotters._is_eligible(bad_correctness))
        missing = dict(good)
        missing.pop("timing")
        self.assertFalse(plotters._is_eligible(missing))

    def test_plot_merges_artifacts_and_draws_without_gpu(self) -> None:
        """Full plot path runs in Agg mode from synthetic artifacts."""
        rows = [
            {
                "engine": engine,
                "natoms": natoms,
                "batch_size": batch_size,
                "availability": "available",
                "timing": {"median_ms": float(batch_size * natoms)},
                "correctness": {"status": "pass"},
            }
            for engine in ("gpuxtb-cpu", "xtb", "tblite")
            for natoms in (14, 32)
            for batch_size in (1, 128)
        ]
        rows.append(
            {
                "engine": "gpuxtb-cpu",
                "natoms": 32,
                "batch_size": 1,
                "job": "trajectory",
                "availability": "available",
                "timing": {"median_ms": 0.5},
            }
        )
        metadata = {
            "hardware": {
                "cpu_model": "test-cpu",
                "nvidia_smi": "test-gpu",
            },
            "threads": {"cpu_threads": 4},
            "commit": {"head": "0123456789abcdef"},
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "matrix.json"
            artifact.write_text(
                json.dumps({"metadata": metadata, "rows": rows}),
                encoding="utf-8",
            )
            output = Path(directory) / "figure.png"
            import subprocess  # noqa: PLC0415

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "benchmarks.plot_natoms_cross_engine",
                    "--artifact",
                    str(artifact),
                    "--commit",
                    "0123456",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(output.is_file() and output.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()