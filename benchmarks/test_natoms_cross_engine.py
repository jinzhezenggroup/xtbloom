"""Hardware-free tests for the cross-engine natoms benchmark and plotter."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks import natoms_cross_engine as nce
from benchmarks import plot_natoms_cross_engine as plotters


def artifact_metadata(start_policy: str = "cold") -> dict[str, object]:
    """Return one schema-v2 clean metadata block for plotter tests."""
    return {
        "hardware": {
            "hostname": "test-node",
            "cpu_model": "test-cpu",
            "nvidia_smi": "test-gpu",
        },
        "threads": {
            "cpu_threads": 4,
            "reference_threads": 4,
            "dxtb_cpu_threads": 4,
        },
        "commit": {"head": "0123456789abcdef", "dirty": False},
        "protocol": {
            "warmups": 1,
            "repetitions": 3,
            "start_policy": start_policy,
            "cross_engine_energy_atol_hartree": 2.0e-3,
            "cross_engine_force_atol_hartree_per_bohr": 2.0e-3,
            "scc_max_iterations": 500,
            "scc_charge_tolerance": 1.0e-4,
            "scc_energy_tolerance": 1.0e-4,
        },
    }


def qualified_correctness(engine: str = "gpuxtb-cpu") -> dict[str, object]:
    """Return a complete correctness gate accepted by the plotter."""
    return {
        "status": "pass",
        "cross_engine": {"status": "reference" if engine == "xtb" else "pass"},
    }


class StorageCheck:
    """Minimal storage view to spot-check distinct conformer batches."""

    def __init__(self, storage: nce.BatchStorage) -> None:
        self.storage = storage

    def system_positions(self, index: int) -> list[float]:
        """Return the flattened positions of one slot for a spot check."""
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
        """Batch size one must reproduce the exact input molecule."""
        from benchmarks.natoms_scaling import make_alkane

        molecule = make_alkane(14)
        storage = nce.build_batch(molecule, 1, seed=7)
        self.assertEqual(len(storage.slices), 1)
        self.assertEqual(tuple(storage.positions), tuple(molecule.positions_bohr))

    def test_eligibility_rejects_nonpass_and_nonfinite_rows(self) -> None:
        """Plot selection keeps only successful finite median rows."""
        good = {
            "availability": "available",
            "timing": {"median_ms": 1.25},
            "correctness": qualified_correctness(),
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
        missing_correctness = dict(good)
        missing_correctness.pop("correctness")
        self.assertFalse(plotters._is_eligible(missing_correctness))

    def test_trajectory_row_sweeps_natoms_per_engine(self) -> None:
        """The runner emits one trajectory row per (engine, natoms)."""
        base = {
            "engine": "gpuxtb-cpu",
            "batch_size": 1,
            "frames": 12,
            "job": "trajectory",
            "availability": "available",
            "timing": {"median_ms": 0.5},
            "correctness": qualified_correctness(),
        }
        rows = [dict(base, natoms=natoms) for natoms in (32, 62, 122)]
        rows.append(dict(base, engine="xtb", natoms=62))
        selected: dict[str, list[tuple[int, float]]] = {}
        for engine in ("gpuxtb-cpu", "xtb"):
            qualified = [
                row
                for row in rows
                if plotters._is_eligible(row)
                and row.get("engine") == engine
                and row.get("job") == "trajectory"
                and row.get("batch_size") == 1
            ]
            qualified.sort(key=lambda row: row["natoms"])
            selected[engine] = [
                (row["natoms"], plotters._median_ms(row)) for row in qualified
            ]
        self.assertEqual(selected["gpuxtb-cpu"], [(32, 0.5), (62, 0.5), (122, 0.5)])
        self.assertEqual(selected["xtb"], [(62, 0.5)])

    def test_plot_merges_artifacts_and_draws_without_gpu(self) -> None:
        """Full plot path runs in Agg mode from synthetic artifacts."""
        rows = [
            {
                "engine": engine,
                "natoms": natoms,
                "batch_size": batch_size,
                "availability": "available",
                "timing": {"median_ms": float(batch_size * natoms)},
                "correctness": qualified_correctness(engine),
            }
            for engine in ("gpuxtb-cpu", "xtb", "tblite")
            for natoms in (14, 32)
            for batch_size in (1, 128, 512)
        ]
        rows.append(
            {
                "engine": "gpuxtb-cpu",
                "natoms": 32,
                "batch_size": 1,
                "job": "trajectory",
                "availability": "available",
                "timing": {"median_ms": 0.5},
                "correctness": qualified_correctness(),
            }
        )
        rows.append(
            {
                "engine": "gpuxtb-cpu",
                "natoms": 62,
                "batch_size": 1,
                "job": "trajectory",
                "availability": "available",
                "timing": {"median_ms": 1.0},
                "correctness": qualified_correctness(),
            }
        )
        rows.append(
            {
                "engine": "xtb",
                "natoms": 62,
                "batch_size": 1,
                "job": "trajectory",
                "availability": "available",
                "timing": {"median_ms": 0.8},
                "correctness": qualified_correctness("xtb"),
            }
        )
        metadata = artifact_metadata()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "matrix.json"
            artifact.write_text(
                json.dumps({"schema_version": 2, "metadata": metadata, "rows": rows}),
                encoding="utf-8",
            )
            output = Path(directory) / "figure.png"
            import subprocess

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "benchmarks.plot_natoms_cross_engine",
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(output.is_file() and output.stat().st_size > 0)

    def test_cold_panel_excludes_autowarm_trajectory_leak(self) -> None:
        """Batch=1 rows leaked by an auto-warm trajectory run must not enter.

        The trajectory invocation also measures a steady-state batch=1 cell
        without a ``job`` tag; only rows whose source artifact recorded a
        ``cold`` start policy (or no policy at all, i.e. legacy artifacts) may
        appear in the batch=1 cold-start panel.
        """
        cold_row = {
            "engine": "gpuxtb-cpu",
            "natoms": 62,
            "batch_size": 1,
            "availability": "available",
            "timing": {"median_ms": 76.0},
            "correctness": qualified_correctness(),
        }
        leak_row = dict(cold_row)  # same shape, warm auto-warm measurement
        leak_row["timing"] = {"median_ms": 21.4}
        metadata = artifact_metadata()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cold_artifact = directory / "cold.json"
            cold_artifact.write_text(
                json.dumps(
                    {"schema_version": 2, "metadata": metadata, "rows": [cold_row]}
                ),
                encoding="utf-8",
            )
            auto_warm_metadata = artifact_metadata("auto-warm")
            leak_artifact = directory / "traj.json"
            leak_artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": auto_warm_metadata,
                        "rows": [leak_row],
                    },
                ),
                encoding="utf-8",
            )
            rows, _ = plotters.load_rows([cold_artifact, leak_artifact])
            cold = [
                row
                for row in rows
                if plotters._is_eligible(row)
                and row.get("batch_size") == 1
                and plotters._cold_batch1_row(row)
            ]
            self.assertEqual(
                [(row["natoms"], row["timing"]["median_ms"]) for row in cold],
                [(62, 76.0)],
            )
            cold_loaded = next(
                row for row in rows if row["timing"]["median_ms"] == 76.0
            )
            leak_loaded = next(
                row for row in rows if row["timing"]["median_ms"] == 21.4
            )
            self.assertNotEqual(
                cold_loaded.get("_artifact_start_policy"),
                leak_loaded.get("_artifact_start_policy"),
            )
            self.assertTrue(plotters._cold_batch1_row(cold_loaded))
            self.assertFalse(plotters._cold_batch1_row(leak_loaded))
            self.assertFalse(
                plotters._cold_batch1_row(
                    dict(leak_loaded, job="trajectory", timing={"median_ms": 0.5})
                )
            )
            legacy = dict(cold_loaded)
            legacy.pop("_artifact_start_policy", None)
            self.assertTrue(plotters._cold_batch1_row(legacy))

    def test_measure_cell_retains_complete_force_evidence(self) -> None:
        """Every measured repetition hashes forces and retains the final vector."""

        class FakeRunner:
            def __init__(self) -> None:
                self.invocations = 0

            def set_start_mode(self, _mode: str) -> None:
                return

            def invoke(self) -> None:
                self.invocations += 1

            def snapshot(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
                    "scc_iterations": [4],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

        fragment = nce.measure_cell(
            FakeRunner(),
            (0, 2),
            nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
            1.0e-8,
            1.0e-8,
            "cold",
        )
        self.assertEqual(fragment["forces_hartree_per_bohr"], [0.1, -0.2, 0.3])
        self.assertEqual(fragment["correctness"]["status"], "pass")
        self.assertEqual(fragment["correctness"]["force_value_count"], 3)
        self.assertEqual(len(fragment["raw_samples"]), 2)
        for sample in fragment["raw_samples"]:
            self.assertEqual(sample["force_count"], 3)
            self.assertEqual(len(sample["forces_sha256_binary64_le"]), 64)

    def test_measure_cell_rejects_missing_requested_forces(self) -> None:
        """A force benchmark cannot pass when an adapter omits force output."""

        class MissingForceRunner:
            def set_start_mode(self, _mode: str) -> None:
                return

            def invoke(self) -> None:
                return

            def snapshot(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": None,
                    "scc_iterations": [4],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

        with self.assertRaisesRegex(nce.BenchmarkError, "force values"):
            nce.measure_cell(
                MissingForceRunner(),
                (0, 1),
                nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
                1.0e-8,
                1.0e-8,
                "cold",
            )

    def test_speedup_annotation_is_data_derived(self) -> None:
        """The plotted headline ratio comes from qualified overlapping rows."""
        rows = [
            {
                "engine": engine,
                "natoms": 62,
                "batch_size": 128,
                "availability": "available",
                "timing": {"median_ms": latency},
                "correctness": qualified_correctness(engine),
            }
            for engine, latency in (
                ("gpuxtb-cpu", 100.0),
                ("xtb", 1200.0),
                ("tblite", 900.0),
            )
        ]
        self.assertEqual(
            plotters._speedup_range(rows, 128, 62),
            (100.0, 900.0, 9.0, 12.0),
        )

    def test_plotter_rejects_dirty_artifact(self) -> None:
        """Publication plots cannot silently combine dirty benchmark output."""
        metadata = artifact_metadata()
        metadata["commit"] = {"head": "0123456789abcdef", "dirty": True}
        row = {
            "engine": "xtb",
            "natoms": 14,
            "batch_size": 1,
            "availability": "available",
            "timing": {"median_ms": 1.0},
            "correctness": qualified_correctness("xtb"),
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "dirty.json"
            artifact.write_text(
                json.dumps({"schema_version": 2, "metadata": metadata, "rows": [row]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(plotters.PlotError, "clean-HEAD"):
                plotters.load_rows([artifact])


if __name__ == "__main__":
    unittest.main()
