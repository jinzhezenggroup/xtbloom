"""Hardware-free tests for the cross-engine natoms benchmark and plotter."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks import natoms_cross_engine as nce
from benchmarks import plot_natoms_cross_engine as plotters


def artifact_metadata(
    start_policy: str = "cold",
    *,
    designation: str = "independent_baseline",
    reference_sha256: str | None = None,
) -> dict[str, object]:
    """Return one schema-v2 clean metadata block for plotter tests."""
    return {
        "hardware": {
            "hostname": "test-node",
            "cpu_model": "test-cpu",
            "nvidia_smi": "test-gpu",
            "nvidia_smi_runtime": "test-gpu,test-uuid,test-driver,32768 MiB",
            "process_cpu_affinity": [0, 1, 2, 3],
            "selected_cuda_device": {
                "cuda_ordinal": 0,
                "CUDA_VISIBLE_DEVICES": "0",
                "resolved_visibility_token": "0",
                "device": {"physical_index": "0", "uuid": "test-uuid"},
            },
        },
        "threads": {
            "cpu_threads": 4,
            "reference_threads": 4,
            "dxtb_cpu_threads": 4,
        },
        "commit": {"head": "0123456789abcdef", "dirty": False},
        "evidence_eligibility": {
            "status": "eligible_clean_head",
            "allow_dirty_evidence": False,
        },
        "comparison_reference": {
            "designation": designation,
            "engine": "xtb",
            "artifact_sha256": reference_sha256,
        },
        "runner": {
            "gpuxtb_library_sha256": "g" * 64,
            "gpuxtb_build": {
                "source_state": {"head": "gpuxtb-head", "dirty": False},
                "selected": {"GPUXTB_ENABLE_CUDA": "ON"},
            },
            "gpuxtb_native_identity": {
                "sha256": "g" * 64,
                "resolved_dependencies": [],
                "unresolved_dependencies": [],
            },
            "xtb_library_sha256": "x" * 64,
            "xtb_source": {"head": "xtb-head", "dirty": False},
            "xtb_native_identity": {
                "sha256": "x" * 64,
                "resolved_dependencies": [],
                "unresolved_dependencies": [],
            },
            "tblite_library_sha256": "t" * 64,
            "tblite_source": {"head": "tblite-head", "dirty": False},
            "tblite_native_identity": {
                "sha256": "t" * 64,
                "resolved_dependencies": [],
                "unresolved_dependencies": [],
            },
        },
        "protocol": {
            "warmups": 1,
            "repetitions": 3,
            "start_policy": start_policy,
            "cross_engine_energy_atol_hartree": 2.0e-3,
            "cross_engine_force_atol_hartree_per_bohr": 2.0e-3,
            "repeatability_energy_atol_hartree": 1.0e-10,
            "repeatability_force_atol_hartree_per_bohr": 1.0e-8,
            "scc_max_iterations": 500,
            "scc_charge_tolerance": 1.0e-4,
            "scc_energy_tolerance": 1.0e-6,
            "convergence_contract": {
                "xtb": {"public_accuracy_factor": 1.0},
                "tblite": {"public_accuracy_factor": 1.0},
            },
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

    def test_direct_url_identity_binds_local_source_without_leaking_url(self) -> None:
        """Publication metadata retains a local binding but no URL secrets."""
        identity = nce.sanitize_direct_url_identity(
            json.dumps(
                {
                    "url": "file:///tmp/dxtb%20source",
                    "dir_info": {"editable": True},
                }
            )
        )
        self.assertEqual(
            identity,
            {
                "scheme": "file",
                "local_source_path": str(Path("/tmp/dxtb source").resolve()),
                "editable": True,
            },
        )
        self.assertEqual(
            nce.sanitize_direct_url_identity(
                '{"url":"https://user:secret@example.invalid/dxtb.git"}'
            ),
            {"scheme": "https"},
        )
        self.assertEqual(
            nce.sanitize_direct_url_identity("not-json"),
            {"scheme": None, "parse_status": "invalid"},
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
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")
        rows = [
            {
                "engine": engine,
                "natoms": natoms,
                "batch_size": batch_size,
                "start_policy": ("auto-warm" if batch_size == 128 else "cold"),
                "effective_start_policy": (
                    "auto-warm" if batch_size == 128 else "cold"
                ),
                "availability": "available",
                "timing": {"median_ms": float(batch_size * natoms)},
                "correctness": qualified_correctness(engine),
            }
            for engine in ("gpuxtb-cpu", "xtb", "tblite")
            for natoms in (14, 32)
            for batch_size in (1, 128, 512)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            cold_reference = directory_path / "cold-reference.json"
            cold_reference.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": artifact_metadata("cold"),
                        "rows": [
                            row
                            for row in rows
                            if row["engine"] == "xtb" and row["batch_size"] != 128
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cold_reference_sha256 = nce.sha256_file(cold_reference)
            warm_reference = directory_path / "warm-reference.json"
            warm_reference.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": artifact_metadata("auto-warm"),
                        "rows": [
                            row
                            for row in rows
                            if row["engine"] == "xtb" and row["batch_size"] == 128
                        ],
                    }
                ),
                encoding="utf-8",
            )
            warm_reference_sha256 = nce.sha256_file(warm_reference)

            cold_dependent_rows = [
                row
                for row in rows
                if row["engine"] != "xtb" and row["batch_size"] != 128
            ]
            warm_dependent_rows = [
                row
                for row in rows
                if row["engine"] != "xtb" and row["batch_size"] == 128
            ]
            for dependent_rows, reference_sha256 in (
                (cold_dependent_rows, cold_reference_sha256),
                (warm_dependent_rows, warm_reference_sha256),
            ):
                for row in dependent_rows:
                    row["correctness"]["cross_engine"] = {
                        "status": "pass",
                        "reference_engine": "xtb",
                        "artifact_sha256": reference_sha256,
                    }
            cold_artifact = directory_path / "cold-dependent.json"
            cold_artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": artifact_metadata(
                            "cold",
                            designation="dependent_run",
                            reference_sha256=cold_reference_sha256,
                        ),
                        "rows": cold_dependent_rows,
                    }
                ),
                encoding="utf-8",
            )
            warm_artifact = directory_path / "warm-dependent.json"
            warm_artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": artifact_metadata(
                            "auto-warm",
                            designation="dependent_run",
                            reference_sha256=warm_reference_sha256,
                        ),
                        "rows": warm_dependent_rows,
                    }
                ),
                encoding="utf-8",
            )
            output = directory_path / "figure.svg"
            import subprocess

            completed = subprocess.run(
                [
                    "python3",
                    "-m",
                    "benchmarks.plot_natoms_cross_engine",
                    "--artifact",
                    str(cold_reference),
                    "--artifact",
                    str(warm_reference),
                    "--artifact",
                    str(cold_artifact),
                    "--artifact",
                    str(warm_artifact),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(output.is_file() and output.stat().st_size > 0)
            svg = output.read_text(encoding="utf-8")
            self.assertIn("<title>", svg)
            self.assertIn("<desc>", svg)
            self.assertNotIn("<dc:date>", svg)

    def test_cold_panel_excludes_autowarm_trajectory_leak(self) -> None:
        """Batch=1 rows leaked by an auto-warm trajectory run must not enter.

        The trajectory invocation also measures a steady-state batch=1 cell
        without a ``job`` tag; only rows whose source artifact recorded a
        ``cold`` start policy may appear in the batch=1 cold-start panel.
        """
        cold_row = {
            "engine": "gpuxtb-cpu",
            "natoms": 62,
            "batch_size": 1,
            "start_policy": "cold",
            "effective_start_policy": "cold",
            "_artifact_start_policy": "cold",
            "availability": "available",
            "timing": {"median_ms": 76.0},
            "correctness": qualified_correctness(),
        }
        leak_row = dict(cold_row)  # same shape, warm auto-warm measurement
        leak_row["timing"] = {"median_ms": 21.4}
        leak_row["start_policy"] = "auto-warm"
        leak_row["effective_start_policy"] = "auto-warm"
        leak_row["_artifact_start_policy"] = "auto-warm"
        rows = [cold_row, leak_row]
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
        self.assertTrue(plotters._cold_batch1_row(cold_row))
        self.assertFalse(plotters._cold_batch1_row(leak_row))
        self.assertFalse(
            plotters._cold_batch1_row(
                dict(leak_row, job="trajectory", timing={"median_ms": 0.5})
            )
        )
        legacy = dict(cold_row)
        legacy.pop("_artifact_start_policy", None)
        self.assertFalse(plotters._cold_batch1_row(legacy))

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
            "cold",
        )
        self.assertEqual(fragment["forces_hartree_per_bohr"], [0.1, -0.2, 0.3])
        self.assertEqual(fragment["correctness"]["status"], "pass")
        self.assertEqual(fragment["correctness"]["force_value_count"], 3)
        self.assertEqual(len(fragment["raw_samples"]), 2)
        for sample in fragment["raw_samples"]:
            self.assertEqual(sample["force_count"], 3)
            self.assertEqual(len(sample["forces_sha256_binary64_le"]), 64)

    def test_auto_warm_seeds_even_without_optional_warmups(self) -> None:
        """The untimed FRESH seed is independent of ``--warmups``."""

        class FakeWarmRunner:
            def __init__(self) -> None:
                self.invocations = 0
                self.modes: list[str] = []

            def set_start_mode(self, mode: str) -> None:
                self.modes.append(mode)

            def invoke(self) -> None:
                self.invocations += 1

            def snapshot(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
                    "scc_iterations": [2],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

        runner = FakeWarmRunner()
        fragment = nce.measure_cell(
            runner,
            (0, 1),
            nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
            "auto-warm",
        )
        self.assertEqual(runner.invocations, 2)
        self.assertEqual(runner.modes, ["fresh", "warm"])
        self.assertEqual(fragment["effective_start_policy"], "auto-warm")

    def test_cold_gpuxtb_records_fresh_initialization_inside_public_call(self) -> None:
        """Selecting FRESH is untimed, but gpuxtb initializes it during compute."""

        class FakeGpuxtbRunner:
            def set_start_mode(self, _mode: str) -> None:
                return

            def invoke(self) -> None:
                return

            def snapshot(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
                    "scc_iterations": [2],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

        fragment = nce.measure_cell(
            FakeGpuxtbRunner(),
            (0, 1),
            nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
            "cold",
        )
        self.assertEqual(
            fragment["state_preparation_timing"],
            "fresh_state_initialization_inside_timed_public_call",
        )

    def test_auto_warm_records_drift_without_cold_repeatability_gate(self) -> None:
        """Warm-state refinement is diagnostic, not cold-call repeatability."""

        class RefiningWarmRunner:
            def __init__(self) -> None:
                self.invocations = 0

            def set_start_mode(self, _mode: str) -> None:
                return

            def invoke(self) -> None:
                self.invocations += 1

            def snapshot(self) -> dict[str, object]:
                drift = self.invocations * 1.0e-6
                return {
                    "energies_hartree": [-1.0 + drift],
                    "forces_hartree_per_bohr": [drift, 0.0, 0.0],
                    "scc_iterations": [2],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

        fragment = nce.measure_cell(
            RefiningWarmRunner(),
            (0, 2),
            nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
            "auto-warm",
            repeatability_energy_atol_hartree=1.0e-10,
            repeatability_force_atol_hartree_per_bohr=1.0e-8,
        )
        repeatability = fragment["correctness"]["repeatability"]
        self.assertEqual(fragment["correctness"]["status"], "pass")
        self.assertFalse(repeatability["gate_applied"])
        self.assertGreater(repeatability["max_abs_energy_drift_hartree"], 1.0e-10)
        self.assertGreater(
            repeatability["max_abs_force_drift_hartree_per_bohr"], 1.0e-8
        )

        cold = nce.measure_cell(
            RefiningWarmRunner(),
            (0, 2),
            nce.Cell("gpuxtb-cpu", 1, 1, 1, 0),
            "cold",
            repeatability_energy_atol_hartree=1.0e-10,
            repeatability_force_atol_hartree_per_bohr=1.0e-8,
        )
        self.assertEqual(cold["correctness"]["status"], "fail")
        self.assertTrue(cold["correctness"]["repeatability"]["gate_applied"])

    def test_dxtb_policy_is_recorded_as_cold(self) -> None:
        """A requested auto-warm row must expose dxtb's actual cold behavior."""

        class FakeDxtbRunner:
            always_cold = True

            def __init__(self) -> None:
                self.invocations = 0

            def invoke(self) -> None:
                self.invocations += 1

            def snapshot(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
                    "scc_iterations": None,
                    "scc_converged": None,
                    "per_system_status": None,
                }

        runner = FakeDxtbRunner()
        fragment = nce.measure_cell(
            runner,
            (0, 1),
            nce.Cell("dxtb-cpu", 1, 1, 1, 0),
            "auto-warm",
        )
        self.assertEqual(runner.invocations, 1)
        self.assertEqual(fragment["effective_start_policy"], "cold")
        self.assertEqual(fragment["state_preparation_timing"], "inside_timed_invoke")
        self.assertTrue(fragment["correctness"]["repeatability"]["gate_applied"])
        self.assertIsNone(fragment["correctness"]["scc_converged_ok"])
        self.assertIsNone(fragment["correctness"]["scc_status_ok"])

    def test_force_vector_encoding_round_trips_through_reference_loader(self) -> None:
        """Compressed binary64 forces remain complete and digest-validated."""
        row = {
            "engine": "xtb",
            "natoms": 1,
            "batch_size": 1,
            "availability": "available",
            "start_policy": "cold",
            "effective_start_policy": "cold",
            "energies_hartree": [-1.0],
            "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
            "correctness": qualified_correctness("xtb"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            nce.write_json(
                path,
                {"schema_version": 2, "metadata": artifact_metadata(), "rows": [row]},
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            encoded = document["rows"][0]["forces_binary64_le_zlib_base64"]
            self.assertEqual(encoded["count"], 3)
            self.assertNotIn("forces_hartree_per_bohr", document["rows"][0])
            artifact = nce.load_reference_artifact(path)
        self.assertEqual(
            artifact.rows[(1, 1)]["forces_hartree_per_bohr"], [0.1, -0.2, 0.3]
        )

    def test_tblite_can_be_designated_as_independent_reference(self) -> None:
        """Reference qualification is not hard-coded to the known xTB force drift."""
        metadata = artifact_metadata("auto-warm")
        metadata["comparison_reference"] = {
            "designation": "independent_baseline",
            "engine": "tblite",
        }
        row = {
            "engine": "tblite",
            "natoms": 1,
            "batch_size": 1,
            "availability": "available",
            "start_policy": "auto-warm",
            "effective_start_policy": "auto-warm",
            "energies_hartree": [-1.0],
            "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
            "correctness": qualified_correctness("xtb"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            nce.write_json(
                path,
                {"schema_version": 2, "metadata": metadata, "rows": [row]},
            )
            artifact = nce.load_reference_artifact(path)
        self.assertEqual(artifact.engine, "tblite")

    def test_cross_engine_gate_checks_energy_and_force_vectors(self) -> None:
        """Either observable exceeding its gate makes a timing row ineligible."""
        reference = nce.ReferenceArtifact(
            path=Path("reference.json"),
            sha256="a" * 64,
            engine="xtb",
            metadata={},
            rows={
                (1, 1): {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
                }
            },
        )
        for energy, forces in (
            (-0.99, [0.1, -0.2, 0.3]),
            (-1.0, [0.1, -0.2, 0.31]),
        ):
            row = {
                "engine": "gpuxtb-cpu",
                "natoms": 1,
                "batch_size": 1,
                "availability": "available",
                "energies_hartree": [energy],
                "forces_hartree_per_bohr": forces,
                "correctness": {"status": "pass"},
            }
            self.assertTrue(
                nce.apply_cross_engine_reference(row, reference, 2.0e-3, 2.0e-3)
            )
            self.assertEqual(row["correctness"]["status"], "fail")
            self.assertEqual(row["correctness"]["cross_engine"]["status"], "fail")

    def test_cross_engine_gate_checks_every_timed_warm_sample(self) -> None:
        """A final in-tolerance WARM result cannot hide an earlier bad sample."""
        reference = nce.ReferenceArtifact(
            path=Path("reference.json"),
            sha256="a" * 64,
            engine="xtb",
            metadata={},
            rows={
                (1, 1): {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
                }
            },
        )
        row = {
            "engine": "gpuxtb-cpu",
            "natoms": 1,
            "batch_size": 1,
            "availability": "available",
            "raw_samples": [
                {"energies_hartree": [-0.99]},
                {"energies_hartree": [-1.0]},
            ],
            "energies_hartree": [-1.0],
            "forces_hartree_per_bohr": [0.1, -0.2, 0.3],
            "_force_samples_hartree_per_bohr": [
                [0.1, -0.2, 0.31],
                [0.1, -0.2, 0.3],
            ],
            "correctness": {"status": "pass"},
        }
        self.assertTrue(
            nce.apply_cross_engine_reference(row, reference, 2.0e-3, 2.0e-3)
        )
        comparison = row["correctness"]["cross_engine"]
        self.assertEqual(comparison["status"], "fail")
        self.assertEqual(comparison["timed_sample_count_checked"], 2)
        self.assertEqual(
            [item["status"] for item in comparison["timed_sample_deltas"]],
            ["fail", "pass"],
        )

    def test_panel_protocol_filter_rejects_mislabeled_artifacts(self) -> None:
        """Each panel accepts only its named requested/effective start policy."""
        base = {
            "engine": "gpuxtb-cpu",
            "job": None,
            "start_policy": "auto-warm",
            "effective_start_policy": "auto-warm",
            "_artifact_start_policy": "auto-warm",
        }
        self.assertTrue(plotters._matches_panel_protocol(base, 128))
        self.assertFalse(plotters._matches_panel_protocol(base, 512))
        cold = dict(
            base,
            start_policy="cold",
            effective_start_policy="cold",
            _artifact_start_policy="cold",
        )
        self.assertTrue(plotters._matches_panel_protocol(cold, 512))
        self.assertFalse(plotters._matches_panel_protocol(cold, 128))
        dxtb = dict(base, engine="dxtb-cpu", effective_start_policy="cold")
        self.assertTrue(plotters._matches_panel_protocol(dxtb, 128))

    def test_figure_protocol_note_is_metadata_derived(self) -> None:
        """Figure header must not silently hard-code publication settings."""
        note = plotters._protocol_note(artifact_metadata())
        self.assertIn("4 CPU threads", note)
        self.assertIn("median n=3", note)
        self.assertIn("10⁻⁴", note)

    def test_artifact_pair_refuses_overwrite(self) -> None:
        """Publication must not replace a stale JSON/CSV pair."""
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "matrix.json"
            csv_path = Path(directory) / "matrix.csv"
            json_path.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(nce.BenchmarkError, "refusing to overwrite"):
                nce.publish_artifacts(
                    json_path,
                    csv_path,
                    {"schema_version": 2, "metadata": {}, "rows": []},
                    [],
                )
            self.assertEqual(json_path.read_text(encoding="utf-8"), "stale\n")
            self.assertFalse(csv_path.exists())

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
                "cold",
            )

    def test_speedup_annotation_is_data_derived(self) -> None:
        """The plotted headline ratio comes from qualified overlapping rows."""
        rows = [
            {
                "engine": engine,
                "natoms": 62,
                "batch_size": 128,
                "start_policy": "auto-warm",
                "effective_start_policy": "auto-warm",
                "_artifact_start_policy": "auto-warm",
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
            (100.0, 900.0, 9.0, 12.0, "xTB / tblite"),
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

    def test_plotter_rejects_clean_diagnostic_override(self) -> None:
        """A clean checkout cannot make --allow-dirty publication-eligible."""
        metadata = artifact_metadata()
        metadata["evidence_eligibility"] = {
            "status": "diagnostic_override",
            "allow_dirty_evidence": True,
        }
        row = {
            "engine": "xtb",
            "natoms": 14,
            "batch_size": 1,
            "availability": "available",
            "timing": {"median_ms": 1.0},
            "correctness": qualified_correctness("xtb"),
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "diagnostic.json"
            artifact.write_text(
                json.dumps({"schema_version": 2, "metadata": metadata, "rows": [row]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(plotters.PlotError, "diagnostic"):
                plotters.load_rows([artifact])

    def test_plotter_rejects_mismatched_process_affinity(self) -> None:
        """Speed claims cannot merge runs from different CPU allocations."""
        metadata_a = artifact_metadata()
        metadata_b = artifact_metadata()
        metadata_b["hardware"]["process_cpu_affinity"] = [4, 5, 6, 7]

        def reference_row(natoms: int) -> dict[str, object]:
            return {
                "engine": "xtb",
                "natoms": natoms,
                "batch_size": 1,
                "availability": "available",
                "timing": {"median_ms": 1.0},
                "correctness": qualified_correctness("xtb"),
            }

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a.json", Path(directory) / "b.json"]
            for path, metadata, natoms in zip(
                paths, (metadata_a, metadata_b), (14, 32), strict=True
            ):
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "metadata": metadata,
                            "rows": [reference_row(natoms)],
                        }
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(plotters.PlotError, "incompatible run"):
                plotters.load_rows(paths)


if __name__ == "__main__":
    unittest.main()
