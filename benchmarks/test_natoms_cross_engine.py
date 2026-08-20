"""Hardware-free tests for the cross-engine natoms benchmark and plotter."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import build_natoms_cross_engine_table as publication
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
                "runtime_uuid": "test-uuid",
                "resolved_visibility_token": "0",
                "device": {
                    "physical_index": "0",
                    "uuid": "test-uuid",
                    "name": "NVIDIA GeForce RTX 5090",
                    "driver": "580.95.05",
                    "memory_mib": "32607",
                },
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
            "xtbloom_library_sha256": "g" * 64,
            "xtbloom_build": {
                "source_state": {"head": "xtbloom-head", "dirty": False},
                "selected": {"XTBLOOM_ENABLE_CUDA": "ON"},
            },
            "xtbloom_native_identity": {
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
            "dxtb_source": {"head": "dxtb-head", "dirty": False},
            "python_distributions": {
                name: {
                    "version": "test-version",
                    "payload_verification": {
                        "status": "verified",
                        "payload_sha256": name[0] * 64,
                    },
                    "direct_url_identity": None,
                }
                for name in ("dxtb", "torch", "tad-libcint")
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
            "perturb_sigma_bohr": nce.PERTURB_SIGMA_BOHR,
            "scc_max_iterations": 500,
            "scc_charge_tolerance": 1.0e-4,
            "scc_energy_tolerance": 1.0e-6,
            "convergence_contract": {
                "xtbloom": {
                    "charge_tolerance": 1.0e-4,
                    "energy_tolerance": 1.0e-6,
                },
                "xtb": {"public_accuracy_factor": 1.0},
                "tblite": {"public_accuracy_factor": 1.0},
                "dxtb": {
                    "x_atol": 1.0e-4,
                    "x_atol_max": 1.0e-5,
                    "f_atol": 1.0e-4,
                    "force_convergence": True,
                },
            },
        },
    }


def qualified_correctness(engine: str = "xtbloom-cpu") -> dict[str, object]:
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
            self.assertEqual(storage.efields, [None] * 8)
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

    def test_timing_summary_retains_observed_maximum(self) -> None:
        """Compact publication rows must not infer the upper whisker."""
        summary = nce.timing_summary((3.0, 1.0, 2.0), batch_size=4)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 3.0)
        self.assertEqual(summary["p95_ms"], 3.0)

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
        self.assertEqual(
            nce.sanitize_direct_url_identity("[]"),
            {"scheme": None, "parse_status": "invalid"},
        )
        self.assertEqual(
            nce.sanitize_direct_url_identity(
                '{"url":"file://remote.example/tmp/dxtb"}'
            ),
            {"scheme": "file", "parse_status": "nonlocal_authority"},
        )
        self.assertEqual(
            nce.sanitize_direct_url_identity(
                '{"url":"file:///tmp/dxtb","dir_info":"editable"}'
            ),
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
            "engine": "xtbloom-cpu",
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
        for engine in ("xtbloom-cpu", "xtb"):
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
        self.assertEqual(selected["xtbloom-cpu"], [(32, 0.5), (62, 0.5), (122, 0.5)])
        self.assertEqual(selected["xtb"], [(62, 0.5)])

    def test_plot_merges_artifacts_and_draws_without_gpu(self) -> None:
        """Full plot path runs in Agg mode from synthetic artifacts."""
        if os.environ.get("XTBLOOM_RUN_PLOT_TEST") != "1":
            self.skipTest("set XTBLOOM_RUN_PLOT_TEST=1 for optional render coverage")
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable")
        latency_scale = {
            "xtbloom-cpu": 1.0,
            "xtbloom-cuda": 1.35,
            "xtb": 2.0,
            "tblite": 3.0,
            "dxtb-cpu": 6.0,
            "dxtb-cuda": 4.5,
        }
        rows = [
            {
                "engine": engine,
                "natoms": natoms,
                "batch_size": batch_size,
                "start_policy": ("auto-warm" if batch_size == 128 else "cold"),
                "effective_start_policy": (
                    "cold"
                    if engine.startswith("dxtb")
                    else ("auto-warm" if batch_size == 128 else "cold")
                ),
                "availability": "available",
                # Keep synthetic series visibly distinct so a rendered layout
                # preview cannot hide reference engines under the xtbloom line.
                "timing": {
                    "median_ms": float(batch_size * natoms * latency_scale[engine])
                },
                "correctness": qualified_correctness(engine),
            }
            for engine in (
                "xtbloom-cpu",
                "xtbloom-cuda",
                "xtb",
                "tblite",
                "dxtb-cpu",
                "dxtb-cuda",
            )
            for natoms in (14, 32)
            for batch_size in (1, 128, 512)
        ]
        unavailable = next(
            row
            for row in rows
            if row["engine"] == "dxtb-cuda"
            and row["natoms"] == 32
            and row["batch_size"] == 512
        )
        unavailable.update(
            availability="error",
            error="CUDA out of memory",
            correctness={"status": "not_run"},
        )
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
                    if row["availability"] != "available":
                        continue
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
            self.assertNotIn("Glyph", completed.stderr)
            for label in (
                "xtbloom CPU",
                "xtbloom CUDA",
                "xTB",
                "tblite",
                "dxtb CPU",
                "dxtb CUDA",
                "unavailable / OOM",
            ):
                self.assertIn(label, svg)

    def test_cold_panel_excludes_autowarm_trajectory_leak(self) -> None:
        """Batch=1 rows leaked by an auto-warm trajectory run must not enter.

        The trajectory invocation also measures a steady-state batch=1 cell
        without a ``job`` tag; only rows whose source artifact recorded a
        ``cold`` start policy may appear in the batch=1 cold-start panel.
        """
        cold_row = {
            "engine": "xtbloom-cpu",
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
            nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
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
            nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
            "auto-warm",
        )
        self.assertEqual(runner.invocations, 2)
        self.assertEqual(runner.modes, ["fresh", "warm"])
        self.assertEqual(fragment["effective_start_policy"], "auto-warm")

    def test_cold_xtbloom_records_fresh_initialization_inside_public_call(self) -> None:
        """Selecting FRESH is untimed, but xtbloom initializes it during compute."""

        class FakeXTBloomRunner:
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
            FakeXTBloomRunner(),
            (0, 1),
            nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
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
            nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
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
            nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
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

    def test_historical_reference_requires_explicit_opt_in(self) -> None:
        """Revision mismatch remains rejected unless the caller opts in."""
        reference = nce.ReferenceArtifact(
            path=Path("reference.json"),
            sha256="a" * 64,
            engine="tblite",
            metadata={"commit": {"head": "old"}},
            rows={},
        )
        with self.assertRaisesRegex(nce.BenchmarkError, "same clean HEAD"):
            nce.validate_reference_revision_policy(reference, "new", False)

    def test_matching_historical_reference_contract_is_accepted(self) -> None:
        """Explicit reuse accepts a complete deterministic workload identity."""
        metadata = artifact_metadata("cold")
        metadata["commit"] = {"head": "old", "dirty": False}
        reference = nce.ReferenceArtifact(
            path=Path("reference.json"),
            sha256="a" * 64,
            engine="tblite",
            metadata=metadata,
            rows={
                (14, 1): {
                    "natoms": 14,
                    "batch_size": 1,
                    "total_atoms_in_batch": 14,
                    "requested_properties": ["energy", "forces"],
                    "workload_seed": 14001,
                }
            },
        )
        nce.validate_reference_revision_policy(reference, "new", True)

    def test_historical_reference_rejects_workload_mismatch(self) -> None:
        """The opt-in does not permit a different workload contract."""
        metadata = artifact_metadata("cold")
        metadata["commit"] = {"head": "old", "dirty": False}
        valid_row = {
            "natoms": 14,
            "batch_size": 1,
            "total_atoms_in_batch": 14,
            "requested_properties": ["energy", "forces"],
            "workload_seed": 14001,
        }
        cases = (
            ("perturb_sigma_bohr", "perturbation"),
            ("workload_seed", "workload seed"),
            ("requested_properties", "requested properties"),
            ("total_atoms_in_batch", "batch extent"),
        )
        for field, message in cases:
            with self.subTest(field=field):
                case_metadata = json.loads(json.dumps(metadata))
                row = dict(valid_row)
                if field == "perturb_sigma_bohr":
                    case_metadata["protocol"][field] = 0.03
                else:
                    row[field] = None
                reference = nce.ReferenceArtifact(
                    path=Path("reference.json"),
                    sha256="a" * 64,
                    engine="tblite",
                    metadata=case_metadata,
                    rows={(14, 1): row},
                )
                with self.assertRaisesRegex(nce.BenchmarkError, message):
                    nce.validate_reference_revision_policy(reference, "new", True)

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
                "engine": "xtbloom-cpu",
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
            "engine": "xtbloom-cpu",
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
            "engine": "xtbloom-cpu",
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
        self.assertIn("CPU budget: 4 threads", note)
        self.assertIn("median n = 3", note)
        self.assertIn("uniform benchmark gate", note)
        self.assertIn(r"2\times10^{-3}", note)
        self.assertIn(r"\max_i|\Delta F_i|", note)
        self.assertNotIn("⁻", note)

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

    def test_meson_sync_precedes_fresh_identity_capture(self) -> None:
        """A rebuild must update the target before hashes/introspection are read."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            build = root / "build"
            info = build / "meson-info"
            source.mkdir()
            info.mkdir(parents=True)
            old_target = build / "libxtb-old.so"
            new_target = build / "libxtb-new.so"
            old_target.write_bytes(b"old")
            library = build / "libxtb.so"
            library.symlink_to(old_target.name)

            def write_metadata(version: str) -> None:
                documents = {
                    "meson-info.json": {
                        "meson_version": {"full": "1.11.2"},
                        "directories": {
                            "source": str(source),
                            "build": str(build),
                        },
                    },
                    "intro-projectinfo.json": {
                        "descriptive_name": "xtb",
                        "version": version,
                    },
                    "intro-targets.json": [
                        {
                            "name": "xtb",
                            "id": "xtb@sha",
                            "type": "shared library",
                            "filename": [str(library)],
                        }
                    ],
                    "intro-compilers.json": {
                        "host": {
                            "c": {
                                "id": "gcc",
                                "version": "test",
                                "exelist": [sys.executable],
                            }
                        }
                    },
                    "intro-buildoptions.json": [
                        {"name": "buildtype", "value": "release"}
                    ],
                }
                for name, document in documents.items():
                    (info / name).write_text(json.dumps(document), encoding="utf-8")

            write_metadata("old")

            def fake_run(command: tuple[str, ...], **_kwargs: object) -> str:
                if command[:2] == ("meson", "compile"):
                    self.assertEqual(command[-1], "xtb:shared_library")
                    new_target.write_bytes(b"new")
                    library.unlink()
                    library.symlink_to(new_target.name)
                    write_metadata("new")
                    return "rebuilt"
                if "rev-parse" in command:
                    return "h" * 40
                if "status" in command:
                    return ""
                raise AssertionError(command)

            nce.meson_build_identity.cache_clear()
            nce.native_library_identity.cache_clear()
            with mock.patch.object(nce, "run_text", side_effect=fake_run):
                sync = nce.synchronize_meson_target(library, source)
                identity = nce.meson_build_identity(library, source)
        self.assertEqual(sync["library_sha256"], hashlib.sha256(b"new").hexdigest())
        self.assertEqual(sync["target_selector"], "xtb:shared_library")
        self.assertEqual(identity["project"]["version"], "new")
        self.assertEqual(
            identity["source_target_sync"]["library_sha256"], sync["library_sha256"]
        )
        self.assertEqual(identity["target"]["filename"], [str(new_target.resolve())])

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
                nce.Cell("xtbloom-cpu", 1, 1, 1, 0),
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
                ("xtbloom-cpu", 100.0),
                ("xtb", 1200.0),
                ("tblite", 900.0),
            )
        ]
        self.assertEqual(
            plotters._speedup_range(rows, 128, 62),
            (100.0, 1200.0, 9.0, 12.0, "xTB/tblite"),
        )
        self.assertEqual(
            plotters._format_speedup(3.24, 4.66),
            "3.2\N{EN DASH}4.7\N{MULTIPLICATION SIGN}",
        )
        self.assertEqual(
            plotters._format_speedup(9.04, 12.2),
            "9.0\N{EN DASH}12.2\N{MULTIPLICATION SIGN}",
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

    def test_plotter_ignores_volatile_gpu_telemetry_for_cpu_artifacts(self) -> None:
        """P-state and current clocks may change between valid CPU processes."""
        metadata_a = artifact_metadata()
        metadata_b = artifact_metadata()
        metadata_a["hardware"]["nvidia_smi_runtime"] = "P8, 210 MHz"
        metadata_b["hardware"]["nvidia_smi_runtime"] = "P0, 2750 MHz"

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
            rows, _metadata = plotters.load_rows(paths)
        self.assertEqual(len(rows), 2)

    def test_cuda_runtime_identity_requires_the_same_selected_gpu(self) -> None:
        """CUDA rows cannot be merged across different selected GPU UUIDs."""
        metadata_b = artifact_metadata()
        metadata_b["hardware"]["selected_cuda_device"]["device"]["uuid"] = (
            "different-uuid"
        )
        with self.assertRaisesRegex(plotters.PlotError, "runtime-verified"):
            plotters._engine_runtime_identity(metadata_b, "xtbloom-cuda")

    def test_plotter_rejects_different_gpus_across_cuda_engines(self) -> None:
        """XTBloom and dxtb CUDA rows must share one physical selected GPU."""
        metadata_a = artifact_metadata(designation="dependent_run")
        metadata_b = artifact_metadata(designation="dependent_run")
        selected_b = metadata_b["hardware"]["selected_cuda_device"]
        selected_b["runtime_uuid"] = "other-uuid"
        selected_b["device"]["uuid"] = "other-uuid"
        rows = (
            {
                "engine": "xtbloom-cuda",
                "natoms": 14,
                "batch_size": 1,
                "availability": "error",
                "correctness": {"status": "fail"},
            },
            {
                "engine": "dxtb-cuda",
                "natoms": 32,
                "batch_size": 1,
                "availability": "error",
                "correctness": {"status": "fail"},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a.json", Path(directory) / "b.json"]
            for path, metadata, row in zip(
                paths, (metadata_a, metadata_b), rows, strict=True
            ):
                path.write_text(
                    json.dumps(
                        {"schema_version": 2, "metadata": metadata, "rows": [row]}
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(plotters.PlotError, "different selected GPUs"):
                plotters.load_rows(paths)

    def test_plotter_rejects_mismatched_convergence_contract(self) -> None:
        """Native stopping-control changes cannot be mixed in one figure."""
        metadata_a = artifact_metadata()
        metadata_b = artifact_metadata()
        metadata_b["protocol"]["convergence_contract"]["tblite"][
            "public_accuracy_factor"
        ] = 0.5

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

    def test_checked_in_manifest_selects_only_issue_467_cuda(self) -> None:
        """The public manifest advances CUDA without changing other sources."""
        manifest_path = Path(__file__).with_name("natoms_cross_engine_publication.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = {source["engine"]: source for source in manifest["sources"]}

        cuda = sources["xtbloom-cuda"]
        evidence_root = "benchmarks/evidence/issue-467/2026-08-20-node3"
        self.assertEqual(cuda["measured_date"], "2026-08-20")
        self.assertEqual(
            cuda["source_revision"],
            "e0a3b0d60a75fbc3efe2fc243a75cafee10f3b68",
        )
        self.assertEqual(cuda["evidence_bundle"], f"{evidence_root}/README.md")
        self.assertEqual(cuda["metadata"], f"{evidence_root}/publication-metadata.json")
        self.assertEqual(
            {artifact["panel"]: artifact["csv"] for artifact in cuda["artifacts"]},
            {
                "cold": f"{evidence_root}/xtbloom-cuda-cold.csv",
                "b128": f"{evidence_root}/xtbloom-cuda-b128.csv",
                "b512": f"{evidence_root}/xtbloom-cuda-b512.csv",
            },
        )

        # This snapshot intentionally binds every non-CUDA source field. A future
        # CPU or third-party refresh must update it in the same reviewed change.
        non_cuda_sources = [
            source
            for source in manifest["sources"]
            if source["engine"] != "xtbloom-cuda"
        ]
        canonical_sources = json.dumps(
            non_cuda_sources, sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            hashlib.sha256(canonical_sources).hexdigest(),
            "9fb7afc9b0443ab0d361173d502ffd44f8e56a986c676b89c0651eb3d7b3cd5a",
        )

    def test_publication_manifest_tracks_independent_engine_revisions(self) -> None:
        """A CUDA refresh can replace its rows without relabelling CPU data."""
        panels = (
            ("cold", 1, "cold"),
            ("b128", 128, "auto-warm"),
            ("b512", 512, "cold"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "benchmarks" / "evidence" / "issue-1" / "run"
            evidence.mkdir(parents=True)
            (
                root / "benchmarks" / "evidence" / "legacy-large-artifacts.tsv"
            ).write_text("sha256\tbytes\tsource_commit\tpath\n", encoding="utf-8")
            (evidence / "README.md").write_text("evidence\n", encoding="utf-8")
            sources: list[dict[str, object]] = []
            metadata_sources: list[dict[str, object]] = []
            metadata_path = evidence / "publication-metadata.json"
            reference_path = evidence / "reference.json"
            reference_path.write_text("reference\n", encoding="utf-8")
            reference_digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            for engine_index, engine in enumerate(publication.SUPPORTED_ENGINES):
                artifacts: list[dict[str, str]] = []
                metadata_artifacts: list[dict[str, object]] = []
                for panel, batch_size, _start_policy in panels:
                    csv_path = evidence / f"{engine}-{panel}.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=(
                                "engine",
                                "natoms",
                                "batch_size",
                                "total_atoms_in_batch",
                                "cpu_threads",
                                "device_id",
                                "job",
                                "availability",
                                "median_ms",
                                "mean_ms",
                                "p95_ms",
                                "min_ms",
                                "max_ms",
                                "systems_per_second_at_median",
                                "correctness_status",
                                "cross_engine_status",
                                "max_abs_energy_delta_hartree",
                                "max_abs_force_delta_hartree_per_bohr",
                            ),
                            lineterminator="\n",
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "engine": engine,
                                "natoms": 14,
                                "batch_size": batch_size,
                                "total_atoms_in_batch": 14 * batch_size,
                                "cpu_threads": 16,
                                "device_id": 0,
                                "job": "",
                                "availability": "available",
                                "median_ms": 10.0 + engine_index,
                                "mean_ms": 10.0 + engine_index,
                                "p95_ms": 11.0 + engine_index,
                                "min_ms": 9.0 + engine_index,
                                "max_ms": 11.0 + engine_index,
                                "systems_per_second_at_median": 1.0,
                                "correctness_status": "pass",
                                "cross_engine_status": (
                                    "reference" if engine == "tblite" else "pass"
                                ),
                                "max_abs_energy_delta_hartree": "",
                                "max_abs_force_delta_hartree_per_bohr": "",
                            }
                        )
                    artifact: dict[str, str] = {
                        "panel": panel,
                        "csv": str(csv_path.relative_to(root)),
                    }
                    if engine != "tblite":
                        artifact["reference_artifact_sha256"] = reference_digest
                    artifacts.append(artifact)
                    metadata_artifacts.append(
                        {
                            "panel": panel,
                            "csv": csv_path.name,
                            "batch_size": batch_size,
                            "start_policy": _start_policy,
                            "reference_artifact_sha256": (
                                None if engine == "tblite" else reference_digest
                            ),
                        }
                    )
                revision = ("1" if engine == "xtbloom-cuda" else "0") * 40
                sources.append(
                    {
                        "engine": engine,
                        "measured_date": "2026-08-20",
                        "source_revision": revision,
                        "runtime_identity": f"{engine}-runtime",
                        "evidence_bundle": str(
                            (evidence / "README.md").relative_to(root)
                        ),
                        "metadata": str(metadata_path.relative_to(root)),
                        "artifacts": artifacts,
                    }
                )
                metadata_sources.append(
                    {
                        "engine": engine,
                        "measured_date": "2026-08-20",
                        "source_revision": revision,
                        "runtime_identity": f"{engine}-runtime",
                        "runner_revision": revision,
                        "runner_dirty": False,
                        "source_dirty": False,
                        "evidence_eligibility": "eligible_clean_head",
                        "artifacts": metadata_artifacts,
                    }
                )
            publication_hardware = {
                "hostname": "node3",
                "cpu_model": "AMD EPYC 7K62",
                "cpu_threads": 16,
                "cuda_device": {
                    "name": "NVIDIA GeForce RTX 5090",
                    "uuid": "test-uuid",
                    "driver": "580.95.05",
                },
            }
            publication_protocol = artifact_metadata()["protocol"]
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": "test-protocol",
                        "hardware": publication_hardware,
                        "protocol": publication_protocol,
                        "panel_coordinates": {
                            panel: [14] for panel, _batch, _start in panels
                        },
                        "reference_bindings": {
                            reference_digest: {
                                "kind": "retained-artifact",
                                "path": str(reference_path.relative_to(root)),
                            }
                        },
                        "sources": metadata_sources,
                    }
                ),
                encoding="utf-8",
            )
            checksum_paths = [*evidence.glob("*.csv"), metadata_path, reference_path]

            def write_checksums() -> None:
                checksum_lines = [
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
                    for path in sorted(checksum_paths)
                ]
                (evidence / "SHA256SUMS").write_text(
                    "\n".join(checksum_lines) + "\n", encoding="utf-8"
                )

            write_checksums()
            manifest = {
                "schema_version": 1,
                "publication": {
                    "protocol_id": "test-protocol",
                    "hardware": publication_hardware,
                    "panels": [
                        {
                            "id": panel,
                            "batch_size": batch_size,
                            "start_policy": start_policy,
                        }
                        for panel, batch_size, start_policy in panels
                    ],
                    "protocol": publication_protocol,
                },
                "sources": sources,
            }
            manifest_path = root / "publication.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(publication, "REPOSITORY_ROOT", root):
                rows, metadata = publication.load_publication(manifest_path)
            self.assertEqual(len(rows), 18)
            revisions = {row["engine"]: row["_source_revision"] for row in rows}
            self.assertEqual(revisions["xtbloom-cpu"], "0" * 40)
            self.assertEqual(revisions["xtbloom-cuda"], "1" * 40)
            self.assertEqual(
                metadata["hardware"]["selected_cuda_device"]["device"]["name"],
                "NVIDIA GeForce RTX 5090",
            )
            rendered = publication.render_table(metadata)
            self.assertIn("NVIDIA GeForce RTX 5090", rendered)
            self.assertIn("xtbloom-cuda", rendered)

            original_metadata = metadata_path.read_text(encoding="utf-8")
            tampered_metadata = json.loads(original_metadata)
            tampered_metadata["hardware"]["cuda_device"]["name"] = "other"
            metadata_path.write_text(json.dumps(tampered_metadata), encoding="utf-8")
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(
                    publication.PublicationError, "metadata does not match SHA256SUMS"
                ),
            ):
                publication.load_publication(manifest_path)
            metadata_path.write_text(original_metadata, encoding="utf-8")

            tampered = json.loads(json.dumps(manifest))
            tampered["publication"]["hardware"]["cuda_device"]["name"] = "other"
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(
                    publication.PublicationError, "publication metadata"
                ),
            ):
                publication.load_publication(manifest_path)

            tampered = json.loads(json.dumps(manifest))
            tampered["sources"][0]["source_revision"] = "f" * 40
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(publication.PublicationError, "source_revision"),
            ):
                publication.load_publication(manifest_path)

            tampered = json.loads(json.dumps(manifest))
            tampered["sources"][0]["artifacts"][0]["reference_artifact_sha256"] = (
                "b" * 64
            )
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(publication.PublicationError, "reference"),
            ):
                publication.load_publication(manifest_path)

            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            unbound_metadata = json.loads(original_metadata)
            unbound_metadata["reference_bindings"][reference_digest]["path"] = (
                "benchmarks/evidence/issue-1/run/missing-reference.json"
            )
            metadata_path.write_text(json.dumps(unbound_metadata), encoding="utf-8")
            write_checksums()
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(
                    publication.PublicationError, "neither retained"
                ),
            ):
                publication.load_publication(manifest_path)
            metadata_path.write_text(original_metadata, encoding="utf-8")
            write_checksums()

            selected_csv = evidence / "xtbloom-cpu-cold.csv"
            original_csv = selected_csv.read_text(encoding="utf-8")
            selected_csv.write_text(
                original_csv.splitlines()[0] + "\n", encoding="utf-8"
            )
            write_checksums()
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(publication.PublicationError, "coordinates"),
            ):
                publication.load_publication(manifest_path)

            selected_csv.write_text(
                original_csv.replace(",pass,pass,", ",fail,pass,", 1),
                encoding="utf-8",
            )
            write_checksums()
            with (
                mock.patch.object(publication, "REPOSITORY_ROOT", root),
                self.assertRaisesRegex(publication.PublicationError, "unqualified"),
            ):
                publication.load_publication(manifest_path)

    def test_publication_protocol_requires_positive_perturbation(self) -> None:
        """A zero geometry perturbation is a different scaling workload."""
        protocol = json.loads(json.dumps(artifact_metadata()["protocol"]))
        protocol["perturb_sigma_bohr"] = 0.0
        with self.assertRaisesRegex(publication.PublicationError, "must be positive"):
            publication._validate_protocol({"protocol": protocol})


if __name__ == "__main__":
    unittest.main()
