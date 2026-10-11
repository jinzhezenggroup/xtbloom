"""Portability and integrity tests for the paper experiment suite."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[1]
SUITE = REPOSITORY / "benchmarks" / "paper"


class PaperSuiteTest(unittest.TestCase):
    """Keep the deployable suite portable and checksum-complete."""

    def test_shell_syntax_and_no_cluster_specific_defaults(self) -> None:
        forbidden = (
            "/home/ztlu",
            "/data/ztlu",
            "/group/software",
            "/feishu/bin",
            "groupServer1",
            "partition=main",
            "RTX 5090",
            "rtx5090-node1",
        )
        shell_files = [
            *SUITE.glob("bin/*.sh"),
            *SUITE.glob("lib/*.sh"),
            *SUITE.glob("slurm/*.sbatch"),
        ]
        self.assertEqual(len(list(SUITE.glob("slurm/*.sbatch"))), 27)
        for path in shell_files:
            subprocess.run(["bash", "-n", str(path)], check=True)
        listed = subprocess.run(
            ["bash", str(SUITE / "bin" / "submit.sh"), "--list"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(len(listed.stdout.splitlines()), 27)
        for path in SUITE.rglob("*"):
            if not path.is_file() or path.name == "SCRIPT_SHA256SUMS":
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{marker!r} leaked into {path}")
        for path in SUITE.glob("slurm/*.sbatch"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("#SBATCH --partition=", text)
            self.assertNotIn("#SBATCH --gres=", text)
            self.assertNotIn("#SBATCH --output=", text)
            self.assertNotIn("#SBATCH --error=", text)

    def test_checksum_inventory_protocol_mirror_and_python_entrypoints(self) -> None:
        expected: dict[str, str] = {}
        for line in (
            (SUITE / "SCRIPT_SHA256SUMS").read_text(encoding="utf-8").splitlines()
        ):
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        actual = {
            f"./{path.relative_to(SUITE).as_posix()}"
            for path in SUITE.rglob("*")
            if path.is_file() and path.name != "SCRIPT_SHA256SUMS"
        }
        self.assertEqual(set(expected), actual)
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256(
                    (SUITE / relative.removeprefix("./")).read_bytes()
                ).hexdigest(),
                digest,
                relative,
            )
        self.assertEqual(
            (REPOSITORY / "docs" / "paper-experiment-plan.md").read_bytes(),
            (SUITE / "protocol" / "paper-experiment-plan.md").read_bytes(),
        )
        self.assertEqual(
            (REPOSITORY / "docs" / "xtbloom_paper_outline.md").read_bytes(),
            (SUITE / "protocol" / "xtbloom-paper-outline.md").read_bytes(),
        )
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for path in SUITE.glob("python/*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            subprocess.run(
                [sys.executable, "-B", str(path), "--help"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

    def test_submitter_uses_only_configured_scheduler_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            data = root / "runs"
            repository.mkdir()
            fake = root / "fake-sbatch"
            log = root / "calls.jsonl"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "log = os.environ['FAKE_SBATCH_LOG']\n"
                "with open(log, 'a', encoding='utf-8') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "print('Submitted batch job 123')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            config = root / "paper.env"
            config.write_text(
                "\n".join(
                    (
                        "export PAPER_RUN_ID=portable-test",
                        f"export PAPER_REPO_ROOT={repository}",
                        f"export PAPER_DATA_ROOT={data}",
                        f"export PAPER_SBATCH={fake}",
                        "export PAPER_CPU_PARTITION=cpu-site",
                        "export PAPER_CPU_HINT=nomultithread",
                        "export PAPER_GPU_PARTITION=gpu-site",
                        "export PAPER_GPU_GRES=gpu:fp64:1",
                        "export PAPER_GPU_CONSTRAINT=sm80",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            environment = {**os.environ, "FAKE_SBATCH_LOG": str(log)}
            subprocess.run(
                [
                    "bash",
                    str(SUITE / "bin" / "submit.sh"),
                    "--config",
                    str(config),
                    "exp1-cpu-native",
                    "exp2-gpu-crossover",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            calls = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(calls), 2)
        self.assertIn("--partition=cpu-site", calls[0])
        self.assertNotIn("--gres=gpu:fp64:1", calls[0])
        self.assertIn("--partition=gpu-site", calls[1])
        self.assertIn("--gres=gpu:fp64:1", calls[1])
        self.assertIn("--constraint=sm80", calls[1])
        self.assertTrue(any("/portable-test/slurm/" in value for value in calls[1]))

    def test_submitter_rejects_data_root_inside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            repository.mkdir()
            config = root / "paper.env"
            config.write_text(
                "\n".join(
                    (
                        "export PAPER_RUN_ID=unsafe-test",
                        f"export PAPER_REPO_ROOT={repository}",
                        f"export PAPER_DATA_ROOT={repository / 'outputs'}",
                        "export PAPER_SBATCH=sbatch",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(SUITE / "bin" / "submit.sh"),
                    "--config",
                    str(config),
                    "exp1-cpu-native",
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cannot be inside", completed.stderr)

    def test_cuda_runtime_resolver_supports_explicit_and_multiarch_layouts(
        self,
    ) -> None:
        module_path = SUITE / "python" / "resolve_visible_gpu.py"
        spec = importlib.util.spec_from_file_location(
            "paper_resolve_visible_gpu", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # The portability scan below intentionally treats every deployable suite
        # file as UTF-8. Keep this test import from leaving a transient bytecode
        # file inside that source tree when the suite runs without ``python -B``.
        previous_dont_write_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(module.platform, "machine", return_value="aarch64"),
        ):
            root = Path(temporary)
            explicit = root / "site" / "libcudart.so"
            explicit.parent.mkdir()
            explicit.write_bytes(b"explicit")
            self.assertEqual(module.resolve_cudart(root, explicit), explicit.resolve())
            explicit.unlink()
            multiarch = root / "targets" / "aarch64-linux" / "lib" / "libcudart.so.12"
            multiarch.parent.mkdir(parents=True)
            multiarch.write_bytes(b"multiarch")
            self.assertEqual(module.resolve_cudart(root), multiarch.resolve())


class PaperEvidenceRegressionTest(unittest.TestCase):
    """Reject incomplete scientific inputs and non-reproducible evidence."""

    def load(self, name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            name, SUITE / "python" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        with (
            mock.patch.object(sys, "dont_write_bytecode", True),
            mock.patch.object(sys, "path", [str(SUITE / "python"), *sys.path]),
        ):
            spec.loader.exec_module(module)
        return module

    def test_cuda_runtime_matches_host_with_multiple_targets(self) -> None:
        module = self.load("resolve_visible_gpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for arch in ("aarch64", "x86_64"):
                paths[arch] = (
                    root / "targets" / f"{arch}-linux" / "lib" / "libcudart.so"
                )
                paths[arch].parent.mkdir(parents=True)
                paths[arch].write_text(arch)
            for host, target in (
                ("x86_64", "x86_64"),
                ("AMD64", "x86_64"),
                ("aarch64", "aarch64"),
                ("arm64", "aarch64"),
            ):
                with (
                    self.subTest(host=host),
                    mock.patch.object(module.platform, "machine", return_value=host),
                ):
                    self.assertEqual(module.resolve_cudart(root), paths[target])
            paths["x86_64"].unlink()
            with mock.patch.object(module.platform, "machine", return_value="x86_64"):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    module.resolve_cudart(root)
                fallback = root / "lib64" / "libcudart.so"
                fallback.parent.mkdir()
                fallback.write_text("fallback")
                self.assertEqual(module.resolve_cudart(root), fallback)

    def test_archive_rejects_external_symlink_but_accepts_internal(self) -> None:
        module = self.load("archive_bundle")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            (stage / "logs").mkdir(parents=True)
            (stage / "logs" / "environment.txt").write_text(
                "phase=formal\neligibility=eligible\n"
            )
            (stage / "data").write_text("data")
            outside = root / "outside"
            outside.write_text("data")
            link = stage / "link"
            link.symlink_to(outside)

            def seal() -> None:
                files = sorted(
                    p
                    for p in stage.rglob("*")
                    if p.is_file() and p.name != "SHA256SUMS"
                )
                (stage / "SHA256SUMS").write_text(
                    "".join(f"{module.sha256(p)}  {p}\n" for p in files)
                )

            seal()
            self.assertTrue(
                any(
                    "escapes stage" in reason
                    for reason in module.verify_stage_checksum(stage)
                )
            )
            link.unlink()
            link.symlink_to(stage / "data")
            seal()
            self.assertEqual(module.verify_stage_checksum(stage), [])

    def test_freeze_rejects_malformed_and_unknown_subset(self) -> None:
        module = self.load("manifest_freeze")
        row = SimpleNamespace(system_id="bad", subset="main", system=None)
        with self.assertRaisesRegex(module.FreezeError, "malformed"):
            module.validate_selected_contract("qm9_main", [row])
        row.system = SimpleNamespace(
            manifest={
                "sampling_probability": "0.5",
                "sampling_stratum": "s",
                "gfn2_n_ao": "2",
                "gfn2_ao_bin": "Q1",
            },
            strata={},
            input_sha256="a" * 64,
        )
        row.subset = "unlabelled"
        with self.assertRaisesRegex(module.FreezeError, "unexpected subset"):
            module.validate_selected_contract("omol25", [row])
        for subset in ("main", "stress", "performance"):
            row.subset = subset
            module.validate_selected_contract("omol25", [row])

    def test_peer_selection_uses_dataset_failure_categories(self) -> None:
        module = self.load("failure_peer_probe")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            for category in (
                "scc_not_converged",
                "eigensolver_or_numerical_failure",
                "unknown_error",
            ):
                rows = [
                    {
                        "program": {"engine": "xtbloom", "backend": "cpu"},
                        "status": {"category": status},
                        "input": {"system_id": name},
                    }
                    for name, status in (("healthy", "success"), ("failed", category))
                ]
                path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows))
                self.assertEqual(module.choose_ids(path, "cpu"), ("healthy", "failed"))

    def test_process_pool_rejects_duplicate_reference_and_invalid_counts(self) -> None:
        module = self.load("process_pool_matrix")
        for value in ("", "0", "1,-2"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                module.parse_csv_ints(value)
        self.assertEqual(module.parse_csv_ints("1,4"), (1, 4))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {
                "program": {"engine": "xtb"},
                "status": {"category": "success"},
                "input": {"dataset": "qm9", "system_id": "one"},
            }
            path = root / "results.jsonl"
            path.write_text("\n" + json.dumps(row) + "\n\n")
            self.assertEqual(module.load_reference(root, "xtb"), {("qm9", "one"): row})
            with path.open("a") as handle:
                handle.write(json.dumps(row))
            with self.assertRaisesRegex(module.PoolError, "duplicate"):
                module.load_reference(root, "xtb")

    def test_wide_selection_is_distinct_with_overlapping_bins(self) -> None:
        module = self.load("performance_matrix")
        items = [
            SimpleNamespace(
                system_id=str(i),
                system=SimpleNamespace(manifest={"gfn2_n_ao": "5"}, strata={}),
            )
            for i in range(3)
        ]
        bins = ((1, 10, "a"), (4, 8, "b"))
        selected = module.controlled_wide(items, bins, 3, "seed")
        self.assertEqual(len({item.system_id for item in selected}), 3)
        with self.assertRaisesRegex(module.PerformanceError, "only 3"):
            module.controlled_wide(items, bins, 4, "seed")

    def test_resource_probe_cleanup_before_start(self) -> None:
        module = self.load("performance_matrix")
        probe = module.ResourceProbe(None)
        probe.stop("cpu", 0)
        self.assertFalse(probe.thread.is_alive())

    def test_warm_bootstrap_uses_paired_trajectory_repetitions(self) -> None:
        module = self.load("warm_trajectory")
        calls = []

        def run(warm: bool) -> list[dict[str, float]]:
            calls.append(warm)
            return [
                {"elapsed_ms": value / (2 if warm else 1)} for value in (1, 10, 100)
            ]

        pairs = module.paired_trajectories(run, 2, 4)
        self.assertEqual(calls, [False, True, True, False] * 3)
        self.assertEqual(len(pairs), 4)
        result = module.paired_speedup_interval(pairs, 100, "seed")
        self.assertEqual(result["warm_speedup"], 2)
        self.assertEqual(result["warm_speedup_bootstrap_95_ci"], [2, 2])
        self.assertEqual(result["fresh_median_trajectory_ms"], 111)
        with self.assertRaises(ValueError):
            module.paired_speedup_interval(pairs[:1], 100, "seed")
        with self.assertRaises(ValueError):
            module.paired_speedup_interval(pairs, 0, "seed")
        pairs[0]["warm"][0]["elapsed_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            module.paired_speedup_interval(pairs, 100, "seed")

    def test_warm_cli_rejects_zero_bootstrap_before_loading_runtime(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SUITE / "python" / "warm_trajectory.py"),
                "--repo",
                "/unused",
                "--library",
                "/unused",
                "--trajectory",
                "/unused",
                "--backend",
                "cpu",
                "--max-scc-iterations",
                "500",
                "--electronic-temperature-kelvin",
                "300",
                "--bootstrap-samples",
                "0",
                "--output",
                "/unused",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--bootstrap-samples must be positive", completed.stderr)

    def test_optional_shell_guards_do_not_trigger_errexit(self) -> None:
        # A false non-final operand of an AND list is exempt from Bash errexit.
        completed = subprocess.run(
            [
                "bash",
                "-Eeuc",
                'DXTB_ARGS=(); [[ -n "${PAPER_DXTB_SOURCE:-}" ]] && '
                'DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE"); '
                'DXTB_GATE=(); [[ "${PAPER_REQUIRE_DXTB:-0}" == 1 ]] && '
                "DXTB_GATE=(--require-dxtb); printf reached",
            ],
            capture_output=True,
            text=True,
            env={"PATH": os.environ["PATH"]},
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "reached")

    def test_config_requires_hardware_identity_and_contained_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            config = root / "paper.env"
            template = (SUITE / "config" / "paper.env.example").read_text()
            for phase, hardware, run_id, expected in (
                ("formal", "CHANGE-ME", "safe", "PAPER_HARDWARE_ID"),
                ("formal", "", "safe", "PAPER_HARDWARE_ID"),
                ("development", "host", ".", "unsafe PAPER_RUN_ID"),
                ("development", "host", "..", "unsafe PAPER_RUN_ID"),
                ("formal", "host", "safe", None),
            ):
                config.write_text(
                    template
                    + f"\nPAPER_PHASE={phase}\nPAPER_HARDWARE_ID={hardware}\n"
                    + f"PAPER_RUN_ID={run_id}\nPAPER_REPO_ROOT={repo}\n"
                    + f"PAPER_DATA_ROOT={root / 'runs'}\n"
                    + f"PAPER_PYTHON={sys.executable}\n"
                    + "PAPER_NUMACTL=/bin/true\n"
                    + "PAPER_CPU_SCALING_GOVERNOR=performance\n"
                )
                completed = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; PAPER_CONFIG=$2; paper_load_config',
                        "test",
                        str(SUITE / "lib" / "common.sh"),
                        str(config),
                    ],
                    capture_output=True,
                    text=True,
                )
                with self.subTest(phase=phase, hardware=hardware, run_id=run_id):
                    if expected:
                        self.assertEqual(completed.returncode, 2)
                        self.assertIn(expected, completed.stderr)
                    else:
                        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cpu_affinity_never_pins_empty_mask(self) -> None:
        module = self.load("performance_matrix")
        with mock.patch.object(module.os, "sched_setaffinity") as set_affinity:
            module.pin_cpu_budget("cpu", {0, 1}, [], 1)
            set_affinity.assert_not_called()
            module.pin_cpu_budget("cpu", {0, 1}, [0, 1], 1)
            set_affinity.assert_called_once_with(0, {0})

    def test_finite_difference_requires_frozen_input_hash(self) -> None:
        module = self.load("finite_difference_check")
        system = SimpleNamespace(input_sha256="abc")
        for row in ({}, {"input_sha256": None}, {"input_sha256": "other"}):
            with self.assertRaisesRegex(module.FiniteDifferenceError, "hash mismatch"):
                module.verify_input_hash("system", row, system)
        module.verify_input_hash("system", {"input_sha256": "abc"}, system)

    def test_warm_cli_retains_repeats_and_rejects_nonfinite_forces(self) -> None:
        module = self.load("warm_trajectory")
        # Load before the scoped module patch so NumPy is not unloaded afterward.
        __import__("numpy")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "trajectory.json"
            trajectory.write_text(
                json.dumps(
                    {
                        "atomic_numbers": [1],
                        "frames_bohr": [[[0, 0, i]] for i in range(3)],
                    }
                )
            )
            library = root / "lib.so"
            library.write_text("fixture")
            output = root / "output.json"
            fake = ModuleType("xtbloom")
            fake.__file__ = str(root / "fake.py")
            calculator = mock.Mock()
            result = SimpleNamespace(
                energy=1.0,
                forces=[0, 0, 0],
                scc_iterations=3,
                scc_status=0,
                scc_converged=True,
            )
            calculator.singlepoint.side_effect = lambda: result
            fake.Calculator = mock.Mock(return_value=calculator)
            argv = [
                "warm_trajectory",
                "--repo",
                str(root),
                "--library",
                str(library),
                "--trajectory",
                str(trajectory),
                "--backend",
                "cpu",
                "--max-scc-iterations",
                "500",
                "--electronic-temperature-kelvin",
                "300",
                "--bootstrap-samples",
                "10",
                "--warmups",
                "0",
                "--repetitions",
                "2",
                "--output",
                str(output),
            ]
            for nonfinite in (False, True):
                result.forces = [0, float("nan") if nonfinite else 0, 0]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.dict(sys.modules, {"xtbloom": fake}),
                    mock.patch.dict(os.environ),
                    mock.patch.object(sys, "path", list(sys.path)),
                ):
                    self.assertEqual(module.main(), 1 if nonfinite else 0)
                document = json.loads(output.read_text())
                self.assertEqual(len(document["paired_repetitions"]), 2)
                self.assertEqual(
                    [p["order"] for p in document["paired_repetitions"]],
                    [["fresh", "warm"], ["warm", "fresh"]],
                )
                self.assertEqual(document["passed"], not nonfinite)
                output.unlink()
            self.assertEqual(fake.Calculator.call_count, 8)
            self.assertEqual(calculator.close.call_count, 8)

    def test_second_hardware_guard_reads_the_recorded_identity_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm").mkdir()
            (root / "lib").mkdir()
            run = root / "runs"
            identity = run / "p0a-canonical-gpu" / "derived" / "hardware-identity.tsv"
            identity.parent.mkdir(parents=True)
            host = subprocess.check_output(["hostname"], text=True).strip()
            identity.write_text(f"primary-id\t{host}\tGPU-1\tExample GPU\n")
            script = root / "slurm" / "second.sbatch"
            script.write_text(
                (SUITE / "slurm" / "45_si_second_hardware.sbatch").read_text()
            )
            (root / "lib" / "common.sh").write_text(
                "paper_begin() { :; }\npaper_require_complete() { :; }\n"
                "paper_check_gpu() { PAPER_RESOLVED_GPU_UUID=GPU-2; "
                "PAPER_RESOLVED_GPU_NAME=GPU; }\n"
                'paper_require_file() { test -f "$1"; }\n'
                'paper_die() { echo "$*" >&2; exit 2; }\n'
            )
            completed = subprocess.run(
                ["bash", str(script)],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PAPER_RUN_ROOT": str(run),
                    "PAPER_RUN_SECOND_HARDWARE": "1",
                    "PAPER_HARDWARE_ID": "primary-id",
                    "PAPER_SECOND_HARDWARE_ID": "second-id",
                    "PAPER_SECOND_GPU_REGEX": "GPU",
                },
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("reuses the primary host or GPU UUID", completed.stderr)

    def test_each_timed_sample_is_qualified_before_overwrite(self) -> None:
        module = self.load("performance_matrix")

        def qualification(output: dict[str, float]) -> dict[str, object]:
            if output["error"] > 1:
                raise module.PerformanceError("sample exceeds gate")
            return {
                "status": "pass",
                "system_outcomes": [output.get("status", 0)],
                "performance_claim_eligible": True,
                "missing_reference_ids": [],
                "output_contract_failures": [],
                "max_abs_energy_error_hartree": output["error"],
            }

        invoke = mock.Mock()
        with self.assertRaisesRegex(module.PerformanceError, "exceeds gate"):
            module.qualified_measurements(
                invoke,
                mock.Mock(side_effect=[{"error": 0.1}, {"error": 2}, {"error": 0.1}]),
                qualification,
                3,
            )
        self.assertEqual(invoke.call_count, 2)
        with self.assertRaisesRegex(module.PerformanceError, "outcomes changed"):
            module.qualified_measurements(
                mock.Mock(),
                mock.Mock(
                    side_effect=[
                        {"error": 0.1, "status": 0},
                        {"error": 0.1, "status": 7},
                    ]
                ),
                qualification,
                2,
            )
        measured, _, result = module.qualified_measurements(
            mock.Mock(),
            mock.Mock(side_effect=[{"error": 0.5}, {"error": 0.1}]),
            qualification,
            2,
        )
        self.assertEqual(len(measured), 2)
        self.assertEqual(result["checked_samples"], 2)
        self.assertEqual(result["max_abs_energy_error_hartree"], 0.5)

    def test_missing_reference_and_failed_timing_never_become_eligible(self) -> None:
        module = self.load("performance_matrix")
        selected = [
            SimpleNamespace(
                system_id="one",
                dataset="qm9",
                system=SimpleNamespace(atomic_numbers=[1]),
            )
        ]
        outputs = {
            "energies_hartree": [-1.0],
            "per_system_status": [0],
            "scc_converged": [1],
        }
        missing = module.qualify(
            selected, outputs, {}, "xtbloom", 5e-7, 5e-6, 5e-7, "energy", False
        )
        self.assertEqual(missing["status"], "unavailable")
        self.assertFalse(missing["performance_claim_eligible"])
        passed = {
            "status": "pass",
            "performance_claim_eligible": True,
            "max_abs_energy_error_hartree": 0,
        }
        failed = {
            "status": "fail",
            "performance_claim_eligible": False,
            "max_abs_energy_error_hartree": 1,
        }
        for early in (missing, failed):
            for documents in ((early, passed, passed), (passed, early, passed)):
                combined = module.combine_qualifications(*documents)
                self.assertFalse(combined["performance_claim_eligible"])
                self.assertNotEqual(combined["status"], "pass")
                self.assertFalse(
                    module.row_claim_eligible(
                        {
                            "availability": "available",
                            "claim_eligible": True,
                            "correctness": combined,
                            "memory": {"status": "available"},
                        }
                    )
                )
        self.assertTrue(
            module.combine_qualifications(passed, passed)["performance_claim_eligible"]
        )
        self.assertFalse(
            module.row_claim_eligible(
                {
                    "availability": "available",
                    "claim_eligible": True,
                    "correctness": passed,
                    "memory": {"status": "unavailable"},
                }
            )
        )

    def test_figures_exclude_unqualified_speedup_rows(self) -> None:
        module = self.load("make_figure_inputs")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for engine, eligible in (("xtb", True), ("xtbloom-ragged", False)):
                rows.append(
                    {
                        "suite": "cpu-native",
                        "dataset": "qm9",
                        "ao_bin": "1-64",
                        "batch_size": 1,
                        "cpu_threads": 1,
                        "engine": engine,
                        "backend": "cpu",
                        "memory_mode": "host",
                        "availability": "available",
                        "claim_eligible": eligible,
                        "correctness": {
                            "status": "pass",
                            "performance_claim_eligible": eligible,
                        },
                        "memory": {"status": "available"},
                        "steady_fresh": {"raw_ms": [1, 1], "median_ms": 1},
                    }
                )
            (root / "timing.json").write_text(json.dumps({"rows": rows}))
            output = module.performance_rows(root)
            candidate = next(row for row in output if row["engine"] == "xtbloom-ragged")
            self.assertFalse(candidate["claim_eligible"])
            self.assertNotIn("speedup_vs_xtb", candidate)

    def test_comparison_workload_requires_ordered_ids_and_complete_masks(self) -> None:
        module = self.load("paper_runtime")
        row = {
            "batch_size": 2,
            "system_ids": ["a", "b"],
            "correctness": {
                "system_outcomes": [
                    {"system_id": "a", "successful": True},
                    {"system_id": "b", "successful": False},
                ]
            },
        }
        self.assertEqual(module.comparison_workload(row), (("a", "b"), (True, False)))
        for field, value in (("system_ids", ["a", "a"]), ("batch_size", 1)):
            with self.subTest(field=field):
                self.assertIsNone(module.comparison_workload({**row, field: value}))
        for outcomes in (
            [],
            [{"system_id": "b", "successful": True}],
            [
                {"system_id": "a", "successful": 1},
                {"system_id": "b", "successful": False},
            ],
        ):
            self.assertIsNone(
                module.comparison_workload(
                    {**row, "correctness": {"system_outcomes": outcomes}}
                )
            )

    def test_speedups_require_matched_inputs_success_masks_and_claims(self) -> None:
        module = self.load("make_figure_inputs")

        def row(engine: str, backend: str = "cpu") -> dict:
            return {
                "suite": "gpu-crossover",
                "dataset": "omol25",
                "ao_bin": "1-64",
                "batch_size": 2,
                "cpu_threads": 1,
                "engine": engine,
                "backend": backend,
                "memory_mode": "host",
                "system_ids": ["a", "b"],
                "availability": "available",
                "claim_eligible": True,
                "memory": {"status": "available"},
                "correctness": {
                    "status": "pass",
                    "performance_claim_eligible": True,
                    "system_outcomes": [
                        {"system_id": "a", "successful": True},
                        {"system_id": "b", "successful": False},
                    ],
                },
                "steady_fresh": {"raw_ms": [2, 2], "median_ms": 2},
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in (
                "matched",
                "different_inputs",
                "reordered",
                "different_mask",
                "missing_outcomes",
                "unqualified",
                "different_threads",
                "different_source",
            ):
                with self.subTest(mode=mode):
                    baseline = row("xtbloom-ragged")
                    candidate = row("xtbloom-ragged", "cuda")
                    if mode == "different_inputs":
                        candidate["system_ids"][1] = "c"
                        candidate["correctness"]["system_outcomes"][1]["system_id"] = (
                            "c"
                        )
                    elif mode == "reordered":
                        candidate["system_ids"].reverse()
                        candidate["correctness"]["system_outcomes"].reverse()
                    elif mode == "different_mask":
                        candidate["correctness"]["system_outcomes"][1]["successful"] = (
                            True
                        )
                    elif mode == "missing_outcomes":
                        candidate["correctness"].pop("system_outcomes")
                    elif mode == "unqualified":
                        candidate["claim_eligible"] = False
                    elif mode == "different_threads":
                        candidate["cpu_threads"] = 2
                    (root / "a.json").write_text(
                        json.dumps(
                            {
                                "rows": [baseline]
                                if mode == "different_source"
                                else [baseline, candidate]
                            }
                        )
                    )
                    if mode == "different_source":
                        (root / "b.json").write_text(json.dumps({"rows": [candidate]}))
                    output = module.performance_rows(root)
                    module.annotate_gpu_throughput(output)
                    gpu = next(item for item in output if item["backend"] == "cuda")
                    self.assertEqual("speedup_vs_cpu" in gpu, mode == "matched")
                    self.assertEqual(
                        gpu["gpu_cpu_throughput_ratio"],
                        1.0 if mode == "matched" else None,
                    )
            (root / "b.json").unlink()
            # CPU engine/sequential comparisons use the identical matching gate.
            for engine, label in (
                ("xtb", "speedup_vs_xtb"),
                ("xtbloom-sequential", "speedup_vs_sequential"),
            ):
                baseline, candidate = row(engine), row("xtbloom-ragged")
                (root / "a.json").write_text(
                    json.dumps({"rows": [baseline, candidate]})
                )
                output = module.performance_rows(root)
                self.assertEqual(output[1][label], 1.0)
                candidate["correctness"]["system_outcomes"][0]["successful"] = False
                (root / "a.json").write_text(
                    json.dumps({"rows": [baseline, candidate]})
                )
                self.assertNotIn(label, module.performance_rows(root)[1])

    def test_process_pool_ratios_require_matched_successful_inputs(self) -> None:
        module = self.load("make_figure_inputs")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "si-cpu-process-pool" / "raw"
            stage.mkdir(parents=True)
            outcomes = [{"system_id": "a", "successful": True}]
            common = {
                "dataset": "qm9",
                "ao_bin": "1-64",
                "batch_size": 1,
                "system_ids": ["a"],
                "availability": "available",
                "claim_eligible": True,
                "correctness": {
                    "status": "pass",
                    "performance_claim_eligible": True,
                    "system_outcomes": outcomes,
                },
            }
            baseline = {
                **common,
                "cpu_threads": 1,
                "memory": {"status": "available"},
                "steady_fresh": {"raw_ms": [1, 1], "median_ms": 1},
            }
            pool = {
                **common,
                "engine": "xtb",
                "processes": 1,
                "raw_samples": [{"wall_ms": 2}, {"wall_ms": 2}],
                "summary": {"median_wall_ms": 2},
            }
            (stage / "qm9-xtbloom.json").write_text(json.dumps({"rows": [baseline]}))
            (stage / "qm9-xtb.json").write_text(json.dumps({"rows": [pool]}))
            self.assertEqual(
                module.process_pool_comparisons(root)[0]["xtbloom_speedup"], 2.0
            )
            for key, value in (("system_ids", ["b"]), ("correctness", {})):
                (stage / "qm9-xtb.json").write_text(
                    json.dumps({"rows": [{**pool, key: value}]})
                )
                self.assertEqual(module.process_pool_comparisons(root), [])

    def test_governor_is_verified_for_every_assigned_cpu(self) -> None:
        module = self.load("resolve_cpu_policy")

        def read(path: Path) -> str | None:
            return (
                None if "cpu1/cpufreq/scaling_governor" in str(path) else "performance"
            )

        with (
            mock.patch.object(
                sys, "argv", ["policy", "--expected-governor", "performance"]
            ),
            mock.patch.object(module.os, "sched_getaffinity", return_value={0, 1}),
            mock.patch.object(module, "read", side_effect=read),
            self.assertRaisesRegex(RuntimeError, "CPU 1 governor None"),
        ):
            module.main()

    def test_pool_startup_and_inference_timeouts_are_bounded(self) -> None:
        module = self.load("process_pool_matrix")
        connection = mock.Mock()
        connection.poll.return_value = False
        with self.assertRaisesRegex(module.PoolError, "startup timed out"):
            module.receive_ready([connection], 0.001)
        with (
            mock.patch.object(module.time, "monotonic", side_effect=[0, 0, 2]),
            mock.patch.object(module.time, "sleep"),
            mock.patch.object(module, "current_rss_bytes", return_value=10),
            self.assertRaisesRegex(module.PoolError, "inference timed out"),
        ):
            module.collect_responses((connection,), (123,), 1, 0)
        connection.poll.return_value = True
        connection.recv.side_effect = EOFError
        with self.assertRaisesRegex(module.PoolError, "before startup"):
            module.receive_ready([connection], 1)
        with self.assertRaisesRegex(module.PoolError, "during inference"):
            module.collect_responses((connection,), (123,), 1, 0)

    def test_pool_failure_closes_pipes_and_terminates_owned_children(self) -> None:
        module = self.load("process_pool_matrix")
        connection = mock.Mock()
        connection.send.side_effect = BrokenPipeError
        process = mock.Mock(exitcode=-15)
        process.is_alive.side_effect = [True, True, False]
        with (
            self.assertRaisesRegex(ValueError, "primary failure"),
            module.pool_lifecycle([connection], [process]),
        ):
            raise ValueError("primary failure")
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 3)
        self.assertTrue(
            all(call.kwargs["timeout"] == 5 for call in process.join.call_args_list)
        )
        connection.close.assert_called_once()
        process = mock.Mock(exitcode=2)
        process.is_alive.return_value = False
        with (
            self.assertRaisesRegex(module.PoolError, "worker exited with 2"),
            module.pool_lifecycle([], [process]),
        ):
            pass


if __name__ == "__main__":
    unittest.main()
