"""Portability and integrity tests for the paper experiment suite."""

from __future__ import annotations

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
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
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


if __name__ == "__main__":
    unittest.main()
