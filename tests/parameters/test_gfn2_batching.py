"""Verify generated exact-AO metadata and shared finite-queue policy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPOSITORY = Path(__file__).resolve().parents[2]
PARAMETER_JSON = REPOSITORY / "data" / "parameters" / "gfn2.json"
DATA_DIR = PARAMETER_JSON.parent
BASIS_MODULE = REPOSITORY / "python" / "xtbloom" / "_gfn2_basis.py"
BATCHING_MODULE = REPOSITORY / "python" / "xtbloom" / "batching.py"
GENERATOR_PATH = REPOSITORY / "tools" / "parameters" / "generate_gfn2.py"


def _copy_generator_tree(root: Path) -> tuple[Path, Path]:
    """Create an isolated canonical parameter tree for generator CLI checks."""
    generator_path = root / "tools" / "parameters" / "generate_gfn2.py"
    generator_path.parent.mkdir(parents=True)
    shutil.copyfile(GENERATOR_PATH, generator_path)

    parameter_dir = root / "data" / "parameters"
    parameter_dir.mkdir(parents=True)
    for filename in (
        GENERATOR.RAW_FILENAME,
        GENERATOR.JSON_FILENAME,
        GENERATOR.HEADER_FILENAME,
        GENERATOR.MANIFEST_FILENAME,
    ):
        shutil.copyfile(DATA_DIR / filename, parameter_dir / filename)

    package_dir = root / "python" / "xtbloom"
    package_dir.mkdir(parents=True)
    basis_module = package_dir / "_gfn2_basis.py"
    shutil.copyfile(BASIS_MODULE, basis_module)
    return generator_path, basis_module


def _load_module(name: str, path: Path) -> ModuleType:
    """Inspect generated and shared modules without importing the native package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASIS = _load_module("xtbloom_test_gfn2_basis", BASIS_MODULE)
BATCHING = _load_module("xtbloom_test_batching", BATCHING_MODULE)
GENERATOR = _load_module("xtbloom_test_generate_gfn2", GENERATOR_PATH)


class Gfn2BatchingTests(unittest.TestCase):
    """Keep basis counts and queue policy tied to their canonical inputs."""

    def test_generated_counts_match_every_canonical_element_shell(self) -> None:
        """Verify exact shell degeneracies and the byte-level JSON digest."""
        payload = PARAMETER_JSON.read_bytes()
        document = json.loads(payload)
        expected = {
            element["atomic_number"]: sum(
                2 * shell["angular_momentum"] + 1 for shell in element["shells"]
            )
            for element in document["elements"]
        }
        self.assertEqual(BASIS.GFN2_AO_COUNTS, expected)
        self.assertEqual(BASIS.BASIS_SHA256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            GENERATOR.render_basis_module(payload), BASIS_MODULE.read_bytes()
        )

    def test_canonical_check_covers_the_python_basis_module(self) -> None:
        """Default canonical checks also reject stale generated Python data."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn2-check-") as directory:
            root = Path(directory)
            generator_path, basis_module = _copy_generator_tree(root)
            result = subprocess.run(
                (sys.executable, str(generator_path), "--check"),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            basis_module.write_bytes(b"stale generated basis module\n")
            result = subprocess.run(
                (sys.executable, str(generator_path), "--check"),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(GENERATOR.BASIS_MODULE_FILENAME, result.stderr)

    def test_custom_output_does_not_write_python_data_unless_requested(self) -> None:
        """Keep custom parameter generations isolated from the package tree."""
        with tempfile.TemporaryDirectory(prefix="xtbloom-gfn2-output-") as directory:
            root = Path(directory)
            generator_path, basis_module = _copy_generator_tree(root)
            preserved_contents = b"leave the canonical package output alone\n"
            basis_module.write_bytes(preserved_contents)

            custom_output = root / "generated-parameters"
            custom_output.mkdir()
            for filename in (
                GENERATOR.RAW_FILENAME,
                GENERATOR.MANIFEST_FILENAME,
            ):
                shutil.copyfile(DATA_DIR / filename, custom_output / filename)

            result = subprocess.run(
                (
                    sys.executable,
                    str(generator_path),
                    "--output-dir",
                    str(custom_output),
                ),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(basis_module.read_bytes(), preserved_contents)

            explicit_python_output = root / "generated-python"
            result = subprocess.run(
                (
                    sys.executable,
                    str(generator_path),
                    "--output-dir",
                    str(custom_output),
                    "--python-output-dir",
                    str(explicit_python_output),
                ),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (explicit_python_output / GENERATOR.BASIS_MODULE_FILENAME).read_bytes(),
                GENERATOR.render_basis_module(
                    (custom_output / GENERATOR.JSON_FILENAME).read_bytes()
                ),
            )
            self.assertEqual(basis_module.read_bytes(), preserved_contents)

    def test_policy_auto_groups_only_cross_batch_ao_reuse(self) -> None:
        """Use exact-AO grouping only when it can unite original batches."""
        resolve = BATCHING.resolve_policy
        self.assertEqual(
            resolve(
                "auto",
                ineligible_reason=None,
                ao_counts=(8, 4, 8, 6),
                original_batches=((0, 1), (2, 3)),
            ),
            ("exact-ao", "cross_batch_exact_ao"),
        )
        self.assertEqual(
            resolve(
                "auto",
                ineligible_reason=None,
                ao_counts=(8, 4, 6),
                original_batches=((0, 1), (2,)),
            ),
            ("original", "no_cross_batch_reuse"),
        )

    def test_policy_auto_keeps_singleton_homogeneous_and_ineligible_queues(
        self,
    ) -> None:
        """Retain the original schedule for queues without a useful grouping."""
        resolve = BATCHING.resolve_policy
        common = {"ineligible_reason": None}
        self.assertEqual(
            resolve(
                "auto",
                **common,
                ao_counts=(8,),
                original_batches=((0,),),
            ),
            ("original", "singleton"),
        )
        self.assertEqual(
            resolve(
                "auto",
                **common,
                ao_counts=(8, 8),
                original_batches=((0,), (1,)),
            ),
            ("original", "homogeneous_ao"),
        )
        self.assertEqual(
            resolve(
                "auto",
                ineligible_reason="warm_start",
                ao_counts=None,
                original_batches=((0, 1),),
            ),
            ("original", "warm_start"),
        )
        self.assertEqual(
            resolve(
                "auto",
                ineligible_reason=None,
                ao_counts=None,
                original_batches=(),
            ),
            ("original", "no_cross_batch_reuse"),
        )

    def test_policy_explicit_modes_and_ineligible_exact_ao(self) -> None:
        """Honor explicit modes and reject exact-AO on unsupported queues."""
        resolve = BATCHING.resolve_policy
        arguments = {
            "ineligible_reason": None,
            "ao_counts": (8, 4),
            "original_batches": ((0, 1),),
        }
        self.assertEqual(
            resolve("original", **(arguments | {"ineligible_reason": "warm_start"})),
            ("original", "explicit_original"),
        )
        self.assertEqual(resolve("off", **arguments), ("original", "explicit_original"))
        self.assertEqual(
            resolve("exact-ao", **arguments), ("exact-ao", "explicit_exact_ao")
        )
        with self.assertRaisesRegex(
            BATCHING.AOGroupingError, "unavailable: warm_start"
        ):
            resolve(
                "exact-ao",
                ineligible_reason="warm_start",
                ao_counts=None,
                original_batches=((0, 1),),
            )
        with self.assertRaises(BATCHING.AOGroupingError):
            resolve("automatic", **arguments)

    def test_benchmark_facade_does_not_import_package_or_numpy(self) -> None:
        """Load the source facade with only the Python standard library."""
        script = "\n".join(
            (
                "import sys",
                "from benchmarks import ao_grouping",
                "assert 'xtbloom' not in sys.modules",
                "assert 'numpy' not in sys.modules",
                "assert ao_grouping.resolve_policy(",
                "    'off', ineligible_reason=None, ao_counts=None,",
                "    original_batches=()",
                ") == ('original', 'explicit_original')",
            )
        )
        subprocess.run(
            (sys.executable, "-S", "-c", script),
            cwd=REPOSITORY,
            check=True,
        )

    def test_benchmark_facade_ignores_another_installed_batching_module(self) -> None:
        """Use this checkout's planner when an unrelated package is loaded."""
        script = "\n".join(
            (
                "import pathlib, sys, types",
                "package = types.ModuleType('xtbloom')",
                "package.__path__ = []",
                "stale = types.ModuleType('xtbloom.batching')",
                "stale.__file__ = '/installed/xtbloom/batching.py'",
                "package.batching = stale",
                "sys.modules['xtbloom'] = package",
                "sys.modules['xtbloom.batching'] = stale",
                "from benchmarks import ao_grouping",
                "assert ao_grouping.AOGroupingError.__module__ != stale.__name__",
                "source = pathlib.Path('python/xtbloom/batching.py').resolve()",
                "loaded = sys.modules[ao_grouping.AOGroupingError.__module__]",
                "assert pathlib.Path(loaded.__file__).resolve() == source",
                "assert 'numpy' not in sys.modules",
            )
        )
        subprocess.run(
            (sys.executable, "-S", "-c", script),
            cwd=REPOSITORY,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
