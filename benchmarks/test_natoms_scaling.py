"""Hardware-free tests for the audit-ready natoms FRESH/WARM protocol."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from benchmarks import natoms_scaling
from benchmarks.tblite_adapter import TbliteAdapter, TbliteState


class FakeRunner:
    """Deterministic public-runner stand-in that records strict mode ordering."""

    def __init__(
        self,
        iterations: dict[str, int] | None = None,
        forces: dict[str, list[float]] | None = None,
    ) -> None:
        self.mode = "fresh"
        self.events: list[tuple[str, str]] = []
        self.compute_options = {"flags": 3}
        self.iterations = iterations or {"fresh": 13, "warm": 2}
        self.forces = forces or {
            "fresh": [0.1, -0.2, 0.3],
            "warm": [0.1, -0.2, 0.3],
            "persistent": [0.1, -0.2, 0.3],
        }
        self.closed = False

    def set_start_mode(self, mode: str) -> None:
        """Record the requested benchmark start mode."""
        self.mode = mode
        self.events.append(("mode", mode))

    def invoke(self) -> None:
        """Record one deterministic invocation in the active mode."""
        self.events.append(("invoke", self.mode))

    def snapshot(self) -> dict[str, Any]:
        """Return deterministic observables for the active mode."""
        self.events.append(("snapshot", self.mode))
        return {
            "energies_hartree": [-10.0],
            "scc_iterations": [self.iterations[self.mode]],
            "scc_converged": [1],
            "per_system_status": [0],
            "forces_hartree_per_bohr": list(self.forces[self.mode]),
        }

    def close(self) -> None:
        """Record that the fake runner released its resources."""
        self.closed = True


def make_reference_document(
    natoms: int = 32, batch_size: int = 1, property_name: str = "force"
) -> dict[str, Any]:
    """Create one fully valid in-memory xtbloom FRESH reference artifact."""
    molecule = natoms_scaling.make_alkane(natoms)
    cell = natoms_scaling.Cell("xtbloom", molecule, batch_size, "cpu", property_name)
    flags = natoms_scaling.public_api.XTBLOOM_COMPUTE_ENERGY
    if property_name == "force":
        flags |= natoms_scaling.public_api.XTBLOOM_COMPUTE_FORCES
    options = {
        "engine": "xtbloom",
        "model": natoms_scaling.public_api.XTBLOOM_MODEL_GFN2_XTB,
        "flags": flags,
        "max_scc_iterations": natoms_scaling.XTBLOOM_CONFORMANCE_MAX_SCC_ITERATIONS,
        "charge_tolerance": natoms_scaling.XTBLOOM_CONFORMANCE_CHARGE_TOLERANCE,
        "energy_tolerance": natoms_scaling.XTBLOOM_CONFORMANCE_ENERGY_TOLERANCE,
        "electronic_temperature_hartree": (
            natoms_scaling.XTBLOOM_CONFORMANCE_ELECTRONIC_TEMPERATURE
        ),
        "cpu_threads": 1,
        "device_id": 0,
        "total_atoms": natoms * batch_size,
    }
    force_reference = (
        [0.001 * index for index in range(3 * natoms * batch_size)]
        if property_name == "force"
        else None
    )
    revision = "a" * 40
    source_git = {"revision": revision, "dirty": False, "path": "/source"}
    identity = {
        "argv": ["reference"],
        "repository": dict(source_git),
        "library": {
            "engine": "xtbloom",
            "path": "/build/libxtbloom.so",
            "sha256": "b" * 64,
            "build": {
                "build_system": "cmake",
                "source": {"git": dict(source_git)},
            },
        },
    }
    row = {
        "availability": "available",
        "engine": "xtbloom",
        "backend": "cpu",
        "memory_mode": "host",
        "molecule": molecule.name,
        "natoms": natoms,
        "batch_size": batch_size,
        "property": property_name,
        "start_mode": "fresh",
        "warmups": 3,
        "repetitions": 5,
        "compute_options": options,
        "workload_identity": natoms_scaling.workload_identity(cell),
        "run_identity": identity,
        "raw_samples": [
            {
                "sample_index": sample_index,
                "start_mode": "fresh",
                "latency_ms": 1.0,
                "energies_hartree": [-10.0] * batch_size,
                "forces_hartree_per_bohr": (
                    list(force_reference) if force_reference is not None else None
                ),
            }
            for sample_index in range(5)
        ],
        "correctness": {
            "status": "pass",
            "energy_reference_hartree": [-10.0] * batch_size,
            "force_reference_hartree_per_bohr": force_reference,
            "energy_atol_hartree": 1.0e-8,
            "force_atol_hartree_per_bohr": 1.0e-7,
            "max_abs_energy_drift_hartree": 0.0,
            "max_abs_force_drift_hartree_per_bohr": (
                0.0 if property_name == "force" else None
            ),
        },
    }
    return natoms_scaling.build_document(
        [row],
        identity,
        natoms_scaling.Protocol("fresh", 3, 5, 1.0e-8, 1.0e-7),
    )


def set_measured_observables(
    row: dict[str, Any],
    energy_samples: list[list[float]],
    force_samples: list[list[float] | None],
) -> None:
    """Replace a dependent row's measured vectors without changing its summary."""
    if len(energy_samples) != len(force_samples):
        raise ValueError("energy and force samples must have equal length")
    row["repetitions"] = len(energy_samples)
    row["raw_samples"] = [
        {
            "sample_index": sample_index,
            "start_mode": row["start_mode"],
            "latency_ms": 1.0,
            "energies_hartree": energies,
            "forces_hartree_per_bohr": forces,
        }
        for sample_index, (energies, forces) in enumerate(
            zip(energy_samples, force_samples, strict=True)
        )
    ]


class NatomsScalingTest(unittest.TestCase):
    """Exercise parser, protocol ordering, provenance, and serialization."""

    def test_run_identity_records_cpu_precision_policy(self) -> None:
        """Distinguish FP64 and adaptive CPU evidence in the environment record."""
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libxtbloom.so"
            library.write_bytes(b"xtbloom")
            with (
                mock.patch.dict(
                    os.environ, {"XTBLOOM_CPU_PRECISION": "adaptive"}, clear=False
                ),
                mock.patch.object(
                    natoms_scaling,
                    "git_state",
                    return_value={"revision": "a" * 40, "dirty": False},
                ),
                mock.patch.object(
                    natoms_scaling,
                    "build_metadata",
                    return_value={"build_system": "test"},
                ),
            ):
                identity = natoms_scaling.collect_run_identity(
                    "xtbloom", library, ["--start-mode", "fresh"], None
                )

        self.assertEqual(
            identity["thread_environment"]["XTBLOOM_CPU_PRECISION"], "adaptive"
        )

    def test_storage_records_absent_uniform_electric_fields(self) -> None:
        """Keep benchmark storage compatible with the public batch builder."""
        molecule = natoms_scaling.make_alkane(14)
        storage = natoms_scaling.make_storage(molecule, 3)
        self.assertEqual(storage.efields, [None, None, None])

    def test_parser_requires_explicit_mode_and_output_paths(self) -> None:
        """Require explicit mode and artifact paths for reproducible runs."""
        parser = natoms_scaling.build_parser()
        arguments = parser.parse_args(
            [
                "--library",
                "libxtbloom.so",
                "--output-json",
                "fresh.json",
                "--output-csv",
                "fresh.csv",
                "--start-mode",
                "fresh",
                "--natoms",
                "32,62",
                "--batch-sizes",
                "1,8",
            ]
        )
        self.assertEqual(arguments.start_mode, "fresh")
        self.assertEqual(arguments.natoms, (32, 62))
        self.assertEqual(arguments.batch_sizes, (1, 8))
        self.assertEqual(arguments.force_atol, 5.0e-7)
        self.assertEqual(arguments.cross_engine_energy_atol, 5.0e-7)
        self.assertEqual(arguments.cross_engine_force_atol, 5.0e-6)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--library", "libxtbloom.so"])

    def test_xtbloom_benchmark_pins_conformance_scc_and_retains_300k(self) -> None:
        """Pin SCC tolerances while retaining the initialized temperature."""
        options = SimpleNamespace(
            max_scc_iterations=250,
            charge_tolerance=1.0e-6,
            energy_tolerance=1.0e-8,
            electronic_temperature=(
                natoms_scaling.XTBLOOM_CONFORMANCE_ELECTRONIC_TEMPERATURE
            ),
        )
        original_temperature = options.electronic_temperature
        natoms_scaling.configure_xtbloom_conformance_scc(options)
        self.assertEqual(
            options.max_scc_iterations,
            natoms_scaling.XTBLOOM_CONFORMANCE_MAX_SCC_ITERATIONS,
        )
        self.assertEqual(
            options.charge_tolerance,
            natoms_scaling.XTBLOOM_CONFORMANCE_CHARGE_TOLERANCE,
        )
        self.assertEqual(
            options.energy_tolerance,
            natoms_scaling.XTBLOOM_CONFORMANCE_ENERGY_TOLERANCE,
        )
        self.assertEqual(options.electronic_temperature, original_temperature)

    def test_nonfresh_runs_require_reference_and_gates_cannot_be_widened(self) -> None:
        """Require FRESH evidence and reject relaxed correctness gates."""
        parser = natoms_scaling.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "lib.so"
            library.write_bytes(b"library")
            reference = root / "fresh.json"
            reference.write_text("{}", encoding="utf-8")

            def arguments(engine: str, mode: str | None) -> list[str]:
                values = [
                    "--engine",
                    engine,
                    "--library",
                    str(library),
                    "--output-json",
                    str(root / f"{engine}-{mode}.json"),
                    "--output-csv",
                    str(root / f"{engine}-{mode}.csv"),
                ]
                if mode is not None:
                    values.extend(("--start-mode", mode))
                return values

            fresh = parser.parse_args(arguments("xtbloom", "fresh"))
            natoms_scaling.validate_arguments(fresh)
            for engine, mode in (
                ("xtbloom", "warm"),
                ("tblite", None),
                ("xtb", None),
            ):
                with (
                    self.subTest(engine=engine),
                    self.assertRaisesRegex(
                        natoms_scaling.BenchmarkError, "require --energy-reference-json"
                    ),
                ):
                    natoms_scaling.validate_arguments(
                        parser.parse_args(arguments(engine, mode))
                    )

            strict = [
                *arguments("xtbloom", "warm"),
                "--energy-reference-json",
                str(reference),
            ]
            natoms_scaling.validate_arguments(parser.parse_args(strict))
            for flag, value in (
                ("--force-atol", "5.1e-7"),
                ("--cross-engine-energy-atol", "5.1e-7"),
                ("--cross-engine-force-atol", "5.1e-6"),
            ):
                with (
                    self.subTest(flag=flag),
                    self.assertRaises(natoms_scaling.BenchmarkError),
                ):
                    natoms_scaling.validate_arguments(
                        parser.parse_args([*strict, flag, value])
                    )

    def test_default_alkanes_have_physical_bonds_and_no_nonbonded_overlap(self) -> None:
        """Keep generated alkane geometries physically separated and bonded."""
        for natoms in natoms_scaling.DEFAULT_NATOMS:
            with self.subTest(natoms=natoms):
                molecule = natoms_scaling.make_alkane(natoms)
                ncarbon = (natoms - 2) // 3
                nhydrogen = 2 * ncarbon + 2
                positions = [
                    tuple(
                        coordinate / natoms_scaling.ANGSTROM_TO_BOHR
                        for coordinate in molecule.positions_bohr[
                            3 * index : 3 * index + 3
                        ]
                    )
                    for index in range(natoms)
                ]
                bonds: set[tuple[int, int]] = set()
                hydrogen = 0
                for carbon in range(ncarbon):
                    count = 3 if carbon in (0, ncarbon - 1) else 2
                    for _ in range(count):
                        bonds.add((hydrogen, nhydrogen + carbon))
                        hydrogen += 1
                for carbon in range(ncarbon - 1):
                    bonds.add((nhydrogen + carbon, nhydrogen + carbon + 1))

                bonded_distances = [
                    math.dist(positions[left], positions[right])
                    for left, right in bonds
                ]
                nonbonded_distances = [
                    math.dist(positions[left], positions[right])
                    for right in range(natoms)
                    for left in range(right)
                    if (left, right) not in bonds
                ]
                self.assertGreaterEqual(min(bonded_distances), 1.089999999)
                self.assertLessEqual(max(bonded_distances), 1.540000001)
                self.assertGreater(min(nonbonded_distances), 1.75)

    def test_dirty_current_run_is_rejected_or_marked_development_only(self) -> None:
        """Reject dirty evidence unless explicitly labeled development-only."""
        identity = {
            "repository": {"revision": "a" * 40, "dirty": True},
            "library": {
                "build": {"source": {"git": {"revision": "a" * 40, "dirty": True}}}
            },
        }
        with self.assertRaisesRegex(natoms_scaling.BenchmarkError, "dirty sources"):
            natoms_scaling.apply_current_evidence_policy(identity, False)
        natoms_scaling.apply_current_evidence_policy(identity, True)
        self.assertEqual(
            identity["evidence_eligibility"]["status"], "development_only_dirty"
        )

    def test_warm_protocol_seeds_once_then_measures_only_warm(self) -> None:
        """Seed WARM state once and exclude that FRESH call from samples."""
        runner = FakeRunner()
        clock_values = iter((0, 1_000_000, 2_000_000, 4_000_000))
        result = natoms_scaling.measure_runner(
            runner,
            natoms_scaling.Protocol("warm", 1, 2, 1.0e-8, 1.0e-7),
            1,
            3,
            clock_ns=lambda: next(clock_values),
        )
        invocations = [event for event in runner.events if event[0] == "invoke"]
        self.assertEqual(
            invocations,
            [
                ("invoke", "fresh"),
                ("invoke", "warm"),
                ("invoke", "warm"),
                ("invoke", "warm"),
            ],
        )
        self.assertEqual(result["seed"]["scc_iterations"], [13])
        self.assertEqual(result["timing"]["samples_ms"], [1.0, 2.0])
        self.assertEqual(
            [sample["start_mode"] for sample in result["raw_samples"]],
            ["warm", "warm"],
        )
        self.assertEqual(
            result["correctness"]["force_reference_hartree_per_bohr"],
            [0.1, -0.2, 0.3],
        )
        self.assertEqual(
            result["correctness"]["max_abs_force_drift_hartree_per_bohr"], 0.0
        )
        self.assertEqual(result["correctness"]["status"], "pass")

    def test_warm_force_vector_mismatch_fails_correctness(self) -> None:
        """Fail WARM correctness when any published force exceeds tolerance."""
        runner = FakeRunner(
            forces={
                "fresh": [0.1, -0.2, 0.3],
                "warm": [0.1, -0.2, 0.300001],
                "persistent": [0.1, -0.2, 0.3],
            }
        )
        clock_values = iter((0, 1_000_000))
        result = natoms_scaling.measure_runner(
            runner,
            natoms_scaling.Protocol("warm", 0, 1, 1.0e-8, 1.0e-7),
            1,
            3,
            clock_ns=lambda: next(clock_values),
        )
        self.assertEqual(result["correctness"]["status"], "fail")
        self.assertAlmostEqual(
            result["correctness"]["max_abs_force_drift_hartree_per_bohr"],
            1.0e-6,
        )

    def test_document_repeats_mode_and_run_identity_per_row(self) -> None:
        """Repeat self-contained mode and provenance identity in each row."""
        identity = {
            "argv": ["python", "natoms_scaling.py", "--start-mode", "fresh"],
            "repository": {"revision": "abc", "dirty": False},
            "library": {"path": "/tmp/libxtbloom.so", "sha256": "123"},
        }
        row = {
            "availability": "available",
            "engine": "xtbloom",
            "start_mode": "fresh",
            "run_identity": identity,
            "raw_samples": [{"latency_ms": 1.0, "scc_iterations": [13]}],
        }
        document = natoms_scaling.build_document(
            [row], identity, natoms_scaling.Protocol("fresh", 3, 5, 1.0e-8, 1.0e-7)
        )
        self.assertEqual(document["run_identity"], identity)
        self.assertEqual(document["protocol"]["start_mode"], "fresh")
        self.assertEqual(document["rows"][0]["run_identity"], identity)
        self.assertEqual(document["rows"][0]["raw_samples"][0]["latency_ms"], 1.0)

    def test_cmake_metadata_records_flags_compiler_source_and_provider(self) -> None:
        """Capture CMake flags, compiler, source, and numerical provider."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            build = root / "build"
            source.mkdir()
            build.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n", encoding="utf-8"
            )
            provider = root / "libblas.so"
            provider.write_bytes(b"blas")
            library = build / "libxtbloom.so"
            library.write_bytes(b"xtbloom")
            cache = build / "CMakeCache.txt"
            cache.write_text(
                "\n".join(
                    (
                        "BUILD_SHARED_LIBS:BOOL=ON",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                        f"CMAKE_CXX_COMPILER:FILEPATH={sys.executable}",
                        "CMAKE_CXX_FLAGS:STRING=-march=x86-64",
                        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                        "CMAKE_GENERATOR:INTERNAL=Ninja",
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}",
                        "XTBLOOM_ENABLE_CUDA:STRING=OFF",
                        f"XTBLOOM_CPU_LINALG_LIBRARY:FILEPATH={provider}",
                    )
                ),
                encoding="utf-8",
            )
            clean = {"path": str(source), "revision": "a" * 40, "dirty": False}
            with mock.patch.object(natoms_scaling, "git_state", return_value=clean):
                metadata = natoms_scaling.build_metadata(library)
        self.assertEqual(metadata["build_system"], "cmake")
        self.assertEqual(metadata["cache_entries"]["CMAKE_BUILD_TYPE"], "Release")
        self.assertEqual(
            metadata["cache_entries"]["CMAKE_CXX_FLAGS_RELEASE"], "-O3 -DNDEBUG"
        )
        self.assertTrue(metadata["compiler"]["is_file"])
        self.assertEqual(
            metadata["dependency_provider"]["sha256"],
            hashlib.sha256(b"blas").hexdigest(),
        )
        self.assertEqual(metadata["source"]["git"], clean)

    def test_meson_metadata_records_build_options_compilers_and_dependencies(
        self,
    ) -> None:
        """Capture hashed Meson introspection and dependency provenance."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            build = root / "build"
            info = build / "meson-info"
            source.mkdir()
            info.mkdir(parents=True)
            (source / "meson.build").write_text(
                "project('tblite', 'fortran')\n", encoding="utf-8"
            )
            library = build / "libtblite.so"
            library.write_bytes(b"tblite")
            provider = root / "libopenblas.so"
            provider.write_bytes(b"openblas")
            documents = {
                "meson-info.json": {
                    "meson_version": {"full": "1.6.0"},
                    "directories": {"source": str(source), "build": str(build)},
                },
                "intro-projectinfo.json": {
                    "descriptive_name": "tblite",
                    "version": "0.7.0",
                },
                "intro-compilers.json": {
                    "host": {
                        "fortran": {
                            "id": "gcc",
                            "version": "13.4.0",
                            "exelist": [sys.executable],
                        }
                    }
                },
                "intro-buildoptions.json": [
                    {"name": "buildtype", "value": "release"},
                    {"name": "optimization", "value": "3"},
                    {"name": "fortran_args", "value": ["-march=x86-64"]},
                ],
                "intro-dependencies.json": [
                    {
                        "name": "openblas",
                        "type": "pkgconfig",
                        "version": "0.3.33",
                        "compile_args": [],
                        "link_args": [str(provider)],
                    }
                ],
            }
            for name, document in documents.items():
                (info / name).write_text(json.dumps(document), encoding="utf-8")
            clean = {"path": str(source), "revision": "c" * 40, "dirty": False}
            with mock.patch.object(natoms_scaling, "git_state", return_value=clean):
                metadata = natoms_scaling.build_metadata(library)
        self.assertEqual(metadata["build_system"], "meson")
        self.assertEqual(metadata["build_options"]["buildtype"], "release")
        self.assertEqual(metadata["build_options"]["optimization"], "3")
        self.assertEqual(metadata["compilers"]["host"]["fortran"]["id"], "gcc")
        self.assertTrue(
            metadata["compilers"]["host"]["fortran"]["executable_files"][0]["sha256"]
        )
        self.assertEqual(
            metadata["dependencies"][0]["linked_files"][0]["sha256"],
            hashlib.sha256(b"openblas").hexdigest(),
        )
        self.assertEqual(
            set(metadata["introspection_files"]),
            set(documents),
        )

    def test_meson_compiler_provenance_does_not_rebind_bare_path_entry(
        self,
    ) -> None:
        """Do not resolve a configure-time compiler name through a later PATH."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_compiler = root / "configured" / "cc"
            benchmark_path_compiler = root / "benchmark-path" / "cc"
            configured_compiler.parent.mkdir()
            benchmark_path_compiler.parent.mkdir()
            configured_compiler.write_bytes(b"configured compiler")
            benchmark_path_compiler.write_bytes(b"different compiler")
            benchmark_path_compiler.chmod(0o755)
            with mock.patch.dict(
                os.environ, {"PATH": str(benchmark_path_compiler.parent)}
            ):
                files, unresolved = (
                    natoms_scaling._meson_compiler_executable_provenance(
                        ["cc", str(configured_compiler)]
                    )
                )
        self.assertEqual(
            files,
            [
                {
                    "path": str(configured_compiler.resolve()),
                    "sha256": hashlib.sha256(b"configured compiler").hexdigest(),
                    "is_file": True,
                }
            ],
        )
        self.assertEqual(
            unresolved,
            [{"entry": "cc", "reason": "non_absolute_configure_time_entry"}],
        )

    def test_reference_protocol_records_cold_and_persistent_samples(self) -> None:
        """Separate the first reference invocation from persistent samples."""
        runner = FakeRunner({"fresh": 13, "warm": 2, "persistent": 0})
        clock_values = iter((0, 5_000_000, 10_000_000, 12_000_000))
        result = natoms_scaling.measure_reference_runner(
            runner,
            natoms_scaling.Protocol("persistent", 1, 1, 1.0e-8, 1.0e-7),
            1,
            3,
            clock_ns=lambda: next(clock_values),
        )
        invocations = [event for event in runner.events if event[0] == "invoke"]
        self.assertEqual(
            invocations,
            [
                ("invoke", "persistent"),
                ("invoke", "persistent"),
                ("invoke", "persistent"),
            ],
        )
        self.assertEqual(result["cold_sample"]["latency_ms"], 5.0)
        self.assertEqual(result["timing"]["samples_ms"], [2.0])

    def test_artifacts_retain_raw_samples_and_refuse_overwrite(self) -> None:
        """Retain raw JSON samples and reject accidental artifact overwrite."""
        document = {
            "schema_version": 1,
            "rows": [
                {
                    "availability": "available",
                    "start_mode": "warm",
                    "raw_samples": [
                        {
                            "latency_ms": 2.5,
                            "energies_hartree": [-1.0],
                            "scc_iterations": [2],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "warm.json"
            csv_path = root / "warm.csv"
            natoms_scaling.write_artifacts(
                json_path, csv_path, document, allow_overwrite=False
            )
            written = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(written["rows"][0]["raw_samples"][0]["latency_ms"], 2.5)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertNotIn("raw_samples", row)
            with self.assertRaises(FileExistsError):
                natoms_scaling.write_artifacts(
                    json_path, csv_path, document, allow_overwrite=False
                )

    def test_csv_serialization_uses_lf_and_remains_parseable(self) -> None:
        """Serialize portable LF-only CSV that the standard reader parses."""
        document = {
            "schema_version": 1,
            "rows": [{"label": "sample", "values": [1, 2]}],
        }
        serialized = natoms_scaling._serialize_csv(document)
        self.assertNotIn("\r", serialized)
        self.assertEqual(
            next(csv.DictReader(io.StringIO(serialized))),
            {"label": "sample", "values": "[1, 2]"},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            natoms_scaling.write_artifacts(
                json_path, csv_path, document, allow_overwrite=False
            )
            csv_bytes = csv_path.read_bytes()
            self.assertNotIn(b"\r", csv_bytes)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(
                    next(csv.DictReader(handle)),
                    {"label": "sample", "values": "[1, 2]"},
                )

    def test_csv_omits_realistic_raw_vectors_for_default_reader(self) -> None:
        """Keep large raw vectors in JSON and compact CSV to summary fields."""
        force_vector = [index / 123456789.0 for index in range(3 * 122)]
        raw_samples = [
            {
                "sample_index": index,
                "latency_ms": 70.0 + index,
                "energies_hartree": [-120.0],
                "forces_hartree_per_bohr": force_vector,
                "scc_iterations": [2],
            }
            for index in range(30)
        ]
        self.assertGreater(len(json.dumps(raw_samples)), csv.field_size_limit())
        document = {
            "schema_version": 1,
            "rows": [
                {
                    "label": "c40h82",
                    "run_identity": {"compiler": "complete provenance in JSON"},
                    "seed": {"forces_hartree_per_bohr": force_vector},
                    "cold_sample": {"forces_hartree_per_bohr": force_vector},
                    "raw_samples": raw_samples,
                    "timing": {
                        "median_ms": 72.0,
                        "samples_ms": [sample["latency_ms"] for sample in raw_samples],
                    },
                    "correctness": {
                        "status": "pass",
                        "energy_reference_hartree": [-120.0],
                        "force_reference_hartree_per_bohr": force_vector,
                        "max_abs_force_drift_hartree_per_bohr": 1.0e-11,
                    },
                }
            ],
        }
        serialized = natoms_scaling._serialize_csv(document)
        row = next(csv.DictReader(io.StringIO(serialized)))
        self.assertEqual(row["label"], "c40h82")
        self.assertNotIn("run_identity", row)
        self.assertNotIn("seed", row)
        self.assertNotIn("cold_sample", row)
        self.assertNotIn("raw_samples", row)
        self.assertEqual(json.loads(row["timing"]), {"median_ms": 72.0})
        self.assertEqual(
            json.loads(row["correctness"]),
            {
                "max_abs_force_drift_hartree_per_bohr": 1.0e-11,
                "status": "pass",
            },
        )

    def test_artifact_pair_rolls_back_when_second_publish_fails(self) -> None:
        """Remove the first new artifact when publishing its peer fails."""
        document = {"schema_version": 1, "rows": [{"value": 1}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            real_link = os.link
            calls = 0

            def fail_second_link(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic CSV publication failure")
                real_link(source, destination)

            with (
                mock.patch.object(
                    natoms_scaling.os, "link", side_effect=fail_second_link
                ),
                self.assertRaisesRegex(OSError, "synthetic CSV"),
            ):
                natoms_scaling.write_artifacts(
                    json_path, csv_path, document, allow_overwrite=False
                )
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob("*.lock")), [])

    def test_artifact_pair_restores_both_originals_after_replace_failure(self) -> None:
        """Restore both original artifacts after partial replacement failure."""
        document = {"schema_version": 1, "rows": [{"value": "new"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            json_path.write_text("old-json", encoding="utf-8")
            csv_path.write_text("old-csv", encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_second_publication(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise OSError("synthetic replacement failure")
                real_replace(source, destination)

            with (
                mock.patch.object(
                    natoms_scaling.os, "replace", side_effect=fail_second_publication
                ),
                self.assertRaisesRegex(OSError, "synthetic replacement"),
            ):
                natoms_scaling.write_artifacts(
                    json_path, csv_path, document, allow_overwrite=True
                )
            self.assertEqual(json_path.read_text(encoding="utf-8"), "old-json")
            self.assertEqual(csv_path.read_text(encoding="utf-8"), "old-csv")
            self.assertEqual(list(root.glob("*.backup")), [])
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_artifact_pair_rejects_stale_partial_and_concurrent_lock(self) -> None:
        """Reject incomplete old pairs and active pair reservations."""
        document = {"schema_version": 1, "rows": [{"value": 1}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            json_path.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "incomplete/stale"):
                natoms_scaling.write_artifacts(
                    json_path, csv_path, document, allow_overwrite=True
                )
            json_path.unlink()
            lock_path = natoms_scaling._pair_lock_path(json_path, csv_path)
            lock_path.write_text("concurrent", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "concurrent or stale"):
                natoms_scaling.write_artifacts(
                    json_path, csv_path, document, allow_overwrite=False
                )
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())

    def test_collect_rows_records_independent_cell_errors(self) -> None:
        """Preserve successful peers when one benchmark cell setup fails."""
        molecule_a = natoms_scaling.make_alkane(32)
        molecule_b = natoms_scaling.make_alkane(62)
        created: list[FakeRunner] = []

        def factory(
            _library: Path,
            cell: natoms_scaling.Cell,
            _cpu_threads: int,
            _device_id: int,
        ) -> FakeRunner:
            if cell.molecule.natoms == 62:
                raise natoms_scaling.BenchmarkError("synthetic setup failure")
            force_vector = [0.0] * (3 * cell.molecule.natoms * cell.batch_size)
            runner = FakeRunner(
                forces={
                    "fresh": force_vector,
                    "warm": force_vector,
                    "persistent": force_vector,
                }
            )
            created.append(runner)
            return runner

        rows, failed = natoms_scaling.collect_rows(
            [
                natoms_scaling.Cell("xtbloom", molecule_a, 1, "cpu", "force"),
                natoms_scaling.Cell("xtbloom", molecule_b, 1, "cpu", "force"),
            ],
            natoms_scaling.Protocol("fresh", 0, 1, 1.0e-8, 1.0e-7),
            Path("libxtbloom.so"),
            1,
            0,
            {"argv": ["test"]},
            runner_factory=factory,
        )
        self.assertTrue(failed)
        self.assertEqual(rows[0]["availability"], "available")
        self.assertEqual(rows[1]["availability"], "error")
        self.assertIn("synthetic setup failure", rows[1]["error"])
        self.assertTrue(created[0].closed)

    def test_reference_artifact_hashes_the_exact_bytes_that_are_parsed(self) -> None:
        """Hash the same in-memory bytes used to parse reference evidence."""
        document = make_reference_document()
        original = json.dumps(document, sort_keys=True).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_bytes(original)
            artifact = natoms_scaling.load_reference_artifact(path)
            path.write_text('{"corrupted": true}', encoding="utf-8")
            self.assertEqual(artifact.sha256, hashlib.sha256(original).hexdigest())
            self.assertEqual(len(artifact.rows), 1)

    def test_reference_artifact_rejects_schema_identity_and_duplicate_rows(
        self,
    ) -> None:
        """Reject malformed, inconsistent, or duplicate reference rows."""
        mutations = {
            "schema": lambda document: document.update(schema_version=999),
            "engine": lambda document: document.update(engine="tblite"),
            "protocol": lambda document: document["protocol"].update(start_mode="warm"),
            "wide_protocol_gate": lambda document: document["protocol"].update(
                force_atol_hartree_per_bohr=1.0e-4
            ),
            "row_status": lambda document: document["rows"][0]["correctness"].update(
                status="fail"
            ),
            "correctness_type": lambda document: document["rows"][0].update(
                correctness=[]
            ),
            "workload": lambda document: document["rows"][0][
                "workload_identity"
            ].update(batch_size=2),
            "options": lambda document: document["rows"][0]["compute_options"].pop(
                "charge_tolerance"
            ),
            "unpinned_options": lambda document: document["rows"][0][
                "compute_options"
            ].update(charge_tolerance=1.0e-6),
            "missing_raw_samples": lambda document: document["rows"][0].pop(
                "raw_samples"
            ),
            "missing_measured_sample": lambda document: document["rows"][0][
                "raw_samples"
            ].pop(),
            "nonconsecutive_sample_index": lambda document: document["rows"][0][
                "raw_samples"
            ][1].update(sample_index=4),
            "short_sample_energy": lambda document: document["rows"][0]["raw_samples"][
                0
            ].update(energies_hartree=[]),
            "string_sample_energy": lambda document: document["rows"][0]["raw_samples"][
                0
            ].update(energies_hartree=["-10.0"]),
            "nonfinite_sample_force": lambda document: document["rows"][0][
                "raw_samples"
            ][0]["forces_hartree_per_bohr"].__setitem__(0, math.nan),
            "tampered_later_force_sample": lambda document: document["rows"][0][
                "raw_samples"
            ][4]["forces_hartree_per_bohr"].__setitem__(0, 1.0e-9),
            "row_repetition_mismatch": lambda document: document["rows"][0].update(
                repetitions=4
            ),
            "tampered_energy_reference": lambda document: document["rows"][0][
                "correctness"
            ].update(energy_reference_hartree=[-10.0 + 1.0e-9]),
            "tampered_energy_drift": lambda document: document["rows"][0][
                "correctness"
            ].update(max_abs_energy_drift_hartree=1.0e-9),
            "tampered_force_reference": lambda document: document["rows"][0][
                "correctness"
            ]["force_reference_hartree_per_bohr"].__setitem__(0, 1.0),
            "tampered_force_drift": lambda document: document["rows"][0][
                "correctness"
            ].update(max_abs_force_drift_hartree_per_bohr=1.0e-9),
            "dirty_reference": lambda document: document["run_identity"][
                "repository"
            ].update(dirty=True),
            "binary_source_revision": lambda document: document["run_identity"][
                "library"
            ]["build"]["source"]["git"].update(revision="d" * 40),
            "library_sha": lambda document: document["run_identity"]["library"].update(
                sha256="not-a-sha"
            ),
            "duplicate": lambda document: document["rows"].append(
                dict(document["rows"][0])
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    document = make_reference_document()
                    mutate(document)
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(natoms_scaling.BenchmarkError):
                        natoms_scaling.load_reference_artifact(path)

    def test_cross_engine_force_and_energy_use_validated_reference(self) -> None:
        """Compare measured energy and force samples with validated FRESH data."""
        document = make_reference_document()
        reference_row = document["rows"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            row = {
                **reference_row,
                "engine": "tblite",
                "compute_options": {"engine": "tblite"},
                "correctness": {
                    "status": "pass",
                    "energy_reference_hartree": [-10.0 + 4.0e-7],
                    "force_reference_hartree_per_bohr": [
                        value + 4.0e-6
                        for value in reference_row["correctness"][
                            "force_reference_hartree_per_bohr"
                        ]
                    ],
                },
            }
            set_measured_observables(
                row,
                [[-10.0 + 1.0e-7], [-10.0 + 4.0e-7]],
                [
                    [
                        value + offset
                        for value in reference_row["correctness"][
                            "force_reference_hartree_per_bohr"
                        ]
                    ]
                    for offset in (1.0e-6, 4.0e-6)
                ],
            )
            failed = natoms_scaling.apply_cross_engine_correctness(
                [row],
                artifact,
                natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
            )
        self.assertFalse(failed)
        comparison = row["correctness"]["fresh_reference_comparison"]
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(
            comparison["xtbloom_option_identity"], "not_comparable_cross_engine"
        )
        self.assertAlmostEqual(comparison["energy"]["max_abs_delta_hartree"], 4e-7)
        self.assertAlmostEqual(
            comparison["force"]["max_abs_delta_hartree_per_bohr"], 4e-6
        )
        self.assertEqual(comparison["measured_samples"]["count"], 2)

    def test_cross_engine_checks_later_samples_not_only_cold_summary(self) -> None:
        """Gate every measured sample rather than only a cold summary vector."""
        document = make_reference_document()
        reference_row = document["rows"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            row = json.loads(json.dumps(reference_row))
            row["engine"] = "tblite"
            row["compute_options"] = {"engine": "tblite"}
            oracle_forces = reference_row["correctness"][
                "force_reference_hartree_per_bohr"
            ]
            set_measured_observables(
                row,
                [[-10.0], [-10.0]],
                [list(oracle_forces), [value + 6.0e-6 for value in oracle_forces]],
            )
            failed = natoms_scaling.apply_cross_engine_correctness(
                [row],
                artifact,
                natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
            )
        comparison = row["correctness"]["fresh_reference_comparison"]
        self.assertTrue(failed)
        self.assertEqual(comparison["status"], "fail")
        self.assertAlmostEqual(
            comparison["force"]["max_abs_delta_hartree_per_bohr"], 6.0e-6
        )

    def test_cross_engine_malformed_measured_vectors_fail_cleanly(self) -> None:
        """Report malformed measured vectors as clean correctness failures."""
        document = make_reference_document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            for mutation in ("missing_samples", "missing_force", "nonfinite_energy"):
                with self.subTest(mutation=mutation):
                    row = json.loads(json.dumps(document["rows"][0]))
                    row["engine"] = "tblite"
                    row["compute_options"] = {"engine": "tblite"}
                    if mutation == "missing_samples":
                        row.pop("raw_samples")
                    elif mutation == "missing_force":
                        row["raw_samples"][1].pop("forces_hartree_per_bohr")
                    else:
                        row["raw_samples"][1]["energies_hartree"][0] = math.inf
                    failed = natoms_scaling.apply_cross_engine_correctness(
                        [row],
                        artifact,
                        natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                        natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
                    )
                    comparison = row["correctness"]["fresh_reference_comparison"]
                    self.assertTrue(failed)
                    self.assertEqual(comparison["status"], "fail")
                    self.assertEqual(comparison["measured_samples"]["status"], "fail")

    def test_xtbloom_reference_requires_exact_compute_option_identity(self) -> None:
        """Require exact xtbloom compute options for same-engine comparison."""
        document = make_reference_document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            row = dict(document["rows"][0])
            row["compute_options"] = dict(row["compute_options"])
            row["compute_options"]["charge_tolerance"] *= 2.0
            row["correctness"] = dict(row["correctness"])
            failed = natoms_scaling.apply_cross_engine_correctness(
                [row],
                artifact,
                natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
            )
        self.assertTrue(failed)
        comparison = row["correctness"]["fresh_reference_comparison"]
        self.assertEqual(comparison["status"], "fail")
        self.assertEqual(comparison["xtbloom_option_identity"], "fail")

    def test_xtbloom_warm_reference_requires_same_binary_and_revision(self) -> None:
        """Require WARM rows to match the FRESH binary and source revision."""
        document = make_reference_document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            row = json.loads(json.dumps(document["rows"][0]))
            row["start_mode"] = "warm"
            row["run_identity"]["library"]["sha256"] = "c" * 64
            row["run_identity"]["repository"]["revision"] = "d" * 40
            row["run_identity"]["library"]["build"]["source"]["git"]["revision"] = (
                "d" * 40
            )
            failed = natoms_scaling.apply_cross_engine_correctness(
                [row],
                artifact,
                natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
            )
        comparison = row["correctness"]["fresh_reference_comparison"]
        self.assertTrue(failed)
        self.assertEqual(comparison["xtbloom_binary_identity"], "fail")
        self.assertEqual(comparison["xtbloom_repository_revision"], "fail")

    def test_cross_engine_force_above_manifest_gate_is_not_performance_evidence(
        self,
    ) -> None:
        """Reject performance evidence whose forces exceed the manifest gate."""
        document = make_reference_document()
        reference_row = document["rows"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            artifact = natoms_scaling.load_reference_artifact(path)
            row = {
                **reference_row,
                "engine": "xtb",
                "compute_options": {"engine": "xtb"},
                "correctness": {
                    "status": "pass",
                    "energy_reference_hartree": [-10.0],
                    "force_reference_hartree_per_bohr": [
                        value + 7.9e-4
                        for value in reference_row["correctness"][
                            "force_reference_hartree_per_bohr"
                        ]
                    ],
                },
            }
            set_measured_observables(
                row,
                [[-10.0]],
                [
                    [
                        value + 7.9e-4
                        for value in reference_row["correctness"][
                            "force_reference_hartree_per_bohr"
                        ]
                    ]
                ],
            )
            failed = natoms_scaling.apply_cross_engine_correctness(
                [row],
                artifact,
                natoms_scaling.DEFAULT_CROSS_ENGINE_ENERGY_ATOL,
                natoms_scaling.DEFAULT_CROSS_ENGINE_FORCE_ATOL,
            )
        comparison = row["correctness"]["fresh_reference_comparison"]
        self.assertTrue(failed)
        self.assertEqual(comparison["status"], "fail")
        self.assertEqual(comparison["force"]["status"], "fail")

    def test_tblite_charge_getter_can_be_excluded_from_timed_protocol(self) -> None:
        """Allow tblite atomic-charge publication outside the timed protocol."""
        adapter = object.__new__(TbliteAdapter)
        adapter.property_name = "force"
        adapter.collect_atomic_charges = False
        adapter.library = SimpleNamespace(
            tblite_update_structure_geometry=mock.Mock(),
            tblite_get_singlepoint=mock.Mock(),
            tblite_get_result_energy=mock.Mock(),
            tblite_get_result_gradient=mock.Mock(),
            tblite_get_result_charges=mock.Mock(),
        )
        adapter._check_error = mock.Mock()
        adapter._check_context = mock.Mock()
        adapter.states = [
            TbliteState(
                error=None,
                context=None,
                structure=None,
                calculator=None,
                result=None,
                positions=None,
                energy=natoms_scaling.ctypes.c_double(-3.0),
                gradient=[1.0, -2.0, 3.0],
                charges=None,
                keepalive=(),
            )
        ]
        adapter.invoke()
        adapter.library.tblite_get_result_energy.assert_called_once()
        adapter.library.tblite_get_result_gradient.assert_called_once()
        adapter.library.tblite_get_result_charges.assert_not_called()
        self.assertNotIn("atomic_charges_e", adapter.results())


if __name__ == "__main__":
    unittest.main()
