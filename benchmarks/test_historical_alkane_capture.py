"""Offline tests for the bounded historical native capture protocol."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmarks import historical_alkane_capture as capture


def make_valid_trace(
    statuses: list[int], converged: list[int], iterations: list[int]
) -> dict[str, object]:
    """Build a minimal valid diagnostic record correlated to public peer data."""
    terminal = [
        {
            "system_index": index,
            "iterations": iterations[index],
            "status": statuses[index],
            "converged": converged[index],
        }
        for index in range(len(statuses))
    ]
    converged_count = sum(value == 1 for value in converged)
    exhausted_count = sum(status == 7 for status in statuses)
    failed_count = sum(status != 0 and status != 7 for status in statuses)
    return {
        "schema": "xtbloom.cuda.scc_diagnostics.v1",
        "instrumented": True,
        "execution_mode": "device_dispatch_chain",
        "fallback_reason": 0,
        "batch_size": len(statuses),
        "maximum_iterations": 500,
        "start_policy": capture.SCC_START_FRESH,
        "buckets": [
            {
                "bucket_index": 0,
                "ao": 1,
                "system_capacity": len(statuses),
                "channel_capacity": 2 * len(statuses),
            }
        ],
        "terminal_buckets": [
            {
                "bucket_index": 0,
                "converged_systems": converged_count,
                "failed_systems": failed_count,
                "exhausted_systems": exhausted_count,
                "unfinished_systems": len(statuses)
                - converged_count
                - failed_count
                - exhausted_count,
            }
        ],
        "terminal_systems": terminal,
        "iterations": [],
    }


def make_synthetic_source_document() -> dict[str, object]:
    """Build a hashable fake corpus with all original alkane slots present."""
    workloads: list[dict[str, object]] = []
    for name, natoms in capture.CAPTURE_WORKLOADS:
        numbers_per_slot = [6, *([1] * (natoms - 1))]
        one_geometry = [0.5606069543851774, -1.982044895130482, 1.25] * natoms
        charges = [0] * 256
        unpaired = [0] * 256
        spin_channels = [1] * 256
        charges[7] = -1
        unpaired[7] = 1
        spin_channels[7] = 2
        workloads.append(
            {
                "name": name,
                "natoms": natoms,
                "batch_size": 256,
                "seed": natoms * 1000,
                "perturb_sigma_bohr": 0.02,
                "atom_offsets": [slot * natoms for slot in range(257)],
                "atomic_numbers": [*numbers_per_slot] * 256,
                "positions_bohr": one_geometry * 256,
                "molecular_charges": charges,
                "unpaired_electrons": unpaired,
                "spin_channels": spin_channels,
            }
        )
    return {
        "resource_stub": "synthetic input fixture; not a historical corpus",
        "source_commit": "c9c0a432947f122d25cb91d0a4624af0a3e761ad",
        "source_hashes": {"fixture.py": "a" * 64},
        "workloads": workloads,
    }


class FakeMemory:
    """Host-only descriptor memory whose download boundary is observable."""

    def __init__(self) -> None:
        self.download_count = 0

    def download_outputs(self) -> None:
        """Record that raw caller-owned buffers were inspected after timing."""
        self.download_count += 1


class FakeAdapter:
    """Persistent fake owner with caller buffers and an optional JSONL sink."""

    created: list[FakeAdapter]
    fail_peer = False

    def __init__(
        self,
        library_path: Path,
        manifest_path: Path,
        manifest: dict[str, object],
        case_sequence: tuple[dict[str, object], ...],
        workload: str,
        property_set: str,
    ) -> None:
        self.library_path = library_path
        self.storage = capture.run.public_api.assemble_batch(
            manifest_path, manifest, case_sequence
        )
        self.systems = len(case_sequence)
        self.atoms = len(self.storage.atomic_numbers)
        self.property_set = property_set
        self.cell = SimpleNamespace(backend="cuda", memory_mode="host")
        self.options = SimpleNamespace(
            model=capture.run.public_api.XTBLOOM_MODEL_GFN2_XTB,
            max_scc_iterations=500,
            energy_tolerance=1.0e-8,
            charge_tolerance=1.0e-6,
            electronic_temperature=300.0
            * capture.run.public_api.XTBLOOM_KELVIN_TO_HARTREE,
            scc_start_mode=capture.SCC_START_FRESH,
            scc_mixer=capture.SCC_MIXER_MODIFIED_BROYDEN,
            scc_mixer_history=8,
            scc_mixer_damping=0.4,
            determinism=capture.DETERMINISM_DEFAULT,
        )
        self.energies = (ctypes.c_double * self.systems)()
        self.forces = (ctypes.c_double * (3 * self.atoms))()
        self.charges = (
            (ctypes.c_double * self.atoms)() if property_set == "EFq" else None
        )
        self.iterations = (ctypes.c_int32 * self.systems)()
        self.converged = (ctypes.c_uint8 * self.systems)()
        self.statuses = (ctypes.c_int32 * self.systems)()
        self.result = SimpleNamespace(flags=0)
        self.library = SimpleNamespace(
            xtbloom_status_string=lambda status: f"status-{status}".encode(),
            xtbloom_get_last_error=lambda: b"fake native error",
        )
        self.memory = FakeMemory()
        self.last_call_status: int | None = None
        self.last_call_error: str | None = None
        self.calls = 0
        self.closed = False
        self.fail_call = False
        type(self).created.append(self)

    def reset_outputs(self) -> None:
        """Use the same poison values as the native adapter before each call."""
        self.result.flags = capture.RESULT_FLAGS_SENTINEL
        for index in range(self.systems):
            self.energies[index] = capture.ENERGY_SENTINEL
            self.iterations[index] = capture.ITERATIONS_SENTINEL
            self.statuses[index] = capture.STATUS_SENTINEL
            self.converged[index] = capture.CONVERGED_SENTINEL
        for index in range(len(self.forces)):
            self.forces[index] = capture.FORCE_SENTINEL
        if self.charges is not None:
            for index in range(len(self.charges)):
                self.charges[index] = capture.CHARGE_SENTINEL

    def invoke(self) -> None:
        """Publish fake finite peers or a deliberate partial call-level failure."""
        self.calls += 1
        if self.fail_call and self.calls == 1:
            self.last_call_status = capture.STATUS_INTERNAL_ERROR
            self.last_call_error = "injected failure after partial publication"
            self.result.flags = 9
            self.energies[0] = 123.5
            return

        self.last_call_status = 0
        self.last_call_error = None
        self.result.flags = 0
        for index in range(self.systems):
            self.energies[index] = -1.0 - index
            self.iterations[index] = 1
            self.statuses[index] = 0
            self.converged[index] = 1
        for index in range(len(self.forces)):
            self.forces[index] = 0.25
        if self.charges is not None:
            for index in range(len(self.charges)):
                self.charges[index] = -0.125
        if type(self).fail_peer:
            self.statuses[0] = 7
            self.converged[0] = 0
            self.iterations[0] = 500
            self.energies[0] = math.nan
            for index in range(3 * self.storage.atom_offsets[1]):
                self.forces[index] = math.nan
            if self.charges is not None:
                for index in range(self.storage.atom_offsets[1]):
                    self.charges[index] = math.nan

        sink = os.environ.get(capture.DIAGNOSTIC_SINK_ENV)
        if sink is not None:
            with Path(sink).open("a", encoding="utf-8") as output:
                json.dump(
                    make_valid_trace(
                        [int(value) for value in self.statuses],
                        [int(value) for value in self.converged],
                        [int(value) for value in self.iterations],
                    ),
                    output,
                    separators=(",", ":"),
                )
                output.write("\n")

    def synchronize(self) -> None:
        """Represent the existing synchronized completion boundary."""

    def close(self) -> None:
        """Record owner cleanup for lifetime assertions."""
        self.closed = True


class HistoricalAlkaneCaptureTest(unittest.TestCase):
    """Test protocol behavior with fake native calls, clocks, and trace sinks."""

    @classmethod
    def setUpClass(cls) -> None:
        """Materialize one synthetic, hash-valid 512-slot corpus offline."""
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source_path = cls.root / "source.json"
        source_document = make_synthetic_source_document()
        source_bytes = json.dumps(
            source_document, allow_nan=True, separators=(",", ":")
        ).encode()
        cls.source_path.write_bytes(source_bytes)
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        cls.manifest_dir = cls.root / "materialized"
        capture.historical_alkane_inputs.materialize(
            cls.source_path, cls.manifest_dir, source_hash
        )
        cls.manifest_path = cls.manifest_dir / "manifest.json"
        cls.manifest_sha256 = hashlib.sha256(cls.manifest_path.read_bytes()).hexdigest()
        cls.manifest = capture.run.conformance.load_json(cls.manifest_path)
        cls.pin_path = cls.root / "library-pins.json"
        cls.pin_document = {
            "schema": capture.PIN_SCHEMA,
            "builds": [
                {
                    "role": role,
                    "dso_path": str(cls.root / f"{role}.so"),
                    "dso_sha256": "a" * 64,
                    "cache_path": str(cls.root / f"{role}.cache"),
                    "cache_sha256": "b" * 64,
                    "source_dir": str(cls.root / f"{role}-source"),
                    "source_commit": "c9c0a432947f122d25cb91d0a4624af0a3e761ad",
                    "source_sha256": "d" * 64,
                }
                for role in ("baseline", "off", "on")
            ],
        }
        cls.pin_path.write_text(json.dumps(cls.pin_document), encoding="utf-8")
        cls.selected: dict[str, tuple[dict[str, object], ...]] = {
            workload: capture.historical_alkane_inputs.select_workload_cases(
                cls.manifest, workload
            )
            for workload, _ in capture.CAPTURE_WORKLOADS
        }
        cls.pins = capture._validate_library_pins(cls.pin_path)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove only the test-owned synthetic input tree."""
        cls.temporary.cleanup()

    def setUp(self) -> None:
        """Reset fake owners and isolate the diagnostic sink environment."""
        FakeAdapter.created = []
        FakeAdapter.fail_peer = False
        os.environ.pop(capture.DIAGNOSTIC_SINK_ENV, None)

    def fake_timed_invoke(self, adapter: FakeAdapter) -> float:
        """Exercise invocation and synchronization while returning a fixed time."""
        adapter.invoke()
        adapter.synchronize()
        return 2.5

    def fake_clock(self) -> object:
        """Return a monotonically advancing deterministic setup clock."""
        ticks = (index * 500_000_000 for index in range(10_000))
        return lambda: next(ticks)

    def fake_pin_verification(self, *_: object) -> dict[str, dict[str, object]]:
        """Return identical immutable pre/post identities without Git or builds."""
        return {
            role: {
                "source_commit": pin["source_commit"],
                "source_clean": True,
                "source_sha256": pin["source_sha256"],
                "dso_sha256": pin["dso_sha256"],
                "cache_sha256": pin["cache_sha256"],
            }
            for role, pin in self.pins.items()
        }

    def run_fake_capture(
        self,
        output_directory: Path,
        *,
        adapter_factory: object | None = None,
        timed_invoke: object | None = None,
        warmups: int = 0,
        measured: int = 0,
        properties: tuple[tuple[str, int], ...] = (("EF", 3),),
        workloads: tuple[tuple[str, int], ...] = (capture.CAPTURE_WORKLOADS[0],),
    ) -> int:
        """Run the real scheduler around fake owners, a fake timer, and fake pins."""
        selected = {workload: self.selected[workload] for workload, _ in workloads}
        factory = FakeAdapter if adapter_factory is None else adapter_factory
        with (
            mock.patch.object(
                capture, "_verify_all_builds", side_effect=self.fake_pin_verification
            ),
            mock.patch.object(
                capture.run,
                "timed_invoke",
                self.fake_timed_invoke if timed_invoke is None else timed_invoke,
            ),
        ):
            return capture.run_capture(
                self.manifest_path,
                self.manifest,
                selected,
                self.manifest_sha256,
                self.pins,
                output_directory,
                adapter_factory=factory,
                clock_ns=self.fake_clock(),
                warmups=warmups,
                measured=measured,
                workloads=workloads,
                properties=properties,
            )

    def test_native_options_match_pins_and_label_extra_fields(self) -> None:
        """Check adapter flags and fixed capture options without loading a DSO."""
        received: list[tuple[object, ...]] = []

        def initialize_base(
            adapter: capture.NativeAdapter,
            library_path: Path,
            manifest_path: Path,
            manifest: dict[str, object],
            case_sequence: tuple[dict[str, object], ...],
            cell: object,
            device_id: int,
            cpu_threads: int,
        ) -> None:
            received.append((cell, device_id, cpu_threads, len(case_sequence)))
            adapter.library = SimpleNamespace()
            adapter.context = object()
            adapter.memory = SimpleNamespace(output=lambda owner, role: (owner, role))
            adapter.systems = 256
            adapter.atoms = 512
            adapter.energies = (ctypes.c_double * 256)()
            adapter.forces = (ctypes.c_double * (3 * 512))()
            adapter.storage = SimpleNamespace(point_charge_values=[])
            adapter.options = SimpleNamespace()
            adapter.result = SimpleNamespace(atomic_charges=None)

        cases = tuple({"id": str(index)} for index in range(256))
        with mock.patch.object(capture.run.XTBloomAdapter, "__init__", initialize_base):
            ef = capture.NativeAdapter(
                Path("/absolute/baseline.so"),
                Path("/absolute/manifest.json"),
                {},
                cases,
                "alkane32-b256",
                "EF",
            )
            efq = capture.NativeAdapter(
                Path("/absolute/on.so"),
                Path("/absolute/manifest.json"),
                {},
                cases,
                "alkane32-b256",
                "EFq",
            )

        self.assertEqual([entry[1:] for entry in received], [(0, 1, 256)] * 2)
        for cell, _, _, _ in received:
            self.assertEqual(cell.backend, "cuda")
            self.assertEqual(cell.memory_mode, "host")
            self.assertEqual(cell.property, "force")
        self.assertEqual(ef.options.flags, 3)
        self.assertEqual(efq.options.flags, 7)
        for adapter in (ef, efq):
            self.assertEqual(
                adapter.options.model, capture.run.public_api.XTBLOOM_MODEL_GFN2_XTB
            )
            self.assertEqual(adapter.options.max_scc_iterations, 500)
            self.assertEqual(adapter.options.energy_tolerance, 1.0e-8)
            self.assertEqual(adapter.options.charge_tolerance, 1.0e-6)
            self.assertEqual(
                adapter.options.electronic_temperature,
                300.0 * capture.run.public_api.XTBLOOM_KELVIN_TO_HARTREE,
            )
            self.assertEqual(
                adapter.options.electronic_temperature,
                capture.HISTORICAL_REQUEST["electronic_temperature_hartree"],
            )
            self.assertEqual(adapter.options.scc_start_mode, 1)
            self.assertEqual(adapter.options.scc_mixer, 1)
            self.assertEqual(adapter.options.scc_mixer_history, 8)
            self.assertEqual(adapter.options.scc_mixer_damping, 0.4)
            self.assertEqual(adapter.options.determinism, 0)
        self.assertIsNone(ef.charges)
        self.assertEqual(efq.result.atomic_charges[1], "atomic_charges")

    def test_clean_detached_baseline_snapshot_matches_all_pins(self) -> None:
        """Accept a pinned clean main commit without requiring a local branch."""
        source_dir = self.root / "detached-baseline-source"
        source_dir.mkdir()
        dso_path = self.root / "detached-baseline.so"
        cache_path = self.root / "detached-baseline.cache"
        dso_path.write_bytes(b"fake DSO")
        cache_path.write_bytes(b"fake CMake cache")
        commit = "c9c0a432947f122d25cb91d0a4624af0a3e761ad"
        pin = {
            "role": "baseline",
            "dso_path": str(dso_path),
            "dso_sha256": "a" * 64,
            "cache_path": str(cache_path),
            "cache_sha256": "b" * 64,
            "source_dir": str(source_dir),
            "source_commit": commit,
            "source_sha256": "d" * 64,
        }
        git_commands: list[tuple[str, ...]] = []

        def detached_git_output(path: Path, *arguments: str) -> str:
            self.assertEqual(path, source_dir)
            git_commands.append(arguments)
            if arguments == ("rev-parse", "HEAD"):
                return commit
            if arguments == (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ):
                return ""
            self.fail(f"unexpected Git query for detached source: {arguments}")
            raise AssertionError("unreachable")

        def pinned_file_hash(path: Path) -> str:
            return pin["dso_sha256"] if path == dso_path else pin["cache_sha256"]

        with (
            mock.patch.object(capture, "_git_output", detached_git_output),
            mock.patch.object(capture, "_git_archive_sha256", return_value="d" * 64),
            mock.patch.object(capture, "_sha256_file", side_effect=pinned_file_hash),
        ):
            actual = capture._verify_one_build("baseline", pin)

        self.assertEqual(
            git_commands,
            [
                ("rev-parse", "HEAD"),
                ("status", "--porcelain=v1", "--untracked-files=all"),
            ],
        )
        self.assertEqual(actual["source_commit"], commit)
        self.assertTrue(actual["source_clean"])
        self.assertEqual(actual["source_sha256"], pin["source_sha256"])
        self.assertEqual(actual["dso_sha256"], pin["dso_sha256"])
        self.assertEqual(actual["cache_sha256"], pin["cache_sha256"])

    def test_offline_describe_never_loads_a_native_library(self) -> None:
        """Describe the fixed matrix with all native loading patched to fail."""
        standard_output = io.StringIO()
        argv = [
            "--manifest",
            str(self.manifest_path),
            "--manifest-sha256",
            self.manifest_sha256,
            "--historical-bridge-raw-sha256",
            capture.HISTORICAL_BRIDGE_RAW_SHA256,
            "--library-pins",
            str(self.pin_path),
            "--describe",
        ]
        with (
            mock.patch.object(
                capture,
                "NativeAdapter",
                side_effect=AssertionError("native adapter must remain offline"),
            ),
            mock.patch.object(
                capture, "_verify_all_builds", side_effect=AssertionError("Git check")
            ),
            redirect_stdout(standard_output),
        ):
            self.assertEqual(capture.main(argv), 0)
        document = json.loads(standard_output.getvalue())
        self.assertFalse(document["native_library_loaded"])
        self.assertEqual(
            [item["case_count"] for item in document["workloads"]], [256, 256]
        )

    def test_default_plan_has_the_bounded_issue_512_matrix(self) -> None:
        """Pin the full counts without writing hundreds of megabytes of fake data."""
        self.assertEqual(len(capture._planned_rounds()), 36)
        self.assertEqual(capture._planned_rounds()[0], ("cold", 0))
        self.assertEqual(
            capture._planned_rounds()[1:6], [("warmup", index) for index in range(1, 6)]
        )
        self.assertEqual(
            capture._planned_rounds()[6:],
            [("measured", index) for index in range(1, 31)],
        )
        owner_count = (
            len(capture.CAPTURE_WORKLOADS)
            * len(capture.CAPTURE_PROPERTIES)
            * len(capture.CAPTURE_VARIANTS)
        )
        calls_per_owner = 1 + capture.EXCLUDED_WARMUPS + capture.MEASURED_CALLS
        self.assertEqual(owner_count, 16)
        self.assertEqual(owner_count * calls_per_owner, 576)
        self.assertEqual(256 * owner_count * calls_per_owner, 147456)
        self.assertEqual(
            len(capture.CAPTURE_WORKLOADS)
            * len(capture.CAPTURE_PROPERTIES)
            * calls_per_owner,
            144,
        )

    def test_scheduler_uses_fresh_owners_and_validates_one_sink_record(self) -> None:
        """Keep four owners persistent and parse only the on-sink owner's append."""
        output_directory = self.root / "successful-capture"
        self.assertEqual(self.run_fake_capture(output_directory), 0)
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["phase"] == "cold" for row in rows))
        self.assertTrue(
            all(row["elapsed_synchronized_public_invoke_ms"] == 2.5 for row in rows)
        )
        self.assertEqual(len(FakeAdapter.created), 4)
        self.assertTrue(
            all(
                adapter.calls == 1 and adapter.closed for adapter in FakeAdapter.created
            )
        )
        sink_row = next(row for row in rows if row["variant"] == "on-sink")
        self.assertEqual(sink_row["diagnostic_trace"]["parsed_records"], 1)
        self.assertTrue(
            all(row["diagnostic_trace"] is None for row in rows if row is not sink_row)
        )
        metadata = json.loads((output_directory / "metadata.json").read_text())
        self.assertEqual(metadata["capture_options"]["model"], 2)
        self.assertEqual(metadata["capture_options"]["model_name"], "GFN2-xTB")
        self.assertTrue(all(row["compute_options"]["model"] == 2 for row in rows))
        self.assertTrue(metadata["capture_valid"])
        self.assertFalse(metadata["claim_eligible"])
        self.assertEqual(
            metadata["historical_bridge_raw_source_sha256"],
            capture.HISTORICAL_BRIDGE_RAW_SHA256,
        )
        self.assertEqual(
            metadata["capture_options_not_emitted_by_historical_bridge_raw"][
                "historically_unemitted_fields"
            ],
            ["scc_mixer", "determinism", "CPU ISA"],
        )
        self.assertEqual(metadata["diagnostic_trace_records"], 1)
        self.assertNotIn(capture.DIAGNOSTIC_SINK_ENV, os.environ)

    def test_synchronization_error_keeps_published_buffers_and_trace(self) -> None:
        """Retain caller data and trace even when the sync boundary raises."""

        def fail_after_invoke(adapter: FakeAdapter) -> float:
            adapter.invoke()
            raise RuntimeError("injected synchronization failure")

        output_directory = self.root / "synchronization-failure-capture"
        self.assertEqual(
            self.run_fake_capture(output_directory, timed_invoke=fail_after_invoke),
            1,
        )
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["state"] == "ERROR" for row in rows))
        self.assertTrue(all(row["call_status"] == 0 for row in rows))
        self.assertTrue(all(len(row["energies_hartree"]) == 256 for row in rows))
        self.assertTrue(
            all(
                row["timing_error"] == "RuntimeError: injected synchronization failure"
                for row in rows
            )
        )
        sink_row = next(row for row in rows if row["variant"] == "on-sink")
        self.assertEqual(sink_row["diagnostic_trace"]["parsed_records"], 1)
        self.assertTrue(all(adapter.closed for adapter in FakeAdapter.created))
        self.assertNotIn(capture.DIAGNOSTIC_SINK_ENV, os.environ)

    def test_failed_invoke_does_not_reuse_previous_call_status(self) -> None:
        """Clear owner status before each call so an exception cannot reuse zero."""
        calls_by_owner: dict[int, int] = {}

        def fail_after_each_owners_first_call(adapter: FakeAdapter) -> float:
            owner_id = id(adapter)
            calls_by_owner[owner_id] = calls_by_owner.get(owner_id, 0) + 1
            if calls_by_owner[owner_id] == 1:
                return self.fake_timed_invoke(adapter)
            raise RuntimeError("injected failure before invoke")

        output_directory = self.root / "pre-invoke-failure-capture"
        self.assertEqual(
            self.run_fake_capture(
                output_directory,
                timed_invoke=fail_after_each_owners_first_call,
                warmups=1,
            ),
            1,
        )
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        failed_rows = [row for row in rows if row["phase"] == "warmup"]
        self.assertEqual(len(failed_rows), 4)
        self.assertTrue(all(row["call_status"] is None for row in failed_rows))
        self.assertTrue(
            all(
                row["call_error"] == "RuntimeError: injected failure before invoke"
                for row in failed_rows
            )
        )
        self.assertTrue(
            all(
                adapter.calls == 1 and adapter.closed for adapter in FakeAdapter.created
            )
        )

    def test_failed_peer_iteration_sentinel_invalidates_publication(self) -> None:
        """Failed floating slices do not excuse an untouched iteration buffer."""
        FakeAdapter.fail_peer = True

        def leave_failed_iteration_untouched(adapter: FakeAdapter) -> float:
            elapsed = self.fake_timed_invoke(adapter)
            adapter.iterations[0] = capture.ITERATIONS_SENTINEL
            return elapsed

        output_directory = self.root / "failed-iteration-sentinel-capture"
        self.assertEqual(
            self.run_fake_capture(
                output_directory,
                timed_invoke=leave_failed_iteration_untouched,
                warmups=1,
            ),
            1,
        )
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl").read_text().splitlines()
        ]
        cold_rows = [row for row in rows if row["phase"] == "cold"]
        self.assertEqual(len(cold_rows), 4)
        self.assertTrue(
            all(
                "system 0 retained a per-system iteration sentinel"
                in row["capture_issues"]
                for row in cold_rows
            )
        )
        self.assertTrue(
            all(row["state"] == "NOT_RUN" for row in rows if row["phase"] == "warmup")
        )
        self.assertTrue(
            all(
                adapter.calls == 1 and adapter.closed for adapter in FakeAdapter.created
            )
        )

    def test_peer_failures_keep_nan_slices_and_do_not_poison_fresh_owner(self) -> None:
        """Preserve failed-peer data while continuing safe call-level successes."""
        FakeAdapter.fail_peer = True
        output_directory = self.root / "peer-failure-capture"
        result = self.run_fake_capture(
            output_directory,
            warmups=1,
            properties=(("EFq", 7),),
        )
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(result, 1)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["state"] == "CAPTURED" for row in rows))
        self.assertEqual(len(FakeAdapter.created), 4)
        self.assertTrue(
            all(
                adapter.calls == 2 and adapter.closed for adapter in FakeAdapter.created
            )
        )
        for row in rows:
            self.assertEqual(row["peer_failures"][0]["status"], 7)
            self.assertEqual(row["energies_hartree"][0], {"nonfinite": "nan"})
            self.assertTrue(
                all(
                    value == {"nonfinite": "nan"}
                    for value in row["forces_hartree_per_bohr"][: 3 * 32]
                )
            )
            self.assertTrue(
                all(
                    value == {"nonfinite": "nan"}
                    for value in row["atomic_charges_e"][:32]
                )
            )
        metadata = json.loads((output_directory / "metadata.json").read_text())
        self.assertFalse(metadata["capture_valid"])

    def test_call_level_failure_poisons_owner_and_keeps_partial_buffers(self) -> None:
        """Do not retry an owner after INTERNAL_ERROR or discard modified outputs."""

        def factory(*args: object) -> FakeAdapter:
            adapter = FakeAdapter(*args)
            adapter.fail_call = adapter.library_path.name == "baseline.so"
            return adapter

        output_directory = self.root / "call-failure-capture"
        self.assertEqual(
            self.run_fake_capture(
                output_directory,
                adapter_factory=factory,
                warmups=1,
            ),
            1,
        )
        rows = [
            json.loads(line)
            for line in (output_directory / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        baseline_rows = [row for row in rows if row["variant"] == "baseline"]
        self.assertEqual([row["state"] for row in baseline_rows], ["ERROR", "NOT_RUN"])
        self.assertEqual(baseline_rows[0]["call_status"], capture.STATUS_INTERNAL_ERROR)
        self.assertEqual(baseline_rows[0]["energies_hartree"][0], 123.5)
        self.assertEqual(baseline_rows[0]["result_flags_after"], 9)
        self.assertIn(
            "modified caller outputs", " ".join(baseline_rows[0]["capture_issues"])
        )
        self.assertTrue(all(adapter.closed for adapter in FakeAdapter.created))

    def test_malformed_or_missing_trace_is_invalid_and_raw_sink_survives(self) -> None:
        """Reject malformed sink bytes and retain their exact file for diagnosis."""
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            raw_trace = b"{not-json}\n"
            trace_path.write_bytes(raw_trace)
            result, next_offset, issues = capture._trace_delta(
                trace_path, 0, True, 256, []
            )
            self.assertEqual(trace_path.read_bytes(), raw_trace)
        self.assertGreater(next_offset, 0)
        self.assertEqual(result["parsed_records"], 0)
        self.assertTrue(any("malformed" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
