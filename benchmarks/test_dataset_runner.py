"""Hardware-independent tests for the dataset manifest runner."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from benchmarks import dataset_runner
from benchmarks import run as benchmark_run
from benchmarks.dxtb_adapter import DxtbAdapter
from benchmarks.tblite_adapter import TbliteAdapter
from benchmarks.xtb_adapter import XtbAdapter


def manifest_row(
    system_id: str,
    order: int,
    *,
    unit: str = "bohr",
    sample_set: str = "main",
) -> dict[str, str]:
    """Create one minimal QM9 manifest fixture row."""
    return {
        "sample_set": sample_set,
        "sample_order": str(order),
        "sample_id": system_id,
        "source_index": str(order + 10),
        "source_split": "train",
        "source_file": "fixture.pkl",
        "charge": "0",
        "multiplicity": "1",
        "unpaired_electrons": "0",
        "natoms": "2",
        "heavy_atom_count": "1",
        "gfn2_n_ao": "5",
        "element_signature": "H",
        "gfn2_ao_bin": "Q1",
        "sampling_stratum": f"stratum-{order}",
        "sampling_probability": "0.5",
        "coordinate_source_unit": "angstrom",
        "coordinate_output_unit": unit,
        "angstrom_per_bohr": "0.529177210903",
        "xtbloom_input_sha256": f"hash-{system_id}",
    }


def sample(row: dict[str, str]) -> dict[str, object]:
    """Create the JSONL sample paired with a QM9 manifest row."""
    return {
        "sample_id": row["sample_id"],
        "source_index": int(row["source_index"]),
        "atomic_numbers": [1, 1],
        "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        "charge": 0,
        "multiplicity": 1,
        "unpaired_electrons": 0,
        "xtbloom_input_sha256": row["xtbloom_input_sha256"],
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write fixture rows with their insertion-order header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_qm9_bundle(root: Path, rows: list[dict[str, str]]) -> Path:
    """Create the real two-file QM9 bundle contract."""
    manifest = root / "manifests" / "selected_samples.csv"
    write_csv(manifest, rows)
    sample_sets = {row["sample_set"] for row in rows}
    if len(sample_sets) != 1:
        raise ValueError("fixture requires exactly one QM9 sample set")
    sample_path = root / "samples" / f"{next(iter(sample_sets))}.jsonl.gz"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["sample_order"]))
    with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(sample(row)) + "\n")
    return manifest


def omol_row(
    system_id: str, subset: str, priority: str, charge: int = 0
) -> dict[str, str]:
    """Create one minimal OMol25 metadata row."""
    return {
        "sample_set": subset,
        "configuration_id": system_id,
        "property_id": f"property-{system_id}",
        "source_shard": "co_0.parquet",
        "source_row": "1",
        "source_data_id": "fixture",
        "source_path": "fixture/path",
        "charge": str(charge),
        "multiplicity": "1",
        "unpaired_electrons": "0",
        "natoms": "2",
        "gfn2_n_ao": "5",
        "domain": "community",
        "configuration_class": "equilibrium",
        "charge_class": "neutral" if charge == 0 else "positive",
        "spin_class": "singlet",
        "element_class": "light_main_group",
        "natoms_bin": "Q1",
        "gfn2_ao_bin": "Q1",
        "sampling_stratum": f"{subset}-stratum",
        "sampling_probability": "0.25",
        "selection_priority_sha256": priority,
        "coordinate_source_unit": "angstrom",
        "coordinate_output_unit": "bohr",
        "angstrom_per_bohr": "0.529177210903",
        "xtbloom_input_sha256": f"hash-{system_id}",
    }


def write_npz(path: Path, rows: list[dict[str, str]]) -> None:
    """Write one ragged-v1 OMol25 subset container."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    np.savez_compressed(
        path,
        format_version=np.asarray(dataset_runner.OMOL25_FORMAT_VERSION),
        sample_ids=np.asarray([row["configuration_id"] for row in rows]),
        offsets=np.arange(0, 2 * count + 1, 2, dtype=np.int64),
        atomic_numbers=np.tile(np.asarray([1, 1], dtype=np.int32), count),
        positions_bohr=np.tile(
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=np.float64),
            (count, 1),
        ),
        charges=np.asarray([int(row["charge"]) for row in rows], dtype=np.int32),
        multiplicities=np.ones(count, dtype=np.int32),
        unpaired_electrons=np.zeros(count, dtype=np.int32),
        natoms=np.full(count, 2, dtype=np.int32),
        gfn2_n_ao=np.full(count, 5, dtype=np.int32),
        input_sha256=np.asarray([row["xtbloom_input_sha256"] for row in rows]),
    )


def write_omol_bundle(root: Path) -> Path:
    """Create interleaved CSV metadata and canonical NPZ subset sequences."""
    main = [omol_row("main-b", "main", "02"), omol_row("main-a", "main", "01")]
    stress = [omol_row("stress-a", "stress", "00", charge=1)]
    performance = [omol_row("performance-a", "performance", "03")]
    # The CSV is intentionally not the canonical execution sequence.
    manifest = root / "manifests" / "selected_samples.csv"
    write_csv(manifest, [stress[0], main[0], performance[0], main[1]])
    source_manifest = root / "provenance" / "source_manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        json.dumps({"sampling": {"order": ["main", "stress", "performance"]}}),
        encoding="utf-8",
    )
    # NPZ sample_ids are authoritative inside each declared subset.
    write_npz(root / "samples" / "main.npz", [main[1], main[0]])
    write_npz(root / "samples" / "stress.npz", stress)
    write_npz(root / "samples" / "performance.npz", performance)
    return manifest


def default_args(root: Path) -> Namespace:
    """Return a complete runner namespace for adapter-unit tests."""
    return Namespace(
        manifest=root / "manifests" / "selected_samples.csv",
        output_dir=root / "output",
        dataset="qm9",
        subsets=None,
        limit=None,
        shard_count=1,
        shard_index=0,
        engines=("tblite",),
        batch_size=2,
        fsync_every=1,
        library=None,
        xtb_library=None,
        tblite_library=None,
        dxtb_source=None,
        backend="cpu",
        memory_mode="host",
        device_id=0,
        cpu_threads=1,
        accuracy=1.0e-4,
        max_scc_iterations=500,
        electronic_temperature_kelvin=300.0,
    )


class DatasetManifestTest(unittest.TestCase):
    """Verify the two frozen input contracts and deterministic selection."""

    def test_qm9_uses_sample_order_and_preserves_units_charge_spin_and_strata(
        self,
    ) -> None:
        """Treat sample_order as authoritative and retain physical metadata."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = manifest_row("qm9-first", 0)
            second = manifest_row("qm9-second", 1)
            manifest = write_qm9_bundle(root, [second, first])

            items = list(dataset_runner.load_manifest(manifest, "qm9"))

        self.assertEqual(
            [item.system_id for item in items], ["qm9-first", "qm9-second"]
        )
        system = items[0].system
        assert system is not None
        self.assertEqual(system.atomic_numbers, (1, 1))
        self.assertEqual(system.positions_bohr[1], (0.0, 0.0, 1.4))
        self.assertEqual(
            (system.charge, system.multiplicity, system.unpaired_electrons), (0, 1, 0)
        )
        self.assertEqual(system.manifest["coordinate_output_unit"], "bohr")
        self.assertEqual(system.strata["sampling_stratum"], "stratum-0")

    def test_qm9_malformed_row_is_retained_and_does_not_hide_next_system(self) -> None:
        """Keep an invalid-unit row while continuing to its healthy peer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = manifest_row("bad-unit", 0, unit="angstrom")
            good = manifest_row("good", 1)
            manifest = write_qm9_bundle(root, [bad, good])

            items = list(dataset_runner.load_manifest(manifest, "qm9"))

        self.assertIsNone(items[0].system)
        self.assertIn("already be in bohr", items[0].error or "")
        self.assertIsNotNone(items[1].system)

    def test_qm9_selects_the_manifest_sample_set_and_rejects_mixed_sets(self) -> None:
        """Load a disjoint performance payload and reject ambiguous bundles."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            performance = manifest_row("performance", 0, sample_set="performance")
            manifest = write_qm9_bundle(root, [performance])
            items = list(dataset_runner.load_manifest(manifest, "qm9"))
            self.assertEqual([item.system_id for item in items], ["performance"])

            mixed = root / "mixed" / "manifests" / "selected_samples.csv"
            write_csv(
                mixed,
                [performance, manifest_row("main", 1, sample_set="main")],
            )
            with self.assertRaisesRegex(
                dataset_runner.DatasetRunnerError, "exactly one sample_set"
            ):
                list(dataset_runner.load_manifest(mixed, "qm9"))

    def test_omol25_uses_declared_subset_and_npz_id_order_not_csv_order(self) -> None:
        """Use source-manifest subset order and NPZ IDs instead of CSV order."""
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_omol_bundle(Path(temporary))
            items = list(dataset_runner.load_manifest(manifest, "omol25"))

        self.assertEqual(
            [item.system_id for item in items],
            ["main-a", "main-b", "stress-a", "performance-a"],
        )
        self.assertEqual([item.canonical_ordinal for item in items], list(range(4)))
        stress = items[2].system
        assert stress is not None
        self.assertEqual(stress.charge, 1)
        self.assertEqual(stress.strata["sampling_stratum"], "stress-stratum")

    def test_limit_precedes_sharding_and_shards_have_no_duplicates_or_gaps(
        self,
    ) -> None:
        """Partition the same globally limited canonical sequence exactly once."""
        items = [
            dataset_runner.LoadedItem("qm9", "main", str(index), index, {}, None, "x")
            for index in range(10)
        ]
        shards = []
        for shard_index in range(3):
            selected = list(dataset_runner.select_items(items, None, 8, 3, shard_index))
            shards.append([ordinal for ordinal, _ in selected])
        flattened = [ordinal for shard in shards for ordinal in shard]
        self.assertEqual(sorted(flattened), list(range(8)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(shards[0], [0, 3, 6])

    def test_unknown_subset_is_rejected_instead_of_succeeding_empty(self) -> None:
        """Catch subset typos before any empty output directory is created."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_qm9_bundle(root, [manifest_row("qm9", 0)])
            args = default_args(root)
            args.manifest = manifest
            args.subsets = ("typo",)

            with self.assertRaisesRegex(
                dataset_runner.DatasetRunnerError, "unknown subsets: typo"
            ):
                dataset_runner.run_dataset(args)

            self.assertFalse(args.output_dir.exists())


class AdapterReuseAndSerializationTest(unittest.TestCase):
    """Verify shared adapter identities, failure isolation, and JSON output."""

    def test_runner_reuses_the_real_shared_adapter_classes(self) -> None:
        """Bind every engine name to the existing benchmark adapter class."""
        self.assertIs(
            dataset_runner.SHARED_ADAPTERS["xtbloom"], benchmark_run.XTBloomAdapter
        )
        self.assertIs(dataset_runner.SHARED_ADAPTERS["xtb"], XtbAdapter)
        self.assertIs(dataset_runner.SHARED_ADAPTERS["tblite"], TbliteAdapter)
        self.assertIs(dataset_runner.SHARED_ADAPTERS["dxtb"], DxtbAdapter)

    def test_xtbloom_receives_the_requested_electronic_temperature(self) -> None:
        """Keep recorded and executed non-default temperatures identical."""
        args = default_args(Path("."))
        args.engines = ("xtbloom",)
        args.library = Path("libxtbloom.so")
        args.electronic_temperature_kelvin = 123.5
        storage = SimpleNamespace(slices=[SimpleNamespace()])
        sentinel = object()
        with (
            mock.patch.object(
                dataset_runner.SHARED_ADAPTERS["xtbloom"],
                "from_storage",
                return_value=sentinel,
            ) as from_storage,
            mock.patch.object(
                dataset_runner, "_require_library", return_value=args.library
            ),
        ):
            created = dataset_runner.create_adapter("xtbloom", storage, args)
        self.assertIs(created, sentinel)
        self.assertAlmostEqual(
            from_storage.call_args.kwargs["electronic_temperature_hartree"],
            123.5 * dataset_runner.public_api.XTBLOOM_KELVIN_TO_HARTREE,
        )

    def test_schema_v2_taxonomy_residual_and_dxtb_device_are_explicit(self) -> None:
        """Make unavailable diagnostics distinguishable from omitted fields."""
        self.assertEqual(dataset_runner.SCHEMA_VERSION, 2)
        self.assertEqual(dataset_runner._status_category(7, 0), "scc_not_converged")
        self.assertEqual(dataset_runner._status_category(4, 0), "resource_or_oom")
        self.assertEqual(
            dataset_runner._status_category(8, 0), "eigensolver_or_numerical_failure"
        )
        self.assertEqual(dataset_runner._status_category(99, 0), "unknown_error")
        args = default_args(Path("."))
        args.backend = "cuda"
        args.memory_mode = "device"
        with mock.patch.object(
            benchmark_run,
            "git_state",
            return_value={"revision": "fixture", "dirty": False},
        ):
            program = dataset_runner._program_document("dxtb", args)
        self.assertEqual(program["memory_mode"], "device")

        item = dataset_runner.LoadedItem("qm9", "main", "bad", 0, {}, None, "failure")
        record = dataset_runner._failure_record(
            "dxtb", 0, item, "unknown_error", "failure", args, program
        )
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(
            record["results"]["final_scc_residual"]["availability"], "unavailable"
        )

    def test_reference_failure_isolated_per_system_and_later_result_survives(
        self,
    ) -> None:
        """Continue after a middle-system OOM in a reference adapter."""
        systems = []
        for index, system_id in enumerate(("good-0", "oom", "good-2")):
            systems.append(
                dataset_runner.DatasetSystem(
                    dataset="qm9",
                    subset="main",
                    system_id=system_id,
                    canonical_ordinal=index,
                    atomic_numbers=(1, 1),
                    positions_bohr=((0.0, 0.0, 0.0), (0.0, 0.0, 1.4)),
                    charge=0,
                    multiplicity=1,
                    unpaired_electrons=0,
                    input_sha256=f"hash-{index}",
                    strata={},
                    manifest={
                        "coordinate_source_unit": "angstrom",
                        "angstrom_per_bohr": "0.529177210903",
                    },
                )
            )
        selected = [
            (
                index,
                dataset_runner.LoadedItem(
                    "qm9", "main", system.system_id, index, {}, system, None
                ),
            )
            for index, system in enumerate(systems)
        ]
        args = default_args(Path("."))

        class FakeAdapter:
            version = 700
            accuracy = 1.0e-4
            electronic_temperature_hartree = 9.5e-4
            max_iterations = 500
            threads = 1

            def __init__(self, storage: object) -> None:
                self.storage = storage

            def invoke(self) -> None:
                if self.storage.slices[0].case["id"] == "oom":
                    raise MemoryError("fixture OOM")

            def results(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.0] * 6,
                    "atomic_charges_e": [0.0, 0.0],
                }

            def close(self) -> None:
                return None

        records = []
        with mock.patch.object(
            dataset_runner,
            "create_adapter",
            side_effect=lambda _engine, storage, _args: FakeAdapter(storage),
        ):
            for entry in selected:
                records.extend(dataset_runner.execute_chunk("tblite", [entry], args))

        self.assertEqual(
            [record["status"]["category"] for record in records],
            ["success", "resource_or_oom", "success"],
        )
        self.assertEqual(records[2]["results"]["energy"]["availability"], "available")

    def test_xtbloom_closes_failed_batch_before_per_system_retries(self) -> None:
        """Release failed batch resources before attempting OOM isolation."""
        systems = [
            dataset_runner.DatasetSystem(
                dataset="qm9",
                subset="main",
                system_id=f"system-{index}",
                canonical_ordinal=index,
                atomic_numbers=(1,),
                positions_bohr=((0.0, 0.0, 0.0),),
                charge=0,
                multiplicity=1,
                unpaired_electrons=0,
                input_sha256=f"hash-{index}",
                strata={},
                manifest={},
            )
            for index in range(2)
        ]
        selected = [
            (
                index,
                dataset_runner.LoadedItem(
                    "qm9", "main", system.system_id, index, {}, system, None
                ),
            )
            for index, system in enumerate(systems)
        ]
        events: list[str] = []

        class FakeAdapter:
            def __init__(self, storage: object) -> None:
                self.storage = storage
                self.options = SimpleNamespace(
                    api_version=2,
                    charge_tolerance=1.0e-10,
                    energy_tolerance=1.0e-12,
                    electronic_temperature=9.5e-4,
                    max_scc_iterations=500,
                )
                events.append(f"create-{len(storage.slices)}")

            def invoke(self) -> None:
                events.append(f"invoke-{len(self.storage.slices)}")
                if len(self.storage.slices) > 1:
                    raise MemoryError("fixture aggregate OOM")

            def synchronize(self) -> None:
                return None

            def raw_results(self) -> dict[str, object]:
                return {
                    "energies_hartree": [-1.0],
                    "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
                    "atomic_charges_e": [0.0],
                    "scc_iterations": [7],
                    "scc_converged": [1],
                    "per_system_status": [0],
                }

            def close(self) -> None:
                events.append(f"close-{len(self.storage.slices)}")

        with mock.patch.object(
            dataset_runner,
            "create_adapter",
            side_effect=lambda _engine, storage, _args: FakeAdapter(storage),
        ):
            records = dataset_runner.execute_chunk(
                "xtbloom", selected, default_args(Path(".")), {"engine": "xtbloom"}
            )

        self.assertEqual(events[:4], ["create-2", "invoke-2", "close-2", "create-1"])
        self.assertEqual(
            [record["status"]["category"] for record in records],
            ["success", "success"],
        )

    def test_nonfinite_adapter_value_is_serialized_as_unavailable_standard_json(
        self,
    ) -> None:
        """Replace NaN and classify the required result as numerical failure."""
        system = dataset_runner.DatasetSystem(
            "qm9",
            "main",
            "nan-energy",
            0,
            (1,),
            ((0.0, 0.0, 0.0),),
            0,
            1,
            0,
            "hash",
            {},
            {},
        )
        item = dataset_runner.LoadedItem(
            "qm9", "main", system.system_id, 0, {}, system, None
        )
        storage = dataset_runner.storage_from_systems([system])
        adapter = SimpleNamespace(
            version=700,
            accuracy=1.0e-4,
            electronic_temperature_hartree=9.5e-4,
            max_iterations=500,
            threads=1,
        )
        records = dataset_runner._records_from_outputs(
            "tblite",
            [(0, item)],
            storage,
            {
                "energies_hartree": [float("nan")],
                "forces_hartree_per_bohr": [0.0, 0.0, 0.0],
                "atomic_charges_e": [0.0],
            },
            adapter,
            default_args(Path(".")),
            {"scope": "fixture"},
            {"engine": "tblite"},
        )
        field = records[0]["results"]["energy"]
        self.assertEqual(
            records[0]["status"]["category"],
            "eigensolver_or_numerical_failure",
        )
        self.assertEqual(field["availability"], "unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            with dataset_runner.JsonlSink(path, fsync_every=1) as sink:
                sink.write(records[0])
            parsed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["results"]["energy"]["availability"], "unavailable")

    def test_program_identity_probes_are_constant_per_engine(self) -> None:
        """Avoid per-record Git subprocesses and repeated library hashing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                manifest_row(f"bad-{index}", index, unit="angstrom")
                for index in range(3)
            ]
            manifest = write_qm9_bundle(root, rows)
            args = default_args(root)
            args.manifest = manifest
            args.engines = ("xtbloom", "xtb")
            args.library = root / "libxtbloom.so"
            args.xtb_library = root / "libxtb.so"
            args.library.write_bytes(b"xtbloom")
            args.xtb_library.write_bytes(b"xtb")
            with (
                mock.patch.object(
                    benchmark_run,
                    "git_state",
                    return_value={"revision": "fixture", "dirty": True},
                ) as git_state,
                mock.patch.object(
                    benchmark_run,
                    "sha256_file",
                    wraps=benchmark_run.sha256_file,
                ) as sha256_file,
            ):
                summary = dataset_runner.run_dataset(args)

        self.assertEqual(summary["record_count"], 6)
        self.assertEqual(git_state.call_count, 3)  # runner plus two engines
        self.assertEqual(sha256_file.call_count, 3)  # manifest plus two libraries

    def test_storage_preserves_ragged_geometry_charge_and_spin(self) -> None:
        """Pass ragged geometry and electronic state unchanged to adapters."""
        system = dataset_runner.DatasetSystem(
            dataset="omol25",
            subset="stress",
            system_id="open-shell",
            canonical_ordinal=0,
            atomic_numbers=(6, 1, 1),
            positions_bohr=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            charge=1,
            multiplicity=3,
            unpaired_electrons=2,
            input_sha256="hash",
            strata={},
            manifest={},
        )
        storage = dataset_runner.storage_from_systems([system])
        self.assertEqual(storage.atom_offsets, [0, 3])
        self.assertEqual(storage.molecular_charges, [1.0])
        self.assertEqual(storage.unpaired_electrons, [2])
        self.assertEqual(storage.spin_channels, [2])
        self.assertEqual(storage.positions[-3:], [-1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
