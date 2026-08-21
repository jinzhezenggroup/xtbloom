#!/usr/bin/env python3
"""Run frozen QM9/OMol25 manifests through the shared benchmark adapters.

The runner preserves the frozen bundle's canonical order and writes one
independent JSONL record per engine/system pair.  It intentionally does not
compare against the datasets' DFT labels: the frozen bundles contain GFN2-xTB
inputs and provenance only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

if __package__ in (None, ""):
    repository_root = Path(
        os.environ.get("PAPER_REPO_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    sys.path.insert(0, str(repository_root / "benchmarks"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run as benchmark_run
    from dxtb_adapter import DxtbAdapter
    from tblite_adapter import TbliteAdapter
    from xtb_adapter import XtbAdapter
else:
    from . import run as benchmark_run
    from .dxtb_adapter import DxtbAdapter
    from .tblite_adapter import TbliteAdapter
    from .xtb_adapter import XtbAdapter

public_api = benchmark_run.public_api

SCHEMA_VERSION = 2
OMOL25_FORMAT_VERSION = "xtbloom-omol25-ragged-v1"
DEFAULT_OMOL25_SET_ORDER = ("main", "stress", "performance")
STATUS_NAMES = {
    0: "success",
    1: "invalid_argument",
    2: "backend_unavailable",
    3: "not_supported",
    4: "allocation_failed",
    5: "not_implemented",
    6: "internal_error",
    7: "scc_not_converged",
    8: "eigensolver_failed",
}
QM9_STRATA_FIELDS = (
    "heavy_atom_count",
    "element_signature",
    "natoms",
    "gfn2_n_ao",
    "gfn2_ao_bin",
    "sampling_stratum",
    "stratum_population",
    "stratum_sample_count",
    "sampling_probability",
)
OMOL25_STRATA_FIELDS = (
    "domain",
    "configuration_class",
    "charge_class",
    "spin_class",
    "element_class",
    "natoms_bin",
    "gfn2_ao_bin",
    "contains_metal",
    "contains_transition_metal",
    "contains_f_block",
    "contains_heavy",
    "sampling_stratum",
    "sampling_probability",
)

# These identities are deliberately public and tested.  The runner owns only
# dataset parsing, isolation, and serialization; the calculations stay in the
# same adapters used by benchmarks/run.py.
SHARED_ADAPTERS = {
    "xtbloom": benchmark_run.XTBloomAdapter,
    "xtb": XtbAdapter,
    "tblite": TbliteAdapter,
    "dxtb": DxtbAdapter,
}
Adapter = benchmark_run.XTBloomAdapter | XtbAdapter | TbliteAdapter | DxtbAdapter


class DatasetRunnerError(RuntimeError):
    """A malformed bundle, invalid CLI request, or unsafe output operation."""


class EngineUnavailable(DatasetRunnerError):
    """A requested adapter cannot be constructed in the selected environment."""


@dataclass(frozen=True)
class DatasetSystem:
    """One validated GFN2-xTB input plus its frozen manifest metadata."""

    dataset: str
    subset: str
    system_id: str
    canonical_ordinal: int
    atomic_numbers: tuple[int, ...]
    positions_bohr: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    unpaired_electrons: int
    input_sha256: str | None
    strata: dict[str, str]
    manifest: dict[str, str]


@dataclass(frozen=True)
class LoadedItem:
    """A canonical manifest row, valid or explicitly malformed."""

    dataset: str
    subset: str
    system_id: str
    canonical_ordinal: int
    manifest: dict[str, str]
    system: DatasetSystem | None
    error: str | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetRunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetRunnerError(f"JSON document must be an object: {path}")
    return document


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetRunnerError(f"manifest has no header: {path}")
            rows = [dict(row) for row in reader]
            return list(reader.fieldnames), rows
    except OSError as exc:
        raise DatasetRunnerError(f"cannot read manifest {path}: {exc}") from exc


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise DatasetRunnerError(f"missing required manifest field {key}")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise DatasetRunnerError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetRunnerError(f"{label} must be an integer") from exc
    if isinstance(value, float) and value != converted:
        raise DatasetRunnerError(f"{label} must be an integer")
    return converted


def _validate_geometry(
    atomic_numbers: Sequence[Any], positions: Sequence[Sequence[Any]]
) -> tuple[tuple[int, ...], tuple[tuple[float, float, float], ...]]:
    numbers = tuple(_integer(value, "atomic number") for value in atomic_numbers)
    if not numbers or any(value <= 0 for value in numbers):
        raise DatasetRunnerError("atomic numbers must be nonempty positive integers")
    if len(positions) != len(numbers):
        raise DatasetRunnerError("positions length does not match atomic numbers")
    coordinates: list[tuple[float, float, float]] = []
    for row in positions:
        if len(row) != 3:
            raise DatasetRunnerError("each position must contain three components")
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            raise DatasetRunnerError("positions contain a non-finite value")
        coordinates.append(values)
    return numbers, tuple(coordinates)


def _electronic_state(
    charge: object, multiplicity: object, unpaired_electrons: object
) -> tuple[int, int, int]:
    converted_charge = _integer(charge, "charge")
    converted_multiplicity = _integer(multiplicity, "multiplicity")
    converted_unpaired = _integer(unpaired_electrons, "unpaired electrons")
    if converted_multiplicity <= 0 or converted_unpaired < 0:
        raise DatasetRunnerError("multiplicity must be positive and spin nonnegative")
    if converted_multiplicity != converted_unpaired + 1:
        raise DatasetRunnerError(
            "multiplicity must equal unpaired_electrons + 1 in frozen bundles"
        )
    return converted_charge, converted_multiplicity, converted_unpaired


def _check_bohr_unit(row: dict[str, str]) -> None:
    if _required(row, "coordinate_output_unit").lower() != "bohr":
        raise DatasetRunnerError(
            "frozen runtime coordinates must already be in bohr; no implicit "
            "unit conversion is performed"
        )


def _malformed_item(
    dataset: str,
    subset: str,
    system_id: str,
    ordinal: int,
    manifest: dict[str, str],
    exc: BaseException,
) -> LoadedItem:
    return LoadedItem(
        dataset=dataset,
        subset=subset,
        system_id=system_id,
        canonical_ordinal=ordinal,
        manifest=manifest,
        system=None,
        error=str(exc),
    )


def _qm9_items(manifest_path: Path) -> Iterator[LoadedItem]:
    _, rows = _csv_rows(manifest_path)
    try:
        ordered = sorted(
            rows, key=lambda row: _integer(row.get("sample_order"), "sample_order")
        )
    except DatasetRunnerError:
        raise
    orders = [_integer(row.get("sample_order"), "sample_order") for row in ordered]
    if orders != list(range(len(rows))):
        raise DatasetRunnerError(
            "QM9 sample_order must be unique and contiguous from zero"
        )
    sample_sets = {_required(row, "sample_set") for row in rows}
    if len(sample_sets) != 1:
        raise DatasetRunnerError(
            "one QM9 manifest must select exactly one sample_set so its sample "
            "payload remains independently hash-pinned"
        )
    samples_path = (
        manifest_path.parents[1] / "samples" / f"{next(iter(sample_sets))}.jsonl.gz"
    )
    try:
        sample_handle = gzip.open(  # noqa: SIM115 - wrap open errors with bundle path
            samples_path, "rt", encoding="utf-8"
        )
    except OSError as exc:
        raise DatasetRunnerError(
            f"cannot open QM9 samples {samples_path}: {exc}"
        ) from exc
    with sample_handle:
        for ordinal, pair in enumerate(
            itertools.zip_longest(ordered, sample_handle, fillvalue=None)
        ):
            row, line = pair
            if row is None:
                raise DatasetRunnerError(
                    "QM9 samples contain more rows than the manifest"
                )
            system_id = row.get("sample_id") or f"manifest-row-{ordinal}"
            subset = row.get("sample_set") or "main"
            try:
                if line is None:
                    raise DatasetRunnerError(
                        "QM9 sample JSONL ended before the manifest"
                    )
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise DatasetRunnerError("QM9 sample line must be a JSON object")
                if sample.get("sample_id") != system_id:
                    raise DatasetRunnerError(
                        "QM9 manifest/sample system IDs do not match"
                    )
                _check_bohr_unit(row)
                numbers, positions = _validate_geometry(
                    sample.get("atomic_numbers", ()), sample.get("positions_bohr", ())
                )
                charge, multiplicity, unpaired = _electronic_state(
                    sample.get("charge"),
                    sample.get("multiplicity"),
                    sample.get("unpaired_electrons"),
                )
                for key, actual in (
                    ("charge", charge),
                    ("multiplicity", multiplicity),
                    ("unpaired_electrons", unpaired),
                    ("natoms", len(numbers)),
                ):
                    if _integer(row.get(key), key) != actual:
                        raise DatasetRunnerError(f"QM9 manifest/sample {key} mismatch")
                input_hash = sample.get("xtbloom_input_sha256")
                if input_hash != row.get("xtbloom_input_sha256"):
                    raise DatasetRunnerError("QM9 manifest/sample input hash mismatch")
                system = DatasetSystem(
                    dataset="qm9",
                    subset=subset,
                    system_id=system_id,
                    canonical_ordinal=ordinal,
                    atomic_numbers=numbers,
                    positions_bohr=positions,
                    charge=charge,
                    multiplicity=multiplicity,
                    unpaired_electrons=unpaired,
                    input_sha256=str(input_hash) if input_hash is not None else None,
                    strata={key: row[key] for key in QM9_STRATA_FIELDS if key in row},
                    manifest=row,
                )
                yield LoadedItem(
                    dataset="qm9",
                    subset=subset,
                    system_id=system_id,
                    canonical_ordinal=ordinal,
                    manifest=row,
                    system=system,
                    error=None,
                )
            except (
                DatasetRunnerError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                yield _malformed_item("qm9", subset, system_id, ordinal, row, exc)


def _omol25_set_order(bundle_root: Path) -> tuple[str, ...]:
    source_manifest_path = bundle_root / "provenance" / "source_manifest.json"
    if not source_manifest_path.is_file():
        return DEFAULT_OMOL25_SET_ORDER
    document = _load_json(source_manifest_path)
    raw = document.get("sampling", {}).get("order")
    if not isinstance(raw, list) or not raw or not all(isinstance(x, str) for x in raw):
        raise DatasetRunnerError("OMol25 source manifest sampling.order is invalid")
    return tuple(raw)


def _omol25_items(manifest_path: Path) -> Iterator[LoadedItem]:
    _, rows = _csv_rows(manifest_path)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        system_id = _required(row, "configuration_id")
        if system_id in by_id:
            raise DatasetRunnerError(f"duplicate OMol25 configuration_id: {system_id}")
        by_id[system_id] = row
    bundle_root = manifest_path.parents[1]
    ordinal = 0
    seen: set[str] = set()
    for subset in _omol25_set_order(bundle_root):
        sample_path = bundle_root / "samples" / f"{subset}.npz"
        try:
            payload_context = np.load(sample_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DatasetRunnerError(
                f"cannot load OMol25 samples {sample_path}: {exc}"
            ) from exc
        with payload_context as payload:
            required = {
                "format_version",
                "sample_ids",
                "offsets",
                "atomic_numbers",
                "positions_bohr",
                "charges",
                "multiplicities",
                "unpaired_electrons",
                "natoms",
                "input_sha256",
            }
            missing = required - set(payload.files)
            if missing:
                raise DatasetRunnerError(
                    f"OMol25 {subset} NPZ missing arrays: {', '.join(sorted(missing))}"
                )
            if str(payload["format_version"].item()) != OMOL25_FORMAT_VERSION:
                raise DatasetRunnerError(
                    f"unsupported OMol25 NPZ format: {sample_path}"
                )
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            offsets = payload["offsets"]
            numbers_array = payload["atomic_numbers"]
            positions_array = payload["positions_bohr"]
            arrays = {
                "charges": payload["charges"],
                "multiplicities": payload["multiplicities"],
                "unpaired_electrons": payload["unpaired_electrons"],
                "natoms": payload["natoms"],
                "input_sha256": payload["input_sha256"],
            }
            if offsets.shape != (len(sample_ids) + 1,):
                raise DatasetRunnerError(f"OMol25 {subset} offsets shape is invalid")
            for key, values in arrays.items():
                if values.shape != (len(sample_ids),):
                    raise DatasetRunnerError(f"OMol25 {subset} {key} shape is invalid")
            for local_index, system_id in enumerate(sample_ids):
                row = by_id.get(system_id, {})
                try:
                    if not row:
                        raise DatasetRunnerError(
                            "OMol25 NPZ ID is absent from manifest"
                        )
                    if system_id in seen:
                        raise DatasetRunnerError(
                            "OMol25 sample ID appears in multiple subsets"
                        )
                    if row.get("sample_set") != subset:
                        raise DatasetRunnerError("OMol25 manifest/NPZ subset mismatch")
                    _check_bohr_unit(row)
                    begin = _integer(offsets[local_index].item(), "atom offset")
                    end = _integer(offsets[local_index + 1].item(), "atom offset")
                    if begin < 0 or end <= begin or end > len(numbers_array):
                        raise DatasetRunnerError("OMol25 ragged atom extent is invalid")
                    numbers, positions = _validate_geometry(
                        numbers_array[begin:end].tolist(),
                        positions_array[begin:end].tolist(),
                    )
                    charge, multiplicity, unpaired = _electronic_state(
                        arrays["charges"][local_index].item(),
                        arrays["multiplicities"][local_index].item(),
                        arrays["unpaired_electrons"][local_index].item(),
                    )
                    expected_values = {
                        "charge": charge,
                        "multiplicity": multiplicity,
                        "unpaired_electrons": unpaired,
                        "natoms": len(numbers),
                    }
                    for key, actual in expected_values.items():
                        if _integer(row.get(key), key) != actual:
                            raise DatasetRunnerError(
                                f"OMol25 manifest/NPZ {key} mismatch"
                            )
                    input_hash = str(arrays["input_sha256"][local_index])
                    if row.get("xtbloom_input_sha256") != input_hash:
                        raise DatasetRunnerError(
                            "OMol25 manifest/NPZ input hash mismatch"
                        )
                    system = DatasetSystem(
                        dataset="omol25",
                        subset=subset,
                        system_id=system_id,
                        canonical_ordinal=ordinal,
                        atomic_numbers=numbers,
                        positions_bohr=positions,
                        charge=charge,
                        multiplicity=multiplicity,
                        unpaired_electrons=unpaired,
                        input_sha256=input_hash,
                        strata={
                            key: row[key] for key in OMOL25_STRATA_FIELDS if key in row
                        },
                        manifest=row,
                    )
                    yield LoadedItem(
                        dataset="omol25",
                        subset=subset,
                        system_id=system_id,
                        canonical_ordinal=ordinal,
                        manifest=row,
                        system=system,
                        error=None,
                    )
                except (DatasetRunnerError, TypeError, ValueError, IndexError) as exc:
                    yield _malformed_item(
                        "omol25", subset, system_id, ordinal, row, exc
                    )
                finally:
                    seen.add(system_id)
                    ordinal += 1
    missing_ids = set(by_id) - seen
    if missing_ids:
        raise DatasetRunnerError(
            "OMol25 manifest rows are absent from canonical NPZ sets: "
            + ", ".join(sorted(missing_ids)[:10])
        )


def detect_dataset(manifest_path: Path) -> str:
    """Identify one of the two frozen contracts from its real CSV header."""
    fieldnames, _ = _csv_rows(manifest_path)
    fields = set(fieldnames)
    if {"sample_order", "sample_id", "source_split"} <= fields:
        return "qm9"
    if {"sample_set", "configuration_id", "property_id"} <= fields:
        return "omol25"
    raise DatasetRunnerError(
        "manifest does not match the frozen QM9 or OMol25 contract"
    )


def load_manifest(manifest_path: Path, dataset: str = "auto") -> Iterator[LoadedItem]:
    """Yield the frozen canonical sequence without inventing another format."""
    manifest_path = manifest_path.resolve()
    selected = detect_dataset(manifest_path) if dataset == "auto" else dataset
    if selected == "qm9":
        yield from _qm9_items(manifest_path)
    elif selected == "omol25":
        yield from _omol25_items(manifest_path)
    else:
        raise DatasetRunnerError(f"unsupported dataset contract: {selected}")


def select_items(
    items: Iterable[LoadedItem],
    subsets: set[str] | None,
    limit: int | None,
    shard_count: int,
    shard_index: int,
) -> Iterator[tuple[int, LoadedItem]]:
    """Apply subset filter, then global limit, then deterministic modulo shard."""
    if limit is not None and limit < 0:
        raise DatasetRunnerError("limit must be nonnegative")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise DatasetRunnerError("shard index must be in [0, shard_count)")
    selected_ordinal = 0
    for item in items:
        if subsets is not None and item.subset not in subsets:
            continue
        if limit is not None and selected_ordinal >= limit:
            break
        if selected_ordinal % shard_count == shard_index:
            yield selected_ordinal, item
        selected_ordinal += 1


def storage_from_systems(
    systems: Sequence[DatasetSystem],
) -> public_api.PublicBatchStorage:
    """Assemble the same duck-typed storage consumed by all shared adapters."""
    atom_offsets = [0]
    atomic_numbers: list[int] = []
    positions: list[float] = []
    slices: list[Any] = []
    charges: list[float] = []
    unpaired: list[int] = []
    spin_channels: list[int] = []
    for system in systems:
        begin = len(atomic_numbers)
        atomic_numbers.extend(system.atomic_numbers)
        positions.extend(value for row in system.positions_bohr for value in row)
        end = len(atomic_numbers)
        atom_offsets.append(end)
        charges.append(float(system.charge))
        unpaired.append(system.unpaired_electrons)
        spin_channels.append(2 if system.unpaired_electrons else 1)
        slices.append(
            public_api.CaseSlice(
                case={"id": system.system_id},
                atom_begin=begin,
                atom_end=end,
                point_begin=0,
                point_end=0,
                expected={},
            )
        )
    return public_api.PublicBatchStorage(
        atom_offsets=atom_offsets,
        atomic_numbers=atomic_numbers,
        positions=positions,
        molecular_charges=charges,
        unpaired_electrons=unpaired,
        spin_channels=spin_channels,
        point_charge_offsets=[0] * (len(systems) + 1),
        point_charge_positions=[],
        point_charge_values=[],
        point_charge_gammas=[],
        slices=slices,
        keepalive=[],
        efields=[None] * len(systems),
    )


def _require_library(path: Path | None, engine: str) -> Path:
    if path is None or not path.is_file():
        raise EngineUnavailable(f"{engine} shared library is unavailable: {path}")
    return path.resolve()


def create_adapter(
    engine: str,
    storage: public_api.PublicBatchStorage,
    args: argparse.Namespace,
) -> Adapter:
    """Construct the real shared adapter selected by the CLI."""
    if engine == "xtbloom":
        library = _require_library(args.library, engine)
        cell = benchmark_run.Cell(
            "xtbloom",
            args.backend,
            args.memory_mode,
            "dataset-manifest",
            "force",
            len(storage.slices),
        )
        return SHARED_ADAPTERS[engine].from_storage(
            library,
            storage,
            cell,
            args.device_id,
            args.cpu_threads,
            collect_atomic_charges=True,
            max_scc_iterations=args.max_scc_iterations,
            electronic_temperature_hartree=(
                args.electronic_temperature_kelvin
                * public_api.XTBLOOM_KELVIN_TO_HARTREE
            ),
        )
    if engine == "xtb":
        return SHARED_ADAPTERS[engine](
            _require_library(args.xtb_library, engine),
            storage,
            "force",
            None,
            accuracy=args.accuracy,
            max_iterations=args.max_scc_iterations,
            electronic_temperature_kelvin=args.electronic_temperature_kelvin,
            threads=args.cpu_threads,
        )
    if engine == "tblite":
        return SHARED_ADAPTERS[engine](
            _require_library(args.tblite_library, engine),
            storage,
            "force",
            accuracy=args.accuracy,
            max_iterations=args.max_scc_iterations,
            electronic_temperature_hartree=(
                args.electronic_temperature_kelvin
                * public_api.XTBLOOM_KELVIN_TO_HARTREE
            ),
            collect_atomic_charges=True,
            threads=args.cpu_threads,
        )
    if engine == "dxtb":
        return SHARED_ADAPTERS[engine](
            storage,
            "force",
            args.backend,
            device_id=args.device_id,
            cpu_threads=args.cpu_threads,
            source_root=args.dxtb_source,
            accuracy=args.accuracy,
            force_convergence=True,
            max_iterations=args.max_scc_iterations,
        )
    raise DatasetRunnerError(f"unknown engine: {engine}")


def _adapter_versions(engine: str, adapter: Adapter) -> dict[str, Any]:
    details: dict[str, Any] = {"adapter_class": type(adapter).__name__}
    if engine == "xtbloom":
        details.update(
            {
                "api_version": int(adapter.options.api_version),
                "model": "GFN2-xTB",
                "settings": {
                    "charge_tolerance": float(adapter.options.charge_tolerance),
                    "energy_tolerance_hartree": float(adapter.options.energy_tolerance),
                    "electronic_temperature_hartree": float(
                        adapter.options.electronic_temperature
                    ),
                    "max_scc_iterations": int(adapter.options.max_scc_iterations),
                },
            }
        )
    elif engine == "xtb":
        details.update(
            {
                "api_version": adapter.api_version,
                "settings": {
                    "accuracy": adapter.accuracy,
                    "electronic_temperature_kelvin": (
                        adapter.electronic_temperature_kelvin
                    ),
                    "max_scc_iterations": adapter.max_iterations,
                    "threads": adapter.threads,
                },
            }
        )
    elif engine == "tblite":
        details.update(
            {
                "api_version": adapter.version,
                "settings": {
                    "accuracy": adapter.accuracy,
                    "electronic_temperature_hartree": (
                        adapter.electronic_temperature_hartree
                    ),
                    "max_scc_iterations": adapter.max_iterations,
                    "threads": adapter.threads,
                },
            }
        )
    elif engine == "dxtb":
        details.update(
            {
                "dxtb_version": adapter.version,
                "torch_version": adapter.torch_version,
                "module_path": adapter.module_path,
                "settings": {
                    "electronic_temperature": _unavailable(
                        "shared dxtb adapter does not expose a temperature option"
                    ),
                    "fixed_point_l2_tolerance": adapter.accuracy,
                    "fixed_point_max_norm_tolerance": adapter.max_norm_tolerance,
                    "function_residual_tolerance": adapter.function_tolerance,
                    "force_convergence": adapter.force_convergence,
                    "max_scc_iterations": adapter.max_iterations,
                    "threads": adapter.cpu_threads,
                },
            }
        )
    return details


def _unavailable(reason: str) -> dict[str, Any]:
    return {"availability": "unavailable", "reason": reason}


def _available(value: object, unit: str | None = None) -> dict[str, Any]:
    def sanitize(item: object) -> object:
        if isinstance(item, (list, tuple)):
            return [sanitize(value) for value in item]
        if isinstance(item, (float, np.floating)):
            converted = float(item)
            if not math.isfinite(converted):
                raise ValueError("non-finite value")
            return converted
        if isinstance(item, np.integer):
            return int(item)
        return item

    try:
        cleaned = sanitize(value)
    except ValueError:
        return _unavailable("adapter did not publish a finite value")
    field = {"availability": "available", "value": cleaned}
    if unit is not None:
        field["unit"] = unit
    return field


def _input_document(system: DatasetSystem) -> dict[str, Any]:
    return {
        "dataset": system.dataset,
        "subset": system.subset,
        "system_id": system.system_id,
        "canonical_ordinal": system.canonical_ordinal,
        "atomic_numbers": list(system.atomic_numbers),
        "positions": {
            "value": [list(row) for row in system.positions_bohr],
            "unit": "bohr",
            "source_unit": system.manifest.get("coordinate_source_unit"),
            "angstrom_per_bohr": system.manifest.get("angstrom_per_bohr"),
        },
        "charge": system.charge,
        "multiplicity": system.multiplicity,
        "unpaired_electrons": system.unpaired_electrons,
        "spin_channel_policy": (
            "two channels when unpaired_electrons > 0; otherwise one"
        ),
        "input_sha256": system.input_sha256,
        "strata": system.strata,
        "provenance": {"manifest_row": system.manifest},
    }


def _program_document(
    engine: str, args: argparse.Namespace, adapter: Adapter | None = None
) -> dict[str, Any]:
    library = {
        "xtbloom": args.library,
        "xtb": args.xtb_library,
        "tblite": args.tblite_library,
        "dxtb": None,
    }[engine]
    document: dict[str, Any] = {
        "engine": engine,
        "model": "GFN2-xTB",
        "backend": args.backend if engine in {"xtbloom", "dxtb"} else "cpu",
        # dxtb tensors reside on the requested CPU/CUDA device even though it
        # does not expose xTBloom's descriptor tags.  Preserve that execution
        # coordinate so CUDA dxtb rows match the preregistered device ledger.
        "memory_mode": args.memory_mode if engine in {"xtbloom", "dxtb"} else "host",
        "source": benchmark_run.git_state(benchmark_run.REPOSITORY_ROOT),
        "library": str(library.resolve())
        if library is not None and library.is_file()
        else None,
        "library_sha256": benchmark_run.sha256_file(library)
        if library is not None and library.is_file()
        else None,
        "requested_settings": {
            "max_scc_iterations": args.max_scc_iterations,
            **(
                {
                    "accuracy": args.accuracy,
                    "electronic_temperature_kelvin": (
                        args.electronic_temperature_kelvin
                    ),
                }
                if engine in {"xtb", "tblite"}
                else {}
            ),
            **({"fixed_point_l2_tolerance": args.accuracy} if engine == "dxtb" else {}),
        },
    }
    if adapter is not None:
        document["adapter"] = _adapter_versions(engine, adapter)
    else:
        document["adapter"] = {"adapter_class": SHARED_ADAPTERS[engine].__name__}
    return document


def _status_category(native_status: int, converged: int) -> str:
    if native_status == 0 and converged == 1:
        return "success"
    if native_status == 7 or (native_status == 0 and converged != 1):
        return "scc_not_converged"
    if native_status in {2, 3, 5}:
        return "unsupported"
    if native_status == 4:
        return "resource_or_oom"
    if native_status == 1:
        return "invalid_input"
    if native_status == 8:
        return "eigensolver_or_numerical_failure"
    return "unknown_error"


def _exception_category(exc: BaseException) -> str:
    current: BaseException | None = exc
    seen: set[int] = set()
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}".lower())
        if isinstance(current, MemoryError):
            return "resource_or_oom"
        exception_type = type(current)
        if exception_type.__name__ in {
            "OutOfMemoryError",
            "CUDAOutOfMemoryError",
        } and (
            exception_type.__module__.startswith("torch")
            or exception_type.__module__.startswith("cuda")
        ):
            return "resource_or_oom"
        if isinstance(
            current, (EngineUnavailable, public_api.BackendUnavailable, ImportError)
        ):
            return "unsupported"
        current = current.__cause__ or current.__context__
    diagnostic = " | ".join(messages)
    # These mappings are deliberately narrow and auditable.  Unknown text is
    # retained as unknown_error rather than guessed into a scientific class.
    if any(
        marker in diagnostic
        for marker in (
            "out of memory",
            "allocation failed",
            "cannot allocate memory",
            "cuda_error_out_of_memory",
            "cublas_status_alloc_failed",
        )
    ):
        return "resource_or_oom"
    if any(
        marker in diagnostic
        for marker in (
            "unsupported",
            "unavailable",
            "not implemented",
            "not compiled",
            "shared library is missing",
        )
    ):
        return "unsupported"
    if any(marker in diagnostic for marker in ("scc", "scf", "converg")) and any(
        marker in diagnostic
        for marker in ("failed", "failure", "not converg", "did not converg")
    ):
        return "scc_not_converged"
    if any(
        marker in diagnostic
        for marker in (
            "eigensolver",
            "eigenvalue",
            "linalg",
            "lapack",
            "cholesky",
            "singular",
            "numerical",
            "non-finite",
            "not finite",
            "nan",
        )
    ):
        return "eigensolver_or_numerical_failure"
    if any(
        marker in diagnostic
        for marker in (
            "invalid argument",
            "invalid input",
            "malformed",
            "shape mismatch",
            "dimension mismatch",
            "charge/spin mismatch",
        )
    ):
        return "invalid_input"
    return "unknown_error"


def _failure_record(
    engine: str,
    selected_ordinal: int,
    item: LoadedItem,
    category: str,
    diagnostic: str,
    args: argparse.Namespace,
    program: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_document: dict[str, Any]
    if item.system is None:
        input_document = {
            "dataset": item.dataset,
            "subset": item.subset,
            "system_id": item.system_id,
            "canonical_ordinal": item.canonical_ordinal,
            "provenance": {"manifest_row": item.manifest},
        }
    else:
        input_document = _input_document(item.system)
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_ordinal": selected_ordinal,
        "shard": {"count": args.shard_count, "index": args.shard_index},
        "input": input_document,
        "program": program or _program_document(engine, args),
        "status": {
            "category": category,
            "native_status": _unavailable("adapter call did not publish native status"),
            "scc_converged": _unavailable("adapter call did not publish convergence"),
            "diagnostics": [diagnostic],
        },
        "results": {
            "energy": _unavailable(diagnostic),
            "forces": _unavailable(diagnostic),
            "atomic_charges": _unavailable(diagnostic),
            "scc_iterations": _unavailable(diagnostic),
            "final_scc_residual": _unavailable(
                "adapter call did not publish a final SCC residual"
            ),
        },
        "timing": timing or _unavailable("adapter setup did not complete"),
    }


def _records_from_outputs(
    engine: str,
    selected: Sequence[tuple[int, LoadedItem]],
    storage: public_api.PublicBatchStorage,
    outputs: dict[str, Any],
    adapter: Adapter,
    args: argparse.Namespace,
    timing: dict[str, Any],
    program: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    program_document = (
        program if program is not None else _program_document(engine, args)
    )
    adapter_document = program_document.get("adapter")
    if not isinstance(adapter_document, dict) or "settings" not in adapter_document:
        program_document["adapter"] = _adapter_versions(engine, adapter)
    statuses = outputs.get("per_system_status")
    converged_values = outputs.get("scc_converged")
    iterations = outputs.get("scc_iterations")
    energies = outputs.get("energies_hartree", ())
    forces = outputs.get("forces_hartree_per_bohr")
    charges = outputs.get("atomic_charges_e")
    for local_index, (selected_ordinal, item) in enumerate(selected):
        assert item.system is not None
        atom_begin = storage.slices[local_index].atom_begin
        atom_end = storage.slices[local_index].atom_end
        if statuses is None:
            category = "success"
            native_status = _unavailable(
                f"{engine} adapter does not expose per-system status"
            )
            convergence = _unavailable(
                f"{engine} adapter does not expose SCC convergence"
            )
            status_name = None
        else:
            native = int(statuses[local_index])
            converged = int(converged_values[local_index])
            category = _status_category(native, converged)
            status_name = STATUS_NAMES.get(native, "unknown")
            native_status = _available({"code": native, "name": status_name})
            convergence = _available(bool(converged))
        failure_reason = (
            f"engine status is {status_name or category}"
            if category != "success"
            else None
        )
        energy = (
            _available(energies[local_index], "Hartree")
            if failure_reason is None
            else _unavailable(failure_reason)
        )
        force_field = _unavailable(f"{engine} adapter does not provide forces")
        if forces is not None and failure_reason is None:
            flat = forces[3 * atom_begin : 3 * atom_end]
            force_field = _available(
                [flat[index : index + 3] for index in range(0, len(flat), 3)],
                "Hartree/bohr",
            )
        charge_field = _unavailable(f"{engine} adapter does not provide atomic charges")
        if charges is not None and failure_reason is None:
            charge_field = _available(charges[atom_begin:atom_end], "e")
        iteration_field = (
            _available(int(iterations[local_index]))
            if iterations is not None
            else _unavailable(f"{engine} adapter does not expose SCC iterations")
        )
        numerical_fields = [energy]
        if forces is not None:
            numerical_fields.append(force_field)
        if charges is not None:
            numerical_fields.append(charge_field)
        if category == "success" and any(
            field["availability"] == "unavailable" for field in numerical_fields
        ):
            category = "eigensolver_or_numerical_failure"
            failure_reason = "adapter published a non-finite required result"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "selected_ordinal": selected_ordinal,
                "shard": {"count": args.shard_count, "index": args.shard_index},
                "input": _input_document(item.system),
                "program": program_document,
                "status": {
                    "category": category,
                    "native_status": native_status,
                    "scc_converged": convergence,
                    "diagnostics": (
                        []
                        if category == "success"
                        else [
                            failure_reason
                            or (
                                f"per_system_status={status_name}; "
                                f"scc_converged={converged_values[local_index]}"
                            )
                        ]
                    ),
                },
                "results": {
                    "energy": energy,
                    "forces": force_field,
                    "atomic_charges": charge_field,
                    "scc_iterations": iteration_field,
                    "final_scc_residual": _unavailable(
                        f"{engine} adapter/public result does not expose "
                        "a final SCC residual"
                    ),
                },
                "timing": timing,
            }
        )
    return records


def execute_chunk(
    engine: str,
    selected: Sequence[tuple[int, LoadedItem]],
    args: argparse.Namespace,
    program: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute one chunk, retrying xTBloom call-level failures per system."""
    systems = [item.system for _, item in selected]
    if not all(system is not None for system in systems):
        raise DatasetRunnerError("execute_chunk received a malformed item")
    storage = storage_from_systems([system for system in systems if system is not None])
    adapter = None
    setup_start = time.perf_counter_ns()
    try:
        adapter = create_adapter(engine, storage, args)
        setup_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        compute_start = time.perf_counter_ns()
        adapter.invoke()
        if hasattr(adapter, "synchronize"):
            adapter.synchronize()
        compute_ms = (time.perf_counter_ns() - compute_start) * 1.0e-6
        publication_start = time.perf_counter_ns()
        outputs = adapter.raw_results() if engine == "xtbloom" else adapter.results()
        publication_ms = (time.perf_counter_ns() - publication_start) * 1.0e-6
        timing = {
            "scope": (
                "whole adapter chunk; values are not divided into per-system latency"
            ),
            "chunk_size": len(selected),
            "setup_ms": setup_ms,
            "compute_ms": compute_ms,
            "publication_ms": publication_ms,
            "total_ms": setup_ms + compute_ms + publication_ms,
        }
        return _records_from_outputs(
            engine, selected, storage, outputs, adapter, args, timing, program
        )
    except Exception as exc:  # noqa: BLE001 - isolate every third-party failure
        elapsed_ms = (time.perf_counter_ns() - setup_start) * 1.0e-6
        if engine == "xtbloom" and len(selected) > 1:
            if adapter is not None:
                with suppress(Exception):
                    adapter.close()
                adapter = None
            records: list[dict[str, Any]] = []
            for entry in selected:
                records.extend(execute_chunk(engine, [entry], args, program))
            return records
        return [
            _failure_record(
                engine,
                selected_ordinal,
                item,
                _exception_category(exc),
                f"{type(exc).__name__}: {exc}",
                args,
                program,
                {
                    "scope": "failed adapter setup or call",
                    "chunk_size": len(selected),
                    "elapsed_ms": elapsed_ms,
                },
            )
            for selected_ordinal, item in selected
        ]
    finally:
        if adapter is not None:
            with suppress(Exception):
                adapter.close()


class JsonlSink:
    """Append standards-compliant records and durably flush at a fixed cadence."""

    def __init__(self, path: Path, fsync_every: int) -> None:
        self.path = path
        self.fsync_every = fsync_every
        self.count = 0
        self.handle = path.open("x", encoding="utf-8", newline="\n")

    def write(self, record: dict[str, Any]) -> None:
        """Append and flush one complete JSON object."""
        self.handle.write(
            json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        self.count += 1
        self.handle.flush()
        if self.fsync_every > 0 and self.count % self.fsync_every == 0:
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        """Durably flush and close the output stream."""
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def __enter__(self) -> JsonlSink:  # noqa: PYI034 - Python 3.10 compatibility
        """Return the open sink."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the sink at the context boundary."""
        self.close()


def _bundle_metadata(manifest_path: Path, dataset: str) -> dict[str, Any]:
    bundle_root = manifest_path.parents[1]
    documents: dict[str, Any] = {}
    for relative in (
        Path("provenance/source_manifest.json"),
        Path("provenance/sampling_config.json"),
        Path("manifests/set_summary.json"),
    ):
        path = bundle_root / relative
        if path.is_file():
            documents[str(relative)] = _load_json(path)
    checksums = bundle_root / "SHA256SUMS"
    return {
        "dataset": dataset,
        "bundle_root": str(bundle_root),
        "manifest": str(manifest_path),
        "manifest_sha256": benchmark_run.sha256_file(manifest_path),
        "sha256sums": checksums.read_text(encoding="utf-8").splitlines()
        if checksums.is_file()
        else None,
        "documents": documents,
    }


def _chunks(
    values: Sequence[tuple[int, LoadedItem]], size: int
) -> Iterator[list[tuple[int, LoadedItem]]]:
    for begin in range(0, len(values), size):
        yield list(values[begin : begin + size])


def _write_pending(
    engine: str,
    pending: list[tuple[int, LoadedItem]],
    args: argparse.Namespace,
    sink: JsonlSink,
    counts: Counter[tuple[str, str]],
    program: dict[str, Any],
) -> None:
    """Execute, serialize, and clear one stable-order adapter chunk."""
    if not pending:
        return
    for record in execute_chunk(engine, list(pending), args, program):
        sink.write(record)
        counts[(engine, record["status"]["category"])] += 1
    pending.clear()


def run_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected shard and retain every malformed or failed record."""
    validate_args(args)
    manifest_path = args.manifest.resolve()
    dataset = detect_dataset(manifest_path) if args.dataset == "auto" else args.dataset
    if args.subsets:
        requested_subsets = set(args.subsets)
        if dataset == "qm9":
            _, subset_rows = _csv_rows(manifest_path)
            available_subsets = {row.get("sample_set") or "main" for row in subset_rows}
        else:
            available_subsets = set(_omol25_set_order(manifest_path.parents[1]))
        unknown_subsets = requested_subsets - available_subsets
        if unknown_subsets:
            raise DatasetRunnerError(
                "unknown subsets: " + ", ".join(sorted(unknown_subsets))
            )
    subset_selection = set(args.subsets) if args.subsets else None

    def selected_items() -> Iterator[tuple[int, LoadedItem]]:
        return select_items(
            load_manifest(manifest_path, dataset),
            subset_selection,
            args.limit,
            args.shard_count,
            args.shard_index,
        )

    # Count and validate once without retaining Python copies of all geometries.
    # Execution reopens the immutable bundle for each engine and holds only one
    # adapter chunk at a time.
    selected_count = sum(1 for _ in selected_items())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    metadata_path = args.output_dir / "run_metadata.json"
    summary_path = args.output_dir / "summary.json"
    for path in (results_path, metadata_path, summary_path):
        if path.exists():
            raise DatasetRunnerError(f"refusing to overwrite existing output: {path}")
    programs = {engine: _program_document(engine, args) for engine in args.engines}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "runner": {
            "python": sys.version,
            "platform": platform.platform(),
            "source": benchmark_run.git_state(benchmark_run.REPOSITORY_ROOT),
        },
        "bundle": _bundle_metadata(manifest_path, dataset),
        "selection": {
            "subsets": list(args.subsets) if args.subsets else None,
            "limit_before_sharding": args.limit,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "selected_records_in_shard": selected_count,
        },
        "protocol": {
            "model": "GFN2-xTB",
            "properties": ["energy", "forces", "atomic_charges"],
            "coordinate_unit": "bohr",
            "engines": list(args.engines),
            "xtbloom_batch_size": args.batch_size,
            "reference_batch_size": 1,
            "failure_isolation": (
                "malformed rows are serialized directly; reference adapters run one "
                "system at a time; xTBloom raw per-system statuses are retained and "
                "call-level chunk failures are retried one system at a time"
            ),
        },
    }
    benchmark_run.write_json(metadata_path, metadata)
    counts: Counter[tuple[str, str]] = Counter()
    with JsonlSink(results_path, args.fsync_every) as sink:
        for engine in args.engines:
            pending: list[tuple[int, LoadedItem]] = []
            batch_size = args.batch_size if engine == "xtbloom" else 1
            for selected_ordinal, item in selected_items():
                if item.system is None:
                    _write_pending(
                        engine, pending, args, sink, counts, programs[engine]
                    )
                    record = _failure_record(
                        engine,
                        selected_ordinal,
                        item,
                        "malformed",
                        item.error or "malformed manifest record",
                        args,
                        programs[engine],
                    )
                    sink.write(record)
                    counts[(engine, "malformed")] += 1
                else:
                    pending.append((selected_ordinal, item))
                    if len(pending) == batch_size:
                        _write_pending(
                            engine, pending, args, sink, counts, programs[engine]
                        )
            _write_pending(engine, pending, args, sink, counts, programs[engine])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "results": str(results_path),
        "metadata": str(metadata_path),
        "record_count": sum(counts.values()),
        "counts": {
            engine: {
                category: count
                for (selected_engine, category), count in sorted(counts.items())
                if selected_engine == engine
            }
            for engine in args.engines
        },
    }
    benchmark_run.write_json(summary_path, summary)
    return summary


def _csv_selection(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("selection must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Define the frozen-bundle runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("auto", "qm9", "omol25"), default="auto")
    parser.add_argument("--subsets", type=_csv_selection)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--engines", type=_csv_selection, default=("xtbloom",))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fsync-every", type=int, default=1)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--xtb-library", type=Path)
    parser.add_argument("--tblite-library", type=Path)
    parser.add_argument("--dxtb-source", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--memory-mode", choices=("host", "device", "mixed"), default="host"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--accuracy", type=float, default=1.0e-4)
    parser.add_argument("--max-scc-iterations", type=int, default=500)
    parser.add_argument("--electronic-temperature-kelvin", type=float, default=300.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid selections before creating output artifacts."""
    unknown = set(args.engines) - set(SHARED_ADAPTERS)
    if unknown:
        raise DatasetRunnerError(f"unknown engines: {', '.join(sorted(unknown))}")
    if args.limit is not None and args.limit < 0:
        raise DatasetRunnerError("limit must be nonnegative")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise DatasetRunnerError("shard index must be in [0, shard_count)")
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        raise DatasetRunnerError("batch size and CPU threads must be positive")
    if args.device_id < 0 or args.max_scc_iterations <= 0:
        raise DatasetRunnerError("device ID must be nonnegative and SCC limit positive")
    if args.accuracy <= 0.0 or args.electronic_temperature_kelvin < 0.0:
        raise DatasetRunnerError(
            "accuracy must be positive and temperature nonnegative"
        )
    if args.backend == "cpu" and args.memory_mode != "host":
        raise DatasetRunnerError("CPU xTBloom execution requires host memory mode")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested shard and print its compact summary."""
    args = build_parser().parse_args(argv)
    try:
        summary = run_dataset(args)
    except DatasetRunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - CLI diagnostic
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201 - CLI result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
